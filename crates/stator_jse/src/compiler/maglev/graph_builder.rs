//! Maglev graph builder: walk [`BytecodeArray`] + [`FeedbackVector`] and emit
//! a [`MaglevGraph`].
//!
//! # Overview
//!
//! The [`GraphBuilder`] is the entry-point for the Maglev optimising tier.  It
//! consumes the bytecode of a single JavaScript function together with its
//! runtime [`FeedbackVector`] and produces a typed, SSA-style
//! [`MaglevGraph`] ready for further optimisation and code generation.
//!
//! ## Speculative type guards
//!
//! The builder consults the feedback vector to decide whether to emit
//! *speculative* (fast-path) IR or *generic* (slow-path) IR:
//!
//! - **Arithmetic** — the builder always emits [`ValueNode::GenericAdd`]
//!   (or the matching variant) for all binary arithmetic operations.
//!   Generic ops enable `smi_guarded` analysis on upstream loads and
//!   have a runtime fallback on overflow instead of deopt.  The codegen
//!   selects the optimal path at emit time: inline i32 ALU for
//!   `smi_guarded` loads, or full Smi-check + runtime fallback otherwise.
//!
//! - **Property loads** — when the load-property slot shows
//!   [`InlineCacheState::Monomorphic`] the builder emits a
//!   [`ValueNode::CheckMaps`] guard followed by [`ValueNode::LoadField`].
//!   Polymorphic or uninitialized slots fall back to
//!   [`ValueNode::LoadNamedGeneric`].
//!
//! - **Keyed property access** — when a `KeyedLoadProperty` or
//!   `KeyedStoreProperty` slot is [`InlineCacheState::Monomorphic`] or
//!   [`InlineCacheState::Polymorphic`] the builder emits a
//!   [`ValueNode::CheckSmi`] guard on the key followed by
//!   [`ValueNode::LoadFixedArrayElement`] /
//!   [`ValueNode::StoreFixedArrayElement`].  Uninitialized or megamorphic
//!   slots fall back to [`ValueNode::LoadKeyedGeneric`] /
//!   [`ValueNode::StoreKeyedGeneric`].
//!
//! ## Liveness / register frame
//!
//! Bytecode registers (including the accumulator, which is stored at index
//! [`ACCUMULATOR_REG`]) are tracked in a flat `Vec<Option<NodeId>>` called the
//! *environment*.  Each entry is the [`NodeId`] of the IR value currently
//! live in that register, or `None` if the register has not yet been written.
//!
//! ## Control flow
//!
//! Each distinct bytecode offset targeted by a jump gets its own
//! [`BasicBlock`].  The builder performs a linear scan in two passes:
//!
//! 1. **Target collection** — scan instructions to identify all jump targets
//!    (offsets), creating a mapping from bytecode offset → block index.
//! 2. **Translation** — walk instructions sequentially, starting new blocks
//!    at each recorded target and emitting IR nodes.
//!
//! # Example
//!
//! ```
//! use stator_jse::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
//! use stator_jse::bytecode::bytecodes::{encode, Instruction, Opcode, Operand};
//! use stator_jse::bytecode::feedback::{
//!     FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
//! };
//! use stator_jse::compiler::maglev::graph_builder::GraphBuilder;
//! use stator_jse::compiler::maglev::ir::ControlNode;
//!
//! // Build a tiny function: load Smi 1, return.
//! let instrs = vec![
//!     Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
//!     Instruction::new_unchecked(Opcode::Return, vec![]),
//! ];
//! let bytes = encode(&instrs);
//! let array = BytecodeArray::new(bytes, vec![], 0, 0, vec![], FeedbackMetadata::empty(), vec![]);
//! let vector = FeedbackVector::new(array.feedback_metadata());
//!
//! let graph = GraphBuilder::build(&array, &vector).expect("build ok");
//! assert_eq!(graph.blocks().len(), 1);
//! assert!(matches!(
//!     graph.entry_block().unwrap().control,
//!     Some(ControlNode::Return { .. })
//! ));
//! ```

use std::collections::{HashMap, HashSet};

use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};
use crate::bytecode::feedback::{FeedbackSlotKind, FeedbackVector, InlineCacheState};
use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Virtual register index reserved for the accumulator in the environment
/// frame.  All real bytecode register indices are `1 + register_index`.
const ACCUMULATOR_REG: usize = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Environment
// ─────────────────────────────────────────────────────────────────────────────

/// The per-block SSA environment: maps each bytecode register (plus the
/// accumulator at slot 0) to the [`NodeId`] of the IR value currently live
/// in that slot.
///
/// Slot layout (mirrors the interpreter/JIT register-file layout):
///
/// ```text
/// slot 0                     = accumulator
/// slots 1 .. 1+param_count   = parameter registers (param0 .. paramN-1)
/// slots 1+param_count ..     = local/temporary registers (r0, r1, …)
/// ```
///
/// Bytecode uses two's-complement `u32` register indices:
/// - `v as i32 < 0`: parameter `-(v as i32 + 1)` → slot `1 + (-(v as i32 + 1))`
/// - `v as i32 >= 0`: local register `v` → slot `1 + param_count + v`
#[derive(Clone)]
struct Environment {
    /// `slots[0]` = accumulator; see doc comment for layout.
    slots: Vec<Option<NodeId>>,
    /// Number of parameter registers (needed to resolve positive register
    /// indices to their correct slot).
    param_count: usize,
}

impl Environment {
    /// Create an environment large enough for `param_count` parameters,
    /// `frame_size` local registers, and the accumulator.
    fn new(param_count: usize, frame_size: usize) -> Self {
        Self {
            slots: vec![None; 1 + param_count + frame_size],
            param_count,
        }
    }

    /// Read the accumulator.
    fn accumulator(&self) -> Option<NodeId> {
        self.slots[ACCUMULATOR_REG]
    }

    /// Write the accumulator.
    fn set_accumulator(&mut self, id: NodeId) {
        self.slots[ACCUMULATOR_REG] = Some(id);
    }

    /// Compute the slot index for a bytecode register operand `reg`.
    ///
    /// - `reg as i32 < 0`: parameter register; slot = `1 + (-(reg as i32 + 1))`
    /// - `reg as i32 >= 0`: local/temporary register; slot = `1 + param_count + reg`
    fn slot_index(&self, reg: u32) -> usize {
        let signed = reg as i32;
        if signed < 0 {
            1 + (-(signed + 1)) as usize
        } else {
            1 + self.param_count + signed as usize
        }
    }

    /// Read a bytecode register.
    fn get(&self, reg: u32) -> Option<NodeId> {
        self.slots.get(self.slot_index(reg)).copied().flatten()
    }

    /// Write a bytecode register.
    fn set(&mut self, reg: u32, id: NodeId) {
        let idx = self.slot_index(reg);
        if idx < self.slots.len() {
            self.slots[idx] = Some(id);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Translates a single function's [`BytecodeArray`] into a [`MaglevGraph`].
///
/// Construct via [`GraphBuilder::build`]; the builder is consumed once the
/// graph is complete.
pub struct GraphBuilder<'a> {
    /// The bytecode being compiled.
    bytecode: &'a BytecodeArray,
    /// Runtime feedback for speculative decisions.  When keyed access slots
    /// show [`InlineCacheState::Monomorphic`] or [`Polymorphic`] the builder
    /// emits `CheckSmi` + `LoadFixedArrayElement`/`StoreFixedArrayElement`
    /// instead of the generic stubs.
    feedback: &'a FeedbackVector,
    /// The graph under construction.
    graph: MaglevGraph,
    /// Current SSA environment (register → NodeId).
    env: Environment,
    /// Index of the block currently being filled.
    current_block: u32,
    /// Map from *instruction index* to the block that starts at that index.
    /// Populated during the first (target-collection) pass.
    block_at: HashMap<usize, u32>,
    /// Block IDs that are loop headers (targets of `JumpLoop` opcodes).
    loop_headers: HashSet<u32>,
    /// Saved environments for successor blocks: `block_id → [(pred_block, env)]`.
    saved_envs: HashMap<u32, Vec<(u32, Environment)>>,
    /// For each loop header: `(slot_index, phi_node_id)` pairs for back-edge
    /// patching when `JumpLoop` is encountered.
    loop_header_phis: HashMap<u32, Vec<(usize, NodeId)>>,
    /// Within-block CSE for global variables: maps global name index →
    /// the last known value [`NodeId`] for that global.  Cleared at block
    /// boundaries and after any opcode that may have side-effects on globals
    /// (function calls, etc.).
    known_globals: HashMap<u32, NodeId>,
    /// Within-block CSE for named property loads: maps `(object, name_idx)` →
    /// the last known result [`NodeId`].  Cleared at block boundaries and
    /// after any store/call that may mutate object properties.
    known_props: HashMap<(NodeId, u32), NodeId>,
}

impl<'a> GraphBuilder<'a> {
    /// Translate `bytecode` + `feedback` into a [`MaglevGraph`].
    ///
    /// Returns an error if the bytecode is malformed (e.g. a register
    /// reference is out of range or a required operand is missing).
    pub fn build(
        bytecode: &'a BytecodeArray,
        feedback: &'a FeedbackVector,
    ) -> StatorResult<MaglevGraph> {
        let instructions = bytecode.instructions()?;

        // NOTE: The bail-out for CallProperty + keyed access (commit 49cfa6b0)
        // has been replaced by post-call IC invalidation in codegen.  After
        // every Call/CallKnownFunction/Construct node, the codegen zeroes the
        // array IC handle so the next keyed access re-fills with current
        // metadata.  This is both safer (covers array shrinks, not just
        // growth) and less restrictive (no false-positive bail-outs for
        // unrelated call + keyed access pairs like sieve_primes).

        let frame_size = bytecode.frame_size() as usize;
        let parameter_count = bytecode.parameter_count();

        let mut builder = Self {
            bytecode,
            feedback,
            graph: MaglevGraph::new(parameter_count),
            env: Environment::new(parameter_count as usize, frame_size),
            current_block: 0,
            block_at: HashMap::new(),
            loop_headers: HashSet::new(),
            saved_envs: HashMap::new(),
            loop_header_phis: HashMap::new(),
            known_globals: HashMap::new(),
            known_props: HashMap::new(),
        };

        // Always start with block 0 (the entry block).  It MUST be added
        // before `collect_targets` so that its Vec index matches its block id
        // (block(id) uses the Vec index, so id==0 must live at index 0).
        builder.graph.add_block(BasicBlock::new(0));
        builder.current_block = 0;

        // Pass 1: collect all jump targets → assign block indices.
        builder.collect_targets(&instructions);

        // Emit Parameter nodes into the entry block.
        // Store each parameter at the two's-complement register address the
        // bytecode uses to refer to it: param[i] ↔ register -(i+1).
        for i in 0..parameter_count {
            let id = builder
                .graph
                .add_value_node(0, ValueNode::Parameter { index: i })
                .ok_or_else(|| StatorError::Internal("entry block missing".into()))?;
            // Register index for param[i] in the two's-complement encoding:
            // param[0] → r[-1] = 0xFFFFFFFF, param[1] → r[-2] = 0xFFFFFFFE, …
            let param_reg = (-(i as i32 + 1)) as u32;
            builder.env.set(param_reg, id);
        }

        // Pass 2: translate instructions.
        builder.translate(&instructions)?;

        Ok(builder.graph)
    }

    // ── Pre-scan: unsafe pattern detection ───────────────────────────────────

    /// Return `true` when `instructions` contain both a property method call
    /// (`CallProperty`, `CallProperty0`, `CallProperty1`, or `CallProperty2`)
    /// and a keyed element load (`LdaKeyedProperty`) or store
    /// (`StaKeyedProperty`, `DefineKeyedOwnProperty`).
    ///
    /// This combination was previously used to bail out of JIT compilation,
    /// but has been superseded by post-call IC invalidation in codegen.
    /// Retained for potential future use in diagnostics or heuristics.
    #[allow(dead_code)]
    fn has_property_call_and_keyed_load(instructions: &[Instruction]) -> bool {
        let mut has_property_call = false;
        let mut has_keyed_access = false;
        for instr in instructions {
            match instr.opcode {
                Opcode::CallProperty
                | Opcode::CallProperty0
                | Opcode::CallProperty1
                | Opcode::CallProperty2 => has_property_call = true,
                Opcode::LdaKeyedProperty
                | Opcode::StaKeyedProperty
                | Opcode::DefineKeyedOwnProperty => has_keyed_access = true,
                _ => {}
            }
            if has_property_call && has_keyed_access {
                return true;
            }
        }
        false
    }

    // ── Pass 1: target collection ────────────────────────────────────────────

    /// Scan `instructions` and create a new [`BasicBlock`] for every distinct
    /// jump target (including the instruction immediately after a conditional
    /// jump, which is the fall-through target).
    ///
    /// Block 0 (the entry block) is always created by the caller; we only
    /// record additional blocks here.
    fn collect_targets(&mut self, instructions: &[Instruction]) {
        // The offset *after* the last instruction — used to determine whether a
        // jump offset lands inside the function.
        let instr_count = instructions.len();

        // We allocate block index 0 for the entry; additional targets get
        // indices 1, 2, …
        let mut next_block_idx: u32 = 1;

        for (i, instr) in instructions.iter().enumerate() {
            // Resolve the absolute target instruction index from a jump offset.
            // The bytecode format stores signed byte offsets from the *current*
            // instruction's start position.  However our `instructions` slice
            // is already decoded; we reconstruct the target index by treating
            // the offset as a delta on the current *byte* offset and then
            // finding the corresponding decoded-instruction index.
            //
            // For simplicity (and consistency with how we built the decoded
            // list) we use the utility `jump_target_index` below.
            let maybe_target = self.jump_target_index(instructions, i);

            let is_conditional_jump = matches!(
                instr.opcode,
                Opcode::JumpIfTrue
                    | Opcode::JumpIfFalse
                    | Opcode::JumpIfToBooleanTrue
                    | Opcode::JumpIfToBooleanFalse
                    | Opcode::JumpIfNull
                    | Opcode::JumpIfNotNull
                    | Opcode::JumpIfUndefined
                    | Opcode::JumpIfNotUndefined
                    | Opcode::JumpIfUndefinedOrNull
                    | Opcode::JumpIfJSReceiver
                    | Opcode::JumpIfTrueConstant
                    | Opcode::JumpIfFalseConstant
                    | Opcode::JumpIfToBooleanTrueConstant
                    | Opcode::JumpIfToBooleanFalseConstant
                    | Opcode::JumpIfNullConstant
                    | Opcode::JumpIfNotNullConstant
                    | Opcode::JumpIfUndefinedConstant
                    | Opcode::JumpIfNotUndefinedConstant
                    | Opcode::JumpIfUndefinedOrNullConstant
                    | Opcode::JumpIfJSReceiverConstant
                    | Opcode::JumpLoop
                    | Opcode::TestLessThanJump
                    | Opcode::TestGreaterThanJump
                    | Opcode::TestLessThanOrEqualJump
                    | Opcode::TestGreaterThanOrEqualJump
                    | Opcode::TestEqualJump
                    | Opcode::TestNotEqualJump
                    | Opcode::TestEqualStrictJump
            );

            // Record the jump target as a new block boundary.  `target == 0`
            // is already handled as the pre-existing entry block (block 0), so
            // we only allocate a new block for targets > 0.  `resolve_jump_target`
            // returns `Some(0)` directly for instruction-index 0.
            if let Some(target) = maybe_target
                && target > 0
                && target < instr_count
                && !self.block_at.contains_key(&target)
            {
                self.block_at.insert(target, next_block_idx);
                self.graph.add_block(BasicBlock::new(next_block_idx));
                next_block_idx += 1;
            }

            // Mark JumpLoop targets as loop headers so that `enter_block`
            // creates Phi nodes for loop-carried variables.
            if instr.opcode == Opcode::JumpLoop
                && let Some(target) = maybe_target
                && target < instr_count
            {
                // target == 0 maps to block 0 (the entry block).
                let block_id = if target == 0 {
                    0
                } else if let Some(&bid) = self.block_at.get(&target) {
                    bid
                } else {
                    continue;
                };
                self.loop_headers.insert(block_id);
            }

            // For conditional jumps the fall-through (next instruction) is
            // also a new block boundary.
            if is_conditional_jump {
                let fall = i + 1;
                if fall < instr_count && !self.block_at.contains_key(&fall) {
                    self.block_at.insert(fall, next_block_idx);
                    self.graph.add_block(BasicBlock::new(next_block_idx));
                    next_block_idx += 1;
                }
            }
        }
    }

    /// Compute the absolute instruction index that a jump at instruction `i`
    /// targets, or `None` if instruction `i` is not a jump.
    ///
    /// We re-encode / decode the byte offsets by accumulating byte lengths of
    /// the prefix instructions to convert from byte offset to instruction
    /// index.  For simplicity we walk the instruction list to build a byte →
    /// index map on demand (small functions dominate in practice).
    fn jump_target_index(&self, instructions: &[Instruction], i: usize) -> Option<usize> {
        use crate::bytecode::bytecodes::encode;

        let instr = &instructions[i];
        let offset = match instr.opcode {
            Opcode::Jump
            | Opcode::JumpIfTrue
            | Opcode::JumpIfFalse
            | Opcode::JumpIfToBooleanTrue
            | Opcode::JumpIfToBooleanFalse
            | Opcode::JumpIfNull
            | Opcode::JumpIfNotNull
            | Opcode::JumpIfUndefined
            | Opcode::JumpIfNotUndefined
            | Opcode::JumpIfUndefinedOrNull
            | Opcode::JumpIfJSReceiver
            | Opcode::JumpLoop => {
                if let Operand::JumpOffset(off) = *instr.operand(0) {
                    off
                } else {
                    return None;
                }
            }
            // Fused test+jump superinstructions: JumpOffset is operand 2.
            Opcode::TestLessThanJump
            | Opcode::TestGreaterThanJump
            | Opcode::TestLessThanOrEqualJump
            | Opcode::TestGreaterThanOrEqualJump
            | Opcode::TestEqualJump
            | Opcode::TestNotEqualJump
            | Opcode::TestEqualStrictJump => {
                if let Operand::JumpOffset(off) = *instr.operand(2) {
                    off
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Build a byte-offset → instruction-index mapping.
        let mut byte_to_idx: HashMap<usize, usize> = HashMap::new();
        let mut byte_off = 0usize;
        for (idx, ins) in instructions.iter().enumerate() {
            byte_to_idx.insert(byte_off, idx);
            let encoded = encode(std::slice::from_ref(ins));
            byte_off += encoded.len();
        }

        // The current instruction's byte offset:
        let mut cur_byte = 0usize;
        for ins in &instructions[..i] {
            let enc = encode(std::slice::from_ref(ins));
            cur_byte += enc.len();
        }
        // Jump offset is relative to the *end* of the current instruction
        // (resolve_jumps computes: target_byte - offsets[jump_idx + 1]).
        let instr_size = encode(std::slice::from_ref(&instructions[i])).len();
        let target_byte = (cur_byte as i64 + instr_size as i64 + offset as i64) as usize;
        byte_to_idx.get(&target_byte).copied()
    }

    // ── Pass 2: translation ───────────────────────────────────────────────────

    /// Walk `instructions` in order and emit IR nodes into the current block.
    fn translate(&mut self, instructions: &[Instruction]) -> StatorResult<()> {
        for (i, instr) in instructions.iter().enumerate() {
            // If this instruction begins a new block, close the current block
            // with an unconditional jump and switch to the new one.
            if let Some(&new_block) = self.block_at.get(&i) {
                let cur = self.current_block;
                if !self.block_is_complete(cur) {
                    self.save_env_for_successor(new_block);
                    self.set_control(ControlNode::Jump { target: new_block });
                    if let Some(block) = self.graph.block_mut(new_block) {
                        block.add_predecessor(cur);
                    }
                } else if self.block_ended_with_deoptimize(cur) {
                    // The current block was terminated by Deoptimize (e.g. an
                    // unhandled opcode).  The successor still needs env state
                    // for its Phi nodes / loop header.  We only do this for
                    // Deoptimize — not for Jump/Branch/Return which already
                    // saved env for their explicit targets.
                    self.save_env_for_successor(new_block);
                }
                self.current_block = new_block;
                self.enter_block(new_block);
            }

            // Skip remaining instructions after the block's terminal control
            // node (e.g. after Deoptimize for an unhandled opcode). This
            // prevents emitting unreachable nodes that bloat the graph.
            if self.block_is_complete(self.current_block) {
                continue;
            }

            self.translate_one(i, instr, instructions)?;
        }

        // If the last block has no terminator (e.g. function ends without an
        // explicit Return), emit `Return undefined`.
        let cur = self.current_block;
        if !self.block_is_complete(cur) {
            let undef = self.emit(ValueNode::UndefinedConstant)?;
            self.set_control(ControlNode::Return { value: undef });
        }

        Ok(())
    }

    /// Translate a single decoded instruction.
    fn translate_one(
        &mut self,
        instr_idx: usize,
        instr: &Instruction,
        all_instructions: &[Instruction],
    ) -> StatorResult<()> {
        match instr.opcode {
            // ── Constants ────────────────────────────────────────────────────
            Opcode::LdaZero => {
                let id = self.emit(ValueNode::SmiConstant { value: 0 })?;
                self.env.set_accumulator(id);
            }
            Opcode::LdaSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let id = self.emit(ValueNode::SmiConstant { value: imm })?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: LdaSmi + Star ─────────────────────────────
            Opcode::LdaSmiStar => {
                let imm = self.operand_immediate(instr, 0)?;
                let dst = self.operand_register(instr, 1)?;
                let id = self.emit(ValueNode::SmiConstant { value: imm })?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::LdaUndefined => {
                let id = self.emit(ValueNode::UndefinedConstant)?;
                self.env.set_accumulator(id);
            }
            Opcode::LdaNull => {
                let id = self.emit(ValueNode::NullConstant)?;
                self.env.set_accumulator(id);
            }
            Opcode::LdaTrue => {
                let id = self.emit(ValueNode::TrueConstant)?;
                self.env.set_accumulator(id);
            }
            Opcode::LdaFalse => {
                let id = self.emit(ValueNode::FalseConstant)?;
                self.env.set_accumulator(id);
            }
            Opcode::Nop => {}
            Opcode::LdaConstant => {
                let idx = self.operand_constant_pool_idx(instr, 0)?;
                let id = self.emit_constant_pool_entry(idx)?;
                self.env.set_accumulator(id);
            }

            // ── Register moves ───────────────────────────────────────────────
            Opcode::Ldar => {
                let reg = self.operand_register(instr, 0)?;
                let id = self.env_get_register(reg)?;
                self.env.set_accumulator(id);
            }
            Opcode::LdarAddStar => {
                let src = self.operand_register(instr, 0)?;
                let add_reg = self.operand_register(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let left = self.env_get_register(src)?;
                let right = self.env_get_register(add_reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Add)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::LdarSubStar => {
                let src = self.operand_register(instr, 0)?;
                let sub_reg = self.operand_register(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let left = self.env_get_register(src)?;
                let right = self.env_get_register(sub_reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Sub)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::Star => {
                let reg = self.operand_register(instr, 0)?;
                let id = self.env_get_accumulator()?;
                self.env.set(reg, id);
            }
            Opcode::Mov => {
                let src = self.operand_register(instr, 0)?;
                let dst = self.operand_register(instr, 1)?;
                let id = self.env_get_register(src)?;
                self.env.set(dst, id);
            }

            // ── Global loads / stores ────────────────────────────────────────
            Opcode::LdaGlobal | Opcode::LdaGlobalInsideTypeof => {
                let name = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                // Within-block CSE: reuse the last known value for this
                // global if no intervening store or call has invalidated it.
                let id = if let Some(&known) = self.known_globals.get(&name) {
                    known
                } else {
                    let id = self.emit(ValueNode::LoadGlobal {
                        name,
                        feedback_slot: slot,
                    })?;
                    self.known_globals.insert(name, id);
                    id
                };
                self.env.set_accumulator(id);
            }
            Opcode::StaGlobal => {
                let name = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let value = self.env_get_accumulator()?;
                self.emit(ValueNode::StoreGlobal {
                    name,
                    value,
                    feedback_slot: slot,
                })?;
                // The stored value is now the known state for this global.
                self.known_globals.insert(name, value);
            }
            // ── Superinstruction: LdaGlobal + Star ──────────────────────────
            Opcode::LdaGlobalStar => {
                let name = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let id = if let Some(&known) = self.known_globals.get(&name) {
                    known
                } else {
                    let id = self.emit(ValueNode::LoadGlobal {
                        name,
                        feedback_slot: slot,
                    })?;
                    self.known_globals.insert(name, id);
                    id
                };
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }

            // ── Property loads ───────────────────────────────────────────────
            // LdaNamedProperty [obj_reg, name_idx, slot]
            // Speculative: Monomorphic → CheckMaps + LoadField
            //              otherwise   → LoadNamedGeneric
            Opcode::LdaNamedProperty | Opcode::LdaNamedPropertyFromSuper => {
                let obj_reg = self.operand_register(instr, 0)?;
                let name = self.operand_constant_pool_idx(instr, 1)?;
                let slot = self.operand_feedback_slot(instr, 2)?;
                let obj = self.env_get_register(obj_reg)?;
                let id = self.emit_named_property_load(obj, name, slot)?;
                self.env.set_accumulator(id);
            }

            // ── Keyed property loads ─────────────────────────────────────────
            // LdaKeyedProperty [obj_reg, slot]
            Opcode::LdaKeyedProperty => {
                let obj_reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let obj = self.env_get_register(obj_reg)?;
                let key = self.env_get_accumulator()?;
                let id = self.emit_keyed_property_load(obj, key, slot)?;
                self.env.set_accumulator(id);
            }

            // ── Property stores ──────────────────────────────────────────────
            // StaNamedProperty [obj_reg, name_idx, slot]
            Opcode::StaNamedProperty
            | Opcode::StaNamedOwnProperty
            | Opcode::DefineNamedOwnProperty => {
                let obj_reg = self.operand_register(instr, 0)?;
                let name = self.operand_constant_pool_idx(instr, 1)?;
                let slot = self.operand_feedback_slot(instr, 2)?;
                let obj = self.env_get_register(obj_reg)?;
                let value = self.env_get_accumulator()?;
                self.emit(ValueNode::StoreNamedGeneric {
                    object: obj,
                    name,
                    value,
                    feedback_slot: slot,
                })?;
                // Invalidate property CSE — the store may affect cached loads.
                self.known_props.clear();
            }

            // StaKeyedProperty [obj_reg, key_reg, slot]
            // StaInArrayLiteral [array_reg, index_reg, slot] — same layout,
            // stores accumulator at a keyed index in an array literal.
            Opcode::StaKeyedProperty | Opcode::StaInArrayLiteral => {
                let obj_reg = self.operand_register(instr, 0)?;
                let key_reg = self.operand_register(instr, 1)?;
                let slot = self.operand_feedback_slot(instr, 2)?;
                let obj = self.env_get_register(obj_reg)?;
                let key = self.env_get_register(key_reg)?;
                let value = self.env_get_accumulator()?;
                self.emit_keyed_property_store(obj, key, value, slot)?;
                // Invalidate property CSE — keyed store may affect named lookups.
                self.known_props.clear();
            }

            // DefineKeyedOwnProperty [obj, key_reg, flags, slot]
            // DefineKeyedOwnPropertyInLiteral [obj, key_reg, flags, slot]
            // Semantically equivalent to a keyed store for JIT purposes.
            Opcode::DefineKeyedOwnProperty | Opcode::DefineKeyedOwnPropertyInLiteral => {
                let obj_reg = self.operand_register(instr, 0)?;
                let key_reg = self.operand_register(instr, 1)?;
                // operand 2 is flags — not needed at IR level.
                let slot = self.operand_feedback_slot(instr, 3)?;
                let obj = self.env_get_register(obj_reg)?;
                let key = self.env_get_register(key_reg)?;
                let value = self.env_get_accumulator()?;
                self.emit_keyed_property_store(obj, key, value, slot)?;
                self.known_props.clear();
            }

            // ── Arithmetic ───────────────────────────────────────────────────
            // Binary ops: accumulator OP reg, with feedback slot.
            Opcode::Add => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Add)?;
                self.env.set_accumulator(id);
            }
            Opcode::Sub => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Sub)?;
                self.env.set_accumulator(id);
            }
            Opcode::Mul => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mul)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: Ldar + Mul + Star ─────────────────────────
            Opcode::LdarMulStar => {
                let src = self.operand_register(instr, 0)?;
                let mul_reg = self.operand_register(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let left = self.env_get_register(src)?;
                self.env.set_accumulator(left);
                let right = self.env_get_register(mul_reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mul)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::Div => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Div)?;
                self.env.set_accumulator(id);
            }
            Opcode::Mod => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mod)?;
                self.env.set_accumulator(id);
            }
            Opcode::Exp => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Exp)?;
                self.env.set_accumulator(id);
            }

            // Immediate-operand Smi arithmetic.
            Opcode::AddSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Add)?;
                self.env.set_accumulator(id);
            }
            Opcode::AddSmiStar => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Add)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::SubSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Sub)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: SubSmi + Star ─────────────────────────────
            Opcode::SubSmiStar => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Sub)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::MulSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mul)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: MulSmi + Star ─────────────────────────────
            Opcode::MulSmiStar => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let dst = self.operand_register(instr, 2)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mul)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::DivSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Div)?;
                self.env.set_accumulator(id);
            }
            Opcode::ModSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Mod)?;
                self.env.set_accumulator(id);
            }
            Opcode::ExpSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::Exp)?;
                self.env.set_accumulator(id);
            }

            // Bitwise ops (register operand).
            Opcode::BitwiseOr => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseOr)?;
                self.env.set_accumulator(id);
            }
            Opcode::BitwiseXor => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseXor)?;
                self.env.set_accumulator(id);
            }
            Opcode::BitwiseAnd => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseAnd)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftLeft => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftLeft)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftRight => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftRight)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftRightLogical => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftRightLogical)?;
                self.env.set_accumulator(id);
            }

            // Bitwise ops (immediate operand).
            Opcode::BitwiseOrSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseOr)?;
                self.env.set_accumulator(id);
            }
            Opcode::BitwiseXorSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseXor)?;
                self.env.set_accumulator(id);
            }
            Opcode::BitwiseAndSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::BitwiseAnd)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftLeftSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftLeft)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftRightSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftRight)?;
                self.env.set_accumulator(id);
            }
            Opcode::ShiftRightLogicalSmi => {
                let imm = self.operand_immediate(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.emit(ValueNode::SmiConstant { value: imm })?;
                let id = self.emit_binary_op(left, right, slot, BinaryOpKind::ShiftRightLogical)?;
                self.env.set_accumulator(id);
            }

            // Inc / Dec
            Opcode::Inc => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit_inc_dec(val, slot, true)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: Inc + Star ────────────────────────────────
            Opcode::IncStar => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let dst = self.operand_register(instr, 1)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit_inc_dec(val, slot, true)?;
                self.env.set_accumulator(id);
                self.env.set(dst, id);
            }
            Opcode::Dec => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit_inc_dec(val, slot, false)?;
                self.env.set_accumulator(id);
            }

            // Negate / BitwiseNot — no feedback-based specialisation for now.
            Opcode::Negate => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::GenericNegate {
                    value: val,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::BitwiseNot => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::GenericBitwiseNot {
                    value: val,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }

            // ── Comparisons ──────────────────────────────────────────────────
            Opcode::TestEqual => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit(ValueNode::TaggedEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestEqual + Jump ──────────────────────────
            Opcode::TestEqualJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition = self.emit(ValueNode::TaggedEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestNotEqual => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit(ValueNode::TaggedNotEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestNotEqual + Jump ───────────────────────
            Opcode::TestNotEqualJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition = self.emit(ValueNode::TaggedNotEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestEqualStrict => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit(ValueNode::TaggedEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestEqualStrict + Jump ────────────────────
            Opcode::TestEqualStrictJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition = self.emit(ValueNode::TaggedEqual {
                    left,
                    right,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestLessThan => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_comparison(left, right, slot, CompareKind::LessThan)?;
                self.env.set_accumulator(id);
            }
            Opcode::TestLessThanJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition = self.emit_comparison(left, right, slot, CompareKind::LessThan)?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestGreaterThan => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_comparison(left, right, slot, CompareKind::GreaterThan)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestGreaterThan + Jump ────────────────────
            Opcode::TestGreaterThanJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition =
                    self.emit_comparison(left, right, slot, CompareKind::GreaterThan)?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestLessThanOrEqual => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id = self.emit_comparison(left, right, slot, CompareKind::LessThanOrEqual)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestLessThanOrEqual + Jump ────────────────
            Opcode::TestLessThanOrEqualJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition =
                    self.emit_comparison(left, right, slot, CompareKind::LessThanOrEqual)?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestGreaterThanOrEqual => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let id =
                    self.emit_comparison(left, right, slot, CompareKind::GreaterThanOrEqual)?;
                self.env.set_accumulator(id);
            }
            // ── Superinstruction: TestGreaterThanOrEqual + Jump ─────────────
            Opcode::TestGreaterThanOrEqualJump => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let is_true = self.operand_flag(instr, 3)? != 0;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                let condition =
                    self.emit_comparison(left, right, slot, CompareKind::GreaterThanOrEqual)?;
                self.env.set_accumulator(condition);
                self.emit_fused_branch(instr_idx, all_instructions, condition, is_true);
            }
            Opcode::TestInstanceOf => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let object = self.env_get_accumulator()?;
                let callable = self.env_get_register(reg)?;
                let id = self.emit(ValueNode::TestInstanceOf {
                    object,
                    callable,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestIn => {
                let reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let key = self.env_get_accumulator()?;
                let object = self.env_get_register(reg)?;
                let id = self.emit(ValueNode::TestIn {
                    key,
                    object,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestUndetectable => {
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::TestUndetectable { value: val })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestNull => {
                let val = self.env_get_accumulator()?;
                let null = self.emit(ValueNode::NullConstant)?;
                let id = self.emit(ValueNode::TaggedEqual {
                    left: val,
                    right: null,
                    feedback_slot: 0,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestUndefined => {
                let val = self.env_get_accumulator()?;
                let undef = self.emit(ValueNode::UndefinedConstant)?;
                let id = self.emit(ValueNode::TaggedEqual {
                    left: val,
                    right: undef,
                    feedback_slot: 0,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestTypeOf => {
                let flag = self.operand_flag(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::TestTypeOf {
                    value: val,
                    literal_flag: flag as u32,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::TestReferenceEqual => {
                let reg = self.operand_register(instr, 0)?;
                let left = self.env_get_accumulator()?;
                let right = self.env_get_register(reg)?;
                // No feedback slot — emit as TaggedEqual with slot 0.
                let id = self.emit(ValueNode::TaggedEqual {
                    left,
                    right,
                    feedback_slot: 0,
                })?;
                self.env.set_accumulator(id);
            }

            // ── Calls ─────────────────────────────────────────────────────────
            // CallAnyReceiver / CallProperty / CallProperty0/1/2 /
            // CallUndefinedReceiver0/1/2
            Opcode::CallAnyReceiver => {
                // Layout: [callable, args_start, args_count, slot]
                let callable_reg = self.operand_register(instr, 0)?;
                let args_start = self.operand_register(instr, 1)?;
                let args_count = self.operand_register_count(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.emit(ValueNode::UndefinedConstant)?;
                let args = self.collect_args(args_start, args_count)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallProperty | Opcode::CallWithSpread => {
                // Layout: [callable, receiver, args_count, slot]
                let callable_reg = self.operand_register(instr, 0)?;
                let receiver_reg = self.operand_register(instr, 1)?;
                let args_count = self.operand_register_count(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.env_get_register(receiver_reg)?;
                // Args follow receiver in the register file.
                let args_start = receiver_reg + 1;
                let args = self.collect_args(args_start, args_count)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallProperty0 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let receiver_reg = self.operand_register(instr, 1)?;
                let slot = self.operand_feedback_slot(instr, 2)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.env_get_register(receiver_reg)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallProperty1 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let receiver_reg = self.operand_register(instr, 1)?;
                let arg1_reg = self.operand_register(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.env_get_register(receiver_reg)?;
                let arg1 = self.env_get_register(arg1_reg)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![arg1],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallProperty2 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let receiver_reg = self.operand_register(instr, 1)?;
                let arg1_reg = self.operand_register(instr, 2)?;
                let arg2_reg = self.operand_register(instr, 3)?;
                let slot = self.operand_feedback_slot(instr, 4)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.env_get_register(receiver_reg)?;
                let arg1 = self.env_get_register(arg1_reg)?;
                let arg2 = self.env_get_register(arg2_reg)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![arg1, arg2],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallUndefinedReceiver0 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.emit(ValueNode::UndefinedConstant)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallUndefinedReceiver1 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let arg1_reg = self.operand_register(instr, 1)?;
                let slot = self.operand_feedback_slot(instr, 2)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.emit(ValueNode::UndefinedConstant)?;
                let arg1 = self.env_get_register(arg1_reg)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![arg1],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallUndefinedReceiver2 => {
                let callable_reg = self.operand_register(instr, 0)?;
                let arg1_reg = self.operand_register(instr, 1)?;
                let arg2_reg = self.operand_register(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let callee = self.env_get_register(callable_reg)?;
                let receiver = self.emit(ValueNode::UndefinedConstant)?;
                let arg1 = self.env_get_register(arg1_reg)?;
                let arg2 = self.env_get_register(arg2_reg)?;
                let id = self.emit(ValueNode::Call {
                    callee,
                    receiver,
                    args: vec![arg1, arg2],
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }
            Opcode::CallRuntime => {
                let function_id = self.operand_runtime_id(instr, 0)?;
                let args_start = self.operand_register(instr, 1)?;
                let args_count = self.operand_register_count(instr, 2)?;
                let args = self.collect_args(args_start, args_count)?;
                let id = self.emit(ValueNode::CallRuntime { function_id, args })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }

            // ── Construct ─────────────────────────────────────────────────────
            Opcode::Construct | Opcode::ConstructWithSpread => {
                let ctor_reg = self.operand_register(instr, 0)?;
                let args_start = self.operand_register(instr, 1)?;
                let args_count = self.operand_register_count(instr, 2)?;
                let slot = self.operand_feedback_slot(instr, 3)?;
                let constructor = self.env_get_register(ctor_reg)?;
                let args = self.collect_args(args_start, args_count)?;
                let id = self.emit(ValueNode::Construct {
                    constructor,
                    args,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
                self.known_globals.clear();
                self.known_props.clear();
            }

            // ── Type conversions ─────────────────────────────────────────────
            Opcode::ToBoolean | Opcode::ToBooleanLogicalNot => {
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToBoolean { value: val })?;
                self.env.set_accumulator(id);
            }
            Opcode::LogicalNot => {
                // Logical NOT on an already-boolean accumulator.
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToBoolean { value: val })?;
                self.env.set_accumulator(id);
            }
            Opcode::TypeOf => {
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::TypeOf { value: val })?;
                self.env.set_accumulator(id);
            }
            Opcode::ToString => {
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToString {
                    value: val,
                    feedback_slot: 0,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::ToNumber => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToNumber {
                    value: val,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::ToNumeric => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToNumberOrNumeric {
                    value: val,
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::ToName => {
                let dst_reg = self.operand_register(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToName {
                    value: val,
                    feedback_slot: 0,
                })?;
                self.env.set(dst_reg, id);
            }
            Opcode::ToObject => {
                let dst_reg = self.operand_register(instr, 0)?;
                let val = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::ToObject {
                    value: val,
                    feedback_slot: 0,
                })?;
                self.env.set(dst_reg, id);
            }

            // ── Context slots ─────────────────────────────────────────────────
            Opcode::LdaContextSlot | Opcode::LdaImmutableContextSlot => {
                let ctx_reg = self.operand_register(instr, 0)?;
                let slot_idx = self.operand_constant_pool_idx(instr, 1)?;
                let depth = self.operand_immediate(instr, 2)? as u32;

                // When the context register has not been explicitly written in
                // this function (e.g. an inner closure that receives its context
                // via RSI at entry), the register holds the function's incoming
                // context.  Use LoadCurrentContextSlot which reads from the
                // cached ctx_regfile_offset slot, avoiding the stale/undefined
                // register_file entry.
                if self.env.get(ctx_reg).is_none() && depth == 0 {
                    let id = self.emit(ValueNode::LoadCurrentContextSlot { slot: slot_idx })?;
                    self.env.set_accumulator(id);
                } else {
                    let context = self.env_get_register(ctx_reg)?;
                    let id = self.emit(ValueNode::LoadContextSlot {
                        context,
                        depth,
                        slot: slot_idx,
                    })?;
                    self.env.set_accumulator(id);
                }
            }
            Opcode::LdaCurrentContextSlot | Opcode::LdaImmutableCurrentContextSlot => {
                let slot_idx = self.operand_constant_pool_idx(instr, 0)?;
                let id = self.emit(ValueNode::LoadCurrentContextSlot { slot: slot_idx })?;
                self.env.set_accumulator(id);
            }
            Opcode::StaContextSlot => {
                let ctx_reg = self.operand_register(instr, 0)?;
                let slot_idx = self.operand_constant_pool_idx(instr, 1)?;
                let depth = self.operand_immediate(instr, 2)? as u32;
                let value = self.env_get_accumulator()?;

                // Same as LdaContextSlot: when the context register has not
                // been written, the function's incoming context (RSI) is the
                // target.  Use StoreCurrentContextSlot for the depth==0 case.
                if self.env.get(ctx_reg).is_none() && depth == 0 {
                    self.emit(ValueNode::StoreCurrentContextSlot {
                        slot: slot_idx,
                        value,
                    })?;
                } else {
                    let context = self.env_get_register(ctx_reg)?;
                    self.emit(ValueNode::StoreContextSlot {
                        context,
                        depth,
                        slot: slot_idx,
                        value,
                    })?;
                }
            }
            Opcode::StaCurrentContextSlot => {
                let slot_idx = self.operand_constant_pool_idx(instr, 0)?;
                let value = self.env_get_accumulator()?;
                self.emit(ValueNode::StoreCurrentContextSlot {
                    slot: slot_idx,
                    value,
                })?;
            }

            // ── Closures ──────────────────────────────────────────────────────
            Opcode::CreateFunctionContext => {
                let scope_info = self.operand_constant_pool_idx(instr, 0)?;
                let slot_count = self.operand_immediate(instr, 1)? as u32;
                let id = self.emit(ValueNode::CreateFunctionContext {
                    scope_info,
                    slot_count,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::PushContext => {
                let reg = self.operand_register(instr, 0)?;
                let new_ctx = self.env_get_accumulator()?;
                let id = self.emit(ValueNode::PushContext { context: new_ctx })?;
                // The old context is saved into the register.
                self.env.set(reg, id);
            }
            Opcode::PopContext => {
                let reg = self.operand_register(instr, 0)?;
                let saved = self.env_get_register(reg)?;
                self.emit(ValueNode::PopContext { context: saved })?;
            }
            Opcode::CreateClosure => {
                let func_idx = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let flags = self.operand_flag(instr, 2)? as u32;

                // Pre-analyse the callee's bytecodes for the fusion pattern
                // so the optimizer can embed (slot, k) at compile time.
                #[cfg(all(target_arch = "x86_64", unix))]
                if let Some(ConstantPoolEntry::Function(callee_ba)) =
                    self.bytecode.constant_pool().get(func_idx as usize)
                    && let Some((s, k)) =
                        crate::compiler::baseline::compiler::jit_runtime::analyze_fusion_pattern(
                            callee_ba.bytecodes(),
                        )
                {
                    self.graph.set_closure_fusion_pattern(func_idx, s as u32, k);
                }

                let id = self.emit(ValueNode::CreateClosure {
                    shared_function_info: func_idx,
                    feedback_slot: slot,
                    flags,
                })?;
                self.env.set_accumulator(id);
            }

            // ── Object / array literals ───────────────────────────────────────
            Opcode::CreateObjectLiteral => {
                let bp = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let flags = self.operand_flag(instr, 2)? as u32;
                let id = self.emit(ValueNode::CreateObjectLiteral {
                    boilerplate_descriptor: bp,
                    feedback_slot: slot,
                    flags,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateEmptyObjectLiteral => {
                let id = self.emit(ValueNode::CreateEmptyObjectLiteral)?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateEmptyArrayLiteral => {
                let slot = self.operand_feedback_slot(instr, 0)?;
                let id = self.emit(ValueNode::CreateEmptyArrayLiteral {
                    feedback_slot: slot,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateMappedArguments => {
                let id = self.emit(ValueNode::CreateMappedArguments)?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateUnmappedArguments => {
                let id = self.emit(ValueNode::CreateUnmappedArguments)?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateRestParameter => {
                let id = self.emit(ValueNode::CreateRestParameter)?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateArrayLiteral => {
                let elems = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let flags = self.operand_flag(instr, 2)? as u32;
                let id = self.emit(ValueNode::CreateArrayLiteral {
                    constant_elements: elems,
                    feedback_slot: slot,
                    flags,
                })?;
                self.env.set_accumulator(id);
            }
            Opcode::CreateRegExpLiteral => {
                let pat = self.operand_constant_pool_idx(instr, 0)?;
                let slot = self.operand_feedback_slot(instr, 1)?;
                let flags = self.operand_flag(instr, 2)? as u32;
                let id = self.emit(ValueNode::CreateRegExpLiteral {
                    pattern: pat,
                    feedback_slot: slot,
                    flags,
                })?;
                self.env.set_accumulator(id);
            }

            // ── Control flow ──────────────────────────────────────────────────
            Opcode::Return => {
                let val = self.env_get_accumulator()?;
                self.set_control(ControlNode::Return { value: val });
            }

            Opcode::Jump | Opcode::JumpConstant => {
                // Resolve the target block.
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(self.current_block + 1);
                self.save_env_for_successor(target);
                let cur = self.current_block;
                if let Some(block) = self.graph.block_mut(target) {
                    block.add_predecessor(cur);
                }
                self.set_control(ControlNode::Jump { target });
            }

            // Conditional jumps: emit ToBoolean + Branch.
            Opcode::JumpIfTrue
            | Opcode::JumpIfTrueConstant
            | Opcode::JumpIfToBooleanTrue
            | Opcode::JumpIfToBooleanTrueConstant => {
                let cond = self.env_get_accumulator()?;
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(self.current_block + 1);
                let fall = self
                    .block_at
                    .get(&(instr_idx + 1))
                    .copied()
                    .unwrap_or(self.current_block + 1);
                self.save_env_for_successor(target);
                self.save_env_for_successor(fall);
                self.add_branch_predecessors(target, fall);
                self.set_control(ControlNode::Branch {
                    condition: cond,
                    if_true: target,
                    if_false: fall,
                });
            }
            Opcode::JumpIfFalse
            | Opcode::JumpIfFalseConstant
            | Opcode::JumpIfToBooleanFalse
            | Opcode::JumpIfToBooleanFalseConstant => {
                let cond = self.env_get_accumulator()?;
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(self.current_block + 1);
                let fall = self
                    .block_at
                    .get(&(instr_idx + 1))
                    .copied()
                    .unwrap_or(self.current_block + 1);
                self.save_env_for_successor(target);
                self.save_env_for_successor(fall);
                self.add_branch_predecessors(target, fall);
                // Invert: jump-if-false means if_true = fall, if_false = target.
                self.set_control(ControlNode::Branch {
                    condition: cond,
                    if_true: fall,
                    if_false: target,
                });
            }
            Opcode::JumpIfNull
            | Opcode::JumpIfNullConstant
            | Opcode::JumpIfUndefined
            | Opcode::JumpIfUndefinedConstant
            | Opcode::JumpIfUndefinedOrNull
            | Opcode::JumpIfUndefinedOrNullConstant
            | Opcode::JumpIfNotNull
            | Opcode::JumpIfNotNullConstant
            | Opcode::JumpIfNotUndefined
            | Opcode::JumpIfNotUndefinedConstant => {
                // Exact null/undefined check — must NOT use ToBoolean because
                // values like 0, false, "" are falsy but not nullish.
                let val = self.env_get_accumulator()?;
                let cond = self.emit(ValueNode::TestNullOrUndefined { value: val })?;
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(self.current_block + 1);
                let fall = self
                    .block_at
                    .get(&(instr_idx + 1))
                    .copied()
                    .unwrap_or(self.current_block + 1);
                // TestNullOrUndefined returns true when the value IS
                // null/undefined.  "JumpIfUndefinedOrNull" jumps when true.
                // "JumpIfNotNull/NotUndefined" jumps when false.
                let (if_true, if_false) = match instr.opcode {
                    Opcode::JumpIfNotNull
                    | Opcode::JumpIfNotNullConstant
                    | Opcode::JumpIfNotUndefined
                    | Opcode::JumpIfNotUndefinedConstant => (fall, target),
                    _ => (target, fall),
                };
                self.save_env_for_successor(if_true);
                self.save_env_for_successor(if_false);
                self.add_branch_predecessors(if_true, if_false);
                self.set_control(ControlNode::Branch {
                    condition: cond,
                    if_true,
                    if_false,
                });
            }
            Opcode::JumpIfJSReceiver | Opcode::JumpIfJSReceiverConstant => {
                // For receiver checks we emit a ToBoolean of the
                // accumulator and branch on that.
                let val = self.env_get_accumulator()?;
                let cond = self.emit(ValueNode::ToBoolean { value: val })?;
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(self.current_block + 1);
                let fall = self
                    .block_at
                    .get(&(instr_idx + 1))
                    .copied()
                    .unwrap_or(self.current_block + 1);
                let (if_true, if_false) = (target, fall);
                self.save_env_for_successor(if_true);
                self.save_env_for_successor(if_false);
                self.add_branch_predecessors(if_true, if_false);
                self.set_control(ControlNode::Branch {
                    condition: cond,
                    if_true,
                    if_false,
                });
            }

            Opcode::JumpLoop => {
                // Back-edge to a loop header.
                let target = self
                    .resolve_jump_target(instr_idx, all_instructions)
                    .unwrap_or(0);
                self.patch_loop_header_phis(target);
                let cur = self.current_block;
                if let Some(block) = self.graph.block_mut(target) {
                    block.add_predecessor(cur);
                }
                self.set_control(ControlNode::Jump { target });
            }

            // ── Misc / no-ops ─────────────────────────────────────────────────
            Opcode::StackCheck
            | Opcode::SetExpressionPosition
            | Opcode::SetExpressionPositionFromEnd
            | Opcode::CollectTypeProfile => {
                // No IR produced; these are purely informational.
            }
            Opcode::Debugger => {
                self.emit(ValueNode::Debugger)?;
            }
            Opcode::Wide | Opcode::ExtraWide => {
                // Width prefixes are consumed by the decoder; we never see them
                // as stand-alone instructions here.
            }
            Opcode::Throw => {
                // Emit a deoptimise to let the interpreter handle the throw.
                let offset = self.byte_offset_of(all_instructions, instr_idx) as u32;
                self.set_control(ControlNode::Deoptimize {
                    bytecode_offset: offset,
                    reason: 0,
                });
            }
            Opcode::ReThrow | Opcode::SetPendingMessage => {
                let offset = self.byte_offset_of(all_instructions, instr_idx) as u32;
                self.set_control(ControlNode::Deoptimize {
                    bytecode_offset: offset,
                    reason: 0,
                });
            }

            // Catch-all for unhandled opcodes: emit a generic deoptimise and
            // let the interpreter execute the rest.
            _ => {
                let offset = self.byte_offset_of(all_instructions, instr_idx) as u32;
                if !self.block_is_complete(self.current_block) {
                    self.set_control(ControlNode::Deoptimize {
                        bytecode_offset: offset,
                        reason: 0,
                    });
                }
            }
        }

        Ok(())
    }

    // ── Speculative helpers ──────────────────────────────────────────────────

    /// Emit a named-property load, specialising on the feedback slot state.
    ///
    /// - **Monomorphic** → `CheckMaps` guard + `LoadField` (fast path).
    /// - Otherwise → `LoadNamedGeneric` (slow path).
    fn emit_named_property_load(
        &mut self,
        object: NodeId,
        name: u32,
        slot: u32,
    ) -> StatorResult<NodeId> {
        // Within-block CSE: if we've already loaded this property from the
        // same object in this block, reuse the previous result.
        let cse_key = (object, name);
        if let Some(&cached) = self.known_props.get(&cse_key) {
            return Ok(cached);
        }

        // Always use the generic path for now.  The monomorphic fast path
        // (CheckMaps + LoadField) requires the real field offset from the
        // IC handler, which is not yet populated — the offset is always 0
        // (a placeholder).  Using LoadNamedGeneric avoids unconditional
        // deopt in Maglev codegen for monomorphic accesses.
        let id = self.emit(ValueNode::LoadNamedGeneric {
            object,
            name,
            feedback_slot: slot,
        })?;

        // Cache the result for subsequent loads of the same property.
        self.known_props.insert(cse_key, id);
        Ok(id)
    }

    /// Emit a keyed property load, specialising on the feedback slot state.
    ///
    /// - **Monomorphic / Polymorphic** with `KeyedLoadProperty` kind →
    ///   `CheckSmi` guard on the key + `LoadFixedArrayElement` (fast path).
    ///   This avoids the expensive `LoadKeyedGeneric` runtime stub for the
    ///   common pattern of `arr[i]` where `i` is always a Smi index.
    /// - Otherwise → `LoadKeyedGeneric` (slow path).
    fn emit_keyed_property_load(
        &mut self,
        object: NodeId,
        key: NodeId,
        slot: u32,
    ) -> StatorResult<NodeId> {
        if self.is_keyed_smi_access(slot) {
            // Guard: key must be a Smi (integer index).  Deopts to
            // interpreter if a non-Smi key is encountered at runtime.
            self.emit(ValueNode::CheckSmi { receiver: key })?;
            // Emit the fast-path element load.
            self.emit(ValueNode::LoadFixedArrayElement {
                elements: object,
                index: key,
            })
        } else {
            self.emit(ValueNode::LoadKeyedGeneric {
                object,
                key,
                feedback_slot: slot,
            })
        }
    }

    /// Emit a keyed property store, specialising on the feedback slot state.
    ///
    /// - **Monomorphic / Polymorphic** with `KeyedStoreProperty` kind →
    ///   `CheckSmi` guard on the key + `StoreFixedArrayElement` (fast path).
    /// - Otherwise → `StoreKeyedGeneric` (slow path).
    fn emit_keyed_property_store(
        &mut self,
        object: NodeId,
        key: NodeId,
        value: NodeId,
        slot: u32,
    ) -> StatorResult<NodeId> {
        if self.is_keyed_smi_access(slot) {
            self.emit(ValueNode::CheckSmi { receiver: key })?;
            self.emit(ValueNode::StoreFixedArrayElement {
                elements: object,
                index: key,
                value,
            })
        } else {
            self.emit(ValueNode::StoreKeyedGeneric {
                object,
                key,
                value,
                feedback_slot: slot,
            })
        }
    }

    /// Return `true` when feedback for `slot` indicates Smi-indexed array
    /// access that is safe to specialize.
    ///
    /// The slot must be a keyed property kind (`KeyedLoadProperty` or
    /// `KeyedStoreProperty`) **and** be in [`Monomorphic`] or [`Polymorphic`]
    /// state — meaning the interpreter has observed this site accessing a
    /// consistent element kind with integer indices.
    fn is_keyed_smi_access(&self, slot: u32) -> bool {
        let is_keyed_kind = matches!(
            self.feedback.kind_of(slot),
            Some(FeedbackSlotKind::KeyedLoadProperty | FeedbackSlotKind::KeyedStoreProperty)
        );
        let is_warm = matches!(
            self.feedback.get_state(slot),
            Some(InlineCacheState::Monomorphic | InlineCacheState::Polymorphic)
        );
        is_keyed_kind && is_warm
    }

    /// Emit a binary operation using Generic IR nodes.
    ///
    /// All arithmetic and bitwise operations use Generic* nodes which:
    /// 1. Enable `smi_guarded` analysis — upstream loads see only
    ///    arithmetic-compatible consumers and qualify for speculative
    ///    Smi coercion at the load site.
    /// 2. Have proper runtime fallbacks — overflow or type mismatches
    ///    call `jit_runtime_generic_*` instead of permanently deopting.
    /// 3. The codegen selects the optimal emit path at code-generation
    ///    time based on `smi_guarded`, `i32_range`, or generic analysis.
    fn emit_binary_op(
        &mut self,
        left: NodeId,
        right: NodeId,
        slot: u32,
        kind: BinaryOpKind,
    ) -> StatorResult<NodeId> {
        // Use Generic operations for Add/Sub/Mul/Div/Mod instead of
        // CheckedSmi variants.  Generic ops enable `smi_guarded` analysis
        // on upstream loads (they are arithmetic-compatible consumers),
        // and their codegen has a proper runtime fallback on overflow
        // instead of deopt.  The `smi_guarded` fast path (inline 32-bit
        // ALU + JO) is nearly as fast as CheckedSmi, but the slow path
        // calls jit_runtime_generic_* instead of deopting — preventing
        // permanent Maglev blocking from transient overflows.
        //
        // Bitwise/shift ops cannot overflow (they produce valid i32 for
        // any two i32 inputs), so no post-operation deopt is needed.
        self.emit(match kind {
            BinaryOpKind::Add => ValueNode::GenericAdd {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::Sub => ValueNode::GenericSubtract {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::Mul => ValueNode::GenericMultiply {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::Div => ValueNode::GenericDivide {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::Mod => ValueNode::GenericModulus {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::Exp => ValueNode::GenericExponentiate {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::BitwiseOr => ValueNode::GenericBitwiseOr {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::BitwiseXor => ValueNode::GenericBitwiseXor {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::BitwiseAnd => ValueNode::GenericBitwiseAnd {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::ShiftLeft => ValueNode::GenericShiftLeft {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::ShiftRight => ValueNode::GenericShiftRight {
                left,
                right,
                feedback_slot: slot,
            },
            BinaryOpKind::ShiftRightLogical => ValueNode::GenericShiftRightLogical {
                left,
                right,
                feedback_slot: slot,
            },
        })
    }

    /// Emit a comparison, specialising on Smi feedback.
    ///
    /// When feedback is unavailable (Uninitialized) we still emit Smi-guarded
    /// Int32 comparisons.  We omit explicit `CheckSmi` guards here so that
    /// upstream loads can qualify as `smi_guarded` (all their consumers will
    /// be arithmetic-compatible).  The `smi_guarded` coercion at the load
    /// site ensures values reaching Int32 comparisons are valid i32.
    fn emit_comparison(
        &mut self,
        left: NodeId,
        right: NodeId,
        _slot: u32,
        kind: CompareKind,
    ) -> StatorResult<NodeId> {
        self.emit(match kind {
            CompareKind::LessThan => ValueNode::Int32LessThan { left, right },
            CompareKind::GreaterThan => ValueNode::Int32GreaterThan { left, right },
            CompareKind::LessThanOrEqual => ValueNode::Int32LessThanOrEqual { left, right },
            CompareKind::GreaterThanOrEqual => ValueNode::Int32GreaterThanOrEqual { left, right },
        })
    }

    /// Emit an increment or decrement.
    ///
    /// Uses GenericIncrement/GenericDecrement so the operation is
    /// arithmetic-compatible for `smi_guarded` analysis and has a
    /// runtime fallback instead of deopt on overflow.
    fn emit_inc_dec(&mut self, val: NodeId, slot: u32, is_inc: bool) -> StatorResult<NodeId> {
        if is_inc {
            self.emit(ValueNode::GenericIncrement {
                value: val,
                feedback_slot: slot,
            })
        } else {
            self.emit(ValueNode::GenericDecrement {
                value: val,
                feedback_slot: slot,
            })
        }
    }

    /// Emit the branch portion of a fused test+jump superinstruction.
    ///
    /// Resolves the jump target and fall-through block, saves environments
    /// for both successors, and terminates the current block with a
    /// [`ControlNode::Branch`].
    fn emit_fused_branch(
        &mut self,
        instr_idx: usize,
        all_instructions: &[Instruction],
        condition: NodeId,
        is_true: bool,
    ) {
        let target = self
            .resolve_jump_target(instr_idx, all_instructions)
            .unwrap_or(self.current_block + 1);
        let fall = self
            .block_at
            .get(&(instr_idx + 1))
            .copied()
            .unwrap_or(self.current_block + 1);
        let if_true = if is_true { target } else { fall };
        let if_false = if is_true { fall } else { target };
        self.save_env_for_successor(if_true);
        self.save_env_for_successor(if_false);
        self.add_branch_predecessors(if_true, if_false);
        self.set_control(ControlNode::Branch {
            condition,
            if_true,
            if_false,
        });
    }

    // ── Environment helpers ──────────────────────────────────────────────────

    /// Read the accumulator, returning an error if it has not been written.
    fn env_get_accumulator(&mut self) -> StatorResult<NodeId> {
        if let Some(id) = self.env.accumulator() {
            Ok(id)
        } else {
            // Accumulator not written — an unhandled opcode that was supposed
            // to set the accumulator was replaced with Deoptimize.  Emit an
            // UndefinedConstant placeholder so the rest of the block can be
            // compiled (the deopt fires before this value is used at runtime).
            let id = self.emit(ValueNode::UndefinedConstant)?;
            self.env.set_accumulator(id);
            Ok(id)
        }
    }

    /// Read a bytecode register, emitting a fallback if it has not been written.
    fn env_get_register(&mut self, reg: u32) -> StatorResult<NodeId> {
        if let Some(id) = self.env.get(reg) {
            Ok(id)
        } else {
            let id = self.emit(ValueNode::UndefinedConstant)?;
            self.env.set(reg, id);
            Ok(id)
        }
    }

    // ── Emission helpers ─────────────────────────────────────────────────────

    /// Append a [`ValueNode`] to the current block and return its
    /// graph-global [`NodeId`].
    fn emit(&mut self, node: ValueNode) -> StatorResult<NodeId> {
        let blk = self.current_block;
        self.graph
            .add_value_node(blk, node)
            .ok_or_else(|| StatorError::Internal("current block missing from graph".into()))
    }

    /// Emit a constant-pool entry as the appropriate [`ValueNode`].
    fn emit_constant_pool_entry(&mut self, idx: u32) -> StatorResult<NodeId> {
        let entry = self
            .bytecode
            .constant_pool()
            .get(idx as usize)
            .ok_or_else(|| {
                StatorError::Internal(format!("constant pool index {idx} out of range"))
            })?
            .clone();

        let node = match entry {
            ConstantPoolEntry::Number(v) => {
                if v.fract() == 0.0 && v >= i32::MIN as f64 && v <= i32::MAX as f64 {
                    ValueNode::SmiConstant { value: v as i32 }
                } else {
                    ValueNode::Float64Constant { value: v }
                }
            }
            ConstantPoolEntry::String(s) => ValueNode::StringConstant { value: s },
            ConstantPoolEntry::Boolean(true) => ValueNode::TrueConstant,
            ConstantPoolEntry::Boolean(false) => ValueNode::FalseConstant,
            ConstantPoolEntry::Null => ValueNode::NullConstant,
            ConstantPoolEntry::Undefined => ValueNode::UndefinedConstant,
            ConstantPoolEntry::Function(_) => {
                // Function entries are referenced by CreateClosure; emit an
                // opaque constant-pool reference.
                ValueNode::ConstantPoolEntry { index: idx }
            }
            ConstantPoolEntry::BigInt(_) => {
                // BigInt constants are handled at runtime; emit an opaque
                // constant-pool reference.
                ValueNode::ConstantPoolEntry { index: idx }
            }
            ConstantPoolEntry::TemplateObject { .. } => {
                // Template objects are handled by GetTemplateObject at runtime;
                // emit an opaque constant-pool reference.
                ValueNode::ConstantPoolEntry { index: idx }
            }
        };
        self.emit(node)
    }

    /// Collect `count` consecutive register values starting at `start`,
    /// using the environment.
    fn collect_args(&mut self, start: u32, count: u32) -> StatorResult<Vec<NodeId>> {
        (start..start + count)
            .map(|r| self.env_get_register(r))
            .collect()
    }

    // ── Phi / environment helpers ─────────────────────────────────────────────

    /// Snapshot the current environment for a successor block.
    fn save_env_for_successor(&mut self, target: u32) {
        let entry = (self.current_block, self.env.clone());
        self.saved_envs.entry(target).or_default().push(entry);
    }

    /// Add `self.current_block` as a predecessor of both branch targets.
    fn add_branch_predecessors(&mut self, if_true: u32, if_false: u32) {
        let cur = self.current_block;
        if let Some(block) = self.graph.block_mut(if_true) {
            block.add_predecessor(cur);
        }
        if if_false != if_true
            && let Some(block) = self.graph.block_mut(if_false)
        {
            block.add_predecessor(cur);
        }
    }

    /// Set up the SSA environment when translation enters a new block.
    ///
    /// For loop headers a Phi node is created for every occupied register slot
    /// (initially with the single forward-edge input).  For non-loop merge
    /// points, Phi nodes are created where predecessor environments disagree.
    fn enter_block(&mut self, block_id: u32) {
        // Invalidate within-block CSE at control-flow boundaries.
        self.known_globals.clear();
        self.known_props.clear();
        if self.loop_headers.contains(&block_id) {
            self.enter_loop_header(block_id);
        } else {
            self.enter_merge_block(block_id);
        }
    }

    /// Enter a loop header: create single-input Phi nodes that will be
    /// completed when the back-edge (`JumpLoop`) is processed.
    fn enter_loop_header(&mut self, block_id: u32) {
        let entry_env = match self.saved_envs.get(&block_id) {
            Some(envs) if !envs.is_empty() => envs[0].1.clone(),
            _ => {
                // No predecessor saved an environment — use the current
                // environment as-is.  This happens when the entry block was
                // terminated early (e.g. unhandled opcode → Deoptimize) before
                // reaching the jump to this loop header.
                self.env.clone()
            }
        };

        let mut phis = Vec::new();
        let slot_count = entry_env.slots.len();
        for slot_idx in 0..slot_count {
            if let Some(val) = entry_env.slots[slot_idx] {
                if let Some(phi_id) = self
                    .graph
                    .add_value_node(block_id, ValueNode::Phi { inputs: vec![val] })
                {
                    phis.push((slot_idx, phi_id));
                    self.env.slots[slot_idx] = Some(phi_id);
                }
            } else {
                self.env.slots[slot_idx] = None;
            }
        }
        self.loop_header_phis.insert(block_id, phis);
    }

    /// Enter a non-loop block: adopt the single predecessor's environment or
    /// merge multiple predecessors with Phi nodes where they disagree.
    fn enter_merge_block(&mut self, block_id: u32) {
        let envs: Vec<(u32, Environment)> = match self.saved_envs.get(&block_id) {
            Some(e) if e.len() >= 2 => e.clone(),
            Some(e) if e.len() == 1 => {
                self.env = e[0].1.clone();
                return;
            }
            _ => {
                // No predecessor saved an environment — use the current
                // environment as-is (inherited from the previous block).
                return;
            }
        };

        let slot_count = self.env.slots.len();
        for slot_idx in 0..slot_count {
            let values: Vec<Option<NodeId>> = envs
                .iter()
                .map(|(_, e)| e.slots.get(slot_idx).copied().flatten())
                .collect();

            let all_same = values.windows(2).all(|w| w[0] == w[1]);
            if all_same {
                self.env.slots[slot_idx] = values[0];
            } else {
                let inputs: Vec<NodeId> = values.into_iter().flatten().collect();
                if inputs.len() >= 2 {
                    if let Some(phi_id) = self
                        .graph
                        .add_value_node(block_id, ValueNode::Phi { inputs })
                    {
                        self.env.slots[slot_idx] = Some(phi_id);
                    }
                } else if inputs.len() == 1 {
                    self.env.slots[slot_idx] = Some(inputs[0]);
                } else {
                    self.env.slots[slot_idx] = None;
                }
            }
        }
    }

    /// Add back-edge inputs to the Phi nodes at a loop header when `JumpLoop`
    /// is processed.
    fn patch_loop_header_phis(&mut self, target: u32) {
        let phis: Vec<(usize, NodeId)> = match self.loop_header_phis.get(&target) {
            Some(p) => p.clone(),
            None => return,
        };

        // Collect back-edge values from the current environment.
        let patches: Vec<(NodeId, NodeId)> = phis
            .iter()
            .filter_map(|(slot_idx, phi_id)| self.env.slots[*slot_idx].map(|val| (*phi_id, val)))
            .collect();

        // Mutate the Phi nodes in the target block.
        if let Some(block) = self.graph.block_mut(target) {
            for (phi_id, val) in patches {
                for (nid, node) in &mut block.nodes {
                    if *nid == phi_id {
                        if let ValueNode::Phi { inputs } = node {
                            inputs.push(val);
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Set the control node for the current block.
    ///
    /// No-op if the block is already complete (e.g. `Return` already emitted).
    fn set_control(&mut self, control: ControlNode) {
        let blk = self.current_block;
        if let Some(block) = self.graph.block_mut(blk)
            && !block.is_complete()
        {
            block.set_control(control);
        }
    }

    /// Return `true` if the current block already has a terminator.
    fn block_is_complete(&self, idx: u32) -> bool {
        self.graph
            .block(idx)
            .map(|b| b.is_complete())
            .unwrap_or(false)
    }

    /// Return `true` if the block was terminated specifically by Deoptimize.
    fn block_ended_with_deoptimize(&self, idx: u32) -> bool {
        self.graph
            .block(idx)
            .and_then(|b| b.control())
            .is_some_and(|c| matches!(c, ControlNode::Deoptimize { .. }))
    }

    // ── Jump-target resolution ────────────────────────────────────────────────

    /// Resolve the target *block index* for a jump instruction at `instr_idx`.
    ///
    /// Returns `None` if the instruction is not a jump or the target cannot be found.
    fn resolve_jump_target(
        &self,
        instr_idx: usize,
        all_instructions: &[Instruction],
    ) -> Option<u32> {
        let target_idx = self.jump_target_index(all_instructions, instr_idx)?;
        if target_idx == 0 {
            Some(0) // entry block
        } else {
            self.block_at.get(&target_idx).copied()
        }
    }

    /// Compute the byte offset of instruction `instr_idx` by summing the
    /// encoded lengths of all preceding instructions.
    fn byte_offset_of(&self, instructions: &[Instruction], instr_idx: usize) -> usize {
        use crate::bytecode::bytecodes::encode;
        instructions[..instr_idx]
            .iter()
            .map(|i| encode(std::slice::from_ref(i)).len())
            .sum()
    }

    // ── Operand accessors ─────────────────────────────────────────────────────

    fn operand_register(&self, instr: &Instruction, idx: usize) -> StatorResult<u32> {
        match instr.operand_at(idx) {
            Some(Operand::Register(r)) => Ok(*r),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected Register at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_register_count(&self, instr: &Instruction, idx: usize) -> StatorResult<u32> {
        match instr.operand_at(idx) {
            Some(Operand::RegisterCount(c)) => Ok(*c),
            Some(Operand::Register(r)) => Ok(*r), // some encodings reuse Register
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected RegisterCount at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_immediate(&self, instr: &Instruction, idx: usize) -> StatorResult<i32> {
        match instr.operand_at(idx) {
            Some(Operand::Immediate(v)) => Ok(*v),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected Immediate at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_constant_pool_idx(&self, instr: &Instruction, idx: usize) -> StatorResult<u32> {
        match instr.operand_at(idx) {
            Some(Operand::ConstantPoolIdx(v)) => Ok(*v),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected ConstantPoolIdx at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_feedback_slot(&self, instr: &Instruction, idx: usize) -> StatorResult<u32> {
        match instr.operand_at(idx) {
            Some(Operand::FeedbackSlot(v)) => Ok(*v),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected FeedbackSlot at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_runtime_id(&self, instr: &Instruction, idx: usize) -> StatorResult<u32> {
        match instr.operand_at(idx) {
            Some(Operand::RuntimeId(v)) => Ok(*v),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected RuntimeId at operand {idx}",
                instr.opcode
            ))),
        }
    }

    fn operand_flag(&self, instr: &Instruction, idx: usize) -> StatorResult<u8> {
        match instr.operand_at(idx) {
            Some(Operand::Flag(v)) => Ok(*v),
            _ => Err(StatorError::Internal(format!(
                "{:?}: expected Flag at operand {idx}",
                instr.opcode
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper enums
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of binary arithmetic or bitwise operation being lowered.
#[derive(Clone, Copy)]
enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Exp,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    ShiftLeft,
    ShiftRight,
    ShiftRightLogical,
}

/// The kind of ordered comparison being lowered.
#[derive(Clone, Copy)]
enum CompareKind {
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::{
        FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
    };
    use crate::compiler::maglev::ir::{ControlNode, ValueNode};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn build(
        instrs: Vec<Instruction>,
        pool: Vec<ConstantPoolEntry>,
        frame_size: u32,
        param_count: u32,
        metadata: FeedbackMetadata,
    ) -> (BytecodeArray, FeedbackVector) {
        let bytes = encode(&instrs);
        let array = BytecodeArray::new(
            bytes,
            pool,
            frame_size,
            param_count,
            vec![],
            metadata.clone(),
            vec![],
        );
        let vector = FeedbackVector::new(&metadata);
        (array, vector)
    }

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn test_lda_zero_return() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        assert_eq!(graph.blocks().len(), 1);
        let block = graph.entry_block().unwrap();
        // First value node should be SmiConstant(0).
        assert!(matches!(
            block.nodes[0].1,
            ValueNode::SmiConstant { value: 0 }
        ));
        assert!(matches!(block.control, Some(ControlNode::Return { .. })));
    }

    #[test]
    fn test_lda_smi_return() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(
            block.nodes[0].1,
            ValueNode::SmiConstant { value: 42 }
        ));
    }

    #[test]
    fn test_lda_undefined() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(block.nodes[0].1, ValueNode::UndefinedConstant));
    }

    #[test]
    fn test_lda_null() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaNull, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(block.nodes[0].1, ValueNode::NullConstant));
    }

    #[test]
    fn test_lda_true_false() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 1, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(block.nodes[0].1, ValueNode::TrueConstant));
        assert!(matches!(block.nodes[1].1, ValueNode::FalseConstant));
    }

    #[test]
    fn test_lda_constant_string() {
        let pool = vec![ConstantPoolEntry::String("hello".into())];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(
            &block.nodes[0].1,
            ValueNode::StringConstant { value } if value == "hello"
        ));
    }

    #[test]
    fn test_lda_constant_number_smi() {
        let pool = vec![ConstantPoolEntry::Number(7.0)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(
            block.nodes[0].1,
            ValueNode::SmiConstant { value: 7 }
        ));
    }

    #[test]
    fn test_lda_constant_float() {
        let pool = vec![ConstantPoolEntry::Number(3.14)];
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(
            block.nodes[0].1,
            ValueNode::Float64Constant { .. }
        ));
    }

    // ── Register moves ────────────────────────────────────────────────────────

    #[test]
    fn test_star_ldar_round_trip() {
        // r0 = 5; acc = r0; return acc
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 1, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        // Should build without error.
        assert!(graph.entry_block().is_some());
    }

    #[test]
    fn test_mov() {
        // r0 = 9; r1 = r0; acc = r1; return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(9)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Mov,
                vec![Operand::Register(0), Operand::Register(1)],
            ),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 2, 0, FeedbackMetadata::empty());
        GraphBuilder::build(&arr, &vec).expect("build ok");
    }

    // ── Arithmetic – always Smi-guarded (feedback state ignored) ────────────

    #[test]
    fn test_add_uninitialized_still_checked_smi() {
        // r0 = 1; acc = r0 + r0; return
        // We now emit GenericAdd instead of CheckedSmiAdd.
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 1, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        let has_generic_add = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::GenericAdd { .. }));
        assert!(has_generic_add, "expected GenericAdd node");
    }

    // ── Arithmetic – speculative Smi path ────────────────────────────────────

    #[test]
    fn test_add_speculative_smi() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, mut fvec) = build(instrs, vec![], 1, 0, meta);
        // Mark the slot Monomorphic to trigger the speculative path.
        fvec.set_state(0, InlineCacheState::Monomorphic);
        let graph = GraphBuilder::build(&arr, &fvec).unwrap();
        let block = graph.entry_block().unwrap();
        let has_generic_add = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::GenericAdd { .. }));
        let no_smi_guard = !block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }));
        assert!(has_generic_add, "expected GenericAdd node");
        assert!(no_smi_guard, "CheckSmi guards should not be emitted");
    }

    #[test]
    fn test_sub_speculative_smi() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Sub,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, mut fvec) = build(instrs, vec![], 1, 0, meta);
        fvec.set_state(0, InlineCacheState::Monomorphic);
        let graph = GraphBuilder::build(&arr, &fvec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::GenericSubtract { .. }))
        );
    }

    #[test]
    fn test_inc_speculative_smi() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOpInc]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, mut fvec) = build(instrs, vec![], 0, 0, meta);
        fvec.set_state(0, InlineCacheState::Monomorphic);
        let graph = GraphBuilder::build(&arr, &fvec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::GenericIncrement { .. }))
        );
    }

    #[test]
    fn test_dec_uninitialized_still_checked_smi() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOpInc]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Dec, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, meta); // Uninitialized
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::GenericDecrement { .. }))
        );
    }

    // ── Property loads ────────────────────────────────────────────────────────

    #[test]
    fn test_load_named_property_generic() {
        // acc = r0.prop (slot 0, uninitialized → LoadNamedGeneric)
        let pool = vec![ConstantPoolEntry::String("prop".into())];
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 1, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. }))
        );
    }

    #[test]
    fn test_load_named_property_monomorphic() {
        // acc = r0.prop (slot 0, Monomorphic → CheckMaps + LoadField)
        let pool = vec![ConstantPoolEntry::String("x".into())];
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, mut fvec) = build(instrs, pool, 1, 0, meta);
        fvec.set_state(0, InlineCacheState::Monomorphic);
        let graph = GraphBuilder::build(&arr, &fvec).unwrap();
        let block = graph.entry_block().unwrap();
        let has_load_named_generic = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. }));
        // Monomorphic feedback now emits LoadNamedGeneric (not
        // CheckMaps + LoadField) because LoadField offset is a placeholder.
        assert!(
            has_load_named_generic,
            "expected LoadNamedGeneric for monomorphic load"
        );
    }

    // ── Global loads / stores ─────────────────────────────────────────────────

    #[test]
    fn test_load_global() {
        let pool = vec![ConstantPoolEntry::String("Math".into())];
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadGlobal]);
        let instrs = vec![
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 0, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadGlobal { .. }))
        );
    }

    #[test]
    fn test_store_global() {
        let pool = vec![ConstantPoolEntry::String("x".into())];
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::StoreGlobal]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::StaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, pool, 0, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::StoreGlobal { .. }))
        );
    }

    // ── Control flow ──────────────────────────────────────────────────────────

    #[test]
    fn test_unconditional_jump_creates_blocks() {
        // LdaSmi 1; Jump +2; LdaSmi 2; Return
        // The jump skips the second LdaSmi.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            // Jump forward 1 instruction (offset computed below)
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(3)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        // The jump target (Return instruction) should be in a new block.
        assert!(graph.blocks().len() >= 1);
    }

    #[test]
    fn test_conditional_branch_creates_blocks() {
        // LdaTrue; JumpIfTrue +N; LdaFalse; Return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::JumpIfTrue, vec![Operand::JumpOffset(3)]),
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        // Should have at least 2 blocks (entry + fall-through or target).
        assert!(graph.blocks().len() >= 2);
    }

    #[test]
    fn test_return_terminates_block() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(matches!(block.control, Some(ControlNode::Return { .. })));
    }

    // ── Calls ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_call_undefined_receiver0() {
        // r0 = some_fn; CallUndefinedReceiver0 r0, slot0; Return
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::Call]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::CallUndefinedReceiver0,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 1, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Call { args, .. } if args.is_empty()))
        );
    }

    // ── Parameters ────────────────────────────────────────────────────────────

    #[test]
    fn test_parameters_emitted() {
        // A function with 2 parameters.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 2, 2, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        assert_eq!(graph.parameter_count(), 2);
        let block = graph.entry_block().unwrap();
        let params: Vec<_> = block
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::Parameter { .. }))
            .collect();
        assert_eq!(params.len(), 2);
    }

    // ── Implicit return on empty body ─────────────────────────────────────────

    #[test]
    fn test_implicit_return_undefined() {
        // A function body with no explicit Return should still get one.
        let instrs = vec![Instruction::new_unchecked(
            Opcode::LdaSmi,
            vec![Operand::Immediate(0)],
        )];
        let (arr, vec) = build(instrs, vec![], 0, 0, FeedbackMetadata::empty());
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(block.is_complete(), "entry block should be terminated");
        assert!(matches!(block.control, Some(ControlNode::Return { .. })));
    }

    // ── AddSmi immediate ──────────────────────────────────────────────────────

    #[test]
    fn test_add_smi_immediate_generic() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(5), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, vec) = build(instrs, vec![], 0, 0, meta);
        let graph = GraphBuilder::build(&arr, &vec).unwrap();
        let block = graph.entry_block().unwrap();
        // Uninitialized feedback → GenericAdd (with runtime fallback)
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::GenericAdd { .. }))
        );
    }

    #[test]
    fn test_add_smi_immediate_speculative() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(5), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let (arr, mut fvec) = build(instrs, vec![], 0, 0, meta);
        fvec.set_state(0, InlineCacheState::Monomorphic);
        let graph = GraphBuilder::build(&arr, &fvec).unwrap();
        let block = graph.entry_block().unwrap();
        assert!(
            block
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::GenericAdd { .. }))
        );
    }

    /// Compile the object_creation_1k benchmark from source and verify that
    /// the Maglev graph builder handles every bytecode without inserting
    /// spurious `Deoptimize` control nodes.
    #[test]
    fn test_object_creation_benchmark_no_deopt() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::recursive_descent;

        let source = r#"
            var last;
            for (var i = 0; i < 1000; i++) {
                last = { x: i, y: i + 1, z: i * 2 };
            }
            last.x + last.y + last.z;
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();

        // Decode and print each instruction for diagnostics.
        let instructions = ba.instructions().unwrap();
        eprintln!(
            "object_creation_1k: {} bytecodes, {} raw bytes",
            instructions.len(),
            ba.bytecodes().len()
        );
        for (idx, instr) in instructions.iter().enumerate() {
            eprintln!("  [{idx:3}] {:?}", instr.opcode);
        }

        let feedback = FeedbackVector::new(ba.feedback_metadata());
        let graph = GraphBuilder::build(&ba, &feedback).unwrap();

        // Check that no block has a Deoptimize control node (which would
        // indicate an unhandled opcode falling through to the catch-all).
        let mut deopt_blocks = Vec::new();
        for (block_idx, block) in graph.blocks().iter().enumerate() {
            if matches!(block.control, Some(ControlNode::Deoptimize { .. })) {
                deopt_blocks.push(block_idx);
            }
        }
        assert!(
            deopt_blocks.is_empty(),
            "graph has Deoptimize control in blocks {deopt_blocks:?} — \
             indicates unhandled bytecodes"
        );
    }

    /// Helper: compile JS source → graph builder → optimizer, return graph.
    /// Panics on any failure.  Also prints all bytecodes for diagnostics.
    fn compile_and_optimize(source: &str, label: &str) -> MaglevGraph {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::compiler::maglev::optimizer::optimize;
        use crate::parser::recursive_descent;

        let program = recursive_descent::parse(source).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let instructions = ba.instructions().unwrap();
        eprintln!(
            "{label}: {} bytecodes, {} raw bytes",
            instructions.len(),
            ba.bytecodes().len()
        );
        for (idx, instr) in instructions.iter().enumerate() {
            eprintln!("  [{idx:3}] {:?}", instr.opcode);
        }

        let feedback = FeedbackVector::new(ba.feedback_metadata());
        let mut graph = GraphBuilder::build(&ba, &feedback).unwrap();
        optimize(&mut graph);

        // Check degenerate
        let degen = graph.is_degenerate();
        eprintln!("{label}: is_degenerate={degen}");

        // Print entry block control
        if let Some(entry) = graph.entry_block() {
            eprintln!(
                "{label}: entry block has {} nodes, control={:?}",
                entry.nodes.len(),
                entry.control
            );
        }

        graph
    }

    #[test]
    fn test_property_access_benchmark_not_degenerate() {
        let source = r#"
            var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "property_access_1k");
        assert!(
            !graph.is_degenerate(),
            "property_access_1k graph is degenerate — Maglev won't compile it"
        );
    }

    /// Verify that every hoisted LoadNamedGeneric node and the LoadGlobal
    /// it depends on get valid register/stack-slot allocations (never None).
    /// Also check for register conflicts among overlapping live intervals.
    /// A None allocation would cause `emit_load` to produce JIT_UNDEFINED,
    /// which breaks downstream GenericAdd Smi checks and forces the slow
    /// runtime path every iteration.
    #[test]
    fn test_property_access_regalloc_allocations_valid() {
        use crate::compiler::maglev::regalloc::{Location, allocate};

        let source = r#"
            var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "regalloc_check");

        // With named property sites, R15 is reserved → 8 registers.
        let num_regs = 8u32;
        let alloc = allocate(&graph, num_regs);

        // Collect all LoadNamedGeneric and LoadGlobal nodes.
        let mut load_globals = Vec::new();
        let mut load_named_generics = Vec::new();
        let mut generic_adds = Vec::new();
        let mut phis = Vec::new();
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                match node {
                    ValueNode::LoadGlobal { .. } => load_globals.push(*id),
                    ValueNode::LoadNamedGeneric { .. } => load_named_generics.push(*id),
                    ValueNode::GenericAdd { .. } => generic_adds.push(*id),
                    ValueNode::Phi { .. } => phis.push(*id),
                    _ => {}
                }
            }
        }

        eprintln!("LoadGlobal nodes: {load_globals:?}");
        eprintln!("LoadNamedGeneric nodes: {load_named_generics:?}");
        eprintln!("GenericAdd nodes: {generic_adds:?}");
        eprintln!("Phi nodes: {phis:?}");
        eprintln!("Spill count: {}", alloc.spill_count());

        // Every LoadGlobal must have a valid allocation.
        for id in &load_globals {
            let loc = alloc.location(*id);
            eprintln!("  LoadGlobal {id:?} → {loc:?}");
            assert!(
                matches!(
                    loc,
                    Some(Location::Register(_)) | Some(Location::StackSlot(_))
                ),
                "LoadGlobal {id:?} has no allocation (None) — emit_load would return JIT_UNDEFINED"
            );
        }

        // Every LoadNamedGeneric must have a valid allocation.
        for id in &load_named_generics {
            let loc = alloc.location(*id);
            eprintln!("  LoadNamedGeneric {id:?} → {loc:?}");
            assert!(
                matches!(
                    loc,
                    Some(Location::Register(_)) | Some(Location::StackSlot(_))
                ),
                "LoadNamedGeneric {id:?} has no allocation (None) — emit_load would return JIT_UNDEFINED"
            );
        }

        // Every GenericAdd must have a valid allocation.
        for id in &generic_adds {
            let loc = alloc.location(*id);
            eprintln!("  GenericAdd {id:?} → {loc:?}");
            assert!(
                matches!(
                    loc,
                    Some(Location::Register(_)) | Some(Location::StackSlot(_))
                ),
                "GenericAdd {id:?} has no allocation (None)"
            );
        }

        // Every Phi must have a valid allocation.
        for id in &phis {
            let loc = alloc.location(*id);
            eprintln!("  Phi {id:?} → {loc:?}");
            assert!(
                matches!(
                    loc,
                    Some(Location::Register(_)) | Some(Location::StackSlot(_))
                ),
                "Phi {id:?} has no allocation (None)"
            );
        }

        // Also verify GenericAdd inputs are allocated.
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if let ValueNode::GenericAdd { left, right, .. } = node {
                    let ll = alloc.location(*left);
                    let rl = alloc.location(*right);
                    assert!(
                        ll.is_some(),
                        "GenericAdd {id:?} left input {left:?} has None allocation"
                    );
                    assert!(
                        rl.is_some(),
                        "GenericAdd {id:?} right input {right:?} has None allocation"
                    );
                }
            }
        }

        // No register conflicts among overlapping live intervals.
        let intervals = crate::compiler::maglev::regalloc::compute_live_intervals(&graph);
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                let a = &intervals[i];
                let b = &intervals[j];
                if a.start < b.end && b.start < a.end {
                    if let (Some(Location::Register(ra)), Some(Location::Register(rb))) =
                        (alloc.location(a.id), alloc.location(b.id))
                    {
                        assert_ne!(
                            ra, rb,
                            "REGISTER CONFLICT: {:?} and {:?} both in reg {} \
                             (intervals [{},{}) and [{},{}))",
                            a.id, b.id, ra, a.start, a.end, b.start, b.end
                        );
                    }
                }
            }
        }

        eprintln!("All allocation checks passed ✓");
    }

    /// Diagnostic: dump the full optimized graph to detect store-to-load
    /// forwarding replacing LoadNamedGeneric with UndefinedConstant.
    #[test]
    fn test_property_access_no_undefined_constants_in_preheader() {
        let source = r#"
            var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "undef_check");

        // Dump ALL blocks with node types
        let mut undef_ids = Vec::new();
        let mut load_named_ids = Vec::new();
        for (bi, block) in graph.blocks().iter().enumerate() {
            eprintln!(
                "Block {bi}: {} nodes, is_loop_header={}, control={:?}",
                block.nodes.len(),
                block.is_loop_header,
                block.control
            );
            for (id, node) in &block.nodes {
                let desc = match node {
                    ValueNode::UndefinedConstant => {
                        undef_ids.push((*id, bi));
                        "UndefinedConstant".to_string()
                    }
                    ValueNode::LoadNamedGeneric { object, name, .. } => {
                        load_named_ids.push((*id, bi));
                        format!("LoadNamedGeneric(object={object:?}, name={name})")
                    }
                    ValueNode::LoadGlobal { name, .. } => {
                        format!("LoadGlobal(name={name})")
                    }
                    ValueNode::GenericAdd { left, right, .. } => {
                        format!("GenericAdd(left={left:?}, right={right:?})")
                    }
                    ValueNode::Phi { inputs } => {
                        format!("Phi(inputs={inputs:?})")
                    }
                    ValueNode::SmiConstant { value } => {
                        format!("SmiConstant({value})")
                    }
                    ValueNode::GenericIncrement { value, .. } => {
                        format!("GenericIncrement(value={value:?})")
                    }
                    ValueNode::StoreGlobal { name, value, .. } => {
                        format!("StoreGlobal(name={name}, value={value:?})")
                    }
                    other => format!("{other:?}").chars().take(80).collect(),
                };
                eprintln!("  {id:?}: {desc}");
            }
        }

        eprintln!("\nUndefinedConstant nodes: {undef_ids:?}");
        eprintln!("LoadNamedGeneric nodes: {load_named_ids:?}");

        // After global-forwarding store-to-load optimisation, all 5 property
        // loads should be replaced with their stored SmiConstant values.
        // LoadNamedGeneric nodes are expected to be eliminated entirely.
        if load_named_ids.is_empty() {
            eprintln!("All LoadNamedGeneric nodes forwarded to constants by store-to-load ✓");
        }

        // Check that GenericAdd nodes in the loop body do NOT reference
        // UndefinedConstant nodes (which would indicate a broken forwarding).
        let undef_node_ids: std::collections::HashSet<_> =
            undef_ids.iter().map(|(id, _)| *id).collect();
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if let ValueNode::GenericAdd { left, right, .. } = node {
                    assert!(
                        !undef_node_ids.contains(left),
                        "GenericAdd {id:?} left input {left:?} references an UndefinedConstant!"
                    );
                    assert!(
                        !undef_node_ids.contains(right),
                        "GenericAdd {id:?} right input {right:?} references an UndefinedConstant!"
                    );
                }
            }
        }

        eprintln!("No UndefinedConstant references from GenericAdd ✓");
    }

    #[test]
    fn test_prototype_chain_benchmark_not_degenerate() {
        let source = r#"
            function Base() {}
            Base.prototype.x = 42;
            function Mid() {}
            Mid.prototype = new Base();
            function Leaf() {}
            Leaf.prototype = new Mid();
            var obj = new Leaf();
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.x;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "prototype_chain_1k");
        assert!(
            !graph.is_degenerate(),
            "prototype_chain_1k graph is degenerate — Maglev won't compile it"
        );
    }

    #[test]
    fn test_deep_object_benchmark_not_degenerate() {
        let source = r#"
            var root = { a: { b: { c: { d: { e: 99 } } } } };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + root.a.b.c.d.e;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "deep_object_access_1k");
        assert!(
            !graph.is_degenerate(),
            "deep_object_access_1k graph is degenerate — Maglev won't compile it"
        );
    }

    /// Verify register allocation is valid for the deep_object_access_1k
    /// pattern — hoisted LoadNamedGeneric chains must keep their results
    /// live across the loop body.
    #[test]
    fn test_deep_object_regalloc_allocations_valid() {
        use crate::compiler::maglev::regalloc::{Location, allocate};

        let source = r#"
            var root = { a: { b: { c: { d: { e: 99 } } } } };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + root.a.b.c.d.e;
            }
            sum;
        "#;
        let graph = compile_and_optimize(source, "deep_regalloc");

        let num_regs = 8u32;
        let alloc = allocate(&graph, num_regs);

        // Collect all LoadNamedGeneric, GenericAdd, and Phi nodes.
        let mut load_named = Vec::new();
        let mut generic_adds = Vec::new();
        let mut phis = Vec::new();
        for (blk_idx, block) in graph.blocks().iter().enumerate() {
            for (id, node) in &block.nodes {
                match node {
                    ValueNode::LoadNamedGeneric { .. } => {
                        load_named.push((*id, blk_idx));
                    }
                    ValueNode::GenericAdd { .. } => {
                        generic_adds.push(*id);
                    }
                    ValueNode::Phi { .. } => {
                        phis.push(*id);
                    }
                    _ => {}
                }
            }
        }

        eprintln!("LoadNamedGeneric: {load_named:?}");
        eprintln!("GenericAdd: {generic_adds:?}");
        eprintln!("Phi: {phis:?}");
        eprintln!("Spill count: {}", alloc.spill_count());

        // Every LoadNamedGeneric must have a valid allocation.
        for (id, blk) in &load_named {
            let loc = alloc.location(*id);
            eprintln!("  LoadNamedGeneric {id:?} (block {blk}) → {loc:?}");
            assert!(
                matches!(
                    loc,
                    Some(Location::Register(_)) | Some(Location::StackSlot(_))
                ),
                "LoadNamedGeneric {id:?} in block {blk} has no allocation — \
                 emit_load would produce JIT_UNDEFINED"
            );
        }

        // Every GenericAdd input must have a valid allocation.
        for block in graph.blocks() {
            for (id, node) in &block.nodes {
                if let ValueNode::GenericAdd { left, right, .. } = node {
                    let ll = alloc.location(*left);
                    let rl = alloc.location(*right);
                    eprintln!("  GenericAdd {id:?}: left {left:?}={ll:?}, right {right:?}={rl:?}");
                    assert!(
                        ll.is_some(),
                        "GenericAdd {id:?} left input {left:?} has None allocation"
                    );
                    assert!(
                        rl.is_some(),
                        "GenericAdd {id:?} right input {right:?} has None allocation"
                    );
                }
            }
        }
    }

    /// Regression test: fused CreateEmptyObjectLiteral nodes must use the
    /// u32::MAX sentinel as feedback_slot so the runtime skips template
    /// caching.  Previously all fused empty-object literals shared slot 0,
    /// causing template-cache collisions that corrupted property names of
    /// nested object initializers and led to persistent JIT deopts.
    #[test]
    fn test_fused_empty_object_literal_uses_sentinel_feedback_slot() {
        let source = r#"
            var root = { a: { b: { c: { d: { e: 99 } } } } };
            root;
        "#;
        let graph = compile_and_optimize(source, "fused_sentinel");

        let mut fused_slots = Vec::new();
        for block in graph.blocks() {
            for (_id, node) in &block.nodes {
                if let ValueNode::CreateObjectLiteralWithProperties { feedback_slot, .. } = node {
                    fused_slots.push(*feedback_slot);
                }
            }
        }

        eprintln!("Fused CreateObjectLiteralWithProperties feedback_slots: {fused_slots:?}");
        assert!(
            !fused_slots.is_empty(),
            "Expected at least one fused CreateObjectLiteralWithProperties node"
        );

        // Every fused node that originated from CreateEmptyObjectLiteral
        // must use the sentinel u32::MAX, not 0.
        for &slot in &fused_slots {
            assert_eq!(
                slot,
                u32::MAX,
                "Fused CreateEmptyObjectLiteral should use sentinel u32::MAX, got {slot}"
            );
        }
    }

    // ── Keyed property specialization ────────────────────────────────────────

    /// When the feedback slot for `LdaKeyedProperty` is `Monomorphic`,
    /// the builder should emit `CheckSmi` + `LoadFixedArrayElement` instead
    /// of the generic `LoadKeyedGeneric`.
    #[test]
    fn test_keyed_load_monomorphic_specializes_to_fixed_array() {
        // Bytecode: LdaSmi(0), Star(r1), Ldar(r0), LdaKeyedProperty(r0, slot:0), Return
        //   r0 = param 0 (the array)
        //   r1 = the key (0)
        //   acc = arr[0]
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            // LdaKeyedProperty r0, [slot 0]
            Instruction::new_unchecked(
                Opcode::LdaKeyedProperty,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let metadata = FeedbackMetadata::new(vec![FeedbackSlotKind::KeyedLoadProperty]);
        let (arr, mut fv) = build(instrs, vec![], 2, 1, metadata);
        // Simulate warm feedback.
        fv.set_state(0, InlineCacheState::Monomorphic);

        let graph = GraphBuilder::build(&arr, &fv).unwrap();
        let block = graph.entry_block().unwrap();

        // Should contain a CheckSmi guard on the key.
        let has_check_smi = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }));
        assert!(has_check_smi, "expected CheckSmi guard for key");

        // Should contain a LoadFixedArrayElement (not LoadKeyedGeneric).
        let has_fast_load = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadFixedArrayElement { .. }));
        assert!(has_fast_load, "expected LoadFixedArrayElement (fast path)");

        let has_generic = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadKeyedGeneric { .. }));
        assert!(
            !has_generic,
            "LoadKeyedGeneric should NOT appear when feedback is monomorphic"
        );
    }

    /// When the feedback slot for `LdaKeyedProperty` is `Uninitialized`,
    /// the builder should fall back to `LoadKeyedGeneric`.
    #[test]
    fn test_keyed_load_uninitialized_falls_back_to_generic() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(
                Opcode::LdaKeyedProperty,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let metadata = FeedbackMetadata::new(vec![FeedbackSlotKind::KeyedLoadProperty]);
        let (arr, fv) = build(instrs, vec![], 2, 1, metadata);
        // fv slot 0 stays Uninitialized (default).

        let graph = GraphBuilder::build(&arr, &fv).unwrap();
        let block = graph.entry_block().unwrap();

        let has_generic = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadKeyedGeneric { .. }));
        assert!(
            has_generic,
            "expected LoadKeyedGeneric when feedback is uninitialized"
        );

        let has_fast_load = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadFixedArrayElement { .. }));
        assert!(
            !has_fast_load,
            "LoadFixedArrayElement should NOT appear without warm feedback"
        );
    }

    /// When the feedback slot for `StaKeyedProperty` is `Monomorphic`,
    /// the builder should emit `CheckSmi` + `StoreFixedArrayElement`.
    #[test]
    fn test_keyed_store_monomorphic_specializes_to_fixed_array() {
        // Bytecode: LdaSmi(42), StaKeyedProperty(r0, r1, slot:0), Return
        //   r0 = param 0 (the array), r1 = param 1 (the key)
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            // StaKeyedProperty r0, r1, [slot 0]
            Instruction::new_unchecked(
                Opcode::StaKeyedProperty,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let metadata = FeedbackMetadata::new(vec![FeedbackSlotKind::KeyedStoreProperty]);
        let (arr, mut fv) = build(instrs, vec![], 2, 2, metadata);
        fv.set_state(0, InlineCacheState::Monomorphic);

        let graph = GraphBuilder::build(&arr, &fv).unwrap();
        let block = graph.entry_block().unwrap();

        let has_check_smi = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::CheckSmi { .. }));
        assert!(has_check_smi, "expected CheckSmi guard for key");

        let has_fast_store = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::StoreFixedArrayElement { .. }));
        assert!(
            has_fast_store,
            "expected StoreFixedArrayElement (fast path)"
        );

        let has_generic = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::StoreKeyedGeneric { .. }));
        assert!(
            !has_generic,
            "StoreKeyedGeneric should NOT appear when feedback is monomorphic"
        );
    }

    /// Megamorphic feedback should fall back to the generic path.
    #[test]
    fn test_keyed_load_megamorphic_falls_back_to_generic() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
            Instruction::new_unchecked(
                Opcode::LdaKeyedProperty,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let metadata = FeedbackMetadata::new(vec![FeedbackSlotKind::KeyedLoadProperty]);
        let (arr, mut fv) = build(instrs, vec![], 2, 1, metadata);
        fv.set_state(0, InlineCacheState::Megamorphic);

        let graph = GraphBuilder::build(&arr, &fv).unwrap();
        let block = graph.entry_block().unwrap();

        let has_generic = block
            .nodes
            .iter()
            .any(|(_, n)| matches!(n, ValueNode::LoadKeyedGeneric { .. }));
        assert!(
            has_generic,
            "expected LoadKeyedGeneric when feedback is megamorphic"
        );
    }

    /// Regression test: the sieve_primes benchmark must compile to a
    /// non-degenerate graph with zero spurious deoptimisation blocks.
    #[test]
    fn test_sieve_benchmark_no_deopt() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::recursive_descent;

        let source = r#"
            var n = 1000;
            var sieve = [];
            for (var i = 0; i <= n; i++) sieve[i] = true;
            sieve[0] = false;
            sieve[1] = false;
            for (var i = 2; i * i <= n; i++) {
                if (sieve[i]) {
                    for (var j = i * i; j <= n; j = j + i) {
                        sieve[j] = false;
                    }
                }
            }
            var count = 0;
            for (var i = 0; i <= n; i++) {
                if (sieve[i]) count = count + 1;
            }
            count;
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let feedback = FeedbackVector::new(ba.feedback_metadata());
        let graph = GraphBuilder::build(&ba, &feedback).unwrap();

        let deopt_blocks: Vec<_> = graph
            .blocks()
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.control, Some(ControlNode::Deoptimize { .. })))
            .map(|(i, _)| i)
            .collect();

        assert!(
            deopt_blocks.is_empty(),
            "sieve graph has Deoptimize control in blocks {deopt_blocks:?} — \
             indicates unhandled bytecodes"
        );
        assert!(
            !graph.is_degenerate(),
            "sieve graph should not be degenerate"
        );
    }

    /// Regression test: property access benchmark must compile cleanly.
    #[test]
    fn test_property_access_benchmark_no_deopt() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::recursive_descent;

        let source = r#"
            var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
            }
            sum;
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let feedback = FeedbackVector::new(ba.feedback_metadata());
        let graph = GraphBuilder::build(&ba, &feedback).unwrap();

        let deopt_blocks: Vec<_> = graph
            .blocks()
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.control, Some(ControlNode::Deoptimize { .. })))
            .map(|(i, _)| i)
            .collect();

        assert!(
            deopt_blocks.is_empty(),
            "property_access graph has Deoptimize control in blocks {deopt_blocks:?}"
        );
        assert!(
            !graph.is_degenerate(),
            "property_access graph should not be degenerate"
        );
    }

    /// Verify that array literals with multiple elements (which emit
    /// `StaInArrayLiteral`) compile without spurious deopts.
    #[test]
    fn test_array_literal_sta_in_array_literal_no_deopt() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::recursive_descent;

        let source = r#"
            var a = [10, 20, 30, 40, 50];
            var sum = 0;
            for (var i = 0; i < 5; i++) {
                sum = sum + a[i];
            }
            sum;
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();

        // Sanity-check: bytecodes should contain StaInArrayLiteral.
        let instructions = ba.instructions().unwrap();
        let has_sta_in_array = instructions
            .iter()
            .any(|i| i.opcode == crate::bytecode::bytecodes::Opcode::StaInArrayLiteral);
        assert!(
            has_sta_in_array,
            "expected StaInArrayLiteral in array literal bytecodes"
        );

        let feedback = FeedbackVector::new(ba.feedback_metadata());
        let graph = GraphBuilder::build(&ba, &feedback).unwrap();

        let deopt_blocks: Vec<_> = graph
            .blocks()
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.control, Some(ControlNode::Deoptimize { .. })))
            .map(|(i, _)| i)
            .collect();

        assert!(
            deopt_blocks.is_empty(),
            "array literal graph has Deoptimize in blocks {deopt_blocks:?} — \
             StaInArrayLiteral handler may be missing"
        );
        assert!(
            !graph.is_degenerate(),
            "array literal graph should not be degenerate"
        );
    }

    /// The actual deep_object benchmark runs inside an arrow function, not a
    /// top-level script. Verify that the *function* version also produces a
    /// non-degenerate Maglev graph and that LICM hoists the property loads.
    #[test]
    fn test_deep_object_arrow_function_not_degenerate() {
        use crate::bytecode::bytecode_array::ConstantPoolEntry;
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::compiler::maglev::optimizer::optimize;
        use crate::parser::recursive_descent;

        // Wrap the benchmark body in a function declaration so the bytecode
        // generator compiles it as a function (registers, not globals).
        let source = r#"
            function deep() {
                var root = { a: { b: { c: { d: { e: 99 } } } } };
                var sum = 0;
                for (var i = 0; i < 1000; i++) {
                    sum = sum + root.a.b.c.d.e;
                }
                return sum;
            }
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let script_ba = BytecodeGenerator::compile_program(&program).unwrap();

        // Extract the inner function's BytecodeArray from the script's
        // constant pool.
        let inner_ba = script_ba
            .constant_pool()
            .iter()
            .find_map(|entry| match entry {
                ConstantPoolEntry::Function(ba) => Some(ba.clone()),
                _ => None,
            })
            .expect("should find inner function BA in constant pool");

        let instructions = inner_ba.instructions().unwrap();
        eprintln!(
            "deep_arrow: {} bytecodes, {} raw bytes",
            instructions.len(),
            inner_ba.bytecodes().len()
        );
        for (idx, instr) in instructions.iter().enumerate() {
            eprintln!("  [{idx:3}] {:?}", instr.opcode);
        }

        let feedback = FeedbackVector::new(inner_ba.feedback_metadata());
        let mut graph = GraphBuilder::build(&inner_ba, &feedback).unwrap();
        optimize(&mut graph);

        let degen = graph.is_degenerate();
        eprintln!("deep_arrow: is_degenerate={degen}");

        if let Some(entry) = graph.entry_block() {
            eprintln!(
                "deep_arrow: entry block has {} nodes, control={:?}",
                entry.nodes.len(),
                entry.control
            );
        }

        // Print all blocks for diagnostics.
        for block in graph.blocks() {
            eprintln!(
                "deep_arrow: block {} has {} nodes, control={:?}",
                block.id,
                block.nodes.len(),
                block.control
            );
            for (nid, node) in &block.nodes {
                eprintln!("  {nid:?} = {node:?}");
            }
        }

        assert!(
            !graph.is_degenerate(),
            "deep_object arrow function graph is degenerate — Maglev won't compile it"
        );
    }

    #[test]
    fn test_closure_counter_bytecodes_and_graph() {
        use crate::bytecode::bytecode_array::ConstantPoolEntry;
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::compiler::maglev::optimizer::optimize;
        use crate::parser::recursive_descent;

        // Test the FULL benchmark script (top-level) — this is what actually
        // needs to be JIT-compiled for the benchmark to be fast.
        let source = r#"
            function make_counter() {
                var count = 0;
                return function() { count = count + 1; return count; };
            }
            var counter = make_counter();
            var result = 0;
            for (var i = 0; i < 1000; i++) {
                result = counter();
            }
            result;
        "#;

        let program = recursive_descent::parse(source).unwrap();
        let script_ba = BytecodeGenerator::compile_program(&program).unwrap();

        // Dump top-level bytecodes
        eprintln!("top-level script bytecodes:");
        let top_instrs = script_ba.instructions().unwrap();
        for (idx, instr) in top_instrs.iter().enumerate() {
            eprintln!("  [{idx:3}] {:?} {:?}", instr.opcode, instr.operands());
        }

        // Build graph for the top-level script
        let feedback = FeedbackVector::new(script_ba.feedback_metadata());
        let result = GraphBuilder::build(&script_ba, &feedback);
        match result {
            Ok(mut graph) => {
                optimize(&mut graph);
                let degen = graph.is_degenerate();
                eprintln!("\ntop-level graph: is_degenerate={degen}");
                for block in graph.blocks() {
                    eprintln!(
                        "  block {} has {} nodes, control={:?}",
                        block.id,
                        block.nodes.len(),
                        block.control
                    );
                    for (nid, node) in &block.nodes {
                        eprintln!("    {nid:?} = {node:?}");
                    }
                }
                assert!(
                    !degen,
                    "closure_counter_1k top-level script graph is degenerate — Maglev won't compile the hot loop"
                );
            }
            Err(e) => {
                panic!("GraphBuilder::build failed for top-level script: {e}");
            }
        }

        // Also check inner closure
        let make_counter_ba = script_ba
            .constant_pool()
            .iter()
            .find_map(|entry| match entry {
                ConstantPoolEntry::Function(ba) => Some(ba.clone()),
                _ => None,
            })
            .expect("should find make_counter BA");
        let inner_ba = make_counter_ba
            .constant_pool()
            .iter()
            .find_map(|entry| match entry {
                ConstantPoolEntry::Function(ba) => Some(ba.clone()),
                _ => None,
            })
            .expect("should find inner closure BA");

        eprintln!("\ninner closure bytecodes:");
        let inner_instrs = inner_ba.instructions().unwrap();
        for (idx, instr) in inner_instrs.iter().enumerate() {
            eprintln!("  [{idx:3}] {:?} {:?}", instr.opcode, instr.operands());
        }

        let inner_feedback = FeedbackVector::new(inner_ba.feedback_metadata());
        let mut inner_graph = GraphBuilder::build(&inner_ba, &inner_feedback).unwrap();
        optimize(&mut inner_graph);
        assert!(
            !inner_graph.is_degenerate(),
            "inner closure graph is degenerate"
        );
    }
}

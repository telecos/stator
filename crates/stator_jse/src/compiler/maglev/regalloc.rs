//! Maglev linear-scan register allocator.
//!
//! # Overview
//!
//! This module implements a linear-scan register allocator over a
//! [`MaglevGraph`].  The allocator:
//!
//! 1. **Linearises** the graph — blocks are visited in index order; within
//!    each block value nodes are visited in their stored order, followed by the
//!    block's control node.  Each position receives a unique *program point*
//!    (a monotonically increasing `u32`).
//!
//! 2. **Computes live ranges** — for every [`NodeId`] the allocator records
//!    the program point at which the value is *defined* (the node that
//!    produces it) and the *last* program point at which it is *used* as an
//!    input.
//!
//! 3. **Runs linear scan** — live intervals are sorted by their start point.
//!    The allocator greedily assigns one of `num_regs` physical registers.
//!    When all registers are occupied it *spills* the live interval with the
//!    farthest endpoint, freeing a register for the new interval (Poletto &
//!    Sarkar, 1999).
//!
//! # Output
//!
//! The allocator produces an [`AllocationResult`] that maps each [`NodeId`]
//! to a [`Location`]:
//!
//! - [`Location::Register(u32)`] — the value lives in physical register `n`.
//! - [`Location::StackSlot(u32)`] — the value has been spilled to stack slot
//!   `n`.
//!
//! # Example
//!
//! ```
//! use stator_jse::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_jse::compiler::maglev::regalloc::{allocate, Location};
//!
//! let mut graph = MaglevGraph::new(1);
//! let mut block = BasicBlock::new(0);
//! let p0 = block.push_value(ValueNode::Parameter { index: 0 });
//! block.set_control(ControlNode::Return { value: p0 });
//! graph.add_block(block);
//!
//! let result = allocate(&graph, 8);
//! // With 8 registers and a single value, no spill occurs.
//! assert!(matches!(result.location(p0), Some(Location::Register(_))));
//! assert_eq!(result.spill_count(), 0);
//! ```

use std::collections::{HashMap, HashSet};

use crate::compiler::maglev::ir::{ControlNode, MaglevGraph, NodeId, ValueNode};
use crate::compiler::maglev::licm::detect_loops;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The physical location assigned to a [`NodeId`] by the register allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// The value lives in physical register `n`.
    Register(u32),
    /// The value has been spilled to stack slot `n`.
    StackSlot(u32),
}

/// The result of register allocation for a [`MaglevGraph`].
///
/// Obtained by calling [`allocate`].
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Map from [`NodeId`] to the assigned [`Location`].
    assignments: HashMap<NodeId, Location>,
    /// Total number of stack slots allocated (i.e. the required stack-frame
    /// size in slots).
    spill_count: u32,
    /// Per-node bitmask of caller-saved allocatable registers (indices 1–5:
    /// RCX, RDX, RSI, R8, R9) that hold live values **across** that node.
    /// A register is included only when its value is still needed *after* the
    /// node, so stub-call arguments consumed at the call site are excluded.
    live_caller_saved: HashMap<NodeId, u8>,
}

impl AllocationResult {
    /// Look up the [`Location`] assigned to `id`.  Returns `None` for nodes
    /// that produce no value (this should not happen for a well-formed graph).
    pub fn location(&self, id: NodeId) -> Option<Location> {
        self.assignments.get(&id).copied()
    }

    /// The number of stack slots required by the function's frame.
    pub fn spill_count(&self) -> u32 {
        self.spill_count
    }

    /// Returns a bitmask of caller-saved allocatable registers (bits 1–5)
    /// that hold values live *across* the given node.  Returns `0` when no
    /// caller-saved register needs saving at that point.
    pub fn live_caller_saved_at(&self, id: NodeId) -> u8 {
        self.live_caller_saved.get(&id).copied().unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live-range analysis
// ─────────────────────────────────────────────────────────────────────────────

/// A half-open live interval `[start, end)` for a single [`NodeId`].
///
/// `start` is the program point at which the value is *defined*; `end` is one
/// past the last program point at which the value is *used* as an input.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LiveInterval {
    pub(crate) id: NodeId,
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// For loop-header Phi values whose intervals were extended to cover
    /// the full loop body: the end point *before* the extension.
    /// The coalescing pass uses this to allow Phi/back-edge-input
    /// coalescing even though the Phi's `end` was artificially enlarged.
    pre_ext_end: Option<u32>,
}

/// Compute a live interval for every value-producing node in `graph`.
///
/// Program points are assigned by visiting blocks in index order; within each
/// block value nodes are visited in their stored order and the control node
/// occupies the next program point after the last value node.
pub(crate) fn compute_live_intervals(graph: &MaglevGraph) -> Vec<LiveInterval> {
    // Pass 1: assign a definition program point to every node.
    let mut def_point: HashMap<NodeId, u32> = HashMap::new();
    let mut pp: u32 = 0;
    for block in graph.blocks() {
        for &(id, _) in &block.nodes {
            def_point.insert(id, pp);
            pp += 1;
        }
        if block.control.is_some() {
            pp += 1; // control node occupies one slot
        }
    }

    // Pass 2: for each use of a NodeId record the program point and track the
    // farthest (last) use.
    let mut end_point: HashMap<NodeId, u32> = HashMap::new();

    let mut record_use = |id: NodeId, at: u32| {
        let e = end_point.entry(id).or_insert(at);
        if at > *e {
            *e = at;
        }
    };

    let mut pp: u32 = 0;
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            let node_pp = pp;
            collect_inputs(node, &mut |inp| record_use(inp, node_pp));
            pp += 1;
        }
        if let Some(ctrl) = &block.control {
            let ctrl_pp = pp;
            collect_control_inputs(ctrl, &mut |inp| record_use(inp, ctrl_pp));
            pp += 1;
        }
    }

    // Compute per-block program-point ranges: (first_pp, one_past_last_pp).
    let mut block_pp_ranges: Vec<(u32, u32)> = Vec::new();
    {
        let mut bp: u32 = 0;
        for block in graph.blocks() {
            let first = bp;
            bp += block.nodes.len() as u32;
            if block.control.is_some() {
                bp += 1;
            }
            block_pp_ranges.push((first, bp));
        }
    }

    // ── Natural loop detection ────────────────────────────────────────
    //
    // Reuse the LICM module's `detect_loops` to correctly identify all
    // natural loops and their full body block sets.  This handles nested
    // loops, reversed block ordering (body before header), and
    // multi-back-edge loops — all of which the previous ad-hoc
    // predecessor-skip heuristic missed.
    let loops = detect_loops(graph);

    // ── Pass 2.5: Fix Phi intervals for reversed-order loop headers ──
    //
    // When the block ordering puts a loop body *before* its header,
    // Phi definitions at the header have a higher program point than
    // their uses in the body.  Extend Phi def_points backward to the
    // earliest body block's start pp.
    for lp in &loops {
        let header_pp = block_pp_ranges[lp.header as usize].0;
        let earliest_body_pp = lp
            .body
            .iter()
            .filter(|&&b| b != lp.header)
            .filter_map(|&b| block_pp_ranges.get(b as usize).map(|&(s, _)| s))
            .min();

        if let Some(min_pp) = earliest_body_pp
            && min_pp < header_pp
            && let Some(header_block) = graph.block(lp.header)
        {
            for (phi_id, node) in &header_block.nodes {
                if let ValueNode::Phi { .. } = node
                    && let Some(def) = def_point.get_mut(phi_id)
                    && min_pp < *def
                {
                    *def = min_pp;
                }
            }
        }
    }

    // ── Pass 3: Extend intervals across loop back-edges ──────────────
    //
    // For each natural loop, extend any interval whose last use falls
    // inside the loop body so it covers the entire loop.  This prevents
    // the allocator from freeing a register mid-loop when the value is
    // still needed on the next iteration.
    //
    // For Phi values defined at loop headers we record the pre-extension
    // end so the coalescing pass can still coalesce the Phi with its
    // back-edge input (the extension is for register preservation, not a
    // genuine use).
    //
    // Iterate to a fixed point so that nested loop extensions propagate
    // outward: an inner loop extension may push an interval's end into
    // the range of an enclosing outer loop.
    let mut header_phi_ids: HashSet<NodeId> = HashSet::new();
    for lp in &loops {
        if let Some(header_block) = graph.block(lp.header) {
            for (phi_id, node) in &header_block.nodes {
                if let ValueNode::Phi { .. } = node {
                    header_phi_ids.insert(*phi_id);
                }
            }
        }
    }
    let mut pre_ext_ends: HashMap<NodeId, u32> = HashMap::new();
    loop {
        let mut changed = false;
        for lp in &loops {
            let loop_start_pp = lp
                .body
                .iter()
                .filter_map(|&b| block_pp_ranges.get(b as usize).map(|&(s, _)| s))
                .min()
                .unwrap_or(0);
            let loop_end_pp = lp
                .body
                .iter()
                .filter_map(|&b| block_pp_ranges.get(b as usize).map(|&(_, e)| e))
                .max()
                .unwrap_or(0);

            if loop_start_pp >= loop_end_pp {
                continue;
            }

            let target_end = loop_end_pp.saturating_sub(1);

            // Collect Phi inputs from the preheader that are NOT used
            // elsewhere in the loop body.  These "entry-only" values are
            // consumed once on loop entry and don't need their intervals
            // extended across the full loop.
            let header_first_pp = block_pp_ranges[lp.header as usize].0;
            let mut phi_entry_only: HashSet<NodeId> = HashSet::new();
            if let Some(header_block) = graph.block(lp.header) {
                let preheader_pos = header_block
                    .predecessors
                    .iter()
                    .position(|&p| p == lp.preheader);
                if let Some(pos) = preheader_pos {
                    for (_, node) in &header_block.nodes {
                        if let ValueNode::Phi { inputs, .. } = node
                            && let Some(&entry_id) = inputs.get(pos)
                            && end_point
                                .get(&entry_id)
                                .is_none_or(|&e| e <= header_first_pp)
                        {
                            phi_entry_only.insert(entry_id);
                        }
                    }
                }
            }

            for (id, end) in end_point.iter_mut() {
                if phi_entry_only.contains(id) {
                    continue;
                }
                if *end >= loop_start_pp
                    && *end < loop_end_pp
                    && let Some(&start) = def_point.get(id)
                    && start < loop_end_pp
                    && *end < target_end
                {
                    // Record original end for header Phis before extending.
                    if header_phi_ids.contains(id) && !pre_ext_ends.contains_key(id) {
                        pre_ext_ends.insert(*id, *end);
                    }
                    *end = target_end;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Build the interval list from the definition map.
    def_point
        .into_iter()
        .map(|(id, start)| {
            // end is one past the last use; if never used, [start, start+1).
            let end = end_point.get(&id).map_or(start + 1, |&e| e + 1);
            let pre_ext_end = pre_ext_ends.get(&id).map(|&e| e + 1);
            LiveInterval {
                id,
                start,
                end,
                pre_ext_end,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Input-collection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Call `f` for each [`NodeId`] operand of `node`.
#[allow(clippy::too_many_lines)]
fn collect_inputs(node: &ValueNode, f: &mut impl FnMut(NodeId)) {
    match node {
        // ── Nodes with no inputs ─────────────────────────────────────────────
        ValueNode::SmiConstant { .. }
        | ValueNode::Float64Constant { .. }
        | ValueNode::Int32Constant { .. }
        | ValueNode::Uint32Constant { .. }
        | ValueNode::BigIntConstant { .. }
        | ValueNode::TrueConstant
        | ValueNode::FalseConstant
        | ValueNode::NullConstant
        | ValueNode::UndefinedConstant
        | ValueNode::RootConstant { .. }
        | ValueNode::ExternalConstant { .. }
        | ValueNode::StringConstant { .. }
        | ValueNode::ConstantPoolEntry { .. }
        | ValueNode::Parameter { .. }
        | ValueNode::RegisterInput { .. }
        | ValueNode::ArgumentsLength
        | ValueNode::RestLength
        | ValueNode::LoadGlobal { .. }
        | ValueNode::LoadCurrentContextSlot { .. }
        | ValueNode::ArgumentsElements { .. }
        | ValueNode::RestElements { .. }
        | ValueNode::VirtualObject { .. }
        | ValueNode::CreateFunctionContext { .. }
        | ValueNode::CreateBlockContext { .. }
        | ValueNode::CreateShallowObjectLiteral { .. }
        | ValueNode::CreateShallowArrayLiteral { .. }
        | ValueNode::CreateObjectLiteral { .. }
        | ValueNode::CreateArrayLiteral { .. }
        | ValueNode::CreateEmptyObjectLiteral
        | ValueNode::CreateEmptyArrayLiteral { .. }
        | ValueNode::CreateMappedArguments
        | ValueNode::CreateUnmappedArguments
        | ValueNode::CreateRestParameter
        | ValueNode::CreateRegExpLiteral { .. }
        | ValueNode::CreateClosure { .. }
        | ValueNode::FastCreateClosure { .. }
        | ValueNode::GetTemplateObject { .. }
        | ValueNode::Debugger
        | ValueNode::Abort { .. } => {}

        // ── Single-input nodes ────────────────────────────────────────────────
        ValueNode::GetArgument { index } => f(*index),

        ValueNode::CheckedSmiIncrement { value }
        | ValueNode::CheckedSmiDecrement { value }
        | ValueNode::Int32Negate { value }
        | ValueNode::Int32Increment { value }
        | ValueNode::Int32Decrement { value }
        | ValueNode::Float64Negate { value }
        | ValueNode::Float64Ieee754Unary { value, .. }
        | ValueNode::GenericBitwiseNot { value, .. }
        | ValueNode::GenericNegate { value, .. }
        | ValueNode::GenericIncrement { value, .. }
        | ValueNode::GenericDecrement { value, .. }
        | ValueNode::ChangeInt32ToFloat64 { input: value }
        | ValueNode::ChangeUint32ToFloat64 { input: value }
        | ValueNode::ChangeFloat64ToInt32 { input: value }
        | ValueNode::CheckedFloat64ToInt32 { input: value }
        | ValueNode::ChangeInt32ToTagged { input: value }
        | ValueNode::ChangeUint32ToTagged { input: value }
        | ValueNode::ChangeFloat64ToTagged { input: value }
        | ValueNode::ChangeTaggedToInt32 { input: value }
        | ValueNode::ChangeTaggedToUint32 { input: value }
        | ValueNode::ChangeTaggedToFloat64 { input: value }
        | ValueNode::CheckedTaggedToInt32 { input: value }
        | ValueNode::CheckedTaggedToFloat64 { input: value }
        | ValueNode::ToBoolean { value }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::ToName { value, .. }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToNumberOrNumeric { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::NumberToString { value, .. }
        | ValueNode::TestUndetectable { value }
        | ValueNode::TestNullOrUndefined { value }
        | ValueNode::TestTypeOf { value, .. } => f(*value),

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => f(*receiver),

        ValueNode::CheckDynamicValue { receiver, expected } => {
            f(*receiver);
            f(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => f(*input),

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::ForInPrepare {
            enumerator: object, ..
        }
        | ValueNode::StringLength { string: object } => f(*object),

        ValueNode::LoadEnumCacheLength { map } => f(*map),

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            f(*object);
            f(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            f(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => f(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            f(*context);
            f(*value);
        }

        ValueNode::PushContext { context } | ValueNode::PopContext { context } => f(*context),

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            f(*elements);
            f(*index);
        }

        ValueNode::StoreFixedArrayElement {
            elements,
            index,
            value,
        }
        | ValueNode::StoreFixedDoubleArrayElement {
            elements,
            index,
            value,
        } => {
            f(*elements);
            f(*index);
            f(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for &v in values {
                f(v);
            }
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            f(*object);
            f(*key);
            f(*value);
        }

        // ── Binary nodes ──────────────────────────────────────────────────────
        ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::CheckedSmiDivide { left, right }
        | ValueNode::CheckedSmiModulus { left, right }
        | ValueNode::Int32Add { left, right }
        | ValueNode::Int32Subtract { left, right }
        | ValueNode::Int32Multiply { left, right }
        | ValueNode::Int32Divide { left, right }
        | ValueNode::Int32Modulus { left, right }
        | ValueNode::Int32BitwiseAnd { left, right }
        | ValueNode::Int32BitwiseOr { left, right }
        | ValueNode::Int32BitwiseXor { left, right }
        | ValueNode::Int32ShiftLeft { left, right }
        | ValueNode::Int32ShiftRight { left, right }
        | ValueNode::Int32ShiftRightLogical { left, right }
        | ValueNode::Uint32Add { left, right }
        | ValueNode::Uint32Subtract { left, right }
        | ValueNode::Uint32Multiply { left, right }
        | ValueNode::Uint32Divide { left, right }
        | ValueNode::Uint32Modulus { left, right }
        | ValueNode::Float64Add { left, right }
        | ValueNode::Float64Subtract { left, right }
        | ValueNode::Float64Multiply { left, right }
        | ValueNode::Float64Divide { left, right }
        | ValueNode::Float64Modulus { left, right }
        | ValueNode::Float64Exponentiate { left, right }
        | ValueNode::Int32Equal { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32LessThanOrEqual { left, right }
        | ValueNode::Int32GreaterThan { left, right }
        | ValueNode::Int32GreaterThanOrEqual { left, right }
        | ValueNode::Float64Equal { left, right }
        | ValueNode::Float64LessThan { left, right }
        | ValueNode::Float64LessThanOrEqual { left, right }
        | ValueNode::Float64GreaterThan { left, right }
        | ValueNode::Float64GreaterThanOrEqual { left, right }
        | ValueNode::StringConcat { left, right }
        | ValueNode::StringEqual { left, right }
        | ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. }
        | ValueNode::GenericDivide { left, right, .. }
        | ValueNode::GenericModulus { left, right, .. }
        | ValueNode::GenericExponentiate { left, right, .. }
        | ValueNode::GenericBitwiseAnd { left, right, .. }
        | ValueNode::GenericBitwiseOr { left, right, .. }
        | ValueNode::GenericBitwiseXor { left, right, .. }
        | ValueNode::GenericShiftLeft { left, right, .. }
        | ValueNode::GenericShiftRight { left, right, .. }
        | ValueNode::GenericShiftRightLogical { left, right, .. }
        | ValueNode::TaggedEqual { left, right, .. }
        | ValueNode::TaggedNotEqual { left, right, .. } => {
            f(*left);
            f(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            f(*left);
            f(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            f(*receiver);
            f(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            f(*object);
            f(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            f(*key);
            f(*object);
        }

        ValueNode::StringAt { string, index } => {
            f(*string);
            f(*index);
        }

        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            f(*receiver);
            f(*cache_index);
            f(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => f(*exception),
        ValueNode::CreateWithContext { object, .. } => f(*object),

        ValueNode::Call {
            callee,
            receiver,
            args,
            ..
        }
        | ValueNode::CallWithSpread {
            callee,
            receiver,
            args,
            ..
        }
        | ValueNode::CallKnownFunction {
            callee,
            receiver,
            args,
        } => {
            f(*callee);
            f(*receiver);
            for &a in args {
                f(a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for &a in args {
                f(a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            f(*constructor);
            for &a in args {
                f(a);
            }
        }

        ValueNode::Phi { inputs } => {
            for &inp in inputs {
                f(inp);
            }
        }
    }
}

/// Call `f` for each [`NodeId`] operand of `ctrl`.
fn collect_control_inputs(ctrl: &ControlNode, f: &mut impl FnMut(NodeId)) {
    match ctrl {
        ControlNode::Return { value } => f(*value),
        ControlNode::Branch { condition, .. } => f(*condition),
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear-scan register allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Run linear-scan register allocation over `graph` using `num_regs` physical
/// registers.
///
/// Values whose live intervals cannot be assigned a register are *spilled* to
/// numbered stack slots.  The algorithm follows Poletto & Sarkar (1999):
/// intervals are processed in start-point order; when all registers are in use
/// the active interval with the farthest endpoint is spilled if it ends later
/// than the current interval, otherwise the current interval is spilled.
///
/// # Loop awareness
///
/// The current implementation does **not** weight spill decisions by loop
/// depth.  Values live across a loop back-edge naturally have long intervals
/// and are therefore *more* likely to be spilled — the opposite of what we
/// want.  A future improvement would detect loop back-edges, compute loop
/// nesting depth for each interval, and multiply the spill cost by the depth
/// so that loop-live values are preferentially kept in registers.
///
/// # Panics
///
/// Panics if `num_regs` is `0`.
pub fn allocate(graph: &MaglevGraph, num_regs: u32) -> AllocationResult {
    assert!(num_regs > 0, "allocate: num_regs must be > 0");

    let mut intervals = compute_live_intervals(graph);
    // Sort by start point; break ties by NodeId for determinism.
    intervals.sort_by_key(|iv| (iv.start, iv.id.0));

    let mut assignments: HashMap<NodeId, Location> = HashMap::new();
    // Active intervals: (register, interval).
    let mut active: Vec<(u32, LiveInterval)> = Vec::new();
    let mut next_spill: u32 = 0;
    // Free-register pool (lower-numbered registers are preferred).
    let mut free_regs: Vec<u32> = (0..num_regs).rev().collect();

    // Build Phi affinity hints: back-edge input → Phi's NodeId.
    // After allocating Phis, we populate a second map from
    // back-edge input → preferred register (the Phi's register).
    let mut phi_affinity_phi: HashMap<NodeId, NodeId> = HashMap::new();
    for block in graph.blocks() {
        let back_edge_positions: Vec<usize> = block
            .predecessors
            .iter()
            .enumerate()
            .filter(|&(_, &pred_idx)| pred_idx >= block.id)
            .map(|(pos, _)| pos)
            .collect();
        if back_edge_positions.is_empty() {
            continue;
        }
        for (phi_id, node) in &block.nodes {
            if let ValueNode::Phi { inputs } = node {
                for &pos in &back_edge_positions {
                    if let Some(&back_input) = inputs.get(pos)
                        && back_input != *phi_id
                    {
                        phi_affinity_phi.insert(back_input, *phi_id);
                    }
                }
            }
        }
    }

    // Filled lazily: back-edge input → preferred register number.
    let mut phi_affinity_reg: HashMap<NodeId, u32> = HashMap::new();

    // Reverse map: Phi → list of back-edge inputs that want its register.
    let mut phi_back_inputs: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (&back_input, &phi_id) in &phi_affinity_phi {
        phi_back_inputs.entry(phi_id).or_default().push(back_input);
    }

    for iv in &intervals {
        // Expire intervals whose live range ends at or before this start point,
        // reclaiming their registers.
        let mut still_active: Vec<(u32, LiveInterval)> = Vec::new();
        for (reg, active_iv) in active.drain(..) {
            if active_iv.end <= iv.start {
                free_regs.push(reg);
            } else {
                still_active.push((reg, active_iv));
            }
        }
        active = still_active;

        // ── Early Phi expiry ───────────────────────────────────────────
        //
        // When the current interval is a back-edge input with a Phi
        // affinity, and the Phi's *real* live range (before loop
        // extension) has ended, expire the Phi from the active set.
        // This frees the Phi's register so the normal affinity-hint
        // logic can immediately reuse it, avoiding an unnecessary MOV
        // and reducing register pressure in the loop body.
        if let Some(&pref_reg) = phi_affinity_reg.get(&iv.id)
            && let Some(&phi_id) = phi_affinity_phi.get(&iv.id)
            && let Some(active_idx) = active
                .iter()
                .position(|(r, aiv)| *r == pref_reg && aiv.id == phi_id)
        {
            let phi_eff_end = active[active_idx]
                .1
                .pre_ext_end
                .unwrap_or(active[active_idx].1.end);
            if phi_eff_end <= iv.start + 1 {
                active.remove(active_idx);
                free_regs.push(pref_reg);
            }
        }

        if let Some(reg) = {
            // Prefer the Phi-affinity register if available.
            let preferred = phi_affinity_reg.get(&iv.id).copied();
            if let Some(pref) = preferred {
                if let Some(pos) = free_regs.iter().position(|&r| r == pref) {
                    Some(free_regs.remove(pos))
                } else if !free_regs.is_empty() {
                    // The preferred register is occupied by a non-Phi
                    // value.  If a different free register exists,
                    // shuffle: move the occupant to the free register
                    // and hand the preferred one to the current interval.
                    if let Some(occ_idx) = active.iter().position(|(r, _)| *r == pref) {
                        let alt_reg = free_regs.pop().unwrap();
                        let occupant_id = active[occ_idx].1.id;
                        assignments.insert(occupant_id, Location::Register(alt_reg));
                        active[occ_idx].0 = alt_reg;
                        Some(pref)
                    } else {
                        free_regs.pop()
                    }
                } else {
                    None
                }
            } else {
                free_regs.pop()
            }
        } {
            // A free register is available — assign it.
            assignments.insert(iv.id, Location::Register(reg));
            active.push((reg, *iv));

            // If this value is a Phi, record affinity for its back-edge inputs.
            if let Some(back_inputs) = phi_back_inputs.get(&iv.id) {
                for &back_input in back_inputs {
                    phi_affinity_reg.insert(back_input, reg);
                }
            }
        } else {
            // All registers occupied — spill the interval with the farthest
            // endpoint.  If that interval ends later than the current one,
            // evict it and give its register to the current interval; otherwise
            // spill the current interval.
            let spill_idx = active
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, a))| a.end)
                .map(|(i, _)| i);

            // active is non-empty because free_regs is empty and the
            // invariant active.len() + free_regs.len() == num_regs holds.
            let idx = spill_idx.expect("active must be non-empty when no free regs");
            if active[idx].1.end > iv.end {
                // Evict the farthest-ending active interval.
                let (reg, spilled_iv) = active.remove(idx);
                let slot = next_spill;
                next_spill += 1;
                assignments.insert(spilled_iv.id, Location::StackSlot(slot));
                assignments.insert(iv.id, Location::Register(reg));
                active.push((reg, *iv));
            } else {
                // Spill the current interval.
                let slot = next_spill;
                next_spill += 1;
                assignments.insert(iv.id, Location::StackSlot(slot));
            }
        }
    }

    // ── Phi coalescing ─────────────────────────────────────────────────────
    //
    // After the main allocation, try to coalesce loop-carried Phi values
    // with their back-edge inputs so that the codegen can skip the MOV at
    // the loop back-edge.
    coalesce_loop_phis(&mut assignments, graph, &intervals);

    // ── Operand-conflict fixup ──────────────────────────────────────────────
    //
    // Safety net: ensure that no non-Phi instruction has two *different*
    // input NodeIds allocated to the same physical register.  This can
    // happen when interval computation is slightly conservative or when
    // Phi coalescing reassigns a back-edge input to a register that
    // conflicts with a co-operand of the same instruction.  When detected,
    // the second operand is demoted to a fresh stack slot.
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            if matches!(node, ValueNode::Phi { .. }) {
                continue;
            }
            let mut seen: Vec<(u32, NodeId)> = Vec::new();
            collect_inputs(node, &mut |inp| {
                if let Some(&Location::Register(r)) = assignments.get(&inp) {
                    let dominated = seen.iter().any(|&(sr, sid)| sr == r && sid != inp);
                    if dominated {
                        // Conflict: a previous input already occupies this
                        // register.  Spill this operand.
                        let slot = next_spill;
                        next_spill += 1;
                        assignments.insert(inp, Location::StackSlot(slot));
                    } else if !seen.iter().any(|&(_, sid)| sid == inp) {
                        seen.push((r, inp));
                    }
                }
            });
        }
    }

    // ── Per-node caller-saved liveness ──────────────────────────────────────
    //
    // For each node at program point P, compute a bitmask of caller-saved
    // register indices (1–5) whose live intervals span *across* P, i.e. the
    // value is defined before P and still needed after P.  This lets codegen
    // save only the registers that are truly live at each stub-call site.

    // Collect intervals that ended up in caller-saved registers.
    let cs_intervals: Vec<(u32, u32, u8)> = intervals
        .iter()
        .filter_map(|iv| {
            if let Some(Location::Register(n)) = assignments.get(&iv.id)
                && (1..=5).contains(n)
            {
                return Some((iv.start, iv.end, 1u8 << n));
            }
            None
        })
        .collect();

    let mut live_caller_saved: HashMap<NodeId, u8> = HashMap::new();

    if !cs_intervals.is_empty() {
        // Rebuild the program-point map (same iteration order as
        // compute_live_intervals).
        let mut pp: u32 = 0;
        for block in graph.blocks() {
            for &(nid, _) in &block.nodes {
                let mut mask: u8 = 0;
                for &(start, end, bit) in &cs_intervals {
                    // The value is live *across* this node when it was
                    // defined strictly before and its last use is strictly
                    // after this program point.
                    if start < pp && end > pp + 1 {
                        mask |= bit;
                    }
                }
                if mask != 0 {
                    live_caller_saved.insert(nid, mask);
                }
                pp += 1;
            }
            if block.control.is_some() {
                pp += 1;
            }
        }
    }

    AllocationResult {
        assignments,
        spill_count: next_spill,
        live_caller_saved,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phi coalescing
// ─────────────────────────────────────────────────────────────────────────────

/// Post-allocation pass that coalesces loop-carried Phi values with their
/// back-edge inputs.
///
/// For each Phi in a loop header block, if the Phi occupies `Register(A)` and
/// its back-edge input occupies `Register(B)` where `A ≠ B`, the pass checks
/// two conditions:
///
/// 1. **No live-range conflict** — the Phi's last use must be at or before the
///    back-edge input's definition point.  This ensures the Phi is dead (or
///    consumed as the immediate input to the defining instruction) when the
///    result is written, so sharing a register is safe.
///
/// 2. **No third-party interference** — no other value assigned to
///    `Register(A)` has a live interval overlapping the back-edge input's
///    interval.
///
/// When both conditions hold the back-edge input is reassigned to
/// `Register(A)`, and the codegen's `emit_phi_copies_for_successor` will skip
/// the MOV at the loop back-edge because source and destination now match.
fn coalesce_loop_phis(
    assignments: &mut HashMap<NodeId, Location>,
    graph: &MaglevGraph,
    intervals: &[LiveInterval],
) {
    // Build a map from NodeId to its live interval for O(1) lookup.
    let interval_map: HashMap<NodeId, &LiveInterval> =
        intervals.iter().map(|iv| (iv.id, iv)).collect();

    for block in graph.blocks() {
        // A loop header has at least one *back-edge*: a predecessor whose
        // block index is ≥ the current block's index (i.e. a backward or
        // self-referential edge in the CFG).
        let back_edge_positions: Vec<usize> = block
            .predecessors
            .iter()
            .enumerate()
            .filter(|&(_, &pred_idx)| pred_idx >= block.id)
            .map(|(pos, _)| pos)
            .collect();

        if back_edge_positions.is_empty() {
            continue;
        }

        for (phi_id, node) in &block.nodes {
            let inputs = match node {
                ValueNode::Phi { inputs } => inputs,
                _ => continue,
            };

            let phi_reg = match assignments.get(phi_id) {
                Some(Location::Register(r)) => *r,
                _ => continue,
            };

            let phi_iv = match interval_map.get(phi_id) {
                Some(iv) => iv,
                None => continue,
            };
            // Use the pre-extension end for Phis whose intervals were
            // artificially extended to cover the loop body.  The extension
            // prevents register reuse but doesn't represent a genuine use,
            // so it should not block coalescing.
            let phi_effective_end = phi_iv.pre_ext_end.unwrap_or(phi_iv.end);

            for &pos in &back_edge_positions {
                let back_input = match inputs.get(pos) {
                    Some(&id) => id,
                    None => continue,
                };

                // Skip self-referential Phis (e.g. `phi = Phi([…, phi])`).
                if back_input == *phi_id {
                    continue;
                }

                let back_reg = match assignments.get(&back_input) {
                    Some(Location::Register(r)) => *r,
                    _ => continue,
                };

                // Already in the same register — early Phi expiry
                // succeeded; nothing more to do.
                if back_reg == phi_reg {
                    #[cfg(feature = "regalloc-trace")]
                    eprintln!(
                        "[regalloc] coalesce: phi {:?} (R{}) ← back {:?}: \
                         already coalesced (early expiry)",
                        phi_id, phi_reg, back_input
                    );
                    continue;
                }

                let back_iv = match interval_map.get(&back_input) {
                    Some(iv) => iv,
                    None => continue,
                };

                // Condition 1: the Phi must be dead (or at its very last use)
                // by the time the back-edge input is defined.  An overlap of
                // exactly one program point is acceptable — that is the point
                // where the defining instruction reads the Phi and writes the
                // result into the same register.
                if phi_effective_end > back_iv.start + 1 {
                    #[cfg(feature = "regalloc-trace")]
                    eprintln!(
                        "[regalloc] coalesce FAIL (live-range overlap): \
                         phi {:?} (R{}, eff_end={}) ← back {:?} (R{}, start={})",
                        phi_id, phi_reg, phi_effective_end, back_input, back_reg, back_iv.start
                    );
                    continue;
                }

                // Condition 2: no other value in Register(phi_reg) has an
                // interval that overlaps the back-edge input's interval.  We
                // exclude the Phi itself (already checked above) and the
                // back-edge input (we are moving it).
                let has_conflict = intervals.iter().any(|iv| {
                    if iv.id == back_input || iv.id == *phi_id {
                        return false;
                    }
                    // Intervals overlap when a.start < b.end && b.start < a.end.
                    if iv.start < back_iv.end && back_iv.start < iv.end {
                        matches!(
                            assignments.get(&iv.id),
                            Some(Location::Register(r)) if *r == phi_reg
                        )
                    } else {
                        false
                    }
                });

                if !has_conflict {
                    #[cfg(feature = "regalloc-trace")]
                    eprintln!(
                        "[regalloc] coalesce OK (post-alloc): \
                         phi {:?} (R{}) ← back {:?} (was R{})",
                        phi_id, phi_reg, back_input, back_reg
                    );
                    assignments.insert(back_input, Location::Register(phi_reg));
                } else {
                    #[cfg(feature = "regalloc-trace")]
                    eprintln!(
                        "[regalloc] coalesce FAIL (interference): \
                         phi {:?} (R{}) ← back {:?} (R{})",
                        phi_id, phi_reg, back_input, back_reg
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Assert that no two overlapping live intervals are assigned the same
    /// physical register.
    fn assert_no_conflicts(graph: &MaglevGraph, result: &AllocationResult, num_regs: u32) {
        let intervals = compute_live_intervals(graph);
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                let a = &intervals[i];
                let b = &intervals[j];
                // Intervals overlap when a.start < b.end && b.start < a.end.
                if a.start < b.end && b.start < a.end {
                    if let (Some(Location::Register(ra)), Some(Location::Register(rb))) =
                        (result.location(a.id), result.location(b.id))
                    {
                        assert_ne!(
                            ra, rb,
                            "register conflict: {:?} and {:?} both in register {} \
                             (intervals [{},{}) and [{},{}) with {} regs)",
                            a.id, b.id, ra, a.start, a.end, b.start, b.end, num_regs
                        );
                    }
                }
            }
        }
    }

    // ── Test: single parameter gets a register ────────────────────────────────

    #[test]
    fn test_single_parameter_gets_register() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        block.set_control(ControlNode::Return { value: p0 });
        graph.add_block(block);

        let result = allocate(&graph, 8);
        assert!(matches!(result.location(p0), Some(Location::Register(_))));
        assert_eq!(result.spill_count(), 0);
        assert_no_conflicts(&graph, &result, 8);
    }

    // ── Test: two independent constants, both get registers ───────────────────

    #[test]
    fn test_two_independent_values_no_spill() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let c2 = block.push_value(ValueNode::Int32Constant { value: 2 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c1,
            right: c2,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let result = allocate(&graph, 8);
        assert_no_conflicts(&graph, &result, 8);
        assert_eq!(result.spill_count(), 0);
    }

    // ── Test: forced spill with only 1 register available ────────────────────

    #[test]
    fn test_spill_with_one_register() {
        // p0 and p1 are both live at the Int32Add node; with 1 register one
        // of them must be spilled.
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let add = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let result = allocate(&graph, 1);
        assert_no_conflicts(&graph, &result, 1);
        assert!(result.spill_count() > 0, "expected at least one spill");
    }

    // ── Test: two-block graph with unconditional jump ─────────────────────────

    #[test]
    fn test_two_block_graph() {
        let mut graph = MaglevGraph::new(1);

        // block 0: define parameter, jump to block 1
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // block 1: return the parameter defined in block 0
        let mut b1 = BasicBlock::new(1);
        b1.set_control(ControlNode::Return { value: p0 });
        graph.add_block(b1);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
        assert!(matches!(result.location(p0), Some(Location::Register(_))));
    }

    // ── Test: branch graph ────────────────────────────────────────────────────

    #[test]
    fn test_branch_graph() {
        let mut graph = MaglevGraph::new(1);

        // block 0: condition based on parameter; branch to block 1 or block 2
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        // block 1: return TrueConstant
        let mut b1 = BasicBlock::new(1);
        let t = b1.push_value(ValueNode::TrueConstant);
        b1.set_control(ControlNode::Return { value: t });
        graph.add_block(b1);

        // block 2: return FalseConstant
        let mut b2 = BasicBlock::new(2);
        let f = b2.push_value(ValueNode::FalseConstant);
        b2.set_control(ControlNode::Return { value: f });
        graph.add_block(b2);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
    }

    // ── Test: chain of unary ops — at most 2 values live at once ─────────────

    #[test]
    fn test_no_spills_with_sufficient_registers() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c0 = block.push_value(ValueNode::Int32Constant { value: 42 });
        let neg0 = block.push_value(ValueNode::Int32Negate { value: c0 });
        let neg1 = block.push_value(ValueNode::Int32Negate { value: neg0 });
        let neg2 = block.push_value(ValueNode::Int32Negate { value: neg1 });
        block.set_control(ControlNode::Return { value: neg2 });
        graph.add_block(block);

        // At most 2 values are live simultaneously along this chain.
        let result = allocate(&graph, 4);
        assert_eq!(result.spill_count(), 0);
        assert_no_conflicts(&graph, &result, 4);
    }

    // ── Test: every node receives a location ─────────────────────────────────

    #[test]
    fn test_all_nodes_get_location() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let add = block.push_value(ValueNode::Int32Add {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let result = allocate(&graph, 2);
        assert!(result.location(p0).is_some(), "p0 missing location");
        assert!(result.location(p1).is_some(), "p1 missing location");
        assert!(result.location(add).is_some(), "add missing location");
        assert_no_conflicts(&graph, &result, 2);
    }

    // ── Test: Phi node — inputs collected across blocks ───────────────────────

    #[test]
    fn test_phi_node_allocation() {
        let mut graph = MaglevGraph::new(1);

        // block 0: branch on parameter
        let mut b0 = BasicBlock::new(0);
        let p0 = b0.push_value(ValueNode::Parameter { index: 0 });
        let cond = b0.push_value(ValueNode::ToBoolean { value: p0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        // block 1: produce value 1
        let mut b1 = BasicBlock::new(1);
        let v1 = b1.push_value(ValueNode::Int32Constant { value: 1 });
        b1.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b1);

        // block 2: produce value 2
        let mut b2 = BasicBlock::new(2);
        let v2 = b2.push_value(ValueNode::Int32Constant { value: 2 });
        b2.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b2);

        // block 3: phi merging v1 and v2, then return
        let mut b3 = BasicBlock::new(3);
        let phi = b3.push_value(ValueNode::Phi {
            inputs: vec![v1, v2],
        });
        b3.set_control(ControlNode::Return { value: phi });
        graph.add_block(b3);

        let result = allocate(&graph, 4);
        assert_no_conflicts(&graph, &result, 4);
        assert!(result.location(phi).is_some(), "phi missing location");
    }

    // ── Test: Phi coalescing eliminates back-edge MOV ─────────────────────────

    #[test]
    fn test_phi_coalescing_loop() {
        // Build a counting loop where the branch condition comes from block 0
        // (a parameter), so no extra node in the loop body steals the Phi's
        // register after it expires:
        //
        //   block 0:  param = Parameter(0)
        //             init  = Const(0)
        //             c1    = Const(1)
        //             Jump → block 1
        //
        //   block 1:  phi = Phi([init, add])      ← loop header
        //             add = Int32Add(phi, c1)
        //             Branch(param) → block 1 | block 2
        //
        //   block 2:  Return(add)
        //
        // Without coalescing `phi` and `add` end up in different registers,
        // causing a MOV at the back-edge.  The coalescing pass reassigns
        // `add` to the same register as `phi`.

        let mut graph = MaglevGraph::new(1);

        // ── Block 0 ─────────────────────────────────────────────────────────
        graph.add_block(BasicBlock::new(0));
        let param = graph
            .add_value_node(0, ValueNode::Parameter { index: 0 })
            .unwrap();
        let init = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let c1 = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 1 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        // ── Block 1 (loop header) ───────────────────────────────────────────
        graph.add_block(BasicBlock::new(1));
        graph.block_mut(1).unwrap().add_predecessor(0);
        graph.block_mut(1).unwrap().add_predecessor(1); // back-edge

        // Pre-allocate a NodeId for `add` so it can appear in the Phi.
        let add_id = graph.alloc_node_id();

        let phi = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init, add_id],
                },
            )
            .unwrap();

        // Insert `add` with its pre-allocated ID.
        graph.block_mut(1).unwrap().push_with_id(
            add_id,
            ValueNode::Int32Add {
                left: phi,
                right: c1,
            },
        );

        // Use the parameter (from block 0) as branch condition — no new
        // value node in the loop body that could take the Phi's register.
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: param,
                if_true: 1,
                if_false: 2,
            });

        // ── Block 2 (exit) ──────────────────────────────────────────────────
        graph.add_block(BasicBlock::new(2));
        graph.block_mut(2).unwrap().add_predecessor(1);
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Return { value: add_id });

        let result = allocate(&graph, 4);

        // Both phi and add must have locations.
        assert!(result.location(phi).is_some(), "phi missing location");
        assert!(result.location(add_id).is_some(), "add missing location");

        // The coalescing pass should place phi and its back-edge input (add)
        // in the same physical register, eliminating the back-edge MOV.
        assert!(
            matches!(result.location(phi), Some(Location::Register(_))),
            "phi should be in a register, got {:?}",
            result.location(phi)
        );
        assert_eq!(
            result.location(phi),
            result.location(add_id),
            "phi and back-edge input (add) should be coalesced into the \
             same register — phi={:?}, add={:?}",
            result.location(phi),
            result.location(add_id)
        );

        // Suppress unused-variable warning — `param` is used only as a
        // branch condition, not asserted on.
        let _ = param;
    }

    // ── Test: coalescing is skipped when a conflict exists ────────────────────

    #[test]
    fn test_phi_coalescing_skipped_on_conflict() {
        // Same loop structure, but with a value that occupies the Phi's
        // register during the back-edge input's live range, blocking
        // coalescing.  The allocator must leave the original (conflict-free)
        // assignment in place.

        let mut graph = MaglevGraph::new(0);

        graph.add_block(BasicBlock::new(0));
        let init = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        graph.add_block(BasicBlock::new(1));
        graph.block_mut(1).unwrap().add_predecessor(0);
        graph.block_mut(1).unwrap().add_predecessor(1);

        let add_id = graph.alloc_node_id();

        let phi = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![init, add_id],
                },
            )
            .unwrap();

        // Use phi in BOTH the add and a separate check — this keeps phi
        // alive past add's definition, which violates the safety condition
        // (phi_end > back_start + 1) and prevents coalescing.
        graph.block_mut(1).unwrap().push_with_id(
            add_id,
            ValueNode::Int32Add {
                left: phi,
                right: init,
            },
        );
        let extra_use = graph
            .add_value_node(1, ValueNode::ToBoolean { value: phi })
            .unwrap();
        let cond = graph
            .add_value_node(
                1,
                ValueNode::Int32Add {
                    left: add_id,
                    right: extra_use,
                },
            )
            .unwrap();
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: cond,
                if_true: 1,
                if_false: 2,
            });

        graph.add_block(BasicBlock::new(2));
        graph.block_mut(2).unwrap().add_predecessor(1);
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Return { value: cond });

        let result = allocate(&graph, 4);
        // With enough registers the original allocation should have no
        // conflicts (verified by the standard helper).
        assert_no_conflicts(&graph, &result, 4);
    }

    // ── Test: arithmetic loop (two Phis, multi-op body) coalesces both ────────

    #[test]
    fn test_arithmetic_loop_dual_phi_coalescing() {
        // Mimics the benchmark:
        //   for (var i = 0; i < 10000; i++) { n = (n + i * 3 - 1) | 0; }
        //
        // IR pattern:
        //   Block 0 (preheader):
        //     n_init = Const(0)
        //     i_init = Const(0)
        //     limit  = Const(10000)
        //     Jump → Block 1
        //
        //   Block 1 (loop header):
        //     n_phi  = Phi([n_init, n_new])
        //     i_phi  = Phi([i_init, i_new])
        //     t1     = i_phi * 3         (strength-reduced to LEA)
        //     n_new  = n_phi + t1 - 1    (fused LEA, after |0 elim)
        //     i_new  = i_phi + 1         (increment)
        //     cmp    = i_new < limit
        //     Branch(cmp) → Block 1 | Block 2
        //
        //   Block 2 (exit):
        //     Return(n_new)
        //
        // Both n_phi/n_new and i_phi/i_new must coalesce (same register).

        let mut graph = MaglevGraph::new(0);

        // ── Block 0 (preheader) ────────────────────────────────────────────
        graph.add_block(BasicBlock::new(0));
        let n_init = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let i_init = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 0 })
            .unwrap();
        let limit = graph
            .add_value_node(0, ValueNode::Int32Constant { value: 10000 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Jump { target: 1 });

        // ── Block 1 (loop header) ──────────────────────────────────────────
        graph.add_block(BasicBlock::new(1));
        graph.block_mut(1).unwrap().add_predecessor(0);
        graph.block_mut(1).unwrap().add_predecessor(1); // back-edge

        // Pre-allocate IDs for back-edge inputs so Phis can reference them.
        let n_new_id = graph.alloc_node_id();
        let i_new_id = graph.alloc_node_id();

        let n_phi = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![n_init, n_new_id],
                },
            )
            .unwrap();
        let i_phi = graph
            .add_value_node(
                1,
                ValueNode::Phi {
                    inputs: vec![i_init, i_new_id],
                },
            )
            .unwrap();

        // t1 = i_phi * 3  (modelled as Int32Multiply)
        let t1 = graph
            .add_value_node(
                1,
                ValueNode::Int32Multiply {
                    left: i_phi,
                    right: limit, // reuse constant node as stand-in
                },
            )
            .unwrap();

        // n_new = n_phi + t1  (stands in for the fused add-sub LEA)
        graph.block_mut(1).unwrap().push_with_id(
            n_new_id,
            ValueNode::Int32Add {
                left: n_phi,
                right: t1,
            },
        );

        // i_new = i_phi + 1
        graph.block_mut(1).unwrap().push_with_id(
            i_new_id,
            ValueNode::Int32Add {
                left: i_phi,
                right: i_init, // reuse 0 constant as stand-in
            },
        );

        // Branch: i_new < limit → loop | exit
        let cmp = graph
            .add_value_node(
                1,
                ValueNode::Int32LessThan {
                    left: i_new_id,
                    right: limit,
                },
            )
            .unwrap();
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: cmp,
                if_true: 1,
                if_false: 2,
            });

        // ── Block 2 (exit) ─────────────────────────────────────────────────
        graph.add_block(BasicBlock::new(2));
        graph.block_mut(2).unwrap().add_predecessor(1);
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Return { value: n_new_id });

        let result = allocate(&graph, 8);

        // Both Phis should be in registers.
        assert!(
            matches!(result.location(n_phi), Some(Location::Register(_))),
            "n_phi should be in a register, got {:?}",
            result.location(n_phi)
        );
        assert!(
            matches!(result.location(i_phi), Some(Location::Register(_))),
            "i_phi should be in a register, got {:?}",
            result.location(i_phi)
        );

        // Both back-edge inputs must coalesce with their Phis.
        assert_eq!(
            result.location(n_phi),
            result.location(n_new_id),
            "n_phi and n_new should share a register — n_phi={:?}, n_new={:?}",
            result.location(n_phi),
            result.location(n_new_id)
        );
        assert_eq!(
            result.location(i_phi),
            result.location(i_new_id),
            "i_phi and i_new should share a register — i_phi={:?}, i_new={:?}",
            result.location(i_phi),
            result.location(i_new_id)
        );

        // Suppress unused-variable warnings.
        let _ = (t1, cmp);
    }
}

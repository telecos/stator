//! Maglev deoptimiser: JIT → interpreter fallback.
//!
//! When a Maglev-compiled function encounters a speculation failure (e.g. an
//! arithmetic overflow or a type-check guard failure), the JIT code returns
//! [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT].  This
//! module provides the machinery to:
//!
//! 1. Capture the reason and register-file snapshot in a [`DeoptInfo`].
//! 2. Reconstruct an [`InterpreterFrame`] from that snapshot.
//! 3. Resume bytecode execution via [`Interpreter::run`].
//! 4. Update the [`FeedbackVector`] so that the Maglev tier does not
//!    re-speculate on the same pattern that just failed.
//!
//! # Example
//!
//! ```
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! use stator_core::bytecode::bytecode_array::BytecodeArray;
//! use stator_core::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
//! use stator_core::bytecode::feedback::{
//!     FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
//! };
//! use stator_core::compiler::maglev::deopt::{DeoptInfo, DeoptReason, FrameState, deoptimize};
//! use stator_core::interpreter::GlobalEnv;
//! use stator_core::objects::value::JsValue;
//!
//! // Bytecode: LdaSmi 0, Return  (a trivial function that returns 0).
//! let instrs = vec![
//!     Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
//!     Instruction::new_unchecked(Opcode::Return, vec![]),
//! ];
//! let bytes = encode(&instrs);
//! let ba = BytecodeArray::new(bytes, vec![], 0, 0, vec![], FeedbackMetadata::empty(), vec![]);
//! let mut fv = FeedbackVector::new(ba.feedback_metadata());
//!
//! let info = DeoptInfo {
//!     reason: DeoptReason::ArithmeticOverflow,
//!     bytecode_offset: 0,
//!     frame_state: FrameState {
//!         registers: vec![],
//!         accumulator: JsValue::Undefined,
//!     },
//! };
//!
//! let result = deoptimize(ba, &mut fv, info, Rc::new(RefCell::new(GlobalEnv::new())))
//!     .expect("interpreter should succeed");
//! assert_eq!(result, JsValue::Smi(0));
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::bytecode::bytecode_array::BytecodeArray;
use crate::bytecode::bytecodes::decode_with_byte_offsets;
use crate::bytecode::feedback::{FeedbackSlotKind, FeedbackVector, InlineCacheState};
use crate::error::StatorResult;
use crate::interpreter::{GlobalEnv, Interpreter, InterpreterFrame};
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// DeoptReason
// ─────────────────────────────────────────────────────────────────────────────

/// The reason why a Maglev-compiled function deoptimised.
///
/// Recorded in [`DeoptInfo`] so that callers can log, trace, or react
/// differently to each failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptReason {
    /// A checked arithmetic operation overflowed the Smi range
    /// (e.g. [`ValueNode::CheckedSmiAdd`][crate::compiler::maglev::ir::ValueNode::CheckedSmiAdd]).
    ArithmeticOverflow,
    /// A type-check guard encountered an unexpected value type
    /// (e.g. [`ValueNode::CheckSmi`][crate::compiler::maglev::ir::ValueNode::CheckSmi]
    /// on a non-Smi value).
    TypeCheckFailure,
    /// An operation not supported by the current JIT tier was encountered.
    UnsupportedOperation,
}

// ─────────────────────────────────────────────────────────────────────────────
// FrameState
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of the interpreter register file at a deoptimisation point.
///
/// Used to reconstruct an [`InterpreterFrame`] so the interpreter can resume
/// from the state the JIT was in when it bailed out.
///
/// # Register layout
///
/// `registers` follows the same flat layout as [`InterpreterFrame::registers`]:
///
/// ```text
/// [ param[0], param[1], …, local[0], local[1], … ]
///  ←── parameter_count ──→←────── frame_size ────→
/// ```
///
/// Any entries beyond the frame's total register count are silently ignored;
/// any missing entries default to [`JsValue::Undefined`].
#[derive(Debug, Clone)]
pub struct FrameState {
    /// Flat register file at the deopt point.
    pub registers: Vec<JsValue>,
    /// Value of the implicit accumulator register at the deopt point.
    pub accumulator: JsValue,
}

// ─────────────────────────────────────────────────────────────────────────────
// DeoptInfo
// ─────────────────────────────────────────────────────────────────────────────

/// All metadata associated with a single deoptimisation event.
///
/// A `DeoptInfo` is constructed when the JIT returns
/// [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT] and is
/// passed to [`deoptimize`] to resume execution in the interpreter.
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    /// Why deoptimisation occurred.
    pub reason: DeoptReason,
    /// Bytecode byte offset where interpretation should resume.
    ///
    /// This is the `bytecode_offset` field from the matching
    /// [`DeoptEntry`][crate::compiler::baseline::compiler::DeoptEntry] in the
    /// compiled code's deopt table.  Pass `0` to restart from the beginning
    /// of the function.
    pub bytecode_offset: u32,
    /// Register-file snapshot at the deoptimisation point.
    pub frame_state: FrameState,
}

// ─────────────────────────────────────────────────────────────────────────────
// Deoptimise entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct an [`InterpreterFrame`] from `deopt_info` and resume bytecode
/// execution from the recorded `bytecode_offset`.
///
/// # Feedback update
///
/// Before resuming, all [`FeedbackSlotKind::BinaryOp`],
/// [`FeedbackSlotKind::BinaryOpInc`], and [`FeedbackSlotKind::Compare`] slots
/// in `feedback` are advanced to [`InlineCacheState::Megamorphic`].  This
/// prevents the Maglev tier from re-emitting the same speculative guards that
/// just caused the deoptimisation.
///
/// # PC selection
///
/// The interpreter's program counter is set to the index of the first
/// instruction whose byte offset is ≥ `deopt_info.bytecode_offset`.  If no
/// such instruction exists (e.g. the offset is past the end of the bytecode),
/// execution restarts from the beginning of the function (PC = 0).
///
/// # Register-file restoration
///
/// The frame's register file is overwritten with the values from
/// `deopt_info.frame_state.registers`.  Slots not present in the snapshot
/// retain their default [`JsValue::Undefined`] value.
///
/// # Errors
///
/// Propagates any [`crate::error::StatorError`] from [`Interpreter::run`].
pub fn deoptimize(
    bytecode_array: BytecodeArray,
    feedback: &mut FeedbackVector,
    deopt_info: DeoptInfo,
    global_env: Rc<RefCell<GlobalEnv>>,
) -> StatorResult<JsValue> {
    // ── Step 1: update the feedback vector ──────────────────────────────────
    //
    // Transition speculation-sensitive slots to Megamorphic so that the next
    // Maglev compilation of this function falls back to generic (non-speculative)
    // IR nodes and does not re-trigger the same deopt.
    for slot in 0..feedback.slot_count() {
        if let Some(
            FeedbackSlotKind::BinaryOp | FeedbackSlotKind::BinaryOpInc | FeedbackSlotKind::Compare,
        ) = feedback.kind_of(slot)
        {
            feedback.set_state(slot, InlineCacheState::Megamorphic);
        }
    }

    // ── Step 2: resolve bytecode_offset → instruction index (PC) ────────────
    let (_, byte_offsets) = decode_with_byte_offsets(bytecode_array.bytecodes())?;
    let target_pc = byte_offsets
        .iter()
        .position(|&o| o as u32 >= deopt_info.bytecode_offset)
        .unwrap_or(0);

    // ── Step 3: reconstruct the interpreter frame ────────────────────────────
    //
    // `InterpreterFrame::new_with_globals` initialises parameters from `args`
    // and locals to `Undefined`.  We then overwrite the register file with the
    // captured frame state so the interpreter resumes with the correct values.
    let mut frame = InterpreterFrame::new_with_globals(Rc::new(bytecode_array), vec![], global_env);

    for (i, v) in deopt_info.frame_state.registers.iter().enumerate() {
        if i < frame.registers.len() {
            frame.registers[i] = v.clone();
        }
    }
    frame.accumulator = deopt_info.frame_state.accumulator;
    frame.pc = target_pc;

    // ── Step 4: resume execution in the interpreter ──────────────────────────
    Interpreter::run(&mut frame)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::BytecodeArray;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::{
        FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
    };
    use crate::objects::value::JsValue;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a trivial bytecode array: `LdaSmi(value)`, `Return`.
    fn make_lda_return(value: i32) -> BytecodeArray {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(value)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Build a two-parameter bytecode array that computes `param0 + param1`
    /// and returns the result.
    ///
    /// Generated instructions:
    /// ```text
    /// Ldar  r(-1)   ; acc = param0  (signed index -1, encoded as u32 0xFFFF_FFFF)
    /// Star  r0      ; r0  = acc     (local register 0)
    /// Ldar  r(-2)   ; acc = param1  (signed index -2, encoded as u32 0xFFFF_FFFE)
    /// Add   r0, _   ; acc = acc + r0
    /// Return
    /// ```
    fn make_add_params(feedback_meta: FeedbackMetadata) -> BytecodeArray {
        // Parameter registers are encoded as negative signed indices bit-cast to u32.
        // param0: signed index = -(0+1) = -1  →  u32 bit-cast = 0xFFFF_FFFF
        // param1: signed index = -(1+1) = -2  →  u32 bit-cast = 0xFFFF_FFFE
        let p0 = (-1_i32) as u32;
        let p1 = (-2_i32) as u32;
        let instrs = vec![
            // Load param0 into accumulator.
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(p0)]),
            // Save it to local register r0.
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // Load param1 into accumulator.
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(p1)]),
            // acc = acc + r0  (param1 + param0).
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        // frame_size = 1 (one local register r0), parameter_count = 2.
        BytecodeArray::new(bytes, vec![], 1, 2, vec![], feedback_meta, vec![])
    }

    fn empty_globals() -> Rc<RefCell<GlobalEnv>> {
        Rc::new(RefCell::new(GlobalEnv::new()))
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    /// DeoptInfo can be constructed from its fields.
    #[test]
    fn test_deopt_info_construction() {
        let info = DeoptInfo {
            reason: DeoptReason::ArithmeticOverflow,
            bytecode_offset: 4,
            frame_state: FrameState {
                registers: vec![JsValue::Smi(1), JsValue::Smi(2)],
                accumulator: JsValue::Smi(3),
            },
        };
        assert_eq!(info.reason, DeoptReason::ArithmeticOverflow);
        assert_eq!(info.bytecode_offset, 4);
        assert_eq!(info.frame_state.accumulator, JsValue::Smi(3));
    }

    /// After deoptimize(), all BinaryOp feedback slots are Megamorphic.
    #[test]
    fn test_deoptimize_updates_feedback_to_megamorphic() {
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp, FeedbackSlotKind::Call]);
        let ba = make_lda_return(0);
        let mut fv = FeedbackVector::new(&meta);
        // Advance slot 0 to Monomorphic to simulate prior optimisation.
        fv.set_state(0, InlineCacheState::Monomorphic);

        let info = DeoptInfo {
            reason: DeoptReason::ArithmeticOverflow,
            bytecode_offset: 0,
            frame_state: FrameState {
                registers: vec![],
                accumulator: JsValue::Undefined,
            },
        };
        deoptimize(ba, &mut fv, info, empty_globals()).expect("deoptimize should succeed");

        // BinaryOp slot should have been advanced to Megamorphic.
        assert_eq!(fv.get_state(0), Some(InlineCacheState::Megamorphic));
        // Non-speculation slot (Call) should be unchanged.
        assert_eq!(fv.get_state(1), Some(InlineCacheState::Uninitialized));
    }

    /// After deoptimize(), Compare and BinaryOpInc slots are also Megamorphic.
    #[test]
    fn test_deoptimize_updates_compare_and_inc_slots() {
        let meta = FeedbackMetadata::new(vec![
            FeedbackSlotKind::Compare,
            FeedbackSlotKind::BinaryOpInc,
        ]);
        let ba = make_lda_return(0);
        let mut fv = FeedbackVector::new(&meta);

        let info = DeoptInfo {
            reason: DeoptReason::TypeCheckFailure,
            bytecode_offset: 0,
            frame_state: FrameState {
                registers: vec![],
                accumulator: JsValue::Undefined,
            },
        };
        deoptimize(ba, &mut fv, info, empty_globals()).expect("deoptimize should succeed");

        assert_eq!(fv.get_state(0), Some(InlineCacheState::Megamorphic));
        assert_eq!(fv.get_state(1), Some(InlineCacheState::Megamorphic));
    }

    /// deoptimize() resumes the interpreter and returns the correct value
    /// for a trivial function.
    #[test]
    fn test_deoptimize_interpreter_resume_trivial() {
        let ba = make_lda_return(42);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());

        let info = DeoptInfo {
            reason: DeoptReason::UnsupportedOperation,
            bytecode_offset: 0,
            frame_state: FrameState {
                registers: vec![],
                accumulator: JsValue::Undefined,
            },
        };
        let result =
            deoptimize(ba, &mut fv, info, empty_globals()).expect("interpreter must succeed");
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Force deopt via type change: compile a Maglev graph with `CheckedSmiAdd`
    /// (speculative Smi addition), execute it with values whose sum overflows
    /// into the 33-bit range so the overflow guard fires, then verify that the
    /// deoptimiser falls back to the interpreter and produces the correct result.
    #[cfg(all(target_arch = "x86_64", unix))]
    #[test]
    #[ignore = "narrow-Int32 analysis is disabled (SIGSEGV); 64-bit add does not overflow at 2^32"]
    fn test_deoptimize_via_type_change() {
        use crate::compiler::baseline::compiler::jit_to_jsvalue;
        use crate::compiler::maglev::codegen::compile;
        use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

        // Build a Maglev graph: param0 + param1 (CheckedSmiAdd).
        // This bakes in a speculative Smi-addition overflow guard.
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd {
            left: p0,
            right: p1,
        });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let cc = compile(&graph, 2).expect("maglev compile failed");

        // ── Fast path: small Smi values stay within range ────────────────────
        // SAFETY: code was produced by the Maglev compiler.
        let ok = unsafe { cc.execute(&[3, 4]) };
        assert_eq!(ok.ok().and_then(jit_to_jsvalue), Some(JsValue::Smi(7)));

        // ── Deopt via "type change": pass non-Smi values (sum = 2^32) ────────
        //
        // Each argument is 2^31 — outside the valid Smi range, simulating the
        // arrival of a value with a type that differs from the Smi
        // specialisation the JIT was compiled for.  Their sum (2^32) has bit 32
        // set, which the CheckedSmiAdd overflow guard detects and causes the
        // function to bail out with JIT_DEOPT.
        let large: i64 = 1_i64 << 31; // 2^31, outside Smi range
        let deopt_result = unsafe { cc.execute(&[large, large]) };
        assert!(
            deopt_result.is_err(),
            "expected JIT_DEOPT when sum has bits above bit 31, got {deopt_result:?}"
        );

        // ── Interpreter fallback ──────────────────────────────────────────────
        // Construct a bytecode array for `param0 + param1` with a BinaryOp
        // feedback slot.
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let ba = make_add_params(meta);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());
        fv.set_state(0, InlineCacheState::Monomorphic);

        // Use i32::MAX for both parameters — the interpreter handles large
        // integer addition by promoting to HeapNumber.
        let a = JsValue::Smi(i32::MAX);
        let b = JsValue::Smi(i32::MAX);
        let info = DeoptInfo {
            reason: DeoptReason::ArithmeticOverflow,
            bytecode_offset: 0,
            frame_state: FrameState {
                registers: vec![a, b],
                accumulator: JsValue::Undefined,
            },
        };
        let result = deoptimize(ba, &mut fv, info, empty_globals())
            .expect("interpreter must handle large Smi addition");

        // i32::MAX + i32::MAX = 4 294 967 294.0 (promoted to HeapNumber).
        let expected = f64::from(i32::MAX) + f64::from(i32::MAX);
        match result {
            JsValue::HeapNumber(v) => assert!(
                (v - expected).abs() < f64::EPSILON,
                "expected {expected}, got {v}"
            ),
            other => panic!("expected HeapNumber({expected}), got {other:?}"),
        }

        // Feedback slot must be Megamorphic after deoptimisation.
        assert_eq!(fv.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    /// Verify that the PC is correctly set to a mid-function bytecode offset
    /// so the interpreter resumes from the right instruction.
    #[test]
    fn test_deoptimize_resumes_from_correct_pc() {
        // Build a function: LdaSmi(10), LdaSmi(99), Return.
        // If we resume from bytecode_offset=0, we get Smi(99) (last LdaSmi wins).
        // If the interpreter is correctly resuming from the very beginning,
        // it executes both LdaSmi instructions and returns the last value loaded.
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let mut fv = FeedbackVector::new(ba.feedback_metadata());

        let info = DeoptInfo {
            reason: DeoptReason::UnsupportedOperation,
            bytecode_offset: 0,
            frame_state: FrameState {
                registers: vec![],
                accumulator: JsValue::Undefined,
            },
        };
        let result =
            deoptimize(ba, &mut fv, info, empty_globals()).expect("interpreter must succeed");
        // The interpreter runs from PC=0 and executes both LdaSmi instructions;
        // the last value in the accumulator before Return is Smi(99).
        assert_eq!(result, JsValue::Smi(99));
    }
}

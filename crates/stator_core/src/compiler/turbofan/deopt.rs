//! Turbofan deoptimiser: Cranelift JIT → interpreter fallback.
//!
//! When a Turbofan-compiled function encounters a speculation failure the
//! generated code writes the zero-based deopt-site index to an extra
//! register-file slot and returns
//! [`JIT_DEOPT`][crate::compiler::baseline::compiler::JIT_DEOPT].
//!
//! This module provides the machinery to:
//!
//! 1. Capture a raw register-file snapshot in a [`TurbofanFrameState`].
//! 2. Locate the matching [`DeoptPoint`][super::DeoptPoint] by its recorded
//!    index to recover the `bytecode_offset` and `reason`.
//! 3. Convert the raw `i64` register-file entries to [`JsValue`]s using
//!    [`jit_to_jsvalue`][crate::compiler::baseline::compiler::jit_to_jsvalue].
//! 4. Delegate to the Maglev deoptimiser
//!    ([`maglev::deopt::deoptimize`][crate::compiler::maglev::deopt::deoptimize])
//!    to reconstruct an [`InterpreterFrame`][crate::interpreter::InterpreterFrame]
//!    and resume bytecode execution.
//!
//! # Calling convention for deopt-index encoding
//!
//! The register file passed to a Turbofan-compiled function is allocated with
//! one *extra* trailing slot (at byte offset `register_file_slots × 8`).
//! Before branching to the deopt epilogue the JIT code stores the deopt-site
//! index into that extra slot so the caller can identify which check failed.
//!
//! # Example
//!
//! ```
//! use std::cell::RefCell;
//! use std::collections::HashMap;
//! use std::rc::Rc;
//!
//! use stator_core::bytecode::bytecode_array::BytecodeArray;
//! use stator_core::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
//! use stator_core::bytecode::feedback::{FeedbackMetadata, FeedbackVector};
//! use stator_core::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};
//! use stator_core::compiler::turbofan::compile;
//! use stator_core::compiler::turbofan::deopt::{TurbofanFrameState, deoptimize_turbofan};
//! use stator_core::objects::value::JsValue;
//!
//! // Build a trivial graph: return the Int32 constant 0.
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! let c = block.push_value(ValueNode::Int32Constant { value: 0 });
//! block.set_control(ControlNode::Deoptimize {
//!     bytecode_offset: 0,
//!     reason: 0,
//! });
//! graph.add_block(block);
//!
//! let compiled = compile(&graph, 0).expect("turbofan compile failed");
//!
//! // Bytecode: LdaSmi 0, Return.
//! let instrs = vec![
//!     Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(0)]),
//!     Instruction::new_unchecked(Opcode::Return, vec![]),
//! ];
//! let bytes = encode(&instrs);
//! let ba = BytecodeArray::new(
//!     bytes,
//!     vec![],
//!     0,
//!     0,
//!     vec![],
//!     FeedbackMetadata::empty(),
//!     vec![],
//! );
//! let mut fv = FeedbackVector::new(ba.feedback_metadata());
//!
//! // Execute and deoptimise.
//! // SAFETY: compiled code is produced by cranelift-jit from a well-formed graph.
//! let result = unsafe {
//!     compiled
//!         .execute_with_deopt(ba, &mut fv, Rc::new(RefCell::new(HashMap::new())))
//! }
//! .expect("interpreter fallback must succeed");
//! assert_eq!(result, JsValue::Smi(0));
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::bytecode::bytecode_array::BytecodeArray;
use crate::bytecode::feedback::FeedbackVector;
use crate::compiler::baseline::compiler::jit_to_jsvalue;
use crate::compiler::maglev::deopt::{DeoptInfo, DeoptReason, FrameState, deoptimize};
use crate::compiler::turbofan::DeoptPoint;
use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// TurbofanFrameState
// ─────────────────────────────────────────────────────────────────────────────

/// A raw snapshot of the register file at a Turbofan deoptimisation point.
///
/// The `registers` slice contains the `i64` values held in the register file
/// (parameters and any extra computed values) at the moment the deopt guard
/// fired.  It does **not** include the trailing deopt-index slot.
///
/// The caller (typically [`TurbofanCompiledCode::execute_with_deopt`]) is
/// responsible for splitting the register file into `registers` and
/// `deopt_index` before constructing this value.
#[derive(Debug, Clone)]
pub struct TurbofanFrameState {
    /// Raw `i64` register-file entries at the deopt point, one per parameter.
    pub registers: Vec<i64>,
    /// Zero-based index of the deopt site that fired.
    pub deopt_index: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// deoptimize_turbofan
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct an [`InterpreterFrame`][crate::interpreter::InterpreterFrame]
/// from a Turbofan deoptimisation and resume bytecode execution.
///
/// # Steps
///
/// 1. Locate the [`DeoptPoint`] whose `index` matches `frame.deopt_index`.
/// 2. Map the raw `i64` register values to [`JsValue`]s using
///    [`jit_to_jsvalue`].  Values that cannot be decoded become
///    [`JsValue::Undefined`].
/// 3. Convert the string `reason` field to a
///    [`DeoptReason`][crate::compiler::maglev::deopt::DeoptReason].
/// 4. Delegate to [`deoptimize`][crate::compiler::maglev::deopt::deoptimize]
///    to update the feedback vector, reconstruct the frame, and resume the
///    interpreter.
///
/// # Errors
///
/// Returns [`StatorError::Internal`] if no deopt point with a matching index
/// is found, or if the interpreter itself returns an error.
pub fn deoptimize_turbofan(
    bytecode_array: BytecodeArray,
    feedback: &mut FeedbackVector,
    deopt_points: &[DeoptPoint],
    frame: TurbofanFrameState,
    global_env: Rc<RefCell<HashMap<String, JsValue>>>,
) -> StatorResult<JsValue> {
    // ── Step 1: find the matching deopt point ────────────────────────────────
    let point = deopt_points
        .iter()
        .find(|p| p.index == frame.deopt_index)
        .ok_or_else(|| {
            StatorError::Internal(format!(
                "turbofan deopt: no deopt point with index {}",
                frame.deopt_index
            ))
        })?;

    // ── Step 2: map raw register values to JsValues ──────────────────────────
    let registers: Vec<JsValue> = frame
        .registers
        .iter()
        .map(|&r| jit_to_jsvalue(r).unwrap_or(JsValue::Undefined))
        .collect();

    // ── Step 3: map reason string to DeoptReason ─────────────────────────────
    let reason_lower = point.reason.to_ascii_lowercase();
    let reason = if reason_lower.contains("overflow") {
        DeoptReason::ArithmeticOverflow
    } else if reason_lower.contains("typecheck") || reason_lower.contains("type") {
        DeoptReason::TypeCheckFailure
    } else {
        DeoptReason::UnsupportedOperation
    };

    // ── Step 4: delegate to the Maglev deoptimiser ───────────────────────────
    let info = DeoptInfo {
        reason,
        bytecode_offset: point.bytecode_offset,
        frame_state: FrameState {
            registers,
            accumulator: JsValue::Undefined,
        },
    };

    deoptimize(bytecode_array, feedback, info, global_env)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::{
        FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
    };
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};
    use crate::compiler::turbofan::compile;

    fn empty_globals() -> Rc<RefCell<HashMap<String, JsValue>>> {
        Rc::new(RefCell::new(HashMap::new()))
    }

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

    // ── Unit tests ────────────────────────────────────────────────────────────

    /// `TurbofanFrameState` can be constructed and its fields are accessible.
    #[test]
    fn test_turbofan_frame_state_construction() {
        let fs = TurbofanFrameState {
            registers: vec![1, 2, 3],
            deopt_index: 0,
        };
        assert_eq!(fs.registers, vec![1, 2, 3]);
        assert_eq!(fs.deopt_index, 0);
    }

    /// `deoptimize_turbofan` returns an error when the deopt index is out of
    /// bounds.
    #[test]
    fn test_deoptimize_turbofan_missing_deopt_point() {
        let ba = make_lda_return(0);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());
        let frame = TurbofanFrameState {
            registers: vec![],
            deopt_index: 99, // no such deopt point
        };
        let result = deoptimize_turbofan(ba, &mut fv, &[], frame, empty_globals());
        assert!(
            result.is_err(),
            "expected error for missing deopt point, got {result:?}"
        );
    }

    /// `deoptimize_turbofan` resumes the interpreter for a trivial function.
    #[test]
    fn test_deoptimize_turbofan_trivial_resume() {
        let ba = make_lda_return(55);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());

        let deopt_points = vec![DeoptPoint {
            index: 0,
            bytecode_offset: 0,
            reason: "UnsupportedOperation",
        }];
        let frame = TurbofanFrameState {
            registers: vec![],
            deopt_index: 0,
        };
        let result =
            deoptimize_turbofan(ba, &mut fv, &deopt_points, frame, empty_globals()).unwrap();
        assert_eq!(result, JsValue::Smi(55));
    }

    /// Overflow deopt reason is mapped to `ArithmeticOverflow`.
    #[test]
    fn test_deoptimize_turbofan_overflow_reason() {
        let ba = make_lda_return(0);
        let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
        let mut fv = FeedbackVector::new(&meta);
        fv.set_state(0, InlineCacheState::Monomorphic);

        let deopt_points = vec![DeoptPoint {
            index: 0,
            bytecode_offset: 0,
            reason: "CheckedSmiAdd overflow",
        }];
        let frame = TurbofanFrameState {
            registers: vec![],
            deopt_index: 0,
        };
        deoptimize_turbofan(ba, &mut fv, &deopt_points, frame, empty_globals())
            .expect("should succeed");
        // BinaryOp slot should be Megamorphic after deopt.
        assert_eq!(fv.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    /// End-to-end: compile a graph with `ControlNode::Deoptimize`, execute it,
    /// verify the JIT triggers deopt, then resume in the interpreter.
    #[test]
    fn test_execute_with_deopt_interpreter_resume() {
        // Graph: always deoptimize at bytecode offset 0.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let _ = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Deoptimize {
            bytecode_offset: 0,
            reason: 0,
        });
        graph.add_block(block);

        let compiled = compile(&graph, 0).expect("turbofan compile failed");
        assert_eq!(compiled.deopt_points.len(), 1);

        // Bytecode: LdaSmi(77), Return.
        let ba = make_lda_return(77);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());

        // SAFETY: compiled code is produced by cranelift-jit from a well-formed
        // graph constructed in this unit test.
        let result = unsafe { compiled.execute_with_deopt(ba, &mut fv, empty_globals()) }
            .expect("interpreter fallback must succeed");
        assert_eq!(result, JsValue::Smi(77));
    }

    /// End-to-end: `CheckedSmiAdd` overflow triggers deopt; interpreter resumes
    /// and computes the correct result for large Smi values.
    #[test]
    fn test_execute_with_deopt_checked_smi_overflow() {
        // Graph: CheckedSmiAdd(SmiConstant(i32::MAX), SmiConstant(1)).
        // The overflow guard should fire at runtime.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: i32::MAX });
        let b = block.push_value(ValueNode::SmiConstant { value: 1 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        let compiled = compile(&graph, 0).expect("turbofan compile failed");

        // Bytecode: LdaSmi(42), Return — the interpreter fallback just returns 42.
        let ba = make_lda_return(42);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());

        // SAFETY: compiled code is produced by cranelift-jit from a well-formed graph.
        let result = unsafe { compiled.execute_with_deopt(ba, &mut fv, empty_globals()) }
            .expect("interpreter fallback must succeed");
        // The JIT overflowed and fell back to the interpreter which returns Smi(42).
        assert_eq!(result, JsValue::Smi(42));
    }
}

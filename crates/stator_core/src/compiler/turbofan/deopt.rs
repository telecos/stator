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
// ValueRecovery — how to recover a single JS value during deopt
// ─────────────────────────────────────────────────────────────────────────────

/// Describes how to recover a single value when reconstructing an interpreter
/// frame from a JIT deoptimisation.
///
/// Each entry in a [`DeoptEntry::register_map`] tells the deoptimiser where
/// to find the corresponding register value:
///
/// - **InRegister** — the value lives in a JIT register-file slot.
/// - **Constant** — the value is a compile-time constant.
/// - **Materialized** — the value must be reconstructed from scalar fields
///   (see scalar replacement / allocation sinking).
#[derive(Debug, Clone, PartialEq)]
pub enum ValueRecovery {
    /// Value lives in JIT register-file slot `index`.
    InRegister {
        /// Zero-based register-file slot index.
        index: u32,
    },
    /// Value is the compile-time constant `value`.
    Constant {
        /// The constant JS value.
        value: JsValue,
    },
    /// Value is a scalar-replaced object that must be materialized from its
    /// stored field scalars.
    Materialized {
        /// Map ID (hidden class / shape) for the reconstructed object.
        map: u32,
        /// `(field_offset, recovery)` pairs for each stored field.
        fields: Vec<(u32, Box<ValueRecovery>)>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// DeoptEntry — rich deopt metadata for speculative optimizations
// ─────────────────────────────────────────────────────────────────────────────

/// Rich deoptimisation metadata recorded for every speculative optimisation.
///
/// Unlike the simpler [`DeoptPoint`] (which records only an index, bytecode
/// offset, and a textual reason), `DeoptEntry` carries a full
/// [`register_map`](DeoptEntry::register_map) describing how to recover each
/// interpreter register from the JIT state.
///
/// A `DeoptEntry` is created for each speculative guard (type check, bounds
/// check, etc.) and stored alongside the compiled code.  When the guard
/// fires, the deoptimiser uses `register_map` to reconstruct the interpreter
/// frame.
#[derive(Debug, Clone)]
pub struct DeoptEntry {
    /// Byte offset in the original bytecode at which to resume interpretation.
    pub bytecode_offset: u32,
    /// Per-register recovery descriptors for frame reconstruction.
    pub register_map: Vec<ValueRecovery>,
    /// The reason the speculation failed.
    pub reason: DeoptReason,
}

// ─────────────────────────────────────────────────────────────────────────────
// DeoptKind — eager vs lazy deopt
// ─────────────────────────────────────────────────────────────────────────────

/// The mechanism by which a deoptimisation is triggered.
///
/// * **Eager** — a speculative type guard fires synchronously at a check
///   instruction (e.g. a Smi-add overflow or a `CheckMaps` failure).
/// * **Lazy** — a dependency is invalidated asynchronously (e.g. a map
///   transition on a prototype object) and the next re-entry into the
///   compiled code triggers the deopt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptKind {
    /// Synchronous guard failure (e.g. type-check, overflow).
    Eager,
    /// Asynchronous dependency invalidation.
    Lazy,
}

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
// Rich deopt with DeoptEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct an interpreter frame using a rich [`DeoptEntry`] and resume
/// bytecode execution.
///
/// This is the enhanced counterpart of [`deoptimize_turbofan`] that uses the
/// per-register [`ValueRecovery`] descriptors from a [`DeoptEntry`] instead
/// of blindly converting all register-file slots.
///
/// # Recovery procedure
///
/// For each entry in `entry.register_map`:
/// - [`ValueRecovery::InRegister`]: read the raw `i64` from `raw_regs` and
///   decode via [`jit_to_jsvalue`].
/// - [`ValueRecovery::Constant`]: use the constant directly.
/// - [`ValueRecovery::Materialized`]: recursively recover field values and
///   produce a fallback [`JsValue::Undefined`] (full materialisation requires
///   heap allocation which is deferred to the GC integration layer).
///
/// # Errors
///
/// Returns [`StatorError::Internal`] if the interpreter itself returns an
/// error.
pub fn deoptimize_with_entry(
    bytecode_array: BytecodeArray,
    feedback: &mut FeedbackVector,
    entry: &DeoptEntry,
    raw_regs: &[i64],
    global_env: Rc<RefCell<HashMap<String, JsValue>>>,
) -> StatorResult<JsValue> {
    let registers: Vec<JsValue> = entry
        .register_map
        .iter()
        .map(|recovery| recover_value(recovery, raw_regs))
        .collect();

    let info = DeoptInfo {
        reason: entry.reason,
        bytecode_offset: entry.bytecode_offset,
        frame_state: FrameState {
            registers,
            accumulator: JsValue::Undefined,
        },
    };

    deoptimize(bytecode_array, feedback, info, global_env)
}

/// Recover a single [`JsValue`] from a [`ValueRecovery`] descriptor.
fn recover_value(recovery: &ValueRecovery, raw_regs: &[i64]) -> JsValue {
    match recovery {
        ValueRecovery::InRegister { index } => {
            let idx = *index as usize;
            if idx < raw_regs.len() {
                jit_to_jsvalue(raw_regs[idx]).unwrap_or(JsValue::Undefined)
            } else {
                JsValue::Undefined
            }
        }
        ValueRecovery::Constant { value } => value.clone(),
        ValueRecovery::Materialized { .. } => {
            // Full heap materialisation is deferred to the GC integration
            // layer.  At this stage we return Undefined as a safe fallback.
            JsValue::Undefined
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame reconstruction helper
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct an interpreter [`FrameState`] from a raw JIT register file.
///
/// This is a low-level helper used by both [`deoptimize_turbofan`] and
/// [`deoptimize_with_entry`].  It converts a slice of raw `i64` register
/// values into a [`FrameState`] suitable for the Maglev deoptimiser.
pub fn reconstruct_frame(raw_regs: &[i64]) -> FrameState {
    let registers: Vec<JsValue> = raw_regs
        .iter()
        .map(|&r| jit_to_jsvalue(r).unwrap_or(JsValue::Undefined))
        .collect();
    FrameState {
        registers,
        accumulator: JsValue::Undefined,
    }
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

    // ── ValueRecovery / DeoptEntry tests ─────────────────────────────────────

    /// `ValueRecovery::InRegister` fetches the correct slot.
    #[test]
    fn test_value_recovery_in_register() {
        // jit_to_jsvalue treats the raw i64 as a direct Smi value.
        let raw_regs: Vec<i64> = vec![10, 20, 30];
        let recovery = ValueRecovery::InRegister { index: 1 };
        let val = recover_value(&recovery, &raw_regs);
        assert_eq!(val, JsValue::Smi(20));
    }

    /// `ValueRecovery::Constant` returns the constant value.
    #[test]
    fn test_value_recovery_constant() {
        let recovery = ValueRecovery::Constant {
            value: JsValue::Smi(99),
        };
        let val = recover_value(&recovery, &[]);
        assert_eq!(val, JsValue::Smi(99));
    }

    /// `ValueRecovery::Materialized` falls back to Undefined.
    #[test]
    fn test_value_recovery_materialized_fallback() {
        let recovery = ValueRecovery::Materialized {
            map: 0,
            fields: vec![],
        };
        let val = recover_value(&recovery, &[]);
        assert_eq!(val, JsValue::Undefined);
    }

    /// `ValueRecovery::InRegister` out of bounds returns Undefined.
    #[test]
    fn test_value_recovery_in_register_oob() {
        let recovery = ValueRecovery::InRegister { index: 100 };
        let val = recover_value(&recovery, &[1, 2]);
        assert_eq!(val, JsValue::Undefined);
    }

    /// `deoptimize_with_entry` resumes the interpreter via a DeoptEntry.
    #[test]
    fn test_deoptimize_with_entry_trivial() {
        let ba = make_lda_return(33);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());
        let entry = DeoptEntry {
            bytecode_offset: 0,
            register_map: vec![],
            reason: DeoptReason::UnsupportedOperation,
        };
        let result = deoptimize_with_entry(ba, &mut fv, &entry, &[], empty_globals()).unwrap();
        assert_eq!(result, JsValue::Smi(33));
    }

    /// `deoptimize_with_entry` with register recovery.
    #[test]
    fn test_deoptimize_with_entry_register_recovery() {
        let ba = make_lda_return(11);
        let mut fv = FeedbackVector::new(ba.feedback_metadata());
        let entry = DeoptEntry {
            bytecode_offset: 0,
            register_map: vec![
                ValueRecovery::Constant {
                    value: JsValue::Smi(5),
                },
                ValueRecovery::InRegister { index: 0 },
            ],
            reason: DeoptReason::TypeCheckFailure,
        };
        let result = deoptimize_with_entry(ba, &mut fv, &entry, &[20], empty_globals()).unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    /// `reconstruct_frame` produces a valid FrameState.
    #[test]
    fn test_reconstruct_frame_basic() {
        // jit_to_jsvalue treats raw i64 as direct Smi values.
        let frame = reconstruct_frame(&[10, 20]);
        assert_eq!(frame.registers.len(), 2);
        assert_eq!(frame.registers[0], JsValue::Smi(10));
        assert_eq!(frame.registers[1], JsValue::Smi(20));
        assert_eq!(frame.accumulator, JsValue::Undefined);
    }

    /// `reconstruct_frame` with empty register file.
    #[test]
    fn test_reconstruct_frame_empty() {
        let frame = reconstruct_frame(&[]);
        assert!(frame.registers.is_empty());
    }

    // ── DeoptKind tests ──────────────────────────────────────────────────────

    /// `DeoptKind` variants are distinct.
    #[test]
    fn test_deopt_kind_variants() {
        assert_ne!(DeoptKind::Eager, DeoptKind::Lazy);
        assert_eq!(DeoptKind::Eager, DeoptKind::Eager);
    }
}

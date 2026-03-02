//! ECMAScript §27 Iterator and Generator protocol support.
//!
//! This module provides:
//!
//! - [`SYMBOL_ITERATOR`] / [`SYMBOL_ASYNC_ITERATOR`] — opaque identifiers for
//!   the `@@iterator` and `@@asyncIterator` well-known symbols.
//! - [`IteratorRecord`] — the result of one iterator step (`{value, done}`).
//! - Helper constructors for built-in iterables: [`make_array_iterator`],
//!   [`make_string_iterator`], [`make_map_iterator`], [`make_set_iterator`].
//! - [`iterator_next`] — advance any [`JsValue`] iterator one step.
//! - [`iterator_to_vec`] — exhaust an iterator into a `Vec<JsValue>`.

use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::value::{GeneratorStep, JsValue, NativeIterator};

// ─────────────────────────────────────────────────────────────────────────────
// Well-known symbol IDs (§6.1.5.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for `Symbol.iterator` (the `@@iterator` well-known
/// symbol).
///
/// Used as the key when looking up the iterator factory method on an object.
pub const SYMBOL_ITERATOR: u64 = 1;

/// Opaque identifier for `Symbol.asyncIterator` (the `@@asyncIterator`
/// well-known symbol).
///
/// Used as the key when looking up the async iterator factory method on an
/// object.
pub const SYMBOL_ASYNC_ITERATOR: u64 = 2;

// ─────────────────────────────────────────────────────────────────────────────
// IteratorRecord (§7.4.1)
// ─────────────────────────────────────────────────────────────────────────────

/// The result of one step of an iterator (equivalent to the JS `{value, done}`
/// iterator-result object described in §27.1.2).
///
/// # Example
///
/// ```
/// use stator_core::builtins::iterator::{make_array_iterator, iterator_next};
/// use stator_core::objects::value::JsValue;
///
/// let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
/// let r = iterator_next(&iter).unwrap();
/// assert_eq!(r.value, JsValue::Smi(1));
/// assert!(!r.done);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IteratorRecord {
    /// The current iteration value.
    pub value: JsValue,
    /// `true` when the iterator is exhausted.
    pub done: bool,
}

impl IteratorRecord {
    /// Create a non-terminal iterator result.
    pub fn value(v: JsValue) -> Self {
        Self {
            value: v,
            done: false,
        }
    }

    /// Create a terminal iterator result with an optional return value.
    pub fn done(return_val: JsValue) -> Self {
        Self {
            value: return_val,
            done: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in iterator constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Create a [`JsValue::Iterator`] that yields the elements of `items` in
/// order (ECMAScript §23.1.5 `Array Iterator`).
///
/// # Example
///
/// ```
/// use stator_core::builtins::iterator::{make_array_iterator, iterator_next};
/// use stator_core::objects::value::JsValue;
///
/// let iter = make_array_iterator(vec![JsValue::Smi(10), JsValue::Smi(20)]);
/// assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(10));
/// assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(20));
/// assert!(iterator_next(&iter).unwrap().done);
/// ```
pub fn make_array_iterator(items: Vec<JsValue>) -> JsValue {
    JsValue::Iterator(NativeIterator::from_items(items))
}

/// Create a [`JsValue::Iterator`] that yields the individual Unicode scalar
/// values of `s` as single-character strings (ECMAScript §22.1.5 `String
/// Iterator`).
///
/// # Example
///
/// ```
/// use stator_core::builtins::iterator::{make_string_iterator, iterator_next};
/// use stator_core::objects::value::JsValue;
///
/// let iter = make_string_iterator("hi");
/// assert_eq!(
///     iterator_next(&iter).unwrap().value,
///     JsValue::String("h".to_string())
/// );
/// ```
pub fn make_string_iterator(s: &str) -> JsValue {
    JsValue::Iterator(NativeIterator::from_string(s))
}

/// Create a [`JsValue::Iterator`] that yields the `[key, value]` pairs from
/// `entries` in insertion order (ECMAScript §24.1.5 `Map Iterator`).
///
/// Each yielded value is a `JsValue::Array` containing `[key, value]`.
///
/// # Example
///
/// ```
/// use std::rc::Rc;
/// use stator_core::builtins::iterator::{make_map_iterator, iterator_next};
/// use stator_core::objects::value::JsValue;
///
/// let entries = vec![
///     (JsValue::String("a".into()), JsValue::Smi(1)),
/// ];
/// let iter = make_map_iterator(entries);
/// let result = iterator_next(&iter).unwrap();
/// assert_eq!(
///     result.value,
///     JsValue::Array(Rc::new(vec![JsValue::String("a".into()), JsValue::Smi(1)]))
/// );
/// ```
pub fn make_map_iterator(entries: Vec<(JsValue, JsValue)>) -> JsValue {
    let items = entries
        .into_iter()
        .map(|(k, v)| JsValue::Array(Rc::new(vec![k, v])))
        .collect();
    JsValue::Iterator(NativeIterator::from_items(items))
}

/// Create a [`JsValue::Iterator`] that yields the values of a `Set` in
/// insertion order (ECMAScript §24.2.5 `Set Iterator`).
///
/// # Example
///
/// ```
/// use stator_core::builtins::iterator::{make_set_iterator, iterator_next};
/// use stator_core::objects::value::JsValue;
///
/// let iter = make_set_iterator(vec![JsValue::Smi(3), JsValue::Smi(7)]);
/// assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(3));
/// assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(7));
/// assert!(iterator_next(&iter).unwrap().done);
/// ```
pub fn make_set_iterator(values: Vec<JsValue>) -> JsValue {
    JsValue::Iterator(NativeIterator::from_items(values))
}

// ─────────────────────────────────────────────────────────────────────────────
// iterator_next (§7.4.4)
// ─────────────────────────────────────────────────────────────────────────────

/// Advance an iterator one step, returning an [`IteratorRecord`].
///
/// Supported iterator types:
///
/// - [`JsValue::Iterator`] — advances the underlying [`NativeIterator`].
/// - [`JsValue::Generator`] — calls
///   [`Interpreter::run_generator_step`][crate::interpreter::Interpreter::run_generator_step]
///   with `undefined` as the sent value.
///
/// Returns a [`StatorError::TypeError`] for all other value types.
pub fn iterator_next(iter: &JsValue) -> StatorResult<IteratorRecord> {
    match iter {
        JsValue::Iterator(ni) => match ni.borrow_mut().next_item() {
            Some(v) => Ok(IteratorRecord::value(v)),
            None => Ok(IteratorRecord::done(JsValue::Undefined)),
        },
        JsValue::Generator(gs) => {
            use crate::interpreter::Interpreter;
            match Interpreter::run_generator_step(gs, JsValue::Undefined)? {
                GeneratorStep::Yield(v) => Ok(IteratorRecord::value(v)),
                GeneratorStep::Return(v) => Ok(IteratorRecord::done(v)),
            }
        }
        other => Err(StatorError::TypeError(format!(
            "iterator_next: value is not an iterator (got {other:?})"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// iterator_to_vec (convenience)
// ─────────────────────────────────────────────────────────────────────────────

/// Fully exhaust `iter`, collecting all yielded values into a `Vec<JsValue>`.
///
/// # Errors
///
/// Propagates any [`StatorResult`] error returned by the iterator.
pub fn iterator_to_vec(iter: &JsValue) -> StatorResult<Vec<JsValue>> {
    let mut out = Vec::new();
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        out.push(rec.value);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::BytecodeArray;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;
    use crate::objects::value::GeneratorState;

    // ── IteratorRecord ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_record_value() {
        let r = IteratorRecord::value(JsValue::Smi(42));
        assert_eq!(r.value, JsValue::Smi(42));
        assert!(!r.done);
    }

    #[test]
    fn test_iterator_record_done() {
        let r = IteratorRecord::done(JsValue::Undefined);
        assert_eq!(r.value, JsValue::Undefined);
        assert!(r.done);
    }

    // ── make_array_iterator ───────────────────────────────────────────────────

    #[test]
    fn test_array_iterator_yields_items_in_order() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert_eq!(
            iterator_next(&iter).unwrap(),
            IteratorRecord::value(JsValue::Smi(1))
        );
        assert_eq!(
            iterator_next(&iter).unwrap(),
            IteratorRecord::value(JsValue::Smi(2))
        );
        assert_eq!(
            iterator_next(&iter).unwrap(),
            IteratorRecord::value(JsValue::Smi(3))
        );
        assert!(iterator_next(&iter).unwrap().done);
    }

    #[test]
    fn test_array_iterator_empty() {
        let iter = make_array_iterator(vec![]);
        assert!(iterator_next(&iter).unwrap().done);
    }

    // ── make_string_iterator ──────────────────────────────────────────────────

    #[test]
    fn test_string_iterator_yields_chars() {
        let iter = make_string_iterator("ab");
        assert_eq!(
            iterator_next(&iter).unwrap().value,
            JsValue::String("a".to_string())
        );
        assert_eq!(
            iterator_next(&iter).unwrap().value,
            JsValue::String("b".to_string())
        );
        assert!(iterator_next(&iter).unwrap().done);
    }

    #[test]
    fn test_string_iterator_empty() {
        let iter = make_string_iterator("");
        assert!(iterator_next(&iter).unwrap().done);
    }

    // ── make_map_iterator ─────────────────────────────────────────────────────

    #[test]
    fn test_map_iterator_yields_key_value_pairs() {
        use std::rc::Rc;
        let iter = make_map_iterator(vec![
            (JsValue::String("x".into()), JsValue::Smi(10)),
            (JsValue::String("y".into()), JsValue::Smi(20)),
        ]);
        let r1 = iterator_next(&iter).unwrap();
        assert_eq!(
            r1.value,
            JsValue::Array(Rc::new(vec![JsValue::String("x".into()), JsValue::Smi(10)]))
        );
        assert!(!r1.done);
        let r2 = iterator_next(&iter).unwrap();
        assert_eq!(
            r2.value,
            JsValue::Array(Rc::new(vec![JsValue::String("y".into()), JsValue::Smi(20)]))
        );
        assert!(!r2.done);
        assert!(iterator_next(&iter).unwrap().done);
    }

    // ── make_set_iterator ─────────────────────────────────────────────────────

    #[test]
    fn test_set_iterator_yields_values() {
        let iter = make_set_iterator(vec![JsValue::Smi(5), JsValue::Smi(6)]);
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(5));
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(6));
        assert!(iterator_next(&iter).unwrap().done);
    }

    // ── iterator_to_vec ───────────────────────────────────────────────────────

    #[test]
    fn test_iterator_to_vec() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let v = iterator_to_vec(&iter).unwrap();
        assert_eq!(v, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    #[test]
    fn test_iterator_to_vec_empty() {
        let iter = make_array_iterator(vec![]);
        let v = iterator_to_vec(&iter).unwrap();
        assert!(v.is_empty());
    }

    // ── generator iterator ────────────────────────────────────────────────────

    /// Build a tiny generator function body:
    /// ```js
    /// function* () { yield 10; yield 20; }
    /// ```
    fn make_gen_yield_10_20() -> JsValue {
        // r0 = dummy generator-state register (ignored by interpreter)
        let gen_reg = Operand::Register(0);
        let instrs = vec![
            // SwitchOnGeneratorState — jump to resume point on re-entry
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![gen_reg]),
            // yield 10
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    gen_reg,
                    gen_reg,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![gen_reg, gen_reg, Operand::RegisterCount(0)],
            ),
            // yield 20
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(20)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    gen_reg,
                    gen_reg,
                    Operand::RegisterCount(0),
                    Operand::Immediate(1),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![gen_reg, gen_reg, Operand::RegisterCount(0)],
            ),
            // return undefined
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_generator_flag(true);
        JsValue::Generator(GeneratorState::new(ba))
    }

    #[test]
    fn test_generator_iterator_yields_sequence() {
        let generator = make_gen_yield_10_20();
        let r1 = iterator_next(&generator).unwrap();
        assert_eq!(r1.value, JsValue::Smi(10));
        assert!(!r1.done);

        let r2 = iterator_next(&generator).unwrap();
        assert_eq!(r2.value, JsValue::Smi(20));
        assert!(!r2.done);

        let r3 = iterator_next(&generator).unwrap();
        assert!(r3.done);
    }

    #[test]
    fn test_generator_iterator_to_vec() {
        let generator = make_gen_yield_10_20();
        let v = iterator_to_vec(&generator).unwrap();
        assert_eq!(v, vec![JsValue::Smi(10), JsValue::Smi(20)]);
    }

    #[test]
    fn test_iterator_next_type_error_on_non_iterator() {
        let err = iterator_next(&JsValue::Smi(42)).unwrap_err();
        assert!(matches!(err, crate::error::StatorError::TypeError(_)));
    }

    // ── symbol constants ──────────────────────────────────────────────────────

    #[test]
    fn test_symbol_iterator_is_symbol_value() {
        let sym = JsValue::Symbol(SYMBOL_ITERATOR);
        assert!(sym.is_symbol());
    }

    #[test]
    fn test_symbol_async_iterator_distinct_from_iterator() {
        assert_ne!(SYMBOL_ITERATOR, SYMBOL_ASYNC_ITERATOR);
    }

    // ── custom iterable (Rust-level simulation) ───────────────────────────────
    //
    // A "custom iterable" is any value whose @@iterator returns an iterator.
    // Here we simulate one by constructing the iterator directly (since we
    // don't yet have full object-property semantics).

    #[test]
    fn test_custom_iterable_simulation() {
        // Suppose we have a custom iterable that produces [100, 200, 300].
        let custom_iter = make_array_iterator(vec![
            JsValue::Smi(100),
            JsValue::Smi(200),
            JsValue::Smi(300),
        ]);
        let collected = iterator_to_vec(&custom_iter).unwrap();
        assert_eq!(
            collected,
            vec![JsValue::Smi(100), JsValue::Smi(200), JsValue::Smi(300)]
        );
    }
}

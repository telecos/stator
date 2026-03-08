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

use crate::builtins::promise::{MicrotaskQueue, promise_reject, promise_resolve};
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
/// if let JsValue::Array(arr) = &result.value {
///     assert_eq!(*arr.borrow(), vec![JsValue::String("a".into()), JsValue::Smi(1)]);
/// } else {
///     panic!("expected Array");
/// }
/// ```
pub fn make_map_iterator(entries: Vec<(JsValue, JsValue)>) -> JsValue {
    let items = entries
        .into_iter()
        .map(|(k, v)| JsValue::new_array(vec![k, v]))
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
// Private: call a JsValue callback
// ─────────────────────────────────────────────────────────────────────────────

/// Invoke `func` with the given arguments.
///
/// Returns [`StatorError::TypeError`] if `func` is not a callable
/// [`JsValue::NativeFunction`].
fn call_fn(func: &JsValue, args: Vec<JsValue>) -> StatorResult<JsValue> {
    match func {
        JsValue::NativeFunction(f) => f(args),
        _ => Err(StatorError::TypeError(
            "iterator helper: callback is not a function".into(),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterator Helpers — ES2025 (§27.1.4)
// ─────────────────────────────────────────────────────────────────────────────

/// ES2025 `Iterator.prototype.map(mapper)`.
///
/// Returns a new iterator that yields the result of applying `mapper` to each
/// element of `iter`.
///
/// # Errors
///
/// Propagates any error from the source iterator or the mapper callback.
pub fn iterator_map(iter: &JsValue, mapper: &JsValue) -> StatorResult<JsValue> {
    let mut out = Vec::new();
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        let mapped = call_fn(mapper, vec![rec.value, JsValue::HeapNumber(index as f64)])?;
        out.push(mapped);
        index += 1;
    }
    Ok(make_array_iterator(out))
}

/// ES2025 `Iterator.prototype.filter(predicate)`.
///
/// Returns a new iterator that yields only the elements of `iter` for which
/// `predicate` returns a truthy value.
///
/// # Errors
///
/// Propagates any error from the source iterator or the predicate callback.
pub fn iterator_filter(iter: &JsValue, predicate: &JsValue) -> StatorResult<JsValue> {
    let mut out = Vec::new();
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        let result = call_fn(
            predicate,
            vec![rec.value.clone(), JsValue::HeapNumber(index as f64)],
        )?;
        if result.to_boolean() {
            out.push(rec.value);
        }
        index += 1;
    }
    Ok(make_array_iterator(out))
}

/// ES2025 `Iterator.prototype.take(limit)`.
///
/// Returns a new iterator that yields at most `limit` elements from `iter`.
///
/// # Errors
///
/// Propagates any error from the source iterator.
pub fn iterator_take(iter: &JsValue, limit: usize) -> StatorResult<JsValue> {
    let mut out = Vec::new();
    for _ in 0..limit {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        out.push(rec.value);
    }
    Ok(make_array_iterator(out))
}

/// ES2025 `Iterator.prototype.drop(count)`.
///
/// Returns a new iterator that skips the first `count` elements of `iter`,
/// then yields the rest.
///
/// # Errors
///
/// Propagates any error from the source iterator.
pub fn iterator_drop(iter: &JsValue, count: usize) -> StatorResult<JsValue> {
    // Skip the first `count` items.
    for _ in 0..count {
        let rec = iterator_next(iter)?;
        if rec.done {
            return Ok(make_array_iterator(vec![]));
        }
    }
    // Collect the rest.
    let mut out = Vec::new();
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        out.push(rec.value);
    }
    Ok(make_array_iterator(out))
}

/// ES2025 `Iterator.prototype.flatMap(mapper)`.
///
/// Returns a new iterator that yields the concatenation of iterators (or
/// single values) produced by applying `mapper` to each element of `iter`.
/// If the mapper returns an iterator or array, its elements are flattened
/// one level deep.
///
/// # Errors
///
/// Propagates any error from the source iterator or the mapper callback.
pub fn iterator_flat_map(iter: &JsValue, mapper: &JsValue) -> StatorResult<JsValue> {
    let mut out = Vec::new();
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        let mapped = call_fn(mapper, vec![rec.value, JsValue::HeapNumber(index as f64)])?;
        match &mapped {
            JsValue::Iterator(_) | JsValue::Generator(_) => {
                let inner = iterator_to_vec(&mapped)?;
                out.extend(inner);
            }
            JsValue::Array(arr) => {
                out.extend(arr.borrow().iter().cloned());
            }
            _ => {
                out.push(mapped);
            }
        }
        index += 1;
    }
    Ok(make_array_iterator(out))
}

/// ES2025 `Iterator.prototype.reduce(reducer, initialValue)`.
///
/// Reduces the elements of `iter` to a single value by repeatedly calling
/// `reducer(accumulator, currentValue)`. If `initial` is `None`, the first
/// element is used as the initial accumulator.
///
/// # Errors
///
/// - [`StatorError::TypeError`] if `iter` is empty and no `initial` value is
///   provided.
/// - Propagates any error from the source iterator or the reducer callback.
pub fn iterator_reduce(
    iter: &JsValue,
    reducer: &JsValue,
    initial: Option<JsValue>,
) -> StatorResult<JsValue> {
    let mut acc = match initial {
        Some(v) => v,
        None => {
            let rec = iterator_next(iter)?;
            if rec.done {
                return Err(StatorError::TypeError(
                    "Reduce of empty iterator with no initial value".into(),
                ));
            }
            rec.value
        }
    };
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        acc = call_fn(reducer, vec![acc, rec.value])?;
    }
    Ok(acc)
}

/// ES2025 `Iterator.prototype.toArray()`.
///
/// Eagerly collects all remaining elements of `iter` into a
/// [`JsValue::Array`].
///
/// # Errors
///
/// Propagates any error from the source iterator.
pub fn iterator_to_array(iter: &JsValue) -> StatorResult<JsValue> {
    let items = iterator_to_vec(iter)?;
    Ok(JsValue::new_array(items))
}

/// ES2025 `Iterator.prototype.forEach(callback)`.
///
/// Calls `callback` for each element of `iter` and returns `undefined`.
///
/// # Errors
///
/// Propagates any error from the source iterator or the callback.
pub fn iterator_for_each(iter: &JsValue, callback: &JsValue) -> StatorResult<JsValue> {
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            break;
        }
        call_fn(callback, vec![rec.value, JsValue::HeapNumber(index as f64)])?;
        index += 1;
    }
    Ok(JsValue::Undefined)
}

/// ES2025 `Iterator.prototype.some(predicate)`.
///
/// Returns `true` if `predicate` returns a truthy value for any element of
/// `iter`, otherwise `false`.
///
/// # Errors
///
/// Propagates any error from the source iterator or the predicate callback.
pub fn iterator_some(iter: &JsValue, predicate: &JsValue) -> StatorResult<JsValue> {
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            return Ok(JsValue::Boolean(false));
        }
        let result = call_fn(
            predicate,
            vec![rec.value, JsValue::HeapNumber(index as f64)],
        )?;
        if result.to_boolean() {
            return Ok(JsValue::Boolean(true));
        }
        index += 1;
    }
}

/// ES2025 `Iterator.prototype.every(predicate)`.
///
/// Returns `true` if `predicate` returns a truthy value for every element of
/// `iter`, otherwise `false`.
///
/// # Errors
///
/// Propagates any error from the source iterator or the predicate callback.
pub fn iterator_every(iter: &JsValue, predicate: &JsValue) -> StatorResult<JsValue> {
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            return Ok(JsValue::Boolean(true));
        }
        let result = call_fn(
            predicate,
            vec![rec.value, JsValue::HeapNumber(index as f64)],
        )?;
        if !result.to_boolean() {
            return Ok(JsValue::Boolean(false));
        }
        index += 1;
    }
}

/// ES2025 `Iterator.prototype.find(predicate)`.
///
/// Returns the first element of `iter` for which `predicate` returns a truthy
/// value, or `undefined` if none match.
///
/// # Errors
///
/// Propagates any error from the source iterator or the predicate callback.
pub fn iterator_find(iter: &JsValue, predicate: &JsValue) -> StatorResult<JsValue> {
    let mut index = 0i64;
    loop {
        let rec = iterator_next(iter)?;
        if rec.done {
            return Ok(JsValue::Undefined);
        }
        let result = call_fn(
            predicate,
            vec![rec.value.clone(), JsValue::HeapNumber(index as f64)],
        )?;
        if result.to_boolean() {
            return Ok(rec.value);
        }
        index += 1;
    }
}

/// ES2025 `Iterator.from(iterable)`.
///
/// If `iterable` is already an iterator, returns it unchanged. If it is an
/// array, wraps it in an array iterator. Otherwise returns a
/// [`StatorError::TypeError`].
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] if `iterable` is not an iterator or array.
pub fn iterator_from(iterable: &JsValue) -> StatorResult<JsValue> {
    match iterable {
        JsValue::Iterator(_) | JsValue::Generator(_) => Ok(iterable.clone()),
        JsValue::Array(arr) => Ok(make_array_iterator(arr.borrow().clone())),
        JsValue::String(s) => Ok(make_string_iterator(s)),
        _ => Err(StatorError::TypeError(format!(
            "Iterator.from: value is not iterable (got {iterable:?})"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Async Iterator Helpers — ES2025 (§27.1.4.2)
//
// Each async helper mirrors the corresponding sync helper but wraps the
// result (or error) in a Promise.  Since the engine currently models async
// iteration through the same Iterator/Generator types, the helpers eagerly
// consume the source and settle the returned promise synchronously.
// ─────────────────────────────────────────────────────────────────────────────

/// Async version of [`iterator_map`].
///
/// Returns a `Promise` that fulfills with a new iterator whose elements are
/// the result of applying `mapper` to each element of `iter`.
pub fn async_iterator_map(iter: &JsValue, mapper: &JsValue, queue: &MicrotaskQueue) -> JsValue {
    match iterator_map(iter, mapper) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_filter`].
///
/// Returns a `Promise` that fulfills with a new iterator containing only the
/// elements for which `predicate` returns a truthy value.
pub fn async_iterator_filter(
    iter: &JsValue,
    predicate: &JsValue,
    queue: &MicrotaskQueue,
) -> JsValue {
    match iterator_filter(iter, predicate) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_take`].
///
/// Returns a `Promise` that fulfills with a new iterator yielding at most
/// `limit` elements.
pub fn async_iterator_take(iter: &JsValue, limit: usize, queue: &MicrotaskQueue) -> JsValue {
    match iterator_take(iter, limit) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_drop`].
///
/// Returns a `Promise` that fulfills with a new iterator that skips the first
/// `count` elements.
pub fn async_iterator_drop(iter: &JsValue, count: usize, queue: &MicrotaskQueue) -> JsValue {
    match iterator_drop(iter, count) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_flat_map`].
///
/// Returns a `Promise` that fulfills with a new iterator whose elements are
/// the concatenation of iterators produced by `mapper`.
pub fn async_iterator_flat_map(
    iter: &JsValue,
    mapper: &JsValue,
    queue: &MicrotaskQueue,
) -> JsValue {
    match iterator_flat_map(iter, mapper) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_reduce`].
///
/// Returns a `Promise` that fulfills with the reduced value.
pub fn async_iterator_reduce(
    iter: &JsValue,
    reducer: &JsValue,
    initial: Option<JsValue>,
    queue: &MicrotaskQueue,
) -> JsValue {
    match iterator_reduce(iter, reducer, initial) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_to_array`].
///
/// Returns a `Promise` that fulfills with a `JsValue::Array` of all elements.
pub fn async_iterator_to_array(iter: &JsValue, queue: &MicrotaskQueue) -> JsValue {
    match iterator_to_array(iter) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_for_each`].
///
/// Returns a `Promise` that fulfills with `undefined` after calling `callback`
/// for each element.
pub fn async_iterator_for_each(
    iter: &JsValue,
    callback: &JsValue,
    queue: &MicrotaskQueue,
) -> JsValue {
    match iterator_for_each(iter, callback) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_some`].
///
/// Returns a `Promise` that fulfills with `true` if `predicate` returns truthy
/// for any element, `false` otherwise.
pub fn async_iterator_some(iter: &JsValue, predicate: &JsValue, queue: &MicrotaskQueue) -> JsValue {
    match iterator_some(iter, predicate) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_every`].
///
/// Returns a `Promise` that fulfills with `true` if `predicate` returns truthy
/// for every element, `false` otherwise.
pub fn async_iterator_every(
    iter: &JsValue,
    predicate: &JsValue,
    queue: &MicrotaskQueue,
) -> JsValue {
    match iterator_every(iter, predicate) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_find`].
///
/// Returns a `Promise` that fulfills with the first matching element, or
/// `undefined` if none match.
pub fn async_iterator_find(iter: &JsValue, predicate: &JsValue, queue: &MicrotaskQueue) -> JsValue {
    match iterator_find(iter, predicate) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
}

/// Async version of [`iterator_from`].
///
/// Returns a `Promise` that fulfills with an iterator wrapping the given
/// iterable.
pub fn async_iterator_from(iterable: &JsValue, queue: &MicrotaskQueue) -> JsValue {
    match iterator_from(iterable) {
        Ok(v) => JsValue::Promise(promise_resolve(v, queue)),
        Err(e) => JsValue::Promise(promise_reject(JsValue::String(e.to_string()), queue)),
    }
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
    use std::rc::Rc;

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
        let iter = make_map_iterator(vec![
            (JsValue::String("x".into()), JsValue::Smi(10)),
            (JsValue::String("y".into()), JsValue::Smi(20)),
        ]);
        let r1 = iterator_next(&iter).unwrap();
        if let JsValue::Array(arr) = &r1.value {
            assert_eq!(
                *arr.borrow(),
                vec![JsValue::String("x".into()), JsValue::Smi(10)]
            );
        } else {
            panic!("expected Array");
        }
        assert!(!r1.done);
        let r2 = iterator_next(&iter).unwrap();
        if let JsValue::Array(arr) = &r2.value {
            assert_eq!(
                *arr.borrow(),
                vec![JsValue::String("y".into()), JsValue::Smi(20)]
            );
        } else {
            panic!("expected Array");
        }
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

    // ── Helper: make a native function for testing ────────────────────────────

    fn make_native_fn(f: impl Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static) -> JsValue {
        JsValue::NativeFunction(Rc::new(f))
    }

    // ── iterator_map ─────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_map_doubles_values() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let mapper = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Smi((n * 2.0) as i32))
        });
        let result = iterator_map(&iter, &mapper).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(
            items,
            vec![JsValue::Smi(2), JsValue::Smi(4), JsValue::Smi(6)]
        );
    }

    #[test]
    fn test_iterator_map_empty() {
        let iter = make_array_iterator(vec![]);
        let mapper = make_native_fn(|args| Ok(args[0].clone()));
        let result = iterator_map(&iter, &mapper).unwrap();
        assert!(iterator_to_vec(&result).unwrap().is_empty());
    }

    // ── iterator_filter ──────────────────────────────────────────────────────

    #[test]
    fn test_iterator_filter_even() {
        let iter = make_array_iterator(vec![
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n as i64 % 2 == 0))
        });
        let result = iterator_filter(&iter, &pred).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(2), JsValue::Smi(4)]);
    }

    #[test]
    fn test_iterator_filter_none_match() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(3)]);
        let pred = make_native_fn(|_| Ok(JsValue::Boolean(false)));
        let result = iterator_filter(&iter, &pred).unwrap();
        assert!(iterator_to_vec(&result).unwrap().is_empty());
    }

    // ── iterator_take ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_take() {
        let iter = make_array_iterator(vec![
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ]);
        let result = iterator_take(&iter, 2).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    #[test]
    fn test_iterator_take_more_than_available() {
        let iter = make_array_iterator(vec![JsValue::Smi(1)]);
        let result = iterator_take(&iter, 10).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(1)]);
    }

    #[test]
    fn test_iterator_take_zero() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let result = iterator_take(&iter, 0).unwrap();
        assert!(iterator_to_vec(&result).unwrap().is_empty());
    }

    // ── iterator_drop ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_drop() {
        let iter = make_array_iterator(vec![
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ]);
        let result = iterator_drop(&iter, 2).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(3), JsValue::Smi(4)]);
    }

    #[test]
    fn test_iterator_drop_all() {
        let iter = make_array_iterator(vec![JsValue::Smi(1)]);
        let result = iterator_drop(&iter, 5).unwrap();
        assert!(iterator_to_vec(&result).unwrap().is_empty());
    }

    #[test]
    fn test_iterator_drop_zero() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let result = iterator_drop(&iter, 0).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    // ── iterator_flat_map ────────────────────────────────────────────────────

    #[test]
    fn test_iterator_flat_map_with_arrays() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let mapper = make_native_fn(|args| {
            let n = args[0].to_number()? as i32;
            Ok(JsValue::new_array(vec![
                JsValue::Smi(n),
                JsValue::Smi(n * 10),
            ]))
        });
        let result = iterator_flat_map(&iter, &mapper).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(
            items,
            vec![
                JsValue::Smi(1),
                JsValue::Smi(10),
                JsValue::Smi(2),
                JsValue::Smi(20)
            ]
        );
    }

    #[test]
    fn test_iterator_flat_map_with_scalars() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let mapper = make_native_fn(|args| Ok(args[0].clone()));
        let result = iterator_flat_map(&iter, &mapper).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    // ── iterator_reduce ──────────────────────────────────────────────────────

    #[test]
    fn test_iterator_reduce_sum() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let reducer = make_native_fn(|args| {
            let a = args[0].to_number()?;
            let b = args[1].to_number()?;
            Ok(JsValue::Smi((a + b) as i32))
        });
        let result = iterator_reduce(&iter, &reducer, Some(JsValue::Smi(0))).unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn test_iterator_reduce_no_initial() {
        let iter = make_array_iterator(vec![JsValue::Smi(10), JsValue::Smi(20)]);
        let reducer = make_native_fn(|args| {
            let a = args[0].to_number()?;
            let b = args[1].to_number()?;
            Ok(JsValue::Smi((a + b) as i32))
        });
        let result = iterator_reduce(&iter, &reducer, None).unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    #[test]
    fn test_iterator_reduce_empty_no_initial_errors() {
        let iter = make_array_iterator(vec![]);
        let reducer = make_native_fn(|_| Ok(JsValue::Undefined));
        let err = iterator_reduce(&iter, &reducer, None).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── iterator_to_array ────────────────────────────────────────────────────

    #[test]
    fn test_iterator_to_array_returns_js_array() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let result = iterator_to_array(&iter).unwrap();
        if let JsValue::Array(arr) = result {
            assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(2)]);
        } else {
            panic!("expected Array");
        }
    }

    #[test]
    fn test_iterator_to_array_empty() {
        let iter = make_array_iterator(vec![]);
        let result = iterator_to_array(&iter).unwrap();
        if let JsValue::Array(arr) = result {
            assert!(arr.borrow().is_empty());
        } else {
            panic!("expected Array");
        }
    }

    // ── iterator_for_each ────────────────────────────────────────────────────

    #[test]
    fn test_iterator_for_each_calls_callback() {
        use std::cell::Cell;
        let count = Rc::new(Cell::new(0i32));
        let count_clone = count.clone();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let callback = make_native_fn(move |_| {
            count_clone.set(count_clone.get() + 1);
            Ok(JsValue::Undefined)
        });
        let result = iterator_for_each(&iter, &callback).unwrap();
        assert_eq!(result, JsValue::Undefined);
        assert_eq!(count.get(), 3);
    }

    // ── iterator_some ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_some_found() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 2.0))
        });
        assert_eq!(iterator_some(&iter, &pred).unwrap(), JsValue::Boolean(true));
    }

    #[test]
    fn test_iterator_some_not_found() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 2.0))
        });
        assert_eq!(
            iterator_some(&iter, &pred).unwrap(),
            JsValue::Boolean(false)
        );
    }

    #[test]
    fn test_iterator_some_empty() {
        let iter = make_array_iterator(vec![]);
        let pred = make_native_fn(|_| Ok(JsValue::Boolean(true)));
        assert_eq!(
            iterator_some(&iter, &pred).unwrap(),
            JsValue::Boolean(false)
        );
    }

    // ── iterator_every ───────────────────────────────────────────────────────

    #[test]
    fn test_iterator_every_all_pass() {
        let iter = make_array_iterator(vec![JsValue::Smi(2), JsValue::Smi(4), JsValue::Smi(6)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n as i64 % 2 == 0))
        });
        assert_eq!(
            iterator_every(&iter, &pred).unwrap(),
            JsValue::Boolean(true)
        );
    }

    #[test]
    fn test_iterator_every_one_fails() {
        let iter = make_array_iterator(vec![JsValue::Smi(2), JsValue::Smi(3), JsValue::Smi(4)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n as i64 % 2 == 0))
        });
        assert_eq!(
            iterator_every(&iter, &pred).unwrap(),
            JsValue::Boolean(false)
        );
    }

    #[test]
    fn test_iterator_every_empty() {
        let iter = make_array_iterator(vec![]);
        let pred = make_native_fn(|_| Ok(JsValue::Boolean(false)));
        assert_eq!(
            iterator_every(&iter, &pred).unwrap(),
            JsValue::Boolean(true)
        );
    }

    // ── iterator_find ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_find_found() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 2.0))
        });
        assert_eq!(iterator_find(&iter, &pred).unwrap(), JsValue::Smi(2));
    }

    #[test]
    fn test_iterator_find_not_found() {
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 99.0))
        });
        assert_eq!(iterator_find(&iter, &pred).unwrap(), JsValue::Undefined);
    }

    // ── iterator_from ────────────────────────────────────────────────────────

    #[test]
    fn test_iterator_from_array() {
        let arr = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let iter = iterator_from(&arr).unwrap();
        let items = iterator_to_vec(&iter).unwrap();
        assert_eq!(items, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    #[test]
    fn test_iterator_from_string() {
        let s = JsValue::String("ab".into());
        let iter = iterator_from(&s).unwrap();
        let items = iterator_to_vec(&iter).unwrap();
        assert_eq!(
            items,
            vec![JsValue::String("a".into()), JsValue::String("b".into())]
        );
    }

    #[test]
    fn test_iterator_from_iterator_passthrough() {
        let orig = make_array_iterator(vec![JsValue::Smi(5)]);
        let result = iterator_from(&orig).unwrap();
        let items = iterator_to_vec(&result).unwrap();
        assert_eq!(items, vec![JsValue::Smi(5)]);
    }

    #[test]
    fn test_iterator_from_non_iterable_errors() {
        let err = iterator_from(&JsValue::Smi(42)).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── async iterator helpers ───────────────────────────────────────────────

    fn make_queue() -> MicrotaskQueue {
        MicrotaskQueue::new()
    }

    #[test]
    fn test_async_iterator_map() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let mapper = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Smi((n * 2.0) as i32))
        });
        let result = async_iterator_map(&iter, &mapper, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(items, vec![JsValue::Smi(2), JsValue::Smi(4)]);
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_filter() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n as i64 % 2 == 0))
        });
        let result = async_iterator_filter(&iter, &pred, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(items, vec![JsValue::Smi(2)]);
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_take() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = async_iterator_take(&iter, 2, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(items, vec![JsValue::Smi(1), JsValue::Smi(2)]);
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_drop() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = async_iterator_drop(&iter, 1, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(items, vec![JsValue::Smi(2), JsValue::Smi(3)]);
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_flat_map() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let mapper = make_native_fn(|args| {
            let n = args[0].to_number()? as i32;
            Ok(JsValue::new_array(vec![
                JsValue::Smi(n),
                JsValue::Smi(n * 10),
            ]))
        });
        let result = async_iterator_flat_map(&iter, &mapper, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(
                items,
                vec![
                    JsValue::Smi(1),
                    JsValue::Smi(10),
                    JsValue::Smi(2),
                    JsValue::Smi(20)
                ]
            );
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_reduce() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let reducer = make_native_fn(|args| {
            let a = args[0].to_number()?;
            let b = args[1].to_number()?;
            Ok(JsValue::Smi((a + b) as i32))
        });
        let result = async_iterator_reduce(&iter, &reducer, Some(JsValue::Smi(0)), &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            assert_eq!(p.value(), Some(JsValue::Smi(6)));
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_reduce_error_rejects() {
        let q = make_queue();
        let iter = make_array_iterator(vec![]);
        let reducer = make_native_fn(|_| Ok(JsValue::Undefined));
        let result = async_iterator_reduce(&iter, &reducer, None, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_rejected());
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_to_array() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let result = async_iterator_to_array(&iter, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            if let Some(JsValue::Array(arr)) = p.value() {
                assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(2)]);
            } else {
                panic!("expected Array value in promise");
            }
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_for_each() {
        use std::cell::Cell;
        let q = make_queue();
        let count = Rc::new(Cell::new(0i32));
        let count_clone = count.clone();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let callback = make_native_fn(move |_| {
            count_clone.set(count_clone.get() + 1);
            Ok(JsValue::Undefined)
        });
        let result = async_iterator_for_each(&iter, &callback, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            assert_eq!(p.value(), Some(JsValue::Undefined));
        } else {
            panic!("expected Promise");
        }
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn test_async_iterator_some() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 2.0))
        });
        let result = async_iterator_some(&iter, &pred, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            assert_eq!(p.value(), Some(JsValue::Boolean(true)));
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_every() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(2), JsValue::Smi(4)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n as i64 % 2 == 0))
        });
        let result = async_iterator_every(&iter, &pred, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            assert_eq!(p.value(), Some(JsValue::Boolean(true)));
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_find() {
        let q = make_queue();
        let iter = make_array_iterator(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let pred = make_native_fn(|args| {
            let n = args[0].to_number()?;
            Ok(JsValue::Boolean(n == 2.0))
        });
        let result = async_iterator_find(&iter, &pred, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            assert_eq!(p.value(), Some(JsValue::Smi(2)));
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_from_array() {
        let q = make_queue();
        let arr = JsValue::new_array(vec![JsValue::Smi(1)]);
        let result = async_iterator_from(&arr, &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_fulfilled());
            let inner = p.value().unwrap();
            let items = iterator_to_vec(&inner).unwrap();
            assert_eq!(items, vec![JsValue::Smi(1)]);
        } else {
            panic!("expected Promise");
        }
    }

    #[test]
    fn test_async_iterator_from_non_iterable_rejects() {
        let q = make_queue();
        let result = async_iterator_from(&JsValue::Smi(42), &q);
        if let JsValue::Promise(p) = result {
            assert!(p.is_rejected());
        } else {
            panic!("expected Promise");
        }
    }
}

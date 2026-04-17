//! ECMAScript §23.1 `Array` built-in static methods and prototype equivalents.
//!
//! Every function in this module is a direct Rust equivalent of either a static
//! property of the JavaScript `Array` constructor or a method on
//! `Array.prototype`.  They operate on [`JsArray`] and [`JsValue`] values and
//! have no side-effects beyond the arrays passed in.
//!
//! # Naming convention
//!
//! Each function is prefixed `array_` to avoid ambiguity with similarly-named
//! standard-library items (e.g. `array_map` vs `Iterator::map`).
//!
//! # Callback conventions
//!
//! Methods that accept callbacks in JavaScript (e.g. `map`, `filter`) take Rust
//! closures instead.  The signature mirrors the ECMAScript callback prototype
//! `(element, index)` — the third `array` argument is omitted because the
//! caller already has a reference to it.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §23.1 — *The Array Constructor*

use crate::error::{StatorError, StatorResult};
use crate::objects::js_array::{ElementKind, JsArray};
use crate::objects::js_object::JsObject;
use crate::objects::map::InstanceType;
use crate::objects::value::JsValue;

// ── Array.isArray ─────────────────────────────────────────────────────────────

/// ECMAScript §23.1.2.2 `Array.isArray(obj)`.
///
/// Returns `true` if `obj`'s hidden class carries [`InstanceType::JsArray`],
/// which is the marker set by [`JsArray::new`].
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::array_is_array;
/// use stator_js::objects::js_array::JsArray;
///
/// let arr = JsArray::new();
/// assert!(array_is_array(arr.as_object()));
///
/// use stator_js::objects::js_object::JsObject;
/// let obj = JsObject::new();
/// assert!(!array_is_array(&obj));
/// ```
pub fn array_is_array(obj: &JsObject) -> bool {
    obj.map().instance_type() == InstanceType::JsArray
}

// ── Array.from ────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.2.1 `Array.from(iterable)`.
///
/// Creates a new [`JsArray`] from any Rust iterable of [`JsValue`]s.
/// Element kinds are widened as each element is pushed (see [`JsArray::push`]).
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::array_from;
/// use stator_js::objects::value::JsValue;
///
/// let arr = array_from([JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
/// assert_eq!(arr.length(), 3);
/// assert_eq!(arr.get(1), JsValue::Smi(2));
/// ```
pub fn array_from(items: impl IntoIterator<Item = JsValue>) -> JsArray {
    let mut arr = JsArray::new();
    for v in items {
        arr.push(v);
    }
    arr
}

// ── Array.from (with mapFn) ──────────────────────────────────────────────────

/// ECMAScript §23.1.2.1 `Array.from(iterable, mapFn)`.
///
/// Creates a new [`JsArray`] from any Rust iterable of [`JsValue`]s, applying
/// `map_fn(element, index)` to each element before inserting it.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::array_from_with_map_fn;
/// use stator_js::objects::value::JsValue;
///
/// let arr = array_from_with_map_fn(
///     [JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
///     |v, _i| if let JsValue::Smi(n) = v { JsValue::Smi(n * 2) } else { v },
/// );
/// assert_eq!(arr.length(), 3);
/// assert_eq!(arr.get(0), JsValue::Smi(2));
/// assert_eq!(arr.get(2), JsValue::Smi(6));
/// ```
pub fn array_from_with_map_fn(
    items: impl IntoIterator<Item = JsValue>,
    mut map_fn: impl FnMut(JsValue, u32) -> JsValue,
) -> JsArray {
    let mut arr = JsArray::new();
    for (i, v) in items.into_iter().enumerate() {
        arr.push(map_fn(v, i as u32));
    }
    arr
}

// ── Array.fromAsync ──────────────────────────────────────────────────────────

/// ECMAScript §23.1.2.1.1 `Array.fromAsync(asyncIterable, mapFn?)`.
///
/// Creates a new [`JsArray`] from an iterable (sync or async) and wraps the
/// result in a resolved [`JsPromise`].  When an optional `map_fn` is provided,
/// each element is mapped through `map_fn(element, index)` before insertion.
///
/// This is a simplified synchronous-collection implementation that handles the
/// common case where the iterable yields items synchronously.  True async
/// iterables that depend on I/O scheduling would need the full event-loop;
/// this version collects eagerly and resolves immediately.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::array_from_async;
/// use stator_js::builtins::promise::{MicrotaskQueue, PromiseState};
/// use stator_js::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = array_from_async(
///     vec![JsValue::Smi(1), JsValue::Smi(2)],
///     None::<fn(JsValue, u32) -> JsValue>,
///     &queue,
/// );
/// queue.drain();
/// if let PromiseState::Fulfilled(JsValue::Array(arr)) = p.state() {
///     assert_eq!(arr.borrow().len(), 2);
/// } else {
///     panic!("expected fulfilled promise with array");
/// }
/// ```
pub fn array_from_async(
    items: impl IntoIterator<Item = JsValue>,
    map_fn: Option<impl FnMut(JsValue, u32) -> JsValue>,
    queue: &crate::builtins::promise::MicrotaskQueue,
) -> crate::builtins::promise::JsPromise {
    let result = match map_fn {
        Some(mf) => array_from_with_map_fn(items, mf),
        None => array_from(items),
    };
    let arr_value = JsValue::new_array(crate::builtins::array::array_values(&result));
    crate::builtins::promise::promise_resolve(arr_value, queue)
}

// ── Array.of ──────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.2.3 `Array.of(...items)`.
///
/// Creates a new [`JsArray`] from a slice of [`JsValue`]s.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::array_of;
/// use stator_js::objects::value::JsValue;
///
/// let arr = array_of(&[JsValue::Smi(7), JsValue::Boolean(true)]);
/// assert_eq!(arr.length(), 2);
/// assert_eq!(arr.get(0), JsValue::Smi(7));
/// ```
pub fn array_of(items: &[JsValue]) -> JsArray {
    array_from(items.iter().cloned())
}

// ── push ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.20 `Array.prototype.push(...items)`.
///
/// Appends every element of `values` to the end of `arr` and returns the new
/// `length`.
pub fn array_push(arr: &mut JsArray, values: &[JsValue]) -> u32 {
    for v in values {
        arr.push(v.clone());
    }
    arr.length()
}

// ── pop ───────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.19 `Array.prototype.pop()`.
///
/// Removes and returns the last element.  Returns [`JsValue::Undefined`] for
/// an empty array.
pub fn array_pop(arr: &mut JsArray) -> JsValue {
    arr.pop()
}

// ── shift ─────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.27 `Array.prototype.shift()`.
///
/// Removes and returns the first element, shifting all remaining elements down
/// by one index.  Returns [`JsValue::Undefined`] for an empty array.
pub fn array_shift(arr: &mut JsArray) -> JsValue {
    let len = arr.length();
    if len == 0 {
        return JsValue::Undefined;
    }
    let first = arr.get(0);
    for i in 1..len {
        let v = arr.get(i);
        arr.set(i - 1, v);
    }
    arr.as_object_mut().truncate_elements((len - 1) as usize);
    first
}

// ── unshift ───────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.32 `Array.prototype.unshift(...items)`.
///
/// Inserts all elements of `values` at the beginning of `arr` (in the order
/// they appear in `values`), shifting existing elements up, and returns the new
/// `length`.
pub fn array_unshift(arr: &mut JsArray, values: &[JsValue]) -> u32 {
    if values.is_empty() {
        return arr.length();
    }
    let count = values.len() as u32;
    let old_len = arr.length();
    // Extend the array to make room, then shift existing elements up.
    for i in (0..old_len).rev() {
        let v = arr.get(i);
        arr.set(i + count, v);
    }
    // Write the new elements at the front.
    for (i, v) in values.iter().enumerate() {
        arr.set(i as u32, v.clone());
    }
    arr.length()
}

// ── splice ────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.30 `Array.prototype.splice(start, deleteCount?, ...items)`.
///
/// Removes `delete_count` elements starting at `start` (clamped to `[0, len]`)
/// and inserts `items` in their place.  Returns a new [`JsArray`] containing
/// the deleted elements.
///
/// Negative `start` is interpreted as an offset from the end.
pub fn array_splice(
    arr: &mut JsArray,
    start: i64,
    delete_count: Option<u32>,
    items: &[JsValue],
) -> JsArray {
    let len = arr.length() as i64;
    // Clamp start to [0, len].
    let actual_start = if start < 0 {
        (len + start).max(0) as u32
    } else {
        (start.min(len)) as u32
    };
    let max_delete = (len - actual_start as i64).max(0) as u32;
    let actual_delete = delete_count.unwrap_or(max_delete).min(max_delete);

    // Collect the deleted elements.
    let mut deleted = JsArray::new();
    for i in 0..actual_delete {
        deleted.push(arr.get(actual_start + i));
    }

    // Build the new element list.
    let old_len = arr.length();
    let mut new_elements: Vec<JsValue> =
        Vec::with_capacity((old_len - actual_delete + items.len() as u32) as usize);
    for i in 0..actual_start {
        new_elements.push(arr.get(i));
    }
    for item in items {
        new_elements.push(item.clone());
    }
    for i in (actual_start + actual_delete)..old_len {
        new_elements.push(arr.get(i));
    }

    // Rebuild arr in place.
    arr.as_object_mut().truncate_elements(0);
    for v in new_elements {
        arr.push(v);
    }

    deleted
}

// ── map ───────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.15 `Array.prototype.map(callbackFn)`.
///
/// Calls `f(element, index)` for each element and returns a new [`JsArray`]
/// containing the results.  Holes (sparse slots) remain as
/// [`JsValue::Undefined`].
pub fn array_map(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> JsValue) -> JsArray {
    let len = arr.length();
    let mut result = JsArray::new();
    for i in 0..len {
        let v = arr.get(i);
        result.push(f(&v, i));
    }
    result
}

// ── filter ────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.7 `Array.prototype.filter(callbackFn)`.
///
/// Returns a new [`JsArray`] containing only the elements for which
/// `f(element, index)` returns `true`.
pub fn array_filter(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> JsArray {
    let len = arr.length();
    let mut result = JsArray::new();
    for i in 0..len {
        let v = arr.get(i);
        if f(&v, i) {
            result.push(v);
        }
    }
    result
}

// ── reduce ────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.22 `Array.prototype.reduce(callbackFn, initialValue?)`.
///
/// Calls `f(accumulator, element, index)` for each element and returns the
/// final accumulator.
///
/// Returns [`StatorError::TypeError`] if the array is empty and no
/// `initial` value is provided.
pub fn array_reduce(
    arr: &JsArray,
    mut f: impl FnMut(JsValue, &JsValue, u32) -> JsValue,
    initial: Option<JsValue>,
) -> StatorResult<JsValue> {
    let len = arr.length();
    let (mut acc, start) = if let Some(init) = initial {
        (init, 0u32)
    } else {
        if len == 0 {
            return Err(StatorError::TypeError(
                "Reduce of empty array with no initial value".to_string(),
            ));
        }
        (arr.get(0), 1u32)
    };
    for i in start..len {
        let v = arr.get(i);
        acc = f(acc, &v, i);
    }
    Ok(acc)
}

// ── forEach ───────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.12 `Array.prototype.forEach(callbackFn)`.
///
/// Calls `f(element, index)` for each element.  Sparse slots are visited with
/// [`JsValue::Undefined`].
pub fn array_for_each(arr: &JsArray, mut f: impl FnMut(&JsValue, u32)) {
    for i in 0..arr.length() {
        let v = arr.get(i);
        f(&v, i);
    }
}

// ── find ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.9 `Array.prototype.find(callbackFn)`.
///
/// Returns the first element for which `f(element, index)` is `true`, or
/// [`JsValue::Undefined`] if none is found.
pub fn array_find(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> JsValue {
    for i in 0..arr.length() {
        let v = arr.get(i);
        if f(&v, i) {
            return v;
        }
    }
    JsValue::Undefined
}

// ── findIndex ─────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.10 `Array.prototype.findIndex(callbackFn)`.
///
/// Returns `Some(index)` of the first element for which `f(element, index)` is
/// `true`, or `None` if none is found.
pub fn array_find_index(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> Option<u32> {
    for i in 0..arr.length() {
        let v = arr.get(i);
        if f(&v, i) {
            return Some(i);
        }
    }
    None
}

// ── some ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.28 `Array.prototype.some(callbackFn)`.
///
/// Returns `true` if `f(element, index)` returns `true` for at least one
/// element.
pub fn array_some(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> bool {
    for i in 0..arr.length() {
        let v = arr.get(i);
        if f(&v, i) {
            return true;
        }
    }
    false
}

// ── every ─────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.5 `Array.prototype.every(callbackFn)`.
///
/// Returns `true` if `f(element, index)` returns `true` for every element.
/// Vacuously `true` for empty arrays.
pub fn array_every(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> bool {
    for i in 0..arr.length() {
        let v = arr.get(i);
        if !f(&v, i) {
            return false;
        }
    }
    true
}

// ── includes ──────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.13 `Array.prototype.includes(searchElement, fromIndex?)`.
///
/// Returns `true` if `value` is in the array at or after index `from_index`.
/// Uses SameValueZero semantics (`NaN === NaN`, `+0 === -0`).
///
/// Negative `from_index` is interpreted as an offset from the end (clamped to
/// `0` if the result would be negative).
pub fn array_includes(arr: &JsArray, value: &JsValue, from_index: Option<i64>) -> bool {
    let len = arr.length();
    let start = resolve_relative_index(from_index.unwrap_or(0), len);
    // Fast path: when the array holds only Smis and the needle is a Smi, we
    // can compare raw i32 values directly without dynamic dispatch.
    if matches!(
        arr.element_kind(),
        ElementKind::PackedSmi | ElementKind::HoleSmi
    ) && let JsValue::Smi(needle) = value
    {
        let slice = arr.elements_as_slice();
        for v in &slice[start as usize..] {
            if let JsValue::Smi(s) = v
                && *s == *needle
            {
                return true;
            }
        }
        return false;
    }
    for i in start..len {
        if same_value_zero(arr.get(i), value) {
            return true;
        }
    }
    false
}

// ── indexOf ───────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.14 `Array.prototype.indexOf(searchElement, fromIndex?)`.
///
/// Returns `Some(index)` of the first element strictly equal to `value` at or
/// after `from_index`, using strict equality (`===`; `NaN !== NaN`).
///
/// Negative `from_index` is interpreted as an offset from the end.
pub fn array_index_of(arr: &JsArray, value: &JsValue, from_index: Option<i64>) -> Option<u32> {
    let len = arr.length();
    let start = resolve_relative_index(from_index.unwrap_or(0), len);
    // Fast path: Smi-only arrays with a Smi needle.
    if matches!(
        arr.element_kind(),
        ElementKind::PackedSmi | ElementKind::HoleSmi
    ) && let JsValue::Smi(needle) = value
    {
        let slice = arr.elements_as_slice();
        for (i, v) in slice[start as usize..].iter().enumerate() {
            if let JsValue::Smi(s) = v
                && *s == *needle
            {
                return Some(start + i as u32);
            }
        }
        return None;
    }
    (start..len).find(|&i| strict_equal(&arr.get(i), value))
}

// ── concat ────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.1 `Array.prototype.concat(...values)`.
///
/// Returns a new [`JsArray`] consisting of the elements of `arr` followed by
/// the elements of each array in `others`.
pub fn array_concat(arr: &JsArray, others: &[&JsArray]) -> JsArray {
    let mut result = JsArray::new();
    for i in 0..arr.length() {
        result.push(arr.get(i));
    }
    for other in others {
        for i in 0..other.length() {
            result.push(other.get(i));
        }
    }
    result
}

// ── slice ─────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.25 `Array.prototype.slice(start?, end?)`.
///
/// Returns a shallow copy of the portion of `arr` between `start` (inclusive)
/// and `end` (exclusive).  Both are optional; omitting `start` defaults to `0`
/// and omitting `end` defaults to `length`.
///
/// Negative indices are interpreted as offsets from the end.
pub fn array_slice(arr: &JsArray, start: Option<i64>, end: Option<i64>) -> JsArray {
    let len = arr.length();
    let begin = resolve_relative_index(start.unwrap_or(0), len);
    let finish = end.map(|e| resolve_relative_index(e, len)).unwrap_or(len);
    let mut result = JsArray::new();
    for i in begin..finish {
        result.push(arr.get(i));
    }
    result
}

// ── join ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.16 `Array.prototype.join(separator?)`.
///
/// Returns a string created by concatenating all elements with `separator`
/// between them.  `None` separator defaults to `","`.
///
/// Returns [`StatorError::TypeError`] if any element cannot be converted to a
/// string (e.g. `Symbol`).  `undefined` and `null` elements are treated as the
/// empty string, per the spec.
pub fn array_join(arr: &JsArray, separator: Option<&str>) -> StatorResult<String> {
    let sep = separator.unwrap_or(",");
    let len = arr.length();
    if len == 0 {
        return Ok(String::new());
    }
    // Fast path: Smi-only arrays can format integers directly.
    if matches!(
        arr.element_kind(),
        ElementKind::PackedSmi | ElementKind::HoleSmi
    ) {
        let mut out = String::with_capacity(len as usize * 4);
        let slice = arr.elements_as_slice();
        for (i, v) in slice.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            match v {
                JsValue::Smi(n) => {
                    use std::fmt::Write;
                    let _ = write!(out, "{n}");
                }
                JsValue::Undefined => {}
                other => out.push_str(&other.to_js_string()?),
            }
        }
        return Ok(out);
    }
    let mut out = String::with_capacity(len as usize * 8);
    for i in 0..len {
        if i > 0 {
            out.push_str(sep);
        }
        let v = arr.get(i);
        match &v {
            JsValue::Undefined | JsValue::Null => {}
            other => out.push_str(&other.to_js_string()?),
        }
    }
    Ok(out)
}

// ── reverse ───────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.24 `Array.prototype.reverse()`.
///
/// Reverses the elements of `arr` **in place** and returns a reference to the
/// same array.
pub fn array_reverse(arr: &mut JsArray) {
    let len = arr.length();
    if len < 2 {
        return;
    }
    // Fast path: reverse the backing slice directly instead of going through
    // get/set (which re-widen element kinds on each write).
    arr.elements_as_mut_slice().reverse();
}

// ── sort ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.29 `Array.prototype.sort(compareFn?)`.
///
/// Sorts the elements of `arr` **in place**.
///
/// If `comparator` is `Some`, it is used to compare elements.  If `None`,
/// elements are compared as strings (ECMAScript default sort order).
///
/// Returns [`StatorError::TypeError`] if the default string conversion fails
/// (e.g. an element is a `Symbol`).
pub fn array_sort(
    arr: &mut JsArray,
    comparator: Option<impl FnMut(&JsValue, &JsValue) -> std::cmp::Ordering>,
) -> StatorResult<()> {
    let len = arr.length() as usize;
    if len < 2 {
        return Ok(());
    }

    if let Some(mut cmp) = comparator {
        // User-supplied comparator: snapshot elements, sort, write back.
        let mut elems: Vec<JsValue> = (0..len as u32).map(|i| arr.get(i)).collect();
        elems.sort_by(|a, b| cmp(a, b));
        arr.as_object_mut().truncate_elements(0);
        for v in elems {
            arr.push(v);
        }
    } else if matches!(
        arr.element_kind(),
        ElementKind::PackedSmi | ElementKind::HoleSmi
    ) {
        // Fast path: Smi-only arrays can be sorted numerically in-place
        // without string conversion.
        let slice = arr.elements_as_mut_slice();
        slice.sort_by(|a, b| {
            let ai = if let JsValue::Smi(v) = a { *v } else { 0 };
            let bi = if let JsValue::Smi(v) = b { *v } else { 0 };
            // Default ECMAScript sort is string-based; for Smis this means
            // comparing their string representations lexicographically.
            let sa = ai.to_string();
            let sb = bi.to_string();
            sa.cmp(&sb)
        });
    } else {
        // Default: sort as strings.
        let mut elems: Vec<JsValue> = (0..len as u32).map(|i| arr.get(i)).collect();
        let mut strs: Vec<String> = Vec::with_capacity(len);
        for v in &elems {
            strs.push(v.to_js_string()?);
        }
        let mut paired: Vec<(JsValue, String)> = elems.into_iter().zip(strs).collect();
        paired.sort_by(|(_, a), (_, b)| a.cmp(b));
        elems = paired.into_iter().map(|(v, _)| v).collect();
        arr.as_object_mut().truncate_elements(0);
        for v in elems {
            arr.push(v);
        }
    }
    Ok(())
}

// ── flat ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.11 `Array.prototype.flat(depth?)`.
///
/// Returns a new [`JsArray`] with sub-arrays flattened up to `depth` levels.
///
/// Because [`JsValue`] has no built-in array variant, callers must supply an
/// `expand` callback that returns `Some(JsArray)` when a value should be
/// flattened, or `None` to leave it as a leaf.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::array::{array_flat, array_of};
/// use stator_js::objects::value::JsValue;
///
/// // Build outer = [1, [2, 3], 4].  The inner array is kept as a pre-built
/// // JsArray; the expand callback recognises it by a sentinel string value.
/// let mut outer = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(4)]);
/// // Without an expand function that matches any element, nothing is flattened.
/// let flat = array_flat(&outer, 1, |_| None);
/// assert_eq!(flat.length(), 3);
/// ```
pub fn array_flat<F>(arr: &JsArray, depth: u32, expand: F) -> JsArray
where
    F: Fn(&JsValue) -> Option<JsArray>,
{
    let mut result = JsArray::new();
    flat_into(arr, depth, &expand, &mut result);
    result
}

fn flat_into<F>(arr: &JsArray, depth: u32, expand: &F, out: &mut JsArray)
where
    F: Fn(&JsValue) -> Option<JsArray>,
{
    for i in 0..arr.length() {
        let v = arr.get(i);
        if depth > 0
            && let Some(inner) = expand(&v)
        {
            flat_into(&inner, depth - 1, expand, out);
            continue;
        }
        out.push(v);
    }
}

// ── flatMap ───────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.11 `Array.prototype.flatMap(callbackFn)`.
///
/// Calls `f(element, index)` for each element, then flattens the result
/// arrays one level into a new [`JsArray`].
pub fn array_flat_map(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> JsArray) -> JsArray {
    let mut result = JsArray::new();
    for i in 0..arr.length() {
        let v = arr.get(i);
        let inner = f(&v, i);
        for j in 0..inner.length() {
            result.push(inner.get(j));
        }
    }
    result
}

// ── fill ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.6 `Array.prototype.fill(value, start?, end?)`.
///
/// Sets every element in the range `[start, end)` (clamped to `[0, length]`) to
/// `value`.  Omitting `start` defaults to `0`; omitting `end` defaults to
/// `length`.
///
/// Negative indices are interpreted as offsets from the end.
pub fn array_fill(arr: &mut JsArray, value: JsValue, start: Option<i64>, end: Option<i64>) {
    let len = arr.length();
    let begin = resolve_relative_index(start.unwrap_or(0), len);
    let finish = end.map(|e| resolve_relative_index(e, len)).unwrap_or(len);
    for i in begin..finish {
        arr.set(i, value.clone());
    }
}

// ── at ────────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.2 `Array.prototype.at(index)`.
///
/// Returns the element at `index`.  Negative indices count from the end.
/// Returns [`JsValue::Undefined`] if the computed index is out of bounds.
pub fn array_at(arr: &JsArray, index: i64) -> JsValue {
    let len = arr.length() as i64;
    let actual = if index < 0 { len + index } else { index };
    if actual < 0 || actual >= len {
        JsValue::Undefined
    } else {
        arr.get(actual as u32)
    }
}

// ── keys / values / entries / Symbol.iterator ─────────────────────────────────

/// ECMAScript §23.1.3.17 `Array.prototype.keys()`.
///
/// Returns a [`Vec<u32>`] of all valid indices (i.e. `0..length`).
pub fn array_keys(arr: &JsArray) -> Vec<u32> {
    (0..arr.length()).collect()
}

/// ECMAScript §23.1.3.33 `Array.prototype.values()`.
///
/// Returns a [`Vec<JsValue>`] of all elements in index order.  Sparse slots
/// appear as [`JsValue::Undefined`].
pub fn array_values(arr: &JsArray) -> Vec<JsValue> {
    (0..arr.length()).map(|i| arr.get(i)).collect()
}

/// ECMAScript §23.1.3.4 `Array.prototype.entries()`.
///
/// Returns a [`Vec`] of `(index, value)` pairs in index order.
pub fn array_entries(arr: &JsArray) -> Vec<(u32, JsValue)> {
    (0..arr.length()).map(|i| (i, arr.get(i))).collect()
}

/// ECMAScript §23.1.3.34 `Array.prototype[Symbol.iterator]()`.
///
/// Returns the same sequence as [`array_values`] — an in-order snapshot of
/// all element values.  Rust has no language-level iterator protocol, so this
/// simply delegates to [`array_values`].
pub fn array_symbol_iterator(arr: &JsArray) -> Vec<JsValue> {
    array_values(arr)
}

// ── findLast ──────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.11 `Array.prototype.findLast(callbackFn)`.
///
/// Returns the last element for which `f(element, index)` is `true`, or
/// [`JsValue::Undefined`] if none is found.
pub fn array_find_last(arr: &JsArray, mut f: impl FnMut(&JsValue, u32) -> bool) -> JsValue {
    let len = arr.length();
    for i in (0..len).rev() {
        let v = arr.get(i);
        if f(&v, i) {
            return v;
        }
    }
    JsValue::Undefined
}

// ── findLastIndex ─────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.12 `Array.prototype.findLastIndex(callbackFn)`.
///
/// Returns `Some(index)` of the last element for which `f(element, index)` is
/// `true`, or `None` if none is found.
pub fn array_find_last_index(
    arr: &JsArray,
    mut f: impl FnMut(&JsValue, u32) -> bool,
) -> Option<u32> {
    let len = arr.length();
    for i in (0..len).rev() {
        let v = arr.get(i);
        if f(&v, i) {
            return Some(i);
        }
    }
    None
}

// ── lastIndexOf ───────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.17 `Array.prototype.lastIndexOf(searchElement, fromIndex?)`.
///
/// Returns `Some(index)` of the last element strictly equal to `value` at or
/// before `from_index`, using strict equality (`===`; `NaN !== NaN`).
///
/// If `from_index` is `None`, the search starts at the last element.
/// Negative `from_index` is interpreted as an offset from the end.
pub fn array_last_index_of(arr: &JsArray, value: &JsValue, from_index: Option<i64>) -> Option<u32> {
    let len = arr.length();
    if len == 0 {
        return None;
    }
    let start = match from_index {
        Some(fi) if fi < 0 => {
            let idx = len as i64 + fi;
            if idx < 0 {
                return None;
            }
            idx as u32
        }
        Some(fi) => (fi as u32).min(len - 1),
        None => len - 1,
    };
    (0..=start)
        .rev()
        .find(|&i| strict_equal(&arr.get(i), value))
}

// ── reduceRight ───────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.23 `Array.prototype.reduceRight(callbackFn, initialValue?)`.
///
/// Like [`array_reduce`] but iterates from the last element to the first.
pub fn array_reduce_right(
    arr: &JsArray,
    mut f: impl FnMut(JsValue, &JsValue, u32) -> JsValue,
    initial: Option<JsValue>,
) -> StatorResult<JsValue> {
    let len = arr.length();
    let (mut acc, start_inclusive) = if let Some(init) = initial {
        if len == 0 {
            return Ok(init);
        }
        (init, len - 1)
    } else {
        if len == 0 {
            return Err(StatorError::TypeError(
                "Reduce of empty array with no initial value".to_string(),
            ));
        }
        if len == 1 {
            return Ok(arr.get(0));
        }
        (arr.get(len - 1), len - 2)
    };
    for i in (0..=start_inclusive).rev() {
        let v = arr.get(i);
        acc = f(acc, &v, i);
    }
    Ok(acc)
}

// ── copyWithin ────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.3 `Array.prototype.copyWithin(target, start, end?)`.
///
/// Copies the sequence of elements from `[start, end)` to position `target`,
/// without changing the array's length.  Negative indices are relative to the
/// end.
pub fn array_copy_within(arr: &mut JsArray, target: i64, start: i64, end: Option<i64>) {
    let len = arr.length();
    let to = resolve_relative_index(target, len);
    let from = resolve_relative_index(start, len);
    let fin = end.map(|e| resolve_relative_index(e, len)).unwrap_or(len);
    let count = ((fin as i64 - from as i64).max(0) as u32).min(len - to);
    // Copy to a temp buffer to handle overlapping regions.
    let buf: Vec<JsValue> = (0..count).map(|i| arr.get(from + i)).collect();
    for (i, v) in buf.into_iter().enumerate() {
        arr.set(to + i as u32, v);
    }
}

// ── toReversed ────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.31 `Array.prototype.toReversed()`.
///
/// Returns a new [`JsArray`] with the elements in reverse order.
/// The original array is unchanged.
pub fn array_to_reversed(arr: &JsArray) -> JsArray {
    let len = arr.length();
    let mut result = JsArray::new();
    for i in (0..len).rev() {
        result.push(arr.get(i));
    }
    result
}

// ── toSorted ──────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.32 `Array.prototype.toSorted(compareFn?)`.
///
/// Returns a new sorted [`JsArray`] without mutating the original.
pub fn array_to_sorted(
    arr: &JsArray,
    comparator: Option<impl FnMut(&JsValue, &JsValue) -> std::cmp::Ordering>,
) -> StatorResult<JsArray> {
    let mut copy = array_slice(arr, None, None);
    array_sort(&mut copy, comparator)?;
    Ok(copy)
}

// ── toSpliced ─────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.33 `Array.prototype.toSpliced(start, deleteCount, ...items)`.
///
/// Returns a new [`JsArray`] with the splice applied, leaving the original
/// unchanged.
pub fn array_to_spliced(
    arr: &JsArray,
    start: i64,
    delete_count: Option<u32>,
    items: &[JsValue],
) -> JsArray {
    let mut copy = array_slice(arr, None, None);
    let _ = array_splice(&mut copy, start, delete_count, items);
    copy
}

// ── with ──────────────────────────────────────────────────────────────────────

/// ECMAScript §23.1.3.36 `Array.prototype.with(index, value)`.
///
/// Returns a new [`JsArray`] identical to `arr` except that the element at
/// `index` is replaced with `value`.
///
/// Returns [`StatorError::RangeError`] if `index` is out of bounds.
pub fn array_with(arr: &JsArray, index: i64, value: JsValue) -> StatorResult<JsArray> {
    let len = arr.length() as i64;
    let actual = if index < 0 { len + index } else { index };
    if actual < 0 || actual >= len {
        return Err(StatorError::RangeError(format!("Invalid index : {index}")));
    }
    let mut result = array_slice(arr, None, None);
    result.set(actual as u32, value);
    Ok(result)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Converts a relative integer index to an absolute `u32` clamped to
/// `[0, length]`.
///
/// Negative indices are treated as `length + index`, following the ECMAScript
/// convention used by `slice`, `splice`, `indexOf`, etc.
fn resolve_relative_index(index: i64, length: u32) -> u32 {
    let len = length as i64;
    if index < 0 {
        (len + index).max(0) as u32
    } else {
        index.min(len) as u32
    }
}

/// ECMAScript §7.2.14 **SameValueZero**.
///
/// Like `===` but treats `NaN` as equal to itself (used by `includes`).
/// `+0` and `-0` are considered equal.
fn same_value_zero(a: JsValue, b: &JsValue) -> bool {
    match (&a, b) {
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Symbol(x), JsValue::Symbol(y)) => x == y,
        (JsValue::BigInt(x), JsValue::BigInt(y)) => x == y,
        (JsValue::Smi(x), JsValue::Smi(y)) => x == y,
        (JsValue::Smi(x), JsValue::HeapNumber(y)) => f64::from(*x) == *y,
        (JsValue::HeapNumber(x), JsValue::Smi(y)) => *x == f64::from(*y),
        // NaN == NaN under SameValueZero; +0 == -0.
        (JsValue::HeapNumber(x), JsValue::HeapNumber(y)) => {
            if x.is_nan() && y.is_nan() {
                return true;
            }
            x == y
        }
        _ => false,
    }
}

/// ECMAScript §7.2.15 **IsStrictlyEqual** — the `===` operator.
///
/// Used by `indexOf`: `NaN !== NaN`, `+0 === -0`.
fn strict_equal(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Symbol(x), JsValue::Symbol(y)) => x == y,
        (JsValue::BigInt(x), JsValue::BigInt(y)) => x == y,
        (JsValue::Smi(x), JsValue::Smi(y)) => x == y,
        (JsValue::Smi(x), JsValue::HeapNumber(y)) => f64::from(*x) == *y,
        (JsValue::HeapNumber(x), JsValue::Smi(y)) => *x == f64::from(*y),
        // NaN !== NaN; +0 === -0.
        (JsValue::HeapNumber(x), JsValue::HeapNumber(y)) => {
            if x.is_nan() || y.is_nan() {
                return false;
            }
            x == y
        }
        _ => false,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::js_array::JsArray;
    use crate::objects::js_object::JsObject;
    use crate::objects::value::JsValue;

    // ── array_is_array ────────────────────────────────────────────────────────

    #[test]
    fn test_array_is_array_with_array() {
        let arr = JsArray::new();
        assert!(array_is_array(arr.as_object()));
    }

    #[test]
    fn test_array_is_array_with_plain_object() {
        let obj = JsObject::new();
        assert!(!array_is_array(&obj));
    }

    // ── array_from ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_from_empty() {
        let arr = array_from(std::iter::empty());
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_array_from_values() {
        let arr = array_from([JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    // ── array_of ──────────────────────────────────────────────────────────────

    #[test]
    fn test_array_of_empty() {
        let arr = array_of(&[]);
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_array_of_values() {
        let arr = array_of(&[JsValue::Boolean(true), JsValue::Null]);
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.get(0), JsValue::Boolean(true));
        assert_eq!(arr.get(1), JsValue::Null);
    }

    // ── array_push ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_push_multiple() {
        let mut arr = JsArray::new();
        let new_len = array_push(&mut arr, &[JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(new_len, 2);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(2));
    }

    #[test]
    fn test_array_push_empty_slice() {
        let mut arr = array_of(&[JsValue::Smi(1)]);
        let new_len = array_push(&mut arr, &[]);
        assert_eq!(new_len, 1);
    }

    // ── array_pop ─────────────────────────────────────────────────────────────

    #[test]
    fn test_array_pop_returns_last() {
        let mut arr = array_of(&[JsValue::Smi(10), JsValue::Smi(20)]);
        assert_eq!(array_pop(&mut arr), JsValue::Smi(20));
        assert_eq!(arr.length(), 1);
    }

    #[test]
    fn test_array_pop_empty_returns_undefined() {
        let mut arr = JsArray::new();
        assert_eq!(array_pop(&mut arr), JsValue::Undefined);
    }

    // ── array_shift ───────────────────────────────────────────────────────────

    #[test]
    fn test_array_shift_removes_first() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let v = array_shift(&mut arr);
        assert_eq!(v, JsValue::Smi(1));
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.get(0), JsValue::Smi(2));
        assert_eq!(arr.get(1), JsValue::Smi(3));
    }

    #[test]
    fn test_array_shift_empty_returns_undefined() {
        let mut arr = JsArray::new();
        assert_eq!(array_shift(&mut arr), JsValue::Undefined);
    }

    // ── array_unshift ─────────────────────────────────────────────────────────

    #[test]
    fn test_array_unshift_prepends_elements() {
        let mut arr = array_of(&[JsValue::Smi(3), JsValue::Smi(4)]);
        let new_len = array_unshift(&mut arr, &[JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(new_len, 4);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(2));
        assert_eq!(arr.get(2), JsValue::Smi(3));
        assert_eq!(arr.get(3), JsValue::Smi(4));
    }

    #[test]
    fn test_array_unshift_empty_slice_is_noop() {
        let mut arr = array_of(&[JsValue::Smi(1)]);
        let new_len = array_unshift(&mut arr, &[]);
        assert_eq!(new_len, 1);
        assert_eq!(arr.get(0), JsValue::Smi(1));
    }

    // ── array_splice ──────────────────────────────────────────────────────────

    #[test]
    fn test_array_splice_delete_from_middle() {
        let mut arr = array_of(&[
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ]);
        let removed = array_splice(&mut arr, 1, Some(2), &[]);
        assert_eq!(removed.length(), 2);
        assert_eq!(removed.get(0), JsValue::Smi(2));
        assert_eq!(removed.get(1), JsValue::Smi(3));
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(4));
    }

    #[test]
    fn test_array_splice_insert_without_deletion() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(3)]);
        let removed = array_splice(&mut arr, 1, Some(0), &[JsValue::Smi(2)]);
        assert_eq!(removed.length(), 0);
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(2));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    #[test]
    fn test_array_splice_negative_start() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        // start = -1 → index 2.
        let removed = array_splice(&mut arr, -1, Some(1), &[]);
        assert_eq!(removed.length(), 1);
        assert_eq!(removed.get(0), JsValue::Smi(3));
        assert_eq!(arr.length(), 2);
    }

    #[test]
    fn test_array_splice_no_delete_count_removes_to_end() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let removed = array_splice(&mut arr, 1, None, &[]);
        assert_eq!(removed.length(), 2);
        assert_eq!(arr.length(), 1);
    }

    // ── array_map ─────────────────────────────────────────────────────────────

    #[test]
    fn test_array_map_doubles_smis() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_map(&arr, |v, _| {
            if let JsValue::Smi(n) = v {
                JsValue::Smi(n * 2)
            } else {
                v.clone()
            }
        });
        assert_eq!(result.length(), 3);
        assert_eq!(result.get(0), JsValue::Smi(2));
        assert_eq!(result.get(1), JsValue::Smi(4));
        assert_eq!(result.get(2), JsValue::Smi(6));
    }

    #[test]
    fn test_array_map_empty_array() {
        let arr = JsArray::new();
        let result = array_map(&arr, |v, _| v.clone());
        assert_eq!(result.length(), 0);
    }

    #[test]
    fn test_array_map_provides_index() {
        let arr = array_of(&[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);
        let mut indices = Vec::new();
        array_map(&arr, |_, i| {
            indices.push(i);
            JsValue::Undefined
        });
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // ── array_filter ──────────────────────────────────────────────────────────

    #[test]
    fn test_array_filter_keeps_positives() {
        let arr = array_of(&[
            JsValue::Smi(-1),
            JsValue::Smi(2),
            JsValue::Smi(-3),
            JsValue::Smi(4),
        ]);
        let result = array_filter(&arr, |v, _| matches!(v, JsValue::Smi(n) if *n > 0));
        assert_eq!(result.length(), 2);
        assert_eq!(result.get(0), JsValue::Smi(2));
        assert_eq!(result.get(1), JsValue::Smi(4));
    }

    #[test]
    fn test_array_filter_none_match_returns_empty() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        let result = array_filter(&arr, |_, _| false);
        assert_eq!(result.length(), 0);
    }

    // ── array_reduce ──────────────────────────────────────────────────────────

    #[test]
    fn test_array_reduce_sum_with_initial() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let sum = array_reduce(
            &arr,
            |acc, v, _| {
                if let (JsValue::Smi(a), JsValue::Smi(b)) = (acc, v) {
                    JsValue::Smi(a + b)
                } else {
                    JsValue::Undefined
                }
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(sum, JsValue::Smi(6));
    }

    #[test]
    fn test_array_reduce_without_initial_uses_first_element() {
        let arr = array_of(&[JsValue::Smi(10), JsValue::Smi(5)]);
        let result = array_reduce(
            &arr,
            |acc, v, _| {
                if let (JsValue::Smi(a), JsValue::Smi(b)) = (acc, v) {
                    JsValue::Smi(a - b)
                } else {
                    JsValue::Undefined
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn test_array_reduce_empty_without_initial_is_error() {
        let arr = JsArray::new();
        let err = array_reduce(&arr, |acc, _, _| acc, None);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── array_for_each ────────────────────────────────────────────────────────

    #[test]
    fn test_array_for_each_visits_all_elements() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let mut sum = 0i32;
        array_for_each(&arr, |v, _| {
            if let JsValue::Smi(n) = v {
                sum += n;
            }
        });
        assert_eq!(sum, 6);
    }

    // ── array_find ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_find_returns_first_match() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(3), JsValue::Smi(5)]);
        let found = array_find(&arr, |v, _| matches!(v, JsValue::Smi(n) if *n > 2));
        assert_eq!(found, JsValue::Smi(3));
    }

    #[test]
    fn test_array_find_no_match_returns_undefined() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        let found = array_find(&arr, |_, _| false);
        assert_eq!(found, JsValue::Undefined);
    }

    // ── array_find_index ──────────────────────────────────────────────────────

    #[test]
    fn test_array_find_index_returns_correct_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let idx = array_find_index(&arr, |v, _| matches!(v, JsValue::Smi(n) if *n == 2));
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_array_find_index_no_match_returns_none() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert_eq!(array_find_index(&arr, |_, _| false), None);
    }

    // ── array_some ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_some_true_when_any_matches() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert!(array_some(
            &arr,
            |v, _| matches!(v, JsValue::Smi(n) if *n > 2)
        ));
    }

    #[test]
    fn test_array_some_false_when_none_matches() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        assert!(!array_some(&arr, |_, _| false));
    }

    #[test]
    fn test_array_some_false_on_empty() {
        assert!(!array_some(&JsArray::new(), |_, _| true));
    }

    // ── array_every ───────────────────────────────────────────────────────────

    #[test]
    fn test_array_every_true_when_all_match() {
        let arr = array_of(&[JsValue::Smi(2), JsValue::Smi(4), JsValue::Smi(6)]);
        assert!(array_every(
            &arr,
            |v, _| matches!(v, JsValue::Smi(n) if n % 2 == 0)
        ));
    }

    #[test]
    fn test_array_every_false_when_one_fails() {
        let arr = array_of(&[JsValue::Smi(2), JsValue::Smi(3)]);
        assert!(!array_every(
            &arr,
            |v, _| matches!(v, JsValue::Smi(n) if n % 2 == 0)
        ));
    }

    #[test]
    fn test_array_every_true_on_empty() {
        assert!(array_every(&JsArray::new(), |_, _| false));
    }

    // ── array_includes ────────────────────────────────────────────────────────

    #[test]
    fn test_array_includes_finds_element() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert!(array_includes(&arr, &JsValue::Smi(2), None));
    }

    #[test]
    fn test_array_includes_not_found() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        assert!(!array_includes(&arr, &JsValue::Smi(99), None));
    }

    #[test]
    fn test_array_includes_nan_equals_nan() {
        let arr = array_of(&[JsValue::HeapNumber(f64::NAN)]);
        assert!(array_includes(&arr, &JsValue::HeapNumber(f64::NAN), None));
    }

    #[test]
    fn test_array_includes_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        // Value 1 exists at index 2 only when searching from index 1.
        assert!(array_includes(&arr, &JsValue::Smi(1), Some(1)));
    }

    #[test]
    fn test_array_includes_negative_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        // from_index = -1 → start at index 2 (last element).
        assert!(array_includes(&arr, &JsValue::Smi(3), Some(-1)));
        assert!(!array_includes(&arr, &JsValue::Smi(1), Some(-1)));
    }

    // ── array_index_of ────────────────────────────────────────────────────────

    #[test]
    fn test_array_index_of_finds_first() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        assert_eq!(array_index_of(&arr, &JsValue::Smi(1), None), Some(0));
    }

    #[test]
    fn test_array_index_of_not_found() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert_eq!(array_index_of(&arr, &JsValue::Smi(9), None), None);
    }

    #[test]
    fn test_array_index_of_nan_not_equal_nan() {
        let arr = array_of(&[JsValue::HeapNumber(f64::NAN)]);
        // indexOf uses strict equality: NaN !== NaN.
        assert_eq!(
            array_index_of(&arr, &JsValue::HeapNumber(f64::NAN), None),
            None
        );
    }

    #[test]
    fn test_array_index_of_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        assert_eq!(array_index_of(&arr, &JsValue::Smi(1), Some(1)), Some(2));
    }

    #[test]
    fn test_array_index_of_negative_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        // from = -1 → index 2.
        assert_eq!(array_index_of(&arr, &JsValue::Smi(3), Some(-1)), Some(2));
    }

    // ── array_concat ──────────────────────────────────────────────────────────

    #[test]
    fn test_array_concat_two_arrays() {
        let a = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        let b = array_of(&[JsValue::Smi(3), JsValue::Smi(4)]);
        let result = array_concat(&a, &[&b]);
        assert_eq!(result.length(), 4);
        assert_eq!(result.get(2), JsValue::Smi(3));
    }

    #[test]
    fn test_array_concat_with_empty() {
        let a = array_of(&[JsValue::Smi(1)]);
        let empty = JsArray::new();
        let result = array_concat(&a, &[&empty]);
        assert_eq!(result.length(), 1);
    }

    // ── array_slice ───────────────────────────────────────────────────────────

    #[test]
    fn test_array_slice_basic() {
        let arr = array_of(&[
            JsValue::Smi(0),
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
        ]);
        let s = array_slice(&arr, Some(1), Some(3));
        assert_eq!(s.length(), 2);
        assert_eq!(s.get(0), JsValue::Smi(1));
        assert_eq!(s.get(1), JsValue::Smi(2));
    }

    #[test]
    fn test_array_slice_no_args_copies_all() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        let s = array_slice(&arr, None, None);
        assert_eq!(s.length(), 2);
    }

    #[test]
    fn test_array_slice_negative_start() {
        let arr = array_of(&[JsValue::Smi(0), JsValue::Smi(1), JsValue::Smi(2)]);
        let s = array_slice(&arr, Some(-2), None);
        assert_eq!(s.length(), 2);
        assert_eq!(s.get(0), JsValue::Smi(1));
    }

    #[test]
    fn test_array_slice_negative_end() {
        let arr = array_of(&[JsValue::Smi(0), JsValue::Smi(1), JsValue::Smi(2)]);
        let s = array_slice(&arr, None, Some(-1));
        assert_eq!(s.length(), 2);
    }

    // ── array_join ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_join_comma_separator() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert_eq!(array_join(&arr, None).unwrap(), "1,2,3");
    }

    #[test]
    fn test_array_join_custom_separator() {
        let arr = array_of(&[
            JsValue::String("a".to_string().into()),
            JsValue::String("b".to_string().into()),
        ]);
        assert_eq!(array_join(&arr, Some("-")).unwrap(), "a-b");
    }

    #[test]
    fn test_array_join_null_and_undefined_become_empty() {
        let arr = array_of(&[JsValue::Null, JsValue::Undefined, JsValue::Smi(1)]);
        assert_eq!(array_join(&arr, Some(",")).unwrap(), ",,1");
    }

    #[test]
    fn test_array_join_empty_array() {
        let arr = JsArray::new();
        assert_eq!(array_join(&arr, None).unwrap(), "");
    }

    // ── array_reverse ─────────────────────────────────────────────────────────

    #[test]
    fn test_array_reverse_odd_length() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        array_reverse(&mut arr);
        assert_eq!(arr.get(0), JsValue::Smi(3));
        assert_eq!(arr.get(1), JsValue::Smi(2));
        assert_eq!(arr.get(2), JsValue::Smi(1));
    }

    #[test]
    fn test_array_reverse_single_element() {
        let mut arr = array_of(&[JsValue::Smi(42)]);
        array_reverse(&mut arr);
        assert_eq!(arr.get(0), JsValue::Smi(42));
    }

    #[test]
    fn test_array_reverse_empty() {
        let mut arr = JsArray::new();
        array_reverse(&mut arr); // must not panic
        assert_eq!(arr.length(), 0);
    }

    // ── array_sort ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_sort_default_lexicographic() {
        let mut arr = array_of(&[
            JsValue::Smi(10),
            JsValue::Smi(9),
            JsValue::Smi(2),
            JsValue::Smi(1),
        ]);
        array_sort(
            &mut arr,
            None::<fn(&JsValue, &JsValue) -> std::cmp::Ordering>,
        )
        .unwrap();
        // Lexicographic order: "1", "10", "2", "9".
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(10));
        assert_eq!(arr.get(2), JsValue::Smi(2));
        assert_eq!(arr.get(3), JsValue::Smi(9));
    }

    #[test]
    fn test_array_sort_numeric_comparator() {
        let mut arr = array_of(&[JsValue::Smi(3), JsValue::Smi(1), JsValue::Smi(2)]);
        array_sort(
            &mut arr,
            Some(|a: &JsValue, b: &JsValue| {
                let na = if let JsValue::Smi(n) = a { *n } else { 0 };
                let nb = if let JsValue::Smi(n) = b { *n } else { 0 };
                na.cmp(&nb)
            }),
        )
        .unwrap();
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(2));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    // ── array_flat ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_flat_depth_zero_no_expansion() {
        let inner = array_of(&[JsValue::Smi(2), JsValue::Smi(3)]);
        let mut outer = array_of(&[JsValue::Smi(1)]);
        // depth=0 → nothing expanded regardless of the callback.
        for i in 0..inner.length() {
            outer.push(inner.get(i));
        }
        let flat = array_flat(&outer, 0, |_| None);
        assert_eq!(flat.length(), 3);
    }

    #[test]
    fn test_array_flat_expands_inner_arrays() {
        // Simulate [1, [2, 3], 4] where the "inner" array is tagged via a
        // sentinel string value in the outer array.  We represent it differently:
        // outer = [10, 20, 30]; expand doubles even numbers as [n, n].
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let flat = array_flat(&arr, 1, |v| {
            if let JsValue::Smi(n) = v {
                if n % 2 == 0 {
                    return Some(array_of(&[JsValue::Smi(*n), JsValue::Smi(*n)]));
                }
            }
            None
        });
        // 1 stays, 2 → [2,2], 3 stays → [1, 2, 2, 3].
        assert_eq!(flat.length(), 4);
        assert_eq!(flat.get(0), JsValue::Smi(1));
        assert_eq!(flat.get(1), JsValue::Smi(2));
        assert_eq!(flat.get(2), JsValue::Smi(2));
        assert_eq!(flat.get(3), JsValue::Smi(3));
    }

    // ── array_flat_map ────────────────────────────────────────────────────────

    #[test]
    fn test_array_flat_map_doubles_each() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_flat_map(&arr, |v, _| array_of(&[v.clone(), v.clone()]));
        assert_eq!(result.length(), 6);
        assert_eq!(result.get(0), JsValue::Smi(1));
        assert_eq!(result.get(1), JsValue::Smi(1));
        assert_eq!(result.get(4), JsValue::Smi(3));
    }

    // ── array_fill ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_fill_all() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        array_fill(&mut arr, JsValue::Smi(0), None, None);
        assert_eq!(arr.get(0), JsValue::Smi(0));
        assert_eq!(arr.get(1), JsValue::Smi(0));
        assert_eq!(arr.get(2), JsValue::Smi(0));
    }

    #[test]
    fn test_array_fill_range() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        array_fill(&mut arr, JsValue::Smi(9), Some(1), Some(2));
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(9));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    #[test]
    fn test_array_fill_negative_indices() {
        let mut arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        array_fill(&mut arr, JsValue::Smi(5), Some(-2), Some(-1));
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(5));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    // ── array_at ──────────────────────────────────────────────────────────────

    #[test]
    fn test_array_at_positive_index() {
        let arr = array_of(&[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);
        assert_eq!(array_at(&arr, 1), JsValue::Smi(20));
    }

    #[test]
    fn test_array_at_negative_index() {
        let arr = array_of(&[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)]);
        assert_eq!(array_at(&arr, -1), JsValue::Smi(30));
        assert_eq!(array_at(&arr, -3), JsValue::Smi(10));
    }

    #[test]
    fn test_array_at_out_of_bounds() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert_eq!(array_at(&arr, 5), JsValue::Undefined);
        assert_eq!(array_at(&arr, -5), JsValue::Undefined);
    }

    // ── array_keys / values / entries ─────────────────────────────────────────

    #[test]
    fn test_array_keys() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        assert_eq!(array_keys(&arr), vec![0, 1, 2]);
    }

    #[test]
    fn test_array_keys_empty() {
        assert_eq!(array_keys(&JsArray::new()), Vec::<u32>::new());
    }

    #[test]
    fn test_array_values() {
        let arr = array_of(&[JsValue::Smi(7), JsValue::Boolean(false)]);
        let vals = array_values(&arr);
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], JsValue::Smi(7));
        assert_eq!(vals[1], JsValue::Boolean(false));
    }

    #[test]
    fn test_array_entries() {
        let arr = array_of(&[JsValue::Smi(10), JsValue::Smi(20)]);
        let entries = array_entries(&arr);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (0, JsValue::Smi(10)));
        assert_eq!(entries[1], (1, JsValue::Smi(20)));
    }

    #[test]
    fn test_array_symbol_iterator_same_as_values() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(array_symbol_iterator(&arr), array_values(&arr));
    }

    // ── sparse array tests ────────────────────────────────────────────────────

    #[test]
    fn test_sparse_array_map_visits_holes_as_undefined() {
        let mut arr = JsArray::new();
        arr.set(3, JsValue::Smi(99)); // indices 0-2 are holes.
        let result = array_map(&arr, |v, _| v.clone());
        assert_eq!(result.length(), 4);
        assert_eq!(result.get(0), JsValue::Undefined);
        assert_eq!(result.get(3), JsValue::Smi(99));
    }

    #[test]
    fn test_sparse_array_filter_excludes_holes() {
        let mut arr = JsArray::new();
        arr.set(0, JsValue::Smi(1));
        arr.set(2, JsValue::Smi(3)); // index 1 is a hole.
        let result = array_filter(&arr, |v, _| !v.is_undefined());
        assert_eq!(result.length(), 2);
        assert_eq!(result.get(0), JsValue::Smi(1));
        assert_eq!(result.get(1), JsValue::Smi(3));
    }

    #[test]
    fn test_sparse_array_includes_treats_holes_as_undefined() {
        let mut arr = JsArray::new();
        arr.set(2, JsValue::Smi(5)); // indices 0,1 are holes.
        assert!(array_includes(&arr, &JsValue::Undefined, None));
        assert!(array_includes(&arr, &JsValue::Smi(5), None));
    }

    #[test]
    fn test_sparse_array_join_holes_as_empty_string() {
        let mut arr = JsArray::new();
        arr.set(0, JsValue::Smi(1));
        arr.set(2, JsValue::Smi(3)); // hole at index 1.
        let s = array_join(&arr, Some(",")).unwrap();
        assert_eq!(s, "1,,3");
    }

    #[test]
    fn test_sparse_array_entries_includes_holes() {
        let mut arr = JsArray::new();
        arr.set(2, JsValue::Smi(7));
        let entries = array_entries(&arr);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], (0, JsValue::Undefined));
        assert_eq!(entries[2], (2, JsValue::Smi(7)));
    }

    // ── array_find_last ───────────────────────────────────────────────────────

    #[test]
    fn test_array_find_last_returns_last_match() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(3), JsValue::Smi(5)]);
        let found = array_find_last(&arr, |v, _| matches!(v, JsValue::Smi(n) if *n > 2));
        assert_eq!(found, JsValue::Smi(5));
    }

    #[test]
    fn test_array_find_last_no_match_returns_undefined() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(array_find_last(&arr, |_, _| false), JsValue::Undefined);
    }

    // ── array_find_last_index ─────────────────────────────────────────────────

    #[test]
    fn test_array_find_last_index_returns_last() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let idx = array_find_last_index(&arr, |v, _| matches!(v, JsValue::Smi(n) if *n >= 2));
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn test_array_find_last_index_no_match_returns_none() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert_eq!(array_find_last_index(&arr, |_, _| false), None);
    }

    // ── array_last_index_of ───────────────────────────────────────────────────

    #[test]
    fn test_array_last_index_of_finds_last() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        assert_eq!(array_last_index_of(&arr, &JsValue::Smi(1), None), Some(2));
    }

    #[test]
    fn test_array_last_index_of_with_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        assert_eq!(
            array_last_index_of(&arr, &JsValue::Smi(1), Some(1)),
            Some(0)
        );
    }

    #[test]
    fn test_array_last_index_of_not_found() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert_eq!(array_last_index_of(&arr, &JsValue::Smi(9), None), None);
    }

    #[test]
    fn test_array_last_index_of_negative_from_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)]);
        // from = -2 → index 1.
        assert_eq!(
            array_last_index_of(&arr, &JsValue::Smi(1), Some(-2)),
            Some(0)
        );
    }

    #[test]
    fn test_array_last_index_of_empty_array() {
        let arr = JsArray::new();
        assert_eq!(array_last_index_of(&arr, &JsValue::Smi(1), None), None);
    }

    // ── array_reduce_right ────────────────────────────────────────────────────

    #[test]
    fn test_array_reduce_right_with_initial() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_reduce_right(
            &arr,
            |acc, v, _| {
                if let (JsValue::Smi(a), JsValue::Smi(b)) = (acc, v) {
                    JsValue::Smi(a + b)
                } else {
                    JsValue::Undefined
                }
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn test_array_reduce_right_without_initial() {
        // [1, 2, 3] reduceRight with subtraction: 3 - 2 - 1 = 0
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_reduce_right(
            &arr,
            |acc, v, _| {
                if let (JsValue::Smi(a), JsValue::Smi(b)) = (acc, v) {
                    JsValue::Smi(a - b)
                } else {
                    JsValue::Undefined
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_array_reduce_right_empty_without_initial_is_error() {
        let arr = JsArray::new();
        let err = array_reduce_right(&arr, |acc, _, _| acc, None);
        assert!(matches!(err, Err(StatorError::TypeError(_))));
    }

    // ── array_copy_within ─────────────────────────────────────────────────────

    #[test]
    fn test_array_copy_within_basic() {
        let mut arr = array_of(&[
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
            JsValue::Smi(5),
        ]);
        array_copy_within(&mut arr, 0, 3, None);
        assert_eq!(arr.get(0), JsValue::Smi(4));
        assert_eq!(arr.get(1), JsValue::Smi(5));
        assert_eq!(arr.get(2), JsValue::Smi(3));
    }

    #[test]
    fn test_array_copy_within_negative_target() {
        let mut arr = array_of(&[
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
            JsValue::Smi(4),
        ]);
        // target=-2 → index 2, copy from 0..2 → [1,2] into positions 2,3
        array_copy_within(&mut arr, -2, 0, Some(2));
        assert_eq!(arr.get(2), JsValue::Smi(1));
        assert_eq!(arr.get(3), JsValue::Smi(2));
    }

    // ── array_to_reversed ─────────────────────────────────────────────────────

    #[test]
    fn test_array_to_reversed_returns_new_array() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let reversed = array_to_reversed(&arr);
        assert_eq!(reversed.get(0), JsValue::Smi(3));
        assert_eq!(reversed.get(1), JsValue::Smi(2));
        assert_eq!(reversed.get(2), JsValue::Smi(1));
        // Original unchanged.
        assert_eq!(arr.get(0), JsValue::Smi(1));
    }

    // ── array_to_sorted ───────────────────────────────────────────────────────

    #[test]
    fn test_array_to_sorted_returns_new_array() {
        let arr = array_of(&[JsValue::Smi(3), JsValue::Smi(1), JsValue::Smi(2)]);
        let sorted = array_to_sorted(
            &arr,
            Some(|a: &JsValue, b: &JsValue| {
                let na = if let JsValue::Smi(n) = a { *n } else { 0 };
                let nb = if let JsValue::Smi(n) = b { *n } else { 0 };
                na.cmp(&nb)
            }),
        )
        .unwrap();
        assert_eq!(sorted.get(0), JsValue::Smi(1));
        assert_eq!(sorted.get(1), JsValue::Smi(2));
        assert_eq!(sorted.get(2), JsValue::Smi(3));
        // Original unchanged.
        assert_eq!(arr.get(0), JsValue::Smi(3));
    }

    // ── array_to_spliced ──────────────────────────────────────────────────────

    #[test]
    fn test_array_to_spliced_returns_new_array() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let spliced = array_to_spliced(&arr, 1, Some(1), &[JsValue::Smi(99)]);
        assert_eq!(spliced.length(), 3);
        assert_eq!(spliced.get(1), JsValue::Smi(99));
        // Original unchanged.
        assert_eq!(arr.get(1), JsValue::Smi(2));
    }

    // ── array_with ────────────────────────────────────────────────────────────

    #[test]
    fn test_array_with_positive_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_with(&arr, 1, JsValue::Smi(99)).unwrap();
        assert_eq!(result.get(1), JsValue::Smi(99));
        // Original unchanged.
        assert_eq!(arr.get(1), JsValue::Smi(2));
    }

    #[test]
    fn test_array_with_negative_index() {
        let arr = array_of(&[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = array_with(&arr, -1, JsValue::Smi(99)).unwrap();
        assert_eq!(result.get(2), JsValue::Smi(99));
    }

    #[test]
    fn test_array_with_out_of_bounds() {
        let arr = array_of(&[JsValue::Smi(1)]);
        assert!(matches!(
            array_with(&arr, 5, JsValue::Smi(1)),
            Err(StatorError::RangeError(_))
        ));
    }

    // ── array_from_with_map_fn ────────────────────────────────────────────────

    #[test]
    fn test_array_from_with_map_fn_doubles() {
        let arr = array_from_with_map_fn(
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
            |v, _| {
                if let JsValue::Smi(n) = v {
                    JsValue::Smi(n * 2)
                } else {
                    v
                }
            },
        );
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.get(0), JsValue::Smi(2));
        assert_eq!(arr.get(1), JsValue::Smi(4));
        assert_eq!(arr.get(2), JsValue::Smi(6));
    }

    #[test]
    fn test_array_from_with_map_fn_uses_index() {
        let arr = array_from_with_map_fn(vec![JsValue::Smi(10), JsValue::Smi(20)], |_, i| {
            JsValue::Smi(i as i32)
        });
        assert_eq!(arr.get(0), JsValue::Smi(0));
        assert_eq!(arr.get(1), JsValue::Smi(1));
    }

    #[test]
    fn test_array_from_with_map_fn_empty() {
        let arr = array_from_with_map_fn(Vec::<JsValue>::new(), |v, _| v);
        assert_eq!(arr.length(), 0);
    }

    // ── array_from_async ──────────────────────────────────────────────────────

    #[test]
    fn test_array_from_async_basic() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let p = array_from_async(
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
            None::<fn(JsValue, u32) -> JsValue>,
            &queue,
        );
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 3);
                assert_eq!(arr.borrow()[0], JsValue::Smi(1));
            }
            other => panic!("expected fulfilled array, got {other:?}"),
        }
    }

    #[test]
    fn test_array_from_async_empty() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let p = array_from_async(
            Vec::<JsValue>::new(),
            None::<fn(JsValue, u32) -> JsValue>,
            &queue,
        );
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 0);
            }
            other => panic!("expected fulfilled empty array, got {other:?}"),
        }
    }

    #[test]
    fn test_array_from_async_with_map_fn() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let p = array_from_async(
            vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
            Some(|v: JsValue, _i: u32| {
                if let JsValue::Smi(n) = v {
                    JsValue::Smi(n * 10)
                } else {
                    v
                }
            }),
            &queue,
        );
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                let borrowed = arr.borrow();
                assert_eq!(borrowed.len(), 3);
                assert_eq!(borrowed[0], JsValue::Smi(10));
                assert_eq!(borrowed[1], JsValue::Smi(20));
                assert_eq!(borrowed[2], JsValue::Smi(30));
            }
            other => panic!("expected fulfilled array, got {other:?}"),
        }
    }

    #[test]
    fn test_array_from_async_with_map_fn_uses_index() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let p = array_from_async(
            vec![JsValue::Smi(100), JsValue::Smi(200)],
            Some(|_v: JsValue, i: u32| JsValue::Smi(i as i32)),
            &queue,
        );
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                let borrowed = arr.borrow();
                assert_eq!(borrowed[0], JsValue::Smi(0));
                assert_eq!(borrowed[1], JsValue::Smi(1));
            }
            other => panic!("expected fulfilled array, got {other:?}"),
        }
    }

    #[test]
    fn test_array_from_async_returns_promise() {
        use crate::builtins::promise::MicrotaskQueue;

        let queue = MicrotaskQueue::new();
        let p = array_from_async(
            vec![JsValue::Smi(1)],
            None::<fn(JsValue, u32) -> JsValue>,
            &queue,
        );
        // Even before drain, it's a fulfilled promise (sync collection).
        assert!(p.is_fulfilled());
    }

    #[test]
    fn test_array_from_async_sync_iterable_strings() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let items = vec![
            JsValue::String("a".into()),
            JsValue::String("b".into()),
            JsValue::String("c".into()),
        ];
        let p = array_from_async(items, None::<fn(JsValue, u32) -> JsValue>, &queue);
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 3);
                assert_eq!(arr.borrow()[0], JsValue::String("a".into()));
            }
            other => panic!("expected fulfilled array, got {other:?}"),
        }
    }

    #[test]
    fn test_array_from_async_sparse_undefined() {
        use crate::builtins::promise::{MicrotaskQueue, PromiseState};

        let queue = MicrotaskQueue::new();
        let items = vec![JsValue::Undefined, JsValue::Smi(1), JsValue::Undefined];
        let p = array_from_async(items, None::<fn(JsValue, u32) -> JsValue>, &queue);
        queue.drain();
        match p.state() {
            PromiseState::Fulfilled(JsValue::Array(arr)) => {
                let b = arr.borrow();
                assert_eq!(b.len(), 3);
                assert_eq!(b[0], JsValue::Undefined);
                assert_eq!(b[1], JsValue::Smi(1));
                assert_eq!(b[2], JsValue::Undefined);
            }
            other => panic!("expected fulfilled array, got {other:?}"),
        }
    }
}

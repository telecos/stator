//! ECMAScript §24.2 `Set` built-in.
//!
//! Provides [`JsSet`], a collection of unique values that preserves insertion
//! order and uses the **SameValueZero** comparison algorithm for membership
//! tests (identical to `===` except that `NaN` is considered equal to itself
//! and `-0` is considered equal to `+0`).
//!
//! # Naming convention
//!
//! Each function is prefixed `set_` to avoid ambiguity with similarly-named
//! standard-library items.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §24.2 — *The Set Objects*

use crate::objects::value::{JsValue, NativeIterator};

use super::util::same_value_zero;

/// ECMAScript §24.2.3.1 step 5: if *value* is **-0**𝔽, set *value* to
/// **+0**𝔽.
///
/// Mirrors the normalisation applied by [`super::map::map_set`] for `Map`
/// keys.
fn normalize_negative_zero(v: JsValue) -> JsValue {
    if let JsValue::HeapNumber(n) = &v
        && *n == 0.0
        && n.is_sign_negative()
    {
        return JsValue::HeapNumber(0.0);
    }
    v
}

// ── SetIteratorKind ───────────────────────────────────────────────────────────

/// The iteration kind for a `Set` iterator (ECMAScript §24.2.5.1).
///
/// Determines what the iterator yields:
/// - `Values` → individual values
/// - `Entries` → `[value, value]` pairs (matching `Map` entry format)
/// - `Keys` → alias for `Values` (per spec, `Set.prototype.keys === Set.prototype.values`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetIteratorKind {
    /// Yield individual values.
    Values,
    /// Yield `[value, value]` pairs (matching the Map entry format).
    Entries,
    /// Alias for `Values` — `Set.prototype.keys` is the same function as
    /// `Set.prototype.values` per the ECMAScript specification.
    Keys,
}

// ── JsSet ─────────────────────────────────────────────────────────────────────

/// A JavaScript `Set` object per ECMAScript §24.2.
///
/// Values are stored in a [`Vec`] in insertion order, matching the ECMAScript
/// requirement that `Set` operations iterate in insertion order.  Membership
/// tests use a linear scan with [`same_value_zero`] comparison.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_has, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(1));
/// set_add(&mut s, JsValue::Smi(1)); // duplicate — ignored
/// assert_eq!(set_size(&s), 1);
/// assert!(set_has(&s, &JsValue::Smi(1)));
/// ```
#[derive(Debug, Clone, Default)]
pub struct JsSet {
    /// Values stored in insertion order.  Duplicates are never inserted.
    values: Vec<JsValue>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

/// ECMAScript §24.2.1.1 `new Set()`.
///
/// Creates an empty [`JsSet`].
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_size};
///
/// let s = set_new();
/// assert_eq!(set_size(&s), 0);
/// ```
pub fn set_new() -> JsSet {
    JsSet::default()
}

/// ECMAScript §24.2.1.1 `new Set(iterable)`.
///
/// Creates a [`JsSet`] from an iterable of values.  Duplicates (per
/// `SameValueZero`) are silently ignored.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_from_iterable, set_has, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)];
/// let s = set_from_iterable(items);
/// assert_eq!(set_size(&s), 2);
/// assert!(set_has(&s, &JsValue::Smi(1)));
/// ```
pub fn set_from_iterable(items: Vec<JsValue>) -> JsSet {
    let mut s = set_new();
    for v in items {
        set_add(&mut s, v);
    }
    s
}

// ── set_add ───────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.1 `Set.prototype.add(value)`.
///
/// Adds `value` to the set if no `SameValueZero`-equal value is already
/// present.  If a matching value already exists this is a no-op.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(5));
/// set_add(&mut s, JsValue::Smi(5));
/// assert_eq!(set_size(&s), 1);
/// ```
pub fn set_add(set: &mut JsSet, value: JsValue) {
    // §24.2.3.1 step 5: normalise -0 → +0.
    let value = normalize_negative_zero(value);
    if !set.values.iter().any(|v| same_value_zero(v, &value)) {
        set.values.push(value);
    }
}

// ── set_has ───────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.6 `Set.prototype.has(value)`.
///
/// Returns `true` if a `SameValueZero`-equal value is present in the set.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_has};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Boolean(true));
/// assert!(set_has(&s, &JsValue::Boolean(true)));
/// assert!(!set_has(&s, &JsValue::Boolean(false)));
/// ```
pub fn set_has(set: &JsSet, value: &JsValue) -> bool {
    set.values.iter().any(|v| same_value_zero(v, value))
}

// ── set_delete ────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.4 `Set.prototype.delete(value)`.
///
/// Removes the first value that is `SameValueZero`-equal to `value`.  Returns
/// `true` if a value was removed, `false` if `value` was not found.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_delete, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(3));
/// assert!(set_delete(&mut s, &JsValue::Smi(3)));
/// assert!(!set_delete(&mut s, &JsValue::Smi(3)));
/// assert_eq!(set_size(&s), 0);
/// ```
pub fn set_delete(set: &mut JsSet, value: &JsValue) -> bool {
    if let Some(pos) = set.values.iter().position(|v| same_value_zero(v, value)) {
        set.values.remove(pos);
        true
    } else {
        false
    }
}

// ── set_clear ─────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.2 `Set.prototype.clear()`.
///
/// Removes all values from the set.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_clear, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(1));
/// set_clear(&mut s);
/// assert_eq!(set_size(&s), 0);
/// ```
pub fn set_clear(set: &mut JsSet) {
    set.values.clear();
}

// ── set_size ──────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.9 `Set.prototype.size` (getter).
///
/// Returns the number of values currently in the set.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(10));
/// assert_eq!(set_size(&s), 1);
/// ```
pub fn set_size(set: &JsSet) -> usize {
    set.values.len()
}

// ── set_for_each ──────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.5 `Set.prototype.forEach(callback)`.
///
/// Calls `callback(value)` for each value in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_for_each};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(1));
/// set_add(&mut s, JsValue::Smi(2));
/// let mut out = Vec::new();
/// set_for_each(&s, |v| out.push(v.clone()));
/// assert_eq!(out, vec![JsValue::Smi(1), JsValue::Smi(2)]);
/// ```
pub fn set_for_each(set: &JsSet, mut callback: impl FnMut(&JsValue)) {
    for v in &set.values {
        callback(v);
    }
}

// ── set_values ────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.10 `Set.prototype.values()`.
///
/// Returns all values in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_values};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(7));
/// set_add(&mut s, JsValue::Smi(3));
/// assert_eq!(set_values(&s), vec![JsValue::Smi(7), JsValue::Smi(3)]);
/// ```
pub fn set_values(set: &JsSet) -> Vec<JsValue> {
    set.values.clone()
}

/// ECMAScript §24.2.3.8 `Set.prototype.keys()`.
///
/// Returns all values in insertion order.  Per the ECMAScript specification,
/// `Set.prototype.keys` is the **same function** as `Set.prototype.values`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_keys};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(7));
/// assert_eq!(set_keys(&s), vec![JsValue::Smi(7)]);
/// ```
pub fn set_keys(set: &JsSet) -> Vec<JsValue> {
    set_values(set)
}

// ── set_entries ───────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3.5 `Set.prototype.entries()`.
///
/// Returns `(value, value)` pairs in insertion order.  The key and value of
/// each entry are identical, matching the spec's requirement that `Set`
/// entries mirror the `Map` entry format.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_entries};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(1));
/// assert_eq!(set_entries(&s), vec![(JsValue::Smi(1), JsValue::Smi(1))]);
/// ```
pub fn set_entries(set: &JsSet) -> Vec<(JsValue, JsValue)> {
    set.values.iter().map(|v| (v.clone(), v.clone())).collect()
}

// ── set_iter ──────────────────────────────────────────────────────────────────

/// ECMAScript §24.2.3 `Set.prototype[@@iterator]()`.
///
/// Returns all values in insertion order.  This is the same as
/// [`set_values`] and fulfils the `Symbol.iterator` protocol for `Set`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_iter};
/// use stator_core::objects::value::JsValue;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(5));
/// let vals = set_iter(&s);
/// assert_eq!(vals.len(), 1);
/// ```
pub fn set_iter(set: &JsSet) -> Vec<JsValue> {
    set_values(set)
}

// ── set_create_iterator ──────────────────────────────────────────────────────

/// Create a [`JsValue::Iterator`] from a `Set` with the given iteration kind
/// (ECMAScript §24.2.5 `CreateSetIterator`).
///
/// - [`SetIteratorKind::Values`] / [`SetIteratorKind::Keys`] yields values.
/// - [`SetIteratorKind::Entries`] yields `[value, value]` arrays.
///
/// The iterator snapshots the current values and yields them in insertion
/// order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_create_iterator, SetIteratorKind};
/// use stator_core::builtins::iterator::iterator_next;
/// use stator_core::objects::value::JsValue;
/// use std::rc::Rc;
///
/// let mut s = set_new();
/// set_add(&mut s, JsValue::Smi(1));
///
/// let iter = set_create_iterator(&s, SetIteratorKind::Values);
/// let r = iterator_next(&iter).unwrap();
/// assert_eq!(r.value, JsValue::Smi(1));
///
/// let iter = set_create_iterator(&s, SetIteratorKind::Entries);
/// let r = iterator_next(&iter).unwrap();
/// if let JsValue::Array(arr) = &r.value {
///     assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(1)]);
/// } else {
///     panic!("expected Array");
/// }
/// ```
pub fn set_create_iterator(set: &JsSet, kind: SetIteratorKind) -> JsValue {
    let items: Vec<JsValue> = match kind {
        SetIteratorKind::Values | SetIteratorKind::Keys => set.values.clone(),
        SetIteratorKind::Entries => set
            .values
            .iter()
            .map(|v| JsValue::new_array(vec![v.clone(), v.clone()]))
            .collect(),
    };
    JsValue::Iterator(NativeIterator::from_items(items))
}

// ── ES2025 Set composition methods ────────────────────────────────────────────

/// ECMAScript §24.2.3.12 `Set.prototype.union(other)`.
///
/// Returns a new `JsSet` containing all values from both `self` and `other`,
/// preserving insertion order (elements of `self` first, then new elements
/// from `other`).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_union, set_size, set_has};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// set_add(&mut a, JsValue::Smi(2));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(2));
/// set_add(&mut b, JsValue::Smi(3));
/// let u = set_union(&a, &b);
/// assert_eq!(set_size(&u), 3);
/// assert!(set_has(&u, &JsValue::Smi(1)));
/// assert!(set_has(&u, &JsValue::Smi(3)));
/// ```
pub fn set_union(a: &JsSet, b: &JsSet) -> JsSet {
    let mut result = a.clone();
    for v in &b.values {
        if !set_has(&result, v) {
            result.values.push(v.clone());
        }
    }
    result
}

/// ECMAScript §24.2.3.7 `Set.prototype.intersection(other)`.
///
/// Returns a new `JsSet` containing only values present in both sets.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_intersection, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// set_add(&mut a, JsValue::Smi(2));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(2));
/// set_add(&mut b, JsValue::Smi(3));
/// let i = set_intersection(&a, &b);
/// assert_eq!(set_size(&i), 1);
/// ```
pub fn set_intersection(a: &JsSet, b: &JsSet) -> JsSet {
    let mut result = set_new();
    for v in &a.values {
        if set_has(b, v) {
            result.values.push(v.clone());
        }
    }
    result
}

/// ECMAScript §24.2.3.3 `Set.prototype.difference(other)`.
///
/// Returns a new `JsSet` containing values in `a` that are not in `b`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_difference, set_size, set_has};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// set_add(&mut a, JsValue::Smi(2));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(2));
/// let d = set_difference(&a, &b);
/// assert_eq!(set_size(&d), 1);
/// assert!(set_has(&d, &JsValue::Smi(1)));
/// ```
pub fn set_difference(a: &JsSet, b: &JsSet) -> JsSet {
    let mut result = set_new();
    for v in &a.values {
        if !set_has(b, v) {
            result.values.push(v.clone());
        }
    }
    result
}

/// ECMAScript §24.2.3.11 `Set.prototype.symmetricDifference(other)`.
///
/// Returns a new `JsSet` containing values in either set but not both.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_symmetric_difference, set_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// set_add(&mut a, JsValue::Smi(2));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(2));
/// set_add(&mut b, JsValue::Smi(3));
/// let sd = set_symmetric_difference(&a, &b);
/// assert_eq!(set_size(&sd), 2);
/// ```
pub fn set_symmetric_difference(a: &JsSet, b: &JsSet) -> JsSet {
    let mut result = set_new();
    for v in &a.values {
        if !set_has(b, v) {
            result.values.push(v.clone());
        }
    }
    for v in &b.values {
        if !set_has(a, v) {
            result.values.push(v.clone());
        }
    }
    result
}

/// ECMAScript §24.2.3.9 `Set.prototype.isSubsetOf(other)`.
///
/// Returns `true` if every element of `a` is also in `b`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_is_subset_of};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(1));
/// set_add(&mut b, JsValue::Smi(2));
/// assert!(set_is_subset_of(&a, &b));
/// assert!(!set_is_subset_of(&b, &a));
/// ```
pub fn set_is_subset_of(a: &JsSet, b: &JsSet) -> bool {
    a.values.iter().all(|v| set_has(b, v))
}

/// ECMAScript §24.2.3.10 `Set.prototype.isSupersetOf(other)`.
///
/// Returns `true` if every element of `b` is also in `a`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_is_superset_of};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// set_add(&mut a, JsValue::Smi(2));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(1));
/// assert!(set_is_superset_of(&a, &b));
/// assert!(!set_is_superset_of(&b, &a));
/// ```
pub fn set_is_superset_of(a: &JsSet, b: &JsSet) -> bool {
    b.values.iter().all(|v| set_has(a, v))
}

/// ECMAScript §24.2.3.8 `Set.prototype.isDisjointFrom(other)`.
///
/// Returns `true` if the two sets have no elements in common.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::set::{set_new, set_add, set_is_disjoint_from};
/// use stator_core::objects::value::JsValue;
///
/// let mut a = set_new();
/// set_add(&mut a, JsValue::Smi(1));
/// let mut b = set_new();
/// set_add(&mut b, JsValue::Smi(2));
/// assert!(set_is_disjoint_from(&a, &b));
/// set_add(&mut b, JsValue::Smi(1));
/// assert!(!set_is_disjoint_from(&a, &b));
/// ```
pub fn set_is_disjoint_from(a: &JsSet, b: &JsSet) -> bool {
    !a.values.iter().any(|v| set_has(b, v))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── set_from_iterable ─────────────────────────────────────────────────

    #[test]
    fn test_set_from_iterable() {
        let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)];
        let s = set_from_iterable(items);
        assert_eq!(set_size(&s), 3);
        assert!(set_has(&s, &JsValue::Smi(1)));
        assert!(set_has(&s, &JsValue::Smi(3)));
    }

    #[test]
    fn test_set_from_iterable_deduplicates() {
        let items = vec![JsValue::Smi(1), JsValue::Smi(1), JsValue::Smi(2)];
        let s = set_from_iterable(items);
        assert_eq!(set_size(&s), 2);
    }

    // ── set_keys ──────────────────────────────────────────────────────────

    #[test]
    fn test_set_keys_equals_values() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(10));
        set_add(&mut s, JsValue::Smi(20));
        assert_eq!(set_keys(&s), set_values(&s));
    }

    // ── set_entries ───────────────────────────────────────────────────────

    #[test]
    fn test_set_entries_returns_value_value_pairs() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        assert_eq!(
            set_entries(&s),
            vec![
                (JsValue::Smi(1), JsValue::Smi(1)),
                (JsValue::Smi(2), JsValue::Smi(2)),
            ]
        );
    }

    // ── set_iter ──────────────────────────────────────────────────────────

    #[test]
    fn test_set_iter_returns_values() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(7));
        set_add(&mut s, JsValue::Smi(3));
        assert_eq!(set_iter(&s), set_values(&s));
    }

    // ── set_add / set_has / set_size ──────────────────────────────────────────

    #[test]
    fn test_set_add_unique_values() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        assert_eq!(set_size(&s), 2);
    }

    #[test]
    fn test_set_add_ignores_duplicates() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(1));
        assert_eq!(set_size(&s), 1);
    }

    #[test]
    fn test_set_has_present_value() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Boolean(false));
        assert!(set_has(&s, &JsValue::Boolean(false)));
    }

    #[test]
    fn test_set_has_absent_value() {
        let s = set_new();
        assert!(!set_has(&s, &JsValue::Null));
    }

    // ── set_delete ────────────────────────────────────────────────────────────

    #[test]
    fn test_set_delete_existing_value() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(42));
        assert!(set_delete(&mut s, &JsValue::Smi(42)));
        assert_eq!(set_size(&s), 0);
    }

    #[test]
    fn test_set_delete_missing_value_returns_false() {
        let mut s = set_new();
        assert!(!set_delete(&mut s, &JsValue::Smi(1)));
    }

    // ── set_clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_set_clear() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        set_clear(&mut s);
        assert_eq!(set_size(&s), 0);
    }

    // ── iteration order ───────────────────────────────────────────────────────

    #[test]
    fn test_set_insertion_order_preserved() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(3));
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        assert_eq!(
            set_values(&s),
            vec![JsValue::Smi(3), JsValue::Smi(1), JsValue::Smi(2)]
        );
    }

    // ── set_for_each ──────────────────────────────────────────────────────────

    #[test]
    fn test_set_for_each_visits_all_values() {
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(10));
        set_add(&mut s, JsValue::Smi(20));
        let mut out = Vec::new();
        set_for_each(&s, |v| out.push(v.clone()));
        assert_eq!(out, vec![JsValue::Smi(10), JsValue::Smi(20)]);
    }

    // ── SameValueZero edge cases ──────────────────────────────────────────────

    #[test]
    fn test_set_nan_deduplication() {
        let mut s = set_new();
        set_add(&mut s, JsValue::HeapNumber(f64::NAN));
        set_add(&mut s, JsValue::HeapNumber(f64::NAN));
        assert_eq!(set_size(&s), 1);
        assert!(set_has(&s, &JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_set_negative_zero_deduplication() {
        let mut s = set_new();
        set_add(&mut s, JsValue::HeapNumber(0.0_f64));
        set_add(&mut s, JsValue::HeapNumber(-0.0_f64));
        assert_eq!(set_size(&s), 1);
    }

    // ── set_create_iterator ───────────────────────────────────────────────────

    #[test]
    fn test_set_create_iterator_values() {
        use crate::builtins::iterator::iterator_next;
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(10));
        set_add(&mut s, JsValue::Smi(20));
        let iter = set_create_iterator(&s, SetIteratorKind::Values);
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(10));
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(20));
        assert!(iterator_next(&iter).unwrap().done);
    }

    #[test]
    fn test_set_create_iterator_keys_same_as_values() {
        use crate::builtins::iterator::iterator_next;
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(5));
        let keys_iter = set_create_iterator(&s, SetIteratorKind::Keys);
        let vals_iter = set_create_iterator(&s, SetIteratorKind::Values);
        assert_eq!(
            iterator_next(&keys_iter).unwrap().value,
            iterator_next(&vals_iter).unwrap().value
        );
    }

    #[test]
    fn test_set_create_iterator_entries() {
        use crate::builtins::iterator::iterator_next;
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        let iter = set_create_iterator(&s, SetIteratorKind::Entries);
        let r1 = iterator_next(&iter).unwrap();
        if let JsValue::Array(arr) = &r1.value {
            assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(1)]);
        } else {
            panic!("expected Array");
        }
        let r2 = iterator_next(&iter).unwrap();
        if let JsValue::Array(arr) = &r2.value {
            assert_eq!(*arr.borrow(), vec![JsValue::Smi(2), JsValue::Smi(2)]);
        } else {
            panic!("expected Array");
        }
        assert!(iterator_next(&iter).unwrap().done);
    }

    #[test]
    fn test_set_create_iterator_empty() {
        use crate::builtins::iterator::iterator_next;
        let s = set_new();
        let iter = set_create_iterator(&s, SetIteratorKind::Values);
        assert!(iterator_next(&iter).unwrap().done);
    }

    #[test]
    fn test_set_create_iterator_insertion_order() {
        use crate::builtins::iterator::iterator_next;
        let mut s = set_new();
        set_add(&mut s, JsValue::Smi(3));
        set_add(&mut s, JsValue::Smi(1));
        set_add(&mut s, JsValue::Smi(2));
        let iter = set_create_iterator(&s, SetIteratorKind::Values);
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(3));
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(1));
        assert_eq!(iterator_next(&iter).unwrap().value, JsValue::Smi(2));
        assert!(iterator_next(&iter).unwrap().done);
    }
}

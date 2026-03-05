//! ECMAScript §24.1 `Map` built-in.
//!
//! Provides [`JsMap`], a key-value collection that preserves insertion order and
//! uses the **SameValueZero** comparison algorithm for key equality (identical
//! to `===` except that `NaN` is considered equal to itself and `-0` is
//! considered equal to `+0`).
//!
//! # Naming convention
//!
//! Each function is prefixed `map_` to avoid ambiguity with similarly-named
//! standard-library items.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §24.1 — *The Map Objects*

use crate::objects::value::JsValue;

use super::util::same_value_zero;

// ── JsMap ─────────────────────────────────────────────────────────────────────

/// A JavaScript `Map` object per ECMAScript §24.1.
///
/// Entries are stored in a [`Vec`] in insertion order, matching the ECMAScript
/// requirement that `Map` operations iterate in insertion order.  Key lookup
/// uses a linear scan with [`same_value_zero`] comparison, which is correct
/// for all `JsValue` kinds.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_get, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::String("key".into()), JsValue::Smi(42));
/// assert_eq!(map_size(&m), 1);
/// assert_eq!(map_get(&m, &JsValue::String("key".into())), JsValue::Smi(42));
/// ```
#[derive(Debug, Clone, Default)]
pub struct JsMap {
    /// Entries stored in insertion order.  Deleted entries are removed in-place.
    entries: Vec<(JsValue, JsValue)>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

/// ECMAScript §24.1.1.1 `new Map()`.
///
/// Creates an empty [`JsMap`].
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_size};
///
/// let m = map_new();
/// assert_eq!(map_size(&m), 0);
/// ```
pub fn map_new() -> JsMap {
    JsMap::default()
}

/// ECMAScript §24.1.1.1 `new Map(iterable)`.
///
/// Creates a [`JsMap`] from an iterable of `[key, value]` pairs.  Each pair
/// is a two-element slice.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_from_iterable, map_get, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let pairs = vec![
///     (JsValue::Smi(1), JsValue::String("one".into())),
///     (JsValue::Smi(2), JsValue::String("two".into())),
/// ];
/// let m = map_from_iterable(pairs);
/// assert_eq!(map_size(&m), 2);
/// assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::String("one".into()));
/// ```
pub fn map_from_iterable(pairs: Vec<(JsValue, JsValue)>) -> JsMap {
    let mut m = map_new();
    for (k, v) in pairs {
        map_set(&mut m, k, v);
    }
    m
}

// ── map_set ───────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.9 `Map.prototype.set(key, value)`.
///
/// If an entry with a key that is `SameValueZero`-equal to `key` already
/// exists it is overwritten in-place (preserving its position in the iteration
/// order).  Otherwise a new entry is appended.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_get, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::Boolean(true));
/// // Updating an existing key does not change size.
/// map_set(&mut m, JsValue::Smi(1), JsValue::Boolean(false));
/// assert_eq!(map_size(&m), 1);
/// assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::Boolean(false));
/// ```
pub fn map_set(map: &mut JsMap, key: JsValue, value: JsValue) {
    if let Some(entry) = map
        .entries
        .iter_mut()
        .find(|(k, _)| same_value_zero(k, &key))
    {
        entry.1 = value;
    } else {
        map.entries.push((key, value));
    }
}

// ── map_get ───────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.6 `Map.prototype.get(key)`.
///
/// Returns the value associated with `key`, or [`JsValue::Undefined`] if no
/// such entry exists.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_get};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// assert_eq!(map_get(&m, &JsValue::Smi(99)), JsValue::Undefined);
/// map_set(&mut m, JsValue::Smi(99), JsValue::String("hi".into()));
/// assert_eq!(map_get(&m, &JsValue::Smi(99)), JsValue::String("hi".into()));
/// ```
pub fn map_get(map: &JsMap, key: &JsValue) -> JsValue {
    map.entries
        .iter()
        .find(|(k, _)| same_value_zero(k, key))
        .map(|(_, v)| v.clone())
        .unwrap_or(JsValue::Undefined)
}

// ── map_has ───────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.7 `Map.prototype.has(key)`.
///
/// Returns `true` if an entry whose key is `SameValueZero`-equal to `key`
/// exists in the map.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_has};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// assert!(!map_has(&m, &JsValue::Null));
/// map_set(&mut m, JsValue::Null, JsValue::Smi(0));
/// assert!(map_has(&m, &JsValue::Null));
/// ```
pub fn map_has(map: &JsMap, key: &JsValue) -> bool {
    map.entries.iter().any(|(k, _)| same_value_zero(k, key))
}

// ── map_delete ────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.3 `Map.prototype.delete(key)`.
///
/// Removes the entry whose key is `SameValueZero`-equal to `key`.  Returns
/// `true` if an entry was removed, `false` if `key` was not found.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_delete, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::Smi(10));
/// assert!(map_delete(&mut m, &JsValue::Smi(1)));
/// assert!(!map_delete(&mut m, &JsValue::Smi(1)));
/// assert_eq!(map_size(&m), 0);
/// ```
pub fn map_delete(map: &mut JsMap, key: &JsValue) -> bool {
    if let Some(pos) = map
        .entries
        .iter()
        .position(|(k, _)| same_value_zero(k, key))
    {
        map.entries.remove(pos);
        true
    } else {
        false
    }
}

// ── map_clear ─────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.2 `Map.prototype.clear()`.
///
/// Removes all entries from the map.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_clear, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::Smi(2));
/// map_clear(&mut m);
/// assert_eq!(map_size(&m), 0);
/// ```
pub fn map_clear(map: &mut JsMap) {
    map.entries.clear();
}

// ── map_size ──────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.10 `Map.prototype.size` (getter).
///
/// Returns the number of entries currently in the map.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_size};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// assert_eq!(map_size(&m), 0);
/// map_set(&mut m, JsValue::Smi(1), JsValue::Smi(1));
/// assert_eq!(map_size(&m), 1);
/// ```
pub fn map_size(map: &JsMap) -> usize {
    map.entries.len()
}

// ── map_for_each ──────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.5 `Map.prototype.forEach(callback)`.
///
/// Calls `callback(value, key)` for each entry in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_for_each};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::String("one".into()));
/// let mut out = Vec::new();
/// map_for_each(&m, |v, k| out.push((k.clone(), v.clone())));
/// assert_eq!(out, vec![(JsValue::Smi(1), JsValue::String("one".into()))]);
/// ```
pub fn map_for_each(map: &JsMap, mut callback: impl FnMut(&JsValue, &JsValue)) {
    for (k, v) in &map.entries {
        callback(v, k);
    }
}

// ── map_keys ──────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.8 `Map.prototype.keys()`.
///
/// Returns the keys of all entries in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_keys};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::Undefined);
/// map_set(&mut m, JsValue::Smi(2), JsValue::Undefined);
/// assert_eq!(map_keys(&m), vec![JsValue::Smi(1), JsValue::Smi(2)]);
/// ```
pub fn map_keys(map: &JsMap) -> Vec<JsValue> {
    map.entries.iter().map(|(k, _)| k.clone()).collect()
}

// ── map_values ────────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.11 `Map.prototype.values()`.
///
/// Returns the values of all entries in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_values};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(0), JsValue::String("a".into()));
/// map_set(&mut m, JsValue::Smi(1), JsValue::String("b".into()));
/// assert_eq!(
///     map_values(&m),
///     vec![JsValue::String("a".into()), JsValue::String("b".into())]
/// );
/// ```
pub fn map_values(map: &JsMap) -> Vec<JsValue> {
    map.entries.iter().map(|(_, v)| v.clone()).collect()
}

// ── map_entries ───────────────────────────────────────────────────────────────

/// ECMAScript §24.1.3.4 `Map.prototype.entries()`.
///
/// Returns all `(key, value)` pairs in insertion order.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_entries};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Boolean(true), JsValue::Smi(1));
/// assert_eq!(
///     map_entries(&m),
///     vec![(JsValue::Boolean(true), JsValue::Smi(1))]
/// );
/// ```
pub fn map_entries(map: &JsMap) -> Vec<(JsValue, JsValue)> {
    map.entries.clone()
}

/// ECMAScript §24.1.3 `Map.prototype[@@iterator]()`.
///
/// Returns all `(key, value)` pairs in insertion order.  This is the same as
/// [`map_entries`] and fulfils the `Symbol.iterator` protocol for `Map`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::map::{map_new, map_set, map_iter};
/// use stator_core::objects::value::JsValue;
///
/// let mut m = map_new();
/// map_set(&mut m, JsValue::Smi(1), JsValue::Boolean(true));
/// let pairs = map_iter(&m);
/// assert_eq!(pairs.len(), 1);
/// ```
pub fn map_iter(map: &JsMap) -> Vec<(JsValue, JsValue)> {
    map_entries(map)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── map_from_iterable ─────────────────────────────────────────────────

    #[test]
    fn test_map_from_iterable() {
        let pairs = vec![
            (JsValue::Smi(1), JsValue::String("a".into())),
            (JsValue::Smi(2), JsValue::String("b".into())),
        ];
        let m = map_from_iterable(pairs);
        assert_eq!(map_size(&m), 2);
        assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::String("a".into()));
        assert_eq!(map_get(&m, &JsValue::Smi(2)), JsValue::String("b".into()));
    }

    #[test]
    fn test_map_from_iterable_deduplicates() {
        let pairs = vec![
            (JsValue::Smi(1), JsValue::String("first".into())),
            (JsValue::Smi(1), JsValue::String("second".into())),
        ];
        let m = map_from_iterable(pairs);
        assert_eq!(map_size(&m), 1);
        assert_eq!(
            map_get(&m, &JsValue::Smi(1)),
            JsValue::String("second".into())
        );
    }

    // ── map_iter ──────────────────────────────────────────────────────────

    #[test]
    fn test_map_iter_returns_entries() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::Smi(10));
        map_set(&mut m, JsValue::Smi(2), JsValue::Smi(20));
        let entries = map_iter(&m);
        assert_eq!(
            entries,
            vec![
                (JsValue::Smi(1), JsValue::Smi(10)),
                (JsValue::Smi(2), JsValue::Smi(20)),
            ]
        );
    }

    // ── map_set / map_get / map_size ──────────────────────────────────────────

    #[test]
    fn test_map_set_and_get() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::String("one".into()));
        assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::String("one".into()));
    }

    #[test]
    fn test_map_get_missing_returns_undefined() {
        let m = map_new();
        assert_eq!(map_get(&m, &JsValue::Smi(99)), JsValue::Undefined);
    }

    #[test]
    fn test_map_set_overwrites_existing_key() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::Boolean(true));
        map_set(&mut m, JsValue::Smi(1), JsValue::Boolean(false));
        assert_eq!(map_size(&m), 1);
        assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::Boolean(false));
    }

    // ── map_has ───────────────────────────────────────────────────────────────

    #[test]
    fn test_map_has_existing_key() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Null, JsValue::Smi(0));
        assert!(map_has(&m, &JsValue::Null));
    }

    #[test]
    fn test_map_has_missing_key() {
        let m = map_new();
        assert!(!map_has(&m, &JsValue::Boolean(true)));
    }

    // ── map_delete ────────────────────────────────────────────────────────────

    #[test]
    fn test_map_delete_existing_key() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(5), JsValue::Smi(50));
        assert!(map_delete(&mut m, &JsValue::Smi(5)));
        assert_eq!(map_size(&m), 0);
        assert!(!map_has(&m, &JsValue::Smi(5)));
    }

    #[test]
    fn test_map_delete_missing_key_returns_false() {
        let mut m = map_new();
        assert!(!map_delete(&mut m, &JsValue::Smi(1)));
    }

    // ── map_clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_map_clear() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::Smi(1));
        map_set(&mut m, JsValue::Smi(2), JsValue::Smi(2));
        map_clear(&mut m);
        assert_eq!(map_size(&m), 0);
    }

    // ── iteration order ───────────────────────────────────────────────────────

    #[test]
    fn test_map_insertion_order_preserved() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(3), JsValue::Undefined);
        map_set(&mut m, JsValue::Smi(1), JsValue::Undefined);
        map_set(&mut m, JsValue::Smi(2), JsValue::Undefined);
        assert_eq!(
            map_keys(&m),
            vec![JsValue::Smi(3), JsValue::Smi(1), JsValue::Smi(2)]
        );
    }

    #[test]
    fn test_map_update_preserves_insertion_position() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::Smi(10));
        map_set(&mut m, JsValue::Smi(2), JsValue::Smi(20));
        map_set(&mut m, JsValue::Smi(1), JsValue::Smi(99)); // update, not re-insert
        assert_eq!(map_keys(&m), vec![JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(map_get(&m, &JsValue::Smi(1)), JsValue::Smi(99));
    }

    // ── map_for_each ──────────────────────────────────────────────────────────

    #[test]
    fn test_map_for_each_visits_all_entries() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(1), JsValue::String("a".into()));
        map_set(&mut m, JsValue::Smi(2), JsValue::String("b".into()));
        let mut pairs: Vec<(JsValue, JsValue)> = Vec::new();
        map_for_each(&m, |v, k| pairs.push((k.clone(), v.clone())));
        assert_eq!(
            pairs,
            vec![
                (JsValue::Smi(1), JsValue::String("a".into())),
                (JsValue::Smi(2), JsValue::String("b".into())),
            ]
        );
    }

    // ── map_values / map_entries ──────────────────────────────────────────────

    #[test]
    fn test_map_values() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Smi(0), JsValue::String("x".into()));
        map_set(&mut m, JsValue::Smi(1), JsValue::String("y".into()));
        assert_eq!(
            map_values(&m),
            vec![JsValue::String("x".into()), JsValue::String("y".into())]
        );
    }

    #[test]
    fn test_map_entries() {
        let mut m = map_new();
        map_set(&mut m, JsValue::Boolean(true), JsValue::Smi(1));
        map_set(&mut m, JsValue::Boolean(false), JsValue::Smi(0));
        assert_eq!(
            map_entries(&m),
            vec![
                (JsValue::Boolean(true), JsValue::Smi(1)),
                (JsValue::Boolean(false), JsValue::Smi(0)),
            ]
        );
    }

    // ── SameValueZero edge cases ──────────────────────────────────────────────

    #[test]
    fn test_map_nan_key_equal_to_nan() {
        let mut m = map_new();
        map_set(&mut m, JsValue::HeapNumber(f64::NAN), JsValue::Smi(1));
        assert!(map_has(&m, &JsValue::HeapNumber(f64::NAN)));
        assert_eq!(map_get(&m, &JsValue::HeapNumber(f64::NAN)), JsValue::Smi(1));
    }

    #[test]
    fn test_map_negative_zero_equals_positive_zero() {
        let mut m = map_new();
        map_set(&mut m, JsValue::HeapNumber(0.0_f64), JsValue::Smi(7));
        assert!(map_has(&m, &JsValue::HeapNumber(-0.0_f64)));
        assert_eq!(map_size(&m), 1);
    }
}

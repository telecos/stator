//! ECMAScript §24.3 `WeakMap` built-in.
//!
//! Provides [`JsWeakMap`], a key-value collection whose keys are **object
//! references** (raw `*mut HeapObject` pointers).  Entries are keyed by
//! pointer identity (address), not by value equality.
//!
//! # Ephemeron GC semantics
//!
//! A `WeakMap` entry does **not** prevent its key from being garbage-collected.
//! When the GC determines that a key object is no longer reachable from any
//! strong root, it calls [`weak_map_remove_object`] to prune the entry.
//! Callers (i.e. the GC sweep phase) are responsible for invoking this hook
//! for every collected object.
//!
//! No enumeration of keys or values is provided; the ECMAScript specification
//! intentionally omits iteration to preserve the weak semantics.
//!
//! # Safety
//!
//! All functions that accept a `*mut HeapObject` key require that the pointer
//! is either null (which is rejected as invalid) or points to a live,
//! properly-aligned [`HeapObject`] that is managed by the engine heap.  The
//! caller must ensure pointer validity; storing a dangling key pointer leads to
//! undefined behaviour when the GC sweep hook is not invoked before the object
//! is freed.
//!
//! # Naming convention
//!
//! Each function is prefixed `weak_map_` to avoid ambiguity.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §24.3 — *The WeakMap Objects*

use std::collections::HashMap;

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;
use crate::objects::value::JsValue;

// ── JsWeakMap ──────────────────────────────────────────────────────────────────

/// A JavaScript `WeakMap` object per ECMAScript §24.3.
///
/// Keys are raw `*mut HeapObject` pointers stored as `usize` in a
/// [`HashMap`].  This allows O(1) average-case lookup while keeping the
/// collection's keys invisible to the GC tracer (weak semantics).
///
/// Call [`weak_map_remove_object`] during the GC sweep phase to implement
/// ephemeron clearing.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{weak_map_new, weak_map_set, weak_map_has};
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// let key = &raw mut obj;
/// weak_map_set(&mut wm, key, JsValue::Smi(1)).unwrap();
/// assert!(weak_map_has(&wm, key));
/// ```
#[derive(Debug, Default)]
pub struct JsWeakMap {
    /// Entries keyed by the raw address of the heap object (`usize`).
    entries: HashMap<usize, JsValue>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

/// ECMAScript §24.3.1.1 `new WeakMap()`.
///
/// Creates an empty [`JsWeakMap`].
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::weak_map_new;
///
/// let wm = weak_map_new();
/// ```
pub fn weak_map_new() -> JsWeakMap {
    JsWeakMap::default()
}

/// ECMAScript §24.3.1.1 `new WeakMap(iterable)`.
///
/// Creates a [`JsWeakMap`] from an iterable of `(key, value)` pairs.
///
/// Returns [`StatorError::TypeError`] if any key is null.
pub fn weak_map_from_iterable(entries: Vec<(*mut HeapObject, JsValue)>) -> StatorResult<JsWeakMap> {
    let mut map = weak_map_new();
    for (key, value) in entries {
        weak_map_set(&mut map, key, value)?;
    }
    Ok(map)
}

// ── weak_map_set ──────────────────────────────────────────────────────────────

/// ECMAScript §24.3.3.5 `WeakMap.prototype.set(key, value)`.
///
/// Associates `value` with `key`.  If an entry for `key` already exists it is
/// overwritten.
///
/// Returns [`StatorError::TypeError`] if `key` is null (null pointers cannot
/// be used as `WeakMap` keys).
///
/// # Safety
///
/// `key` must either be null (rejected) or point to a live, aligned
/// [`HeapObject`] managed by the engine heap.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{weak_map_new, weak_map_set, weak_map_get};
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// let key = &raw mut obj;
/// weak_map_set(&mut wm, key, JsValue::Smi(42)).unwrap();
/// assert_eq!(weak_map_get(&wm, key), JsValue::Smi(42));
/// ```
pub fn weak_map_set(map: &mut JsWeakMap, key: *mut HeapObject, value: JsValue) -> StatorResult<()> {
    if key.is_null() {
        return Err(StatorError::TypeError(
            "WeakMap key must be an object".into(),
        ));
    }
    map.entries.insert(key as usize, value);
    Ok(())
}

// ── weak_map_get ──────────────────────────────────────────────────────────────

/// ECMAScript §24.3.3.3 `WeakMap.prototype.get(key)`.
///
/// Returns the value associated with `key`, or [`JsValue::Undefined`] if no
/// entry exists (or `key` is null).
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{weak_map_new, weak_map_get};
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// assert_eq!(weak_map_get(&wm, &raw mut obj), JsValue::Undefined);
/// ```
pub fn weak_map_get(map: &JsWeakMap, key: *mut HeapObject) -> JsValue {
    if key.is_null() {
        return JsValue::Undefined;
    }
    map.entries
        .get(&(key as usize))
        .cloned()
        .unwrap_or(JsValue::Undefined)
}

// ── weak_map_has ──────────────────────────────────────────────────────────────

/// ECMAScript §24.3.3.4 `WeakMap.prototype.has(key)`.
///
/// Returns `true` if an entry for `key` exists.  Returns `false` for null
/// keys.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{weak_map_new, weak_map_set, weak_map_has};
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// let key = &raw mut obj;
/// assert!(!weak_map_has(&wm, key));
/// weak_map_set(&mut wm, key, JsValue::Undefined).unwrap();
/// assert!(weak_map_has(&wm, key));
/// ```
pub fn weak_map_has(map: &JsWeakMap, key: *mut HeapObject) -> bool {
    if key.is_null() {
        return false;
    }
    map.entries.contains_key(&(key as usize))
}

// ── weak_map_delete ───────────────────────────────────────────────────────────

/// ECMAScript §24.3.3.2 `WeakMap.prototype.delete(key)`.
///
/// Removes the entry for `key`.  Returns `true` if an entry was removed,
/// `false` if `key` was not found or was null.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{weak_map_new, weak_map_set, weak_map_delete};
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// let key = &raw mut obj;
/// weak_map_set(&mut wm, key, JsValue::Smi(1)).unwrap();
/// assert!(weak_map_delete(&mut wm, key));
/// assert!(!weak_map_delete(&mut wm, key));
/// ```
pub fn weak_map_delete(map: &mut JsWeakMap, key: *mut HeapObject) -> bool {
    if key.is_null() {
        return false;
    }
    map.entries.remove(&(key as usize)).is_some()
}

// ── weak_map_remove_object ────────────────────────────────────────────────────

/// GC ephemeron hook: remove the entry for `key` if it exists.
///
/// This function is called by the garbage collector sweep phase when `key` is
/// determined to be unreachable.  It clears the corresponding entry from the
/// `WeakMap`, implementing the ephemeron semantics required by ECMAScript
/// §24.3.
///
/// Calling this with a null pointer is safe and is a no-op.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::weak_map::{
///     weak_map_new, weak_map_set, weak_map_has, weak_map_remove_object,
/// };
/// use stator_jse::objects::heap_object::HeapObject;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wm = weak_map_new();
/// let mut obj = HeapObject::new_null();
/// let key = &raw mut obj;
/// weak_map_set(&mut wm, key, JsValue::Boolean(true)).unwrap();
/// // Simulate GC collecting `obj`.
/// weak_map_remove_object(&mut wm, key);
/// assert!(!weak_map_has(&wm, key));
/// ```
pub fn weak_map_remove_object(map: &mut JsWeakMap, key: *mut HeapObject) {
    if !key.is_null() {
        map.entries.remove(&(key as usize));
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CRUD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_weak_map_set_and_get() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        let key = &raw mut obj;
        weak_map_set(&mut wm, key, JsValue::Smi(99)).unwrap();
        assert_eq!(weak_map_get(&wm, key), JsValue::Smi(99));
    }

    #[test]
    fn test_weak_map_get_missing_returns_undefined() {
        let wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        assert_eq!(weak_map_get(&wm, &raw mut obj), JsValue::Undefined);
    }

    #[test]
    fn test_weak_map_has() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        let key = &raw mut obj;
        assert!(!weak_map_has(&wm, key));
        weak_map_set(&mut wm, key, JsValue::Undefined).unwrap();
        assert!(weak_map_has(&wm, key));
    }

    #[test]
    fn test_weak_map_delete_existing() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        let key = &raw mut obj;
        weak_map_set(&mut wm, key, JsValue::Smi(1)).unwrap();
        assert!(weak_map_delete(&mut wm, key));
        assert!(!weak_map_has(&wm, key));
    }

    #[test]
    fn test_weak_map_delete_missing_returns_false() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        assert!(!weak_map_delete(&mut wm, &raw mut obj));
    }

    #[test]
    fn test_weak_map_null_key_returns_error() {
        let mut wm = weak_map_new();
        assert!(weak_map_set(&mut wm, std::ptr::null_mut(), JsValue::Smi(1)).is_err());
    }

    #[test]
    fn test_weak_map_null_key_get_returns_undefined() {
        let wm = weak_map_new();
        assert_eq!(weak_map_get(&wm, std::ptr::null_mut()), JsValue::Undefined);
    }

    #[test]
    fn test_weak_map_null_key_has_returns_false() {
        let wm = weak_map_new();
        assert!(!weak_map_has(&wm, std::ptr::null_mut()));
    }

    #[test]
    fn test_weak_map_null_key_delete_returns_false() {
        let mut wm = weak_map_new();
        assert!(!weak_map_delete(&mut wm, std::ptr::null_mut()));
    }

    #[test]
    fn test_weak_map_from_iterable_populates_entries() {
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let wm = weak_map_from_iterable(vec![
            (&raw mut a, JsValue::Smi(1)),
            (&raw mut b, JsValue::Smi(2)),
        ])
        .unwrap();
        assert_eq!(weak_map_get(&wm, &raw mut a), JsValue::Smi(1));
        assert_eq!(weak_map_get(&wm, &raw mut b), JsValue::Smi(2));
    }

    #[test]
    fn test_weak_map_from_iterable_rejects_null_key() {
        let err =
            weak_map_from_iterable(vec![(std::ptr::null_mut(), JsValue::Smi(1))]).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── Pointer identity ──────────────────────────────────────────────────────

    #[test]
    fn test_weak_map_keys_are_distinct_by_address() {
        let mut wm = weak_map_new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let ka = &raw mut a;
        let kb = &raw mut b;
        weak_map_set(&mut wm, ka, JsValue::Smi(1)).unwrap();
        weak_map_set(&mut wm, kb, JsValue::Smi(2)).unwrap();
        assert_eq!(weak_map_get(&wm, ka), JsValue::Smi(1));
        assert_eq!(weak_map_get(&wm, kb), JsValue::Smi(2));
    }

    // ── Ephemeron / GC hook ────────────────────────────────────────────────────

    #[test]
    fn test_weak_map_remove_object_clears_entry() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        let key = &raw mut obj;
        weak_map_set(&mut wm, key, JsValue::Boolean(true)).unwrap();
        weak_map_remove_object(&mut wm, key);
        assert!(!weak_map_has(&wm, key));
        assert_eq!(weak_map_get(&wm, key), JsValue::Undefined);
    }

    #[test]
    fn test_weak_map_remove_object_null_is_noop() {
        let mut wm = weak_map_new();
        // Should not panic.
        weak_map_remove_object(&mut wm, std::ptr::null_mut());
    }

    #[test]
    fn test_weak_map_overwrite_existing_key() {
        let mut wm = weak_map_new();
        let mut obj = HeapObject::new_null();
        let key = &raw mut obj;
        assert!(weak_map_set(&mut wm, key, JsValue::Smi(1)).is_ok());
        assert!(weak_map_set(&mut wm, key, JsValue::Smi(2)).is_ok());
        assert_eq!(weak_map_get(&wm, key), JsValue::Smi(2));
    }
}

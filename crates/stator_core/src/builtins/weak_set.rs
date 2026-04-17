//! ECMAScript §24.4 `WeakSet` built-in.
//!
//! Provides [`JsWeakSet`], a collection of **object references**
//! (`*mut HeapObject` pointers) held weakly.  Membership is determined by
//! pointer identity (address), not by value equality.
//!
//! # Ephemeron GC semantics
//!
//! A `WeakSet` entry does **not** prevent its object from being
//! garbage-collected.  When the GC determines that an object is no longer
//! reachable from any strong root, it calls [`weak_set_remove_object`] to
//! prune the entry.  Callers (i.e. the GC sweep phase) are responsible for
//! invoking this hook for every collected object.
//!
//! No enumeration of values is provided; the ECMAScript specification
//! intentionally omits iteration to preserve the weak semantics.
//!
//! # Safety
//!
//! All functions that accept a `*mut HeapObject` argument require that the
//! pointer is either null (which is rejected as invalid) or points to a live,
//! properly-aligned [`HeapObject`] managed by the engine heap.  The caller
//! must ensure pointer validity; storing a dangling pointer leads to undefined
//! behaviour if the GC sweep hook is not invoked before the object is freed.
//!
//! # Naming convention
//!
//! Each function is prefixed `weak_set_` to avoid ambiguity.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §24.4 — *The WeakSet Objects*

use std::collections::HashSet;

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;

// ── JsWeakSet ──────────────────────────────────────────────────────────────────

/// A JavaScript `WeakSet` object per ECMAScript §24.4.
///
/// Members are raw `*mut HeapObject` pointers stored as `usize` in a
/// [`HashSet`], allowing O(1) average-case membership tests while keeping the
/// pointers invisible to the GC tracer (weak semantics).
///
/// Call [`weak_set_remove_object`] during the GC sweep phase to implement
/// ephemeron clearing.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::{weak_set_new, weak_set_add, weak_set_has};
/// use stator_js::objects::heap_object::HeapObject;
///
/// let mut ws = weak_set_new();
/// let mut obj = HeapObject::new_null();
/// let ptr = &raw mut obj;
/// weak_set_add(&mut ws, ptr).unwrap();
/// assert!(weak_set_has(&ws, ptr));
/// ```
#[derive(Debug, Default)]
pub struct JsWeakSet {
    /// Members stored as raw address values (`usize`).
    entries: HashSet<usize>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

/// ECMAScript §24.4.1.1 `new WeakSet()`.
///
/// Creates an empty [`JsWeakSet`].
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::weak_set_new;
///
/// let ws = weak_set_new();
/// ```
pub fn weak_set_new() -> JsWeakSet {
    JsWeakSet::default()
}

/// ECMAScript §24.4.1.1 `new WeakSet(iterable)`.
///
/// Creates a [`JsWeakSet`] from an iterable of object pointers.
///
/// Returns [`StatorError::TypeError`] if any value is null.
pub fn weak_set_from_iterable(values: Vec<*mut HeapObject>) -> StatorResult<JsWeakSet> {
    let mut set = weak_set_new();
    for value in values {
        weak_set_add(&mut set, value)?;
    }
    Ok(set)
}

// ── weak_set_add ──────────────────────────────────────────────────────────────

/// ECMAScript §24.4.3.1 `WeakSet.prototype.add(value)`.
///
/// Adds `value` to the set.  If `value` is already a member this is a no-op.
///
/// Returns [`StatorError::TypeError`] if `value` is null (null pointers cannot
/// be used as `WeakSet` members).
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::{weak_set_new, weak_set_add, weak_set_has};
/// use stator_js::objects::heap_object::HeapObject;
///
/// let mut ws = weak_set_new();
/// let mut obj = HeapObject::new_null();
/// let ptr = &raw mut obj;
/// weak_set_add(&mut ws, ptr).unwrap();
/// assert!(weak_set_has(&ws, ptr));
/// ```
pub fn weak_set_add(set: &mut JsWeakSet, value: *mut HeapObject) -> StatorResult<()> {
    if value.is_null() {
        return Err(StatorError::TypeError(
            "WeakSet value must be an object".into(),
        ));
    }
    set.entries.insert(value as usize);
    Ok(())
}

// ── weak_set_has ──────────────────────────────────────────────────────────────

/// ECMAScript §24.4.3.4 `WeakSet.prototype.has(value)`.
///
/// Returns `true` if `value` is a member of the set.  Returns `false` for
/// null pointers.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::{weak_set_new, weak_set_add, weak_set_has};
/// use stator_js::objects::heap_object::HeapObject;
///
/// let mut ws = weak_set_new();
/// let mut obj = HeapObject::new_null();
/// let ptr = &raw mut obj;
/// assert!(!weak_set_has(&ws, ptr));
/// weak_set_add(&mut ws, ptr).unwrap();
/// assert!(weak_set_has(&ws, ptr));
/// ```
pub fn weak_set_has(set: &JsWeakSet, value: *mut HeapObject) -> bool {
    if value.is_null() {
        return false;
    }
    set.entries.contains(&(value as usize))
}

// ── weak_set_delete ───────────────────────────────────────────────────────────

/// ECMAScript §24.4.3.3 `WeakSet.prototype.delete(value)`.
///
/// Removes `value` from the set.  Returns `true` if `value` was a member,
/// `false` if it was not found or was null.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::{weak_set_new, weak_set_add, weak_set_delete};
/// use stator_js::objects::heap_object::HeapObject;
///
/// let mut ws = weak_set_new();
/// let mut obj = HeapObject::new_null();
/// let ptr = &raw mut obj;
/// weak_set_add(&mut ws, ptr).unwrap();
/// assert!(weak_set_delete(&mut ws, ptr));
/// assert!(!weak_set_delete(&mut ws, ptr));
/// ```
pub fn weak_set_delete(set: &mut JsWeakSet, value: *mut HeapObject) -> bool {
    if value.is_null() {
        return false;
    }
    set.entries.remove(&(value as usize))
}

// ── weak_set_remove_object ────────────────────────────────────────────────────

/// GC ephemeron hook: remove `value` from the set if it is a member.
///
/// This function is called by the garbage collector sweep phase when `value`
/// is determined to be unreachable.  It clears the corresponding entry from
/// the `WeakSet`, implementing the ephemeron semantics required by ECMAScript
/// §24.4.
///
/// Calling this with a null pointer is safe and is a no-op.
///
/// # Examples
///
/// ```
/// use stator_js::builtins::weak_set::{
///     weak_set_new, weak_set_add, weak_set_has, weak_set_remove_object,
/// };
/// use stator_js::objects::heap_object::HeapObject;
///
/// let mut ws = weak_set_new();
/// let mut obj = HeapObject::new_null();
/// let ptr = &raw mut obj;
/// weak_set_add(&mut ws, ptr).unwrap();
/// // Simulate GC collecting `obj`.
/// weak_set_remove_object(&mut ws, ptr);
/// assert!(!weak_set_has(&ws, ptr));
/// ```
pub fn weak_set_remove_object(set: &mut JsWeakSet, value: *mut HeapObject) {
    if !value.is_null() {
        set.entries.remove(&(value as usize));
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CRUD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_weak_set_add_and_has() {
        let mut ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(weak_set_add(&mut ws, ptr).is_ok());
        assert!(weak_set_has(&ws, ptr));
    }

    #[test]
    fn test_weak_set_has_missing_returns_false() {
        let ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        assert!(!weak_set_has(&ws, &raw mut obj));
    }

    #[test]
    fn test_weak_set_add_duplicate_is_noop() {
        let mut ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        weak_set_add(&mut ws, ptr).unwrap();
        weak_set_add(&mut ws, ptr).unwrap(); // duplicate
        assert!(weak_set_has(&ws, ptr));
    }

    #[test]
    fn test_weak_set_delete_existing() {
        let mut ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        weak_set_add(&mut ws, ptr).unwrap();
        assert!(weak_set_delete(&mut ws, ptr));
        assert!(!weak_set_has(&ws, ptr));
    }

    #[test]
    fn test_weak_set_delete_missing_returns_false() {
        let mut ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        assert!(!weak_set_delete(&mut ws, &raw mut obj));
    }

    #[test]
    fn test_weak_set_null_add_returns_error() {
        let mut ws = weak_set_new();
        assert!(weak_set_add(&mut ws, std::ptr::null_mut()).is_err());
    }

    #[test]
    fn test_weak_set_null_has_returns_false() {
        let ws = weak_set_new();
        assert!(!weak_set_has(&ws, std::ptr::null_mut()));
    }

    #[test]
    fn test_weak_set_null_delete_returns_false() {
        let mut ws = weak_set_new();
        assert!(!weak_set_delete(&mut ws, std::ptr::null_mut()));
    }

    #[test]
    fn test_weak_set_from_iterable_populates_entries() {
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let ws = weak_set_from_iterable(vec![&raw mut a, &raw mut b]).unwrap();
        assert!(weak_set_has(&ws, &raw mut a));
        assert!(weak_set_has(&ws, &raw mut b));
    }

    #[test]
    fn test_weak_set_from_iterable_rejects_null_value() {
        let err = weak_set_from_iterable(vec![std::ptr::null_mut()]).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── Pointer identity ──────────────────────────────────────────────────────

    #[test]
    fn test_weak_set_members_are_distinct_by_address() {
        let mut ws = weak_set_new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let pa = &raw mut a;
        let pb = &raw mut b;
        weak_set_add(&mut ws, pa).unwrap();
        assert!(weak_set_has(&ws, pa));
        assert!(!weak_set_has(&ws, pb));
    }

    // ── Ephemeron / GC hook ────────────────────────────────────────────────────

    #[test]
    fn test_weak_set_remove_object_clears_entry() {
        let mut ws = weak_set_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        weak_set_add(&mut ws, ptr).unwrap();
        weak_set_remove_object(&mut ws, ptr);
        assert!(!weak_set_has(&ws, ptr));
    }

    #[test]
    fn test_weak_set_remove_object_null_is_noop() {
        let mut ws = weak_set_new();
        // Should not panic.
        weak_set_remove_object(&mut ws, std::ptr::null_mut());
    }
}

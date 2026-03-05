//! ECMAScript §26.1 `WeakRef` built-in.
//!
//! Provides [`JsWeakRef`], a weak reference to an object that does **not**
//! prevent garbage collection of the target.
//!
//! # Current limitations
//!
//! True GC-weak semantics require integration with the garbage collector
//! (issue #267).  For now the reference is held strongly; the API surface is
//! correct so that downstream code can rely on the interface while the GC
//! integration is completed separately.
//!
//! # GC integration
//!
//! When the GC determines that the target object is no longer strongly
//! reachable, it should call [`weak_ref_clear`] to set the internal slot to
//! `None`, after which [`weak_ref_deref`] will return
//! [`JsValue::Undefined`].
//!
//! # Safety
//!
//! All functions that accept a `*mut HeapObject` target require that the
//! pointer is either null (which is rejected as invalid) or points to a live,
//! properly-aligned [`HeapObject`] managed by the engine heap.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §26.1 — *WeakRef Objects*

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;
use crate::objects::value::JsValue;

// ── JsWeakRef ────────────────────────────────────────────────────────────────

/// A JavaScript `WeakRef` object per ECMAScript §26.1.
///
/// Holds an optional raw `*mut HeapObject` pointer.  When the target is
/// collected by the GC, the internal slot is cleared to `None`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::weak_ref::{weak_ref_new, weak_ref_deref};
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut obj = HeapObject::new_null();
/// let wr = weak_ref_new(&raw mut obj).unwrap();
/// assert_ne!(weak_ref_deref(&wr), JsValue::Undefined);
/// ```
#[derive(Debug)]
pub struct JsWeakRef {
    /// The weak target stored as a raw address, or `None` if collected.
    target: Option<*mut HeapObject>,
}

// ── Constructors ─────────────────────────────────────────────────────────────

/// ECMAScript §26.1.1.1 `new WeakRef(target)`.
///
/// Creates a new [`JsWeakRef`] holding a reference to `target`.
///
/// Returns [`StatorError::TypeError`] if `target` is null (only objects may
/// be weak-referenced).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::weak_ref::weak_ref_new;
/// use stator_core::objects::heap_object::HeapObject;
///
/// let mut obj = HeapObject::new_null();
/// let wr = weak_ref_new(&raw mut obj).unwrap();
/// ```
pub fn weak_ref_new(target: *mut HeapObject) -> StatorResult<JsWeakRef> {
    if target.is_null() {
        return Err(StatorError::TypeError(
            "WeakRef target must be an object".into(),
        ));
    }
    Ok(JsWeakRef {
        target: Some(target),
    })
}

// ── weak_ref_deref ───────────────────────────────────────────────────────────

/// ECMAScript §26.1.3.2 `WeakRef.prototype.deref()`.
///
/// Returns a [`JsValue::Object`] containing the target pointer if the target
/// has not been collected, or [`JsValue::Undefined`] if it has.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::weak_ref::{weak_ref_new, weak_ref_deref};
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut obj = HeapObject::new_null();
/// let wr = weak_ref_new(&raw mut obj).unwrap();
/// assert_ne!(weak_ref_deref(&wr), JsValue::Undefined);
/// ```
pub fn weak_ref_deref(wr: &JsWeakRef) -> JsValue {
    match wr.target {
        Some(ptr) => JsValue::Object(ptr),
        None => JsValue::Undefined,
    }
}

// ── weak_ref_clear ───────────────────────────────────────────────────────────

/// GC hook: clear the weak reference when the target is collected.
///
/// After this call, [`weak_ref_deref`] will return [`JsValue::Undefined`].
/// This function is called by the garbage collector sweep phase when the
/// target object is determined to be unreachable.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::weak_ref::{weak_ref_new, weak_ref_deref, weak_ref_clear};
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut obj = HeapObject::new_null();
/// let mut wr = weak_ref_new(&raw mut obj).unwrap();
/// weak_ref_clear(&mut wr);
/// assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
/// ```
pub fn weak_ref_clear(wr: &mut JsWeakRef) {
    wr.target = None;
}

// ── weak_ref_has_target ──────────────────────────────────────────────────────

/// Returns `true` if the weak reference still holds a live target.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::weak_ref::{weak_ref_new, weak_ref_has_target, weak_ref_clear};
/// use stator_core::objects::heap_object::HeapObject;
///
/// let mut obj = HeapObject::new_null();
/// let mut wr = weak_ref_new(&raw mut obj).unwrap();
/// assert!(weak_ref_has_target(&wr));
/// weak_ref_clear(&mut wr);
/// assert!(!weak_ref_has_target(&wr));
/// ```
pub fn weak_ref_has_target(wr: &JsWeakRef) -> bool {
    wr.target.is_some()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_new_with_valid_target() {
        let mut obj = HeapObject::new_null();
        let wr = weak_ref_new(&raw mut obj);
        assert!(wr.is_ok());
    }

    #[test]
    fn test_weak_ref_new_null_target_returns_error() {
        let result = weak_ref_new(std::ptr::null_mut());
        assert!(result.is_err());
    }

    #[test]
    fn test_weak_ref_new_null_error_is_type_error() {
        let err = weak_ref_new(std::ptr::null_mut()).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── Deref ────────────────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_deref_returns_object() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        let wr = weak_ref_new(ptr).unwrap();
        let val = weak_ref_deref(&wr);
        assert!(matches!(val, JsValue::Object(p) if p == ptr));
    }

    #[test]
    fn test_weak_ref_deref_after_clear_returns_undefined() {
        let mut obj = HeapObject::new_null();
        let mut wr = weak_ref_new(&raw mut obj).unwrap();
        weak_ref_clear(&mut wr);
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    #[test]
    fn test_weak_ref_deref_preserves_pointer_identity() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        let wr = weak_ref_new(ptr).unwrap();
        if let JsValue::Object(p) = weak_ref_deref(&wr) {
            assert_eq!(p, ptr);
        } else {
            panic!("deref should return Object");
        }
    }

    // ── Clear (GC hook) ──────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_clear_makes_deref_undefined() {
        let mut obj = HeapObject::new_null();
        let mut wr = weak_ref_new(&raw mut obj).unwrap();
        weak_ref_clear(&mut wr);
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    #[test]
    fn test_weak_ref_clear_idempotent() {
        let mut obj = HeapObject::new_null();
        let mut wr = weak_ref_new(&raw mut obj).unwrap();
        weak_ref_clear(&mut wr);
        weak_ref_clear(&mut wr); // second clear is a no-op
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    // ── has_target ───────────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_has_target_true_initially() {
        let mut obj = HeapObject::new_null();
        let wr = weak_ref_new(&raw mut obj).unwrap();
        assert!(weak_ref_has_target(&wr));
    }

    #[test]
    fn test_weak_ref_has_target_false_after_clear() {
        let mut obj = HeapObject::new_null();
        let mut wr = weak_ref_new(&raw mut obj).unwrap();
        weak_ref_clear(&mut wr);
        assert!(!weak_ref_has_target(&wr));
    }

    // ── Distinct targets ─────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_distinct_targets() {
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let pa = &raw mut a;
        let pb = &raw mut b;
        let wr_a = weak_ref_new(pa).unwrap();
        let wr_b = weak_ref_new(pb).unwrap();
        if let (JsValue::Object(ra), JsValue::Object(rb)) =
            (weak_ref_deref(&wr_a), weak_ref_deref(&wr_b))
        {
            assert_eq!(ra, pa);
            assert_eq!(rb, pb);
            assert_ne!(ra, rb);
        } else {
            panic!("both derefs should return Object");
        }
    }

    #[test]
    fn test_weak_ref_clear_one_does_not_affect_other() {
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let mut wr_a = weak_ref_new(&raw mut a).unwrap();
        let wr_b = weak_ref_new(&raw mut b).unwrap();
        weak_ref_clear(&mut wr_a);
        assert!(!weak_ref_has_target(&wr_a));
        assert!(weak_ref_has_target(&wr_b));
    }

    // ── Multiple derefs ──────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_deref_is_repeatable() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        let wr = weak_ref_new(ptr).unwrap();
        let v1 = weak_ref_deref(&wr);
        let v2 = weak_ref_deref(&wr);
        assert!(matches!(v1, JsValue::Object(p) if p == ptr));
        assert!(matches!(v2, JsValue::Object(p) if p == ptr));
    }
}

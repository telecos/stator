//! ECMAScript §26.1 `WeakRef` built-in.
//!
//! Provides [`JsWeakRef`], a weak reference to an object that does **not**
//! prevent garbage collection of the target.
//!
//! # Weak semantics
//!
//! For `PlainObject` targets (backed by `Rc<RefCell<PropertyMap>>`), this
//! module uses [`std::rc::Weak`] to hold a true weak reference.  When all
//! strong `Rc` handles are dropped, [`weak_ref_deref`] automatically returns
//! [`JsValue::Undefined`] — no external GC hook is required.
//!
//! For raw `*mut HeapObject` targets managed by the GC heap, the reference
//! is stored as a raw pointer.  The GC sweep phase should call
//! [`weak_ref_clear`] when the target becomes unreachable.
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

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── WeakTarget ───────────────────────────────────────────────────────────────

/// The internal representation of a weak reference target.
#[derive(Debug, Clone)]
pub(crate) enum WeakTarget {
    /// An Rc-managed `PlainObject`: uses `Weak<RefCell<PropertyMap>>` so that
    /// `deref()` returns `Undefined` once all strong references are dropped.
    Plain(Weak<RefCell<PropertyMap>>),
    /// A GC-managed `HeapObject`: raw pointer cleared by the GC sweep phase.
    Heap(Option<*mut HeapObject>),
}

// ── JsWeakRef ────────────────────────────────────────────────────────────────

/// A JavaScript `WeakRef` object per ECMAScript §26.1.
///
/// Supports both Rc-managed `PlainObject` targets (true weak semantics via
/// [`std::rc::Weak`]) and GC-managed `HeapObject` targets (manual clearing).
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
#[derive(Debug, Clone)]
pub struct JsWeakRef {
    /// The weak target, supporting both Rc-managed and GC-managed objects.
    target: WeakTarget,
}

// ── Constructors ─────────────────────────────────────────────────────────────

/// ECMAScript §26.1.1.1 `new WeakRef(target)` for GC-managed heap objects.
///
/// Creates a new [`JsWeakRef`] holding a raw pointer to `target`.
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
        target: WeakTarget::Heap(Some(target)),
    })
}

/// ECMAScript §26.1.1.1 `new WeakRef(target)` for Rc-managed plain objects.
///
/// Creates a new [`JsWeakRef`] using [`Rc::downgrade`] so that the weak
/// reference does **not** prevent the target from being dropped.  Once all
/// strong `Rc` handles are released, [`weak_ref_deref`] returns
/// [`JsValue::Undefined`].
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_core::builtins::weak_ref::{weak_ref_new_plain, weak_ref_deref};
/// use stator_core::objects::property_map::PropertyMap;
/// use stator_core::objects::value::JsValue;
///
/// let obj = Rc::new(RefCell::new(PropertyMap::new()));
/// let wr = weak_ref_new_plain(&obj);
/// assert_ne!(weak_ref_deref(&wr), JsValue::Undefined);
///
/// drop(obj);
/// assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
/// ```
pub fn weak_ref_new_plain(rc: &Rc<RefCell<PropertyMap>>) -> JsWeakRef {
    JsWeakRef {
        target: WeakTarget::Plain(Rc::downgrade(rc)),
    }
}

// ── weak_ref_deref ───────────────────────────────────────────────────────────

/// ECMAScript §26.1.3.2 `WeakRef.prototype.deref()`.
///
/// Returns the target as a [`JsValue`] if it is still alive, or
/// [`JsValue::Undefined`] if the target has been collected.
///
/// For `PlainObject` targets this checks `Weak::upgrade()`.  For
/// `HeapObject` targets this returns the stored raw pointer (or `Undefined`
/// if the GC has cleared it via [`weak_ref_clear`]).
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
    match &wr.target {
        WeakTarget::Plain(weak) => match weak.upgrade() {
            Some(rc) => JsValue::PlainObject(rc),
            None => JsValue::Undefined,
        },
        WeakTarget::Heap(Some(ptr)) => JsValue::Object(*ptr),
        WeakTarget::Heap(None) => JsValue::Undefined,
    }
}

// ── weak_ref_clear ───────────────────────────────────────────────────────────

/// GC hook: clear the weak reference when the target is collected.
///
/// After this call, [`weak_ref_deref`] will return [`JsValue::Undefined`].
/// For `PlainObject` targets this is a no-op (clearing is automatic).
/// For `HeapObject` targets this sets the internal slot to `None`.
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
    match &mut wr.target {
        WeakTarget::Plain(_) => {
            // Rc-managed targets are cleared automatically when all strong
            // references are dropped; replace with an already-expired Weak.
            wr.target = WeakTarget::Plain(Weak::new());
        }
        WeakTarget::Heap(opt) => {
            *opt = None;
        }
    }
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
    match &wr.target {
        WeakTarget::Plain(weak) => weak.strong_count() > 0,
        WeakTarget::Heap(opt) => opt.is_some(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Heap target: Construction ────────────────────────────────────────────

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

    // ── Heap target: Deref ───────────────────────────────────────────────────

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

    // ── Heap target: Clear (GC hook) ─────────────────────────────────────────

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

    // ── Heap target: has_target ──────────────────────────────────────────────

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

    // ── Heap target: Distinct targets ────────────────────────────────────────

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

    // ── Heap target: Multiple derefs ─────────────────────────────────────────

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

    // ── PlainObject target: Construction ─────────────────────────────────────

    #[test]
    fn test_weak_ref_new_plain_creates_weak_ref() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let wr = weak_ref_new_plain(&obj);
        assert!(weak_ref_has_target(&wr));
    }

    // ── PlainObject target: Deref returns PlainObject ────────────────────────

    #[test]
    fn test_weak_ref_plain_deref_returns_plain_object() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let wr = weak_ref_new_plain(&obj);
        assert!(matches!(weak_ref_deref(&wr), JsValue::PlainObject(_)));
    }

    // ── PlainObject target: Deref returns Undefined after drop ───────────────

    #[test]
    fn test_weak_ref_plain_deref_undefined_after_drop() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let wr = weak_ref_new_plain(&obj);
        drop(obj);
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    // ── PlainObject target: has_target false after drop ──────────────────────

    #[test]
    fn test_weak_ref_plain_has_target_false_after_drop() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let wr = weak_ref_new_plain(&obj);
        drop(obj);
        assert!(!weak_ref_has_target(&wr));
    }

    // ── PlainObject target: Clear ────────────────────────────────────────────

    #[test]
    fn test_weak_ref_plain_clear() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let mut wr = weak_ref_new_plain(&obj);
        weak_ref_clear(&mut wr);
        assert!(!weak_ref_has_target(&wr));
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    // ── PlainObject target: Multiple clones keep alive ───────────────────────

    #[test]
    fn test_weak_ref_plain_alive_while_rc_exists() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let obj2 = Rc::clone(&obj);
        let wr = weak_ref_new_plain(&obj);
        drop(obj);
        // obj2 still holds a strong reference
        assert!(weak_ref_has_target(&wr));
        assert!(matches!(weak_ref_deref(&wr), JsValue::PlainObject(_)));
        drop(obj2);
        // now all strong refs are gone
        assert!(!weak_ref_has_target(&wr));
        assert_eq!(weak_ref_deref(&wr), JsValue::Undefined);
    }

    // ── PlainObject target: Deref is repeatable ──────────────────────────────

    #[test]
    fn test_weak_ref_plain_deref_is_repeatable() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let wr = weak_ref_new_plain(&obj);
        let v1 = weak_ref_deref(&wr);
        let v2 = weak_ref_deref(&wr);
        assert!(matches!(v1, JsValue::PlainObject(_)));
        assert!(matches!(v2, JsValue::PlainObject(_)));
    }
}

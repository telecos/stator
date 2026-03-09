//! ECMAScript §26.2 `FinalizationRegistry` built-in.
//!
//! Provides [`JsFinalizationRegistry`], a registry that holds cleanup
//! callbacks to be invoked after registered target objects are
//! garbage-collected.
//!
//! # Weak semantics
//!
//! For `PlainObject` targets (backed by `Rc<RefCell<PropertyMap>>`), this
//! module stores [`std::rc::Weak`] references.  The [`finalization_registry_sweep_plain`]
//! function checks for expired weak references and moves matching held values
//! to the cleanup queue — no external GC notification is needed.
//!
//! For raw `*mut HeapObject` targets managed by the GC heap, the GC sweep
//! phase should call [`finalization_registry_notify`] when a target becomes
//! unreachable.
//!
//! # Safety
//!
//! All functions that accept a `*mut HeapObject` require that the pointer is
//! either null (rejected as invalid) or points to a live, properly-aligned
//! [`HeapObject`] managed by the engine heap.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §26.2 — *FinalizationRegistry Objects*

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::error::{StatorError, StatorResult};
use crate::objects::heap_object::HeapObject;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── Registration ─────────────────────────────────────────────────────────────

/// A single registration entry inside a [`JsFinalizationRegistry`].
#[derive(Debug, Clone)]
struct Registration {
    /// The raw address of the target object (`usize`).
    target: usize,
    /// The value passed to the cleanup callback when the target is collected.
    held_value: JsValue,
    /// Optional unregister token stored as a raw address (`usize`).
    unregister_token: Option<usize>,
}

/// A single registration entry for an Rc-managed `PlainObject` target.
#[derive(Debug, Clone)]
struct PlainRegistration {
    /// Weak reference to the target `PlainObject`.
    target: Weak<RefCell<PropertyMap>>,
    /// The value passed to the cleanup callback when the target is collected.
    held_value: JsValue,
    /// Optional unregister token stored as the raw pointer of an
    /// `Rc<RefCell<PropertyMap>>` (using `Rc::as_ptr` for identity).
    unregister_token: Option<usize>,
}

// ── JsFinalizationRegistry ───────────────────────────────────────────────────

/// A JavaScript `FinalizationRegistry` object per ECMAScript §26.2.
///
/// Registrations are stored in a `Vec`; when a target is collected the
/// corresponding entries move to a *cleanup queue* from which held values
/// can be drained.
///
/// Supports both GC-managed `HeapObject` targets (via [`finalization_registry_notify`])
/// and Rc-managed `PlainObject` targets (via [`finalization_registry_sweep_plain`]).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register,
///     finalization_registry_notify, finalization_registry_drain,
/// };
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let mut obj = HeapObject::new_null();
/// let target = &raw mut obj;
/// finalization_registry_register(&mut fr, target, JsValue::Smi(42), None).unwrap();
/// finalization_registry_notify(&mut fr, target);
/// let held = finalization_registry_drain(&mut fr);
/// assert_eq!(held, vec![JsValue::Smi(42)]);
/// ```
#[derive(Debug, Default)]
pub struct JsFinalizationRegistry {
    /// Active registrations for GC-managed targets, indexed by auto-incrementing ID.
    registrations: HashMap<u64, Registration>,
    /// Active registrations for Rc-managed `PlainObject` targets.
    plain_registrations: HashMap<u64, PlainRegistration>,
    /// Monotonically increasing counter for registration IDs.
    next_id: u64,
    /// Held values ready to be delivered to the cleanup callback.
    cleanup_queue: Vec<JsValue>,
}

// ── Constructors ─────────────────────────────────────────────────────────────

/// ECMAScript §26.2.1.1 `new FinalizationRegistry(cleanupCallback)`.
///
/// Creates an empty [`JsFinalizationRegistry`].  The actual cleanup callback
/// is held at the JS layer; this struct only manages registrations.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::finalization_registry_new;
///
/// let fr = finalization_registry_new();
/// ```
pub fn finalization_registry_new() -> JsFinalizationRegistry {
    JsFinalizationRegistry::default()
}

// ── finalization_registry_register ───────────────────────────────────────────

/// ECMAScript §26.2.3.2
/// `FinalizationRegistry.prototype.register(target, heldValue, unregisterToken)`.
///
/// Adds a registration so that when `target` is garbage-collected the
/// `held_value` will be passed to the cleanup callback.
///
/// `unregister_token` is an optional object pointer that can later be passed
/// to [`finalization_registry_unregister`] to remove all registrations
/// sharing that token.
///
/// Returns [`StatorError::TypeError`] if:
/// - `target` is null.
/// - `unregister_token` is explicitly `Some(null)`.
/// - `target` and `unregister_token` are the same pointer (the spec
///   allows this, but our implementation currently permits it — this
///   note is for future reference).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register,
/// };
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let mut obj = HeapObject::new_null();
/// finalization_registry_register(&mut fr, &raw mut obj, JsValue::Smi(1), None).unwrap();
/// ```
pub fn finalization_registry_register(
    registry: &mut JsFinalizationRegistry,
    target: *mut HeapObject,
    held_value: JsValue,
    unregister_token: Option<*mut HeapObject>,
) -> StatorResult<()> {
    if target.is_null() {
        return Err(StatorError::TypeError(
            "FinalizationRegistry target must be an object".into(),
        ));
    }
    if let Some(tok) = unregister_token
        && tok.is_null()
    {
        return Err(StatorError::TypeError(
            "FinalizationRegistry unregister token must be an object".into(),
        ));
    }
    let id = registry.next_id;
    registry.next_id += 1;
    registry.registrations.insert(
        id,
        Registration {
            target: target as usize,
            held_value,
            unregister_token: unregister_token.map(|t| t as usize),
        },
    );
    Ok(())
}

// ── finalization_registry_unregister ─────────────────────────────────────────

/// ECMAScript §26.2.3.3
/// `FinalizationRegistry.prototype.unregister(unregisterToken)`.
///
/// Removes all registrations whose unregister token matches `token`.
/// Returns `true` if at least one registration was removed, `false`
/// otherwise.
///
/// Returns [`StatorError::TypeError`] if `token` is null.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register,
///     finalization_registry_unregister,
/// };
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let mut obj = HeapObject::new_null();
/// let mut tok = HeapObject::new_null();
/// let target = &raw mut obj;
/// let token = &raw mut tok;
/// finalization_registry_register(&mut fr, target, JsValue::Smi(1), Some(token)).unwrap();
/// assert!(finalization_registry_unregister(&mut fr, token).unwrap());
/// assert!(!finalization_registry_unregister(&mut fr, token).unwrap());
/// ```
pub fn finalization_registry_unregister(
    registry: &mut JsFinalizationRegistry,
    token: *mut HeapObject,
) -> StatorResult<bool> {
    if token.is_null() {
        return Err(StatorError::TypeError(
            "FinalizationRegistry unregister token must be an object".into(),
        ));
    }
    let addr = token as usize;
    let before = registry.registrations.len();
    registry
        .registrations
        .retain(|_, reg| reg.unregister_token != Some(addr));
    Ok(registry.registrations.len() < before)
}

// ── finalization_registry_notify ─────────────────────────────────────────────

/// GC hook: notify the registry that `target` has been collected.
///
/// Moves all registrations whose target matches `target` into the cleanup
/// queue.  Their held values can then be retrieved via
/// [`finalization_registry_drain`].
///
/// Calling with a null pointer is safe and is a no-op.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register,
///     finalization_registry_notify, finalization_registry_drain,
/// };
/// use stator_core::objects::heap_object::HeapObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let mut obj = HeapObject::new_null();
/// let target = &raw mut obj;
/// finalization_registry_register(&mut fr, target, JsValue::Boolean(true), None).unwrap();
/// finalization_registry_notify(&mut fr, target);
/// assert_eq!(finalization_registry_drain(&mut fr), vec![JsValue::Boolean(true)]);
/// ```
pub fn finalization_registry_notify(
    registry: &mut JsFinalizationRegistry,
    target: *mut HeapObject,
) {
    if target.is_null() {
        return;
    }
    let addr = target as usize;
    let mut remaining = HashMap::new();
    for (id, reg) in registry.registrations.drain() {
        if reg.target == addr {
            registry.cleanup_queue.push(reg.held_value);
        } else {
            remaining.insert(id, reg);
        }
    }
    registry.registrations = remaining;
}

// ── finalization_registry_drain ──────────────────────────────────────────────

/// Drain the cleanup queue, returning all held values ready for callback
/// invocation.
///
/// After this call the cleanup queue is empty.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_drain,
/// };
///
/// let mut fr = finalization_registry_new();
/// assert!(finalization_registry_drain(&mut fr).is_empty());
/// ```
pub fn finalization_registry_drain(registry: &mut JsFinalizationRegistry) -> Vec<JsValue> {
    std::mem::take(&mut registry.cleanup_queue)
}

// ── finalization_registry_register_plain ──────────────────────────────────────

/// Register an Rc-managed `PlainObject` target.
///
/// Uses [`Rc::downgrade`] to store a weak reference.  When all strong `Rc`
/// handles to the target are dropped, [`finalization_registry_sweep_plain`]
/// will move the held value to the cleanup queue.
///
/// `unregister_token` is an optional `Rc<RefCell<PropertyMap>>` whose raw
/// pointer is used as the token identity for [`finalization_registry_unregister_plain`].
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register_plain,
///     finalization_registry_sweep_plain, finalization_registry_drain,
/// };
/// use stator_core::objects::property_map::PropertyMap;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let obj = Rc::new(RefCell::new(PropertyMap::new()));
/// finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(42), None);
/// drop(obj);
/// finalization_registry_sweep_plain(&mut fr);
/// let held = finalization_registry_drain(&mut fr);
/// assert_eq!(held, vec![JsValue::Smi(42)]);
/// ```
pub fn finalization_registry_register_plain(
    registry: &mut JsFinalizationRegistry,
    target: &Rc<RefCell<PropertyMap>>,
    held_value: JsValue,
    unregister_token: Option<&Rc<RefCell<PropertyMap>>>,
) {
    let id = registry.next_id;
    registry.next_id += 1;
    registry.plain_registrations.insert(
        id,
        PlainRegistration {
            target: Rc::downgrade(target),
            held_value,
            unregister_token: unregister_token.map(|t| Rc::as_ptr(t) as usize),
        },
    );
}

// ── finalization_registry_unregister_plain ────────────────────────────────────

/// Unregister all `PlainObject` registrations matching the given token.
///
/// Returns `true` if at least one registration was removed.
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register_plain,
///     finalization_registry_unregister_plain,
/// };
/// use stator_core::objects::property_map::PropertyMap;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let obj = Rc::new(RefCell::new(PropertyMap::new()));
/// let tok = Rc::new(RefCell::new(PropertyMap::new()));
/// finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(1), Some(&tok));
/// assert!(finalization_registry_unregister_plain(&mut fr, &tok));
/// assert!(!finalization_registry_unregister_plain(&mut fr, &tok));
/// ```
pub fn finalization_registry_unregister_plain(
    registry: &mut JsFinalizationRegistry,
    token: &Rc<RefCell<PropertyMap>>,
) -> bool {
    let addr = Rc::as_ptr(token) as usize;
    let before = registry.plain_registrations.len();
    registry
        .plain_registrations
        .retain(|_, reg| reg.unregister_token != Some(addr));
    registry.plain_registrations.len() < before
}

// ── finalization_registry_sweep_plain ─────────────────────────────────────────

/// Check for collected `PlainObject` targets and move their held values to
/// the cleanup queue.
///
/// This should be called periodically (e.g. before `cleanupSome()`) to
/// detect Rc-managed targets that have been dropped.
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_register_plain,
///     finalization_registry_sweep_plain, finalization_registry_drain,
/// };
/// use stator_core::objects::property_map::PropertyMap;
/// use stator_core::objects::value::JsValue;
///
/// let mut fr = finalization_registry_new();
/// let obj = Rc::new(RefCell::new(PropertyMap::new()));
/// finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(99), None);
/// drop(obj);
/// finalization_registry_sweep_plain(&mut fr);
/// assert_eq!(finalization_registry_drain(&mut fr), vec![JsValue::Smi(99)]);
/// ```
pub fn finalization_registry_sweep_plain(registry: &mut JsFinalizationRegistry) {
    let mut remaining = HashMap::new();
    for (id, reg) in registry.plain_registrations.drain() {
        if reg.target.strong_count() == 0 {
            registry.cleanup_queue.push(reg.held_value);
        } else {
            remaining.insert(id, reg);
        }
    }
    registry.plain_registrations = remaining;
}

// ── finalization_registry_has_registrations ───────────────────────────────────

/// Returns `true` if the registry has any active registrations (heap or plain).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::finalization_registry::{
///     finalization_registry_new, finalization_registry_has_registrations,
/// };
///
/// let fr = finalization_registry_new();
/// assert!(!finalization_registry_has_registrations(&fr));
/// ```
pub fn finalization_registry_has_registrations(registry: &JsFinalizationRegistry) -> bool {
    !registry.registrations.is_empty() || !registry.plain_registrations.is_empty()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_new_is_empty() {
        let fr = finalization_registry_new();
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_drain_empty_returns_empty() {
        let mut fr = finalization_registry_new();
        assert!(finalization_registry_drain(&mut fr).is_empty());
    }

    // ── Register ─────────────────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_register_valid() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let result = finalization_registry_register(&mut fr, &raw mut obj, JsValue::Smi(1), None);
        assert!(result.is_ok());
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_register_null_target_error() {
        let mut fr = finalization_registry_new();
        let result =
            finalization_registry_register(&mut fr, std::ptr::null_mut(), JsValue::Smi(1), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_finalization_registry_register_null_target_is_type_error() {
        let mut fr = finalization_registry_new();
        let err =
            finalization_registry_register(&mut fr, std::ptr::null_mut(), JsValue::Smi(1), None)
                .unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_finalization_registry_register_null_token_error() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let result = finalization_registry_register(
            &mut fr,
            &raw mut obj,
            JsValue::Smi(1),
            Some(std::ptr::null_mut()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_finalization_registry_register_with_token() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let mut tok = HeapObject::new_null();
        let result = finalization_registry_register(
            &mut fr,
            &raw mut obj,
            JsValue::Smi(1),
            Some(&raw mut tok),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_finalization_registry_register_multiple() {
        let mut fr = finalization_registry_new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        finalization_registry_register(&mut fr, &raw mut a, JsValue::Smi(1), None).unwrap();
        finalization_registry_register(&mut fr, &raw mut b, JsValue::Smi(2), None).unwrap();
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_register_same_target_twice() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        finalization_registry_register(&mut fr, ptr, JsValue::Smi(1), None).unwrap();
        finalization_registry_register(&mut fr, ptr, JsValue::Smi(2), None).unwrap();
        finalization_registry_notify(&mut fr, ptr);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held.len(), 2);
    }

    // ── Unregister ───────────────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_unregister_removes_matching() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let mut tok = HeapObject::new_null();
        let target = &raw mut obj;
        let token = &raw mut tok;
        finalization_registry_register(&mut fr, target, JsValue::Smi(1), Some(token)).unwrap();
        assert!(finalization_registry_unregister(&mut fr, token).unwrap());
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_unregister_returns_false_when_no_match() {
        let mut fr = finalization_registry_new();
        let mut tok = HeapObject::new_null();
        assert!(!finalization_registry_unregister(&mut fr, &raw mut tok).unwrap());
    }

    #[test]
    fn test_finalization_registry_unregister_null_token_error() {
        let mut fr = finalization_registry_new();
        let result = finalization_registry_unregister(&mut fr, std::ptr::null_mut());
        assert!(result.is_err());
    }

    #[test]
    fn test_finalization_registry_unregister_only_removes_matching_token() {
        let mut fr = finalization_registry_new();
        let mut obj1 = HeapObject::new_null();
        let mut obj2 = HeapObject::new_null();
        let mut tok1 = HeapObject::new_null();
        let mut tok2 = HeapObject::new_null();
        finalization_registry_register(
            &mut fr,
            &raw mut obj1,
            JsValue::Smi(1),
            Some(&raw mut tok1),
        )
        .unwrap();
        finalization_registry_register(
            &mut fr,
            &raw mut obj2,
            JsValue::Smi(2),
            Some(&raw mut tok2),
        )
        .unwrap();
        finalization_registry_unregister(&mut fr, &raw mut tok1).unwrap();
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_unregister_removes_multiple_with_same_token() {
        let mut fr = finalization_registry_new();
        let mut obj1 = HeapObject::new_null();
        let mut obj2 = HeapObject::new_null();
        let mut tok = HeapObject::new_null();
        let token = &raw mut tok;
        finalization_registry_register(&mut fr, &raw mut obj1, JsValue::Smi(1), Some(token))
            .unwrap();
        finalization_registry_register(&mut fr, &raw mut obj2, JsValue::Smi(2), Some(token))
            .unwrap();
        assert!(finalization_registry_unregister(&mut fr, token).unwrap());
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_unregister_does_not_affect_no_token_entries() {
        let mut fr = finalization_registry_new();
        let mut obj1 = HeapObject::new_null();
        let mut obj2 = HeapObject::new_null();
        let mut tok = HeapObject::new_null();
        let token = &raw mut tok;
        finalization_registry_register(&mut fr, &raw mut obj1, JsValue::Smi(1), Some(token))
            .unwrap();
        finalization_registry_register(&mut fr, &raw mut obj2, JsValue::Smi(2), None).unwrap();
        finalization_registry_unregister(&mut fr, token).unwrap();
        // Entry without token should still exist
        assert!(finalization_registry_has_registrations(&fr));
    }

    // ── Notify (GC hook) ─────────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_notify_moves_to_cleanup_queue() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::Smi(42), None).unwrap();
        finalization_registry_notify(&mut fr, target);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Smi(42)]);
    }

    #[test]
    fn test_finalization_registry_notify_removes_registration() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::Smi(1), None).unwrap();
        finalization_registry_notify(&mut fr, target);
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_notify_null_is_noop() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        finalization_registry_register(&mut fr, &raw mut obj, JsValue::Smi(1), None).unwrap();
        finalization_registry_notify(&mut fr, std::ptr::null_mut());
        assert!(finalization_registry_has_registrations(&fr));
        assert!(finalization_registry_drain(&mut fr).is_empty());
    }

    #[test]
    fn test_finalization_registry_notify_only_matching_target() {
        let mut fr = finalization_registry_new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let pa = &raw mut a;
        let pb = &raw mut b;
        finalization_registry_register(&mut fr, pa, JsValue::Smi(1), None).unwrap();
        finalization_registry_register(&mut fr, pb, JsValue::Smi(2), None).unwrap();
        finalization_registry_notify(&mut fr, pa);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Smi(1)]);
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_notify_multiple_registrations_same_target() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        finalization_registry_register(&mut fr, ptr, JsValue::Smi(10), None).unwrap();
        finalization_registry_register(&mut fr, ptr, JsValue::Smi(20), None).unwrap();
        finalization_registry_notify(&mut fr, ptr);
        let mut held = finalization_registry_drain(&mut fr);
        held.sort_by_key(|v| match v {
            JsValue::Smi(n) => *n,
            _ => 0,
        });
        assert_eq!(held, vec![JsValue::Smi(10), JsValue::Smi(20)]);
    }

    #[test]
    fn test_finalization_registry_notify_already_unregistered_is_noop() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let mut tok = HeapObject::new_null();
        let target = &raw mut obj;
        let token = &raw mut tok;
        finalization_registry_register(&mut fr, target, JsValue::Smi(1), Some(token)).unwrap();
        finalization_registry_unregister(&mut fr, token).unwrap();
        finalization_registry_notify(&mut fr, target);
        assert!(finalization_registry_drain(&mut fr).is_empty());
    }

    // ── Drain ────────────────────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_drain_clears_queue() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::Smi(1), None).unwrap();
        finalization_registry_notify(&mut fr, target);
        let first = finalization_registry_drain(&mut fr);
        assert_eq!(first.len(), 1);
        let second = finalization_registry_drain(&mut fr);
        assert!(second.is_empty());
    }

    #[test]
    fn test_finalization_registry_drain_preserves_order_per_notify() {
        let mut fr = finalization_registry_new();
        let mut a = HeapObject::new_null();
        let mut b = HeapObject::new_null();
        let pa = &raw mut a;
        let pb = &raw mut b;
        finalization_registry_register(&mut fr, pa, JsValue::Smi(1), None).unwrap();
        finalization_registry_register(&mut fr, pb, JsValue::Smi(2), None).unwrap();
        finalization_registry_notify(&mut fr, pa);
        finalization_registry_notify(&mut fr, pb);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Smi(1), JsValue::Smi(2)]);
    }

    // ── Integration scenarios ────────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_register_then_notify_then_drain() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::Boolean(true), None).unwrap();
        finalization_registry_notify(&mut fr, target);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Boolean(true)]);
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_held_value_undefined() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::Undefined, None).unwrap();
        finalization_registry_notify(&mut fr, target);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Undefined]);
    }

    #[test]
    fn test_finalization_registry_held_value_string() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let target = &raw mut obj;
        finalization_registry_register(&mut fr, target, JsValue::String("cleanup".into()), None)
            .unwrap();
        finalization_registry_notify(&mut fr, target);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::String("cleanup".into())]);
    }

    #[test]
    fn test_finalization_registry_no_notify_means_no_drain() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        finalization_registry_register(&mut fr, &raw mut obj, JsValue::Smi(1), None).unwrap();
        // No notify call
        assert!(finalization_registry_drain(&mut fr).is_empty());
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_notify_nonexistent_target_is_noop() {
        let mut fr = finalization_registry_new();
        let mut obj = HeapObject::new_null();
        let mut other = HeapObject::new_null();
        finalization_registry_register(&mut fr, &raw mut obj, JsValue::Smi(1), None).unwrap();
        finalization_registry_notify(&mut fr, &raw mut other);
        assert!(finalization_registry_drain(&mut fr).is_empty());
        assert!(finalization_registry_has_registrations(&fr));
    }

    // ── PlainObject registration ─────────────────────────────────────────────

    #[test]
    fn test_finalization_registry_register_plain_has_registrations() {
        let mut fr = finalization_registry_new();
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(1), None);
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_sweep_plain_after_drop() {
        let mut fr = finalization_registry_new();
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(42), None);
        drop(obj);
        finalization_registry_sweep_plain(&mut fr);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Smi(42)]);
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_sweep_plain_alive_target_not_drained() {
        let mut fr = finalization_registry_new();
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(1), None);
        finalization_registry_sweep_plain(&mut fr);
        assert!(finalization_registry_drain(&mut fr).is_empty());
        assert!(finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_unregister_plain_removes_matching() {
        let mut fr = finalization_registry_new();
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        let tok = Rc::new(RefCell::new(PropertyMap::new()));
        finalization_registry_register_plain(&mut fr, &obj, JsValue::Smi(1), Some(&tok));
        assert!(finalization_registry_unregister_plain(&mut fr, &tok));
        assert!(!finalization_registry_has_registrations(&fr));
    }

    #[test]
    fn test_finalization_registry_unregister_plain_returns_false_no_match() {
        let mut fr = finalization_registry_new();
        let tok = Rc::new(RefCell::new(PropertyMap::new()));
        assert!(!finalization_registry_unregister_plain(&mut fr, &tok));
    }

    #[test]
    fn test_finalization_registry_sweep_plain_multiple_targets() {
        let mut fr = finalization_registry_new();
        let a = Rc::new(RefCell::new(PropertyMap::new()));
        let b = Rc::new(RefCell::new(PropertyMap::new()));
        finalization_registry_register_plain(&mut fr, &a, JsValue::Smi(1), None);
        finalization_registry_register_plain(&mut fr, &b, JsValue::Smi(2), None);
        drop(a);
        finalization_registry_sweep_plain(&mut fr);
        let held = finalization_registry_drain(&mut fr);
        assert_eq!(held, vec![JsValue::Smi(1)]);
        // b is still alive
        assert!(finalization_registry_has_registrations(&fr));
    }
}

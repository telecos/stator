//! ECMAScript §28.2 `Proxy` built-in and §10.5 Proxy invariant enforcement.
//!
//! A [`JsProxy`] wraps a *target* [`JsObject`] together with a
//! [`ProxyHandler`] that holds up to 13 optional Rust-closure traps.  When an
//! operation is performed on the proxy each trap is consulted first; if no trap
//! is installed the operation falls through to the target object unchanged
//! (the *default behaviour*).
//!
//! Revocable proxies ([`proxy_revocable`]) set the `revoked` flag on the proxy
//! instance; all subsequent operations return a
//! [`StatorError::TypeError`][crate::error::StatorError::TypeError].
//!
//! # Invariant enforcement (ECMAScript §10.5)
//!
//! The following proxy invariants from ECMAScript §10.5 are enforced:
//!
//! * **`get`** – if the corresponding target property is non-configurable and
//!   non-writable the trap result must equal the target's value.
//! * **`set`** – a trap may not successfully change a non-configurable,
//!   non-writable data property to a different value.
//! * **`has`** – a trap may not report `false` for a property that is
//!   non-configurable on the target, or for any property when the target is
//!   non-extensible.
//! * **`deleteProperty`** – a trap may not report success for a
//!   non-configurable property.
//! * **`getOwnPropertyDescriptor`** – a trap may not report a property as
//!   non-existent if the corresponding target property is non-configurable.
//! * **`defineProperty`** – a trap may not define a property that would
//!   violate the non-configurable property constraints of the target.
//! * **`getPrototypeOf`** – for a non-extensible target, the trap result must
//!   equal the target's actual prototype.
//! * **`setPrototypeOf`** – for a non-extensible target, a successful trap
//!   result requires the requested prototype to equal the target's actual
//!   prototype.
//! * **`isExtensible`** – the trap result must match the target's
//!   extensibility.
//! * **`preventExtensions`** – a trap may only return `true` if the target is
//!   non-extensible.
//! * **`ownKeys`** – the trap result must include every non-configurable own
//!   key, and for non-extensible targets it must contain every existing own key
//!   and no extra keys.
//!
//! # Naming convention
//!
//! Each function is prefixed `proxy_` to mirror ECMAScript `Proxy.*` and to
//! avoid ambiguity with standard-library items.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §28.2 — *The Proxy Constructor*
//! * ECMAScript 2025 Language Specification §10.5 — *Proxy Object Internal Methods*

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::js_object::JsObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ── Type aliases for trap closures ────────────────────────────────────────────

/// Trap for `[[Get]](target, key, receiver)`.
pub type GetTrap = Box<dyn Fn(&JsObject, &str, &JsValue) -> StatorResult<JsValue>>;
/// Trap for `[[Set]](target, key, value, receiver)` → boolean.
pub type SetTrap = Box<dyn Fn(&mut JsObject, &str, JsValue, &JsValue) -> StatorResult<bool>>;
/// Trap for `[[Has]](target, key)` → boolean.
pub type HasTrap = Box<dyn Fn(&JsObject, &str) -> StatorResult<bool>>;
/// Trap for `[[Delete]](target, key)` → boolean.
pub type DeletePropertyTrap = Box<dyn Fn(&mut JsObject, &str) -> StatorResult<bool>>;
/// Trap for `[[DefineOwnProperty]](target, key, value, attributes)` → boolean.
pub type DefinePropertyTrap =
    Box<dyn Fn(&mut JsObject, &str, JsValue, PropertyAttributes) -> StatorResult<bool>>;
/// Trap for `[[GetOwnProperty]](target, key)`.
pub type GetOwnPropertyDescriptorTrap =
    Box<dyn Fn(&JsObject, &str) -> StatorResult<Option<(JsValue, PropertyAttributes)>>>;
/// Trap for `[[GetPrototypeOf]](target)`.
pub type GetPrototypeOfTrap = Box<dyn Fn(&JsObject) -> StatorResult<JsValue>>;
/// Trap for `[[SetPrototypeOf]](target, proto)` → boolean.
pub type SetPrototypeOfTrap = Box<dyn Fn(&mut JsObject, JsValue) -> StatorResult<bool>>;
/// Trap for `[[IsExtensible]](target)` → boolean.
pub type IsExtensibleTrap = Box<dyn Fn(&JsObject) -> StatorResult<bool>>;
/// Trap for `[[PreventExtensions]](target)` → boolean.
pub type PreventExtensionsTrap = Box<dyn Fn(&mut JsObject) -> StatorResult<bool>>;
/// Trap for `[[OwnPropertyKeys]](target)` → keys.
pub type OwnKeysTrap = Box<dyn Fn(&JsObject) -> StatorResult<Vec<JsValue>>>;
/// Trap for `[[Call]](thisArg, args)` → value.
pub type ApplyTrap = Box<dyn Fn(JsValue, Vec<JsValue>) -> StatorResult<JsValue>>;
/// Trap for `[[Construct]](args, newTarget)` → object.
pub type ConstructTrap = Box<dyn Fn(Vec<JsValue>, JsValue) -> StatorResult<JsValue>>;

// ── ProxyHandler ──────────────────────────────────────────────────────────────

/// Holds the 13 optional trap closures for a [`JsProxy`].
///
/// A `None` trap means the default behaviour (fall through to the target
/// object's own internal method) is used.
///
/// # Examples
///
/// ```ignore
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_get};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut handler = ProxyHandler::default();
/// // Install a get-trap that intercepts every property read.
/// handler.get = Some(Box::new(|_target, _key| Ok(JsValue::Smi(42))));
///
/// let target = JsObject::new();
/// let mut proxy = proxy_new(target, handler);
/// assert_eq!(proxy_get(&proxy, "anything").unwrap(), JsValue::Smi(42));
/// ```
#[derive(Default)]
pub struct ProxyHandler {
    /// `[[Get]]` trap.
    pub get: Option<GetTrap>,
    /// `[[Set]]` trap.
    pub set: Option<SetTrap>,
    /// `[[Has]]` trap.
    pub has: Option<HasTrap>,
    /// `[[Delete]]` trap.
    pub delete_property: Option<DeletePropertyTrap>,
    /// `[[DefineOwnProperty]]` trap.
    pub define_property: Option<DefinePropertyTrap>,
    /// `[[GetOwnProperty]]` trap.
    pub get_own_property_descriptor: Option<GetOwnPropertyDescriptorTrap>,
    /// `[[GetPrototypeOf]]` trap.
    pub get_prototype_of: Option<GetPrototypeOfTrap>,
    /// `[[SetPrototypeOf]]` trap.
    pub set_prototype_of: Option<SetPrototypeOfTrap>,
    /// `[[IsExtensible]]` trap.
    pub is_extensible: Option<IsExtensibleTrap>,
    /// `[[PreventExtensions]]` trap.
    pub prevent_extensions: Option<PreventExtensionsTrap>,
    /// `[[OwnPropertyKeys]]` trap.
    pub own_keys: Option<OwnKeysTrap>,
    /// `[[Call]]` trap.
    pub apply: Option<ApplyTrap>,
    /// `[[Construct]]` trap.
    pub construct: Option<ConstructTrap>,
}

// ── JsProxy ───────────────────────────────────────────────────────────────────

/// A JavaScript `Proxy` object per ECMAScript §28.2.
///
/// Wraps a `target` [`JsObject`] and a [`ProxyHandler`].  Operations that are
/// performed on the proxy check the handler for a corresponding trap first; if
/// no trap is installed the default (target) behaviour applies.
///
/// # Revocation
///
/// Calling [`proxy_revoke`] sets the internal `revoked` flag.  Any subsequent
/// operation on a revoked proxy returns
/// [`StatorError::TypeError`][crate::error::StatorError::TypeError].
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_get, proxy_set};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("x", JsValue::Smi(1)).unwrap();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert_eq!(proxy_get(&proxy, "x").unwrap(), JsValue::Smi(1));
/// ```
pub struct JsProxy {
    /// The target object.
    pub(crate) target: JsObject,
    /// The handler holding optional traps.
    pub(crate) handler: ProxyHandler,
    /// The original JavaScript target value for host-created proxies.
    pub(crate) target_value: Option<JsValue>,
    /// Whether this proxy has been revoked.
    revoked: bool,
    /// Whether the proxy's target is callable (`[[Call]]` internal method).
    /// Per ECMAScript §10.5.12, `typeof proxy` returns `"function"` when
    /// the target is callable.
    pub(crate) callable: bool,
}

// ── Constructors ──────────────────────────────────────────────────────────────

/// ECMAScript §28.2.1.1 `new Proxy(target, handler)`.
///
/// Creates a new, non-revocable [`JsProxy`] wrapping `target` and using
/// `handler` for trap dispatch.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new};
/// use stator_core::objects::js_object::JsObject;
///
/// let p = proxy_new(JsObject::new(), ProxyHandler::default());
/// assert!(!p.is_revoked());
/// ```
pub fn proxy_new(target: JsObject, handler: ProxyHandler) -> JsProxy {
    JsProxy {
        target,
        handler,
        target_value: None,
        revoked: false,
        callable: false,
    }
}

/// Creates a new [`JsProxy`] whose target is callable.
///
/// This sets the internal `callable` flag so that `typeof proxy` correctly
/// returns `"function"` per ECMAScript §10.5.12.
pub fn proxy_new_callable(target: JsObject, handler: ProxyHandler) -> JsProxy {
    JsProxy {
        target,
        handler,
        target_value: None,
        revoked: false,
        callable: true,
    }
}

/// ECMAScript §28.2.2 `Proxy.revocable(target, handler)`.
///
/// Creates a [`JsProxy`] that may later be revoked by calling
/// [`proxy_revoke`] on it.  The returned proxy behaves identically to one
/// created with [`proxy_new`] until it is revoked.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{
///     ProxyHandler, proxy_revocable, proxy_revoke, proxy_get,
/// };
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("k", JsValue::Smi(1)).unwrap();
///
/// let mut p = proxy_revocable(target, ProxyHandler::default());
/// assert_eq!(proxy_get(&p, "k").unwrap(), JsValue::Smi(1));
/// proxy_revoke(&mut p);
/// assert!(proxy_get(&p, "k").is_err());
/// ```
pub fn proxy_revocable(target: JsObject, handler: ProxyHandler) -> JsProxy {
    proxy_new(target, handler)
}

/// Revokes the proxy created by [`proxy_revocable`].
///
/// After calling this function every subsequent operation on `proxy` will
/// return [`StatorError::TypeError`].
pub fn proxy_revoke(proxy: &mut JsProxy) {
    proxy.revoked = true;
}

impl JsProxy {
    /// Returns `true` if this proxy has been revoked.
    pub fn is_revoked(&self) -> bool {
        self.revoked
    }

    /// Returns `true` if the proxy's target is callable.
    pub fn is_callable(&self) -> bool {
        self.callable
    }

    /// Returns an error if this proxy is revoked.
    fn check_revoked(&self) -> StatorResult<()> {
        if self.revoked {
            Err(StatorError::TypeError(
                "Cannot perform operation on a revoked proxy".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn target_is_extensible(&self) -> bool {
        match &self.target_value {
            Some(JsValue::PlainObject(map)) => map.borrow().extensible,
            Some(JsValue::Error(error)) => error.props.borrow().extensible,
            _ => self.target.is_extensible(),
        }
    }

    fn target_prototype_value(&self) -> JsValue {
        match &self.target_value {
            Some(JsValue::PlainObject(map)) => map
                .borrow()
                .get("__proto__")
                .cloned()
                .unwrap_or(JsValue::Null),
            Some(JsValue::Error(error)) => error
                .props
                .borrow()
                .get("__proto__")
                .cloned()
                .unwrap_or(JsValue::Null),
            _ => JsValue::Null,
        }
    }

    fn target_own_property_descriptor(&self, key: &str) -> Option<(JsValue, PropertyAttributes)> {
        match &self.target_value {
            Some(JsValue::PlainObject(map)) => {
                plain_visible_own_property_descriptor(&map.borrow(), key)
            }
            Some(JsValue::Error(error)) => {
                plain_visible_own_property_descriptor(&error.props.borrow(), key)
            }
            _ => self.target.get_own_property_descriptor(key),
        }
    }

    fn target_has_own_property(&self, key: &str) -> bool {
        self.target_own_property_descriptor(key).is_some()
    }

    fn target_own_property_keys(&self) -> Vec<String> {
        match &self.target_value {
            Some(JsValue::PlainObject(map)) => plain_visible_own_property_keys(&map.borrow()),
            Some(JsValue::Error(error)) => plain_visible_own_property_keys(&error.props.borrow()),
            _ => self.target.own_property_keys(),
        }
    }
}

fn accessor_property_name(key: &str) -> Option<&str> {
    if let Some(rest) = key
        .strip_prefix("__get_")
        .or_else(|| key.strip_prefix("__set_"))
    {
        rest.strip_suffix("__")
    } else {
        None
    }
}

fn plain_visible_own_property_descriptor(
    map: &PropertyMap,
    key: &str,
) -> Option<(JsValue, PropertyAttributes)> {
    if let Some(value) = map.get(key).cloned() {
        let mut attrs = PropertyAttributes::empty();
        if map.is_writable(key) {
            attrs |= PropertyAttributes::WRITABLE;
        }
        if map.is_enumerable(key) {
            attrs |= PropertyAttributes::ENUMERABLE;
        }
        if map.is_configurable(key) {
            attrs |= PropertyAttributes::CONFIGURABLE;
        }
        return Some((value, attrs));
    }

    let getter_key = format!("__get_{key}__");
    let setter_key = format!("__set_{key}__");
    if map.contains_key(&getter_key) || map.contains_key(&setter_key) {
        let attr_key = if map.contains_key(&getter_key) {
            getter_key.as_str()
        } else {
            setter_key.as_str()
        };
        let mut attrs = PropertyAttributes::empty();
        if map.is_enumerable(attr_key) {
            attrs |= PropertyAttributes::ENUMERABLE;
        }
        if map.is_configurable(attr_key) {
            attrs |= PropertyAttributes::CONFIGURABLE;
        }
        return Some((JsValue::Undefined, attrs));
    }

    None
}

fn plain_visible_own_property_keys(map: &PropertyMap) -> Vec<String> {
    let mut keys = Vec::new();
    let mut seen = HashSet::new();

    for raw_key in map.keys() {
        if &**raw_key == "__proto__" {
            continue;
        }
        if let Some(name) = accessor_property_name(raw_key) {
            if seen.insert(name.to_string()) {
                keys.push(name.to_string());
            }
            continue;
        }
        if seen.insert(raw_key.to_string()) {
            keys.push(raw_key.to_string());
        }
    }

    keys
}

// ── proxy_get ─────────────────────────────────────────────────────────────────

/// ECMAScript §10.5.8 `[[Get]]` for Proxy.
///
/// Invokes the `get` trap if installed, otherwise falls through to the target.
/// Enforces the invariant: if the corresponding target own property is
/// non-configurable and non-writable, the trap result must equal the target
/// property value.
///
/// Returns [`StatorError::TypeError`] for revoked proxies or invariant
/// violations.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_get};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("x", JsValue::Smi(3)).unwrap();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert_eq!(proxy_get(&proxy, "x").unwrap(), JsValue::Smi(3));
/// ```
pub fn proxy_get(proxy: &JsProxy, key: &str) -> StatorResult<JsValue> {
    proxy_get_with_receiver(proxy, key, &JsValue::Undefined)
}

/// ECMAScript §10.5.8 `[[Get]]` for Proxy with an explicit receiver.
pub fn proxy_get_with_receiver(
    proxy: &JsProxy,
    key: &str,
    receiver: &JsValue,
) -> StatorResult<JsValue> {
    proxy.check_revoked()?;
    let trap_result = if let Some(trap) = &proxy.handler.get {
        trap(&proxy.target, key, receiver)?
    } else {
        proxy.target.get_property(key)
    };
    // Invariant §10.5.8 step 11: non-configurable + non-writable target property
    // must equal the trap result.
    if let Some((target_val, attrs)) = proxy.target_own_property_descriptor(key)
        && !attrs.contains(PropertyAttributes::CONFIGURABLE)
        && !attrs.contains(PropertyAttributes::WRITABLE)
        && trap_result != target_val
    {
        return Err(StatorError::TypeError(format!(
            "Proxy get trap returned a value incompatible with the \
             non-configurable, non-writable own property '{key}'"
        )));
    }
    Ok(trap_result)
}

// ── proxy_set ─────────────────────────────────────────────────────────────────

/// ECMAScript §10.5.9 `[[Set]]` for Proxy.
///
/// Invokes the `set` trap if installed, otherwise falls through to the target.
/// Returns `true` on success, `false` if the target rejected the assignment.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_set, proxy_get};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let target = JsObject::new();
/// let mut proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_set(&mut proxy, "y", JsValue::Smi(7)).unwrap());
/// assert_eq!(proxy_get(&proxy, "y").unwrap(), JsValue::Smi(7));
/// ```
pub fn proxy_set(proxy: &mut JsProxy, key: &str, value: JsValue) -> StatorResult<bool> {
    proxy_set_with_receiver(proxy, key, value, &JsValue::Undefined)
}

/// ECMAScript §10.5.9 `[[Set]]` for Proxy with an explicit receiver.
pub fn proxy_set_with_receiver(
    proxy: &mut JsProxy,
    key: &str,
    value: JsValue,
    receiver: &JsValue,
) -> StatorResult<bool> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.set {
        let result = trap(&mut proxy.target, key, value.clone(), receiver)?;
        // Invariant §10.5.9 step 11: if the target property is non-configurable
        // and non-writable, the trap must not successfully set a different value.
        if result
            && let Some((target_val, attrs)) = proxy.target_own_property_descriptor(key)
            && !attrs.contains(PropertyAttributes::CONFIGURABLE)
            && !attrs.contains(PropertyAttributes::WRITABLE)
            && value != target_val
        {
            return Err(StatorError::TypeError(format!(
                "Proxy set trap cannot change a non-configurable, \
                 non-writable own property '{key}'"
            )));
        }
        Ok(result)
    } else {
        Ok(proxy.target.set_property(key, value).is_ok())
    }
}

// ── proxy_has ─────────────────────────────────────────────────────────────────

/// ECMAScript §10.5.7 `[[HasProperty]]` for Proxy.
///
/// Invokes the `has` trap if installed, otherwise falls through.
/// Enforces invariants:
/// * A trap may not report `false` for a non-configurable property on the
///   target.
/// * A trap may not report `false` for any property when the target is
///   non-extensible.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_has};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("p", JsValue::Null).unwrap();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_has(&proxy, "p").unwrap());
/// assert!(!proxy_has(&proxy, "q").unwrap());
/// ```
pub fn proxy_has(proxy: &JsProxy, key: &str) -> StatorResult<bool> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.has {
        trap(&proxy.target, key)?
    } else {
        return Ok(proxy.target.has_property(key));
    };
    // Invariant §10.5.7 step 10: trap may not return false for a non-configurable
    // own property of the target.
    if !result {
        if let Some((_, attrs)) = proxy.target_own_property_descriptor(key)
            && !attrs.contains(PropertyAttributes::CONFIGURABLE)
        {
            return Err(StatorError::TypeError(format!(
                "Proxy has trap returned false for non-configurable property '{key}'"
            )));
        }
        if !proxy.target_is_extensible() && proxy.target_has_own_property(key) {
            return Err(StatorError::TypeError(format!(
                "Proxy has trap returned false for existing property '{key}' \
                 on a non-extensible target"
            )));
        }
    }
    Ok(result)
}

// ── proxy_delete_property ─────────────────────────────────────────────────────

/// ECMAScript §10.5.10 `[[Delete]]` for Proxy.
///
/// Invokes the `deleteProperty` trap if installed, otherwise falls through.
/// Enforces the invariant: a trap may not return `true` for a non-configurable
/// own property of the target.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_delete_property};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("d", JsValue::Smi(1)).unwrap();
/// let mut proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_delete_property(&mut proxy, "d").unwrap());
/// ```
pub fn proxy_delete_property(proxy: &mut JsProxy, key: &str) -> StatorResult<bool> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.delete_property {
        trap(&mut proxy.target, key)?
    } else {
        return proxy.target.delete_own_property(key);
    };
    // Invariant §10.5.10 step 11: trap may not return true for a non-configurable property.
    if result
        && let Some((_, attrs)) = proxy.target_own_property_descriptor(key)
        && !attrs.contains(PropertyAttributes::CONFIGURABLE)
    {
        return Err(StatorError::TypeError(format!(
            "Proxy deleteProperty trap returned true for \
             non-configurable property '{key}'"
        )));
    }
    Ok(result)
}

// ── proxy_define_property ─────────────────────────────────────────────────────

/// ECMAScript §10.5.6 `[[DefineOwnProperty]]` for Proxy.
///
/// Invokes the `defineProperty` trap if installed, otherwise falls through.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_define_property};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::map::PropertyAttributes;
/// use stator_core::objects::value::JsValue;
///
/// let target = JsObject::new();
/// let mut proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_define_property(
///     &mut proxy,
///     "n",
///     JsValue::Smi(5),
///     PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE,
/// ).unwrap());
/// ```
pub fn proxy_define_property(
    proxy: &mut JsProxy,
    key: &str,
    value: JsValue,
    attributes: PropertyAttributes,
) -> StatorResult<bool> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.define_property {
        let result = trap(&mut proxy.target, key, value.clone(), attributes)?;
        if result {
            // Invariant §10.5.6 step 19: if the target has a non-configurable
            // property, the trap cannot redefine it as configurable.
            if let Some((target_val, target_attrs)) = proxy.target_own_property_descriptor(key)
                && !target_attrs.contains(PropertyAttributes::CONFIGURABLE)
                && (attributes.contains(PropertyAttributes::CONFIGURABLE)
                    || attributes.contains(PropertyAttributes::ENUMERABLE)
                        != target_attrs.contains(PropertyAttributes::ENUMERABLE)
                    || (!target_attrs.contains(PropertyAttributes::WRITABLE)
                        && (attributes.contains(PropertyAttributes::WRITABLE)
                            || value != target_val)))
            {
                return Err(StatorError::TypeError(format!(
                    "Proxy defineProperty trap cannot report an incompatible \
                     descriptor for non-configurable property '{key}'"
                )));
            }
            // Invariant §10.5.6 step 20: cannot add a new property if the
            // target is non-extensible.
            if !proxy.target_is_extensible() && proxy.target_own_property_descriptor(key).is_none()
            {
                return Err(StatorError::TypeError(format!(
                    "Proxy defineProperty trap cannot add property '{key}' \
                     to a non-extensible target"
                )));
            }
        }
        Ok(result)
    } else {
        Ok(proxy
            .target
            .define_own_property(key, value, attributes)
            .is_ok())
    }
}

// ── proxy_get_own_property_descriptor ────────────────────────────────────────

/// ECMAScript §10.5.5 `[[GetOwnProperty]]` for Proxy.
///
/// Invokes the `getOwnPropertyDescriptor` trap if installed, otherwise falls
/// through.
/// Enforces the invariant: a trap may not report `None` (property does not
/// exist) for a non-configurable own property of the target.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_get_own_property_descriptor};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::map::PropertyAttributes;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("k", JsValue::Smi(1)).unwrap();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_get_own_property_descriptor(&proxy, "k").unwrap().is_some());
/// assert!(proxy_get_own_property_descriptor(&proxy, "nope").unwrap().is_none());
/// ```
pub fn proxy_get_own_property_descriptor(
    proxy: &JsProxy,
    key: &str,
) -> StatorResult<Option<(JsValue, PropertyAttributes)>> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.get_own_property_descriptor {
        trap(&proxy.target, key)?
    } else {
        return Ok(proxy.target_own_property_descriptor(key));
    };
    // Invariant §10.5.5 step 16: if the trap says undefined but the target has
    // a non-configurable property, that is a violation.
    let target_desc = proxy.target_own_property_descriptor(key);
    if result.is_none() {
        if let Some((_, attrs)) = &target_desc
            && (!attrs.contains(PropertyAttributes::CONFIGURABLE) || !proxy.target_is_extensible())
        {
            return Err(StatorError::TypeError(format!(
                "Proxy getOwnPropertyDescriptor trap returned undefined \
                 for incompatible property '{key}'"
            )));
        }
    } else if let Some((_, trap_attrs)) = &result {
        if let Some((target_val, target_attrs)) = target_desc {
            if !target_attrs.contains(PropertyAttributes::CONFIGURABLE)
                && (trap_attrs.contains(PropertyAttributes::CONFIGURABLE)
                    || trap_attrs.contains(PropertyAttributes::ENUMERABLE)
                        != target_attrs.contains(PropertyAttributes::ENUMERABLE)
                    || (!target_attrs.contains(PropertyAttributes::WRITABLE)
                        && (trap_attrs.contains(PropertyAttributes::WRITABLE)
                            || result
                                .as_ref()
                                .is_some_and(|(trap_val, _)| *trap_val != target_val))))
            {
                return Err(StatorError::TypeError(format!(
                    "Proxy getOwnPropertyDescriptor trap reported incompatible \
                     descriptor for non-configurable property '{key}'"
                )));
            }
        } else if !proxy.target_is_extensible() {
            return Err(StatorError::TypeError(format!(
                "Proxy getOwnPropertyDescriptor trap reported a new property \
                 '{key}' on a non-extensible target"
            )));
        }
    }
    Ok(result)
}

// ── proxy_get_prototype_of ────────────────────────────────────────────────────

/// ECMAScript §10.5.1 `[[GetPrototypeOf]]` for Proxy.
///
/// Invokes the `getPrototypeOf` trap if installed, otherwise falls through.
///
/// # Examples
///
/// ```ignore
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_get_prototype_of};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let target = JsObject::new();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert_eq!(proxy_get_prototype_of(&proxy).unwrap(), JsValue::Null);
/// ```
pub fn proxy_get_prototype_of(proxy: &JsProxy) -> StatorResult<JsValue> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.get_prototype_of {
        let result = trap(&proxy.target)?;
        if !matches!(result, JsValue::Null) && !result.is_object_like() {
            return Err(StatorError::TypeError(
                "Proxy getPrototypeOf trap must return an object or null".to_string(),
            ));
        }
        // Invariant §10.5.1 step 9: if the target is non-extensible, the trap
        // must return the target's actual prototype.
        if !proxy.target_is_extensible() && !result.same_value(&proxy.target_prototype_value()) {
            return Err(StatorError::TypeError(
                "Proxy getPrototypeOf trap returned a different prototype \
                 than the non-extensible target's prototype"
                    .to_string(),
            ));
        }
        Ok(result)
    } else {
        Ok(proxy.target_prototype_value())
    }
}

// ── proxy_set_prototype_of ────────────────────────────────────────────────────

/// ECMAScript §10.5.2 `[[SetPrototypeOf]]` for Proxy.
///
/// Invokes the `setPrototypeOf` trap if installed, otherwise falls through.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_set_prototype_of};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let target = JsObject::new();
/// let mut proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_set_prototype_of(
///     &mut proxy,
///     JsValue::Null
/// ).unwrap());
/// ```
pub fn proxy_set_prototype_of(proxy: &mut JsProxy, proto: JsValue) -> StatorResult<bool> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.set_prototype_of {
        let result = trap(&mut proxy.target, proto.clone())?;
        // Invariant §10.5.2 step 12: if the trap returns true and the target
        // is non-extensible, the provided prototype must be the same as the
        // target's current prototype.
        if result
            && !proxy.target_is_extensible()
            && !proto.same_value(&proxy.target_prototype_value())
        {
            return Err(StatorError::TypeError(
                "Proxy setPrototypeOf trap returned true for a non-extensible \
                 target with a different prototype"
                    .to_string(),
            ));
        }
        Ok(result)
    } else {
        use crate::builtins::object::object_set_prototype_of;
        if matches!(proto, JsValue::Null) {
            Ok(object_set_prototype_of(&mut proxy.target, None).is_ok())
        } else if proto.is_object_like() {
            Ok(object_set_prototype_of(
                &mut proxy.target,
                Some(Rc::new(RefCell::new(JsObject::new()))),
            )
            .is_ok())
        } else {
            Err(StatorError::TypeError(
                "Object prototype may only be an Object or null".to_string(),
            ))
        }
    }
}

// ── proxy_is_extensible ───────────────────────────────────────────────────────

/// ECMAScript §10.5.3 `[[IsExtensible]]` for Proxy.
///
/// Invokes the `isExtensible` trap if installed, otherwise falls through.
/// Enforces the invariant: the trap result must match the target's
/// extensibility.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_is_extensible};
/// use stator_core::objects::js_object::JsObject;
///
/// let target = JsObject::new();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_is_extensible(&proxy).unwrap());
/// ```
pub fn proxy_is_extensible(proxy: &JsProxy) -> StatorResult<bool> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.is_extensible {
        trap(&proxy.target)?
    } else {
        return Ok(proxy.target.is_extensible());
    };
    // Invariant §10.5.3 step 7: result must match target.
    let target_ext = proxy.target_is_extensible();
    if result != target_ext {
        return Err(StatorError::TypeError(format!(
            "Proxy isExtensible trap returned {} but target is {}",
            result, target_ext
        )));
    }
    Ok(result)
}

// ── proxy_prevent_extensions ──────────────────────────────────────────────────

/// ECMAScript §10.5.4 `[[PreventExtensions]]` for Proxy.
///
/// Invokes the `preventExtensions` trap if installed, otherwise falls through.
/// Enforces the invariant: the trap may only return `true` if the target is
/// non-extensible.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_prevent_extensions, proxy_is_extensible};
/// use stator_core::objects::js_object::JsObject;
///
/// let target = JsObject::new();
/// let mut proxy = proxy_new(target, ProxyHandler::default());
/// assert!(proxy_prevent_extensions(&mut proxy).unwrap());
/// assert!(!proxy_is_extensible(&proxy).unwrap());
/// ```
pub fn proxy_prevent_extensions(proxy: &mut JsProxy) -> StatorResult<bool> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.prevent_extensions {
        trap(&mut proxy.target)?
    } else {
        proxy.target.prevent_extensions();
        return Ok(true);
    };
    // Invariant §10.5.4 step 8: trap may only return true if target is
    // already non-extensible.
    if result && proxy.target_is_extensible() {
        return Err(StatorError::TypeError(
            "Proxy preventExtensions trap returned true but target is still extensible".to_string(),
        ));
    }
    Ok(result)
}

// ── proxy_own_keys ────────────────────────────────────────────────────────────

/// ECMAScript §10.5.11 `[[OwnPropertyKeys]]` for Proxy.
///
/// Invokes the `ownKeys` trap if installed, otherwise falls through.
/// Enforces the invariant: when the target is non-extensible, the trap result
/// must contain every existing own key of the target and no extra keys.
///
/// # Examples
///
/// ```ignore
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_own_keys};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut target = JsObject::new();
/// target.set_property("a", JsValue::Smi(1)).unwrap();
/// let proxy = proxy_new(target, ProxyHandler::default());
/// let keys = proxy_own_keys(&proxy).unwrap();
/// assert!(keys.contains(&"a".to_string()));
/// ```
pub fn proxy_own_keys(proxy: &JsProxy) -> StatorResult<Vec<JsValue>> {
    proxy.check_revoked()?;
    let result = if let Some(trap) = &proxy.handler.own_keys {
        trap(&proxy.target)?
    } else {
        return Ok(proxy
            .target
            .own_property_keys()
            .into_iter()
            .map(|key| JsValue::String(key.into()))
            .collect());
    };

    let normalize_key = |value: &JsValue| -> StatorResult<String> {
        match value {
            JsValue::String(s) => Ok(s.to_string()),
            JsValue::Symbol(id) => Ok(crate::builtins::symbol::symbol_to_property_key(*id)),
            _ => Err(StatorError::TypeError(
                "Proxy ownKeys trap must return only strings or symbols".to_string(),
            )),
        }
    };

    let normalized_result: Vec<String> = result
        .iter()
        .map(normalize_key)
        .collect::<StatorResult<Vec<_>>>()?;
    let mut seen = HashSet::new();
    for key in &normalized_result {
        if !seen.insert(key.clone()) {
            return Err(StatorError::TypeError(format!(
                "Proxy ownKeys trap returned duplicate key '{key}'"
            )));
        }
    }

    let target_keys = proxy.target_own_property_keys();
    for tk in &target_keys {
        if let Some((_, attrs)) = proxy.target_own_property_descriptor(tk)
            && !attrs.contains(PropertyAttributes::CONFIGURABLE)
            && !normalized_result.contains(tk)
        {
            return Err(StatorError::TypeError(format!(
                "Proxy ownKeys trap omitted non-configurable own key '{tk}'"
            )));
        }
    }
    // Invariant §10.5.11 step 26/27: when target is non-extensible, result must
    // exactly cover all target own keys.
    if !proxy.target_is_extensible() {
        for tk in &target_keys {
            if !normalized_result.contains(tk) {
                return Err(StatorError::TypeError(format!(
                    "Proxy ownKeys trap omitted existing own key '{tk}' \
                     on a non-extensible target"
                )));
            }
        }
        for rk in &normalized_result {
            if !target_keys.contains(rk) {
                return Err(StatorError::TypeError(format!(
                    "Proxy ownKeys trap added extra key '{rk}' \
                     on a non-extensible target"
                )));
            }
        }
    }
    Ok(result)
}

// ── proxy_apply ───────────────────────────────────────────────────────────────

/// ECMAScript §10.5.12 `[[Call]]` for Proxy.
///
/// Invokes the `apply` trap if installed.  Returns
/// [`StatorError::TypeError`] if no trap is installed (since the underlying
/// target object is not callable in the pure-Rust model).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_apply};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut handler = ProxyHandler::default();
/// handler.apply = Some(Box::new(|_this, args| {
///     Ok(args.first().cloned().unwrap_or(JsValue::Undefined))
/// }));
/// let mut proxy = proxy_new(JsObject::new(), handler);
/// let result = proxy_apply(&mut proxy, JsValue::Undefined, vec![JsValue::Smi(9)]).unwrap();
/// assert_eq!(result, JsValue::Smi(9));
/// ```
pub fn proxy_apply(
    proxy: &mut JsProxy,
    this_argument: JsValue,
    arguments_list: Vec<JsValue>,
) -> StatorResult<JsValue> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.apply {
        trap(this_argument, arguments_list)
    } else {
        Err(StatorError::TypeError(
            "proxy_apply: target object is not callable".to_string(),
        ))
    }
}

// ── proxy_construct ───────────────────────────────────────────────────────────

/// ECMAScript §10.5.13 `[[Construct]]` for Proxy.
///
/// Invokes the `construct` trap if installed, passing both the
/// `arguments_list` and the `new_target` (§10.5.13 step 7).  Returns
/// [`StatorError::TypeError`] if no trap is installed.
///
/// # Examples
///
/// ```ignore
/// use stator_core::builtins::proxy::{ProxyHandler, proxy_new, proxy_construct};
/// use stator_core::objects::js_object::JsObject;
/// use stator_core::objects::value::JsValue;
///
/// let mut handler = ProxyHandler::default();
/// handler.construct = Some(Box::new(|args, _new_target| {
///     let mut o = PropertyMap::new();
///     if let Some(v) = args.first() { o.insert("v".to_string(), v.clone()); }
///     Ok(JsValue::PlainObject(Rc::new(RefCell::new(o))))
/// }));
/// let mut proxy = proxy_new(JsObject::new(), handler);
/// let obj = proxy_construct(&mut proxy, vec![JsValue::Smi(5)], JsValue::Undefined).unwrap();
/// assert!(matches!(obj, JsValue::PlainObject(_)));
/// ```
pub fn proxy_construct(
    proxy: &mut JsProxy,
    arguments_list: Vec<JsValue>,
    new_target: JsValue,
) -> StatorResult<JsValue> {
    proxy.check_revoked()?;
    if let Some(trap) = &proxy.handler.construct {
        let result = trap(arguments_list, new_target)?;
        if !result.is_object_like() {
            return Err(StatorError::TypeError(
                "proxy_construct: construct trap must return an object".to_string(),
            ));
        }
        Ok(result)
    } else {
        Err(StatorError::TypeError(
            "proxy_construct: target object is not constructible".to_string(),
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::objects::js_object::JsObject;
    use crate::objects::map::PropertyAttributes;
    use crate::objects::value::JsValue;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn target_with(key: &str, value: JsValue) -> JsObject {
        let mut t = JsObject::new();
        t.set_property(key, value).unwrap();
        t
    }

    // ── proxy_new / is_revoked ────────────────────────────────────────────────

    #[test]
    fn test_proxy_new_not_revoked() {
        let p = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(!p.is_revoked());
    }

    // ── proxy_revocable / proxy_revoke ────────────────────────────────────────

    #[test]
    fn test_proxy_revoke_sets_revoked_flag() {
        let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
        assert!(!p.is_revoked());
        proxy_revoke(&mut p);
        assert!(p.is_revoked());
    }

    #[test]
    fn test_revoked_proxy_get_returns_error() {
        let mut p = proxy_revocable(target_with("x", JsValue::Smi(1)), ProxyHandler::default());
        proxy_revoke(&mut p);
        assert!(matches!(proxy_get(&p, "x"), Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_revoked_proxy_set_returns_error() {
        let mut p = proxy_revocable(JsObject::new(), ProxyHandler::default());
        proxy_revoke(&mut p);
        assert!(matches!(
            proxy_set(&mut p, "x", JsValue::Smi(1)),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_revoked_proxy_has_returns_error() {
        let p = {
            let mut tmp =
                proxy_revocable(target_with("x", JsValue::Smi(1)), ProxyHandler::default());
            proxy_revoke(&mut tmp);
            tmp
        };
        assert!(matches!(proxy_has(&p, "x"), Err(StatorError::TypeError(_))));
    }

    // ── proxy_get ─────────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_get_no_trap_falls_through() {
        let proxy = proxy_new(target_with("k", JsValue::Smi(42)), ProxyHandler::default());
        assert_eq!(proxy_get(&proxy, "k").unwrap(), JsValue::Smi(42));
    }

    #[test]
    fn test_proxy_get_trap_overrides() {
        let mut handler = ProxyHandler::default();
        handler.get = Some(Box::new(|_t, _k, _r| Ok(JsValue::Smi(99))));
        let proxy = proxy_new(JsObject::new(), handler);
        assert_eq!(proxy_get(&proxy, "anything").unwrap(), JsValue::Smi(99));
    }

    #[test]
    fn test_proxy_get_invariant_non_configurable_non_writable() {
        // Target has a non-configurable, non-writable property = 1.
        // Trap returns 2 → invariant violation.
        let mut target = JsObject::new();
        target
            .define_own_property("ro", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.get = Some(Box::new(|_t, _k, _r| Ok(JsValue::Smi(2))));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_get(&proxy, "ro"),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_get_invariant_same_value_allowed() {
        let mut target = JsObject::new();
        target
            .define_own_property("ro", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.get = Some(Box::new(|_t, _k, _r| Ok(JsValue::Smi(1))));
        let proxy = proxy_new(target, handler);
        assert_eq!(proxy_get(&proxy, "ro").unwrap(), JsValue::Smi(1));
    }

    // ── proxy_set ─────────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_set_no_trap_falls_through() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(proxy_set(&mut proxy, "y", JsValue::Smi(7)).unwrap());
        assert_eq!(proxy_get(&proxy, "y").unwrap(), JsValue::Smi(7));
    }

    #[test]
    fn test_proxy_set_trap_overrides() {
        let mut handler = ProxyHandler::default();
        handler.set = Some(Box::new(|_t, _k, _v, _r| Ok(false)));
        let mut proxy = proxy_new(JsObject::new(), handler);
        assert!(!proxy_set(&mut proxy, "z", JsValue::Smi(1)).unwrap());
    }

    #[test]
    fn test_proxy_set_invariant_non_configurable_non_writable() {
        let mut target = JsObject::new();
        target
            .define_own_property("ro", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.set = Some(Box::new(|_t, _k, _v, _r| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_set(&mut proxy, "ro", JsValue::Smi(2)),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_set_invariant_same_value_allowed() {
        let mut target = JsObject::new();
        target
            .define_own_property("ro", JsValue::Smi(1), PropertyAttributes::empty())
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.set = Some(Box::new(|_t, _k, _v, _r| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(proxy_set(&mut proxy, "ro", JsValue::Smi(1)).unwrap());
    }

    // ── proxy_has ─────────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_has_no_trap_falls_through() {
        let proxy = proxy_new(target_with("p", JsValue::Null), ProxyHandler::default());
        assert!(proxy_has(&proxy, "p").unwrap());
        assert!(!proxy_has(&proxy, "q").unwrap());
    }

    #[test]
    fn test_proxy_has_trap_overrides() {
        let mut handler = ProxyHandler::default();
        handler.has = Some(Box::new(|_t, _k| Ok(true)));
        let proxy = proxy_new(JsObject::new(), handler);
        assert!(proxy_has(&proxy, "anything").unwrap());
    }

    #[test]
    fn test_proxy_has_invariant_non_configurable_property() {
        let mut target = JsObject::new();
        target
            .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.has = Some(Box::new(|_t, _k| Ok(false)));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_has(&proxy, "nc"),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_has_invariant_non_extensible_target() {
        let mut target = target_with("x", JsValue::Smi(1));
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.has = Some(Box::new(|_t, _k| Ok(false)));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_has(&proxy, "x"),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_delete_property ─────────────────────────────────────────────────

    #[test]
    fn test_proxy_delete_property_no_trap() {
        let mut proxy = proxy_new(target_with("d", JsValue::Smi(1)), ProxyHandler::default());
        assert!(proxy_delete_property(&mut proxy, "d").unwrap());
    }

    #[test]
    fn test_proxy_delete_property_trap_overrides() {
        let mut handler = ProxyHandler::default();
        handler.delete_property = Some(Box::new(|_t, _k| Ok(false)));
        let mut proxy = proxy_new(target_with("d", JsValue::Smi(1)), ProxyHandler::default());
        proxy.handler = handler;
        assert!(!proxy_delete_property(&mut proxy, "d").unwrap());
    }

    #[test]
    fn test_proxy_delete_property_invariant_non_configurable() {
        let mut target = JsObject::new();
        target
            .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.delete_property = Some(Box::new(|_t, _k| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_delete_property(&mut proxy, "nc"),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_define_property ─────────────────────────────────────────────────

    #[test]
    fn test_proxy_define_property_no_trap() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(
            proxy_define_property(
                &mut proxy,
                "m",
                JsValue::Smi(3),
                PropertyAttributes::WRITABLE
                    | PropertyAttributes::ENUMERABLE
                    | PropertyAttributes::CONFIGURABLE,
            )
            .unwrap()
        );
        assert_eq!(proxy.target.get_own_property("m"), Some(JsValue::Smi(3)));
    }

    #[test]
    fn test_proxy_define_property_trap_overrides() {
        let mut handler = ProxyHandler::default();
        handler.define_property = Some(Box::new(|_t, _k, _v, _a| Ok(false)));
        let mut proxy = proxy_new(JsObject::new(), handler);
        assert!(
            !proxy_define_property(
                &mut proxy,
                "m",
                JsValue::Smi(1),
                PropertyAttributes::WRITABLE,
            )
            .unwrap()
        );
    }

    #[test]
    fn test_proxy_define_property_invariant_non_configurable_to_configurable() {
        let mut target = JsObject::new();
        target
            .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.define_property = Some(Box::new(|_t, _k, _v, _a| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_define_property(
                &mut proxy,
                "nc",
                JsValue::Smi(2),
                PropertyAttributes::CONFIGURABLE,
            ),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_define_property_invariant_non_extensible_new_property() {
        let mut target = JsObject::new();
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.define_property = Some(Box::new(|_t, _k, _v, _a| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_define_property(
                &mut proxy,
                "new_key",
                JsValue::Smi(1),
                PropertyAttributes::WRITABLE,
            ),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_get_own_property_descriptor ─────────────────────────────────────

    #[test]
    fn test_proxy_get_own_property_descriptor_no_trap() {
        let proxy = proxy_new(target_with("k", JsValue::Smi(1)), ProxyHandler::default());
        assert!(
            proxy_get_own_property_descriptor(&proxy, "k")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn test_proxy_get_own_property_descriptor_invariant() {
        let mut target = JsObject::new();
        target
            .define_own_property("nc", JsValue::Smi(1), PropertyAttributes::WRITABLE)
            .unwrap();
        let mut handler = ProxyHandler::default();
        handler.get_own_property_descriptor = Some(Box::new(|_t, _k| Ok(None)));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_get_own_property_descriptor(&proxy, "nc"),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_get_prototype_of ────────────────────────────────────────────────

    #[test]
    fn test_proxy_get_prototype_of_no_trap_no_prototype() {
        let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert_eq!(proxy_get_prototype_of(&proxy).unwrap(), JsValue::Null);
    }

    #[test]
    fn test_proxy_get_prototype_of_trap_overrides() {
        let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let proto_clone = proto.clone();
        let mut handler = ProxyHandler::default();
        handler.get_prototype_of = Some(Box::new(move |_t| Ok(proto_clone.clone())));
        let proxy = proxy_new(JsObject::new(), handler);
        assert!(proxy_get_prototype_of(&proxy).unwrap().same_value(&proto));
    }

    #[test]
    fn test_proxy_get_prototype_of_invariant_non_extensible() {
        // Non-extensible target with no prototype; trap returns Some → violation.
        let mut target = JsObject::new();
        target.prevent_extensions();
        let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let proto_clone = proto.clone();
        let mut handler = ProxyHandler::default();
        handler.get_prototype_of = Some(Box::new(move |_t| Ok(proto_clone.clone())));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_get_prototype_of(&proxy),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_get_prototype_of_invariant_non_extensible_same_ok() {
        // Non-extensible target with null prototype; trap returns the same → OK.
        let mut target = JsObject::new();
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.get_prototype_of = Some(Box::new(move |_t| Ok(JsValue::Null)));
        let proxy = proxy_new(target, handler);
        assert_eq!(proxy_get_prototype_of(&proxy).unwrap(), JsValue::Null);
    }

    // ── proxy_set_prototype_of ────────────────────────────────────────────────

    #[test]
    fn test_proxy_set_prototype_of_no_trap() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert!(proxy_set_prototype_of(&mut proxy, proto).unwrap());
        assert!(proxy.target.prototype().is_some());
    }

    #[test]
    fn test_proxy_set_prototype_of_invariant_non_extensible_different() {
        // Non-extensible target; trap returns true with a different prototype → violation.
        let mut target = JsObject::new();
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.set_prototype_of = Some(Box::new(|_t, _p| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        let new_proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert!(matches!(
            proxy_set_prototype_of(&mut proxy, new_proto),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_set_prototype_of_invariant_non_extensible_null_ok() {
        // Non-extensible target with no prototype; trap returns true with None → OK.
        let mut target = JsObject::new();
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.set_prototype_of = Some(Box::new(|_t, _p| Ok(true)));
        let mut proxy = proxy_new(target, handler);
        assert!(proxy_set_prototype_of(&mut proxy, JsValue::Null).unwrap());
    }

    // ── proxy_is_extensible ───────────────────────────────────────────────────

    #[test]
    fn test_proxy_is_extensible_no_trap() {
        let proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(proxy_is_extensible(&proxy).unwrap());
    }

    #[test]
    fn test_proxy_is_extensible_invariant_violation() {
        // Target is extensible but trap says false → violation.
        let mut handler = ProxyHandler::default();
        handler.is_extensible = Some(Box::new(|_t| Ok(false)));
        let proxy = proxy_new(JsObject::new(), handler); // target is extensible
        assert!(matches!(
            proxy_is_extensible(&proxy),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_is_extensible_invariant_consistent() {
        // Target is extensible and trap says true → OK.
        let mut handler = ProxyHandler::default();
        handler.is_extensible = Some(Box::new(|_t| Ok(true)));
        let proxy = proxy_new(JsObject::new(), handler);
        assert!(proxy_is_extensible(&proxy).unwrap());
    }

    // ── proxy_prevent_extensions ──────────────────────────────────────────────

    #[test]
    fn test_proxy_prevent_extensions_no_trap() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(proxy_prevent_extensions(&mut proxy).unwrap());
        assert!(!proxy.target.is_extensible());
    }

    #[test]
    fn test_proxy_prevent_extensions_invariant_trap_true_but_target_still_extensible() {
        let mut handler = ProxyHandler::default();
        // Trap returns true without actually preventing extensions on the target.
        handler.prevent_extensions = Some(Box::new(|_t| Ok(true)));
        let mut proxy = proxy_new(JsObject::new(), handler); // target stays extensible
        assert!(matches!(
            proxy_prevent_extensions(&mut proxy),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_own_keys ────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_own_keys_no_trap() {
        let mut target = JsObject::new();
        target.set_property("a", JsValue::Smi(1)).unwrap();
        let proxy = proxy_new(target, ProxyHandler::default());
        let keys = proxy_own_keys(&proxy).unwrap();
        assert!(keys.contains(&JsValue::String("a".into())));
    }

    #[test]
    fn test_proxy_own_keys_invariant_non_extensible_missing_key() {
        let mut target = target_with("a", JsValue::Smi(1));
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.own_keys = Some(Box::new(|_t| Ok(vec![])));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_own_keys(&proxy),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_proxy_own_keys_invariant_non_extensible_extra_key() {
        let mut target = target_with("a", JsValue::Smi(1));
        target.prevent_extensions();
        let mut handler = ProxyHandler::default();
        handler.own_keys = Some(Box::new(|_t| {
            Ok(vec![
                JsValue::String("a".into()),
                JsValue::String("b".into()),
            ])
        }));
        let proxy = proxy_new(target, handler);
        assert!(matches!(
            proxy_own_keys(&proxy),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_apply ───────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_apply_with_trap() {
        let mut handler = ProxyHandler::default();
        handler.apply = Some(Box::new(|_this, args| {
            Ok(args.first().cloned().unwrap_or(JsValue::Undefined))
        }));
        let mut proxy = proxy_new(JsObject::new(), handler);
        let r = proxy_apply(&mut proxy, JsValue::Undefined, vec![JsValue::Smi(5)]).unwrap();
        assert_eq!(r, JsValue::Smi(5));
    }

    #[test]
    fn test_proxy_apply_no_trap_returns_type_error() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(matches!(
            proxy_apply(&mut proxy, JsValue::Undefined, vec![]),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── proxy_construct ───────────────────────────────────────────────────────

    #[test]
    fn test_proxy_construct_with_trap() {
        let mut handler = ProxyHandler::default();
        handler.construct = Some(Box::new(|args, _new_target| {
            let mut o = PropertyMap::new();
            if let Some(v) = args.first() {
                o.insert("v".to_string(), v.clone());
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(o))))
        }));
        let mut proxy = proxy_new(JsObject::new(), handler);
        let obj = proxy_construct(&mut proxy, vec![JsValue::Smi(3)], JsValue::Undefined).unwrap();
        let JsValue::PlainObject(map) = obj else {
            panic!("expected plain object");
        };
        assert_eq!(map.borrow().get("v"), Some(&JsValue::Smi(3)));
    }

    #[test]
    fn test_proxy_construct_no_trap_returns_type_error() {
        let mut proxy = proxy_new(JsObject::new(), ProxyHandler::default());
        assert!(matches!(
            proxy_construct(&mut proxy, vec![], JsValue::Undefined),
            Err(StatorError::TypeError(_))
        ));
    }

    // ── Proxy ownKeys trap order conformance ─────────────────────────────

    #[test]
    fn test_proxy_own_keys_no_trap_follows_spec_order() {
        let mut target = JsObject::new();
        target.set_property("b", JsValue::Smi(1)).unwrap();
        target.set_property("3", JsValue::Smi(2)).unwrap();
        target.set_property("a", JsValue::Smi(3)).unwrap();
        target.set_property("0", JsValue::Smi(4)).unwrap();
        let proxy = proxy_new(target, ProxyHandler::default());
        let keys = proxy_own_keys(&proxy).unwrap();
        let key_strings: Vec<String> = keys
            .iter()
            .map(|k| match k {
                JsValue::String(s) => s.to_string(),
                _ => panic!("expected string key"),
            })
            .collect();
        assert_eq!(key_strings, &["0", "3", "b", "a"]);
    }

    #[test]
    fn test_proxy_own_keys_trap_order_respected() {
        let mut target = JsObject::new();
        target.set_property("a", JsValue::Smi(1)).unwrap();
        target.set_property("b", JsValue::Smi(2)).unwrap();
        let mut handler = ProxyHandler::default();
        handler.own_keys = Some(Box::new(|_target| {
            Ok(vec![
                JsValue::String("b".into()),
                JsValue::String("a".into()),
            ])
        }));
        let proxy = proxy_new(target, handler);
        let keys = proxy_own_keys(&proxy).unwrap();
        let key_strings: Vec<String> = keys
            .iter()
            .map(|k| match k {
                JsValue::String(s) => s.to_string(),
                _ => panic!("expected string key"),
            })
            .collect();
        // Trap order should be preserved
        assert_eq!(key_strings, &["b", "a"]);
    }
}

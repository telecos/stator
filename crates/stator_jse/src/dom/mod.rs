//! DOM integration layer for bridging Stator JS engine with web platform APIs.
//!
//! This module provides the infrastructure that allows a C++ embedder (such as
//! a browser's DOM implementation) to expose native DOM nodes as JavaScript
//! objects.  The design mirrors V8's `ObjectTemplate` / `FunctionTemplate`
//! approach:
//!
//! - **Object wrapping** — [`DomObjectWrap`] pairs a JS object with a fixed
//!   number of *internal fields* that store opaque `*mut c_void` pointers back
//!   to the C++ DOM node.
//! - **Property interceptors** — [`NamedPropertyHandlerConfig`] and
//!   [`IndexedPropertyHandlerConfig`] let the embedder intercept property
//!   reads/writes so that `element.id` or `nodeList[0]` call into native code.
//! - **Weak references** — [`DomWeakRef`] wraps a pointer to a
//!   [`DomObjectWrap`] with an invoke-on-GC weak callback so that the C++ side
//!   can release the backing DOM node when the JS wrapper is collected.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Interceptor callback types
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by a named-property getter interceptor.
///
/// `Some(value)` means the interceptor handled the access; `None` means the
/// engine should fall through to the object's own properties.
pub type NamedGetterResult = Option<JsValue>;

/// Result returned by a named-property setter interceptor.
///
/// `true` means the interceptor handled the write; `false` means the engine
/// should fall through to the normal property store.
pub type NamedSetterResult = bool;

/// Callback for intercepting named-property **get** (e.g. `element.id`).
pub type NamedGetterCallback = Box<dyn Fn(&str, *mut c_void) -> NamedGetterResult>;

/// Callback for intercepting named-property **set** (e.g. `element.id = "x"`).
pub type NamedSetterCallback = Box<dyn Fn(&str, &JsValue, *mut c_void) -> NamedSetterResult>;

/// Callback for intercepting named-property **query** (e.g. `"id" in element`).
///
/// Returns `Some(attributes)` if the property exists, `None` otherwise.
/// The integer encodes `v8::PropertyAttribute` flags (0 = `None`).
pub type NamedQueryCallback = Box<dyn Fn(&str, *mut c_void) -> Option<u32>>;

/// Callback for intercepting named-property **delete**.
pub type NamedDeleterCallback = Box<dyn Fn(&str, *mut c_void) -> bool>;

/// Callback that enumerates intercepted named properties.
pub type NamedEnumeratorCallback = Box<dyn Fn(*mut c_void) -> Vec<String>>;

/// Result returned by an indexed-property getter interceptor.
pub type IndexedGetterResult = Option<JsValue>;

/// Result returned by an indexed-property setter interceptor.
pub type IndexedSetterResult = bool;

/// Callback for intercepting indexed-property **get** (e.g. `nodeList[0]`).
pub type IndexedGetterCallback = Box<dyn Fn(u32, *mut c_void) -> IndexedGetterResult>;

/// Callback for intercepting indexed-property **set** (e.g. `nodeList[0] = x`).
pub type IndexedSetterCallback = Box<dyn Fn(u32, &JsValue, *mut c_void) -> IndexedSetterResult>;

/// Callback for intercepting indexed-property **query**.
pub type IndexedQueryCallback = Box<dyn Fn(u32, *mut c_void) -> Option<u32>>;

/// Callback for intercepting indexed-property **length** query.
pub type IndexedLengthCallback = Box<dyn Fn(*mut c_void) -> u32>;

/// Borrowed named-property getter reference.
type NamedGetterRef<'a> = Option<&'a dyn Fn(&str, *mut c_void) -> NamedGetterResult>;

/// Borrowed named-property setter reference.
type NamedSetterRef<'a> = Option<&'a dyn Fn(&str, &JsValue, *mut c_void) -> NamedSetterResult>;

/// Borrowed named-property query reference.
type NamedQueryRef<'a> = Option<&'a dyn Fn(&str, *mut c_void) -> Option<u32>>;

/// Borrowed named-property deleter reference.
type NamedDeleterRef<'a> = Option<&'a dyn Fn(&str, *mut c_void) -> bool>;

/// Borrowed indexed-property setter reference.
type IndexedSetterRef<'a> = Option<&'a dyn Fn(u32, &JsValue, *mut c_void) -> IndexedSetterResult>;

// ─────────────────────────────────────────────────────────────────────────────
// PropertyHandlerFlags
// ─────────────────────────────────────────────────────────────────────────────

/// Bitmask flags that modify the behaviour of a
/// [`NamedPropertyHandlerConfig`], mirroring V8/Blink's
/// `v8::PropertyHandlerFlags` for named interceptors.
///
/// Flags are an additive, backwards-compatible refinement of interceptor
/// semantics: the default value [`NamedPropertyHandlerFlags::NONE`]
/// preserves the historical behaviour, and any combination of the
/// documented bits below is valid.  Setting unknown bits is an error
/// reported by [`NamedPropertyHandlerFlags::validate`].
///
/// # Supported semantics
///
/// | Flag | Effect |
/// |------|--------|
/// | [`NamedPropertyHandlerFlags::NONE`] | Default; the interceptor is consulted before own properties on every operation. |
/// | [`NamedPropertyHandlerFlags::ALL_CAN_READ`] | Read-side operations (`get`, `query`) still consult the interceptor even when the installed access-check callback would otherwise deny the access.  Writes/deletes/enumeration remain access-checked. |
/// | [`NamedPropertyHandlerFlags::NON_MASKING`] | The interceptor is consulted **only** when the wrapper's own-property map does not already define the key.  Own properties therefore "mask" interceptor entries. |
/// | [`NamedPropertyHandlerFlags::ONLY_INTERCEPT_STRINGS`] | Stored-only metadata.  Stator does not yet route symbol-keyed access through DOM wrappers, so this flag is already implicitly the steady-state behaviour.  Documented for future symbol support. |
/// | [`NamedPropertyHandlerFlags::INTERCEPT_SYMBOLS`] | Stored-only metadata.  Documents that the embedder *wants* symbol keys forwarded to the interceptor once Stator routes them; today it has no runtime effect. |
/// | [`NamedPropertyHandlerFlags::HAS_NO_SIDE_EFFECT`] | Stored-only metadata, intended for profilers/debuggers that need to call interceptors without observable side effects.  No runtime effect today. |
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NamedPropertyHandlerFlags(u32);

impl NamedPropertyHandlerFlags {
    /// No flags set; default V8-compatible interceptor behaviour.
    pub const NONE: Self = Self(0);
    /// Read-side operations bypass an access-check denial.
    pub const ALL_CAN_READ: Self = Self(1 << 0);
    /// Interceptor only fires when own properties don't already define the key.
    pub const NON_MASKING: Self = Self(1 << 1);
    /// Stored-only metadata; the interceptor is intended to fire only for string keys.
    pub const ONLY_INTERCEPT_STRINGS: Self = Self(1 << 2);
    /// Stored-only metadata; the interceptor is intended to fire for symbol keys as well.
    pub const INTERCEPT_SYMBOLS: Self = Self(1 << 3);
    /// Stored-only metadata; the interceptor is side-effect-free for diagnostics use.
    pub const HAS_NO_SIDE_EFFECT: Self = Self(1 << 4);

    /// Bitmask of every flag bit recognised by this Stator build.  Setting
    /// any other bit will cause [`NamedPropertyHandlerFlags::validate`] to
    /// reject the value.
    pub const ALL: Self = Self(
        Self::ALL_CAN_READ.0
            | Self::NON_MASKING.0
            | Self::ONLY_INTERCEPT_STRINGS.0
            | Self::INTERCEPT_SYMBOLS.0
            | Self::HAS_NO_SIDE_EFFECT.0,
    );

    /// Construct a flag set from a raw bitmask **without** validation.
    /// Prefer [`NamedPropertyHandlerFlags::from_bits`] for embedder-supplied
    /// values.
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Construct a flag set from a raw bitmask, rejecting any unrecognised
    /// bits (fail-closed) so embedders cannot silently activate semantics
    /// the current Stator build does not understand.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL.0 != 0 {
            None
        } else {
            Some(Self(bits))
        }
    }

    /// Return the underlying bitmask.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Return `true` if every bit in `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Bitwise union of two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Return `Ok(())` when only documented bits are set, else
    /// `Err(unknown_bits)`.
    pub const fn validate(self) -> Result<(), u32> {
        let bad = self.0 & !Self::ALL.0;
        if bad != 0 { Err(bad) } else { Ok(()) }
    }
}

impl std::ops::BitOr for NamedPropertyHandlerFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Bitmask flags that modify the behaviour of an
/// [`IndexedPropertyHandlerConfig`], mirroring V8/Blink's
/// `v8::PropertyHandlerFlags` for indexed interceptors.
///
/// Indexed interceptors never see string keys, so the
/// `OnlyInterceptStrings`/`InterceptSymbols` flags do not apply.  All
/// remaining flags carry the same semantics as their named counterparts
/// (see [`NamedPropertyHandlerFlags`]), restricted to indexed operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct IndexedPropertyHandlerFlags(u32);

impl IndexedPropertyHandlerFlags {
    /// No flags set; default V8-compatible interceptor behaviour.
    pub const NONE: Self = Self(0);
    /// Read-side operations bypass an access-check denial.
    pub const ALL_CAN_READ: Self = Self(1 << 0);
    /// Interceptor only fires when own properties don't already define the key.
    /// Indexed wrappers do not maintain a per-index own-property map today,
    /// so this flag is stored-only and reserved for future symmetry with
    /// named interceptors.
    pub const NON_MASKING: Self = Self(1 << 1);
    /// Stored-only metadata; the interceptor is side-effect-free for
    /// diagnostics use.  No runtime effect today.
    pub const HAS_NO_SIDE_EFFECT: Self = Self(1 << 4);

    /// Bitmask of every flag bit recognised by this Stator build.
    pub const ALL: Self =
        Self(Self::ALL_CAN_READ.0 | Self::NON_MASKING.0 | Self::HAS_NO_SIDE_EFFECT.0);

    /// Construct a flag set from a raw bitmask **without** validation.
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Construct a flag set from a raw bitmask, rejecting any unrecognised
    /// bits (fail-closed).
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL.0 != 0 {
            None
        } else {
            Some(Self(bits))
        }
    }

    /// Return the underlying bitmask.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Return `true` if every bit in `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Bitwise union of two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Return `Ok(())` when only documented bits are set, else
    /// `Err(unknown_bits)`.
    pub const fn validate(self) -> Result<(), u32> {
        let bad = self.0 & !Self::ALL.0;
        if bad != 0 { Err(bad) } else { Ok(()) }
    }
}

impl std::ops::BitOr for IndexedPropertyHandlerFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NamedPropertyHandlerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for intercepting named-property access on a DOM-wrapped
/// object, analogous to `v8::NamedPropertyHandlerConfiguration`.
///
/// Only the getter is required; all other callbacks default to no-op
/// fall-throughs when left as `None`.
///
/// # Example
///
/// ```
/// use stator_jse::dom::NamedPropertyHandlerConfig;
/// use stator_jse::objects::value::JsValue;
///
/// let cfg = NamedPropertyHandlerConfig::builder()
///     .getter(|name, _data| {
///         if name == "id" { Some(JsValue::String("my-div".into())) }
///         else { None }
///     })
///     .build();
///
/// assert!(cfg.getter().is_some());
/// assert!(cfg.setter().is_none());
/// ```
pub struct NamedPropertyHandlerConfig {
    getter: Option<NamedGetterCallback>,
    setter: Option<NamedSetterCallback>,
    query: Option<NamedQueryCallback>,
    deleter: Option<NamedDeleterCallback>,
    enumerator: Option<NamedEnumeratorCallback>,
    flags: NamedPropertyHandlerFlags,
}

impl NamedPropertyHandlerConfig {
    /// Create a [`NamedPropertyHandlerConfigBuilder`].
    pub fn builder() -> NamedPropertyHandlerConfigBuilder {
        NamedPropertyHandlerConfigBuilder::default()
    }

    /// Return a reference to the getter callback, if installed.
    pub fn getter(&self) -> NamedGetterRef<'_> {
        self.getter.as_deref()
    }

    /// Return a reference to the setter callback, if installed.
    pub fn setter(&self) -> NamedSetterRef<'_> {
        self.setter.as_deref()
    }

    /// Return a reference to the query callback, if installed.
    pub fn query(&self) -> NamedQueryRef<'_> {
        self.query.as_deref()
    }

    /// Return a reference to the deleter callback, if installed.
    pub fn deleter(&self) -> NamedDeleterRef<'_> {
        self.deleter.as_deref()
    }

    /// Return a reference to the enumerator callback, if installed.
    pub fn enumerator(&self) -> Option<&dyn Fn(*mut c_void) -> Vec<String>> {
        self.enumerator.as_deref()
    }

    /// Return the flag bitset configured on this handler.
    pub fn flags(&self) -> NamedPropertyHandlerFlags {
        self.flags
    }
}

impl std::fmt::Debug for NamedPropertyHandlerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedPropertyHandlerConfig")
            .field("has_getter", &self.getter.is_some())
            .field("has_setter", &self.setter.is_some())
            .field("has_query", &self.query.is_some())
            .field("has_deleter", &self.deleter.is_some())
            .field("has_enumerator", &self.enumerator.is_some())
            .field("flags", &self.flags)
            .finish()
    }
}

/// Builder for [`NamedPropertyHandlerConfig`].
#[derive(Default)]
pub struct NamedPropertyHandlerConfigBuilder {
    getter: Option<NamedGetterCallback>,
    setter: Option<NamedSetterCallback>,
    query: Option<NamedQueryCallback>,
    deleter: Option<NamedDeleterCallback>,
    enumerator: Option<NamedEnumeratorCallback>,
    flags: NamedPropertyHandlerFlags,
}

impl NamedPropertyHandlerConfigBuilder {
    /// Install a named-property getter interceptor.
    pub fn getter(mut self, cb: impl Fn(&str, *mut c_void) -> NamedGetterResult + 'static) -> Self {
        self.getter = Some(Box::new(cb));
        self
    }

    /// Install a named-property setter interceptor.
    pub fn setter(
        mut self,
        cb: impl Fn(&str, &JsValue, *mut c_void) -> NamedSetterResult + 'static,
    ) -> Self {
        self.setter = Some(Box::new(cb));
        self
    }

    /// Install a named-property query interceptor.
    pub fn query(mut self, cb: impl Fn(&str, *mut c_void) -> Option<u32> + 'static) -> Self {
        self.query = Some(Box::new(cb));
        self
    }

    /// Install a named-property deleter interceptor.
    pub fn deleter(mut self, cb: impl Fn(&str, *mut c_void) -> bool + 'static) -> Self {
        self.deleter = Some(Box::new(cb));
        self
    }

    /// Install a named-property enumerator callback.
    pub fn enumerator(mut self, cb: impl Fn(*mut c_void) -> Vec<String> + 'static) -> Self {
        self.enumerator = Some(Box::new(cb));
        self
    }

    /// Set the [`NamedPropertyHandlerFlags`] bitmask that modifies the
    /// behaviour of the configured interceptors.  Replaces any previously
    /// set flags.
    pub fn flags(mut self, flags: NamedPropertyHandlerFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Consume the builder and return the finished configuration.
    pub fn build(self) -> NamedPropertyHandlerConfig {
        NamedPropertyHandlerConfig {
            getter: self.getter,
            setter: self.setter,
            query: self.query,
            deleter: self.deleter,
            enumerator: self.enumerator,
            flags: self.flags,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexedPropertyHandlerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for intercepting indexed-property access on a DOM-wrapped
/// object (e.g. `NodeList[0]`, `HTMLCollection[i]`), analogous to
/// `v8::IndexedPropertyHandlerConfiguration`.
///
/// # Example
///
/// ```
/// use stator_jse::dom::IndexedPropertyHandlerConfig;
/// use stator_jse::objects::value::JsValue;
///
/// let cfg = IndexedPropertyHandlerConfig::builder()
///     .getter(|index, _data| {
///         if index == 0 { Some(JsValue::Smi(42)) }
///         else { None }
///     })
///     .length(|_data| 1)
///     .build();
///
/// assert!(cfg.getter().is_some());
/// assert!(cfg.length().is_some());
/// ```
pub struct IndexedPropertyHandlerConfig {
    getter: Option<IndexedGetterCallback>,
    setter: Option<IndexedSetterCallback>,
    query: Option<IndexedQueryCallback>,
    length: Option<IndexedLengthCallback>,
    flags: IndexedPropertyHandlerFlags,
}

impl IndexedPropertyHandlerConfig {
    /// Create an [`IndexedPropertyHandlerConfigBuilder`].
    pub fn builder() -> IndexedPropertyHandlerConfigBuilder {
        IndexedPropertyHandlerConfigBuilder::default()
    }

    /// Return a reference to the getter callback, if installed.
    pub fn getter(&self) -> Option<&dyn Fn(u32, *mut c_void) -> IndexedGetterResult> {
        self.getter.as_deref()
    }

    /// Return a reference to the setter callback, if installed.
    pub fn setter(&self) -> IndexedSetterRef<'_> {
        self.setter.as_deref()
    }

    /// Return a reference to the query callback, if installed.
    pub fn query(&self) -> Option<&dyn Fn(u32, *mut c_void) -> Option<u32>> {
        self.query.as_deref()
    }

    /// Return a reference to the length callback, if installed.
    pub fn length(&self) -> Option<&dyn Fn(*mut c_void) -> u32> {
        self.length.as_deref()
    }

    /// Return the flag bitset configured on this handler.
    pub fn flags(&self) -> IndexedPropertyHandlerFlags {
        self.flags
    }
}

impl std::fmt::Debug for IndexedPropertyHandlerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedPropertyHandlerConfig")
            .field("has_getter", &self.getter.is_some())
            .field("has_setter", &self.setter.is_some())
            .field("has_query", &self.query.is_some())
            .field("has_length", &self.length.is_some())
            .field("flags", &self.flags)
            .finish()
    }
}

/// Builder for [`IndexedPropertyHandlerConfig`].
#[derive(Default)]
pub struct IndexedPropertyHandlerConfigBuilder {
    getter: Option<IndexedGetterCallback>,
    setter: Option<IndexedSetterCallback>,
    query: Option<IndexedQueryCallback>,
    length: Option<IndexedLengthCallback>,
    flags: IndexedPropertyHandlerFlags,
}

impl IndexedPropertyHandlerConfigBuilder {
    /// Install an indexed-property getter interceptor.
    pub fn getter(
        mut self,
        cb: impl Fn(u32, *mut c_void) -> IndexedGetterResult + 'static,
    ) -> Self {
        self.getter = Some(Box::new(cb));
        self
    }

    /// Install an indexed-property setter interceptor.
    pub fn setter(
        mut self,
        cb: impl Fn(u32, &JsValue, *mut c_void) -> IndexedSetterResult + 'static,
    ) -> Self {
        self.setter = Some(Box::new(cb));
        self
    }

    /// Install an indexed-property query interceptor.
    pub fn query(mut self, cb: impl Fn(u32, *mut c_void) -> Option<u32> + 'static) -> Self {
        self.query = Some(Box::new(cb));
        self
    }

    /// Install an indexed-property length callback.
    pub fn length(mut self, cb: impl Fn(*mut c_void) -> u32 + 'static) -> Self {
        self.length = Some(Box::new(cb));
        self
    }

    /// Set the [`IndexedPropertyHandlerFlags`] bitmask that modifies the
    /// behaviour of the configured interceptors.  Replaces any previously
    /// set flags.
    pub fn flags(mut self, flags: IndexedPropertyHandlerFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Consume the builder and return the finished configuration.
    pub fn build(self) -> IndexedPropertyHandlerConfig {
        IndexedPropertyHandlerConfig {
            getter: self.getter,
            setter: self.setter,
            query: self.query,
            length: self.length,
            flags: self.flags,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Access-check (security) callback
// ─────────────────────────────────────────────────────────────────────────────

/// Operation being attempted on a DOM wrapper, passed to an
/// [`AccessCheckCallback`] so the embedder can apply a policy decision
/// (e.g. cross-origin allow/deny) before the engine consults the
/// interceptor or the wrapper's own properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessCheckOperation {
    /// Named-property read (e.g. `element.id`).
    NamedGet,
    /// Named-property write (e.g. `element.id = "x"`).
    NamedSet,
    /// Named-property `in`/has query (e.g. `"id" in element`).
    NamedQuery,
    /// Named-property `delete` (e.g. `delete element.id`).
    NamedDelete,
    /// Named-property enumeration (e.g. `for-in`, `Object.keys`).
    NamedEnumerate,
    /// Indexed-property read (e.g. `nodeList[0]`).
    IndexedGet,
    /// Indexed-property write (e.g. `nodeList[0] = x`).
    IndexedSet,
    /// Indexed-property query.
    IndexedQuery,
    /// Indexed-collection length query.
    IndexedLength,
}

/// Key associated with an [`AccessCheckOperation`].
#[derive(Debug, Clone, Copy)]
pub enum AccessCheckKey<'a> {
    /// No specific key — used for [`AccessCheckOperation::NamedEnumerate`]
    /// and [`AccessCheckOperation::IndexedLength`].
    None,
    /// A named property key.
    Named(&'a str),
    /// An indexed property key.
    Indexed(u32),
}

/// Embedder callback that decides whether a DOM wrapper property
/// operation should be allowed.
///
/// The third argument is the embedder data pointer captured when the
/// access check was installed.  The callback **must** return `true` to
/// allow the operation, and `false` to deny it (fail closed).
pub type AccessCheckCallback =
    Box<dyn Fn(AccessCheckOperation, AccessCheckKey<'_>, *mut c_void) -> bool>;

// ─────────────────────────────────────────────────────────────────────────────
// DOM wrapper class-id registry
// ─────────────────────────────────────────────────────────────────────────────

/// Reserved DOM wrapper class id meaning "unassigned".
pub const DOM_CLASS_ID_UNASSIGNED: u32 = 0;

/// Stable metadata registered for an embedder DOM wrapper class id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomClassInfo {
    class_id: u32,
    parent_class_id: Option<u32>,
    name: String,
    flags: u32,
}

impl DomClassInfo {
    /// Create metadata for a DOM wrapper class id.
    pub fn new(
        class_id: u32,
        parent_class_id: Option<u32>,
        name: impl Into<String>,
        flags: u32,
    ) -> Self {
        Self {
            class_id,
            parent_class_id,
            name: name.into(),
            flags,
        }
    }

    /// Return this class id.
    pub fn class_id(&self) -> u32 {
        self.class_id
    }

    /// Return the optional parent/base class id.
    pub fn parent_class_id(&self) -> Option<u32> {
        self.parent_class_id
    }

    /// Return the registered debug label.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return embedder-defined metadata flags.
    pub fn flags(&self) -> u32 {
        self.flags
    }
}

/// Error returned by [`DomClassIdRegistry`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DomClassRegistryError {
    /// Class id 0 is reserved for unassigned wrappers.
    #[error("DOM class id 0 is reserved")]
    InvalidClassId,
    /// The requested class id is already registered.
    #[error("DOM class id is already registered")]
    DuplicateClassId,
    /// The requested parent class id is invalid or unregistered.
    #[error("DOM parent class id is invalid or unregistered")]
    InvalidParentClassId,
    /// The class cannot be unregistered while registered children refer to it.
    #[error("DOM class id still has registered children")]
    ClassHasChildren,
    /// The requested class id is not registered.
    #[error("DOM class id is not registered")]
    NotRegistered,
}

/// Isolate-level registry for DOM wrapper class ids and inheritance metadata.
#[derive(Debug, Default, Clone)]
pub struct DomClassIdRegistry {
    classes: HashMap<u32, DomClassInfo>,
}

impl DomClassIdRegistry {
    /// Create an empty DOM class-id registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a class id with optional parent metadata.
    pub fn register(&mut self, info: DomClassInfo) -> Result<(), DomClassRegistryError> {
        if info.class_id == DOM_CLASS_ID_UNASSIGNED {
            return Err(DomClassRegistryError::InvalidClassId);
        }
        if self.classes.contains_key(&info.class_id) {
            return Err(DomClassRegistryError::DuplicateClassId);
        }
        if let Some(parent_id) = info.parent_class_id
            && (parent_id == DOM_CLASS_ID_UNASSIGNED
                || parent_id == info.class_id
                || !self.classes.contains_key(&parent_id))
        {
            return Err(DomClassRegistryError::InvalidParentClassId);
        }
        self.classes.insert(info.class_id, info);
        Ok(())
    }

    /// Unregister a class id. Fails if registered children still refer to it.
    pub fn unregister(&mut self, class_id: u32) -> Result<(), DomClassRegistryError> {
        if class_id == DOM_CLASS_ID_UNASSIGNED {
            return Err(DomClassRegistryError::InvalidClassId);
        }
        if !self.classes.contains_key(&class_id) {
            return Err(DomClassRegistryError::NotRegistered);
        }
        if self
            .classes
            .values()
            .any(|info| info.parent_class_id == Some(class_id))
        {
            return Err(DomClassRegistryError::ClassHasChildren);
        }
        self.classes.remove(&class_id);
        Ok(())
    }

    /// Return metadata for a registered class id.
    pub fn get(&self, class_id: u32) -> Option<&DomClassInfo> {
        if class_id == DOM_CLASS_ID_UNASSIGNED {
            None
        } else {
            self.classes.get(&class_id)
        }
    }

    /// Return `true` if `actual_class_id` exactly matches `expected_class_id`.
    pub fn is_exact_match(&self, actual_class_id: u32, expected_class_id: u32) -> bool {
        actual_class_id == expected_class_id
            && self.get(actual_class_id).is_some()
            && self.get(expected_class_id).is_some()
    }

    /// Return `true` if `actual_class_id` is `expected_class_id` or derives from it.
    pub fn is_derived_match(&self, actual_class_id: u32, expected_class_id: u32) -> bool {
        if self.get(expected_class_id).is_none() {
            return false;
        }
        let mut current = Some(actual_class_id);
        let mut depth = 0usize;
        while let Some(class_id) = current {
            if class_id == expected_class_id {
                return self.get(class_id).is_some();
            }
            let Some(info) = self.get(class_id) else {
                return false;
            };
            current = info.parent_class_id;
            depth += 1;
            if depth > self.classes.len() {
                return false;
            }
        }
        false
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// DomObjectWrap
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of internal fields a single [`DomObjectWrap`] may hold.
pub const MAX_INTERNAL_FIELDS: usize = 16;

/// A JavaScript object that wraps a native (C++) DOM node.
///
/// `DomObjectWrap` extends the basic [`JsValue::PlainObject`] concept with:
///
/// - **Internal fields** — a fixed-size vector of opaque `*mut c_void`
///   pointers that the embedder uses to point back at C++ DOM objects (e.g.
///   `blink::Element*`).
/// - **Named property interceptors** — optional
///   [`NamedPropertyHandlerConfig`] so that reads like `element.id` are
///   forwarded to the embedder instead of looking up a JS property.
/// - **Indexed property interceptors** — optional
///   [`IndexedPropertyHandlerConfig`] so that reads like `nodeList[0]` are
///   forwarded.
///
/// # Safety contract
///
/// The raw pointers stored in internal fields are **owned by the embedder**.
/// The engine never dereferences them; it only stores and returns them.  The
/// embedder must ensure the pointers remain valid for the lifetime of the
/// wrapper (or register a weak callback via [`DomWeakRef`] to clean up).
///
/// # Example
///
/// ```
/// use stator_jse::dom::DomObjectWrap;
/// use stator_jse::objects::value::JsValue;
///
/// let mut wrap = DomObjectWrap::new(2); // 2 internal fields
/// assert_eq!(wrap.internal_field_count(), 2);
///
/// wrap.set_property("tagName", JsValue::String("DIV".into()));
/// assert_eq!(wrap.get_property("tagName"), JsValue::String("DIV".into()));
/// ```
pub struct DomObjectWrap {
    /// Own JS properties (for the rare case where the embedder stores values
    /// directly on the wrapper rather than going through interceptors).
    properties: Rc<RefCell<PropertyMap>>,
    /// Opaque embedder pointers (e.g. `blink::Element*`).
    internal_fields: Vec<*mut c_void>,
    /// Optional named-property interceptor configuration.
    named_handler: Option<NamedPropertyHandlerConfig>,
    /// Optional indexed-property interceptor configuration.
    indexed_handler: Option<IndexedPropertyHandlerConfig>,
    /// Optional access-check (security) callback.  When installed, every
    /// property operation on this wrapper consults the callback *before*
    /// the interceptor or own-property paths; a `false` return short-
    /// circuits the operation in a fail-closed manner (reads observe
    /// `undefined`, writes/deletes/queries report failure, enumeration
    /// yields the empty list).
    access_check: Option<AccessCheckCallback>,
}

// SAFETY: `DomObjectWrap` holds raw `*mut c_void` pointers in
// `internal_fields` that are only ever accessed on the owning thread.  The
// embedder is responsible for external synchronisation if the wrapper is
// transferred across threads.
unsafe impl Send for DomObjectWrap {}

impl DomObjectWrap {
    /// Create a new wrapper with `field_count` internal-field slots, all
    /// initialised to null.
    ///
    /// # Panics
    ///
    /// Panics if `field_count` exceeds [`MAX_INTERNAL_FIELDS`].
    pub fn new(field_count: usize) -> Self {
        assert!(
            field_count <= MAX_INTERNAL_FIELDS,
            "field_count ({field_count}) exceeds MAX_INTERNAL_FIELDS ({MAX_INTERNAL_FIELDS})"
        );
        Self {
            properties: Rc::new(RefCell::new(PropertyMap::new())),
            internal_fields: vec![std::ptr::null_mut(); field_count],
            named_handler: None,
            indexed_handler: None,
            access_check: None,
        }
    }

    // ── internal fields ──────────────────────────────────────────────────

    /// Return the number of internal-field slots.
    pub fn internal_field_count(&self) -> usize {
        self.internal_fields.len()
    }

    /// Store an opaque pointer in internal field `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= internal_field_count()`.
    pub fn set_internal_field(&mut self, index: usize, ptr: *mut c_void) {
        assert!(
            index < self.internal_fields.len(),
            "internal field index {index} out of range (count = {})",
            self.internal_fields.len()
        );
        self.internal_fields[index] = ptr;
    }

    /// Retrieve the opaque pointer from internal field `index`.
    ///
    /// Returns a null pointer if `index` is out of range.
    pub fn get_internal_field(&self, index: usize) -> *mut c_void {
        self.internal_fields
            .get(index)
            .copied()
            .unwrap_or(std::ptr::null_mut())
    }

    // ── property access (with interceptor support) ───────────────────────

    /// Return the embedder data pointer (internal field 0, or null).
    fn data_ptr(&self) -> *mut c_void {
        self.internal_fields
            .first()
            .copied()
            .unwrap_or(std::ptr::null_mut())
    }

    /// Run the installed access-check callback for `op`/`key`.  Returns
    /// `true` when access is allowed (the default when no callback is
    /// installed) and `false` when the embedder denies the operation.
    pub fn check_access(&self, op: AccessCheckOperation, key: AccessCheckKey<'_>) -> bool {
        match &self.access_check {
            Some(cb) => cb(op, key, self.data_ptr()),
            None => true,
        }
    }

    /// Read a named property, consulting the access-check callback first,
    /// then the interceptor, then own properties.
    ///
    /// Resolution order (modified by [`NamedPropertyHandlerFlags`] on the
    /// installed handler, if any):
    /// 1. Access-check callback (if installed).  A denial returns
    ///    `JsValue::Undefined` without consulting either the interceptor
    ///    or own properties — unless the handler sets
    ///    [`NamedPropertyHandlerFlags::ALL_CAN_READ`], in which case the
    ///    interceptor is still consulted (own properties remain hidden).
    /// 2. Named-property interceptor getter (if installed).  When the
    ///    handler sets [`NamedPropertyHandlerFlags::NON_MASKING`] the
    ///    interceptor is skipped if an own property with the same key
    ///    already exists.
    /// 3. Own properties on this wrapper.
    /// 4. `JsValue::Undefined`.
    pub fn get_property(&self, key: &str) -> JsValue {
        let allowed = self.check_access(AccessCheckOperation::NamedGet, AccessCheckKey::Named(key));
        let flags = self
            .named_handler
            .as_ref()
            .map(|c| c.flags())
            .unwrap_or(NamedPropertyHandlerFlags::NONE);
        let all_can_read = flags.contains(NamedPropertyHandlerFlags::ALL_CAN_READ);
        let non_masking = flags.contains(NamedPropertyHandlerFlags::NON_MASKING);

        // 1. Interceptor (subject to access check and non-masking).
        if let Some(cfg) = &self.named_handler
            && let Some(getter) = cfg.getter()
            && (allowed || all_can_read)
            && (!non_masking || !self.properties.borrow().contains_key(key))
            && let Some(val) = getter(key, self.data_ptr())
        {
            return val;
        }
        if !allowed {
            // Access denied and interceptor either absent or did not
            // handle the key; own properties remain hidden.
            return JsValue::Undefined;
        }
        // 2. Own properties
        self.properties
            .borrow()
            .get(key)
            .cloned()
            .unwrap_or(JsValue::Undefined)
    }

    /// Write a named property, consulting the access-check callback first,
    /// then the interceptor.
    ///
    /// If the access check denies the write, the value is *silently
    /// discarded* (fail-closed); the wrapper's own-property map is not
    /// mutated.  Otherwise, if the interceptor handles the write it short-
    /// circuits the own-property store as before.  When the handler sets
    /// [`NamedPropertyHandlerFlags::NON_MASKING`] the interceptor is
    /// skipped when an own property already exists for `key`, so the
    /// existing own property is updated directly.
    pub fn set_property(&mut self, key: &str, value: JsValue) {
        if !self.check_access(AccessCheckOperation::NamedSet, AccessCheckKey::Named(key)) {
            return;
        }
        let non_masking = self
            .named_handler
            .as_ref()
            .map(|c| c.flags().contains(NamedPropertyHandlerFlags::NON_MASKING))
            .unwrap_or(false);
        let already_own = non_masking && self.properties.borrow().contains_key(key);
        if !already_own
            && let Some(cfg) = &self.named_handler
            && let Some(setter) = cfg.setter()
            && setter(key, &value, self.data_ptr())
        {
            return;
        }
        self.properties.borrow_mut().insert(key.to_string(), value);
    }

    /// Attempt to write a named property through the interceptor only.
    ///
    /// Returns `true` when an installed setter handled the write or when
    /// the access check denied the write (the write is silently dropped
    /// and the engine treats the operation as handled so it does not fall
    /// through to a generic store).  Returns `false` when no setter is
    /// installed and access was allowed.  When the handler sets
    /// [`NamedPropertyHandlerFlags::NON_MASKING`] and the wrapper already
    /// owns `key`, the interceptor is skipped and `false` is returned so
    /// the caller updates the own-property store.
    pub fn set_intercepted_property(&self, key: &str, value: JsValue) -> bool {
        if !self.check_access(AccessCheckOperation::NamedSet, AccessCheckKey::Named(key)) {
            // Access denied: treat as handled so the caller does not fall
            // through to an own-property store.
            return true;
        }
        if let Some(cfg) = &self.named_handler
            && let Some(setter) = cfg.setter()
        {
            if cfg.flags().contains(NamedPropertyHandlerFlags::NON_MASKING)
                && self.properties.borrow().contains_key(key)
            {
                return false;
            }
            return setter(key, &value, self.data_ptr());
        }
        false
    }

    /// Query whether a named property exists, consulting the access-check
    /// callback first.  A denial reports the property as absent unless the
    /// handler sets [`NamedPropertyHandlerFlags::ALL_CAN_READ`], in which
    /// case the interceptor query is still consulted (own properties stay
    /// hidden).  When the handler sets
    /// [`NamedPropertyHandlerFlags::NON_MASKING`] the interceptor query is
    /// skipped if an own property already covers `key`.
    pub fn has_property(&self, key: &str) -> bool {
        let allowed =
            self.check_access(AccessCheckOperation::NamedQuery, AccessCheckKey::Named(key));
        let flags = self
            .named_handler
            .as_ref()
            .map(|c| c.flags())
            .unwrap_or(NamedPropertyHandlerFlags::NONE);
        let all_can_read = flags.contains(NamedPropertyHandlerFlags::ALL_CAN_READ);
        let non_masking = flags.contains(NamedPropertyHandlerFlags::NON_MASKING);
        if let Some(cfg) = &self.named_handler
            && let Some(query) = cfg.query()
            && (allowed || all_can_read)
            && (!non_masking || !self.properties.borrow().contains_key(key))
            && query(key, self.data_ptr()).is_some()
        {
            return true;
        }
        if !allowed {
            return false;
        }
        self.properties.borrow().contains_key(key)
    }

    /// Delete a named property, consulting the access-check callback
    /// first.  A denial returns `false` and leaves the property in place.
    pub fn delete_property(&mut self, key: &str) -> bool {
        if !self.check_access(
            AccessCheckOperation::NamedDelete,
            AccessCheckKey::Named(key),
        ) {
            return false;
        }
        if let Some(cfg) = &self.named_handler
            && let Some(deleter) = cfg.deleter()
            && deleter(key, self.data_ptr())
        {
            return true;
        }
        self.properties.borrow_mut().remove(key).is_some()
    }

    /// Enumerate own property names, consulting the access-check callback
    /// first.  A denial returns an empty list — neither interceptor names
    /// nor own-property names are reported.
    pub fn property_names(&self) -> Vec<String> {
        if !self.check_access(AccessCheckOperation::NamedEnumerate, AccessCheckKey::None) {
            return Vec::new();
        }
        let mut names: Vec<String> = Vec::new();
        // Interceptor-reported names come first.
        if let Some(cfg) = &self.named_handler
            && let Some(enumerator) = cfg.enumerator()
        {
            names.extend(enumerator(self.data_ptr()));
        }
        // Then own-property names.
        let mut own: Vec<String> = self
            .properties
            .borrow()
            .keys()
            .map(|k| k.to_string())
            .collect();
        own.sort();
        names.extend(own);
        names
    }

    // ── indexed property access ──────────────────────────────────────────

    /// Read an indexed property via the interceptor.
    ///
    /// Returns `JsValue::Undefined` if the access-check callback denies
    /// the read (unless the handler sets
    /// [`IndexedPropertyHandlerFlags::ALL_CAN_READ`], in which case the
    /// interceptor is still consulted), if no indexed-property
    /// interceptor is installed, or if the interceptor does not handle
    /// this index.
    pub fn get_indexed(&self, index: u32) -> JsValue {
        let allowed = self.check_access(
            AccessCheckOperation::IndexedGet,
            AccessCheckKey::Indexed(index),
        );
        let all_can_read = self
            .indexed_handler
            .as_ref()
            .map(|c| {
                c.flags()
                    .contains(IndexedPropertyHandlerFlags::ALL_CAN_READ)
            })
            .unwrap_or(false);
        if !allowed && !all_can_read {
            return JsValue::Undefined;
        }
        if let Some(cfg) = &self.indexed_handler
            && let Some(getter) = cfg.getter()
            && let Some(val) = getter(index, self.data_ptr())
        {
            return val;
        }
        JsValue::Undefined
    }

    /// Write an indexed property via the interceptor.
    ///
    /// Returns `true` when the interceptor handled the write *or* when the
    /// access check denied it (the write is silently discarded and the
    /// engine treats the operation as handled).  Returns `false` when no
    /// setter is installed and access was allowed.
    pub fn set_indexed(&mut self, index: u32, value: &JsValue) -> bool {
        if !self.check_access(
            AccessCheckOperation::IndexedSet,
            AccessCheckKey::Indexed(index),
        ) {
            return true;
        }
        if let Some(cfg) = &self.indexed_handler
            && let Some(setter) = cfg.setter()
        {
            return setter(index, value, self.data_ptr());
        }
        false
    }

    /// Query the length of the indexed collection via the interceptor.
    ///
    /// Returns `0` when the access-check callback denies the query (unless
    /// the handler sets [`IndexedPropertyHandlerFlags::ALL_CAN_READ`], in
    /// which case the length callback is still consulted) or when no
    /// length callback is installed.
    pub fn indexed_length(&self) -> u32 {
        let allowed = self.check_access(AccessCheckOperation::IndexedLength, AccessCheckKey::None);
        let all_can_read = self
            .indexed_handler
            .as_ref()
            .map(|c| {
                c.flags()
                    .contains(IndexedPropertyHandlerFlags::ALL_CAN_READ)
            })
            .unwrap_or(false);
        if !allowed && !all_can_read {
            return 0;
        }
        if let Some(cfg) = &self.indexed_handler
            && let Some(length_cb) = cfg.length()
        {
            return length_cb(self.data_ptr());
        }
        0
    }

    // ── interceptor installation ─────────────────────────────────────────

    /// Install a named-property handler configuration.
    pub fn set_named_handler(&mut self, config: NamedPropertyHandlerConfig) {
        self.named_handler = Some(config);
    }

    /// Return `true` if a named-property handler is installed.
    pub fn has_named_handler(&self) -> bool {
        self.named_handler.is_some()
    }

    /// Install an indexed-property handler configuration.
    pub fn set_indexed_handler(&mut self, config: IndexedPropertyHandlerConfig) {
        self.indexed_handler = Some(config);
    }

    /// Return `true` if an indexed-property handler is installed.
    pub fn has_indexed_handler(&self) -> bool {
        self.indexed_handler.is_some()
    }

    /// Return the [`NamedPropertyHandlerFlags`] currently configured on
    /// the installed named handler, or [`NamedPropertyHandlerFlags::NONE`]
    /// when no handler is installed.
    pub fn named_handler_flags(&self) -> NamedPropertyHandlerFlags {
        self.named_handler
            .as_ref()
            .map(|c| c.flags())
            .unwrap_or(NamedPropertyHandlerFlags::NONE)
    }

    /// Replace the [`NamedPropertyHandlerFlags`] on the installed named
    /// handler.  Returns `true` when a handler was present (and its flags
    /// were updated), `false` when no handler is installed.
    pub fn set_named_handler_flags(&mut self, flags: NamedPropertyHandlerFlags) -> bool {
        match &mut self.named_handler {
            Some(cfg) => {
                cfg.flags = flags;
                true
            }
            None => false,
        }
    }

    /// Return the [`IndexedPropertyHandlerFlags`] currently configured on
    /// the installed indexed handler, or
    /// [`IndexedPropertyHandlerFlags::NONE`] when no handler is installed.
    pub fn indexed_handler_flags(&self) -> IndexedPropertyHandlerFlags {
        self.indexed_handler
            .as_ref()
            .map(|c| c.flags())
            .unwrap_or(IndexedPropertyHandlerFlags::NONE)
    }

    /// Replace the [`IndexedPropertyHandlerFlags`] on the installed
    /// indexed handler.  Returns `true` when a handler was present (and
    /// its flags were updated), `false` when no handler is installed.
    pub fn set_indexed_handler_flags(&mut self, flags: IndexedPropertyHandlerFlags) -> bool {
        match &mut self.indexed_handler {
            Some(cfg) => {
                cfg.flags = flags;
                true
            }
            None => false,
        }
    }

    // ── access-check installation ────────────────────────────────────────

    /// Install a fail-closed access-check callback.
    ///
    /// The callback is invoked before every property operation on this
    /// wrapper (named and indexed get/set/query/delete, enumeration, and
    /// indexed length).  Returning `true` allows the operation to
    /// proceed; returning `false` denies it.  Replaces any previously
    /// installed access-check callback.
    pub fn set_access_check(&mut self, callback: AccessCheckCallback) {
        self.access_check = Some(callback);
    }

    /// Remove any installed access-check callback.  Returns `true` if a
    /// callback was previously installed.
    pub fn clear_access_check(&mut self) -> bool {
        self.access_check.take().is_some()
    }

    /// Return `true` if an access-check callback is installed.
    pub fn has_access_check(&self) -> bool {
        self.access_check.is_some()
    }

    // ── conversion ───────────────────────────────────────────────────────

    /// Return a [`JsValue::PlainObject`] view of the wrapper's own properties.
    ///
    /// This is useful when handing the wrapper to parts of the engine that
    /// only understand plain objects.  Note that the returned value shares
    /// the property map by reference; mutations through the `JsValue` are
    /// visible through the wrapper and vice versa.
    pub fn as_js_value(&self) -> JsValue {
        JsValue::PlainObject(Rc::clone(&self.properties))
    }
}

impl std::fmt::Debug for DomObjectWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomObjectWrap")
            .field("internal_field_count", &self.internal_fields.len())
            .field("property_count", &self.properties.borrow().len())
            .field("has_named_handler", &self.named_handler.is_some())
            .field("has_indexed_handler", &self.indexed_handler.is_some())
            .field("has_access_check", &self.access_check.is_some())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DomWeakRef
// ─────────────────────────────────────────────────────────────────────────────

/// Weak-callback type invoked when the wrapped object is collected.
///
/// The parameter is the embedder data pointer previously stored in internal
/// field 0 of the associated [`DomObjectWrap`].
pub type DomWeakCallback = Box<dyn FnOnce(*mut c_void)>;

/// A weak reference to a [`DomObjectWrap`] that invokes a callback when the
/// wrapped object becomes unreachable.
///
/// This mirrors V8's `v8::Persistent<T>::SetWeak()` mechanism, allowing the
/// C++ embedder to release native resources (e.g. calling
/// `Release()` on a ref-counted DOM node) when the JavaScript wrapper is
/// garbage-collected.
///
/// # Example
///
/// ```
/// use std::cell::Cell;
/// use std::rc::Rc;
/// use stator_jse::dom::{DomObjectWrap, DomWeakRef};
///
/// let wrap = DomObjectWrap::new(1);
/// let invoked = Rc::new(Cell::new(false));
/// let flag = Rc::clone(&invoked);
///
/// let weak = DomWeakRef::new(&wrap, move |_ptr| {
///     flag.set(true);
/// });
///
/// assert!(weak.is_alive());
/// weak.invoke_callback();
/// assert!(!weak.is_alive());
/// assert!(invoked.get());
/// ```
pub struct DomWeakRef {
    /// Embedder data pointer (copied from internal field 0 at creation time).
    data: *mut c_void,
    /// The weak callback, wrapped in `Option` so we can `take()` it exactly
    /// once.
    callback: RefCell<Option<DomWeakCallback>>,
    /// Liveness flag; set to `false` after the callback fires.
    alive: RefCell<bool>,
}

// SAFETY: `DomWeakRef` holds a raw `*mut c_void` that is only accessed on the
// owning thread.  The embedder is responsible for external synchronisation.
unsafe impl Send for DomWeakRef {}

impl DomWeakRef {
    /// Create a new weak reference for `wrap`.
    ///
    /// `callback` will be invoked *at most once* when [`invoke_callback`] is
    /// called (typically by the GC).  The pointer passed to the callback is
    /// the value of internal field 0 at creation time (or null if the wrapper
    /// has no internal fields).
    ///
    /// [`invoke_callback`]: DomWeakRef::invoke_callback
    pub fn new(wrap: &DomObjectWrap, callback: impl FnOnce(*mut c_void) + 'static) -> Self {
        let data = wrap.get_internal_field(0);
        Self {
            data,
            callback: RefCell::new(Some(Box::new(callback))),
            alive: RefCell::new(true),
        }
    }

    /// Return `true` if the weak reference has not yet been invalidated.
    pub fn is_alive(&self) -> bool {
        *self.alive.borrow()
    }

    /// Return the embedder data pointer captured at creation time.
    pub fn data(&self) -> *mut c_void {
        self.data
    }

    /// Fire the weak callback (if it has not already been fired) and mark
    /// the reference as dead.
    ///
    /// This is idempotent: calling it a second time is a no-op.
    pub fn invoke_callback(&self) {
        if let Some(cb) = self.callback.borrow_mut().take() {
            cb(self.data);
        }
        *self.alive.borrow_mut() = false;
    }

    /// Reset the weak reference without invoking the callback.
    ///
    /// After this call, [`is_alive`] returns `false` and any future
    /// [`invoke_callback`] call is a no-op.
    ///
    /// [`is_alive`]: DomWeakRef::is_alive
    /// [`invoke_callback`]: DomWeakRef::invoke_callback
    pub fn clear(&self) {
        self.callback.borrow_mut().take();
        *self.alive.borrow_mut() = false;
    }
}

impl std::fmt::Debug for DomWeakRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomWeakRef")
            .field("alive", &*self.alive.borrow())
            .field("data", &self.data)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // ── DomObjectWrap — internal fields ──────────────────────────────────

    #[test]
    fn test_wrap_new_zero_fields() {
        let wrap = DomObjectWrap::new(0);
        assert_eq!(wrap.internal_field_count(), 0);
    }

    #[test]
    fn test_wrap_new_with_fields() {
        let wrap = DomObjectWrap::new(3);
        assert_eq!(wrap.internal_field_count(), 3);
        for i in 0..3 {
            assert!(wrap.get_internal_field(i).is_null());
        }
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_INTERNAL_FIELDS")]
    fn test_wrap_new_too_many_fields() {
        let _wrap = DomObjectWrap::new(MAX_INTERNAL_FIELDS + 1);
    }

    #[test]
    fn test_wrap_set_get_internal_field() {
        let mut wrap = DomObjectWrap::new(2);
        let sentinel: usize = 0xDEAD_BEEF;
        wrap.set_internal_field(0, sentinel as *mut c_void);
        assert_eq!(wrap.get_internal_field(0) as usize, sentinel);
        assert!(wrap.get_internal_field(1).is_null());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_wrap_set_internal_field_out_of_range() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_internal_field(1, std::ptr::null_mut());
    }

    #[test]
    fn test_wrap_get_internal_field_out_of_range_returns_null() {
        let wrap = DomObjectWrap::new(1);
        assert!(wrap.get_internal_field(99).is_null());
    }

    // ── DomObjectWrap — plain property access ────────────────────────────

    #[test]
    fn test_wrap_property_crud() {
        let mut wrap = DomObjectWrap::new(0);
        assert_eq!(wrap.get_property("x"), JsValue::Undefined);
        assert!(!wrap.has_property("x"));

        wrap.set_property("x", JsValue::Smi(42));
        assert_eq!(wrap.get_property("x"), JsValue::Smi(42));
        assert!(wrap.has_property("x"));

        assert!(wrap.delete_property("x"));
        assert!(!wrap.has_property("x"));
    }

    #[test]
    fn test_wrap_property_names_sorted() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("z", JsValue::Smi(1));
        wrap.set_property("a", JsValue::Smi(2));
        wrap.set_property("m", JsValue::Smi(3));
        // Own properties are sorted.
        assert_eq!(wrap.property_names(), vec!["a", "m", "z"]);
    }

    // ── DomObjectWrap — named interceptors ───────────────────────────────

    #[test]
    fn test_wrap_named_getter_interceptor() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|name, _data| {
                    if name == "id" {
                        Some(JsValue::String("my-div".into()))
                    } else {
                        None
                    }
                })
                .build(),
        );

        // Interceptor handles "id".
        assert_eq!(wrap.get_property("id"), JsValue::String("my-div".into()));
        // Interceptor falls through for "class"; returns Undefined.
        assert_eq!(wrap.get_property("class"), JsValue::Undefined);
    }

    #[test]
    fn test_wrap_named_setter_interceptor() {
        let intercepted = Rc::new(Cell::new(false));
        let flag = Rc::clone(&intercepted);

        let mut wrap = DomObjectWrap::new(1);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .setter(move |name, _val, _data| {
                    if name == "id" {
                        flag.set(true);
                        true // handled
                    } else {
                        false
                    }
                })
                .build(),
        );

        wrap.set_property("id", JsValue::String("new-id".into()));
        assert!(intercepted.get());
        // "id" was handled by interceptor, so it should NOT be in own properties.
        assert!(!wrap.properties.borrow().contains_key("id"));

        wrap.set_property("class", JsValue::String("foo".into()));
        // "class" was not intercepted, so it IS in own properties.
        assert!(wrap.properties.borrow().contains_key("class"));
    }

    #[test]
    fn test_wrap_named_query_interceptor() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .query(|name, _data| {
                    if name == "id" {
                        Some(0) // property exists with no special attributes
                    } else {
                        None
                    }
                })
                .build(),
        );

        assert!(wrap.has_property("id"));
        assert!(!wrap.has_property("class"));
    }

    #[test]
    fn test_wrap_named_enumerator() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_property("ownProp", JsValue::Smi(1));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .enumerator(|_data| vec!["id".to_string(), "className".to_string()])
                .build(),
        );

        let names = wrap.property_names();
        // Interceptor names come first, then sorted own properties.
        assert_eq!(names, vec!["id", "className", "ownProp"]);
    }

    // ── DomObjectWrap — indexed interceptors ─────────────────────────────

    #[test]
    fn test_wrap_indexed_getter() {
        let items = vec![JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)];
        let items_clone = items.clone();

        let mut wrap = DomObjectWrap::new(1);
        wrap.set_indexed_handler(
            IndexedPropertyHandlerConfig::builder()
                .getter(move |idx, _data| items_clone.get(idx as usize).cloned())
                .length(|_data| 3)
                .build(),
        );

        assert_eq!(wrap.get_indexed(0), JsValue::Smi(10));
        assert_eq!(wrap.get_indexed(2), JsValue::Smi(30));
        assert_eq!(wrap.get_indexed(3), JsValue::Undefined);
        assert_eq!(wrap.indexed_length(), 3);
    }

    #[test]
    fn test_wrap_indexed_setter() {
        let stored = Rc::new(RefCell::new(Vec::<(u32, JsValue)>::new()));
        let stored_clone = Rc::clone(&stored);

        let mut wrap = DomObjectWrap::new(1);
        wrap.set_indexed_handler(
            IndexedPropertyHandlerConfig::builder()
                .setter(move |idx, val, _data| {
                    stored_clone.borrow_mut().push((idx, val.clone()));
                    true
                })
                .build(),
        );

        assert!(wrap.set_indexed(0, &JsValue::Smi(99)));
        assert_eq!(stored.borrow().len(), 1);
        assert_eq!(stored.borrow()[0], (0, JsValue::Smi(99)));
    }

    #[test]
    fn test_wrap_no_indexed_handler() {
        let wrap = DomObjectWrap::new(0);
        assert_eq!(wrap.get_indexed(0), JsValue::Undefined);
        assert_eq!(wrap.indexed_length(), 0);
    }

    // ── DomObjectWrap — access-check callback ────────────────────────────

    #[test]
    fn test_wrap_access_check_default_allows_all() {
        let wrap = DomObjectWrap::new(0);
        assert!(!wrap.has_access_check());
        assert!(wrap.check_access(AccessCheckOperation::NamedGet, AccessCheckKey::Named("x")));
        assert!(wrap.check_access(AccessCheckOperation::IndexedGet, AccessCheckKey::Indexed(0)));
        assert!(wrap.check_access(AccessCheckOperation::NamedEnumerate, AccessCheckKey::None));
    }

    #[test]
    fn test_wrap_access_check_allow_lets_interceptor_run() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|name, _data| {
                    if name == "id" {
                        Some(JsValue::String("ok".into()))
                    } else {
                        None
                    }
                })
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| true));
        assert!(wrap.has_access_check());
        assert_eq!(wrap.get_property("id"), JsValue::String("ok".into()));
    }

    #[test]
    fn test_wrap_access_check_deny_named_get_fails_closed() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(42));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|_name, _data| Some(JsValue::String("intercepted".into())))
                .build(),
        );
        wrap.set_access_check(Box::new(|op, _key, _data| {
            !matches!(op, AccessCheckOperation::NamedGet)
        }));
        // Denied: neither interceptor nor own-property is observed.
        assert_eq!(wrap.get_property("x"), JsValue::Undefined);
    }

    #[test]
    fn test_wrap_access_check_deny_named_set_discards_write() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_access_check(Box::new(|op, _key, _data| {
            !matches!(op, AccessCheckOperation::NamedSet)
        }));
        wrap.set_property("x", JsValue::Smi(99));
        // The write was silently discarded.
        assert_eq!(wrap.get_property("x"), JsValue::Undefined);
    }

    #[test]
    fn test_wrap_access_check_deny_query_and_delete() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(1));
        wrap.set_access_check(Box::new(|op, _key, _data| {
            !matches!(
                op,
                AccessCheckOperation::NamedQuery | AccessCheckOperation::NamedDelete
            )
        }));
        assert!(!wrap.has_property("x"));
        assert!(!wrap.delete_property("x"));
    }

    #[test]
    fn test_wrap_access_check_deny_enumerate_returns_empty() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("a", JsValue::Smi(1));
        wrap.set_property("b", JsValue::Smi(2));
        wrap.set_access_check(Box::new(|op, _key, _data| {
            !matches!(op, AccessCheckOperation::NamedEnumerate)
        }));
        assert!(wrap.property_names().is_empty());
    }

    #[test]
    fn test_wrap_access_check_deny_indexed_operations() {
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_indexed_handler(
            IndexedPropertyHandlerConfig::builder()
                .getter(|_idx, _data| Some(JsValue::Smi(7)))
                .setter(|_idx, _val, _data| true)
                .length(|_data| 3)
                .build(),
        );
        wrap.set_access_check(Box::new(|op, _key, _data| {
            !matches!(
                op,
                AccessCheckOperation::IndexedGet
                    | AccessCheckOperation::IndexedSet
                    | AccessCheckOperation::IndexedLength
            )
        }));
        assert_eq!(wrap.get_indexed(0), JsValue::Undefined);
        assert!(wrap.set_indexed(0, &JsValue::Smi(1))); // handled (silently dropped)
        assert_eq!(wrap.indexed_length(), 0);
    }

    #[test]
    fn test_wrap_access_check_receives_key_and_embedder_data() {
        let observed: Rc<RefCell<Vec<(AccessCheckOperation, String, usize)>>> =
            Rc::new(RefCell::new(Vec::new()));
        let log = Rc::clone(&observed);

        let sentinel: usize = 0xBADC_0FFE;
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_internal_field(0, sentinel as *mut c_void);
        wrap.set_access_check(Box::new(move |op, key, data| {
            let key_str = match key {
                AccessCheckKey::Named(name) => name.to_string(),
                AccessCheckKey::Indexed(i) => format!("[{i}]"),
                AccessCheckKey::None => String::new(),
            };
            log.borrow_mut().push((op, key_str, data as usize));
            true
        }));

        let _ = wrap.get_property("foo");
        let _ = wrap.get_indexed(7);
        let _ = wrap.property_names();

        let entries = observed.borrow();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, AccessCheckOperation::NamedGet);
        assert_eq!(entries[0].1, "foo");
        assert_eq!(entries[0].2, sentinel);
        assert_eq!(entries[1].0, AccessCheckOperation::IndexedGet);
        assert_eq!(entries[1].1, "[7]");
        assert_eq!(entries[1].2, sentinel);
        assert_eq!(entries[2].0, AccessCheckOperation::NamedEnumerate);
        assert_eq!(entries[2].1, "");
    }

    #[test]
    fn test_wrap_clear_access_check_restores_default_allow() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(5));
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        assert!(wrap.has_access_check());
        assert_eq!(wrap.get_property("x"), JsValue::Undefined);

        assert!(wrap.clear_access_check());
        assert!(!wrap.has_access_check());
        assert_eq!(wrap.get_property("x"), JsValue::Smi(5));
        // clear is idempotent
        assert!(!wrap.clear_access_check());
    }

    // ── DomObjectWrap — as_js_value ──────────────────────────────────────

    #[test]
    fn test_wrap_as_js_value_shares_properties() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(1));
        let jv = wrap.as_js_value();
        if let JsValue::PlainObject(map) = &jv {
            assert_eq!(map.borrow().get("x"), Some(&JsValue::Smi(1)));
            // Mutate through JsValue; visible through wrap.
            map.borrow_mut().insert("y".to_string(), JsValue::Smi(2));
        } else {
            panic!("expected PlainObject");
        }
        assert_eq!(wrap.get_property("y"), JsValue::Smi(2));
    }

    // ── DomObjectWrap — Debug ────────────────────────────────────────────

    #[test]
    fn test_wrap_debug() {
        let wrap = DomObjectWrap::new(2);
        let s = format!("{wrap:?}");
        assert!(s.contains("DomObjectWrap"));
        assert!(s.contains("internal_field_count: 2"));
    }

    // ── DomWeakRef ───────────────────────────────────────────────────────

    #[test]
    fn test_weak_ref_invoke() {
        let invoked = Rc::new(Cell::new(false));
        let flag = Rc::clone(&invoked);

        let wrap = DomObjectWrap::new(1);
        let weak = DomWeakRef::new(&wrap, move |_ptr| {
            flag.set(true);
        });

        assert!(weak.is_alive());
        weak.invoke_callback();
        assert!(!weak.is_alive());
        assert!(invoked.get());
    }

    #[test]
    fn test_weak_ref_invoke_idempotent() {
        let count = Rc::new(Cell::new(0u32));
        let ctr = Rc::clone(&count);

        let wrap = DomObjectWrap::new(1);
        let weak = DomWeakRef::new(&wrap, move |_ptr| {
            ctr.set(ctr.get() + 1);
        });

        weak.invoke_callback();
        weak.invoke_callback(); // no-op
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn test_weak_ref_clear() {
        let invoked = Rc::new(Cell::new(false));
        let flag = Rc::clone(&invoked);

        let wrap = DomObjectWrap::new(1);
        let weak = DomWeakRef::new(&wrap, move |_ptr| {
            flag.set(true);
        });

        weak.clear();
        assert!(!weak.is_alive());
        weak.invoke_callback(); // no-op after clear
        assert!(!invoked.get());
    }

    #[test]
    fn test_weak_ref_captures_data_pointer() {
        let sentinel: usize = 0xCAFE;
        let mut wrap = DomObjectWrap::new(1);
        wrap.set_internal_field(0, sentinel as *mut c_void);

        let captured = Rc::new(Cell::new(0usize));
        let cap = Rc::clone(&captured);

        let weak = DomWeakRef::new(&wrap, move |ptr| {
            cap.set(ptr as usize);
        });

        assert_eq!(weak.data() as usize, sentinel);
        weak.invoke_callback();
        assert_eq!(captured.get(), sentinel);
    }

    #[test]
    fn test_weak_ref_debug() {
        let wrap = DomObjectWrap::new(0);
        let weak = DomWeakRef::new(&wrap, |_| {});
        let s = format!("{weak:?}");
        assert!(s.contains("DomWeakRef"));
        assert!(s.contains("alive: true"));
    }

    // ── NamedPropertyHandlerConfig ───────────────────────────────────────

    #[test]
    fn test_named_handler_config_builder_empty() {
        let cfg = NamedPropertyHandlerConfig::builder().build();
        assert!(cfg.getter().is_none());
        assert!(cfg.setter().is_none());
        assert!(cfg.query().is_none());
        assert!(cfg.deleter().is_none());
        assert!(cfg.enumerator().is_none());
    }

    #[test]
    fn test_named_handler_config_debug() {
        let cfg = NamedPropertyHandlerConfig::builder()
            .getter(|_, _| None)
            .build();
        let s = format!("{cfg:?}");
        assert!(s.contains("has_getter: true"));
        assert!(s.contains("has_setter: false"));
    }

    // ── IndexedPropertyHandlerConfig ─────────────────────────────────────

    #[test]
    fn test_indexed_handler_config_builder_empty() {
        let cfg = IndexedPropertyHandlerConfig::builder().build();
        assert!(cfg.getter().is_none());
        assert!(cfg.setter().is_none());
        assert!(cfg.query().is_none());
        assert!(cfg.length().is_none());
    }

    #[test]
    fn test_indexed_handler_config_debug() {
        let cfg = IndexedPropertyHandlerConfig::builder()
            .getter(|_, _| None)
            .length(|_| 0)
            .build();
        let s = format!("{cfg:?}");
        assert!(s.contains("has_getter: true"));
        assert!(s.contains("has_length: true"));
    }

    // ── PropertyHandlerFlags — bit/validation helpers ────────────────────

    #[test]
    fn test_named_flags_default_is_none() {
        let f: NamedPropertyHandlerFlags = NamedPropertyHandlerFlags::default();
        assert_eq!(f, NamedPropertyHandlerFlags::NONE);
        assert_eq!(f.bits(), 0);
        assert!(f.validate().is_ok());
    }

    #[test]
    fn test_named_flags_from_bits_round_trip() {
        let raw = NamedPropertyHandlerFlags::ALL_CAN_READ.bits()
            | NamedPropertyHandlerFlags::NON_MASKING.bits();
        let parsed = NamedPropertyHandlerFlags::from_bits(raw).expect("valid bits");
        assert_eq!(parsed.bits(), raw);
        assert!(parsed.contains(NamedPropertyHandlerFlags::ALL_CAN_READ));
        assert!(parsed.contains(NamedPropertyHandlerFlags::NON_MASKING));
        assert!(!parsed.contains(NamedPropertyHandlerFlags::HAS_NO_SIDE_EFFECT));
    }

    #[test]
    fn test_named_flags_reject_unknown_bits() {
        let bad = 1u32 << 31;
        assert!(NamedPropertyHandlerFlags::from_bits(bad).is_none());
        let truncated = NamedPropertyHandlerFlags::from_bits_truncate(bad);
        assert_eq!(truncated.bits(), 0);
    }

    #[test]
    fn test_indexed_flags_reject_unknown_bits() {
        // Indexed flags do not include ONLY_INTERCEPT_STRINGS/INTERCEPT_SYMBOLS.
        let bad = 1u32 << 2;
        assert!(IndexedPropertyHandlerFlags::from_bits(bad).is_none());
    }

    #[test]
    fn test_named_flags_bitor_union() {
        let f = NamedPropertyHandlerFlags::ALL_CAN_READ | NamedPropertyHandlerFlags::NON_MASKING;
        assert!(f.contains(NamedPropertyHandlerFlags::ALL_CAN_READ));
        assert!(f.contains(NamedPropertyHandlerFlags::NON_MASKING));
    }

    // ── Default flags / builder integration ──────────────────────────────

    #[test]
    fn test_named_config_default_flags_none() {
        let cfg = NamedPropertyHandlerConfig::builder()
            .getter(|_, _| None)
            .build();
        assert_eq!(cfg.flags(), NamedPropertyHandlerFlags::NONE);
    }

    #[test]
    fn test_named_config_builder_sets_flags() {
        let cfg = NamedPropertyHandlerConfig::builder()
            .getter(|_, _| None)
            .flags(NamedPropertyHandlerFlags::ALL_CAN_READ)
            .build();
        assert!(
            cfg.flags()
                .contains(NamedPropertyHandlerFlags::ALL_CAN_READ)
        );
    }

    #[test]
    fn test_indexed_config_default_flags_none() {
        let cfg = IndexedPropertyHandlerConfig::builder().build();
        assert_eq!(cfg.flags(), IndexedPropertyHandlerFlags::NONE);
    }

    #[test]
    fn test_wrap_named_flags_accessors_when_no_handler() {
        let mut wrap = DomObjectWrap::new(0);
        assert_eq!(wrap.named_handler_flags(), NamedPropertyHandlerFlags::NONE);
        // No-op when no handler is installed.
        assert!(!wrap.set_named_handler_flags(NamedPropertyHandlerFlags::ALL_CAN_READ));
        assert_eq!(wrap.named_handler_flags(), NamedPropertyHandlerFlags::NONE);
    }

    #[test]
    fn test_wrap_indexed_flags_accessors_when_no_handler() {
        let mut wrap = DomObjectWrap::new(0);
        assert_eq!(
            wrap.indexed_handler_flags(),
            IndexedPropertyHandlerFlags::NONE
        );
        assert!(!wrap.set_indexed_handler_flags(IndexedPropertyHandlerFlags::ALL_CAN_READ));
    }

    #[test]
    fn test_wrap_named_flags_update_after_install() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|_, _| None)
                .build(),
        );
        assert_eq!(wrap.named_handler_flags(), NamedPropertyHandlerFlags::NONE);
        assert!(wrap.set_named_handler_flags(
            NamedPropertyHandlerFlags::ALL_CAN_READ | NamedPropertyHandlerFlags::NON_MASKING
        ));
        assert!(
            wrap.named_handler_flags()
                .contains(NamedPropertyHandlerFlags::ALL_CAN_READ)
        );
        assert!(
            wrap.named_handler_flags()
                .contains(NamedPropertyHandlerFlags::NON_MASKING)
        );
    }

    // ── DOM class-id registry ─────────────────────────────────────────────

    #[test]
    fn test_dom_class_registry_rejects_invalid_and_duplicate_ids() {
        let mut registry = DomClassIdRegistry::new();
        assert_eq!(
            registry.register(DomClassInfo::new(0, None, "Invalid", 0)),
            Err(DomClassRegistryError::InvalidClassId)
        );
        registry
            .register(DomClassInfo::new(1, None, "Node", 0))
            .expect("register node");
        assert_eq!(
            registry.register(DomClassInfo::new(1, None, "Other", 0)),
            Err(DomClassRegistryError::DuplicateClassId)
        );
        assert_eq!(
            registry.register(DomClassInfo::new(2, Some(99), "Element", 0)),
            Err(DomClassRegistryError::InvalidParentClassId)
        );
    }

    #[test]
    fn test_dom_class_registry_exact_and_derived_matching() {
        let mut registry = DomClassIdRegistry::new();
        registry
            .register(DomClassInfo::new(1, None, "Node", 0))
            .expect("register node");
        registry
            .register(DomClassInfo::new(2, Some(1), "Element", 0))
            .expect("register element");
        registry
            .register(DomClassInfo::new(3, Some(2), "HTMLDivElement", 0))
            .expect("register div");

        assert!(registry.is_exact_match(2, 2));
        assert!(!registry.is_exact_match(3, 2));
        assert!(registry.is_derived_match(3, 3));
        assert!(registry.is_derived_match(3, 2));
        assert!(registry.is_derived_match(3, 1));
        assert!(!registry.is_derived_match(1, 3));
        assert!(!registry.is_derived_match(99, 1));
        assert!(!registry.is_derived_match(3, 99));
    }

    #[test]
    fn test_dom_class_registry_unregister_rejects_registered_children() {
        let mut registry = DomClassIdRegistry::new();
        registry
            .register(DomClassInfo::new(1, None, "Node", 0))
            .expect("register node");
        registry
            .register(DomClassInfo::new(2, Some(1), "Element", 0))
            .expect("register element");
        assert_eq!(
            registry.unregister(1),
            Err(DomClassRegistryError::ClassHasChildren)
        );
        registry.unregister(2).expect("unregister child");
        registry.unregister(1).expect("unregister parent");
        assert!(!registry.is_exact_match(1, 1));
    }
    // ── ALL_CAN_READ semantics — named ───────────────────────────────────

    #[test]
    fn test_named_all_can_read_bypasses_access_check_for_get() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|name, _data| {
                    if name == "id" {
                        Some(JsValue::String("ok".into()))
                    } else {
                        None
                    }
                })
                .flags(NamedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        // Deny everything.
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        // ALL_CAN_READ still routes reads through the interceptor.
        assert_eq!(wrap.get_property("id"), JsValue::String("ok".into()));
        // Keys not handled by the interceptor still return Undefined.
        assert_eq!(wrap.get_property("class"), JsValue::Undefined);
    }

    #[test]
    fn test_named_all_can_read_does_not_expose_own_props() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(42));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(|_name, _data| None)
                .flags(NamedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        // Own property "x" stays hidden because access was denied.
        assert_eq!(wrap.get_property("x"), JsValue::Undefined);
    }

    #[test]
    fn test_named_all_can_read_bypass_for_query() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .query(|name, _data| if name == "id" { Some(0) } else { None })
                .flags(NamedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        assert!(wrap.has_property("id"));
        assert!(!wrap.has_property("missing"));
    }

    #[test]
    fn test_named_all_can_read_does_not_bypass_writes_or_deletes() {
        let intercepted = Rc::new(Cell::new(false));
        let flag = Rc::clone(&intercepted);
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .setter(move |_n, _v, _d| {
                    flag.set(true);
                    true
                })
                .flags(NamedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        wrap.set_property("x", JsValue::Smi(1));
        // Write must still be denied: setter never ran.
        assert!(!intercepted.get());
    }

    // ── NON_MASKING semantics — named ────────────────────────────────────

    #[test]
    fn test_named_non_masking_skips_interceptor_for_existing_own_prop() {
        let intercept_calls = Rc::new(Cell::new(0u32));
        let counter = Rc::clone(&intercept_calls);
        let mut wrap = DomObjectWrap::new(0);
        // Own property "x" already set before the interceptor is installed.
        wrap.set_property("x", JsValue::Smi(42));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .getter(move |_name, _data| {
                    counter.set(counter.get() + 1);
                    Some(JsValue::String("intercepted".into()))
                })
                .flags(NamedPropertyHandlerFlags::NON_MASKING)
                .build(),
        );

        // Existing own property — interceptor must NOT run, own value wins.
        assert_eq!(wrap.get_property("x"), JsValue::Smi(42));
        assert_eq!(intercept_calls.get(), 0);

        // Missing own property — interceptor runs.
        assert_eq!(
            wrap.get_property("y"),
            JsValue::String("intercepted".into())
        );
        assert_eq!(intercept_calls.get(), 1);
    }

    #[test]
    fn test_named_non_masking_setter_falls_back_to_own_for_existing_key() {
        let intercept_calls = Rc::new(Cell::new(0u32));
        let counter = Rc::clone(&intercept_calls);
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(1));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .setter(move |_n, _v, _d| {
                    counter.set(counter.get() + 1);
                    true
                })
                .flags(NamedPropertyHandlerFlags::NON_MASKING)
                .build(),
        );
        wrap.set_property("x", JsValue::Smi(99));
        // Interceptor skipped; own property updated in place.
        assert_eq!(intercept_calls.get(), 0);
        assert_eq!(wrap.get_property("x"), JsValue::Smi(99));
    }

    #[test]
    fn test_named_non_masking_set_intercepted_returns_false_for_existing_own() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_property("x", JsValue::Smi(1));
        wrap.set_named_handler(
            NamedPropertyHandlerConfig::builder()
                .setter(|_n, _v, _d| true)
                .flags(NamedPropertyHandlerFlags::NON_MASKING)
                .build(),
        );
        // For an existing own prop, the interceptor write is bypassed.
        assert!(!wrap.set_intercepted_property("x", JsValue::Smi(2)));
        // For a fresh key, the interceptor handles the write.
        assert!(wrap.set_intercepted_property("y", JsValue::Smi(3)));
    }

    // ── ALL_CAN_READ semantics — indexed ─────────────────────────────────

    #[test]
    fn test_indexed_all_can_read_bypasses_access_check_for_get_and_length() {
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_indexed_handler(
            IndexedPropertyHandlerConfig::builder()
                .getter(|idx, _d| Some(JsValue::Smi(idx as i32)))
                .length(|_d| 3)
                .flags(IndexedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        assert_eq!(wrap.get_indexed(0), JsValue::Smi(0));
        assert_eq!(wrap.get_indexed(2), JsValue::Smi(2));
        assert_eq!(wrap.indexed_length(), 3);
    }

    #[test]
    fn test_indexed_all_can_read_does_not_bypass_set() {
        let intercepted = Rc::new(Cell::new(false));
        let flag = Rc::clone(&intercepted);
        let mut wrap = DomObjectWrap::new(0);
        wrap.set_indexed_handler(
            IndexedPropertyHandlerConfig::builder()
                .setter(move |_idx, _v, _d| {
                    flag.set(true);
                    true
                })
                .flags(IndexedPropertyHandlerFlags::ALL_CAN_READ)
                .build(),
        );
        wrap.set_access_check(Box::new(|_op, _key, _data| false));
        let handled = wrap.set_indexed(0, &JsValue::Smi(7));
        // Access check denial still treats the write as handled and
        // silently drops it; the setter must NOT have run.
        assert!(handled);
        assert!(!intercepted.get());
    }

    // ── Stored-only flags ────────────────────────────────────────────────

    #[test]
    fn test_stored_only_flags_preserved_in_config() {
        let cfg = NamedPropertyHandlerConfig::builder()
            .getter(|_, _| None)
            .flags(
                NamedPropertyHandlerFlags::ONLY_INTERCEPT_STRINGS
                    | NamedPropertyHandlerFlags::HAS_NO_SIDE_EFFECT,
            )
            .build();
        assert!(
            cfg.flags()
                .contains(NamedPropertyHandlerFlags::ONLY_INTERCEPT_STRINGS)
        );
        assert!(
            cfg.flags()
                .contains(NamedPropertyHandlerFlags::HAS_NO_SIDE_EFFECT)
        );
    }
}

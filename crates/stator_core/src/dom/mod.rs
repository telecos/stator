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
/// use stator_core::dom::NamedPropertyHandlerConfig;
/// use stator_core::objects::value::JsValue;
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
}

impl std::fmt::Debug for NamedPropertyHandlerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedPropertyHandlerConfig")
            .field("has_getter", &self.getter.is_some())
            .field("has_setter", &self.setter.is_some())
            .field("has_query", &self.query.is_some())
            .field("has_deleter", &self.deleter.is_some())
            .field("has_enumerator", &self.enumerator.is_some())
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

    /// Consume the builder and return the finished configuration.
    pub fn build(self) -> NamedPropertyHandlerConfig {
        NamedPropertyHandlerConfig {
            getter: self.getter,
            setter: self.setter,
            query: self.query,
            deleter: self.deleter,
            enumerator: self.enumerator,
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
/// use stator_core::dom::IndexedPropertyHandlerConfig;
/// use stator_core::objects::value::JsValue;
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
}

impl std::fmt::Debug for IndexedPropertyHandlerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedPropertyHandlerConfig")
            .field("has_getter", &self.getter.is_some())
            .field("has_setter", &self.setter.is_some())
            .field("has_query", &self.query.is_some())
            .field("has_length", &self.length.is_some())
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

    /// Consume the builder and return the finished configuration.
    pub fn build(self) -> IndexedPropertyHandlerConfig {
        IndexedPropertyHandlerConfig {
            getter: self.getter,
            setter: self.setter,
            query: self.query,
            length: self.length,
        }
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
/// use stator_core::dom::DomObjectWrap;
/// use stator_core::objects::value::JsValue;
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

    /// Read a named property, consulting the interceptor first.
    ///
    /// Resolution order:
    /// 1. Named-property interceptor getter (if installed).
    /// 2. Own properties on this wrapper.
    /// 3. `JsValue::Undefined`.
    pub fn get_property(&self, key: &str) -> JsValue {
        // 1. Interceptor
        if let Some(cfg) = &self.named_handler
            && let Some(getter) = cfg.getter()
            && let Some(val) = getter(key, self.data_ptr())
        {
            return val;
        }
        // 2. Own properties
        self.properties
            .borrow()
            .get(key)
            .cloned()
            .unwrap_or(JsValue::Undefined)
    }

    /// Write a named property, consulting the interceptor first.
    ///
    /// If the interceptor handles the write (returns `true`), the value is
    /// *not* stored in the own-property map.
    pub fn set_property(&mut self, key: &str, value: JsValue) {
        if let Some(cfg) = &self.named_handler
            && let Some(setter) = cfg.setter()
            && setter(key, &value, self.data_ptr())
        {
            return;
        }
        self.properties.borrow_mut().insert(key.to_string(), value);
    }

    /// Query whether a named property exists, consulting the interceptor.
    pub fn has_property(&self, key: &str) -> bool {
        if let Some(cfg) = &self.named_handler
            && let Some(query) = cfg.query()
            && query(key, self.data_ptr()).is_some()
        {
            return true;
        }
        self.properties.borrow().contains_key(key)
    }

    /// Delete a named property, consulting the interceptor first.
    pub fn delete_property(&mut self, key: &str) -> bool {
        if let Some(cfg) = &self.named_handler
            && let Some(deleter) = cfg.deleter()
            && deleter(key, self.data_ptr())
        {
            return true;
        }
        self.properties.borrow_mut().remove(key).is_some()
    }

    /// Enumerate own property names, including those reported by the
    /// interceptor.
    pub fn property_names(&self) -> Vec<String> {
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
    /// Returns `JsValue::Undefined` if no indexed-property interceptor is
    /// installed or the interceptor does not handle this index.
    pub fn get_indexed(&self, index: u32) -> JsValue {
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
    /// Returns `true` if the interceptor handled the write.
    pub fn set_indexed(&mut self, index: u32, value: &JsValue) -> bool {
        if let Some(cfg) = &self.indexed_handler
            && let Some(setter) = cfg.setter()
        {
            return setter(index, value, self.data_ptr());
        }
        false
    }

    /// Query the length of the indexed collection via the interceptor.
    ///
    /// Returns `0` if no length callback is installed.
    pub fn indexed_length(&self) -> u32 {
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
/// use stator_core::dom::{DomObjectWrap, DomWeakRef};
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
}

//! V8-compatible FFI wrapper types for the Stator JavaScript engine.
//!
//! This module provides Rust types that mirror V8's C++ API types
//! (`v8::Object`, `v8::Array`, `v8::Number`, `v8::Function`, `v8::String`,
//! `v8::Boolean`, `v8::Value`, `v8::Promise`, `v8::TryCatch`,
//! `v8::ObjectTemplate`, `v8::FunctionTemplate`), allowing Chromium embedders
//! to interact with JavaScript values through a familiar V8-style interface.
//!
//! Each wrapper holds a [`JsValue`] internally and provides typed access
//! methods.  Wrappers implement [`From<JsValue>`] and [`Into<JsValue>`]
//! conversions, and panicking constructors validate that the inner value
//! matches the expected type.
//!
//! # Example
//!
//! ```
//! use stator_core::ffi::{V8Number, V8String, V8Object, V8Array, V8Promise, PromiseState};
//! use stator_core::objects::value::JsValue;
//!
//! let num = V8Number::new(3.14);
//! assert_eq!(num.value(), 3.14);
//!
//! let s = V8String::new("hello");
//! assert_eq!(s.value(), "hello");
//! assert_eq!(s.length(), 5);
//!
//! let mut obj = V8Object::new();
//! obj.set("x", JsValue::Smi(42));
//! assert!(obj.has("x"));
//! assert_eq!(obj.get("x"), JsValue::Smi(42));
//!
//! let mut arr = V8Array::new(0);
//! arr.push(JsValue::Smi(1));
//! arr.push(JsValue::Smi(2));
//! assert_eq!(arr.length(), 2);
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// V8Value — base wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A generic wrapper around any [`JsValue`], analogous to `v8::Value`.
///
/// This is the untyped base from which specific wrappers can be extracted
/// via the `try_into_*` conversion methods.
#[derive(Debug, Clone, PartialEq)]
pub struct V8Value {
    inner: JsValue,
}

impl V8Value {
    /// Wrap an arbitrary [`JsValue`].
    pub fn new(value: JsValue) -> Self {
        Self { inner: value }
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }

    /// Returns `true` if the inner value is `undefined`.
    pub fn is_undefined(&self) -> bool {
        self.inner.is_undefined()
    }

    /// Returns `true` if the inner value is `null`.
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    /// Returns `true` if the inner value is `null` or `undefined`.
    pub fn is_null_or_undefined(&self) -> bool {
        self.inner.is_nullish()
    }

    /// Returns `true` if the inner value is a boolean.
    pub fn is_boolean(&self) -> bool {
        self.inner.is_boolean()
    }

    /// Returns `true` if the inner value is a number (Smi or HeapNumber).
    pub fn is_number(&self) -> bool {
        self.inner.is_number()
    }

    /// Returns `true` if the inner value is a string.
    pub fn is_string(&self) -> bool {
        self.inner.is_string()
    }

    /// Returns `true` if the inner value is an object (PlainObject variant).
    pub fn is_object(&self) -> bool {
        matches!(self.inner, JsValue::PlainObject(_))
    }

    /// Returns `true` if the inner value is an array.
    pub fn is_array(&self) -> bool {
        self.inner.is_array()
    }

    /// Returns `true` if the inner value is a function.
    pub fn is_function(&self) -> bool {
        self.inner.is_function()
    }

    /// Try to convert into a [`V8Object`].
    pub fn try_into_object(self) -> Result<V8Object, Self> {
        if matches!(self.inner, JsValue::PlainObject(_)) {
            Ok(V8Object { inner: self.inner })
        } else {
            Err(self)
        }
    }

    /// Try to convert into a [`V8Array`].
    pub fn try_into_array(self) -> Result<V8Array, Self> {
        if self.inner.is_array() {
            Ok(V8Array { inner: self.inner })
        } else {
            Err(self)
        }
    }

    /// Try to convert into a [`V8Number`].
    pub fn try_into_number(self) -> Result<V8Number, Self> {
        if self.inner.is_number() {
            Ok(V8Number { inner: self.inner })
        } else {
            Err(self)
        }
    }

    /// Try to convert into a [`V8String`].
    pub fn try_into_string(self) -> Result<V8String, Self> {
        if self.inner.is_string() {
            Ok(V8String { inner: self.inner })
        } else {
            Err(self)
        }
    }

    /// Try to convert into a [`V8Function`].
    pub fn try_into_function(self) -> Result<V8Function, Self> {
        if self.inner.is_function() {
            Ok(V8Function { inner: self.inner })
        } else {
            Err(self)
        }
    }

    /// Try to convert into a [`V8Boolean`].
    pub fn try_into_boolean(self) -> Result<V8Boolean, Self> {
        if self.inner.is_boolean() {
            Ok(V8Boolean { inner: self.inner })
        } else {
            Err(self)
        }
    }
}

impl From<JsValue> for V8Value {
    fn from(value: JsValue) -> Self {
        Self { inner: value }
    }
}

impl From<V8Value> for JsValue {
    fn from(wrapper: V8Value) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Object
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a [`JsValue::PlainObject`] that provides V8-style Object
/// API.
///
/// This mirrors the `v8::Object` interface with property get/set/has/delete
/// and property-name enumeration.
#[derive(Debug, Clone, PartialEq)]
pub struct V8Object {
    inner: JsValue,
}

impl V8Object {
    /// Create a new, empty JavaScript object.
    pub fn new() -> Self {
        Self {
            inner: JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new()))),
        }
    }

    /// Get the value of a named property, or `JsValue::Undefined` if absent.
    pub fn get(&self, key: &str) -> JsValue {
        match &self.inner {
            JsValue::PlainObject(map) => {
                map.borrow().get(key).cloned().unwrap_or(JsValue::Undefined)
            }
            _ => JsValue::Undefined,
        }
    }

    /// Set (or overwrite) a named property on this object.
    pub fn set(&mut self, key: &str, value: JsValue) {
        if let JsValue::PlainObject(map) = &self.inner {
            map.borrow_mut().insert(key.to_string(), value);
        }
    }

    /// Returns `true` if this object has the named property.
    pub fn has(&self, key: &str) -> bool {
        match &self.inner {
            JsValue::PlainObject(map) => map.borrow().contains_key(key),
            _ => false,
        }
    }

    /// Delete a named property.  Returns `true` if the property was removed.
    pub fn delete(&mut self, key: &str) -> bool {
        match &self.inner {
            JsValue::PlainObject(map) => map.borrow_mut().remove(key).is_some(),
            _ => false,
        }
    }

    /// Return the own property names of this object as a sorted list.
    pub fn get_property_names(&self) -> Vec<String> {
        match &self.inner {
            JsValue::PlainObject(map) => {
                let mut names: Vec<String> = map.borrow().keys().cloned().collect();
                names.sort();
                names
            }
            _ => Vec::new(),
        }
    }

    /// Return the number of own properties.
    pub fn property_count(&self) -> usize {
        match &self.inner {
            JsValue::PlainObject(map) => map.borrow().len(),
            _ => 0,
        }
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl Default for V8Object {
    fn default() -> Self {
        Self::new()
    }
}

impl From<JsValue> for V8Object {
    /// Convert a [`JsValue`] into a [`V8Object`].
    ///
    /// # Panics
    /// Panics if the value is not a `PlainObject`.
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::PlainObject(_)),
            "V8Object::from: expected PlainObject, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8Object> for JsValue {
    fn from(wrapper: V8Object) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Array
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a [`JsValue::Array`] that provides V8-style Array API.
///
/// Mirrors the `v8::Array` interface with indexed get/set, `length`, and
/// `push`/`pop` convenience methods.
#[derive(Debug, Clone, PartialEq)]
pub struct V8Array {
    inner: JsValue,
}

impl V8Array {
    /// Create a new array pre-filled with `len` `undefined` elements.
    pub fn new(len: u32) -> Self {
        let items: Vec<JsValue> = vec![JsValue::Undefined; len as usize];
        Self {
            inner: JsValue::new_array(items),
        }
    }

    /// Create a new, empty array.
    pub fn empty() -> Self {
        Self::new(0)
    }

    /// Create an array from a `Vec<JsValue>`.
    pub fn from_vec(items: Vec<JsValue>) -> Self {
        Self {
            inner: JsValue::new_array(items),
        }
    }

    /// Return the number of elements.
    pub fn length(&self) -> u32 {
        match &self.inner {
            JsValue::Array(items) => items.borrow().len() as u32,
            _ => 0,
        }
    }

    /// Get the element at `index`, or `JsValue::Undefined` if out of bounds.
    pub fn get(&self, index: u32) -> JsValue {
        match &self.inner {
            JsValue::Array(items) => items
                .borrow()
                .get(index as usize)
                .cloned()
                .unwrap_or(JsValue::Undefined),
            _ => JsValue::Undefined,
        }
    }

    /// Set the element at `index`.
    ///
    /// If `index` is beyond the current length the array is extended with
    /// `undefined` elements to accommodate the new index.
    pub fn set(&mut self, index: u32, value: JsValue) {
        // JsValue::Array uses Rc<Vec<JsValue>>, so we need to make_mut.
        // Since Rc doesn't have make_mut, we reconstruct.
        if let JsValue::Array(items) = &self.inner {
            let mut new_items: Vec<JsValue> = items.borrow().clone();
            let idx = index as usize;
            if idx >= new_items.len() {
                new_items.resize(idx + 1, JsValue::Undefined);
            }
            new_items[idx] = value;
            self.inner = JsValue::new_array(new_items);
        }
    }

    /// Append a value to the end of the array.
    pub fn push(&mut self, value: JsValue) {
        if let JsValue::Array(items) = &self.inner {
            let mut new_items: Vec<JsValue> = items.borrow().clone();
            new_items.push(value);
            self.inner = JsValue::new_array(new_items);
        }
    }

    /// Remove and return the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<JsValue> {
        if let JsValue::Array(items) = &self.inner {
            let mut new_items: Vec<JsValue> = items.borrow().clone();
            let result = new_items.pop();
            self.inner = JsValue::new_array(new_items);
            result
        } else {
            None
        }
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl From<JsValue> for V8Array {
    /// Convert a [`JsValue`] into a [`V8Array`].
    ///
    /// # Panics
    /// Panics if the value is not an `Array`.
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::Array(_)),
            "V8Array::from: expected Array, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8Array> for JsValue {
    fn from(wrapper: V8Array) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Number
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a numeric [`JsValue`] that provides V8-style Number API.
///
/// Accepts both `Smi` and `HeapNumber` variants internally.
#[derive(Debug, Clone, PartialEq)]
pub struct V8Number {
    inner: JsValue,
}

impl V8Number {
    /// Create a new number value.
    ///
    /// Uses `Smi` for integer values in the i32 range, `HeapNumber` otherwise.
    pub fn new(value: f64) -> Self {
        let inner = if value.fract() == 0.0
            && !value.is_nan()
            && !value.is_infinite()
            && value >= f64::from(i32::MIN)
            && value <= f64::from(i32::MAX)
            && !(value == 0.0 && value.is_sign_negative())
        {
            JsValue::Smi(value as i32)
        } else {
            JsValue::HeapNumber(value)
        };
        Self { inner }
    }

    /// Return the numeric value as `f64`.
    pub fn value(&self) -> f64 {
        match &self.inner {
            JsValue::Smi(n) => f64::from(*n),
            JsValue::HeapNumber(n) => *n,
            _ => f64::NAN,
        }
    }

    /// Returns `true` if this number is an integer that fits in an `i32`.
    pub fn is_int32(&self) -> bool {
        matches!(self.inner, JsValue::Smi(_))
            || matches!(self.inner, JsValue::HeapNumber(n) if n.fract() == 0.0
                && !n.is_nan()
                && !n.is_infinite()
                && n >= f64::from(i32::MIN)
                && n <= f64::from(i32::MAX))
    }

    /// Return the value as `i32` if it is an integer, otherwise `None`.
    pub fn to_int32(&self) -> Option<i32> {
        match &self.inner {
            JsValue::Smi(n) => Some(*n),
            JsValue::HeapNumber(n)
                if n.fract() == 0.0
                    && !n.is_nan()
                    && !n.is_infinite()
                    && *n >= f64::from(i32::MIN)
                    && *n <= f64::from(i32::MAX) =>
            {
                Some(*n as i32)
            }
            _ => None,
        }
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl From<f64> for V8Number {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl From<i32> for V8Number {
    fn from(value: i32) -> Self {
        Self {
            inner: JsValue::Smi(value),
        }
    }
}

impl From<JsValue> for V8Number {
    /// Convert a [`JsValue`] into a [`V8Number`].
    ///
    /// # Panics
    /// Panics if the value is not numeric (`Smi` or `HeapNumber`).
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::Smi(_) | JsValue::HeapNumber(_)),
            "V8Number::from: expected Smi or HeapNumber, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8Number> for JsValue {
    fn from(wrapper: V8Number) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8String
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a [`JsValue::String`] that provides V8-style String API.
#[derive(Debug, Clone, PartialEq)]
pub struct V8String {
    inner: JsValue,
}

impl V8String {
    /// Create a new string value.
    pub fn new(value: &str) -> Self {
        Self {
            inner: JsValue::String(value.to_string()),
        }
    }

    /// Create from an owned `String`.
    pub fn from_owned(value: String) -> Self {
        Self {
            inner: JsValue::String(value),
        }
    }

    /// Return the string contents.
    pub fn value(&self) -> String {
        match &self.inner {
            JsValue::String(s) => s.clone(),
            _ => String::new(),
        }
    }

    /// Return the length in UTF-8 bytes (consistent with Rust `str::len`).
    pub fn length(&self) -> usize {
        match &self.inner {
            JsValue::String(s) => s.len(),
            _ => 0,
        }
    }

    /// Return the length in UTF-16 code units (consistent with JavaScript
    /// `String.prototype.length`).
    pub fn utf16_length(&self) -> usize {
        match &self.inner {
            JsValue::String(s) => s.encode_utf16().count(),
            _ => 0,
        }
    }

    /// Returns `true` if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.length() == 0
    }

    /// Concatenate this string with another, returning a new [`V8String`].
    pub fn concat(&self, other: &V8String) -> V8String {
        let mut s = self.value();
        s.push_str(&other.value());
        V8String::from_owned(s)
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl From<&str> for V8String {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for V8String {
    fn from(value: String) -> Self {
        Self::from_owned(value)
    }
}

impl From<JsValue> for V8String {
    /// Convert a [`JsValue`] into a [`V8String`].
    ///
    /// # Panics
    /// Panics if the value is not a `String`.
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::String(_)),
            "V8String::from: expected String, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8String> for JsValue {
    fn from(wrapper: V8String) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Boolean
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a [`JsValue::Boolean`] that provides V8-style Boolean API.
#[derive(Debug, Clone, PartialEq)]
pub struct V8Boolean {
    inner: JsValue,
}

impl V8Boolean {
    /// Create a new boolean value.
    pub fn new(value: bool) -> Self {
        Self {
            inner: JsValue::Boolean(value),
        }
    }

    /// Return the boolean value.
    pub fn value(&self) -> bool {
        match &self.inner {
            JsValue::Boolean(b) => *b,
            _ => false,
        }
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl From<bool> for V8Boolean {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<JsValue> for V8Boolean {
    /// Convert a [`JsValue`] into a [`V8Boolean`].
    ///
    /// # Panics
    /// Panics if the value is not a `Boolean`.
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::Boolean(_)),
            "V8Boolean::from: expected Boolean, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8Boolean> for JsValue {
    fn from(wrapper: V8Boolean) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Function
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper around a callable [`JsValue`] that provides V8-style Function API.
///
/// Supports both bytecode-compiled functions (`JsValue::Function`) and native
/// Rust functions (`JsValue::NativeFunction`).
#[derive(Debug, Clone, PartialEq)]
pub struct V8Function {
    inner: JsValue,
}

impl V8Function {
    /// Create a `V8Function` wrapping a native Rust callback.
    ///
    /// The callback receives a `Vec<JsValue>` of positional arguments and
    /// returns a `StatorResult<JsValue>`.
    pub fn from_native<F>(f: F) -> Self
    where
        F: Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static,
    {
        Self {
            inner: JsValue::NativeFunction(Rc::new(f)),
        }
    }

    /// Call this function with the given receiver and arguments.
    ///
    /// For native functions the receiver (`_recv`) is currently ignored;
    /// the function is invoked with `args` only.  Bytecode-compiled functions
    /// require the interpreter to execute, which is outside the scope of
    /// this wrapper — calling a bytecode function returns a `TypeError`.
    pub fn call(&self, _recv: JsValue, args: &[JsValue]) -> StatorResult<JsValue> {
        match &self.inner {
            JsValue::NativeFunction(f) => f(args.to_vec()),
            JsValue::Function(_) => Err(StatorError::TypeError(
                "V8Function::call: bytecode functions require the interpreter to execute"
                    .to_string(),
            )),
            _ => Err(StatorError::TypeError(
                "V8Function::call: value is not callable".to_string(),
            )),
        }
    }

    /// Returns `true` if this is a native (Rust) function.
    pub fn is_native(&self) -> bool {
        matches!(self.inner, JsValue::NativeFunction(_))
    }

    /// Returns `true` if this is a bytecode-compiled function.
    pub fn is_compiled(&self) -> bool {
        matches!(self.inner, JsValue::Function(_))
    }

    /// Return a reference to the underlying [`JsValue`].
    pub fn inner(&self) -> &JsValue {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`JsValue`].
    pub fn into_inner(self) -> JsValue {
        self.inner
    }
}

impl From<JsValue> for V8Function {
    /// Convert a [`JsValue`] into a [`V8Function`].
    ///
    /// # Panics
    /// Panics if the value is not callable (`Function` or `NativeFunction`).
    fn from(value: JsValue) -> Self {
        assert!(
            matches!(value, JsValue::Function(_) | JsValue::NativeFunction(_)),
            "V8Function::from: expected Function or NativeFunction, got {value:?}"
        );
        Self { inner: value }
    }
}

impl From<V8Function> for JsValue {
    fn from(wrapper: V8Function) -> Self {
        wrapper.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8TryCatch
// ─────────────────────────────────────────────────────────────────────────────

/// A scope guard that catches JavaScript exceptions, analogous to `v8::TryCatch`.
///
/// # Example
///
/// ```
/// use stator_core::ffi::V8TryCatch;
/// use stator_core::objects::value::JsValue;
///
/// let mut tc = V8TryCatch::new();
/// assert!(!tc.has_caught());
///
/// tc.set_caught(JsValue::String("error".into()));
/// assert!(tc.has_caught());
/// assert_eq!(tc.exception(), Some(&JsValue::String("error".into())));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct V8TryCatch {
    has_caught: bool,
    exception: Option<JsValue>,
    message: Option<String>,
}

impl V8TryCatch {
    /// Create a new, empty `TryCatch` scope with no caught exception.
    pub fn new() -> Self {
        Self {
            has_caught: false,
            exception: None,
            message: None,
        }
    }

    /// Returns `true` if an exception has been caught.
    pub fn has_caught(&self) -> bool {
        self.has_caught
    }

    /// Return the caught exception value, if any.
    pub fn exception(&self) -> Option<&JsValue> {
        self.exception.as_ref()
    }

    /// Return the exception message string, if any.
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Reset the `TryCatch`, clearing any caught exception.
    pub fn reset(&mut self) {
        self.has_caught = false;
        self.exception = None;
        self.message = None;
    }

    /// Record a caught exception.
    ///
    /// If the exception is a `JsValue::String`, its content is also stored
    /// as the message.  For `JsValue::Error`, the error's `Display`
    /// representation is used.  Otherwise the message is set to a generic
    /// `"[object]"` string.
    pub fn set_caught(&mut self, exception: JsValue) {
        self.has_caught = true;
        self.message = Some(match &exception {
            JsValue::String(s) => s.clone(),
            JsValue::Smi(n) => n.to_string(),
            JsValue::HeapNumber(n) => n.to_string(),
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Undefined => "undefined".to_string(),
            JsValue::Null => "null".to_string(),
            JsValue::Error(e) => format!("{:?}", e),
            _ => "[object]".to_string(),
        });
        self.exception = Some(exception);
    }
}

impl Default for V8TryCatch {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8Promise
// ─────────────────────────────────────────────────────────────────────────────

/// The state of a JavaScript promise, analogous to V8's `v8::Promise::PromiseState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromiseState {
    /// The promise has not yet been settled.
    Pending,
    /// The promise was resolved with a value.
    Fulfilled,
    /// The promise was rejected with a reason.
    Rejected,
}

/// A JavaScript promise wrapper, analogous to `v8::Promise`.
///
/// # Example
///
/// ```
/// use stator_core::ffi::{V8Promise, PromiseState};
/// use stator_core::objects::value::JsValue;
///
/// let mut p = V8Promise::new();
/// assert_eq!(*p.state(), PromiseState::Pending);
///
/// p.resolve(JsValue::Smi(42));
/// assert_eq!(*p.state(), PromiseState::Fulfilled);
/// assert_eq!(p.result(), Some(&JsValue::Smi(42)));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct V8Promise {
    state: PromiseState,
    result: Option<JsValue>,
}

impl V8Promise {
    /// Create a new pending promise with no result.
    pub fn new() -> Self {
        Self {
            state: PromiseState::Pending,
            result: None,
        }
    }

    /// Return a reference to the current promise state.
    pub fn state(&self) -> &PromiseState {
        &self.state
    }

    /// Return the result value (fulfillment value or rejection reason), if settled.
    pub fn result(&self) -> Option<&JsValue> {
        self.result.as_ref()
    }

    /// Resolve the promise with a fulfillment value.
    ///
    /// If the promise has already been settled this is a no-op, matching the
    /// V8 behaviour where a promise can only be settled once.
    pub fn resolve(&mut self, value: JsValue) {
        if self.state == PromiseState::Pending {
            self.state = PromiseState::Fulfilled;
            self.result = Some(value);
        }
    }

    /// Reject the promise with a reason.
    ///
    /// If the promise has already been settled this is a no-op.
    pub fn reject(&mut self, reason: JsValue) {
        if self.state == PromiseState::Pending {
            self.state = PromiseState::Rejected;
            self.result = Some(reason);
        }
    }
}

impl Default for V8Promise {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8ObjectTemplate
// ─────────────────────────────────────────────────────────────────────────────

/// Callback type for named-property getter interceptors on [`V8ObjectTemplate`].
pub type NamedGetterCb = Box<dyn Fn(&str) -> JsValue>;

/// A template for creating JavaScript objects with predefined properties,
/// analogous to `v8::ObjectTemplate`.
///
/// # Example
///
/// ```
/// use stator_core::ffi::V8ObjectTemplate;
/// use stator_core::objects::value::JsValue;
///
/// let mut tmpl = V8ObjectTemplate::new();
/// tmpl.set("x", JsValue::Smi(10));
/// tmpl.set("y", JsValue::Smi(20));
///
/// let instance = tmpl.new_instance();
/// // instance is a PlainObject with properties "x" and "y".
/// ```
pub struct V8ObjectTemplate {
    properties: HashMap<String, JsValue>,
    named_getter: Option<NamedGetterCb>,
}

impl V8ObjectTemplate {
    /// Create a new, empty object template.
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            named_getter: None,
        }
    }

    /// Set a property that will be present on every instance created from
    /// this template.
    pub fn set(&mut self, key: &str, value: JsValue) {
        self.properties.insert(key.to_string(), value);
    }

    /// Install a named-property getter interceptor.
    ///
    /// When set, this callback can be used by embedders to lazily provide
    /// property values.  The interceptor is stored on the template but is
    /// not automatically wired into instances created by [`new_instance`];
    /// it is available for embedder code that needs to consult the getter
    /// at object-creation time.
    ///
    /// [`new_instance`]: V8ObjectTemplate::new_instance
    pub fn set_named_getter<F>(&mut self, getter: F)
    where
        F: Fn(&str) -> JsValue + 'static,
    {
        self.named_getter = Some(Box::new(getter));
    }

    /// Return a reference to the named-property getter, if installed.
    pub fn named_getter(&self) -> Option<&dyn Fn(&str) -> JsValue> {
        self.named_getter.as_deref()
    }

    /// Return the number of predefined properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Return the predefined property names (sorted).
    pub fn property_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.properties.keys().cloned().collect();
        names.sort();
        names
    }

    /// Create a new [`JsValue::PlainObject`] instance populated with the
    /// predefined properties from this template.
    pub fn new_instance(&self) -> JsValue {
        let mut map = PropertyMap::new();
        for (k, v) in &self.properties {
            map.insert(k.clone(), v.clone());
        }
        JsValue::PlainObject(Rc::new(RefCell::new(map)))
    }
}

impl Default for V8ObjectTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for V8ObjectTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V8ObjectTemplate")
            .field("properties", &self.properties)
            .field("has_named_getter", &self.named_getter.is_some())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// V8FunctionTemplate
// ─────────────────────────────────────────────────────────────────────────────

/// A template for creating JavaScript function objects, analogous to
/// `v8::FunctionTemplate`.
///
/// # Example
///
/// ```
/// use stator_core::ffi::V8FunctionTemplate;
/// use stator_core::objects::value::JsValue;
///
/// let tmpl = V8FunctionTemplate::new(|args| {
///     let sum: i32 = args.iter().filter_map(|v| match v {
///         JsValue::Smi(n) => Some(*n),
///         _ => None,
///     }).sum();
///     JsValue::Smi(sum)
/// });
///
/// let func = tmpl.get_function();
/// assert!(func.is_function());
/// ```
pub struct V8FunctionTemplate {
    callback: Rc<dyn Fn(Vec<JsValue>) -> JsValue>,
    class_name: Option<String>,
}

impl V8FunctionTemplate {
    /// Create a new function template wrapping the given callback.
    ///
    /// The callback receives a `Vec<JsValue>` of positional arguments and
    /// returns a `JsValue` result.
    pub fn new(callback: impl Fn(Vec<JsValue>) -> JsValue + 'static) -> Self {
        Self {
            callback: Rc::new(callback),
            class_name: None,
        }
    }

    /// Set the class name for instances created from this template.
    ///
    /// This mirrors `v8::FunctionTemplate::SetClassName()`.
    pub fn set_class_name(&mut self, name: &str) {
        self.class_name = Some(name.to_string());
    }

    /// Return the class name, if set.
    pub fn class_name(&self) -> Option<&str> {
        self.class_name.as_deref()
    }

    /// Create a [`JsValue::NativeFunction`] from this template.
    ///
    /// The returned value wraps the template's callback so that it can be
    /// called from the interpreter like any other native function.
    pub fn get_function(&self) -> JsValue {
        let cb = Rc::clone(&self.callback);
        JsValue::NativeFunction(Rc::new(move |args| Ok(cb(args))))
    }
}

impl std::fmt::Debug for V8FunctionTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V8FunctionTemplate")
            .field("class_name", &self.class_name)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── V8Value ──────────────────────────────────────────────────────────────

    #[test]
    fn test_v8value_type_predicates() {
        let undef = V8Value::new(JsValue::Undefined);
        assert!(undef.is_undefined());
        assert!(!undef.is_null());

        let null = V8Value::new(JsValue::Null);
        assert!(null.is_null());
        assert!(null.is_null_or_undefined());

        let num = V8Value::new(JsValue::Smi(42));
        assert!(num.is_number());
        assert!(!num.is_string());

        let s = V8Value::new(JsValue::String("hi".into()));
        assert!(s.is_string());

        let b = V8Value::new(JsValue::Boolean(true));
        assert!(b.is_boolean());
    }

    #[test]
    fn test_v8value_try_into_conversions() {
        let num_val = V8Value::new(JsValue::Smi(7));
        let num = num_val.try_into_number().unwrap();
        assert_eq!(num.value(), 7.0);

        let str_val = V8Value::new(JsValue::String("hello".into()));
        let s = str_val.try_into_string().unwrap();
        assert_eq!(s.value(), "hello");

        // Failing conversion returns Err(self).
        let bad = V8Value::new(JsValue::Null);
        assert!(bad.try_into_number().is_err());
    }

    #[test]
    fn test_v8value_from_into_jsvalue() {
        let original = JsValue::Smi(99);
        let wrapped = V8Value::from(original.clone());
        let back: JsValue = wrapped.into();
        assert_eq!(back, original);
    }

    // ── V8Object ─────────────────────────────────────────────────────────────

    #[test]
    fn test_v8object_new_is_empty() {
        let obj = V8Object::new();
        assert_eq!(obj.property_count(), 0);
        assert!(obj.get_property_names().is_empty());
    }

    #[test]
    fn test_v8object_set_get_has() {
        let mut obj = V8Object::new();
        assert!(!obj.has("x"));
        assert_eq!(obj.get("x"), JsValue::Undefined);

        obj.set("x", JsValue::Smi(42));
        assert!(obj.has("x"));
        assert_eq!(obj.get("x"), JsValue::Smi(42));

        // Overwrite.
        obj.set("x", JsValue::String("hello".into()));
        assert_eq!(obj.get("x"), JsValue::String("hello".into()));
    }

    #[test]
    fn test_v8object_delete() {
        let mut obj = V8Object::new();
        obj.set("a", JsValue::Smi(1));
        assert!(obj.has("a"));
        assert!(obj.delete("a"));
        assert!(!obj.has("a"));
        // Deleting non-existent returns false.
        assert!(!obj.delete("b"));
    }

    #[test]
    fn test_v8object_get_property_names() {
        let mut obj = V8Object::new();
        obj.set("b", JsValue::Smi(2));
        obj.set("a", JsValue::Smi(1));
        obj.set("c", JsValue::Smi(3));
        assert_eq!(obj.get_property_names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_v8object_from_jsvalue() {
        let map = Rc::new(RefCell::new(PropertyMap::new()));
        map.borrow_mut().insert("key".to_string(), JsValue::Smi(10));
        let val = JsValue::PlainObject(map);
        let obj = V8Object::from(val);
        assert_eq!(obj.get("key"), JsValue::Smi(10));
    }

    #[test]
    #[should_panic(expected = "expected PlainObject")]
    fn test_v8object_from_wrong_type_panics() {
        let _ = V8Object::from(JsValue::Smi(1));
    }

    #[test]
    fn test_v8object_into_jsvalue() {
        let mut obj = V8Object::new();
        obj.set("k", JsValue::Boolean(true));
        let val: JsValue = obj.into();
        assert!(matches!(val, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_v8object_default() {
        let obj = V8Object::default();
        assert_eq!(obj.property_count(), 0);
    }

    // ── V8Array ──────────────────────────────────────────────────────────────

    #[test]
    fn test_v8array_new_with_length() {
        let arr = V8Array::new(3);
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.get(0), JsValue::Undefined);
        assert_eq!(arr.get(1), JsValue::Undefined);
        assert_eq!(arr.get(2), JsValue::Undefined);
    }

    #[test]
    fn test_v8array_empty() {
        let arr = V8Array::empty();
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_v8array_from_vec() {
        let arr = V8Array::from_vec(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.get(0), JsValue::Smi(1));
        assert_eq!(arr.get(1), JsValue::Smi(2));
    }

    #[test]
    fn test_v8array_set() {
        let mut arr = V8Array::new(2);
        arr.set(0, JsValue::Smi(10));
        arr.set(1, JsValue::Smi(20));
        assert_eq!(arr.get(0), JsValue::Smi(10));
        assert_eq!(arr.get(1), JsValue::Smi(20));
    }

    #[test]
    fn test_v8array_set_extends() {
        let mut arr = V8Array::new(0);
        arr.set(2, JsValue::Smi(99));
        assert_eq!(arr.length(), 3);
        assert_eq!(arr.get(0), JsValue::Undefined);
        assert_eq!(arr.get(1), JsValue::Undefined);
        assert_eq!(arr.get(2), JsValue::Smi(99));
    }

    #[test]
    fn test_v8array_push_pop() {
        let mut arr = V8Array::empty();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::Smi(2));
        assert_eq!(arr.length(), 2);

        assert_eq!(arr.pop(), Some(JsValue::Smi(2)));
        assert_eq!(arr.length(), 1);
        assert_eq!(arr.pop(), Some(JsValue::Smi(1)));
        assert_eq!(arr.pop(), None);
    }

    #[test]
    fn test_v8array_get_out_of_bounds() {
        let arr = V8Array::new(1);
        assert_eq!(arr.get(100), JsValue::Undefined);
    }

    #[test]
    fn test_v8array_from_jsvalue() {
        let val = JsValue::new_array(vec![JsValue::Smi(5)]);
        let arr = V8Array::from(val);
        assert_eq!(arr.length(), 1);
        assert_eq!(arr.get(0), JsValue::Smi(5));
    }

    #[test]
    #[should_panic(expected = "expected Array")]
    fn test_v8array_from_wrong_type_panics() {
        let _ = V8Array::from(JsValue::Null);
    }

    #[test]
    fn test_v8array_into_jsvalue() {
        let arr = V8Array::from_vec(vec![JsValue::Smi(1)]);
        let val: JsValue = arr.into();
        assert!(matches!(val, JsValue::Array(_)));
    }

    // ── V8Number ─────────────────────────────────────────────────────────────

    #[test]
    fn test_v8number_new_integer() {
        let n = V8Number::new(42.0);
        assert_eq!(n.value(), 42.0);
        // Should be stored as Smi for integer values.
        assert!(matches!(n.inner(), JsValue::Smi(42)));
    }

    #[test]
    fn test_v8number_new_float() {
        let n = V8Number::new(3.14);
        assert_eq!(n.value(), 3.14);
        assert!(matches!(n.inner(), JsValue::HeapNumber(_)));
    }

    #[test]
    fn test_v8number_new_nan() {
        let n = V8Number::new(f64::NAN);
        assert!(n.value().is_nan());
    }

    #[test]
    fn test_v8number_new_infinity() {
        let pos = V8Number::new(f64::INFINITY);
        assert_eq!(pos.value(), f64::INFINITY);
        let neg = V8Number::new(f64::NEG_INFINITY);
        assert_eq!(neg.value(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_v8number_new_neg_zero() {
        let n = V8Number::new(-0.0);
        assert_eq!(n.value(), 0.0);
        assert!(n.value().is_sign_negative());
        // Negative zero should be stored as HeapNumber, not Smi.
        assert!(matches!(n.inner(), JsValue::HeapNumber(_)));
    }

    #[test]
    fn test_v8number_is_int32() {
        assert!(V8Number::new(0.0).is_int32());
        assert!(V8Number::new(100.0).is_int32());
        assert!(V8Number::new(-100.0).is_int32());
        assert!(!V8Number::new(3.14).is_int32());
        assert!(!V8Number::new(f64::NAN).is_int32());
        assert!(!V8Number::new(f64::INFINITY).is_int32());
    }

    #[test]
    fn test_v8number_to_int32() {
        assert_eq!(V8Number::new(42.0).to_int32(), Some(42));
        assert_eq!(V8Number::new(-7.0).to_int32(), Some(-7));
        assert_eq!(V8Number::new(3.14).to_int32(), None);
        assert_eq!(V8Number::new(f64::NAN).to_int32(), None);
    }

    #[test]
    fn test_v8number_from_i32() {
        let n = V8Number::from(42i32);
        assert_eq!(n.value(), 42.0);
        assert!(matches!(n.inner(), JsValue::Smi(42)));
    }

    #[test]
    fn test_v8number_from_f64() {
        let n = V8Number::from(2.5f64);
        assert_eq!(n.value(), 2.5);
    }

    #[test]
    fn test_v8number_from_jsvalue() {
        let n = V8Number::from(JsValue::Smi(10));
        assert_eq!(n.value(), 10.0);
        let n2 = V8Number::from(JsValue::HeapNumber(1.5));
        assert_eq!(n2.value(), 1.5);
    }

    #[test]
    #[should_panic(expected = "expected Smi or HeapNumber")]
    fn test_v8number_from_wrong_type_panics() {
        let _ = V8Number::from(JsValue::String("oops".into()));
    }

    #[test]
    fn test_v8number_into_jsvalue() {
        let n = V8Number::new(5.0);
        let val: JsValue = n.into();
        assert_eq!(val, JsValue::Smi(5));
    }

    // ── V8String ─────────────────────────────────────────────────────────────

    #[test]
    fn test_v8string_new() {
        let s = V8String::new("hello");
        assert_eq!(s.value(), "hello");
        assert_eq!(s.length(), 5);
    }

    #[test]
    fn test_v8string_empty() {
        let s = V8String::new("");
        assert!(s.is_empty());
        assert_eq!(s.length(), 0);
    }

    #[test]
    fn test_v8string_from_owned() {
        let s = V8String::from_owned("world".to_string());
        assert_eq!(s.value(), "world");
    }

    #[test]
    fn test_v8string_utf16_length() {
        // ASCII: each char is 1 UTF-16 code unit.
        let ascii = V8String::new("abc");
        assert_eq!(ascii.utf16_length(), 3);

        // Emoji: U+1F600 is 2 UTF-16 code units (surrogate pair).
        let emoji = V8String::new("😀");
        assert_eq!(emoji.utf16_length(), 2);
        assert_eq!(emoji.length(), 4); // 4 UTF-8 bytes
    }

    #[test]
    fn test_v8string_concat() {
        let a = V8String::new("hello");
        let b = V8String::new(" world");
        let c = a.concat(&b);
        assert_eq!(c.value(), "hello world");
    }

    #[test]
    fn test_v8string_from_str() {
        let s: V8String = "test".into();
        assert_eq!(s.value(), "test");
    }

    #[test]
    fn test_v8string_from_string() {
        let s: V8String = String::from("test").into();
        assert_eq!(s.value(), "test");
    }

    #[test]
    fn test_v8string_from_jsvalue() {
        let s = V8String::from(JsValue::String("hi".into()));
        assert_eq!(s.value(), "hi");
    }

    #[test]
    #[should_panic(expected = "expected String")]
    fn test_v8string_from_wrong_type_panics() {
        let _ = V8String::from(JsValue::Smi(1));
    }

    #[test]
    fn test_v8string_into_jsvalue() {
        let s = V8String::new("bye");
        let val: JsValue = s.into();
        assert_eq!(val, JsValue::String("bye".into()));
    }

    // ── V8Boolean ────────────────────────────────────────────────────────────

    #[test]
    fn test_v8boolean_true() {
        let b = V8Boolean::new(true);
        assert!(b.value());
    }

    #[test]
    fn test_v8boolean_false() {
        let b = V8Boolean::new(false);
        assert!(!b.value());
    }

    #[test]
    fn test_v8boolean_from_bool() {
        let b: V8Boolean = true.into();
        assert!(b.value());
    }

    #[test]
    fn test_v8boolean_from_jsvalue() {
        let b = V8Boolean::from(JsValue::Boolean(false));
        assert!(!b.value());
    }

    #[test]
    #[should_panic(expected = "expected Boolean")]
    fn test_v8boolean_from_wrong_type_panics() {
        let _ = V8Boolean::from(JsValue::Smi(0));
    }

    #[test]
    fn test_v8boolean_into_jsvalue() {
        let b = V8Boolean::new(true);
        let val: JsValue = b.into();
        assert_eq!(val, JsValue::Boolean(true));
    }

    // ── V8Function ───────────────────────────────────────────────────────────

    #[test]
    fn test_v8function_from_native_call() {
        let f = V8Function::from_native(|args| {
            // Sum all Smi arguments.
            let sum: i32 = args
                .iter()
                .filter_map(|v| match v {
                    JsValue::Smi(n) => Some(*n),
                    _ => None,
                })
                .sum();
            Ok(JsValue::Smi(sum))
        });
        assert!(f.is_native());
        assert!(!f.is_compiled());

        let result = f
            .call(
                JsValue::Undefined,
                &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
            )
            .unwrap();
        assert_eq!(result, JsValue::Smi(60));
    }

    #[test]
    fn test_v8function_call_no_args() {
        let f = V8Function::from_native(|_args| Ok(JsValue::String("ok".into())));
        let result = f.call(JsValue::Undefined, &[]).unwrap();
        assert_eq!(result, JsValue::String("ok".into()));
    }

    #[test]
    fn test_v8function_call_error() {
        let f = V8Function::from_native(|_args| Err(StatorError::TypeError("boom".to_string())));
        let result = f.call(JsValue::Undefined, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_v8function_from_jsvalue() {
        let native: JsValue = JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined)));
        let f = V8Function::from(native);
        assert!(f.is_native());
    }

    #[test]
    #[should_panic(expected = "expected Function or NativeFunction")]
    fn test_v8function_from_wrong_type_panics() {
        let _ = V8Function::from(JsValue::Smi(1));
    }

    #[test]
    fn test_v8function_into_jsvalue() {
        let f = V8Function::from_native(|_| Ok(JsValue::Undefined));
        let val: JsValue = f.into();
        assert!(matches!(val, JsValue::NativeFunction(_)));
    }

    // ── V8Value try_into with object/array/function ──────────────────────────

    #[test]
    fn test_v8value_try_into_object() {
        let obj_val = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let v = V8Value::new(obj_val);
        assert!(v.is_object());
        let obj = v.try_into_object().unwrap();
        assert_eq!(obj.property_count(), 0);
    }

    #[test]
    fn test_v8value_try_into_array() {
        let arr_val = JsValue::new_array(vec![JsValue::Smi(1)]);
        let v = V8Value::new(arr_val);
        assert!(v.is_array());
        let arr = v.try_into_array().unwrap();
        assert_eq!(arr.length(), 1);
    }

    #[test]
    fn test_v8value_try_into_function() {
        let f_val = JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined)));
        let v = V8Value::new(f_val);
        assert!(v.is_function());
        let f = v.try_into_function().unwrap();
        assert!(f.is_native());
    }

    #[test]
    fn test_v8value_try_into_boolean() {
        let v = V8Value::new(JsValue::Boolean(true));
        let b = v.try_into_boolean().unwrap();
        assert!(b.value());
    }

    // ── Cross-type round-trip ────────────────────────────────────────────────

    #[test]
    fn test_object_with_nested_values() {
        let mut obj = V8Object::new();
        let arr = V8Array::from_vec(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let num = V8Number::new(3.14);
        let s = V8String::new("hello");

        obj.set("array", arr.into_inner());
        obj.set("number", num.into_inner());
        obj.set("string", s.into_inner());

        assert_eq!(obj.property_count(), 3);

        // Extract and verify the array.
        let arr_val = obj.get("array");
        let arr2 = V8Array::from(arr_val);
        assert_eq!(arr2.length(), 2);
        assert_eq!(arr2.get(0), JsValue::Smi(1));

        // Extract and verify the number.
        let num_val = obj.get("number");
        let num2 = V8Number::from(num_val);
        assert_eq!(num2.value(), 3.14);

        // Extract and verify the string.
        let str_val = obj.get("string");
        let str2 = V8String::from(str_val);
        assert_eq!(str2.value(), "hello");
    }

    #[test]
    fn test_array_of_objects() {
        let mut obj1 = V8Object::new();
        obj1.set("name", JsValue::String("Alice".into()));
        let mut obj2 = V8Object::new();
        obj2.set("name", JsValue::String("Bob".into()));

        let arr = V8Array::from_vec(vec![obj1.into_inner(), obj2.into_inner()]);
        assert_eq!(arr.length(), 2);

        let first = V8Object::from(arr.get(0));
        assert_eq!(first.get("name"), JsValue::String("Alice".into()));
    }

    // ── V8TryCatch ───────────────────────────────────────────────────────────

    #[test]
    fn test_trycatch_new_is_empty() {
        let tc = V8TryCatch::new();
        assert!(!tc.has_caught());
        assert!(tc.exception().is_none());
        assert!(tc.message().is_none());
    }

    #[test]
    fn test_trycatch_default() {
        let tc = V8TryCatch::default();
        assert!(!tc.has_caught());
    }

    #[test]
    fn test_trycatch_set_caught_string() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::String("oops".into()));
        assert!(tc.has_caught());
        assert_eq!(tc.exception(), Some(&JsValue::String("oops".into())));
        assert_eq!(tc.message(), Some("oops"));
    }

    #[test]
    fn test_trycatch_set_caught_number() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::Smi(42));
        assert!(tc.has_caught());
        assert_eq!(tc.exception(), Some(&JsValue::Smi(42)));
        assert_eq!(tc.message(), Some("42"));
    }

    #[test]
    fn test_trycatch_set_caught_boolean() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::Boolean(false));
        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("false"));
    }

    #[test]
    fn test_trycatch_set_caught_undefined() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::Undefined);
        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("undefined"));
    }

    #[test]
    fn test_trycatch_set_caught_null() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::Null);
        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("null"));
    }

    #[test]
    fn test_trycatch_set_caught_object() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::PlainObject(Rc::new(RefCell::new(
            PropertyMap::new(),
        ))));
        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("[object]"));
    }

    #[test]
    fn test_trycatch_set_caught_heap_number() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::HeapNumber(3.14));
        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("3.14"));
    }

    #[test]
    fn test_trycatch_reset() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::String("err".into()));
        assert!(tc.has_caught());

        tc.reset();
        assert!(!tc.has_caught());
        assert!(tc.exception().is_none());
        assert!(tc.message().is_none());
    }

    #[test]
    fn test_trycatch_multiple_catches() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::String("first".into()));
        assert_eq!(tc.message(), Some("first"));

        // Catch again overwrites the previous exception.
        tc.set_caught(JsValue::String("second".into()));
        assert_eq!(tc.message(), Some("second"));
        assert_eq!(tc.exception(), Some(&JsValue::String("second".into())));
    }

    #[test]
    fn test_trycatch_reset_then_catch_again() {
        let mut tc = V8TryCatch::new();
        tc.set_caught(JsValue::Smi(1));
        tc.reset();
        assert!(!tc.has_caught());

        tc.set_caught(JsValue::Smi(2));
        assert!(tc.has_caught());
        assert_eq!(tc.exception(), Some(&JsValue::Smi(2)));
    }

    // ── V8Promise ────────────────────────────────────────────────────────────

    #[test]
    fn test_promise_new_is_pending() {
        let p = V8Promise::new();
        assert_eq!(*p.state(), PromiseState::Pending);
        assert!(p.result().is_none());
    }

    #[test]
    fn test_promise_default() {
        let p = V8Promise::default();
        assert_eq!(*p.state(), PromiseState::Pending);
    }

    #[test]
    fn test_promise_resolve() {
        let mut p = V8Promise::new();
        p.resolve(JsValue::Smi(42));
        assert_eq!(*p.state(), PromiseState::Fulfilled);
        assert_eq!(p.result(), Some(&JsValue::Smi(42)));
    }

    #[test]
    fn test_promise_reject() {
        let mut p = V8Promise::new();
        p.reject(JsValue::String("fail".into()));
        assert_eq!(*p.state(), PromiseState::Rejected);
        assert_eq!(p.result(), Some(&JsValue::String("fail".into())));
    }

    #[test]
    fn test_promise_resolve_only_once() {
        let mut p = V8Promise::new();
        p.resolve(JsValue::Smi(1));
        // Second resolve is a no-op.
        p.resolve(JsValue::Smi(2));
        assert_eq!(*p.state(), PromiseState::Fulfilled);
        assert_eq!(p.result(), Some(&JsValue::Smi(1)));
    }

    #[test]
    fn test_promise_reject_only_once() {
        let mut p = V8Promise::new();
        p.reject(JsValue::String("a".into()));
        p.reject(JsValue::String("b".into()));
        assert_eq!(*p.state(), PromiseState::Rejected);
        assert_eq!(p.result(), Some(&JsValue::String("a".into())));
    }

    #[test]
    fn test_promise_resolve_then_reject_noop() {
        let mut p = V8Promise::new();
        p.resolve(JsValue::Smi(10));
        p.reject(JsValue::String("err".into()));
        assert_eq!(*p.state(), PromiseState::Fulfilled);
        assert_eq!(p.result(), Some(&JsValue::Smi(10)));
    }

    #[test]
    fn test_promise_reject_then_resolve_noop() {
        let mut p = V8Promise::new();
        p.reject(JsValue::String("err".into()));
        p.resolve(JsValue::Smi(10));
        assert_eq!(*p.state(), PromiseState::Rejected);
        assert_eq!(p.result(), Some(&JsValue::String("err".into())));
    }

    #[test]
    fn test_promise_resolve_undefined() {
        let mut p = V8Promise::new();
        p.resolve(JsValue::Undefined);
        assert_eq!(*p.state(), PromiseState::Fulfilled);
        assert_eq!(p.result(), Some(&JsValue::Undefined));
    }

    #[test]
    fn test_promise_state_enum_equality() {
        assert_eq!(PromiseState::Pending, PromiseState::Pending);
        assert_ne!(PromiseState::Pending, PromiseState::Fulfilled);
        assert_ne!(PromiseState::Fulfilled, PromiseState::Rejected);
    }

    // ── V8ObjectTemplate ─────────────────────────────────────────────────────

    #[test]
    fn test_object_template_new_empty() {
        let tmpl = V8ObjectTemplate::new();
        assert_eq!(tmpl.property_count(), 0);
        assert!(tmpl.property_names().is_empty());
        assert!(tmpl.named_getter().is_none());
    }

    #[test]
    fn test_object_template_default() {
        let tmpl = V8ObjectTemplate::default();
        assert_eq!(tmpl.property_count(), 0);
    }

    #[test]
    fn test_object_template_set_properties() {
        let mut tmpl = V8ObjectTemplate::new();
        tmpl.set("x", JsValue::Smi(10));
        tmpl.set("y", JsValue::Smi(20));
        assert_eq!(tmpl.property_count(), 2);
        assert_eq!(tmpl.property_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_object_template_overwrite_property() {
        let mut tmpl = V8ObjectTemplate::new();
        tmpl.set("x", JsValue::Smi(1));
        tmpl.set("x", JsValue::Smi(2));
        assert_eq!(tmpl.property_count(), 1);

        let instance = tmpl.new_instance();
        let obj = V8Object::from(instance);
        assert_eq!(obj.get("x"), JsValue::Smi(2));
    }

    #[test]
    fn test_object_template_new_instance() {
        let mut tmpl = V8ObjectTemplate::new();
        tmpl.set("a", JsValue::Smi(1));
        tmpl.set("b", JsValue::String("hello".into()));

        let instance = tmpl.new_instance();
        assert!(matches!(instance, JsValue::PlainObject(_)));

        let obj = V8Object::from(instance);
        assert_eq!(obj.get("a"), JsValue::Smi(1));
        assert_eq!(obj.get("b"), JsValue::String("hello".into()));
        assert_eq!(obj.property_count(), 2);
    }

    #[test]
    fn test_object_template_instances_are_independent() {
        let mut tmpl = V8ObjectTemplate::new();
        tmpl.set("x", JsValue::Smi(10));

        let i1 = tmpl.new_instance();
        let i2 = tmpl.new_instance();

        // Mutating one instance does not affect the other.
        let mut obj1 = V8Object::from(i1);
        obj1.set("x", JsValue::Smi(99));

        let obj2 = V8Object::from(i2);
        assert_eq!(obj2.get("x"), JsValue::Smi(10));
    }

    #[test]
    fn test_object_template_set_named_getter() {
        let mut tmpl = V8ObjectTemplate::new();
        tmpl.set_named_getter(|key| {
            if key == "magic" {
                JsValue::Smi(42)
            } else {
                JsValue::Undefined
            }
        });

        assert!(tmpl.named_getter().is_some());
        let getter = tmpl.named_getter().unwrap();
        assert_eq!(getter("magic"), JsValue::Smi(42));
        assert_eq!(getter("other"), JsValue::Undefined);
    }

    #[test]
    fn test_object_template_debug() {
        let tmpl = V8ObjectTemplate::new();
        let debug_str = format!("{:?}", tmpl);
        assert!(debug_str.contains("V8ObjectTemplate"));
    }

    // ── V8FunctionTemplate ───────────────────────────────────────────────────

    #[test]
    fn test_function_template_new() {
        let tmpl = V8FunctionTemplate::new(|_args| JsValue::Undefined);
        assert!(tmpl.class_name().is_none());
    }

    #[test]
    fn test_function_template_set_class_name() {
        let mut tmpl = V8FunctionTemplate::new(|_| JsValue::Undefined);
        tmpl.set_class_name("MyClass");
        assert_eq!(tmpl.class_name(), Some("MyClass"));
    }

    #[test]
    fn test_function_template_get_function() {
        let tmpl = V8FunctionTemplate::new(|_args| JsValue::Smi(99));
        let func = tmpl.get_function();
        assert!(func.is_function());

        // Call the resulting NativeFunction.
        if let JsValue::NativeFunction(f) = &func {
            let result = f(vec![]).unwrap();
            assert_eq!(result, JsValue::Smi(99));
        } else {
            panic!("expected NativeFunction");
        }
    }

    #[test]
    fn test_function_template_get_function_with_args() {
        let tmpl = V8FunctionTemplate::new(|args| {
            let sum: i32 = args
                .iter()
                .filter_map(|v| match v {
                    JsValue::Smi(n) => Some(*n),
                    _ => None,
                })
                .sum();
            JsValue::Smi(sum)
        });

        let func = tmpl.get_function();
        if let JsValue::NativeFunction(f) = &func {
            let result = f(vec![JsValue::Smi(10), JsValue::Smi(20)]).unwrap();
            assert_eq!(result, JsValue::Smi(30));
        } else {
            panic!("expected NativeFunction");
        }
    }

    #[test]
    fn test_function_template_get_function_via_v8function() {
        let tmpl = V8FunctionTemplate::new(|_| JsValue::String("ok".into()));
        let func_val = tmpl.get_function();
        let f = V8Function::from(func_val);
        assert!(f.is_native());

        let result = f.call(JsValue::Undefined, &[]).unwrap();
        assert_eq!(result, JsValue::String("ok".into()));
    }

    #[test]
    fn test_function_template_multiple_functions_share_callback() {
        let tmpl = V8FunctionTemplate::new(|_| JsValue::Smi(7));
        let f1 = tmpl.get_function();
        let f2 = tmpl.get_function();

        // Both functions produce the same result.
        if let (JsValue::NativeFunction(fn1), JsValue::NativeFunction(fn2)) = (&f1, &f2) {
            assert_eq!(fn1(vec![]).unwrap(), JsValue::Smi(7));
            assert_eq!(fn2(vec![]).unwrap(), JsValue::Smi(7));
        } else {
            panic!("expected NativeFunction");
        }
    }

    #[test]
    fn test_function_template_debug() {
        let mut tmpl = V8FunctionTemplate::new(|_| JsValue::Undefined);
        tmpl.set_class_name("Foo");
        let debug_str = format!("{:?}", tmpl);
        assert!(debug_str.contains("V8FunctionTemplate"));
        assert!(debug_str.contains("Foo"));
    }

    // ── Cross-type: template + promise + trycatch ────────────────────────────

    #[test]
    fn test_object_template_with_native_function() {
        let mut tmpl = V8ObjectTemplate::new();
        let fn_tmpl = V8FunctionTemplate::new(|args| {
            if let Some(JsValue::Smi(n)) = args.first() {
                JsValue::Smi(n * 2)
            } else {
                JsValue::Undefined
            }
        });
        tmpl.set("double", fn_tmpl.get_function());

        let instance = tmpl.new_instance();
        let obj = V8Object::from(instance);
        let func_val = obj.get("double");
        let f = V8Function::from(func_val);
        let result = f.call(JsValue::Undefined, &[JsValue::Smi(5)]).unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn test_trycatch_with_function_error() {
        let mut tc = V8TryCatch::new();
        let f = V8Function::from_native(|_| Err(StatorError::TypeError("kaboom".to_string())));

        match f.call(JsValue::Undefined, &[]) {
            Err(StatorError::TypeError(msg)) => {
                tc.set_caught(JsValue::String(msg.clone()));
            }
            _ => panic!("expected error"),
        }

        assert!(tc.has_caught());
        assert_eq!(tc.message(), Some("kaboom"));
    }
}

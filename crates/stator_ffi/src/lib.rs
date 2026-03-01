//! `stator_ffi` — C-ABI FFI surface for the Stator JavaScript engine.
//!
//! This crate exposes a stable C API (`extern "C"`, `#[no_mangle]`) so that
//! embedders (e.g. Chromium's content layer) can link against Stator without
//! depending on Rust tooling.
//!
//! # Design
//! All opaque handle types are prefixed `Stator` and passed as raw pointers.
//! Memory is always owned by the Stator side: callers obtain handles through
//! `_create` functions and must release them with the corresponding `_destroy`
//! function.

use std::ffi::{CStr, CString, c_char};

use stator_core::gc::heap::Heap;
use stator_core::objects::js_object::JsObject;
use stator_core::objects::value::JsValue;

/// An opaque isolate handle.
///
/// An isolate is an independent instance of the Stator engine with its own
/// heap and root set.  Isolates are not thread-safe by default; access from
/// multiple threads requires external synchronisation.
pub struct StatorIsolate {
    heap: Heap,
    /// Number of live `StatorValue` / `StatorObject` handles currently owned
    /// by the embedder.  Incremented on `_new_*` / `_object_new`, decremented
    /// on `_destroy`.
    live_objects: usize,
}

/// Create a new isolate.
///
/// The returned pointer must eventually be passed to [`stator_isolate_destroy`]
/// to free all associated resources.  Returns a null pointer on allocation
/// failure (extremely unlikely in practice).
#[unsafe(no_mangle)]
pub extern "C" fn stator_isolate_create() -> *mut StatorIsolate {
    Box::into_raw(Box::new(StatorIsolate {
        heap: Heap::new(),
        live_objects: 0,
    }))
}

/// Destroy an isolate previously created with [`stator_isolate_create`].
///
/// After this call the pointer is invalid and must not be used.
///
/// # Safety
/// - `isolate` must be a non-null pointer returned by `stator_isolate_create`.
/// - `isolate` must not be used again after this call.
/// - This function must not be called more than once for the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_destroy(isolate: *mut StatorIsolate) {
    if !isolate.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in `stator_isolate_create`.
        drop(unsafe { Box::from_raw(isolate) });
    }
}

/// Trigger a minor (young-generation) garbage collection on the isolate's heap.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live `StatorIsolate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_gc(isolate: *mut StatorIsolate) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe { (*isolate).heap.collect() };
    }
}

// ── Context ──────────────────────────────────────────────────────────────────

/// An opaque context handle.
///
/// A context represents an execution environment associated with an
/// [`StatorIsolate`].  The caller must pass the returned pointer to
/// [`stator_context_destroy`] when the context is no longer needed.
pub struct StatorContext {
    _isolate: *mut StatorIsolate,
}

// SAFETY: `StatorContext` only holds a pointer that is valid for the lifetime
// of the parent `StatorIsolate`; access is single-threaded.
unsafe impl Send for StatorContext {}

/// Create a new context associated with `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_context_destroy`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_new(isolate: *mut StatorIsolate) -> *mut StatorContext {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(StatorContext { _isolate: isolate }))
}

/// Destroy a context previously created with [`stator_context_new`].
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `stator_context_new` and must
/// not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_destroy(ctx: *mut StatorContext) {
    if !ctx.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in `stator_context_new`.
        drop(unsafe { Box::from_raw(ctx) });
    }
}

// ── Value ─────────────────────────────────────────────────────────────────────

/// Internal storage for a [`StatorValue`].
enum StatorValueInner {
    /// A double-precision floating-point number.
    Number(f64),
    /// A UTF-8 string stored as a null-terminated C string for easy FFI access.
    Str(CString),
}

/// An opaque handle to a JavaScript value (number or string).
///
/// Created by [`stator_value_new_number`] or [`stator_value_new_string`] and
/// destroyed by [`stator_value_destroy`].  The live-object counter on the
/// parent isolate is incremented on creation and decremented on destruction.
pub struct StatorValue {
    inner: StatorValueInner,
    isolate: *mut StatorIsolate,
}

// SAFETY: `StatorValue` is single-threaded; the raw pointer to `StatorIsolate`
// is only ever accessed while the isolate is alive.
unsafe impl Send for StatorValue {}

/// Create a new number value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_number(
    isolate: *mut StatorIsolate,
    val: f64,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Number(val),
        isolate,
    }))
}

/// Create a new string value from a buffer of `len` bytes.
///
/// `data` need not be null-terminated; `len` bytes are copied.  Any embedded
/// null bytes are replaced by truncating at the first null.  Returns a null
/// pointer if `isolate` or `data` is null.
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `data` must be valid for reads of `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_string(
    isolate: *mut StatorIsolate,
    data: *const c_char,
    len: usize,
) -> *mut StatorValue {
    if isolate.is_null() || data.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `data` is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
    // Truncate at the first embedded null byte, if any.
    let valid_len = bytes.iter().position(|&b| b == 0).unwrap_or(len);
    // SAFETY: `bytes[..valid_len]` contains no null bytes by construction.
    let cstring = unsafe { CString::from_vec_unchecked(bytes[..valid_len].to_vec()) };
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Str(cstring),
        isolate,
    }))
}

/// Destroy a value and decrement the isolate's live-object counter.
///
/// # Safety
/// `val` must be a non-null pointer returned by `stator_value_new_number` or
/// `stator_value_new_string` and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_destroy(val: *mut StatorValue) {
    if !val.is_null() {
        // SAFETY: caller guarantees `val` is a valid, live pointer.
        let iso = unsafe { (*val).isolate };
        if !iso.is_null() {
            // SAFETY: `iso` is guaranteed valid for the lifetime of `val`.
            unsafe { (*iso).live_objects = (*iso).live_objects.saturating_sub(1) };
        }
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(val) });
    }
}

/// Return a static C string describing the type of `val`: `"number"` or
/// `"string"`.
///
/// Returns `"undefined"` when `val` is null.  The returned pointer is valid
/// for the lifetime of the process.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_type(val: *const StatorValue) -> *const c_char {
    if val.is_null() {
        return c"undefined".as_ptr();
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Number(_) => c"number".as_ptr(),
        StatorValueInner::Str(_) => c"string".as_ptr(),
    }
}

/// Return the numeric value stored in `val`, or `NaN` if `val` is null or not
/// a number.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_as_number(val: *const StatorValue) -> f64 {
    if val.is_null() {
        return f64::NAN;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Number(n) => *n,
        StatorValueInner::Str(_) => f64::NAN,
    }
}

/// Return a null-terminated C string for the string stored in `val`.
///
/// Returns a pointer to an empty string (`""`) when `val` is null or not a
/// string.  The pointer is valid only as long as `val` is alive.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_as_string(val: *const StatorValue) -> *const c_char {
    if val.is_null() {
        return c"".as_ptr();
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Str(cs) => cs.as_ptr(),
        StatorValueInner::Number(_) => c"".as_ptr(),
    }
}

// ── Object ────────────────────────────────────────────────────────────────────

/// An opaque handle to a JavaScript object.
///
/// Created by [`stator_object_new`] and destroyed by [`stator_object_destroy`].
/// Named properties can be set and retrieved via [`stator_object_set`] and
/// [`stator_object_get`].
pub struct StatorObject {
    inner: JsObject,
    isolate: *mut StatorIsolate,
}

// SAFETY: `StatorObject` is single-threaded; see [`StatorValue`] rationale.
unsafe impl Send for StatorObject {}

/// Create a new, empty JavaScript object.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_new(isolate: *mut StatorIsolate) -> *mut StatorObject {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorObject {
        inner: JsObject::new(),
        isolate,
    }))
}

/// Destroy an object and decrement the isolate's live-object counter.
///
/// # Safety
/// `obj` must be a non-null pointer returned by `stator_object_new` and must
/// not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_destroy(obj: *mut StatorObject) {
    if !obj.is_null() {
        // SAFETY: caller guarantees `obj` is a valid, live pointer.
        let iso = unsafe { (*obj).isolate };
        if !iso.is_null() {
            // SAFETY: `iso` is guaranteed valid for the lifetime of `obj`.
            unsafe { (*iso).live_objects = (*iso).live_objects.saturating_sub(1) };
        }
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(obj) });
    }
}

/// Set (or overwrite) the named property `key` on `obj` to `val`.
///
/// The value is copied by value into the object; the caller retains ownership
/// of `val` and must destroy it independently.  Does nothing if any argument
/// is null.
///
/// # Safety
/// - `obj` must be a valid, live [`StatorObject`] pointer.
/// - `key` must be a valid, null-terminated C string.
/// - `val` must be a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_set(
    obj: *mut StatorObject,
    key: *const c_char,
    val: *const StatorValue,
) {
    if obj.is_null() || key.is_null() || val.is_null() {
        return;
    }
    // SAFETY: caller guarantees `key` is a valid null-terminated string.
    let key_str = unsafe { CStr::from_ptr(key) }.to_string_lossy();
    // SAFETY: caller guarantees `val` is valid.
    let js_val = match unsafe { &(*val).inner } {
        StatorValueInner::Number(n) => JsValue::HeapNumber(*n),
        StatorValueInner::Str(cs) => JsValue::String(cs.to_string_lossy().into_owned()),
    };
    // SAFETY: caller guarantees `obj` is valid.
    let _ = unsafe { (*obj).inner.set_property(&key_str, js_val) };
}

/// Get the named property `key` from `obj` as a new [`StatorValue`].
///
/// Returns a null pointer if `obj` or `key` is null, or if the property does
/// not exist.  The caller owns the returned pointer and must pass it to
/// [`stator_value_destroy`].
///
/// # Safety
/// - `obj` must be a valid, live [`StatorObject`] pointer.
/// - `key` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_get(
    obj: *const StatorObject,
    key: *const c_char,
) -> *mut StatorValue {
    if obj.is_null() || key.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `key` is a valid null-terminated string.
    let key_str = unsafe { CStr::from_ptr(key) }.to_string_lossy();
    // SAFETY: caller guarantees `obj` is valid.
    let js_val = unsafe { (*obj).inner.get_property(&key_str) };
    let inner = match js_val {
        JsValue::HeapNumber(n) => StatorValueInner::Number(n),
        JsValue::Smi(n) => StatorValueInner::Number(f64::from(n)),
        JsValue::String(s) => {
            // SAFETY: `s` contains no null bytes iff it came through our API;
            // we truncate at the first null to be safe.
            let valid_len = s.as_bytes().iter().position(|&b| b == 0).unwrap_or(s.len());
            // SAFETY: `&s.as_bytes()[..valid_len]` has no null bytes.
            unsafe {
                StatorValueInner::Str(CString::from_vec_unchecked(
                    s.as_bytes()[..valid_len].to_vec(),
                ))
            }
        }
        // Non-existent or non-primitive property → return null.
        _ => return std::ptr::null_mut(),
    };
    // SAFETY: caller guarantees `obj` is valid, so `isolate` is valid too.
    let isolate = unsafe { (*obj).isolate };
    if !isolate.is_null() {
        // SAFETY: `isolate` is valid for the lifetime of `obj`.
        unsafe { (*isolate).live_objects += 1 };
    }
    Box::into_raw(Box::new(StatorValue { inner, isolate }))
}

// ── GC / heap stats ───────────────────────────────────────────────────────────

/// Trigger a minor (young-generation) GC on the isolate heap.
///
/// This is the preferred spelling for embedders using the Phase 1 object
/// model; it is equivalent to [`stator_isolate_gc`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_gc_collect(isolate: *mut StatorIsolate) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe { (*isolate).heap.collect() };
    }
}

/// Return the number of live embedder-owned handles (values + objects) that
/// have been created on `isolate` but not yet destroyed.
///
/// After a GC cycle this count reflects how many objects would survive because
/// they are still reachable through live handles.
///
/// Returns 0 when `isolate` is null.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_live_object_count(isolate: *const StatorIsolate) -> usize {
    if isolate.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects }
}

/// Return the number of bytes currently allocated in the young-generation
/// from-space.  This is 0 immediately after a GC cycle.
///
/// Returns 0 when `isolate` is null.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_heap_used(isolate: *const StatorIsolate) -> usize {
    if isolate.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).heap.young_space.used() }
}

/// Return the total capacity (in bytes) of the young-generation semi-space,
/// counting both the from-space and to-space halves.
///
/// Returns 0 when `isolate` is null.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_heap_capacity(isolate: *const StatorIsolate) -> usize {
    if isolate.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    // The young generation uses a two-space (Cheney) copying design where
    // `capacity()` returns the size of one half (from-space). Multiply by 2
    // to report the total young-space footprint (from-space + to-space).
    unsafe { (*isolate).heap.young_space.capacity() * 2 }
}

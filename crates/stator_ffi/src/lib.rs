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

use std::ffi::{CStr, CString, c_char, c_void};
use std::io::Write as _;

use stator_core::bytecode::bytecode_array::BytecodeArray;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::bytecode::bytecodes::{Operand, decode};
use stator_core::gc::heap::Heap;
use stator_core::objects::js_object::JsObject;
use stator_core::objects::value::JsValue;
use stator_core::parser;

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
    /// Per-slot embedder data.  Grows on demand; slots not yet set hold null.
    /// The pointers are owned by the embedder and are not freed by the isolate.
    embedder_data: Vec<*mut c_void>,
    /// Number of times the isolate has been entered without a matching exit.
    enter_count: u32,
    /// The context most recently set as current via [`stator_context_new`] or
    /// cleared via [`stator_context_destroy`].  Non-owning; the context is
    /// owned by the embedder and must outlive the isolate or be destroyed first.
    current_context: *mut StatorContext,
    /// Pending exception stored by [`stator_isolate_throw_exception`].
    /// Non-owning; the value is owned by the embedder.  `None` means no
    /// pending exception is set.
    pending_exception: Option<*mut StatorValue>,
}

// SAFETY: `StatorIsolate` contains raw pointer fields that are only ever
// accessed on the owning thread; the embedder is responsible for external
// synchronisation if the isolate is passed across threads.
unsafe impl Send for StatorIsolate {}

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
        embedder_data: Vec::new(),
        enter_count: 0,
        current_context: std::ptr::null_mut(),
        pending_exception: None,
    }))
}

/// Create a new isolate.
///
/// This is the preferred spelling for the V8-compatible lifecycle API.
/// Equivalent to [`stator_isolate_create`].  The returned pointer must
/// eventually be passed to [`stator_isolate_dispose`].
#[unsafe(no_mangle)]
pub extern "C" fn stator_isolate_new() -> *mut StatorIsolate {
    stator_isolate_create()
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

/// Dispose an isolate previously created with [`stator_isolate_new`].
///
/// This is the preferred spelling for the V8-compatible lifecycle API.
/// Equivalent to [`stator_isolate_destroy`].
///
/// # Safety
/// - `isolate` must be a non-null pointer returned by `stator_isolate_new`.
/// - `isolate` must not be used again after this call.
/// - This function must not be called more than once for the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_dispose(isolate: *mut StatorIsolate) {
    // SAFETY: same requirements as `stator_isolate_destroy`.
    unsafe { stator_isolate_destroy(isolate) };
}

/// Mark `isolate` as entered on the current thread.
///
/// Each call to `stator_isolate_enter` must be balanced by a corresponding
/// call to [`stator_isolate_exit`].  Does nothing when `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_enter(isolate: *mut StatorIsolate) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe { (*isolate).enter_count = (*isolate).enter_count.saturating_add(1) };
    }
}

/// Unmark `isolate` as entered on the current thread.
///
/// Must be called once for every preceding [`stator_isolate_enter`] call.
/// Does nothing when `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_exit(isolate: *mut StatorIsolate) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe { (*isolate).enter_count = (*isolate).enter_count.saturating_sub(1) };
    }
}

/// Store an opaque embedder-defined pointer at `slot` on the isolate.
///
/// Slots are zero-indexed.  Storing `NULL` at a slot is permitted and
/// equivalent to clearing it.  Does nothing when `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_set_data(
    isolate: *mut StatorIsolate,
    slot: u32,
    data: *mut c_void,
) {
    if isolate.is_null() {
        return;
    }
    let slot = slot as usize;
    // SAFETY: caller guarantees `isolate` is valid.
    let slots = unsafe { &mut (*isolate).embedder_data };
    if slot >= slots.len() {
        slots.resize(slot + 1, std::ptr::null_mut());
    }
    slots[slot] = data;
}

/// Retrieve the embedder-defined pointer previously stored at `slot`.
///
/// Returns a null pointer when `isolate` is null or `slot` has not been set.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_get_data(
    isolate: *const StatorIsolate,
    slot: u32,
) -> *mut c_void {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let slot = slot as usize;
    // SAFETY: caller guarantees `isolate` is valid.
    let slots = unsafe { &(*isolate).embedder_data };
    if slot < slots.len() {
        slots[slot]
    } else {
        std::ptr::null_mut()
    }
}

/// Record `exception` as the pending exception on `isolate`.
///
/// At most one pending exception is stored at a time; a subsequent call
/// replaces the previous value.  Does nothing when `isolate` is null.
///
/// The caller retains ownership of `exception`; the isolate only holds a
/// raw pointer and does not free it.
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `exception` must be either null or a valid, live [`StatorValue`] pointer
///   that outlives the pending-exception window.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_throw_exception(
    isolate: *mut StatorIsolate,
    exception: *mut StatorValue,
) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe {
            (*isolate).pending_exception = if exception.is_null() {
                None
            } else {
                Some(exception)
            }
        };
    }
}

/// Return the context most recently made current on `isolate`.
///
/// Returns a null pointer when `isolate` is null or no context has been made
/// current (i.e. no context has been created on this isolate yet, or the
/// most recent one was destroyed).
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_get_current_context(
    isolate: *const StatorIsolate,
) -> *mut StatorContext {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).current_context }
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
    /// Number of times the context has been entered without a matching exit.
    enter_count: u32,
    /// The global object for this context.  Owned by the context; embedders
    /// receive a non-owning pointer via [`stator_context_global`] and must
    /// **not** call [`stator_object_destroy`] on it.
    global: StatorObject,
}

// SAFETY: `StatorContext` only holds a pointer that is valid for the lifetime
// of the parent `StatorIsolate`; access is single-threaded.
unsafe impl Send for StatorContext {}

/// Create a new context associated with `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_context_destroy`].
///
/// Creating a context automatically makes it the current context on `isolate`
/// (equivalent to entering it).  Destroying the context via
/// [`stator_context_destroy`] clears the current-context slot when it matches.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_new(isolate: *mut StatorIsolate) -> *mut StatorContext {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = Box::into_raw(Box::new(StatorContext {
        _isolate: isolate,
        enter_count: 0,
        global: StatorObject {
            inner: JsObject::new(),
            isolate,
        },
    }));
    // SAFETY: caller guarantees `isolate` is valid; `ctx` was just created.
    unsafe { (*isolate).current_context = ctx };
    ctx
}

/// Destroy a context previously created with [`stator_context_new`].
///
/// If `ctx` is the current context on its associated isolate, the
/// current-context slot is cleared (set to null).
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `stator_context_new` and must
/// not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_destroy(ctx: *mut StatorContext) {
    if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        let isolate = unsafe { (*ctx)._isolate };
        if !isolate.is_null() {
            // SAFETY: `isolate` is valid for the lifetime of any context
            // created from it; the caller must not have already destroyed it.
            unsafe {
                if (*isolate).current_context == ctx {
                    (*isolate).current_context = std::ptr::null_mut();
                }
            }
        }
        // SAFETY: pointer was created by `Box::into_raw` in `stator_context_new`.
        drop(unsafe { Box::from_raw(ctx) });
    }
}

/// Mark `ctx` as entered on the current thread.
///
/// Each call to `stator_context_enter` must be balanced by a corresponding
/// call to [`stator_context_exit`].  Entering a context also makes it the
/// current context on its associated isolate.  Does nothing when `ctx` is null.
///
/// # Safety
/// `ctx` must be either null or a valid, live [`StatorContext`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_enter(ctx: *mut StatorContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    unsafe {
        (*ctx).enter_count = (*ctx).enter_count.saturating_add(1);
        let isolate = (*ctx)._isolate;
        if !isolate.is_null() {
            (*isolate).current_context = ctx;
        }
    }
}

/// Unmark `ctx` as entered on the current thread.
///
/// Must be called once for every preceding [`stator_context_enter`] call.
/// When the enter count reaches zero the context is no longer recorded as
/// current on its associated isolate.  Does nothing when `ctx` is null.
///
/// # Safety
/// `ctx` must be either null or a valid, live [`StatorContext`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_exit(ctx: *mut StatorContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    unsafe {
        (*ctx).enter_count = (*ctx).enter_count.saturating_sub(1);
        if (*ctx).enter_count == 0 {
            let isolate = (*ctx)._isolate;
            if !isolate.is_null() && (*isolate).current_context == ctx {
                (*isolate).current_context = std::ptr::null_mut();
            }
        }
    }
}

/// Return a non-owning pointer to the global object of `ctx`.
///
/// The returned pointer is valid for as long as `ctx` is alive.  **The caller
/// must not pass the returned pointer to [`stator_object_destroy`]**; the
/// global object is owned by the context and is freed when the context is
/// destroyed via [`stator_context_destroy`].
///
/// Returns a null pointer when `ctx` is null.
///
/// # Safety
/// `ctx` must be either null or a valid, live [`StatorContext`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_global(ctx: *mut StatorContext) -> *mut StatorObject {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `ctx` is valid; we return a pointer to the
    // field inside the allocation, which is valid as long as `ctx` is alive.
    unsafe { &raw mut (*ctx).global }
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

// ── Script (Phase 2: parsing + compilation) ───────────────────────────────────

/// The outcome of a `stator_script_compile` call.
///
/// Created by [`stator_script_compile`] and released by [`stator_script_free`].
pub struct StatorScript {
    /// Compiled bytecodes on success; `None` on parse / compile error.
    bytecodes: Option<BytecodeArray>,
    /// Human-readable error message, or `None` on success.
    error: Option<CString>,
}

/// Compile `source` (a UTF-8 string of `source_len` bytes) into bytecode.
///
/// Returns a non-null [`StatorScript`] pointer in all cases (even on error).
/// Call [`stator_script_get_error`] to check whether compilation succeeded.
/// The caller must eventually pass the returned pointer to
/// [`stator_script_free`].
///
/// Returns a null pointer only on an internal allocation failure (extremely
/// unlikely in practice).
///
/// # Safety
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - `source` must be valid for reads of `source_len` bytes of valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_compile(
    _ctx: *mut StatorContext,
    source: *const c_char,
    source_len: usize,
) -> *mut StatorScript {
    if source.is_null() {
        let script = Box::new(StatorScript {
            bytecodes: None,
            error: Some(c"null source pointer".into()),
        });
        return Box::into_raw(script);
    }

    // SAFETY: caller guarantees `source` is valid for `source_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, source_len) };
    let src = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            let script = Box::new(StatorScript {
                bytecodes: None,
                error: Some(c"source is not valid UTF-8".into()),
            });
            return Box::into_raw(script);
        }
    };

    // Parse then compile.
    let result =
        parser::parse(src).and_then(|program| BytecodeGenerator::compile_program(&program));

    let script = match result {
        Ok(bytecodes) => Box::new(StatorScript {
            bytecodes: Some(bytecodes),
            error: None,
        }),
        Err(e) => {
            let msg = e.to_string();
            let cstring = CString::new(msg).unwrap_or_else(|_| c"compilation error".into());
            Box::new(StatorScript {
                bytecodes: None,
                error: Some(cstring),
            })
        }
    };
    Box::into_raw(script)
}

/// Return a null-terminated error message if `script` compiled with an error.
///
/// Returns a null pointer when `script` compiled successfully.  The returned
/// pointer is valid as long as `script` is alive (i.e. until
/// [`stator_script_free`] is called).
///
/// # Safety
/// `script` must be a non-null pointer returned by [`stator_script_compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_get_error(script: *const StatorScript) -> *const c_char {
    if script.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `script` is valid.
    match unsafe { &(*script).error } {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Print the decoded bytecode listing for `script` to standard output.
///
/// Each instruction is printed on a new line in the form:
/// `  <OpcodeName> [<operand>, …]`
///
/// Does nothing if `script` is null or compiled with an error.
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_bytecode_dump(script: *const StatorScript) {
    if script.is_null() {
        return;
    }
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return,
    };
    match decode(bytecodes.bytecodes()) {
        Ok(instructions) => {
            for instr in &instructions {
                let operand_strs: Vec<String> = instr
                    .operands
                    .iter()
                    .map(|op| format_operand(*op))
                    .collect();
                if operand_strs.is_empty() {
                    println!("  {:?}", instr.opcode);
                } else {
                    println!("  {:?} {}", instr.opcode, operand_strs.join(", "));
                }
            }
        }
        Err(e) => {
            println!("  <decode error: {e}>");
        }
    }
    // Flush Rust's stdout immediately so output appears in-order relative to
    // the surrounding C stdio output.  When stdout is a pipe (e.g. captured
    // by a shell), both Rust and C maintain separate internal buffers for the
    // same fd.  Without an explicit flush here the Rust lines would not reach
    // fd 1 until Rust's runtime teardown, which happens before C's atexit
    // handlers — causing the bytecodes to appear before all C printf output.
    let _ = std::io::stdout().flush();
}

/// Format a single [`Operand`] for human-readable bytecode listing.
///
/// Conventions:
/// - `Register(r)`: non-negative values become `r{n}` (local/temp registers);
///   negative values stored as two's-complement `u32` become `a{n}` (argument
///   registers, where `n = !r`).
/// - `Immediate(v)`: printed as a signed integer.
/// - `ConstantPoolIdx(i)`: printed as `[{i}]`.
/// - `FeedbackSlot(s)`: printed as `slot({s})`.
/// - All other variants use a descriptive prefix followed by the raw value.
fn format_operand(op: Operand) -> String {
    match op {
        Operand::Register(r) => {
            // Negative register indices (stored as two's-complement u32)
            // represent parameter registers.
            let signed = r as i32;
            if signed < 0 {
                format!("a{}", !signed)
            } else {
                format!("r{r}")
            }
        }
        Operand::RegisterCount(n) => format!("count({n})"),
        Operand::Immediate(v) => format!("{v}"),
        Operand::ConstantPoolIdx(i) => format!("[{i}]"),
        Operand::FeedbackSlot(s) => format!("slot({s})"),
        Operand::RuntimeId(id) => format!("rt({id})"),
        Operand::JumpOffset(o) => format!("{o:+}"),
        Operand::Flag(f) => format!("#{f}"),
    }
}

/// Return the number of bytecode instructions in a successfully compiled script.
///
/// Returns 0 if `script` is null, compiled with an error, or the bytecode
/// stream cannot be decoded.
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_bytecode_count(script: *const StatorScript) -> usize {
    if script.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return 0,
    };
    decode(bytecodes.bytecodes())
        .map(|instrs| instrs.len())
        .unwrap_or(0)
}

/// Free a [`StatorScript`] previously returned by [`stator_script_compile`].
///
/// # Safety
/// `script` must be a non-null pointer returned by [`stator_script_compile`]
/// and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_free(script: *mut StatorScript) {
    if !script.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in
        // `stator_script_compile`.
        drop(unsafe { Box::from_raw(script) });
    }
}

// ── Platform vtable ───────────────────────────────────────────────────────────

/// C-compatible vtable that embedders fill in to customise engine behaviour.
///
/// All fields are optional (`Option<…>`).  When a field is `None` the engine
/// falls back to a default (no-op or zero).
///
/// # Safety requirements for callers
/// Every function pointer, if set, must remain valid for the entire lifetime of
/// the owning [`StatorPlatform`].  Calling [`stator_platform_destroy`] on the
/// platform is the only safe point at which the function pointers may be
/// invalidated.
#[repr(C)]
pub struct StatorPlatformVTable {
    /// Return the number of worker threads available for background work.
    ///
    /// May be `None`; in that case the engine defaults to `0`.
    pub number_of_worker_threads: Option<unsafe extern "C" fn() -> u32>,

    /// Schedule `task` for execution on a platform thread.
    ///
    /// The platform takes ownership of `task`.  May be `None` (tasks are
    /// silently dropped).
    pub post_task: Option<unsafe extern "C" fn(task: *mut c_void)>,

    /// Return a monotonically increasing time in seconds.
    ///
    /// May be `None`; the engine returns `0.0` when unset.
    pub monotonically_increasing_time: Option<unsafe extern "C" fn() -> f64>,

    /// Return the current wall-clock time in milliseconds since the Unix epoch.
    ///
    /// May be `None`; the engine returns `0.0` when unset.
    pub current_clock_time_millis: Option<unsafe extern "C" fn() -> f64>,
}

/// Rust-side wrapper around a [`StatorPlatformVTable`] that implements the
/// [`stator_core::platform::Platform`] trait.
struct VTablePlatformImpl {
    vtable: StatorPlatformVTable,
}

// SAFETY: The vtable function pointers are inherently thread-safe (they are
// stateless C function pointers, not closures); the embedder is responsible
// for ensuring any global state they access is thread-safe.
unsafe impl Send for VTablePlatformImpl {}

impl stator_core::platform::Platform for VTablePlatformImpl {
    fn number_of_worker_threads(&self) -> u32 {
        // SAFETY: The vtable pointer is valid for the lifetime of the platform.
        self.vtable
            .number_of_worker_threads
            .map_or(0, |f| unsafe { f() })
    }

    unsafe fn post_task(&self, task: *mut c_void) {
        if let Some(f) = self.vtable.post_task {
            // SAFETY: caller upholds the `post_task` contract; the vtable
            // pointer is valid for the lifetime of the platform.
            unsafe { f(task) };
        }
    }

    fn monotonically_increasing_time(&self) -> f64 {
        // SAFETY: The vtable pointer is valid for the lifetime of the platform.
        self.vtable
            .monotonically_increasing_time
            .map_or(0.0, |f| unsafe { f() })
    }

    fn current_clock_time_millis(&self) -> f64 {
        // SAFETY: The vtable pointer is valid for the lifetime of the platform.
        self.vtable
            .current_clock_time_millis
            .map_or(0.0, |f| unsafe { f() })
    }
}

/// An opaque handle to an embedder-provided platform implementation.
///
/// Created by [`stator_platform_new`] and destroyed by
/// [`stator_platform_destroy`].
pub struct StatorPlatform {
    inner: Box<dyn stator_core::platform::Platform>,
}

/// Create a new platform from an embedder-supplied vtable.
///
/// The vtable is copied by value; the caller does not need to keep the
/// original `StatorPlatformVTable` alive after this call returns.
///
/// Returns a null pointer if `vtable` is null.  The caller must eventually
/// pass the returned pointer to [`stator_platform_destroy`].
///
/// # Safety
/// - `vtable` must be either null or a valid, readable pointer to a
///   [`StatorPlatformVTable`].
/// - All non-null function pointers stored in the vtable must remain callable
///   for the entire lifetime of the returned [`StatorPlatform`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_new(
    vtable: *const StatorPlatformVTable,
) -> *mut StatorPlatform {
    if vtable.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `vtable` is a valid, readable pointer.
    let copied = unsafe {
        StatorPlatformVTable {
            number_of_worker_threads: (*vtable).number_of_worker_threads,
            post_task: (*vtable).post_task,
            monotonically_increasing_time: (*vtable).monotonically_increasing_time,
            current_clock_time_millis: (*vtable).current_clock_time_millis,
        }
    };
    Box::into_raw(Box::new(StatorPlatform {
        inner: Box::new(VTablePlatformImpl { vtable: copied }),
    }))
}

/// Destroy a platform previously created with [`stator_platform_new`].
///
/// After this call the pointer is invalid and must not be used.
///
/// # Safety
/// - `platform` must be a non-null pointer returned by `stator_platform_new`.
/// - `platform` must not be used again after this call.
/// - This function must not be called more than once for the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_destroy(platform: *mut StatorPlatform) {
    if !platform.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in `stator_platform_new`.
        drop(unsafe { Box::from_raw(platform) });
    }
}

/// Return the number of worker threads reported by `platform`.
///
/// Returns `0` when `platform` is null or the vtable entry was not set.
///
/// # Safety
/// `platform` must be either null or a valid, live [`StatorPlatform`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_number_of_worker_threads(
    platform: *const StatorPlatform,
) -> u32 {
    if platform.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `platform` is valid.
    unsafe { (*platform).inner.number_of_worker_threads() }
}

/// Schedule `task` for execution on `platform`.
///
/// Does nothing when `platform` is null, `task` is null, or the vtable entry
/// was not set.
///
/// # Safety
/// - `platform` must be either null or a valid, live [`StatorPlatform`] pointer.
/// - `task` must be a non-null pointer whose lifetime the caller manages;
///   ownership of `task` is transferred to the platform on this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_post_task(
    platform: *const StatorPlatform,
    task: *mut c_void,
) {
    if platform.is_null() || task.is_null() {
        return;
    }
    // SAFETY: caller guarantees both pointers are valid.
    unsafe { (*platform).inner.post_task(task) };
}

/// Return the monotonically increasing time (seconds) from `platform`.
///
/// Returns `0.0` when `platform` is null or the vtable entry was not set.
///
/// # Safety
/// `platform` must be either null or a valid, live [`StatorPlatform`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_monotonically_increasing_time(
    platform: *const StatorPlatform,
) -> f64 {
    if platform.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees `platform` is valid.
    unsafe { (*platform).inner.monotonically_increasing_time() }
}

/// Return the current clock time in milliseconds from `platform`.
///
/// Returns `0.0` when `platform` is null or the vtable entry was not set.
///
/// # Safety
/// `platform` must be either null or a valid, live [`StatorPlatform`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_platform_current_clock_time_millis(
    platform: *const StatorPlatform,
) -> f64 {
    if platform.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees `platform` is valid.
    unsafe { (*platform).inner.current_clock_time_millis() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// Helper: create an isolate and automatically destroy it after the test.
    struct IsolateGuard(*mut StatorIsolate);

    impl IsolateGuard {
        fn new() -> Self {
            Self(stator_isolate_create())
        }

        fn as_ptr(&self) -> *mut StatorIsolate {
            self.0
        }
    }

    impl Drop for IsolateGuard {
        fn drop(&mut self) {
            // SAFETY: pointer was created by `stator_isolate_create` in this guard.
            unsafe { stator_isolate_destroy(self.0) };
        }
    }

    #[test]
    fn test_isolate_create_returns_nonnull() {
        let iso = IsolateGuard::new();
        assert!(!iso.as_ptr().is_null());
    }

    #[test]
    fn test_isolate_destroy_null_is_safe() {
        // SAFETY: passing null is explicitly documented as a no-op.
        unsafe { stator_isolate_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_value_new_number_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is non-null and live.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 3.14) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let n = unsafe { stator_value_as_number(val) };
        assert!((n - 3.14).abs() < f64::EPSILON);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_new_string_roundtrip() {
        let iso = IsolateGuard::new();
        let s = b"hello\0";
        // SAFETY: `iso` is valid; `s` pointer is valid for 5 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 5) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let ptr = unsafe { stator_value_as_string(val) };
        // SAFETY: returned pointer is valid while `val` is alive.
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "hello");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_type_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: `val` is non-null and live.
        let type_ptr = unsafe { stator_value_type(val) };
        // SAFETY: returned pointer is static.
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "number");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_type_string() {
        let iso = IsolateGuard::new();
        let s = b"x\0";
        // SAFETY: `iso` is valid; `s` pointer is valid for 1 byte.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 1) };
        // SAFETY: `val` is non-null and live.
        let type_ptr = unsafe { stator_value_type(val) };
        // SAFETY: returned pointer is static.
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "string");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_type_null_returns_undefined() {
        // SAFETY: null is explicitly documented to return "undefined".
        let type_ptr = unsafe { stator_value_type(std::ptr::null()) };
        // SAFETY: returned pointer is static.
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "undefined");
    }

    #[test]
    fn test_value_as_number_on_string_returns_nan() {
        let iso = IsolateGuard::new();
        let s = b"hello\0";
        // SAFETY: `iso` is valid; `s` pointer is valid for 5 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 5) };
        // SAFETY: `val` is non-null and live.
        let n = unsafe { stator_value_as_number(val) };
        assert!(n.is_nan());
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_new_number_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let val = unsafe { stator_value_new_number(std::ptr::null_mut(), 1.0) };
        assert!(val.is_null());
    }

    #[test]
    fn test_live_object_count_tracks_values() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
        // SAFETY: `iso` is valid.
        let v1 = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);
        // SAFETY: `iso` is valid.
        let v2 = unsafe { stator_value_new_number(iso.as_ptr(), 2.0) };
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 2);
        // SAFETY: `v1` is non-null and live.
        unsafe { stator_value_destroy(v1) };
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);
        // SAFETY: `v2` is non-null and live.
        unsafe { stator_value_destroy(v2) };
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    #[test]
    fn test_object_set_and_get_number_property() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        assert!(!obj.is_null());
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        let key = c"answer";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_set(obj, key.as_ptr(), val) };
        // SAFETY: `obj` and `key` are valid.
        let got = unsafe { stator_object_get(obj, key.as_ptr()) };
        assert!(!got.is_null());
        // SAFETY: `got` is non-null and live.
        let n = unsafe { stator_value_as_number(got) };
        assert!((n - 42.0).abs() < f64::EPSILON);
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(val);
            stator_value_destroy(got);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_object_get_missing_property_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let key = c"missing";
        // SAFETY: `obj` and `key` are valid.
        let got = unsafe { stator_object_get(obj, key.as_ptr()) };
        assert!(got.is_null());
        // SAFETY: `obj` is non-null and live.
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_context_create_and_destroy() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let ctx = unsafe { stator_context_new(std::ptr::null_mut()) };
        assert!(ctx.is_null());
    }

    #[test]
    fn test_heap_capacity_nonzero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let cap = unsafe { stator_heap_capacity(iso.as_ptr()) };
        assert!(cap > 0);
    }

    #[test]
    fn test_gc_collect_does_not_crash() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_gc(iso.as_ptr()) };
        // SAFETY: `iso` is valid.
        unsafe { stator_gc_collect(iso.as_ptr()) };
    }

    // ── Isolate lifecycle (Phase 4) ───────────────────────────────────────────

    #[test]
    fn test_isolate_new_returns_nonnull() {
        let iso = stator_isolate_new();
        assert!(!iso.is_null());
        // SAFETY: `iso` was just returned by `stator_isolate_new`.
        unsafe { stator_isolate_dispose(iso) };
    }

    #[test]
    fn test_isolate_dispose_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_isolate_dispose(std::ptr::null_mut()) };
    }

    #[test]
    fn test_isolate_enter_exit_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_enter(iso.as_ptr()) };
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_exit(iso.as_ptr()) };
        // No assertion needed — the test passes if it does not panic or crash.
    }

    #[test]
    fn test_isolate_enter_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_isolate_enter(std::ptr::null_mut()) };
    }

    #[test]
    fn test_isolate_exit_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_isolate_exit(std::ptr::null_mut()) };
    }

    #[test]
    fn test_isolate_exit_without_enter_does_not_underflow() {
        let iso = IsolateGuard::new();
        // Calling exit without a matching enter must not panic or underflow.
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_exit(iso.as_ptr()) };
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_exit(iso.as_ptr()) };
    }

    #[test]
    fn test_isolate_set_get_data_roundtrip() {
        let iso = IsolateGuard::new();
        let sentinel = 0xDEAD_BEEFusize as *mut c_void;
        // SAFETY: `iso` is valid; `sentinel` is a valid (if fake) data pointer.
        unsafe { stator_isolate_set_data(iso.as_ptr(), 0, sentinel) };
        // SAFETY: `iso` is valid.
        let got = unsafe { stator_isolate_get_data(iso.as_ptr(), 0) };
        assert_eq!(got, sentinel);
    }

    #[test]
    fn test_isolate_set_get_data_multiple_slots() {
        let iso = IsolateGuard::new();
        let a = 0x1111usize as *mut c_void;
        let b = 0x2222usize as *mut c_void;
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_set_data(iso.as_ptr(), 0, a) };
        unsafe { stator_isolate_set_data(iso.as_ptr(), 2, b) };
        // Slot 1 was never set and must be null.
        assert_eq!(unsafe { stator_isolate_get_data(iso.as_ptr(), 0) }, a);
        assert_eq!(
            unsafe { stator_isolate_get_data(iso.as_ptr(), 1) },
            std::ptr::null_mut()
        );
        assert_eq!(unsafe { stator_isolate_get_data(iso.as_ptr(), 2) }, b);
    }

    #[test]
    fn test_isolate_get_data_unset_slot_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid; slot 99 has never been set.
        let got = unsafe { stator_isolate_get_data(iso.as_ptr(), 99) };
        assert!(got.is_null());
    }

    #[test]
    fn test_isolate_get_data_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let got = unsafe { stator_isolate_get_data(std::ptr::null(), 0) };
        assert!(got.is_null());
    }

    #[test]
    fn test_isolate_set_data_null_isolate_is_safe() {
        // SAFETY: null isolate is documented as a no-op.
        unsafe {
            stator_isolate_set_data(std::ptr::null_mut(), 0, 0xDEADusize as *mut c_void);
        }
    }

    #[test]
    fn test_isolate_throw_exception_stores_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        assert!(!val.is_null());
        // SAFETY: `iso` and `val` are valid.
        unsafe { stator_isolate_throw_exception(iso.as_ptr(), val) };
        // Verify the exception is recorded (access through the struct directly
        // in the test module is intentional — tests are in the same crate).
        // SAFETY: `iso` is valid.
        let recorded = unsafe { (*iso.as_ptr()).pending_exception };
        assert_eq!(recorded, Some(val));
        // Clean up.
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_isolate_throw_exception_null_isolate_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_isolate_throw_exception(std::ptr::null_mut(), std::ptr::null_mut()) };
    }

    #[test]
    fn test_isolate_get_current_context_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let ctx = unsafe { stator_isolate_get_current_context(std::ptr::null()) };
        assert!(ctx.is_null());
    }

    #[test]
    fn test_isolate_get_current_context_set_by_context_new() {
        let iso = IsolateGuard::new();
        // Before any context is created the current context must be null.
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso.as_ptr()) }.is_null());

        // Creating a context makes it the current one.
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        // SAFETY: `iso` is valid.
        let current = unsafe { stator_isolate_get_current_context(iso.as_ptr()) };
        assert_eq!(current, ctx);

        // Destroying the context clears the current-context slot.
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso.as_ptr()) }.is_null());
    }

    #[test]
    fn test_full_isolate_lifecycle() {
        // Exercises the full V8-style isolate lifecycle in one test.
        let iso = stator_isolate_new();
        assert!(!iso.is_null());

        // Enter the isolate.
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_enter(iso) };

        // Create a context — it becomes current.
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso) };
        assert!(!ctx.is_null());
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_isolate_get_current_context(iso) }, ctx);

        // Store and retrieve embedder data.
        let tag = 0xCAFEusize as *mut c_void;
        // SAFETY: `iso` is valid; `tag` is a valid data pointer for this test.
        unsafe { stator_isolate_set_data(iso, 0, tag) };
        assert_eq!(unsafe { stator_isolate_get_data(iso, 0) }, tag);

        // Throw an exception value.
        // SAFETY: `iso` is valid.
        let exc = unsafe { stator_value_new_number(iso, -1.0) };
        // SAFETY: `iso` and `exc` are valid.
        unsafe { stator_isolate_throw_exception(iso, exc) };

        // Destroy the context.
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso) }.is_null());

        // Exit the isolate.
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_exit(iso) };

        // Clean up exception value before disposing.
        // SAFETY: `exc` is non-null and live.
        unsafe { stator_value_destroy(exc) };

        // Dispose — must not crash.
        // SAFETY: `iso` is non-null and live.
        unsafe { stator_isolate_dispose(iso) };
    }

    // ── Context enter / exit / global (P4) ───────────────────────────────────

    #[test]
    fn test_context_enter_makes_context_current() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        // After `context_new` the context is already current; exit it so we
        // can verify that `enter` restores it.
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_exit(ctx) };
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso.as_ptr()) }.is_null());
        // Entering the context must make it current again.
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_enter(ctx) };
        // SAFETY: `iso` is valid.
        assert_eq!(
            unsafe { stator_isolate_get_current_context(iso.as_ptr()) },
            ctx
        );
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_exit(ctx) };
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_exit_clears_current_when_count_reaches_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // Enter twice, exit twice — current should be cleared after the last exit.
        // SAFETY: `ctx` is valid.
        unsafe { stator_context_enter(ctx) };
        unsafe { stator_context_enter(ctx) };
        unsafe { stator_context_exit(ctx) };
        // Still entered once; should still be current.
        // SAFETY: `iso` is valid.
        assert_eq!(
            unsafe { stator_isolate_get_current_context(iso.as_ptr()) },
            ctx
        );
        // SAFETY: `ctx` is valid.
        unsafe { stator_context_exit(ctx) };
        // Enter count is now 0; context should no longer be current.
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso.as_ptr()) }.is_null());
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_exit_without_enter_does_not_underflow() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // Extra exits must not panic or underflow.
        // SAFETY: `ctx` is valid.
        unsafe { stator_context_exit(ctx) };
        unsafe { stator_context_exit(ctx) };
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_enter_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_context_enter(std::ptr::null_mut()) };
    }

    #[test]
    fn test_context_exit_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_context_exit(std::ptr::null_mut()) };
    }

    #[test]
    fn test_context_global_returns_nonnull() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        // SAFETY: `ctx` is non-null and live.
        let global = unsafe { stator_context_global(ctx) };
        assert!(!global.is_null());
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_global_null_returns_null() {
        // SAFETY: null context is documented to return null.
        let global = unsafe { stator_context_global(std::ptr::null_mut()) };
        assert!(global.is_null());
    }

    #[test]
    fn test_context_global_set_and_get_property() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // SAFETY: `ctx` is non-null and live.
        let global = unsafe { stator_context_global(ctx) };
        assert!(!global.is_null());

        // Store a property on the global object.
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 99.0) };
        let key = c"answer";
        // SAFETY: `global`, `key`, and `val` are all valid.
        unsafe { stator_object_set(global, key.as_ptr(), val) };

        // Retrieve the property from the global object.
        // SAFETY: `global` and `key` are valid.
        let got = unsafe { stator_object_get(global, key.as_ptr()) };
        assert!(!got.is_null());
        // SAFETY: `got` is non-null and live.
        let n = unsafe { stator_value_as_number(got) };
        assert!((n - 99.0).abs() < f64::EPSILON);

        // Clean up owned handles.
        // SAFETY: both pointers are non-null and live.
        unsafe {
            stator_value_destroy(val);
            stator_value_destroy(got);
        }
        // Do NOT call stator_object_destroy(global) — it is owned by the context.
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_full_context_lifecycle() {
        // create → enter → access global → set property → exit → dispose
        let iso = stator_isolate_new();
        assert!(!iso.is_null());
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_enter(iso) };

        // Create context and enter it.
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso) };
        assert!(!ctx.is_null());
        // SAFETY: `ctx` is valid.
        unsafe { stator_context_enter(ctx) };
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_isolate_get_current_context(iso) }, ctx);

        // Access and use the global object.
        // SAFETY: `ctx` is valid.
        let global = unsafe { stator_context_global(ctx) };
        assert!(!global.is_null());
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso, 7.0) };
        let key = c"lucky";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_set(global, key.as_ptr(), val) };
        // SAFETY: `global` and `key` are valid.
        let got = unsafe { stator_object_get(global, key.as_ptr()) };
        assert!(!got.is_null());
        // SAFETY: `got` is non-null and live.
        assert!((unsafe { stator_value_as_number(got) } - 7.0).abs() < f64::EPSILON);
        // SAFETY: both pointers are non-null and live.
        unsafe {
            stator_value_destroy(val);
            stator_value_destroy(got);
        }

        // Exit and destroy context.
        // SAFETY: `ctx` is valid.
        unsafe { stator_context_exit(ctx) };
        // SAFETY: `iso` is valid.
        assert!(unsafe { stator_isolate_get_current_context(iso) }.is_null());
        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };

        // Exit and dispose isolate.
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_exit(iso) };
        // SAFETY: `iso` is non-null and live.
        unsafe { stator_isolate_dispose(iso) };
    }

    // ── Script / Phase 2 ─────────────────────────────────────────────────────

    /// Helper: compile a source string and return the `StatorScript` pointer.
    ///
    /// The returned pointer must be freed with `stator_script_free`.
    fn compile_src(src: &str) -> *mut StatorScript {
        let bytes = src.as_bytes();
        // SAFETY: null ctx is permitted; `bytes` is valid UTF-8.
        unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                bytes.as_ptr() as *const c_char,
                bytes.len(),
            )
        }
    }

    #[test]
    fn test_script_compile_simple_returns_nonnull() {
        let script = compile_src("var x = 1 + 2;");
        assert!(!script.is_null());
        // SAFETY: `script` was just returned by `stator_script_compile`.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_compile_no_error_on_valid_source() {
        let script = compile_src("var x = 1 + 2;");
        // SAFETY: `script` is non-null and live.
        let err_ptr = unsafe { stator_script_get_error(script) };
        assert!(err_ptr.is_null(), "expected no error");
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_compile_error_on_syntax_error() {
        let script = compile_src("var = ;");
        // SAFETY: `script` is non-null and live.
        let err_ptr = unsafe { stator_script_get_error(script) };
        assert!(!err_ptr.is_null(), "expected an error");
        // SAFETY: returned pointer is valid while `script` is alive.
        let msg = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap();
        assert!(
            msg.contains("SyntaxError"),
            "expected 'SyntaxError' in {msg:?}"
        );
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_bytecode_count_nonzero_on_success() {
        let script = compile_src("var x = 1 + 2;");
        // SAFETY: `script` is non-null and live.
        let count = unsafe { stator_script_bytecode_count(script) };
        assert!(count > 0, "expected bytecodes, got 0");
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_bytecode_count_zero_on_error() {
        let script = compile_src("var = ;");
        // SAFETY: `script` is non-null and live.
        let count = unsafe { stator_script_bytecode_count(script) };
        assert_eq!(count, 0, "expected 0 bytecodes on error");
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_free_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_script_free(std::ptr::null_mut()) };
    }

    #[test]
    fn test_bytecode_dump_does_not_crash() {
        let script = compile_src("var x = 1 + 2;");
        // SAFETY: `script` is non-null and live.
        unsafe { stator_bytecode_dump(script) };
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    // ── Platform vtable ───────────────────────────────────────────────────────

    /// C callback: returns a fixed thread count of 4.
    unsafe extern "C" fn cb_worker_threads() -> u32 {
        4
    }

    /// C callback: returns a fixed monotonic time.
    unsafe extern "C" fn cb_monotonic_time() -> f64 {
        1.5
    }

    /// C callback: returns a fixed clock time.
    unsafe extern "C" fn cb_clock_millis() -> f64 {
        1_000.0
    }

    /// C callback for post_task: stores the task pointer in a global so tests
    /// can verify it was received.
    static LAST_TASK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    unsafe extern "C" fn cb_post_task(task: *mut c_void) {
        LAST_TASK.store(task as usize, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn test_platform_new_returns_nonnull() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: Some(cb_worker_threads),
            post_task: Some(cb_post_task),
            monotonically_increasing_time: Some(cb_monotonic_time),
            current_clock_time_millis: Some(cb_clock_millis),
        };
        // SAFETY: `vtable` is a valid, fully-initialised vtable.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        assert!(!platform.is_null());
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }

    #[test]
    fn test_platform_null_vtable_returns_null() {
        // SAFETY: null vtable is documented to return null.
        let platform = unsafe { stator_platform_new(std::ptr::null()) };
        assert!(platform.is_null());
    }

    #[test]
    fn test_platform_destroy_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_platform_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_platform_number_of_worker_threads_callback() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: Some(cb_worker_threads),
            post_task: None,
            monotonically_increasing_time: None,
            current_clock_time_millis: None,
        };
        // SAFETY: `vtable` is a valid pointer.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        assert!(!platform.is_null());
        // SAFETY: `platform` is non-null and live.
        let threads = unsafe { stator_platform_number_of_worker_threads(platform) };
        assert_eq!(threads, 4);
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }

    #[test]
    fn test_platform_monotonically_increasing_time_callback() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: None,
            post_task: None,
            monotonically_increasing_time: Some(cb_monotonic_time),
            current_clock_time_millis: None,
        };
        // SAFETY: `vtable` is a valid pointer.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        // SAFETY: `platform` is non-null and live.
        let t = unsafe { stator_platform_monotonically_increasing_time(platform) };
        assert!((t - 1.5).abs() < f64::EPSILON);
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }

    #[test]
    fn test_platform_current_clock_time_millis_callback() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: None,
            post_task: None,
            monotonically_increasing_time: None,
            current_clock_time_millis: Some(cb_clock_millis),
        };
        // SAFETY: `vtable` is a valid pointer.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        // SAFETY: `platform` is non-null and live.
        let ms = unsafe { stator_platform_current_clock_time_millis(platform) };
        assert!((ms - 1_000.0).abs() < f64::EPSILON);
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }

    #[test]
    fn test_platform_post_task_callback() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: None,
            post_task: Some(cb_post_task),
            monotonically_increasing_time: None,
            current_clock_time_millis: None,
        };
        // SAFETY: `vtable` is a valid pointer.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        let sentinel = 0xDEAD_BEEFusize as *mut c_void;
        // Reset global before the call.
        LAST_TASK.store(0, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: `platform` is non-null and live; `sentinel` is a valid (if
        // fake) task pointer for the purposes of this test.
        unsafe { stator_platform_post_task(platform, sentinel) };
        assert_eq!(
            LAST_TASK.load(std::sync::atomic::Ordering::SeqCst),
            sentinel as usize,
            "post_task callback should have stored the sentinel pointer"
        );
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }

    #[test]
    fn test_platform_none_callbacks_return_defaults() {
        let vtable = StatorPlatformVTable {
            number_of_worker_threads: None,
            post_task: None,
            monotonically_increasing_time: None,
            current_clock_time_millis: None,
        };
        // SAFETY: `vtable` is a valid pointer.
        let platform = unsafe { stator_platform_new(&raw const vtable) };
        // SAFETY: `platform` is non-null and live.
        assert_eq!(
            unsafe { stator_platform_number_of_worker_threads(platform) },
            0
        );
        assert!(
            (unsafe { stator_platform_monotonically_increasing_time(platform) } - 0.0).abs()
                < f64::EPSILON
        );
        assert!(
            (unsafe { stator_platform_current_clock_time_millis(platform) } - 0.0).abs()
                < f64::EPSILON
        );
        // SAFETY: `platform` is non-null and live.
        unsafe { stator_platform_destroy(platform) };
    }
}

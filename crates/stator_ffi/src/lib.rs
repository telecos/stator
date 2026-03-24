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

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::Write as _;
use std::rc::Rc;

use stator_core::bytecode::bytecode_array::BytecodeArray;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::bytecode::bytecodes::{Operand, decode};
use stator_core::dom::{
    DomObjectWrap, DomWeakRef, IndexedPropertyHandlerConfig, NamedPropertyHandlerConfig,
};
use stator_core::gc::heap::Heap;
use stator_core::interpreter::{GlobalEnv, Interpreter, InterpreterFrame};
use stator_core::objects::js_object::JsObject;
use stator_core::objects::property_map::PropertyMap;
use stator_core::objects::value::{JsValue, NativeFn};
use stator_core::parser;
use stator_core::wasm::{WasmEngine, WasmInstance, WasmModule};

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
    /// The innermost active [`StatorHandleScope`] on this isolate, or null if
    /// no scope is currently open.  This forms a linked list via each scope's
    /// `previous` field.
    active_handle_scope: *mut StatorHandleScope,
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
        active_handle_scope: std::ptr::null_mut(),
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

/// Return `true` if there is a pending exception on `isolate`.
///
/// Returns `false` when `isolate` is null.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_has_pending_exception(
    isolate: *const StatorIsolate,
) -> bool {
    if isolate.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).pending_exception.is_some() }
}

/// Clear the pending exception on `isolate` and return it.
///
/// Returns a null pointer when `isolate` is null or no pending exception is
/// set.  The caller owns the returned pointer and must eventually pass it to
/// [`stator_value_destroy`].
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_clear_pending_exception(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe {
        (*isolate)
            .pending_exception
            .take()
            .unwrap_or(std::ptr::null_mut())
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

/// Compilation tier statistics for an isolate.
///
/// Filled in by `stator_isolate_get_stats`.  All counts are thread-local
/// totals accumulated since the process started (or since the last
/// interpreter reset on this thread).
///
/// On platforms where the baseline JIT is not available (non-x86-64 or
/// non-Unix), all fields will always be zero.
#[repr(C)]
pub struct StatorCompilationStats {
    /// Number of JavaScript functions that have been compiled to baseline JIT.
    pub jit_functions_compiled: u32,
    /// Total bytes of machine code produced by the baseline JIT compiler.
    pub jit_code_bytes: usize,
}

/// Fill `*stats` with the current compilation tier statistics.
///
/// Reports the cumulative number of baseline-JIT-compiled functions and the
/// total machine-code bytes produced on this thread since the process started.
///
/// On platforms where the baseline JIT is not available all counts will be
/// zero.
///
/// Does nothing when `stats` is null.
///
/// # Safety
/// - `isolate` must be either null or a valid, live `StatorIsolate` pointer.
/// - `stats` must be a non-null, properly aligned pointer to a
///   `StatorCompilationStats` that is valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_get_stats(
    _isolate: *const StatorIsolate,
    stats: *mut StatorCompilationStats,
) {
    if stats.is_null() {
        return;
    }
    let (count, bytes) = stator_core::interpreter::jit_stats();
    // SAFETY: caller guarantees `stats` is valid for writes.
    unsafe {
        (*stats).jit_functions_compiled = count;
        (*stats).jit_code_bytes = bytes;
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
    /// Shared global-variable environment for script execution.
    ///
    /// Populated by [`stator_register_native_function`] and consumed by
    /// [`stator_script_run`].  The map owns [`JsValue`] values (including
    /// [`JsValue::NativeFunction`] and [`JsValue::PlainObject`] wrappers).
    pub(crate) globals: Rc<RefCell<GlobalEnv>>,
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
        globals: Rc::new(RefCell::new(GlobalEnv::new())),
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
    /// The ECMAScript `undefined` value.
    Undefined,
    /// The ECMAScript `null` value.
    Null,
    /// A JavaScript boolean (`true` or `false`).
    Boolean(bool),
    /// A double-precision floating-point number.
    Number(f64),
    /// A UTF-8 string stored as a null-terminated C string for easy FFI access.
    Str(CString),
    /// A JavaScript object (plain objects, GC-heap objects, generators,
    /// iterators, errors).
    Object,
    /// A callable JavaScript function (bytecode or native).
    Function,
    /// A native function created via a [`StatorFunctionTemplate`], carrying
    /// its callable [`NativeFn`] so it can be installed into a context and
    /// called from JavaScript.
    NativeFunctionValue(NativeFn),
    /// A JavaScript array.
    Array,
    /// A JavaScript `Date` object.
    Date,
    /// A JavaScript `RegExp` object.
    RegExp,
    /// A JavaScript `Promise` object.
    Promise,
    /// A JavaScript `Map` object.
    Map,
    /// A JavaScript `Set` object.
    Set,
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
    let val = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Number(val),
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(val);
        }
    }
    val
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
    let val = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Str(cstring),
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(val);
        }
    }
    val
}

/// Create a new boolean value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_boolean(
    isolate: *mut StatorIsolate,
    val: bool,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Boolean(val),
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `undefined` value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_undefined(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Undefined,
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `null` value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_null(isolate: *mut StatorIsolate) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Null,
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new plain-object value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_object(isolate: *mut StatorIsolate) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Object,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new function-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_function_tag(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Function,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new array-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_array_tag(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Array,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `Date`-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_date_tag(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Date,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `RegExp`-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_regexp_tag(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::RegExp,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `Promise`-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_promise_tag(
    isolate: *mut StatorIsolate,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Promise,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `Map`-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_map_tag(isolate: *mut StatorIsolate) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Map,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Create a new `Set`-tagged value.
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_new_set_tag(isolate: *mut StatorIsolate) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Set,
        isolate,
    }));
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

// ── V8-style typed constructors ───────────────────────────────────────────────

/// Create a new string value from a UTF-8 byte buffer of `len` bytes.
///
/// This is the V8-style spelling of [`stator_value_new_string`].  `data` need
/// not be null-terminated; `len` bytes are copied.  Any embedded null bytes
/// cause the string to be truncated at the first such byte.  Returns a null
/// pointer if `isolate` or `data` is null.
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `data` must be valid for reads of `len` bytes and must point to valid
///   UTF-8 data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_new_from_utf8(
    isolate: *mut StatorIsolate,
    data: *const c_char,
    len: usize,
) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_string`.
    unsafe { stator_value_new_string(isolate, data, len) }
}

/// Return the number of UTF-8 bytes required to represent `val`.
///
/// Returns `0` when `val` is null or not a string.  The returned length does
/// **not** include a terminating null byte.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_utf8_length(val: *const StatorValue) -> usize {
    if val.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Str(cs) => cs.as_bytes().len(),
        _ => 0,
    }
}

/// Write the UTF-8 representation of `val` into `buf`.
///
/// At most `buf_size` bytes are written.  The output is **not**
/// null-terminated (unlike `stator_value_as_string`).  If `nchars_ref` is
/// non-null, `*nchars_ref` is set to the number of bytes written.  Returns
/// the number of bytes written, or `0` when `val` is null or not a string or
/// `buf` is null.
///
/// # Safety
/// - `val` must be either null or a valid, live [`StatorValue`] pointer.
/// - `buf` must be valid for writes of `buf_size` bytes (unless null).
/// - `nchars_ref` must be either null or a valid pointer to a `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_write_utf8(
    val: *const StatorValue,
    buf: *mut c_char,
    buf_size: usize,
    nchars_ref: *mut usize,
) -> usize {
    if val.is_null() || buf.is_null() {
        if !nchars_ref.is_null() {
            // SAFETY: caller guarantees `nchars_ref` is valid.
            unsafe { *nchars_ref = 0 };
        }
        return 0;
    }
    // SAFETY: caller guarantees `val` is valid.
    let bytes = match unsafe { &(*val).inner } {
        StatorValueInner::Str(cs) => cs.as_bytes(),
        _ => {
            if !nchars_ref.is_null() {
                // SAFETY: caller guarantees `nchars_ref` is valid.
                unsafe { *nchars_ref = 0 };
            }
            return 0;
        }
    };
    let n = bytes.len().min(buf_size);
    // SAFETY: `buf` is valid for `buf_size` bytes; `n <= buf_size`.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n) };
    if !nchars_ref.is_null() {
        // SAFETY: caller guarantees `nchars_ref` is valid.
        unsafe { *nchars_ref = n };
    }
    n
}

/// Create a new number value (V8-style spelling of [`stator_value_new_number`]).
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_number_new(
    isolate: *mut StatorIsolate,
    val: f64,
) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_number`.
    unsafe { stator_value_new_number(isolate, val) }
}

/// Create a new integer value from a signed 64-bit integer.
///
/// The value is stored as an ECMAScript `Number` (IEEE 754 double).  Values
/// outside the safe-integer range (`±2⁵³−1`) may lose precision.  Returns a
/// null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_integer_new(
    isolate: *mut StatorIsolate,
    val: i64,
) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_number`.
    unsafe { stator_value_new_number(isolate, val as f64) }
}

/// Create a new boolean value (V8-style spelling of [`stator_value_new_boolean`]).
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_boolean_new(
    isolate: *mut StatorIsolate,
    val: bool,
) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_boolean`.
    unsafe { stator_value_new_boolean(isolate, val) }
}

/// Create a new `undefined` value (V8-style spelling of [`stator_value_new_undefined`]).
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_undefined_new(isolate: *mut StatorIsolate) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_undefined`.
    unsafe { stator_value_new_undefined(isolate) }
}

/// Create a new `null` value (V8-style spelling of [`stator_value_new_null`]).
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_null_new(isolate: *mut StatorIsolate) -> *mut StatorValue {
    // SAFETY: delegated to `stator_value_new_null`.
    unsafe { stator_value_new_null(isolate) }
}

/// Destroy a value and decrement the isolate's live-object counter.
///
/// # Safety
/// `val` must be a non-null pointer returned by any `stator_value_new_*`
/// function and must not be used again after this call.
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
        StatorValueInner::Undefined => c"undefined".as_ptr(),
        StatorValueInner::Null => c"object".as_ptr(),
        StatorValueInner::Boolean(_) => c"boolean".as_ptr(),
        StatorValueInner::Function | StatorValueInner::NativeFunctionValue(_) => {
            c"function".as_ptr()
        }
        StatorValueInner::Object
        | StatorValueInner::Array
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => c"object".as_ptr(),
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
        StatorValueInner::Str(_) | StatorValueInner::Undefined => f64::NAN,
        StatorValueInner::Null => 0.0,
        StatorValueInner::Boolean(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        StatorValueInner::Object
        | StatorValueInner::Function
        | StatorValueInner::NativeFunctionValue(_)
        | StatorValueInner::Array
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => f64::NAN,
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
        StatorValueInner::Number(_)
        | StatorValueInner::Undefined
        | StatorValueInner::Null
        | StatorValueInner::Boolean(_)
        | StatorValueInner::Object
        | StatorValueInner::Function
        | StatorValueInner::NativeFunctionValue(_)
        | StatorValueInner::Array
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => c"".as_ptr(),
    }
}

// ── Value type-checking predicates ───────────────────────────────────────────

/// Returns `true` if `val` is the ECMAScript `undefined` value.
///
/// A null `val` pointer is treated as `undefined` and also returns `true`.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_undefined(val: *const StatorValue) -> bool {
    if val.is_null() {
        return true;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Undefined)
}

/// Returns `true` if `val` is the ECMAScript `null` value.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_null(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Null)
}

/// Returns `true` if `val` holds a JavaScript string.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_string(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Str(_))
}

/// Returns `true` if `val` holds a JavaScript number.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_number(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Number(_))
}

/// Returns `true` if `val` holds a JavaScript boolean.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_boolean(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Boolean(_))
}

/// Returns `true` if `val` is a JavaScript object (excludes `null`).
///
/// Arrays, dates, regexps, promises, maps, and sets are all objects in
/// ECMAScript and therefore also return `true`.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_object(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(
        unsafe { &(*val).inner },
        StatorValueInner::Object
            | StatorValueInner::Array
            | StatorValueInner::Date
            | StatorValueInner::RegExp
            | StatorValueInner::Promise
            | StatorValueInner::Map
            | StatorValueInner::Set
    )
}

/// Returns `true` if `val` is a callable JavaScript function.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_function(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(
        unsafe { &(*val).inner },
        StatorValueInner::Function | StatorValueInner::NativeFunctionValue(_)
    )
}

/// Returns `true` if `val` is a JavaScript array.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_array(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Array)
}

/// Returns `true` if `val` is a number whose value is a signed 32-bit integer.
///
/// A value is considered an int32 when it is a finite number equal to its own
/// `ToInt32` conversion: i.e. it is an integer in the range `[−2³¹, 2³¹−1]`.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_int32(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    if let StatorValueInner::Number(n) = unsafe { &(*val).inner } {
        n.is_finite() && n.fract() == 0.0 && *n >= i32::MIN as f64 && *n <= i32::MAX as f64
    } else {
        false
    }
}

/// Returns `true` if `val` is a number whose value is an unsigned 32-bit integer.
///
/// A value is considered a uint32 when it is a finite number equal to its own
/// `ToUint32` conversion: i.e. it is an integer in the range `[0, 2³²−1]`.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_uint32(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    if let StatorValueInner::Number(n) = unsafe { &(*val).inner } {
        n.is_finite() && n.fract() == 0.0 && *n >= 0.0 && *n <= u32::MAX as f64
    } else {
        false
    }
}

/// Returns `true` if `val` is a JavaScript `Date` object.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_date(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Date)
}

/// Returns `true` if `val` is a JavaScript `RegExp` object.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_regexp(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::RegExp)
}

/// Returns `true` if `val` is a JavaScript `Promise` object.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_promise(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Promise)
}

/// Returns `true` if `val` is a JavaScript `Map` object.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_map(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Map)
}

/// Returns `true` if `val` is a JavaScript `Set` object.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_is_set(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    matches!(unsafe { &(*val).inner }, StatorValueInner::Set)
}

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
    let js_val = stator_value_inner_to_jsvalue(unsafe { &(*val).inner });
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

/// Return `true` if `obj` has a property with the given `key` (own or
/// inherited via the prototype chain).
///
/// Returns `false` if any argument is null.
///
/// # Safety
/// - `obj` must be a valid, live [`StatorObject`] pointer.
/// - `key` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_has(obj: *const StatorObject, key: *const c_char) -> bool {
    if obj.is_null() || key.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `key` is a valid null-terminated string.
    let key_str = unsafe { CStr::from_ptr(key) }.to_string_lossy();
    // SAFETY: caller guarantees `obj` is valid.
    unsafe { (*obj).inner.has_property(&key_str) }
}

/// Delete the named property `key` from `obj`.
///
/// Returns `true` if the property was successfully deleted (or did not exist),
/// `false` if the deletion fails (e.g. the property is non-configurable, the
/// object is non-extensible in a way that prevents deletion, or any argument
/// is null).
///
/// # Safety
/// - `obj` must be a valid, live [`StatorObject`] pointer.
/// - `key` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_delete(obj: *mut StatorObject, key: *const c_char) -> bool {
    if obj.is_null() || key.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `key` is a valid null-terminated string.
    let key_str = unsafe { CStr::from_ptr(key) }.to_string_lossy();
    // SAFETY: caller guarantees `obj` is valid.
    // `delete_own_property` returns `Ok(false)` when the property is
    // non-configurable and `Err(_)` on internal engine errors; both cases
    // map to `false` here, consistent with ECMAScript's [[Delete]] semantics.
    unsafe { (*obj).inner.delete_own_property(&key_str).unwrap_or(false) }
}

/// An opaque snapshot of the own property names of a JavaScript object.
///
/// Created by [`stator_object_get_property_names`] and freed by
/// [`stator_property_names_destroy`].  The individual name strings are owned
/// by this snapshot and remain valid until it is destroyed.
pub struct StatorPropertyNames {
    /// The property names as null-terminated C strings.
    names: Vec<CString>,
}

/// Collect the own enumerable property names of `obj` into a new
/// [`StatorPropertyNames`] snapshot.
///
/// Returns a null pointer if `obj` is null.  The caller must eventually pass
/// the returned pointer to [`stator_property_names_destroy`].
///
/// # Safety
/// `obj` must be a valid, live [`StatorObject`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_get_property_names(
    obj: *const StatorObject,
) -> *mut StatorPropertyNames {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `obj` is valid.
    let keys = unsafe { (*obj).inner.own_property_keys() };
    let names = keys
        .into_iter()
        .map(|k| {
            // Truncate at the first embedded NUL byte to produce a valid
            // null-terminated string; `from_vec_unchecked` skips the NUL-byte
            // check performed by `CString::new`, so we must do it manually.
            let valid_len = k.as_bytes().iter().position(|&b| b == 0).unwrap_or(k.len());
            // SAFETY: `&k.as_bytes()[..valid_len]` contains no NUL bytes.
            unsafe { CString::from_vec_unchecked(k.as_bytes()[..valid_len].to_vec()) }
        })
        .collect();
    Box::into_raw(Box::new(StatorPropertyNames { names }))
}

/// Return the number of property names in `names`.
///
/// Returns `0` if `names` is null.
///
/// # Safety
/// `names` must be either null or a valid, live [`StatorPropertyNames`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_property_names_count(names: *const StatorPropertyNames) -> usize {
    if names.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `names` is valid.
    unsafe { (*names).names.len() }
}

/// Return a pointer to the null-terminated property name at position `index`.
///
/// The returned pointer is valid for as long as `names` is alive and has not
/// been destroyed.  Returns a null pointer if `names` is null or `index` is
/// out of range.
///
/// # Safety
/// `names` must be either null or a valid, live [`StatorPropertyNames`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_property_names_get(
    names: *const StatorPropertyNames,
    index: usize,
) -> *const c_char {
    if names.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `names` is valid.
    let names_ref = unsafe { &*names };
    names_ref
        .names
        .get(index)
        .map_or(std::ptr::null(), |s| s.as_ptr())
}

/// Destroy a [`StatorPropertyNames`] snapshot previously returned by
/// [`stator_object_get_property_names`].
///
/// After this call the pointer and all name pointers obtained from it are
/// invalid and must not be used.
///
/// # Safety
/// `names` must be a non-null pointer returned by
/// `stator_object_get_property_names` and must not be used again after this
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_property_names_destroy(names: *mut StatorPropertyNames) {
    if !names.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(names) });
    }
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
                    .operands()
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

// ── Script execution (Phase 3) ────────────────────────────────────────────────

/// C-callable native-function signature.
///
/// The callback receives the active context, an array of `argc` argument
/// pointers (owned by the Rust side; **do not free them**), and the count.
/// It must return either a new [`StatorValue`] (caller must free it) or a
/// null pointer (treated as `undefined`).
type StatorNativeCallback = unsafe extern "C" fn(
    ctx: *mut StatorContext,
    args: *const *const StatorValue,
    argc: i32,
) -> *mut StatorValue;

/// Execute a compiled script in `ctx` and return the result as a
/// [`StatorValue`].
///
/// The script must have been successfully compiled (i.e.
/// [`stator_script_get_error`] returns null).  If the script produces an
/// exception the call returns null.
///
/// The caller must pass the returned pointer to [`stator_value_destroy`] when
/// done, or ignore a null return.
///
/// # Safety
/// - `script` must be a non-null pointer returned by [`stator_script_compile`].
/// - `ctx` must be a valid, live [`StatorContext`] pointer, or null (in which
///   case an empty global environment is used).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_run(
    script: *const StatorScript,
    ctx: *mut StatorContext,
) -> *mut StatorValue {
    if script.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b.clone(),
        None => return std::ptr::null_mut(),
    };

    // Borrow the global environment from the context (if any).
    let global_env = if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        Rc::clone(unsafe { &(*ctx).globals })
    } else {
        Rc::new(RefCell::new(GlobalEnv::new()))
    };

    let mut frame = InterpreterFrame::new_with_globals(Rc::new(bytecodes), vec![], global_env);
    match Interpreter::run(&mut frame) {
        Ok(val) => {
            // Wrap the result in a StatorValue.  Use the isolate from the
            // context when available; null isolate means the handle is
            // "untracked" (live_objects is not incremented).
            let isolate = if ctx.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: ctx is valid.
                unsafe { (*ctx)._isolate }
            };
            let inner = jsvalue_to_stator_value_inner(&val);
            if !isolate.is_null() {
                // SAFETY: isolate is valid.
                unsafe { (*isolate).live_objects += 1 };
            }
            Box::into_raw(Box::new(StatorValue { inner, isolate }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Convert the value `val` to a UTF-8 string and write it into `buf`.
///
/// At most `buf_len` bytes (including the NUL terminator) are written.
/// Returns the number of bytes written **excluding** the NUL terminator, or
/// `-1` on error.  Returns `0` when `val` is null (writes `""` and a NUL).
///
/// # Safety
/// - `val` must be either null or a valid, live [`StatorValue`] pointer.
/// - `buf` must be valid for writes of `buf_len` bytes, or null when
///   `buf_len` is 0 (in which case only the required length is returned).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_string_utf8(
    val: *const StatorValue,
    buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    let s = if val.is_null() {
        "undefined".to_owned()
    } else {
        // SAFETY: caller guarantees `val` is valid.
        value_inner_to_js_string(unsafe { &(*val).inner })
    };
    if buf.is_null() || buf_len == 0 {
        return s.len() as i32;
    }
    let bytes = s.as_bytes();
    // Copy at most buf_len-1 bytes so there is always room for a NUL.
    let copy_len = bytes.len().min(buf_len - 1);
    // SAFETY: buf is valid for buf_len bytes; we only write copy_len + 1.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, copy_len);
        *buf.add(copy_len) = 0;
    }
    copy_len as i32
}

// ── Value conversion (ECMAScript coercion) ───────────────────────────────────

/// Coerce `val` to a number following ECMAScript §7.1.4 **ToNumber**.
///
/// | Value type | Result |
/// |---|---|
/// | `undefined` (or null pointer) | `NaN` |
/// | `null` | `+0` |
/// | `true` | `1` |
/// | `false` | `+0` |
/// | number | the number itself |
/// | string | parsed numeric value; `NaN` if unparseable |
/// | object / function / … | `NaN` |
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_number(val: *const StatorValue) -> f64 {
    if val.is_null() {
        return f64::NAN;
    }
    // SAFETY: caller guarantees `val` is valid.
    value_inner_to_number(unsafe { &(*val).inner })
}

/// Coerce `val` to a string following ECMAScript §7.1.17 **ToString** and
/// return it as a new [`StatorValue`] of string type.
///
/// The caller owns the returned pointer and must pass it to
/// [`stator_value_destroy`] (or let a handle scope manage it).
///
/// Returns a null pointer if `isolate` is null.
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_string(
    isolate: *mut StatorIsolate,
    val: *const StatorValue,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let s = if val.is_null() {
        "undefined".to_owned()
    } else {
        // SAFETY: caller guarantees `val` is valid.
        value_inner_to_js_string(unsafe { &(*val).inner })
    };
    let valid_len = s.as_bytes().iter().position(|&b| b == 0).unwrap_or(s.len());
    // SAFETY: `&s.as_bytes()[..valid_len]` contains no NUL bytes.
    let cstring = unsafe { CString::from_vec_unchecked(s.as_bytes()[..valid_len].to_vec()) };
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let v = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::Str(cstring),
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(v);
        }
    }
    v
}

/// Coerce `val` to a signed 32-bit integer following ECMAScript §7.1.7
/// **ToInt32**.
///
/// Applies `ToNumber` first, then reduces modulo 2³² and maps to the signed
/// range `[−2³¹, 2³¹−1]`.  `NaN`, `±0`, and `±Infinity` all convert to `0`.
///
/// Returns `0` when `val` is null.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_int32(val: *const StatorValue) -> i32 {
    let n = if val.is_null() {
        f64::NAN
    } else {
        // SAFETY: caller guarantees `val` is valid.
        value_inner_to_number(unsafe { &(*val).inner })
    };
    number_to_int32(n)
}

/// Coerce `val` to an unsigned 32-bit integer following ECMAScript §7.1.8
/// **ToUint32**.
///
/// Applies `ToNumber` first, then reduces modulo 2³².  `NaN`, `±0`, and
/// `±Infinity` all convert to `0`.
///
/// Returns `0` when `val` is null.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_uint32(val: *const StatorValue) -> u32 {
    let n = if val.is_null() {
        f64::NAN
    } else {
        // SAFETY: caller guarantees `val` is valid.
        value_inner_to_number(unsafe { &(*val).inner })
    };
    number_to_uint32(n)
}

/// Coerce `val` to a boolean following ECMAScript §7.1.2 **ToBoolean**.
///
/// Falsy values: `undefined`, `null`, `false`, `+0`, `-0`, `NaN`, `""`.
/// Everything else is truthy.  A null pointer is treated as `undefined` and
/// returns `false`.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_to_boolean(val: *const StatorValue) -> bool {
    if val.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Undefined | StatorValueInner::Null => false,
        StatorValueInner::Boolean(b) => *b,
        StatorValueInner::Number(n) => !n.is_nan() && *n != 0.0,
        StatorValueInner::Str(cs) => !cs.as_bytes().is_empty(),
        // All object-like values are truthy.
        StatorValueInner::Object
        | StatorValueInner::Function
        | StatorValueInner::NativeFunctionValue(_)
        | StatorValueInner::Array
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => true,
    }
}

/// Test whether `a === b` following ECMAScript §7.2.15 **IsStrictlyEqual**.
///
/// Rules:
/// - Different types → `false`.
/// - `undefined === undefined` → `true`.
/// - `null === null` → `true`.
/// - Numbers: `NaN !== NaN`; `+0 === -0`.
/// - Strings: byte-for-byte equality.
/// - Booleans: value equality.
/// - Object / function / array / … : `false` (no shared identity in FFI
///   handles; two distinct handles are never the same object).
///
/// Both null pointers are treated as `undefined`.
///
/// # Safety
/// `a` and `b` must each be either null or a valid, live [`StatorValue`]
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_strict_equals(
    a: *const StatorValue,
    b: *const StatorValue,
) -> bool {
    let a_inner = if a.is_null() {
        &StatorValueInner::Undefined
    } else {
        // SAFETY: caller guarantees `a` is valid.
        unsafe { &(*a).inner }
    };
    let b_inner = if b.is_null() {
        &StatorValueInner::Undefined
    } else {
        // SAFETY: caller guarantees `b` is valid.
        unsafe { &(*b).inner }
    };
    match (a_inner, b_inner) {
        (StatorValueInner::Undefined, StatorValueInner::Undefined) => true,
        (StatorValueInner::Null, StatorValueInner::Null) => true,
        (StatorValueInner::Boolean(x), StatorValueInner::Boolean(y)) => x == y,
        (StatorValueInner::Number(x), StatorValueInner::Number(y)) => {
            // NaN !== NaN; +0 === -0 (IEEE 0.0 == -0.0 is true in Rust).
            if x.is_nan() || y.is_nan() {
                return false;
            }
            x == y
        }
        (StatorValueInner::Str(x), StatorValueInner::Str(y)) => x == y,
        // Object-like tags carry no identity in FFI handles → never equal.
        _ => false,
    }
}

/// Apply ECMAScript §7.1.7 **ToInt32** to a finite number.
///
/// Reduces `n` modulo 2³² and maps into the signed range `[−2³¹, 2³¹−1]`.
fn number_to_int32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    const TWO_32: f64 = 4_294_967_296.0_f64;
    const TWO_31: f64 = 2_147_483_648.0_f64;
    let int = n.trunc();
    // Mathematical modulo: result is always in [0, TWO_32).
    let int32bit = ((int % TWO_32) + TWO_32) % TWO_32;
    if int32bit >= TWO_31 {
        (int32bit - TWO_32) as i32
    } else {
        int32bit as i32
    }
}

/// Apply ECMAScript §7.1.8 **ToUint32** to a finite number.
///
/// Reduces `n` modulo 2³² into the range `[0, 2³²−1]`.
fn number_to_uint32(n: f64) -> u32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    const TWO_32: f64 = 4_294_967_296.0_f64;
    let int = n.trunc();
    // Mathematical modulo: result is always in [0, TWO_32).
    let int32bit = ((int % TWO_32) + TWO_32) % TWO_32;
    int32bit as u32
}

/// Register a native function named `name` on `ctx`.
///
/// After registration, JavaScript code running via [`stator_script_run`] can
/// call `name(…)` as a global function.  The C `callback` is invoked with the
/// active context, an array of argument pointers (valid only for the duration
/// of the call), and the argument count.
///
/// The name `"console"` has special semantics: if not yet defined it creates a
/// plain object; the function is then installed as `console.<method>` where
/// `<method>` is derived from the dot notation in `name`.  For example,
/// `name = "console.log"` registers `log` on the `console` object.
///
/// Does nothing when `ctx` or `name` is null.
///
/// # Safety
/// - `ctx` must be a valid, live [`StatorContext`] pointer.
/// - `name` must be a valid, null-terminated C string.
/// - `callback` must remain callable for the lifetime of `ctx`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_register_native_function(
    ctx: *mut StatorContext,
    name: *const c_char,
    callback: StatorNativeCallback,
) {
    if ctx.is_null() || name.is_null() {
        return;
    }
    // SAFETY: caller guarantees `name` is a valid C string.
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();

    // Build a Rust closure that wraps the C callback.
    let ctx_ptr = ctx;
    let native: NativeFn = Rc::new(move |args: Vec<JsValue>| {
        // Convert JsValue args to temporary StatorValue* for the C side.
        let c_vals: Vec<StatorValue> = args
            .iter()
            .map(|v| StatorValue {
                inner: jsvalue_to_stator_value_inner(v),
                isolate: std::ptr::null_mut(),
            })
            .collect();
        let c_ptrs: Vec<*const StatorValue> =
            c_vals.iter().map(|v| v as *const StatorValue).collect();

        // SAFETY: the callback is valid for the lifetime of `ctx` (per the
        // FFI contract), and the argument pointers are valid for this call.
        let ret = unsafe { callback(ctx_ptr, c_ptrs.as_ptr(), c_ptrs.len() as i32) };

        if ret.is_null() {
            Ok(JsValue::Undefined)
        } else {
            // Convert returned StatorValue to JsValue, then free it.
            // SAFETY: `ret` was allocated by Box::into_raw in the callback.
            let owned = unsafe { Box::from_raw(ret) };
            Ok(stator_value_inner_to_jsvalue(&owned.inner))
        }
    });

    // SAFETY: ctx is valid.
    let globals = unsafe { Rc::clone(&(*ctx).globals) };

    // Check whether the name contains a dot (e.g. "console.log").
    if let Some(dot) = name_str.find('.') {
        let obj_name = &name_str[..dot];
        let method_name = &name_str[dot + 1..];

        let env = globals.borrow_mut();
        // Insert the parent object if it doesn't exist yet.
        let obj = if let Some(existing) = env.get(obj_name) {
            existing.clone()
        } else {
            JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())))
        };
        drop(env); // release borrow before we potentially re-borrow

        if let JsValue::PlainObject(ref map) = obj {
            map.borrow_mut()
                .insert(method_name.to_owned(), JsValue::NativeFunction(native));
        }
        // Ensure the (possibly newly-created) object is stored back.
        globals.borrow_mut().insert(obj_name.to_owned(), obj);
    } else {
        globals
            .borrow_mut()
            .insert(name_str, JsValue::NativeFunction(native));
    }
}

// ── FunctionTemplate (Phase 4) ────────────────────────────────────────────────

/// C-callable function-template callback signature.
///
/// The callback receives a pointer to a [`StatorFunctionCallbackInfo`] which
/// provides access to the call arguments and isolate.  It must return either
/// a new [`StatorValue`] (the caller — i.e. the engine wrapper — owns it and
/// frees it automatically) or a null pointer (treated as `undefined`).
type StatorFunctionTemplateCallback =
    unsafe extern "C" fn(*const StatorFunctionCallbackInfo) -> *mut StatorValue;

/// Call-site information passed to a function-template callback.
///
/// The lifetime of a `StatorFunctionCallbackInfo` value is limited to the
/// duration of the native callback invocation; the embedder must not store the
/// pointer beyond that.
pub struct StatorFunctionCallbackInfo {
    /// Temporary argument values valid for the duration of the call.
    args: Vec<StatorValue>,
    /// The isolate this call is happening on.
    isolate: *mut StatorIsolate,
}

// SAFETY: `StatorFunctionCallbackInfo` contains raw pointer fields that are
// only ever accessed on the owning thread; the embedder is responsible for
// external synchronisation if the value is transferred across threads.
unsafe impl Send for StatorFunctionCallbackInfo {}

/// Return the number of arguments passed to this call.
///
/// Returns `0` when `info` is null.
///
/// # Safety
/// `info` must be either null or a valid pointer to a live
/// [`StatorFunctionCallbackInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_callback_info_length(
    info: *const StatorFunctionCallbackInfo,
) -> i32 {
    if info.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `info` is valid.
    unsafe { (*info).args.len() as i32 }
}

/// Return a pointer to the argument at position `index`.
///
/// The returned pointer is valid only for the duration of the callback
/// invocation.  Returns a null pointer if `info` is null or `index` is out of
/// range.
///
/// # Safety
/// `info` must be either null or a valid pointer to a live
/// [`StatorFunctionCallbackInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_callback_info_get(
    info: *const StatorFunctionCallbackInfo,
    index: i32,
) -> *const StatorValue {
    if info.is_null() || index < 0 {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `info` is valid.
    let args = unsafe { &(*info).args };
    let idx = index as usize;
    if idx >= args.len() {
        return std::ptr::null();
    }
    &args[idx] as *const StatorValue
}

/// Return the isolate associated with this call.
///
/// Returns a null pointer if `info` is null.
///
/// # Safety
/// `info` must be either null or a valid pointer to a live
/// [`StatorFunctionCallbackInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_callback_info_get_isolate(
    info: *const StatorFunctionCallbackInfo,
) -> *mut StatorIsolate {
    if info.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `info` is valid.
    unsafe { (*info).isolate }
}

/// An opaque function template handle.
///
/// A function template associates a C callback with an isolate so that
/// [`stator_function_template_get_function`] can produce a [`StatorValue`]
/// representing the function, which can then be installed into a context's
/// global environment via [`stator_context_global_set`].
pub struct StatorFunctionTemplate {
    isolate: *mut StatorIsolate,
    callback: StatorFunctionTemplateCallback,
}

// SAFETY: `StatorFunctionTemplate` contains raw pointer fields that are only
// ever accessed on the owning thread; the embedder is responsible for external
// synchronisation if the template is transferred across threads.
unsafe impl Send for StatorFunctionTemplate {}

/// Create a new function template associated with `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_function_template_destroy`].
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `callback` must remain valid for the lifetime of the template.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_template_new(
    isolate: *mut StatorIsolate,
    callback: StatorFunctionTemplateCallback,
) -> *mut StatorFunctionTemplate {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(StatorFunctionTemplate { isolate, callback }))
}

/// Destroy a function template previously created with
/// [`stator_function_template_new`].
///
/// Does nothing if `tmpl` is null.
///
/// # Safety
/// `tmpl` must be a non-null pointer returned by
/// [`stator_function_template_new`] and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_template_destroy(tmpl: *mut StatorFunctionTemplate) {
    if !tmpl.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(tmpl) });
    }
}

/// Produce a [`StatorValue`] representing the function described by `tmpl`.
///
/// The returned value has type `"function"` and can be installed into a
/// context's global environment via [`stator_context_global_set`], after which
/// JavaScript code run with [`stator_script_run`] can call it as a global
/// function.
///
/// Returns a null pointer if `tmpl` is null.  The caller owns the returned
/// pointer and must eventually pass it to [`stator_value_destroy`] (or let a
/// handle scope manage it).
///
/// # Safety
/// `tmpl` must be either null or a valid, live [`StatorFunctionTemplate`]
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_template_get_function(
    tmpl: *const StatorFunctionTemplate,
    _ctx: *mut StatorContext,
) -> *mut StatorValue {
    if tmpl.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `tmpl` is valid.
    let isolate = unsafe { (*tmpl).isolate };
    let callback = unsafe { (*tmpl).callback };

    // Build a NativeFn closure that wraps the C callback.
    let native: NativeFn = Rc::new(move |args: Vec<JsValue>| {
        // Convert JsValue args into temporary StatorValue structs on the stack.
        let c_vals: Vec<StatorValue> = args
            .iter()
            .map(|v| StatorValue {
                inner: jsvalue_to_stator_value_inner(v),
                isolate: std::ptr::null_mut(),
            })
            .collect();

        let info = StatorFunctionCallbackInfo {
            args: c_vals,
            isolate,
        };

        // Temporarily disable the active handle scope so that any
        // StatorValue created inside the callback (e.g. via
        // stator_value_new_object) is NOT auto-registered with an outer
        // scope.  Without this, the callback-created value would be owned
        // by both the outer scope and the Box::from_raw below, causing a
        // double-free when the scope later calls stator_value_destroy.
        let saved_scope = if !isolate.is_null() {
            // SAFETY: isolate is valid; we restore the scope pointer below.
            let s = unsafe { (*isolate).active_handle_scope };
            unsafe { (*isolate).active_handle_scope = std::ptr::null_mut() };
            s
        } else {
            std::ptr::null_mut()
        };

        // SAFETY: `callback` is valid for the lifetime of the template (per
        // the FFI contract).  `info` is a local variable live for this call.
        let ret = unsafe { callback(&raw const info) };

        // Restore the handle scope.
        if !isolate.is_null() {
            // SAFETY: isolate is valid; saved_scope was obtained from it.
            unsafe { (*isolate).active_handle_scope = saved_scope };
        }

        if ret.is_null() {
            Ok(JsValue::Undefined)
        } else {
            // Convert the returned StatorValue to JsValue, then free it.
            // The handle scope was disabled during the callback, so `ret`
            // is not scope-owned: taking ownership via Box::from_raw is safe.
            // SAFETY: `ret` was allocated by `Box::into_raw` in the callback.
            let owned = unsafe { Box::from_raw(ret) };
            Ok(stator_value_inner_to_jsvalue(&owned.inner))
        }
    });

    if !isolate.is_null() {
        // SAFETY: isolate is valid.
        unsafe { (*isolate).live_objects += 1 };
    }
    let val = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::NativeFunctionValue(native),
        isolate,
    }));
    // Register with the active handle scope, if any.
    // SAFETY: `isolate` is valid; `active_handle_scope` is either null or a
    // valid live scope pointer.
    if !isolate.is_null() {
        unsafe {
            let scope = (*isolate).active_handle_scope;
            if !scope.is_null() {
                (*scope).handles.push(val);
            }
        }
    }
    val
}

/// Install a value into the context's global environment under `name`.
///
/// After this call, JavaScript code running via [`stator_script_run`] can
/// reference `name` as a global variable.  For function values (created via
/// [`stator_function_template_get_function`] or
/// [`stator_value_new_*`][stator_value_new_number]), the callable is installed
/// so that scripts can invoke it.
///
/// Does nothing if `ctx` or `name` is null.  A null `val` installs
/// `undefined`.
///
/// # Safety
/// - `ctx` must be a valid, live [`StatorContext`] pointer.
/// - `name` must be a valid, null-terminated C string.
/// - `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_global_set(
    ctx: *mut StatorContext,
    name: *const c_char,
    val: *const StatorValue,
) {
    if ctx.is_null() || name.is_null() {
        return;
    }
    // SAFETY: caller guarantees `name` is a valid C string.
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();

    let jsval = if val.is_null() {
        JsValue::Undefined
    } else {
        // SAFETY: caller guarantees `val` is valid.
        stator_value_inner_to_jsvalue(unsafe { &(*val).inner })
    };

    // SAFETY: ctx is valid.
    let globals = unsafe { Rc::clone(&(*ctx).globals) };

    // Support dot-notation (e.g. "document.getElementById") by creating a
    // nested PlainObject for the parent, mirroring stator_register_native_function.
    if let Some(dot) = name_str.find('.') {
        let obj_name = &name_str[..dot];
        let method_name = &name_str[dot + 1..];

        let env = globals.borrow_mut();
        let obj = if let Some(existing) = env.get(obj_name) {
            existing.clone()
        } else {
            JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())))
        };
        drop(env);

        if let JsValue::PlainObject(ref map) = obj {
            map.borrow_mut().insert(method_name.to_owned(), jsval);
        }
        globals.borrow_mut().insert(obj_name.to_owned(), obj);
    } else {
        globals.borrow_mut().insert(name_str, jsval);
    }
}

/// Convert a [`StatorValueInner`] to a number following ECMAScript §7.1.4 **ToNumber**.
///
/// Object-like tags (`Object`, `Function`, `Array`, `Date`, `RegExp`,
/// `Promise`, `Map`, `Set`) return `NaN` because a `ToPrimitive` call is not
/// available at the FFI layer.
fn value_inner_to_number(inner: &StatorValueInner) -> f64 {
    match inner {
        StatorValueInner::Undefined => f64::NAN,
        StatorValueInner::Null => 0.0,
        StatorValueInner::Boolean(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        StatorValueInner::Number(n) => *n,
        StatorValueInner::Str(cs) => {
            let s = cs.to_string_lossy();
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return 0.0;
            }
            if trimmed == "Infinity" || trimmed == "+Infinity" {
                return f64::INFINITY;
            }
            if trimmed == "-Infinity" {
                return f64::NEG_INFINITY;
            }
            if let Some(hex) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                return u64::from_str_radix(hex, 16)
                    .map(|n| n as f64)
                    .unwrap_or(f64::NAN);
            }
            if let Some(bin) = trimmed
                .strip_prefix("0b")
                .or_else(|| trimmed.strip_prefix("0B"))
            {
                return u64::from_str_radix(bin, 2)
                    .map(|n| n as f64)
                    .unwrap_or(f64::NAN);
            }
            if let Some(oct) = trimmed
                .strip_prefix("0o")
                .or_else(|| trimmed.strip_prefix("0O"))
            {
                return u64::from_str_radix(oct, 8)
                    .map(|n| n as f64)
                    .unwrap_or(f64::NAN);
            }
            trimmed.parse::<f64>().unwrap_or(f64::NAN)
        }
        StatorValueInner::Object
        | StatorValueInner::Function
        | StatorValueInner::NativeFunctionValue(_)
        | StatorValueInner::Array
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => f64::NAN,
    }
}

/// Convert a [`StatorValueInner`] to a `String` following ECMAScript §7.1.17
/// **ToString**.
fn value_inner_to_js_string(inner: &StatorValueInner) -> String {
    match inner {
        StatorValueInner::Undefined => "undefined".to_owned(),
        StatorValueInner::Null => "null".to_owned(),
        StatorValueInner::Boolean(b) => if *b { "true" } else { "false" }.to_owned(),
        StatorValueInner::Number(n) => {
            if n.is_nan() {
                "NaN".to_owned()
            } else if n.is_infinite() {
                if *n > 0.0 {
                    "Infinity".to_owned()
                } else {
                    "-Infinity".to_owned()
                }
            } else if *n == 0.0 {
                "0".to_owned()
            } else {
                format!("{n}")
            }
        }
        StatorValueInner::Str(cs) => cs.to_string_lossy().into_owned(),
        StatorValueInner::Object => "[object Object]".to_owned(),
        StatorValueInner::Function | StatorValueInner::NativeFunctionValue(_) => {
            "function () { [native code] }".to_owned()
        }
        StatorValueInner::Array => "".to_owned(),
        StatorValueInner::Date => "[object Date]".to_owned(),
        StatorValueInner::RegExp => "(?:)".to_owned(),
        StatorValueInner::Promise => "[object Promise]".to_owned(),
        StatorValueInner::Map => "[object Map]".to_owned(),
        StatorValueInner::Set => "[object Set]".to_owned(),
    }
}

/// Convert a [`StatorValueInner`] to a [`JsValue`].
///
/// This is the counterpart to [`jsvalue_to_stator_value_inner`] and is used
/// when converting values returned from C native callbacks back to Rust.
fn stator_value_inner_to_jsvalue(inner: &StatorValueInner) -> JsValue {
    match inner {
        StatorValueInner::Number(n) => {
            // Represent exact integers that fit in i32 as Smi for efficiency.
            if n.fract() == 0.0
                && n.is_finite()
                && (*n >= i32::MIN as f64)
                && (*n <= i32::MAX as f64)
            {
                JsValue::Smi(*n as i32)
            } else {
                JsValue::HeapNumber(*n)
            }
        }
        StatorValueInner::Str(cs) => JsValue::String(cs.to_string_lossy().into_owned().into()),
        StatorValueInner::Undefined => JsValue::Undefined,
        StatorValueInner::Null => JsValue::Null,
        StatorValueInner::Boolean(b) => JsValue::Boolean(*b),
        StatorValueInner::Object
        | StatorValueInner::Date
        | StatorValueInner::RegExp
        | StatorValueInner::Promise
        | StatorValueInner::Map
        | StatorValueInner::Set => JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new()))),
        StatorValueInner::Function => {
            // Return a no-op native function to preserve the callable nature.
            JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined)))
        }
        StatorValueInner::NativeFunctionValue(native) => {
            // Share the same callable closure with the interpreter.  The Rc
            // allows both the StatorValue handle and the context globals map to
            // reference the same NativeFn without requiring ownership transfer.
            JsValue::NativeFunction(Rc::clone(native))
        }
        StatorValueInner::Array => JsValue::new_array(vec![]),
    }
}

/// Convert a [`JsValue`] to the inner storage type used by [`StatorValue`].
fn jsvalue_to_stator_value_inner(v: &JsValue) -> StatorValueInner {
    match v {
        JsValue::Undefined | JsValue::TheHole => StatorValueInner::Undefined,
        JsValue::Null => StatorValueInner::Null,
        JsValue::Boolean(b) => StatorValueInner::Boolean(*b),
        JsValue::Smi(n) => StatorValueInner::Number(f64::from(*n)),
        JsValue::HeapNumber(n) => StatorValueInner::Number(*n),
        JsValue::String(s) => {
            // Truncate at the first embedded NUL byte (CString requirement).
            let valid_len = s.as_bytes().iter().position(|&b| b == 0).unwrap_or(s.len());
            // SAFETY: `&s.as_bytes()[..valid_len]` contains no NUL bytes.
            unsafe {
                StatorValueInner::Str(CString::from_vec_unchecked(
                    s.as_bytes()[..valid_len].to_vec(),
                ))
            }
        }
        JsValue::Function(_) | JsValue::NativeFunction(_) => StatorValueInner::Function,
        JsValue::Array(_) => StatorValueInner::Array,
        JsValue::Object(_)
        | JsValue::Generator(_)
        | JsValue::Iterator(_)
        | JsValue::Error(_)
        | JsValue::Promise(_)
        | JsValue::PlainObject(_) => StatorValueInner::Object,
        JsValue::Symbol(_)
        | JsValue::BigInt(_)
        | JsValue::Context(_)
        | JsValue::Proxy(_)
        | JsValue::ArrayBuffer(_)
        | JsValue::TypedArray(_)
        | JsValue::DataView(_) => {
            // Symbols, BigInts, and Contexts are not yet representable in StatorValueInner;
            // fall back to a string representation so callers can inspect them.
            let s = v.to_js_string().unwrap_or_else(|_| "undefined".to_owned());
            let valid_len = s.as_bytes().iter().position(|&b| b == 0).unwrap_or(s.len());
            // SAFETY: `&s.as_bytes()[..valid_len]` contains no NUL bytes.
            unsafe {
                StatorValueInner::Str(CString::from_vec_unchecked(
                    s.as_bytes()[..valid_len].to_vec(),
                ))
            }
        }
    }
}

// ── HandleScope ───────────────────────────────────────────────────────────────

/// An opaque handle scope that manages the lifetime of [`StatorValue`] handles
/// created while the scope is open.
///
/// When a handle scope is open on an isolate, any value created via
/// [`stator_value_new_number`] or [`stator_value_new_string`] is automatically
/// registered with the innermost open scope.  Calling
/// [`stator_handle_scope_close`] destroys all registered values and restores
/// the previous scope.
///
/// Handle scopes nest: opening a second scope while one is already open is
/// valid; closing it restores the outer scope.
///
/// # Ownership
/// Values registered with a scope are **owned by the scope**.  The embedder
/// must **not** call [`stator_value_destroy`] on those values; doing so would
/// result in a double-free.
pub struct StatorHandleScope {
    isolate: *mut StatorIsolate,
    /// Values allocated while this scope was the active scope.
    handles: Vec<*mut StatorValue>,
    /// The scope that was active before this one was pushed.  Restored when
    /// this scope is closed.
    previous: *mut StatorHandleScope,
}

// SAFETY: `StatorHandleScope` is single-threaded; same rationale as
// `StatorIsolate`.
unsafe impl Send for StatorHandleScope {}

/// Open a new handle scope on `isolate`.
///
/// All [`StatorValue`] handles created after this call (and before the
/// corresponding [`stator_handle_scope_close`]) are owned by the returned
/// scope and will be destroyed automatically when the scope is closed.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_handle_scope_close`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_handle_scope_new(
    isolate: *mut StatorIsolate,
) -> *mut StatorHandleScope {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    let previous = unsafe { (*isolate).active_handle_scope };
    let scope = Box::into_raw(Box::new(StatorHandleScope {
        isolate,
        handles: Vec::new(),
        previous,
    }));
    // SAFETY: `isolate` is valid; `scope` was just created.
    unsafe { (*isolate).active_handle_scope = scope };
    scope
}

/// Close a handle scope and destroy all values registered with it.
///
/// After this call, all [`StatorValue`] pointers that were created while
/// `scope` was the active scope are invalid.  The previous scope (if any) is
/// restored as the active scope on the isolate.
///
/// Does nothing if `scope` is null.
///
/// # Safety
/// - `scope` must be a non-null pointer returned by [`stator_handle_scope_new`].
/// - `scope` must be the *innermost* open scope on its isolate (i.e. handle
///   scopes must be closed in LIFO order).
/// - `scope` must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_handle_scope_close(scope: *mut StatorHandleScope) {
    if scope.is_null() {
        return;
    }
    // SAFETY: caller guarantees `scope` is valid.
    unsafe {
        let isolate = (*scope).isolate;
        // Restore the previous scope on the isolate.
        if !isolate.is_null() {
            (*isolate).active_handle_scope = (*scope).previous;
        }
        // Destroy every value registered with this scope.
        for val in (*scope).handles.drain(..) {
            // SAFETY: each `val` was allocated by `stator_value_new_*` and is
            // still live (the embedder must not have called `stator_value_destroy`
            // on scope-owned values).
            stator_value_destroy(val);
        }
        drop(Box::from_raw(scope));
    }
}

// ── EscapableHandleScope ──────────────────────────────────────────────────────

/// An opaque escapable handle scope.
///
/// Works like [`StatorHandleScope`] but allows a single value to be
/// *escaped* — promoted into the enclosing scope (or left as an embedder-owned
/// handle if there is no enclosing scope) — via
/// [`stator_escapable_handle_scope_escape`].
///
/// Created by [`stator_escapable_handle_scope_new`] and closed (and its
/// remaining values destroyed) by [`stator_escapable_handle_scope_close`].
pub struct StatorEscapableHandleScope {
    /// Inner scope that manages handle tracking and nesting.
    inner: StatorHandleScope,
}

// SAFETY: same rationale as `StatorHandleScope`.
unsafe impl Send for StatorEscapableHandleScope {}

/// Open a new escapable handle scope on `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_escapable_handle_scope_close`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_escapable_handle_scope_new(
    isolate: *mut StatorIsolate,
) -> *mut StatorEscapableHandleScope {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    let previous = unsafe { (*isolate).active_handle_scope };
    let scope = Box::into_raw(Box::new(StatorEscapableHandleScope {
        inner: StatorHandleScope {
            isolate,
            handles: Vec::new(),
            previous,
        },
    }));
    // Point the isolate's active scope at the inner `StatorHandleScope` so
    // that value-creation functions register handles in this scope.
    // SAFETY: `scope` was just created; `isolate` is valid.
    unsafe { (*isolate).active_handle_scope = &raw mut (*scope).inner };
    scope
}

/// Escape `val` from `scope`, promoting it into the enclosing scope.
///
/// Removes `val` from `scope`'s tracked handles (so it will not be destroyed
/// when the scope is closed) and, if there is an enclosing scope, registers
/// it there.  If there is no enclosing scope the caller takes ownership and
/// must eventually pass `val` to [`stator_value_destroy`].
///
/// Returns `val` unchanged (for convenient chaining in C/C++ callers), or a
/// null pointer if `scope` or `val` is null.
///
/// # Safety
/// - `scope` must be a non-null, valid pointer to a live
///   [`StatorEscapableHandleScope`].
/// - `val` must be a non-null pointer that is currently registered with
///   `scope` (i.e. it was created while `scope` was the active scope).
/// - Each value may only be escaped once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_escapable_handle_scope_escape(
    scope: *mut StatorEscapableHandleScope,
    val: *mut StatorValue,
) -> *mut StatorValue {
    if scope.is_null() || val.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `scope` is valid.
    unsafe {
        // Remove `val` from this scope's handle list so it won't be destroyed
        // when the scope closes.
        (*scope).inner.handles.retain(|&h| h != val);

        // Register `val` with the enclosing scope (if any).
        let previous = (*scope).inner.previous;
        if !previous.is_null() {
            (*previous).handles.push(val);
        }
    }
    val
}

/// Close an escapable handle scope and destroy all non-escaped values.
///
/// Equivalent in behaviour to [`stator_handle_scope_close`] for the escapable
/// variant.
///
/// # Safety
/// - `scope` must be a non-null pointer returned by
///   [`stator_escapable_handle_scope_new`].
/// - `scope` must be the innermost open scope on its isolate.
/// - `scope` must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_escapable_handle_scope_close(
    scope: *mut StatorEscapableHandleScope,
) {
    if scope.is_null() {
        return;
    }
    // SAFETY: caller guarantees `scope` is valid.
    unsafe {
        let isolate = (*scope).inner.isolate;
        // Restore the previous scope on the isolate.
        if !isolate.is_null() {
            (*isolate).active_handle_scope = (*scope).inner.previous;
        }
        // Destroy every non-escaped value registered with this scope.
        for val in (*scope).inner.handles.drain(..) {
            // SAFETY: same as in `stator_handle_scope_close`.
            stator_value_destroy(val);
        }
        drop(Box::from_raw(scope));
    }
}

// ── ObjectTemplate ────────────────────────────────────────────────────────────

/// An opaque object template handle.
///
/// An object template describes the shape (properties and accessors) of objects
/// that will be created from it.  This mirrors V8's `ObjectTemplate` and is
/// used by Blink to set up DOM prototype objects.
///
/// Created by [`stator_object_template_new`] and freed by
/// [`stator_object_template_destroy`].
pub struct StatorObjectTemplate {
    isolate: *mut StatorIsolate,
    /// Named properties to install on each new instance.  Each entry maps a
    /// property name to a value that is cloned into every instance.
    properties: HashMap<String, StatorValueInner>,
    /// The number of internal fields reserved for embedder data.
    internal_field_count: i32,
}

// SAFETY: `StatorObjectTemplate` is single-threaded; same rationale as
// `StatorIsolate`.
unsafe impl Send for StatorObjectTemplate {}

/// Create a new, empty object template associated with `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_object_template_destroy`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_new(
    isolate: *mut StatorIsolate,
) -> *mut StatorObjectTemplate {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(StatorObjectTemplate {
        isolate,
        properties: HashMap::new(),
        internal_field_count: 0,
    }))
}

/// Destroy an object template previously created with
/// [`stator_object_template_new`].
///
/// Does nothing if `tmpl` is null.
///
/// # Safety
/// `tmpl` must be a non-null pointer returned by
/// [`stator_object_template_new`] and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_destroy(tmpl: *mut StatorObjectTemplate) {
    if !tmpl.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(tmpl) });
    }
}

/// Set a named property on the object template.
///
/// When a new instance is created via [`stator_object_template_new_instance`],
/// it will have this property pre-installed.  Does nothing if any argument is
/// null.
///
/// # Safety
/// - `tmpl` must be a valid, live [`StatorObjectTemplate`] pointer.
/// - `key` must be a valid, null-terminated C string.
/// - `val` must be a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_set(
    tmpl: *mut StatorObjectTemplate,
    key: *const c_char,
    val: *const StatorValue,
) {
    if tmpl.is_null() || key.is_null() || val.is_null() {
        return;
    }
    // SAFETY: caller guarantees pointers are valid.
    let key_str = unsafe { CStr::from_ptr(key) }
        .to_string_lossy()
        .into_owned();
    let inner = match unsafe { &(*val).inner } {
        StatorValueInner::Number(n) => StatorValueInner::Number(*n),
        StatorValueInner::Boolean(b) => StatorValueInner::Boolean(*b),
        StatorValueInner::Undefined => StatorValueInner::Undefined,
        StatorValueInner::Null => StatorValueInner::Null,
        StatorValueInner::Str(cs) => StatorValueInner::Str(cs.clone()),
        StatorValueInner::Object => StatorValueInner::Object,
        StatorValueInner::Function => StatorValueInner::Function,
        StatorValueInner::NativeFunctionValue(f) => {
            StatorValueInner::NativeFunctionValue(Rc::clone(f))
        }
        StatorValueInner::Array => StatorValueInner::Array,
        StatorValueInner::Date => StatorValueInner::Date,
        StatorValueInner::RegExp => StatorValueInner::RegExp,
        StatorValueInner::Promise => StatorValueInner::Promise,
        StatorValueInner::Map => StatorValueInner::Map,
        StatorValueInner::Set => StatorValueInner::Set,
    };
    unsafe { (*tmpl).properties.insert(key_str, inner) };
}

/// Set the number of internal fields on objects created from this template.
///
/// Internal fields are opaque embedder-owned slots (similar to V8's
/// `SetInternalFieldCount`).  Does nothing if `tmpl` is null.
///
/// # Safety
/// `tmpl` must be a valid, live [`StatorObjectTemplate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_set_internal_field_count(
    tmpl: *mut StatorObjectTemplate,
    count: i32,
) {
    if !tmpl.is_null() {
        // SAFETY: caller guarantees `tmpl` is valid.
        unsafe { (*tmpl).internal_field_count = count };
    }
}

/// Return the internal field count configured on this template.
///
/// Returns `0` if `tmpl` is null.
///
/// # Safety
/// `tmpl` must be either null or a valid, live [`StatorObjectTemplate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_internal_field_count(
    tmpl: *const StatorObjectTemplate,
) -> i32 {
    if tmpl.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `tmpl` is valid.
    unsafe { (*tmpl).internal_field_count }
}

/// Create a new [`StatorObject`] instance from an object template.
///
/// The returned object has all properties defined on the template pre-installed.
/// Returns a null pointer if `tmpl` is null.  The caller owns the returned
/// pointer and must eventually pass it to [`stator_object_destroy`].
///
/// # Safety
/// `tmpl` must be either null or a valid, live [`StatorObjectTemplate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_template_new_instance(
    tmpl: *const StatorObjectTemplate,
) -> *mut StatorObject {
    if tmpl.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `tmpl` is valid.
    let isolate = unsafe { (*tmpl).isolate };
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let mut obj = JsObject::new();
    // Install template properties onto the new object.
    // SAFETY: `tmpl` is valid.
    for (key, val_inner) in unsafe { &(*tmpl).properties } {
        let js_val = stator_value_inner_to_jsvalue(val_inner);
        let _ = obj.set_property(key, js_val);
    }
    // SAFETY: isolate is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorObject {
        inner: obj,
        isolate,
    }))
}

// ── TryCatch ─────────────────────────────────────────────────────────────────

/// An opaque try-catch scope for catching exceptions across the FFI boundary.
///
/// Mirrors V8's `TryCatch` RAII scope.  While a try-catch scope is active,
/// any pending exception thrown via [`stator_isolate_throw_exception`] (or
/// raised during script execution) is captured and can be inspected via
/// [`stator_try_catch_has_caught`] and [`stator_try_catch_exception`].
///
/// Created by [`stator_try_catch_new`] and freed by
/// [`stator_try_catch_destroy`].
pub struct StatorTryCatch {
    isolate: *mut StatorIsolate,
    /// The exception value captured when the scope is checked, or null if no
    /// exception was caught.
    exception: *mut StatorValue,
    /// Whether an exception has been caught by this scope.
    has_caught: bool,
}

// SAFETY: `StatorTryCatch` is single-threaded; same rationale as `StatorIsolate`.
unsafe impl Send for StatorTryCatch {}

/// Create a new try-catch scope on `isolate`.
///
/// Returns a null pointer if `isolate` is null.  The caller must eventually
/// pass the returned pointer to [`stator_try_catch_destroy`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_new(isolate: *mut StatorIsolate) -> *mut StatorTryCatch {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(StatorTryCatch {
        isolate,
        exception: std::ptr::null_mut(),
        has_caught: false,
    }))
}

/// Return `true` if this try-catch scope has caught an exception.
///
/// This function checks the isolate for a pending exception and, if found,
/// captures it into the scope.  Returns `false` if `tc` is null.
///
/// # Safety
/// `tc` must be either null or a valid, live [`StatorTryCatch`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_has_caught(tc: *mut StatorTryCatch) -> bool {
    if tc.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `tc` is valid.
    let tc_ref = unsafe { &mut *tc };
    if tc_ref.has_caught {
        return true;
    }
    // Check if the isolate has a pending exception we should capture.
    if !tc_ref.isolate.is_null() {
        // SAFETY: isolate is valid.
        let pending = unsafe { (*tc_ref.isolate).pending_exception.take() };
        if let Some(exc) = pending {
            tc_ref.exception = exc;
            tc_ref.has_caught = true;
        }
    }
    tc_ref.has_caught
}

/// Return the caught exception, or a null pointer if none was caught.
///
/// The returned pointer is owned by the try-catch scope and remains valid
/// until [`stator_try_catch_reset`] or [`stator_try_catch_destroy`] is called.
/// The caller must **not** call [`stator_value_destroy`] on it.
///
/// # Safety
/// `tc` must be either null or a valid, live [`StatorTryCatch`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_exception(tc: *const StatorTryCatch) -> *mut StatorValue {
    if tc.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `tc` is valid.
    unsafe { (*tc).exception }
}

/// Reset the try-catch scope, clearing any caught exception.
///
/// After this call, [`stator_try_catch_has_caught`] returns `false` and
/// [`stator_try_catch_exception`] returns null until a new exception is caught.
///
/// # Safety
/// `tc` must be either null or a valid, live [`StatorTryCatch`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_reset(tc: *mut StatorTryCatch) {
    if !tc.is_null() {
        // SAFETY: caller guarantees `tc` is valid.
        unsafe {
            (*tc).exception = std::ptr::null_mut();
            (*tc).has_caught = false;
        }
    }
}

/// Destroy a try-catch scope previously created with [`stator_try_catch_new`].
///
/// If the scope holds a caught exception that has not been cleared via
/// [`stator_try_catch_reset`], the exception value is **not** destroyed (the
/// caller retains ownership of exception values passed to
/// [`stator_isolate_throw_exception`]).
///
/// Does nothing if `tc` is null.
///
/// # Safety
/// `tc` must be a non-null pointer returned by [`stator_try_catch_new`] and
/// must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_destroy(tc: *mut StatorTryCatch) {
    if !tc.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(tc) });
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

// ── WebAssembly FFI ───────────────────────────────────────────────────────────

/// An opaque handle wrapping a compiled WebAssembly module.
///
/// Created by [`stator_wasm_compile`] and freed by
/// [`stator_wasm_module_destroy`].
pub struct StatorWasmModule {
    engine: WasmEngine,
    module: WasmModule,
}

// SAFETY: `StatorWasmModule` is single-threaded; the embedder is responsible
// for synchronisation if passed across threads.
unsafe impl Send for StatorWasmModule {}

/// An opaque handle wrapping a live WebAssembly module instance.
///
/// Created by [`stator_wasm_instantiate`] and freed by
/// [`stator_wasm_instance_destroy`].
pub struct StatorWasmInstance {
    instance: WasmInstance,
}

// SAFETY: `StatorWasmInstance` is single-threaded; the embedder is responsible
// for synchronisation if passed across threads.
unsafe impl Send for StatorWasmInstance {}

/// Compile a WebAssembly binary into a [`StatorWasmModule`].
///
/// Returns a null pointer if `isolate` or `bytes` is null, `len` is zero, or
/// if the bytes do not represent a valid WebAssembly module.
///
/// The returned pointer must eventually be passed to
/// [`stator_wasm_module_destroy`] to free all associated resources.
///
/// # Safety
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `bytes` must be valid for reads of `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_compile(
    isolate: *mut StatorIsolate,
    bytes: *const u8,
    len: usize,
) -> *mut StatorWasmModule {
    if isolate.is_null() || bytes.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `bytes` is valid for `len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(bytes, len) };
    let engine = WasmEngine::new();
    match WasmModule::from_bytes(&engine, slice) {
        Ok(module) => Box::into_raw(Box::new(StatorWasmModule { engine, module })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a [`StatorWasmModule`] previously returned by [`stator_wasm_compile`].
///
/// Does nothing when `module` is null.
///
/// # Safety
/// `module` must be either null or a valid pointer returned by
/// [`stator_wasm_compile`] that has not been freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_module_destroy(module: *mut StatorWasmModule) {
    if !module.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in `stator_wasm_compile`.
        drop(unsafe { Box::from_raw(module) });
    }
}

/// Instantiate a compiled [`StatorWasmModule`] into a live
/// [`StatorWasmInstance`].
///
/// `ctx` is accepted for future use and may be null.  `imports` is reserved for
/// future use and **must** be null; passing a non-null value is currently
/// ignored.
///
/// Returns a null pointer if `module` is null or if instantiation fails (e.g.
/// the module requires imports that are not provided).
///
/// The returned pointer must eventually be passed to
/// [`stator_wasm_instance_destroy`].
///
/// # Safety
/// - `module` must be a non-null, valid pointer to a live
///   [`StatorWasmModule`].
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - `imports` must be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_instantiate(
    module: *mut StatorWasmModule,
    _ctx: *mut StatorContext,
    _imports: *const c_void,
) -> *mut StatorWasmInstance {
    if module.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `module` is valid.
    let m = unsafe { &*module };
    match WasmInstance::new(&m.engine, &m.module) {
        Ok(instance) => Box::into_raw(Box::new(StatorWasmInstance { instance })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a [`StatorWasmInstance`] previously returned by
/// [`stator_wasm_instantiate`].
///
/// Does nothing when `instance` is null.
///
/// # Safety
/// `instance` must be either null or a valid pointer returned by
/// [`stator_wasm_instantiate`] that has not been freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_instance_destroy(instance: *mut StatorWasmInstance) {
    if !instance.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in `stator_wasm_instantiate`.
        drop(unsafe { Box::from_raw(instance) });
    }
}

/// Return a null-terminated array of export names from `instance`.
///
/// Each element in the array is a heap-allocated, null-terminated UTF-8 string.
/// The array itself is terminated by a null pointer.  The caller owns the
/// returned array and must free it with [`stator_wasm_exports_destroy`] when
/// it is no longer needed.
///
/// Returns a null pointer if `instance` is null.
///
/// # Safety
/// `instance` must be either null or a valid, live [`StatorWasmInstance`]
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_instance_exports(
    instance: *mut StatorWasmInstance,
) -> *mut *mut c_char {
    if instance.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `instance` is valid.
    let inst = unsafe { &mut *instance };
    let names = inst.instance.export_names();
    // Build a null-terminated array of heap-allocated C strings.
    let mut ptrs: Vec<*mut c_char> = names
        .into_iter()
        .map(|n| {
            // Silently replace any embedded NUL bytes to satisfy CString.
            let safe = n.replace('\0', "");
            // SAFETY: `safe` contains no NUL bytes after the replacement above.
            unsafe { CString::from_vec_unchecked(safe.into_bytes()) }.into_raw()
        })
        .collect();
    ptrs.push(std::ptr::null_mut()); // null terminator
    let boxed = ptrs.into_boxed_slice();
    Box::into_raw(boxed) as *mut *mut c_char
}

/// Free the export-name array returned by [`stator_wasm_instance_exports`].
///
/// Does nothing when `exports` is null.
///
/// # Safety
/// `exports` must be either null or a valid pointer returned by
/// [`stator_wasm_instance_exports`] that has not been freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_exports_destroy(exports: *mut *mut c_char) {
    if exports.is_null() {
        return;
    }
    // Count elements and free each string.
    let mut count = 0usize;
    loop {
        // SAFETY: we produced this array in `stator_wasm_instance_exports`.
        let ptr = unsafe { *exports.add(count) };
        if ptr.is_null() {
            break;
        }
        // SAFETY: each non-null string was created by `CString::into_raw`.
        drop(unsafe { CString::from_raw(ptr) });
        count += 1;
    }
    // Reconstruct the boxed slice (length includes the null terminator) and
    // drop it to free the array storage.
    // SAFETY: this slice was produced by `Box::into_raw(boxed)` in
    // `stator_wasm_instance_exports` with `count + 1` elements.
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(exports, count + 1)) });
}

/// Call an exported WebAssembly function by name.
///
/// `args` is an array of `args_len` pointers to [`StatorValue`] handles used
/// as Wasm arguments.  Each null element is treated as `i32(0)`.  A null
/// `args` pointer with `args_len == 0` is valid and means no arguments.
///
/// Returns a new [`StatorValue`] owned by the caller (free with
/// [`stator_value_destroy`]), or a null pointer if any required pointer
/// parameter is null, the named export does not exist, the call traps, or the
/// first result cannot be represented as a [`StatorValue`].  Void (zero-result)
/// functions return `undefined`.
///
/// # Safety
/// - `instance` must be a non-null, valid pointer to a live
///   [`StatorWasmInstance`].
/// - `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
/// - `name` must be a valid, null-terminated C string.
/// - `args` must be valid for reads of `args_len` pointers; each non-null
///   element must be a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_instance_call(
    instance: *mut StatorWasmInstance,
    isolate: *mut StatorIsolate,
    name: *const c_char,
    args: *const *const StatorValue,
    args_len: usize,
) -> *mut StatorValue {
    if instance.is_null() || isolate.is_null() || name.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `name` is a valid C string.
    let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();

    // Convert StatorValue* arguments to JsValue.
    let js_args: Vec<JsValue> = if args.is_null() || args_len == 0 {
        vec![]
    } else {
        // SAFETY: caller guarantees `args` is valid for `args_len` pointers.
        let ptrs = unsafe { std::slice::from_raw_parts(args, args_len) };
        ptrs.iter()
            .map(|&p| {
                if p.is_null() {
                    JsValue::Smi(0)
                } else {
                    // SAFETY: each non-null pointer is a valid StatorValue.
                    stator_value_inner_to_jsvalue(unsafe { &(*p).inner })
                }
            })
            .collect()
    };

    // SAFETY: caller guarantees `instance` is valid.
    let inst = unsafe { &mut *instance };
    match inst.instance.call_with_js_values(&name_str, &js_args) {
        Ok(jsval) => {
            let inner = jsvalue_to_stator_value_inner(&jsval);
            // SAFETY: `isolate` is valid.
            unsafe { (*isolate).live_objects += 1 };
            Box::into_raw(Box::new(StatorValue { inner, isolate }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CDP WebSocket server
// ─────────────────────────────────────────────────────────────────────────────

use stator_core::inspector::cdp::CdpServer;
use stator_core::inspector::debugger::{DebugAction, Debugger};
use stator_core::interpreter::{attach_debugger, detach_debugger};
/// An opaque handle to a CDP WebSocket server.
///
/// Created with [`stator_cdp_server_create`] and freed with
/// [`stator_cdp_server_destroy`].
pub struct StatorCdpServer {
    server: CdpServer,
}

/// Bind a CDP WebSocket server to `127.0.0.1:<port>`.
///
/// Passing `port = 0` lets the OS choose a free port; use
/// [`stator_cdp_server_local_port`] to discover the actual port.
///
/// Returns a non-null handle on success, or null on failure (e.g. port already
/// in use).  The handle must eventually be freed with
/// [`stator_cdp_server_destroy`].
#[unsafe(no_mangle)]
pub extern "C" fn stator_cdp_server_create(port: u16) -> *mut StatorCdpServer {
    let addr = format!("127.0.0.1:{port}");
    match CdpServer::bind(addr) {
        Ok(server) => Box::into_raw(Box::new(StatorCdpServer { server })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Return the TCP port that `server` is bound to.
///
/// # Safety
/// `server` must be a non-null pointer returned by [`stator_cdp_server_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_cdp_server_local_port(server: *const StatorCdpServer) -> u16 {
    if server.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `server` is valid.
    unsafe { &*server }
        .server
        .local_addr()
        .map(|a| a.port())
        .unwrap_or(0)
}

/// Spawn a background OS thread that accepts and serves CDP connections in a
/// loop, transferring ownership of `server` to the new thread.
///
/// After this call the `server` pointer is **consumed** and must **not** be
/// passed to [`stator_cdp_server_destroy`] or any other function.  The
/// background thread runs until the process exits.  Any per-connection errors
/// are silently ignored.
///
/// Does nothing if `server` is null.
///
/// # Safety
/// `server` must be null or a valid, uniquely-owned pointer returned by
/// [`stator_cdp_server_create`] that has not already been consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_cdp_server_run_background(server: *mut StatorCdpServer) {
    if server.is_null() {
        return;
    }
    // Transfer ownership to the background thread.
    // SAFETY: caller guarantees `server` is a valid, uniquely-owned pointer.
    let boxed = unsafe { Box::from_raw(server) };
    std::thread::spawn(move || {
        let _ = boxed.server.accept_loop();
    });
}

/// Free a CDP server returned by [`stator_cdp_server_create`].
///
/// Does nothing if `server` is null.
///
/// # Safety
/// `server` must be null or a valid pointer returned by
/// [`stator_cdp_server_create`].  Must not be called more than once for the
/// same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_cdp_server_destroy(server: *mut StatorCdpServer) {
    if !server.is_null() {
        // SAFETY: caller guarantees `server` is a valid, uniquely-owned pointer.
        drop(unsafe { Box::from_raw(server) });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug session (breakpoint / step / inspect)
// ─────────────────────────────────────────────────────────────────────────────

/// An opaque handle to an interactive debugging session.
///
/// A debug session owns both the compiled [`InterpreterFrame`] and an attached
/// [`Debugger`], allowing the host to:
///
/// 1. Set breakpoints before execution starts.
/// 2. Start execution with [`stator_debug_session_run`] — the call returns
///    `true` when a breakpoint is hit, leaving the session in a paused state.
/// 3. Inspect global variables with [`stator_debug_session_get_global_string`].
/// 4. Resume with [`stator_debug_session_resume`].
/// 5. Repeat until `run` / `resume` return `false` (execution completed).
/// 6. Retrieve the final result with [`stator_debug_session_result`].
///
/// Created with [`stator_debug_session_create`] and freed with
/// [`stator_debug_session_destroy`].
pub struct StatorDebugSession {
    frame: InterpreterFrame,
    dbg: Rc<RefCell<Debugger>>,
    /// True while the session is paused at a breakpoint / step.
    paused: bool,
    /// Line number reported at the last pause (1-based; 0 if unknown).
    pause_line: u32,
    /// Stored result after the session runs to completion.
    result: Option<JsValue>,
    /// Isolate used to wrap the final result as a `StatorValue`.
    isolate: *mut StatorIsolate,
}

/// Create a new debug session for `script`.
///
/// Compiles `script` and prepares an interpreter frame; does **not** start
/// execution.  Call [`stator_debug_session_set_breakpoint_at_line`] to
/// install breakpoints, then [`stator_debug_session_run`] to begin execution.
///
/// Returns null if `script` or `ctx` is null, if `script` has a compile
/// error, or if allocation fails.  The returned handle must eventually be
/// freed with [`stator_debug_session_destroy`].
///
/// # Safety
/// - `script` must be a non-null pointer returned by [`stator_script_compile`]
///   with no compile error.
/// - `ctx` must be a non-null pointer to a live [`StatorContext`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_create(
    script: *const StatorScript,
    ctx: *mut StatorContext,
) -> *mut StatorDebugSession {
    if script.is_null() || ctx.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees both pointers are valid and live.
    let script_ref = unsafe { &*script };
    let ctx_ref = unsafe { &*ctx };
    let bytecodes = match &script_ref.bytecodes {
        Some(bc) => bc.clone(),
        None => return std::ptr::null_mut(),
    };
    let frame =
        InterpreterFrame::new_with_globals(Rc::new(bytecodes), vec![], Rc::clone(&ctx_ref.globals));
    let dbg = Rc::new(RefCell::new(Debugger::new()));
    Box::into_raw(Box::new(StatorDebugSession {
        frame,
        dbg,
        paused: false,
        pause_line: 0,
        result: None,
        isolate: ctx_ref._isolate,
    }))
}

/// Install a breakpoint at the given 1-based source line in `session`.
///
/// Returns `true` if the breakpoint was successfully mapped to a bytecode
/// offset, `false` otherwise (e.g. the line has no executable code).
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_set_breakpoint_at_line(
    session: *mut StatorDebugSession,
    line: u32,
) -> bool {
    if session.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `session` is valid.
    let s = unsafe { &mut *session };
    s.dbg
        .borrow_mut()
        .set_breakpoint_at_line(&s.frame.bytecode_array, line)
        .is_some()
}

/// Run (or continue) the session until a breakpoint is hit or execution
/// completes.
///
/// Returns `true` if execution paused at a breakpoint (the session is now in
/// a paused state and the caller may inspect variables).  Returns `false` when
/// execution finished normally or with an uncaught exception; the final result
/// is available via [`stator_debug_session_result`].
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_run(session: *mut StatorDebugSession) -> bool {
    if session.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `session` is valid.
    let s = unsafe { &mut *session };
    if s.result.is_some() {
        // Already completed.
        return false;
    }

    attach_debugger(Rc::clone(&s.dbg));
    let outcome = Interpreter::run(&mut s.frame);
    detach_debugger();

    match outcome {
        Err(stator_core::error::StatorError::DebuggerPaused { .. }) => {
            s.paused = true;
            // Record the 1-based line from the debugger's last pause location.
            s.pause_line = s.dbg.borrow().last_pause_line();
            true
        }
        Ok(val) => {
            s.paused = false;
            s.result = Some(val);
            false
        }
        Err(_) => {
            s.paused = false;
            s.result = Some(JsValue::Undefined);
            false
        }
    }
}

/// Returns `true` if the session is currently paused at a breakpoint.
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_is_paused(
    session: *const StatorDebugSession,
) -> bool {
    if session.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `session` is valid.
    unsafe { &*session }.paused
}

/// Return the 1-based source line at which execution is currently paused, or
/// 0 if the session is not paused.
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_pause_line(
    session: *const StatorDebugSession,
) -> u32 {
    if session.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `session` is valid.
    let s = unsafe { &*session };
    if s.paused {
        s.dbg.borrow().last_pause_line()
    } else {
        0
    }
}

/// Write the string representation of the global variable `name` into `buf`
/// (up to `buf_len − 1` bytes, always NUL-terminated).
///
/// Returns the number of bytes written (excluding the NUL), or `-1` if the
/// variable does not exist or the session / buffer pointer is null.
///
/// # Safety
/// - `session` must be a non-null pointer returned by
///   [`stator_debug_session_create`].
/// - `name` must be a valid, null-terminated C string.
/// - `buf` must be valid for writes of at least `buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_get_global_string(
    session: *const StatorDebugSession,
    name: *const c_char,
    buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    if session.is_null() || name.is_null() || buf.is_null() || buf_len == 0 {
        return -1;
    }
    // SAFETY: caller guarantees both pointers are valid.
    let s = unsafe { &*session };
    let key = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    let globals = s.frame.global_env.borrow();
    let value = match globals.get(key.as_ref()) {
        Some(v) => v,
        None => return -1,
    };
    let repr = match value.to_js_string() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let bytes = repr.as_bytes();
    let copy_len = bytes.len().min(buf_len - 1);
    // SAFETY: caller guarantees `buf` is valid for `buf_len` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buf, copy_len);
        *buf.add(copy_len) = 0;
    }
    copy_len as i32
}

/// Resume a paused session with a "continue" action (run until the next
/// breakpoint or completion).
///
/// Equivalent to calling [`stator_debug_session_run`] after applying a
/// `Continue` action.  Returns `true` if execution pauses again, `false` if
/// it completes.
///
/// Does nothing and returns `false` if the session is not paused or `session`
/// is null.
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_resume(session: *mut StatorDebugSession) -> bool {
    if session.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `session` is valid.
    let s = unsafe { &mut *session };
    if !s.paused {
        return false;
    }
    s.dbg.borrow_mut().apply_action(DebugAction::Continue);
    s.paused = false;
    // SAFETY: `session` is valid (same pointer).
    unsafe { stator_debug_session_run(session) }
}

/// Return a new [`StatorValue`] containing the final result of a completed
/// debug session, or null if the session has not yet completed.
///
/// The returned value must be freed with [`stator_value_destroy`].
///
/// # Safety
/// `session` must be a non-null pointer returned by
/// [`stator_debug_session_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_result(
    session: *const StatorDebugSession,
) -> *mut StatorValue {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `session` is valid.
    let s = unsafe { &*session };
    let val = match &s.result {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let inner = jsvalue_to_stator_value_inner(val);
    let isolate = s.isolate;
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: isolate is known to be valid for the session's lifetime.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorValue { inner, isolate }))
}

/// Free a debug session returned by [`stator_debug_session_create`].
///
/// Does nothing if `session` is null.
///
/// # Safety
/// `session` must be null or a valid pointer returned by
/// [`stator_debug_session_create`].  Must not be called more than once for
/// the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_debug_session_destroy(session: *mut StatorDebugSession) {
    if !session.is_null() {
        // SAFETY: caller guarantees `session` is a valid, uniquely-owned pointer.
        drop(unsafe { Box::from_raw(session) });
    }
}

// ── DOM integration (object wrapping, interceptors, weak refs) ────────────────

/// C-callable named-property getter callback.
///
/// Receives the property name, an embedder-data pointer, and an out-pointer
/// for the result.  Returns `true` if the interceptor handled the access
/// (and wrote a value into `*out`), `false` to fall through.
///
/// # Safety
/// - `name` must be a valid, null-terminated C string.
/// - `out` must be a valid, writable pointer if the callback returns `true`.
type StatorDomNamedGetterCb = unsafe extern "C" fn(
    name: *const c_char,
    data: *mut c_void,
    out: *mut *mut StatorValue,
) -> bool;

/// C-callable named-property setter callback.
///
/// Returns `true` if the interceptor handled the write.
type StatorDomNamedSetterCb =
    unsafe extern "C" fn(name: *const c_char, value: *const StatorValue, data: *mut c_void) -> bool;

/// C-callable indexed-property getter callback.
///
/// Returns `true` if the interceptor handled the access.
type StatorDomIndexedGetterCb =
    unsafe extern "C" fn(index: u32, data: *mut c_void, out: *mut *mut StatorValue) -> bool;

/// C-callable indexed-property setter callback.
///
/// Returns `true` if the interceptor handled the write.
type StatorDomIndexedSetterCb =
    unsafe extern "C" fn(index: u32, value: *const StatorValue, data: *mut c_void) -> bool;

/// C-callable weak-reference callback, invoked when the wrapped DOM object is
/// garbage-collected.
type StatorDomWeakCb = unsafe extern "C" fn(data: *mut c_void);

/// An opaque handle to a DOM object wrapper.
///
/// Created by [`stator_dom_object_wrap_new`] and freed by
/// [`stator_dom_object_wrap_destroy`].  Stores opaque embedder pointers in
/// *internal fields* and routes property access through optional interceptors.
pub struct StatorDomObjectWrap {
    inner: DomObjectWrap,
    isolate: *mut StatorIsolate,
}

// SAFETY: `StatorDomObjectWrap` is single-threaded; see [`StatorValue`].
unsafe impl Send for StatorDomObjectWrap {}

/// Create a new DOM object wrapper with `field_count` internal-field slots.
///
/// Returns a null pointer if `isolate` is null or `field_count` exceeds the
/// engine's per-object limit (currently 16).  The caller must eventually pass
/// the returned pointer to [`stator_dom_object_wrap_destroy`].
///
/// # Safety
/// `isolate` must be a non-null, valid pointer to a live [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_new(
    isolate: *mut StatorIsolate,
    field_count: u32,
) -> *mut StatorDomObjectWrap {
    if isolate.is_null() || field_count as usize > stator_core::dom::MAX_INTERNAL_FIELDS {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorDomObjectWrap {
        inner: DomObjectWrap::new(field_count as usize),
        isolate,
    }))
}

/// Destroy a DOM object wrapper previously created with
/// [`stator_dom_object_wrap_new`].
///
/// # Safety
/// `wrap` must be a non-null pointer returned by [`stator_dom_object_wrap_new`]
/// and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_destroy(wrap: *mut StatorDomObjectWrap) {
    if !wrap.is_null() {
        // SAFETY: caller guarantees `wrap` is valid.
        let iso = unsafe { (*wrap).isolate };
        if !iso.is_null() {
            // SAFETY: `iso` is valid for the lifetime of `wrap`.
            unsafe { (*iso).live_objects = (*iso).live_objects.saturating_sub(1) };
        }
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(wrap) });
    }
}

/// Return the number of internal-field slots on `wrap`.
///
/// Returns `0` when `wrap` is null.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_internal_field_count(
    wrap: *const StatorDomObjectWrap,
) -> u32 {
    if wrap.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).inner.internal_field_count() as u32 }
}

/// Store an opaque embedder pointer in internal field `index`.
///
/// Does nothing when `wrap` is null or `index` is out of range.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `ptr` is an opaque value; the engine never dereferences it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_internal_field(
    wrap: *mut StatorDomObjectWrap,
    index: u32,
    ptr: *mut c_void,
) {
    if wrap.is_null() {
        return;
    }
    let idx = index as usize;
    // SAFETY: caller guarantees `wrap` is valid.
    let inner = unsafe { &mut (*wrap).inner };
    if idx < inner.internal_field_count() {
        inner.set_internal_field(idx, ptr);
    }
}

/// Retrieve the opaque embedder pointer from internal field `index`.
///
/// Returns a null pointer when `wrap` is null or `index` is out of range.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_get_internal_field(
    wrap: *const StatorDomObjectWrap,
    index: u32,
) -> *mut c_void {
    if wrap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).inner.get_internal_field(index as usize) }
}

/// Install a named-property getter interceptor on `wrap`.
///
/// The callback is invoked for every named-property read (e.g. `element.id`).
/// Does nothing when `wrap` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `cb` must remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_named_getter(
    wrap: *mut StatorDomObjectWrap,
    cb: StatorDomNamedGetterCb,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let inner = unsafe { &mut (*wrap).inner };
    let isolate = unsafe { (*wrap).isolate };
    inner.set_named_handler(
        NamedPropertyHandlerConfig::builder()
            .getter(move |name, data| {
                let c_name = CString::new(name).ok()?;
                let mut out: *mut StatorValue = std::ptr::null_mut();
                // SAFETY: `cb` is a C function pointer that the caller
                // guarantees remains valid.  `c_name` is a valid C string.
                let handled = unsafe { cb(c_name.as_ptr(), data, &mut out) };
                if handled && !out.is_null() {
                    // SAFETY: `out` was written by the callback.
                    let val = unsafe { Box::from_raw(out) };
                    Some(stator_value_inner_to_jsvalue(&val.inner))
                } else {
                    None
                }
            })
            .build(),
    );
    let _ = isolate; // keep variable used for future extensions
}

/// Install a named-property setter interceptor on `wrap`.
///
/// The callback is invoked for every named-property write.
/// Does nothing when `wrap` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `cb` must remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_named_setter(
    wrap: *mut StatorDomObjectWrap,
    cb: StatorDomNamedSetterCb,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let inner = unsafe { &mut (*wrap).inner };
    inner.set_named_handler(
        NamedPropertyHandlerConfig::builder()
            .setter(move |name, value, data| {
                if let Ok(c_name) = CString::new(name) {
                    let c_val = StatorValue {
                        inner: jsvalue_to_stator_value_inner(value),
                        isolate: std::ptr::null_mut(),
                    };
                    // SAFETY: `cb` is guaranteed valid by the caller.
                    unsafe { cb(c_name.as_ptr(), &c_val, data) }
                } else {
                    false
                }
            })
            .build(),
    );
}

/// Install an indexed-property getter interceptor on `wrap`.
///
/// The callback is invoked for every indexed-property read (e.g.
/// `nodeList[0]`).  Does nothing when `wrap` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `cb` must remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_indexed_getter(
    wrap: *mut StatorDomObjectWrap,
    cb: StatorDomIndexedGetterCb,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let inner = unsafe { &mut (*wrap).inner };
    inner.set_indexed_handler(
        IndexedPropertyHandlerConfig::builder()
            .getter(move |index, data| {
                let mut out: *mut StatorValue = std::ptr::null_mut();
                // SAFETY: `cb` is guaranteed valid by the caller.
                let handled = unsafe { cb(index, data, &mut out) };
                if handled && !out.is_null() {
                    // SAFETY: `out` was written by the callback.
                    let val = unsafe { Box::from_raw(out) };
                    Some(stator_value_inner_to_jsvalue(&val.inner))
                } else {
                    None
                }
            })
            .build(),
    );
}

/// Install an indexed-property setter interceptor on `wrap`.
///
/// The callback is invoked for every indexed-property write.
/// Does nothing when `wrap` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `cb` must remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_indexed_setter(
    wrap: *mut StatorDomObjectWrap,
    cb: StatorDomIndexedSetterCb,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let inner = unsafe { &mut (*wrap).inner };
    inner.set_indexed_handler(
        IndexedPropertyHandlerConfig::builder()
            .setter(move |index, value, data| {
                let c_val = StatorValue {
                    inner: jsvalue_to_stator_value_inner(value),
                    isolate: std::ptr::null_mut(),
                };
                // SAFETY: `cb` is guaranteed valid by the caller.
                unsafe { cb(index, &c_val, data) }
            })
            .build(),
    );
}

/// An opaque handle to a DOM weak reference.
///
/// Created by [`stator_dom_weak_ref_new`] and freed by
/// [`stator_dom_weak_ref_destroy`].
pub struct StatorDomWeakRef {
    inner: DomWeakRef,
}

// SAFETY: `StatorDomWeakRef` is single-threaded; see [`StatorValue`].
unsafe impl Send for StatorDomWeakRef {}

/// Create a new weak reference for the given DOM object wrapper.
///
/// When the weak reference is later invoked (by the GC or explicitly), `cb`
/// will be called with the embedder data pointer from internal field 0.
///
/// Returns a null pointer if `wrap` is null.  The caller must eventually pass
/// the returned pointer to [`stator_dom_weak_ref_destroy`].
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `cb` must remain valid until the weak callback fires or the weak ref is
///   destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_weak_ref_new(
    wrap: *const StatorDomObjectWrap,
    cb: StatorDomWeakCb,
) -> *mut StatorDomWeakRef {
    if wrap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let dom_wrap = unsafe { &(*wrap).inner };
    let weak = DomWeakRef::new(dom_wrap, move |data| {
        // SAFETY: `cb` is guaranteed valid by the caller.
        unsafe { cb(data) };
    });
    Box::into_raw(Box::new(StatorDomWeakRef { inner: weak }))
}

/// Return `true` if the weak reference has not yet been invalidated.
///
/// Returns `false` when `weak` is null.
///
/// # Safety
/// `weak` must be either null or a valid, live [`StatorDomWeakRef`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_weak_ref_is_alive(weak: *const StatorDomWeakRef) -> bool {
    if weak.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `weak` is valid.
    unsafe { (*weak).inner.is_alive() }
}

/// Fire the weak callback and mark the reference as dead.
///
/// This is idempotent: calling it after the callback has already fired is a
/// no-op.  Does nothing when `weak` is null.
///
/// # Safety
/// `weak` must be either null or a valid, live [`StatorDomWeakRef`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_weak_ref_invoke(weak: *const StatorDomWeakRef) {
    if !weak.is_null() {
        // SAFETY: caller guarantees `weak` is valid.
        unsafe { (*weak).inner.invoke_callback() };
    }
}

/// Reset the weak reference without invoking the callback.
///
/// Does nothing when `weak` is null.
///
/// # Safety
/// `weak` must be either null or a valid, live [`StatorDomWeakRef`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_weak_ref_clear(weak: *const StatorDomWeakRef) {
    if !weak.is_null() {
        // SAFETY: caller guarantees `weak` is valid.
        unsafe { (*weak).inner.clear() };
    }
}

/// Destroy a weak reference previously created with [`stator_dom_weak_ref_new`].
///
/// If the callback has not yet been invoked it will **not** be called; use
/// [`stator_dom_weak_ref_invoke`] first if you need the callback to fire.
///
/// # Safety
/// `weak` must be a non-null pointer returned by [`stator_dom_weak_ref_new`]
/// and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_weak_ref_destroy(weak: *mut StatorDomWeakRef) {
    if !weak.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(weak) });
    }
}

// ── Event loop FFI ─────────────────────────────────────────────────────────────

use stator_core::builtins::promise::MicrotaskQueue;
use stator_core::event_loop::{DefaultCallbacks, EmbedderCallbacks, EventLoop, TimerHandle};

/// Opaque event loop handle.
pub struct StatorEventLoop {
    inner: EventLoop,
}

/// C function pointer types for embedder callbacks.
type StatorPostTaskFn = unsafe extern "C" fn(task_data: *mut c_void);
type StatorPostDelayedTaskFn = unsafe extern "C" fn(task_data: *mut c_void, delay_secs: f64);
type StatorRequestIdleCallbackFn = unsafe extern "C" fn(cb_data: *mut c_void, idle_time: f64);
type StatorMonotonicTimeFn = unsafe extern "C" fn() -> f64;

/// C-compatible vtable for embedder callbacks.
#[repr(C)]
pub struct StatorEmbedderCallbacks {
    /// Post a task to the embedder's main thread.
    pub post_task: StatorPostTaskFn,
    /// Post a delayed task.
    pub post_delayed_task: StatorPostDelayedTaskFn,
    /// Request an idle callback.
    pub request_idle_callback: StatorRequestIdleCallbackFn,
    /// Monotonic time source.
    pub monotonic_time: StatorMonotonicTimeFn,
}

/// Adapter that wraps C function pointers into the [`EmbedderCallbacks`] trait.
struct FfiCallbacks {
    vtable: StatorEmbedderCallbacks,
}

impl EmbedderCallbacks for FfiCallbacks {
    fn post_task(&self, _task: Box<dyn FnOnce()>) {
        // SAFETY: embedder guarantees the function pointer is valid.
        unsafe { (self.vtable.post_task)(std::ptr::null_mut()) };
    }

    fn post_delayed_task(&self, _task: Box<dyn FnOnce()>, delay_secs: f64) {
        // SAFETY: embedder guarantees the function pointer is valid.
        unsafe { (self.vtable.post_delayed_task)(std::ptr::null_mut(), delay_secs) };
    }

    fn request_idle_callback(&self, _cb: Box<dyn FnOnce(f64)>) {
        // SAFETY: embedder guarantees the function pointer is valid.
        unsafe { (self.vtable.request_idle_callback)(std::ptr::null_mut(), 0.0) };
    }

    fn monotonic_time(&self) -> f64 {
        // SAFETY: embedder guarantees the function pointer is valid.
        unsafe { (self.vtable.monotonic_time)() }
    }
}

/// Create a new event loop with default (no-op) callbacks.
///
/// The returned pointer must be freed with [`stator_event_loop_destroy`].
#[unsafe(no_mangle)]
pub extern "C" fn stator_event_loop_create() -> *mut StatorEventLoop {
    Box::into_raw(Box::new(StatorEventLoop {
        inner: EventLoop::new(MicrotaskQueue::new(), Box::new(DefaultCallbacks)),
    }))
}

/// Create a new event loop with embedder-provided callbacks.
///
/// # Safety
/// All function pointers in `cbs` must be valid for the lifetime of the
/// returned event loop.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_create_with_callbacks(
    cbs: StatorEmbedderCallbacks,
) -> *mut StatorEventLoop {
    let callbacks = FfiCallbacks { vtable: cbs };
    Box::into_raw(Box::new(StatorEventLoop {
        inner: EventLoop::new(MicrotaskQueue::new(), Box::new(callbacks)),
    }))
}

/// Destroy an event loop.
///
/// # Safety
/// `el` must be a non-null pointer returned by `stator_event_loop_create*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_destroy(el: *mut StatorEventLoop) {
    if !el.is_null() {
        // SAFETY: pointer was created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(el) });
    }
}

/// Post a macrotask.  The provided C callback will be invoked with `data` on
/// the next turn of the event loop.
///
/// # Safety
/// `el` must be a valid event loop pointer.  `callback` must be a valid
/// function pointer.  `data` is passed through opaquely.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_post_task(
    el: *mut StatorEventLoop,
    callback: unsafe extern "C" fn(*mut c_void),
    data: *mut c_void,
) {
    if el.is_null() {
        return;
    }
    // SAFETY: caller guarantees `el` is valid.
    let el = unsafe { &*el };
    // Wrap the C callback+data into a Rust closure.
    // SAFETY: the caller guarantees `callback` and `data` remain valid until
    // the task executes.
    el.inner.post_task(Box::new(move || unsafe {
        callback(data);
    }));
}

/// Schedule a one-shot timer.  Returns the timer id (0 on null input).
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_set_timer(
    el: *mut StatorEventLoop,
    delay_secs: f64,
    callback: unsafe extern "C" fn(*mut c_void),
    data: *mut c_void,
) -> u64 {
    if el.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `el` is valid.
    let el = unsafe { &*el };
    let handle = el
        .inner
        .set_timer(delay_secs, Box::new(move || unsafe { callback(data) }));
    handle.id()
}

/// Cancel a previously scheduled timer.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_cancel_timer(el: *mut StatorEventLoop, timer_id: u64) {
    if el.is_null() {
        return;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }
        .inner
        .cancel_timer(TimerHandle::from_raw(timer_id));
}

/// Run one tick of the event loop.  Returns `true` if a macrotask ran.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_tick(el: *mut StatorEventLoop) -> bool {
    if el.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }.inner.tick()
}

/// Spin the event loop until idle.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_run_until_idle(el: *mut StatorEventLoop) {
    if el.is_null() {
        return;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }.inner.run_until_idle();
}

/// Drain pending microtasks.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_drain_microtasks(el: *mut StatorEventLoop) {
    if el.is_null() {
        return;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }.inner.drain_microtasks();
}

/// Returns `true` when the event loop has no pending work.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_is_idle(el: *mut StatorEventLoop) -> bool {
    if el.is_null() {
        return true;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }.inner.is_idle()
}

/// Returns the number of pending macrotasks.
///
/// # Safety
/// `el` must be a valid event loop pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_event_loop_pending_task_count(el: *mut StatorEventLoop) -> usize {
    if el.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `el` is valid.
    unsafe { &*el }.inner.pending_task_count()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── stator_object_has ─────────────────────────────────────────────────────

    #[test]
    fn test_object_has_existing_property_returns_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let key = c"x";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_set(obj, key.as_ptr(), val) };
        // SAFETY: `obj` and `key` are valid.
        assert!(unsafe { stator_object_has(obj, key.as_ptr()) });
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(val);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_object_has_missing_property_returns_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let key = c"nope";
        // SAFETY: `obj` and `key` are valid; property has not been set.
        assert!(!unsafe { stator_object_has(obj, key.as_ptr()) });
        // SAFETY: `obj` is non-null and live.
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_object_has_null_obj_returns_false() {
        let key = c"k";
        // SAFETY: null obj is documented to return false.
        assert!(!unsafe { stator_object_has(std::ptr::null(), key.as_ptr()) });
    }

    // ── stator_object_delete ──────────────────────────────────────────────────

    #[test]
    fn test_object_delete_existing_property() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 7.0) };
        let key = c"y";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_set(obj, key.as_ptr(), val) };
        // Verify the property exists before deletion.
        assert!(unsafe { stator_object_has(obj, key.as_ptr()) });
        // SAFETY: `obj` and `key` are valid.
        let deleted = unsafe { stator_object_delete(obj, key.as_ptr()) };
        assert!(deleted);
        // Property should no longer be present.
        assert!(!unsafe { stator_object_has(obj, key.as_ptr()) });
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(val);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_object_delete_missing_property_returns_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let key = c"nonexistent";
        // Deleting a property that never existed is a no-op success.
        let result = unsafe { stator_object_delete(obj, key.as_ptr()) };
        assert!(result);
        // SAFETY: `obj` is non-null and live.
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_object_delete_null_obj_returns_false() {
        let key = c"k";
        // SAFETY: null obj is documented to return false.
        assert!(!unsafe { stator_object_delete(std::ptr::null_mut(), key.as_ptr()) });
    }

    // ── stator_object_get_property_names ──────────────────────────────────────

    #[test]
    fn test_object_get_property_names_empty_object() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        // SAFETY: `obj` is non-null and live.
        let names = unsafe { stator_object_get_property_names(obj) };
        assert!(!names.is_null());
        // SAFETY: `names` is non-null and live.
        assert_eq!(unsafe { stator_property_names_count(names) }, 0);
        // SAFETY: `names` is non-null and live.
        unsafe { stator_property_names_destroy(names) };
        // SAFETY: `obj` is non-null and live.
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_object_get_property_names_returns_all_keys() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let v1 = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let v2 = unsafe { stator_value_new_number(iso.as_ptr(), 2.0) };
        let ka = c"alpha";
        let kb = c"beta";
        // SAFETY: all pointers are valid.
        unsafe {
            stator_object_set(obj, ka.as_ptr(), v1);
            stator_object_set(obj, kb.as_ptr(), v2);
        }
        // SAFETY: `obj` is valid.
        let names = unsafe { stator_object_get_property_names(obj) };
        assert!(!names.is_null());
        // SAFETY: `names` is non-null and live.
        let count = unsafe { stator_property_names_count(names) };
        assert_eq!(count, 2);
        // Collect the returned names and verify both keys are present.
        let mut got: Vec<String> = Vec::new();
        for i in 0..count {
            // SAFETY: `names` is valid; `i` is within range.
            let ptr = unsafe { stator_property_names_get(names, i) };
            assert!(!ptr.is_null());
            // SAFETY: `ptr` is a valid null-terminated string owned by `names`.
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
            got.push(s);
        }
        got.sort();
        assert_eq!(got, vec!["alpha", "beta"]);
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_property_names_destroy(names);
            stator_value_destroy(v1);
            stator_value_destroy(v2);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_object_get_property_names_after_delete() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let v = unsafe { stator_value_new_number(iso.as_ptr(), 99.0) };
        let key = c"temp";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_set(obj, key.as_ptr(), v) };
        // SAFETY: `obj` and `key` are valid.
        unsafe { stator_object_delete(obj, key.as_ptr()) };
        // SAFETY: `obj` is valid.
        let names = unsafe { stator_object_get_property_names(obj) };
        assert!(!names.is_null());
        // SAFETY: `names` is non-null and live.
        assert_eq!(unsafe { stator_property_names_count(names) }, 0);
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_property_names_destroy(names);
            stator_value_destroy(v);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_property_names_get_out_of_range_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        // SAFETY: `obj` is valid.
        let names = unsafe { stator_object_get_property_names(obj) };
        assert!(!names.is_null());
        // SAFETY: `names` is non-null and live; index 0 is out of range for an
        // empty snapshot.
        assert!(unsafe { stator_property_names_get(names, 0) }.is_null());
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_property_names_destroy(names);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_object_get_property_names_null_returns_null() {
        // SAFETY: null obj is documented to return null.
        let names = unsafe { stator_object_get_property_names(std::ptr::null()) };
        assert!(names.is_null());
    }

    #[test]
    fn test_property_names_count_null_returns_zero() {
        // SAFETY: null names is documented to return 0.
        assert_eq!(unsafe { stator_property_names_count(std::ptr::null()) }, 0);
    }

    #[test]
    fn test_property_names_get_null_names_returns_null() {
        // SAFETY: null names is documented to return null.
        assert!(unsafe { stator_property_names_get(std::ptr::null(), 0) }.is_null());
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

    // ── HandleScope ──────────────────────────────────────────────────────────

    #[test]
    fn test_handle_scope_new_returns_nonnull() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_handle_scope_new(iso.as_ptr()) };
        assert!(!scope.is_null());
        // SAFETY: `scope` is non-null and live; it is the innermost scope.
        unsafe { stator_handle_scope_close(scope) };
    }

    #[test]
    fn test_handle_scope_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let scope = unsafe { stator_handle_scope_new(std::ptr::null_mut()) };
        assert!(scope.is_null());
    }

    #[test]
    fn test_handle_scope_close_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_handle_scope_close(std::ptr::null_mut()) };
    }

    #[test]
    fn test_handle_scope_owns_values_and_destroys_on_close() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_handle_scope_new(iso.as_ptr()) };
        assert!(!scope.is_null());

        // Create two values while the scope is active — they are scope-owned.
        // SAFETY: `iso` is valid.
        let v1 = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let v2 = unsafe { stator_value_new_number(iso.as_ptr(), 2.0) };
        assert!(!v1.is_null());
        assert!(!v2.is_null());

        // Both should be readable before the scope is closed.
        // SAFETY: `v1` and `v2` are non-null and live.
        assert!((unsafe { stator_value_as_number(v1) } - 1.0).abs() < f64::EPSILON);
        assert!((unsafe { stator_value_as_number(v2) } - 2.0).abs() < f64::EPSILON);

        // live_objects should be 2.
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 2);

        // Closing the scope frees both values automatically.
        // SAFETY: `scope` is non-null and live; it is the innermost scope.
        unsafe { stator_handle_scope_close(scope) };

        // live_objects should be back to 0.
        // SAFETY: `iso` is valid.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    #[test]
    fn test_handle_scope_with_string_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_handle_scope_new(iso.as_ptr()) };

        let s = b"scoped\0";
        // SAFETY: `iso` is valid; `s` is valid for 6 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 6) };
        assert!(!val.is_null());

        // SAFETY: `val` is non-null and live.
        let ptr = unsafe { stator_value_as_string(val) };
        // SAFETY: returned pointer is valid while `val` is alive.
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "scoped");

        // Closing the scope frees the string value.
        // SAFETY: `scope` is non-null and live.
        unsafe { stator_handle_scope_close(scope) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    #[test]
    fn test_handle_scope_restores_previous_scope_on_close() {
        let iso = IsolateGuard::new();
        // Open outer scope.
        // SAFETY: `iso` is valid.
        let outer = unsafe { stator_handle_scope_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid; `outer` is the active scope.
        let _ = unsafe { stator_value_new_number(iso.as_ptr(), 10.0) };

        // Open inner scope — it shadows the outer.
        // SAFETY: `iso` is valid.
        let inner = unsafe { stator_handle_scope_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid; `inner` is the active scope.
        let _ = unsafe { stator_value_new_number(iso.as_ptr(), 20.0) };

        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 2);

        // Close inner scope — only the inner value is destroyed; outer scope is restored.
        // SAFETY: `inner` is non-null and live; it is the innermost scope.
        unsafe { stator_handle_scope_close(inner) };
        // Inner value is freed; outer value still lives (owned by outer scope).
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);
        // After closing inner, the outer scope is the active scope again.
        // SAFETY: `iso` is valid.
        assert_eq!(
            unsafe { (*iso.as_ptr()).active_handle_scope },
            outer as *mut StatorHandleScope
        );

        // Close outer scope — the outer value is destroyed.
        // SAFETY: `outer` is non-null and live; it is now the innermost scope.
        unsafe { stator_handle_scope_close(outer) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
        // No active scope remains.
        assert!(unsafe { (*iso.as_ptr()).active_handle_scope }.is_null());
    }

    #[test]
    fn test_values_without_scope_are_embedder_owned() {
        // When no scope is open, behavior is unchanged: the embedder must call
        // stator_value_destroy manually.
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid; no scope is active.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        assert!(!val.is_null());
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    // ── EscapableHandleScope ─────────────────────────────────────────────────

    #[test]
    fn test_escapable_handle_scope_new_returns_nonnull() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_escapable_handle_scope_new(iso.as_ptr()) };
        assert!(!scope.is_null());
        // SAFETY: `scope` is non-null and live.
        unsafe { stator_escapable_handle_scope_close(scope) };
    }

    #[test]
    fn test_escapable_handle_scope_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let scope = unsafe { stator_escapable_handle_scope_new(std::ptr::null_mut()) };
        assert!(scope.is_null());
    }

    #[test]
    fn test_escapable_handle_scope_close_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_escapable_handle_scope_close(std::ptr::null_mut()) };
    }

    #[test]
    fn test_escapable_handle_scope_escape_promotes_to_outer_scope() {
        let iso = IsolateGuard::new();
        // Open outer (plain) scope.
        // SAFETY: `iso` is valid.
        let outer = unsafe { stator_handle_scope_new(iso.as_ptr()) };

        // Open inner escapable scope.
        // SAFETY: `iso` is valid.
        let inner = unsafe { stator_escapable_handle_scope_new(iso.as_ptr()) };

        // Create a value inside the escapable scope.
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 99.0) };
        assert!(!val.is_null());

        // Escape the value — it is promoted into `outer`.
        // SAFETY: `inner` is valid; `val` is registered with `inner`.
        let escaped = unsafe { stator_escapable_handle_scope_escape(inner, val) };
        assert_eq!(escaped, val);

        // Close the inner scope. `val` was escaped so it must NOT be destroyed.
        // SAFETY: `inner` is non-null and live.
        unsafe { stator_escapable_handle_scope_close(inner) };
        // The value is still live (owned by the outer scope now).
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);
        assert!((unsafe { stator_value_as_number(escaped) } - 99.0).abs() < f64::EPSILON);

        // Close the outer scope — the escaped value is now destroyed.
        // SAFETY: `outer` is non-null and live.
        unsafe { stator_handle_scope_close(outer) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    #[test]
    fn test_escapable_handle_scope_non_escaped_values_destroyed_on_close() {
        let iso = IsolateGuard::new();
        // Open outer scope for the escaped value.
        // SAFETY: `iso` is valid.
        let outer = unsafe { stator_handle_scope_new(iso.as_ptr()) };

        // Open inner escapable scope.
        // SAFETY: `iso` is valid.
        let inner = unsafe { stator_escapable_handle_scope_new(iso.as_ptr()) };

        // Create two values — escape only one.
        // SAFETY: `iso` is valid.
        let keep = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let _ = unsafe { stator_value_new_number(iso.as_ptr(), 2.0) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 2);

        // Escape `keep` into the outer scope.
        // SAFETY: `inner` and `keep` are valid.
        let _ = unsafe { stator_escapable_handle_scope_escape(inner, keep) };

        // Close inner scope — only the non-escaped value is destroyed.
        // SAFETY: `inner` is valid.
        unsafe { stator_escapable_handle_scope_close(inner) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);

        // Close outer scope — `keep` (escaped) is now destroyed.
        // SAFETY: `outer` is valid.
        unsafe { stator_handle_scope_close(outer) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    #[test]
    fn test_escapable_handle_scope_escape_null_scope_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: null scope is documented to return null.
        let result = unsafe { stator_escapable_handle_scope_escape(std::ptr::null_mut(), val) };
        assert!(result.is_null());
        // SAFETY: `val` is non-null and live; clean up manually.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_escapable_handle_scope_escape_null_val_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_escapable_handle_scope_new(iso.as_ptr()) };
        // SAFETY: null val is documented to return null.
        let result = unsafe { stator_escapable_handle_scope_escape(scope, std::ptr::null_mut()) };
        assert!(result.is_null());
        // SAFETY: `scope` is valid.
        unsafe { stator_escapable_handle_scope_close(scope) };
    }

    #[test]
    fn test_escapable_handle_scope_no_outer_scope_caller_owns_escaped() {
        let iso = IsolateGuard::new();
        // No outer scope — escaped value has no parent scope to go into.
        // SAFETY: `iso` is valid.
        let scope = unsafe { stator_escapable_handle_scope_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 7.0) };

        // Escape with no outer scope — caller takes ownership.
        // SAFETY: `scope` and `val` are valid.
        let escaped = unsafe { stator_escapable_handle_scope_escape(scope, val) };
        assert_eq!(escaped, val);

        // Close the scope — `val` was escaped so it must NOT be freed here.
        // SAFETY: `scope` is valid.
        unsafe { stator_escapable_handle_scope_close(scope) };
        // Value is still live; the caller owns it.
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 1);

        // Caller's responsibility to destroy the escaped value.
        // SAFETY: `escaped` is non-null and live.
        unsafe { stator_value_destroy(escaped) };
        assert_eq!(unsafe { stator_live_object_count(iso.as_ptr()) }, 0);
    }

    // ── Value type-checking predicates ───────────────────────────────────────

    #[test]
    fn test_is_undefined_null_pointer() {
        // A null StatorValue pointer is treated as undefined.
        // SAFETY: null is documented as "treated as undefined".
        assert!(unsafe { stator_value_is_undefined(std::ptr::null()) });
    }

    #[test]
    fn test_is_undefined_true_for_undefined_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_undefined(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_undefined_false_for_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 0.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_undefined(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_null_null_pointer_returns_false() {
        // A null StatorValue pointer is undefined, not null.
        // SAFETY: null pointer documented as "undefined".
        assert!(!unsafe { stator_value_is_null(std::ptr::null()) });
    }

    #[test]
    fn test_is_null_true_for_null_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_null(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_null_false_for_undefined() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_null(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_string_true() {
        let iso = IsolateGuard::new();
        let s = b"hi\0";
        // SAFETY: `iso` is valid; `s` pointer is valid for 2 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 2) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_string(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_string_false_for_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_string(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_number_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 3.14) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_number(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_number_false_for_string() {
        let iso = IsolateGuard::new();
        let s = b"x\0";
        // SAFETY: `iso` is valid; `s` is valid for 1 byte.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 1) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_number(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_boolean_true_and_false_values() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let t = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: `iso` is valid.
        let f = unsafe { stator_value_new_boolean(iso.as_ptr(), false) };
        // SAFETY: both pointers are non-null and live.
        assert!(unsafe { stator_value_is_boolean(t) });
        assert!(unsafe { stator_value_is_boolean(f) });
        // SAFETY: non-null and live.
        unsafe {
            stator_value_destroy(t);
            stator_value_destroy(f);
        }
    }

    #[test]
    fn test_is_boolean_false_for_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_object_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_object(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_object_false_for_null_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        // null is NOT an object even though typeof null === "object".
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_object(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_object_false_for_null_pointer() {
        // SAFETY: null pointer is documented as undefined.
        assert!(!unsafe { stator_value_is_object(std::ptr::null()) });
    }

    #[test]
    fn test_is_function_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_function_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_function(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_function_false_for_object() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_function(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_array_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_array_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_array(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_array_false_for_object() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_array(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_int32_true_for_integer_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_int32(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_int32_false_for_float() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 3.14) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_int32(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_int32_boundaries() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let min = unsafe { stator_value_new_number(iso.as_ptr(), i32::MIN as f64) };
        let max = unsafe { stator_value_new_number(iso.as_ptr(), i32::MAX as f64) };
        let over = unsafe { stator_value_new_number(iso.as_ptr(), i32::MAX as f64 + 1.0) };
        // SAFETY: pointers are non-null and live.
        assert!(unsafe { stator_value_is_int32(min) });
        assert!(unsafe { stator_value_is_int32(max) });
        assert!(!unsafe { stator_value_is_int32(over) });
        // SAFETY: non-null and live.
        unsafe {
            stator_value_destroy(min);
            stator_value_destroy(max);
            stator_value_destroy(over);
        }
    }

    #[test]
    fn test_is_uint32_true_for_zero_and_max() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let zero = unsafe { stator_value_new_number(iso.as_ptr(), 0.0) };
        let max = unsafe { stator_value_new_number(iso.as_ptr(), u32::MAX as f64) };
        // SAFETY: pointers are non-null and live.
        assert!(unsafe { stator_value_is_uint32(zero) });
        assert!(unsafe { stator_value_is_uint32(max) });
        // SAFETY: non-null and live.
        unsafe {
            stator_value_destroy(zero);
            stator_value_destroy(max);
        }
    }

    #[test]
    fn test_is_uint32_false_for_negative() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), -1.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_uint32(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_date_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_date_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_date(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_date_false_for_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 0.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_date(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_regexp_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_regexp_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_regexp(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_regexp_false_for_string() {
        let iso = IsolateGuard::new();
        let s = b"abc\0";
        // SAFETY: `iso` is valid; `s` is valid for 3 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 3) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_regexp(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_promise_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_promise_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_promise(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_promise_false_for_object() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_promise(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_map_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_map_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_map(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_map_false_for_set() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_set_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_map(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_set_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_set_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_set(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_set_false_for_map() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_map_tag(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_is_set(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_is_object_includes_array_date_regexp_promise_map_set() {
        let iso = IsolateGuard::new();
        // Arrays, dates, regexps, promises, maps, and sets are all objects.
        let arr = unsafe { stator_value_new_array_tag(iso.as_ptr()) };
        let date = unsafe { stator_value_new_date_tag(iso.as_ptr()) };
        let regexp = unsafe { stator_value_new_regexp_tag(iso.as_ptr()) };
        let promise = unsafe { stator_value_new_promise_tag(iso.as_ptr()) };
        let map = unsafe { stator_value_new_map_tag(iso.as_ptr()) };
        let set = unsafe { stator_value_new_set_tag(iso.as_ptr()) };
        // SAFETY: all pointers are non-null and live.
        assert!(unsafe { stator_value_is_object(arr) });
        assert!(unsafe { stator_value_is_object(date) });
        assert!(unsafe { stator_value_is_object(regexp) });
        assert!(unsafe { stator_value_is_object(promise) });
        assert!(unsafe { stator_value_is_object(map) });
        assert!(unsafe { stator_value_is_object(set) });
        // SAFETY: non-null and live.
        unsafe {
            stator_value_destroy(arr);
            stator_value_destroy(date);
            stator_value_destroy(regexp);
            stator_value_destroy(promise);
            stator_value_destroy(map);
            stator_value_destroy(set);
        }
    }

    #[test]
    fn test_is_object_false_for_primitives() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let num = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let undef = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        let null = unsafe { stator_value_new_null(iso.as_ptr()) };
        let s = b"x\0";
        let str_val =
            unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 1) };
        let bool_val = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: all pointers are non-null and live.
        assert!(!unsafe { stator_value_is_object(num) });
        assert!(!unsafe { stator_value_is_object(undef) });
        assert!(!unsafe { stator_value_is_object(null) });
        assert!(!unsafe { stator_value_is_object(str_val) });
        assert!(!unsafe { stator_value_is_object(bool_val) });
        // SAFETY: non-null and live.
        unsafe {
            stator_value_destroy(num);
            stator_value_destroy(undef);
            stator_value_destroy(null);
            stator_value_destroy(str_val);
            stator_value_destroy(bool_val);
        }
    }

    #[test]
    fn test_new_boolean_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let val = unsafe { stator_value_new_boolean(std::ptr::null_mut(), true) };
        assert!(val.is_null());
    }

    #[test]
    fn test_new_undefined_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let val = unsafe { stator_value_new_undefined(std::ptr::null_mut()) };
        assert!(val.is_null());
    }

    #[test]
    fn test_new_null_null_isolate_returns_null() {
        // SAFETY: null isolate is documented to return null.
        let val = unsafe { stator_value_new_null(std::ptr::null_mut()) };
        assert!(val.is_null());
    }

    #[test]
    fn test_value_type_null_value_is_object() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        let type_ptr = unsafe { stator_value_type(val) };
        // SAFETY: returned pointer is static.
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        // ECMAScript: typeof null === "object".
        assert_eq!(type_str, "object");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_value_to_string_utf8_null_value() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        let mut buf = [0u8; 16];
        // SAFETY: `val` is non-null and live; buf is valid for 16 bytes.
        let len =
            unsafe { stator_value_to_string_utf8(val, buf.as_mut_ptr() as *mut c_char, buf.len()) };
        assert_eq!(len, 4);
        assert_eq!(&buf[..4], b"null");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── stator_value_to_boolean ───────────────────────────────────────────────

    #[test]
    fn test_to_boolean_undefined_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_null_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_true_is_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_false_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_boolean(iso.as_ptr(), false) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_zero_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 0.0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_nan_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), f64::NAN) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_nonzero_number_is_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_empty_string_is_false() {
        let iso = IsolateGuard::new();
        let s = b"\0";
        // SAFETY: `iso` is valid; 0-byte string.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 0) };
        // SAFETY: `val` is non-null and live.
        assert!(!unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_nonempty_string_is_true() {
        let iso = IsolateGuard::new();
        let s = b"x\0";
        // SAFETY: `iso` is valid; `s` valid for 1 byte.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 1) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_object_is_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_boolean_null_pointer_is_false() {
        // SAFETY: null pointer is documented to return false.
        assert!(!unsafe { stator_value_to_boolean(std::ptr::null()) });
    }

    // ── stator_value_to_number ────────────────────────────────────────────────

    #[test]
    fn test_to_number_from_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 3.14) };
        // SAFETY: `val` is non-null and live.
        assert!((unsafe { stator_value_to_number(val) } - 3.14).abs() < f64::EPSILON);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_undefined_is_nan() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_number(val) }.is_nan());
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_null_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_null(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 0.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_true_is_one() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 1.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_false_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_boolean(iso.as_ptr(), false) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 0.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_numeric_string() {
        let iso = IsolateGuard::new();
        let s = b"42\0";
        // SAFETY: `iso` is valid; `s` valid for 2 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 2) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 42.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_empty_string_is_zero() {
        let iso = IsolateGuard::new();
        let s = b"\0";
        // SAFETY: `iso` is valid; 0-byte string.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 0) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 0.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_non_numeric_string_is_nan() {
        let iso = IsolateGuard::new();
        let s = b"abc\0";
        // SAFETY: `iso` is valid; `s` valid for 3 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 3) };
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_to_number(val) }.is_nan());
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_from_hex_string() {
        let iso = IsolateGuard::new();
        let s = b"0xff\0";
        // SAFETY: `iso` is valid; `s` valid for 4 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 4) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_number(val) }, 255.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_number_null_pointer_is_nan() {
        // SAFETY: null pointer is documented to return NaN.
        assert!(unsafe { stator_value_to_number(std::ptr::null()) }.is_nan());
    }

    // ── stator_value_to_string ────────────────────────────────────────────────

    #[test]
    fn test_to_string_from_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let num = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        // SAFETY: `iso` and `num` are valid.
        let s = unsafe { stator_value_to_string(iso.as_ptr(), num) };
        assert!(!s.is_null());
        // SAFETY: `s` is non-null.
        let ptr = unsafe { stator_value_as_string(s) };
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "42");
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_value_destroy(num);
            stator_value_destroy(s);
        }
    }

    #[test]
    fn test_to_string_from_boolean_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let b = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: `iso` and `b` are valid.
        let s = unsafe { stator_value_to_string(iso.as_ptr(), b) };
        assert!(!s.is_null());
        // SAFETY: `s` is non-null.
        let ptr = unsafe { stator_value_as_string(s) };
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "true");
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_value_destroy(b);
            stator_value_destroy(s);
        }
    }

    #[test]
    fn test_to_string_from_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let n = unsafe { stator_value_new_null(iso.as_ptr()) };
        // SAFETY: `iso` and `n` are valid.
        let s = unsafe { stator_value_to_string(iso.as_ptr(), n) };
        assert!(!s.is_null());
        // SAFETY: `s` is non-null.
        let ptr = unsafe { stator_value_as_string(s) };
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "null");
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_value_destroy(n);
            stator_value_destroy(s);
        }
    }

    #[test]
    fn test_to_string_null_isolate_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // SAFETY: null isolate documented to return null.
        let s = unsafe { stator_value_to_string(std::ptr::null_mut(), val) };
        assert!(s.is_null());
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── stator_value_to_int32 ─────────────────────────────────────────────────

    #[test]
    fn test_to_int32_from_positive_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.9) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_int32(val) }, 42);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_int32_from_negative_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), -1.0) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_int32(val) }, -1);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_int32_from_nan_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), f64::NAN) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_int32(val) }, 0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_int32_from_infinity_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), f64::INFINITY) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_int32(val) }, 0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_int32_from_large_number_wraps() {
        let iso = IsolateGuard::new();
        // 2^32 + 1 → ToInt32 = 1 (mod 2^32)
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 4_294_967_297.0) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_int32(val) }, 1);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_int32_from_undefined_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: `val` is non-null and live (undefined → NaN → 0).
        assert_eq!(unsafe { stator_value_to_int32(val) }, 0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── stator_value_to_uint32 ────────────────────────────────────────────────

    #[test]
    fn test_to_uint32_from_positive_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.9) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_uint32(val) }, 42u32);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_uint32_from_negative_wraps() {
        let iso = IsolateGuard::new();
        // -1 → ToUint32 = 2^32 - 1 = 4294967295
        let val = unsafe { stator_value_new_number(iso.as_ptr(), -1.0) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_uint32(val) }, 4_294_967_295u32);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_uint32_from_nan_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), f64::NAN) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_uint32(val) }, 0u32);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_uint32_from_infinity_is_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), f64::INFINITY) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_value_to_uint32(val) }, 0u32);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_to_uint32_from_string_parsed() {
        let iso = IsolateGuard::new();
        let s = b"10\0";
        // SAFETY: `iso` is valid; `s` valid for 2 bytes.
        let val = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 2) };
        // SAFETY: `val` is non-null and live ("10" → 10 → 10u32).
        assert_eq!(unsafe { stator_value_to_uint32(val) }, 10u32);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── stator_value_strict_equals ────────────────────────────────────────────

    #[test]
    fn test_strict_equals_same_number() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        let b = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        // SAFETY: both are non-null and live.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_different_numbers() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let b = unsafe { stator_value_new_number(iso.as_ptr(), 2.0) };
        // SAFETY: both are non-null and live.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_nan_is_not_equal_to_nan() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), f64::NAN) };
        let b = unsafe { stator_value_new_number(iso.as_ptr(), f64::NAN) };
        // SAFETY: both are non-null and live.  NaN !== NaN per ECMAScript.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_positive_zero_equals_negative_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), 0.0) };
        let b = unsafe { stator_value_new_number(iso.as_ptr(), -0.0) };
        // SAFETY: both are non-null and live.  +0 === -0 per ECMAScript.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_same_string() {
        let iso = IsolateGuard::new();
        let s = b"hello\0";
        // SAFETY: `iso` is valid; `s` valid for 5 bytes.
        let a = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 5) };
        let b = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 5) };
        // SAFETY: both are non-null and live.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_different_strings() {
        let iso = IsolateGuard::new();
        let s1 = b"hello\0";
        let s2 = b"world\0";
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_string(iso.as_ptr(), s1.as_ptr() as *const c_char, 5) };
        let b = unsafe { stator_value_new_string(iso.as_ptr(), s2.as_ptr() as *const c_char, 5) };
        // SAFETY: both are non-null and live.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_number_and_string_is_false() {
        let iso = IsolateGuard::new();
        let s = b"42\0";
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        let b = unsafe { stator_value_new_string(iso.as_ptr(), s.as_ptr() as *const c_char, 2) };
        // SAFETY: both are non-null and live.  Different types → false.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_undefined_equals_undefined() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        let b = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: both are non-null and live.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_null_equals_null() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_null(iso.as_ptr()) };
        let b = unsafe { stator_value_new_null(iso.as_ptr()) };
        // SAFETY: both are non-null and live.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_null_and_undefined_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_null(iso.as_ptr()) };
        let b = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: both are non-null and live.  Different types → false.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_boolean_true_equals_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        let b = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        // SAFETY: both are non-null and live.
        assert!(unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_boolean_true_and_false_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        let b = unsafe { stator_value_new_boolean(iso.as_ptr(), false) };
        // SAFETY: both are non-null and live.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_two_objects_is_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_object(iso.as_ptr()) };
        let b = unsafe { stator_value_new_object(iso.as_ptr()) };
        // SAFETY: both are non-null and live.  No shared identity → false.
        assert!(!unsafe { stator_value_strict_equals(a, b) });
        unsafe {
            stator_value_destroy(a);
            stator_value_destroy(b);
        }
    }

    #[test]
    fn test_strict_equals_null_ptr_treated_as_undefined() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let a = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        // SAFETY: null pointer is documented to be treated as undefined.
        assert!(unsafe { stator_value_strict_equals(a, std::ptr::null()) });
        // SAFETY: `a` is non-null and live.
        unsafe { stator_value_destroy(a) };
    }

    // ── stator_string_new_from_utf8 / stator_string_utf8_length / stator_string_write_utf8 ──

    #[test]
    fn test_string_new_from_utf8_roundtrip_ascii() {
        let iso = IsolateGuard::new();
        let src = b"hello";
        // SAFETY: `iso` is valid; `src` is valid for 5 bytes.
        let val = unsafe {
            stator_string_new_from_utf8(iso.as_ptr(), src.as_ptr() as *const c_char, src.len())
        };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let len = unsafe { stator_string_utf8_length(val) };
        assert_eq!(len, 5);
        let mut buf = [0u8; 16];
        let mut nchars: usize = 0;
        // SAFETY: `val` is non-null; `buf` is valid for 16 bytes; `nchars` is valid.
        let written = unsafe {
            stator_string_write_utf8(
                val,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &raw mut nchars,
            )
        };
        assert_eq!(written, 5);
        assert_eq!(nchars, 5);
        assert_eq!(&buf[..5], b"hello");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_string_new_from_utf8_roundtrip_multibyte() {
        let iso = IsolateGuard::new();
        // "café" in UTF-8: c, a, f, é(2 bytes) → 5 bytes total
        let src = "café".as_bytes();
        // SAFETY: `iso` is valid; `src` is valid UTF-8.
        let val = unsafe {
            stator_string_new_from_utf8(iso.as_ptr(), src.as_ptr() as *const c_char, src.len())
        };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let len = unsafe { stator_string_utf8_length(val) };
        assert_eq!(len, src.len());
        let mut buf = vec![0u8; src.len()];
        let mut nchars: usize = 0;
        // SAFETY: `val` is non-null; `buf` is valid; `nchars` is valid.
        let written = unsafe {
            stator_string_write_utf8(
                val,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &raw mut nchars,
            )
        };
        assert_eq!(written, src.len());
        assert_eq!(nchars, src.len());
        assert_eq!(&buf[..], src);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_string_write_utf8_truncates_to_buf_size() {
        let iso = IsolateGuard::new();
        let src = b"abcdef";
        // SAFETY: `iso` is valid; `src` is valid for 6 bytes.
        let val = unsafe {
            stator_string_new_from_utf8(iso.as_ptr(), src.as_ptr() as *const c_char, src.len())
        };
        assert!(!val.is_null());
        let mut buf = [0u8; 3];
        let mut nchars: usize = 0;
        // SAFETY: `val` is non-null; `buf` is valid for 3 bytes; `nchars` is valid.
        let written = unsafe {
            stator_string_write_utf8(
                val,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &raw mut nchars,
            )
        };
        // Only 3 bytes fit in the buffer.
        assert_eq!(written, 3);
        assert_eq!(nchars, 3);
        assert_eq!(&buf[..], b"abc");
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_string_utf8_length_non_string_returns_zero() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        // SAFETY: `val` is non-null and live.
        assert_eq!(unsafe { stator_string_utf8_length(val) }, 0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_string_write_utf8_null_val_returns_zero() {
        let mut buf = [0u8; 8];
        let mut nchars: usize = 99;
        // SAFETY: `val` is null (documented as returning 0); `buf` is valid.
        let written = unsafe {
            stator_string_write_utf8(
                std::ptr::null(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &raw mut nchars,
            )
        };
        assert_eq!(written, 0);
        assert_eq!(nchars, 0);
    }

    // ── stator_number_new / stator_integer_new ────────────────────────────────

    #[test]
    fn test_number_new_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_number_new(iso.as_ptr(), 2.718) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let n = unsafe { stator_value_as_number(val) };
        assert!((n - 2.718).abs() < f64::EPSILON);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_integer_new_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_integer_new(iso.as_ptr(), 42) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_number(val) });
        let n = unsafe { stator_value_as_number(val) };
        assert_eq!(n, 42.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_integer_new_negative() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_integer_new(iso.as_ptr(), -7) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        let n = unsafe { stator_value_as_number(val) };
        assert_eq!(n, -7.0);
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── stator_boolean_new / stator_undefined_new / stator_null_new ──────────

    #[test]
    fn test_boolean_new_true() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_boolean_new(iso.as_ptr(), true) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_boolean_new_false() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_boolean_new(iso.as_ptr(), false) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_boolean(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_undefined_new() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_undefined_new(iso.as_ptr()) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_undefined(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_null_new() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let val = unsafe { stator_null_new(iso.as_ptr()) };
        assert!(!val.is_null());
        // SAFETY: `val` is non-null and live.
        assert!(unsafe { stator_value_is_null(val) });
        // SAFETY: `val` is non-null and live.
        unsafe { stator_value_destroy(val) };
    }

    // ── FunctionTemplate / Phase 4 ────────────────────────────────────────────

    #[test]
    fn test_function_template_new_returns_nonnull() {
        let iso = IsolateGuard::new();
        unsafe extern "C" fn noop(_info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            std::ptr::null_mut()
        }
        // SAFETY: `iso` is valid; `noop` is a valid function pointer.
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), noop) };
        assert!(!tmpl.is_null());
        // SAFETY: `tmpl` is non-null and live.
        unsafe { stator_function_template_destroy(tmpl) };
    }

    #[test]
    fn test_function_template_null_isolate_returns_null() {
        unsafe extern "C" fn noop(_info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            std::ptr::null_mut()
        }
        // SAFETY: null isolate is documented to return null.
        let tmpl = unsafe { stator_function_template_new(std::ptr::null_mut(), noop) };
        assert!(tmpl.is_null());
    }

    #[test]
    fn test_function_template_destroy_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_function_template_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_function_template_get_function_returns_function_value() {
        let iso = IsolateGuard::new();
        unsafe extern "C" fn noop(_info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            std::ptr::null_mut()
        }
        // SAFETY: `iso` is valid.
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), noop) };
        assert!(!tmpl.is_null());
        // SAFETY: `tmpl` is non-null and live; null ctx is allowed.
        let fn_val = unsafe { stator_function_template_get_function(tmpl, std::ptr::null_mut()) };
        assert!(!fn_val.is_null());
        // SAFETY: `fn_val` is non-null and live.
        assert!(unsafe { stator_value_is_function(fn_val) });
        // type() should report "function".
        // SAFETY: `fn_val` is non-null and live.
        let type_ptr = unsafe { stator_value_type(fn_val) };
        let type_str = unsafe { CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "function");
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_value_destroy(fn_val);
            stator_function_template_destroy(tmpl);
        }
    }

    #[test]
    fn test_function_callback_info_length_and_get() {
        let iso = IsolateGuard::new();
        // This callback reads its arguments and stores the count in a global.
        static ARG_COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
        static ARG0_VALUE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
        unsafe extern "C" fn recorder(info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            // SAFETY: `info` is valid for the duration of this call.
            let len = unsafe { stator_function_callback_info_length(info) };
            ARG_COUNT.store(len, std::sync::atomic::Ordering::SeqCst);
            if len > 0 {
                // SAFETY: `info` is valid; index 0 is in range.
                let arg = unsafe { stator_function_callback_info_get(info, 0) };
                if !arg.is_null() {
                    // SAFETY: `arg` is valid for this call.
                    let n = unsafe { stator_value_as_number(arg) };
                    ARG0_VALUE.store(n as i64, std::sync::atomic::Ordering::SeqCst);
                }
            }
            std::ptr::null_mut()
        }
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid; `recorder` is a valid function pointer.
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), recorder) };
        // SAFETY: `tmpl` and `ctx` are valid.
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        // Install the function into the context as "rec".
        let name = c"rec";
        // SAFETY: `ctx` and `fn_val` are valid; `name` is a valid C string.
        unsafe { stator_context_global_set(ctx, name.as_ptr(), fn_val) };
        // SAFETY: `fn_val` is non-null and live.
        unsafe { stator_value_destroy(fn_val) };

        // Run a script that calls rec(99).
        let src = b"rec(99)";
        // SAFETY: null ctx is allowed for compile; `src` is valid UTF-8.
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert!(!script.is_null());
        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        // The callback returns null → undefined; result may be null or undefined.
        if !result.is_null() {
            // SAFETY: `result` is non-null and live.
            unsafe { stator_value_destroy(result) };
        }

        // Verify that the callback was invoked with one argument equal to 99.
        assert_eq!(ARG_COUNT.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(ARG0_VALUE.load(std::sync::atomic::Ordering::SeqCst), 99);

        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_script_free(script);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_function_template_get_function_called_returns_value() {
        let iso = IsolateGuard::new();
        // Callback: returns the first argument doubled.
        unsafe extern "C" fn double_it(
            info: *const StatorFunctionCallbackInfo,
        ) -> *mut StatorValue {
            // SAFETY: `info` is valid for this call.
            let isolate = unsafe { stator_function_callback_info_get_isolate(info) };
            let arg = unsafe { stator_function_callback_info_get(info, 0) };
            let n = if arg.is_null() {
                0.0
            } else {
                // SAFETY: `arg` is valid for this call.
                unsafe { stator_value_as_number(arg) }
            };
            // SAFETY: `isolate` is valid.
            let ret = unsafe { stator_value_new_number(isolate, n * 2.0) };
            ret
        }
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid; `double_it` is a valid function pointer.
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), double_it) };
        // SAFETY: `tmpl` and `ctx` are valid.
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        let name = c"dbl";
        // SAFETY: all pointers are valid.
        unsafe { stator_context_global_set(ctx, name.as_ptr(), fn_val) };
        // SAFETY: `fn_val` is non-null and live.
        unsafe { stator_value_destroy(fn_val) };

        // Run a script that calls dbl(21) and uses the result.
        let src = b"dbl(21)";
        // SAFETY: `src` is valid UTF-8.
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert!(!script.is_null());
        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null(), "expected a result from dbl(21)");
        // SAFETY: `result` is non-null and live.
        let n = unsafe { stator_value_as_number(result) };
        assert!(
            (n - 42.0).abs() < f64::EPSILON,
            "expected dbl(21) == 42.0, got {n}"
        );

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_script_compile_and_run_with_context() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"1 + 2";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());
        // SAFETY: `script` is non-null and live.
        let err = unsafe { stator_script_get_error(script) };
        assert!(err.is_null(), "unexpected compile error");
        // SAFETY: `script` and `ctx` are both valid.
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null(), "expected a result");
        // SAFETY: `result` is non-null and live.
        let n = unsafe { stator_value_as_number(result) };
        assert!((n - 3.0).abs() < f64::EPSILON, "expected 3.0, got {n}");
        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_context_global_set_function_then_script_run() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };

        unsafe extern "C" fn add_one(info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            // SAFETY: `info` is valid.
            let isolate = unsafe { stator_function_callback_info_get_isolate(info) };
            let arg = unsafe { stator_function_callback_info_get(info, 0) };
            let n = if arg.is_null() {
                0.0
            } else {
                // SAFETY: `arg` is valid.
                unsafe { stator_value_as_number(arg) }
            };
            // SAFETY: `isolate` is valid.
            unsafe { stator_value_new_number(isolate, n + 1.0) }
        }

        // SAFETY: `iso` is valid.
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), add_one) };
        // SAFETY: `tmpl` and `ctx` are valid.
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        let name = c"addOne";
        // SAFETY: all pointers are valid.
        unsafe { stator_context_global_set(ctx, name.as_ptr(), fn_val) };
        // SAFETY: `fn_val` is non-null.
        unsafe { stator_value_destroy(fn_val) };

        let src = b"addOne(41)";
        // SAFETY: `src` is valid UTF-8.
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        // SAFETY: `result` is non-null and live.
        let n = unsafe { stator_value_as_number(result) };
        assert!((n - 42.0).abs() < f64::EPSILON, "expected 42.0, got {n}");

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    /// `stator_isolate_get_stats` must return valid (non-crashing) stats and
    /// fill the struct with zero or positive counts.
    #[test]
    fn test_isolate_get_stats_fills_struct() {
        let iso = IsolateGuard::new();
        let mut stats = StatorCompilationStats {
            jit_functions_compiled: 0xff,
            jit_code_bytes: 0xdeadbeef,
        };
        // SAFETY: `iso` is valid; `stats` is a valid mutable reference.
        unsafe { stator_isolate_get_stats(iso.as_ptr(), &mut stats as *mut _) };

        // The function must have overwritten the sentinel values.
        assert_ne!(stats.jit_functions_compiled, 0xff, "stats must be filled");
        assert_ne!(stats.jit_code_bytes, 0xdeadbeef, "stats must be filled");
    }

    /// `stator_isolate_get_stats` with a null `stats` pointer must not crash.
    #[test]
    fn test_isolate_get_stats_null_stats_is_safe() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid; null stats pointer is explicitly handled.
        unsafe { stator_isolate_get_stats(iso.as_ptr(), std::ptr::null_mut()) };
    }

    // ── WebAssembly FFI ───────────────────────────────────────────────────────

    #[test]
    fn test_wasm_compile_null_isolate_returns_null() {
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: null isolate is documented to return null.
        let m = unsafe { stator_wasm_compile(std::ptr::null_mut(), bytes.as_ptr(), bytes.len()) };
        assert!(m.is_null());
    }

    #[test]
    fn test_wasm_compile_null_bytes_returns_null() {
        let iso = IsolateGuard::new();
        // SAFETY: null bytes pointer is documented to return null.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), std::ptr::null(), 8) };
        assert!(m.is_null());
    }

    #[test]
    fn test_wasm_compile_zero_len_returns_null() {
        let iso = IsolateGuard::new();
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: zero len is documented to return null.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bytes.as_ptr(), 0) };
        assert!(m.is_null());
    }

    #[test]
    fn test_wasm_compile_invalid_bytes_returns_null() {
        let iso = IsolateGuard::new();
        let bad = b"not wasm";
        // SAFETY: `iso` is valid; `bad` is valid for 8 bytes.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bad.as_ptr(), bad.len()) };
        assert!(m.is_null());
    }

    #[test]
    fn test_wasm_compile_valid_returns_nonnull() {
        let iso = IsolateGuard::new();
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: `iso` is valid; `bytes` is a valid empty Wasm binary.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bytes.as_ptr(), bytes.len()) };
        assert!(!m.is_null());
        // SAFETY: `m` is non-null and live.
        unsafe { stator_wasm_module_destroy(m) };
    }

    #[test]
    fn test_wasm_module_destroy_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_wasm_module_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_wasm_instantiate_null_module_returns_null() {
        // SAFETY: null module is documented to return null.
        let inst = unsafe {
            stator_wasm_instantiate(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null())
        };
        assert!(inst.is_null());
    }

    #[test]
    fn test_wasm_instantiate_valid_returns_nonnull() {
        let iso = IsolateGuard::new();
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: `iso` is valid; `bytes` is a valid empty Wasm binary.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bytes.as_ptr(), bytes.len()) };
        assert!(!m.is_null());
        // SAFETY: `m` is valid; null ctx and imports are allowed.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), std::ptr::null()) };
        assert!(!inst.is_null());
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    #[test]
    fn test_wasm_instance_destroy_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_wasm_instance_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_wasm_instance_exports_null_instance_returns_null() {
        // SAFETY: null instance is documented to return null.
        let exports = unsafe { stator_wasm_instance_exports(std::ptr::null_mut()) };
        assert!(exports.is_null());
    }

    #[test]
    fn test_wasm_instance_exports_empty_module() {
        let iso = IsolateGuard::new();
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: all pointers are valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bytes.as_ptr(), bytes.len()) };
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), std::ptr::null()) };
        // SAFETY: `inst` is valid.
        let exports = unsafe { stator_wasm_instance_exports(inst) };
        assert!(!exports.is_null());
        // The empty module has no exports; the first element must be the null terminator.
        // SAFETY: `exports` is a valid null-terminated array.
        let first = unsafe { *exports };
        assert!(first.is_null(), "empty module should have no exports");
        // SAFETY: `exports` was returned by `stator_wasm_instance_exports`.
        unsafe { stator_wasm_exports_destroy(exports) };
        unsafe {
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    #[test]
    fn test_wasm_exports_destroy_null_is_safe() {
        // SAFETY: null is documented as a no-op.
        unsafe { stator_wasm_exports_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_wasm_compile_instantiate_exports_and_call() {
        let iso = IsolateGuard::new();
        // WAT text is accepted by wasmtime as a Wasm module source.
        let wat: &[u8] = br#"
            (module
                (func $add (export "add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add))
        "#;
        // SAFETY: `iso` is valid; `wat` is valid WAT accepted by wasmtime.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        assert!(!m.is_null(), "stator_wasm_compile must succeed");

        // SAFETY: `m` is valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), std::ptr::null()) };
        assert!(!inst.is_null(), "stator_wasm_instantiate must succeed");

        // Check that "add" appears in the exports list.
        // SAFETY: `inst` is valid.
        let exports = unsafe { stator_wasm_instance_exports(inst) };
        assert!(!exports.is_null());
        let mut found = false;
        let mut i = 0;
        loop {
            // SAFETY: `exports` is a valid null-terminated array.
            let ptr = unsafe { *exports.add(i) };
            if ptr.is_null() {
                break;
            }
            // SAFETY: each non-null element is a valid C string.
            let name = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
            if name == "add" {
                found = true;
            }
            i += 1;
        }
        assert!(found, "'add' must appear in the exports list");
        // SAFETY: `exports` was returned by `stator_wasm_instance_exports`.
        unsafe { stator_wasm_exports_destroy(exports) };

        // Call `add(3, 4)` and verify the result is 7.
        let name = c"add";
        // SAFETY: `iso` is valid.
        let arg_a = unsafe { stator_value_new_number(iso.as_ptr(), 3.0) };
        let arg_b = unsafe { stator_value_new_number(iso.as_ptr(), 4.0) };
        let arg_ptrs: [*const StatorValue; 2] = [arg_a, arg_b];
        // SAFETY: all pointers are valid.
        let result = unsafe {
            stator_wasm_instance_call(
                inst,
                iso.as_ptr(),
                name.as_ptr(),
                arg_ptrs.as_ptr(),
                arg_ptrs.len(),
            )
        };
        assert!(!result.is_null(), "add(3, 4) must return a value");
        // SAFETY: `result` is non-null and live.
        let n = unsafe { stator_value_as_number(result) };
        assert_eq!(n, 7.0, "add(3, 4) must equal 7");

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(result);
            stator_value_destroy(arg_a);
            stator_value_destroy(arg_b);
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    #[test]
    fn test_wasm_instance_call_null_instance_returns_null() {
        let iso = IsolateGuard::new();
        let name = c"add";
        // SAFETY: null instance is documented to return null.
        let result = unsafe {
            stator_wasm_instance_call(
                std::ptr::null_mut(),
                iso.as_ptr(),
                name.as_ptr(),
                std::ptr::null(),
                0,
            )
        };
        assert!(result.is_null());
    }

    #[test]
    fn test_wasm_instance_call_missing_export_returns_null() {
        let iso = IsolateGuard::new();
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        // SAFETY: all pointers are valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), bytes.as_ptr(), bytes.len()) };
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), std::ptr::null()) };
        let name = c"nonexistent";
        // SAFETY: all pointers are valid.
        let result = unsafe {
            stator_wasm_instance_call(inst, iso.as_ptr(), name.as_ptr(), std::ptr::null(), 0)
        };
        assert!(result.is_null(), "missing export must return null");
        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    // ── ObjectTemplate tests ─────────────────────────────────────────────

    #[test]
    fn test_object_template_create_destroy() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let tmpl = unsafe { stator_object_template_new(iso.as_ptr()) };
        assert!(!tmpl.is_null());
        // SAFETY: `tmpl` is non-null and live.
        unsafe { stator_object_template_destroy(tmpl) };
    }

    #[test]
    fn test_object_template_null_isolate_returns_null() {
        // SAFETY: passing null is safe.
        let tmpl = unsafe { stator_object_template_new(std::ptr::null_mut()) };
        assert!(tmpl.is_null());
    }

    #[test]
    fn test_object_template_set_and_new_instance() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let tmpl = unsafe { stator_object_template_new(iso.as_ptr()) };
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 42.0) };
        let key = c"answer";
        // SAFETY: all pointers are valid.
        unsafe { stator_object_template_set(tmpl, key.as_ptr(), val) };
        let obj = unsafe { stator_object_template_new_instance(tmpl) };
        assert!(!obj.is_null());
        // Check that the property was installed.
        let got = unsafe { stator_object_get(obj, key.as_ptr()) };
        assert!(!got.is_null());
        let n = unsafe { stator_value_as_number(got) };
        assert!((n - 42.0).abs() < f64::EPSILON);
        // SAFETY: clean up.
        unsafe {
            stator_value_destroy(got);
            stator_object_destroy(obj);
            stator_value_destroy(val);
            stator_object_template_destroy(tmpl);
        }
    }

    #[test]
    fn test_object_template_internal_field_count() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let tmpl = unsafe { stator_object_template_new(iso.as_ptr()) };
        assert_eq!(
            unsafe { stator_object_template_internal_field_count(tmpl) },
            0
        );
        unsafe { stator_object_template_set_internal_field_count(tmpl, 3) };
        assert_eq!(
            unsafe { stator_object_template_internal_field_count(tmpl) },
            3
        );
        unsafe { stator_object_template_destroy(tmpl) };
    }

    // ── TryCatch tests ───────────────────────────────────────────────────

    #[test]
    fn test_try_catch_no_exception() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        assert!(!tc.is_null());
        assert!(!unsafe { stator_try_catch_has_caught(tc) });
        assert!(unsafe { stator_try_catch_exception(tc) }.is_null());
        unsafe { stator_try_catch_destroy(tc) };
    }

    #[test]
    fn test_try_catch_catches_pending_exception() {
        let iso = IsolateGuard::new();
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let exc = unsafe { stator_value_new_string(iso.as_ptr(), c"error".as_ptr(), 5) };
        // Throw the exception on the isolate.
        unsafe { stator_isolate_throw_exception(iso.as_ptr(), exc) };
        // TryCatch should now capture it.
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let caught = unsafe { stator_try_catch_exception(tc) };
        assert!(!caught.is_null());
        assert_eq!(caught, exc);
        // Clean up.
        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(exc);
        }
    }

    // ── DOM integration FFI ───────────────────────────────────────────────────

    #[test]
    fn test_dom_object_wrap_new_and_destroy() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 2) };
        assert!(!wrap.is_null());
        // SAFETY: `wrap` is non-null and live.
        assert_eq!(
            unsafe { stator_dom_object_wrap_internal_field_count(wrap) },
            2
        );
        // SAFETY: `wrap` is non-null and live.
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_object_wrap_null_isolate() {
        // SAFETY: passing null is explicitly supported.
        let wrap = unsafe { stator_dom_object_wrap_new(std::ptr::null_mut(), 1) };
        assert!(wrap.is_null());
    }

    #[test]
    fn test_dom_object_wrap_too_many_fields() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid; 100 exceeds MAX_INTERNAL_FIELDS.
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 100) };
        assert!(wrap.is_null());
    }

    #[test]
    fn test_dom_object_wrap_internal_fields() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 2) };
        assert!(!wrap.is_null());

        let sentinel: usize = 0xBEEF;
        // SAFETY: `wrap` is non-null and live.
        unsafe {
            stator_dom_object_wrap_set_internal_field(wrap, 0, sentinel as *mut c_void);
        }
        let retrieved = unsafe { stator_dom_object_wrap_get_internal_field(wrap, 0) };
        assert_eq!(retrieved as usize, sentinel);

        // Out-of-range returns null.
        let oob = unsafe { stator_dom_object_wrap_get_internal_field(wrap, 99) };
        assert!(oob.is_null());

        // SAFETY: `wrap` is non-null and live.
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_object_wrap_null_internal_field_count() {
        // SAFETY: passing null is supported.
        assert_eq!(
            unsafe { stator_dom_object_wrap_internal_field_count(std::ptr::null()) },
            0
        );
    }

    #[test]
    fn test_dom_weak_ref_lifecycle() {
        use std::sync::atomic::{AtomicBool, Ordering};

        static INVOKED: AtomicBool = AtomicBool::new(false);

        unsafe extern "C" fn weak_cb(_data: *mut c_void) {
            INVOKED.store(true, Ordering::SeqCst);
        }

        INVOKED.store(false, Ordering::SeqCst);

        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 1) };
        assert!(!wrap.is_null());

        // SAFETY: `wrap` is valid.
        let weak = unsafe { stator_dom_weak_ref_new(wrap, weak_cb) };
        assert!(!weak.is_null());
        // SAFETY: `weak` is valid.
        assert!(unsafe { stator_dom_weak_ref_is_alive(weak) });

        // SAFETY: `weak` is valid.
        unsafe { stator_dom_weak_ref_invoke(weak) };
        assert!(!unsafe { stator_dom_weak_ref_is_alive(weak) });
        assert!(INVOKED.load(Ordering::SeqCst));

        // SAFETY: cleaning up.
        unsafe {
            stator_dom_weak_ref_destroy(weak);
            stator_dom_object_wrap_destroy(wrap);
        }
    }

    #[test]
    fn test_try_catch_reset_clears_exception() {
        let iso = IsolateGuard::new();
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let exc = unsafe { stator_value_new_number(iso.as_ptr(), 99.0) };
        unsafe { stator_isolate_throw_exception(iso.as_ptr(), exc) };
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        unsafe { stator_try_catch_reset(tc) };
        assert!(!unsafe { stator_try_catch_has_caught(tc) });
        assert!(unsafe { stator_try_catch_exception(tc) }.is_null());
        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(exc);
        }
    }

    // ── Pending exception helper tests ───────────────────────────────────

    #[test]
    fn test_isolate_has_pending_exception_initially_false() {
        let iso = IsolateGuard::new();
        assert!(!unsafe { stator_isolate_has_pending_exception(iso.as_ptr()) });
    }

    #[test]
    fn test_isolate_pending_exception_roundtrip() {
        let iso = IsolateGuard::new();
        let exc = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        unsafe { stator_isolate_throw_exception(iso.as_ptr(), exc) };
        assert!(unsafe { stator_isolate_has_pending_exception(iso.as_ptr()) });
        let cleared = unsafe { stator_isolate_clear_pending_exception(iso.as_ptr()) };
        assert_eq!(cleared, exc);
        assert!(!unsafe { stator_isolate_has_pending_exception(iso.as_ptr()) });
        unsafe { stator_value_destroy(exc) };
    }

    #[test]
    fn test_dom_weak_ref_clear() {
        use std::sync::atomic::{AtomicBool, Ordering};

        static CB_CALLED: AtomicBool = AtomicBool::new(false);

        unsafe extern "C" fn weak_cb_clear(_data: *mut c_void) {
            CB_CALLED.store(true, Ordering::SeqCst);
        }

        CB_CALLED.store(false, Ordering::SeqCst);

        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 1) };
        let weak = unsafe { stator_dom_weak_ref_new(wrap, weak_cb_clear) };

        // Clear without invoking.
        unsafe { stator_dom_weak_ref_clear(weak) };
        assert!(!unsafe { stator_dom_weak_ref_is_alive(weak) });

        // Invoke after clear is a no-op.
        unsafe { stator_dom_weak_ref_invoke(weak) };
        assert!(!CB_CALLED.load(Ordering::SeqCst));

        unsafe {
            stator_dom_weak_ref_destroy(weak);
            stator_dom_object_wrap_destroy(wrap);
        }
    }

    #[test]
    fn test_dom_weak_ref_null_wrap() {
        unsafe extern "C" fn noop_cb(_data: *mut c_void) {}
        // SAFETY: passing null is supported.
        let weak = unsafe { stator_dom_weak_ref_new(std::ptr::null(), noop_cb) };
        assert!(weak.is_null());
    }

    #[test]
    fn test_dom_weak_ref_null_operations() {
        // All operations on null should be safe no-ops.
        assert!(!unsafe { stator_dom_weak_ref_is_alive(std::ptr::null()) });
        unsafe { stator_dom_weak_ref_invoke(std::ptr::null()) };
        unsafe { stator_dom_weak_ref_clear(std::ptr::null()) };
        unsafe { stator_dom_weak_ref_destroy(std::ptr::null_mut()) };
    }
}

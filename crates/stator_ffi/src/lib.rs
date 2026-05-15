//! `stator_jse_ffi` — C-ABI FFI surface for the Stator JavaScript engine.
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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::Write as _;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use stator_jse::builtins::error::{ErrorKind, JsError};
use stator_jse::builtins::promise::PromiseState;
use stator_jse::bytecode::bytecode_array::BytecodeArray;
use stator_jse::bytecode::bytecode_generator::BytecodeGenerator;
use stator_jse::bytecode::bytecodes::{Operand, decode};
use stator_jse::compiler::baseline::compiler::{
    first_deopt_counts, reset_first_deopt_counts, reset_stub_deopt_counts, stub_call_counts,
    stub_deopt_counts,
};
use stator_jse::dom::{
    DomObjectWrap, DomWeakRef, IndexedPropertyHandlerConfig, NamedPropertyHandlerConfig,
};
use stator_jse::gc::heap::Heap;
use stator_jse::host::HostModuleLoader;
use stator_jse::interpreter::{GlobalEnv, Interpreter, InterpreterFrame};
use stator_jse::objects::js_object::JsObject;
use stator_jse::objects::map::PropertyAttributes;
use stator_jse::objects::property_map::PropertyMap;
use stator_jse::objects::value::{JsValue, NativeFn};
use stator_jse::parser;
use stator_jse::parser::ast::{
    ImportAttribute, ImportSpecifier, ModuleDecl, ModuleExportName, ObjectPatProp, Pat, Program,
    ProgramItem, Stmt,
};
use stator_jse::wasm::{
    HostFunc, HostFuncCallback, HostVal, HostValKind, WasmEngine, WasmInstance, WasmModule,
};

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
    /// Pending exception stored by [`stator_isolate_throw_exception`] or by
    /// FFI script execution when an interpreter error crosses the boundary.
    /// Embedder-thrown values remain embedder-owned; internally-created script
    /// error values are tracked by `pending_exception_owned`.
    pending_exception: Option<*mut StatorValue>,
    /// Whether `pending_exception` is owned by the FFI layer and should be
    /// destroyed when a try-catch scope clears it.
    pending_exception_owned: bool,
    /// Structured classification of the pending exception, populated by FFI
    /// script execution when an interpreter or compile error crosses the
    /// boundary.  Embedders read it through [`StatorMessage`] APIs and may
    /// inspect it without disturbing `pending_exception`.  `None` when no
    /// structured information is available (e.g. an embedder-thrown raw
    /// exception value).
    pending_message: Option<Box<StatorMessage>>,
    /// The innermost active [`StatorHandleScope`] on this isolate, or null if
    /// no scope is currently open.  This forms a linked list via each scope's
    /// `previous` field.
    active_handle_scope: *mut StatorHandleScope,
    /// When true, FFI script execution blocks Stator's JIT tiers for scripts
    /// run in this isolate.
    jit_disabled: bool,
    /// When true, the embedder has requested that JavaScript execution on
    /// this isolate be terminated.  Mirrors `v8::Isolate::TerminateExecution`.
    ///
    /// Stored as an `AtomicBool` so that other threads can safely request
    /// termination of a script running on the isolate's owning thread.
    /// While a script is executing on the owning thread, the interpreter
    /// polls this flag at backward branches, function-call boundaries, and
    /// between microtasks via the thread-local pointer published by
    /// [`stator_jse::interpreter::set_interrupt_flag`].
    ///
    /// Setting and observing the flag from any thread is well-defined; the
    /// rest of `StatorIsolate` remains single-threaded.
    terminating: AtomicBool,
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
        pending_exception_owned: false,
        pending_message: None,
        active_handle_scope: std::ptr::null_mut(),
        jit_disabled: false,
        terminating: AtomicBool::new(false),
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

/// Request that JavaScript execution on `isolate` be terminated.
///
/// Mirrors `v8::Isolate::TerminateExecution`.  Sets a sticky atomic flag on
/// the isolate; the interpreter polls this flag at backward branches,
/// function-call boundaries, and between microtasks, and unwinds the running
/// script with a script-execution-terminated error visible through the
/// existing pending-exception / try-catch reporting paths.  Subsequent calls
/// to [`stator_script_run`] will also refuse to start a new script while the
/// flag is set.  The flag remains set until explicitly cleared via
/// [`stator_isolate_cancel_terminate_execution`].
///
/// This function is safe to call from any thread; the underlying flag is
/// atomic.  Concurrency on the rest of the isolate is still the embedder's
/// responsibility.
///
/// Limitations:
/// * Baseline / Maglev / Turbofan JIT-emitted machine code does not poll
///   the flag in this slice.  Termination is observed at the next
///   interpreter boundary (JIT return / deopt / runtime stub).  Hostile
///   code that stays inside JIT code indefinitely is not terminable until
///   it re-enters the interpreter; embedders running untrusted code should
///   currently call [`stator_isolate_set_jit_disabled`].
/// * Wasm execution polls the flag at JS↔Wasm entry and through Wasmtime epoch
///   interruption while compiled Wasm is running.  The epoch broadcast is
///   process-wide, but each store only traps when the thread running that store
///   observes its own published Stator interrupt flag.
///
/// Does nothing when `isolate` is null.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_terminate_execution(isolate: *mut StatorIsolate) {
    if isolate.is_null() {
        return;
    }
    // SAFETY: caller guarantees `isolate` is valid.  The atomic store is
    // safe from any thread.
    unsafe { (*isolate).terminating.store(true, Ordering::Relaxed) };
    // Advance the epoch of every live Wasm engine in the process so any
    // in-flight Wasm call reaches its per-store deadline callback at the next
    // epoch check.  The callback checks the running thread's interrupt flag
    // before trapping, so unrelated isolates ignore the broadcast: see
    // `stator_jse::wasm::interrupt_all_wasm_engines`.
    stator_jse::wasm::interrupt_all_wasm_engines();
}

/// Clear a previously requested termination on `isolate`.
///
/// Mirrors `v8::Isolate::CancelTerminateExecution`.  After this call,
/// [`stator_isolate_is_execution_terminating`] returns `false` until a new
/// termination is requested.  Does nothing when `isolate` is null.
///
/// Safe to call from any thread.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_cancel_terminate_execution(isolate: *mut StatorIsolate) {
    if isolate.is_null() {
        return;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).terminating.store(false, Ordering::Relaxed) };
}

/// Return `true` if a termination has been requested on `isolate` and not
/// yet cleared.
///
/// Returns `false` when `isolate` is null.  Safe to call from any thread.
///
/// # Safety
/// `isolate` must be either null or a valid, live [`StatorIsolate`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_is_execution_terminating(
    isolate: *const StatorIsolate,
) -> bool {
    if isolate.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).terminating.load(Ordering::Relaxed) }
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
    // SAFETY: caller guarantees `isolate` and `exception` validity.
    unsafe { set_pending_exception(isolate, exception, false) };
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
        let isolate_ref = &mut *isolate;
        let exception = isolate_ref
            .pending_exception
            .take()
            .unwrap_or(std::ptr::null_mut());
        isolate_ref.pending_exception_owned = false;
        // The structured message is bound to the exception value; clearing
        // the exception also clears the message.  Embedders that want to
        // preserve it across `clear_pending_exception` should call
        // `stator_isolate_take_pending_message` first.
        isolate_ref.pending_message = None;
        exception
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
    let (count, bytes) = stator_jse::interpreter::jit_stats();
    // SAFETY: caller guarantees `stats` is valid for writes.
    unsafe {
        (*stats).jit_functions_compiled = count;
        (*stats).jit_code_bytes = bytes;
    }
}

/// Extended tiering diagnostics for embedders.
#[repr(C)]
#[derive(Debug)]
pub struct StatorTieringStats {
    /// Number of functions compiled by the baseline JIT.
    pub baseline_function_count: u32,
    /// Total baseline JIT code bytes.
    pub baseline_code_bytes: usize,
    /// Number of functions compiled by Maglev.
    pub maglev_function_count: u32,
    /// Total Maglev code bytes.
    pub maglev_code_bytes: usize,
    /// Number of functions compiled by Turbofan.
    pub turbofan_function_count: u32,
    /// Total Turbofan code bytes.
    pub turbofan_code_bytes: usize,
    /// Number of entries into the best-available-JIT dispatch helper.
    pub best_jit_entries: u64,
    /// Number of best-JIT lookups that executed Maglev.
    pub maglev_hits: u64,
    /// Number of best-JIT lookups that missed Maglev.
    pub maglev_misses: u64,
    /// Number of raw Maglev execution attempts.
    pub maglev_tried: u64,
    /// Number of Maglev executions that completed without deopt.
    pub maglev_executed: u64,
    /// Number of Maglev executions that deopted.
    pub maglev_deopts: u64,
    /// Number of times Maglev code was requested before it was ready.
    pub maglev_not_ready: u64,
    /// Number of Maglev attempts blocked by deopt backoff.
    pub maglev_blocked: u64,
    /// Number of times the Maglev executable cache was empty.
    pub maglev_cache_empty: u64,
    /// Number of Maglev compilation jobs started.
    pub maglev_compile_started: u32,
    /// Number of Maglev compilation jobs that failed.
    pub maglev_compile_failed: u32,
    /// Number of Maglev compilation jobs that panicked.
    pub maglev_compile_panicked: u32,
    /// Number of best-JIT lookups intercepted by Turbofan.
    pub turbofan_hits: u64,
    /// Number of interpreter run-inner entries.
    pub run_inner_entries: u64,
    /// Number of dispatch-loop entries.
    pub run_dispatch_entries: u64,
    /// Maglev deopt counters by reason.
    pub maglev_deopt_counts: [u64; 6],
    /// Runtime-stub deopt counters by stub slot.
    pub stub_deopt_counts: [u64; 24],
    /// First-deopt-per-invocation counters by stub slot.
    pub stub_first_deopt_counts: [u64; 24],
    /// Runtime-stub call counters by stub slot.
    pub stub_call_counts: [u64; 24],
}

/// Fill `*stats` with current tiering diagnostics.
///
/// Does nothing when `stats` is null. Passing a null isolate is permitted and
/// still returns process/thread counters.
///
/// # Safety
/// - `isolate` must be either null or a valid, live `StatorIsolate` pointer.
/// - `stats` must be null or valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_get_tiering_stats(
    _isolate: *const StatorIsolate,
    stats: *mut StatorTieringStats,
) {
    if stats.is_null() {
        return;
    }
    let (baseline_function_count, baseline_code_bytes) = stator_jse::interpreter::jit_stats();
    let (maglev_function_count, maglev_code_bytes) = stator_jse::interpreter::maglev_stats();
    let (turbofan_function_count, turbofan_code_bytes) = stator_jse::interpreter::turbofan_stats();
    let (best_jit_entries, maglev_hits, maglev_misses) =
        stator_jse::interpreter::jit_entry_diagnostics();
    let (run_inner_entries, run_dispatch_entries) =
        stator_jse::interpreter::dispatch_entry_diagnostics();
    let (
        maglev_tried,
        maglev_executed,
        maglev_deopts,
        maglev_not_ready,
        _maglev_compilations,
        _maglev_bytes,
        maglev_compile_started,
        maglev_compile_failed,
        maglev_compile_panicked,
        maglev_blocked,
        maglev_cache_empty,
        turbofan_hits,
    ) = stator_jse::interpreter::maglev_diagnostics();

    // SAFETY: caller provided a valid output pointer.
    unsafe {
        *stats = StatorTieringStats {
            baseline_function_count,
            baseline_code_bytes,
            maglev_function_count,
            maglev_code_bytes,
            turbofan_function_count,
            turbofan_code_bytes,
            best_jit_entries,
            maglev_hits,
            maglev_misses,
            maglev_tried,
            maglev_executed,
            maglev_deopts,
            maglev_not_ready,
            maglev_blocked,
            maglev_cache_empty,
            maglev_compile_started,
            maglev_compile_failed,
            maglev_compile_panicked,
            turbofan_hits,
            run_inner_entries,
            run_dispatch_entries,
            maglev_deopt_counts: stator_jse::interpreter::maglev_deopt_categories(),
            stub_deopt_counts: stub_deopt_counts(),
            stub_first_deopt_counts: first_deopt_counts(),
            stub_call_counts: stub_call_counts(),
        };
    }
}

/// Reset tiering diagnostics visible through `stator_isolate_get_tiering_stats`.
///
/// A null isolate is accepted; counters are thread/process diagnostics rather
/// than heap-owned state.
///
/// # Safety
/// `isolate` must be null or a valid, live `StatorIsolate` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_reset_tiering_stats(_isolate: *mut StatorIsolate) {
    stator_jse::interpreter::reset_tiering_stats();
}

/// Enable or disable JIT tiers for scripts run in `isolate`.
///
/// Does nothing when `isolate` is null.
///
/// # Safety
/// `isolate` must be null or a valid, live `StatorIsolate` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_set_jit_disabled(
    isolate: *mut StatorIsolate,
    disabled: bool,
) {
    if !isolate.is_null() {
        // SAFETY: caller guarantees `isolate` is valid.
        unsafe { (*isolate).jit_disabled = disabled };
    }
}

/// Return whether JIT tiers are disabled for `isolate`.
///
/// Returns `false` when `isolate` is null.
///
/// # Safety
/// `isolate` must be null or a valid, live `StatorIsolate` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_jit_disabled(isolate: *const StatorIsolate) -> bool {
    if isolate.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).jit_disabled }
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
    /// Per-slot embedder data, mirroring [`StatorIsolate::embedder_data`] but
    /// scoped to a single context.  Used by browser embedders (e.g. Blink) to
    /// associate a `ScriptState` or `ExecutionContext` with the V8-equivalent
    /// `Context`.  The pointers are owned by the embedder and are not freed
    /// when the context is destroyed.
    embedder_data: Vec<*mut c_void>,
    /// Optional host callback used to resolve static `import` / re-export
    /// specifiers and module-evaluation dynamic `import()` /
    /// `import.meta.resolve` requests for this context. The resolver owns its
    /// `user_data` cleanup contract and is dropped when replaced, cleared, or
    /// when the context is destroyed.
    module_resolver: Option<StatorModuleResolver>,
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
            inner: Rc::new(RefCell::new(JsObject::new())),
            isolate,
        },
        globals: Rc::new(RefCell::new(GlobalEnv::new())),
        embedder_data: Vec::new(),
        module_resolver: None,
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

/// Store an embedder-defined pointer in `slot` on `ctx`.
///
/// Slots grow on demand; previously-unset slots are treated as null.  Mirrors
/// `v8::Context::SetAlignedPointerInEmbedderData` and is intended for use by
/// browser embedders that need to associate per-context state (e.g. a
/// `ScriptState` or `ExecutionContext`) with a Stator context.
///
/// The caller retains ownership of `data`; the context only stores the
/// pointer and does not free it on destruction.
///
/// Does nothing when `ctx` is null.
///
/// # Safety
/// `ctx` must be either null or a valid, live [`StatorContext`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_set_embedder_data(
    ctx: *mut StatorContext,
    slot: u32,
    data: *mut c_void,
) {
    if ctx.is_null() {
        return;
    }
    let slot = slot as usize;
    // SAFETY: caller guarantees `ctx` is valid.
    let slots = unsafe { &mut (*ctx).embedder_data };
    if slot >= slots.len() {
        slots.resize(slot + 1, std::ptr::null_mut());
    }
    slots[slot] = data;
}

/// Retrieve the embedder-defined pointer previously stored at `slot` on `ctx`.
///
/// Returns a null pointer when `ctx` is null or `slot` has not been set.
///
/// # Safety
/// `ctx` must be either null or a valid, live [`StatorContext`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_get_embedder_data(
    ctx: *const StatorContext,
    slot: u32,
) -> *mut c_void {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let slot = slot as usize;
    // SAFETY: caller guarantees `ctx` is valid.
    let slots = unsafe { &(*ctx).embedder_data };
    if slot < slots.len() {
        slots[slot]
    } else {
        std::ptr::null_mut()
    }
}

/// Register, replace, or clear the module resolver callback for `ctx`.
///
/// The resolver is scoped to a single context. Passing a non-null `callback`
/// stores `user_data` and optional `free_user_data`; any previous resolver is
/// dropped first, invoking its free callback exactly once when both the
/// previous `user_data` and previous `free_user_data` are non-null.
///
/// To clear an existing resolver, pass a null `callback`, null `user_data`, and
/// null `free_user_data`. Passing a null callback with non-null cleanup state is
/// rejected and leaves the existing resolver unchanged.
/// [`stator_context_destroy`] also drops the active resolver, if any.
///
/// Resolver callbacks are invoked synchronously by [`stator_module_instantiate`]
/// and, during [`stator_module_evaluate`], by dynamic `import()` and
/// `import.meta.resolve` on the same thread that called into Stator; Stator does
/// not dispatch module resolution to background worker threads. Dynamic imports
/// use the currently-evaluating module as `referrer`, pass an empty attributes
/// slice, instantiate/evaluate the returned module synchronously, and reject the
/// import promise with the same typed status/detail mapping used for static
/// resolver failures. `import.meta.resolve` calls the same resolver and returns
/// the resolved module's resource name when available, otherwise the original
/// specifier.
///
/// The callback must not destroy `ctx`, replace or clear the currently-running
/// resolver, free the `referrer`, or otherwise re-enter APIs that mutate the
/// same module graph while a resolver invocation is active. The FFI
/// context/module graph APIs are single-threaded; embedders must serialize
/// access to a context and its modules.
///
/// Returns `true` on successful registration or clear, and `false` for a null
/// context or malformed clear request.
///
/// # Safety
/// - `ctx` must be null or a valid, live [`StatorContext`] pointer.
/// - `callback`, when non-null, must remain callable until replaced, cleared,
///   or `ctx` is destroyed.
/// - `user_data`, when non-null, must remain valid for callbacks until the
///   resolver is replaced/cleared/destroyed. If `free_user_data` is non-null,
///   Stator calls it synchronously with that same pointer during resolver
///   replacement, resolver clearing, or context destruction. If `user_data` is
///   null, `free_user_data` is stored but not invoked.
/// - `free_user_data`, when non-null, must remain callable until it is invoked
///   or until the resolver is replaced/cleared/destroyed with null `user_data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_set_module_resolver(
    ctx: *mut StatorContext,
    callback: Option<
        unsafe extern "C" fn(
            ctx: *mut StatorContext,
            user_data: *mut c_void,
            referrer: *const StatorModule,
            origin: *const StatorModuleOrigin,
            specifier: *const c_char,
            specifier_len: usize,
            attributes: *const StatorImportAttribute,
            attributes_len: usize,
            out_module: *mut *mut StatorModule,
            out_error: *mut *mut StatorString,
        ) -> StatorResolveStatus,
    >,
    user_data: *mut c_void,
    free_user_data: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
) -> bool {
    if ctx.is_null() {
        return false;
    }

    let resolver = match callback {
        Some(callback) => Some(StatorModuleResolver {
            callback,
            user_data,
            free_user_data,
        }),
        None => {
            if !user_data.is_null() || free_user_data.is_some() {
                return false;
            }
            None
        }
    };

    // SAFETY: caller guarantees `ctx` is valid and uniquely mutable for the
    // duration of this FFI call. Assignment drops the previous resolver, if any.
    unsafe { (*ctx).module_resolver = resolver };
    true
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
    ///
    /// Tag-only placeholder used by [`stator_value_new_object`] and friends.
    /// Carries no per-instance state, so identity is not preserved across
    /// FFI round trips.  Use [`StatorValueInner::ObjectHandle`] (produced by
    /// [`stator_object_as_value`]) when identity must survive.
    Object,
    /// A JavaScript object backed by a real, shared [`JsObject`] storage.
    ///
    /// Produced by [`stator_object_as_value`].  Multiple [`StatorValue`] /
    /// [`StatorObject`] handles that share the same `Rc<RefCell<JsObject>>`
    /// observe one another's mutations and round-trip through
    /// [`stator_value_as_object`] without identity loss.
    ObjectHandle(Rc<RefCell<JsObject>>),
    /// A JavaScript object backed by a DOM wrapper materialized through the
    /// embedder FFI.
    DomWrapHandle {
        plain: Rc<RefCell<PropertyMap>>,
        wrap: *mut StatorDomObjectWrap,
        alive: Rc<Cell<bool>>,
    },
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

type MaterializedDomWrapRegistry = HashMap<usize, (*mut StatorDomObjectWrap, Rc<Cell<bool>>)>;

thread_local! {
    static DOM_WRAP_MATERIALIZED_REGISTRY: RefCell<MaterializedDomWrapRegistry> =
        RefCell::new(HashMap::new());
}

fn dom_wrap_inner_for_plain_object(plain: &Rc<RefCell<PropertyMap>>) -> Option<StatorValueInner> {
    let key = Rc::as_ptr(plain) as usize;
    DOM_WRAP_MATERIALIZED_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let (wrap, alive) = registry.get(&key)?;
        if wrap.is_null() || !alive.get() {
            return None;
        }
        Some(StatorValueInner::DomWrapHandle {
            plain: Rc::clone(plain),
            wrap: *wrap,
            alive: Rc::clone(alive),
        })
    })
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

fn register_handle_with_active_scope(isolate: *mut StatorIsolate, val: *mut StatorValue) {
    if isolate.is_null() || val.is_null() {
        return;
    }
    // SAFETY: callers pass the value's owning isolate.  The active scope, when
    // present, is owned by the same isolate and is valid until closed.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(val);
        }
    }
}

unsafe fn allocate_stator_value(
    isolate: *mut StatorIsolate,
    inner: StatorValueInner,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    let val = Box::into_raw(Box::new(StatorValue { inner, isolate }));
    register_handle_with_active_scope(isolate, val);
    val
}

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
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. }
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
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. }
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
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. }
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
            | StatorValueInner::ObjectHandle(_)
            | StatorValueInner::DomWrapHandle { .. }
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
///
/// The underlying [`JsObject`] storage is reference-counted, allowing the
/// same object identity to be exposed through multiple FFI handles via
/// [`stator_value_as_object`] / [`stator_object_as_value`].  Mutations
/// through one handle are observed through all other handles that share the
/// same backing storage.
pub struct StatorObject {
    inner: Rc<RefCell<JsObject>>,
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
        inner: Rc::new(RefCell::new(JsObject::new())),
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
    let _ = unsafe { (*obj).inner.borrow_mut().set_property(&key_str, js_val) };
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
    let js_val = unsafe { (*obj).inner.borrow().get_property(&key_str) };
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
    unsafe { (*obj).inner.borrow().has_property(&key_str) }
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
    unsafe {
        (*obj)
            .inner
            .borrow_mut()
            .delete_own_property(&key_str)
            .unwrap_or(false)
    }
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
    let keys = unsafe { (*obj).inner.borrow().own_property_keys() };
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
    bytecodes: Option<Rc<BytecodeArray>>,
    /// Human-readable error message, or `None` on success.
    error: Option<CString>,
    /// Structured classification of the compile error, or
    /// [`StatorMessageKind::StatorMessageKindUnknown`] when the script compiled successfully
    /// or the error was not classified.  Parse/compile failures are
    /// classified as [`StatorMessageKind::StatorMessageKindSyntax`].
    error_kind: StatorMessageKind,
    /// Resource name (typically a filename or URL) supplied by the embedder
    /// via [`stator_script_set_origin`].  Mirrors `v8::ScriptOrigin::ResourceName`.
    /// Used by browser embedders to attribute stack frames to source files.
    resource_name: Option<CString>,
    /// 1-based line offset of the script within `resource_name`.  Defaults to 0.
    /// Mirrors `v8::ScriptOrigin::ResourceLineOffset`.
    resource_line_offset: i32,
    /// 1-based column offset of the script within `resource_name`.  Defaults
    /// to 0.  Mirrors `v8::ScriptOrigin::ResourceColumnOffset`.
    resource_column_offset: i32,
}

/// Owned UTF-8 string handle used by module resolver out-parameters.
///
/// Hosts construct values with [`stator_string_new`] and may transfer ownership
/// to Stator through resolver `out_error` parameters. Stator releases received
/// strings with [`stator_string_free`]. The byte buffer is not required to be
/// null-terminated and may legally contain interior NULs.
pub struct StatorString {
    bytes: Vec<u8>,
}

/// Allocate a new owned [`StatorString`] from `len` UTF-8 bytes at `data`.
///
/// Returns a non-null handle on success. When `data` is null and `len` is
/// zero, an empty string is returned. When `data` is null and `len` is
/// non-zero, returns null. The bytes are copied into engine-owned storage and
/// are not interpreted, validated, or required to be null-terminated.
///
/// # Safety
/// - `data` must either be null (with `len == 0`) or point to at least `len`
///   readable bytes for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_new(data: *const c_char, len: usize) -> *mut StatorString {
    if data.is_null() && len != 0 {
        return std::ptr::null_mut();
    }
    let bytes = if len == 0 {
        Vec::new()
    } else {
        // SAFETY: caller guarantees `data` points to `len` readable bytes.
        let slice = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
        slice.to_vec()
    };
    Box::into_raw(Box::new(StatorString { bytes }))
}

/// Return a pointer to the UTF-8 bytes held by `string`.
///
/// The returned pointer is valid until `string` is freed and is **not**
/// guaranteed to be null-terminated. Returns null when `string` is null.
///
/// # Safety
/// - `string` must be either null or a valid pointer to a live
///   [`StatorString`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_data(string: *const StatorString) -> *const c_char {
    if string.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `string` is live.
    let s = unsafe { &*string };
    s.bytes.as_ptr() as *const c_char
}

/// Return the byte length of `string`.
///
/// Returns `0` when `string` is null.
///
/// # Safety
/// - `string` must be either null or a valid pointer to a live
///   [`StatorString`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_len(string: *const StatorString) -> usize {
    if string.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `string` is live.
    unsafe { (*string).bytes.len() }
}

/// Free a [`StatorString`] previously returned by [`stator_string_new`] or
/// transferred from a host resolver to Stator.
///
/// Passing null is a no-op.
///
/// # Safety
/// - `string` must either be null or a pointer obtained from
///   [`stator_string_new`] that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_string_free(string: *mut StatorString) {
    if string.is_null() {
        return;
    }
    // SAFETY: caller transferred ownership of `string` to this call.
    unsafe { drop(Box::from_raw(string)) };
}

/// A browser-facing compiled ES module record.
///
/// Created by [`stator_module_compile`] and released by [`stator_module_free`].
/// The handle owns parsed/compiled module bytecode, origin metadata, and the
/// current module linking/evaluation status.
pub struct StatorModule {
    /// Compiled module bytecodes on success; `None` on parse / compile error.
    bytecodes: Option<Rc<BytecodeArray>>,
    /// Host-visible source kind supplied at compile time.
    source_type: StatorModuleType,
    /// Whether this module needs host import resolution before evaluation.
    has_dependencies: bool,
    /// Static import and re-export requests in source order.
    module_requests: Vec<StatorModuleRequest>,
    /// Names directly exported by this module's source (including `default`
    /// when an explicit default export is present, and aliases introduced by
    /// `export { local as exported }` / `export * as ns from "..."`). This is
    /// used by the link-time validator to detect missing imports/re-exports
    /// without requiring a full live-binding evaluator.
    direct_exports: HashSet<String>,
    /// Link/evaluation status for this module record.
    status: StatorModuleStatus,
    /// Last successful evaluation result.
    last_result: Option<JsValue>,
    /// Human-readable error message, or `None` on success.
    error: Option<CString>,
    /// Structured classification of the compile or evaluation error.
    error_kind: StatorMessageKind,
    /// Resource name (typically a URL) supplied by the embedder.
    resource_name: Option<CString>,
    /// 1-based line offset of the module within `resource_name`. Defaults to 0.
    resource_line_offset: i32,
    /// 1-based column offset of the module within `resource_name`. Defaults to 0.
    resource_column_offset: i32,
    /// Owned copy of the embedder-supplied base URL bytes, if any.
    base_url: Option<Vec<u8>>,
    /// Owned copy of the embedder-supplied Subresource Integrity metadata, if any.
    integrity_metadata: Option<Vec<u8>>,
    /// Browser credentials mode applied when this module was fetched.
    credentials_mode: StatorCredentialsMode,
    /// Browser referrer policy carried by this module.
    referrer_policy: StatorReferrerPolicy,
    /// HTML parser-metadata classification carried by this module.
    parser_metadata: StatorParserMetadata,
}

struct StatorModuleRequest {
    specifier: CString,
    _attributes_storage: Vec<StatorModuleRequestAttribute>,
    attributes: Vec<StatorImportAttribute>,
    /// Names this request expects from the resolved module. Used by the
    /// link-time validator to fail with `SyntaxError` when an imported or
    /// re-exported binding is missing from the source module.
    imports: Vec<RequestedExport>,
    /// Module pointer returned by the host resolver during link, or null
    /// before the request has been resolved. Borrowed (not owned).
    resolved: Cell<*mut StatorModule>,
}

/// A single binding requested from a dependency module.
#[derive(Debug, Clone)]
enum RequestedExport {
    /// Default import or `export { default as ... } from "..."`.
    Default,
    /// Named import / re-export of `name`.
    Named(String),
    /// Namespace import (`import * as ns`) or namespace re-export
    /// (`export * as ns from "..."`). Always satisfied; the namespace object
    /// itself is built at evaluation time.
    Namespace,
    /// `export * from "..."` — bare star re-export. Pulls all non-default
    /// exports of the source through, but never fails name validation on
    /// its own.
    Star,
}

struct StatorModuleRequestAttribute {
    key: CString,
    value: CString,
}

/// A single ECMAScript import attribute passed to a module resolver callback.
///
/// Both `key` and `value` are UTF-8 byte slices and are not required to be
/// null-terminated. They are only valid for the duration of the callback.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StatorImportAttribute {
    /// Pointer to `key_len` bytes of the attribute key.
    pub key: *const c_char,
    /// Number of bytes in `key`.
    pub key_len: usize,
    /// Pointer to `value_len` bytes of the attribute value.
    pub value: *const c_char,
    /// Number of bytes in `value`.
    pub value_len: usize,
}

/// Host-visible source kind for a compiled module record.
///
/// JavaScript modules are parsed and compiled by Stator today. Other source
/// kinds are recorded as metadata so embedders can compare import attributes
/// against host policy while execution support lands separately.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorModuleType {
    /// ECMAScript source text module.
    StatorModuleTypeJavaScript = 0,
    /// JSON module source.
    StatorModuleTypeJson = 1,
    /// WebAssembly module source.
    StatorModuleTypeWebAssembly = 2,
    /// CSS module source.
    StatorModuleTypeCss = 3,
}

/// Browser credentials mode applied to a module fetch.
///
/// Stable persisted numbering — never reorder these variants.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorCredentialsMode {
    /// Default credentials mode (`same-origin` per HTML for module scripts).
    StatorCredentialsModeDefault = 0,
    /// `omit` — never send or store credentials.
    StatorCredentialsModeOmit = 1,
    /// `same-origin` — send credentials only for same-origin requests.
    StatorCredentialsModeSameOrigin = 2,
    /// `include` — always send credentials, even cross-origin.
    StatorCredentialsModeInclude = 3,
}

/// Browser referrer policy applied to subsequent module fetches.
///
/// Mirrors the values of the W3C Referrer Policy spec. Stable persisted
/// numbering — never reorder these variants.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorReferrerPolicy {
    /// Empty / default — defer to the embedder's environment policy.
    StatorReferrerPolicyDefault = 0,
    /// `no-referrer`.
    StatorReferrerPolicyNoReferrer = 1,
    /// `no-referrer-when-downgrade`.
    StatorReferrerPolicyNoReferrerWhenDowngrade = 2,
    /// `same-origin`.
    StatorReferrerPolicySameOrigin = 3,
    /// `origin`.
    StatorReferrerPolicyOrigin = 4,
    /// `strict-origin`.
    StatorReferrerPolicyStrictOrigin = 5,
    /// `origin-when-cross-origin`.
    StatorReferrerPolicyOriginWhenCrossOrigin = 6,
    /// `strict-origin-when-cross-origin`.
    StatorReferrerPolicyStrictOriginWhenCrossOrigin = 7,
    /// `unsafe-url`.
    StatorReferrerPolicyUnsafeUrl = 8,
}

/// HTML parser-metadata classification for a module script.
///
/// Stable persisted numbering — never reorder these variants.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorParserMetadata {
    /// Module was not inserted by the HTML parser (default).
    StatorParserMetadataNotParserInserted = 0,
    /// Module was inserted by the HTML parser.
    StatorParserMetadataParserInserted = 1,
}

/// Read-only view of the browser policy/origin metadata associated with a
/// referrer module, supplied to host module resolver callbacks.
///
/// `base_url` and `integrity_metadata` point to UTF-8 byte slices owned by the
/// referrer [`StatorModule`]; they are not required to be null-terminated and
/// may be null when the corresponding length is zero. The pointed-to bytes
/// (and the [`StatorModuleOrigin`] struct itself) are valid only for the
/// duration of the resolver callback that received them — hosts must copy any
/// data they need to retain past return.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StatorModuleOrigin {
    /// Pointer to `base_url_len` bytes of the referrer's base URL, or null
    /// when no base URL was set.
    pub base_url: *const c_char,
    /// Number of bytes in `base_url`. Zero when `base_url` is null.
    pub base_url_len: usize,
    /// Pointer to `integrity_metadata_len` bytes of Subresource Integrity
    /// metadata associated with the referrer fetch, or null when none.
    pub integrity_metadata: *const c_char,
    /// Number of bytes in `integrity_metadata`. Zero when null.
    pub integrity_metadata_len: usize,
    /// Credentials mode applied to the referrer fetch.
    pub credentials_mode: StatorCredentialsMode,
    /// Referrer policy applied to subsequent module fetches.
    pub referrer_policy: StatorReferrerPolicy,
    /// Parser-metadata classification of the referrer module script.
    pub parser_metadata: StatorParserMetadata,
}

/// Result returned by a host module resolver callback.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorResolveStatus {
    /// Resolution succeeded and `out_module` was written.
    StatorResolveStatusOk = 0,
    /// The specifier was not found by the host.
    StatorResolveStatusNotFound = 1,
    /// Resolution failed due to an I/O or network-layer problem.
    StatorResolveStatusNetworkError = 2,
    /// Resolution failed due to invalid specifier/attribute usage.
    StatorResolveStatusTypeError = 3,
    /// Resolution will complete asynchronously in a future API slice.
    StatorResolveStatusPending = 4,
}

/// Free callback for resolver-owned embedder data.
///
/// When a resolver is replaced, cleared, or its context is destroyed, Stator
/// invokes this callback with the previous `user_data` pointer when both are
/// non-null. The callback runs synchronously on the thread performing the
/// replacement, clear, or context destruction, and Stator never invokes it more
/// than once for a registered resolver instance.
pub type StatorUserDataFreeCallback = unsafe extern "C" fn(user_data: *mut c_void);

/// Synchronous host callback used to resolve an ES module import specifier.
///
/// The callback is invoked depth-first by [`stator_module_instantiate`] for
/// every static `import`, re-export, or import-with-attributes request in the
/// referrer module, on the same thread that called into Stator. On success the
/// host writes a live, compiled module pointer through `out_module`. Stator
/// borrows that module while walking the graph and may update its link status,
/// but does not take ownership, retain a reference count, or free it; the host
/// must keep the module alive for the duration of instantiation and remains
/// responsible for eventually calling [`stator_module_free`]. On failure the
/// host returns a non-`Ok` status and may optionally allocate an owned
/// diagnostic via [`stator_string_new`] and transfer it through `out_error`. The
/// engine consumes any non-null `out_error` and releases it with
/// [`stator_string_free`], so hosts must not retain or free it themselves after
/// returning.
///
/// The status drives the typed error surfaced through the module status/error
/// accessors and any future evaluation rejection:
///
/// - [`StatorResolveStatus::StatorResolveStatusNotFound`] →
///   `ReferenceError` (`StatorMessageKindReference`).
/// - [`StatorResolveStatus::StatorResolveStatusNetworkError`] →
///   `TypeError` (`StatorMessageKindType`) modelled on the HTML module-loading
///   spec for fetch failures.
/// - [`StatorResolveStatus::StatorResolveStatusTypeError`] →
///   `TypeError` (`StatorMessageKindType`).
/// - Other failure statuses surface as internal engine errors.
///
/// The optional `out_error` detail, when supplied, is appended to the engine's
/// canonical message so the actionable host context (URL, fetch failure, etc.)
/// flows through the module error accessors verbatim.
///
/// # Safety
/// - `ctx` is the context on which the resolver was registered.
/// - `user_data` is the exact pointer registered with
///   [`stator_context_set_module_resolver`].
/// - `referrer` is a read-only borrowed module pointer valid only for the
///   duration of the callback.
/// - `origin` is a read-only pointer to a [`StatorModuleOrigin`] view of the
///   referrer's policy/origin metadata. The struct and any byte slices it
///   points to are valid only for the duration of the callback. When the
///   referrer has no metadata configured, the struct's pointers are null with
///   zero lengths and its enum fields hold their `Default` variants.
/// - `specifier` points to `specifier_len` UTF-8 bytes and is not necessarily
///   null-terminated. The bytes are read-only and valid only for this callback.
/// - `attributes` points to `attributes_len` entries, or is null when the
///   length is zero. Each attribute's key/value bytes are read-only and valid
///   only for this callback.
/// - `out_module` and `out_error`, when non-null, are valid for one pointer
///   write each. Any non-null value written through `out_error` must have been
///   produced by [`stator_string_new`] and is owned by the engine after return.
/// - The callback must not destroy `ctx`, replace/clear this resolver, free the
///   `referrer` or returned `out_module`, or recursively instantiate/evaluate
///   the same module graph. Access to the context and graph must be serialized
///   by the embedder.
pub type StatorModuleResolverCallback = unsafe extern "C" fn(
    ctx: *mut StatorContext,
    user_data: *mut c_void,
    referrer: *const StatorModule,
    origin: *const StatorModuleOrigin,
    specifier: *const c_char,
    specifier_len: usize,
    attributes: *const StatorImportAttribute,
    attributes_len: usize,
    out_module: *mut *mut StatorModule,
    out_error: *mut *mut StatorString,
) -> StatorResolveStatus;

struct StatorModuleResolver {
    callback: StatorModuleResolverCallback,
    user_data: *mut c_void,
    free_user_data: Option<StatorUserDataFreeCallback>,
}

impl Drop for StatorModuleResolver {
    fn drop(&mut self) {
        if let Some(free_user_data) = self.free_user_data
            && !self.user_data.is_null()
        {
            // SAFETY: the embedder supplied this callback for exactly this
            // resolver-owned `user_data` pointer.
            unsafe { free_user_data(self.user_data) };
        }
    }
}

/// Link/evaluation state for a [`StatorModule`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorModuleStatus {
    /// The module failed to compile or a null module handle was supplied.
    StatorModuleStatusErrored = -1,
    /// The module compiled but has not been linked.
    StatorModuleStatusUnlinked = 0,
    /// The module is currently linking.
    StatorModuleStatusLinking = 1,
    /// The module has completed dependency-free linking.
    StatorModuleStatusLinked = 2,
    /// The module is currently evaluating.
    StatorModuleStatusEvaluating = 3,
    /// The module completed evaluation successfully.
    StatorModuleStatusEvaluated = 4,
}

fn program_has_module_dependencies(program: &Program) -> bool {
    program.body.iter().any(|item| match item {
        ProgramItem::ModuleDecl(ModuleDecl::Import(_)) => true,
        ProgramItem::ModuleDecl(ModuleDecl::ExportAll(_)) => true,
        ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) => decl.source.is_some(),
        ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(_)) | ProgramItem::Stmt(_) => false,
    })
}

fn module_request_attributes(attributes: &[ImportAttribute]) -> Vec<StatorModuleRequestAttribute> {
    attributes
        .iter()
        .filter_map(|attribute| {
            let key = CString::new(module_request_specifier_value(&attribute.key.name)).ok()?;
            let value =
                CString::new(module_request_specifier_value(&attribute.value.value)).ok()?;
            Some(StatorModuleRequestAttribute { key, value })
        })
        .collect()
}

fn module_request_specifier_value(specifier: &str) -> &str {
    let bytes = specifier.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'))
    {
        &specifier[1..specifier.len() - 1]
    } else {
        specifier
    }
}

fn module_request_for(
    specifier: &str,
    attributes: &[ImportAttribute],
    imports: Vec<RequestedExport>,
) -> Option<StatorModuleRequest> {
    let specifier = CString::new(module_request_specifier_value(specifier)).ok()?;
    let attributes_storage = module_request_attributes(attributes);
    let attributes = attributes_storage
        .iter()
        .map(|attribute| StatorImportAttribute {
            key: attribute.key.as_ptr(),
            key_len: attribute.key.as_bytes().len(),
            value: attribute.value.as_ptr(),
            value_len: attribute.value.as_bytes().len(),
        })
        .collect();
    Some(StatorModuleRequest {
        specifier,
        _attributes_storage: attributes_storage,
        attributes,
        imports,
        resolved: Cell::new(std::ptr::null_mut()),
    })
}

fn module_export_name_str(name: &ModuleExportName) -> &str {
    match name {
        ModuleExportName::Ident(id) => &id.name,
        ModuleExportName::Str(s) => &s.value,
    }
}

fn collect_pat_names(pat: &Pat, names: &mut HashSet<String>) {
    match pat {
        Pat::Ident(id) => {
            names.insert(id.name.clone());
        }
        Pat::Array(arr) => {
            for p in arr.elements.iter().flatten() {
                collect_pat_names(p, names);
            }
        }
        Pat::Object(obj) => {
            for prop in &obj.properties {
                match prop {
                    ObjectPatProp::KeyValue(kv) => collect_pat_names(&kv.value, names),
                    ObjectPatProp::Assign(a) => {
                        names.insert(a.key.name.clone());
                    }
                    ObjectPatProp::Rest(r) => collect_pat_names(&r.argument, names),
                }
            }
        }
        Pat::Rest(r) => collect_pat_names(&r.argument, names),
        Pat::Assign(a) => collect_pat_names(&a.left, names),
        Pat::Expr(_) => {}
    }
}

fn collect_decl_export_names(stmt: &Stmt, exports: &mut HashSet<String>) {
    match stmt {
        Stmt::VarDecl(v) => {
            for d in &v.declarators {
                collect_pat_names(&d.id, exports);
            }
        }
        Stmt::FnDecl(f) => {
            if let Some(id) = &f.id {
                exports.insert(id.name.clone());
            }
        }
        Stmt::ClassDecl(c) => {
            if let Some(id) = &c.id {
                exports.insert(id.name.clone());
            }
        }
        _ => {}
    }
}

fn collect_module_requests(program: &Program) -> (Vec<StatorModuleRequest>, HashSet<String>) {
    let mut requests = Vec::new();
    let mut exports: HashSet<String> = HashSet::new();
    for item in &program.body {
        match item {
            ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) => {
                let imports: Vec<RequestedExport> = decl
                    .specifiers
                    .iter()
                    .map(|spec| match spec {
                        ImportSpecifier::Named(n) => {
                            RequestedExport::Named(module_export_name_str(&n.imported).to_string())
                        }
                        ImportSpecifier::Default(_) => RequestedExport::Default,
                        ImportSpecifier::Namespace(_) => RequestedExport::Namespace,
                    })
                    .collect();
                if let Some(req) = module_request_for(&decl.source.value, &decl.attributes, imports)
                {
                    requests.push(req);
                }
            }
            ProgramItem::ModuleDecl(ModuleDecl::ExportAll(decl)) => {
                let imports = if let Some(name) = &decl.exported {
                    exports.insert(module_export_name_str(name).to_string());
                    vec![RequestedExport::Namespace]
                } else {
                    vec![RequestedExport::Star]
                };
                if let Some(req) = module_request_for(&decl.source.value, &decl.attributes, imports)
                {
                    requests.push(req);
                }
            }
            ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) => {
                if let Some(source) = &decl.source {
                    let mut imports = Vec::new();
                    for spec in &decl.specifiers {
                        let local = module_export_name_str(&spec.local).to_string();
                        let exported = module_export_name_str(&spec.exported).to_string();
                        imports.push(if local == "default" {
                            RequestedExport::Default
                        } else {
                            RequestedExport::Named(local)
                        });
                        exports.insert(exported);
                    }
                    if let Some(req) = module_request_for(&source.value, &decl.attributes, imports)
                    {
                        requests.push(req);
                    }
                } else {
                    for spec in &decl.specifiers {
                        exports.insert(module_export_name_str(&spec.exported).to_string());
                    }
                    if let Some(stmt) = &decl.declaration {
                        collect_decl_export_names(stmt, &mut exports);
                    }
                }
            }
            ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(_)) => {
                exports.insert("default".to_string());
            }
            ProgramItem::Stmt(_) => {}
        }
    }
    (requests, exports)
}

fn module_rejection_to_error(reason: JsValue) -> stator_jse::error::StatorError {
    let message = reason
        .to_js_string()
        .unwrap_or_else(|_| "module evaluation rejected".to_string());
    if let Some(rest) = message.strip_prefix("TypeError: ") {
        stator_jse::error::StatorError::TypeError(rest.to_string())
    } else if let Some(rest) = message.strip_prefix("SyntaxError: ") {
        stator_jse::error::StatorError::SyntaxError(rest.to_string())
    } else if let Some(rest) = message.strip_prefix("ReferenceError: ") {
        stator_jse::error::StatorError::ReferenceError(rest.to_string())
    } else if let Some(rest) = message.strip_prefix("RangeError: ") {
        stator_jse::error::StatorError::RangeError(rest.to_string())
    } else if let Some(rest) = message.strip_prefix("URIError: ") {
        stator_jse::error::StatorError::URIError(rest.to_string())
    } else {
        stator_jse::error::StatorError::JsException(message)
    }
}

fn module_evaluation_completion(result: JsValue) -> stator_jse::error::StatorResult<JsValue> {
    match result {
        JsValue::Promise(promise) => match promise.state() {
            PromiseState::Fulfilled(value) => Ok(value),
            PromiseState::Rejected(reason) => Err(module_rejection_to_error(reason)),
            PromiseState::Pending => Ok(JsValue::Promise(promise)),
        },
        value => Ok(value),
    }
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
            error_kind: StatorMessageKind::StatorMessageKindInternal,
            resource_name: None,
            resource_line_offset: 0,
            resource_column_offset: 0,
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
                error_kind: StatorMessageKind::StatorMessageKindInternal,
                resource_name: None,
                resource_line_offset: 0,
                resource_column_offset: 0,
            });
            return Box::into_raw(script);
        }
    };

    // Parse then compile.
    let result =
        parser::parse(src).and_then(|program| BytecodeGenerator::compile_program(&program));

    let script = match result {
        Ok(bytecodes) => Box::new(StatorScript {
            bytecodes: Some(Rc::new(bytecodes)),
            error: None,
            error_kind: StatorMessageKind::StatorMessageKindUnknown,
            resource_name: None,
            resource_line_offset: 0,
            resource_column_offset: 0,
        }),
        Err(e) => {
            let msg = e.to_string();
            let cstring = CString::new(msg).unwrap_or_else(|_| c"compilation error".into());
            Box::new(StatorScript {
                bytecodes: None,
                error: Some(cstring),
                error_kind: StatorMessageKind::StatorMessageKindSyntax,
                resource_name: None,
                resource_line_offset: 0,
                resource_column_offset: 0,
            })
        }
    };
    Box::into_raw(script)
}

/// Compile `source` (a UTF-8 string of `source_len` bytes) as an ES module.
///
/// Returns a non-null [`StatorModule`] pointer in all cases (even on error).
/// Call [`stator_module_get_error`] to check whether compilation succeeded.
/// The caller must eventually pass the returned pointer to
/// [`stator_module_free`].
///
/// The resulting module can be evaluated with [`stator_module_evaluate`]
/// when it has no static imports or re-exports. Dependency-bearing modules
/// require host import resolution before evaluation.
///
/// # Safety
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - `source` must be valid for reads of `source_len` bytes of valid UTF-8.
unsafe fn compile_module_source(
    _ctx: *mut StatorContext,
    source: *const c_char,
    source_len: usize,
    source_type: StatorModuleType,
) -> *mut StatorModule {
    if source.is_null() {
        let module = Box::new(StatorModule {
            bytecodes: None,
            source_type,
            has_dependencies: false,
            module_requests: Vec::new(),
            direct_exports: HashSet::new(),
            status: StatorModuleStatus::StatorModuleStatusErrored,
            last_result: None,
            error: Some(c"null source pointer".into()),
            error_kind: StatorMessageKind::StatorMessageKindInternal,
            resource_name: None,
            resource_line_offset: 0,
            resource_column_offset: 0,
            base_url: None,
            integrity_metadata: None,
            credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
            referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
            parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
        });
        return Box::into_raw(module);
    }

    // SAFETY: caller guarantees `source` is valid for `source_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, source_len) };
    let src = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            let module = Box::new(StatorModule {
                bytecodes: None,
                source_type,
                has_dependencies: false,
                module_requests: Vec::new(),
                direct_exports: HashSet::new(),
                status: StatorModuleStatus::StatorModuleStatusErrored,
                last_result: None,
                error: Some(c"source is not valid UTF-8".into()),
                error_kind: StatorMessageKind::StatorMessageKindInternal,
                resource_name: None,
                resource_line_offset: 0,
                resource_column_offset: 0,
                base_url: None,
                integrity_metadata: None,
                credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
                referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
                parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
            });
            return Box::into_raw(module);
        }
    };

    if source_type != StatorModuleType::StatorModuleTypeJavaScript {
        let type_name = match source_type {
            StatorModuleType::StatorModuleTypeJavaScript => "JavaScript",
            StatorModuleType::StatorModuleTypeJson => "JSON",
            StatorModuleType::StatorModuleTypeWebAssembly => "WebAssembly",
            StatorModuleType::StatorModuleTypeCss => "CSS",
        };
        let module = Box::new(StatorModule {
            bytecodes: None,
            source_type,
            has_dependencies: false,
            module_requests: Vec::new(),
            direct_exports: HashSet::new(),
            status: StatorModuleStatus::StatorModuleStatusErrored,
            last_result: None,
            error: Some(
                CString::new(format!("{type_name} module source is not executable yet"))
                    .unwrap_or_else(|_| c"typed module source is not executable yet".into()),
            ),
            error_kind: StatorMessageKind::StatorMessageKindInternal,
            resource_name: None,
            resource_line_offset: 0,
            resource_column_offset: 0,
            base_url: None,
            integrity_metadata: None,
            credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
            referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
            parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
        });
        return Box::into_raw(module);
    }

    let result = parser::parse_module(src).and_then(|program| {
        let has_dependencies = program_has_module_dependencies(&program);
        let (module_requests, direct_exports) = collect_module_requests(&program);
        BytecodeGenerator::compile_program(&program)
            .map(|bytecodes| (bytecodes, has_dependencies, module_requests, direct_exports))
    });

    let module = match result {
        Ok((bytecodes, has_dependencies, module_requests, direct_exports)) => {
            Box::new(StatorModule {
                bytecodes: Some(Rc::new(bytecodes)),
                source_type,
                has_dependencies,
                module_requests,
                direct_exports,
                status: StatorModuleStatus::StatorModuleStatusUnlinked,
                last_result: None,
                error: None,
                error_kind: StatorMessageKind::StatorMessageKindUnknown,
                resource_name: None,
                resource_line_offset: 0,
                resource_column_offset: 0,
                base_url: None,
                integrity_metadata: None,
                credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
                referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
                parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
            })
        }
        Err(e) => {
            let msg = e.to_string();
            let cstring = CString::new(msg).unwrap_or_else(|_| c"module compilation error".into());
            Box::new(StatorModule {
                bytecodes: None,
                source_type,
                has_dependencies: false,
                module_requests: Vec::new(),
                direct_exports: HashSet::new(),
                status: StatorModuleStatus::StatorModuleStatusErrored,
                last_result: None,
                error: Some(cstring),
                error_kind: StatorMessageKind::StatorMessageKindSyntax,
                resource_name: None,
                resource_line_offset: 0,
                resource_column_offset: 0,
                base_url: None,
                integrity_metadata: None,
                credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
                referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
                parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
            })
        }
    };
    Box::into_raw(module)
}

/// Compile `source` (a UTF-8 string of `source_len` bytes) as a JavaScript ES module.
///
/// This is equivalent to [`stator_module_compile_typed`] with
/// [`StatorModuleType::StatorModuleTypeJavaScript`].
///
/// # Safety
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - `source` must be valid for reads of `source_len` bytes of valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_compile(
    ctx: *mut StatorContext,
    source: *const c_char,
    source_len: usize,
) -> *mut StatorModule {
    // SAFETY: forwards the caller-provided raw source slice to the shared
    // compiler using the JavaScript/default module type.
    unsafe {
        compile_module_source(
            ctx,
            source,
            source_len,
            StatorModuleType::StatorModuleTypeJavaScript,
        )
    }
}

/// Compile `source` as a module with host-provided source kind metadata.
///
/// JavaScript modules are parsed and compiled normally. JSON, WebAssembly, and
/// CSS module source kinds are recorded on the returned module handle but
/// currently produce an unsupported compile error because their module
/// evaluators are not implemented in Stator yet.
///
/// # Safety
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - `source` must be valid for reads of `source_len` bytes of valid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_compile_typed(
    ctx: *mut StatorContext,
    source: *const c_char,
    source_len: usize,
    source_type: StatorModuleType,
) -> *mut StatorModule {
    // SAFETY: forwards the caller-provided raw source slice to the shared
    // compiler while preserving the requested source type metadata.
    unsafe { compile_module_source(ctx, source, source_len, source_type) }
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

/// Attach origin metadata (resource name and offsets) to `script`.
///
/// Mirrors the embedder-side construction of `v8::ScriptOrigin`.  The
/// metadata is stored on the script and is intended to be surfaced through
/// future stack-trace and error-reporting APIs; today it is plumbing only.
///
/// `resource_name` may be null to clear an existing name; otherwise it must
/// be a valid, null-terminated UTF-8 C string.  The contents are copied; the
/// caller retains ownership of the input buffer.
///
/// Does nothing when `script` is null.
///
/// # Safety
/// - `script` must be either null or a valid, live [`StatorScript`] pointer.
/// - When non-null, `resource_name` must be a valid, null-terminated UTF-8
///   C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_set_origin(
    script: *mut StatorScript,
    resource_name: *const c_char,
    line_offset: i32,
    column_offset: i32,
) {
    if script.is_null() {
        return;
    }
    let name = if resource_name.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `resource_name` is a valid null-terminated string.
        let cstr = unsafe { CStr::from_ptr(resource_name) };
        Some(CString::from(cstr))
    };
    // SAFETY: caller guarantees `script` is valid.
    unsafe {
        (*script).resource_name = name;
        (*script).resource_line_offset = line_offset;
        (*script).resource_column_offset = column_offset;
    }
}

/// Return the resource name previously set on `script`, or null if none has
/// been set.
///
/// The returned pointer is valid as long as `script` is alive (i.e. until
/// [`stator_script_free`] is called) and as long as the origin is not
/// overwritten by another call to [`stator_script_set_origin`].
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_get_resource_name(
    script: *const StatorScript,
) -> *const c_char {
    if script.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `script` is valid.
    match unsafe { &(*script).resource_name } {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Return the line offset previously set on `script` (default 0).
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_get_line_offset(script: *const StatorScript) -> i32 {
    if script.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `script` is valid.
    unsafe { (*script).resource_line_offset }
}

/// Return the column offset previously set on `script` (default 0).
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_get_column_offset(script: *const StatorScript) -> i32 {
    if script.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `script` is valid.
    unsafe { (*script).resource_column_offset }
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

/// Per-script tiering state visible to embedders.
#[repr(C)]
pub struct StatorScriptTierStatus {
    /// Number of bytecode instructions in the script.
    pub bytecode_count: usize,
    /// Number of times the script/function has been invoked.
    pub invocation_count: u32,
    /// Whether baseline JIT code is cached.
    pub baseline_jit_compiled: bool,
    /// Whether Maglev JIT code is cached.
    pub maglev_jit_compiled: bool,
    /// Whether an executable Maglev entry is cached.
    pub maglev_executable_cached: bool,
    /// Whether Maglev compilation has been attempted.
    pub maglev_compile_attempted: bool,
    /// Number of Maglev deopts recorded for this script.
    pub maglev_deopt_count: u32,
    /// Invocation count at which Maglev may be retried.
    pub maglev_next_try_at: u32,
    /// Whether Turbofan JIT code is cached.
    pub turbofan_jit_compiled: bool,
    /// Whether this script is blocked from JIT tier execution.
    pub jit_disabled: bool,
}

/// Fill `*status` with tiering state for `script`.
///
/// Returns `false` when either pointer is null or the script failed to compile.
///
/// # Safety
/// - `script` must be null or a valid, live `StatorScript` pointer.
/// - `status` must be null or valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_get_tier_status(
    script: *const StatorScript,
    status: *mut StatorScriptTierStatus,
) -> bool {
    if script.is_null() || status.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return false,
    };
    // SAFETY: caller provided a valid output pointer.
    unsafe {
        *status = StatorScriptTierStatus {
            bytecode_count: bytecodes.bytecode_count(),
            invocation_count: bytecodes.invocation_count(),
            baseline_jit_compiled: bytecodes.has_baseline_jit_code(),
            maglev_jit_compiled: bytecodes.has_maglev_jit_code(),
            maglev_executable_cached: bytecodes.has_maglev_executable_cached(),
            maglev_compile_attempted: bytecodes.maglev_compile_attempted(),
            maglev_deopt_count: bytecodes.maglev_deopt_count(),
            maglev_next_try_at: bytecodes.maglev_next_try_at(),
            turbofan_jit_compiled: bytecodes.has_turbofan_jit_code(),
            jit_disabled: bytecodes.jit_disabled(),
        };
    }
    true
}

/// Synchronously force Maglev compilation for `script` when the platform supports it.
///
/// Returns `false` for null/failed scripts and on unsupported platforms.
///
/// # Safety
/// `script` must be null or a valid, live `StatorScript` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_force_maglev_compile(script: *mut StatorScript) -> bool {
    if script.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return false,
    };
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    {
        stator_jse::interpreter::compile_maglev_sync(bytecodes)
    }
    #[cfg(not(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    )))]
    {
        let _ = bytecodes;
        false
    }
}

/// Disable Maglev/Turbofan tier-up for a compiled script.
///
/// This is intended for embedders that need deterministic interpreter
/// execution for workloads whose JIT tier is known to be unstable.
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_disable_jit(script: *mut StatorScript) {
    if script.is_null() {
        return;
    }
    // SAFETY: caller guarantees `script` is valid.
    if let Some(bytecodes) = unsafe { &(*script).bytecodes } {
        bytecodes.set_jit_disabled(true);
    }
}

/// Reset JIT deopt backoff state for a compiled script.
///
/// This lets embedders perform a warmup phase, clear transient first-run deopts,
/// and then measure or execute with warmed inline caches. It does not discard
/// compiled JIT code.
///
/// # Safety
/// `script` must be either null or a valid, live [`StatorScript`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_reset_jit_deopts(script: *mut StatorScript) {
    if script.is_null() {
        return;
    }
    // SAFETY: caller guarantees `script` is valid.
    if let Some(bytecodes) = unsafe { &(*script).bytecodes } {
        bytecodes.reset_maglev_deopt_count();
        reset_stub_deopt_counts();
        reset_first_deopt_counts();
    }
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

/// Return a null-terminated error message if `module` compiled with an error.
///
/// Returns a null pointer when `module` compiled successfully. The returned
/// pointer is valid as long as `module` is alive.
///
/// # Safety
/// `module` must be a non-null pointer returned by [`stator_module_compile`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_error(module: *const StatorModule) -> *const c_char {
    if module.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `module` is valid.
    match unsafe { &(*module).error } {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Return the structured [`StatorMessageKind`] of `module`'s compile error.
///
/// Returns [`StatorMessageKind::StatorMessageKindUnknown`] when `module`
/// compiled successfully or is null.
///
/// # Safety
/// `module` must be either null or a valid pointer to a live [`StatorModule`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_error_kind(
    module: *const StatorModule,
) -> StatorMessageKind {
    if module.is_null() {
        return StatorMessageKind::StatorMessageKindUnknown;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).error_kind }
}

/// Attach origin metadata (resource name and offsets) to `module`.
///
/// `resource_name` may be null to clear an existing name; otherwise it must be
/// a valid, null-terminated UTF-8 C string. The contents are copied.
///
/// # Safety
/// - `module` must be either null or a valid, live [`StatorModule`] pointer.
/// - When non-null, `resource_name` must be a valid, null-terminated UTF-8
///   C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_set_origin(
    module: *mut StatorModule,
    resource_name: *const c_char,
    line_offset: i32,
    column_offset: i32,
) {
    if module.is_null() {
        return;
    }
    let name = if resource_name.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `resource_name` is a valid null-terminated string.
        let cstr = unsafe { CStr::from_ptr(resource_name) };
        Some(CString::from(cstr))
    };
    // SAFETY: caller guarantees `module` is valid.
    unsafe {
        (*module).resource_name = name;
        (*module).resource_line_offset = line_offset;
        (*module).resource_column_offset = column_offset;
    }
}

/// Return the resource name previously set on `module`, or null if none.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_resource_name(
    module: *const StatorModule,
) -> *const c_char {
    if module.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `module` is valid.
    match unsafe { &(*module).resource_name } {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Return the line offset previously set on `module` (default 0).
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_line_offset(module: *const StatorModule) -> i32 {
    if module.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).resource_line_offset }
}

/// Return the column offset previously set on `module` (default 0).
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_column_offset(module: *const StatorModule) -> i32 {
    if module.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).resource_column_offset }
}

/// Build a [`StatorModuleOrigin`] view borrowing into a module's stored
/// browser policy/origin metadata.
///
/// The returned struct's pointer fields borrow from `module` and are valid
/// only while `module` is alive and its metadata is not overwritten.
fn module_origin_view(module: &StatorModule) -> StatorModuleOrigin {
    let (base_url, base_url_len) = match module.base_url.as_ref() {
        Some(bytes) if !bytes.is_empty() => (bytes.as_ptr() as *const c_char, bytes.len()),
        _ => (std::ptr::null(), 0),
    };
    let (integrity_metadata, integrity_metadata_len) = match module.integrity_metadata.as_ref() {
        Some(bytes) if !bytes.is_empty() => (bytes.as_ptr() as *const c_char, bytes.len()),
        _ => (std::ptr::null(), 0),
    };
    StatorModuleOrigin {
        base_url,
        base_url_len,
        integrity_metadata,
        integrity_metadata_len,
        credentials_mode: module.credentials_mode,
        referrer_policy: module.referrer_policy,
        parser_metadata: module.parser_metadata,
    }
}

/// Attach browser policy/origin metadata to `module`.
///
/// Records the embedder's per-module-fetch context — base URL, Subresource
/// Integrity metadata, credentials mode, referrer policy, and parser metadata
/// — so that the host module resolver can apply same-origin, CORS, COEP, CSP
/// `script-src`, integrity, referrer-policy, and parser-metadata checks before
/// returning a resolved module from a static `import`, re-export, or
/// `import`-with-attributes request.
///
/// Strings (`base_url`, `integrity_metadata`) are copied into `module`; the
/// caller retains ownership of the input buffers and may free them as soon as
/// this call returns. The stored copies remain stable until either
/// [`stator_module_set_origin_metadata`] is called again or the module is
/// freed by [`stator_module_free`].
///
/// `base_url` / `integrity_metadata` may be null when the corresponding
/// length is zero to clear that field. Passing a null pointer with a non-zero
/// length is rejected as invalid input: the call returns `false` and the
/// previously stored metadata on `module` is preserved unchanged.
///
/// Returns `true` on success, `false` when `module` is null or when any
/// pointer/length pair is inconsistent.
///
/// # Safety
/// - `module` must be either null or a valid, live [`StatorModule`] pointer.
/// - When `base_url_len > 0`, `base_url` must be valid for reads of
///   `base_url_len` bytes.
/// - When `integrity_metadata_len > 0`, `integrity_metadata` must be valid
///   for reads of `integrity_metadata_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_set_origin_metadata(
    module: *mut StatorModule,
    base_url: *const c_char,
    base_url_len: usize,
    credentials_mode: StatorCredentialsMode,
    integrity_metadata: *const c_char,
    integrity_metadata_len: usize,
    referrer_policy: StatorReferrerPolicy,
    parser_metadata: StatorParserMetadata,
) -> bool {
    if module.is_null() {
        return false;
    }
    if base_url.is_null() && base_url_len != 0 {
        return false;
    }
    if integrity_metadata.is_null() && integrity_metadata_len != 0 {
        return false;
    }

    let base_url_owned = if base_url.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `base_url` is valid for `base_url_len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(base_url as *const u8, base_url_len) };
        Some(slice.to_vec())
    };
    let integrity_owned = if integrity_metadata.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `integrity_metadata` is valid for
        // `integrity_metadata_len` bytes.
        let slice = unsafe {
            std::slice::from_raw_parts(integrity_metadata as *const u8, integrity_metadata_len)
        };
        Some(slice.to_vec())
    };

    // SAFETY: caller guarantees `module` is valid.
    unsafe {
        (*module).base_url = base_url_owned;
        (*module).integrity_metadata = integrity_owned;
        (*module).credentials_mode = credentials_mode;
        (*module).referrer_policy = referrer_policy;
        (*module).parser_metadata = parser_metadata;
    }
    true
}

/// Return a pointer to the base-URL bytes previously set on `module`, or null
/// when no base URL is set.
///
/// `out_len`, when non-null, receives the byte length of the base URL (zero
/// when the return value is null). The bytes are not null-terminated and are
/// valid as long as `module` is alive and the metadata is not overwritten by
/// another call to [`stator_module_set_origin_metadata`].
///
/// # Safety
/// - `module` must be either null or a valid, live [`StatorModule`] pointer.
/// - `out_len`, when non-null, must be valid for one `usize` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_base_url(
    module: *const StatorModule,
    out_len: *mut usize,
) -> *const c_char {
    if module.is_null() {
        if !out_len.is_null() {
            // SAFETY: caller guarantees `out_len` is valid.
            unsafe { *out_len = 0 };
        }
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `module` is valid.
    let bytes = unsafe { (*module).base_url.as_ref() };
    let (ptr, len) = match bytes {
        Some(b) if !b.is_empty() => (b.as_ptr() as *const c_char, b.len()),
        _ => (std::ptr::null(), 0),
    };
    if !out_len.is_null() {
        // SAFETY: caller guarantees `out_len` is valid for one write.
        unsafe { *out_len = len };
    }
    ptr
}

/// Return a pointer to the integrity-metadata bytes previously set on
/// `module`, or null when none is set.
///
/// `out_len`, when non-null, receives the byte length (zero when the return
/// value is null). The bytes are not null-terminated and are valid as long as
/// `module` is alive and the metadata is not overwritten by another call to
/// [`stator_module_set_origin_metadata`].
///
/// # Safety
/// - `module` must be either null or a valid, live [`StatorModule`] pointer.
/// - `out_len`, when non-null, must be valid for one `usize` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_integrity_metadata(
    module: *const StatorModule,
    out_len: *mut usize,
) -> *const c_char {
    if module.is_null() {
        if !out_len.is_null() {
            // SAFETY: caller guarantees `out_len` is valid.
            unsafe { *out_len = 0 };
        }
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `module` is valid.
    let bytes = unsafe { (*module).integrity_metadata.as_ref() };
    let (ptr, len) = match bytes {
        Some(b) if !b.is_empty() => (b.as_ptr() as *const c_char, b.len()),
        _ => (std::ptr::null(), 0),
    };
    if !out_len.is_null() {
        // SAFETY: caller guarantees `out_len` is valid for one write.
        unsafe { *out_len = len };
    }
    ptr
}

/// Return the credentials mode previously set on `module`, or
/// [`StatorCredentialsMode::StatorCredentialsModeDefault`] when unset.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_credentials_mode(
    module: *const StatorModule,
) -> StatorCredentialsMode {
    if module.is_null() {
        return StatorCredentialsMode::StatorCredentialsModeDefault;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).credentials_mode }
}

/// Return the referrer policy previously set on `module`, or
/// [`StatorReferrerPolicy::StatorReferrerPolicyDefault`] when unset.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_referrer_policy(
    module: *const StatorModule,
) -> StatorReferrerPolicy {
    if module.is_null() {
        return StatorReferrerPolicy::StatorReferrerPolicyDefault;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).referrer_policy }
}

/// Return the parser-metadata classification previously set on `module`, or
/// [`StatorParserMetadata::StatorParserMetadataNotParserInserted`] when unset.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_parser_metadata(
    module: *const StatorModule,
) -> StatorParserMetadata {
    if module.is_null() {
        return StatorParserMetadata::StatorParserMetadataNotParserInserted;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).parser_metadata }
}

/// Return the number of bytecode instructions in a compiled module.
///
/// Returns 0 if `module` is null, failed to compile, or cannot be decoded.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_bytecode_count(module: *const StatorModule) -> usize {
    if module.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `module` is valid.
    let bytecodes = match unsafe { &(*module).bytecodes } {
        Some(b) => b,
        None => return 0,
    };
    decode(bytecodes.bytecodes())
        .map(|instrs| instrs.len())
        .unwrap_or(0)
}

/// Return whether the compiled module bytecode is marked as async.
///
/// Modules are compiled with async capability so hosts can later drive
/// top-level-await evaluation. Returns false for null or failed modules.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_is_async(module: *const StatorModule) -> bool {
    if module.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).bytecodes.as_ref().is_some_and(|b| b.is_async()) }
}

/// Return whether `module` contains static imports or re-exports.
///
/// Dependency-bearing modules require a host resolver/linker before this FFI
/// layer can safely evaluate them. Dependency-free modules can be evaluated by
/// [`stator_module_evaluate`].
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_has_dependencies(module: *const StatorModule) -> bool {
    if module.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).has_dependencies }
}

/// Return the number of static import/re-export requests in `module`.
///
/// Requests are reported in source order. Returns 0 for null or failed modules.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_request_count(module: *const StatorModule) -> usize {
    if module.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).module_requests.len() }
}

/// Return one static module request's specifier and import attributes.
///
/// All out-pointers are optional. Returned pointers are borrowed from `module`
/// and remain valid until [`stator_module_free`] or recompilation of a future
/// replacement handle. The specifier is not null-terminated; use
/// `out_specifier_len`.
///
/// Returns `false` when `module` is null or `idx` is out of range.
///
/// # Safety
/// - `module` must be either null or a valid, live [`StatorModule`] pointer.
/// - Each non-null out-pointer must be valid for one pointer/length write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_request(
    module: *const StatorModule,
    idx: usize,
    out_specifier: *mut *const c_char,
    out_specifier_len: *mut usize,
    out_attributes: *mut *const StatorImportAttribute,
    out_attributes_len: *mut usize,
) -> bool {
    if module.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `module` is valid.
    let module_ref = unsafe { &*module };
    let request = match module_ref.module_requests.get(idx) {
        Some(request) => request,
        None => return false,
    };

    if !out_specifier.is_null() {
        // SAFETY: caller guarantees the out pointer is valid for one write.
        unsafe { *out_specifier = request.specifier.as_ptr() };
    }
    if !out_specifier_len.is_null() {
        // SAFETY: caller guarantees the out pointer is valid for one write.
        unsafe { *out_specifier_len = request.specifier.as_bytes().len() };
    }
    if !out_attributes.is_null() {
        // SAFETY: caller guarantees the out pointer is valid for one write.
        unsafe {
            *out_attributes = if request.attributes.is_empty() {
                std::ptr::null()
            } else {
                request.attributes.as_ptr()
            };
        }
    }
    if !out_attributes_len.is_null() {
        // SAFETY: caller guarantees the out pointer is valid for one write.
        unsafe { *out_attributes_len = request.attributes.len() };
    }
    true
}

/// Return the current link/evaluation status for `module`.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_status(
    module: *const StatorModule,
) -> StatorModuleStatus {
    if module.is_null() {
        return StatorModuleStatus::StatorModuleStatusErrored;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).status }
}

/// Return the source kind metadata recorded on `module`.
///
/// Null module pointers report JavaScript so embedders can treat legacy/default
/// paths as JavaScript-compatible.
///
/// # Safety
/// `module` must be either null or a valid, live [`StatorModule`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_get_type(module: *const StatorModule) -> StatorModuleType {
    if module.is_null() {
        return StatorModuleType::StatorModuleTypeJavaScript;
    }
    // SAFETY: caller guarantees `module` is valid.
    unsafe { (*module).source_type }
}

unsafe fn module_result_to_value(ctx: *mut StatorContext, val: &JsValue) -> *mut StatorValue {
    let isolate = if ctx.is_null() {
        std::ptr::null_mut()
    } else {
        // SAFETY: caller guarantees `ctx` is valid when non-null.
        unsafe { (*ctx)._isolate }
    };
    let inner = jsvalue_to_stator_value_inner(val);
    if isolate.is_null() {
        Box::into_raw(Box::new(StatorValue { inner, isolate }))
    } else {
        // SAFETY: `isolate` comes from a valid context and owns the new value.
        unsafe { allocate_stator_value(isolate, inner) }
    }
}

unsafe fn set_module_error(
    module: &mut StatorModule,
    ctx: *mut StatorContext,
    error: &stator_jse::error::StatorError,
) {
    let message = error.to_string();
    let terminating = unsafe { context_is_execution_terminating(ctx) };
    let kind = message_kind_for_error(error, terminating);
    module.status = StatorModuleStatus::StatorModuleStatusErrored;
    module.last_result = None;
    module.error = Some(CString::new(message.clone()).unwrap_or_else(|_| c"module error".into()));
    module.error_kind = kind;

    if ctx.is_null() {
        return;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let isolate = unsafe { (*ctx)._isolate };
    if isolate.is_null() || unsafe { isolate_has_pending_exception(isolate) } {
        return;
    }

    // SAFETY: `isolate` is valid and `message` is copied by the callee.
    let exception = unsafe {
        stator_value_new_string(isolate, message.as_ptr() as *const c_char, message.len())
    };
    // SAFETY: `exception` was allocated for ownership by the pending exception.
    unsafe { set_pending_exception(isolate, exception, true) };

    let structured = Box::new(StatorMessage {
        kind,
        text: CString::new(message).unwrap_or_else(|_| c"module error".into()),
        resource_name: module.resource_name.clone(),
        line: None,
        column: None,
        terminated: terminating || matches!(kind, StatorMessageKind::StatorMessageKindTermination),
    });
    // SAFETY: `isolate` is valid.
    unsafe { (*isolate).pending_message = Some(structured) };
}

/// Convert a resolver `out_error` pointer into an owned `String` and free the
/// FFI handle.
///
/// Returns `None` when `out_error` is null or contains no bytes. Invalid UTF-8
/// is replaced with the Unicode replacement character so that non-conforming
/// hosts still surface a printable diagnostic instead of being collapsed to a
/// generic engine message.
unsafe fn take_resolver_error_string(out_error: *mut StatorString) -> Option<String> {
    if out_error.is_null() {
        return None;
    }
    // SAFETY: hosts transfer ownership of `out_error` to the engine on return.
    let owned = unsafe { Box::from_raw(out_error) };
    if owned.bytes.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(&owned.bytes).into_owned())
}

fn resolver_status_error(
    status: StatorResolveStatus,
    specifier: &str,
    detail: Option<&str>,
) -> stator_jse::error::StatorError {
    use stator_jse::error::StatorError;
    let suffix = match detail {
        Some(message) if !message.is_empty() => format!(": {message}"),
        _ => String::new(),
    };
    match status {
        StatorResolveStatus::StatorResolveStatusOk => StatorError::Internal(format!(
            "module resolver returned ok without a module for '{specifier}'{suffix}"
        )),
        StatorResolveStatus::StatorResolveStatusNotFound => StatorError::ReferenceError(format!(
            "module specifier '{specifier}' was not found{suffix}"
        )),
        StatorResolveStatus::StatorResolveStatusNetworkError => {
            StatorError::TypeError(format!("failed to fetch module '{specifier}'{suffix}"))
        }
        StatorResolveStatus::StatorResolveStatusTypeError => StatorError::TypeError(format!(
            "module specifier '{specifier}' was rejected by resolver{suffix}"
        )),
        StatorResolveStatus::StatorResolveStatusPending => StatorError::Internal(format!(
            "module specifier '{specifier}' requires asynchronous resolution{suffix}"
        )),
    }
}

fn js_error_from_stator_error(error: stator_jse::error::StatorError) -> JsError {
    use stator_jse::error::StatorError;
    match error {
        StatorError::TypeError(message) => JsError::new(ErrorKind::TypeError, message),
        StatorError::SyntaxError(message) => JsError::new(ErrorKind::SyntaxError, message),
        StatorError::ReferenceError(message) => JsError::new(ErrorKind::ReferenceError, message),
        StatorError::RangeError(message) => JsError::new(ErrorKind::RangeError, message),
        StatorError::URIError(message) => JsError::new(ErrorKind::URIError, message),
        other => JsError::new(ErrorKind::TypeError, other.to_string()),
    }
}

struct FfiHostModuleLoader {
    ctx: *mut StatorContext,
    referrer: *mut StatorModule,
}

impl FfiHostModuleLoader {
    fn resolve_module(&self, specifier: &str) -> Result<*mut StatorModule, Box<JsError>> {
        if self.ctx.is_null() || self.referrer.is_null() {
            return Err(Box::new(JsError::new(
                ErrorKind::TypeError,
                "module resolver is not installed".to_string(),
            )));
        }

        // SAFETY: `ctx` is valid for the active module evaluation and access is
        // serialized by the FFI contract.
        let (callback, user_data) = match unsafe { (*self.ctx).module_resolver.as_ref() } {
            Some(resolver) => (resolver.callback, resolver.user_data),
            None => {
                return Err(Box::new(JsError::new(
                    ErrorKind::TypeError,
                    "module resolver is not installed".to_string(),
                )));
            }
        };

        // SAFETY: `referrer` is the live module currently being evaluated.
        let origin = unsafe { module_origin_view(&*self.referrer) };
        let mut out_module: *mut StatorModule = std::ptr::null_mut();
        let mut out_error: *mut StatorString = std::ptr::null_mut();
        let specifier_ptr = specifier.as_ptr() as *const c_char;
        // SAFETY: resolver registration guarantees the callback remains
        // callable while installed. `specifier` and `origin` are valid for this
        // synchronous call, attributes are empty for dynamic import, and out
        // pointers are valid locals.
        let status = unsafe {
            callback(
                self.ctx,
                user_data,
                self.referrer,
                &origin,
                specifier_ptr,
                specifier.len(),
                std::ptr::null(),
                0,
                &mut out_module,
                &mut out_error,
            )
        };
        // SAFETY: resolver callbacks transfer any non-null detail string to us.
        let detail = unsafe { take_resolver_error_string(out_error) };

        if status != StatorResolveStatus::StatorResolveStatusOk {
            return Err(Box::new(js_error_from_stator_error(resolver_status_error(
                status,
                specifier,
                detail.as_deref(),
            ))));
        }
        if out_module.is_null() {
            return Err(Box::new(js_error_from_stator_error(resolver_status_error(
                StatorResolveStatus::StatorResolveStatusNotFound,
                specifier,
                detail.as_deref(),
            ))));
        }
        Ok(out_module)
    }
}

impl HostModuleLoader for FfiHostModuleLoader {
    fn dynamic_import(&self, specifier: &str, _referrer: Option<&str>) -> Result<JsValue, JsError> {
        let module = self.resolve_module(specifier).map_err(|error| *error)?;
        let mut visiting = HashSet::new();
        // SAFETY: the resolver returned a live module pointer by contract.
        if let Err(error) = unsafe { instantiate_module_graph(self.ctx, module, &mut visiting) } {
            return Err(js_error_from_stator_error(error));
        }

        // SAFETY: the resolver returned a live module pointer by contract.
        let result = unsafe { stator_module_evaluate(module, self.ctx) };
        if result.is_null() {
            // SAFETY: `module` is live for the duration of this synchronous
            // dynamic import.
            let error = unsafe { module_stored_error(&*module) };
            return Err(js_error_from_stator_error(error));
        }
        // SAFETY: `stator_module_evaluate` returned a value owned by us.
        unsafe { stator_value_destroy(result) };
        Ok(JsValue::PlainObject(Rc::new(RefCell::new(
            PropertyMap::new(),
        ))))
    }

    fn resolve(&self, specifier: &str, _referrer: Option<&str>) -> Result<String, JsError> {
        let module = self.resolve_module(specifier).map_err(|error| *error)?;
        // SAFETY: the resolver returned a live module pointer by contract.
        let module_ref = unsafe { &*module };
        Ok(module_ref
            .resource_name
            .as_ref()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| specifier.to_string()))
    }
}

fn set_module_link_error(module: &mut StatorModule, error: &stator_jse::error::StatorError) {
    module.status = StatorModuleStatus::StatorModuleStatusErrored;
    module.last_result = None;
    module.error = Some(CString::new(error.to_string()).unwrap_or_else(|_| c"module error".into()));
    module.error_kind = message_kind_for_error(error, false);
}

fn module_stored_error(module: &StatorModule) -> stator_jse::error::StatorError {
    let message = module
        .error
        .as_ref()
        .map(|error| error.to_string_lossy().into_owned())
        .unwrap_or_else(|| "module is already errored".to_string());
    match module.error_kind {
        StatorMessageKind::StatorMessageKindSyntax => {
            stator_jse::error::StatorError::SyntaxError(message)
        }
        StatorMessageKind::StatorMessageKindType => {
            stator_jse::error::StatorError::TypeError(message)
        }
        StatorMessageKind::StatorMessageKindReference => {
            stator_jse::error::StatorError::ReferenceError(message)
        }
        StatorMessageKind::StatorMessageKindRange => {
            stator_jse::error::StatorError::RangeError(message)
        }
        StatorMessageKind::StatorMessageKindURI => {
            stator_jse::error::StatorError::URIError(message)
        }
        StatorMessageKind::StatorMessageKindInternal
        | StatorMessageKind::StatorMessageKindTermination
        | StatorMessageKind::StatorMessageKindWasm
        | StatorMessageKind::StatorMessageKindJsException
        | StatorMessageKind::StatorMessageKindOutOfMemory
        | StatorMessageKind::StatorMessageKindSandboxViolation
        | StatorMessageKind::StatorMessageKindUnknown => {
            stator_jse::error::StatorError::Internal(message)
        }
    }
}

unsafe fn instantiate_module_graph(
    ctx: *mut StatorContext,
    module: *mut StatorModule,
    visiting: &mut HashSet<*mut StatorModule>,
) -> Result<(), stator_jse::error::StatorError> {
    if module.is_null() {
        return Err(stator_jse::error::StatorError::Internal(
            "module resolver returned null module".to_string(),
        ));
    }

    if visiting.contains(&module) {
        return Ok(());
    }

    // SAFETY: caller guarantees `module` is a live module pointer.
    let module_ref = unsafe { &mut *module };
    match module_ref.status {
        StatorModuleStatus::StatorModuleStatusLinked
        | StatorModuleStatus::StatorModuleStatusEvaluated => return Ok(()),
        StatorModuleStatus::StatorModuleStatusErrored => {
            return Err(module_stored_error(module_ref));
        }
        StatorModuleStatus::StatorModuleStatusEvaluating
        | StatorModuleStatus::StatorModuleStatusLinking => return Ok(()),
        StatorModuleStatus::StatorModuleStatusUnlinked => {}
    }

    module_ref.status = StatorModuleStatus::StatorModuleStatusLinking;
    visiting.insert(module);

    if module_ref.module_requests.is_empty() {
        module_ref.status = StatorModuleStatus::StatorModuleStatusLinked;
        visiting.remove(&module);
        return Ok(());
    }

    if ctx.is_null() {
        visiting.remove(&module);
        let error = stator_jse::error::StatorError::Internal(
            "module instantiation requires a context with a module resolver".to_string(),
        );
        set_module_link_error(module_ref, &error);
        return Err(error);
    }

    // SAFETY: caller guarantees `ctx` is valid.
    let (callback, user_data) = match unsafe { (*ctx).module_resolver.as_ref() } {
        Some(resolver) => (resolver.callback, resolver.user_data),
        None => {
            visiting.remove(&module);
            let error = stator_jse::error::StatorError::Internal(
                "module instantiation requires a module resolver".to_string(),
            );
            set_module_link_error(module_ref, &error);
            return Err(error);
        }
    };

    let origin = module_origin_view(module_ref);
    for request in &module_ref.module_requests {
        let mut out_module: *mut StatorModule = std::ptr::null_mut();
        let mut out_error: *mut StatorString = std::ptr::null_mut();
        let attributes_ptr = if request.attributes.is_empty() {
            std::ptr::null()
        } else {
            request.attributes.as_ptr()
        };
        // SAFETY: resolver registration requires the callback to remain callable
        // while installed. Request slices and origin metadata are borrowed from
        // `module` for the duration of this synchronous call, and out pointers
        // are valid locals.
        let status = unsafe {
            callback(
                ctx,
                user_data,
                module,
                &origin,
                request.specifier.as_ptr(),
                request.specifier.as_bytes().len(),
                attributes_ptr,
                request.attributes.len(),
                &mut out_module,
                &mut out_error,
            )
        };

        let specifier = request.specifier.to_string_lossy();
        // SAFETY: hosts may transfer ownership of an error string via
        // `out_error`. We always reclaim and free it here, even on success.
        let detail = unsafe { take_resolver_error_string(out_error) };
        if status != StatorResolveStatus::StatorResolveStatusOk {
            visiting.remove(&module);
            let error = resolver_status_error(status, &specifier, detail.as_deref());
            set_module_link_error(module_ref, &error);
            return Err(error);
        }
        if out_module.is_null() {
            visiting.remove(&module);
            let error = resolver_status_error(
                StatorResolveStatus::StatorResolveStatusNotFound,
                &specifier,
                detail.as_deref(),
            );
            set_module_link_error(module_ref, &error);
            return Err(error);
        }

        // SAFETY: the resolver returned a live module pointer by contract.
        if let Err(error) = unsafe { instantiate_module_graph(ctx, out_module, visiting) } {
            visiting.remove(&module);
            set_module_link_error(module_ref, &error);
            return Err(error);
        }
        request.resolved.set(out_module);
    }

    module_ref.status = StatorModuleStatus::StatorModuleStatusLinked;
    visiting.remove(&module);
    Ok(())
}

/// Returns true if `dep` directly or transitively (via `export *` re-exports)
/// exposes an export named `name`. Default exports are never propagated through
/// bare star re-exports, matching ECMAScript module semantics.
///
/// # Safety
/// `dep` must be a non-null pointer to a live, linked module record.
unsafe fn dep_has_export(dep: *mut StatorModule, name: &str) -> bool {
    let mut visited: HashSet<*mut StatorModule> = HashSet::new();
    unsafe { dep_has_export_inner(dep, name, &mut visited) }
}

unsafe fn dep_has_export_inner(
    dep: *mut StatorModule,
    name: &str,
    visited: &mut HashSet<*mut StatorModule>,
) -> bool {
    if dep.is_null() || !visited.insert(dep) {
        return false;
    }
    // SAFETY: caller guarantees `dep` is a live module pointer.
    let module = unsafe { &*dep };
    if module.direct_exports.contains(name) {
        return true;
    }
    if name == "default" {
        return false;
    }
    for request in &module.module_requests {
        if matches!(request.imports.first(), Some(RequestedExport::Star))
            && unsafe { dep_has_export_inner(request.resolved.get(), name, visited) }
        {
            return true;
        }
    }
    false
}

fn missing_export_error(specifier: &str, name: &str) -> stator_jse::error::StatorError {
    stator_jse::error::StatorError::SyntaxError(format!(
        "module '{specifier}' has no export named '{name}'"
    ))
}

fn module_cell_for_binding(binding: &str) -> i32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in binding.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    (hash & 0x3fff_ffff) as i32
}

fn module_cell_key(specifier: &str, binding: &str) -> String {
    format!("__mod:{specifier}:{}", module_cell_for_binding(binding))
}

fn requested_export_name(export: &RequestedExport) -> Option<&str> {
    match export {
        RequestedExport::Default => Some("default"),
        RequestedExport::Named(name) => Some(name),
        RequestedExport::Namespace | RequestedExport::Star => None,
    }
}

fn uninitialized_module_binding_error(binding: &str) -> stator_jse::error::StatorError {
    stator_jse::error::StatorError::ReferenceError(format!(
        "Cannot access imported binding '{binding}' before initialization"
    ))
}

fn sync_module_request_bindings(
    request: &StatorModuleRequest,
    global_env: &Rc<RefCell<GlobalEnv>>,
) -> Result<(), stator_jse::error::StatorError> {
    let specifier = request.specifier.to_string_lossy();
    for import in &request.imports {
        let Some(binding) = requested_export_name(import) else {
            continue;
        };
        let export_key = module_cell_key("", binding);
        let value = global_env
            .borrow()
            .get(&export_key)
            .cloned()
            .ok_or_else(|| uninitialized_module_binding_error(binding))?;
        let import_key = module_cell_key(&specifier, binding);
        global_env.borrow_mut().insert(import_key, value);
    }
    Ok(())
}

/// Build (and publish into `global_env`) the module namespace object that
/// the importer's `GetModuleNamespace` opcode will load for `request`.
///
/// The namespace object collects every name in the dependency module's
/// `direct_exports` set, reading each value out of the dependency's "" cell
/// (the convention used by [`sync_module_request_bindings`]). It is frozen
/// and marked non-extensible so importers cannot mutate or extend the
/// namespace.  Re-exported star bindings sourced from grand-dependencies are
/// not included yet — this slice only covers names the dependency's source
/// declares directly.
fn publish_module_namespace(request: &StatorModuleRequest, global_env: &Rc<RefCell<GlobalEnv>>) {
    let dep = request.resolved.get();
    if dep.is_null() {
        return;
    }
    // SAFETY: `dep` is a borrowed pointer kept alive by the linked module
    // graph. We only read immutable metadata while no other code holds a
    // mutable borrow of the module record.
    let exports: Vec<String> = unsafe { (*dep).direct_exports.iter().cloned().collect() };
    let specifier = request.specifier.to_string_lossy();
    let mut ns = PropertyMap::new();
    for name in exports {
        let cell = module_cell_for_binding(&name);
        let value_key = format!("__mod::{cell}");
        let value = global_env
            .borrow()
            .get(&value_key)
            .cloned()
            .unwrap_or(JsValue::Undefined);
        ns.insert_with_attrs(name, value, PropertyAttributes::ENUMERABLE);
    }
    ns.freeze();
    let ns_value = JsValue::PlainObject(Rc::new(RefCell::new(ns)));
    let ns_key = format!("__mod_ns:{specifier}");
    global_env.borrow_mut().insert(ns_key, ns_value);
}

/// Walk the linked module graph rooted at `root` and ensure every static
/// `import`/`export … from` references a binding the source module actually
/// exports. Cycles are tracked via `validated`. Star re-exports are followed
/// transitively (ignoring `default`) so that `import { x } from "a"` is
/// satisfied when `a` does `export * from "b"` and `b` declares `x`.
///
/// # Safety
/// `root` must be a non-null pointer to a live, fully-linked module record.
/// Each request's `resolved` pointer must be set (this is true after a
/// successful [`instantiate_module_graph`]).
unsafe fn validate_module_graph_exports(
    root: *mut StatorModule,
    validated: &mut HashSet<*mut StatorModule>,
) -> Result<(), stator_jse::error::StatorError> {
    if root.is_null() || !validated.insert(root) {
        return Ok(());
    }
    // SAFETY: caller guarantees `root` is a live module pointer.
    let module = unsafe { &*root };
    for request in &module.module_requests {
        let dep = request.resolved.get();
        if dep.is_null() {
            continue;
        }
        let specifier = request.specifier.to_string_lossy();
        for imp in &request.imports {
            match imp {
                RequestedExport::Default => {
                    // SAFETY: `dep` is a live linked module pointer.
                    let dep_ref = unsafe { &*dep };
                    if !dep_ref.direct_exports.contains("default") {
                        return Err(missing_export_error(&specifier, "default"));
                    }
                }
                RequestedExport::Named(name) => {
                    // SAFETY: `dep` is a live linked module pointer.
                    if !unsafe { dep_has_export(dep, name) } {
                        return Err(missing_export_error(&specifier, name));
                    }
                }
                RequestedExport::Namespace | RequestedExport::Star => {}
            }
        }
        // SAFETY: `dep` is a live linked module pointer.
        unsafe { validate_module_graph_exports(dep, validated)? };
    }
    Ok(())
}

/// Link a compiled module graph by resolving all static import/re-export requests.
///
/// Dependency-free modules transition to `Linked` without a context. Modules
/// with requests require a resolver registered on `ctx`; the resolver is invoked
/// depth-first and cycles are accepted without infinite recursion. This slice
/// validates graph availability only: evaluation of dependency-bearing modules
/// still returns an explicit error until live binding wiring lands.
///
/// Returns `true` on successful graph instantiation. On failure returns `false`
/// and stores an error on the root module.
///
/// # Safety
/// - `module` must be a non-null pointer returned by [`stator_module_compile`].
/// - `ctx` must be null only for dependency-free modules, or a valid live
///   [`StatorContext`] pointer for modules with requests.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_instantiate(
    ctx: *mut StatorContext,
    module: *mut StatorModule,
) -> bool {
    if module.is_null() {
        return false;
    }
    let mut visiting = HashSet::new();
    match unsafe { instantiate_module_graph(ctx, module, &mut visiting) } {
        Ok(()) => {
            let mut validated = HashSet::new();
            // SAFETY: `module` is a non-null, fully-linked module record.
            match unsafe { validate_module_graph_exports(module, &mut validated) } {
                Ok(()) => true,
                Err(error) => {
                    // SAFETY: `module` is non-null and valid by function contract.
                    let module_ref = unsafe { &mut *module };
                    // SAFETY: `module_ref` and `ctx` are valid for this call.
                    unsafe { set_module_error(module_ref, ctx, &error) };
                    false
                }
            }
        }
        Err(error) => {
            // SAFETY: `module` is non-null and valid by function contract.
            let module_ref = unsafe { &mut *module };
            // SAFETY: `module_ref` and `ctx` are valid for this call.
            unsafe { set_module_error(module_ref, ctx, &error) };
            false
        }
    }
}

unsafe fn run_module_bytecodes(
    ctx: *mut StatorContext,
    module: *mut StatorModule,
    bytecodes: Rc<BytecodeArray>,
    global_env: &Rc<RefCell<GlobalEnv>>,
) -> stator_jse::error::StatorResult<JsValue> {
    if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        let isolate = unsafe { (*ctx)._isolate };
        if !isolate.is_null() {
            // SAFETY: context owns a live isolate pointer by construction.
            if unsafe { (*isolate).jit_disabled } {
                bytecodes.set_jit_disabled(true);
            }
        }
    }

    // SAFETY: caller guarantees `module` is a live module pointer.
    let module_ref = unsafe { &*module };
    let module_url = module_ref
        .resource_name
        .as_ref()
        .map(|cs| cs.to_string_lossy().into_owned());
    let host_loader = if ctx.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `ctx` is valid when non-null.
        let has_resolver = unsafe { (*ctx).module_resolver.is_some() };
        has_resolver.then(|| {
            Rc::new(FfiHostModuleLoader {
                ctx,
                referrer: module,
            }) as Rc<dyn HostModuleLoader>
        })
    };
    let _host_scope = stator_jse::host::HostScope::install(host_loader, module_url.as_deref());

    let result = if global_env.borrow().globals_installed {
        Interpreter::run_fast(&bytecodes, &[], global_env)
    } else {
        let mut frame = InterpreterFrame::new_with_globals(
            Rc::clone(&bytecodes),
            vec![],
            Rc::clone(global_env),
        );
        Interpreter::run(&mut frame)
    };
    result.and_then(module_evaluation_completion)
}

unsafe fn evaluate_module_graph(
    ctx: *mut StatorContext,
    module: *mut StatorModule,
    global_env: &Rc<RefCell<GlobalEnv>>,
    active: &mut HashSet<*mut StatorModule>,
) -> stator_jse::error::StatorResult<JsValue> {
    if module.is_null() {
        return Err(stator_jse::error::StatorError::Internal(
            "null module in evaluation graph".to_string(),
        ));
    }

    // SAFETY: caller guarantees `module` is valid for this evaluation.
    let status = unsafe { (*module).status };
    match status {
        StatorModuleStatus::StatorModuleStatusEvaluated => {
            // SAFETY: `module` is valid and evaluated.
            return Ok(unsafe { (*module).last_result.clone() }.unwrap_or(JsValue::Undefined));
        }
        StatorModuleStatus::StatorModuleStatusErrored => {
            return Err(stator_jse::error::StatorError::Internal(
                "module has already failed evaluation".to_string(),
            ));
        }
        StatorModuleStatus::StatorModuleStatusEvaluating => {
            return Err(stator_jse::error::StatorError::ReferenceError(
                "Cannot access cyclic module before initialization".to_string(),
            ));
        }
        StatorModuleStatus::StatorModuleStatusLinking => {
            return Err(stator_jse::error::StatorError::Internal(
                "module is already linking".to_string(),
            ));
        }
        StatorModuleStatus::StatorModuleStatusUnlinked
        | StatorModuleStatus::StatorModuleStatusLinked => {}
    }

    if !active.insert(module) {
        return Err(stator_jse::error::StatorError::ReferenceError(
            "Cannot access cyclic module before initialization".to_string(),
        ));
    }

    // SAFETY: `module` is valid and uniquely driven by this evaluation call.
    let bytecodes = match unsafe { &*module }.bytecodes.as_ref().map(Rc::clone) {
        Some(bytecodes) => bytecodes,
        None => {
            active.remove(&module);
            return Err(stator_jse::error::StatorError::Internal(
                "module has no bytecode".to_string(),
            ));
        }
    };

    // SAFETY: `module` is valid and uniquely driven by this evaluation call.
    let module_ref = unsafe { &mut *module };
    if module_ref.has_dependencies
        && matches!(
            module_ref.status,
            StatorModuleStatus::StatorModuleStatusUnlinked
        )
    {
        active.remove(&module);
        return Err(stator_jse::error::StatorError::Internal(
            "module evaluation requires host import resolution".to_string(),
        ));
    }
    module_ref.status = StatorModuleStatus::StatorModuleStatusEvaluating;

    let result = (|| {
        // SAFETY: `module` remains valid throughout evaluation.
        let request_len = unsafe { (*module).module_requests.len() };
        for request_idx in 0..request_len {
            // SAFETY: request index is in bounds and module is valid.
            let dep = unsafe { (&(*module).module_requests)[request_idx].resolved.get() };
            if dep.is_null() {
                return Err(stator_jse::error::StatorError::Internal(
                    "module evaluation requires host import resolution".to_string(),
                ));
            }
            // SAFETY: dependency pointers are supplied by successful graph instantiation.
            unsafe { evaluate_module_graph(ctx, dep, global_env, active)? };
            // SAFETY: request index is in bounds and module is valid.
            let request = unsafe { &(&(*module).module_requests)[request_idx] };
            sync_module_request_bindings(request, global_env)?;
            publish_module_namespace(request, global_env);
        }
        // SAFETY: bytecode and module are valid for this evaluation.
        unsafe { run_module_bytecodes(ctx, module, bytecodes, global_env) }
    })();

    active.remove(&module);
    match result {
        Ok(result) => {
            // SAFETY: `module` is valid and uniquely driven by this evaluation call.
            let module_ref = unsafe { &mut *module };
            module_ref.status = StatorModuleStatus::StatorModuleStatusEvaluated;
            module_ref.last_result = Some(result.clone());
            module_ref.error = None;
            module_ref.error_kind = StatorMessageKind::StatorMessageKindUnknown;
            Ok(result)
        }
        Err(error) => {
            // SAFETY: `module` is valid and uniquely driven by this evaluation call.
            let module_ref = unsafe { &mut *module };
            // SAFETY: `module_ref` and `ctx` are valid for this call.
            unsafe { set_module_error(module_ref, ctx, &error) };
            Err(error)
        }
    }
}

/// Link and evaluate a compiled module.
///
/// This module-runtime slice drives the module record through
/// `Unlinked -> Linking -> Linked -> Evaluating -> Evaluated`. On success it
/// returns the top-level completion value as a new [`StatorValue`]. On failure
/// it returns null, stores the error on `module`, and publishes a pending
/// exception on `ctx` when a context is supplied.
///
/// For linked module graphs, dependencies are evaluated depth-first and simple
/// named/default import cells are copied from dependency exports before the
/// importing module runs. Namespace objects and TLA remain incomplete.
///
/// # Safety
/// - `module` must be a non-null pointer returned by [`stator_module_compile`].
/// - `ctx` must be a valid, live [`StatorContext`] pointer, or null (in which
///   case an empty global environment is used).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_evaluate(
    module: *mut StatorModule,
    ctx: *mut StatorContext,
) -> *mut StatorValue {
    if module.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `module` is valid.
    let status = unsafe { (*module).status };
    match status {
        StatorModuleStatus::StatorModuleStatusEvaluated => {
            // SAFETY: caller guarantees `module` is valid.
            let module_ref = unsafe { &*module };
            return match &module_ref.last_result {
                Some(result) => unsafe { module_result_to_value(ctx, result) },
                None => unsafe { module_result_to_value(ctx, &JsValue::Undefined) },
            };
        }
        StatorModuleStatus::StatorModuleStatusErrored => return std::ptr::null_mut(),
        StatorModuleStatus::StatorModuleStatusEvaluating
        | StatorModuleStatus::StatorModuleStatusLinking => {
            let error =
                stator_jse::error::StatorError::Internal("module is already active".to_string());
            // SAFETY: caller guarantees `module` is valid.
            let module_ref = unsafe { &mut *module };
            // SAFETY: `module_ref` and `ctx` are valid for this call.
            unsafe { set_module_error(module_ref, ctx, &error) };
            return std::ptr::null_mut();
        }
        StatorModuleStatus::StatorModuleStatusUnlinked
        | StatorModuleStatus::StatorModuleStatusLinked => {}
    }

    if unsafe { context_is_execution_terminating(ctx) } {
        let error =
            stator_jse::error::StatorError::Internal("script execution terminated".to_string());
        // SAFETY: caller guarantees `module` is valid.
        let module_ref = unsafe { &mut *module };
        // SAFETY: `module_ref` and `ctx` are valid for this call.
        unsafe { set_module_error(module_ref, ctx, &error) };
        return std::ptr::null_mut();
    }

    let _interrupt_scope = InterruptFlagScope::new(ctx);
    let owned_global_env;
    let global_env = if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        unsafe { &(*ctx).globals }
    } else {
        owned_global_env = Rc::new(RefCell::new(GlobalEnv::new()));
        &owned_global_env
    };

    let mut active = HashSet::new();
    match unsafe { evaluate_module_graph(ctx, module, global_env, &mut active) } {
        Ok(result) => {
            if !ctx.is_null() {
                // SAFETY: `ctx` is valid and owns a stable isolate pointer.
                unsafe { discard_pending_exception((*ctx)._isolate) };
            }
            // SAFETY: `ctx` is valid when non-null and `result` is live.
            unsafe { module_result_to_value(ctx, &result) }
        }
        Err(error) => {
            // SAFETY: caller guarantees `module` is valid.
            let module_ref = unsafe { &mut *module };
            // SAFETY: `module_ref` and `ctx` are valid for this call.
            unsafe { set_module_error(module_ref, ctx, &error) };
            std::ptr::null_mut()
        }
    }
}

/// Free a [`StatorModule`] previously returned by [`stator_module_compile`].
///
/// # Safety
/// `module` must be a non-null pointer returned by [`stator_module_compile`]
/// and must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_module_free(module: *mut StatorModule) {
    if !module.is_null() {
        // SAFETY: pointer was created by `Box::into_raw` in
        // `stator_module_compile`.
        drop(unsafe { Box::from_raw(module) });
    }
}

// ── Script execution (Phase 3) ────────────────────────────────────────────────

/// C-callable native-function signature.
///
/// The callback receives the active context, an array of `argc` argument
/// pointers (owned by the Rust side; **do not free them**), and the count.
/// It must return either a new [`StatorValue`] (caller must free it) or a
/// null pointer (treated as `undefined` unless the isolate has a pending
/// exception, in which case script execution fails).
type StatorNativeCallback = unsafe extern "C" fn(
    ctx: *mut StatorContext,
    args: *const *const StatorValue,
    argc: i32,
) -> *mut StatorValue;

unsafe fn set_pending_exception(
    isolate: *mut StatorIsolate,
    exception: *mut StatorValue,
    owns_exception: bool,
) {
    if isolate.is_null() {
        return;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    let isolate_ref = unsafe { &mut *isolate };
    if isolate_ref.pending_exception_owned
        && let Some(previous) = isolate_ref.pending_exception.take()
    {
        // SAFETY: owned pending exceptions are allocated by this FFI layer.
        unsafe { stator_value_destroy(previous) };
    }
    isolate_ref.pending_exception = if exception.is_null() {
        None
    } else {
        Some(exception)
    };
    isolate_ref.pending_exception_owned = owns_exception && !exception.is_null();
    // Any new pending exception invalidates a previously-cached structured
    // message.  `record_script_error` re-populates it for engine-raised
    // errors; embedder-thrown values leave it cleared.
    isolate_ref.pending_message = None;
}

unsafe fn discard_pending_exception(isolate: *mut StatorIsolate) {
    if isolate.is_null() {
        return;
    }
    // SAFETY: caller guarantees `isolate` is valid.
    let isolate_ref = unsafe { &mut *isolate };
    if isolate_ref.pending_exception_owned
        && let Some(exception) = isolate_ref.pending_exception.take()
    {
        // SAFETY: owned pending exceptions are allocated by this FFI layer.
        unsafe { stator_value_destroy(exception) };
    } else {
        isolate_ref.pending_exception = None;
    }
    isolate_ref.pending_exception_owned = false;
    isolate_ref.pending_message = None;
}

unsafe fn isolate_has_pending_exception(isolate: *const StatorIsolate) -> bool {
    if isolate.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `isolate` is valid when non-null.
    unsafe { (*isolate).pending_exception.is_some() }
}

unsafe fn context_has_pending_exception(ctx: *mut StatorContext) -> bool {
    if ctx.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let isolate = unsafe { (*ctx)._isolate };
    // SAFETY: `isolate` comes from a valid context.
    unsafe { isolate_has_pending_exception(isolate) }
}

unsafe fn ensure_pending_dom_interceptor_exception(
    isolate: *mut StatorIsolate,
    message: &'static str,
) {
    // SAFETY: caller guarantees `isolate` is valid when non-null.
    if unsafe { isolate_has_pending_exception(isolate) } {
        return;
    }
    // SAFETY: `isolate` is either null or valid; the string is copied.
    let exception = unsafe {
        stator_value_new_string(isolate, message.as_ptr() as *const c_char, message.len())
    };
    // SAFETY: `exception` was allocated by this FFI layer.
    unsafe { set_pending_exception(isolate, exception, true) };
}

unsafe fn pending_exception_error(
    isolate: *const StatorIsolate,
    fallback_message: &str,
) -> Option<stator_jse::error::StatorError> {
    // SAFETY: caller guarantees `isolate` is valid when non-null.
    if !unsafe { isolate_has_pending_exception(isolate) } {
        return None;
    }
    let message = if isolate.is_null() {
        fallback_message.to_string()
    } else {
        // SAFETY: caller guarantees `isolate` is valid when non-null.
        unsafe {
            (*isolate)
                .pending_exception
                .and_then(|exception| {
                    if exception.is_null() {
                        None
                    } else {
                        match &(*exception).inner {
                            StatorValueInner::Str(value) => {
                                Some(value.to_string_lossy().into_owned())
                            }
                            _ => None,
                        }
                    }
                })
                .unwrap_or_else(|| fallback_message.to_string())
        }
    };
    Some(stator_jse::error::StatorError::TypeError(message))
}

unsafe fn pending_dom_interceptor_error(
    isolate: *const StatorIsolate,
) -> Option<stator_jse::error::StatorError> {
    // SAFETY: caller guarantees `isolate` is valid when non-null.
    unsafe { pending_exception_error(isolate, "DOM interceptor exception") }
}

unsafe fn pending_native_callback_error(
    isolate: *const StatorIsolate,
) -> Option<stator_jse::error::StatorError> {
    // SAFETY: caller guarantees `isolate` is valid when non-null.
    unsafe { pending_exception_error(isolate, "Native callback exception") }
}

unsafe fn record_script_error(
    script: *const StatorScript,
    ctx: *mut StatorContext,
    error: &stator_jse::error::StatorError,
) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let isolate = unsafe { (*ctx)._isolate };
    if isolate.is_null() {
        return;
    }
    let message = error.to_string();
    // SAFETY: `isolate` is valid and `message` is copied by the callee.
    let exception = unsafe {
        stator_value_new_string(isolate, message.as_ptr() as *const c_char, message.len())
    };
    // SAFETY: `exception` was allocated for ownership by the pending exception.
    unsafe { set_pending_exception(isolate, exception, true) };

    // Build the structured message *after* `set_pending_exception`, which
    // unconditionally clears any previously-stored message.
    let terminating = !isolate.is_null()
        // SAFETY: `isolate` is valid.
        && unsafe { (*isolate).terminating.load(Ordering::Relaxed) };
    let kind = message_kind_for_error(error, terminating);
    let resource_name = if script.is_null() {
        None
    } else {
        // SAFETY: caller guarantees `script` (when non-null) is valid.
        unsafe { (*script).resource_name.clone() }
    };
    let structured = Box::new(StatorMessage {
        kind,
        text: CString::new(message).unwrap_or_else(|_| c"script error".into()),
        resource_name,
        line: None,
        column: None,
        terminated: terminating || matches!(kind, StatorMessageKind::StatorMessageKindTermination),
    });
    // SAFETY: `isolate` is valid.
    unsafe { (*isolate).pending_message = Some(structured) };
}

unsafe fn context_is_execution_terminating(ctx: *mut StatorContext) -> bool {
    if ctx.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let isolate = unsafe { (*ctx)._isolate };
    !isolate.is_null() && unsafe { (*isolate).terminating.load(Ordering::Relaxed) }
}

/// RAII guard: publish the isolate's termination flag to the interpreter's
/// thread-local interrupt slot for the duration of a script run, and clear
/// it on drop.  See `stator_jse::interpreter::set_interrupt_flag`.
struct InterruptFlagScope {
    prev: *const AtomicBool,
}

impl InterruptFlagScope {
    fn new(ctx: *mut StatorContext) -> Self {
        let prev = stator_jse::interpreter::interrupt_flag_ptr();
        if !ctx.is_null() {
            // SAFETY: caller of run_script_inner / run_script_no_result_inner
            // guarantees `ctx` is valid.
            let isolate = unsafe { (*ctx)._isolate };
            if !isolate.is_null() {
                // SAFETY: `isolate` outlives the script run; the atomic
                // address is stable for the isolate's lifetime.
                let flag_ptr: *const AtomicBool = unsafe { &(*isolate).terminating };
                unsafe { stator_jse::interpreter::set_interrupt_flag(flag_ptr) };
            }
        }
        Self { prev }
    }
}

impl Drop for InterruptFlagScope {
    fn drop(&mut self) {
        // SAFETY: `prev` was the previously published pointer (possibly null);
        // restoring it is always safe.
        unsafe { stator_jse::interpreter::set_interrupt_flag(self.prev) };
    }
}

unsafe fn run_script_inner(
    script: *const StatorScript,
    ctx: *mut StatorContext,
) -> Option<stator_jse::error::StatorResult<JsValue>> {
    if script.is_null() {
        return None;
    }
    if unsafe { context_is_execution_terminating(ctx) } {
        return Some(Err(stator_jse::error::StatorError::Internal(
            "script execution terminated".to_string(),
        )));
    }
    // Publish the isolate's atomic termination flag to the interpreter's
    // thread-local interrupt slot so backward branches, call boundaries, and
    // microtask drains poll it.  Cleared on drop, even on panic.
    let _interrupt_scope = InterruptFlagScope::new(ctx);
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return None,
    };
    if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        let isolate = unsafe { (*ctx)._isolate };
        if !isolate.is_null() {
            // SAFETY: context owns a live isolate pointer by construction.
            if unsafe { (*isolate).jit_disabled } {
                bytecodes.set_jit_disabled(true);
            }
        }
    }

    let owned_global_env;
    let global_env = if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        unsafe { &(*ctx).globals }
    } else {
        owned_global_env = Rc::new(RefCell::new(GlobalEnv::new()));
        &owned_global_env
    };

    Some(if global_env.borrow().globals_installed {
        Interpreter::run_fast(bytecodes, &[], global_env)
    } else {
        let mut frame =
            InterpreterFrame::new_with_globals(Rc::clone(bytecodes), vec![], Rc::clone(global_env));
        Interpreter::run(&mut frame)
    })
}

unsafe fn run_script_no_result_inner(
    script: *const StatorScript,
    ctx: *mut StatorContext,
) -> Option<stator_jse::error::StatorResult<()>> {
    if script.is_null() {
        return None;
    }
    if unsafe { context_is_execution_terminating(ctx) } {
        return Some(Err(stator_jse::error::StatorError::Internal(
            "script execution terminated".to_string(),
        )));
    }
    let _interrupt_scope = InterruptFlagScope::new(ctx);
    // SAFETY: caller guarantees `script` is valid.
    let bytecodes = match unsafe { &(*script).bytecodes } {
        Some(b) => b,
        None => return None,
    };
    if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        let isolate = unsafe { (*ctx)._isolate };
        if !isolate.is_null() {
            // SAFETY: context owns a live isolate pointer by construction.
            if unsafe { (*isolate).jit_disabled } {
                bytecodes.set_jit_disabled(true);
            }
        }
    }

    let owned_global_env;
    let global_env = if !ctx.is_null() {
        // SAFETY: caller guarantees `ctx` is valid.
        unsafe { &(*ctx).globals }
    } else {
        owned_global_env = Rc::new(RefCell::new(GlobalEnv::new()));
        &owned_global_env
    };

    Some(if global_env.borrow().globals_installed {
        Interpreter::run_fast_no_result(bytecodes, &[], global_env)
    } else {
        let mut frame =
            InterpreterFrame::new_with_globals(Rc::clone(bytecodes), vec![], Rc::clone(global_env));
        Interpreter::run(&mut frame).map(|_| ())
    })
}

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
    let result = match unsafe { run_script_inner(script, ctx) } {
        Some(result) => result,
        None => return std::ptr::null_mut(),
    };

    match result {
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
                unsafe { discard_pending_exception(isolate) };
                // SAFETY: isolate is valid.
                unsafe { (*isolate).live_objects += 1 };
            }
            Box::into_raw(Box::new(StatorValue { inner, isolate }))
        }
        Err(error) => {
            // SAFETY: `ctx` is valid for the duration of script execution.
            if unsafe { context_has_pending_exception(ctx) } {
                return std::ptr::null_mut();
            }
            // SAFETY: `ctx` and `script` are valid for the duration of script execution.
            unsafe { record_script_error(script, ctx, &error) };
            std::ptr::null_mut()
        }
    }
}

/// Execute a compiled script in `ctx` and discard the result.
///
/// Returns `true` when execution completes without an exception.  This avoids
/// allocating an embedder-owned [`StatorValue`] for callers that only need to
/// drive script execution and do not inspect the result.
///
/// # Safety
/// - `script` must be a non-null pointer returned by [`stator_script_compile`].
/// - `ctx` must be a valid, live [`StatorContext`] pointer, or null (in which
///   case an empty global environment is used).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_run_no_result(
    script: *const StatorScript,
    ctx: *mut StatorContext,
) -> bool {
    match unsafe { run_script_no_result_inner(script, ctx) } {
        Some(Ok(_)) => {
            if !ctx.is_null() {
                // SAFETY: `ctx` is valid and owns a stable isolate pointer.
                unsafe { discard_pending_exception((*ctx)._isolate) };
            }
            true
        }
        Some(Err(error)) => {
            // SAFETY: `ctx` is valid for the duration of script execution.
            if unsafe { context_has_pending_exception(ctx) } {
                return false;
            }
            // SAFETY: `ctx` and `script` are valid for the duration of script execution.
            unsafe { record_script_error(script, ctx, &error) };
            false
        }
        None => false,
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
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. }
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
        // Two shared-storage object handles compare equal iff they point at
        // the same underlying `Rc<RefCell<JsObject>>` (pointer identity).
        (StatorValueInner::ObjectHandle(x), StatorValueInner::ObjectHandle(y)) => Rc::ptr_eq(x, y),
        (
            StatorValueInner::DomWrapHandle { plain: x, .. },
            StatorValueInner::DomWrapHandle { plain: y, .. },
        ) => Rc::ptr_eq(x, y),
        // Object-like tags carry no identity in FFI handles → never equal.
        _ => false,
    }
}

// ── Status enum, typed accessors, and identity bridge ────────────────────────

/// Stable status code returned by typed-accessor and Maybe-style FFI entry
/// points so that C callers can distinguish a missing value, an invalid
/// argument, an unsupported operation, and a JavaScript exception without
/// reaching for out-of-band channels.
///
/// Variants:
/// * [`StatorStatusOk`][Self::StatorStatusOk] — the call succeeded; any
///   out-pointer was written.
/// * [`StatorStatusFalse`][Self::StatorStatusFalse] — the call succeeded with
///   a structural "no" answer (e.g. property missing, type mismatch on a
///   typed get).  Out-pointers are left untouched or zero-cleared, never
///   carrying a stale value.
/// * [`StatorStatusException`][Self::StatorStatusException] — the operation
///   raised (or captured) a JavaScript exception.  When applicable, the
///   isolate's pending exception is populated and can be inspected through
///   [`stator_isolate_peek_pending_message`] / `stator_try_catch_*`.
/// * [`StatorStatusInvalidArg`][Self::StatorStatusInvalidArg] — at least one
///   required argument was null or otherwise malformed; nothing was mutated.
/// * [`StatorStatusUnsupported`][Self::StatorStatusUnsupported] — the
///   operation is well-formed but not yet implemented for this kind of
///   value/object (e.g. calling a bytecode function via the FFI).  Reserved
///   discriminants are stable; embedders should treat unknown values as
///   `StatorStatusUnsupported`.
///
/// Backed by a C `int`-shaped enum so the discriminants are stable across
/// the C ABI and can be compared by value from C/C++.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StatorStatus {
    /// Operation succeeded.
    StatorStatusOk = 0,
    /// Operation succeeded with a structural "no" answer (missing property,
    /// type mismatch on a typed accessor).
    StatorStatusFalse = 1,
    /// A JavaScript exception was raised or captured.  The isolate's pending
    /// exception is set when this is returned by a Maybe-style API.
    StatorStatusException = 2,
    /// At least one argument was null or malformed.
    StatorStatusInvalidArg = 3,
    /// The operation is not supported for this value/object kind.
    StatorStatusUnsupported = 4,
}

/// Read the boolean value of `val` into `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when `val` holds a JavaScript boolean.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but not a
///   boolean.  `*out` is left untouched.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `out` is null.
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `out` must be either null or a valid pointer to a `bool`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_get_boolean(
    val: *const StatorValue,
    out: *mut bool,
) -> StatorStatus {
    if val.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Boolean(b) => {
            // SAFETY: caller guarantees `out` is valid.
            unsafe { *out = *b };
            StatorStatus::StatorStatusOk
        }
        _ => StatorStatus::StatorStatusFalse,
    }
}

/// Read the numeric value of `val` into `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when `val` holds a JavaScript number.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but not a
///   number.  `*out` is left untouched.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `out` is null.
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `out` must be either null or a valid pointer to an `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_get_number(
    val: *const StatorValue,
    out: *mut f64,
) -> StatorStatus {
    if val.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Number(n) => {
            // SAFETY: caller guarantees `out` is valid.
            unsafe { *out = *n };
            StatorStatus::StatorStatusOk
        }
        _ => StatorStatus::StatorStatusFalse,
    }
}

/// Read `val` as a signed 32-bit integer into `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when `val` is a finite integer in the
///   range `[−2³¹, 2³¹−1]`; `*out` is set to the exact value.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but is not
///   exactly representable as an `i32` (non-number, fractional, ±Infinity,
///   `NaN`, or out of range).  `*out` is left untouched.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `out` is null.
///
/// Mirrors the semantics of `v8::Value::Int32Value` for the success case but
/// without the implicit `ToNumber` coercion — embedders that want the lossy
/// truncating conversion should call [`stator_value_to_int32`] instead.
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `out` must be either null or a valid pointer to an `i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_get_int32(
    val: *const StatorValue,
    out: *mut i32,
) -> StatorStatus {
    if val.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    if let StatorValueInner::Number(n) = unsafe { &(*val).inner }
        && n.is_finite()
        && n.fract() == 0.0
        && *n >= i32::MIN as f64
        && *n <= i32::MAX as f64
    {
        // SAFETY: caller guarantees `out` is valid.
        unsafe { *out = *n as i32 };
        return StatorStatus::StatorStatusOk;
    }
    StatorStatus::StatorStatusFalse
}

/// Read `val` as an unsigned 32-bit integer into `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when `val` is a finite integer in the
///   range `[0, 2³²−1]`; `*out` is set to the exact value.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but is not
///   exactly representable as a `u32`.  `*out` is left untouched.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `out` is null.
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `out` must be either null or a valid pointer to a `u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_get_uint32(
    val: *const StatorValue,
    out: *mut u32,
) -> StatorStatus {
    if val.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    if let StatorValueInner::Number(n) = unsafe { &(*val).inner }
        && n.is_finite()
        && n.fract() == 0.0
        && *n >= 0.0
        && *n <= u32::MAX as f64
    {
        // SAFETY: caller guarantees `out` is valid.
        unsafe { *out = *n as u32 };
        return StatorStatus::StatorStatusOk;
    }
    StatorStatus::StatorStatusFalse
}

/// Read the UTF-8 byte length of the string stored in `val` into `*out`.
///
/// The returned length does **not** include a terminating null byte and is
/// the count of raw UTF-8 bytes — sufficient for sizing a buffer passed to
/// [`stator_value_write_string_utf8`].
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when `val` is a string; `*out` is the
///   byte length.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but not a
///   string.  `*out` is left untouched.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `out` is null.
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `out` must be either null or a valid pointer to a `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_get_string_utf8_length(
    val: *const StatorValue,
    out: *mut usize,
) -> StatorStatus {
    if val.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::Str(cs) => {
            // SAFETY: caller guarantees `out` is valid.
            unsafe { *out = cs.as_bytes().len() };
            StatorStatus::StatorStatusOk
        }
        _ => StatorStatus::StatorStatusFalse,
    }
}

/// Copy the UTF-8 representation of the string stored in `val` into `buf`.
///
/// The write APIs work in explicit byte counts: no implicit `strlen` is
/// performed on the source side and no terminating null byte is appended to
/// the destination.  This makes the API safe to use with input data that
/// contains, or buffers that should accommodate, embedded NUL bytes.
///
/// At most `buf_size` bytes are written.  If `buf_size` is smaller than the
/// string's byte length the output is truncated; callers can size their
/// buffer with [`stator_value_get_string_utf8_length`] and inspect
/// `*written` to detect truncation.
///
/// On success, `*written` (if non-null) is set to the number of bytes
/// actually written.  On any non-OK return `*written` is set to zero so
/// callers never see stale length information.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on a (possibly truncated) successful
///   write.
/// * [`StatorStatus::StatorStatusFalse`] when `val` is non-null but not a
///   string.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `val` or `buf` is null.
///   A null `written` pointer is allowed (and simply discards the length).
///
/// # Safety
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
/// * `buf` must be valid for writes of `buf_size` bytes (unless null).
/// * `written` must be either null or a valid pointer to a `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_write_string_utf8(
    val: *const StatorValue,
    buf: *mut c_char,
    buf_size: usize,
    written: *mut usize,
) -> StatorStatus {
    if !written.is_null() {
        // SAFETY: caller guarantees `written` is valid when non-null.
        unsafe { *written = 0 };
    }
    if val.is_null() || buf.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `val` is valid.
    let bytes = match unsafe { &(*val).inner } {
        StatorValueInner::Str(cs) => cs.as_bytes(),
        _ => return StatorStatus::StatorStatusFalse,
    };
    let n = bytes.len().min(buf_size);
    if n > 0 {
        // SAFETY: `buf` is valid for `buf_size` bytes; `n <= buf_size`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n) };
    }
    if !written.is_null() {
        // SAFETY: caller guarantees `written` is valid when non-null.
        unsafe { *written = n };
    }
    StatorStatus::StatorStatusOk
}

// ── Object / value identity bridge ───────────────────────────────────────────

/// Borrow the shared backing storage of an [`StatorObject`] handle so that
/// the same `Rc<RefCell<JsObject>>` can be reused by both [`StatorObject`]
/// and [`StatorValue`] handles created from it.  Returns `None` if the
/// object's pointer is null.
///
/// # Safety
/// `obj` must be either null or a valid, live [`StatorObject`] pointer.
unsafe fn stator_object_inner_clone(obj: *const StatorObject) -> Option<Rc<RefCell<JsObject>>> {
    if obj.is_null() {
        return None;
    }
    // SAFETY: caller guarantees `obj` is valid.
    Some(Rc::clone(&unsafe { &*obj }.inner))
}

/// Wrap a [`StatorObject`] handle as a fresh [`StatorValue`] handle that
/// shares the same underlying `JsObject` storage.
///
/// The returned value's `typeof` is `"object"`; passing it to
/// [`stator_value_as_object`] yields a new [`StatorObject`] handle whose
/// property mutations are observed through `obj` and vice versa.  This is
/// the FFI mechanism for round-tripping object identity across the
/// value/object boundary.
///
/// Returns a null pointer when `obj` is null.  The caller owns the returned
/// value pointer and must release it with [`stator_value_destroy`] (or rely
/// on the active handle scope).
///
/// # Safety
/// `obj` must be either null or a valid, live [`StatorObject`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_as_value(obj: *const StatorObject) -> *mut StatorValue {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `obj` is valid.
    let rc = match unsafe { stator_object_inner_clone(obj) } {
        Some(rc) => rc,
        None => return std::ptr::null_mut(),
    };
    // SAFETY: caller guarantees `obj` is valid.
    let isolate = unsafe { (*obj).isolate };
    if !isolate.is_null() {
        // SAFETY: `isolate` is valid for the lifetime of `obj`.
        unsafe { (*isolate).live_objects += 1 };
    }
    let val = Box::into_raw(Box::new(StatorValue {
        inner: StatorValueInner::ObjectHandle(rc),
        isolate,
    }));
    if !isolate.is_null() {
        // SAFETY: `isolate` is valid; `active_handle_scope` is null or valid.
        unsafe {
            let scope = (*isolate).active_handle_scope;
            if !scope.is_null() {
                (*scope).handles.push(val);
            }
        }
    }
    val
}

/// Wrap a [`StatorValue`] holding an object-as-value handle as a fresh
/// [`StatorObject`] handle that shares the same underlying `JsObject`
/// storage.
///
/// Identity is only preserved for values produced by
/// [`stator_object_as_value`] (or by future FFI APIs that allocate the
/// shared-storage representation).  Tag-only object values created via
/// [`stator_value_new_object`], `stator_value_new_array_tag`, and the other
/// tag constructors carry no per-instance storage; passing such a value
/// here returns a null pointer to make the limitation explicit at the call
/// site rather than silently materialising a divergent empty object.
///
/// Returns a null pointer when `val` is null, is not an object-like value,
/// or is a tag-only object value.  The caller owns the returned object
/// pointer and must release it with [`stator_object_destroy`].
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_as_object(val: *const StatorValue) -> *mut StatorObject {
    if val.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `val` is valid.
    let rc = match unsafe { &(*val).inner } {
        StatorValueInner::ObjectHandle(rc) => Rc::clone(rc),
        _ => return std::ptr::null_mut(),
    };
    // SAFETY: caller guarantees `val` is valid.
    let isolate = unsafe { (*val).isolate };
    if !isolate.is_null() {
        // SAFETY: `isolate` is valid for the lifetime of `val`.
        unsafe { (*isolate).live_objects += 1 };
    }
    Box::into_raw(Box::new(StatorObject { inner: rc, isolate }))
}

// ── Maybe-style property APIs ────────────────────────────────────────────────

/// Decode a `(ptr, len)` UTF-8 key buffer as a borrowed `&str`.
///
/// Returns `None` when the pointer is null or the bytes are not valid UTF-8.
/// Embedded NUL bytes inside the key are preserved (they participate in the
/// property name verbatim).
///
/// # Safety
/// `key` must be valid for reads of `key_len` bytes, or null.
unsafe fn decode_property_key<'a>(key: *const c_char, key_len: usize) -> Option<&'a str> {
    if key.is_null() {
        return None;
    }
    // SAFETY: caller guarantees `key` is valid for `key_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(key as *const u8, key_len) };
    std::str::from_utf8(bytes).ok()
}

/// Build a fresh owned [`StatorValue`] from a [`JsValue`] returned by an
/// underlying [`JsObject`] property read.
///
/// The new value is registered with the isolate's live-object counter and
/// (if open) the innermost handle scope.  Returns a null pointer when
/// `isolate` is null or allocation fails.
fn js_value_to_owned_stator_value(
    isolate: *mut StatorIsolate,
    js_val: &JsValue,
) -> *mut StatorValue {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let inner = jsvalue_to_stator_value_inner(js_val);
    // SAFETY: caller guarantees `isolate` is valid; bookkeeping only.
    unsafe { (*isolate).live_objects += 1 };
    let val = Box::into_raw(Box::new(StatorValue { inner, isolate }));
    // SAFETY: `isolate` is valid; `active_handle_scope` is null or valid.
    unsafe {
        let scope = (*isolate).active_handle_scope;
        if !scope.is_null() {
            (*scope).handles.push(val);
        }
    }
    val
}

/// Read the property named `(key, key_len)` from `obj` and, on success, write
/// a freshly-allocated [`StatorValue`] handle to `*out_val`.
///
/// Maybe-style semantics:
/// * [`StatorStatus::StatorStatusOk`] when the property exists.  The
///   returned value mirrors the property value, including an explicit
///   `undefined` if that is what is stored.
/// * [`StatorStatus::StatorStatusFalse`] when the property is missing from
///   `obj` and its prototype chain.  `*out_val` is set to null.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `obj`, `key`, or
///   `out_val` is null, when the key bytes are not valid UTF-8, or when the
///   object has no associated isolate.
///
/// The caller owns the [`StatorValue`] handed back through `*out_val` and
/// must release it with [`stator_value_destroy`] (or rely on the active
/// handle scope).
///
/// # Safety
/// * `obj` must be either null or a valid, live [`StatorObject`] pointer.
/// * `key` must be valid for reads of `key_len` bytes (or null).
/// * `out_val` must be either null or a valid pointer to a `*mut StatorValue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_get_property(
    obj: *const StatorObject,
    key: *const c_char,
    key_len: usize,
    out_val: *mut *mut StatorValue,
) -> StatorStatus {
    if !out_val.is_null() {
        // SAFETY: caller guarantees `out_val` is valid when non-null.
        unsafe { *out_val = std::ptr::null_mut() };
    }
    if obj.is_null() || out_val.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `key` is valid for `key_len` bytes.
    let key_str = match unsafe { decode_property_key(key, key_len) } {
        Some(s) => s,
        None => return StatorStatus::StatorStatusInvalidArg,
    };
    // SAFETY: caller guarantees `obj` is valid.
    let inner_rc = Rc::clone(&unsafe { &*obj }.inner);
    // SAFETY: caller guarantees `obj` is valid.
    let isolate = unsafe { (*obj).isolate };
    if isolate.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    let (present, js_val) = {
        let borrowed = inner_rc.borrow();
        let present = borrowed.has_property(key_str);
        if present {
            (true, borrowed.get_property(key_str))
        } else {
            (false, JsValue::Undefined)
        }
    };
    if !present {
        return StatorStatus::StatorStatusFalse;
    }
    let val = js_value_to_owned_stator_value(isolate, &js_val);
    if val.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `out_val` is valid.
    unsafe { *out_val = val };
    StatorStatus::StatorStatusOk
}

/// Write `val` to the property named `(key, key_len)` on `obj`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when the property was created or
///   updated.
/// * [`StatorStatus::StatorStatusException`] when the underlying `[[Set]]`
///   raised an error (e.g. assigning to a read-only property).  The
///   isolate's pending exception is populated with a stringified error and
///   a structured [`StatorMessage`] classified as
///   [`StatorMessageKind::StatorMessageKindType`].
/// * [`StatorStatus::StatorStatusInvalidArg`] when `obj`, `key`, or `val` is
///   null, when the key bytes are not valid UTF-8, or when the object has
///   no associated isolate.
///
/// # Safety
/// * `obj` must be either null or a valid, live [`StatorObject`] pointer.
/// * `key` must be valid for reads of `key_len` bytes (or null).
/// * `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_set_property(
    obj: *mut StatorObject,
    key: *const c_char,
    key_len: usize,
    val: *const StatorValue,
) -> StatorStatus {
    if obj.is_null() || val.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `key` is valid for `key_len` bytes.
    let key_str = match unsafe { decode_property_key(key, key_len) } {
        Some(s) => s,
        None => return StatorStatus::StatorStatusInvalidArg,
    };
    // SAFETY: caller guarantees `obj` is valid.
    let inner_rc = Rc::clone(&unsafe { &*obj }.inner);
    // SAFETY: caller guarantees `val` is valid.
    let js_val = stator_value_inner_to_jsvalue(unsafe { &(*val).inner });
    let result = inner_rc.borrow_mut().set_property(key_str, js_val);
    match result {
        Ok(()) => StatorStatus::StatorStatusOk,
        Err(err) => {
            // SAFETY: caller guarantees `obj` is valid.
            let isolate = unsafe { (*obj).isolate };
            if !isolate.is_null() {
                let message = err.to_string();
                // SAFETY: `isolate` is valid; the string is copied by the callee.
                let exception = unsafe {
                    stator_value_new_string(
                        isolate,
                        message.as_ptr() as *const c_char,
                        message.len(),
                    )
                };
                // SAFETY: `exception` was allocated for ownership.
                unsafe { set_pending_exception(isolate, exception, true) };
                // Build a structured TypeError message so embedders can route
                // it through their existing exception pipeline.
                let structured = Box::new(StatorMessage {
                    kind: StatorMessageKind::StatorMessageKindType,
                    text: CString::new(message).unwrap_or_else(|_| c"property set error".into()),
                    resource_name: None,
                    line: None,
                    column: None,
                    terminated: false,
                });
                // SAFETY: `isolate` is valid.
                unsafe { (*isolate).pending_message = Some(structured) };
            }
            StatorStatus::StatorStatusException
        }
    }
}

/// Test whether `obj` has a property named `(key, key_len)` (own or
/// inherited).  The result is written to `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on success; `*out` is `true` iff the
///   property exists.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `obj`, `key`, or `out` is
///   null, or when the key bytes are not valid UTF-8.
///
/// # Safety
/// * `obj` must be either null or a valid, live [`StatorObject`] pointer.
/// * `key` must be valid for reads of `key_len` bytes (or null).
/// * `out` must be either null or a valid pointer to a `bool`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_has_property(
    obj: *const StatorObject,
    key: *const c_char,
    key_len: usize,
    out: *mut bool,
) -> StatorStatus {
    if obj.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `key` is valid for `key_len` bytes.
    let key_str = match unsafe { decode_property_key(key, key_len) } {
        Some(s) => s,
        None => return StatorStatus::StatorStatusInvalidArg,
    };
    // SAFETY: caller guarantees `obj` is valid.
    let present = unsafe { &*obj }.inner.borrow().has_property(key_str);
    // SAFETY: caller guarantees `out` is valid.
    unsafe { *out = present };
    StatorStatus::StatorStatusOk
}

/// Delete the property named `(key, key_len)` from `obj`.  The boolean
/// outcome of the underlying `[[Delete]]` is written to `*out`.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on success.  `*out` is `true` when
///   the property no longer exists (either it was successfully deleted or
///   was already absent) and `false` when deletion was rejected (e.g. a
///   non-configurable own property).
/// * [`StatorStatus::StatorStatusInvalidArg`] when `obj`, `key`, or `out` is
///   null, or when the key bytes are not valid UTF-8.
///
/// # Safety
/// * `obj` must be either null or a valid, live [`StatorObject`] pointer.
/// * `key` must be valid for reads of `key_len` bytes (or null).
/// * `out` must be either null or a valid pointer to a `bool`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_object_delete_property(
    obj: *mut StatorObject,
    key: *const c_char,
    key_len: usize,
    out: *mut bool,
) -> StatorStatus {
    if obj.is_null() || out.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `key` is valid for `key_len` bytes.
    let key_str = match unsafe { decode_property_key(key, key_len) } {
        Some(s) => s,
        None => return StatorStatus::StatorStatusInvalidArg,
    };
    // SAFETY: caller guarantees `obj` is valid.
    let deleted = unsafe { &*obj }
        .inner
        .borrow_mut()
        .delete_own_property(key_str)
        .unwrap_or(false);
    // SAFETY: caller guarantees `out` is valid.
    unsafe { *out = deleted };
    StatorStatus::StatorStatusOk
}

// ── Native-function call bridge ──────────────────────────────────────────────

/// Invoke a callable [`StatorValue`] with `argc` arguments and, on success,
/// write the returned value to `*out_val`.
///
/// This first slice only supports native functions installed via
/// [`stator_function_template_get_function`] or returned through the FFI by
/// a native callback — i.e. values whose internal representation is a
/// reference-counted [`NativeFn`].  Bytecode-backed function values
/// ([`StatorValueInner::Function`]) cannot yet be invoked through this API
/// because the FFI does not yet model `new.target` or bytecode-function call
/// frames for direct embedder calls; calling such a value returns
/// [`StatorStatus::StatorStatusUnsupported`].  Construct semantics are
/// likewise deferred.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when the native function ran to
///   completion; `*out_val` holds the (caller-owned) result.
/// * [`StatorStatus::StatorStatusException`] when the native callback
///   returned an `Err`.  The isolate's pending exception is populated.
/// * [`StatorStatus::StatorStatusUnsupported`] when `callable` is not a
///   native function (e.g. bytecode function, plain object, primitive).
/// * [`StatorStatus::StatorStatusInvalidArg`] when `ctx`, `callable`, or
///   `out_val` is null, when `argc` is negative, or when `args` is null
///   while `argc > 0`.
///
/// `recv` is used as the native callback receiver (`this`).  A null receiver is
/// treated as JavaScript `undefined`.
///
/// # Safety
/// * `ctx` must be a valid, live [`StatorContext`] pointer.
/// * `callable` must be either null or a valid, live [`StatorValue`] pointer.
/// * `recv` must be either null or a valid, live [`StatorValue`] pointer.
/// * `args` must be valid for `argc` `*const StatorValue` reads when `argc > 0`
///   (or null when `argc == 0`).  Each non-null element must point at a
///   valid, live [`StatorValue`].
/// * `out_val` must be either null or a valid pointer to a `*mut StatorValue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_call(
    ctx: *mut StatorContext,
    callable: *const StatorValue,
    recv: *const StatorValue,
    argc: i32,
    args: *const *const StatorValue,
    out_val: *mut *mut StatorValue,
) -> StatorStatus {
    if !out_val.is_null() {
        // SAFETY: caller guarantees `out_val` is valid when non-null.
        unsafe { *out_val = std::ptr::null_mut() };
    }
    if ctx.is_null() || callable.is_null() || out_val.is_null() || argc < 0 {
        return StatorStatus::StatorStatusInvalidArg;
    }
    if argc > 0 && args.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let isolate = unsafe { (*ctx)._isolate };
    if isolate.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `callable` is valid.
    let native = match unsafe { &(*callable).inner } {
        StatorValueInner::NativeFunctionValue(f) => Rc::clone(f),
        _ => return StatorStatus::StatorStatusUnsupported,
    };
    // Build a JsValue argv from the C-side argv array.
    let mut js_args: Vec<JsValue> = Vec::with_capacity(argc as usize);
    if argc > 0 {
        // SAFETY: caller guarantees `args` is valid for `argc` pointers.
        let slice = unsafe { std::slice::from_raw_parts(args, argc as usize) };
        for &p in slice {
            if p.is_null() {
                js_args.push(JsValue::Undefined);
            } else {
                // SAFETY: each non-null pointer is a valid StatorValue.
                js_args.push(stator_value_inner_to_jsvalue(unsafe { &(*p).inner }));
            }
        }
    }
    let this_val = if recv.is_null() {
        JsValue::Undefined
    } else {
        // SAFETY: caller guarantees `recv` is valid when non-null.
        stator_value_inner_to_jsvalue(unsafe { &(*recv).inner })
    };
    // SAFETY: caller guarantees `ctx` is valid.
    let global_env = unsafe { Rc::clone(&(*ctx).globals) };
    match stator_jse::interpreter::invoke_native_with_this_in_global_env(
        &native, global_env, this_val, js_args,
    ) {
        Ok(js_val) => {
            let val = js_value_to_owned_stator_value(isolate, &js_val);
            if val.is_null() {
                return StatorStatus::StatorStatusInvalidArg;
            }
            // SAFETY: caller guarantees `out_val` is valid.
            unsafe { *out_val = val };
            StatorStatus::StatorStatusOk
        }
        Err(err) => {
            let message = err.to_string();
            // SAFETY: `isolate` is valid.
            let exception = unsafe {
                stator_value_new_string(isolate, message.as_ptr() as *const c_char, message.len())
            };
            // SAFETY: `exception` was allocated for ownership.
            unsafe { set_pending_exception(isolate, exception, true) };
            let kind = message_kind_for_error(&err, false);
            let structured = Box::new(StatorMessage {
                kind,
                text: CString::new(message).unwrap_or_else(|_| c"native call error".into()),
                resource_name: None,
                line: None,
                column: None,
                terminated: false,
            });
            // SAFETY: `isolate` is valid.
            unsafe { (*isolate).pending_message = Some(structured) };
            StatorStatus::StatorStatusException
        }
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

        let isolate = if ctx_ptr.is_null() {
            std::ptr::null_mut()
        } else {
            // SAFETY: `ctx_ptr` is valid for the lifetime of the registered callback.
            unsafe { (*ctx_ptr)._isolate }
        };
        // SAFETY: `isolate` is the context isolate when present.
        if let Some(error) = unsafe { pending_native_callback_error(isolate) } {
            if !ret.is_null() {
                // SAFETY: `ret` was allocated by Box::into_raw in the callback.
                unsafe { stator_value_destroy(ret) };
            }
            Err(error)
        } else if ret.is_null() {
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
/// provides access to the call receiver, arguments, and isolate.  It must
/// return either a new [`StatorValue`] (the caller — i.e. the engine wrapper —
/// owns it and frees it automatically) or a null pointer (treated as
/// `undefined` unless the isolate has a pending exception, in which case script
/// execution fails).
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
    /// Temporary receiver value valid for the duration of the call.
    this_value: StatorValue,
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

/// Return the receiver (`this`) associated with this call.
///
/// The returned pointer is valid only for the duration of the callback
/// invocation.  Returns a null pointer when `info` is null.
///
/// # Safety
/// `info` must be either null or a valid pointer to a live
/// [`StatorFunctionCallbackInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_function_callback_info_get_this(
    info: *const StatorFunctionCallbackInfo,
) -> *const StatorValue {
    if info.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `info` is valid.
    unsafe { &(*info).this_value as *const StatorValue }
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
            this_value: StatorValue {
                inner: stator_jse::interpreter::current_this()
                    .as_ref()
                    .map(jsvalue_to_stator_value_inner)
                    .unwrap_or(StatorValueInner::Undefined),
                isolate: std::ptr::null_mut(),
            },
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

        // SAFETY: `isolate` is the callback isolate and remains valid for the
        // lifetime of the function value.
        if let Some(error) = unsafe { pending_native_callback_error(isolate) } {
            if !ret.is_null() {
                // SAFETY: `ret` was allocated by `Box::into_raw` in the callback.
                unsafe { stator_value_destroy(ret) };
            }
            Err(error)
        } else if ret.is_null() {
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

fn dom_object_wrap_plain_object(
    wrap: *mut StatorDomObjectWrap,
) -> Option<Rc<RefCell<PropertyMap>>> {
    if wrap.is_null() {
        return None;
    }

    // SAFETY: caller guarantees `wrap` is valid.
    if let Some(existing) = unsafe { (*wrap).materialized.borrow().as_ref().cloned() } {
        return Some(existing);
    }

    let wrap_addr = wrap as usize;
    // SAFETY: caller guarantees `wrap` is valid.
    let alive = unsafe { Rc::clone(&(*wrap).alive) };
    // SAFETY: caller guarantees `wrap` is valid for the lifetime of the
    // installed JS global.  We copy the enumerable property names now and route
    // each accessor back through the live wrapper so named handlers stay active.
    let names = unsafe { (*wrap).inner.property_names() };
    let mut object = PropertyMap::new();
    let mut installed_names: Vec<String> = Vec::new();

    for name in names {
        if name.starts_with("__get_")
            || name.starts_with("__set_")
            || installed_names.iter().any(|installed| installed == &name)
        {
            continue;
        }
        installed_names.push(name.clone());

        let getter_name = name.clone();
        let getter_wrap_addr = wrap_addr;
        let getter_alive = Rc::clone(&alive);
        object.insert_builtin(
            format!("__get_{name}__"),
            JsValue::NativeFunction(Rc::new(move |_args| {
                if !getter_alive.get() {
                    return Ok(JsValue::Undefined);
                }
                let wrap = getter_wrap_addr as *mut StatorDomObjectWrap;
                if wrap.is_null() {
                    return Ok(JsValue::Undefined);
                }
                // SAFETY: the embedder must keep the wrapper alive for as long
                // as the installed global can be observed.
                let value = unsafe { (*wrap).inner.get_property(&getter_name) };
                // SAFETY: the wrapper owns the isolate pointer for this DOM object.
                if let Some(error) = unsafe { pending_dom_interceptor_error((*wrap).isolate) } {
                    return Err(error);
                }
                Ok(value)
            })),
        );

        if unsafe { (*wrap).has_named_setter } {
            let setter_name = name.clone();
            let setter_wrap_addr = wrap_addr;
            let setter_alive = Rc::clone(&alive);
            object.insert_builtin(
                format!("__set_{name}__"),
                JsValue::NativeFunction(Rc::new(move |args| {
                    if !setter_alive.get() {
                        return Ok(JsValue::Undefined);
                    }
                    let wrap = setter_wrap_addr as *mut StatorDomObjectWrap;
                    if wrap.is_null() {
                        return Ok(JsValue::Undefined);
                    }
                    let value = args.first().cloned().unwrap_or(JsValue::Undefined);
                    // SAFETY: the embedder must keep the wrapper alive until
                    // the owning context is destroyed.
                    unsafe {
                        (*wrap).inner.set_intercepted_property(&setter_name, value);
                        if let Some(error) = pending_dom_interceptor_error((*wrap).isolate) {
                            return Err(error);
                        }
                    }
                    Ok(JsValue::Undefined)
                })),
            );
        }

        object.insert_with_attrs(name, JsValue::Undefined, PropertyAttributes::ENUMERABLE);
    }

    object.extensible = false;
    let plain = Rc::new(RefCell::new(object));
    DOM_WRAP_MATERIALIZED_REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .insert(Rc::as_ptr(&plain) as usize, (wrap, Rc::clone(&alive)));
    });
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { *(*wrap).materialized.borrow_mut() = Some(Rc::clone(&plain)) };
    Some(plain)
}

fn dom_object_wrap_js_value(wrap: *mut StatorDomObjectWrap) -> Option<JsValue> {
    dom_object_wrap_plain_object(wrap).map(JsValue::PlainObject)
}

/// Wrap a DOM object wrapper as an identity-preserving [`StatorValue`].
///
/// Multiple calls for the same wrapper return values that materialize to the
/// same JS object, so script-visible comparisons such as
/// `document.documentElement === document.documentElement` work without a V8
/// wrapper.  The wrapper must remain valid until all returned values and script
/// aliases have been released or [`stator_dom_object_wrap_invalidate`] has been
/// called.
///
/// Returns null when `wrap` is null or its owning isolate is null.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_as_value(
    wrap: *mut StatorDomObjectWrap,
) -> *mut StatorValue {
    if wrap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `wrap` is valid.
    let isolate = unsafe { (*wrap).isolate };
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    let Some(plain) = dom_object_wrap_plain_object(wrap) else {
        return std::ptr::null_mut();
    };
    // SAFETY: `isolate` is the wrapper's owning isolate and is live while the
    // wrapper is live.
    unsafe {
        allocate_stator_value(
            isolate,
            StatorValueInner::DomWrapHandle {
                plain,
                wrap,
                alive: Rc::clone(&(*wrap).alive),
            },
        )
    }
}

/// Return the originating DOM wrapper for a value produced by
/// [`stator_dom_object_wrap_as_value`].
///
/// Returns null for primitives, ordinary objects, tag-only object values, and
/// DOM wrapper values that have since been invalidated.
///
/// # Safety
/// `val` must be either null or a valid, live [`StatorValue`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_value_as_dom_object_wrap(
    val: *const StatorValue,
) -> *mut StatorDomObjectWrap {
    if val.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `val` is valid.
    match unsafe { &(*val).inner } {
        StatorValueInner::DomWrapHandle { wrap, alive, .. } if alive.get() => *wrap,
        _ => std::ptr::null_mut(),
    }
}

/// Invalidate a DOM object wrapper's script-visible materialization.
///
/// Existing JS aliases to the materialized object remain safe to touch, but
/// generated DOM accessor thunks stop dereferencing the raw wrapper pointer and
/// return `undefined`.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_invalidate(wrap: *mut StatorDomObjectWrap) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe {
        (*wrap).alive.set(false);
        if let Some(plain) = (*wrap).materialized.borrow_mut().take() {
            DOM_WRAP_MATERIALIZED_REGISTRY.with(|registry| {
                registry.borrow_mut().remove(&(Rc::as_ptr(&plain) as usize));
            });
        }
    }
}

/// Install a DOM object wrapper as a named global on `ctx`.
///
/// The installed value is backed by a Stator `PlainObject` whose enumerable
/// properties are snapshotted from the wrapper's named-property enumerator and
/// own-property list at install time.  Reads of those properties dispatch back
/// to the wrapper's named getter; writes dispatch back only when the wrapper has
/// a named setter installed.  This lets embedders expose P0 DOM globals such as
/// `document.title` without V8.
///
/// The global path shares the same cached materialization as
/// [`stator_dom_object_wrap_as_value`], preserving identity with wrapper values
/// returned from getters and methods.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] when the global was installed.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `ctx`, `name`, or `wrap` is
///   null, when `name` is not valid UTF-8, or when `wrap` belongs to a
///   different isolate than `ctx`.
///
/// # Safety
/// - `ctx` must be a valid, live [`StatorContext`] pointer.
/// - `name` must be a valid, null-terminated C string.
/// - `wrap` must be a valid, live [`StatorDomObjectWrap`] pointer. It must
///   remain alive until no script aliases can observe it, or must be invalidated
///   via [`stator_dom_object_wrap_invalidate`] before destruction.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_context_global_set_dom_object_wrap(
    ctx: *mut StatorContext,
    name: *const c_char,
    wrap: *mut StatorDomObjectWrap,
) -> StatorStatus {
    if ctx.is_null() || name.is_null() || wrap.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }

    // SAFETY: caller guarantees pointers are valid.
    let ctx_isolate = unsafe { (*ctx)._isolate };
    let wrap_isolate = unsafe { (*wrap).isolate };
    if ctx_isolate.is_null() || ctx_isolate != wrap_isolate {
        return StatorStatus::StatorStatusInvalidArg;
    }

    // SAFETY: caller guarantees `name` is a valid C string.
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return StatorStatus::StatorStatusInvalidArg,
    };

    let Some(js_value) = dom_object_wrap_js_value(wrap) else {
        return StatorStatus::StatorStatusInvalidArg;
    };

    // SAFETY: caller guarantees `ctx` is valid.
    let globals = unsafe { Rc::clone(&(*ctx).globals) };
    globals.borrow_mut().insert(name_str, js_value);
    StatorStatus::StatorStatusOk
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
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. }
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
        StatorValueInner::Object
        | StatorValueInner::ObjectHandle(_)
        | StatorValueInner::DomWrapHandle { .. } => "[object Object]".to_owned(),
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
        StatorValueInner::ObjectHandle(_) => {
            // Identity-preserving bridge from `StatorValueInner::ObjectHandle`
            // into the interpreter heap is not yet implemented.  Round-trips
            // through this conversion (e.g. when an object handle is passed
            // through a script-visible global) lose identity: a fresh empty
            // `PlainObject` is materialised here.  FFI-level property reads
            // and writes via `stator_object_*` APIs still observe shared
            // state because they operate directly on the `Rc<RefCell<JsObject>>`.
            JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())))
        }
        StatorValueInner::DomWrapHandle { plain, .. } => JsValue::PlainObject(Rc::clone(plain)),
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
        JsValue::PlainObject(plain) => {
            dom_wrap_inner_for_plain_object(plain).unwrap_or(StatorValueInner::Object)
        }
        JsValue::Object(_)
        | JsValue::Generator(_)
        | JsValue::Iterator(_)
        | JsValue::Error(_)
        | JsValue::Promise(_) => StatorValueInner::Object,
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
        StatorValueInner::ObjectHandle(rc) => StatorValueInner::ObjectHandle(Rc::clone(rc)),
        StatorValueInner::DomWrapHandle { plain, wrap, alive } => StatorValueInner::DomWrapHandle {
            plain: Rc::clone(plain),
            wrap: *wrap,
            alive: Rc::clone(alive),
        },
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
        inner: Rc::new(RefCell::new(obj)),
        isolate,
    }))
}

// ── Structured messages ──────────────────────────────────────────────────────

/// Stable classification of a Stator engine error or message.
///
/// Mirrors the structural categories embedders care about — JavaScript
/// built-in error kinds, internal engine errors, termination, and unknown.
/// Backed by a C `int`-shaped enum so the discriminants are stable across
/// the C ABI and can be compared by value from C/C++.
///
/// New variants may be appended in future versions; embedders should treat
/// unknown values as [`StatorMessageKind::StatorMessageKindUnknown`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorMessageKind {
    /// No classification available.
    StatorMessageKindUnknown = 0,
    /// JavaScript `SyntaxError`, or a parse/compile error from the engine
    /// front-end.
    StatorMessageKindSyntax = 1,
    /// JavaScript `TypeError`.
    StatorMessageKindType = 2,
    /// JavaScript `RangeError`.
    StatorMessageKindRange = 3,
    /// JavaScript `ReferenceError`.
    StatorMessageKindReference = 4,
    /// JavaScript `URIError`.
    StatorMessageKindURI = 5,
    /// WebAssembly compilation, instantiation, or execution failure.
    StatorMessageKindWasm = 6,
    /// An internal engine error (`stator_jse::error::StatorError::Internal`)
    /// or a programmer-induced FFI misuse (e.g. null source pointer).
    StatorMessageKindInternal = 7,
    /// Script execution was interrupted by an explicit termination request
    /// (`stator_isolate_terminate_execution` or equivalent).
    StatorMessageKindTermination = 8,
    /// An uncaught JavaScript exception value that does not map to one of
    /// the structured kinds above.
    StatorMessageKindJsException = 9,
    /// The engine ran out of heap memory.
    StatorMessageKindOutOfMemory = 10,
    /// A sandbox bounds-check violation.
    StatorMessageKindSandboxViolation = 11,
}

/// An opaque structured message describing an engine error.
///
/// Carries the information embedders need to surface a Stator failure
/// without parsing the human-readable text:
///
/// - **kind** — a [`StatorMessageKind`] classifying the error.
/// - **text** — the engine's UTF-8 message string.
/// - **resource_name** — the script's resource URL when the message was
///   produced during script execution, otherwise `None`.
/// - **line / column** — script position when available.  Currently not
///   populated by the engine; getters return `false` so callers can detect
///   the missing data rather than treat a zero as truthful.
/// - **terminated** — `true` when the message was raised because execution
///   was being terminated (see [`StatorMessageKind::StatorMessageKindTermination`]).
///
/// Created internally by the engine and exposed via:
/// - [`stator_isolate_take_pending_message`] (ownership transfer)
/// - [`stator_isolate_peek_pending_message`] (borrowed view)
/// - [`stator_try_catch_message`] (borrowed view, valid for the lifetime of
///   the try-catch scope)
///
/// Ownership-transferred messages must be freed with
/// [`stator_message_destroy`].
pub struct StatorMessage {
    kind: StatorMessageKind,
    text: CString,
    resource_name: Option<CString>,
    line: Option<i32>,
    column: Option<i32>,
    terminated: bool,
}

// SAFETY: `StatorMessage` contains only owned heap data; it is `Send` on the
// same single-threaded model as `StatorIsolate`.
unsafe impl Send for StatorMessage {}

/// Classify a [`stator_jse::error::StatorError`] into a stable
/// [`StatorMessageKind`].
///
/// `terminating` is the isolate-wide termination flag observed at the point
/// of error capture; when set it upgrades a generic `Internal("script
/// execution terminated")` (or any error raised while the flag is live) to
/// [`StatorMessageKind::StatorMessageKindTermination`].
fn message_kind_for_error(
    error: &stator_jse::error::StatorError,
    terminating: bool,
) -> StatorMessageKind {
    use stator_jse::error::StatorError as E;
    if terminating {
        return StatorMessageKind::StatorMessageKindTermination;
    }
    match error {
        E::OutOfMemory => StatorMessageKind::StatorMessageKindOutOfMemory,
        E::TypeError(_) => StatorMessageKind::StatorMessageKindType,
        E::SyntaxError(_) => StatorMessageKind::StatorMessageKindSyntax,
        E::ReferenceError(_) => StatorMessageKind::StatorMessageKindReference,
        E::RangeError(_) => StatorMessageKind::StatorMessageKindRange,
        E::URIError(_) => StatorMessageKind::StatorMessageKindURI,
        E::WasmError(_) => StatorMessageKind::StatorMessageKindWasm,
        E::JsException(_) => StatorMessageKind::StatorMessageKindJsException,
        E::SandboxViolation { .. } => StatorMessageKind::StatorMessageKindSandboxViolation,
        E::Internal(msg) => {
            // The FFI layer raises this synthetic error when termination is
            // detected before script entry; classify it as Termination even
            // when the atomic flag has since been cleared.
            if msg == "script execution terminated" {
                StatorMessageKind::StatorMessageKindTermination
            } else {
                StatorMessageKind::StatorMessageKindInternal
            }
        }
        E::DebuggerPaused { .. } => StatorMessageKind::StatorMessageKindInternal,
    }
}

/// Return the structured [`StatorMessageKind`] of `msg`.
///
/// Returns [`StatorMessageKind::StatorMessageKindUnknown`] when `msg` is null.
///
/// # Safety
/// `msg` must be either null or a valid pointer to a live [`StatorMessage`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_kind(msg: *const StatorMessage) -> StatorMessageKind {
    if msg.is_null() {
        return StatorMessageKind::StatorMessageKindUnknown;
    }
    // SAFETY: caller guarantees `msg` is valid.
    unsafe { (*msg).kind }
}

/// Return a null-terminated UTF-8 description of `msg`, or null when `msg`
/// is null.
///
/// The returned pointer is borrowed: it is valid as long as `msg` is alive
/// and must not be freed by the caller.
///
/// # Safety
/// `msg` must be either null or a valid pointer to a live [`StatorMessage`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_text(msg: *const StatorMessage) -> *const c_char {
    if msg.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `msg` is valid.
    unsafe { (*msg).text.as_ptr() }
}

/// Return the script resource name associated with `msg`, or null when no
/// resource name was attached (e.g. for embedder-thrown exceptions) or
/// `msg` is null.
///
/// The returned pointer is borrowed: it is valid as long as `msg` is alive
/// and must not be freed by the caller.
///
/// # Safety
/// `msg` must be either null or a valid pointer to a live [`StatorMessage`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_resource_name(msg: *const StatorMessage) -> *const c_char {
    if msg.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `msg` is valid.
    match unsafe { &(*msg).resource_name } {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// If `msg` carries a 1-based line number, write it into `*out_line` and
/// return `true`; otherwise leave `*out_line` untouched and return `false`.
///
/// Returns `false` when `msg` or `out_line` is null.  Callers must treat a
/// `false` return as "line is unknown" rather than assuming zero.
///
/// # Safety
/// - `msg` must be either null or a valid pointer to a live [`StatorMessage`].
/// - `out_line`, when non-null, must be aligned and valid for a write of an
///   `i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_get_line(
    msg: *const StatorMessage,
    out_line: *mut i32,
) -> bool {
    if msg.is_null() || out_line.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `msg` is valid.
    match unsafe { (*msg).line } {
        Some(line) => {
            // SAFETY: caller guarantees `out_line` is valid for one i32.
            unsafe { *out_line = line };
            true
        }
        None => false,
    }
}

/// If `msg` carries a 1-based column number, write it into `*out_column`
/// and return `true`; otherwise leave `*out_column` untouched and return
/// `false`.
///
/// Returns `false` when `msg` or `out_column` is null.  Callers must treat
/// a `false` return as "column is unknown" rather than assuming zero.
///
/// # Safety
/// - `msg` must be either null or a valid pointer to a live [`StatorMessage`].
/// - `out_column`, when non-null, must be aligned and valid for a write of
///   an `i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_get_column(
    msg: *const StatorMessage,
    out_column: *mut i32,
) -> bool {
    if msg.is_null() || out_column.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `msg` is valid.
    match unsafe { (*msg).column } {
        Some(column) => {
            // SAFETY: caller guarantees `out_column` is valid for one i32.
            unsafe { *out_column = column };
            true
        }
        None => false,
    }
}

/// Return `true` if `msg` was produced because execution was being
/// terminated.  Returns `false` when `msg` is null.
///
/// # Safety
/// `msg` must be either null or a valid pointer to a live [`StatorMessage`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_terminated(msg: *const StatorMessage) -> bool {
    if msg.is_null() {
        return false;
    }
    // SAFETY: caller guarantees `msg` is valid.
    unsafe { (*msg).terminated }
}

/// Destroy a [`StatorMessage`] previously transferred to the embedder by
/// [`stator_isolate_take_pending_message`].
///
/// Does nothing when `msg` is null.  Borrowed pointers returned by
/// `_peek_` / `_try_catch_message` must NOT be passed to this function;
/// they remain owned by the isolate / try-catch scope.
///
/// # Safety
/// `msg` must be either null or a pointer obtained from
/// [`stator_isolate_take_pending_message`] and not previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_message_destroy(msg: *mut StatorMessage) {
    if !msg.is_null() {
        // SAFETY: caller guarantees `msg` came from `Box::into_raw`.
        drop(unsafe { Box::from_raw(msg) });
    }
}

/// Borrow the structured pending message on `isolate`, or return null when
/// none is set.
///
/// The returned pointer is owned by the isolate and remains valid only
/// until the next call that mutates the pending exception slot
/// (`stator_isolate_throw_exception`, `stator_isolate_clear_pending_exception`,
/// `stator_isolate_take_pending_message`, script execution, or a
/// try-catch scope capturing the exception).
///
/// # Safety
/// `isolate` must be either null or a valid pointer to a live
/// [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_peek_pending_message(
    isolate: *const StatorIsolate,
) -> *const StatorMessage {
    if isolate.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    match unsafe { &(*isolate).pending_message } {
        Some(msg) => msg.as_ref() as *const StatorMessage,
        None => std::ptr::null(),
    }
}

/// Take ownership of the structured pending message on `isolate`.
///
/// Returns null when `isolate` is null or no structured message is
/// available.  The caller owns the returned pointer and must eventually
/// pass it to [`stator_message_destroy`].
///
/// This does NOT clear the pending exception value; callers that want to
/// fully consume the exception should also call
/// [`stator_isolate_clear_pending_exception`] (and `stator_value_destroy`
/// on the result).
///
/// # Safety
/// `isolate` must be either null or a valid pointer to a live
/// [`StatorIsolate`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_isolate_take_pending_message(
    isolate: *mut StatorIsolate,
) -> *mut StatorMessage {
    if isolate.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    match unsafe { (*isolate).pending_message.take() } {
        Some(msg) => Box::into_raw(msg),
        None => std::ptr::null_mut(),
    }
}

/// Return the structured [`StatorMessageKind`] of `script`'s compile error,
/// or [`StatorMessageKind::StatorMessageKindUnknown`] when `script` compiled successfully or
/// is null.
///
/// Parse and bytecode-generator failures are classified as
/// [`StatorMessageKind::StatorMessageKindSyntax`]; FFI misuse such as a null source pointer
/// is classified as [`StatorMessageKind::StatorMessageKindInternal`].
///
/// # Safety
/// `script` must be either null or a valid pointer to a live
/// [`StatorScript`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_script_error_kind(
    script: *const StatorScript,
) -> StatorMessageKind {
    if script.is_null() {
        return StatorMessageKind::StatorMessageKindUnknown;
    }
    // SAFETY: caller guarantees `script` is valid.
    unsafe { (*script).error_kind }
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
    owns_exception: bool,
    /// Whether an exception has been caught by this scope.
    has_caught: bool,
    /// Structured classification of the captured exception, taken from the
    /// isolate's `pending_message` slot when the scope first observes a
    /// pending exception.  `None` when no structured information is
    /// available (embedder-thrown raw values).
    caught_message: Option<Box<StatorMessage>>,
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
        owns_exception: false,
        has_caught: false,
        caught_message: None,
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
            tc_ref.owns_exception = unsafe { (*tc_ref.isolate).pending_exception_owned };
            unsafe { (*tc_ref.isolate).pending_exception_owned = false };
            // SAFETY: isolate is valid; transfer the structured message
            // alongside the exception value.
            tc_ref.caught_message = unsafe { (*tc_ref.isolate).pending_message.take() };
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

/// Return a borrowed structured message describing the caught exception,
/// or null when none was caught or no structured information is available
/// (e.g. an embedder-thrown raw exception value).
///
/// The returned pointer is owned by the try-catch scope and remains valid
/// until [`stator_try_catch_reset`] or [`stator_try_catch_destroy`] is
/// called.  The caller must NOT pass it to [`stator_message_destroy`].
///
/// # Safety
/// `tc` must be either null or a valid, live [`StatorTryCatch`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_message(
    tc: *const StatorTryCatch,
) -> *const StatorMessage {
    if tc.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `tc` is valid.
    match unsafe { &(*tc).caught_message } {
        Some(msg) => msg.as_ref() as *const StatorMessage,
        None => std::ptr::null(),
    }
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
            if (*tc).owns_exception && !(*tc).exception.is_null() {
                stator_value_destroy((*tc).exception);
            }
            (*tc).exception = std::ptr::null_mut();
            (*tc).owns_exception = false;
            (*tc).has_caught = false;
            (*tc).caught_message = None;
        }
    }
}

/// Destroy a try-catch scope previously created with [`stator_try_catch_new`].
///
/// If the scope holds a caught exception created internally by script
/// execution, it is destroyed with the scope.  Embedder-owned exception values
/// passed to [`stator_isolate_throw_exception`] are not destroyed.
///
/// Does nothing if `tc` is null.
///
/// # Safety
/// `tc` must be a non-null pointer returned by [`stator_try_catch_new`] and
/// must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_try_catch_destroy(tc: *mut StatorTryCatch) {
    if !tc.is_null() {
        // SAFETY: `tc` is valid and may own its current exception.
        unsafe { stator_try_catch_reset(tc) };
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
/// [`stator_jse::platform::Platform`] trait.
struct VTablePlatformImpl {
    vtable: StatorPlatformVTable,
}

// SAFETY: The vtable function pointers are inherently thread-safe (they are
// stateless C function pointers, not closures); the embedder is responsible
// for ensuring any global state they access is thread-safe.
unsafe impl Send for VTablePlatformImpl {}

impl stator_jse::platform::Platform for VTablePlatformImpl {
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
    inner: Box<dyn stator_jse::platform::Platform>,
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

/// C-ABI Wasm value kind used by host-import marshalling.
///
/// Discriminants are stable and embedders may rely on them.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StatorWasmValueKind {
    /// 32-bit signed integer (`i32`).
    StatorWasmValueKindI32 = 0,
    /// 64-bit signed integer (`i64`).
    StatorWasmValueKindI64 = 1,
    /// 32-bit IEEE-754 float (`f32`).
    StatorWasmValueKindF32 = 2,
    /// 64-bit IEEE-754 float (`f64`).
    StatorWasmValueKindF64 = 3,
}

/// C-ABI union payload of a [`StatorWasmValue`].
///
/// Only the variant indicated by the containing [`StatorWasmValue::kind`] is
/// well-defined; reading any other variant is undefined behavior.
#[repr(C)]
#[derive(Copy, Clone)]
pub union StatorWasmValuePayload {
    /// `i32` storage.
    pub i32_: i32,
    /// `i64` storage.
    pub i64_: i64,
    /// `f32` storage.
    pub f32_: f32,
    /// `f64` storage.
    pub f64_: f64,
}

/// A typed Wasm value used at the host-import boundary.
///
/// `kind` discriminates the active union member.  Embedders must initialise
/// the payload member matching both `kind` and the result kind declared on the
/// host import; Stator traps the Wasm call if a callback returns a mismatched
/// result kind.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StatorWasmValue {
    /// Discriminant for the active member of `value`.
    pub kind: StatorWasmValueKind,
    /// Typed payload.  The active member is selected by `kind`.
    pub value: StatorWasmValuePayload,
}

/// Synchronous host-function callback used to fulfil a Wasm import.
///
/// Invoked synchronously on the thread running the Wasm caller.  Returns
/// `true` after writing exactly `results_len` typed results into `results`.
/// Result kinds must match the declaration in [`StatorWasmHostFunc::results`].
/// Returning `false` or a mismatched result kind traps the running Wasm call:
/// the enclosing
/// [`stator_wasm_instance_call`] then returns null.
///
/// # Safety
/// - `args` is a valid pointer for reads of `args_len` consecutive
///   [`StatorWasmValue`]s.
/// - `results` is a valid pointer for writes of `results_len` consecutive
///   [`StatorWasmValue`]s.  Each slot must be initialised by the callback
///   with the kind declared in [`StatorWasmHostFunc::results`].
/// - `ctx` and `user_data` are the values supplied at instantiate time and
///   must remain live for the duration of the call.
pub type StatorWasmHostFuncCallback = unsafe extern "C" fn(
    ctx: *mut StatorContext,
    user_data: *mut c_void,
    args: *const StatorWasmValue,
    args_len: usize,
    results: *mut StatorWasmValue,
    results_len: usize,
) -> bool;

/// Declaration of a single Wasm host-function import.
///
/// `module_name` and `field_name` form the fully-qualified Wasm import name
/// the module expects (e.g. `env` / `add`).  `params` / `results` arrays
/// declare the signature, must point to valid storage for `params_len` /
/// `results_len` elements respectively, and must match the signature the
/// module declares for that import.  A null `callback` causes instantiation
/// to fail.
#[repr(C)]
pub struct StatorWasmHostFunc {
    /// Null-terminated UTF-8 import module name (e.g. `"env"`).
    pub module_name: *const c_char,
    /// Null-terminated UTF-8 import field name (e.g. `"add"`).
    pub field_name: *const c_char,
    /// Pointer to `params_len` parameter kinds, in order.  May be null when
    /// `params_len` is zero.
    pub params: *const StatorWasmValueKind,
    /// Number of parameter kinds in `params`.
    pub params_len: usize,
    /// Pointer to `results_len` result kinds, in order.  May be null when
    /// `results_len` is zero.
    pub results: *const StatorWasmValueKind,
    /// Number of result kinds in `results`.
    pub results_len: usize,
    /// Callback invoked on each Wasm-side call.  Required; null fails
    /// instantiation.
    pub callback: Option<
        unsafe extern "C" fn(
            ctx: *mut StatorContext,
            user_data: *mut c_void,
            args: *const StatorWasmValue,
            args_len: usize,
            results: *mut StatorWasmValue,
            results_len: usize,
        ) -> bool,
    >,
    /// Opaque embedder data passed unchanged to `callback`.  Stator does not
    /// interpret this pointer.
    pub user_data: *mut c_void,
}

/// A collection of host-function imports passed to [`stator_wasm_instantiate`].
///
/// A null pointer or a structure with `funcs_len == 0` declares "no imports".
#[repr(C)]
pub struct StatorWasmImports {
    /// Pointer to `funcs_len` [`StatorWasmHostFunc`] elements.  May be null
    /// when `funcs_len` is zero.
    pub funcs: *const StatorWasmHostFunc,
    /// Number of elements in `funcs`.
    pub funcs_len: usize,
}

/// Instantiate a compiled [`StatorWasmModule`] into a live
/// [`StatorWasmInstance`].
///
/// `ctx` is captured and forwarded to every host-import callback; it may be
/// null when no callback needs an isolate context.  `imports` may be null (or
/// a struct with `funcs_len == 0`) to instantiate with no host imports; in
/// that case the module must declare no imports of its own or instantiation
/// fails.
///
/// Returns a null pointer if `module` is null, if any required import is
/// missing or malformed (null callback, null name, invalid UTF-8, signature
/// mismatch), or if instantiation fails.
///
/// The returned pointer must eventually be passed to
/// [`stator_wasm_instance_destroy`].
///
/// # Safety
/// - `module` must be a non-null, valid pointer to a live
///   [`StatorWasmModule`].
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer
///   that outlives every host callback invocation triggered through the
///   returned instance.
/// - `imports`, when non-null, must point to a valid [`StatorWasmImports`]
///   whose inner arrays and strings remain valid for the duration of this
///   call.  The structures are only read during instantiation; afterwards
///   the embedder may reuse or free the storage.  However, every
///   `user_data` pointer and the `ctx` pointer captured here must remain
///   valid until [`stator_wasm_instance_destroy`] is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_wasm_instantiate(
    module: *mut StatorWasmModule,
    ctx: *mut StatorContext,
    imports: *const StatorWasmImports,
) -> *mut StatorWasmInstance {
    if module.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `module` is valid.
    let m = unsafe { &*module };

    let host_imports = if imports.is_null() {
        Vec::new()
    } else {
        // SAFETY: caller guarantees `imports` is valid for one read.
        match unsafe { build_host_imports(ctx, imports) } {
            Some(v) => v,
            None => return std::ptr::null_mut(),
        }
    };

    match WasmInstance::new_with_imports(&m.engine, &m.module, host_imports) {
        Ok(instance) => Box::into_raw(Box::new(StatorWasmInstance { instance })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Marshal a C-ABI [`StatorWasmImports`] into a [`Vec<HostFunc>`] suitable for
/// [`WasmInstance::new_with_imports`].
///
/// Returns `None` if any entry is malformed: null callback, null name, name
/// not valid UTF-8, or kind arrays pointing nowhere with non-zero length.
///
/// # Safety
/// `imports` must be a valid pointer to a [`StatorWasmImports`]; its inner
/// arrays / strings must be valid for the duration of this call.
unsafe fn build_host_imports(
    ctx: *mut StatorContext,
    imports: *const StatorWasmImports,
) -> Option<Vec<HostFunc>> {
    // SAFETY: caller guarantees `imports` is a valid pointer.
    let imports = unsafe { &*imports };
    if imports.funcs_len == 0 {
        return Some(Vec::new());
    }
    if imports.funcs.is_null() {
        return None;
    }
    // SAFETY: caller guarantees the slice is valid for `funcs_len` reads.
    let funcs = unsafe { std::slice::from_raw_parts(imports.funcs, imports.funcs_len) };

    let mut out: Vec<HostFunc> = Vec::with_capacity(funcs.len());
    for f in funcs {
        let cb_fn = f.callback?;
        if f.module_name.is_null() || f.field_name.is_null() {
            return None;
        }
        if (f.params.is_null() && f.params_len != 0) || (f.results.is_null() && f.results_len != 0)
        {
            return None;
        }
        // SAFETY: caller guarantees the strings are valid null-terminated C
        // strings for the duration of this call.
        let module_name = match unsafe { CStr::from_ptr(f.module_name) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return None,
        };
        // SAFETY: as above.
        let field_name = match unsafe { CStr::from_ptr(f.field_name) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return None,
        };
        let params: Vec<HostValKind> = if f.params_len == 0 {
            Vec::new()
        } else {
            // SAFETY: validated non-null above.
            unsafe { std::slice::from_raw_parts(f.params, f.params_len) }
                .iter()
                .map(|k| stator_kind_to_host(*k))
                .collect()
        };
        let results: Vec<HostValKind> = if f.results_len == 0 {
            Vec::new()
        } else {
            // SAFETY: validated non-null above.
            unsafe { std::slice::from_raw_parts(f.results, f.results_len) }
                .iter()
                .map(|k| stator_kind_to_host(*k))
                .collect()
        };

        // Wasmtime's `Linker::func_new` requires the closure to be
        // `Send + Sync + 'static`.  Raw pointers are neither, even though
        // Stator's Wasm runtime only invokes host callbacks synchronously on
        // the calling thread.  Smuggle them through `usize` to satisfy the
        // bound; the embedder is responsible for keeping `ctx` and
        // `user_data` valid until the instance is destroyed.
        let ctx_addr = ctx as usize;
        let user_addr = f.user_data as usize;
        let result_kinds = results.clone();
        let callback: HostFuncCallback = Arc::new(move |args, out_results| {
            let c_args: Vec<StatorWasmValue> = args.iter().map(host_val_to_stator).collect();
            let mut c_results: Vec<StatorWasmValue> = result_kinds
                .iter()
                .map(|k| StatorWasmValue {
                    kind: host_kind_to_stator(*k),
                    value: StatorWasmValuePayload { i64_: 0 },
                })
                .collect();
            // SAFETY: `cb_fn` is the embedder-provided extern "C" callback
            // captured at instantiate time; `args` / `results` slices are
            // valid for `len` reads/writes; `ctx` / `user_data` are the
            // values supplied at instantiate time and the embedder is
            // responsible for keeping them live until the instance is
            // destroyed.
            let ok = unsafe {
                cb_fn(
                    ctx_addr as *mut StatorContext,
                    user_addr as *mut c_void,
                    c_args.as_ptr(),
                    c_args.len(),
                    c_results.as_mut_ptr(),
                    c_results.len(),
                )
            };
            if !ok {
                return false;
            }
            if out_results.len() != c_results.len() {
                return false;
            }
            for (expected, c) in result_kinds.iter().zip(c_results.iter()) {
                if c.kind != host_kind_to_stator(*expected) {
                    return false;
                }
            }
            for (slot, c) in out_results.iter_mut().zip(c_results.iter()) {
                *slot = stator_to_host_val(c);
            }
            true
        });

        out.push(HostFunc {
            module: module_name,
            name: field_name,
            params,
            results,
            callback,
        });
    }
    Some(out)
}

fn stator_kind_to_host(k: StatorWasmValueKind) -> HostValKind {
    match k {
        StatorWasmValueKind::StatorWasmValueKindI32 => HostValKind::I32,
        StatorWasmValueKind::StatorWasmValueKindI64 => HostValKind::I64,
        StatorWasmValueKind::StatorWasmValueKindF32 => HostValKind::F32,
        StatorWasmValueKind::StatorWasmValueKindF64 => HostValKind::F64,
    }
}

fn host_kind_to_stator(k: HostValKind) -> StatorWasmValueKind {
    match k {
        HostValKind::I32 => StatorWasmValueKind::StatorWasmValueKindI32,
        HostValKind::I64 => StatorWasmValueKind::StatorWasmValueKindI64,
        HostValKind::F32 => StatorWasmValueKind::StatorWasmValueKindF32,
        HostValKind::F64 => StatorWasmValueKind::StatorWasmValueKindF64,
    }
}

fn host_val_to_stator(v: &HostVal) -> StatorWasmValue {
    match *v {
        HostVal::I32(n) => StatorWasmValue {
            kind: StatorWasmValueKind::StatorWasmValueKindI32,
            value: StatorWasmValuePayload { i32_: n },
        },
        HostVal::I64(n) => StatorWasmValue {
            kind: StatorWasmValueKind::StatorWasmValueKindI64,
            value: StatorWasmValuePayload { i64_: n },
        },
        HostVal::F32(f) => StatorWasmValue {
            kind: StatorWasmValueKind::StatorWasmValueKindF32,
            value: StatorWasmValuePayload { f32_: f },
        },
        HostVal::F64(f) => StatorWasmValue {
            kind: StatorWasmValueKind::StatorWasmValueKindF64,
            value: StatorWasmValuePayload { f64_: f },
        },
    }
}

fn stator_to_host_val(v: &StatorWasmValue) -> HostVal {
    // SAFETY: the embedder must initialise the union member that matches
    // `kind`; reading any other member would be UB.  We document this
    // contract on [`StatorWasmValue`].
    unsafe {
        match v.kind {
            StatorWasmValueKind::StatorWasmValueKindI32 => HostVal::I32(v.value.i32_),
            StatorWasmValueKind::StatorWasmValueKindI64 => HostVal::I64(v.value.i64_),
            StatorWasmValueKind::StatorWasmValueKindF32 => HostVal::F32(v.value.f32_),
            StatorWasmValueKind::StatorWasmValueKindF64 => HostVal::F64(v.value.f64_),
        }
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

use stator_jse::inspector::cdp::CdpServer;
use stator_jse::inspector::debugger::{DebugAction, Debugger};
use stator_jse::interpreter::{attach_debugger, detach_debugger};
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
    let frame = InterpreterFrame::new_with_globals(bytecodes, vec![], Rc::clone(&ctx_ref.globals));
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
        Err(stator_jse::error::StatorError::DebuggerPaused { .. }) => {
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
///
/// In addition to the engine-level wrapper state, the FFI layer tracks two
/// browser-embedder concepts: a *class id* (an embedder-defined integer that
/// identifies the JS-visible interface, e.g. `HTMLDivElement`) and a *native
/// object pointer* (an opaque pointer the embedder uses to identify the
/// underlying C++ object regardless of which internal field is in use).
/// Both default to zero / null and are never dereferenced by the engine.
pub struct StatorDomObjectWrap {
    inner: DomObjectWrap,
    isolate: *mut StatorIsolate,
    /// Whether the current named handler has a setter.  The FFI global-install
    /// helper uses this to avoid manufacturing write access for read-only DOM
    /// properties.
    has_named_setter: bool,
    /// Embedder-assigned class identifier (0 = unassigned).
    class_id: u32,
    /// Opaque embedder-assigned native object identity pointer.  Distinct
    /// from internal fields so embedders can tag the canonical native object
    /// without consuming an internal-field slot.
    native_ptr: *mut c_void,
    /// Cached JS-visible wrapper object.  Reused by globals, named getters,
    /// and method returns so repeated conversions preserve object identity.
    materialized: RefCell<Option<Rc<RefCell<PropertyMap>>>>,
    /// Liveness flag captured by generated accessor closures.  The flag lets
    /// stale JS aliases observe `undefined` instead of dereferencing a wrapper
    /// after the embedder invalidates or destroys it.
    alive: Rc<Cell<bool>>,
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
    if isolate.is_null() || field_count as usize > stator_jse::dom::MAX_INTERNAL_FIELDS {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `isolate` is valid.
    unsafe { (*isolate).live_objects += 1 };
    Box::into_raw(Box::new(StatorDomObjectWrap {
        inner: DomObjectWrap::new(field_count as usize),
        isolate,
        has_named_setter: false,
        class_id: 0,
        native_ptr: std::ptr::null_mut(),
        materialized: RefCell::new(None),
        alive: Rc::new(Cell::new(true)),
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
        // SAFETY: caller guarantees `wrap` is valid and uniquely owned.
        unsafe { stator_dom_object_wrap_invalidate(wrap) };
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
    unsafe { (*wrap).has_named_setter = false };
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
    unsafe { (*wrap).has_named_setter = true };
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

// ── DOM hardened slice: class identity + V2 interceptors ──────────────────────
//
// The functions below extend the basic DOM wrapper FFI with concepts required
// by a real browser embedding (Edge/Blink-style):
//
// * **Class identity** — `class_id` plus an opaque `native_ptr` that lets the
//   embedder recognise a JS object as wrapping a known native C++ type without
//   having to walk internal fields.
// * **Aggregate interceptor install** — `StatorDomNamedHandler` and
//   `StatorDomIndexedHandler` are POD structs of optional function pointers
//   (any field may be `NULL`) that install all callbacks at once.  This is
//   additive on top of the legacy `set_named_getter` / `set_named_setter` /
//   `set_indexed_getter` / `set_indexed_setter` entry points and lets
//   embedders register query, deleter, enumerator, and length callbacks too.
// * **StatorStatus + explicit-length UTF-8 keys** — V2 callbacks accept
//   `(name, name_len)` rather than a null-terminated C string and return
//   [`StatorStatus`] so the embedder can distinguish *handled* from *missing*
//   from *exception*.
//
// ### Exception bridging limitation
//
// A V2 callback returning [`StatorStatus::StatorStatusException`] records a
// pending isolate exception when the embedder has not already done so.  The
// wrapper-backed global accessor thunks then convert that pending exception
// into a script execution failure so embedders can observe it through
// `stator_try_catch_*`.

/// Set the embedder-defined class identifier on `wrap`.
///
/// A class id of 0 is treated as *unassigned*.  Does nothing when `wrap` is
/// null.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_class_id(
    wrap: *mut StatorDomObjectWrap,
    class_id: u32,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).class_id = class_id };
}

/// Return the embedder-defined class identifier previously stored on `wrap`,
/// or `0` when `wrap` is null or unassigned.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_get_class_id(
    wrap: *const StatorDomObjectWrap,
) -> u32 {
    if wrap.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).class_id }
}

/// Store an opaque native-object identity pointer on `wrap`.
///
/// The engine never dereferences this pointer; it is purely an identity tag
/// the embedder can compare against a known native object (e.g. a
/// `blink::Element*`).  Does nothing when `wrap` is null.
///
/// # Safety
/// - `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
/// - `ptr` is treated as opaque by the engine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_set_native_ptr(
    wrap: *mut StatorDomObjectWrap,
    ptr: *mut c_void,
) {
    if wrap.is_null() {
        return;
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).native_ptr = ptr };
}

/// Retrieve the opaque native-object identity pointer previously stored on
/// `wrap`, or null when `wrap` is null or no pointer has been set.
///
/// # Safety
/// `wrap` must be either null or a valid, live [`StatorDomObjectWrap`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_get_native_ptr(
    wrap: *const StatorDomObjectWrap,
) -> *mut c_void {
    if wrap.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).native_ptr }
}

// ── V2 callback signatures (explicit-length keys + StatorStatus) ──────────

/// V2 named-property **getter** callback.
///
/// The property name is passed as a UTF-8 byte range
/// `(name_utf8, name_len)` — it is **not** required to be null-terminated.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] — the interceptor handled the read and
///   wrote a non-null [`StatorValue`] pointer into `*out`.  Ownership of the
///   value transfers to Stator, which will destroy it once consumed.
/// * [`StatorStatus::StatorStatusFalse`] — the interceptor did not handle the
///   read; the engine falls through to the wrapper's own properties.  `*out`
///   is ignored.
/// * [`StatorStatus::StatorStatusException`] — the embedder raised an
///   exception.  If the callback did not populate the isolate's pending
///   exception via [`stator_isolate_throw_exception`], the FFI bridge records a
///   generic DOM-interceptor exception before the wrapper-backed global
///   accessor thunk fails script execution.
/// * Any other status is treated as fall-through.
///
/// # Safety
/// - `name_utf8` must be valid for `name_len` bytes.
/// - `out` must be a writable `*mut StatorValue` slot if [`StatorStatus::StatorStatusOk`]
///   is returned.
pub type StatorDomNamedGetterCbV2 = unsafe extern "C" fn(
    name_utf8: *const c_char,
    name_len: usize,
    data: *mut c_void,
    out: *mut *mut StatorValue,
) -> StatorStatus;

/// V2 named-property **setter** callback.
///
/// Returns [`StatorStatus::StatorStatusOk`] to indicate the write was handled
/// (engine will not fall through to the wrapper's own properties),
/// [`StatorStatus::StatorStatusFalse`] for fall-through, or
/// [`StatorStatus::StatorStatusException`] to fail wrapper-backed global
/// access with a pending exception.
///
/// # Safety
/// `name_utf8` must be valid for `name_len` bytes; `value` must be either
/// null or a valid, live [`StatorValue`] pointer borrowed for the call.
pub type StatorDomNamedSetterCbV2 = unsafe extern "C" fn(
    name_utf8: *const c_char,
    name_len: usize,
    value: *const StatorValue,
    data: *mut c_void,
) -> StatorStatus;

/// V2 named-property **query** callback.
///
/// Returns [`StatorStatus::StatorStatusOk`] when the property exists (writing
/// `v8::PropertyAttribute` flags into `*out_attrs`; `0` = `None`),
/// [`StatorStatus::StatorStatusFalse`] when it does not exist, or
/// [`StatorStatus::StatorStatusException`] (treated as missing).
///
/// # Safety
/// `name_utf8` must be valid for `name_len` bytes; `out_attrs` must be a
/// writable `*mut u32` slot if [`StatorStatus::StatorStatusOk`] is returned.
pub type StatorDomNamedQueryCb = unsafe extern "C" fn(
    name_utf8: *const c_char,
    name_len: usize,
    data: *mut c_void,
    out_attrs: *mut u32,
) -> StatorStatus;

/// V2 named-property **deleter** callback.
///
/// Returns [`StatorStatus::StatorStatusOk`] to indicate the interceptor
/// handled the delete and wrote the JS-visible "deleted" result into
/// `*out_deleted` (`true` for successful delete, `false` for non-configurable
/// / refused).  [`StatorStatus::StatorStatusFalse`] falls through to the
/// wrapper's own properties.  [`StatorStatus::StatorStatusException`] is
/// treated as fall-through.
///
/// # Safety
/// `name_utf8` must be valid for `name_len` bytes; `out_deleted` must be a
/// writable `*mut bool` slot if [`StatorStatus::StatorStatusOk`] is returned.
pub type StatorDomNamedDeleterCb = unsafe extern "C" fn(
    name_utf8: *const c_char,
    name_len: usize,
    data: *mut c_void,
    out_deleted: *mut bool,
) -> StatorStatus;

/// V2 named-property **enumerator** callback.
///
/// The callback should push each enumerable name into `buf` via
/// [`stator_dom_name_buffer_push`] and return [`StatorStatus::StatorStatusOk`]
/// on success.  Any other status is treated as "no names produced".
///
/// # Safety
/// `buf` must be the non-null pointer the FFI bridge passed in and must not
/// outlive the callback invocation.
pub type StatorDomNamedEnumeratorCb =
    unsafe extern "C" fn(buf: *mut StatorDomNameBuffer, data: *mut c_void) -> StatorStatus;

/// V2 indexed-property **getter** callback.
///
/// Semantics mirror [`StatorDomNamedGetterCbV2`] but the property key is a
/// numeric `u32` index.
///
/// # Safety
/// `out` must be a writable `*mut StatorValue` slot if
/// [`StatorStatus::StatorStatusOk`] is returned.
pub type StatorDomIndexedGetterCbV2 =
    unsafe extern "C" fn(index: u32, data: *mut c_void, out: *mut *mut StatorValue) -> StatorStatus;

/// V2 indexed-property **setter** callback.
///
/// Semantics mirror [`StatorDomNamedSetterCbV2`] for indexed access.
///
/// # Safety
/// `value` must be either null or a valid, live [`StatorValue`] pointer
/// borrowed for the call.
pub type StatorDomIndexedSetterCbV2 =
    unsafe extern "C" fn(index: u32, value: *const StatorValue, data: *mut c_void) -> StatorStatus;

/// V2 indexed-property **query** callback.
///
/// Semantics mirror [`StatorDomNamedQueryCb`] for indexed access.
///
/// # Safety
/// `out_attrs` must be a writable `*mut u32` slot if
/// [`StatorStatus::StatorStatusOk`] is returned.
pub type StatorDomIndexedQueryCb =
    unsafe extern "C" fn(index: u32, data: *mut c_void, out_attrs: *mut u32) -> StatorStatus;

/// V2 indexed-collection **length** callback.
///
/// Returns [`StatorStatus::StatorStatusOk`] with the collection length
/// written into `*out_len`.  Any other status is treated as "length unknown"
/// and the engine observes `0` — embedders should therefore reserve other
/// statuses for genuine error conditions, never for ordinary length queries.
///
/// # Safety
/// `out_len` must be a writable `*mut u32` slot if
/// [`StatorStatus::StatorStatusOk`] is returned.
pub type StatorDomIndexedLengthCb =
    unsafe extern "C" fn(data: *mut c_void, out_len: *mut u32) -> StatorStatus;

/// POD bundle of named-property interceptors, installed in one call by
/// [`stator_dom_object_wrap_install_named_handler`].
///
/// Each callback field is optional: pass `NULL` from C (or `None` from Rust)
/// to leave a particular interceptor uninstalled.  At least one callback
/// must be non-null for the install call to succeed.
///
/// `data` is an opaque embedder pointer passed verbatim to each callback;
/// the engine never dereferences it.  It is independent of internal-field 0
/// (which the legacy `set_named_getter` family uses for `data`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StatorDomNamedHandler {
    /// Named-property getter, or null.
    pub getter: Option<
        unsafe extern "C" fn(
            name_utf8: *const c_char,
            name_len: usize,
            data: *mut c_void,
            out: *mut *mut StatorValue,
        ) -> StatorStatus,
    >,
    /// Named-property setter, or null.
    pub setter: Option<
        unsafe extern "C" fn(
            name_utf8: *const c_char,
            name_len: usize,
            value: *const StatorValue,
            data: *mut c_void,
        ) -> StatorStatus,
    >,
    /// Named-property `in`/query callback, or null.
    pub query: Option<
        unsafe extern "C" fn(
            name_utf8: *const c_char,
            name_len: usize,
            data: *mut c_void,
            out_attrs: *mut u32,
        ) -> StatorStatus,
    >,
    /// Named-property `delete` callback, or null.
    pub deleter: Option<
        unsafe extern "C" fn(
            name_utf8: *const c_char,
            name_len: usize,
            data: *mut c_void,
            out_deleted: *mut bool,
        ) -> StatorStatus,
    >,
    /// Named-property enumerator callback, or null.
    pub enumerator: Option<
        unsafe extern "C" fn(buf: *mut StatorDomNameBuffer, data: *mut c_void) -> StatorStatus,
    >,
    /// Opaque embedder data passed to every callback.
    pub data: *mut c_void,
}

/// POD bundle of indexed-property interceptors, installed in one call by
/// [`stator_dom_object_wrap_install_indexed_handler`].
///
/// Each callback field is optional (`NULL` to skip).  At least one callback
/// must be non-null for the install call to succeed.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StatorDomIndexedHandler {
    /// Indexed-property getter, or null.
    pub getter: Option<
        unsafe extern "C" fn(
            index: u32,
            data: *mut c_void,
            out: *mut *mut StatorValue,
        ) -> StatorStatus,
    >,
    /// Indexed-property setter, or null.
    pub setter: Option<
        unsafe extern "C" fn(
            index: u32,
            value: *const StatorValue,
            data: *mut c_void,
        ) -> StatorStatus,
    >,
    /// Indexed-property query callback, or null.
    pub query: Option<
        unsafe extern "C" fn(index: u32, data: *mut c_void, out_attrs: *mut u32) -> StatorStatus,
    >,
    /// Indexed-collection length callback, or null.
    pub length: Option<unsafe extern "C" fn(data: *mut c_void, out_len: *mut u32) -> StatorStatus>,
    /// Opaque embedder data passed to every callback.
    pub data: *mut c_void,
}

/// Opaque name buffer passed to a [`StatorDomNamedEnumeratorCb`] callback.
///
/// The callback fills the buffer by repeatedly calling
/// [`stator_dom_name_buffer_push`].  The buffer is owned by the FFI bridge
/// and must not outlive the callback invocation.
pub struct StatorDomNameBuffer {
    names: Vec<String>,
}

// SAFETY: `StatorDomNameBuffer` only ever holds owned `String`s and is
// accessed on the owning thread during a single callback invocation.
unsafe impl Send for StatorDomNameBuffer {}

/// Append a UTF-8 name to a [`StatorDomNameBuffer`].
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on success.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `buf` is null, when
///   `name_utf8` is null while `name_len` is non-zero, or when the byte
///   range is not valid UTF-8.
///
/// # Safety
/// - `buf` must be either null or a valid pointer to a [`StatorDomNameBuffer`]
///   that is currently borrowed by an enumerator callback.
/// - `name_utf8` must point to at least `name_len` valid bytes when
///   `name_len > 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_name_buffer_push(
    buf: *mut StatorDomNameBuffer,
    name_utf8: *const c_char,
    name_len: usize,
) -> StatorStatus {
    if buf.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    if name_len > 0 && name_utf8.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    let slice: &[u8] = if name_len == 0 {
        &[]
    } else {
        // SAFETY: caller guarantees `name_utf8` is valid for `name_len` bytes.
        unsafe { std::slice::from_raw_parts(name_utf8 as *const u8, name_len) }
    };
    match std::str::from_utf8(slice) {
        Ok(s) => {
            // SAFETY: caller guarantees `buf` is a live name buffer.
            unsafe { (*buf).names.push(s.to_string()) };
            StatorStatus::StatorStatusOk
        }
        Err(_) => StatorStatus::StatorStatusInvalidArg,
    }
}

/// Install an aggregated set of named-property interceptors on `wrap`.
///
/// This **replaces** any previously-installed named-property handler
/// (whether installed through this function or through the legacy
/// [`stator_dom_object_wrap_set_named_getter`] /
/// [`stator_dom_object_wrap_set_named_setter`] pair).  Only non-null
/// callbacks in `handler` are installed; null callback fields are left
/// uninstalled and the engine falls through to the wrapper's own properties
/// for that operation.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on success.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `wrap` or `handler` is
///   null, or when every callback field in `handler` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `handler` must be a non-null, readable pointer to a
///   [`StatorDomNamedHandler`] struct.  The function pointers it carries must
///   remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_install_named_handler(
    wrap: *mut StatorDomObjectWrap,
    handler: *const StatorDomNamedHandler,
) -> StatorStatus {
    if wrap.is_null() || handler.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `handler` is readable.
    let h = unsafe { *handler };
    if h.getter.is_none()
        && h.setter.is_none()
        && h.query.is_none()
        && h.deleter.is_none()
        && h.enumerator.is_none()
    {
        return StatorStatus::StatorStatusInvalidArg;
    }

    let user_data_addr = h.data as usize;
    // SAFETY: caller guarantees `wrap` is valid for this installation.
    let isolate_addr = unsafe { (*wrap).isolate as usize };
    let mut builder = NamedPropertyHandlerConfig::builder();

    if let Some(cb) = h.getter {
        builder = builder.getter(move |name, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let isolate = isolate_addr as *mut StatorIsolate;
            let mut out: *mut StatorValue = std::ptr::null_mut();
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(name.as_ptr() as *const c_char, name.len(), data, &mut out) };
            match status {
                StatorStatus::StatorStatusOk if !out.is_null() => {
                    // SAFETY: callback transferred ownership of `out`.
                    let js = stator_value_inner_to_jsvalue(unsafe { &(*out).inner });
                    // Use `stator_value_destroy` so the isolate's live-object
                    // counter is properly decremented.
                    unsafe { stator_value_destroy(out) };
                    Some(js)
                }
                StatorStatus::StatorStatusException => {
                    // SAFETY: `isolate` is the wrapper's valid isolate pointer.
                    unsafe {
                        ensure_pending_dom_interceptor_exception(
                            isolate,
                            "DOM named getter exception",
                        )
                    };
                    None
                }
                _ => None,
            }
        });
    }
    if let Some(cb) = h.setter {
        builder = builder.setter(move |name, value, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let isolate = isolate_addr as *mut StatorIsolate;
            let c_val = StatorValue {
                inner: jsvalue_to_stator_value_inner(value),
                isolate: std::ptr::null_mut(),
            };
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(name.as_ptr() as *const c_char, name.len(), &c_val, data) };
            if status == StatorStatus::StatorStatusException {
                // SAFETY: `isolate` is the wrapper's valid isolate pointer.
                unsafe {
                    ensure_pending_dom_interceptor_exception(isolate, "DOM named setter exception")
                };
            }
            matches!(status, StatorStatus::StatorStatusOk)
        });
    }
    if let Some(cb) = h.query {
        builder = builder.query(move |name, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut attrs: u32 = 0;
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status =
                unsafe { cb(name.as_ptr() as *const c_char, name.len(), data, &mut attrs) };
            match status {
                StatorStatus::StatorStatusOk => Some(attrs),
                _ => None,
            }
        });
    }
    if let Some(cb) = h.deleter {
        builder = builder.deleter(move |name, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut deleted = false;
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe {
                cb(
                    name.as_ptr() as *const c_char,
                    name.len(),
                    data,
                    &mut deleted,
                )
            };
            matches!(status, StatorStatus::StatorStatusOk) && deleted
        });
    }
    if let Some(cb) = h.enumerator {
        builder = builder.enumerator(move |_data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut buf = StatorDomNameBuffer { names: Vec::new() };
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(&mut buf as *mut StatorDomNameBuffer, data) };
            if matches!(status, StatorStatus::StatorStatusOk) {
                buf.names
            } else {
                Vec::new()
            }
        });
    }

    // SAFETY: caller guarantees `wrap` is valid.
    unsafe {
        (*wrap).inner.set_named_handler(builder.build());
        (*wrap).has_named_setter = h.setter.is_some();
    }
    StatorStatus::StatorStatusOk
}

/// Install an aggregated set of indexed-property interceptors on `wrap`.
///
/// This **replaces** any previously-installed indexed-property handler
/// (whether installed through this function or through the legacy
/// [`stator_dom_object_wrap_set_indexed_getter`] /
/// [`stator_dom_object_wrap_set_indexed_setter`] pair).  Only non-null
/// callbacks in `handler` are installed.
///
/// Returns:
/// * [`StatorStatus::StatorStatusOk`] on success.
/// * [`StatorStatus::StatorStatusInvalidArg`] when `wrap` or `handler` is
///   null, or when every callback field in `handler` is null.
///
/// # Safety
/// - `wrap` must be a non-null, valid pointer to a live [`StatorDomObjectWrap`].
/// - `handler` must be a non-null, readable pointer to a
///   [`StatorDomIndexedHandler`] struct.  The function pointers it carries
///   must remain valid for the lifetime of the wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_dom_object_wrap_install_indexed_handler(
    wrap: *mut StatorDomObjectWrap,
    handler: *const StatorDomIndexedHandler,
) -> StatorStatus {
    if wrap.is_null() || handler.is_null() {
        return StatorStatus::StatorStatusInvalidArg;
    }
    // SAFETY: caller guarantees `handler` is readable.
    let h = unsafe { *handler };
    if h.getter.is_none() && h.setter.is_none() && h.query.is_none() && h.length.is_none() {
        return StatorStatus::StatorStatusInvalidArg;
    }

    let user_data_addr = h.data as usize;
    let mut builder = IndexedPropertyHandlerConfig::builder();

    if let Some(cb) = h.getter {
        builder = builder.getter(move |index, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut out: *mut StatorValue = std::ptr::null_mut();
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(index, data, &mut out) };
            match status {
                StatorStatus::StatorStatusOk if !out.is_null() => {
                    // SAFETY: callback transferred ownership of `out`.
                    let js = stator_value_inner_to_jsvalue(unsafe { &(*out).inner });
                    unsafe { stator_value_destroy(out) };
                    Some(js)
                }
                _ => None,
            }
        });
    }
    if let Some(cb) = h.setter {
        builder = builder.setter(move |index, value, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let c_val = StatorValue {
                inner: jsvalue_to_stator_value_inner(value),
                isolate: std::ptr::null_mut(),
            };
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(index, &c_val, data) };
            matches!(status, StatorStatus::StatorStatusOk)
        });
    }
    if let Some(cb) = h.query {
        builder = builder.query(move |index, _data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut attrs: u32 = 0;
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(index, data, &mut attrs) };
            match status {
                StatorStatus::StatorStatusOk => Some(attrs),
                _ => None,
            }
        });
    }
    if let Some(cb) = h.length {
        builder = builder.length(move |_data_field0| {
            let data = user_data_addr as *mut c_void;
            let mut out_len: u32 = 0;
            // SAFETY: `cb` is a C fn pointer the embedder guarantees valid.
            let status = unsafe { cb(data, &mut out_len) };
            if matches!(status, StatorStatus::StatorStatusOk) {
                out_len
            } else {
                0
            }
        });
    }

    // SAFETY: caller guarantees `wrap` is valid.
    unsafe { (*wrap).inner.set_indexed_handler(builder.build()) };
    StatorStatus::StatorStatusOk
}

// ── Event loop FFI ─────────────────────────────────────────────────────────────

use stator_jse::builtins::promise::{
    MicrotaskQueue, PromiseRejectionEventKind, drain_active_microtask_queue,
    drain_active_promise_rejection_events,
};
use stator_jse::event_loop::{DefaultCallbacks, EmbedderCallbacks, EventLoop, TimerHandle};

/// Drain the active thread-local Promise microtask queue installed by Stator
/// globals.
///
/// This mirrors an embedder microtask checkpoint for scripts that use
/// `Promise.resolve(...).then(...)` and other Promise reaction jobs.  Returns the
/// number of microtasks drained, or 0 when no active queue has been installed on
/// the current thread.
#[unsafe(no_mangle)]
pub extern "C" fn stator_drain_active_microtask_queue() -> usize {
    drain_active_microtask_queue()
}

/// Host-visible Promise rejection event kind.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatorPromiseRejectionEventKind {
    /// A Promise was rejected with no rejection handler attached.
    StatorPromiseRejectionEventKindRejectedWithNoHandler = 0,
    /// A rejection handler was attached after the host observed the rejection.
    StatorPromiseRejectionEventKindHandlerAddedAfterReject = 1,
}

/// C-callable Promise rejection event callback signature.
///
/// `reason_utf8` is valid only for the duration of the callback and is not
/// null-terminated. Embedders must copy it if they need to retain the string.
pub type StatorPromiseRejectionEventCallback = unsafe extern "C" fn(
    kind: StatorPromiseRejectionEventKind,
    promise_id: usize,
    reason_utf8: *const c_char,
    reason_len: usize,
    user_data: *mut c_void,
);

/// Drain active host-visible Promise rejection events.
///
/// Returns the number of drained events.
///
/// # Safety
/// `callback` must be safe to call for each pending event, and `user_data` must
/// be valid for the callback's expectations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_drain_active_promise_rejection_events(
    callback: StatorPromiseRejectionEventCallback,
    user_data: *mut c_void,
) -> usize {
    let events = drain_active_promise_rejection_events();
    let count = events.len();
    for event in events {
        let kind = match event.kind {
            PromiseRejectionEventKind::RejectedWithNoHandler => {
                StatorPromiseRejectionEventKind::StatorPromiseRejectionEventKindRejectedWithNoHandler
            }
            PromiseRejectionEventKind::HandlerAddedAfterReject => {
                StatorPromiseRejectionEventKind::StatorPromiseRejectionEventKindHandlerAddedAfterReject
            }
        };
        let reason = event.reason.as_bytes();
        // SAFETY: `callback` was provided by the embedder, and `reason` points
        // into `event` for the duration of this call.
        unsafe {
            callback(
                kind,
                event.promise_id,
                reason.as_ptr() as *const c_char,
                reason.len(),
                user_data,
            )
        };
    }
    count
}

/// Discard active host-visible Promise rejection events.
///
/// Returns the number of discarded events.
#[unsafe(no_mangle)]
pub extern "C" fn stator_discard_active_promise_rejection_events() -> usize {
    drain_active_promise_rejection_events().len()
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Inspector (in-process CDP) FFI
// ─────────────────────────────────────────────────────────────────────────────
//
// The inspector surface is **additive** — it does not modify or replace the
// existing standalone WebSocket CDP server.  Embedders that prefer to drive
// the CDP dispatcher directly (e.g. Edge DevTools) construct a
// `StatorInspector` from a `StatorContext`, open one or more
// `StatorInspectorSession`s, and exchange JSON-RPC frames synchronously.
//
// # Threading and lifetime
//
// All inspector calls are **synchronous**, **non-reentrant**, and must be
// issued from the isolate's owning thread.  `ctx` must outlive the
// inspector and the inspector must outlive every session it produces.
// Pointers returned by `stator_inspector_next_message` are engine-owned
// and remain valid only until the next inspector call on the same session.

/// Opaque in-process inspector handle.
///
/// Owns a script registry and a set of sessions sharing a single
/// [`StatorContext`]'s globals environment.
pub struct StatorInspector {
    inner: stator_jse::inspector::api::InProcessInspector,
}

/// Opaque inspector session handle.  Lifetime is bounded by the inspector
/// that produced it; never freed directly by the embedder.
pub struct StatorInspectorSession {
    inner: *mut stator_jse::inspector::api::InProcessInspectorSession,
}

/// Build an inspector that shares its context's globals environment.
///
/// Returns a null pointer if `ctx` is null.  The returned pointer must
/// eventually be released via [`stator_inspector_destroy`].
///
/// # Safety
/// - `ctx` must be either null or a valid, live [`StatorContext`] pointer.
/// - The returned inspector borrows `ctx`'s globals; `ctx` must outlive the
///   inspector.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_create(ctx: *mut StatorContext) -> *mut StatorInspector {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `ctx` is valid.
    let globals = unsafe { Rc::clone(&(*ctx).globals) };
    Box::into_raw(Box::new(StatorInspector {
        inner: stator_jse::inspector::api::InProcessInspector::new(globals),
    }))
}

/// Destroy an inspector previously returned by [`stator_inspector_create`].
///
/// All sessions still open on this inspector are dropped first; any
/// session handles previously returned by [`stator_inspector_connect`]
/// become invalid immediately.  Passing a null pointer is a no-op.
///
/// # Safety
/// `inspector` must be either null or a pointer previously returned by
/// [`stator_inspector_create`] and not yet destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_destroy(inspector: *mut StatorInspector) {
    if !inspector.is_null() {
        // SAFETY: caller guarantees `inspector` is a live, unique pointer.
        unsafe {
            drop(Box::from_raw(inspector));
        }
    }
}

/// Open a new CDP session against `inspector` and return a borrowed,
/// engine-owned handle.
///
/// `session_id` is an opaque embedder-supplied identifier echoed back via
/// future inspector APIs; it has no protocol-level meaning.  Returns a
/// null pointer if `inspector` is null.
///
/// # Safety
/// `inspector` must be a non-null pointer returned by
/// [`stator_inspector_create`].  The returned session pointer is owned by
/// the inspector and must not be freed by the embedder; release it via
/// [`stator_inspector_disconnect`] instead.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_connect(
    inspector: *mut StatorInspector,
    session_id: u32,
) -> *mut StatorInspectorSession {
    if inspector.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `inspector` is valid.
    let inner_session = unsafe { (*inspector).inner.connect(session_id) }
        as *mut stator_jse::inspector::api::InProcessInspectorSession;
    Box::into_raw(Box::new(StatorInspectorSession {
        inner: inner_session,
    }))
}

/// Close a session previously returned by [`stator_inspector_connect`].
///
/// The underlying session is removed from the inspector and dropped.  The
/// outer handle is also freed.  Either argument being null is a no-op.
///
/// # Safety
/// - `inspector` must be either null or a live [`StatorInspector`] pointer.
/// - `session` must be either null or a pointer returned by
///   [`stator_inspector_connect`] on `inspector` and not yet disconnected.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_disconnect(
    inspector: *mut StatorInspector,
    session: *mut StatorInspectorSession,
) {
    if inspector.is_null() || session.is_null() {
        return;
    }
    // SAFETY: caller guarantees both pointers are valid.
    unsafe {
        let session_box = Box::from_raw(session);
        (*inspector).inner.disconnect(session_box.inner);
    }
}

/// Submit a JSON-RPC request `json` (`len` bytes, UTF-8) to `session`.
///
/// Returns `0` on success: the request parsed as JSON and a corresponding
/// protocol response (plus any associated events) has been pushed onto the
/// session's outbox.  In-protocol errors such as "unknown method" or
/// missing parameters also return `0`; the error payload is delivered via
/// the outbox.
///
/// Returns `-1` if `session` or `json` is null, or if `len` overflows.
/// Returns `1` if `json` is not a valid UTF-8 JSON-RPC request; a
/// parse-error response is still pushed onto the outbox as a courtesy to
/// the caller.
///
/// # Safety
/// - `session` must be a non-null pointer returned by
///   [`stator_inspector_connect`].
/// - `json` must be valid for reads of `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_dispatch(
    session: *mut StatorInspectorSession,
    json: *const c_char,
    len: usize,
) -> i32 {
    if session.is_null() || json.is_null() {
        return -1;
    }
    // SAFETY: caller guarantees `json` is valid for `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(json as *const u8, len) };
    // SAFETY: caller guarantees `session` is a live FFI wrapper whose
    // `inner` points to a session owned by the parent inspector.
    let inner = unsafe { &mut *(*session).inner };
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            inner.dispatch_parse_error("Parse error: invalid UTF-8 JSON-RPC request".to_string());
            return 1;
        }
    };
    match inner.dispatch_json(text) {
        stator_jse::inspector::cdp::DispatchOutcome::Ok => 0,
        stator_jse::inspector::cdp::DispatchOutcome::ParseError => 1,
    }
}

/// Number of protocol messages currently waiting in `session`'s outbox.
///
/// Returns `0` if `session` is null.
///
/// # Safety
/// `session` must be either null or a non-null pointer returned by
/// [`stator_inspector_connect`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_pending_count(
    session: *const StatorInspectorSession,
) -> usize {
    if session.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `session` is a live wrapper.
    let inner = unsafe { &*(*session).inner };
    inner.pending_count()
}

/// Pop the oldest message from `session`'s outbox.
///
/// On success returns a non-null pointer to UTF-8 bytes (no trailing NUL)
/// and writes the byte length to `*out_len`.  The returned pointer is
/// engine-owned and remains valid until the **next inspector call on this
/// session** (any of `stator_inspector_dispatch`,
/// `stator_inspector_next_message`, or `stator_inspector_disconnect`).
///
/// Returns a null pointer when the outbox is empty.  When `session` is
/// null, returns null without touching `out_len`.  When `out_len` is null,
/// the function still returns the data pointer and the caller must
/// supply the length itself (e.g. via [`stator_inspector_pending_count`]
/// before the call); however, this mode is discouraged.
///
/// # Safety
/// - `session` must be either null or a non-null pointer returned by
///   [`stator_inspector_connect`].
/// - `out_len` must be either null or a valid writable `size_t` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_next_message(
    session: *mut StatorInspectorSession,
    out_len: *mut usize,
) -> *const u8 {
    if session.is_null() {
        return std::ptr::null();
    }
    // SAFETY: caller guarantees `session` is a live wrapper.
    let inner = unsafe { &mut *(*session).inner };
    match inner.take_next_bytes() {
        Some(bytes) => {
            if !out_len.is_null() {
                // SAFETY: caller guarantees `out_len` is writable when non-null.
                unsafe {
                    *out_len = bytes.len();
                }
            }
            bytes.as_ptr()
        }
        None => {
            if !out_len.is_null() {
                // SAFETY: caller guarantees `out_len` is writable when non-null.
                unsafe {
                    *out_len = 0;
                }
            }
            std::ptr::null()
        }
    }
}

/// Register `source` (`source_len` UTF-8 bytes) with `inspector`'s script
/// registry and emit `Debugger.scriptParsed` to every session whose
/// `Debugger` domain has been enabled.
///
/// Returns the freshly assigned, monotonically increasing non-zero script
/// ID, or `0` on error (null inspector, null source, or invalid UTF-8).
/// The `_script` argument is currently retained for future linkage between
/// the script handle and the inspector's registry; passing null is
/// allowed.
///
/// # Safety
/// - `inspector` must be a non-null pointer returned by
///   [`stator_inspector_create`].
/// - `source` must be valid for reads of `source_len` bytes.
/// - `_script` is treated as opaque and may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_inspector_register_script(
    inspector: *mut StatorInspector,
    _script: *mut StatorScript,
    source: *const c_char,
    source_len: usize,
) -> u32 {
    if inspector.is_null() || source.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `source` is valid for `source_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, source_len) };
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    // SAFETY: caller guarantees `inspector` is valid.
    unsafe { (*inspector).inner.register_script(text) }
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

    unsafe extern "C" fn test_resolver_free_user_data(user_data: *mut c_void) {
        assert!(!user_data.is_null());
        // SAFETY: tests pass a valid mutable `usize` pointer as user data.
        let counter = unsafe { &mut *(user_data as *mut usize) };
        *counter += 1;
    }

    static NULL_RESOLVER_FREE_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    unsafe extern "C" fn test_null_resolver_free_user_data(_user_data: *mut c_void) {
        NULL_RESOLVER_FREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    unsafe extern "C" fn test_nullable_module_resolver_cb(
        _ctx: *mut StatorContext,
        _user_data: *mut c_void,
        _referrer: *const StatorModule,
        _origin: *const StatorModuleOrigin,
        _specifier: *const c_char,
        _specifier_len: usize,
        _attributes: *const StatorImportAttribute,
        _attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_module = std::ptr::null_mut() };
        }
        if !out_error.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_error = std::ptr::null_mut() };
        }
        StatorResolveStatus::StatorResolveStatusNotFound
    }

    unsafe extern "C" fn test_module_resolver_cb(
        ctx: *mut StatorContext,
        user_data: *mut c_void,
        referrer: *const StatorModule,
        _origin: *const StatorModuleOrigin,
        specifier: *const c_char,
        specifier_len: usize,
        attributes: *const StatorImportAttribute,
        attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        assert!(!ctx.is_null());
        assert!(!user_data.is_null());
        assert!(!referrer.is_null());
        assert!(!specifier.is_null());
        assert!(attributes.is_null());
        assert_eq!(attributes_len, 0);
        // SAFETY: the test passes a valid UTF-8 specifier buffer.
        let specifier =
            unsafe { std::slice::from_raw_parts(specifier as *const u8, specifier_len) };
        assert_eq!(specifier, b"./dep.js");
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in this test.
            unsafe { *out_module = std::ptr::null_mut() };
        }
        if !out_error.is_null() {
            // SAFETY: out pointer is valid for one write in this test.
            unsafe { *out_error = std::ptr::null_mut() };
        }
        StatorResolveStatus::StatorResolveStatusNotFound
    }

    #[test]
    fn test_context_set_module_resolver_rejects_malformed_clear() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut counter = 0usize;

        // SAFETY: null callback with non-null cleanup state is rejected.
        let ok = unsafe {
            stator_context_set_module_resolver(
                ctx,
                None,
                &mut counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        };
        assert!(!ok);
        assert_eq!(counter, 0);
        // SAFETY: `ctx` is valid and visible inside crate tests.
        assert!(unsafe { (*ctx).module_resolver.is_none() });

        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_set_module_resolver_replaces_and_frees_user_data() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut first_counter = 0usize;
        let mut second_counter = 0usize;

        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut first_counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        assert_eq!(first_counter, 0);

        // SAFETY: replacing drops the previous resolver and frees its data.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut second_counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        assert_eq!(first_counter, 1);
        assert_eq!(second_counter, 0);

        // SAFETY: clearing drops the active resolver and frees its data.
        assert!(unsafe {
            stator_context_set_module_resolver(ctx, None, std::ptr::null_mut(), None)
        });
        assert_eq!(first_counter, 1);
        assert_eq!(second_counter, 1);

        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
    }

    #[test]
    fn test_context_set_module_resolver_clear_frees_user_data_once() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut counter = 0usize;

        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        // SAFETY: clearing drops the active resolver and frees its data once.
        assert!(unsafe {
            stator_context_set_module_resolver(ctx, None, std::ptr::null_mut(), None)
        });
        assert_eq!(counter, 1);

        // SAFETY: destroying after clear must not re-free cleared user data.
        unsafe { stator_context_destroy(ctx) };
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_context_module_resolver_frees_user_data_on_destroy() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut counter = 0usize;

        // SAFETY: callback and user data remain live until context destroy.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        // SAFETY: dropping the context drops the resolver.
        unsafe { stator_context_destroy(ctx) };
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_context_module_resolver_cleanup_not_double_called_after_replace_destroy() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut first_counter = 0usize;
        let mut second_counter = 0usize;

        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut first_counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        // SAFETY: replacing drops the first resolver exactly once.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut second_counter as *mut usize as *mut c_void,
                Some(test_resolver_free_user_data),
            )
        });
        assert_eq!(first_counter, 1);
        assert_eq!(second_counter, 0);

        // SAFETY: destroying drops only the active second resolver.
        unsafe { stator_context_destroy(ctx) };
        assert_eq!(first_counter, 1);
        assert_eq!(second_counter, 1);
    }

    #[test]
    fn test_context_module_resolver_null_user_data_and_callback_combinations_are_safe() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut owned_without_cleanup = 7usize;

        NULL_RESOLVER_FREE_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: null user data with a free callback is allowed; the callback
        // is not invoked because there is no non-null user data to clean up.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_nullable_module_resolver_cb),
                std::ptr::null_mut(),
                Some(test_null_resolver_free_user_data),
            )
        });
        // SAFETY: clearing a resolver with null user data must not call free.
        assert!(unsafe {
            stator_context_set_module_resolver(ctx, None, std::ptr::null_mut(), None)
        });
        assert_eq!(
            NULL_RESOLVER_FREE_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        // SAFETY: non-null user data with null free callback is allowed; Stator
        // borrows but does not own the pointer.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_nullable_module_resolver_cb),
                &mut owned_without_cleanup as *mut usize as *mut c_void,
                None,
            )
        });
        // SAFETY: replacing with null user data and a free callback drops the
        // previous resolver without cleanup and stores the new no-op cleanup
        // state safely.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_nullable_module_resolver_cb),
                std::ptr::null_mut(),
                Some(test_null_resolver_free_user_data),
            )
        });
        assert_eq!(owned_without_cleanup, 7);
        // SAFETY: context destruction must not invoke a free callback for null data.
        unsafe { stator_context_destroy(ctx) };
        assert_eq!(
            NULL_RESOLVER_FREE_CALLS.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[test]
    fn test_context_module_resolver_callback_shape_invokes() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let mut user_data = 7usize;
        let module = compile_module_src("export const value = 1;");
        assert!(!module.is_null());

        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_module_resolver_cb),
                &mut user_data as *mut usize as *mut c_void,
                None,
            )
        });

        let specifier = b"./dep.js";
        let mut out_module: *mut StatorModule = std::ptr::null_mut();
        let mut out_error: *mut StatorString = std::ptr::null_mut();
        let origin = StatorModuleOrigin {
            base_url: std::ptr::null(),
            base_url_len: 0,
            integrity_metadata: std::ptr::null(),
            integrity_metadata_len: 0,
            credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
            referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
            parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
        };
        // SAFETY: the stored callback and all pointers are valid for this test.
        let status = unsafe {
            let resolver = (*ctx).module_resolver.as_ref().unwrap();
            (resolver.callback)(
                ctx,
                resolver.user_data,
                module,
                &origin,
                specifier.as_ptr() as *const c_char,
                specifier.len(),
                std::ptr::null(),
                0,
                &mut out_module,
                &mut out_error,
            )
        };
        assert_eq!(status, StatorResolveStatus::StatorResolveStatusNotFound);
        assert!(out_module.is_null());
        assert!(out_error.is_null());

        // SAFETY: pointers are non-null and live.
        unsafe {
            stator_module_free(module);
            stator_context_destroy(ctx);
        }
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

    /// Helper: compile a module source string and return the `StatorModule` pointer.
    ///
    /// The returned pointer must be freed with `stator_module_free`.
    fn compile_module_src(src: &str) -> *mut StatorModule {
        let bytes = src.as_bytes();
        // SAFETY: null ctx is permitted; `bytes` is valid UTF-8.
        unsafe {
            stator_module_compile(
                std::ptr::null_mut(),
                bytes.as_ptr() as *const c_char,
                bytes.len(),
            )
        }
    }

    struct TestGraphResolverData {
        modules: HashMap<String, *mut StatorModule>,
        status: StatorResolveStatus,
        calls: Vec<String>,
        attributes: Vec<Vec<(String, String)>>,
        cleanup_calls: usize,
    }

    unsafe fn import_attribute_pair(attribute: &StatorImportAttribute) -> (String, String) {
        let key = if attribute.key.is_null() {
            String::new()
        } else {
            // SAFETY: test callers pass valid FFI attribute slices.
            let bytes = unsafe {
                std::slice::from_raw_parts(attribute.key as *const u8, attribute.key_len)
            };
            std::str::from_utf8(bytes).unwrap().to_string()
        };
        let value = if attribute.value.is_null() {
            String::new()
        } else {
            // SAFETY: test callers pass valid FFI attribute slices.
            let bytes = unsafe {
                std::slice::from_raw_parts(attribute.value as *const u8, attribute.value_len)
            };
            std::str::from_utf8(bytes).unwrap().to_string()
        };
        (key, value)
    }

    unsafe extern "C" fn test_graph_resolver_cb(
        _ctx: *mut StatorContext,
        user_data: *mut c_void,
        _referrer: *const StatorModule,
        origin: *const StatorModuleOrigin,
        specifier: *const c_char,
        specifier_len: usize,
        attributes: *const StatorImportAttribute,
        attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        assert!(!user_data.is_null());
        assert!(!specifier.is_null());
        assert!(!origin.is_null());
        if attributes_len == 0 {
            assert!(attributes.is_null());
        }
        // SAFETY: tests pass a valid resolver data pointer.
        let data = unsafe { &mut *(user_data as *mut TestGraphResolverData) };
        // SAFETY: callback contract supplies a valid specifier byte slice.
        let specifier_bytes =
            unsafe { std::slice::from_raw_parts(specifier as *const u8, specifier_len) };
        let specifier = std::str::from_utf8(specifier_bytes).unwrap().to_string();
        data.calls.push(specifier.clone());
        let attributes = if attributes_len == 0 {
            Vec::new()
        } else {
            assert!(!attributes.is_null());
            // SAFETY: callback contract supplies `attributes_len` valid entries.
            let slice = unsafe { std::slice::from_raw_parts(attributes, attributes_len) };
            slice
                .iter()
                .map(|attribute| {
                    // SAFETY: `slice` was built from valid FFI attribute entries.
                    unsafe { import_attribute_pair(attribute) }
                })
                .collect()
        };
        data.attributes.push(attributes);
        if !out_error.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_error = std::ptr::null_mut() };
        }
        if data.status != StatorResolveStatus::StatorResolveStatusOk {
            if !out_module.is_null() {
                // SAFETY: out pointer is valid for one write in these tests.
                unsafe { *out_module = std::ptr::null_mut() };
            }
            return data.status;
        }
        let module = data
            .modules
            .get(&specifier)
            .copied()
            .unwrap_or(std::ptr::null_mut());
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_module = module };
        }
        if module.is_null() {
            StatorResolveStatus::StatorResolveStatusNotFound
        } else {
            StatorResolveStatus::StatorResolveStatusOk
        }
    }

    unsafe extern "C" fn test_graph_resolver_free_user_data(user_data: *mut c_void) {
        assert!(!user_data.is_null());
        // SAFETY: tests pass a valid mutable `TestGraphResolverData` pointer.
        let data = unsafe { &mut *(user_data as *mut TestGraphResolverData) };
        data.cleanup_calls += 1;
    }

    struct TestDetailResolverData {
        status: StatorResolveStatus,
        detail: &'static str,
        calls: Vec<String>,
    }

    unsafe extern "C" fn test_detail_resolver_cb(
        _ctx: *mut StatorContext,
        user_data: *mut c_void,
        _referrer: *const StatorModule,
        _origin: *const StatorModuleOrigin,
        specifier: *const c_char,
        specifier_len: usize,
        _attributes: *const StatorImportAttribute,
        attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        assert_eq!(attributes_len, 0);
        // SAFETY: tests pass a valid resolver data pointer.
        let data = unsafe { &mut *(user_data as *mut TestDetailResolverData) };
        // SAFETY: callback contract supplies a valid specifier byte slice.
        let specifier_bytes =
            unsafe { std::slice::from_raw_parts(specifier as *const u8, specifier_len) };
        data.calls
            .push(std::str::from_utf8(specifier_bytes).unwrap().to_string());
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in this test.
            unsafe { *out_module = std::ptr::null_mut() };
        }
        if !out_error.is_null() {
            let detail = data.detail.as_bytes();
            // SAFETY: detail bytes are valid for the duration of this call and
            // `stator_string_new` copies them.
            unsafe {
                *out_error = stator_string_new(detail.as_ptr() as *const c_char, detail.len());
            }
        }
        data.status
    }

    struct TestAttributeGateResolverData {
        modules: HashMap<String, *mut StatorModule>,
        expected: Vec<(String, String)>,
        calls: Vec<String>,
        attributes: Vec<Vec<(String, String)>>,
    }

    unsafe extern "C" fn test_attribute_gate_resolver_cb(
        _ctx: *mut StatorContext,
        user_data: *mut c_void,
        _referrer: *const StatorModule,
        origin: *const StatorModuleOrigin,
        specifier: *const c_char,
        specifier_len: usize,
        attributes: *const StatorImportAttribute,
        attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        assert!(!origin.is_null());
        // SAFETY: tests pass a valid resolver data pointer.
        let data = unsafe { &mut *(user_data as *mut TestAttributeGateResolverData) };
        // SAFETY: callback contract supplies a valid specifier byte slice.
        let specifier_bytes =
            unsafe { std::slice::from_raw_parts(specifier as *const u8, specifier_len) };
        let specifier = std::str::from_utf8(specifier_bytes).unwrap().to_string();
        data.calls.push(specifier.clone());

        let actual = if attributes_len == 0 {
            Vec::new()
        } else {
            assert!(!attributes.is_null());
            // SAFETY: callback contract supplies `attributes_len` valid entries.
            let slice = unsafe { std::slice::from_raw_parts(attributes, attributes_len) };
            slice
                .iter()
                .map(|attribute| {
                    // SAFETY: `slice` was built from valid FFI attribute entries.
                    unsafe { import_attribute_pair(attribute) }
                })
                .collect()
        };
        data.attributes.push(actual.clone());

        if actual != data.expected {
            if !out_module.is_null() {
                // SAFETY: out pointer is valid for one write in these tests.
                unsafe { *out_module = std::ptr::null_mut() };
            }
            if !out_error.is_null() {
                let detail = b"import attribute mismatch";
                // SAFETY: `detail` is a live byte slice for this call.
                let error =
                    unsafe { stator_string_new(detail.as_ptr() as *const c_char, detail.len()) };
                // SAFETY: out pointer is valid for one write in these tests.
                unsafe { *out_error = error };
            }
            return StatorResolveStatus::StatorResolveStatusTypeError;
        }

        if !out_error.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_error = std::ptr::null_mut() };
        }
        let module = data
            .modules
            .get(&specifier)
            .copied()
            .unwrap_or(std::ptr::null_mut());
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_module = module };
        }
        if module.is_null() {
            StatorResolveStatus::StatorResolveStatusNotFound
        } else {
            StatorResolveStatus::StatorResolveStatusOk
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
    fn test_module_compile_top_level_await_returns_record() {
        let module = compile_module_src("const x = await 1; export { x };");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert!(stator_module_get_error(module).is_null());
            assert!(stator_module_bytecode_count(module) > 0);
            assert!(stator_module_is_async(module));
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusUnlinked
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_evaluate_dependency_free_returns_result() {
        let module = compile_module_src("40 + 2;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live; null context is documented.
        unsafe {
            assert!(!stator_module_has_dependencies(module));
            let result = stator_module_evaluate(module, std::ptr::null_mut());
            assert!(!result.is_null());
            assert_eq!(stator_value_to_number(result), 42.0);
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusEvaluated
            );
            assert!(stator_module_get_error(module).is_null());
            stator_value_destroy(result);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_evaluate_dependency_free_propagates_error() {
        let module = compile_module_src("1n + 1;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live; null context is documented.
        unsafe {
            let result = stator_module_evaluate(module, std::ptr::null_mut());
            assert!(result.is_null());
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindType
            );
            assert!(!stator_module_get_error(module).is_null());
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_evaluate_dynamic_import_uses_ffi_resolver_error_detail() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import('./missing.js');");
        let mut data = TestDetailResolverData {
            status: StatorResolveStatus::StatorResolveStatusNetworkError,
            detail: "offline while fetching",
            calls: Vec::new(),
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_detail_resolver_cb),
                &mut data as *mut TestDetailResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            let result = stator_module_evaluate(root, ctx);
            assert!(result.is_null());
            assert_eq!(data.calls, vec!["./missing.js"]);
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindType
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap();
            assert!(err.contains("failed to fetch module './missing.js'"));
            assert!(err.contains("offline while fetching"));
            stator_module_free(root);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_import_meta_resolve_routes_to_ffi_resolver() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import.meta.resolve('./dep.js');");
        let dep = compile_module_src("export const value = 1;");
        let dep_url = c"https://cdn.example.test/dep.js";
        // SAFETY: `dep` is valid and `dep_url` is a valid C string.
        unsafe { stator_module_set_origin(dep, dep_url.as_ptr(), 0, 0) };
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            assert_eq!(data.calls, vec!["./dep.js"]);
            let resolved = CStr::from_ptr(stator_value_as_string(result))
                .to_str()
                .unwrap();
            assert_eq!(resolved, "https://cdn.example.test/dep.js");
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_import_meta_resolve_fails_closed_without_resolver() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import.meta.resolve('./dep.js');");

        // SAFETY: pointers are non-null and live.
        unsafe {
            let result = stator_module_evaluate(root, ctx);
            assert!(result.is_null());
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindType
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap();
            assert!(err.contains("import.meta.resolve is not supported by this host"));
            stator_module_free(root);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_import_meta_url_uses_resource_name() {
        let module = compile_module_src("import.meta.url;");
        let url = c"https://app.example.test/root.mjs";
        // SAFETY: `module` is valid and `url` is a valid C string.
        unsafe { stator_module_set_origin(module, url.as_ptr(), 0, 0) };

        // SAFETY: `module` is non-null and live; null context is documented.
        unsafe {
            let result = stator_module_evaluate(module, std::ptr::null_mut());
            assert!(!result.is_null());
            let meta_url = CStr::from_ptr(stator_value_as_string(result))
                .to_str()
                .unwrap();
            assert_eq!(meta_url, "https://app.example.test/root.mjs");
            stator_value_destroy(result);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_evaluate_ffi_resolver_replacement_uses_current_resolver() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import.meta.resolve('./dep2.js');");
        let dep1 = compile_module_src("export const value = 1;");
        let dep2 = compile_module_src("export const value = 2;");
        let dep2_url = c"https://cdn.example.test/dep2.js";
        // SAFETY: `dep2` is valid and `dep2_url` is a valid C string.
        unsafe { stator_module_set_origin(dep2, dep2_url.as_ptr(), 0, 0) };

        let mut modules1 = HashMap::new();
        modules1.insert("./dep1.js".to_string(), dep1);
        let mut data1 = TestGraphResolverData {
            modules: modules1,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        let mut modules2 = HashMap::new();
        modules2.insert("./dep2.js".to_string(), dep2);
        let mut data2 = TestGraphResolverData {
            modules: modules2,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };

        // SAFETY: callbacks and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data1 as *mut TestGraphResolverData as *mut c_void,
                Some(test_graph_resolver_free_user_data),
            )
        });
        // SAFETY: replacing the resolver is permitted outside callback entry.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data2 as *mut TestGraphResolverData as *mut c_void,
                Some(test_graph_resolver_free_user_data),
            )
        });
        assert_eq!(data1.cleanup_calls, 1);

        // SAFETY: pointers are non-null and live.
        unsafe {
            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            assert!(data1.calls.is_empty());
            assert_eq!(data2.calls, vec!["./dep2.js"]);
            let resolved = CStr::from_ptr(stator_value_as_string(result))
                .to_str()
                .unwrap();
            assert_eq!(resolved, "https://cdn.example.test/dep2.js");
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep1);
            stator_module_free(dep2);
            stator_context_destroy(ctx);
        }
        assert_eq!(data2.cleanup_calls, 1);
    }

    #[test]
    fn test_module_evaluate_blocks_unlinked_static_dependencies() {
        let module = compile_module_src("import value from 'dep'; value;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live; null context is documented.
        unsafe {
            assert!(stator_module_has_dependencies(module));
            let result = stator_module_evaluate(module, std::ptr::null_mut());
            assert!(result.is_null());
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindInternal
            );
            let err = CStr::from_ptr(stator_module_get_error(module))
                .to_str()
                .unwrap();
            assert!(err.contains("host import resolution"));
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_evaluate_named_import_reads_exported_const() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { value } from './dep.js'; value + 1;");
        let dep = compile_module_src("export const value = 41;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            if result.is_null() {
                let err = CStr::from_ptr(stator_module_get_error(root))
                    .to_str()
                    .unwrap();
                panic!("module evaluation failed: {err}");
            }
            assert_eq!(stator_value_to_number(result), 42.0);
            assert_eq!(
                stator_module_get_status(dep),
                StatorModuleStatus::StatorModuleStatusEvaluated
            );
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_default_import_reads_default_expression() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import answer from './dep.js'; answer + 1;");
        let dep = compile_module_src("export default 41;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            assert_eq!(stator_value_to_number(result), 42.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_exported_let_update_remains_pending() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { counter } from './dep.js'; counter;");
        let dep = compile_module_src("export let counter = 0; counter = 1;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            // Current bytecode only stores the declaration initializer into the
            // module cell; subsequent local assignments do not update it yet.
            assert_eq!(stator_value_to_number(result), 0.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_namespace_import_exposes_named_and_default_exports() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root =
            compile_module_src("import * as ns from './dep.js'; ns.value + ns.other + ns.default;");
        let dep =
            compile_module_src("export const value = 1; export const other = 2; export default 3;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            if result.is_null() {
                let err = CStr::from_ptr(stator_module_get_error(root))
                    .to_str()
                    .unwrap();
                panic!("module evaluation failed: {err}");
            }
            assert_eq!(stator_value_to_number(result), 6.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_namespace_import_missing_property_is_undefined() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src(
            "import * as ns from './dep.js'; typeof ns.notThere === 'undefined' ? 7 : 0;",
        );
        let dep = compile_module_src("export const value = 1;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            assert_eq!(stator_value_to_number(result), 7.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_namespace_reexport_with_alias_is_readable() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { ns } from './re.js'; ns.value + ns.other;");
        // The intermediate module re-exports the dep namespace under name `ns`.
        let re = compile_module_src("export * as ns from './dep.js';");
        let dep = compile_module_src("export const value = 10; export const other = 20;");
        let mut modules = HashMap::new();
        modules.insert("./re.js".to_string(), re);
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            if result.is_null() {
                let err = CStr::from_ptr(stator_module_get_error(root))
                    .to_str()
                    .unwrap();
                panic!("module evaluation failed: {err}");
            }
            assert_eq!(stator_value_to_number(result), 30.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(re);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_namespace_property_assignment_throws_in_strict_mode() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // Modules are always strict, and the module namespace object is
        // frozen, so assignment to a namespace property must throw a
        // TypeError rather than silently mutating the export.
        let root = compile_module_src("import * as ns from './dep.js'; ns.value = 99; ns.value;");
        let dep = compile_module_src("export const value = 41;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            assert!(result.is_null(), "frozen namespace store must throw");
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindType
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_namespace_import_isolates_per_specifier() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src(
            "import * as a from './a.js'; import * as b from './b.js'; a.value + b.value;",
        );
        let a = compile_module_src("export const value = 100;");
        let b = compile_module_src("export const value = 7;");
        let mut modules = HashMap::new();
        modules.insert("./a.js".to_string(), a);
        modules.insert("./b.js".to_string(), b);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            let result = stator_module_evaluate(root, ctx);
            if result.is_null() {
                let err = CStr::from_ptr(stator_module_get_error(root))
                    .to_str()
                    .unwrap();
                panic!("module evaluation failed: {err}");
            }
            assert_eq!(stator_value_to_number(result), 107.0);
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(a);
            stator_module_free(b);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_no_request_module_links_and_evaluates() {
        let module = compile_module_src("40 + 2;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and dependency-free, so null context is permitted.
        unsafe {
            assert!(stator_module_instantiate(std::ptr::null_mut(), module));
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            let result = stator_module_evaluate(module, std::ptr::null_mut());
            assert!(!result.is_null());
            assert_eq!(stator_value_to_number(result), 42.0);
            stator_value_destroy(result);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_request_enumeration_reports_imports_and_reexports() {
        let module = compile_module_src("import a from './a.js'; export { b } from './b.js';");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert_eq!(stator_module_get_request_count(module), 2);

            let mut specifier = std::ptr::null();
            let mut specifier_len = 0usize;
            let mut attributes = std::ptr::null();
            let mut attributes_len = 1usize;
            assert!(stator_module_get_request(
                module,
                0,
                &mut specifier,
                &mut specifier_len,
                &mut attributes,
                &mut attributes_len,
            ));
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(
                    specifier as *const u8,
                    specifier_len
                ))
                .unwrap(),
                "./a.js"
            );
            assert!(attributes.is_null());
            assert_eq!(attributes_len, 0);

            assert!(stator_module_get_request(
                module,
                1,
                &mut specifier,
                &mut specifier_len,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ));
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(
                    specifier as *const u8,
                    specifier_len
                ))
                .unwrap(),
                "./b.js"
            );
            assert!(!stator_module_get_request(
                module,
                2,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ));
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_request_enumeration_reports_import_attributes() {
        let module = compile_module_src(
            "import data from './data.json' with { type: 'json', mode: 'strict' };
             export * from './style.css' assert { type: 'css' };",
        );
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert_eq!(stator_module_get_request_count(module), 2);

            let mut specifier = std::ptr::null();
            let mut specifier_len = 0usize;
            let mut attributes = std::ptr::null();
            let mut attributes_len = 0usize;
            assert!(stator_module_get_request(
                module,
                0,
                &mut specifier,
                &mut specifier_len,
                &mut attributes,
                &mut attributes_len,
            ));
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(
                    specifier as *const u8,
                    specifier_len
                ))
                .unwrap(),
                "./data.json"
            );
            assert_eq!(attributes_len, 2);
            assert!(!attributes.is_null());
            let attrs = std::slice::from_raw_parts(attributes, attributes_len);
            assert_eq!(
                import_attribute_pair(&attrs[0]),
                ("type".into(), "json".into())
            );
            assert_eq!(
                import_attribute_pair(&attrs[1]),
                ("mode".into(), "strict".into())
            );

            assert!(stator_module_get_request(
                module,
                1,
                &mut specifier,
                &mut specifier_len,
                &mut attributes,
                &mut attributes_len,
            ));
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(
                    specifier as *const u8,
                    specifier_len
                ))
                .unwrap(),
                "./style.css"
            );
            assert_eq!(attributes_len, 1);
            let attrs = std::slice::from_raw_parts(attributes, attributes_len);
            assert_eq!(
                import_attribute_pair(&attrs[0]),
                ("type".into(), "css".into())
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_instantiate_reports_missing_resolver() {
        let module = compile_module_src("import './dep.js';");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(std::ptr::null_mut(), module));
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindInternal
            );
            let err = CStr::from_ptr(stator_module_get_error(module))
                .to_str()
                .unwrap();
            assert!(err.contains("module resolver"));
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_instantiate_maps_resolver_type_error() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './dep.js';");
        let mut data = TestGraphResolverData {
            modules: HashMap::new(),
            status: StatorResolveStatus::StatorResolveStatusTypeError,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./dep.js"]);
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindType
            );
            stator_module_free(root);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_accepts_resolved_dependency() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './dep.js';");
        let dep = compile_module_src("export const value = 1;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                Some(test_graph_resolver_free_user_data),
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./dep.js"]);
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            assert_eq!(
                stator_module_get_status(dep),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
        assert_eq!(data.cleanup_calls, 1);
    }

    #[test]
    fn test_module_instantiate_passes_import_attributes_to_resolver() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import data from './data.json' with { type: 'json' };");
        let dep = compile_module_src("export default 1;");
        let mut modules = HashMap::new();
        modules.insert("./data.json".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./data.json"]);
            assert_eq!(data.attributes, vec![vec![("type".into(), "json".into())]]);
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_handles_resolver_cycle() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './dep.js';");
        let dep = compile_module_src("import './root.js';");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        modules.insert("./root.js".to_string(), root);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./dep.js", "./root.js"]);
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            assert_eq!(
                stator_module_get_status(dep),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_static_import_forms_link_graph_only() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src(
            "import def, { named as alias } from './dep.js';
             import * as ns from './ns.js';
             export { alias };
             ns; def;",
        );
        let dep = compile_module_src("export default 1; export const named = 2;");
        let ns = compile_module_src("export const side = 3;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        modules.insert("./ns.js".to_string(), ns);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./dep.js", "./ns.js"]);
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );

            let result = stator_module_evaluate(root, ctx);
            assert!(!result.is_null());
            assert_eq!(stator_value_to_number(result), 1.0);
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindUnknown
            );
            stator_value_destroy(result);
            stator_module_free(root);
            stator_module_free(dep);
            stator_module_free(ns);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_reexport_forms_link_graph() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src(
            "export { foo as bar } from './dep.js';
             export * from './star.js';
             export * as ns from './ns.js';",
        );
        let dep = compile_module_src("export const foo = 1;");
        let star = compile_module_src("export const a = 2;");
        let ns = compile_module_src("export const b = 3;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        modules.insert("./star.js".to_string(), star);
        modules.insert("./ns.js".to_string(), ns);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./dep.js", "./star.js", "./ns.js"]);
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_module_free(star);
            stator_module_free(ns);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_rejects_attribute_mismatch_from_resolver() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import data from './data.json' with { type: 'css' };");
        let dep = compile_module_src("export default 1;");
        let mut modules = HashMap::new();
        modules.insert("./data.json".to_string(), dep);
        let mut data = TestAttributeGateResolverData {
            modules,
            expected: vec![("type".into(), "json".into())],
            calls: Vec::new(),
            attributes: Vec::new(),
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_attribute_gate_resolver_cb),
                &mut data as *mut TestAttributeGateResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./data.json"]);
            assert_eq!(data.attributes, vec![vec![("type".into(), "css".into())]]);
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindType
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap();
            assert!(err.contains("attribute mismatch"));
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_propagates_resolver_returned_compile_error_module() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './bad.js';");
        let bad = compile_module_src("export {");
        let mut modules = HashMap::new();
        modules.insert("./bad.js".to_string(), bad);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./bad.js"]);
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindSyntax
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap();
            assert!(err.contains("SyntaxError"));
            stator_module_free(root);
            stator_module_free(bad);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_evaluate_cyclic_import_bindings_fail_closed_with_reference_error() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { b } from './b.js'; export const a = b + 1;");
        let dep = compile_module_src("import { a } from './a.js'; export const b = 1;");
        let mut modules = HashMap::new();
        modules.insert("./b.js".to_string(), dep);
        modules.insert("./a.js".to_string(), root);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(data.calls, vec!["./b.js", "./a.js"]);
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            let result = stator_module_evaluate(root, ctx);
            assert!(result.is_null());
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindReference
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap();
            assert!(err.contains("cyclic module"));
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    fn run_missing_export_case(root_src: &str, dep_src: &str, expected_substrings: &[&str]) {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src(root_src);
        let dep = compile_module_src(dep_src);
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindSyntax
            );
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap()
                .to_string();
            for needle in expected_substrings {
                assert!(
                    err.contains(needle),
                    "expected error {err:?} to contain {needle:?}"
                );
            }
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_named_import_missing_fails_with_syntax_error() {
        run_missing_export_case(
            "import { missing } from './dep.js';",
            "export const present = 1;",
            &["./dep.js", "missing"],
        );
    }

    #[test]
    fn test_module_instantiate_default_import_missing_fails_with_syntax_error() {
        run_missing_export_case(
            "import x from './dep.js';",
            "export const named = 1;",
            &["./dep.js", "default"],
        );
    }

    #[test]
    fn test_module_instantiate_named_reexport_missing_fails_with_syntax_error() {
        run_missing_export_case(
            "export { missing } from './dep.js';",
            "export const present = 1;",
            &["./dep.js", "missing"],
        );
    }

    #[test]
    fn test_module_instantiate_default_reexport_missing_fails_with_syntax_error() {
        run_missing_export_case(
            "export { default as renamed } from './dep.js';",
            "export const named = 1;",
            &["./dep.js", "default"],
        );
    }

    #[test]
    fn test_module_instantiate_namespace_import_succeeds_without_specific_names() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import * as ns from './dep.js'; ns;");
        let dep = compile_module_src("export const x = 1;");
        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });
        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_named_import_via_star_reexport_succeeds() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { deep } from './bridge.js'; deep;");
        let bridge = compile_module_src("export * from './leaf.js';");
        let leaf = compile_module_src("export const deep = 1;");
        let mut modules = HashMap::new();
        modules.insert("./bridge.js".to_string(), bridge);
        modules.insert("./leaf.js".to_string(), leaf);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });
        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(bridge);
            stator_module_free(leaf);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_named_import_default_via_star_reexport_fails() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import { x } from './bridge.js';");
        let bridge = compile_module_src("export * from './leaf.js';");
        // `default` is not propagated through bare star re-exports.
        let leaf = compile_module_src("export default 1;");
        let mut modules = HashMap::new();
        modules.insert("./bridge.js".to_string(), bridge);
        modules.insert("./leaf.js".to_string(), leaf);
        let mut data = TestGraphResolverData {
            modules,
            status: StatorResolveStatus::StatorResolveStatusOk,
            calls: Vec::new(),
            attributes: Vec::new(),
            cleanup_calls: 0,
        };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_graph_resolver_cb),
                &mut data as *mut TestGraphResolverData as *mut c_void,
                None,
            )
        });
        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap()
                .to_string();
            assert!(err.contains("./bridge.js"), "{err}");
            assert!(err.contains("'x'"), "{err}");
            stator_module_free(root);
            stator_module_free(bridge);
            stator_module_free(leaf);
            stator_context_destroy(ctx);
        }
    }

    struct TestErrorResolverData {
        status: StatorResolveStatus,
        detail: Option<&'static str>,
    }

    unsafe extern "C" fn test_error_resolver_cb(
        _ctx: *mut StatorContext,
        user_data: *mut c_void,
        _referrer: *const StatorModule,
        _origin: *const StatorModuleOrigin,
        _specifier: *const c_char,
        _specifier_len: usize,
        _attributes: *const StatorImportAttribute,
        _attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        // SAFETY: tests pass a valid resolver data pointer.
        let data = unsafe { &*(user_data as *const TestErrorResolverData) };
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_module = std::ptr::null_mut() };
        }
        if !out_error.is_null() {
            let err = if let Some(detail) = data.detail {
                let bytes = detail.as_bytes();
                // SAFETY: `bytes` are valid for the duration of this call.
                unsafe { stator_string_new(bytes.as_ptr() as *const c_char, bytes.len()) }
            } else {
                std::ptr::null_mut()
            };
            // SAFETY: out pointer is valid for one write in these tests.
            unsafe { *out_error = err };
        }
        data.status
    }

    fn run_resolver_failure_case(
        status: StatorResolveStatus,
        detail: Option<&'static str>,
        expected_kind: StatorMessageKind,
        expected_substrings: &[&str],
    ) {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './missing.js';");
        let mut data = TestErrorResolverData { status, detail };
        // SAFETY: callback and user data remain live for this test.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_error_resolver_cb),
                &mut data as *mut TestErrorResolverData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(stator_module_error_kind(root), expected_kind);
            let err = CStr::from_ptr(stator_module_get_error(root))
                .to_str()
                .unwrap()
                .to_string();
            for needle in expected_substrings {
                assert!(
                    err.contains(needle),
                    "expected error {err:?} to contain {needle:?}"
                );
            }
            stator_module_free(root);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_instantiate_not_found_maps_to_reference_error() {
        run_resolver_failure_case(
            StatorResolveStatus::StatorResolveStatusNotFound,
            None,
            StatorMessageKind::StatorMessageKindReference,
            &["./missing.js", "not found"],
        );
    }

    #[test]
    fn test_module_instantiate_not_found_propagates_resolver_detail() {
        run_resolver_failure_case(
            StatorResolveStatus::StatorResolveStatusNotFound,
            Some("file:///app/missing.js (404)"),
            StatorMessageKind::StatorMessageKindReference,
            &["./missing.js", "file:///app/missing.js", "404"],
        );
    }

    #[test]
    fn test_module_instantiate_network_error_maps_to_type_error() {
        run_resolver_failure_case(
            StatorResolveStatus::StatorResolveStatusNetworkError,
            Some("connection reset"),
            StatorMessageKind::StatorMessageKindType,
            &["./missing.js", "fetch", "connection reset"],
        );
    }

    #[test]
    fn test_module_instantiate_type_error_maps_to_type_error() {
        run_resolver_failure_case(
            StatorResolveStatus::StatorResolveStatusTypeError,
            Some("invalid 'type' attribute"),
            StatorMessageKind::StatorMessageKindType,
            &["./missing.js", "rejected", "invalid 'type' attribute"],
        );
    }

    #[test]
    fn test_module_instantiate_distinct_kinds_per_status() {
        // Sanity check that the three concrete failure statuses produce
        // distinct, typed error kinds rather than collapsing to one.
        let cases: Vec<(StatorResolveStatus, StatorMessageKind)> = vec![
            (
                StatorResolveStatus::StatorResolveStatusNotFound,
                StatorMessageKind::StatorMessageKindReference,
            ),
            (
                StatorResolveStatus::StatorResolveStatusNetworkError,
                StatorMessageKind::StatorMessageKindType,
            ),
            (
                StatorResolveStatus::StatorResolveStatusTypeError,
                StatorMessageKind::StatorMessageKindType,
            ),
        ];
        let mut seen_kinds: Vec<StatorMessageKind> = Vec::new();
        for (status, expected_kind) in cases {
            let iso = IsolateGuard::new();
            // SAFETY: `iso` is valid.
            let ctx = unsafe { stator_context_new(iso.as_ptr()) };
            let root = compile_module_src("import './x.js';");
            let mut data = TestErrorResolverData {
                status,
                detail: None,
            };
            // SAFETY: callback and user data are live for this test scope.
            assert!(unsafe {
                stator_context_set_module_resolver(
                    ctx,
                    Some(test_error_resolver_cb),
                    &mut data as *mut TestErrorResolverData as *mut c_void,
                    None,
                )
            });
            // SAFETY: pointers are valid.
            unsafe {
                assert!(!stator_module_instantiate(ctx, root));
                let kind = stator_module_error_kind(root);
                assert_eq!(kind, expected_kind);
                seen_kinds.push(kind);
                stator_module_free(root);
                stator_context_destroy(ctx);
            }
        }
        assert!(seen_kinds.contains(&StatorMessageKind::StatorMessageKindReference));
        assert!(seen_kinds.contains(&StatorMessageKind::StatorMessageKindType));
    }

    #[test]
    fn test_stator_string_new_data_len_roundtrip() {
        let payload = b"hello\0world";
        // SAFETY: `payload` is a live byte slice.
        let s = unsafe { stator_string_new(payload.as_ptr() as *const c_char, payload.len()) };
        assert!(!s.is_null());
        // SAFETY: `s` is live and was just allocated.
        unsafe {
            assert_eq!(stator_string_len(s), payload.len());
            let data = stator_string_data(s);
            assert!(!data.is_null());
            let slice = std::slice::from_raw_parts(data as *const u8, stator_string_len(s));
            assert_eq!(slice, payload);
            stator_string_free(s);
        }
    }

    #[test]
    fn test_stator_string_new_empty_returns_handle() {
        // SAFETY: zero-length payload with null pointer is permitted.
        let s = unsafe { stator_string_new(std::ptr::null(), 0) };
        assert!(!s.is_null());
        // SAFETY: `s` is live.
        unsafe {
            assert_eq!(stator_string_len(s), 0);
            stator_string_free(s);
        }
    }

    #[test]
    fn test_stator_string_new_rejects_null_with_length() {
        // SAFETY: null pointer with non-zero length is invalid by contract.
        let s = unsafe { stator_string_new(std::ptr::null(), 4) };
        assert!(s.is_null());
    }

    #[test]
    fn test_stator_string_free_accepts_null() {
        // SAFETY: passing null is a documented no-op.
        unsafe { stator_string_free(std::ptr::null_mut()) };
    }

    #[test]
    fn test_module_instantiate_resolver_returns_ok_with_null_module_is_reference_error() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './x.js';");
        // Resolver returns Ok but writes a null module — engine treats this
        // as NotFound rather than panicking.
        let mut data = TestErrorResolverData {
            status: StatorResolveStatus::StatorResolveStatusOk,
            detail: None,
        };
        // SAFETY: callback and user data are live for this test scope.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_error_resolver_cb),
                &mut data as *mut TestErrorResolverData as *mut c_void,
                None,
            )
        });
        // SAFETY: pointers are valid.
        unsafe {
            assert!(!stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_error_kind(root),
                StatorMessageKind::StatorMessageKindReference
            );
            stator_module_free(root);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_module_compile_syntax_error_classified() {
        let module = compile_module_src("export {");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert!(!stator_module_get_error(module).is_null());
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindSyntax
            );
            assert_eq!(stator_module_bytecode_count(module), 0);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_compile_null_source_classified_internal() {
        // SAFETY: null source pointer is handled as a compile error record.
        let module = unsafe { stator_module_compile(std::ptr::null_mut(), std::ptr::null(), 0) };
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert!(!stator_module_get_error(module).is_null());
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindInternal
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_compile_defaults_to_javascript_type() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert_eq!(
                stator_module_get_type(module),
                StatorModuleType::StatorModuleTypeJavaScript
            );
            assert!(stator_module_get_error(module).is_null());
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_compile_typed_records_unsupported_source_type() {
        let source = b"{\"answer\":42}";
        // SAFETY: null ctx is permitted; source bytes are valid UTF-8.
        let module = unsafe {
            stator_module_compile_typed(
                std::ptr::null_mut(),
                source.as_ptr() as *const c_char,
                source.len(),
                StatorModuleType::StatorModuleTypeJson,
            )
        };
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            assert_eq!(
                stator_module_get_type(module),
                StatorModuleType::StatorModuleTypeJson
            );
            assert!(!stator_module_get_error(module).is_null());
            assert_eq!(
                stator_module_error_kind(module),
                StatorMessageKind::StatorMessageKindInternal
            );
            assert_eq!(
                stator_module_get_status(module),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_origin_defaults_and_roundtrip() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());

        // SAFETY: `module` is valid.
        unsafe {
            assert!(stator_module_get_resource_name(module).is_null());
            assert_eq!(stator_module_get_line_offset(module), 0);
            assert_eq!(stator_module_get_column_offset(module), 0);
        }

        let name = c"https://example.test/foo.mjs";
        // SAFETY: `module` is valid; `name` is a valid C string.
        unsafe {
            stator_module_set_origin(module, name.as_ptr(), 3, 5);
            let got = stator_module_get_resource_name(module);
            assert!(!got.is_null());
            let got_str = CStr::from_ptr(got).to_str().unwrap();
            assert_eq!(got_str, "https://example.test/foo.mjs");
            assert_eq!(stator_module_get_line_offset(module), 3);
            assert_eq!(stator_module_get_column_offset(module), 5);

            stator_module_set_origin(module, std::ptr::null(), 0, 0);
            assert!(stator_module_get_resource_name(module).is_null());
            assert_eq!(stator_module_get_line_offset(module), 0);
            assert_eq!(stator_module_get_column_offset(module), 0);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_origin_metadata_defaults_are_unset() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());
        // SAFETY: `module` is non-null and live.
        unsafe {
            let mut len: usize = 99;
            assert!(stator_module_get_base_url(module, &mut len).is_null());
            assert_eq!(len, 0);
            len = 99;
            assert!(stator_module_get_integrity_metadata(module, &mut len).is_null());
            assert_eq!(len, 0);
            assert_eq!(
                stator_module_get_credentials_mode(module),
                StatorCredentialsMode::StatorCredentialsModeDefault
            );
            assert_eq!(
                stator_module_get_referrer_policy(module),
                StatorReferrerPolicy::StatorReferrerPolicyDefault
            );
            assert_eq!(
                stator_module_get_parser_metadata(module),
                StatorParserMetadata::StatorParserMetadataNotParserInserted
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_set_origin_metadata_copies_strings_and_survives_caller_mutation() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());
        // Allocate caller-owned buffers we will later mutate / drop.
        let mut base_url = b"https://example.test/app/main.mjs".to_vec();
        let mut integrity = b"sha384-abcdef".to_vec();

        // SAFETY: `module` is non-null and live; buffers are valid.
        let ok = unsafe {
            stator_module_set_origin_metadata(
                module,
                base_url.as_ptr() as *const c_char,
                base_url.len(),
                StatorCredentialsMode::StatorCredentialsModeInclude,
                integrity.as_ptr() as *const c_char,
                integrity.len(),
                StatorReferrerPolicy::StatorReferrerPolicyStrictOriginWhenCrossOrigin,
                StatorParserMetadata::StatorParserMetadataParserInserted,
            )
        };
        assert!(ok);

        // Mutate then drop the caller buffers — engine must own its own copies.
        base_url.iter_mut().for_each(|b| *b = b'!');
        integrity.iter_mut().for_each(|b| *b = b'?');
        drop(base_url);
        drop(integrity);

        // SAFETY: `module` is non-null and live.
        unsafe {
            let mut len: usize = 0;
            let ptr = stator_module_get_base_url(module, &mut len);
            assert!(!ptr.is_null());
            let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
            assert_eq!(bytes, b"https://example.test/app/main.mjs");

            let mut len2: usize = 0;
            let ptr2 = stator_module_get_integrity_metadata(module, &mut len2);
            assert!(!ptr2.is_null());
            let bytes2 = std::slice::from_raw_parts(ptr2 as *const u8, len2);
            assert_eq!(bytes2, b"sha384-abcdef");

            assert_eq!(
                stator_module_get_credentials_mode(module),
                StatorCredentialsMode::StatorCredentialsModeInclude
            );
            assert_eq!(
                stator_module_get_referrer_policy(module),
                StatorReferrerPolicy::StatorReferrerPolicyStrictOriginWhenCrossOrigin
            );
            assert_eq!(
                stator_module_get_parser_metadata(module),
                StatorParserMetadata::StatorParserMetadataParserInserted
            );
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_set_origin_metadata_clears_with_null_zero_length() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());
        let url = b"https://example.test/".to_vec();
        // SAFETY: `module` is non-null and live; `url` is valid.
        unsafe {
            assert!(stator_module_set_origin_metadata(
                module,
                url.as_ptr() as *const c_char,
                url.len(),
                StatorCredentialsMode::StatorCredentialsModeOmit,
                std::ptr::null(),
                0,
                StatorReferrerPolicy::StatorReferrerPolicyNoReferrer,
                StatorParserMetadata::StatorParserMetadataNotParserInserted,
            ));
            assert!(!stator_module_get_base_url(module, std::ptr::null_mut()).is_null());

            // Now clear both string fields by passing null with length 0.
            assert!(stator_module_set_origin_metadata(
                module,
                std::ptr::null(),
                0,
                StatorCredentialsMode::StatorCredentialsModeDefault,
                std::ptr::null(),
                0,
                StatorReferrerPolicy::StatorReferrerPolicyDefault,
                StatorParserMetadata::StatorParserMetadataNotParserInserted,
            ));
            let mut len: usize = 1;
            assert!(stator_module_get_base_url(module, &mut len).is_null());
            assert_eq!(len, 0);
            len = 1;
            assert!(stator_module_get_integrity_metadata(module, &mut len).is_null());
            assert_eq!(len, 0);
            stator_module_free(module);
        }
    }

    #[test]
    fn test_module_set_origin_metadata_rejects_null_pointer_with_nonzero_length() {
        let module = compile_module_src("export const x = 1;");
        assert!(!module.is_null());
        let url = b"https://example.test/".to_vec();
        // SAFETY: `module` is non-null and live.
        unsafe {
            // Seed valid metadata so we can verify it stays unchanged.
            assert!(stator_module_set_origin_metadata(
                module,
                url.as_ptr() as *const c_char,
                url.len(),
                StatorCredentialsMode::StatorCredentialsModeSameOrigin,
                std::ptr::null(),
                0,
                StatorReferrerPolicy::StatorReferrerPolicyOrigin,
                StatorParserMetadata::StatorParserMetadataParserInserted,
            ));

            // Invalid: null base_url with nonzero length.
            assert!(!stator_module_set_origin_metadata(
                module,
                std::ptr::null(),
                7,
                StatorCredentialsMode::StatorCredentialsModeOmit,
                std::ptr::null(),
                0,
                StatorReferrerPolicy::StatorReferrerPolicyNoReferrer,
                StatorParserMetadata::StatorParserMetadataNotParserInserted,
            ));
            // Invalid: null integrity with nonzero length.
            assert!(!stator_module_set_origin_metadata(
                module,
                url.as_ptr() as *const c_char,
                url.len(),
                StatorCredentialsMode::StatorCredentialsModeOmit,
                std::ptr::null(),
                3,
                StatorReferrerPolicy::StatorReferrerPolicyNoReferrer,
                StatorParserMetadata::StatorParserMetadataNotParserInserted,
            ));

            // Pre-existing metadata must still be intact.
            let mut len = 0usize;
            let ptr = stator_module_get_base_url(module, &mut len);
            assert!(!ptr.is_null());
            let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
            assert_eq!(bytes, b"https://example.test/");
            assert_eq!(
                stator_module_get_credentials_mode(module),
                StatorCredentialsMode::StatorCredentialsModeSameOrigin
            );
            assert_eq!(
                stator_module_get_referrer_policy(module),
                StatorReferrerPolicy::StatorReferrerPolicyOrigin
            );
            assert_eq!(
                stator_module_get_parser_metadata(module),
                StatorParserMetadata::StatorParserMetadataParserInserted
            );

            stator_module_free(module);
        }
    }

    struct TestOriginCaptureData {
        modules: HashMap<String, *mut StatorModule>,
        captured_base_url: Option<Vec<u8>>,
        captured_integrity: Option<Vec<u8>>,
        captured_credentials_mode: StatorCredentialsMode,
        captured_referrer_policy: StatorReferrerPolicy,
        captured_parser_metadata: StatorParserMetadata,
        captured_origin_was_null: bool,
    }

    unsafe extern "C" fn test_origin_capture_resolver_cb(
        _ctx: *mut StatorContext,
        user_data: *mut c_void,
        _referrer: *const StatorModule,
        origin: *const StatorModuleOrigin,
        specifier: *const c_char,
        specifier_len: usize,
        _attributes: *const StatorImportAttribute,
        _attributes_len: usize,
        out_module: *mut *mut StatorModule,
        out_error: *mut *mut StatorString,
    ) -> StatorResolveStatus {
        // SAFETY: tests pass a valid resolver data pointer.
        let data = unsafe { &mut *(user_data as *mut TestOriginCaptureData) };
        data.captured_origin_was_null = origin.is_null();
        if !origin.is_null() {
            // SAFETY: callback contract guarantees `origin` is valid for this call.
            let view = unsafe { &*origin };
            data.captured_base_url = if view.base_url.is_null() {
                None
            } else {
                // SAFETY: contract guarantees `base_url_len` valid bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts(view.base_url as *const u8, view.base_url_len)
                };
                Some(bytes.to_vec())
            };
            data.captured_integrity = if view.integrity_metadata.is_null() {
                None
            } else {
                // SAFETY: contract guarantees `integrity_metadata_len` valid bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        view.integrity_metadata as *const u8,
                        view.integrity_metadata_len,
                    )
                };
                Some(bytes.to_vec())
            };
            data.captured_credentials_mode = view.credentials_mode;
            data.captured_referrer_policy = view.referrer_policy;
            data.captured_parser_metadata = view.parser_metadata;
        }
        // SAFETY: callback contract supplies a valid specifier byte slice.
        let specifier_bytes =
            unsafe { std::slice::from_raw_parts(specifier as *const u8, specifier_len) };
        let specifier = std::str::from_utf8(specifier_bytes).unwrap().to_string();
        if !out_error.is_null() {
            // SAFETY: out pointer is valid for one write in this test.
            unsafe { *out_error = std::ptr::null_mut() };
        }
        let module = data
            .modules
            .get(&specifier)
            .copied()
            .unwrap_or(std::ptr::null_mut());
        if !out_module.is_null() {
            // SAFETY: out pointer is valid for one write in this test.
            unsafe { *out_module = module };
        }
        if module.is_null() {
            StatorResolveStatus::StatorResolveStatusNotFound
        } else {
            StatorResolveStatus::StatorResolveStatusOk
        }
    }

    fn fresh_origin_capture_data(
        modules: HashMap<String, *mut StatorModule>,
    ) -> TestOriginCaptureData {
        TestOriginCaptureData {
            modules,
            captured_base_url: None,
            captured_integrity: None,
            captured_credentials_mode: StatorCredentialsMode::StatorCredentialsModeDefault,
            captured_referrer_policy: StatorReferrerPolicy::StatorReferrerPolicyDefault,
            captured_parser_metadata: StatorParserMetadata::StatorParserMetadataNotParserInserted,
            captured_origin_was_null: true,
        }
    }

    #[test]
    fn test_module_resolver_receives_origin_metadata_from_referrer() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './dep.js';");
        let dep = compile_module_src("export const value = 1;");

        let base_url = b"https://example.test/app/main.mjs".to_vec();
        let integrity = b"sha256-deadbeef".to_vec();
        // SAFETY: `root` is valid; buffers are valid.
        unsafe {
            assert!(stator_module_set_origin_metadata(
                root,
                base_url.as_ptr() as *const c_char,
                base_url.len(),
                StatorCredentialsMode::StatorCredentialsModeSameOrigin,
                integrity.as_ptr() as *const c_char,
                integrity.len(),
                StatorReferrerPolicy::StatorReferrerPolicyStrictOrigin,
                StatorParserMetadata::StatorParserMetadataParserInserted,
            ));
        }

        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = fresh_origin_capture_data(modules);

        // SAFETY: callback and user data live for this test scope.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_origin_capture_resolver_cb),
                &mut data as *mut TestOriginCaptureData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            assert_eq!(
                stator_module_get_status(root),
                StatorModuleStatus::StatorModuleStatusLinked
            );
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }

        assert!(!data.captured_origin_was_null);
        assert_eq!(
            data.captured_base_url.as_deref(),
            Some(b"https://example.test/app/main.mjs".as_ref())
        );
        assert_eq!(
            data.captured_integrity.as_deref(),
            Some(b"sha256-deadbeef".as_ref())
        );
        assert_eq!(
            data.captured_credentials_mode,
            StatorCredentialsMode::StatorCredentialsModeSameOrigin
        );
        assert_eq!(
            data.captured_referrer_policy,
            StatorReferrerPolicy::StatorReferrerPolicyStrictOrigin
        );
        assert_eq!(
            data.captured_parser_metadata,
            StatorParserMetadata::StatorParserMetadataParserInserted
        );
    }

    #[test]
    fn test_module_resolver_receives_default_origin_when_metadata_unset() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let root = compile_module_src("import './dep.js';");
        let dep = compile_module_src("export const value = 1;");

        let mut modules = HashMap::new();
        modules.insert("./dep.js".to_string(), dep);
        let mut data = fresh_origin_capture_data(modules);

        // SAFETY: callback and user data live for this test scope.
        assert!(unsafe {
            stator_context_set_module_resolver(
                ctx,
                Some(test_origin_capture_resolver_cb),
                &mut data as *mut TestOriginCaptureData as *mut c_void,
                None,
            )
        });

        // SAFETY: pointers are non-null and live.
        unsafe {
            assert!(stator_module_instantiate(ctx, root));
            stator_module_free(root);
            stator_module_free(dep);
            stator_context_destroy(ctx);
        }

        assert!(!data.captured_origin_was_null);
        assert!(data.captured_base_url.is_none());
        assert!(data.captured_integrity.is_none());
        assert_eq!(
            data.captured_credentials_mode,
            StatorCredentialsMode::StatorCredentialsModeDefault
        );
        assert_eq!(
            data.captured_referrer_policy,
            StatorReferrerPolicy::StatorReferrerPolicyDefault
        );
        assert_eq!(
            data.captured_parser_metadata,
            StatorParserMetadata::StatorParserMetadataNotParserInserted
        );
    }

    #[test]
    fn test_module_null_accessors_are_safe() {
        // SAFETY: passing null is documented as supported by every accessor.
        unsafe {
            assert!(stator_module_get_error(std::ptr::null()).is_null());
            assert_eq!(
                stator_module_error_kind(std::ptr::null()),
                StatorMessageKind::StatorMessageKindUnknown
            );
            assert_eq!(
                stator_module_get_status(std::ptr::null()),
                StatorModuleStatus::StatorModuleStatusErrored
            );
            assert_eq!(
                stator_module_get_type(std::ptr::null()),
                StatorModuleType::StatorModuleTypeJavaScript
            );
            stator_module_set_origin(std::ptr::null_mut(), std::ptr::null(), 1, 2);
            assert!(stator_module_get_resource_name(std::ptr::null()).is_null());
            assert_eq!(stator_module_get_line_offset(std::ptr::null()), 0);
            assert_eq!(stator_module_get_column_offset(std::ptr::null()), 0);
            assert_eq!(stator_module_bytecode_count(std::ptr::null()), 0);
            assert!(!stator_module_is_async(std::ptr::null()));
            assert!(!stator_module_has_dependencies(std::ptr::null()));
            assert_eq!(stator_module_get_request_count(std::ptr::null()), 0);
            assert!(!stator_module_get_request(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ));
            assert!(!stator_module_instantiate(
                std::ptr::null_mut(),
                std::ptr::null_mut()
            ));
            assert!(stator_module_evaluate(std::ptr::null_mut(), std::ptr::null_mut()).is_null());
            assert!(!stator_module_set_origin_metadata(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                StatorCredentialsMode::StatorCredentialsModeDefault,
                std::ptr::null(),
                0,
                StatorReferrerPolicy::StatorReferrerPolicyDefault,
                StatorParserMetadata::StatorParserMetadataNotParserInserted,
            ));
            let mut len = 99usize;
            assert!(stator_module_get_base_url(std::ptr::null(), &mut len).is_null());
            assert_eq!(len, 0);
            len = 99;
            assert!(stator_module_get_integrity_metadata(std::ptr::null(), &mut len).is_null());
            assert_eq!(len, 0);
            assert!(stator_module_get_base_url(std::ptr::null(), std::ptr::null_mut()).is_null());
            assert!(
                stator_module_get_integrity_metadata(std::ptr::null(), std::ptr::null_mut())
                    .is_null()
            );
            assert_eq!(
                stator_module_get_credentials_mode(std::ptr::null()),
                StatorCredentialsMode::StatorCredentialsModeDefault
            );
            assert_eq!(
                stator_module_get_referrer_policy(std::ptr::null()),
                StatorReferrerPolicy::StatorReferrerPolicyDefault
            );
            assert_eq!(
                stator_module_get_parser_metadata(std::ptr::null()),
                StatorParserMetadata::StatorParserMetadataNotParserInserted
            );
            stator_module_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_tiering_stats_null_safety() {
        // SAFETY: null output is documented as a no-op.
        unsafe { stator_isolate_get_tiering_stats(std::ptr::null(), std::ptr::null_mut()) };
        // SAFETY: null isolate is accepted and `stats` is valid for writes.
        let mut stats = unsafe { std::mem::zeroed::<StatorTieringStats>() };
        unsafe { stator_isolate_get_tiering_stats(std::ptr::null(), &mut stats) };
        // SAFETY: null isolate is accepted for resetting process/thread diagnostics.
        unsafe { stator_isolate_reset_tiering_stats(std::ptr::null_mut()) };
    }

    #[test]
    fn test_isolate_jit_disabled_controls_are_null_safe() {
        // SAFETY: null isolate is documented as a no-op.
        unsafe { stator_isolate_set_jit_disabled(std::ptr::null_mut(), true) };
        // SAFETY: null isolate returns false.
        assert!(!unsafe { stator_isolate_jit_disabled(std::ptr::null()) });

        let isolate = stator_isolate_create();
        assert!(!isolate.is_null());
        // SAFETY: `isolate` is non-null and live.
        assert!(!unsafe { stator_isolate_jit_disabled(isolate) });
        // SAFETY: `isolate` is non-null and live.
        unsafe { stator_isolate_set_jit_disabled(isolate, true) };
        // SAFETY: `isolate` is non-null and live.
        assert!(unsafe { stator_isolate_jit_disabled(isolate) });
        // SAFETY: `isolate` is non-null and live.
        unsafe { stator_isolate_destroy(isolate) };
    }

    #[test]
    fn test_script_tier_status_null_safety_and_success() {
        let mut status = unsafe { std::mem::zeroed::<StatorScriptTierStatus>() };
        // SAFETY: null script is accepted and returns false.
        assert!(!unsafe { stator_script_get_tier_status(std::ptr::null(), &mut status) });

        let script = compile_src("function f(){ return 1; } f();");
        // SAFETY: `script` is non-null and live.
        assert!(unsafe { stator_script_get_tier_status(script, &mut status) });
        assert!(status.bytecode_count > 0);
        // SAFETY: null output is accepted and returns false.
        assert!(!unsafe { stator_script_get_tier_status(script, std::ptr::null_mut()) });
        // SAFETY: `script` is non-null and live.
        unsafe { stator_script_free(script) };
    }

    #[test]
    fn test_script_force_maglev_compile_null_safe() {
        // SAFETY: null script is accepted and returns false.
        assert!(!unsafe { stator_script_force_maglev_compile(std::ptr::null_mut()) });

        let script = compile_src("function f(){ return 1; } f();");
        // SAFETY: `script` is non-null and live. Unsupported platforms return false.
        let forced = unsafe { stator_script_force_maglev_compile(script) };
        #[cfg(not(any(
            stator_maglev_jit_x86_64,
            all(target_arch = "x86_64", any(unix, windows))
        )))]
        assert!(!forced);
        #[cfg(any(
            stator_maglev_jit_x86_64,
            all(target_arch = "x86_64", any(unix, windows))
        ))]
        {
            let _ = forced;
        }
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
    fn test_function_callback_info_get_this_null_is_safe() {
        let this = unsafe { stator_function_callback_info_get_this(std::ptr::null()) };
        assert!(this.is_null());
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
    fn test_function_template_pending_exception_fails_script() {
        use std::sync::atomic::{AtomicPtr, Ordering};

        static THROWN_EXCEPTION: AtomicPtr<StatorValue> = AtomicPtr::new(std::ptr::null_mut());

        unsafe extern "C" fn throw_from_template(
            info: *const StatorFunctionCallbackInfo,
        ) -> *mut StatorValue {
            // SAFETY: `info` is valid for this callback invocation.
            let isolate = unsafe { stator_function_callback_info_get_isolate(info) };
            // SAFETY: `isolate` is valid and the string bytes are copied.
            let exception =
                unsafe { stator_value_new_string(isolate, c"template failure".as_ptr(), 16) };
            THROWN_EXCEPTION.store(exception, Ordering::SeqCst);
            // SAFETY: `isolate` and `exception` are valid for the pending-exception window.
            unsafe { stator_isolate_throw_exception(isolate, exception) };
            std::ptr::null_mut()
        }

        THROWN_EXCEPTION.store(std::ptr::null_mut(), Ordering::SeqCst);
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), throw_from_template) };
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        unsafe { stator_context_global_set(ctx, c"boom".as_ptr(), fn_val) };
        unsafe { stator_value_destroy(fn_val) };

        let src = b"boom(); 13";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert!(!script.is_null());
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(result.is_null());
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let caught = unsafe { stator_try_catch_exception(tc) };
        assert_eq!(caught, THROWN_EXCEPTION.load(Ordering::SeqCst));

        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(caught);
            stator_script_free(script);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_registered_native_pending_exception_fails_script() {
        use std::sync::atomic::{AtomicPtr, Ordering};

        static THROWN_EXCEPTION: AtomicPtr<StatorValue> = AtomicPtr::new(std::ptr::null_mut());

        unsafe extern "C" fn throw_from_native(
            ctx: *mut StatorContext,
            _args: *const *const StatorValue,
            _argc: i32,
        ) -> *mut StatorValue {
            let isolate = if ctx.is_null() {
                std::ptr::null_mut()
            } else {
                // SAFETY: `ctx` is valid for this callback invocation.
                unsafe { (*ctx)._isolate }
            };
            // SAFETY: `isolate` is valid and the string bytes are copied.
            let exception =
                unsafe { stator_value_new_string(isolate, c"native failure".as_ptr(), 14) };
            THROWN_EXCEPTION.store(exception, Ordering::SeqCst);
            // SAFETY: `isolate` and `exception` are valid for the pending-exception window.
            unsafe { stator_isolate_throw_exception(isolate, exception) };
            std::ptr::null_mut()
        }

        THROWN_EXCEPTION.store(std::ptr::null_mut(), Ordering::SeqCst);
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        unsafe { stator_register_native_function(ctx, c"boom".as_ptr(), throw_from_native) };

        let src = b"boom(); 13";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert!(!script.is_null());
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(result.is_null());
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let caught = unsafe { stator_try_catch_exception(tc) };
        assert_eq!(caught, THROWN_EXCEPTION.load(Ordering::SeqCst));

        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(caught);
            stator_script_free(script);
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
    fn test_script_run_no_result_reports_success() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"1 + 2";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        // SAFETY: `script` and `ctx` are valid.
        assert!(unsafe { stator_script_run_no_result(script, ctx) });

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_script_globals_reflect_on_global_this() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"
            var edgeGlobalReflection = 42;
            edgeGlobalAssignment = edgeGlobalReflection + 1;
            globalThis.edgeGlobalReflection + ':' + globalThis.edgeGlobalAssignment;
        ";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null(), "expected a result");
        // SAFETY: `result` is non-null and live.
        let result_ptr = unsafe { stator_value_as_string(result) };
        // SAFETY: `result_ptr` points to the live result string.
        let actual = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy();
        assert_eq!(actual, "42:43");

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_drain_active_microtask_queue_flushes_promise_reactions() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"
            var edgeStatorMicrotaskValue = 0;
            Promise.resolve(41).then(function(value) {
                edgeStatorMicrotaskValue = value + 1;
            });
            edgeStatorMicrotaskValue;
        ";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null(), "expected an initial result");
        // SAFETY: `result` is non-null and live.
        unsafe { stator_value_destroy(result) };

        let drained = stator_drain_active_microtask_queue();
        assert!(drained > 0, "expected at least one Promise microtask");

        let read_src = b"edgeStatorMicrotaskValue";
        // SAFETY: `ctx` is valid; `read_src` is valid UTF-8.
        let read_script = unsafe {
            stator_script_compile(ctx, read_src.as_ptr() as *const c_char, read_src.len())
        };
        assert!(!read_script.is_null());
        // SAFETY: `read_script` and `ctx` are valid.
        let read_result = unsafe { stator_script_run(read_script, ctx) };
        assert!(!read_result.is_null(), "expected a read result");
        // SAFETY: `read_result` is non-null and live.
        let n = unsafe { stator_value_as_number(read_result) };
        assert_eq!(n, 42.0);

        // SAFETY: all pointers are non-null and live.
        unsafe {
            stator_value_destroy(read_result);
            stator_script_free(read_script);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct SeenPromiseRejectionEvent {
        kind: StatorPromiseRejectionEventKind,
        promise_id: usize,
        reason: String,
    }

    unsafe extern "C" fn collect_promise_rejection_event(
        kind: StatorPromiseRejectionEventKind,
        promise_id: usize,
        reason_utf8: *const c_char,
        reason_len: usize,
        user_data: *mut c_void,
    ) {
        // SAFETY: the test passes a valid mutable Vec pointer as `user_data`.
        let events = unsafe { &mut *(user_data as *mut Vec<SeenPromiseRejectionEvent>) };
        let reason = if reason_utf8.is_null() || reason_len == 0 {
            String::new()
        } else {
            // SAFETY: the callback contract guarantees the byte slice is valid
            // for this call.
            let bytes = unsafe { std::slice::from_raw_parts(reason_utf8 as *const u8, reason_len) };
            String::from_utf8(bytes.to_vec()).expect("reason must be valid UTF-8")
        };
        events.push(SeenPromiseRejectionEvent {
            kind,
            promise_id,
            reason,
        });
    }

    fn drain_promise_rejection_events_for_test() -> Vec<SeenPromiseRejectionEvent> {
        let mut events = Vec::new();
        // SAFETY: the callback and `events` pointer are valid for this call.
        unsafe {
            stator_drain_active_promise_rejection_events(
                collect_promise_rejection_event,
                &mut events as *mut _ as *mut c_void,
            );
        }
        events
    }

    fn run_script_for_promise_rejection_test(
        ctx: *mut StatorContext,
        source: &[u8],
    ) -> *mut StatorValue {
        // SAFETY: `ctx` is valid and `source` is a valid byte slice.
        let script =
            unsafe { stator_script_compile(ctx, source.as_ptr() as *const c_char, source.len()) };
        assert!(!script.is_null());
        // SAFETY: `script` is non-null.
        assert!(unsafe { stator_script_get_error(script) }.is_null());
        // SAFETY: `script` and `ctx` are valid.
        let result = unsafe { stator_script_run(script, ctx) };
        // SAFETY: `script` is non-null and no longer needed.
        unsafe { stator_script_free(script) };
        result
    }

    #[test]
    fn test_drain_active_promise_rejection_events_reports_unhandled_and_handled() {
        // Drain any thread-local events left by an earlier test running on the
        // same worker thread.
        stator_discard_active_promise_rejection_events();

        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };

        let rejected_result = run_script_for_promise_rejection_test(
            ctx,
            b"var edgeRejectedPromise = Promise.reject('edge-reject'); 'queued';",
        );
        assert!(!rejected_result.is_null());
        // SAFETY: `rejected_result` is non-null and live.
        unsafe { stator_value_destroy(rejected_result) };

        let rejected_events = drain_promise_rejection_events_for_test();
        assert_eq!(rejected_events.len(), 1);
        assert_eq!(
            rejected_events[0].kind,
            StatorPromiseRejectionEventKind::StatorPromiseRejectionEventKindRejectedWithNoHandler
        );
        assert_ne!(rejected_events[0].promise_id, 0);
        assert_eq!(rejected_events[0].reason, "edge-reject");

        let handled_result = run_script_for_promise_rejection_test(
            ctx,
            b"edgeRejectedPromise.catch(function(reason) { return reason; }); 'handled';",
        );
        assert!(!handled_result.is_null());
        // SAFETY: `handled_result` is non-null and live.
        unsafe { stator_value_destroy(handled_result) };

        let handled_events = drain_promise_rejection_events_for_test();
        assert_eq!(handled_events.len(), 1);
        assert_eq!(
            handled_events[0].kind,
            StatorPromiseRejectionEventKind::StatorPromiseRejectionEventKindHandlerAddedAfterReject
        );
        assert_eq!(handled_events[0].promise_id, rejected_events[0].promise_id);
        assert_eq!(handled_events[0].reason, "edge-reject");

        // SAFETY: `ctx` is non-null and live.
        unsafe { stator_context_destroy(ctx) };
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

    // ── Wasm host imports (FFI) ───────────────────────────────────────────

    /// FFI host-import callback: integer add through `env.add(i32,i32) -> i32`.
    unsafe extern "C" fn ffi_add_cb(
        _ctx: *mut StatorContext,
        _user_data: *mut c_void,
        args: *const StatorWasmValue,
        args_len: usize,
        results: *mut StatorWasmValue,
        results_len: usize,
    ) -> bool {
        if args_len != 2 || results_len != 1 {
            return false;
        }
        // SAFETY: caller passed args/results slices of the declared sizes.
        let a = unsafe { (*args.add(0)).value.i32_ };
        let b = unsafe { (*args.add(1)).value.i32_ };
        unsafe {
            (*results.add(0)).kind = StatorWasmValueKind::StatorWasmValueKindI32;
            (*results.add(0)).value.i32_ = a.wrapping_add(b);
        }
        true
    }

    unsafe extern "C" fn ffi_trap_cb(
        _ctx: *mut StatorContext,
        _user_data: *mut c_void,
        _args: *const StatorWasmValue,
        _args_len: usize,
        _results: *mut StatorWasmValue,
        _results_len: usize,
    ) -> bool {
        false
    }

    unsafe extern "C" fn ffi_wrong_result_kind_cb(
        _ctx: *mut StatorContext,
        _user_data: *mut c_void,
        _args: *const StatorWasmValue,
        _args_len: usize,
        results: *mut StatorWasmValue,
        results_len: usize,
    ) -> bool {
        if results_len != 1 {
            return false;
        }
        unsafe {
            (*results.add(0)).kind = StatorWasmValueKind::StatorWasmValueKindF64;
            (*results.add(0)).value.f64_ = 42.0;
        }
        true
    }

    /// FFI host import: compile, instantiate with `env.add`, call exported
    /// `call_add(3, 4)` and observe `7`.
    #[test]
    fn test_ffi_wasm_host_import_add() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "add" (func $add (param i32 i32) (result i32)))
                (func (export "call_add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    call $add))
        "#;
        // SAFETY: `iso` is valid; `wat` is valid WAT.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        assert!(!m.is_null());

        let params = [
            StatorWasmValueKind::StatorWasmValueKindI32,
            StatorWasmValueKind::StatorWasmValueKindI32,
        ];
        let results = [StatorWasmValueKind::StatorWasmValueKindI32];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"add".as_ptr(),
            params: params.as_ptr(),
            params_len: params.len(),
            results: results.as_ptr(),
            results_len: results.len(),
            callback: Some(ffi_add_cb),
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: pointers are valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(
            !inst.is_null(),
            "instantiate with valid imports must succeed"
        );

        let name = c"call_add";
        // SAFETY: iso valid.
        let a = unsafe { stator_value_new_number(iso.as_ptr(), 3.0) };
        let b = unsafe { stator_value_new_number(iso.as_ptr(), 4.0) };
        let argv: [*const StatorValue; 2] = [a, b];
        // SAFETY: all pointers valid.
        let res = unsafe {
            stator_wasm_instance_call(inst, iso.as_ptr(), name.as_ptr(), argv.as_ptr(), argv.len())
        };
        assert!(!res.is_null());
        // SAFETY: res valid.
        let n = unsafe { stator_value_as_number(res) };
        assert_eq!(n, 7.0);
        // SAFETY: all pointers valid.
        unsafe {
            stator_value_destroy(res);
            stator_value_destroy(a);
            stator_value_destroy(b);
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    /// Host callback returning `false` traps the Wasm caller and the FFI call
    /// returns null.
    #[test]
    fn test_ffi_wasm_host_import_trap_returns_null() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "bad" (func $bad (result i32)))
                (func (export "go") (result i32) (call $bad)))
        "#;
        // SAFETY: valid input.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        assert!(!m.is_null());
        let results = [StatorWasmValueKind::StatorWasmValueKindI32];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"bad".as_ptr(),
            params: std::ptr::null(),
            params_len: 0,
            results: results.as_ptr(),
            results_len: results.len(),
            callback: Some(ffi_trap_cb),
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(!inst.is_null());
        let name = c"go";
        // SAFETY: valid.
        let res = unsafe {
            stator_wasm_instance_call(inst, iso.as_ptr(), name.as_ptr(), std::ptr::null(), 0)
        };
        assert!(res.is_null(), "callback returning false must trap the call");
        // SAFETY: valid pointers.
        unsafe {
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    /// A callback that writes a result kind different from the declared Wasm
    /// import signature traps instead of letting Stator reinterpret the union
    /// payload as the wrong type.
    #[test]
    fn test_ffi_wasm_host_import_wrong_result_kind_traps() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "bad" (func $bad (result i32)))
                (func (export "go") (result i32) (call $bad)))
        "#;
        // SAFETY: valid input.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        assert!(!m.is_null());
        let results = [StatorWasmValueKind::StatorWasmValueKindI32];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"bad".as_ptr(),
            params: std::ptr::null(),
            params_len: 0,
            results: results.as_ptr(),
            results_len: results.len(),
            callback: Some(ffi_wrong_result_kind_cb),
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(!inst.is_null());
        let name = c"go";
        // SAFETY: valid.
        let res = unsafe {
            stator_wasm_instance_call(inst, iso.as_ptr(), name.as_ptr(), std::ptr::null(), 0)
        };
        assert!(res.is_null(), "wrong result kind must trap the call");
        // SAFETY: valid pointers.
        unsafe {
            stator_wasm_instance_destroy(inst);
            stator_wasm_module_destroy(m);
        }
    }

    /// Instantiate fails when a required import is not supplied.
    #[test]
    fn test_ffi_wasm_host_import_missing_returns_null() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "missing" (func (result i32))))
        "#;
        // SAFETY: valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        assert!(!m.is_null());
        // No imports supplied — instantiate must fail.
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), std::ptr::null()) };
        assert!(inst.is_null());
        // SAFETY: valid.
        unsafe { stator_wasm_module_destroy(m) };
    }

    /// Instantiate fails when the supplied import is bound under the wrong
    /// field name.
    #[test]
    fn test_ffi_wasm_host_import_bad_name_returns_null() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "add" (func (param i32 i32) (result i32))))
        "#;
        // SAFETY: valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        let params = [
            StatorWasmValueKind::StatorWasmValueKindI32,
            StatorWasmValueKind::StatorWasmValueKindI32,
        ];
        let results = [StatorWasmValueKind::StatorWasmValueKindI32];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"other".as_ptr(),
            params: params.as_ptr(),
            params_len: params.len(),
            results: results.as_ptr(),
            results_len: results.len(),
            callback: Some(ffi_add_cb),
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(inst.is_null(), "bad field name must fail instantiate");
        // SAFETY: valid.
        unsafe { stator_wasm_module_destroy(m) };
    }

    /// Instantiate fails when the supplied import signature does not match
    /// the module's declaration.
    #[test]
    fn test_ffi_wasm_host_import_bad_signature_returns_null() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "f" (func (param i32 i32) (result i32))))
        "#;
        // SAFETY: valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        let params = [StatorWasmValueKind::StatorWasmValueKindF64];
        let results = [StatorWasmValueKind::StatorWasmValueKindF64];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"f".as_ptr(),
            params: params.as_ptr(),
            params_len: params.len(),
            results: results.as_ptr(),
            results_len: results.len(),
            callback: Some(ffi_add_cb),
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(inst.is_null(), "bad signature must fail instantiate");
        // SAFETY: valid.
        unsafe { stator_wasm_module_destroy(m) };
    }

    /// Null callback in a host import declaration is rejected.
    #[test]
    fn test_ffi_wasm_host_import_null_callback_returns_null() {
        let iso = IsolateGuard::new();
        let wat: &[u8] = br#"
            (module
                (import "env" "add" (func (param i32 i32) (result i32))))
        "#;
        // SAFETY: valid.
        let m = unsafe { stator_wasm_compile(iso.as_ptr(), wat.as_ptr(), wat.len()) };
        let params = [
            StatorWasmValueKind::StatorWasmValueKindI32,
            StatorWasmValueKind::StatorWasmValueKindI32,
        ];
        let results = [StatorWasmValueKind::StatorWasmValueKindI32];
        let host = StatorWasmHostFunc {
            module_name: c"env".as_ptr(),
            field_name: c"add".as_ptr(),
            params: params.as_ptr(),
            params_len: params.len(),
            results: results.as_ptr(),
            results_len: results.len(),
            callback: None,
            user_data: std::ptr::null_mut(),
        };
        let imports = StatorWasmImports {
            funcs: &host,
            funcs_len: 1,
        };
        // SAFETY: valid.
        let inst = unsafe { stator_wasm_instantiate(m, std::ptr::null_mut(), &imports) };
        assert!(inst.is_null());
        // SAFETY: valid.
        unsafe { stator_wasm_module_destroy(m) };
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

    #[test]
    fn test_try_catch_catches_script_run_error_message() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        // SAFETY: `iso` is valid.
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        assert!(!tc.is_null());
        let src = b"var edgeStatorNonFunction = 1; edgeStatorNonFunction();";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        // SAFETY: all pointers are valid and live.
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            assert!(stator_try_catch_has_caught(tc));
            let exception = stator_try_catch_exception(tc);
            assert!(!exception.is_null());
            let len = stator_value_to_string_utf8(exception, std::ptr::null_mut(), 0);
            assert!(len > 0);
            let mut buffer = vec![0 as c_char; len as usize + 1];
            stator_value_to_string_utf8(exception, buffer.as_mut_ptr(), buffer.len());
            let message = CStr::from_ptr(buffer.as_ptr()).to_string_lossy();
            assert!(
                message.contains("TypeError") && message.contains("callee is not a function"),
                "unexpected message: {message}"
            );
            stator_try_catch_destroy(tc);
            stator_script_free(script);
            stator_context_destroy(ctx);
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

    // ── DOM hardened slice: class identity + V2 interceptors ───────────────

    #[test]
    fn test_dom_object_wrap_class_id_default_zero() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        assert!(!wrap.is_null());
        assert_eq!(unsafe { stator_dom_object_wrap_get_class_id(wrap) }, 0);
        assert!(unsafe { stator_dom_object_wrap_get_native_ptr(wrap) }.is_null());
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_object_wrap_class_id_roundtrip() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        unsafe { stator_dom_object_wrap_set_class_id(wrap, 42) };
        assert_eq!(unsafe { stator_dom_object_wrap_get_class_id(wrap) }, 42);
        let sentinel: usize = 0xCAFE_F00D;
        unsafe { stator_dom_object_wrap_set_native_ptr(wrap, sentinel as *mut c_void) };
        assert_eq!(
            unsafe { stator_dom_object_wrap_get_native_ptr(wrap) } as usize,
            sentinel
        );
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_object_wrap_class_id_null_safe() {
        // get_* on null returns the zero value, set_* is a no-op.
        assert_eq!(
            unsafe { stator_dom_object_wrap_get_class_id(std::ptr::null()) },
            0
        );
        assert!(unsafe { stator_dom_object_wrap_get_native_ptr(std::ptr::null()) }.is_null());
        unsafe { stator_dom_object_wrap_set_class_id(std::ptr::null_mut(), 7) };
        unsafe {
            stator_dom_object_wrap_set_native_ptr(std::ptr::null_mut(), 1 as *mut c_void);
        }
    }

    #[test]
    fn test_dom_install_named_handler_rejects_null_inputs() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        // null handler
        let status =
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, std::ptr::null()) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        // empty handler (all-null callbacks)
        let empty = StatorDomNamedHandler {
            getter: None,
            setter: None,
            query: None,
            deleter: None,
            enumerator: None,
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_named_handler(wrap, &empty) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        // null wrap
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_always_handled),
            setter: None,
            query: None,
            deleter: None,
            enumerator: None,
            data: std::ptr::null_mut(),
        };
        let status =
            unsafe { stator_dom_object_wrap_install_named_handler(std::ptr::null_mut(), &handler) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_install_indexed_handler_rejects_null_inputs() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let status =
            unsafe { stator_dom_object_wrap_install_indexed_handler(wrap, std::ptr::null()) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        let empty = StatorDomIndexedHandler {
            getter: None,
            setter: None,
            query: None,
            length: None,
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_indexed_handler(wrap, &empty) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    // Helper callbacks used by the V2 interceptor tests.

    unsafe extern "C" fn named_get_always_handled(
        _name: *const c_char,
        _name_len: usize,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let iso = ACTIVE_ISO.with(|c| c.get());
        let v = unsafe { stator_value_new_number(iso, 17.0) };
        unsafe { *out = v };
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_get_id_only(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        // SAFETY: caller passes a valid UTF-8 range.
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key == "id" {
            let iso = ACTIVE_ISO.with(|c| c.get());
            let v = unsafe { stator_value_new_number(iso, 99.0) };
            unsafe { *out = v };
            StatorStatus::StatorStatusOk
        } else if key == "missing" {
            StatorStatus::StatorStatusFalse
        } else if key == "boom" {
            StatorStatus::StatorStatusException
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn named_query_id(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out_attrs: *mut u32,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key == "id" {
            unsafe { *out_attrs = 0 };
            StatorStatus::StatorStatusOk
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn named_delete_id(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out_deleted: *mut bool,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key == "id" {
            unsafe { *out_deleted = true };
            StatorStatus::StatorStatusOk
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn named_enumerate_ab(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        let a = b"alpha";
        let b = b"beta";
        let s1 = unsafe { stator_dom_name_buffer_push(buf, a.as_ptr() as *const c_char, a.len()) };
        assert_eq!(s1, StatorStatus::StatorStatusOk);
        let s2 = unsafe { stator_dom_name_buffer_push(buf, b.as_ptr() as *const c_char, b.len()) };
        assert_eq!(s2, StatorStatus::StatorStatusOk);
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_get_document_property(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        let iso = ACTIVE_ISO.with(|c| c.get());
        let value = match key {
            "title" => ACTIVE_DOCUMENT_TITLE.with(|title| title.borrow().clone()),
            "URL" | "documentURI" => "https://example.test/stator.html".to_string(),
            "readonlyMissing" => return StatorStatus::StatorStatusFalse,
            _ => return StatorStatus::StatorStatusFalse,
        };
        let bytes = value.as_bytes();
        let v =
            unsafe { stator_value_new_string(iso, bytes.as_ptr() as *const c_char, bytes.len()) };
        unsafe { *out = v };
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_get_document_wrapper_property(
        name: *const c_char,
        name_len: usize,
        data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key != "documentElement" {
            return StatorStatus::StatorStatusFalse;
        }
        let value = unsafe { stator_dom_object_wrap_as_value(data as *mut StatorDomObjectWrap) };
        if value.is_null() {
            return StatorStatus::StatorStatusInvalidArg;
        }
        unsafe { *out = value };
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn capture_dom_this(
        info: *const StatorFunctionCallbackInfo,
    ) -> *mut StatorValue {
        let this = unsafe { stator_function_callback_info_get_this(info) };
        if !this.is_null() {
            let wrap = unsafe { stator_value_as_dom_object_wrap(this) };
            if !wrap.is_null() {
                CAPTURED_DOM_THIS_CLASS_ID.store(
                    unsafe { stator_dom_object_wrap_get_class_id(wrap) },
                    std::sync::atomic::Ordering::SeqCst,
                );
                CAPTURED_DOM_THIS_NATIVE_PTR.store(
                    unsafe { stator_dom_object_wrap_get_native_ptr(wrap) } as usize,
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        }
        let iso = unsafe { stator_function_callback_info_get_isolate(info) };
        unsafe { stator_value_new_number(iso, 1.0) }
    }

    unsafe extern "C" fn named_get_capture_method(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key != "captureThis" {
            return StatorStatus::StatorStatusFalse;
        }
        let iso = ACTIVE_ISO.with(|c| c.get());
        let tmpl = unsafe { stator_function_template_new(iso, capture_dom_this) };
        if tmpl.is_null() {
            return StatorStatus::StatorStatusInvalidArg;
        }
        let value = unsafe { stator_function_template_get_function(tmpl, std::ptr::null_mut()) };
        unsafe { stator_function_template_destroy(tmpl) };
        if value.is_null() {
            return StatorStatus::StatorStatusInvalidArg;
        }
        unsafe { *out = value };
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_enumerate_capture_method(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        let name = b"captureThis";
        unsafe { stator_dom_name_buffer_push(buf, name.as_ptr() as *const c_char, name.len()) }
    }

    unsafe extern "C" fn named_get_element_property(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        let iso = ACTIVE_ISO.with(|c| c.get());
        match key {
            "nodeName" => {
                let value = b"HTML";
                let v = unsafe {
                    stator_value_new_string(iso, value.as_ptr() as *const c_char, value.len())
                };
                unsafe { *out = v };
                StatorStatus::StatorStatusOk
            }
            "nodeType" => {
                let v = unsafe { stator_value_new_number(iso, 1.0) };
                unsafe { *out = v };
                StatorStatus::StatorStatusOk
            }
            _ => StatorStatus::StatorStatusFalse,
        }
    }

    unsafe extern "C" fn named_get_throwing_document_property(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        _out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key != "boom" {
            return StatorStatus::StatorStatusFalse;
        }

        let iso = ACTIVE_ISO.with(|c| c.get());
        let message = b"DOM getter boom";
        let exception = unsafe {
            stator_value_new_string(iso, message.as_ptr() as *const c_char, message.len())
        };
        unsafe { stator_isolate_throw_exception(iso, exception) };
        StatorStatus::StatorStatusException
    }

    unsafe extern "C" fn named_get_status_exception_document_property(
        name: *const c_char,
        name_len: usize,
        _data: *mut c_void,
        _out: *mut *mut StatorValue,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key == "boom" {
            StatorStatus::StatorStatusException
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn named_set_document_property(
        name: *const c_char,
        name_len: usize,
        value: *const StatorValue,
        _data: *mut c_void,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key != "title" {
            return StatorStatus::StatorStatusFalse;
        }
        let ptr = unsafe { stator_value_as_string(value) };
        if ptr.is_null() {
            return StatorStatus::StatorStatusInvalidArg;
        }
        let title = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        ACTIVE_DOCUMENT_TITLE.with(|stored| *stored.borrow_mut() = title);
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_set_throwing_document_property(
        name: *const c_char,
        name_len: usize,
        _value: *const StatorValue,
        _data: *mut c_void,
    ) -> StatorStatus {
        let slice = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
        let key = std::str::from_utf8(slice).unwrap();
        if key == "boom" {
            StatorStatus::StatorStatusException
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn named_enumerate_document(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        for name in [
            b"title".as_slice(),
            b"URL".as_slice(),
            b"documentURI".as_slice(),
            b"readonlyMissing".as_slice(),
        ] {
            let status = unsafe {
                stator_dom_name_buffer_push(buf, name.as_ptr() as *const c_char, name.len())
            };
            if status != StatorStatus::StatorStatusOk {
                return status;
            }
        }
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_enumerate_document_wrapper(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        let name = b"documentElement";
        unsafe { stator_dom_name_buffer_push(buf, name.as_ptr() as *const c_char, name.len()) }
    }

    unsafe extern "C" fn named_enumerate_element(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        for name in [b"nodeName".as_slice(), b"nodeType".as_slice()] {
            let status = unsafe {
                stator_dom_name_buffer_push(buf, name.as_ptr() as *const c_char, name.len())
            };
            if status != StatorStatus::StatorStatusOk {
                return status;
            }
        }
        StatorStatus::StatorStatusOk
    }

    unsafe extern "C" fn named_enumerate_boom(
        buf: *mut StatorDomNameBuffer,
        _data: *mut c_void,
    ) -> StatorStatus {
        let name = b"boom";
        unsafe { stator_dom_name_buffer_push(buf, name.as_ptr() as *const c_char, name.len()) }
    }

    unsafe extern "C" fn indexed_get_zero(
        index: u32,
        _data: *mut c_void,
        out: *mut *mut StatorValue,
    ) -> StatorStatus {
        if index == 0 {
            let iso = ACTIVE_ISO.with(|c| c.get());
            let v = unsafe { stator_value_new_number(iso, 7.0) };
            unsafe { *out = v };
            StatorStatus::StatorStatusOk
        } else {
            StatorStatus::StatorStatusFalse
        }
    }

    unsafe extern "C" fn indexed_length_three(
        _data: *mut c_void,
        out_len: *mut u32,
    ) -> StatorStatus {
        unsafe { *out_len = 3 };
        StatorStatus::StatorStatusOk
    }

    thread_local! {
        static ACTIVE_ISO: std::cell::Cell<*mut StatorIsolate> =
            const { std::cell::Cell::new(std::ptr::null_mut()) };
        static ACTIVE_DOCUMENT_TITLE: RefCell<String> = RefCell::new(String::new());
    }
    static CAPTURED_DOM_THIS_CLASS_ID: std::sync::atomic::AtomicU32 =
        std::sync::atomic::AtomicU32::new(0);
    static CAPTURED_DOM_THIS_NATIVE_PTR: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    #[test]
    fn test_dom_named_handler_missing_vs_handled_vs_exception() {
        use stator_jse::objects::value::JsValue;

        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_id_only),
            setter: None,
            query: None,
            deleter: None,
            enumerator: None,
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) };
        assert_eq!(status, StatorStatus::StatorStatusOk);

        // Handled: returns the interceptor value.
        let v_id = unsafe { (*wrap).inner.get_property("id") };
        match v_id {
            JsValue::Smi(n) => assert_eq!(n, 99),
            other => panic!("expected Smi(99), got {other:?}"),
        }
        // Missing (StatusFalse): falls through to own properties -> Undefined.
        assert!(matches!(
            unsafe { (*wrap).inner.get_property("missing") },
            JsValue::Undefined
        ));
        // Exception (StatusException): direct wrapper access still falls
        // through, but the FFI layer records a pending exception for script
        // accessors to consume.
        assert!(matches!(
            unsafe { (*wrap).inner.get_property("boom") },
            JsValue::Undefined
        ));
        let exception = unsafe { stator_isolate_clear_pending_exception(iso.as_ptr()) };
        assert!(!exception.is_null());
        unsafe { stator_value_destroy(exception) };

        unsafe { stator_dom_object_wrap_destroy(wrap) };
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_dom_named_handler_query_and_delete() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: None,
            setter: None,
            query: Some(named_query_id),
            deleter: Some(named_delete_id),
            enumerator: None,
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) };
        assert_eq!(status, StatorStatus::StatorStatusOk);

        assert!(unsafe { (*wrap).inner.has_property("id") });
        assert!(!unsafe { (*wrap).inner.has_property("missing") });
        assert!(unsafe { (*wrap).inner.delete_property("id") });
        assert!(!unsafe { (*wrap).inner.delete_property("missing") });

        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_dom_named_handler_enumerator() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: None,
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_ab),
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) };
        assert_eq!(status, StatorStatus::StatorStatusOk);

        let names = unsafe { (*wrap).inner.property_names() };
        assert!(names.iter().any(|n| n == "alpha"));
        assert!(names.iter().any(|n| n == "beta"));

        unsafe { stator_dom_object_wrap_destroy(wrap) };
    }

    #[test]
    fn test_context_global_set_dom_object_wrap_dispatches_named_handlers() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        ACTIVE_DOCUMENT_TITLE.with(|title| *title.borrow_mut() = "Initial".to_string());

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_document_property),
            setter: Some(named_set_document_property),
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_document),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );

        let global_name = c"document";
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, global_name.as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let src = b"document.title + '|' + document.URL + '|' + document.documentURI";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        let ptr = unsafe { stator_value_as_string(result) };
        let actual = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(
            actual,
            "Initial|https://example.test/stator.html|https://example.test/stator.html"
        );
        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
        }

        let write_src = b"document.title = 'Updated'; document.title";
        let write_script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                write_src.as_ptr() as *const c_char,
                write_src.len(),
            )
        };
        let write_result = unsafe { stator_script_run(write_script, ctx) };
        assert!(!write_result.is_null());
        let write_ptr = unsafe { stator_value_as_string(write_result) };
        let updated = unsafe { CStr::from_ptr(write_ptr) }.to_string_lossy();
        assert_eq!(updated, "Updated");
        ACTIVE_DOCUMENT_TITLE.with(|title| assert_eq!(&*title.borrow(), "Updated"));

        unsafe {
            stator_value_destroy(write_result);
            stator_script_free(write_script);
        }

        let rejected_write_src =
            b"document.readonlyMissing = 'polluted'; typeof document.readonlyMissing";
        let rejected_write_script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                rejected_write_src.as_ptr() as *const c_char,
                rejected_write_src.len(),
            )
        };
        let rejected_write_result = unsafe { stator_script_run(rejected_write_script, ctx) };
        assert!(!rejected_write_result.is_null());
        let rejected_write_ptr = unsafe { stator_value_as_string(rejected_write_result) };
        let rejected_write_type = unsafe { CStr::from_ptr(rejected_write_ptr) }.to_string_lossy();
        assert_eq!(rejected_write_type, "undefined");

        unsafe {
            stator_value_destroy(rejected_write_result);
            stator_script_free(rejected_write_script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
        ACTIVE_DOCUMENT_TITLE.with(|title| title.borrow_mut().clear());
    }

    #[test]
    fn test_dom_object_wrap_as_value_round_trips_identity() {
        let iso = IsolateGuard::new();
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        assert!(!wrap.is_null());

        let first = unsafe { stator_dom_object_wrap_as_value(wrap) };
        let second = unsafe { stator_dom_object_wrap_as_value(wrap) };
        assert!(!first.is_null());
        assert!(!second.is_null());
        assert!(unsafe { stator_value_is_object(first) });
        assert!(unsafe { stator_value_strict_equals(first, second) });
        assert_eq!(unsafe { stator_value_as_dom_object_wrap(first) }, wrap);

        let plain = unsafe { stator_value_new_object(iso.as_ptr()) };
        assert!(unsafe { stator_value_as_dom_object_wrap(plain) }.is_null());

        unsafe { stator_dom_object_wrap_invalidate(wrap) };
        assert!(unsafe { stator_value_as_dom_object_wrap(first) }.is_null());

        unsafe {
            stator_value_destroy(plain);
            stator_value_destroy(first);
            stator_value_destroy(second);
            stator_dom_object_wrap_destroy(wrap);
        }
    }

    #[test]
    fn test_dom_object_wrap_returned_from_getter_preserves_identity() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let document_wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let element_wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };

        let element_handler = StatorDomNamedHandler {
            getter: Some(named_get_element_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_element),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(element_wrap, &element_handler) },
            StatorStatus::StatorStatusOk
        );

        let document_handler = StatorDomNamedHandler {
            getter: Some(named_get_document_wrapper_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_document_wrapper),
            data: element_wrap as *mut c_void,
        };
        assert_eq!(
            unsafe {
                stator_dom_object_wrap_install_named_handler(document_wrap, &document_handler)
            },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe {
                stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), document_wrap)
            },
            StatorStatus::StatorStatusOk
        );

        let src = b"(document.documentElement === document.documentElement) + ':' + document.documentElement.nodeName + ':' + document.documentElement.nodeType";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        let ptr = unsafe { stator_value_as_string(result) };
        let actual = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(actual, "true:HTML:1");

        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(document_wrap);
            stator_dom_object_wrap_destroy(element_wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_dom_object_wrap_method_receiver_round_trips_to_wrap() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        CAPTURED_DOM_THIS_CLASS_ID.store(0, std::sync::atomic::Ordering::SeqCst);
        CAPTURED_DOM_THIS_NATIVE_PTR.store(0, std::sync::atomic::Ordering::SeqCst);

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let mut native_tag = 0u8;
        let native = &mut native_tag as *mut u8 as *mut c_void;
        unsafe {
            stator_dom_object_wrap_set_class_id(wrap, 42);
            stator_dom_object_wrap_set_native_ptr(wrap, native);
        }
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_capture_method),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_capture_method),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let src = b"document.captureThis()";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        assert_eq!(
            CAPTURED_DOM_THIS_CLASS_ID.load(std::sync::atomic::Ordering::SeqCst),
            42
        );
        assert_eq!(
            CAPTURED_DOM_THIS_NATIVE_PTR.load(std::sync::atomic::Ordering::SeqCst),
            native as usize
        );

        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_dom_object_wrap_invalidate_makes_cached_alias_safe() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        ACTIVE_DOCUMENT_TITLE.with(|title| *title.borrow_mut() = "Initial".to_string());

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_document_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_document),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );
        unsafe { stator_dom_object_wrap_invalidate(wrap) };

        let src = b"typeof document.title";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        let ptr = unsafe { stator_value_as_string(result) };
        let actual = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(actual, "undefined");

        unsafe {
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
        ACTIVE_DOCUMENT_TITLE.with(|title| title.borrow_mut().clear());
    }

    #[test]
    fn test_context_global_dom_named_getter_exception_fails_script() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_throwing_document_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_boom),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );

        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let src = b"document.boom";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(result.is_null());
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let exception = unsafe { stator_try_catch_exception(tc) };
        assert!(!exception.is_null());
        let ptr = unsafe { stator_value_as_string(exception) };
        let message = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(message, "DOM getter boom");

        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(exception);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_context_global_dom_named_keyed_getter_exception_fails_script() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_throwing_document_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_boom),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let src = b"document['boom']";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(result.is_null());
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let exception = unsafe { stator_try_catch_exception(tc) };
        assert!(!exception.is_null());
        let ptr = unsafe { stator_value_as_string(exception) };
        let message = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(message, "DOM getter boom");

        unsafe {
            stator_try_catch_destroy(tc);
            stator_value_destroy(exception);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_context_global_dom_named_getter_exception_caught_clears_pending_exception() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: Some(named_get_status_exception_document_property),
            setter: None,
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_boom),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let src = b"try { document.boom; 'missed'; } catch (e) { 'caught'; }";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(!result.is_null());
        let ptr = unsafe { stator_value_as_string(result) };
        let actual = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(actual, "caught");
        assert!(!unsafe { stator_isolate_has_pending_exception(iso.as_ptr()) });

        let followup_src = b"21 + 21";
        let followup_script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                followup_src.as_ptr() as *const c_char,
                followup_src.len(),
            )
        };
        let followup = unsafe { stator_script_run(followup_script, ctx) };
        assert!(!followup.is_null());
        assert_eq!(unsafe { stator_value_as_number(followup) }, 42.0);

        unsafe {
            stator_value_destroy(followup);
            stator_script_free(followup_script);
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_context_global_dom_named_setter_exception_fails_script() {
        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));

        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomNamedHandler {
            getter: None,
            setter: Some(named_set_throwing_document_property),
            query: None,
            deleter: None,
            enumerator: Some(named_enumerate_boom),
            data: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { stator_dom_object_wrap_install_named_handler(wrap, &handler) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, c"document".as_ptr(), wrap) },
            StatorStatus::StatorStatusOk
        );

        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let src = b"document.boom = 1";
        let script = unsafe {
            stator_script_compile(
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        let result = unsafe { stator_script_run(script, ctx) };
        assert!(result.is_null());
        assert!(unsafe { stator_try_catch_has_caught(tc) });
        let exception = unsafe { stator_try_catch_exception(tc) };
        assert!(!exception.is_null());
        let ptr = unsafe { stator_value_as_string(exception) };
        let message = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(message, "DOM named setter exception");

        unsafe {
            stator_try_catch_destroy(tc);
            stator_script_free(script);
            stator_context_destroy(ctx);
            stator_dom_object_wrap_destroy(wrap);
        }
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_context_global_set_dom_object_wrap_rejects_invalid_args() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let name = c"document";

        assert_eq!(
            unsafe {
                stator_context_global_set_dom_object_wrap(std::ptr::null_mut(), name.as_ptr(), wrap)
            },
            StatorStatus::StatorStatusInvalidArg
        );
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, std::ptr::null(), wrap,) },
            StatorStatus::StatorStatusInvalidArg
        );
        assert_eq!(
            unsafe {
                stator_context_global_set_dom_object_wrap(ctx, name.as_ptr(), std::ptr::null_mut())
            },
            StatorStatus::StatorStatusInvalidArg
        );

        let other_iso = IsolateGuard::new();
        let other_wrap = unsafe { stator_dom_object_wrap_new(other_iso.as_ptr(), 0) };
        assert_eq!(
            unsafe { stator_context_global_set_dom_object_wrap(ctx, name.as_ptr(), other_wrap) },
            StatorStatus::StatorStatusInvalidArg
        );

        unsafe {
            stator_dom_object_wrap_destroy(other_wrap);
            stator_dom_object_wrap_destroy(wrap);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_dom_indexed_handler_install_and_dispatch() {
        use stator_jse::objects::value::JsValue;

        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 0) };
        let handler = StatorDomIndexedHandler {
            getter: Some(indexed_get_zero),
            setter: None,
            query: None,
            length: Some(indexed_length_three),
            data: std::ptr::null_mut(),
        };
        let status = unsafe { stator_dom_object_wrap_install_indexed_handler(wrap, &handler) };
        assert_eq!(status, StatorStatus::StatorStatusOk);

        match unsafe { (*wrap).inner.get_indexed(0) } {
            JsValue::Smi(n) => assert_eq!(n, 7),
            other => panic!("expected Smi(7), got {other:?}"),
        }
        // Out-of-range falls through to Undefined.
        assert!(matches!(
            unsafe { (*wrap).inner.get_indexed(5) },
            JsValue::Undefined
        ));
        assert_eq!(unsafe { (*wrap).inner.indexed_length() }, 3);

        unsafe { stator_dom_object_wrap_destroy(wrap) };
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    #[test]
    fn test_dom_name_buffer_push_rejects_invalid_args() {
        // null buf
        let bytes = b"abc";
        let status = unsafe {
            stator_dom_name_buffer_push(
                std::ptr::null_mut(),
                bytes.as_ptr() as *const c_char,
                bytes.len(),
            )
        };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);

        let mut buf = StatorDomNameBuffer { names: Vec::new() };
        // null name with non-zero length
        let status =
            unsafe { stator_dom_name_buffer_push(&mut buf as *mut _, std::ptr::null(), 4) };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
        // empty name with null pointer is allowed.
        let status =
            unsafe { stator_dom_name_buffer_push(&mut buf as *mut _, std::ptr::null(), 0) };
        assert_eq!(status, StatorStatus::StatorStatusOk);
        assert_eq!(buf.names.last().map(String::as_str), Some(""));
        // invalid UTF-8 is rejected.
        let bad = [0xffu8, 0xfe];
        let status = unsafe {
            stator_dom_name_buffer_push(
                &mut buf as *mut _,
                bad.as_ptr() as *const c_char,
                bad.len(),
            )
        };
        assert_eq!(status, StatorStatus::StatorStatusInvalidArg);
    }

    #[test]
    fn test_dom_legacy_v1_named_getter_still_works() {
        use stator_jse::objects::value::JsValue;

        // Preserve existing v1 API: install via the legacy entry point and
        // confirm property reads still go through the interceptor.
        unsafe extern "C" fn v1_getter(
            _name: *const c_char,
            _data: *mut c_void,
            out: *mut *mut StatorValue,
        ) -> bool {
            let iso = ACTIVE_ISO.with(|c| c.get());
            let v = unsafe { stator_value_new_number(iso, 5.0) };
            unsafe { *out = v };
            true
        }

        let iso = IsolateGuard::new();
        ACTIVE_ISO.with(|c| c.set(iso.as_ptr()));
        let wrap = unsafe { stator_dom_object_wrap_new(iso.as_ptr(), 1) };
        unsafe { stator_dom_object_wrap_set_named_getter(wrap, v1_getter) };
        match unsafe { (*wrap).inner.get_property("anything") } {
            JsValue::Smi(n) => assert_eq!(n, 5),
            other => panic!("expected Smi(5), got {other:?}"),
        }
        unsafe { stator_dom_object_wrap_destroy(wrap) };
        ACTIVE_ISO.with(|c| c.set(std::ptr::null_mut()));
    }

    // ── Edge-proof Maglev tiering parity ──────────────────────────────────────

    /// Helper: read tiering stats via the FFI surface (same path Edge uses).
    fn read_tiering_stats(iso: *const StatorIsolate) -> StatorTieringStats {
        let mut stats: StatorTieringStats = unsafe { std::mem::zeroed() };
        unsafe { stator_isolate_get_tiering_stats(iso, &mut stats as *mut _) };
        stats
    }

    /// Run an Edge-proof shaped workload (compile once, warmup + measured)
    /// against `source` and return `(final_number_result, stats)`.
    ///
    /// Mirrors the Edge proof harness: 5 warmup runs followed by 20 measured
    /// runs over the SAME compiled script (the "compiled_run" path).
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    fn run_edge_proof_compiled(source: &[u8]) -> (f64, StatorTieringStats) {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // SAFETY: `iso` is valid.
        unsafe { stator_isolate_reset_tiering_stats(iso.as_ptr()) };

        // SAFETY: `ctx` is valid; `source` is a valid byte slice.
        let script =
            unsafe { stator_script_compile(ctx, source.as_ptr() as *const c_char, source.len()) };
        assert!(!script.is_null(), "script must compile");
        // SAFETY: `script` is non-null.
        let err = unsafe { stator_script_get_error(script) };
        assert!(err.is_null(), "unexpected compile error");

        let mut last = f64::NAN;
        // 5 warmup + 20 measured = 25 total runs over the same compiled BA.
        for i in 0..25 {
            // SAFETY: `script` and `ctx` are valid.
            let result = unsafe { stator_script_run(script, ctx) };
            assert!(!result.is_null(), "run {i} returned null");
            // SAFETY: `result` is non-null.
            last = unsafe { stator_value_as_number(result) };
            // SAFETY: `result` is non-null.
            unsafe { stator_value_destroy(result) };
            // Give the background Maglev thread a moment to install code
            // before the next run picks it up.  Mirrors Edge's per-iteration
            // pacing (each measured iteration is hundreds of microseconds).
            if i < 6 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }

        let stats = read_tiering_stats(iso.as_ptr());

        // SAFETY: pointers are live.
        unsafe {
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
        (last, stats)
    }

    /// Loop2K from Edge `edge_stator_jse_test_cases.cc`: `7000`.
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    #[test]
    fn edge_proof_loop2k_executes_maglev() {
        let src = b"var total = 0; for (var i = 0; i < 2000; ++i) { total += (i & 7); } total;";
        let (result, stats) = run_edge_proof_compiled(src);
        assert!(
            (result - 7000.0).abs() < f64::EPSILON,
            "Loop2K must return 7000, got {result}"
        );
        assert!(
            stats.maglev_function_count > 0,
            "Maglev should have compiled at least one function: {stats:?}"
        );
        assert!(
            stats.maglev_executed > 0,
            "Maglev should have executed at least once over 25 runs (compiled={}, tried={}, executed={}, deopts={}, not_ready={}, blocked={}, cache_empty={})",
            stats.maglev_function_count,
            stats.maglev_tried,
            stats.maglev_executed,
            stats.maglev_deopts,
            stats.maglev_not_ready,
            stats.maglev_blocked,
            stats.maglev_cache_empty,
        );
    }

    /// FunctionLoop1K from Edge `edge_stator_jse_test_cases.cc`: `2000`.
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    #[test]
    fn edge_proof_function_loop1k_executes_maglev() {
        let src = b"function f(n) { var total = 0; for (var i = 0; i < n; ++i) { total += (i % 5); } return total; } f(1000);";
        let (result, stats) = run_edge_proof_compiled(src);
        assert!(
            (result - 2000.0).abs() < f64::EPSILON,
            "FunctionLoop1K must return 2000, got {result}"
        );
        assert!(
            stats.maglev_function_count > 0,
            "Maglev should have compiled at least one function: {stats:?}"
        );
        assert!(
            stats.maglev_executed > 0,
            "Maglev should have executed at least once over 25 runs (compiled={}, tried={}, executed={}, deopts={}, not_ready={}, blocked={}, cache_empty={})",
            stats.maglev_function_count,
            stats.maglev_tried,
            stats.maglev_executed,
            stats.maglev_deopts,
            stats.maglev_not_ready,
            stats.maglev_blocked,
            stats.maglev_cache_empty,
        );
    }

    /// Edge-pattern release-mode timing harness.
    ///
    /// Mirrors the Edge "compiled_run" benchmark exactly:
    ///   - `stator_script_compile` once
    ///   - 5 warmup `stator_script_run` calls
    ///   - `MEASURED` measured `stator_script_run` calls (same compiled script)
    ///
    /// Returns `(median_ns, mean_ns, min_ns, max_ns, last_value, stats)`.
    ///
    /// Uses `Instant::now()` around the *exact* `stator_script_run` body —
    /// the same function Edge wraps in its perf timer.  The harness does not
    /// sleep between iterations once warmup has completed (Edge's measured
    /// loop also runs back-to-back).
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    fn edge_pattern_timed(
        source: &[u8],
        measured: usize,
    ) -> (u128, u128, u128, u128, f64, StatorTieringStats) {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        unsafe { stator_isolate_reset_tiering_stats(iso.as_ptr()) };

        let script =
            unsafe { stator_script_compile(ctx, source.as_ptr() as *const c_char, source.len()) };
        assert!(!script.is_null());
        let err = unsafe { stator_script_get_error(script) };
        assert!(err.is_null(), "unexpected compile error");

        // 5 warmup runs (with brief sleeps so the background Maglev compile
        // has a chance to install code, just like the existing edge-proof
        // tests above).
        let mut last = f64::NAN;
        for _ in 0..5 {
            let r = unsafe { stator_script_run(script, ctx) };
            assert!(!r.is_null(), "warmup run returned null");
            last = unsafe { stator_value_as_number(r) };
            unsafe { stator_value_destroy(r) };
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // Measured runs — back-to-back, no sleeps, no extra work in the
        // timed body.
        let mut samples: Vec<u128> = Vec::with_capacity(measured);
        for _ in 0..measured {
            let t0 = std::time::Instant::now();
            let r = unsafe { stator_script_run(script, ctx) };
            let elapsed = t0.elapsed().as_nanos();
            assert!(!r.is_null(), "measured run returned null");
            last = unsafe { stator_value_as_number(r) };
            unsafe { stator_value_destroy(r) };
            samples.push(elapsed);
        }
        samples.sort_unstable();
        let min = *samples.first().unwrap();
        let max = *samples.last().unwrap();
        let median = samples[samples.len() / 2];
        let mean = samples.iter().sum::<u128>() / samples.len() as u128;

        let stats = read_tiering_stats(iso.as_ptr());
        unsafe {
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
        (median, mean, min, max, last, stats)
    }

    /// Print release-mode timings for the three Edge perf-proof scripts
    /// using exactly the FFI pattern Edge times.  Ignored by default so it
    /// does not slow down `cargo test`; run with:
    ///   `cargo test -p stator_jse_ffi --release \
    ///        edge_pattern_timing_release -- --ignored --nocapture`
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    #[test]
    #[ignore]
    fn edge_pattern_timing_release() {
        let scripts: &[(&str, &[u8], f64)] = &[
            (
                "Loop2K",
                b"var total = 0; for (var i = 0; i < 2000; ++i) { total += (i & 7); } total;",
                7000.0,
            ),
            (
                "FunctionLoop1K",
                b"function f(n) { var total = 0; for (var i = 0; i < n; ++i) { total += (i % 5); } return total; } f(1000);",
                2000.0,
            ),
            (
                "StringAppend200",
                b"var value = ''; for (var i = 0; i < 200; ++i) { value += 'x'; } value.length;",
                200.0,
            ),
        ];

        eprintln!(
            "EDGE_FFI_TIMING(stator_script_run): script | median_us | mean_us | min_us | max_us | maglev_executed/tried | result"
        );
        for (name, src, expected) in scripts {
            let (median, mean, min, max, last, stats) = edge_pattern_timed(src, 25);
            assert!(
                (last - expected).abs() < f64::EPSILON,
                "{name}: expected {expected}, got {last}"
            );
            eprintln!(
                "EDGE_FFI_TIMING: {name:<16} | {:>9.3} | {:>8.3} | {:>7.3} | {:>7.3} | {:>5}/{:<5} | {}",
                median as f64 / 1000.0,
                mean as f64 / 1000.0,
                min as f64 / 1000.0,
                max as f64 / 1000.0,
                stats.maglev_executed,
                stats.maglev_tried,
                last,
            );
        }

        // Reconciliation diagnostic: also time the same scripts with JIT
        // tiers disabled (interpreter only).  The Edge perf proof at 0.2.0
        // reported maglev_executed=0 because the diagnostic counters were
        // gated behind cfg(debug_assertions) before commit 86e2865a; that
        // value, combined with Edge's per-iteration times in the hundreds
        // of microseconds, is consistent with the interpreter path running
        // (i.e. Maglev cache miss / Maglev cfg disabled in Edge build).
        eprintln!(
            "EDGE_FFI_TIMING(jit_disabled, interpreter only): script | median_us | mean_us | min_us | max_us | result"
        );
        for (name, src, expected) in scripts {
            let iso = IsolateGuard::new();
            unsafe { stator_isolate_set_jit_disabled(iso.as_ptr(), true) };
            let ctx = unsafe { stator_context_new(iso.as_ptr()) };
            let script =
                unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
            assert!(!script.is_null());
            // Warmup.
            for _ in 0..5 {
                let r = unsafe { stator_script_run(script, ctx) };
                if !r.is_null() {
                    unsafe { stator_value_destroy(r) };
                }
            }
            let mut samples = Vec::with_capacity(25);
            let mut last = f64::NAN;
            for _ in 0..25 {
                let t0 = std::time::Instant::now();
                let r = unsafe { stator_script_run(script, ctx) };
                let elapsed = t0.elapsed().as_nanos();
                assert!(!r.is_null());
                last = unsafe { stator_value_as_number(r) };
                unsafe { stator_value_destroy(r) };
                samples.push(elapsed);
            }
            samples.sort_unstable();
            assert!((last - expected).abs() < f64::EPSILON);
            eprintln!(
                "EDGE_FFI_TIMING(interp): {name:<16} | {:>9.3} | {:>8.3} | {:>7.3} | {:>7.3} | {}",
                samples[samples.len() / 2] as f64 / 1000.0,
                (samples.iter().sum::<u128>() / samples.len() as u128) as f64 / 1000.0,
                *samples.first().unwrap() as f64 / 1000.0,
                *samples.last().unwrap() as f64 / 1000.0,
                last,
            );
            unsafe {
                stator_script_free(script);
                stator_context_destroy(ctx);
            }
        }
    }

    /// StringAppend200 from Edge `edge_stator_jse_test_cases.cc`: `200`.
    #[cfg(any(
        stator_maglev_jit_x86_64,
        all(target_arch = "x86_64", any(unix, windows))
    ))]
    #[test]
    fn edge_proof_string_append200_executes_maglev() {
        let src = b"var value = ''; for (var i = 0; i < 200; ++i) { value += 'x'; } value.length;";
        let (result, stats) = run_edge_proof_compiled(src);
        assert!(
            (result - 200.0).abs() < f64::EPSILON,
            "StringAppend200 must return 200, got {result}"
        );
        assert!(
            stats.maglev_function_count > 0,
            "Maglev should have compiled at least one function: {stats:?}"
        );
        assert!(
            stats.maglev_executed > 0,
            "Maglev should have executed at least once over 25 runs (compiled={}, tried={}, executed={}, deopts={}, not_ready={}, blocked={}, cache_empty={})",
            stats.maglev_function_count,
            stats.maglev_tried,
            stats.maglev_executed,
            stats.maglev_deopts,
            stats.maglev_not_ready,
            stats.maglev_blocked,
            stats.maglev_cache_empty,
        );
    }

    // ── Browser P0 plumbing tests ──────────────────────────────────────────

    #[test]
    fn test_isolate_terminate_execution_flag_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        unsafe {
            assert!(!stator_isolate_is_execution_terminating(iso.as_ptr()));
            stator_isolate_terminate_execution(iso.as_ptr());
            assert!(stator_isolate_is_execution_terminating(iso.as_ptr()));
            stator_isolate_cancel_terminate_execution(iso.as_ptr());
            assert!(!stator_isolate_is_execution_terminating(iso.as_ptr()));
        }
    }

    #[test]
    fn test_isolate_terminate_execution_blocks_script_start() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let src = b"var edgeStatorTerminated = 1; edgeStatorTerminated;";
        // SAFETY: `ctx` is valid; `src` is valid UTF-8.
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        // SAFETY: all pointers are valid and live.
        unsafe {
            stator_isolate_terminate_execution(iso.as_ptr());
            assert!(stator_script_run(script, ctx).is_null());
            assert!(!stator_script_run_no_result(script, ctx));

            stator_isolate_cancel_terminate_execution(iso.as_ptr());
            let result = stator_script_run(script, ctx);
            assert!(!result.is_null());
            assert_eq!(stator_value_as_number(result), 1.0);
            stator_value_destroy(result);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_isolate_terminate_null_is_safe() {
        // SAFETY: passing null is explicitly documented as a no-op.
        unsafe {
            stator_isolate_terminate_execution(std::ptr::null_mut());
            stator_isolate_cancel_terminate_execution(std::ptr::null_mut());
            assert!(!stator_isolate_is_execution_terminating(std::ptr::null()));
        }
    }

    /// A hostile `while (true) {}` script must be terminable from another
    /// thread within a bounded time, and the isolate must be reusable
    /// after the termination is cancelled.
    #[test]
    fn test_isolate_terminate_aborts_infinite_loop_cross_thread() {
        // Sendable pointer wrapper: `*mut StatorIsolate` is not `Send` by
        // default but the underlying flag is atomic and is safe to set from
        // any thread.
        #[derive(Copy, Clone)]
        struct IsoPtr(usize);
        // SAFETY: the only operations performed on the other thread are
        // atomic stores to the isolate's termination flag, which are
        // thread-safe by construction.
        unsafe impl Send for IsoPtr {}

        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());

        // Disable JIT so the test exercises the interpreter polling path
        // deterministically; the JIT does not poll in this slice.
        unsafe {
            stator_isolate_set_jit_disabled(iso.as_ptr(), true);
        }

        let src = b"while (true) { }";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        let iso_ptr = IsoPtr(iso.as_ptr() as usize);
        let killer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            // SAFETY: the main thread keeps the isolate alive until after
            // join; the atomic store is thread-safe.
            unsafe {
                stator_isolate_terminate_execution(iso_ptr.0 as *mut StatorIsolate);
            }
        });

        let start = std::time::Instant::now();
        let result = unsafe { stator_script_run(script, ctx) };
        let elapsed = start.elapsed();
        killer.join().unwrap();

        assert!(
            result.is_null(),
            "script_run should return null after termination"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "termination took too long: {elapsed:?}"
        );
        // The flag should still be set (it is sticky until cancelled).
        unsafe {
            assert!(stator_isolate_is_execution_terminating(iso.as_ptr()));
        }

        // Cancelling termination must restore the ability to run scripts.
        unsafe {
            stator_isolate_cancel_terminate_execution(iso.as_ptr());
            assert!(!stator_isolate_is_execution_terminating(iso.as_ptr()));

            let src2 = b"1 + 2";
            let script2 = stator_script_compile(ctx, src2.as_ptr() as *const c_char, src2.len());
            assert!(!script2.is_null());
            let r = stator_script_run(script2, ctx);
            assert!(!r.is_null(), "script must run after cancel");
            assert_eq!(stator_value_as_number(r), 3.0);
            stator_value_destroy(r);
            stator_script_free(script2);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    /// A function-call infinite recursion / loop is terminable via the
    /// `run_inner` call-boundary poll.
    #[test]
    fn test_isolate_terminate_aborts_function_call_loop() {
        #[derive(Copy, Clone)]
        struct IsoPtr(usize);
        // SAFETY: only an atomic store is performed off-thread.
        unsafe impl Send for IsoPtr {}

        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        unsafe {
            stator_isolate_set_jit_disabled(iso.as_ptr(), true);
        }

        // Loop body contains a function call so the call-boundary poll is
        // also exercised.
        let src = b"function f(){return 1} while (true) { f(); }";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        let iso_ptr = IsoPtr(iso.as_ptr() as usize);
        let killer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            unsafe {
                stator_isolate_terminate_execution(iso_ptr.0 as *mut StatorIsolate);
            }
        });

        let start = std::time::Instant::now();
        let result = unsafe { stator_script_run(script, ctx) };
        let elapsed = start.elapsed();
        killer.join().unwrap();

        assert!(result.is_null());
        assert!(elapsed < std::time::Duration::from_secs(5));

        unsafe {
            stator_isolate_cancel_terminate_execution(iso.as_ptr());
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    /// A JIT-resident hot loop must observe termination via the Maglev
    /// loop-header poll without first returning to the interpreter.
    #[test]
    fn test_isolate_terminate_aborts_maglev_jit_hot_loop() {
        #[derive(Copy, Clone)]
        struct IsoPtr(usize);
        // SAFETY: only an atomic store is performed off-thread.
        unsafe impl Send for IsoPtr {}

        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());

        // JIT remains enabled (default).  The function is warmed up so
        // Maglev compiles it, then the inner loop runs JIT-resident and
        // must be terminated by the loop-header termination poll.
        let src = b"function f(){var i=0;for(;;){i=i+1|0}} \
                    for (var w=0; w<5000; w++){} \
                    f();";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());

        let iso_ptr = IsoPtr(iso.as_ptr() as usize);
        let killer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(250));
            unsafe {
                stator_isolate_terminate_execution(iso_ptr.0 as *mut StatorIsolate);
            }
        });

        let start = std::time::Instant::now();
        let result = unsafe { stator_script_run(script, ctx) };
        let elapsed = start.elapsed();
        killer.join().unwrap();

        assert!(result.is_null());
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "JIT hot loop did not terminate in time: {:?}",
            elapsed
        );

        unsafe {
            stator_isolate_cancel_terminate_execution(iso.as_ptr());
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    /// A microtask flood must observe termination between tasks.  We drive
    /// the microtask queue directly to keep the test independent of the
    /// FFI surface for Promise microtask draining.
    #[test]
    fn test_microtask_drain_observes_interrupt_flag_between_tasks() {
        use std::cell::Cell as StdCell;
        use std::rc::Rc as StdRc;

        let flag = AtomicBool::new(false);
        // SAFETY: `flag` outlives the publish/clear pair below.
        unsafe { stator_jse::interpreter::set_interrupt_flag(&flag as *const _) };

        let queue = stator_jse::builtins::promise::MicrotaskQueue::new();
        let ran = StdRc::new(StdCell::new(0u32));

        // Enqueue 1000 tasks; the 5th one trips the termination flag.
        // Tasks scheduled after termination must not run during this drain.
        for i in 0..1000u32 {
            let ran = StdRc::clone(&ran);
            let flag_ptr: *const AtomicBool = &flag;
            queue.enqueue(Box::new(move || {
                ran.set(ran.get() + 1);
                if i == 4 {
                    // SAFETY: `flag` outlives the drain.
                    unsafe { (*flag_ptr).store(true, Ordering::Relaxed) };
                }
            }));
        }

        queue.drain();

        // Tasks 0..=4 should have run; tasks >= 5 should not have run during
        // this drain (the between-task poll bails out).
        assert!(
            ran.get() <= 5,
            "expected drain to bail after termination flag was set, ran {} tasks",
            ran.get()
        );
        assert!(
            ran.get() >= 5,
            "expected at least 5 tasks (through the one that set the flag) to run"
        );
        // Remaining tasks are preserved; once we cancel, they would drain.
        assert!(!queue.is_empty());

        flag.store(false, Ordering::Relaxed);
        queue.drain();
        assert!(queue.is_empty());
        assert_eq!(ran.get(), 1000);

        stator_jse::interpreter::clear_interrupt_flag();
    }

    #[test]
    fn test_context_embedder_data_roundtrip() {
        let iso = IsolateGuard::new();
        // SAFETY: `iso` is valid.
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        assert!(!ctx.is_null());
        let sentinel = 0xDEADBEEF_usize as *mut c_void;
        // SAFETY: `ctx` is valid.
        unsafe {
            assert!(stator_context_get_embedder_data(ctx, 0).is_null());
            stator_context_set_embedder_data(ctx, 3, sentinel);
            assert_eq!(stator_context_get_embedder_data(ctx, 3), sentinel);
            assert!(stator_context_get_embedder_data(ctx, 0).is_null());
            assert!(stator_context_get_embedder_data(ctx, 2).is_null());
            assert!(stator_context_get_embedder_data(ctx, 99).is_null());
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_context_embedder_data_null_is_safe() {
        // SAFETY: passing null is documented as a no-op.
        unsafe {
            stator_context_set_embedder_data(std::ptr::null_mut(), 0, std::ptr::null_mut());
            assert!(stator_context_get_embedder_data(std::ptr::null(), 0).is_null());
        }
    }

    #[test]
    fn test_script_origin_defaults_and_roundtrip() {
        let src = b"1 + 1\0";
        // SAFETY: `src` is valid for 5 bytes.
        let script = unsafe {
            stator_script_compile(std::ptr::null_mut(), src.as_ptr() as *const c_char, 5)
        };
        assert!(!script.is_null());

        // SAFETY: `script` is valid.
        unsafe {
            assert!(stator_script_get_resource_name(script).is_null());
            assert_eq!(stator_script_get_line_offset(script), 0);
            assert_eq!(stator_script_get_column_offset(script), 0);
        }

        let name = c"https://example.test/foo.js";
        // SAFETY: `script` is valid; `name` is a valid C string.
        unsafe {
            stator_script_set_origin(script, name.as_ptr(), 7, 13);
            let got = stator_script_get_resource_name(script);
            assert!(!got.is_null());
            let got_str = CStr::from_ptr(got).to_str().unwrap();
            assert_eq!(got_str, "https://example.test/foo.js");
            assert_eq!(stator_script_get_line_offset(script), 7);
            assert_eq!(stator_script_get_column_offset(script), 13);
        }

        // Setting a null resource_name clears the previously-set name.
        // SAFETY: `script` is valid.
        unsafe {
            stator_script_set_origin(script, std::ptr::null(), 0, 0);
            assert!(stator_script_get_resource_name(script).is_null());
            assert_eq!(stator_script_get_line_offset(script), 0);
            assert_eq!(stator_script_get_column_offset(script), 0);
            stator_script_free(script);
        }
    }

    #[test]
    fn test_script_origin_null_script_is_safe() {
        // SAFETY: passing null is documented as a no-op / default.
        unsafe {
            stator_script_set_origin(std::ptr::null_mut(), std::ptr::null(), 1, 2);
            assert!(stator_script_get_resource_name(std::ptr::null()).is_null());
            assert_eq!(stator_script_get_line_offset(std::ptr::null()), 0);
            assert_eq!(stator_script_get_column_offset(std::ptr::null()), 0);
        }
    }

    // ── Structured message FFI ────────────────────────────────────────────

    #[test]
    fn test_message_null_safety() {
        // All accessors on a null message return the documented defaults
        // rather than dereferencing.
        // SAFETY: passing null is documented as supported by every accessor.
        unsafe {
            assert_eq!(
                stator_message_kind(std::ptr::null()),
                StatorMessageKind::StatorMessageKindUnknown
            );
            assert!(stator_message_text(std::ptr::null()).is_null());
            assert!(stator_message_resource_name(std::ptr::null()).is_null());
            let mut line: i32 = 42;
            assert!(!stator_message_get_line(std::ptr::null(), &mut line));
            assert_eq!(line, 42, "out param must not be touched on null msg");
            let mut col: i32 = 7;
            assert!(!stator_message_get_column(std::ptr::null(), &mut col));
            assert_eq!(col, 7, "out param must not be touched on null msg");
            assert!(!stator_message_terminated(std::ptr::null()));
            stator_message_destroy(std::ptr::null_mut());
            assert!(stator_isolate_peek_pending_message(std::ptr::null()).is_null());
            assert!(stator_isolate_take_pending_message(std::ptr::null_mut()).is_null());
            assert!(stator_try_catch_message(std::ptr::null()).is_null());
            assert_eq!(
                stator_script_error_kind(std::ptr::null()),
                StatorMessageKind::StatorMessageKindUnknown
            );
        }
    }

    #[test]
    fn test_message_get_line_column_rejects_null_out_param() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"throw 1;";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
        }
        let msg = unsafe { stator_isolate_peek_pending_message(iso.as_ptr()) };
        assert!(!msg.is_null());
        unsafe {
            assert!(!stator_message_get_line(msg, std::ptr::null_mut()));
            assert!(!stator_message_get_column(msg, std::ptr::null_mut()));
            stator_isolate_clear_pending_exception(iso.as_ptr());
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_script_compile_syntax_error_classified() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        // A malformed source surfaces as a parse error.
        let src = b"function (";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        assert!(!script.is_null());
        unsafe {
            // Existing string-based API still works.
            assert!(!stator_script_get_error(script).is_null());
            // New structured API classifies it as a syntax error.
            assert_eq!(
                stator_script_error_kind(script),
                StatorMessageKind::StatorMessageKindSyntax
            );
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_script_compile_success_has_unknown_kind() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"1 + 1";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            assert!(stator_script_get_error(script).is_null());
            assert_eq!(
                stator_script_error_kind(script),
                StatorMessageKind::StatorMessageKindUnknown
            );
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_script_compile_null_source_classified_internal() {
        let script = unsafe { stator_script_compile(std::ptr::null_mut(), std::ptr::null(), 0) };
        assert!(!script.is_null());
        unsafe {
            assert!(!stator_script_get_error(script).is_null());
            assert_eq!(
                stator_script_error_kind(script),
                StatorMessageKind::StatorMessageKindInternal
            );
            stator_script_free(script);
        }
    }

    #[test]
    fn test_pending_message_runtime_type_error() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"var notFn = 1; notFn();";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        // Attach a resource name so the message surfaces it.
        let resource = c"file:///t.js";
        unsafe { stator_script_set_origin(script, resource.as_ptr(), 0, 0) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            let msg = stator_isolate_peek_pending_message(iso.as_ptr());
            assert!(!msg.is_null(), "structured message must be populated");
            assert_eq!(
                stator_message_kind(msg),
                StatorMessageKind::StatorMessageKindType
            );
            let text = stator_message_text(msg);
            assert!(!text.is_null());
            let text_s = CStr::from_ptr(text).to_string_lossy();
            assert!(text_s.contains("TypeError"), "message text was {text_s:?}");
            let rname = stator_message_resource_name(msg);
            assert!(!rname.is_null());
            assert_eq!(CStr::from_ptr(rname).to_string_lossy(), "file:///t.js");
            // No line/column info plumbed yet — getters must report missing
            // data rather than fabricate zeros.
            let mut line: i32 = -1;
            assert!(!stator_message_get_line(msg, &mut line));
            assert_eq!(line, -1);
            let mut col: i32 = -1;
            assert!(!stator_message_get_column(msg, &mut col));
            assert_eq!(col, -1);
            assert!(!stator_message_terminated(msg));

            // Clean up: drain the pending exception/message.
            let exc = stator_isolate_clear_pending_exception(iso.as_ptr());
            if !exc.is_null() {
                stator_value_destroy(exc);
            }
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_pending_message_classifies_js_exception_throw() {
        // `throw 1;` propagates as an uncaught JS exception value; the
        // engine classifies the resulting structured error as JsException.
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"throw 1;";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            let msg = stator_isolate_peek_pending_message(iso.as_ptr());
            assert!(!msg.is_null());
            assert_eq!(
                stator_message_kind(msg),
                StatorMessageKind::StatorMessageKindJsException
            );
            let exc = stator_isolate_clear_pending_exception(iso.as_ptr());
            if !exc.is_null() {
                stator_value_destroy(exc);
            }
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_pending_message_termination_classified() {
        // Terminating execution before script entry surfaces a Termination
        // message and `terminated == true`.
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"42";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            // SAFETY: `iso` is valid.
            (*iso.as_ptr()).terminating.store(true, Ordering::Relaxed);
            assert!(stator_script_run(script, ctx).is_null());
            let msg = stator_isolate_peek_pending_message(iso.as_ptr());
            assert!(!msg.is_null());
            assert_eq!(
                stator_message_kind(msg),
                StatorMessageKind::StatorMessageKindTermination
            );
            assert!(stator_message_terminated(msg));
            // Reset the termination flag so the destructor path doesn't keep
            // poisoning subsequent helpers.
            (*iso.as_ptr()).terminating.store(false, Ordering::Relaxed);
            let exc = stator_isolate_clear_pending_exception(iso.as_ptr());
            if !exc.is_null() {
                stator_value_destroy(exc);
            }
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_take_pending_message_transfers_ownership() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"throw 1;";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            // peek returns Some, take returns the same data and clears the slot.
            assert!(!stator_isolate_peek_pending_message(iso.as_ptr()).is_null());
            let owned = stator_isolate_take_pending_message(iso.as_ptr());
            assert!(!owned.is_null());
            assert!(stator_isolate_peek_pending_message(iso.as_ptr()).is_null());
            // Owner can still read fields.
            assert!(!stator_message_text(owned).is_null());
            // Destroy via the dedicated API.
            stator_message_destroy(owned);
            // The pending exception value itself is still set and must be
            // released through the existing API.
            let exc = stator_isolate_clear_pending_exception(iso.as_ptr());
            if !exc.is_null() {
                stator_value_destroy(exc);
            }
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_try_catch_message_for_script_run_error() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let src = b"var x = 1; x();";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        let resource = c"file:///tc.js";
        unsafe { stator_script_set_origin(script, resource.as_ptr(), 0, 0) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            assert!(stator_try_catch_has_caught(tc));
            let msg = stator_try_catch_message(tc);
            assert!(!msg.is_null());
            assert_eq!(
                stator_message_kind(msg),
                StatorMessageKind::StatorMessageKindType
            );
            let rname = stator_message_resource_name(msg);
            assert!(!rname.is_null());
            assert_eq!(CStr::from_ptr(rname).to_string_lossy(), "file:///tc.js");
            // Reset clears the structured message as well.
            stator_try_catch_reset(tc);
            assert!(stator_try_catch_message(tc).is_null());

            stator_try_catch_destroy(tc);
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_try_catch_message_absent_for_embedder_thrown_exception() {
        // Embedder-thrown raw values have no structured classification.
        let iso = IsolateGuard::new();
        let tc = unsafe { stator_try_catch_new(iso.as_ptr()) };
        let exc = unsafe { stator_value_new_string(iso.as_ptr(), c"oops".as_ptr(), 4) };
        unsafe { stator_isolate_throw_exception(iso.as_ptr(), exc) };
        unsafe {
            assert!(stator_try_catch_has_caught(tc));
            assert!(stator_try_catch_message(tc).is_null());
            stator_try_catch_destroy(tc);
            stator_value_destroy(exc);
        }
    }

    #[test]
    fn test_clear_pending_exception_drops_message() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let src = b"throw 1;";
        let script =
            unsafe { stator_script_compile(ctx, src.as_ptr() as *const c_char, src.len()) };
        unsafe {
            assert!(stator_script_run(script, ctx).is_null());
            assert!(!stator_isolate_peek_pending_message(iso.as_ptr()).is_null());
            let exc = stator_isolate_clear_pending_exception(iso.as_ptr());
            if !exc.is_null() {
                stator_value_destroy(exc);
            }
            assert!(stator_isolate_peek_pending_message(iso.as_ptr()).is_null());
            stator_script_free(script);
            stator_context_destroy(ctx);
        }
    }

    // ── Status enum / typed accessors / Maybe-style property API tests ──────

    /// Build a non-empty `StatorObject` via a helper to share setup across
    /// the new-API tests.  The returned object has a single own property
    /// `"key"` set to the integer 42.
    fn make_object_with_key_42(iso: *mut StatorIsolate) -> *mut StatorObject {
        // SAFETY: `iso` is valid for the lifetime of the test guard.
        let obj = unsafe { stator_object_new(iso) };
        let key = c"key";
        // SAFETY: `iso` and `obj` are valid.
        let val = unsafe { stator_value_new_number(iso, 42.0) };
        let status = unsafe {
            stator_object_set_property(
                obj,
                key.as_ptr(),
                key.to_bytes().len(),
                val as *const StatorValue,
            )
        };
        assert_eq!(status, StatorStatus::StatorStatusOk);
        unsafe { stator_value_destroy(val) };
        obj
    }

    #[test]
    fn test_status_get_boolean_matrix() {
        let iso = IsolateGuard::new();
        let mut out = false;
        // Null val → InvalidArg.
        assert_eq!(
            unsafe { stator_value_get_boolean(std::ptr::null(), &mut out) },
            StatorStatus::StatorStatusInvalidArg
        );
        let bool_val = unsafe { stator_value_new_boolean(iso.as_ptr(), true) };
        let num_val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // Null out → InvalidArg.
        assert_eq!(
            unsafe { stator_value_get_boolean(bool_val, std::ptr::null_mut()) },
            StatorStatus::StatorStatusInvalidArg
        );
        // Wrong type → False, out untouched.
        out = false;
        assert_eq!(
            unsafe { stator_value_get_boolean(num_val, &mut out) },
            StatorStatus::StatorStatusFalse
        );
        assert!(!out);
        // Right type → Ok, out filled.
        assert_eq!(
            unsafe { stator_value_get_boolean(bool_val, &mut out) },
            StatorStatus::StatorStatusOk
        );
        assert!(out);
        unsafe {
            stator_value_destroy(bool_val);
            stator_value_destroy(num_val);
        }
    }

    #[test]
    fn test_status_get_number_matrix() {
        let iso = IsolateGuard::new();
        let num_val = unsafe { stator_value_new_number(iso.as_ptr(), 3.5) };
        let str_val = unsafe { stator_value_new_string(iso.as_ptr(), c"x".as_ptr(), 1) };
        let mut out = 0.0;
        assert_eq!(
            unsafe { stator_value_get_number(num_val, &mut out) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(out, 3.5);
        assert_eq!(
            unsafe { stator_value_get_number(str_val, &mut out) },
            StatorStatus::StatorStatusFalse
        );
        // Out preserved.
        assert_eq!(out, 3.5);
        unsafe {
            stator_value_destroy(num_val);
            stator_value_destroy(str_val);
        }
    }

    #[test]
    fn test_status_get_int32_in_and_out_of_range() {
        let iso = IsolateGuard::new();
        let in_range = unsafe { stator_value_new_number(iso.as_ptr(), -7.0) };
        let frac = unsafe { stator_value_new_number(iso.as_ptr(), 1.5) };
        let too_big = unsafe { stator_value_new_number(iso.as_ptr(), 2.0_f64.powi(35)) };
        let mut out: i32 = 99;
        assert_eq!(
            unsafe { stator_value_get_int32(in_range, &mut out) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(out, -7);
        out = 99;
        assert_eq!(
            unsafe { stator_value_get_int32(frac, &mut out) },
            StatorStatus::StatorStatusFalse
        );
        assert_eq!(out, 99);
        assert_eq!(
            unsafe { stator_value_get_int32(too_big, &mut out) },
            StatorStatus::StatorStatusFalse
        );
        assert_eq!(out, 99);
        unsafe {
            stator_value_destroy(in_range);
            stator_value_destroy(frac);
            stator_value_destroy(too_big);
        }
    }

    #[test]
    fn test_status_get_uint32_rejects_negative() {
        let iso = IsolateGuard::new();
        let pos = unsafe { stator_value_new_number(iso.as_ptr(), 4_000_000_000.0) };
        let neg = unsafe { stator_value_new_number(iso.as_ptr(), -1.0) };
        let mut out: u32 = 0;
        assert_eq!(
            unsafe { stator_value_get_uint32(pos, &mut out) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(out, 4_000_000_000);
        assert_eq!(
            unsafe { stator_value_get_uint32(neg, &mut out) },
            StatorStatus::StatorStatusFalse
        );
        unsafe {
            stator_value_destroy(pos);
            stator_value_destroy(neg);
        }
    }

    #[test]
    fn test_string_write_utf8_non_ascii_and_truncation() {
        let iso = IsolateGuard::new();
        // "héllo" — h, é (2 bytes in UTF-8), l, l, o → 6 bytes.
        let src = "héllo".as_bytes();
        let val = unsafe {
            stator_value_new_string(iso.as_ptr(), src.as_ptr() as *const c_char, src.len())
        };
        let mut len: usize = 0;
        assert_eq!(
            unsafe { stator_value_get_string_utf8_length(val, &mut len) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(len, 6);

        // Full-size buffer round-trips bytes.
        let mut buf = vec![0u8; 6];
        let mut written: usize = 0;
        assert_eq!(
            unsafe {
                stator_value_write_string_utf8(
                    val,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                    &mut written,
                )
            },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(written, 6);
        assert_eq!(&buf[..], src);

        // Smaller buffer truncates without writing a NUL terminator.
        let mut small = [0xAAu8; 3];
        written = 999;
        assert_eq!(
            unsafe {
                stator_value_write_string_utf8(
                    val,
                    small.as_mut_ptr() as *mut c_char,
                    small.len(),
                    &mut written,
                )
            },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(written, 3);
        assert_eq!(&small[..], &src[..3]);

        // Wrong type → False, written cleared to 0.
        let num = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let mut buf2 = [0u8; 4];
        written = 17;
        assert_eq!(
            unsafe {
                stator_value_write_string_utf8(
                    num,
                    buf2.as_mut_ptr() as *mut c_char,
                    buf2.len(),
                    &mut written,
                )
            },
            StatorStatus::StatorStatusFalse
        );
        assert_eq!(written, 0);

        // Null val / null buf → InvalidArg, written cleared.
        written = 17;
        assert_eq!(
            unsafe {
                stator_value_write_string_utf8(
                    std::ptr::null(),
                    buf2.as_mut_ptr() as *mut c_char,
                    buf2.len(),
                    &mut written,
                )
            },
            StatorStatus::StatorStatusInvalidArg
        );
        assert_eq!(written, 0);

        unsafe {
            stator_value_destroy(val);
            stator_value_destroy(num);
        }
    }

    #[test]
    fn test_string_write_utf8_embedded_nul_in_buffer_args_is_respected() {
        let iso = IsolateGuard::new();
        // Three ASCII bytes, no embedded NUL — the API must NOT call strlen
        // on the source side.  We verify by sizing exactly to byte length.
        let src = b"abc";
        let val = unsafe {
            stator_value_new_string(iso.as_ptr(), src.as_ptr() as *const c_char, src.len())
        };
        // Buffer pre-filled with 0xCC sentinels so we can detect over-write.
        let mut buf = [0xCCu8; 5];
        let mut written: usize = 0;
        assert_eq!(
            unsafe {
                stator_value_write_string_utf8(
                    val,
                    buf.as_mut_ptr() as *mut c_char,
                    3,
                    &mut written,
                )
            },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(written, 3);
        assert_eq!(&buf[..3], b"abc");
        // Bytes past the size limit must remain untouched (no NUL terminator).
        assert_eq!(buf[3], 0xCC);
        assert_eq!(buf[4], 0xCC);
        unsafe { stator_value_destroy(val) };
    }

    #[test]
    fn test_object_as_value_round_trip_preserves_identity() {
        let iso = IsolateGuard::new();
        let obj = make_object_with_key_42(iso.as_ptr());

        let val = unsafe { stator_object_as_value(obj) };
        assert!(!val.is_null());
        assert!(unsafe { stator_value_is_object(val) });

        let obj2 = unsafe { stator_value_as_object(val) };
        assert!(!obj2.is_null());
        // Mutating through obj2 must be visible through obj.
        let key = c"shared";
        let v100 = unsafe { stator_value_new_number(iso.as_ptr(), 100.0) };
        assert_eq!(
            unsafe {
                stator_object_set_property(
                    obj2,
                    key.as_ptr(),
                    key.to_bytes().len(),
                    v100 as *const StatorValue,
                )
            },
            StatorStatus::StatorStatusOk
        );
        let mut has = false;
        assert_eq!(
            unsafe {
                stator_object_has_property(obj, key.as_ptr(), key.to_bytes().len(), &mut has)
            },
            StatorStatus::StatorStatusOk
        );
        assert!(has, "mutations on obj2 should be observed through obj");

        // Strict-equals on two ObjectHandle values pointing at same Rc → true.
        let val2 = unsafe { stator_object_as_value(obj2) };
        assert!(unsafe { stator_value_strict_equals(val, val2) });

        unsafe {
            stator_value_destroy(v100);
            stator_value_destroy(val);
            stator_value_destroy(val2);
            stator_object_destroy(obj);
            stator_object_destroy(obj2);
        }
    }

    #[test]
    fn test_value_as_object_returns_null_for_tag_only() {
        let iso = IsolateGuard::new();
        let val = unsafe { stator_value_new_object(iso.as_ptr()) };
        // Tag-only object value has no shared storage → as_object returns null.
        let obj = unsafe { stator_value_as_object(val) };
        assert!(obj.is_null());
        // Also returns null for primitives.
        let n = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        assert!(unsafe { stator_value_as_object(n) }.is_null());
        // And null for the null value pointer itself.
        assert!(unsafe { stator_value_as_object(std::ptr::null()) }.is_null());
        unsafe {
            stator_value_destroy(val);
            stator_value_destroy(n);
        }
    }

    #[test]
    fn test_maybe_get_missing_returns_false() {
        let iso = IsolateGuard::new();
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let key = c"absent";
        let mut out: *mut StatorValue = 0x1 as *mut StatorValue; // sentinel
        assert_eq!(
            unsafe {
                stator_object_get_property(obj, key.as_ptr(), key.to_bytes().len(), &mut out)
            },
            StatorStatus::StatorStatusFalse
        );
        // The out-pointer must always be cleared on non-OK.
        assert!(out.is_null());
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_maybe_get_existing_undefined_is_ok_with_undefined_value() {
        let iso = IsolateGuard::new();
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let key = c"u";
        let undef = unsafe { stator_value_new_undefined(iso.as_ptr()) };
        assert_eq!(
            unsafe {
                stator_object_set_property(
                    obj,
                    key.as_ptr(),
                    key.to_bytes().len(),
                    undef as *const StatorValue,
                )
            },
            StatorStatus::StatorStatusOk
        );
        let mut got: *mut StatorValue = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                stator_object_get_property(obj, key.as_ptr(), key.to_bytes().len(), &mut got)
            },
            StatorStatus::StatorStatusOk
        );
        assert!(!got.is_null());
        assert!(unsafe { stator_value_is_undefined(got) });
        unsafe {
            stator_value_destroy(got);
            stator_value_destroy(undef);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_maybe_set_get_has_delete_for_value_types() {
        let iso = IsolateGuard::new();
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let cases: &[(&str, *mut StatorValue)] = &[
            ("b", unsafe { stator_value_new_boolean(iso.as_ptr(), true) }),
            ("n", unsafe { stator_value_new_number(iso.as_ptr(), 7.0) }),
            ("nu", unsafe { stator_value_new_null(iso.as_ptr()) }),
            ("s", unsafe {
                stator_value_new_string(iso.as_ptr(), c"hi".as_ptr(), 2)
            }),
            ("o", unsafe { stator_value_new_object(iso.as_ptr()) }),
        ];
        for (k, v) in cases {
            let kb = k.as_bytes();
            assert_eq!(
                unsafe {
                    stator_object_set_property(
                        obj,
                        kb.as_ptr() as *const c_char,
                        kb.len(),
                        *v as *const StatorValue,
                    )
                },
                StatorStatus::StatorStatusOk,
                "set {k}"
            );
            let mut has = false;
            assert_eq!(
                unsafe {
                    stator_object_has_property(
                        obj,
                        kb.as_ptr() as *const c_char,
                        kb.len(),
                        &mut has,
                    )
                },
                StatorStatus::StatorStatusOk
            );
            assert!(has, "has {k}");
            let mut got: *mut StatorValue = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    stator_object_get_property(
                        obj,
                        kb.as_ptr() as *const c_char,
                        kb.len(),
                        &mut got,
                    )
                },
                StatorStatus::StatorStatusOk
            );
            assert!(!got.is_null(), "get {k}");
            unsafe { stator_value_destroy(got) };
            let mut deleted = false;
            assert_eq!(
                unsafe {
                    stator_object_delete_property(
                        obj,
                        kb.as_ptr() as *const c_char,
                        kb.len(),
                        &mut deleted,
                    )
                },
                StatorStatus::StatorStatusOk
            );
            assert!(deleted, "delete {k}");
        }
        for (_, v) in cases {
            unsafe { stator_value_destroy(*v) };
        }
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_maybe_set_invalid_args_and_non_utf8_keys() {
        let iso = IsolateGuard::new();
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        let val = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        // Null obj.
        assert_eq!(
            unsafe {
                stator_object_set_property(
                    std::ptr::null_mut(),
                    c"k".as_ptr(),
                    1,
                    val as *const StatorValue,
                )
            },
            StatorStatus::StatorStatusInvalidArg
        );
        // Null val.
        assert_eq!(
            unsafe { stator_object_set_property(obj, c"k".as_ptr(), 1, std::ptr::null()) },
            StatorStatus::StatorStatusInvalidArg
        );
        // Non-UTF-8 key (0xFF is not a valid UTF-8 lead byte).
        let bad = [0xFFu8];
        assert_eq!(
            unsafe {
                stator_object_set_property(
                    obj,
                    bad.as_ptr() as *const c_char,
                    bad.len(),
                    val as *const StatorValue,
                )
            },
            StatorStatus::StatorStatusInvalidArg
        );
        unsafe {
            stator_value_destroy(val);
            stator_object_destroy(obj);
        }
    }

    #[test]
    fn test_maybe_get_null_safety() {
        let mut got: *mut StatorValue = std::ptr::null_mut();
        // Null obj.
        assert_eq!(
            unsafe { stator_object_get_property(std::ptr::null(), c"k".as_ptr(), 1, &mut got) },
            StatorStatus::StatorStatusInvalidArg
        );
        assert!(got.is_null());
        // Null out.
        let iso = IsolateGuard::new();
        let obj = unsafe { stator_object_new(iso.as_ptr()) };
        assert_eq!(
            unsafe { stator_object_get_property(obj, c"k".as_ptr(), 1, std::ptr::null_mut()) },
            StatorStatus::StatorStatusInvalidArg
        );
        unsafe { stator_object_destroy(obj) };
    }

    #[test]
    fn test_value_call_unsupported_for_bytecode_or_primitive() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let n = unsafe { stator_value_new_number(iso.as_ptr(), 1.0) };
        let mut out: *mut StatorValue = std::ptr::null_mut();
        assert_eq!(
            unsafe { stator_value_call(ctx, n, std::ptr::null(), 0, std::ptr::null(), &mut out,) },
            StatorStatus::StatorStatusUnsupported
        );
        assert!(out.is_null());

        // Bytecode-tag function value → also unsupported in this slice.
        let f_tag = unsafe { stator_value_new_function_tag(iso.as_ptr()) };
        assert_eq!(
            unsafe {
                stator_value_call(ctx, f_tag, std::ptr::null(), 0, std::ptr::null(), &mut out)
            },
            StatorStatus::StatorStatusUnsupported
        );

        // Invalid arg: null ctx, negative argc.
        assert_eq!(
            unsafe {
                stator_value_call(
                    std::ptr::null_mut(),
                    n,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    &mut out,
                )
            },
            StatorStatus::StatorStatusInvalidArg
        );
        assert_eq!(
            unsafe { stator_value_call(ctx, n, std::ptr::null(), -1, std::ptr::null(), &mut out) },
            StatorStatus::StatorStatusInvalidArg
        );
        unsafe {
            stator_value_destroy(n);
            stator_value_destroy(f_tag);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_value_call_native_function_template_round_trip() {
        unsafe extern "C" fn triple(info: *const StatorFunctionCallbackInfo) -> *mut StatorValue {
            // SAFETY: `info` is valid for the duration of the callback.
            let iso = unsafe { stator_function_callback_info_get_isolate(info) };
            let argc = unsafe { stator_function_callback_info_length(info) };
            let n = if argc < 1 {
                0.0
            } else {
                let arg = unsafe { stator_function_callback_info_get(info, 0) };
                if arg.is_null() {
                    0.0
                } else {
                    unsafe { stator_value_as_number(arg) }
                }
            };
            unsafe { stator_value_new_number(iso, n * 3.0) }
        }
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), triple) };
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        assert!(unsafe { stator_value_is_function(fn_val) });

        let arg = unsafe { stator_value_new_number(iso.as_ptr(), 7.0) };
        let argv: [*const StatorValue; 1] = [arg as *const StatorValue];
        let mut out: *mut StatorValue = std::ptr::null_mut();
        let status =
            unsafe { stator_value_call(ctx, fn_val, std::ptr::null(), 1, argv.as_ptr(), &mut out) };
        assert_eq!(status, StatorStatus::StatorStatusOk);
        assert!(!out.is_null());
        let mut got = 0.0;
        assert_eq!(
            unsafe { stator_value_get_number(out, &mut got) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(got, 21.0);
        unsafe {
            stator_value_destroy(out);
            stator_value_destroy(arg);
            stator_value_destroy(fn_val);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_value_call_native_function_uses_receiver() {
        unsafe extern "C" fn return_this(
            info: *const StatorFunctionCallbackInfo,
        ) -> *mut StatorValue {
            let iso = unsafe { stator_function_callback_info_get_isolate(info) };
            let this = unsafe { stator_function_callback_info_get_this(info) };
            let n = if this.is_null() {
                0.0
            } else {
                unsafe { stator_value_as_number(this) }
            };
            unsafe { stator_value_new_number(iso, n) }
        }

        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let tmpl = unsafe { stator_function_template_new(iso.as_ptr(), return_this) };
        let fn_val = unsafe { stator_function_template_get_function(tmpl, ctx) };
        let recv = unsafe { stator_value_new_number(iso.as_ptr(), 321.0) };

        let mut out: *mut StatorValue = std::ptr::null_mut();
        let status = unsafe { stator_value_call(ctx, fn_val, recv, 0, std::ptr::null(), &mut out) };
        assert_eq!(status, StatorStatus::StatorStatusOk);
        assert!(!out.is_null());
        let mut got = 0.0;
        assert_eq!(
            unsafe { stator_value_get_number(out, &mut got) },
            StatorStatus::StatorStatusOk
        );
        assert_eq!(got, 321.0);

        unsafe {
            stator_value_destroy(out);
            stator_value_destroy(recv);
            stator_value_destroy(fn_val);
            stator_function_template_destroy(tmpl);
            stator_context_destroy(ctx);
        }
    }

    // ── Inspector FFI ────────────────────────────────────────────────────────

    /// Drain `session`'s outbox into owned `String`s, asserting that each
    /// pop invalidates the previous engine-owned buffer (cached pointer
    /// changes after every successful `next_message`).
    fn drain_inspector_session(session: *mut StatorInspectorSession) -> Vec<String> {
        let mut out = Vec::new();
        loop {
            let mut len: usize = 0;
            let ptr = unsafe { stator_inspector_next_message(session, &mut len) };
            if ptr.is_null() {
                assert_eq!(len, 0, "out_len must be zeroed when no message available");
                break;
            }
            // SAFETY: returned pointer is valid until the next inspector
            // call on this session; we copy out before re-entering.
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            out.push(String::from_utf8(bytes).unwrap());
        }
        out
    }

    #[test]
    fn test_inspector_create_null_ctx_returns_null() {
        let p = unsafe { stator_inspector_create(std::ptr::null_mut()) };
        assert!(p.is_null());
        // Destroy on null is a no-op.
        unsafe { stator_inspector_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn test_inspector_runtime_enable_emits_event_and_ack() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        assert!(!inspector.is_null());
        let session = unsafe { stator_inspector_connect(inspector, 1) };
        assert!(!session.is_null());

        let req = br#"{"id":7,"method":"Runtime.enable","params":{}}"#;
        let rc =
            unsafe { stator_inspector_dispatch(session, req.as_ptr() as *const c_char, req.len()) };
        assert_eq!(rc, 0);
        assert_eq!(unsafe { stator_inspector_pending_count(session) }, 2);

        let msgs = drain_inspector_session(session);
        assert_eq!(msgs.len(), 2);
        let event: serde_json::Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(event["method"], "Runtime.executionContextCreated");
        let ack: serde_json::Value = serde_json::from_str(&msgs[1]).unwrap();
        assert_eq!(ack["id"], 7u64);
        assert!(ack.get("error").is_none());

        // After draining, pending count is zero.
        assert_eq!(unsafe { stator_inspector_pending_count(session) }, 0);

        unsafe {
            stator_inspector_disconnect(inspector, session);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_unknown_method_returns_protocol_error_ok_status() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        let session = unsafe { stator_inspector_connect(inspector, 1) };

        let req = br#"{"id":5,"method":"NoSuch.method","params":{}}"#;
        let rc =
            unsafe { stator_inspector_dispatch(session, req.as_ptr() as *const c_char, req.len()) };
        assert_eq!(rc, 0, "unknown method must report transport success");
        let msgs = drain_inspector_session(session);
        assert_eq!(msgs.len(), 1);
        let resp: serde_json::Value = serde_json::from_str(&msgs[0]).unwrap();
        assert!(resp.get("error").is_some(), "domain error expected");
        assert_eq!(resp["id"], 5u64);

        unsafe {
            stator_inspector_disconnect(inspector, session);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_malformed_json_returns_transport_error() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        let session = unsafe { stator_inspector_connect(inspector, 1) };

        let req = b"not-json";
        let rc =
            unsafe { stator_inspector_dispatch(session, req.as_ptr() as *const c_char, req.len()) };
        assert_eq!(rc, 1, "malformed JSON must surface as transport error");
        let msgs = drain_inspector_session(session);
        assert_eq!(msgs.len(), 1);
        let resp: serde_json::Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(resp["error"]["code"], -32700);

        unsafe {
            stator_inspector_disconnect(inspector, session);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_invalid_utf8_returns_transport_error_and_response() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        let session = unsafe { stator_inspector_connect(inspector, 1) };

        let req = [0xff_u8, 0xfe_u8, b'{'];
        let rc =
            unsafe { stator_inspector_dispatch(session, req.as_ptr() as *const c_char, req.len()) };
        assert_eq!(rc, 1, "invalid UTF-8 must surface as transport error");
        let msgs = drain_inspector_session(session);
        assert_eq!(msgs.len(), 1);
        let resp: serde_json::Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(resp["error"]["code"], -32700);
        assert!(
            resp["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("invalid UTF-8")
        );

        unsafe {
            stator_inspector_disconnect(inspector, session);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_dispatch_rejects_null_pointers() {
        let rc = unsafe { stator_inspector_dispatch(std::ptr::null_mut(), std::ptr::null(), 0) };
        assert_eq!(rc, -1);
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        let session = unsafe { stator_inspector_connect(inspector, 1) };
        let rc2 = unsafe { stator_inspector_dispatch(session, std::ptr::null(), 0) };
        assert_eq!(rc2, -1);
        unsafe {
            stator_inspector_disconnect(inspector, session);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_next_message_null_safety() {
        let mut len: usize = 99;
        let p = unsafe { stator_inspector_next_message(std::ptr::null_mut(), &mut len) };
        assert!(p.is_null());
        // Null session must not touch out_len.
        assert_eq!(len, 99);
        assert_eq!(
            unsafe { stator_inspector_pending_count(std::ptr::null()) },
            0
        );
    }

    #[test]
    fn test_inspector_register_script_fans_out_to_debugger_enabled() {
        let iso = IsolateGuard::new();
        let ctx = unsafe { stator_context_new(iso.as_ptr()) };
        let inspector = unsafe { stator_inspector_create(ctx) };
        let s1 = unsafe { stator_inspector_connect(inspector, 1) };
        let s2 = unsafe { stator_inspector_connect(inspector, 2) };

        // s1 enables Debugger; s2 does not.
        let enable = br#"{"id":1,"method":"Debugger.enable","params":{}}"#;
        let rc = unsafe {
            stator_inspector_dispatch(s1, enable.as_ptr() as *const c_char, enable.len())
        };
        assert_eq!(rc, 0);
        // Drain the ack on s1 so the outbox starts empty before
        // register_script runs.
        let _ = drain_inspector_session(s1);

        let src = b"var x = 1;";
        let id = unsafe {
            stator_inspector_register_script(
                inspector,
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert_eq!(id, 1);

        assert_eq!(unsafe { stator_inspector_pending_count(s1) }, 1);
        assert_eq!(unsafe { stator_inspector_pending_count(s2) }, 0);

        let msgs = drain_inspector_session(s1);
        let ev: serde_json::Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(ev["method"], "Debugger.scriptParsed");
        assert_eq!(ev["params"]["scriptId"], "1");

        // Second call increments the ID.
        let id2 = unsafe {
            stator_inspector_register_script(
                inspector,
                std::ptr::null_mut(),
                src.as_ptr() as *const c_char,
                src.len(),
            )
        };
        assert_eq!(id2, 2);

        unsafe {
            stator_inspector_disconnect(inspector, s1);
            stator_inspector_disconnect(inspector, s2);
            stator_inspector_destroy(inspector);
            stator_context_destroy(ctx);
        }
    }

    #[test]
    fn test_inspector_register_script_rejects_null_inspector() {
        let id = unsafe {
            stator_inspector_register_script(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                b"x".as_ptr() as *const c_char,
                1,
            )
        };
        assert_eq!(id, 0);
    }
}

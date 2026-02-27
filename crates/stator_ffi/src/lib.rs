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

use stator_core::gc::heap::Heap;

/// An opaque isolate handle.
///
/// An isolate is an independent instance of the Stator engine with its own
/// heap and root set.  Isolates are not thread-safe by default; access from
/// multiple threads requires external synchronisation.
pub struct StatorIsolate {
    heap: Heap,
}

/// Create a new isolate.
///
/// The returned pointer must eventually be passed to [`stator_isolate_destroy`]
/// to free all associated resources.  Returns a null pointer on allocation
/// failure (extremely unlikely in practice).
#[unsafe(no_mangle)]
pub extern "C" fn stator_isolate_create() -> *mut StatorIsolate {
    Box::into_raw(Box::new(StatorIsolate { heap: Heap::new() }))
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

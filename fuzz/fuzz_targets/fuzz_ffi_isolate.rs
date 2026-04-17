#![no_main]

//! Fuzz the isolate/context lifecycle at the FFI boundary.
//!
//! Exercises random sequences of:
//!   - isolate create / enter / exit / dispose
//!   - context new / enter / exit / destroy
//!   - isolate GC trigger
//!   - embedder data set/get
//!
//! Verifies that no use-after-free, double-free, or memory corruption occurs
//! for any valid operation ordering.

use libfuzzer_sys::fuzz_target;
use stator_jse_ffi::{
    stator_context_destroy, stator_context_enter, stator_context_exit, stator_context_new,
    stator_isolate_dispose, stator_isolate_enter, stator_isolate_exit, stator_isolate_gc,
    stator_isolate_get_current_context, stator_isolate_get_data, stator_isolate_new,
    stator_isolate_set_data,
};

fuzz_target!(|data: &[u8]| {
    // Interpret every byte as an operation to perform.
    //
    //  op & 0x0f → operation selector
    //    0  create isolate (at most MAX_ISOLATES live at once)
    //    1  enter the current isolate
    //    2  exit the current isolate
    //    3  trigger a GC on the current isolate
    //    4  set embedder data slot (slot = (byte >> 4) & 0x3, value = non-null sentinel)
    //    5  get embedder data slot (slot = (byte >> 4) & 0x3)
    //    6  create a context on the current isolate
    //    7  enter the current context
    //    8  exit the current context
    //    9  destroy the current context
    //   10  dispose the current isolate (replaces with previous if any)
    //   other → no-op

    const MAX_ISOLATES: usize = 4;
    const MAX_CONTEXTS: usize = 4;

    let mut isolates = Vec::new();
    // index of "current" isolate (last created)
    let mut ctx_stack: Vec<*mut stator_jse_ffi::StatorContext> = Vec::new();

    for &byte in data {
        let op = byte & 0x0f;

        // Pick the last live isolate as "current", if any.
        let iso = isolates.last().copied().unwrap_or(std::ptr::null_mut());

        match op {
            0 if isolates.len() < MAX_ISOLATES => {
                // Create a new isolate (bounded to avoid unbounded memory use).
                let i = stator_isolate_new();
                assert!(!i.is_null(), "stator_isolate_new must not return null");
                isolates.push(i);
            }
            1 if !iso.is_null() => {
                // Enter the current isolate.
                // SAFETY: `iso` is a live pointer we own.
                unsafe { stator_isolate_enter(iso) };
            }
            2 if !iso.is_null() => {
                // Exit the current isolate.
                // SAFETY: `iso` is a live pointer we own.
                unsafe { stator_isolate_exit(iso) };
            }
            3 if !iso.is_null() => {
                // Trigger GC on the current isolate.
                // SAFETY: `iso` is a live pointer we own.
                unsafe { stator_isolate_gc(iso) };
            }
            4 if !iso.is_null() => {
                // Set embedder data at a small slot index.
                let slot = u32::from((byte >> 4) & 0x3);
                // Use the address of a local as a well-known pointer for
                // round-trip verification; the engine only stores it, never
                // dereferences it.
                let mut sentinel: u8 = 42;
                let sentinel_ptr = &raw mut sentinel as *mut std::ffi::c_void;
                // SAFETY: `iso` is a live pointer we own; `sentinel_ptr` is
                // a valid stack address that outlives this block.
                unsafe { stator_isolate_set_data(iso, slot, sentinel_ptr) };
                // Immediately verify the round-trip.
                // SAFETY: same requirements as above.
                let got = unsafe { stator_isolate_get_data(iso, slot) };
                assert_eq!(got, sentinel_ptr, "embedder data round-trip must be lossless");
            }
            5 if !iso.is_null() => {
                // Get embedder data (slot may not have been set; should return null or sentinel).
                let slot = u32::from((byte >> 4) & 0x3);
                // SAFETY: `iso` is a live pointer we own.
                let _got = unsafe { stator_isolate_get_data(iso, slot) };
                // No invariant to assert; just must not crash.
            }
            6 if !iso.is_null() && ctx_stack.len() < MAX_CONTEXTS => {
                // Create a context on the current isolate.
                // SAFETY: `iso` is a live pointer we own.
                let ctx = unsafe { stator_context_new(iso) };
                if !ctx.is_null() {
                    ctx_stack.push(ctx);
                    // The context must now be reported as current.
                    // SAFETY: `iso` is a live pointer we own.
                    let current = unsafe { stator_isolate_get_current_context(iso) };
                    assert_eq!(
                        current, ctx,
                        "newly created context must be the current context"
                    );
                }
            }
            7 => {
                // Enter the current context.
                if let Some(&ctx) = ctx_stack.last() {
                    // SAFETY: `ctx` is a live pointer we own.
                    unsafe { stator_context_enter(ctx) };
                }
            }
            8 => {
                // Exit the current context.
                if let Some(&ctx) = ctx_stack.last() {
                    // SAFETY: `ctx` is a live pointer we own.
                    unsafe { stator_context_exit(ctx) };
                }
            }
            9 => {
                // Destroy the most recently created context.
                if let Some(ctx) = ctx_stack.pop() {
                    // SAFETY: `ctx` is a live pointer we own; we remove it from
                    // the stack before destroying it so it cannot be used again.
                    unsafe { stator_context_destroy(ctx) };
                }
            }
            10 if !iso.is_null() => {
                // Dispose the most recently created isolate.
                //
                // First destroy all contexts associated with it to avoid
                // accessing a freed isolate pointer from inside the context.
                // Destroy all contexts that were created on this isolate.
                for ctx in ctx_stack.drain(..) {
                    // SAFETY: each context was created on `iso` which is
                    // still live at this point.
                    unsafe { stator_context_destroy(ctx) };
                }
                isolates.pop();
                // SAFETY: `iso` was created by `stator_isolate_new` and has
                // not been disposed yet; all contexts on it are already gone.
                unsafe { stator_isolate_dispose(iso) };
            }
            _ => {}
        }
    }

    // Cleanup: destroy contexts then dispose isolates in LIFO order.
    for ctx in ctx_stack.drain(..) {
        // SAFETY: each ctx in the stack is a live pointer we own.
        unsafe { stator_context_destroy(ctx) };
    }
    for iso in isolates.drain(..) {
        // SAFETY: each isolate in the vec is a live pointer we own.
        unsafe { stator_isolate_dispose(iso) };
    }
});

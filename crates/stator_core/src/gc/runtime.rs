//! Thread-local GC runtime: allocation, collection, and root management.
//!
//! This module provides the public API for wiring the GC heap infrastructure
//! into the JavaScript runtime.  All state is thread-local (the engine is
//! single-threaded per isolate).
//!
//! # Usage
//!
//! ```rust,ignore
//! use stator_core::gc::runtime::{gc_alloc, gc_collect, gc_stats};
//!
//! let ptr = gc_alloc::<MyGcType>();
//! // ... use the allocated object ...
//! gc_collect(); // trigger a minor GC
//! let stats = gc_stats();
//! println!("allocated: {}, collections: {}", stats.bytes_allocated, stats.collections);
//! ```

use std::alloc::Layout;
use std::cell::RefCell;

use crate::gc::gc_ptr::{GcObject, GcPtr};
use crate::gc::heap::Heap;
use crate::gc::immix::{ImmixSpace, Tlab};
use crate::objects::heap_object::HeapObject;

thread_local! {
    /// The thread-local GC heap instance.
    static GC_HEAP: RefCell<Heap> = RefCell::new(Heap::new());

    /// Simple allocation counter for triggering GC heuristics.
    static GC_STATS: RefCell<GcRuntimeStats> = RefCell::new(GcRuntimeStats::new());

    /// Thread-local Immix allocation buffer.
    static IMMIX_TLAB: RefCell<Tlab> = RefCell::new(Tlab::new());

    /// Shared Immix block space (thread-local for single-threaded engine).
    static IMMIX_SPACE: RefCell<ImmixSpace> = RefCell::new(ImmixSpace::with_defaults());
}

/// GC runtime statistics.
#[derive(Debug, Clone)]
pub struct GcRuntimeStats {
    /// Total bytes allocated since the last collection.
    pub bytes_allocated: usize,
    /// Total number of minor collections.
    pub collections: u64,
    /// Allocation threshold (bytes) that triggers an automatic minor GC.
    pub gc_threshold: usize,
}

impl GcRuntimeStats {
    /// Default allocation threshold: 1 MiB.
    const DEFAULT_THRESHOLD: usize = 1024 * 1024;

    fn new() -> Self {
        Self {
            bytes_allocated: 0,
            collections: 0,
            gc_threshold: Self::DEFAULT_THRESHOLD,
        }
    }
}

/// Allocate a GC-managed object of type `T` on the thread-local heap.
///
/// Returns `Some(GcPtr<T>)` if allocation succeeds, `None` if the heap is
/// exhausted even after a minor GC attempt.
///
/// # Safety
///
/// The returned `GcPtr` is **unrooted**.  The caller must root it (via
/// `HandleScope` or `PersistentRoots`) before any code path that might
/// trigger garbage collection.
pub fn gc_alloc<T: GcObject>() -> Option<GcPtr<T>> {
    let layout = Layout::new::<T>();

    GC_HEAP.with(|heap| {
        let ptr = heap.borrow_mut().allocate(layout);
        if ptr.is_null() {
            return None;
        }

        // Update stats and check if we should auto-collect.
        GC_STATS.with(|stats| {
            let mut s = stats.borrow_mut();
            s.bytes_allocated += layout.size();
        });

        // SAFETY: ptr is non-null, properly aligned, and zero-initialized by
        // Heap::allocate.  We cast it to T* which is valid because T has a
        // HeapObject header as its first field (GcObject contract).
        Some(unsafe { GcPtr::from_heap_object_ptr(ptr) })
    })
}

/// Allocate raw bytes on the thread-local heap.
///
/// Returns a raw `*mut HeapObject` pointer, or null if allocation fails.
/// This is the low-level interface for allocation; prefer [`gc_alloc`] for
/// typed allocation.
pub fn gc_alloc_raw(layout: Layout) -> *mut HeapObject {
    GC_HEAP.with(|heap| {
        let ptr = heap.borrow_mut().allocate(layout);

        if !ptr.is_null() {
            GC_STATS.with(|stats| {
                let mut s = stats.borrow_mut();
                s.bytes_allocated += layout.size();
            });
        }

        ptr
    })
}

/// Trigger a minor (young-generation) garbage collection.
///
/// This swaps the semi-space halves and resets the nursery.  Objects that
/// are still reachable from roots should be rooted via `HandleScope` or
/// `PersistentRoots` before calling this.
pub fn gc_collect() {
    GC_HEAP.with(|heap| {
        heap.borrow_mut().collect();
    });
    GC_STATS.with(|stats| {
        let mut s = stats.borrow_mut();
        s.bytes_allocated = 0;
        s.collections += 1;
    });
}

/// Check whether the allocation counter has exceeded the GC threshold,
/// and if so, trigger a minor collection.
///
/// This is intended to be called periodically (e.g. at loop back-edges or
/// function entry) to bound pause times and memory usage.
pub fn gc_safepoint() {
    let should_collect = GC_STATS.with(|stats| {
        let s = stats.borrow();
        s.bytes_allocated >= s.gc_threshold
    });
    if should_collect {
        gc_collect();
    }
}

/// Returns a snapshot of the GC runtime statistics.
pub fn gc_stats() -> GcRuntimeStats {
    GC_STATS.with(|stats| stats.borrow().clone())
}

/// Set the allocation threshold (in bytes) that triggers automatic minor GC.
pub fn gc_set_threshold(threshold: usize) {
    GC_STATS.with(|stats| {
        stats.borrow_mut().gc_threshold = threshold;
    });
}

/// Access the thread-local heap directly for advanced operations.
///
/// # Safety
///
/// The caller must not trigger re-entrant allocation or collection from
/// within the closure `f`.
pub fn with_heap<R>(f: impl FnOnce(&mut Heap) -> R) -> R {
    GC_HEAP.with(|heap| f(&mut heap.borrow_mut()))
}

/// Allocate from the Immix block-based allocator via the thread-local TLAB.
///
/// Returns a pointer to the allocated bytes, or `None` if the Immix space
/// cannot provide a block.  The returned pointer is **not** typed; callers
/// must initialise the object header.
pub fn gc_alloc_immix(layout: Layout) -> Option<*mut u8> {
    IMMIX_TLAB.with(|tlab| {
        IMMIX_SPACE.with(|space| {
            let ptr = tlab
                .borrow_mut()
                .allocate(layout, &mut space.borrow_mut())?;

            GC_STATS.with(|stats| {
                let mut s = stats.borrow_mut();
                s.bytes_allocated += layout.size();
            });

            Some(ptr)
        })
    })
}

/// Access the thread-local Immix space for collection operations.
pub fn with_immix_space<R>(f: impl FnOnce(&mut ImmixSpace) -> R) -> R {
    IMMIX_SPACE.with(|space| f(&mut space.borrow_mut()))
}

/// Flush the thread-local TLAB back to the Immix space (e.g. before GC).
pub fn flush_immix_tlab() {
    IMMIX_TLAB.with(|tlab| {
        IMMIX_SPACE.with(|space| {
            tlab.borrow_mut().flush(&mut space.borrow_mut());
        });
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_stats_initial() {
        // Reset by creating a new stats instance.
        let stats = GcRuntimeStats::new();
        assert_eq!(stats.bytes_allocated, 0);
        assert_eq!(stats.collections, 0);
        assert_eq!(stats.gc_threshold, GcRuntimeStats::DEFAULT_THRESHOLD);
    }

    #[test]
    fn test_gc_alloc_raw_returns_non_null() {
        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr = gc_alloc_raw(layout);
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_gc_collect_resets_allocated_bytes() {
        // Allocate some bytes.
        let layout = Layout::from_size_align(128, 8).unwrap();
        let _ = gc_alloc_raw(layout);
        // Collect should reset the counter.
        gc_collect();
        let stats = gc_stats();
        assert_eq!(stats.bytes_allocated, 0);
        assert!(stats.collections >= 1);
    }

    #[test]
    fn test_gc_safepoint_does_not_collect_below_threshold() {
        let initial = gc_stats().collections;
        gc_safepoint();
        assert_eq!(gc_stats().collections, initial);
    }

    #[test]
    fn test_gc_set_threshold() {
        gc_set_threshold(512);
        assert_eq!(gc_stats().gc_threshold, 512);
        // Reset to default.
        gc_set_threshold(GcRuntimeStats::DEFAULT_THRESHOLD);
    }
}

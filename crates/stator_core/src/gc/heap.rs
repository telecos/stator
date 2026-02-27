use std::alloc::{Layout, alloc, dealloc};
use std::mem::align_of;

pub use crate::objects::heap_object::HeapObject;

/// A contiguous, fixed-size memory region used as a semi-space or generation.
///
/// Allocation is a simple pointer-bump; deallocation happens by resetting the
/// cursor to the start of the region (`reset`).
pub struct MemoryRegion {
    base: *mut u8,
    capacity: usize,
    used: usize,
}

// SAFETY: `MemoryRegion` owns its backing allocation and is never aliased.
unsafe impl Send for MemoryRegion {}

impl MemoryRegion {
    /// Allocate a new region of `capacity` bytes.
    ///
    /// # Panics
    /// Panics if the system allocator returns a null pointer.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "region capacity must be non-zero");
        // SAFETY: capacity > 0 and alignment 8 is a power of two, so the
        // layout is valid.  We abort on alloc failure below.
        let layout = Layout::from_size_align(capacity, 8).expect("valid layout");
        let base = unsafe { alloc(layout) };
        assert!(!base.is_null(), "heap region allocation failed");
        Self {
            base,
            capacity,
            used: 0,
        }
    }

    /// Bump-allocate `layout` bytes, returning a pointer to the start of the
    /// allocation, or `None` if the region does not have enough free space.
    pub fn bump_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        let align = layout.align();
        let size = layout.size();
        // Round up `used` to the required alignment.
        let current = self.base as usize + self.used;
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let end = aligned.checked_add(size)?;
        let new_used = end - self.base as usize;
        if new_used > self.capacity {
            return None;
        }
        self.used = new_used;
        Some(aligned as *mut u8)
    }

    /// Reset the region's allocation cursor to the beginning.
    ///
    /// All pointers into this region are invalidated after a call to `reset`.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Number of bytes currently in use.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: `base` was allocated with the same layout in `new`.
            let layout = Layout::from_size_align(self.capacity, 8).expect("valid layout");
            unsafe { dealloc(self.base, layout) };
        }
    }
}

// Default sizes for the two generations.
const YOUNG_SPACE_SIZE: usize = 8 * 1024 * 1024; //  8 MiB — nursery
const OLD_SPACE_SIZE: usize = 64 * 1024 * 1024; // 64 MiB — tenured objects

/// The garbage-collected heap, composed of a young generation (nursery) and
/// an old generation (tenured space).
///
/// New allocations land in `young_space`.  When the nursery is full, `collect`
/// must be called; surviving objects will eventually be promoted to `old_space`.
pub struct Heap {
    /// The young (nursery) generation where new allocations land.
    pub young_space: MemoryRegion,
    /// The old (tenured) generation for long-lived objects.
    pub old_space: MemoryRegion,
}

impl Heap {
    /// Create a heap with default nursery and old-generation sizes.
    pub fn new() -> Self {
        Self {
            young_space: MemoryRegion::new(YOUNG_SPACE_SIZE),
            old_space: MemoryRegion::new(OLD_SPACE_SIZE),
        }
    }

    /// Bump-allocate enough memory for `layout` bytes in the young generation
    /// and return a pointer to a zero-initialised `HeapObject` header.
    ///
    /// Returns a null pointer when the young space is exhausted; the caller
    /// is responsible for triggering a collection and retrying.
    pub fn allocate(&mut self, layout: Layout) -> *mut HeapObject {
        // Ensure the allocation is at least as aligned as `HeapObject`.
        let layout = layout
            .align_to(align_of::<HeapObject>())
            .expect("alignment adjustment failed")
            .pad_to_align();

        match self.young_space.bump_alloc(layout) {
            Some(ptr) => {
                // Zero-initialise so map_word starts as null (no map set yet).
                // SAFETY: `ptr` is freshly allocated with `layout` bytes.
                unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
                ptr as *mut HeapObject
            }
            None => std::ptr::null_mut(),
        }
    }

    /// Perform a minor (young-generation) collection.
    ///
    /// In this initial skeleton the young space is simply reset.  A full
    /// copying/tracing GC will be layered on top once the object model and
    /// the `Trace` infrastructure are complete.
    pub fn collect(&mut self) {
        self.young_space.reset();
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_returns_non_null_for_small_object() {
        let mut heap = Heap::new();
        let layout = Layout::new::<HeapObject>();
        let ptr = heap.allocate(layout);
        assert!(!ptr.is_null());
    }

    #[test]
    fn collect_resets_young_space() {
        let mut heap = Heap::new();
        let layout = Layout::new::<HeapObject>();
        heap.allocate(layout);
        let used_before = heap.young_space.used();
        assert!(used_before > 0);
        heap.collect();
        assert_eq!(heap.young_space.used(), 0);
    }

    #[test]
    fn allocate_returns_null_when_young_space_full() {
        let mut heap = Heap::new();
        // Fill the young space with 1-byte allocations to exhaust it.
        let layout = Layout::from_size_align(YOUNG_SPACE_SIZE, 1).unwrap();
        let first = heap.allocate(layout);
        assert!(!first.is_null());
        // Next allocation must fail.
        let second = heap.allocate(Layout::new::<HeapObject>());
        assert!(second.is_null());
    }
}

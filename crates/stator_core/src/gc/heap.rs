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

    /// Raw base pointer of this region.
    ///
    /// Used by the scavenger to determine whether a given heap pointer falls
    /// within this region's address range.
    pub(crate) fn base_ptr(&self) -> *mut u8 {
        self.base
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

/// Capacity of each semi-space half in the young (nursery) generation.
///
/// Value: 4 MiB (4 × 1024 × 1024 = 4,194,304 bytes).
/// The total young-generation footprint is `2 × YOUNG_SEMI_SPACE_SIZE` because
/// Cheney's algorithm requires a mirrored *to-space* of equal size.
const YOUNG_SEMI_SPACE_SIZE: usize = 4 * 1024 * 1024;

/// Capacity of the old (tenured) generation (64 MiB).
const OLD_SPACE_SIZE: usize = 64 * 1024 * 1024;

/// Objects whose adjusted allocation size meets or exceeds this threshold
/// bypass the nursery and are placed directly in the [`LargeObjectSpace`].
///
/// Set equal to [`YOUNG_SEMI_SPACE_SIZE`] so that an object guaranteed never
/// to fit in a single semi-space half is routed to the LOS immediately.
const LARGE_OBJECT_THRESHOLD: usize = YOUNG_SEMI_SPACE_SIZE;

/// A Cheney-style semi-space for the young (nursery) generation.
///
/// The semi-space is split into two equally-sized halves: *from-space* and
/// *to-space*.  New allocations always land in the from-space via a bump
/// pointer.  During a minor GC the collector copies surviving objects into the
/// to-space, then swaps the two halves so the old from-space becomes the new
/// (empty) to-space, ready to receive survivors in the next cycle.
///
/// In this skeletal implementation live-object copying is not yet performed
/// (the root-scanning infrastructure is incomplete).  The from-space is simply
/// reset after a swap, treating every nursery object as dead — a safe
/// approximation for a nursery whose live set is fully re-rooted on every
/// minor GC.
pub struct SemiSpace {
    from_space: MemoryRegion,
    to_space: MemoryRegion,
}

impl SemiSpace {
    /// Create a semi-space with two halves each of `semi_size` bytes.
    pub fn new(semi_size: usize) -> Self {
        Self {
            from_space: MemoryRegion::new(semi_size),
            to_space: MemoryRegion::new(semi_size),
        }
    }

    /// Bump-allocate `layout` bytes from the active from-space.
    ///
    /// Returns `None` when the from-space is exhausted.
    pub fn bump_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        self.from_space.bump_alloc(layout)
    }

    /// Bump-allocate `layout` bytes into the **to-space**.
    ///
    /// Used by the scavenger to copy live objects from the from-space into
    /// the to-space during a Cheney collection.  Returns `None` when the
    /// to-space is exhausted.
    pub(crate) fn bump_alloc_to_space(&mut self, layout: Layout) -> Option<*mut u8> {
        self.to_space.bump_alloc(layout)
    }

    /// Returns `true` if `ptr` points into the active from-space.
    ///
    /// Used by the scavenger to decide whether a pointer needs to be copied.
    pub fn is_in_from_space(&self, ptr: *mut HeapObject) -> bool {
        let base = self.from_space.base_ptr() as usize;
        let end = base + self.from_space.capacity();
        let addr = ptr as usize;
        addr >= base && addr < end
    }

    /// Bytes currently used in the to-space (live objects copied so far).
    pub(crate) fn to_space_used(&self) -> usize {
        self.to_space.used()
    }

    /// Raw base pointer of the to-space region.
    ///
    /// Used by the Cheney BFS scan to walk newly-copied objects.
    pub(crate) fn to_space_base(&self) -> *mut u8 {
        self.to_space.base_ptr()
    }

    /// Perform a Cheney-style semi-space collection.
    ///
    /// Swaps the from-space and to-space halves, then resets the new
    /// to-space (the old from-space) so it is ready to receive copies in the
    /// next scavenge cycle.
    ///
    /// After this call the new from-space contains whatever was written into
    /// the old to-space (i.e. the live objects copied by the scavenger),
    /// while the new to-space is empty.
    pub fn collect(&mut self) {
        std::mem::swap(&mut self.from_space, &mut self.to_space);
        // Reset the *new* to-space (which is the old from-space that held
        // both live and dead nursery objects).
        self.to_space.reset();
    }

    /// Bytes currently allocated in the active from-space.
    pub fn used(&self) -> usize {
        self.from_space.used()
    }

    /// Capacity of each semi-space half in bytes.
    pub fn capacity(&self) -> usize {
        self.from_space.capacity()
    }
}

/// The old (tenured) generation: a simple bump allocator for long-lived objects.
///
/// Objects are promoted here from the young generation when they survive
/// enough minor collections.  Compaction and major GC are not yet implemented.
pub struct OldSpace {
    region: MemoryRegion,
}

impl OldSpace {
    /// Create an old-space region of `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            region: MemoryRegion::new(capacity),
        }
    }

    /// Bump-allocate `layout` bytes, returning a pointer or `None` if full.
    pub fn bump_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        self.region.bump_alloc(layout)
    }

    /// Bytes currently in use.
    pub fn used(&self) -> usize {
        self.region.used()
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.region.capacity()
    }
}

/// Space for objects too large to fit in the young generation's semi-space.
///
/// Each large object is backed by an individual allocation from the system
/// allocator and tracked in an internal list.  Objects are reclaimed when the
/// space is dropped.
pub struct LargeObjectSpace {
    objects: Vec<(*mut u8, Layout)>,
}

// SAFETY: The raw pointers stored here are owned exclusively by this space;
// no aliases exist outside of it.
unsafe impl Send for LargeObjectSpace {}

impl LargeObjectSpace {
    /// Create an empty large-object space.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Allocate a large object for the given `layout`.
    ///
    /// Returns a pointer to zero-initialised memory, or a null pointer when
    /// the system allocator fails.
    pub fn allocate(&mut self, layout: Layout) -> *mut HeapObject {
        // SAFETY: `layout.align()` is guaranteed to be a non-zero power of two
        // by the `Layout` type invariant.  `layout.size() > 0` is a documented
        // precondition of `alloc`; all call sites in this crate supply layouts
        // derived from `HeapObject`-aligned base layouts that are never zero-sized.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: `ptr` is valid and `layout.size()` bytes are accessible.
        unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
        // Record the allocation size in the header so the scavenger can copy
        // the object without additional size information.
        // SAFETY: ptr is non-null, properly aligned, and zero-initialised.
        unsafe { (*(ptr as *mut HeapObject)).init_alloc_size(layout.size() as u32) };
        self.objects.push((ptr, layout));
        ptr as *mut HeapObject
    }

    /// Number of large objects currently tracked in this space.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
}

impl Default for LargeObjectSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LargeObjectSpace {
    fn drop(&mut self) {
        for (ptr, layout) in self.objects.drain(..) {
            // SAFETY: each pointer was allocated with its stored layout.
            unsafe { dealloc(ptr, layout) };
        }
    }
}

/// The garbage-collected heap, composed of a young generation (nursery),
/// an old generation, and a large-object space.
///
/// New allocations land first in the young generation's from-space.  When the
/// from-space is full, a minor collection is triggered automatically and the
/// allocation is retried.  Objects too large for the semi-space are forwarded
/// directly to the large-object space.
pub struct Heap {
    /// Young (nursery) generation: a Cheney-style semi-space.
    pub young_space: SemiSpace,
    /// Old (tenured) generation: a bump allocator for promoted objects.
    pub old_space: OldSpace,
    /// Space for objects that are too large for the nursery.
    pub large_object_space: LargeObjectSpace,
}

impl Heap {
    /// Create a heap with default generation sizes.
    pub fn new() -> Self {
        Self {
            young_space: SemiSpace::new(YOUNG_SEMI_SPACE_SIZE),
            old_space: OldSpace::new(OLD_SPACE_SIZE),
            large_object_space: LargeObjectSpace::new(),
        }
    }

    /// Allocate a zero-initialised [`HeapObject`] header for the given layout.
    ///
    /// The allocation follows this policy:
    ///
    /// 1. Large objects (adjusted size ≥ [`LARGE_OBJECT_THRESHOLD`]) are
    ///    placed directly in the large-object space.
    /// 2. Small objects are bump-allocated in the young from-space.
    /// 3. If the from-space is exhausted, a minor collection is triggered and
    ///    the allocation is retried once.
    /// 4. A null pointer is returned only if the retry also fails (i.e., the
    ///    requested size exceeds the semi-space capacity).
    pub fn allocate(&mut self, layout: Layout) -> *mut HeapObject {
        // Ensure the allocation is at least as aligned as `HeapObject`.
        let layout = layout
            .align_to(align_of::<HeapObject>())
            .expect("alignment adjustment failed")
            .pad_to_align();

        // Large objects bypass the nursery entirely.
        if layout.size() >= LARGE_OBJECT_THRESHOLD {
            return self.large_object_space.allocate(layout);
        }

        // Fast path: bump-allocate in the young from-space.
        if let Some(ptr) = self.young_space.bump_alloc(layout) {
            // SAFETY: `ptr` is freshly allocated with `layout` bytes.
            unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
            // SAFETY: ptr is non-null, properly aligned, and zero-initialised.
            unsafe { (*(ptr as *mut HeapObject)).init_alloc_size(layout.size() as u32) };
            return ptr as *mut HeapObject;
        }

        // Slow path: nursery is full — run a minor GC and retry once.
        self.collect();
        match self.young_space.bump_alloc(layout) {
            Some(ptr) => {
                // SAFETY: `ptr` is freshly allocated with `layout` bytes.
                unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
                // SAFETY: ptr is non-null, properly aligned, and zero-initialised.
                unsafe { (*(ptr as *mut HeapObject)).init_alloc_size(layout.size() as u32) };
                ptr as *mut HeapObject
            }
            None => std::ptr::null_mut(),
        }
    }

    /// Perform a minor (young-generation) collection.
    ///
    /// Swaps the semi-space halves and resets the old from-space (now the
    /// to-space), making it ready for the next scavenge cycle.  This
    /// "collect-all-as-dead" variant is used when no root information is
    /// available; for a proper live-object copy use
    /// [`scavenge_with_roots`][Self::scavenge_with_roots].
    pub fn collect(&mut self) {
        self.young_space.collect();
    }

    /// Perform a minor GC using the Cheney scavenger with explicit roots.
    ///
    /// Each entry in `roots` is a `*mut *mut HeapObject` — a mutable pointer
    /// to a root-pointer slot.  After the scavenge, each slot is updated to
    /// the object's new address (in the young to-space or, for promoted
    /// objects, in old-space).
    ///
    /// The `remembered_set` provides additional roots: old-space objects that
    /// hold references into the young generation.  The set is cleared at the
    /// end of the cycle.
    ///
    /// # Safety
    /// Every pointer reachable from `roots` or `remembered_set` must be a
    /// valid, aligned, live heap object currently residing in the young
    /// generation's from-space.
    pub unsafe fn scavenge_with_roots(
        &mut self,
        roots: &mut [*mut *mut HeapObject],
        remembered_set: &mut crate::gc::scavenger::RememberedSet,
    ) {
        // SAFETY: caller upholds the preconditions.
        unsafe {
            crate::gc::scavenger::Scavenger::new(&mut self.young_space, &mut self.old_space)
                .scavenge(roots, remembered_set);
        }
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
    fn allocate_triggers_minor_gc_and_retries_when_young_space_full() {
        let mut heap = Heap::new();
        let layout = Layout::new::<HeapObject>();
        // Exhaust the young from-space directly (bypassing the auto-GC path).
        while heap.young_space.bump_alloc(layout).is_some() {}
        // heap.allocate() must detect exhaustion, run a minor GC, and succeed.
        let ptr = heap.allocate(layout);
        assert!(
            !ptr.is_null(),
            "allocate must succeed after implicit minor GC"
        );
    }

    #[test]
    fn semi_space_collect_resets_from_space() {
        let mut ss = SemiSpace::new(1024);
        let layout = Layout::from_size_align(64, 8).unwrap();
        assert!(ss.bump_alloc(layout).is_some());
        assert!(ss.used() > 0);
        ss.collect();
        assert_eq!(ss.used(), 0);
        // Fresh allocation is possible in the new from-space.
        assert!(ss.bump_alloc(layout).is_some());
    }

    #[test]
    fn old_space_bump_alloc_and_capacity() {
        let mut os = OldSpace::new(4096);
        let layout = Layout::from_size_align(64, 8).unwrap();
        assert!(os.bump_alloc(layout).is_some());
        assert_eq!(os.used(), 64);
        assert_eq!(os.capacity(), 4096);
    }

    #[test]
    fn large_object_space_allocates_and_tracks_objects() {
        let mut los = LargeObjectSpace::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = los.allocate(layout);
        assert!(!ptr.is_null());
        assert_eq!(los.object_count(), 1);
    }

    #[test]
    fn large_object_bypasses_young_space() {
        let mut heap = Heap::new();
        // An allocation at or above LARGE_OBJECT_THRESHOLD is routed to LOS.
        let large_layout = Layout::from_size_align(LARGE_OBJECT_THRESHOLD, 8).unwrap();
        let young_used_before = heap.young_space.used();
        let ptr = heap.allocate(large_layout);
        assert!(!ptr.is_null());
        assert_eq!(
            heap.young_space.used(),
            young_used_before,
            "young space must be untouched for large objects"
        );
        assert_eq!(heap.large_object_space.object_count(), 1);
    }

    #[test]
    fn allocate_until_young_full_then_verify_after_collection() {
        let mut heap = Heap::new();
        let layout = Layout::new::<HeapObject>();
        // Fill the young from-space by bumping directly.
        while heap.young_space.bump_alloc(layout).is_some() {}
        assert!(
            heap.young_space.used() > 0,
            "from-space must be non-empty after fill"
        );
        // Trigger a manual collection.
        heap.collect();
        assert_eq!(
            heap.young_space.used(),
            0,
            "young space must be empty after collection"
        );
        // Allocation succeeds again after collection.
        let ptr = heap.allocate(layout);
        assert!(!ptr.is_null(), "allocation after collection must succeed");
    }
}

//! Mark-Sweep-Compact collector for the old (tenured) generation.
//!
//! # Phases
//!
//! 1. **Mark** – Starting from explicit roots, a grey-stack BFS walk marks
//!    every old-space object reachable from those roots.
//!
//! 2. **Sweep** – Every object that was *not* marked is treated as garbage.
//!    Its byte range is recorded in the returned free-region list.  The
//!    allocation cursor is not yet changed in this phase.
//!
//! 3. **Compact** – Live objects are slid toward the base of the old-space
//!    region, filling in all holes.  Root slots are updated to the new
//!    addresses.  The allocation cursor is set to the new high-water mark and
//!    all marks are cleared.
//!
//! # Entry point
//!
//! [`MarkSweepCompactor::collect`] runs all three phases in order and is the
//! primary entry point for a major (old-generation) GC.

use std::collections::HashSet;

use crate::gc::heap::OldSpace;
use crate::objects::heap_object::HeapObject;

/// Performs a Mark-Sweep-Compact collection on the old (tenured) generation.
///
/// The collector is created once per GC cycle, borrows the [`OldSpace`] for
/// the duration of the cycle, and is dropped when the cycle ends.
///
/// # Example
///
/// ```ignore
/// unsafe {
///     MarkSweepCompactor::new(&mut heap.old_space).collect(&mut roots);
/// }
/// ```
pub struct MarkSweepCompactor<'heap> {
    old_space: &'heap mut OldSpace,
    /// Addresses (as `usize`) of old-space objects that have been marked
    /// reachable in the current cycle.
    mark_set: HashSet<usize>,
}

impl<'heap> MarkSweepCompactor<'heap> {
    /// Create a new compactor that will operate on `old_space`.
    pub fn new(old_space: &'heap mut OldSpace) -> Self {
        Self {
            old_space,
            mark_set: HashSet::new(),
        }
    }

    // ── Phase helpers ─────────────────────────────────────────────────────

    /// Return `true` if the old-space object at `ptr` has been marked.
    fn is_marked(&self, ptr: *mut HeapObject) -> bool {
        self.mark_set.contains(&(ptr as usize))
    }

    /// Mark the old-space object at `ptr` as reachable.
    fn mark_ptr(&mut self, ptr: *mut HeapObject) {
        self.mark_set.insert(ptr as usize);
    }

    /// Discard all marks, preparing for the next GC cycle.
    fn clear_marks(&mut self) {
        self.mark_set.clear();
    }

    // ── Public phase methods ──────────────────────────────────────────────

    /// **Mark phase**: mark every old-space object reachable from `roots`.
    ///
    /// Each root slot that points into old-space is pushed onto a grey stack
    /// and then drained with a BFS walk.  Full child-pointer traversal (via
    /// a `Relocate`/`Trace` dispatch on concrete object types) will be wired
    /// in once the complete object model is available; the grey-stack
    /// infrastructure is already in place.
    ///
    /// # Safety
    /// Every non-null root pointer in `roots` that falls within old-space must
    /// point to a valid, live `HeapObject` with an initialised `alloc_size`.
    pub unsafe fn mark(&mut self, roots: &[*mut *mut HeapObject]) {
        let mut grey_stack: Vec<*mut HeapObject> = Vec::new();

        // Seed the grey stack with old-space roots.
        for &slot in roots {
            // SAFETY: slot is a valid root pointer slot; caller guarantees it.
            let ptr = unsafe { *slot };
            if ptr.is_null() {
                continue;
            }
            if self.old_space.contains(ptr as *mut u8) && !self.is_marked(ptr) {
                self.mark_ptr(ptr);
                grey_stack.push(ptr);
            }
        }

        // BFS: drain the grey stack.
        //
        // TODO: call a `Trace`/`Relocate` dispatch on each object to also
        // push its embedded child pointers once the full object model carries
        // instance-type information in `HeapObject`.
        while let Some(_obj) = grey_stack.pop() {
            // Child traversal will be added here.
        }
    }

    /// **Sweep phase**: identify dead objects and return their byte ranges.
    ///
    /// Walks every object in old-space in address order.  For each object that
    /// was **not** marked, its `(byte_offset_from_base, byte_size)` is
    /// appended to the returned list.
    ///
    /// This phase does **not** change the old-space allocation cursor.  The
    /// caller should follow up with [`compact`][Self::compact] to reclaim the
    /// dead bytes or use the returned list to build a free-list allocator.
    ///
    /// # Safety
    /// Old-space must contain a contiguous sequence of valid `HeapObject`
    /// records from offset 0 to `old_space.used()`, each with `alloc_size > 0`.
    pub unsafe fn sweep(&self) -> Vec<(usize, usize)> {
        let base = self.old_space.base_ptr();
        let used = self.old_space.used();
        let mut free_regions: Vec<(usize, usize)> = Vec::new();

        let mut offset = 0usize;
        while offset < used {
            let obj_ptr = unsafe { base.add(offset) } as *mut HeapObject;
            // SAFETY: offset is within [0, used); alloc_size was written by
            // the allocator.
            let size = unsafe { (*obj_ptr).alloc_size() } as usize;
            if size == 0 {
                // Defensive stop: uninitialised header — do not walk further.
                break;
            }
            if !self.is_marked(obj_ptr) {
                free_regions.push((offset, size));
            }
            offset += size;
        }

        free_regions
    }

    /// **Compact phase**: relocate live objects toward the base of old-space.
    ///
    /// Three sub-passes are performed:
    ///
    /// 1. **Plan** – Walk all objects; for each live (marked) object compute
    ///    its new address by packing them in order from the base.
    /// 2. **Update roots** – Rewrite every root slot whose value was relocated.
    /// 3. **Move** – Copy each live object to its planned destination using
    ///    [`ptr::copy`] (handles overlapping ranges correctly since all moves
    ///    are toward lower or equal addresses).
    ///
    /// After compaction:
    /// * `old_space.used()` equals the sum of all live object sizes.
    /// * All marks are cleared.
    ///
    /// # Safety
    /// * Every non-null root slot that points into old-space must be a marked
    ///   (live) object as established by a preceding [`mark`][Self::mark] call.
    /// * Old-space must contain a contiguous sequence of valid `HeapObject`
    ///   records from offset 0 to `old_space.used()`.
    pub unsafe fn compact(&mut self, roots: &mut [*mut *mut HeapObject]) {
        let base = self.old_space.base_ptr();
        let used = self.old_space.used();

        // ── Sub-pass 1: build the compaction plan ─────────────────────────
        // Each entry is (old_ptr, new_ptr, size_in_bytes).
        let mut plan: Vec<(*mut HeapObject, *mut HeapObject, usize)> = Vec::new();
        let mut new_offset: usize = 0;

        let mut offset = 0usize;
        while offset < used {
            let old_ptr = unsafe { base.add(offset) } as *mut HeapObject;
            // SAFETY: offset is within [0, used); header is valid.
            let size = unsafe { (*old_ptr).alloc_size() } as usize;
            if size == 0 {
                break;
            }
            if self.is_marked(old_ptr) {
                let new_ptr = unsafe { base.add(new_offset) } as *mut HeapObject;
                plan.push((old_ptr, new_ptr, size));
                new_offset += size;
            }
            offset += size;
        }

        // ── Sub-pass 2: update root slots ─────────────────────────────────
        for slot in roots.iter_mut() {
            // SAFETY: slot is a valid, dereferenceable pointer.
            let old_ptr = unsafe { **slot };
            if old_ptr.is_null() {
                continue;
            }
            // Linear search is acceptable: root sets are typically small.
            if let Some(&(_, new_ptr, _)) = plan.iter().find(|(o, _, _)| *o == old_ptr) {
                // SAFETY: slot is valid and new_ptr is the live copy.
                unsafe { **slot = new_ptr };
            }
        }

        // ── Sub-pass 3: physically move live objects ──────────────────────
        // Objects are moved toward lower addresses (forward compaction).
        // Processing in address order guarantees that the destination of each
        // move precedes — or equals — its source, so there is no harmful
        // overlap.  `ptr::copy` (memmove semantics) is used for safety.
        for &(old_ptr, new_ptr, size) in &plan {
            if old_ptr != new_ptr {
                // SAFETY: both ranges are valid; destination ≤ source in
                // memory so the copy does not corrupt still-unprocessed
                // source data.
                unsafe {
                    std::ptr::copy(old_ptr as *const u8, new_ptr as *mut u8, size);
                }
            }
        }

        // ── Finalise ──────────────────────────────────────────────────────
        self.old_space.force_used(new_offset);
        self.clear_marks();
    }

    /// Run a complete **Mark → Sweep → Compact** cycle.
    ///
    /// This is the primary entry point for a major (old-generation) GC.
    /// After this call:
    ///
    /// * Unreachable old-space objects have been discarded.
    /// * Surviving objects have been relocated toward the base of old-space.
    /// * Every root slot in `roots` that pointed into old-space has been
    ///   updated to the object's new address.
    /// * `old_space.used()` equals the total byte size of surviving objects.
    ///
    /// # Safety
    /// See [`mark`][Self::mark] and [`compact`][Self::compact] for the full
    /// safety contracts.
    pub unsafe fn collect(&mut self, roots: &mut [*mut *mut HeapObject]) {
        // Mark phase: identify live objects.
        // SAFETY: caller upholds mark's preconditions.
        unsafe { self.mark(roots) };

        // Sweep phase: enumerate dead byte ranges (informational).
        // The compact phase below makes the free-region list obsolete, but
        // callers that want to build a free-list allocator instead of
        // compacting can use the return value of `sweep` directly.
        let _free_regions = unsafe { self.sweep() };

        // Compact phase: move survivors and update roots.
        // SAFETY: caller upholds compact's preconditions.
        unsafe { self.compact(roots) };
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::heap::OldSpace;
    use crate::objects::heap_object::HeapObject;
    use std::alloc::Layout;
    use std::mem::align_of;

    /// Allocate a single `HeapObject`-sized slot in `old_space`, zero-init it,
    /// and write the `alloc_size` header field.  Returns the pointer.
    fn alloc_in_old(old: &mut OldSpace) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = old.bump_alloc(layout).expect("old space has free space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid and zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    /// Allocate a slot of `extra` bytes on top of `HeapObject` in old-space.
    fn alloc_sized_in_old(old: &mut OldSpace, extra: usize) -> *mut HeapObject {
        let base_size = std::mem::size_of::<HeapObject>();
        let total = base_size + extra;
        let layout = Layout::from_size_align(total, align_of::<HeapObject>())
            .expect("valid layout")
            .pad_to_align();
        let raw = old.bump_alloc(layout).expect("old space has free space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid and zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    // ── Mark phase ────────────────────────────────────────────────────────

    #[test]
    fn test_mark_marks_rooted_object() {
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut root: *mut HeapObject = obj;
        let mut msc = MarkSweepCompactor::new(&mut old);

        let roots = [&raw mut root as *mut *mut HeapObject];
        // SAFETY: root is a valid old-space object.
        unsafe { msc.mark(&roots) };

        assert!(msc.is_marked(obj), "rooted old-space object must be marked");
    }

    #[test]
    fn test_mark_does_not_mark_null_root() {
        let mut old = OldSpace::new(65536);
        let mut root: *mut HeapObject = std::ptr::null_mut();
        let mut msc = MarkSweepCompactor::new(&mut old);

        let roots = [&raw mut root as *mut *mut HeapObject];
        // SAFETY: null root must be skipped gracefully.
        unsafe { msc.mark(&roots) };

        assert!(msc.mark_set.is_empty(), "null root must not add any marks");
    }

    #[test]
    fn test_mark_ignores_non_old_space_pointer() {
        let mut old = OldSpace::new(65536);
        // Stack-allocated object — not in old-space.
        let mut stack_obj = HeapObject::new_null();
        let stack_ptr: *mut HeapObject = &raw mut stack_obj;

        let mut root: *mut HeapObject = stack_ptr;
        let mut msc = MarkSweepCompactor::new(&mut old);

        let roots = [&raw mut root as *mut *mut HeapObject];
        // SAFETY: stack_ptr is not in old-space; the mark phase should skip it.
        unsafe { msc.mark(&roots) };

        assert!(
            msc.mark_set.is_empty(),
            "pointer outside old-space must not be marked"
        );
    }

    // ── Sweep phase ───────────────────────────────────────────────────────

    #[test]
    fn test_sweep_returns_free_region_for_unmarked_object() {
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        // Do NOT mark obj — it is dead.
        let msc = MarkSweepCompactor::new(&mut old);

        // SAFETY: old-space contains a contiguous sequence of valid objects.
        let free_regions = unsafe { msc.sweep() };

        assert_eq!(free_regions.len(), 1, "one dead object → one free region");
        let (offset, size) = free_regions[0];
        // The object starts at offset 0 (first allocation).
        assert_eq!(offset, 0, "dead object must be at offset 0");
        let expected_size = unsafe { (*obj).alloc_size() } as usize;
        assert_eq!(
            size, expected_size,
            "free region size must match alloc_size"
        );
    }

    #[test]
    fn test_sweep_returns_no_free_regions_when_all_marked() {
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);

        let mut msc = MarkSweepCompactor::new(&mut old);
        msc.mark_ptr(obj); // manually mark as live

        // SAFETY: old-space contains valid objects.
        let free_regions = unsafe { msc.sweep() };

        assert!(
            free_regions.is_empty(),
            "marked object must not appear in the free-region list"
        );
    }

    #[test]
    fn test_sweep_mixed_live_dead() {
        let mut old = OldSpace::new(65536);
        let live = alloc_in_old(&mut old);
        let _dead = alloc_in_old(&mut old);
        let live2 = alloc_in_old(&mut old);

        let mut msc = MarkSweepCompactor::new(&mut old);
        msc.mark_ptr(live);
        msc.mark_ptr(live2);

        // SAFETY: old-space contains valid objects.
        let free_regions = unsafe { msc.sweep() };

        assert_eq!(
            free_regions.len(),
            1,
            "exactly one dead object → one free region"
        );
    }

    // ── Compact phase ─────────────────────────────────────────────────────

    /// After compacting with a single live object the `used` counter must
    /// equal that object's `alloc_size`.
    #[test]
    fn test_compact_single_live_object_updates_used() {
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);
        let obj_size = unsafe { (*obj).alloc_size() } as usize;

        // Allocate a second (dead) object to create fragmentation.
        let _dead = alloc_in_old(&mut old);

        let mut msc = MarkSweepCompactor::new(&mut old);
        msc.mark_ptr(obj);

        let mut root: *mut HeapObject = obj;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        // SAFETY: root is a valid marked old-space object.
        unsafe { msc.compact(&mut roots) };

        assert_eq!(
            old.used(),
            obj_size,
            "used must equal exactly one live object's size after compaction"
        );
    }

    /// Compacting with zero live objects must leave `used == 0`.
    #[test]
    fn test_compact_no_live_objects_zeros_used() {
        let mut old = OldSpace::new(65536);
        let _dead1 = alloc_in_old(&mut old);
        let _dead2 = alloc_in_old(&mut old);

        // Mark nothing.
        let mut msc = MarkSweepCompactor::new(&mut old);
        let mut roots: [*mut *mut HeapObject; 0] = [];

        // SAFETY: no live objects; compact must handle an empty plan.
        unsafe { msc.compact(&mut roots) };

        assert_eq!(
            old.used(),
            0,
            "used must be 0 after compacting all-dead space"
        );
    }

    /// Root slot is updated to the object's new (compacted) address.
    #[test]
    fn test_compact_updates_root_slot() {
        let mut old = OldSpace::new(65536);

        // Allocate a dead object first so the live object gets relocated.
        let _dead = alloc_in_old(&mut old);
        let live = alloc_in_old(&mut old);

        let original_addr = live as usize;

        let mut msc = MarkSweepCompactor::new(&mut old);
        msc.mark_ptr(live);

        let mut root: *mut HeapObject = live;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        // SAFETY: root is a valid marked old-space object.
        unsafe { msc.compact(&mut roots) };

        let new_addr = root as usize;
        assert_ne!(
            new_addr, original_addr,
            "root must point to the relocated (compacted) address"
        );
        // The compacted address must be at the start of the region.
        let base = old.base_ptr() as usize;
        assert_eq!(new_addr, base, "single survivor must be at the region base");
    }

    // ── collect (all phases) ──────────────────────────────────────────────

    /// Old-space object survival: a rooted object persists across a full
    /// mark-sweep-compact cycle.
    #[test]
    fn test_collect_old_space_object_survives() {
        let mut old = OldSpace::new(65536);
        let obj = alloc_in_old(&mut old);
        let obj_size = unsafe { (*obj).alloc_size() } as usize;

        let mut root: *mut HeapObject = obj;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        // SAFETY: root is a valid old-space object.
        unsafe { MarkSweepCompactor::new(&mut old).collect(&mut roots) };

        assert!(
            !root.is_null(),
            "root must remain non-null after a full MSC cycle"
        );
        assert_eq!(
            old.used(),
            obj_size,
            "used must equal the surviving object's size"
        );
    }

    /// Full GC reclamation: an unreachable object is freed after a full cycle.
    #[test]
    fn test_collect_unrooted_object_is_reclaimed() {
        let mut old = OldSpace::new(65536);

        let live = alloc_in_old(&mut old);
        let live_size = unsafe { (*live).alloc_size() } as usize;
        let _dead = alloc_in_old(&mut old); // no root → unreachable

        let mut root: *mut HeapObject = live;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        // SAFETY: root is a valid old-space object; _dead has no root.
        unsafe { MarkSweepCompactor::new(&mut old).collect(&mut roots) };

        assert_eq!(
            old.used(),
            live_size,
            "dead object must have been reclaimed; only the live object remains"
        );
    }

    /// Fragmentation compaction: alternating live/dead objects collapse to a
    /// contiguous region containing only the survivors.
    #[test]
    fn test_collect_fragmentation_is_eliminated() {
        let mut old = OldSpace::new(65536);

        // Allocate 4 objects: keep objects at indices 0 and 2 (alternating).
        let obj0 = alloc_sized_in_old(&mut old, 0);
        let _obj1 = alloc_sized_in_old(&mut old, 0); // dead
        let obj2 = alloc_sized_in_old(&mut old, 0);
        let _obj3 = alloc_sized_in_old(&mut old, 0); // dead

        let live_size =
            (unsafe { (*obj0).alloc_size() } + unsafe { (*obj2).alloc_size() }) as usize;

        let mut root0: *mut HeapObject = obj0;
        let mut root2: *mut HeapObject = obj2;
        let mut roots = [
            &raw mut root0 as *mut *mut HeapObject,
            &raw mut root2 as *mut *mut HeapObject,
        ];

        // SAFETY: root0 and root2 are valid old-space objects.
        unsafe { MarkSweepCompactor::new(&mut old).collect(&mut roots) };

        assert_eq!(
            old.used(),
            live_size,
            "after compaction, used must equal the total size of the two survivors"
        );
        // The two survivors must be packed at the start of the region.
        let base = old.base_ptr() as usize;
        let first_size = unsafe { (*root0).alloc_size() } as usize;
        assert_eq!(
            root0 as usize, base,
            "first survivor must be at the region base after compaction"
        );
        assert_eq!(
            root2 as usize,
            base + first_size,
            "second survivor must immediately follow the first"
        );
    }

    /// Repeated collect cycles: the space remains in a consistent state
    /// across multiple MSC runs with evolving root sets.
    #[test]
    fn test_collect_multiple_cycles_consistent() {
        let mut old = OldSpace::new(65536);

        let obj = alloc_in_old(&mut old);
        let obj_size = unsafe { (*obj).alloc_size() } as usize;

        let mut root: *mut HeapObject = obj;

        for _ in 0..3 {
            let mut roots = [&raw mut root as *mut *mut HeapObject];
            // SAFETY: root is a valid old-space object throughout all cycles.
            unsafe { MarkSweepCompactor::new(&mut old).collect(&mut roots) };

            assert!(!root.is_null(), "root must survive every collect cycle");
            assert_eq!(
                old.used(),
                obj_size,
                "used must be stable across repeated cycles with the same root"
            );
        }
    }
}

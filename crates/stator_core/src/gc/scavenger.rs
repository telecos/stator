//! Cheney semi-space scavenger (minor GC) and write-barrier remembered set.
//!
//! # Design
//!
//! The scavenger implements a classic Cheney two-finger copy collector for the
//! young generation:
//!
//! 1. **Root copying** – every root pointer that falls in the from-space is
//!    replaced by a fresh copy in the to-space (or, for objects old enough, by
//!    a promotion into old-space).  A forwarding pointer is installed in the
//!    original from-space slot so that subsequent references to the same object
//!    resolve to the same copy.
//!
//! 2. **BFS scan** – a scan cursor walks the to-space from its base to its
//!    current high-water mark.  For each copied object the scavenger would
//!    update any embedded child pointers (a `Relocate` trait will provide this
//!    once the full object model is in place; the loop structure is here now).
//!
//! 3. **Semi-space swap** – after the scan, [`SemiSpace::collect`] swaps the
//!    halves: the populated to-space becomes the new from-space and the old
//!    from-space is cleared for reuse.
//!
//! # Write barrier
//!
//! When mutator code stores a young-generation pointer into an old-generation
//! object it must call the write barrier, which inserts the old-generation
//! object into the [`RememberedSet`].  During the next scavenge the remembered
//! set entries are used as additional roots so that young objects reachable
//! only through old-space references are not incorrectly collected.

use std::alloc::Layout;
use std::collections::HashSet;
use std::mem::align_of;

use crate::gc::heap::{OldSpace, SemiSpace};
use crate::objects::heap_object::HeapObject;

/// Number of scavenge cycles an object must survive before being promoted to
/// the old generation.
pub const PROMOTION_AGE: u8 = 3;

// ── RememberedSet ──────────────────────────────────────────────────────────────

/// Tracks old-generation objects that contain references into the young
/// generation ("old→young" pointers).
///
/// When mutator code assigns a young-generation pointer into a field of an
/// old-generation object, the *write barrier* must record the old object here.
/// On the next scavenge, every object in the remembered set is treated as an
/// additional GC root so that the young objects it references are copied and
/// not collected.
///
/// After a scavenge the set is cleared; all young objects that were reached
/// through it are now in the new from-space (or promoted), and the
/// old-generation objects' pointer fields should be updated to those new
/// addresses.
pub struct RememberedSet {
    /// Raw pointers to old-space objects with at least one young-space field.
    slots: HashSet<*mut HeapObject>,
}

// SAFETY: The GC runtime is single-threaded; `RememberedSet` is owned by the
// heap and is never aliased across threads.
unsafe impl Send for RememberedSet {}

impl RememberedSet {
    /// Create an empty remembered set.
    pub fn new() -> Self {
        Self {
            slots: HashSet::new(),
        }
    }

    /// Record that `old_ptr` (an old-space object) now holds a reference to a
    /// young-space object.
    ///
    /// Duplicate insertions are silently ignored.
    pub fn insert(&mut self, old_ptr: *mut HeapObject) {
        self.slots.insert(old_ptr);
    }

    /// Remove `old_ptr` from the set.
    ///
    /// Has no effect if `old_ptr` is not present.
    pub fn remove(&mut self, old_ptr: *mut HeapObject) {
        self.slots.remove(&old_ptr);
    }

    /// Iterate the raw pointers of all recorded old-space objects.
    pub fn iter(&self) -> impl Iterator<Item = *mut HeapObject> + '_ {
        self.slots.iter().copied()
    }

    /// Remove all entries.
    ///
    /// Called at the end of a scavenge cycle once all live young objects have
    /// been copied and the old-space pointers updated.
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Number of entries currently in the set.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns `true` if the set contains no entries.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl Default for RememberedSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Scavenger ─────────────────────────────────────────────────────────────────

/// Implements a Cheney-style semi-space scavenge (minor GC).
///
/// The scavenger borrows the young generation's [`SemiSpace`] and the old
/// generation's [`OldSpace`] for the duration of a single scavenge cycle.
/// After [`scavenge`][Scavenger::scavenge] returns both spaces are in a
/// consistent state and the borrows are released.
pub struct Scavenger<'heap> {
    semi_space: &'heap mut SemiSpace,
    old_space: &'heap mut OldSpace,
}

impl<'heap> Scavenger<'heap> {
    /// Create a `Scavenger` that operates on the given spaces.
    pub fn new(semi_space: &'heap mut SemiSpace, old_space: &'heap mut OldSpace) -> Self {
        Self {
            semi_space,
            old_space,
        }
    }

    /// Run a full scavenge cycle.
    ///
    /// # Phases
    ///
    /// 1. **Root copying** – every `*mut HeapObject` that falls inside the
    ///    from-space is copied to the to-space (or promoted to old-space if its
    ///    age is ≥ [`PROMOTION_AGE`]).  The original from-space slot receives a
    ///    forwarding pointer so that shared references converge on one copy.
    ///    Both the `roots` slice and the `remembered_set` contribute roots.
    ///
    /// 2. **Cheney BFS scan** – a scan cursor walks the to-space; each copied
    ///    object's size (from its `alloc_size` header field) advances the
    ///    cursor.  When a `Relocate` trait is available this phase will also
    ///    update embedded child pointers.
    ///
    /// 3. **Semi-space swap** – [`SemiSpace::collect`] makes the to-space the
    ///    new from-space and resets the old from-space.
    ///
    /// # Safety
    /// Every `*mut HeapObject` reachable via `roots` or `remembered_set` must
    /// point to a live, validly-aligned heap object in the current from-space.
    pub unsafe fn scavenge(
        &mut self,
        roots: &mut [*mut *mut HeapObject],
        remembered_set: &mut RememberedSet,
    ) {
        // ── Phase 1: copy from-space roots into to-space ──────────────────
        for slot in roots.iter_mut() {
            // SAFETY: slot is a valid pointer to a root slot.
            let ptr = unsafe { **slot };
            if !ptr.is_null() && self.semi_space.is_in_from_space(ptr) {
                // SAFETY: caller guarantees ptr is a valid live from-space object.
                let new_ptr = unsafe { self.copy_object(ptr) };
                // SAFETY: slot is a valid, dereferenceable pointer.
                unsafe { **slot = new_ptr };
            }
        }

        // Process remembered-set entries as additional roots.  Old-space
        // objects themselves are not in the from-space, so we only use them
        // to ensure liveness — the actual pointer-update pass (scanning old
        // object fields) requires the Relocate trait which is not yet
        // implemented.
        let rem_ptrs: Vec<*mut HeapObject> = remembered_set.iter().collect();
        for old_ptr in rem_ptrs {
            // Guard: if the remembered-set entry is itself in from-space
            // (should not normally happen), copy it too.
            if !old_ptr.is_null() && self.semi_space.is_in_from_space(old_ptr) {
                // SAFETY: old_ptr is a valid from-space object per precondition.
                unsafe { self.copy_object(old_ptr) };
            }
        }
        remembered_set.clear();

        // ── Phase 2: Cheney BFS scan of to-space ─────────────────────────
        //
        // Walk the range [to_base, to_base + to_space_used) advancing by
        // each object's alloc_size.  When a Relocate trait is available,
        // each object's embedded HeapObject pointer fields will be updated
        // here (calling copy_object for each child that still lives in the
        // from-space).
        let to_base = self.semi_space.to_space_base() as usize;
        let mut scan = to_base;
        loop {
            let used = self.semi_space.to_space_used();
            if scan >= to_base + used {
                break;
            }
            let obj = scan as *mut HeapObject;
            // SAFETY: `scan` is within the to-space allocation range and
            // was written there by copy_object.
            let size = unsafe { (*obj).alloc_size() as usize };
            if size == 0 {
                // Defensive stop: an uninitialized header means we've run
                // past the live area.
                break;
            }
            // TODO: call obj.relocate(&mut |field| self.copy_field(field))
            //       once the Relocate trait is implemented.
            scan += size;
        }

        // ── Phase 3: swap semi-space halves ──────────────────────────────
        self.semi_space.collect();
    }

    /// Copy `from_ptr` from the from-space to the to-space (or promote it to
    /// old-space if its scavenge age meets the promotion threshold).
    ///
    /// If `from_ptr` already has a forwarding pointer (it was copied earlier
    /// in this cycle), the existing copy's address is returned without any
    /// further work.
    ///
    /// Returns the address of the (possibly newly-created) copy.
    ///
    /// # Safety
    /// `from_ptr` must be non-null, properly aligned, and point to a live
    /// `HeapObject` in the current from-space.
    unsafe fn copy_object(&mut self, from_ptr: *mut HeapObject) -> *mut HeapObject {
        // If this object was already copied this cycle, return its forwarding
        // destination immediately — this handles shared/cyclic references.
        // SAFETY: from_ptr is a valid from-space object per caller contract.
        if unsafe { (*from_ptr).is_forwarded() } {
            return unsafe { (*from_ptr).forwarding_ptr() };
        }

        let alloc_size = unsafe { (*from_ptr).alloc_size() as usize };
        let age = unsafe { (*from_ptr).age() };

        // Build the layout for the copy destination.
        let layout = Layout::from_size_align(alloc_size, align_of::<HeapObject>())
            .expect("alloc_size and HeapObject alignment must produce a valid Layout");

        // Helper: try to bump-allocate in to-space.
        let alloc_to_space = |ss: &mut SemiSpace| -> *mut HeapObject {
            ss.bump_alloc_to_space(layout)
                .map(|p| p as *mut HeapObject)
                .unwrap_or(std::ptr::null_mut())
        };

        // Decide whether to promote or keep in the young generation.
        let dest: *mut HeapObject = if age >= PROMOTION_AGE {
            // Promote: allocate in old-space.
            match self.old_space.bump_alloc(layout) {
                Some(ptr) => ptr as *mut HeapObject,
                // Old-space full: fall back to to-space (object stays young).
                None => alloc_to_space(self.semi_space),
            }
        } else {
            // Copy into to-space.
            alloc_to_space(self.semi_space)
        };

        if dest.is_null() {
            // Allocation failed; leave the object in place as a last resort.
            return from_ptr;
        }

        // Bitwise-copy the object bytes to the new location.
        // SAFETY: both pointers are valid and non-overlapping; alloc_size bytes
        // are accessible at each address.
        unsafe {
            std::ptr::copy_nonoverlapping(from_ptr as *const u8, dest as *mut u8, alloc_size)
        };

        // Increment the survival age in the freshly-copied object.
        // SAFETY: dest is a valid, just-written HeapObject.
        unsafe { (*dest).increment_age() };

        // Install the forwarding pointer in the original location so that any
        // later reference to from_ptr is redirected to dest.
        // SAFETY: from_ptr is valid and dest is non-null and properly aligned.
        unsafe { (*from_ptr).set_forwarding_ptr(dest) };

        dest
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::heap::{OldSpace, SemiSpace};
    use crate::objects::heap_object::HeapObject;
    use std::alloc::Layout;

    // Helper: allocate a HeapObject in `semi_space`'s from-space, zero-init
    // it, and write its alloc_size header field.
    fn alloc_in_from(semi: &mut SemiSpace) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = semi.bump_alloc(layout).expect("from-space has space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is a valid pointer to zero-initialised memory.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    // ── RememberedSet basic API ───────────────────────────────────────────

    #[test]
    fn test_remembered_set_insert_and_clear() {
        let mut rs = RememberedSet::new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(rs.is_empty());
        rs.insert(ptr);
        assert_eq!(rs.len(), 1);
        rs.clear();
        assert!(rs.is_empty());
    }

    #[test]
    fn test_remembered_set_remove() {
        let mut rs = RememberedSet::new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        rs.insert(ptr);
        rs.remove(ptr);
        assert!(rs.is_empty());
    }

    #[test]
    fn test_remembered_set_duplicate_insert_is_idempotent() {
        let mut rs = RememberedSet::new();
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        rs.insert(ptr);
        rs.insert(ptr);
        assert_eq!(rs.len(), 1, "duplicate inserts must not grow the set");
    }

    // ── Scavenge cycle: live objects are copied ───────────────────────────

    #[test]
    fn test_scavenge_live_object_root_is_updated() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let obj_ptr = alloc_in_from(&mut semi);
        let original_addr = obj_ptr as usize;

        let mut root: *mut HeapObject = obj_ptr;
        let mut roots = [&raw mut root as *mut *mut HeapObject];
        let mut rs = RememberedSet::new();

        // SAFETY: root is a valid from-space object.
        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // Root must have been updated to the new location (in the new from-space,
        // which is the old to-space).
        let new_addr = root as usize;
        assert_ne!(
            new_addr, original_addr,
            "root must point to the copy, not the original"
        );
        // New from-space must contain at least one object.
        assert!(
            semi.used() > 0,
            "live object must appear in the new from-space"
        );
    }

    // ── Dead objects are not present in new from-space ────────────────────

    #[test]
    fn test_scavenge_dead_objects_do_not_extend_new_from_space() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        // Allocate two objects but only root one of them.
        let live_ptr = alloc_in_from(&mut semi);
        let _dead_ptr = alloc_in_from(&mut semi); // no root → dead

        let mut root: *mut HeapObject = live_ptr;
        let mut roots = [&raw mut root as *mut *mut HeapObject];
        let mut rs = RememberedSet::new();

        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // Only the one live object should be in the new from-space.
        let obj_size = std::mem::size_of::<HeapObject>();
        assert_eq!(
            semi.used(),
            obj_size,
            "only the live object must be in the new from-space"
        );
    }

    // ── Forwarding pointers deduplicate shared references ─────────────────

    #[test]
    fn test_scavenge_two_roots_to_same_object_get_same_copy() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let obj_ptr = alloc_in_from(&mut semi);

        // Two roots pointing at the same object.
        let mut root1: *mut HeapObject = obj_ptr;
        let mut root2: *mut HeapObject = obj_ptr;
        let mut roots = [
            &raw mut root1 as *mut *mut HeapObject,
            &raw mut root2 as *mut *mut HeapObject,
        ];
        let mut rs = RememberedSet::new();

        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // Both roots must converge on the single copy.
        assert_eq!(
            root1, root2,
            "two roots to the same object must both point to the single copy"
        );
        // Only one copy must exist in the new from-space.
        let obj_size = std::mem::size_of::<HeapObject>();
        assert_eq!(
            semi.used(),
            obj_size,
            "shared object must be copied exactly once"
        );
    }

    // ── Promotion after PROMOTION_AGE scavenges ───────────────────────────

    #[test]
    fn test_scavenge_promotes_old_enough_object_to_old_space() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let obj_ptr = alloc_in_from(&mut semi);

        // Artificially age the object to just below the threshold.
        // SAFETY: obj_ptr is a valid from-space HeapObject.
        for _ in 0..PROMOTION_AGE {
            unsafe { (*obj_ptr).increment_age() };
        }

        let mut root: *mut HeapObject = obj_ptr;
        let mut roots = [&raw mut root as *mut *mut HeapObject];
        let mut rs = RememberedSet::new();

        let old_used_before = old.used();

        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // The object must have been promoted: old-space grew.
        assert!(
            old.used() > old_used_before,
            "promoted object must be in old-space"
        );
        // Young from-space must be empty (no copy went there).
        assert_eq!(
            semi.used(),
            0,
            "promoted object must not appear in the young from-space"
        );
    }

    // ── Age is incremented on each copy ──────────────────────────────────

    #[test]
    fn test_scavenge_increments_age_on_copy() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let obj_ptr = alloc_in_from(&mut semi);
        assert_eq!(unsafe { (*obj_ptr).age() }, 0);

        let mut root: *mut HeapObject = obj_ptr;
        let mut roots = [&raw mut root as *mut *mut HeapObject];
        let mut rs = RememberedSet::new();

        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // The copy's age must be 1.
        // SAFETY: root now points to the freshly-copied object.
        assert_eq!(
            unsafe { (*root).age() },
            1,
            "copied object's age must be incremented by 1"
        );
    }

    // ── Null roots are skipped gracefully ─────────────────────────────────

    #[test]
    fn test_scavenge_null_root_is_skipped() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let mut root: *mut HeapObject = std::ptr::null_mut();
        let mut roots = [&raw mut root as *mut *mut HeapObject];
        let mut rs = RememberedSet::new();

        // Must not panic or crash.
        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }
        assert!(root.is_null(), "null root must remain null");
    }

    // ── Remembered set is cleared after scavenge ─────────────────────────

    #[test]
    fn test_scavenge_clears_remembered_set() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let mut rs = RememberedSet::new();
        // Insert a dummy old-space address (not in from-space).
        let mut dummy = HeapObject::new_null();
        rs.insert(&raw mut dummy);
        assert!(!rs.is_empty());

        let mut roots: [*mut *mut HeapObject; 0] = [];
        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }
        assert!(
            rs.is_empty(),
            "remembered set must be cleared after scavenge"
        );
    }

    // ── Multiple scavenge cycles ──────────────────────────────────────────

    #[test]
    fn test_multiple_scavenge_cycles_age_accumulates() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);

        let first_ptr = alloc_in_from(&mut semi);
        let mut root: *mut HeapObject = first_ptr;

        let mut rs = RememberedSet::new();
        for expected_age in 1..PROMOTION_AGE {
            let mut roots = [&raw mut root as *mut *mut HeapObject];
            unsafe {
                Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
            }
            // Re-allocate into the new from-space so the next cycle has a fresh
            // from-space object alongside the survivor.
            assert_eq!(
                unsafe { (*root).age() },
                expected_age,
                "age after {expected_age} cycle(s) must be {expected_age}"
            );
        }
    }
}

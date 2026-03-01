//! Write barrier for the generational garbage collector.
//!
//! # What is a write barrier?
//!
//! When mutator code stores a pointer to a *young-generation* object into a
//! field of an *old-generation* object, the garbage collector must be notified.
//! Without this notification a subsequent minor GC (scavenge) would not know
//! that the young object is still reachable through the old object, and could
//! incorrectly collect it.
//!
//! The write barrier is the mechanism that performs this notification.  Every
//! heap-pointer store that **might** create an old→young edge must be followed
//! by a call to [`WriteBarrier::record`].
//!
//! # Implementation
//!
//! This implementation uses a *store-buffer* style remembered set (see
//! [`RememberedSet`]).  When [`record`][WriteBarrier::record] detects an
//! old→young edge it inserts the old-generation *host* object into the
//! [`RememberedSet`].  The scavenger already accepts the remembered set as
//! additional roots; once the [`Relocate`][crate::gc::trace] trait is
//! implemented the scavenger will also update the stored pointer fields.
//!
//! # Usage
//!
//! ```rust,ignore
//! // After writing `value` into a field of the object at `host`:
//! unsafe { barrier.record(host, slot, &value); }
//! ```

use crate::gc::heap::{OldSpace, SemiSpace};
use crate::gc::scavenger::RememberedSet;
use crate::objects::heap_object::HeapObject;
use crate::objects::value::JsValue;

/// Tracks old-generation → young-generation pointer edges via a store buffer.
///
/// Holds shared references to the old-space and young-space regions (used to
/// classify pointers) and a mutable reference to the [`RememberedSet`] (used
/// to record old-space host objects that contain young-space references).
///
/// Every heap-pointer store that may create an old→young edge must call
/// [`record`][WriteBarrier::record].
pub struct WriteBarrier<'h> {
    old_space: &'h OldSpace,
    semi_space: &'h SemiSpace,
    remembered_set: &'h mut RememberedSet,
}

impl<'h> WriteBarrier<'h> {
    /// Create a new `WriteBarrier` backed by the given heap regions and
    /// remembered set.
    pub fn new(
        old_space: &'h OldSpace,
        semi_space: &'h SemiSpace,
        remembered_set: &'h mut RememberedSet,
    ) -> Self {
        Self {
            old_space,
            semi_space,
            remembered_set,
        }
    }

    /// Record a potential old-generation → young-generation pointer edge.
    ///
    /// This must be called after **every** heap-pointer store.  The barrier
    /// performs the following checks:
    ///
    /// 1. `value` must be a [`JsValue::Object`] wrapping a non-null pointer —
    ///    primitive values cannot create GC edges and are skipped cheaply.
    /// 2. `host` must reside in the old generation (i.e.
    ///    [`OldSpace::contains`] returns `true`).
    /// 3. The pointer embedded in `value` must reside in the young
    ///    generation's active from-space (i.e.
    ///    [`SemiSpace::is_in_from_space`] returns `true`).
    ///
    /// When all three conditions hold, `host` is inserted into the
    /// [`RememberedSet`] so that the scavenger treats it as an additional
    /// GC root during the next minor collection.
    ///
    /// The `slot` parameter is the address of the specific pointer field
    /// within `host` that was written.  It is accepted for API completeness
    /// — future card-table implementations may use per-card dirty bits
    /// derived from `slot` — but the current store-buffer implementation
    /// tracks at the per-object (`host`) granularity and does not use it.
    ///
    /// # Safety
    ///
    /// * `host` must be non-null and point to a live, validly-aligned heap
    ///   object (in either old-space or young-space).
    /// * `slot` must be a valid pointer to a field within `host`, or null
    ///   if the specific field address is unknown.
    pub unsafe fn record(&mut self, host: *mut HeapObject, _slot: *const JsValue, value: &JsValue) {
        // Only Object (heap pointer) values can create old→young edges.
        let JsValue::Object(value_ptr) = value else {
            return;
        };
        if value_ptr.is_null() {
            return;
        }
        // The host must be in old-space; young→young stores are handled by
        // the normal root-set and do not require remembered-set tracking.
        if !self.old_space.contains(host as *mut u8) {
            return;
        }
        // The value must be in the young from-space; old→old stores do not
        // create edges that could be missed by a minor GC.
        if !self.semi_space.is_in_from_space(*value_ptr) {
            return;
        }
        // Record the old-space host so the scavenger treats it as a root.
        self.remembered_set.insert(host);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::heap::{OldSpace, SemiSpace};
    use crate::gc::scavenger::{RememberedSet, Scavenger};
    use crate::objects::heap_object::HeapObject;
    use std::alloc::Layout;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Bump-allocate a zero-initialised `HeapObject` in the young from-space.
    fn alloc_young(semi: &mut SemiSpace) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = semi.bump_alloc(layout).expect("young from-space has space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid, zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    /// Bump-allocate a zero-initialised `HeapObject` in old-space.
    fn alloc_old(old: &mut OldSpace) -> *mut HeapObject {
        let layout = Layout::new::<HeapObject>();
        let raw = old.bump_alloc(layout).expect("old-space has space");
        // SAFETY: raw is valid and layout.size() bytes are accessible.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        let obj = raw as *mut HeapObject;
        // SAFETY: obj is valid, zero-initialised.
        unsafe { (*obj).init_alloc_size(layout.size() as u32) };
        obj
    }

    // ── WriteBarrier::record: old→young edge is recorded ─────────────────────

    #[test]
    fn test_barrier_records_old_to_young_edge() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let old_obj = alloc_old(&mut old);
        let young_obj = alloc_young(&mut semi);

        let value = JsValue::Object(young_obj);
        // SAFETY: old_obj and young_obj are valid, live heap objects.
        unsafe {
            WriteBarrier::new(&old, &semi, &mut rs).record(old_obj, std::ptr::null(), &value);
        }

        assert_eq!(
            rs.len(),
            1,
            "old→young edge must be recorded in the remembered set"
        );
        assert!(
            rs.iter().any(|p| p == old_obj),
            "the old-space host must be in the remembered set"
        );
    }

    // ── WriteBarrier::record: young→young edge is NOT recorded ────────────────

    #[test]
    fn test_barrier_skips_young_to_young_edge() {
        let mut semi = SemiSpace::new(4096);
        let old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let young_host = alloc_young(&mut semi);
        let young_value_obj = alloc_young(&mut semi);

        let value = JsValue::Object(young_value_obj);
        // SAFETY: both objects are valid, live heap objects.
        unsafe {
            WriteBarrier::new(&old, &semi, &mut rs).record(young_host, std::ptr::null(), &value);
        }

        assert!(
            rs.is_empty(),
            "young→young store must not populate the remembered set"
        );
    }

    // ── WriteBarrier::record: old→old edge is NOT recorded ───────────────────

    #[test]
    fn test_barrier_skips_old_to_old_edge() {
        let semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let old_host = alloc_old(&mut old);
        let old_value_obj = alloc_old(&mut old);

        let value = JsValue::Object(old_value_obj);
        // SAFETY: both objects are valid, live heap objects.
        unsafe {
            WriteBarrier::new(&old, &semi, &mut rs).record(old_host, std::ptr::null(), &value);
        }

        assert!(
            rs.is_empty(),
            "old→old store must not populate the remembered set"
        );
    }

    // ── WriteBarrier::record: primitive value is NOT recorded ────────────────

    #[test]
    fn test_barrier_skips_primitive_values() {
        let semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let old_obj = alloc_old(&mut old);

        for value in [
            JsValue::Smi(42),
            JsValue::Undefined,
            JsValue::Null,
            JsValue::Boolean(true),
            JsValue::HeapNumber(3.14),
            JsValue::String("hello".to_string()),
        ] {
            // SAFETY: old_obj is a valid live heap object.
            unsafe {
                WriteBarrier::new(&old, &semi, &mut rs).record(old_obj, std::ptr::null(), &value);
            }
        }

        assert!(
            rs.is_empty(),
            "primitive value stores must not populate the remembered set"
        );
    }

    // ── WriteBarrier::record: null Object pointer is NOT recorded ─────────────

    #[test]
    fn test_barrier_skips_null_object_pointer() {
        let semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let old_obj = alloc_old(&mut old);
        let value = JsValue::Object(std::ptr::null_mut());

        // SAFETY: old_obj is a valid live heap object.
        unsafe {
            WriteBarrier::new(&old, &semi, &mut rs).record(old_obj, std::ptr::null(), &value);
        }

        assert!(
            rs.is_empty(),
            "null Object pointer must not populate the remembered set"
        );
    }

    // ── WriteBarrier::record: duplicate insertions are idempotent ─────────────

    #[test]
    fn test_barrier_duplicate_records_are_idempotent() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        let old_obj = alloc_old(&mut old);
        let young_obj = alloc_young(&mut semi);
        let value = JsValue::Object(young_obj);

        // Record the same edge twice.
        for _ in 0..3 {
            // SAFETY: both objects are valid live heap objects.
            unsafe {
                WriteBarrier::new(&old, &semi, &mut rs).record(old_obj, std::ptr::null(), &value);
            }
        }

        assert_eq!(
            rs.len(),
            1,
            "duplicate records for the same host must be deduplicated"
        );
    }

    // ── Store young ref in old object; scavenge; verify RS is populated ───────
    //
    // This test validates the end-to-end write-barrier workflow:
    //
    //   1. A young object is allocated in the nursery.
    //   2. The write barrier records the old→young edge in the remembered set.
    //   3. A scavenge runs; the young object is also passed as a direct root so
    //      it is guaranteed to be copied even though the Relocate trait (which
    //      would let the scavenger scan old-space object fields) is not yet
    //      implemented.
    //   4. After the scavenge the remembered set is cleared and the young object
    //      has been moved to the new from-space.
    //
    // Once the Relocate trait is available, step 3 will not require a direct
    // root: the scavenger will discover the young object by scanning the
    // old-space object's fields.

    #[test]
    fn test_store_young_ref_in_old_object_scavenge_verify_survival() {
        let mut semi = SemiSpace::new(4096);
        let mut old = OldSpace::new(65536);
        let mut rs = RememberedSet::new();

        // Allocate the host in old-space and the value in young from-space.
        let old_obj = alloc_old(&mut old);
        let young_obj = alloc_young(&mut semi);
        let original_young_addr = young_obj as usize;

        // Simulate storing `young_obj` into a field of `old_obj`.
        let value = JsValue::Object(young_obj);

        // Write barrier: record the old→young edge.
        // SAFETY: both objects are valid live heap objects.
        unsafe {
            WriteBarrier::new(&old, &semi, &mut rs).record(old_obj, std::ptr::null(), &value);
        }

        // The old-space host must now be in the remembered set.
        assert_eq!(
            rs.len(),
            1,
            "write barrier must record the old-space host in the remembered set"
        );

        // Run a scavenge.  Pass `young_obj` as an explicit root so the
        // scavenger copies it (the Relocate trait would otherwise be needed
        // to discover it through the old-space object's fields).
        let mut root: *mut HeapObject = young_obj;
        let mut roots = [&raw mut root as *mut *mut HeapObject];

        // SAFETY: root is a valid from-space object.
        unsafe {
            Scavenger::new(&mut semi, &mut old).scavenge(&mut roots, &mut rs);
        }

        // The young object must have survived: its address changed because it
        // was evacuated to the new from-space.
        let new_young_addr = root as usize;
        assert_ne!(
            new_young_addr, original_young_addr,
            "young object must have been evacuated to the new from-space"
        );
        assert!(
            semi.used() > 0,
            "new from-space must contain the evacuated young object"
        );

        // The remembered set must have been cleared by the scavenge.
        assert!(
            rs.is_empty(),
            "remembered set must be cleared after a scavenge cycle"
        );
    }
}

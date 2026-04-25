//! The [`HeapObject`] base type shared by all GC-managed objects.

use crate::objects::map::{InstanceType, Map};
use crate::objects::tagged::TaggedValue;

/// Tag bit written into `map_word` to indicate a GC forwarding pointer.
///
/// Heap objects are at least 8-byte aligned, so bits 0–2 of a valid heap
/// address are always zero.  Bit 1 is reserved as the forwarding tag: when
/// `map_word.0 & FORWARDING_TAG != 0`, the object has already been copied and
/// the forwarding destination is `map_word.0 & !FORWARDING_TAG`.
const FORWARDING_TAG: usize = 0b10;

/// The base header for every GC-managed heap object.
///
/// # Memory layout
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
///      0     8  map_word   (Map pointer or GC forwarding pointer)
///      8     4  alloc_size (padded byte size of this allocation)
///     12     1  age        (number of scavenge cycles survived)
///     13     3  _pad       (explicit alignment padding)
/// ```
///
/// Every heap object starts with a `map_word`: a [`TaggedValue`] whose raw
/// bits hold the address of the object's [`Map`] (hidden class).  Derived
/// types embed `HeapObject` as their first `#[repr(C)]` field so that a
/// `*mut DerivedType` can be safely cast to `*mut HeapObject`.
///
/// During a GC scavenge, `map_word` is temporarily overwritten with the
/// object's new address OR-ed with [`FORWARDING_TAG`] (bit 1).  All map
/// accesses must go through [`HeapObject::map`] rather than reading
/// `map_word` directly.
///
/// # Alignment
///
/// `HeapObject` is forced to 8-byte alignment so that the low three bits of
/// every `*mut HeapObject` are zero.  The NaN-boxing scheme in
/// [`crate::objects::nan_boxing`] reuses bits 0 and 60–63 of the pointer for
/// type tags, and the GC forwarding scheme reuses bit 1 of `map_word`
/// (see [`FORWARDING_TAG`]).  Without an explicit `align(8)`, on 32-bit
/// targets this struct would only be 4-byte aligned (since `usize`/`u32`
/// have alignment 4), which would corrupt those tag bits and trip the
/// `pointer must be 8-byte aligned` assertion in
/// [`crate::objects::nan_boxing::NanBoxedValue::from_heap_ptr`].
#[repr(C, align(8))]
pub struct HeapObject {
    /// Tagged pointer to this object's [`Map`] (hidden class), or a
    /// forwarding pointer during a GC scavenge cycle.
    map_word: TaggedValue,
    /// Padded size (in bytes) of this allocation, as recorded by the heap
    /// allocator.  Used by the scavenger to copy the object verbatim.
    alloc_size: u32,
    /// Number of minor GC (scavenge) cycles this object has survived.
    /// When this reaches the promotion threshold the object is copied to
    /// the old generation instead of the young generation's to-space.
    age: u8,
    _pad: [u8; 3],
}

impl HeapObject {
    /// Creates a `HeapObject` header with a null map pointer.
    ///
    /// The returned value has no valid map set.  Callers **must** initialise
    /// the map before invoking [`map`][Self::map] or
    /// [`instance_type`][Self::instance_type].
    ///
    /// This constructor is used during allocation: the heap returns a
    /// zero-initialised block, and the allocator then writes the correct map
    /// pointer.
    pub fn new_null() -> Self {
        Self {
            map_word: TaggedValue(0),
            alloc_size: 0,
            age: 0,
            _pad: [0; 3],
        }
    }

    // ── GC metadata accessors ─────────────────────────────────────────────

    /// Returns the padded allocation size recorded for this object by the
    /// heap allocator.
    ///
    /// A value of `0` indicates the field has not been initialised (the
    /// object was not allocated through [`Heap::allocate`]).
    pub fn alloc_size(&self) -> u32 {
        self.alloc_size
    }

    /// Writes the allocation size into this object's header.
    ///
    /// Called exactly once by the heap allocator immediately after
    /// zero-initialising the backing memory.  Must not be called after the
    /// object is in use.
    pub(crate) fn init_alloc_size(&mut self, size: u32) {
        self.alloc_size = size;
    }

    /// Returns the number of scavenge cycles this object has survived.
    pub fn age(&self) -> u8 {
        self.age
    }

    /// Increment the scavenge-survival counter by one.
    ///
    /// Called by the scavenger when it copies the object into to-space or
    /// promotes it to old-space.
    pub(crate) fn increment_age(&mut self) {
        self.age = self.age.saturating_add(1);
    }

    // ── Forwarding pointer support ────────────────────────────────────────

    /// Returns `true` if this object has been copied by the scavenger and
    /// its `map_word` now holds a forwarding pointer.
    ///
    /// When `true`, the actual destination address can be obtained via
    /// [`forwarding_ptr`][Self::forwarding_ptr].
    #[inline]
    pub fn is_forwarded(&self) -> bool {
        self.map_word.0 & FORWARDING_TAG != 0
    }

    /// Returns the forwarding destination set by the scavenger.
    ///
    /// The return value is only meaningful when [`is_forwarded`][Self::is_forwarded]
    /// returns `true`.
    #[inline]
    pub fn forwarding_ptr(&self) -> *mut HeapObject {
        (self.map_word.0 & !FORWARDING_TAG) as *mut HeapObject
    }

    /// Overwrite `map_word` with a forwarding pointer to `dest`.
    ///
    /// After this call [`is_forwarded`] returns `true` and
    /// [`forwarding_ptr`] returns `dest`.
    ///
    /// # Safety
    /// `dest` must be non-null, 4-byte aligned (so that
    /// `FORWARDING_TAG` does not overlap a real address bit), and must point
    /// to the live copy of this object.
    pub(crate) unsafe fn set_forwarding_ptr(&mut self, dest: *mut HeapObject) {
        debug_assert!(!dest.is_null(), "forwarding destination must be non-null");
        debug_assert!(
            dest as usize & FORWARDING_TAG == 0,
            "forwarding destination must not have bit 1 set"
        );
        self.map_word = TaggedValue(dest as usize | FORWARDING_TAG);
    }

    /// Returns a raw pointer to this object's [`Map`].
    ///
    /// # Safety
    /// `map_word` must contain a valid, non-null, heap-aligned address of a
    /// live [`Map`] object.  Calling this when `map_word` is null, holds a
    /// forwarding pointer, or encodes a `Smi` is undefined behaviour.
    #[inline]
    pub unsafe fn map(&self) -> *mut Map {
        // map_word stores the raw Map address with bit 0 == 0 (naturally
        // aligned heap pointer).
        // SAFETY: caller guarantees map_word is a valid, non-null Map pointer.
        self.map_word.0 as *mut Map
    }

    /// Returns the [`InstanceType`] of this object by reading the [`Map`].
    ///
    /// # Safety
    /// `map_word` must contain a valid, non-null, heap-aligned address of a
    /// live [`Map`] object.
    #[inline]
    pub unsafe fn instance_type(&self) -> InstanceType {
        // SAFETY: caller guarantees the Map pointer is valid and the Map is live.
        unsafe { (*self.map()).instance_type() }
    }

    /// Returns `true` if the `map_word` is non-null and not a forwarding
    /// pointer.  This is a *necessary* but not *sufficient* condition for
    /// [`map`][Self::map] to be safe — the pointer could still be dangling.
    #[inline]
    pub fn has_map(&self) -> bool {
        let raw = self.map_word.0;
        // Null or forwarding pointer (bit 1 set) → no valid map.
        raw != 0 && (raw & FORWARDING_TAG) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `HeapObject` must be at least 8-byte aligned on every target so that
    /// the low three bits of every `*mut HeapObject` are free for use by the
    /// NaN-boxing scheme (bit 0) and the GC forwarding marker (bit 1).
    /// Without `#[repr(C, align(8))]`, this would silently regress to
    /// alignment 4 on 32-bit targets.
    #[test]
    fn test_heap_object_is_8_byte_aligned() {
        assert!(
            std::mem::align_of::<HeapObject>() >= 8,
            "HeapObject must be at least 8-byte aligned on all targets"
        );
    }

    #[test]
    fn test_map_ptr_round_trip() {
        let mut map = Map::new(InstanceType::Map, 0);
        let obj = HeapObject {
            map_word: TaggedValue(&raw mut map as usize),
            ..HeapObject::new_null()
        };
        // SAFETY: map is live for the duration of this test.
        let recovered = unsafe { obj.map() };
        assert_eq!(recovered, &raw mut map);
    }

    #[test]
    fn test_instance_type_via_map() {
        let mut map = Map::new(InstanceType::Map, 0);
        let obj = HeapObject {
            map_word: TaggedValue(&raw mut map as usize),
            ..HeapObject::new_null()
        };
        // SAFETY: map is live for the duration of this test.
        let ty = unsafe { obj.instance_type() };
        assert_eq!(ty, InstanceType::Map);
    }

    #[test]
    fn test_new_null_has_zero_map_word() {
        let obj = HeapObject::new_null();
        assert_eq!(obj.map_word.raw(), 0);
    }

    // ── Forwarding pointer tests ──────────────────────────────────────────

    #[test]
    fn test_not_forwarded_by_default() {
        let obj = HeapObject::new_null();
        assert!(!obj.is_forwarded());
    }

    #[test]
    fn test_forwarding_ptr_round_trip() {
        let mut dest = HeapObject::new_null();
        let dest_ptr = &raw mut dest;
        let mut src = HeapObject::new_null();
        // SAFETY: dest_ptr is non-null and properly aligned.
        unsafe { src.set_forwarding_ptr(dest_ptr) };
        assert!(src.is_forwarded());
        assert_eq!(src.forwarding_ptr(), dest_ptr);
    }

    #[test]
    fn test_age_increments() {
        let mut obj = HeapObject::new_null();
        assert_eq!(obj.age(), 0);
        obj.increment_age();
        assert_eq!(obj.age(), 1);
        obj.increment_age();
        assert_eq!(obj.age(), 2);
    }

    #[test]
    fn test_alloc_size_init() {
        let mut obj = HeapObject::new_null();
        assert_eq!(obj.alloc_size(), 0);
        obj.init_alloc_size(128);
        assert_eq!(obj.alloc_size(), 128);
    }
}

//! The [`HeapObject`] base type shared by all GC-managed objects.

use crate::objects::map::{InstanceType, Map};
use crate::objects::tagged::TaggedValue;

/// The base header for every GC-managed heap object.
///
/// # Memory layout
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
///      0     8  map_word  (raw address of this object's Map)
/// ```
///
/// Every heap object starts with a `map_word`: a [`TaggedValue`] whose raw
/// bits hold the address of the object's [`Map`] (hidden class).  Derived
/// types embed `HeapObject` as their first `#[repr(C)]` field so that a
/// `*mut DerivedType` can be safely cast to `*mut HeapObject`.
///
/// During a GC copy/compaction phase the runtime may temporarily replace
/// `map_word` with a forwarding pointer; therefore all map accesses must go
/// through [`HeapObject::map`] rather than reading `map_word` directly.
#[repr(C)]
pub struct HeapObject {
    /// Tagged pointer to this object's [`Map`] (hidden class).
    map_word: TaggedValue,
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
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_ptr_round_trip() {
        let mut map = Map::new(InstanceType::Map);
        let obj = HeapObject {
            map_word: TaggedValue(&raw mut map as usize),
        };
        // SAFETY: map is live for the duration of this test.
        let recovered = unsafe { obj.map() };
        assert_eq!(recovered, &raw mut map);
    }

    #[test]
    fn test_instance_type_via_map() {
        let mut map = Map::new(InstanceType::Map);
        let obj = HeapObject {
            map_word: TaggedValue(&raw mut map as usize),
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
}

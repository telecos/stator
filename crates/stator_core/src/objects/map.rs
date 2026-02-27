//! The hidden class ([`Map`]) and instance-type tag for heap objects.

/// Identifies the concrete type of a [`HeapObject`][crate::objects::heap_object::HeapObject].
///
/// Each variant corresponds to a distinct Stator / JavaScript object shape
/// and is stored directly in the [`Map`] for O(1) lookup.  Values are
/// intentionally left with gaps to mirror the V8 numbering convention and
/// leave room for future additions.
#[repr(u16)]
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstanceType {
    /// A [`Map`] object itself (the hidden class of a `Map` is also a `Map`).
    Map = 0x0080,
}

/// The hidden class (shape descriptor) of a heap object.
///
/// Every [`HeapObject`][crate::objects::heap_object::HeapObject] starts with
/// a pointer to its `Map`.  The `Map` records the object's [`InstanceType`]
/// (and, in a full implementation, its field layout, prototype chain, and
/// other shape metadata).
#[repr(C)]
pub struct Map {
    /// The concrete object type described by this `Map`.
    instance_type: InstanceType,
}

impl Map {
    /// Creates a `Map` with the given instance type.
    pub fn new(instance_type: InstanceType) -> Self {
        Self { instance_type }
    }

    /// Returns the instance type of objects described by this `Map`.
    #[inline]
    pub fn instance_type(&self) -> InstanceType {
        self.instance_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_instance_type_roundtrip() {
        let map = Map::new(InstanceType::Map);
        assert_eq!(map.instance_type(), InstanceType::Map);
    }
}

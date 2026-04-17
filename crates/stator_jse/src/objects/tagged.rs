use crate::gc::trace::{Trace, Tracer};
use crate::objects::heap_object::HeapObject;

/// A pointer-sized tagged value that can represent either a small integer
/// (`Smi`) or a pointer to a [`HeapObject`].
///
/// # Encoding
///
/// ```text
/// Bit 0 = 1  →  Smi  (bits [usize::BITS-1 .. 1] hold a shifted i31 value)
/// Bit 0 = 0  →  HeapObject pointer (naturally aligned, so bit 0 is always 0)
/// ```
///
/// This matches the V8 tagging scheme on 64-bit platforms, where a 31-bit
/// signed integer is stored shifted left by one bit with the tag in bit 0.
/// Heap pointers need no adjustment because all `HeapObject` allocations are
/// at least 2-byte aligned.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TaggedValue(pub usize);

const SMI_TAG: usize = 1;
const SMI_TAG_MASK: usize = 1;
const SMI_SHIFT: usize = 1;

impl TaggedValue {
    /// Encode a small integer as a `TaggedValue`.
    ///
    /// The integer is stored in bits `[usize::BITS-1 .. 1]` with bit 0 set to
    /// `1` as the Smi tag.
    #[inline]
    pub fn from_smi(value: i32) -> Self {
        let tagged = ((value as usize) << SMI_SHIFT) | SMI_TAG;
        TaggedValue(tagged)
    }

    /// Encode a [`HeapObject`] pointer as a `TaggedValue`.
    ///
    /// # Safety
    /// - `ptr` must be non-null.
    /// - `ptr` must be at least 2-byte aligned (bit 0 must be 0).
    /// - `ptr` must point to a live `HeapObject` managed by the [`Heap`][crate::gc::heap::Heap].
    #[inline]
    pub unsafe fn from_heap_object(ptr: *mut HeapObject) -> Self {
        debug_assert!(!ptr.is_null(), "heap pointer must be non-null");
        debug_assert!(
            ptr as usize & SMI_TAG_MASK == 0,
            "heap pointer must be at least 2-byte aligned"
        );
        TaggedValue(ptr as usize)
    }

    /// Returns `true` if this value encodes a `Smi`.
    #[inline]
    pub fn is_smi(self) -> bool {
        self.0 & SMI_TAG_MASK == SMI_TAG
    }

    /// Returns `true` if this value is a pointer to a heap object.
    #[inline]
    pub fn is_heap_object(self) -> bool {
        !self.is_smi()
    }

    /// Decode the `Smi` integer value.
    ///
    /// Returns `None` if this `TaggedValue` encodes a heap-object pointer.
    #[inline]
    pub fn as_smi(self) -> Option<i32> {
        if self.is_smi() {
            // Arithmetic right-shift preserves the sign of the stored integer.
            Some((self.0 as isize >> SMI_SHIFT) as i32)
        } else {
            None
        }
    }

    /// Return a raw pointer to the [`HeapObject`].
    ///
    /// Returns `None` if this `TaggedValue` encodes a `Smi`.
    ///
    /// # Safety
    /// The returned pointer is only valid as long as the heap has not moved or
    /// freed the object (i.e., no collection has occurred since this value was
    /// created).
    #[inline]
    pub unsafe fn as_heap_object(self) -> Option<*mut HeapObject> {
        if self.is_heap_object() {
            Some(self.0 as *mut HeapObject)
        } else {
            None
        }
    }

    /// Return the raw `usize` representation.
    #[inline]
    pub fn raw(self) -> usize {
        self.0
    }
}

impl Trace for TaggedValue {
    /// Mark the heap object this tagged value points to, if any.
    ///
    /// Smi-encoded values carry no heap reference and are silently ignored.
    fn trace(&self, tracer: &mut Tracer) {
        if self.is_heap_object() {
            // SAFETY: is_heap_object() guarantees bit 0 == 0, so self.0 is a
            // naturally-aligned pointer to a live HeapObject managed by the heap.
            unsafe { tracer.mark_raw(self.0 as *mut u8) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smi_round_trip_positive() {
        let tv = TaggedValue::from_smi(42);
        assert!(tv.is_smi());
        assert!(!tv.is_heap_object());
        assert_eq!(tv.as_smi(), Some(42));
    }

    #[test]
    fn smi_round_trip_negative() {
        let tv = TaggedValue::from_smi(-1);
        assert!(tv.is_smi());
        assert_eq!(tv.as_smi(), Some(-1));
    }

    #[test]
    fn smi_round_trip_zero() {
        let tv = TaggedValue::from_smi(0);
        assert!(tv.is_smi());
        assert_eq!(tv.as_smi(), Some(0));
    }

    #[test]
    fn smi_round_trip_i31_max() {
        // The maximum value encodable in a 31-bit signed integer.
        let max = i32::MAX >> 1; // 2^30 - 1
        let tv = TaggedValue::from_smi(max);
        assert_eq!(tv.as_smi(), Some(max));
    }

    #[test]
    fn smi_round_trip_i31_min() {
        let min = i32::MIN >> 1; // -(2^30)
        let tv = TaggedValue::from_smi(min);
        assert_eq!(tv.as_smi(), Some(min));
    }

    #[test]
    fn heap_ptr_round_trip() {
        // Use a null-map HeapObject as a stand-in; we only need the address.
        let mut obj = HeapObject::new_null();
        let ptr = &mut obj as *mut HeapObject;
        // SAFETY: ptr is non-null and properly aligned.
        let tv = unsafe { TaggedValue::from_heap_object(ptr) };
        assert!(tv.is_heap_object());
        assert!(!tv.is_smi());
        assert_eq!(tv.as_smi(), None);
        // SAFETY: obj is still live.
        let recovered = unsafe { tv.as_heap_object() };
        assert_eq!(recovered, Some(ptr));
    }
}

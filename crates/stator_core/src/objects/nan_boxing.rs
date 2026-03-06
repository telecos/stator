//! NaN-boxed (tagged-pointer) value representation for JavaScript values.
//!
//! [`NanBoxedValue`] compresses the multi-variant [`super::value::JsValue`]
//! enum into a single `u64`, using a tagged-pointer encoding scheme modelled
//! on V8:
//!
//! ```text
//! Smi (31-bit signed):  xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxx1  (bit 0 = 1)
//! HeapPtr:              0TTTAAAA AAAAAAAA AAAAAAAA AAAAAAA0  (bit 0 = 0, T = type tag)
//!
//! Type tags (3-bit): Object=000, String=001, Symbol=010,
//!                    BigInt=011, Function=100, Array=101
//!
//! Special values:
//!   false=0x06, true=0x16, null=0x26, undefined=0x36, hole=0x46
//! ```
//!
//! See [issue #265](https://github.com/telecos/stator/issues/265) for the full
//! design rationale.

use crate::gc::trace::{Trace, Tracer};
use crate::objects::heap_object::HeapObject;
use crate::objects::value::JsValue;

// ─── Smi ────────────────────────────────────────────────────────────────────

const SMI_TAG: u64 = 1;
const SMI_TAG_MASK: u64 = 1;
const SMI_SHIFT: u32 = 1;

// ─── Special sentinels ──────────────────────────────────────────────────────

/// Bit pattern for `false`.
const FALSE_BITS: u64 = 0x06;
/// Bit pattern for `true`.
const TRUE_BITS: u64 = 0x16;
/// Bit pattern for `null`.
const NULL_BITS: u64 = 0x26;
/// Bit pattern for `undefined`.
const UNDEFINED_BITS: u64 = 0x36;
/// Bit pattern for the internal *hole* sentinel.
const HOLE_BITS: u64 = 0x46;

/// Low-nibble mask shared by all special sentinels.
const SPECIAL_TAG: u64 = 0x06;
/// Mask that isolates the low nibble for the special-value check.
const SPECIAL_MASK: u64 = 0x0F;

// ─── Heap-pointer type tags ─────────────────────────────────────────────────

/// Bit position of the 3-bit type tag inside a heap-pointer word.
const HEAP_TAG_SHIFT: u32 = 60;
/// Mask that isolates bits 60–62 (the type tag).
const HEAP_TAG_MASK: u64 = 0x7_u64 << HEAP_TAG_SHIFT;
/// Mask that isolates the lower 48 address bits (x86-64 canonical form).
const POINTER_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// 3-bit type tag stored in bits 60–62 of a heap-pointer [`NanBoxedValue`].
///
/// These tags let the runtime determine the concrete heap-object kind without
/// dereferencing the pointer and reading the object's [`Map`][crate::objects::map::Map].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HeapTypeTag {
    /// A generic JS object (ordinary or exotic).
    Object = 0,
    /// A heap-allocated JavaScript string.
    String = 1,
    /// A JavaScript `Symbol`.
    Symbol = 2,
    /// A `BigInt` value.
    BigInt = 3,
    /// A callable JavaScript function.
    Function = 4,
    /// A JavaScript `Array`.
    Array = 5,
}

impl HeapTypeTag {
    /// Try to interpret a raw 3-bit value as a [`HeapTypeTag`].
    const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::Object),
            1 => Some(Self::String),
            2 => Some(Self::Symbol),
            3 => Some(Self::BigInt),
            4 => Some(Self::Function),
            5 => Some(Self::Array),
            _ => None,
        }
    }
}

// ─── NanBoxedValue ──────────────────────────────────────────────────────────

/// A 64-bit tagged representation of a JavaScript value.
///
/// `NanBoxedValue` packs the type discriminant and payload into a single `u64`
/// word.  At 8 bytes it is 3× smaller than the current [`JsValue`] enum and
/// enables branchless type checks via simple bit-masking.
///
/// # Encoding table
///
/// | Kind        | Bit pattern                                         |
/// |-------------|-----------------------------------------------------|
/// | Smi         | `payload << 1 \| 1` (bit 0 = 1)                    |
/// | HeapPtr     | `tag << 60 \| address` (bit 0 = 0, 8-byte aligned) |
/// | `false`     | `0x06`                                              |
/// | `true`      | `0x16`                                              |
/// | `null`      | `0x26`                                              |
/// | `undefined` | `0x36`                                              |
/// | *hole*      | `0x46`                                              |
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct NanBoxedValue(u64);

// ── Constructors ────────────────────────────────────────────────────────────

impl NanBoxedValue {
    /// Encode a small integer as a [`NanBoxedValue`].
    ///
    /// The full `i32` range is supported: the value is sign-extended to 64
    /// bits, shifted left by one, and OR-ed with the Smi tag in bit 0.
    #[inline]
    pub fn from_smi(value: i32) -> Self {
        Self(((value as i64 as u64) << SMI_SHIFT) | SMI_TAG)
    }

    /// Create the `undefined` sentinel.
    #[inline]
    pub const fn undefined() -> Self {
        Self(UNDEFINED_BITS)
    }

    /// Create the `null` sentinel.
    #[inline]
    pub const fn null() -> Self {
        Self(NULL_BITS)
    }

    /// Encode a boolean value.
    #[inline]
    pub const fn from_bool(b: bool) -> Self {
        if b { Self(TRUE_BITS) } else { Self(FALSE_BITS) }
    }

    /// Create the internal *hole* sentinel used by sparse arrays.
    #[inline]
    pub const fn hole() -> Self {
        Self(HOLE_BITS)
    }

    /// Encode a raw heap-object pointer with the given type tag.
    ///
    /// # Safety
    ///
    /// * `ptr` must be non-null and at least 8-byte aligned.
    /// * `ptr` must point to a live object managed by the engine heap.
    /// * The pointer must fit in 48 bits (valid user-space address).
    #[inline]
    pub unsafe fn from_heap_ptr(ptr: *mut HeapObject, tag: HeapTypeTag) -> Self {
        let addr = ptr as u64;
        debug_assert!(addr & 0x07 == 0, "pointer must be 8-byte aligned");
        debug_assert!(addr & !POINTER_MASK == 0, "pointer must fit in 48 bits");
        Self(((tag as u64) << HEAP_TAG_SHIFT) | addr)
    }
}

// ── Predicates ──────────────────────────────────────────────────────────────

impl NanBoxedValue {
    /// Returns `true` if this value encodes a small integer.
    #[inline]
    pub const fn is_smi(self) -> bool {
        self.0 & SMI_TAG_MASK == SMI_TAG
    }

    /// Returns `true` if this value is one of the five special sentinels
    /// (`undefined`, `null`, `true`, `false`, or *hole*).
    #[inline]
    pub const fn is_special(self) -> bool {
        self.0 & SPECIAL_MASK == SPECIAL_TAG && self.0 <= HOLE_BITS
    }

    /// Returns `true` if this value is a tagged heap-object pointer.
    #[inline]
    pub const fn is_heap_ptr(self) -> bool {
        !self.is_smi() && !self.is_special()
    }

    /// Returns `true` if this value is `undefined`.
    #[inline]
    pub const fn is_undefined(self) -> bool {
        self.0 == UNDEFINED_BITS
    }

    /// Returns `true` if this value is `null`.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == NULL_BITS
    }

    /// Returns `true` if this value is the boolean `true`.
    #[inline]
    pub const fn is_true(self) -> bool {
        self.0 == TRUE_BITS
    }

    /// Returns `true` if this value is the boolean `false`.
    #[inline]
    pub const fn is_false(self) -> bool {
        self.0 == FALSE_BITS
    }

    /// Returns `true` if this value is a boolean (`true` or `false`).
    #[inline]
    pub const fn is_boolean(self) -> bool {
        self.is_true() || self.is_false()
    }

    /// Returns `true` if this value is the internal *hole* sentinel.
    #[inline]
    pub const fn is_hole(self) -> bool {
        self.0 == HOLE_BITS
    }

    /// Returns `true` if this value is either `null` or `undefined`.
    #[inline]
    pub const fn is_nullish(self) -> bool {
        self.is_null() || self.is_undefined()
    }
}

// ── Accessors ───────────────────────────────────────────────────────────────

impl NanBoxedValue {
    /// Decode the small integer, or `None` if this is not a Smi.
    #[inline]
    pub const fn as_smi(self) -> Option<i32> {
        if self.is_smi() {
            // Arithmetic right-shift preserves sign.
            Some(((self.0 as i64) >> SMI_SHIFT) as i32)
        } else {
            None
        }
    }

    /// Decode the boolean value, or `None` if this is not a boolean.
    #[inline]
    pub const fn as_bool(self) -> Option<bool> {
        if self.is_true() {
            Some(true)
        } else if self.is_false() {
            Some(false)
        } else {
            None
        }
    }

    /// Extract the 3-bit heap type tag, or `None` if this is not a heap
    /// pointer.
    #[inline]
    pub fn heap_type_tag(self) -> Option<HeapTypeTag> {
        if self.is_heap_ptr() {
            let bits = ((self.0 & HEAP_TAG_MASK) >> HEAP_TAG_SHIFT) as u8;
            HeapTypeTag::from_bits(bits)
        } else {
            None
        }
    }

    /// Extract the raw heap-object pointer, or `None` if this is not a heap
    /// pointer.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid if no GC compaction has occurred
    /// since this value was created.
    #[inline]
    pub unsafe fn as_heap_ptr(self) -> Option<*mut HeapObject> {
        if self.is_heap_ptr() {
            Some((self.0 & POINTER_MASK) as *mut HeapObject)
        } else {
            None
        }
    }

    /// Return the raw `u64` bit representation.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Create a [`NanBoxedValue`] from a raw `u64`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `bits` is a valid encoding produced by
    /// one of the safe constructors or by [`raw`][Self::raw].
    #[inline]
    pub const unsafe fn from_raw(bits: u64) -> Self {
        Self(bits)
    }
}

// ── JsValue conversions ─────────────────────────────────────────────────────

impl NanBoxedValue {
    /// Try to convert a [`JsValue`] reference into a [`NanBoxedValue`].
    ///
    /// Returns `Some` for the variants that can be represented inline:
    /// `Undefined`, `Null`, `Boolean`, `Smi`, and `Object` (raw heap
    /// pointer).  Returns `None` for heap-allocated variants (`String`,
    /// `HeapNumber`, `BigInt`, `Function`, `Array`, etc.) whose data lives
    /// behind an `Rc` and cannot be losslessly encoded without a GC heap.
    pub fn try_from_js_value(value: &JsValue) -> Option<Self> {
        match value {
            JsValue::Undefined => Some(Self::undefined()),
            JsValue::Null => Some(Self::null()),
            JsValue::Boolean(b) => Some(Self::from_bool(*b)),
            JsValue::Smi(i) => Some(Self::from_smi(*i)),
            JsValue::Object(ptr) => {
                // SAFETY: JsValue::Object guarantees the pointer is to a live,
                // aligned HeapObject.
                Some(unsafe { Self::from_heap_ptr(*ptr, HeapTypeTag::Object) })
            }
            _ => None,
        }
    }

    /// Convert back to a [`JsValue`], if possible.
    ///
    /// Heap-pointer values are returned as `JsValue::Object` regardless of
    /// their type tag because reconstructing the high-level Rust wrapper
    /// (e.g. `Rc<BytecodeArray>`) requires access to the GC heap.
    ///
    /// Returns `None` for the *hole* sentinel, which has no `JsValue`
    /// equivalent.
    ///
    /// # Safety
    ///
    /// If this value is a heap pointer, the pointer must still be valid
    /// (no GC compaction since encoding).
    pub unsafe fn to_js_value(self) -> Option<JsValue> {
        if self.is_undefined() {
            Some(JsValue::Undefined)
        } else if self.is_null() {
            Some(JsValue::Null)
        } else if let Some(b) = self.as_bool() {
            Some(JsValue::Boolean(b))
        } else if let Some(i) = self.as_smi() {
            Some(JsValue::Smi(i))
        } else if self.is_hole() {
            None
        } else if self.is_heap_ptr() {
            // SAFETY: caller guarantees the pointer is still valid;
            // `is_heap_ptr()` ensures `as_heap_ptr()` returns `Some`.
            let ptr = unsafe { self.as_heap_ptr().unwrap_unchecked() };
            Some(JsValue::Object(ptr))
        } else {
            None
        }
    }
}

// ── Trait impls ─────────────────────────────────────────────────────────────

impl std::fmt::Debug for NanBoxedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_undefined() {
            write!(f, "NanBoxedValue(undefined)")
        } else if self.is_null() {
            write!(f, "NanBoxedValue(null)")
        } else if let Some(b) = self.as_bool() {
            write!(f, "NanBoxedValue({b})")
        } else if self.is_hole() {
            write!(f, "NanBoxedValue(hole)")
        } else if let Some(i) = self.as_smi() {
            write!(f, "NanBoxedValue(Smi({i}))")
        } else if let Some(tag) = self.heap_type_tag() {
            write!(
                f,
                "NanBoxedValue(HeapPtr({tag:?}, 0x{:012x}))",
                self.0 & POINTER_MASK
            )
        } else {
            write!(f, "NanBoxedValue(0x{:016x})", self.0)
        }
    }
}

impl Trace for NanBoxedValue {
    /// Mark the heap object this value points to, if any.
    ///
    /// Smi and special sentinel values carry no heap reference and are
    /// silently ignored.
    fn trace(&self, tracer: &mut Tracer) {
        if self.is_heap_ptr() {
            let ptr = (self.0 & POINTER_MASK) as *mut u8;
            // SAFETY: `is_heap_ptr()` guarantees the lower 48 bits hold a
            // valid, aligned heap-object address.
            unsafe { tracer.mark_raw(ptr) };
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Size guarantee ──────────────────────────────────────────────────

    #[test]
    fn test_size_is_8_bytes() {
        assert_eq!(std::mem::size_of::<NanBoxedValue>(), 8);
    }

    // ── Smi round-trips ─────────────────────────────────────────────────

    #[test]
    fn test_smi_zero() {
        let v = NanBoxedValue::from_smi(0);
        assert!(v.is_smi());
        assert!(!v.is_special());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.as_smi(), Some(0));
    }

    #[test]
    fn test_smi_positive() {
        let v = NanBoxedValue::from_smi(42);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), Some(42));
    }

    #[test]
    fn test_smi_negative() {
        let v = NanBoxedValue::from_smi(-1);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), Some(-1));
    }

    #[test]
    fn test_smi_i32_max() {
        let v = NanBoxedValue::from_smi(i32::MAX);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), Some(i32::MAX));
    }

    #[test]
    fn test_smi_i32_min() {
        let v = NanBoxedValue::from_smi(i32::MIN);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), Some(i32::MIN));
    }

    #[test]
    fn test_smi_i31_max() {
        let max = i32::MAX >> 1; // 2^30 - 1
        let v = NanBoxedValue::from_smi(max);
        assert_eq!(v.as_smi(), Some(max));
    }

    #[test]
    fn test_smi_i31_min() {
        let min = i32::MIN >> 1; // -(2^30)
        let v = NanBoxedValue::from_smi(min);
        assert_eq!(v.as_smi(), Some(min));
    }

    // ── Special values ──────────────────────────────────────────────────

    #[test]
    fn test_undefined() {
        let v = NanBoxedValue::undefined();
        assert!(v.is_undefined());
        assert!(v.is_special());
        assert!(v.is_nullish());
        assert!(!v.is_smi());
        assert!(!v.is_null());
        assert!(!v.is_boolean());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.raw(), UNDEFINED_BITS);
    }

    #[test]
    fn test_null() {
        let v = NanBoxedValue::null();
        assert!(v.is_null());
        assert!(v.is_special());
        assert!(v.is_nullish());
        assert!(!v.is_undefined());
        assert!(!v.is_smi());
        assert!(!v.is_boolean());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.raw(), NULL_BITS);
    }

    #[test]
    fn test_true() {
        let v = NanBoxedValue::from_bool(true);
        assert!(v.is_true());
        assert!(v.is_boolean());
        assert!(v.is_special());
        assert!(!v.is_false());
        assert!(!v.is_smi());
        assert!(!v.is_nullish());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(v.raw(), TRUE_BITS);
    }

    #[test]
    fn test_false() {
        let v = NanBoxedValue::from_bool(false);
        assert!(v.is_false());
        assert!(v.is_boolean());
        assert!(v.is_special());
        assert!(!v.is_true());
        assert!(!v.is_smi());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.as_bool(), Some(false));
        assert_eq!(v.raw(), FALSE_BITS);
    }

    #[test]
    fn test_hole() {
        let v = NanBoxedValue::hole();
        assert!(v.is_hole());
        assert!(v.is_special());
        assert!(!v.is_smi());
        assert!(!v.is_boolean());
        assert!(!v.is_heap_ptr());
        assert_eq!(v.raw(), HOLE_BITS);
    }

    // ── Heap-pointer round-trips ────────────────────────────────────────

    #[test]
    fn test_heap_ptr_object_tag() {
        let mut obj = HeapObject::new_null();
        let ptr = &mut obj as *mut HeapObject;
        // SAFETY: ptr is valid and aligned.
        let v = unsafe { NanBoxedValue::from_heap_ptr(ptr, HeapTypeTag::Object) };
        assert!(v.is_heap_ptr());
        assert!(!v.is_smi());
        assert!(!v.is_special());
        assert_eq!(v.heap_type_tag(), Some(HeapTypeTag::Object));
        // SAFETY: obj is still live.
        let recovered = unsafe { v.as_heap_ptr() };
        assert_eq!(recovered, Some(ptr));
    }

    #[test]
    fn test_heap_ptr_all_tags() {
        let tags = [
            HeapTypeTag::Object,
            HeapTypeTag::String,
            HeapTypeTag::Symbol,
            HeapTypeTag::BigInt,
            HeapTypeTag::Function,
            HeapTypeTag::Array,
        ];
        for &tag in &tags {
            let mut obj = HeapObject::new_null();
            let ptr = &mut obj as *mut HeapObject;
            // SAFETY: ptr is valid and aligned.
            let v = unsafe { NanBoxedValue::from_heap_ptr(ptr, tag) };
            assert!(v.is_heap_ptr(), "tag {tag:?} should be heap ptr");
            assert_eq!(v.heap_type_tag(), Some(tag), "tag {tag:?} round-trip");
            // SAFETY: obj is still live.
            let recovered = unsafe { v.as_heap_ptr() };
            assert_eq!(recovered, Some(ptr), "pointer {tag:?} round-trip");
        }
    }

    // ── Cross-type discrimination ───────────────────────────────────────

    #[test]
    fn test_smi_is_not_special_or_heap() {
        for i in [0, 1, -1, 42, -42, i32::MAX, i32::MIN] {
            let v = NanBoxedValue::from_smi(i);
            assert!(v.is_smi(), "Smi({i}) must be smi");
            assert!(!v.is_special(), "Smi({i}) must not be special");
            assert!(!v.is_heap_ptr(), "Smi({i}) must not be heap ptr");
            assert_eq!(v.as_bool(), None, "Smi({i}) as_bool must be None");
            assert!(!v.is_undefined());
            assert!(!v.is_null());
        }
    }

    #[test]
    fn test_specials_are_not_smi_or_heap() {
        let specials = [
            NanBoxedValue::undefined(),
            NanBoxedValue::null(),
            NanBoxedValue::from_bool(true),
            NanBoxedValue::from_bool(false),
            NanBoxedValue::hole(),
        ];
        for v in specials {
            assert!(v.is_special(), "{v:?} must be special");
            assert!(!v.is_smi(), "{v:?} must not be smi");
            assert!(!v.is_heap_ptr(), "{v:?} must not be heap ptr");
            assert_eq!(v.as_smi(), None, "{v:?} as_smi must be None");
        }
    }

    #[test]
    fn test_heap_ptr_is_not_smi_or_special() {
        let mut obj = HeapObject::new_null();
        let ptr = &mut obj as *mut HeapObject;
        // SAFETY: ptr is valid.
        let v = unsafe { NanBoxedValue::from_heap_ptr(ptr, HeapTypeTag::Function) };
        assert!(v.is_heap_ptr());
        assert!(!v.is_smi());
        assert!(!v.is_special());
        assert_eq!(v.as_smi(), None);
        assert_eq!(v.as_bool(), None);
    }

    // ── All special sentinels are distinct ───────────────────────────────

    #[test]
    fn test_all_specials_distinct() {
        let values = [
            NanBoxedValue::undefined(),
            NanBoxedValue::null(),
            NanBoxedValue::from_bool(true),
            NanBoxedValue::from_bool(false),
            NanBoxedValue::hole(),
        ];
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b, "specials at index {i} and {j} must differ");
                }
            }
        }
    }

    // ── JsValue round-trip conversions ──────────────────────────────────

    #[test]
    fn test_jsvalue_undefined_round_trip() {
        let orig = JsValue::Undefined;
        let boxed = NanBoxedValue::try_from_js_value(&orig).unwrap();
        assert!(boxed.is_undefined());
        // SAFETY: no heap pointers involved.
        let back = unsafe { boxed.to_js_value() }.unwrap();
        assert_eq!(back, JsValue::Undefined);
    }

    #[test]
    fn test_jsvalue_null_round_trip() {
        let orig = JsValue::Null;
        let boxed = NanBoxedValue::try_from_js_value(&orig).unwrap();
        assert!(boxed.is_null());
        // SAFETY: no heap pointers involved.
        let back = unsafe { boxed.to_js_value() }.unwrap();
        assert_eq!(back, JsValue::Null);
    }

    #[test]
    fn test_jsvalue_bool_round_trip() {
        for b in [true, false] {
            let orig = JsValue::Boolean(b);
            let boxed = NanBoxedValue::try_from_js_value(&orig).unwrap();
            assert_eq!(boxed.as_bool(), Some(b));
            // SAFETY: no heap pointers involved.
            let back = unsafe { boxed.to_js_value() }.unwrap();
            assert_eq!(back, JsValue::Boolean(b));
        }
    }

    #[test]
    fn test_jsvalue_smi_round_trip() {
        for i in [0, 1, -1, 42, -42, i32::MAX, i32::MIN] {
            let orig = JsValue::Smi(i);
            let boxed = NanBoxedValue::try_from_js_value(&orig).unwrap();
            assert_eq!(boxed.as_smi(), Some(i));
            // SAFETY: no heap pointers involved.
            let back = unsafe { boxed.to_js_value() }.unwrap();
            assert_eq!(back, JsValue::Smi(i));
        }
    }

    #[test]
    fn test_jsvalue_object_round_trip() {
        let mut obj = HeapObject::new_null();
        let ptr = &mut obj as *mut HeapObject;
        let orig = JsValue::Object(ptr);
        let boxed = NanBoxedValue::try_from_js_value(&orig).unwrap();
        assert!(boxed.is_heap_ptr());
        assert_eq!(boxed.heap_type_tag(), Some(HeapTypeTag::Object));
        // SAFETY: obj is still live.
        let back = unsafe { boxed.to_js_value() }.unwrap();
        assert_eq!(back, JsValue::Object(ptr));
    }

    #[test]
    fn test_jsvalue_unsupported_returns_none() {
        let unsupported = [
            JsValue::HeapNumber(3.14),
            JsValue::String("hello".into()),
            JsValue::Symbol(42),
            JsValue::BigInt(123),
        ];
        for v in &unsupported {
            assert!(
                NanBoxedValue::try_from_js_value(v).is_none(),
                "{v:?} should not be convertible"
            );
        }
    }

    // ── Raw round-trip ──────────────────────────────────────────────────

    #[test]
    fn test_raw_round_trip() {
        let v = NanBoxedValue::from_smi(12345);
        // SAFETY: we just encoded a valid Smi.
        let v2 = unsafe { NanBoxedValue::from_raw(v.raw()) };
        assert_eq!(v, v2);
        assert_eq!(v2.as_smi(), Some(12345));
    }

    // ── Debug formatting ────────────────────────────────────────────────

    #[test]
    fn test_debug_formatting() {
        assert_eq!(
            format!("{:?}", NanBoxedValue::undefined()),
            "NanBoxedValue(undefined)"
        );
        assert_eq!(
            format!("{:?}", NanBoxedValue::null()),
            "NanBoxedValue(null)"
        );
        assert_eq!(
            format!("{:?}", NanBoxedValue::from_bool(true)),
            "NanBoxedValue(true)"
        );
        assert_eq!(
            format!("{:?}", NanBoxedValue::from_bool(false)),
            "NanBoxedValue(false)"
        );
        assert_eq!(
            format!("{:?}", NanBoxedValue::hole()),
            "NanBoxedValue(hole)"
        );
        assert!(format!("{:?}", NanBoxedValue::from_smi(7)).contains("Smi(7)"));
    }

    // ── Trace implementation ────────────────────────────────────────────

    #[test]
    fn test_trace_smi_does_nothing() {
        let mut tracer = Tracer::new();
        let v = NanBoxedValue::from_smi(99);
        v.trace(&mut tracer);
        assert!(tracer.gray_stack.is_empty());
    }

    #[test]
    fn test_trace_special_does_nothing() {
        let mut tracer = Tracer::new();
        NanBoxedValue::undefined().trace(&mut tracer);
        NanBoxedValue::null().trace(&mut tracer);
        NanBoxedValue::from_bool(true).trace(&mut tracer);
        NanBoxedValue::hole().trace(&mut tracer);
        assert!(tracer.gray_stack.is_empty());
    }

    #[test]
    fn test_trace_heap_ptr_marks() {
        let mut obj = HeapObject::new_null();
        let ptr = &mut obj as *mut HeapObject;
        // SAFETY: ptr is valid.
        let v = unsafe { NanBoxedValue::from_heap_ptr(ptr, HeapTypeTag::Object) };

        let mut tracer = Tracer::new();
        v.trace(&mut tracer);
        assert_eq!(tracer.gray_stack.len(), 1);
        assert_eq!(tracer.gray_stack[0], ptr as *mut u8);
    }
}

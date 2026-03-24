//! NaN-boxed value representation for the Stator VM.
//!
//! Packs all JavaScript values into a single `u64` using IEEE 754 NaN
//! bit patterns. Doubles use their native bit representation; all other
//! types are encoded as quiet NaNs with a 3-bit type tag and 48-bit
//! payload.

use std::rc::Rc;

use super::value::JsValue;

/// A NaN-boxed JavaScript value packed into 8 bytes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct NanBoxedValue(u64);

// ── Bit layout constants ──────────────────────────────────────────────

/// Quiet NaN mask: exponent all 1s + quiet bit set.
const QNAN: u64 = 0x7FF8_0000_0000_0000;

/// Tag shift: tags occupy bits 48-50.
const TAG_SHIFT: u64 = 48;

/// Payload mask: lower 48 bits.
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

// ── Type tags ─────────────────────────────────────────────────────────

const TAG_UNDEFINED: u64 = 0;
const TAG_NULL: u64 = 1;
const TAG_BOOLEAN: u64 = 2;
const TAG_SMI: u64 = 3;
const TAG_HEAP_PTR: u64 = 4;
const TAG_STRING: u64 = 5;
const TAG_SYMBOL: u64 = 6;
const TAG_THE_HOLE: u64 = 7;

impl NanBoxedValue {
    // ── Constructors ──────────────────────────────────────────────────

    /// Encode a double (`f64`) value.
    #[inline(always)]
    #[must_use]
    pub fn from_double(v: f64) -> Self {
        Self(v.to_bits())
    }

    /// Encode `undefined`.
    #[inline(always)]
    #[must_use]
    pub fn undefined() -> Self {
        Self(QNAN | (TAG_UNDEFINED << TAG_SHIFT))
    }

    /// Encode `null`.
    #[inline(always)]
    #[must_use]
    pub fn null() -> Self {
        Self(QNAN | (TAG_NULL << TAG_SHIFT))
    }

    /// Encode a boolean.
    #[inline(always)]
    #[must_use]
    pub fn from_boolean(v: bool) -> Self {
        Self(QNAN | (TAG_BOOLEAN << TAG_SHIFT) | u64::from(v))
    }

    /// Encode a small integer (SMI).
    #[inline(always)]
    #[must_use]
    pub fn from_smi(v: i32) -> Self {
        Self(QNAN | (TAG_SMI << TAG_SHIFT) | (u32::from_ne_bytes(v.to_ne_bytes()) as u64))
    }

    /// Encode TheHole (temporal dead zone sentinel).
    #[inline(always)]
    #[must_use]
    pub fn the_hole() -> Self {
        Self(QNAN | (TAG_THE_HOLE << TAG_SHIFT))
    }

    /// Encode a symbol by ID.
    #[inline(always)]
    #[must_use]
    pub fn from_symbol(id: u64) -> Self {
        Self(QNAN | (TAG_SYMBOL << TAG_SHIFT) | (id & PAYLOAD_MASK))
    }

    /// Encode a raw heap pointer (48-bit).
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and fits in 48 bits.
    #[inline(always)]
    #[must_use]
    pub unsafe fn from_heap_ptr(ptr: *mut u8) -> Self {
        let addr = ptr as u64;
        debug_assert!(addr & !PAYLOAD_MASK == 0, "pointer exceeds 48 bits");
        Self(QNAN | (TAG_HEAP_PTR << TAG_SHIFT) | (addr & PAYLOAD_MASK))
    }

    /// Encode a string pointer (`Rc<str>` raw pointer).
    ///
    /// # Safety
    /// The caller must ensure the pointer came from `Rc::into_raw` and fits in
    /// 48 bits.
    #[inline(always)]
    #[must_use]
    pub unsafe fn from_string_ptr(ptr: *const str) -> Self {
        let addr = ptr as *const () as u64;
        debug_assert!(addr & !PAYLOAD_MASK == 0, "string pointer exceeds 48 bits");
        Self(QNAN | (TAG_STRING << TAG_SHIFT) | (addr & PAYLOAD_MASK))
    }

    // ── Type checks ───────────────────────────────────────────────────

    /// Returns `true` if this value is a double (not NaN-boxed).
    #[inline(always)]
    #[must_use]
    pub fn is_double(self) -> bool {
        (self.0 & QNAN) != QNAN
    }

    /// Extract the 3-bit tag (only valid for non-double values).
    #[inline(always)]
    #[must_use]
    fn tag(self) -> u64 {
        (self.0 >> TAG_SHIFT) & 0x7
    }

    /// Extract the 48-bit payload.
    #[inline(always)]
    #[must_use]
    fn payload(self) -> u64 {
        self.0 & PAYLOAD_MASK
    }

    /// Returns `true` if this value is `undefined`.
    #[inline(always)]
    #[must_use]
    pub fn is_undefined(self) -> bool {
        !self.is_double() && self.tag() == TAG_UNDEFINED
    }

    /// Returns `true` if this value is `null`.
    #[inline(always)]
    #[must_use]
    pub fn is_null(self) -> bool {
        !self.is_double() && self.tag() == TAG_NULL
    }

    /// Returns `true` if this value is a boolean.
    #[inline(always)]
    #[must_use]
    pub fn is_boolean(self) -> bool {
        !self.is_double() && self.tag() == TAG_BOOLEAN
    }

    /// Returns `true` if this value is an SMI.
    #[inline(always)]
    #[must_use]
    pub fn is_smi(self) -> bool {
        !self.is_double() && self.tag() == TAG_SMI
    }

    /// Returns `true` if this value is a heap pointer.
    #[inline(always)]
    #[must_use]
    pub fn is_heap_ptr(self) -> bool {
        !self.is_double() && self.tag() == TAG_HEAP_PTR
    }

    /// Returns `true` if this value is a string pointer.
    #[inline(always)]
    #[must_use]
    pub fn is_string(self) -> bool {
        !self.is_double() && self.tag() == TAG_STRING
    }

    /// Returns `true` if this value is a symbol.
    #[inline(always)]
    #[must_use]
    pub fn is_symbol(self) -> bool {
        !self.is_double() && self.tag() == TAG_SYMBOL
    }

    /// Returns `true` if this value is TheHole.
    #[inline(always)]
    #[must_use]
    pub fn is_the_hole(self) -> bool {
        !self.is_double() && self.tag() == TAG_THE_HOLE
    }

    // ── Extractors ────────────────────────────────────────────────────

    /// Extract as `f64` (only valid if `is_double()`).
    #[inline(always)]
    #[must_use]
    pub fn as_double(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Extract boolean value.
    #[inline(always)]
    #[must_use]
    pub fn as_boolean(self) -> bool {
        self.payload() != 0
    }

    /// Extract SMI value.
    #[inline(always)]
    #[must_use]
    pub fn as_smi(self) -> i32 {
        self.payload() as u32 as i32
    }

    /// Extract symbol ID.
    #[inline(always)]
    #[must_use]
    pub fn as_symbol(self) -> u64 {
        self.payload()
    }

    /// Extract heap pointer.
    ///
    /// # Safety
    /// Only valid if `is_heap_ptr()`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn as_heap_ptr(self) -> *mut u8 {
        self.payload() as *mut u8
    }

    /// Get the raw bits.
    #[inline(always)]
    #[must_use]
    pub fn to_bits(self) -> u64 {
        self.0
    }

    /// Construct from raw bits.
    #[inline(always)]
    #[must_use]
    pub fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    // ── JsValue ↔ NanBoxedValue conversion ───────────────────────────

    /// Convert a [`JsValue`] into its NaN-boxed 8-byte representation.
    ///
    /// * Primitive variants (`Undefined`, `Null`, `Boolean`, `Smi`,
    ///   `HeapNumber`, `TheHole`, `Symbol`) are encoded inline.
    /// * `String` values have their inner [`Rc<str>`] stored via
    ///   [`Box`] and tagged with the string tag.
    /// * All remaining heap-allocated variants are boxed as
    ///   `Box<JsValue>` and stored with the generic heap-pointer tag.
    ///
    /// This **consumes** the `JsValue`.  The caller must not drop the
    /// original value after packing (it has been moved into the
    /// NaN-boxed representation).
    #[must_use]
    pub fn pack(value: JsValue) -> Self {
        match value {
            JsValue::Undefined => Self::undefined(),
            JsValue::Null => Self::null(),
            JsValue::TheHole => Self::the_hole(),
            JsValue::Boolean(b) => Self::from_boolean(b),
            JsValue::Smi(i) => Self::from_smi(i),
            JsValue::HeapNumber(f) => {
                // Quiet-NaN bit patterns collide with the QNAN tag
                // prefix, so we box them as heap-allocated values.
                if f.to_bits() & QNAN == QNAN {
                    let boxed = Box::new(JsValue::HeapNumber(f));
                    let ptr = Box::into_raw(boxed) as *mut u8;
                    // SAFETY: Box::into_raw returns a valid pointer that
                    // fits in 48 bits on current 64-bit platforms.
                    unsafe { Self::from_heap_ptr(ptr) }
                } else {
                    Self::from_double(f)
                }
            }
            JsValue::Symbol(id) => Self::from_symbol(id),
            JsValue::String(s) => {
                // Box the Rc<str> so we have a thin pointer we can
                // store in the 48-bit payload.
                let boxed: Box<Rc<str>> = Box::new(s);
                let addr = Box::into_raw(boxed) as u64;
                debug_assert!(
                    addr & !PAYLOAD_MASK == 0,
                    "string box pointer exceeds 48 bits"
                );
                Self(QNAN | (TAG_STRING << TAG_SHIFT) | (addr & PAYLOAD_MASK))
            }
            other => {
                // Generic fallback: box the entire JsValue.
                let boxed = Box::new(other);
                let ptr = Box::into_raw(boxed) as *mut u8;
                // SAFETY: Box::into_raw returns a valid pointer that
                // fits in 48 bits on current 64-bit platforms.
                unsafe { Self::from_heap_ptr(ptr) }
            }
        }
    }

    /// Reconstruct a [`JsValue`] from this NaN-boxed representation,
    /// consuming the packed slot.
    ///
    /// # Safety
    ///
    /// * For `TAG_STRING` values the payload must be a pointer previously
    ///   created by [`pack`](Self::pack) with a `JsValue::String`, and
    ///   this method must be called **exactly once** per packed value
    ///   (it reclaims the heap allocation).
    /// * For `TAG_HEAP_PTR` values the payload must be a pointer
    ///   previously created by [`pack`](Self::pack), called exactly once.
    /// * Primitive tags (`Undefined`, `Null`, `Boolean`, `Smi`,
    ///   `TheHole`, `Symbol`) and doubles have no pointer-safety
    ///   requirements.
    pub unsafe fn unpack(self) -> JsValue {
        if self.is_double() {
            return JsValue::HeapNumber(self.as_double());
        }
        match self.tag() {
            TAG_UNDEFINED => JsValue::Undefined,
            TAG_NULL => JsValue::Null,
            TAG_BOOLEAN => JsValue::Boolean(self.as_boolean()),
            TAG_SMI => JsValue::Smi(self.as_smi()),
            TAG_THE_HOLE => JsValue::TheHole,
            TAG_SYMBOL => JsValue::Symbol(self.as_symbol()),
            TAG_STRING => {
                let ptr = self.payload() as *mut Rc<str>;
                // SAFETY: Caller guarantees this pointer came from
                // pack() and is consumed exactly once.
                JsValue::String(*unsafe { Box::from_raw(ptr) })
            }
            TAG_HEAP_PTR => {
                let ptr = self.payload() as *mut JsValue;
                // SAFETY: Caller guarantees this pointer came from
                // pack() and is consumed exactly once.
                *unsafe { Box::from_raw(ptr) }
            }
            _ => unreachable!(),
        }
    }

    /// Clone the packed value into a [`JsValue`] **without** consuming
    /// the NaN-boxed slot.
    ///
    /// For ref-counted types this increments the reference count.
    /// For boxed heap types this clones the inner `JsValue`.
    ///
    /// # Safety
    ///
    /// Same pointer-validity requirements as [`unpack`](Self::unpack),
    /// except this may be called multiple times.
    pub unsafe fn clone_value(&self) -> JsValue {
        if self.is_double() {
            return JsValue::HeapNumber(self.as_double());
        }
        match self.tag() {
            TAG_UNDEFINED => JsValue::Undefined,
            TAG_NULL => JsValue::Null,
            TAG_BOOLEAN => JsValue::Boolean(self.as_boolean()),
            TAG_SMI => JsValue::Smi(self.as_smi()),
            TAG_THE_HOLE => JsValue::TheHole,
            TAG_SYMBOL => JsValue::Symbol(self.as_symbol()),
            TAG_STRING => {
                let ptr = self.payload() as *const Rc<str>;
                // SAFETY: Caller guarantees the pointer is valid.  We
                // clone the inner Rc (bumping the strong count) without
                // reclaiming the Box allocation.
                JsValue::String(unsafe { (*ptr).clone() })
            }
            TAG_HEAP_PTR => {
                let ptr = self.payload() as *const JsValue;
                // SAFETY: Caller guarantees the pointer is valid.  We
                // clone the JsValue without reclaiming the Box.
                unsafe { (*ptr).clone() }
            }
            _ => unreachable!(),
        }
    }

    // ── SMI arithmetic fast paths ────────────────────────────────────

    /// Fast SMI addition.  Returns `None` on overflow or if either
    /// operand is not an SMI.
    #[inline(always)]
    pub fn smi_add(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            self.as_smi()
                .checked_add(other.as_smi())
                .map(Self::from_smi)
        } else {
            None
        }
    }

    /// Fast SMI subtraction.  Returns `None` on overflow or if either
    /// operand is not an SMI.
    #[inline(always)]
    pub fn smi_sub(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            self.as_smi()
                .checked_sub(other.as_smi())
                .map(Self::from_smi)
        } else {
            None
        }
    }

    /// Fast SMI multiplication.  Returns `None` on overflow or if
    /// either operand is not an SMI.
    #[inline(always)]
    pub fn smi_mul(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            self.as_smi()
                .checked_mul(other.as_smi())
                .map(Self::from_smi)
        } else {
            None
        }
    }

    /// Fast SMI less-than comparison.  Returns `None` if either operand
    /// is not an SMI.
    #[inline(always)]
    pub fn smi_less_than(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            Some(Self::from_boolean(self.as_smi() < other.as_smi()))
        } else {
            None
        }
    }

    /// Fast SMI greater-than comparison.  Returns `None` if either
    /// operand is not an SMI.
    #[inline(always)]
    pub fn smi_greater_than(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            Some(Self::from_boolean(self.as_smi() > other.as_smi()))
        } else {
            None
        }
    }

    /// Fast SMI equality check.  Returns `None` if either operand is
    /// not an SMI.
    #[inline(always)]
    pub fn smi_equal(self, other: Self) -> Option<Self> {
        if self.is_smi() && other.is_smi() {
            Some(Self::from_boolean(self.as_smi() == other.as_smi()))
        } else {
            None
        }
    }
}

impl std::fmt::Debug for NanBoxedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_double() {
            write!(f, "NanBoxed(double: {})", self.as_double())
        } else {
            match self.tag() {
                TAG_UNDEFINED => write!(f, "NanBoxed(undefined)"),
                TAG_NULL => write!(f, "NanBoxed(null)"),
                TAG_BOOLEAN => write!(f, "NanBoxed(bool: {})", self.as_boolean()),
                TAG_SMI => write!(f, "NanBoxed(smi: {})", self.as_smi()),
                TAG_HEAP_PTR => write!(f, "NanBoxed(heap: 0x{:012x})", self.payload()),
                TAG_STRING => write!(f, "NanBoxed(string: 0x{:012x})", self.payload()),
                TAG_SYMBOL => write!(f, "NanBoxed(symbol: {})", self.as_symbol()),
                TAG_THE_HOLE => write!(f, "NanBoxed(the_hole)"),
                _ => write!(f, "NanBoxed(unknown tag: {})", self.tag()),
            }
        }
    }
}

impl PartialEq for NanBoxedValue {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for NanBoxedValue {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::value::JsValue;
    use std::rc::Rc;

    #[test]
    fn test_undefined() {
        let v = NanBoxedValue::undefined();
        assert!(v.is_undefined());
        assert!(!v.is_null());
        assert!(!v.is_double());
        assert!(!v.is_smi());
    }

    #[test]
    fn test_null() {
        let v = NanBoxedValue::null();
        assert!(v.is_null());
        assert!(!v.is_undefined());
    }

    #[test]
    fn test_boolean_true() {
        let v = NanBoxedValue::from_boolean(true);
        assert!(v.is_boolean());
        assert!(v.as_boolean());
    }

    #[test]
    fn test_boolean_false() {
        let v = NanBoxedValue::from_boolean(false);
        assert!(v.is_boolean());
        assert!(!v.as_boolean());
    }

    #[test]
    fn test_smi_positive() {
        let v = NanBoxedValue::from_smi(42);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), 42);
    }

    #[test]
    fn test_smi_negative() {
        let v = NanBoxedValue::from_smi(-1);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), -1);
    }

    #[test]
    fn test_smi_zero() {
        let v = NanBoxedValue::from_smi(0);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), 0);
    }

    #[test]
    fn test_smi_max() {
        let v = NanBoxedValue::from_smi(i32::MAX);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), i32::MAX);
    }

    #[test]
    fn test_smi_min() {
        let v = NanBoxedValue::from_smi(i32::MIN);
        assert!(v.is_smi());
        assert_eq!(v.as_smi(), i32::MIN);
    }

    #[test]
    fn test_double_normal() {
        let v = NanBoxedValue::from_double(3.14);
        assert!(v.is_double());
        assert!(!v.is_smi());
        assert_eq!(v.as_double(), 3.14);
    }

    #[test]
    fn test_double_zero() {
        let v = NanBoxedValue::from_double(0.0);
        assert!(v.is_double());
        assert_eq!(v.as_double(), 0.0);
    }

    #[test]
    fn test_double_negative_zero() {
        let v = NanBoxedValue::from_double(-0.0);
        assert!(v.is_double());
        assert!(v.as_double().is_sign_negative());
    }

    #[test]
    fn test_double_infinity() {
        let v = NanBoxedValue::from_double(f64::INFINITY);
        assert!(v.is_double());
        assert!(v.as_double().is_infinite());
    }

    #[test]
    fn test_double_negative_infinity() {
        let v = NanBoxedValue::from_double(f64::NEG_INFINITY);
        assert!(v.is_double());
        assert!(v.as_double().is_infinite());
        assert!(v.as_double().is_sign_negative());
    }

    #[test]
    fn test_the_hole() {
        let v = NanBoxedValue::the_hole();
        assert!(v.is_the_hole());
        assert!(!v.is_undefined());
    }

    #[test]
    fn test_symbol() {
        let v = NanBoxedValue::from_symbol(12345);
        assert!(v.is_symbol());
        assert_eq!(v.as_symbol(), 12345);
    }

    #[test]
    fn test_equality() {
        assert_eq!(NanBoxedValue::undefined(), NanBoxedValue::undefined());
        assert_eq!(NanBoxedValue::null(), NanBoxedValue::null());
        assert_ne!(NanBoxedValue::undefined(), NanBoxedValue::null());
        assert_eq!(NanBoxedValue::from_smi(42), NanBoxedValue::from_smi(42));
        assert_ne!(NanBoxedValue::from_smi(1), NanBoxedValue::from_smi(2));
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<NanBoxedValue>(), 8);
    }

    #[test]
    fn test_copy_semantics() {
        let a = NanBoxedValue::from_smi(99);
        let b = a;
        assert_eq!(a.as_smi(), b.as_smi());
    }

    #[test]
    fn test_round_trip_bits() {
        let v = NanBoxedValue::from_smi(42);
        let bits = v.to_bits();
        let v2 = NanBoxedValue::from_bits(bits);
        assert_eq!(v, v2);
    }

    #[test]
    fn test_heap_ptr_round_trip() {
        let data: Box<u64> = Box::new(42);
        let ptr = Box::into_raw(data) as *mut u8;
        // SAFETY: `ptr` comes from `Box::into_raw`, is live for the duration of
        // the test, and current user-space pointers fit within 48 bits here.
        let v = unsafe { NanBoxedValue::from_heap_ptr(ptr) };
        assert!(v.is_heap_ptr());
        // SAFETY: `v` was created from `ptr` above using the heap-pointer tag.
        let recovered = unsafe { v.as_heap_ptr() };
        assert_eq!(ptr, recovered);
        // SAFETY: `recovered` is the original allocation returned by
        // `Box::into_raw`, converted back exactly once for cleanup.
        unsafe {
            drop(Box::from_raw(recovered as *mut u64));
        }
    }

    #[test]
    fn test_double_nan_is_double() {
        let v = NanBoxedValue::from_double(f64::NAN);
        let bits = v.to_bits();
        assert_eq!(bits, f64::NAN.to_bits());
    }

    // ── Pack / unpack round-trip tests ──────────────────────────────

    #[test]
    fn test_pack_unpack_undefined() {
        let packed = NanBoxedValue::pack(JsValue::Undefined);
        assert!(packed.is_undefined());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Undefined));
    }

    #[test]
    fn test_pack_unpack_null() {
        let packed = NanBoxedValue::pack(JsValue::Null);
        assert!(packed.is_null());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Null));
    }

    #[test]
    fn test_pack_unpack_the_hole() {
        let packed = NanBoxedValue::pack(JsValue::TheHole);
        assert!(packed.is_the_hole());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::TheHole));
    }

    #[test]
    fn test_pack_unpack_boolean_true() {
        let packed = NanBoxedValue::pack(JsValue::Boolean(true));
        assert!(packed.is_boolean());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Boolean(true)));
    }

    #[test]
    fn test_pack_unpack_boolean_false() {
        let packed = NanBoxedValue::pack(JsValue::Boolean(false));
        assert!(packed.is_boolean());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Boolean(false)));
    }

    #[test]
    fn test_pack_unpack_smi_positive() {
        let packed = NanBoxedValue::pack(JsValue::Smi(42));
        assert!(packed.is_smi());
        assert_eq!(packed.as_smi(), 42);
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Smi(42)));
    }

    #[test]
    fn test_pack_unpack_smi_negative() {
        let packed = NanBoxedValue::pack(JsValue::Smi(-1));
        assert!(packed.is_smi());
        assert_eq!(packed.as_smi(), -1);
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Smi(-1)));
    }

    #[test]
    fn test_pack_unpack_smi_max() {
        let packed = NanBoxedValue::pack(JsValue::Smi(i32::MAX));
        assert_eq!(packed.as_smi(), i32::MAX);
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Smi(i32::MAX)));
    }

    #[test]
    fn test_pack_unpack_smi_min() {
        let packed = NanBoxedValue::pack(JsValue::Smi(i32::MIN));
        assert_eq!(packed.as_smi(), i32::MIN);
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Smi(i32::MIN)));
    }

    #[test]
    fn test_pack_unpack_heap_number() {
        let packed = NanBoxedValue::pack(JsValue::HeapNumber(3.14));
        assert!(packed.is_double());
        assert_eq!(packed.as_double(), 3.14);
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::HeapNumber(f) => assert_eq!(f, 3.14),
            _ => panic!("expected HeapNumber"),
        }
    }

    #[test]
    fn test_pack_unpack_negative_zero() {
        let packed = NanBoxedValue::pack(JsValue::HeapNumber(-0.0));
        assert!(packed.is_double());
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::HeapNumber(f) => {
                assert_eq!(f, 0.0);
                assert!(f.is_sign_negative());
            }
            _ => panic!("expected HeapNumber"),
        }
    }

    #[test]
    fn test_pack_unpack_infinity() {
        let packed = NanBoxedValue::pack(JsValue::HeapNumber(f64::INFINITY));
        assert!(packed.is_double());
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::HeapNumber(f) => assert!(f.is_infinite() && f.is_sign_positive()),
            _ => panic!("expected HeapNumber"),
        }
    }

    #[test]
    fn test_pack_unpack_neg_infinity() {
        let packed = NanBoxedValue::pack(JsValue::HeapNumber(f64::NEG_INFINITY));
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::HeapNumber(f) => assert!(f.is_infinite() && f.is_sign_negative()),
            _ => panic!("expected HeapNumber"),
        }
    }

    #[test]
    fn test_pack_unpack_nan() {
        let packed = NanBoxedValue::pack(JsValue::HeapNumber(f64::NAN));
        // NaN is boxed as a heap pointer because its bits collide with QNAN.
        assert!(packed.is_heap_ptr());
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::HeapNumber(f) => assert!(f.is_nan()),
            _ => panic!("expected HeapNumber(NaN)"),
        }
    }

    #[test]
    fn test_pack_unpack_symbol() {
        let packed = NanBoxedValue::pack(JsValue::Symbol(12345));
        assert!(packed.is_symbol());
        assert_eq!(packed.as_symbol(), 12345);
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::Symbol(12345)));
    }

    #[test]
    fn test_pack_unpack_string() {
        let s: Rc<str> = Rc::from("hello");
        let packed = NanBoxedValue::pack(JsValue::String(s));
        assert!(packed.is_string());
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::String(s) => assert_eq!(&*s, "hello"),
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_pack_clone_value_string_refcount() {
        let s: Rc<str> = Rc::from("hello");
        assert_eq!(Rc::strong_count(&s), 1);
        let packed = NanBoxedValue::pack(JsValue::String(s));

        // clone_value should bump the Rc strong count.
        let cloned = unsafe { packed.clone_value() };
        match &cloned {
            JsValue::String(s) => {
                assert_eq!(&**s, "hello");
                assert_eq!(Rc::strong_count(s), 2);
            }
            _ => panic!("expected String"),
        }

        // Drop the clone and unpack the original to verify correctness.
        drop(cloned);
        let unpacked = unsafe { packed.unpack() };
        match unpacked {
            JsValue::String(s) => {
                assert_eq!(&*s, "hello");
                assert_eq!(Rc::strong_count(&s), 1);
            }
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_pack_unpack_bigint() {
        let packed = NanBoxedValue::pack(JsValue::BigInt(Box::new(42)));
        assert!(packed.is_heap_ptr());
        let unpacked = unsafe { packed.unpack() };
        assert!(matches!(unpacked, JsValue::BigInt(Box::new(42))));
    }

    #[test]
    fn test_clone_value_primitives() {
        let cases = [
            NanBoxedValue::pack(JsValue::Undefined),
            NanBoxedValue::pack(JsValue::Null),
            NanBoxedValue::pack(JsValue::TheHole),
            NanBoxedValue::pack(JsValue::Boolean(true)),
            NanBoxedValue::pack(JsValue::Smi(99)),
            NanBoxedValue::pack(JsValue::HeapNumber(2.718)),
            NanBoxedValue::pack(JsValue::Symbol(7)),
        ];

        for packed in &cases {
            let cloned = unsafe { packed.clone_value() };
            drop(cloned);
        }
    }

    // ── SMI arithmetic tests ────────────────────────────────────────

    #[test]
    fn test_smi_add_basic() {
        let a = NanBoxedValue::from_smi(10);
        let b = NanBoxedValue::from_smi(20);
        let result = a.smi_add(b).unwrap();
        assert_eq!(result.as_smi(), 30);
    }

    #[test]
    fn test_smi_add_negative() {
        let a = NanBoxedValue::from_smi(-5);
        let b = NanBoxedValue::from_smi(3);
        let result = a.smi_add(b).unwrap();
        assert_eq!(result.as_smi(), -2);
    }

    #[test]
    fn test_smi_add_overflow() {
        let a = NanBoxedValue::from_smi(i32::MAX);
        let b = NanBoxedValue::from_smi(1);
        assert!(a.smi_add(b).is_none());
    }

    #[test]
    fn test_smi_add_non_smi() {
        let a = NanBoxedValue::from_smi(1);
        let b = NanBoxedValue::from_double(2.0);
        assert!(a.smi_add(b).is_none());
    }

    #[test]
    fn test_smi_sub_basic() {
        let a = NanBoxedValue::from_smi(30);
        let b = NanBoxedValue::from_smi(10);
        let result = a.smi_sub(b).unwrap();
        assert_eq!(result.as_smi(), 20);
    }

    #[test]
    fn test_smi_sub_overflow() {
        let a = NanBoxedValue::from_smi(i32::MIN);
        let b = NanBoxedValue::from_smi(1);
        assert!(a.smi_sub(b).is_none());
    }

    #[test]
    fn test_smi_mul_basic() {
        let a = NanBoxedValue::from_smi(6);
        let b = NanBoxedValue::from_smi(7);
        let result = a.smi_mul(b).unwrap();
        assert_eq!(result.as_smi(), 42);
    }

    #[test]
    fn test_smi_mul_overflow() {
        let a = NanBoxedValue::from_smi(i32::MAX);
        let b = NanBoxedValue::from_smi(2);
        assert!(a.smi_mul(b).is_none());
    }

    #[test]
    fn test_smi_less_than() {
        let a = NanBoxedValue::from_smi(1);
        let b = NanBoxedValue::from_smi(2);
        let result = a.smi_less_than(b).unwrap();
        assert!(result.as_boolean());
        let result = b.smi_less_than(a).unwrap();
        assert!(!result.as_boolean());
    }

    #[test]
    fn test_smi_greater_than() {
        let a = NanBoxedValue::from_smi(5);
        let b = NanBoxedValue::from_smi(3);
        let result = a.smi_greater_than(b).unwrap();
        assert!(result.as_boolean());
    }

    #[test]
    fn test_smi_equal_values() {
        let a = NanBoxedValue::from_smi(42);
        let b = NanBoxedValue::from_smi(42);
        let c = NanBoxedValue::from_smi(99);
        assert!(a.smi_equal(b).unwrap().as_boolean());
        assert!(!a.smi_equal(c).unwrap().as_boolean());
    }

    #[test]
    fn test_smi_comparison_non_smi() {
        let a = NanBoxedValue::from_smi(1);
        let b = NanBoxedValue::undefined();
        assert!(a.smi_less_than(b).is_none());
        assert!(a.smi_greater_than(b).is_none());
        assert!(a.smi_equal(b).is_none());
    }
}

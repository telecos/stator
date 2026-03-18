//! NaN-boxed value representation for the Stator VM.
//!
//! Packs all JavaScript values into a single `u64` using IEEE 754 NaN
//! bit patterns. Doubles use their native bit representation; all other
//! types are encoded as quiet NaNs with a 3-bit type tag and 48-bit
//! payload.

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
}

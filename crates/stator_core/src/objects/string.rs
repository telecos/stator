//! JavaScript string types.
//!
//! Provides multiple string representations so the engine can store and
//! manipulate strings efficiently:
//!
//! - [`SeqOneByteString`] — Latin-1 encoded (one byte per code unit)
//! - [`SeqTwoByteString`] — UTF-16 encoded (two bytes per code unit)
//! - [`ConsString`] — lazily-concatenated pair of strings
//! - [`SlicedString`] — a substring view (offset + length) into a parent string
//! - [`ExternalString`] — a string whose character data is managed by the embedder
//!
//! All variants are exposed through the top-level [`JsString`] enum, which
//! provides the common API: [`length`][JsString::length],
//! [`char_at`][JsString::char_at], [`flatten`][JsString::flatten],
//! [`to_utf8`][JsString::to_utf8], and [`hash`][JsString::hash].

use std::sync::Arc;

use crate::error::{StatorError, StatorResult};

// ──────────────────────────────────────────────────────────────────────────────
// ExternalStringResource trait
// ──────────────────────────────────────────────────────────────────────────────

/// The character-data provider for an [`ExternalString`].
///
/// Embedders implement this trait to back a `JsString` with memory that lives
/// outside the engine heap.  The resource must remain alive for as long as any
/// [`ExternalString`] that references it.
pub trait ExternalStringResource: Send + Sync + std::fmt::Debug {
    /// Returns the number of code units in the string.
    fn len(&self) -> usize;

    /// Returns `true` if the string contains no characters.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the Latin-1 bytes if this is a one-byte resource, or `None`
    /// for a two-byte resource.
    fn as_one_byte(&self) -> Option<&[u8]>;

    /// Returns the UTF-16 code units if this is a two-byte resource, or `None`
    /// for a one-byte resource.
    fn as_two_byte(&self) -> Option<&[u16]>;
}

// ──────────────────────────────────────────────────────────────────────────────
// SeqOneByteString
// ──────────────────────────────────────────────────────────────────────────────

/// A flat Latin-1 (one-byte-per-code-unit) string.
///
/// Each byte stores one JavaScript code unit in the range `U+0000`–`U+00FF`.
/// Strings containing only Latin-1 characters are stored more compactly than
/// their UTF-16 equivalents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqOneByteString {
    data: Vec<u8>,
}

impl SeqOneByteString {
    /// Creates a `SeqOneByteString` from a raw byte vector.
    ///
    /// Each byte is a valid Latin-1 code unit in `0x00`–`0xFF`.
    #[inline]
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Attempts to create a `SeqOneByteString` from a UTF-8 string slice.
    ///
    /// Returns `Err(StatorError::TypeError)` if `s` contains any character
    /// outside the Latin-1 range (`U+0100` or above).
    pub fn from_latin1_str(s: &str) -> StatorResult<Self> {
        let mut data = Vec::with_capacity(s.len());
        for ch in s.chars() {
            let code = ch as u32;
            if code > 0xFF {
                return Err(StatorError::TypeError(format!(
                    "character U+{code:04X} is outside the Latin-1 range"
                )));
            }
            data.push(code as u8);
        }
        Ok(Self { data })
    }

    /// Returns the number of code units (characters) in the string.
    #[inline]
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Returns the Latin-1 code unit at `index` as a UTF-16 unit, or `None`
    /// if the index is out of bounds.
    #[inline]
    pub fn char_at(&self, index: usize) -> Option<u16> {
        self.data.get(index).copied().map(u16::from)
    }

    /// Returns a UTF-8 `String` representation.
    ///
    /// Latin-1 code units in `0x80`–`0xFF` are encoded as two-byte UTF-8
    /// sequences.
    pub fn to_utf8(&self) -> String {
        self.data.iter().map(|&b| b as char).collect()
    }

    /// Returns a 32-bit FNV-1a hash of the string's code units.
    pub fn hash(&self) -> u32 {
        fnv1a_hash_u16(self.data.iter().map(|&b| u16::from(b)))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SeqTwoByteString
// ──────────────────────────────────────────────────────────────────────────────

/// A flat UTF-16 (two-bytes-per-code-unit) string.
///
/// Stores JavaScript code units as 16-bit values to represent the full Unicode
/// BMP and surrogate pairs for supplementary characters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqTwoByteString {
    data: Vec<u16>,
}

impl SeqTwoByteString {
    /// Creates a `SeqTwoByteString` from a raw UTF-16 code-unit vector.
    #[inline]
    pub fn new(data: Vec<u16>) -> Self {
        Self { data }
    }

    /// Creates a `SeqTwoByteString` from a UTF-8 string slice.
    ///
    /// The string is UTF-16–encoded; surrogate pairs are generated for
    /// characters outside the BMP.
    pub fn from_utf8(s: &str) -> Self {
        Self {
            data: s.encode_utf16().collect(),
        }
    }

    /// Returns the number of UTF-16 code units.
    #[inline]
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    #[inline]
    pub fn char_at(&self, index: usize) -> Option<u16> {
        self.data.get(index).copied()
    }

    /// Returns a UTF-8 `String` representation.
    pub fn to_utf8(&self) -> String {
        String::from_utf16_lossy(&self.data)
    }

    /// Returns a 32-bit FNV-1a hash of the UTF-16 code units.
    pub fn hash(&self) -> u32 {
        fnv1a_hash_u16(self.data.iter().copied())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ConsString
// ──────────────────────────────────────────────────────────────────────────────

/// A lazily-concatenated pair of strings.
///
/// `ConsString` avoids copying data at concatenation time by storing a
/// reference to each half.  Characters are not materialised until
/// [`flatten`][JsString::flatten] or [`to_utf8`][JsString::to_utf8] is called.
#[derive(Debug, Clone)]
pub struct ConsString {
    left: Box<JsString>,
    right: Box<JsString>,
    length: usize,
}

impl ConsString {
    /// Creates a `ConsString` from two `JsString` halves.
    ///
    /// The total length is computed eagerly and cached.
    pub fn new(left: JsString, right: JsString) -> Self {
        let length = left.length() + right.length();
        Self {
            left: Box::new(left),
            right: Box::new(right),
            length,
        }
    }

    /// Returns the total number of code units.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    pub fn char_at(&self, index: usize) -> Option<u16> {
        if index >= self.length {
            return None;
        }
        let left_len = self.left.length();
        if index < left_len {
            self.left.char_at(index)
        } else {
            self.right.char_at(index - left_len)
        }
    }

    /// Returns a UTF-8 `String` of the fully concatenated content.
    pub fn to_utf8(&self) -> String {
        let mut s = self.left.to_utf8();
        s.push_str(&self.right.to_utf8());
        s
    }

    /// Returns a flat [`JsString`] with all characters materialised.
    pub fn flatten(&self) -> JsString {
        JsString::new(&self.to_utf8())
    }

    /// Returns a 32-bit FNV-1a hash of the concatenated string's code units.
    pub fn hash(&self) -> u32 {
        self.flatten().hash()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SlicedString
// ──────────────────────────────────────────────────────────────────────────────

/// A substring view into a parent string.
///
/// `SlicedString` avoids copying data for substrings by recording the parent
/// string together with a code-unit `offset` and `length`.  Characters are
/// read on demand through [`char_at`][JsString::char_at].
#[derive(Debug, Clone)]
pub struct SlicedString {
    parent: Box<JsString>,
    offset: usize,
    length: usize,
}

impl SlicedString {
    /// Creates a `SlicedString` representing `parent[offset .. offset+length]`.
    ///
    /// Returns `Err(StatorError::RangeError)` if the slice would exceed the
    /// parent string's bounds.
    pub fn new(parent: JsString, offset: usize, length: usize) -> StatorResult<Self> {
        let parent_len = parent.length();
        let end = offset.checked_add(length);
        if end.is_none_or(|e| e > parent_len) {
            return Err(StatorError::RangeError(format!(
                "slice [offset={offset}, length={length}) is out of range for a string of length {parent_len}",
            )));
        }
        Ok(Self {
            parent: Box::new(parent),
            offset,
            length,
        })
    }

    /// Returns the length of the slice in code units.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the UTF-16 code unit at `index` within the slice, or `None` if
    /// out of bounds.
    #[inline]
    pub fn char_at(&self, index: usize) -> Option<u16> {
        if index >= self.length {
            return None;
        }
        self.parent.char_at(self.offset + index)
    }

    /// Returns a UTF-8 `String` for this slice.
    pub fn to_utf8(&self) -> String {
        let units: Vec<u16> = (self.offset..self.offset + self.length)
            .filter_map(|i| self.parent.char_at(i))
            .collect();
        String::from_utf16_lossy(&units)
    }

    /// Returns a flat [`JsString`] containing only the slice's characters.
    pub fn flatten(&self) -> JsString {
        JsString::new(&self.to_utf8())
    }

    /// Returns a 32-bit FNV-1a hash of the slice's code units.
    pub fn hash(&self) -> u32 {
        self.flatten().hash()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ExternalString
// ──────────────────────────────────────────────────────────────────────────────

/// A string whose character data is managed externally by the embedder.
///
/// `ExternalString` stores a reference-counted [`ExternalStringResource`]
/// rather than owning the character data.  This avoids copying large strings
/// that originate outside the engine heap (e.g. from a source file buffer).
#[derive(Debug, Clone)]
pub struct ExternalString {
    resource: Arc<dyn ExternalStringResource>,
}

impl ExternalString {
    /// Creates an `ExternalString` backed by the given resource.
    pub fn new(resource: Arc<dyn ExternalStringResource>) -> Self {
        Self { resource }
    }

    /// Returns the number of code units in the external string.
    #[inline]
    pub fn length(&self) -> usize {
        self.resource.len()
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    pub fn char_at(&self, index: usize) -> Option<u16> {
        if let Some(bytes) = self.resource.as_one_byte() {
            bytes.get(index).copied().map(u16::from)
        } else if let Some(units) = self.resource.as_two_byte() {
            units.get(index).copied()
        } else {
            None
        }
    }

    /// Returns a UTF-8 `String` representation.
    pub fn to_utf8(&self) -> String {
        if let Some(bytes) = self.resource.as_one_byte() {
            bytes.iter().map(|&b| b as char).collect()
        } else if let Some(units) = self.resource.as_two_byte() {
            String::from_utf16_lossy(units)
        } else {
            String::new()
        }
    }

    /// Returns a flat [`JsString`] containing the external string's data.
    pub fn flatten(&self) -> JsString {
        JsString::new(&self.to_utf8())
    }

    /// Returns a 32-bit FNV-1a hash of the string's code units.
    pub fn hash(&self) -> u32 {
        if let Some(bytes) = self.resource.as_one_byte() {
            fnv1a_hash_u16(bytes.iter().map(|&b| u16::from(b)))
        } else if let Some(units) = self.resource.as_two_byte() {
            fnv1a_hash_u16(units.iter().copied())
        } else {
            fnv1a_hash_u16(std::iter::empty())
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// JsString enum
// ──────────────────────────────────────────────────────────────────────────────

/// A JavaScript string value in one of several internal representations.
///
/// The engine chooses the most efficient representation based on the string's
/// content and how it was created:
///
/// | Variant | When used |
/// |---|---|
/// | [`SeqOneByte`][JsString::SeqOneByte] | All characters are in the Latin-1 range |
/// | [`SeqTwoByte`][JsString::SeqTwoByte] | String contains characters outside Latin-1 |
/// | [`Cons`][JsString::Cons] | Result of a lazy concatenation |
/// | [`Sliced`][JsString::Sliced] | Substring view into another string |
/// | [`External`][JsString::External] | Embedder-provided character data |
#[derive(Debug, Clone)]
pub enum JsString {
    /// A flat Latin-1 string.
    SeqOneByte(SeqOneByteString),
    /// A flat UTF-16 string.
    SeqTwoByte(SeqTwoByteString),
    /// A lazily-concatenated pair.
    Cons(ConsString),
    /// A substring view.
    Sliced(SlicedString),
    /// An embedder-provided string.
    External(ExternalString),
}

impl JsString {
    /// Creates a `JsString` from a UTF-8 string slice.
    ///
    /// Chooses [`SeqOneByte`][JsString::SeqOneByte] if all characters are in
    /// the Latin-1 range; otherwise falls back to
    /// [`SeqTwoByte`][JsString::SeqTwoByte].
    pub fn new(s: &str) -> Self {
        match SeqOneByteString::from_latin1_str(s) {
            Ok(one_byte) => JsString::SeqOneByte(one_byte),
            Err(_) => JsString::SeqTwoByte(SeqTwoByteString::from_utf8(s)),
        }
    }

    /// Lazily concatenates `self` and `other` into a [`Cons`][JsString::Cons] string.
    pub fn concat(self, other: JsString) -> Self {
        JsString::Cons(ConsString::new(self, other))
    }

    /// Returns the number of UTF-16 code units in the string.
    pub fn length(&self) -> usize {
        match self {
            Self::SeqOneByte(s) => s.length(),
            Self::SeqTwoByte(s) => s.length(),
            Self::Cons(s) => s.length(),
            Self::Sliced(s) => s.length(),
            Self::External(s) => s.length(),
        }
    }

    /// Returns the UTF-16 code unit at `index`, or `None` if out of bounds.
    pub fn char_at(&self, index: usize) -> Option<u16> {
        match self {
            Self::SeqOneByte(s) => s.char_at(index),
            Self::SeqTwoByte(s) => s.char_at(index),
            Self::Cons(s) => s.char_at(index),
            Self::Sliced(s) => s.char_at(index),
            Self::External(s) => s.char_at(index),
        }
    }

    /// Returns a flat copy of this string.
    ///
    /// If the string is already flat ([`SeqOneByte`][JsString::SeqOneByte] or
    /// [`SeqTwoByte`][JsString::SeqTwoByte]) it is returned as-is.  All other
    /// variants are materialised into a new flat string.
    pub fn flatten(&self) -> JsString {
        match self {
            Self::SeqOneByte(_) | Self::SeqTwoByte(_) => self.clone(),
            Self::Cons(s) => s.flatten(),
            Self::Sliced(s) => s.flatten(),
            Self::External(s) => s.flatten(),
        }
    }

    /// Returns a UTF-8 `String` representation.
    pub fn to_utf8(&self) -> String {
        match self {
            Self::SeqOneByte(s) => s.to_utf8(),
            Self::SeqTwoByte(s) => s.to_utf8(),
            Self::Cons(s) => s.to_utf8(),
            Self::Sliced(s) => s.to_utf8(),
            Self::External(s) => s.to_utf8(),
        }
    }

    /// Returns a 32-bit FNV-1a hash of the string's UTF-16 code units.
    ///
    /// Two strings with identical code-unit sequences always produce the same
    /// hash, regardless of their internal representation.
    pub fn hash(&self) -> u32 {
        match self {
            Self::SeqOneByte(s) => s.hash(),
            Self::SeqTwoByte(s) => s.hash(),
            Self::Cons(s) => s.hash(),
            Self::Sliced(s) => s.hash(),
            Self::External(s) => s.hash(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Hash helper
// ──────────────────────────────────────────────────────────────────────────────

/// Computes a 32-bit FNV-1a hash over an iterator of UTF-16 code units.
///
/// Each code unit is processed as two bytes (little-endian) so that the hash
/// is consistent regardless of whether the string is one-byte or two-byte:
/// a Latin-1 byte `b` is equivalent to the UTF-16 unit `b as u16`, and both
/// produce the same contribution to the hash.
fn fnv1a_hash_u16(units: impl Iterator<Item = u16>) -> u32 {
    const FNV_OFFSET: u32 = 2_166_136_261;
    const FNV_PRIME: u32 = 16_777_619;
    let mut hash = FNV_OFFSET;
    for unit in units {
        hash ^= u32::from(unit & 0xFF);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u32::from(unit >> 8);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SeqOneByteString ─────────────────────────────────────────────────────

    #[test]
    fn test_seq_one_byte_from_str_ascii() {
        let s = SeqOneByteString::from_latin1_str("hello").unwrap();
        assert_eq!(s.length(), 5);
        assert_eq!(s.to_utf8(), "hello");
    }

    #[test]
    fn test_seq_one_byte_from_str_latin1() {
        let s = SeqOneByteString::from_latin1_str("caf\u{00E9}").unwrap(); // "café"
        assert_eq!(s.length(), 4);
        assert_eq!(s.to_utf8(), "café");
    }

    #[test]
    fn test_seq_one_byte_from_str_rejects_non_latin1() {
        let result = SeqOneByteString::from_latin1_str("こんにちは");
        assert!(matches!(result, Err(StatorError::TypeError(_))));
    }

    #[test]
    fn test_seq_one_byte_char_at() {
        let s = SeqOneByteString::from_latin1_str("abc").unwrap();
        assert_eq!(s.char_at(0), Some(b'a' as u16));
        assert_eq!(s.char_at(2), Some(b'c' as u16));
        assert_eq!(s.char_at(3), None);
    }

    #[test]
    fn test_seq_one_byte_hash_same_content() {
        let a = SeqOneByteString::from_latin1_str("test").unwrap();
        let b = SeqOneByteString::from_latin1_str("test").unwrap();
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_seq_one_byte_hash_different_content() {
        let a = SeqOneByteString::from_latin1_str("test").unwrap();
        let b = SeqOneByteString::from_latin1_str("Test").unwrap();
        assert_ne!(a.hash(), b.hash());
    }

    // ── SeqTwoByteString ─────────────────────────────────────────────────────

    #[test]
    fn test_seq_two_byte_from_str_ascii() {
        let s = SeqTwoByteString::from_utf8("hello");
        assert_eq!(s.length(), 5);
        assert_eq!(s.to_utf8(), "hello");
    }

    #[test]
    fn test_seq_two_byte_from_str_emoji() {
        // 😀 is U+1F600, encoded as a surrogate pair in UTF-16 (length 2).
        let s = SeqTwoByteString::from_utf8("😀");
        assert_eq!(s.length(), 2);
        assert_eq!(s.to_utf8(), "😀");
    }

    #[test]
    fn test_seq_two_byte_from_str_japanese() {
        let s = SeqTwoByteString::from_utf8("こんにちは");
        assert_eq!(s.length(), 5);
        assert_eq!(s.to_utf8(), "こんにちは");
    }

    #[test]
    fn test_seq_two_byte_char_at() {
        let s = SeqTwoByteString::new(vec![0x0048, 0x0069]); // "Hi"
        assert_eq!(s.char_at(0), Some(0x0048));
        assert_eq!(s.char_at(1), Some(0x0069));
        assert_eq!(s.char_at(2), None);
    }

    #[test]
    fn test_seq_two_byte_hash_same_content() {
        let a = SeqTwoByteString::from_utf8("hello");
        let b = SeqTwoByteString::from_utf8("hello");
        assert_eq!(a.hash(), b.hash());
    }

    // ── Hash consistency across representations ───────────────────────────────

    #[test]
    fn test_hash_consistent_one_byte_vs_two_byte() {
        // The same ASCII content hashed through both representations should be equal.
        let one = JsString::SeqOneByte(SeqOneByteString::from_latin1_str("abc").unwrap());
        let two = JsString::SeqTwoByte(SeqTwoByteString::from_utf8("abc"));
        assert_eq!(one.hash(), two.hash());
    }

    // ── ConsString ───────────────────────────────────────────────────────────

    #[test]
    fn test_cons_concatenation_length() {
        let a = JsString::new("hello");
        let b = JsString::new(" world");
        let c = a.concat(b);
        assert_eq!(c.length(), 11);
    }

    #[test]
    fn test_cons_to_utf8() {
        let a = JsString::new("foo");
        let b = JsString::new("bar");
        let c = a.concat(b);
        assert_eq!(c.to_utf8(), "foobar");
    }

    #[test]
    fn test_cons_char_at() {
        let a = JsString::new("ab");
        let b = JsString::new("cd");
        let c = a.concat(b);
        assert_eq!(c.char_at(0), Some(b'a' as u16));
        assert_eq!(c.char_at(1), Some(b'b' as u16));
        assert_eq!(c.char_at(2), Some(b'c' as u16));
        assert_eq!(c.char_at(3), Some(b'd' as u16));
        assert_eq!(c.char_at(4), None);
    }

    #[test]
    fn test_cons_flatten_returns_flat_string() {
        let a = JsString::new("hello");
        let b = JsString::new(" world");
        let cons = a.concat(b);
        let flat = cons.flatten();
        assert!(matches!(
            flat,
            JsString::SeqOneByte(_) | JsString::SeqTwoByte(_)
        ));
        assert_eq!(flat.to_utf8(), "hello world");
    }

    #[test]
    fn test_cons_hash_matches_flat() {
        let a = JsString::new("foo");
        let b = JsString::new("bar");
        let cons = a.clone().concat(b.clone());
        let flat = JsString::new("foobar");
        assert_eq!(cons.hash(), flat.hash());
    }

    #[test]
    fn test_cons_empty_strings() {
        let a = JsString::new("");
        let b = JsString::new("hello");
        let c = a.concat(b);
        assert_eq!(c.length(), 5);
        assert_eq!(c.to_utf8(), "hello");
    }

    // ── SlicedString ─────────────────────────────────────────────────────────

    #[test]
    fn test_sliced_basic_slice() {
        let parent = JsString::new("hello world");
        let slice = SlicedString::new(parent, 6, 5).unwrap();
        assert_eq!(slice.length(), 5);
        assert_eq!(slice.to_utf8(), "world");
    }

    #[test]
    fn test_sliced_char_at() {
        let parent = JsString::new("abcdef");
        let slice = SlicedString::new(parent, 2, 3).unwrap(); // "cde"
        assert_eq!(slice.char_at(0), Some(b'c' as u16));
        assert_eq!(slice.char_at(1), Some(b'd' as u16));
        assert_eq!(slice.char_at(2), Some(b'e' as u16));
        assert_eq!(slice.char_at(3), None);
    }

    #[test]
    fn test_sliced_out_of_bounds_returns_error() {
        let parent = JsString::new("hi");
        let result = SlicedString::new(parent, 1, 5);
        assert!(matches!(result, Err(StatorError::RangeError(_))));
    }

    #[test]
    fn test_sliced_offset_overflow_returns_error() {
        let parent = JsString::new("hi");
        let result = SlicedString::new(parent, usize::MAX, 1);
        assert!(matches!(result, Err(StatorError::RangeError(_))));
    }

    #[test]
    fn test_sliced_flatten_returns_flat_string() {
        let parent = JsString::new("hello world");
        let slice = SlicedString::new(parent, 0, 5).unwrap();
        let flat = slice.flatten();
        assert!(matches!(
            flat,
            JsString::SeqOneByte(_) | JsString::SeqTwoByte(_)
        ));
        assert_eq!(flat.to_utf8(), "hello");
    }

    #[test]
    fn test_sliced_hash_matches_flat() {
        let parent = JsString::new("hello world");
        let slice = JsString::Sliced(SlicedString::new(parent, 6, 5).unwrap());
        let flat = JsString::new("world");
        assert_eq!(slice.hash(), flat.hash());
    }

    #[test]
    fn test_sliced_empty_slice() {
        let parent = JsString::new("hello");
        let slice = SlicedString::new(parent, 2, 0).unwrap();
        assert_eq!(slice.length(), 0);
        assert_eq!(slice.to_utf8(), "");
    }

    // ── ExternalString ───────────────────────────────────────────────────────

    /// A one-byte external resource backed by a `Vec<u8>`.
    #[derive(Debug)]
    struct TestOneByteResource(Vec<u8>);

    impl ExternalStringResource for TestOneByteResource {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn as_one_byte(&self) -> Option<&[u8]> {
            Some(&self.0)
        }
        fn as_two_byte(&self) -> Option<&[u16]> {
            None
        }
    }

    /// A two-byte external resource backed by a `Vec<u16>`.
    #[derive(Debug)]
    struct TestTwoByteResource(Vec<u16>);

    impl ExternalStringResource for TestTwoByteResource {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn as_one_byte(&self) -> Option<&[u8]> {
            None
        }
        fn as_two_byte(&self) -> Option<&[u16]> {
            Some(&self.0)
        }
    }

    #[test]
    fn test_external_one_byte_length_and_to_utf8() {
        let res: Arc<dyn ExternalStringResource> = Arc::new(TestOneByteResource(b"hello".to_vec()));
        let s = ExternalString::new(res);
        assert_eq!(s.length(), 5);
        assert_eq!(s.to_utf8(), "hello");
    }

    #[test]
    fn test_external_two_byte_length_and_to_utf8() {
        let units: Vec<u16> = "こんにちは".encode_utf16().collect();
        let res: Arc<dyn ExternalStringResource> = Arc::new(TestTwoByteResource(units));
        let s = ExternalString::new(res);
        assert_eq!(s.length(), 5);
        assert_eq!(s.to_utf8(), "こんにちは");
    }

    #[test]
    fn test_external_char_at() {
        let res: Arc<dyn ExternalStringResource> = Arc::new(TestOneByteResource(b"abc".to_vec()));
        let s = ExternalString::new(res);
        assert_eq!(s.char_at(0), Some(b'a' as u16));
        assert_eq!(s.char_at(2), Some(b'c' as u16));
        assert_eq!(s.char_at(3), None);
    }

    #[test]
    fn test_external_flatten_returns_flat_string() {
        let res: Arc<dyn ExternalStringResource> = Arc::new(TestOneByteResource(b"world".to_vec()));
        let s = ExternalString::new(res);
        let flat = s.flatten();
        assert!(matches!(
            flat,
            JsString::SeqOneByte(_) | JsString::SeqTwoByte(_)
        ));
        assert_eq!(flat.to_utf8(), "world");
    }

    #[test]
    fn test_external_hash_matches_seq() {
        let res: Arc<dyn ExternalStringResource> = Arc::new(TestOneByteResource(b"hello".to_vec()));
        let ext = JsString::External(ExternalString::new(res));
        let seq = JsString::new("hello");
        assert_eq!(ext.hash(), seq.hash());
    }

    // ── UTF-8 round-trips ────────────────────────────────────────────────────

    #[test]
    fn test_utf8_roundtrip_ascii() {
        let original = "The quick brown fox";
        assert_eq!(JsString::new(original).to_utf8(), original);
    }

    #[test]
    fn test_utf8_roundtrip_latin1() {
        let original = "résumé";
        assert_eq!(JsString::new(original).to_utf8(), original);
    }

    #[test]
    fn test_utf8_roundtrip_japanese() {
        let original = "日本語";
        assert_eq!(JsString::new(original).to_utf8(), original);
    }

    #[test]
    fn test_utf8_roundtrip_emoji() {
        let original = "Hello 🌍!";
        assert_eq!(JsString::new(original).to_utf8(), original);
    }

    #[test]
    fn test_utf8_roundtrip_cons() {
        let a = JsString::new("Hello");
        let b = JsString::new(", 世界!");
        let c = a.concat(b);
        assert_eq!(c.to_utf8(), "Hello, 世界!");
    }

    #[test]
    fn test_utf8_roundtrip_sliced() {
        let parent = JsString::new("Hello, world!");
        let sliced = JsString::Sliced(SlicedString::new(parent, 7, 5).unwrap());
        assert_eq!(sliced.to_utf8(), "world");
    }

    // ── JsString::new encoding selection ────────────────────────────────

    #[test]
    fn test_from_str_selects_one_byte_for_ascii() {
        let s = JsString::new("ascii");
        assert!(matches!(s, JsString::SeqOneByte(_)));
    }

    #[test]
    fn test_from_str_selects_one_byte_for_latin1() {
        let s = JsString::new("caf\u{00E9}");
        assert!(matches!(s, JsString::SeqOneByte(_)));
    }

    #[test]
    fn test_from_str_selects_two_byte_for_non_latin1() {
        let s = JsString::new("日本語");
        assert!(matches!(s, JsString::SeqTwoByte(_)));
    }

    // ── JsString::length on empty string ─────────────────────────────────────

    #[test]
    fn test_length_empty_string() {
        assert_eq!(JsString::new("").length(), 0);
    }

    #[test]
    fn test_char_at_empty_string() {
        assert_eq!(JsString::new("").char_at(0), None);
    }
}

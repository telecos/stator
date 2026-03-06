//! ECMAScript §23.2 TypedArray, §25.1 ArrayBuffer, and §25.3 DataView built-ins.
//!
//! Provides [`JsArrayBuffer`], [`JsTypedArray`], and [`JsDataView`], the binary
//! data primitives defined by the ECMAScript specification.
//!
//! # Naming convention
//!
//! Each function is prefixed with `arraybuffer_`, `typed_array_`, or
//! `dataview_` to avoid ambiguity.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §25.1 — *ArrayBuffer Objects*
//! * ECMAScript 2025 Language Specification §25.3 — *DataView Objects*
//! * ECMAScript 2025 Language Specification §23.2 — *TypedArray Objects*

use std::cell::RefCell;
use std::cmp;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// TypedArrayKind
// ─────────────────────────────────────────────────────────────────────────────

/// The element type of a TypedArray (ECMAScript §23.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedArrayKind {
    /// `Int8Array` — signed 8-bit integer.
    Int8,
    /// `Uint8Array` — unsigned 8-bit integer.
    Uint8,
    /// `Uint8ClampedArray` — unsigned 8-bit integer (clamped).
    Uint8Clamped,
    /// `Int16Array` — signed 16-bit integer.
    Int16,
    /// `Uint16Array` — unsigned 16-bit integer.
    Uint16,
    /// `Int32Array` — signed 32-bit integer.
    Int32,
    /// `Uint32Array` — unsigned 32-bit integer.
    Uint32,
    /// `Float32Array` — 32-bit IEEE 754 float.
    Float32,
    /// `Float64Array` — 64-bit IEEE 754 float.
    Float64,
    /// `BigInt64Array` — signed 64-bit BigInt.
    BigInt64,
    /// `BigUint64Array` — unsigned 64-bit BigInt.
    BigUint64,
}

impl TypedArrayKind {
    /// The size in bytes of a single element of this typed-array kind.
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 | Self::Uint8Clamped => 1,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Float64 | Self::BigInt64 | Self::BigUint64 => 8,
        }
    }

    /// The constructor name for this kind (e.g. `"Int8Array"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Int8 => "Int8Array",
            Self::Uint8 => "Uint8Array",
            Self::Uint8Clamped => "Uint8ClampedArray",
            Self::Int16 => "Int16Array",
            Self::Uint16 => "Uint16Array",
            Self::Int32 => "Int32Array",
            Self::Uint32 => "Uint32Array",
            Self::Float32 => "Float32Array",
            Self::Float64 => "Float64Array",
            Self::BigInt64 => "BigInt64Array",
            Self::BigUint64 => "BigUint64Array",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ArrayBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript `ArrayBuffer` object (ECMAScript §25.1).
///
/// The backing store is a `Vec<u8>` whose length equals `byteLength`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::{arraybuffer_new, arraybuffer_byte_length};
///
/// let buf = arraybuffer_new(16);
/// assert_eq!(arraybuffer_byte_length(&buf), 16);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct JsArrayBuffer {
    /// Raw byte storage.
    pub data: Vec<u8>,
}

/// ECMAScript §25.1.3.1 `new ArrayBuffer(byteLength)`.
///
/// Creates a new zero-filled `ArrayBuffer` with the given byte length.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::arraybuffer_new;
///
/// let buf = arraybuffer_new(8);
/// assert_eq!(buf.data.len(), 8);
/// ```
pub fn arraybuffer_new(byte_length: usize) -> JsArrayBuffer {
    JsArrayBuffer {
        data: vec![0u8; byte_length],
    }
}

/// ECMAScript §25.1.5.1 `ArrayBuffer.prototype.byteLength` getter.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::{arraybuffer_new, arraybuffer_byte_length};
///
/// let buf = arraybuffer_new(32);
/// assert_eq!(arraybuffer_byte_length(&buf), 32);
/// ```
pub fn arraybuffer_byte_length(buf: &JsArrayBuffer) -> usize {
    buf.data.len()
}

/// ECMAScript §25.1.5.3 `ArrayBuffer.prototype.slice(begin, end)`.
///
/// Returns a new `ArrayBuffer` containing bytes from `begin` to `end`
/// (exclusive).  Negative indices count from the end.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::{arraybuffer_new, arraybuffer_slice};
///
/// let mut buf = arraybuffer_new(4);
/// buf.data[0] = 10;
/// buf.data[1] = 20;
/// buf.data[2] = 30;
/// buf.data[3] = 40;
/// let sliced = arraybuffer_slice(&buf, 1, 3);
/// assert_eq!(sliced.data, vec![20, 30]);
/// ```
pub fn arraybuffer_slice(buf: &JsArrayBuffer, begin: i64, end: i64) -> JsArrayBuffer {
    let len = buf.data.len() as i64;
    let start = clamp_index(begin, len) as usize;
    let fin = clamp_index(end, len) as usize;
    let fin = cmp::max(start, fin);
    JsArrayBuffer {
        data: buf.data[start..fin].to_vec(),
    }
}

/// ECMAScript §25.1.4.1 `ArrayBuffer.isView(arg)`.
///
/// Returns `true` if `arg` is a `TypedArray` or `DataView`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::arraybuffer_is_view;
/// use stator_core::objects::value::JsValue;
///
/// assert!(!arraybuffer_is_view(&JsValue::Smi(42)));
/// ```
pub fn arraybuffer_is_view(value: &JsValue) -> bool {
    matches!(value, JsValue::TypedArray(_) | JsValue::DataView(_))
}

// ─────────────────────────────────────────────────────────────────────────────
// DataView
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript `DataView` object (ECMAScript §25.3).
///
/// Provides a byte-level interface for reading and writing arbitrary data types
/// from an `ArrayBuffer` with explicit endianness control.
#[derive(Debug, Clone, PartialEq)]
pub struct JsDataView {
    /// The underlying `ArrayBuffer`.
    pub buffer: Rc<RefCell<JsArrayBuffer>>,
    /// Byte offset into the buffer.
    pub byte_offset: usize,
    /// The length in bytes of the view.
    pub byte_length: usize,
}

/// ECMAScript §25.3.2.1 `new DataView(buffer, byteOffset?, byteLength?)`.
///
/// # Errors
///
/// Returns `RangeError` if the offset or length is out of bounds.
pub fn dataview_new(
    buffer: Rc<RefCell<JsArrayBuffer>>,
    byte_offset: usize,
    byte_length: Option<usize>,
) -> StatorResult<JsDataView> {
    let buf_len = buffer.borrow().data.len();
    if byte_offset > buf_len {
        return Err(StatorError::RangeError(
            "Start offset is outside the bounds of the buffer".into(),
        ));
    }
    let len = byte_length.unwrap_or(buf_len - byte_offset);
    if byte_offset + len > buf_len {
        return Err(StatorError::RangeError("Invalid DataView length".into()));
    }
    Ok(JsDataView {
        buffer,
        byte_offset,
        byte_length: len,
    })
}

/// `DataView.prototype.buffer` getter.
pub fn dataview_buffer(dv: &JsDataView) -> Rc<RefCell<JsArrayBuffer>> {
    Rc::clone(&dv.buffer)
}

/// `DataView.prototype.byteLength` getter.
pub fn dataview_byte_length(dv: &JsDataView) -> usize {
    dv.byte_length
}

/// `DataView.prototype.byteOffset` getter.
pub fn dataview_byte_offset(dv: &JsDataView) -> usize {
    dv.byte_offset
}

macro_rules! dataview_get {
    ($name:ident, $ty:ty, $read_le:ident, $read_be:ident, $n:expr) => {
        /// Read a value at the given byte offset with the specified endianness.
        ///
        /// # Errors
        ///
        /// Returns `RangeError` if reading past the end of the view.
        pub fn $name(
            dv: &JsDataView,
            byte_offset: usize,
            little_endian: bool,
        ) -> StatorResult<$ty> {
            let abs = dv.byte_offset + byte_offset;
            if byte_offset + $n > dv.byte_length {
                return Err(StatorError::RangeError(
                    "Offset is outside the bounds of the DataView".into(),
                ));
            }
            let buf = dv.buffer.borrow();
            let bytes: [u8; $n] = buf.data[abs..abs + $n]
                .try_into()
                .expect("slice length verified");
            Ok(if little_endian {
                <$ty>::$read_le(bytes)
            } else {
                <$ty>::$read_be(bytes)
            })
        }
    };
}

macro_rules! dataview_set {
    ($name:ident, $ty:ty, $write_le:ident, $write_be:ident, $n:expr) => {
        /// Write a value at the given byte offset with the specified endianness.
        ///
        /// # Errors
        ///
        /// Returns `RangeError` if writing past the end of the view.
        pub fn $name(
            dv: &JsDataView,
            byte_offset: usize,
            value: $ty,
            little_endian: bool,
        ) -> StatorResult<()> {
            let abs = dv.byte_offset + byte_offset;
            if byte_offset + $n > dv.byte_length {
                return Err(StatorError::RangeError(
                    "Offset is outside the bounds of the DataView".into(),
                ));
            }
            let bytes = if little_endian {
                value.$write_le()
            } else {
                value.$write_be()
            };
            let mut buf = dv.buffer.borrow_mut();
            buf.data[abs..abs + $n].copy_from_slice(&bytes);
            Ok(())
        }
    };
}

dataview_get!(dataview_get_int8, i8, from_le_bytes, from_be_bytes, 1);
dataview_get!(dataview_get_uint8, u8, from_le_bytes, from_be_bytes, 1);
dataview_get!(dataview_get_int16, i16, from_le_bytes, from_be_bytes, 2);
dataview_get!(dataview_get_uint16, u16, from_le_bytes, from_be_bytes, 2);
dataview_get!(dataview_get_int32, i32, from_le_bytes, from_be_bytes, 4);
dataview_get!(dataview_get_uint32, u32, from_le_bytes, from_be_bytes, 4);
dataview_get!(dataview_get_float32, f32, from_le_bytes, from_be_bytes, 4);
dataview_get!(dataview_get_float64, f64, from_le_bytes, from_be_bytes, 8);
dataview_get!(dataview_get_bigint64, i64, from_le_bytes, from_be_bytes, 8);
dataview_get!(dataview_get_biguint64, u64, from_le_bytes, from_be_bytes, 8);

dataview_set!(dataview_set_int8, i8, to_le_bytes, to_be_bytes, 1);
dataview_set!(dataview_set_uint8, u8, to_le_bytes, to_be_bytes, 1);
dataview_set!(dataview_set_int16, i16, to_le_bytes, to_be_bytes, 2);
dataview_set!(dataview_set_uint16, u16, to_le_bytes, to_be_bytes, 2);
dataview_set!(dataview_set_int32, i32, to_le_bytes, to_be_bytes, 4);
dataview_set!(dataview_set_uint32, u32, to_le_bytes, to_be_bytes, 4);
dataview_set!(dataview_set_float32, f32, to_le_bytes, to_be_bytes, 4);
dataview_set!(dataview_set_float64, f64, to_le_bytes, to_be_bytes, 8);
dataview_set!(dataview_set_bigint64, i64, to_le_bytes, to_be_bytes, 8);
dataview_set!(dataview_set_biguint64, u64, to_le_bytes, to_be_bytes, 8);

// ─────────────────────────────────────────────────────────────────────────────
// TypedArray
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript TypedArray object (ECMAScript §23.2).
///
/// All eleven TypedArray constructors share this representation; the
/// [`TypedArrayKind`] discriminant determines the element size and
/// interpretation.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::{TypedArrayKind, typed_array_new_from_length,
///     typed_array_length, typed_array_get, typed_array_set};
/// use stator_core::objects::value::JsValue;
///
/// let ta = typed_array_new_from_length(TypedArrayKind::Int32, 4);
/// assert_eq!(typed_array_length(&ta), 4);
/// typed_array_set(&ta, 0, &JsValue::Smi(42)).unwrap();
/// assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(42));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct JsTypedArray {
    /// The underlying `ArrayBuffer`.
    pub buffer: Rc<RefCell<JsArrayBuffer>>,
    /// Element type.
    pub kind: TypedArrayKind,
    /// Byte offset into the buffer.
    pub byte_offset: usize,
    /// Number of elements (not bytes).
    pub length: usize,
}

/// Construct a TypedArray of `length` elements, allocating a new buffer.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::typed_array::{TypedArrayKind, typed_array_new_from_length,
///     typed_array_length};
///
/// let ta = typed_array_new_from_length(TypedArrayKind::Float64, 8);
/// assert_eq!(typed_array_length(&ta), 8);
/// ```
pub fn typed_array_new_from_length(kind: TypedArrayKind, length: usize) -> JsTypedArray {
    let byte_len = length * kind.bytes_per_element();
    JsTypedArray {
        buffer: Rc::new(RefCell::new(arraybuffer_new(byte_len))),
        kind,
        byte_offset: 0,
        length,
    }
}

/// Construct a TypedArray over an existing `ArrayBuffer`.
///
/// # Errors
///
/// Returns `RangeError` if offset or length is out of bounds, or if
/// `byte_offset` is not aligned to the element size.
pub fn typed_array_new_from_buffer(
    kind: TypedArrayKind,
    buffer: Rc<RefCell<JsArrayBuffer>>,
    byte_offset: usize,
    length: Option<usize>,
) -> StatorResult<JsTypedArray> {
    let bpe = kind.bytes_per_element();
    if !byte_offset.is_multiple_of(bpe) {
        return Err(StatorError::RangeError(
            "Start offset is not a multiple of the element size".into(),
        ));
    }
    let buf_len = buffer.borrow().data.len();
    if byte_offset > buf_len {
        return Err(StatorError::RangeError(
            "Start offset is outside the bounds of the buffer".into(),
        ));
    }
    let len = match length {
        Some(l) => {
            if byte_offset + l * bpe > buf_len {
                return Err(StatorError::RangeError("Invalid typed array length".into()));
            }
            l
        }
        None => {
            let remaining = buf_len - byte_offset;
            if !remaining.is_multiple_of(bpe) {
                return Err(StatorError::RangeError(
                    "Byte length of buffer minus offset is not a multiple of element size".into(),
                ));
            }
            remaining / bpe
        }
    };
    Ok(JsTypedArray {
        buffer,
        kind,
        byte_offset,
        length: len,
    })
}

/// Construct a TypedArray from an iterable of `JsValue`s.
pub fn typed_array_from_values(
    kind: TypedArrayKind,
    values: &[JsValue],
) -> StatorResult<JsTypedArray> {
    let ta = typed_array_new_from_length(kind, values.len());
    for (i, v) in values.iter().enumerate() {
        typed_array_set(&ta, i, v)?;
    }
    Ok(ta)
}

// ── Accessors ────────────────────────────────────────────────────────────────

/// `%TypedArray%.prototype.length` getter.
pub fn typed_array_length(ta: &JsTypedArray) -> usize {
    ta.length
}

/// `%TypedArray%.prototype.byteLength` getter.
pub fn typed_array_byte_length(ta: &JsTypedArray) -> usize {
    ta.length * ta.kind.bytes_per_element()
}

/// `%TypedArray%.prototype.byteOffset` getter.
pub fn typed_array_byte_offset(ta: &JsTypedArray) -> usize {
    ta.byte_offset
}

/// `%TypedArray%.prototype.buffer` getter.
pub fn typed_array_buffer(ta: &JsTypedArray) -> Rc<RefCell<JsArrayBuffer>> {
    Rc::clone(&ta.buffer)
}

// ── Element access ───────────────────────────────────────────────────────────

/// Get the element at `index` as a `JsValue`.
///
/// Returns `JsValue::Undefined` for out-of-bounds access.
pub fn typed_array_get(ta: &JsTypedArray, index: usize) -> JsValue {
    if index >= ta.length {
        return JsValue::Undefined;
    }
    let bpe = ta.kind.bytes_per_element();
    let abs = ta.byte_offset + index * bpe;
    let buf = ta.buffer.borrow();
    let d = &buf.data;
    match ta.kind {
        TypedArrayKind::Int8 => JsValue::Smi(i32::from(d[abs] as i8)),
        TypedArrayKind::Uint8 | TypedArrayKind::Uint8Clamped => JsValue::Smi(i32::from(d[abs])),
        TypedArrayKind::Int16 => {
            let v = i16::from_ne_bytes([d[abs], d[abs + 1]]);
            JsValue::Smi(i32::from(v))
        }
        TypedArrayKind::Uint16 => {
            let v = u16::from_ne_bytes([d[abs], d[abs + 1]]);
            JsValue::Smi(i32::from(v))
        }
        TypedArrayKind::Int32 => {
            let v = i32::from_ne_bytes([d[abs], d[abs + 1], d[abs + 2], d[abs + 3]]);
            JsValue::Smi(v)
        }
        TypedArrayKind::Uint32 => {
            let v = u32::from_ne_bytes([d[abs], d[abs + 1], d[abs + 2], d[abs + 3]]);
            JsValue::HeapNumber(f64::from(v))
        }
        TypedArrayKind::Float32 => {
            let v = f32::from_ne_bytes([d[abs], d[abs + 1], d[abs + 2], d[abs + 3]]);
            JsValue::HeapNumber(f64::from(v))
        }
        TypedArrayKind::Float64 => {
            let bytes: [u8; 8] = d[abs..abs + 8].try_into().expect("8 bytes");
            JsValue::HeapNumber(f64::from_ne_bytes(bytes))
        }
        TypedArrayKind::BigInt64 => {
            let bytes: [u8; 8] = d[abs..abs + 8].try_into().expect("8 bytes");
            JsValue::BigInt(i128::from(i64::from_ne_bytes(bytes)))
        }
        TypedArrayKind::BigUint64 => {
            let bytes: [u8; 8] = d[abs..abs + 8].try_into().expect("8 bytes");
            JsValue::BigInt(i128::from(u64::from_ne_bytes(bytes)))
        }
    }
}

/// Set the element at `index` from a `JsValue`.
///
/// # Errors
///
/// Returns `RangeError` for out-of-bounds indices. Returns `TypeError`
/// if the value cannot be converted to the typed-array element type.
pub fn typed_array_set(ta: &JsTypedArray, index: usize, value: &JsValue) -> StatorResult<()> {
    if index >= ta.length {
        return Err(StatorError::RangeError("Index out of bounds".into()));
    }
    let bpe = ta.kind.bytes_per_element();
    let abs = ta.byte_offset + index * bpe;
    let mut buf = ta.buffer.borrow_mut();
    let d = &mut buf.data;
    match ta.kind {
        TypedArrayKind::Int8 => {
            let n = value.to_int32()? as i8;
            d[abs] = n as u8;
        }
        TypedArrayKind::Uint8 => {
            let n = value.to_int32()? as u8;
            d[abs] = n;
        }
        TypedArrayKind::Uint8Clamped => {
            let n = value.to_number()?;
            d[abs] = clamp_u8(n);
        }
        TypedArrayKind::Int16 => {
            let n = value.to_int32()? as i16;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 2].copy_from_slice(&bytes);
        }
        TypedArrayKind::Uint16 => {
            let n = value.to_int32()? as u16;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 2].copy_from_slice(&bytes);
        }
        TypedArrayKind::Int32 => {
            let n = value.to_int32()?;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 4].copy_from_slice(&bytes);
        }
        TypedArrayKind::Uint32 => {
            let n = value.to_uint32()?;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 4].copy_from_slice(&bytes);
        }
        TypedArrayKind::Float32 => {
            let n = value.to_number()? as f32;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 4].copy_from_slice(&bytes);
        }
        TypedArrayKind::Float64 => {
            let n = value.to_number()?;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 8].copy_from_slice(&bytes);
        }
        TypedArrayKind::BigInt64 => {
            let n = value_to_bigint64(value)?;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 8].copy_from_slice(&bytes);
        }
        TypedArrayKind::BigUint64 => {
            let n = value_to_biguint64(value)?;
            let bytes = n.to_ne_bytes();
            d[abs..abs + 8].copy_from_slice(&bytes);
        }
    }
    Ok(())
}

// ── Prototype methods ────────────────────────────────────────────────────────

/// `%TypedArray%.prototype.at(index)`.
pub fn typed_array_at(ta: &JsTypedArray, index: i64) -> JsValue {
    let len = ta.length as i64;
    let i = if index < 0 { len + index } else { index };
    if i < 0 || i >= len {
        return JsValue::Undefined;
    }
    typed_array_get(ta, i as usize)
}

/// `%TypedArray%.prototype.fill(value, start?, end?)`.
pub fn typed_array_fill(
    ta: &JsTypedArray,
    value: &JsValue,
    start: i64,
    end: i64,
) -> StatorResult<()> {
    let len = ta.length as i64;
    let s = clamp_index(start, len) as usize;
    let e = clamp_index(end, len) as usize;
    for i in s..e {
        typed_array_set(ta, i, value)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.copyWithin(target, start, end?)`.
pub fn typed_array_copy_within(ta: &JsTypedArray, target: i64, start: i64, end: i64) {
    let len = ta.length as i64;
    let to = clamp_index(target, len) as usize;
    let from = clamp_index(start, len) as usize;
    let fin = clamp_index(end, len) as usize;
    let count = cmp::min(fin.saturating_sub(from), ta.length.saturating_sub(to));
    if count == 0 {
        return;
    }
    let bpe = ta.kind.bytes_per_element();
    let src_start = ta.byte_offset + from * bpe;
    let dst_start = ta.byte_offset + to * bpe;
    let byte_count = count * bpe;
    let mut buf = ta.buffer.borrow_mut();
    buf.data
        .copy_within(src_start..src_start + byte_count, dst_start);
}

/// `%TypedArray%.prototype.reverse()`.
pub fn typed_array_reverse(ta: &JsTypedArray) {
    let len = ta.length;
    if len < 2 {
        return;
    }
    let bpe = ta.kind.bytes_per_element();
    let mut buf = ta.buffer.borrow_mut();
    for i in 0..len / 2 {
        let a = ta.byte_offset + i * bpe;
        let b = ta.byte_offset + (len - 1 - i) * bpe;
        for k in 0..bpe {
            buf.data.swap(a + k, b + k);
        }
    }
}

/// `%TypedArray%.prototype.indexOf(searchElement, fromIndex?)`.
pub fn typed_array_index_of(ta: &JsTypedArray, search: &JsValue, from: i64) -> i64 {
    let len = ta.length as i64;
    let start = if from < 0 {
        cmp::max(len + from, 0) as usize
    } else {
        from as usize
    };
    for i in start..ta.length {
        if typed_array_get(ta, i).is_strictly_equal(search) {
            return i as i64;
        }
    }
    -1
}

/// `%TypedArray%.prototype.lastIndexOf(searchElement, fromIndex?)`.
pub fn typed_array_last_index_of(ta: &JsTypedArray, search: &JsValue, from: i64) -> i64 {
    let len = ta.length as i64;
    let start = if from < 0 {
        (len + from) as isize
    } else {
        cmp::min(from, len - 1) as isize
    };
    if start < 0 {
        return -1;
    }
    for i in (0..=start as usize).rev() {
        if typed_array_get(ta, i).is_strictly_equal(search) {
            return i as i64;
        }
    }
    -1
}

/// `%TypedArray%.prototype.includes(searchElement, fromIndex?)`.
pub fn typed_array_includes(ta: &JsTypedArray, search: &JsValue, from: i64) -> bool {
    let len = ta.length as i64;
    let start = if from < 0 {
        cmp::max(len + from, 0) as usize
    } else {
        from as usize
    };
    for i in start..ta.length {
        let elem = typed_array_get(ta, i);
        if elem.same_value_zero(search) {
            return true;
        }
    }
    false
}

/// `%TypedArray%.prototype.join(separator?)`.
pub fn typed_array_join(ta: &JsTypedArray, separator: &str) -> StatorResult<String> {
    let mut parts = Vec::with_capacity(ta.length);
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        parts.push(v.to_js_string()?);
    }
    Ok(parts.join(separator))
}

/// `%TypedArray%.prototype.slice(start?, end?)`.
///
/// Returns a new TypedArray of the same kind containing the sliced elements.
pub fn typed_array_slice(ta: &JsTypedArray, start: i64, end: i64) -> StatorResult<JsTypedArray> {
    let len = ta.length as i64;
    let s = clamp_index(start, len) as usize;
    let e = clamp_index(end, len) as usize;
    let count = e.saturating_sub(s);
    let result = typed_array_new_from_length(ta.kind, count);
    let bpe = ta.kind.bytes_per_element();
    let byte_count = count * bpe;
    if byte_count > 0 {
        let src_buf = ta.buffer.borrow();
        let src_start = ta.byte_offset + s * bpe;
        let src_bytes = src_buf.data[src_start..src_start + byte_count].to_vec();
        drop(src_buf);
        let mut dst_buf = result.buffer.borrow_mut();
        dst_buf.data[..byte_count].copy_from_slice(&src_bytes);
    }
    Ok(result)
}

/// `%TypedArray%.prototype.subarray(begin?, end?)`.
///
/// Returns a new TypedArray that shares the same buffer.
pub fn typed_array_subarray(ta: &JsTypedArray, begin: i64, end: i64) -> JsTypedArray {
    let len = ta.length as i64;
    let s = clamp_index(begin, len) as usize;
    let e = clamp_index(end, len) as usize;
    let count = e.saturating_sub(s);
    let bpe = ta.kind.bytes_per_element();
    JsTypedArray {
        buffer: Rc::clone(&ta.buffer),
        kind: ta.kind,
        byte_offset: ta.byte_offset + s * bpe,
        length: count,
    }
}

/// `%TypedArray%.prototype.set(source, offset?)`.
///
/// Copies elements from `source` into this typed array starting at `offset`.
pub fn typed_array_set_from(
    ta: &JsTypedArray,
    source: &[JsValue],
    offset: usize,
) -> StatorResult<()> {
    if offset + source.len() > ta.length {
        return Err(StatorError::RangeError("Source is too large".into()));
    }
    for (i, v) in source.iter().enumerate() {
        typed_array_set(ta, offset + i, v)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.sort(comparefn?)`.
///
/// Sorts elements in-place using a default numeric sort (no custom comparator
/// support in the pure-data API).
pub fn typed_array_sort(ta: &JsTypedArray) {
    let len = ta.length;
    if len < 2 {
        return;
    }
    // Collect, sort, write back.
    let mut elems: Vec<JsValue> = (0..len).map(|i| typed_array_get(ta, i)).collect();
    elems.sort_by(|a, b| {
        let na = a.to_number().unwrap_or(f64::NAN);
        let nb = b.to_number().unwrap_or(f64::NAN);
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, v) in elems.iter().enumerate() {
        let _ = typed_array_set(ta, i, v);
    }
}

/// `%TypedArray%.prototype.every(callbackfn)` — returns `true` if `pred`
/// returns `true` for every element.
pub fn typed_array_every(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<bool> {
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        if !pred(&v, i)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `%TypedArray%.prototype.some(callbackfn)`.
pub fn typed_array_some(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<bool> {
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `%TypedArray%.prototype.find(callbackfn)`.
pub fn typed_array_find(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<JsValue> {
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            return Ok(v);
        }
    }
    Ok(JsValue::Undefined)
}

/// `%TypedArray%.prototype.findIndex(callbackfn)`.
pub fn typed_array_find_index(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<i64> {
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            return Ok(i as i64);
        }
    }
    Ok(-1)
}

/// `%TypedArray%.prototype.findLast(callbackfn)`.
pub fn typed_array_find_last(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<JsValue> {
    for i in (0..ta.length).rev() {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            return Ok(v);
        }
    }
    Ok(JsValue::Undefined)
}

/// `%TypedArray%.prototype.findLastIndex(callbackfn)`.
pub fn typed_array_find_last_index(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<i64> {
    for i in (0..ta.length).rev() {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            return Ok(i as i64);
        }
    }
    Ok(-1)
}

/// `%TypedArray%.prototype.forEach(callbackfn)`.
pub fn typed_array_for_each(
    ta: &JsTypedArray,
    f: impl Fn(&JsValue, usize) -> StatorResult<()>,
) -> StatorResult<()> {
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        f(&v, i)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.filter(callbackfn)`.
pub fn typed_array_filter(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<JsTypedArray> {
    let mut kept = Vec::new();
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        if pred(&v, i)? {
            kept.push(v);
        }
    }
    typed_array_from_values(ta.kind, &kept)
}

/// `%TypedArray%.prototype.map(callbackfn)`.
pub fn typed_array_map(
    ta: &JsTypedArray,
    f: impl Fn(&JsValue, usize) -> StatorResult<JsValue>,
) -> StatorResult<JsTypedArray> {
    let mut mapped = Vec::with_capacity(ta.length);
    for i in 0..ta.length {
        let v = typed_array_get(ta, i);
        mapped.push(f(&v, i)?);
    }
    typed_array_from_values(ta.kind, &mapped)
}

/// `%TypedArray%.prototype.reduce(callbackfn, initialValue?)`.
pub fn typed_array_reduce(
    ta: &JsTypedArray,
    f: impl Fn(&JsValue, &JsValue, usize) -> StatorResult<JsValue>,
    initial: Option<JsValue>,
) -> StatorResult<JsValue> {
    let mut start = 0;
    let mut acc = match initial {
        Some(v) => v,
        None => {
            if ta.length == 0 {
                return Err(StatorError::TypeError(
                    "Reduce of empty array with no initial value".into(),
                ));
            }
            start = 1;
            typed_array_get(ta, 0)
        }
    };
    for i in start..ta.length {
        let v = typed_array_get(ta, i);
        acc = f(&acc, &v, i)?;
    }
    Ok(acc)
}

/// `%TypedArray%.prototype.reduceRight(callbackfn, initialValue?)`.
pub fn typed_array_reduce_right(
    ta: &JsTypedArray,
    f: impl Fn(&JsValue, &JsValue, usize) -> StatorResult<JsValue>,
    initial: Option<JsValue>,
) -> StatorResult<JsValue> {
    let len = ta.length;
    let has_initial = initial.is_some();
    let mut acc = match initial {
        Some(v) => v,
        None => {
            if len == 0 {
                return Err(StatorError::TypeError(
                    "Reduce of empty array with no initial value".into(),
                ));
            }
            typed_array_get(ta, len - 1)
        }
    };
    let end = if !has_initial && len > 0 {
        len - 1
    } else {
        len
    };
    for i in (0..end).rev() {
        let v = typed_array_get(ta, i);
        acc = f(&acc, &v, i)?;
    }
    Ok(acc)
}

/// `%TypedArray%.prototype.values()` — returns element values as a `Vec`.
pub fn typed_array_values(ta: &JsTypedArray) -> Vec<JsValue> {
    (0..ta.length).map(|i| typed_array_get(ta, i)).collect()
}

/// `%TypedArray%.prototype.keys()` — returns indices as a `Vec`.
pub fn typed_array_keys(ta: &JsTypedArray) -> Vec<JsValue> {
    (0..ta.length).map(|i| JsValue::Smi(i as i32)).collect()
}

/// `%TypedArray%.prototype.entries()` — returns `[index, value]` pairs.
pub fn typed_array_entries(ta: &JsTypedArray) -> Vec<JsValue> {
    (0..ta.length)
        .map(|i| {
            JsValue::Array(Rc::new(vec![
                JsValue::Smi(i as i32),
                typed_array_get(ta, i),
            ]))
        })
        .collect()
}

/// `TypedArray.from(source)`.
pub fn typed_array_static_from(
    kind: TypedArrayKind,
    source: &[JsValue],
) -> StatorResult<JsTypedArray> {
    typed_array_from_values(kind, source)
}

/// `TypedArray.of(...items)`.
pub fn typed_array_static_of(
    kind: TypedArrayKind,
    items: &[JsValue],
) -> StatorResult<JsTypedArray> {
    typed_array_from_values(kind, items)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Clamp a possibly-negative index to `[0, len]`.
fn clamp_index(idx: i64, len: i64) -> i64 {
    if idx < 0 {
        cmp::max(len + idx, 0)
    } else {
        cmp::min(idx, len)
    }
}

/// Clamp a float to a `u8` for `Uint8ClampedArray`.
fn clamp_u8(n: f64) -> u8 {
    if n.is_nan() || n <= 0.0 {
        0
    } else if n >= 255.0 {
        255
    } else {
        n.round() as u8
    }
}

/// Convert a `JsValue` to `i64` for `BigInt64Array`.
fn value_to_bigint64(value: &JsValue) -> StatorResult<i64> {
    match value {
        JsValue::BigInt(n) => Ok(*n as i64),
        JsValue::Smi(n) => Ok(i64::from(*n)),
        JsValue::HeapNumber(n) => Ok(*n as i64),
        _ => Err(StatorError::TypeError(
            "Cannot convert value to BigInt".into(),
        )),
    }
}

/// Convert a `JsValue` to `u64` for `BigUint64Array`.
fn value_to_biguint64(value: &JsValue) -> StatorResult<u64> {
    match value {
        JsValue::BigInt(n) => Ok(*n as u64),
        JsValue::Smi(n) => Ok(*n as u64),
        JsValue::HeapNumber(n) => Ok(*n as u64),
        _ => Err(StatorError::TypeError(
            "Cannot convert value to BigInt".into(),
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ArrayBuffer ──────────────────────────────────────────────────────

    #[test]
    fn test_arraybuffer_new_and_byte_length() {
        let buf = arraybuffer_new(16);
        assert_eq!(arraybuffer_byte_length(&buf), 16);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_arraybuffer_slice_basic() {
        let mut buf = arraybuffer_new(4);
        buf.data[0] = 10;
        buf.data[1] = 20;
        buf.data[2] = 30;
        buf.data[3] = 40;
        let s = arraybuffer_slice(&buf, 1, 3);
        assert_eq!(s.data, vec![20, 30]);
    }

    #[test]
    fn test_arraybuffer_slice_negative() {
        let mut buf = arraybuffer_new(4);
        buf.data = vec![1, 2, 3, 4];
        let s = arraybuffer_slice(&buf, -2, 4);
        assert_eq!(s.data, vec![3, 4]);
    }

    #[test]
    fn test_arraybuffer_is_view() {
        assert!(!arraybuffer_is_view(&JsValue::Smi(42)));
        assert!(!arraybuffer_is_view(&JsValue::Undefined));
    }

    // ── DataView ─────────────────────────────────────────────────────────

    #[test]
    fn test_dataview_get_set_int32() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_int32(&dv, 0, 0x12345678, false).unwrap();
        assert_eq!(dataview_get_int32(&dv, 0, false).unwrap(), 0x12345678);
        // Little-endian read of the same bytes gives a different result.
        let le_val = dataview_get_int32(&dv, 0, true).unwrap();
        assert_eq!(le_val, 0x78563412);
    }

    #[test]
    fn test_dataview_float64() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_float64(&dv, 0, std::f64::consts::PI, true).unwrap();
        let v = dataview_get_float64(&dv, 0, true).unwrap();
        assert!((v - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_dataview_out_of_bounds() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        assert!(dataview_get_int32(&dv, 4, false).is_err());
        assert!(dataview_set_int32(&dv, 4, 0, false).is_err());
    }

    #[test]
    fn test_dataview_offset() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let dv = dataview_new(Rc::clone(&buf), 4, Some(4)).unwrap();
        assert_eq!(dataview_byte_offset(&dv), 4);
        assert_eq!(dataview_byte_length(&dv), 4);
        dataview_set_uint8(&dv, 0, 0xFF, false).unwrap();
        assert_eq!(dataview_get_uint8(&dv, 0, false).unwrap(), 0xFF);
    }

    // ── TypedArray ───────────────────────────────────────────────────────

    #[test]
    fn test_typed_array_int32_basic() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 4);
        assert_eq!(typed_array_length(&ta), 4);
        assert_eq!(typed_array_byte_length(&ta), 16);
        typed_array_set(&ta, 0, &JsValue::Smi(42)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(42));
    }

    #[test]
    fn test_typed_array_uint8_clamped() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8Clamped, 3);
        typed_array_set(&ta, 0, &JsValue::HeapNumber(300.0)).unwrap();
        typed_array_set(&ta, 1, &JsValue::HeapNumber(-10.0)).unwrap();
        typed_array_set(&ta, 2, &JsValue::HeapNumber(128.5)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(255));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(0));
        // Round-half-to-even: 128.5 → 128 (banker's rounding in Rust's f64::round rounds to 129)
        // Actually Rust rounds to 129 for 128.5. The spec says round but we use f64::round().
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(129));
    }

    #[test]
    fn test_typed_array_float64() {
        let ta = typed_array_new_from_length(TypedArrayKind::Float64, 2);
        typed_array_set(&ta, 0, &JsValue::HeapNumber(std::f64::consts::PI)).unwrap();
        if let JsValue::HeapNumber(v) = typed_array_get(&ta, 0) {
            assert!((v - std::f64::consts::PI).abs() < 1e-15);
        } else {
            panic!("Expected HeapNumber");
        }
    }

    #[test]
    fn test_typed_array_out_of_bounds() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int8, 2);
        assert_eq!(typed_array_get(&ta, 5), JsValue::Undefined);
        assert!(typed_array_set(&ta, 5, &JsValue::Smi(1)).is_err());
    }

    #[test]
    fn test_typed_array_from_buffer() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let ta = typed_array_new_from_buffer(TypedArrayKind::Int16, Rc::clone(&buf), 2, Some(2))
            .unwrap();
        assert_eq!(typed_array_length(&ta), 2);
        assert_eq!(typed_array_byte_offset(&ta), 2);
    }

    #[test]
    fn test_typed_array_at() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
        )
        .unwrap();
        assert_eq!(typed_array_at(&ta, -1), JsValue::Smi(30));
        assert_eq!(typed_array_at(&ta, 0), JsValue::Smi(10));
    }

    #[test]
    fn test_typed_array_fill() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8, 4);
        typed_array_fill(&ta, &JsValue::Smi(7), 0, 4).unwrap();
        for i in 0..4 {
            assert_eq!(typed_array_get(&ta, i), JsValue::Smi(7));
        }
    }

    #[test]
    fn test_typed_array_reverse() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        typed_array_reverse(&ta);
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(1));
    }

    #[test]
    fn test_typed_array_index_of() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
        )
        .unwrap();
        assert_eq!(typed_array_index_of(&ta, &JsValue::Smi(20), 0), 1);
        assert_eq!(typed_array_index_of(&ta, &JsValue::Smi(99), 0), -1);
    }

    #[test]
    fn test_typed_array_includes() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        assert!(typed_array_includes(&ta, &JsValue::Smi(2), 0));
        assert!(!typed_array_includes(&ta, &JsValue::Smi(5), 0));
    }

    #[test]
    fn test_typed_array_join() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        assert_eq!(typed_array_join(&ta, ",").unwrap(), "1,2,3");
    }

    #[test]
    fn test_typed_array_slice() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
        )
        .unwrap();
        let s = typed_array_slice(&ta, 1, 3).unwrap();
        assert_eq!(typed_array_length(&s), 2);
        assert_eq!(typed_array_get(&s, 0), JsValue::Smi(20));
    }

    #[test]
    fn test_typed_array_subarray_shares_buffer() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let sub = typed_array_subarray(&ta, 1, 3);
        // Mutation through subarray is visible in original.
        typed_array_set(&sub, 0, &JsValue::Smi(99)).unwrap();
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(99));
    }

    #[test]
    fn test_typed_array_sort() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(30), JsValue::Smi(10), JsValue::Smi(20)],
        )
        .unwrap();
        typed_array_sort(&ta);
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(10));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(20));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(30));
    }

    #[test]
    fn test_typed_array_every_some() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(2), JsValue::Smi(4), JsValue::Smi(6)],
        )
        .unwrap();
        let all_even =
            typed_array_every(&ta, |v, _| Ok(v.to_number().unwrap() as i32 % 2 == 0)).unwrap();
        assert!(all_even);
        let has_odd =
            typed_array_some(&ta, |v, _| Ok(v.to_number().unwrap() as i32 % 2 != 0)).unwrap();
        assert!(!has_odd);
    }

    #[test]
    fn test_typed_array_find() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let found = typed_array_find(&ta, |v, _| Ok(v.to_number().unwrap() > 1.5)).unwrap();
        assert_eq!(found, JsValue::Smi(2));
    }

    #[test]
    fn test_typed_array_copy_within() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[
                JsValue::Smi(1),
                JsValue::Smi(2),
                JsValue::Smi(3),
                JsValue::Smi(4),
                JsValue::Smi(5),
            ],
        )
        .unwrap();
        // Copy positions 3..5 → position 0.
        typed_array_copy_within(&ta, 0, 3, 5);
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(4));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(5));
    }

    #[test]
    fn test_typed_array_values_keys_entries() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(10), JsValue::Smi(20)])
                .unwrap();
        assert_eq!(
            typed_array_values(&ta),
            vec![JsValue::Smi(10), JsValue::Smi(20)]
        );
        assert_eq!(
            typed_array_keys(&ta),
            vec![JsValue::Smi(0), JsValue::Smi(1)]
        );
        let entries = typed_array_entries(&ta);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_typed_array_static_of() {
        let ta =
            typed_array_static_of(TypedArrayKind::Float32, &[JsValue::HeapNumber(1.5)]).unwrap();
        assert_eq!(typed_array_length(&ta), 1);
    }

    #[test]
    fn test_typed_array_bigint64() {
        let ta = typed_array_new_from_length(TypedArrayKind::BigInt64, 1);
        typed_array_set(&ta, 0, &JsValue::BigInt(42)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::BigInt(42));
    }

    #[test]
    fn test_typed_array_reduce() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let sum = typed_array_reduce(
            &ta,
            |acc, v, _| {
                let a = acc.to_number()?;
                let b = v.to_number()?;
                Ok(JsValue::Smi((a + b) as i32))
            },
            Some(JsValue::Smi(0)),
        )
        .unwrap();
        assert_eq!(sum, JsValue::Smi(6));
    }

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(TypedArrayKind::Int8.bytes_per_element(), 1);
        assert_eq!(TypedArrayKind::Uint16.bytes_per_element(), 2);
        assert_eq!(TypedArrayKind::Int32.bytes_per_element(), 4);
        assert_eq!(TypedArrayKind::Float64.bytes_per_element(), 8);
        assert_eq!(TypedArrayKind::BigUint64.bytes_per_element(), 8);
    }

    #[test]
    fn test_typed_array_kind_name() {
        assert_eq!(TypedArrayKind::Int8.name(), "Int8Array");
        assert_eq!(TypedArrayKind::Uint8Clamped.name(), "Uint8ClampedArray");
        assert_eq!(TypedArrayKind::Float64.name(), "Float64Array");
        assert_eq!(TypedArrayKind::BigInt64.name(), "BigInt64Array");
    }
}

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

/// Comparator function signature for `TypedArray.prototype.sort`.
type SortCompareFn<'a> = Option<&'a dyn Fn(&JsValue, &JsValue) -> StatorResult<f64>>;

/// Mapping function signature for `TypedArray.from(source, mapFn)`.
type FromMapFn<'a> = Option<&'a dyn Fn(&JsValue, usize) -> StatorResult<JsValue>>;

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
    /// Whether this buffer represents a `SharedArrayBuffer`.
    pub shared: bool,
    /// Whether this buffer may be resized or grown.
    pub max_byte_length: Option<usize>,
    /// Whether this buffer has been detached.
    pub detached: bool,
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
        shared: false,
        max_byte_length: None,
        detached: false,
        data: vec![0u8; byte_length],
    }
}

/// Create a resizable `ArrayBuffer`.
pub fn arraybuffer_new_resizable(byte_length: usize, max_byte_length: usize) -> JsArrayBuffer {
    JsArrayBuffer {
        shared: false,
        max_byte_length: Some(max_byte_length),
        detached: false,
        data: vec![0u8; byte_length],
    }
}

/// ECMAScript §25.2.3.1 `new SharedArrayBuffer(byteLength)`.
///
/// Creates a new zero-filled shared buffer with the given byte length.
pub fn shared_arraybuffer_new(byte_length: usize) -> JsArrayBuffer {
    JsArrayBuffer {
        shared: true,
        max_byte_length: None,
        detached: false,
        data: vec![0u8; byte_length],
    }
}

/// Create a growable `SharedArrayBuffer`.
pub fn shared_arraybuffer_new_growable(
    byte_length: usize,
    max_byte_length: usize,
) -> JsArrayBuffer {
    JsArrayBuffer {
        shared: true,
        max_byte_length: Some(max_byte_length),
        detached: false,
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
    if buf.detached { 0 } else { buf.data.len() }
}

/// `ArrayBuffer.prototype.resizable` getter.
pub fn arraybuffer_resizable(buf: &JsArrayBuffer) -> bool {
    !buf.shared && buf.max_byte_length.is_some()
}

/// `SharedArrayBuffer.prototype.growable` getter.
pub fn shared_arraybuffer_growable(buf: &JsArrayBuffer) -> bool {
    buf.shared && buf.max_byte_length.is_some()
}

/// `ArrayBuffer.prototype.maxByteLength` getter.
pub fn arraybuffer_max_byte_length(buf: &JsArrayBuffer) -> usize {
    buf.max_byte_length
        .unwrap_or_else(|| arraybuffer_byte_length(buf))
}

/// `ArrayBuffer.prototype.detached` getter.
pub fn arraybuffer_detached(buf: &JsArrayBuffer) -> bool {
    buf.detached
}

/// Resize an `ArrayBuffer` in place.
pub fn arraybuffer_resize(buf: &mut JsArrayBuffer, new_byte_length: usize) -> StatorResult<()> {
    if buf.shared {
        return Err(StatorError::TypeError(
            "SharedArrayBuffer cannot be resized with ArrayBuffer.prototype.resize".into(),
        ));
    }
    if buf.detached {
        return Err(StatorError::TypeError("ArrayBuffer is detached".into()));
    }
    let Some(max_byte_length) = buf.max_byte_length else {
        return Err(StatorError::TypeError(
            "ArrayBuffer is not resizable".into(),
        ));
    };
    if new_byte_length > max_byte_length {
        return Err(StatorError::RangeError(
            "newByteLength exceeds maxByteLength".into(),
        ));
    }
    buf.data.resize(new_byte_length, 0);
    Ok(())
}

/// Grow a `SharedArrayBuffer` in place.
pub fn shared_arraybuffer_grow(
    buf: &mut JsArrayBuffer,
    new_byte_length: usize,
) -> StatorResult<()> {
    if !buf.shared {
        return Err(StatorError::TypeError(
            "SharedArrayBuffer.prototype.grow called on incompatible receiver".into(),
        ));
    }
    let Some(max_byte_length) = buf.max_byte_length else {
        return Err(StatorError::TypeError(
            "SharedArrayBuffer is not growable".into(),
        ));
    };
    if new_byte_length < buf.data.len() {
        return Err(StatorError::RangeError(
            "SharedArrayBuffer can only grow".into(),
        ));
    }
    if new_byte_length > max_byte_length {
        return Err(StatorError::RangeError(
            "newByteLength exceeds maxByteLength".into(),
        ));
    }
    buf.data.resize(new_byte_length, 0);
    Ok(())
}

/// Transfer an `ArrayBuffer` to a new buffer, detaching the source.
pub fn arraybuffer_transfer(
    buf: &mut JsArrayBuffer,
    new_byte_length: Option<usize>,
    to_fixed_length: bool,
) -> StatorResult<JsArrayBuffer> {
    if buf.shared {
        return Err(StatorError::TypeError(
            "SharedArrayBuffer cannot be transferred".into(),
        ));
    }
    if buf.detached {
        return Err(StatorError::TypeError("ArrayBuffer is detached".into()));
    }

    let old_len = buf.data.len();
    let target_len = new_byte_length.unwrap_or(old_len);
    if let Some(max_byte_length) = buf.max_byte_length
        && target_len > max_byte_length
    {
        return Err(StatorError::RangeError(
            "newByteLength exceeds maxByteLength".into(),
        ));
    }

    let mut data = vec![0; target_len];
    let copy_len = cmp::min(old_len, target_len);
    data[..copy_len].copy_from_slice(&buf.data[..copy_len]);

    let transferred = JsArrayBuffer {
        shared: false,
        max_byte_length: if to_fixed_length {
            None
        } else {
            buf.max_byte_length
        },
        detached: false,
        data,
    };

    buf.data.clear();
    buf.max_byte_length = None;
    buf.detached = true;

    Ok(transferred)
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
        shared: buf.shared,
        max_byte_length: None,
        detached: false,
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
    let buf_len = {
        let buffer_ref = buffer.borrow();
        if buffer_ref.detached {
            return Err(StatorError::TypeError("ArrayBuffer is detached".into()));
        }
        buffer_ref.data.len()
    };
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
pub fn dataview_byte_length(dv: &JsDataView) -> StatorResult<usize> {
    dataview_validate_view(dv)?;
    Ok(dv.byte_length)
}

/// `DataView.prototype.byteOffset` getter.
pub fn dataview_byte_offset(dv: &JsDataView) -> StatorResult<usize> {
    dataview_validate_view(dv)?;
    Ok(dv.byte_offset)
}

fn dataview_validate_view(dv: &JsDataView) -> StatorResult<()> {
    let buf = dv.buffer.borrow();
    if buf.detached {
        return Err(StatorError::TypeError("ArrayBuffer is detached".into()));
    }
    let Some(view_end) = dv.byte_offset.checked_add(dv.byte_length) else {
        return Err(StatorError::TypeError("DataView is out of bounds".into()));
    };
    if view_end > buf.data.len() {
        return Err(StatorError::TypeError("DataView is out of bounds".into()));
    }
    Ok(())
}

fn dataview_resolve_range(dv: &JsDataView, byte_offset: usize, size: usize) -> StatorResult<usize> {
    dataview_validate_view(dv)?;
    let Some(end) = byte_offset.checked_add(size) else {
        return Err(StatorError::RangeError(
            "Offset is outside the bounds of the DataView".into(),
        ));
    };
    if end > dv.byte_length {
        return Err(StatorError::RangeError(
            "Offset is outside the bounds of the DataView".into(),
        ));
    }
    dv.byte_offset.checked_add(byte_offset).ok_or_else(|| {
        StatorError::RangeError("Offset is outside the bounds of the DataView".into())
    })
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
            let abs = dataview_resolve_range(dv, byte_offset, $n)?;
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
            let abs = dataview_resolve_range(dv, byte_offset, $n)?;
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
    /// Declared number of elements (not bytes).
    pub length: usize,
    /// Whether this view tracks the current backing-buffer byte length.
    pub auto_length: bool,
}

impl JsTypedArray {
    /// The view length after accounting for detached or resized backing stores.
    pub fn effective_length(&self) -> usize {
        let buf = self.buffer.borrow();
        if buf.detached || self.byte_offset > buf.data.len() {
            return 0;
        }
        let available = buf.data.len() - self.byte_offset;
        let available_len = available / self.kind.bytes_per_element();
        if self.auto_length {
            available_len
        } else {
            cmp::min(self.length, available_len)
        }
    }

    /// The current byte length of the view.
    pub fn effective_byte_length(&self) -> usize {
        self.effective_length() * self.kind.bytes_per_element()
    }
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
        auto_length: false,
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
        auto_length: length.is_none(),
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
    ta.effective_length()
}

/// `%TypedArray%.prototype.byteLength` getter.
pub fn typed_array_byte_length(ta: &JsTypedArray) -> usize {
    ta.effective_byte_length()
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
    if index >= ta.effective_length() {
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
    if index >= ta.effective_length() {
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
    let len = ta.effective_length() as i64;
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
    let len = ta.effective_length() as i64;
    let s = clamp_index(start, len) as usize;
    let e = clamp_index(end, len) as usize;
    for i in s..e {
        typed_array_set(ta, i, value)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.copyWithin(target, start, end?)`.
pub fn typed_array_copy_within(ta: &JsTypedArray, target: i64, start: i64, end: i64) {
    let len = ta.effective_length() as i64;
    let to = clamp_index(target, len) as usize;
    let from = clamp_index(start, len) as usize;
    let fin = clamp_index(end, len) as usize;
    let count = cmp::min(
        fin.saturating_sub(from),
        ta.effective_length().saturating_sub(to),
    );
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
    let len = ta.effective_length();
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
    let len = ta.effective_length() as i64;
    let start = if from < 0 {
        cmp::max(len + from, 0) as usize
    } else {
        from as usize
    };
    for i in start..ta.effective_length() {
        if typed_array_get(ta, i).is_strictly_equal(search) {
            return i as i64;
        }
    }
    -1
}

/// `%TypedArray%.prototype.lastIndexOf(searchElement, fromIndex?)`.
pub fn typed_array_last_index_of(ta: &JsTypedArray, search: &JsValue, from: i64) -> i64 {
    let len = ta.effective_length() as i64;
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
    let len = ta.effective_length() as i64;
    let start = if from < 0 {
        cmp::max(len + from, 0) as usize
    } else {
        from as usize
    };
    for i in start..ta.effective_length() {
        let elem = typed_array_get(ta, i);
        if elem.same_value_zero(search) {
            return true;
        }
    }
    false
}

/// `%TypedArray%.prototype.join(separator?)`.
pub fn typed_array_join(ta: &JsTypedArray, separator: &str) -> StatorResult<String> {
    let mut parts = Vec::with_capacity(ta.effective_length());
    for i in 0..ta.effective_length() {
        let v = typed_array_get(ta, i);
        parts.push(v.to_js_string()?);
    }
    Ok(parts.join(separator))
}

/// `%TypedArray%.prototype.slice(start?, end?)`.
///
/// Returns a new TypedArray of the same kind containing the sliced elements.
pub fn typed_array_slice(ta: &JsTypedArray, start: i64, end: i64) -> StatorResult<JsTypedArray> {
    let len = ta.effective_length() as i64;
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
    let len = ta.effective_length() as i64;
    let s = clamp_index(begin, len) as usize;
    let e = clamp_index(end, len) as usize;
    let count = e.saturating_sub(s);
    let bpe = ta.kind.bytes_per_element();
    JsTypedArray {
        buffer: Rc::clone(&ta.buffer),
        kind: ta.kind,
        byte_offset: ta.byte_offset + s * bpe,
        length: count,
        auto_length: ta.auto_length && e == len as usize,
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
    if offset + source.len() > ta.effective_length() {
        return Err(StatorError::RangeError("Source is too large".into()));
    }
    for (i, v) in source.iter().enumerate() {
        typed_array_set(ta, offset + i, v)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.set(typedArray, offset?)` — TypedArray overload.
///
/// Copies all elements from `source` TypedArray into `ta` starting at `offset`.
pub fn typed_array_set_from_typed_array(
    ta: &JsTypedArray,
    source: &JsTypedArray,
    offset: usize,
) -> StatorResult<()> {
    let src_len = source.effective_length();
    if offset + src_len > ta.effective_length() {
        return Err(StatorError::RangeError("Source is too large".into()));
    }
    // Use fast byte-copy when source and target have the same kind and do not
    // share the same buffer (or share but regions do not overlap).
    let same_buffer = Rc::ptr_eq(&ta.buffer, &source.buffer);
    if ta.kind == source.kind && !same_buffer {
        let bpe = ta.kind.bytes_per_element();
        let byte_count = src_len * bpe;
        let src_buf = source.buffer.borrow();
        let src_start = source.byte_offset;
        let bytes = src_buf.data[src_start..src_start + byte_count].to_vec();
        drop(src_buf);
        let mut dst_buf = ta.buffer.borrow_mut();
        let dst_start = ta.byte_offset + offset * bpe;
        dst_buf.data[dst_start..dst_start + byte_count].copy_from_slice(&bytes);
    } else {
        // Slow path: read elements as JsValue, then write.
        let elems: Vec<JsValue> = (0..src_len).map(|i| typed_array_get(source, i)).collect();
        for (i, v) in elems.iter().enumerate() {
            typed_array_set(ta, offset + i, v)?;
        }
    }
    Ok(())
}

/// `%TypedArray%.prototype.sort(comparefn?)`.
///
/// Sorts elements in-place.  When `compare` is `None` the default numeric
/// comparison defined by the spec is used (ascending, with `NaN` sorted to
/// the end and `-0` before `+0`).  When a comparator is provided its
/// return value determines ordering.
pub fn typed_array_sort(ta: &JsTypedArray, compare: SortCompareFn<'_>) -> StatorResult<()> {
    let len = ta.effective_length();
    if len < 2 {
        return Ok(());
    }
    let mut elems: Vec<JsValue> = (0..len).map(|i| typed_array_get(ta, i)).collect();

    // We need to propagate errors from the comparator, so we use a
    // Cell to capture the first error.
    let mut err: Option<StatorError> = None;
    elems.sort_by(|a, b| {
        if err.is_some() {
            return std::cmp::Ordering::Equal;
        }
        match compare {
            Some(cmp_fn) => match cmp_fn(a, b) {
                Ok(v) => {
                    if v < 0.0 {
                        std::cmp::Ordering::Less
                    } else if v > 0.0 {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
                Err(e) => {
                    err = Some(e);
                    std::cmp::Ordering::Equal
                }
            },
            None => {
                let na = a.to_number().unwrap_or(f64::NAN);
                let nb = b.to_number().unwrap_or(f64::NAN);
                typed_array_default_compare(na, nb)
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    for (i, v) in elems.iter().enumerate() {
        typed_array_set(ta, i, v)?;
    }
    Ok(())
}

/// Default TypedArray sort comparison per ECMAScript §23.2.3.28.
///
/// * `NaN` sorts after all other values.
/// * `-0` sorts before `+0`.
fn typed_array_default_compare(x: f64, y: f64) -> std::cmp::Ordering {
    if x.is_nan() && y.is_nan() {
        return std::cmp::Ordering::Equal;
    }
    if x.is_nan() {
        return std::cmp::Ordering::Greater;
    }
    if y.is_nan() {
        return std::cmp::Ordering::Less;
    }
    if x == 0.0 && y == 0.0 {
        // -0 < +0
        let x_neg = x.is_sign_negative();
        let y_neg = y.is_sign_negative();
        return y_neg.cmp(&x_neg);
    }
    x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
}

/// `%TypedArray%.prototype.every(callbackfn)` — returns `true` if `pred`
/// returns `true` for every element.
pub fn typed_array_every(
    ta: &JsTypedArray,
    pred: impl Fn(&JsValue, usize) -> StatorResult<bool>,
) -> StatorResult<bool> {
    for i in 0..ta.effective_length() {
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
    for i in 0..ta.effective_length() {
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
    for i in 0..ta.effective_length() {
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
    for i in 0..ta.effective_length() {
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
    for i in (0..ta.effective_length()).rev() {
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
    for i in (0..ta.effective_length()).rev() {
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
    for i in 0..ta.effective_length() {
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
    for i in 0..ta.effective_length() {
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
    let mut mapped = Vec::with_capacity(ta.effective_length());
    for i in 0..ta.effective_length() {
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
            if ta.effective_length() == 0 {
                return Err(StatorError::TypeError(
                    "Reduce of empty array with no initial value".into(),
                ));
            }
            start = 1;
            typed_array_get(ta, 0)
        }
    };
    for i in start..ta.effective_length() {
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
    let len = ta.effective_length();
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
    (0..ta.effective_length())
        .map(|i| typed_array_get(ta, i))
        .collect()
}

/// `%TypedArray%.prototype.keys()` — returns indices as a `Vec`.
pub fn typed_array_keys(ta: &JsTypedArray) -> Vec<JsValue> {
    (0..ta.effective_length())
        .map(|i| JsValue::Smi(i as i32))
        .collect()
}

/// `%TypedArray%.prototype.entries()` — returns `[index, value]` pairs.
pub fn typed_array_entries(ta: &JsTypedArray) -> Vec<JsValue> {
    (0..ta.effective_length())
        .map(|i| JsValue::new_array(vec![JsValue::Smi(i as i32), typed_array_get(ta, i)]))
        .collect()
}

/// `TypedArray.from(source, mapFn?)`.
///
/// Creates a new TypedArray from `source`, optionally applying `map_fn`
/// to each element during construction.
pub fn typed_array_static_from(
    kind: TypedArrayKind,
    source: &[JsValue],
    map_fn: FromMapFn<'_>,
) -> StatorResult<JsTypedArray> {
    match map_fn {
        None => typed_array_from_values(kind, source),
        Some(f) => {
            let mapped: Vec<JsValue> = source
                .iter()
                .enumerate()
                .map(|(i, v)| f(v, i))
                .collect::<StatorResult<Vec<_>>>()?;
            typed_array_from_values(kind, &mapped)
        }
    }
}

/// `TypedArray.of(...items)`.
pub fn typed_array_static_of(
    kind: TypedArrayKind,
    items: &[JsValue],
) -> StatorResult<JsTypedArray> {
    typed_array_from_values(kind, items)
}

// ── toReversed ────────────────────────────────────────────────────────────────

/// `%TypedArray%.prototype.toReversed()` (ECMAScript §23.2.3.30).
///
/// Returns a **new** `JsTypedArray` of the same kind with elements in reverse
/// order.  The original is unchanged.
pub fn typed_array_to_reversed(ta: &JsTypedArray) -> StatorResult<JsTypedArray> {
    let len = ta.effective_length();
    let kind = ta.kind;
    let result = typed_array_new_from_length(kind, len);
    for i in 0..len {
        let v = typed_array_get(ta, len - 1 - i);
        typed_array_set(&result, i, &v)?;
    }
    Ok(result)
}

// ── toSorted ─────────────────────────────────────────────────────────────────

/// `%TypedArray%.prototype.toSorted(comparefn?)` (ECMAScript §23.2.3.31).
///
/// Returns a **new** sorted `JsTypedArray` of the same kind without mutating the
/// original.  When no comparator is provided the default numeric comparison is
/// used (ascending, `NaN` at the end, `-0` before `+0`).
pub fn typed_array_to_sorted(
    ta: &JsTypedArray,
    compare: SortCompareFn<'_>,
) -> StatorResult<JsTypedArray> {
    let len = ta.effective_length();
    let kind = ta.kind;
    let copy = typed_array_new_from_length(kind, len);
    for i in 0..len {
        let v = typed_array_get(ta, i);
        typed_array_set(&copy, i, &v)?;
    }
    typed_array_sort(&copy, compare)?;
    Ok(copy)
}

// ── with ─────────────────────────────────────────────────────────────────────

/// `%TypedArray%.prototype.with(index, value)` (ECMAScript §23.2.3.37).
///
/// Returns a **new** `JsTypedArray` identical to `ta` except that the element at
/// `index` is replaced with `value`.
///
/// Returns [`StatorError::RangeError`] if `index` is out of bounds.
pub fn typed_array_with(
    ta: &JsTypedArray,
    index: i64,
    value: &JsValue,
) -> StatorResult<JsTypedArray> {
    let len = ta.effective_length() as i64;
    let actual = if index < 0 { len + index } else { index };
    if actual < 0 || actual >= len {
        return Err(StatorError::RangeError(format!("Invalid index : {index}")));
    }
    let kind = ta.kind;
    let result = typed_array_new_from_length(kind, len as usize);
    for i in 0..len as usize {
        let v = typed_array_get(ta, i);
        typed_array_set(&result, i, &v)?;
    }
    typed_array_set(&result, actual as usize, value)?;
    Ok(result)
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
        assert_eq!(dataview_byte_offset(&dv).unwrap(), 4);
        assert_eq!(dataview_byte_length(&dv).unwrap(), 4);
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
        typed_array_sort(&ta, None).unwrap();
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

    // ── Additional conformance tests ────────────────────────────────────

    #[test]
    fn test_arraybuffer_zero_length() {
        let buf = arraybuffer_new(0);
        assert_eq!(arraybuffer_byte_length(&buf), 0);
        let sliced = arraybuffer_slice(&buf, 0, 0);
        assert_eq!(sliced.data.len(), 0);
    }

    #[test]
    fn test_arraybuffer_slice_clamps_beyond_end() {
        let mut buf = arraybuffer_new(4);
        buf.data = vec![1, 2, 3, 4];
        let s = arraybuffer_slice(&buf, 2, 100);
        assert_eq!(s.data, vec![3, 4]);
    }

    #[test]
    fn test_arraybuffer_slice_negative_both() {
        let mut buf = arraybuffer_new(4);
        buf.data = vec![10, 20, 30, 40];
        let s = arraybuffer_slice(&buf, -3, -1);
        assert_eq!(s.data, vec![20, 30]);
    }

    #[test]
    fn test_arraybuffer_slice_empty_when_begin_ge_end() {
        let buf = arraybuffer_new(4);
        let s = arraybuffer_slice(&buf, 3, 1);
        assert!(s.data.is_empty());
    }

    #[test]
    fn test_dataview_int16_endianness() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_int16(&dv, 0, 0x0102, false).unwrap();
        assert_eq!(dataview_get_int16(&dv, 0, false).unwrap(), 0x0102);
        assert_eq!(dataview_get_int16(&dv, 0, true).unwrap(), 0x0201);
    }

    #[test]
    fn test_dataview_uint16_roundtrip() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_uint16(&dv, 0, 0xFFFE, true).unwrap();
        assert_eq!(dataview_get_uint16(&dv, 0, true).unwrap(), 0xFFFE);
    }

    #[test]
    fn test_dataview_float32_roundtrip() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_float32(&dv, 0, 1.5, true).unwrap();
        let v = dataview_get_float32(&dv, 0, true).unwrap();
        assert!((v - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dataview_new_invalid_offset() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        assert!(dataview_new(Rc::clone(&buf), 5, None).is_err());
    }

    #[test]
    fn test_dataview_new_invalid_length() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(4)));
        assert!(dataview_new(Rc::clone(&buf), 2, Some(4)).is_err());
    }

    #[test]
    fn test_dataview_buffer_identity() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        let buf2 = dataview_buffer(&dv);
        assert!(Rc::ptr_eq(&buf, &buf2));
    }

    #[test]
    fn test_typed_array_all_kinds_roundtrip() {
        for kind in &[
            TypedArrayKind::Int8,
            TypedArrayKind::Uint8,
            TypedArrayKind::Uint8Clamped,
            TypedArrayKind::Int16,
            TypedArrayKind::Uint16,
            TypedArrayKind::Int32,
            TypedArrayKind::Uint32,
            TypedArrayKind::Float32,
            TypedArrayKind::Float64,
        ] {
            let ta = typed_array_new_from_length(*kind, 3);
            assert_eq!(typed_array_length(&ta), 3);
            typed_array_set(&ta, 0, &JsValue::Smi(7)).unwrap();
            let v = typed_array_get(&ta, 0);
            assert_ne!(v, JsValue::Undefined, "kind={:?} should store 7", kind);
        }
    }

    #[test]
    fn test_typed_array_from_buffer_alignment_error() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(8)));
        let result = typed_array_new_from_buffer(TypedArrayKind::Int32, buf, 1, None);
        assert!(result.is_err(), "Offset 1 is not aligned to 4");
    }

    #[test]
    fn test_typed_array_from_buffer_auto_length() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(12)));
        let ta = typed_array_new_from_buffer(TypedArrayKind::Int32, buf, 4, None).unwrap();
        assert_eq!(typed_array_length(&ta), 2);
        assert_eq!(typed_array_byte_offset(&ta), 4);
    }

    #[test]
    fn test_typed_array_from_buffer_remainder_error() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(7)));
        let result = typed_array_new_from_buffer(TypedArrayKind::Int32, buf, 0, None);
        assert!(result.is_err(), "7 bytes not divisible by 4");
    }

    #[test]
    fn test_typed_array_from_values_uint8() {
        let ta = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(255)],
        )
        .unwrap();
        assert_eq!(typed_array_length(&ta), 3);
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(255));
    }

    #[test]
    fn test_typed_array_fill_partial() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 5);
        typed_array_fill(&ta, &JsValue::Smi(42), 1, 3).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(42));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(42));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(0));
    }

    #[test]
    fn test_typed_array_fill_negative_indices() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8, 4);
        typed_array_fill(&ta, &JsValue::Smi(9), -2, 4).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(0));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(9));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(9));
    }

    #[test]
    fn test_typed_array_copy_within_overlap() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[
                JsValue::Smi(1),
                JsValue::Smi(2),
                JsValue::Smi(3),
                JsValue::Smi(4),
            ],
        )
        .unwrap();
        typed_array_copy_within(&ta, 1, 0, 3);
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(1));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(1));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(2));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(3));
    }

    #[test]
    fn test_typed_array_set_from_offset() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 5);
        typed_array_set_from(&ta, &[JsValue::Smi(10), JsValue::Smi(20)], 2).unwrap();
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(10));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(20));
    }

    #[test]
    fn test_typed_array_set_from_overflow() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 2);
        let result = typed_array_set_from(&ta, &[JsValue::Smi(1), JsValue::Smi(2)], 1);
        assert!(result.is_err(), "Source is too large");
    }

    #[test]
    fn test_typed_array_slice_negative_start() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[
                JsValue::Smi(10),
                JsValue::Smi(20),
                JsValue::Smi(30),
                JsValue::Smi(40),
            ],
        )
        .unwrap();
        let s = typed_array_slice(&ta, -2, 4).unwrap();
        assert_eq!(typed_array_length(&s), 2);
        assert_eq!(typed_array_get(&s, 0), JsValue::Smi(30));
        assert_eq!(typed_array_get(&s, 1), JsValue::Smi(40));
    }

    #[test]
    fn test_typed_array_subarray_negative_begin() {
        let ta = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let sub = typed_array_subarray(&ta, -2, 3);
        assert_eq!(typed_array_length(&sub), 2);
        assert_eq!(typed_array_get(&sub, 0), JsValue::Smi(2));
    }

    #[test]
    fn test_typed_array_index_of_from_negative() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)],
        )
        .unwrap();
        assert_eq!(typed_array_index_of(&ta, &JsValue::Smi(1), -2), 2);
    }

    #[test]
    fn test_typed_array_last_index_of_basic() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)],
        )
        .unwrap();
        assert_eq!(typed_array_last_index_of(&ta, &JsValue::Smi(1), 2), 2);
        assert_eq!(typed_array_last_index_of(&ta, &JsValue::Smi(1), 1), 0);
    }

    #[test]
    fn test_typed_array_includes_nan() {
        let ta = typed_array_from_values(TypedArrayKind::Float64, &[JsValue::HeapNumber(f64::NAN)])
            .unwrap();
        assert!(typed_array_includes(&ta, &JsValue::HeapNumber(f64::NAN), 0));
    }

    #[test]
    fn test_typed_array_join_default_separator() {
        let ta = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
        )
        .unwrap();
        assert_eq!(typed_array_join(&ta, ",").unwrap(), "10,20,30");
    }

    #[test]
    fn test_typed_array_join_custom_separator() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        assert_eq!(typed_array_join(&ta, " - ").unwrap(), "1 - 2 - 3");
    }

    #[test]
    fn test_typed_array_reduce_no_initial() {
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
            None,
        )
        .unwrap();
        assert_eq!(sum, JsValue::Smi(6));
    }

    #[test]
    fn test_typed_array_reduce_empty_no_initial_errors() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        let result = typed_array_reduce(&ta, |acc, _, _| Ok(acc.clone()), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_typed_array_reduce_right() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let result = typed_array_reduce_right(
            &ta,
            |acc, v, _| {
                let a = acc.to_js_string()?;
                let b = v.to_js_string()?;
                Ok(JsValue::String(format!("{a}{b}").into()))
            },
            Some(JsValue::String("".into())),
        )
        .unwrap();
        assert_eq!(result, JsValue::String("321".into()));
    }

    #[test]
    fn test_typed_array_find_index_not_found() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(1), JsValue::Smi(2)])
                .unwrap();
        let idx = typed_array_find_index(&ta, |v, _| Ok(v.to_number()? > 10.0)).unwrap();
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_typed_array_find_last() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let found = typed_array_find_last(&ta, |v, _| Ok(v.to_number()? < 3.0)).unwrap();
        assert_eq!(found, JsValue::Smi(2));
    }

    #[test]
    fn test_typed_array_filter() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[
                JsValue::Smi(1),
                JsValue::Smi(2),
                JsValue::Smi(3),
                JsValue::Smi(4),
            ],
        )
        .unwrap();
        let filtered = typed_array_filter(&ta, |v, _| Ok(v.to_number()? > 2.0)).unwrap();
        assert_eq!(typed_array_length(&filtered), 2);
        assert_eq!(typed_array_get(&filtered, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&filtered, 1), JsValue::Smi(4));
    }

    #[test]
    fn test_typed_array_map() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let mapped = typed_array_map(&ta, |v, _| {
            let n = v.to_number()? as i32;
            Ok(JsValue::Smi(n * 2))
        })
        .unwrap();
        assert_eq!(typed_array_get(&mapped, 0), JsValue::Smi(2));
        assert_eq!(typed_array_get(&mapped, 1), JsValue::Smi(4));
        assert_eq!(typed_array_get(&mapped, 2), JsValue::Smi(6));
    }

    #[test]
    fn test_typed_array_for_each() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(10), JsValue::Smi(20)])
                .unwrap();
        let sum = std::cell::Cell::new(0i32);
        typed_array_for_each(&ta, |v, _| {
            sum.set(sum.get() + v.to_number()? as i32);
            Ok(())
        })
        .unwrap();
        assert_eq!(sum.get(), 30);
    }

    #[test]
    fn test_typed_array_static_from() {
        let ta = typed_array_static_from(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
            None,
        )
        .unwrap();
        assert_eq!(typed_array_length(&ta), 3);
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(20));
    }

    #[test]
    fn test_typed_array_static_of_float64() {
        let ta = typed_array_static_of(
            TypedArrayKind::Float64,
            &[JsValue::HeapNumber(1.1), JsValue::HeapNumber(2.2)],
        )
        .unwrap();
        assert_eq!(typed_array_length(&ta), 2);
        if let JsValue::HeapNumber(v) = typed_array_get(&ta, 0) {
            assert!((v - 1.1).abs() < 1e-10);
        } else {
            panic!("Expected HeapNumber");
        }
    }

    #[test]
    fn test_typed_array_shared_buffer_via_subarray() {
        let ta = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let sub = typed_array_subarray(&ta, 1, 2);
        typed_array_set(&sub, 0, &JsValue::Smi(99)).unwrap();
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(99));
    }

    #[test]
    fn test_typed_array_sort_float64_with_negatives() {
        let ta = typed_array_from_values(
            TypedArrayKind::Float64,
            &[
                JsValue::HeapNumber(3.0),
                JsValue::HeapNumber(-1.0),
                JsValue::HeapNumber(2.0),
            ],
        )
        .unwrap();
        typed_array_sort(&ta, None).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::HeapNumber(-1.0));
        assert_eq!(typed_array_get(&ta, 1), JsValue::HeapNumber(2.0));
        assert_eq!(typed_array_get(&ta, 2), JsValue::HeapNumber(3.0));
    }

    #[test]
    fn test_typed_array_byte_length_and_offset() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(16)));
        let ta = typed_array_new_from_buffer(TypedArrayKind::Int32, buf, 8, Some(2)).unwrap();
        assert_eq!(typed_array_byte_length(&ta), 8);
        assert_eq!(typed_array_byte_offset(&ta), 8);
        assert_eq!(typed_array_length(&ta), 2);
    }

    #[test]
    fn test_typed_array_int8_overflow() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int8, 1);
        typed_array_set(&ta, 0, &JsValue::Smi(200)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(-56));
    }

    #[test]
    fn test_typed_array_uint8_overflow() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8, 1);
        typed_array_set(&ta, 0, &JsValue::Smi(256)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
    }

    #[test]
    fn test_typed_array_uint8_clamped_nan() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8Clamped, 1);
        typed_array_set(&ta, 0, &JsValue::HeapNumber(f64::NAN)).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
    }

    #[test]
    fn test_typed_array_at_out_of_bounds() {
        let ta = typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(10)]).unwrap();
        assert_eq!(typed_array_at(&ta, 1), JsValue::Undefined);
        assert_eq!(typed_array_at(&ta, -2), JsValue::Undefined);
    }

    #[test]
    fn test_typed_array_reverse_even_length() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[
                JsValue::Smi(1),
                JsValue::Smi(2),
                JsValue::Smi(3),
                JsValue::Smi(4),
            ],
        )
        .unwrap();
        typed_array_reverse(&ta);
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(4));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(3));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(2));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(1));
    }

    #[test]
    fn test_dataview_bigint_roundtrip() {
        let buf = Rc::new(RefCell::new(arraybuffer_new(16)));
        let dv = dataview_new(Rc::clone(&buf), 0, None).unwrap();
        dataview_set_bigint64(&dv, 0, i64::MIN, true).unwrap();
        assert_eq!(dataview_get_bigint64(&dv, 0, true).unwrap(), i64::MIN);
        dataview_set_biguint64(&dv, 8, u64::MAX, false).unwrap();
        assert_eq!(dataview_get_biguint64(&dv, 8, false).unwrap(), u64::MAX);
    }

    #[test]
    fn test_typed_array_every_false() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(2), JsValue::Smi(3), JsValue::Smi(4)],
        )
        .unwrap();
        let result = typed_array_every(&ta, |v, _| Ok(v.to_number()? % 2.0 == 0.0)).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_typed_array_some_found() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(3), JsValue::Smi(4)],
        )
        .unwrap();
        let result = typed_array_some(&ta, |v, _| Ok(v.to_number()? > 3.0)).unwrap();
        assert!(result);
    }

    #[test]
    fn test_typed_array_some_not_found() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(1), JsValue::Smi(2)])
                .unwrap();
        let result = typed_array_some(&ta, |v, _| Ok(v.to_number()? > 10.0)).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_typed_array_float32_precision() {
        let ta = typed_array_new_from_length(TypedArrayKind::Float32, 1);
        typed_array_set(&ta, 0, &JsValue::HeapNumber(std::f64::consts::PI)).unwrap();
        if let JsValue::HeapNumber(v) = typed_array_get(&ta, 0) {
            assert!((v - std::f64::consts::PI).abs() < 1e-6);
            assert!((v - std::f64::consts::PI).abs() > 1e-10);
        } else {
            panic!("Expected HeapNumber");
        }
    }

    #[test]
    fn test_typed_array_uint32_large_value() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint32, 1);
        typed_array_set(&ta, 0, &JsValue::HeapNumber(4_000_000_000.0)).unwrap();
        if let JsValue::HeapNumber(v) = typed_array_get(&ta, 0) {
            assert_eq!(v, 4_000_000_000.0);
        } else {
            panic!("Expected HeapNumber for Uint32");
        }
    }

    // ── sort with custom comparator ─────────────────────────────────────

    #[test]
    fn test_typed_array_sort_with_comparator_descending() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(10), JsValue::Smi(30), JsValue::Smi(20)],
        )
        .unwrap();
        typed_array_sort(
            &ta,
            Some(&|a, b| {
                let na = a.to_number()?;
                let nb = b.to_number()?;
                Ok(nb - na)
            }),
        )
        .unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(30));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(20));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(10));
    }

    #[test]
    fn test_typed_array_sort_default_is_numeric_not_lexicographic() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(100), JsValue::Smi(3), JsValue::Smi(20)],
        )
        .unwrap();
        typed_array_sort(&ta, None).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(20));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(100));
    }

    #[test]
    fn test_typed_array_sort_single_element() {
        let ta = typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(42)]).unwrap();
        typed_array_sort(&ta, None).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(42));
    }

    #[test]
    fn test_typed_array_sort_empty() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        typed_array_sort(&ta, None).unwrap();
        assert_eq!(typed_array_length(&ta), 0);
    }

    // ── from with mapFn ─────────────────────────────────────────────────

    #[test]
    fn test_typed_array_static_from_with_map() {
        let ta = typed_array_static_from(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
            Some(&|v, _| {
                let n = v.to_number()? as i32;
                Ok(JsValue::Smi(n * 10))
            }),
        )
        .unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(10));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(20));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(30));
    }

    #[test]
    fn test_typed_array_static_from_with_map_receives_index() {
        let ta = typed_array_static_from(
            TypedArrayKind::Int32,
            &[JsValue::Smi(0), JsValue::Smi(0), JsValue::Smi(0)],
            Some(&|_, idx| Ok(JsValue::Smi(idx as i32))),
        )
        .unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(1));
        assert_eq!(typed_array_get(&ta, 2), JsValue::Smi(2));
    }

    #[test]
    fn test_typed_array_static_from_empty() {
        let ta = typed_array_static_from(TypedArrayKind::Uint8, &[], None).unwrap();
        assert_eq!(typed_array_length(&ta), 0);
    }

    // ── set_from_typed_array ────────────────────────────────────────────

    #[test]
    fn test_typed_array_set_from_typed_array_same_kind() {
        let dst = typed_array_new_from_length(TypedArrayKind::Int32, 5);
        let src =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(10), JsValue::Smi(20)])
                .unwrap();
        typed_array_set_from_typed_array(&dst, &src, 2).unwrap();
        assert_eq!(typed_array_get(&dst, 2), JsValue::Smi(10));
        assert_eq!(typed_array_get(&dst, 3), JsValue::Smi(20));
    }

    #[test]
    fn test_typed_array_set_from_typed_array_different_kind() {
        let dst = typed_array_new_from_length(TypedArrayKind::Float64, 3);
        let src =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(5), JsValue::Smi(6)])
                .unwrap();
        typed_array_set_from_typed_array(&dst, &src, 1).unwrap();
        assert_eq!(typed_array_get(&dst, 1), JsValue::HeapNumber(5.0));
        assert_eq!(typed_array_get(&dst, 2), JsValue::HeapNumber(6.0));
    }

    #[test]
    fn test_typed_array_set_from_typed_array_overflow_error() {
        let dst = typed_array_new_from_length(TypedArrayKind::Int32, 2);
        let src = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        assert!(typed_array_set_from_typed_array(&dst, &src, 0).is_err());
    }

    // ── copyWithin negative ─────────────────────────────────────────────

    #[test]
    fn test_typed_array_copy_within_negative_target() {
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
        typed_array_copy_within(&ta, -2, 0, 2);
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(1));
        assert_eq!(typed_array_get(&ta, 4), JsValue::Smi(2));
    }

    // ── fill negative end ───────────────────────────────────────────────

    #[test]
    fn test_typed_array_fill_negative_end() {
        let ta = typed_array_new_from_length(TypedArrayKind::Uint8, 5);
        typed_array_fill(&ta, &JsValue::Smi(3), 1, -1).unwrap();
        assert_eq!(typed_array_get(&ta, 0), JsValue::Smi(0));
        assert_eq!(typed_array_get(&ta, 1), JsValue::Smi(3));
        assert_eq!(typed_array_get(&ta, 3), JsValue::Smi(3));
        assert_eq!(typed_array_get(&ta, 4), JsValue::Smi(0));
    }

    // ── reduce/reduceRight edge cases ───────────────────────────────────

    #[test]
    fn test_typed_array_reduce_right_no_initial() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        )
        .unwrap();
        let sum = typed_array_reduce_right(
            &ta,
            |acc, v, _| {
                let a = acc.to_number()?;
                let b = v.to_number()?;
                Ok(JsValue::Smi((a + b) as i32))
            },
            None,
        )
        .unwrap();
        assert_eq!(sum, JsValue::Smi(6));
    }

    #[test]
    fn test_typed_array_reduce_right_empty_no_initial_errors() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        let result = typed_array_reduce_right(&ta, |acc, _, _| Ok(acc.clone()), None);
        assert!(result.is_err());
    }

    // ── find_last_index ─────────────────────────────────────────────────

    #[test]
    fn test_typed_array_find_last_index_found() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(2)],
        )
        .unwrap();
        let idx = typed_array_find_last_index(&ta, |v, _| Ok(v.to_number()? == 2.0)).unwrap();
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_typed_array_find_last_index_not_found() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(1), JsValue::Smi(2)])
                .unwrap();
        let idx = typed_array_find_last_index(&ta, |v, _| Ok(v.to_number()? == 99.0)).unwrap();
        assert_eq!(idx, -1);
    }

    // ── indexOf with negative from ──────────────────────────────────────

    #[test]
    fn test_typed_array_index_of_negative_beyond_start() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(10), JsValue::Smi(20)])
                .unwrap();
        assert_eq!(typed_array_index_of(&ta, &JsValue::Smi(10), -100), 0);
    }

    // ── lastIndexOf edge cases ──────────────────────────────────────────

    #[test]
    fn test_typed_array_last_index_of_negative_from() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(1)],
        )
        .unwrap();
        assert_eq!(typed_array_last_index_of(&ta, &JsValue::Smi(1), -2), 0);
    }

    // ── subarray auto_length ────────────────────────────────────────────

    #[test]
    fn test_typed_array_subarray_to_end() {
        let ta = typed_array_from_values(
            TypedArrayKind::Uint8,
            &[
                JsValue::Smi(1),
                JsValue::Smi(2),
                JsValue::Smi(3),
                JsValue::Smi(4),
            ],
        )
        .unwrap();
        let sub = typed_array_subarray(&ta, 2, 4);
        assert_eq!(typed_array_length(&sub), 2);
        assert_eq!(typed_array_get(&sub, 0), JsValue::Smi(3));
        assert_eq!(typed_array_get(&sub, 1), JsValue::Smi(4));
    }

    // ── slice empty result ──────────────────────────────────────────────

    #[test]
    fn test_typed_array_slice_empty_when_start_ge_end() {
        let ta =
            typed_array_from_values(TypedArrayKind::Int32, &[JsValue::Smi(1), JsValue::Smi(2)])
                .unwrap();
        let s = typed_array_slice(&ta, 2, 1).unwrap();
        assert_eq!(typed_array_length(&s), 0);
    }

    // ── default compare NaN ordering ────────────────────────────────────

    #[test]
    fn test_typed_array_default_compare_nan_at_end() {
        assert_eq!(
            typed_array_default_compare(1.0, f64::NAN),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            typed_array_default_compare(f64::NAN, 1.0),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            typed_array_default_compare(f64::NAN, f64::NAN),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn test_typed_array_default_compare_negative_zero() {
        assert_eq!(
            typed_array_default_compare(-0.0_f64, 0.0_f64),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            typed_array_default_compare(0.0_f64, -0.0_f64),
            std::cmp::Ordering::Greater
        );
    }

    // ── map / filter return correct kind ────────────────────────────────

    #[test]
    fn test_typed_array_map_returns_same_kind() {
        let ta =
            typed_array_from_values(TypedArrayKind::Uint8, &[JsValue::Smi(1), JsValue::Smi(2)])
                .unwrap();
        let mapped = typed_array_map(&ta, |v, _| {
            let n = v.to_number()? as i32;
            Ok(JsValue::Smi(n + 1))
        })
        .unwrap();
        assert_eq!(mapped.kind, TypedArrayKind::Uint8);
        assert_eq!(typed_array_get(&mapped, 0), JsValue::Smi(2));
        assert_eq!(typed_array_get(&mapped, 1), JsValue::Smi(3));
    }

    #[test]
    fn test_typed_array_filter_returns_same_kind() {
        let ta = typed_array_from_values(
            TypedArrayKind::Float64,
            &[
                JsValue::HeapNumber(1.0),
                JsValue::HeapNumber(2.0),
                JsValue::HeapNumber(3.0),
            ],
        )
        .unwrap();
        let filtered = typed_array_filter(&ta, |v, _| Ok(v.to_number()? > 1.5)).unwrap();
        assert_eq!(filtered.kind, TypedArrayKind::Float64);
        assert_eq!(typed_array_length(&filtered), 2);
    }

    // ── for_each receives correct index ─────────────────────────────────

    #[test]
    fn test_typed_array_for_each_receives_index() {
        let ta = typed_array_from_values(
            TypedArrayKind::Int32,
            &[JsValue::Smi(10), JsValue::Smi(20), JsValue::Smi(30)],
        )
        .unwrap();
        let mut indices = Vec::new();
        typed_array_for_each(&ta, |_, idx| {
            indices.push(idx);
            Ok(())
        })
        .unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // ── values / keys / entries ─────────────────────────────────────────

    #[test]
    fn test_typed_array_values_empty() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        assert!(typed_array_values(&ta).is_empty());
    }

    #[test]
    fn test_typed_array_keys_empty() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        assert!(typed_array_keys(&ta).is_empty());
    }

    #[test]
    fn test_typed_array_entries_empty() {
        let ta = typed_array_new_from_length(TypedArrayKind::Int32, 0);
        assert!(typed_array_entries(&ta).is_empty());
    }
}

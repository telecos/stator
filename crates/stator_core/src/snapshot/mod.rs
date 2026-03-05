//! Startup snapshot — binary serialization and deserialization of heap state.
//!
//! A *startup snapshot* captures the engine's initial global environment (all
//! entries in the global `HashMap<String, JsValue>`) into a compact binary blob
//! that can be persisted to disk and restored later to skip the bootstrapping
//! phase on subsequent engine starts.
//!
//! # Binary format
//!
//! ```text
//! Header (8 bytes):
//!   magic   : [u8; 4]  = b"STSS"  (Stator Startup Snapshot)
//!   version : u32 LE   = SNAPSHOT_VERSION
//!
//! Globals section:
//!   count   : u32 LE
//!   [entry] * count:
//!     key   : str32    (u32-length-prefixed UTF-8)
//!     value : JsValue  (tagged encoding described below)
//!
//! JsValue tag byte:
//!   0x00  Undefined
//!   0x01  Null
//!   0x02  Boolean      : u8   (0 = false, 1 = true)
//!   0x03  Smi          : i32 LE
//!   0x04  HeapNumber   : f64 LE (IEEE 754)
//!   0x05  String       : str32
//!   0x06  Symbol       : u64 LE
//!   0x07  BigInt       : i128 LE (16 bytes)
//!   0x08  Function     : BytecodeArray (see below)
//!   0x09  Array        : u32 LE count, then count × JsValue
//!   0x0A  PlainObject  : u32 LE count, then count × (str32, JsValue)
//!   0x0B  Error        : u8 ErrorKind, str32 message
//!   0x0C  NativeFunction placeholder : str32 name
//!   -- Object, Generator, Iterator serialize as Undefined (0x00) --
//!
//! BytecodeArray encoding:
//!   bytecodes       : bytes32  (u32-length-prefixed bytes)
//!   constant_pool   : u32 LE count, then count × ConstantPoolEntry
//!   frame_size      : u32 LE
//!   parameter_count : u32 LE
//!   source_positions: u32 LE count, then count × (u32, u32, u32)
//!   feedback_meta   : u32 LE count, then count × u8 FeedbackSlotKind
//!   handler_table   : u32 LE count, then count × (u32, u32, u32, u8)
//!   is_generator    : u8  (0 or 1)
//!
//! ConstantPoolEntry tag byte:
//!   0x00  Number    : f64 LE
//!   0x01  String    : str32
//!   0x02  Boolean   : u8
//!   0x03  Null
//!   0x04  Undefined
//!   0x05  Function  : BytecodeArray (nested)
//! ```
//!
//! # Limitations
//!
//! - `JsValue::Object` (raw GC pointer), `Generator`, and `Iterator` are
//!   **not** serializable; they round-trip as `Undefined`.
//! - `NativeFunction` values are stored as a named placeholder
//!   (`NativeFunctionPlaceholder`).  On deserialization the placeholder is
//!   restored as a [`JsValue::NativeFunction`] that returns `Undefined` for
//!   every call.  Embedders that need to restore real host callbacks should
//!   re-install them by name after calling [`deserialize_globals`].
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//! use stator_core::objects::value::JsValue;
//! use stator_core::snapshot::{StartupSnapshot, serialize_globals, deserialize_globals};
//!
//! let mut globals: HashMap<String, JsValue> = HashMap::new();
//! globals.insert("answer".to_string(), JsValue::Smi(42));
//! globals.insert("greeting".to_string(), JsValue::String("hello".to_string()));
//!
//! let snapshot = serialize_globals(&globals);
//! let restored = deserialize_globals(snapshot.as_bytes()).expect("valid snapshot");
//!
//! assert_eq!(restored.get("answer"), Some(&JsValue::Smi(42)));
//! assert_eq!(restored.get("greeting"), Some(&JsValue::String("hello".to_string())));
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::builtins::error::{ErrorKind, JsError};
use crate::bytecode::bytecode_array::{
    BytecodeArray, ConstantPoolEntry, HandlerTableEntry, SourcePosition,
};
use crate::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes that identify a Stator Startup Snapshot blob.
const MAGIC: [u8; 4] = *b"STSS";

/// Format version; increment when the layout changes in an incompatible way.
const SNAPSHOT_VERSION: u32 = 1;

// JsValue tag bytes
const TAG_UNDEFINED: u8 = 0x00;
const TAG_NULL: u8 = 0x01;
const TAG_BOOLEAN: u8 = 0x02;
const TAG_SMI: u8 = 0x03;
const TAG_HEAP_NUMBER: u8 = 0x04;
const TAG_STRING: u8 = 0x05;
const TAG_SYMBOL: u8 = 0x06;
const TAG_BIGINT: u8 = 0x07;
const TAG_FUNCTION: u8 = 0x08;
const TAG_ARRAY: u8 = 0x09;
const TAG_PLAIN_OBJECT: u8 = 0x0A;
const TAG_ERROR: u8 = 0x0B;
const TAG_NATIVE_FUNCTION: u8 = 0x0C;

// ConstantPoolEntry tag bytes
const CPE_NUMBER: u8 = 0x00;
const CPE_STRING: u8 = 0x01;
const CPE_BOOLEAN: u8 = 0x02;
const CPE_NULL: u8 = 0x03;
const CPE_UNDEFINED: u8 = 0x04;
const CPE_FUNCTION: u8 = 0x05;

// ─────────────────────────────────────────────────────────────────────────────
// StartupSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// An opaque binary blob produced by [`serialize_globals`].
///
/// The blob can be stored to disk and later passed to [`deserialize_globals`]
/// to restore the engine's initial global environment without re-running the
/// bootstrap script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupSnapshot {
    data: Vec<u8>,
}

impl StartupSnapshot {
    /// Create a `StartupSnapshot` from raw bytes.
    ///
    /// This is the inverse of [`StartupSnapshot::as_bytes`]; it does **not**
    /// validate the content — call [`deserialize_globals`] to perform
    /// validation and extraction.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Return the raw byte representation of this snapshot.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the snapshot and return its underlying `Vec<u8>`.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize the engine's global variable table into a [`StartupSnapshot`].
///
/// Every entry in `globals` is encoded using the binary format described in the
/// [module documentation](self).  Values that cannot be serialized (raw
/// `Object` pointers, live `Generator`s, and `Iterator`s) are stored as
/// `Undefined` placeholders.  `NativeFunction` values are stored as named
/// placeholders that restore to no-op stubs on deserialization.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use stator_core::objects::value::JsValue;
/// use stator_core::snapshot::serialize_globals;
///
/// let mut globals = HashMap::new();
/// globals.insert("x".to_string(), JsValue::Smi(7));
/// let snap = serialize_globals(&globals);
/// assert!(!snap.as_bytes().is_empty());
/// ```
pub fn serialize_globals(globals: &HashMap<String, JsValue>) -> StartupSnapshot {
    let mut buf = Vec::new();
    write_magic_header(&mut buf);
    write_u32(&mut buf, globals.len() as u32);
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = globals.keys().collect();
    keys.sort();
    for key in keys {
        let value = &globals[key];
        write_str32(&mut buf, key);
        write_jsvalue(&mut buf, value);
    }
    StartupSnapshot { data: buf }
}

/// Deserialize a [`StartupSnapshot`] into a globals map.
///
/// # Errors
///
/// Returns [`StatorError::Internal`] if the blob is malformed, the magic
/// bytes do not match, or the version is unsupported.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use stator_core::objects::value::JsValue;
/// use stator_core::snapshot::{serialize_globals, deserialize_globals};
///
/// let mut globals = HashMap::new();
/// globals.insert("pi".to_string(), JsValue::HeapNumber(3.14));
/// let snap = serialize_globals(&globals);
/// let restored = deserialize_globals(snap.as_bytes()).unwrap();
/// assert_eq!(restored.get("pi"), Some(&JsValue::HeapNumber(3.14)));
/// ```
pub fn deserialize_globals(bytes: &[u8]) -> StatorResult<HashMap<String, JsValue>> {
    let mut cursor = 0usize;
    read_magic_header(bytes, &mut cursor)?;
    let count = read_u32(bytes, &mut cursor)?;
    let mut globals = HashMap::with_capacity(count as usize);
    for _ in 0..count {
        let key = read_str32(bytes, &mut cursor)?;
        let value = read_jsvalue(bytes, &mut cursor)?;
        globals.insert(key, value);
    }
    Ok(globals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Write helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_magic_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(&MAGIC);
    write_u32(buf, SNAPSHOT_VERSION);
}

fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_i128(buf: &mut Vec<u8>, v: i128) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Write a `u32`-length-prefixed UTF-8 string.
fn write_str32(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    write_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

fn write_jsvalue(buf: &mut Vec<u8>, value: &JsValue) {
    match value {
        JsValue::Undefined => write_u8(buf, TAG_UNDEFINED),
        JsValue::Null => write_u8(buf, TAG_NULL),
        JsValue::Boolean(b) => {
            write_u8(buf, TAG_BOOLEAN);
            write_u8(buf, if *b { 1 } else { 0 });
        }
        JsValue::Smi(n) => {
            write_u8(buf, TAG_SMI);
            write_i32(buf, *n);
        }
        JsValue::HeapNumber(n) => {
            write_u8(buf, TAG_HEAP_NUMBER);
            write_f64(buf, *n);
        }
        JsValue::String(s) => {
            write_u8(buf, TAG_STRING);
            write_str32(buf, s);
        }
        JsValue::Symbol(id) => {
            write_u8(buf, TAG_SYMBOL);
            write_u64(buf, *id);
        }
        JsValue::BigInt(n) => {
            write_u8(buf, TAG_BIGINT);
            write_i128(buf, *n);
        }
        JsValue::Function(rc) => {
            write_u8(buf, TAG_FUNCTION);
            write_bytecode_array(buf, rc);
        }
        JsValue::Array(items) => {
            write_u8(buf, TAG_ARRAY);
            write_u32(buf, items.len() as u32);
            for item in items.iter() {
                write_jsvalue(buf, item);
            }
        }
        JsValue::PlainObject(map) => {
            let borrow = map.borrow();
            write_u8(buf, TAG_PLAIN_OBJECT);
            write_u32(buf, borrow.len() as u32);
            // Sort for determinism.
            let mut entries: Vec<(&String, &JsValue)> = borrow.iter().collect();
            entries.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in entries {
                write_str32(buf, k);
                write_jsvalue(buf, v);
            }
        }
        JsValue::Error(rc) => {
            write_u8(buf, TAG_ERROR);
            write_u8(buf, error_kind_to_byte(rc.kind));
            write_str32(buf, &rc.message);
        }
        JsValue::NativeFunction(_) => {
            // Native closures cannot be serialized; store a placeholder.
            write_u8(buf, TAG_NATIVE_FUNCTION);
            write_str32(buf, "<native>");
        }
        // Object (raw pointer), Generator, Iterator, and Context cannot be serialized.
        JsValue::Object(_) | JsValue::Generator(_) | JsValue::Iterator(_) | JsValue::Context(_) => {
            write_u8(buf, TAG_UNDEFINED);
        }
    }
}

fn write_bytecode_array(buf: &mut Vec<u8>, ba: &BytecodeArray) {
    // bytecodes
    let bc = ba.bytecodes();
    write_u32(buf, bc.len() as u32);
    buf.extend_from_slice(bc);

    // constant pool
    let pool = ba.constant_pool();
    write_u32(buf, pool.len() as u32);
    for entry in pool {
        write_constant_pool_entry(buf, entry);
    }

    // frame_size, parameter_count
    write_u32(buf, ba.frame_size());
    write_u32(buf, ba.parameter_count());

    // source_positions
    let sp = ba.source_positions();
    write_u32(buf, sp.len() as u32);
    for pos in sp {
        write_u32(buf, pos.bytecode_offset);
        write_u32(buf, pos.line);
        write_u32(buf, pos.column);
    }

    // feedback_metadata slot kinds
    let kinds = ba.feedback_metadata().slot_kinds();
    write_u32(buf, kinds.len() as u32);
    for kind in kinds {
        write_u8(buf, feedback_slot_kind_to_byte(*kind));
    }

    // handler_table
    let ht = ba.handler_table();
    write_u32(buf, ht.len() as u32);
    for entry in ht {
        write_u32(buf, entry.try_start);
        write_u32(buf, entry.try_end);
        write_u32(buf, entry.handler);
        write_u8(buf, if entry.is_finally { 1 } else { 0 });
    }

    // is_generator flag
    write_u8(buf, if ba.is_generator() { 1 } else { 0 });
}

fn write_constant_pool_entry(buf: &mut Vec<u8>, entry: &ConstantPoolEntry) {
    match entry {
        ConstantPoolEntry::Number(n) => {
            write_u8(buf, CPE_NUMBER);
            write_f64(buf, *n);
        }
        ConstantPoolEntry::String(s) => {
            write_u8(buf, CPE_STRING);
            write_str32(buf, s);
        }
        ConstantPoolEntry::Boolean(b) => {
            write_u8(buf, CPE_BOOLEAN);
            write_u8(buf, if *b { 1 } else { 0 });
        }
        ConstantPoolEntry::Null => write_u8(buf, CPE_NULL),
        ConstantPoolEntry::Undefined => write_u8(buf, CPE_UNDEFINED),
        ConstantPoolEntry::Function(ba) => {
            write_u8(buf, CPE_FUNCTION);
            write_bytecode_array(buf, ba);
        }
    }
}

fn error_kind_to_byte(kind: ErrorKind) -> u8 {
    match kind {
        ErrorKind::Error => 0,
        ErrorKind::TypeError => 1,
        ErrorKind::RangeError => 2,
        ErrorKind::ReferenceError => 3,
        ErrorKind::SyntaxError => 4,
        ErrorKind::URIError => 5,
        ErrorKind::EvalError => 6,
        ErrorKind::AggregateError => 7,
    }
}

fn feedback_slot_kind_to_byte(kind: FeedbackSlotKind) -> u8 {
    match kind {
        FeedbackSlotKind::Call => 0,
        FeedbackSlotKind::LoadProperty => 1,
        FeedbackSlotKind::StoreProperty => 2,
        FeedbackSlotKind::KeyedLoadProperty => 3,
        FeedbackSlotKind::KeyedStoreProperty => 4,
        FeedbackSlotKind::BinaryOp => 5,
        FeedbackSlotKind::Compare => 6,
        FeedbackSlotKind::ForIn => 7,
        FeedbackSlotKind::TypeOf => 8,
        FeedbackSlotKind::CreateClosure => 9,
        FeedbackSlotKind::LoadGlobal => 10,
        FeedbackSlotKind::StoreGlobal => 11,
        FeedbackSlotKind::InstanceOf => 12,
        FeedbackSlotKind::BinaryOpInc => 13,
        FeedbackSlotKind::UnaryOp => 14,
        FeedbackSlotKind::Literal => 15,
        FeedbackSlotKind::DefineAccessor => 16,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Read helpers
// ─────────────────────────────────────────────────────────────────────────────

fn need(bytes: &[u8], cursor: usize, n: usize) -> StatorResult<()> {
    if cursor + n > bytes.len() {
        Err(StatorError::Internal(format!(
            "snapshot: unexpected end of data (need {n} bytes at offset {cursor}, have {})",
            bytes.len()
        )))
    } else {
        Ok(())
    }
}

fn read_magic_header(bytes: &[u8], cursor: &mut usize) -> StatorResult<()> {
    need(bytes, *cursor, 8)?;
    let magic = &bytes[*cursor..*cursor + 4];
    if magic != MAGIC {
        return Err(StatorError::Internal(format!(
            "snapshot: invalid magic bytes {magic:?}"
        )));
    }
    *cursor += 4;
    let version = read_u32(bytes, cursor)?;
    if version != SNAPSHOT_VERSION {
        return Err(StatorError::Internal(format!(
            "snapshot: unsupported version {version} (expected {SNAPSHOT_VERSION})"
        )));
    }
    Ok(())
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> StatorResult<u8> {
    need(bytes, *cursor, 1)?;
    let v = bytes[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> StatorResult<u32> {
    need(bytes, *cursor, 4)?;
    let v = u32::from_le_bytes(
        bytes[*cursor..*cursor + 4]
            .try_into()
            .expect("bounds already checked"),
    );
    *cursor += 4;
    Ok(v)
}

fn read_i32(bytes: &[u8], cursor: &mut usize) -> StatorResult<i32> {
    need(bytes, *cursor, 4)?;
    let v = i32::from_le_bytes(
        bytes[*cursor..*cursor + 4]
            .try_into()
            .expect("bounds already checked"),
    );
    *cursor += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> StatorResult<u64> {
    need(bytes, *cursor, 8)?;
    let v = u64::from_le_bytes(
        bytes[*cursor..*cursor + 8]
            .try_into()
            .expect("bounds already checked"),
    );
    *cursor += 8;
    Ok(v)
}

fn read_i128(bytes: &[u8], cursor: &mut usize) -> StatorResult<i128> {
    need(bytes, *cursor, 16)?;
    let v = i128::from_le_bytes(
        bytes[*cursor..*cursor + 16]
            .try_into()
            .expect("bounds already checked"),
    );
    *cursor += 16;
    Ok(v)
}

fn read_f64(bytes: &[u8], cursor: &mut usize) -> StatorResult<f64> {
    need(bytes, *cursor, 8)?;
    let v = f64::from_le_bytes(
        bytes[*cursor..*cursor + 8]
            .try_into()
            .expect("bounds already checked"),
    );
    *cursor += 8;
    Ok(v)
}

/// Read a `u32`-length-prefixed UTF-8 string.
fn read_str32(bytes: &[u8], cursor: &mut usize) -> StatorResult<String> {
    let len = read_u32(bytes, cursor)? as usize;
    need(bytes, *cursor, len)?;
    let s = std::str::from_utf8(&bytes[*cursor..*cursor + len])
        .map_err(|e| StatorError::Internal(format!("snapshot: invalid UTF-8 string: {e}")))?;
    let owned = s.to_owned();
    *cursor += len;
    Ok(owned)
}

fn read_jsvalue(bytes: &[u8], cursor: &mut usize) -> StatorResult<JsValue> {
    let tag = read_u8(bytes, cursor)?;
    match tag {
        TAG_UNDEFINED => Ok(JsValue::Undefined),
        TAG_NULL => Ok(JsValue::Null),
        TAG_BOOLEAN => {
            let b = read_u8(bytes, cursor)?;
            Ok(JsValue::Boolean(b != 0))
        }
        TAG_SMI => {
            let n = read_i32(bytes, cursor)?;
            Ok(JsValue::Smi(n))
        }
        TAG_HEAP_NUMBER => {
            let n = read_f64(bytes, cursor)?;
            Ok(JsValue::HeapNumber(n))
        }
        TAG_STRING => {
            let s = read_str32(bytes, cursor)?;
            Ok(JsValue::String(s))
        }
        TAG_SYMBOL => {
            let id = read_u64(bytes, cursor)?;
            Ok(JsValue::Symbol(id))
        }
        TAG_BIGINT => {
            let n = read_i128(bytes, cursor)?;
            Ok(JsValue::BigInt(n))
        }
        TAG_FUNCTION => {
            let ba = read_bytecode_array(bytes, cursor)?;
            Ok(JsValue::Function(Rc::new(ba)))
        }
        TAG_ARRAY => {
            let count = read_u32(bytes, cursor)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_jsvalue(bytes, cursor)?);
            }
            Ok(JsValue::Array(Rc::new(items)))
        }
        TAG_PLAIN_OBJECT => {
            let count = read_u32(bytes, cursor)? as usize;
            let mut map = HashMap::with_capacity(count);
            for _ in 0..count {
                let k = read_str32(bytes, cursor)?;
                let v = read_jsvalue(bytes, cursor)?;
                map.insert(k, v);
            }
            Ok(JsValue::PlainObject(Rc::new(RefCell::new(map))))
        }
        TAG_ERROR => {
            let kind_byte = read_u8(bytes, cursor)?;
            let kind = byte_to_error_kind(kind_byte)?;
            let message = read_str32(bytes, cursor)?;
            Ok(JsValue::Error(Rc::new(JsError::new(kind, message))))
        }
        TAG_NATIVE_FUNCTION => {
            // Discard the `<native>` placeholder; restore as a no-op stub.
            let _placeholder_name = read_str32(bytes, cursor)?;
            Ok(JsValue::NativeFunction(Rc::new(|_args| {
                Ok(JsValue::Undefined)
            })))
        }
        other => Err(StatorError::Internal(format!(
            "snapshot: unknown JsValue tag {other:#04x}"
        ))),
    }
}

fn read_bytecode_array(bytes: &[u8], cursor: &mut usize) -> StatorResult<BytecodeArray> {
    // bytecodes
    let bc_len = read_u32(bytes, cursor)? as usize;
    need(bytes, *cursor, bc_len)?;
    let bytecodes = bytes[*cursor..*cursor + bc_len].to_vec();
    *cursor += bc_len;

    // constant pool
    let pool_len = read_u32(bytes, cursor)? as usize;
    let mut constant_pool = Vec::with_capacity(pool_len);
    for _ in 0..pool_len {
        constant_pool.push(read_constant_pool_entry(bytes, cursor)?);
    }

    // frame_size, parameter_count
    let frame_size = read_u32(bytes, cursor)?;
    let parameter_count = read_u32(bytes, cursor)?;

    // source_positions
    let sp_len = read_u32(bytes, cursor)? as usize;
    let mut source_positions = Vec::with_capacity(sp_len);
    for _ in 0..sp_len {
        let offset = read_u32(bytes, cursor)?;
        let line = read_u32(bytes, cursor)?;
        let column = read_u32(bytes, cursor)?;
        source_positions.push(SourcePosition::new(offset, line, column));
    }

    // feedback_metadata
    let fm_len = read_u32(bytes, cursor)? as usize;
    let mut slot_kinds = Vec::with_capacity(fm_len);
    for _ in 0..fm_len {
        let b = read_u8(bytes, cursor)?;
        slot_kinds.push(byte_to_feedback_slot_kind(b)?);
    }
    let feedback_metadata = FeedbackMetadata::new(slot_kinds);

    // handler_table
    let ht_len = read_u32(bytes, cursor)? as usize;
    let mut handler_table = Vec::with_capacity(ht_len);
    for _ in 0..ht_len {
        let try_start = read_u32(bytes, cursor)?;
        let try_end = read_u32(bytes, cursor)?;
        let handler = read_u32(bytes, cursor)?;
        let is_finally = read_u8(bytes, cursor)? != 0;
        handler_table.push(HandlerTableEntry {
            try_start,
            try_end,
            handler,
            is_finally,
        });
    }

    // is_generator
    let is_generator = read_u8(bytes, cursor)? != 0;

    Ok(BytecodeArray::new(
        bytecodes,
        constant_pool,
        frame_size,
        parameter_count,
        source_positions,
        feedback_metadata,
        handler_table,
    )
    .with_generator_flag(is_generator))
}

fn read_constant_pool_entry(bytes: &[u8], cursor: &mut usize) -> StatorResult<ConstantPoolEntry> {
    let tag = read_u8(bytes, cursor)?;
    match tag {
        CPE_NUMBER => {
            let n = read_f64(bytes, cursor)?;
            Ok(ConstantPoolEntry::Number(n))
        }
        CPE_STRING => {
            let s = read_str32(bytes, cursor)?;
            Ok(ConstantPoolEntry::String(s))
        }
        CPE_BOOLEAN => {
            let b = read_u8(bytes, cursor)?;
            Ok(ConstantPoolEntry::Boolean(b != 0))
        }
        CPE_NULL => Ok(ConstantPoolEntry::Null),
        CPE_UNDEFINED => Ok(ConstantPoolEntry::Undefined),
        CPE_FUNCTION => {
            let ba = read_bytecode_array(bytes, cursor)?;
            Ok(ConstantPoolEntry::Function(Box::new(ba)))
        }
        other => Err(StatorError::Internal(format!(
            "snapshot: unknown ConstantPoolEntry tag {other:#04x}"
        ))),
    }
}

fn byte_to_error_kind(b: u8) -> StatorResult<ErrorKind> {
    match b {
        0 => Ok(ErrorKind::Error),
        1 => Ok(ErrorKind::TypeError),
        2 => Ok(ErrorKind::RangeError),
        3 => Ok(ErrorKind::ReferenceError),
        4 => Ok(ErrorKind::SyntaxError),
        5 => Ok(ErrorKind::URIError),
        6 => Ok(ErrorKind::EvalError),
        7 => Ok(ErrorKind::AggregateError),
        other => Err(StatorError::Internal(format!(
            "snapshot: unknown ErrorKind byte {other}"
        ))),
    }
}

fn byte_to_feedback_slot_kind(b: u8) -> StatorResult<FeedbackSlotKind> {
    match b {
        0 => Ok(FeedbackSlotKind::Call),
        1 => Ok(FeedbackSlotKind::LoadProperty),
        2 => Ok(FeedbackSlotKind::StoreProperty),
        3 => Ok(FeedbackSlotKind::KeyedLoadProperty),
        4 => Ok(FeedbackSlotKind::KeyedStoreProperty),
        5 => Ok(FeedbackSlotKind::BinaryOp),
        6 => Ok(FeedbackSlotKind::Compare),
        7 => Ok(FeedbackSlotKind::ForIn),
        8 => Ok(FeedbackSlotKind::TypeOf),
        9 => Ok(FeedbackSlotKind::CreateClosure),
        10 => Ok(FeedbackSlotKind::LoadGlobal),
        11 => Ok(FeedbackSlotKind::StoreGlobal),
        12 => Ok(FeedbackSlotKind::InstanceOf),
        13 => Ok(FeedbackSlotKind::BinaryOpInc),
        14 => Ok(FeedbackSlotKind::UnaryOp),
        15 => Ok(FeedbackSlotKind::Literal),
        16 => Ok(FeedbackSlotKind::DefineAccessor),
        other => Err(StatorError::Internal(format!(
            "snapshot: unknown FeedbackSlotKind byte {other}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
    use crate::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind};
    use crate::objects::value::JsValue;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn round_trip(globals: HashMap<String, JsValue>) -> HashMap<String, JsValue> {
        let snap = serialize_globals(&globals);
        deserialize_globals(snap.as_bytes()).expect("deserialization should succeed")
    }

    // ── primitive values ──────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_undefined() {
        let mut g = HashMap::new();
        g.insert("u".to_string(), JsValue::Undefined);
        assert_eq!(round_trip(g).get("u"), Some(&JsValue::Undefined));
    }

    #[test]
    fn test_round_trip_null() {
        let mut g = HashMap::new();
        g.insert("n".to_string(), JsValue::Null);
        assert_eq!(round_trip(g).get("n"), Some(&JsValue::Null));
    }

    #[test]
    fn test_round_trip_boolean() {
        let mut g = HashMap::new();
        g.insert("t".to_string(), JsValue::Boolean(true));
        g.insert("f".to_string(), JsValue::Boolean(false));
        let r = round_trip(g);
        assert_eq!(r.get("t"), Some(&JsValue::Boolean(true)));
        assert_eq!(r.get("f"), Some(&JsValue::Boolean(false)));
    }

    #[test]
    fn test_round_trip_smi() {
        let mut g = HashMap::new();
        g.insert("pos".to_string(), JsValue::Smi(42));
        g.insert("neg".to_string(), JsValue::Smi(-7));
        g.insert("zero".to_string(), JsValue::Smi(0));
        let r = round_trip(g);
        assert_eq!(r.get("pos"), Some(&JsValue::Smi(42)));
        assert_eq!(r.get("neg"), Some(&JsValue::Smi(-7)));
        assert_eq!(r.get("zero"), Some(&JsValue::Smi(0)));
    }

    #[test]
    fn test_round_trip_heap_number() {
        let mut g = HashMap::new();
        g.insert("pi".to_string(), JsValue::HeapNumber(3.141_592_653_589_793));
        g.insert("nan".to_string(), JsValue::HeapNumber(f64::NAN));
        g.insert("inf".to_string(), JsValue::HeapNumber(f64::INFINITY));
        g.insert(
            "neg_inf".to_string(),
            JsValue::HeapNumber(f64::NEG_INFINITY),
        );
        let r = round_trip(g);
        assert_eq!(
            r.get("pi"),
            Some(&JsValue::HeapNumber(3.141_592_653_589_793))
        );
        // NaN != NaN by IEEE 754; check bit pattern instead.
        if let Some(JsValue::HeapNumber(v)) = r.get("nan") {
            assert!(v.is_nan());
        } else {
            panic!("expected HeapNumber(NaN)");
        }
        assert_eq!(r.get("inf"), Some(&JsValue::HeapNumber(f64::INFINITY)));
        assert_eq!(
            r.get("neg_inf"),
            Some(&JsValue::HeapNumber(f64::NEG_INFINITY))
        );
    }

    #[test]
    fn test_round_trip_string() {
        let mut g = HashMap::new();
        g.insert(
            "s".to_string(),
            JsValue::String("hello, world 🌍".to_string()),
        );
        g.insert("empty".to_string(), JsValue::String(String::new()));
        let r = round_trip(g);
        assert_eq!(
            r.get("s"),
            Some(&JsValue::String("hello, world 🌍".to_string()))
        );
        assert_eq!(r.get("empty"), Some(&JsValue::String(String::new())));
    }

    #[test]
    fn test_round_trip_symbol() {
        let mut g = HashMap::new();
        g.insert("sym".to_string(), JsValue::Symbol(0xDEAD_BEEF_CAFE_1234));
        let r = round_trip(g);
        assert_eq!(r.get("sym"), Some(&JsValue::Symbol(0xDEAD_BEEF_CAFE_1234)));
    }

    #[test]
    fn test_round_trip_bigint() {
        let big = i128::MAX;
        let mut g = HashMap::new();
        g.insert("big".to_string(), JsValue::BigInt(big));
        g.insert("neg".to_string(), JsValue::BigInt(-1_i128));
        let r = round_trip(g);
        assert_eq!(r.get("big"), Some(&JsValue::BigInt(big)));
        assert_eq!(r.get("neg"), Some(&JsValue::BigInt(-1)));
    }

    // ── composite values ──────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_array() {
        let items = vec![
            JsValue::Smi(1),
            JsValue::String("two".to_string()),
            JsValue::Null,
        ];
        let mut g = HashMap::new();
        g.insert("arr".to_string(), JsValue::Array(Rc::new(items.clone())));
        let r = round_trip(g);
        if let Some(JsValue::Array(restored)) = r.get("arr") {
            assert_eq!(restored.as_ref(), &items);
        } else {
            panic!("expected Array");
        }
    }

    #[test]
    fn test_round_trip_plain_object() {
        let mut map = HashMap::new();
        map.insert("x".to_string(), JsValue::Smi(10));
        map.insert("y".to_string(), JsValue::Boolean(true));
        let mut g = HashMap::new();
        g.insert(
            "obj".to_string(),
            JsValue::PlainObject(Rc::new(RefCell::new(map.clone()))),
        );
        let r = round_trip(g);
        if let Some(JsValue::PlainObject(restored)) = r.get("obj") {
            assert_eq!(*restored.borrow(), map);
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_round_trip_error() {
        use crate::builtins::error::ErrorKind;
        let mut g = HashMap::new();
        g.insert(
            "err".to_string(),
            JsValue::Error(Rc::new(JsError::new(
                ErrorKind::TypeError,
                "not a function".to_string(),
            ))),
        );
        let r = round_trip(g);
        if let Some(JsValue::Error(e)) = r.get("err") {
            assert_eq!(e.kind, ErrorKind::TypeError);
            assert_eq!(e.message, "not a function");
        } else {
            panic!("expected Error");
        }
    }

    #[test]
    fn test_round_trip_native_function_becomes_stub() {
        let mut g = HashMap::new();
        g.insert(
            "nf".to_string(),
            JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Smi(99)))),
        );
        let r = round_trip(g);
        // NativeFunction should restore as a no-op stub (returns Undefined).
        if let Some(JsValue::NativeFunction(f)) = r.get("nf") {
            let result = f(vec![]).expect("stub should not error");
            assert_eq!(result, JsValue::Undefined);
        } else {
            panic!("expected NativeFunction stub");
        }
    }

    // ── function (BytecodeArray) ───────────────────────────────────────────────

    #[test]
    fn test_round_trip_function() {
        let pool = vec![
            ConstantPoolEntry::Number(1.5),
            ConstantPoolEntry::String("hello".to_string()),
            ConstantPoolEntry::Boolean(true),
            ConstantPoolEntry::Null,
            ConstantPoolEntry::Undefined,
        ];
        let feedback = FeedbackMetadata::new(vec![
            FeedbackSlotKind::Call,
            FeedbackSlotKind::BinaryOp,
            FeedbackSlotKind::LoadGlobal,
        ]);
        let source_positions = vec![SourcePosition::new(0, 1, 1), SourcePosition::new(4, 2, 5)];
        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 10,
            handler: 12,
            is_finally: false,
        }];
        let ba = BytecodeArray::new(
            vec![0x01, 0x02, 0x03],
            pool.clone(),
            3,
            2,
            source_positions,
            feedback,
            handler_table,
        );
        let mut g = HashMap::new();
        g.insert("fn".to_string(), JsValue::Function(Rc::new(ba.clone())));
        let r = round_trip(g);
        if let Some(JsValue::Function(restored)) = r.get("fn") {
            assert_eq!(restored.as_ref(), &ba);
        } else {
            panic!("expected Function");
        }
    }

    #[test]
    fn test_round_trip_generator_function() {
        let ba = BytecodeArray::new(
            vec![0xAB],
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
        .with_generator_flag(true);
        let mut g = HashMap::new();
        g.insert("gen".to_string(), JsValue::Function(Rc::new(ba.clone())));
        let r = round_trip(g);
        if let Some(JsValue::Function(restored)) = r.get("gen") {
            assert!(restored.is_generator(), "generator flag must be preserved");
            assert_eq!(restored.as_ref(), &ba);
        } else {
            panic!("expected Function");
        }
    }

    #[test]
    fn test_round_trip_nested_function_in_constant_pool() {
        let inner_ba = BytecodeArray::new(
            vec![0x10, 0x11],
            vec![],
            1,
            1,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let pool = vec![ConstantPoolEntry::Function(Box::new(inner_ba.clone()))];
        let outer_ba = BytecodeArray::new(
            vec![0x20],
            pool,
            2,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let mut g = HashMap::new();
        g.insert(
            "outer".to_string(),
            JsValue::Function(Rc::new(outer_ba.clone())),
        );
        let r = round_trip(g);
        if let Some(JsValue::Function(restored)) = r.get("outer") {
            assert_eq!(restored.as_ref(), &outer_ba);
            if let ConstantPoolEntry::Function(nested) = &restored.constant_pool()[0] {
                assert_eq!(nested.as_ref(), &inner_ba);
            } else {
                panic!("expected nested Function in constant pool");
            }
        } else {
            panic!("expected outer Function");
        }
    }

    // ── non-serializable values fall back to Undefined ────────────────────────

    #[test]
    fn test_object_pointer_becomes_undefined() {
        let mut g = HashMap::new();
        // We can't create a valid HeapObject pointer in a unit test; use
        // a null pointer (which is never dereferenced during serialization).
        g.insert("obj".to_string(), JsValue::Object(std::ptr::null_mut()));
        let r = round_trip(g);
        assert_eq!(r.get("obj"), Some(&JsValue::Undefined));
    }

    // ── snapshot metadata ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_globals() {
        let g: HashMap<String, JsValue> = HashMap::new();
        let snap = serialize_globals(&g);
        let restored = deserialize_globals(snap.as_bytes()).expect("should not fail");
        assert!(restored.is_empty());
    }

    #[test]
    fn test_snapshot_magic_bytes() {
        let g: HashMap<String, JsValue> = HashMap::new();
        let snap = serialize_globals(&g);
        let bytes = snap.as_bytes();
        assert_eq!(&bytes[0..4], b"STSS");
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let mut bytes = serialize_globals(&HashMap::new()).into_bytes();
        bytes[0] = b'X';
        let err = deserialize_globals(&bytes).unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_unsupported_version_rejected() {
        let mut bytes = serialize_globals(&HashMap::new()).into_bytes();
        // version is at bytes[4..8], set it to 999
        let v: u32 = 999;
        bytes[4..8].copy_from_slice(&v.to_le_bytes());
        let err = deserialize_globals(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported version"));
    }

    #[test]
    fn test_truncated_snapshot_rejected() {
        let mut g = HashMap::new();
        g.insert("x".to_string(), JsValue::Smi(1));
        let bytes = serialize_globals(&g).into_bytes();
        // Truncate at half the length.
        let truncated = &bytes[..bytes.len() / 2];
        assert!(deserialize_globals(truncated).is_err());
    }

    #[test]
    fn test_snapshot_from_bytes_round_trip() {
        let mut g = HashMap::new();
        g.insert("answer".to_string(), JsValue::Smi(42));
        let snap = serialize_globals(&g);
        let bytes = snap.as_bytes().to_vec();
        let snap2 = StartupSnapshot::from_bytes(bytes.clone());
        assert_eq!(snap2.as_bytes(), bytes.as_slice());
        assert_eq!(snap2.into_bytes(), bytes);
    }

    #[test]
    fn test_large_globals_map() {
        let mut g: HashMap<String, JsValue> = HashMap::new();
        for i in 0..100u32 {
            g.insert(format!("key{i}"), JsValue::Smi(i as i32));
        }
        let r = round_trip(g.clone());
        assert_eq!(r.len(), 100);
        for i in 0..100u32 {
            assert_eq!(r.get(&format!("key{i}")), Some(&JsValue::Smi(i as i32)));
        }
    }
}

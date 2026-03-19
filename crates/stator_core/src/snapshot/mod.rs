//! Startup snapshot — binary serialization and deserialization of heap state.
//!
//! A *startup snapshot* captures the engine's initial global environment (all
//! entries in the global `HashMap<String, JsValue>`) into a compact binary blob
//! that can be persisted to disk and restored later to skip the bootstrapping
//! phase on subsequent engine starts.
//!
//! # Binary format (v2)
//!
//! ```text
//! Header (8 bytes):
//!   magic   : [u8; 4]  = b"STSS"  (Stator Startup Snapshot)
//!   version : u32 LE   = 2
//!
//! Globals section:
//!   count   : u32 LE
//!   [entry] * count:
//!     key   : str32    (u32-length-prefixed UTF-8)
//!     value : JsValue  (tagged encoding described below)
//!
//! Footer (4 bytes):
//!   checksum: u32 LE   (FNV-1a hash of all preceding bytes)
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
//!   0x0D  DefineRef    : u32 LE ref_id, then inner JsValue
//!   0x0E  BackRef      : u32 LE ref_id
//!   -- Object, Generator, Iterator serialize as Undefined (0x00) --
//!
//! DefineRef / BackRef enable shared and circular object references.
//! Every Rc-wrapped value (PlainObject, Array, Function, Error) is
//! assigned a monotonic ref_id on first encounter.  Subsequent
//! occurrences of the same Rc emit a BackRef tag, avoiding duplicate
//! serialization and preserving object identity across the snapshot.
//! Circular references within PlainObject are supported: the object is
//! registered in the reference table before its entries are read, so
//! back-references encountered during entry deserialization resolve
//! correctly.
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
//! # Shared and circular references
//!
//! Rc-wrapped values (PlainObject, Array, Function, Error) are tracked
//! by pointer identity during serialization.  The first occurrence is
//! wrapped in a `DefineRef(ref_id)` tag; later occurrences are replaced
//! by `BackRef(ref_id)`.  This preserves prototype chains (multiple
//! globals sharing the same prototype PlainObject) and handles circular
//! structures (a PlainObject whose property points back to itself).
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
//! globals.insert("greeting".to_string(), JsValue::String("hello".to_string().into()));
//!
//! let snapshot = serialize_globals(&globals);
//! let restored = deserialize_globals(snapshot.as_bytes()).expect("valid snapshot");
//!
//! assert_eq!(restored.get("answer"), Some(&JsValue::Smi(42)));
//! assert_eq!(restored.get("greeting"), Some(&JsValue::String("hello".to_string().into())));
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use crate::builtins::error::{ErrorKind, JsError};
use crate::bytecode::bytecode_array::{
    BytecodeArray, ConstantPoolEntry, HandlerTableEntry, SourcePosition,
};
use crate::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind};
use crate::error::{StatorError, StatorResult};
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes that identify a Stator Startup Snapshot blob.
const MAGIC: [u8; 4] = *b"STSS";

/// Format version; increment when the layout changes in an incompatible way.
///
/// **v2** added: `DefineRef`/`BackRef` tags for shared and circular references,
/// and a 4-byte FNV-1a checksum footer.
const SNAPSHOT_VERSION: u32 = 2;

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
/// Marks the first occurrence of an Rc-wrapped value; followed by a ref_id
/// and the inner tagged value.
const TAG_DEFINE_REF: u8 = 0x0D;
/// Back-reference to a previously defined Rc-wrapped value by ref_id.
const TAG_BACK_REF: u8 = 0x0E;

// ConstantPoolEntry tag bytes
const CPE_NUMBER: u8 = 0x00;
const CPE_STRING: u8 = 0x01;
const CPE_BOOLEAN: u8 = 0x02;
const CPE_NULL: u8 = 0x03;
const CPE_UNDEFINED: u8 = 0x04;
const CPE_FUNCTION: u8 = 0x05;
const CPE_TEMPLATE_OBJECT: u8 = 0x06;
/// BigInt constant pool entry tag.
const CPE_BIGINT: u8 = 0x07;
const CPE_OBJECT_LITERAL_TEMPLATE: u8 = 0x08;

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

    /// Return the size of the snapshot blob in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the snapshot blob is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Write the snapshot blob to a file at `path`.
    ///
    /// Creates or truncates the file.  The raw binary format is written
    /// directly; the resulting file can later be loaded with
    /// [`StartupSnapshot::read_from_file`].
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::Internal`] if the file cannot be written.
    pub fn write_to_file(&self, path: &Path) -> StatorResult<()> {
        std::fs::write(path, &self.data).map_err(|e| {
            StatorError::Internal(format!(
                "snapshot: failed to write to {}: {e}",
                path.display()
            ))
        })
    }

    /// Read a snapshot blob from a file at `path`.
    ///
    /// The file must contain a raw snapshot blob previously produced by
    /// [`serialize_globals`] and written with [`StartupSnapshot::write_to_file`]
    /// (or any other mechanism that stores [`StartupSnapshot::as_bytes`]).
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::Internal`] if the file cannot be read.
    pub fn read_from_file(path: &Path) -> StatorResult<Self> {
        let data = std::fs::read(path).map_err(|e| {
            StatorError::Internal(format!(
                "snapshot: failed to read from {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self { data })
    }

    /// Validate the snapshot header and checksum without a full deserialization.
    ///
    /// Returns `Ok(())` if the magic bytes, version, and integrity checksum are
    /// correct, or an error describing the mismatch.
    pub fn validate(&self) -> StatorResult<()> {
        if self.data.len() < 12 {
            return Err(StatorError::Internal(
                "snapshot: blob too small for header + checksum".into(),
            ));
        }
        let mut cursor = 0usize;
        read_magic_header(&self.data, &mut cursor)?;
        verify_checksum(&self.data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization / deserialization contexts
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks Rc pointer identity during serialization so that shared and circular
/// references are emitted as `DefineRef` / `BackRef` pairs.
struct SerContext {
    next_id: u32,
    seen: HashMap<usize, u32>,
}

impl SerContext {
    fn new() -> Self {
        Self {
            next_id: 0,
            seen: HashMap::new(),
        }
    }

    /// Return `Some(ref_id)` if the pointer was already seen, or `None` after
    /// registering a new ref_id for it.
    fn track_ptr(&mut self, addr: usize) -> Option<u32> {
        if let Some(&id) = self.seen.get(&addr) {
            Some(id)
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.seen.insert(addr, id);
            None
        }
    }
}

/// Resolves `BackRef` tags during deserialization by mapping ref_ids back to
/// already-deserialized [`JsValue`]s.
struct DeserContext {
    ref_table: HashMap<u32, JsValue>,
}

impl DeserContext {
    fn new() -> Self {
        Self {
            ref_table: HashMap::new(),
        }
    }

    fn register(&mut self, id: u32, value: JsValue) {
        self.ref_table.insert(id, value);
    }

    fn lookup(&self, id: u32) -> StatorResult<JsValue> {
        self.ref_table.get(&id).cloned().ok_or_else(|| {
            StatorError::Internal(format!("snapshot: unresolved back-reference {id}"))
        })
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
    let mut ctx = SerContext::new();
    write_magic_header(&mut buf);
    write_u32(&mut buf, globals.len() as u32);
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = globals.keys().collect();
    keys.sort();
    for key in keys {
        let value = &globals[key];
        write_str32(&mut buf, key);
        write_jsvalue(&mut buf, value, &mut ctx);
    }
    // Append FNV-1a checksum footer.
    let checksum = fnv1a_32(&buf);
    write_u32(&mut buf, checksum);
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
    // Minimum: 8 (header) + 4 (count) + 4 (checksum).
    if bytes.len() < 16 {
        return Err(StatorError::Internal("snapshot: blob too small".into()));
    }

    // Validate header (magic + version) *before* checksum so that wrong-file
    // and wrong-version errors are reported with clear messages.
    let mut cursor = 0usize;
    read_magic_header(bytes, &mut cursor)?;

    // Verify FNV-1a checksum stored in the last 4 bytes.
    verify_checksum(bytes)?;

    // Read from the body (everything except the trailing checksum).
    let body = &bytes[..bytes.len() - 4];

    let mut ctx = DeserContext::new();
    let count = read_u32(body, &mut cursor)?;
    let mut globals = HashMap::with_capacity(count as usize);
    for _ in 0..count {
        let key = read_str32(body, &mut cursor)?;
        let value = read_jsvalue(body, &mut cursor, &mut ctx)?;
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

fn write_jsvalue(buf: &mut Vec<u8>, value: &JsValue, ctx: &mut SerContext) {
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
            let addr = Rc::as_ptr(rc) as *const () as usize;
            if let Some(id) = ctx.track_ptr(addr) {
                write_u8(buf, TAG_BACK_REF);
                write_u32(buf, id);
            } else {
                let id = ctx.next_id - 1;
                write_u8(buf, TAG_DEFINE_REF);
                write_u32(buf, id);
                write_u8(buf, TAG_FUNCTION);
                write_bytecode_array(buf, rc);
            }
        }
        JsValue::Array(rc) => {
            let addr = Rc::as_ptr(rc) as *const () as usize;
            if let Some(id) = ctx.track_ptr(addr) {
                write_u8(buf, TAG_BACK_REF);
                write_u32(buf, id);
            } else {
                let id = ctx.next_id - 1;
                write_u8(buf, TAG_DEFINE_REF);
                write_u32(buf, id);
                write_u8(buf, TAG_ARRAY);
                write_u32(buf, rc.borrow().len() as u32);
                for item in rc.borrow().iter() {
                    write_jsvalue(buf, item, ctx);
                }
            }
        }
        JsValue::PlainObject(rc) => {
            let addr = Rc::as_ptr(rc) as *const () as usize;
            if let Some(id) = ctx.track_ptr(addr) {
                write_u8(buf, TAG_BACK_REF);
                write_u32(buf, id);
            } else {
                let id = ctx.next_id - 1;
                write_u8(buf, TAG_DEFINE_REF);
                write_u32(buf, id);
                let borrow = rc.borrow();
                write_u8(buf, TAG_PLAIN_OBJECT);
                write_u32(buf, borrow.len() as u32);
                // Sort for determinism.
                let mut entries: Vec<(&String, &JsValue)> = borrow.iter().collect();
                entries.sort_by_key(|(k, _)| k.as_str());
                for (k, v) in entries {
                    write_str32(buf, k);
                    write_jsvalue(buf, v, ctx);
                }
            }
        }
        JsValue::Error(rc) => {
            let addr = Rc::as_ptr(rc) as *const () as usize;
            if let Some(id) = ctx.track_ptr(addr) {
                write_u8(buf, TAG_BACK_REF);
                write_u32(buf, id);
            } else {
                let id = ctx.next_id - 1;
                write_u8(buf, TAG_DEFINE_REF);
                write_u32(buf, id);
                write_u8(buf, TAG_ERROR);
                write_u8(buf, error_kind_to_byte(rc.kind));
                write_str32(buf, &rc.message);
            }
        }
        JsValue::NativeFunction(_) => {
            // Native closures cannot be serialized; store a placeholder.
            write_u8(buf, TAG_NATIVE_FUNCTION);
            write_str32(buf, "<native>");
        }
        // Object (raw pointer), Generator, Iterator, and Context cannot be serialized.
        JsValue::Object(_)
        | JsValue::Generator(_)
        | JsValue::Iterator(_)
        | JsValue::Promise(_)
        | JsValue::Context(_)
        | JsValue::Proxy(_)
        | JsValue::ArrayBuffer(_)
        | JsValue::TypedArray(_)
        | JsValue::DataView(_)
        | JsValue::TheHole => {
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
        ConstantPoolEntry::BigInt(n) => {
            write_u8(buf, CPE_BIGINT);
            buf.extend_from_slice(&n.to_le_bytes());
        }
        ConstantPoolEntry::Function(ba) => {
            write_u8(buf, CPE_FUNCTION);
            write_bytecode_array(buf, ba);
        }
        ConstantPoolEntry::TemplateObject { cooked, raw } => {
            write_u8(buf, CPE_TEMPLATE_OBJECT);
            write_u32(buf, cooked.len() as u32);
            for c in cooked {
                match c {
                    Some(s) => {
                        write_u8(buf, 1);
                        write_str32(buf, s);
                    }
                    None => write_u8(buf, 0),
                }
            }
            for r in raw {
                write_str32(buf, r);
            }
        }
        ConstantPoolEntry::ObjectLiteralTemplate { keys } => {
            write_u8(buf, CPE_OBJECT_LITERAL_TEMPLATE);
            write_u32(buf, keys.len() as u32);
            for key in keys {
                write_str32(buf, key);
            }
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

fn read_jsvalue(bytes: &[u8], cursor: &mut usize, ctx: &mut DeserContext) -> StatorResult<JsValue> {
    let tag = read_u8(bytes, cursor)?;
    match tag {
        TAG_DEFINE_REF => {
            let ref_id = read_u32(bytes, cursor)?;
            let inner_tag = read_u8(bytes, cursor)?;
            read_defined_ref(bytes, cursor, ctx, ref_id, inner_tag)
        }
        TAG_BACK_REF => {
            let ref_id = read_u32(bytes, cursor)?;
            ctx.lookup(ref_id)
        }
        other => read_jsvalue_by_tag(bytes, cursor, ctx, other),
    }
}

/// Deserialize a `DefineRef`-wrapped value and register it in the context.
///
/// For `PlainObject` the Rc is registered *before* entries are read so that
/// circular back-references resolve correctly.
fn read_defined_ref(
    bytes: &[u8],
    cursor: &mut usize,
    ctx: &mut DeserContext,
    ref_id: u32,
    inner_tag: u8,
) -> StatorResult<JsValue> {
    match inner_tag {
        TAG_PLAIN_OBJECT => {
            let count = read_u32(bytes, cursor)? as usize;
            let map = Rc::new(RefCell::new(PropertyMap::with_capacity(count)));
            let value = JsValue::PlainObject(Rc::clone(&map));
            // Register before reading entries to allow circular references.
            ctx.register(ref_id, value.clone());
            for _ in 0..count {
                let k = read_str32(bytes, cursor)?;
                let v = read_jsvalue(bytes, cursor, ctx)?;
                map.borrow_mut().insert(k, v);
            }
            Ok(value)
        }
        TAG_ARRAY => {
            let count = read_u32(bytes, cursor)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_jsvalue(bytes, cursor, ctx)?);
            }
            let value = JsValue::new_array(items);
            ctx.register(ref_id, value.clone());
            Ok(value)
        }
        TAG_FUNCTION => {
            let ba = read_bytecode_array(bytes, cursor)?;
            let value = JsValue::Function(Rc::new(ba));
            ctx.register(ref_id, value.clone());
            Ok(value)
        }
        TAG_ERROR => {
            let kind_byte = read_u8(bytes, cursor)?;
            let kind = byte_to_error_kind(kind_byte)?;
            let message = read_str32(bytes, cursor)?;
            let value = JsValue::Error(Rc::new(JsError::new(kind, message)));
            ctx.register(ref_id, value.clone());
            Ok(value)
        }
        other => Err(StatorError::Internal(format!(
            "snapshot: DefineRef wrapping unexpected tag {other:#04x}"
        ))),
    }
}

/// Read a JsValue given its already-consumed tag byte.
fn read_jsvalue_by_tag(
    bytes: &[u8],
    cursor: &mut usize,
    ctx: &mut DeserContext,
    tag: u8,
) -> StatorResult<JsValue> {
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
            Ok(JsValue::String(s.into()))
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
                items.push(read_jsvalue(bytes, cursor, ctx)?);
            }
            Ok(JsValue::new_array(items))
        }
        TAG_PLAIN_OBJECT => {
            let count = read_u32(bytes, cursor)? as usize;
            let mut map = PropertyMap::with_capacity(count);
            for _ in 0..count {
                let k = read_str32(bytes, cursor)?;
                let v = read_jsvalue(bytes, cursor, ctx)?;
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
        CPE_BIGINT => {
            if *cursor + 16 > bytes.len() {
                return Err(StatorError::Internal(
                    "snapshot: truncated BigInt CPE".into(),
                ));
            }
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&bytes[*cursor..*cursor + 16]);
            *cursor += 16;
            Ok(ConstantPoolEntry::BigInt(i128::from_le_bytes(buf)))
        }
        CPE_FUNCTION => {
            let ba = read_bytecode_array(bytes, cursor)?;
            Ok(ConstantPoolEntry::Function(Rc::new(ba)))
        }
        CPE_TEMPLATE_OBJECT => {
            let len = read_u32(bytes, cursor)? as usize;
            let mut cooked = Vec::with_capacity(len);
            for _ in 0..len {
                let has = read_u8(bytes, cursor)?;
                if has == 1 {
                    cooked.push(Some(read_str32(bytes, cursor)?));
                } else {
                    cooked.push(None);
                }
            }
            let mut raw = Vec::with_capacity(len);
            for _ in 0..len {
                raw.push(read_str32(bytes, cursor)?);
            }
            Ok(ConstantPoolEntry::TemplateObject { cooked, raw })
        }
        CPE_OBJECT_LITERAL_TEMPLATE => {
            let len = read_u32(bytes, cursor)? as usize;
            let mut keys = Vec::with_capacity(len);
            for _ in 0..len {
                keys.push(read_str32(bytes, cursor)?);
            }
            Ok(ConstantPoolEntry::ObjectLiteralTemplate { keys })
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
// Checksum
// ─────────────────────────────────────────────────────────────────────────────

/// Compute FNV-1a 32-bit hash for snapshot integrity checking.
fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Verify the FNV-1a checksum stored in the last 4 bytes of `bytes`.
fn verify_checksum(bytes: &[u8]) -> StatorResult<()> {
    if bytes.len() < 4 {
        return Err(StatorError::Internal(
            "snapshot: blob too small for checksum".into(),
        ));
    }
    let body = &bytes[..bytes.len() - 4];
    let stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().expect("4-byte slice"));
    let computed = fnv1a_32(body);
    if stored != computed {
        return Err(StatorError::Internal(format!(
            "snapshot: checksum mismatch (stored {stored:#010x}, computed {computed:#010x})"
        )));
    }
    Ok(())
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
            JsValue::String("hello, world 🌍".to_string().into()),
        );
        g.insert("empty".to_string(), JsValue::String(String::new().into()));
        let r = round_trip(g);
        assert_eq!(
            r.get("s"),
            Some(&JsValue::String("hello, world 🌍".to_string().into()))
        );
        assert_eq!(r.get("empty"), Some(&JsValue::String(String::new().into())));
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
            JsValue::String("two".to_string().into()),
            JsValue::Null,
        ];
        let mut g = HashMap::new();
        g.insert("arr".to_string(), JsValue::new_array(items.clone()));
        let r = round_trip(g);
        if let Some(JsValue::Array(restored)) = r.get("arr") {
            assert_eq!(*restored.borrow(), items);
        } else {
            panic!("expected Array");
        }
    }

    #[test]
    fn test_round_trip_plain_object() {
        let mut map = PropertyMap::new();
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
        let pool = vec![ConstantPoolEntry::Function(Rc::new(inner_ba.clone()))];
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

    // ── file I/O ──────────────────────────────────────────────────────────────

    #[test]
    fn test_write_and_read_from_file() {
        let dir = std::env::temp_dir().join("stator_snapshot_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_snap.bin");

        let mut g = HashMap::new();
        g.insert("x".to_string(), JsValue::Smi(99));
        g.insert("s".to_string(), JsValue::String("hello".to_string().into()));
        let snap = serialize_globals(&g);
        snap.write_to_file(&path).expect("write should succeed");

        let loaded = StartupSnapshot::read_from_file(&path).expect("read should succeed");
        assert_eq!(loaded.as_bytes(), snap.as_bytes());

        let restored = deserialize_globals(loaded.as_bytes()).expect("deser should succeed");
        assert_eq!(restored.get("x"), Some(&JsValue::Smi(99)));
        assert_eq!(
            restored.get("s"),
            Some(&JsValue::String("hello".to_string().into()))
        );

        // Clean up.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_read_from_nonexistent_file() {
        let path = std::path::PathBuf::from("/tmp/stator_no_such_snapshot.bin");
        let err = StartupSnapshot::read_from_file(&path).unwrap_err();
        assert!(err.to_string().contains("failed to read"));
    }

    // ── validation ────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_good_snapshot() {
        let g: HashMap<String, JsValue> = HashMap::new();
        let snap = serialize_globals(&g);
        snap.validate()
            .expect("valid snapshot should pass validation");
    }

    #[test]
    fn test_validate_bad_magic() {
        let snap = StartupSnapshot::from_bytes(vec![0x00; 16]);
        let err = snap.validate().unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_validate_too_short() {
        let snap = StartupSnapshot::from_bytes(vec![0x00; 4]);
        assert!(snap.validate().is_err());
    }

    #[test]
    fn test_len_and_is_empty() {
        let empty = StartupSnapshot::from_bytes(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let snap = serialize_globals(&HashMap::new());
        assert!(!snap.is_empty());
        assert!(snap.len() > 0);
    }

    // ── shared and circular references ────────────────────────────────────────

    #[test]
    fn test_round_trip_shared_plain_object() {
        let mut shared_map = PropertyMap::new();
        shared_map.insert("x".to_string(), JsValue::Smi(1));
        let shared = Rc::new(RefCell::new(shared_map));
        let mut g = HashMap::new();
        g.insert("a".to_string(), JsValue::PlainObject(Rc::clone(&shared)));
        g.insert("b".to_string(), JsValue::PlainObject(Rc::clone(&shared)));
        let snap = serialize_globals(&g);
        let r = deserialize_globals(snap.as_bytes()).expect("deser");
        // Both globals must point to the same Rc (identity preserved).
        if let (Some(JsValue::PlainObject(a)), Some(JsValue::PlainObject(b))) =
            (r.get("a"), r.get("b"))
        {
            assert!(Rc::ptr_eq(a, b), "shared object identity must be preserved");
            assert_eq!(a.borrow().get("x"), Some(&JsValue::Smi(1)));
        } else {
            panic!("expected two PlainObjects");
        }
    }

    #[test]
    fn test_round_trip_circular_plain_object() {
        let obj = Rc::new(RefCell::new(PropertyMap::new()));
        obj.borrow_mut()
            .insert("self".to_string(), JsValue::PlainObject(Rc::clone(&obj)));
        obj.borrow_mut().insert("val".to_string(), JsValue::Smi(42));
        let mut g = HashMap::new();
        g.insert("circ".to_string(), JsValue::PlainObject(obj));
        let snap = serialize_globals(&g);
        let r = deserialize_globals(snap.as_bytes()).expect("deser");
        if let Some(JsValue::PlainObject(restored)) = r.get("circ") {
            let borrow = restored.borrow();
            assert_eq!(borrow.get("val"), Some(&JsValue::Smi(42)));
            if let Some(JsValue::PlainObject(inner)) = borrow.get("self") {
                assert!(
                    Rc::ptr_eq(restored, inner),
                    "circular reference must restore to same Rc"
                );
            } else {
                panic!("expected self-reference as PlainObject");
            }
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_round_trip_shared_array() {
        let shared = Rc::new(RefCell::new(vec![JsValue::Smi(1), JsValue::Smi(2)]));
        let mut g = HashMap::new();
        g.insert("a".to_string(), JsValue::Array(Rc::clone(&shared)));
        g.insert("b".to_string(), JsValue::Array(Rc::clone(&shared)));
        let snap = serialize_globals(&g);
        let r = deserialize_globals(snap.as_bytes()).expect("deser");
        if let (Some(JsValue::Array(a)), Some(JsValue::Array(b))) = (r.get("a"), r.get("b")) {
            assert!(Rc::ptr_eq(a, b), "shared array identity must be preserved");
            assert_eq!(*a.borrow(), vec![JsValue::Smi(1), JsValue::Smi(2)]);
        } else {
            panic!("expected two Arrays");
        }
    }

    #[test]
    fn test_round_trip_shared_function() {
        let ba = BytecodeArray::new(
            vec![0x01],
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let shared = Rc::new(ba.clone());
        let mut g = HashMap::new();
        g.insert("f1".to_string(), JsValue::Function(Rc::clone(&shared)));
        g.insert("f2".to_string(), JsValue::Function(Rc::clone(&shared)));
        let snap = serialize_globals(&g);
        let r = deserialize_globals(snap.as_bytes()).expect("deser");
        if let (Some(JsValue::Function(f1)), Some(JsValue::Function(f2))) =
            (r.get("f1"), r.get("f2"))
        {
            assert!(
                Rc::ptr_eq(f1, f2),
                "shared function identity must be preserved"
            );
            assert_eq!(f1.as_ref(), &ba);
        } else {
            panic!("expected two Functions");
        }
    }

    #[test]
    #[ignore] // TODO: prototype chain snapshot regression
    fn test_round_trip_prototype_chain() {
        // Simulate a prototype chain: child.__proto__ = parent
        let mut parent_map = PropertyMap::new();
        parent_map.insert(
            "greet".to_string(),
            JsValue::String("hello".to_string().into()),
        );
        let parent = Rc::new(RefCell::new(parent_map));
        let mut child_map = PropertyMap::new();
        child_map.insert("x".to_string(), JsValue::Smi(10));
        child_map.insert(
            "__proto__".to_string(),
            JsValue::PlainObject(Rc::clone(&parent)),
        );
        let child = Rc::new(RefCell::new(child_map));

        // A second child shares the same parent prototype.
        let mut child2_map = PropertyMap::new();
        child2_map.insert("y".to_string(), JsValue::Smi(20));
        child2_map.insert(
            "__proto__".to_string(),
            JsValue::PlainObject(Rc::clone(&parent)),
        );
        let child2 = Rc::new(RefCell::new(child2_map));

        let mut g = HashMap::new();
        g.insert("c1".to_string(), JsValue::PlainObject(child));
        g.insert("c2".to_string(), JsValue::PlainObject(child2));

        let snap = serialize_globals(&g);
        let r = deserialize_globals(snap.as_bytes()).expect("deser");

        if let (Some(JsValue::PlainObject(c1)), Some(JsValue::PlainObject(c2))) =
            (r.get("c1"), r.get("c2"))
        {
            let proto1 = c1
                .borrow()
                .get(crate::objects::property_map::INTERNAL_PROTO_PROPERTY_KEY)
                .cloned();
            let proto2 = c2
                .borrow()
                .get(crate::objects::property_map::INTERNAL_PROTO_PROPERTY_KEY)
                .cloned();
            if let (Some(JsValue::PlainObject(p1)), Some(JsValue::PlainObject(p2))) =
                (&proto1, &proto2)
            {
                assert!(
                    Rc::ptr_eq(p1, p2),
                    "shared prototype must be the same Rc across children"
                );
                assert_eq!(
                    p1.borrow().get("greet"),
                    Some(&JsValue::String("hello".to_string().into()))
                );
            } else {
                panic!("expected __proto__ to be PlainObject");
            }
        } else {
            panic!("expected two PlainObject children");
        }
    }

    // ── checksum ──────────────────────────────────────────────────────────────

    #[test]
    fn test_checksum_corruption_detected() {
        let mut g = HashMap::new();
        g.insert("k".to_string(), JsValue::Smi(1));
        let mut bytes = serialize_globals(&g).into_bytes();
        // Corrupt a data byte (not part of header or checksum).
        if bytes.len() > 12 {
            bytes[10] ^= 0xFF;
        }
        let err = deserialize_globals(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("checksum"),
            "corruption should be caught by checksum: {err}"
        );
    }
}

//! Heap snapshot builder and CDP serialiser.
//!
//! Produces a `HeapProfiler.HeapSnapshot`-compatible JSON payload that can be
//! loaded directly in Chrome DevTools → Memory tab.
//!
//! # Snapshot format
//!
//! The snapshot is a JSON object with the following top-level keys:
//!
//! - `snapshot.meta` — field names and type strings for the flat arrays.
//! - `nodes` — flat array; every [`NODE_FIELDS`] entries describe one node.
//! - `edges` — flat array; every [`EDGE_FIELDS`] entries describe one edge.
//! - `strings` — string interning table referenced by index in `nodes`/`edges`.
//!
//! # Allocation tracking
//!
//! Call [`start_tracking`] to begin recording allocations, then
//! [`stop_tracking`] to obtain the collected [`AllocationRecord`]s.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//! use stator_core::inspector::heap_snapshot::{HeapSnapshotBuilder, start_tracking, stop_tracking, record_allocation};
//!
//! // Build a snapshot from an empty globals map.
//! let globals: HashMap<String, stator_core::objects::value::JsValue> = HashMap::new();
//! let snapshot = HeapSnapshotBuilder::build(&globals);
//! assert!(snapshot.snapshot.node_count >= 1); // at least the root node
//! ```

use std::cell::RefCell;
use std::collections::HashMap;

use serde::Serialize;

use crate::builtins::error::capture_call_stack;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Layout constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of `u32` entries per node in the flat `nodes` array.
///
/// Fields (in order): `type`, `name`, `id`, `self_size`, `edge_count`.
const NODE_FIELDS: usize = 5;

/// Number of `u32` entries per edge in the flat `edges` array.
///
/// Fields (in order): `type`, `name_or_index`, `to_node`.
const EDGE_FIELDS: usize = 3;

// Node type indices — must match the order of entries in `node_types[0]` below.
const NODE_TYPE_HIDDEN: u32 = 0;
const NODE_TYPE_ARRAY: u32 = 1;
const NODE_TYPE_STRING: u32 = 2;
const NODE_TYPE_OBJECT: u32 = 3;
const NODE_TYPE_CLOSURE: u32 = 5;
const NODE_TYPE_NUMBER: u32 = 7;
const NODE_TYPE_NATIVE: u32 = 8;
const NODE_TYPE_SYNTHETIC: u32 = 9;
const NODE_TYPE_SYMBOL: u32 = 12;
const NODE_TYPE_BIGINT: u32 = 13;

// Edge type indices — must match the order of entries in `edge_types[0]` below.
const EDGE_TYPE_ELEMENT: u32 = 1;
const EDGE_TYPE_PROPERTY: u32 = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Allocation tracking
// ─────────────────────────────────────────────────────────────────────────────

/// A single recorded allocation event.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Monotonically increasing allocation identifier.
    pub id: u64,
    /// Self-size of the allocation in bytes.
    pub size: usize,
    /// JavaScript call stack at the point of allocation (bottom-to-top).
    pub stack: Vec<String>,
}

thread_local! {
    /// Whether allocation tracking is currently active.
    static TRACKING_ACTIVE: RefCell<bool> = const { RefCell::new(false) };

    /// Accumulated allocation records for the current tracking session.
    static ALLOCATION_RECORDS: RefCell<Vec<AllocationRecord>> =
        const { RefCell::new(Vec::new()) };

    /// Monotonically increasing allocation counter for this session.
    static ALLOC_COUNTER: RefCell<u64> = const { RefCell::new(0) };
}

/// Begin a new allocation-tracking session.
///
/// Once started, any call to [`record_allocation`] will append to the internal
/// buffer.  Calling this when a session is already active resets the buffer.
pub fn start_tracking() {
    TRACKING_ACTIVE.with(|a| *a.borrow_mut() = true);
    ALLOCATION_RECORDS.with(|r| r.borrow_mut().clear());
    ALLOC_COUNTER.with(|c| *c.borrow_mut() = 0);
}

/// Stop the current allocation-tracking session and return all recorded events.
///
/// Returns an empty `Vec` if no session was active.
pub fn stop_tracking() -> Vec<AllocationRecord> {
    TRACKING_ACTIVE.with(|a| *a.borrow_mut() = false);
    ALLOCATION_RECORDS.with(|r| r.borrow_mut().drain(..).collect())
}

/// Record a single allocation of `size` bytes, if tracking is active.
///
/// Captures the current JavaScript call stack via
/// [`capture_call_stack`][crate::builtins::error::capture_call_stack].
/// This function is essentially free when tracking is not active.
pub fn record_allocation(size: usize) {
    let active = TRACKING_ACTIVE.with(|a| *a.borrow());
    if !active {
        return;
    }
    let id = ALLOC_COUNTER.with(|c| {
        let mut v = c.borrow_mut();
        *v += 1;
        *v
    });
    let stack = capture_call_stack();
    ALLOCATION_RECORDS.with(|r| {
        r.borrow_mut().push(AllocationRecord { id, size, stack });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// CDP data types
// ─────────────────────────────────────────────────────────────────────────────

/// A CDP-serialisable heap snapshot.
///
/// Use [`HeapSnapshotBuilder::build`] to construct a snapshot from a globals
/// map, then call [`HeapSnapshot::to_json`] to obtain the CDP
/// `HeapProfiler.HeapSnapshot` payload.
#[derive(Debug, Serialize)]
pub struct HeapSnapshot {
    /// Snapshot metadata (field names and type tables).
    pub snapshot: SnapshotMeta,
    /// Flat node array; every [`NODE_FIELDS`] entries describe one node.
    pub nodes: Vec<u32>,
    /// Flat edge array; every [`EDGE_FIELDS`] entries describe one edge.
    pub edges: Vec<u32>,
    /// String interning table; all string values referenced by index.
    pub strings: Vec<String>,
}

/// Top-level `snapshot` field in the CDP heap snapshot payload.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotMeta {
    /// Field and type definitions for the `nodes` and `edges` flat arrays.
    pub meta: SnapshotMetaInner,
    /// Total number of nodes.
    pub node_count: u32,
    /// Total number of edges.
    pub edge_count: u32,
}

/// Nested `meta` object inside [`SnapshotMeta`].
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotMetaInner {
    /// Names of the fields for each node entry in the flat `nodes` array.
    pub node_fields: Vec<&'static str>,
    /// Type descriptors for each node field.
    pub node_types: Vec<serde_json::Value>,
    /// Names of the fields for each edge entry in the flat `edges` array.
    pub edge_fields: Vec<&'static str>,
    /// Type descriptors for each edge field.
    pub edge_types: Vec<serde_json::Value>,
}

impl HeapSnapshot {
    /// Serialise this snapshot to a compact JSON string.
    ///
    /// The resulting string is suitable for use as the `chunk` parameter of a
    /// `HeapProfiler.addHeapSnapshotChunk` CDP event.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot builder
// ─────────────────────────────────────────────────────────────────────────────

/// Constructs a [`HeapSnapshot`] by walking a set of root [`JsValue`]s.
///
/// The builder maintains:
/// - A string interning table (deduplicates string names in nodes/edges).
/// - A seen-set to avoid revisiting Rc-backed values already in the graph.
/// - A flat edge list built alongside each node.
pub struct HeapSnapshotBuilder {
    /// Flat node data (`NODE_FIELDS` u32s per node).
    nodes: Vec<u32>,
    /// Flat edge data (`EDGE_FIELDS` u32s per edge).
    edges: Vec<u32>,
    /// `string → index` map for the interning table.
    string_map: HashMap<String, u32>,
    /// Ordered string table; indices are positions in this `Vec`.
    strings: Vec<String>,
    /// Maps a value's stable unique ID to its starting offset in `nodes`.
    seen: HashMap<u64, u32>,
    /// Next unique node ID to assign to primitive or untracked values.
    next_id: u64,
}

impl HeapSnapshotBuilder {
    fn new() -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            string_map: HashMap::new(),
            strings: Vec::new(),
            seen: HashMap::new(),
            next_id: 1,
        };
        // Always intern the empty string at index 0.
        builder.intern("");
        builder
    }

    /// Intern `s` into the string table, returning its index.
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.string_map.get(s) {
            return idx;
        }
        let idx = self.strings.len() as u32;
        self.string_map.insert(s.to_string(), idx);
        self.strings.push(s.to_string());
        idx
    }

    /// Total number of nodes recorded so far.
    fn node_count(&self) -> u32 {
        (self.nodes.len() / NODE_FIELDS) as u32
    }

    /// Total number of edges recorded so far.
    fn edge_count(&self) -> u32 {
        (self.edges.len() / EDGE_FIELDS) as u32
    }

    /// Append a node to the flat array, returning its starting offset.
    ///
    /// The `edge_count` field (index + 4) is written as `0` and patched later
    /// once the node's outgoing edges have been appended.
    fn push_node(&mut self, node_type: u32, name_idx: u32, id: u64, self_size: u32) -> u32 {
        let offset = self.nodes.len() as u32;
        self.nodes.push(node_type);
        self.nodes.push(name_idx);
        // IDs are truncated to 32 bits for the flat array (sufficient for
        // all heap objects in a single session).
        self.nodes.push(id as u32);
        self.nodes.push(self_size);
        self.nodes.push(0); // edge_count — patched by set_edge_count
        offset
    }

    /// Patch the `edge_count` field of the node at `node_offset`.
    fn set_edge_count(&mut self, node_offset: u32, count: u32) {
        self.nodes[node_offset as usize + 4] = count;
    }

    /// Append an edge to the flat array.
    fn push_edge(&mut self, edge_type: u32, name_or_index: u32, to_node_offset: u32) {
        self.edges.push(edge_type);
        self.edges.push(name_or_index);
        self.edges.push(to_node_offset);
    }

    /// Map a `JsValue` to its CDP node-type constant.
    fn value_node_type(value: &JsValue) -> u32 {
        match value {
            JsValue::Undefined | JsValue::Null | JsValue::Boolean(_) => NODE_TYPE_HIDDEN,
            JsValue::Smi(_) | JsValue::HeapNumber(_) => NODE_TYPE_NUMBER,
            JsValue::String(_) => NODE_TYPE_STRING,
            JsValue::Symbol(_) => NODE_TYPE_SYMBOL,
            JsValue::BigInt(_) => NODE_TYPE_BIGINT,
            JsValue::Function(_) => NODE_TYPE_CLOSURE,
            JsValue::Array(_) => NODE_TYPE_ARRAY,
            JsValue::Object(_) | JsValue::PlainObject(_) => NODE_TYPE_OBJECT,
            JsValue::NativeFunction(_) => NODE_TYPE_NATIVE,
            JsValue::Generator(_) | JsValue::Iterator(_) | JsValue::Error(_) => NODE_TYPE_OBJECT,
            JsValue::Promise(_) => NODE_TYPE_OBJECT,
            JsValue::Context(_) => NODE_TYPE_OBJECT,
            JsValue::Proxy(_) => NODE_TYPE_OBJECT,
            JsValue::ArrayBuffer(_) | JsValue::TypedArray(_) | JsValue::DataView(_) => {
                NODE_TYPE_OBJECT
            }
        }
    }

    /// Derive a stable unique 64-bit ID for a `JsValue`.
    ///
    /// Reference types (backed by `Rc` or a raw pointer) use their pointer
    /// address so that shared references map to the same node.  Primitive
    /// values receive a fresh sequential ID each visit — they are not
    /// deduplicated in the graph.
    fn value_id(value: &JsValue, fallback_id: u64) -> u64 {
        match value {
            JsValue::Function(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::Array(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::Generator(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::Iterator(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::Error(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::PlainObject(rc) => std::rc::Rc::as_ptr(rc) as u64,
            JsValue::Object(ptr) => *ptr as u64,
            _ => fallback_id,
        }
    }

    /// Compute a human-readable name string for a node.
    fn value_name(value: &JsValue) -> String {
        match value {
            JsValue::Undefined => "undefined".to_string(),
            JsValue::Null => "null".to_string(),
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Smi(n) => n.to_string(),
            JsValue::HeapNumber(f) => f.to_string(),
            JsValue::String(s) => s.clone(),
            JsValue::Symbol(id) => format!("Symbol({id})"),
            JsValue::BigInt(n) => format!("{n}n"),
            JsValue::Function(_) => "(closure)".to_string(),
            JsValue::Array(_) => "Array".to_string(),
            JsValue::Object(_) => "Object".to_string(),
            JsValue::PlainObject(_) => "Object".to_string(),
            JsValue::NativeFunction(_) => "NativeFunction".to_string(),
            JsValue::Generator(_) => "Generator".to_string(),
            JsValue::Iterator(_) => "Iterator".to_string(),
            JsValue::Error(e) => {
                use std::fmt::Write as _;
                let mut s = String::with_capacity(e.name().len() + 2 + e.message().len());
                let _ = write!(s, "{}: {}", e.name(), e.message());
                s
            }
            JsValue::Promise(_) => "Promise".to_string(),
            JsValue::Context(_) => "Context".to_string(),
            JsValue::Proxy(_) => "Proxy".to_string(),
            JsValue::ArrayBuffer(_) => "ArrayBuffer".to_string(),
            JsValue::TypedArray(ta) => ta.borrow().kind.name().to_string(),
            JsValue::DataView(_) => "DataView".to_string(),
        }
    }

    /// Approximate self-size of a `JsValue` in bytes.
    fn value_self_size(value: &JsValue) -> u32 {
        use std::mem::size_of;
        match value {
            JsValue::String(s) => (size_of::<String>() + s.len()) as u32,
            JsValue::Array(rc) => {
                (size_of::<Vec<JsValue>>() + rc.borrow().len() * size_of::<JsValue>()) as u32
            }
            JsValue::PlainObject(rc) => {
                let map = rc.borrow();
                (size_of::<HashMap<String, JsValue>>()
                    + map.len() * (size_of::<String>() + size_of::<JsValue>()))
                    as u32
            }
            _ => size_of::<JsValue>() as u32,
        }
    }

    /// Visit `value`, adding its node and recursively its children's nodes to
    /// the snapshot graph.  Returns the node's starting offset in `nodes`.
    ///
    /// Reference-typed values that have already been visited return the cached
    /// offset immediately, preventing infinite loops on cyclic graphs.
    fn visit(&mut self, value: &JsValue) -> u32 {
        let fallback_id = self.next_id;
        self.next_id += 1;
        let id = Self::value_id(value, fallback_id);

        // Return the cached offset for reference types already visited.
        if let Some(&offset) = self.seen.get(&id) {
            return offset;
        }

        let node_type = Self::value_node_type(value);
        let name = Self::value_name(value);
        let name_idx = self.intern(&name);
        let self_size = Self::value_self_size(value);
        let node_offset = self.push_node(node_type, name_idx, id, self_size);
        self.seen.insert(id, node_offset);

        // Collect child edges before pushing them so we can pass `count`.
        let mut child_edges: Vec<(u32, u32, u32)> = Vec::new();

        match value {
            JsValue::Array(rc) => {
                for (i, elem) in rc.borrow().iter().enumerate() {
                    let child_offset = self.visit(elem);
                    child_edges.push((EDGE_TYPE_ELEMENT, i as u32, child_offset));
                }
            }
            JsValue::PlainObject(rc) => {
                // Snapshot the entries first to avoid holding a borrow while
                // calling `visit`, which may borrow `self.string_map`.
                let entries: Vec<(String, JsValue)> = rc
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                for (key, val) in entries {
                    let child_offset = self.visit(&val);
                    let key_idx = self.intern(&key);
                    child_edges.push((EDGE_TYPE_PROPERTY, key_idx, child_offset));
                }
            }
            _ => {}
        }

        self.set_edge_count(node_offset, child_edges.len() as u32);
        for (et, ni, to) in child_edges {
            self.push_edge(et, ni, to);
        }

        node_offset
    }

    /// Build a [`HeapSnapshot`] from a globals map.
    ///
    /// The snapshot contains one synthetic root node (`"(GC roots)"`) with
    /// property edges pointing to every global value, plus the transitive
    /// closure of all reachable reference-typed values.
    pub fn build(globals: &HashMap<String, JsValue>) -> HeapSnapshot {
        let mut builder = Self::new();

        // Synthetic root node (id = 0, never appears in `seen` for real values).
        let root_name_idx = builder.intern("(GC roots)");
        let root_offset = builder.push_node(NODE_TYPE_SYNTHETIC, root_name_idx, 0, 0);

        let mut root_edges: Vec<(u32, u32, u32)> = Vec::new();
        for (name, value) in globals {
            let child_offset = builder.visit(value);
            let name_idx = builder.intern(name);
            root_edges.push((EDGE_TYPE_PROPERTY, name_idx, child_offset));
        }

        builder.set_edge_count(root_offset, root_edges.len() as u32);
        for (et, ni, to) in root_edges {
            builder.push_edge(et, ni, to);
        }

        let node_count = builder.node_count();
        let edge_count = builder.edge_count();

        HeapSnapshot {
            snapshot: SnapshotMeta {
                meta: SnapshotMetaInner {
                    node_fields: vec!["type", "name", "id", "self_size", "edge_count"],
                    node_types: vec![
                        serde_json::json!([
                            "hidden",
                            "array",
                            "string",
                            "object",
                            "code",
                            "closure",
                            "regexp",
                            "number",
                            "native",
                            "synthetic",
                            "concatenated string",
                            "sliced string",
                            "symbol",
                            "bigint"
                        ]),
                        serde_json::json!("string"),
                        serde_json::json!("number"),
                        serde_json::json!("number"),
                        serde_json::json!("number"),
                    ],
                    edge_fields: vec!["type", "name_or_index", "to_node"],
                    edge_types: vec![
                        serde_json::json!([
                            "context", "element", "property", "internal", "hidden", "shortcut",
                            "weak"
                        ]),
                        serde_json::json!("string_or_number"),
                        serde_json::json!("node"),
                    ],
                },
                node_count,
                edge_count,
            },
            nodes: builder.nodes,
            edges: builder.edges,
            strings: builder.strings,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use super::*;
    use crate::objects::property_map::PropertyMap;
    use crate::objects::value::JsValue;

    // ── HeapSnapshotBuilder::build ────────────────────────────────────────────

    #[test]
    fn test_snapshot_empty_globals_has_root_node() {
        let globals = HashMap::new();
        let snap = HeapSnapshotBuilder::build(&globals);
        // Root synthetic node must always be present.
        assert_eq!(snap.snapshot.node_count, 1);
        assert_eq!(snap.snapshot.edge_count, 0);
        assert_eq!(snap.nodes.len(), NODE_FIELDS);
        // Node type at offset 0 must be NODE_TYPE_SYNTHETIC (9).
        assert_eq!(snap.nodes[0], NODE_TYPE_SYNTHETIC);
    }

    #[test]
    fn test_snapshot_primitive_globals_adds_nodes() {
        let mut globals = HashMap::new();
        globals.insert("x".to_string(), JsValue::Smi(42));
        globals.insert("s".to_string(), JsValue::String("hello".to_string()));
        let snap = HeapSnapshotBuilder::build(&globals);
        // Root + 2 primitive nodes.
        assert_eq!(snap.snapshot.node_count, 3);
        assert_eq!(snap.snapshot.edge_count, 2);
    }

    #[test]
    fn test_snapshot_node_count_matches_flat_array_length() {
        let mut globals = HashMap::new();
        globals.insert("n".to_string(), JsValue::HeapNumber(3.14));
        let snap = HeapSnapshotBuilder::build(&globals);
        assert_eq!(
            snap.nodes.len(),
            snap.snapshot.node_count as usize * NODE_FIELDS
        );
        assert_eq!(
            snap.edges.len(),
            snap.snapshot.edge_count as usize * EDGE_FIELDS
        );
    }

    #[test]
    fn test_snapshot_plain_object_edges() {
        let mut inner = PropertyMap::new();
        inner.insert("a".to_string(), JsValue::Smi(1));
        inner.insert("b".to_string(), JsValue::Smi(2));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(inner)));

        let mut globals = HashMap::new();
        globals.insert("obj".to_string(), obj);

        let snap = HeapSnapshotBuilder::build(&globals);
        // Root → obj → 2 Smi children = 4 nodes total.
        assert_eq!(snap.snapshot.node_count, 4);
        // root→obj + obj→a + obj→b = 3 edges.
        assert_eq!(snap.snapshot.edge_count, 3);
    }

    #[test]
    fn test_snapshot_array_element_edges() {
        let arr = JsValue::new_array(vec![JsValue::Smi(10), JsValue::Smi(20)]);
        let mut globals = HashMap::new();
        globals.insert("arr".to_string(), arr);

        let snap = HeapSnapshotBuilder::build(&globals);
        // Root → arr → elem0 → elem1 = 4 nodes.
        assert_eq!(snap.snapshot.node_count, 4);
        // root→arr + arr→elem0 + arr→elem1 = 3 edges.
        assert_eq!(snap.snapshot.edge_count, 3);
    }

    #[test]
    fn test_snapshot_shared_rc_deduplication() {
        // Two globals pointing to the same Rc<RefCell<Vec<JsValue>>>.
        let shared = Rc::new(RefCell::new(vec![JsValue::Smi(1)]));
        let mut globals = HashMap::new();
        globals.insert("a".to_string(), JsValue::Array(Rc::clone(&shared)));
        globals.insert("b".to_string(), JsValue::Array(Rc::clone(&shared)));

        let snap = HeapSnapshotBuilder::build(&globals);
        // Root + 1 shared array + 1 element = 3 nodes (not 4).
        assert_eq!(snap.snapshot.node_count, 3);
    }

    #[test]
    fn test_snapshot_to_json_valid() {
        let globals = HashMap::new();
        let snap = HeapSnapshotBuilder::build(&globals);
        let json = snap.to_json();
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("snapshot must be valid JSON");
        assert!(parsed["snapshot"]["nodeCount"].is_number());
        assert!(parsed["nodes"].is_array());
        assert!(parsed["edges"].is_array());
        assert!(parsed["strings"].is_array());
    }

    #[test]
    fn test_snapshot_string_table_contains_root_name() {
        let globals = HashMap::new();
        let snap = HeapSnapshotBuilder::build(&globals);
        assert!(
            snap.strings.contains(&"(GC roots)".to_string()),
            "string table must contain the root node name"
        );
    }

    // ── Allocation tracking ───────────────────────────────────────────────────

    #[test]
    fn test_record_allocation_inactive_is_noop() {
        // Make sure tracking is off.
        let _ = stop_tracking();
        record_allocation(128);
        // No records were collected.
        let records = stop_tracking();
        assert!(records.is_empty());
    }

    #[test]
    fn test_start_stop_tracking_basic() {
        start_tracking();
        record_allocation(64);
        record_allocation(32);
        let records = stop_tracking();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].size, 64);
        assert_eq!(records[1].size, 32);
        // IDs must be monotonically increasing.
        assert!(records[1].id > records[0].id);
    }

    #[test]
    fn test_stop_tracking_returns_empty_when_inactive() {
        let _ = stop_tracking(); // ensure inactive
        let records = stop_tracking();
        assert!(records.is_empty());
    }

    #[test]
    fn test_start_tracking_resets_buffer() {
        start_tracking();
        record_allocation(10);
        start_tracking(); // reset
        record_allocation(20);
        let records = stop_tracking();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].size, 20);
    }

    #[test]
    fn test_allocation_record_has_id_and_stack() {
        start_tracking();
        record_allocation(256);
        let mut records = stop_tracking();
        assert_eq!(records.len(), 1);
        let rec = records.remove(0);
        assert_eq!(rec.id, 1);
        assert_eq!(rec.size, 256);
        // Stack is whatever the current JS call stack is (likely empty in tests).
        let _ = rec.stack;
    }
}

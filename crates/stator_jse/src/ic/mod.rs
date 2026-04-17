//! Inline-cache (IC) runtime for the Stator VM.
//!
//! This module implements the adaptive property-access and call-site optimisation
//! layer used by the bytecode interpreter.  Each operation is guarded by a
//! **feedback slot** in the function's [`FeedbackVector`]; the slot records
//! the current speculation state and the IC handler caches the observed
//! object shapes so that subsequent accesses can use a direct index instead of
//! a full descriptor scan.
//!
//! # Shape-based validation
//!
//! Inline caches use [`ShapeId`] from the transition-tree shape system
//! ([`crate::objects::shapes`]) to validate receivers in O(1) — a single
//! integer comparison replaces the previous O(n) descriptor-name scan.
//! When the object carries a [`ShapeId`], the IC records it together with the
//! property's `field_index` obtained from the [`ShapeTable`].  On
//! subsequent accesses the cached `ShapeId` is compared against the
//! receiver's current shape; on a match the property value is read/written
//! directly at the cached offset.
//!
//! Objects that do **not** participate in the shape system (no [`ShapeId`])
//! fall back to a legacy name-list comparison that inspects the [`Map`]
//! descriptor array.
//!
//! # State machine
//!
//! Every IC slot follows the same forward-only state machine:
//!
//! ```text
//! Uninitialized → Monomorphic → Polymorphic → Megamorphic
//! ```
//!
//! * **Uninitialized** — no type feedback yet; performs a full runtime lookup
//!   and records the first cache entry.
//! * **Monomorphic** — exactly one shape observed; verifies the receiver shape
//!   with a single `ShapeId` comparison (or a name-list fallback) and, on a
//!   hit, reads or writes the property directly by its cached offset.
//! * **Polymorphic** — 2–[`POLY_MAX`] distinct shapes observed; checks each
//!   entry in turn.
//! * **Megamorphic** — more than [`POLY_MAX`] distinct shapes; all speculation
//!   abandoned; always falls back to the full runtime path.
//!
//! # Structures
//!
//! | Type                | Feedback kind      | Description                                  |
//! |---------------------|--------------------|----------------------------------------------|
//! | [`PropertyLoadIc`]  | `LoadProperty`     | Named property load (`LdaNamedProperty`)     |
//! | [`PropertyStoreIc`] | `StoreProperty`    | Named property store (`StaNamedProperty`)    |
//! | [`CallIc`]          | `Call`             | Function call site state tracking            |
//! | [`IcStats`]         | —                  | Aggregate hit/miss/transition counters       |
//!
//! # Example
//!
//! ```
//! use stator_jse::ic::PropertyLoadIc;
//! use stator_jse::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind, FeedbackVector};
//! use stator_jse::objects::js_object::JsObject;
//! use stator_jse::objects::shapes::ShapeTable;
//! use stator_jse::objects::map::PropertyAttributes;
//! use stator_jse::objects::value::JsValue;
//!
//! let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]);
//! let mut feedback = FeedbackVector::new(&meta);
//! let mut ic = PropertyLoadIc::new();
//! let mut table = ShapeTable::new();
//! let attrs = PropertyAttributes::WRITABLE
//!     | PropertyAttributes::ENUMERABLE
//!     | PropertyAttributes::CONFIGURABLE;
//!
//! // Create an object with shape tracking.
//! let shape = table.transition(table.root(), "x", attrs);
//! let mut obj = JsObject::new();
//! obj.set_property("x", JsValue::Smi(42)).unwrap();
//! obj.set_shape_id(shape);
//!
//! // First load: Uninitialized → Monomorphic; returns 42.
//! let val = ic.load(&obj, "x", &mut feedback, 0, &table);
//! assert_eq!(val, JsValue::Smi(42));
//!
//! // Second load: Monomorphic fast path (ShapeId check); returns 42.
//! let val2 = ic.load(&obj, "x", &mut feedback, 0, &table);
//! assert_eq!(val2, JsValue::Smi(42));
//! ```

use std::rc::Rc;

use crate::bytecode::bytecode_array::BytecodeArray;
use crate::bytecode::feedback::{FeedbackVector, InlineCacheState};
use crate::error::StatorResult;
use crate::objects::js_object::JsObject;
use crate::objects::shapes::{ShapeId, ShapeTable};
use crate::objects::value::JsValue;

/// Maximum number of distinct shapes cached before a slot goes megamorphic.
pub const POLY_MAX: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// IcStats — aggregate hit / miss / transition counters
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for inline-cache activity.
///
/// An `IcStats` instance can be shared across all IC handlers for a single
/// function (or globally) to collect performance data for profiling and JIT
/// heuristics.
#[derive(Debug, Clone, Default)]
pub struct IcStats {
    /// Number of accesses that hit the monomorphic fast path.
    pub mono_hits: u64,
    /// Number of accesses that hit a polymorphic fast-path entry.
    pub poly_hits: u64,
    /// Number of accesses that fell through to the full runtime lookup.
    pub misses: u64,
    /// Number of state transitions (e.g. Uninitialized → Monomorphic).
    pub transitions: u64,
}

impl IcStats {
    /// Creates a zeroed statistics block.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of IC lookups (hits + misses).
    pub fn total(&self) -> u64 {
        self.mono_hits + self.poly_hits + self.misses
    }

    /// Returns the overall hit ratio as a value in `[0.0, 1.0]`, or `0.0` if
    /// no lookups have been recorded.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.mono_hits + self.poly_hits) as f64 / total as f64
    }

    /// Resets all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyEntry — cached (shape, offset) pair
// ─────────────────────────────────────────────────────────────────────────────

/// A single cached entry for a named-property IC.
///
/// When the object carries a [`ShapeId`], the entry stores the shape id and
/// the property's `field_index` obtained from the [`ShapeTable`].  Matching
/// is then a single `u32` comparison.
///
/// For legacy objects without a `ShapeId`, the entry falls back to storing
/// the ordered property-name fingerprint from the [`Map`] descriptor array
/// so that shape equivalence can still be verified (at higher cost).
///
/// `fast_index` is the zero-based position of the target property in the
/// object's fast-properties storage, allowing a direct `O(1)` read or write
/// after the shape check passes.
#[derive(Debug, Clone)]
struct PropertyEntry {
    /// Shape-based identity (preferred fast path).
    shape_id: Option<ShapeId>,
    /// Legacy ordered property-name fingerprint (fallback when no `ShapeId`).
    legacy_shape: Option<Vec<String>>,
    /// Fast-mode slot index of the cached property within this shape.
    fast_index: usize,
}

impl PropertyEntry {
    /// Return `true` if `obj` matches this entry's cached shape.
    ///
    /// Prefers the O(1) [`ShapeId`] comparison when available, falling back
    /// to the O(n) legacy name-list comparison otherwise.
    fn matches(&self, obj: &JsObject) -> bool {
        // Fast path: ShapeId comparison (single u32 compare).
        if let Some(cached_sid) = self.shape_id {
            return obj.shape_id() == Some(cached_sid);
        }
        // Legacy fallback: compare descriptor name lists.
        if let Some(ref names) = self.legacy_shape {
            if !obj.is_fast_mode() {
                return false;
            }
            let descs = obj.map().descriptors();
            return descs.len() == names.len()
                && descs.iter().zip(names.iter()).all(|(d, s)| d.key() == s);
        }
        false
    }

    /// Returns `true` if this entry uses the shape-id fast path.
    #[cfg(test)]
    fn is_shape_based(&self) -> bool {
        self.shape_id.is_some()
    }
}

/// Build a [`PropertyEntry`] for `key` on `obj`.
///
/// When the object has an associated [`ShapeId`], the entry is built from the
/// shape table (O(1) matching).  Otherwise the legacy name-list fingerprint
/// is used.  Returns `None` when the property cannot be found or the object
/// is in slow (dictionary) mode.
fn make_property_entry(obj: &JsObject, key: &str, table: &ShapeTable) -> Option<PropertyEntry> {
    // Preferred: ShapeId-based entry.
    if let Some(sid) = obj.shape_id() {
        let desc = table.lookup(sid, key)?;
        return Some(PropertyEntry {
            shape_id: Some(sid),
            legacy_shape: None,
            fast_index: desc.field_index() as usize,
        });
    }
    // Legacy fallback: descriptor name-list fingerprint.
    if !obj.is_fast_mode() {
        return None;
    }
    let descs = obj.map().descriptors();
    let fast_index = descs.iter().position(|d| d.key() == key)?;
    let names = descs.iter().map(|d| d.key().to_string()).collect();
    Some(PropertyEntry {
        shape_id: None,
        legacy_shape: Some(names),
        fast_index,
    })
}

/// Returns `true` if two entries refer to the same shape.
fn same_shape(a: &PropertyEntry, b: &PropertyEntry) -> bool {
    if let (Some(sa), Some(sb)) = (a.shape_id, b.shape_id) {
        return sa == sb;
    }
    if let (Some(la), Some(lb)) = (&a.legacy_shape, &b.legacy_shape) {
        return la == lb;
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyLoadIc
// ─────────────────────────────────────────────────────────────────────────────

/// IC handler for named property loads (`LdaNamedProperty`).
///
/// Create one `PropertyLoadIc` per `LoadProperty` feedback slot and call
/// [`load`](PropertyLoadIc::load) at every corresponding instruction.
///
/// # Fast path
///
/// On a shape hit the property value is read directly by its cached offset,
/// bypassing the linear descriptor scan performed by
/// [`JsObject::get_own_property`].  When the object carries a [`ShapeId`],
/// the shape check is a single `u32` comparison.
#[derive(Debug, Default)]
pub struct PropertyLoadIc {
    entries: Vec<PropertyEntry>,
}

impl PropertyLoadIc {
    /// Create a new, empty (uninitialized) property-load IC.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns the number of cached shape entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Load property `key` from `obj`, using the IC fast path when possible.
    ///
    /// The [`ShapeTable`] is consulted to build shape-based cache entries
    /// when the object participates in the transition-tree shape system.
    /// Checks `feedback[slot]` for the current speculation state, attempts a
    /// fast read on a shape hit, and falls back to
    /// [`JsObject::get_property`] on a miss.  Transitions the feedback slot
    /// forward whenever a new shape is observed.
    ///
    /// When `stats` is `Some`, hit/miss/transition counters are updated.
    pub fn load(
        &mut self,
        obj: &JsObject,
        key: &str,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
    ) -> JsValue {
        self.load_inner(obj, key, feedback, slot, table, None)
    }

    /// Like [`load`](Self::load) but also records statistics in `stats`.
    pub fn load_with_stats(
        &mut self,
        obj: &JsObject,
        key: &str,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
        stats: &mut IcStats,
    ) -> JsValue {
        self.load_inner(obj, key, feedback, slot, table, Some(stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn load_inner(
        &mut self,
        obj: &JsObject,
        key: &str,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
        mut stats: Option<&mut IcStats>,
    ) -> JsValue {
        match feedback.get_state(slot) {
            None => {
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.get_property(key)
            }

            Some(InlineCacheState::Uninitialized) => {
                let val = obj.get_property(key);
                if let Some(entry) = make_property_entry(obj, key, table) {
                    self.entries.push(entry);
                    feedback.transition(slot, InlineCacheState::Monomorphic);
                    if let Some(s) = stats.as_mut() {
                        s.transitions += 1;
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                val
            }

            Some(InlineCacheState::Monomorphic) => {
                if let Some(entry) = self.entries.first() {
                    if entry.matches(obj) {
                        // ── Monomorphic fast path ─────────────────────────
                        if let Some(s) = stats.as_mut() {
                            s.mono_hits += 1;
                        }
                        return obj
                            .get_fast_property_at_index(entry.fast_index)
                            .unwrap_or(JsValue::Undefined);
                    }
                    // Shape mismatch: record the new shape (if unique) and
                    // advance to polymorphic.
                    if let Some(new_entry) = make_property_entry(obj, key, table)
                        && !self.entries.iter().any(|e| same_shape(e, &new_entry))
                    {
                        self.entries.push(new_entry);
                    }
                    feedback.transition(slot, InlineCacheState::Polymorphic);
                    if let Some(s) = stats.as_mut() {
                        s.transitions += 1;
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.get_property(key)
            }

            Some(InlineCacheState::Polymorphic) => {
                if let Some(entry) = self.entries.iter().find(|e| e.matches(obj)) {
                    // ── Polymorphic fast path ─────────────────────────────
                    if let Some(s) = stats.as_mut() {
                        s.poly_hits += 1;
                    }
                    return obj
                        .get_fast_property_at_index(entry.fast_index)
                        .unwrap_or(JsValue::Undefined);
                }
                if let Some(new_entry) = make_property_entry(obj, key, table)
                    && !self.entries.iter().any(|e| same_shape(e, &new_entry))
                {
                    if self.entries.len() < POLY_MAX {
                        self.entries.push(new_entry);
                    } else {
                        self.entries.clear();
                        feedback.transition(slot, InlineCacheState::Megamorphic);
                        if let Some(s) = stats.as_mut() {
                            s.transitions += 1;
                        }
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.get_property(key)
            }

            Some(InlineCacheState::Megamorphic) => {
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.get_property(key)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyStoreIc
// ─────────────────────────────────────────────────────────────────────────────

/// IC handler for named property stores (`StaNamedProperty`).
///
/// Create one `PropertyStoreIc` per `StoreProperty` feedback slot and call
/// [`store`](PropertyStoreIc::store) at every corresponding instruction.
///
/// # Fast path
///
/// When the receiver's shape matches a cached entry and the property already
/// exists in the object's fast-properties array, the value is written directly
/// by index, bypassing the descriptor scan and attribute checks performed by
/// [`JsObject::set_property`].  With the shape-system integration, the
/// shape check is a single [`ShapeId`] comparison.
///
/// # New-property additions
///
/// When the property does not yet exist on the receiver (the object's shape
/// does not match the cached post-write shape), the IC falls back to the full
/// [`JsObject::set_property`] path.  The resulting post-write shape is
/// recorded so that subsequent stores to objects that already carry the
/// property can use the fast path.
#[derive(Debug, Default)]
pub struct PropertyStoreIc {
    entries: Vec<PropertyEntry>,
}

impl PropertyStoreIc {
    /// Create a new, empty (uninitialized) property-store IC.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns the number of cached shape entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Store `value` into property `key` on `obj`, using the IC fast path
    /// when possible.
    ///
    /// The [`ShapeTable`] is consulted to build shape-based cache entries.
    /// Checks `feedback[slot]` for the current speculation state, attempts a
    /// fast write on a shape hit, and falls back to
    /// [`JsObject::set_property`] on a miss.  Transitions the feedback slot
    /// forward whenever a new shape is observed.
    pub fn store(
        &mut self,
        obj: &mut JsObject,
        key: &str,
        value: JsValue,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
    ) -> StatorResult<()> {
        self.store_inner(obj, key, value, feedback, slot, table, None)
    }

    /// Like [`store`](Self::store) but also records statistics in `stats`.
    #[allow(clippy::too_many_arguments)]
    pub fn store_with_stats(
        &mut self,
        obj: &mut JsObject,
        key: &str,
        value: JsValue,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
        stats: &mut IcStats,
    ) -> StatorResult<()> {
        self.store_inner(obj, key, value, feedback, slot, table, Some(stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn store_inner(
        &mut self,
        obj: &mut JsObject,
        key: &str,
        value: JsValue,
        feedback: &mut FeedbackVector,
        slot: u32,
        table: &ShapeTable,
        mut stats: Option<&mut IcStats>,
    ) -> StatorResult<()> {
        match feedback.get_state(slot) {
            None => {
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.set_property(key, value)
            }

            Some(InlineCacheState::Uninitialized) => {
                obj.set_property(key, value)?;
                if let Some(entry) = make_property_entry(obj, key, table) {
                    self.entries.push(entry);
                    feedback.transition(slot, InlineCacheState::Monomorphic);
                    if let Some(s) = stats.as_mut() {
                        s.transitions += 1;
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                Ok(())
            }

            Some(InlineCacheState::Monomorphic) => {
                let entry_opt = self.entries.first().cloned();
                if let Some(entry) = entry_opt {
                    if entry.matches(obj) {
                        // ── Monomorphic fast path ─────────────────────────
                        obj.set_fast_property_at_index(entry.fast_index, value);
                        if let Some(s) = stats.as_mut() {
                            s.mono_hits += 1;
                        }
                        return Ok(());
                    }
                    // Shape mismatch: runtime store.
                    obj.set_property(key, value)?;
                    if let Some(new_entry) = make_property_entry(obj, key, table)
                        && !self.entries.iter().any(|e| same_shape(e, &new_entry))
                    {
                        self.entries.push(new_entry);
                        feedback.transition(slot, InlineCacheState::Polymorphic);
                        if let Some(s) = stats.as_mut() {
                            s.transitions += 1;
                        }
                    }
                } else {
                    obj.set_property(key, value)?;
                    if let Some(entry) = make_property_entry(obj, key, table) {
                        self.entries.push(entry);
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                Ok(())
            }

            Some(InlineCacheState::Polymorphic) => {
                let entry_opt = self.entries.iter().find(|e| e.matches(obj)).cloned();
                if let Some(entry) = entry_opt {
                    // ── Polymorphic fast path ─────────────────────────────
                    obj.set_fast_property_at_index(entry.fast_index, value);
                    if let Some(s) = stats.as_mut() {
                        s.poly_hits += 1;
                    }
                    return Ok(());
                }
                // Miss: runtime store.
                obj.set_property(key, value)?;
                if let Some(new_entry) = make_property_entry(obj, key, table)
                    && !self.entries.iter().any(|e| same_shape(e, &new_entry))
                {
                    if self.entries.len() < POLY_MAX {
                        self.entries.push(new_entry);
                    } else {
                        self.entries.clear();
                        feedback.transition(slot, InlineCacheState::Megamorphic);
                        if let Some(s) = stats.as_mut() {
                            s.transitions += 1;
                        }
                    }
                }
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                Ok(())
            }

            Some(InlineCacheState::Megamorphic) => {
                if let Some(s) = stats.as_mut() {
                    s.misses += 1;
                }
                obj.set_property(key, value)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CallIc
// ─────────────────────────────────────────────────────────────────────────────

/// IC handler for function call sites.
///
/// Tracks the set of distinct callees observed at a single call site and
/// updates the corresponding [`FeedbackVector`] slot accordingly.
///
/// The call IC does **not** perform the call itself; the interpreter is
/// responsible for executing the callee.  Call [`record`](CallIc::record)
/// before or after each call at the site to keep the IC state current.
///
/// Callee identity is determined by the raw pointer of the
/// [`Rc`]-managed [`BytecodeArray`](crate::bytecode::bytecode_array::BytecodeArray):
/// two `JsValue::Function` values are considered the *same* callee iff they
/// share the same underlying `Rc` allocation.  Raw pointers are compared with
/// [`std::ptr::eq`] rather than casting to `usize`, which keeps the
/// comparison Miri-safe.
#[derive(Debug, Default)]
pub struct CallIc {
    /// Raw pointers to the `BytecodeArray` backing store of each observed callee.
    callee_ids: Vec<*const BytecodeArray>,
}

impl CallIc {
    /// Create a new, empty (uninitialized) call IC.
    pub fn new() -> Self {
        Self {
            callee_ids: Vec::new(),
        }
    }

    /// Record an observation of `callee` at this call site and update
    /// `feedback[slot]`.
    ///
    /// * If `callee` is a new callee and the slot is not yet megamorphic, the
    ///   callee identity is added to the cache and the slot advances.
    /// * Once the slot becomes megamorphic no further tracking is performed.
    /// * Non-`Function` callees (e.g., `undefined`) are silently ignored; the
    ///   interpreter is expected to emit a `TypeError` separately.
    pub fn record(&mut self, callee: &JsValue, feedback: &mut FeedbackVector, slot: u32) {
        let Some(id) = callee_identity(callee) else {
            return;
        };
        match feedback.get_state(slot) {
            None => {}

            Some(InlineCacheState::Uninitialized) => {
                self.callee_ids.push(id);
                feedback.transition(slot, InlineCacheState::Monomorphic);
            }

            Some(InlineCacheState::Monomorphic) => {
                if !self.callee_ids.iter().any(|&p| std::ptr::eq(p, id)) {
                    self.callee_ids.push(id);
                    feedback.transition(slot, InlineCacheState::Polymorphic);
                }
            }

            Some(InlineCacheState::Polymorphic) => {
                if !self.callee_ids.iter().any(|&p| std::ptr::eq(p, id)) {
                    if self.callee_ids.len() >= POLY_MAX {
                        self.callee_ids.clear();
                        feedback.transition(slot, InlineCacheState::Megamorphic);
                    } else {
                        self.callee_ids.push(id);
                    }
                }
            }

            Some(InlineCacheState::Megamorphic) => {
                // Already megamorphic; nothing more to record.
            }
        }
    }

    /// Return the number of distinct callees currently cached.
    pub fn callee_count(&self) -> usize {
        self.callee_ids.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a stable pointer identity for a `JsValue::Function` callee.
///
/// Returns the raw `*const BytecodeArray` pointer from the underlying `Rc`
/// so that two `Function` values backed by the same `Rc` allocation compare
/// equal via [`std::ptr::eq`].  This avoids a pointer-to-integer cast, keeping
/// the implementation Miri-safe.  Returns `None` for any non-`Function` value.
fn callee_identity(callee: &JsValue) -> Option<*const BytecodeArray> {
    if let JsValue::Function(rc) = callee {
        Some(Rc::as_ptr(rc))
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind};
    use crate::objects::map::PropertyAttributes;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Default property attributes used when building shapes.
    fn default_attrs() -> PropertyAttributes {
        PropertyAttributes::WRITABLE
            | PropertyAttributes::ENUMERABLE
            | PropertyAttributes::CONFIGURABLE
    }

    fn load_feedback() -> FeedbackVector {
        FeedbackVector::new(&FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]))
    }

    fn store_feedback() -> FeedbackVector {
        FeedbackVector::new(&FeedbackMetadata::new(vec![
            FeedbackSlotKind::StoreProperty,
        ]))
    }

    fn call_feedback() -> FeedbackVector {
        FeedbackVector::new(&FeedbackMetadata::new(vec![FeedbackSlotKind::Call]))
    }

    /// Build an object with the given properties set to `Smi(0)` using legacy
    /// (no `ShapeId`) mode.
    fn obj_with_props(keys: &[&str]) -> JsObject {
        let mut obj = JsObject::new();
        for key in keys {
            obj.set_property(key, JsValue::Smi(0)).unwrap();
        }
        obj
    }

    /// Build an object with the given properties **and** a matching `ShapeId`
    /// from `table`.
    fn shaped_obj(table: &mut ShapeTable, keys: &[&str], values: &[JsValue]) -> JsObject {
        let attrs = default_attrs();
        let mut shape = table.root();
        for key in keys {
            shape = table.transition(shape, key, attrs);
        }
        let mut obj = JsObject::new();
        for (key, val) in keys.iter().zip(values.iter()) {
            obj.set_property(key, val.clone()).unwrap();
        }
        obj.set_shape_id(shape);
        obj
    }

    // ── PropertyLoadIc: ShapeId-based mono/poly/mega transitions ─────────────

    #[test]
    fn test_load_ic_shape_uninitialized_to_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();
        let obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(7)]);

        let val = ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Smi(7));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert!(ic.entries[0].is_shape_based());
    }

    #[test]
    fn test_load_ic_shape_mono_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();
        let obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(42)]);

        ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        let val = ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Smi(42));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_shape_mono_to_poly_on_shape_mismatch() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();

        // Shape 1: {x}
        let obj1 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(1)]);
        // Shape 2: {y, x}
        let obj2 = shaped_obj(&mut table, &["y", "x"], &[JsValue::Smi(0), JsValue::Smi(2)]);

        ic.load(&obj1, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_load_ic_shape_poly_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();

        let obj1 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(10)]);
        let obj2 = shaped_obj(
            &mut table,
            &["y", "x"],
            &[JsValue::Smi(0), JsValue::Smi(20)],
        );

        ic.load(&obj1, "x", &mut fb, 0, &table);
        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        assert_eq!(ic.load(&obj1, "x", &mut fb, 0, &table), JsValue::Smi(10));
        assert_eq!(ic.load(&obj2, "x", &mut fb, 0, &table), JsValue::Smi(20));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_load_ic_shape_poly_to_mega() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut keys: Vec<String> = (0..i).map(|j| format!("p{j}")).collect();
            keys.push("x".to_string());
            let mut vals: Vec<JsValue> = (0..i).map(|_| JsValue::Smi(0)).collect();
            vals.push(JsValue::Smi(i as i32));

            let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
            let obj = shaped_obj(&mut table, &key_refs, &vals);
            ic.load(&obj, "x", &mut fb, 0, &table);
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_load_ic_shape_same_shape_stays_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();

        let obj1 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(1)]);
        let obj2 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(2)]);
        // Both share the same ShapeId because shaped_obj deduplicates.
        assert_eq!(obj1.shape_id(), obj2.shape_id());

        ic.load(&obj1, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    // ── Legacy (no ShapeId) PropertyLoadIc tests ────────────────────────────

    #[test]
    fn test_load_ic_legacy_uninitialized_to_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(7)).unwrap();

        let val = ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Smi(7));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert!(!ic.entries[0].is_shape_based());
    }

    #[test]
    fn test_load_ic_legacy_mono_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(42)).unwrap();

        ic.load(&obj, "x", &mut fb, 0, &table);
        let val = ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Smi(42));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_legacy_mono_to_poly_on_shape_mismatch() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(1)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(2)).unwrap();

        ic.load(&obj1, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_load_ic_legacy_poly_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(10)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(20)).unwrap();

        ic.load(&obj1, "x", &mut fb, 0, &table);
        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        assert_eq!(ic.load(&obj1, "x", &mut fb, 0, &table), JsValue::Smi(10));
        assert_eq!(ic.load(&obj2, "x", &mut fb, 0, &table), JsValue::Smi(20));
    }

    #[test]
    fn test_load_ic_legacy_poly_to_mega() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(i as i32)).unwrap();
            ic.load(&obj, "x", &mut fb, 0, &table);
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_load_ic_megamorphic_always_runtime() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(i as i32)).unwrap();
            ic.load(&obj, "x", &mut fb, 0, &table);
        }

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(99)).unwrap();
        let val = ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Smi(99));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_load_ic_missing_property_returns_undefined() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let obj = JsObject::new();
        let val = ic.load(&obj, "missing", &mut fb, 0, &table);
        assert_eq!(val, JsValue::Undefined);
    }

    #[test]
    fn test_load_ic_legacy_same_shape_stays_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(1)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("x", JsValue::Smi(2)).unwrap();

        ic.load(&obj1, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.load(&obj2, "x", &mut fb, 0, &table);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_out_of_range_slot() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(5)).unwrap();

        let val = ic.load(&obj, "x", &mut fb, 99, &table);
        assert_eq!(val, JsValue::Smi(5));
    }

    // ── PropertyStoreIc: ShapeId-based tests ────────────────────────────────

    #[test]
    fn test_store_ic_shape_uninitialized_to_mono() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let mut table = ShapeTable::new();
        let mut obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);

        ic.store(&mut obj, "x", JsValue::Smi(7), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(7)));
        assert!(ic.entries[0].is_shape_based());
    }

    #[test]
    fn test_store_ic_shape_mono_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let mut table = ShapeTable::new();
        let mut obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);

        ic.store(&mut obj, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        let mut obj2 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);
        ic.store(&mut obj2, "x", JsValue::Smi(99), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(99)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_store_ic_shape_mono_to_poly() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let mut table = ShapeTable::new();

        let mut obj1 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);
        let mut obj2 = shaped_obj(&mut table, &["y", "x"], &[JsValue::Smi(0), JsValue::Smi(0)]);

        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_store_ic_shape_poly_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let mut table = ShapeTable::new();

        let mut obj1 = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);
        let mut obj2 = shaped_obj(&mut table, &["y", "x"], &[JsValue::Smi(0), JsValue::Smi(0)]);

        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        ic.store(&mut obj1, "x", JsValue::Smi(10), &mut fb, 0, &table)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(20), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(obj1.get_own_property("x"), Some(JsValue::Smi(10)));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(20)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    // ── Legacy PropertyStoreIc tests ────────────────────────────────────────

    #[test]
    fn test_store_ic_legacy_uninitialized_to_mono() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();

        ic.store(&mut obj, "x", JsValue::Smi(7), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(7)));
    }

    #[test]
    fn test_store_ic_legacy_mono_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();

        ic.store(&mut obj, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        let mut obj2 = JsObject::new();
        obj2.set_property("x", JsValue::Smi(0)).unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(99), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(99)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_store_ic_legacy_mono_to_poly_on_shape_mismatch() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(0)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(0)).unwrap();

        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_store_ic_legacy_poly_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        let mut obj1 = obj_with_props(&["x"]);
        let mut obj2 = obj_with_props(&["y", "x"]);

        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0, &table)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        ic.store(&mut obj1, "x", JsValue::Smi(10), &mut fb, 0, &table)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(20), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(obj1.get_own_property("x"), Some(JsValue::Smi(10)));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(20)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_store_ic_poly_to_mega() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(0)).unwrap();
            ic.store(&mut obj, "x", JsValue::Smi(i as i32), &mut fb, 0, &table)
                .unwrap();
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_store_ic_megamorphic_still_writes() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let table = ShapeTable::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(0)).unwrap();
            ic.store(&mut obj, "x", JsValue::Smi(i as i32), &mut fb, 0, &table)
                .unwrap();
        }

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();
        ic.store(&mut obj, "x", JsValue::Smi(77), &mut fb, 0, &table)
            .unwrap();
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(77)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    // ── CallIc: mono/poly/mega transitions ────────────────────────────────────

    fn make_function() -> JsValue {
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Opcode, encode};
        use crate::bytecode::feedback::FeedbackMetadata;

        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = BytecodeArray::new(
            encode(&instrs),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        JsValue::Function(Rc::new(ba))
    }

    #[test]
    fn test_call_ic_uninitialized_to_mono() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();
        let f = make_function();

        ic.record(&f, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(ic.callee_count(), 1);
    }

    #[test]
    fn test_call_ic_same_callee_stays_mono() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();
        let f = make_function();

        ic.record(&f, &mut fb, 0);
        ic.record(&f, &mut fb, 0);
        ic.record(&f, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(ic.callee_count(), 1);
    }

    #[test]
    fn test_call_ic_mono_to_poly_on_different_callee() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();

        let f1 = make_function();
        let f2 = make_function(); // distinct Rc allocation

        ic.record(&f1, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.record(&f2, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
        assert_eq!(ic.callee_count(), 2);
    }

    #[test]
    fn test_call_ic_poly_to_mega() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();

        let fns: Vec<JsValue> = (0..=POLY_MAX).map(|_| make_function()).collect();
        for f in &fns {
            ic.record(f, &mut fb, 0);
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
        assert_eq!(ic.callee_count(), 0); // cleared on megamorphic transition
    }

    #[test]
    fn test_call_ic_megamorphic_no_further_changes() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();

        let fns: Vec<JsValue> = (0..=POLY_MAX).map(|_| make_function()).collect();
        for f in &fns {
            ic.record(f, &mut fb, 0);
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));

        let extra = make_function();
        ic.record(&extra, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_call_ic_non_function_ignored() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();

        ic.record(&JsValue::Undefined, &mut fb, 0);
        ic.record(&JsValue::Smi(42), &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Uninitialized));
        assert_eq!(ic.callee_count(), 0);
    }

    // ── JsObject fast-index helpers ───────────────────────────────────────────

    #[test]
    fn test_get_fast_property_at_index() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(1)).unwrap();
        obj.set_property("b", JsValue::Smi(2)).unwrap();

        assert_eq!(obj.get_fast_property_at_index(0), Some(JsValue::Smi(1)));
        assert_eq!(obj.get_fast_property_at_index(1), Some(JsValue::Smi(2)));
        assert_eq!(obj.get_fast_property_at_index(2), None);
    }

    #[test]
    fn test_set_fast_property_at_index() {
        let mut obj = JsObject::new();
        obj.set_property("a", JsValue::Smi(0)).unwrap();
        obj.set_property("b", JsValue::Smi(0)).unwrap();

        assert!(obj.set_fast_property_at_index(1, JsValue::Smi(99)));
        assert_eq!(obj.get_fast_property_at_index(1), Some(JsValue::Smi(99)));

        // Out of range.
        assert!(!obj.set_fast_property_at_index(5, JsValue::Smi(0)));
    }

    // ── PropertyEntry shape matching ─────────────────────────────────────────

    #[test]
    fn test_property_entry_shape_id_matches() {
        let mut table = ShapeTable::new();
        let obj = shaped_obj(&mut table, &["x", "y"], &[JsValue::Smi(1), JsValue::Smi(2)]);
        let entry = make_property_entry(&obj, "x", &table).unwrap();
        assert!(entry.is_shape_based());
        assert!(entry.matches(&obj));
        assert_eq!(entry.fast_index, 0);
    }

    #[test]
    fn test_property_entry_shape_id_no_match_different_shape() {
        let mut table = ShapeTable::new();
        let obj1 = shaped_obj(&mut table, &["x", "y"], &[JsValue::Smi(1), JsValue::Smi(2)]);
        let entry = make_property_entry(&obj1, "x", &table).unwrap();

        // Different shape: {x, y, z}
        let obj2 = shaped_obj(
            &mut table,
            &["x", "y", "z"],
            &[JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)],
        );
        assert!(!entry.matches(&obj2));
    }

    #[test]
    fn test_property_entry_legacy_matches_same_shape() {
        let table = ShapeTable::new();
        let obj = obj_with_props(&["x", "y"]);
        let entry = make_property_entry(&obj, "x", &table).unwrap();
        assert!(!entry.is_shape_based());
        assert!(entry.matches(&obj));
    }

    #[test]
    fn test_property_entry_legacy_no_match_after_mutation() {
        let table = ShapeTable::new();
        let mut obj = obj_with_props(&["x", "y"]);
        let entry = make_property_entry(&obj, "x", &table).unwrap();
        assert!(entry.matches(&obj));

        obj.set_property("z", JsValue::Smi(0)).unwrap();
        assert!(!entry.matches(&obj));
    }

    #[test]
    fn test_property_entry_no_match_slow_mode() {
        use crate::objects::js_object::MAX_FAST_PROPERTIES;
        let _table = ShapeTable::new();
        let mut obj = JsObject::new();
        for i in 0..=MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(0)).unwrap();
        }
        assert!(!obj.is_fast_mode());

        let entry = PropertyEntry {
            shape_id: None,
            legacy_shape: Some(vec!["p0".to_string()]),
            fast_index: 0,
        };
        assert!(!entry.matches(&obj));
    }

    // ── IcStats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ic_stats_default() {
        let stats = IcStats::new();
        assert_eq!(stats.mono_hits, 0);
        assert_eq!(stats.poly_hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.transitions, 0);
        assert_eq!(stats.total(), 0);
        assert_eq!(stats.hit_ratio(), 0.0);
    }

    #[test]
    fn test_ic_stats_total_and_ratio() {
        let mut stats = IcStats::new();
        stats.mono_hits = 8;
        stats.poly_hits = 2;
        stats.misses = 10;
        assert_eq!(stats.total(), 20);
        assert!((stats.hit_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ic_stats_reset() {
        let mut stats = IcStats::new();
        stats.mono_hits = 5;
        stats.transitions = 3;
        stats.reset();
        assert_eq!(stats.total(), 0);
        assert_eq!(stats.transitions, 0);
    }

    #[test]
    fn test_load_ic_with_stats() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let mut table = ShapeTable::new();
        let mut stats = IcStats::new();

        let obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(42)]);

        // First load: miss (uninitialized → monomorphic).
        ic.load_with_stats(&obj, "x", &mut fb, 0, &table, &mut stats);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.transitions, 1);

        // Second load: mono hit.
        ic.load_with_stats(&obj, "x", &mut fb, 0, &table, &mut stats);
        assert_eq!(stats.mono_hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_store_ic_with_stats() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();
        let mut table = ShapeTable::new();
        let mut stats = IcStats::new();

        let mut obj = shaped_obj(&mut table, &["x"], &[JsValue::Smi(0)]);

        // First store: miss.
        ic.store_with_stats(
            &mut obj,
            "x",
            JsValue::Smi(1),
            &mut fb,
            0,
            &table,
            &mut stats,
        )
        .unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.transitions, 1);

        // Second store: mono hit.
        ic.store_with_stats(
            &mut obj,
            "x",
            JsValue::Smi(2),
            &mut fb,
            0,
            &table,
            &mut stats,
        )
        .unwrap();
        assert_eq!(stats.mono_hits, 1);
    }

    #[test]
    fn test_entry_count_accessors() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();
        let table = ShapeTable::new();

        assert_eq!(ic.entry_count(), 0);

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(1)).unwrap();
        ic.load(&obj, "x", &mut fb, 0, &table);
        assert_eq!(ic.entry_count(), 1);
    }
}

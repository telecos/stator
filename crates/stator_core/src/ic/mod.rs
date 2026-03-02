//! Inline-cache (IC) runtime for the Stator VM.
//!
//! This module implements the adaptive property-access and call-site optimisation
//! layer used by the bytecode interpreter.  Each operation is guarded by a
//! **feedback slot** in the function's [`FeedbackVector`]; the slot records
//! the current speculation state and the IC handler caches the observed
//! object shapes so that subsequent accesses can use a direct index instead of
//! a full descriptor scan.
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
//!   with an `O(n)` descriptor-name comparison and, on a hit, reads or writes
//!   the property directly by its cached fast index, skipping the descriptor
//!   scan entirely.
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
//!
//! # Example
//!
//! ```
//! use stator_core::ic::PropertyLoadIc;
//! use stator_core::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind, FeedbackVector};
//! use stator_core::objects::js_object::JsObject;
//! use stator_core::objects::value::JsValue;
//!
//! let meta = FeedbackMetadata::new(vec![FeedbackSlotKind::LoadProperty]);
//! let mut feedback = FeedbackVector::new(&meta);
//! let mut ic = PropertyLoadIc::new();
//!
//! let mut obj = JsObject::new();
//! obj.set_property("x", JsValue::Smi(42)).unwrap();
//!
//! // First load: Uninitialized → Monomorphic; returns 42.
//! let val = ic.load(&obj, "x", &mut feedback, 0);
//! assert_eq!(val, JsValue::Smi(42));
//!
//! // Second load: Monomorphic fast path; returns 42 without a descriptor scan.
//! let val2 = ic.load(&obj, "x", &mut feedback, 0);
//! assert_eq!(val2, JsValue::Smi(42));
//! ```

use std::rc::Rc;

use crate::bytecode::feedback::{FeedbackVector, InlineCacheState};
use crate::error::StatorResult;
use crate::objects::js_object::JsObject;
use crate::objects::value::JsValue;

/// Maximum number of distinct shapes cached before a slot goes megamorphic.
pub const POLY_MAX: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Internal shape helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A single cached entry for a named-property IC.
///
/// `shape` is the ordered list of property names present in the object's
/// hidden-class [`Map`](crate::objects::map::Map) when the entry was
/// recorded.  Two objects are considered to have the *same shape* when their
/// descriptor lists are the same length and the names appear in the same order.
///
/// `fast_index` is the zero-based position of the target property in the
/// fast-properties [`SmallVec`](smallvec::SmallVec) of the object, allowing a
/// direct `O(1)` read or write after the shape check passes.
#[derive(Debug, Clone)]
struct PropertyEntry {
    /// Ordered property-name fingerprint for the shape check.
    shape: Vec<String>,
    /// Fast-mode slot index of the cached property within this shape.
    fast_index: usize,
}

impl PropertyEntry {
    /// Return `true` if `obj` is in fast mode and has the exact same shape as
    /// this entry (same number of descriptors, same names in the same order).
    fn matches(&self, obj: &JsObject) -> bool {
        if !obj.is_fast_mode() {
            return false;
        }
        let descs = obj.map().descriptors();
        descs.len() == self.shape.len()
            && descs
                .iter()
                .zip(self.shape.iter())
                .all(|(d, s)| d.key() == s)
    }
}

/// Build a [`PropertyEntry`] for `key` on a fast-mode `obj`.
///
/// Returns `None` when `obj` is in slow (dictionary) mode or when `key` does
/// not appear in the fast-mode descriptor table.
fn make_property_entry(obj: &JsObject, key: &str) -> Option<PropertyEntry> {
    if !obj.is_fast_mode() {
        return None;
    }
    let descs = obj.map().descriptors();
    let fast_index = descs.iter().position(|d| d.key() == key)?;
    let shape = descs.iter().map(|d| d.key().to_string()).collect();
    Some(PropertyEntry { shape, fast_index })
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
/// On a shape hit the property value is read directly by its cached fast
/// index, bypassing the linear descriptor scan performed by
/// [`JsObject::get_own_property`].
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

    /// Load property `key` from `obj`, using the IC fast path when possible.
    ///
    /// Checks `feedback[slot]` for the current speculation state, attempts a
    /// fast read on a shape hit, and falls back to
    /// [`JsObject::get_property`] on a miss.  Transitions the feedback slot
    /// forward whenever a new shape is observed.
    pub fn load(
        &mut self,
        obj: &JsObject,
        key: &str,
        feedback: &mut FeedbackVector,
        slot: u32,
    ) -> JsValue {
        match feedback.get_state(slot) {
            None => {
                // Slot index out of range: safe fallback, no state change.
                obj.get_property(key)
            }

            Some(InlineCacheState::Uninitialized) => {
                // First execution: perform the runtime lookup, record an entry
                // if the receiver is in fast mode, then go monomorphic.
                let val = obj.get_property(key);
                if let Some(entry) = make_property_entry(obj, key) {
                    self.entries.push(entry);
                    feedback.transition(slot, InlineCacheState::Monomorphic);
                }
                val
            }

            Some(InlineCacheState::Monomorphic) => {
                if let Some(entry) = self.entries.first() {
                    if entry.matches(obj) {
                        // ── Monomorphic fast path ─────────────────────────
                        return obj
                            .get_fast_property_at_index(entry.fast_index)
                            .unwrap_or(JsValue::Undefined);
                    }
                    // Shape mismatch: record the new shape (if unique) and
                    // advance to polymorphic.
                    if let Some(new_entry) = make_property_entry(obj, key)
                        && !self.entries.iter().any(|e| e.shape == new_entry.shape)
                    {
                        self.entries.push(new_entry);
                    }
                    feedback.transition(slot, InlineCacheState::Polymorphic);
                }
                obj.get_property(key)
            }

            Some(InlineCacheState::Polymorphic) => {
                // Search cached entries.
                if let Some(entry) = self.entries.iter().find(|e| e.matches(obj)) {
                    // ── Polymorphic fast path ─────────────────────────────
                    return obj
                        .get_fast_property_at_index(entry.fast_index)
                        .unwrap_or(JsValue::Undefined);
                }
                // Miss: try to add the new shape.
                if let Some(new_entry) = make_property_entry(obj, key)
                    && !self.entries.iter().any(|e| e.shape == new_entry.shape)
                {
                    if self.entries.len() < POLY_MAX {
                        self.entries.push(new_entry);
                    } else {
                        // Exceeded POLY_MAX distinct shapes: go megamorphic.
                        self.entries.clear();
                        feedback.transition(slot, InlineCacheState::Megamorphic);
                    }
                }
                obj.get_property(key)
            }

            Some(InlineCacheState::Megamorphic) => {
                // No speculation: always use the full runtime lookup.
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
/// [`JsObject::set_property`].
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

    /// Store `value` into property `key` on `obj`, using the IC fast path
    /// when possible.
    ///
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
    ) -> StatorResult<()> {
        match feedback.get_state(slot) {
            None => {
                // Slot out of range: safe fallback.
                obj.set_property(key, value)
            }

            Some(InlineCacheState::Uninitialized) => {
                // First execution: runtime store, then record the post-write
                // shape so the fast path can be used next time.
                obj.set_property(key, value)?;
                if let Some(entry) = make_property_entry(obj, key) {
                    self.entries.push(entry);
                    feedback.transition(slot, InlineCacheState::Monomorphic);
                }
                Ok(())
            }

            Some(InlineCacheState::Monomorphic) => {
                // Clone the entry to release the immutable borrow before the
                // mutable `set_property` / `set_fast_property_at_index` call.
                let entry_opt = self.entries.first().cloned();
                if let Some(entry) = entry_opt {
                    if entry.matches(obj) {
                        // ── Monomorphic fast path ─────────────────────────
                        // Shape is verified; index is guaranteed in-bounds.
                        obj.set_fast_property_at_index(entry.fast_index, value);
                        return Ok(());
                    }
                    // Shape mismatch: runtime store.
                    obj.set_property(key, value)?;
                    // Record the new post-write shape if genuinely different.
                    if let Some(new_entry) = make_property_entry(obj, key)
                        && !self.entries.iter().any(|e| e.shape == new_entry.shape)
                    {
                        self.entries.push(new_entry);
                        feedback.transition(slot, InlineCacheState::Polymorphic);
                    }
                } else {
                    // No entry was recorded (slow-mode object on first call).
                    obj.set_property(key, value)?;
                    if let Some(entry) = make_property_entry(obj, key) {
                        self.entries.push(entry);
                    }
                }
                Ok(())
            }

            Some(InlineCacheState::Polymorphic) => {
                let entry_opt = self.entries.iter().find(|e| e.matches(obj)).cloned();
                if let Some(entry) = entry_opt {
                    // ── Polymorphic fast path ─────────────────────────────
                    // Shape is verified; index is guaranteed in-bounds.
                    obj.set_fast_property_at_index(entry.fast_index, value);
                    return Ok(());
                }
                // Miss: runtime store.
                obj.set_property(key, value)?;
                // Add the new post-write shape if it is genuinely new.
                if let Some(new_entry) = make_property_entry(obj, key)
                    && !self.entries.iter().any(|e| e.shape == new_entry.shape)
                {
                    if self.entries.len() < POLY_MAX {
                        self.entries.push(new_entry);
                    } else {
                        // Too many shapes: go megamorphic.
                        self.entries.clear();
                        feedback.transition(slot, InlineCacheState::Megamorphic);
                    }
                }
                Ok(())
            }

            Some(InlineCacheState::Megamorphic) => {
                // No speculation: full runtime store.
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
/// Callee identity is determined by the raw pointer address of the
/// [`Rc`]-managed [`BytecodeArray`](crate::bytecode::bytecode_array::BytecodeArray):
/// two `JsValue::Function` values are considered the *same* callee iff they
/// share the same underlying `Rc` allocation.
#[derive(Debug, Default)]
pub struct CallIc {
    /// Raw `Rc` pointer addresses of observed callees.
    callee_ids: Vec<usize>,
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
                if !self.callee_ids.contains(&id) {
                    self.callee_ids.push(id);
                    feedback.transition(slot, InlineCacheState::Polymorphic);
                }
            }

            Some(InlineCacheState::Polymorphic) => {
                if !self.callee_ids.contains(&id) {
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

/// Extract a stable numeric identity for a `JsValue::Function` callee.
///
/// Returns the raw pointer of the `Rc<BytecodeArray>` as a `usize` so that
/// two `Function` values backed by the same `Rc` allocation compare equal.
/// Returns `None` for any non-`Function` value.
fn callee_identity(callee: &JsValue) -> Option<usize> {
    if let JsValue::Function(rc) = callee {
        Some(Rc::as_ptr(rc) as usize)
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

    // ── Helpers ──────────────────────────────────────────────────────────────

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

    /// Build an object with the given properties set to `Smi(0)`.
    fn obj_with_props(keys: &[&str]) -> JsObject {
        let mut obj = JsObject::new();
        for key in keys {
            obj.set_property(key, JsValue::Smi(0)).unwrap();
        }
        obj
    }

    // ── PropertyLoadIc: mono/poly/mega transitions ────────────────────────────

    #[test]
    fn test_load_ic_uninitialized_to_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(7)).unwrap();

        // First call: Uninitialized → Monomorphic.
        let val = ic.load(&obj, "x", &mut fb, 0);
        assert_eq!(val, JsValue::Smi(7));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_mono_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(42)).unwrap();

        // Warm up.
        ic.load(&obj, "x", &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        // Second call: hits the monomorphic fast path.
        let val = ic.load(&obj, "x", &mut fb, 0);
        assert_eq!(val, JsValue::Smi(42));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_mono_to_poly_on_shape_mismatch() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        // Shape 1: {x}
        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(1)).unwrap();

        // Shape 2: {y, x}
        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(2)).unwrap();

        ic.load(&obj1, "x", &mut fb, 0); // Uninitialized → Monomorphic
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.load(&obj2, "x", &mut fb, 0); // Monomorphic → Polymorphic
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_load_ic_poly_fast_path() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(10)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(20)).unwrap();

        // Warm up both shapes.
        ic.load(&obj1, "x", &mut fb, 0);
        ic.load(&obj2, "x", &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        // Fast path should work for both shapes.
        assert_eq!(ic.load(&obj1, "x", &mut fb, 0), JsValue::Smi(10));
        assert_eq!(ic.load(&obj2, "x", &mut fb, 0), JsValue::Smi(20));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_load_ic_poly_to_mega() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        // POLY_MAX + 1 distinct shapes to trigger Megamorphic.
        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            // Give each object a unique prefix of extra properties so that
            // each one has a distinct shape.
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(i as i32)).unwrap();
            ic.load(&obj, "x", &mut fb, 0);
        }

        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_load_ic_megamorphic_always_runtime() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        // Drive to Megamorphic.
        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(i as i32)).unwrap();
            ic.load(&obj, "x", &mut fb, 0);
        }

        // In megamorphic state the IC still returns the correct value via the
        // runtime path.
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(99)).unwrap();
        let val = ic.load(&obj, "x", &mut fb, 0);
        assert_eq!(val, JsValue::Smi(99));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_load_ic_missing_property_returns_undefined() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        let obj = JsObject::new();
        let val = ic.load(&obj, "missing", &mut fb, 0);
        assert_eq!(val, JsValue::Undefined);
    }

    #[test]
    fn test_load_ic_same_shape_stays_mono() {
        let mut fb = load_feedback();
        let mut ic = PropertyLoadIc::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(1)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("x", JsValue::Smi(2)).unwrap();

        ic.load(&obj1, "x", &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        // Same shape: still monomorphic.
        ic.load(&obj2, "x", &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_load_ic_out_of_range_slot() {
        let mut fb = load_feedback(); // only slot 0 exists
        let mut ic = PropertyLoadIc::new();
        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(5)).unwrap();

        // Slot 99 does not exist; the IC should safely fall back without
        // panicking.
        let val = ic.load(&obj, "x", &mut fb, 99);
        assert_eq!(val, JsValue::Smi(5));
    }

    // ── PropertyStoreIc: mono/poly/mega transitions ───────────────────────────

    #[test]
    fn test_store_ic_uninitialized_to_mono() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();

        ic.store(&mut obj, "x", JsValue::Smi(7), &mut fb, 0)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
        assert_eq!(obj.get_own_property("x"), Some(JsValue::Smi(7)));
    }

    #[test]
    fn test_store_ic_mono_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();

        // Warm up: Uninitialized → Monomorphic.
        ic.store(&mut obj, "x", JsValue::Smi(1), &mut fb, 0)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        // Second store with same-shape object: fast path.
        let mut obj2 = JsObject::new();
        obj2.set_property("x", JsValue::Smi(0)).unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(99), &mut fb, 0)
            .unwrap();
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(99)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));
    }

    #[test]
    fn test_store_ic_mono_to_poly_on_shape_mismatch() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        let mut obj1 = JsObject::new();
        obj1.set_property("x", JsValue::Smi(0)).unwrap();

        let mut obj2 = JsObject::new();
        obj2.set_property("y", JsValue::Smi(0)).unwrap();
        obj2.set_property("x", JsValue::Smi(0)).unwrap();

        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Monomorphic));

        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_store_ic_poly_fast_path() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        let mut obj1 = obj_with_props(&["x"]);
        let mut obj2 = obj_with_props(&["y", "x"]);

        // Warm up both shapes.
        ic.store(&mut obj1, "x", JsValue::Smi(1), &mut fb, 0)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(2), &mut fb, 0)
            .unwrap();
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));

        // Fast path for both shapes.
        ic.store(&mut obj1, "x", JsValue::Smi(10), &mut fb, 0)
            .unwrap();
        ic.store(&mut obj2, "x", JsValue::Smi(20), &mut fb, 0)
            .unwrap();
        assert_eq!(obj1.get_own_property("x"), Some(JsValue::Smi(10)));
        assert_eq!(obj2.get_own_property("x"), Some(JsValue::Smi(20)));
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Polymorphic));
    }

    #[test]
    fn test_store_ic_poly_to_mega() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(0)).unwrap();
            ic.store(&mut obj, "x", JsValue::Smi(i as i32), &mut fb, 0)
                .unwrap();
        }

        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_store_ic_megamorphic_still_writes() {
        let mut fb = store_feedback();
        let mut ic = PropertyStoreIc::new();

        // Drive to Megamorphic.
        for i in 0..=(POLY_MAX as u32) {
            let mut obj = JsObject::new();
            for j in 0..i {
                obj.set_property(&format!("p{j}"), JsValue::Smi(0)).unwrap();
            }
            obj.set_property("x", JsValue::Smi(0)).unwrap();
            ic.store(&mut obj, "x", JsValue::Smi(i as i32), &mut fb, 0)
                .unwrap();
        }

        let mut obj = JsObject::new();
        obj.set_property("x", JsValue::Smi(0)).unwrap();
        ic.store(&mut obj, "x", JsValue::Smi(77), &mut fb, 0)
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

        // Drive to megamorphic; keep all functions alive to prevent the
        // allocator from reusing pointer addresses across loop iterations.
        let fns: Vec<JsValue> = (0..=POLY_MAX).map(|_| make_function()).collect();
        for f in &fns {
            ic.record(f, &mut fb, 0);
        }
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));

        // Further records should not change state or panic.
        let extra = make_function();
        ic.record(&extra, &mut fb, 0);
        assert_eq!(fb.get_state(0), Some(InlineCacheState::Megamorphic));
    }

    #[test]
    fn test_call_ic_non_function_ignored() {
        let mut fb = call_feedback();
        let mut ic = CallIc::new();

        // Non-function callees should be silently ignored.
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
    fn test_property_entry_matches_same_shape() {
        let mut obj = obj_with_props(&["x", "y"]);
        let entry = make_property_entry(&obj, "x").unwrap();
        assert!(entry.matches(&obj));

        // Add a property to change the shape.
        obj.set_property("z", JsValue::Smi(0)).unwrap();
        assert!(!entry.matches(&obj));
    }

    #[test]
    fn test_property_entry_no_match_slow_mode() {
        // Force slow mode by adding more than MAX_FAST_PROPERTIES properties.
        use crate::objects::js_object::MAX_FAST_PROPERTIES;
        let mut obj = JsObject::new();
        for i in 0..=MAX_FAST_PROPERTIES {
            obj.set_property(&format!("p{i}"), JsValue::Smi(0)).unwrap();
        }
        assert!(!obj.is_fast_mode());

        let entry = PropertyEntry {
            shape: vec!["p0".to_string()],
            fast_index: 0,
        };
        assert!(!entry.matches(&obj));
    }
}

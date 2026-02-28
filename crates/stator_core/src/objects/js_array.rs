//! JavaScript Array object with element-kind tracking.
//!
//! This module provides [`JsArray`], which wraps a [`JsObject`] and adds
//! array-specific behaviour modelled after V8's element-kind optimisations.
//!
//! # Element kinds
//!
//! Every `JsArray` maintains an [`ElementKind`] that describes the most general
//! element type seen so far and whether any "holes" (sparse slots) exist.  The
//! lattice of possible kinds is:
//!
//! ```text
//! PackedSmi  →  PackedDouble  →  PackedElements
//!     ↓               ↓                 ↓
//! HoleSmi   →  HoleDouble   →  HoleElements
//! ```
//!
//! Transitions are **monotone**: once an array moves to a more general kind it
//! never narrows back.  "Packed" means every slot `0..length` holds a concrete
//! value; "Holey" means at least one slot contains a `JsValue::Undefined`
//! hole.

use crate::gc::trace::{Trace, Tracer};
use crate::objects::js_object::JsObject;
use crate::objects::map::InstanceType;
use crate::objects::value::JsValue;

/// Classifies the most-general element type and hole-presence seen in a
/// [`JsArray`] at any point in its lifetime.
///
/// Variants are ordered from most-specific (`PackedSmi`) to most-general
/// (`HoleElements`).  Once an array transitions to a more general kind it
/// never returns to a narrower one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    /// All elements are `Smi` integers; no holes.
    PackedSmi,
    /// All elements are numeric (`Smi` or `HeapNumber`); no holes.
    PackedDouble,
    /// Elements may be any `JsValue`; no holes.
    PackedElements,
    /// All elements are `Smi` integers; at least one hole exists.
    HoleSmi,
    /// All elements are numeric (`Smi` or `HeapNumber`); at least one hole exists.
    HoleDouble,
    /// Elements may be any `JsValue`; at least one hole exists.
    HoleElements,
}

impl ElementKind {
    /// Returns the holey counterpart of this kind.
    ///
    /// If the kind is already holey, it is returned unchanged.
    fn to_holey(self) -> Self {
        match self {
            Self::PackedSmi | Self::HoleSmi => Self::HoleSmi,
            Self::PackedDouble | Self::HoleDouble => Self::HoleDouble,
            Self::PackedElements | Self::HoleElements => Self::HoleElements,
        }
    }

    /// Returns `true` if this kind permits holes.
    pub fn is_holey(self) -> bool {
        matches!(self, Self::HoleSmi | Self::HoleDouble | Self::HoleElements)
    }

    /// Returns this kind widened to accommodate `value`.
    ///
    /// The hole-state is preserved; only the type rank can increase:
    ///
    /// | value type | packed rank change | holey rank change |
    /// |---|---|---|
    /// | `Smi` | no change | no change |
    /// | `HeapNumber` | `PackedSmi → PackedDouble` | `HoleSmi → HoleDouble` |
    /// | anything else | → `PackedElements` | → `HoleElements` |
    fn widen_for_value(self, value: &JsValue) -> Self {
        match (self, value) {
            // Smi fits in any kind — no widening needed.
            (k, JsValue::Smi(_)) => k,
            // HeapNumber: widen Smi kinds to Double; Double and above unchanged.
            (Self::PackedSmi, JsValue::HeapNumber(_)) => Self::PackedDouble,
            (Self::HoleSmi, JsValue::HeapNumber(_)) => Self::HoleDouble,
            (k, JsValue::HeapNumber(_)) => k,
            // Any other value: widen to Elements (preserving hole-state).
            (Self::PackedSmi | Self::PackedDouble | Self::PackedElements, _) => {
                Self::PackedElements
            }
            _ => Self::HoleElements,
        }
    }
}

/// A JavaScript Array per ECMAScript §10.4.2.
///
/// `JsArray` wraps a [`JsObject`] and layers on top of it:
///
/// * **Length semantics** — `length` reflects the highest index assigned plus
///   one, mirroring the ECMAScript `Array.prototype.length` invariant.
/// * **Element-kind tracking** — every mutation updates the [`ElementKind`]
///   via monotone transitions (see the [module-level docs][self]).
pub struct JsArray {
    /// Underlying ordinary object (provides named-property and element storage).
    object: JsObject,
    /// Most-general element kind seen so far.
    element_kind: ElementKind,
}

impl JsArray {
    /// Creates an empty array with [`ElementKind::PackedSmi`].
    pub fn new() -> Self {
        Self {
            object: JsObject::new_with_instance_type(InstanceType::JsArray),
            element_kind: ElementKind::PackedSmi,
        }
    }

    /// Returns the current [`ElementKind`] of this array.
    pub fn element_kind(&self) -> ElementKind {
        self.element_kind
    }

    /// Returns the number of elements in the array (ECMAScript `length`).
    ///
    /// This equals the highest index ever written plus one; sparse assignments
    /// that create holes are included.
    pub fn length(&self) -> u32 {
        self.object.elements_length() as u32
    }

    /// Returns the element at `index`.
    ///
    /// Returns [`JsValue::Undefined`] if `index` is out of bounds.
    pub fn get(&self, index: u32) -> JsValue {
        self.object.get_element(index as usize)
    }

    /// Sets the element at `index`, widening the element kind as necessary.
    ///
    /// If `index > length()`, the intermediate slots become `JsValue::Undefined`
    /// holes and the kind transitions to a holey variant.
    pub fn set(&mut self, index: u32, value: JsValue) {
        let current_len = self.object.elements_length() as u32;
        // A gap between the current last index and the new index creates holes.
        if index > current_len {
            self.element_kind = self.element_kind.to_holey();
        }
        self.element_kind = self.element_kind.widen_for_value(&value);
        self.object.set_element(index as usize, value);
    }

    /// Appends `value` to the end of the array and returns the new length.
    ///
    /// The element kind is widened if necessary.
    pub fn push(&mut self, value: JsValue) -> u32 {
        let idx = self.object.elements_length();
        self.element_kind = self.element_kind.widen_for_value(&value);
        self.object.set_element(idx, value);
        self.object.elements_length() as u32
    }

    /// Removes and returns the last element of the array.
    ///
    /// Returns [`JsValue::Undefined`] if the array is empty.
    ///
    /// The element kind is **not** narrowed after a pop — transitions are
    /// monotone.
    pub fn pop(&mut self) -> JsValue {
        let len = self.object.elements_length();
        if len == 0 {
            return JsValue::Undefined;
        }
        let value = self.object.get_element(len - 1);
        self.object.truncate_elements(len - 1);
        value
    }

    /// Returns a reference to the underlying [`JsObject`].
    pub fn as_object(&self) -> &JsObject {
        &self.object
    }

    /// Returns a mutable reference to the underlying [`JsObject`].
    pub fn as_object_mut(&mut self) -> &mut JsObject {
        &mut self.object
    }
}

impl Default for JsArray {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace for JsArray {
    /// Delegate tracing to the underlying [`JsObject`].
    ///
    /// All GC-reachable heap references (named properties, indexed elements,
    /// and the prototype chain) are owned by the inner object.
    fn trace(&self, tracer: &mut Tracer) {
        self.object.trace(tracer);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Element kind transitions ──────────────────────────────────────────────

    #[test]
    fn test_new_array_starts_packed_smi() {
        let arr = JsArray::new();
        assert_eq!(arr.element_kind(), ElementKind::PackedSmi);
    }

    #[test]
    fn test_push_smi_stays_packed_smi() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::Smi(2));
        assert_eq!(arr.element_kind(), ElementKind::PackedSmi);
    }

    #[test]
    fn test_push_heap_number_transitions_to_packed_double() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::HeapNumber(3.14));
        assert_eq!(arr.element_kind(), ElementKind::PackedDouble);
    }

    #[test]
    fn test_push_string_transitions_to_packed_elements() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::String("hello".to_string()));
        assert_eq!(arr.element_kind(), ElementKind::PackedElements);
    }

    #[test]
    fn test_push_double_then_string_transitions_to_packed_elements() {
        let mut arr = JsArray::new();
        arr.push(JsValue::HeapNumber(1.5));
        assert_eq!(arr.element_kind(), ElementKind::PackedDouble);
        arr.push(JsValue::String("x".to_string()));
        assert_eq!(arr.element_kind(), ElementKind::PackedElements);
    }

    #[test]
    fn test_sparse_set_transitions_to_holey_smi() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1)); // length = 1
        arr.set(3, JsValue::Smi(2)); // gap at index 1 and 2 → holey
        assert_eq!(arr.element_kind(), ElementKind::HoleSmi);
    }

    #[test]
    fn test_sparse_set_with_double_transitions_to_hole_double() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1)); // length = 1
        arr.set(3, JsValue::HeapNumber(2.5)); // gap → holey, value → double
        assert_eq!(arr.element_kind(), ElementKind::HoleDouble);
    }

    #[test]
    fn test_sparse_set_with_string_transitions_to_hole_elements() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.set(5, JsValue::String("z".to_string()));
        assert_eq!(arr.element_kind(), ElementKind::HoleElements);
    }

    #[test]
    fn test_transition_never_narrows_after_pop() {
        let mut arr = JsArray::new();
        arr.push(JsValue::HeapNumber(1.0)); // → PackedDouble
        arr.pop();
        // Kind stays PackedDouble even though array is empty.
        assert_eq!(arr.element_kind(), ElementKind::PackedDouble);
    }

    #[test]
    fn test_transition_packed_to_holey_after_sparse_then_stays_holey() {
        let mut arr = JsArray::new();
        arr.set(2, JsValue::Smi(5)); // indices 0,1 become holes
        assert_eq!(arr.element_kind(), ElementKind::HoleSmi);
        // Pushing more Smis does not narrow back to packed.
        arr.push(JsValue::Smi(9));
        assert_eq!(arr.element_kind(), ElementKind::HoleSmi);
    }

    // ── Length semantics ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_array_length_is_zero() {
        let arr = JsArray::new();
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_push_increments_length() {
        let mut arr = JsArray::new();
        assert_eq!(arr.push(JsValue::Smi(1)), 1);
        assert_eq!(arr.push(JsValue::Smi(2)), 2);
        assert_eq!(arr.push(JsValue::Smi(3)), 3);
        assert_eq!(arr.length(), 3);
    }

    #[test]
    fn test_pop_decrements_length() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(10));
        arr.push(JsValue::Smi(20));
        arr.pop();
        assert_eq!(arr.length(), 1);
    }

    #[test]
    fn test_pop_empty_returns_undefined_and_length_stays_zero() {
        let mut arr = JsArray::new();
        let v = arr.pop();
        assert_eq!(v, JsValue::Undefined);
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_sparse_set_updates_length() {
        let mut arr = JsArray::new();
        arr.set(4, JsValue::Smi(1));
        assert_eq!(arr.length(), 5); // indices 0-3 are holes, index 4 is set
    }

    #[test]
    fn test_set_within_bounds_does_not_change_length() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::Smi(2));
        arr.push(JsValue::Smi(3));
        arr.set(1, JsValue::Smi(99));
        assert_eq!(arr.length(), 3);
    }

    // ── Push and pop ──────────────────────────────────────────────────────────

    #[test]
    fn test_push_and_pop_roundtrip() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(42));
        arr.push(JsValue::Smi(7));
        assert_eq!(arr.pop(), JsValue::Smi(7));
        assert_eq!(arr.pop(), JsValue::Smi(42));
        assert_eq!(arr.pop(), JsValue::Undefined);
    }

    #[test]
    fn test_push_after_pop_reuses_slot() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::Smi(2));
        arr.pop();
        arr.push(JsValue::Smi(3));
        assert_eq!(arr.length(), 2);
        assert_eq!(arr.get(1), JsValue::Smi(3));
    }

    // ── Index access ──────────────────────────────────────────────────────────

    #[test]
    fn test_get_returns_correct_elements() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(10));
        arr.push(JsValue::Smi(20));
        arr.push(JsValue::Smi(30));
        assert_eq!(arr.get(0), JsValue::Smi(10));
        assert_eq!(arr.get(1), JsValue::Smi(20));
        assert_eq!(arr.get(2), JsValue::Smi(30));
    }

    #[test]
    fn test_get_out_of_bounds_returns_undefined() {
        let arr = JsArray::new();
        assert_eq!(arr.get(0), JsValue::Undefined);
        assert_eq!(arr.get(100), JsValue::Undefined);
    }

    #[test]
    fn test_set_updates_existing_element() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(1));
        arr.push(JsValue::Smi(2));
        arr.set(0, JsValue::Smi(99));
        assert_eq!(arr.get(0), JsValue::Smi(99));
        assert_eq!(arr.get(1), JsValue::Smi(2));
    }

    #[test]
    fn test_sparse_holes_read_as_undefined() {
        let mut arr = JsArray::new();
        arr.set(3, JsValue::Smi(7));
        assert_eq!(arr.get(0), JsValue::Undefined);
        assert_eq!(arr.get(1), JsValue::Undefined);
        assert_eq!(arr.get(2), JsValue::Undefined);
        assert_eq!(arr.get(3), JsValue::Smi(7));
    }

    // ── Default / as_object ───────────────────────────────────────────────────

    #[test]
    fn test_default_equals_new() {
        let arr: JsArray = JsArray::default();
        assert_eq!(arr.element_kind(), ElementKind::PackedSmi);
        assert_eq!(arr.length(), 0);
    }

    #[test]
    fn test_as_object_reflects_elements() {
        let mut arr = JsArray::new();
        arr.push(JsValue::Smi(5));
        assert_eq!(arr.as_object().get_element(0), JsValue::Smi(5));
    }
}

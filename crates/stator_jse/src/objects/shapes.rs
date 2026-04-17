//! V8-style hidden-class (shape) system with transition trees and descriptor
//! arrays.
//!
//! # Overview
//!
//! A [`Shape`] is an ordered property-layout descriptor: it records which named
//! properties an object has, in what order they were added, and at which slot
//! offset each property's value lives.  Objects that receive the same sequence
//! of property additions share a single `Shape`, enabling the inline-cache (IC)
//! runtime to cache a property's slot offset keyed on [`ShapeId`] alone.
//!
//! # Transition trees
//!
//! Shapes form a tree rooted at an empty "root" shape.  Adding a property to an
//! object whose current shape is *S* produces a **transition** from *S* to a
//! child shape *S′* that extends the descriptor array by one entry.  If a
//! transition for the same `(name, attributes)` pair already exists, the
//! existing child is reused — this is the deduplication mechanism.
//!
//! # `ShapeTable`
//!
//! [`ShapeTable`] is the global registry that owns every [`Shape`] allocated
//! during the lifetime of the engine.  Shapes are identified by [`ShapeId`]
//! (a `u32` index into the table's backing `Vec`), so property-access fast
//! paths compare a single integer instead of chasing pointers.

use smallvec::SmallVec;

use crate::objects::map::PropertyAttributes;

// ─────────────────────────────────────────────────────────────────────────────
// ShapeId
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight handle to a [`Shape`] inside a [`ShapeTable`].
///
/// Internally this is a `u32` index into the table's backing `Vec<Shape>`,
/// so copying, comparing, and hashing are all O(1) and pointer-free.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeId(u32);

impl ShapeId {
    /// Returns the raw `u32` index.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constness
// ─────────────────────────────────────────────────────────────────────────────

/// Whether a property slot is considered constant (never reassigned after
/// initial definition) or mutable.
///
/// Inline caches can specialize more aggressively on `Const` properties
/// because their value is known not to change.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Constness {
    /// The property has been assigned exactly once and is treated as constant.
    Const,
    /// The property may be reassigned at any time.
    Mutable,
}

// ─────────────────────────────────────────────────────────────────────────────
// ShapeDescriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in a [`Shape`]'s descriptor array.
///
/// Each descriptor records the property's name, its slot offset (field index)
/// in the object's in-object / overflow property storage, the ECMAScript
/// attribute flags, and whether the property is considered constant.
#[derive(Clone, Debug)]
pub struct ShapeDescriptor {
    /// Property name.
    key: String,
    /// Zero-based slot index into the object's property storage.
    field_index: u32,
    /// ECMAScript property attributes (writable, enumerable, configurable).
    attributes: PropertyAttributes,
    /// Whether the engine considers this slot constant.
    constness: Constness,
}

impl ShapeDescriptor {
    /// Creates a new descriptor.
    pub fn new(
        key: impl Into<String>,
        field_index: u32,
        attributes: PropertyAttributes,
        constness: Constness,
    ) -> Self {
        Self {
            key: key.into(),
            field_index,
            attributes,
            constness,
        }
    }

    /// Returns the property name.
    #[inline]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the zero-based slot index.
    #[inline]
    pub fn field_index(&self) -> u32 {
        self.field_index
    }

    /// Returns the property attribute flags.
    #[inline]
    pub fn attributes(&self) -> PropertyAttributes {
        self.attributes
    }

    /// Returns the constness of this property slot.
    #[inline]
    pub fn constness(&self) -> Constness {
        self.constness
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShapeTransition
// ─────────────────────────────────────────────────────────────────────────────

/// A single edge in the shape transition tree.
///
/// When a property named `property_name` with `attributes` is added to an
/// object whose current shape is the parent, the object transitions to the
/// child shape identified by `target`.
#[derive(Clone, Debug)]
pub struct ShapeTransition {
    /// The property name that triggers this transition.
    property_name: String,
    /// The attribute flags of the added property.
    attributes: PropertyAttributes,
    /// The child [`ShapeId`] reached by this transition.
    target: ShapeId,
}

impl ShapeTransition {
    /// Returns the property name of this transition edge.
    #[inline]
    pub fn property_name(&self) -> &str {
        &self.property_name
    }

    /// Returns the attribute flags of this transition edge.
    #[inline]
    pub fn attributes(&self) -> PropertyAttributes {
        self.attributes
    }

    /// Returns the target [`ShapeId`].
    #[inline]
    pub fn target(&self) -> ShapeId {
        self.target
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of transition edges stored inline before spilling to the
/// heap.  Empirically most shapes have ≤ 4 transitions.
const TRANSITION_INLINE_CAP: usize = 4;

/// A hidden class describing the property layout of a JavaScript object.
///
/// Every `Shape` holds:
///
/// * A **descriptor array** — the flattened, ordered list of
///   [`ShapeDescriptor`]s for every property present in this shape (including
///   those inherited from ancestor shapes in the transition tree).
/// * A **transition table** — outgoing edges to child shapes, keyed on
///   `(property_name, attributes)`.
/// * A link to the **parent** shape (if any).
///
/// Two objects are said to *share a shape* when their `ShapeId` values are
/// equal — meaning they have exactly the same property names, in the same
/// insertion order, with the same attribute flags.
pub struct Shape {
    /// This shape's unique identifier inside the [`ShapeTable`].
    id: ShapeId,
    /// Parent shape in the transition tree, or `None` for the root.
    parent: Option<ShapeId>,
    /// Flattened descriptor array (includes all ancestor descriptors).
    descriptors: Vec<ShapeDescriptor>,
    /// Total number of properties described by this shape.
    property_count: u16,
    /// Instance size hint: suggested number of in-object property slots.
    instance_size: u16,
    /// Number of properties that fit in fixed in-object slots (before
    /// overflow to a separate property array).
    n_in_object_properties: u8,
    /// Outgoing transitions to child shapes.
    transitions: SmallVec<[ShapeTransition; TRANSITION_INLINE_CAP]>,
}

impl Shape {
    /// Returns this shape's [`ShapeId`].
    #[inline]
    pub fn id(&self) -> ShapeId {
        self.id
    }

    /// Returns the parent shape, or `None` for the root shape.
    #[inline]
    pub fn parent(&self) -> Option<ShapeId> {
        self.parent
    }

    /// Returns the full descriptor array for this shape.
    #[inline]
    pub fn descriptors(&self) -> &[ShapeDescriptor] {
        &self.descriptors
    }

    /// Returns the total number of properties described by this shape.
    #[inline]
    pub fn property_count(&self) -> u16 {
        self.property_count
    }

    /// Returns the suggested in-object instance size (number of slots).
    #[inline]
    pub fn instance_size(&self) -> u16 {
        self.instance_size
    }

    /// Returns how many properties fit in the fixed in-object slots.
    #[inline]
    pub fn n_in_object_properties(&self) -> u8 {
        self.n_in_object_properties
    }

    /// Returns the outgoing transition edges.
    #[inline]
    pub fn transitions(&self) -> &[ShapeTransition] {
        &self.transitions
    }

    /// Looks up a property by name in the descriptor array.
    ///
    /// Returns the matching [`ShapeDescriptor`] if found, or `None`.
    pub fn lookup(&self, key: &str) -> Option<&ShapeDescriptor> {
        self.descriptors.iter().find(|d| d.key == key)
    }

    /// Finds an existing transition for `(name, attrs)`, if one exists.
    fn find_transition(&self, name: &str, attrs: PropertyAttributes) -> Option<ShapeId> {
        self.transitions
            .iter()
            .find(|t| t.property_name == name && t.attributes == attrs)
            .map(|t| t.target)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShapeTable
// ─────────────────────────────────────────────────────────────────────────────

/// Default number of in-object property slots for newly created objects.
const DEFAULT_IN_OBJECT_SLOTS: u8 = 4;

/// Global registry of all [`Shape`]s.
///
/// Shapes are arena-allocated into a `Vec<Shape>` and identified by
/// [`ShapeId`] (a `u32` index).  The table is created once per engine
/// instance and owns the entire transition tree.
///
/// # Deduplication
///
/// Adding a property to a shape first checks the parent's transition table.
/// If a child with the same `(name, attributes)` pair already exists, the
/// existing [`ShapeId`] is returned.  This ensures that objects which
/// receive the same sequence of property additions converge on the same
/// shape — the key property that makes inline caches effective.
pub struct ShapeTable {
    /// Arena of all allocated shapes.
    shapes: Vec<Shape>,
}

impl ShapeTable {
    /// Creates a new `ShapeTable` containing only the empty root shape.
    pub fn new() -> Self {
        let root = Shape {
            id: ShapeId(0),
            parent: None,
            descriptors: Vec::new(),
            property_count: 0,
            instance_size: 0,
            n_in_object_properties: DEFAULT_IN_OBJECT_SLOTS,
            transitions: SmallVec::new(),
        };
        Self { shapes: vec![root] }
    }

    /// Returns the [`ShapeId`] of the empty root shape.
    #[inline]
    pub fn root(&self) -> ShapeId {
        ShapeId(0)
    }

    /// Returns a reference to the [`Shape`] identified by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range (debug builds only).
    #[inline]
    pub fn get(&self, id: ShapeId) -> &Shape {
        &self.shapes[id.0 as usize]
    }

    /// Returns the total number of shapes in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Returns `true` if the table contains only the root shape.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shapes.len() <= 1
    }

    /// Performs a shape transition: adds property `name` with `attrs` to
    /// `parent` and returns the resulting child [`ShapeId`].
    ///
    /// If a child with the same `(name, attrs)` pair already exists under
    /// `parent`, its [`ShapeId`] is reused (deduplication).  Otherwise a new
    /// shape is allocated, appended to the table, and linked into the parent's
    /// transition table.
    pub fn transition(
        &mut self,
        parent: ShapeId,
        name: &str,
        attrs: PropertyAttributes,
    ) -> ShapeId {
        // Fast path: check for an existing transition.
        if let Some(existing) = self.shapes[parent.0 as usize].find_transition(name, attrs) {
            return existing;
        }

        // Allocate a new child shape.
        let child_id = ShapeId(self.shapes.len() as u32);
        let parent_shape = &self.shapes[parent.0 as usize];
        let field_index = parent_shape.property_count as u32;

        let mut descriptors = parent_shape.descriptors.clone();
        descriptors.push(ShapeDescriptor::new(
            name,
            field_index,
            attrs,
            Constness::Mutable,
        ));

        let property_count = parent_shape.property_count + 1;
        let n_in_object = parent_shape.n_in_object_properties;

        let child = Shape {
            id: child_id,
            parent: Some(parent),
            descriptors,
            property_count,
            instance_size: property_count,
            n_in_object_properties: n_in_object,
            transitions: SmallVec::new(),
        };
        self.shapes.push(child);

        // Record the transition on the parent.
        self.shapes[parent.0 as usize]
            .transitions
            .push(ShapeTransition {
                property_name: name.to_string(),
                attributes: attrs,
                target: child_id,
            });

        child_id
    }

    /// Looks up a property by name in the shape identified by `id`.
    ///
    /// Returns the matching [`ShapeDescriptor`] (with its `field_index`) if
    /// found, or `None` if the property is not part of this shape.
    #[inline]
    pub fn lookup(&self, id: ShapeId, key: &str) -> Option<&ShapeDescriptor> {
        self.get(id).lookup(key)
    }
}

impl Default for ShapeTable {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShapeId ──────────────────────────────────────────────────────────────

    #[test]
    fn test_shape_id_index() {
        let id = ShapeId(42);
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn test_shape_id_equality() {
        assert_eq!(ShapeId(0), ShapeId(0));
        assert_ne!(ShapeId(0), ShapeId(1));
    }

    // ── ShapeDescriptor ─────────────────────────────────────────────────────

    #[test]
    fn test_shape_descriptor_fields() {
        let desc = ShapeDescriptor::new(
            "x",
            0,
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
            Constness::Mutable,
        );
        assert_eq!(desc.key(), "x");
        assert_eq!(desc.field_index(), 0);
        assert!(desc.attributes().contains(PropertyAttributes::WRITABLE));
        assert!(desc.attributes().contains(PropertyAttributes::ENUMERABLE));
        assert!(!desc.attributes().contains(PropertyAttributes::CONFIGURABLE));
        assert_eq!(desc.constness(), Constness::Mutable);
    }

    #[test]
    fn test_shape_descriptor_const() {
        let desc = ShapeDescriptor::new("c", 3, PropertyAttributes::empty(), Constness::Const);
        assert_eq!(desc.constness(), Constness::Const);
    }

    // ── ShapeTable basics ───────────────────────────────────────────────────

    #[test]
    fn test_table_new_has_root() {
        let table = ShapeTable::new();
        assert_eq!(table.len(), 1);
        assert!(table.is_empty()); // only root
        let root = table.get(table.root());
        assert_eq!(root.property_count(), 0);
        assert!(root.descriptors().is_empty());
        assert!(root.parent().is_none());
        assert!(root.transitions().is_empty());
    }

    #[test]
    fn test_table_default_eq_new() {
        let a = ShapeTable::new();
        let b = ShapeTable::default();
        assert_eq!(a.len(), b.len());
        assert_eq!(a.root(), b.root());
    }

    // ── Single transition ───────────────────────────────────────────────────

    #[test]
    fn test_single_transition() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        let child = table.transition(root, "x", attrs);

        assert_ne!(child, root);
        assert_eq!(table.len(), 2);

        let shape = table.get(child);
        assert_eq!(shape.property_count(), 1);
        assert_eq!(shape.parent(), Some(root));
        assert_eq!(shape.descriptors().len(), 1);
        assert_eq!(shape.descriptors()[0].key(), "x");
        assert_eq!(shape.descriptors()[0].field_index(), 0);
        assert_eq!(shape.descriptors()[0].attributes(), attrs);
    }

    // ── Transition deduplication ────────────────────────────────────────────

    #[test]
    fn test_transition_deduplication() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE;
        let a = table.transition(root, "x", attrs);
        let b = table.transition(root, "x", attrs);
        assert_eq!(a, b, "same (name, attrs) must deduplicate");
        assert_eq!(table.len(), 2); // root + one child
    }

    #[test]
    fn test_different_attrs_no_dedup() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let a = table.transition(root, "x", PropertyAttributes::WRITABLE);
        let b = table.transition(root, "x", PropertyAttributes::ENUMERABLE);
        assert_ne!(a, b, "different attrs must produce different shapes");
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn test_different_names_no_dedup() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE;
        let a = table.transition(root, "x", attrs);
        let b = table.transition(root, "y", attrs);
        assert_ne!(a, b);
        assert_eq!(table.len(), 3);
    }

    // ── Chain of transitions ────────────────────────────────────────────────

    #[test]
    fn test_transition_chain() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE
            | PropertyAttributes::ENUMERABLE
            | PropertyAttributes::CONFIGURABLE;

        let s1 = table.transition(root, "a", attrs);
        let s2 = table.transition(s1, "b", attrs);
        let s3 = table.transition(s2, "c", attrs);

        assert_eq!(table.len(), 4); // root + 3

        let shape = table.get(s3);
        assert_eq!(shape.property_count(), 3);
        assert_eq!(shape.parent(), Some(s2));

        // Verify full descriptor array.
        let descs = shape.descriptors();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].key(), "a");
        assert_eq!(descs[0].field_index(), 0);
        assert_eq!(descs[1].key(), "b");
        assert_eq!(descs[1].field_index(), 1);
        assert_eq!(descs[2].key(), "c");
        assert_eq!(descs[2].field_index(), 2);
    }

    // ── Shared prefix convergence ───────────────────────────────────────────

    #[test]
    fn test_shared_prefix_shapes() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE;

        // Path 1: root → x → y
        let sx = table.transition(root, "x", attrs);
        let sxy = table.transition(sx, "y", attrs);

        // Path 2: root → x → z
        let sx2 = table.transition(root, "x", attrs);
        let sxz = table.transition(sx2, "z", attrs);

        // sx and sx2 must be the same shape (dedup).
        assert_eq!(sx, sx2);
        // But sxy and sxz diverge.
        assert_ne!(sxy, sxz);

        // root has 1 child transition ("x"), sx has 2 ("y", "z").
        assert_eq!(table.get(root).transitions().len(), 1);
        assert_eq!(table.get(sx).transitions().len(), 2);
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_found() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE;
        let s1 = table.transition(root, "x", attrs);
        let s2 = table.transition(s1, "y", attrs);

        let desc = table.lookup(s2, "x").expect("x should exist");
        assert_eq!(desc.field_index(), 0);

        let desc = table.lookup(s2, "y").expect("y should exist");
        assert_eq!(desc.field_index(), 1);
    }

    #[test]
    fn test_lookup_not_found() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let s1 = table.transition(root, "x", PropertyAttributes::WRITABLE);
        assert!(table.lookup(s1, "z").is_none());
    }

    #[test]
    fn test_lookup_on_root_returns_none() {
        let table = ShapeTable::new();
        assert!(table.lookup(table.root(), "anything").is_none());
    }

    // ── Shape accessors ─────────────────────────────────────────────────────

    #[test]
    fn test_shape_instance_size() {
        let mut table = ShapeTable::new();
        let root = table.root();
        assert_eq!(table.get(root).instance_size(), 0);

        let s1 = table.transition(root, "a", PropertyAttributes::WRITABLE);
        assert_eq!(table.get(s1).instance_size(), 1);

        let s2 = table.transition(s1, "b", PropertyAttributes::WRITABLE);
        assert_eq!(table.get(s2).instance_size(), 2);
    }

    #[test]
    fn test_shape_n_in_object_properties() {
        let table = ShapeTable::new();
        assert_eq!(
            table.get(table.root()).n_in_object_properties(),
            DEFAULT_IN_OBJECT_SLOTS
        );
    }

    #[test]
    fn test_shape_id_accessor() {
        let table = ShapeTable::new();
        let root = table.root();
        assert_eq!(table.get(root).id(), root);
    }

    // ── Constness on descriptors ────────────────────────────────────────────

    #[test]
    fn test_transition_creates_mutable_descriptors() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let child = table.transition(root, "val", PropertyAttributes::WRITABLE);
        let desc = &table.get(child).descriptors()[0];
        assert_eq!(desc.constness(), Constness::Mutable);
    }

    // ── Large transition fan-out ────────────────────────────────────────────

    #[test]
    fn test_many_transitions_from_root() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE;
        let mut children = Vec::new();
        for i in 0..20 {
            let name = format!("prop{i}");
            children.push(table.transition(root, &name, attrs));
        }
        // All 20 children must be distinct.
        let unique: std::collections::HashSet<_> = children.iter().collect();
        assert_eq!(unique.len(), 20);
        // Root must have 20 transition edges.
        assert_eq!(table.get(root).transitions().len(), 20);
        // Table size: root + 20 children.
        assert_eq!(table.len(), 21);
    }

    // ── ShapeTransition accessors ───────────────────────────────────────────

    #[test]
    fn test_shape_transition_accessors() {
        let mut table = ShapeTable::new();
        let root = table.root();
        let attrs = PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE;
        let child = table.transition(root, "foo", attrs);

        let t = &table.get(root).transitions()[0];
        assert_eq!(t.property_name(), "foo");
        assert_eq!(t.attributes(), attrs);
        assert_eq!(t.target(), child);
    }
}

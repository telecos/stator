//! The hidden class ([`Map`]) and instance-type tag for heap objects.

use bitflags::bitflags;

use crate::objects::tagged::TaggedValue;

/// Identifies the concrete type of a [`HeapObject`][crate::objects::heap_object::HeapObject].
///
/// Each variant corresponds to a distinct Stator / JavaScript object shape
/// and is stored directly in the [`Map`] for O(1) lookup.  Values are
/// intentionally left with gaps to mirror the V8 numbering convention and
/// leave room for future additions.
#[repr(u16)]
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstanceType {
    // ── Internal / structural engine types ──────────────────────────────────
    /// A [`Map`] object itself (the hidden class of a `Map` is also a `Map`).
    Map = 0x0080,
    /// A fixed-length array of tagged values (internal engine type).
    FixedArray = 0x0088,
    /// A raw byte array (internal engine type).
    ByteArray = 0x0090,
    /// Shared function metadata shared across closures (internal engine type).
    SharedFunctionInfo = 0x00c0,
    /// A compiled code object (internal engine type).
    Code = 0x00c8,
    /// A function template used by the embedding API.
    FunctionTemplate = 0x00d0,

    // ── Primitive value wrappers ─────────────────────────────────────────────
    /// A boxed floating-point number stored on the heap.
    HeapNumber = 0x0400,
    /// A BigInt arbitrary-precision integer value.
    BigInt = 0x0408,
    /// A unique Symbol value.
    Symbol = 0x0410,

    // ── String types ─────────────────────────────────────────────────────────
    /// A JavaScript string value.
    JsString = 0x0800,

    // ── JavaScript object types ───────────────────────────────────────────────
    /// A plain JavaScript object (`{}`).
    JsObject = 0x1000,
    /// A JavaScript `Array`.
    JsArray = 0x1008,
    /// A JavaScript function.
    JsFunction = 0x1010,
    /// A JavaScript `RegExp`.
    JsRegExp = 0x1018,
    /// A JavaScript `Date`.
    JsDate = 0x1020,
    /// A JavaScript `Map` (ES2015).
    JsMap = 0x1028,
    /// A JavaScript `Set` (ES2015).
    JsSet = 0x1030,
    /// A JavaScript `WeakMap`.
    JsWeakMap = 0x1038,
    /// A JavaScript `WeakSet`.
    JsWeakSet = 0x1040,
    /// A JavaScript `Promise`.
    JsPromise = 0x1048,
    /// A JavaScript `Proxy`.
    JsProxy = 0x1050,
    /// A JavaScript `Error` (or subclass thereof).
    JsError = 0x1058,
    /// A JavaScript `arguments` object.
    JsArguments = 0x1060,
    /// A JavaScript generator object (returned by a generator function call).
    JsGeneratorObject = 0x1068,
    /// A JavaScript async-function activation object.
    JsAsyncFunctionObject = 0x1070,
}

bitflags! {
    /// Attribute flags for a [`PropertyDescriptor`].
    ///
    /// These correspond directly to the ECMAScript property attribute fields
    /// `[[Writable]]`, `[[Enumerable]]`, and `[[Configurable]]`.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub struct PropertyAttributes: u8 {
        /// The property value may be changed with an assignment.
        const WRITABLE     = 0b001;
        /// The property key shows up during enumeration (`for…in`, `Object.keys`).
        const ENUMERABLE   = 0b010;
        /// The property descriptor may be changed and the property may be deleted.
        const CONFIGURABLE = 0b100;
    }
}

/// A descriptor for a single named property attached to a [`Map`].
///
/// Mirrors the essential information from the ECMAScript *property descriptor*
/// specification record: a string key and [`PropertyAttributes`] flags.
pub struct PropertyDescriptor {
    /// The string name of the property.
    key: String,
    /// Attribute flags: writable, enumerable, configurable.
    attributes: PropertyAttributes,
}

impl PropertyDescriptor {
    /// Creates a new property descriptor with `key` and `attributes`.
    pub fn new(key: impl Into<String>, attributes: PropertyAttributes) -> Self {
        Self {
            key: key.into(),
            attributes,
        }
    }

    /// Returns the property key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the property attribute flags.
    pub fn attributes(&self) -> PropertyAttributes {
        self.attributes
    }
}

/// The hidden class (shape descriptor) of a heap object.
///
/// Every [`HeapObject`][crate::objects::heap_object::HeapObject] starts with
/// a pointer to its `Map`.  The `Map` records the object's [`InstanceType`],
/// instance size, prototype, and named-property descriptors.
///
/// # Layout note
/// `Map` is **not** marked `#[repr(C)]` because it contains a
/// [`Vec<PropertyDescriptor>`] whose internal layout is not C-compatible.
/// The `Map` is always accessed through Rust code via a raw pointer stored in
/// `HeapObject::map_word`; no C code dereferences `Map` fields directly.
pub struct Map {
    /// The concrete object type described by this `Map`.
    instance_type: InstanceType,
    /// The size in bytes of an object instance described by this `Map`.
    instance_size: u32,
    /// Tagged pointer to the prototype object.
    ///
    /// `TaggedValue(0)` means no prototype is set.
    prototype: TaggedValue,
    /// Named-property descriptors for this object shape.
    descriptors: Vec<PropertyDescriptor>,
}

impl Map {
    /// Creates a `Map` with the given instance type and instance size.
    ///
    /// `prototype` is initialised to `TaggedValue(0)` (no prototype) and
    /// `descriptors` is left empty.
    pub fn new(instance_type: InstanceType, instance_size: u32) -> Self {
        Self {
            instance_type,
            instance_size,
            prototype: TaggedValue(0),
            descriptors: Vec::new(),
        }
    }

    /// Returns the instance type of objects described by this `Map`.
    #[inline]
    pub fn instance_type(&self) -> InstanceType {
        self.instance_type
    }

    /// Returns the in-object size (in bytes) of instances described by this `Map`.
    #[inline]
    pub fn instance_size(&self) -> u32 {
        self.instance_size
    }

    /// Returns the tagged prototype pointer.
    ///
    /// A value of `TaggedValue(0)` means no prototype is set.
    #[inline]
    pub fn prototype(&self) -> TaggedValue {
        self.prototype
    }

    /// Sets the prototype pointer.
    pub fn set_prototype(&mut self, prototype: TaggedValue) {
        self.prototype = prototype;
    }

    /// Returns a slice of the property descriptors belonging to this shape.
    pub fn descriptors(&self) -> &[PropertyDescriptor] {
        &self.descriptors
    }

    /// Appends a [`PropertyDescriptor`] to this `Map`.
    pub fn add_descriptor(&mut self, descriptor: PropertyDescriptor) {
        self.descriptors.push(descriptor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_instance_type_roundtrip() {
        let map = Map::new(InstanceType::Map, 0);
        assert_eq!(map.instance_type(), InstanceType::Map);
    }

    #[test]
    fn test_map_instance_size() {
        let map = Map::new(InstanceType::JsObject, 32);
        assert_eq!(map.instance_type(), InstanceType::JsObject);
        assert_eq!(map.instance_size(), 32);
    }

    #[test]
    fn test_map_prototype_default_is_zero() {
        let map = Map::new(InstanceType::JsObject, 16);
        assert_eq!(map.prototype(), TaggedValue(0));
    }

    #[test]
    fn test_map_set_prototype() {
        let mut map = Map::new(InstanceType::JsObject, 16);
        let proto = TaggedValue(0x1000);
        map.set_prototype(proto);
        assert_eq!(map.prototype(), proto);
    }

    #[test]
    fn test_map_descriptors_empty_by_default() {
        let map = Map::new(InstanceType::JsObject, 16);
        assert!(map.descriptors().is_empty());
    }

    #[test]
    fn test_map_add_and_lookup_descriptor() {
        let mut map = Map::new(InstanceType::JsObject, 16);
        let desc = PropertyDescriptor::new(
            "x",
            PropertyAttributes::WRITABLE | PropertyAttributes::ENUMERABLE,
        );
        map.add_descriptor(desc);
        assert_eq!(map.descriptors().len(), 1);
        assert_eq!(map.descriptors()[0].key(), "x");
        assert!(
            map.descriptors()[0]
                .attributes()
                .contains(PropertyAttributes::WRITABLE)
        );
        assert!(
            map.descriptors()[0]
                .attributes()
                .contains(PropertyAttributes::ENUMERABLE)
        );
        assert!(
            !map.descriptors()[0]
                .attributes()
                .contains(PropertyAttributes::CONFIGURABLE)
        );
    }

    #[test]
    fn test_map_multiple_descriptors() {
        let mut map = Map::new(InstanceType::JsObject, 32);
        map.add_descriptor(PropertyDescriptor::new("a", PropertyAttributes::WRITABLE));
        map.add_descriptor(PropertyDescriptor::new(
            "b",
            PropertyAttributes::ENUMERABLE | PropertyAttributes::CONFIGURABLE,
        ));
        assert_eq!(map.descriptors().len(), 2);
        assert_eq!(map.descriptors()[0].key(), "a");
        assert_eq!(map.descriptors()[1].key(), "b");
    }

    #[test]
    fn test_all_instance_type_discriminants_are_unique() {
        let types: &[InstanceType] = &[
            InstanceType::Map,
            InstanceType::FixedArray,
            InstanceType::ByteArray,
            InstanceType::SharedFunctionInfo,
            InstanceType::Code,
            InstanceType::FunctionTemplate,
            InstanceType::HeapNumber,
            InstanceType::BigInt,
            InstanceType::Symbol,
            InstanceType::JsString,
            InstanceType::JsObject,
            InstanceType::JsArray,
            InstanceType::JsFunction,
            InstanceType::JsRegExp,
            InstanceType::JsDate,
            InstanceType::JsMap,
            InstanceType::JsSet,
            InstanceType::JsWeakMap,
            InstanceType::JsWeakSet,
            InstanceType::JsPromise,
            InstanceType::JsProxy,
            InstanceType::JsError,
            InstanceType::JsArguments,
            InstanceType::JsGeneratorObject,
            InstanceType::JsAsyncFunctionObject,
        ];
        // Verify all 25 variants exist.
        assert_eq!(types.len(), 25);
        // Verify every discriminant value is unique (O(n) via a HashSet).
        let discriminants: std::collections::HashSet<u16> =
            types.iter().map(|&t| t as u16).collect();
        assert_eq!(
            discriminants.len(),
            types.len(),
            "duplicate discriminant values found"
        );
    }
}

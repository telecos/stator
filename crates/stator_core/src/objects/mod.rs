/// The `HeapObject` base type shared by all GC-managed objects.
pub mod heap_object;
/// JavaScript Array with element-kind tracking and length semantics.
pub mod js_array;
/// JavaScript ordinary objects with in-object, overflow, and element storage.
pub mod js_object;
/// Hidden class ([`map::Map`]) and instance-type tag for heap objects.
pub mod map;
/// JavaScript string types with multiple internal representations.
pub mod string;
/// Tagged pointer representation for JavaScript values.
pub mod tagged;
/// Top-level JavaScript value enum and ECMAScript §7.1 type conversions.
pub mod value;

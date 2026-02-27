/// The `HeapObject` base type shared by all GC-managed objects.
pub mod heap_object;
/// Hidden class ([`map::Map`]) and instance-type tag for heap objects.
pub mod map;
/// Tagged pointer representation for JavaScript values.
pub mod tagged;
/// Top-level JavaScript value enum and ECMAScript §7.1 type conversions.
pub mod value;

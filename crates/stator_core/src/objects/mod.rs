/// The `HeapObject` base type shared by all GC-managed objects.
pub mod heap_object;
/// JavaScript Array with element-kind tracking and length semantics.
pub mod js_array;
/// JavaScript function objects: [`js_function::SharedFunctionInfo`] and [`js_function::JsFunction`].
pub mod js_function;
/// JavaScript ordinary objects with in-object, overflow, and element storage.
pub mod js_object;
/// Hidden class ([`map::Map`]) and instance-type tag for heap objects.
pub mod map;
/// NaN-boxed (tagged-pointer) 64-bit value representation ([issue #265]).
///
/// [issue #265]: https://github.com/telecos/stator/issues/265
pub mod nan_boxing;
/// ECMAScript §6.2.6 Property Descriptor specification type with data,
/// accessor, and generic variants plus validation logic.
pub mod property_descriptor;
/// JavaScript `RegExp` object with ECMAScript flag and built-in method support.
pub mod regexp;
/// V8-style hidden-class (shape) system with transition trees and descriptor
/// arrays for fast property access.
pub mod shapes;
/// JavaScript string types with multiple internal representations.
pub mod string;
/// Tagged pointer representation for JavaScript values.
pub mod tagged;
/// Top-level JavaScript value enum and ECMAScript §7.1 type conversions.
pub mod value;

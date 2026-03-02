//! Built-in JavaScript objects and their static methods.
//!
//! Each sub-module provides pure-Rust implementations of the corresponding
//! ECMAScript built-in namespace object.  These functions operate directly on
//! the engine's internal [`JsObject`][crate::objects::js_object::JsObject] and
//! [`JsValue`][crate::objects::value::JsValue] types and do **not** touch the
//! GC heap or the interpreter; they can therefore be unit-tested in isolation.

/// ECMAScript §23.1 `Array` built-in static methods and prototype equivalents.
pub mod array;
/// ECMAScript §21.3 `Math` built-in static methods.
pub mod math;
/// ECMAScript §21.1 `Number` built-in static methods and prototype equivalents.
pub mod number;
/// ECMAScript §20.1 `Object` built-in static methods.
pub mod object;
/// ECMAScript §27 `Promise` built-in and microtask queue.
pub mod promise;
/// ECMAScript §22.1 `String` built-in static methods and prototype equivalents.
pub mod string;

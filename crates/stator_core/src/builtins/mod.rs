//! Built-in JavaScript objects and their static methods.
//!
//! Each sub-module provides pure-Rust implementations of the corresponding
//! ECMAScript built-in namespace object.  These functions operate directly on
//! the engine's internal [`JsObject`][crate::objects::js_object::JsObject] and
//! [`JsValue`][crate::objects::value::JsValue] types and do **not** touch the
//! GC heap or the interpreter; they can therefore be unit-tested in isolation.

/// ECMAScript §23.1 `Array` built-in static methods and prototype equivalents.
pub mod array;
/// ECMAScript §20.5 `Error` built-in and error hierarchy (`TypeError`, `RangeError`, etc.).
pub mod error;
/// ECMAScript §19 global object properties and global functions (`isNaN`, `isFinite`,
/// `parseInt`, `parseFloat`, `encodeURI`, `decodeURI`, `encodeURIComponent`,
/// `decodeURIComponent`, `eval`, and the `NaN`/`Infinity`/`undefined` constants).
pub mod global;
/// ECMAScript §27 Iterator and Generator protocol: `Symbol.iterator`,
/// `IteratorRecord`, and built-in iterators for `Array`, `String`, `Map`, `Set`.
pub mod iterator;
/// ECMAScript §25.5 `JSON` built-in — `JSON.parse` and `JSON.stringify`.
pub mod json;
/// ECMAScript §24.1 `Map` built-in — insertion-ordered key-value collection.
pub mod map;
/// ECMAScript §21.3 `Math` built-in static methods.
pub mod math;
/// ECMAScript §21.1 `Number` built-in static methods and prototype equivalents.
pub mod number;
/// ECMAScript §20.1 `Object` built-in static methods.
pub mod object;
/// ECMAScript §27 `Promise` built-in and microtask queue.
pub mod promise;
/// ECMAScript §28.2 `Proxy` built-in and §10.5 invariant enforcement.
pub mod proxy;
/// ECMAScript §28.1 `Reflect` built-in static methods.
pub mod reflect;
/// ECMAScript §24.2 `Set` built-in — insertion-ordered unique-value collection.
pub mod set;
/// ECMAScript §22.1 `String` built-in static methods and prototype equivalents.
pub mod string;
/// Shared utility functions (e.g. `SameValueZero`) used across built-in sub-modules.
pub(crate) mod util;
/// ECMAScript §24.3 `WeakMap` built-in — object-keyed weak map with ephemeron GC semantics.
pub mod weak_map;
/// ECMAScript §24.4 `WeakSet` built-in — object-keyed weak set with ephemeron GC semantics.
pub mod weak_set;

//! JavaScript value representation.
//!
//! This module provides [`JsValue`], the top-level enum that can hold any
//! ECMAScript value, together with type-checking predicates and the abstract
//! type-conversion operations defined in ECMAScript §7.1 / §7.2:
//!
//! * [`to_boolean`][JsValue::to_boolean] (§7.1.2)
//! * [`to_number`][JsValue::to_number] (§7.1.4) — now with full ToPrimitive path
//! * [`to_int32`][JsValue::to_int32] (§7.1.6)
//! * [`to_uint32`][JsValue::to_uint32] (§7.1.7)
//! * [`to_int16`][JsValue::to_int16] (§7.1.9)
//! * [`to_integer_or_infinity`][JsValue::to_integer_or_infinity] (§7.1.5)
//! * [`to_length`][JsValue::to_length] (§7.1.15)
//! * [`to_numeric`][JsValue::to_numeric] (§7.1.3)
//! * [`to_js_string`][JsValue::to_js_string] (§7.1.17 ToString)
//! * [`to_object`][JsValue::to_object] (§7.1.18)
//! * [`to_property_key`][JsValue::to_property_key] (§7.1.19)
//! * [`to_primitive`][JsValue::to_primitive] (§7.1.1) with OrdinaryToPrimitive
//! * [`same_value`][JsValue::same_value] (§7.2.10)
//! * [`same_value_zero`][JsValue::same_value_zero] (§7.2.11)
//! * [`is_loosely_equal`][JsValue::is_loosely_equal] (§7.2.13)
//! * [`is_strictly_equal`][JsValue::is_strictly_equal] (§7.2.15)
//!
//! This module also defines the generator and iterator support types used by
//! the bytecode interpreter and built-in iterators:
//!
//! - [`GeneratorStatus`] / [`GeneratorState`] / [`GeneratorStep`] — the
//!   low-level execution state for generator functions (see
//!   [`crate::interpreter::Interpreter::run_generator_step`]).
//! - [`NativeIterator`] — a Rust-level iterator that wraps a pre-collected
//!   `Vec<JsValue>` and is surfaced as [`JsValue::Iterator`].

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::error::JsError;
use crate::builtins::proxy::JsProxy;
use crate::builtins::symbol::{SYMBOL_TO_PRIMITIVE, symbol_description};
use crate::bytecode::bytecode_array::BytecodeArray;
use crate::error::{StatorError, StatorResult};
use crate::gc::trace::{Trace, Tracer};
use crate::objects::heap_object::HeapObject;
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;

// ─────────────────────────────────────────────────────────────────────────────
// Generator support types
// ─────────────────────────────────────────────────────────────────────────────

/// How a generator should be resumed on the next step.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorResumeMode {
    /// Normal `.next(value)` — continue from the yield point.
    Normal,
    /// `.throw(value)` — throw an exception at the yield point.
    Throw(JsValue),
    /// `.return(value)` — force a return completion at the yield point,
    /// triggering any enclosing `finally` blocks.
    Return(JsValue),
}

/// Lifecycle status of a JavaScript generator object.
///
/// Maps to V8's `JSGeneratorObject::ResumeMode` / `GeneratorState` integers:
/// `Executing` = −2, `Completed` = −1, `SuspendedAtStart` = 0,
/// `SuspendedAtYield` = 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorStatus {
    /// The generator function has not yet been advanced by `.next()`.
    SuspendedAtStart,
    /// The generator is paused at a `yield` expression.
    SuspendedAtYield,
    /// The generator is currently executing (re-entrancy is invalid).
    Executing,
    /// The generator function returned; iteration is complete.
    Completed,
}

impl GeneratorStatus {
    /// Integer encoding compatible with V8's `GetGeneratorState` / `SetGeneratorState`.
    pub fn to_smi(self) -> i32 {
        match self {
            Self::Executing => -2,
            Self::Completed => -1,
            Self::SuspendedAtStart => 0,
            Self::SuspendedAtYield => 1,
        }
    }

    /// Decode from the integer produced by [`Self::to_smi`].
    ///
    /// Any value that does not map to a known status is treated as
    /// `SuspendedAtYield`.
    pub fn from_smi(n: i32) -> Self {
        match n {
            -2 => Self::Executing,
            -1 => Self::Completed,
            0 => Self::SuspendedAtStart,
            _ => Self::SuspendedAtYield,
        }
    }
}

/// The saved execution state of a suspended JavaScript generator.
///
/// Holds all data needed to resume execution of a generator function after it
/// has been suspended at a `yield` expression or before the first `.next()`
/// call.
///
/// Use [`GeneratorState::new`] to create a fresh generator, then drive it
/// with [`crate::interpreter::Interpreter::run_generator_step`].
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorState {
    /// Bytecode for the generator function body.
    pub bytecode_array: BytecodeArray,
    /// Prototype for the generator object returned by the call.
    ///
    /// This is usually the callee's own `.prototype` object, falling back to
    /// the intrinsic `%GeneratorPrototype%` when no per-function prototype is
    /// available.
    pub prototype: Option<JsValue>,
    /// Saved register file at the point of suspension (empty before the
    /// first [`crate::bytecode::bytecodes::Opcode::SuspendGenerator`]).
    pub registers: Vec<JsValue>,
    /// Original call arguments for the activation.
    pub call_args: Vec<JsValue>,
    /// Global environment used to execute/resume the activation.
    pub global_env: Option<Rc<RefCell<crate::interpreter::GlobalEnv>>>,
    /// Instruction index to resume from; `0` = start of function body.
    pub resume_pc: usize,
    /// Current lifecycle status.
    pub status: GeneratorStatus,
    /// How the generator should be resumed on the next step.
    pub resume_mode: GeneratorResumeMode,
    /// The `new.target` value visible to the activation.
    pub new_target: JsValue,
}

impl GeneratorState {
    /// Create a new generator ready to execute `bytecode_array` from the
    /// beginning on the first call to
    /// [`crate::interpreter::Interpreter::run_generator_step`].
    pub fn new(bytecode_array: BytecodeArray) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            bytecode_array,
            prototype: None,
            registers: Vec::new(),
            call_args: Vec::new(),
            global_env: None,
            resume_pc: 0,
            status: GeneratorStatus::SuspendedAtStart,
            resume_mode: GeneratorResumeMode::Normal,
            new_target: JsValue::Undefined,
        }))
    }
}

/// The result of running one step of a generator via
/// [`crate::interpreter::Interpreter::run_generator_step`].
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorStep {
    /// The generator suspended at a `yield` expression; the contained value
    /// is the yielded result.  The generator may be resumed.
    Yield(JsValue),
    /// The generator's function body returned; the contained value is the
    /// final return value.  The generator is now exhausted.
    Return(JsValue),
}

// ─────────────────────────────────────────────────────────────────────────────
// Native iterator
// ─────────────────────────────────────────────────────────────────────────────

/// A Rust-level iterator over a pre-collected sequence of [`JsValue`]s.
///
/// Built-in iterators for `Array`, `String`, `Map`, and `Set` are represented
/// as a `NativeIterator` wrapped in [`JsValue::Iterator`].  The iterator
/// advances by incrementing an internal index into the item vector.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeIterator {
    /// The items to iterate over, collected eagerly.
    pub items: Vec<JsValue>,
    /// Zero-based index of the next item to yield.
    pub index: usize,
}

impl NativeIterator {
    /// Create a `NativeIterator` from a pre-collected item vector.
    pub fn from_items(items: Vec<JsValue>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self { items, index: 0 }))
    }

    /// Create a `NativeIterator` that yields the individual Unicode scalar
    /// values of `s` as single-character strings.
    pub fn from_string(s: &str) -> Rc<RefCell<Self>> {
        let items = s
            .chars()
            .map(|c| JsValue::String(c.to_string().into()))
            .collect();
        Rc::new(RefCell::new(Self { items, index: 0 }))
    }

    /// Advance the iterator.
    ///
    /// Returns `Some(value)` if there are remaining items, or `None` when
    /// the sequence is exhausted.
    pub fn next_item(&mut self) -> Option<JsValue> {
        if self.index < self.items.len() {
            let val = self.items[self.index].clone();
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

/// Signature for a native Rust function callable from JavaScript.
///
/// Receives the list of positional arguments and returns a [`JsValue`] result.
/// Used by [`JsValue::NativeFunction`].
pub type NativeFn = Rc<dyn Fn(Vec<JsValue>) -> StatorResult<JsValue>>;

/// Hint for the [`JsValue::to_primitive`] abstract operation (ECMAScript §7.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToPrimitiveHint {
    /// No preference — equivalent to `Number` for ordinary objects.
    Default,
    /// Prefer a numeric result (`valueOf` before `toString`).
    Number,
    /// Prefer a string result (`toString` before `valueOf`).
    String,
}

/// Any ECMAScript value.
///
/// Primitive variants carry their data inline; `Object` holds a raw pointer to
/// a GC-managed [`HeapObject`]; `Function` holds a reference-counted
/// [`BytecodeArray`] representing a callable closure; `Array` holds a
/// reference-counted vector used by built-in combinators (e.g. `Promise.all`)
/// that need to produce array values without allocating on the GC heap.
/// `Generator` wraps a suspended generator function's execution state.
/// `Iterator` wraps a [`NativeIterator`] over a pre-collected sequence.
/// `NativeFunction` wraps an embedder-provided Rust closure.
/// `PlainObject` wraps a simple property map for lightweight object creation.
/// `Context` wraps a scope context for closure variable capture.
///
/// # Safety – `Object` variant
///
/// The pointer stored in `Object(ptr)` must refer to a live object managed by
/// the engine heap.  It is the caller's responsibility to ensure the object
/// outlives the `JsValue` that wraps it and that no GC compaction has
/// invalidated the pointer.
#[derive(Clone)]
pub enum JsValue {
    /// The ECMAScript `undefined` primitive.
    Undefined,
    /// The ECMAScript `null` primitive.
    Null,
    /// Internal sentinel for uninitialized `let`/`const`/`class` bindings
    /// (the Temporal Dead Zone).
    ///
    /// This is **not** a user-visible JavaScript value.  It exists only so
    /// that the interpreter can distinguish "not yet initialised" from an
    /// explicit `undefined` assignment.  The `ThrowReferenceErrorIfHole`
    /// family of opcodes checks for this value and throws a
    /// `ReferenceError` when it is encountered.
    TheHole,
    /// A JavaScript boolean (`true` or `false`).
    Boolean(bool),
    /// A small (31-bit signed) integer, stored inline without heap allocation.
    Smi(i32),
    /// A double-precision floating-point number stored inline.
    HeapNumber(f64),
    /// A JavaScript string value.
    ///
    /// Stored as `Rc<str>` so that cloning a string value is O(1) (reference
    /// count bump) instead of O(n) (full heap copy).
    String(Rc<str>),
    /// A unique JavaScript symbol, identified by an opaque 64-bit descriptor.
    Symbol(u64),
    /// A pointer to a GC-managed heap object.
    Object(*mut HeapObject),
    /// A JavaScript `BigInt` value (represented as a 128-bit signed integer).
    BigInt(i128),
    /// A callable JavaScript function backed by a [`BytecodeArray`] closure.
    ///
    /// The [`Rc`] allows function values to be cheaply cloned and shared
    /// without copying the bytecode.
    Function(Rc<BytecodeArray>),
    /// A mutable JavaScript array backed by a reference-counted [`RefCell`]
    /// wrapping a [`Vec`].
    ///
    /// Used by built-in combinators such as `Promise.all` that need to return
    /// array values without interacting with the GC heap.  The [`RefCell`]
    /// layer enables in-place mutation required by `Array.prototype.push`,
    /// `splice`, `sort`, etc.
    Array(Rc<RefCell<Vec<JsValue>>>),
    /// A JavaScript generator object holding the suspended execution state of
    /// a generator function.
    ///
    /// Generators are produced by calling a function compiled with
    /// `is_generator = true`; they are iterable (they are their own
    /// `@@iterator`) and are advanced by the [`Opcode::IteratorNext`] opcode
    /// or by [`crate::interpreter::Interpreter::run_generator_step`].
    Generator(Rc<RefCell<GeneratorState>>),
    /// A built-in iterator over a pre-collected sequence (Array, String, …).
    ///
    /// Created by the [`Opcode::GetIterator`] opcode when the iterable is a
    /// [`JsValue::Array`] or [`JsValue::String`].
    Iterator(Rc<RefCell<NativeIterator>>),
    /// A JavaScript `Error` object with `name`, `message`, and `stack` properties.
    ///
    /// Covers all eight ECMAScript error types: `Error`, `TypeError`,
    /// `RangeError`, `ReferenceError`, `SyntaxError`, `URIError`, `EvalError`,
    /// and `AggregateError`.
    Error(Rc<JsError>),
    /// A native Rust function callable from JavaScript.
    ///
    /// Used to expose host-provided functionality (e.g. `console.log`,
    /// `print`) to JavaScript code without compiling a bytecode body.
    NativeFunction(NativeFn),
    /// A lightweight property map representing a plain JavaScript object.
    ///
    /// Each property carries [`PropertyAttributes`] flags alongside its
    /// value, enabling ECMAScript attribute enforcement (writable,
    /// enumerable, configurable) at the interpreter level.
    PlainObject(Rc<RefCell<crate::objects::property_map::PropertyMap>>),
    /// A JavaScript `Promise` object backed by the [`JsPromise`] state machine.
    ///
    /// Wraps the shared-state [`JsPromise`] handle from the promise module so
    /// that promises can be stored and passed through the value pipeline just
    /// like any other JavaScript value.
    Promise(crate::builtins::promise::JsPromise),
    /// A scope context for closure variable capture.
    ///
    /// A context holds numbered slots for captured variables and an optional
    /// parent pointer forming the scope chain.  Used by context-slot opcodes
    /// (`LdaContextSlot`, `StaContextSlot`, etc.) in the interpreter.
    Context(Rc<RefCell<JsContext>>),
    /// A JavaScript `Proxy` object wrapping a target with handler traps.
    ///
    /// Created by `new Proxy(target, handler)` or `Proxy.revocable(target,
    /// handler)`.  Operations performed on the proxy are intercepted by the
    /// handler traps; if no trap is installed the operation falls through to
    /// the target.
    Proxy(Rc<RefCell<JsProxy>>),
    /// A JavaScript `ArrayBuffer` (ECMAScript §25.1) — raw binary data store.
    ArrayBuffer(Rc<RefCell<crate::builtins::typed_array::JsArrayBuffer>>),
    /// A JavaScript TypedArray (ECMAScript §23.2) — typed view over an `ArrayBuffer`.
    TypedArray(Rc<RefCell<crate::builtins::typed_array::JsTypedArray>>),
    /// A JavaScript `DataView` (ECMAScript §25.3) — byte-level buffer accessor.
    DataView(Rc<RefCell<crate::builtins::typed_array::JsDataView>>),
}

/// A scope context representing the environment for captured variables.
///
/// Each context contains a vector of numbered slots and an optional parent
/// pointer so that inner scopes can walk the chain to reach outer scopes.
#[derive(Debug, Clone, PartialEq)]
pub struct JsContext {
    /// The captured variable slots.
    pub slots: Vec<JsValue>,
    /// The enclosing (parent) context, or `None` for the outermost scope.
    pub parent: Option<Rc<RefCell<JsContext>>>,
}

impl JsContext {
    /// Create a new context with `slot_count` slots initialised to `undefined`,
    /// chained to an optional parent context.
    pub fn new(slot_count: usize, parent: Option<Rc<RefCell<JsContext>>>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            slots: vec![JsValue::Undefined; slot_count],
            parent,
        }))
    }
}

// Manual Debug impl: NativeFunction can't derive Debug (dyn Fn).
impl std::fmt::Debug for JsValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undefined => write!(f, "Undefined"),
            Self::Null => write!(f, "Null"),
            Self::TheHole => write!(f, "TheHole"),
            Self::Boolean(b) => write!(f, "Boolean({b})"),
            Self::Smi(n) => write!(f, "Smi({n})"),
            Self::HeapNumber(n) => write!(f, "HeapNumber({n})"),
            Self::String(s) => write!(f, "String({s:?})"),
            Self::Symbol(id) => write!(f, "Symbol({id})"),
            Self::Object(ptr) => write!(f, "Object({ptr:?})"),
            Self::BigInt(n) => write!(f, "BigInt({n})"),
            Self::Function(ba) => write!(f, "Function({ba:?})"),
            Self::Array(arr) => write!(f, "Array({:?})", arr.borrow()),
            Self::Generator(g) => write!(f, "Generator({g:?})"),
            Self::Iterator(i) => write!(f, "Iterator({i:?})"),
            Self::Error(e) => write!(f, "Error({e:?})"),
            Self::NativeFunction(_) => write!(f, "NativeFunction"),
            Self::PlainObject(map) => write!(f, "PlainObject({map:?})"),
            Self::Promise(p) => write!(f, "Promise({:?})", p.state()),
            Self::Context(ctx) => write!(f, "Context({ctx:?})"),
            Self::Proxy(p) => write!(f, "Proxy(revoked={})", p.borrow().is_revoked()),
            Self::ArrayBuffer(buf) => write!(f, "ArrayBuffer({})", buf.borrow().data.len()),
            Self::TypedArray(ta) => {
                let ta = ta.borrow();
                write!(f, "TypedArray({}, len={})", ta.kind.name(), ta.length)
            }
            Self::DataView(dv) => write!(f, "DataView(len={})", dv.borrow().byte_length),
        }
    }
}

// Manual PartialEq: NativeFunction uses Rc pointer equality.
impl PartialEq for JsValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Undefined, Self::Undefined) => true,
            (Self::Null, Self::Null) => true,
            (Self::TheHole, Self::TheHole) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Smi(a), Self::Smi(b)) => a == b,
            (Self::HeapNumber(a), Self::HeapNumber(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Symbol(a), Self::Symbol(b)) => a == b,
            // Object identity — `JsValue::Object` holds a raw `*mut HeapObject`
            // pointer; equality is pointer equality (same allocation).
            (Self::Object(a), Self::Object(b)) => std::ptr::eq(*a, *b),
            (Self::BigInt(a), Self::BigInt(b)) => a == b,
            // Reference types: compare contents (matching the original derive behaviour).
            (Self::Function(a), Self::Function(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => Rc::ptr_eq(a, b),
            (Self::Generator(a), Self::Generator(b)) => Rc::ptr_eq(a, b),
            (Self::Iterator(a), Self::Iterator(b)) => Rc::ptr_eq(a, b),
            (Self::Error(a), Self::Error(b)) => Rc::ptr_eq(a, b),
            // NativeFunction has no comparable content; use pointer identity.
            (Self::NativeFunction(a), Self::NativeFunction(b)) => Rc::ptr_eq(a, b),
            (Self::PlainObject(a), Self::PlainObject(b)) => Rc::ptr_eq(a, b),
            (Self::Promise(a), Self::Promise(b)) => a == b,
            (Self::Context(a), Self::Context(b)) => Rc::ptr_eq(a, b),
            (Self::Proxy(a), Self::Proxy(b)) => Rc::ptr_eq(a, b),
            (Self::ArrayBuffer(a), Self::ArrayBuffer(b)) => Rc::ptr_eq(a, b),
            (Self::TypedArray(a), Self::TypedArray(b)) => Rc::ptr_eq(a, b),
            (Self::DataView(a), Self::DataView(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Type-checking predicates
// ──────────────────────────────────────────────────────────────────────────────

impl JsValue {
    /// Create a new mutable `Array` value from a `Vec<JsValue>`.
    ///
    /// This is the preferred constructor—callers should use this instead of
    /// manually wrapping in `Rc<RefCell<…>>`.
    #[inline]
    pub fn new_array(items: Vec<JsValue>) -> Self {
        Self::Array(Rc::new(RefCell::new(items)))
    }

    /// Returns `true` if this value is `undefined`.
    #[inline]
    pub fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }

    /// Returns `true` if this value is the internal hole sentinel.
    #[inline]
    pub fn is_the_hole(&self) -> bool {
        matches!(self, Self::TheHole)
    }

    /// Returns `true` if this value is `null`.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this value is `null` or `undefined`.
    #[inline]
    pub fn is_nullish(&self) -> bool {
        matches!(self, Self::Null | Self::Undefined)
    }

    /// Returns `true` if this value is a boolean.
    #[inline]
    pub fn is_boolean(&self) -> bool {
        matches!(self, Self::Boolean(_))
    }

    /// Returns `true` if this value is a small integer ([`Smi`][JsValue::Smi]).
    #[inline]
    pub fn is_smi(&self) -> bool {
        matches!(self, Self::Smi(_))
    }

    /// Returns `true` if this value is a heap number ([`HeapNumber`][JsValue::HeapNumber]).
    #[inline]
    pub fn is_heap_number(&self) -> bool {
        matches!(self, Self::HeapNumber(_))
    }

    /// Returns `true` if this value is any numeric type (`Smi` or `HeapNumber`).
    #[inline]
    pub fn is_number(&self) -> bool {
        matches!(self, Self::Smi(_) | Self::HeapNumber(_))
    }

    /// Returns `true` if this value is a string.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    /// Returns `true` if this value is a symbol.
    #[inline]
    pub fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
    }

    /// Returns `true` if this value is an object.
    #[inline]
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// Returns `true` if this value is a `BigInt`.
    #[inline]
    pub fn is_bigint(&self) -> bool {
        matches!(self, Self::BigInt(_))
    }

    /// Returns `true` if this value is a callable function.
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_) | Self::NativeFunction(_))
    }

    /// Returns `true` if this value is a lightweight array ([`Array`][JsValue::Array]).
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Returns `true` if this value is a generator object ([`Generator`][JsValue::Generator]).
    #[inline]
    pub fn is_generator(&self) -> bool {
        matches!(self, Self::Generator(_))
    }

    /// Returns `true` if this value is a native iterator ([`Iterator`][JsValue::Iterator]).
    #[inline]
    pub fn is_iterator(&self) -> bool {
        matches!(self, Self::Iterator(_))
    }

    /// Returns `true` if this value is a JavaScript `Error` object ([`Error`][JsValue::Error]).
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns `true` if this value is a JavaScript `Promise` ([`Promise`][JsValue::Promise]).
    #[inline]
    pub fn is_promise(&self) -> bool {
        matches!(self, Self::Promise(_))
    }

    /// Returns `true` if this value is an ECMAScript primitive
    /// (Undefined, Null, Boolean, Number, String, Symbol, or BigInt).
    #[inline]
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Self::Undefined
                | Self::Null
                | Self::Boolean(_)
                | Self::Smi(_)
                | Self::HeapNumber(_)
                | Self::String(_)
                | Self::Symbol(_)
                | Self::BigInt(_)
        )
    }

    /// Returns `true` if this value is an object-like type (not a primitive).
    #[inline]
    pub fn is_object_like(&self) -> bool {
        !self.is_primitive()
    }

    /// Clone this value with minimal overhead.
    ///
    /// Scalar variants (`Undefined`, `Null`, `TheHole`, `Boolean`, `Smi`,
    /// `HeapNumber`, `Symbol`, `Object`, `BigInt`) are plain bitwise copies
    /// with no reference-count traffic.  Reference-counted variants fall
    /// back to the derived [`Clone`] implementation.
    #[inline(always)]
    pub fn cheap_clone(&self) -> Self {
        match self {
            Self::Undefined => Self::Undefined,
            Self::Null => Self::Null,
            Self::TheHole => Self::TheHole,
            Self::Boolean(b) => Self::Boolean(*b),
            Self::Smi(n) => Self::Smi(*n),
            Self::HeapNumber(n) => Self::HeapNumber(*n),
            Self::Symbol(s) => Self::Symbol(*s),
            Self::Object(p) => Self::Object(*p),
            Self::BigInt(n) => Self::BigInt(*n),
            other => other.clone(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Abstract type-conversion operations (ECMAScript §7.1)
// ──────────────────────────────────────────────────────────────────────────────

impl JsValue {
    /// ECMAScript §7.1.2 **ToBoolean**.
    ///
    /// | Value type | Result |
    /// |---|---|
    /// | `Undefined` | `false` |
    /// | `Null` | `false` |
    /// | `Boolean` | the boolean itself |
    /// | `Smi` | `false` if `0`, otherwise `true` |
    /// | `HeapNumber` | `false` if `+0.0`, `-0.0`, or `NaN`; otherwise `true` |
    /// | `String` | `false` if the string is empty; otherwise `true` |
    /// | `Symbol` | `true` |
    /// | `Object` | `true` |
    /// | `BigInt` | `false` if `0`, otherwise `true` |
    #[inline]
    pub fn to_boolean(&self) -> bool {
        match self {
            Self::Undefined | Self::Null | Self::TheHole => false,
            Self::Boolean(b) => *b,
            Self::Smi(n) => *n != 0,
            Self::HeapNumber(n) => !n.is_nan() && *n != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::Symbol(_)
            | Self::Object(_)
            | Self::Function(_)
            | Self::Array(_)
            | Self::Error(_)
            | Self::Generator(_)
            | Self::Iterator(_)
            | Self::NativeFunction(_)
            | Self::PlainObject(_)
            | Self::Promise(_)
            | Self::Context(_)
            | Self::Proxy(_)
            | Self::ArrayBuffer(_)
            | Self::TypedArray(_)
            | Self::DataView(_) => true,
            Self::BigInt(n) => *n != 0,
        }
    }

    /// ECMAScript §7.1.1 **ToPrimitive(input, preferredType)**.
    ///
    /// If the value is already a primitive, returns it unchanged.
    /// For object-like types, first looks up `@@toPrimitive` and calls it when
    /// present.  Otherwise it performs **OrdinaryToPrimitive**:
    /// - With [`ToPrimitiveHint::String`]: tries `toString` then `valueOf`.
    /// - With [`ToPrimitiveHint::Number`] or [`ToPrimitiveHint::Default`]:
    ///   tries `valueOf` then `toString`.
    ///
    /// `Date` instances inherit `Date.prototype[@@toPrimitive]`, so the
    /// `"default"` hint behaves like `"string"` for them as required by the
    /// ECMAScript spec.
    pub fn to_primitive(&self, hint: ToPrimitiveHint) -> StatorResult<JsValue> {
        match self {
            // Primitives return themselves (§7.1.1 step 1).
            Self::Undefined
            | Self::Null
            | Self::Boolean(_)
            | Self::Smi(_)
            | Self::HeapNumber(_)
            | Self::String(_)
            | Self::Symbol(_)
            | Self::BigInt(_) => Ok(self.clone()),

            // TheHole is an internal sentinel — treat as undefined.
            Self::TheHole => Ok(JsValue::Undefined),

            // Object-like values: @@toPrimitive, then OrdinaryToPrimitive.
            _ => {
                let exotic = crate::interpreter::dispatch_get_property_value(
                    self,
                    JsValue::Symbol(SYMBOL_TO_PRIMITIVE),
                )?;
                if !matches!(exotic, JsValue::Undefined | JsValue::Null) {
                    if !is_callable_for_to_primitive(&exotic) {
                        return Err(StatorError::TypeError(
                            "Symbol.toPrimitive is not a function".into(),
                        ));
                    }
                    let result = call_to_primitive_method(
                        &exotic,
                        self,
                        vec![JsValue::String(to_primitive_hint_str(hint).into())],
                    )?;
                    if result.is_primitive() {
                        return Ok(result);
                    }
                    return Err(StatorError::TypeError(
                        "Symbol.toPrimitive returned a non-primitive".into(),
                    ));
                }

                ordinary_to_primitive(self, hint)
            }
        }
    }

    /// ECMAScript §7.1.4 **ToNumber**.
    ///
    /// Returns `Err(StatorError::TypeError)` for `Symbol` and `BigInt`.
    /// Object-like types go through [`to_primitive`][Self::to_primitive] with a
    /// `Number` hint, then the resulting primitive is converted to a number.
    ///
    /// String parsing handles hex (`0x`), octal (`0o`), and binary (`0b`)
    /// integer literals as well as `"Infinity"` / `"-Infinity"`.
    #[inline]
    pub fn to_number(&self) -> StatorResult<f64> {
        match self {
            Self::Undefined | Self::TheHole => Ok(f64::NAN),
            Self::Null => Ok(0.0),
            Self::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
            Self::Smi(n) => Ok(f64::from(*n)),
            Self::HeapNumber(n) => Ok(*n),
            Self::String(s) => Ok(string_to_number(s)),
            Self::Symbol(_) => Err(StatorError::TypeError(
                "Cannot convert a Symbol value to a number".to_string(),
            )),
            Self::BigInt(_) => Err(StatorError::TypeError(
                "Cannot convert a BigInt value to a number".to_string(),
            )),
            // Object-like types: ToPrimitive(input, number) then ToNumber.
            _ => {
                let prim = self.to_primitive(ToPrimitiveHint::Number)?;
                prim.to_number()
            }
        }
    }

    /// ECMAScript §7.1.17 **ToString**.
    ///
    /// Named `to_js_string` to avoid ambiguity with [`ToString::to_string`].
    ///
    /// Returns `Err(StatorError::TypeError)` for `Symbol`.
    /// Object-like types go through [`to_primitive`][Self::to_primitive] with a
    /// `String` hint, then the resulting primitive is converted to a string.
    pub fn to_js_string(&self) -> StatorResult<String> {
        match self {
            Self::Undefined | Self::TheHole => Ok("undefined".to_string()),
            Self::Null => Ok("null".to_string()),
            Self::Boolean(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            Self::Smi(n) => Ok(n.to_string()),
            Self::HeapNumber(n) => Ok(number_to_string(*n)),
            Self::String(s) => Ok(s.to_string()),
            Self::Symbol(_) => Err(StatorError::TypeError(
                "Cannot convert a Symbol value to a string".to_string(),
            )),
            Self::BigInt(n) => Ok(n.to_string()),
            // Array.prototype.toString — join elements with ",".
            // Handles the common case directly without a ToPrimitive roundtrip.
            Self::Array(items) => {
                // Clone elements before iterating so the RefCell borrow is
                // released before any recursive `to_js_string()` call.
                let snapshot: Vec<JsValue> = items.borrow().clone();
                let parts: Vec<String> = snapshot
                    .iter()
                    .map(|v| match v {
                        Self::Null | Self::Undefined => String::new(),
                        other => other.to_js_string().unwrap_or_default(),
                    })
                    .collect();
                Ok(parts.join(","))
            }
            // Object-like types: ToPrimitive(input, string) then ToString.
            _ => {
                let prim = self.to_primitive(ToPrimitiveHint::String)?;
                prim.to_js_string()
            }
        }
    }

    /// Console-safe string conversion that never throws.
    ///
    /// Unlike [`to_js_string`](Self::to_js_string), this handles `Symbol`
    /// values by formatting them as `"Symbol(<id>)"` rather than returning
    /// a `TypeError`.  Used by `console.log` and similar debugging output.
    pub fn to_display_string(&self) -> String {
        match self {
            Self::Symbol(id) => format!("Symbol({id})"),
            _ => self.to_js_string().unwrap_or_else(|_| format!("{self:?}")),
        }
    }

    /// ECMAScript §7.1.6 **ToInt32**.
    ///
    /// Converts the value to a number via [`to_number`][Self::to_number], then
    /// applies the modulo-2³² truncation algorithm.  `NaN`, `±0`, and `±∞`
    /// all map to `0`.
    pub fn to_int32(&self) -> StatorResult<i32> {
        let n = self.to_number()?;
        Ok(f64_to_int32(n))
    }

    /// ECMAScript §7.1.7 **ToUint32**.
    ///
    /// Converts the value to a number via [`to_number`][Self::to_number], then
    /// applies the modulo-2³² truncation algorithm returning an unsigned result.
    pub fn to_uint32(&self) -> StatorResult<u32> {
        let n = self.to_number()?;
        Ok(f64_to_uint32(n))
    }

    /// ECMAScript §7.1.9 **ToInt16**.
    ///
    /// Converts the value to a number via [`to_number`][Self::to_number], then
    /// applies the modulo-2¹⁶ truncation algorithm.
    pub fn to_int16(&self) -> StatorResult<i16> {
        let n = self.to_number()?;
        Ok(f64_to_int16(n))
    }

    /// ECMAScript §7.1.5 **ToIntegerOrInfinity**.
    ///
    /// Used extensively by `Array`, `String`, and `TypedArray` built-in methods
    /// (e.g. `slice`, `indexOf`, `at`) to clamp numeric arguments.
    ///
    /// * `NaN` → `0`
    /// * `+∞` / `−∞` → `+∞` / `−∞`
    /// * Otherwise → truncate toward zero (`Math.trunc` semantics)
    pub fn to_integer_or_infinity(&self) -> StatorResult<f64> {
        let number = self.to_number()?;
        if number.is_nan() || number == 0.0 {
            Ok(0.0)
        } else if number.is_infinite() {
            Ok(number)
        } else {
            Ok(number.trunc())
        }
    }

    /// ECMAScript §7.1.15 **ToLength**.
    ///
    /// Clamps the result of [`to_integer_or_infinity`][Self::to_integer_or_infinity]
    /// to the range \[0, 2⁵³ − 1\].  Used by array/string methods that accept
    /// a `length` argument.
    pub fn to_length(&self) -> StatorResult<u64> {
        let len = self.to_integer_or_infinity()?;
        if len <= 0.0 {
            Ok(0)
        } else {
            // 2^53 − 1
            Ok((len.min(9_007_199_254_740_991.0)) as u64)
        }
    }

    /// ECMAScript §7.1.3 **ToNumeric**.
    ///
    /// Returns the value as either a `Number` ([`HeapNumber`][Self::HeapNumber])
    /// or a [`BigInt`][Self::BigInt].  Object-like types are first converted
    /// via [`to_primitive`][Self::to_primitive] with a `Number` hint.
    pub fn to_numeric(&self) -> StatorResult<JsValue> {
        let prim = self.to_primitive(ToPrimitiveHint::Number)?;
        if matches!(prim, JsValue::BigInt(_)) {
            Ok(prim)
        } else {
            Ok(JsValue::HeapNumber(prim.to_number()?))
        }
    }

    /// ECMAScript §7.1.18 **ToObject**.
    ///
    /// Returns `Err(TypeError)` for `undefined` and `null`.
    /// Primitive values are wrapped in a [`PlainObject`][Self::PlainObject]
    /// with a `"[[PrimitiveValue]]"` property holding the original value.
    /// Object-like values are returned unchanged.
    pub fn to_object(&self) -> StatorResult<JsValue> {
        match self {
            Self::Undefined => Err(StatorError::TypeError(
                "Cannot convert undefined to object".to_string(),
            )),
            Self::Null => Err(StatorError::TypeError(
                "Cannot convert null to object".to_string(),
            )),
            // Primitive wrappers.
            Self::Boolean(_)
            | Self::Smi(_)
            | Self::HeapNumber(_)
            | Self::String(_)
            | Self::Symbol(_)
            | Self::BigInt(_) => {
                let primitive_value = self.clone();
                let string_value = match self {
                    Self::Boolean(b) => {
                        JsValue::String(if *b { "true" } else { "false" }.to_string().into())
                    }
                    Self::Smi(n) => JsValue::String(n.to_string().into()),
                    Self::HeapNumber(n) => JsValue::String(number_to_string(*n).into()),
                    Self::String(s) => JsValue::String(s.clone()),
                    Self::Symbol(id) => {
                        let text = match symbol_description(*id) {
                            Some(desc) => format!("Symbol({desc})"),
                            None => "Symbol()".to_string(),
                        };
                        JsValue::String(text.into())
                    }
                    Self::BigInt(n) => JsValue::String(n.to_string().into()),
                    _ => unreachable!("only primitive wrappers reach this branch"),
                };
                let to_string_tag = match self {
                    Self::Boolean(_) => "Boolean",
                    Self::Smi(_) | Self::HeapNumber(_) => "Number",
                    Self::String(_) => "String",
                    Self::Symbol(_) => "Symbol",
                    Self::BigInt(_) => "BigInt",
                    _ => unreachable!("only primitive wrappers reach this branch"),
                };
                let primitive_for_value_of = primitive_value.clone();
                let string_for_to_string = string_value.clone();
                let mut map = PropertyMap::new();
                map.insert("[[PrimitiveValue]]".to_string(), primitive_value);
                if matches!(self, Self::Symbol(_)) {
                    map.insert("__wrapped__".to_string(), self.clone());
                }
                map.insert(
                    "@@toStringTag".to_string(),
                    JsValue::String(to_string_tag.to_string().into()),
                );
                map.insert(
                    "valueOf".to_string(),
                    JsValue::NativeFunction(Rc::new(move |_| Ok(primitive_for_value_of.clone()))),
                );
                map.insert(
                    "toString".to_string(),
                    JsValue::NativeFunction(Rc::new(move |_| Ok(string_for_to_string.clone()))),
                );
                if let Self::String(s) = self {
                    map.insert_with_attrs(
                        "length".to_string(),
                        JsValue::Smi(s.encode_utf16().count() as i32),
                        PropertyAttributes::empty(),
                    );
                    let utf16: Vec<u16> = s.encode_utf16().collect();
                    for (index, unit) in utf16.iter().enumerate() {
                        let ch = String::from_utf16_lossy(std::slice::from_ref(unit));
                        map.insert(index.to_string(), JsValue::String(ch.into()));
                    }
                }
                let ctor_name = match self {
                    Self::Boolean(_) => Some("Boolean"),
                    Self::Smi(_) | Self::HeapNumber(_) => Some("Number"),
                    Self::String(_) => Some("String"),
                    Self::Symbol(_) => Some("Symbol"),
                    Self::BigInt(_) => Some("BigInt"),
                    _ => None,
                };
                if let Some(ctor_name) = ctor_name
                    && let Some(globals) = crate::interpreter::current_global_env()
                {
                    let prototype = {
                        let globals = globals.borrow();
                        globals.get(ctor_name).and_then(|ctor| match ctor {
                            JsValue::PlainObject(map) => map.borrow().get("prototype").cloned(),
                            _ => None,
                        })
                    };
                    if let Some(prototype) = prototype {
                        map.insert("__proto__".to_string(), prototype);
                    }
                }
                map.make_all_non_enumerable();
                if matches!(self, Self::String(_)) {
                    for raw_key in map.keys().cloned().collect::<Vec<_>>() {
                        if raw_key.chars().all(|ch| ch.is_ascii_digit()) {
                            map.set_enumerable(&raw_key, true);
                        }
                    }
                }
                Ok(JsValue::PlainObject(Rc::new(RefCell::new(map))))
            }
            // Object-like types return themselves.
            _ => Ok(self.clone()),
        }
    }

    /// ECMAScript §7.1.19 **ToPropertyKey**.
    ///
    /// Converts via [`to_primitive`][Self::to_primitive] with a `String` hint,
    /// then returns the string representation. Symbols are encoded using the
    /// engine's internal symbol-property-key format so they remain distinct
    /// from user-visible string keys.
    pub fn to_property_key(&self) -> StatorResult<String> {
        let key = self.to_primitive(ToPrimitiveHint::String)?;
        match key {
            Self::Symbol(id) => Ok(crate::builtins::symbol::symbol_to_property_key(id)),
            other => other.to_js_string(),
        }
    }

    /// ECMAScript §20.1.3.6 **Object.prototype.toString** tag.
    ///
    /// Returns the `[object X]` classification string for any [`JsValue`].
    /// Used by `Object.prototype.toString.call(value)` to identify types.
    pub fn obj_to_string_tag(&self) -> String {
        match self {
            Self::Undefined => "[object Undefined]".to_string(),
            Self::Null => "[object Null]".to_string(),
            Self::Boolean(_) => "[object Boolean]".to_string(),
            Self::Smi(_) | Self::HeapNumber(_) => "[object Number]".to_string(),
            Self::String(_) => "[object String]".to_string(),
            Self::Symbol(_) => "[object Symbol]".to_string(),
            Self::BigInt(_) => "[object BigInt]".to_string(),
            Self::Array(_) => "[object Array]".to_string(),
            Self::Function(ba) => {
                if ba.is_generator() {
                    "[object GeneratorFunction]".to_string()
                } else {
                    "[object Function]".to_string()
                }
            }
            Self::NativeFunction(_) => "[object Function]".to_string(),
            Self::Error(_) => "[object Error]".to_string(),
            Self::Generator(_) => "[object Generator]".to_string(),
            Self::Iterator(_) => "[object Iterator]".to_string(),
            Self::Promise(_) => "[object Promise]".to_string(),
            Self::ArrayBuffer(buf) => {
                if buf.borrow().shared {
                    "[object SharedArrayBuffer]".to_string()
                } else {
                    "[object ArrayBuffer]".to_string()
                }
            }
            Self::TypedArray(ta) => format!("[object {}]", ta.borrow().kind.name()),
            Self::DataView(_) => "[object DataView]".to_string(),
            Self::PlainObject(map) => {
                let borrow = map.borrow();
                if let Some(Self::String(tag)) = borrow.get("@@toStringTag").cloned() {
                    return format!("[object {tag}]");
                }
                if matches!(borrow.get("__is_array__"), Some(Self::Boolean(true))) {
                    return "[object Array]".to_string();
                }
                if matches!(borrow.get("__is_regexp__"), Some(Self::Boolean(true))) {
                    return "[object RegExp]".to_string();
                }
                if matches!(borrow.get("__is_date__"), Some(Self::Boolean(true))) {
                    return "[object Date]".to_string();
                }
                if borrow.get("__call__").is_some() {
                    return "[object Function]".to_string();
                }
                if matches!(borrow.get("__is_error__"), Some(Self::Boolean(true))) {
                    return "[object Error]".to_string();
                }
                "[object Object]".to_string()
            }
            _ => "[object Object]".to_string(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Abstract comparison operations (ECMAScript §7.2)
// ──────────────────────────────────────────────────────────────────────────────

impl JsValue {
    /// ECMAScript §7.2.10 **SameValue(x, y)**.
    ///
    /// Like `===` except:
    /// * `NaN` is considered equal to `NaN`.
    /// * `+0` is **not** considered equal to `-0`.
    pub fn same_value(&self, other: &JsValue) -> bool {
        match (self, other) {
            (Self::HeapNumber(x), Self::HeapNumber(y)) => {
                if x.is_nan() && y.is_nan() {
                    return true;
                }
                // Distinguish +0 and -0.
                if *x == 0.0 && *y == 0.0 {
                    return x.is_sign_positive() == y.is_sign_positive();
                }
                x == y
            }
            // Smi(0) is always +0.
            (Self::Smi(x), Self::HeapNumber(y)) => {
                if *x == 0 && *y == 0.0 {
                    return y.is_sign_positive();
                }
                f64::from(*x) == *y
            }
            (Self::HeapNumber(x), Self::Smi(y)) => {
                if *x == 0.0 && *y == 0 {
                    return x.is_sign_positive();
                }
                *x == f64::from(*y)
            }
            _ => self == other,
        }
    }

    /// ECMAScript §7.2.11 **SameValueZero(x, y)**.
    ///
    /// Like [`same_value`](Self::same_value) except `+0` **is** considered
    /// equal to `-0`.  Used by `Map`, `Set`, and `Array.prototype.includes`.
    pub fn same_value_zero(&self, other: &JsValue) -> bool {
        match (self, other) {
            (Self::HeapNumber(x), Self::HeapNumber(y)) => {
                if x.is_nan() && y.is_nan() {
                    true
                } else {
                    x == y
                }
            }
            (Self::Smi(x), Self::HeapNumber(y)) => f64::from(*x) == *y,
            (Self::HeapNumber(x), Self::Smi(y)) => *x == f64::from(*y),
            _ => self == other,
        }
    }

    /// ECMAScript §7.2.13 **IsLooselyEqual(x, y)** — the `==` operator.
    ///
    /// Implements the full 10-step algorithm including ToPrimitive coercion,
    /// null/undefined equivalence, and boolean→number promotion.
    pub fn is_loosely_equal(&self, other: &JsValue) -> StatorResult<bool> {
        is_loosely_equal_inner(self, other, 0)
    }

    /// ECMAScript §7.2.15 **IsStrictlyEqual(x, y)** — the `===` operator.
    ///
    /// No type coercion is performed.  `NaN !== NaN` and `+0 === -0`
    /// per IEEE 754 semantics.
    pub fn is_strictly_equal(&self, other: &JsValue) -> bool {
        match (self, other) {
            (Self::Undefined, Self::Undefined) | (Self::Null, Self::Null) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Symbol(a), Self::Symbol(b)) => a == b,
            (Self::BigInt(a), Self::BigInt(b)) => a == b,
            // Numeric — IEEE 754: NaN !== NaN handled by f64 PartialEq.
            (Self::Smi(a), Self::Smi(b)) => a == b,
            (Self::HeapNumber(a), Self::HeapNumber(b)) => a == b,
            (Self::Smi(a), Self::HeapNumber(b)) => f64::from(*a) == *b,
            (Self::HeapNumber(a), Self::Smi(b)) => *a == f64::from(*b),
            // Object identity.
            (Self::Object(a), Self::Object(b)) => std::ptr::eq(*a, *b),
            (Self::Function(a), Self::Function(b)) => Rc::ptr_eq(a, b),
            (Self::Array(a), Self::Array(b)) => Rc::ptr_eq(a, b),
            (Self::Generator(a), Self::Generator(b)) => Rc::ptr_eq(a, b),
            (Self::Iterator(a), Self::Iterator(b)) => Rc::ptr_eq(a, b),
            (Self::Error(a), Self::Error(b)) => Rc::ptr_eq(a, b),
            (Self::NativeFunction(a), Self::NativeFunction(b)) => Rc::ptr_eq(a, b),
            (Self::PlainObject(a), Self::PlainObject(b)) => Rc::ptr_eq(a, b),
            (Self::Promise(a), Self::Promise(b)) => a == b,
            (Self::Context(a), Self::Context(b)) => Rc::ptr_eq(a, b),
            (Self::Proxy(a), Self::Proxy(b)) => Rc::ptr_eq(a, b),
            (Self::ArrayBuffer(a), Self::ArrayBuffer(b)) => Rc::ptr_eq(a, b),
            (Self::TypedArray(a), Self::TypedArray(b)) => Rc::ptr_eq(a, b),
            (Self::DataView(a), Self::DataView(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// ECMAScript §7.2.14 **IsLessThan(x, y, LeftFirst)**.
    ///
    /// Performs the Abstract Relational Comparison algorithm:
    /// 1. Converts both operands to primitives via [`ToPrimitive`] with a
    ///    `Number` hint.
    /// 2. If **both** primitives are strings, compares them lexicographically.
    /// 3. Otherwise converts to numeric values and compares numerically,
    ///    including mixed `BigInt` × `Number` comparisons.
    ///
    /// Returns `Ok(None)` (**undefined**) when either operand is `NaN`,
    /// `Ok(Some(true))` when `x < y`, and `Ok(Some(false))` otherwise.
    ///
    /// The `left_first` parameter controls evaluation order (ES spec
    /// §7.2.14 step 1 vs step 2). Pass `true` for `<` and `<=`, `false`
    /// for `>` and `>=`.
    pub fn abstract_relational_comparison(
        x: &JsValue,
        y: &JsValue,
        left_first: bool,
    ) -> StatorResult<Option<bool>> {
        // Step 1/2: ToPrimitive with hint Number.
        let (px, py) = if left_first {
            let px = x.to_primitive(ToPrimitiveHint::Number)?;
            let py = y.to_primitive(ToPrimitiveHint::Number)?;
            (px, py)
        } else {
            let py = y.to_primitive(ToPrimitiveHint::Number)?;
            let px = x.to_primitive(ToPrimitiveHint::Number)?;
            (px, py)
        };

        // Step 3: If both are strings, compare by UTF-16 code units (ES spec).
        if let (JsValue::String(a), JsValue::String(b)) = (&px, &py) {
            return Ok(Some(compare_utf16(a, b) == std::cmp::Ordering::Less));
        }

        // Step 4a: BigInt × String
        if let (JsValue::BigInt(n), JsValue::String(s)) = (&px, &py) {
            return Ok(s.trim().parse::<i128>().ok().map(|parsed| *n < parsed));
        }
        // Step 4b: String × BigInt
        if let (JsValue::String(s), JsValue::BigInt(n)) = (&px, &py) {
            return Ok(s.trim().parse::<i128>().ok().map(|parsed| parsed < *n));
        }

        // Step 4d-e: ToNumeric on both sides.
        let nx = px.to_numeric()?;
        let ny = py.to_numeric()?;

        // Step 4f: Same numeric type.
        match (&nx, &ny) {
            // Number × Number
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                if a.is_nan() || b.is_nan() {
                    return Ok(None); // undefined
                }
                Ok(Some(a < b))
            }
            // BigInt × BigInt
            (JsValue::BigInt(a), JsValue::BigInt(b)) => Ok(Some(a < b)),
            // Step 4g-h: BigInt × Number
            (JsValue::BigInt(a), JsValue::HeapNumber(b)) => {
                if b.is_nan() {
                    return Ok(None);
                }
                if b.is_infinite() {
                    return Ok(Some(b.is_sign_positive()));
                }
                Ok(Some((*a as f64) < *b))
            }
            // Number × BigInt
            (JsValue::HeapNumber(a), JsValue::BigInt(b)) => {
                if a.is_nan() {
                    return Ok(None);
                }
                if a.is_infinite() {
                    return Ok(Some(a.is_sign_negative()));
                }
                Ok(Some(*a < (*b as f64)))
            }
            _ => Ok(None),
        }
    }

    /// Convenience: `x < y` using [`abstract_relational_comparison`] with
    /// `left_first = true`.
    ///
    /// Returns `false` when the spec result is **undefined** (i.e. NaN
    /// comparisons), matching the runtime semantics of the `<` operator.
    pub fn js_less_than(x: &JsValue, y: &JsValue) -> StatorResult<bool> {
        Ok(Self::abstract_relational_comparison(x, y, true)?.unwrap_or(false))
    }

    /// Convenience: `x > y` — equivalent to `IsLessThan(y, x, false)`.
    pub fn js_greater_than(x: &JsValue, y: &JsValue) -> StatorResult<bool> {
        Ok(Self::abstract_relational_comparison(y, x, false)?.unwrap_or(false))
    }

    /// Convenience: `x <= y` — equivalent to `!(y < x)`.
    pub fn js_less_than_or_equal(x: &JsValue, y: &JsValue) -> StatorResult<bool> {
        Ok(Self::abstract_relational_comparison(y, x, false)?
            .map(|r| !r)
            .unwrap_or(false))
    }

    /// Convenience: `x >= y` — equivalent to `!(x < y)`.
    pub fn js_greater_than_or_equal(x: &JsValue, y: &JsValue) -> StatorResult<bool> {
        Ok(Self::abstract_relational_comparison(x, y, true)?
            .map(|r| !r)
            .unwrap_or(false))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GC Trace
// ──────────────────────────────────────────────────────────────────────────────

impl Trace for JsValue {
    /// Report any GC-managed heap pointer embedded in this value to the tracer.
    ///
    /// The [`JsValue::Object`] variant holds a raw heap pointer that is
    /// reported directly.  [`JsValue::Array`] may contain nested `Object`
    /// values, so each element is traced recursively.  [`JsValue::Generator`]
    /// traces the saved register file.  [`JsValue::Iterator`] traces its item
    /// vector.  All other variants carry no GC reference and are silently
    /// ignored.
    fn trace(&self, tracer: &mut Tracer) {
        match self {
            Self::Object(ptr) => {
                // SAFETY: Object pointers must refer to live, GC-managed HeapObjects.
                // The caller is responsible for ensuring the value is not used after
                // a collection that may have freed or moved the object.
                unsafe { tracer.mark_raw(*ptr as *mut u8) };
            }
            Self::Array(items) => {
                for item in items.borrow().iter() {
                    item.trace(tracer);
                }
            }
            Self::Generator(state) => {
                if let Some(proto) = &state.borrow().prototype {
                    proto.trace(tracer);
                }
                for reg in &state.borrow().registers {
                    reg.trace(tracer);
                }
            }
            Self::Iterator(iter) => {
                for item in &iter.borrow().items {
                    item.trace(tracer);
                }
            }
            Self::Context(ctx) => {
                for slot in &ctx.borrow().slots {
                    slot.trace(tracer);
                }
                // Parent contexts are traced transitively through the slots/parent chain.
                if let Some(parent) = &ctx.borrow().parent {
                    JsValue::Context(Rc::clone(parent)).trace(tracer);
                }
            }
            // JsError (including AggregateError inner errors) contains no raw
            // GC heap pointers — only Strings, Rc-reference-counted JsErrors,
            // and an ErrorKind enum.  Nothing to report to the tracer.
            Self::Error(_) => {}
            // JsPromise uses Rc<RefCell<_>> internally with no raw GC pointers.
            Self::Promise(_) => {}
            // JsProxy uses Rc<RefCell<_>> internally with no raw GC pointers.
            Self::Proxy(_) => {}
            // ArrayBuffer, TypedArray, DataView use Rc<RefCell<_>> — no raw GC pointers.
            Self::ArrayBuffer(_) | Self::TypedArray(_) | Self::DataView(_) => {}
            _ => {}
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compare two strings by their UTF-16 code units, as required by the
/// ECMAScript specification for relational comparison of strings.
///
/// This differs from Rust's default `str` comparison (which is byte-level
/// UTF-8, i.e. code-point order) only for strings containing supplementary
/// characters (above U+FFFF), where UTF-16 surrogate pairs sort differently
/// than their code-point order.
fn compare_utf16(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_units = a.encode_utf16();
    let mut b_units = b.encode_utf16();
    loop {
        match (a_units.next(), b_units.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(au), Some(bu)) => match au.cmp(&bu) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            },
        }
    }
}

/// Formats an `f64` as a JavaScript number string (ECMAScript §7.1.12.1).
///
/// Special cases: `NaN → "NaN"`, `+∞ → "Infinity"`, `-∞ → "-Infinity"`,
/// and both `+0.0` and `-0.0` → `"0"`.
///
/// For values ≥ 10²¹ or in (0, 10⁻⁶) ECMAScript mandates exponential
/// notation with an explicit `+` sign on positive exponents (e.g. `"1e+21"`).
/// All other finite values use the shortest decimal representation via Rust's
/// `Display` formatting.
pub(crate) fn number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n == 0.0 {
        // Both +0.0 and -0.0 produce "0".
        return "0".to_string();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    // Negative: "-" + NumberToString(abs(n)).
    if n < 0.0 {
        return format!("-{}", number_to_string(-n));
    }

    // ECMAScript §7.1.12.1: values >= 1e21 or < 1e-6 use exponential notation.
    if !(1e-6..1e21).contains(&n) {
        let s = format!("{n:e}");
        // Rust omits the '+' on positive exponents; ECMAScript requires it.
        if let Some(e_pos) = s.find('e') {
            let (mantissa, rest) = s.split_at(e_pos);
            let exp = &rest[1..]; // skip 'e'
            return if exp.starts_with('-') {
                format!("{mantissa}e{exp}")
            } else {
                format!("{mantissa}e+{exp}")
            };
        }
        return s;
    }

    // Normal range: Rust's Display produces the correct shortest representation.
    format!("{n}")
}

/// ECMAScript §7.1.4.1 **StringToNumber** — parse a string into a numeric
/// value.
///
/// Handles `"Infinity"`, `"+Infinity"`, `"-Infinity"`, hexadecimal (`0x`),
/// octal (`0o`), and binary (`0b`) integer literals, in addition to standard
/// decimal notation.  Returns `NaN` for strings that are not valid ECMAScript
/// numeric literals.
pub(crate) fn string_to_number(s: &str) -> f64 {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return 0.0;
    }

    // ECMAScript allows "Infinity", "+Infinity", "-Infinity".
    match trimmed {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }

    // Hex integer literal: 0x / 0X.
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        return u64::from_str_radix(hex, 16)
            .map(|n| n as f64)
            .unwrap_or_else(|_| parse_radix_digits(hex, 16));
    }
    // Octal integer literal: 0o / 0O.
    if let Some(oct) = trimmed
        .strip_prefix("0o")
        .or_else(|| trimmed.strip_prefix("0O"))
    {
        return u64::from_str_radix(oct, 8)
            .map(|n| n as f64)
            .unwrap_or_else(|_| parse_radix_digits(oct, 8));
    }
    // Binary integer literal: 0b / 0B.
    if let Some(bin) = trimmed
        .strip_prefix("0b")
        .or_else(|| trimmed.strip_prefix("0B"))
    {
        return u64::from_str_radix(bin, 2)
            .map(|n| n as f64)
            .unwrap_or_else(|_| parse_radix_digits(bin, 2));
    }

    // Reject Rust-specific float strings not valid in ECMAScript (e.g.
    // "inf", "infinity", "nan", "+inf").
    let check = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if check.eq_ignore_ascii_case("inf")
        || check.eq_ignore_ascii_case("infinity")
        || check.eq_ignore_ascii_case("nan")
    {
        return f64::NAN;
    }

    trimmed.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse a radix-prefixed digit string into an `f64` by accumulating
/// digit-by-digit.
///
/// Used as a fallback when the integer value exceeds [`u64::MAX`], which
/// [`u64::from_str_radix`] cannot handle.  Returns [`f64::NAN`] for empty
/// input or invalid digit characters.
fn parse_radix_digits(s: &str, radix: u32) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let base = f64::from(radix);
    s.chars()
        .try_fold(0.0_f64, |acc, c| {
            let digit = c.to_digit(radix)?;
            Some(acc.mul_add(base, f64::from(digit)))
        })
        .unwrap_or(f64::NAN)
}

fn to_primitive_hint_str(hint: ToPrimitiveHint) -> &'static str {
    match hint {
        ToPrimitiveHint::Default => "default",
        ToPrimitiveHint::Number => "number",
        ToPrimitiveHint::String => "string",
    }
}

fn is_callable_for_to_primitive(value: &JsValue) -> bool {
    match value {
        JsValue::Function(_) | JsValue::NativeFunction(_) => true,
        JsValue::PlainObject(map) => map.borrow().contains_key("__call__"),
        JsValue::Proxy(proxy) => proxy.borrow().is_callable(),
        _ => false,
    }
}

fn call_to_primitive_method(
    callee: &JsValue,
    receiver: &JsValue,
    args: Vec<JsValue>,
) -> StatorResult<JsValue> {
    match callee {
        JsValue::Function(_) | JsValue::Proxy(_) => {
            crate::interpreter::dispatch_call_with_this(callee, receiver.clone(), args)
        }
        JsValue::NativeFunction(f) => {
            let mut call_args = Vec::with_capacity(args.len() + 1);
            call_args.push(receiver.clone());
            call_args.extend(args);
            f(call_args)
        }
        JsValue::PlainObject(map) => {
            let call_fn = map
                .borrow()
                .get("__call__")
                .cloned()
                .ok_or_else(|| StatorError::TypeError("value is not a function".to_string()))?;
            call_to_primitive_method(&call_fn, receiver, args)
        }
        _ => Err(StatorError::TypeError(
            "value is not a function".to_string(),
        )),
    }
}

/// OrdinaryToPrimitive for object-like values (ECMAScript §7.1.1.1).
///
/// Looks for callable properties named `"valueOf"` / `"toString"` in the order
/// dictated by `hint`. Returns the first primitive result or throws a
/// `TypeError` if neither method produces one.
fn ordinary_to_primitive(value: &JsValue, hint: ToPrimitiveHint) -> StatorResult<JsValue> {
    let method_names: [&str; 2] = match hint {
        ToPrimitiveHint::String => ["toString", "valueOf"],
        ToPrimitiveHint::Number | ToPrimitiveHint::Default => ["valueOf", "toString"],
    };

    for name in &method_names {
        let maybe_method = crate::interpreter::dispatch_get_property_value(
            value,
            JsValue::String((*name).into()),
        )?;
        if !is_callable_for_to_primitive(&maybe_method) {
            continue;
        }
        let result = call_to_primitive_method(&maybe_method, value, Vec::new())?;
        if result.is_primitive() {
            return Ok(result);
        }
    }

    Err(StatorError::TypeError(
        "Cannot convert object to primitive value".into(),
    ))
}

/// ECMAScript §7.1.6 helper: truncate an `f64` to a signed 32-bit integer.
fn f64_to_int32(n: f64) -> i32 {
    if n.is_nan() || n.is_infinite() || n == 0.0 {
        return 0;
    }
    let int32bit = n.trunc().rem_euclid(4_294_967_296.0); // 2^32
    if int32bit >= 2_147_483_648.0 {
        // 2^31
        (int32bit - 4_294_967_296.0) as i32
    } else {
        int32bit as i32
    }
}

/// ECMAScript §7.1.7 helper: truncate an `f64` to an unsigned 32-bit integer.
fn f64_to_uint32(n: f64) -> u32 {
    if n.is_nan() || n.is_infinite() || n == 0.0 {
        return 0;
    }
    n.trunc().rem_euclid(4_294_967_296.0) as u32 // 2^32
}

/// ECMAScript §7.1.9 helper: truncate an `f64` to a signed 16-bit integer.
fn f64_to_int16(n: f64) -> i16 {
    if n.is_nan() || n.is_infinite() || n == 0.0 {
        return 0;
    }
    let int16bit = n.trunc().rem_euclid(65536.0); // 2^16
    if int16bit >= 32768.0 {
        // 2^15
        (int16bit - 65536.0) as i16
    } else {
        int16bit as i16
    }
}

/// Recursion-limited implementation of IsLooselyEqual (ECMAScript §7.2.13).
///
/// The `depth` parameter guards against infinite recursion caused by cyclic
/// ToPrimitive / ToNumber coercions.
fn is_loosely_equal_inner(lhs: &JsValue, rhs: &JsValue, depth: u8) -> StatorResult<bool> {
    const MAX_DEPTH: u8 = 8;
    if depth > MAX_DEPTH {
        return Ok(false);
    }

    // Step 1: Same ECMAScript type → strict equality.
    if es_type(lhs) == es_type(rhs) {
        return Ok(lhs.is_strictly_equal(rhs));
    }

    // Step 2: null == undefined (and vice versa).
    if (lhs.is_null() && rhs.is_undefined()) || (lhs.is_undefined() && rhs.is_null()) {
        return Ok(true);
    }

    // Steps 3-4: Number vs String → coerce string to number.
    if lhs.is_number() && rhs.is_string() {
        let y = JsValue::HeapNumber(string_to_number(
            rhs.to_js_string().unwrap_or_default().as_str(),
        ));
        return is_loosely_equal_inner(lhs, &y, depth + 1);
    }
    if lhs.is_string() && rhs.is_number() {
        let x = JsValue::HeapNumber(string_to_number(
            lhs.to_js_string().unwrap_or_default().as_str(),
        ));
        return is_loosely_equal_inner(&x, rhs, depth + 1);
    }

    // Steps 5-6: BigInt vs String.
    if lhs.is_bigint() && rhs.is_string() {
        if let JsValue::String(s) = rhs
            && let Ok(n) = s.trim().parse::<i128>()
        {
            return is_loosely_equal_inner(lhs, &JsValue::BigInt(n), depth + 1);
        }
        return Ok(false);
    }
    if lhs.is_string() && rhs.is_bigint() {
        return is_loosely_equal_inner(rhs, lhs, depth + 1);
    }

    // Steps 7-8: Boolean on either side → coerce to number.
    if let JsValue::Boolean(b) = lhs {
        return is_loosely_equal_inner(&JsValue::Smi(i32::from(*b)), rhs, depth + 1);
    }
    if let JsValue::Boolean(b) = rhs {
        return is_loosely_equal_inner(lhs, &JsValue::Smi(i32::from(*b)), depth + 1);
    }

    // Steps 9-10: Object vs primitive → ToPrimitive.
    if (lhs.is_string() || lhs.is_number() || lhs.is_bigint() || lhs.is_symbol())
        && rhs.is_object_like()
    {
        let y_prim = rhs.to_primitive(ToPrimitiveHint::Default)?;
        return is_loosely_equal_inner(lhs, &y_prim, depth + 1);
    }
    if lhs.is_object_like()
        && (rhs.is_string() || rhs.is_number() || rhs.is_bigint() || rhs.is_symbol())
    {
        let x_prim = lhs.to_primitive(ToPrimitiveHint::Default)?;
        return is_loosely_equal_inner(&x_prim, rhs, depth + 1);
    }

    // Steps 11-12: BigInt vs Number.
    if let (JsValue::BigInt(x), _) = (lhs, rhs)
        && rhs.is_number()
    {
        let y = rhs.to_number()?;
        if y.is_nan() || y.is_infinite() {
            return Ok(false);
        }
        // Compare exactly: the BigInt must equal the truncated f64.
        return Ok(*x as f64 == y && y as i128 == *x);
    }
    if let (_, JsValue::BigInt(y)) = (lhs, rhs)
        && lhs.is_number()
    {
        let x = lhs.to_number()?;
        if x.is_nan() || x.is_infinite() {
            return Ok(false);
        }
        return Ok(*y as f64 == x && x as i128 == *y);
    }

    // Step 13: Return false.
    Ok(false)
}

/// Map a [`JsValue`] to its ECMAScript language type (§6.1).
fn es_type(v: &JsValue) -> u8 {
    match v {
        JsValue::Undefined => 0,
        JsValue::Null => 1,
        JsValue::Boolean(_) => 2,
        JsValue::Smi(_) | JsValue::HeapNumber(_) => 3,
        JsValue::String(_) => 4,
        JsValue::Symbol(_) => 5,
        JsValue::BigInt(_) => 6,
        // All object-like variants share the same type tag.
        _ => 7,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::heap_object::HeapObject;

    // ── is_* predicates ──────────────────────────────────────────────────────

    #[test]
    fn test_is_undefined() {
        assert!(JsValue::Undefined.is_undefined());
        assert!(!JsValue::Null.is_undefined());
        assert!(!JsValue::Smi(0).is_undefined());
    }

    #[test]
    fn test_is_null() {
        assert!(JsValue::Null.is_null());
        assert!(!JsValue::Undefined.is_null());
        assert!(!JsValue::Boolean(false).is_null());
    }

    #[test]
    fn test_is_nullish() {
        assert!(JsValue::Undefined.is_nullish());
        assert!(JsValue::Null.is_nullish());
        assert!(!JsValue::Boolean(false).is_nullish());
        assert!(!JsValue::Smi(0).is_nullish());
    }

    #[test]
    fn test_is_boolean() {
        assert!(JsValue::Boolean(true).is_boolean());
        assert!(JsValue::Boolean(false).is_boolean());
        assert!(!JsValue::Smi(0).is_boolean());
    }

    #[test]
    fn test_is_smi() {
        assert!(JsValue::Smi(42).is_smi());
        assert!(!JsValue::HeapNumber(42.0).is_smi());
    }

    #[test]
    fn test_is_heap_number() {
        assert!(JsValue::HeapNumber(3.14).is_heap_number());
        assert!(!JsValue::Smi(3).is_heap_number());
    }

    #[test]
    fn test_is_number() {
        assert!(JsValue::Smi(0).is_number());
        assert!(JsValue::HeapNumber(0.0).is_number());
        assert!(!JsValue::Boolean(false).is_number());
        assert!(!JsValue::Null.is_number());
    }

    #[test]
    fn test_is_string() {
        assert!(JsValue::String("hello".to_string().into()).is_string());
        assert!(!JsValue::Smi(0).is_string());
    }

    #[test]
    fn test_is_symbol() {
        assert!(JsValue::Symbol(1).is_symbol());
        assert!(!JsValue::String("sym".to_string().into()).is_symbol());
    }

    #[test]
    fn test_is_object() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(JsValue::Object(ptr).is_object());
        assert!(!JsValue::Null.is_object());
    }

    #[test]
    fn test_is_bigint() {
        assert!(JsValue::BigInt(0).is_bigint());
        assert!(!JsValue::Smi(0).is_bigint());
    }

    #[test]
    fn test_is_primitive() {
        assert!(JsValue::Undefined.is_primitive());
        assert!(JsValue::Null.is_primitive());
        assert!(JsValue::Boolean(true).is_primitive());
        assert!(JsValue::Smi(0).is_primitive());
        assert!(JsValue::HeapNumber(0.0).is_primitive());
        assert!(JsValue::String("".to_string().into()).is_primitive());
        assert!(JsValue::Symbol(0).is_primitive());
        assert!(JsValue::BigInt(0).is_primitive());
        assert!(!JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new()))).is_primitive());
        assert!(!JsValue::new_array(vec![]).is_primitive());
    }

    #[test]
    fn test_is_object_like() {
        assert!(!JsValue::Undefined.is_object_like());
        assert!(!JsValue::Smi(42).is_object_like());
        assert!(JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new()))).is_object_like());
        assert!(JsValue::new_array(vec![]).is_object_like());
    }

    // ── to_boolean ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_boolean_undefined_is_false() {
        assert!(!JsValue::Undefined.to_boolean());
    }

    #[test]
    fn test_to_boolean_null_is_false() {
        assert!(!JsValue::Null.to_boolean());
    }

    #[test]
    fn test_to_boolean_boolean_passthrough() {
        assert!(JsValue::Boolean(true).to_boolean());
        assert!(!JsValue::Boolean(false).to_boolean());
    }

    #[test]
    fn test_to_boolean_smi_zero_is_false() {
        assert!(!JsValue::Smi(0).to_boolean());
        assert!(JsValue::Smi(1).to_boolean());
        assert!(JsValue::Smi(-1).to_boolean());
    }

    #[test]
    fn test_to_boolean_heap_number_special_cases() {
        assert!(!JsValue::HeapNumber(0.0).to_boolean());
        assert!(!JsValue::HeapNumber(-0.0).to_boolean());
        assert!(!JsValue::HeapNumber(f64::NAN).to_boolean());
        assert!(JsValue::HeapNumber(1.0).to_boolean());
        assert!(JsValue::HeapNumber(-1.0).to_boolean());
        assert!(JsValue::HeapNumber(f64::INFINITY).to_boolean());
    }

    #[test]
    fn test_to_boolean_string() {
        assert!(!JsValue::String(String::new().into()).to_boolean());
        assert!(JsValue::String("x".to_string().into()).to_boolean());
        assert!(JsValue::String("false".to_string().into()).to_boolean());
    }

    #[test]
    fn test_to_boolean_symbol_is_true() {
        assert!(JsValue::Symbol(0).to_boolean());
        assert!(JsValue::Symbol(u64::MAX).to_boolean());
    }

    #[test]
    fn test_to_boolean_object_is_true() {
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert!(JsValue::Object(ptr).to_boolean());
    }

    #[test]
    fn test_to_boolean_bigint_zero_is_false() {
        assert!(!JsValue::BigInt(0).to_boolean());
        assert!(JsValue::BigInt(1).to_boolean());
        assert!(JsValue::BigInt(-1).to_boolean());
    }

    #[test]
    fn test_to_boolean_thehole_is_false() {
        assert!(!JsValue::TheHole.to_boolean());
    }

    #[test]
    fn test_to_boolean_function_is_true() {
        let f: NativeFn = Rc::new(|_| Ok(JsValue::Undefined));
        assert!(JsValue::NativeFunction(f).to_boolean());
    }

    #[test]
    fn test_to_boolean_array_is_true() {
        assert!(JsValue::new_array(vec![]).to_boolean());
    }

    #[test]
    fn test_to_boolean_plain_object_is_true() {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert!(obj.to_boolean());
    }

    // ── to_primitive ─────────────────────────────────────────────────────────

    #[test]
    fn test_to_primitive_primitives_passthrough() {
        let vals = [
            JsValue::Undefined,
            JsValue::Null,
            JsValue::Boolean(true),
            JsValue::Smi(42),
            JsValue::HeapNumber(3.14),
            JsValue::String("hi".to_string().into()),
            JsValue::Symbol(7),
            JsValue::BigInt(99),
        ];
        for v in &vals {
            assert_eq!(&v.to_primitive(ToPrimitiveHint::Default).unwrap(), v);
        }
    }

    #[test]
    fn test_to_primitive_plain_object_default() {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let prim = obj.to_primitive(ToPrimitiveHint::Default).unwrap();
        assert_eq!(prim, JsValue::String("[object Object]".to_string().into()));
    }

    #[test]
    fn test_to_primitive_thehole_becomes_undefined() {
        let prim = JsValue::TheHole
            .to_primitive(ToPrimitiveHint::Default)
            .unwrap();
        assert_eq!(prim, JsValue::Undefined);
    }

    #[test]
    fn test_to_primitive_plain_object_with_valueof() {
        let mut map = PropertyMap::new();
        let f: NativeFn = Rc::new(|_| Ok(JsValue::Smi(42)));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(f));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        // Number hint: valueOf is tried first.
        let prim = obj.to_primitive(ToPrimitiveHint::Number).unwrap();
        assert_eq!(prim, JsValue::Smi(42));
    }

    #[test]
    fn test_to_primitive_plain_object_with_tostring() {
        let mut map = PropertyMap::new();
        let f: NativeFn = Rc::new(|_| Ok(JsValue::String("custom".to_string().into())));
        map.insert("toString".to_string(), JsValue::NativeFunction(f));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        // String hint: toString is tried first.
        let prim = obj.to_primitive(ToPrimitiveHint::String).unwrap();
        assert_eq!(prim, JsValue::String("custom".to_string().into()));
    }

    #[test]
    fn test_to_primitive_array() {
        let arr = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let prim = arr.to_primitive(ToPrimitiveHint::String).unwrap();
        assert_eq!(prim, JsValue::String("1,2".to_string().into()));
    }

    #[test]
    fn test_to_primitive_with_symbol_to_primitive() {
        let mut map = PropertyMap::new();
        let f: NativeFn = Rc::new(|args| {
            let hint = args.get(1).unwrap_or(&JsValue::Undefined).clone();
            if let JsValue::String(h) = hint {
                Ok(JsValue::String(format!("hint:{h}").into()))
            } else {
                Ok(JsValue::Smi(0))
            }
        });
        map.insert("@@toPrimitive".to_string(), JsValue::NativeFunction(f));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        let prim = obj.to_primitive(ToPrimitiveHint::Number).unwrap();
        assert_eq!(prim, JsValue::String("hint:number".to_string().into()));
    }

    #[test]
    fn test_to_primitive_with_non_callable_symbol_to_primitive_is_type_error() {
        let mut map = PropertyMap::new();
        map.insert("@@toPrimitive".to_string(), JsValue::Smi(1));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(matches!(
            obj.to_primitive(ToPrimitiveHint::Number),
            Err(StatorError::TypeError(message)) if message == "Symbol.toPrimitive is not a function"
        ));
    }

    #[test]
    fn test_to_primitive_plain_object_both_methods_non_primitive_is_type_error() {
        let mut map = PropertyMap::new();
        let value_of: NativeFn = Rc::new(|_| Ok(JsValue::new_array(vec![])));
        let to_string: NativeFn = Rc::new(|_| Ok(JsValue::new_array(vec![])));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(value_of));
        map.insert("toString".to_string(), JsValue::NativeFunction(to_string));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(matches!(
            obj.to_primitive(ToPrimitiveHint::Number),
            Err(StatorError::TypeError(message)) if message == "Cannot convert object to primitive value"
        ));
    }

    #[test]
    fn test_to_string_tag() {
        let mut map = PropertyMap::new();
        map.insert(
            "@@toStringTag".to_string(),
            JsValue::String("CustomType".to_string().into()),
        );
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object CustomType]".to_string());
    }

    #[test]
    fn test_obj_to_string_tag_undefined() {
        assert_eq!(JsValue::Undefined.obj_to_string_tag(), "[object Undefined]");
    }

    #[test]
    fn test_obj_to_string_tag_null() {
        assert_eq!(JsValue::Null.obj_to_string_tag(), "[object Null]");
    }

    #[test]
    fn test_obj_to_string_tag_boolean() {
        assert_eq!(
            JsValue::Boolean(true).obj_to_string_tag(),
            "[object Boolean]"
        );
    }

    #[test]
    fn test_obj_to_string_tag_number() {
        assert_eq!(JsValue::Smi(42).obj_to_string_tag(), "[object Number]");
        assert_eq!(
            JsValue::HeapNumber(3.14).obj_to_string_tag(),
            "[object Number]"
        );
    }

    #[test]
    fn test_obj_to_string_tag_string() {
        assert_eq!(
            JsValue::String("hi".into()).obj_to_string_tag(),
            "[object String]"
        );
    }

    #[test]
    fn test_obj_to_string_tag_array() {
        let arr = JsValue::Array(Rc::new(RefCell::new(vec![])));
        assert_eq!(arr.obj_to_string_tag(), "[object Array]");
    }

    #[test]
    fn test_obj_to_string_tag_function() {
        let f: Rc<dyn Fn(Vec<JsValue>) -> crate::error::StatorResult<JsValue>> =
            Rc::new(|_| Ok(JsValue::Undefined));
        assert_eq!(
            JsValue::NativeFunction(f).obj_to_string_tag(),
            "[object Function]"
        );
    }

    #[test]
    fn test_obj_to_string_tag_error() {
        let e = JsValue::Error(Rc::new(crate::builtins::error::JsError::new(
            crate::builtins::error::ErrorKind::Error,
            "test".to_string(),
        )));
        assert_eq!(e.obj_to_string_tag(), "[object Error]");
    }

    #[test]
    fn test_obj_to_string_tag_plain_object() {
        let map = PropertyMap::new();
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object Object]");
    }

    #[test]
    fn test_obj_to_string_tag_custom_tag() {
        let mut map = PropertyMap::new();
        map.insert(
            "@@toStringTag".to_string(),
            JsValue::String("MyClass".into()),
        );
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object MyClass]");
    }

    #[test]
    fn test_obj_to_string_tag_regexp() {
        let mut map = PropertyMap::new();
        map.insert("__is_regexp__".to_string(), JsValue::Boolean(true));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object RegExp]");
    }

    // ── to_number ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_number_undefined_is_nan_v2() {
        let n = JsValue::Undefined.to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_null_is_zero_v2() {
        assert_eq!(JsValue::Null.to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_boolean() {
        assert_eq!(JsValue::Boolean(true).to_number().unwrap(), 1.0);
        assert_eq!(JsValue::Boolean(false).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_smi() {
        assert_eq!(JsValue::Smi(42).to_number().unwrap(), 42.0);
        assert_eq!(JsValue::Smi(-1).to_number().unwrap(), -1.0);
        assert_eq!(JsValue::Smi(0).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_heap_number_passthrough() {
        assert_eq!(JsValue::HeapNumber(3.14).to_number().unwrap(), 3.14);
        let n = JsValue::HeapNumber(f64::NAN).to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_string_numeric() {
        assert_eq!(
            JsValue::String("42".to_string().into())
                .to_number()
                .unwrap(),
            42.0
        );
        assert_eq!(
            JsValue::String("  3.14  ".to_string().into())
                .to_number()
                .unwrap(),
            3.14
        );
        assert_eq!(
            JsValue::String("".to_string().into()).to_number().unwrap(),
            0.0
        );
        assert_eq!(
            JsValue::String("   ".to_string().into())
                .to_number()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_to_number_string_non_numeric_gives_nan() {
        let n = JsValue::String("abc".to_string().into())
            .to_number()
            .unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_string_hex() {
        assert_eq!(
            JsValue::String("0xFF".to_string().into())
                .to_number()
                .unwrap(),
            255.0
        );
        assert_eq!(
            JsValue::String("0x1A".to_string().into())
                .to_number()
                .unwrap(),
            26.0
        );
        assert_eq!(
            JsValue::String("0X10".to_string().into())
                .to_number()
                .unwrap(),
            16.0
        );
    }

    #[test]
    fn test_to_number_string_octal() {
        assert_eq!(
            JsValue::String("0o17".to_string().into())
                .to_number()
                .unwrap(),
            15.0
        );
        assert_eq!(
            JsValue::String("0O10".to_string().into())
                .to_number()
                .unwrap(),
            8.0
        );
    }

    #[test]
    fn test_to_number_string_binary() {
        assert_eq!(
            JsValue::String("0b1010".to_string().into())
                .to_number()
                .unwrap(),
            10.0
        );
        assert_eq!(
            JsValue::String("0B11".to_string().into())
                .to_number()
                .unwrap(),
            3.0
        );
    }

    #[test]
    fn test_to_number_string_infinity() {
        assert_eq!(
            JsValue::String("Infinity".to_string().into())
                .to_number()
                .unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            JsValue::String("+Infinity".to_string().into())
                .to_number()
                .unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            JsValue::String("-Infinity".to_string().into())
                .to_number()
                .unwrap(),
            f64::NEG_INFINITY
        );
        // "inf" is not valid ECMAScript.
        let n = JsValue::String("inf".to_string().into())
            .to_number()
            .unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_symbol_is_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_number(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_to_number_object_via_to_primitive() {
        // Object goes through ToPrimitive → "[object Object]" → NaN.
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        let n = JsValue::Object(ptr).to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_plain_object_is_nan() {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let n = obj.to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_empty_array_is_zero() {
        let arr = JsValue::new_array(vec![]);
        assert_eq!(arr.to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_single_element_array() {
        let arr = JsValue::new_array(vec![JsValue::Smi(42)]);
        assert_eq!(arr.to_number().unwrap(), 42.0);
    }

    #[test]
    fn test_to_number_multi_element_array_is_nan() {
        let arr = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2)]);
        let n = arr.to_number().unwrap();
        assert!(n.is_nan());
    }

    #[test]
    fn test_to_number_bigint_is_type_error() {
        assert!(matches!(
            JsValue::BigInt(42).to_number(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_number_bigint_error_message() {
        let err = JsValue::BigInt(0).to_number().unwrap_err();
        match err {
            StatorError::TypeError(msg) => {
                assert_eq!(msg, "Cannot convert a BigInt value to a number");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_to_number_symbol_error_message() {
        let err = JsValue::Symbol(0).to_number().unwrap_err();
        match err {
            StatorError::TypeError(msg) => {
                assert_eq!(msg, "Cannot convert a Symbol value to a number");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_to_number_thehole_is_nan() {
        let n = JsValue::TheHole.to_number().unwrap();
        assert!(n.is_nan());
    }

    // ── to_js_string ─────────────────────────────────────────────────────────

    #[test]
    fn test_to_js_string_undefined_v2() {
        assert_eq!(JsValue::Undefined.to_js_string().unwrap(), "undefined");
    }

    #[test]
    fn test_to_js_string_null_v2() {
        assert_eq!(JsValue::Null.to_js_string().unwrap(), "null");
    }

    #[test]
    fn test_to_js_string_boolean() {
        assert_eq!(JsValue::Boolean(true).to_js_string().unwrap(), "true");
        assert_eq!(JsValue::Boolean(false).to_js_string().unwrap(), "false");
    }

    #[test]
    fn test_to_js_string_smi() {
        assert_eq!(JsValue::Smi(0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::Smi(42).to_js_string().unwrap(), "42");
        assert_eq!(JsValue::Smi(-7).to_js_string().unwrap(), "-7");
    }

    #[test]
    fn test_to_js_string_heap_number_special_cases() {
        assert_eq!(JsValue::HeapNumber(f64::NAN).to_js_string().unwrap(), "NaN");
        assert_eq!(
            JsValue::HeapNumber(f64::INFINITY).to_js_string().unwrap(),
            "Infinity"
        );
        assert_eq!(
            JsValue::HeapNumber(f64::NEG_INFINITY)
                .to_js_string()
                .unwrap(),
            "-Infinity"
        );
        assert_eq!(JsValue::HeapNumber(0.0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::HeapNumber(-0.0).to_js_string().unwrap(), "0");
    }

    #[test]
    fn test_to_js_string_heap_number_normal() {
        assert_eq!(JsValue::HeapNumber(42.0).to_js_string().unwrap(), "42");
        assert_eq!(JsValue::HeapNumber(3.14).to_js_string().unwrap(), "3.14");
    }

    #[test]
    fn test_to_js_string_string_passthrough() {
        assert_eq!(
            JsValue::String("hello".to_string().into())
                .to_js_string()
                .unwrap(),
            "hello"
        );
        assert_eq!(
            JsValue::String(String::new().into())
                .to_js_string()
                .unwrap(),
            ""
        );
    }

    #[test]
    fn test_to_js_string_symbol_is_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_js_string(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_js_string_thehole_is_undefined() {
        assert_eq!(JsValue::TheHole.to_js_string().unwrap(), "undefined");
    }

    #[test]
    #[ignore] // TODO: conformance — not yet passing
    fn test_to_js_string_object_via_to_primitive() {
        // Object goes through ToPrimitive → "[object Object]".
        let mut obj = HeapObject::new_null();
        let ptr = &raw mut obj;
        assert_eq!(
            JsValue::Object(ptr).to_js_string().unwrap(),
            "[object Object]"
        );
    }

    #[test]
    fn test_to_js_string_plain_object() {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert_eq!(obj.to_js_string().unwrap(), "[object Object]");
    }

    #[test]
    fn test_to_js_string_array() {
        let arr = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Null, JsValue::Smi(3)]);
        assert_eq!(arr.to_js_string().unwrap(), "1,,3");
    }

    #[test]
    fn test_to_js_string_bigint_v2() {
        assert_eq!(JsValue::BigInt(0).to_js_string().unwrap(), "0");
        assert_eq!(JsValue::BigInt(12345).to_js_string().unwrap(), "12345");
        assert_eq!(JsValue::BigInt(-99).to_js_string().unwrap(), "-99");
    }

    // ── to_int32 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_int32_basic() {
        assert_eq!(JsValue::Smi(42).to_int32().unwrap(), 42);
        assert_eq!(JsValue::Smi(-1).to_int32().unwrap(), -1);
        assert_eq!(JsValue::HeapNumber(3.7).to_int32().unwrap(), 3);
        assert_eq!(JsValue::HeapNumber(-3.7).to_int32().unwrap(), -3);
    }

    #[test]
    fn test_to_int32_special_values() {
        assert_eq!(JsValue::HeapNumber(f64::NAN).to_int32().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(f64::INFINITY).to_int32().unwrap(), 0);
        assert_eq!(
            JsValue::HeapNumber(f64::NEG_INFINITY).to_int32().unwrap(),
            0
        );
        assert_eq!(JsValue::HeapNumber(0.0).to_int32().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(-0.0).to_int32().unwrap(), 0);
    }

    #[test]
    fn test_to_int32_wraparound() {
        // 2^31 wraps to -2^31
        assert_eq!(
            JsValue::HeapNumber(2_147_483_648.0).to_int32().unwrap(),
            -2_147_483_648
        );
        // 2^32 wraps to 0
        assert_eq!(JsValue::HeapNumber(4_294_967_296.0).to_int32().unwrap(), 0);
        // 2^32 + 1 wraps to 1
        assert_eq!(JsValue::HeapNumber(4_294_967_297.0).to_int32().unwrap(), 1);
    }

    #[test]
    fn test_to_int32_null_is_zero() {
        assert_eq!(JsValue::Null.to_int32().unwrap(), 0);
    }

    #[test]
    fn test_to_int32_undefined_is_zero() {
        assert_eq!(JsValue::Undefined.to_int32().unwrap(), 0);
    }

    // ── to_uint32 ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_uint32_basic() {
        assert_eq!(JsValue::Smi(42).to_uint32().unwrap(), 42);
        assert_eq!(JsValue::HeapNumber(3.7).to_uint32().unwrap(), 3);
    }

    #[test]
    fn test_to_uint32_negative() {
        // -1 → 2^32 - 1 = 4294967295
        assert_eq!(JsValue::Smi(-1).to_uint32().unwrap(), 4_294_967_295);
    }

    #[test]
    fn test_to_uint32_special_values() {
        assert_eq!(JsValue::HeapNumber(f64::NAN).to_uint32().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(f64::INFINITY).to_uint32().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(0.0).to_uint32().unwrap(), 0);
    }

    #[test]
    fn test_to_uint32_wraparound() {
        // 2^32 wraps to 0
        assert_eq!(JsValue::HeapNumber(4_294_967_296.0).to_uint32().unwrap(), 0);
    }

    // ── to_int16 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_int16_basic() {
        assert_eq!(JsValue::Smi(42).to_int16().unwrap(), 42);
        assert_eq!(JsValue::Smi(-1).to_int16().unwrap(), -1);
    }

    #[test]
    fn test_to_int16_wraparound() {
        // 32768 wraps to -32768
        assert_eq!(JsValue::HeapNumber(32768.0).to_int16().unwrap(), -32768);
        // 65536 wraps to 0
        assert_eq!(JsValue::HeapNumber(65536.0).to_int16().unwrap(), 0);
    }

    #[test]
    fn test_to_int16_special_values() {
        assert_eq!(JsValue::HeapNumber(f64::NAN).to_int16().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(f64::INFINITY).to_int16().unwrap(), 0);
    }

    // ── to_object ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_object_undefined_is_type_error() {
        assert!(matches!(
            JsValue::Undefined.to_object(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_object_null_is_type_error() {
        assert!(matches!(
            JsValue::Null.to_object(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_to_object_primitive_wrapping() {
        let wrapper = JsValue::Smi(42).to_object().unwrap();
        if let JsValue::PlainObject(map) = wrapper {
            assert_eq!(
                map.borrow().get("[[PrimitiveValue]]"),
                Some(&JsValue::Smi(42))
            );
        } else {
            panic!("Expected PlainObject wrapper");
        }
    }

    #[test]
    fn test_to_object_object_passthrough() {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let result = obj.to_object().unwrap();
        assert!(result.is_object_like());
    }

    // ── to_property_key ──────────────────────────────────────────────────────

    #[test]
    fn test_to_property_key_string() {
        assert_eq!(
            JsValue::String("foo".to_string().into())
                .to_property_key()
                .unwrap(),
            "foo"
        );
    }

    #[test]
    fn test_to_property_key_number() {
        assert_eq!(JsValue::Smi(42).to_property_key().unwrap(), "42");
    }

    #[test]
    fn test_to_property_key_symbol() {
        assert_eq!(
            JsValue::Symbol(7).to_property_key().unwrap(),
            crate::builtins::symbol::symbol_to_property_key(7)
        );
    }

    #[test]
    fn test_to_property_key_null() {
        assert_eq!(JsValue::Null.to_property_key().unwrap(), "null");
    }

    // ── same_value ───────────────────────────────────────────────────────────

    #[test]
    fn test_same_value_nan_equals_nan() {
        assert!(JsValue::HeapNumber(f64::NAN).same_value(&JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_same_value_positive_zero_not_negative_zero() {
        assert!(!JsValue::HeapNumber(0.0).same_value(&JsValue::HeapNumber(-0.0)));
        assert!(!JsValue::HeapNumber(-0.0).same_value(&JsValue::HeapNumber(0.0)));
    }

    #[test]
    fn test_same_value_smi_zero_vs_negative_zero() {
        assert!(!JsValue::Smi(0).same_value(&JsValue::HeapNumber(-0.0)));
        assert!(!JsValue::HeapNumber(-0.0).same_value(&JsValue::Smi(0)));
    }

    #[test]
    fn test_same_value_positive_zero_same() {
        assert!(JsValue::HeapNumber(0.0).same_value(&JsValue::HeapNumber(0.0)));
        assert!(JsValue::Smi(0).same_value(&JsValue::HeapNumber(0.0)));
    }

    #[test]
    fn test_same_value_ordinary_numbers() {
        assert!(JsValue::HeapNumber(42.0).same_value(&JsValue::HeapNumber(42.0)));
        assert!(!JsValue::HeapNumber(1.0).same_value(&JsValue::HeapNumber(2.0)));
    }

    #[test]
    fn test_same_value_primitives() {
        assert!(JsValue::Smi(1).same_value(&JsValue::Smi(1)));
        assert!(!JsValue::Smi(1).same_value(&JsValue::Smi(2)));
        assert!(JsValue::Null.same_value(&JsValue::Null));
        assert!(!JsValue::Null.same_value(&JsValue::Undefined));
    }

    // ── same_value_zero ──────────────────────────────────────────────────────

    #[test]
    fn test_same_value_zero_nan_equals_nan() {
        assert!(JsValue::HeapNumber(f64::NAN).same_value_zero(&JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_same_value_zero_positive_zero_equals_negative_zero() {
        assert!(JsValue::HeapNumber(0.0).same_value_zero(&JsValue::HeapNumber(-0.0)));
        assert!(JsValue::HeapNumber(-0.0).same_value_zero(&JsValue::HeapNumber(0.0)));
    }

    #[test]
    fn test_same_value_zero_cross_number_types() {
        assert!(JsValue::Smi(0).same_value_zero(&JsValue::HeapNumber(0.0)));
        assert!(JsValue::HeapNumber(42.0).same_value_zero(&JsValue::Smi(42)));
    }

    #[test]
    fn test_same_value_zero_distinct_numbers() {
        assert!(!JsValue::HeapNumber(1.0).same_value_zero(&JsValue::HeapNumber(2.0)));
    }

    #[test]
    fn test_same_value_zero_primitives() {
        assert!(JsValue::Smi(42).same_value_zero(&JsValue::Smi(42)));
        assert!(!JsValue::Smi(1).same_value_zero(&JsValue::Smi(2)));
        assert!(JsValue::Null.same_value_zero(&JsValue::Null));
        assert!(!JsValue::Null.same_value_zero(&JsValue::Undefined));
    }

    // ── is_strictly_equal ────────────────────────────────────────────────────

    #[test]
    fn test_is_strictly_equal_same_type() {
        assert!(JsValue::Smi(1).is_strictly_equal(&JsValue::Smi(1)));
        assert!(!JsValue::Smi(1).is_strictly_equal(&JsValue::Smi(2)));
        assert!(
            JsValue::String("a".to_string().into())
                .is_strictly_equal(&JsValue::String("a".to_string().into()))
        );
    }

    #[test]
    fn test_is_strictly_equal_nan_not_equal() {
        assert!(!JsValue::HeapNumber(f64::NAN).is_strictly_equal(&JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_is_strictly_equal_zero_signs() {
        // +0 === -0 per IEEE 754.
        assert!(JsValue::HeapNumber(0.0).is_strictly_equal(&JsValue::HeapNumber(-0.0)));
    }

    #[test]
    fn test_is_strictly_equal_cross_type() {
        // Different types → false.
        assert!(!JsValue::Smi(1).is_strictly_equal(&JsValue::String("1".to_string().into())));
        assert!(!JsValue::Null.is_strictly_equal(&JsValue::Undefined));
    }

    #[test]
    fn test_is_strictly_equal_smi_heap_number() {
        assert!(JsValue::Smi(42).is_strictly_equal(&JsValue::HeapNumber(42.0)));
        assert!(JsValue::HeapNumber(42.0).is_strictly_equal(&JsValue::Smi(42)));
    }

    // ── is_loosely_equal ─────────────────────────────────────────────────────

    #[test]
    fn test_is_loosely_equal_same_type() {
        assert!(JsValue::Smi(1).is_loosely_equal(&JsValue::Smi(1)).unwrap());
        assert!(!JsValue::Smi(1).is_loosely_equal(&JsValue::Smi(2)).unwrap());
    }

    #[test]
    fn test_is_loosely_equal_null_undefined() {
        assert!(JsValue::Null.is_loosely_equal(&JsValue::Undefined).unwrap());
        assert!(JsValue::Undefined.is_loosely_equal(&JsValue::Null).unwrap());
    }

    #[test]
    fn test_is_loosely_equal_null_not_zero() {
        assert!(!JsValue::Null.is_loosely_equal(&JsValue::Smi(0)).unwrap());
        assert!(
            !JsValue::Null
                .is_loosely_equal(&JsValue::String("".to_string().into()))
                .unwrap()
        );
    }

    #[test]
    fn test_is_loosely_equal_number_string() {
        assert!(
            JsValue::Smi(1)
                .is_loosely_equal(&JsValue::String("1".to_string().into()))
                .unwrap()
        );
        assert!(
            JsValue::String("42".to_string().into())
                .is_loosely_equal(&JsValue::Smi(42))
                .unwrap()
        );
        assert!(
            !JsValue::Smi(1)
                .is_loosely_equal(&JsValue::String("2".to_string().into()))
                .unwrap()
        );
    }

    #[test]
    fn test_is_loosely_equal_boolean_coercion() {
        // true == 1
        assert!(
            JsValue::Boolean(true)
                .is_loosely_equal(&JsValue::Smi(1))
                .unwrap()
        );
        // false == 0
        assert!(
            JsValue::Boolean(false)
                .is_loosely_equal(&JsValue::Smi(0))
                .unwrap()
        );
        // true == "1"
        assert!(
            JsValue::Boolean(true)
                .is_loosely_equal(&JsValue::String("1".to_string().into()))
                .unwrap()
        );
        // false != "false"
        assert!(
            !JsValue::Boolean(false)
                .is_loosely_equal(&JsValue::String("false".to_string().into()))
                .unwrap()
        );
    }

    #[test]
    fn test_is_loosely_equal_object_to_primitive() {
        let arr = JsValue::new_array(vec![]);
        // [] == false → "" == 0 → 0 == 0 → true
        assert!(arr.is_loosely_equal(&JsValue::Boolean(false)).unwrap());
        // [] == 0 → "" == 0 → 0 == 0 → true
        assert!(
            JsValue::new_array(vec![])
                .is_loosely_equal(&JsValue::Smi(0))
                .unwrap()
        );
    }

    #[test]
    fn test_is_loosely_equal_bigint_number() {
        assert!(
            JsValue::BigInt(1)
                .is_loosely_equal(&JsValue::Smi(1))
                .unwrap()
        );
        assert!(
            !JsValue::BigInt(1)
                .is_loosely_equal(&JsValue::Smi(2))
                .unwrap()
        );
        // NaN != BigInt
        assert!(
            !JsValue::BigInt(0)
                .is_loosely_equal(&JsValue::HeapNumber(f64::NAN))
                .unwrap()
        );
    }

    #[test]
    fn test_is_loosely_equal_bigint_string() {
        assert!(
            JsValue::BigInt(42)
                .is_loosely_equal(&JsValue::String("42".to_string().into()))
                .unwrap()
        );
        assert!(
            !JsValue::BigInt(42)
                .is_loosely_equal(&JsValue::String("abc".to_string().into()))
                .unwrap()
        );
    }

    // ── string_to_number helper ──────────────────────────────────────────────

    #[test]
    fn test_string_to_number_empty() {
        assert_eq!(string_to_number(""), 0.0);
        assert_eq!(string_to_number("   "), 0.0);
    }

    #[test]
    fn test_string_to_number_decimal() {
        assert_eq!(string_to_number("42"), 42.0);
        assert_eq!(string_to_number("  3.14  "), 3.14);
        assert_eq!(string_to_number("-7"), -7.0);
    }

    #[test]
    fn test_string_to_number_infinity() {
        assert_eq!(string_to_number("Infinity"), f64::INFINITY);
        assert_eq!(string_to_number("+Infinity"), f64::INFINITY);
        assert_eq!(string_to_number("-Infinity"), f64::NEG_INFINITY);
        assert!(string_to_number("infinity").is_nan());
        assert!(string_to_number("inf").is_nan());
    }

    #[test]
    fn test_string_to_number_hex_v2() {
        assert_eq!(string_to_number("0xff"), 255.0);
        assert_eq!(string_to_number("0xFF"), 255.0);
    }

    #[test]
    fn test_string_to_number_octal_v2() {
        assert_eq!(string_to_number("0o77"), 63.0);
    }

    #[test]
    fn test_string_to_number_binary_v2() {
        assert_eq!(string_to_number("0b1010"), 10.0);
    }

    // ── number_to_string helper ──────────────────────────────────────────────

    #[test]
    fn test_number_to_string_nan() {
        assert_eq!(number_to_string(f64::NAN), "NaN");
    }

    #[test]
    fn test_number_to_string_positive_infinity() {
        assert_eq!(number_to_string(f64::INFINITY), "Infinity");
    }

    #[test]
    fn test_number_to_string_negative_infinity() {
        assert_eq!(number_to_string(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn test_number_to_string_negative_zero() {
        assert_eq!(number_to_string(-0.0), "0");
    }

    #[test]
    fn test_number_to_string_positive_zero() {
        assert_eq!(number_to_string(0.0), "0");
    }

    #[test]
    fn test_number_to_string_nan_v2() {
        assert_eq!(number_to_string(f64::NAN), "NaN");
    }

    #[test]
    fn test_f64_to_uint32_negative_wrap() {
        assert_eq!(f64_to_uint32(-1.0), 4_294_967_295);
    }

    #[test]
    fn test_f64_to_int16_wrap() {
        assert_eq!(f64_to_int16(32768.0), -32768);
        assert_eq!(f64_to_int16(-1.0), -1);
    }

    // ── number_to_string exponential notation ────────────────────────────────

    #[test]
    fn test_number_to_string_large_exponential() {
        // Values >= 1e21 use exponential notation with e+.
        assert_eq!(number_to_string(1e21), "1e+21");
        assert_eq!(number_to_string(1e25), "1e+25");
        assert_eq!(number_to_string(1.5e21), "1.5e+21");
    }

    #[test]
    fn test_number_to_string_small_exponential() {
        // Values < 1e-6 use exponential notation.
        assert_eq!(number_to_string(5e-7), "5e-7");
        assert_eq!(number_to_string(1e-7), "1e-7");
        assert_eq!(number_to_string(1.5e-8), "1.5e-8");
    }

    #[test]
    fn test_number_to_string_boundary_no_exponential() {
        // Exactly 1e-6 stays decimal.
        assert_eq!(number_to_string(1e-6), "0.000001");
        // Values just below 1e21 stay decimal.
        assert_eq!(number_to_string(1e20), "100000000000000000000");
    }

    #[test]
    fn test_number_to_string_negative_exponential() {
        assert_eq!(number_to_string(-1e21), "-1e+21");
        assert_eq!(number_to_string(-5e-8), "-5e-8");
    }

    #[test]
    fn test_number_to_string_ordinary_values() {
        assert_eq!(number_to_string(42.0), "42");
        assert_eq!(number_to_string(3.14), "3.14");
        assert_eq!(number_to_string(0.5), "0.5");
        assert_eq!(number_to_string(1.0), "1");
        assert_eq!(number_to_string(-42.0), "-42");
        assert_eq!(number_to_string(-3.14), "-3.14");
    }

    // ── string_to_number overflow handling ───────────────────────────────────

    #[test]
    fn test_string_to_number_hex_overflow_u64() {
        // 0x10000000000000000 = 2^64, which exceeds u64::MAX.
        let n = string_to_number("0x10000000000000000");
        assert!(n.is_finite());
        // u64::MAX rounds up to 2^64 in f64, so check >=.
        assert!(n >= u64::MAX as f64);
    }

    #[test]
    fn test_string_to_number_binary_overflow_u64() {
        // 65 binary ones exceed u64.
        let s = format!("0b{}", "1".repeat(65));
        let n = string_to_number(&s);
        assert!(n.is_finite());
        // u64::MAX rounds up to 2^64 in f64, so check >=.
        assert!(n >= u64::MAX as f64);
    }

    #[test]
    fn test_string_to_number_hex_empty_after_prefix() {
        assert!(string_to_number("0x").is_nan());
        assert!(string_to_number("0o").is_nan());
        assert!(string_to_number("0b").is_nan());
    }

    // ── parse_radix_digits helper ────────────────────────────────────────────

    #[test]
    fn test_parse_radix_digits_empty() {
        assert!(parse_radix_digits("", 16).is_nan());
    }

    #[test]
    fn test_parse_radix_digits_hex() {
        assert_eq!(parse_radix_digits("FF", 16), 255.0);
        assert_eq!(parse_radix_digits("0", 16), 0.0);
        assert_eq!(parse_radix_digits("a", 16), 10.0);
    }

    #[test]
    fn test_parse_radix_digits_invalid() {
        assert!(parse_radix_digits("ZZ", 16).is_nan());
        assert!(parse_radix_digits("9", 8).is_nan());
        assert!(parse_radix_digits("2", 2).is_nan());
    }

    // ── to_integer_or_infinity ───────────────────────────────────────────────

    #[test]
    fn test_to_integer_or_infinity_nan_is_zero() {
        assert_eq!(JsValue::Undefined.to_integer_or_infinity().unwrap(), 0.0);
        assert_eq!(
            JsValue::HeapNumber(f64::NAN)
                .to_integer_or_infinity()
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_to_integer_or_infinity_infinity_passthrough() {
        assert_eq!(
            JsValue::HeapNumber(f64::INFINITY)
                .to_integer_or_infinity()
                .unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            JsValue::HeapNumber(f64::NEG_INFINITY)
                .to_integer_or_infinity()
                .unwrap(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_to_integer_or_infinity_truncation() {
        assert_eq!(
            JsValue::HeapNumber(3.9).to_integer_or_infinity().unwrap(),
            3.0
        );
        assert_eq!(
            JsValue::HeapNumber(-3.9).to_integer_or_infinity().unwrap(),
            -3.0
        );
        assert_eq!(
            JsValue::HeapNumber(0.5).to_integer_or_infinity().unwrap(),
            0.0
        );
    }

    #[test]
    fn test_to_integer_or_infinity_zero() {
        assert_eq!(JsValue::Smi(0).to_integer_or_infinity().unwrap(), 0.0);
        assert_eq!(JsValue::Null.to_integer_or_infinity().unwrap(), 0.0);
    }

    #[test]
    fn test_to_integer_or_infinity_whole_numbers() {
        assert_eq!(JsValue::Smi(42).to_integer_or_infinity().unwrap(), 42.0);
        assert_eq!(JsValue::Smi(-7).to_integer_or_infinity().unwrap(), -7.0);
    }

    #[test]
    fn test_to_integer_or_infinity_string_coercion() {
        assert_eq!(
            JsValue::String("3.7".into())
                .to_integer_or_infinity()
                .unwrap(),
            3.0
        );
        assert_eq!(
            JsValue::String("".into()).to_integer_or_infinity().unwrap(),
            0.0
        );
    }

    #[test]
    fn test_to_integer_or_infinity_bigint_error() {
        assert!(JsValue::BigInt(42).to_integer_or_infinity().is_err());
    }

    // ── to_length ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_length_positive() {
        assert_eq!(JsValue::Smi(5).to_length().unwrap(), 5);
        assert_eq!(JsValue::HeapNumber(3.9).to_length().unwrap(), 3);
    }

    #[test]
    fn test_to_length_negative_clamps_to_zero() {
        assert_eq!(JsValue::Smi(-1).to_length().unwrap(), 0);
        assert_eq!(JsValue::HeapNumber(-100.5).to_length().unwrap(), 0);
    }

    #[test]
    fn test_to_length_infinity_clamps_to_max() {
        let max_safe = 9_007_199_254_740_991_u64;
        assert_eq!(
            JsValue::HeapNumber(f64::INFINITY).to_length().unwrap(),
            max_safe
        );
    }

    #[test]
    fn test_to_length_nan_is_zero() {
        assert_eq!(JsValue::Undefined.to_length().unwrap(), 0);
    }

    #[test]
    fn test_to_length_zero() {
        assert_eq!(JsValue::Smi(0).to_length().unwrap(), 0);
        assert_eq!(JsValue::Null.to_length().unwrap(), 0);
    }

    // ── to_numeric ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_numeric_number() {
        let result = JsValue::Smi(42).to_numeric().unwrap();
        assert_eq!(result, JsValue::HeapNumber(42.0));
    }

    #[test]
    fn test_to_numeric_bigint_passthrough() {
        let result = JsValue::BigInt(42).to_numeric().unwrap();
        assert_eq!(result, JsValue::BigInt(42));
    }

    #[test]
    fn test_to_numeric_string_coercion() {
        let result = JsValue::String("3.14".into()).to_numeric().unwrap();
        assert_eq!(result, JsValue::HeapNumber(3.14));
    }

    #[test]
    fn test_to_numeric_boolean_coercion() {
        assert_eq!(
            JsValue::Boolean(true).to_numeric().unwrap(),
            JsValue::HeapNumber(1.0)
        );
        assert_eq!(
            JsValue::Boolean(false).to_numeric().unwrap(),
            JsValue::HeapNumber(0.0)
        );
    }

    #[test]
    fn test_to_numeric_undefined_is_nan() {
        if let JsValue::HeapNumber(n) = JsValue::Undefined.to_numeric().unwrap() {
            assert!(n.is_nan());
        } else {
            panic!("Expected HeapNumber");
        }
    }

    #[test]
    fn test_to_numeric_symbol_error() {
        assert!(JsValue::Symbol(0).to_numeric().is_err());
    }

    // ── abstract_relational_comparison / js_less_than ────────────────────────

    #[test]
    fn test_less_than_numbers() {
        assert!(JsValue::js_less_than(&JsValue::Smi(1), &JsValue::Smi(2)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::Smi(2), &JsValue::Smi(1)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::Smi(1), &JsValue::Smi(1)).unwrap());
        assert!(
            JsValue::js_less_than(&JsValue::HeapNumber(1.5), &JsValue::HeapNumber(2.5)).unwrap()
        );
    }

    #[test]
    fn test_less_than_nan_is_false() {
        // NaN < anything → false; anything < NaN → false
        assert!(!JsValue::js_less_than(&JsValue::HeapNumber(f64::NAN), &JsValue::Smi(1)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::Smi(1), &JsValue::HeapNumber(f64::NAN)).unwrap());
        assert!(
            !JsValue::js_less_than(
                &JsValue::HeapNumber(f64::NAN),
                &JsValue::HeapNumber(f64::NAN)
            )
            .unwrap()
        );
    }

    #[test]
    fn test_less_than_returns_undefined_for_nan() {
        // The raw comparison returns None for NaN.
        assert_eq!(
            JsValue::abstract_relational_comparison(
                &JsValue::HeapNumber(f64::NAN),
                &JsValue::Smi(1),
                true
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn test_less_than_strings_lexicographic() {
        // String comparison is lexicographic, not numeric.
        assert!(
            JsValue::js_less_than(&JsValue::String("a".into()), &JsValue::String("b".into()))
                .unwrap()
        );
        assert!(
            !JsValue::js_less_than(&JsValue::String("b".into()), &JsValue::String("a".into()))
                .unwrap()
        );
        // "10" < "9" is true lexicographically (digit '1' < digit '9').
        assert!(
            JsValue::js_less_than(&JsValue::String("10".into()), &JsValue::String("9".into()))
                .unwrap()
        );
    }

    #[test]
    fn test_less_than_string_vs_number_coercion() {
        // "5" < 10 → 5 < 10 → true (string converted to number)
        assert!(JsValue::js_less_than(&JsValue::String("5".into()), &JsValue::Smi(10)).unwrap());
        // 10 < "5" → 10 < 5 → false
        assert!(!JsValue::js_less_than(&JsValue::Smi(10), &JsValue::String("5".into())).unwrap());
    }

    #[test]
    fn test_less_than_boolean_coercion() {
        // true → 1, false → 0
        assert!(JsValue::js_less_than(&JsValue::Boolean(false), &JsValue::Boolean(true)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::Boolean(true), &JsValue::Boolean(false)).unwrap());
        // false (0) < 1 → true
        assert!(JsValue::js_less_than(&JsValue::Boolean(false), &JsValue::Smi(1)).unwrap());
    }

    #[test]
    fn test_less_than_null_undefined() {
        // null → 0, undefined → NaN
        assert!(JsValue::js_less_than(&JsValue::Null, &JsValue::Smi(1)).unwrap());
        // undefined → NaN, NaN < 1 → false
        assert!(!JsValue::js_less_than(&JsValue::Undefined, &JsValue::Smi(1)).unwrap());
    }

    #[test]
    fn test_less_than_bigint() {
        assert!(JsValue::js_less_than(&JsValue::BigInt(1), &JsValue::BigInt(2)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::BigInt(2), &JsValue::BigInt(1)).unwrap());
    }

    #[test]
    fn test_less_than_bigint_vs_number() {
        assert!(JsValue::js_less_than(&JsValue::BigInt(1), &JsValue::HeapNumber(1.5)).unwrap());
        assert!(!JsValue::js_less_than(&JsValue::HeapNumber(2.5), &JsValue::BigInt(2)).unwrap());
    }

    #[test]
    fn test_less_than_bigint_vs_string() {
        // BigInt(1) < "2" → 1 < 2 → true
        assert!(JsValue::js_less_than(&JsValue::BigInt(1), &JsValue::String("2".into())).unwrap());
        // "1" < BigInt(2) → 1 < 2 → true
        assert!(JsValue::js_less_than(&JsValue::String("1".into()), &JsValue::BigInt(2)).unwrap());
        // BigInt vs unparseable string → None → false
        assert!(
            !JsValue::js_less_than(&JsValue::BigInt(1), &JsValue::String("abc".into())).unwrap()
        );
    }

    #[test]
    fn test_less_than_infinity() {
        assert!(
            JsValue::js_less_than(&JsValue::HeapNumber(f64::NEG_INFINITY), &JsValue::Smi(0))
                .unwrap()
        );
        assert!(
            !JsValue::js_less_than(&JsValue::HeapNumber(f64::INFINITY), &JsValue::Smi(0)).unwrap()
        );
    }

    #[test]
    fn test_greater_than() {
        assert!(JsValue::js_greater_than(&JsValue::Smi(2), &JsValue::Smi(1)).unwrap());
        assert!(!JsValue::js_greater_than(&JsValue::Smi(1), &JsValue::Smi(2)).unwrap());
    }

    #[test]
    fn test_less_than_or_equal() {
        assert!(JsValue::js_less_than_or_equal(&JsValue::Smi(1), &JsValue::Smi(2)).unwrap());
        assert!(JsValue::js_less_than_or_equal(&JsValue::Smi(1), &JsValue::Smi(1)).unwrap());
        assert!(!JsValue::js_less_than_or_equal(&JsValue::Smi(2), &JsValue::Smi(1)).unwrap());
    }

    #[test]
    fn test_greater_than_or_equal() {
        assert!(JsValue::js_greater_than_or_equal(&JsValue::Smi(2), &JsValue::Smi(1)).unwrap());
        assert!(JsValue::js_greater_than_or_equal(&JsValue::Smi(1), &JsValue::Smi(1)).unwrap());
        assert!(!JsValue::js_greater_than_or_equal(&JsValue::Smi(1), &JsValue::Smi(2)).unwrap());
    }

    #[test]
    fn test_relational_nan_all_false() {
        let nan = JsValue::HeapNumber(f64::NAN);
        let one = JsValue::Smi(1);
        // NaN makes all relational operators return false.
        assert!(!JsValue::js_less_than(&nan, &one).unwrap());
        assert!(!JsValue::js_greater_than(&nan, &one).unwrap());
        assert!(!JsValue::js_less_than_or_equal(&nan, &one).unwrap());
        assert!(!JsValue::js_greater_than_or_equal(&nan, &one).unwrap());
    }

    #[test]
    fn test_less_than_object_to_primitive_string_path() {
        use crate::objects::property_map::PropertyMap;
        use std::cell::RefCell;
        use std::rc::Rc;

        // Create a PlainObject with valueOf returning "abc".
        let mut map_a = PropertyMap::new();
        let val_fn_a: Rc<dyn Fn(Vec<JsValue>) -> StatorResult<JsValue>> =
            Rc::new(|_| Ok(JsValue::String("abc".into())));
        map_a.insert("valueOf".to_string(), JsValue::NativeFunction(val_fn_a));
        let obj_a = JsValue::PlainObject(Rc::new(RefCell::new(map_a)));

        // Create a PlainObject with valueOf returning "xyz".
        let mut map_b = PropertyMap::new();
        let val_fn_b: Rc<dyn Fn(Vec<JsValue>) -> StatorResult<JsValue>> =
            Rc::new(|_| Ok(JsValue::String("xyz".into())));
        map_b.insert("valueOf".to_string(), JsValue::NativeFunction(val_fn_b));
        let obj_b = JsValue::PlainObject(Rc::new(RefCell::new(map_b)));

        // Both ToPrimitive to strings → lexicographic: "abc" < "xyz" → true.
        assert!(JsValue::js_less_than(&obj_a, &obj_b).unwrap());
        assert!(!JsValue::js_less_than(&obj_b, &obj_a).unwrap());
    }

    #[test]
    fn test_less_than_object_to_primitive_number_path() {
        use crate::objects::property_map::PropertyMap;
        use std::cell::RefCell;
        use std::rc::Rc;

        // PlainObject with valueOf returning 10.
        let mut map = PropertyMap::new();
        let val_fn: Rc<dyn Fn(Vec<JsValue>) -> StatorResult<JsValue>> =
            Rc::new(|_| Ok(JsValue::Smi(10)));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(val_fn));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));

        // obj (→ 10) < 20 → true
        assert!(JsValue::js_less_than(&obj, &JsValue::Smi(20)).unwrap());
        // 5 < obj (→ 10) → true
        assert!(JsValue::js_less_than(&JsValue::Smi(5), &obj).unwrap());
    }

    #[test]
    fn test_less_than_object_with_toprimitive() {
        use crate::objects::property_map::PropertyMap;
        use std::cell::RefCell;
        use std::rc::Rc;

        // PlainObject with @@toPrimitive returning 42.
        let mut map = PropertyMap::new();
        let tp_fn: Rc<dyn Fn(Vec<JsValue>) -> StatorResult<JsValue>> =
            Rc::new(|_| Ok(JsValue::Smi(42)));
        map.insert("@@toPrimitive".to_string(), JsValue::NativeFunction(tp_fn));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));

        assert!(JsValue::js_less_than(&obj, &JsValue::Smi(100)).unwrap());
        assert!(!JsValue::js_less_than(&obj, &JsValue::Smi(10)).unwrap());
    }

    // ── to_primitive: Array ─────────────────────────────────────────────────

    #[test]
    fn test_to_primitive_array_number_hint() {
        let items = vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)];
        let arr = JsValue::Array(Rc::new(RefCell::new(items)));
        let result = arr.to_primitive(ToPrimitiveHint::Number).unwrap();
        assert_eq!(result, JsValue::String("1,2,3".into()));
    }

    #[test]
    fn test_to_primitive_array_with_null_undefined() {
        let items = vec![
            JsValue::Smi(1),
            JsValue::Null,
            JsValue::Undefined,
            JsValue::Smi(4),
        ];
        let arr = JsValue::Array(Rc::new(RefCell::new(items)));
        let result = arr.to_primitive(ToPrimitiveHint::Default).unwrap();
        assert_eq!(result, JsValue::String("1,,,4".into()));
    }

    #[test]
    fn test_to_primitive_empty_array() {
        let items: Vec<JsValue> = vec![];
        let arr = JsValue::Array(Rc::new(RefCell::new(items)));
        let result = arr.to_primitive(ToPrimitiveHint::Number).unwrap();
        assert_eq!(result, JsValue::String("".into()));
    }

    // ── to_primitive: Error ─────────────────────────────────────────────────

    #[test]
    fn test_to_primitive_error() {
        let err = JsValue::Error(Rc::new(crate::builtins::error::JsError::new(
            crate::builtins::error::ErrorKind::TypeError,
            "test error".to_string(),
        )));
        let result = err.to_primitive(ToPrimitiveHint::String).unwrap();
        if let JsValue::String(s) = result {
            assert!(s.contains("TypeError"));
            assert!(s.contains("test error"));
        } else {
            panic!("Expected string from error ToPrimitive");
        }
    }

    // ── Symbol / BigInt TypeError conformance (ES §7.1) ─────────────────────

    #[test]
    fn test_symbol_to_number_throws() {
        let sym = JsValue::Symbol(42);
        assert!(sym.to_number().is_err());
        match sym.to_number() {
            Err(StatorError::TypeError(msg)) => {
                assert_eq!(msg, "Cannot convert a Symbol value to a number");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_symbol_to_js_string_throws() {
        let sym = JsValue::Symbol(7);
        assert!(sym.to_js_string().is_err());
        match sym.to_js_string() {
            Err(StatorError::TypeError(msg)) => {
                assert_eq!(msg, "Cannot convert a Symbol value to a string");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_symbol_to_display_string_does_not_throw() {
        // Explicit String(sym) path — should succeed, not throw.
        let sym = JsValue::Symbol(99);
        let s = sym.to_display_string();
        assert_eq!(s, "Symbol(99)");
    }

    #[test]
    fn test_symbol_to_int32_propagates_type_error() {
        // to_int32 delegates to to_number, which must throw for Symbol.
        assert!(matches!(
            JsValue::Symbol(1).to_int32(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_symbol_to_uint32_propagates_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_uint32(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_symbol_to_integer_or_infinity_propagates_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_integer_or_infinity(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_symbol_to_length_propagates_type_error() {
        assert!(matches!(
            JsValue::Symbol(1).to_length(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_bigint_to_number_throws() {
        let big = JsValue::BigInt(123);
        assert!(big.to_number().is_err());
        match big.to_number() {
            Err(StatorError::TypeError(msg)) => {
                assert_eq!(msg, "Cannot convert a BigInt value to a number");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_bigint_to_int32_propagates_type_error() {
        assert!(matches!(
            JsValue::BigInt(1).to_int32(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_bigint_to_uint32_propagates_type_error() {
        assert!(matches!(
            JsValue::BigInt(1).to_uint32(),
            Err(StatorError::TypeError(_))
        ));
    }

    #[test]
    fn test_symbol_to_numeric_throws_type_error() {
        // to_numeric → to_primitive (no-op for Symbol) → to_number → TypeError.
        match JsValue::Symbol(5).to_numeric() {
            Err(StatorError::TypeError(msg)) => {
                assert_eq!(msg, "Cannot convert a Symbol value to a number");
            }
            other => panic!("Expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_symbol_to_boolean_is_true() {
        // Symbols are truthy (ES §7.1.2).
        assert!(JsValue::Symbol(0).to_boolean());
    }

    #[test]
    fn test_symbol_is_primitive() {
        assert!(JsValue::Symbol(1).is_primitive());
    }

    #[test]
    fn test_symbol_to_property_key_succeeds() {
        // ToPropertyKey on Symbol should not throw.
        let key = JsValue::Symbol(42).to_property_key().unwrap();
        assert_eq!(key, crate::builtins::symbol::symbol_to_property_key(42));
    }

    #[test]
    fn test_bigint_to_js_string_succeeds() {
        // BigInt → ToString should produce the decimal representation.
        assert_eq!(JsValue::BigInt(42).to_js_string().unwrap(), "42");
        assert_eq!(JsValue::BigInt(-1).to_js_string().unwrap(), "-1");
        assert_eq!(JsValue::BigInt(0).to_js_string().unwrap(), "0");
    }

    #[test]
    fn test_bigint_to_numeric_passthrough() {
        // to_numeric on BigInt returns itself, not HeapNumber.
        assert_eq!(
            JsValue::BigInt(99).to_numeric().unwrap(),
            JsValue::BigInt(99)
        );
    }

    // ── to_number: PlainObject valueOf/toString chain ─────────────────────────

    #[test]
    fn test_to_number_plain_object_valueof_returns_number() {
        let mut map = PropertyMap::new();
        let f: NativeFn = Rc::new(|_| Ok(JsValue::Smi(7)));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(f));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.to_number().unwrap(), 7.0);
    }

    #[test]
    fn test_to_number_plain_object_valueof_non_primitive_falls_to_tostring() {
        let mut map = PropertyMap::new();
        // valueOf returns an array (non-primitive) → skipped
        let val_fn: NativeFn = Rc::new(|_| Ok(JsValue::new_array(vec![])));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(val_fn));
        // toString returns "42" → used instead
        let ts_fn: NativeFn = Rc::new(|_| Ok(JsValue::String("42".into())));
        map.insert("toString".to_string(), JsValue::NativeFunction(ts_fn));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.to_number().unwrap(), 42.0);
    }

    #[test]
    fn test_to_number_plain_object_both_non_primitive_is_type_error() {
        let mut map = PropertyMap::new();
        // Both methods return non-primitives → TypeError per OrdinaryToPrimitive.
        let val_fn: NativeFn = Rc::new(|_| Ok(JsValue::new_array(vec![])));
        map.insert("valueOf".to_string(), JsValue::NativeFunction(val_fn));
        let ts_fn: NativeFn = Rc::new(|_| Ok(JsValue::new_array(vec![])));
        map.insert("toString".to_string(), JsValue::NativeFunction(ts_fn));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert!(matches!(
            obj.to_number(),
            Err(StatorError::TypeError(message)) if message == "Cannot convert object to primitive value"
        ));
    }

    #[test]
    fn test_to_number_plain_object_tostring_only() {
        let mut map = PropertyMap::new();
        // No valueOf, toString returns "100"
        let ts_fn: NativeFn = Rc::new(|_| Ok(JsValue::String("100".into())));
        map.insert("toString".to_string(), JsValue::NativeFunction(ts_fn));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.to_number().unwrap(), 100.0);
    }

    // ── to_js_string: Array direct arm ───────────────────────────────────────

    #[test]
    fn test_to_js_string_empty_array() {
        assert_eq!(JsValue::new_array(vec![]).to_js_string().unwrap(), "");
    }

    #[test]
    fn test_to_js_string_array_single_element() {
        let arr = JsValue::new_array(vec![JsValue::Smi(42)]);
        assert_eq!(arr.to_js_string().unwrap(), "42");
    }

    #[test]
    fn test_to_js_string_array_nested() {
        // [[3,4]] → inner toString is "3,4" → outer join is "3,4"
        let inner = JsValue::new_array(vec![JsValue::Smi(3), JsValue::Smi(4)]);
        let outer = JsValue::new_array(vec![JsValue::Smi(1), JsValue::Smi(2), inner]);
        assert_eq!(outer.to_js_string().unwrap(), "1,2,3,4");
    }

    #[test]
    fn test_to_js_string_array_with_booleans() {
        let arr = JsValue::new_array(vec![JsValue::Boolean(true), JsValue::Boolean(false)]);
        assert_eq!(arr.to_js_string().unwrap(), "true,false");
    }

    #[test]
    fn test_to_js_string_array_with_mixed_types() {
        let arr = JsValue::new_array(vec![
            JsValue::Smi(1),
            JsValue::String("hello".into()),
            JsValue::Null,
            JsValue::Undefined,
            JsValue::Boolean(true),
        ]);
        assert_eq!(arr.to_js_string().unwrap(), "1,hello,,,true");
    }

    // ── to_boolean: comprehensive edge cases ─────────────────────────────────

    #[test]
    fn test_to_boolean_empty_string_is_false() {
        assert!(!JsValue::String("".into()).to_boolean());
    }

    #[test]
    fn test_to_boolean_whitespace_string_is_true() {
        // " " is non-empty → true
        assert!(JsValue::String(" ".into()).to_boolean());
    }

    #[test]
    fn test_to_boolean_string_zero_is_true() {
        // "0" is non-empty → true
        assert!(JsValue::String("0".into()).to_boolean());
    }

    #[test]
    fn test_to_boolean_smi_negative_is_true() {
        assert!(JsValue::Smi(-42).to_boolean());
    }

    #[test]
    fn test_to_boolean_heap_number_neg_infinity_is_true() {
        assert!(JsValue::HeapNumber(f64::NEG_INFINITY).to_boolean());
    }

    #[test]
    fn test_to_boolean_bigint_negative_is_true() {
        assert!(JsValue::BigInt(-1).to_boolean());
    }

    // ── to_int32 / to_uint32: cross-type coercion ────────────────────────────

    #[test]
    fn test_to_int32_from_string() {
        assert_eq!(JsValue::String("42".into()).to_int32().unwrap(), 42);
        assert_eq!(JsValue::String("".into()).to_int32().unwrap(), 0);
        assert_eq!(JsValue::String("3.9".into()).to_int32().unwrap(), 3);
    }

    #[test]
    fn test_to_int32_from_boolean() {
        assert_eq!(JsValue::Boolean(true).to_int32().unwrap(), 1);
        assert_eq!(JsValue::Boolean(false).to_int32().unwrap(), 0);
    }

    #[test]
    fn test_to_uint32_from_string() {
        assert_eq!(JsValue::String("42".into()).to_uint32().unwrap(), 42);
        assert_eq!(JsValue::String("".into()).to_uint32().unwrap(), 0);
    }

    #[test]
    fn test_to_uint32_from_boolean() {
        assert_eq!(JsValue::Boolean(true).to_uint32().unwrap(), 1);
        assert_eq!(JsValue::Boolean(false).to_uint32().unwrap(), 0);
    }

    #[test]
    fn test_to_int32_large_negative() {
        // -2^31 - 1 wraps to 2^31 - 1 = 2147483647
        assert_eq!(
            JsValue::HeapNumber(-2_147_483_649.0).to_int32().unwrap(),
            2_147_483_647
        );
    }

    #[test]
    fn test_to_uint32_large_wrap() {
        // 2^32 + 5 wraps to 5
        assert_eq!(JsValue::HeapNumber(4_294_967_301.0).to_uint32().unwrap(), 5);
    }

    // ── same_value: cross-type numeric edge cases ────────────────────────────

    #[test]
    fn test_same_value_smi_vs_heap_number_equal() {
        assert!(JsValue::Smi(42).same_value(&JsValue::HeapNumber(42.0)));
        assert!(JsValue::HeapNumber(42.0).same_value(&JsValue::Smi(42)));
    }

    #[test]
    fn test_same_value_smi_vs_heap_number_not_equal() {
        assert!(!JsValue::Smi(1).same_value(&JsValue::HeapNumber(2.0)));
        assert!(!JsValue::HeapNumber(2.0).same_value(&JsValue::Smi(1)));
    }

    #[test]
    fn test_same_value_negative_zero_vs_negative_zero() {
        assert!(JsValue::HeapNumber(-0.0).same_value(&JsValue::HeapNumber(-0.0)));
    }

    #[test]
    fn test_same_value_strings() {
        assert!(JsValue::String("abc".into()).same_value(&JsValue::String("abc".into())));
        assert!(!JsValue::String("abc".into()).same_value(&JsValue::String("xyz".into())));
    }

    #[test]
    fn test_same_value_different_types() {
        assert!(!JsValue::Smi(0).same_value(&JsValue::Boolean(false)));
        assert!(!JsValue::Smi(0).same_value(&JsValue::String("0".into())));
        assert!(!JsValue::Null.same_value(&JsValue::Smi(0)));
    }

    // ── same_value_zero: cross-type numeric edge cases ───────────────────────

    #[test]
    fn test_same_value_zero_smi_vs_negative_zero() {
        // SameValueZero: +0 === -0
        assert!(JsValue::Smi(0).same_value_zero(&JsValue::HeapNumber(-0.0)));
        assert!(JsValue::HeapNumber(-0.0).same_value_zero(&JsValue::Smi(0)));
    }

    #[test]
    fn test_same_value_zero_negative_zero_vs_negative_zero() {
        assert!(JsValue::HeapNumber(-0.0).same_value_zero(&JsValue::HeapNumber(-0.0)));
    }

    #[test]
    fn test_same_value_zero_different_types() {
        assert!(!JsValue::Smi(0).same_value_zero(&JsValue::Boolean(false)));
        assert!(!JsValue::Smi(0).same_value_zero(&JsValue::String("0".into())));
    }

    // ── conformance round-21: additional coverage ────────────────────────────

    #[test]
    fn test_to_number_empty_string() {
        assert_eq!(JsValue::String("".into()).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_whitespace() {
        assert_eq!(JsValue::String("  ".into()).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_hex_string() {
        assert_eq!(JsValue::String("0xff".into()).to_number().unwrap(), 255.0);
    }

    #[test]
    fn test_to_string_neg_zero() {
        assert_eq!(
            JsValue::HeapNumber(-0.0).to_js_string().unwrap(),
            "0".to_string()
        );
    }

    #[test]
    fn test_to_string_infinity() {
        assert_eq!(
            JsValue::HeapNumber(f64::INFINITY).to_js_string().unwrap(),
            "Infinity".to_string()
        );
    }

    #[test]
    fn test_to_string_nan() {
        assert_eq!(
            JsValue::HeapNumber(f64::NAN).to_js_string().unwrap(),
            "NaN".to_string()
        );
    }

    #[test]
    fn test_to_number_true() {
        assert_eq!(JsValue::Boolean(true).to_number().unwrap(), 1.0);
    }

    #[test]
    fn test_to_number_false() {
        assert_eq!(JsValue::Boolean(false).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_obj_to_string_tag_plain_object_date() {
        let mut map = PropertyMap::new();
        map.insert("__is_date__".to_string(), JsValue::Boolean(true));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object Date]");
    }

    #[test]
    fn test_obj_to_string_tag_plain_object_callable() {
        let mut map = PropertyMap::new();
        let f: NativeFn = Rc::new(|_| Ok(JsValue::Undefined));
        map.insert("__call__".to_string(), JsValue::NativeFunction(f));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object Function]");
    }

    #[test]
    fn test_obj_to_string_tag_plain_object_error() {
        let mut map = PropertyMap::new();
        map.insert("__is_error__".to_string(), JsValue::Boolean(true));
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
        assert_eq!(obj.obj_to_string_tag(), "[object Error]");
    }

    #[test]
    fn test_compare_utf16_ascii() {
        assert!(
            JsValue::js_less_than(&JsValue::String("a".into()), &JsValue::String("b".into()))
                .unwrap()
        );
        assert!(
            !JsValue::js_less_than(&JsValue::String("b".into()), &JsValue::String("a".into()))
                .unwrap()
        );
        assert!(
            !JsValue::js_less_than(&JsValue::String("a".into()), &JsValue::String("a".into()))
                .unwrap()
        );
    }

    // ── number_to_string edge cases ──────────────────────────────────────

    #[test]
    fn test_number_to_string_negative_zero_v2() {
        assert_eq!(number_to_string(-0.0), "0");
    }

    #[test]
    fn test_number_to_string_positive_zero_v2() {
        assert_eq!(number_to_string(0.0), "0");
    }

    #[test]
    fn test_number_to_string_nan_v3() {
        assert_eq!(number_to_string(f64::NAN), "NaN");
    }

    #[test]
    fn test_number_to_string_infinity_v2() {
        assert_eq!(number_to_string(f64::INFINITY), "Infinity");
        assert_eq!(number_to_string(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn test_number_to_string_large_exponent() {
        assert_eq!(number_to_string(1e21), "1e+21");
    }

    #[test]
    fn test_number_to_string_small_exponent() {
        assert_eq!(number_to_string(5e-7), "5e-7");
    }

    // ── string_to_number edge cases ──────────────────────────────────────

    #[test]
    fn test_string_to_number_empty_is_zero() {
        assert_eq!(string_to_number(""), 0.0);
    }

    #[test]
    fn test_string_to_number_whitespace_is_zero() {
        assert_eq!(string_to_number("   "), 0.0);
    }

    #[test]
    fn test_string_to_number_hex() {
        assert_eq!(string_to_number("0xff"), 255.0);
        assert_eq!(string_to_number("0xFF"), 255.0);
    }

    #[test]
    fn test_string_to_number_octal() {
        assert_eq!(string_to_number("0o77"), 63.0);
    }

    #[test]
    fn test_string_to_number_binary() {
        assert_eq!(string_to_number("0b1010"), 10.0);
    }

    #[test]
    fn test_string_to_number_infinity_variants() {
        assert_eq!(string_to_number("Infinity"), f64::INFINITY);
        assert_eq!(string_to_number("+Infinity"), f64::INFINITY);
        assert_eq!(string_to_number("-Infinity"), f64::NEG_INFINITY);
    }

    #[test]
    fn test_string_to_number_leading_whitespace() {
        assert_eq!(string_to_number("  42  "), 42.0);
    }

    #[test]
    fn test_string_to_number_nan_cases() {
        assert!(string_to_number("hello").is_nan());
        assert!(string_to_number("0x").is_nan());
        assert!(string_to_number("0xG").is_nan());
    }

    // ── to_js_string (ToString) edge cases ───────────────────────────────

    #[test]
    fn test_to_js_string_null() {
        assert_eq!(JsValue::Null.to_js_string().unwrap(), "null");
    }

    #[test]
    fn test_to_js_string_undefined() {
        assert_eq!(JsValue::Undefined.to_js_string().unwrap(), "undefined");
    }

    #[test]
    fn test_to_js_string_negative_zero() {
        assert_eq!(JsValue::HeapNumber(-0.0).to_js_string().unwrap(), "0");
    }

    #[test]
    fn test_to_js_string_symbol_throws() {
        assert!(JsValue::Symbol(42).to_js_string().is_err());
    }

    #[test]
    fn test_to_js_string_bigint() {
        assert_eq!(JsValue::BigInt(123).to_js_string().unwrap(), "123");
    }

    // ── to_number (ToNumber) edge cases ──────────────────────────────────

    #[test]
    fn test_to_number_null_is_zero() {
        assert_eq!(JsValue::Null.to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_undefined_is_nan() {
        assert!(JsValue::Undefined.to_number().unwrap().is_nan());
    }

    #[test]
    fn test_to_number_true_is_one() {
        assert_eq!(JsValue::Boolean(true).to_number().unwrap(), 1.0);
    }

    #[test]
    fn test_to_number_false_is_zero() {
        assert_eq!(JsValue::Boolean(false).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_empty_string_is_zero() {
        assert_eq!(JsValue::String("".into()).to_number().unwrap(), 0.0);
    }

    #[test]
    fn test_to_number_symbol_throws() {
        assert!(JsValue::Symbol(1).to_number().is_err());
    }

    #[test]
    fn test_to_number_bigint_throws() {
        assert!(JsValue::BigInt(1).to_number().is_err());
    }

    // ── is_loosely_equal (==) edge cases ─────────────────────────────────

    #[test]
    fn test_loose_eq_null_undefined() {
        assert!(JsValue::Null.is_loosely_equal(&JsValue::Undefined).unwrap());
        assert!(JsValue::Undefined.is_loosely_equal(&JsValue::Null).unwrap());
    }

    #[test]
    fn test_loose_eq_null_not_zero() {
        assert!(!JsValue::Null.is_loosely_equal(&JsValue::Smi(0)).unwrap());
    }

    #[test]
    fn test_loose_eq_null_not_empty_string() {
        assert!(
            !JsValue::Null
                .is_loosely_equal(&JsValue::String("".into()))
                .unwrap()
        );
    }

    #[test]
    fn test_loose_eq_null_not_false() {
        assert!(
            !JsValue::Null
                .is_loosely_equal(&JsValue::Boolean(false))
                .unwrap()
        );
    }

    #[test]
    fn test_loose_eq_number_string_coercion() {
        // 1 == "1" → true (string coerced to number)
        assert!(
            JsValue::Smi(1)
                .is_loosely_equal(&JsValue::String("1".into()))
                .unwrap()
        );
        // 0 == "" → true (empty string coerced to 0)
        assert!(
            JsValue::Smi(0)
                .is_loosely_equal(&JsValue::String("".into()))
                .unwrap()
        );
    }

    #[test]
    fn test_loose_eq_boolean_coercion() {
        // true == 1 → true (boolean coerced to 1)
        assert!(
            JsValue::Boolean(true)
                .is_loosely_equal(&JsValue::Smi(1))
                .unwrap()
        );
        // false == 0 → true (boolean coerced to 0)
        assert!(
            JsValue::Boolean(false)
                .is_loosely_equal(&JsValue::Smi(0))
                .unwrap()
        );
        // true == "1" → true (boolean → 1, then 1 == "1")
        assert!(
            JsValue::Boolean(true)
                .is_loosely_equal(&JsValue::String("1".into()))
                .unwrap()
        );
    }

    #[test]
    fn test_loose_eq_symbol_not_equal() {
        assert!(
            !JsValue::Symbol(1)
                .is_loosely_equal(&JsValue::Symbol(2))
                .unwrap()
        );
        assert!(
            JsValue::Symbol(1)
                .is_loosely_equal(&JsValue::Symbol(1))
                .unwrap()
        );
    }

    // ── is_strictly_equal (===) edge cases ───────────────────────────────

    #[test]
    fn test_strict_eq_positive_negative_zero() {
        // +0 === -0 should be true per IEEE 754 / ES spec
        assert!(JsValue::Smi(0).is_strictly_equal(&JsValue::HeapNumber(-0.0)));
        assert!(JsValue::HeapNumber(-0.0).is_strictly_equal(&JsValue::Smi(0)));
        assert!(JsValue::HeapNumber(0.0).is_strictly_equal(&JsValue::HeapNumber(-0.0)));
    }

    #[test]
    fn test_strict_eq_nan_not_equal() {
        // NaN === NaN should be false
        assert!(!JsValue::HeapNumber(f64::NAN).is_strictly_equal(&JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_strict_eq_null_not_undefined() {
        assert!(!JsValue::Null.is_strictly_equal(&JsValue::Undefined));
    }

    // ── abstract_relational_comparison edge cases ────────────────────────

    #[test]
    fn test_relational_nan_always_undefined() {
        let nan = JsValue::HeapNumber(f64::NAN);
        let one = JsValue::Smi(1);
        // NaN < 1 → undefined (None)
        assert_eq!(
            JsValue::abstract_relational_comparison(&nan, &one, true).unwrap(),
            None
        );
        // 1 < NaN → undefined (None)
        assert_eq!(
            JsValue::abstract_relational_comparison(&one, &nan, true).unwrap(),
            None
        );
        // NaN < NaN → undefined (None)
        assert_eq!(
            JsValue::abstract_relational_comparison(&nan, &nan, true).unwrap(),
            None
        );
    }

    #[test]
    fn test_less_than_or_equal_nan() {
        let nan = JsValue::HeapNumber(f64::NAN);
        let one = JsValue::Smi(1);
        // NaN <= 1 → false
        assert!(!JsValue::js_less_than_or_equal(&nan, &one).unwrap());
        // 1 <= NaN → false
        assert!(!JsValue::js_less_than_or_equal(&one, &nan).unwrap());
    }

    #[test]
    fn test_greater_than_or_equal_nan() {
        let nan = JsValue::HeapNumber(f64::NAN);
        let one = JsValue::Smi(1);
        // NaN >= 1 → false
        assert!(!JsValue::js_greater_than_or_equal(&nan, &one).unwrap());
        // 1 >= NaN → false
        assert!(!JsValue::js_greater_than_or_equal(&one, &nan).unwrap());
    }

    #[test]
    fn test_greater_than_nan() {
        let nan = JsValue::HeapNumber(f64::NAN);
        let one = JsValue::Smi(1);
        // NaN > 1 → false
        assert!(!JsValue::js_greater_than(&nan, &one).unwrap());
        // 1 > NaN → false
        assert!(!JsValue::js_greater_than(&one, &nan).unwrap());
    }

    // ── SameValue edge cases ─────────────────────────────────────────────

    #[test]
    fn test_same_value_nan_equals_nan_v2() {
        assert!(JsValue::HeapNumber(f64::NAN).same_value(&JsValue::HeapNumber(f64::NAN)));
    }

    #[test]
    fn test_same_value_zero_signs_differ() {
        // SameValue(+0, -0) → false
        assert!(!JsValue::HeapNumber(0.0).same_value(&JsValue::HeapNumber(-0.0)));
        assert!(!JsValue::Smi(0).same_value(&JsValue::HeapNumber(-0.0)));
    }

    // ── to_display_string (safe symbol) ──────────────────────────────────

    #[test]
    fn test_to_display_string_symbol() {
        let sym = JsValue::Symbol(42);
        assert_eq!(sym.to_display_string(), "Symbol(42)");
    }

    #[test]
    fn test_to_display_string_number_neg_zero() {
        // to_display_string delegates to to_js_string which uses number_to_string
        assert_eq!(JsValue::HeapNumber(-0.0).to_display_string(), "0");
    }
}

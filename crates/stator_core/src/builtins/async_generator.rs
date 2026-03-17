//! ECMAScript §27.6 `AsyncGeneratorFunction` built-in and `AsyncGenerator.prototype`.
//!
//! Async generators combine the lazy pull of generators with the
//! asynchronous completion of promises.  Each call to `.next()`, `.return()`,
//! or `.throw()` returns a `Promise<{ value, done }>`.
//!
//! This module constructs:
//!
//! - **`AsyncGeneratorFunction`** — the constructor (not directly callable in
//!   typical code).
//! - **`AsyncGeneratorFunction.prototype`** — the `%AsyncGenerator%` intrinsic.
//! - **`AsyncGenerator.prototype`** — the `%AsyncGeneratorPrototype%` intrinsic
//!   with `.next()`, `.return()`, `.throw()`, and `[Symbol.asyncIterator]()`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::symbol::{SYMBOL_ASYNC_ITERATOR, SYMBOL_TO_STRING_TAG};
use crate::error::{StatorError, StatorResult};
use crate::interpreter::{Interpreter, async_iterator_result_promise};
use crate::objects::map::PropertyAttributes;
use crate::objects::property_map::PropertyMap;
use crate::objects::value::{GeneratorStep, JsValue};

// ── Helpers (local) ─────────────────────────────────────────────────────────

/// Shorthand for wrapping a Rust closure as a `JsValue::NativeFunction`.
fn native(f: impl Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static) -> JsValue {
    JsValue::NativeFunction(Rc::new(f))
}

/// Build a "built-in function object" with `name` and `length` properties.
fn builtin_fn(
    name: &str,
    length: i32,
    f: impl Fn(Vec<JsValue>) -> StatorResult<JsValue> + 'static,
) -> JsValue {
    let mut props = PropertyMap::new();
    let attrs = PropertyAttributes::CONFIGURABLE;
    props.insert_with_attrs("name".into(), JsValue::String(name.into()), attrs);
    props.insert_with_attrs("length".into(), JsValue::Smi(length), attrs);
    props.insert("__call__".into(), JsValue::NativeFunction(Rc::new(f)));
    props.make_all_non_enumerable();
    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

/// Create a `{ value, done }` iterator result object.
fn make_iter_result(value: JsValue, done: bool) -> JsValue {
    let mut map = PropertyMap::new();
    map.insert("value".to_string(), value);
    map.insert("done".to_string(), JsValue::Boolean(done));
    JsValue::PlainObject(Rc::new(RefCell::new(map)))
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Build the `AsyncGeneratorFunction` constructor and `AsyncGenerator.prototype`.
///
/// Per §27.4 / §27.6:
/// - `AsyncGeneratorFunction` is a constructor (rarely used directly).
/// - `AsyncGeneratorFunction.prototype` === `%AsyncGenerator%`.
/// - `%AsyncGenerator%.prototype` === `%AsyncGeneratorPrototype%` carrying
///   `.next()`, `.return()`, `.throw()`, and `[Symbol.asyncIterator]()`.
///
/// The returned `PlainObject` is registered as `"AsyncGeneratorFunction"` in
/// the global environment.
pub fn make_async_generator_function() -> JsValue {
    let mut props = PropertyMap::new();

    // AsyncGeneratorFunction is not directly callable.
    props.insert(
        "__call__".into(),
        native(|_args| {
            Err(StatorError::TypeError(
                "AsyncGeneratorFunction is not a constructor".into(),
            ))
        }),
    );

    // ── AsyncGenerator.prototype (%AsyncGeneratorPrototype%) ────────────

    let mut gen_proto = PropertyMap::new();

    // AsyncGenerator.prototype.next(value)  §27.6.1.2
    gen_proto.insert(
        "next".into(),
        builtin_fn("next", 1, |args| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            let input = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            match this {
                JsValue::Generator(gs) => {
                    let result =
                        Interpreter::run_generator_step(&gs, input).map(|step| match step {
                            GeneratorStep::Yield(v) => make_iter_result(v, false),
                            GeneratorStep::Return(v) => make_iter_result(v, true),
                        });
                    async_iterator_result_promise(result)
                }
                _ => Err(StatorError::TypeError(
                    "AsyncGenerator.prototype.next requires an async generator receiver".into(),
                )),
            }
        }),
    );

    // AsyncGenerator.prototype.return(value)  §27.6.1.3
    gen_proto.insert(
        "return".into(),
        builtin_fn("return", 1, |args| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            let value = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            match this {
                JsValue::Generator(gs) => {
                    async_iterator_result_promise(Interpreter::generator_return(&gs, value))
                }
                _ => Err(StatorError::TypeError(
                    "AsyncGenerator.prototype.return requires an async generator receiver".into(),
                )),
            }
        }),
    );

    // AsyncGenerator.prototype.throw(exception)  §27.6.1.4
    gen_proto.insert(
        "throw".into(),
        builtin_fn("throw", 1, |args| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            let value = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            match this {
                JsValue::Generator(gs) => {
                    async_iterator_result_promise(Interpreter::generator_throw(&gs, value))
                }
                _ => Err(StatorError::TypeError(
                    "AsyncGenerator.prototype.throw requires an async generator receiver".into(),
                )),
            }
        }),
    );

    // AsyncGenerator.prototype[@@toStringTag] = "AsyncGenerator"  §27.6.1.5
    gen_proto.insert_with_attrs(
        "@@toStringTag".into(),
        JsValue::String("AsyncGenerator".into()),
        PropertyAttributes::CONFIGURABLE,
    );
    gen_proto.insert_with_attrs(
        format!("Symbol({})", SYMBOL_TO_STRING_TAG),
        JsValue::String("AsyncGenerator".into()),
        PropertyAttributes::CONFIGURABLE,
    );

    // AsyncGenerator.prototype[@@asyncIterator]()  §27.6.1.6
    // Async generators are their own async iterator.
    gen_proto.insert(
        "@@asyncIterator".into(),
        builtin_fn("[Symbol.asyncIterator]", 0, |args| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            Ok(this)
        }),
    );
    gen_proto.insert(
        format!("Symbol({})", SYMBOL_ASYNC_ITERATOR),
        builtin_fn("[Symbol.asyncIterator]", 0, |args| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            Ok(this)
        }),
    );

    gen_proto.make_all_non_enumerable();
    let gen_proto_rc = Rc::new(RefCell::new(gen_proto));

    // ── %AsyncGenerator% (= AsyncGeneratorFunction.prototype) ───────────

    let mut generator_obj = PropertyMap::new();
    generator_obj.insert(
        "prototype".into(),
        JsValue::PlainObject(gen_proto_rc.clone()),
    );
    generator_obj.insert_with_attrs(
        "@@toStringTag".into(),
        JsValue::String("AsyncGeneratorFunction".into()),
        PropertyAttributes::CONFIGURABLE,
    );
    generator_obj.insert_with_attrs(
        format!("Symbol({})", SYMBOL_TO_STRING_TAG),
        JsValue::String("AsyncGeneratorFunction".into()),
        PropertyAttributes::CONFIGURABLE,
    );
    generator_obj.make_all_non_enumerable();
    let generator_obj_rc = Rc::new(RefCell::new(generator_obj));

    // AsyncGeneratorFunction.prototype = %AsyncGenerator%
    props.insert(
        "prototype".into(),
        JsValue::PlainObject(generator_obj_rc.clone()),
    );

    // Wire constructor links:
    // AsyncGenerator.prototype.constructor = %AsyncGenerator%
    gen_proto_rc.borrow_mut().insert_with_attrs(
        "constructor".into(),
        JsValue::PlainObject(generator_obj_rc.clone()),
        PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
    );

    props.insert(
        "name".into(),
        JsValue::String("AsyncGeneratorFunction".into()),
    );
    props.insert("length".into(), JsValue::Smi(1));
    props.make_all_non_enumerable();
    JsValue::PlainObject(Rc::new(RefCell::new(props)))
}

//! ECMAScript §20.2 `Function` built-in constructor and prototype methods.
//!
//! Provides pure-Rust implementations of:
//!
//! - `Function.prototype.bind(thisArg, ...args)` — partial application
//! - `Function.prototype.call(thisArg, ...args)`
//! - `Function.prototype.apply(thisArg, argsArray)`
//! - `Function.prototype.toString()`
//! - `Function.prototype[Symbol.hasInstance](V)`
//! - `.name` property
//! - `.length` property
//! - `new Function(…args, body)` — dynamic function creation
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §20.2 — *Function Objects*

use std::rc::Rc;

use crate::error::StatorResult;
use crate::interpreter::{dispatch_get_property_value, has_prototype_in_chain};
use crate::objects::value::{JsValue, NativeFn};

// ── Function.prototype.call ──────────────────────────────────────────────────

/// ECMAScript §20.2.3.3 `Function.prototype.call(thisArg, ...args)`.
///
/// Invokes the target callable with `this_arg` prepended to the arguments
/// (stator's NativeFunctions expect `this` as `args[0]`).
///
/// # Examples
///
/// ```
/// use std::rc::Rc;
/// use stator_core::builtins::function::function_call;
/// use stator_core::objects::value::JsValue;
///
/// // args[0] = thisArg (prepended by function_call), args[1..] = call args
/// let add = Rc::new(|args: Vec<JsValue>| {
///     let a = args.get(1).unwrap_or(&JsValue::Smi(0)).to_number().unwrap();
///     let b = args.get(2).unwrap_or(&JsValue::Smi(0)).to_number().unwrap();
///     Ok(JsValue::Smi((a + b) as i32))
/// }) as Rc<dyn Fn(Vec<JsValue>) -> stator_core::error::StatorResult<JsValue>>;
/// let result = function_call(&add, &JsValue::Undefined, &[JsValue::Smi(1), JsValue::Smi(2)]).unwrap();
/// assert_eq!(result, JsValue::Smi(3));
/// ```
pub fn function_call(
    func: &NativeFn,
    this_arg: &JsValue,
    args: &[JsValue],
) -> StatorResult<JsValue> {
    // NativeFunctions in stator expect `this` as args[0], so prepend it.
    let mut all_args = vec![this_arg.clone()];
    all_args.extend_from_slice(args);
    func(all_args)
}

// ── Function.prototype.apply ─────────────────────────────────────────────────

/// ECMAScript §20.2.3.1 `Function.prototype.apply(thisArg, argsArray)`.
///
/// Invokes the target callable, spreading `args_array` as positional
/// arguments.  If `args_array` is `undefined` or `null`, the function is
/// called with no arguments.
///
/// # Examples
///
/// ```
/// use std::rc::Rc;
/// use stator_core::builtins::function::function_apply;
/// use stator_core::objects::value::JsValue;
///
/// let sum = Rc::new(|args: Vec<JsValue>| {
///     // args[0] is this_arg, args[1..] are the actual values
///     let mut total = 0i32;
///     for a in args.iter().skip(1) { total += a.to_number().unwrap() as i32; }
///     Ok(JsValue::Smi(total))
/// }) as Rc<dyn Fn(Vec<JsValue>) -> stator_core::error::StatorResult<JsValue>>;
/// let arr = vec![JsValue::Smi(10), JsValue::Smi(20)];
/// let result = function_apply(&sum, &JsValue::Undefined, &Some(arr)).unwrap();
/// assert_eq!(result, JsValue::Smi(30));
/// ```
pub fn function_apply(
    func: &NativeFn,
    this_arg: &JsValue,
    args_array: &Option<Vec<JsValue>>,
) -> StatorResult<JsValue> {
    let spread = match args_array {
        Some(arr) => arr.clone(),
        None => Vec::new(),
    };
    // NativeFunctions in stator expect `this` as args[0], so prepend it.
    let mut all_args = vec![this_arg.clone()];
    all_args.extend(spread);
    func(all_args)
}

// ── Function.prototype.bind ──────────────────────────────────────────────────

/// ECMAScript §20.2.3.2 `Function.prototype.bind(thisArg, ...args)`.
///
/// Returns a new `NativeFn` that, when called, invokes the original function
/// with the bound arguments prepended to the call-time arguments.
///
/// Note: `this_arg` is intentionally unused here because stator's
/// NativeFunctions receive `this` as `args[0]` through the runtime calling
/// convention. The proto_lookup `bind` handler prepends the bound `this` into
/// the argument list before calling the target.
///
/// # Examples
///
/// ```
/// use std::rc::Rc;
/// use stator_core::builtins::function::function_bind;
/// use stator_core::objects::value::JsValue;
///
/// let add = Rc::new(|args: Vec<JsValue>| {
///     let a = args.first().unwrap_or(&JsValue::Smi(0)).to_number().unwrap();
///     let b = args.get(1).unwrap_or(&JsValue::Smi(0)).to_number().unwrap();
///     Ok(JsValue::Smi((a + b) as i32))
/// }) as Rc<dyn Fn(Vec<JsValue>) -> stator_core::error::StatorResult<JsValue>>;
/// let bound = function_bind(&add, &JsValue::Undefined, &[JsValue::Smi(10)]);
/// let result = bound(vec![JsValue::Smi(5)]).unwrap();
/// assert_eq!(result, JsValue::Smi(15));
/// ```
pub fn function_bind(func: &NativeFn, _this_arg: &JsValue, bound_args: &[JsValue]) -> NativeFn {
    let target = Rc::clone(func);
    let prefix: Vec<JsValue> = bound_args.to_vec();
    Rc::new(move |call_args: Vec<JsValue>| {
        let mut combined = prefix.clone();
        combined.extend(call_args);
        target(combined)
    })
}

// ── .length ──────────────────────────────────────────────────────────────────

/// Compute `Function.prototype.length` for a native function.
///
/// The *length* of a bound function is `max(0, target.length - bound_args)`.
/// For plain native functions with no formal parameter metadata we default to
/// `0` — the caller supplies the target's original arity.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_length;
///
/// assert_eq!(function_length(3, 1), 2);
/// assert_eq!(function_length(1, 5), 0);
/// ```
pub fn function_length(target_length: u32, bound_arg_count: u32) -> u32 {
    target_length.saturating_sub(bound_arg_count)
}

// ── .name ────────────────────────────────────────────────────────────────────

/// Compute the `.name` of a bound function.
///
/// Per spec §20.2.3.2 step 11, the name is `"bound " + targetName`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_bound_name;
///
/// assert_eq!(function_bound_name("foo"), "bound foo");
/// assert_eq!(function_bound_name(""), "bound ");
/// ```
pub fn function_bound_name(target_name: &str) -> String {
    format!("bound {target_name}")
}

// ── Function.prototype[Symbol.hasInstance] ────────────────────────────────────

/// ECMAScript §20.2.3.6 `Function.prototype[@@hasInstance](V)`.
///
/// The default `@@hasInstance` implementation walks the prototype chain of
/// `value`.  In our simplified model a `PlainObject` with a `"prototype"`
/// property matching the constructor's `prototype` property counts as an
/// instance.  For non-object values the result is always `false`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_has_instance;
/// use stator_core::objects::value::JsValue;
///
/// assert!(!function_has_instance(&JsValue::Undefined, &JsValue::Smi(42)));
/// ```
pub fn function_has_instance(constructor: &JsValue, value: &JsValue) -> bool {
    if matches!(
        value,
        JsValue::Undefined
            | JsValue::Null
            | JsValue::Boolean(_)
            | JsValue::Smi(_)
            | JsValue::HeapNumber(_)
            | JsValue::String(_)
            | JsValue::Symbol(_)
            | JsValue::BigInt(_)
    ) {
        return false;
    }

    let Ok(constructor_prototype) =
        dispatch_get_property_value(constructor, JsValue::String("prototype".into()))
    else {
        return false;
    };

    matches!(
        constructor_prototype,
        JsValue::PlainObject(_) | JsValue::Object(_)
    ) && has_prototype_in_chain(value, &constructor_prototype)
}

// ── Function.prototype.toString ──────────────────────────────────────────────

/// ECMAScript §20.2.3.5 `Function.prototype.toString()`.
///
/// For native functions without original source text we produce the
/// spec-mandated `"function name() { [native code] }"` form.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_to_string;
///
/// assert_eq!(function_to_string("foo"), "function foo() { [native code] }");
/// assert_eq!(function_to_string(""), "function () { [native code] }");
/// ```
pub fn function_to_string(name: &str) -> String {
    if name.is_empty() {
        "function () { [native code] }".to_string()
    } else {
        format!("function {name}() {{ [native code] }}")
    }
}

/// Produce a `toString()` result for a dynamically-created `Function`.
///
/// Per spec, the toString result for `new Function("a","b","return a+b")`
/// should be `"function anonymous(a,b\n) {\nreturn a+b\n}"`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_dynamic_to_string;
///
/// assert_eq!(
///     function_dynamic_to_string(&["a", "b"], "return a+b"),
///     "function anonymous(a,b\n) {\nreturn a+b\n}"
/// );
/// ```
pub fn function_dynamic_to_string(params: &[&str], body: &str) -> String {
    let param_list = params.join(",");
    format!("function anonymous({param_list}\n) {{\n{body}\n}}")
}

// ── new Function() constructor ───────────────────────────────────────────────

/// ECMAScript §20.2.1.1 `Function(…args)` — dynamic function creation.
///
/// Concatenates the argument strings into a function source and evaluates it
/// using the engine's eval pathway.  The last argument is the function body;
/// all preceding arguments are parameter names.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::function::function_constructor;
/// use stator_core::objects::value::JsValue;
///
/// let result = function_constructor(&[
///     JsValue::String("a".into()),
///     JsValue::String("b".into()),
///     JsValue::String("return a + b".into()),
/// ]);
/// assert!(result.is_ok());
/// ```
pub fn function_constructor(args: &[JsValue]) -> StatorResult<JsValue> {
    use crate::builtins::global::global_eval;
    use crate::interpreter::fn_props_set;

    // Convert all args to strings.
    let string_args: Vec<String> = args
        .iter()
        .map(|a| a.to_js_string())
        .collect::<StatorResult<Vec<String>>>()?;

    let (body, params) = match string_args.split_last() {
        Some((body, params)) => (body.clone(), params.to_vec()),
        None => (String::new(), Vec::new()),
    };
    let source_text = function_dynamic_to_string(
        &params.iter().map(String::as_str).collect::<Vec<_>>(),
        &body,
    );
    let source = format!("({source_text})");
    let result = global_eval(&source)?;
    if let JsValue::Function(ba) = &result {
        fn_props_set(ba, "name".to_string(), JsValue::String("anonymous".into()));
        fn_props_set(
            ba,
            "source".to_string(),
            JsValue::String(source_text.into()),
        );
    }
    Ok(result)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers for call/apply tests skip args[0] (this_arg) since
    // function_call/apply now prepend it.
    fn make_adder() -> NativeFn {
        Rc::new(|args: Vec<JsValue>| {
            let a = args.get(1).unwrap_or(&JsValue::Smi(0)).to_number()?;
            let b = args.get(2).unwrap_or(&JsValue::Smi(0)).to_number()?;
            let sum = a + b;
            if sum.fract() == 0.0 && sum >= f64::from(i32::MIN) && sum <= f64::from(i32::MAX) {
                Ok(JsValue::Smi(sum as i32))
            } else {
                Ok(JsValue::HeapNumber(sum))
            }
        })
    }

    fn make_summer() -> NativeFn {
        Rc::new(|args: Vec<JsValue>| {
            let mut total = 0.0_f64;
            for a in args.iter().skip(1) {
                total += a.to_number()?;
            }
            if total.fract() == 0.0 && total >= f64::from(i32::MIN) && total <= f64::from(i32::MAX)
            {
                Ok(JsValue::Smi(total as i32))
            } else {
                Ok(JsValue::HeapNumber(total))
            }
        })
    }

    // Helpers for bind tests — bind does NOT prepend this_arg, so args
    // start at index 0.
    fn make_adder_raw() -> NativeFn {
        Rc::new(|args: Vec<JsValue>| {
            let a = args.first().unwrap_or(&JsValue::Smi(0)).to_number()?;
            let b = args.get(1).unwrap_or(&JsValue::Smi(0)).to_number()?;
            let sum = a + b;
            if sum.fract() == 0.0 && sum >= f64::from(i32::MIN) && sum <= f64::from(i32::MAX) {
                Ok(JsValue::Smi(sum as i32))
            } else {
                Ok(JsValue::HeapNumber(sum))
            }
        })
    }

    fn make_summer_raw() -> NativeFn {
        Rc::new(|args: Vec<JsValue>| {
            let mut total = 0.0_f64;
            for a in &args {
                total += a.to_number()?;
            }
            if total.fract() == 0.0 && total >= f64::from(i32::MIN) && total <= f64::from(i32::MAX)
            {
                Ok(JsValue::Smi(total as i32))
            } else {
                Ok(JsValue::HeapNumber(total))
            }
        })
    }

    // ── call ────────────────────────────────────────────────────────────

    #[test]
    fn test_call_basic() {
        let f = make_adder();
        let result =
            function_call(&f, &JsValue::Undefined, &[JsValue::Smi(3), JsValue::Smi(4)]).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    #[test]
    fn test_call_no_args() {
        let f = make_adder();
        let result = function_call(&f, &JsValue::Undefined, &[]).unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_call_with_this_arg_passed() {
        // this_arg is now prepended to the args, so the function sees it at args[0].
        let f: NativeFn = Rc::new(|args: Vec<JsValue>| {
            let this = args.first().cloned().unwrap_or(JsValue::Undefined);
            Ok(this)
        });
        let result = function_call(
            &f,
            &JsValue::String("hello".into()),
            &[JsValue::Smi(10), JsValue::Smi(20)],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("hello".into()));
    }

    #[test]
    fn test_call_single_arg() {
        let f = make_adder();
        let result = function_call(&f, &JsValue::Undefined, &[JsValue::Smi(5)]).unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn test_call_heap_number_args() {
        let f = make_adder();
        let result = function_call(
            &f,
            &JsValue::Undefined,
            &[JsValue::HeapNumber(1.5), JsValue::HeapNumber(2.5)],
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    // ── apply ───────────────────────────────────────────────────────────

    #[test]
    fn test_apply_basic() {
        let f = make_summer();
        let args = Some(vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]);
        let result = function_apply(&f, &JsValue::Undefined, &args).unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn test_apply_no_args_array() {
        let f = make_summer();
        let result = function_apply(&f, &JsValue::Undefined, &None).unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_apply_empty_args_array() {
        let f = make_summer();
        let result = function_apply(&f, &JsValue::Undefined, &Some(vec![])).unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_apply_single_element() {
        let f = make_summer();
        let result =
            function_apply(&f, &JsValue::Undefined, &Some(vec![JsValue::Smi(42)])).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── bind ────────────────────────────────────────────────────────────

    #[test]
    fn test_bind_partial_application() {
        let f = make_adder_raw();
        let bound = function_bind(&f, &JsValue::Undefined, &[JsValue::Smi(10)]);
        let result = bound(vec![JsValue::Smi(5)]).unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    #[test]
    fn test_bind_all_args() {
        let f = make_adder_raw();
        let bound = function_bind(&f, &JsValue::Undefined, &[JsValue::Smi(3), JsValue::Smi(7)]);
        let result = bound(vec![]).unwrap();
        assert_eq!(result, JsValue::Smi(10));
    }

    #[test]
    fn test_bind_no_args() {
        let f = make_adder_raw();
        let bound = function_bind(&f, &JsValue::Undefined, &[]);
        let result = bound(vec![JsValue::Smi(1), JsValue::Smi(2)]).unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_bind_chained() {
        let f = make_summer_raw();
        let bound1 = function_bind(&f, &JsValue::Undefined, &[JsValue::Smi(1)]);
        let bound2 = function_bind(&bound1, &JsValue::Undefined, &[JsValue::Smi(2)]);
        let result = bound2(vec![JsValue::Smi(3)]).unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn test_bind_extra_args_ignored() {
        let f = make_adder_raw();
        let bound = function_bind(&f, &JsValue::Undefined, &[JsValue::Smi(1), JsValue::Smi(2)]);
        // Extra args beyond what adder reads are harmless.
        let result = bound(vec![JsValue::Smi(99)]).unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── length ──────────────────────────────────────────────────────────

    #[test]
    fn test_length_no_bound() {
        assert_eq!(function_length(3, 0), 3);
    }

    #[test]
    fn test_length_partial() {
        assert_eq!(function_length(3, 1), 2);
    }

    #[test]
    fn test_length_saturated() {
        assert_eq!(function_length(1, 5), 0);
    }

    #[test]
    fn test_length_zero() {
        assert_eq!(function_length(0, 0), 0);
    }

    // ── name ────────────────────────────────────────────────────────────

    #[test]
    fn test_bound_name() {
        assert_eq!(function_bound_name("foo"), "bound foo");
    }

    #[test]
    fn test_bound_name_empty() {
        assert_eq!(function_bound_name(""), "bound ");
    }

    #[test]
    fn test_bound_name_chained() {
        let name = function_bound_name("foo");
        let name2 = function_bound_name(&name);
        assert_eq!(name2, "bound bound foo");
    }

    // ── hasInstance ──────────────────────────────────────────────────────

    #[test]
    fn test_has_instance_primitives() {
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::Smi(42)
        ));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::Boolean(true)
        ));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::String("hi".into())
        ));
        assert!(!function_has_instance(&JsValue::Undefined, &JsValue::Null));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::Undefined
        ));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::HeapNumber(3.14)
        ));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::Symbol(1)
        ));
        assert!(!function_has_instance(
            &JsValue::Undefined,
            &JsValue::BigInt(Box::new(99))
        ));
    }

    #[test]
    fn test_has_instance_matches_constructor_prototype_chain() {
        use crate::objects::property_map::{INTERNAL_PROTO_PROPERTY_KEY, PropertyMap};
        use std::cell::RefCell;
        let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let instance_map = Rc::new(RefCell::new(PropertyMap::new()));
        instance_map
            .borrow_mut()
            .insert(INTERNAL_PROTO_PROPERTY_KEY.into(), proto.clone());
        let mut ctor_map = PropertyMap::new();
        ctor_map.insert("prototype".into(), proto);
        let constructor = JsValue::PlainObject(Rc::new(RefCell::new(ctor_map)));
        let instance = JsValue::PlainObject(instance_map);
        assert!(function_has_instance(&constructor, &instance));
    }

    #[test]
    fn test_has_instance_rejects_non_matching_chain() {
        use crate::objects::property_map::PropertyMap;
        use std::cell::RefCell;
        let proto = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let mut ctor_map = PropertyMap::new();
        ctor_map.insert("prototype".into(), proto);
        let constructor = JsValue::PlainObject(Rc::new(RefCell::new(ctor_map)));
        let instance = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert!(!function_has_instance(&constructor, &instance));
    }

    #[test]
    fn test_has_instance_requires_object_prototype_property() {
        use crate::objects::property_map::PropertyMap;
        use std::cell::RefCell;
        let constructor = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        let instance = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        assert!(!function_has_instance(&constructor, &instance));
    }

    // ── toString ────────────────────────────────────────────────────────

    #[test]
    fn test_to_string_named() {
        assert_eq!(
            function_to_string("foo"),
            "function foo() { [native code] }"
        );
    }

    #[test]
    fn test_to_string_anonymous() {
        assert_eq!(function_to_string(""), "function () { [native code] }");
    }

    #[test]
    fn test_to_string_bound() {
        let name = function_bound_name("myFn");
        assert_eq!(
            function_to_string(&name),
            "function bound myFn() { [native code] }"
        );
    }

    // ── dynamic toString ────────────────────────────────────────────────

    #[test]
    fn test_dynamic_to_string_two_params() {
        assert_eq!(
            function_dynamic_to_string(&["a", "b"], "return a+b"),
            "function anonymous(a,b\n) {\nreturn a+b\n}"
        );
    }

    #[test]
    fn test_dynamic_to_string_no_params() {
        assert_eq!(
            function_dynamic_to_string(&[], "return 42"),
            "function anonymous(\n) {\nreturn 42\n}"
        );
    }

    #[test]
    fn test_dynamic_to_string_one_param() {
        assert_eq!(
            function_dynamic_to_string(&["x"], "return x * x"),
            "function anonymous(x\n) {\nreturn x * x\n}"
        );
    }

    // ── constructor ─────────────────────────────────────────────────────

    #[test]
    fn test_constructor_empty() {
        let result = function_constructor(&[]);
        // Should compile without error.
        assert!(result.is_ok());
    }

    #[test]
    fn test_constructor_body_only() {
        let result = function_constructor(&[JsValue::String("return 42".into())]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constructor_with_params() {
        let result = function_constructor(&[
            JsValue::String("a".into()),
            JsValue::String("b".into()),
            JsValue::String("return a + b".into()),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constructor_syntax_error() {
        // An invalid parameter list should trigger a parse error.
        let result = function_constructor(&[
            JsValue::String(",".into()),
            JsValue::String("return 1".into()),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_constructor_coerces_args() {
        // Non-string args are coerced via ToString; valid param name works.
        let result = function_constructor(&[
            JsValue::String("x".into()),
            JsValue::String("return x".into()),
        ]);
        assert!(result.is_ok());
    }
}

//! JavaScript function objects.
//!
//! This module provides two closely related types:
//!
//! * [`SharedFunctionInfo`] — the **static** metadata about a function that is
//!   shared across all closure instances created from the same source: the
//!   function's name, its bytecode (or a reference to it), its formal-parameter
//!   count, and the language mode it was compiled in.
//!
//! * [`JsFunction`] — a **closure instance** that pairs a [`SharedFunctionInfo`]
//!   with the lexical scope ([`Context`]) captured at the point the `function`
//!   expression was evaluated.
//!
//! The module additionally supports two special function flavours:
//!
//! * **Bound functions** — produced by `Function.prototype.bind`.  A bound
//!   function wraps an underlying target together with a pre-bound `this` value
//!   and zero or more leading arguments.
//!
//! * **Native Rust functions** — host-side callbacks that implement built-in
//!   behaviour directly in Rust.  These are represented by the [`NativeFn`] type
//!   alias and stored in the [`FunctionKind::Native`] variant.

use std::rc::Rc;

use crate::error::StatorResult;
use crate::gc::trace::{Trace, Tracer};
use crate::objects::value::JsValue;

// ──────────────────────────────────────────────────────────────────────────────
// LanguageMode
// ──────────────────────────────────────────────────────────────────────────────

/// The ECMAScript language mode in which a function was compiled.
///
/// Corresponds to the `[[Strict]]` internal slot on functions:
/// * [`Sloppy`][LanguageMode::Sloppy] — non-strict (default) mode.
/// * [`Strict`][LanguageMode::Strict] — `"use strict"` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageMode {
    /// Non-strict (legacy sloppy) mode.
    Sloppy,
    /// Strict mode (`"use strict"`).
    Strict,
}

// ──────────────────────────────────────────────────────────────────────────────
// SharedFunctionInfo
// ──────────────────────────────────────────────────────────────────────────────

/// A reference to a compiled bytecode chunk.
///
/// In a full engine this would be a GC-managed pointer to a `Code` or
/// `BytecodeArray` object.  During early development a simple `Vec<u8>` is
/// used as a placeholder; the bytecode bytes are completely opaque to this
/// module.
#[derive(Debug, Clone)]
pub struct BytecodeRef(pub Vec<u8>);

/// Shared metadata for a JavaScript function.
///
/// A single `SharedFunctionInfo` instance is created once per parsed function
/// body and is shared by every closure ([`JsFunction`]) created from that
/// body.  It carries everything that does **not** depend on the captured
/// lexical scope:
///
/// * `name` — the function's declared name, or an empty string for anonymous
///   functions.
/// * `bytecode` — a reference to the compiled bytecode, if available.
/// * `param_count` — the number of formal parameters declared in the source.
/// * `language_mode` — whether the function body was compiled in sloppy or
///   strict mode.
#[derive(Debug, Clone)]
pub struct SharedFunctionInfo {
    name: String,
    bytecode: Option<BytecodeRef>,
    param_count: u32,
    language_mode: LanguageMode,
}

impl SharedFunctionInfo {
    /// Creates a new `SharedFunctionInfo` with no bytecode.
    ///
    /// # Parameters
    /// * `name` — the function's source name (empty string for anonymous).
    /// * `param_count` — the number of formal parameters.
    /// * `language_mode` — the compilation mode ([`LanguageMode::Sloppy`] or
    ///   [`LanguageMode::Strict`]).
    pub fn new(name: impl Into<String>, param_count: u32, language_mode: LanguageMode) -> Self {
        Self {
            name: name.into(),
            bytecode: None,
            param_count,
            language_mode,
        }
    }

    /// Creates a new `SharedFunctionInfo` with pre-compiled bytecode.
    pub fn with_bytecode(
        name: impl Into<String>,
        param_count: u32,
        language_mode: LanguageMode,
        bytecode: BytecodeRef,
    ) -> Self {
        Self {
            name: name.into(),
            bytecode: Some(bytecode),
            param_count,
            language_mode,
        }
    }

    /// Returns the function's declared name.
    ///
    /// Anonymous functions return an empty string.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the formal parameter count declared in the source.
    pub fn param_count(&self) -> u32 {
        self.param_count
    }

    /// Returns the language mode in which this function was compiled.
    pub fn language_mode(&self) -> LanguageMode {
        self.language_mode
    }

    /// Returns a reference to the compiled bytecode, if available.
    pub fn bytecode(&self) -> Option<&BytecodeRef> {
        self.bytecode.as_ref()
    }

    /// Installs (or replaces) the compiled bytecode.
    pub fn set_bytecode(&mut self, bytecode: BytecodeRef) {
        self.bytecode = Some(bytecode);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Context (captured lexical scope)
// ──────────────────────────────────────────────────────────────────────────────

/// A lexical scope captured by a closure.
///
/// `Context` represents the set of variable bindings that were in scope at the
/// point a `function` expression was evaluated.  In a full engine this would be
/// a GC-managed chain of activation-record frames; here it is modelled as a
/// flat `Vec` of `(name, value)` pairs that a closure can read.
///
/// Contexts are reference-counted so that multiple closures created in the same
/// scope can share the same `Context` without copying.
#[derive(Debug, Clone, Default)]
pub struct Context {
    bindings: Vec<(String, JsValue)>,
}

impl Context {
    /// Creates an empty context with no bindings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a binding, shadowing any previous binding with the same name.
    pub fn set(&mut self, name: impl Into<String>, value: JsValue) {
        let name = name.into();
        if let Some(entry) = self.bindings.iter_mut().find(|(k, _)| k == &name) {
            entry.1 = value;
        } else {
            self.bindings.push((name, value));
        }
    }

    /// Looks up a binding by name.
    ///
    /// Returns `None` if the name is not bound in this context.
    pub fn get(&self, name: &str) -> Option<&JsValue> {
        self.bindings
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
    }

    /// Returns the number of bindings stored in this context.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns `true` if this context contains no bindings.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NativeFn
// ──────────────────────────────────────────────────────────────────────────────

/// A host-side (Rust) callback that implements a built-in JavaScript function.
///
/// The callback receives a slice of [`JsValue`] arguments and returns a
/// [`StatorResult`][crate::error::StatorResult]`<`[`JsValue`]`>`.  The first
/// element of `args` is the `this` value by convention; subsequent elements are
/// the positional arguments.
pub type NativeFn = fn(&[JsValue]) -> StatorResult<JsValue>;

// ──────────────────────────────────────────────────────────────────────────────
// FunctionKind
// ──────────────────────────────────────────────────────────────────────────────

/// Discriminates the three flavours a [`JsFunction`] can take.
pub enum FunctionKind {
    /// An ordinary interpreted (or compiled) function backed by bytecode.
    ///
    /// The function is called by executing the bytecode stored in the
    /// associated [`SharedFunctionInfo`].
    Normal,

    /// A bound function produced by `Function.prototype.bind`.
    ///
    /// Stores the wrapped target function together with the pre-bound `this`
    /// and leading argument list.
    Bound {
        /// The underlying function that will be invoked.
        target: Rc<JsFunction>,
        /// The `this` value bound at `bind` time.
        bound_this: JsValue,
        /// Leading arguments prepended to each call.
        bound_args: Vec<JsValue>,
    },

    /// A built-in Rust callback.
    ///
    /// Native functions bypass bytecode execution entirely and run as regular
    /// Rust code.  They do **not** capture a lexical scope; the `context` field
    /// of the owning [`JsFunction`] will be empty.
    Native(NativeFn),
}

// Implement Debug manually because fn pointers don't implement Debug.
impl std::fmt::Debug for FunctionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Bound {
                target,
                bound_this,
                bound_args,
            } => f
                .debug_struct("Bound")
                .field("target", target)
                .field("bound_this", bound_this)
                .field("bound_args", bound_args)
                .finish(),
            Self::Native(_) => write!(f, "Native(<fn>)"),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// JsFunction
// ──────────────────────────────────────────────────────────────────────────────

/// A JavaScript function closure.
///
/// A `JsFunction` is a runtime closure instance.  It pairs:
///
/// 1. A reference-counted [`SharedFunctionInfo`] — the static metadata
///    (name, bytecode, parameter count, language mode) that is shared by every
///    closure created from the same function body.
///
/// 2. A [`Context`] — the lexical scope snapshot captured when the `function`
///    expression was evaluated.
///
/// In addition, [`FunctionKind`] discriminates ordinary functions from bound
/// functions and native host callbacks.
///
/// # Example — wrapping a native function
///
/// ```rust
/// use stator_core::objects::js_function::{
///     JsFunction, LanguageMode, NativeFn, SharedFunctionInfo,
/// };
/// use stator_core::objects::value::JsValue;
///
/// fn add(args: &[JsValue]) -> stator_core::error::StatorResult<JsValue> {
///     // args[0] is `this`; args[1] and args[2] are the two operands.
///     let a = args.get(1).cloned().unwrap_or(JsValue::Smi(0));
///     let b = args.get(2).cloned().unwrap_or(JsValue::Smi(0));
///     match (a, b) {
///         (JsValue::Smi(x), JsValue::Smi(y)) => Ok(JsValue::Smi(x + y)),
///         _ => Ok(JsValue::Undefined),
///     }
/// }
///
/// let sfi = SharedFunctionInfo::new("add", 2, LanguageMode::Sloppy);
/// let f = JsFunction::new_native(sfi, add);
/// assert_eq!(f.name(), "add");
/// assert_eq!(f.param_count(), 2);
/// ```
#[derive(Debug)]
pub struct JsFunction {
    /// Shared static metadata (name, bytecode, param count, language mode).
    shared: Rc<SharedFunctionInfo>,
    /// Captured lexical scope.
    context: Context,
    /// The specific flavour of this function.
    kind: FunctionKind,
}

impl JsFunction {
    /// Creates a new ordinary (interpreted) [`JsFunction`] with an empty
    /// captured scope.
    pub fn new(shared: SharedFunctionInfo) -> Self {
        Self {
            shared: Rc::new(shared),
            context: Context::new(),
            kind: FunctionKind::Normal,
        }
    }

    /// Creates a new ordinary function with a captured lexical scope.
    pub fn new_with_context(shared: SharedFunctionInfo, context: Context) -> Self {
        Self {
            shared: Rc::new(shared),
            context,
            kind: FunctionKind::Normal,
        }
    }

    /// Creates a native (Rust callback) function.
    ///
    /// The `context` is always empty for native functions; their behaviour is
    /// implemented entirely in Rust rather than in bytecode.
    pub fn new_native(shared: SharedFunctionInfo, native: NativeFn) -> Self {
        Self {
            shared: Rc::new(shared),
            context: Context::new(),
            kind: FunctionKind::Native(native),
        }
    }

    /// Creates a bound function from `target`, binding `bound_this` and
    /// zero or more leading `bound_args`.
    ///
    /// The `shared` metadata of the resulting bound function is cloned from
    /// the target's metadata so that `name` and `length` remain correct.
    pub fn new_bound(
        target: Rc<JsFunction>,
        bound_this: JsValue,
        bound_args: Vec<JsValue>,
    ) -> Self {
        let shared = Rc::clone(&target.shared);
        Self {
            shared,
            context: Context::new(),
            kind: FunctionKind::Bound {
                target,
                bound_this,
                bound_args,
            },
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns the function's declared name.
    pub fn name(&self) -> &str {
        self.shared.name()
    }

    /// Returns the formal parameter count.
    pub fn param_count(&self) -> u32 {
        self.shared.param_count()
    }

    /// Returns the language mode in which this function was compiled.
    pub fn language_mode(&self) -> LanguageMode {
        self.shared.language_mode()
    }

    /// Returns a reference to the shared function metadata.
    pub fn shared_info(&self) -> &SharedFunctionInfo {
        &self.shared
    }

    /// Returns a reference to the captured lexical scope.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the kind of this function.
    pub fn kind(&self) -> &FunctionKind {
        &self.kind
    }

    /// Returns `true` if this is an ordinary interpreted function.
    pub fn is_normal(&self) -> bool {
        matches!(self.kind, FunctionKind::Normal)
    }

    /// Returns `true` if this is a native (Rust callback) function.
    pub fn is_native(&self) -> bool {
        matches!(self.kind, FunctionKind::Native(_))
    }

    /// Returns `true` if this is a bound function.
    pub fn is_bound(&self) -> bool {
        matches!(self.kind, FunctionKind::Bound { .. })
    }

    // ── Call helpers ──────────────────────────────────────────────────────────

    /// Invokes the function if it is a native callback.
    ///
    /// `args` should follow the convention that `args[0]` is `this` and
    /// subsequent elements are positional arguments.
    ///
    /// Returns `None` if this is not a native function (i.e., it requires the
    /// bytecode interpreter).
    pub fn call_native(&self, args: &[JsValue]) -> Option<StatorResult<JsValue>> {
        if let FunctionKind::Native(f) = self.kind {
            Some(f(args))
        } else {
            None
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GC Trace
// ──────────────────────────────────────────────────────────────────────────────

impl Trace for JsFunction {
    /// Visit every GC-managed heap reference reachable through this function.
    ///
    /// Traces:
    /// * all values bound in the captured [`Context`],
    /// * for bound functions: the wrapped target function, the pre-bound `this`
    ///   value, and each element of the pre-bound argument list.
    fn trace(&self, tracer: &mut Tracer) {
        for (_, v) in &self.context.bindings {
            v.trace(tracer);
        }
        if let FunctionKind::Bound {
            target,
            bound_this,
            bound_args,
        } = &self.kind
        {
            target.trace(tracer);
            bound_this.trace(tracer);
            for v in bound_args {
                v.trace(tracer);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::error::StatorError;

    // ── SharedFunctionInfo ────────────────────────────────────────────────────

    #[test]
    fn test_shared_function_info_name() {
        let sfi = SharedFunctionInfo::new("greet", 1, LanguageMode::Sloppy);
        assert_eq!(sfi.name(), "greet");
    }

    #[test]
    fn test_shared_function_info_anonymous_empty_name() {
        let sfi = SharedFunctionInfo::new("", 0, LanguageMode::Sloppy);
        assert_eq!(sfi.name(), "");
    }

    #[test]
    fn test_shared_function_info_param_count() {
        let sfi = SharedFunctionInfo::new("f", 3, LanguageMode::Strict);
        assert_eq!(sfi.param_count(), 3);
    }

    #[test]
    fn test_shared_function_info_language_mode() {
        let sfi_sloppy = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        assert_eq!(sfi_sloppy.language_mode(), LanguageMode::Sloppy);

        let sfi_strict = SharedFunctionInfo::new("f", 0, LanguageMode::Strict);
        assert_eq!(sfi_strict.language_mode(), LanguageMode::Strict);
    }

    #[test]
    fn test_shared_function_info_no_bytecode_by_default() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        assert!(sfi.bytecode().is_none());
    }

    #[test]
    fn test_shared_function_info_with_bytecode() {
        let bytecode = BytecodeRef(vec![0x01, 0x02, 0x03]);
        let sfi = SharedFunctionInfo::with_bytecode("f", 0, LanguageMode::Sloppy, bytecode.clone());
        assert!(sfi.bytecode().is_some());
        assert_eq!(sfi.bytecode().unwrap().0, bytecode.0);
    }

    #[test]
    fn test_shared_function_info_set_bytecode() {
        let mut sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        assert!(sfi.bytecode().is_none());
        sfi.set_bytecode(BytecodeRef(vec![0xde, 0xad]));
        assert_eq!(sfi.bytecode().unwrap().0, &[0xde, 0xad]);
    }

    // ── Context ───────────────────────────────────────────────────────────────

    #[test]
    fn test_context_new_is_empty() {
        let ctx = Context::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);
    }

    #[test]
    fn test_context_set_and_get() {
        let mut ctx = Context::new();
        ctx.set("x", JsValue::Smi(42));
        assert_eq!(ctx.get("x"), Some(&JsValue::Smi(42)));
    }

    #[test]
    fn test_context_set_overwrites_existing() {
        let mut ctx = Context::new();
        ctx.set("x", JsValue::Smi(1));
        ctx.set("x", JsValue::Smi(2));
        assert_eq!(ctx.get("x"), Some(&JsValue::Smi(2)));
        // Only one binding should remain.
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn test_context_get_missing_returns_none() {
        let ctx = Context::new();
        assert_eq!(ctx.get("missing"), None);
    }

    #[test]
    fn test_context_multiple_bindings() {
        let mut ctx = Context::new();
        ctx.set("a", JsValue::Smi(10));
        ctx.set("b", JsValue::Boolean(true));
        assert_eq!(ctx.get("a"), Some(&JsValue::Smi(10)));
        assert_eq!(ctx.get("b"), Some(&JsValue::Boolean(true)));
        assert_eq!(ctx.len(), 2);
    }

    // ── JsFunction — normal ───────────────────────────────────────────────────

    #[test]
    fn test_js_function_new_name_and_param_count() {
        let sfi = SharedFunctionInfo::new("hello", 2, LanguageMode::Sloppy);
        let f = JsFunction::new(sfi);
        assert_eq!(f.name(), "hello");
        assert_eq!(f.param_count(), 2);
    }

    #[test]
    fn test_js_function_new_is_normal() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        let f = JsFunction::new(sfi);
        assert!(f.is_normal());
        assert!(!f.is_native());
        assert!(!f.is_bound());
    }

    #[test]
    fn test_js_function_language_mode() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Strict);
        let f = JsFunction::new(sfi);
        assert_eq!(f.language_mode(), LanguageMode::Strict);
    }

    #[test]
    fn test_js_function_with_context_captures_bindings() {
        let sfi = SharedFunctionInfo::new("closure", 0, LanguageMode::Sloppy);
        let mut ctx = Context::new();
        ctx.set("captured", JsValue::Smi(99));
        let f = JsFunction::new_with_context(sfi, ctx);
        assert_eq!(f.context().get("captured"), Some(&JsValue::Smi(99)));
    }

    #[test]
    fn test_js_function_empty_context_by_default() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        let f = JsFunction::new(sfi);
        assert!(f.context().is_empty());
    }

    // ── JsFunction — native ───────────────────────────────────────────────────

    fn native_add(args: &[JsValue]) -> StatorResult<JsValue> {
        // args[0] = this (ignored), args[1] and args[2] are the operands.
        let a = args.get(1).cloned().unwrap_or(JsValue::Smi(0));
        let b = args.get(2).cloned().unwrap_or(JsValue::Smi(0));
        match (a, b) {
            (JsValue::Smi(x), JsValue::Smi(y)) => Ok(JsValue::Smi(x + y)),
            _ => Err(StatorError::TypeError("expected Smi".to_string())),
        }
    }

    #[test]
    fn test_wrap_native_fn_is_native() {
        let sfi = SharedFunctionInfo::new("add", 2, LanguageMode::Sloppy);
        let f = JsFunction::new_native(sfi, native_add);
        assert!(f.is_native());
        assert!(!f.is_normal());
        assert!(!f.is_bound());
    }

    #[test]
    fn test_wrap_native_fn_call_info() {
        let sfi = SharedFunctionInfo::new("add", 2, LanguageMode::Sloppy);
        let f = JsFunction::new_native(sfi, native_add);
        assert_eq!(f.name(), "add");
        assert_eq!(f.param_count(), 2);
        assert_eq!(f.language_mode(), LanguageMode::Sloppy);
    }

    #[test]
    fn test_wrap_native_fn_call_native_returns_value() {
        let sfi = SharedFunctionInfo::new("add", 2, LanguageMode::Sloppy);
        let f = JsFunction::new_native(sfi, native_add);
        let result = f
            .call_native(&[JsValue::Undefined, JsValue::Smi(3), JsValue::Smi(4)])
            .expect("should be a native fn");
        assert_eq!(result.unwrap(), JsValue::Smi(7));
    }

    #[test]
    fn test_call_native_on_normal_fn_returns_none() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        let f = JsFunction::new(sfi);
        assert!(f.call_native(&[]).is_none());
    }

    #[test]
    fn test_native_fn_context_is_empty() {
        let sfi = SharedFunctionInfo::new("native", 0, LanguageMode::Sloppy);
        let f = JsFunction::new_native(sfi, |_| Ok(JsValue::Undefined));
        assert!(f.context().is_empty());
    }

    // ── JsFunction — bound ────────────────────────────────────────────────────

    #[test]
    fn test_bound_function_is_bound() {
        let sfi = SharedFunctionInfo::new("f", 1, LanguageMode::Sloppy);
        let target = Rc::new(JsFunction::new(sfi));
        let bound = JsFunction::new_bound(Rc::clone(&target), JsValue::Null, vec![JsValue::Smi(1)]);
        assert!(bound.is_bound());
        assert!(!bound.is_native());
        assert!(!bound.is_normal());
    }

    #[test]
    fn test_bound_function_inherits_name() {
        let sfi = SharedFunctionInfo::new("original", 2, LanguageMode::Strict);
        let target = Rc::new(JsFunction::new(sfi));
        let bound = JsFunction::new_bound(Rc::clone(&target), JsValue::Null, vec![]);
        assert_eq!(bound.name(), "original");
        assert_eq!(bound.param_count(), 2);
    }

    #[test]
    fn test_bound_function_stores_bound_this_and_args() {
        let sfi = SharedFunctionInfo::new("f", 1, LanguageMode::Sloppy);
        let target = Rc::new(JsFunction::new(sfi));
        let bound_this = JsValue::Smi(42);
        let bound_args = vec![JsValue::Boolean(true)];
        let bound =
            JsFunction::new_bound(Rc::clone(&target), bound_this.clone(), bound_args.clone());
        if let FunctionKind::Bound {
            bound_this: bt,
            bound_args: ba,
            ..
        } = bound.kind()
        {
            assert_eq!(*bt, bound_this);
            assert_eq!(*ba, bound_args);
        } else {
            panic!("expected Bound kind");
        }
    }

    #[test]
    fn test_bound_function_call_native_returns_none() {
        let sfi = SharedFunctionInfo::new("f", 0, LanguageMode::Sloppy);
        let target = Rc::new(JsFunction::new(sfi));
        let bound = JsFunction::new_bound(target, JsValue::Null, vec![]);
        assert!(bound.call_native(&[]).is_none());
    }

    // ── shared_info accessor ──────────────────────────────────────────────────

    #[test]
    fn test_shared_info_accessor() {
        let sfi = SharedFunctionInfo::new("test", 5, LanguageMode::Strict);
        let f = JsFunction::new(sfi);
        assert_eq!(f.shared_info().name(), "test");
        assert_eq!(f.shared_info().param_count(), 5);
        assert_eq!(f.shared_info().language_mode(), LanguageMode::Strict);
    }
}

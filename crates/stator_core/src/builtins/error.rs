//! ECMAScript §20.5 `Error` built-in and error hierarchy.
//!
//! This module provides [`JsError`], the Rust representation of a JavaScript
//! `Error` object, together with constructor helpers for every error type in
//! the ECMAScript standard hierarchy and the V8-compatible
//! `Error.captureStackTrace` / `Error.stackTraceLimit` extension.
//!
//! # Error kinds
//!
//! | Kind | JS constructor |
//! |---|---|
//! | [`ErrorKind::Error`] | `Error` |
//! | [`ErrorKind::TypeError`] | `TypeError` |
//! | [`ErrorKind::RangeError`] | `RangeError` |
//! | [`ErrorKind::ReferenceError`] | `ReferenceError` |
//! | [`ErrorKind::SyntaxError`] | `SyntaxError` |
//! | [`ErrorKind::URIError`] | `URIError` |
//! | [`ErrorKind::EvalError`] | `EvalError` |
//! | [`ErrorKind::AggregateError`] | `AggregateError` |
//!
//! # Stack traces
//!
//! Each error records the JavaScript call stack at the point of construction,
//! capped at [`STACK_TRACE_LIMIT`] frames.  The interpreter pushes and pops
//! frame names into the thread-local [`CALL_STACK`] before and after every
//! function call, so the captured trace is always meaningful.
//!
//! # V8 extensions
//!
//! [`error_capture_stack_trace`] and [`STACK_TRACE_LIMIT`] replicate the V8
//! `Error.captureStackTrace` / `Error.stackTraceLimit` API.

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::property_map::PropertyMap;
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Stack-trace limit
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of stack frames captured in an error's `stack` property
/// (mirrors V8's `Error.stackTraceLimit` default of 10).
///
/// Modify at runtime via [`set_stack_trace_limit`] /
/// [`get_stack_trace_limit`].
static STACK_TRACE_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10);

/// Return the current `Error.stackTraceLimit` value.
pub fn get_stack_trace_limit() -> usize {
    STACK_TRACE_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set a new `Error.stackTraceLimit` value.
pub fn set_stack_trace_limit(limit: usize) {
    STACK_TRACE_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread-local call stack for stack-trace capture
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// The JavaScript call stack, maintained by the interpreter.
    ///
    /// Each entry is the name of a function frame (or `"<anonymous>"` when the
    /// function has no name).  The interpreter pushes a name before entering a
    /// nested call and pops it on return.  [`capture_stack_trace`] reads this
    /// list when a new error is created.
    static CALL_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Maximum JavaScript call-stack depth.
///
/// When the number of nested interpreter frames reaches this limit,
/// [`push_call_frame`] returns
/// `Err(StatorError::RangeError("Maximum call stack size exceeded"))`,
/// matching the behaviour of V8 and SpiderMonkey for infinite-recursion
/// programs.  Keeping this below the OS thread-stack size prevents a fatal
/// `stack overflow, aborting` abort that cannot be caught.
///
/// The interpreter uses `stacker::maybe_grow` to dynamically extend the
/// native stack via `mmap`/`VirtualAlloc` when headroom is low, so each
/// recursive `Interpreter::run` call is safe from a raw stack overflow.
/// A limit of 1024 is high enough for realistic JavaScript programs
/// (V8 and SpiderMonkey allow similar depths) while still catching
/// infinite-recursion bugs with a proper `RangeError` long before memory
/// is exhausted.
pub const MAX_CALL_STACK_DEPTH: usize = 1024;

/// Push a frame name onto the thread-local call stack.
///
/// Returns `Err(StatorError::RangeError)` when the call stack would exceed
/// [`MAX_CALL_STACK_DEPTH`], so that the interpreter can surface a proper
/// JavaScript `RangeError` instead of aborting on a native stack overflow.
///
/// Call this immediately before entering a nested interpreter call.
pub fn push_call_frame(name: impl Into<String>) -> StatorResult<()> {
    CALL_STACK.with(|cs| {
        let mut stack = cs.borrow_mut();
        if stack.len() >= MAX_CALL_STACK_DEPTH {
            return Err(StatorError::RangeError(
                "Maximum call stack size exceeded".to_string(),
            ));
        }
        stack.push(name.into());
        Ok(())
    })
}

/// Return the current depth of the thread-local call stack.
///
/// The depth is 0 at the top-level script, 1 inside the first function call,
/// and so on.  The debugger uses this to implement step-over and step-out:
/// step-over pauses when the depth returns to (or below) the depth at the
/// time the step was requested; step-out pauses when the depth drops *below*
/// the saved depth.
pub fn call_stack_depth() -> usize {
    CALL_STACK.with(|cs| cs.borrow().len())
}

/// Pop the most recently pushed frame name from the thread-local call stack.
///
/// Call this immediately after returning from a nested interpreter call.
pub fn pop_call_frame() {
    CALL_STACK.with(|cs| {
        cs.borrow_mut().pop();
    });
}

/// Return a snapshot of the current JS call stack as a `Vec<String>`.
///
/// Each entry is a function-frame name (or `"<anonymous>"`).  The outermost
/// caller is at index 0; the most-recently-entered frame is last.
///
/// Used by the CPU profiler to record samples at safe points.
pub fn capture_call_stack() -> Vec<String> {
    CALL_STACK.with(|cs| cs.borrow().clone())
}

/// Clear the thread-local call stack entirely.
///
/// This is used by the Test262 runner (and similar harnesses) to reset the
/// call-stack state between test cases.  A `catch_unwind`-caught panic inside
/// the interpreter may leave frames on the stack (because the panic bypasses
/// the normal `pop_call_frame` calls), which would cause every subsequent test
/// to fail immediately with "Maximum call stack size exceeded".  Calling this
/// function after each test guarantees a clean starting state.
pub fn clear_call_stack() {
    CALL_STACK.with(|cs| cs.borrow_mut().clear());
}

/// Capture the current call stack as a formatted `stack` property string.
///
/// The returned string has the format:
/// ```text
/// ErrorName: message
///     at frame1
///     at frame2
///     …
/// ```
///
/// The number of frames is capped at [`get_stack_trace_limit`].
pub fn capture_stack_trace(error_name: &str, message: &str) -> String {
    let limit = get_stack_trace_limit();
    let mut result = format!("{error_name}: {message}");
    CALL_STACK.with(|cs| {
        let stack = cs.borrow();
        // Most-recent frame is at the end; iterate in reverse order.
        for frame in stack.iter().rev().take(limit) {
            result.push_str("\n    at ");
            result.push_str(frame);
        }
    });
    result
}

/// V8 extension: `Error.captureStackTrace(targetObject, constructorOpt)`.
///
/// Attaches a formatted stack trace string to `target`'s `stack` field.  The
/// optional `constructor_name` argument is used to strip frames up to (and
/// including) the named constructor from the trace, but the current
/// implementation ignores it and always captures the full current stack.
///
/// # Example
///
/// ```
/// use stator_core::builtins::error::{error_capture_stack_trace, JsError};
/// let mut err = JsError::new(stator_core::builtins::error::ErrorKind::Error, "oops".to_string());
/// error_capture_stack_trace(&mut err, None);
/// assert!(err.stack().starts_with("Error: oops"));
/// ```
pub fn error_capture_stack_trace(target: &mut JsError, _constructor_name: Option<&str>) {
    let new_stack = capture_stack_trace(target.name(), &target.message);
    target.stack = new_stack;
}

// ─────────────────────────────────────────────────────────────────────────────
// ErrorKind
// ─────────────────────────────────────────────────────────────────────────────

/// The kind (class name) of a JavaScript `Error` object.
///
/// Each variant maps to one of the standard ECMAScript error constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Generic `Error`.
    Error,
    /// `TypeError` — a value is not of the expected type.
    TypeError,
    /// `RangeError` — a value is out of the allowed range.
    RangeError,
    /// `ReferenceError` — a variable that does not exist was accessed.
    ReferenceError,
    /// `SyntaxError` — the source text could not be parsed.
    SyntaxError,
    /// `URIError` — a URI handling function received a malformed URI.
    URIError,
    /// `EvalError` — an error related to the global `eval` function.
    EvalError,
    /// `AggregateError` — wraps multiple errors (e.g. from `Promise.any`).
    AggregateError,
}

impl ErrorKind {
    /// Return the ECMAScript `name` property string for this error kind.
    ///
    /// ```
    /// use stator_core::builtins::error::ErrorKind;
    /// assert_eq!(ErrorKind::TypeError.as_name(), "TypeError");
    /// assert_eq!(ErrorKind::AggregateError.as_name(), "AggregateError");
    /// ```
    pub fn as_name(self) -> &'static str {
        match self {
            Self::Error => "Error",
            Self::TypeError => "TypeError",
            Self::RangeError => "RangeError",
            Self::ReferenceError => "ReferenceError",
            Self::SyntaxError => "SyntaxError",
            Self::URIError => "URIError",
            Self::EvalError => "EvalError",
            Self::AggregateError => "AggregateError",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JsError
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript `Error` object.
///
/// Holds the error `kind` (which determines the `name` property), the
/// human-readable `message`, and the captured `stack` trace string.
///
/// For `AggregateError`, the inner `errors` field holds the list of
/// constituent errors.
///
/// # ECMAScript properties
///
/// | Property | Accessor |
/// |---|---|
/// | `name` | [`JsError::name`] |
/// | `message` | [`JsError::message`] |
/// | `stack` | [`JsError::stack`] |
/// | `cause` | [`JsError::cause`] (ES2022) |
/// | `errors` | [`JsError::errors`] (`AggregateError` only) |
#[derive(Debug, Clone, PartialEq)]
pub struct JsError {
    /// The kind of this error (determines the `name` property).
    pub kind: ErrorKind,
    /// The human-readable error description.
    pub message: String,
    /// The formatted stack trace string (captured at construction time).
    pub stack: String,
    /// Inner errors for `AggregateError` (empty for all other kinds).
    pub errors: Vec<Rc<JsError>>,
    /// The ES2022 `cause` property — the underlying reason for this error.
    ///
    /// Set when the constructor receives an options object with a `cause`
    /// property, e.g. `new Error("msg", { cause: originalError })`.
    pub cause: Option<JsValue>,
    /// User-set property overlay.
    ///
    /// Stores values written by JS code (e.g. `err.message = "new"`).
    /// [`proto_lookup`](crate::interpreter::Interpreter) checks this map
    /// first, falling back to the built-in fields above when a key is absent.
    pub props: RefCell<PropertyMap>,
}

impl JsError {
    /// Create a new `JsError` with the given kind and message.
    ///
    /// The `stack` property is populated by capturing the current thread-local
    /// call stack at the point of this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use stator_core::builtins::error::{JsError, ErrorKind};
    ///
    /// let e = JsError::new(ErrorKind::TypeError, "not a function".to_string());
    /// assert_eq!(e.name(), "TypeError");
    /// assert_eq!(e.message(), "not a function");
    /// assert!(e.stack().starts_with("TypeError: not a function"));
    /// ```
    pub fn new(kind: ErrorKind, message: String) -> Self {
        let stack = capture_stack_trace(kind.as_name(), &message);
        Self {
            kind,
            message,
            stack,
            errors: Vec::new(),
            cause: None,
            props: RefCell::new(PropertyMap::new()),
        }
    }

    /// Create a new `AggregateError` wrapping `errors` with the given message.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    /// use stator_core::builtins::error::{JsError, ErrorKind};
    ///
    /// let e1 = Rc::new(JsError::new(ErrorKind::TypeError, "bad type".to_string()));
    /// let e2 = Rc::new(JsError::new(ErrorKind::RangeError, "out of range".to_string()));
    /// let agg = JsError::new_aggregate(vec![e1, e2], "All promises rejected".to_string());
    /// assert_eq!(agg.name(), "AggregateError");
    /// assert_eq!(agg.errors.len(), 2);
    /// ```
    pub fn new_aggregate(errors: Vec<Rc<Self>>, message: String) -> Self {
        let stack = capture_stack_trace("AggregateError", &message);
        Self {
            kind: ErrorKind::AggregateError,
            message,
            stack,
            errors,
            cause: None,
            props: RefCell::new(PropertyMap::new()),
        }
    }

    /// The ECMAScript `name` property — the error constructor name.
    pub fn name(&self) -> &str {
        self.kind.as_name()
    }

    /// The ECMAScript `message` property — the human-readable description.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The ECMAScript `stack` property — the formatted stack trace.
    pub fn stack(&self) -> &str {
        &self.stack
    }

    /// The ES2022 `cause` property — the underlying error cause, if any.
    ///
    /// Returns `None` when the error was constructed without a `cause` option.
    pub fn cause(&self) -> Option<&JsValue> {
        self.cause.as_ref()
    }

    /// Builder: set the `cause` property on this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use stator_core::builtins::error::{JsError, ErrorKind};
    /// use stator_core::objects::value::JsValue;
    ///
    /// let inner = JsValue::String("disk full".to_string().into());
    /// let e = JsError::new(ErrorKind::Error, "write failed".to_string())
    ///     .with_cause(inner.clone());
    /// assert_eq!(e.cause(), Some(&inner));
    /// ```
    pub fn with_cause(mut self, cause: JsValue) -> Self {
        self.cause = Some(cause);
        self
    }

    /// ECMAScript §20.5.3.4 `Error.prototype.toString()`.
    ///
    /// Returns `"name: message"`, or just `"name"` when `message` is empty,
    /// or just `"message"` when `name` is empty.
    /// Respects user-set overrides in the property overlay.
    ///
    /// ```
    /// use stator_core::builtins::error::{JsError, ErrorKind};
    ///
    /// let e = JsError::new(ErrorKind::RangeError, "index out of bounds".to_string());
    /// assert_eq!(e.to_error_string(), "RangeError: index out of bounds");
    ///
    /// let e2 = JsError::new(ErrorKind::Error, String::new());
    /// assert_eq!(e2.to_error_string(), "Error");
    /// ```
    pub fn to_error_string(&self) -> String {
        let props = self.props.borrow();
        let name = match props.get("name") {
            Some(JsValue::String(s)) => s.to_string(),
            _ => self.kind.as_name().to_string(),
        };
        let msg = match props.get("message") {
            Some(JsValue::String(s)) => s.to_string(),
            _ => self.message.clone(),
        };
        drop(props);

        if name.is_empty() {
            return msg;
        }
        if msg.is_empty() {
            return name;
        }
        format!("{name}: {msg}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Create a new `Error` object.
///
/// ```
/// use stator_core::builtins::error::{error_new, ErrorKind};
/// let e = error_new("something went wrong".to_string());
/// assert_eq!(e.kind, ErrorKind::Error);
/// ```
pub fn error_new(message: String) -> JsError {
    JsError::new(ErrorKind::Error, message)
}

/// Create a new `TypeError` object.
///
/// ```
/// use stator_core::builtins::error::{type_error_new, ErrorKind};
/// let e = type_error_new("not a function".to_string());
/// assert_eq!(e.kind, ErrorKind::TypeError);
/// ```
pub fn type_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::TypeError, message)
}

/// Create a new `RangeError` object.
///
/// ```
/// use stator_core::builtins::error::{range_error_new, ErrorKind};
/// let e = range_error_new("stack overflow".to_string());
/// assert_eq!(e.kind, ErrorKind::RangeError);
/// ```
pub fn range_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::RangeError, message)
}

/// Create a new `ReferenceError` object.
///
/// ```
/// use stator_core::builtins::error::{reference_error_new, ErrorKind};
/// let e = reference_error_new("x is not defined".to_string());
/// assert_eq!(e.kind, ErrorKind::ReferenceError);
/// ```
pub fn reference_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::ReferenceError, message)
}

/// Create a new `SyntaxError` object.
///
/// ```
/// use stator_core::builtins::error::{syntax_error_new, ErrorKind};
/// let e = syntax_error_new("unexpected token '}'".to_string());
/// assert_eq!(e.kind, ErrorKind::SyntaxError);
/// ```
pub fn syntax_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::SyntaxError, message)
}

/// Create a new `URIError` object.
///
/// ```
/// use stator_core::builtins::error::{uri_error_new, ErrorKind};
/// let e = uri_error_new("malformed URI sequence".to_string());
/// assert_eq!(e.kind, ErrorKind::URIError);
/// ```
pub fn uri_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::URIError, message)
}

/// Create a new `EvalError` object.
///
/// ```
/// use stator_core::builtins::error::{eval_error_new, ErrorKind};
/// let e = eval_error_new("eval is not supported".to_string());
/// assert_eq!(e.kind, ErrorKind::EvalError);
/// ```
pub fn eval_error_new(message: String) -> JsError {
    JsError::new(ErrorKind::EvalError, message)
}

/// Create a new `AggregateError` wrapping `errors` with the given message.
///
/// ```
/// use std::rc::Rc;
/// use stator_core::builtins::error::{aggregate_error_new, type_error_new, ErrorKind};
/// let inner = Rc::new(type_error_new("bad type".to_string()));
/// let e = aggregate_error_new(vec![inner], "All promises rejected".to_string());
/// assert_eq!(e.kind, ErrorKind::AggregateError);
/// assert_eq!(e.errors.len(), 1);
/// ```
pub fn aggregate_error_new(errors: Vec<Rc<JsError>>, message: String) -> JsError {
    JsError::new_aggregate(errors, message)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ErrorKind ────────────────────────────────────────────────────────────

    #[test]
    fn test_error_kind_names() {
        assert_eq!(ErrorKind::Error.as_name(), "Error");
        assert_eq!(ErrorKind::TypeError.as_name(), "TypeError");
        assert_eq!(ErrorKind::RangeError.as_name(), "RangeError");
        assert_eq!(ErrorKind::ReferenceError.as_name(), "ReferenceError");
        assert_eq!(ErrorKind::SyntaxError.as_name(), "SyntaxError");
        assert_eq!(ErrorKind::URIError.as_name(), "URIError");
        assert_eq!(ErrorKind::EvalError.as_name(), "EvalError");
        assert_eq!(ErrorKind::AggregateError.as_name(), "AggregateError");
    }

    // ── JsError::new ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_new_type_error() {
        let e = JsError::new(ErrorKind::TypeError, "not a function".to_string());
        assert_eq!(e.kind, ErrorKind::TypeError);
        assert_eq!(e.message(), "not a function");
        assert_eq!(e.name(), "TypeError");
        assert!(e.stack().starts_with("TypeError: not a function"));
        assert!(e.errors.is_empty());
    }

    #[test]
    fn test_error_new_range_error() {
        let e = range_error_new("index out of range".to_string());
        assert_eq!(e.kind, ErrorKind::RangeError);
        assert_eq!(e.to_error_string(), "RangeError: index out of range");
    }

    #[test]
    fn test_error_new_reference_error() {
        let e = reference_error_new("x is not defined".to_string());
        assert_eq!(e.name(), "ReferenceError");
        assert_eq!(e.message(), "x is not defined");
    }

    #[test]
    fn test_error_new_syntax_error() {
        let e = syntax_error_new("unexpected token".to_string());
        assert_eq!(e.name(), "SyntaxError");
        assert_eq!(e.to_error_string(), "SyntaxError: unexpected token");
    }

    #[test]
    fn test_error_new_uri_error() {
        let e = uri_error_new("malformed URI".to_string());
        assert_eq!(e.name(), "URIError");
    }

    #[test]
    fn test_error_new_eval_error() {
        let e = eval_error_new("eval is not supported".to_string());
        assert_eq!(e.name(), "EvalError");
    }

    #[test]
    fn test_error_empty_message() {
        let e = error_new(String::new());
        assert_eq!(e.to_error_string(), "Error");
        assert!(e.stack().starts_with("Error"));
    }

    // ── AggregateError ───────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_error_new() {
        let e1 = Rc::new(type_error_new("bad type".to_string()));
        let e2 = Rc::new(range_error_new("out of range".to_string()));
        let agg = aggregate_error_new(
            vec![e1.clone(), e2.clone()],
            "multiple failures".to_string(),
        );
        assert_eq!(agg.kind, ErrorKind::AggregateError);
        assert_eq!(agg.message(), "multiple failures");
        assert_eq!(agg.name(), "AggregateError");
        assert_eq!(agg.errors.len(), 2);
        assert_eq!(agg.errors[0].kind, ErrorKind::TypeError);
        assert_eq!(agg.errors[1].kind, ErrorKind::RangeError);
    }

    // ── to_error_string ──────────────────────────────────────────────────────

    #[test]
    fn test_to_error_string_with_message() {
        let e = JsError::new(ErrorKind::Error, "something failed".to_string());
        assert_eq!(e.to_error_string(), "Error: something failed");
    }

    #[test]
    fn test_to_error_string_empty_message() {
        let e = JsError::new(ErrorKind::TypeError, String::new());
        assert_eq!(e.to_error_string(), "TypeError");
    }

    // ── property overlay (settable name/message) ─────────────────────────────

    #[test]
    fn test_to_error_string_overridden_message() {
        let e = JsError::new(ErrorKind::Error, "original".to_string());
        e.props
            .borrow_mut()
            .insert("message".to_string(), JsValue::String("overridden".into()));
        assert_eq!(e.to_error_string(), "Error: overridden");
    }

    #[test]
    fn test_to_error_string_overridden_name() {
        let e = JsError::new(ErrorKind::Error, "msg".to_string());
        e.props
            .borrow_mut()
            .insert("name".to_string(), JsValue::String("CustomError".into()));
        assert_eq!(e.to_error_string(), "CustomError: msg");
    }

    #[test]
    fn test_to_error_string_overridden_name_empty() {
        let e = JsError::new(ErrorKind::Error, "msg".to_string());
        e.props
            .borrow_mut()
            .insert("name".to_string(), JsValue::String(String::new().into()));
        assert_eq!(e.to_error_string(), "msg");
    }

    #[test]
    fn test_props_overlay_custom_property() {
        let e = JsError::new(ErrorKind::Error, "test".to_string());
        e.props
            .borrow_mut()
            .insert("code".to_string(), JsValue::Smi(42));
        assert_eq!(e.props.borrow().get("code"), Some(&JsValue::Smi(42)));
    }

    // ── stack traces ─────────────────────────────────────────────────────────

    #[test]
    fn test_stack_trace_no_frames() {
        // Outside any interpreter call — call stack is empty.
        let e = JsError::new(ErrorKind::Error, "test".to_string());
        assert_eq!(e.stack(), "Error: test");
    }

    #[test]
    fn test_stack_trace_with_frames() {
        push_call_frame("outer");
        push_call_frame("inner");
        let e = JsError::new(ErrorKind::TypeError, "oops".to_string());
        pop_call_frame();
        pop_call_frame();

        assert!(e.stack().starts_with("TypeError: oops"));
        assert!(e.stack().contains("inner"));
        assert!(e.stack().contains("outer"));
    }

    #[test]
    fn test_stack_trace_limit() {
        let old_limit = get_stack_trace_limit();
        set_stack_trace_limit(2);

        push_call_frame("frameA");
        push_call_frame("frameB");
        push_call_frame("frameC");
        push_call_frame("frameD");
        let e = JsError::new(ErrorKind::Error, "limited".to_string());
        pop_call_frame();
        pop_call_frame();
        pop_call_frame();
        pop_call_frame();

        // Only the 2 most-recent frames should appear.
        let stack = e.stack();
        assert!(
            stack.contains("frameD"),
            "most recent frame 'frameD' missing: {stack}"
        );
        assert!(
            stack.contains("frameC"),
            "second frame 'frameC' missing: {stack}"
        );
        assert!(
            !stack.contains("frameB"),
            "frame 'frameB' should be truncated: {stack}"
        );
        assert!(
            !stack.contains("frameA"),
            "frame 'frameA' should be truncated: {stack}"
        );

        set_stack_trace_limit(old_limit);
    }

    // ── Error.captureStackTrace ──────────────────────────────────────────────

    #[test]
    fn test_capture_stack_trace() {
        push_call_frame("myFunction");
        let mut e = JsError::new(ErrorKind::Error, "captured".to_string());
        // Re-capture the stack trace on the existing error.
        error_capture_stack_trace(&mut e, None);
        pop_call_frame();

        assert!(e.stack().starts_with("Error: captured"));
        assert!(e.stack().contains("myFunction"));
    }

    // ── stack_trace_limit getter/setter ──────────────────────────────────────

    #[test]
    fn test_stack_trace_limit_getter_setter() {
        let original = get_stack_trace_limit();
        set_stack_trace_limit(5);
        assert_eq!(get_stack_trace_limit(), 5);
        set_stack_trace_limit(original);
    }

    // ── call-depth guard ─────────────────────────────────────────────────────

    #[test]
    fn test_push_call_frame_exceeds_limit_returns_range_error() {
        // Push MAX_CALL_STACK_DEPTH frames, then verify the next push fails.
        for _ in 0..MAX_CALL_STACK_DEPTH {
            push_call_frame("<test>").expect("should not fail below the limit");
        }
        let result = push_call_frame("<test>");
        // Clean up: pop all the frames we pushed.
        for _ in 0..MAX_CALL_STACK_DEPTH {
            pop_call_frame();
        }
        assert!(
            matches!(result, Err(crate::error::StatorError::RangeError(_))),
            "expected RangeError when call stack is full, got {result:?}"
        );
    }

    // ── cause property (ES2022) ──────────────────────────────────────────────

    #[test]
    fn test_error_cause_none_by_default() {
        let e = JsError::new(ErrorKind::Error, "no cause".to_string());
        assert!(e.cause().is_none());
    }

    #[test]
    fn test_error_with_cause() {
        let cause = JsValue::String("disk full".to_string().into());
        let e =
            JsError::new(ErrorKind::Error, "write failed".to_string()).with_cause(cause.clone());
        assert_eq!(e.cause(), Some(&cause));
    }

    #[test]
    fn test_error_cause_can_be_error_value() {
        let inner = JsValue::Error(Rc::new(type_error_new("bad type".to_string())));
        let outer = JsError::new(ErrorKind::Error, "wrapper".to_string()).with_cause(inner.clone());
        assert_eq!(outer.cause(), Some(&inner));
    }

    #[test]
    fn test_aggregate_error_cause_none_by_default() {
        let agg = aggregate_error_new(vec![], "agg".to_string());
        assert!(agg.cause().is_none());
    }

    #[test]
    fn test_aggregate_error_with_cause() {
        let cause = JsValue::Smi(42);
        let mut agg = aggregate_error_new(vec![], "agg".to_string());
        agg.cause = Some(cause.clone());
        assert_eq!(agg.cause(), Some(&cause));
    }

    // ── name/message inheritance ─────────────────────────────────────────────

    #[test]
    fn test_all_error_kinds_have_correct_names() {
        let kinds = [
            (ErrorKind::Error, "Error"),
            (ErrorKind::TypeError, "TypeError"),
            (ErrorKind::RangeError, "RangeError"),
            (ErrorKind::ReferenceError, "ReferenceError"),
            (ErrorKind::SyntaxError, "SyntaxError"),
            (ErrorKind::URIError, "URIError"),
            (ErrorKind::EvalError, "EvalError"),
            (ErrorKind::AggregateError, "AggregateError"),
        ];
        for (kind, expected_name) in kinds {
            let e = JsError::new(kind, "test".to_string());
            assert_eq!(e.name(), expected_name, "wrong name for {kind:?}");
            assert_eq!(e.message(), "test");
            assert!(
                e.stack().starts_with(&format!("{expected_name}: test")),
                "stack should start with error string for {kind:?}"
            );
        }
    }
}

//! Error types for the Stator engine.

use thiserror::Error;

/// All errors that can be produced by the Stator engine.
#[derive(Debug, Error)]
pub enum StatorError {
    /// The engine ran out of heap memory.
    #[error("out of memory")]
    OutOfMemory,

    /// A JavaScript TypeError was raised.
    #[error("TypeError: {0}")]
    TypeError(String),

    /// A JavaScript SyntaxError was raised.
    #[error("SyntaxError: {0}")]
    SyntaxError(String),

    /// A JavaScript ReferenceError was raised.
    #[error("ReferenceError: {0}")]
    ReferenceError(String),

    /// A JavaScript RangeError was raised.
    #[error("RangeError: {0}")]
    RangeError(String),

    /// A JavaScript URIError was raised (malformed URI in encodeURI / decodeURI).
    #[error("URIError: {0}")]
    URIError(String),

    /// An internal engine error that should not occur in normal operation.
    #[error("internal error: {0}")]
    Internal(String),

    /// A WebAssembly error (compilation, instantiation, or execution failure).
    #[error("WasmError: {0}")]
    WasmError(String),

    /// A JavaScript exception was thrown and propagated out of the current frame
    /// without being caught.  The inner string is the debug representation of
    /// the thrown value, kept to avoid a dependency cycle.
    #[error("Uncaught exception: {0}")]
    JsException(String),

    /// The interpreter was suspended by the debugger at the given bytecode
    /// byte offset.
    ///
    /// This error is returned from [`crate::interpreter::Interpreter::run`]
    /// when a breakpoint, `debugger;` statement, step condition, or
    /// pause-on-exception fires.  The interpreter frame is left in a
    /// consistent state so that execution can be resumed by calling
    /// [`crate::interpreter::Interpreter::run`] again after applying a
    /// [`crate::inspector::debugger::DebugAction`] via
    /// [`crate::inspector::debugger::Debugger::apply_action`].
    #[error("debugger paused at bytecode offset {bytecode_offset}")]
    DebuggerPaused {
        /// The bytecode byte offset of the instruction at which execution was
        /// paused.
        bytecode_offset: u32,
    },

    /// A pointer operation targeted memory outside the sandbox bounds.
    ///
    /// This error is returned by
    /// [`crate::sandbox::Sandbox::check_in_bounds`] when a raw pointer
    /// does not fall within the sandbox's reserved virtual-address range
    /// `[sandbox_base, sandbox_base + sandbox_size)`.
    #[error(
        "sandbox violation: address {address:#x} is outside sandbox \
         [{sandbox_base:#x}, {end:#x})",
        end = sandbox_base + sandbox_size
    )]
    SandboxViolation {
        /// The out-of-bounds address that triggered the violation.
        address: usize,
        /// Base address of the sandbox virtual-address range.
        sandbox_base: usize,
        /// Size of the sandbox virtual-address range in bytes.
        sandbox_size: usize,
    },
}

/// Convenient `Result` alias for fallible engine operations.
pub type StatorResult<T> = Result<T, StatorError>;

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
}

/// Convenient `Result` alias for fallible engine operations.
pub type StatorResult<T> = Result<T, StatorError>;

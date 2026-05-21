//! Error types for the Stator engine.

use thiserror::Error;

/// All errors that can be produced by the Stator engine.
#[derive(Debug, Error)]
pub enum StatorError {
    /// The engine ran out of heap memory.
    #[error("out of memory")]
    OutOfMemory,

    /// A JavaScript Error was raised.
    #[error("Error: {0}")]
    Error(String),
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

    /// A strict snapshot serializer encountered a heap value that cannot be
    /// safely captured.
    ///
    /// This is returned by the strict serialization entry points (e.g.
    /// [`crate::snapshot::serialize_globals_strict`]) instead of silently
    /// downgrading the value to `Undefined`, so embedders fail closed when
    /// asked to persist unsupported host/native state — native function
    /// closures, raw `Object` GC pointers, generators, iterators, promises,
    /// contexts, proxies, ArrayBuffers / TypedArrays / DataViews, internal
    /// `TheHole` sentinels, or any other non-deterministic host wrapper.
    #[error(
        "snapshot: unsupported heap value of class `{class}` at `{path}`{}",
        match reason {
            Some(r) => format!(": {r}"),
            None => String::new(),
        }
    )]
    SnapshotUnsupportedValue {
        /// Symbolic name of the rejected value class
        /// (e.g. `"NativeFunction"`, `"Promise"`, `"Proxy"`).
        class: &'static str,
        /// Dotted path from the snapshot root to the rejected value
        /// (e.g. `"globals.window.document"` or `"globals.arr[3].cb"`).
        path: String,
        /// Optional free-form reason describing why this class is unsafe to
        /// serialize for warm-context snapshots.
        reason: Option<&'static str>,
    },

    /// A manifest-aware snapshot was loaded with a callback manifest
    /// whose digest does not match the digest captured at snapshot
    /// create time.
    ///
    /// Emitted by
    /// [`crate::snapshot::reinstall_globals_with_manifest`] when the
    /// load-time [`crate::snapshot::SnapshotCallbackManifest`] does not
    /// exactly match (by digest and id set) the manifest that produced
    /// the snapshot.  v1 does not implement an `allow_extra` mode; any
    /// id present on one side but not the other is fatal.
    #[error(
        "snapshot: callback manifest mismatch (expected digest {expected}, found {found}; \
         missing_ids={missing_ids:?}, extra_ids={extra_ids:?})"
    )]
    SnapshotManifestMismatch {
        /// Hex-encoded digest captured in the snapshot header.
        expected: String,
        /// Hex-encoded digest of the load-time manifest.
        found: String,
        /// Ids present in the snapshot header but absent from the
        /// load-time manifest.
        missing_ids: Vec<String>,
        /// Ids present in the load-time manifest but absent from the
        /// snapshot header.
        extra_ids: Vec<String>,
    },

    /// A warm-context (`STWC`) snapshot header field did not match the
    /// load-time engine/embedder environment.
    ///
    /// Emitted by [`crate::snapshot::load_globals_stwc`] before any
    /// payload is decoded.  `field` names the compatibility key (e.g.
    /// `"magic"`, `"snapshot_format_ver"`, `"bytecode_format_ver"`,
    /// `"engine_crate_ver"`, `"ffi_abi_version"`, `"target_triple"`,
    /// `"build_id"`, `"build_features_hash"`, `"edge_release_hash"`,
    /// `"payload_len"`).  The found/expected strings are
    /// human-readable renderings of the snapshot vs. load-time
    /// values, suitable for direct surfacing through telemetry.
    #[error(
        "snapshot: warm-context compatibility mismatch on field `{field}` \
         (found {found}, expected {expected})"
    )]
    SnapshotCompatibilityMismatch {
        /// Name of the compatibility field that mismatched.
        field: &'static str,
        /// Human-readable rendering of the value stored in the snapshot
        /// header.
        found: String,
        /// Human-readable rendering of the value expected by the
        /// load-time engine/embedder environment.
        expected: String,
    },

    /// A warm-context (`STWC`) snapshot footer digest did not verify.
    ///
    /// Emitted by [`crate::snapshot::load_globals_stwc`] when the
    /// computed digest of the header + payload does not match the
    /// digest stored in the footer.  Any digest mismatch is fatal and
    /// no payload is decoded.
    #[error("snapshot: warm-context digest mismatch (expected {expected}, found {found})")]
    SnapshotDigestMismatch {
        /// Hex-encoded digest stored in the snapshot footer.
        expected: String,
        /// Hex-encoded digest recomputed from the on-disk bytes.
        found: String,
    },
}

/// Convenient `Result` alias for fallible engine operations.
pub type StatorResult<T> = Result<T, StatorError>;

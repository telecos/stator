//! WebAssembly backend for the Stator engine.
//!
//! This module integrates [Wasmtime] as the WebAssembly execution backend.
//! It provides three thin wrapper types that map naturally onto the Wasmtime
//! API while exposing a Stator-idiomatic interface:
//!
//! - [`WasmEngine`] — a configured Wasmtime engine (compilation settings,
//!   resource limits).  Cheap to clone; the underlying engine is
//!   reference-counted.
//! - [`WasmModule`] — a compiled Wasm module.  Created from raw `.wasm`
//!   bytes or textual WAT source via [`WasmModule::from_bytes`] /
//!   [`WasmModule::from_wat`].
//! - [`WasmInstance`] — a live module instance with its own linear memory
//!   and table.  Created from a [`WasmModule`] via
//!   [`WasmInstance::new`].
//!
//! # Value conversion
//!
//! [`js_value_to_wasm_val`] and [`wasm_val_to_js_value`] convert between
//! [`JsValue`] and [`wasmtime::Val`]:
//!
//! | JavaScript value      | Wasmtime value type       |
//! |-----------------------|---------------------------|
//! | `Smi(n)` / `HeapNumber(n as f64 → i32)` | `Val::I32`  |
//! | `HeapNumber(f)`       | `Val::F64`                |
//! | `Undefined` / `Null`  | `Val::I32(0)`             |
//! | `Boolean(b)`          | `Val::I32(b as i32)`      |
//!
//! # Example
//!
//! ```rust
//! use stator_jse::wasm::{WasmEngine, WasmInstance, WasmModule};
//!
//! let engine = WasmEngine::new();
//! let wat = r#"
//!     (module
//!         (func $add (export "add") (param i32 i32) (result i32)
//!             local.get 0
//!             local.get 1
//!             i32.add))
//! "#;
//! let module = WasmModule::from_wat(&engine, wat).unwrap();
//! let mut instance = WasmInstance::new(&engine, &module).unwrap();
//! let result = instance.call("add", &[1i32.into(), 2i32.into()]).unwrap();
//! assert_eq!(result[0].unwrap_i32(), 3);
//! ```

use std::sync::{Arc, Mutex};

use wasmtime::{
    Config, Engine, FuncType, Global, GlobalType, Instance, Linker, Module, Mutability, Store,
    UpdateDeadline, V128, Val, ValType,
};
pub use wasmtime::{MemoryType, SharedMemory};

use crate::error::{StatorError, StatorResult};
use crate::interpreter::{SCRIPT_TERMINATED_MESSAGE, check_interrupt_flag};
use crate::objects::value::JsValue;

// ─────────────────────────────────────────────────────────────────────────────
// Termination plumbing
// ─────────────────────────────────────────────────────────────────────────────
//
// Wasmtime provides an "epoch interruption" mechanism: compiled Wasm code
// periodically checks the engine's epoch counter against a per-store
// deadline.  When the deadline is reached the configured callback decides
// whether to trap the running call.  This is the only safe way to interrupt
// a Wasm call from another thread without relying on signals or signals
// emulation.  See:
//   https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#method.epoch_interruption
//
// Polling sites in this slice:
//   * Every [`WasmInstance::call`] entry checks the Stator interrupt flag
//     and short-circuits with [`script_terminated_error`] so a JS→Wasm
//     boundary is always observable, even if Wasm compiled code never
//     reaches an epoch check (e.g. very short call, no loops).
//   * Every [`WasmEngine::new`] enables epoch interruption and every
//     [`Store`] starts with `epoch_deadline = 1` and a callback that checks
//     this thread's published Stator interrupt flag.
//     [`interrupt_all_wasm_engines`] (called by the FFI termination
//     entry point) advances every live engine's epoch so in-flight Wasm calls
//     reach the callback at their next epoch check.
//
// Scope and limitations:
//   * The engine registry is process-global.  Triggering termination on one
//     isolate also bumps the epoch of Wasm engines belonging to other
//     isolates that happen to be running on this process.  The other
//     isolates will simply observe an extra (idempotent) epoch tick — they
//     will not trap unless their own embedder is concurrently asking for
//     termination — but the conservative behavior is to keep this in mind
//     for future per-isolate scoping.

/// Process-wide registry of live Wasm engines.  Used by
/// [`interrupt_all_wasm_engines`] to advance every engine's epoch when the
/// host requests termination.  Entries are added by [`WasmEngine::new`].
///
/// Engines are reference-counted internally by Wasmtime; storing a clone
/// here keeps the underlying engine alive even if the embedder drops its
/// last [`WasmEngine`].  For dogfood the number of engines is small and
/// bounded by isolate count, so unbounded growth is not a concern in
/// practice.
static WASM_ENGINES: Mutex<Vec<Engine>> = Mutex::new(Vec::new());

/// Advance the epoch of every registered Wasm [`Engine`] so any in-flight
/// Wasm call observes its deadline at its next epoch check.
///
/// Safe to call from any thread; safe to call when no Wasm is in flight.
/// Idempotent: calling twice in a row simply increments the epoch twice
/// (the deadline is already past after the first call).  The per-store
/// deadline callback decides whether the epoch tick represents this isolate's
/// own termination request or an unrelated cross-isolate broadcast.
pub fn interrupt_all_wasm_engines() {
    let guard = WASM_ENGINES
        .lock()
        .expect("Stator Wasm engine registry mutex poisoned");
    for engine in guard.iter() {
        engine.increment_epoch();
    }
}

fn register_engine(engine: &Engine) {
    WASM_ENGINES
        .lock()
        .expect("Stator Wasm engine registry mutex poisoned")
        .push(engine.clone());
}

// ─────────────────────────────────────────────────────────────────────────────
// WasmEngine
// ─────────────────────────────────────────────────────────────────────────────

/// A configured Wasmtime engine.
///
/// Wraps [`wasmtime::Engine`].  The engine holds JIT compilation settings and
/// is cheap to clone (the underlying engine is reference-counted by Wasmtime).
#[derive(Clone, Debug)]
pub struct WasmEngine {
    inner: Engine,
}

impl WasmEngine {
    /// Create a new [`WasmEngine`] with default compilation settings and
    /// epoch interruption enabled.
    ///
    /// Epoch interruption is the mechanism [`interrupt_all_wasm_engines`]
    /// uses to terminate an in-flight Wasm call.  Enabling it on the engine
    /// is a no-op unless a [`Store`] also configures a deadline; both are
    /// set up here for every Stator-created Wasm execution.
    pub fn new() -> Self {
        let mut config = Config::new();
        config.epoch_interruption(true);
        // Enable the threads proposal so importer modules can declare
        // `(memory ... shared)` imports and Stator can bind a
        // [`SharedMemory`] across stores. Shared memory is the only memory
        // primitive that can be safely passed between independent
        // [`wasmtime::Store`]s, so this is required by the module-graph
        // memory import contract documented on [`HostMemory`].
        config.wasm_threads(true);
        // Allow externally-created `SharedMemory` handles. Required to
        // route a Wasm-dependency module's exported shared memory through
        // the linker as an import to a separate Wasm instance.
        config.shared_memory(true);
        // `Engine::new` only fails for invalid `Config`s; the defaults plus
        // `epoch_interruption(true)` and `wasm_threads(true)` are always
        // valid.
        let inner = Engine::new(&config).expect("default wasmtime Config should be valid");
        register_engine(&inner);
        Self { inner }
    }

    /// Return a reference to the underlying [`wasmtime::Engine`].
    pub fn inner(&self) -> &Engine {
        &self.inner
    }
}

impl Default for WasmEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WasmModule
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled WebAssembly module.
///
/// Wraps [`wasmtime::Module`].  A module is produced by compiling raw Wasm
/// bytes (or WAT text) and can be instantiated multiple times with different
/// imports.
#[derive(Clone, Debug)]
pub struct WasmModule {
    inner: Module,
}

impl WasmModule {
    /// Compile a [`WasmModule`] from raw WebAssembly binary bytes.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if the bytes are not valid Wasm or
    /// if compilation fails.
    pub fn from_bytes(engine: &WasmEngine, bytes: &[u8]) -> StatorResult<Self> {
        Module::new(engine.inner(), bytes)
            .map(|inner| Self { inner })
            .map_err(|e| StatorError::WasmError(e.to_string()))
    }

    /// Compile a [`WasmModule`] from WebAssembly text format (WAT) source.
    ///
    /// This is a convenience wrapper around [`wasmtime::Module::new`] that
    /// first converts the WAT string to binary using Wasmtime's built-in WAT
    /// parser.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if the WAT is invalid or if
    /// compilation fails.
    pub fn from_wat(engine: &WasmEngine, wat: &str) -> StatorResult<Self> {
        Module::new(engine.inner(), wat)
            .map(|inner| Self { inner })
            .map_err(|e| StatorError::WasmError(e.to_string()))
    }

    /// Return a reference to the underlying [`wasmtime::Module`].
    pub fn inner(&self) -> &Module {
        &self.inner
    }

    /// Enumerate this module's import descriptors as a list of
    /// [`WasmImportDescriptor`] values, including typed metadata for imports
    /// that Stator does not yet bind at runtime.
    ///
    /// Iteration order matches the import section of the Wasm module.
    pub fn imports(&self) -> Vec<WasmImportDescriptor> {
        self.inner
            .imports()
            .map(|import| {
                let kind = match import.ty() {
                    wasmtime::ExternType::Func(ft) => WasmExternKind::Func {
                        params: ft.params().map(value_type_to_host_kind).collect(),
                        results: ft.results().map(value_type_to_host_kind).collect(),
                    },
                    wasmtime::ExternType::Global(gt) => WasmExternKind::Global {
                        value_type: value_type_to_host_kind(gt.content().clone()),
                        mutable: gt.mutability().is_var(),
                    },
                    wasmtime::ExternType::Memory(mt) => WasmExternKind::Memory {
                        minimum: mt.minimum(),
                        maximum: mt.maximum(),
                        memory64: mt.is_64(),
                        shared: mt.is_shared(),
                        page_size_log2: mt.page_size_log2(),
                    },
                    wasmtime::ExternType::Table(tt) => WasmExternKind::Table {
                        element: tt.element().to_string(),
                        minimum: tt.minimum(),
                        maximum: tt.maximum(),
                        table64: tt.is_64(),
                    },
                    _ => WasmExternKind::Other,
                };
                WasmImportDescriptor {
                    module: import.module().to_string(),
                    name: import.name().to_string(),
                    kind,
                }
            })
            .collect()
    }

    /// Look up the kind of a named function export.
    ///
    /// Returns `None` when the export does not exist or is not a function, or
    /// when the function uses Wasm value types that have no [`HostValKind`]
    /// representation (`v128`, references).
    pub fn exported_function_signature(
        &self,
        name: &str,
    ) -> Option<(Vec<HostValKind>, Vec<HostValKind>)> {
        let export = self.inner.exports().find(|e| e.name() == name)?;
        let func_ty = export.ty().func()?.clone();
        let params = func_ty
            .params()
            .map(value_type_to_host_kind)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let results = func_ty
            .results()
            .map(value_type_to_host_kind)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        Some((params, results))
    }

    /// Look up the type of a named global export.
    ///
    /// Returns `None` when the export does not exist, is not a global, or
    /// uses a Wasm value type that has no [`HostValKind`] representation
    /// (reference types). `v128` globals are represented by
    /// [`HostValKind::V128`].
    pub fn exported_global_type(&self, name: &str) -> Option<(HostValKind, bool)> {
        let export = self.inner.exports().find(|e| e.name() == name)?;
        let gt = export.ty().global()?.clone();
        let kind = value_type_to_host_kind(gt.content().clone()).ok()?;
        Some((kind, gt.mutability().is_var()))
    }

    /// Look up the declared type of a named memory export.
    ///
    /// Returns `None` when the export does not exist or is not a memory.
    /// The returned [`WasmMemoryTypeInfo`] mirrors the shape of
    /// [`WasmExternKind::Memory`] so import/export compatibility can be
    /// validated structurally (minimum/maximum pages, memory64 flag, shared
    /// flag, page-size log2).
    pub fn exported_memory_type(&self, name: &str) -> Option<WasmMemoryTypeInfo> {
        let export = self.inner.exports().find(|e| e.name() == name)?;
        let mt = export.ty().memory()?.clone();
        Some(WasmMemoryTypeInfo {
            minimum: mt.minimum(),
            maximum: mt.maximum(),
            memory64: mt.is_64(),
            shared: mt.is_shared(),
            page_size_log2: mt.page_size_log2(),
        })
    }

    /// Look up the declared type of a named table export.
    ///
    /// Returns `None` when the export does not exist or is not a table.
    /// The returned [`WasmTableTypeInfo`] mirrors the shape of
    /// [`WasmExternKind::Table`] so the FFI layer can structurally compare
    /// a dependency module's exported table type against an importer's
    /// expected table type (element reference type, min, max, table64) at
    /// compile/link time even though Stator cannot yet bind a table import
    /// at runtime.
    pub fn exported_table_type(&self, name: &str) -> Option<WasmTableTypeInfo> {
        let export = self.inner.exports().find(|e| e.name() == name)?;
        let tt = export.ty().table()?.clone();
        Some(WasmTableTypeInfo {
            element: tt.element().to_string(),
            minimum: tt.minimum(),
            maximum: tt.maximum(),
            table64: tt.is_64(),
        })
    }
}

/// Structural type information for a Wasm memory import or export, matching
/// the metadata fields of [`WasmExternKind::Memory`]. Used by the FFI layer
/// to validate that a dependency module's exported memory is compatible
/// with the importer's expected memory type before sharing a
/// [`SharedMemory`] across instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WasmMemoryTypeInfo {
    /// Minimum page count.
    pub minimum: u64,
    /// Optional maximum page count.
    pub maximum: Option<u64>,
    /// Whether this is a memory64 import/export.
    pub memory64: bool,
    /// Whether this is a shared-memory import/export.
    pub shared: bool,
    /// Log2 of the page size in bytes (16 ⇒ 64 KiB pages by default).
    pub page_size_log2: u8,
}

/// Structural type information for a Wasm table import or export, matching
/// the metadata fields of [`WasmExternKind::Table`]. Used by the FFI layer
/// to surface a precise type mismatch in the typed fail-closed diagnostic
/// emitted when a Wasm-to-Wasm table import cannot be bound because
/// Wasmtime tables are [`Store`]-bound and there is no `SharedTable`
/// primitive that can route a table across independent stores yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmTableTypeInfo {
    /// Wasmtime display string for the table element reference type
    /// (e.g. `"funcref"`).
    pub element: String,
    /// Minimum element count.
    pub minimum: u64,
    /// Optional maximum element count.
    pub maximum: Option<u64>,
    /// Whether this is a 64-bit table import/export.
    pub table64: bool,
}

fn value_type_to_host_kind(ty: wasmtime::ValType) -> Result<HostValKind, String> {
    Ok(match ty {
        wasmtime::ValType::I32 => HostValKind::I32,
        wasmtime::ValType::I64 => HostValKind::I64,
        wasmtime::ValType::F32 => HostValKind::F32,
        wasmtime::ValType::F64 => HostValKind::F64,
        wasmtime::ValType::V128 => HostValKind::V128,
        other => return Err(format!("unsupported Wasm value type {other}")),
    })
}

/// Description of a single Wasm import descriptor used for module-graph
/// integration. Mirrors the structure of `wasmtime::ImportType` in a form
/// that does not leak `wasmtime` types across the [`stator_jse`] boundary.
#[derive(Debug, Clone)]
pub struct WasmImportDescriptor {
    /// Wasm import module namespace (e.g. `"env"`).
    pub module: String,
    /// Wasm import field name (e.g. `"add"`).
    pub name: String,
    /// Kind and declared type metadata.
    pub kind: WasmExternKind,
}

/// The kind of a Wasm import or export, with declared type metadata. The
/// module graph integration in this slice only bridges function imports;
/// callers fail closed on globals, memories and tables with these precise
/// descriptors so unsupported imports cannot accidentally instantiate.
#[derive(Debug, Clone)]
pub enum WasmExternKind {
    /// A function with the declared parameter and result kinds.
    ///
    /// Each parameter / result is itself a `Result<HostValKind, String>` to
    /// surface unsupported value types (`v128`, references) without requiring
    /// the caller to depend on `wasmtime` directly.
    Func {
        /// Parameter kinds, in order.
        params: Vec<Result<HostValKind, String>>,
        /// Result kinds, in order.
        results: Vec<Result<HostValKind, String>>,
    },
    /// A global value.
    Global {
        /// Global value kind, or an unsupported value-type diagnostic.
        value_type: Result<HostValKind, String>,
        /// Whether the import requires a mutable global.
        mutable: bool,
    },
    /// A linear memory.
    Memory {
        /// Minimum page count.
        minimum: u64,
        /// Optional maximum page count.
        maximum: Option<u64>,
        /// Whether this is a memory64 import.
        memory64: bool,
        /// Whether this is a shared-memory import.
        shared: bool,
        /// Log2 page size in bytes.
        page_size_log2: u8,
    },
    /// A table.
    Table {
        /// Wasmtime display string for the table element reference type.
        element: String,
        /// Minimum element count.
        minimum: u64,
        /// Optional maximum element count.
        maximum: Option<u64>,
        /// Whether this is a 64-bit table import.
        table64: bool,
    },
    /// Any other extern kind not bridged by this slice (e.g. tag/event types).
    Other,
}

// ─────────────────────────────────────────────────────────────────────────────
// Host imports
// ─────────────────────────────────────────────────────────────────────────────
//
// `WasmInstance::new_with_imports` lets embedders bind host functions into the
// Wasm module's import namespace.  Each [`HostFunc`] declares a fully-qualified
// import name (`module::name`), a signature (parameter + result kinds), and a
// synchronous callback that runs on the thread executing the calling Wasm.
//
// Polling:
//   * Before invoking the embedder callback the host shim observes the Stator
//     interrupt flag and traps with [`SCRIPT_TERMINATED_MESSAGE`] if set.
//   * After the embedder callback returns, the flag is observed again.  This
//     covers the case where the embedder ran user-attributable work that
//     responded to a termination request.
//   * The callback returning `false` is reported as a Wasm trap to the running
//     module; the outer [`WasmInstance::call`] surfaces this as a
//     [`StatorError::WasmError`].

/// The C-ABI–compatible value kind used by [`HostVal`] and [`HostFunc`].
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HostValKind {
    /// Wasm `i32`.
    I32 = 0,
    /// Wasm `i64`.
    I64 = 1,
    /// Wasm `f32`.
    F32 = 2,
    /// Wasm `f64`.
    F64 = 3,
    /// Wasm `v128`. Reserved for Wasm-to-Wasm host-global imports; there is
    /// no JS-side representation, so any JS↔Wasm boundary conversion that
    /// encounters this kind fails closed.
    V128 = 4,
}

impl HostValKind {
    fn to_wasmtime(self) -> ValType {
        match self {
            HostValKind::I32 => ValType::I32,
            HostValKind::I64 => ValType::I64,
            HostValKind::F32 => ValType::F32,
            HostValKind::F64 => ValType::F64,
            HostValKind::V128 => ValType::V128,
        }
    }
}

/// A typed Wasm value used at the host-import boundary.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HostVal {
    /// Wasm `i32`.
    I32(i32),
    /// Wasm `i64`.
    I64(i64),
    /// Wasm `f32`.
    F32(f32),
    /// Wasm `f64`.
    F64(f64),
    /// Wasm `v128` carried as raw 128-bit bits in little-endian-lane order.
    /// Only used by the Wasm-to-Wasm global-import bridge; this value never
    /// participates in any JS↔Wasm conversion.
    V128(u128),
}

impl HostVal {
    /// Return the [`HostValKind`] discriminant of this value.
    pub fn kind(&self) -> HostValKind {
        match self {
            HostVal::I32(_) => HostValKind::I32,
            HostVal::I64(_) => HostValKind::I64,
            HostVal::F32(_) => HostValKind::F32,
            HostVal::F64(_) => HostValKind::F64,
            HostVal::V128(_) => HostValKind::V128,
        }
    }

    /// Return the zero value of the given kind.
    pub fn zero_of(kind: HostValKind) -> HostVal {
        match kind {
            HostValKind::I32 => HostVal::I32(0),
            HostValKind::I64 => HostVal::I64(0),
            HostValKind::F32 => HostVal::F32(0.0),
            HostValKind::F64 => HostVal::F64(0.0),
            HostValKind::V128 => HostVal::V128(0),
        }
    }

    fn from_wasm(v: &Val) -> Option<HostVal> {
        match v {
            Val::I32(n) => Some(HostVal::I32(*n)),
            Val::I64(n) => Some(HostVal::I64(*n)),
            Val::F32(b) => Some(HostVal::F32(f32::from_bits(*b))),
            Val::F64(b) => Some(HostVal::F64(f64::from_bits(*b))),
            Val::V128(v) => Some(HostVal::V128(v.as_u128())),
            _ => None,
        }
    }

    fn to_wasm(self) -> Val {
        match self {
            HostVal::I32(n) => Val::I32(n),
            HostVal::I64(n) => Val::I64(n),
            HostVal::F32(f) => Val::F32(f.to_bits()),
            HostVal::F64(f) => Val::F64(f.to_bits()),
            HostVal::V128(bits) => Val::V128(V128::from(bits)),
        }
    }
}

/// A synchronous host callback bound into a Wasm module's import namespace.
///
/// `args` carries the Wasm-side argument values in the order declared by
/// [`HostFunc::params`].  The callback writes its results into `results` whose
/// length and kinds match [`HostFunc::results`].  Returning `false` traps the
/// caller; the outer Wasm call surfaces this as [`StatorError::WasmError`].
pub type HostFuncCallback = Arc<dyn Fn(&[HostVal], &mut [HostVal]) -> bool + Send + Sync>;

/// A single host function import.
///
/// `module` and `name` form the fully-qualified Wasm import name (e.g. `env`
/// and `add`).  `params` and `results` declare the function's signature and
/// are matched against the module's expected import signature at instantiate
/// time: any mismatch fails instantiation.
pub struct HostFunc {
    /// Wasm import module name (e.g. `"env"`).
    pub module: String,
    /// Wasm import field name (e.g. `"add"`).
    pub name: String,
    /// Parameter kinds, in order.
    pub params: Vec<HostValKind>,
    /// Result kinds, in order.
    pub results: Vec<HostValKind>,
    /// Synchronous callback invoked when imported Wasm code calls this
    /// function.  Runs on the thread executing the calling Wasm.
    pub callback: HostFuncCallback,
}

/// A single host global import bound into a Wasm module's import namespace.
///
/// `module` and `name` form the fully-qualified Wasm import name (e.g. `env`
/// and `g`).  `value` carries the initial value and its declared
/// [`HostValKind`]; `mutable` determines whether the bound global is of type
/// `const` or `var` and is matched against the module's expected global type
/// at instantiate time (any mismatch fails instantiation).
///
/// Globals created from a [`HostGlobal`] are owned by the Wasm store and
/// outlive the embedder's view of the import; writes performed by the running
/// Wasm module are not propagated back to any host-side value.  Callers that
/// need to observe Wasm-side mutations must keep that responsibility outside
/// this slice.
pub struct HostGlobal {
    /// Wasm import module name (e.g. `"env"`).
    pub module: String,
    /// Wasm import field name (e.g. `"g"`).
    pub name: String,
    /// Initial value and declared kind.
    pub value: HostVal,
    /// Whether the import requires a mutable (`var`) global.
    pub mutable: bool,
}

/// A single host shared-memory import bound into a Wasm module's import
/// namespace.
///
/// `module` and `name` form the fully-qualified Wasm import name (e.g.
/// `env` and `mem`). `memory` is a [`SharedMemory`] whose declared type is
/// matched structurally by the Wasmtime linker against the module's
/// expected memory type at instantiate time (any mismatch fails
/// instantiation).
///
/// Sharing semantics: a [`SharedMemory`] is a thread-safe linear memory
/// that lives outside any single [`Store`], so the same `SharedMemory`
/// handle can be supplied as an import to multiple instances in different
/// stores. Writes performed by one importer are immediately observable by
/// every other importer (and by the dependency module that originally
/// allocated or owns it). This is the only Wasm memory primitive Stator
/// can safely route through the ES module graph today; unshared memories
/// cannot be passed across stores in Wasmtime, so non-shared memory
/// imports continue to fail closed.
pub struct HostMemory {
    /// Wasm import module name (e.g. `"env"`).
    pub module: String,
    /// Wasm import field name (e.g. `"mem"`).
    pub name: String,
    /// Externally-owned shared memory bound under `(module, name)`.
    pub memory: SharedMemory,
}

// ─────────────────────────────────────────────────────────────────────────────
// WasmInstance
// ─────────────────────────────────────────────────────────────────────────────

/// A live WebAssembly module instance.
///
/// Wraps [`wasmtime::Instance`] together with the [`wasmtime::Store`] it lives
/// in.  Every instance has its own linear memory and tables.
///
/// Use [`WasmInstance::call`] to invoke an exported function by name.
pub struct WasmInstance {
    store: Store<()>,
    inner: Instance,
}

impl WasmInstance {
    /// Instantiate a [`WasmModule`] with no imports.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if instantiation fails (e.g. the
    /// module requires imports that are not provided).
    pub fn new(engine: &WasmEngine, module: &WasmModule) -> StatorResult<Self> {
        Self::new_with_imports(engine, module, Vec::new())
    }

    /// Instantiate a [`WasmModule`] binding the given host-function imports.
    ///
    /// Equivalent to calling [`WasmInstance::new_with_extern_imports`] with
    /// no host global imports.  See that method for the import-binding
    /// contract and termination/polling guarantees.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if linker registration fails (e.g.
    /// duplicate import names) or if instantiation fails.
    pub fn new_with_imports(
        engine: &WasmEngine,
        module: &WasmModule,
        imports: Vec<HostFunc>,
    ) -> StatorResult<Self> {
        Self::new_with_extern_imports(engine, module, imports, Vec::new())
    }

    /// Instantiate a [`WasmModule`] binding the given host-function and
    /// host-global imports.
    ///
    /// Each [`HostFunc`] is registered in the [`wasmtime::Linker`] under its
    /// `(module, name)` pair using a [`FuncType`] derived from the declared
    /// `params` / `results`.  Each [`HostGlobal`] is materialised as a
    /// [`wasmtime::Global`] owned by the store and registered under its
    /// `(module, name)` pair with the declared `value` kind and `mutable`
    /// flag.  Instantiation fails if the module expects an import that is
    /// not provided, or if a provided import's declared signature/type does
    /// not match the one the module expects (Wasmtime performs the
    /// structural match).
    ///
    /// Host callbacks run synchronously on the thread that called
    /// [`WasmInstance::call`]: see the "Host imports" section above for the
    /// termination / polling contract.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if linker registration fails (e.g.
    /// duplicate import names, conflicting `(module, name)` entries) or if
    /// instantiation fails.
    pub fn new_with_extern_imports(
        engine: &WasmEngine,
        module: &WasmModule,
        funcs: Vec<HostFunc>,
        globals: Vec<HostGlobal>,
    ) -> StatorResult<Self> {
        Self::new_with_extern_imports_full(engine, module, funcs, globals, Vec::new())
    }

    /// Instantiate a [`WasmModule`] binding the given host-function,
    /// host-global, and host-memory imports.
    ///
    /// Behaves identically to [`WasmInstance::new_with_extern_imports`] for
    /// functions and globals, and additionally registers each
    /// [`HostMemory`] under its `(module, name)` pair using its
    /// [`SharedMemory`] handle. The Wasmtime linker then performs the
    /// structural import-type match (minimum/maximum pages, memory64,
    /// shared, page-size log2) against the module's expected memory type
    /// and fails instantiation on any mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if linker registration fails
    /// (e.g. duplicate import names, conflicting `(module, name)` entries)
    /// or if instantiation fails.
    pub fn new_with_extern_imports_full(
        engine: &WasmEngine,
        module: &WasmModule,
        funcs: Vec<HostFunc>,
        globals: Vec<HostGlobal>,
        memories: Vec<HostMemory>,
    ) -> StatorResult<Self> {
        let mut store: Store<()> = Store::new(engine.inner(), ());
        // Configure the store so that every epoch advance reaches a callback
        // which checks the current thread's Stator interrupt flag.  This lets
        // a process-wide epoch broadcast terminate the isolate that owns the
        // running script without trapping unrelated isolates that share the
        // same process-global Wasm engine registry.
        store.epoch_deadline_callback(|_| {
            if check_interrupt_flag() {
                return Ok(UpdateDeadline::Interrupt);
            }
            Ok(UpdateDeadline::Continue(1))
        });
        store.set_epoch_deadline(1);

        let mut linker: Linker<()> = Linker::new(engine.inner());
        for imp in funcs {
            let HostFunc {
                module: m_name,
                name: f_name,
                params,
                results,
                callback,
            } = imp;
            let ty = FuncType::new(
                engine.inner(),
                params.iter().map(|k| k.to_wasmtime()),
                results.iter().map(|k| k.to_wasmtime()),
            );
            let results_kinds = results.clone();
            linker
                .func_new(&m_name, &f_name, ty, move |_caller, args, out| {
                    // Pre-callback termination poll: short-circuit before
                    // re-entering embedder code.
                    if check_interrupt_flag() {
                        return Err(wasmtime::Error::msg(SCRIPT_TERMINATED_MESSAGE));
                    }
                    let host_args: Vec<HostVal> =
                        args.iter().filter_map(HostVal::from_wasm).collect();
                    if host_args.len() != args.len() {
                        return Err(wasmtime::Error::msg(
                            "unsupported Wasm value type at host import boundary",
                        ));
                    }
                    let mut host_results: Vec<HostVal> =
                        results_kinds.iter().map(|k| HostVal::zero_of(*k)).collect();
                    let ok = (callback)(&host_args, &mut host_results);
                    if !ok {
                        // Map a host-requested trap.  If the embedder triggered
                        // it via the termination flag, surface the canonical
                        // terminated message instead.
                        if check_interrupt_flag() {
                            return Err(wasmtime::Error::msg(SCRIPT_TERMINATED_MESSAGE));
                        }
                        return Err(wasmtime::Error::msg("host function trapped"));
                    }
                    // Post-callback termination poll.
                    if check_interrupt_flag() {
                        return Err(wasmtime::Error::msg(SCRIPT_TERMINATED_MESSAGE));
                    }
                    for (slot, hv) in out.iter_mut().zip(host_results.iter()) {
                        *slot = hv.to_wasm();
                    }
                    Ok(())
                })
                .map_err(|e| StatorError::WasmError(e.to_string()))?;
        }

        for HostGlobal {
            module: m_name,
            name: g_name,
            value,
            mutable,
        } in globals
        {
            let mutability = if mutable {
                Mutability::Var
            } else {
                Mutability::Const
            };
            let ty = GlobalType::new(value.kind().to_wasmtime(), mutability);
            let global = Global::new(&mut store, ty, value.to_wasm())
                .map_err(|e| StatorError::WasmError(e.to_string()))?;
            linker
                .define(&store, &m_name, &g_name, global)
                .map_err(|e| StatorError::WasmError(e.to_string()))?;
        }

        for HostMemory {
            module: m_name,
            name: mem_name,
            memory,
        } in memories
        {
            linker
                .define(&store, &m_name, &mem_name, memory)
                .map_err(|e| StatorError::WasmError(e.to_string()))?;
        }

        let instance = linker
            .instantiate(&mut store, module.inner())
            .map_err(|e| StatorError::WasmError(e.to_string()))?;
        Ok(Self {
            store,
            inner: instance,
        })
    }

    /// Call an exported function by name with the given arguments.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if:
    /// - No export with the given name exists, or the export is not a
    ///   function.
    /// - The call traps or returns an error.
    pub fn call(&mut self, name: &str, args: &[Val]) -> StatorResult<Vec<Val>> {
        // JS↔Wasm boundary termination poll.  If the embedder has requested
        // termination on this thread we short-circuit before entering Wasm
        // so the embedder observes the canonical "script execution
        // terminated" error rather than whatever value a partial Wasm call
        // might happen to return.
        if check_interrupt_flag() {
            return Err(StatorError::WasmError(
                SCRIPT_TERMINATED_MESSAGE.to_string(),
            ));
        }

        let func = self
            .inner
            .get_func(&mut self.store, name)
            .ok_or_else(|| StatorError::WasmError(format!("no exported function '{name}'")))?;

        let ty = func.ty(&self.store);
        let result_count = ty.results().len();
        let mut results = vec![Val::I32(0); result_count];

        func.call(&mut self.store, args, &mut results)
            .map_err(|e| {
                if check_interrupt_flag() {
                    return StatorError::WasmError(SCRIPT_TERMINATED_MESSAGE.to_string());
                }
                StatorError::WasmError(e.to_string())
            })?;

        Ok(results)
    }

    /// Return the names of all exports from this instance.
    ///
    /// Each element is the UTF-8 name of one export (function, memory, table,
    /// or global).
    pub fn export_names(&mut self) -> Vec<String> {
        self.inner
            .exports(&mut self.store)
            .map(|e| e.name().to_owned())
            .collect()
    }

    /// Look up a named shared-memory export on this instance.
    ///
    /// Returns `None` when the export does not exist or is not a shared
    /// memory. Stator only exposes shared memories across the module-graph
    /// import boundary because they are the only Wasm memory primitive
    /// safely shareable between independent [`Store`]s; callers that need
    /// to bind a memory import from a Wasm dependency module use this
    /// accessor to obtain the dependency-owned [`SharedMemory`] handle.
    pub fn exported_shared_memory(&mut self, name: &str) -> Option<SharedMemory> {
        self.inner
            .get_export(&mut self.store, name)?
            .into_shared_memory()
    }

    /// Read the current 128-bit value of a named `v128` global export on
    /// this instance.
    ///
    /// Returns `None` when the export does not exist, is not a global, or
    /// is not of type `v128`. The value is returned as raw bits in the
    /// little-endian-lane order used by [`wasmtime::V128::as_u128`].
    ///
    /// Stator's module-graph integration uses this accessor to copy an
    /// immutable `v128` global value from a Wasm dependency's store into a
    /// fresh per-importer-store [`Global`]. Because the source global is
    /// declared immutable, this single-shot snapshot is semantically
    /// equivalent to sharing the same global instance across stores.
    pub fn exported_v128_global(&mut self, name: &str) -> Option<u128> {
        let export = self.inner.get_export(&mut self.store, name)?;
        let global = export.into_global()?;
        if !matches!(global.ty(&self.store).content(), ValType::V128) {
            return None;
        }
        match global.get(&mut self.store) {
            Val::V128(v) => Some(v.as_u128()),
            _ => None,
        }
    }

    /// Call an exported function by name using [`JsValue`] arguments.
    ///
    /// This is a convenience wrapper around [`WasmInstance::call`] that
    /// converts the [`JsValue`] arguments to [`wasmtime::Val`] before the call
    /// and converts the first result back to a [`JsValue`] after.  For void
    /// functions (zero results) [`JsValue::Undefined`] is returned.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::WasmError`] if:
    /// - An argument cannot be converted to a Wasm value.
    /// - The named export does not exist or is not a function.
    /// - The call traps.
    /// - The first result cannot be converted to a [`JsValue`].
    pub fn call_with_js_values(&mut self, name: &str, args: &[JsValue]) -> StatorResult<JsValue> {
        let wasm_args: Vec<Val> = args
            .iter()
            .map(js_value_to_wasm_val)
            .collect::<StatorResult<Vec<_>>>()?;
        let results = self.call(name, &wasm_args)?;
        if results.is_empty() {
            Ok(JsValue::Undefined)
        } else {
            wasm_val_to_js_value(&results[0])
        }
    }

    /// Return a reference to the underlying [`wasmtime::Instance`].
    pub fn inner(&self) -> &Instance {
        &self.inner
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JS ↔ Wasm value conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a [`JsValue`] to a [`wasmtime::Val`].
///
/// | JavaScript value          | Wasm value              |
/// |---------------------------|-------------------------|
/// | `Smi(n)`                  | `Val::I32(n)`           |
/// | `HeapNumber(f)` (in i32)  | `Val::I32(f as i32)`    |
/// | `HeapNumber(f)`           | `Val::F64(f.to_bits())` |
/// | `Boolean(b)`              | `Val::I32(b as i32)`    |
/// | `Undefined` / `Null`      | `Val::I32(0)`           |
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] for value types that have no natural
/// Wasm representation (e.g. objects, functions).
pub fn js_value_to_wasm_val(value: &JsValue) -> StatorResult<Val> {
    match value {
        JsValue::Smi(n) => Ok(Val::I32(*n)),
        JsValue::HeapNumber(f) => {
            // If the double is an exact i32, prefer the integer representation.
            let as_i32 = *f as i32;
            if f64::from(as_i32) == *f {
                Ok(Val::I32(as_i32))
            } else {
                Ok(Val::F64(f.to_bits()))
            }
        }
        JsValue::Boolean(b) => Ok(Val::I32(i32::from(*b))),
        JsValue::Undefined | JsValue::Null => Ok(Val::I32(0)),
        other => Err(StatorError::WasmError(format!(
            "cannot convert JsValue::{:?} to a Wasm value",
            std::mem::discriminant(other)
        ))),
    }
}

/// Convert a [`wasmtime::Val`] to a [`JsValue`].
///
/// | Wasm value   | JavaScript value      |
/// |--------------|-----------------------|
/// | `I32(n)`     | `Smi(n)`              |
/// | `I64(n)`     | `HeapNumber(n as f64)`|
/// | `F32(bits)`  | `HeapNumber(f64)`     |
/// | `F64(bits)`  | `HeapNumber(f64)`     |
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] for Wasm value types that cannot be
/// represented as a [`JsValue`] (e.g. `externref`, `funcref`).
pub fn wasm_val_to_js_value(val: &Val) -> StatorResult<JsValue> {
    match val {
        Val::I32(n) => Ok(JsValue::Smi(*n)),
        Val::I64(n) => Ok(JsValue::HeapNumber(*n as f64)),
        Val::F32(bits) => Ok(JsValue::HeapNumber(f64::from(f32::from_bits(*bits)))),
        Val::F64(bits) => Ok(JsValue::HeapNumber(f64::from_bits(*bits))),
        other => Err(StatorError::WasmError(format!(
            "cannot convert Wasm value {:?} to a JsValue",
            other
        ))),
    }
}

/// Convert a [`HostVal`] to a [`JsValue`].
///
/// This is the typed-host-import counterpart to [`wasm_val_to_js_value`] used
/// when the embedder works in [`HostVal`] without ever materialising a
/// [`wasmtime::Val`].
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] for [`HostVal::V128`], which has no
/// JS-side representation. Stator's module-graph integration routes `v128`
/// values directly between Wasm stores without ever passing through
/// [`JsValue`], so this is fail-closed by construction.
pub fn host_val_to_js_value(val: HostVal) -> StatorResult<JsValue> {
    Ok(match val {
        HostVal::I32(n) => JsValue::Smi(n),
        HostVal::I64(n) => JsValue::HeapNumber(n as f64),
        HostVal::F32(f) => JsValue::HeapNumber(f64::from(f)),
        HostVal::F64(f) => JsValue::HeapNumber(f),
        HostVal::V128(_) => {
            return Err(StatorError::WasmError(
                "cannot convert Wasm v128 value to a JsValue".to_string(),
            ));
        }
    })
}

/// Convert a [`JsValue`] into a [`HostVal`] of the requested kind.
///
/// Numeric coercion follows the same rules as
/// [`js_value_to_wasm_val`]: booleans and `undefined` / `null` map to zero of
/// the requested kind, and numbers truncate / convert as required.
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] when the value cannot be represented as
/// the requested [`HostValKind`] (e.g. an object), and unconditionally for
/// [`HostValKind::V128`] because no [`JsValue`] can carry 128 bits without
/// loss; Wasm-to-Wasm v128 globals are bound through
/// [`WasmInstance::exported_v128_global`] instead of the JS namespace path.
pub fn js_value_to_host_val(value: &JsValue, kind: HostValKind) -> StatorResult<HostVal> {
    if matches!(kind, HostValKind::V128) {
        return Err(StatorError::WasmError(
            "cannot convert a JsValue to a Wasm v128 value".to_string(),
        ));
    }
    let n = match value {
        JsValue::Smi(n) => Some(f64::from(*n)),
        JsValue::HeapNumber(f) => Some(*f),
        JsValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
        JsValue::Undefined | JsValue::Null => Some(0.0),
        _ => None,
    };
    let Some(n) = n else {
        return Err(StatorError::WasmError(format!(
            "cannot convert JsValue::{:?} to a Wasm value of kind {:?}",
            std::mem::discriminant(value),
            kind
        )));
    };
    Ok(match kind {
        HostValKind::I32 => HostVal::I32(n as i32),
        HostValKind::I64 => HostVal::I64(n as i64),
        HostValKind::F32 => HostVal::F32(n as f32),
        HostValKind::F64 => HostVal::F64(n),
        HostValKind::V128 => unreachable!("guarded above"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WasmEngine ──────────────────────────────────────────────────────────

    #[test]
    fn test_wasm_engine_new() {
        let engine = WasmEngine::new();
        // Engine::default() produces a valid engine; just check it is accessible.
        let _ = engine.inner();
    }

    #[test]
    fn test_wasm_engine_default() {
        let _engine: WasmEngine = WasmEngine::default();
    }

    #[test]
    fn test_wasm_engine_clone() {
        let engine = WasmEngine::new();
        let _cloned = engine.clone();
    }

    // ── WasmModule ──────────────────────────────────────────────────────────

    const MINIMAL_WAT: &str = r#"(module)"#;

    const ADD_WAT: &str = r#"
        (module
            (func $add (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add))
    "#;

    const ADD_F64_WAT: &str = r#"
        (module
            (func $addf (export "addf") (param f64 f64) (result f64)
                local.get 0
                local.get 1
                f64.add))
    "#;

    const FACTORIAL_WAT: &str = r#"
        (module
            (func $factorial (export "factorial") (param i32) (result i32)
                (if (result i32) (i32.le_s (local.get 0) (i32.const 1))
                    (then (i32.const 1))
                    (else
                        (i32.mul
                            (local.get 0)
                            (call $factorial (i32.sub (local.get 0) (i32.const 1)))))))
        )
    "#;

    #[test]
    fn test_wasm_module_from_wat_minimal() {
        let engine = WasmEngine::new();
        WasmModule::from_wat(&engine, MINIMAL_WAT).expect("minimal WAT should compile");
    }

    #[test]
    fn test_wasm_module_from_wat_invalid() {
        let engine = WasmEngine::new();
        let err = WasmModule::from_wat(&engine, "not valid wat").unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    #[test]
    fn test_wasm_module_from_bytes_valid() {
        // WAT → binary bytes → WasmModule::from_bytes
        let engine = WasmEngine::new();
        // Build a minimal valid Wasm binary (empty module: magic + version)
        let bytes: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        WasmModule::from_bytes(&engine, bytes).expect("empty wasm binary should compile");
    }

    #[test]
    fn test_wasm_module_from_bytes_invalid() {
        let engine = WasmEngine::new();
        let err = WasmModule::from_bytes(&engine, b"not wasm").unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    #[test]
    fn test_wasm_module_clone() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, MINIMAL_WAT).unwrap();
        let _cloned = module.clone();
    }

    #[test]
    fn test_wasm_module_imports_report_global_type_metadata() {
        let engine = WasmEngine::new();
        let module =
            WasmModule::from_wat(&engine, r#"(module (import "env" "g" (global (mut i64))))"#)
                .unwrap();
        let imports = module.imports();
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "env");
        assert_eq!(imports[0].name, "g");
        match &imports[0].kind {
            WasmExternKind::Global {
                value_type,
                mutable,
            } => {
                assert_eq!(*value_type, Ok(HostValKind::I64));
                assert!(*mutable);
            }
            other => panic!("expected global import, got {other:?}"),
        }
    }

    #[test]
    fn test_wasm_module_imports_report_memory_type_metadata() {
        let engine = WasmEngine::new();
        let module =
            WasmModule::from_wat(&engine, r#"(module (import "env" "mem" (memory 2 5)))"#).unwrap();
        let imports = module.imports();
        assert_eq!(imports.len(), 1);
        match &imports[0].kind {
            WasmExternKind::Memory {
                minimum,
                maximum,
                memory64,
                shared,
                page_size_log2,
            } => {
                assert_eq!(*minimum, 2);
                assert_eq!(*maximum, Some(5));
                assert!(!*memory64);
                assert!(!*shared);
                assert_eq!(*page_size_log2, 16);
            }
            other => panic!("expected memory import, got {other:?}"),
        }
    }

    #[test]
    fn test_wasm_module_imports_report_table_type_metadata() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(
            &engine,
            r#"(module (import "env" "tab" (table 1 3 funcref)))"#,
        )
        .unwrap();
        let imports = module.imports();
        assert_eq!(imports.len(), 1);
        match &imports[0].kind {
            WasmExternKind::Table {
                element,
                minimum,
                maximum,
                table64,
            } => {
                assert!(element.contains("func"), "unexpected element: {element}");
                assert_eq!(*minimum, 1);
                assert_eq!(*maximum, Some(3));
                assert!(!*table64);
            }
            other => panic!("expected table import, got {other:?}"),
        }
    }

    // ── WasmInstance ────────────────────────────────────────────────────────

    #[test]
    fn test_wasm_instance_new() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, MINIMAL_WAT).unwrap();
        WasmInstance::new(&engine, &module)
            .expect("instantiation of minimal module should succeed");
    }

    #[test]
    fn test_wasm_instance_call_add() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, ADD_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let result = instance
            .call("add", &[Val::I32(3), Val::I32(4)])
            .expect("add(3, 4) should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].unwrap_i32(), 7);
    }

    #[test]
    fn test_wasm_instance_call_add_f64() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, ADD_F64_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let a = 1.5_f64;
        let b = 2.25_f64;
        let result = instance
            .call("addf", &[Val::F64(a.to_bits()), Val::F64(b.to_bits())])
            .expect("addf(1.5, 2.25) should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].unwrap_f64(), 3.75);
    }

    #[test]
    fn test_wasm_instance_call_factorial() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, FACTORIAL_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let result = instance
            .call("factorial", &[Val::I32(5)])
            .expect("factorial(5) should succeed");
        assert_eq!(result[0].unwrap_i32(), 120);
    }

    #[test]
    fn test_wasm_instance_call_missing_export() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, MINIMAL_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let err = instance.call("nonexistent", &[]).unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    // ── Termination at JS↔Wasm boundary ─────────────────────────────────────

    /// `WasmInstance::call` must observe the interpreter interrupt flag at
    /// entry and short-circuit with a terminated `WasmError`, even when the
    /// callee is a trivial constant function that would otherwise return
    /// immediately.
    #[test]
    fn test_wasm_call_entry_observes_interrupt_flag() {
        use crate::interpreter::{clear_interrupt_flag, set_interrupt_flag};
        use std::sync::atomic::{AtomicBool, Ordering};

        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, ADD_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        // Sanity: with the flag clear, the call succeeds normally.
        let ok = instance.call("add", &[Val::I32(1), Val::I32(2)]).unwrap();
        assert_eq!(ok[0].unwrap_i32(), 3);

        let flag = AtomicBool::new(true);
        // SAFETY: `flag` lives for the duration of the `set`/`clear` pair.
        unsafe { set_interrupt_flag(&flag as *const _) };

        let err = instance
            .call("add", &[Val::I32(1), Val::I32(2)])
            .expect_err("call must observe the interrupt flag at entry");
        match err {
            StatorError::WasmError(msg) => {
                assert_eq!(SCRIPT_TERMINATED_MESSAGE, msg);
            }
            other => panic!("expected WasmError, got: {other:?}"),
        }

        // After clearing, the call succeeds again so we know the boundary
        // observation is gated on the flag and not a sticky state.
        flag.store(false, Ordering::Relaxed);
        clear_interrupt_flag();
        let ok = instance.call("add", &[Val::I32(10), Val::I32(20)]).unwrap();
        assert_eq!(ok[0].unwrap_i32(), 30);
    }

    /// [`interrupt_all_wasm_engines`] must run without panicking even when
    /// no Wasm call is in flight (idempotency / safety contract).
    #[test]
    fn test_interrupt_all_wasm_engines_safe_when_idle() {
        // Ensure at least one engine is registered.
        let _engine = WasmEngine::new();
        interrupt_all_wasm_engines();
        interrupt_all_wasm_engines();
    }

    /// A process-global epoch tick must not trap an unrelated isolate.  The
    /// callback should continue execution unless this thread's Stator interrupt
    /// flag is set.
    #[test]
    fn test_wasm_epoch_bump_without_interrupt_flag_continues() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, ADD_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        engine.inner().increment_epoch();

        let ok = instance.call("add", &[Val::I32(4), Val::I32(5)]).unwrap();
        assert_eq!(ok[0].unwrap_i32(), 9);
    }

    /// After an epoch tick, a running Wasm call traps only when the Stator
    /// interrupt flag published on the executing thread is set.  We call
    /// Wasmtime directly here so this validates the epoch callback rather than
    /// the public `WasmInstance::call` entry check.
    #[test]
    fn test_wasm_call_mid_execution_traps_when_interrupt_flag_set() {
        use crate::interpreter::{clear_interrupt_flag, set_interrupt_flag};
        use std::sync::atomic::AtomicBool;

        // (module
        //   (func (export "spin")
        //     (loop $L (br $L))))
        let spin_wat = r#"
            (module
                (func (export "spin")
                    (loop $L (br $L))))
        "#;

        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, spin_wat).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let func = instance
            .inner
            .get_func(&mut instance.store, "spin")
            .expect("spin export should exist");
        let mut results = Vec::new();
        let flag = AtomicBool::new(true);

        // Pre-advance the engine's epoch so the store's deadline callback runs
        // at the first loop epoch check.  With the thread-local flag set, the
        // callback interrupts instead of continuing.
        engine.inner().increment_epoch();

        // SAFETY: `flag` lives for the duration of the `set`/`clear` pair.
        unsafe { set_interrupt_flag(&flag as *const _) };
        let err = func
            .call(&mut instance.store, &[], &mut results)
            .expect_err("spin must trap when the interrupt flag is set");
        clear_interrupt_flag();
        // Wasmtime surfaces the trap as a generic error string.  We only
        // require that the call errors out instead of hanging — the test
        // would time out if epoch interruption was not wired up.
        assert!(!err.to_string().is_empty());
    }

    /// Public Wasm calls that are interrupted after entry should map
    /// Wasmtime's generic interrupt trap back to Stator's canonical
    /// termination message.
    #[test]
    fn test_wasm_call_mid_execution_maps_interrupt_to_terminated_error() {
        use crate::interpreter::{clear_interrupt_flag, set_interrupt_flag};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        let spin_wat = r#"
            (module
                (func (export "spin")
                    (loop $L (br $L))))
        "#;

        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, spin_wat).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let trigger_flag = Arc::clone(&flag);
        let trigger = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            trigger_flag.store(true, Ordering::Relaxed);
            interrupt_all_wasm_engines();
        });

        // SAFETY: `flag` stays alive until after the call returns and the
        // thread-local association is cleared.
        unsafe { set_interrupt_flag(Arc::as_ptr(&flag)) };
        let err = instance
            .call("spin", &[])
            .expect_err("spin must terminate after the async epoch bump");
        clear_interrupt_flag();
        trigger.join().unwrap();

        match err {
            StatorError::WasmError(msg) => {
                assert_eq!(SCRIPT_TERMINATED_MESSAGE, msg);
            }
            other => panic!("expected WasmError, got: {other:?}"),
        }
    }

    // ── Host imports ────────────────────────────────────────────────────────

    /// Imported `env.add(i32, i32) -> i32` is bound and observable from a Wasm
    /// export that simply forwards its arguments to the import.
    #[test]
    fn test_wasm_host_import_add_i32() {
        let wat = r#"
            (module
                (import "env" "add" (func $add (param i32 i32) (result i32)))
                (func (export "call_add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    call $add))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();

        let cb: HostFuncCallback = Arc::new(|args, results| {
            let a = match args[0] {
                HostVal::I32(n) => n,
                _ => return false,
            };
            let b = match args[1] {
                HostVal::I32(n) => n,
                _ => return false,
            };
            results[0] = HostVal::I32(a.wrapping_add(b));
            true
        });
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "add".to_string(),
            params: vec![HostValKind::I32, HostValKind::I32],
            results: vec![HostValKind::I32],
            callback: cb,
        }];
        let mut instance = WasmInstance::new_with_imports(&engine, &module, imports).unwrap();
        let r = instance
            .call("call_add", &[Val::I32(7), Val::I32(35)])
            .unwrap();
        assert_eq!(r[0].unwrap_i32(), 42);
    }

    /// A host callback returning `false` traps the running Wasm call and the
    /// outer call surfaces a `WasmError`.
    #[test]
    fn test_wasm_host_import_callback_false_traps() {
        let wat = r#"
            (module
                (import "env" "bad" (func $bad (result i32)))
                (func (export "go") (result i32) (call $bad)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let cb: HostFuncCallback = Arc::new(|_args, _results| false);
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "bad".to_string(),
            params: vec![],
            results: vec![HostValKind::I32],
            callback: cb,
        }];
        let mut instance = WasmInstance::new_with_imports(&engine, &module, imports).unwrap();
        let err = instance.call("go", &[]).unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    /// Instantiation fails if a required import is not supplied.
    #[test]
    fn test_wasm_host_import_missing_fails_instantiate() {
        let wat = r#"
            (module
                (import "env" "missing" (func (result i32))))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let result = WasmInstance::new_with_imports(&engine, &module, vec![]);
        match result {
            Ok(_) => panic!("instantiation should fail when imports are missing"),
            Err(StatorError::WasmError(_)) => {}
            Err(other) => panic!("expected WasmError, got: {other:?}"),
        }
    }

    /// Instantiation fails if the supplied import is bound to the wrong module
    /// or field name (the linker reports the missing required import).
    #[test]
    fn test_wasm_host_import_bad_name_fails_instantiate() {
        let wat = r#"
            (module
                (import "env" "add" (func (param i32 i32) (result i32))))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let cb: HostFuncCallback = Arc::new(|_args, _results| true);
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "other".to_string(),
            params: vec![HostValKind::I32, HostValKind::I32],
            results: vec![HostValKind::I32],
            callback: cb,
        }];
        let result = WasmInstance::new_with_imports(&engine, &module, imports);
        match result {
            Ok(_) => panic!("instantiation should fail with mismatched import name"),
            Err(StatorError::WasmError(_)) => {}
            Err(other) => panic!("expected WasmError, got: {other:?}"),
        }
    }

    /// Instantiation fails when the supplied host signature does not match the
    /// module's import declaration.
    #[test]
    fn test_wasm_host_import_signature_mismatch_fails_instantiate() {
        let wat = r#"
            (module
                (import "env" "f" (func (param i32 i32) (result i32))))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let cb: HostFuncCallback = Arc::new(|_args, _results| true);
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "f".to_string(),
            // Wrong: declared as f64/f64 -> f64.
            params: vec![HostValKind::F64, HostValKind::F64],
            results: vec![HostValKind::F64],
            callback: cb,
        }];
        let result = WasmInstance::new_with_imports(&engine, &module, imports);
        match result {
            Ok(_) => panic!("instantiation should fail with mismatched signature"),
            Err(StatorError::WasmError(_)) => {}
            Err(other) => panic!("expected WasmError, got: {other:?}"),
        }
    }

    /// A termination request issued before any Wasm runs must short-circuit
    /// the next call with the canonical terminated message, even when the
    /// module declares host imports.
    #[test]
    fn test_wasm_host_import_termination_before_call_returns_terminated() {
        use crate::interpreter::{clear_interrupt_flag, set_interrupt_flag};
        use std::sync::atomic::AtomicBool;

        let wat = r#"
            (module
                (import "env" "add" (func $add (param i32 i32) (result i32)))
                (func (export "go") (result i32)
                    (call $add (i32.const 1) (i32.const 2))))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let cb: HostFuncCallback = Arc::new(|args, results| {
            if let (HostVal::I32(a), HostVal::I32(b)) = (args[0], args[1]) {
                results[0] = HostVal::I32(a + b);
                true
            } else {
                false
            }
        });
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "add".to_string(),
            params: vec![HostValKind::I32, HostValKind::I32],
            results: vec![HostValKind::I32],
            callback: cb,
        }];
        let mut instance = WasmInstance::new_with_imports(&engine, &module, imports).unwrap();

        let flag = AtomicBool::new(true);
        // SAFETY: `flag` outlives the set/clear pair.
        unsafe { set_interrupt_flag(&flag as *const _) };
        let err = instance.call("go", &[]).unwrap_err();
        clear_interrupt_flag();
        match err {
            StatorError::WasmError(msg) => assert_eq!(SCRIPT_TERMINATED_MESSAGE, msg),
            other => panic!("expected WasmError, got: {other:?}"),
        }
    }

    /// Host callbacks must run synchronously on the thread invoking the Wasm
    /// call, not on any worker thread.
    #[test]
    fn test_wasm_host_import_runs_on_calling_thread() {
        use std::sync::Mutex as StdMutex;
        use std::thread::ThreadId;

        let wat = r#"
            (module
                (import "env" "probe" (func $probe))
                (func (export "go") (call $probe)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let observed: Arc<StdMutex<Option<ThreadId>>> = Arc::new(StdMutex::new(None));
        let observed_cb = Arc::clone(&observed);
        let cb: HostFuncCallback = Arc::new(move |_args, _results| {
            *observed_cb.lock().unwrap() = Some(std::thread::current().id());
            true
        });
        let imports = vec![HostFunc {
            module: "env".to_string(),
            name: "probe".to_string(),
            params: vec![],
            results: vec![],
            callback: cb,
        }];
        let mut instance = WasmInstance::new_with_imports(&engine, &module, imports).unwrap();
        instance.call("go", &[]).unwrap();
        let seen = observed.lock().unwrap().expect("callback must have run");
        assert_eq!(seen, std::thread::current().id());
    }

    // ── Host globals ────────────────────────────────────────────────────────

    /// Importing an immutable `i32` global binds the supplied initial value
    /// and the imported module observes it via `global.get`.
    #[test]
    fn test_wasm_host_global_import_immutable_i32() {
        let wat = r#"
            (module
                (import "env" "g" (global i32))
                (func (export "get") (result i32) (global.get 0)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let globals = vec![HostGlobal {
            module: "env".to_string(),
            name: "g".to_string(),
            value: HostVal::I32(123),
            mutable: false,
        }];
        let mut instance =
            WasmInstance::new_with_extern_imports(&engine, &module, Vec::new(), globals).unwrap();
        let result = instance.call("get", &[]).unwrap();
        assert_eq!(result[0].unwrap_i32(), 123);
    }

    /// Importing a mutable `i64` global with a matching `(mut i64)` import
    /// declaration succeeds and the importer can both read and write it.
    #[test]
    fn test_wasm_host_global_import_mutable_i64_roundtrip() {
        let wat = r#"
            (module
                (import "env" "g" (global (mut i64)))
                (func (export "get") (result i64) (global.get 0))
                (func (export "set") (param i64) (local.get 0) (global.set 0)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let globals = vec![HostGlobal {
            module: "env".to_string(),
            name: "g".to_string(),
            value: HostVal::I64(7),
            mutable: true,
        }];
        let mut instance =
            WasmInstance::new_with_extern_imports(&engine, &module, Vec::new(), globals).unwrap();
        let initial = instance.call("get", &[]).unwrap();
        assert_eq!(initial[0].unwrap_i64(), 7);
        instance.call("set", &[Val::I64(42)]).unwrap();
        let updated = instance.call("get", &[]).unwrap();
        assert_eq!(updated[0].unwrap_i64(), 42);
    }

    /// Supplying a `const` host global for an import declared `mut` fails
    /// instantiation rather than silently coercing the mutability.
    #[test]
    fn test_wasm_host_global_import_mutability_mismatch_fails() {
        let wat = r#"
            (module (import "env" "g" (global (mut i32))))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let globals = vec![HostGlobal {
            module: "env".to_string(),
            name: "g".to_string(),
            value: HostVal::I32(0),
            mutable: false,
        }];
        let err = match WasmInstance::new_with_extern_imports(&engine, &module, Vec::new(), globals)
        {
            Ok(_) => panic!("instantiation should fail with mutability mismatch"),
            Err(e) => e,
        };
        assert!(matches!(err, StatorError::WasmError(_)), "{err:?}");
    }

    /// Supplying a host global of the wrong value kind fails instantiation.
    #[test]
    fn test_wasm_host_global_import_kind_mismatch_fails() {
        let wat = r#"
            (module (import "env" "g" (global i32)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let globals = vec![HostGlobal {
            module: "env".to_string(),
            name: "g".to_string(),
            value: HostVal::F64(1.5),
            mutable: false,
        }];
        let err = match WasmInstance::new_with_extern_imports(&engine, &module, Vec::new(), globals)
        {
            Ok(_) => panic!("instantiation should fail with kind mismatch"),
            Err(e) => e,
        };
        assert!(matches!(err, StatorError::WasmError(_)), "{err:?}");
    }

    /// `exported_global_type` returns the correct kind/mutability for known
    /// global exports and `None` for missing or non-global exports.
    #[test]
    fn test_wasm_module_exported_global_type_reports_metadata() {
        let wat = r#"
            (module
                (global (export "g") (mut f32) (f32.const 1.5))
                (global (export "h") i64 (i64.const 9))
                (global (export "v") v128 (v128.const i64x2 0x1 0x2))
                (func (export "f")))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        assert_eq!(
            module.exported_global_type("g"),
            Some((HostValKind::F32, true))
        );
        assert_eq!(
            module.exported_global_type("h"),
            Some((HostValKind::I64, false))
        );
        assert_eq!(
            module.exported_global_type("v"),
            Some((HostValKind::V128, false))
        );
        assert_eq!(module.exported_global_type("f"), None);
        assert_eq!(module.exported_global_type("missing"), None);
    }

    /// Snapshotting a Wasm instance's exported `v128` global returns the
    /// declared constant bits in the lane order documented on
    /// [`WasmInstance::exported_v128_global`]; non-v128 globals and
    /// missing names return `None`.
    #[test]
    fn test_wasm_instance_exported_v128_global_round_trips_bits() {
        let wat = r#"
            (module
                (global (export "v") v128
                    (v128.const i64x2 0x1122334455667788 0x99aabbccddeeff00))
                (global (export "n") i32 (i32.const 42))
                (func (export "f")))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();
        let expected: u128 = ((0x99aabbccddeeff00_u128) << 64) | (0x1122334455667788_u128);
        assert_eq!(instance.exported_v128_global("v"), Some(expected));
        assert_eq!(instance.exported_v128_global("n"), None);
        assert_eq!(instance.exported_v128_global("f"), None);
        assert_eq!(instance.exported_v128_global("missing"), None);
    }

    /// Binding an immutable v128 host global through the typed host-import
    /// path materialises a per-store global whose value matches the
    /// supplied [`HostVal::V128`] bits, observable by reading the global
    /// back from the freshly instantiated instance.
    #[test]
    fn test_wasm_host_global_import_v128_immutable_round_trip() {
        let wat = r#"
            (module
                (import "env" "g" (global v128))
                (global (export "out") v128 (global.get 0)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let bits: u128 = 0xdeadbeef_cafef00d_0123456789abcdef;
        let globals = vec![HostGlobal {
            module: "env".to_string(),
            name: "g".to_string(),
            value: HostVal::V128(bits),
            mutable: false,
        }];
        let mut instance =
            WasmInstance::new_with_extern_imports(&engine, &module, Vec::new(), globals).unwrap();
        assert_eq!(instance.exported_v128_global("out"), Some(bits));
    }

    // ── Host shared-memory imports ──────────────────────────────────────────

    /// Importing a shared memory binds the supplied [`SharedMemory`] handle;
    /// writes performed by the importer Wasm are observable through the
    /// same handle from outside the instance because both observe the same
    /// underlying linear memory.
    #[test]
    fn test_wasm_host_shared_memory_import_roundtrips_through_handle() {
        let wat = r#"
            (module
                (import "env" "mem" (memory 1 1 shared))
                (func (export "store") (param i32 i32)
                    (local.get 0) (local.get 1) (i32.store)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let memory = SharedMemory::new(engine.inner(), MemoryType::shared(1, 1)).unwrap();
        let memories = vec![HostMemory {
            module: "env".to_string(),
            name: "mem".to_string(),
            memory: memory.clone(),
        }];
        let mut instance = WasmInstance::new_with_extern_imports_full(
            &engine,
            &module,
            Vec::new(),
            Vec::new(),
            memories,
        )
        .unwrap();
        instance
            .call("store", &[Val::I32(0), Val::I32(0x12345678)])
            .unwrap();
        // SAFETY: SharedMemory::data returns a slice of UnsafeCell<u8>; we
        // only read the bytes we just wrote and never alias them mutably.
        let observed = unsafe {
            let slice = memory.data();
            let mut bytes = [0u8; 4];
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = *slice[i].get();
            }
            u32::from_le_bytes(bytes)
        };
        assert_eq!(observed, 0x12345678);
    }

    /// Supplying a shared memory whose declared maximum exceeds the
    /// importer's expected maximum fails instantiation rather than silently
    /// substituting a wider memory.
    #[test]
    fn test_wasm_host_shared_memory_import_max_mismatch_fails() {
        let wat = r#"
            (module (import "env" "mem" (memory 1 1 shared)))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let memory = SharedMemory::new(engine.inner(), MemoryType::shared(1, 4)).unwrap();
        let memories = vec![HostMemory {
            module: "env".to_string(),
            name: "mem".to_string(),
            memory,
        }];
        let err = match WasmInstance::new_with_extern_imports_full(
            &engine,
            &module,
            Vec::new(),
            Vec::new(),
            memories,
        ) {
            Ok(_) => panic!("instantiation should fail with memory max mismatch"),
            Err(e) => e,
        };
        assert!(matches!(err, StatorError::WasmError(_)), "{err:?}");
    }

    /// `exported_memory_type` returns structural type information for
    /// declared memory exports and `None` for missing or non-memory exports.
    #[test]
    fn test_wasm_module_exported_memory_type_reports_metadata() {
        let wat = r#"
            (module
                (memory (export "m") 2 5 shared)
                (func (export "f")))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let info = module
            .exported_memory_type("m")
            .expect("memory export must be reported");
        assert_eq!(info.minimum, 2);
        assert_eq!(info.maximum, Some(5));
        assert!(!info.memory64);
        assert!(info.shared);
        assert_eq!(info.page_size_log2, 16);
        assert_eq!(module.exported_memory_type("f"), None);
        assert_eq!(module.exported_memory_type("missing"), None);
    }

    /// `exported_table_type` returns structural type information for
    /// declared table exports and `None` for missing or non-table exports.
    /// The accessor is in place so that a future Wasm-to-Wasm table
    /// import implementation can validate dependency/importer table types
    /// structurally; today such imports still fail closed because
    /// Wasmtime tables are [`Store`]-bound and there is no `SharedTable`
    /// primitive that can route a table across independent stores.
    #[test]
    fn test_wasm_module_exported_table_type_reports_metadata() {
        let wat = r#"
            (module
                (table (export "tab") 2 5 funcref)
                (func (export "f")))
        "#;
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, wat).unwrap();
        let info = module
            .exported_table_type("tab")
            .expect("table export must be reported");
        assert!(
            info.element.contains("func"),
            "unexpected element: {}",
            info.element
        );
        assert_eq!(info.minimum, 2);
        assert_eq!(info.maximum, Some(5));
        assert!(!info.table64);
        assert_eq!(module.exported_table_type("f"), None);
        assert_eq!(module.exported_table_type("missing"), None);
    }

    /// A shared memory exported by one Wasm instance can be observed
    /// through `exported_shared_memory` and used as the import for a
    /// second, independently-instantiated Wasm module. Writes by either
    /// importer are visible to the dependency instance through the same
    /// underlying memory.
    #[test]
    fn test_wasm_shared_memory_routes_across_two_instances() {
        let dep_wat = r#"
            (module (memory (export "mem") 1 1 shared))
        "#;
        let importer_wat = r#"
            (module
                (import "env" "mem" (memory 1 1 shared))
                (func (export "store") (param i32 i32)
                    (local.get 0) (local.get 1) (i32.store))
                (func (export "load") (param i32) (result i32)
                    (local.get 0) (i32.load)))
        "#;
        let engine = WasmEngine::new();
        let dep_module = WasmModule::from_wat(&engine, dep_wat).unwrap();
        let importer_module = WasmModule::from_wat(&engine, importer_wat).unwrap();

        let mut dep_instance = WasmInstance::new(&engine, &dep_module).unwrap();
        let dep_mem = dep_instance
            .exported_shared_memory("mem")
            .expect("dep must export shared memory");

        let mut importer_instance = WasmInstance::new_with_extern_imports_full(
            &engine,
            &importer_module,
            Vec::new(),
            Vec::new(),
            vec![HostMemory {
                module: "env".to_string(),
                name: "mem".to_string(),
                memory: dep_mem,
            }],
        )
        .unwrap();

        importer_instance
            .call("store", &[Val::I32(16), Val::I32(0xfeedface_u32 as i32)])
            .unwrap();
        let read_back = importer_instance.call("load", &[Val::I32(16)]).unwrap();
        assert_eq!(read_back[0].unwrap_i32() as u32, 0xfeedface);

        // The dep also sees the same memory: its exported handle observes
        // the importer's write.
        let dep_view = dep_instance.exported_shared_memory("mem").unwrap();
        // SAFETY: bytes 16..20 were just written by the importer; we only
        // read them here and never alias mutably.
        let observed = unsafe {
            let slice = dep_view.data();
            let mut bytes = [0u8; 4];
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = *slice[16 + i].get();
            }
            u32::from_le_bytes(bytes)
        };
        assert_eq!(observed, 0xfeedface);
    }

    // ── Value conversion ────────────────────────────────────────────────────

    #[test]
    fn test_js_value_to_wasm_val_smi() {
        let v = js_value_to_wasm_val(&JsValue::Smi(42)).unwrap();
        assert_eq!(v.unwrap_i32(), 42);
    }

    #[test]
    fn test_js_value_to_wasm_val_heap_number_exact_i32() {
        let v = js_value_to_wasm_val(&JsValue::HeapNumber(7.0)).unwrap();
        assert_eq!(v.unwrap_i32(), 7);
    }

    #[test]
    fn test_js_value_to_wasm_val_heap_number_f64() {
        let v = js_value_to_wasm_val(&JsValue::HeapNumber(3.14)).unwrap();
        // Should produce F64
        assert_eq!(v.unwrap_f64(), 3.14);
    }

    #[test]
    fn test_js_value_to_wasm_val_boolean_true() {
        let v = js_value_to_wasm_val(&JsValue::Boolean(true)).unwrap();
        assert_eq!(v.unwrap_i32(), 1);
    }

    #[test]
    fn test_js_value_to_wasm_val_boolean_false() {
        let v = js_value_to_wasm_val(&JsValue::Boolean(false)).unwrap();
        assert_eq!(v.unwrap_i32(), 0);
    }

    #[test]
    fn test_js_value_to_wasm_val_undefined() {
        let v = js_value_to_wasm_val(&JsValue::Undefined).unwrap();
        assert_eq!(v.unwrap_i32(), 0);
    }

    #[test]
    fn test_js_value_to_wasm_val_null() {
        let v = js_value_to_wasm_val(&JsValue::Null).unwrap();
        assert_eq!(v.unwrap_i32(), 0);
    }

    #[test]
    fn test_js_value_to_wasm_val_unsupported() {
        let err = js_value_to_wasm_val(&JsValue::String("hello".into())).unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    #[test]
    fn test_wasm_val_to_js_value_i32() {
        let v = wasm_val_to_js_value(&Val::I32(10)).unwrap();
        assert_eq!(v, JsValue::Smi(10));
    }

    #[test]
    fn test_wasm_val_to_js_value_i64() {
        let v = wasm_val_to_js_value(&Val::I64(1_000_000_000_000_i64)).unwrap();
        assert_eq!(v, JsValue::HeapNumber(1_000_000_000_000_f64));
    }

    #[test]
    fn test_wasm_val_to_js_value_f32() {
        let bits = 1.5_f32.to_bits();
        let v = wasm_val_to_js_value(&Val::F32(bits)).unwrap();
        if let JsValue::HeapNumber(f) = v {
            assert!((f - 1.5).abs() < 1e-6);
        } else {
            panic!("expected HeapNumber");
        }
    }

    #[test]
    fn test_wasm_val_to_js_value_f64() {
        let bits = 2.718_f64.to_bits();
        let v = wasm_val_to_js_value(&Val::F64(bits)).unwrap();
        assert_eq!(v, JsValue::HeapNumber(2.718));
    }

    // ── End-to-end: use JS value conversion to call a Wasm function ─────────

    #[test]
    fn test_end_to_end_js_value_add() {
        let engine = WasmEngine::new();
        let module = WasmModule::from_wat(&engine, ADD_WAT).unwrap();
        let mut instance = WasmInstance::new(&engine, &module).unwrap();

        let a = js_value_to_wasm_val(&JsValue::Smi(10)).unwrap();
        let b = js_value_to_wasm_val(&JsValue::Smi(32)).unwrap();
        let wasm_results = instance.call("add", &[a, b]).unwrap();

        let js_result = wasm_val_to_js_value(&wasm_results[0]).unwrap();
        assert_eq!(js_result, JsValue::Smi(42));
    }
}

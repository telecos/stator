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
    Config, Engine, FuncType, Instance, Linker, Module, Store, UpdateDeadline, Val, ValType,
};

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
        // `Engine::new` only fails for invalid `Config`s; the defaults plus
        // `epoch_interruption(true)` are always valid.
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
}

impl HostValKind {
    fn to_wasmtime(self) -> ValType {
        match self {
            HostValKind::I32 => ValType::I32,
            HostValKind::I64 => ValType::I64,
            HostValKind::F32 => ValType::F32,
            HostValKind::F64 => ValType::F64,
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
}

impl HostVal {
    /// Return the [`HostValKind`] discriminant of this value.
    pub fn kind(&self) -> HostValKind {
        match self {
            HostVal::I32(_) => HostValKind::I32,
            HostVal::I64(_) => HostValKind::I64,
            HostVal::F32(_) => HostValKind::F32,
            HostVal::F64(_) => HostValKind::F64,
        }
    }

    /// Return the zero value of the given kind.
    pub fn zero_of(kind: HostValKind) -> HostVal {
        match kind {
            HostValKind::I32 => HostVal::I32(0),
            HostValKind::I64 => HostVal::I64(0),
            HostValKind::F32 => HostVal::F32(0.0),
            HostValKind::F64 => HostVal::F64(0.0),
        }
    }

    fn from_wasm(v: &Val) -> Option<HostVal> {
        match v {
            Val::I32(n) => Some(HostVal::I32(*n)),
            Val::I64(n) => Some(HostVal::I64(*n)),
            Val::F32(b) => Some(HostVal::F32(f32::from_bits(*b))),
            Val::F64(b) => Some(HostVal::F64(f64::from_bits(*b))),
            _ => None,
        }
    }

    fn to_wasm(self) -> Val {
        match self {
            HostVal::I32(n) => Val::I32(n),
            HostVal::I64(n) => Val::I64(n),
            HostVal::F32(f) => Val::F32(f.to_bits()),
            HostVal::F64(f) => Val::F64(f.to_bits()),
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
    /// Each [`HostFunc`] is registered in the [`wasmtime::Linker`] under its
    /// `(module, name)` pair using a [`FuncType`] derived from the declared
    /// `params` / `results`.  Instantiation fails if the module expects an
    /// import that is not provided, or if a provided import's declared
    /// signature does not match the one the module expects (Wasmtime performs
    /// the structural match).
    ///
    /// Host callbacks run synchronously on the thread that called
    /// [`WasmInstance::call`]: see the "Host imports" section above for the
    /// termination / polling contract.
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
        for imp in imports {
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

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
//! use stator_js::wasm::{WasmEngine, WasmInstance, WasmModule};
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

use wasmtime::{Engine, Instance, Linker, Module, Store, Val};

use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

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
    /// Create a new [`WasmEngine`] with default compilation settings.
    pub fn new() -> Self {
        Self {
            inner: Engine::default(),
        }
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
        let mut store: Store<()> = Store::new(engine.inner(), ());
        let linker: Linker<()> = Linker::new(engine.inner());
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
        let func = self
            .inner
            .get_func(&mut self.store, name)
            .ok_or_else(|| StatorError::WasmError(format!("no exported function '{name}'")))?;

        let ty = func.ty(&self.store);
        let result_count = ty.results().len();
        let mut results = vec![Val::I32(0); result_count];

        func.call(&mut self.store, args, &mut results)
            .map_err(|e| StatorError::WasmError(e.to_string()))?;

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

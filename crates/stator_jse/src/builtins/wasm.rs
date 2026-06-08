//! WebAssembly JS API built-in objects.
//!
//! This module implements the [WebAssembly JavaScript API] as a set of pure-Rust
//! functions that operate on the engine's [`JsValue`] type.  The public entry
//! point is [`make_webassembly_object`], which constructs the `WebAssembly`
//! namespace object that should be installed as a global.
//!
//! # Implemented API surface
//!
//! | JavaScript                               | Rust function            |
//! |------------------------------------------|--------------------------|
//! | `WebAssembly.validate(bytes)`            | [`wasm_validate`]        |
//! | `WebAssembly.compile(bytes)`             | [`wasm_compile`]         |
//! | `WebAssembly.instantiate(src, imports?)` | [`wasm_instantiate`]     |
//! | `new WebAssembly.Module(bytes)`          | [`wasm_module_ctor`]     |
//! | `new WebAssembly.Instance(mod, imp?)`    | [`wasm_instance_ctor`]   |
//!
//! # Fail-closed surface
//!
//! A few additional names from the modern WebAssembly JS API are exposed on
//! the `WebAssembly` namespace so feature-detection code finds them, but they
//! intentionally throw `TypeError` instead of returning fake objects, because
//! Stator does not implement the underlying semantics:
//!
//! | JavaScript                          | Reason                                  |
//! |-------------------------------------|-----------------------------------------|
//! | `WebAssembly.compileStreaming`      | Requires `Response`/`fetch` integration |
//! | `WebAssembly.instantiateStreaming`  | Requires `Response`/`fetch` integration |
//! | `new WebAssembly.Memory(...)`       | Wasm memory/ArrayBuffer bridge not implemented |
//! | `new WebAssembly.Table(...)`        | Wasm table/reference bridge not implemented |
//! | `new WebAssembly.Global(...)`       | Wasm global/import bridge not implemented |
//! | `new WebAssembly.Tag(...)`          | Wasm exception-handling not implemented |
//! | `new WebAssembly.Exception(...)`    | Wasm exception-handling not implemented |
//! | `new WebAssembly.CompileError(...)` | Real Error subclassing not available    |
//! | `new WebAssembly.LinkError(...)`    | Real Error subclassing not available    |
//! | `new WebAssembly.RuntimeError(...)` | Real Error subclassing not available    |
//!
//! These names exist as own properties of `WebAssembly` so `"compileStreaming"
//! in WebAssembly` is `true`, but invoking any of them throws
//! `TypeError: <name>: not implemented`.  No partial Error subclass hierarchy
//! is fabricated.
//!
//! # Byte input format
//!
//! Per the WebAssembly JS API, Wasm entry points accept a `BufferSource`
//! (ArrayBuffer or any ArrayBufferView).  Stator additionally accepts a few
//! convenience forms for environments that don't construct buffers directly:
//!
//! - **`JsValue::ArrayBuffer`** — bytes copied from `[0, byteLength)`.
//! - **`JsValue::TypedArray`** — bytes copied from the underlying buffer,
//!   honouring `byteOffset` and the (possibly auto-tracked) byte length.
//! - **`JsValue::DataView`** — bytes copied from the underlying buffer,
//!   honouring `byteOffset` and `byteLength`.
//! - **`JsValue::Array` of `Smi`/`HeapNumber` values (0–255)** — raw binary.
//! - **`JsValue::String`** — WebAssembly Text Format (WAT); compiled via
//!   Wasmtime's built-in WAT parser.
//!
//! Detached `ArrayBuffer`s (and views over them) raise a `TypeError`.
//!
//! # Object representation
//!
//! Wasm host objects are represented as [`JsValue::PlainObject`] values with a
//! discriminating `"__wasm_type__"` string property:
//!
//! | `__wasm_type__`          | additional properties                        |
//! |--------------------------|----------------------------------------------|
//! | `"WebAssembly.Module"`   | `exports` (Array of descriptors), `__wasm_bytes__` |
//! | `"WebAssembly.Instance"` | `exports` (PlainObject of callable exports)  |
//!
//! [WebAssembly JavaScript API]: https://webassembly.github.io/spec/js-api/

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::{StatorError, StatorResult};
use crate::objects::property_map::PropertyMap;
use crate::objects::value::{JsValue, NativeFn};
use crate::wasm::{
    WasmEngine, WasmInstance, WasmModule, js_value_to_wasm_val, wasm_val_to_js_value,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return `true` if `bytes` look like WAT (WebAssembly Text Format) rather
/// than a binary Wasm module.
///
/// Heuristic: a valid Wasm binary always starts with the magic bytes
/// `\0asm` (0x00 0x61 0x73 0x6d).  Everything else is treated as WAT.
fn is_wat(bytes: &[u8]) -> bool {
    !bytes.starts_with(b"\0asm")
}

/// Extract a `Vec<u8>` from a [`JsValue`] for use as Wasm source bytes.
///
/// Accepts the WebAssembly JS API `BufferSource` shapes plus a couple of
/// engine-specific conveniences:
///
/// - `JsValue::ArrayBuffer` → a copy of the buffer's bytes.
/// - `JsValue::TypedArray` → a copy of the bytes covered by the view
///   (`byteOffset .. byteOffset + byteLength`).
/// - `JsValue::DataView` → a copy of the bytes covered by the view
///   (`byteOffset .. byteOffset + byteLength`).
/// - `JsValue::Array` of `Smi`/`HeapNumber` → raw bytes (each value in 0–255).
/// - `JsValue::String` → UTF-8 encoding of the string (treated as WAT source).
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] if `val` is not one of the supported
/// shapes, if an array element is not a number in `[0, 255]`, or if an
/// `ArrayBuffer` (or the buffer backing a view) is detached.  Returns
/// [`StatorError::RangeError`] if a view's `byteOffset`/`byteLength` extends
/// past the end of its underlying buffer.
fn bytes_from_js_value(val: &JsValue) -> StatorResult<Vec<u8>> {
    match val {
        JsValue::ArrayBuffer(buf) => {
            let buf = buf.borrow();
            if buf.detached {
                return Err(StatorError::TypeError(
                    "WebAssembly source ArrayBuffer is detached".into(),
                ));
            }
            Ok(buf.data.clone())
        }
        JsValue::TypedArray(ta) => {
            let ta = ta.borrow();
            let buf = ta.buffer.borrow();
            if buf.detached {
                return Err(StatorError::TypeError(
                    "WebAssembly source TypedArray is backed by a detached ArrayBuffer".into(),
                ));
            }
            let byte_len = ta.effective_byte_length();
            let start = ta.byte_offset;
            let end = start
                .checked_add(byte_len)
                .ok_or_else(|| StatorError::RangeError("TypedArray range overflow".into()))?;
            if end > buf.data.len() {
                return Err(StatorError::RangeError(
                    "WebAssembly source TypedArray extends past end of ArrayBuffer".into(),
                ));
            }
            Ok(buf.data[start..end].to_vec())
        }
        JsValue::DataView(dv) => {
            let dv = dv.borrow();
            let buf = dv.buffer.borrow();
            if buf.detached {
                return Err(StatorError::TypeError(
                    "WebAssembly source DataView is backed by a detached ArrayBuffer".into(),
                ));
            }
            let start = dv.byte_offset;
            let end = start
                .checked_add(dv.byte_length)
                .ok_or_else(|| StatorError::RangeError("DataView range overflow".into()))?;
            if end > buf.data.len() {
                return Err(StatorError::RangeError(
                    "WebAssembly source DataView extends past end of ArrayBuffer".into(),
                ));
            }
            Ok(buf.data[start..end].to_vec())
        }
        JsValue::Array(items) => {
            let mut bytes = Vec::with_capacity(items.borrow().len());
            for (i, item) in items.borrow().iter().enumerate() {
                let n = match item {
                    JsValue::Smi(n) => *n,
                    JsValue::HeapNumber(f) => *f as i32,
                    other => {
                        return Err(StatorError::TypeError(format!(
                            "WebAssembly bytes[{i}]: expected number, got {other:?}"
                        )));
                    }
                };
                if !(0..=255).contains(&n) {
                    return Err(StatorError::TypeError(format!(
                        "WebAssembly bytes[{i}]: value {n} is out of range 0..=255"
                    )));
                }
                bytes.push(n as u8);
            }
            Ok(bytes)
        }
        JsValue::String(s) => Ok(s.as_bytes().to_vec()),
        other => Err(StatorError::TypeError(format!(
            "WebAssembly source must be an ArrayBuffer, ArrayBufferView, Array of bytes, or WAT string, got {other:?}"
        ))),
    }
}

/// Compile `bytes` to a [`WasmModule`], auto-detecting binary vs. WAT.
fn compile_bytes(engine: &WasmEngine, bytes: &[u8]) -> StatorResult<WasmModule> {
    if is_wat(bytes) {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| StatorError::WasmError(format!("invalid UTF-8 in WAT source: {e}")))?;
        WasmModule::from_wat(engine, text)
    } else {
        WasmModule::from_bytes(engine, bytes)
    }
}

/// Build a `WebAssembly.Module` [`JsValue::PlainObject`] from a compiled module.
///
/// The returned object has:
/// - `__wasm_type__` → `"WebAssembly.Module"`
/// - `__wasm_bytes__` → `Array` of `Smi` (byte values 0–255)
/// - `exports` → `Array` of export descriptor objects `{name, kind}`
fn make_module_object(module: &WasmModule, bytes: Vec<u8>) -> JsValue {
    let map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));

    // Tag the object type for later detection.
    map.borrow_mut().insert(
        "__wasm_type__".to_string(),
        JsValue::String("WebAssembly.Module".to_string().into()),
    );

    // Store the original bytes so we can re-compile when instantiating.
    let js_bytes: Vec<JsValue> = bytes.iter().map(|&b| JsValue::Smi(i32::from(b))).collect();
    map.borrow_mut()
        .insert("__wasm_bytes__".to_string(), JsValue::new_array(js_bytes));

    // Build the exports descriptor array (`WebAssembly.Module.exports(mod)`).
    let export_descs: Vec<JsValue> = module
        .inner()
        .exports()
        .map(|exp| {
            let desc: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
            desc.borrow_mut().insert(
                "name".to_string(),
                JsValue::String(exp.name().to_string().into()),
            );
            let kind = match exp.ty() {
                wasmtime::ExternType::Func(_) => "function",
                wasmtime::ExternType::Memory(_) => "memory",
                wasmtime::ExternType::Table(_) => "table",
                wasmtime::ExternType::Global(_) => "global",
                _ => "other",
            };
            desc.borrow_mut()
                .insert("kind".to_string(), JsValue::String(kind.to_string().into()));
            JsValue::PlainObject(desc)
        })
        .collect();
    map.borrow_mut()
        .insert("exports".to_string(), JsValue::new_array(export_descs));

    JsValue::PlainObject(map)
}

/// Extract the raw bytes stored inside a `WebAssembly.Module` object.
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] when `module_obj` is not a
/// `WebAssembly.Module` [`JsValue::PlainObject`].
fn extract_bytes_from_module(module_obj: &JsValue) -> StatorResult<Vec<u8>> {
    let map = match module_obj {
        JsValue::PlainObject(m) => m,
        _ => {
            return Err(StatorError::TypeError(
                "expected a WebAssembly.Module object".to_string(),
            ));
        }
    };

    let wasm_type = map.borrow().get("__wasm_type__").cloned();
    match &wasm_type {
        Some(JsValue::String(s)) if &**s == "WebAssembly.Module" => {}
        _ => {
            return Err(StatorError::TypeError(
                "not a WebAssembly.Module object".to_string(),
            ));
        }
    }

    match map.borrow().get("__wasm_bytes__").cloned() {
        Some(JsValue::Array(arr)) => arr
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                JsValue::Smi(n) => Ok(*n as u8),
                _ => Err(StatorError::WasmError(format!(
                    "__wasm_bytes__[{i}] is not a Smi"
                ))),
            })
            .collect(),
        _ => Err(StatorError::WasmError(
            "missing __wasm_bytes__ in WebAssembly.Module object".to_string(),
        )),
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn format_wasm_val_types(types: impl Iterator<Item = wasmtime::ValType>) -> String {
    let types: Vec<String> = types.map(|ty| ty.to_string()).collect();
    format!("[{}]", types.join(", "))
}

fn describe_unsupported_wasm_import_kind(kind: wasmtime::ExternType) -> String {
    match kind {
        wasmtime::ExternType::Func(func) => format!(
            "function import (params={}, results={})",
            format_wasm_val_types(func.params()),
            format_wasm_val_types(func.results())
        ),
        wasmtime::ExternType::Memory(memory) => format!(
            "memory import (min={}, max={}, memory64={}, shared={}, page_size_log2={})",
            memory.minimum(),
            format_optional_u64(memory.maximum()),
            memory.is_64(),
            memory.is_shared(),
            memory.page_size_log2()
        ),
        wasmtime::ExternType::Table(table) => format!(
            "table import (element={}, min={}, max={}, table64={})",
            table.element(),
            table.minimum(),
            format_optional_u64(table.maximum()),
            table.is_64()
        ),
        wasmtime::ExternType::Global(global) => format!(
            "global import (value_type={}, mutable={})",
            global.content(),
            global.mutability().is_var()
        ),
        _ => "extern import kind not represented by Stator".to_string(),
    }
}

/// Instantiate a compiled [`WasmModule`] and wrap the live instance in a
/// `WebAssembly.Instance` [`JsValue::PlainObject`].
///
/// The `exports` property of the returned object is a [`JsValue::PlainObject`]
/// where each exported *function* name is mapped to a
/// [`JsValue::NativeFunction`] that, when called, invokes that export on the
/// live Wasmtime instance.
///
/// Modules with imports or non-function exports (memories, tables, globals)
/// fail closed because the corresponding host bridges are not implemented.
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] for unsupported imports/non-function
/// exports, or [`StatorError::WasmError`] if Wasmtime instantiation fails.
fn make_instance_object(module: &WasmModule, engine: &WasmEngine) -> StatorResult<JsValue> {
    if let Some(import) = module.inner().imports().next() {
        return Err(StatorError::TypeError(format!(
            "WebAssembly.Instance: unsupported {} '{}.{}'; imports are not implemented",
            describe_unsupported_wasm_import_kind(import.ty()),
            import.module(),
            import.name()
        )));
    }
    if let Some(export) = module
        .inner()
        .exports()
        .find(|export| !matches!(export.ty(), wasmtime::ExternType::Func(_)))
    {
        return Err(StatorError::TypeError(format!(
            "WebAssembly.Instance: {} export '{}' is not implemented",
            match export.ty() {
                wasmtime::ExternType::Memory(_) => "memory",
                wasmtime::ExternType::Table(_) => "table",
                wasmtime::ExternType::Global(_) => "global",
                _ => "non-function",
            },
            export.name()
        )));
    }

    // Collect (name, is_func) pairs *before* consuming `module` in
    // WasmInstance::new, so that the borrow checker is happy.
    let export_info: Vec<(String, bool)> = module
        .inner()
        .exports()
        .map(|exp| {
            let is_func = matches!(exp.ty(), wasmtime::ExternType::Func(_));
            (exp.name().to_string(), is_func)
        })
        .collect();

    let instance_rc: Rc<RefCell<WasmInstance>> =
        Rc::new(RefCell::new(WasmInstance::new(engine, module)?));

    // Build the exports map with one NativeFunction per exported function.
    let exports_map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    for (name, is_func) in export_info {
        if is_func {
            let inst_ref = Rc::clone(&instance_rc);
            let fn_name = name.clone();
            let f: NativeFn = Rc::new(move |args: Vec<JsValue>| {
                let mut inst = inst_ref.borrow_mut();
                // Convert JS arguments to Wasm values.
                let wasm_args: Vec<wasmtime::Val> = args
                    .iter()
                    .map(js_value_to_wasm_val)
                    .collect::<StatorResult<Vec<_>>>()?;
                let results = inst.call(&fn_name, &wasm_args)?;
                if results.is_empty() {
                    Ok(JsValue::Undefined)
                } else {
                    wasm_val_to_js_value(&results[0])
                }
            });
            exports_map
                .borrow_mut()
                .insert(name, JsValue::NativeFunction(f));
        }
        // Non-function exports (memory, table, global) are not yet exposed.
    }

    let instance_map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
    instance_map.borrow_mut().insert(
        "__wasm_type__".to_string(),
        JsValue::String("WebAssembly.Instance".to_string().into()),
    );
    instance_map
        .borrow_mut()
        .insert("exports".to_string(), JsValue::PlainObject(exports_map));

    Ok(JsValue::PlainObject(instance_map))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API functions
// ─────────────────────────────────────────────────────────────────────────────

/// `WebAssembly.validate(bytes)` — ECMAScript WebAssembly API §7.1.
///
/// Returns `true` if `bytes` represent a valid WebAssembly module, `false`
/// otherwise.  Never throws.
///
/// `bytes` may be:
/// - a [`JsValue::Array`] of byte values (0–255), or
/// - a [`JsValue::String`] containing WAT source.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::wasm::wasm_validate;
/// use stator_jse::objects::value::JsValue;
///
/// // Minimal Wasm binary (magic + version).
/// let bytes: Vec<JsValue> = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
///     .into_iter()
///     .map(|b: u8| JsValue::Smi(b as i32))
///     .collect();
/// let result = wasm_validate(vec![JsValue::new_array(bytes)]).unwrap();
/// assert_eq!(result, JsValue::Boolean(true));
/// ```
pub fn wasm_validate(args: Vec<JsValue>) -> StatorResult<JsValue> {
    let source = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let bytes = match bytes_from_js_value(&source) {
        Ok(b) => b,
        Err(_) => return Ok(JsValue::Boolean(false)),
    };
    let engine = WasmEngine::new();
    let valid = compile_bytes(&engine, &bytes).is_ok();
    Ok(JsValue::Boolean(valid))
}

/// `WebAssembly.compile(bytes)` — ECMAScript WebAssembly API §7.2.
///
/// In the browser API this returns a `Promise<WebAssembly.Module>`; here it
/// compiles synchronously and returns the module object directly.
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] if the bytes are not a valid Wasm module.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::wasm::wasm_compile;
/// use stator_jse::objects::value::JsValue;
///
/// let wat = JsValue::String(r#"(module)"#.to_string().into());
/// let module = wasm_compile(vec![wat]).unwrap();
/// assert!(matches!(module, JsValue::PlainObject(_)));
/// ```
pub fn wasm_compile(args: Vec<JsValue>) -> StatorResult<JsValue> {
    let source = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let bytes = bytes_from_js_value(&source)?;
    let engine = WasmEngine::new();
    let module = compile_bytes(&engine, &bytes)?;
    Ok(make_module_object(&module, bytes))
}

/// `WebAssembly.instantiate(source, importObject?)` — ECMAScript WebAssembly API §7.3.
///
/// In the browser API this returns a `Promise`; here it executes synchronously.
///
/// - When `source` is a **byte array or WAT string**: compiles and instantiates,
///   returning a [`JsValue::PlainObject`] `{module, instance}`.
/// - When `source` is an **existing `WebAssembly.Module` object**: instantiates
///   directly and returns just the [`JsValue::PlainObject`] instance.
///
/// `importObject` is accepted but currently ignored.
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] on compilation or instantiation failure,
/// or [`StatorError::TypeError`] for invalid arguments.
pub fn wasm_instantiate(args: Vec<JsValue>) -> StatorResult<JsValue> {
    let mut iter = args.into_iter();
    let source = iter.next().unwrap_or(JsValue::Undefined);
    // importObject is accepted for spec-compliance but not yet implemented.
    let _import_object = iter.next();

    let engine = WasmEngine::new();

    // Determine whether `source` is already a compiled Module object.
    let is_module_obj = if let JsValue::PlainObject(ref m) = source {
        matches!(
            m.borrow().get("__wasm_type__").cloned(),
            Some(JsValue::String(ref s)) if &**s == "WebAssembly.Module"
        )
    } else {
        false
    };

    if is_module_obj {
        // source is a Module → return just the Instance.
        let bytes = extract_bytes_from_module(&source)?;
        let module = compile_bytes(&engine, &bytes)?;
        make_instance_object(&module, &engine)
    } else {
        // source is bytes → return {module, instance}.
        let bytes = bytes_from_js_value(&source)?;
        let module = compile_bytes(&engine, &bytes)?;
        let module_obj = make_module_object(&module, bytes);
        let instance_obj = make_instance_object(&module, &engine)?;

        let result: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
        result.borrow_mut().insert("module".to_string(), module_obj);
        result
            .borrow_mut()
            .insert("instance".to_string(), instance_obj);
        Ok(JsValue::PlainObject(result))
    }
}

/// `new WebAssembly.Module(bytes)` — ECMAScript WebAssembly API §5.
///
/// Compiles `bytes` synchronously and returns a `WebAssembly.Module` object.
///
/// # Errors
///
/// Returns [`StatorError::WasmError`] on compilation failure or
/// [`StatorError::TypeError`] for invalid input.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::wasm::wasm_module_ctor;
/// use stator_jse::objects::value::JsValue;
///
/// let wat = JsValue::String("(module)".to_string().into());
/// let m = wasm_module_ctor(vec![wat]).unwrap();
/// assert!(matches!(m, JsValue::PlainObject(_)));
/// ```
pub fn wasm_module_ctor(args: Vec<JsValue>) -> StatorResult<JsValue> {
    let source = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let bytes = bytes_from_js_value(&source)?;
    let engine = WasmEngine::new();
    let module = compile_bytes(&engine, &bytes)?;
    Ok(make_module_object(&module, bytes))
}

/// `new WebAssembly.Instance(module, importObject?)` — ECMAScript WebAssembly API §6.
///
/// Instantiates an existing `WebAssembly.Module` object.
///
/// `importObject` is accepted for spec-compliance but not yet implemented.
///
/// # Errors
///
/// Returns [`StatorError::TypeError`] if `module` is not a `WebAssembly.Module`
/// object, or [`StatorError::WasmError`] on instantiation failure.
pub fn wasm_instance_ctor(args: Vec<JsValue>) -> StatorResult<JsValue> {
    let mut iter = args.into_iter();
    let module_obj = iter.next().unwrap_or(JsValue::Undefined);
    let _import_object = iter.next();

    let bytes = extract_bytes_from_module(&module_obj)?;
    let engine = WasmEngine::new();
    let module = compile_bytes(&engine, &bytes)?;
    make_instance_object(&module, &engine)
}

/// `new WebAssembly.Memory(...)` — fail closed until the memory/ArrayBuffer
/// bridge is implemented.
pub fn wasm_memory_ctor(_args: Vec<JsValue>) -> StatorResult<JsValue> {
    Err(wasm_unsupported_error("Memory"))
}

/// `new WebAssembly.Table(...)` — fail closed until the table/reference bridge
/// is implemented.
pub fn wasm_table_ctor(_args: Vec<JsValue>) -> StatorResult<JsValue> {
    Err(wasm_unsupported_error("Table"))
}

/// `new WebAssembly.Global(...)` — fail closed until the global/import bridge
/// is implemented.
pub fn wasm_global_ctor(_args: Vec<JsValue>) -> StatorResult<JsValue> {
    Err(wasm_unsupported_error("Global"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Namespace factory
// ─────────────────────────────────────────────────────────────────────────────

/// Build the `WebAssembly` global namespace object.
///
/// Returns a [`JsValue::PlainObject`] with the following properties, each a
/// [`JsValue::NativeFunction`]:
///
/// - `validate` → [`wasm_validate`]
/// - `compile` → [`wasm_compile`]
/// - `instantiate` → [`wasm_instantiate`]
/// - `Module` → [`wasm_module_ctor`]
/// - `Instance` → [`wasm_instance_ctor`]
/// - `Memory`/`Table`/`Global` → fail-closed until their host bridges exist
///
/// Install this value as the `"WebAssembly"` key in the interpreter's global
/// environment to expose the full WebAssembly JS API to executing scripts.
///
/// # Examples
///
/// ```
/// use stator_jse::builtins::wasm::make_webassembly_object;
/// use stator_jse::objects::value::JsValue;
///
/// let wasm_obj = make_webassembly_object();
/// assert!(matches!(wasm_obj, JsValue::PlainObject(_)));
/// ```
/// Build a fail-closed native function that always throws `TypeError`.
///
/// Used for entries on the `WebAssembly` namespace whose underlying semantics
/// (streaming compile/instantiate, wasm exception handling, real Error
/// subclassing) are not implemented.  The function is exposed so
/// feature-detection succeeds, but calling or constructing it never returns a
/// plausible-looking fake object.
fn make_wasm_unsupported(name: &'static str) -> JsValue {
    let f: NativeFn = Rc::new(move |_args: Vec<JsValue>| Err(wasm_unsupported_error(name)));
    JsValue::NativeFunction(f)
}

fn wasm_unsupported_error(name: &str) -> StatorError {
    StatorError::TypeError(format!("WebAssembly.{name}: not implemented"))
}

/// Build the `WebAssembly` namespace object.
///
/// Real implementations are wired for `validate`, `compile`, `instantiate`,
/// and function-only `Module`/`Instance` use. The remaining modern-browser and
/// bridge-dependent surface is exposed as fail-closed entries that throw
/// `TypeError` because Stator does not implement the underlying semantics; see
/// the module docs.
pub fn make_webassembly_object() -> JsValue {
    let map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));

    {
        let mut m = map.borrow_mut();
        m.insert(
            "validate".to_string(),
            JsValue::NativeFunction(Rc::new(wasm_validate)),
        );
        m.insert(
            "compile".to_string(),
            JsValue::NativeFunction(Rc::new(wasm_compile)),
        );
        m.insert(
            "instantiate".to_string(),
            JsValue::NativeFunction(Rc::new(wasm_instantiate)),
        );
        m.insert(
            "Module".to_string(),
            JsValue::NativeFunction(Rc::new(wasm_module_ctor)),
        );
        m.insert(
            "Instance".to_string(),
            JsValue::NativeFunction(Rc::new(wasm_instance_ctor)),
        );
        // Fail-closed entries: see module-level docs.
        for name in [
            "Memory",
            "Table",
            "Global",
            "compileStreaming",
            "instantiateStreaming",
            "Tag",
            "Exception",
            "CompileError",
            "LinkError",
            "RuntimeError",
        ] {
            m.insert(name.to_string(), make_wasm_unsupported(name));
        }
    }

    JsValue::PlainObject(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::objects::property_map::PropertyMap;

    // ── WAT helpers ──────────────────────────────────────────────────────────

    /// Minimal empty module in WAT.
    const EMPTY_WAT: &str = "(module)";

    /// Module that exports a single `add(i32, i32) → i32` function.
    const ADD_WAT: &str = r#"
        (module
            (func $add (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add))
    "#;

    /// Module that exports a single `double(i32) → i32` function.
    const DOUBLE_WAT: &str = r#"
        (module
            (func $double (export "double") (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.mul))
    "#;

    /// Module with two exported functions.
    const MULTI_EXPORT_WAT: &str = r#"
        (module
            (func $noop (export "noop"))
            (func $identity (export "identity") (param i32) (result i32)
                local.get 0))
    "#;

    const IMPORTED_FUNC_WAT: &str = r#"
        (module
            (import "env" "f" (func $f))
            (func (export "call_import")
                call $f))
    "#;

    const IMPORTED_TABLE_WAT: &str = r#"
        (module
            (import "env" "tab" (table 1 3 funcref)))
    "#;

    const IMPORTED_NON_SHARED_MEMORY_WAT: &str = r#"
        (module
            (import "env" "mem" (memory 1 2)))
    "#;

    const IMPORTED_SHARED_MEMORY_WAT: &str = r#"
        (module
            (import "env" "mem" (memory 1 1 shared)))
    "#;

    const IMPORTED_MUTABLE_GLOBAL_WAT: &str = r#"
        (module
            (import "env" "g" (global (mut i32))))
    "#;

    const IMPORTED_V128_GLOBAL_WAT: &str = r#"
        (module
            (import "env" "g" (global v128)))
    "#;

    const IMPORTED_MUTABLE_V128_GLOBAL_WAT: &str = r#"
        (module
            (import "env" "g" (global (mut v128))))
    "#;

    const IMPORTED_EXTERNREF_GLOBAL_WAT: &str = r#"
        (module
            (import "env" "g" (global externref)))
    "#;

    const IMPORTED_MEMORY64_WAT_CANDIDATES: &[&str] = &[
        r#"(module (import "env" "mem64" (memory i64 1 2)))"#,
        r#"(module (import "env" "mem64" (memory (i64) 1 2)))"#,
        r#"(module (import "env" "mem64" (memory64 1 2)))"#,
    ];

    const IMPORTED_CUSTOM_PAGE_MEMORY_WAT_CANDIDATES: &[&str] = &[
        r#"(module (import "env" "mem" (memory (page_size_log2 12) 1 2)))"#,
        r#"(module (import "env" "mem" (memory (pagesize 4096) 1 2)))"#,
        r#"(module (import "env" "mem" (memory 1 2 (page_size_log2 12))))"#,
    ];

    const EXPORTED_MEMORY_WAT: &str = r#"
        (module
            (memory (export "memory") 1))
    "#;

    /// Helper: convert a WAT string to a `JsValue::String`.
    fn wat_val(wat: &str) -> JsValue {
        JsValue::String(wat.to_string().into())
    }

    /// Helper: minimal valid Wasm binary (empty module magic + version).
    fn empty_wasm_binary_val() -> JsValue {
        let bytes: Vec<JsValue> = [0x00u8, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
            .iter()
            .map(|&b| JsValue::Smi(i32::from(b)))
            .collect();
        JsValue::new_array(bytes)
    }

    /// Helper: plain descriptor object `{key: value}`.
    fn descriptor(pairs: &[(&str, JsValue)]) -> JsValue {
        let map: Rc<RefCell<PropertyMap>> = Rc::new(RefCell::new(PropertyMap::new()));
        for (k, v) in pairs {
            map.borrow_mut().insert(k.to_string(), v.clone());
        }
        JsValue::PlainObject(map)
    }

    fn assert_type_error_contains(err: StatorError, expected_terms: &[&str]) {
        match err {
            StatorError::TypeError(msg) => {
                for term in expected_terms {
                    assert!(msg.contains(term), "missing '{term}' in: {msg}");
                }
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    fn assert_instance_import_wat_fails_closed(wat: &str, expected_terms: &[&str]) {
        let module = wasm_module_ctor(vec![wat_val(wat)]).unwrap();
        let err = wasm_instance_ctor(vec![module, descriptor(&[])]).unwrap_err();
        assert_type_error_contains(err, expected_terms);
    }

    fn assert_optional_instance_import_wat_fails_closed(
        candidates: &[&str],
        expected_terms: &[&str],
    ) {
        for wat in candidates {
            let Ok(module) = wasm_module_ctor(vec![wat_val(wat)]) else {
                continue;
            };
            let err = wasm_instance_ctor(vec![module, descriptor(&[])]).unwrap_err();
            assert_type_error_contains(err, expected_terms);
            return;
        }
    }

    // ── bytes_from_js_value ───────────────────────────────────────────────────

    #[test]
    fn test_bytes_from_array_of_smis() {
        let arr = JsValue::new_array(vec![JsValue::Smi(0), JsValue::Smi(255)]);
        let bytes = bytes_from_js_value(&arr).unwrap();
        assert_eq!(bytes, vec![0u8, 255u8]);
    }

    #[test]
    fn test_bytes_from_heap_number() {
        let arr = JsValue::new_array(vec![JsValue::HeapNumber(1.0)]);
        let bytes = bytes_from_js_value(&arr).unwrap();
        assert_eq!(bytes, vec![1u8]);
    }

    #[test]
    fn test_bytes_from_string() {
        let s = JsValue::String("(module)".to_string().into());
        let bytes = bytes_from_js_value(&s).unwrap();
        assert_eq!(bytes, b"(module)");
    }

    #[test]
    fn test_bytes_from_invalid_type_returns_error() {
        let err = bytes_from_js_value(&JsValue::Smi(42)).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_bytes_out_of_range_returns_error() {
        let arr = JsValue::new_array(vec![JsValue::Smi(256)]);
        let err = bytes_from_js_value(&arr).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── bytes_from_js_value: ArrayBuffer / TypedArray / DataView ──────────────

    fn make_buffer(bytes: &[u8]) -> Rc<RefCell<crate::builtins::typed_array::JsArrayBuffer>> {
        let mut buf = crate::builtins::typed_array::arraybuffer_new(bytes.len());
        buf.data.copy_from_slice(bytes);
        Rc::new(RefCell::new(buf))
    }

    #[test]
    fn test_bytes_from_arraybuffer() {
        let buf = make_buffer(&[0x00, 0x61, 0x73, 0x6d]);
        let val = JsValue::ArrayBuffer(buf);
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![0x00, 0x61, 0x73, 0x6d]);
    }

    #[test]
    fn test_bytes_from_detached_arraybuffer_errors() {
        let buf = make_buffer(&[1, 2, 3]);
        buf.borrow_mut().detached = true;
        let err = bytes_from_js_value(&JsValue::ArrayBuffer(buf)).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_bytes_from_typed_array_full_view() {
        let buf = make_buffer(&[10, 20, 30, 40]);
        let ta = crate::builtins::typed_array::JsTypedArray {
            buffer: buf,
            kind: crate::builtins::typed_array::TypedArrayKind::Uint8,
            byte_offset: 0,
            length: 4,
            auto_length: false,
        };
        let val = JsValue::TypedArray(Rc::new(RefCell::new(ta)));
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_bytes_from_typed_array_subarray_with_offset() {
        let buf = make_buffer(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ta = crate::builtins::typed_array::JsTypedArray {
            buffer: buf,
            kind: crate::builtins::typed_array::TypedArrayKind::Uint8,
            byte_offset: 2,
            length: 4,
            auto_length: false,
        };
        let val = JsValue::TypedArray(Rc::new(RefCell::new(ta)));
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_bytes_from_typed_array_multibyte_element() {
        // Uint16Array of length 2 starting at byte_offset 2 over an 8-byte buffer:
        // covers bytes [2..6).
        let buf = make_buffer(&[0xAA, 0xBB, 0x01, 0x02, 0x03, 0x04, 0xCC, 0xDD]);
        let ta = crate::builtins::typed_array::JsTypedArray {
            buffer: buf,
            kind: crate::builtins::typed_array::TypedArrayKind::Uint16,
            byte_offset: 2,
            length: 2,
            auto_length: false,
        };
        let val = JsValue::TypedArray(Rc::new(RefCell::new(ta)));
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_bytes_from_typed_array_detached_buffer_errors() {
        let buf = make_buffer(&[1, 2, 3, 4]);
        let ta = crate::builtins::typed_array::JsTypedArray {
            buffer: Rc::clone(&buf),
            kind: crate::builtins::typed_array::TypedArrayKind::Uint8,
            byte_offset: 0,
            length: 4,
            auto_length: false,
        };
        buf.borrow_mut().detached = true;
        let err = bytes_from_js_value(&JsValue::TypedArray(Rc::new(RefCell::new(ta)))).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_bytes_from_dataview_full() {
        let buf = make_buffer(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let dv = crate::builtins::typed_array::dataview_new(buf, 0, None).unwrap();
        let val = JsValue::DataView(Rc::new(RefCell::new(dv)));
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_bytes_from_dataview_with_offset_and_length() {
        let buf = make_buffer(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let dv = crate::builtins::typed_array::dataview_new(buf, 3, Some(4)).unwrap();
        let val = JsValue::DataView(Rc::new(RefCell::new(dv)));
        let bytes = bytes_from_js_value(&val).unwrap();
        assert_eq!(bytes, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_bytes_from_dataview_detached_buffer_errors() {
        let buf = make_buffer(&[1, 2, 3, 4]);
        let dv = crate::builtins::typed_array::dataview_new(Rc::clone(&buf), 0, None).unwrap();
        buf.borrow_mut().detached = true;
        let err = bytes_from_js_value(&JsValue::DataView(Rc::new(RefCell::new(dv)))).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_validate_accepts_arraybuffer_input() {
        let buf = make_buffer(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        let result = wasm_validate(vec![JsValue::ArrayBuffer(buf)]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_validate_accepts_typed_array_subview() {
        // Pad the magic bytes so we can exercise byte_offset.
        let mut data = vec![0xFFu8, 0xFFu8];
        data.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        let buf = make_buffer(&data);
        let ta = crate::builtins::typed_array::JsTypedArray {
            buffer: buf,
            kind: crate::builtins::typed_array::TypedArrayKind::Uint8,
            byte_offset: 2,
            length: 8,
            auto_length: false,
        };
        let result = wasm_validate(vec![JsValue::TypedArray(Rc::new(RefCell::new(ta)))]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_validate_accepts_dataview_subrange() {
        let mut data = vec![0xCCu8];
        data.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        data.push(0xCC);
        let buf = make_buffer(&data);
        let dv = crate::builtins::typed_array::dataview_new(buf, 1, Some(8)).unwrap();
        let result = wasm_validate(vec![JsValue::DataView(Rc::new(RefCell::new(dv)))]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── is_wat ───────────────────────────────────────────────────────────────

    #[test]
    fn test_is_wat_for_text_module() {
        assert!(is_wat(b"(module)"));
    }

    #[test]
    fn test_is_wat_false_for_wasm_magic() {
        assert!(!is_wat(b"\0asm\x01\0\0\0"));
    }

    // ── wasm_validate ─────────────────────────────────────────────────────────

    #[test]
    fn test_validate_empty_module_wat() {
        let result = wasm_validate(vec![wat_val(EMPTY_WAT)]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_validate_valid_binary() {
        let result = wasm_validate(vec![empty_wasm_binary_val()]).unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_validate_invalid_bytes_returns_false() {
        let bad = JsValue::new_array(vec![JsValue::Smi(0x00), JsValue::Smi(0x00)]);
        let result = wasm_validate(vec![bad]).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_validate_invalid_wat_returns_false() {
        let result =
            wasm_validate(vec![JsValue::String("not wasm at all".to_string().into())]).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_validate_no_args_returns_false() {
        let result = wasm_validate(vec![]).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_validate_wrong_type_returns_false() {
        let result = wasm_validate(vec![JsValue::Smi(42)]).unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── wasm_compile ──────────────────────────────────────────────────────────

    #[test]
    fn test_compile_empty_wat_returns_plain_object() {
        let m = wasm_compile(vec![wat_val(EMPTY_WAT)]).unwrap();
        assert!(matches!(m, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_compile_sets_wasm_type() {
        let m = wasm_compile(vec![wat_val(EMPTY_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            let ty = map.borrow().get("__wasm_type__").cloned();
            assert_eq!(
                ty,
                Some(JsValue::String("WebAssembly.Module".to_string().into()))
            );
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_compile_exports_is_array() {
        let m = wasm_compile(vec![wat_val(ADD_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            let exports = map.borrow().get("exports").cloned();
            assert!(matches!(exports, Some(JsValue::Array(_))));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_compile_export_descriptor_has_name_and_kind() {
        let m = wasm_compile(vec![wat_val(ADD_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            let exports = map.borrow().get("exports").cloned().unwrap();
            if let JsValue::Array(arr) = exports {
                assert_eq!(arr.borrow().len(), 1);
                if let JsValue::PlainObject(desc) = &arr.borrow()[0] {
                    let name = desc.borrow().get("name").cloned();
                    let kind = desc.borrow().get("kind").cloned();
                    assert_eq!(name, Some(JsValue::String("add".to_string().into())));
                    assert_eq!(kind, Some(JsValue::String("function".to_string().into())));
                } else {
                    panic!("expected PlainObject descriptor");
                }
            } else {
                panic!("expected Array exports");
            }
        } else {
            panic!("expected PlainObject module");
        }
    }

    #[test]
    fn test_compile_invalid_wat_returns_error() {
        let err = wasm_compile(vec![JsValue::String("bad wat".to_string().into())]).unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    #[test]
    fn test_compile_binary_wasm() {
        let m = wasm_compile(vec![empty_wasm_binary_val()]).unwrap();
        assert!(matches!(m, JsValue::PlainObject(_)));
    }

    // ── wasm_module_ctor ──────────────────────────────────────────────────────

    #[test]
    fn test_module_ctor_returns_module_object() {
        let m = wasm_module_ctor(vec![wat_val(EMPTY_WAT)]).unwrap();
        assert!(matches!(m, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_module_ctor_wasm_bytes_stored() {
        let m = wasm_module_ctor(vec![wat_val(EMPTY_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            let bytes = map.borrow().get("__wasm_bytes__").cloned();
            assert!(matches!(bytes, Some(JsValue::Array(_))));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_module_ctor_add_exports_descriptor() {
        let m = wasm_module_ctor(vec![wat_val(ADD_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            if let Some(JsValue::Array(arr)) = map.borrow().get("exports").cloned() {
                assert_eq!(arr.borrow().len(), 1);
            } else {
                panic!("expected exports array with one entry");
            }
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_module_ctor_multi_export_descriptor() {
        let m = wasm_module_ctor(vec![wat_val(MULTI_EXPORT_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = m {
            if let Some(JsValue::Array(arr)) = map.borrow().get("exports").cloned() {
                assert_eq!(arr.borrow().len(), 2);
            } else {
                panic!("expected exports array with two entries");
            }
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_module_ctor_invalid_returns_error() {
        let err =
            wasm_module_ctor(vec![JsValue::String("nonsense".to_string().into())]).unwrap_err();
        assert!(matches!(err, StatorError::WasmError(_)));
    }

    #[test]
    fn test_module_ctor_type_error_on_bad_input() {
        let err = wasm_module_ctor(vec![JsValue::Smi(0)]).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── wasm_instance_ctor ────────────────────────────────────────────────────

    #[test]
    fn test_instance_ctor_from_empty_module() {
        let module = wasm_module_ctor(vec![wat_val(EMPTY_WAT)]).unwrap();
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        assert!(matches!(inst, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_instance_ctor_has_exports_property() {
        let module = wasm_module_ctor(vec![wat_val(EMPTY_WAT)]).unwrap();
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        if let JsValue::PlainObject(map) = inst {
            assert!(map.borrow().contains_key("exports"));
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_instance_ctor_exports_contains_native_function() {
        let module = wasm_module_ctor(vec![wat_val(ADD_WAT)]).unwrap();
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        if let JsValue::PlainObject(imap) = inst {
            if let Some(JsValue::PlainObject(exp)) = imap.borrow().get("exports").cloned() {
                let add = exp.borrow().get("add").cloned();
                assert!(matches!(add, Some(JsValue::NativeFunction(_))));
            } else {
                panic!("expected exports PlainObject");
            }
        } else {
            panic!("expected instance PlainObject");
        }
    }

    #[test]
    fn test_instance_ctor_exported_function_callable() {
        let module = wasm_module_ctor(vec![wat_val(ADD_WAT)]).unwrap();
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        if let JsValue::PlainObject(imap) = inst {
            if let Some(JsValue::PlainObject(exp)) = imap.borrow().get("exports").cloned() {
                if let Some(JsValue::NativeFunction(add_fn)) = exp.borrow().get("add").cloned() {
                    let result = add_fn(vec![JsValue::Smi(3), JsValue::Smi(4)]).unwrap();
                    assert_eq!(result, JsValue::Smi(7));
                } else {
                    panic!("expected NativeFunction for 'add'");
                }
            } else {
                panic!("expected exports PlainObject");
            }
        } else {
            panic!("expected instance PlainObject");
        }
    }

    #[test]
    fn test_instance_ctor_double_function() {
        let module = wasm_module_ctor(vec![wat_val(DOUBLE_WAT)]).unwrap();
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        if let JsValue::PlainObject(imap) = inst {
            if let Some(JsValue::PlainObject(exp)) = imap.borrow().get("exports").cloned() {
                if let Some(JsValue::NativeFunction(f)) = exp.borrow().get("double").cloned() {
                    let result = f(vec![JsValue::Smi(6)]).unwrap();
                    assert_eq!(result, JsValue::Smi(12));
                } else {
                    panic!("expected NativeFunction for 'double'");
                }
            } else {
                panic!("expected exports PlainObject");
            }
        } else {
            panic!("expected instance PlainObject");
        }
    }

    #[test]
    fn test_instance_ctor_type_error_on_non_module() {
        let err = wasm_instance_ctor(vec![JsValue::Smi(0)]).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── wasm_instantiate ──────────────────────────────────────────────────────

    #[test]
    fn test_instantiate_from_bytes_returns_module_and_instance() {
        let result = wasm_instantiate(vec![wat_val(ADD_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = result {
            assert!(map.borrow().contains_key("module"));
            assert!(map.borrow().contains_key("instance"));
        } else {
            panic!("expected PlainObject {{module, instance}}");
        }
    }

    #[test]
    fn test_instantiate_from_module_returns_instance_directly() {
        let module = wasm_module_ctor(vec![wat_val(ADD_WAT)]).unwrap();
        let inst = wasm_instantiate(vec![module]).unwrap();
        if let JsValue::PlainObject(map) = &inst {
            let ty = map.borrow().get("__wasm_type__").cloned();
            assert_eq!(
                ty,
                Some(JsValue::String("WebAssembly.Instance".to_string().into()))
            );
        } else {
            panic!("expected instance PlainObject");
        }
    }

    #[test]
    fn test_instantiate_from_bytes_exports_callable() {
        let result = wasm_instantiate(vec![wat_val(ADD_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = result {
            if let Some(JsValue::PlainObject(inst_map)) = map.borrow().get("instance").cloned() {
                if let Some(JsValue::PlainObject(exp)) = inst_map.borrow().get("exports").cloned() {
                    if let Some(JsValue::NativeFunction(add_fn)) = exp.borrow().get("add").cloned()
                    {
                        let r = add_fn(vec![JsValue::Smi(10), JsValue::Smi(20)]).unwrap();
                        assert_eq!(r, JsValue::Smi(30));
                        return;
                    }
                }
            }
        }
        panic!("could not reach add export through instantiate result");
    }

    #[test]
    fn test_instantiate_type_error_on_bad_source() {
        let err = wasm_instantiate(vec![JsValue::Smi(0)]).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_imports() {
        let module = wasm_module_ctor(vec![wat_val(IMPORTED_FUNC_WAT)]).unwrap();
        let err = wasm_instance_ctor(vec![module, descriptor(&[])]).unwrap_err();
        match err {
            StatorError::TypeError(msg) => {
                assert!(msg.contains("imports are not implemented"));
                assert!(msg.contains("env.f"));
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_table_import() {
        assert_instance_import_wat_fails_closed(
            IMPORTED_TABLE_WAT,
            &[
                "env.tab",
                "table import",
                "element",
                "min=1",
                "table64=false",
            ],
        );
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_non_shared_memory_import() {
        assert_instance_import_wat_fails_closed(
            IMPORTED_NON_SHARED_MEMORY_WAT,
            &[
                "env.mem",
                "memory import",
                "min=1",
                "max=2",
                "memory64=false",
                "shared=false",
            ],
        );
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_shared_memory_import() {
        assert_instance_import_wat_fails_closed(
            IMPORTED_SHARED_MEMORY_WAT,
            &[
                "env.mem",
                "memory import",
                "min=1",
                "max=1",
                "memory64=false",
                "shared=true",
            ],
        );
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_reference_global_import() {
        assert_instance_import_wat_fails_closed(
            IMPORTED_EXTERNREF_GLOBAL_WAT,
            &["env.g", "global import", "extern", "mutable=false"],
        );
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_mutable_and_v128_global_imports() {
        for (wat, terms) in [
            (
                IMPORTED_MUTABLE_GLOBAL_WAT,
                &["env.g", "global import", "i32", "mutable=true"][..],
            ),
            (
                IMPORTED_V128_GLOBAL_WAT,
                &["env.g", "global import", "v128", "mutable=false"][..],
            ),
            (
                IMPORTED_MUTABLE_V128_GLOBAL_WAT,
                &["env.g", "global import", "v128", "mutable=true"][..],
            ),
        ] {
            assert_instance_import_wat_fails_closed(wat, terms);
        }
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_memory64_import_if_supported() {
        assert_optional_instance_import_wat_fails_closed(
            IMPORTED_MEMORY64_WAT_CANDIDATES,
            &["env.mem64", "memory import", "memory64=true"],
        );
    }

    #[test]
    fn test_instance_ctor_fails_closed_for_custom_page_memory_import_if_supported() {
        assert_optional_instance_import_wat_fails_closed(
            IMPORTED_CUSTOM_PAGE_MEMORY_WAT_CANDIDATES,
            &["env.mem", "memory import", "page_size_log2=12"],
        );
    }

    #[test]
    fn test_instantiate_fails_closed_for_non_function_exports() {
        let err = wasm_instantiate(vec![wat_val(EXPORTED_MEMORY_WAT)]).unwrap_err();
        match err {
            StatorError::TypeError(msg) => {
                assert!(msg.contains("memory export"));
                assert!(msg.contains("not implemented"));
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    // ── Wasm constructors that require missing host bridges ───────────────────

    #[test]
    fn test_memory_ctor_fails_closed_without_fake_buffer() {
        let desc = descriptor(&[("initial", JsValue::Smi(1))]);
        let err = wasm_memory_ctor(vec![desc]).unwrap_err();
        assert!(
            matches!(err, StatorError::TypeError(msg) if msg.contains("WebAssembly.Memory") && msg.contains("not implemented"))
        );
    }

    #[test]
    fn test_table_ctor_fails_closed_without_fake_slots() {
        let desc = descriptor(&[
            ("element", JsValue::String("anyfunc".to_string().into())),
            ("initial", JsValue::Smi(4)),
        ]);
        let err = wasm_table_ctor(vec![desc]).unwrap_err();
        assert!(
            matches!(err, StatorError::TypeError(msg) if msg.contains("WebAssembly.Table") && msg.contains("not implemented"))
        );
    }

    #[test]
    fn test_global_ctor_fails_closed_without_fake_mutability() {
        let desc = descriptor(&[("value", JsValue::String("i32".to_string().into()))]);
        let err = wasm_global_ctor(vec![desc, JsValue::Smi(42)]).unwrap_err();
        assert!(
            matches!(err, StatorError::TypeError(msg) if msg.contains("WebAssembly.Global") && msg.contains("not implemented"))
        );
    }

    // ── make_webassembly_object ───────────────────────────────────────────────

    #[test]
    fn test_make_webassembly_object_returns_plain_object() {
        let wasm = make_webassembly_object();
        assert!(matches!(wasm, JsValue::PlainObject(_)));
    }

    #[test]
    fn test_make_webassembly_object_has_all_api_keys() {
        let wasm = make_webassembly_object();
        if let JsValue::PlainObject(map) = wasm {
            let keys: Vec<&str> = [
                "validate",
                "compile",
                "instantiate",
                "Module",
                "Instance",
                "Memory",
                "Table",
                "Global",
            ]
            .as_slice()
            .iter()
            .copied()
            .collect();
            for key in keys {
                assert!(map.borrow().contains_key(key), "missing key: {key}");
            }
        } else {
            panic!("expected PlainObject");
        }
    }

    #[test]
    fn test_make_webassembly_object_all_values_are_native_functions() {
        let wasm = make_webassembly_object();
        if let JsValue::PlainObject(map) = wasm {
            for (key, val) in map.borrow().iter() {
                assert!(
                    matches!(val, JsValue::NativeFunction(_)),
                    "WebAssembly.{key} is not a NativeFunction"
                );
            }
        }
    }

    // ── full integration: compile + instantiate + call ────────────────────────

    #[test]
    fn test_full_api_compile_then_instance_then_call() {
        // 1. Compile
        let module = wasm_module_ctor(vec![wat_val(ADD_WAT)]).unwrap();
        // 2. Instantiate
        let inst = wasm_instance_ctor(vec![module]).unwrap();
        // 3. Call
        if let JsValue::PlainObject(imap) = inst {
            if let Some(JsValue::PlainObject(exp)) = imap.borrow().get("exports").cloned() {
                if let Some(JsValue::NativeFunction(add)) = exp.borrow().get("add").cloned() {
                    let r = add(vec![JsValue::Smi(100), JsValue::Smi(200)]).unwrap();
                    assert_eq!(r, JsValue::Smi(300));
                    return;
                }
            }
        }
        panic!("integration test failed");
    }

    #[test]
    fn test_full_api_instantiate_from_bytes() {
        let result = wasm_instantiate(vec![wat_val(DOUBLE_WAT)]).unwrap();
        if let JsValue::PlainObject(map) = result {
            if let Some(JsValue::PlainObject(inst_map)) = map.borrow().get("instance").cloned() {
                if let Some(JsValue::PlainObject(exp)) = inst_map.borrow().get("exports").cloned() {
                    if let Some(JsValue::NativeFunction(f)) = exp.borrow().get("double").cloned() {
                        let r = f(vec![JsValue::Smi(7)]).unwrap();
                        assert_eq!(r, JsValue::Smi(14));
                        return;
                    }
                }
            }
        }
        panic!("full instantiate from bytes failed");
    }

    // ── fail-closed surface ──────────────────────────────────────────────────

    const UNSUPPORTED_NAMES: &[&str] = &[
        "Memory",
        "Table",
        "Global",
        "compileStreaming",
        "instantiateStreaming",
        "Tag",
        "Exception",
        "CompileError",
        "LinkError",
        "RuntimeError",
    ];

    #[test]
    fn test_make_wasm_unsupported_returns_type_error() {
        let f = make_wasm_unsupported("compileStreaming");
        let JsValue::NativeFunction(callable) = f else {
            panic!("expected NativeFunction");
        };
        let err = callable(vec![]).unwrap_err();
        match err {
            StatorError::TypeError(msg) => {
                assert!(msg.contains("compileStreaming"));
                assert!(msg.contains("not implemented"));
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_webassembly_exposes_failclosed_keys() {
        let wasm = make_webassembly_object();
        let JsValue::PlainObject(map) = wasm else {
            panic!("expected PlainObject");
        };
        for key in UNSUPPORTED_NAMES {
            assert!(
                map.borrow().contains_key(*key),
                "missing fail-closed key: WebAssembly.{key}"
            );
        }
    }

    #[test]
    fn test_webassembly_failclosed_keys_throw_typeerror() {
        let wasm = make_webassembly_object();
        let JsValue::PlainObject(map) = wasm else {
            panic!("expected PlainObject");
        };
        for key in UNSUPPORTED_NAMES {
            let val = map.borrow().get(*key).cloned();
            let Some(JsValue::NativeFunction(callable)) = val else {
                panic!("WebAssembly.{key} is not a NativeFunction");
            };
            let err = callable(vec![]).unwrap_err();
            match err {
                StatorError::TypeError(msg) => {
                    assert!(
                        msg.contains(key),
                        "TypeError for WebAssembly.{key} did not mention name: {msg}"
                    );
                    assert!(
                        msg.contains("not implemented"),
                        "TypeError for WebAssembly.{key} did not mention 'not implemented': {msg}"
                    );
                }
                other => panic!("WebAssembly.{key} expected TypeError, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_webassembly_failclosed_does_not_shadow_real_apis() {
        // Adding fail-closed entries must not remove or replace any of the real
        // APIs that already work, and should keep explicit placeholders for
        // bridge-dependent constructors.
        let wasm = make_webassembly_object();
        let JsValue::PlainObject(map) = wasm else {
            panic!("expected PlainObject");
        };
        for key in [
            "validate",
            "compile",
            "instantiate",
            "Module",
            "Instance",
            "Memory",
            "Table",
            "Global",
        ] {
            assert!(
                map.borrow().contains_key(key),
                "lost WebAssembly key: {key}"
            );
        }
        // Spot-check that the real APIs still actually work.
        let v = wasm_validate(vec![wat_val(EMPTY_WAT)]).unwrap();
        assert_eq!(v, JsValue::Boolean(true));
    }

    #[test]
    fn test_webassembly_no_streaming_fake_promise_returned() {
        // compileStreaming / instantiateStreaming must NOT return a Promise or
        // any object; they must throw synchronously so callers cannot mistake
        // the fail-closed shape for a working implementation.
        let wasm = make_webassembly_object();
        let JsValue::PlainObject(map) = wasm else {
            panic!("expected PlainObject");
        };
        for key in ["compileStreaming", "instantiateStreaming"] {
            let Some(JsValue::NativeFunction(callable)) = map.borrow().get(key).cloned() else {
                panic!("missing {key}");
            };
            let result = callable(vec![JsValue::Undefined]);
            assert!(
                matches!(result, Err(StatorError::TypeError(_))),
                "WebAssembly.{key} should throw TypeError, got {result:?}"
            );
        }
    }

    #[test]
    fn test_validate_and_compile_consistency() {
        // If validate says true, compile must succeed (and vice versa).
        let valid_result = wasm_validate(vec![wat_val(ADD_WAT)]).unwrap();
        let compile_result = wasm_compile(vec![wat_val(ADD_WAT)]);
        assert_eq!(valid_result, JsValue::Boolean(true));
        assert!(compile_result.is_ok());

        let invalid = JsValue::String("not wasm".to_string().into());
        let invalid_result = wasm_validate(vec![invalid.clone()]).unwrap();
        let compile_err = wasm_compile(vec![invalid]);
        assert_eq!(invalid_result, JsValue::Boolean(false));
        assert!(compile_err.is_err());
    }
}

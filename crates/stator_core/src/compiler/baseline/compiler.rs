//! Baseline (non-optimising) JIT compiler.
//!
//! Translates a [`BytecodeArray`] to x86-64 machine code by compiling each
//! bytecode instruction to a fixed sequence of native instructions.
//!
//! # Design
//!
//! ## JIT value representation
//!
//! Inside JIT code all values are `i64`.  The encoding is:
//!
//! | JavaScript value | `i64` representation |
//! |------------------|----------------------|
//! | `Smi(v)`         | `v as i64`           |
//! | `true`           | [`JIT_TRUE`]         |
//! | `false`          | [`JIT_FALSE`]        |
//! | `undefined`      | [`JIT_UNDEFINED`]    |
//! | `null`           | [`JIT_NULL`]         |
//!
//! Smi values occupy `i32::MIN..=i32::MAX` (≈ ±2.1 billion).
//! The special sentinels are all above `i32::MAX` and therefore disjoint.
//!
//! ## Register conventions
//!
//! | Register | Role |
//! |----------|------|
//! | `R12`    | accumulator (callee-saved) |
//! | `R14`    | register-file base pointer (`*mut i64`, callee-saved) |
//! | `R11`    | scratch (caller-saved) |
//!
//! ## JIT function signature
//!
//! ```text
//! extern "C" fn(regs: *mut i64) -> i64
//! ```
//!
//! `regs` points to an array of `parameter_count + frame_size` `i64` slots.
//! The caller pre-initialises parameter slots; the function returns the
//! accumulator value on normal completion or [`JIT_DEOPT`] if it encounters
//! a bytecode it cannot handle.
//!
//! ## Register-file slot layout
//!
//! Slot indices follow the interpreter's convention:
//!
//! ```text
//! [ param[0], param[1], …, local[0], local[1], … ]
//!  ^------ parameter_count ------^^---- frame_size ----^
//! ```
//!
//! For a bytecode operand `v` (encoded as `u32`):
//! - `v as i32 ≥ 0`: flat index = `parameter_count + (v as usize)`.
//! - `v as i32 < 0`: flat index = `-(v as i32 + 1) as usize` (parameter).

use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, decode_with_byte_offsets};
use crate::compiler::baseline::masm_x64::{CondCode, Label, MacroAssembler, Reg64};
use crate::error::{StatorError, StatorResult};

#[cfg(all(target_arch = "x86_64", unix))]
use std::cell::{Cell, RefCell};
#[cfg(all(target_arch = "x86_64", unix))]
use std::rc::Rc;

// ─────────────────────────────────────────────────────────────────────────────
// Table serialization constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic number written in the last 4 bytes of the code buffer to mark the
/// presence of serialized safepoint and deopt tables.  ASCII `STAT` in
/// little-endian order.
pub const METADATA_MAGIC: u32 = 0x5441_5453;

/// Size in bytes of the fixed metadata footer appended at the very end of the
/// code buffer (3 × `u32` = 12 bytes).
pub const FOOTER_SIZE: usize = 12;

/// Serialized size of a single [`SafepointEntry`] in bytes
/// (`code_offset u32` + `bytecode_index u32` + `gc_map u64` = 16 bytes).
const SAFEPOINT_ENTRY_SIZE: usize = 16;

/// Serialized size of a single [`DeoptEntry`] in bytes
/// (`code_offset u32` + `bytecode_offset u32` + `liveness_map u64` = 16 bytes).
const DEOPT_ENTRY_SIZE: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// JIT value sentinels
// ─────────────────────────────────────────────────────────────────────────────

/// JIT representation of `false`.
///
/// Chosen to be just above `i32::MAX` so it cannot be confused with any Smi.
pub const JIT_FALSE: i64 = 0x1_0000_0000_i64;
/// JIT representation of `true` (`JIT_FALSE + 1`).
pub const JIT_TRUE: i64 = 0x1_0000_0001_i64;
/// JIT representation of `undefined`.
pub const JIT_UNDEFINED: i64 = 0x1_0000_0002_i64;
/// JIT representation of `null`.
pub const JIT_NULL: i64 = 0x1_0000_0003_i64;
/// Sentinel returned by the JIT function when it encounters an unsupported
/// bytecode and must fall back to the interpreter.
pub const JIT_DEOPT: i64 = i64::MIN;

/// Base tag for heap-object handles in the JIT `i64` register file.
///
/// Complex JavaScript values (objects, arrays, functions, strings) that
/// cannot be represented as a plain `i64` are stored in a thread-local side
/// table.  The handle `JIT_HEAP_TAG + index` is placed in the register file
/// instead.
#[cfg(all(target_arch = "x86_64", unix))]
pub(crate) const JIT_HEAP_TAG: i64 = 0x2_0000_0000_i64;

/// Convert a JIT `i64` value back to a [`crate::objects::value::JsValue`].
///
/// Returns `None` if `v` is not a recognised sentinel and does not fit in
/// `i32` (i.e. is an out-of-range or deopt value).
pub fn jit_to_jsvalue(v: i64) -> Option<crate::objects::value::JsValue> {
    use crate::objects::value::JsValue;
    if v == JIT_FALSE {
        Some(JsValue::Boolean(false))
    } else if v == JIT_TRUE {
        Some(JsValue::Boolean(true))
    } else if v == JIT_UNDEFINED {
        Some(JsValue::Undefined)
    } else if v == JIT_NULL {
        Some(JsValue::Null)
    } else if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
        Some(JsValue::Smi(v as i32))
    } else {
        // Large integer outside Smi range: promote to HeapNumber (f64).
        // All integers up to 2^53 are exactly representable as f64, so
        // arithmetic results that overflow i32 can be returned faithfully.
        Some(JsValue::HeapNumber(v as f64))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime call-stub infrastructure (x86-64 + Unix only)
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-local state used by the JIT runtime trampoline to access the
/// constant pool and to store heap-allocated JavaScript objects that cannot
/// be encoded as plain `i64` values.
#[cfg(all(target_arch = "x86_64", unix))]
pub(crate) mod jit_runtime {
    use super::*;
    use crate::bytecode::bytecode_array::JitExecutableCode;
    use crate::interpreter::GlobalEnv;
    use crate::objects::property_map::{INTERNAL_PROTO_PROPERTY_KEY, PropertyMap};
    use crate::objects::value::{JsContext, JsValue};

    /// Combined inline caches for named-property stubs.
    ///
    /// Merging the LDA and prototype ICs into one TLS variable saves
    /// one `thread_local!` lookup on the IC-miss path (where both
    /// caches are probed sequentially).
    struct JitPropertyIcState {
        /// Own-property IC: `(name_idx, shape_id, offset)`.
        /// Direct-mapped by `name_idx & 63`.
        lda: [(u32, u64, usize); 64],
        /// Prototype-chain IC: `(name_idx, own_shape_id, cached_value)`.
        /// Direct-mapped by `name_idx & 15`.
        proto: [(u32, u64, i64); 16],
    }

    /// Combined global-environment + inline-cache state for
    /// `LdaGlobal`/`StaGlobal` stubs.  Stored in a single
    /// `thread_local!` to save one TLS lookup per global access.
    struct JitGlobalState {
        env: Option<Rc<RefCell<GlobalEnv>>>,
        /// Direct-mapped IC: `(name_idx, slot_index, generation)`.
        ic: [(u32, usize, u64); 64],
    }

    thread_local! {
        /// Raw pointer to the [`BytecodeArray`] of the currently-executing
        /// JIT function.  Set by [`jit_runtime_setup`], cleared by
        /// [`jit_runtime_teardown`].
        ///
        /// # Safety
        ///
        /// The pointee is alive for the entire JIT execution because the
        /// caller of `CompiledCode::execute` holds an `Rc<BytecodeArray>`.
        static RT_BYTECODE: Cell<*const BytecodeArray> = const { Cell::new(std::ptr::null()) };

        /// Side table for heap-allocated [`JsValue`]s.
        ///
        /// Objects, arrays, functions, and strings cannot be encoded as `i64`.
        /// Instead, they are pushed into this table and the register file
        /// stores `JIT_HEAP_TAG + index` as a handle.
        static RT_HEAP: RefCell<Vec<JsValue>> = const { RefCell::new(Vec::new()) };

        /// Combined own-property + prototype inline caches for
        /// `LdaNamedProperty` stubs.  Cleared by [`jit_runtime_setup`]
        /// to prevent cross-function constant-pool index collisions.
        static RT_PROP_IC: RefCell<JitPropertyIcState> = const {
            RefCell::new(JitPropertyIcState {
                lda: [(u32::MAX, 0, 0); 64],
                proto: [(u32::MAX, 0, 0); 16],
            })
        };

        /// Combined global environment + IC for `LdaGlobal`/`StaGlobal`
        /// runtime stubs.  Set once via [`jit_runtime_set_global_env`] at
        /// the top of `run_dispatch` and reused by all JIT calls within
        /// that execution.  Merging the IC array into the same TLS
        /// variable saves one `thread_local!` lookup per global access on
        /// the IC-hit path.
        static RT_GLOBAL: RefCell<JitGlobalState> = const {
            RefCell::new(JitGlobalState {
                env: None,
                ic: [(u32::MAX, 0, 0); 64],
            })
        };

        /// Reference to the current closure context for
        /// `LdaCurrentContextSlot`/`StaCurrentContextSlot` stubs.
        /// Set via [`jit_runtime_set_context`] before JIT execution of
        /// closures.
        static RT_CONTEXT: RefCell<Option<Rc<RefCell<JsContext>>>> = const { RefCell::new(None) };

        /// Cached raw pointers to the three most-accessed TLS variables.
        /// Populated once by [`cache_rt_ptrs`] and reused by
        /// [`exec_jit_callee`] to avoid repeated `.with()` lookups in
        /// the nested-call hot path.
        static RT_PTRS: Cell<RtPtrs> = const { Cell::new(RtPtrs::EMPTY) };
    }

    /// Cached raw pointers to frequently-accessed TLS variables.
    ///
    /// # Safety
    ///
    /// Thread-local storage in Rust has a stable address for the
    /// thread's lifetime.  These pointers are set once during
    /// [`cache_rt_ptrs`] and are only dereferenced from the same
    /// thread.
    #[derive(Clone, Copy)]
    struct RtPtrs {
        heap: *const RefCell<Vec<JsValue>>,
        context: *const RefCell<Option<Rc<RefCell<JsContext>>>>,
        bytecode: *const Cell<*const BytecodeArray>,
        global: *const RefCell<JitGlobalState>,
        prop_ic: *const RefCell<JitPropertyIcState>,
    }

    impl RtPtrs {
        const EMPTY: Self = Self {
            heap: std::ptr::null(),
            context: std::ptr::null(),
            bytecode: std::ptr::null(),
            global: std::ptr::null(),
            prop_ic: std::ptr::null(),
        };

        fn is_cached(&self) -> bool {
            !self.heap.is_null()
        }
    }

    /// Populate [`RT_PTRS`] so that [`exec_jit_callee`] can bypass
    /// per-variable `.with()` calls.
    fn cache_rt_ptrs() {
        RT_PTRS.with(|p| {
            if p.get().is_cached() {
                return;
            }
            let ptrs = RtPtrs {
                heap: RT_HEAP.with(|h| h as *const RefCell<Vec<JsValue>>),
                context: RT_CONTEXT.with(|c| c as *const RefCell<Option<Rc<RefCell<JsContext>>>>),
                bytecode: RT_BYTECODE.with(|b| b as *const Cell<*const BytecodeArray>),
                global: RT_GLOBAL.with(|g| g as *const RefCell<JitGlobalState>),
                prop_ic: RT_PROP_IC.with(|ic| ic as *const RefCell<JitPropertyIcState>),
            };
            p.set(ptrs);
        });
    }

    // ── Setup / teardown ─────────────────────────────────────────────────

    /// Prepare thread-local state before JIT execution.
    ///
    /// Must be called before `CompiledCode::execute` so that runtime call
    /// stubs can access the constant pool.
    pub fn jit_runtime_setup(ba: &BytecodeArray) {
        cache_rt_ptrs();
        RT_BYTECODE.with(|b| b.set(ba as *const BytecodeArray));
        RT_HEAP.with(|h| h.borrow_mut().clear());
        RT_PROP_IC.with(|ic| {
            let mut c = ic.borrow_mut();
            c.lda = [(u32::MAX, 0, 0); 64];
            c.proto = [(u32::MAX, 0, 0); 16];
        });
    }

    /// Set the global environment for `LdaGlobal`/`StaGlobal` stubs.
    ///
    /// Called once at the top of `run_dispatch`.  The reference is
    /// `Rc`-cloned so it stays alive even if the interpreter frame is
    /// rebuilt (e.g. tail-call).  Not cleared by [`jit_runtime_teardown`]
    /// so it persists across multiple JIT entries within the same
    /// `run_dispatch` invocation.
    ///
    /// When the environment `Rc` is the **same** as the currently stored
    /// one (pointer-equal), the direct-mapped IC is kept warm — avoiding
    /// cold-start IC misses on every criterion iteration.
    pub fn jit_runtime_set_global_env(env: Rc<RefCell<GlobalEnv>>) {
        RT_GLOBAL.with(|g| {
            let mut state = g.borrow_mut();
            let same_env = state
                .env
                .as_ref()
                .map_or(false, |old| Rc::ptr_eq(old, &env));
            if same_env {
                return;
            }
            state.env = Some(env);
            state.ic = [(u32::MAX, 0, 0); 64];
        });
    }

    /// Clean up thread-local state after JIT execution.
    ///
    /// Note: does NOT clear `RT_GLOBAL` — it persists across JIT
    /// calls within a single `run_dispatch` invocation.
    pub fn jit_runtime_teardown() {
        RT_BYTECODE.with(|b| b.set(std::ptr::null()));
        RT_HEAP.with(|h| h.borrow_mut().clear());
        RT_PROP_IC.with(|ic| {
            let mut c = ic.borrow_mut();
            c.lda = [(u32::MAX, 0, 0); 64];
            c.proto = [(u32::MAX, 0, 0); 16];
        });
    }

    /// Set the current closure context for context-slot stubs.
    ///
    /// Called before JIT execution of closure bodies that use
    /// `LdaCurrentContextSlot` / `StaCurrentContextSlot`.
    pub fn jit_runtime_set_context(ctx: Option<Rc<RefCell<JsContext>>>) {
        RT_CONTEXT.with(|c| *c.borrow_mut() = ctx);
    }

    // ── Heap-handle helpers ──────────────────────────────────────────────

    /// Returns `true` if `v` is a heap-object handle.
    #[inline]
    fn is_heap_handle(v: i64) -> bool {
        v >= JIT_HEAP_TAG && v < JIT_HEAP_TAG + 0x1_0000_0000
    }

    /// Allocate a new heap handle for `val`, returning the `i64` handle.
    fn alloc_heap_handle(val: JsValue) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime; no concurrent borrows.
            let heap = unsafe { &mut *(&*ptrs.heap).as_ptr() };
            let idx = heap.len();
            heap.push(val);
            JIT_HEAP_TAG + idx as i64
        } else {
            RT_HEAP.with(|heap| {
                let mut heap = heap.borrow_mut();
                let idx = heap.len();
                heap.push(val);
                JIT_HEAP_TAG + idx as i64
            })
        }
    }

    /// Public entry point for allocating a JIT heap handle.
    ///
    /// Used by the interpreter's JIT entry path to convert non-primitive
    /// [`JsValue`] types (objects, arrays, functions, strings) into `i64`
    /// handles that JIT code and runtime stubs can work with.
    pub fn alloc_jit_heap_handle(val: JsValue) -> i64 {
        alloc_heap_handle(val)
    }

    /// Retrieve the [`JsValue`] stored behind `handle`.
    fn get_heap_object(handle: i64) -> Option<JsValue> {
        if !is_heap_handle(handle) {
            return None;
        }
        let idx = (handle - JIT_HEAP_TAG) as usize;
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime; no concurrent borrows.
            let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
            heap.get(idx).cloned()
        } else {
            RT_HEAP.with(|heap| heap.borrow().get(idx).cloned())
        }
    }

    /// Convert a JIT `i64` (possibly a heap handle) to a [`JsValue`].
    fn jit_i64_to_jsvalue(v: i64) -> JsValue {
        if is_heap_handle(v) {
            get_heap_object(v).unwrap_or(JsValue::Undefined)
        } else {
            super::jit_to_jsvalue(v).unwrap_or(JsValue::Undefined)
        }
    }

    /// Convert a [`JsValue`] to a JIT `i64`, allocating a heap handle for
    /// complex types.
    fn jsvalue_to_jit_i64(v: JsValue) -> i64 {
        match &v {
            JsValue::Smi(n) => i64::from(*n),
            JsValue::Boolean(true) => JIT_TRUE,
            JsValue::Boolean(false) => JIT_FALSE,
            JsValue::Undefined => JIT_UNDEFINED,
            JsValue::Null => JIT_NULL,
            JsValue::HeapNumber(f) => {
                let f = *f;
                if f.fract() == 0.0 && f >= i32::MIN as f64 && f <= i32::MAX as f64 {
                    f as i64
                } else {
                    alloc_heap_handle(v)
                }
            }
            _ => alloc_heap_handle(v),
        }
    }

    /// Extended `jit_to_jsvalue` that also resolves heap handles.
    ///
    /// Called by the interpreter after JIT execution to convert the return
    /// value back to a [`JsValue`].
    pub fn jit_to_jsvalue_ext(v: i64) -> Option<JsValue> {
        if is_heap_handle(v) {
            get_heap_object(v)
        } else {
            super::jit_to_jsvalue(v)
        }
    }

    // ── String constant helper ───────────────────────────────────────────

    /// Read a string from the constant pool of the currently-executing
    /// JIT function.
    fn get_rt_string_constant(idx: u32) -> Option<String> {
        let ptrs = RT_PTRS.with(|p| p.get());
        let ptr = if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime.
            unsafe { &*ptrs.bytecode }.get()
        } else {
            RT_BYTECODE.with(|ba_ptr| ba_ptr.get())
        };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: The pointer is valid for the duration of JIT execution
        // (set by jit_runtime_setup, cleared by jit_runtime_teardown).
        let ba = unsafe { &*ptr };
        match ba.get_constant(idx) {
            Some(ConstantPoolEntry::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Like [`get_rt_string_constant`] but returns a `&str` reference
    /// without cloning. The reference is valid for the duration of JIT
    /// execution (the backing `BytecodeArray` is alive on the call stack).
    ///
    /// # Safety
    ///
    /// The `RT_BYTECODE` / `RT_PTRS.bytecode` pointer must be valid.
    fn get_rt_string_constant_ref(idx: u32) -> Option<&'static str> {
        let ptrs = RT_PTRS.with(|p| p.get());
        let ptr = if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime.
            unsafe { &*ptrs.bytecode }.get()
        } else {
            RT_BYTECODE.with(|ba_ptr| ba_ptr.get())
        };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: The pointer is valid for the duration of JIT execution
        // (set by jit_runtime_setup, cleared by jit_runtime_teardown).
        let ba = unsafe { &*ptr };
        match ba.get_constant(idx) {
            Some(ConstantPoolEntry::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    // ── Trampoline entry point ───────────────────────────────────────────

    /// Runtime trampoline called from JIT machine code for opcodes that
    /// cannot be compiled to native instructions.
    ///
    /// # Calling convention
    ///
    /// System V AMD64: arguments in RDI, RSI, RDX, RCX, R8; return in RAX.
    ///
    /// # Returns
    ///
    /// The new accumulator value as `i64` on success, or [`JIT_DEOPT`] if
    /// the operation cannot be handled (causing the entire JIT function to
    /// fall back to the interpreter).
    ///
    /// # Safety
    ///
    /// `regs` must point to a valid, adequately-sized JIT register file
    /// (`*mut i64`) that outlives this call.
    pub extern "C" fn jit_runtime_trampoline(
        opcode: u32,
        regs: *mut i64,
        acc: i64,
        operand1: i64,
        operand2: i64,
    ) -> i64 {
        jit_runtime_dispatch(opcode, regs, acc, operand1, operand2).unwrap_or(JIT_DEOPT)
    }

    /// Inner dispatch for the trampoline.  Returns `None` to signal deopt.
    fn jit_runtime_dispatch(
        opcode: u32,
        regs: *mut i64,
        acc: i64,
        operand1: i64,
        operand2: i64,
    ) -> Option<i64> {
        let op = Opcode::try_from_u8(opcode as u8).ok()?;

        match op {
            // ── Object / array creation ──────────────────────────────────
            Opcode::CreateEmptyObjectLiteral => {
                let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
                Some(jsvalue_to_jit_i64(obj))
            }

            Opcode::CreateObjectLiteral => {
                let feedback_slot = operand1;
                let capacity = operand2.max(4) as usize;

                // Try template caching via the BytecodeArray's object
                // literal template table (same mechanism the interpreter
                // uses).  operand1 carries the feedback slot; -1 means no
                // slot.
                if feedback_slot >= 0 {
                    let slot = feedback_slot as u32;

                    // SAFETY: RT_BYTECODE is valid for the lifetime of
                    // this JIT call.
                    let ba = RT_BYTECODE.with(|b| b.get());
                    if !ba.is_null() {
                        // SAFETY: pointer is valid and points to a live
                        // BytecodeArray.
                        let ba_ref = unsafe { &*ba };

                        // Fast path: clone a previously cached template.
                        if let Some(map) = ba_ref.clone_object_literal_template(slot) {
                            let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
                            return Some(jsvalue_to_jit_i64(obj));
                        }

                        // Second execution: promote pending → cached.
                        if let Some(map) = ba_ref.promote_object_literal_template(slot) {
                            let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
                            return Some(jsvalue_to_jit_i64(obj));
                        }

                        // First execution: create fresh and register as
                        // pending for future promotion.
                        let map = PropertyMap::with_capacity(capacity);
                        let rc = Rc::new(RefCell::new(map));
                        ba_ref.set_object_literal_pending(slot, Rc::clone(&rc));
                        let obj = JsValue::PlainObject(rc);
                        return Some(jsvalue_to_jit_i64(obj));
                    }
                }

                // Fallback: no feedback slot or no BA pointer.
                let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::with_capacity(
                    capacity,
                ))));
                Some(jsvalue_to_jit_i64(obj))
            }

            Opcode::CreateEmptyArrayLiteral | Opcode::CreateArrayLiteral => {
                let arr = JsValue::Array(Rc::new(RefCell::new(Vec::new())));
                Some(jsvalue_to_jit_i64(arr))
            }

            // ── Named property load ──────────────────────────────────────
            Opcode::LdaNamedProperty => {
                let obj_flat = operand1 as usize;
                let name_idx = operand2 as u32;
                // SAFETY: obj_flat was computed by the compiler from a valid
                // bytecode register operand and is within bounds.
                let obj_i64 = unsafe { *regs.add(obj_flat) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let prop_name = get_rt_string_constant(name_idx)?;

                let result = match &obj {
                    JsValue::PlainObject(map) => map
                        .borrow()
                        .get(&prop_name)
                        .cloned()
                        .unwrap_or(JsValue::Undefined),
                    JsValue::Array(arr) => {
                        if prop_name == "length" {
                            JsValue::Smi(arr.borrow().len() as i32)
                        } else {
                            JsValue::Undefined
                        }
                    }
                    _ => return None,
                };
                Some(jsvalue_to_jit_i64(result))
            }

            // ── Named property store ─────────────────────────────────────
            Opcode::StaNamedProperty
            | Opcode::StaNamedOwnProperty
            | Opcode::DefineNamedOwnProperty => {
                let obj_flat = operand1 as usize;
                let name_idx = operand2 as u32;
                // SAFETY: valid index (see above).
                let obj_i64 = unsafe { *regs.add(obj_flat) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let value = jit_i64_to_jsvalue(acc);
                let prop_name = get_rt_string_constant(name_idx)?;

                match &obj {
                    JsValue::PlainObject(map) => {
                        map.borrow_mut().insert(prop_name, value);
                    }
                    _ => return None,
                }
                Some(acc) // accumulator unchanged
            }

            // ── Keyed property load ──────────────────────────────────────
            Opcode::LdaKeyedProperty => {
                let obj_flat = operand1 as usize;
                // SAFETY: valid index.
                let obj_i64 = unsafe { *regs.add(obj_flat) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let key = jit_i64_to_jsvalue(acc);

                let result = match (&obj, &key) {
                    (JsValue::Array(arr), JsValue::Smi(idx)) if *idx >= 0 => {
                        let i = *idx as usize;
                        let borrow = arr.borrow();
                        match borrow.get(i) {
                            Some(v) if !matches!(v, JsValue::TheHole) => v.clone(),
                            _ => JsValue::Undefined,
                        }
                    }
                    (JsValue::PlainObject(map), JsValue::Smi(idx)) if *idx >= 0 => {
                        let key_str = idx.to_string();
                        map.borrow()
                            .get(&key_str)
                            .cloned()
                            .unwrap_or(JsValue::Undefined)
                    }
                    (JsValue::PlainObject(map), JsValue::String(s)) => map
                        .borrow()
                        .get(&**s)
                        .cloned()
                        .unwrap_or(JsValue::Undefined),
                    _ => return None,
                };
                Some(jsvalue_to_jit_i64(result))
            }

            // ── Keyed property store ─────────────────────────────────────
            Opcode::StaKeyedProperty => {
                let obj_flat = operand1 as usize;
                let key_flat = operand2 as usize;
                // SAFETY: valid indices.
                let obj_i64 = unsafe { *regs.add(obj_flat) };
                let key_i64 = unsafe { *regs.add(key_flat) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let key = jit_i64_to_jsvalue(key_i64);
                let value = jit_i64_to_jsvalue(acc);

                match (&obj, &key) {
                    (JsValue::Array(arr), JsValue::Smi(idx)) if *idx >= 0 => {
                        let i = *idx as usize;
                        let mut v = arr.borrow_mut();
                        if i >= v.len() {
                            let cur_len = v.len();
                            let new_cap = (i + 1).next_power_of_two();
                            v.reserve(new_cap - cur_len);
                            v.resize(i + 1, JsValue::TheHole);
                        }
                        v[i] = value;
                    }
                    (JsValue::PlainObject(map), JsValue::Smi(idx)) if *idx >= 0 => {
                        let key_str = idx.to_string();
                        map.borrow_mut().insert(key_str, value);
                    }
                    (JsValue::PlainObject(map), JsValue::String(s)) => {
                        map.borrow_mut().insert(s.to_string(), value);
                    }
                    _ => return None,
                }
                Some(acc)
            }

            // ── Array literal store ──────────────────────────────────────
            Opcode::StaInArrayLiteral => {
                let arr_flat = operand1 as usize;
                let idx_flat = operand2 as usize;
                // SAFETY: valid indices.
                let arr_i64 = unsafe { *regs.add(arr_flat) };
                let idx_i64 = unsafe { *regs.add(idx_flat) };
                let arr = jit_i64_to_jsvalue(arr_i64);
                let idx = jit_i64_to_jsvalue(idx_i64);
                let value = jit_i64_to_jsvalue(acc);

                match (&arr, &idx) {
                    (JsValue::Array(arr), JsValue::Smi(i)) if *i >= 0 => {
                        let i = *i as usize;
                        let mut v = arr.borrow_mut();
                        if i >= v.len() {
                            let cur_len = v.len();
                            let new_cap = (i + 1).next_power_of_two();
                            v.reserve(new_cap - cur_len);
                            v.resize(i + 1, JsValue::TheHole);
                        }
                        v[i] = value;
                    }
                    _ => return None,
                }
                Some(acc)
            }

            // ── Function calls (native functions only) ───────────────────
            Opcode::CallUndefinedReceiver0 => {
                let callee_flat = operand1 as usize;
                // SAFETY: valid index.
                let callee_i64 = unsafe { *regs.add(callee_flat) };
                let callee = jit_i64_to_jsvalue(callee_i64);

                // Save the bytecode pointer in case the native function
                // triggers inner JIT execution that overwrites it.
                let saved_ba = RT_BYTECODE.with(|b| b.get());

                let result = match &callee {
                    JsValue::NativeFunction(f) => {
                        let r = f(vec![]);
                        RT_BYTECODE.with(|b| b.set(saved_ba));
                        match r {
                            Ok(v) => Some(jsvalue_to_jit_i64(v)),
                            Err(_) => None,
                        }
                    }
                    JsValue::Function(ba) => call_js_function(ba, vec![], &[], saved_ba),
                    _ => {
                        RT_BYTECODE.with(|b| b.set(saved_ba));
                        None
                    }
                };
                result
            }

            Opcode::CallUndefinedReceiver1 => {
                let callee_flat = operand1 as usize;
                let arg1_flat = operand2 as usize;
                // SAFETY: valid indices.
                let callee_i64 = unsafe { *regs.add(callee_flat) };
                let arg1_i64 = unsafe { *regs.add(arg1_flat) };
                let callee = jit_i64_to_jsvalue(callee_i64);
                let arg1 = jit_i64_to_jsvalue(arg1_i64);

                let saved_ba = RT_BYTECODE.with(|b| b.get());

                let result = match &callee {
                    JsValue::NativeFunction(f) => {
                        let r = f(vec![arg1]);
                        RT_BYTECODE.with(|b| b.set(saved_ba));
                        match r {
                            Ok(v) => Some(jsvalue_to_jit_i64(v)),
                            Err(_) => None,
                        }
                    }
                    JsValue::Function(ba) => {
                        call_js_function(ba, vec![arg1], &[arg1_i64], saved_ba)
                    }
                    _ => {
                        RT_BYTECODE.with(|b| b.set(saved_ba));
                        None
                    }
                };
                result
            }

            // ── Context slot loads/stores ────────────────────────────────
            Opcode::LdaCurrentContextSlot | Opcode::LdaImmutableCurrentContextSlot => {
                let slot_idx = operand1 as usize;
                RT_CONTEXT.with(|ctx_cell| {
                    let ctx_opt = ctx_cell.borrow();
                    let ctx_rc = ctx_opt.as_ref()?;
                    let ctx = ctx_rc.borrow();
                    let value = ctx
                        .slots
                        .get(slot_idx)
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    Some(jsvalue_to_jit_i64(value))
                })
            }

            Opcode::StaCurrentContextSlot => {
                let slot_idx = operand1 as usize;
                let value = jit_i64_to_jsvalue(acc);
                RT_CONTEXT.with(|ctx_cell| {
                    let ctx_opt = ctx_cell.borrow();
                    let ctx_rc = ctx_opt.as_ref()?;
                    let mut ctx = ctx_rc.borrow_mut();
                    if slot_idx >= ctx.slots.len() {
                        ctx.slots.resize(slot_idx + 1, JsValue::Undefined);
                    }
                    ctx.slots[slot_idx] = value;
                    Some(acc)
                })
            }

            Opcode::LdaContextSlot | Opcode::LdaImmutableContextSlot => {
                // operand1 = slot_idx, operand2 = register containing context
                let slot_idx = operand1 as usize;
                let ctx_flat = operand2 as usize;
                // SAFETY: ctx_flat was computed by the compiler from a valid
                // bytecode register operand.
                let ctx_i64 = unsafe { *regs.add(ctx_flat) };
                let ctx_val = jit_i64_to_jsvalue(ctx_i64);
                match ctx_val {
                    JsValue::Context(ctx_rc) => {
                        let ctx = ctx_rc.borrow();
                        let value = ctx
                            .slots
                            .get(slot_idx)
                            .cloned()
                            .unwrap_or(JsValue::Undefined);
                        Some(jsvalue_to_jit_i64(value))
                    }
                    _ => None,
                }
            }

            Opcode::StaContextSlot => {
                let slot_idx = operand1 as usize;
                let ctx_flat = operand2 as usize;
                let value = jit_i64_to_jsvalue(acc);
                // SAFETY: ctx_flat was computed by the compiler.
                let ctx_i64 = unsafe { *regs.add(ctx_flat) };
                let ctx_val = jit_i64_to_jsvalue(ctx_i64);
                match ctx_val {
                    JsValue::Context(ctx_rc) => {
                        let mut ctx = ctx_rc.borrow_mut();
                        if slot_idx >= ctx.slots.len() {
                            ctx.slots.resize(slot_idx + 1, JsValue::Undefined);
                        }
                        ctx.slots[slot_idx] = value;
                        Some(acc)
                    }
                    _ => None,
                }
            }

            // ── Global variable access ──────────────────────────────────
            // NOTE: Usually reached via specialized stubs, but the generic
            // trampoline path delegates to the shared inner functions.
            Opcode::LdaGlobal => lda_global_inner(operand1 as u32),
            Opcode::StaGlobal => sta_global_inner(operand1 as u32, acc),

            // ── Construct (new) ─────────────────────────────────────────
            Opcode::Construct => {
                // operand1 = ctor_flat, operand2 packs args_start (high 16)
                // and arg_count (low 16).
                let ctor_flat = operand1 as usize;
                let args_start_flat = ((operand2 >> 16) & 0xFFFF) as usize;
                let arg_count = (operand2 & 0xFFFF) as usize;

                // SAFETY: ctor_flat is within the register file.
                let ctor_i64 = unsafe { *regs.add(ctor_flat) };
                let ctor_val = jit_i64_to_jsvalue(ctor_i64);

                let saved_ba = RT_BYTECODE.with(|b| b.get());

                let result = construct_inner(&ctor_val, regs, args_start_flat, arg_count, saved_ba);

                RT_BYTECODE.with(|b| b.set(saved_ba));
                result
            }

            // ── Dynamic scope lookup ────────────────────────────────────
            Opcode::LdaLookupSlot => None,

            // ── ToString ────────────────────────────────────────────────
            Opcode::ToString => {
                let val = jit_i64_to_jsvalue(acc);
                let s = match &val {
                    JsValue::String(_) => return Some(acc),
                    JsValue::Smi(n) => n.to_string(),
                    JsValue::Boolean(true) => "true".to_string(),
                    JsValue::Boolean(false) => "false".to_string(),
                    JsValue::Undefined => "undefined".to_string(),
                    JsValue::Null => "null".to_string(),
                    JsValue::HeapNumber(f) => {
                        if f.is_nan() {
                            "NaN".to_string()
                        } else if f.is_infinite() {
                            if *f > 0.0 {
                                "Infinity".to_string()
                            } else {
                                "-Infinity".to_string()
                            }
                        } else {
                            format!("{f}")
                        }
                    }
                    _ => return None,
                };
                Some(jsvalue_to_jit_i64(JsValue::String(Rc::from(s.as_str()))))
            }

            // ── TypeOf ──────────────────────────────────────────────────
            Opcode::TypeOf => {
                let val = jit_i64_to_jsvalue(acc);
                let type_str = match &val {
                    JsValue::Undefined => "undefined",
                    JsValue::Null => "object",
                    JsValue::Boolean(_) => "boolean",
                    JsValue::Smi(_) | JsValue::HeapNumber(_) => "number",
                    JsValue::String(_) => "string",
                    JsValue::NativeFunction(_) | JsValue::Function(_) => "function",
                    _ => "object",
                };
                Some(jsvalue_to_jit_i64(JsValue::String(Rc::from(type_str))))
            }

            // ── Bit shift operations ────────────────────────────────────
            Opcode::ShiftLeft => {
                let lhs_flat = operand1 as usize;
                // SAFETY: lhs_flat was computed by the compiler from a valid
                // bytecode register operand and is within bounds.
                let lhs_i64 = unsafe { *regs.add(lhs_flat) };
                let lhs = jit_i64_to_jsvalue(lhs_i64);
                let rhs = jit_i64_to_jsvalue(acc);
                match (&lhs, &rhs) {
                    (JsValue::Smi(l), JsValue::Smi(r)) => {
                        let result = l << ((*r as u32) & 0x1f);
                        Some(i64::from(result))
                    }
                    _ => None,
                }
            }

            Opcode::ShiftRight => {
                let lhs_flat = operand1 as usize;
                // SAFETY: valid index (see ShiftLeft above).
                let lhs_i64 = unsafe { *regs.add(lhs_flat) };
                let lhs = jit_i64_to_jsvalue(lhs_i64);
                let rhs = jit_i64_to_jsvalue(acc);
                match (&lhs, &rhs) {
                    (JsValue::Smi(l), JsValue::Smi(r)) => {
                        let result = l >> ((*r as u32) & 0x1f);
                        Some(i64::from(result))
                    }
                    _ => None,
                }
            }

            Opcode::ShiftRightLogical => {
                let lhs_flat = operand1 as usize;
                // SAFETY: valid index (see ShiftLeft above).
                let lhs_i64 = unsafe { *regs.add(lhs_flat) };
                let lhs = jit_i64_to_jsvalue(lhs_i64);
                let rhs = jit_i64_to_jsvalue(acc);
                match (&lhs, &rhs) {
                    (JsValue::Smi(l), JsValue::Smi(r)) => {
                        let result = (*l as u32) >> ((*r as u32) & 0x1f);
                        if result <= i32::MAX as u32 {
                            Some(i64::from(result as i32))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }

            // ── TestTypeOf ────────────────────────────────────────────────
            // Flag: 0=number, 1=string, 2=symbol, 3=boolean, 4=bigint,
            //       5=undefined, 6=function, 7=object
            Opcode::TestTypeOf => {
                let val = jit_i64_to_jsvalue(acc);
                let flag = operand1 as u8;
                let result = match flag {
                    0 => matches!(val, JsValue::Smi(_) | JsValue::HeapNumber(_)),
                    1 => matches!(val, JsValue::String(_)),
                    2 => false, // symbols not yet tracked in JIT
                    3 => matches!(val, JsValue::Boolean(_)),
                    4 => false, // BigInt not yet supported in JIT
                    5 => matches!(val, JsValue::Undefined | JsValue::TheHole),
                    6 => matches!(val, JsValue::NativeFunction(_) | JsValue::Function(_)),
                    7 => matches!(
                        val,
                        JsValue::Null | JsValue::PlainObject(_) | JsValue::Array(_)
                    ),
                    _ => return None,
                };
                Some(if result { JIT_TRUE } else { JIT_FALSE })
            }

            // ── TestInstanceOf ───────────────────────────────────────────
            Opcode::TestInstanceOf => {
                // operand1 = register flat index of RHS constructor
                let rhs_i64 = unsafe { *regs.add(operand1 as usize) };
                let _rhs = jit_i64_to_jsvalue(rhs_i64);
                let _lhs = jit_i64_to_jsvalue(acc);
                // Full instanceof requires @@hasInstance / prototype chain
                // walk — deopt for now to the interpreter.
                None
            }

            // ── TestIn ───────────────────────────────────────────────────
            Opcode::TestIn => {
                let obj_i64 = unsafe { *regs.add(operand1 as usize) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let key = jit_i64_to_jsvalue(acc);
                match (&key, &obj) {
                    (JsValue::String(k), JsValue::PlainObject(map)) => {
                        let found = map.borrow().contains_key(k.as_ref());
                        Some(if found { JIT_TRUE } else { JIT_FALSE })
                    }
                    (JsValue::Smi(idx), JsValue::Array(arr)) => {
                        let found = (*idx as usize) < arr.borrow().len();
                        Some(if found { JIT_TRUE } else { JIT_FALSE })
                    }
                    _ => None,
                }
            }

            // ── ThrowReferenceErrorIfHole ────────────────────────────────
            Opcode::ThrowReferenceErrorIfHole => {
                let val = jit_i64_to_jsvalue(acc);
                if matches!(val, JsValue::TheHole) {
                    // TDZ violation — deopt to interpreter which will throw.
                    None
                } else {
                    Some(acc)
                }
            }

            // ── LdaTheHole ──────────────────────────────────────────────
            Opcode::LdaTheHole => Some(jsvalue_to_jit_i64(JsValue::TheHole)),

            // ── ToNumber / ToNumeric ─────────────────────────────────────
            Opcode::ToNumber | Opcode::ToNumeric => {
                let val = jit_i64_to_jsvalue(acc);
                match &val {
                    JsValue::Smi(_) | JsValue::HeapNumber(_) => Some(acc),
                    JsValue::Boolean(true) => Some(1_i64),
                    JsValue::Boolean(false) => Some(0_i64),
                    JsValue::Undefined => Some(jsvalue_to_jit_i64(JsValue::HeapNumber(f64::NAN))),
                    JsValue::Null => Some(0_i64),
                    _ => None,
                }
            }

            // ── CreateClosure ────────────────────────────────────────────
            Opcode::CreateClosure => {
                let cp_idx = operand1 as u32;
                // SAFETY: RT_BYTECODE is always set by jit_runtime_setup
                // before any JIT code runs, and points at a live
                // BytecodeArray for the duration of the JIT execution.
                let ba_ptr = RT_BYTECODE.with(|b| b.get());
                if ba_ptr.is_null() {
                    return None;
                }
                let ba = unsafe { &*ba_ptr };
                let entry = ba.get_constant(cp_idx)?;
                let ConstantPoolEntry::Function(inner_ba) = entry else {
                    return None;
                };
                // Capture the current context (if any) for the closure.
                let closure_ctx = RT_CONTEXT.with(|ctx_cell| {
                    let ctx_opt = ctx_cell.borrow();
                    ctx_opt.as_ref().map(Rc::clone)
                });
                let func = Rc::new(inner_ba.clone_for_closure(closure_ctx));
                let val = JsValue::Function(func);
                Some(jsvalue_to_jit_i64(val))
            }

            // ── CreateFunctionContext ─────────────────────────────────────
            Opcode::CreateFunctionContext => {
                let slot_count = operand1 as usize;
                // Parent context comes from RT_CONTEXT (the current
                // closure's captured context chain).
                let parent = RT_CONTEXT.with(|ctx_cell| {
                    let ctx_opt = ctx_cell.borrow();
                    ctx_opt.as_ref().map(Rc::clone)
                });
                let js_ctx = JsContext::new(slot_count, parent);
                let val = JsValue::Context(js_ctx);
                Some(jsvalue_to_jit_i64(val))
            }

            // ── PushContext ──────────────────────────────────────────────
            // operand1 = flat register index to save old context into
            Opcode::PushContext => {
                let reg_flat = operand1 as usize;
                // Save the current context into the specified register.
                let old_ctx = RT_CONTEXT.with(|ctx_cell| {
                    let ctx_opt = ctx_cell.borrow();
                    ctx_opt
                        .as_ref()
                        .map(|c| JsValue::Context(Rc::clone(c)))
                        .unwrap_or(JsValue::Undefined)
                });
                let old_i64 = jsvalue_to_jit_i64(old_ctx);
                // SAFETY: reg_flat was computed by the compiler from a
                // valid bytecode register operand.
                unsafe { *regs.add(reg_flat) = old_i64 };
                // Set the new context from the accumulator.
                let new_ctx_val = jit_i64_to_jsvalue(acc);
                if let JsValue::Context(c) = new_ctx_val {
                    RT_CONTEXT.with(|ctx_cell| {
                        *ctx_cell.borrow_mut() = Some(c);
                    });
                }
                Some(acc)
            }

            // ── PopContext ───────────────────────────────────────────────
            // operand1 = flat register index holding the saved context
            Opcode::PopContext => {
                let reg_flat = operand1 as usize;
                // SAFETY: reg_flat was computed by the compiler.
                let saved_i64 = unsafe { *regs.add(reg_flat) };
                let saved = jit_i64_to_jsvalue(saved_i64);
                match saved {
                    JsValue::Context(c) => {
                        RT_CONTEXT.with(|ctx_cell| {
                            *ctx_cell.borrow_mut() = Some(c);
                        });
                    }
                    _ => {
                        RT_CONTEXT.with(|ctx_cell| {
                            *ctx_cell.borrow_mut() = None;
                        });
                    }
                }
                Some(acc)
            }

            // ── Div / Mod ───────────────────────────────────────────────
            Opcode::Div => {
                let lhs_i64 = unsafe { *regs.add(operand1 as usize) };
                let lhs = jit_i64_to_jsvalue(lhs_i64);
                let rhs = jit_i64_to_jsvalue(acc);
                match (&lhs, &rhs) {
                    (JsValue::Smi(a), JsValue::Smi(b)) if *b != 0 => {
                        if *a % *b == 0 {
                            Some(jsvalue_to_jit_i64(JsValue::Smi(*a / *b)))
                        } else {
                            Some(jsvalue_to_jit_i64(JsValue::HeapNumber(
                                *a as f64 / *b as f64,
                            )))
                        }
                    }
                    _ => None,
                }
            }

            Opcode::Mod => {
                let lhs_i64 = unsafe { *regs.add(operand1 as usize) };
                let lhs = jit_i64_to_jsvalue(lhs_i64);
                let rhs = jit_i64_to_jsvalue(acc);
                match (&lhs, &rhs) {
                    (JsValue::Smi(a), JsValue::Smi(b)) if *b != 0 => {
                        Some(jsvalue_to_jit_i64(JsValue::Smi(*a % *b)))
                    }
                    _ => None,
                }
            }

            // ── CallProperty0 ────────────────────────────────────────────
            // operand1 = flat register of the callee, operand2 = flat
            // register of the receiver.  Zero additional arguments.
            Opcode::CallProperty0 => {
                let callee_i64 = unsafe { *regs.add(operand1 as usize) };
                let receiver_i64 = unsafe { *regs.add(operand2 as usize) };
                call_property0_inner(callee_i64, receiver_i64)
            }

            // ── DivSmi / ModSmi ──────────────────────────────────────────
            Opcode::DivSmi => {
                let rhs = operand1 as i32;
                let lhs = jit_i64_to_jsvalue(acc);
                match lhs {
                    JsValue::Smi(a) if rhs != 0 => {
                        let a64 = i64::from(a);
                        let r64 = i64::from(rhs);
                        if a64 % r64 == 0 {
                            Some(jsvalue_to_jit_i64(JsValue::Smi((a64 / r64) as i32)))
                        } else {
                            Some(jsvalue_to_jit_i64(JsValue::HeapNumber(
                                a64 as f64 / r64 as f64,
                            )))
                        }
                    }
                    _ => None,
                }
            }

            Opcode::ModSmi => {
                let rhs = operand1 as i32;
                let lhs = jit_i64_to_jsvalue(acc);
                match lhs {
                    JsValue::Smi(a) if rhs != 0 => Some(jsvalue_to_jit_i64(JsValue::Smi(a % rhs))),
                    _ => None,
                }
            }

            // ── CallProperty1 ────────────────────────────────────────────
            // operand1 encodes callee_flat (low 16 bits) and receiver_flat
            // (bits 16..31).  operand2 = flat register of arg0.
            Opcode::CallProperty1 => {
                let callee_flat = (operand1 & 0xFFFF) as usize;
                let receiver_flat = ((operand1 >> 16) & 0xFFFF) as usize;
                let arg0_flat = operand2 as usize;
                let callee_i64 = unsafe { *regs.add(callee_flat) };
                let receiver_i64 = unsafe { *regs.add(receiver_flat) };
                let arg0_i64 = unsafe { *regs.add(arg0_flat) };
                call_property1_inner(callee_i64, receiver_i64, arg0_i64)
            }

            _ => None,
        }
    }

    /// Specialized runtime stub for `LdaNamedProperty`.
    ///
    /// Avoids the generic opcode dispatch overhead of
    /// [`jit_runtime_trampoline`].  Maintains a simple shape-based
    /// inline cache so that repeated accesses to the same property on
    /// objects with an unchanged shape use O(1) offset lookups instead
    /// of the full `PropertyMap::get` path.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – the JIT i64 encoding of the receiver object.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    /// * `RDX` (`_feedback_slot`) – reserved for future feedback-vector use.
    ///
    /// Returns the property value as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_lda_named_property(
        obj_i64: i64,
        name_idx: u32,
        _feedback_slot: u32,
    ) -> i64 {
        lda_named_property_inner(obj_i64, name_idx).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_lda_named_property`].
    ///
    /// Uses a two-phase approach:
    /// 1. **Fast path** – borrow the heap object without cloning it and
    ///    check the inline cache.  On IC hit with a primitive result the
    ///    value is returned immediately (zero Rc clones).
    /// 2. **Slow path** – clone-based fallback for IC misses and complex
    ///    result types that need a heap-handle allocation.
    fn lda_named_property_inner(obj_i64: i64, name_idx: u32) -> Option<i64> {
        if !is_heap_handle(obj_i64) {
            return None;
        }
        let idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        let lda_slot = (name_idx & 63) as usize;

        let ptrs = RT_PTRS.with(|p| p.get());

        // ── Phase 1: IC hit via borrowed heap (no input clone) ──────
        // Returns Ok(i64) for a direct/primitive result, Err(JsValue)
        // for an object result that still needs a heap handle, or None
        // on IC miss.
        let fast: Option<Result<i64, JsValue>> = if ptrs.is_cached() {
            // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            let ic_ref = unsafe { &*ptrs.prop_ic };

            // SAFETY: JIT execution is single-threaded and no mutable borrows
            // of RT_HEAP or PropertyMaps can be active during a load-property
            // fast path.  Skipping the RefCell borrow check eliminates ~6ns of
            // overhead per stub call.
            let heap = unsafe { &*heap_ref.as_ptr() };
            let obj = heap.get(idx)?;

            match obj {
                JsValue::PlainObject(map_rc) => {
                    // SAFETY: same single-thread guarantee as above.
                    let map = unsafe { &*map_rc.as_ptr() };
                    let shape = map.shape_id();
                    let cache = unsafe { &*ic_ref.as_ptr() };

                    // Own-property IC
                    let entry = &cache.lda[lda_slot];
                    if entry.0 == name_idx && entry.1 == shape {
                        return Some(
                            match map
                                .get_by_offset(entry.2)
                                .map(encode_or_clone_ref)
                                .unwrap_or(Ok(JIT_UNDEFINED))
                            {
                                Ok(val) => val,
                                Err(obj_val) => alloc_heap_handle(obj_val),
                            },
                        );
                    }

                    // Prototype IC (value pre-encoded as i64)
                    let proto_slot = (name_idx & 15) as usize;
                    let pe = &cache.proto[proto_slot];
                    if pe.0 == name_idx && pe.1 == shape {
                        return Some(pe.2);
                    }

                    None // IC miss
                }
                JsValue::Array(arr) => {
                    let prop_name = get_rt_string_constant_ref(name_idx)?;
                    if prop_name == "length" {
                        // SAFETY: single-threaded JIT; no concurrent borrows.
                        Some(Ok(unsafe { &*arr.as_ptr() }.len() as i64))
                    } else {
                        Some(Ok(JIT_UNDEFINED))
                    }
                }
                _ => None,
            }
        } else {
            RT_HEAP.with(|heap| {
                let heap = heap.borrow();
                let obj = heap.get(idx)?;

                match obj {
                    JsValue::PlainObject(map_rc) => {
                        let map = map_rc.borrow();
                        let shape = map.shape_id();

                        RT_PROP_IC.with(|ic| {
                            let cache = ic.borrow();

                            let entry = &cache.lda[lda_slot];
                            if entry.0 == name_idx && entry.1 == shape {
                                return Some(
                                    map.get_by_offset(entry.2)
                                        .map(encode_or_clone_ref)
                                        .unwrap_or(Ok(JIT_UNDEFINED)),
                                );
                            }

                            let proto_slot = (name_idx & 15) as usize;
                            let pe = &cache.proto[proto_slot];
                            if pe.0 == name_idx && pe.1 == shape {
                                return Some(Ok(pe.2));
                            }

                            None
                        })
                    }
                    JsValue::Array(arr) => {
                        let prop_name = get_rt_string_constant_ref(name_idx)?;
                        if prop_name == "length" {
                            Some(Ok(arr.borrow().len() as i64))
                        } else {
                            Some(Ok(JIT_UNDEFINED))
                        }
                    }
                    _ => None,
                }
            })
        };

        if let Some(result) = fast {
            return Some(match result {
                Ok(val) => val,
                Err(obj_val) => alloc_heap_handle(obj_val),
            });
        }

        // ── Phase 2: IC miss — clone-based slow path ────────────────
        let obj = jit_i64_to_jsvalue(obj_i64);
        match &obj {
            JsValue::PlainObject(map_rc) => {
                let map = map_rc.borrow();
                let shape = map.shape_id();

                let prop_name = get_rt_string_constant(name_idx)?;

                if let Some(offset) = map.offset_of(&prop_name) {
                    if ptrs.is_cached() {
                        // SAFETY: cached pointer valid for thread lifetime.
                        unsafe { &*ptrs.prop_ic }.borrow_mut().lda[lda_slot] =
                            (name_idx, shape, offset);
                    } else {
                        RT_PROP_IC.with(|ic| {
                            ic.borrow_mut().lda[lda_slot] = (name_idx, shape, offset);
                        });
                    }
                    Some(
                        map.get_by_offset(offset)
                            .map(jsvalue_ref_to_jit_i64)
                            .unwrap_or(JIT_UNDEFINED),
                    )
                } else {
                    // Not on own object — walk the prototype chain.
                    let proto = map
                        .get(INTERNAL_PROTO_PROPERTY_KEY)
                        .or_else(|| map.get("__proto__"))
                        .cloned();
                    drop(map);
                    let result = jit_proto_chain_walk(proto.as_ref(), &prop_name);

                    let proto_slot = (name_idx & 15) as usize;
                    if ptrs.is_cached() {
                        unsafe { &*ptrs.prop_ic }.borrow_mut().proto[proto_slot] =
                            (name_idx, shape, result);
                    } else {
                        RT_PROP_IC.with(|ic| {
                            ic.borrow_mut().proto[proto_slot] = (name_idx, shape, result);
                        });
                    }

                    Some(result)
                }
            }
            JsValue::Array(arr) => {
                let prop_name = get_rt_string_constant(name_idx)?;
                if prop_name == "length" {
                    Some(arr.borrow().len() as i64)
                } else {
                    Some(JIT_UNDEFINED)
                }
            }
            _ => None,
        }
    }

    /// Encode a `&JsValue` as `i64` if it is a primitive (no heap
    /// allocation needed), otherwise clone the value for later
    /// [`alloc_heap_handle`].
    #[inline]
    fn encode_or_clone_ref(val: &JsValue) -> Result<i64, JsValue> {
        match val {
            JsValue::Smi(n) => Ok(i64::from(*n)),
            JsValue::Boolean(true) => Ok(JIT_TRUE),
            JsValue::Boolean(false) => Ok(JIT_FALSE),
            JsValue::Undefined => Ok(JIT_UNDEFINED),
            JsValue::Null => Ok(JIT_NULL),
            JsValue::HeapNumber(f) => {
                let f = *f;
                if f.fract() == 0.0 && f >= i32::MIN as f64 && f <= i32::MAX as f64 {
                    Ok(f as i64)
                } else {
                    Err(val.clone())
                }
            }
            _ => Err(val.clone()),
        }
    }

    /// Walk a prototype chain looking for `key`, returning the JIT i64
    /// encoding of the property value found (or [`JIT_UNDEFINED`]).
    ///
    /// Handles up to 64 prototype hops to guard against cycles.
    fn jit_proto_chain_walk(start: Option<&JsValue>, key: &str) -> i64 {
        let Some(first) = start else {
            return JIT_UNDEFINED;
        };
        let mut current = first.clone();
        for _ in 0..64 {
            if matches!(current, JsValue::Null | JsValue::Undefined) {
                break;
            }
            if let JsValue::PlainObject(ref map_rc) = current {
                let borrow = map_rc.borrow();
                if let Some(val) = borrow.get(key) {
                    return jsvalue_ref_to_jit_i64(val);
                }
                let next = borrow
                    .get(INTERNAL_PROTO_PROPERTY_KEY)
                    .or_else(|| borrow.get("__proto__"))
                    .cloned();
                drop(borrow);
                match next {
                    Some(p) => current = p,
                    None => break,
                }
            } else {
                break;
            }
        }
        JIT_UNDEFINED
    }

    /// Convert a `&JsValue` to JIT i64 *without* cloning when possible.
    ///
    /// Smi and Boolean values are encoded directly as i64 constants.
    /// Other types fall back to [`jsvalue_to_jit_i64`] which clones and
    /// allocates a heap handle.
    #[inline]
    fn jsvalue_ref_to_jit_i64(val: &JsValue) -> i64 {
        match val {
            JsValue::Smi(n) => i64::from(*n),
            JsValue::Boolean(true) => JIT_TRUE,
            JsValue::Boolean(false) => JIT_FALSE,
            JsValue::Undefined => JIT_UNDEFINED,
            JsValue::Null => JIT_NULL,
            other => jsvalue_to_jit_i64(other.clone()),
        }
    }

    // ── Specialized CallUndefinedReceiver0 stub ──────────────────────────

    /// Specialized runtime stub for `CallUndefinedReceiver0`.
    ///
    /// Eliminates generic opcode dispatch for zero-argument function
    /// calls.  Handles native functions directly and attempts JIT
    /// execution for bytecode-backed `Function` values.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – the JIT i64 encoding of the callee.
    ///
    /// Returns the call result as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_call_undefined_receiver0(callee_i64: i64) -> i64 {
        call_undefined_receiver0_inner(callee_i64).unwrap_or(JIT_DEOPT)
    }

    /// Execute a JS `Function` callee via JIT (preferred) or interpreter
    /// (fallback).
    ///
    /// 1. Eagerly compiles the callee via [`BaselineCompiler`] if it has no
    ///    JIT code yet.
    /// 2. Tries the persistent exec-cache JIT path.
    /// 3. Falls back to the interpreter so the **caller** is never forced to
    ///    deopt just because the callee was not yet compiled.
    fn call_js_function(
        ba: &Rc<BytecodeArray>,
        args: Vec<JsValue>,
        jit_args: &[i64],
        saved_ba: *const BytecodeArray,
    ) -> Option<i64> {
        // ── Fast path: exec cache already populated ─────────────────
        // On the second+ call, skip the compilation and init checks.
        // SAFETY: single-threaded JIT; no concurrent borrows of exec cache.
        let exec_cache = ba.jit_executable_cache();
        {
            let cache_ref = unsafe { &*exec_cache.as_ptr() };
            if let Some(exec) = cache_ref.as_ref() {
                return exec_jit_callee(ba, exec, jit_args, saved_ba);
            }
        }

        // ── Slow path: compile + init exec cache ────────────────────
        if ba.try_get_jit_code().is_none() && !ba.jit_baseline_has_deopted() {
            if let Ok(cc) = BaselineCompiler::compile(ba) {
                ba.store_jit_code(cc.code, cc.register_file_slots);
            } else {
                ba.mark_jit_baseline_deopted();
            }
        }

        {
            let needs_init = exec_cache.borrow().is_none();
            if needs_init {
                if let Some((code, reg_slots)) = ba.try_get_jit_code() {
                    // SAFETY: code was produced by the baseline compiler.
                    let exec = unsafe { JitExecutableCode::new(&code, reg_slots) };
                    *exec_cache.borrow_mut() = exec;
                }
            }
        }

        {
            let cache_ref = exec_cache.borrow();
            if let Some(exec) = cache_ref.as_ref() {
                return exec_jit_callee(ba, exec, jit_args, saved_ba);
            }
        }

        // ── Interpreter fallback ────────────────────────────────────
        interpreter_call_fallback(ba, args, saved_ba)
    }

    /// Execute a JIT-compiled callee with state save/restore.
    ///
    /// Uses cached TLS pointers from [`RT_PTRS`] to bypass repeated
    /// `.with()` lookups — one TLS access instead of six.
    fn exec_jit_callee(
        ba: &Rc<BytecodeArray>,
        exec: &JitExecutableCode,
        jit_args: &[i64],
        saved_ba: *const BytecodeArray,
    ) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());

        if ptrs.is_cached() {
            // SAFETY: cached pointers point to thread-local storage that
            // outlives this function call.  Single-threaded access is
            // guaranteed because JIT stubs only run on the interpreter
            // thread.  We bypass RefCell borrow checks here because:
            // 1. No concurrent mutable borrows exist during JIT execution
            // 2. The heap/context are only accessed from this thread
            // 3. This eliminates ~4 RefCell borrows per closure call
            let heap_ref = unsafe { &*ptrs.heap };
            let ctx_ref = unsafe { &*ptrs.context };
            let bc_ref = unsafe { &*ptrs.bytecode };

            // SAFETY: no active borrows; read length via raw pointer.
            let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

            // Compare callee context with current context — if they're
            // the same (common for closures in tight loops), skip the
            // expensive clone + swap + restore cycle entirely.
            let callee_ctx_raw = ba.closure_context();
            let callee_ctx_ptr = callee_ctx_raw.map(Rc::as_ptr).unwrap_or(std::ptr::null());
            // SAFETY: no active borrows; read current context pointer.
            let current_ctx_ptr = unsafe {
                (*ctx_ref.as_ptr())
                    .as_ref()
                    .map(Rc::as_ptr)
                    .unwrap_or(std::ptr::null())
            };
            let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);

            let ctx_ptr = callee_ctx_ptr as i64;
            let saved_ctx = if same_context {
                // Same context: no clone/swap needed.
                None
            } else {
                let callee_ctx = callee_ctx_raw.cloned();
                // SAFETY: no active borrows; swap context via raw pointer.
                Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
            };

            bc_ref.set(&**ba as *const BytecodeArray);

            // SAFETY: cached executable code was produced by the
            // baseline compiler and contains valid x86-64 instructions.
            let jit_result = unsafe { exec.execute(jit_args, ctx_ptr) };

            let result_val = if jit_result == JIT_DEOPT {
                None
            } else {
                jit_to_jsvalue_ext(jit_result)
            };

            bc_ref.set(saved_ba);
            // SAFETY: no active borrows; truncate/restore via raw pointer.
            unsafe { (*heap_ref.as_ptr()).truncate(heap_base) };
            if let Some(ctx) = saved_ctx {
                // SAFETY: no active borrows; restore context via raw pointer.
                unsafe { *ctx_ref.as_ptr() = ctx };
            }

            return result_val.map(jsvalue_to_jit_i64);
        }

        // Fallback: pointers not cached yet — use .with() calls.
        let heap_base = RT_HEAP.with(|h| h.borrow().len());

        let callee_ctx = ba.closure_context().cloned();
        let ctx_ptr = callee_ctx
            .as_ref()
            .map(|rc| Rc::as_ptr(rc) as i64)
            .unwrap_or(0);

        let saved_ctx = RT_CONTEXT.with(|c| std::mem::replace(&mut *c.borrow_mut(), callee_ctx));
        RT_BYTECODE.with(|b| b.set(&**ba as *const BytecodeArray));

        let jit_result = unsafe { exec.execute(jit_args, ctx_ptr) };

        let result_val = if jit_result == JIT_DEOPT {
            None
        } else {
            jit_to_jsvalue_ext(jit_result)
        };

        RT_BYTECODE.with(|b| b.set(saved_ba));
        RT_HEAP.with(|h| h.borrow_mut().truncate(heap_base));
        RT_CONTEXT.with(|c| *c.borrow_mut() = saved_ctx);

        result_val.map(jsvalue_to_jit_i64)
    }

    /// Execute a JS function via the interpreter.
    ///
    /// Used when the callee has no JIT code or the JIT code deopted.
    /// This prevents the *caller* from deopting just because the callee
    /// could not run in JIT mode.
    fn interpreter_call_fallback(
        ba: &Rc<BytecodeArray>,
        args: Vec<JsValue>,
        saved_ba: *const BytecodeArray,
    ) -> Option<i64> {
        use crate::interpreter::{Interpreter, InterpreterFrame};

        let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());

        let result = if let Some(env) = env_opt {
            let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), args, env);
            Interpreter::run(&mut frame)
        } else {
            let mut frame = InterpreterFrame::new(Rc::clone(ba), args);
            Interpreter::run(&mut frame)
        };

        // Restore RT_BYTECODE — the interpreter may have changed it
        // through its own JIT setup/teardown.
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
            unsafe { &*ptrs.bytecode }.set(saved_ba);
        } else {
            RT_BYTECODE.with(|b| b.set(saved_ba));
        }

        match result {
            Ok(val) => Some(jsvalue_to_jit_i64(val)),
            Err(_) => None,
        }
    }

    /// Inner implementation for the `Construct` runtime stub.
    ///
    /// Handles `new Ctor(args...)` by creating the `this` object, running the
    /// constructor body through the interpreter, and returning the
    /// constructed object (or the explicitly returned one).
    fn construct_inner(
        ctor_val: &JsValue,
        regs: *mut i64,
        args_start_flat: usize,
        arg_count: usize,
        saved_ba: *const BytecodeArray,
    ) -> Option<i64> {
        use crate::interpreter::{
            Interpreter, InterpreterFrame, make_construct_this, maybe_cache_construct_boilerplate,
            resolve_construct_proto, restore_closure_context,
        };

        match ctor_val {
            JsValue::Function(ba) => {
                if ba.is_arrow() {
                    return None;
                }

                // Collect args from the register file.
                let mut args = Vec::with_capacity(arg_count);
                for i in 0..arg_count {
                    // SAFETY: args_start_flat + i is within the register file.
                    let arg_i64 = unsafe { *regs.add(args_start_flat + i) };
                    args.push(jit_i64_to_jsvalue(arg_i64));
                }

                let ctor_proto = resolve_construct_proto(&JsValue::Function(Rc::clone(ba)), ba);
                let this_val = make_construct_this(ba, &ctor_proto);

                let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());
                let env = env_opt?;

                let mut callee_frame = InterpreterFrame::new_with_globals(Rc::clone(ba), args, env);
                restore_closure_context(&mut callee_frame, ba);
                callee_frame.new_target = JsValue::Function(Rc::clone(ba));
                callee_frame
                    .global_env
                    .borrow_mut()
                    .set_this(this_val.clone());

                let result = Interpreter::run(&mut callee_frame);

                RT_BYTECODE.with(|b| b.set(saved_ba));

                let val = result.ok()?;
                let constructed = match val {
                    JsValue::PlainObject(_) | JsValue::Object(_) => val,
                    _ => {
                        maybe_cache_construct_boilerplate(ba, &this_val);
                        this_val
                    }
                };
                Some(jsvalue_to_jit_i64(constructed))
            }

            JsValue::NativeFunction(f) => {
                let mut args = Vec::with_capacity(arg_count);
                for i in 0..arg_count {
                    // SAFETY: args_start_flat + i is within the register file.
                    let arg_i64 = unsafe { *regs.add(args_start_flat + i) };
                    args.push(jit_i64_to_jsvalue(arg_i64));
                }
                let result = f(args);
                RT_BYTECODE.with(|b| b.set(saved_ba));
                match result {
                    Ok(val) => Some(jsvalue_to_jit_i64(val)),
                    Err(_) => None,
                }
            }

            // Proxy, PlainObject with __call__, etc. — fall back to deopt.
            _ => None,
        }
    }

    /// Inner implementation for [`jit_runtime_call_undefined_receiver0`].
    ///
    /// Uses cached TLS pointers for the entire call path so only one
    /// `.with()` lookup is needed per closure invocation.
    fn call_undefined_receiver0_inner(callee_i64: i64) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());

        if ptrs.is_cached() {
            // SAFETY: pointers set by cache_rt_ptrs; valid for thread lifetime.
            let bc_ref = unsafe { &*ptrs.bytecode };
            let saved_ba = bc_ref.get();

            // Decode the callee without an extra jit_i64_to_jsvalue TLS access.
            if callee_i64 >= JIT_HEAP_TAG {
                let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
                let heap_ref = unsafe { &*ptrs.heap };
                // SAFETY: no concurrent mutable borrows; single-threaded JIT.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => {
                        let ba = Rc::clone(ba);
                        return call_js_function(&ba, vec![], &[], saved_ba);
                    }
                    Some(JsValue::NativeFunction(f)) => {
                        let f = Rc::clone(f);
                        let result = f(vec![]);
                        bc_ref.set(saved_ba);
                        return match result {
                            Ok(v) => Some(jsvalue_to_jit_i64(v)),
                            Err(_) => None,
                        };
                    }
                    _ => {
                        bc_ref.set(saved_ba);
                        return None;
                    }
                }
            }

            // Non-heap callee (e.g. Smi / bool) — cannot be called.
            bc_ref.set(saved_ba);
            return None;
        }

        // Slow path: pointers not cached.
        let callee = jit_i64_to_jsvalue(callee_i64);
        let saved_ba = RT_BYTECODE.with(|b| b.get());

        match &callee {
            JsValue::NativeFunction(f) => {
                let result = f(vec![]);
                RT_BYTECODE.with(|b| b.set(saved_ba));
                match result {
                    Ok(v) => Some(jsvalue_to_jit_i64(v)),
                    Err(_) => None,
                }
            }
            JsValue::Function(ba) => call_js_function(ba, vec![], &[], saved_ba),
            _ => {
                RT_BYTECODE.with(|b| b.set(saved_ba));
                None
            }
        }
    }

    // ── Specialized CallUndefinedReceiver1 stub ─────────────────────────

    /// Specialized runtime stub for `CallUndefinedReceiver1`.
    ///
    /// Calls a JS function with one argument and `undefined` as receiver.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`arg0_i64`) – JIT i64 encoding of the first argument.
    ///
    /// Returns the result as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_call_undefined_receiver1(callee_i64: i64, arg0_i64: i64) -> i64 {
        call_undefined_receiver1_inner(callee_i64, arg0_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_call_undefined_receiver1`].
    fn call_undefined_receiver1_inner(callee_i64: i64, arg0_i64: i64) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());

        if ptrs.is_cached() {
            // SAFETY: pointers set by cache_rt_ptrs; valid for thread lifetime.
            let bc_ref = unsafe { &*ptrs.bytecode };
            let saved_ba = bc_ref.get();

            if callee_i64 >= JIT_HEAP_TAG {
                let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
                let heap_ref = unsafe { &*ptrs.heap };
                // SAFETY: no concurrent mutable borrows; single-threaded JIT.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => {
                        let ba = Rc::clone(ba);
                        let arg0 = jit_i64_to_jsvalue(arg0_i64);
                        return call_js_function(&ba, vec![arg0], &[arg0_i64], saved_ba);
                    }
                    Some(JsValue::NativeFunction(f)) => {
                        let f = Rc::clone(f);
                        let arg0 = jit_i64_to_jsvalue(arg0_i64);
                        let result = f(vec![arg0]);
                        bc_ref.set(saved_ba);
                        return match result {
                            Ok(v) => Some(jsvalue_to_jit_i64(v)),
                            Err(_) => None,
                        };
                    }
                    _ => {
                        bc_ref.set(saved_ba);
                        return None;
                    }
                }
            }

            bc_ref.set(saved_ba);
            return None;
        }

        // Slow path: pointers not cached.
        let callee = jit_i64_to_jsvalue(callee_i64);
        let saved_ba = RT_BYTECODE.with(|b| b.get());

        match &callee {
            JsValue::NativeFunction(f) => {
                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                let result = f(vec![arg0]);
                RT_BYTECODE.with(|b| b.set(saved_ba));
                match result {
                    Ok(v) => Some(jsvalue_to_jit_i64(v)),
                    Err(_) => None,
                }
            }
            JsValue::Function(ba) => {
                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                call_js_function(ba, vec![arg0], &[arg0_i64], saved_ba)
            }
            _ => {
                RT_BYTECODE.with(|b| b.set(saved_ba));
                None
            }
        }
    }

    /// Specialized stub for calling a function with two arguments.
    pub extern "C" fn jit_runtime_call_undefined_receiver2(
        callee_i64: i64,
        arg0_i64: i64,
        arg1_i64: i64,
    ) -> i64 {
        call_undefined_receiver2_inner(callee_i64, arg0_i64, arg1_i64).unwrap_or(JIT_DEOPT)
    }

    fn call_undefined_receiver2_inner(
        callee_i64: i64,
        arg0_i64: i64,
        arg1_i64: i64,
    ) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());

        if ptrs.is_cached() {
            let bc_ref = unsafe { &*ptrs.bytecode };
            let saved_ba = bc_ref.get();

            if callee_i64 >= JIT_HEAP_TAG {
                let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
                let heap_ref = unsafe { &*ptrs.heap };
                // SAFETY: no concurrent mutable borrows; single-threaded JIT.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => {
                        let ba = Rc::clone(ba);
                        let arg0 = jit_i64_to_jsvalue(arg0_i64);
                        let arg1 = jit_i64_to_jsvalue(arg1_i64);
                        return call_js_function(
                            &ba,
                            vec![arg0, arg1],
                            &[arg0_i64, arg1_i64],
                            saved_ba,
                        );
                    }
                    Some(JsValue::NativeFunction(f)) => {
                        let f = Rc::clone(f);
                        let arg0 = jit_i64_to_jsvalue(arg0_i64);
                        let arg1 = jit_i64_to_jsvalue(arg1_i64);
                        let result = f(vec![arg0, arg1]);
                        bc_ref.set(saved_ba);
                        return match result {
                            Ok(v) => Some(jsvalue_to_jit_i64(v)),
                            Err(_) => None,
                        };
                    }
                    _ => {
                        bc_ref.set(saved_ba);
                        return None;
                    }
                }
            }

            bc_ref.set(saved_ba);
            return None;
        }

        let callee = jit_i64_to_jsvalue(callee_i64);
        let saved_ba = RT_BYTECODE.with(|b| b.get());

        match &callee {
            JsValue::NativeFunction(f) => {
                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                let arg1 = jit_i64_to_jsvalue(arg1_i64);
                let result = f(vec![arg0, arg1]);
                RT_BYTECODE.with(|b| b.set(saved_ba));
                match result {
                    Ok(v) => Some(jsvalue_to_jit_i64(v)),
                    Err(_) => None,
                }
            }
            JsValue::Function(ba) => {
                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                let arg1 = jit_i64_to_jsvalue(arg1_i64);
                call_js_function(ba, vec![arg0, arg1], &[arg0_i64, arg1_i64], saved_ba)
            }
            _ => {
                RT_BYTECODE.with(|b| b.set(saved_ba));
                None
            }
        }
    }

    /// Specialized runtime stub for `LdaGlobal`.
    ///
    /// Avoids the generic opcode dispatch overhead of
    /// [`jit_runtime_trampoline`].  Uses `RT_GLOBAL` thread-local
    /// directly for O(1) indexed access on IC hit.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`name_idx`) – constant-pool index of the variable name.
    ///
    /// Returns the variable value as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_lda_global(name_idx: i64) -> i64 {
        lda_global_inner(name_idx as u32).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_lda_global`].
    fn lda_global_inner(name_idx: u32) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());
        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let g_ref = if ptrs.is_cached() {
            unsafe { &*ptrs.global }
        } else {
            return RT_GLOBAL.with(|g| lda_global_from_ref(g, name_idx));
        };

        lda_global_from_ref(g_ref, name_idx)
    }

    /// Shared global-load logic used by both the cached-pointer and
    /// `.with()` paths.
    fn lda_global_from_ref(g: &RefCell<JitGlobalState>, name_idx: u32) -> Option<i64> {
        // SAFETY: JIT execution is single-threaded.  No concurrent mutable
        // borrows of the global state or environment can be active during a
        // load-global fast path.
        let state = unsafe { &*g.as_ptr() };
        let env_rc = state.env.as_ref()?;
        let env = unsafe { &*env_rc.as_ptr() };

        // Fast path: direct-mapped IC hit.
        let entry = &state.ic[(name_idx & 63) as usize];
        if entry.0 == name_idx {
            let (slot_idx, cached_gen) = (entry.1, entry.2);
            if env.generation() == cached_gen && slot_idx < env.slot_count() {
                let value = env.get_by_index(slot_idx);
                if *value != JsValue::TheHole {
                    return Some(jsvalue_ref_to_jit_i64(value));
                }
            }
        }

        // Slow path: HashMap lookup.
        let name = get_rt_string_constant(name_idx)?;
        let value = env.get(&name).unwrap_or(&JsValue::Undefined);
        let result = jsvalue_ref_to_jit_i64(value);

        // Populate IC — need mutable borrow; must use safe borrow here.
        let slot_idx = env.slot_index_for(&name);
        let cur_gen = env.generation();
        if let Some(idx) = slot_idx {
            g.borrow_mut().ic[(name_idx & 63) as usize] = (name_idx, idx, cur_gen);
        }

        Some(result)
    }

    // ── Specialized StaGlobal stub ──────────────────────────────────────

    /// Specialized runtime stub for `StaGlobal`.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`name_idx`) – constant-pool index of the variable name.
    /// * `RSI` (`value_i64`) – the JIT i64 encoding of the value to store.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_global(name_idx: i64, value_i64: i64) -> i64 {
        sta_global_inner(name_idx as u32, value_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_sta_global`].
    fn sta_global_inner(name_idx: u32, value_i64: i64) -> Option<i64> {
        let value = jit_i64_to_jsvalue(value_i64);

        let ptrs = RT_PTRS.with(|p| p.get());
        let g_ref = if ptrs.is_cached() {
            // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
            unsafe { &*ptrs.global }
        } else {
            return RT_GLOBAL.with(|g| sta_global_from_ref(g, name_idx, value_i64, value));
        };

        sta_global_from_ref(g_ref, name_idx, value_i64, value)
    }

    /// Shared global-store logic used by both the cached-pointer and
    /// `.with()` paths.
    fn sta_global_from_ref(
        g: &RefCell<JitGlobalState>,
        name_idx: u32,
        value_i64: i64,
        value: JsValue,
    ) -> Option<i64> {
        // SAFETY: JIT execution is single-threaded. No concurrent borrows.
        let state = unsafe { &*g.as_ptr() };
        let env_rc = state.env.as_ref()?;
        let env = unsafe { &mut *env_rc.as_ptr() };

        // Fast path: direct-mapped IC hit — store by index.
        let entry = &state.ic[(name_idx & 63) as usize];
        if entry.0 == name_idx {
            let (slot_idx, cached_gen) = (entry.1, entry.2);
            if env.generation() == cached_gen && slot_idx < env.slot_count() {
                let name = get_rt_string_constant(name_idx)?;
                env.store_by_index_fast(slot_idx, &name, value);
                return Some(value_i64);
            }
        }

        // Slow path: insert via HashMap.
        let name = get_rt_string_constant(name_idx)?;
        let slot_idx = env.slot_index_for(&name);
        if let Some(idx) = slot_idx {
            env.store_by_index_fast(idx, &name, value);
        } else {
            env.insert(name.clone(), value);
        }

        // Populate / update IC.
        let cur_gen = env.generation();
        let new_slot_idx = env.slot_index_for(&name);
        // SAFETY: single-threaded JIT; update IC via raw pointer.
        if let Some(idx) = new_slot_idx {
            unsafe { (*g.as_ptr()).ic[(name_idx & 63) as usize] = (name_idx, idx, cur_gen) };
        }

        Some(value_i64)
    }

    // ── Specialized LdaKeyedProperty stub ───────────────────────────────

    /// Specialized runtime stub for `LdaKeyedProperty`.
    ///
    /// Skips generic opcode dispatch.  Handles `Array[Smi]` and
    /// `PlainObject[Smi|String]` element access directly.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`key_i64`) – JIT i64 encoding of the key (accumulator).
    ///
    /// Returns the element value as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_lda_keyed_property(obj_i64: i64, key_i64: i64) -> i64 {
        lda_keyed_property_inner(obj_i64, key_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_lda_keyed_property`].
    fn lda_keyed_property_inner(obj_i64: i64, key_i64: i64) -> Option<i64> {
        // Fast path: Array[Smi] without cloning the input array.
        if is_heap_handle(obj_i64) && !is_heap_handle(key_i64) {
            if let Some(JsValue::Smi(smi_key)) = super::jit_to_jsvalue(key_i64) {
                if smi_key >= 0 {
                    let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
                    let ptrs = RT_PTRS.with(|p| p.get());
                    let fast = if ptrs.is_cached() {
                        // SAFETY: cached pointers set by cache_rt_ptrs;
                        // valid for thread lifetime.
                        let heap_ref = unsafe { &*ptrs.heap };
                        // SAFETY: single-threaded JIT; no concurrent mutable
                        // borrows during a load-keyed fast path.
                        let heap = unsafe { &*heap_ref.as_ptr() };
                        match heap.get(obj_idx)? {
                            JsValue::Array(arr) => {
                                let borrow = unsafe { &*arr.as_ptr() };
                                match borrow.get(smi_key as usize) {
                                    Some(v) if !matches!(v, JsValue::TheHole) => {
                                        Some(encode_or_clone_ref(v))
                                    }
                                    _ => Some(Ok(JIT_UNDEFINED)),
                                }
                            }
                            JsValue::PlainObject(map_rc) => {
                                let key_str = smi_key.to_string();
                                let val = unsafe { &*map_rc.as_ptr() }
                                    .get(&key_str)
                                    .map(encode_or_clone_ref)
                                    .unwrap_or(Ok(JIT_UNDEFINED));
                                Some(val)
                            }
                            _ => None,
                        }
                    } else {
                        RT_HEAP.with(|heap| {
                            let heap = heap.borrow();
                            match heap.get(obj_idx)? {
                                JsValue::Array(arr) => {
                                    let borrow = arr.borrow();
                                    match borrow.get(smi_key as usize) {
                                        Some(v) if !matches!(v, JsValue::TheHole) => {
                                            Some(encode_or_clone_ref(v))
                                        }
                                        _ => Some(Ok(JIT_UNDEFINED)),
                                    }
                                }
                                JsValue::PlainObject(map_rc) => {
                                    let key_str = smi_key.to_string();
                                    let val = map_rc
                                        .borrow()
                                        .get(&key_str)
                                        .map(encode_or_clone_ref)
                                        .unwrap_or(Ok(JIT_UNDEFINED));
                                    Some(val)
                                }
                                _ => None,
                            }
                        })
                    };
                    if let Some(result) = fast {
                        return Some(match result {
                            Ok(val) => val,
                            Err(obj_val) => alloc_heap_handle(obj_val),
                        });
                    }
                }
            }
        }

        // Slow path: clone-based fallback.
        let obj = jit_i64_to_jsvalue(obj_i64);
        let key = jit_i64_to_jsvalue(key_i64);

        match (&obj, &key) {
            (JsValue::Array(arr), JsValue::Smi(idx)) if *idx >= 0 => {
                let i = *idx as usize;
                let borrow = arr.borrow();
                match borrow.get(i) {
                    Some(v) if !matches!(v, JsValue::TheHole) => Some(jsvalue_ref_to_jit_i64(v)),
                    _ => Some(JIT_UNDEFINED),
                }
            }
            (JsValue::PlainObject(map_rc), JsValue::Smi(idx)) if *idx >= 0 => {
                let key_str = idx.to_string();
                Some(
                    map_rc
                        .borrow()
                        .get(&key_str)
                        .map(jsvalue_ref_to_jit_i64)
                        .unwrap_or(JIT_UNDEFINED),
                )
            }
            (JsValue::PlainObject(map_rc), JsValue::String(s)) => Some(
                map_rc
                    .borrow()
                    .get(&**s)
                    .map(jsvalue_ref_to_jit_i64)
                    .unwrap_or(JIT_UNDEFINED),
            ),
            _ => None,
        }
    }

    // ── Specialized StaKeyedProperty stub ───────────────────────────────

    /// Specialized runtime stub for `StaKeyedProperty`.
    ///
    /// Handles `Array[Smi]` and `PlainObject[Smi|String]` element
    /// stores directly, skipping generic opcode dispatch.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`key_i64`) – JIT i64 encoding of the key.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value (accumulator).
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_keyed_property(
        obj_i64: i64,
        key_i64: i64,
        value_i64: i64,
    ) -> i64 {
        sta_keyed_property_inner(obj_i64, key_i64, value_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_sta_keyed_property`].
    fn sta_keyed_property_inner(obj_i64: i64, key_i64: i64, value_i64: i64) -> Option<i64> {
        // Fast path: Array[Smi] = primitive, without cloning the array.
        // The value must NOT be a heap handle (would conflict with the
        // immutable RT_HEAP borrow used to resolve the receiver).
        if is_heap_handle(obj_i64) && !is_heap_handle(key_i64) && !is_heap_handle(value_i64) {
            if let (Some(JsValue::Smi(smi_key)), Some(value)) = (
                super::jit_to_jsvalue(key_i64),
                super::jit_to_jsvalue(value_i64),
            ) {
                if smi_key >= 0 {
                    let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
                    let ptrs = RT_PTRS.with(|p| p.get());
                    let fast = if ptrs.is_cached() {
                        // SAFETY: cached pointers set by cache_rt_ptrs;
                        // valid for thread lifetime.
                        let heap_ref = unsafe { &*ptrs.heap };
                        // SAFETY: single-threaded JIT; no concurrent
                        // mutable borrows of RT_HEAP during keyed store.
                        let heap = unsafe { &*heap_ref.as_ptr() };
                        match heap.get(obj_idx)? {
                            JsValue::Array(arr) => {
                                let i = smi_key as usize;
                                let mut v = arr.borrow_mut();
                                if i >= v.len() {
                                    let cur_len = v.len();
                                    let new_cap = (i + 1).next_power_of_two();
                                    v.reserve(new_cap - cur_len);
                                    v.resize(i + 1, JsValue::TheHole);
                                }
                                v[i] = value;
                                Some(())
                            }
                            JsValue::PlainObject(map_rc) => {
                                let key_str = smi_key.to_string();
                                map_rc.borrow_mut().insert(key_str, value);
                                Some(())
                            }
                            _ => None,
                        }
                    } else {
                        RT_HEAP.with(|heap| {
                            let heap = heap.borrow();
                            match heap.get(obj_idx)? {
                                JsValue::Array(arr) => {
                                    let i = smi_key as usize;
                                    let mut v = arr.borrow_mut();
                                    if i >= v.len() {
                                        let cur_len = v.len();
                                        let new_cap = (i + 1).next_power_of_two();
                                        v.reserve(new_cap - cur_len);
                                        v.resize(i + 1, JsValue::TheHole);
                                    }
                                    v[i] = value;
                                    Some(())
                                }
                                JsValue::PlainObject(map_rc) => {
                                    let key_str = smi_key.to_string();
                                    map_rc.borrow_mut().insert(key_str, value);
                                    Some(())
                                }
                                _ => None,
                            }
                        })
                    };
                    if fast.is_some() {
                        return Some(value_i64);
                    }
                }
            }
        }

        // Slow path: clone-based fallback.
        let obj = jit_i64_to_jsvalue(obj_i64);
        let key = jit_i64_to_jsvalue(key_i64);
        let value = jit_i64_to_jsvalue(value_i64);

        match (&obj, &key) {
            (JsValue::Array(arr), JsValue::Smi(idx)) if *idx >= 0 => {
                let i = *idx as usize;
                let mut v = arr.borrow_mut();
                if i >= v.len() {
                    let cur_len = v.len();
                    let new_cap = (i + 1).next_power_of_two();
                    v.reserve(new_cap - cur_len);
                    v.resize(i + 1, JsValue::TheHole);
                }
                v[i] = value;
            }
            (JsValue::PlainObject(map_rc), JsValue::Smi(idx)) if *idx >= 0 => {
                let key_str = idx.to_string();
                map_rc.borrow_mut().insert(key_str, value);
            }
            (JsValue::PlainObject(map_rc), JsValue::String(s)) => {
                map_rc.borrow_mut().insert(s.to_string(), value);
            }
            _ => return None,
        }
        Some(value_i64)
    }

    // ── Specialized StaNamedProperty stub ───────────────────────────────

    /// Specialized runtime stub for `StaNamedProperty`.
    ///
    /// Stores the accumulator value as a named property on a
    /// `PlainObject`, skipping the generic opcode dispatch trampoline.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver object.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    ///
    /// Returns `value_i64` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_named_property(
        obj_i64: i64,
        name_idx: u32,
        value_i64: i64,
    ) -> i64 {
        sta_named_property_inner(obj_i64, name_idx, value_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_sta_named_property`].
    fn sta_named_property_inner(obj_i64: i64, name_idx: u32, value_i64: i64) -> Option<i64> {
        let obj = jit_i64_to_jsvalue(obj_i64);
        let value = jit_i64_to_jsvalue(value_i64);

        match &obj {
            JsValue::PlainObject(map_rc) => {
                let prop_name = get_rt_string_constant(name_idx)?;
                map_rc.borrow_mut().insert(prop_name.to_string(), value);
                Some(value_i64)
            }
            _ => None,
        }
    }

    // ── Specialized CallProperty0 stub ──────────────────────────────────

    /// Specialized runtime stub for `CallProperty0`.
    ///
    /// Handles zero-argument `NativeFunction(receiver)` calls directly.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`receiver_i64`) – JIT i64 encoding of the receiver.
    ///
    /// Returns the call result as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_call_property0(callee_i64: i64, receiver_i64: i64) -> i64 {
        call_property0_inner(callee_i64, receiver_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_call_property0`].
    ///
    /// Handles fast array methods (e.g. `arr.pop()`) and generic
    /// `NativeFunction` calls.
    fn call_property0_inner(callee_i64: i64, receiver_i64: i64) -> Option<i64> {
        // ── Fast path: use cached heap to detect fast array methods ──
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() && is_heap_handle(callee_i64) {
            let callee_idx = (callee_i64 - JIT_HEAP_TAG) as usize;
            // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            // SAFETY: single-threaded JIT; no concurrent mutable borrows.
            let heap = unsafe { &*heap_ref.as_ptr() };

            if let Some(JsValue::PlainObject(map_rc)) = heap.get(callee_idx) {
                // SAFETY: single-threaded, no concurrent mutation of callee.
                let map = unsafe { &*map_rc.as_ptr() };
                if let Some(JsValue::String(method_name)) = map.get("\0stator.fast_array_method") {
                    if is_heap_handle(receiver_i64) {
                        let recv_idx = (receiver_i64 - JIT_HEAP_TAG) as usize;
                        if let Some(JsValue::Array(arr)) = heap.get(recv_idx) {
                            match method_name.as_ref() {
                                "pop" => {
                                    let val = arr.borrow_mut().pop().unwrap_or(JsValue::Undefined);
                                    return Some(jsvalue_to_jit_i64(val));
                                }
                                "shift" => {
                                    let val = if arr.borrow().is_empty() {
                                        JsValue::Undefined
                                    } else {
                                        arr.borrow_mut().remove(0)
                                    };
                                    return Some(jsvalue_to_jit_i64(val));
                                }
                                _ => {}
                            }
                        }
                    }
                    return None;
                }

                // PlainObject with __call__.
                if let Some(JsValue::NativeFunction(nf)) = map.get("__call__") {
                    let nf = Rc::clone(nf);
                    drop(map);
                    let receiver = if is_heap_handle(receiver_i64) {
                        heap.get((receiver_i64 - JIT_HEAP_TAG) as usize)
                            .cloned()
                            .unwrap_or(JsValue::Undefined)
                    } else {
                        super::jit_to_jsvalue(receiver_i64).unwrap_or(JsValue::Undefined)
                    };
                    drop(heap);
                    return match nf(vec![receiver]) {
                        Ok(val) => Some(jsvalue_to_jit_i64(val)),
                        Err(_) => None,
                    };
                }
                drop(map);
            }

            if let Some(JsValue::NativeFunction(nf)) = heap.get(callee_idx) {
                let nf = Rc::clone(nf);
                let receiver = if is_heap_handle(receiver_i64) {
                    heap.get((receiver_i64 - JIT_HEAP_TAG) as usize)
                        .cloned()
                        .unwrap_or(JsValue::Undefined)
                } else {
                    super::jit_to_jsvalue(receiver_i64).unwrap_or(JsValue::Undefined)
                };
                drop(heap);
                return match nf(vec![receiver]) {
                    Ok(val) => Some(jsvalue_to_jit_i64(val)),
                    Err(_) => None,
                };
            }

            drop(heap);
            return None;
        }

        // ── Slow path: pointers not cached ──────────────────────────────
        let callee = jit_i64_to_jsvalue(callee_i64);
        match callee {
            JsValue::NativeFunction(nf) => {
                let receiver = jit_i64_to_jsvalue(receiver_i64);
                match nf(vec![receiver]) {
                    Ok(val) => Some(jsvalue_to_jit_i64(val)),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    // ── Specialized CallProperty1 stub ──────────────────────────────────

    /// Specialized runtime stub for `CallProperty1`.
    ///
    /// Handles `NativeFunction(receiver, arg0)` calls directly, avoiding
    /// the generic trampoline dispatch.  This is the fast path for
    /// built-in methods like `Array.prototype.push`.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`receiver_i64`) – JIT i64 encoding of the receiver.
    /// * `RDX` (`arg0_i64`) – JIT i64 encoding of the first argument.
    ///
    /// Returns the call result as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_call_property1(
        callee_i64: i64,
        receiver_i64: i64,
        arg0_i64: i64,
    ) -> i64 {
        call_property1_inner(callee_i64, receiver_i64, arg0_i64).unwrap_or(JIT_DEOPT)
    }

    /// Inner implementation for [`jit_runtime_call_property1`].
    ///
    /// Handles three callee shapes:
    /// 1. Fast array method (PlainObject with `\0stator.fast_array_method`)
    ///    — inlines `push`, `pop`, etc. directly.
    /// 2. `NativeFunction` — generic Rust-closure dispatch.
    /// 3. Anything else → DEOPT.
    fn call_property1_inner(callee_i64: i64, receiver_i64: i64, arg0_i64: i64) -> Option<i64> {
        // ── Fast path: use cached heap to avoid jit_i64_to_jsvalue clones ──
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() && is_heap_handle(callee_i64) {
            let callee_idx = (callee_i64 - JIT_HEAP_TAG) as usize;
            // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            // SAFETY: single-threaded JIT; no concurrent mutable borrows.
            let heap = unsafe { &*heap_ref.as_ptr() };

            // Check for fast array method pattern (PlainObject with marker).
            if let Some(JsValue::PlainObject(map_rc)) = heap.get(callee_idx) {
                // SAFETY: single-threaded; callee PlainObject not mutated during read.
                let map = unsafe { &*map_rc.as_ptr() };
                if let Some(JsValue::String(method_name)) = map.get("\0stator.fast_array_method") {
                    // Inline fast array method — currently supports "push".
                    if is_heap_handle(receiver_i64) {
                        let recv_idx = (receiver_i64 - JIT_HEAP_TAG) as usize;
                        if let Some(JsValue::Array(arr)) = heap.get(recv_idx) {
                            match method_name.as_ref() {
                                "push" => {
                                    let arg0 = if is_heap_handle(arg0_i64) {
                                        let a_idx = (arg0_i64 - JIT_HEAP_TAG) as usize;
                                        heap.get(a_idx).cloned().unwrap_or(JsValue::Undefined)
                                    } else {
                                        super::jit_to_jsvalue(arg0_i64)
                                            .unwrap_or(JsValue::Undefined)
                                    };
                                    let mut items = arr.borrow_mut();
                                    items.push(arg0);
                                    return Some(items.len() as i64);
                                }
                                "pop" => {
                                    let val = arr.borrow_mut().pop().unwrap_or(JsValue::Undefined);
                                    return Some(jsvalue_to_jit_i64(val));
                                }
                                _ => {}
                            }
                        }
                    }
                    // Unrecognised fast-array method or non-Array receiver — DEOPT.
                    return None;
                }

                // PlainObject with __call__ (generic callable object).
                if let Some(JsValue::NativeFunction(nf)) = map.get("__call__") {
                    let nf = Rc::clone(nf);
                    let receiver = heap
                        .get((receiver_i64 - JIT_HEAP_TAG) as usize)
                        .cloned()
                        .unwrap_or(JsValue::Undefined);
                    let arg0 = if is_heap_handle(arg0_i64) {
                        heap.get((arg0_i64 - JIT_HEAP_TAG) as usize)
                            .cloned()
                            .unwrap_or(JsValue::Undefined)
                    } else {
                        super::jit_to_jsvalue(arg0_i64).unwrap_or(JsValue::Undefined)
                    };
                    return match nf(vec![receiver, arg0]) {
                        Ok(val) => Some(jsvalue_to_jit_i64(val)),
                        Err(_) => None,
                    };
                }

                return None;
            }

            // NativeFunction path.
            if let Some(JsValue::NativeFunction(nf)) = heap.get(callee_idx) {
                let nf = Rc::clone(nf);
                let receiver = heap
                    .get((receiver_i64 - JIT_HEAP_TAG) as usize)
                    .cloned()
                    .unwrap_or(JsValue::Undefined);
                let arg0 = if is_heap_handle(arg0_i64) {
                    heap.get((arg0_i64 - JIT_HEAP_TAG) as usize)
                        .cloned()
                        .unwrap_or(JsValue::Undefined)
                } else {
                    super::jit_to_jsvalue(arg0_i64).unwrap_or(JsValue::Undefined)
                };
                return match nf(vec![receiver, arg0]) {
                    Ok(val) => Some(jsvalue_to_jit_i64(val)),
                    Err(_) => None,
                };
            }

            drop(heap);
            return None;
        }

        // ── Slow path: pointers not cached ──────────────────────────────
        let callee = jit_i64_to_jsvalue(callee_i64);
        match callee {
            JsValue::NativeFunction(nf) => {
                let receiver = jit_i64_to_jsvalue(receiver_i64);
                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                match nf(vec![receiver, arg0]) {
                    Ok(val) => Some(jsvalue_to_jit_i64(val)),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    // ── Specialized context slot stubs ──────────────────────────────────

    /// Specialized runtime stub for `LdaCurrentContextSlot` and
    /// `LdaImmutableCurrentContextSlot`.
    ///
    /// Eliminates generic opcode dispatch.  Accesses the closure context
    /// stored in `RT_CONTEXT` directly.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`slot_idx`) – context slot index.
    ///
    /// Returns the slot value as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_lda_context_slot(slot_idx: i64) -> i64 {
        RT_CONTEXT
            .with(|ctx_cell| {
                let ctx_opt = ctx_cell.borrow();
                let ctx_rc = ctx_opt.as_ref()?;
                let ctx = ctx_rc.borrow();
                Some(
                    ctx.slots
                        .get(slot_idx as usize)
                        .map(jsvalue_ref_to_jit_i64)
                        .unwrap_or(JIT_UNDEFINED),
                )
            })
            .unwrap_or(JIT_DEOPT)
    }

    /// Specialized runtime stub for `StaCurrentContextSlot`.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`slot_idx`) – context slot index.
    /// * `RSI` (`value_i64`) – JIT i64 encoding of the value (accumulator).
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_context_slot(slot_idx: i64, value_i64: i64) -> i64 {
        let value = jit_i64_to_jsvalue(value_i64);
        RT_CONTEXT
            .with(|ctx_cell| {
                let ctx_opt = ctx_cell.borrow();
                let ctx_rc = ctx_opt.as_ref()?;
                let mut ctx = ctx_rc.borrow_mut();
                let slot = slot_idx as usize;
                if slot >= ctx.slots.len() {
                    ctx.slots.resize(slot + 1, JsValue::Undefined);
                }
                ctx.slots[slot] = value;
                Some(value_i64)
            })
            .unwrap_or(JIT_DEOPT)
    }

    // ── Direct context-slot stubs (no TLS) ──────────────────────────────

    /// Load a closure-context slot using a raw `RefCell<JsContext>` pointer
    /// passed in `RDI` (RBX at the caller).  Eliminates the
    /// `RT_CONTEXT` TLS lookup.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_raw`) – raw pointer to `RefCell<JsContext>`.
    /// * `RSI` (`slot_idx`) – context slot index.
    ///
    /// Returns the slot value as `i64` in `RAX`, or [`JIT_DEOPT`].
    ///
    /// # Safety
    ///
    /// `ctx_raw` must point to a live `RefCell<JsContext>` for the
    /// duration of this call.  The caller guarantees this because the
    /// `Rc<RefCell<JsContext>>` is kept alive by `RT_CONTEXT`.
    pub extern "C" fn jit_runtime_lda_context_slot_direct(ctx_raw: i64, slot_idx: i64) -> i64 {
        if ctx_raw == 0 {
            return JIT_DEOPT;
        }
        // SAFETY: ctx_raw points to a live RefCell<JsContext> kept alive
        // by the Rc in RT_CONTEXT (set by call_js_function).
        let ctx_ref = unsafe { &*(ctx_raw as *const RefCell<JsContext>) };
        let ctx = ctx_ref.borrow();
        ctx.slots
            .get(slot_idx as usize)
            .map(jsvalue_ref_to_jit_i64)
            .unwrap_or(JIT_UNDEFINED)
    }

    /// Store a value into a closure-context slot using a raw pointer.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_raw`) – raw pointer to `RefCell<JsContext>`.
    /// * `RSI` (`slot_idx`) – context slot index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_context_slot_direct(
        ctx_raw: i64,
        slot_idx: i64,
        value_i64: i64,
    ) -> i64 {
        if ctx_raw == 0 {
            return JIT_DEOPT;
        }
        let value = jit_i64_to_jsvalue(value_i64);
        // SAFETY: see jit_runtime_lda_context_slot_direct.
        let ctx_ref = unsafe { &*(ctx_raw as *const RefCell<JsContext>) };
        let mut ctx = ctx_ref.borrow_mut();
        let slot = slot_idx as usize;
        if slot >= ctx.slots.len() {
            ctx.slots.resize(slot + 1, JsValue::Undefined);
        }
        ctx.slots[slot] = value;
        value_i64
    }

    // ── Generic arithmetic stubs for Maglev ─────────────────────────────────

    /// Generic Add: handles Smi + Smi (with overflow), HeapNumber, and
    /// string concatenation.  Deopts on complex cases.
    pub extern "C" fn jit_runtime_generic_add(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => match a.checked_add(*b) {
                Some(sum) => jsvalue_to_jit_i64(JsValue::Smi(sum)),
                None => jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 + *b as f64)),
            },
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a + *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 + *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a + *b as f64))
            }
            _ => JIT_DEOPT,
        }
    }

    /// Generic Subtract.
    pub extern "C" fn jit_runtime_generic_sub(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => match a.checked_sub(*b) {
                Some(d) => jsvalue_to_jit_i64(JsValue::Smi(d)),
                None => jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 - *b as f64)),
            },
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a - *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 - *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a - *b as f64))
            }
            _ => JIT_DEOPT,
        }
    }

    /// Generic Multiply.
    pub extern "C" fn jit_runtime_generic_mul(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => match a.checked_mul(*b) {
                Some(p) => jsvalue_to_jit_i64(JsValue::Smi(p)),
                None => jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 * *b as f64)),
            },
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a * *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 * *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a * *b as f64))
            }
            _ => JIT_DEOPT,
        }
    }

    /// Generic Divide.
    pub extern "C" fn jit_runtime_generic_div(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        let (a, b) = match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => (*a as f64, *b as f64),
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => (*a, *b),
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => (*a as f64, *b),
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => (*a, *b as f64),
            _ => return JIT_DEOPT,
        };
        let result = a / b;
        if result.fract() == 0.0 && result >= i32::MIN as f64 && result <= i32::MAX as f64 {
            jsvalue_to_jit_i64(JsValue::Smi(result as i32))
        } else {
            jsvalue_to_jit_i64(JsValue::HeapNumber(result))
        }
    }

    /// Generic Modulus.
    pub extern "C" fn jit_runtime_generic_mod(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) if *b != 0 => {
                jsvalue_to_jit_i64(JsValue::Smi(*a % *b))
            }
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) if *b != 0.0 => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a % *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) if *b != 0.0 => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 % *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) if *b != 0 => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a % *b as f64))
            }
            _ => JIT_DEOPT,
        }
    }

    /// Generic bitwise helpers — convert to i32 first.
    fn to_int32(v: &JsValue) -> Option<i32> {
        match v {
            JsValue::Smi(n) => Some(*n),
            JsValue::HeapNumber(f) => Some(*f as i32),
            JsValue::Boolean(true) => Some(1),
            JsValue::Boolean(false) | JsValue::Null => Some(0),
            JsValue::Undefined => Some(0),
            _ => None,
        }
    }

    pub extern "C" fn jit_runtime_generic_bitwise_and(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => jsvalue_to_jit_i64(JsValue::Smi(a & b)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_bitwise_or(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => jsvalue_to_jit_i64(JsValue::Smi(a | b)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_bitwise_xor(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => jsvalue_to_jit_i64(JsValue::Smi(a ^ b)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_shift_left(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => jsvalue_to_jit_i64(JsValue::Smi(a << (b as u32 & 0x1f))),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_shift_right(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => jsvalue_to_jit_i64(JsValue::Smi(a >> (b as u32 & 0x1f))),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_shift_right_logical(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => {
                let result = (a as u32) >> (b as u32 & 0x1f);
                jsvalue_to_jit_i64(JsValue::Smi(result as i32))
            }
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_negate(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::Smi(n) => match n.checked_neg() {
                Some(neg) if neg != 0 || *n == 0 => jsvalue_to_jit_i64(JsValue::Smi(neg)),
                _ => jsvalue_to_jit_i64(JsValue::HeapNumber(-(*n as f64))),
            },
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(-*f)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_increment(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::Smi(n) => match n.checked_add(1) {
                Some(inc) => jsvalue_to_jit_i64(JsValue::Smi(inc)),
                None => jsvalue_to_jit_i64(JsValue::HeapNumber(*n as f64 + 1.0)),
            },
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(*f + 1.0)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_decrement(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::Smi(n) => match n.checked_sub(1) {
                Some(dec) => jsvalue_to_jit_i64(JsValue::Smi(dec)),
                None => jsvalue_to_jit_i64(JsValue::HeapNumber(*n as f64 - 1.0)),
            },
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(*f - 1.0)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_bitwise_not(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        match to_int32(&v) {
            Some(n) => jsvalue_to_jit_i64(JsValue::Smi(!n)),
            None => JIT_DEOPT,
        }
    }

    // ── Type conversion stubs ───────────────────────────────────────────────

    /// ToString: convert a value to a string.
    pub extern "C" fn jit_runtime_tostring(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        let s: Rc<str> = match &v {
            JsValue::String(_) => return value,
            JsValue::Smi(n) => Rc::from(n.to_string().as_str()),
            JsValue::HeapNumber(f) => Rc::from(f.to_string().as_str()),
            JsValue::Boolean(true) => Rc::from("true"),
            JsValue::Boolean(false) => Rc::from("false"),
            JsValue::Null => Rc::from("null"),
            JsValue::Undefined => Rc::from("undefined"),
            _ => return JIT_DEOPT,
        };
        jsvalue_to_jit_i64(JsValue::String(s))
    }

    /// ToNumber: convert a value to a number.
    pub extern "C" fn jit_runtime_tonumber(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::Smi(_) | JsValue::HeapNumber(_) => value,
            JsValue::Boolean(true) => jsvalue_to_jit_i64(JsValue::Smi(1)),
            JsValue::Boolean(false) | JsValue::Null => jsvalue_to_jit_i64(JsValue::Smi(0)),
            JsValue::Undefined => jsvalue_to_jit_i64(JsValue::HeapNumber(f64::NAN)),
            _ => JIT_DEOPT,
        }
    }

    /// TypeOf: return the typeof string for a value.
    pub extern "C" fn jit_runtime_typeof(value: i64) -> i64 {
        let v = jit_i64_to_jsvalue(value);
        let s: &str = match &v {
            JsValue::Smi(_) | JsValue::HeapNumber(_) => "number",
            JsValue::Boolean(_) => "boolean",
            JsValue::String(_) => "string",
            JsValue::Undefined => "undefined",
            JsValue::Null => "object",
            JsValue::Function(_) | JsValue::NativeFunction(_) => "function",
            JsValue::Symbol(_) => "symbol",
            JsValue::BigInt(_) => "bigint",
            _ => "object",
        };
        jsvalue_to_jit_i64(JsValue::String(Rc::from(s)))
    }

    // ── Tagged equality stubs ───────────────────────────────────────────────

    /// Strict equality (`===`) for tagged JIT values.
    pub extern "C" fn jit_runtime_tagged_equal(left: i64, right: i64) -> i64 {
        if left == right {
            return JIT_TRUE;
        }
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        let result = match (&l, &r) {
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => *a == *b,
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => *a as f64 == *b,
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => *a == *b as f64,
            (JsValue::String(a), JsValue::String(b)) => **a == **b,
            _ => false,
        };
        if result { JIT_TRUE } else { JIT_FALSE }
    }

    /// Strict inequality (`!==`) for tagged JIT values.
    pub extern "C" fn jit_runtime_tagged_not_equal(left: i64, right: i64) -> i64 {
        if left == right {
            return JIT_FALSE;
        }
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        let result = match (&l, &r) {
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => *a == *b,
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => *a as f64 == *b,
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => *a == *b as f64,
            (JsValue::String(a), JsValue::String(b)) => **a == **b,
            _ => false,
        };
        if result { JIT_FALSE } else { JIT_TRUE }
    }

    // ── Construct stub for Maglev ───────────────────────────────────────────

    /// Simplified construct for 0 arguments — takes the constructor value
    /// directly instead of reading from the register file.
    pub extern "C" fn jit_runtime_construct0(ctor_i64: i64) -> i64 {
        construct0_inner(ctor_i64).unwrap_or(JIT_DEOPT)
    }

    fn construct0_inner(ctor_i64: i64) -> Option<i64> {
        use crate::interpreter::{
            Interpreter, InterpreterFrame, make_construct_this, maybe_cache_construct_boilerplate,
            resolve_construct_proto, restore_closure_context,
        };

        let ctor_val = jit_i64_to_jsvalue(ctor_i64);
        match &ctor_val {
            JsValue::Function(ba) => {
                if ba.is_arrow() {
                    return None;
                }

                let ctor_proto = resolve_construct_proto(&JsValue::Function(Rc::clone(ba)), ba);
                let this_val = make_construct_this(ba, &ctor_proto);

                let saved_ba = RT_BYTECODE.with(|b| b.get());
                let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());
                let env = env_opt?;

                let mut callee_frame =
                    InterpreterFrame::new_with_globals(Rc::clone(ba), vec![], env);
                restore_closure_context(&mut callee_frame, ba);
                callee_frame.new_target = JsValue::Function(Rc::clone(ba));
                callee_frame
                    .global_env
                    .borrow_mut()
                    .set_this(this_val.clone());

                let result = Interpreter::run(&mut callee_frame);

                RT_BYTECODE.with(|b| b.set(saved_ba));

                let val = result.ok()?;
                let constructed = match val {
                    JsValue::PlainObject(_) | JsValue::Object(_) => val,
                    _ => {
                        maybe_cache_construct_boilerplate(ba, &this_val);
                        this_val
                    }
                };
                Some(jsvalue_to_jit_i64(constructed))
            }
            _ => None,
        }
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
pub use jit_runtime::{
    alloc_jit_heap_handle, jit_runtime_set_context, jit_runtime_set_global_env, jit_runtime_setup,
    jit_runtime_teardown, jit_to_jsvalue_ext,
};

/// A single entry in the safepoint table.
///
/// A safepoint is any code location at which the garbage collector is allowed
/// to run.  The entry records the byte offset in the JIT code buffer and the
/// corresponding bytecode instruction index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafepointEntry {
    /// Byte offset in the JIT code buffer.
    pub code_offset: u32,
    /// Index of the bytecode instruction this safepoint corresponds to.
    pub bytecode_index: u32,
    /// Bitmask of register-file slots that hold GC-managed heap pointers at
    /// this safepoint.  Bit `i` set means slot `i` holds a pointer that must
    /// be reported to the garbage collector during stack scanning.
    /// Zero for the current Smi-only JIT tier (no heap-object values).
    pub gc_map: u64,
}

/// A single entry in the deoptimization table.
///
/// When a JIT compiled function encounters a value or operation it cannot
/// handle natively, it jumps to the deopt epilogue and returns [`JIT_DEOPT`].
/// The deopt table maps those code offsets back to bytecode offsets so that
/// the interpreter can resume from the correct point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeoptEntry {
    /// Byte offset in the JIT code buffer of the deopt point.
    pub code_offset: u32,
    /// Bytecode byte offset where interpretation should resume.
    pub bytecode_offset: u32,
    /// Bitmask of register-file slots that hold live values at this deopt
    /// point.  Bit `i` set means slot `i` must be preserved when
    /// reconstructing the interpreter frame.  Conservatively set to all slots
    /// live in the current baseline tier.
    pub liveness_map: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompiledCode
// ─────────────────────────────────────────────────────────────────────────────

/// The output of the [`BaselineCompiler`]: machine code bytes plus metadata.
pub struct CompiledCode {
    /// Emitted x86-64 machine code followed by the serialized safepoint and
    /// deopt tables and a 12-byte footer.  The footer is the last
    /// [`FOOTER_SIZE`] bytes and encodes the byte offsets of each table within
    /// `code`, plus [`METADATA_MAGIC`] as a sanity check.
    pub code: Vec<u8>,
    /// Number of bytes in `code` that are native machine instructions.
    /// Bytes in `code[native_code_len..]` are metadata tables.
    pub native_code_len: usize,
    /// Number of `i64` slots in the register file
    /// (`parameter_count + frame_size`).
    pub register_file_slots: usize,
    /// Safepoint table (code offset → bytecode instruction index + gc_map).
    pub safepoints: Vec<SafepointEntry>,
    /// Deoptimization table (code offset → bytecode byte offset + liveness).
    pub deopt_entries: Vec<DeoptEntry>,
}

/// Internal representation of the 12-byte metadata footer.
struct MetadataFooter {
    safepoint_table_start: u32,
    deopt_table_start: u32,
}

impl CompiledCode {
    /// Look up the safepoint entry whose `code_offset` equals `offset`.
    ///
    /// Returns `None` if no safepoint at that exact offset is recorded.
    pub fn find_safepoint(&self, offset: u32) -> Option<&SafepointEntry> {
        self.safepoints.iter().find(|e| e.code_offset == offset)
    }

    /// Look up the deopt entry whose `code_offset` equals `offset`.
    ///
    /// Returns `None` if no deopt entry at that exact offset is recorded.
    pub fn find_deopt(&self, offset: u32) -> Option<&DeoptEntry> {
        self.deopt_entries.iter().find(|e| e.code_offset == offset)
    }

    /// Parse the safepoint table from a slice of serialized code bytes.
    ///
    /// Returns `None` if the buffer does not contain a valid footer with
    /// [`METADATA_MAGIC`], or if the encoded table would overflow the buffer.
    pub fn parse_safepoints(code: &[u8]) -> Option<Vec<SafepointEntry>> {
        let footer = Self::read_footer(code)?;
        let start = footer.safepoint_table_start as usize;
        let data = code.get(start..)?;
        let count = u32::from_le_bytes(data.get(..4)?.try_into().ok()?) as usize;
        // Sanity-check: cap allocation to the number of entries that can
        // physically fit in the remaining buffer.
        let max_entries = data.len().saturating_sub(4) / SAFEPOINT_ENTRY_SIZE;
        if count > max_entries {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * SAFEPOINT_ENTRY_SIZE;
            let e = data.get(off..off + SAFEPOINT_ENTRY_SIZE)?;
            out.push(SafepointEntry {
                code_offset: u32::from_le_bytes(e[0..4].try_into().ok()?),
                bytecode_index: u32::from_le_bytes(e[4..8].try_into().ok()?),
                gc_map: u64::from_le_bytes(e[8..16].try_into().ok()?),
            });
        }
        Some(out)
    }

    /// Parse the deopt table from a slice of serialized code bytes.
    ///
    /// Returns `None` if the buffer does not contain a valid footer with
    /// [`METADATA_MAGIC`], or if the encoded table would overflow the buffer.
    pub fn parse_deopt_entries(code: &[u8]) -> Option<Vec<DeoptEntry>> {
        let footer = Self::read_footer(code)?;
        let start = footer.deopt_table_start as usize;
        let data = code.get(start..)?;
        let count = u32::from_le_bytes(data.get(..4)?.try_into().ok()?) as usize;
        // Sanity-check: cap allocation to the number of entries that can
        // physically fit in the remaining buffer.
        let max_entries = data.len().saturating_sub(4) / DEOPT_ENTRY_SIZE;
        if count > max_entries {
            return None;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * DEOPT_ENTRY_SIZE;
            let e = data.get(off..off + DEOPT_ENTRY_SIZE)?;
            out.push(DeoptEntry {
                code_offset: u32::from_le_bytes(e[0..4].try_into().ok()?),
                bytecode_offset: u32::from_le_bytes(e[4..8].try_into().ok()?),
                liveness_map: u64::from_le_bytes(e[8..16].try_into().ok()?),
            });
        }
        Some(out)
    }

    /// Read and validate the 12-byte footer at the end of `code`.
    fn read_footer(code: &[u8]) -> Option<MetadataFooter> {
        if code.len() < FOOTER_SIZE {
            return None;
        }
        let f = &code[code.len() - FOOTER_SIZE..];
        let magic = u32::from_le_bytes(f[8..12].try_into().ok()?);
        if magic != METADATA_MAGIC {
            return None;
        }
        Some(MetadataFooter {
            safepoint_table_start: u32::from_le_bytes(f[0..4].try_into().ok()?),
            deopt_table_start: u32::from_le_bytes(f[4..8].try_into().ok()?),
        })
    }

    /// Execute the compiled code on x86-64 Linux/macOS by allocating a page of
    /// read-write-execute memory with `mmap`, copying the code bytes into it,
    /// and invoking the JIT function.
    ///
    /// `args` provides the initial values for the parameter slots.  Missing
    /// arguments are filled with `0` (`Smi(0)`); extra arguments are ignored.
    ///
    /// Returns the accumulator value on success.  Returns
    /// [`StatorError::Internal`] if `mmap` fails.
    ///
    /// If the JIT function returns [`JIT_DEOPT`], the result is
    /// `Err(StatorError::Internal("jit deopt"))`.
    ///
    /// # Safety
    ///
    /// The `code` bytes inside this [`CompiledCode`] must be valid x86-64
    /// machine code emitted by [`BaselineCompiler`].  Executing arbitrary or
    /// malformed bytes via `mmap` and a function pointer is undefined behaviour.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        use std::ptr;

        let code_size = self.code.len();
        if code_size == 0 {
            return Err(StatorError::Internal("compiled code is empty".into()));
        }

        // Allocate a page of read/write/execute memory.
        //
        // SAFETY: arguments are valid; the return value is checked against
        // MAP_FAILED before use.
        let mem = unsafe {
            libc::mmap(
                ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mem == libc::MAP_FAILED {
            return Err(StatorError::Internal("mmap failed for JIT code".into()));
        }

        // SAFETY: `mem` is valid, page-aligned, and sized for `code_size` bytes.
        unsafe {
            ptr::copy_nonoverlapping(self.code.as_ptr(), mem.cast::<u8>(), code_size);
        }

        // Build the register file: fill with zeros, then overwrite with args.
        let mut regs = vec![0i64; self.register_file_slots];
        for (i, &v) in args.iter().enumerate().take(regs.len()) {
            regs[i] = v;
        }

        // Transmute and call the JIT function.
        //
        // SAFETY:
        // - `mem` contains correctly-encoded x86-64 machine code emitted by
        //   the baseline compiler.
        // - The function signature matches the JIT calling convention:
        //   `extern "C" fn(*mut i64) -> i64` (SysV AMD64).
        // - `regs.as_mut_ptr()` is valid for the lifetime of the call.
        let result = unsafe {
            let f: extern "C" fn(*mut i64) -> i64 = std::mem::transmute(mem);
            f(regs.as_mut_ptr())
        };

        // SAFETY: `mem` is a valid mapping of `code_size` bytes.
        unsafe {
            libc::munmap(mem, code_size);
        }

        if result == JIT_DEOPT {
            Err(StatorError::Internal("jit deopt".into()))
        } else {
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BaselineCompiler
// ─────────────────────────────────────────────────────────────────────────────

/// Baseline (non-optimising) JIT compiler.
///
/// Compiles a [`BytecodeArray`] to x86-64 machine code.  Each bytecode
/// instruction is translated to a fixed native-code sequence; complex
/// operations (property access, function calls, generators) emit a jump to
/// the deopt epilogue that returns [`JIT_DEOPT`].
pub struct BaselineCompiler<'a> {
    bytecode: &'a BytecodeArray,
    masm: MacroAssembler,
    param_count: usize,
    safepoints: Vec<SafepointEntry>,
    deopt_entries: Vec<DeoptEntry>,
    /// Per-instruction label (bound at the start of each instruction's code).
    labels: Vec<Label>,
    /// Label for the shared deopt epilogue.
    deopt_label: Label,
    /// Promoted global variables: `(name_idx, flat_slot_index)` pairs.
    ///
    /// During compilation, `LdaGlobal`/`StaGlobal` for these `name_idx` values
    /// are lowered to direct register-file loads/stores instead of runtime stub
    /// calls.  The prologue loads each global once, and the epilogue writes back
    /// any modifications.  Around Call/Construct opcodes the promoted slots are
    /// flushed and reloaded to preserve correctness.
    promoted_globals: Vec<(u32, usize)>,
    /// Number of extra register-file slots allocated for promoted globals.
    promoted_extra_slots: usize,
}

impl<'a> BaselineCompiler<'a> {
    /// Compile `bytecode` to native x86-64 machine code.
    ///
    /// Returns a [`CompiledCode`] containing the emitted bytes and all
    /// associated metadata.  The safepoint and deopt tables are serialized and
    /// appended to the code buffer after the native instructions; a 12-byte
    /// footer records the table start offsets and [`METADATA_MAGIC`].
    pub fn compile(bytecode: &'a BytecodeArray) -> StatorResult<CompiledCode> {
        let mut c = Self {
            bytecode,
            masm: MacroAssembler::new(),
            param_count: bytecode.parameter_count() as usize,
            safepoints: Vec::new(),
            deopt_entries: Vec::new(),
            labels: Vec::new(),
            deopt_label: Label::new(),
            promoted_globals: Vec::new(),
            promoted_extra_slots: 0,
        };
        c.compile_function()?;
        let register_file_slots = bytecode.parameter_count() as usize
            + bytecode.frame_size() as usize
            + c.promoted_extra_slots;
        let mut code = c.masm.into_code();
        let native_code_len = code.len();

        // ── Serialize safepoint table ────────────────────────────────────────
        let safepoint_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let sp_count = c.safepoints.len() as u32;
        code.extend_from_slice(&sp_count.to_le_bytes());
        for e in &c.safepoints {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_index.to_le_bytes());
            code.extend_from_slice(&e.gc_map.to_le_bytes());
        }

        // ── Serialize deopt table ────────────────────────────────────────────
        let deopt_table_start = u32::try_from(code.len())
            .map_err(|_| StatorError::Internal("compiled code exceeds 4 GiB limit".into()))?;
        let de_count = c.deopt_entries.len() as u32;
        code.extend_from_slice(&de_count.to_le_bytes());
        for e in &c.deopt_entries {
            code.extend_from_slice(&e.code_offset.to_le_bytes());
            code.extend_from_slice(&e.bytecode_offset.to_le_bytes());
            code.extend_from_slice(&e.liveness_map.to_le_bytes());
        }

        // ── Serialize footer (last FOOTER_SIZE bytes) ────────────────────────
        code.extend_from_slice(&safepoint_table_start.to_le_bytes());
        code.extend_from_slice(&deopt_table_start.to_le_bytes());
        code.extend_from_slice(&METADATA_MAGIC.to_le_bytes());

        Ok(CompiledCode {
            code,
            native_code_len,
            register_file_slots,
            safepoints: c.safepoints,
            deopt_entries: c.deopt_entries,
        })
    }

    // ── Prologue / epilogue ──────────────────────────────────────────────────

    /// Emit the standard function prologue.
    ///
    /// Sets up the call frame and loads the register-file pointer into R14.
    /// Five callee-saved registers are pushed (odd count) so that the stack
    /// is 16-byte aligned after the return-address push by the caller.
    ///
    /// ```text
    /// push rbp
    /// mov  rbp, rsp
    /// push rbx        ; callee-saved context pointer
    /// push r12        ; callee-saved accumulator
    /// push r14        ; callee-saved register-file pointer
    /// push r15        ; alignment pad (reserved for future use)
    /// mov  r14, rdi   ; r14 = regs argument
    /// mov  rbx, rsi   ; rbx = closure context pointer
    /// xor  r12, r12   ; accumulator = 0
    /// ```
    fn emit_prologue(&mut self) {
        self.masm.push(Reg64::Rbp);
        self.masm.mov_rr(Reg64::Rbp, Reg64::Rsp);
        self.masm.push(Reg64::Rbx);
        self.masm.push(Reg64::R12);
        self.masm.push(Reg64::R14);
        // Push R15 for 16-byte stack alignment (5 pushes + return addr = even).
        self.masm.push(Reg64::R15);
        self.masm.mov_rr(Reg64::R14, Reg64::Rdi);
        // RSI carries the raw closure-context pointer (passed by execute).
        // Store in RBX (callee-saved) for use by context-slot stubs.
        self.masm.mov_rr(Reg64::Rbx, Reg64::Rsi);
        self.masm.xor_rr(Reg64::R12, Reg64::R12);
    }

    /// Emit the normal function epilogue.
    ///
    /// ```text
    /// mov rax, r12    ; return accumulator
    /// pop r15         ; alignment pad
    /// pop r14
    /// pop r12
    /// pop rbx
    /// pop rbp
    /// ret
    /// ```
    fn emit_normal_epilogue(&mut self) {
        self.masm.mov_rr(Reg64::Rax, Reg64::R12);
        self.masm.pop(Reg64::R15);
        self.masm.pop(Reg64::R14);
        self.masm.pop(Reg64::R12);
        self.masm.pop(Reg64::Rbx);
        self.masm.pop(Reg64::Rbp);
        self.masm.ret();
    }

    /// Emit the deopt epilogue.
    ///
    /// Loads [`JIT_DEOPT`] into the return-value register and then falls
    /// through to the standard register-restore / `ret` sequence.
    ///
    /// ```text
    /// deopt_label:
    ///   mov r12, JIT_DEOPT
    ///   mov rax, r12
    ///   pop r15
    ///   pop r14
    ///   pop r12
    ///   pop rbx
    ///   pop rbp
    ///   ret
    /// ```
    fn emit_deopt_epilogue(&mut self) {
        self.masm.bind_label(&mut self.deopt_label);
        self.masm.mov_ri(Reg64::R12, JIT_DEOPT);
        self.emit_normal_epilogue();
    }

    // ── Register-file helpers ────────────────────────────────────────────────

    /// Compute the byte offset from R14 (register-file base) for a bytecode
    /// register operand value `v`.
    fn reg_offset(&self, v: u32) -> i32 {
        let signed = v as i32;
        let flat_index = if signed >= 0 {
            self.param_count + signed as usize
        } else {
            (-(signed + 1)) as usize
        };
        (flat_index * 8) as i32
    }

    /// Compute the flat slot index for a bytecode register operand `v`.
    ///
    /// This is the element index into the `*mut i64` register file (not a
    /// byte offset).
    fn reg_flat_index(&self, v: u32) -> usize {
        let signed = v as i32;
        if signed >= 0 {
            self.param_count + signed as usize
        } else {
            (-(signed + 1)) as usize
        }
    }

    /// Emit code to load register `v` into `dst`.
    fn emit_load_reg(&mut self, dst: Reg64, v: u32) {
        let off = self.reg_offset(v);
        self.masm.mov_load_base_disp32(dst, Reg64::R14, off);
    }

    /// Emit code to store `src` into register `v`.
    fn emit_store_reg(&mut self, v: u32, src: Reg64) {
        let off = self.reg_offset(v);
        self.masm.mov_store_base_disp32(Reg64::R14, off, src);
    }

    // ── Global register promotion helpers ────────────────────────────────────

    /// Scan the decoded instruction stream and collect every unique
    /// `ConstantPoolIdx` operand used by `LdaGlobal` / `StaGlobal`.
    ///
    /// Each unique `name_idx` is allocated a fresh register-file slot beyond
    /// the original `parameter_count + frame_size` range.
    fn scan_and_promote_globals(&mut self, instructions: &[Instruction]) {
        let base_slots = self.param_count + self.bytecode.frame_size() as usize;

        let mut seen = std::collections::HashSet::new();
        for instr in instructions {
            if matches!(instr.opcode, Opcode::LdaGlobal | Opcode::StaGlobal)
                && let Operand::ConstantPoolIdx(idx) = instr.operands[0]
            {
                seen.insert(idx);
            }
        }

        let mut sorted: Vec<u32> = seen.into_iter().collect();
        sorted.sort_unstable();

        self.promoted_globals = sorted
            .iter()
            .enumerate()
            .map(|(i, &name_idx)| (name_idx, base_slots + i))
            .collect();
        self.promoted_extra_slots = sorted.len();
    }

    /// Look up the promoted register-file flat-slot index for `name_idx`.
    fn promoted_slot_for(&self, name_idx: u32) -> Option<usize> {
        self.promoted_globals
            .iter()
            .find(|(n, _)| *n == name_idx)
            .map(|&(_, slot)| slot)
    }

    /// Byte offset into the register file for a promoted flat-slot index.
    #[allow(dead_code)]
    fn promoted_offset(flat: usize) -> i32 {
        (flat * 8) as i32
    }

    /// Emit code to load **all** promoted globals from `GlobalEnv` into their
    /// register-file slots.
    ///
    /// Each global is loaded via [`jit_runtime_lda_global`]; a deopt sentinel
    /// check is omitted here because the globals are already initialised by the
    /// time baseline JIT fires (they were created during the first interpreter
    /// pass).  If a load does return `JIT_DEOPT` the slot will hold the deopt
    /// sentinel and will be caught later when the value is used.
    #[allow(unused_variables)]
    fn emit_promoted_global_loads(&mut self) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let addr = jit_runtime::jit_runtime_lda_global as *const () as usize as i64;
            for &(name_idx, flat) in &self.promoted_globals.clone() {
                // RDI = name_idx
                self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
                // Store result into promoted slot: [r14 + flat*8] = rax
                self.masm.mov_store_base_disp32(
                    Reg64::R14,
                    Self::promoted_offset(flat),
                    Reg64::Rax,
                );
            }
        }
    }

    /// Emit code to flush **all** promoted globals from their register-file
    /// slots back to `GlobalEnv` via [`jit_runtime_sta_global`].
    #[allow(unused_variables)]
    fn emit_promoted_global_stores(&mut self) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let addr = jit_runtime::jit_runtime_sta_global as *const () as usize as i64;
            for &(name_idx, flat) in &self.promoted_globals.clone() {
                // RDI = name_idx
                self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));
                // RSI = value from promoted slot
                self.masm
                    .mov_load_base_disp32(Reg64::Rsi, Reg64::R14, Self::promoted_offset(flat));
                self.masm.mov_ri(Reg64::R11, addr);
                self.masm.call_reg(Reg64::R11);
            }
        }
    }

    // ── Comparison helper ────────────────────────────────────────────────────

    /// Emit code that:
    /// 1. Loads the RHS register `v` into R11.
    /// 2. Compares R12 (accumulator) against R11.
    /// 3. Sets AL via `SETCC cc`.
    /// 4. Zero-extends AL into R12.
    /// 5. Converts the 0/1 result in R12 to [`JIT_FALSE`] / [`JIT_TRUE`].
    fn emit_compare_and_set(&mut self, v: u32, cc: CondCode) {
        self.emit_load_reg(Reg64::R11, v);
        self.masm.cmp_rr(Reg64::R12, Reg64::R11);
        self.masm.setcc_al(cc);
        self.masm.movzx_r64_al(Reg64::R12);
        // Convert raw 0/1 → JIT_FALSE/JIT_TRUE by adding JIT_FALSE.
        self.masm.mov_ri(Reg64::R11, JIT_FALSE);
        self.masm.add_rr(Reg64::R12, Reg64::R11);
    }

    /// Try to fuse a comparison opcode with a following conditional jump.
    ///
    /// If the instruction at `idx + 1` is `JumpIfFalse`, `JumpIfTrue`,
    /// `JumpIfToBooleanFalse`, or `JumpIfToBooleanTrue`, emit a fused
    /// `mov r11,[r14+off]; cmp r12,r11; jcc target` sequence and return
    /// `Ok(1)` (one extra instruction consumed). Otherwise return `Ok(0)`.
    fn try_fuse_compare_jump(
        &mut self,
        idx: usize,
        instructions: &[Instruction],
        byte_offsets: &[usize],
        v: u32,
        cc: CondCode,
    ) -> StatorResult<usize> {
        let n = instructions.len();
        let next_idx = idx + 1;
        if next_idx >= n {
            return Ok(0);
        }

        let next = &instructions[next_idx];
        let fused_cc = match next.opcode {
            Opcode::JumpIfFalse | Opcode::JumpIfToBooleanFalse => negate_cc(cc),
            Opcode::JumpIfTrue | Opcode::JumpIfToBooleanTrue => cc,
            _ => return Ok(0),
        };

        let Operand::JumpOffset(delta) = next.operands[0] else {
            return Ok(0);
        };

        let target = Self::resolve_target(
            jump_target_byte(next_idx, delta, byte_offsets),
            byte_offsets,
            n,
        )?;

        // Emit fused: load RHS, compare, conditional jump.
        self.emit_load_reg(Reg64::R11, v);
        self.masm.cmp_rr(Reg64::R12, Reg64::R11);
        self.emit_cond_jump(fused_cc, target);

        Ok(1)
    }

    /// Emit code that compares R12 against `sentinel` and converts the result
    /// to JIT_FALSE / JIT_TRUE (1 if equal, 0 if not).
    fn emit_test_sentinel(&mut self, sentinel: i64) {
        self.masm.mov_ri(Reg64::R11, sentinel);
        self.masm.cmp_rr(Reg64::R12, Reg64::R11);
        self.masm.setcc_al(CondCode::Equal);
        self.masm.movzx_r64_al(Reg64::R12);
        self.masm.mov_ri(Reg64::R11, JIT_FALSE);
        self.masm.add_rr(Reg64::R12, Reg64::R11);
    }

    // ── Jump helpers ─────────────────────────────────────────────────────────

    /// Find the instruction index whose bytecode byte offset equals
    /// `target_byte`, given the `byte_offsets` table.
    fn resolve_target(
        target_byte: usize,
        byte_offsets: &[usize],
        instr_count: usize,
    ) -> StatorResult<usize> {
        byte_offsets[..instr_count]
            .binary_search(&target_byte)
            .map_err(|_| {
                StatorError::Internal(format!(
                    "jump target {target_byte} is not at an instruction boundary"
                ))
            })
    }

    /// Emit an unconditional JMP to the instruction at `target_idx`.
    fn emit_jump(&mut self, target_idx: usize) {
        self.masm.jmp(&mut self.labels[target_idx]);
    }

    /// Emit a conditional JMP (`cc`) to `target_idx`.
    fn emit_cond_jump(&mut self, cc: CondCode, target_idx: usize) {
        self.masm.jcc(cc, &mut self.labels[target_idx]);
    }

    // ── Deopt helper ─────────────────────────────────────────────────────────

    /// Compute a conservative liveness bitmask covering all register-file
    /// slots for use in deopt entries.
    ///
    /// Sets bit `i` for every slot index `i` in `0..register_file_slots`.
    /// This is a conservative over-approximation: the interpreter will
    /// preserve all slots when reconstructing the frame.
    fn all_slots_live(&self) -> u64 {
        let slots = self.param_count + self.bytecode.frame_size() as usize;
        if slots >= 64 {
            u64::MAX
        } else {
            (1u64 << slots).wrapping_sub(1)
        }
    }

    /// Record a deopt point at the current code position and emit a JMP to
    /// the deopt epilogue.
    fn emit_deopt(&mut self, bytecode_offset: u32) {
        let code_off = self.masm.position() as u32;
        let liveness_map = self.all_slots_live();
        self.deopt_entries.push(DeoptEntry {
            code_offset: code_off,
            bytecode_offset,
            liveness_map,
        });
        self.masm.jmp(&mut self.deopt_label);
    }

    // ── Runtime call-stub emission ───────────────────────────────────────────

    /// Emit a call to the runtime trampoline for an opcode that cannot be
    /// natively compiled.
    ///
    /// On x86-64 + Unix this emits the full call sequence (parameter setup,
    /// indirect `CALL`, deopt check, accumulator update).  On all other
    /// platforms it falls back to [`emit_deopt`](Self::emit_deopt).
    ///
    /// # Parameters
    ///
    /// * `opcode` – discriminant of the [`Opcode`] to execute at runtime.
    /// * `operand1` / `operand2` – opcode-specific operand values (typically
    ///   flat register-file slot indices or constant-pool indices).
    /// * `bytecode_offset` – byte offset in the bytecode stream (for the
    ///   deopt entry if the trampoline returns [`JIT_DEOPT`]).
    #[allow(unused_variables)]
    fn emit_runtime_stub(
        &mut self,
        opcode: Opcode,
        operand1: i64,
        operand2: i64,
        bytecode_offset: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // ── Set up SysV AMD64 arguments ─────────────────────────────
            //   RDI = opcode (u32)
            //   RSI = register-file base pointer (R14)
            //   RDX = accumulator value (R12)
            //   RCX = operand1
            //   R8  = operand2
            self.masm.mov_ri(Reg64::Rdi, opcode as u8 as i64);
            self.masm.mov_rr(Reg64::Rsi, Reg64::R14);
            self.masm.mov_rr(Reg64::Rdx, Reg64::R12);
            self.masm.mov_ri(Reg64::Rcx, operand1);
            self.masm.mov_ri(Reg64::R8, operand2);

            // Load trampoline address into R11 and call.
            let trampoline_addr = jit_runtime::jit_runtime_trampoline as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, trampoline_addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        // Non-x86-64 fallback: deopt.
        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_lda_named_property`] for
    /// `LdaNamedProperty` bytecodes.
    ///
    /// Unlike [`emit_runtime_stub`], this loads the object value from the
    /// register file in JIT code and calls a dedicated runtime function that
    /// skips generic opcode dispatch and uses shape-based inline caching.
    #[allow(unused_variables)]
    fn emit_lda_named_property_stub(&mut self, obj_flat: i64, name_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // Load object value from register file: RDI = regs[obj_flat].
            let byte_offset = (obj_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, byte_offset);

            // RSI = name_idx (constant-pool index).
            self.masm.mov_ri(Reg64::Rsi, i64::from(name_idx));

            // RDX = feedback slot (reserved, zero for now).
            self.masm.xor_rr(Reg64::Rdx, Reg64::Rdx);

            let addr = jit_runtime::jit_runtime_lda_named_property as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_call_undefined_receiver0`] for
    /// `CallUndefinedReceiver0` bytecodes.
    ///
    /// Loads the callee from the register file in JIT code and calls a
    /// dedicated function that handles native and JIT-compiled callees
    /// without generic opcode dispatch.
    #[allow(unused_variables)]
    fn emit_call_undefined_receiver0_stub(&mut self, callee_flat: i64, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // Load callee value from register file: RDI = regs[callee_flat].
            let byte_offset = (callee_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, byte_offset);

            let addr =
                jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to [`jit_runtime::jit_runtime_lda_global`]
    /// for `LdaGlobal` bytecodes.
    ///
    /// Unlike [`emit_runtime_stub`], this passes only the constant-pool
    /// name index and calls a dedicated function that skips generic opcode
    /// dispatch.
    #[allow(unused_variables)]
    fn emit_lda_global_stub(&mut self, name_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = name_idx (constant-pool index).
            self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));

            let addr = jit_runtime::jit_runtime_lda_global as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to [`jit_runtime::jit_runtime_sta_global`]
    /// for `StaGlobal` bytecodes.
    ///
    /// Passes the constant-pool name index and the current accumulator
    /// value to a dedicated function.
    #[allow(unused_variables)]
    fn emit_sta_global_stub(&mut self, name_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = name_idx (constant-pool index).
            self.masm.mov_ri(Reg64::Rdi, i64::from(name_idx));
            // RSI = accumulator value (the value to store).
            self.masm.mov_rr(Reg64::Rsi, Reg64::R12);

            let addr = jit_runtime::jit_runtime_sta_global as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_lda_keyed_property`] for
    /// `LdaKeyedProperty` bytecodes.
    ///
    /// Loads the receiver from the register file and passes both the
    /// receiver and the accumulator (key) to a dedicated function that
    /// skips generic opcode dispatch.
    #[allow(unused_variables)]
    fn emit_lda_keyed_property_stub(&mut self, obj_flat: i64, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = receiver object from register file.
            let byte_offset = (obj_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, byte_offset);

            // RSI = key (current accumulator value).
            self.masm.mov_rr(Reg64::Rsi, Reg64::R12);

            let addr = jit_runtime::jit_runtime_lda_keyed_property as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_sta_keyed_property`] for
    /// `StaKeyedProperty` bytecodes.
    ///
    /// Passes receiver, key, and the accumulator (value) to a dedicated
    /// function that skips generic opcode dispatch.
    #[allow(unused_variables)]
    fn emit_sta_keyed_property_stub(&mut self, obj_flat: i64, key_flat: i64, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = receiver object from register file.
            let obj_byte_offset = (obj_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, obj_byte_offset);

            // RSI = key from register file.
            let key_byte_offset = (key_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rsi, Reg64::R14, key_byte_offset);

            // RDX = value (current accumulator).
            self.masm.mov_rr(Reg64::Rdx, Reg64::R12);

            let addr = jit_runtime::jit_runtime_sta_keyed_property as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ─────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to the direct context-slot stub
    /// [`jit_runtime::jit_runtime_lda_context_slot_direct`] for
    /// `LdaCurrentContextSlot` / `LdaImmutableCurrentContextSlot`.
    ///
    /// RBX carries the raw `RefCell<JsContext>` pointer (set in the
    /// prologue from the RSI parameter passed by `execute`).
    #[allow(unused_variables)]
    fn emit_lda_context_slot_stub(&mut self, slot_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = ctx_raw (from RBX), RSI = slot_idx.
            self.masm.mov_rr(Reg64::Rdi, Reg64::Rbx);
            self.masm.mov_ri(Reg64::Rsi, i64::from(slot_idx));

            let addr =
                jit_runtime::jit_runtime_lda_context_slot_direct as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to the direct context-slot stub
    /// [`jit_runtime::jit_runtime_sta_context_slot_direct`] for
    /// `StaCurrentContextSlot`.
    ///
    /// RBX carries the raw `RefCell<JsContext>` pointer.
    #[allow(unused_variables)]
    fn emit_sta_context_slot_stub(&mut self, slot_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = ctx_raw (from RBX), RSI = slot_idx, RDX = value (R12).
            self.masm.mov_rr(Reg64::Rdi, Reg64::Rbx);
            self.masm.mov_ri(Reg64::Rsi, i64::from(slot_idx));
            self.masm.mov_rr(Reg64::Rdx, Reg64::R12);

            let addr =
                jit_runtime::jit_runtime_sta_context_slot_direct as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_sta_named_property`] for
    /// `StaNamedProperty` / `StaNamedOwnProperty` / `DefineNamedOwnProperty`.
    #[allow(unused_variables)]
    fn emit_sta_named_property_stub(&mut self, obj_flat: i64, name_idx: u32, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = receiver object from register file.
            let obj_byte_offset = (obj_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, obj_byte_offset);

            // RSI = name_idx (constant-pool index).
            self.masm.mov_ri(Reg64::Rsi, i64::from(name_idx));

            // RDX = value (current accumulator).
            self.masm.mov_rr(Reg64::Rdx, Reg64::R12);

            let addr = jit_runtime::jit_runtime_sta_named_property as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ──────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_call_property0`] for `CallProperty0`.
    #[allow(unused_variables)]
    fn emit_call_property0_stub(
        &mut self,
        callee_flat: i64,
        receiver_flat: i64,
        bytecode_offset: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = callee from register file.
            let callee_byte_offset = (callee_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, callee_byte_offset);

            // RSI = receiver from register file.
            let receiver_byte_offset = (receiver_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rsi, Reg64::R14, receiver_byte_offset);

            let addr = jit_runtime::jit_runtime_call_property0 as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ──────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    /// Emit a specialized call to
    /// [`jit_runtime::jit_runtime_call_property1`] for `CallProperty1`.
    #[allow(unused_variables)]
    fn emit_call_property1_stub(
        &mut self,
        callee_flat: i64,
        receiver_flat: i64,
        arg0_flat: i64,
        bytecode_offset: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // RDI = callee from register file.
            let callee_byte_offset = (callee_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::R14, callee_byte_offset);

            // RSI = receiver from register file.
            let receiver_byte_offset = (receiver_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rsi, Reg64::R14, receiver_byte_offset);

            // RDX = arg0 from register file.
            let arg0_byte_offset = (arg0_flat as i32) * 8;
            self.masm
                .mov_load_base_disp32(Reg64::Rdx, Reg64::R14, arg0_byte_offset);

            let addr = jit_runtime::jit_runtime_call_property1 as *const () as usize as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);

            // ── Check for deopt ─────────────────────────────────────────
            self.masm.mov_ri(Reg64::R11, JIT_DEOPT);
            self.masm.cmp_rr(Reg64::Rax, Reg64::R11);

            let code_off = self.masm.position() as u32;
            let liveness_map = self.all_slots_live();
            self.deopt_entries.push(DeoptEntry {
                code_offset: code_off,
                bytecode_offset,
                liveness_map,
            });
            self.masm.je(&mut self.deopt_label);

            // ── Success: update accumulator ──────────────────────────────
            self.masm.mov_rr(Reg64::R12, Reg64::Rax);
            return;
        }

        #[allow(unreachable_code)]
        self.emit_deopt(bytecode_offset);
    }

    // ── Main compilation pass ────────────────────────────────────────────────

    fn compile_function(&mut self) -> StatorResult<()> {
        let (instructions, byte_offsets) = decode_with_byte_offsets(self.bytecode.bytecodes())?;
        let n = instructions.len();

        // ── Global register promotion analysis ───────────────────────────────
        // Scan the bytecode for LdaGlobal/StaGlobal operands and allocate
        // extra register-file slots so the hot loop accesses memory via [R14]
        // instead of calling runtime stubs.
        self.scan_and_promote_globals(&instructions);

        // Pre-create one label per instruction.
        self.labels = (0..n).map(|_| Label::new()).collect();

        self.emit_prologue();

        // Load all promoted globals into their register-file slots (once).
        self.emit_promoted_global_loads();

        let mut idx = 0;
        while idx < n {
            let instr = &instructions[idx];

            // Bind the label for this instruction to the current code position.
            self.masm.bind_label(&mut self.labels[idx]);

            // Safepoint at every instruction.
            self.safepoints.push(SafepointEntry {
                code_offset: self.masm.position() as u32,
                bytecode_index: idx as u32,
                gc_map: 0,
            });

            let extra = self.compile_instruction(
                idx,
                &instructions,
                &byte_offsets,
                instr,
                byte_offsets[idx] as u32,
            )?;

            // Bind labels and record safepoints for any fused (skipped)
            // instructions so that jump targets into them still resolve.
            for skip_i in 1..=extra {
                let fused_idx = idx + skip_i;
                self.masm.bind_label(&mut self.labels[fused_idx]);
                self.safepoints.push(SafepointEntry {
                    code_offset: self.masm.position() as u32,
                    bytecode_index: fused_idx as u32,
                    gc_map: 0,
                });
            }

            idx += 1 + extra;
        }

        // Emit the deopt epilogue after the normal instruction stream.
        self.emit_deopt_epilogue();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_instruction(
        &mut self,
        idx: usize,
        instructions: &[Instruction],
        byte_offsets: &[usize],
        instr: &Instruction,
        bytecode_offset: u32,
    ) -> StatorResult<usize> {
        let n = instructions.len();

        match instr.opcode {
            // ── Load immediates ──────────────────────────────────────────────
            Opcode::LdaZero => {
                self.masm.xor_rr(Reg64::R12, Reg64::R12);
            }
            Opcode::LdaSmi => {
                let Operand::Immediate(v) = instr.operands[0] else {
                    return Err(bad_operand("LdaSmi", 0));
                };
                self.masm.mov_ri(Reg64::R12, v as i64);
            }
            Opcode::LdaUndefined => {
                self.masm.mov_ri(Reg64::R12, JIT_UNDEFINED);
            }
            Opcode::LdaNull => {
                self.masm.mov_ri(Reg64::R12, JIT_NULL);
            }
            Opcode::LdaTrue => {
                self.masm.mov_ri(Reg64::R12, JIT_TRUE);
            }
            Opcode::LdaFalse => {
                self.masm.mov_ri(Reg64::R12, JIT_FALSE);
            }
            Opcode::Nop => {}
            Opcode::LdaConstant => {
                let Operand::ConstantPoolIdx(idx_cp) = instr.operands[0] else {
                    return Err(bad_operand("LdaConstant", 0));
                };
                match self.bytecode.get_constant(idx_cp) {
                    Some(ConstantPoolEntry::Number(f)) => {
                        let f = *f;
                        if f.fract() == 0.0
                            && (i32::MIN as f64..=i32::MAX as f64).contains(&f)
                            && f.is_finite()
                        {
                            self.masm.mov_ri(Reg64::R12, f as i64);
                        } else {
                            // Non-integer numbers require a HeapNumber; deopt.
                            self.emit_deopt(bytecode_offset);
                        }
                    }
                    Some(ConstantPoolEntry::Boolean(true)) => {
                        self.masm.mov_ri(Reg64::R12, JIT_TRUE);
                    }
                    Some(ConstantPoolEntry::Boolean(false)) => {
                        self.masm.mov_ri(Reg64::R12, JIT_FALSE);
                    }
                    Some(ConstantPoolEntry::Null) => {
                        self.masm.mov_ri(Reg64::R12, JIT_NULL);
                    }
                    Some(ConstantPoolEntry::Undefined) => {
                        self.masm.mov_ri(Reg64::R12, JIT_UNDEFINED);
                    }
                    _ => {
                        // Strings and Function entries require heap allocation; deopt.
                        self.emit_deopt(bytecode_offset);
                    }
                }
            }

            // ── Register moves ───────────────────────────────────────────────
            Opcode::Ldar => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Ldar", 0));
                };
                self.emit_load_reg(Reg64::R12, v);
            }
            Opcode::LdarAddStar => {
                let Operand::Register(src) = instr.operands[0] else {
                    return Err(bad_operand("LdarAddStar", 0));
                };
                let Operand::Register(add_reg) = instr.operands[1] else {
                    return Err(bad_operand("LdarAddStar", 1));
                };
                let Operand::Register(dst) = instr.operands[2] else {
                    return Err(bad_operand("LdarAddStar", 2));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[3] else {
                    return Err(bad_operand("LdarAddStar", 3));
                };
                self.emit_load_reg(Reg64::R12, src);
                self.emit_load_reg(Reg64::R11, add_reg);
                self.masm.add_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::LdarSubStar => {
                let Operand::Register(src) = instr.operands[0] else {
                    return Err(bad_operand("LdarSubStar", 0));
                };
                let Operand::Register(sub_reg) = instr.operands[1] else {
                    return Err(bad_operand("LdarSubStar", 1));
                };
                let Operand::Register(dst) = instr.operands[2] else {
                    return Err(bad_operand("LdarSubStar", 2));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[3] else {
                    return Err(bad_operand("LdarSubStar", 3));
                };
                self.emit_load_reg(Reg64::R12, src);
                self.emit_load_reg(Reg64::R11, sub_reg);
                self.masm.sub_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::Star => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Star", 0));
                };
                self.emit_store_reg(v, Reg64::R12);
            }
            Opcode::Mov => {
                let Operand::Register(src) = instr.operands[0] else {
                    return Err(bad_operand("Mov", 0));
                };
                let Operand::Register(dst) = instr.operands[1] else {
                    return Err(bad_operand("Mov", 1));
                };
                self.emit_load_reg(Reg64::R11, src);
                self.emit_store_reg(dst, Reg64::R11);
            }

            // ── Arithmetic ───────────────────────────────────────────────────
            Opcode::Add => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Add", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.add_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Sub => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Sub", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.sub_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Mul => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("Mul", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Inc => {
                self.masm.add_ri(Reg64::R12, 1);
            }
            Opcode::Dec => {
                self.masm.sub_ri(Reg64::R12, 1);
            }
            Opcode::AddSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("AddSmi", 0));
                };
                self.masm.add_ri(Reg64::R12, imm);
            }
            Opcode::AddSmiStar => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("AddSmiStar", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("AddSmiStar", 1));
                };
                let Operand::Register(dst) = instr.operands[2] else {
                    return Err(bad_operand("AddSmiStar", 2));
                };
                self.masm.add_ri(Reg64::R12, imm);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::SubSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("SubSmi", 0));
                };
                self.masm.sub_ri(Reg64::R12, imm);
            }
            Opcode::MulSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("MulSmi", 0));
                };
                self.masm.mov_ri(Reg64::R11, imm as i64);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::Negate => {
                self.masm.neg_r(Reg64::R12);
            }

            // ── Bitwise ─────────────────────────────────────────────────────
            Opcode::BitwiseOr => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseOr", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.or_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseAnd => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseAnd", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.and_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseXor => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseXor", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.xor_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseOrSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseOrSmi", 0));
                };
                self.masm.or_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseAndSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseAndSmi", 0));
                };
                self.masm.and_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseXorSmi => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("BitwiseXorSmi", 0));
                };
                self.masm.xor_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseNot => {
                self.masm.not_r(Reg64::R12);
            }

            // ── Comparisons ──────────────────────────────────────────────────
            Opcode::TestLessThan => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestLessThan", 0));
                };
                let fused =
                    self.try_fuse_compare_jump(idx, instructions, byte_offsets, v, CondCode::Less)?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::Less);
            }
            Opcode::TestLessThanJump => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestLessThanJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("TestLessThanJump", 1));
                };
                let Operand::JumpOffset(delta) = instr.operands[2] else {
                    return Err(bad_operand("TestLessThanJump", 2));
                };
                let Operand::Flag(is_true_flag) = instr.operands[3] else {
                    return Err(bad_operand("TestLessThanJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::Less)
                } else {
                    CondCode::Less
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::TestGreaterThanJump => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestGreaterThanJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("TestGreaterThanJump", 1));
                };
                let Operand::JumpOffset(delta) = instr.operands[2] else {
                    return Err(bad_operand("TestGreaterThanJump", 2));
                };
                let Operand::Flag(is_true_flag) = instr.operands[3] else {
                    return Err(bad_operand("TestGreaterThanJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::Greater)
                } else {
                    CondCode::Greater
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::TestEqualJump => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestEqualJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("TestEqualJump", 1));
                };
                let Operand::JumpOffset(delta) = instr.operands[2] else {
                    return Err(bad_operand("TestEqualJump", 2));
                };
                let Operand::Flag(is_true_flag) = instr.operands[3] else {
                    return Err(bad_operand("TestEqualJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::Equal)
                } else {
                    CondCode::Equal
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::TestEqualStrictJump => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestEqualStrictJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("TestEqualStrictJump", 1));
                };
                let Operand::JumpOffset(delta) = instr.operands[2] else {
                    return Err(bad_operand("TestEqualStrictJump", 2));
                };
                let Operand::Flag(is_true_flag) = instr.operands[3] else {
                    return Err(bad_operand("TestEqualStrictJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::Equal)
                } else {
                    CondCode::Equal
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::SubSmiStar => {
                let Operand::Immediate(imm) = instr.operands[0] else {
                    return Err(bad_operand("SubSmiStar", 0));
                };
                let Operand::FeedbackSlot(_slot) = instr.operands[1] else {
                    return Err(bad_operand("SubSmiStar", 1));
                };
                let Operand::Register(dst) = instr.operands[2] else {
                    return Err(bad_operand("SubSmiStar", 2));
                };
                self.masm.sub_ri(Reg64::R12, imm);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::TestGreaterThan => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestGreaterThan", 0));
                };
                let fused = self.try_fuse_compare_jump(
                    idx,
                    instructions,
                    byte_offsets,
                    v,
                    CondCode::Greater,
                )?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::Greater);
            }
            Opcode::TestLessThanOrEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestLessThanOrEqual", 0));
                };
                let fused = self.try_fuse_compare_jump(
                    idx,
                    instructions,
                    byte_offsets,
                    v,
                    CondCode::LessEq,
                )?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::LessEq);
            }
            Opcode::TestGreaterThanOrEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestGreaterThanOrEqual", 0));
                };
                let fused = self.try_fuse_compare_jump(
                    idx,
                    instructions,
                    byte_offsets,
                    v,
                    CondCode::GreaterEq,
                )?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::GreaterEq);
            }
            Opcode::TestEqual | Opcode::TestEqualStrict => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestEqual", 0));
                };
                let fused = self.try_fuse_compare_jump(
                    idx,
                    instructions,
                    byte_offsets,
                    v,
                    CondCode::Equal,
                )?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::Equal);
            }
            Opcode::TestNotEqual => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("TestNotEqual", 0));
                };
                let fused = self.try_fuse_compare_jump(
                    idx,
                    instructions,
                    byte_offsets,
                    v,
                    CondCode::NotEqual,
                )?;
                if fused > 0 {
                    return Ok(fused);
                }
                self.emit_compare_and_set(v, CondCode::NotEqual);
            }
            Opcode::TestNull => {
                self.emit_test_sentinel(JIT_NULL);
            }
            Opcode::TestUndefined => {
                self.emit_test_sentinel(JIT_UNDEFINED);
            }

            // ── Logical ──────────────────────────────────────────────────────
            Opcode::LogicalNot | Opcode::ToBooleanLogicalNot => {
                // Flip JIT_FALSE ↔ JIT_TRUE by toggling bit 0.
                // JIT_FALSE = 0x1_0000_0000 (bit 0 = 0) → XOR 1 → JIT_TRUE
                // JIT_TRUE  = 0x1_0000_0001 (bit 0 = 1) → XOR 1 → JIT_FALSE
                self.masm.xor_ri(Reg64::R12, 1);
            }

            // ── Control flow ─────────────────────────────────────────────────
            Opcode::Jump => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("Jump", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.emit_jump(target);
            }
            Opcode::JumpLoop => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpLoop", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.emit_jump(target);
            }
            Opcode::JumpIfTrue => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfTrue", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_TRUE
                self.masm.mov_ri(Reg64::R11, JIT_TRUE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfFalse => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfFalse", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfToBooleanTrue => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfToBooleanTrue", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc != JIT_FALSE (i.e. truthy).
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfToBooleanFalse => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfToBooleanFalse", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_FALSE (i.e. falsy).
                self.masm.mov_ri(Reg64::R11, JIT_FALSE);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNotNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNotNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfUndefined => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfUndefined", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }
            Opcode::JumpIfNotUndefined => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfNotUndefined", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::NotEqual, target);
            }
            Opcode::JumpIfUndefinedOrNull => {
                let Operand::JumpOffset(delta) = instr.operands[0] else {
                    return Err(bad_operand("JumpIfUndefinedOrNull", 0));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                // Jump if acc == JIT_UNDEFINED.
                self.masm.mov_ri(Reg64::R11, JIT_UNDEFINED);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
                // Also jump if acc == JIT_NULL.
                self.masm.mov_ri(Reg64::R11, JIT_NULL);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(CondCode::Equal, target);
            }

            // ── Return ───────────────────────────────────────────────────────
            Opcode::Return => {
                // Flush promoted globals back to GlobalEnv before returning.
                self.emit_promoted_global_stores();
                self.emit_normal_epilogue();
            }

            // ── No-ops / metadata ─────────────────────────────────────────────
            Opcode::StackCheck
            | Opcode::SetExpressionPosition
            | Opcode::SetExpressionPositionFromEnd
            | Opcode::CollectTypeProfile => {
                // These carry no runtime semantics; emit nothing.
            }

            // ── Runtime call stubs ───────────────────────────────────────────
            //
            // These opcodes call into the Rust runtime trampoline instead of
            // deopting the whole function.  The loop skeleton still runs
            // natively; only the complex operations go through Rust.
            Opcode::CreateEmptyObjectLiteral => {
                self.emit_runtime_stub(Opcode::CreateEmptyObjectLiteral, 0, 0, bytecode_offset);
            }

            Opcode::CreateObjectLiteral => {
                let feedback_slot = match instr.operands.get(1) {
                    Some(Operand::FeedbackSlot(s)) => i64::from(*s),
                    _ => -1,
                };
                let capacity = match instr.operands.get(2) {
                    Some(Operand::Flag(count)) if *count > 0 => i64::from(*count),
                    _ => 4,
                };
                self.emit_runtime_stub(
                    Opcode::CreateObjectLiteral,
                    feedback_slot,
                    capacity,
                    bytecode_offset,
                );
            }

            Opcode::CreateEmptyArrayLiteral => {
                self.emit_runtime_stub(Opcode::CreateEmptyArrayLiteral, 0, 0, bytecode_offset);
            }

            Opcode::CreateArrayLiteral => {
                self.emit_runtime_stub(Opcode::CreateArrayLiteral, 0, 0, bytecode_offset);
            }

            Opcode::LdaNamedProperty => {
                let Operand::Register(obj_v) = instr.operands[0] else {
                    return Err(bad_operand("LdaNamedProperty", 0));
                };
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
                    return Err(bad_operand("LdaNamedProperty", 1));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_lda_named_property_stub(obj_flat, name_idx, bytecode_offset);
            }

            Opcode::StaNamedProperty
            | Opcode::StaNamedOwnProperty
            | Opcode::DefineNamedOwnProperty => {
                let opname = match instr.opcode {
                    Opcode::StaNamedProperty => "StaNamedProperty",
                    Opcode::StaNamedOwnProperty => "StaNamedOwnProperty",
                    _ => "DefineNamedOwnProperty",
                };
                let Operand::Register(obj_v) = instr.operands[0] else {
                    return Err(bad_operand(opname, 0));
                };
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[1] else {
                    return Err(bad_operand(opname, 1));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_sta_named_property_stub(obj_flat, name_idx, bytecode_offset);
            }

            Opcode::LdaKeyedProperty => {
                let Operand::Register(obj_v) = instr.operands[0] else {
                    return Err(bad_operand("LdaKeyedProperty", 0));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_lda_keyed_property_stub(obj_flat, bytecode_offset);
            }

            Opcode::StaKeyedProperty => {
                let Operand::Register(obj_v) = instr.operands[0] else {
                    return Err(bad_operand("StaKeyedProperty", 0));
                };
                let Operand::Register(key_v) = instr.operands[1] else {
                    return Err(bad_operand("StaKeyedProperty", 1));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                let key_flat = self.reg_flat_index(key_v) as i64;
                self.emit_sta_keyed_property_stub(obj_flat, key_flat, bytecode_offset);
            }

            Opcode::StaInArrayLiteral => {
                let Operand::Register(arr_v) = instr.operands[0] else {
                    return Err(bad_operand("StaInArrayLiteral", 0));
                };
                let Operand::Register(idx_v) = instr.operands[1] else {
                    return Err(bad_operand("StaInArrayLiteral", 1));
                };
                let arr_flat = self.reg_flat_index(arr_v) as i64;
                let idx_flat = self.reg_flat_index(idx_v) as i64;
                self.emit_runtime_stub(
                    Opcode::StaInArrayLiteral,
                    arr_flat,
                    idx_flat,
                    bytecode_offset,
                );
            }

            Opcode::CallUndefinedReceiver0 => {
                let Operand::Register(callee_v) = instr.operands[0] else {
                    return Err(bad_operand("CallUndefinedReceiver0", 0));
                };
                let callee_flat = self.reg_flat_index(callee_v) as i64;
                self.emit_promoted_global_stores();
                self.emit_call_undefined_receiver0_stub(callee_flat, bytecode_offset);
                self.emit_promoted_global_loads();
            }

            Opcode::CallUndefinedReceiver1 => {
                let Operand::Register(callee_v) = instr.operands[0] else {
                    return Err(bad_operand("CallUndefinedReceiver1", 0));
                };
                let Operand::Register(arg1_v) = instr.operands[1] else {
                    return Err(bad_operand("CallUndefinedReceiver1", 1));
                };
                let callee_flat = self.reg_flat_index(callee_v) as i64;
                let arg1_flat = self.reg_flat_index(arg1_v) as i64;
                self.emit_promoted_global_stores();
                self.emit_runtime_stub(
                    Opcode::CallUndefinedReceiver1,
                    callee_flat,
                    arg1_flat,
                    bytecode_offset,
                );
                self.emit_promoted_global_loads();
            }

            // ── Context slot runtime stubs ───────────────────────────────────
            //
            // These use RT_CONTEXT thread-local to access the current closure
            // context. LdaCurrentContextSlot reads from the current context;
            // LdaContextSlot walks the chain via a register-held context.
            Opcode::LdaCurrentContextSlot | Opcode::LdaImmutableCurrentContextSlot => {
                let Operand::ConstantPoolIdx(slot) = instr.operands[0] else {
                    return Err(bad_operand("LdaCurrentContextSlot", 0));
                };
                self.emit_lda_context_slot_stub(slot, bytecode_offset);
            }

            Opcode::StaCurrentContextSlot => {
                let Operand::ConstantPoolIdx(slot) = instr.operands[0] else {
                    return Err(bad_operand("StaCurrentContextSlot", 0));
                };
                self.emit_sta_context_slot_stub(slot, bytecode_offset);
            }

            Opcode::LdaContextSlot | Opcode::LdaImmutableContextSlot => {
                let Operand::Register(ctx_v) = instr.operands[0] else {
                    return Err(bad_operand("LdaContextSlot", 0));
                };
                let Operand::ConstantPoolIdx(slot) = instr.operands[1] else {
                    return Err(bad_operand("LdaContextSlot", 1));
                };
                let ctx_flat = self.reg_flat_index(ctx_v) as i64;
                self.emit_runtime_stub(instr.opcode, i64::from(slot), ctx_flat, bytecode_offset);
            }

            Opcode::StaContextSlot => {
                let Operand::Register(ctx_v) = instr.operands[0] else {
                    return Err(bad_operand("StaContextSlot", 0));
                };
                let Operand::ConstantPoolIdx(slot) = instr.operands[1] else {
                    return Err(bad_operand("StaContextSlot", 1));
                };
                let ctx_flat = self.reg_flat_index(ctx_v) as i64;
                self.emit_runtime_stub(
                    Opcode::StaContextSlot,
                    i64::from(slot),
                    ctx_flat,
                    bytecode_offset,
                );
            }

            // ── Construct (new) runtime stub ─────────────────────────────────
            Opcode::Construct => {
                let Operand::Register(ctor_v) = instr.operands[0] else {
                    return Err(bad_operand("Construct", 0));
                };
                let Operand::Register(args_start_v) = instr.operands[1] else {
                    return Err(bad_operand("Construct", 1));
                };
                let Operand::RegisterCount(arg_count) = instr.operands[2] else {
                    return Err(bad_operand("Construct", 2));
                };
                let ctor_flat = self.reg_flat_index(ctor_v) as i64;
                let args_start_flat = self.reg_flat_index(args_start_v) as i64;
                // Pack args_start (high 16) and arg_count (low 16) into
                // operand2 so they fit the two-operand stub calling
                // convention.
                let packed = ((args_start_flat & 0xFFFF) << 16) | (i64::from(arg_count) & 0xFFFF);
                self.emit_promoted_global_stores();
                self.emit_runtime_stub(Opcode::Construct, ctor_flat, packed, bytecode_offset);
                self.emit_promoted_global_loads();
            }

            // ── Global variable specialized stubs ────────────────────────────
            Opcode::LdaGlobal => {
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
                    return Err(bad_operand("LdaGlobal", 0));
                };
                #[allow(unused_variables)]
                if let Some(flat) = self.promoted_slot_for(name_idx) {
                    // Promoted: load directly from register file.
                    #[cfg(all(target_arch = "x86_64", unix))]
                    {
                        self.masm.mov_load_base_disp32(
                            Reg64::R12,
                            Reg64::R14,
                            Self::promoted_offset(flat),
                        );
                    }
                    #[cfg(not(all(target_arch = "x86_64", unix)))]
                    {
                        self.emit_deopt(bytecode_offset);
                    }
                } else {
                    self.emit_lda_global_stub(name_idx, bytecode_offset);
                }
            }

            Opcode::StaGlobal => {
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
                    return Err(bad_operand("StaGlobal", 0));
                };
                #[allow(unused_variables)]
                if let Some(flat) = self.promoted_slot_for(name_idx) {
                    // Promoted: store directly into register file.
                    #[cfg(all(target_arch = "x86_64", unix))]
                    {
                        self.masm.mov_store_base_disp32(
                            Reg64::R14,
                            Self::promoted_offset(flat),
                            Reg64::R12,
                        );
                    }
                    #[cfg(not(all(target_arch = "x86_64", unix)))]
                    {
                        self.emit_deopt(bytecode_offset);
                    }
                } else {
                    self.emit_sta_global_stub(name_idx, bytecode_offset);
                }
            }

            // ── Dynamic scope lookup stub ────────────────────────────────────
            Opcode::LdaLookupSlot => {
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
                    return Err(bad_operand("LdaLookupSlot", 0));
                };
                self.emit_runtime_stub(
                    Opcode::LdaLookupSlot,
                    i64::from(name_idx),
                    0,
                    bytecode_offset,
                );
            }

            // ── ToString / TypeOf runtime stubs ──────────────────────────────
            Opcode::ToString => {
                self.emit_runtime_stub(Opcode::ToString, 0, 0, bytecode_offset);
            }

            Opcode::TypeOf => {
                self.emit_runtime_stub(Opcode::TypeOf, 0, 0, bytecode_offset);
            }

            // ── Bit shift runtime stubs ──────────────────────────────────────
            Opcode::ShiftLeft | Opcode::ShiftRight | Opcode::ShiftRightLogical => {
                let opname = match instr.opcode {
                    Opcode::ShiftLeft => "ShiftLeft",
                    Opcode::ShiftRight => "ShiftRight",
                    _ => "ShiftRightLogical",
                };
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand(opname, 0));
                };
                let lhs_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(instr.opcode, lhs_flat, 0, bytecode_offset);
            }

            // ── TestTypeOf runtime stub ──────────────────────────────────────
            Opcode::TestTypeOf => {
                let Operand::Flag(type_flag) = instr.operands[0] else {
                    return Err(bad_operand("TestTypeOf", 0));
                };
                self.emit_runtime_stub(
                    Opcode::TestTypeOf,
                    i64::from(type_flag),
                    0,
                    bytecode_offset,
                );
            }

            // ── TestInstanceOf runtime stub ──────────────────────────────────
            Opcode::TestInstanceOf => {
                let Operand::Register(rhs_v) = instr.operands[0] else {
                    return Err(bad_operand("TestInstanceOf", 0));
                };
                let rhs_flat = self.reg_flat_index(rhs_v) as i64;
                self.emit_runtime_stub(Opcode::TestInstanceOf, rhs_flat, 0, bytecode_offset);
            }

            // ── TestIn runtime stub ─────────────────────────────────────────
            Opcode::TestIn => {
                let Operand::Register(obj_v) = instr.operands[0] else {
                    return Err(bad_operand("TestIn", 0));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_runtime_stub(Opcode::TestIn, obj_flat, 0, bytecode_offset);
            }

            // ── ThrowReferenceErrorIfHole runtime stub ──────────────────────
            Opcode::ThrowReferenceErrorIfHole => {
                let Operand::ConstantPoolIdx(name_idx) = instr.operands[0] else {
                    return Err(bad_operand("ThrowReferenceErrorIfHole", 0));
                };
                self.emit_runtime_stub(
                    Opcode::ThrowReferenceErrorIfHole,
                    i64::from(name_idx),
                    0,
                    bytecode_offset,
                );
            }

            // ── LdaTheHole runtime stub ─────────────────────────────────────
            Opcode::LdaTheHole => {
                self.emit_runtime_stub(Opcode::LdaTheHole, 0, 0, bytecode_offset);
            }

            // ── ToNumber / ToNumeric runtime stubs ──────────────────────────
            Opcode::ToNumber | Opcode::ToNumeric => {
                self.emit_runtime_stub(instr.opcode, 0, 0, bytecode_offset);
            }

            // ── CreateClosure runtime stub ──────────────────────────────────
            Opcode::CreateClosure => {
                let Operand::ConstantPoolIdx(cp_idx) = instr.operands[0] else {
                    return Err(bad_operand("CreateClosure", 0));
                };
                self.emit_runtime_stub(
                    Opcode::CreateClosure,
                    i64::from(cp_idx),
                    0,
                    bytecode_offset,
                );
            }

            // ── CreateFunctionContext runtime stub ───────────────────────────
            Opcode::CreateFunctionContext => {
                let slot_count = match instr.operands.get(1) {
                    Some(Operand::Immediate(n)) => i64::from(*n),
                    _ => 0,
                };
                self.emit_runtime_stub(
                    Opcode::CreateFunctionContext,
                    slot_count,
                    0,
                    bytecode_offset,
                );
            }

            // ── PushContext / PopContext runtime stubs ────────────────────────
            Opcode::PushContext => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("PushContext", 0));
                };
                let reg_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(Opcode::PushContext, reg_flat, 0, bytecode_offset);
            }

            Opcode::PopContext => {
                let Operand::Register(v) = instr.operands[0] else {
                    return Err(bad_operand("PopContext", 0));
                };
                let reg_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(Opcode::PopContext, reg_flat, 0, bytecode_offset);
            }

            // ── Div / Mod runtime stubs ─────────────────────────────────────
            Opcode::Div => {
                let Operand::Register(lhs_v) = instr.operands[0] else {
                    return Err(bad_operand("Div", 0));
                };
                let lhs_flat = self.reg_flat_index(lhs_v) as i64;
                self.emit_runtime_stub(Opcode::Div, lhs_flat, 0, bytecode_offset);
            }

            Opcode::Mod => {
                let Operand::Register(lhs_v) = instr.operands[0] else {
                    return Err(bad_operand("Mod", 0));
                };
                let lhs_flat = self.reg_flat_index(lhs_v) as i64;
                self.emit_runtime_stub(Opcode::Mod, lhs_flat, 0, bytecode_offset);
            }

            // ── CallProperty0 specialized stub ─────────────────────────────
            Opcode::CallProperty0 => {
                let Operand::Register(callee_v) = instr.operands[0] else {
                    return Err(bad_operand("CallProperty0", 0));
                };
                let Operand::Register(receiver_v) = instr.operands[1] else {
                    return Err(bad_operand("CallProperty0", 1));
                };
                let callee_flat = self.reg_flat_index(callee_v) as i64;
                let receiver_flat = self.reg_flat_index(receiver_v) as i64;
                self.emit_promoted_global_stores();
                self.emit_call_property0_stub(callee_flat, receiver_flat, bytecode_offset);
                self.emit_promoted_global_loads();
            }

            // ── CallProperty1 specialized stub ─────────────────────────────
            Opcode::CallProperty1 => {
                let Operand::Register(callee_v) = instr.operands[0] else {
                    return Err(bad_operand("CallProperty1", 0));
                };
                let Operand::Register(receiver_v) = instr.operands[1] else {
                    return Err(bad_operand("CallProperty1", 1));
                };
                let Operand::Register(arg0_v) = instr.operands[2] else {
                    return Err(bad_operand("CallProperty1", 2));
                };
                let callee_flat = self.reg_flat_index(callee_v) as i64;
                let receiver_flat = self.reg_flat_index(receiver_v) as i64;
                let arg0_flat = self.reg_flat_index(arg0_v) as i64;
                self.emit_promoted_global_stores();
                self.emit_call_property1_stub(
                    callee_flat,
                    receiver_flat,
                    arg0_flat,
                    bytecode_offset,
                );
                self.emit_promoted_global_loads();
            }

            // ── DivSmi / ModSmi runtime stubs ───────────────────────────────
            Opcode::DivSmi => {
                let imm = match instr.operands.first() {
                    Some(Operand::Immediate(n)) => i64::from(*n),
                    _ => return Err(bad_operand("DivSmi", 0)),
                };
                self.emit_runtime_stub(Opcode::DivSmi, imm, 0, bytecode_offset);
            }

            Opcode::ModSmi => {
                let imm = match instr.operands.first() {
                    Some(Operand::Immediate(n)) => i64::from(*n),
                    _ => return Err(bad_operand("ModSmi", 0)),
                };
                self.emit_runtime_stub(Opcode::ModSmi, imm, 0, bytecode_offset);
            }

            // ── Remaining unsupported opcodes → deopt ────────────────────────
            Opcode::LdaNamedPropertyFromSuper
            | Opcode::LdaEnumeratedKeyedProperty
            | Opcode::DefineKeyedOwnProperty
            | Opcode::DefineKeyedOwnPropertyInLiteral
            | Opcode::SetLiteralPrototype
            | Opcode::LdaGlobalInsideTypeof
            | Opcode::LdaLookupContextSlot
            | Opcode::LdaLookupGlobalSlot
            | Opcode::LdaLookupSlotInsideTypeof
            | Opcode::LdaLookupContextSlotInsideTypeof
            | Opcode::LdaLookupGlobalSlotInsideTypeof
            | Opcode::StaLookupSlot
            | Opcode::DeleteLookupSlot
            | Opcode::CallAnyReceiver
            | Opcode::CallProperty
            | Opcode::CallProperty2
            | Opcode::CallUndefinedReceiver2
            | Opcode::CallWithSpread
            | Opcode::CallRuntime
            | Opcode::CallRuntimeForPair
            | Opcode::CallJSRuntime
            | Opcode::InvokeIntrinsic
            | Opcode::CallDirectEval
            | Opcode::TailCall
            | Opcode::ConstructWithSpread
            | Opcode::ConstructForwardAllArgs
            | Opcode::CreateBlockContext
            | Opcode::CreateCatchContext
            | Opcode::CreateEvalContext
            | Opcode::CreateWithContext
            | Opcode::CreateMappedArguments
            | Opcode::CreateUnmappedArguments
            | Opcode::CreateRestParameter
            | Opcode::CreateRegExpLiteral
            | Opcode::CreateArrayFromIterable
            | Opcode::CreateObjectFromIterable
            | Opcode::GetIterator
            | Opcode::GetAsyncIterator
            | Opcode::IteratorNext
            | Opcode::IteratorClose
            | Opcode::ForInEnumerate
            | Opcode::ForInPrepare
            | Opcode::ForInNext
            | Opcode::ForInStep
            | Opcode::JumpIfForInDone
            | Opcode::GetTemplateObject
            | Opcode::SwitchOnGeneratorState
            | Opcode::SuspendGenerator
            | Opcode::ResumeGenerator
            | Opcode::GetGeneratorState
            | Opcode::SetGeneratorState
            | Opcode::ToName
            | Opcode::ToObject
            | Opcode::ToBoolean
            | Opcode::DeletePropertyStrict
            | Opcode::DeletePropertySloppy
            | Opcode::TestReferenceEqual
            | Opcode::TestUndetectable
            | Opcode::Throw
            | Opcode::ReThrow
            | Opcode::SetPendingMessage
            | Opcode::ThrowSuperNotCalledIfHole
            | Opcode::ThrowSuperAlreadyCalledIfNotHole
            | Opcode::ThrowIfNullOrUndefined
            | Opcode::Debugger
            | Opcode::JumpConstant
            | Opcode::JumpIfTrueConstant
            | Opcode::JumpIfFalseConstant
            | Opcode::JumpIfToBooleanTrueConstant
            | Opcode::JumpIfToBooleanFalseConstant
            | Opcode::JumpIfNullConstant
            | Opcode::JumpIfNotNullConstant
            | Opcode::JumpIfUndefinedConstant
            | Opcode::JumpIfNotUndefinedConstant
            | Opcode::JumpIfUndefinedOrNullConstant
            | Opcode::JumpIfJSReceiver
            | Opcode::JumpIfJSReceiverConstant
            | Opcode::Exp
            | Opcode::ExpSmi
            | Opcode::ShiftLeftSmi
            | Opcode::ShiftRightSmi
            | Opcode::ShiftRightLogicalSmi
            | Opcode::DefineGetterProperty
            | Opcode::DefineSetterProperty
            | Opcode::DefineKeyedGetterProperty
            | Opcode::DefineKeyedSetterProperty
            | Opcode::DefineClassNamedOwnProperty
            | Opcode::DefineClassGetterProperty
            | Opcode::DefineClassSetterProperty
            | Opcode::DefineClassKeyedOwnProperty
            | Opcode::DefineClassKeyedGetterProperty
            | Opcode::DefineClassKeyedSetterProperty
            | Opcode::CopyDataProperties => {
                self.emit_deopt(bytecode_offset);
            }

            // These prefix/trap opcodes should never appear here.
            Opcode::CreateClass
            | Opcode::TestPrivateBrand
            | Opcode::DefinePrivateBrand
            | Opcode::LdaModuleVariable
            | Opcode::StaModuleVariable
            | Opcode::LdaImportMeta
            | Opcode::GetModuleNamespace
            | Opcode::Wide
            | Opcode::ExtraWide
            | Opcode::LdaNewTarget
            | Opcode::Illegal => {
                return Err(StatorError::Internal(format!(
                    "unexpected opcode in compilation: {:?}",
                    instr.opcode
                )));
            }
        }
        Ok(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the target byte offset for a jump with the given `delta`.
///
/// `idx` is the instruction index of the jump; `byte_offsets` is the table
/// produced by [`decode_with_byte_offsets`].  The target byte is `delta`
/// bytes past the end of the jump instruction.
fn jump_target_byte(idx: usize, delta: i32, byte_offsets: &[usize]) -> usize {
    // byte_offsets[idx + 1] = byte offset one past the jump instruction.
    let after = byte_offsets[idx + 1];
    (after as i64 + delta as i64) as usize
}

/// Construct a [`StatorError::Internal`] for an unexpected operand kind.
#[cold]
fn bad_operand(opcode: &'static str, i: usize) -> StatorError {
    StatorError::Internal(format!("{opcode}: unexpected operand at index {i}"))
}

/// Return the inverse condition code (e.g. `Less` → `GreaterEq`).
///
/// Used when fusing a comparison with `JumpIfFalse`: the jump fires when the
/// comparison was *false*, so we negate the original condition.
fn negate_cc(cc: CondCode) -> CondCode {
    match cc {
        CondCode::Less => CondCode::GreaterEq,
        CondCode::Greater => CondCode::LessEq,
        CondCode::LessEq => CondCode::Greater,
        CondCode::GreaterEq => CondCode::Less,
        CondCode::Equal => CondCode::NotEqual,
        CondCode::NotEqual => CondCode::Equal,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::BytecodeArray;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;
    use crate::interpreter::{Interpreter, InterpreterFrame};
    use crate::objects::value::JsValue;
    use std::rc::Rc;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a [`BytecodeArray`] from a list of instructions with no constant
    /// pool, one register and zero parameters.
    fn bytecode(instrs: Vec<Instruction>) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            4,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Run `ba` through the Stator interpreter and return the accumulator.
    fn interp_run(ba: BytecodeArray) -> JsValue {
        let mut frame = InterpreterFrame::new(Rc::new(ba), vec![]);
        Interpreter::run(&mut frame).expect("interpreter error")
    }

    // ── compile-only tests (no execution, all platforms) ─────────────────────

    #[test]
    fn test_compile_produces_non_empty_code() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert!(!cc.code.is_empty(), "compiled code must be non-empty");
    }

    #[test]
    fn test_compile_builds_safepoint_table() {
        // LdaSmi(7), Return  →  2 safepoint entries
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.safepoints.len(), 2);
        assert_eq!(cc.safepoints[0].bytecode_index, 0);
        assert_eq!(cc.safepoints[1].bytecode_index, 1);
    }

    #[test]
    fn test_compile_deopt_for_unsupported_opcode() {
        // LdaGlobal emits a deopt entry.
        let ba = bytecode(vec![
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.deopt_entries.len(), 1);
    }

    #[test]
    fn test_compile_register_file_slots() {
        let ba = BytecodeArray::new(
            encode(&[Instruction::new_unchecked(Opcode::Return, vec![])]),
            vec![],
            5, // frame_size
            2, // parameter_count
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.register_file_slots, 7); // 2 params + 5 locals
    }

    #[test]
    fn test_compile_empty_bytecode_errors() {
        let ba = BytecodeArray::new(
            vec![],
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        // An empty bytecode stream has no Return; compile succeeds but the
        // resulting code (prologue + deopt epilogue only) is still valid bytes.
        let cc = BaselineCompiler::compile(&ba).expect("compile should not error on empty stream");
        assert!(!cc.code.is_empty());
    }

    // ── execution tests (x86-64 + Unix only) ─────────────────────────────────

    /// Compile `ba`, execute the JIT code with `args`, and return the raw i64
    /// accumulator value.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn jit_run(ba: &BytecodeArray, args: &[i64]) -> i64 {
        let cc = BaselineCompiler::compile(ba).expect("compile failed");
        unsafe { cc.execute(args).expect("jit execute failed") }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_smi_return() {
        // LdaSmi(42), Return  →  42
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let result = jit_run(&ba, &[]);
        assert_eq!(result, 42);
        assert_eq!(interp_run(ba), JsValue::Smi(42));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_zero_return() {
        // LdaZero, Return  →  0
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let result = jit_run(&ba, &[]);
        assert_eq!(result, 0);
        assert_eq!(interp_run(ba), JsValue::Smi(0));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_addition() {
        // LdaSmi(3), Star(r0), LdaSmi(2), Add(r0, _), Return  →  5
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 5);
        assert_eq!(interp_result, JsValue::Smi(5));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_subtraction() {
        // LdaSmi(10) → r0; LdaSmi(4) → acc; Sub(r0): acc = acc − r0 = 4 − 10 = −6
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
            Instruction::new_unchecked(
                Opcode::Sub,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, -6);
        assert_eq!(interp_result, JsValue::Smi(-6));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_multiplication() {
        // LdaSmi(6), Star(r0), LdaSmi(7), Mul(r0, _), Return  →  42
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(6)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::Mul,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 42);
        assert_eq!(interp_result, JsValue::Smi(42));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_increment() {
        // LdaSmi(99), Inc(_), Return  →  100
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 100);
        assert_eq!(interp_result, JsValue::Smi(100));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_negate() {
        // LdaSmi(7), Negate(_), Return  →  -7
        // Note: Negate is not yet implemented in the tree-walking interpreter,
        // so we only compare the JIT result to the expected value.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Negate, vec![Operand::FeedbackSlot(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        assert_eq!(jit_result, -7);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_comparison_less_than_true() {
        // Store 5 in r0; load 3 into acc; TestLessThan(r0): acc = (3 < 5) = true → JIT_TRUE
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, JIT_TRUE);
        assert_eq!(interp_result, JsValue::Boolean(true));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_comparison_less_than_false() {
        // Store 3 in r0; load 5 into acc; TestLessThan(r0): acc = (5 < 3) = false → JIT_FALSE
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, JIT_FALSE);
        assert_eq!(interp_result, JsValue::Boolean(false));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_unconditional_jump() {
        // LdaSmi(1), Jump(+2 bytes → Return), LdaSmi(99), Return
        // (Jump skips the LdaSmi(99); should return 1)
        //
        // Byte layout (all narrow, each opcode + 1-byte operand = 2 bytes):
        //   0: LdaSmi(1)   2 bytes
        //   2: Jump(+2)    2 bytes  → target = end_of_jump(4) + 2 = 6 = Return
        //   4: LdaSmi(99)  2 bytes  (skipped)
        //   6: Return      1 byte
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(2)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = bytecode(instrs);
        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        assert_eq!(jit_result, 1);
        assert_eq!(interp_result, JsValue::Smi(1));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_sum_loop() {
        // Compute sum = 0; for (let i = 1; i <= 5; i++) sum += i;
        // Equivalent bytecode:
        //   LdaZero                ; acc = 0
        //   Star r0                ; sum = 0
        //   LdaSmi 1               ; acc = 1
        //   Star r1                ; i = 1
        // loop:
        //   LdaSmi 5
        //   Star r2                ; r2 = 5
        //   Ldar r1                ; acc = i
        //   TestLessThanOrEqual r2 ; acc = (i <= 5)
        //   JumpIfToBooleanFalse exit
        //   Ldar r0                ; acc = sum
        //   Add r1                 ; acc = sum + i
        //   Star r0                ; sum = acc
        //   Ldar r1                ; acc = i
        //   Inc                    ; acc = i + 1
        //   Star r1                ; i = acc
        //   Jump loop
        // exit:
        //   Ldar r0                ; acc = sum
        //   Return
        //
        // We need to compute the jump offsets manually based on the encoded size.
        // Let's build and encode step by step.

        // Opcode sizes (narrow encoding): each opcode is 1 byte + operands.
        // LdaZero: 1+0=1
        // Star r0: 1+1=2
        // LdaSmi 1: 1+1=2
        // Star r1: 1+1=2
        // --- loop starts here (offset = 7) ---
        // LdaSmi 5: 2
        // Star r2: 2
        // Ldar r1: 2
        // TestLessThanOrEqual r2: 1+1+1=3
        // JumpIfToBooleanFalse +N: 1+1=2  (need to compute N)
        // Ldar r0: 2
        // Add r1: 1+1+1=3
        // Star r0: 2
        // Ldar r1: 2
        // Inc: 1+1=2
        // Star r1: 2
        // Jump -M: 1+1=2  (need to compute M)
        // --- exit starts here ---
        // Ldar r0: 2
        // Return: 1

        // Let's compute offsets:
        // 0:  LdaZero (1)
        // 1:  Star r0 (2)
        // 3:  LdaSmi 1 (2)
        // 5:  Star r1 (2)
        // 7:  LdaSmi 5 (2)          ← loop top
        // 9:  Star r2 (2)
        // 11: Ldar r1 (2)
        // 13: TestLessThanOrEqual r2,slot (3)
        // 16: JumpIfToBooleanFalse δ (2) ← δ from end of this (byte 18) to exit
        // 18: Ldar r0 (2)
        // 20: Add r1,slot (3)
        // 23: Star r0 (2)
        // 25: Ldar r1 (2)
        // 27: Inc slot (2)
        // 29: Star r1 (2)
        // 31: Jump δ2 (2)            ← δ2 from byte 33 back to 7 = 7-33 = -26
        // 33: Ldar r0 (2)            ← exit
        // 35: Return (1)
        // JumpIfToBooleanFalse: from byte 18 to byte 33 → δ = 33 - 18 = 15
        // Jump: from byte 33 to byte 7 → δ = 7 - 33 = -26

        let instrs = vec![
            // 0: LdaZero
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            // 1: Star r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // 3: LdaSmi 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            // 5: Star r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // 7: LdaSmi 5   ← loop
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            // 9: Star r2
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // 11: Ldar r1
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            // 13: TestLessThanOrEqual r2
            Instruction::new_unchecked(
                Opcode::TestLessThanOrEqual,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            // 16: JumpIfToBooleanFalse +15
            Instruction::new_unchecked(Opcode::JumpIfToBooleanFalse, vec![Operand::JumpOffset(15)]),
            // 18: Ldar r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // 20: Add r1
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            // 23: Star r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // 25: Ldar r1
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            // 27: Inc
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
            // 29: Star r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // 31: Jump -26
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(-26)]),
            // 33: Ldar r0   ← exit
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // 35: Return
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instrs);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            4,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );

        let jit_result = jit_run(&ba, &[]);
        let interp_result = interp_run(ba);
        // sum = 1+2+3+4+5 = 15
        assert_eq!(jit_result, 15);
        assert_eq!(interp_result, JsValue::Smi(15));
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_lda_true_false() {
        let ba_true = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba_true, &[]), JIT_TRUE);

        let ba_false = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba_false, &[]), JIT_FALSE);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_logical_not() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::LogicalNot, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        assert_eq!(jit_run(&ba, &[]), JIT_FALSE);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_jit_deopt_for_global_load() {
        let ba = bytecode(vec![
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        // execute() should return Err("jit deopt") since LdaGlobal deopts.
        let result = unsafe { cc.execute(&[]) };
        assert!(
            matches!(result, Err(ref e) if e.to_string().contains("deopt")),
            "expected deopt error, got: {result:?}"
        );
    }

    // ── safepoint / deopt table tests (all platforms) ────────────────────────

    #[test]
    fn test_safepoint_gc_map_is_zero_for_smi_only() {
        // The current JIT tier only handles Smi/boolean/null/undefined values;
        // no slot ever holds a GC-managed heap pointer → gc_map must be zero.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        for sp in &cc.safepoints {
            assert_eq!(sp.gc_map, 0, "no GC pointers in Smi-only JIT code");
        }
    }

    #[test]
    fn test_deopt_liveness_map_covers_all_slots() {
        // 2 params + 3 locals = 5 register-file slots.
        // The deopt liveness_map must have bits 0–4 set (= 0b1_1111 = 31).
        let ba = BytecodeArray::new(
            encode(&[
                Instruction::new_unchecked(
                    Opcode::LdaGlobal,
                    vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ]),
            vec![],
            3, // frame_size = 3 locals
            2, // parameter_count = 2 params
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert_eq!(cc.deopt_entries.len(), 1);
        let expected_liveness = (1u64 << 5) - 1; // bits 0..4 set
        assert_eq!(
            cc.deopt_entries[0].liveness_map, expected_liveness,
            "liveness_map must cover all 5 register-file slots"
        );
    }

    #[test]
    fn test_native_code_len_excludes_tables() {
        // The metadata tables are appended after the native instructions, so
        // native_code_len must be strictly less than the total code length.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");
        assert!(
            cc.native_code_len < cc.code.len(),
            "metadata tables must be appended after native code"
        );
        // The last FOOTER_SIZE bytes must contain the magic number.
        let footer_off = cc.code.len() - FOOTER_SIZE;
        let magic =
            u32::from_le_bytes(cc.code[footer_off + 8..footer_off + 12].try_into().unwrap());
        assert_eq!(
            magic, METADATA_MAGIC,
            "footer magic must match METADATA_MAGIC"
        );
    }

    #[test]
    fn test_tables_round_trip_through_serialized_code() {
        // The safepoint and deopt tables serialized into code bytes must be
        // parseable back to entries that are identical to the in-memory ones.
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");

        // Round-trip safepoint table.
        let parsed_sp =
            CompiledCode::parse_safepoints(&cc.code).expect("safepoint table must be parseable");
        assert_eq!(parsed_sp.len(), cc.safepoints.len());
        for (parsed, orig) in parsed_sp.iter().zip(&cc.safepoints) {
            assert_eq!(parsed.code_offset, orig.code_offset);
            assert_eq!(parsed.bytecode_index, orig.bytecode_index);
            assert_eq!(parsed.gc_map, orig.gc_map);
        }

        // Round-trip deopt table.
        let parsed_de =
            CompiledCode::parse_deopt_entries(&cc.code).expect("deopt table must be parseable");
        assert_eq!(parsed_de.len(), cc.deopt_entries.len());
        for (parsed, orig) in parsed_de.iter().zip(&cc.deopt_entries) {
            assert_eq!(parsed.code_offset, orig.code_offset);
            assert_eq!(parsed.bytecode_offset, orig.bytecode_offset);
            assert_eq!(parsed.liveness_map, orig.liveness_map);
        }
    }

    #[test]
    fn test_simulate_gc_stack_scan() {
        // Simulate what the GC would do at a safepoint:
        //   2. Use the PC (code_offset) to look up the SafepointEntry.
        //   3. Read gc_map to determine which register-file slots hold GC roots.
        // In the current Smi-only JIT, gc_map must always be zero (no roots).
        let ba = bytecode(vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
        let cc = BaselineCompiler::compile(&ba).expect("compile failed");

        // The first safepoint is at the very first instruction.
        let sp0 = &cc.safepoints[0];

        // Simulate GC: look up safepoint by code_offset.
        let found = cc
            .find_safepoint(sp0.code_offset)
            .expect("safepoint lookup must succeed");

        // No GC roots in a Smi-only JIT frame.
        assert_eq!(found.gc_map, 0, "no GC roots in Smi-only JIT frame");
        assert_eq!(found.bytecode_index, 0);

        // Also verify via the parsed tables (as the GC would use them from the
        // serialized code bytes embedded in the Code object).
        let parsed =
            CompiledCode::parse_safepoints(&cc.code).expect("must parse from serialized code");
        let found_parsed = parsed
            .iter()
            .find(|e| e.code_offset == sp0.code_offset)
            .expect("parsed safepoint lookup must succeed");
        assert_eq!(found_parsed.gc_map, 0);
    }

    /// `jit_to_jsvalue` must promote i64 values outside the i32 range to
    /// `HeapNumber` so that large-integer JIT results (e.g. sum 1..1_000_000)
    /// are returned correctly without deoptimizing.
    #[test]
    fn test_jit_to_jsvalue_large_integer() {
        use crate::objects::value::JsValue;
        // A value that fits in i32 must remain a Smi.
        assert_eq!(jit_to_jsvalue(42), Some(JsValue::Smi(42)));
        assert_eq!(
            jit_to_jsvalue(i32::MAX as i64),
            Some(JsValue::Smi(i32::MAX))
        );
        assert_eq!(
            jit_to_jsvalue(i32::MIN as i64),
            Some(JsValue::Smi(i32::MIN))
        );

        // A value outside the i32 range must be promoted to HeapNumber.
        let large: i64 = (i32::MAX as i64) + 1;
        match jit_to_jsvalue(large) {
            Some(JsValue::HeapNumber(n)) => {
                assert!((n - large as f64).abs() < 1.0, "HeapNumber value mismatch");
            }
            other => panic!("expected HeapNumber, got {:?}", other),
        }

        // 500_000_500_000 — the result of sum(1..1_000_000).
        let sum_result: i64 = 500_000_500_000;
        match jit_to_jsvalue(sum_result) {
            Some(JsValue::HeapNumber(n)) => {
                assert!((n - sum_result as f64).abs() < 1.0);
            }
            other => panic!("expected HeapNumber for large sum, got {:?}", other),
        }
    }
}

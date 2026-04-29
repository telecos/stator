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
use std::collections::HashMap;
#[cfg(all(target_arch = "x86_64", unix))]
use std::rc::Rc;
#[cfg(all(target_arch = "x86_64", unix))]
use std::sync::atomic::AtomicU32;

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
/// Deopt reason: CheckedSmi overflow (arithmetic result out of i32 range).
pub const JIT_DEOPT_OVERFLOW: i64 = i64::MIN + 1;
/// Deopt reason: a runtime stub returned JIT_DEOPT.
pub const JIT_DEOPT_STUB: i64 = i64::MIN + 2;
/// Deopt reason: promoted global load failed.
pub const JIT_DEOPT_GLOBAL: i64 = i64::MIN + 3;
/// Deopt reason: integer division by zero.
pub const JIT_DEOPT_DIVZERO: i64 = i64::MIN + 5;

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
/// Returns `None` if `v` is the [`JIT_DEOPT`] sentinel (the JIT requested
/// a fall-back to the interpreter).  All other values are mapped to a
/// concrete `JsValue`: sentinels → booleans/undefined/null, i32-range →
/// Smi, and everything else is reinterpreted as an IEEE 754 `f64` bit
/// pattern (the inverse of the `jsvalue_to_jit` HeapNumber encoding).
pub fn jit_to_jsvalue(v: i64) -> Option<crate::objects::value::JsValue> {
    use crate::objects::value::JsValue;
    // Fast path: Smi values are the overwhelmingly common case (loop
    // counters, arithmetic results, array indices).  Check the i32
    // range FIRST to avoid 4 wasted comparisons against magic sentinels.
    if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
        Some(JsValue::Smi(v as i32))
    } else if v == JIT_FALSE {
        Some(JsValue::Boolean(false))
    } else if v == JIT_TRUE {
        Some(JsValue::Boolean(true))
    } else if v == JIT_UNDEFINED {
        Some(JsValue::Undefined)
    } else if v == JIT_NULL {
        Some(JsValue::Null)
    } else {
        // Value outside Smi range — promote to HeapNumber via lossy f64 cast.
        // JIT arithmetic that overflows i32 produces large i64 values that
        // must be presented to JS as floating-point numbers.
        Some(JsValue::HeapNumber(v as f64))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime call-stub infrastructure (x86-64 + Unix only)
// ─────────────────────────────────────────────────────────────────────────────

// Re-export `jit_full_teardown` so benchmark crates can access it as
// `compiler::baseline::compiler::jit_full_teardown()`.
#[cfg(all(target_arch = "x86_64", unix))]
pub use jit_runtime::jit_full_teardown;

/// Thread-local state used by the JIT runtime trampoline to access the
/// constant pool and to store heap-allocated JavaScript objects that cannot
/// be encoded as plain `i64` values.
#[cfg(all(target_arch = "x86_64", unix))]
pub(crate) mod jit_runtime {
    use super::*;
    use crate::bytecode::bytecode_array::JitExecutableCode;
    use crate::interpreter::GlobalEnv;
    use crate::objects::map::PropertyAttributes;
    use crate::objects::property_map::{
        INTERNAL_PROTO_PROPERTY_KEY, ObjectLiteralTemplate, PropertyMap,
        acquire_object_rc_from_template_cached, acquire_object_rc_from_template_with_values_cached,
        recycle_object_rc,
    };
    use crate::objects::value::{JsContext, JsValue, NativeIterator, recycle_context_rc};

    /// One slot of the own-property inline cache for `LdaNamedProperty`.
    #[derive(Clone, Copy)]
    struct LdaIcEntry {
        /// Constant-pool index of the property name.
        name_idx: u32,
        /// Shape id of the object at the time the IC was populated.
        shape: u64,
        /// Fast property offset inside the object's property storage.
        offset: usize,
        /// Heap-handle value for the cached result (0 if not heap).
        cached_handle: i64,
        /// Raw `Rc` pointer of the cached heap value (0 if not heap).
        cached_ptr: usize,
    }

    impl LdaIcEntry {
        /// Sentinel value used to initialise / clear the IC.
        const EMPTY: Self = Self {
            name_idx: u32::MAX,
            shape: 0,
            offset: 0,
            cached_handle: 0,
            cached_ptr: 0,
        };
    }

    /// One slot of the prototype-chain inline cache.
    ///
    /// In addition to the shape-based validation (existing), this entry
    /// stores the receiver's JIT heap handle and the global prototype
    /// mutation epoch at the time the IC was populated.  This enables a
    /// **handle-based fast path** that returns the cached value without
    /// dereferencing the heap→Rc→PropertyMap→shape chain, as long as the
    /// global epoch has not advanced (i.e. no prototype in the program has
    /// been mutated since the entry was written).
    #[derive(Clone, Copy)]
    struct ProtoIcEntry {
        /// Constant-pool index of the property name.
        name_idx: u32,
        /// Shape id of the receiver at IC population time.
        shape: u64,
        /// JIT i64 heap handle of the receiver object.
        receiver_handle: i64,
        /// Global prototype mutation epoch at IC population time.
        global_epoch: u64,
        /// Pre-encoded JIT i64 result value.
        cached_value: i64,
    }

    impl ProtoIcEntry {
        const EMPTY: Self = Self {
            name_idx: u32::MAX,
            shape: 0,
            receiver_handle: 0,
            global_epoch: u64::MAX,
            cached_value: 0,
        };
    }

    /// Combined inline caches for named-property stubs.
    ///
    /// Merging the LDA and prototype ICs into one TLS variable saves
    /// one `thread_local!` lookup on the IC-miss path (where both
    /// caches are probed sequentially).
    struct JitPropertyIcState {
        /// Own-property IC, indexed by `name_idx & 63`.
        lda: [LdaIcEntry; 64],
        /// Prototype-chain IC, direct-mapped by `name_idx & 31`.
        proto: [ProtoIcEntry; 32],
        /// Set to `true` on every IC fill.  Checked by `jit_runtime_setup`
        /// to skip the per-entry handle invalidation loop when no fills
        /// occurred since the last setup (~50-80ns savings).
        dirty: bool,
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

        /// Identity-based handle deduplication map.
        ///
        /// Maps `Rc::as_ptr()` identity (as `usize`) to the heap index
        /// returned by [`alloc_heap_handle`].  Ensures that the same
        /// reference-counted object always gets the same JIT handle,
        /// which is critical for inline-cache stability on chained
        /// property accesses (e.g. `obj.a.b.c.d.e`).
        static RT_HANDLE_DEDUP: RefCell<HashMap<usize, usize>> =
            RefCell::new(HashMap::new());

        /// Combined own-property + prototype inline caches for
        /// `LdaNamedProperty` stubs.  Cleared by [`jit_runtime_setup`]
        /// to prevent cross-function constant-pool index collisions.
        static RT_PROP_IC: RefCell<JitPropertyIcState> = const {
            RefCell::new(JitPropertyIcState {
                lda: [LdaIcEntry::EMPTY; 64],
                proto: [ProtoIcEntry::EMPTY; 32],
                dirty: false,
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

        /// Monotonically increasing counter bumped by [`jit_runtime_setup`].
        /// Compared against `RtPtrs::generation` to detect stale caches
        /// after nested JIT execution re-initialises thread-local state.
        static RT_SETUP_GEN: Cell<u64> = const { Cell::new(0) };

        /// Set by [`jit_runtime_teardown`] after draining the heap.
        /// Checked by [`jit_runtime_setup`] to skip the redundant
        /// `recycle_and_clear_heap` + cache clears when teardown already
        /// performed them.
        static RT_HEAP_CLEAN: Cell<bool> = const { Cell::new(false) };

        /// Monotonically increasing counter bumped by every named-property
        /// store (STA) in JIT stubs.  Combined with the prototype-mutation
        /// epoch to guard proto-IC entries that cache own-property values.
        static RT_VALUE_WRITE_EPOCH: Cell<u64> = const { Cell::new(0) };

        /// Last [`BytecodeArray`] pointer **and bytecodes length** passed
        /// to [`jit_runtime_setup`].  Used to skip IC reset when the same
        /// function re-enters (e.g. criterion iterations measuring the
        /// same benchmark).
        ///
        /// Storing the length alongside the pointer detects allocator
        /// address reuse: when a `BytecodeArray` is dropped and a new one
        /// is allocated at the same address, the lengths will almost
        /// certainly differ, forcing an IC reset.
        static RT_LAST_BA: Cell<(*const BytecodeArray, usize)> = const {
            Cell::new((std::ptr::null(), 0))
        };

        /// Single-entry IC that caches the most recent array method
        /// lookup.  Avoids recreating the fast-array-method
        /// `PlainObject` wrapper on every iteration of a tight
        /// `arr.push()` / `arr.pop()` loop.
        ///
        /// Cleared by [`jit_runtime_setup`] to avoid stale handles.
        static RT_ARRAY_METHOD_IC: Cell<ArrayMethodIcEntry> = const {
            Cell::new(ArrayMethodIcEntry::EMPTY)
        };

        /// Single-entry IC for `call_property1` dispatch.  Caches the
        /// callee heap handle so that repeated `arr.push(v)` calls in
        /// a tight loop skip the `PropertyMap` lookup used to identify
        /// the fast-array-method tag.
        ///
        /// Unconditionally cleared by [`jit_runtime_setup`] because
        /// heap handles are recycled between invocations.
        static RT_CALL_PROP1_IC: Cell<CallProp1IcEntry> = const {
            Cell::new(CallProp1IcEntry::EMPTY)
        };

        /// Single-entry IC that caches the most recent object-literal
        /// template pointer.  Avoids the `RefCell` borrow and `HashMap`
        /// lookup on the `object_literal_templates` map for repeated
        /// creation of the same literal shape (e.g. `{x:1, y:2, z:3}`
        /// inside a tight loop).
        ///
        /// Cleared by [`jit_runtime_setup`] when the [`BytecodeArray`]
        /// changes.
        static RT_OBJECT_IC: Cell<ObjectLiteralIcEntry> = const {
            Cell::new(ObjectLiteralIcEntry::EMPTY)
        };

        /// Cached resolved pattern for [`SpeculativeCallFusion`] runtime
        /// stub.  Stores only value-type data (no pointers) so it is
        /// safe to keep across JIT invocations without invalidation.
        ///
        /// On a cache hit (same `callee` heap handle, which is
        /// deterministic when the heap is cleared between iterations),
        /// the fusion function skips the [`Rc`] deref into the
        /// [`OnceCell`]-based pattern cache and the pattern-match
        /// destructure — going straight to the heap lookup for the
        /// closure context.
        static RT_FUSION_CACHE: Cell<FusionFastCache> = const {
            Cell::new(FusionFastCache::EMPTY)
        };

        /// Cached Vec pointer for `jit_runtime_array_push`.  Avoids
        /// repeated heap → JsValue::Array → Rc<RefCell<Vec>> dereferences
        /// when the receiver is the same array across iterations.
        static RT_PUSH_CACHE: Cell<PushVecCache> = const {
            Cell::new(PushVecCache::EMPTY)
        };
    }

    // ── Direct JIT-to-JIT call state ────────────────────────────────
    //
    // Saved by `jit_runtime_get_jit_entry`, restored by
    // `jit_runtime_finish_direct_call`.
    //
    // Bundled into a single `Cell` so that save/restore is one TLS
    // access instead of three.

    /// Per-call state stashed before a direct JIT-to-JIT call.
    #[derive(Clone, Copy)]
    struct DirectCallState {
        saved_ba: *const BytecodeArray,
        heap_base: usize,
        ctx_changed: bool,
    }

    thread_local! {
        static DIRECT_CALL_STATE: Cell<DirectCallState> = const {
            Cell::new(DirectCallState {
                saved_ba: std::ptr::null(),
                heap_base: 0,
                ctx_changed: false,
            })
        };

        /// Previous context saved before a direct-call context swap.
        static DIRECT_CALL_OLD_CTX: RefCell<Option<Rc<RefCell<JsContext>>>> =
            const { RefCell::new(None) };

        /// Single-entry callee cache for `call_undefined_receiver0_inner`.
        ///
        /// When a tight loop calls the same closure repeatedly, the heap
        /// index (`callee_i64`) is identical on every iteration.  This
        /// cache stores `(callee_i64, *const BytecodeArray)` so that the
        /// heap lookup + `JsValue` match can be skipped on a hit.
        ///
        /// # Safety
        ///
        /// The raw pointer is derived from an `Rc<BytecodeArray>` that
        /// is kept alive in the heap (`RT_HEAP`) for the duration of
        /// the JIT execution.  The cache is only read on the same
        /// thread and is invalidated whenever `callee_i64` changes.
        static CACHED_CALLEE: Cell<CachedCalleeEntry> =
            const { Cell::new(CachedCalleeEntry::EMPTY) };

        /// When `true`, the `CreateMappedArguments` trampoline returns
        /// `JIT_UNDEFINED` immediately instead of building a full arguments
        /// object.  Set by the closure call path when the callee has 0
        /// formal parameters (the arguments object is dead code).
        static SKIP_MAPPED_ARGS: Cell<bool> = const { Cell::new(false) };

        /// Per-stub deopt failure tracking.  Each index corresponds to a
        /// specific runtime stub (see `STUB_*` constants).
        static RT_STUB_DEOPT_COUNTS: Cell<[u64; STUB_DEOPT_SLOTS]> =
            const { Cell::new([0; STUB_DEOPT_SLOTS]) };

        /// Whether the first deopt in the current invocation has already
        /// been recorded.  Reset to `false` by [`jit_runtime_setup`].
        static RT_FIRST_DEOPT_REPORTED: Cell<bool> = const { Cell::new(false) };

        /// Per-stub *first*-deopt counts.  Like `RT_STUB_DEOPT_COUNTS`,
        /// but only incremented for the very first deopt in each
        /// invocation (i.e. after each [`jit_runtime_setup`] call).
        static RT_FIRST_DEOPT_COUNTS: Cell<[u64; STUB_DEOPT_SLOTS]> =
            const { Cell::new([0; STUB_DEOPT_SLOTS]) };

        /// Per-stub total call counts.  Incremented on every FFI stub
        /// entry (both fast and slow paths) to diagnose whether inline
        /// IC fast paths are firing or all calls fall to the FFI.
        static RT_STUB_CALL_COUNTS: Cell<[u64; STUB_DEOPT_SLOTS]> =
            const { Cell::new([0; STUB_DEOPT_SLOTS]) };

        /// Repeat-callee cache for [`exec_maglev_callee`].
        ///
        /// When a tight loop calls the same closure repeatedly, this cache
        /// stores the resolved context pointer and comparison result so that
        /// per-call context lookups and pointer comparisons are skipped.
        static MAGLEV_CALLEE_CACHE: Cell<MaglevCalleeCache> =
            const { Cell::new(MaglevCalleeCache::EMPTY) };
    }

    // ── Per-stub deopt tracking ─────────────────────────────────────────

    /// Number of slots in the per-stub deopt counter array.
    pub const STUB_DEOPT_SLOTS: usize = 24;

    /// Stub deopt tracking indices.
    pub const STUB_LDA_NAMED: usize = 0;
    pub const STUB_STA_NAMED: usize = 1;
    pub const STUB_LDA_GLOBAL: usize = 2;
    pub const STUB_STA_GLOBAL: usize = 3;
    pub const STUB_CONSTRUCT0: usize = 4;
    pub const STUB_CONSTRUCT1: usize = 5;
    pub const STUB_CONSTRUCT2: usize = 6;
    pub const STUB_CREATE_OBJ_WITH_PROPS: usize = 7;
    pub const STUB_FAST_CREATE_OBJ: usize = 8;
    pub const STUB_LDA_KEYED: usize = 9;
    pub const STUB_STA_KEYED: usize = 10;
    pub const STUB_CALL_UNDEF0: usize = 11;
    pub const STUB_CREATE_CLOSURE: usize = 12;
    pub const STUB_STA_NAMED_OWN: usize = 13;
    pub const STUB_CALL_PROP0: usize = 14;
    pub const STUB_CALL_PROP1: usize = 15;
    pub const STUB_CALL_UNDEF1: usize = 16;
    pub const STUB_CALL_UNDEF2: usize = 17;
    pub const STUB_FAST_ARRAY_LOAD: usize = 18;
    pub const STUB_FAST_ARRAY_STORE: usize = 19;
    pub const STUB_FAST_ARRAY_PUSH: usize = 20;
    pub const STUB_TRAMPOLINE: usize = 21;
    pub const STUB_GENERIC_ARITH: usize = 22;

    /// Record a deopt for the stub at the given index.
    #[inline]
    fn track_stub_deopt(idx: usize) {
        RT_STUB_DEOPT_COUNTS.with(|c| {
            let mut arr = c.get();
            arr[idx] = arr[idx].saturating_add(1);
            c.set(arr);
        });
        // Track first-deopt-per-invocation.
        RT_FIRST_DEOPT_REPORTED.with(|reported| {
            if !reported.get() {
                reported.set(true);
                RT_FIRST_DEOPT_COUNTS.with(|c| {
                    let mut arr = c.get();
                    arr[idx] = arr[idx].saturating_add(1);
                    c.set(arr);
                });
            }
        });
    }

    /// Return the current per-stub deopt counts.
    pub fn stub_deopt_counts() -> [u64; STUB_DEOPT_SLOTS] {
        RT_STUB_DEOPT_COUNTS.with(|c| c.get())
    }

    /// Reset all per-stub deopt counts to zero.
    pub fn reset_stub_deopt_counts() {
        RT_STUB_DEOPT_COUNTS.with(|c| c.set([0; STUB_DEOPT_SLOTS]));
    }

    /// Return the current per-stub *first*-deopt counts.
    pub fn first_deopt_counts() -> [u64; STUB_DEOPT_SLOTS] {
        RT_FIRST_DEOPT_COUNTS.with(|c| c.get())
    }

    /// Reset all per-stub first-deopt counts to zero.
    pub fn reset_first_deopt_counts() {
        RT_FIRST_DEOPT_COUNTS.with(|c| c.set([0; STUB_DEOPT_SLOTS]));
    }

    /// Record a call to the FFI stub at the given index.
    #[inline]
    fn track_stub_call(idx: usize) {
        RT_STUB_CALL_COUNTS.with(|c| {
            let mut arr = c.get();
            arr[idx] = arr[idx].saturating_add(1);
            c.set(arr);
        });
    }

    /// Return the current per-stub call counts.
    pub fn stub_call_counts() -> [u64; STUB_DEOPT_SLOTS] {
        RT_STUB_CALL_COUNTS.with(|c| c.get())
    }

    /// Reset all per-stub call counts to zero.
    pub fn reset_stub_call_counts() {
        RT_STUB_CALL_COUNTS.with(|c| c.set([0; STUB_DEOPT_SLOTS]));
    }

    /// Cached fusion-pattern data for [`SpeculativeCallFusion`] runtime stub.
    ///
    /// Stores only value-type data (heap-handle identity, slot index,
    /// increment constant) — no raw pointers — so the cache is safe to
    /// keep across `jit_runtime_setup` / `jit_runtime_teardown` cycles.
    #[derive(Clone, Copy)]
    struct FusionFastCache {
        /// Callee heap handle used as the cache key.
        callee: i64,
        /// Context-slot index from the fusion pattern.
        slot: usize,
        /// Increment constant from the fusion pattern.
        k: i64,
    }

    impl FusionFastCache {
        const EMPTY: Self = Self {
            callee: 0,
            slot: 0,
            k: 0,
        };
    }

    // ── Global fusion context cache (for Maglev inline fast path) ───
    //
    // Caches the result of the last `jit_runtime_fusion_ctx_slot_ptr`
    // call so that the Maglev codegen can check it directly from
    // generated machine code (via the global address embedded as an
    // immediate) without any function call on cache hit.
    //
    // The engine is single-threaded, so `AtomicI64` with `Relaxed`
    // ordering compiles to plain loads/stores on x86.

    use std::sync::atomic::{AtomicI64, Ordering};

    /// Cached callee heap handle for the fusion context inline cache.
    static FUSION_CTX_CALLEE: AtomicI64 = AtomicI64::new(0);

    /// Cached raw pointer to the target `JsValue` slot for the fusion
    /// context inline cache.
    static FUSION_CTX_SLOT_PTR: AtomicI64 = AtomicI64::new(0);

    /// Return the address of [`FUSION_CTX_CALLEE`] for embedding as
    /// an absolute address immediate in JIT machine code.
    pub fn fusion_ctx_callee_addr() -> usize {
        &FUSION_CTX_CALLEE as *const AtomicI64 as usize
    }

    /// Return the address of [`FUSION_CTX_SLOT_PTR`] for embedding as
    /// an absolute address immediate in JIT machine code.
    pub fn fusion_ctx_slot_ptr_addr() -> usize {
        &FUSION_CTX_SLOT_PTR as *const AtomicI64 as usize
    }

    /// Lightweight fusion resolve stub: given a callee heap handle and
    /// a compile-time–known slot index, returns a raw pointer to the
    /// target `JsValue` slot in the closure context.
    ///
    /// Unlike [`jit_runtime_fusion_resolve`], this skips bytecode pattern
    /// analysis entirely — `(slot, k)` are compile-time constants embedded
    /// in the IR.  On success the global inline cache is also updated so
    /// subsequent calls from the Maglev fast path hit the cache and
    /// bypass this stub entirely.
    ///
    /// Returns `0` on any failure (non-heap-handle, missing context, etc.).
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_fusion_ctx_slot_ptr(callee_i64: i64, slot_index: i64) -> i64 {
        if !is_heap_handle(callee_i64) {
            return 0;
        }
        let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
        let slot = slot_index as usize;

        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return 0;
        }
        // SAFETY: cached pointer valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if idx >= heap.len() {
            return 0;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds check above.
        let ba = match unsafe { heap.get_unchecked(idx) } {
            JsValue::Function(ba) => ba,
            _ => return 0,
        };

        let Some(ctx_rc) = ba.closure_context() else {
            return 0;
        };
        // SAFETY: single-threaded JIT execution — no concurrent borrows.
        let ctx = unsafe { &*ctx_rc.as_ptr() };
        if slot >= ctx.slots.len() {
            return 0;
        }
        let slot_ptr = unsafe { ctx.slots.as_ptr().add(slot) } as i64;

        // Update the global inline cache.
        FUSION_CTX_CALLEE.store(callee_i64, Ordering::Relaxed);
        FUSION_CTX_SLOT_PTR.store(slot_ptr, Ordering::Relaxed);

        slot_ptr
    }

    /// Extended single-entry callee cache entry.
    ///
    /// Beyond the `(callee_i64, ba_ptr)` pair, this also caches the
    /// closure context pointer, JIT entry function pointer, and
    /// register-file slot count.  On a full cache hit with `entry_fn != 0`,
    /// the dispatch path can bypass `ba.jit_executable_cache()`,
    /// `ba.closure_context()`, and `exec.execute()` entirely — calling
    /// the JIT code pointer directly.
    #[derive(Clone, Copy)]
    struct CachedCalleeEntry {
        callee_i64: i64,
        ba_ptr: *const BytecodeArray,
        ctx_ptr: i64,
        entry_fn: usize,
        reg_slots: usize,
        /// Cached Maglev code entry point (0 = not available).
        maglev_fn: usize,
        /// Maglev register-file slot count.
        maglev_reg_slots: usize,
        /// Cached `ba.parameter_count() == 0` — avoids a virtual call
        /// on every cached hit.
        skip_args: bool,
        /// Raw pointer from `Rc::into_raw(Rc::clone(ctx))`.  Holds one
        /// strong reference count so that per-call context swaps can use
        /// `Rc::from_raw` / `Rc::into_raw` without touching atomic
        /// refcounts.  Must be dropped via [`drop_cached_ctx_rc_raw`]
        /// when the entry is invalidated.
        ctx_rc_raw: *const RefCell<JsContext>,
    }

    impl CachedCalleeEntry {
        const EMPTY: Self = Self {
            callee_i64: 0,
            ba_ptr: std::ptr::null(),
            ctx_ptr: 0,
            entry_fn: 0,
            reg_slots: 0,
            maglev_fn: 0,
            maglev_reg_slots: 0,
            skip_args: false,
            ctx_rc_raw: std::ptr::null(),
        };
    }

    /// Repeat-callee cache for [`exec_maglev_callee`].
    ///
    /// Caches the resolved callee identity and context-comparison result
    /// so that repeated calls to the same closure skip the
    /// `closure_context()` lookup, context pointer read, and pointer
    /// comparison on every iteration.
    #[derive(Clone, Copy)]
    struct MaglevCalleeCache {
        /// Raw pointer to the callee's [`BytecodeArray`].  `null` = empty.
        ba_ptr: *const BytecodeArray,
        /// Cached callee context pointer (as `i64` for direct use in JIT).
        ctx_ptr: i64,
        /// Whether the callee's context pointer matched the runtime
        /// context on the last call.  When `true`, all context
        /// save/restore work can be skipped.
        same_context: bool,
        /// Cached `ba.parameter_count() == 0`.
        skip_args: bool,
        /// Raw pointer from `Rc::into_raw(Rc::clone(ctx))`.  Holds one
        /// strong reference count so that the cached `ctx_ptr` remains
        /// valid even after `set_closure_context` replaces the inner
        /// `Rc`.  Must be dropped via [`drop_cached_ctx_rc_raw`] when
        /// the entry is invalidated or replaced.
        ctx_rc_raw: *const RefCell<JsContext>,
    }

    impl MaglevCalleeCache {
        const EMPTY: Self = Self {
            ba_ptr: std::ptr::null(),
            ctx_ptr: 0,
            same_context: false,
            skip_args: false,
            ctx_rc_raw: std::ptr::null(),
        };
    }

    /// Drop the `Rc` refcount held by a [`CachedCalleeEntry::ctx_rc_raw`]
    /// pointer.  Must be called before overwriting an entry whose
    /// `ctx_rc_raw` is non-null with a different callee or EMPTY.
    fn drop_cached_ctx_rc_raw(raw: *const RefCell<JsContext>) {
        if !raw.is_null() {
            // SAFETY: `raw` was produced by `Rc::into_raw(Rc::clone(..))`.
            // The matching decrement here balances that clone.
            unsafe {
                drop(Rc::from_raw(raw));
            }
        }
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
    #[repr(C)]
    struct RtPtrs {
        heap: *const RefCell<Vec<JsValue>>,
        context: *const RefCell<Option<Rc<RefCell<JsContext>>>>,
        bytecode: *const Cell<*const BytecodeArray>,
        global: *const RefCell<JitGlobalState>,
        prop_ic: *const RefCell<JitPropertyIcState>,
        /// Pointer to the `Cell<DirectCallState>` thread-local, cached
        /// so that `prepare` and `finish` avoid a second TLS lookup.
        direct_call: *const Cell<DirectCallState>,
        /// Pointer to the `Cell<CachedCalleeEntry>` thread-local.
        cached_callee: *const Cell<CachedCalleeEntry>,
        /// Generation counter set at cache time.  Compared against
        /// [`RT_SETUP_GEN`] before use; a mismatch forces a re-cache.
        #[allow(dead_code)]
        generation: u64,
        /// Pointer to the `Cell<bool>` for `SKIP_MAPPED_ARGS`.
        skip_mapped_args: *const Cell<bool>,
        /// Cached pointer to the `Cell<ObjectLiteralIcEntry>` thread-
        /// local, eliminating a TLS lookup on every object creation.
        object_ic: *const Cell<ObjectLiteralIcEntry>,
        /// Cached pointer to the `OBJECT_RC_POOL` thread-local, so that
        /// `acquire_object_rc_from_template*` variants can skip TLS.
        object_rc_pool: *const RefCell<Vec<Rc<RefCell<PropertyMap>>>>,
        /// Cached pointer to the `MAGLEV_CALLEE_CACHE` thread-local.
        maglev_callee_cache: *const Cell<MaglevCalleeCache>,
    }

    impl RtPtrs {
        const EMPTY: Self = Self {
            heap: std::ptr::null(),
            context: std::ptr::null(),
            bytecode: std::ptr::null(),
            global: std::ptr::null(),
            prop_ic: std::ptr::null(),
            direct_call: std::ptr::null(),
            cached_callee: std::ptr::null(),
            generation: 0,
            skip_mapped_args: std::ptr::null(),
            object_ic: std::ptr::null(),
            object_rc_pool: std::ptr::null(),
            maglev_callee_cache: std::ptr::null(),
        };

        fn is_cached(&self) -> bool {
            !self.heap.is_null()
                && !self.bytecode.is_null()
                && (self.bytecode as usize)
                    .is_multiple_of(std::mem::align_of::<Cell<*const BytecodeArray>>())
        }

        /// Store [`DirectCallState`] via the cached pointer, avoiding a
        /// separate `DIRECT_CALL_STATE.with()` TLS lookup.
        fn set_direct_call(&self, state: DirectCallState) {
            if !self.direct_call.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.direct_call }.set(state);
            } else {
                DIRECT_CALL_STATE.with(|c| c.set(state));
            }
        }

        /// Read [`DirectCallState`] via the cached pointer.
        fn get_direct_call(&self) -> DirectCallState {
            if !self.direct_call.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.direct_call }.get()
            } else {
                DIRECT_CALL_STATE.with(|c| c.get())
            }
        }

        /// Read [`CachedCalleeEntry`] via the cached pointer.
        fn get_cached_callee(&self) -> CachedCalleeEntry {
            if !self.cached_callee.is_null() {
                unsafe { &*self.cached_callee }.get()
            } else {
                CACHED_CALLEE.with(|c| c.get())
            }
        }

        /// Store [`CachedCalleeEntry`] via the cached pointer.
        fn set_cached_callee(&self, entry: CachedCalleeEntry) {
            if !self.cached_callee.is_null() {
                unsafe { &*self.cached_callee }.set(entry);
            } else {
                CACHED_CALLEE.with(|c| c.set(entry));
            }
        }

        /// Read `SKIP_MAPPED_ARGS` via the cached pointer.
        fn get_skip_mapped_args(&self) -> bool {
            if !self.skip_mapped_args.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.skip_mapped_args }.get()
            } else {
                SKIP_MAPPED_ARGS.with(|c| c.get())
            }
        }

        /// Store `SKIP_MAPPED_ARGS` via the cached pointer.
        fn set_skip_mapped_args(&self, val: bool) {
            if !self.skip_mapped_args.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.skip_mapped_args }.set(val);
            } else {
                SKIP_MAPPED_ARGS.with(|c| c.set(val));
            }
        }

        /// Read the object-literal IC entry via the cached pointer.
        #[inline(always)]
        fn get_object_ic(&self) -> ObjectLiteralIcEntry {
            if !self.object_ic.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.object_ic }.get()
            } else {
                RT_OBJECT_IC.with(|c| c.get())
            }
        }

        /// Store the object-literal IC entry via the cached pointer.
        #[inline(always)]
        fn set_object_ic(&self, entry: ObjectLiteralIcEntry) {
            if !self.object_ic.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.object_ic }.set(entry);
            } else {
                RT_OBJECT_IC.with(|c| c.set(entry));
            }
        }

        /// Read [`MaglevCalleeCache`] via the cached pointer.
        #[inline(always)]
        fn get_maglev_callee_cache(&self) -> MaglevCalleeCache {
            if !self.maglev_callee_cache.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.maglev_callee_cache }.get()
            } else {
                MAGLEV_CALLEE_CACHE.with(|c| c.get())
            }
        }

        /// Store [`MaglevCalleeCache`] via the cached pointer.
        #[inline(always)]
        fn set_maglev_callee_cache(&self, entry: MaglevCalleeCache) {
            if !self.maglev_callee_cache.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*self.maglev_callee_cache }.set(entry);
            } else {
                MAGLEV_CALLEE_CACHE.with(|c| c.set(entry));
            }
        }
    }

    /// fast-array-method `PlainObject` wrappers in tight loops.
    /// fast-array-method `PlainObject` wrappers in tight loops.
    #[derive(Clone, Copy)]
    struct ArrayMethodIcEntry {
        /// Heap handle of the receiver array.
        receiver: i64,
        /// Constant-pool index of the property name.
        name_idx: u32,
        /// Heap handle of the cached method `PlainObject`.
        method: i64,
    }

    impl ArrayMethodIcEntry {
        const EMPTY: Self = Self {
            receiver: 0,
            name_idx: 0,
            method: 0,
        };
    }

    /// Single-entry IC for `CallProperty1` dispatch.
    ///
    /// Caches the callee heap handle together with a tag that identifies
    /// the fast method (currently only `Array.prototype.push`).  On an IC
    /// hit, [`call_property1_inner`] skips the `PlainObject` deref and
    /// `PropertyMap::get("\0stator.fast_array_method")` hash-map probe
    /// that otherwise dominates tight `arr.push()` loops.
    #[derive(Clone, Copy)]
    struct CallProp1IcEntry {
        /// Heap handle of the callee last seen.
        callee: i64,
        /// Identified method tag — `0` means empty/miss.
        tag: u8,
    }

    impl CallProp1IcEntry {
        const EMPTY: Self = Self { callee: 0, tag: 0 };
        const TAG_ARRAY_PUSH: u8 = 1;
    }

    /// Single-entry cache for `jit_runtime_array_push`.
    ///
    /// Caches the `*mut Vec<JsValue>` raw pointer obtained from the last
    /// receiver array so that repeated `arr.push(v)` calls in a tight
    /// loop skip the heap dereference, `JsValue::Array` match, and
    /// `Rc<RefCell<_>>::as_ptr()` chain.
    ///
    /// # Safety
    ///
    /// The `vec_ptr` is derived from `Rc<RefCell<Vec<JsValue>>>::as_ptr()`.
    /// The owning `Rc` lives in the heap, which is alive during JIT
    /// execution.  The cache is cleared in [`jit_runtime_setup`] when the
    /// heap is recycled.
    #[derive(Clone, Copy)]
    struct PushVecCache {
        /// Receiver heap handle of the last push call.
        receiver: i64,
        /// Raw pointer to the inner `Vec<JsValue>`.
        vec_ptr: *mut Vec<JsValue>,
    }

    impl PushVecCache {
        const EMPTY: Self = Self {
            receiver: 0,
            vec_ptr: std::ptr::null_mut(),
        };
    }

    // SAFETY: PushVecCache is only accessed from the JIT thread.
    unsafe impl Send for PushVecCache {}

    /// Single-entry inline cache for object-literal template lookups.
    ///
    /// Caches a raw pointer to the [`ObjectLiteralTemplate`] stored inside
    /// the [`BytecodeArray`]'s `object_literal_templates` map so that the
    /// hot path can skip the `RefCell` borrow and `HashMap` probe on
    /// repeated calls with the same `(slot, ba)` pair.
    ///
    /// # Safety
    ///
    /// The `template` pointer is derived from a
    /// `Box<ObjectLiteralTemplate>` inside
    /// `ObjectLiteralCacheEntry::Cached`.  That heap allocation is stable
    /// because cached entries are never removed or replaced.  The IC is
    /// invalidated when the [`BytecodeArray`] changes (see
    /// [`jit_runtime_setup`]).
    #[derive(Clone, Copy)]
    struct ObjectLiteralIcEntry {
        /// Feedback slot this entry is for.
        slot: u32,
        /// BytecodeArray pointer this entry belongs to.
        ba: *const BytecodeArray,
        /// Raw pointer to the cached template.
        template: *const ObjectLiteralTemplate,
    }

    impl ObjectLiteralIcEntry {
        const EMPTY: Self = Self {
            slot: u32::MAX,
            ba: std::ptr::null(),
            template: std::ptr::null(),
        };
    }

    /// Populate [`RT_PTRS`] so that runtime stubs can bypass per-variable
    /// `.with()` calls.
    ///
    /// # Safety
    ///
    /// The cached pointers reference thread-local `RefCell`s whose addresses
    /// are stable for the thread's lifetime.  Stubs create short-lived
    /// references from `as_ptr()` and never hold them across nested calls,
    /// so no aliasing violations occur even when Maglev stubs invoke
    /// `call_js_function` (which re-enters the interpreter on a fresh frame).
    fn cache_rt_ptrs() {
        RT_HEAP.with(|heap| {
            RT_CONTEXT.with(|ctx| {
                RT_BYTECODE.with(|bc| {
                    RT_GLOBAL.with(|g| {
                        RT_PROP_IC.with(|ic| {
                            DIRECT_CALL_STATE.with(|dc| {
                                CACHED_CALLEE.with(|cc| {
                                    SKIP_MAPPED_ARGS.with(|sma| {
                                        RT_OBJECT_IC.with(|oic| {
                                            MAGLEV_CALLEE_CACHE.with(|mcc| {
                                                let pool_ptr =
                                                    crate::objects::property_map::object_rc_pool_ptr();
                                                let setup_gen = RT_SETUP_GEN.with(|g| g.get());
                                                RT_PTRS.with(|p| {
                                                    p.set(RtPtrs {
                                                        heap: heap as *const RefCell<Vec<JsValue>>,
                                                        context: ctx as *const RefCell<
                                                            Option<Rc<RefCell<JsContext>>>,
                                                        >,
                                                        bytecode: bc as *const Cell<
                                                            *const BytecodeArray,
                                                        >,
                                                        global: g
                                                            as *const RefCell<JitGlobalState>,
                                                        prop_ic: ic as *const RefCell<
                                                            JitPropertyIcState,
                                                        >,
                                                        direct_call: dc
                                                            as *const Cell<DirectCallState>,
                                                        cached_callee: cc as *const Cell<
                                                            CachedCalleeEntry,
                                                        >,
                                                        generation: setup_gen,
                                                        skip_mapped_args: sma
                                                            as *const Cell<bool>,
                                                        object_ic: oic as *const Cell<
                                                            ObjectLiteralIcEntry,
                                                        >,
                                                        object_rc_pool: pool_ptr,
                                                        maglev_callee_cache: mcc
                                                            as *const Cell<MaglevCalleeCache>,
                                                    });
                                                });
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    }

    // ── Setup / teardown ─────────────────────────────────────────────────

    /// Prepare thread-local state before JIT execution.
    ///
    /// Must be called before `CompiledCode::execute` so that runtime call
    /// stubs can access the constant pool.
    pub fn jit_runtime_setup(ba: &BytecodeArray) {
        // Bump the generation counter so that stale cached pointers from
        // a previous setup epoch are detected and re-cached.
        RT_SETUP_GEN.with(|g| g.set(g.get().wrapping_add(1)));
        // Reset the first-deopt flag so the next deopt is recorded.
        RT_FIRST_DEOPT_REPORTED.with(|r| r.set(false));
        // Skip the expensive 13-TLS-call cache_rt_ptrs() when pointers
        // are already cached — TLS cell addresses are stable for the
        // thread's lifetime.  Only update the generation field.
        let already_cached = RT_PTRS.with(|p| {
            let ptrs = p.get();
            if ptrs.is_cached() {
                let new_gen = RT_SETUP_GEN.with(|g| g.get());
                p.set(RtPtrs {
                    generation: new_gen,
                    ..ptrs
                });
                true
            } else {
                false
            }
        });
        if !already_cached {
            cache_rt_ptrs();
        }
        let ba_ptr = ba as *const BytecodeArray;
        RT_BYTECODE.with(|b| b.set(ba_ptr));
        // Skip heap recycle + cache clears when teardown already did them.
        let heap_clean = RT_HEAP_CLEAN.with(|c| c.replace(false));
        if !heap_clean {
            recycle_and_clear_heap();
            RT_CALL_PROP1_IC.with(|c| c.set(CallProp1IcEntry::EMPTY));
            RT_PUSH_CACHE.with(|c| c.set(PushVecCache::EMPTY));
            CACHED_CALLEE.with(|c| {
                let old = c.get();
                drop_cached_ctx_rc_raw(old.ctx_rc_raw);
                c.set(CachedCalleeEntry::EMPTY);
            });
            MAGLEV_CALLEE_CACHE.with(|c| {
                let old = c.get();
                drop_cached_ctx_rc_raw(old.ctx_rc_raw);
                c.set(MaglevCalleeCache::EMPTY);
            });
        }
        // Only reset the property and array-method ICs when the function
        // changes.  Re-entrant calls to the same BytecodeArray (e.g.
        // criterion iterations) keep warm IC state, saving ~190ns per
        // named property access on IC-miss paths.
        //
        // We compare both the pointer AND the bytecodes length to detect
        // allocator address reuse: when a BytecodeArray is dropped and a
        // new one is allocated at the same address, the bytecodes length
        // will almost certainly differ, forcing an IC reset.
        let ba_len = ba.bytecodes().len();
        let same_ba = RT_LAST_BA.with(|prev| {
            let (was_ptr, was_len) = prev.get();
            prev.set((ba_ptr, ba_len));
            was_ptr == ba_ptr && was_len == ba_len
        });
        if !same_ba {
            RT_PROP_IC.with(|ic| {
                let mut c = ic.borrow_mut();
                c.lda = [LdaIcEntry::EMPTY; 64];
                c.proto = [ProtoIcEntry::EMPTY; 32];
            });
            RT_ARRAY_METHOD_IC.with(|c| c.set(ArrayMethodIcEntry::EMPTY));
            RT_OBJECT_IC.with(|c| c.set(ObjectLiteralIcEntry::EMPTY));
        } else {
            // Only invalidate IC handles if fills happened since the
            // last setup.  For benchmarks that don't access named
            // properties (e.g. closure_counter), this skips the entire
            // 96-entry loop (~50-80ns savings).
            let needs_invalidation = RT_PROP_IC.with(|ic| ic.borrow().dirty);
            if needs_invalidation {
                // IC kept warm for performance, but heap handles cached inside
                // the IC entries are stale after `recycle_and_clear_heap()`.
                // Invalidate only the handle fields — shape, offset, and name
                // information remains valid so the next access re-populates
                // quickly (single IC-miss instead of a full cold start).
                RT_PROP_IC.with(|ic| {
                    let mut c = ic.borrow_mut();
                    for entry in c.lda.iter_mut() {
                        entry.cached_handle = 0;
                        entry.cached_ptr = 0;
                    }
                    for entry in c.proto.iter_mut() {
                        entry.receiver_handle = 0;
                        if is_heap_handle(entry.cached_value) {
                            *entry = ProtoIcEntry::EMPTY;
                        }
                    }
                    c.dirty = false;
                });
                // Array-method IC caches receiver and method heap handles.
                RT_ARRAY_METHOD_IC.with(|c| c.set(ArrayMethodIcEntry::EMPTY));
            }
        }
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
            let same_env = state.env.as_ref().is_some_and(|old| Rc::ptr_eq(old, &env));
            if same_env {
                return;
            }
            state.env = Some(env);
            state.ic = [(u32::MAX, 0, 0); 64];
        });
    }

    /// Clean up thread-local state after JIT execution.
    ///
    /// Performs the minimal cleanup needed for correctness: clears the
    /// bytecode pointer, recycles heap objects, and drops stale Maglev
    /// callee cache entries.
    ///
    /// Note: does NOT clear `RT_GLOBAL`, `RT_PROP_IC`, or `RT_PTRS` —
    /// they persist across JIT calls so that repeated invocations of the
    /// same function benefit from warm IC state and skip the expensive
    /// 13-TLS-call `cache_rt_ptrs()` in setup.
    pub fn jit_runtime_teardown() {
        RT_BYTECODE.with(|b| b.set(std::ptr::null()));
        recycle_and_clear_heap();
        // Drop stale Rc held by the Maglev callee cache so the old
        // closure context is freed promptly.
        MAGLEV_CALLEE_CACHE.with(|c| {
            let old = c.get();
            drop_cached_ctx_rc_raw(old.ctx_rc_raw);
            c.set(MaglevCalleeCache::EMPTY);
        });
        // Also clear the caches that setup would clear anyway — then
        // set the flag so setup can skip the redundant work.
        RT_CALL_PROP1_IC.with(|c| c.set(CallProp1IcEntry::EMPTY));
        RT_PUSH_CACHE.with(|c| c.set(PushVecCache::EMPTY));
        CACHED_CALLEE.with(|c| {
            let old = c.get();
            drop_cached_ctx_rc_raw(old.ctx_rc_raw);
            c.set(CachedCalleeEntry::EMPTY);
        });
        RT_HEAP_CLEAN.with(|c| c.set(true));
    }

    /// Comprehensive teardown of **all** JIT thread-local state.
    ///
    /// Unlike [`jit_runtime_teardown`], which intentionally preserves warm
    /// IC state and cached contexts for the next JIT call, this function
    /// releases *every* `Rc` and raw-pointer reference held in JIT
    /// thread-locals.  Call it once when the thread is about to exit
    /// (e.g. at the end of `main()` in benchmark binaries) so that
    /// all reference-counted objects are dropped while the TLS variables
    /// they transitively reference are still alive — preventing
    /// use-after-free crashes during non-deterministic TLS destruction.
    #[allow(dead_code)] // Called from bench binaries, not from the lib itself.
    pub fn jit_full_teardown() {
        // ── Standard teardown ──────────────────────────────────────
        RT_BYTECODE.with(|b| b.set(std::ptr::null()));
        recycle_and_clear_heap();
        RT_PTRS.with(|p| p.set(RtPtrs::EMPTY));

        // ── Drop leaked Rc in callee caches ────────────────────────
        CACHED_CALLEE.with(|c| {
            let entry = c.get();
            drop_cached_ctx_rc_raw(entry.ctx_rc_raw);
            c.set(CachedCalleeEntry::EMPTY);
        });
        MAGLEV_CALLEE_CACHE.with(|c| {
            let cache = c.get();
            drop_cached_ctx_rc_raw(cache.ctx_rc_raw);
            c.set(MaglevCalleeCache::EMPTY);
        });

        // ── Release Rc-holding runtime TLS ─────────────────────────
        RT_CONTEXT.with(|c| *c.borrow_mut() = None);
        RT_GLOBAL.with(|g| {
            let mut state = g.borrow_mut();
            state.env = None;
            state.ic = [(u32::MAX, 0, 0); 64];
        });
        DIRECT_CALL_OLD_CTX.with(|c| *c.borrow_mut() = None);
        RT_LAST_BA.with(|b| b.set((std::ptr::null(), 0)));
        RT_FUSION_CACHE.with(|c| c.set(FusionFastCache::EMPTY));
        RT_PUSH_CACHE.with(|c| c.set(PushVecCache::EMPTY));
        FUSION_CTX_CALLEE.store(0, Ordering::Relaxed);
        FUSION_CTX_SLOT_PTR.store(0, Ordering::Relaxed);

        // ── Clear heap so recycled PlainObjects don't linger ───────
        RT_HEAP.with(|h| h.borrow_mut().clear());
        RT_HANDLE_DEDUP.with(|m| m.borrow_mut().clear());

        // ── Drain property-map pools that may hold Rc/JsValue ──────
        crate::objects::property_map::clear_property_map_pools();

        // ── Drain context recycling pool ───────────────────────────
        crate::objects::value::clear_context_pool();

        // ── Drain array-vec recycling pool ─────────────────────────
        crate::objects::value::clear_array_vec_pool();
    }

    /// Set the current closure context for context-slot stubs.
    ///
    /// Called before JIT execution of closure bodies that use
    /// `LdaCurrentContextSlot` / `StaCurrentContextSlot`.
    pub fn jit_runtime_set_context(ctx: Option<Rc<RefCell<JsContext>>>) {
        RT_CONTEXT.with(|c| *c.borrow_mut() = ctx);
    }

    // ── Heap-handle helpers ──────────────────────────────────────────────

    /// Drain the JIT heap, recycling `PlainObject` `Rc` wrappers into the
    /// thread-local object pool for reuse by future allocations.
    fn recycle_and_clear_heap() {
        RT_HEAP.with(|h| {
            let mut heap = h.borrow_mut();
            for val in heap.drain(..) {
                match val {
                    JsValue::PlainObject(rc) => recycle_object_rc(rc),
                    JsValue::Context(rc) => recycle_context_rc(rc),
                    JsValue::Array(rc) => {
                        // Recycle the inner Vec buffer if we're the sole owner.
                        if let Ok(inner) = Rc::try_unwrap(rc) {
                            crate::objects::value::recycle_array_vec(inner.into_inner());
                        }
                    }
                    _ => {}
                }
            }
            // Pre-reserve capacity on first use so that object-heavy
            // loops (e.g. 1000 iterations) avoid repeated Vec
            // reallocations on their first execution.
            if heap.capacity() < 1024 {
                heap.reserve(1024);
            }
        });
        RT_HANDLE_DEDUP.with(|m| m.borrow_mut().clear());
    }

    /// Truncate the JIT heap back to `base`, recycling `PlainObject` `Rc`
    /// wrappers from the truncated region.
    fn recycle_and_truncate_heap(base: usize) {
        RT_HEAP.with(|h| {
            let mut heap = h.borrow_mut();
            for val in heap.drain(base..) {
                match val {
                    JsValue::PlainObject(rc) => recycle_object_rc(rc),
                    JsValue::Context(rc) => recycle_context_rc(rc),
                    JsValue::Array(rc) => {
                        if let Ok(inner) = Rc::try_unwrap(rc) {
                            crate::objects::value::recycle_array_vec(inner.into_inner());
                        }
                    }
                    _ => {}
                }
            }
        });
        RT_HANDLE_DEDUP.with(|m| m.borrow_mut().retain(|_, idx| *idx < base));
    }

    /// Returns `true` if `v` is a heap-object handle.
    #[inline]
    fn is_heap_handle(v: i64) -> bool {
        (JIT_HEAP_TAG..JIT_HEAP_TAG + 0x1_0000_0000).contains(&v)
    }

    /// Bump the value-write epoch.  Called on every named-property store
    /// so that proto-IC entries caching own-property values are
    /// invalidated.
    #[inline(always)]
    fn bump_value_write_epoch() {
        RT_VALUE_WRITE_EPOCH.with(|e| e.set(e.get().wrapping_add(1)));
    }

    /// Read the current value-write epoch.
    #[inline(always)]
    fn current_value_write_epoch() -> u64 {
        RT_VALUE_WRITE_EPOCH.with(Cell::get)
    }

    /// Compute a combined epoch from the prototype-mutation counter and
    /// the value-write counter.  Used as a single guard for proto-IC
    /// entries that may cache own-property values.
    #[inline(always)]
    fn combined_ic_epoch() -> u64 {
        let proto = crate::objects::property_map::PropertyMap::global_proto_mutation_epoch();
        let value = current_value_write_epoch();
        proto.wrapping_add(value)
    }

    /// Returns `true` when `v` is a Smi (fits in i32 range).
    #[inline(always)]
    fn is_smi(v: i64) -> bool {
        v == (v as i32) as i64
    }

    /// Returns `true` when `v` is any JIT deopt sentinel
    /// (`JIT_DEOPT` through `JIT_DEOPT_DIVZERO`).
    #[inline]
    pub(crate) fn is_jit_deopt(v: i64) -> bool {
        (v as u64).wrapping_sub(JIT_DEOPT as u64) <= 5
    }

    /// Allocate a new heap handle for `val`, returning the `i64` handle.
    fn alloc_heap_handle(val: JsValue) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        alloc_heap_handle_with_ptrs(val, &ptrs)
    }

    /// Extract a stable pointer identity from reference-counted JsValue
    /// variants.  Returns `None` for inline types (Smi, bool, etc.)
    /// that don't carry a heap pointer.
    #[inline]
    fn jsvalue_rc_identity(val: &JsValue) -> Option<usize> {
        match val {
            JsValue::PlainObject(rc) => Some(Rc::as_ptr(rc) as usize),
            JsValue::Array(rc) => Some(Rc::as_ptr(rc) as usize),
            JsValue::Function(rc) => Some(Rc::as_ptr(rc) as usize),
            JsValue::String(rc) => Some(Rc::as_ptr(rc) as *const () as usize),
            JsValue::Error(rc) => Some(Rc::as_ptr(rc) as usize),
            JsValue::Generator(rc) => Some(Rc::as_ptr(rc) as usize),
            JsValue::Iterator(rc) => Some(Rc::as_ptr(rc) as usize),
            _ => None,
        }
    }

    /// Allocate a heap handle using pre-fetched [`RtPtrs`].
    ///
    /// Deduplicates by `Rc` pointer identity so that the same
    /// reference-counted object always receives the same handle.
    /// This is critical for inline-cache stability on chained property
    /// accesses (e.g. `obj.a.b.c.d.e`).
    #[inline]
    fn alloc_heap_handle_with_ptrs(val: JsValue, ptrs: &RtPtrs) -> i64 {
        // Fast path: check dedup map for existing handle.
        if let Some(identity) = jsvalue_rc_identity(&val) {
            let hit = RT_HANDLE_DEDUP.with(|m| m.borrow().get(&identity).copied());
            if let Some(idx) = hit {
                // Validate: the dedup entry may be stale if a raw
                // `truncate()` shortened the heap without pruning
                // the dedup map.
                let valid = if ptrs.is_cached() {
                    // SAFETY: cached pointer valid for thread lifetime.
                    let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
                    idx < heap.len()
                } else {
                    RT_HEAP.with(|h| idx < h.borrow().len())
                };
                if valid {
                    return JIT_HEAP_TAG + idx as i64;
                }
                // Stale entry — remove it and fall through to allocate
                // a fresh handle.
                RT_HANDLE_DEDUP.with(|m| {
                    m.borrow_mut().remove(&identity);
                });
            }
        }

        let identity = jsvalue_rc_identity(&val);

        let idx = if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime; no concurrent borrows.
            let heap = unsafe { &mut *(&*ptrs.heap).as_ptr() };
            let idx = heap.len();
            heap.push(val);
            idx
        } else {
            RT_HEAP.with(|heap| {
                let mut heap = heap.borrow_mut();
                let idx = heap.len();
                heap.push(val);
                idx
            })
        };

        if let Some(key) = identity {
            RT_HANDLE_DEDUP.with(|m| {
                m.borrow_mut().insert(key, idx);
            });
        }

        JIT_HEAP_TAG + idx as i64
    }

    /// Allocate a heap handle for a freshly created object, skipping the
    /// [`RT_HANDLE_DEDUP`] map entirely.
    ///
    /// This is safe for objects returned by `CreateObjectLiteral` /
    /// `CreateObjectLiteralWithProperties` because:
    ///
    /// * The `Rc` was just popped from the pool or freshly allocated, so
    ///   it cannot already be in the dedup map under a *valid* index.
    /// * The caller's handle is the only reference that JIT code uses;
    ///   no other code path will call `alloc_heap_handle` with the same
    ///   `Rc` identity during this JIT execution.
    ///
    /// Eliminating the dedup map avoids two TLS lookups and two
    /// `HashMap` operations (get + insert) per object creation — the
    /// dominant cost on the IC-hit hot path.
    #[inline(always)]
    fn alloc_heap_handle_no_dedup(val: JsValue, ptrs: &RtPtrs) -> i64 {
        let idx = if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime; no
            // concurrent borrows.
            let heap = unsafe { &mut *(&*ptrs.heap).as_ptr() };
            let idx = heap.len();
            heap.push(val);
            idx
        } else {
            RT_HEAP.with(|heap| {
                let mut heap = heap.borrow_mut();
                let idx = heap.len();
                heap.push(val);
                idx
            })
        };
        JIT_HEAP_TAG + idx as i64
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
            match get_heap_object(v) {
                Some(val) => val,
                None => JsValue::Undefined,
            }
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
    #[allow(dead_code)]
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
        get_rt_string_constant_ref_with_ptrs(idx, &ptrs)
    }

    /// Like [`get_rt_string_constant_ref`] but accepts pre-fetched
    /// [`RtPtrs`] to avoid a redundant TLS lookup when the caller has
    /// already obtained the cached pointers.
    #[inline]
    fn get_rt_string_constant_ref_with_ptrs(idx: u32, ptrs: &RtPtrs) -> Option<&'static str> {
        let ptr = if ptrs.is_cached() {
            // SAFETY: cached pointer valid for thread lifetime; alignment
            // verified by is_cached().
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
        jit_runtime_dispatch(opcode, regs, acc, operand1, operand2).unwrap_or_else(|| {
            track_stub_deopt(STUB_TRAMPOLINE);
            JIT_DEOPT
        })
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

                    // Use cached bytecode pointer to avoid a separate
                    // RT_BYTECODE TLS lookup.
                    let ptrs = RT_PTRS.with(|p| p.get());
                    let ba = if ptrs.is_cached() {
                        // SAFETY: cached pointer valid for thread lifetime.
                        unsafe { &*ptrs.bytecode }.get()
                    } else {
                        RT_BYTECODE.with(|b| b.get())
                    };
                    if !ba.is_null() {
                        // SAFETY: pointer is valid and points to a live
                        // BytecodeArray.
                        let ba_ref = unsafe { &*ba };

                        // Fast path: clone a previously cached template.
                        if let Some(rc) = ba_ref.clone_object_literal_template_pooled(slot) {
                            let obj = JsValue::PlainObject(rc);
                            return Some(jsvalue_to_jit_i64(obj));
                        }

                        // Second execution: promote pending → cached.
                        if let Some(rc) = ba_ref.promote_object_literal_template_pooled(slot) {
                            let obj = JsValue::PlainObject(rc);
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

            // ── Arguments objects ────────────────────────────────────────
            // Build arguments objects from the JIT register file which
            // stores parameters at flat indices 0..parameter_count.
            Opcode::CreateMappedArguments => {
                // Fast path: when the caller has determined that the
                // arguments object is unused (0-param closure), return
                // undefined immediately to avoid the expensive allocation.
                let ptrs = RT_PTRS.with(|p| p.get());
                if ptrs.get_skip_mapped_args() {
                    return Some(JIT_UNDEFINED);
                }

                let ba_ptr = RT_BYTECODE.with(|b| b.get());
                if ba_ptr.is_null() {
                    return None;
                }
                // SAFETY: pointer set by jit_runtime_setup and valid for
                // the duration of JIT execution.
                let ba = unsafe { &*ba_ptr };
                let param_count = ba.parameter_count() as usize;

                let mut args: Vec<JsValue> = Vec::with_capacity(param_count);
                for i in 0..param_count {
                    // SAFETY: register file is allocated with at least
                    // parameter_count + frame_size slots.
                    let jit_val = unsafe { *regs.add(i) };
                    args.push(jit_i64_to_jsvalue(jit_val));
                }

                let mut map = PropertyMap::new();
                for (i, v) in args.iter().enumerate() {
                    map.insert(i.to_string(), v.clone());
                }
                map.insert_with_attrs(
                    "length".to_string(),
                    JsValue::Smi(args.len() as i32),
                    PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
                );
                // callee: reference to the executing function (sloppy mode).
                // Reconstruct an Rc from the raw pointer — the original Rc
                // in the interpreter frame keeps the allocation alive.
                // SAFETY: ba_ptr came from `&*Rc<BytecodeArray>` and the Rc
                // is alive for the entire JIT execution.
                let callee = unsafe {
                    Rc::increment_strong_count(ba_ptr);
                    JsValue::Function(Rc::from_raw(ba_ptr))
                };
                map.insert("callee".to_string(), callee);
                let args_for_iter = args;
                map.insert(
                    "@@iterator".to_string(),
                    JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
                        Ok(JsValue::Iterator(NativeIterator::from_items(
                            args_for_iter.clone(),
                        )))
                    })),
                );
                let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
                Some(jsvalue_to_jit_i64(obj))
            }

            Opcode::CreateUnmappedArguments => {
                let ba_ptr = RT_BYTECODE.with(|b| b.get());
                if ba_ptr.is_null() {
                    return None;
                }
                // SAFETY: pointer set by jit_runtime_setup and valid for
                // the duration of JIT execution.
                let ba = unsafe { &*ba_ptr };
                let param_count = ba.parameter_count() as usize;

                let mut args: Vec<JsValue> = Vec::with_capacity(param_count);
                for i in 0..param_count {
                    // SAFETY: register file slots are valid.
                    let jit_val = unsafe { *regs.add(i) };
                    args.push(jit_i64_to_jsvalue(jit_val));
                }

                let mut map = PropertyMap::new();
                for (i, v) in args.iter().enumerate() {
                    map.insert(i.to_string(), v.clone());
                }
                map.insert_with_attrs(
                    "length".to_string(),
                    JsValue::Smi(args.len() as i32),
                    PropertyAttributes::WRITABLE | PropertyAttributes::CONFIGURABLE,
                );
                // callee: TypeError thrower (strict mode)
                map.insert(
                    "callee".to_string(),
                    JsValue::NativeFunction(Rc::new(|_args: Vec<JsValue>| {
                        Err(StatorError::TypeError(
                            "'caller', 'callee', and 'arguments' properties may not be accessed on strict mode functions or the arguments objects for calls to them".into(),
                        ))
                    })),
                );
                let args_for_iter = args;
                map.insert(
                    "@@iterator".to_string(),
                    JsValue::NativeFunction(Rc::new(move |_args: Vec<JsValue>| {
                        Ok(JsValue::Iterator(NativeIterator::from_items(
                            args_for_iter.clone(),
                        )))
                    })),
                );
                let obj = JsValue::PlainObject(Rc::new(RefCell::new(map)));
                Some(jsvalue_to_jit_i64(obj))
            }

            Opcode::CreateRestParameter => None,

            // ── Named property load ──────────────────────────────────────
            Opcode::LdaNamedProperty => {
                let obj_flat = operand1 as usize;
                let name_idx = operand2 as u32;
                // SAFETY: obj_flat was computed by the compiler from a valid
                // bytecode register operand and is within bounds.
                let obj_i64 = unsafe { *regs.add(obj_flat) };
                let obj = jit_i64_to_jsvalue(obj_i64);
                let prop_name = get_rt_string_constant_ref(name_idx)?;

                let result = match &obj {
                    JsValue::PlainObject(map) => map
                        .borrow()
                        .get(prop_name)
                        .cloned()
                        .unwrap_or(JsValue::Undefined),
                    JsValue::Array(arr) => {
                        if prop_name == "length" {
                            JsValue::Smi(arr.borrow().len() as i32)
                        } else {
                            JsValue::Undefined
                        }
                    }
                    JsValue::Function(ba) => {
                        let val = crate::interpreter::fn_props_get(ba, prop_name);
                        if !matches!(val, JsValue::Undefined) {
                            val
                        } else if prop_name == "prototype" && !ba.is_arrow() && !ba.is_generator() {
                            let func_val = JsValue::Function(Rc::clone(ba));
                            let mut proto_map = PropertyMap::new();
                            proto_map.insert("constructor".to_string(), func_val);
                            let proto_obj = JsValue::PlainObject(Rc::new(RefCell::new(proto_map)));
                            crate::interpreter::fn_props_set(
                                ba,
                                "prototype".to_string(),
                                proto_obj.clone(),
                            );
                            proto_obj
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
                let prop_name = get_rt_string_constant_ref(name_idx)?;

                // Fast path: borrow object directly from heap via cached
                // ptrs, avoiding the jit_i64_to_jsvalue Rc clone.
                if is_heap_handle(obj_i64) {
                    let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
                    let ptrs = RT_PTRS.with(|p| p.get());
                    if ptrs.is_cached() {
                        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
                        if let Some(JsValue::PlainObject(map_rc)) = heap.get(obj_idx) {
                            // Inline Smi decode for the value (common case
                            // for numeric literals and loop counters).
                            let value = if acc >= i32::MIN as i64 && acc <= i32::MAX as i64 {
                                JsValue::Smi(acc as i32)
                            } else {
                                jit_i64_to_jsvalue(acc)
                            };
                            // SAFETY: single-threaded JIT; no concurrent borrows.
                            let map = unsafe { &mut *map_rc.as_ptr() };
                            match map.try_template_fill(prop_name, value) {
                                Ok(_) => return Some(acc),
                                Err(value) => {
                                    map.insert(prop_name.to_string(), value);
                                    return Some(acc);
                                }
                            }
                        }
                    }
                }

                // Slow path: clone-based fallback.
                let obj = jit_i64_to_jsvalue(obj_i64);
                let value = jit_i64_to_jsvalue(acc);

                match &obj {
                    JsValue::PlainObject(map) => {
                        map.borrow_mut().insert(prop_name.to_string(), value);
                    }
                    JsValue::Function(ba) => {
                        crate::interpreter::fn_props_set(ba, prop_name.to_string(), value);
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
                    (JsValue::PlainObject(map), JsValue::String(s)) => {
                        map.borrow().get(s).cloned().unwrap_or(JsValue::Undefined)
                    }
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

                match &callee {
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
                }
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

                match &callee {
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
                }
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
        track_stub_call(STUB_LDA_NAMED);
        lda_named_property_inner(obj_i64, name_idx)
            .or_else(|| lda_named_fallback(obj_i64, name_idx))
            .unwrap_or_else(|| {
                track_stub_deopt(STUB_LDA_NAMED);
                JIT_DEOPT
            })
    }

    /// Like [`jit_runtime_lda_named_property`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell in the 4th argument, eliminating
    /// one TLS lookup per property load.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – the JIT i64 encoding of the receiver object.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    /// * `RDX` (`_feedback_slot`) – reserved for future feedback-vector use.
    /// * `RCX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns the property value as `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_lda_named_property_r15(
        obj_i64: i64,
        name_idx: u32,
        _feedback_slot: u32,
        rt_ptrs_cell: i64,
    ) -> i64 {
        track_stub_call(STUB_LDA_NAMED);
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        lda_named_property_inner_with_ptrs(obj_i64, name_idx, ptrs)
            .or_else(|| lda_named_fallback(obj_i64, name_idx))
            .unwrap_or_else(|| {
                track_stub_deopt(STUB_LDA_NAMED);
                JIT_DEOPT
            })
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
        let ptrs = RT_PTRS.with(|p| p.get());
        lda_named_property_inner_with_ptrs(obj_i64, name_idx, ptrs)
    }

    /// Inner implementation for [`jit_runtime_lda_named_property`] with
    /// pre-fetched [`RtPtrs`].
    ///
    /// Uses a two-phase approach:
    /// 1. **Fast path** – borrow the heap object without cloning it and
    ///    check the inline cache.  On IC hit with a primitive result the
    ///    value is returned immediately (zero Rc clones).
    /// 2. **Slow path** – clone-based fallback for IC misses and complex
    ///    result types that need a heap-handle allocation.
    fn lda_named_property_inner_with_ptrs(
        obj_i64: i64,
        name_idx: u32,
        ptrs: RtPtrs,
    ) -> Option<i64> {
        if !is_heap_handle(obj_i64) {
            return None;
        }

        // ── Phase 0: handle-based proto IC (skips heap dereference) ─────
        // When the receiver handle + name + global prototype epoch all
        // match, the cached value is guaranteed valid without touching
        // the heap→Rc→PropertyMap→shape chain.  This is the hottest path
        // for repeated prototype-chain property reads in tight loops.
        let proto_slot = (name_idx & 31) as usize;
        if ptrs.is_cached() {
            let pe = {
                // SAFETY: cached pointer valid for thread lifetime.
                let cache = unsafe { &*(&*ptrs.prop_ic).as_ptr() };
                cache.proto[proto_slot]
            };
            if pe.receiver_handle == obj_i64 && pe.name_idx == name_idx {
                let epoch = combined_ic_epoch();
                if pe.global_epoch == epoch {
                    return Some(pe.cached_value);
                }
            }
        }

        let idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        let lda_slot = (name_idx & 63) as usize;

        // ── Phase 1: IC hit via borrowed heap (no input clone) ──────
        // Returns Ok(i64) for a direct/primitive result, Err(JsValue)
        // for an object result that still needs a heap handle, or None
        // on IC miss.
        //
        // The property ICs (own + prototype) are checked FIRST, before
        // the array-method IC, because the property ICs are accessed
        // via the already-cached `RT_PTRS` pointer.  Deferring the
        // `RT_ARRAY_METHOD_IC` thread-local lookup saves one TLS
        // access (~5-10 ns) on every property IC hit.
        let fast: Option<Result<i64, JsValue>> = if ptrs.is_cached() {
            // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            let ic_ref = unsafe { &*ptrs.prop_ic };

            // Pre-check: peek at the heap object via raw pointer to
            // extract the shape_id WITHOUT cloning.  This allows the
            // proto IC (which only needs name_idx + shape) to short-
            // circuit before any Rc clone.
            let heap = unsafe { &*heap_ref.as_ptr() };
            let heap_val = heap.get(idx)?;
            if let JsValue::PlainObject(map_rc) = heap_val {
                // SAFETY: PropertyMap lives in Rc, separate from heap Vec.
                let map = unsafe { &*map_rc.as_ptr() };
                let shape = map.shape_id();

                let (ic_entry, proto_entry) = {
                    // SAFETY: scoped borrow; dropped before IC mutation.
                    let cache = unsafe { &*ic_ref.as_ptr() };
                    (cache.lda[lda_slot], cache.proto[(name_idx & 31) as usize])
                };

                // ── Own-property IC hit (zero clone) ────────────
                if ic_entry.name_idx == name_idx && ic_entry.shape == shape {
                    // SAFETY: shape_id unchanged since IC population ⇒
                    // offset is still valid (shape changes on every
                    // structural mutation that could invalidate an offset).
                    let val = unsafe { map.get_by_offset_unchecked(ic_entry.offset) };
                    return Some(encode_ic_cached(
                        val,
                        ic_entry.cached_ptr,
                        ic_entry.cached_handle,
                        ic_ref,
                        lda_slot,
                    ));
                }

                // ── Prototype IC hit (shape-based fallback) ─────
                // Phase 0 (handle+epoch) already ran above and missed,
                // so try the shape-based check that tolerates epoch
                // changes as long as the receiver's shape is unchanged.
                if proto_entry.name_idx == name_idx && proto_entry.shape == shape {
                    // Re-stamp the entry with the current epoch so that
                    // Phase 0 will hit on the next access.
                    let epoch = combined_ic_epoch();
                    // SAFETY: scoped borrow dropped above; single-threaded.
                    let cache_mut = unsafe { &mut *ic_ref.as_ptr() };
                    let pe = &mut cache_mut.proto[proto_slot];
                    pe.receiver_handle = obj_i64;
                    pe.global_epoch = epoch;
                    return Some(proto_entry.cached_value);
                }
            }

            // ── Deferred array method IC check ──────────────────
            // Checked after property ICs: saves one TLS access on every
            // own-property or prototype IC hit for plain objects.
            let arr_ic = RT_ARRAY_METHOD_IC.with(|c| c.get());
            if arr_ic.receiver == obj_i64 && arr_ic.name_idx == name_idx && arr_ic.method != 0 {
                return Some(arr_ic.method);
            }

            // IC miss or non-PlainObject — need a proper clone for the
            // slow path (heap ref must be dropped before any mutation).
            let obj = heap_val.clone();
            let _ = heap;

            match &obj {
                JsValue::PlainObject(map_rc) => {
                    let map = unsafe { &*map_rc.as_ptr() };
                    let shape = map.shape_id();
                    let proto_slot = (name_idx & 31) as usize;

                    // ── IC miss: full lookup without cloning ────────
                    let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;
                    if let Some(offset) = map.offset_of(prop_name) {
                        // Safe bounds-checked access.
                        if let Some(val) = map.get_by_offset(offset) {
                            let result = jsvalue_ref_to_jit_i64(val);
                            let ptr = jsvalue_rc_ptr(val);
                            let handle = if ptr != 0 { result } else { 0 };
                            // SAFETY: IC `cache` ref was dropped above; no
                            // aliased references.  Single-threaded JIT.
                            let cache_mut = unsafe { &mut *ic_ref.as_ptr() };
                            cache_mut.lda[lda_slot] = LdaIcEntry {
                                name_idx,
                                shape,
                                offset,
                                cached_handle: handle,
                                cached_ptr: ptr,
                            };
                            cache_mut.dirty = true;
                            return Some(result);
                        }
                        // Property slot exists in shape but value is None
                        // — structural inconsistency.
                        return Some(JIT_UNDEFINED);
                    }

                    // Prototype chain walk.
                    let proto = map
                        .get(INTERNAL_PROTO_PROPERTY_KEY)
                        .or_else(|| map.get("__proto__"));
                    let (result, _proto_handle) = jit_proto_chain_walk(proto, prop_name);
                    // SAFETY: IC `cache` ref was dropped; single-threaded JIT.
                    let cache_mut = unsafe { &mut *ic_ref.as_ptr() };
                    cache_mut.proto[proto_slot] = ProtoIcEntry {
                        name_idx,
                        shape,
                        receiver_handle: obj_i64,
                        global_epoch: combined_ic_epoch(),
                        cached_value: result,
                    };
                    cache_mut.dirty = true;
                    return Some(result);
                }
                JsValue::Array(arr) => {
                    let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;
                    match prop_name {
                        "length" => Some(Ok(arr.borrow().len() as i64)),
                        "push" | "pop" | "indexOf" | "includes" => {
                            let arr_val = JsValue::Array(Rc::clone(arr));
                            let len = if prop_name == "pop" { 0 } else { 1 };
                            let method = crate::interpreter::make_fast_array_method(
                                &arr_val, prop_name, len,
                            );
                            let handle = alloc_heap_handle(method);
                            RT_ARRAY_METHOD_IC.with(|c| {
                                c.set(ArrayMethodIcEntry {
                                    receiver: obj_i64,
                                    name_idx,
                                    method: handle,
                                })
                            });
                            Some(Ok(handle))
                        }
                        _ => Some(Ok(JIT_UNDEFINED)),
                    }
                }
                JsValue::Function(ba) => {
                    // Look up ad-hoc properties stored in the
                    // thread-local side table (e.g. `.prototype`).
                    let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;
                    let val = crate::interpreter::fn_props_get(ba, prop_name);
                    if matches!(val, JsValue::Undefined) {
                        if prop_name == "prototype" && !ba.is_arrow() && !ba.is_generator() {
                            // Lazy prototype creation (ES §10.2.5).
                            let func_val = JsValue::Function(Rc::clone(ba));
                            let mut proto_map = PropertyMap::new();
                            proto_map.insert("constructor".to_string(), func_val);
                            let proto_obj = JsValue::PlainObject(Rc::new(RefCell::new(proto_map)));
                            crate::interpreter::fn_props_set(
                                ba,
                                "prototype".to_string(),
                                proto_obj.clone(),
                            );
                            Some(Err(proto_obj))
                        } else {
                            Some(Ok(JIT_UNDEFINED))
                        }
                    } else {
                        Some(encode_or_clone_ref(&val).map_err(|_| val))
                    }
                }
                _ => None,
            }
        } else {
            // Deferred array method IC check for the non-cached path.
            let arr_ic = RT_ARRAY_METHOD_IC.with(|c| c.get());
            if arr_ic.receiver == obj_i64 && arr_ic.name_idx == name_idx && arr_ic.method != 0 {
                return Some(arr_ic.method);
            }

            RT_HEAP.with(|heap| {
                let heap = heap.borrow();
                let obj = heap.get(idx)?;

                match obj {
                    JsValue::PlainObject(map_rc) => {
                        let map = map_rc.borrow();
                        let shape = map.shape_id();

                        // IC hit check.
                        let ic_hit = RT_PROP_IC.with(|ic| {
                            let cache = ic.borrow();

                            let entry = &cache.lda[lda_slot];
                            if entry.name_idx == name_idx && entry.shape == shape {
                                return Some(
                                    map.get_by_offset(entry.offset)
                                        .map(encode_or_clone_ref)
                                        .unwrap_or(Ok(JIT_UNDEFINED)),
                                );
                            }

                            let proto_slot = (name_idx & 31) as usize;
                            let pe = &cache.proto[proto_slot];
                            if pe.name_idx == name_idx && pe.shape == shape {
                                return Some(Ok(pe.cached_value));
                            }

                            None
                        });
                        if ic_hit.is_some() {
                            return ic_hit;
                        }

                        // IC miss: full lookup (avoids cloning receiver).
                        let prop_name = get_rt_string_constant_ref(name_idx)?;
                        if let Some(offset) = map.offset_of(prop_name) {
                            let result = map
                                .get_by_offset(offset)
                                .map(encode_or_clone_ref)
                                .unwrap_or(Ok(JIT_UNDEFINED));
                            RT_PROP_IC.with(|ic| {
                                let mut c = ic.borrow_mut();
                                c.lda[lda_slot] = LdaIcEntry {
                                    name_idx,
                                    shape,
                                    offset,
                                    cached_handle: 0,
                                    cached_ptr: 0,
                                };
                                c.dirty = true;
                            });
                            return Some(result);
                        }

                        // Prototype chain walk.
                        let proto = map
                            .get(INTERNAL_PROTO_PROPERTY_KEY)
                            .or_else(|| map.get("__proto__"));
                        let (result, _proto_handle) = jit_proto_chain_walk(proto, prop_name);
                        let proto_slot = (name_idx & 31) as usize;
                        RT_PROP_IC.with(|ic| {
                            let mut c = ic.borrow_mut();
                            c.proto[proto_slot] = ProtoIcEntry {
                                name_idx,
                                shape,
                                receiver_handle: obj_i64,
                                global_epoch: combined_ic_epoch(),
                                cached_value: result,
                            };
                            c.dirty = true;
                        });
                        Some(Ok(result))
                    }
                    JsValue::Array(arr) => {
                        let prop_name = get_rt_string_constant_ref(name_idx)?;
                        match prop_name {
                            "length" => Some(Ok(arr.borrow().len() as i64)),
                            "push" | "pop" | "indexOf" | "includes" => {
                                let arr_val = JsValue::Array(Rc::clone(arr));
                                let len = if prop_name == "pop" { 0 } else { 1 };
                                let method = crate::interpreter::make_fast_array_method(
                                    &arr_val, prop_name, len,
                                );
                                let handle = alloc_heap_handle(method);
                                RT_ARRAY_METHOD_IC.with(|c| {
                                    c.set(ArrayMethodIcEntry {
                                        receiver: obj_i64,
                                        name_idx,
                                        method: handle,
                                    })
                                });
                                Some(Ok(handle))
                            }
                            _ => Some(Ok(JIT_UNDEFINED)),
                        }
                    }
                    JsValue::Function(ba) => {
                        let prop_name = get_rt_string_constant_ref(name_idx)?;
                        let val = crate::interpreter::fn_props_get(ba, prop_name);
                        if matches!(val, JsValue::Undefined) {
                            if prop_name == "prototype" && !ba.is_arrow() && !ba.is_generator() {
                                // Lazy prototype creation (ES §10.2.5).
                                let func_val = JsValue::Function(Rc::clone(ba));
                                let mut proto_map = PropertyMap::new();
                                proto_map.insert("constructor".to_string(), func_val);
                                let proto_obj =
                                    JsValue::PlainObject(Rc::new(RefCell::new(proto_map)));
                                crate::interpreter::fn_props_set(
                                    ba,
                                    "prototype".to_string(),
                                    proto_obj.clone(),
                                );
                                Some(Err(proto_obj))
                            } else {
                                Some(Ok(JIT_UNDEFINED))
                            }
                        } else {
                            Some(encode_or_clone_ref(&val).map_err(|_| val))
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

                let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;

                if let Some(offset) = map.offset_of(prop_name) {
                    if ptrs.is_cached() {
                        // SAFETY: cached pointer valid for thread lifetime.
                        let c = unsafe { &*ptrs.prop_ic };
                        let mut c = c.borrow_mut();
                        c.lda[lda_slot] = LdaIcEntry {
                            name_idx,
                            shape,
                            offset,
                            cached_handle: 0,
                            cached_ptr: 0,
                        };
                        c.dirty = true;
                    } else {
                        RT_PROP_IC.with(|ic| {
                            let mut c = ic.borrow_mut();
                            c.lda[lda_slot] = LdaIcEntry {
                                name_idx,
                                shape,
                                offset,
                                cached_handle: 0,
                                cached_ptr: 0,
                            };
                            c.dirty = true;
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
                        .or_else(|| map.get("__proto__"));
                    let (result, _proto_handle) = jit_proto_chain_walk(proto, prop_name);

                    let proto_slot = (name_idx & 31) as usize;
                    let new_entry = ProtoIcEntry {
                        name_idx,
                        shape,
                        receiver_handle: obj_i64,
                        global_epoch: combined_ic_epoch(),
                        cached_value: result,
                    };
                    if ptrs.is_cached() {
                        let c = unsafe { &*ptrs.prop_ic };
                        let mut c = c.borrow_mut();
                        c.proto[proto_slot] = new_entry;
                        c.dirty = true;
                    } else {
                        RT_PROP_IC.with(|ic| {
                            let mut c = ic.borrow_mut();
                            c.proto[proto_slot] = new_entry;
                            c.dirty = true;
                        });
                    }

                    Some(result)
                }
            }
            JsValue::Array(arr) => {
                let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;
                match prop_name {
                    "length" => Some(arr.borrow().len() as i64),
                    "push" | "pop" | "indexOf" | "includes" => {
                        let len = if prop_name == "pop" { 0 } else { 1 };
                        let method =
                            crate::interpreter::make_fast_array_method(&obj, prop_name, len);
                        Some(jsvalue_to_jit_i64(method))
                    }
                    _ => Some(JIT_UNDEFINED),
                }
            }
            JsValue::Function(ba) => {
                let prop_name = get_rt_string_constant_ref_with_ptrs(name_idx, &ptrs)?;
                let val = crate::interpreter::fn_props_get(ba, prop_name);
                if !matches!(val, JsValue::Undefined) {
                    Some(jsvalue_to_jit_i64(val))
                } else if prop_name == "prototype" && !ba.is_arrow() && !ba.is_generator() {
                    // Lazy prototype creation (ES §10.2.5).
                    let func_val = JsValue::Function(Rc::clone(ba));
                    let mut proto_map = PropertyMap::new();
                    proto_map.insert("constructor".to_string(), func_val);
                    let proto_obj = JsValue::PlainObject(Rc::new(RefCell::new(proto_map)));
                    crate::interpreter::fn_props_set(
                        ba,
                        "prototype".to_string(),
                        proto_obj.clone(),
                    );
                    Some(jsvalue_to_jit_i64(proto_obj))
                } else {
                    Some(JIT_UNDEFINED)
                }
            }
            _ => None,
        }
    }

    /// Fallback property lookup used when [`lda_named_property_inner_with_ptrs`]
    /// returns `None`.  Performs a simple clone-based lookup without any IC
    /// machinery, avoiding a full JIT deopt for edge cases (unsupported
    /// receiver types, stale heap handles, constant-pool misses on the
    /// cached pointer path, etc.).
    ///
    /// Returns `Some(value)` on success, `None` only when the constant-pool
    /// lookup fails (indicating a fundamental BA-pointer problem).
    fn lda_named_fallback(obj_i64: i64, name_idx: u32) -> Option<i64> {
        if !is_heap_handle(obj_i64) {
            // Non-heap receiver (primitive, boolean, null, undefined).
            // Property access on these returns undefined in the JIT.
            return Some(JIT_UNDEFINED);
        }

        let prop_name = get_rt_string_constant_ref(name_idx)?;
        let obj = jit_i64_to_jsvalue(obj_i64);

        match &obj {
            JsValue::PlainObject(map_rc) => {
                let map = map_rc.borrow();
                if let Some(val) = map.get(prop_name) {
                    let encoded = jsvalue_ref_to_jit_i64(val);
                    return Some(encoded);
                }
                // Not an own property — walk prototype chain.
                let proto = map
                    .get(INTERNAL_PROTO_PROPERTY_KEY)
                    .or_else(|| map.get("__proto__"));
                let (result, _) = jit_proto_chain_walk(proto, prop_name);
                Some(result)
            }
            JsValue::Array(arr) => match prop_name {
                "length" => Some(arr.borrow().len() as i64),
                _ => Some(JIT_UNDEFINED),
            },
            JsValue::Function(ba) => {
                let val = crate::interpreter::fn_props_get(ba, prop_name);
                if !matches!(val, JsValue::Undefined) {
                    Some(jsvalue_to_jit_i64(val))
                } else {
                    Some(JIT_UNDEFINED)
                }
            }
            JsValue::String(s) if prop_name == "length" => Some(s.len() as i64),
            _ => Some(JIT_UNDEFINED),
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

    /// Extract a raw pointer that uniquely identifies the storage behind
    /// a heap-object `JsValue`.  Returns `0` for primitive types and for
    /// `NativeFunction` (which wraps a fat `Rc<dyn Fn>` pointer).
    #[inline]
    fn jsvalue_rc_ptr(val: &JsValue) -> usize {
        match val {
            JsValue::PlainObject(rc) => Rc::as_ptr(rc) as usize,
            JsValue::Array(rc) => Rc::as_ptr(rc) as usize,
            JsValue::Function(rc) => Rc::as_ptr(rc) as usize,
            JsValue::String(s) => (**s).as_ptr() as usize,
            _ => 0,
        }
    }

    /// Encode a `&JsValue` from an IC hit with heap-handle caching.
    ///
    /// For primitives this is a plain encode (no allocation).  For heap
    /// objects the `(cached_ptr, cached_handle)` pair from the IC entry
    /// is compared with the current value's `Rc` pointer; on match the
    /// cached handle is returned without an `Rc::clone`.
    ///
    /// The IC entry's `cached_ptr` and `cached_handle` are passed **by
    /// value** to avoid holding a `&LdaIcEntry` reference that would
    /// alias the `&mut` used to update the IC slot.
    #[inline]
    fn encode_ic_cached(
        val: &JsValue,
        entry_cached_ptr: usize,
        entry_cached_handle: i64,
        ic_ref: *const RefCell<JitPropertyIcState>,
        lda_slot: usize,
    ) -> i64 {
        match val {
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
                    alloc_heap_handle(val.clone())
                }
            }
            _ => {
                let ptr = jsvalue_rc_ptr(val);
                if ptr != 0 && ptr == entry_cached_ptr {
                    return entry_cached_handle;
                }
                let handle = alloc_heap_handle(val.clone());
                // SAFETY: `entry_cached_ptr` and `entry_cached_handle` were
                // copied by value before this call, so no `&LdaIcEntry`
                // aliases `lda[lda_slot]`.  Single-threaded JIT execution
                // ensures no concurrent IC access.
                let e = unsafe { &mut (*(*ic_ref).as_ptr()).lda[lda_slot] };
                e.cached_handle = handle;
                e.cached_ptr = ptr;
                handle
            }
        }
    }

    /// Walk a prototype chain looking for `key`, returning the JIT i64
    /// encoding of the property value found (or [`JIT_UNDEFINED`]) and
    /// the `shape_id` (as `i64`) of the prototype where the property
    /// was found (0 when not found).
    ///
    /// Handles up to 64 prototype hops to guard against cycles.
    ///
    /// # Safety contract
    ///
    /// Uses `RefCell::as_ptr()` to avoid per-iteration borrow overhead.
    /// This is safe because the interpreter is single-threaded and the
    /// walk performs only read access.
    fn jit_proto_chain_walk(start: Option<&JsValue>, key: &str) -> (i64, i64) {
        let mut current: &JsValue = match start {
            Some(v) => v,
            None => return (JIT_UNDEFINED, 0),
        };
        for _ in 0..64 {
            let JsValue::PlainObject(map_rc) = current else {
                break;
            };
            // SAFETY: single-threaded interpreter; read-only access
            // during the walk.  No mutable borrows exist on this
            // RefCell.
            let map = unsafe { &*map_rc.as_ptr() };
            if let Some(val) = map.get(key) {
                return (jsvalue_ref_to_jit_i64(val), map.shape_id() as i64);
            }
            // Advance to the next prototype without cloning—the
            // reference is valid as long as the Rc keeps the
            // PropertyMap alive.
            current = match map
                .get(INTERNAL_PROTO_PROPERTY_KEY)
                .or_else(|| map.get("__proto__"))
            {
                Some(next) => next,
                None => break,
            };
        }
        (JIT_UNDEFINED, 0)
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
        call_undefined_receiver0_inner(callee_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_UNDEF0);
            JIT_DEOPT
        })
    }

    /// Return type for [`jit_runtime_fusion_resolve`].
    ///
    /// Returned in RAX:RDX per the SysV AMD64 ABI (two-member C struct).
    /// The Maglev codegen uses the resolved pointers to inline the
    /// closed-form arithmetic, avoiding a full stub call for the hot path.
    #[repr(C)]
    pub struct FusionResolved {
        /// Raw pointer to the start of `slots[slot]` in the closure
        /// context's `Vec<JsValue>` data buffer, or 0 on failure.
        pub slot_ptr: i64,
        /// The increment constant `k` from the fusion pattern.
        pub k: i64,
    }

    /// Lightweight resolve helper for inline speculative call fusion.
    ///
    /// Resolves `callee_i64` → `BytecodeArray` → closure context → slot
    /// data pointer and pattern `(slot, k)`.  Returns the raw pointer to
    /// the target `JsValue` slot entry and the increment constant `k`.
    ///
    /// The Maglev inline fast path then performs the actual load, add,
    /// overflow check, and store in machine code — no further function
    /// call needed.
    ///
    /// Returns `slot_ptr == 0` on any failure (non-heap-handle, missing
    /// context, non-Smi slot, etc.).
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_fusion_resolve(callee_i64: i64) -> FusionResolved {
        let fail = FusionResolved { slot_ptr: 0, k: 0 };

        if !is_heap_handle(callee_i64) {
            return fail;
        }
        let idx = (callee_i64 - JIT_HEAP_TAG) as usize;

        // ── Resolve BytecodeArray from heap ──────────────────────────
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return fail;
        }
        // SAFETY: cached pointer valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if idx >= heap.len() {
            return fail;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds check above.
        let ba = match unsafe { heap.get_unchecked(idx) } {
            JsValue::Function(ba) => ba,
            _ => return fail,
        };

        // ── Resolve (slot, k) pattern (with TLS fast cache) ─────────
        let cache = RT_FUSION_CACHE.with(|c| c.get());
        let (slot, k) = if cache.callee == callee_i64 && cache.callee != 0 {
            (cache.slot, cache.k)
        } else {
            let &pattern = ba
                .fusion_pattern_cache()
                .get_or_init(|| analyze_fusion_pattern(ba.bytecodes()));
            let Some((slot, k)) = pattern else {
                return fail;
            };
            RT_FUSION_CACHE.with(|c| {
                c.set(FusionFastCache {
                    callee: callee_i64,
                    slot,
                    k,
                });
            });
            (slot, k)
        };

        // ── Get closure-context slot pointer ─────────────────────────
        let Some(ctx_rc) = ba.closure_context() else {
            return fail;
        };
        // SAFETY: single-threaded JIT execution — no concurrent borrows.
        let ctx = unsafe { &*ctx_rc.as_ptr() };
        if slot >= ctx.slots.len() {
            return fail;
        }
        let slot_ptr = unsafe { ctx.slots.as_ptr().add(slot) } as i64;

        FusionResolved { slot_ptr, k }
    }

    /// Inner implementation for speculative call fusion.
    ///
    /// Inlines the heap lookup (instead of going through
    /// Analyse raw bytecodes for the context-slot increment pattern used by
    /// [`SpeculativeCallFusion`](crate::compiler::maglev::ir::ValueNode::SpeculativeCallFusion).
    ///
    /// Returns `Some((slot_index, k_value))` when the bytecodes match:
    /// ```text
    /// LdaCurrentContextSlot(S) … Add/AddSmi(K) … StaCurrentContextSlot(S) … Return
    /// ```
    /// and `None` otherwise.
    pub(crate) fn analyze_fusion_pattern(bytecodes: &[u8]) -> Option<(usize, i64)> {
        use crate::bytecode::bytecodes::{Opcode, Operand, decode};

        let instrs = decode(bytecodes).ok()?;

        let mut slot: Option<usize> = None;
        let mut k_value: Option<i64> = None;
        let mut store_slot: Option<usize> = None;
        let mut has_return = false;

        // Track: LdaSmi value that might be used by a subsequent Add
        let mut pending_lda_smi: Option<i64> = None;

        for instr in &instrs {
            match instr.opcode {
                Opcode::LdaCurrentContextSlot => {
                    if let Some(Operand::ConstantPoolIdx(s)) = instr.operand_at(0)
                        && slot.is_none()
                    {
                        slot = Some(*s as usize);
                    }
                }
                Opcode::AddSmi => {
                    if let Some(Operand::Immediate(imm)) = instr.operand_at(0) {
                        k_value = Some(*imm as i64);
                    }
                }
                Opcode::Add => {
                    // Add uses register operand; the K comes from a prior LdaSmi
                    if let Some(k) = pending_lda_smi {
                        k_value = Some(k);
                    }
                }
                Opcode::StaCurrentContextSlot => {
                    if let Some(Operand::ConstantPoolIdx(s)) = instr.operand_at(0) {
                        store_slot = Some(*s as usize);
                    }
                }
                Opcode::LdaSmi => {
                    if let Some(Operand::Immediate(imm)) = instr.operand_at(0) {
                        pending_lda_smi = Some(*imm as i64);
                    }
                }
                Opcode::Return => {
                    has_return = true;
                }
                // Skip benign prefix/suffix instructions
                Opcode::CreateMappedArguments | Opcode::Star => {}
                // Any other opcode → not a simple increment pattern
                _ => return None,
            }
        }

        let slot = slot?;
        let k = k_value?;
        let store_s = store_slot?;
        if store_s != slot || !has_return {
            return None;
        }
        Some((slot, k))
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
        // ── Maglev fast path: skip invocation count when ready ──────
        // Once Maglev code is compiled the invocation count is only
        // used for tiering decisions that have already been made.
        // Skipping the increment avoids a Cell read+write per call.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            if !ba.jit_maglev_has_deopted() {
                let maglev_cache = ba.maglev_executable_cache();
                // SAFETY: single-threaded JIT; no concurrent mutation.
                let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                if let Some(maglev_exec) = maglev_ref.as_ref() {
                    if let Some(result) = exec_maglev_callee(ba, maglev_exec, jit_args, saved_ba) {
                        return Some(result);
                    }
                    // Callee Maglev deopted — mark the flag so future
                    // invocations skip JIT.  Do NOT clear the cache here:
                    // a parent frame may still be executing the same JIT
                    // code (recursive calls).  borrow_mut() would panic
                    // if try_execute_maglev holds an immutable borrow.
                    ba.mark_jit_maglev_deopted();
                }
            }
        }

        // ── Increment invocation count ──────────────────────────────
        // When called from JIT stubs, the interpreter's dispatch_call
        // never runs, so the callee's counter would stall.  Bump it
        // here so Maglev compilation eventually triggers.
        let inv_count = ba.increment_invocation_count();

        // ── Try to trigger Maglev compilation ───────────────────────
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            use crate::bytecode::bytecode_array::MAGLEV_TIERING_THRESHOLD;
            use crate::compiler::maglev::codegen::CachedMaglevCode;

            // Kick off background Maglev compilation early.
            if inv_count >= MAGLEV_TIERING_THRESHOLD {
                crate::interpreter::maybe_compile_maglev(ba);
            }

            if !ba.jit_maglev_has_deopted() {
                let maglev_cache = ba.maglev_executable_cache();
                // SAFETY: single-threaded JIT; no concurrent mutation.
                let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                if maglev_ref.is_none() {
                    // Lazy transfer: background thread writes to the Arc<Mutex>,
                    // but JIT stubs check the Rc<RefCell>.  Bridge the gap by
                    // copying compiled code from the Arc into the Rc cache.
                    let jit_cache = ba.maglev_jit_cache_arc();
                    let cached_data = jit_cache.lock().ok().and_then(|guard| {
                        guard
                            .as_ref()
                            .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
                    });
                    if let Some((code, register_file_slots)) = cached_data {
                        // SAFETY: `code` was produced by `maglev_codegen::compile`.
                        let exec = unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                        *maglev_cache.borrow_mut() = exec;
                    }
                }
                let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                if let Some(maglev_exec) = maglev_ref.as_ref() {
                    if let Some(result) = exec_maglev_callee(ba, maglev_exec, jit_args, saved_ba) {
                        return Some(result);
                    }
                    // Callee Maglev deopted — only mark the flag.
                    // Do NOT clear the cache: a parent recursive call
                    // may hold a return address pointing into this JIT
                    // code page.  Clearing would munmap it → SIGSEGV.
                    ba.mark_jit_maglev_deopted();
                }
            }
        }

        // ── Baseline JIT fast path ──────────────────────────────────
        // On the second+ call, skip the compilation and init checks.
        // SAFETY: single-threaded JIT; no concurrent borrows of exec cache.
        let exec_cache = ba.jit_executable_cache();
        {
            let cache_ref = unsafe { &*exec_cache.as_ptr() };
            if let Some(exec) = cache_ref.as_ref() {
                if let Some(result) = exec_jit_callee(ba, exec, jit_args, saved_ba) {
                    return Some(result);
                }
                // Callee baseline JIT deopted — fall through to interpreter.
                ba.mark_jit_baseline_deopted();
            }
        }

        // ── Slow path: compile + init exec cache ────────────────────
        if ba.try_get_jit_code().is_none() && !ba.jit_baseline_has_deopted() {
            if let Ok(cc) = BaselineCompiler::compile(ba) {
                // SAFETY: `cc.code` is valid x86-64 machine code from the
                // baseline compiler.
                match unsafe {
                    CachedExecutableCode::from_compiled(&cc.code, cc.register_file_slots)
                } {
                    Ok(cached) => ba.store_jit_code(cached),
                    Err(_) => ba.mark_jit_baseline_deopted(),
                }
            } else {
                ba.mark_jit_baseline_deopted();
            }
        }

        {
            let needs_init = exec_cache.borrow().is_none();
            if needs_init {
                let jit_ref = ba.try_get_jit_code();
                if let Some(cached) = jit_ref.as_ref() {
                    // SAFETY: code was produced by the baseline compiler.
                    let exec = unsafe {
                        JitExecutableCode::new(cached.code_bytes(), cached.register_file_slots)
                    };
                    *exec_cache.borrow_mut() = exec;
                }
            }
        }

        {
            let cache_ref = exec_cache.borrow();
            if let Some(exec) = cache_ref.as_ref()
                && let Some(result) = exec_jit_callee(ba, exec, jit_args, saved_ba)
            {
                return Some(result);
            }
            // fall through to interpreter
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

            let skip_args = ba.parameter_count() == 0;

            // Always read heap length so we can truncate after the
            // callee returns — even 0-param closures may allocate
            // heap handles (context loads, intermediate values).
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
            if skip_args {
                ptrs.set_skip_mapped_args(true);
            }

            // SAFETY: cached executable code was produced by the
            // baseline compiler and contains valid x86-64 instructions.
            let jit_result = unsafe { exec.execute(jit_args, ctx_ptr) };

            bc_ref.set(saved_ba);
            if skip_args {
                ptrs.set_skip_mapped_args(false);
            }

            // Fast path: non-heap results (Smi, bool, undefined, null) don't
            // reference heap handles and survive truncation unchanged.  Skip
            // the jit_to_jsvalue_ext → jsvalue_to_jit_i64 round-trip.
            if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                // SAFETY: no active borrows; skip truncation when the
                // callee allocated no heap handles.
                // SAFETY: truncate heap handles allocated by callee.
                unsafe {
                    if (*heap_ref.as_ptr()).len() != heap_base {
                        (*heap_ref.as_ptr()).truncate(heap_base);
                    }
                }
                if let Some(ctx) = saved_ctx {
                    unsafe { *ctx_ref.as_ptr() = ctx };
                }
                return Some(jit_result);
            }

            // heap_base was read before the callee executed (always
            // valid — no deferred sentinel).

            // Heap handle or deopt: resolve the handle before truncation
            // destroys it, then re-encode after cleanup.
            let result_val = if is_jit_deopt(jit_result) {
                None
            } else {
                jit_to_jsvalue_ext(jit_result)
            };

            // SAFETY: no active borrows; skip truncation when the
            // callee allocated no heap handles.
            unsafe {
                if (*heap_ref.as_ptr()).len() != heap_base {
                    (*heap_ref.as_ptr()).truncate(heap_base);
                }
            }
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

        RT_BYTECODE.with(|b| b.set(saved_ba));

        // Fast path: non-heap results survive truncation unchanged.
        if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
            recycle_and_truncate_heap(heap_base);
            RT_CONTEXT.with(|c| *c.borrow_mut() = saved_ctx);
            return Some(jit_result);
        }

        let result_val = if is_jit_deopt(jit_result) {
            None
        } else {
            jit_to_jsvalue_ext(jit_result)
        };

        recycle_and_truncate_heap(heap_base);
        RT_CONTEXT.with(|c| *c.borrow_mut() = saved_ctx);

        result_val.map(jsvalue_to_jit_i64)
    }

    /// Execute a JS function via its cached Maglev code.
    ///
    /// Similar to [`exec_jit_callee`] but uses the Maglev JIT tier's
    /// cached code.  Maglev reads context from `RT_CONTEXT` TLS rather
    /// than receiving it as a function parameter.
    ///
    /// Uses [`MaglevCalleeCache`] to skip `closure_context()` lookup,
    /// context pointer comparison, and `skip_mapped_args` toggling when
    /// the same callee is called repeatedly (e.g. closures in a tight loop).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn exec_maglev_callee(
        ba: &Rc<BytecodeArray>,
        maglev_exec: &crate::compiler::maglev::codegen::CachedMaglevCode,
        jit_args: &[i64],
        saved_ba: *const BytecodeArray,
    ) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());

        if ptrs.is_cached() {
            // SAFETY: cached pointers point to thread-local storage that
            // outlives this call.  Single-threaded access guaranteed.
            let heap_ref = unsafe { &*ptrs.heap };
            let ctx_ref = unsafe { &*ptrs.context };
            let bc_ref = unsafe { &*ptrs.bytecode };
            let ba_ptr = &**ba as *const BytecodeArray;

            // ── Repeat-callee cache: skip context lookup on cache hit ──
            let cache = ptrs.get_maglev_callee_cache();
            let (skip_args, ctx_raw, same_context) = if cache.ba_ptr == ba_ptr {
                // Cache hit — verify the closure context hasn't been
                // replaced by a subsequent `set_closure_context` call
                // (e.g. when the same closure bytecode is re-instantiated
                // with a fresh context in a new iteration).
                let current_ctx_ptr = ba
                    .closure_context()
                    .map(Rc::as_ptr)
                    .unwrap_or(std::ptr::null()) as i64;
                if cache.ctx_ptr == current_ctx_ptr {
                    // Full hit — reuse resolved context pointer and
                    // comparison result from the previous call.
                    (cache.skip_args, cache.ctx_ptr, cache.same_context)
                } else {
                    // Context changed — drop old Rc, re-resolve.
                    drop_cached_ctx_rc_raw(cache.ctx_rc_raw);
                    let skip_args = ba.parameter_count() == 0;
                    let callee_ctx = ba.closure_context();
                    let ctx_rc_raw = callee_ctx
                        .map(|rc| Rc::into_raw(Rc::clone(rc)))
                        .unwrap_or(std::ptr::null());
                    let ctx_raw = ctx_rc_raw as i64;
                    // SAFETY: no active borrows; read current context pointer.
                    let runtime_ctx_ptr = unsafe {
                        (*ctx_ref.as_ptr())
                            .as_ref()
                            .map(Rc::as_ptr)
                            .unwrap_or(std::ptr::null())
                    };
                    let same_context = std::ptr::eq(ctx_rc_raw, runtime_ctx_ptr);
                    ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                        ba_ptr,
                        ctx_ptr: ctx_raw,
                        same_context,
                        skip_args,
                        ctx_rc_raw,
                    });
                    (skip_args, ctx_raw, same_context)
                }
            } else {
                // Cache miss — drop old Rc (if any), resolve and populate.
                drop_cached_ctx_rc_raw(cache.ctx_rc_raw);
                let skip_args = ba.parameter_count() == 0;
                let callee_ctx = ba.closure_context();
                let ctx_rc_raw = callee_ctx
                    .map(|rc| Rc::into_raw(Rc::clone(rc)))
                    .unwrap_or(std::ptr::null());
                let ctx_raw = ctx_rc_raw as i64;
                // SAFETY: no active borrows; read current context pointer.
                let current_ctx_ptr = unsafe {
                    (*ctx_ref.as_ptr())
                        .as_ref()
                        .map(Rc::as_ptr)
                        .unwrap_or(std::ptr::null())
                };
                let same_context = std::ptr::eq(ctx_rc_raw, current_ctx_ptr);
                ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                    ba_ptr,
                    ctx_ptr: ctx_raw,
                    same_context,
                    skip_args,
                    ctx_rc_raw,
                });
                (skip_args, ctx_raw, same_context)
            };

            // Always read heap length so we can truncate after the
            // callee returns — even 0-param closures may allocate
            // heap handles (context loads, intermediate values).
            // SAFETY: no active borrows; read length via raw pointer.
            let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

            let saved_ctx = if same_context {
                None
            } else {
                let callee_ctx = ba.closure_context().cloned();
                // SAFETY: no active borrows; swap context via raw pointer.
                Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
            };

            bc_ref.set(ba_ptr);
            if skip_args {
                ptrs.set_skip_mapped_args(true);
            }
            // SAFETY: Maglev code is valid x86-64.  Context set in TLS.
            let jit_result = unsafe { maglev_exec.execute(jit_args, ctx_raw) };
            bc_ref.set(saved_ba);
            if skip_args {
                ptrs.set_skip_mapped_args(false);
            }

            // ── Fast return for Smi results ───────────────────────
            if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                // Always truncate heap — even 0-param closures may
                // allocate handles that must be cleaned up.
                // SAFETY: no active borrows; truncate only when the
                // callee allocated heap handles.
                unsafe {
                    if (*heap_ref.as_ptr()).len() != heap_base {
                        (*heap_ref.as_ptr()).truncate(heap_base);
                    }
                }
                if let Some(ctx) = saved_ctx {
                    unsafe { *ctx_ref.as_ptr() = ctx };
                }
                return Some(jit_result);
            }

            // heap_base was read before the callee executed (always
            // valid — no deferred sentinel).

            let result_val = if is_jit_deopt(jit_result) {
                // Callee deopted — invalidate the repeat-callee cache
                // so the next call re-evaluates context state.
                // Re-read from TLS since the cache may have been
                // updated after the initial `cache` snapshot.
                let current_cache = ptrs.get_maglev_callee_cache();
                drop_cached_ctx_rc_raw(current_cache.ctx_rc_raw);
                ptrs.set_maglev_callee_cache(MaglevCalleeCache::EMPTY);
                None
            } else {
                jit_to_jsvalue_ext(jit_result)
            };
            // SAFETY: no active borrows; skip truncation when the
            // callee allocated no heap handles.
            unsafe {
                if (*heap_ref.as_ptr()).len() != heap_base {
                    (*heap_ref.as_ptr()).truncate(heap_base);
                }
            }
            if let Some(ctx) = saved_ctx {
                unsafe { *ctx_ref.as_ptr() = ctx };
            }
            return result_val.map(jsvalue_to_jit_i64);
        }

        // Fallback: pointers not cached yet — use .with() calls.
        let heap_base = RT_HEAP.with(|h| h.borrow().len());
        let callee_ctx = ba.closure_context();
        let ctx_raw = callee_ctx.map(Rc::as_ptr).unwrap_or(std::ptr::null()) as i64;
        let callee_ctx_owned = callee_ctx.cloned();
        let saved_ctx =
            RT_CONTEXT.with(|c| std::mem::replace(&mut *c.borrow_mut(), callee_ctx_owned));
        RT_BYTECODE.with(|b| b.set(&**ba as *const BytecodeArray));

        // SAFETY: Maglev code is valid x86-64.
        let jit_result = unsafe { maglev_exec.execute(jit_args, ctx_raw) };
        RT_BYTECODE.with(|b| b.set(saved_ba));

        if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
            recycle_and_truncate_heap(heap_base);
            RT_CONTEXT.with(|c| *c.borrow_mut() = saved_ctx);
            return Some(jit_result);
        }

        let result_val = if is_jit_deopt(jit_result) {
            None
        } else {
            jit_to_jsvalue_ext(jit_result)
        };
        recycle_and_truncate_heap(heap_base);
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
        use crate::interpreter::{Interpreter, InterpreterFrame, restore_closure_context};

        let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());

        let result = if let Some(env) = env_opt {
            let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), args, env);
            restore_closure_context(&mut frame, ba);
            Interpreter::run(&mut frame)
        } else {
            let mut frame = InterpreterFrame::new(Rc::clone(ba), args);
            restore_closure_context(&mut frame, ba);
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

    /// Fast interpreter call for cached 0-arg closures.
    ///
    /// Skips `call_js_function`'s JIT cache checks, invocation count
    /// increment, and compilation attempts — all of which have already
    /// been performed or are unavailable.  Uses the pre-cached
    /// [`RtPtrs::global`] pointer to read the global environment without
    /// an extra TLS lookup.
    fn fast_interpreter_call_0(
        ba: &Rc<BytecodeArray>,
        saved_ba: *const BytecodeArray,
        ptrs: &RtPtrs,
    ) -> Option<i64> {
        use crate::interpreter::{
            Interpreter, InterpreterFrame, restore_closure_context, try_inline_small_function,
        };

        // Ultra-fast path: try to inline tiny closure bodies (e.g.
        // `count = count + 1; return count;`) without creating a frame.
        // This avoids InterpreterFrame allocation, register acquisition,
        // CURRENT_GLOBALS TLS checks, JIT entry checks, and stacker
        // overhead — saving ~200-300ns per call for the closure_counter
        // pattern.
        let env_opt = if !ptrs.global.is_null() {
            // SAFETY: pointer set by cache_rt_ptrs; valid for thread
            // lifetime.  Single-threaded access; no concurrent borrows.
            let g = unsafe { &*(*ptrs.global).as_ptr() };
            g.env.as_ref().cloned()
        } else {
            RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned())
        };
        if let Some(ref env) = env_opt
            && let Some(val) = try_inline_small_function(ba, &[], env)
        {
            // Restore RT_BYTECODE — the inline path doesn't touch it
            // but the caller expects it to be restored.
            if !ptrs.bytecode.is_null() {
                // SAFETY: pointer set by cache_rt_ptrs; valid for thread lifetime.
                unsafe { &*ptrs.bytecode }.set(saved_ba);
            } else {
                RT_BYTECODE.with(|b| b.set(saved_ba));
            }
            return Some(jsvalue_to_jit_i64(val));
        }

        // Bump invocation count so Maglev tiering still triggers for
        // callees that graduate from the interpreter path.
        ba.increment_invocation_count();
        #[cfg(all(target_arch = "x86_64", unix))]
        crate::interpreter::maybe_compile_maglev(ba);

        let result = if let Some(env) = env_opt {
            let mut frame = InterpreterFrame::new_with_globals(Rc::clone(ba), vec![], env);
            restore_closure_context(&mut frame, ba);
            Interpreter::run(&mut frame)
        } else {
            let mut frame = InterpreterFrame::new(Rc::clone(ba), vec![]);
            restore_closure_context(&mut frame, ba);
            Interpreter::run(&mut frame)
        };

        // Restore RT_BYTECODE via cached pointer (avoids TLS lookup).
        if !ptrs.bytecode.is_null() {
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

    /// Extract the raw JIT entry point and register-file slot count from
    /// a `BytecodeArray`'s executable cache, if available.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn extract_exec_entry(ba: &BytecodeArray) -> (usize, usize) {
        let exec_cache = ba.jit_executable_cache();
        // SAFETY: single-threaded JIT; no concurrent mutation.
        let cache_ref = unsafe { &*exec_cache.as_ptr() };
        if let Some(exec) = cache_ref.as_ref() {
            let base = exec as *const JitExecutableCode as *const u8;
            // SAFETY: first field (ptr) is at offset 0.
            let entry = unsafe { *(base as *const usize) };
            let slots = exec.register_file_slots;
            (entry, slots)
        } else {
            (0, 0)
        }
    }

    /// Extract the Maglev entry point and register-file slot count from
    /// a `BytecodeArray`'s Maglev executable cache, if available and not
    /// deopted.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn extract_maglev_entry(ba: &BytecodeArray) -> (usize, usize) {
        if ba.jit_maglev_has_deopted() {
            return (0, 0);
        }
        let maglev_cache = ba.maglev_executable_cache();
        // SAFETY: single-threaded JIT; no concurrent mutation.
        let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
        if let Some(maglev_exec) = maglev_ref.as_ref() {
            (
                maglev_exec.entry_point() as usize,
                maglev_exec.register_file_slots,
            )
        } else {
            (0, 0)
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
                let heap_ref = unsafe { &*ptrs.heap };

                // ── Callee cache: skip heap lookup for repeated calls ──
                //
                // In a tight loop calling the same closure, `callee_i64`
                // is identical on every iteration.  Caching the resolved
                // `*const BytecodeArray` avoids the heap index, clone,
                // and enum discriminant check on every call.
                let (ba_ptr, cached_entry): (*const BytecodeArray, CachedCalleeEntry) = {
                    let cached = ptrs.get_cached_callee();
                    if cached.callee_i64 == callee_i64 && !cached.ba_ptr.is_null() {
                        (cached.ba_ptr, cached)
                    } else {
                        // Cache miss — look up the heap in a scoped borrow.
                        let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
                        let callee_val = {
                            // SAFETY: scoped immutable borrow; dropped before
                            // any heap mutation can occur.
                            let heap = unsafe { &*heap_ref.as_ptr() };
                            heap.get(idx).cloned()
                        };

                        match callee_val.as_ref() {
                            Some(JsValue::Function(ba)) => {
                                let ptr = Rc::as_ptr(ba);
                                let ba_ref: &BytecodeArray = unsafe { &*ptr };
                                let ctx_ptr = ba_ref
                                    .closure_context()
                                    .map(Rc::as_ptr)
                                    .unwrap_or(std::ptr::null())
                                    as i64;
                                // Clone the closure context Rc and convert
                                // to a raw pointer via `Rc::into_raw`.
                                // This holds one strong reference count so
                                // that per-call context swaps can use
                                // `Rc::from_raw` / `Rc::into_raw` without
                                // touching atomic refcounts.
                                let ctx_rc_raw = ba_ref
                                    .closure_context()
                                    .map(|rc| Rc::into_raw(Rc::clone(rc)))
                                    .unwrap_or(std::ptr::null());
                                #[cfg(all(target_arch = "x86_64", unix))]
                                let (entry_fn, reg_slots) = extract_exec_entry(ba_ref);
                                #[cfg(not(all(target_arch = "x86_64", unix)))]
                                let (entry_fn, reg_slots) = (0usize, 0usize);
                                #[cfg(all(target_arch = "x86_64", unix))]
                                let (maglev_fn, maglev_reg_slots) = extract_maglev_entry(ba_ref);
                                #[cfg(not(all(target_arch = "x86_64", unix)))]
                                let (maglev_fn, maglev_reg_slots) = (0usize, 0usize);
                                let entry = CachedCalleeEntry {
                                    callee_i64,
                                    ba_ptr: ptr,
                                    ctx_ptr,
                                    entry_fn,
                                    reg_slots,
                                    maglev_fn,
                                    maglev_reg_slots,
                                    skip_args: ba_ref.parameter_count() == 0,
                                    ctx_rc_raw,
                                };
                                // Drop old entry's Rc refcount if callee changed.
                                let old = ptrs.get_cached_callee();
                                if !old.ctx_rc_raw.is_null() && old.callee_i64 != callee_i64 {
                                    drop_cached_ctx_rc_raw(old.ctx_rc_raw);
                                }
                                ptrs.set_cached_callee(entry);
                                (ptr, entry)
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
                };

                // SAFETY: `ba_ptr` was obtained from `Rc::as_ptr` on an
                // `Rc<BytecodeArray>` stored in the RT_HEAP.  The heap
                // entry outlives this call because heap truncation only
                // removes entries above `heap_base` (set after this point).
                let ba: &BytecodeArray = unsafe { &*ba_ptr };

                // Skip CreateMappedArguments for closures with 0 formal
                // parameters — the arguments object is dead code in this
                // common case (e.g. `function() { return count++; }`).
                let skip_args = cached_entry.skip_args;

                // ── Cached Maglev ultra-fast path ──────────────
                // When the Maglev entry pointer is cached, skip the
                // RefCell lookup, lazy transfer check, and
                // CachedMaglevCode::execute overhead.
                //
                // Uses MaglevCalleeCache for the context comparison so
                // that repeated calls to the same closure skip the
                // TLS context read and pointer comparison.
                #[cfg(all(target_arch = "x86_64", unix))]
                if cached_entry.maglev_fn != 0 && !ba.jit_maglev_has_deopted() {
                    let ctx_ref = unsafe { &*ptrs.context };

                    // For 0-param closures whose body never heap-allocates
                    // (common: `count = count + 1; return count`), defer
                    // the heap-length snapshot until we actually need it.
                    // On the Smi fast path this read is skipped entirely.
                    // SAFETY: always snapshot heap length before calling the callee
                    // so we can truncate afterwards — even 0-param closures
                    // may allocate heap handles.
                    let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                    // Repeat-callee cache: reuse the context comparison
                    // result when the same callee is called again.
                    let mcache = ptrs.get_maglev_callee_cache();
                    let same_context = if mcache.ba_ptr == ba_ptr {
                        // Verify context hasn't changed.
                        let current_ctx_ptr =
                            ba.closure_context()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null()) as i64;
                        if mcache.ctx_ptr == current_ctx_ptr {
                            mcache.same_context
                        } else {
                            // Context changed — drop old Rc, re-resolve.
                            drop_cached_ctx_rc_raw(mcache.ctx_rc_raw);
                            let callee_ctx_ptr = current_ctx_ptr as *const RefCell<JsContext>;
                            let current_ctx_raw = unsafe {
                                (*ctx_ref.as_ptr())
                                    .as_ref()
                                    .map(Rc::as_ptr)
                                    .unwrap_or(std::ptr::null())
                            };
                            let same = std::ptr::eq(callee_ctx_ptr, current_ctx_raw);
                            // Hold Rc reference to keep context alive.
                            let ctx_rc_raw = if !cached_entry.ctx_rc_raw.is_null() {
                                unsafe {
                                    Rc::increment_strong_count(cached_entry.ctx_rc_raw);
                                }
                                cached_entry.ctx_rc_raw
                            } else {
                                std::ptr::null()
                            };
                            ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                                ba_ptr,
                                ctx_ptr: cached_entry.ctx_ptr,
                                same_context: same,
                                skip_args,
                                ctx_rc_raw,
                            });
                            same
                        }
                    } else {
                        // Cache miss — drop old Rc, resolve.
                        drop_cached_ctx_rc_raw(mcache.ctx_rc_raw);
                        let callee_ctx_ptr = cached_entry.ctx_ptr as *const RefCell<JsContext>;
                        let current_ctx_ptr = unsafe {
                            (*ctx_ref.as_ptr())
                                .as_ref()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let same = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);
                        // Hold Rc reference to keep context alive.
                        let ctx_rc_raw = if !cached_entry.ctx_rc_raw.is_null() {
                            unsafe {
                                Rc::increment_strong_count(cached_entry.ctx_rc_raw);
                            }
                            cached_entry.ctx_rc_raw
                        } else {
                            std::ptr::null()
                        };
                        ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                            ba_ptr,
                            ctx_ptr: cached_entry.ctx_ptr,
                            same_context: same,
                            skip_args,
                            ctx_rc_raw,
                        });
                        same
                    };
                    let saved_ctx = if same_context {
                        None
                    } else {
                        // Zero-cost context swap: create an Rc from the
                        // cached raw pointer without touching atomic
                        // refcounts.  The matching `Rc::into_raw` on
                        // restore converts it back without decrementing.
                        let callee_ctx = if !cached_entry.ctx_rc_raw.is_null() {
                            Some(unsafe { Rc::from_raw(cached_entry.ctx_rc_raw) })
                        } else {
                            None
                        };
                        Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
                    };

                    bc_ref.set(ba_ptr);
                    if skip_args {
                        ptrs.set_skip_mapped_args(true);
                    }
                    let mut reg_file = [0i64; 16];
                    // SAFETY: Maglev entry is valid x86-64 with signature
                    // `extern "C" fn(*mut i64, i64) -> i64`.
                    let f: extern "C" fn(*mut i64, i64) -> i64 =
                        unsafe { std::mem::transmute(cached_entry.maglev_fn as *const ()) };
                    let jit_result = f(reg_file.as_mut_ptr(), cached_entry.ctx_ptr);
                    bc_ref.set(saved_ba);
                    if skip_args {
                        ptrs.set_skip_mapped_args(false);
                    }

                    if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                        // Smi result — skip heap truncation entirely for
                        // 0-param closures (heap_base was never read).
                        // SAFETY: truncate heap handles allocated by callee.
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            // Zero-cost restore: swap back, convert the
                            // removed callee Rc to raw (no refcount change).
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        return Some(jit_result);
                    }

                    // heap_base was always read before callee execution.

                    if is_jit_deopt(jit_result) {
                        ba.mark_jit_maglev_deopted();
                        // Invalidate cached Maglev pointer in the direct-call
                        // cache so future calls don't try the JIT fast path.
                        // Do NOT clear maglev_cache: a parent recursive frame
                        // may hold a return address into this JIT code page.
                        let mut entry = cached_entry;
                        entry.maglev_fn = 0;
                        entry.maglev_reg_slots = 0;
                        ptrs.set_cached_callee(entry);
                        // Invalidate repeat-callee cache on deopt.
                        // Re-read from TLS since the cache may have been
                        // updated after the initial `mcache` snapshot.
                        let current_cache = ptrs.get_maglev_callee_cache();
                        drop_cached_ctx_rc_raw(current_cache.ctx_rc_raw);
                        ptrs.set_maglev_callee_cache(MaglevCalleeCache::EMPTY);
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        // Fall through to baseline JIT / interpreter.
                    } else {
                        let result_val = jit_to_jsvalue_ext(jit_result);
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        return result_val.map(jsvalue_to_jit_i64);
                    }
                }

                // ── Maglev fast path (preferred) ───────────────
                #[cfg(all(target_arch = "x86_64", unix))]
                if !ba.jit_maglev_has_deopted() {
                    use crate::compiler::maglev::codegen::CachedMaglevCode;

                    let maglev_cache = ba.maglev_executable_cache();
                    // SAFETY: single-threaded; no concurrent mutation.
                    let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                    if maglev_ref.is_none() {
                        // Lazy transfer from Arc<Mutex> (background thread)
                        // to Rc<RefCell> (JIT fast path).
                        let jit_cache = ba.maglev_jit_cache_arc();
                        let cached_data = jit_cache.lock().ok().and_then(|guard| {
                            guard
                                .as_ref()
                                .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
                        });
                        if let Some((code, register_file_slots)) = cached_data {
                            // SAFETY: `code` was produced by `maglev_codegen::compile`.
                            let exec = unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                            *maglev_cache.borrow_mut() = exec;
                        }
                    }
                    let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                    if let Some(maglev_exec) = maglev_ref.as_ref() {
                        // Upgrade the callee cache with the Maglev entry.
                        let mut entry = cached_entry;
                        entry.maglev_fn = maglev_exec.entry_point() as usize;
                        entry.maglev_reg_slots = maglev_exec.register_file_slots;
                        ptrs.set_cached_callee(entry);

                        let ctx_ref = unsafe { &*ptrs.context };
                        // SAFETY: always read heap length before callee execution.
                        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                        let callee_ctx_raw = ba.closure_context();
                        let callee_ctx_ptr =
                            callee_ctx_raw.map(Rc::as_ptr).unwrap_or(std::ptr::null());
                        let current_ctx_ptr = unsafe {
                            (*ctx_ref.as_ptr())
                                .as_ref()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);
                        let saved_ctx = if same_context {
                            None
                        } else {
                            let callee_ctx = if !cached_entry.ctx_rc_raw.is_null() {
                                Some(unsafe { Rc::from_raw(cached_entry.ctx_rc_raw) })
                            } else {
                                callee_ctx_raw.cloned()
                            };
                            Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
                        };

                        bc_ref.set(ba_ptr);
                        if skip_args {
                            ptrs.set_skip_mapped_args(true);
                        }
                        let ctx_raw = callee_ctx_ptr as i64;
                        // SAFETY: Maglev code is valid x86-64.
                        let jit_result = unsafe { maglev_exec.execute(&[], ctx_raw) };
                        bc_ref.set(saved_ba);
                        if skip_args {
                            ptrs.set_skip_mapped_args(false);
                        }

                        if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                            // SAFETY: truncate heap handles allocated by callee.
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                            return Some(jit_result);
                        }

                        // heap_base was always read before callee execution.

                        if is_jit_deopt(jit_result) {
                            // Callee Maglev deopted — mark + invalidate,
                            // then fall through to baseline/interpreter so
                            // the CALLER is not forced to deopt.
                            // Do NOT clear maglev_cache: a parent recursive
                            // frame may hold a return address into this JIT
                            // code page.  The deopt flag prevents future use.
                            ba.mark_jit_maglev_deopted();
                            // NOTE: Arc not cleared — deopt guard on lazy
                            // transfer prevents reload.
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                            // Fall through to baseline JIT / interpreter.
                        } else {
                            let result_val = jit_to_jsvalue_ext(jit_result);
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                            return result_val.map(jsvalue_to_jit_i64);
                        }
                    }
                }

                // ── Cached baseline fast path ───────────────────
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    let mut entry = cached_entry;
                    if entry.entry_fn == 0 {
                        let (ef, rs) = extract_exec_entry(ba);
                        if ef != 0 {
                            entry.entry_fn = ef;
                            entry.reg_slots = rs;
                            entry.ctx_ptr = ba
                                .closure_context()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null())
                                as i64;
                            ptrs.set_cached_callee(entry);
                        }
                    }

                    if entry.entry_fn != 0 {
                        let ctx_ref = unsafe { &*ptrs.context };
                        // SAFETY: always read heap length before callee execution.
                        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                        let callee_ctx_ptr = entry.ctx_ptr as *const RefCell<JsContext>;
                        let current_ctx_ptr = unsafe {
                            (*ctx_ref.as_ptr())
                                .as_ref()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);
                        let saved_ctx = if same_context {
                            None
                        } else {
                            let callee_ctx = if !entry.ctx_rc_raw.is_null() {
                                Some(unsafe { Rc::from_raw(entry.ctx_rc_raw) })
                            } else {
                                ba.closure_context().cloned()
                            };
                            Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
                        };

                        bc_ref.set(ba_ptr);
                        if skip_args {
                            ptrs.set_skip_mapped_args(true);
                        }
                        let mut reg_file = [0i64; 16];
                        let f: extern "C" fn(*mut i64, i64) -> i64 =
                            unsafe { std::mem::transmute(entry.entry_fn as *const ()) };
                        let jit_result = f(reg_file.as_mut_ptr(), entry.ctx_ptr);
                        bc_ref.set(saved_ba);
                        if skip_args {
                            ptrs.set_skip_mapped_args(false);
                        }

                        if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                            // SAFETY: truncate heap handles allocated by callee.
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                            return Some(jit_result);
                        }

                        // heap_base was always read before callee execution.

                        if is_jit_deopt(jit_result) {
                            ba.mark_jit_baseline_deopted();
                            // Drop ctx_rc_raw before clearing the entry.
                            drop_cached_ctx_rc_raw(entry.ctx_rc_raw);
                            ptrs.set_cached_callee(CachedCalleeEntry::EMPTY);
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                        } else {
                            let result_val = jit_to_jsvalue_ext(jit_result);
                            unsafe {
                                if (*heap_ref.as_ptr()).len() != heap_base {
                                    (*heap_ref.as_ptr()).truncate(heap_base);
                                }
                            }
                            if let Some(saved) = saved_ctx {
                                let removed =
                                    unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                                if let Some(rc) = removed {
                                    let _ = Rc::into_raw(rc);
                                }
                            }
                            return result_val.map(jsvalue_to_jit_i64);
                        }
                    }
                }

                // ── Baseline JIT fast path ─────────────────────
                let exec_cache = ba.jit_executable_cache();
                // SAFETY: single-threaded JIT; no concurrent
                // mutation of exec cache.
                let cache_ref = unsafe { &*exec_cache.as_ptr() };
                if let Some(exec) = cache_ref.as_ref() {
                    let ctx_ref = unsafe { &*ptrs.context };
                    // SAFETY: always snapshot heap length before calling the
                    // callee so we can truncate afterwards — even 0-param
                    // closures may allocate heap handles.
                    let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                    let callee_ctx_raw = ba.closure_context();
                    let callee_ctx_ptr = callee_ctx_raw.map(Rc::as_ptr).unwrap_or(std::ptr::null());
                    let current_ctx_ptr = unsafe {
                        (*ctx_ref.as_ptr())
                            .as_ref()
                            .map(Rc::as_ptr)
                            .unwrap_or(std::ptr::null())
                    };
                    let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);

                    let ctx_ptr_i64 = callee_ctx_ptr as i64;
                    let saved_ctx = if same_context {
                        None
                    } else {
                        let callee_ctx = if !cached_entry.ctx_rc_raw.is_null() {
                            Some(unsafe { Rc::from_raw(cached_entry.ctx_rc_raw) })
                        } else {
                            callee_ctx_raw.cloned()
                        };
                        // SAFETY: no active borrows of context RefCell.
                        Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
                    };

                    bc_ref.set(ba_ptr);
                    if skip_args {
                        ptrs.set_skip_mapped_args(true);
                    }
                    // SAFETY: cached executable code contains valid
                    // x86-64 instructions.  No `&Vec<JsValue>` ref
                    // is alive during this call.
                    let jit_result = unsafe { exec.execute(&[], ctx_ptr_i64) };

                    bc_ref.set(saved_ba);
                    if skip_args {
                        ptrs.set_skip_mapped_args(false);
                    }

                    // Non-heap results skip the round-trip conversion.
                    if !is_jit_deopt(jit_result) && jit_result < JIT_HEAP_TAG {
                        // SAFETY: no active heap borrows; skip truncation
                        // when the callee allocated no heap handles.
                        // SAFETY: truncate heap handles allocated by callee.
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        return Some(jit_result);
                    }

                    // heap_base was always read before callee execution.

                    if is_jit_deopt(jit_result) {
                        // Callee baseline JIT deopted — mark it and
                        // fall through to interpreter so the CALLER
                        // is not forced to deopt.
                        ba.mark_jit_baseline_deopted();
                        // SAFETY: no active heap borrows.
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        // Fall through to interpreter below.
                    } else {
                        let result_val = jit_to_jsvalue_ext(jit_result);
                        // SAFETY: no active heap borrows.
                        unsafe {
                            if (*heap_ref.as_ptr()).len() != heap_base {
                                (*heap_ref.as_ptr()).truncate(heap_base);
                            }
                        }
                        if let Some(saved) = saved_ctx {
                            let removed =
                                unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), saved) };
                            if let Some(rc) = removed {
                                let _ = Rc::into_raw(rc);
                            }
                        }
                        return result_val.map(jsvalue_to_jit_i64);
                    }
                }

                // No JIT code at all: direct interpreter call, bypassing
                // call_js_function's JIT cache checks and compilation
                // attempts which have already been tried on cache fill.
                // SAFETY: ba_ptr was obtained from Rc::as_ptr on an Rc
                // that is still alive in the heap.  Bumping the strong
                // count before from_raw produces a valid owned Rc.
                unsafe { Rc::increment_strong_count(ba_ptr) };
                // SAFETY: pointer + refcount are valid per above.
                let ba_rc = unsafe { Rc::from_raw(ba_ptr) };
                return fast_interpreter_call_0(&ba_rc, saved_ba, &ptrs);
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
        call_undefined_receiver1_inner(callee_i64, arg0_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_UNDEF1);
            JIT_DEOPT
        })
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
                // Scoped heap borrow: dropped before call_js_function
                // or native function invocation that could reallocate.
                let callee_val = {
                    // SAFETY: scoped immutable borrow.
                    let heap = unsafe { &*heap_ref.as_ptr() };
                    heap.get(idx).cloned()
                };
                match callee_val.as_ref() {
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
        call_undefined_receiver2_inner(callee_i64, arg0_i64, arg1_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_UNDEF2);
            JIT_DEOPT
        })
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
                // Scoped heap borrow: dropped before call_js_function
                // or native function invocation that could reallocate.
                let callee_val = {
                    // SAFETY: scoped immutable borrow.
                    let heap = unsafe { &*heap_ref.as_ptr() };
                    heap.get(idx).cloned()
                };
                match callee_val.as_ref() {
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

    // ── Direct JIT-to-JIT call support ─────────────────────────────────
    //
    // The following functions enable Maglev-generated code to call into
    // baseline JIT code directly, bypassing the full runtime stub overhead.
    //
    // Flow:
    //   1. Generated code calls `jit_runtime_get_jit_entry(callee_i64)`
    //      which returns a `JitEntryInfo { entry_point, ctx_ptr }`.
    //   2. If `entry_point != 0`, generated code allocates a register file
    //      on the stack, stores arguments, and calls the entry point directly.
    //   3. After the call, generated code calls
    //      `jit_runtime_finish_direct_call(result)` to restore TLS state
    //      and encode the result.
    //   4. If `entry_point == 0`, generated code falls back to the normal
    //      runtime stub (e.g. `jit_runtime_call_undefined_receiver0`).

    /// Return type for [`jit_runtime_get_jit_entry`].
    ///
    /// Returned in RAX:RDX per the SysV AMD64 ABI (two-member C struct).
    #[repr(C)]
    pub struct JitEntryInfo {
        /// JIT code entry point, or 0 if no JIT code is available.
        pub entry_point: usize,
        /// Closure context raw pointer to pass as the second argument
        /// to the baseline JIT entry.  Only valid when `entry_point != 0`.
        pub ctx_ptr: i64,
    }

    /// Look up the baseline JIT entry point for a callee and prepare TLS
    /// state for a direct call.
    ///
    /// When a valid entry point is returned, this function has already:
    ///   - Saved the current `RT_BYTECODE` pointer (for later restore).
    ///   - Set `RT_BYTECODE` to the callee's [`BytecodeArray`].
    ///   - Saved the current heap length (for post-call truncation).
    ///   - Saved and swapped the closure context if necessary.
    ///
    /// The caller **must** call [`jit_runtime_finish_direct_call`] after
    /// the direct call completes to restore TLS state.
    ///
    /// Returns `JitEntryInfo { 0, 0 }` if the callee does not have
    /// baseline JIT code cached (fall back to the full stub).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    ///
    /// Returns `JitEntryInfo` in `RAX` (entry_point) and `RDX` (ctx_ptr).
    pub extern "C" fn jit_runtime_get_jit_entry(callee_i64: i64) -> JitEntryInfo {
        let null_info = JitEntryInfo {
            entry_point: 0,
            ctx_ptr: 0,
        };

        if callee_i64 < JIT_HEAP_TAG {
            return null_info;
        }

        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return null_info;
        }

        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let heap_ref = unsafe { &*ptrs.heap };
        let bc_ref = unsafe { &*ptrs.bytecode };
        let ctx_ref = unsafe { &*ptrs.context };

        let idx = (callee_i64 - JIT_HEAP_TAG) as usize;

        // Scoped heap borrow: extract the exec-cache pointer and context
        // info, then drop the heap borrow before calling JIT code.
        let (exec_ptr, reg_file_slots, ba_ptr, callee_ctx_ptr) = {
            // SAFETY: single-threaded JIT; no concurrent heap mutation.
            let heap = unsafe { &*heap_ref.as_ptr() };
            let callee = match heap.get(idx) {
                Some(v) => v,
                None => return null_info,
            };
            match callee {
                JsValue::Function(ba) => {
                    let ctx = ba
                        .closure_context()
                        .map(|rc| Rc::as_ptr(rc) as usize)
                        .unwrap_or(0);
                    let ba_raw = &**ba as *const BytecodeArray;

                    // Try Maglev executable cache first (preferred).
                    // CachedMaglevCode has identical memory layout to
                    // JitExecutableCode (ptr, size, register_file_slots),
                    // so the pointer extraction in the caller works.
                    let maglev_cache = ba.maglev_executable_cache();
                    // SAFETY: single-threaded; no concurrent mutation.
                    let maglev_ref_init = unsafe { &*maglev_cache.as_ptr() };
                    if maglev_ref_init.is_none() {
                        // Lazy transfer from Arc<Mutex> (background
                        // compilation) to Rc<RefCell> (JIT fast path).
                        let jit_cache = ba.maglev_jit_cache_arc();
                        let cached_data = jit_cache.lock().ok().and_then(|guard| {
                            guard
                                .as_ref()
                                .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
                        });
                        if let Some((code, register_file_slots)) = cached_data {
                            use crate::compiler::maglev::codegen::CachedMaglevCode;
                            // SAFETY: `code` from `maglev_codegen::compile`.
                            let exec = unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                            *maglev_cache.borrow_mut() = exec;
                        }
                    }
                    let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                    if let Some(maglev_exec) =
                        maglev_ref.as_ref().filter(|e| e.register_file_slots <= 16)
                    {
                        // SAFETY: CachedMaglevCode layout matches
                        // JitExecutableCode (ptr, size, register_file_slots).
                        let exec_raw = maglev_exec
                            as *const crate::compiler::maglev::codegen::CachedMaglevCode
                            as *const JitExecutableCode;
                        (exec_raw, maglev_exec.register_file_slots, ba_raw, ctx)
                    } else {
                        // Fallback: try baseline JIT cache.
                        let exec_cache = ba.jit_executable_cache();
                        // SAFETY: single-threaded; exec cache is not being
                        // mutated during JIT execution.
                        let cache_ref = unsafe { &*exec_cache.as_ptr() };
                        if let Some(exec) =
                            cache_ref.as_ref().filter(|e| e.register_file_slots <= 16)
                        {
                            let exec_raw = exec as *const JitExecutableCode;
                            (exec_raw, exec.register_file_slots, ba_raw, ctx)
                        } else {
                            // ── Eager compile for inner closures ──
                            // Neither Maglev nor baseline cache has code.
                            // Compile Maglev synchronously so the MIC caches
                            // the optimised entry point from the very first
                            // call (inline Smi paths for context slots
                            // instead of per-access FFI stubs).
                            let ba_ref: &BytecodeArray = unsafe { &*ba_raw };

                            // Synchronous Maglev compilation (fast for small
                            // closure bodies like `count = count + 1`).
                            if !ba_ref.jit_maglev_has_deopted() {
                                crate::interpreter::compile_maglev_sync(ba_ref);
                            }
                            let maglev_ref2 = unsafe { &*maglev_cache.as_ptr() };
                            if let Some(maglev_exec) =
                                maglev_ref2.as_ref().filter(|e| e.register_file_slots <= 16)
                            {
                                let exec_raw = maglev_exec
                                    as *const crate::compiler::maglev::codegen::CachedMaglevCode
                                    as *const JitExecutableCode;
                                (exec_raw, maglev_exec.register_file_slots, ba_raw, ctx)
                            } else if !ba_ref.jit_baseline_has_deopted() {
                                // Maglev failed — fall back to eager baseline.
                                if let Ok(cc) = BaselineCompiler::compile(ba_ref) {
                                    if let Ok(cached) = unsafe {
                                        CachedExecutableCode::from_compiled(
                                            &cc.code,
                                            cc.register_file_slots,
                                        )
                                    } {
                                        ba_ref.store_jit_code(cached);
                                        crate::interpreter::maybe_compile_maglev(ba_ref);
                                        let jit_ref = ba_ref.try_get_jit_code();
                                        if let Some(cached) = jit_ref.as_ref() {
                                            let exec = unsafe {
                                                JitExecutableCode::new(
                                                    cached.code_bytes(),
                                                    cached.register_file_slots,
                                                )
                                            };
                                            *exec_cache.borrow_mut() = exec;
                                            let cache_ref = unsafe { &*exec_cache.as_ptr() };
                                            if let Some(exec) = cache_ref
                                                .as_ref()
                                                .filter(|e| e.register_file_slots <= 16)
                                            {
                                                let exec_raw = exec as *const JitExecutableCode;
                                                (exec_raw, exec.register_file_slots, ba_raw, ctx)
                                            } else {
                                                return null_info;
                                            }
                                        } else {
                                            return null_info;
                                        }
                                    } else {
                                        ba_ref.mark_jit_baseline_deopted();
                                        return null_info;
                                    }
                                } else {
                                    ba_ref.mark_jit_baseline_deopted();
                                    return null_info;
                                }
                            } else {
                                return null_info;
                            }
                        }
                    }
                }
                _ => return null_info,
            }
        };
        // Heap borrow is dropped.

        // Save TLS state for jit_runtime_finish_direct_call.
        let saved_ba = bc_ref.get();
        // SAFETY: no active heap borrows.
        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

        // Context swap (skip if same).
        let callee_ctx_raw_ptr = callee_ctx_ptr as *const RefCell<JsContext>;
        // SAFETY: no active borrows of context RefCell.
        let current_ctx_ptr = unsafe {
            (*ctx_ref.as_ptr())
                .as_ref()
                .map(Rc::as_ptr)
                .unwrap_or(std::ptr::null())
        };
        let same_context = std::ptr::eq(callee_ctx_raw_ptr, current_ctx_ptr);
        if !same_context {
            // Save the OLD context before swapping. We read it from
            // the raw ptr (which is still the old value).
            DIRECT_CALL_OLD_CTX.with(|c| {
                // SAFETY: single-threaded; no concurrent borrow.
                *c.borrow_mut() = unsafe { (*ctx_ref.as_ptr()).clone() };
            });

            // Re-borrow heap to clone callee's context Rc.
            let callee_ctx = {
                // SAFETY: single-threaded.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => ba.closure_context().cloned(),
                    _ => None,
                }
            };
            // SAFETY: no active borrows of context RefCell.
            unsafe { *ctx_ref.as_ptr() = callee_ctx };
        }

        // Set RT_BYTECODE to callee.
        bc_ref.set(ba_ptr);

        // Stash saved state via cached pointer (avoids TLS lookup).
        ptrs.set_direct_call(DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed: !same_context,
        });

        // Extract the raw function pointer from JitExecutableCode.
        // SAFETY: `exec_raw` points to a valid `JitExecutableCode` whose
        // first field is `ptr: *mut u8`.  We call `execute` with a
        // pre-allocated register file to obtain the same function pointer.
        // Instead, we compute the entry point by reading the struct's
        // internal pointer.  JitExecutableCode has fields:
        //   ptr: *mut u8, size: usize, register_file_slots: usize
        // The public `register_file_slots` at a known offset lets us
        // verify the layout assumption.
        let entry_point = {
            // SAFETY: exec_raw is alive (held by Rc in the heap).
            let exec = unsafe { &*exec_ptr };
            // The only way to obtain the code pointer is through the
            // struct's memory layout.  JitExecutableCode is a plain
            // struct with 3 pointer-sized fields in declaration order.
            let base = exec as *const JitExecutableCode as *const u8;
            // Validate: register_file_slots (3rd field) should be at
            // offset 2 * size_of::<usize>().
            let expected_offset = 2 * std::mem::size_of::<usize>();
            // SAFETY: reading within the struct's allocation.
            let slots_at_offset = unsafe { *(base.add(expected_offset) as *const usize) };
            debug_assert_eq!(
                slots_at_offset, reg_file_slots,
                "JitExecutableCode layout assumption violated"
            );
            // SAFETY: first field (ptr) is at offset 0.
            unsafe { *(base as *const usize) }
        };

        if entry_point == 0 {
            // Undo TLS changes.
            bc_ref.set(saved_ba);
            return null_info;
        }

        JitEntryInfo {
            entry_point,
            ctx_ptr: callee_ctx_ptr as i64,
        }
    }

    /// Restore TLS state after a direct JIT-to-JIT call.
    ///
    /// Handles heap truncation, bytecode-pointer restore, context restore,
    /// and result encoding (heap handle → re-encoded i64).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`result`) – raw `i64` result from the callee JIT code.
    ///
    /// Returns the final result as `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)]
    pub extern "C" fn jit_runtime_finish_direct_call(result: i64) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        finish_direct_call_inner(result, ptrs)
    }

    /// Shared implementation for [`jit_runtime_finish_direct_call`] and
    /// its `_r15` variant.
    fn finish_direct_call_inner(result: i64, ptrs: RtPtrs) -> i64 {
        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };
        let ctx_ref = unsafe { &*ptrs.context };

        let DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed,
        } = ptrs.get_direct_call();

        // Restore bytecode pointer.
        bc_ref.set(saved_ba);

        // Fast path: non-heap results don't reference heap handles.
        if result != JIT_DEOPT && result < JIT_HEAP_TAG {
            // SAFETY: no active heap borrows; skip truncation when
            // the callee allocated no heap handles.
            unsafe {
                if (*heap_ref.as_ptr()).len() != heap_base {
                    (*heap_ref.as_ptr()).truncate(heap_base);
                }
            }
            if ctx_changed {
                DIRECT_CALL_OLD_CTX.with(|c| {
                    let old = c.borrow_mut().take();
                    // SAFETY: single-threaded; no concurrent borrow.
                    unsafe { *ctx_ref.as_ptr() = old };
                });
            }
            return result;
        }

        // Heap handle or deopt: resolve before truncation.
        let result_val = if result == JIT_DEOPT {
            None
        } else {
            jit_to_jsvalue_ext(result)
        };

        // SAFETY: no active heap borrows; skip truncation when
        // the callee allocated no heap handles.
        unsafe {
            if (*heap_ref.as_ptr()).len() != heap_base {
                (*heap_ref.as_ptr()).truncate(heap_base);
            }
        }
        if ctx_changed {
            DIRECT_CALL_OLD_CTX.with(|c| {
                let old = c.borrow_mut().take();
                // SAFETY: single-threaded; no concurrent borrow.
                unsafe { *ctx_ref.as_ptr() = old };
            });
        }

        result_val.map(jsvalue_to_jit_i64).unwrap_or(JIT_DEOPT)
    }

    /// Like [`jit_runtime_finish_direct_call`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell, eliminating one TLS lookup.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`result`) – the callee's return value.
    /// * `RSI` (`rt_ptrs_cell`) – raw pointer to the `Cell<RtPtrs>` TLS
    ///   slot, previously obtained via [`jit_runtime_get_rt_ptrs_cell_addr`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_finish_direct_call_r15(result: i64, rt_ptrs_cell: i64) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        finish_direct_call_inner(result, ptrs)
    }

    /// Lightweight TLS setup for a monomorphic JIT-to-JIT call.
    ///
    /// Replaces [`jit_runtime_get_jit_entry`] on the fast path when the
    /// caller has already verified that the callee identity matches a
    /// cached entry.  Skips the exec-cache / maglev-cache lookup and
    /// entry-point extraction.  The caller passes the cached BA pointer
    /// and context pointer directly.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`ba_ptr`) – cached raw pointer to callee's [`BytecodeArray`].
    /// * `RDX` (`cached_ctx_ptr`) – cached raw context pointer
    ///   (from the first call via [`jit_runtime_get_jit_entry`]).
    ///
    /// Returns `1` in `RAX` on success (caller should use its cached
    /// entry point), or `0` on failure (fall back to stub).
    pub extern "C" fn jit_runtime_mono_call_prepare(
        callee_i64: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
    ) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return 0;
        }

        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };
        let ctx_ref = unsafe { &*ptrs.context };

        // Save current BA and set callee's BA.
        let saved_ba = bc_ref.get();
        bc_ref.set(ba_ptr as *const BytecodeArray);

        // Save heap base for post-call truncation.
        // SAFETY: single-threaded JIT; no concurrent heap mutation.
        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

        // Context swap: compare cached callee context against current.
        let callee_ctx_raw = cached_ctx_ptr as *const RefCell<JsContext>;
        // SAFETY: no active borrows of context RefCell.
        let current_ctx = unsafe {
            (*ctx_ref.as_ptr())
                .as_ref()
                .map(Rc::as_ptr)
                .unwrap_or(std::ptr::null())
        };
        let same_context = std::ptr::eq(callee_ctx_raw, current_ctx);

        if !same_context {
            // Save old context.
            DIRECT_CALL_OLD_CTX.with(|c| {
                // SAFETY: single-threaded; no concurrent borrow.
                *c.borrow_mut() = unsafe { (*ctx_ref.as_ptr()).clone() };
            });

            // Clone callee's context from heap.
            let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
            let callee_ctx = {
                // SAFETY: single-threaded.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => ba.closure_context().cloned(),
                    _ => {
                        // Callee disappeared — roll back BA and fail.
                        bc_ref.set(saved_ba);
                        return 0;
                    }
                }
            };
            // SAFETY: no active borrows of context RefCell.
            unsafe { *ctx_ref.as_ptr() = callee_ctx };
        }

        // Stash saved state via cached pointer.
        ptrs.set_direct_call(DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed: !same_context,
        });

        // Check whether the callee has Maglev code available.  If so,
        // return its entry point (> 1) so the codegen can upgrade the
        // mono-cache slot.  Otherwise return 1 (plain success).
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
            if let Some(ep) = try_get_maglev_entry_point(ba) {
                return ep;
            }
        }

        1 // success, no Maglev upgrade
    }

    /// Returns the current runtime context pointer as an `i64`.
    ///
    /// Used by Maglev codegen to compare against the cached callee
    /// context before deciding whether the full `mono_call_prepare` is
    /// required.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_get_current_ctx_ptr() -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return 0;
        }
        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let ctx_ref = unsafe { &*ptrs.context };
        let ctx_borrow = ctx_ref.borrow();
        match ctx_borrow.as_ref() {
            Some(rc) => Rc::as_ptr(rc) as i64,
            None => 0,
        }
    }

    /// Combined context-check and prepare for monomorphic calls.
    ///
    /// Merges `get_current_ctx_ptr` + `mono_call_prepare_same_ctx` into
    /// a single TLS access, saving ~40-60ns per closure call.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`ba_ptr`) – cached raw pointer to callee's [`BytecodeArray`].
    /// * `RDX` (`cached_ctx_ptr`) – cached callee context pointer.
    ///
    /// Returns `0` on failure, `1` on success (use cached entry), or
    /// `> 1` for a Maglev upgrade entry point.  When context differs,
    /// falls through to the full `mono_call_prepare`.
    #[allow(dead_code)]
    pub extern "C" fn jit_runtime_mono_call_prepare_check_ctx(
        callee_i64: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
    ) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return 0;
        }
        prepare_check_ctx_inner(callee_i64, ba_ptr, cached_ctx_ptr, ptrs)
    }

    /// Lightweight mono-call prepare for the **same-context** fast path.
    ///
    /// Called when the cached callee context pointer already matches the
    /// current runtime context, so no context save/restore is needed.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`ba_ptr`) – cached raw pointer to callee's [`BytecodeArray`].
    ///
    /// Returns a Maglev entry point (> 1) in `RAX` when an upgrade is
    /// available, `1` for plain success, or `0` on failure.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_mono_call_prepare_same_ctx(_callee_i64: i64, ba_ptr: i64) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return 0;
        }

        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };

        // Save current BA and set callee's BA.
        let saved_ba = bc_ref.get();
        bc_ref.set(ba_ptr as *const BytecodeArray);

        // Save heap base for post-call truncation.
        // SAFETY: single-threaded JIT; no concurrent heap mutation.
        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

        // Stash saved state for jit_runtime_finish_direct_call.
        // Context did not change (the codegen already verified the match).
        ptrs.set_direct_call(DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed: false,
        });

        // Check whether the callee has Maglev code available.
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
            if let Some(ep) = try_get_maglev_entry_point(ba) {
                return ep;
            }
        }

        1 // success, no Maglev upgrade
    }

    /// Analyze a callee's bytecode for speculative context-slot inlining.
    ///
    /// Called from Maglev-generated code on the first monomorphic cache hit
    /// to determine if the callee is a trivial context-slot-update closure.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ba_ptr`) – raw pointer to callee's [`BytecodeArray`].
    ///
    /// Returns a packed result in `RAX`:
    /// - If inlinable: `(slot << 48) | (op << 40) | (imm << 8) | 1`
    ///   where slot = context slot index (u16), op = tag (u8), imm = i32
    /// - If not inlinable: `0`
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_analyze_callee_inline(ba_ptr: i64) -> i64 {
        if ba_ptr == 0 {
            return 0;
        }
        // SAFETY: ba_ptr was obtained from the mono cache, which stores
        // a valid BytecodeArray pointer for the callee.
        let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
        if ba.bytecode_count() > 40 || ba.has_exception_handler() {
            return 0;
        }
        // Reuse the interpreter's existing pattern matcher.
        use crate::interpreter::extract_inline_call_cache_pub;
        match extract_inline_call_cache_pub(ba) {
            Some((slot, imm, op)) => {
                // Pre-compute delta so the hot path skips op dispatch:
                //   AddSmi → +imm, SubSmi → -imm, Inc → +1, Dec → -1.
                // Post-ops (4-7) use same delta as pre-ops (0-3).
                let base_op = op & 3;
                let delta: i32 = match base_op {
                    0 => imm,
                    1 => match imm.checked_neg() {
                        Some(d) => d,
                        None => return 0, // i32::MIN overflow
                    },
                    2 => 1,
                    3 => -1,
                    _ => return 0,
                };
                // Pack: flag=1, slot (16 bits), op (8 bits), delta (32 bits)
                let slot_u16 = slot as u16;
                let op_u8 = op;
                ((slot_u16 as i64) << 48)
                    | ((op_u8 as i64) << 40)
                    | (((delta as u32) as i64) << 8)
                    | 1
            }
            None => 0,
        }
    }

    /// Try to obtain the Maglev entry-point for `ba`, performing the lazy
    /// Arc→Rc transfer if necessary.
    ///
    /// Returns `Some(entry_point_as_i64)` (always > 1) when Maglev code is
    /// available, or `None` otherwise.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn try_get_maglev_entry_point(ba: &BytecodeArray) -> Option<i64> {
        use crate::compiler::maglev::codegen::CachedMaglevCode;

        let maglev_cache = ba.maglev_executable_cache();
        // SAFETY: single-threaded JIT; no concurrent mutation.
        let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
        if maglev_ref.is_none() {
            // Lazy transfer from Arc<Mutex> (background compilation)
            // to Rc<RefCell> (JIT fast path).
            let jit_cache = ba.maglev_jit_cache_arc();
            let cached_data = jit_cache.lock().ok().and_then(|guard| {
                guard
                    .as_ref()
                    .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
            });
            if let Some((code, register_file_slots)) = cached_data {
                // SAFETY: `code` was produced by `maglev_codegen::compile`.
                let exec = unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                *maglev_cache.borrow_mut() = exec;
            }
        }
        let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
        maglev_ref
            .as_ref()
            .map(|maglev_exec| maglev_exec.entry_point() as i64)
    }

    /// Read the current `RT_BYTECODE` pointer for caching after a
    /// successful [`jit_runtime_get_jit_entry`] call.
    ///
    /// Returns the raw `*const BytecodeArray` as `i64`.
    pub extern "C" fn jit_runtime_read_current_ba() -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            let bc_ref = unsafe { &*ptrs.bytecode };
            bc_ref.get() as i64
        } else {
            0
        }
    }

    /// Read the register file slot count from a [`BytecodeArray`] pointer.
    ///
    /// Returns `parameter_count + frame_size`, clamped to `[0, 16]`.
    #[allow(dead_code)]
    pub extern "C" fn jit_runtime_read_reg_slots(ba_ptr: i64) -> i64 {
        if ba_ptr == 0 {
            return 16;
        }
        // SAFETY: caller guarantees ba_ptr is valid.
        let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
        let slots = (ba.parameter_count() + ba.frame_size()) as i64;
        slots.clamp(1, 16)
    }

    /// Re-read the closure context pointer from a [`BytecodeArray`].
    ///
    /// Used by the inline mono-cache hit path to ensure the context
    /// pointer is fresh — `set_closure_context` may have replaced the
    /// inner `Rc`, making a previously cached raw pointer dangling.
    ///
    /// Returns the raw `*const RefCell<JsContext>` as `i64`, or `0` if
    /// `ba_ptr` is null or the function has no closure context.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_read_ba_ctx(ba_ptr: i64) -> i64 {
        if ba_ptr == 0 {
            return 0;
        }
        // SAFETY: caller guarantees ba_ptr points to a live
        // BytecodeArray (the callee's Rc<BytecodeArray> keeps it
        // alive while the heap handle exists).
        let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
        ba.closure_context()
            .map(|rc| Rc::as_ptr(rc) as i64)
            .unwrap_or(0)
    }

    /// Returns the address of the `RT_PTRS` TLS `Cell` so Maglev-compiled
    /// code can cache it in a callee-saved register (R15) and pass it to
    /// `_r15` variants of prepare/finish, eliminating per-call TLS lookups.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the entire lifetime of the
    /// current thread.  It must only be used on the same thread.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_get_rt_ptrs_cell_addr() -> i64 {
        RT_PTRS.with(|p| p as *const Cell<RtPtrs> as i64)
    }

    /// Byte offset of the `global` field within `Cell<RtPtrs>`.
    ///
    /// Used by Maglev codegen to load the global-state pointer from R15
    /// (`[R15 + RT_PTRS_GLOBAL_OFFSET]`) without a full TLS lookup.
    pub const RT_PTRS_GLOBAL_OFFSET: i32 = std::mem::offset_of!(RtPtrs, global) as i32;

    /// Like [`jit_runtime_mono_call_prepare_check_ctx`] but accepts a
    /// pre-cached pointer to the `RT_PTRS` TLS cell in the 4th argument,
    /// eliminating one TLS lookup per call.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`callee_i64`) – JIT i64 encoding of the callee.
    /// * `RSI` (`ba_ptr`) – cached BA pointer.
    /// * `RDX` (`cached_ctx_ptr`) – cached callee context pointer.
    /// * `RCX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns `0` on failure, `1` on success, or `> 1` for Maglev upgrade.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_mono_call_prepare_check_ctx_r15(
        callee_i64: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return 0;
        }
        prepare_check_ctx_inner(callee_i64, ba_ptr, cached_ctx_ptr, ptrs)
    }

    /// Shared implementation for the context-check + prepare fast path.
    fn prepare_check_ctx_inner(
        callee_i64: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
        ptrs: RtPtrs,
    ) -> i64 {
        // SAFETY: cached pointer set by cache_rt_ptrs; valid for thread lifetime.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };
        let ctx_ref = unsafe { &*ptrs.context };

        // Compare cached callee context against current runtime context.
        let callee_ctx_raw = cached_ctx_ptr as *const RefCell<JsContext>;
        // SAFETY: no active borrows of context RefCell.
        let current_ctx = unsafe {
            (*ctx_ref.as_ptr())
                .as_ref()
                .map(Rc::as_ptr)
                .unwrap_or(std::ptr::null())
        };

        if !std::ptr::eq(callee_ctx_raw, current_ctx) {
            return jit_runtime_mono_call_prepare(callee_i64, ba_ptr, cached_ctx_ptr);
        }

        // Context matches — same-ctx fast path (no context save/restore).
        let saved_ba = bc_ref.get();
        bc_ref.set(ba_ptr as *const BytecodeArray);

        // SAFETY: single-threaded JIT; no concurrent heap mutation.
        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

        ptrs.set_direct_call(DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed: false,
        });

        #[cfg(all(target_arch = "x86_64", unix))]
        {
            let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
            if let Some(ep) = try_get_maglev_entry_point(ba) {
                return ep;
            }
        }

        1
    }

    /// Combined prepare + call + finish for monomorphic 0-arg direct calls.
    ///
    /// Fuses the entire call sequence into a single Rust function, saving:
    ///
    /// * 2 extra Rust function call prologues/epilogues
    /// * 2 `Cell<RtPtrs>::get()` reads (shared across prepare + finish)
    /// * 1 `DirectCallState` TLS write + read (kept on Rust stack)
    /// * 1 Maglev upgrade check per call
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`entry_point`) — cached entry point of the callee
    /// * `RSI` (`ba_ptr`) — cached BA pointer for callee
    /// * `RDX` (`cached_ctx_ptr`) — cached context pointer
    /// * `RCX` (`rt_ptrs_cell`) — raw pointer to `Cell<RtPtrs>` (from R15)
    /// * `R8`  (`reg_slots`) — cached `register_file_slots`
    /// * `R9`  (`callee_i64`) — JIT i64 encoding of the callee
    ///
    /// Returns callee result in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_mono_dispatch_call_0_r15(
        entry_point: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
        rt_ptrs_cell: i64,
        reg_slots: i64,
        callee_i64: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        mono_dispatch_call_0_inner(
            entry_point,
            ba_ptr,
            cached_ctx_ptr,
            ptrs,
            callee_i64,
            reg_slots as usize,
        )
    }

    /// Inner implementation for [`jit_runtime_mono_dispatch_call_0_r15`].
    fn mono_dispatch_call_0_inner(
        entry_point: i64,
        ba_ptr: i64,
        cached_ctx_ptr: i64,
        ptrs: RtPtrs,
        callee_i64: i64,
        _reg_slots: usize,
    ) -> i64 {
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread
        // lifetime.  Single-threaded access guaranteed.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };
        let ctx_ref = unsafe { &*ptrs.context };

        // ── Prepare ─────────────────────────────────────────────
        let saved_ba = bc_ref.get();
        bc_ref.set(ba_ptr as *const BytecodeArray);

        // Use repeat-callee cache for the context comparison: when
        // the same callee is dispatched repeatedly, skip the TLS
        // context read and pointer comparison.
        let ba_raw = ba_ptr as *const BytecodeArray;
        let mcache = ptrs.get_maglev_callee_cache();
        let same_context = if mcache.ba_ptr == ba_raw && mcache.ctx_ptr == cached_ctx_ptr {
            mcache.same_context
        } else {
            drop_cached_ctx_rc_raw(mcache.ctx_rc_raw);
            let callee_ctx_raw = cached_ctx_ptr as *const RefCell<JsContext>;
            // SAFETY: no active borrows of context RefCell.
            let current_ctx = unsafe {
                (*ctx_ref.as_ptr())
                    .as_ref()
                    .map(Rc::as_ptr)
                    .unwrap_or(std::ptr::null())
            };
            let same = std::ptr::eq(callee_ctx_raw, current_ctx);
            // Hold Rc reference: the cached_ctx_ptr is alive this call
            // (backed by the callee's closure_context Rc), so
            // Rc::increment_strong_count keeps it alive across calls.
            let ctx_rc_raw = if cached_ctx_ptr != 0 {
                unsafe {
                    Rc::increment_strong_count(callee_ctx_raw);
                }
                callee_ctx_raw
            } else {
                std::ptr::null()
            };
            ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                ba_ptr: ba_raw,
                ctx_ptr: cached_ctx_ptr,
                same_context: same,
                skip_args: true, // 0-arg dispatch
                ctx_rc_raw,
            });
            same
        };

        // SAFETY: single-threaded JIT; no concurrent heap mutation.
        // Defer heap_base snapshot: for same-context 0-arg closures
        // returning Smi, the heap is never touched and truncation is
        // skipped entirely.
        let heap_base = if same_context {
            0usize // sentinel — unused when same_context + Smi result
        } else {
            unsafe { (*heap_ref.as_ptr()).len() }
        };

        let saved_ctx = if same_context {
            None
        } else {
            let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
            let callee_ctx = {
                // SAFETY: single-threaded.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => ba.closure_context().cloned(),
                    _ => {
                        bc_ref.set(saved_ba);
                        return JIT_DEOPT;
                    }
                }
            };
            // SAFETY: no active borrows; swap context via raw pointer.
            Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
        };

        // ── Call ─────────────────────────────────────────────────
        // Allocate a zero-initialized register file on the Rust stack.
        let mut reg_file = [0i64; 16];
        // SAFETY: entry_point is a valid JIT code pointer produced by
        // the baseline or Maglev compiler.
        let f: extern "C" fn(*mut i64, i64) -> i64 =
            unsafe { std::mem::transmute(entry_point as usize as *const ()) };
        let result = f(reg_file.as_mut_ptr(), cached_ctx_ptr);

        // ── Finish ──────────────────────────────────────────────
        bc_ref.set(saved_ba);

        if result != JIT_DEOPT && result < JIT_HEAP_TAG {
            // Smi result fast path: skip heap truncation when
            // same_context (heap_base was deferred) and skip context
            // restore (no swap occurred).
            if !same_context {
                // SAFETY: no active borrows; truncate when callee
                // allocated heap handles.
                unsafe {
                    if (*heap_ref.as_ptr()).len() != heap_base {
                        (*heap_ref.as_ptr()).truncate(heap_base);
                    }
                }
            }
            if let Some(ctx) = saved_ctx {
                // SAFETY: no active borrows; restore context.
                unsafe { *ctx_ref.as_ptr() = ctx };
            }
            return result;
        }

        // Heap handle or deopt: need real heap_base for truncation.
        let heap_base = if same_context {
            unsafe { (*heap_ref.as_ptr()).len() }
        } else {
            heap_base
        };

        // Heap handle or deopt: resolve before truncation.
        let result_val = if result == JIT_DEOPT {
            None
        } else {
            jit_to_jsvalue_ext(result)
        };

        // SAFETY: no active borrows; truncate/restore via raw pointer.
        unsafe {
            if (*heap_ref.as_ptr()).len() != heap_base {
                (*heap_ref.as_ptr()).truncate(heap_base);
            }
        }
        if let Some(ctx) = saved_ctx {
            unsafe { *ctx_ref.as_ptr() = ctx };
        }

        result_val.map(jsvalue_to_jit_i64).unwrap_or(JIT_DEOPT)
    }

    /// Lightweight prepare for inline monomorphic 0-arg calls.
    ///
    /// Combines the `read_ba_ctx` context refresh with the full
    /// prepare sequence (BA save/set, MaglevCalleeCache, context
    /// swap) in a single FFI call.  The caller then allocates the
    /// register file on the JIT stack and calls the entry point
    /// directly, followed by [`jit_runtime_finish_direct_call_r15`].
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ba_ptr`)       – cached BA pointer for callee
    /// * `RSI` (`rt_ptrs_cell`) – raw pointer to `Cell<RtPtrs>` (R15)
    /// * `RDX` (`callee_i64`)   – JIT i64 encoding of the callee
    ///
    /// Returns the fresh `ctx_ptr` in `RAX` on success, or
    /// [`JIT_DEOPT`] on failure (caller should fall back to stub).
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_mono_inline_prepare_r15(
        ba_ptr: i64,
        rt_ptrs_cell: i64,
        callee_i64: i64,
    ) -> i64 {
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        mono_inline_prepare_inner(ba_ptr, ptrs, callee_i64)
    }

    /// Inner implementation for [`jit_runtime_mono_inline_prepare_r15`].
    fn mono_inline_prepare_inner(ba_ptr: i64, ptrs: RtPtrs, callee_i64: i64) -> i64 {
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread
        // lifetime.  Single-threaded access guaranteed.
        let bc_ref = unsafe { &*ptrs.bytecode };
        let heap_ref = unsafe { &*ptrs.heap };
        let ctx_ref = unsafe { &*ptrs.context };

        // Read fresh context from the BytecodeArray.
        let cached_ctx_ptr = if ba_ptr != 0 {
            let ba = unsafe { &*(ba_ptr as *const BytecodeArray) };
            ba.closure_context()
                .map(|rc| Rc::as_ptr(rc) as i64)
                .unwrap_or(0)
        } else {
            0
        };

        // Save current BA and set callee's BA.
        let saved_ba = bc_ref.get();
        bc_ref.set(ba_ptr as *const BytecodeArray);

        // Use MaglevCalleeCache for repeat-callee optimization.
        let ba_raw = ba_ptr as *const BytecodeArray;
        let mcache = ptrs.get_maglev_callee_cache();
        let same_context = if mcache.ba_ptr == ba_raw && mcache.ctx_ptr == cached_ctx_ptr {
            mcache.same_context
        } else {
            drop_cached_ctx_rc_raw(mcache.ctx_rc_raw);
            let callee_ctx_raw = cached_ctx_ptr as *const RefCell<JsContext>;
            let current_ctx = unsafe {
                (*ctx_ref.as_ptr())
                    .as_ref()
                    .map(Rc::as_ptr)
                    .unwrap_or(std::ptr::null())
            };
            let same = std::ptr::eq(callee_ctx_raw, current_ctx);
            let ctx_rc_raw = if cached_ctx_ptr != 0 {
                unsafe {
                    Rc::increment_strong_count(callee_ctx_raw);
                }
                callee_ctx_raw
            } else {
                std::ptr::null()
            };
            ptrs.set_maglev_callee_cache(MaglevCalleeCache {
                ba_ptr: ba_raw,
                ctx_ptr: cached_ctx_ptr,
                same_context: same,
                skip_args: true,
                ctx_rc_raw,
            });
            same
        };

        // Heap base snapshot (deferred for same-context Smi results).
        let heap_base = if same_context {
            0usize
        } else {
            unsafe { (*heap_ref.as_ptr()).len() }
        };

        // Context swap for different-context calls.
        let ctx_changed = if same_context {
            false
        } else {
            let idx = (callee_i64 - JIT_HEAP_TAG) as usize;
            let callee_ctx = {
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(idx) {
                    Some(JsValue::Function(ba)) => ba.closure_context().cloned(),
                    _ => {
                        bc_ref.set(saved_ba);
                        return JIT_DEOPT;
                    }
                }
            };
            let old_ctx = unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) };
            DIRECT_CALL_OLD_CTX.with(|c| {
                *c.borrow_mut() = old_ctx;
            });
            true
        };

        // Store state for finish_direct_call_r15.
        ptrs.set_direct_call(DirectCallState {
            saved_ba,
            heap_base,
            ctx_changed,
        });

        cached_ctx_ptr
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
        track_stub_call(STUB_LDA_GLOBAL);
        let result = lda_global_inner(name_idx as u32).unwrap_or(JIT_DEOPT);
        if result == JIT_DEOPT {
            track_stub_deopt(STUB_LDA_GLOBAL);
            // Track global-load failures during Maglev execution for
            // diagnostics.
            crate::interpreter::maglev_track_global_deopt();
        }
        result
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
        // SAFETY: JIT execution is single-threaded.  No concurrent
        // borrows of the global state can be active during a
        // load-global stub.
        let state = unsafe { &*g.as_ptr() };
        let env_rc = state.env.as_ref()?;
        // Access the GlobalEnv via raw pointer, avoiding Rc::clone/drop
        // overhead (~10ns of atomic refcount ops on every call).
        let env = unsafe { &*env_rc.as_ptr() };
        let ic_entry = state.ic[(name_idx & 63) as usize];

        // Fast path: direct-mapped IC hit.
        if ic_entry.0 == name_idx {
            let (slot_idx, cached_gen) = (ic_entry.1, ic_entry.2);
            if env.generation() == cached_gen && slot_idx < env.slot_count() {
                let value = env.get_by_index(slot_idx);
                if *value != JsValue::TheHole {
                    return Some(jsvalue_ref_to_jit_i64(value));
                }
            }
        }

        // Slow path: HashMap lookup.
        let name = get_rt_string_constant_ref(name_idx)?;
        let value = env.get(name).unwrap_or(&JsValue::Undefined);
        let result = jsvalue_ref_to_jit_i64(value);

        // Populate IC via raw pointer — single-threaded JIT, safe.
        let slot_idx = env.slot_index_for(name);
        let cur_gen = env.generation();
        if let Some(idx) = slot_idx {
            unsafe { (*g.as_ptr()).ic[(name_idx & 63) as usize] = (name_idx, idx, cur_gen) };
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
        sta_global_inner(name_idx as u32, value_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_STA_GLOBAL);
            JIT_DEOPT
        })
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
        // Access the GlobalEnv via raw pointer, avoiding Rc::clone/drop
        // overhead (~10ns of atomic refcount ops on every call).
        let env = unsafe { &mut *env_rc.as_ptr() };
        let ic_entry = state.ic[(name_idx & 63) as usize];

        // Fast path: direct-mapped IC hit — store by index.
        if ic_entry.0 == name_idx {
            let (slot_idx, cached_gen) = (ic_entry.1, ic_entry.2);
            if env.generation() == cached_gen && slot_idx < env.slot_count() {
                let name = get_rt_string_constant_ref(name_idx)?;
                env.store_by_index_sync(slot_idx, name, value);
                return Some(value_i64);
            }
        }

        // Slow path: insert via HashMap.
        let name = get_rt_string_constant_ref(name_idx)?;
        let slot_idx = env.slot_index_for(name);
        if let Some(idx) = slot_idx {
            env.store_by_index_sync(idx, name, value);
            // Populate / update IC with the slot index we already found.
            let cur_gen = env.generation();
            // SAFETY: single-threaded JIT, no aliased borrows.
            unsafe { (*g.as_ptr()).ic[(name_idx & 63) as usize] = (name_idx, idx, cur_gen) };
        } else {
            env.insert(name.to_string(), value);
            // After insert, populate IC with the new slot index.
            let cur_gen = env.generation();
            if let Some(idx) = env.slot_index_for(name) {
                // SAFETY: same as above — no aliased references.
                unsafe { (*g.as_ptr()).ic[(name_idx & 63) as usize] = (name_idx, idx, cur_gen) };
            }
        }

        Some(value_i64)
    }

    // ── Fast R15-based global stubs ─────────────────────────────────────
    //
    // When Maglev codegen has R15 available (pointing to the cached
    // `Cell<RtPtrs>`), it can pass the global-state pointer directly,
    // skipping the TLS lookup + Cell::get copy that the normal stubs do.

    /// Fast `LdaGlobal` stub that receives the global-state pointer
    /// directly from the codegen (loaded from `[R15 + offset]`).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`name_idx`) – constant-pool index of the variable name.
    /// * `RSI` (`global_ptr`) – raw pointer to `RefCell<JitGlobalState>`.
    ///
    /// Returns the variable value as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_lda_global_fast(name_idx: i64, global_ptr: i64) -> i64 {
        track_stub_call(STUB_LDA_GLOBAL);
        let g = unsafe { &*(global_ptr as *const RefCell<JitGlobalState>) };
        let result = lda_global_from_ref(g, name_idx as u32).unwrap_or(JIT_DEOPT);
        if result == JIT_DEOPT {
            track_stub_deopt(STUB_LDA_GLOBAL);
            crate::interpreter::maglev_track_global_deopt();
        }
        result
    }

    /// Fast `StaGlobal` stub that receives the global-state pointer
    /// directly from the codegen (loaded from `[R15 + offset]`).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`name_idx`) – constant-pool index of the variable name.
    /// * `RSI` (`value_i64`) – the JIT i64 encoding of the value to store.
    /// * `RDX` (`global_ptr`) – raw pointer to `RefCell<JitGlobalState>`.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_global_fast(
        name_idx: i64,
        value_i64: i64,
        global_ptr: i64,
    ) -> i64 {
        track_stub_call(STUB_STA_GLOBAL);
        let g = unsafe { &*(global_ptr as *const RefCell<JitGlobalState>) };
        let value = jit_i64_to_jsvalue(value_i64);
        sta_global_from_ref(g, name_idx as u32, value_i64, value).unwrap_or_else(|| {
            track_stub_deopt(STUB_STA_GLOBAL);
            JIT_DEOPT
        })
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
        track_stub_call(STUB_LDA_KEYED);
        lda_keyed_property_inner(obj_i64, key_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_LDA_KEYED);
            JIT_DEOPT
        })
    }

    /// Like [`jit_runtime_lda_keyed_property`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell in the 3rd argument (R15),
    /// eliminating one TLS lookup per keyed load on the generic fallback
    /// path.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`key_i64`) – JIT i64 encoding of the key (accumulator).
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns the element value as `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_lda_keyed_property_r15(
        obj_i64: i64,
        key_i64: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        track_stub_call(STUB_LDA_KEYED);
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        lda_keyed_property_with_ptrs(obj_i64, key_i64, ptrs).unwrap_or_else(|| {
            track_stub_deopt(STUB_LDA_KEYED);
            JIT_UNDEFINED
        })
    }

    /// Combined hole-check + encode for array elements.  Returns the JIT
    /// i64 encoding directly — `TheHole` maps to `JIT_UNDEFINED`, common
    /// primitives encode inline, and heap types return `None` for
    /// fallback through `encode_or_clone_ref`.
    ///
    /// Handles the hole-check and encoding in a single match, eliminating
    /// one branch per element access on the hot path.
    #[inline(always)]
    pub(super) fn encode_array_element(v: &JsValue) -> Option<i64> {
        match v {
            JsValue::Boolean(b) => Some(if *b { JIT_TRUE } else { JIT_FALSE }),
            JsValue::Smi(n) => Some(i64::from(*n)),
            JsValue::Undefined | JsValue::TheHole => Some(JIT_UNDEFINED),
            JsValue::Null => Some(JIT_NULL),
            _ => None,
        }
    }

    /// Decode a non-heap JIT i64 value to `JsValue`.  Checks booleans
    /// first (hot for sieve-like benchmarks), then Smi range, then
    /// other constants.  Returns `None` if the encoding is unrecognised.
    #[inline(always)]
    pub(super) fn decode_non_heap_value_fast(value_i64: i64) -> Option<JsValue> {
        // Boolean constants are the hottest values in sieve-style
        // benchmarks — check them before the wider Smi range test.
        if value_i64 == JIT_TRUE {
            Some(JsValue::Boolean(true))
        } else if value_i64 == JIT_FALSE {
            Some(JsValue::Boolean(false))
        } else if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
            Some(JsValue::Smi(value_i64 as i32))
        } else if value_i64 == JIT_UNDEFINED {
            Some(JsValue::Undefined)
        } else {
            super::jit_to_jsvalue(value_i64)
        }
    }

    /// Inner implementation for [`jit_runtime_lda_keyed_property`].
    fn lda_keyed_property_inner(obj_i64: i64, key_i64: i64) -> Option<i64> {
        let ptrs = RT_PTRS.with(|p| p.get());
        lda_keyed_property_with_ptrs(obj_i64, key_i64, ptrs)
    }

    /// Like [`lda_keyed_property_inner`] but accepts pre-resolved
    /// [`RtPtrs`], eliminating a redundant TLS lookup when the caller
    /// already has cached pointers (e.g. from R15).
    #[inline(always)]
    fn lda_keyed_property_with_ptrs(obj_i64: i64, key_i64: i64, ptrs: RtPtrs) -> Option<i64> {
        // Ultra-fast path: positive Smi index (0..=i32::MAX).
        // Avoids jit_to_jsvalue conversion entirely — just use the i64 as usize.
        if is_heap_handle(obj_i64) && key_i64 >= 0 && key_i64 <= i32::MAX as i64 {
            let smi_key = key_i64 as usize;
            let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
            let fast = if ptrs.is_cached() {
                // SAFETY: cached pointers set by cache_rt_ptrs;
                // valid for thread lifetime.
                let heap_ref = unsafe { &*ptrs.heap };
                // SAFETY: single-threaded JIT; no concurrent mutable
                // borrows during a load-keyed fast path.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(obj_idx)? {
                    JsValue::Array(arr) => {
                        // SAFETY: single-threaded JIT; no concurrent borrows.
                        let data = unsafe { &*arr.as_ptr() };
                        if smi_key < data.len() {
                            // SAFETY: bounds verified above.
                            let v = unsafe { data.get_unchecked(smi_key) };
                            // Combined hole-check + encode in a single
                            // match — one fewer branch per element access.
                            if let Some(fast_val) = encode_array_element(v) {
                                Some(Ok(fast_val))
                            } else {
                                Some(encode_or_clone_ref(v))
                            }
                        } else {
                            Some(Ok(JIT_UNDEFINED))
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
                            let data = arr.borrow();
                            if smi_key < data.len() {
                                // SAFETY: bounds verified above.
                                let v = unsafe { data.get_unchecked(smi_key) };
                                if let Some(fast_val) = encode_array_element(v) {
                                    Some(Ok(fast_val))
                                } else {
                                    Some(encode_or_clone_ref(v))
                                }
                            } else {
                                Some(Ok(JIT_UNDEFINED))
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
                    .get(s)
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
        sta_keyed_property_inner(obj_i64, key_i64, value_i64).unwrap_or_else(|| {
            static STA_DEOPT_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = STA_DEOPT_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < 5 {
                eprintln!(
                    "[diag:sta_keyed] #{n} obj=0x{obj_i64:016x} key=0x{key_i64:016x} \
                     val=0x{value_i64:016x}"
                );
            }
            track_stub_deopt(STUB_STA_KEYED);
            value_i64
        })
    }

    /// Like [`jit_runtime_sta_keyed_property`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell in the 4th argument,
    /// eliminating one TLS lookup per keyed store.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`)      – JIT i64 heap handle for the receiver.
    /// * `RSI` (`key_i64`)      – non-negative Smi key.
    /// * `RDX` (`value_i64`)    – JIT i64 encoding of the value.
    /// * `RCX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns `value_i64` on success, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_sta_keyed_property_r15(
        obj_i64: i64,
        key_i64: i64,
        value_i64: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        sta_keyed_property_with_ptrs(obj_i64, key_i64, value_i64, ptrs).unwrap_or_else(|| {
            static STA_R15_DEOPT_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = STA_R15_DEOPT_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < 5 {
                eprintln!(
                    "[diag:sta_keyed_r15] #{n} obj=0x{obj_i64:016x} key=0x{key_i64:016x} \
                     val=0x{value_i64:016x}"
                );
            }
            track_stub_deopt(STUB_STA_KEYED);
            value_i64
        })
    }

    /// Inner implementation for [`jit_runtime_sta_keyed_property`].
    fn sta_keyed_property_inner(obj_i64: i64, key_i64: i64, value_i64: i64) -> Option<i64> {
        // Ultra-fast path: Array[positive Smi] with inline value decode.
        // Avoids jit_to_jsvalue for the key AND common value types.
        if is_heap_handle(obj_i64)
            && key_i64 >= 0
            && key_i64 <= i32::MAX as i64
            && !is_heap_handle(value_i64)
        {
            // Decode value: booleans first (hot for sieve-like patterns).
            let value = decode_non_heap_value_fast(value_i64)?;
            let smi_key = key_i64 as usize;
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
                        // SAFETY: single-threaded JIT; no concurrent
                        // borrows of this array during keyed store.
                        let v = unsafe { &mut *arr.as_ptr() };
                        if smi_key < v.len() {
                            // SAFETY: bounds verified above.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        } else {
                            let cur_len = v.len();
                            let new_cap = (smi_key + 1).next_power_of_two();
                            v.reserve(new_cap - cur_len);
                            v.resize(smi_key + 1, JsValue::TheHole);
                            // SAFETY: resize just made smi_key valid.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        }
                        Some(())
                    }
                    JsValue::PlainObject(map_rc) => {
                        let key_str = smi_key.to_string();
                        // SAFETY: single-threaded JIT; no concurrent borrows.
                        unsafe { &mut *map_rc.as_ptr() }.insert(key_str, value);
                        Some(())
                    }
                    _ => None,
                }
            } else {
                RT_HEAP.with(|heap| {
                    let heap = heap.borrow();
                    match heap.get(obj_idx)? {
                        JsValue::Array(arr) => {
                            let mut v = arr.borrow_mut();
                            if smi_key < v.len() {
                                // SAFETY: bounds verified above.
                                unsafe { *v.get_unchecked_mut(smi_key) = value };
                            } else {
                                let cur_len = v.len();
                                let new_cap = (smi_key + 1).next_power_of_two();
                                v.reserve(new_cap - cur_len);
                                v.resize(smi_key + 1, JsValue::TheHole);
                                // SAFETY: resize just made smi_key valid.
                                unsafe { *v.get_unchecked_mut(smi_key) = value };
                            }
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

    /// Variant of [`sta_keyed_property_inner`] that accepts pre-fetched
    /// [`RtPtrs`] to avoid a TLS lookup on the fast path.
    fn sta_keyed_property_with_ptrs(
        obj_i64: i64,
        key_i64: i64,
        value_i64: i64,
        ptrs: RtPtrs,
    ) -> Option<i64> {
        if is_heap_handle(obj_i64)
            && key_i64 >= 0
            && key_i64 <= i32::MAX as i64
            && !is_heap_handle(value_i64)
        {
            // Decode value: booleans first (hot for sieve-like patterns).
            let value = match decode_non_heap_value_fast(value_i64) {
                Some(v) => v,
                None => {
                    return None;
                }
            };
            let smi_key = key_i64 as usize;
            let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
            let fast = if ptrs.is_cached() {
                // SAFETY: cached pointers set by cache_rt_ptrs;
                // valid for thread lifetime.
                let heap_ref = unsafe { &*ptrs.heap };
                // SAFETY: single-threaded JIT; no concurrent
                // mutable borrows of RT_HEAP during keyed store.
                let heap = unsafe { &*heap_ref.as_ptr() };
                match heap.get(obj_idx)? {
                    JsValue::Array(arr) => {
                        // SAFETY: single-threaded JIT; no concurrent
                        // borrows of this array during keyed store.
                        let v = unsafe { &mut *arr.as_ptr() };
                        if smi_key < v.len() {
                            // SAFETY: bounds verified above.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        } else {
                            let cur_len = v.len();
                            let new_cap = (smi_key + 1).next_power_of_two();
                            v.reserve(new_cap - cur_len);
                            v.resize(smi_key + 1, JsValue::TheHole);
                            // SAFETY: resize just made smi_key valid.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        }
                        Some(())
                    }
                    JsValue::PlainObject(map_rc) => {
                        let key_str = smi_key.to_string();
                        // SAFETY: single-threaded JIT; no concurrent borrows.
                        unsafe { &mut *map_rc.as_ptr() }.insert(key_str, value);
                        Some(())
                    }
                    _ => None,
                }
            } else {
                RT_HEAP.with(|heap| {
                    let heap = heap.borrow();
                    match heap.get(obj_idx)? {
                        JsValue::Array(arr) => {
                            let mut v = arr.borrow_mut();
                            if smi_key < v.len() {
                                // SAFETY: bounds verified above.
                                unsafe { *v.get_unchecked_mut(smi_key) = value };
                            } else {
                                let cur_len = v.len();
                                let new_cap = (smi_key + 1).next_power_of_two();
                                v.reserve(new_cap - cur_len);
                                v.resize(smi_key + 1, JsValue::TheHole);
                                // SAFETY: resize just made smi_key valid.
                                unsafe { *v.get_unchecked_mut(smi_key) = value };
                            }
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
        } else {
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
            _ => {
                return None;
            }
        }
        Some(value_i64)
    }

    /// Keyed store with IC update: performs the store AND writes the
    /// array's current `(handle, data_ptr, len)` back to the caller's IC
    /// slots.  This avoids the "invalidate + miss next time" pattern that
    /// hurts growing arrays (e.g. the sieve fill loop).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`)   – JIT i64 heap handle for the receiver.
    /// * `RSI` (`key_i64`)   – non-negative Smi key.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    /// * `RCX` (`ic_slots`)  – pointer to 3 consecutive `i64` IC slots
    ///   `[handle, data_ptr, len]`.
    ///
    /// Returns `value_i64` on success, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_sta_keyed_property_with_ic(
        obj_i64: i64,
        key_i64: i64,
        value_i64: i64,
        ic_slots: *mut i64,
    ) -> i64 {
        let result = sta_keyed_property_inner(obj_i64, key_i64, value_i64);
        if result.is_some() && is_heap_handle(obj_i64) {
            // After a successful store (which may have grown the Vec),
            // update the caller's IC slots so the next inline fast-path
            // check can succeed.
            let ptrs = RT_PTRS.with(|p| p.get());
            if ptrs.is_cached() {
                let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
                // SAFETY: cached pointers valid for thread lifetime.
                let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
                if let Some(JsValue::Array(arr)) = heap.get(obj_idx) {
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let data = unsafe { &*arr.as_ptr() };
                    // SAFETY: ic_slots points to 4 writable i64s on the
                    // caller's stack frame (Maglev-generated code).
                    unsafe {
                        *ic_slots = obj_i64;
                        *ic_slots.add(1) = data.as_ptr() as i64;
                        *ic_slots.add(2) = data.len() as i64;
                        *ic_slots.add(3) = arr.as_ptr() as *const Vec<JsValue> as i64;
                    }
                }
            }
        }
        result.unwrap_or_else(|| {
            track_stub_deopt(STUB_STA_KEYED);
            value_i64
        })
    }

    /// R15 variant of [`jit_runtime_sta_keyed_property_with_ic`] that
    /// accepts a pre-cached `Cell<RtPtrs>` pointer, eliminating two
    /// TLS lookups (one for the store, one for the IC update).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`)      – JIT i64 heap handle for the receiver.
    /// * `RSI` (`key_i64`)      – non-negative Smi key.
    /// * `RDX` (`value_i64`)    – JIT i64 encoding of the value.
    /// * `RCX` (`ic_slots`)     – pointer to 4 consecutive `i64` IC slots.
    /// * `R8`  (`rt_ptrs_cell`) – pointer to the TLS `Cell<RtPtrs>`.
    ///
    /// Returns `value_i64` on success, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI entry point; caller guarantees valid pointer.
    pub extern "C" fn jit_runtime_sta_keyed_property_with_ic_r15(
        obj_i64: i64,
        key_i64: i64,
        value_i64: i64,
        ic_slots: *mut i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        let result = sta_keyed_property_with_ptrs(obj_i64, key_i64, value_i64, ptrs);
        if result.is_some() && is_heap_handle(obj_i64) && ptrs.is_cached() {
            let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
            // SAFETY: cached pointers valid for thread lifetime.
            let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
            if let Some(JsValue::Array(arr)) = heap.get(obj_idx) {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let data = unsafe { &*arr.as_ptr() };
                // SAFETY: ic_slots points to 4 writable i64s on the
                // caller's stack frame (Maglev-generated code).
                unsafe {
                    *ic_slots = obj_i64;
                    *ic_slots.add(1) = data.as_ptr() as i64;
                    *ic_slots.add(2) = data.len() as i64;
                    *ic_slots.add(3) = arr.as_ptr() as *const Vec<JsValue> as i64;
                }
            }
        }
        result.unwrap_or_else(|| {
            track_stub_deopt(STUB_STA_KEYED);
            value_i64
        })
    }

    // ── Specialized fast-path array element stubs ────────────────────────

    /// Fast-path array element load: `array[integer_index]`.
    ///
    /// Skips the generic keyed-property dispatch and only handles the
    /// `JsValue::Array` case with a non-negative integer index.  Returns
    /// [`JIT_DEOPT`] if the receiver is not an array or any other
    /// precondition fails, letting the caller fall back to the generic
    /// stub or deoptimize.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_handle`) – JIT i64 heap handle for the receiver.
    /// * `RSI` (`index`) – non-negative Smi-encoded integer index.
    ///
    /// Returns the element as a JIT `i64` in `RAX`, or [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_fast_array_load(obj_handle: i64, index: i64) -> i64 {
        fast_array_load_inner(obj_handle, index).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_ARRAY_LOAD);
            JIT_DEOPT
        })
    }

    /// Like [`jit_runtime_fast_array_load`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell, eliminating one TLS lookup.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_handle`) – JIT i64 heap handle for the receiver.
    /// * `RSI` (`index`) – non-negative Smi-encoded integer index.
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns the element as a JIT `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_fast_array_load_r15(
        obj_handle: i64,
        index: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        fast_array_load_with_ptrs(obj_handle, index, ptrs).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_ARRAY_LOAD);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_fast_array_load`].
    fn fast_array_load_inner(obj_handle: i64, index: i64) -> Option<i64> {
        if !is_heap_handle(obj_handle) || index < 0 || index > i32::MAX as i64 {
            return None;
        }
        let idx = index as usize;
        let obj_idx = (obj_handle - JIT_HEAP_TAG) as usize;
        let ptrs = RT_PTRS.with(|p| p.get());
        fast_array_load_core(idx, obj_idx, ptrs)
    }

    /// Like [`fast_array_load_inner`] but accepts pre-resolved pointers.
    #[inline(always)]
    fn fast_array_load_with_ptrs(obj_handle: i64, index: i64, ptrs: RtPtrs) -> Option<i64> {
        if !is_heap_handle(obj_handle) || index < 0 || index > i32::MAX as i64 {
            return None;
        }
        fast_array_load_core(index as usize, (obj_handle - JIT_HEAP_TAG) as usize, ptrs)
    }

    /// Shared core for fast array load with already-resolved indices.
    #[inline(always)]
    fn fast_array_load_core(idx: usize, obj_idx: usize, ptrs: RtPtrs) -> Option<i64> {
        if ptrs.is_cached() {
            // SAFETY: cached pointers set by cache_rt_ptrs;
            // valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            // SAFETY: single-threaded JIT; no concurrent mutable
            // borrows during a load fast path.
            let heap = unsafe { &*heap_ref.as_ptr() };
            match heap.get(obj_idx)? {
                JsValue::Array(arr) => {
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let data = unsafe { &*arr.as_ptr() };
                    if idx < data.len() {
                        // SAFETY: bounds verified above.
                        let v = unsafe { data.get_unchecked(idx) };
                        if let Some(fast_val) = encode_array_element(v) {
                            Some(fast_val)
                        } else {
                            Some(match encode_or_clone_ref(v) {
                                Ok(val) => val,
                                Err(obj_val) => alloc_heap_handle(obj_val),
                            })
                        }
                    } else {
                        Some(JIT_UNDEFINED)
                    }
                }
                _ => None,
            }
        } else {
            RT_HEAP.with(|heap| {
                let heap = heap.borrow();
                match heap.get(obj_idx)? {
                    JsValue::Array(arr) => {
                        let data = arr.borrow();
                        if idx < data.len() {
                            // SAFETY: bounds verified above.
                            let v = unsafe { data.get_unchecked(idx) };
                            if let Some(fast_val) = encode_array_element(v) {
                                Some(fast_val)
                            } else {
                                Some(match encode_or_clone_ref(v) {
                                    Ok(val) => val,
                                    Err(obj_val) => alloc_heap_handle(obj_val),
                                })
                            }
                        } else {
                            Some(JIT_UNDEFINED)
                        }
                    }
                    _ => None,
                }
            })
        }
    }

    /// Fast-path array element store: `array[integer_index] = value`.
    ///
    /// Handles only `JsValue::Array` receivers with a non-negative integer
    /// index.  Grows the backing `Vec` when `index >= len` (needed for
    /// patterns like `sieve[i] = true`).  Returns [`JIT_DEOPT`] on any
    /// type mismatch.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_handle`) – JIT i64 heap handle for the receiver.
    /// * `RSI` (`index`) – non-negative Smi-encoded integer index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_fast_array_store(
        obj_handle: i64,
        index: i64,
        value_i64: i64,
    ) -> i64 {
        fast_array_store_inner(obj_handle, index, value_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_ARRAY_STORE);
            JIT_DEOPT
        })
    }

    /// Like [`jit_runtime_fast_array_store`] but accepts a pre-cached
    /// pointer to the `RT_PTRS` TLS cell, eliminating one TLS lookup.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_handle`) – JIT i64 heap handle for the receiver.
    /// * `RSI` (`index`) – non-negative Smi-encoded integer index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    /// * `RCX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_fast_array_store_r15(
        obj_handle: i64,
        index: i64,
        value_i64: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        fast_array_store_with_ptrs(obj_handle, index, value_i64, ptrs).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_ARRAY_STORE);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_fast_array_store`].
    fn fast_array_store_inner(obj_handle: i64, index: i64, value_i64: i64) -> Option<i64> {
        if !is_heap_handle(obj_handle) || index < 0 || index > i32::MAX as i64 {
            return None;
        }
        let smi_key = index as usize;
        let obj_idx = (obj_handle - JIT_HEAP_TAG) as usize;

        // Decode value: booleans first (hot for sieve-like patterns),
        // then Smi range, with heap handle fallback.
        let value = if !is_heap_handle(value_i64) {
            decode_non_heap_value_fast(value_i64)?
        } else {
            jit_i64_to_jsvalue(value_i64)
        };

        let ptrs = RT_PTRS.with(|p| p.get());
        fast_array_store_core(smi_key, obj_idx, value, value_i64, ptrs)
    }

    /// Like [`fast_array_store_inner`] but accepts pre-resolved pointers.
    #[inline(always)]
    fn fast_array_store_with_ptrs(
        obj_handle: i64,
        index: i64,
        value_i64: i64,
        ptrs: RtPtrs,
    ) -> Option<i64> {
        if !is_heap_handle(obj_handle) || index < 0 || index > i32::MAX as i64 {
            return None;
        }
        let smi_key = index as usize;
        let obj_idx = (obj_handle - JIT_HEAP_TAG) as usize;
        let value = if !is_heap_handle(value_i64) {
            decode_non_heap_value_fast(value_i64)?
        } else {
            jit_i64_to_jsvalue(value_i64)
        };
        fast_array_store_core(smi_key, obj_idx, value, value_i64, ptrs)
    }

    /// Shared core for fast array store with already-resolved values.
    #[inline(always)]
    fn fast_array_store_core(
        smi_key: usize,
        obj_idx: usize,
        value: JsValue,
        value_i64: i64,
        ptrs: RtPtrs,
    ) -> Option<i64> {
        if ptrs.is_cached() {
            // SAFETY: cached pointers set by cache_rt_ptrs;
            // valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            // SAFETY: single-threaded JIT; no concurrent mutable
            // borrows during a store fast path.
            let heap = unsafe { &*heap_ref.as_ptr() };
            match heap.get(obj_idx)? {
                JsValue::Array(arr) => {
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let v = unsafe { &mut *arr.as_ptr() };
                    if smi_key < v.len() {
                        // SAFETY: bounds verified above.
                        unsafe { *v.get_unchecked_mut(smi_key) = value };
                    } else {
                        let cur_len = v.len();
                        let new_cap = (smi_key + 1).next_power_of_two();
                        v.reserve(new_cap - cur_len);
                        v.resize(smi_key + 1, JsValue::TheHole);
                        // SAFETY: resize just made smi_key valid.
                        unsafe { *v.get_unchecked_mut(smi_key) = value };
                    }
                    Some(value_i64)
                }
                _ => None,
            }
        } else {
            RT_HEAP.with(|heap| {
                let heap = heap.borrow();
                match heap.get(obj_idx)? {
                    JsValue::Array(arr) => {
                        let mut v = arr.borrow_mut();
                        if smi_key < v.len() {
                            // SAFETY: bounds verified above.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        } else {
                            let cur_len = v.len();
                            let new_cap = (smi_key + 1).next_power_of_two();
                            v.reserve(new_cap - cur_len);
                            v.resize(smi_key + 1, JsValue::TheHole);
                            // SAFETY: resize just made smi_key valid.
                            unsafe { *v.get_unchecked_mut(smi_key) = value };
                        }
                        Some(value_i64)
                    }
                    _ => None,
                }
            })
        }
    }

    /// Fast-path `Array.prototype.push(value)` stub.
    ///
    /// Checks that the receiver is a `JsValue::Array`, appends `value` to
    /// its backing `Vec`, and returns the new length as a Smi.  Returns
    /// [`JIT_DEOPT`] if the receiver is not an array.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_handle`) – JIT i64 heap handle for the array.
    /// * `RSI` (`method_handle`) – ignored (present for call-site compat).
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to push.
    ///
    /// Returns the new array length as a Smi `i64`, or [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_fast_array_push(
        obj_handle: i64,
        _method_handle: i64,
        value_i64: i64,
    ) -> i64 {
        fast_array_push_inner(obj_handle, value_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_ARRAY_PUSH);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_fast_array_push`].
    fn fast_array_push_inner(obj_handle: i64, value_i64: i64) -> Option<i64> {
        if !is_heap_handle(obj_handle) {
            return None;
        }
        let obj_idx = (obj_handle - JIT_HEAP_TAG) as usize;

        // Decode value inline for common types.
        let value = if !is_heap_handle(value_i64) {
            if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
                JsValue::Smi(value_i64 as i32)
            } else if value_i64 == JIT_TRUE {
                JsValue::Boolean(true)
            } else if value_i64 == JIT_FALSE {
                JsValue::Boolean(false)
            } else if value_i64 == JIT_UNDEFINED {
                JsValue::Undefined
            } else {
                super::jit_to_jsvalue(value_i64)?
            }
        } else {
            jit_i64_to_jsvalue(value_i64)
        };

        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers set by cache_rt_ptrs;
            // valid for thread lifetime.
            let heap_ref = unsafe { &*ptrs.heap };
            // SAFETY: single-threaded JIT; no concurrent mutable
            // borrows during push fast path.
            let heap = unsafe { &*heap_ref.as_ptr() };
            match heap.get(obj_idx)? {
                JsValue::Array(arr) => {
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let v = unsafe { &mut *arr.as_ptr() };
                    v.push(value);
                    let new_len = v.len() as i64;
                    Some(new_len)
                }
                _ => None,
            }
        } else {
            RT_HEAP.with(|heap| {
                let heap = heap.borrow();
                match heap.get(obj_idx)? {
                    JsValue::Array(arr) => {
                        let mut v = arr.borrow_mut();
                        v.push(value);
                        let new_len = v.len() as i64;
                        Some(new_len)
                    }
                    _ => None,
                }
            })
        }
    }
    // ── Inline-friendly keyed-property helpers for Maglev ──────────────

    /// Result from inline keyed-property helpers.
    ///
    /// On the SysV AMD64 ABI this 16-byte `#[repr(C)]` struct is
    /// returned in `RAX` (`value`) and `RDX` (`hit`) — no memory
    /// indirection needed.
    #[repr(C)]
    pub struct InlineKeyedResult {
        /// JIT `i64` element value (meaningful only when `hit != 0`).
        pub value: i64,
        /// Non-zero when the inline path succeeded; zero when the
        /// caller must fall back to the full generic stub.
        pub hit: i64,
    }

    impl InlineKeyedResult {
        /// Wraps an `Option<i64>` from a full generic stub into a result.
        /// `Some(v)` → hit, `None` → miss (fall back to generic path).
        #[inline(always)]
        fn from_generic(v: Option<i64>) -> Self {
            match v {
                Some(val) => Self { value: val, hit: 1 },
                None => Self { value: 0, hit: 0 },
            }
        }
    }

    /// Inline-friendly array element **load** for known-integer keys.
    ///
    /// Handles `Array[non-negative-int]` where the element is a common
    /// inline-encodable type (Smi, Boolean, Undefined, Null).  Returns
    /// `hit=0` for any case needing the full generic stub (non-array
    /// receiver, non-cached `RT_PTRS`, heap-object elements, etc.).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`smi_key`) – non-negative integer index.
    ///
    /// Returns [`InlineKeyedResult`] in `RAX:RDX`.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_inline_load_keyed_smi(
        obj_i64: i64,
        smi_key: i64,
    ) -> InlineKeyedResult {
        let ptrs = RT_PTRS.with(|p| p.get());
        inline_load_keyed_with_ptrs(obj_i64, smi_key, ptrs)
    }

    /// Like [`jit_runtime_inline_load_keyed_smi`] but accepts a
    /// pre-cached pointer to the `RT_PTRS` TLS cell in the 3rd argument,
    /// eliminating one TLS lookup per array load.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`smi_key`) – non-negative integer index.
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns [`InlineKeyedResult`] in `RAX:RDX`.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_inline_load_keyed_smi_r15(
        obj_i64: i64,
        smi_key: i64,
        rt_ptrs_cell: i64,
    ) -> InlineKeyedResult {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        inline_load_keyed_with_ptrs(obj_i64, smi_key, ptrs)
    }

    fn inline_load_keyed_with_ptrs(obj_i64: i64, smi_key: i64, ptrs: RtPtrs) -> InlineKeyedResult {
        if !is_heap_handle(obj_i64) || smi_key < 0 {
            return InlineKeyedResult::from_generic(lda_keyed_property_with_ptrs(
                obj_i64, smi_key, ptrs,
            ));
        }
        let key = smi_key as usize;
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        if !ptrs.is_cached() {
            return InlineKeyedResult::from_generic(lda_keyed_property_with_ptrs(
                obj_i64, smi_key, ptrs,
            ));
        }
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for
        // thread lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        match heap.get(obj_idx) {
            Some(JsValue::Array(arr)) => {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let data = unsafe { &*arr.as_ptr() };
                if key >= data.len() {
                    return InlineKeyedResult {
                        value: JIT_UNDEFINED,
                        hit: 1,
                    };
                }
                // SAFETY: bounds verified above.
                let v = unsafe { data.get_unchecked(key) };
                // Combined hole-check + encode via encode_array_element.
                if let Some(encoded) = encode_array_element(v) {
                    InlineKeyedResult {
                        value: encoded,
                        hit: 1,
                    }
                } else {
                    // Heap-object element — delegate to full generic stub
                    // for heap-handle allocation.
                    InlineKeyedResult::from_generic(lda_keyed_property_with_ptrs(
                        obj_i64, smi_key, ptrs,
                    ))
                }
            }
            _ => InlineKeyedResult::from_generic(lda_keyed_property_with_ptrs(
                obj_i64, smi_key, ptrs,
            )),
        }
    }

    // ── Inline named-property IC helpers ────────────────────────────────────

    /// Result from inline named-property helpers.
    ///
    /// On the SysV AMD64 ABI this 16-byte `#[repr(C)]` struct is
    /// returned in `RAX` (`value`) and `RDX` (`hit`) — no memory
    /// indirection needed.
    #[repr(C)]
    pub struct InlineNamedResult {
        /// JIT `i64` property value (meaningful only when `hit != 0`).
        pub value: i64,
        /// Non-zero when the inline path succeeded; zero when the
        /// caller must fall back to the full generic stub.
        pub hit: i64,
    }

    impl InlineNamedResult {
        const MISS: Self = Self { value: 0, hit: 0 };
    }

    /// Inline-friendly **named property load** using the IC fast path.
    ///
    /// Probes both the own-property and prototype inline caches
    /// without allocating or cloning.  Returns `hit=0` when the caller
    /// must fall back to the full [`jit_runtime_lda_named_property`]
    /// stub (IC miss, non-PlainObject receiver, heap-object result
    /// that needs a new handle, etc.).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    ///
    /// Returns [`InlineNamedResult`] in `RAX:RDX`.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_inline_lda_named_property(
        obj_i64: i64,
        name_idx: u32,
    ) -> InlineNamedResult {
        let ptrs = RT_PTRS.with(|p| p.get());
        inline_lda_named_with_ptrs(obj_i64, name_idx, ptrs)
    }

    /// Like [`jit_runtime_inline_lda_named_property`] but accepts a
    /// pre-cached pointer to the `RT_PTRS` TLS cell in the 3rd argument,
    /// eliminating one TLS lookup per property load.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns [`InlineNamedResult`] in `RAX:RDX`.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_inline_lda_named_property_r15(
        obj_i64: i64,
        name_idx: u32,
        rt_ptrs_cell: i64,
    ) -> InlineNamedResult {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        inline_lda_named_with_ptrs(obj_i64, name_idx, ptrs)
    }

    /// Shared implementation for inline named-property load.
    ///
    /// Probes the own-property IC and prototype IC.  Returns a hit only
    /// when the result can be encoded without allocating a new heap
    /// handle (primitives, or heap objects whose cached handle is still
    /// valid).
    fn inline_lda_named_with_ptrs(obj_i64: i64, name_idx: u32, ptrs: RtPtrs) -> InlineNamedResult {
        if !is_heap_handle(obj_i64) || !ptrs.is_cached() {
            return InlineNamedResult::MISS;
        }

        // ── Handle-based proto IC (skips heap dereference) ──────────
        let ic_ref = unsafe { &*ptrs.prop_ic };
        let proto_slot = (name_idx & 31) as usize;
        {
            // SAFETY: scoped borrow; dropped before any IC mutation.
            let cache = unsafe { &*ic_ref.as_ptr() };
            let pe = cache.proto[proto_slot];
            if pe.receiver_handle == obj_i64 && pe.name_idx == name_idx {
                let epoch = combined_ic_epoch();
                if pe.global_epoch == epoch {
                    return InlineNamedResult {
                        value: pe.cached_value,
                        hit: 1,
                    };
                }
            }
        }

        let idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        let lda_slot = (name_idx & 63) as usize;

        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };

        let heap_val = match heap.get(idx) {
            Some(v) => v,
            None => return InlineNamedResult::MISS,
        };

        if let JsValue::PlainObject(map_rc) = heap_val {
            // SAFETY: PropertyMap lives in Rc, separate from heap Vec.
            let map = unsafe { &*map_rc.as_ptr() };
            let shape = map.shape_id();

            let (ic_entry, proto_entry) = {
                // SAFETY: scoped borrow; dropped before IC mutation.
                let cache = unsafe { &*ic_ref.as_ptr() };
                (cache.lda[lda_slot], cache.proto[proto_slot])
            };

            // ── Own-property IC hit ────────────────────────────────
            if ic_entry.name_idx == name_idx && ic_entry.shape == shape {
                // SAFETY: shape_id unchanged since IC population ⇒
                // offset is still valid (shape changes on every
                // structural mutation that could invalidate an offset).
                let val = unsafe { map.get_by_offset_unchecked(ic_entry.offset) };
                // Only return a hit for inline-encodable values
                // (primitives and heap objects with a valid cached
                // handle).  Anything else needs the full stub for
                // heap-handle allocation.
                if let Some(result) =
                    try_encode_ic_inline(val, ic_entry.cached_ptr, ic_entry.cached_handle)
                {
                    // Promote to proto IC so the next access for the
                    // same handle skips the heap dereference entirely.
                    let epoch = combined_ic_epoch();
                    let cache_mut = unsafe { &mut *ic_ref.as_ptr() };
                    let pe = &mut cache_mut.proto[proto_slot];
                    pe.name_idx = name_idx;
                    pe.shape = shape;
                    pe.receiver_handle = obj_i64;
                    pe.global_epoch = epoch;
                    pe.cached_value = result;

                    return InlineNamedResult {
                        value: result,
                        hit: 1,
                    };
                }
            }

            // ── Prototype IC hit (shape-based fallback) ────────────
            if proto_entry.name_idx == name_idx && proto_entry.shape == shape {
                // Re-stamp the entry so handle+epoch check hits next time.
                let epoch = combined_ic_epoch();
                let cache_mut = unsafe { &mut *ic_ref.as_ptr() };
                let pe = &mut cache_mut.proto[proto_slot];
                pe.receiver_handle = obj_i64;
                pe.global_epoch = epoch;
                return InlineNamedResult {
                    value: proto_entry.cached_value,
                    hit: 1,
                };
            }
        }

        InlineNamedResult::MISS
    }

    /// Try to encode a property value as an `i64` without allocating a
    /// new heap handle.  Returns `None` when the value requires a fresh
    /// heap-handle allocation (which the full stub handles).
    #[inline]
    fn try_encode_ic_inline(
        val: &JsValue,
        entry_cached_ptr: usize,
        entry_cached_handle: i64,
    ) -> Option<i64> {
        match val {
            JsValue::Smi(n) => Some(i64::from(*n)),
            JsValue::Boolean(true) => Some(JIT_TRUE),
            JsValue::Boolean(false) => Some(JIT_FALSE),
            JsValue::Undefined => Some(JIT_UNDEFINED),
            JsValue::Null => Some(JIT_NULL),
            JsValue::HeapNumber(f) => {
                let f = *f;
                if f.fract() == 0.0 && f >= i32::MIN as f64 && f <= i32::MAX as f64 {
                    Some(f as i64)
                } else {
                    None // needs heap handle
                }
            }
            _ => {
                // For heap objects, check if the cached handle is still valid.
                let ptr = jsvalue_rc_ptr(val);
                if ptr != 0 && ptr == entry_cached_ptr && entry_cached_handle != 0 {
                    Some(entry_cached_handle)
                } else {
                    None // needs fresh heap handle → full stub
                }
            }
        }
    }

    /// Inline-friendly array element **store** for known-integer keys.
    ///
    /// Handles **in-bounds** `Array[non-negative-int] = value` where the
    /// value is a common inline-decodable type (Smi, Boolean, Undefined).
    /// Returns `hit=0` for any case needing the full generic stub
    /// (non-array receiver, out-of-bounds index, heap-object values).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`smi_key`) – non-negative integer index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value.
    ///
    /// Returns [`InlineKeyedResult`] in `RAX:RDX`.
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_inline_store_keyed_smi(
        obj_i64: i64,
        smi_key: i64,
        value_i64: i64,
    ) -> InlineKeyedResult {
        let ptrs = RT_PTRS.with(|p| p.get());
        inline_store_keyed_with_ptrs(obj_i64, smi_key, value_i64, ptrs)
    }

    /// Like [`jit_runtime_inline_store_keyed_smi`] but accepts a
    /// pre-cached pointer to the `RT_PTRS` TLS cell in the 4th argument,
    /// eliminating one TLS lookup per array store.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`smi_key`) – non-negative integer index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value.
    /// * `RCX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns [`InlineKeyedResult`] in `RAX:RDX`.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_inline_store_keyed_smi_r15(
        obj_i64: i64,
        smi_key: i64,
        value_i64: i64,
        rt_ptrs_cell: i64,
    ) -> InlineKeyedResult {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        inline_store_keyed_with_ptrs(obj_i64, smi_key, value_i64, ptrs)
    }

    /// Combined inline store + growth + IC update.  Handles both in-bounds
    /// stores and out-of-bounds sequential appends in a single FFI call,
    /// eliminating the need for the generic stub fallback on array growth.
    pub extern "C" fn jit_runtime_inline_store_keyed_grow_ic_r15(
        obj_i64: i64,
        smi_key: i64,
        value_i64: i64,
        ic_ptr: i64,
        rt_ptrs_cell: i64,
    ) -> InlineKeyedResult {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        inline_store_keyed_grow_ic_with_ptrs(obj_i64, smi_key, value_i64, ic_ptr, ptrs)
    }

    fn inline_store_keyed_with_ptrs(
        obj_i64: i64,
        smi_key: i64,
        value_i64: i64,
        ptrs: RtPtrs,
    ) -> InlineKeyedResult {
        if !is_heap_handle(obj_i64) || smi_key < 0 || is_heap_handle(value_i64) {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        }
        let key = smi_key as usize;
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        if !ptrs.is_cached() {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        }
        // Decode value inline for common types only.
        // Check booleans first — hot path for sieve-like patterns where
        // every element is `true`/`false`, avoiding the wider Smi range
        // comparison on each store.
        let value = if value_i64 == JIT_TRUE {
            JsValue::Boolean(true)
        } else if value_i64 == JIT_FALSE {
            JsValue::Boolean(false)
        } else if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
            JsValue::Smi(value_i64 as i32)
        } else if value_i64 == JIT_UNDEFINED {
            JsValue::Undefined
        } else {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        };

        // SAFETY: cached pointers valid for thread lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        match heap.get(obj_idx) {
            Some(JsValue::Array(arr)) => {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let v = unsafe { &mut *arr.as_ptr() };
                if key < v.len() {
                    // SAFETY: bounds verified above.
                    unsafe { *v.get_unchecked_mut(key) = value };
                    InlineKeyedResult {
                        value: value_i64,
                        hit: 1,
                    }
                } else {
                    // Out-of-bounds: needs full stub for Vec growth.
                    InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                        obj_i64, smi_key, value_i64, ptrs,
                    ))
                }
            }
            _ => InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            )),
        }
    }

    /// Inline store with array growth and IC update.  Handles both
    /// in-bounds and out-of-bounds stores in a single call.
    ///
    /// When growth is needed the Vec is pre-extended to its full
    /// capacity (filled with `TheHole`) so that subsequent Maglev
    /// inline IC checks (`key < cached_len`) succeed without calling
    /// back into Rust.  This turns O(n) IC misses during sequential
    /// fills (e.g. `for (i=0;i<=n;i++) arr[i]=v`) into O(log n).
    fn inline_store_keyed_grow_ic_with_ptrs(
        obj_i64: i64,
        smi_key: i64,
        value_i64: i64,
        ic_ptr: i64,
        ptrs: RtPtrs,
    ) -> InlineKeyedResult {
        if !is_heap_handle(obj_i64) || smi_key < 0 || is_heap_handle(value_i64) {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        }
        let key = smi_key as usize;
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        if !ptrs.is_cached() {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        }
        // Decode value — check booleans first (hot for sieve-like
        // patterns) to avoid the wider Smi range comparison.
        let value = if value_i64 == JIT_TRUE {
            JsValue::Boolean(true)
        } else if value_i64 == JIT_FALSE {
            JsValue::Boolean(false)
        } else if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
            JsValue::Smi(value_i64 as i32)
        } else if value_i64 == JIT_UNDEFINED {
            JsValue::Undefined
        } else {
            return InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            ));
        };

        // SAFETY: cached pointers valid for thread lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        match heap.get(obj_idx) {
            Some(JsValue::Array(arr)) => {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let v = unsafe { &mut *arr.as_ptr() };
                if key < v.len() {
                    // SAFETY: bounds verified above.
                    unsafe { *v.get_unchecked_mut(key) = value };
                } else {
                    // Out-of-bounds — grow allocation if needed.
                    if key >= v.capacity() {
                        let new_cap = (key + 1).next_power_of_two();
                        v.reserve(new_cap.saturating_sub(v.len()));
                    }
                    // Pre-extend to full capacity with TheHole so that
                    // the Maglev inline IC can service all subsequent
                    // stores up to this capacity without another FFI
                    // call.  The cost is O(cap) TheHole writes, but it
                    // happens only O(log n) times (at each power-of-2
                    // boundary).
                    let cap = v.capacity();
                    v.resize(cap, JsValue::TheHole);
                    // SAFETY: key < cap after resize.
                    unsafe { *v.get_unchecked_mut(key) = value };
                }

                // Update caller's IC slots so subsequent x86 inline
                // checks can succeed with the new data pointer and
                // length.  After pre-extend, len == capacity, so the
                // inline IC will accept every key < capacity.
                if ic_ptr != 0 {
                    // SAFETY: ic_ptr points to 4 writable i64s on the
                    // caller's stack frame (Maglev-generated code).
                    let ic_slots = ic_ptr as *mut i64;
                    unsafe {
                        *ic_slots = obj_i64;
                        *ic_slots.add(1) = v.as_ptr() as i64;
                        *ic_slots.add(2) = v.len() as i64;
                        *ic_slots.add(3) = v as *mut Vec<JsValue> as i64;
                    }
                }
                InlineKeyedResult {
                    value: value_i64,
                    hit: 1,
                }
            }
            _ => InlineKeyedResult::from_generic(sta_keyed_property_with_ptrs(
                obj_i64, smi_key, value_i64, ptrs,
            )),
        }
    }

    // ── JsValue layout probing for true-inline array access ────────────

    /// Compile-time-discovered byte layout of [`JsValue`] for the Smi and
    /// Boolean variants.  Used by the Maglev codegen to emit inline x86
    /// loads/stores without calling into Rust.
    #[derive(Debug, Clone, Copy)]
    pub struct JsValueLayout {
        /// `std::mem::size_of::<JsValue>()`.
        pub jsvalue_size: usize,
        /// Byte offset of the enum discriminant within a `JsValue`.
        pub disc_offset: usize,
        /// Discriminant byte value for `JsValue::Smi(_)`.
        pub smi_disc: u8,
        /// Byte offset of the `i32` payload inside `JsValue::Smi(_)`.
        pub smi_payload_offset: usize,
        /// Discriminant byte value for `JsValue::Boolean(_)`.
        pub bool_disc: u8,
        /// Byte offset of the `bool` payload inside `JsValue::Boolean(_)`.
        pub bool_payload_offset: usize,
        /// Discriminant byte value for `JsValue::Undefined`.
        pub undef_disc: u8,
        /// Discriminant byte value for `JsValue::Null`.
        pub null_disc: u8,
        /// Discriminant byte value for `JsValue::TheHole`.
        pub hole_disc: u8,
    }

    /// Probe the in-memory byte layout of [`JsValue`] for discriminant
    /// and payload offsets.  Panics if the layout cannot be determined
    /// (i.e. niche optimisation has made the layout non-trivial).
    pub fn probe_jsvalue_layout() -> JsValueLayout {
        let size = std::mem::size_of::<JsValue>();
        // Transmute trick: read the raw bytes of known JsValue instances.
        // SAFETY: we only read; the values are never used as JsValue again.
        fn raw(v: &JsValue, size: usize) -> Vec<u8> {
            let ptr = v as *const JsValue as *const u8;
            (0..size).map(|i| unsafe { *ptr.add(i) }).collect()
        }

        let undef = JsValue::Undefined;
        let null = JsValue::Null;
        let hole = JsValue::TheHole;
        let bool_false = JsValue::Boolean(false);
        let bool_true = JsValue::Boolean(true);
        let smi_probe = JsValue::Smi(0x12345678_i32);
        let smi_zero = JsValue::Smi(0);

        let b_undef = raw(&undef, size);
        let b_null = raw(&null, size);
        let b_hole = raw(&hole, size);
        let b_false = raw(&bool_false, size);
        let b_true = raw(&bool_true, size);
        let b_smi = raw(&smi_probe, size);
        let b_smi0 = raw(&smi_zero, size);

        // ── Find the discriminant offset ──
        // All field-less / small variants differ only in the discriminant
        // byte.  We find the first byte position where Undefined, Null,
        // Boolean(false), and Smi(0) all differ pairwise.
        let disc_offset = (0..size)
            .find(|&i| {
                let vals = [b_undef[i], b_null[i], b_false[i], b_smi0[i]];
                // All four must be distinct.
                vals[0] != vals[1]
                    && vals[0] != vals[2]
                    && vals[0] != vals[3]
                    && vals[1] != vals[2]
                    && vals[1] != vals[3]
                    && vals[2] != vals[3]
            })
            .expect("JsValue: cannot find discriminant byte — layout may use niche optimisation");

        let undef_disc = b_undef[disc_offset];
        let null_disc = b_null[disc_offset];
        let hole_disc = b_hole[disc_offset];
        let bool_disc = b_false[disc_offset];
        let smi_disc = b_smi[disc_offset];

        // ── Find the Smi i32 payload offset ──
        // Look for 0x78 0x56 0x34 0x12 (LE bytes of 0x12345678).
        let smi_payload_offset = b_smi
            .windows(4)
            .position(|w| w == [0x78, 0x56, 0x34, 0x12])
            .expect("JsValue: cannot find Smi i32 payload in byte representation");

        // ── Find the Boolean payload offset ──
        // Boolean(false) and Boolean(true) share the same discriminant;
        // the first byte that differs is the bool payload.
        let bool_payload_offset = (0..size)
            .find(|&i| b_false[i] != b_true[i])
            .expect("JsValue: cannot find Boolean payload");

        // Sanity: the bool discriminant byte must NOT be the payload byte.
        assert_ne!(disc_offset, bool_payload_offset);

        JsValueLayout {
            jsvalue_size: size,
            disc_offset,
            smi_disc,
            smi_payload_offset,
            bool_disc,
            bool_payload_offset,
            undef_disc,
            null_disc,
            hole_disc,
        }
    }

    // ── JsContext layout probing for inline context-slot access ────────

    /// Compile-time-discovered byte layout of `RefCell<JsContext>` used
    /// by Maglev codegen to emit inline x86 loads/stores of closure
    /// context slots without calling into Rust.
    #[derive(Debug, Clone, Copy)]
    #[allow(dead_code)]
    pub struct JsContextLayout {
        /// Byte offset from a `*const RefCell<JsContext>` to the
        /// `Vec<JsValue>` data-pointer field inside `JsContext::slots`.
        pub slots_data_ptr_offset: usize,
        /// Byte offset to the `Vec<JsValue>` len field.
        pub slots_len_offset: usize,
    }

    /// Probe the in-memory byte layout of `RefCell<JsContext>` to find
    /// the offsets of the slots `Vec` fields (data pointer and length).
    pub fn probe_jscontext_layout() -> JsContextLayout {
        use std::cell::RefCell;

        // Create a JsContext with 3 slots so the Vec has a non-null data ptr.
        let ctx = JsContext {
            slots: vec![JsValue::Smi(0); 3],
            parent: None,
        };
        let data_ptr = ctx.slots.as_ptr() as usize;
        let data_len = ctx.slots.len();

        let rc = RefCell::new(ctx);
        let rc_ptr = &rc as *const RefCell<JsContext> as *const u8;
        let rc_size = std::mem::size_of::<RefCell<JsContext>>();

        // Scan for the data pointer (as a usize / 8-byte value).
        let ptr_bytes = data_ptr.to_ne_bytes();
        let mut slots_data_ptr_offset = None;
        for i in 0..=(rc_size.saturating_sub(8)) {
            // SAFETY: we only read bytes of the RefCell on the stack.
            let found = (0..8).all(|j| unsafe { *rc_ptr.add(i + j) } == ptr_bytes[j]);
            if found {
                slots_data_ptr_offset = Some(i);
                break;
            }
        }
        let slots_data_ptr_offset = slots_data_ptr_offset
            .expect("JsContext: cannot find Vec data pointer in RefCell layout");

        // The Vec len field is at the next usize slot after data pointer.
        let len_offset = slots_data_ptr_offset + std::mem::size_of::<usize>();
        // SAFETY: read the len field and verify it matches.
        let probed_len = unsafe { *(rc_ptr.add(len_offset) as *const usize) };
        assert_eq!(
            probed_len, data_len,
            "JsContext: Vec len field not at expected offset"
        );

        JsContextLayout {
            slots_data_ptr_offset,
            slots_len_offset: len_offset,
        }
    }

    // ── Vec<JsValue> layout probing for inline array append ────────────

    /// Compile-time-discovered byte layout of `Vec<JsValue>`.
    ///
    /// Used by Maglev codegen to read `len` and `capacity` fields and to
    /// increment `len` directly from x86 during inline array append,
    /// bypassing any FFI call.
    #[derive(Debug, Clone, Copy)]
    pub struct VecJsValueLayout {
        /// Byte offset of the data-pointer field within `Vec<JsValue>`.
        pub ptr_offset: usize,
        /// Byte offset of the `len` field within `Vec<JsValue>`.
        pub len_offset: usize,
        /// Byte offset of the `capacity` field within `Vec<JsValue>`.
        pub cap_offset: usize,
    }

    /// Probe the in-memory byte layout of `Vec<JsValue>` to find
    /// the offsets of the data-pointer, `len`, and `capacity` fields.
    ///
    /// Uses distinctive element counts to locate each field by scanning
    /// the raw bytes of a stack-allocated Vec.
    pub fn probe_vec_jsvalue_layout() -> VecJsValueLayout {
        // Use distinctive values unlikely to appear as pointer bytes.
        let mut v: Vec<JsValue> = Vec::with_capacity(97);
        for _ in 0..37 {
            v.push(JsValue::Undefined);
        }
        assert_eq!(v.len(), 37);
        assert_eq!(v.capacity(), 97);

        let vec_ptr = &v as *const Vec<JsValue> as *const u8;
        let vec_size = std::mem::size_of::<Vec<JsValue>>();

        let len_bytes = 37usize.to_ne_bytes();
        let cap_bytes = 97usize.to_ne_bytes();
        let data_ptr_val = v.as_ptr() as usize;
        let ptr_bytes = data_ptr_val.to_ne_bytes();

        let mut ptr_offset = None;
        let mut len_offset = None;
        let mut cap_offset = None;

        for i in 0..=(vec_size.saturating_sub(8)) {
            // SAFETY: reading bytes of a stack-allocated Vec.
            let matches_ptr = (0..8).all(|j| unsafe { *vec_ptr.add(i + j) } == ptr_bytes[j]);
            let matches_len = (0..8).all(|j| unsafe { *vec_ptr.add(i + j) } == len_bytes[j]);
            let matches_cap = (0..8).all(|j| unsafe { *vec_ptr.add(i + j) } == cap_bytes[j]);

            if matches_ptr && ptr_offset.is_none() {
                ptr_offset = Some(i);
            }
            if matches_len && len_offset.is_none() {
                len_offset = Some(i);
            }
            if matches_cap && cap_offset.is_none() {
                cap_offset = Some(i);
            }
        }

        let ptr_offset =
            ptr_offset.expect("Vec<JsValue>: cannot find ptr field in byte representation");
        let len_offset =
            len_offset.expect("Vec<JsValue>: cannot find len field in byte representation");
        let cap_offset =
            cap_offset.expect("Vec<JsValue>: cannot find capacity field in byte representation");

        // Cross-check: push one more element and verify len changed.
        v.push(JsValue::Undefined);
        let probed_len = unsafe { *(vec_ptr.add(len_offset) as *const usize) };
        assert_eq!(
            probed_len, 38,
            "Vec<JsValue>: len field not at expected offset"
        );
        let probed_cap = unsafe { *(vec_ptr.add(cap_offset) as *const usize) };
        assert_eq!(
            probed_cap, 97,
            "Vec<JsValue>: capacity field not at expected offset"
        );
        // Cross-check the data pointer — may have moved after push.
        let probed_ptr = unsafe { *(vec_ptr.add(ptr_offset) as *const usize) };
        assert_eq!(
            probed_ptr,
            v.as_ptr() as usize,
            "Vec<JsValue>: ptr field not at expected offset"
        );

        VecJsValueLayout {
            ptr_offset,
            len_offset,
            cap_offset,
        }
    }

    // ── PropertyMap layout probing for inline named-property IC ────────

    /// Compile-time-discovered byte layout of [`PropertyMap`] for the
    /// inline named-property IC fast path in Maglev codegen.
    ///
    /// The inline IC needs to read `shape_id` and the `values` Vec data
    /// pointer directly from a `PropertyMap` pointer using only x86
    /// instructions (no Rust function calls).
    #[derive(Debug, Clone, Copy)]
    pub struct PropertyMapLayout {
        /// Byte offset of `shape_id: u64` within `PropertyMap`.
        #[allow(dead_code)]
        pub shape_id_offset: usize,
        /// Byte offset of the `values` Vec data-pointer field within
        /// `PropertyMap`.  The Vec layout is `[ptr, len, cap]`; this
        /// offset points to the `ptr` field.
        #[allow(dead_code)]
        pub values_data_ptr_offset: usize,
    }

    /// Probe the in-memory byte layout of [`PropertyMap`] to find the
    /// offsets of `shape_id` and `values` Vec data pointer.
    ///
    /// Uses distinctive sentinel values to locate each field by scanning
    /// the raw bytes of a stack-allocated PropertyMap.
    ///
    /// **Important:** creates a throwaway `PropertyMap` first to advance
    /// the thread-local shape counter.  Without this, `shape_id` and
    /// `values.len()` can have identical byte patterns (both equal the
    /// number of inserts) — causing the scan to pick the wrong offset
    /// and panicking on the Maglev compilation thread.
    pub fn probe_propertymap_layout() -> PropertyMapLayout {
        // Burn one shape_id so the counter is non-zero when the probe
        // map is constructed.  This breaks the lock-step between
        // shape_id (= S0 + num_inserts) and values.len (= num_inserts)
        // that occurs when S0 = 0 on a fresh thread.
        {
            let _burn = PropertyMap::with_capacity(0);
        }

        // Create a PropertyMap with a distinctive shape_id.
        let mut map = PropertyMap::with_capacity(4);
        // Insert properties so the values Vec has a non-null data ptr.
        map.insert("__probe_a__".to_string(), JsValue::Smi(0));
        map.insert("__probe_b__".to_string(), JsValue::Smi(0));

        let data_ptr = map.values_as_slice().as_ptr() as usize;

        // Set a distinctive shape_id by performing structural mutations.
        // Each insert bumps shape_id.  Read the current value.
        let shape = map.shape_id();
        assert_ne!(shape, 0, "probe: shape_id should be non-zero after inserts");

        let map_ptr = &map as *const PropertyMap as *const u8;
        let map_size = std::mem::size_of::<PropertyMap>();

        // ── Find shape_id offset ──
        // Collect ALL candidate offsets that match the current shape_id
        // byte pattern — even with the burn above, small counter values
        // could still alias with other fields on some platforms.
        let shape_bytes = shape.to_ne_bytes();
        let mut shape_candidates: Vec<usize> = Vec::new();
        for i in 0..=(map_size.saturating_sub(8)) {
            // SAFETY: reading bytes of a stack-allocated PropertyMap.
            let matches = (0..8).all(|j| unsafe { *map_ptr.add(i + j) } == shape_bytes[j]);
            if matches {
                shape_candidates.push(i);
            }
        }
        assert!(
            !shape_candidates.is_empty(),
            "PropertyMap: cannot find shape_id field in byte representation"
        );

        // If there's only one candidate, use it directly.  Otherwise,
        // insert another property and use the mutation to disambiguate.
        let shape_id_offset = if shape_candidates.len() == 1 {
            shape_candidates[0]
        } else {
            map.insert("__probe_c__".to_string(), JsValue::Smi(0));
            let new_shape = map.shape_id();
            shape_candidates
                .iter()
                .copied()
                .find(|&offset| {
                    let probed = unsafe { *(map_ptr.add(offset) as *const u64) };
                    probed == new_shape
                })
                .expect(
                    "PropertyMap: shape_id offset ambiguous — none of the \
                     candidate offsets track shape_id after mutation",
                )
        };

        // ── Find values Vec data-pointer offset ──
        let ptr_bytes = data_ptr.to_ne_bytes();
        let mut values_data_ptr_offset = None;
        for i in 0..=(map_size.saturating_sub(8)) {
            // SAFETY: reading bytes of a stack-allocated PropertyMap.
            let matches = (0..8).all(|j| unsafe { *map_ptr.add(i + j) } == ptr_bytes[j]);
            if matches {
                values_data_ptr_offset = Some(i);
                break;
            }
        }
        let values_data_ptr_offset = values_data_ptr_offset
            .expect("PropertyMap: cannot find values Vec data pointer in byte representation");

        // Final cross-check: verify shape_id reads correctly.
        let final_probed = unsafe { *(map_ptr.add(shape_id_offset) as *const u64) };
        assert_eq!(
            final_probed,
            map.shape_id(),
            "PropertyMap: shape_id field not at expected offset"
        );

        // Cross-check: verify values data pointer reads correctly.
        let new_data_ptr = map.values_as_slice().as_ptr() as usize;
        let probed_ptr = unsafe { *(map_ptr.add(values_data_ptr_offset) as *const usize) };
        assert_eq!(
            probed_ptr, new_data_ptr,
            "PropertyMap: values data pointer not at expected offset"
        );

        PropertyMapLayout {
            shape_id_offset,
            values_data_ptr_offset,
        }
    }

    // ── Named property inline-cache fill ────────────────────────────────

    /// Result returned by [`jit_runtime_fill_named_ic_r15`].
    ///
    /// Returned via SysV AMD64 ABI in RAX:RDX.
    #[repr(C)]
    pub struct NamedIcResult {
        /// JIT i64 property value (meaningful only when `hit != 0`).
        pub value: i64,
        /// Non-zero when the inline path succeeded.
        pub hit: i64,
    }

    impl NamedIcResult {
        const MISS: Self = Self { value: 0, hit: 0 };
    }

    /// Combined named-property IC fill **and** load.
    ///
    /// Resolves the heap handle, fills the 4-slot inline cache at
    /// `ic_slots` (`[handle, map_ptr, shape_id, elem_addr]`), and performs
    /// the property load — all in a single call.
    ///
    /// Only fills the IC for own data-property hits on `PlainObject`
    /// receivers (not prototype hits, getters, array length, etc.).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`name_idx`) – constant-pool index of the property name.
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    /// * `RCX` (`ic_slots`) – pointer to 4 consecutive `i64` on the
    ///   stack: `[handle, map_ptr, shape_id, elem_addr]`.
    ///
    /// Returns [`NamedIcResult`] in `RAX:RDX`.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub extern "C" fn jit_runtime_fill_named_ic_r15(
        obj_i64: i64,
        name_idx: u32,
        rt_ptrs_cell: i64,
        ic_slots: *mut [i64; 4],
    ) -> NamedIcResult {
        if !is_heap_handle(obj_i64) {
            return NamedIcResult::MISS;
        }
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return NamedIcResult::MISS;
        }
        // SAFETY: cached pointers valid for thread lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        let heap_val = match heap.get(obj_idx) {
            Some(v) => v,
            None => {
                return NamedIcResult::MISS;
            }
        };

        if let JsValue::PlainObject(map_rc) = heap_val {
            let prop_name = match get_rt_string_constant_ref(name_idx) {
                Some(n) => n,
                None => {
                    return NamedIcResult::MISS;
                }
            };
            // SAFETY: single-threaded JIT; no concurrent borrows.
            let map = unsafe { &*map_rc.as_ptr() };
            let shape = map.shape_id();
            if let Some(offset) = map.offset_of(prop_name) {
                // SAFETY: offset valid while shape_id unchanged.
                let val = unsafe { map.get_by_offset_unchecked(offset) };
                // Encode the value as a JIT i64.  Primitives (Smi,
                // Boolean, Undefined, Null) are encoded directly.
                // For heap objects (PlainObject, Array, String, …) we
                // allocate a heap handle so the IC is filled for ALL
                // property types — this avoids falling through to the
                // expensive generic-fallback path on IC miss.
                let encoded = match try_encode_named_ic_value(val) {
                    Some(enc) => enc,
                    None => alloc_heap_handle(val.clone()),
                };
                // Fill IC slots: [handle, map_ptr, shape_id, elem_addr]
                // Store the absolute element address (data_ptr + offset *
                // sizeof(JsValue)) so the inline fast path can skip the
                // values-data-pointer load and index multiplication.
                let map_raw = map_rc.as_ptr() as i64;
                let elem_addr = unsafe {
                    (map.values_as_slice().as_ptr() as *const u8)
                        .add(offset * std::mem::size_of::<JsValue>())
                } as i64;
                unsafe {
                    (*ic_slots)[0] = obj_i64;
                    (*ic_slots)[1] = map_raw;
                    (*ic_slots)[2] = shape as i64;
                    (*ic_slots)[3] = elem_addr;
                }
                return NamedIcResult {
                    value: encoded,
                    hit: 1,
                };
            }
        } else if let JsValue::Array(arr_rc) = heap_val {
            // Array `.length` fast path: return current length directly
            // without filling the IC (length is dynamic, not cacheable
            // at a fixed address).  This avoids the expensive fallthrough
            // to `jit_runtime_lda_named_property`.
            let prop_name = match get_rt_string_constant_ref(name_idx) {
                Some(n) => n,
                None => return NamedIcResult::MISS,
            };
            if prop_name == "length" {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let len = unsafe { &*arr_rc.as_ptr() }.len() as i64;
                return NamedIcResult { value: len, hit: 1 };
            }
        }

        // Fall back to the existing inline helper (probes proto IC too).
        let result = inline_lda_named_with_ptrs(obj_i64, name_idx, ptrs);
        let r = NamedIcResult {
            value: result.value,
            hit: result.hit,
        };
        r
    }

    /// Try to encode a property value as a JIT i64 without allocating.
    ///
    /// Returns `None` for heap objects that would need a new handle.
    fn try_encode_named_ic_value(val: &JsValue) -> Option<i64> {
        match val {
            JsValue::Smi(n) => Some(i64::from(*n)),
            JsValue::Boolean(b) => Some(if *b { JIT_TRUE } else { JIT_FALSE }),
            JsValue::Undefined => Some(JIT_UNDEFINED),
            JsValue::Null => Some(JIT_NULL),
            _ => None,
        }
    }

    /// Clone the [`JsValue`] at the given raw pointer and allocate a JIT
    /// heap handle for it.
    ///
    /// Called from Maglev-generated code when the inline IC hits a
    /// non-primitive property (Object, Array, String, etc.) that cannot
    /// be encoded without a heap allocation.  Much cheaper than a full
    /// IC fill call because it skips property-name lookup and shape
    /// validation — the caller (inline IC fast path) has already
    /// verified handle + shape.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`jsvalue_ptr`) – raw pointer to the `JsValue` in the
    ///   PropertyMap storage (element address computed by inline IC).
    ///
    /// Returns the JIT heap handle (`JIT_HEAP_TAG + idx`) in `RAX`.
    ///
    /// # Safety
    ///
    /// `jsvalue_ptr` must be a valid pointer to a live `JsValue` that
    /// will not be moved or deallocated during this call.  This is
    /// guaranteed by the inline IC: the shape_id check ensures the
    /// PropertyMap layout has not changed since the IC was populated.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_alloc_handle_for_jsvalue(jsvalue_ptr: i64) -> i64 {
        // SAFETY: caller guarantees jsvalue_ptr is a valid &JsValue.
        let val: &JsValue = unsafe { &*(jsvalue_ptr as *const JsValue) };
        alloc_heap_handle(val.clone())
    }

    /// Diagnostic: called from JIT when smi_guard fails at done_label.
    ///
    /// Prints the failing value and node ID so CI logs reveal the root cause.
    /// The value is the raw i64 in RAX at the point of failure.  Known
    /// encodings: Smi = sign-extended i32, JIT_UNDEFINED = 0x1_0000_0002,
    /// JIT_FALSE/TRUE = 0x1_0000_000{0,1}, heap handle = 0x2_0000_0000+idx.
    #[allow(dead_code)]
    pub extern "C" fn jit_smi_guard_fail_log(value: i64, node_id: i64) {
        eprintln!(
            "[SMI_GUARD_FAIL] node={} value=0x{:016x} ({})",
            node_id, value as u64, value
        );
    }

    // ── Array inline-cache fill + load ──────────────────────────────────

    /// Result returned by [`jit_runtime_fill_array_ic_r15`].
    ///
    /// Returned in `RAX:RDX` via SysV AMD64 ABI.
    #[repr(C)]
    pub struct ArrayIcInfo {
        /// Raw pointer to the `Vec<JsValue>` data buffer, or 0 on failure.
        pub data_ptr: i64,
        /// Current length of the `Vec`, or 0 on failure.
        pub len: i64,
    }

    impl ArrayIcInfo {
        const MISS: Self = Self {
            data_ptr: 0,
            len: 0,
        };
    }

    /// Resolve a heap handle to its underlying array element buffer and
    /// fill the caller-provided inline-cache slots.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    ///
    /// Returns [`ArrayIcInfo`] in `RAX:RDX` (`data_ptr`, `len`).
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_fill_array_ic_r15(
        obj_i64: i64,
        rt_ptrs_cell: i64,
    ) -> ArrayIcInfo {
        if !is_heap_handle(obj_i64) {
            return ArrayIcInfo::MISS;
        }
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();
        if !ptrs.is_cached() {
            return ArrayIcInfo::MISS;
        }
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread
        // lifetime.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        match heap.get(obj_idx) {
            Some(JsValue::Array(arr)) => {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let data = unsafe { &*arr.as_ptr() };
                ArrayIcInfo {
                    data_ptr: data.as_ptr() as i64,
                    len: data.len() as i64,
                }
            }
            _ => ArrayIcInfo::MISS,
        }
    }

    /// Combined array IC fill **and** keyed load.
    ///
    /// Resolves the heap handle, fills the 3-slot inline cache at
    /// `ic_slots` (`[handle, data_ptr, len]`), and performs the element
    /// load — all in a single call.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`) – JIT i64 encoding of the receiver.
    /// * `RSI` (`smi_key`) – non-negative integer index.
    /// * `RDX` (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    /// * `RCX` (`ic_slots`) – pointer to 3 consecutive `i64` on the
    ///   stack: `[handle, data_ptr, len]`.
    ///
    /// Returns [`InlineKeyedResult`] in `RAX:RDX`.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI entry point; caller guarantees valid pointer.
    pub extern "C" fn jit_runtime_inline_load_keyed_smi_ic_r15(
        obj_i64: i64,
        smi_key: i64,
        rt_ptrs_cell: i64,
        ic_slots: *mut [i64; 4],
    ) -> InlineKeyedResult {
        // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and the
        // TLS slot lives for the thread's entire lifetime.
        let ptrs = unsafe { &*(rt_ptrs_cell as *const Cell<RtPtrs>) }.get();

        // Combined IC fill + element load: resolve the heap handle once
        // and reuse the array reference for both the IC update and the
        // element access, avoiding a redundant second heap resolution.
        if is_heap_handle(obj_i64) && smi_key >= 0 && ptrs.is_cached() {
            let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;
            // SAFETY: cached pointers valid for thread lifetime.
            let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
            if let Some(JsValue::Array(arr)) = heap.get(obj_idx) {
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let data = unsafe { &*arr.as_ptr() };
                // Fill IC for subsequent fast-path hits.
                unsafe {
                    (*ic_slots)[0] = obj_i64;
                    (*ic_slots)[1] = data.as_ptr() as i64;
                    (*ic_slots)[2] = data.len() as i64;
                    (*ic_slots)[3] = arr.as_ptr() as *const Vec<JsValue> as i64;
                }
                // Element load — reuse the already-resolved `data` ref.
                let key = smi_key as usize;
                if key < data.len() {
                    // SAFETY: bounds verified above.
                    let v = unsafe { data.get_unchecked(key) };
                    return if let Some(encoded) = encode_array_element(v) {
                        InlineKeyedResult {
                            value: encoded,
                            hit: 1,
                        }
                    } else {
                        InlineKeyedResult::from_generic(lda_keyed_property_with_ptrs(
                            obj_i64, smi_key, ptrs,
                        ))
                    };
                }
                return InlineKeyedResult {
                    value: JIT_UNDEFINED,
                    hit: 1,
                };
            }
        }

        // Fallback for non-array receiver or uncached pointers.
        inline_load_keyed_with_ptrs(obj_i64, smi_key, ptrs)
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
        sta_named_property_inner(obj_i64, name_idx, value_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_STA_NAMED);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_sta_named_property`].
    ///
    /// Uses cached TLS pointers to borrow the target object directly
    /// from the heap (avoiding an Rc clone) and tries the template-fill
    /// fast path before falling back to a full `insert`.
    fn sta_named_property_inner(obj_i64: i64, name_idx: u32, value_i64: i64) -> Option<i64> {
        // Bump the value-write epoch so proto-IC entries caching
        // own-property values are invalidated.
        bump_value_write_epoch();

        if !is_heap_handle(obj_i64) {
            return None;
        }
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;

        // Fast path: borrow directly from heap via cached ptrs.
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
            match heap.get(obj_idx) {
                Some(JsValue::PlainObject(map_rc)) => {
                    let prop_name = get_rt_string_constant_ref(name_idx)?;
                    // Inline Smi decode (common for numeric literals and
                    // loop counters) to avoid heap-handle lookups.
                    let value = if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
                        JsValue::Smi(value_i64 as i32)
                    } else {
                        jit_i64_to_jsvalue(value_i64)
                    };
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let map = unsafe { &mut *map_rc.as_ptr() };
                    match map.try_template_fill(prop_name, value) {
                        Ok(_) => return Some(value_i64),
                        Err(value) => {
                            map.insert(prop_name.to_string(), value);
                            return Some(value_i64);
                        }
                    }
                }
                Some(JsValue::Function(ba)) => {
                    let prop_name = get_rt_string_constant_ref(name_idx)?;
                    let value = jit_i64_to_jsvalue(value_i64);
                    crate::interpreter::fn_props_set(ba, prop_name.to_string(), value);
                    return Some(value_i64);
                }
                _ => return None,
            }
        }

        // Slow path: clone-based fallback.
        let obj = jit_i64_to_jsvalue(obj_i64);
        let value = jit_i64_to_jsvalue(value_i64);

        match &obj {
            JsValue::PlainObject(map_rc) => {
                let prop_name = get_rt_string_constant_ref(name_idx)?;
                map_rc.borrow_mut().insert(prop_name.to_string(), value);
                Some(value_i64)
            }
            JsValue::Function(ba) => {
                let prop_name = get_rt_string_constant_ref(name_idx)?;
                crate::interpreter::fn_props_set(ba, prop_name.to_string(), value);
                Some(value_i64)
            }
            _ => None,
        }
    }

    // ── Fast CreateObjectLiteral stub ───────────────────────────────────

    /// Direct runtime stub for `CreateObjectLiteral` that bypasses the
    /// generic trampoline dispatch.
    ///
    /// Uses the same template-caching mechanism as the trampoline handler
    /// but avoids the 5-argument setup and opcode-match overhead on every
    /// call.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`feedback_slot`) – feedback vector slot index (−1 = none).
    /// * `RSI` (`capacity`)      – minimum property capacity / flags.
    ///
    /// Returns a JIT i64 heap handle for the new `PlainObject`, or
    /// [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_fast_create_object_literal(
        feedback_slot: i64,
        capacity: i64,
    ) -> i64 {
        fast_create_object_literal_inner(feedback_slot, capacity).unwrap_or_else(|| {
            track_stub_deopt(STUB_FAST_CREATE_OBJ);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_fast_create_object_literal`].
    ///
    /// The hot path (cached template hit) avoids redundant TLS lookups by
    /// reusing the [`RtPtrs`] fetched at entry for the subsequent heap
    /// allocation.  Combined with the pooled `Rc<RefCell<PropertyMap>>`
    /// recycling this achieves **zero** per-object heap allocations on
    /// repeated calls: one TLS read, one template-cache probe, one pool
    /// pop + in-place reinitialise, and one heap-vec push.
    fn fast_create_object_literal_inner(feedback_slot: i64, capacity: i64) -> Option<i64> {
        let capacity = capacity.max(4) as usize;

        if feedback_slot >= 0 {
            let slot = feedback_slot as u32;

            // Use as_ptr() to borrow in-place, avoiding a full struct copy.
            // SAFETY: single-threaded JIT; Cell is not mutated during call.
            let ptrs: &RtPtrs = unsafe { &*RT_PTRS.with(|p| p.as_ptr()) };
            let ba = if ptrs.is_cached() {
                // SAFETY: cached pointer valid for thread lifetime.
                unsafe { &*ptrs.bytecode }.get()
            } else {
                RT_BYTECODE.with(|b| b.get())
            };
            if !ba.is_null() {
                // SAFETY: pointer is valid and points to a live
                // BytecodeArray.
                let ba_ref = unsafe { &*ba };

                // Ultra-fast path: single-entry IC hit via cached
                // pointer — no extra TLS lookup.
                let ic = ptrs.get_object_ic();
                if ic.slot == slot && ic.ba == ba {
                    // SAFETY: template pointer is valid — it points to
                    // Box heap memory owned by the BytecodeArray's
                    // template map which is alive for the entire JIT
                    // execution.  Cached entries are never removed or
                    // replaced.
                    let template = unsafe { &*ic.template };
                    let rc = acquire_object_rc_from_template_cached(template, ptrs.object_rc_pool);
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // Fast path: template cached in BA — populate the IC.
                if let Some(rc) = ba_ref.clone_object_literal_template_pooled(slot) {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // Second execution: promote pending → cached, then
                // populate the IC for future hits.
                if let Some(rc) = ba_ref.promote_object_literal_template_pooled(slot) {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // First execution: create fresh and register as
                // pending for future promotion.
                let map = PropertyMap::with_capacity(capacity);
                let rc = Rc::new(RefCell::new(map));
                ba_ref.set_object_literal_pending(slot, Rc::clone(&rc));
                let obj = JsValue::PlainObject(rc);
                return Some(alloc_heap_handle_no_dedup(obj, ptrs));
            }
        }

        // Fallback: no feedback slot or no BA pointer.
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::with_capacity(capacity))));
        Some(alloc_heap_handle(obj))
    }

    // ── Fused CreateObjectLiteral + StoreNamedGeneric stub ──────────────

    /// Create an object literal and fill up to 5 own properties in a
    /// single call.
    ///
    /// This is the runtime half of the `CreateObjectLiteralWithProperties`
    /// Maglev IR node produced by the optimizer's store-fusion pass.
    /// Performing both creation and property initialisation in one stub
    /// call eliminates N separate `StoreNamedGeneric` calls (and their
    /// per-call TLS lookups).
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`feedback_slot`) – feedback vector slot index (−1 = none).
    /// * `RSI` (`names_packed`)  – up to 5 constant-pool name indices
    ///   packed as 12-bit values: `name0 | (name1 << 12) | … |
    ///   (name4 << 48)`, with the property count in bits 60–63.
    /// * `RDX` (`val0`) – first property value (JIT i64).
    /// * `RCX` (`val1`) – second property value (JIT i64), or 0 if fewer.
    /// * `R8`  (`val2`) – third property value (JIT i64), or 0 if fewer.
    /// * `R9`  (`val3`) – fourth property value (JIT i64), or 0 if fewer.
    /// * Stack (`val4`) – fifth property value (JIT i64), or 0 if fewer.
    ///
    /// Returns a JIT i64 heap handle for the new `PlainObject`, or
    /// [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_create_object_with_props(
        feedback_slot: i64,
        names_packed: i64,
        val0: i64,
        val1: i64,
        val2: i64,
        val3: i64,
        val4: i64,
    ) -> i64 {
        create_object_with_props_inner(feedback_slot, names_packed, val0, val1, val2, val3, val4)
            .unwrap_or_else(|| {
                track_stub_deopt(STUB_CREATE_OBJ_WITH_PROPS);
                JIT_DEOPT
            })
    }

    /// Inner implementation for [`jit_runtime_create_object_with_props`].
    ///
    /// The hot path (template cache hit) converts the register-encoded
    /// values to a `Vec<JsValue>` and hands them to
    /// [`BytecodeArray::clone_object_literal_template_with_values`] which
    /// constructs the [`PropertyMap`] with values pre-filled — avoiding
    /// the per-property name lookups and `try_template_fill` comparisons
    /// that the first-execution path pays.
    fn create_object_with_props_inner(
        feedback_slot: i64,
        names_packed: i64,
        val0: i64,
        val1: i64,
        val2: i64,
        val3: i64,
        val4: i64,
    ) -> Option<i64> {
        let names_packed_u64 = names_packed as u64;
        let count = ((names_packed_u64 >> 60) & 0xF) as usize;
        let vals = [val0, val1, val2, val3, val4];

        // Use as_ptr() to borrow the RtPtrs in place, avoiding a 96-byte
        // struct copy on the hot path.
        // SAFETY: single-threaded JIT; the Cell is not mutated during this call.
        let ptrs: &RtPtrs = unsafe { &*RT_PTRS.with(|p| p.as_ptr()) };

        if feedback_slot >= 0 {
            let slot = feedback_slot as u32;
            let ba = if ptrs.is_cached() {
                // SAFETY: cached pointer valid for thread lifetime; alignment
                // verified by is_cached().
                unsafe { &*ptrs.bytecode }.get()
            } else {
                RT_BYTECODE.with(|b| b.get())
            };
            if !ba.is_null() {
                // SAFETY: pointer is valid and points to a live BytecodeArray.
                let ba_ref = unsafe { &*ba };

                // Decode to a stack array to avoid a Vec allocation.
                let js_values = decode_jit_values_array(count, &vals);

                // ── Ultra-fast path: IC hit via cached pointer ──
                let ic = ptrs.get_object_ic();
                if ic.slot == slot && ic.ba == ba {
                    // SAFETY: template pointer valid — see IC safety
                    // comment on ObjectLiteralIcEntry.
                    let template = unsafe { &*ic.template };
                    let rc = acquire_object_rc_from_template_with_values_cached(
                        template,
                        &js_values[..count],
                        ptrs.object_rc_pool,
                    );
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── Hot path: template cached in BA — populate IC ──
                if let Some(rc) = ba_ref
                    .clone_object_literal_template_with_values_pooled(slot, &js_values[..count])
                {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── Promote path (second execution) ──
                if let Some(rc) = ba_ref
                    .promote_object_literal_template_with_values_pooled(slot, &js_values[..count])
                {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── First execution: create fresh and fill via property stores ──
                let names = unpack_names(names_packed_u64);
                let capacity = 4i64.max(count as i64) as usize;
                let map = PropertyMap::with_capacity(capacity);
                let rc = Rc::new(RefCell::new(map));
                ba_ref.set_object_literal_pending(slot, Rc::clone(&rc));
                {
                    // SAFETY: single-threaded JIT; no concurrent borrows.
                    let map = unsafe { &mut *rc.as_ptr() };
                    for i in 0..count {
                        let prop_name = get_rt_string_constant_ref(names[i])?;
                        let value = decode_one_jit_value(vals[i]);
                        match map.try_template_fill(prop_name, value) {
                            Ok(_) => {}
                            Err(value) => {
                                map.insert(prop_name.to_string(), value);
                            }
                        }
                    }
                }
                let obj = JsValue::PlainObject(rc);
                return Some(alloc_heap_handle_no_dedup(obj, ptrs));
            }
        }

        // Fallback: no feedback slot or no BA pointer.
        let names = unpack_names(names_packed_u64);
        let capacity = 4i64.max(count as i64) as usize;
        let map_rc = Rc::new(RefCell::new(PropertyMap::with_capacity(capacity)));
        {
            // SAFETY: single-threaded JIT; no concurrent borrows.
            let map = unsafe { &mut *map_rc.as_ptr() };
            for i in 0..count {
                let prop_name = get_rt_string_constant_ref(names[i])?;
                let value = decode_one_jit_value(vals[i]);
                match map.try_template_fill(prop_name, value) {
                    Ok(_) => {}
                    Err(value) => {
                        map.insert(prop_name.to_string(), value);
                    }
                }
            }
        }
        let obj = JsValue::PlainObject(map_rc);
        Some(alloc_heap_handle_no_dedup(obj, ptrs))
    }

    /// Fast-path object creation that accepts the pre-cached `RT_PTRS`
    /// cell pointer (R15), eliminating the per-call TLS lookup for
    /// [`RT_PTRS`].
    ///
    /// Called by Maglev-generated code when `needs_r15` is true.  Falls
    /// back to [`jit_runtime_create_object_with_props`] if the cached
    /// pointers are stale.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`feedback_slot`) – feedback vector slot index.
    /// * `RSI` (`names_packed`)  – packed name indices + count.
    /// * `RDX` (`val0`) – first property value.
    /// * `RCX` (`val1`) – second property value, or 0.
    /// * `R8`  (`val2`) – third property value, or 0.
    /// * `R9`  (`val3`) – fourth property value, or 0.
    /// * Stack+0 (`val4`) – fifth property value, or 0.
    /// * Stack+8 (`rt_ptrs_cell`) – pointer to `Cell<RtPtrs>` TLS slot.
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    pub extern "C" fn jit_runtime_create_object_with_props_r15(
        feedback_slot: i64,
        names_packed: i64,
        val0: i64,
        val1: i64,
        val2: i64,
        val3: i64,
        val4: i64,
        rt_ptrs_cell: i64,
    ) -> i64 {
        create_object_with_props_inner_r15(
            feedback_slot,
            names_packed,
            val0,
            val1,
            val2,
            val3,
            val4,
            rt_ptrs_cell,
        )
        .unwrap_or_else(|| {
            track_stub_deopt(STUB_CREATE_OBJ_WITH_PROPS);
            JIT_DEOPT
        })
    }

    /// Inner implementation for [`jit_runtime_create_object_with_props_r15`].
    ///
    /// Identical to [`create_object_with_props_inner`] except that it
    /// resolves [`RtPtrs`] from the passed cell pointer instead of the
    /// TLS lookup.
    #[allow(clippy::too_many_arguments)]
    fn create_object_with_props_inner_r15(
        feedback_slot: i64,
        names_packed: i64,
        val0: i64,
        val1: i64,
        val2: i64,
        val3: i64,
        val4: i64,
        rt_ptrs_cell: i64,
    ) -> Option<i64> {
        let names_packed_u64 = names_packed as u64;
        let count = ((names_packed_u64 >> 60) & 0xF) as usize;
        let vals = [val0, val1, val2, val3, val4];

        // Resolve RtPtrs from the cached cell pointer, falling back to
        // TLS if the pointer is null or stale.  Use as_ptr() to obtain a
        // reference instead of Cell::get() which copies the full 96-byte
        // struct — on the IC-hit hot path only a few fields are read.
        let ptrs: &RtPtrs = if rt_ptrs_cell != 0 {
            // SAFETY: rt_ptrs_cell was obtained from RT_PTRS.with() and
            // the TLS slot lives for the thread's entire lifetime.
            // No mutation of the Cell occurs during this function call.
            unsafe { &*(&*(rt_ptrs_cell as *const Cell<RtPtrs>)).as_ptr() }
        } else {
            // SAFETY: single-threaded JIT; Cell is not mutated during call.
            unsafe { &*RT_PTRS.with(|p| p.as_ptr()) }
        };

        if feedback_slot >= 0 {
            let slot = feedback_slot as u32;
            let ba = if ptrs.is_cached() {
                unsafe { &*ptrs.bytecode }.get()
            } else {
                RT_BYTECODE.with(|b| b.get())
            };
            if !ba.is_null() {
                let ba_ref = unsafe { &*ba };
                let js_values = decode_jit_values_array(count, &vals);

                // ── Ultra-fast path: IC hit ──
                let ic = ptrs.get_object_ic();
                if ic.slot == slot && ic.ba == ba {
                    let template = unsafe { &*ic.template };
                    let rc = acquire_object_rc_from_template_with_values_cached(
                        template,
                        &js_values[..count],
                        ptrs.object_rc_pool,
                    );
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── Hot path: template cached in BA — populate IC ──
                if let Some(rc) = ba_ref
                    .clone_object_literal_template_with_values_pooled(slot, &js_values[..count])
                {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── Promote path ──
                if let Some(rc) = ba_ref
                    .promote_object_literal_template_with_values_pooled(slot, &js_values[..count])
                {
                    let tmpl_ptr = ba_ref.get_cached_template_ptr(slot);
                    if !tmpl_ptr.is_null() {
                        ptrs.set_object_ic(ObjectLiteralIcEntry {
                            slot,
                            ba,
                            template: tmpl_ptr,
                        });
                    }
                    let obj = JsValue::PlainObject(rc);
                    return Some(alloc_heap_handle_no_dedup(obj, ptrs));
                }

                // ── First execution ──
                let names = unpack_names(names_packed_u64);
                let capacity = 4i64.max(count as i64) as usize;
                let map = PropertyMap::with_capacity(capacity);
                let rc = Rc::new(RefCell::new(map));
                ba_ref.set_object_literal_pending(slot, Rc::clone(&rc));
                {
                    let map = unsafe { &mut *rc.as_ptr() };
                    for i in 0..count {
                        let prop_name = get_rt_string_constant_ref(names[i])?;
                        let value = decode_one_jit_value(vals[i]);
                        match map.try_template_fill(prop_name, value) {
                            Ok(_) => {}
                            Err(value) => {
                                map.insert(prop_name.to_string(), value);
                            }
                        }
                    }
                }
                let obj = JsValue::PlainObject(rc);
                return Some(alloc_heap_handle_no_dedup(obj, ptrs));
            }
        }

        // Fallback.
        let names = unpack_names(names_packed_u64);
        let capacity = 4i64.max(count as i64) as usize;
        let map_rc = Rc::new(RefCell::new(PropertyMap::with_capacity(capacity)));
        {
            let map = unsafe { &mut *map_rc.as_ptr() };
            for i in 0..count {
                let prop_name = get_rt_string_constant_ref(names[i])?;
                let value = decode_one_jit_value(vals[i]);
                match map.try_template_fill(prop_name, value) {
                    Ok(_) => {}
                    Err(value) => {
                        map.insert(prop_name.to_string(), value);
                    }
                }
            }
        }
        let obj = JsValue::PlainObject(map_rc);
        Some(alloc_heap_handle_no_dedup(obj, ptrs))
    }

    /// Decode a single JIT `i64` register value into a [`JsValue`].
    #[inline(always)]
    fn decode_one_jit_value(v: i64) -> JsValue {
        if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
            JsValue::Smi(v as i32)
        } else {
            jit_i64_to_jsvalue(v)
        }
    }

    /// Decode up to 5 JIT `i64` register values into a stack-allocated
    /// array, avoiding a `Vec` heap allocation on the fused hot path.
    #[inline(always)]
    fn decode_jit_values_array(count: usize, vals: &[i64; 5]) -> [JsValue; 5] {
        let mut out = [
            JsValue::Undefined,
            JsValue::Undefined,
            JsValue::Undefined,
            JsValue::Undefined,
            JsValue::Undefined,
        ];
        for i in 0..count.min(5) {
            out[i] = decode_one_jit_value(vals[i]);
        }
        out
    }

    /// Unpack up to 5 constant-pool name indices from the packed `u64`
    /// (12-bit encoding).
    #[inline]
    fn unpack_names(packed: u64) -> [u32; 5] {
        [
            (packed & 0xFFF) as u32,
            ((packed >> 12) & 0xFFF) as u32,
            ((packed >> 24) & 0xFFF) as u32,
            ((packed >> 36) & 0xFFF) as u32,
            ((packed >> 48) & 0xFFF) as u32,
        ]
    }

    // ── Fast StaNamedOwnProperty stub ───────────────────────────────────

    /// Specialized runtime stub for `StaNamedOwnProperty` /
    /// `DefineNamedOwnProperty` on freshly created plain objects.
    ///
    /// Compared to [`jit_runtime_sta_named_property`] this stub:
    /// * skips the `Function` receiver check (own-property stores target
    ///   plain objects),
    /// * always attempts the template-fill fast path first.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`obj_i64`)   – JIT i64 heap handle of the receiver.
    /// * `RSI` (`name_idx`)  – constant-pool index of the property name.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    ///
    /// Returns `value_i64` on success, or [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_fast_sta_named_own_property(
        obj_i64: i64,
        name_idx: i64,
        value_i64: i64,
    ) -> i64 {
        fast_sta_named_own_property_inner(obj_i64, name_idx as u32, value_i64).unwrap_or_else(
            || {
                track_stub_deopt(STUB_STA_NAMED_OWN);
                JIT_DEOPT
            },
        )
    }

    /// Inner implementation for [`jit_runtime_fast_sta_named_own_property`].
    fn fast_sta_named_own_property_inner(
        obj_i64: i64,
        name_idx: u32,
        value_i64: i64,
    ) -> Option<i64> {
        if !is_heap_handle(obj_i64) {
            return None;
        }
        let obj_idx = (obj_i64 - JIT_HEAP_TAG) as usize;

        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
            if let Some(JsValue::PlainObject(map_rc)) = heap.get(obj_idx) {
                let prop_name = get_rt_string_constant_ref(name_idx)?;
                // Inline Smi decode (common for numeric literals and
                // loop counters) to avoid heap-handle lookups.
                let value = if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
                    JsValue::Smi(value_i64 as i32)
                } else {
                    jit_i64_to_jsvalue(value_i64)
                };
                // SAFETY: single-threaded JIT; no concurrent borrows.
                let map = unsafe { &mut *map_rc.as_ptr() };
                match map.try_template_fill(prop_name, value) {
                    Ok(_) => return Some(value_i64),
                    Err(value) => {
                        map.insert(prop_name.to_string(), value);
                        return Some(value_i64);
                    }
                }
            }
        }

        // Slow path: clone-based fallback.
        let obj = jit_i64_to_jsvalue(obj_i64);
        let value = jit_i64_to_jsvalue(value_i64);

        match &obj {
            JsValue::PlainObject(map_rc) => {
                let prop_name = get_rt_string_constant_ref(name_idx)?;
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
        call_property0_inner(callee_i64, receiver_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_PROP0);
            JIT_DEOPT
        })
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

            // Phase 1: Extract callee info from the heap in a scoped
            // borrow.  Dropped before any native function invocation or
            // jsvalue_to_jit_i64 call that could allocate heap handles.
            enum CalleeInfo {
                FastArrayMethod {
                    method_name: Rc<str>,
                    arr: Option<Rc<RefCell<Vec<JsValue>>>>,
                },
                NativeCall(Rc<dyn Fn(Vec<JsValue>) -> Result<JsValue, StatorError>>),
                JsFunction(*const BytecodeArray),
                Other,
            }

            let callee_info = {
                // SAFETY: scoped immutable borrow of the heap.
                let heap = unsafe { &*heap_ref.as_ptr() };
                if let Some(JsValue::PlainObject(map_rc)) = heap.get(callee_idx) {
                    let map = unsafe { &*map_rc.as_ptr() };
                    if let Some(JsValue::String(method_name)) =
                        map.get("\0stator.fast_array_method")
                    {
                        let arr = if is_heap_handle(receiver_i64) {
                            let recv_idx = (receiver_i64 - JIT_HEAP_TAG) as usize;
                            if let Some(JsValue::Array(arr)) = heap.get(recv_idx) {
                                Some(Rc::clone(arr))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        CalleeInfo::FastArrayMethod {
                            method_name: Rc::clone(method_name),
                            arr,
                        }
                    } else if let Some(JsValue::NativeFunction(nf)) = map.get("__call__") {
                        CalleeInfo::NativeCall(Rc::clone(nf))
                    } else {
                        CalleeInfo::Other
                    }
                } else if let Some(JsValue::NativeFunction(nf)) = heap.get(callee_idx) {
                    CalleeInfo::NativeCall(Rc::clone(nf))
                } else if let Some(JsValue::Function(ba)) = heap.get(callee_idx) {
                    CalleeInfo::JsFunction(Rc::as_ptr(ba))
                } else {
                    CalleeInfo::Other
                }
            }; // heap dropped here

            match callee_info {
                CalleeInfo::FastArrayMethod { method_name, arr } => {
                    if let Some(arr) = arr {
                        match method_name.as_ref() {
                            "pop" => {
                                let val = arr.borrow_mut().pop().unwrap_or(JsValue::Undefined);
                                return Some(jsvalue_to_jit_i64(val));
                            }
                            "shift" => {
                                let mut items = arr.borrow_mut();
                                let val = if items.is_empty() {
                                    JsValue::Undefined
                                } else {
                                    items.remove(0)
                                };
                                drop(items);
                                return Some(jsvalue_to_jit_i64(val));
                            }
                            _ => {}
                        }
                    }
                    return None;
                }
                CalleeInfo::NativeCall(nf) => {
                    let receiver = jit_i64_to_jsvalue(receiver_i64);
                    return match nf(vec![receiver]) {
                        Ok(val) => Some(jsvalue_to_jit_i64(val)),
                        Err(_) => None,
                    };
                }
                #[cfg(all(target_arch = "x86_64", unix))]
                CalleeInfo::JsFunction(ba_ptr) => {
                    // User-defined JS function (e.g. obj.getX()).
                    // Execute via Maglev if available, passing receiver
                    // as args[0] (the `this` binding).
                    // SAFETY: ba_ptr from Rc::as_ptr; heap entry outlives call.
                    let ba: &BytecodeArray = unsafe { &*ba_ptr };
                    let bc_ref = unsafe { &*ptrs.bytecode };
                    let saved_ba = bc_ref.get();

                    // ── Maglev fast path ──
                    {
                        use crate::compiler::maglev::codegen::CachedMaglevCode;
                        let maglev_cache = ba.maglev_executable_cache();
                        let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                        if maglev_ref.is_none() {
                            let jit_cache = ba.maglev_jit_cache_arc();
                            let cached_data = jit_cache.lock().ok().and_then(|guard| {
                                guard
                                    .as_ref()
                                    .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
                            });
                            if let Some((code, register_file_slots)) = cached_data {
                                let exec =
                                    unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                                *maglev_cache.borrow_mut() = exec;
                            }
                        }
                        let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                        if let Some(maglev_exec) = maglev_ref.as_ref() {
                            let ctx_ref = unsafe { &*ptrs.context };
                            let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                            let callee_ctx_raw = ba.closure_context();
                            let callee_ctx_ptr =
                                callee_ctx_raw.map(Rc::as_ptr).unwrap_or(std::ptr::null());
                            let current_ctx_ptr = unsafe {
                                (*ctx_ref.as_ptr())
                                    .as_ref()
                                    .map(Rc::as_ptr)
                                    .unwrap_or(std::ptr::null())
                            };
                            let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);
                            let saved_ctx = if same_context {
                                None
                            } else {
                                let callee_ctx = callee_ctx_raw.cloned();
                                Some(unsafe {
                                    std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx)
                                })
                            };

                            bc_ref.set(ba_ptr);
                            let ctx_raw = callee_ctx_ptr as i64;
                            // Pass receiver as args[0] (the `this` binding).
                            let jit_result =
                                unsafe { maglev_exec.execute(&[receiver_i64], ctx_raw) };
                            bc_ref.set(saved_ba);

                            if jit_result != JIT_DEOPT && jit_result < JIT_HEAP_TAG {
                                unsafe { (*heap_ref.as_ptr()).truncate(heap_base) };
                                if let Some(ctx) = saved_ctx {
                                    unsafe { *ctx_ref.as_ptr() = ctx };
                                }
                                return Some(jit_result);
                            }

                            let result_val = if jit_result == JIT_DEOPT {
                                None
                            } else {
                                jit_to_jsvalue_ext(jit_result)
                            };
                            unsafe { (*heap_ref.as_ptr()).truncate(heap_base) };
                            if let Some(ctx) = saved_ctx {
                                unsafe { *ctx_ref.as_ptr() = ctx };
                            }
                            return result_val.map(jsvalue_to_jit_i64);
                        }
                    }

                    // No Maglev code — deopt to interpreter.
                    return None;
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                CalleeInfo::JsFunction(_) => return None,
                CalleeInfo::Other => return None,
            }
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
        call_property1_inner(callee_i64, receiver_i64, arg0_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_PROP1);
            JIT_DEOPT
        })
    }

    /// Specialized `Array.prototype.push` runtime for `CallArrayPush`.
    ///
    /// Skips callee resolution entirely — the graph builder already
    /// verified the callee is `push` at compile time.  Only needs
    /// the receiver (array) and the argument.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`receiver_i64`) – JIT i64 encoding of the receiver array.
    /// * `RSI` (`arg0_i64`) – JIT i64 encoding of the value to push.
    ///
    /// Returns the new array length as `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_array_push(receiver_i64: i64, arg0_i64: i64) -> i64 {
        call_array_push_inner(receiver_i64, arg0_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CALL_PROP1);
            JIT_DEOPT
        })
    }

    /// Array push that also fills the Maglev array IC on success.
    ///
    /// `ic_ptr` points to `[handle, data_ptr, len, vec_ptr]` (4 × i64)
    /// on the JIT stack frame.  After a successful push, populates all
    /// four IC fields so that the next inline push fast-path hits.
    ///
    /// Returns the new array length as `i64`, or [`JIT_DEOPT`].
    #[unsafe(no_mangle)]
    pub extern "C" fn jit_runtime_array_push_fill_ic(
        receiver_i64: i64,
        arg0_i64: i64,
        ic_ptr: i64,
    ) -> i64 {
        track_stub_call(STUB_FAST_ARRAY_PUSH);
        match call_array_push_inner(receiver_i64, arg0_i64) {
            Some(new_len) => {
                // Fill the array IC so subsequent inline pushes hit.
                if ic_ptr != 0 {
                    let cache = RT_PUSH_CACHE.with(|c| c.get());
                    if cache.receiver == receiver_i64 && !cache.vec_ptr.is_null() {
                        // SAFETY: ic_ptr is a valid stack-frame address
                        // from `[RBP + array_ic_base]`.  vec_ptr is valid
                        // for the duration of JIT execution.
                        unsafe {
                            let ic = ic_ptr as *mut i64;
                            let vec = &*cache.vec_ptr;
                            // IC layout: [handle(+0), data_ptr(+8), len(+16), vec_ptr(+24)]
                            *ic = receiver_i64;
                            *ic.add(1) = vec.as_ptr() as i64;
                            *ic.add(2) = vec.len() as i64;
                            *ic.add(3) = cache.vec_ptr as i64;
                        }
                    }
                }
                new_len
            }
            None => {
                track_stub_deopt(STUB_CALL_PROP1);
                JIT_DEOPT
            }
        }
    }

    /// IC-hit fast path for `Array.prototype.push` in `CallProperty1`.
    ///
    /// Called when [`RT_CALL_PROP1_IC`] identifies the callee as a
    /// previously-seen array-push method.  Performs only a receiver
    /// type check, value decode, and the actual `Vec::push` — skipping
    /// the `PlainObject` deref and `PropertyMap` probe entirely.
    #[inline(always)]
    fn call_property1_ic_push(receiver_i64: i64, arg0_i64: i64) -> Option<i64> {
        if !is_heap_handle(receiver_i64) {
            return None;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return None;
        }
        let recv_idx = (receiver_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
        let heap_ref = unsafe { &*ptrs.heap };
        // SAFETY: single-threaded JIT; no concurrent mutable borrows.
        let heap = unsafe { &*heap_ref.as_ptr() };
        match heap.get(recv_idx)? {
            JsValue::Array(arr) => {
                let arg0 = if is_heap_handle(arg0_i64) {
                    let a_idx = (arg0_i64 - JIT_HEAP_TAG) as usize;
                    heap.get(a_idx).cloned().unwrap_or(JsValue::Undefined)
                } else if arg0_i64 >= i32::MIN as i64 && arg0_i64 <= i32::MAX as i64 {
                    JsValue::Smi(arg0_i64 as i32)
                } else {
                    super::jit_to_jsvalue(arg0_i64).unwrap_or(JsValue::Undefined)
                };
                // SAFETY: single-threaded JIT; no concurrent borrows of
                // this array during push.
                let items = unsafe { &mut *arr.as_ptr() };
                items.push(arg0);
                Some(items.len() as i64)
            }
            _ => None,
        }
    }

    /// Direct array push — no callee resolution, no IC.
    ///
    /// Used by `jit_runtime_array_push` when the graph builder has
    /// already proven the callee is `Array.prototype.push`.
    #[inline(always)]
    fn call_array_push_inner(receiver_i64: i64, arg0_i64: i64) -> Option<i64> {
        // Decode the argument (Smi fast path first — common in push loops).
        let arg0 = if arg0_i64 >= i32::MIN as i64 && arg0_i64 <= i32::MAX as i64 {
            JsValue::Smi(arg0_i64 as i32)
        } else if is_heap_handle(arg0_i64) {
            let ptrs = RT_PTRS.with(|p| p.get());
            if !ptrs.is_cached() {
                return None;
            }
            let heap_ref = unsafe { &*ptrs.heap };
            let heap = unsafe { &*heap_ref.as_ptr() };
            let a_idx = (arg0_i64 - JIT_HEAP_TAG) as usize;
            heap.get(a_idx).cloned().unwrap_or(JsValue::Undefined)
        } else {
            super::jit_to_jsvalue(arg0_i64).unwrap_or(JsValue::Undefined)
        };

        // ── Cache fast path: reuse the Vec pointer from previous push ──
        let cache = RT_PUSH_CACHE.with(|c| c.get());
        if cache.receiver == receiver_i64 && !cache.vec_ptr.is_null() {
            // SAFETY: vec_ptr was set from Rc<RefCell<Vec<JsValue>>>::as_ptr()
            // in the slow path below.  The Rc lives in the heap which is
            // alive during JIT execution (single-threaded, no GC).
            let items = unsafe { &mut *cache.vec_ptr };
            items.push(arg0);
            return Some(items.len() as i64);
        }

        // ── Slow path: resolve receiver and populate cache ─────────────
        if !is_heap_handle(receiver_i64) {
            return None;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return None;
        }
        let recv_idx = (receiver_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers set by cache_rt_ptrs; valid for thread lifetime.
        let heap_ref = unsafe { &*ptrs.heap };
        // SAFETY: single-threaded JIT; no concurrent mutable borrows.
        let heap = unsafe { &*heap_ref.as_ptr() };
        match heap.get(recv_idx)? {
            JsValue::Array(arr) => {
                let vec_ptr = arr.as_ptr();
                // Populate cache for subsequent iterations.
                RT_PUSH_CACHE.with(|c| {
                    c.set(PushVecCache {
                        receiver: receiver_i64,
                        vec_ptr,
                    })
                });
                // SAFETY: single-threaded JIT; no concurrent borrows of
                // this array during push.
                let items = unsafe { &mut *vec_ptr };
                // Pre-allocate when pushing into an empty array.  Code that
                // enters a push loop will typically push many elements;
                // reserving up front avoids ~10 reallocations + memcpys for
                // a 1000-element loop (saves ~24 KB of redundant copying).
                if items.is_empty() {
                    items.reserve(1024);
                }
                items.push(arg0);
                Some(items.len() as i64)
            }
            _ => None,
        }
    }

    /// Inner implementation for [`jit_runtime_call_property1`].
    ///
    /// Handles three callee shapes:
    /// 1. Fast array method (PlainObject with `\0stator.fast_array_method`)
    ///    — inlines `push`, `pop`, etc. directly.
    /// 2. `NativeFunction` — generic Rust-closure dispatch.
    /// 3. Anything else → DEOPT.
    fn call_property1_inner(callee_i64: i64, receiver_i64: i64, arg0_i64: i64) -> Option<i64> {
        // ── IC fast path: skip callee identification on cache hit ─────
        //
        // In tight `arr.push(v)` loops the callee heap handle is the
        // same on every iteration.  The IC lets us jump straight to the
        // push implementation without dereffing the PlainObject or
        // probing its PropertyMap for the "\0stator.fast_array_method"
        // marker — the two most expensive operations on the slow path.
        let ic = RT_CALL_PROP1_IC.with(|c| c.get());
        if ic.callee == callee_i64 && ic.tag == CallProp1IcEntry::TAG_ARRAY_PUSH {
            if let Some(result) = call_property1_ic_push(receiver_i64, arg0_i64) {
                return Some(result);
            }
            // Receiver was not an Array → invalidate IC, fall through.
            RT_CALL_PROP1_IC.with(|c| c.set(CallProp1IcEntry::EMPTY));
        }

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
                                    // Populate IC so subsequent iterations
                                    // skip the PropertyMap probe.
                                    RT_CALL_PROP1_IC.with(|c| {
                                        c.set(CallProp1IcEntry {
                                            callee: callee_i64,
                                            tag: CallProp1IcEntry::TAG_ARRAY_PUSH,
                                        })
                                    });
                                    // Inline value decode for common types.
                                    let arg0 = if is_heap_handle(arg0_i64) {
                                        let a_idx = (arg0_i64 - JIT_HEAP_TAG) as usize;
                                        heap.get(a_idx).cloned().unwrap_or(JsValue::Undefined)
                                    } else if arg0_i64 >= i32::MIN as i64
                                        && arg0_i64 <= i32::MAX as i64
                                    {
                                        JsValue::Smi(arg0_i64 as i32)
                                    } else {
                                        super::jit_to_jsvalue(arg0_i64)
                                            .unwrap_or(JsValue::Undefined)
                                    };
                                    // SAFETY: single-threaded JIT; no concurrent
                                    // borrows of this array during push.
                                    let items = unsafe { &mut *arr.as_ptr() };
                                    items.push(arg0);
                                    return Some(items.len() as i64);
                                }
                                "pop" => {
                                    // SAFETY: single-threaded JIT; no concurrent borrows.
                                    let val = unsafe { &mut *arr.as_ptr() }
                                        .pop()
                                        .unwrap_or(JsValue::Undefined);
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

            // User-defined JS function (e.g. Base.call(this) in constructors).
            #[cfg(all(target_arch = "x86_64", unix))]
            if let Some(JsValue::Function(ba_rc)) = heap.get(callee_idx) {
                let ba_ptr: *const BytecodeArray = Rc::as_ptr(ba_rc);
                // SAFETY: ba_ptr from Rc::as_ptr; heap entry outlives call.
                let ba: &BytecodeArray = unsafe { &*ba_ptr };
                let bc_ref = unsafe { &*ptrs.bytecode };
                let saved_ba = bc_ref.get();

                // ── Maglev fast path ──
                {
                    use crate::compiler::maglev::codegen::CachedMaglevCode;
                    let maglev_cache = ba.maglev_executable_cache();
                    let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                    if maglev_ref.is_none() {
                        let jit_cache = ba.maglev_jit_cache_arc();
                        let cached_data = jit_cache.lock().ok().and_then(|guard| {
                            guard
                                .as_ref()
                                .map(|c| (c.as_bytes().to_vec(), c.register_file_slots))
                        });
                        if let Some((code, register_file_slots)) = cached_data {
                            let exec = unsafe { CachedMaglevCode::new(&code, register_file_slots) };
                            *maglev_cache.borrow_mut() = exec;
                        }
                    }
                    let maglev_ref = unsafe { &*maglev_cache.as_ptr() };
                    if let Some(maglev_exec) = maglev_ref.as_ref() {
                        let ctx_ref = unsafe { &*ptrs.context };
                        let heap_base = unsafe { (*heap_ref.as_ptr()).len() };

                        let callee_ctx_raw = ba.closure_context();
                        let callee_ctx_ptr =
                            callee_ctx_raw.map(Rc::as_ptr).unwrap_or(std::ptr::null());
                        let current_ctx_ptr = unsafe {
                            (*ctx_ref.as_ptr())
                                .as_ref()
                                .map(Rc::as_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let same_context = std::ptr::eq(callee_ctx_ptr, current_ctx_ptr);
                        let saved_ctx = if same_context {
                            None
                        } else {
                            let callee_ctx = callee_ctx_raw.cloned();
                            Some(unsafe { std::mem::replace(&mut *ctx_ref.as_ptr(), callee_ctx) })
                        };

                        bc_ref.set(ba_ptr);
                        let ctx_raw = callee_ctx_ptr as i64;
                        // Pass receiver as args[0] and arg0 as args[1].
                        let jit_result =
                            unsafe { maglev_exec.execute(&[receiver_i64, arg0_i64], ctx_raw) };
                        bc_ref.set(saved_ba);

                        if jit_result != JIT_DEOPT && jit_result < JIT_HEAP_TAG {
                            unsafe { (*heap_ref.as_ptr()).truncate(heap_base) };
                            if let Some(ctx) = saved_ctx {
                                unsafe { *ctx_ref.as_ptr() = ctx };
                            }
                            return Some(jit_result);
                        }

                        let result_val = if jit_result == JIT_DEOPT {
                            None
                        } else {
                            jit_to_jsvalue_ext(jit_result)
                        };
                        unsafe { (*heap_ref.as_ptr()).truncate(heap_base) };
                        if let Some(ctx) = saved_ctx {
                            unsafe { *ctx_ref.as_ptr() = ctx };
                        }
                        return result_val.map(jsvalue_to_jit_i64);
                    }
                }

                // No Maglev code — deopt to interpreter.
                return None;
            }

            let _ = heap;
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
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let ctx_opt = unsafe { &*(*ptrs.context).as_ptr() };
            if let Some(ctx_rc) = ctx_opt.as_ref() {
                let ctx = unsafe { &*ctx_rc.as_ptr() };
                return ctx
                    .slots
                    .get(slot_idx as usize)
                    .map(jsvalue_ref_to_jit_i64)
                    .unwrap_or(JIT_UNDEFINED);
            }
            return JIT_DEOPT;
        }
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
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let ctx_opt = unsafe { &*(*ptrs.context).as_ptr() };
            if let Some(ctx_rc) = ctx_opt.as_ref() {
                let ctx = unsafe { &mut *ctx_rc.as_ptr() };
                let slot = slot_idx as usize;
                if slot >= ctx.slots.len() {
                    ctx.slots.resize(slot + 1, JsValue::Undefined);
                }
                ctx.slots[slot] = value;
                return value_i64;
            }
            return JIT_DEOPT;
        }
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
        // Single-threaded JIT: no concurrent borrows during slot load.
        let ctx_ref = unsafe { &*(ctx_raw as *const RefCell<JsContext>) };
        let ctx = unsafe { &*ctx_ref.as_ptr() };
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
        // Fast path: decode Smi inline (most common case for counters,
        // loop variables, etc.) to avoid the 4 magic-constant comparisons
        // in `jit_i64_to_jsvalue`.
        let value = if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
            JsValue::Smi(value_i64 as i32)
        } else {
            jit_i64_to_jsvalue(value_i64)
        };
        // SAFETY: single-threaded JIT; no concurrent borrows during
        // context slot store.
        let ctx_ref = unsafe { &*(ctx_raw as *const RefCell<JsContext>) };
        let ctx = unsafe { &mut *ctx_ref.as_ptr() };
        let slot = slot_idx as usize;
        if slot >= ctx.slots.len() {
            ctx.slots.resize(slot + 1, JsValue::Undefined);
        }
        ctx.slots[slot] = value;
        value_i64
    }

    /// Load a value from a context slot using a JIT-encoded context value.
    ///
    /// Handles two encoding formats:
    /// 1. **Heap handle** – from `CreateFunctionContext` (in JIT_HEAP_TAG
    ///    range). Decoded via `jit_i64_to_jsvalue`.
    /// 2. **Raw pointer** – from the direct-call cache's closure context
    ///    (`Rc::as_ptr`). Dereferenced directly as
    ///    `*const RefCell<JsContext>`.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_i64`) – JIT i64 encoding of the context value.
    /// * `RSI` (`slot_idx`) – context slot index.
    ///
    /// Returns the slot value as `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-emitted assembly, not Rust code.
    pub extern "C" fn jit_runtime_load_context_slot_direct(ctx_i64: i64, slot_idx: i64) -> i64 {
        let slot = slot_idx as usize;

        // Path 1: heap handle (from CreateFunctionContext etc.)
        if is_heap_handle(ctx_i64) {
            let ctx_val = jit_i64_to_jsvalue(ctx_i64);
            return match ctx_val {
                JsValue::Context(ctx_rc) => {
                    let ctx = ctx_rc.borrow();
                    ctx.slots
                        .get(slot)
                        .map(jsvalue_ref_to_jit_i64)
                        .unwrap_or(JIT_UNDEFINED)
                }
                _ => JIT_DEOPT,
            };
        }

        // Path 2: raw pointer (from direct-call closure context cache).
        // SAFETY: ctx_i64 is `Rc::as_ptr(rc) as usize` where `rc` is
        // held alive by the callee's BytecodeArray. Single-threaded
        // JIT guarantees no concurrent borrows.
        if ctx_i64 != 0 {
            let ctx_ref = unsafe { &*(ctx_i64 as *const RefCell<JsContext>) };
            let ctx = unsafe { &*ctx_ref.as_ptr() };
            return ctx
                .slots
                .get(slot)
                .map(jsvalue_ref_to_jit_i64)
                .unwrap_or(JIT_UNDEFINED);
        }

        JIT_DEOPT
    }

    /// Store a value into a context slot using a JIT-encoded context value.
    ///
    /// Handles two encoding formats:
    /// 1. **Heap handle** – from `CreateFunctionContext`.
    /// 2. **Raw pointer** – from the direct-call closure context cache.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_i64`) – JIT i64 encoding of the context value.
    /// * `RSI` (`slot_idx`) – context slot index.
    /// * `RDX` (`value_i64`) – JIT i64 encoding of the value to store.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-emitted assembly, not Rust code.
    pub extern "C" fn jit_runtime_store_context_slot_direct(
        ctx_i64: i64,
        slot_idx: i64,
        value_i64: i64,
    ) -> i64 {
        let slot = slot_idx as usize;

        // Decode value (needed for both paths).
        let value = if value_i64 >= i32::MIN as i64 && value_i64 <= i32::MAX as i64 {
            JsValue::Smi(value_i64 as i32)
        } else {
            jit_i64_to_jsvalue(value_i64)
        };

        // Path 1: heap handle (from CreateFunctionContext etc.)
        if is_heap_handle(ctx_i64) {
            let ctx_val = jit_i64_to_jsvalue(ctx_i64);
            return match ctx_val {
                JsValue::Context(ctx_rc) => {
                    let mut ctx = ctx_rc.borrow_mut();
                    if slot >= ctx.slots.len() {
                        ctx.slots.resize(slot + 1, JsValue::Undefined);
                    }
                    ctx.slots[slot] = value;
                    value_i64
                }
                _ => JIT_DEOPT,
            };
        }

        // Path 2: raw pointer (from direct-call closure context cache).
        // SAFETY: ctx_i64 is `Rc::as_ptr(rc) as usize` where `rc` is
        // held alive by the callee's BytecodeArray. Single-threaded
        // JIT guarantees no concurrent borrows.
        if ctx_i64 != 0 {
            let ctx_ref = unsafe { &*(ctx_i64 as *const RefCell<JsContext>) };
            let ctx = unsafe { &mut *ctx_ref.as_ptr() };
            if slot >= ctx.slots.len() {
                ctx.slots.resize(slot + 1, JsValue::Undefined);
            }
            ctx.slots[slot] = value;
            return value_i64;
        }

        JIT_DEOPT
    }

    // ── Lean context-slot helpers (Maglev fast path) ──────────────────────
    //
    // These handle **only** the raw-pointer context path used by depth-0
    // closure contexts (stored as `Rc::as_ptr()`).  By dropping the
    // heap-handle decoding path and the `RefCell::borrow()` it requires,
    // the generated machine code is roughly half the size of the `_direct`
    // variants, which improves icache utilisation and branch prediction in
    // tight closure loops (e.g. `closure_counter_1k`).
    //
    // When the context is a heap handle (from `CreateFunctionContext`) or
    // null, the lean helpers return [`JIT_DEOPT`], letting Maglev fall
    // back to the interpreter.

    /// Lean context-slot **loader** for Maglev codegen.
    ///
    /// Reads `ctx.slots[slot]` from a raw-pointer context and encodes the
    /// result as a JIT `i64`.  Returns [`JIT_DEOPT`] for heap-handle or
    /// null contexts.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_i64`) – raw `*const RefCell<JsContext>` as `i64`.
    /// * `RSI` (`slot_idx`) – context slot index.
    ///
    /// Returns the slot value as `i64` in `RAX`, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    #[inline(never)]
    pub extern "C" fn jit_runtime_load_context_slot_lean(ctx_i64: i64, slot_idx: i64) -> i64 {
        if ctx_i64 == 0 {
            return JIT_DEOPT;
        }
        let slot = slot_idx as usize;

        // Path 1: heap handle (from CreateFunctionContext in closures).
        if is_heap_handle(ctx_i64) {
            let ctx_val = jit_i64_to_jsvalue(ctx_i64);
            return match ctx_val {
                JsValue::Context(ctx_rc) => {
                    let ctx = ctx_rc.borrow();
                    ctx.slots
                        .get(slot)
                        .map(jsvalue_ref_to_jit_i64)
                        .unwrap_or(JIT_UNDEFINED)
                }
                _ => JIT_DEOPT,
            };
        }

        // Path 2: raw pointer (from direct-call closure context cache).
        // SAFETY: ctx_i64 is `Rc::as_ptr(rc) as usize` where `rc` is held
        // alive by the callee's BytecodeArray.  Single-threaded JIT
        // guarantees no concurrent borrows.
        let ctx_ref = unsafe { &*(ctx_i64 as *const RefCell<JsContext>) };
        let ctx = unsafe { &*ctx_ref.as_ptr() };
        ctx.slots
            .get(slot)
            .map(jsvalue_ref_to_jit_i64)
            .unwrap_or(JIT_UNDEFINED)
    }

    /// Lean context-slot **store** for Maglev codegen.
    ///
    /// Writes `value` into `ctx.slots[slot]`.  Handles both raw-pointer
    /// and heap-handle contexts (closures use heap handles).
    /// Returns [`JIT_DEOPT`] for null contexts.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`ctx_i64`) – raw `*const RefCell<JsContext>` as `i64`.
    /// * `RSI` (`slot_idx`) – context slot index.
    /// * `RDX` (`value_i64`) – JIT `i64` encoding of the value to store.
    ///
    /// Returns `value_i64` in `RAX` on success, or [`JIT_DEOPT`].
    #[allow(dead_code)] // Called from JIT-generated machine code, not Rust.
    #[inline(never)]
    pub extern "C" fn jit_runtime_store_context_slot_lean(
        ctx_i64: i64,
        slot_idx: i64,
        value_i64: i64,
    ) -> i64 {
        if ctx_i64 == 0 {
            return JIT_DEOPT;
        }
        let slot = slot_idx as usize;
        let value = if is_smi(value_i64) {
            JsValue::Smi(value_i64 as i32)
        } else {
            jit_i64_to_jsvalue(value_i64)
        };

        // Path 1: heap handle (from CreateFunctionContext in closures).
        if is_heap_handle(ctx_i64) {
            let ctx_val = jit_i64_to_jsvalue(ctx_i64);
            return match ctx_val {
                JsValue::Context(ctx_rc) => {
                    let mut ctx = ctx_rc.borrow_mut();
                    if slot >= ctx.slots.len() {
                        ctx.slots.resize(slot + 1, JsValue::Undefined);
                    }
                    ctx.slots[slot] = value;
                    value_i64
                }
                _ => JIT_DEOPT,
            };
        }

        // Path 2: raw pointer.
        // SAFETY: ctx_i64 is `Rc::as_ptr(rc) as usize` where `rc` is held
        // alive by the callee's BytecodeArray.  Single-threaded JIT
        // guarantees no concurrent borrows.
        let ctx_ref = unsafe { &*(ctx_i64 as *const RefCell<JsContext>) };
        let ctx = unsafe { &mut *ctx_ref.as_ptr() };
        if slot >= ctx.slots.len() {
            ctx.slots.resize(slot + 1, JsValue::Undefined);
        }
        ctx.slots[slot] = value;
        value_i64
    }

    // ── CreateFunctionContext / CreateClosure / Object & Array stubs ────────

    /// Create an empty plain object.
    ///
    /// Replaces the trampoline dispatch for `CreateEmptyObjectLiteral` in
    /// Maglev — a zero-argument operation.
    pub extern "C" fn jit_runtime_create_empty_object() -> i64 {
        let obj = JsValue::PlainObject(Rc::new(RefCell::new(PropertyMap::new())));
        alloc_heap_handle(obj)
    }

    /// Create an empty array.
    ///
    /// Replaces the trampoline dispatch for `CreateEmptyArrayLiteral` and
    /// `CreateArrayLiteral` in Maglev.
    pub extern "C" fn jit_runtime_create_empty_array() -> i64 {
        let vec = crate::objects::value::take_recycled_array_vec();
        let arr = JsValue::Array(Rc::new(RefCell::new(vec)));
        alloc_heap_handle(arr)
    }

    /// Create a new function context with `slot_count` slots.
    ///
    /// This replaces the trampoline dispatch for `CreateFunctionContext` in
    /// Maglev, saving ~30 instructions of trampoline overhead.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`slot_count`) – number of context slots.
    ///
    /// Returns the new context as JIT `i64` in `RAX`.
    pub extern "C" fn jit_runtime_create_function_context(slot_count: i64) -> i64 {
        let parent = RT_CONTEXT.with(|ctx_cell| {
            let ctx_opt = ctx_cell.borrow();
            ctx_opt.as_ref().map(Rc::clone)
        });
        let js_ctx = JsContext::new(slot_count as usize, parent);
        let val = JsValue::Context(js_ctx);
        jsvalue_to_jit_i64(val)
    }

    /// Create a closure from a constant-pool entry.
    ///
    /// This replaces the trampoline dispatch for `CreateClosure` /
    /// `FastCreateClosure` in Maglev.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`cp_idx`) – constant pool index of the inner function.
    ///
    /// Returns the new closure as JIT `i64` in `RAX`, or [`JIT_DEOPT`].
    pub extern "C" fn jit_runtime_create_closure(cp_idx: i64) -> i64 {
        // SAFETY: RT_BYTECODE is always set by jit_runtime_setup
        // before any JIT code runs, and points at a live
        // BytecodeArray for the duration of the JIT execution.
        let ba_ptr = RT_BYTECODE.with(|b| b.get());
        if ba_ptr.is_null() {
            track_stub_deopt(STUB_CREATE_CLOSURE);
            return JIT_DEOPT;
        }
        let ba = unsafe { &*ba_ptr };
        let Some(entry) = ba.get_constant(cp_idx as u32) else {
            track_stub_deopt(STUB_CREATE_CLOSURE);
            return JIT_DEOPT;
        };
        let ConstantPoolEntry::Function(inner_ba) = entry else {
            track_stub_deopt(STUB_CREATE_CLOSURE);
            return JIT_DEOPT;
        };
        let closure_ctx = RT_CONTEXT.with(|ctx_cell| {
            let ctx_opt = ctx_cell.borrow();
            ctx_opt.as_ref().map(Rc::clone)
        });
        let func = Rc::new(inner_ba.clone_for_closure(closure_ctx));
        let val = JsValue::Function(func);
        jsvalue_to_jit_i64(val)
    }

    // ── Context push/pop stubs for Maglev ──────────────────────────────────

    /// Push a new context as the active closure context, returning the old
    /// context.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`new_ctx_i64`) – JIT i64 encoding of the new context value.
    ///
    /// Returns the **old** context as JIT `i64` in `RAX`.
    pub extern "C" fn jit_runtime_push_context(new_ctx_i64: i64) -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let ctx_ref = unsafe { &*ptrs.context };
            // Read the old context in a scoped borrow so that `ctx_opt`
            // is dropped before the write through `ctx_ref.as_ptr()`.
            let old = {
                let ctx_opt = unsafe { &*ctx_ref.as_ptr() };
                ctx_opt
                    .as_ref()
                    .map(|c| JsValue::Context(Rc::clone(c)))
                    .unwrap_or(JsValue::Undefined)
            };
            let old_i64 = jsvalue_to_jit_i64(old);
            let new_ctx_val = jit_i64_to_jsvalue(new_ctx_i64);
            if let JsValue::Context(c) = new_ctx_val {
                // SAFETY: `ctx_opt` was dropped; no aliased references.
                // Single-threaded JIT.
                unsafe { *ctx_ref.as_ptr() = Some(c) };
            }
            return old_i64;
        }
        let old = RT_CONTEXT.with(|ctx_cell| {
            let ctx_opt = ctx_cell.borrow();
            ctx_opt
                .as_ref()
                .map(|c| JsValue::Context(Rc::clone(c)))
                .unwrap_or(JsValue::Undefined)
        });
        let old_i64 = jsvalue_to_jit_i64(old);
        let new_ctx_val = jit_i64_to_jsvalue(new_ctx_i64);
        if let JsValue::Context(c) = new_ctx_val {
            RT_CONTEXT.with(|ctx_cell| {
                *ctx_cell.borrow_mut() = Some(c);
            });
        }
        old_i64
    }

    /// Restore a previously saved context as the active closure context.
    ///
    /// # Calling convention (SysV AMD64)
    ///
    /// * `RDI` (`saved_ctx_i64`) – JIT i64 encoding of the saved context.
    ///
    /// Returns `JIT_UNDEFINED` in `RAX`.
    pub extern "C" fn jit_runtime_pop_context(saved_ctx_i64: i64) -> i64 {
        let saved = jit_i64_to_jsvalue(saved_ctx_i64);
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime;
            // single-threaded JIT, no concurrent borrows.
            let ctx_ref = unsafe { &*ptrs.context };
            match saved {
                JsValue::Context(c) => unsafe { *ctx_ref.as_ptr() = Some(c) },
                _ => unsafe { *ctx_ref.as_ptr() = None },
            }
            return JIT_UNDEFINED;
        }
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
        JIT_UNDEFINED
    }

    /// Return the raw pointer of the current TLS closure context.
    ///
    /// This is used by the Maglev codegen after `PushContext` / `PopContext`
    /// stub calls to update the register-file context cache so that
    /// subsequent `LoadCurrentContextSlot` / `StoreCurrentContextSlot`
    /// operations find the correct context pointer.
    ///
    /// Returns `Rc::as_ptr()` cast to `i64`, or `0` when no context is set.
    pub extern "C" fn jit_runtime_get_current_ctx_raw_ptr() -> i64 {
        let ptrs = RT_PTRS.with(|p| p.get());
        if ptrs.is_cached() {
            // SAFETY: cached pointers valid for thread lifetime.
            let ctx_ref = unsafe { &*ptrs.context };
            let ctx_opt = unsafe { &*ctx_ref.as_ptr() };
            return ctx_opt.as_ref().map(|c| Rc::as_ptr(c) as i64).unwrap_or(0);
        }
        RT_CONTEXT.with(|ctx_cell| {
            ctx_cell
                .borrow()
                .as_ref()
                .map(|c| Rc::as_ptr(c) as i64)
                .unwrap_or(0)
        })
    }

    // ── Generic arithmetic stubs for Maglev ─────────────────────────────────

    /// Generic Add: handles Smi + Smi (with overflow), HeapNumber, and
    /// string concatenation.  Deopts on complex cases.
    pub extern "C" fn jit_runtime_generic_add(left: i64, right: i64) -> i64 {
        // Fast path: both operands are Smi (i32 range).  Uses sign-
        // extension check (single comparison per operand) and i32
        // checked_add (single overflow-flag test) for a tighter code
        // sequence than the 6-comparison alternative.
        if is_smi(left) && is_smi(right) {
            if let Some(sum) = (left as i32).checked_add(right as i32) {
                return sum as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(left as f64 + right as f64));
        }
        generic_add_slow(left, right)
    }

    /// Convert a [`JsValue`] to an `f64` following ECMAScript `ToNumber` semantics.
    fn js_to_number(v: &JsValue) -> f64 {
        match v {
            JsValue::Smi(n) => *n as f64,
            JsValue::HeapNumber(f) => *f,
            JsValue::Boolean(true) => 1.0,
            JsValue::Boolean(false) => 0.0,
            JsValue::Null => 0.0,
            JsValue::Undefined => f64::NAN,
            JsValue::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    0.0
                } else {
                    trimmed.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            _ => f64::NAN, // Object, Function, Array, etc.
        }
    }

    /// Slow path for [`jit_runtime_generic_add`] — handles all JS type
    /// combinations using ToNumber / string-concatenation semantics.
    #[cold]
    fn generic_add_slow(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            // Smi+Smi: happens when one/both operands arrive as heap handles
            // that resolve to Smi after jit_i64_to_jsvalue.
            (JsValue::Smi(a), JsValue::Smi(b)) => {
                if let Some(sum) = a.checked_add(*b) {
                    jsvalue_to_jit_i64(JsValue::Smi(sum))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 + *b as f64))
                }
            }
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a + *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 + *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a + *b as f64))
            }
            // String + anything → string concatenation (add only)
            (JsValue::String(s), other) => {
                let rhs = other.to_js_string().unwrap_or_else(|_| String::new());
                let result: Rc<str> = format!("{s}{rhs}").into();
                jsvalue_to_jit_i64(JsValue::String(result))
            }
            (other, JsValue::String(s)) => {
                let lhs = other.to_js_string().unwrap_or_else(|_| String::new());
                let result: Rc<str> = format!("{lhs}{s}").into();
                jsvalue_to_jit_i64(JsValue::String(result))
            }
            // All other types: ToNumber on both operands, then add as f64.
            _ => {
                let a = js_to_number(&l);
                let b = js_to_number(&r);
                let result = a + b;
                if result.is_finite()
                    && result.fract() == 0.0
                    && result >= i32::MIN as f64
                    && result <= i32::MAX as f64
                {
                    jsvalue_to_jit_i64(JsValue::Smi(result as i32))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(result))
                }
            }
        }
    }

    /// Generic Subtract.
    pub extern "C" fn jit_runtime_generic_sub(left: i64, right: i64) -> i64 {
        // Fast path: Smi - Smi.
        if is_smi(left) && is_smi(right) {
            if let Some(diff) = (left as i32).checked_sub(right as i32) {
                return diff as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(left as f64 - right as f64));
        }
        generic_sub_slow(left, right)
    }

    /// Slow path for [`jit_runtime_generic_sub`].
    #[cold]
    fn generic_sub_slow(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => {
                if let Some(diff) = a.checked_sub(*b) {
                    jsvalue_to_jit_i64(JsValue::Smi(diff))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 - *b as f64))
                }
            }
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a - *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 - *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a - *b as f64))
            }
            // All other types: ToNumber on both operands, then subtract as f64.
            _ => {
                let a = js_to_number(&l);
                let b = js_to_number(&r);
                let result = a - b;
                if result.is_finite()
                    && result.fract() == 0.0
                    && result >= i32::MIN as f64
                    && result <= i32::MAX as f64
                {
                    jsvalue_to_jit_i64(JsValue::Smi(result as i32))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(result))
                }
            }
        }
    }

    /// Generic Multiply.
    pub extern "C" fn jit_runtime_generic_mul(left: i64, right: i64) -> i64 {
        // Fast path: Smi * Smi.
        if is_smi(left) && is_smi(right) {
            if let Some(product) = (left as i32).checked_mul(right as i32) {
                return product as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(left as f64 * right as f64));
        }
        generic_mul_slow(left, right)
    }

    /// Slow path for [`jit_runtime_generic_mul`].
    #[cold]
    fn generic_mul_slow(left: i64, right: i64) -> i64 {
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (&l, &r) {
            (JsValue::Smi(a), JsValue::Smi(b)) => {
                if let Some(product) = a.checked_mul(*b) {
                    jsvalue_to_jit_i64(JsValue::Smi(product))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 * *b as f64))
                }
            }
            (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a * *b))
            }
            (JsValue::Smi(a), JsValue::HeapNumber(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a as f64 * *b))
            }
            (JsValue::HeapNumber(a), JsValue::Smi(b)) => {
                jsvalue_to_jit_i64(JsValue::HeapNumber(*a * *b as f64))
            }
            // All other types: ToNumber on both operands, then multiply as f64.
            _ => {
                let a = js_to_number(&l);
                let b = js_to_number(&r);
                let result = a * b;
                if result.is_finite()
                    && result.fract() == 0.0
                    && result >= i32::MIN as f64
                    && result <= i32::MAX as f64
                {
                    jsvalue_to_jit_i64(JsValue::Smi(result as i32))
                } else {
                    jsvalue_to_jit_i64(JsValue::HeapNumber(result))
                }
            }
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
        // Fast path: both Smi — do bitwise AND directly on i64 encoding.
        if is_smi(left) && is_smi(right) {
            return (left as i32 & right as i32) as i64;
        }
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => (a & b) as i64,
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_bitwise_or(left: i64, right: i64) -> i64 {
        if is_smi(left) && is_smi(right) {
            return (left as i32 | right as i32) as i64;
        }
        let l = jit_i64_to_jsvalue(left);
        let r = jit_i64_to_jsvalue(right);
        match (to_int32(&l), to_int32(&r)) {
            (Some(a), Some(b)) => (a | b) as i64,
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
        // Fast path: Smi (i32 range) — negate directly.
        if is_smi(value) {
            if value == 0 {
                // -0 in JS is -0.0 (negative zero), but 0 negated stays 0
                // as Smi. JS `-0` is only produced by `-0` literal or
                // specific fp ops. Smi 0 negated → Smi 0.
                return 0;
            }
            if let Some(neg) = (value as i32).checked_neg() {
                return neg as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(-(value as f64)));
        }
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(-*f)),
            _ => JIT_DEOPT,
        }
    }

    pub extern "C" fn jit_runtime_generic_increment(value: i64) -> i64 {
        // Fast path: Smi (i32 range) — skip JsValue construction.
        if is_smi(value) {
            if let Some(inc) = (value as i32).checked_add(1) {
                return inc as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(value as f64 + 1.0));
        }
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(*f + 1.0)),
            _ => {
                track_stub_deopt(STUB_GENERIC_ARITH);
                JIT_DEOPT
            }
        }
    }

    pub extern "C" fn jit_runtime_generic_decrement(value: i64) -> i64 {
        // Fast path: Smi (i32 range) — skip JsValue construction.
        if is_smi(value) {
            if let Some(dec) = (value as i32).checked_sub(1) {
                return dec as i64;
            }
            return alloc_heap_handle(JsValue::HeapNumber(value as f64 - 1.0));
        }
        let v = jit_i64_to_jsvalue(value);
        match &v {
            JsValue::HeapNumber(f) => jsvalue_to_jit_i64(JsValue::HeapNumber(*f - 1.0)),
            _ => {
                track_stub_deopt(STUB_GENERIC_ARITH);
                JIT_DEOPT
            }
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
        construct0_inner(ctor_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CONSTRUCT0);
            JIT_DEOPT
        })
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

                // Fast path: if the constructor body is trivial (only
                // LdaUndefined + Return, or CreateMappedArguments +
                // LdaUndefined + Return), skip interpreter re-entry entirely
                // and return `this` directly.  Empty constructors like
                // `function Base() {}` hit this path.
                if ba.has_trivial_body() {
                    maybe_cache_construct_boilerplate(ba, &this_val);
                    return Some(jsvalue_to_jit_i64(this_val));
                }

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

    // ── Construct stub for Maglev (1 argument) ──────────────────────────────

    /// Construct with exactly 1 argument — takes the constructor and one arg
    /// as packed i64 values.
    pub extern "C" fn jit_runtime_construct1(ctor_i64: i64, arg0_i64: i64) -> i64 {
        construct1_inner(ctor_i64, arg0_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CONSTRUCT1);
            JIT_DEOPT
        })
    }

    fn construct1_inner(ctor_i64: i64, arg0_i64: i64) -> Option<i64> {
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

                let arg0 = jit_i64_to_jsvalue(arg0_i64);

                let saved_ba = RT_BYTECODE.with(|b| b.get());
                let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());
                let env = env_opt?;

                let mut callee_frame =
                    InterpreterFrame::new_with_globals(Rc::clone(ba), vec![arg0], env);
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

    // ── Construct stub for Maglev (2 arguments) ─────────────────────────────

    /// Construct with exactly 2 arguments — takes the constructor and two args
    /// as packed i64 values.
    pub extern "C" fn jit_runtime_construct2(ctor_i64: i64, arg0_i64: i64, arg1_i64: i64) -> i64 {
        construct2_inner(ctor_i64, arg0_i64, arg1_i64).unwrap_or_else(|| {
            track_stub_deopt(STUB_CONSTRUCT2);
            JIT_DEOPT
        })
    }

    fn construct2_inner(ctor_i64: i64, arg0_i64: i64, arg1_i64: i64) -> Option<i64> {
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

                let arg0 = jit_i64_to_jsvalue(arg0_i64);
                let arg1 = jit_i64_to_jsvalue(arg1_i64);

                let saved_ba = RT_BYTECODE.with(|b| b.get());
                let env_opt = RT_GLOBAL.with(|g| g.borrow().env.as_ref().cloned());
                let env = env_opt?;

                let mut callee_frame =
                    InterpreterFrame::new_with_globals(Rc::clone(ba), vec![arg0, arg1], env);
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

    /// Native sum of all Smi elements in an array.
    ///
    /// Called by [`ValueNode::SpeculativeSumFusion`] codegen.  Given a JIT
    /// heap handle pointing to a `JsValue::Array`, iterates all elements
    /// and returns their integer sum.  Returns [`JIT_DEOPT`] if:
    /// - `arr_i64` is not a valid heap handle,
    /// - the heap value is not `JsValue::Array`,
    /// - any element is not `JsValue::Smi`,
    /// - the sum overflows `i32`.
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_batch_sum_smi(arr_i64: i64) -> i64 {
        if !is_heap_handle(arr_i64) {
            return JIT_DEOPT;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        let arr_idx = (arr_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if arr_idx >= heap.len() {
            return JIT_DEOPT;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds checked above.
        let items_rc = match unsafe { heap.get_unchecked(arr_idx) } {
            JsValue::Array(rc) => rc,
            _ => return JIT_DEOPT,
        };
        // SAFETY: single-threaded JIT; no concurrent borrows.
        let items = unsafe { &*items_rc.as_ptr() };
        let layout = jit_runtime::probe_jsvalue_layout();
        // Fast path: use raw pointer arithmetic to read Smi payloads
        // at a fixed stride, avoiding per-element pattern matching.
        if layout.disc_offset == 0 && layout.smi_payload_offset == 4 {
            let base = items.as_ptr() as *const u8;
            let stride = layout.jsvalue_size;
            let count = items.len();
            let smi_disc = layout.smi_disc;
            let mut sum: i64 = 0;
            // Branchless discriminant accumulation: XOR each disc with the
            // expected Smi discriminant and OR into `disc_mismatch`.  If any
            // element is not a Smi the result is non-zero → deopt once at the
            // end instead of branching per element.  This lets the CPU
            // pipeline the loop without mispredicts.
            let mut disc_mismatch: u8 = 0;
            for i in 0..count {
                // SAFETY: i < count ≤ items.len(); stride * i + 7 < stride * count
                // which fits in the Vec's allocation.
                let elem_ptr = unsafe { base.add(stride * i) };
                disc_mismatch |= unsafe { *elem_ptr } ^ smi_disc;
                // Read i32 payload at offset 4 (smi_payload_offset).
                let payload = unsafe { *(elem_ptr.add(4) as *const i32) };
                sum += payload as i64;
            }
            if disc_mismatch != 0 {
                return JIT_DEOPT;
            }
            if sum < i32::MIN as i64 || sum > i32::MAX as i64 {
                return JIT_DEOPT;
            }
            return sum;
        }
        // Fallback: safe pattern-matching path.
        let mut sum: i64 = 0;
        for elem in items.iter() {
            match elem {
                JsValue::Smi(v) => sum += *v as i64,
                _ => return JIT_DEOPT,
            }
        }
        // Check i32 range — Smi result must fit in i32.
        if sum < i32::MIN as i64 || sum > i32::MAX as i64 {
            return JIT_DEOPT;
        }
        sum
    }

    /// Native batch push of sequential Smi values `0..count` into an array.
    ///
    /// Called by [`ValueNode::SpeculativePushFusion`] codegen.  Given a JIT
    /// heap handle pointing to a `JsValue::Array`, pushes `count` sequential
    /// Smi values starting from 0.  The array is pre-allocated with
    /// `reserve(count)` for a single allocation.
    ///
    /// Returns `count` (the new array length) on success, or [`JIT_DEOPT`]
    /// if the handle is invalid or not an Array.
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_batch_push_smi_range(arr_i64: i64, count: i64) -> i64 {
        if !is_heap_handle(arr_i64) {
            return JIT_DEOPT;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        let arr_idx = (arr_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if arr_idx >= heap.len() {
            return JIT_DEOPT;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds checked above.
        let items_rc = match unsafe { heap.get_unchecked(arr_idx) } {
            JsValue::Array(rc) => rc,
            _ => return JIT_DEOPT,
        };
        let count = count as usize;
        // SAFETY: single-threaded JIT; no concurrent borrows.
        let items = unsafe { &mut *items_rc.as_ptr() };
        items.reserve(count);
        // SAFETY: reserve guarantees capacity; we write exactly `count` values.
        let base_len = items.len();
        unsafe {
            let ptr = items.as_mut_ptr().add(base_len);
            for i in 0..count {
                ptr.add(i).write(JsValue::Smi(i as i32));
            }
            items.set_len(base_len + count);
        }
        items.len() as i64
    }

    /// Native batch fill of an empty array with `true` values.
    ///
    /// Called by Maglev's sieve-style fill fusion.  The optimizer only emits
    /// this for freshly-created arrays, and the runtime verifies that shape
    /// before mutating so a stale speculative match deopts instead of changing
    /// semantics.
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_batch_fill_true(arr_i64: i64, count_i64: i64) -> i64 {
        if !is_heap_handle(arr_i64) || !(0..=100_000).contains(&count_i64) {
            return JIT_DEOPT;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        let arr_idx = (arr_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if arr_idx >= heap.len() {
            return JIT_DEOPT;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds checked above.
        let items_rc = match unsafe { heap.get_unchecked(arr_idx) } {
            JsValue::Array(rc) => rc,
            _ => return JIT_DEOPT,
        };
        // SAFETY: single-threaded JIT; no concurrent borrows.
        let items = unsafe { &mut *items_rc.as_ptr() };
        if !items.is_empty() {
            return JIT_DEOPT;
        }
        let count = count_i64 as usize;
        items.reserve(count);
        // SAFETY: reserve guarantees capacity; we write exactly `count` values.
        unsafe {
            let ptr = items.as_mut_ptr();
            for i in 0..count {
                ptr.add(i).write(JsValue::Boolean(true));
            }
            items.set_len(count);
        }
        count_i64
    }

    /// Native count of boolean-true entries in an array prefix.
    ///
    /// Called by Maglev's sieve count-loop fusion.  It intentionally accepts
    /// only booleans so speculative matches deopt rather than applying JS
    /// truthiness to shapes this narrow optimizer did not prove.
    #[allow(dead_code)] // Called from JIT-generated machine code.
    pub extern "C" fn jit_runtime_count_bool_true(arr_i64: i64, count_i64: i64) -> i64 {
        if !is_heap_handle(arr_i64) || !(0..=100_000).contains(&count_i64) {
            return JIT_DEOPT;
        }
        let ptrs = RT_PTRS.with(|p| p.get());
        if !ptrs.is_cached() {
            return JIT_DEOPT;
        }
        let arr_idx = (arr_i64 - JIT_HEAP_TAG) as usize;
        // SAFETY: cached pointers valid for thread lifetime; single-threaded.
        let heap = unsafe { &*(&*ptrs.heap).as_ptr() };
        if arr_idx >= heap.len() {
            return JIT_DEOPT;
        }
        use crate::objects::value::JsValue;
        // SAFETY: bounds checked above.
        let items_rc = match unsafe { heap.get_unchecked(arr_idx) } {
            JsValue::Array(rc) => rc,
            _ => return JIT_DEOPT,
        };
        // SAFETY: single-threaded JIT; no concurrent borrows.
        let items = unsafe { &*items_rc.as_ptr() };
        let count = count_i64 as usize;
        if count > items.len() {
            return JIT_DEOPT;
        }
        let mut total = 0i64;
        for item in &items[..count] {
            match item {
                JsValue::Boolean(true) => total += 1,
                JsValue::Boolean(false) => {}
                _ => return JIT_DEOPT,
            }
        }
        total
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
pub use jit_runtime::{
    alloc_jit_heap_handle, jit_runtime_set_context, jit_runtime_set_global_env, jit_runtime_setup,
    jit_runtime_teardown, jit_to_jsvalue_ext,
};

#[cfg(all(target_arch = "x86_64", unix))]
pub use jit_runtime::{
    ArrayIcInfo, JsContextLayout, JsValueLayout, VecJsValueLayout, jit_runtime_fill_array_ic_r15,
    jit_runtime_inline_load_keyed_smi_ic_r15, probe_jscontext_layout, probe_jsvalue_layout,
    probe_vec_jsvalue_layout,
};

// ── Per-stub deopt tracking (platform-independent API) ──────────────────

/// Number of slots in the per-stub deopt counter array.
pub const STUB_DEOPT_SLOTS: usize = 24;

/// Human-readable names for each stub index (for diagnostic printing).
pub const STUB_NAMES: [&str; STUB_DEOPT_SLOTS] = [
    "lda_named",
    "sta_named",
    "lda_global",
    "sta_global",
    "construct0",
    "construct1",
    "construct2",
    "create_obj_props",
    "fast_create_obj",
    "lda_keyed",
    "sta_keyed",
    "call_undef0",
    "create_closure",
    "sta_named_own",
    "call_prop0",
    "call_prop1",
    "call_undef1",
    "call_undef2",
    "fast_array_load",
    "fast_array_store",
    "fast_array_push",
    "trampoline",
    "generic_arith",
    "_reserved23",
];

/// Return the current per-stub deopt counts.
///
/// On platforms without the JIT (non-x86-64/non-Unix), returns all zeros.
pub fn stub_deopt_counts() -> [u64; STUB_DEOPT_SLOTS] {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::stub_deopt_counts()
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    {
        [0; STUB_DEOPT_SLOTS]
    }
}

/// Reset all per-stub deopt counts to zero.
///
/// On platforms without the JIT this is a no-op.
pub fn reset_stub_deopt_counts() {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::reset_stub_deopt_counts();
    }
}

/// Return per-stub *first*-deopt-per-invocation counts.
///
/// Each slot counts how many times the corresponding stub was the
/// **first** stub to deopt in a given invocation (i.e. since the
/// last [`jit_runtime_setup`] call).
///
/// On platforms without the JIT (non-x86-64/non-Unix), returns all zeros.
pub fn first_deopt_counts() -> [u64; STUB_DEOPT_SLOTS] {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::first_deopt_counts()
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    {
        [0; STUB_DEOPT_SLOTS]
    }
}

/// Reset all per-stub first-deopt counts to zero.
///
/// On platforms without the JIT this is a no-op.
pub fn reset_first_deopt_counts() {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::reset_first_deopt_counts();
    }
}

/// Return the current per-stub FFI call counts.
///
/// On platforms without the JIT (non-x86-64/non-Unix), returns all zeros.
pub fn stub_call_counts() -> [u64; STUB_DEOPT_SLOTS] {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::stub_call_counts()
    }
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    {
        [0; STUB_DEOPT_SLOTS]
    }
}

/// Reset all per-stub call counts to zero.
///
/// On platforms without the JIT this is a no-op.
pub fn reset_stub_call_counts() {
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        jit_runtime::reset_stub_call_counts();
    }
}

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

#[cfg(all(target_arch = "x86_64", unix))]
const STACK_REGISTER_FILE_SLOTS: usize = 32;

#[cfg(all(target_arch = "x86_64", unix))]
fn init_register_file(
    args: &[i64],
    register_file_slots: usize,
) -> smallvec::SmallVec<[i64; STACK_REGISTER_FILE_SLOTS]> {
    use smallvec::{SmallVec, smallvec};

    let mut regs: SmallVec<[i64; STACK_REGISTER_FILE_SLOTS]> = smallvec![0i64; register_file_slots];
    for (i, &value) in args.iter().enumerate().take(regs.len()) {
        regs[i] = value;
    }
    regs
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

        // Keep small register files on the stack to avoid per-call heap churn
        // on hot JIT entry paths.
        let mut regs = init_register_file(args, self.register_file_slots);

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

        if jit_runtime::is_jit_deopt(result) {
            Err(StatorError::Internal("jit deopt".into()))
        } else {
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Persistent JIT code page that caches the `mmap`'d executable memory.
///
/// The page is allocated once (by [`CachedExecutableCode::from_compiled`]) and
/// reused for all subsequent calls, eliminating the per-call `mmap`/`munmap`
/// syscall overhead.
#[cfg(all(target_arch = "x86_64", unix))]
pub struct CachedExecutableCode {
    /// Pointer to the `mmap`'d executable memory.
    ptr: *mut u8,
    /// Size of the `mmap`'d region.
    len: usize,
    /// Function pointer to the JIT code (transmuted from `ptr`).
    func: extern "C" fn(*mut i64) -> i64,
    /// Number of `i64` slots needed in the register file.
    pub register_file_slots: usize,
}

// SAFETY: The mmap'd code is position-independent and can be shared across
// threads.  The memory is owned exclusively by this struct and freed on Drop.
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Send for CachedExecutableCode {}

// SAFETY: The mmap'd code is immutable after construction and the function
// pointer is safe to call from any thread (the register file is caller-owned).
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Sync for CachedExecutableCode {}

#[cfg(all(target_arch = "x86_64", unix))]
impl std::fmt::Debug for CachedExecutableCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedExecutableCode")
            .field("len", &self.len)
            .field("register_file_slots", &self.register_file_slots)
            .finish()
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
impl Drop for CachedExecutableCode {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` were set by a successful `mmap` call in
        // `from_compiled`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
impl CachedExecutableCode {
    /// Allocate executable memory, copy `code` into it, and transmute to a
    /// persistent function pointer.
    ///
    /// This performs the `mmap` + `copy_nonoverlapping` + `transmute` sequence
    /// **once**.  The resulting [`CachedExecutableCode`] can then be called
    /// many times via [`execute`](Self::execute) without further syscalls.
    ///
    /// # Errors
    ///
    /// Returns [`StatorError::Internal`] if `code` is empty or `mmap` fails.
    ///
    /// # Safety
    ///
    /// `code` must be valid x86-64 machine code emitted by the baseline or
    /// Maglev compiler.
    pub unsafe fn from_compiled(code: &[u8], register_file_slots: usize) -> StatorResult<Self> {
        use std::ptr;

        let code_size = code.len();
        if code_size == 0 {
            return Err(StatorError::Internal("compiled code is empty".into()));
        }

        // SAFETY: arguments are valid; MAP_FAILED is checked before use.
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
            ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), code_size);
        }

        // SAFETY:
        // - `mem` contains correctly-encoded x86-64 machine code.
        // - The function signature matches the JIT calling convention:
        //   `extern "C" fn(*mut i64) -> i64` (SysV AMD64).
        let func: extern "C" fn(*mut i64) -> i64 = unsafe { std::mem::transmute(mem) };

        Ok(Self {
            ptr: mem.cast::<u8>(),
            len: code_size,
            func,
            register_file_slots,
        })
    }

    /// Returns the mmap'd executable code as a byte slice.
    ///
    /// Used by the interpreter to lazily initialise a
    /// [`JitExecutableCode`](crate::bytecode::bytecode_array::JitExecutableCode)
    /// or [`CachedMaglevCode`](crate::compiler::maglev::codegen::CachedMaglevCode)
    /// from a previously-cached compilation result.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.ptr` and `self.len` were set by a successful `mmap`
        // call in `from_compiled`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Execute the cached JIT code with the given arguments.
    ///
    /// Allocates only the register file and calls the persistent function
    /// pointer.  Small register files stay on the stack; no `mmap`/`munmap`
    /// syscalls are issued.
    ///
    /// `args` provides the initial values for the parameter slots.  Missing
    /// arguments are filled with `0` (`Smi(0)`); extra arguments are ignored.
    ///
    /// Returns the accumulator value on success, or
    /// [`StatorError::Internal`] if the JIT function returns [`JIT_DEOPT`].
    ///
    /// # Safety
    ///
    /// The caller must ensure this `CachedExecutableCode` was constructed from
    /// valid x86-64 machine code emitted by the baseline or Maglev compiler.
    pub unsafe fn execute(&self, args: &[i64]) -> StatorResult<i64> {
        let mut regs = init_register_file(args, self.register_file_slots);

        let result = (self.func)(regs.as_mut_ptr());

        if jit_runtime::is_jit_deopt(result) {
            Err(StatorError::Internal("jit deopt".into()))
        } else {
            Ok(result)
        }
    }

    /// Returns the raw machine-code bytes stored in the `mmap`'d region.
    ///
    /// Used to seed a [`JitExecutableCode`](crate::bytecode::bytecode_array::JitExecutableCode)
    /// from the persistent cache without re-compiling.
    pub fn code_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` and `len` were set by a successful `mmap` in
        // `from_compiled`; the region remains valid until `Drop`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
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
    /// Mapping from register-file operand value to a callee-saved physical
    /// register used as a cache.  Loads from cached slots read the physical
    /// register directly; stores write both memory and the cached register.
    #[cfg(all(target_arch = "x86_64", unix))]
    cache_map: Vec<(u32, Reg64)>,
    /// Number of `CallUndefinedReceiver0` call sites in the bytecode.
    /// Used to allocate monomorphic inline cache slots in the frame.
    #[cfg(all(target_arch = "x86_64", unix))]
    mono_call_sites: i32,
    /// Counter used during emission to assign sequential cache-slot
    /// offsets to each `CallUndefinedReceiver0` site.
    #[cfg(all(target_arch = "x86_64", unix))]
    next_mono_site: i32,
    /// Total bytes reserved on the stack for monomorphic call cache
    /// slots (aligned to preserve RSP ≡ 0 mod 16 after sub).
    #[cfg(all(target_arch = "x86_64", unix))]
    mono_cache_bytes: i32,
    /// RBP offset of the first mono-cache slot's callee field.
    /// E.g. −40 means `[RBP − 40]` holds slot 0's callee_i64.
    #[cfg(all(target_arch = "x86_64", unix))]
    mono_cache_base: i32,
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
            #[cfg(all(target_arch = "x86_64", unix))]
            cache_map: Vec::new(),
            #[cfg(all(target_arch = "x86_64", unix))]
            mono_call_sites: 0,
            #[cfg(all(target_arch = "x86_64", unix))]
            next_mono_site: 0,
            #[cfg(all(target_arch = "x86_64", unix))]
            mono_cache_bytes: 0,
            #[cfg(all(target_arch = "x86_64", unix))]
            mono_cache_base: 0,
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
    /// When register caching is active, also pushes callee-saved cache
    /// registers and pre-loads them from the register file.
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
        // Push callee-saved registers used for register-file caching.
        #[cfg(all(target_arch = "x86_64", unix))]
        for &(_, phys_reg) in &self.cache_map {
            self.masm.push(phys_reg);
        }
        self.masm.mov_rr(Reg64::R14, Reg64::Rdi);
        // RSI carries the raw closure-context pointer (passed by execute).
        // Store in RBX (callee-saved) for use by context-slot stubs.
        self.masm.mov_rr(Reg64::Rbx, Reg64::Rsi);
        self.masm.xor_rr(Reg64::R12, Reg64::R12);
        // Pre-load cached registers from the register file.
        #[cfg(all(target_arch = "x86_64", unix))]
        for &(virt_reg, phys_reg) in &self.cache_map {
            let off = self.reg_offset(virt_reg);
            self.masm.mov_load_base_disp32(phys_reg, Reg64::R14, off);
        }

        // When there are CallUndefinedReceiver0 sites, load R15 with the
        // RT_PTRS TLS Cell address for zero-FFI monomorphic call dispatch.
        #[cfg(all(target_arch = "x86_64", unix))]
        if self.mono_call_sites > 0 {
            // After fixed pushes (5) + cache_map pushes (N):
            //   N even → RSP ≡ 0 mod 16 → call-ready
            //   N odd  → RSP ≡ 8 mod 16 → need padding
            let need_pad = self.cache_map.len() % 2 == 1;
            if need_pad {
                self.masm.push(Reg64::R11);
            }
            let addr = jit_runtime::jit_runtime_get_rt_ptrs_cell_addr as *const () as i64;
            self.masm.mov_ri(Reg64::R11, addr);
            self.masm.call_reg(Reg64::R11);
            self.masm.mov_rr(Reg64::R15, Reg64::Rax);
            if need_pad {
                self.masm.pop(Reg64::R11);
            }

            // Allocate and zero mono-cache slots.
            self.masm.sub_ri(Reg64::Rsp, self.mono_cache_bytes);
            self.masm.xor_rr(Reg64::R11, Reg64::R11);
            let slots = self.mono_cache_bytes / 8;
            for i in 0..slots {
                self.masm
                    .mov_store_base_disp32(Reg64::Rsp, i * 8, Reg64::R11);
            }
        }
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
        // Deallocate mono-cache slots.
        #[cfg(all(target_arch = "x86_64", unix))]
        if self.mono_cache_bytes > 0 {
            self.masm.add_ri(Reg64::Rsp, self.mono_cache_bytes);
        }
        // Pop cache registers in reverse push order.
        #[cfg(all(target_arch = "x86_64", unix))]
        for &(_, phys_reg) in self.cache_map.iter().rev() {
            self.masm.pop(phys_reg);
        }
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
    ///
    /// When register caching is active and `v` is cached in a physical
    /// register, emits a register-to-register move instead of a memory load.
    fn emit_load_reg(&mut self, dst: Reg64, v: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        if let Some(&(_, phys)) = self.cache_map.iter().find(|(vr, _)| *vr == v) {
            if dst != phys {
                self.masm.mov_rr(dst, phys);
            }
            return;
        }
        let off = self.reg_offset(v);
        self.masm.mov_load_base_disp32(dst, Reg64::R14, off);
    }

    /// Emit code to store `src` into register `v`.
    ///
    /// Always writes to memory (so the register file is up-to-date for deopt).
    /// When register caching is active, also updates the cached physical
    /// register.
    fn emit_store_reg(&mut self, v: u32, src: Reg64) {
        let off = self.reg_offset(v);
        self.masm.mov_store_base_disp32(Reg64::R14, off, src);
        #[cfg(all(target_arch = "x86_64", unix))]
        if let Some(&(_, phys)) = self.cache_map.iter().find(|(vr, _)| *vr == v)
            && src != phys
        {
            self.masm.mov_rr(phys, src);
        }
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
                && let Operand::ConstantPoolIdx(idx) = *instr.operand(0)
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

        let Operand::JumpOffset(delta) = *next.operand(0) else {
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

    // ── Codegen optimizations (x86-64 + unix only) ──────────────────────────

    /// Emit an optimized load of a Smi immediate into the accumulator (R12).
    ///
    /// Uses shorter instruction encodings when possible:
    /// - `xor r12d, r12d` for zero (3 bytes vs 7).
    /// - `mov r12d, imm32` for positive values (6 bytes vs 7).
    /// - `mov r12, imm64` (sign-extended i32) for negative values (7 bytes).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_lda_smi_optimized(&mut self, imm: i32) {
        if imm == 0 {
            self.masm.xor_rr(Reg64::R12, Reg64::R12);
        } else if imm > 0 {
            self.masm.mov_ri32(Reg64::R12, imm as u32);
        } else {
            self.masm.mov_ri(Reg64::R12, imm as i64);
        }
    }

    /// Emit an arithmetic operation (add/sub) with i64 overflow checking.
    ///
    /// On overflow, jumps to a deopt stub.  On success, commits the result
    /// from the scratch register (R11) into the accumulator (R12).
    ///
    /// `is_sub` selects between ADD and SUB.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_arith_with_overflow_check(&mut self, v: u32, bytecode_offset: u32, is_sub: bool) {
        self.emit_load_reg(Reg64::Rax, v);
        self.masm.mov_rr(Reg64::R11, Reg64::R12);
        if is_sub {
            self.masm.sub_rr(Reg64::R11, Reg64::Rax);
        } else {
            self.masm.add_rr(Reg64::R11, Reg64::Rax);
        }
        let mut overflow = Label::new();
        let mut done = Label::new();
        self.masm.jo(&mut overflow);
        // Fast path: commit result.
        self.masm.mov_rr(Reg64::R12, Reg64::R11);
        self.masm.jmp(&mut done);
        // Slow path: overflow → deopt.
        self.masm.bind_label(&mut overflow);
        self.emit_deopt(bytecode_offset);
        self.masm.bind_label(&mut done);
    }

    /// Emit a multiply with i64 overflow checking.
    ///
    /// On overflow, jumps to a deopt stub.  On success, commits the result
    /// from the scratch register (R11) into the accumulator (R12).
    #[cfg(all(target_arch = "x86_64", unix))]
    fn emit_mul_with_overflow_check(&mut self, v: u32, bytecode_offset: u32) {
        self.emit_load_reg(Reg64::Rax, v);
        self.masm.mov_rr(Reg64::R11, Reg64::R12);
        self.masm.imul_rr(Reg64::R11, Reg64::Rax);
        let mut overflow = Label::new();
        let mut done = Label::new();
        self.masm.jo(&mut overflow);
        self.masm.mov_rr(Reg64::R12, Reg64::R11);
        self.masm.jmp(&mut done);
        self.masm.bind_label(&mut overflow);
        self.emit_deopt(bytecode_offset);
        self.masm.bind_label(&mut done);
    }

    /// Analyse bytecodes to identify hot registers inside loop bodies and
    /// build a cache mapping from virtual register-file slots to spare
    /// callee-saved physical registers (RBX, R13, R15).
    ///
    /// Only the first (outermost) loop is considered.  At most 3 registers
    /// are cached.
    #[cfg(all(target_arch = "x86_64", unix))]
    fn analyze_register_caching(
        instructions: &[Instruction],
        byte_offsets: &[usize],
        n: usize,
    ) -> Vec<(u32, Reg64)> {
        const AVAILABLE: [Reg64; 3] = [Reg64::Rbx, Reg64::R13, Reg64::R15];

        // Find backward jumps to identify loop bodies.
        let mut best_loop: Option<(usize, usize)> = None;
        for (idx, instr) in instructions.iter().enumerate() {
            let delta = match instr.opcode {
                Opcode::JumpLoop => {
                    if let Operand::JumpOffset(d) = *instr.operand(0) {
                        d
                    } else {
                        continue;
                    }
                }
                Opcode::Jump => {
                    if let Operand::JumpOffset(d) = *instr.operand(0) {
                        if d >= 0 {
                            continue;
                        }
                        d
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let target_byte = jump_target_byte(idx, delta, byte_offsets);
            if let Ok(target_idx) = Self::resolve_target(target_byte, byte_offsets, n) {
                // Pick the loop with the most iterations (longest body).
                let body_len = idx - target_idx;
                if best_loop.is_none_or(|(_, prev_len)| body_len > prev_len) {
                    best_loop = Some((target_idx, body_len));
                }
            }
        }

        let (entry_idx, body_len) = match best_loop {
            Some(v) => v,
            None => return Vec::new(),
        };

        // Count register-file accesses within the loop body.
        let mut counts: Vec<(u32, usize)> = Vec::new();
        let end = (entry_idx + body_len + 1).min(n);
        for instr in &instructions[entry_idx..end] {
            for op in instr.operands() {
                if let Operand::Register(v) = op {
                    if let Some(entry) = counts.iter_mut().find(|(vr, _)| *vr == *v) {
                        entry.1 += 1;
                    } else {
                        counts.push((*v, 1));
                    }
                }
            }
        }

        // Sort by descending access count and take at most AVAILABLE.len().
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts
            .into_iter()
            .filter(|&(_, c)| c >= 2) // only cache if accessed at least twice
            .take(AVAILABLE.len())
            .enumerate()
            .map(|(i, (v, _))| (v, AVAILABLE[i]))
            .collect()
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

    /// Emit a monomorphic-inline-cache fast path (or fallback stub call) for
    /// `CallUndefinedReceiver0` bytecodes.
    ///
    /// When mono call sites were detected, emits a zero-FFI cache-hit path
    /// identical to the Maglev MIC: compare callee identity → set BA inline
    /// → allocate register file → direct call → restore BA.  On cache miss,
    /// populates the cache via [`jit_runtime_get_jit_entry`] and falls back
    /// to the full runtime stub when no JIT entry is available.
    #[allow(unused_variables)]
    fn emit_call_undefined_receiver0_stub(&mut self, callee_flat: i64, bytecode_offset: u32) {
        #[cfg(all(target_arch = "x86_64", unix))]
        if self.mono_call_sites > 0 {
            let byte_offset = (callee_flat as i32) * 8;

            // Claim a mono-cache slot.
            let site = self.next_mono_site;
            self.next_mono_site += 1;
            let slot_base = self.mono_cache_base - site * 40;
            let off_callee = slot_base;
            let off_entry = slot_base - 8;
            let off_ctx = slot_base - 16;
            let off_ba = slot_base - 24;
            let off_caller_ba = slot_base - 32;

            // Load callee into R11 (scratch).
            self.masm
                .mov_load_base_disp32(Reg64::R11, Reg64::R14, byte_offset);

            let mut cache_miss = Label::new();
            let mut done_label = Label::new();

            // ── Cache check ─────────────────────────────────────────────
            self.masm.cmp_rm(Reg64::R11, Reg64::Rbp, off_callee);
            self.masm.jne(&mut cache_miss);

            // ── Mono HIT: fully inline dispatch (zero FFI) ──────────────
            const RTPTRS_BYTECODE_OFF: i32 = 16;

            // Inline BA set (callee BA from mono cache).
            self.masm
                .mov_load_base_disp32(Reg64::R10, Reg64::R15, RTPTRS_BYTECODE_OFF);
            self.masm
                .mov_load_base_disp32(Reg64::Rcx, Reg64::Rbp, off_ba);
            self.masm.mov_store_base_disp32(Reg64::R10, 0, Reg64::Rcx);

            // Allocate 128-byte register file.
            self.masm.sub_ri(Reg64::Rsp, 128);

            // Zero register file with unrolled stores.
            self.masm.xor_rr(Reg64::Rax, Reg64::Rax);
            for i in 0..16i32 {
                self.masm
                    .mov_store_base_disp32(Reg64::Rsp, i * 8, Reg64::Rax);
            }

            // Load entry + ctx from mono cache.
            self.masm
                .mov_load_base_disp32(Reg64::R11, Reg64::Rbp, off_entry);
            self.masm
                .mov_load_base_disp32(Reg64::Rsi, Reg64::Rbp, off_ctx);

            // CALL entry point: RDI = register file, RSI = ctx_ptr.
            self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
            self.masm.call_reg(Reg64::R11);

            // Deallocate register file.
            self.masm.add_ri(Reg64::Rsp, 128);

            // Inline BA restore (from cached caller BA).
            self.masm
                .mov_load_base_disp32(Reg64::R10, Reg64::R15, RTPTRS_BYTECODE_OFF);
            self.masm
                .mov_load_base_disp32(Reg64::Rcx, Reg64::Rbp, off_caller_ba);
            self.masm.mov_store_base_disp32(Reg64::R10, 0, Reg64::Rcx);

            // Deopt check: RAX >= i32::MIN means valid result.
            self.masm.cmp_ri(Reg64::Rax, i32::MIN);
            self.masm.jcc(CondCode::GreaterEq, &mut done_label);

            // Callee deopt after inline dispatch: fall back to stub.
            self.masm
                .mov_load_base_disp32(Reg64::Rdi, Reg64::Rbp, off_callee);
            let mono_retry_addr =
                jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize;
            self.masm.mov_ri(Reg64::R11, mono_retry_addr as i64);
            self.masm.call_reg(Reg64::R11);
            self.masm.jmp(&mut done_label);

            // ── Cache miss: full jit_runtime_get_jit_entry path ─────────
            self.masm.bind_label(&mut cache_miss);

            self.masm.mov_rr(Reg64::Rdi, Reg64::R11); // callee
            // Push callee + padding (2 pushes → alignment preserved).
            self.masm.push(Reg64::Rdi);
            self.masm.push(Reg64::Rdi);

            let get_entry_addr = jit_runtime::jit_runtime_get_jit_entry as *const () as usize;
            self.masm.mov_ri(Reg64::R11, get_entry_addr as i64);
            self.masm.call_reg(Reg64::R11);

            // Check entry_point == 0 → stub fallback.
            let mut stub_fallback = Label::new();
            self.masm.test_rr(Reg64::Rax, Reg64::Rax);
            self.masm.je(&mut stub_fallback);

            // RAX = entry_point, RDX = ctx_ptr.
            self.masm.mov_rr(Reg64::R11, Reg64::Rax); // entry
            self.masm.mov_rr(Reg64::R10, Reg64::Rdx); // ctx

            // ── Populate mono cache ─────────────────────────────────────
            self.masm.mov_load_base_disp32(Reg64::Rax, Reg64::Rsp, 0);
            self.masm
                .mov_store_base_disp32(Reg64::Rbp, off_callee, Reg64::Rax);
            self.masm
                .mov_store_base_disp32(Reg64::Rbp, off_entry, Reg64::R11);
            self.masm
                .mov_store_base_disp32(Reg64::Rbp, off_ctx, Reg64::R10);

            // Read the current BA for caching.
            self.masm.push(Reg64::R11);
            self.masm.push(Reg64::R10);
            let ba_addr = jit_runtime::jit_runtime_read_current_ba as *const () as usize;
            self.masm.mov_ri(Reg64::R11, ba_addr as i64);
            self.masm.call_reg(Reg64::R11);
            self.masm
                .mov_store_base_disp32(Reg64::Rbp, off_ba, Reg64::Rax);
            self.masm.pop(Reg64::R10);
            self.masm.pop(Reg64::R11);

            // ── Direct-call path (cache miss only) ──────────────────────
            self.masm.sub_ri(Reg64::Rsp, 128);
            self.masm.xor_rr(Reg64::Rax, Reg64::Rax);
            for i in 0..16i32 {
                self.masm
                    .mov_store_base_disp32(Reg64::Rsp, i * 8, Reg64::Rax);
            }
            self.masm.mov_rr(Reg64::Rdi, Reg64::Rsp);
            self.masm.mov_rr(Reg64::Rsi, Reg64::R10);
            self.masm.call_reg(Reg64::R11);
            self.masm.add_ri(Reg64::Rsp, 128);

            // Finish direct call (restores BA, context, truncates heap).
            self.masm.mov_rr(Reg64::Rdi, Reg64::Rax);
            self.masm.mov_rr(Reg64::Rsi, Reg64::R15);
            let finish_addr = jit_runtime::jit_runtime_finish_direct_call_r15 as *const () as usize;
            self.masm.mov_ri(Reg64::R11, finish_addr as i64);
            self.masm.call_reg(Reg64::R11);

            // Cache caller's BA for future cache hits.
            {
                self.masm
                    .mov_load_base_disp32(Reg64::R10, Reg64::R15, RTPTRS_BYTECODE_OFF);
                self.masm.mov_load_base_disp32(Reg64::Rcx, Reg64::R10, 0);
                self.masm
                    .mov_store_base_disp32(Reg64::Rbp, off_caller_ba, Reg64::Rcx);
            }

            // If callee JIT returned JIT_DEOPT, fall back to runtime stub.
            self.masm.cmp_ri(Reg64::Rax, i32::MIN);
            let mut miss_ok = Label::new();
            self.masm.jcc(CondCode::GreaterEq, &mut miss_ok);
            self.masm.pop(Reg64::Rdi); // callee
            self.masm.add_ri(Reg64::Rsp, 8); // padding
            let miss_retry_addr =
                jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize;
            self.masm.mov_ri(Reg64::R11, miss_retry_addr as i64);
            self.masm.call_reg(Reg64::R11);
            self.masm.jmp(&mut done_label);

            self.masm.bind_label(&mut miss_ok);
            self.masm.add_ri(Reg64::Rsp, 16); // pop callee + padding
            self.masm.jmp(&mut done_label);

            // ── Stub fallback (get_jit_entry returned 0) ────────────────
            self.masm.bind_label(&mut stub_fallback);
            self.masm.pop(Reg64::R11); // padding
            self.masm.pop(Reg64::Rdi); // callee
            let stub_addr = jit_runtime::jit_runtime_call_undefined_receiver0 as *const () as usize;
            self.masm.mov_ri(Reg64::R11, stub_addr as i64);
            self.masm.call_reg(Reg64::R11);

            // ── Common exit ─────────────────────────────────────────────
            self.masm.bind_label(&mut done_label);

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

        // Analyse register usage in loop bodies and set up the cache map
        // before emitting the prologue (which pushes the cache registers).
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            // Count CallUndefinedReceiver0 sites for monomorphic inline cache.
            let call0_sites = instructions
                .iter()
                .filter(|i| i.opcode == Opcode::CallUndefinedReceiver0)
                .count() as i32;
            self.mono_call_sites = call0_sites;

            let mut cache_map = Self::analyze_register_caching(&instructions, &byte_offsets, n);
            // If there are call sites, reserve R15 for RT_PTRS instead of
            // register caching.
            if call0_sites > 0 {
                cache_map.retain(|&(_, reg)| reg != Reg64::R15);
            }
            self.cache_map = cache_map;

            // Compute mono cache layout.
            if call0_sites > 0 {
                const SLOT_BYTES: i32 = 40;
                let raw = call0_sites * SLOT_BYTES;
                let total_pushes = 32 + 8 * self.cache_map.len() as i32;
                let total = total_pushes + raw;
                let aligned_total = (total + 15) & !15;
                self.mono_cache_bytes = aligned_total - total_pushes;
                self.mono_cache_base = -(total_pushes + 8);
            }
        }

        self.emit_prologue();

        // Load all promoted globals into their register-file slots (once).
        self.emit_promoted_global_loads();

        // Use a while loop instead of for so that compile_instruction can
        // consume extra instructions (e.g. comparison + jump fusion).
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
                let Operand::Immediate(v) = *instr.operand(0) else {
                    return Err(bad_operand("LdaSmi", 0));
                };
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    self.emit_lda_smi_optimized(v);
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                {
                    self.masm.mov_ri(Reg64::R12, v as i64);
                }
            }
            Opcode::LdaSmiStar => {
                let Operand::Immediate(v) = *instr.operand(0) else {
                    return Err(bad_operand("LdaSmiStar", 0));
                };
                let Operand::Register(dst) = *instr.operand(1) else {
                    return Err(bad_operand("LdaSmiStar", 1));
                };
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    self.emit_lda_smi_optimized(v);
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                {
                    self.masm.mov_ri(Reg64::R12, v as i64);
                }
                self.emit_store_reg(dst, Reg64::R12);
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
                let Operand::ConstantPoolIdx(idx_cp) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("Ldar", 0));
                };
                self.emit_load_reg(Reg64::R12, v);
            }
            Opcode::LdarAddStar => {
                let Operand::Register(src) = *instr.operand(0) else {
                    return Err(bad_operand("LdarAddStar", 0));
                };
                let Operand::Register(add_reg) = *instr.operand(1) else {
                    return Err(bad_operand("LdarAddStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("LdarAddStar", 2));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(3) else {
                    return Err(bad_operand("LdarAddStar", 3));
                };
                self.emit_load_reg(Reg64::R12, src);
                self.emit_load_reg(Reg64::R11, add_reg);
                self.masm.add_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::LdarSubStar => {
                let Operand::Register(src) = *instr.operand(0) else {
                    return Err(bad_operand("LdarSubStar", 0));
                };
                let Operand::Register(sub_reg) = *instr.operand(1) else {
                    return Err(bad_operand("LdarSubStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("LdarSubStar", 2));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(3) else {
                    return Err(bad_operand("LdarSubStar", 3));
                };
                self.emit_load_reg(Reg64::R12, src);
                self.emit_load_reg(Reg64::R11, sub_reg);
                self.masm.sub_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::LdarMulStar => {
                let Operand::Register(src) = *instr.operand(0) else {
                    return Err(bad_operand("LdarMulStar", 0));
                };
                let Operand::Register(mul_reg) = *instr.operand(1) else {
                    return Err(bad_operand("LdarMulStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("LdarMulStar", 2));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(3) else {
                    return Err(bad_operand("LdarMulStar", 3));
                };
                self.emit_load_reg(Reg64::R12, src);
                self.emit_load_reg(Reg64::R11, mul_reg);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::Star => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("Star", 0));
                };
                self.emit_store_reg(v, Reg64::R12);
            }
            Opcode::Mov => {
                let Operand::Register(src) = *instr.operand(0) else {
                    return Err(bad_operand("Mov", 0));
                };
                let Operand::Register(dst) = *instr.operand(1) else {
                    return Err(bad_operand("Mov", 1));
                };
                self.emit_load_reg(Reg64::R11, src);
                self.emit_store_reg(dst, Reg64::R11);
            }

            // ── Arithmetic ───────────────────────────────────────────────────
            Opcode::Add => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("Add", 0));
                };
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    self.emit_arith_with_overflow_check(v, bytecode_offset, false);
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                {
                    self.emit_load_reg(Reg64::R11, v);
                    self.masm.add_rr(Reg64::R12, Reg64::R11);
                }
            }
            Opcode::Sub => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("Sub", 0));
                };
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    self.emit_arith_with_overflow_check(v, bytecode_offset, true);
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                {
                    self.emit_load_reg(Reg64::R11, v);
                    self.masm.sub_rr(Reg64::R12, Reg64::R11);
                }
            }
            Opcode::Mul => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("Mul", 0));
                };
                #[cfg(all(target_arch = "x86_64", unix))]
                {
                    self.emit_mul_with_overflow_check(v, bytecode_offset);
                }
                #[cfg(not(all(target_arch = "x86_64", unix)))]
                {
                    self.emit_load_reg(Reg64::R11, v);
                    self.masm.imul_rr(Reg64::R12, Reg64::R11);
                }
            }
            Opcode::Inc => {
                self.masm.add_ri(Reg64::R12, 1);
            }
            Opcode::Dec => {
                self.masm.sub_ri(Reg64::R12, 1);
            }
            Opcode::AddSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("AddSmi", 0));
                };
                self.masm.add_ri(Reg64::R12, imm);
            }
            Opcode::AddSmiStar => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("AddSmiStar", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("AddSmiStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("AddSmiStar", 2));
                };
                self.masm.add_ri(Reg64::R12, imm);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::SubSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("SubSmi", 0));
                };
                self.masm.sub_ri(Reg64::R12, imm);
            }
            Opcode::MulSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("MulSmi", 0));
                };
                self.masm.mov_ri(Reg64::R11, imm as i64);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::MulSmiStar => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("MulSmiStar", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("MulSmiStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("MulSmiStar", 2));
                };
                self.masm.mov_ri(Reg64::R11, imm as i64);
                self.masm.imul_rr(Reg64::R12, Reg64::R11);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::Negate => {
                self.masm.neg_r(Reg64::R12);
            }
            Opcode::IncStar => {
                let Operand::FeedbackSlot(_slot) = *instr.operand(0) else {
                    return Err(bad_operand("IncStar", 0));
                };
                let Operand::Register(dst) = *instr.operand(1) else {
                    return Err(bad_operand("IncStar", 1));
                };
                self.masm.add_ri(Reg64::R12, 1);
                self.emit_store_reg(dst, Reg64::R12);
            }

            // ── Bitwise ─────────────────────────────────────────────────────
            Opcode::BitwiseOr => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseOr", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.or_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseAnd => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseAnd", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.and_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseXor => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseXor", 0));
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.xor_rr(Reg64::R12, Reg64::R11);
            }
            Opcode::BitwiseOrSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseOrSmi", 0));
                };
                self.masm.or_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseAndSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseAndSmi", 0));
                };
                self.masm.and_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseXorSmi => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("BitwiseXorSmi", 0));
                };
                self.masm.xor_ri(Reg64::R12, imm);
            }
            Opcode::BitwiseNot => {
                self.masm.not_r(Reg64::R12);
            }

            // ── Comparisons ──────────────────────────────────────────────────
            Opcode::TestLessThan => {
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestLessThanJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestLessThanJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestLessThanJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestGreaterThanJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestGreaterThanJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestGreaterThanJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestEqualJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestEqualJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestEqualJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
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
            Opcode::TestNotEqualJump => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestNotEqualJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestNotEqualJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestNotEqualJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
                    return Err(bad_operand("TestNotEqualJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::NotEqual)
                } else {
                    CondCode::NotEqual
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::TestEqualStrictJump => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestEqualStrictJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestEqualStrictJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestEqualStrictJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
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
            Opcode::TestLessThanOrEqualJump => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestLessThanOrEqualJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestLessThanOrEqualJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestLessThanOrEqualJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
                    return Err(bad_operand("TestLessThanOrEqualJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::LessEq)
                } else {
                    CondCode::LessEq
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::TestGreaterThanOrEqualJump => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("TestGreaterThanOrEqualJump", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("TestGreaterThanOrEqualJump", 1));
                };
                let Operand::JumpOffset(delta) = *instr.operand(2) else {
                    return Err(bad_operand("TestGreaterThanOrEqualJump", 2));
                };
                let Operand::Flag(is_true_flag) = *instr.operand(3) else {
                    return Err(bad_operand("TestGreaterThanOrEqualJump", 3));
                };
                let target = Self::resolve_target(
                    jump_target_byte(idx, delta, byte_offsets),
                    byte_offsets,
                    n,
                )?;
                let fused_cc = if is_true_flag == 0 {
                    negate_cc(CondCode::GreaterEq)
                } else {
                    CondCode::GreaterEq
                };
                self.emit_load_reg(Reg64::R11, v);
                self.masm.cmp_rr(Reg64::R12, Reg64::R11);
                self.emit_cond_jump(fused_cc, target);
            }
            Opcode::SubSmiStar => {
                let Operand::Immediate(imm) = *instr.operand(0) else {
                    return Err(bad_operand("SubSmiStar", 0));
                };
                let Operand::FeedbackSlot(_slot) = *instr.operand(1) else {
                    return Err(bad_operand("SubSmiStar", 1));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("SubSmiStar", 2));
                };
                self.masm.sub_ri(Reg64::R12, imm);
                self.emit_store_reg(dst, Reg64::R12);
            }
            Opcode::TestGreaterThan => {
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let Operand::JumpOffset(delta) = *instr.operand(0) else {
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
                let feedback_slot = match instr.operand_at(1) {
                    Some(Operand::FeedbackSlot(s)) => i64::from(*s),
                    _ => -1,
                };
                let capacity = match instr.operand_at(2) {
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
                let Operand::Register(obj_v) = *instr.operand(0) else {
                    return Err(bad_operand("LdaNamedProperty", 0));
                };
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(1) else {
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
                let Operand::Register(obj_v) = *instr.operand(0) else {
                    return Err(bad_operand(opname, 0));
                };
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(1) else {
                    return Err(bad_operand(opname, 1));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_sta_named_property_stub(obj_flat, name_idx, bytecode_offset);
            }

            Opcode::LdaKeyedProperty => {
                let Operand::Register(obj_v) = *instr.operand(0) else {
                    return Err(bad_operand("LdaKeyedProperty", 0));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_lda_keyed_property_stub(obj_flat, bytecode_offset);
            }

            Opcode::StaKeyedProperty => {
                let Operand::Register(obj_v) = *instr.operand(0) else {
                    return Err(bad_operand("StaKeyedProperty", 0));
                };
                let Operand::Register(key_v) = *instr.operand(1) else {
                    return Err(bad_operand("StaKeyedProperty", 1));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                let key_flat = self.reg_flat_index(key_v) as i64;
                self.emit_sta_keyed_property_stub(obj_flat, key_flat, bytecode_offset);
            }

            Opcode::StaInArrayLiteral => {
                let Operand::Register(arr_v) = *instr.operand(0) else {
                    return Err(bad_operand("StaInArrayLiteral", 0));
                };
                let Operand::Register(idx_v) = *instr.operand(1) else {
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
                let Operand::Register(callee_v) = *instr.operand(0) else {
                    return Err(bad_operand("CallUndefinedReceiver0", 0));
                };
                let callee_flat = self.reg_flat_index(callee_v) as i64;
                self.emit_promoted_global_stores();
                self.emit_call_undefined_receiver0_stub(callee_flat, bytecode_offset);
                self.emit_promoted_global_loads();
            }

            Opcode::CallUndefinedReceiver1 => {
                let Operand::Register(callee_v) = *instr.operand(0) else {
                    return Err(bad_operand("CallUndefinedReceiver1", 0));
                };
                let Operand::Register(arg1_v) = *instr.operand(1) else {
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
                let Operand::ConstantPoolIdx(slot) = *instr.operand(0) else {
                    return Err(bad_operand("LdaCurrentContextSlot", 0));
                };
                self.emit_lda_context_slot_stub(slot, bytecode_offset);
            }

            Opcode::StaCurrentContextSlot => {
                let Operand::ConstantPoolIdx(slot) = *instr.operand(0) else {
                    return Err(bad_operand("StaCurrentContextSlot", 0));
                };
                self.emit_sta_context_slot_stub(slot, bytecode_offset);
            }

            Opcode::LdaContextSlot | Opcode::LdaImmutableContextSlot => {
                let Operand::Register(ctx_v) = *instr.operand(0) else {
                    return Err(bad_operand("LdaContextSlot", 0));
                };
                let Operand::ConstantPoolIdx(slot) = *instr.operand(1) else {
                    return Err(bad_operand("LdaContextSlot", 1));
                };
                let ctx_flat = self.reg_flat_index(ctx_v) as i64;
                self.emit_runtime_stub(instr.opcode, i64::from(slot), ctx_flat, bytecode_offset);
            }

            Opcode::StaContextSlot => {
                let Operand::Register(ctx_v) = *instr.operand(0) else {
                    return Err(bad_operand("StaContextSlot", 0));
                };
                let Operand::ConstantPoolIdx(slot) = *instr.operand(1) else {
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
                let Operand::Register(ctor_v) = *instr.operand(0) else {
                    return Err(bad_operand("Construct", 0));
                };
                let Operand::Register(args_start_v) = *instr.operand(1) else {
                    return Err(bad_operand("Construct", 1));
                };
                let Operand::RegisterCount(arg_count) = *instr.operand(2) else {
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
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(0) else {
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

            Opcode::LdaGlobalStar => {
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(0) else {
                    return Err(bad_operand("LdaGlobalStar", 0));
                };
                let Operand::Register(dst) = *instr.operand(2) else {
                    return Err(bad_operand("LdaGlobalStar", 2));
                };
                #[allow(unused_variables)]
                if let Some(flat) = self.promoted_slot_for(name_idx) {
                    #[cfg(all(target_arch = "x86_64", unix))]
                    {
                        self.masm.mov_load_base_disp32(
                            Reg64::R12,
                            Reg64::R14,
                            Self::promoted_offset(flat),
                        );
                        self.emit_store_reg(dst, Reg64::R12);
                    }
                    #[cfg(not(all(target_arch = "x86_64", unix)))]
                    {
                        self.emit_deopt(bytecode_offset);
                    }
                } else {
                    self.emit_lda_global_stub(name_idx, bytecode_offset);
                    self.emit_store_reg(dst, Reg64::R12);
                }
            }

            Opcode::StaGlobal => {
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(0) else {
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
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(0) else {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand(opname, 0));
                };
                let lhs_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(instr.opcode, lhs_flat, 0, bytecode_offset);
            }

            // ── TestTypeOf runtime stub ──────────────────────────────────────
            Opcode::TestTypeOf => {
                let Operand::Flag(type_flag) = *instr.operand(0) else {
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
                let Operand::Register(rhs_v) = *instr.operand(0) else {
                    return Err(bad_operand("TestInstanceOf", 0));
                };
                let rhs_flat = self.reg_flat_index(rhs_v) as i64;
                self.emit_runtime_stub(Opcode::TestInstanceOf, rhs_flat, 0, bytecode_offset);
            }

            // ── TestIn runtime stub ─────────────────────────────────────────
            Opcode::TestIn => {
                let Operand::Register(obj_v) = *instr.operand(0) else {
                    return Err(bad_operand("TestIn", 0));
                };
                let obj_flat = self.reg_flat_index(obj_v) as i64;
                self.emit_runtime_stub(Opcode::TestIn, obj_flat, 0, bytecode_offset);
            }

            // ── ThrowReferenceErrorIfHole runtime stub ──────────────────────
            Opcode::ThrowReferenceErrorIfHole => {
                let Operand::ConstantPoolIdx(name_idx) = *instr.operand(0) else {
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
                let Operand::ConstantPoolIdx(cp_idx) = *instr.operand(0) else {
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
                let slot_count = match instr.operand_at(1) {
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
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("PushContext", 0));
                };
                let reg_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(Opcode::PushContext, reg_flat, 0, bytecode_offset);
            }

            Opcode::PopContext => {
                let Operand::Register(v) = *instr.operand(0) else {
                    return Err(bad_operand("PopContext", 0));
                };
                let reg_flat = self.reg_flat_index(v) as i64;
                self.emit_runtime_stub(Opcode::PopContext, reg_flat, 0, bytecode_offset);
            }

            // ── Div / Mod runtime stubs ─────────────────────────────────────
            Opcode::Div => {
                let Operand::Register(lhs_v) = *instr.operand(0) else {
                    return Err(bad_operand("Div", 0));
                };
                let lhs_flat = self.reg_flat_index(lhs_v) as i64;
                self.emit_runtime_stub(Opcode::Div, lhs_flat, 0, bytecode_offset);
            }

            Opcode::Mod => {
                let Operand::Register(lhs_v) = *instr.operand(0) else {
                    return Err(bad_operand("Mod", 0));
                };
                let lhs_flat = self.reg_flat_index(lhs_v) as i64;
                self.emit_runtime_stub(Opcode::Mod, lhs_flat, 0, bytecode_offset);
            }

            // ── CallProperty0 specialized stub ─────────────────────────────
            Opcode::CallProperty0 => {
                let Operand::Register(callee_v) = *instr.operand(0) else {
                    return Err(bad_operand("CallProperty0", 0));
                };
                let Operand::Register(receiver_v) = *instr.operand(1) else {
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
                let Operand::Register(callee_v) = *instr.operand(0) else {
                    return Err(bad_operand("CallProperty1", 0));
                };
                let Operand::Register(receiver_v) = *instr.operand(1) else {
                    return Err(bad_operand("CallProperty1", 1));
                };
                let Operand::Register(arg0_v) = *instr.operand(2) else {
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
                let imm = match instr.operand_at(0) {
                    Some(Operand::Immediate(n)) => i64::from(*n),
                    _ => return Err(bad_operand("DivSmi", 0)),
                };
                self.emit_runtime_stub(Opcode::DivSmi, imm, 0, bytecode_offset);
            }

            Opcode::ModSmi => {
                let imm = match instr.operand_at(0) {
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
        // Overflow is never used in compare-branch fusion.
        CondCode::Overflow => CondCode::Overflow,
        CondCode::AboveEq => CondCode::Below,
        CondCode::Below => CondCode::AboveEq,
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
        // Exp still emits a deopt entry (no runtime stub for it).
        let ba = bytecode(vec![
            Instruction::new_unchecked(
                Opcode::Exp,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
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
                    Opcode::Exp,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
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

    /// `jit_to_jsvalue` must decode `JIT_TRUE` and `JIT_FALSE` to
    /// `JsValue::Boolean` so that boolean stores through keyed stubs
    /// do not deoptimize.
    #[test]
    fn test_jit_to_jsvalue_booleans() {
        assert_eq!(jit_to_jsvalue(JIT_TRUE), Some(JsValue::Boolean(true)));
        assert_eq!(jit_to_jsvalue(JIT_FALSE), Some(JsValue::Boolean(false)));
    }

    /// `decode_non_heap_value_fast` must handle `JIT_TRUE` / `JIT_FALSE`
    /// as booleans — this is the hot path for sieve-style benchmarks
    /// where `sta_keyed` stores booleans into arrays.
    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_decode_non_heap_value_fast_booleans() {
        use super::jit_runtime::decode_non_heap_value_fast;
        assert_eq!(
            decode_non_heap_value_fast(JIT_TRUE),
            Some(JsValue::Boolean(true))
        );
        assert_eq!(
            decode_non_heap_value_fast(JIT_FALSE),
            Some(JsValue::Boolean(false))
        );
    }

    /// `decode_non_heap_value_fast` must still handle Smi and Undefined.
    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_decode_non_heap_value_fast_smi_and_undefined() {
        use super::jit_runtime::decode_non_heap_value_fast;
        assert_eq!(decode_non_heap_value_fast(0), Some(JsValue::Smi(0)));
        assert_eq!(decode_non_heap_value_fast(42), Some(JsValue::Smi(42)));
        assert_eq!(decode_non_heap_value_fast(-1), Some(JsValue::Smi(-1)));
        assert_eq!(
            decode_non_heap_value_fast(JIT_UNDEFINED),
            Some(JsValue::Undefined)
        );
    }

    /// `encode_array_element` must roundtrip booleans through the JIT
    /// encoding so that `lda_keyed` returns the correct boolean values.
    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_encode_array_element_booleans() {
        use super::jit_runtime::encode_array_element;
        assert_eq!(
            encode_array_element(&JsValue::Boolean(true)),
            Some(JIT_TRUE)
        );
        assert_eq!(
            encode_array_element(&JsValue::Boolean(false)),
            Some(JIT_FALSE)
        );
    }

    /// Boolean JIT roundtrip: encoding then decoding must recover the
    /// original `JsValue::Boolean`.
    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn test_boolean_jit_roundtrip() {
        use super::jit_runtime::{decode_non_heap_value_fast, encode_array_element};
        for b in [true, false] {
            let original = JsValue::Boolean(b);
            let encoded = encode_array_element(&original).expect("encode boolean");
            let decoded = decode_non_heap_value_fast(encoded).expect("decode boolean");
            assert_eq!(decoded, original);
        }
    }
}

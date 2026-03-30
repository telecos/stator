//! [`BytecodeArray`] — the immutable, compact bytecode representation used by
//! the Stator VM interpreter.
//!
//! A [`BytecodeArray`] bundles together:
//!
//! - The raw bytecode stream (`Vec<u8>`) produced by the compiler.
//! - A **constant pool** holding all literals (numbers, strings, booleans)
//!   referenced by index from [`bytecodes::Opcode::LdaConstant`] instructions.
//! - Interpreter-level metadata: `frame_size` (number of virtual registers
//!   needed) and `parameter_count`.
//! - Optional **source-position table** that maps bytecode offsets back to
//!   source line/column pairs for stack traces and debugger support.
//! - A **feedback metadata** descriptor that lists the [`FeedbackSlotKind`] for
//!   every inline-cache slot allocated by the compiler.
//!
//! # Example
//!
//! ```
//! use stator_core::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry};
//! use stator_core::bytecode::bytecodes::{Instruction, Operand, Opcode, encode};
//! use stator_core::bytecode::feedback::FeedbackMetadata;
//!
//! // Build a tiny function: load constant 0 (42.0), return.
//! let instructions = vec![
//!     Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
//!     Instruction::new_unchecked(Opcode::Return, vec![]),
//! ];
//! let bytes = encode(&instructions);
//!
//! let pool = vec![ConstantPoolEntry::Number(42.0)];
//! let array = BytecodeArray::new(bytes, pool, 1, 0, vec![], FeedbackMetadata::empty(), vec![]);
//!
//! assert_eq!(array.parameter_count(), 0);
//! assert_eq!(array.frame_size(), 1);
//! assert_eq!(array.constant_pool().len(), 1);
//!
//! let decoded = array.instructions().expect("valid bytecode");
//! assert_eq!(decoded.len(), 2);
//! ```

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::bytecode::feedback::{FeedbackMetadata, FeedbackVector, InlineCacheState};
use crate::bytecode::{
    bytecodes::{self, Instruction, Operand},
    peephole,
};
use crate::compiler::turbofan::TurbofanCompiledCode;
use crate::error::{StatorError, StatorResult};
use crate::objects::property_map::{ObjectLiteralTemplate, PropertyMap};
use crate::objects::value::JsContext;

// ─────────────────────────────────────────────────────────────────────────────
// HandlerTableEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in a function's exception handler table.
///
/// Each entry describes a contiguous range of bytecode instructions (by
/// zero-based instruction *index* in the pre-decoded list) that is protected
/// by a catch or finally handler.
///
/// When the interpreter encounters a `Throw` or `ReThrow` instruction it walks
/// the handler table to find the first entry whose `[try_start, try_end)` range
/// contains the current program counter.  The innermost handler always appears
/// earlier in the table (it is pushed before outer handlers during compilation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerTableEntry {
    /// Instruction index of the first instruction covered by this handler
    /// (inclusive).
    pub try_start: u32,
    /// Instruction index one past the last instruction covered by this handler
    /// (exclusive).
    pub try_end: u32,
    /// Instruction index of the handler entry point (first instruction of the
    /// catch or finally block).
    pub handler: u32,
    /// `true` for a `finally` handler; `false` for a `catch` handler.
    ///
    /// When `true` the interpreter saves the thrown value before entering the
    /// handler so the finally block can re-throw it with `ReThrow`.
    pub is_finally: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstantPoolEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in a function's constant pool.
///
/// The bytecode instruction [`bytecodes::Opcode::LdaConstant`] references
/// these by zero-based index.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantPoolEntry {
    /// A 64-bit IEEE 754 floating-point number (covers all JS numbers).
    Number(f64),
    /// An interned string literal.
    String(String),
    /// A boolean literal (`true` / `false`).
    Boolean(bool),
    /// The JavaScript `null` literal.
    Null,
    /// The JavaScript `undefined` literal.
    Undefined,
    /// A BigInt literal (128-bit signed integer).
    BigInt(i128),
    /// A compiled nested function or closure.
    Function(Rc<BytecodeArray>),
    /// A template-literal descriptor for [`Opcode::GetTemplateObject`](super::bytecodes::Opcode::GetTemplateObject).
    ///
    /// Holds the cooked strings (`None` when the segment has an invalid escape)
    /// and the raw strings (backslash sequences preserved).
    TemplateObject {
        /// Cooked template strings.
        cooked: Vec<Option<String>>,
        /// Raw template strings.
        raw: Vec<String>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// SourcePosition
// ─────────────────────────────────────────────────────────────────────────────

/// Maps a bytecode offset to a location in the original JavaScript source.
///
/// The source-position table is a sorted, sparse list of `SourcePosition`
/// entries.  Any bytecode offset between two entries is attributed to the
/// earlier entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    /// Byte offset within the encoded [`BytecodeArray::bytecodes`] slice.
    pub bytecode_offset: u32,
    /// 1-based source line number.
    pub line: u32,
    /// 1-based source column number.
    pub column: u32,
}

impl SourcePosition {
    /// Construct a new `SourcePosition`.
    pub fn new(bytecode_offset: u32, line: u32, column: u32) -> Self {
        Self {
            bytecode_offset,
            line,
            column,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BytecodeArray
// ─────────────────────────────────────────────────────────────────────────────

/// Shared JIT code cache stored in a [`BytecodeArray`].
///
/// On x86-64 Unix this stores a [`CachedExecutableCode`] that owns a
/// persistent `mmap`'d page of executable memory, eliminating per-call
/// `mmap`/`munmap` overhead.  On other platforms the cache stores raw bytes
/// and the register-file slot count.
///
/// The outer [`Rc`] allows all clones of a [`BytecodeArray`] to share the
/// same cache without copying.
#[cfg(all(target_arch = "x86_64", unix))]
type JitCodeCache = Rc<RefCell<Option<crate::compiler::baseline::compiler::CachedExecutableCode>>>;

/// Shared JIT code cache stored in a [`BytecodeArray`].
///
/// On non-JIT platforms this is a no-op placeholder.
#[cfg(not(all(target_arch = "x86_64", unix)))]
type JitCodeCache = Rc<RefCell<Option<(Vec<u8>, usize)>>>;

/// Persistent executable JIT code region.
///
/// Holds an `mmap`'d region of executable memory that persists across calls,
/// avoiding the cost of `mmap`/`munmap` per invocation (two syscalls per call).
/// Created lazily on first JIT execution and freed when the last
/// [`BytecodeArray`] clone referencing it is dropped.
#[cfg(all(target_arch = "x86_64", unix))]
#[derive(Debug)]
pub struct JitExecutableCode {
    /// Pointer to `mmap`'d executable memory.
    ptr: *mut u8,
    /// Size of the `mmap`'d region in bytes.
    size: usize,
    /// Number of `i64` register-file slots the JIT code expects.
    pub register_file_slots: usize,
}

#[cfg(all(target_arch = "x86_64", unix))]
impl JitExecutableCode {
    /// Create a new executable code region from raw machine code bytes.
    ///
    /// Allocates a read-write-execute memory region via `mmap`, copies the
    /// code into it, and returns the persistent handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure `code` contains valid x86-64 machine code
    /// following the JIT calling convention.
    pub unsafe fn new(code: &[u8], register_file_slots: usize) -> Option<Self> {
        use std::ptr;
        if code.is_empty() {
            return None;
        }
        let size = code.len();
        // SAFETY: arguments are valid; MAP_FAILED is checked below.
        let mem = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mem == libc::MAP_FAILED {
            return None;
        }
        // SAFETY: `mem` is valid, page-aligned, and sized for `size` bytes.
        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), mem.cast::<u8>(), size);
        }
        Some(Self {
            ptr: mem.cast::<u8>(),
            size,
            register_file_slots,
        })
    }

    /// Execute the cached JIT code with the given register-file arguments.
    ///
    /// # Safety
    ///
    /// The stored code must still be valid x86-64 machine code that follows
    /// the JIT calling convention (`extern "C" fn(*mut i64) -> i64`).
    pub unsafe fn execute(&self, args: &[i64], ctx_ptr: i64) -> i64 {
        let n = self.register_file_slots;

        // Fast path: small register files use a stack-allocated array,
        // completely avoiding the TLS lookup + RefCell borrow.
        if n <= 16 {
            let mut regs = [0i64; 16];
            for (i, &v) in args.iter().enumerate().take(n) {
                regs[i] = v;
            }
            // SAFETY: `self.ptr` contains valid x86-64 JIT code.
            let f: extern "C" fn(*mut i64, i64) -> i64 = unsafe { std::mem::transmute(self.ptr) };
            return f(regs.as_mut_ptr(), ctx_ptr);
        }

        // Large register files: reuse pooled Vec via TLS.
        thread_local! {
            static REG_FILE: std::cell::RefCell<Vec<i64>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }
        REG_FILE.with(|pool| {
            let mut regs = pool.borrow_mut();
            regs.clear();
            regs.resize(n, 0);
            for (i, &v) in args.iter().enumerate().take(n) {
                regs[i] = v;
            }
            // SAFETY: `self.ptr` contains valid x86-64 JIT code.
            // Second parameter (RSI) carries the raw context pointer.
            let f: extern "C" fn(*mut i64, i64) -> i64 = unsafe { std::mem::transmute(self.ptr) };
            f(regs.as_mut_ptr(), ctx_ptr)
        })
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
impl Drop for JitExecutableCode {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.size > 0 {
            // SAFETY: `self.ptr` is a valid `mmap`'d region of `self.size` bytes.
            unsafe {
                libc::munmap(self.ptr.cast(), self.size);
            }
        }
    }
}

// SAFETY: JitExecutableCode is only accessed from the interpreter's single
// thread.  The mmap'd memory is read-only after initial copy.
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Send for JitExecutableCode {}
#[cfg(all(target_arch = "x86_64", unix))]
unsafe impl Sync for JitExecutableCode {}

/// Shared persistent executable JIT code cache.
///
/// Created lazily on first baseline JIT execution.  All clones of a
/// [`BytecodeArray`] share the same cache via [`Rc`].
#[cfg(all(target_arch = "x86_64", unix))]
pub type JitExecutableCache = Rc<RefCell<Option<JitExecutableCode>>>;
/// Stub type for platforms without JIT support.
#[cfg(not(all(target_arch = "x86_64", unix)))]
pub type JitExecutableCache = Rc<RefCell<Option<()>>>;

/// Shared decoded bytecode cache stored in a [`BytecodeArray`].
///
/// The cache is filled on first decode and then shared by all clones of the
/// bytecode array so repeated function calls avoid re-decoding the same
/// bytecode stream.
pub(crate) type JumpTargetMap = Vec<Option<usize>>;
type DecodedBytecode = (Vec<Instruction>, Vec<usize>, JumpTargetMap);
type DecodedBytecodeRef<'a> = (&'a [Instruction], &'a [usize], &'a [Option<usize>]);
type DecodedBytecodeCache = Rc<OnceCell<Rc<DecodedBytecode>>>;
type SharedFeedbackVector = Rc<RefCell<FeedbackVector>>;

/// Shared Maglev JIT code cache stored in a [`BytecodeArray`].
///
/// On x86-64 Unix this stores a [`CachedExecutableCode`] that owns a
/// persistent `mmap`'d page of executable memory.  Uses [`Arc`] + [`Mutex`]
/// so that the Maglev background compilation thread can write the compiled
/// code into the cache while the interpreter runs on the main thread.
#[cfg(all(target_arch = "x86_64", unix))]
pub type MaglevJitCodeCache =
    Arc<Mutex<Option<crate::compiler::baseline::compiler::CachedExecutableCode>>>;

/// Shared Maglev JIT code cache stored in a [`BytecodeArray`].
///
/// On non-JIT platforms this stores raw bytes and register-file slot count.
#[cfg(not(all(target_arch = "x86_64", unix)))]
pub type MaglevJitCodeCache = Arc<Mutex<Option<(Vec<u8>, usize)>>>;

/// Persistent executable Maglev code cache shared among clones of a
/// [`BytecodeArray`].
///
/// On x86-64 Unix, this wraps a [`CachedMaglevCode`] that keeps a
/// persistent `mmap`'d page.  Lazily initialised from the raw code bytes
/// in [`MaglevJitCodeCache`] on first execution.
#[cfg(all(target_arch = "x86_64", unix))]
pub type MaglevExecutableCache =
    Rc<RefCell<Option<crate::compiler::maglev::codegen::CachedMaglevCode>>>;
/// Stub type for platforms without JIT support.
#[cfg(not(all(target_arch = "x86_64", unix)))]
pub type MaglevExecutableCache = Rc<RefCell<Option<()>>>;

/// Shared Turbofan JIT code cache stored in a [`BytecodeArray`].
///
/// Stores a fully-compiled [`TurbofanCompiledCode`] produced by the Turbofan
/// background thread.  Uses [`Arc`] + [`Mutex`] so the background thread can
/// write the code while the interpreter runs on the main thread.
pub type TurbofanJitCodeCache = Arc<Mutex<Option<TurbofanCompiledCode>>>;

/// Invocation-count threshold that triggers baseline JIT compilation.
///
/// When a function's `invocation_count` reaches this value (3 calls) the
/// interpreter requests a baseline-compiled version; all subsequent calls
/// that can be represented in the current JIT tier execute via native code.
pub const TIERING_THRESHOLD: u32 = 10;

/// Invocation-count threshold that triggers Maglev JIT compilation.
///
/// When a function's `invocation_count` reaches this value (20 calls) and
/// baseline JIT code is already present, the interpreter schedules a
/// background Maglev compilation.  Once compilation finishes the cached
/// Maglev code replaces the baseline tier for future calls.
pub const MAGLEV_TIERING_THRESHOLD: u32 = 50;

/// Invocation-count threshold that triggers Turbofan (Cranelift optimising)
/// JIT compilation.
///
/// When a function's `invocation_count` reaches this value (100 calls) a
/// background Turbofan compilation is scheduled.  Turbofan runs the full
/// Maglev graph builder followed by Cranelift CLIF lowering and
/// optimisation, producing code that is expected to reach within 90 % of
/// peak throughput.
pub const TURBOFAN_TIERING_THRESHOLD: u32 = 1_000;

/// Shared, lazily-allocated megamorphic IC state persisted on a
/// [`BytecodeArray`] across invocations.
type SharedMegamorphicIc = Rc<RefCell<Option<Box<crate::interpreter::MegamorphicIc>>>>;

/// Shared, lazily-allocated prototype-chain IC state.
type SharedProtoLoadIc = Rc<RefCell<Option<Box<HashMap<u32, crate::interpreter::ProtoLoadIc>>>>>;

/// Shared, lazily-allocated global-variable IC state.
type SharedGlobalIc = Rc<RefCell<Option<Box<HashMap<u32, (usize, u64)>>>>>;

/// An immutable, compact representation of the bytecode for a single
/// JavaScript function.
#[derive(Debug, Clone)]
pub struct BytecodeArray {
    /// The encoded bytecode stream.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same allocation
    /// instead of copying the entire byte buffer.
    bytecodes: Rc<[u8]>,
    /// Literals referenced by [`bytecodes::Opcode::LdaConstant`].
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same pool.
    constant_pool: Rc<[ConstantPoolEntry]>,
    /// Number of virtual registers (locals + temporaries) required.
    frame_size: u32,
    /// Number of formal parameters declared by the function.
    parameter_count: u32,
    /// `Function.prototype.length` metadata for this function.
    function_length: u32,
    /// Declared or inferred function name.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same string.
    function_name: Rc<str>,
    /// Optional source text used by `Function.prototype.toString()`.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same source text.
    source_text: Option<Rc<str>>,
    /// Visible binding-to-register mapping for direct `eval()`.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same map.
    binding_registers: Rc<HashMap<String, i32>>,
    /// Sparse mapping from bytecode offsets to source locations.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same table.
    source_positions: Rc<[SourcePosition]>,
    /// Compile-time description of all inline-cache feedback slots.
    ///
    /// Wrapped in [`Rc`] so that closure clones share the same metadata.
    feedback_metadata: Rc<FeedbackMetadata>,
    /// Runtime inline-cache feedback shared by all clones of this function.
    ///
    /// Higher JIT tiers snapshot this vector when compiling so optimisation
    /// decisions reflect observed runtime behaviour rather than empty metadata.
    feedback_vector: SharedFeedbackVector,
    /// Per-function exception handler table (pre-peephole instruction indices).
    handler_table: Rc<Vec<HandlerTableEntry>>,
    /// Lazily-populated remapped handler table (post-peephole instruction
    /// indices).  Populated on first decode when the peephole pass compacts
    /// instructions.
    handler_table_remapped: Rc<OnceCell<Rc<Vec<HandlerTableEntry>>>>,
    /// Lazily-populated decoded instruction cache shared across clones.
    cached_decode: DecodedBytecodeCache,
    /// Cached template objects keyed by bytecode offset.
    ///
    /// Tagged template sites must reuse the same frozen template object across
    /// executions of the same compiled function. Clones of this bytecode array
    /// share the cache so repeated calls observe the same identity.
    template_cache: Rc<RefCell<HashMap<u32, crate::objects::value::JsValue>>>,
    /// `true` if this bytecode belongs to a generator function (`function*`).
    ///
    /// When a generator function is *called*, the interpreter creates a fresh
    /// [`crate::objects::value::GeneratorState`] and returns it as
    /// [`crate::objects::value::JsValue::Generator`] without executing the
    /// body immediately.
    is_generator: bool,
    /// `true` if this bytecode belongs to an async function or async generator.
    is_async: bool,
    /// `true` if this bytecode belongs to an ES module (as opposed to a script).
    is_module: bool,
    /// `true` if this bytecode was compiled in strict mode (`"use strict"`).
    is_strict: bool,
    /// `true` if this bytecode belongs to an arrow function (`=>`).
    ///
    /// Arrow functions are not constructable — invoking them with `new`
    /// must throw a `TypeError`.
    is_arrow: bool,
    /// `true` if this bytecode belongs to a top-level script (not a function).
    ///
    /// Top-level scripts use global variables (`LdaGlobal` / `StaGlobal`)
    /// for all `var` declarations, making speculative optimisations
    /// unreliable.  Maglev and Turbofan skip these bytecodes and let
    /// the baseline JIT handle them instead.
    is_top_level: bool,
    // ─── Tiering state (shared across clones via Rc / Arc) ───────────────────
    /// Number of times this function has been invoked.
    ///
    /// Wrapped in `Rc<Cell<_>>` so that every clone of this `BytecodeArray`
    /// (including copies moved into interpreter frames or nested closures)
    /// contributes to the same counter.  When the count reaches
    /// [`TIERING_THRESHOLD`] the interpreter triggers baseline JIT compilation.
    invocation_count: Rc<Cell<u32>>,
    /// Cached baseline-JIT machine code and register-file slot count.
    ///
    /// Stores `(code_bytes, register_file_slots)` produced by
    /// [`BaselineCompiler`][crate::compiler::baseline::compiler::BaselineCompiler].
    /// `None` until tiering has been triggered and compilation succeeded.
    jit_code: JitCodeCache,
    /// Fast, shared flag indicating that some baseline JIT code is cached.
    has_jit_code: Rc<Cell<bool>>,
    /// Persistent executable JIT code cache.
    ///
    /// Created lazily on first baseline JIT execution.  Holds a persistent
    /// `mmap`'d code region so subsequent calls avoid syscall overhead.
    jit_executable: JitExecutableCache,
    /// Cached Maglev-JIT machine code and register-file slot count.
    ///
    /// Uses [`Arc`] + [`Mutex`] so the background Maglev compilation thread
    /// can write results while the interpreter runs on the main thread.
    /// `None` until Maglev compilation finishes successfully.
    maglev_jit_code: MaglevJitCodeCache,
    /// Persistent executable Maglev code cache.
    ///
    /// Lazily initialised on first Maglev execution from the raw code bytes
    /// in [`maglev_jit_code`].  Avoids `mmap`/`munmap` per call.
    maglev_executable: MaglevExecutableCache,
    /// Fast, shared flag indicating that Maglev JIT code has been cached.
    ///
    /// Set atomically by the background compilation thread after storing code
    /// into `maglev_jit_code`, allowing the hot dispatch path to skip the
    /// mutex lock when no Maglev code is available yet.
    has_maglev_jit_code_flag: Arc<AtomicBool>,
    /// Set to `true` (via compare-exchange) when a Maglev compilation has been
    /// scheduled so that only one background thread is spawned per function.
    maglev_compile_started: Arc<AtomicBool>,
    /// Cached Turbofan (Cranelift optimising) JIT compiled code.
    ///
    /// Uses [`Arc`] + [`Mutex`] so the background Turbofan compilation thread
    /// can write results while the interpreter runs on the main thread.
    /// `None` until Turbofan compilation finishes successfully.
    turbofan_jit_code: TurbofanJitCodeCache,
    /// Fast, shared flag indicating that Turbofan JIT code has been cached.
    ///
    /// Set atomically by the background compilation thread after storing code
    /// into `turbofan_jit_code`, allowing the hot dispatch path to skip the
    /// mutex lock when no Turbofan code is available yet.
    has_turbofan_jit_code_flag: Arc<AtomicBool>,
    /// Set to `true` (via compare-exchange) when a Turbofan compilation has
    /// been scheduled so that only one background thread is spawned per
    /// function.
    turbofan_compile_started: Arc<AtomicBool>,
    /// Set to `true` when baseline JIT code has deopted at least once.
    ///
    /// When `true`, the interpreter skips all baseline JIT execution attempts
    /// for this function.  This prevents the overhead of repeatedly entering
    /// and immediately exiting JIT code that contains unsupported opcodes
    /// (e.g. `LdaCurrentContextSlot` in closures).
    jit_baseline_deopted: Rc<Cell<bool>>,
    /// When `true`, the interpreter skips all Maglev JIT execution attempts
    /// for this function.
    jit_maglev_deopted: Cell<bool>,
    /// Captured closure context set by `CreateClosure`.
    ///
    /// When a function is created as a closure, this holds the enclosing
    /// scope's context so that inner code can walk the context chain to
    /// reach captured variables.  `None` for top-level scripts and functions
    /// that do not close over any variables.
    closure_context: Option<Rc<RefCell<JsContext>>>,
    /// `true` if this function's bytecode writes to any captured closure
    /// variable (via `StaContextSlot` or `StaCurrentContextSlot`).
    ///
    /// When `false` the [`CreateClosure`](super::bytecodes::Opcode::CreateClosure)
    /// handler can create a lightweight clone that shares all immutable
    /// bytecode data with the template, because the closure only *reads*
    /// from captured variables.
    writes_closure_vars: bool,
    /// `true` if `fn_props_set` has been called on this `BytecodeArray`.
    ///
    /// When `false`, `restore_closure_context` can skip the expensive
    /// `FUNCTION_PROPS` thread-local HashMap lookup entirely.  Set to
    /// `true` by `fn_props_set` on first write.
    has_fn_props: Cell<bool>,
    /// Register index for the named function expression self-reference.
    ///
    /// When a named function expression (`var f = function g() { … }`) is
    /// called, the interpreter writes the callee function value into this
    /// register so that the body can reference the function by its own
    /// name.  `None` for anonymous functions, arrow functions, and
    /// function declarations (which are hoisted into the enclosing scope
    /// instead).
    self_name_register: Option<i32>,
    // ─── Constructor fast-path caches (shared across clones via Rc) ──────
    /// Cached `.prototype` value resolved during the first `[[Construct]]`
    /// call.  Avoids repeated `proto_lookup(&ctor, "prototype")` on
    /// subsequent `new` invocations of the same constructor.
    construct_proto_cache: Rc<RefCell<Option<crate::objects::value::JsValue>>>,
    /// Cached "boilerplate" shape (property keys + attributes) of the
    /// `this` object produced by the first successful `[[Construct]]` call.
    /// Subsequent constructions clone this shape and fill in fresh values,
    /// avoiding per-property `insert` overhead.
    construct_boilerplate: Rc<RefCell<Option<ConstructBoilerplate>>>,
    // ─── Object-literal template cache (shared across clones via Rc) ────
    /// Cached object-literal property templates keyed by feedback-slot
    /// index.
    ///
    /// On the first execution of a `CreateObjectLiteral` bytecode the
    /// resulting [`PropertyMap`] is recorded as
    /// [`ObjectLiteralCacheEntry::Pending`].  The second execution
    /// promotes the entry to [`ObjectLiteralCacheEntry::Cached`] by
    /// capturing the first object's finalised key layout.  Third and
    /// subsequent executions clone the cached template via
    /// [`PropertyMap::clone_shape`] for O(1) object creation.
    object_literal_templates: Rc<RefCell<HashMap<u32, ObjectLiteralCacheEntry>>>,
    /// Shared megamorphic IC state that persists across invocations of the same
    /// function.  New [`InterpreterFrame`]s seed their per-frame IC from this
    /// shared cache and write back on return, giving subsequent calls a warm IC
    /// without the cost of re-populating from scratch.
    ///
    /// Protected by `RefCell` and lazily allocated on first IC miss.
    shared_mega_load_ic: SharedMegamorphicIc,
    /// Shared megamorphic IC for property stores, mirroring `shared_mega_load_ic`.
    shared_mega_store_ic: SharedMegamorphicIc,
    /// Shared prototype-chain IC that persists across invocations.
    /// Maps feedback slot to cached inherited-property lookup results.
    shared_proto_load_ic: SharedProtoLoadIc,
    /// Shared global variable IC (constant_pool_idx → (slot_index, generation)).
    shared_global_ic: SharedGlobalIc,
}

/// Cached property-key layout captured after the first successful
/// `[[Construct]]` execution of a constructor function.
///
/// On subsequent `new` calls the interpreter pre-allocates a
/// [`crate::objects::property_map::PropertyMap`] with this shape so that
/// the constructor body only needs to overwrite slot values instead of
/// performing full property insertions.
#[derive(Debug, Clone)]
pub struct ConstructBoilerplate {
    /// Property key names in insertion order.
    pub keys: Vec<Rc<str>>,
    /// Per-key attribute flags.
    pub attrs: Vec<crate::objects::map::PropertyAttributes>,
}

/// Cache entry for an object-literal template, keyed by feedback slot
/// index inside a [`BytecodeArray`].
///
/// The cache transitions through two states:
///
/// 1. **[`Pending`](Self::Pending)** — recorded on the *first* execution
///    of a `CreateObjectLiteral` bytecode.  Holds an `Rc` reference to
///    the live [`PropertyMap`] being populated by subsequent
///    `DefineNamedOwnProperty` instructions.
/// 2. **[`Cached`](Self::Cached)** — promoted on the *second* execution.
///    The first object's final key layout is captured as a compact
///    [`ObjectLiteralTemplate`]. All further executions instantiate fresh
///    [`PropertyMap`]s from that cached shape.
#[derive(Debug)]
pub(crate) enum ObjectLiteralCacheEntry {
    /// First execution recorded; waiting for second to promote.
    Pending(Rc<RefCell<PropertyMap>>),
    /// Fully captured template ready for cloning.
    Cached(Box<ObjectLiteralTemplate>),
}

impl PartialEq for BytecodeArray {
    /// Two [`BytecodeArray`]s are equal when their static bytecode and metadata
    /// are identical.  The tiering state (`invocation_count`, `jit_code`,
    /// `has_jit_code`, `maglev_jit_code`, `has_maglev_jit_code_flag`,
    /// `turbofan_jit_code`, `has_turbofan_jit_code_flag`) and runtime
    /// caches (`template_cache`) are intentionally excluded from the
    /// comparison.
    fn eq(&self, other: &Self) -> bool {
        self.bytecodes == other.bytecodes
            && self.constant_pool == other.constant_pool
            && self.frame_size == other.frame_size
            && self.parameter_count == other.parameter_count
            && self.function_length == other.function_length
            && self.function_name == other.function_name
            && self.source_text == other.source_text
            && self.binding_registers == other.binding_registers
            && self.source_positions == other.source_positions
            && self.feedback_metadata == other.feedback_metadata
            && self.handler_table == other.handler_table
            && self.is_generator == other.is_generator
            && self.is_async == other.is_async
            && self.is_module == other.is_module
            && self.is_strict == other.is_strict
            && self.is_arrow == other.is_arrow
            && self.is_top_level == other.is_top_level
            && self.self_name_register == other.self_name_register
            && self.writes_closure_vars == other.writes_closure_vars
    }
}

impl BytecodeArray {
    /// Construct a new [`BytecodeArray`].
    ///
    /// - `bytecodes` — the raw encoded bytecode produced by
    ///   [`bytecodes::encode`].
    /// - `constant_pool` — all literals referenced from the bytecode.
    /// - `frame_size` — number of virtual registers needed at runtime.
    /// - `parameter_count` — number of formal parameters.
    /// - `source_positions` — optional source-position table (may be empty).
    /// - `feedback_metadata` — inline-cache slot descriptor produced by the
    ///   compiler (use [`FeedbackMetadata::empty`] when there are no IC slots).
    /// - `handler_table` — exception handler entries for `try`/`catch`/`finally`
    ///   (use an empty `Vec` when there are no try blocks).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bytecodes: Vec<u8>,
        constant_pool: Vec<ConstantPoolEntry>,
        frame_size: u32,
        parameter_count: u32,
        source_positions: Vec<SourcePosition>,
        feedback_metadata: FeedbackMetadata,
        handler_table: Vec<HandlerTableEntry>,
    ) -> Self {
        let feedback_vector = FeedbackVector::new(&feedback_metadata);
        Self {
            bytecodes: bytecodes.into(),
            constant_pool: constant_pool.into(),
            frame_size,
            parameter_count,
            function_length: parameter_count,
            function_name: Rc::from(""),
            source_text: None,
            binding_registers: Rc::new(HashMap::new()),
            source_positions: source_positions.into(),
            feedback_metadata: Rc::new(feedback_metadata),
            feedback_vector: Rc::new(RefCell::new(feedback_vector)),
            handler_table: Rc::new(handler_table),
            handler_table_remapped: Rc::new(OnceCell::new()),
            cached_decode: Rc::new(OnceCell::new()),
            template_cache: Rc::new(RefCell::new(HashMap::new())),
            is_generator: false,
            is_async: false,
            is_module: false,
            is_strict: false,
            is_arrow: false,
            is_top_level: false,
            invocation_count: Rc::new(Cell::new(0)),
            jit_code: Rc::new(RefCell::new(None)),
            has_jit_code: Rc::new(Cell::new(false)),
            jit_executable: Rc::new(RefCell::new(None)),
            maglev_jit_code: Arc::new(Mutex::new(None)),
            maglev_executable: Rc::new(RefCell::new(None)),
            has_maglev_jit_code_flag: Arc::new(AtomicBool::new(false)),
            maglev_compile_started: Arc::new(AtomicBool::new(false)),
            turbofan_jit_code: Arc::new(Mutex::new(None)),
            has_turbofan_jit_code_flag: Arc::new(AtomicBool::new(false)),
            turbofan_compile_started: Arc::new(AtomicBool::new(false)),
            jit_baseline_deopted: Rc::new(Cell::new(false)),
            jit_maglev_deopted: Cell::new(false),
            closure_context: None,
            writes_closure_vars: false,
            has_fn_props: Cell::new(false),
            self_name_register: None,
            construct_proto_cache: Rc::new(RefCell::new(None)),
            construct_boilerplate: Rc::new(RefCell::new(None)),
            object_literal_templates: Rc::new(RefCell::new(HashMap::new())),
            shared_mega_load_ic: Rc::new(RefCell::new(None)),
            shared_mega_store_ic: Rc::new(RefCell::new(None)),
            shared_proto_load_ic: Rc::new(RefCell::new(None)),
            shared_global_ic: Rc::new(RefCell::new(None)),
        }
    }

    /// Return a cached template object for the given bytecode offset, if any.
    pub fn cached_template_object(
        &self,
        bytecode_offset: u32,
    ) -> Option<crate::objects::value::JsValue> {
        self.template_cache.borrow().get(&bytecode_offset).cloned()
    }

    /// Cache a template object for the given bytecode offset.
    pub fn cache_template_object(
        &self,
        bytecode_offset: u32,
        value: crate::objects::value::JsValue,
    ) {
        self.template_cache
            .borrow_mut()
            .insert(bytecode_offset, value);
    }

    /// Return the number of cached template objects.
    #[cfg(test)]
    pub(crate) fn template_cache_len(&self) -> usize {
        self.template_cache.borrow().len()
    }

    /// Mark this [`BytecodeArray`] as belonging to a generator function.
    ///
    /// Returns `self` so this can be chained onto [`BytecodeArray::new`]:
    /// ```
    /// # use stator_core::bytecode::bytecode_array::BytecodeArray;
    /// # use stator_core::bytecode::feedback::FeedbackMetadata;
    /// let ba = BytecodeArray::new(vec![], vec![], 0, 0, vec![], FeedbackMetadata::empty(), vec![])
    ///     .with_generator_flag(true);
    /// assert!(ba.is_generator());
    /// ```
    pub fn with_generator_flag(mut self, flag: bool) -> Self {
        self.is_generator = flag;
        self
    }

    /// Returns `true` if this bytecode belongs to a `function*` generator.
    pub fn is_generator(&self) -> bool {
        self.is_generator
    }

    /// Mark this [`BytecodeArray`] as belonging to an async function.
    ///
    /// When combined with [`BytecodeArray::with_generator_flag`] this marks
    /// the function as an async generator (`async function*`).
    pub fn with_async_flag(mut self, flag: bool) -> Self {
        self.is_async = flag;
        self
    }

    /// Returns `true` if this bytecode belongs to an `async function` or
    /// `async function*`.
    pub fn is_async(&self) -> bool {
        self.is_async
    }

    /// Mark this [`BytecodeArray`] as belonging to an ES module.
    pub fn with_module_flag(mut self, flag: bool) -> Self {
        self.is_module = flag;
        self
    }

    /// Returns `true` if this bytecode belongs to an ES module.
    pub fn is_module(&self) -> bool {
        self.is_module
    }

    /// Mark this [`BytecodeArray`] as compiled in strict mode.
    pub fn with_strict_flag(mut self, flag: bool) -> Self {
        self.is_strict = flag;
        self
    }

    /// Returns `true` if this bytecode was compiled in strict mode.
    pub fn is_strict(&self) -> bool {
        self.is_strict
    }

    /// Mark this [`BytecodeArray`] as belonging to an arrow function.
    ///
    /// Arrow functions are not constructable — invoking them with `new`
    /// must throw a `TypeError` per ES §15.3.4.
    pub fn with_arrow_flag(mut self, flag: bool) -> Self {
        self.is_arrow = flag;
        self
    }

    /// Returns `true` if this bytecode belongs to an arrow function (`=>`).
    pub fn is_arrow(&self) -> bool {
        self.is_arrow
    }

    /// Mark this bytecode as a top-level script.
    pub fn set_top_level(&mut self, flag: bool) {
        self.is_top_level = flag;
    }

    /// Returns `true` if this bytecode belongs to a top-level script.
    pub fn is_top_level(&self) -> bool {
        self.is_top_level
    }

    /// Builder-style: mark this bytecode as a top-level script.
    pub fn with_top_level_flag(mut self, flag: bool) -> Self {
        self.is_top_level = flag;
        self
    }

    /// Returns the captured closure context, if any.
    pub fn closure_context(&self) -> Option<&Rc<RefCell<JsContext>>> {
        self.closure_context.as_ref()
    }

    /// Attach a captured closure context to this [`BytecodeArray`].
    pub fn set_closure_context(&mut self, ctx: Rc<RefCell<JsContext>>) {
        self.closure_context = Some(ctx);
    }

    /// Returns `true` if this function's bytecode writes to any captured
    /// closure variable.
    pub fn writes_closure_vars(&self) -> bool {
        self.writes_closure_vars
    }

    /// Returns `true` if `fn_props_set` has been called on this bytecode
    /// array (or any clone sharing the same `has_fn_props` flag).
    #[inline(always)]
    pub fn has_fn_props(&self) -> bool {
        self.has_fn_props.get()
    }

    /// Mark that function-properties side-table entries exist for this
    /// bytecode array.
    #[inline(always)]
    pub fn mark_has_fn_props(&self) {
        self.has_fn_props.set(true);
    }

    /// Mark whether this function writes to captured closure variables.
    pub fn with_writes_closure_vars(mut self, flag: bool) -> Self {
        self.writes_closure_vars = flag;
        self
    }

    /// Create a lightweight clone for use as a closure, attaching the given
    /// closure context.
    ///
    /// All immutable bytecode data (bytecodes, constant pool, source
    /// positions, etc.) is shared with the original via [`Rc`], making
    /// this operation O(1) regardless of function size.
    pub fn clone_for_closure(&self, ctx: Option<Rc<RefCell<JsContext>>>) -> Self {
        Self {
            closure_context: ctx,
            ..self.clone()
        }
    }

    /// Register index for a named function expression's self-reference.
    pub fn self_name_register(&self) -> Option<i32> {
        self.self_name_register
    }

    /// Set the self-name register for named function expressions.
    pub fn with_self_name_register(mut self, reg: i32) -> Self {
        self.self_name_register = Some(reg);
        self
    }

    // ─── Constructor fast-path cache accessors ───────────────────────────

    /// Returns the cached constructor `.prototype` value, if populated.
    pub fn cached_construct_proto(&self) -> Option<crate::objects::value::JsValue> {
        self.construct_proto_cache.borrow().clone()
    }

    /// Stores a constructor `.prototype` value for reuse on subsequent
    /// `[[Construct]]` calls.
    pub fn set_construct_proto_cache(&self, proto: crate::objects::value::JsValue) {
        *self.construct_proto_cache.borrow_mut() = Some(proto);
    }

    /// Returns a clone of the cached construct boilerplate, if populated.
    pub fn cached_construct_boilerplate(&self) -> Option<ConstructBoilerplate> {
        self.construct_boilerplate.borrow().clone()
    }

    /// Stores a construct boilerplate captured from the first successful
    /// `[[Construct]]` execution.
    pub fn set_construct_boilerplate(&self, bp: ConstructBoilerplate) {
        *self.construct_boilerplate.borrow_mut() = Some(bp);
    }

    // ─── Object-literal template cache API ───────────────────────────────

    /// If a cached object-literal template exists for `slot`, returns a
    /// fresh [`PropertyMap`] instantiated from it.
    pub fn clone_object_literal_template(&self, slot: u32) -> Option<PropertyMap> {
        let borrow = self.object_literal_templates.borrow();
        match borrow.get(&slot) {
            Some(ObjectLiteralCacheEntry::Cached(template)) => Some(template.instantiate()),
            _ => None,
        }
    }

    /// If a [`Pending`](ObjectLiteralCacheEntry::Pending) first-instance
    /// exists for `slot`, promotes it to
    /// [`Cached`](ObjectLiteralCacheEntry::Cached) by capturing its
    /// finalised key layout and returns a template clone.
    ///
    /// Returns `None` if no pending entry exists or the first instance has
    /// no properties (nothing worth caching).
    pub fn promote_object_literal_template(&self, slot: u32) -> Option<PropertyMap> {
        let pending_rc = {
            let borrow = self.object_literal_templates.borrow();
            match borrow.get(&slot) {
                Some(ObjectLiteralCacheEntry::Pending(rc)) => Some(Rc::clone(rc)),
                _ => None,
            }
        };
        let first_rc = pending_rc?;
        let first_borrow = first_rc.borrow();
        let template = ObjectLiteralTemplate::capture(&first_borrow)?;
        let cloned = template.instantiate();
        drop(first_borrow);
        self.object_literal_templates
            .borrow_mut()
            .insert(slot, ObjectLiteralCacheEntry::Cached(Box::new(template)));
        Some(cloned)
    }

    /// Records a [`Pending`](ObjectLiteralCacheEntry::Pending)
    /// first-instance for `slot`.
    pub fn set_object_literal_pending(&self, slot: u32, map: Rc<RefCell<PropertyMap>>) {
        self.object_literal_templates
            .borrow_mut()
            .insert(slot, ObjectLiteralCacheEntry::Pending(map));
    }

    // ─── Shared megamorphic IC accessors ─────────────────────────────────

    /// Returns a clone of the shared megamorphic load IC, if one has been
    /// populated by a previous invocation.
    pub fn shared_mega_load_ic(&self) -> Option<Box<crate::interpreter::MegamorphicIc>> {
        self.shared_mega_load_ic.borrow().clone()
    }

    /// Writes back a megamorphic load IC to the shared cache so that
    /// subsequent invocations start warm.
    pub fn set_shared_mega_load_ic(&self, ic: Box<crate::interpreter::MegamorphicIc>) {
        *self.shared_mega_load_ic.borrow_mut() = Some(ic);
    }

    /// Returns a clone of the shared megamorphic store IC, if one has been
    /// populated by a previous invocation.
    pub fn shared_mega_store_ic(&self) -> Option<Box<crate::interpreter::MegamorphicIc>> {
        self.shared_mega_store_ic.borrow().clone()
    }

    /// Writes back a megamorphic store IC to the shared cache so that
    /// subsequent invocations start warm.
    pub fn set_shared_mega_store_ic(&self, ic: Box<crate::interpreter::MegamorphicIc>) {
        *self.shared_mega_store_ic.borrow_mut() = Some(ic);
    }

    /// Returns a clone of the shared prototype-chain load IC, if populated.
    pub fn shared_proto_load_ic(
        &self,
    ) -> Option<Box<HashMap<u32, crate::interpreter::ProtoLoadIc>>> {
        self.shared_proto_load_ic.borrow().clone()
    }

    /// Writes back a prototype-chain load IC to the shared cache.
    pub fn set_shared_proto_load_ic(&self, ic: Box<HashMap<u32, crate::interpreter::ProtoLoadIc>>) {
        *self.shared_proto_load_ic.borrow_mut() = Some(ic);
    }

    /// Returns a clone of the shared global variable IC, if populated.
    pub fn shared_global_ic(&self) -> Option<Box<HashMap<u32, (usize, u64)>>> {
        self.shared_global_ic.borrow().clone()
    }

    /// Writes back a global variable IC to the shared cache.
    pub fn set_shared_global_ic(&self, ic: Box<HashMap<u32, (usize, u64)>>) {
        *self.shared_global_ic.borrow_mut() = Some(ic);
    }

    /// The raw encoded bytecode bytes.
    pub fn bytecodes(&self) -> &[u8] {
        &self.bytecodes
    }

    /// The constant pool for this function.
    pub fn constant_pool(&self) -> &[ConstantPoolEntry] {
        &self.constant_pool
    }

    /// Number of virtual registers required by this function's frame.
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }

    /// Number of formal parameters declared by this function.
    pub fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Number of decoded bytecode instructions after peephole fusion.
    ///
    /// Returns [`usize::MAX`] for malformed bytecode so hot-path callers can
    /// conservatively skip optimizations that depend on a valid instruction
    /// count.
    pub fn bytecode_count(&self) -> usize {
        self.ensure_decoded_instructions()
            .map_or(usize::MAX, |decoded| decoded.0.len())
    }

    /// Returns `true` when this function has at least one exception handler.
    pub fn has_exception_handler(&self) -> bool {
        !self.handler_table.is_empty()
    }

    /// `Function.prototype.length` for this function.
    pub fn function_length(&self) -> u32 {
        self.function_length
    }

    /// Set `Function.prototype.length` metadata.
    pub fn with_function_length(mut self, length: u32) -> Self {
        self.function_length = length;
        self
    }

    /// Declared or inferred function name.
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Set the declared or inferred function name.
    pub fn with_function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = Rc::from(name.into());
        self
    }

    /// Source text used by `Function.prototype.toString()`, if any.
    pub fn source_text(&self) -> Option<&str> {
        self.source_text.as_deref()
    }

    /// Set the source text used by `Function.prototype.toString()`.
    pub fn with_source_text(mut self, source_text: impl Into<String>) -> Self {
        self.source_text = Some(Rc::from(source_text.into()));
        self
    }

    /// Visible binding-to-register mapping for direct `eval()`.
    pub fn binding_registers(&self) -> &HashMap<String, i32> {
        &self.binding_registers
    }

    /// Set the binding-to-register mapping used by direct `eval()`.
    pub fn with_binding_registers(mut self, binding_registers: HashMap<String, i32>) -> Self {
        self.binding_registers = Rc::new(binding_registers);
        self
    }

    /// The source-position table (may be empty if debug info was stripped).
    pub fn source_positions(&self) -> &[SourcePosition] {
        &self.source_positions
    }

    /// The compile-time feedback metadata for all inline-cache slots.
    pub fn feedback_metadata(&self) -> &FeedbackMetadata {
        &self.feedback_metadata
    }

    /// Return a snapshot of the current runtime feedback vector.
    pub fn feedback_vector_snapshot(&self) -> FeedbackVector {
        self.feedback_vector.borrow().clone()
    }

    /// Return the current inline-cache state for `slot`.
    pub fn feedback_state(&self, slot: u32) -> Option<InlineCacheState> {
        self.feedback_vector.borrow().get_state(slot)
    }

    /// Advance the feedback state for `slot` if `new_state` is hotter.
    pub fn feedback_transition(&self, slot: u32, new_state: InlineCacheState) -> bool {
        self.feedback_vector
            .borrow_mut()
            .transition(slot, new_state)
    }

    /// Overwrite the feedback state for `slot`.
    pub fn set_feedback_state(&self, slot: u32, state: InlineCacheState) -> bool {
        self.feedback_vector.borrow_mut().set_state(slot, state)
    }

    /// The per-function exception handler table.
    ///
    /// Each entry maps a `[try_start, try_end)` instruction-index range to a
    /// handler entry point.  Entries are ordered so that the innermost handler
    /// for any given instruction always appears before outer handlers.
    ///
    /// After the peephole pass compacts instructions, this returns the
    /// remapped table with post-compaction indices.
    pub fn handler_table(&self) -> &[HandlerTableEntry] {
        // Trigger decode + remap if not done yet.
        let _ = self.ensure_decoded_instructions();
        // If we have a remapped table, we can't return a direct reference to
        // it because it's behind RefCell.  Fall back to the original table
        // (which is only used before first decode in practice).
        self.handler_table.as_slice()
    }

    /// Return a shared reference-counted handle to the exception handler
    /// table, remapped for post-peephole instruction indices if available.
    pub(crate) fn shared_handler_table(&self) -> Rc<Vec<HandlerTableEntry>> {
        if let Some(remapped) = self.handler_table_remapped.get() {
            Rc::clone(remapped)
        } else {
            Rc::clone(&self.handler_table)
        }
    }

    /// Decode the bytecode stream and return the list of [`Instruction`]s.
    ///
    /// Returns an error if the byte stream is malformed.
    pub fn instructions(&self) -> StatorResult<Vec<Instruction>> {
        bytecodes::decode(&self.bytecodes)
    }

    fn ensure_decoded_instructions(&self) -> StatorResult<&Rc<DecodedBytecode>> {
        if self.cached_decode.get().is_none() {
            let (mut instructions, mut byte_offsets) =
                bytecodes::decode_with_byte_offsets(&self.bytecodes)?;

            let remap = peephole::fuse_instructions_with_remap(
                &mut instructions,
                &mut byte_offsets,
                Some(self.constant_pool()),
            );

            // Helper: map a pre-peephole instruction index to the first
            // surviving post-peephole instruction at or after that position.
            let remap_index = |idx: u32, fallback: usize| -> u32 {
                remap
                    .iter()
                    .skip(idx as usize)
                    .find_map(|mapped| *mapped)
                    .unwrap_or(fallback) as u32
            };

            // Remap the handler table entries from pre-peephole instruction
            // indices to post-peephole instruction indices.
            let remapped_handlers: Vec<HandlerTableEntry> = self
                .handler_table
                .iter()
                .map(|entry| HandlerTableEntry {
                    try_start: remap_index(entry.try_start, 0),
                    try_end: remap_index(entry.try_end, instructions.len()),
                    handler: remap_index(entry.handler, 0),
                    is_finally: entry.is_finally,
                })
                .collect();
            // Replace the handler table with the remapped version.
            // SAFETY: Interior mutability is acceptable here — the handler
            // table is initialised once during first decode, just like
            // cached_decode.
            let _ = self.handler_table_remapped.set(Rc::new(remapped_handlers));

            let mut jump_targets = vec![None; instructions.len()];
            for (instruction_index, instruction) in instructions.iter().enumerate() {
                for operand in instruction.operands() {
                    let Operand::JumpOffset(delta) = operand else {
                        continue;
                    };
                    let pc_after_jump = instruction_index + 1;
                    let end_byte = *byte_offsets.get(pc_after_jump).ok_or_else(|| {
                        StatorError::Internal(format!(
                            "missing post-jump byte offset for instruction {instruction_index}"
                        ))
                    })?;
                    let target_byte = (end_byte as i64 + i64::from(*delta)) as usize;
                    let target_index = byte_offsets
                        .binary_search(&target_byte)
                        .map_err(|_| {
                            StatorError::Internal(format!(
                                "jump target byte offset {target_byte} is not at an instruction boundary"
                            ))
                        })?;
                    jump_targets[instruction_index] = Some(target_index);
                }
            }
            let decoded = Rc::new((instructions, byte_offsets, jump_targets));
            let _ = self.cached_decode.set(decoded);
        }
        Ok(self
            .cached_decode
            .get()
            .expect("decoded bytecode cache must be initialized"))
    }

    /// Decode the bytecode stream once and return cached instructions, byte
    /// offsets, and pre-computed jump targets on subsequent calls.
    pub fn decoded_instructions(&mut self) -> StatorResult<DecodedBytecodeRef<'_>> {
        let decoded = self.ensure_decoded_instructions()?;
        Ok((
            decoded.0.as_slice(),
            decoded.1.as_slice(),
            decoded.2.as_slice(),
        ))
    }

    /// Return a shared handle to the cached decoded instruction stream.
    pub(crate) fn shared_decoded_instructions(&self) -> StatorResult<Rc<DecodedBytecode>> {
        Ok(Rc::clone(self.ensure_decoded_instructions()?))
    }

    /// Look up a constant-pool entry by zero-based `index`.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn get_constant(&self, index: u32) -> Option<&ConstantPoolEntry> {
        self.constant_pool.get(index as usize)
    }

    /// Return the [`SourcePosition`] that covers `bytecode_offset`, or `None`
    /// if the source-position table is empty or no entry precedes the offset.
    ///
    /// The table must be sorted by `bytecode_offset` (ascending).  The lookup
    /// uses binary search and returns the last entry whose `bytecode_offset`
    /// is ≤ the given offset.
    pub fn source_position_for(&self, bytecode_offset: u32) -> Option<&SourcePosition> {
        let idx = self
            .source_positions
            .partition_point(|sp| sp.bytecode_offset <= bytecode_offset);
        idx.checked_sub(1).map(|i| &self.source_positions[i])
    }

    // ─── Tiering helpers ──────────────────────────────────────────────────────

    /// Atomically increment the invocation counter and return the **new** value.
    ///
    /// All clones of this [`BytecodeArray`] share the same counter via the
    /// inner [`Rc`], so every copy — whether still held in a
    /// [`JsValue::Function`][crate::objects::value::JsValue] or already moved
    /// into an [`crate::interpreter::InterpreterFrame`] — increments the same
    /// counter.
    #[inline(always)]
    pub fn increment_invocation_count(&self) -> u32 {
        let old = self.invocation_count.get();
        if old >= TURBOFAN_TIERING_THRESHOLD {
            return old;
        }
        let new = old + 1;
        self.invocation_count.set(new);
        new
    }

    /// Returns the current invocation count without modifying it.
    pub fn invocation_count(&self) -> u32 {
        self.invocation_count.get()
    }

    /// Store baseline-JIT cached executable code produced by the compiler.
    ///
    /// `cached` is a [`CachedExecutableCode`] that owns a persistent `mmap`'d
    /// page of executable memory.
    ///
    /// All clones of this [`BytecodeArray`] share the same JIT cache.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub fn store_jit_code(
        &self,
        cached: crate::compiler::baseline::compiler::CachedExecutableCode,
    ) {
        *self.jit_code.borrow_mut() = Some(cached);
        self.has_jit_code.set(true);
    }

    /// Store baseline-JIT machine code produced by the compiler (non-JIT
    /// platform fallback).
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    pub fn store_jit_code(&self, code: Vec<u8>, register_file_slots: usize) {
        *self.jit_code.borrow_mut() = Some((code, register_file_slots));
        self.has_jit_code.set(true);
    }

    /// Fast check for whether any JIT tier has compiled code for this function.
    ///
    /// Checks the fast boolean flags for baseline, Maglev, and Turbofan tiers
    /// using short-circuit evaluation to avoid unnecessary atomic loads when
    /// an earlier tier is already compiled.
    #[inline(always)]
    pub fn has_any_jit_code(&self) -> bool {
        self.has_jit_code.get()
            || self.has_maglev_jit_code_flag.load(Ordering::Relaxed)
            || self.has_turbofan_jit_code_flag.load(Ordering::Relaxed)
    }

    /// Borrows the cached JIT executable code, or returns `None` if baseline
    /// compilation has not been triggered yet.
    ///
    /// The caller can call [`CachedExecutableCode::execute`] on the borrowed
    /// reference without cloning or allocating executable memory.
    #[cfg(all(target_arch = "x86_64", unix))]
    pub fn try_get_jit_code(
        &self,
    ) -> std::cell::Ref<'_, Option<crate::compiler::baseline::compiler::CachedExecutableCode>> {
        self.jit_code.borrow()
    }

    /// Returns a clone of the cached JIT machine code and register-file slot
    /// count, or `None` if baseline compilation has not been triggered yet
    /// (non-JIT platform fallback).
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    pub fn try_get_jit_code(&self) -> Option<(Vec<u8>, usize)> {
        self.jit_code.borrow().clone()
    }

    /// Returns `true` when Maglev JIT code has been cached for this function.
    pub fn has_maglev_jit_code(&self) -> bool {
        self.maglev_jit_code
            .lock()
            .ok()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Returns an [`Arc`] clone of the Maglev JIT code cache.
    ///
    /// The background compilation thread receives this `Arc` and writes the
    /// compiled code into it when compilation succeeds.
    pub fn maglev_jit_cache_arc(&self) -> MaglevJitCodeCache {
        Arc::clone(&self.maglev_jit_code)
    }

    /// Returns an [`Arc`] clone of the Maglev JIT-code-ready flag.
    ///
    /// The background compilation thread sets this to `true` after storing
    /// compiled code, so the hot dispatch path can skip the mutex lock.
    pub fn maglev_jit_code_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.has_maglev_jit_code_flag)
    }

    /// Attempt to atomically mark this function as having a Maglev compilation
    /// in flight.
    ///
    /// Returns `true` if the caller successfully claimed the compilation slot
    /// (the previous state was `false`); returns `false` if a compilation was
    /// already started or has been scheduled by another caller.
    pub fn try_start_maglev_compile(&self) -> bool {
        self.maglev_compile_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Returns `true` if Turbofan compilation has finished and compiled code
    /// is available.
    pub fn has_turbofan_jit_code(&self) -> bool {
        self.turbofan_jit_code
            .lock()
            .ok()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Returns an [`Arc`] clone of the Turbofan JIT code cache.
    ///
    /// The background compilation thread receives this `Arc` and writes the
    /// compiled [`TurbofanCompiledCode`] into it when compilation succeeds.
    pub fn turbofan_jit_cache_arc(&self) -> TurbofanJitCodeCache {
        Arc::clone(&self.turbofan_jit_code)
    }

    /// Returns an [`Arc`] clone of the Turbofan JIT-code-ready flag.
    ///
    /// The background compilation thread sets this to `true` after storing
    /// compiled code, so the hot dispatch path can skip the mutex lock.
    pub fn turbofan_jit_code_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.has_turbofan_jit_code_flag)
    }

    /// Attempt to atomically mark this function as having a Turbofan
    /// compilation in flight.
    ///
    /// Returns `true` if the caller successfully claimed the compilation slot
    /// (the previous state was `false`); returns `false` if a compilation was
    /// already started or has been scheduled by another caller.
    pub fn try_start_turbofan_compile(&self) -> bool {
        self.turbofan_compile_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Returns a shared reference to the persistent executable JIT code cache.
    ///
    /// On the first call after baseline JIT compilation, the cache is empty
    /// and the caller should populate it via [`JitExecutableCode::new`].
    /// Subsequent calls return the cached executable code directly.
    pub fn jit_executable_cache(&self) -> &JitExecutableCache {
        &self.jit_executable
    }

    /// Returns a reference to the cached Maglev executable code.
    ///
    /// The cache is lazily initialised on first Maglev execution from the raw
    /// compiled code bytes.
    pub fn maglev_executable_cache(&self) -> &MaglevExecutableCache {
        &self.maglev_executable
    }

    /// Returns `true` if baseline JIT code has deopted at least once,
    /// indicating the generated code contains unsupported opcodes and
    /// should not be re-attempted.
    pub fn jit_baseline_has_deopted(&self) -> bool {
        self.jit_baseline_deopted.get()
    }

    /// Mark this function's baseline JIT code as having deopted.
    ///
    /// Once set, the interpreter will skip all baseline JIT execution
    /// attempts for this function to avoid the overhead of repeatedly
    /// entering and immediately exiting always-deopting code.
    pub fn mark_jit_baseline_deopted(&self) {
        self.jit_baseline_deopted.set(true);
    }

    /// Returns `true` if Maglev JIT code has deopted at least once.
    pub fn jit_maglev_has_deopted(&self) -> bool {
        self.jit_maglev_deopted.get()
    }

    /// Mark this function's Maglev JIT code as having deopted.
    pub fn mark_jit_maglev_deopted(&self) {
        self.jit_maglev_deopted.set(true);
    }

    /// Returns a clone of the cached Maglev-JIT machine code and
    /// register-file slot count, or `None` if Maglev compilation has not
    /// finished yet.
    ///
    /// Only available on non-JIT platforms where [`MaglevJitCodeCache`] stores
    /// raw bytes.  On x86-64 Unix, use [`maglev_jit_cache_arc`](Self::maglev_jit_cache_arc) directly.
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    pub fn try_get_maglev_jit_code(&self) -> Option<(Vec<u8>, usize)> {
        self.maglev_jit_code.lock().ok()?.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;
    use crate::objects::property_map::PropertyMap;
    use crate::objects::value::JsValue;

    fn make_simple_array() -> BytecodeArray {
        // load smi 7 → r0, return
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instructions);
        BytecodeArray::new(
            bytes,
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    fn make_jump_array() -> BytecodeArray {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instructions);
        let (_, offsets) = bytecodes::decode_with_byte_offsets(&bytes).expect("valid bytecode");
        let target_byte = offsets[2];
        let jump_end_byte = offsets[1];
        let mut resolved = instructions;
        *resolved[0].operand_mut(0) =
            Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);
        BytecodeArray::new(
            encode(&resolved),
            vec![],
            1,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    #[test]
    fn test_create_bytecode_array() {
        let array = make_simple_array();
        assert_eq!(array.frame_size(), 1);
        assert_eq!(array.parameter_count(), 0);
        assert!(array.constant_pool().is_empty());
        assert!(array.source_positions().is_empty());
        assert!(!array.bytecodes().is_empty());
    }

    #[test]
    fn test_iterate_instructions() {
        let array = make_simple_array();
        let instrs = array.instructions().expect("valid bytecode");
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[0].opcode, Opcode::LdaSmi);
        assert_eq!(instrs[1].opcode, Opcode::Star);
        assert_eq!(instrs[2].opcode, Opcode::Return);
    }

    #[test]
    fn test_constant_pool() {
        let instructions = vec![
            Instruction::new_unchecked(Opcode::LdaConstant, vec![Operand::ConstantPoolIdx(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&instructions);
        let pool = vec![
            ConstantPoolEntry::Number(3.14),
            ConstantPoolEntry::String("hello".to_owned()),
            ConstantPoolEntry::Boolean(true),
            ConstantPoolEntry::Null,
            ConstantPoolEntry::Undefined,
        ];
        let array =
            BytecodeArray::new(bytes, pool, 0, 1, vec![], FeedbackMetadata::empty(), vec![]);

        assert_eq!(array.constant_pool().len(), 5);
        assert_eq!(
            array.get_constant(0),
            Some(&ConstantPoolEntry::Number(3.14))
        );
        assert_eq!(
            array.get_constant(1),
            Some(&ConstantPoolEntry::String("hello".to_owned()))
        );
        assert_eq!(
            array.get_constant(2),
            Some(&ConstantPoolEntry::Boolean(true))
        );
        assert_eq!(array.get_constant(3), Some(&ConstantPoolEntry::Null));
        assert_eq!(array.get_constant(4), Some(&ConstantPoolEntry::Undefined));
        assert_eq!(array.get_constant(5), None);
    }

    #[test]
    fn test_object_literal_template_cache_stores_shape_only() {
        let array = make_simple_array();
        let mut first = PropertyMap::with_capacity(4);
        first.insert("x".to_string(), JsValue::Smi(1));
        first.insert("y".to_string(), JsValue::Smi(2));

        let first = Rc::new(RefCell::new(first));
        array.set_object_literal_pending(7, Rc::clone(&first));

        let promoted = array
            .promote_object_literal_template(7)
            .expect("template should be promoted");
        assert_eq!(promoted.layout_id(), first.borrow().layout_id());
        assert_eq!(promoted.get("x"), Some(&JsValue::Undefined));
        assert_eq!(promoted.get("y"), Some(&JsValue::Undefined));

        first.borrow_mut().insert("x".to_string(), JsValue::Smi(99));
        let cached = array
            .clone_object_literal_template(7)
            .expect("cached template should instantiate");
        assert_eq!(cached.layout_id(), promoted.layout_id());
        assert_eq!(cached.get("x"), Some(&JsValue::Undefined));
        assert_eq!(cached.get("y"), Some(&JsValue::Undefined));
    }

    #[test]
    fn test_source_positions() {
        let array = BytecodeArray::new(
            vec![],
            vec![],
            0,
            0,
            vec![
                SourcePosition::new(0, 1, 1),
                SourcePosition::new(4, 2, 5),
                SourcePosition::new(8, 3, 1),
            ],
            FeedbackMetadata::empty(),
            vec![],
        );

        assert_eq!(
            array.source_position_for(0),
            Some(&SourcePosition::new(0, 1, 1))
        );
        assert_eq!(
            array.source_position_for(2),
            Some(&SourcePosition::new(0, 1, 1))
        );
        assert_eq!(
            array.source_position_for(4),
            Some(&SourcePosition::new(4, 2, 5))
        );
        assert_eq!(
            array.source_position_for(10),
            Some(&SourcePosition::new(8, 3, 1))
        );
    }

    #[test]
    fn test_source_position_empty_table() {
        let array = make_simple_array();
        assert_eq!(array.source_position_for(0), None);
    }

    #[test]
    fn test_instructions_decode_error() {
        // Truncated LdaSmi (opcode only, no operand byte) → decode error.
        let array = BytecodeArray::new(
            vec![Opcode::LdaSmi as u8],
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        assert!(array.instructions().is_err());
    }

    #[test]
    fn test_feedback_metadata_stored_in_array() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        let metadata =
            FeedbackMetadata::new(vec![FeedbackSlotKind::Call, FeedbackSlotKind::LoadProperty]);
        let array = BytecodeArray::new(vec![], vec![], 0, 0, vec![], metadata, vec![]);
        assert_eq!(array.feedback_metadata().slot_count(), 2);
        assert_eq!(
            array.feedback_metadata().kind_of(0),
            Some(FeedbackSlotKind::Call)
        );
        assert_eq!(
            array.feedback_metadata().kind_of(1),
            Some(FeedbackSlotKind::LoadProperty)
        );
    }

    #[test]
    fn test_decoded_instructions_are_cached() {
        let mut array = make_simple_array();

        // Decode fresh to compare against cached version.
        let expected_offsets = bytecodes::decode_with_byte_offsets(array.bytecodes())
            .expect("valid bytecode")
            .1;

        // First call populates the cache (uses &mut self).
        // The peephole optimizer fuses LdaSmi+Star into LdaSmiStar, yielding
        // two instructions instead of three.
        {
            let (instructions, offsets, jump_targets) =
                array.decoded_instructions().expect("valid bytecode");
            assert_eq!(instructions.len(), 2);
            assert_eq!(instructions[0].opcode, Opcode::LdaSmiStar);
            assert_eq!(instructions[1].opcode, Opcode::Return);
            // Fused offsets differ from the raw decode; just verify the count
            // matches instructions.len() + 1 (includes the end-of-bytecode
            // sentinel).
            assert_eq!(offsets.len(), instructions.len() + 1);
            // No jump instructions in simple bytecode, so all entries are None.
            assert!(jump_targets.iter().all(|t| t.is_none()));
        }

        // Second call returns the same cached Rc allocation.
        let first = array
            .shared_decoded_instructions()
            .expect("cached bytecode");
        let second = array
            .shared_decoded_instructions()
            .expect("cached bytecode 2");
        assert!(std::ptr::eq(first.0.as_ptr(), second.0.as_ptr()));
        assert!(std::ptr::eq(first.1.as_ptr(), second.1.as_ptr()));
        assert!(std::ptr::eq(&first.2, &second.2));
    }

    #[test]
    fn test_decoded_instructions_cache_is_shared_across_clones() {
        let array = make_simple_array();
        let clone = array.clone();
        let decoded_orig = array.shared_decoded_instructions().expect("valid bytecode");
        let decoded_clone = clone
            .shared_decoded_instructions()
            .expect("shared cached bytecode");

        assert!(std::ptr::eq(
            decoded_orig.0.as_ptr(),
            decoded_clone.0.as_ptr()
        ));
        assert!(std::ptr::eq(
            decoded_orig.1.as_ptr(),
            decoded_clone.1.as_ptr()
        ));
        assert!(std::ptr::eq(&decoded_orig.2, &decoded_clone.2));
    }

    #[test]
    fn test_decoded_instructions_cache_includes_jump_targets() {
        let mut array = make_jump_array();

        let (instructions, _offsets, jump_targets) =
            array.decoded_instructions().expect("valid bytecode");

        assert_eq!(instructions[0].opcode, Opcode::Jump);
        assert_eq!(jump_targets[0], Some(2));
    }

    #[test]
    fn test_decoded_instructions_apply_fusion_before_jump_resolution() {
        let unresolved = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(3), Operand::FeedbackSlot(4)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfTrue, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let bytes = encode(&unresolved);
        let (_, offsets) = bytecodes::decode_with_byte_offsets(&bytes).expect("valid bytecode");
        let mut resolved = unresolved;
        let target_byte = offsets[6];
        let jump_end_byte = offsets[5];
        *resolved[4].operand_mut(0) =
            Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);

        let mut array = BytecodeArray::new(
            encode(&resolved),
            vec![],
            4,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let (instructions, _offsets, jump_targets) =
            array.decoded_instructions().expect("valid bytecode");

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdarAddStar,
                Opcode::TestLessThanJump,
                Opcode::LdaZero,
                Opcode::Return,
            ]
        );
        assert_eq!(jump_targets[1], Some(3));
    }
}

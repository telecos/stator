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

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::bytecode::bytecodes::{self, Instruction};
use crate::bytecode::feedback::FeedbackMetadata;
use crate::compiler::turbofan::TurbofanCompiledCode;
use crate::error::StatorResult;

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
    /// A compiled nested function or closure.
    Function(Box<BytecodeArray>),
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
/// Contains the raw x86-64 machine code bytes produced by the baseline
/// compiler and the number of `i64` register-file slots required by the
/// generated code.  The outer [`Rc`] allows all clones of a [`BytecodeArray`]
/// to share the same cache without copying.
type JitCodeCache = Rc<RefCell<Option<(Vec<u8>, usize)>>>;

/// Shared Maglev JIT code cache stored in a [`BytecodeArray`].
///
/// Uses [`Arc`] + [`Mutex`] so that the Maglev background compilation thread
/// can write the compiled code into the cache while the interpreter runs on
/// the main thread.
pub type MaglevJitCodeCache = Arc<Mutex<Option<(Vec<u8>, usize)>>>;

/// Shared Turbofan JIT code cache stored in a [`BytecodeArray`].
///
/// Stores a fully-compiled [`TurbofanCompiledCode`] produced by the Turbofan
/// background thread.  Uses [`Arc`] + [`Mutex`] so the background thread can
/// write the code while the interpreter runs on the main thread.
pub type TurbofanJitCodeCache = Arc<Mutex<Option<TurbofanCompiledCode>>>;

/// Invocation-count threshold that triggers baseline JIT compilation.
///
/// When a function's `invocation_count` reaches this value the interpreter
/// requests a baseline-compiled version; all subsequent calls that can be
/// represented in the current JIT tier execute via native code.
pub const TIERING_THRESHOLD: u32 = 100;

/// Invocation-count threshold that triggers Maglev JIT compilation.
///
/// When a function's `invocation_count` reaches this value and baseline JIT
/// code is already present, the interpreter schedules a background Maglev
/// compilation.  Once compilation finishes the cached Maglev code replaces
/// the baseline tier for future calls.
pub const MAGLEV_TIERING_THRESHOLD: u32 = 1_000;

/// Invocation-count threshold that triggers Turbofan (Cranelift optimising)
/// JIT compilation.
///
/// When a function's `invocation_count` reaches this value a background
/// Turbofan compilation is scheduled.  Turbofan runs the full Maglev graph
/// builder followed by Cranelift CLIF lowering and optimisation, producing
/// code that is expected to reach within 90 % of peak throughput.
pub const TURBOFAN_TIERING_THRESHOLD: u32 = 10_000;

/// An immutable, compact representation of the bytecode for a single
/// JavaScript function.
///
/// The raw bytes are the V8 Ignition-style encoding produced by
/// [`bytecodes::encode`].  Use [`BytecodeArray::instructions`] to decode them
/// back into a [`Vec<Instruction>`] when needed.
///
/// In addition to the static bytecode and metadata, a `BytecodeArray` carries
/// **tiering state** that is shared across all clones via [`Rc`] (baseline)
/// or [`Arc`] (Maglev, shared with the background compilation thread):
///
/// - An *invocation counter* that is incremented on every call.
/// - A *baseline JIT code cache* once invocation count reaches [`TIERING_THRESHOLD`].
/// - A *Maglev JIT code cache* once invocation count reaches [`MAGLEV_TIERING_THRESHOLD`].
#[derive(Debug, Clone)]
pub struct BytecodeArray {
    /// The encoded bytecode stream.
    bytecodes: Vec<u8>,
    /// Literals referenced by [`bytecodes::Opcode::LdaConstant`].
    constant_pool: Vec<ConstantPoolEntry>,
    /// Number of virtual registers (locals + temporaries) required.
    frame_size: u32,
    /// Number of formal parameters declared by the function.
    parameter_count: u32,
    /// Sparse mapping from bytecode offsets to source locations.
    source_positions: Vec<SourcePosition>,
    /// Compile-time description of all inline-cache feedback slots.
    feedback_metadata: FeedbackMetadata,
    /// Per-function exception handler table.
    handler_table: Vec<HandlerTableEntry>,
    /// `true` if this bytecode belongs to a generator function (`function*`).
    ///
    /// When a generator function is *called*, the interpreter creates a fresh
    /// [`crate::objects::value::GeneratorState`] and returns it as
    /// [`crate::objects::value::JsValue::Generator`] without executing the
    /// body immediately.
    is_generator: bool,
    /// `true` if this bytecode belongs to an async function or async generator.
    is_async: bool,
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
    /// Cached Maglev-JIT machine code and register-file slot count.
    ///
    /// Uses [`Arc`] + [`Mutex`] so the background Maglev compilation thread
    /// can write results while the interpreter runs on the main thread.
    /// `None` until Maglev compilation finishes successfully.
    maglev_jit_code: MaglevJitCodeCache,
    /// Set to `true` (via compare-exchange) when a Maglev compilation has been
    /// scheduled so that only one background thread is spawned per function.
    maglev_compile_started: Arc<AtomicBool>,
    /// Cached Turbofan (Cranelift optimising) JIT compiled code.
    ///
    /// Uses [`Arc`] + [`Mutex`] so the background Turbofan compilation thread
    /// can write results while the interpreter runs on the main thread.
    /// `None` until Turbofan compilation finishes successfully.
    turbofan_jit_code: TurbofanJitCodeCache,
    /// Set to `true` (via compare-exchange) when a Turbofan compilation has
    /// been scheduled so that only one background thread is spawned per
    /// function.
    turbofan_compile_started: Arc<AtomicBool>,
}

impl PartialEq for BytecodeArray {
    /// Two [`BytecodeArray`]s are equal when their static bytecode and metadata
    /// are identical.  The tiering state (`invocation_count`, `jit_code`,
    /// `maglev_jit_code`, `turbofan_jit_code`) is intentionally excluded from
    /// the comparison.
    fn eq(&self, other: &Self) -> bool {
        self.bytecodes == other.bytecodes
            && self.constant_pool == other.constant_pool
            && self.frame_size == other.frame_size
            && self.parameter_count == other.parameter_count
            && self.source_positions == other.source_positions
            && self.feedback_metadata == other.feedback_metadata
            && self.handler_table == other.handler_table
            && self.is_generator == other.is_generator
            && self.is_async == other.is_async
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
        Self {
            bytecodes,
            constant_pool,
            frame_size,
            parameter_count,
            source_positions,
            feedback_metadata,
            handler_table,
            is_generator: false,
            is_async: false,
            invocation_count: Rc::new(Cell::new(0)),
            jit_code: Rc::new(RefCell::new(None)),
            maglev_jit_code: Arc::new(Mutex::new(None)),
            maglev_compile_started: Arc::new(AtomicBool::new(false)),
            turbofan_jit_code: Arc::new(Mutex::new(None)),
            turbofan_compile_started: Arc::new(AtomicBool::new(false)),
        }
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

    /// The source-position table (may be empty if debug info was stripped).
    pub fn source_positions(&self) -> &[SourcePosition] {
        &self.source_positions
    }

    /// The compile-time feedback metadata for all inline-cache slots.
    pub fn feedback_metadata(&self) -> &FeedbackMetadata {
        &self.feedback_metadata
    }

    /// The per-function exception handler table.
    ///
    /// Each entry maps a `[try_start, try_end)` instruction-index range to a
    /// handler entry point.  Entries are ordered so that the innermost handler
    /// for any given instruction always appears before outer handlers.
    pub fn handler_table(&self) -> &[HandlerTableEntry] {
        &self.handler_table
    }

    /// Decode the bytecode stream and return the list of [`Instruction`]s.
    ///
    /// Returns an error if the byte stream is malformed.
    pub fn instructions(&self) -> StatorResult<Vec<Instruction>> {
        bytecodes::decode(&self.bytecodes)
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
    pub fn increment_invocation_count(&self) -> u32 {
        let new = self.invocation_count.get().saturating_add(1);
        self.invocation_count.set(new);
        new
    }

    /// Returns the current invocation count without modifying it.
    pub fn invocation_count(&self) -> u32 {
        self.invocation_count.get()
    }

    /// Store baseline-JIT machine code produced by the compiler.
    ///
    /// `code` is the raw x86-64 code buffer (including metadata tables
    /// appended by [`BaselineCompiler`][crate::compiler::baseline::compiler::BaselineCompiler]).
    /// `register_file_slots` is the number of `i64` slots required by the
    /// JIT's register file (`parameter_count + frame_size`).
    ///
    /// All clones of this [`BytecodeArray`] share the same JIT cache.
    pub fn store_jit_code(&self, code: Vec<u8>, register_file_slots: usize) {
        *self.jit_code.borrow_mut() = Some((code, register_file_slots));
    }

    /// Returns a clone of the cached JIT machine code and register-file slot
    /// count, or `None` if baseline compilation has not been triggered yet.
    ///
    /// The caller is responsible for ensuring that the code bytes are executed
    /// only on the platform and CPU that produced them.
    pub fn try_get_jit_code(&self) -> Option<(Vec<u8>, usize)> {
        self.jit_code.borrow().clone()
    }

    /// Returns a clone of the cached Maglev-JIT machine code and
    /// register-file slot count, or `None` if Maglev compilation has not
    /// finished yet.
    pub fn try_get_maglev_jit_code(&self) -> Option<(Vec<u8>, usize)> {
        self.maglev_jit_code.lock().ok()?.clone()
    }

    /// Returns an [`Arc`] clone of the Maglev JIT code cache.
    ///
    /// The background compilation thread receives this `Arc` and writes the
    /// compiled code into it when compilation succeeds.
    pub fn maglev_jit_cache_arc(&self) -> MaglevJitCodeCache {
        Arc::clone(&self.maglev_jit_code)
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;

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
}

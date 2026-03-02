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

use crate::bytecode::bytecodes::{self, Instruction};
use crate::bytecode::feedback::FeedbackMetadata;
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

/// An immutable, compact representation of the bytecode for a single
/// JavaScript function.
///
/// The raw bytes are the V8 Ignition-style encoding produced by
/// [`bytecodes::encode`].  Use [`BytecodeArray::instructions`] to decode them
/// back into a [`Vec<Instruction>`] when needed.
#[derive(Debug, Clone, PartialEq)]
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

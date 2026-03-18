//! Bytecode module: instruction set and encoding for the Stator VM.
//!
//! The instruction set matches V8 Ignition semantics (~174 opcodes).
//!
//! - [`bytecodes`] — Opcode table, operand encoding, [`bytecodes::Instruction`]
//!   type, and [`bytecodes::encode`]/[`bytecodes::decode`] utilities.
//! - [`bytecode_array`] — [`bytecode_array::BytecodeArray`]: compact bytecode
//!   storage, constant pool, frame metadata, and source-position table.
//! - [`register`] — [`register::Register`] virtual-register type and
//!   [`register::RegisterAllocator`] for register assignment during
//!   compilation.
//! - [`bytecode_generator`] — [`bytecode_generator::BytecodeGenerator`]:
//!   AST-to-bytecode compiler that produces a [`bytecode_array::BytecodeArray`]
//!   from a [`crate::parser::ast::Program`].
//! - [`feedback`] — [`feedback::FeedbackVector`], [`feedback::FeedbackMetadata`],
//!   [`feedback::FeedbackSlotKind`], and [`feedback::InlineCacheState`]: inline
//!   cache feedback slots allocated during compilation and updated at runtime.
//! - [`source_positions`] — [`source_positions::SourcePositionTable`] and
//!   [`source_positions::SourcePositionEntry`]: maps bytecode offsets to source
//!   locations for stack traces, debugger stepping, and source maps.

/// Compact bytecode storage with constant pool and source-position table.
pub mod bytecode_array;
/// Bytecode generator: compiles a JavaScript AST into a [`bytecode_array::BytecodeArray`].
pub mod bytecode_generator;
/// Bytecode instruction set, operand encoding, and encode/decode utilities.
pub mod bytecodes;
/// Feedback vectors and inline-cache state for adaptive optimisation.
pub mod feedback;
/// Peephole fusion pass for decoded bytecode streams.
pub mod peephole;
/// Virtual register type and register allocator for bytecode compilation.
pub mod register;
/// Source-position table: maps bytecode offsets to source locations for stack
/// traces, debugger stepping, and source maps.
pub mod source_positions;

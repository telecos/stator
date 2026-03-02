//! Bytecode module: instruction set and encoding for the Stator VM.
//!
//! The instruction set matches V8 Ignition semantics (~174 opcodes).
//!
//! - [`bytecodes`] — Opcode table, operand encoding, [`bytecodes::Instruction`]
//!   type, and [`bytecodes::encode`]/[`bytecodes::decode`] utilities.
//! - [`bytecode_array`] — [`bytecode_array::BytecodeArray`]: compact bytecode
//!   storage, constant pool, frame metadata, and source-position table.

/// Compact bytecode storage with constant pool and source-position table.
pub mod bytecode_array;
/// Bytecode instruction set, operand encoding, and encode/decode utilities.
pub mod bytecodes;

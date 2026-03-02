//! Bytecode module: instruction set and encoding for the Stator VM.
//!
//! The instruction set matches V8 Ignition semantics (~174 opcodes).
//!
//! - [`bytecodes`] — Opcode table, operand encoding, [`bytecodes::Instruction`]
//!   type, and [`bytecodes::encode`]/[`bytecodes::decode`] utilities.

/// Bytecode instruction set, operand encoding, and encode/decode utilities.
pub mod bytecodes;

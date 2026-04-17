//! Baseline (non-optimising) JIT compiler.
//!
//! - [`masm_x64`] — x86-64 macro-assembler: emit machine code into a buffer,
//!   label binding/patching, register encoding, and RIP-relative addressing.
//! - [`compiler`] — [`compiler::BaselineCompiler`]: walks a
//!   [`crate::bytecode::bytecode_array::BytecodeArray`], emits native x86-64
//!   machine code per bytecode instruction, maps virtual registers to
//!   register-file slots, and generates a safepoint table and deopt metadata.

/// Baseline JIT compiler: bytecode → machine code.
pub mod compiler;
/// x86-64 macro-assembler for the baseline compiler.
pub mod masm_x64;

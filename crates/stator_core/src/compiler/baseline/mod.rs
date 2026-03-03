//! Baseline (non-optimising) JIT compiler.
//!
//! - [`masm_x64`] — x86-64 macro-assembler: emit machine code into a buffer,
//!   label binding/patching, register encoding, and RIP-relative addressing.

/// x86-64 macro-assembler for the baseline compiler.
pub mod masm_x64;

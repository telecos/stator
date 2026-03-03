//! Compiler infrastructure for the Stator JIT.
//!
//! # Crate layout
//!
//! - [`baseline`] — Non-optimising baseline JIT compiler tier.
//!   - [`baseline::masm_x64`] — x86-64 macro-assembler: emits raw machine code
//!     into a byte buffer with label patching and RIP-relative addressing.

/// Non-optimising baseline JIT compiler.
pub mod baseline;

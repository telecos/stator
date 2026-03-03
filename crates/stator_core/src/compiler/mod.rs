//! Compiler infrastructure for the Stator JIT.
//!
//! # Crate layout
//!
//! - [`baseline`] — Non-optimising baseline JIT compiler tier.
//!   - [`baseline::masm_x64`] — x86-64 macro-assembler: emits raw machine code
//!     into a byte buffer with label patching and RIP-relative addressing.
//! - [`maglev`] — Optimising Maglev compiler tier.
//!   - [`maglev::ir`] — Typed IR node types ([`maglev::ir::ValueNode`],
//!     [`maglev::ir::ControlNode`], [`maglev::ir::BasicBlock`],
//!     [`maglev::ir::MaglevGraph`]).
//!   - [`maglev::graph_builder`] — Bytecode-to-IR graph builder
//!     ([`maglev::graph_builder::GraphBuilder`]).
//!   - [`maglev::optimizer`] — Optimisation passes (constant folding, DCE,
//!     redundant-CheckMaps removal).
//!   - [`maglev::regalloc`] — Linear-scan register allocator.
//!   - [`maglev::codegen`] — Code generator: walks a register-allocated
//!     [`maglev::ir::MaglevGraph`] and emits x86-64 machine code.

/// Non-optimising baseline JIT compiler.
pub mod baseline;
/// Maglev optimising compiler tier.
pub mod maglev;

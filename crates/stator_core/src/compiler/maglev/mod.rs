//! Maglev optimising compiler tier.
//!
//! # Modules
//!
//! - [`ir`] — Typed IR node types: [`ir::ValueNode`], [`ir::ControlNode`],
//!   [`ir::BasicBlock`], and [`ir::MaglevGraph`].
//! - [`graph_builder`] — Bytecode-to-IR graph builder: walks a
//!   [`crate::bytecode::bytecode_array::BytecodeArray`] together with a
//!   [`crate::bytecode::feedback::FeedbackVector`] and emits a
//!   [`ir::MaglevGraph`] with speculative type guards.

/// Code generator: walk register-allocated [`ir::MaglevGraph`] and emit
/// x86-64 machine code.
pub mod codegen;
/// Bytecode-to-IR graph builder.
pub mod graph_builder;
/// Typed IR node definitions for the Maglev compiler.
pub mod ir;
/// Optimisation passes: constant folding, DCE, redundant-CheckMaps removal.
pub mod optimizer;
/// Linear-scan register allocator over [`ir::MaglevGraph`].
pub mod regalloc;

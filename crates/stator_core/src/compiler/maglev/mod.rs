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

/// Bytecode-to-IR graph builder.
pub mod graph_builder;
/// Typed IR node definitions for the Maglev compiler.
pub mod ir;

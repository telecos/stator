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
//! - [`type_specialization`] — Replaces generic arithmetic with typed
//!   fast-path equivalents using IC feedback.
//! - [`range_analysis`] — Integer range analysis that eliminates
//!   unnecessary overflow-check deoptimisations.
//! - [`licm`] — Loop-invariant code motion: hoists pure invariant nodes
//!   out of natural loops.
//! - [`type_guards`] — Inserts type-guard nodes with deoptimisation
//!   bailout before unguarded typed arithmetic.

/// Code generator: walk register-allocated [`ir::MaglevGraph`] and emit
/// x86-64 machine code.
pub mod codegen;
/// Deoptimiser: JIT → interpreter fallback on speculation failure.
pub mod deopt;
/// Bytecode-to-IR graph builder.
pub mod graph_builder;
/// Typed IR node definitions for the Maglev compiler.
pub mod ir;
/// Loop-invariant code motion (LICM).
pub mod licm;
/// Optimisation passes: constant folding, DCE, redundant-CheckMaps removal.
pub mod optimizer;
/// Integer range analysis for overflow-check elimination.
pub mod range_analysis;
/// Linear-scan register allocator over [`ir::MaglevGraph`].
pub mod regalloc;
/// Type-guard insertion with deoptimisation bailout.
pub mod type_guards;
/// Type specialisation from inline-cache (IC) feedback.
pub mod type_specialization;

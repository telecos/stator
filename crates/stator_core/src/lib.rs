//! `stator_core` — the foundational engine library for the Stator JavaScript
//! engine.
//!
//! # Crate layout
//!
//! - [`error`] — Engine error types and `StatorResult` alias.
//! - [`gc`] — Garbage collector infrastructure (heap, tracing, handles).
//! - [`objects`] — JavaScript value representation and heap object types.
//! - [`zone`] — Bump-pointer region allocator for compiler temporaries.

/// Engine error types and [`StatorResult`] alias.
pub mod error;
/// Garbage collector infrastructure: heap, tracing, and handle scopes.
pub mod gc;
/// JavaScript value representation and heap object types.
pub mod objects;
/// JavaScript parser infrastructure (lexer, future AST, …).
pub mod parser;
/// Bump-pointer region allocator for compiler temporaries.
pub mod zone;

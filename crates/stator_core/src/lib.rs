//! `stator_core` — the foundational engine library for the Stator JavaScript
//! engine.
//!
//! # Crate layout
//!
//! - [`gc`] — Garbage collector infrastructure (heap, tracing, handles).
//! - [`objects`] — JavaScript value representation and heap object types.

/// Garbage collector infrastructure: heap, tracing, and handle scopes.
pub mod gc;
/// JavaScript value representation and heap object types.
pub mod objects;

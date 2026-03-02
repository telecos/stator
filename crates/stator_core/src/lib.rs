//! `stator_core` — the foundational engine library for the Stator JavaScript
//! engine.
//!
//! # Crate layout
//!
//! - [`error`] — Engine error types and `StatorResult` alias.
//! - [`gc`] — Garbage collector infrastructure (heap, tracing, handles).
//! - [`objects`] — JavaScript value representation and heap object types.
//! - [`zone`] — Bump-pointer region allocator for compiler temporaries.
//! - [`parser`] — Lexer ([`parser::scanner`]), AST ([`parser::ast`]), and
//!   scope analysis ([`parser::scope`]).
//! - [`bytecode`] — Bytecode instruction set ([`bytecode::bytecodes`]):
//!   ~174 opcodes matching V8 Ignition semantics, operand encoding, and
//!   encode/decode utilities.

/// Bytecode instruction set and encode/decode utilities.
pub mod bytecode;
/// Engine error types and [`StatorResult`] alias.
pub mod error;
/// Garbage collector infrastructure: heap, tracing, and handle scopes.
pub mod gc;
/// JavaScript value representation and heap object types.
pub mod objects;
/// JavaScript parser infrastructure (lexer and AST node definitions).
pub mod parser;
/// Bump-pointer region allocator for compiler temporaries.
pub mod zone;

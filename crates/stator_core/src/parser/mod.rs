//! JavaScript parser infrastructure.
//!
//! - [`scanner`] — ES2025 lexer that converts raw source text into a
//!   stream of [`scanner::Token`]s.
//! - [`ast`] — ES2025 Abstract Syntax Tree node definitions.
//! - [`scope`] — Scope analysis and variable resolution.

/// ES2025 Abstract Syntax Tree node types.
pub mod ast;
/// ES2025 JavaScript lexer.
pub mod scanner;
/// Scope analysis and variable resolution.
pub mod scope;

//! JavaScript parser infrastructure.
//!
//! - [`scanner`] — ES2025 lexer that converts raw source text into a
//!   stream of [`scanner::Token`]s.
//! - [`ast`] — ES2025 Abstract Syntax Tree node definitions.
//! - [`scope`] — Scope analysis and variable resolution.
//! - [`preparser`] — Lazy parsing: fast scan that records
//!   [`preparser::LazyCompileData`] for each function body without building a
//!   full AST.
//! - [`recursive_descent`] — Minimal recursive-descent parser: converts source
//!   text into a [`ast::Program`] AST.

/// ES2025 Abstract Syntax Tree node types.
pub mod ast;
/// Lazy parsing (pre-parser): fast function-body scan without full AST.
pub mod preparser;
/// Minimal recursive-descent JavaScript parser.
pub mod recursive_descent;
/// ES2025 JavaScript lexer.
pub mod scanner;
/// Scope analysis and variable resolution.
pub mod scope;

pub use recursive_descent::parse;

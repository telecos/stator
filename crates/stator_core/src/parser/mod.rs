//! JavaScript parser infrastructure.
//!
//! - [`scanner`] — ES2025 lexer that converts raw source text into a
//!   stream of [`scanner::Token`]s.

/// ES2025 JavaScript lexer.
pub mod scanner;

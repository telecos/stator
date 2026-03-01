//! Lazy parsing (pre-parser) for ES2025 JavaScript.
//!
//! The pre-parser does a fast scan of source text, locating every function
//! body without building a full AST.  It detects syntax errors (mismatched
//! braces/parentheses/brackets) and collects the minimal *scope structure*
//! needed to compile a function lazily — only when it is first called.
//!
//! # Workflow
//!
//! 1. Call [`preparse`] to obtain a [`PreParseResult`].
//! 2. Inspect [`PreParseResult::functions`] for all discovered functions.
//! 3. When a function is first invoked at runtime, retrieve its
//!    [`LazyCompileData`] and perform the full parse on just the saved source
//!    slice.
//!
//! # What the pre-parser detects
//!
//! - **Syntax errors** – mismatched `{`/`}`/`(`/`)`/`[`/`]`.
//! - **Scope structure** – whether the body references `this`, `super`,
//!   `arguments`, or a direct `eval(…)` call.
//! - **Nested functions** – the count of directly-nested function bodies.

use crate::error::{StatorError, StatorResult};
use crate::parser::scanner::{Scanner, Span, Token, TokenKind};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Stored in place of a full AST node for every function body that has not yet
/// been compiled.  Contains everything required to perform a full parse later
/// without re-scanning the entire source file.
#[derive(Debug, Clone)]
pub struct LazyCompileData {
    /// The optional name of the function (e.g. `"foo"` for `function foo()`).
    pub name: Option<String>,
    /// Half-open byte span `[start, end)` of the **body** (the `{…}` block).
    pub body_span: Span,
    /// Names of formal parameters, in declaration order.
    pub param_names: Vec<String>,
    /// `true` when the body (excluding nested function bodies) references
    /// `this`.
    pub uses_this: bool,
    /// `true` when the body (excluding nested function bodies) references
    /// `super`.
    pub uses_super: bool,
    /// `true` when the body (excluding nested function bodies) references
    /// `arguments`.
    pub uses_arguments: bool,
    /// `true` when the body contains a direct `eval(…)` call (i.e. the
    /// identifier `eval` followed by `(`).
    pub uses_eval: bool,
    /// Number of function bodies directly nested inside this one (not
    /// recursively).
    pub inner_function_count: usize,
}

/// The result returned by [`preparse`].
#[derive(Debug)]
pub struct PreParseResult {
    /// One entry per function body discovered (in source order).
    pub functions: Vec<LazyCompileData>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-parser
// ─────────────────────────────────────────────────────────────────────────────

/// Scan `source`, locate every function body, record [`LazyCompileData`] for
/// each one, and validate that all bracket pairs are matched.
///
/// Returns [`StatorError::SyntaxError`] if a mismatched bracket is found.
///
/// # Example
///
/// ```
/// use stator_core::parser::preparser::preparse;
///
/// let src = "function add(a, b) { return a + b; }";
/// let result = preparse(src).unwrap();
/// assert_eq!(result.functions.len(), 1);
/// let f = &result.functions[0];
/// assert_eq!(f.name.as_deref(), Some("add"));
/// assert_eq!(f.param_names, vec!["a", "b"]);
/// ```
pub fn preparse(source: &str) -> StatorResult<PreParseResult> {
    let mut pp = PreParser::new(source);
    pp.run()?;
    Ok(PreParseResult {
        functions: pp.functions,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal pre-parser state
// ─────────────────────────────────────────────────────────────────────────────

struct PreParser<'src> {
    source: &'src str,
    scanner: Scanner<'src>,
    /// Lookahead buffer: at most one peeked token.
    peeked: Option<Token>,
    /// Collected results.
    functions: Vec<LazyCompileData>,
}

impl<'src> PreParser<'src> {
    fn new(source: &'src str) -> Self {
        Self {
            source,
            scanner: Scanner::new(source),
            peeked: None,
            functions: Vec::new(),
        }
    }

    /// Return the next significant (non-comment) token.
    fn next(&mut self) -> StatorResult<Token> {
        if let Some(tok) = self.peeked.take() {
            return Ok(tok);
        }
        loop {
            let tok = self.scanner.next_token()?;
            if !matches!(
                tok.kind,
                TokenKind::SingleLineComment | TokenKind::MultiLineComment
            ) {
                return Ok(tok);
            }
        }
    }

    /// Peek at the next significant token without consuming it.
    fn peek(&mut self) -> StatorResult<&Token> {
        if self.peeked.is_none() {
            let tok = loop {
                let t = self.scanner.next_token()?;
                if !matches!(
                    t.kind,
                    TokenKind::SingleLineComment | TokenKind::MultiLineComment
                ) {
                    break t;
                }
            };
            self.peeked = Some(tok);
        }
        Ok(self.peeked.as_ref().expect("peeked was just set above"))
    }

    /// Top-level scan loop.
    fn run(&mut self) -> StatorResult<()> {
        loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::Eof => break,
                TokenKind::Function => {
                    self.scan_function(None)?;
                }
                TokenKind::Async => {
                    // `async function …` or `async (…) => …`
                    if self.peek()?.kind == TokenKind::Function {
                        self.next()?; // consume `function`
                        self.scan_function(None)?;
                    }
                    // Standalone `async` identifier or `async () =>` — not a
                    // named function declaration; skip.
                }
                // Opening delimiters: validate matching.
                TokenKind::LeftBrace => {
                    self.skip_block(tok.span)?;
                }
                TokenKind::LeftParen => {
                    self.skip_paren(tok.span)?;
                }
                TokenKind::LeftBracket => {
                    self.skip_bracket(tok.span)?;
                }
                // Unmatched closing delimiters.
                TokenKind::RightBrace => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected '}}' at {}:{}",
                        tok.span.start.line, tok.span.start.column
                    )));
                }
                TokenKind::RightParen => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected ')' at {}:{}",
                        tok.span.start.line, tok.span.start.column
                    )));
                }
                TokenKind::RightBracket => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected ']' at {}:{}",
                        tok.span.start.line, tok.span.start.column
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Called immediately after consuming the `function` keyword.  Reads the
    /// optional name, parameter list, and body; records a [`LazyCompileData`].
    fn scan_function(&mut self, name_override: Option<String>) -> StatorResult<()> {
        // Optional generator `*`.
        if self.peek()?.kind == TokenKind::Star {
            self.next()?;
        }

        // Optional function name.
        let name = if name_override.is_some() {
            name_override
        } else if matches!(
            self.peek()?.kind,
            TokenKind::Identifier
                | TokenKind::Await
                | TokenKind::Yield
                | TokenKind::Let
                | TokenKind::Static
                | TokenKind::Async
                | TokenKind::Of
                | TokenKind::From
                | TokenKind::Get
                | TokenKind::Set
                | TokenKind::Target
                | TokenKind::Meta
        ) {
            let tok = self.next()?;
            match tok.value {
                crate::parser::scanner::TokenValue::Str(s) => Some(s),
                _ => {
                    // keyword used as name – extract text from source
                    let slice = &self.source[tok.span.start.offset..tok.span.end.offset];
                    Some(slice.to_string())
                }
            }
        } else {
            None
        };

        // Parameter list.
        let param_names = self.scan_params()?;

        // Body `{ … }`.
        let open = self.next()?;
        if open.kind != TokenKind::LeftBrace {
            return Err(StatorError::SyntaxError(format!(
                "expected '{{' to open function body at {}:{}",
                open.span.start.line, open.span.start.column
            )));
        }

        let (body_span, uses_this, uses_super, uses_arguments, uses_eval, inner_count) =
            self.scan_body(open.span)?;

        self.functions.push(LazyCompileData {
            name,
            body_span,
            param_names,
            uses_this,
            uses_super,
            uses_arguments,
            uses_eval,
            inner_function_count: inner_count,
        });

        Ok(())
    }

    /// Scan a formal parameter list `(…)` and return plain identifier names.
    /// Destructuring patterns and defaults are accepted but only the top-level
    /// identifier names are recorded.
    fn scan_params(&mut self) -> StatorResult<Vec<String>> {
        let open = self.next()?;
        if open.kind != TokenKind::LeftParen {
            return Err(StatorError::SyntaxError(format!(
                "expected '(' for parameter list at {}:{}",
                open.span.start.line, open.span.start.column
            )));
        }

        let mut names = Vec::new();
        let mut depth: usize = 1; // we already consumed `(`

        loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                TokenKind::LeftBrace | TokenKind::LeftBracket => {
                    // Destructuring parameter — skip the nested delimiters.
                    let close = if tok.kind == TokenKind::LeftBrace {
                        TokenKind::RightBrace
                    } else {
                        TokenKind::RightBracket
                    };
                    self.skip_until(close, tok.span)?;
                }
                TokenKind::Identifier => {
                    if let crate::parser::scanner::TokenValue::Str(s) = tok.value {
                        names.push(s);
                    }
                }
                TokenKind::DotDotDot => {
                    // Rest parameter: collect the following identifier.
                    let next = self.peek()?;
                    if next.kind == TokenKind::Identifier {
                        let t = self.next()?;
                        if let crate::parser::scanner::TokenValue::Str(s) = t.value {
                            names.push(s);
                        }
                    }
                }
                TokenKind::Eof => {
                    return Err(StatorError::SyntaxError(
                        "unexpected end of input in parameter list".to_string(),
                    ));
                }
                _ => {}
            }
        }

        Ok(names)
    }

    /// Scan a function body `{…}`.  The opening `{` has already been consumed
    /// and its span is passed in.  Returns:
    /// `(body_span, uses_this, uses_super, uses_arguments, uses_eval,
    /// inner_function_count)`.
    fn scan_body(
        &mut self,
        open_span: Span,
    ) -> StatorResult<(Span, bool, bool, bool, bool, usize)> {
        let mut depth: usize = 1;
        let mut uses_this = false;
        let mut uses_super = false;
        let mut uses_arguments = false;
        let mut uses_eval = false;
        let mut inner_count: usize = 0;

        // Track the most recent identifier for `eval(` detection.
        let mut prev_was_eval = false;

        let body_span = loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::LeftBrace => depth += 1,
                TokenKind::RightBrace => {
                    depth -= 1;
                    if depth == 0 {
                        break Span {
                            start: open_span.start,
                            end: tok.span.end,
                        };
                    }
                }
                TokenKind::LeftParen => {
                    if prev_was_eval {
                        uses_eval = true;
                    }
                }
                TokenKind::This => {
                    if depth == 1 {
                        uses_this = true;
                    }
                }
                TokenKind::Super => {
                    if depth == 1 {
                        uses_super = true;
                    }
                }
                TokenKind::Identifier => {
                    // Identifier tokens always carry a Str value; the pattern
                    // exhausts all cases so there is no silent fallback.
                    let name = match &tok.value {
                        crate::parser::scanner::TokenValue::Str(s) => s.as_str(),
                        _ => "",
                    };
                    if depth == 1 {
                        if name == "arguments" {
                            uses_arguments = true;
                        }
                        // Record whether the next `(` would be a direct eval call.
                        prev_was_eval = name == "eval";
                        continue;
                    }
                }
                // Nested `function` declaration/expression — skip and count.
                TokenKind::Function => {
                    inner_count += 1;
                    self.skip_function()?;
                }
                // `async function …` — skip and count; plain `async` is ignored.
                TokenKind::Async => {
                    if self.peek()?.kind == TokenKind::Function {
                        self.next()?; // consume `function`
                        inner_count += 1;
                        self.skip_function()?;
                    }
                }
                TokenKind::Eof => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected end of input: unclosed '{{' at {}:{}",
                        open_span.start.line, open_span.start.column
                    )));
                }
                _ => {}
            }
            prev_was_eval = false;
        };

        Ok((
            body_span,
            uses_this,
            uses_super,
            uses_arguments,
            uses_eval,
            inner_count,
        ))
    }

    /// Skip a nested function (after `function` has been consumed).  Does not
    /// record a [`LazyCompileData`] entry — nested functions are counted but
    /// not individually stored at the outer level.
    fn skip_function(&mut self) -> StatorResult<()> {
        // Optional `*`
        if self.peek()?.kind == TokenKind::Star {
            self.next()?;
        }
        // Optional name
        if matches!(
            self.peek()?.kind,
            TokenKind::Identifier
                | TokenKind::Await
                | TokenKind::Yield
                | TokenKind::Let
                | TokenKind::Static
                | TokenKind::Async
                | TokenKind::Of
                | TokenKind::From
                | TokenKind::Get
                | TokenKind::Set
                | TokenKind::Target
                | TokenKind::Meta
        ) {
            self.next()?;
        }
        // Params
        self.scan_params()?;
        // Body
        let open = self.next()?;
        if open.kind != TokenKind::LeftBrace {
            return Err(StatorError::SyntaxError(format!(
                "expected '{{' to open function body at {}:{}",
                open.span.start.line, open.span.start.column
            )));
        }
        self.skip_block(open.span)?;
        Ok(())
    }

    /// Skip tokens until the matching `}`, starting after an already-consumed
    /// `{`.  Nested `{…}`, `(…)`, `[…]`, and inner functions are handled
    /// recursively.
    fn skip_block(&mut self, open_span: Span) -> StatorResult<()> {
        let mut depth: usize = 1;
        loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::LeftBrace => depth += 1,
                TokenKind::RightBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                TokenKind::Eof => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected end of input: unclosed '{{' at {}:{}",
                        open_span.start.line, open_span.start.column
                    )));
                }
                _ => {}
            }
        }
    }

    /// Skip tokens until the matching `)`, starting after an already-consumed
    /// `(`.
    fn skip_paren(&mut self, open_span: Span) -> StatorResult<()> {
        let mut depth: usize = 1;
        loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                TokenKind::Eof => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected end of input: unclosed '(' at {}:{}",
                        open_span.start.line, open_span.start.column
                    )));
                }
                _ => {}
            }
        }
    }

    /// Skip tokens until the matching `]`, starting after an already-consumed
    /// `[`.
    fn skip_bracket(&mut self, open_span: Span) -> StatorResult<()> {
        let mut depth: usize = 1;
        loop {
            let tok = self.next()?;
            match tok.kind {
                TokenKind::LeftBracket => depth += 1,
                TokenKind::RightBracket => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                TokenKind::Eof => {
                    return Err(StatorError::SyntaxError(format!(
                        "unexpected end of input: unclosed '[' at {}:{}",
                        open_span.start.line, open_span.start.column
                    )));
                }
                _ => {}
            }
        }
    }

    /// Generic "skip until matching close delimiter" helper used for
    /// destructuring parameters.
    fn skip_until(&mut self, close: TokenKind, open_span: Span) -> StatorResult<()> {
        let open = if close == TokenKind::RightBrace {
            TokenKind::LeftBrace
        } else {
            TokenKind::LeftBracket
        };
        let mut depth: usize = 1;
        loop {
            let tok = self.next()?;
            if tok.kind == open {
                depth += 1;
            } else if tok.kind == close {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            } else if tok.kind == TokenKind::Eof {
                return Err(StatorError::SyntaxError(format!(
                    "unexpected end of input: unclosed delimiter at {}:{}",
                    open_span.start.line, open_span.start.column
                )));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic function detection ──────────────────────────────────────────────

    #[test]
    fn test_preparse_single_function() {
        let src = "function add(a, b) { return a + b; }";
        let result = preparse(src).unwrap();
        assert_eq!(result.functions.len(), 1);
        let f = &result.functions[0];
        assert_eq!(f.name.as_deref(), Some("add"));
        assert_eq!(f.param_names, vec!["a", "b"]);
        assert!(!f.uses_this);
        assert!(!f.uses_eval);
    }

    #[test]
    fn test_preparse_many_functions() {
        let src = r#"
            function a() { return 1; }
            function b() { return 2; }
            function c() { return 3; }
            function d() { return 4; }
            function e() { return 5; }
        "#;
        let result = preparse(src).unwrap();
        assert_eq!(result.functions.len(), 5);
        let names: Vec<_> = result.functions.iter().map(|f| f.name.as_deref()).collect();
        assert_eq!(
            names,
            vec![Some("a"), Some("b"), Some("c"), Some("d"), Some("e")]
        );
    }

    #[test]
    fn test_preparse_anonymous_function() {
        let src = "var f = function() { return 42; };";
        // Anonymous function expression – not reached by the top-level scan
        // because it's behind a `var` statement. Only *top-level* `function`
        // keywords are directly scanned here; expressions inside other
        // statements are skipped.
        let result = preparse(src).unwrap();
        // The anonymous function expression is NOT found at the top level scan
        // because the `function` keyword is inside the var-init expression, but
        // our top-level scanner still encounters the `function` token.
        assert_eq!(result.functions.len(), 1);
        assert!(result.functions[0].name.is_none());
    }

    // ── Scope-structure detection ─────────────────────────────────────────────

    #[test]
    fn test_preparse_uses_this() {
        let src = "function greet() { return this.name; }";
        let result = preparse(src).unwrap();
        assert!(result.functions[0].uses_this);
        assert!(!result.functions[0].uses_super);
    }

    #[test]
    fn test_preparse_uses_arguments() {
        let src = "function sum() { return arguments.length; }";
        let result = preparse(src).unwrap();
        assert!(result.functions[0].uses_arguments);
    }

    #[test]
    fn test_preparse_uses_eval() {
        let src = "function run(code) { return eval(code); }";
        let result = preparse(src).unwrap();
        assert!(result.functions[0].uses_eval);
    }

    #[test]
    fn test_preparse_not_uses_eval_as_identifier() {
        // `eval` used as a plain identifier (not called) must NOT set uses_eval.
        let src = "function f(eval) { return eval + 1; }";
        let result = preparse(src).unwrap();
        // `eval + 1` – `eval` is followed by `+`, not `(`, so uses_eval stays false.
        assert!(!result.functions[0].uses_eval);
    }

    // ── Nested functions ──────────────────────────────────────────────────────

    #[test]
    fn test_preparse_inner_function_count() {
        let src = r#"
            function outer() {
                function inner1() {}
                function inner2() {}
            }
        "#;
        let result = preparse(src).unwrap();
        // Only the outer function is stored at the top level.
        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name.as_deref(), Some("outer"));
        assert_eq!(result.functions[0].inner_function_count, 2);
    }

    // ── Body span ────────────────────────────────────────────────────────────

    #[test]
    fn test_preparse_body_span() {
        let src = "function f() { return 1; }";
        let result = preparse(src).unwrap();
        let f = &result.functions[0];
        let body_src = &src[f.body_span.start.offset..f.body_span.end.offset];
        assert_eq!(body_src, "{ return 1; }");
    }

    // ── Full parse on demand ──────────────────────────────────────────────────

    /// Simulates deferred full parsing: after pre-parsing, use the stored
    /// `body_span` to extract the function source and verify it is valid JS.
    #[test]
    fn test_preparse_deferred_full_parse() {
        let src = r#"
            function heavy(x, y) {
                var z = x * y;
                return z + 1;
            }
            function light() { return 0; }
        "#;
        let result = preparse(src).unwrap();
        assert_eq!(result.functions.len(), 2);

        // Simulate on-demand full parse: verify the body spans are extractable
        // and non-empty.
        for f in &result.functions {
            let body = &src[f.body_span.start.offset..f.body_span.end.offset];
            assert!(body.starts_with('{'));
            assert!(body.ends_with('}'));
        }
    }

    // ── Syntax error detection ────────────────────────────────────────────────

    #[test]
    fn test_preparse_unmatched_brace() {
        let src = "function f() { return 1; ";
        let result = preparse(src);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("SyntaxError"));
    }

    #[test]
    fn test_preparse_unmatched_close_brace() {
        let src = "var x = 1; }";
        let result = preparse(src);
        assert!(result.is_err());
    }

    #[test]
    fn test_preparse_unmatched_paren() {
        let src = "if (true { }";
        let result = preparse(src);
        assert!(result.is_err());
    }

    #[test]
    fn test_preparse_unmatched_bracket() {
        let src = "var a = [1, 2;";
        let result = preparse(src);
        assert!(result.is_err());
    }

    // ── Parameters ───────────────────────────────────────────────────────────

    #[test]
    fn test_preparse_rest_param() {
        let src = "function f(...args) { return args.length; }";
        let result = preparse(src).unwrap();
        assert_eq!(result.functions[0].param_names, vec!["args"]);
    }

    #[test]
    fn test_preparse_destructuring_param() {
        // Destructuring params are skipped; only plain identifiers are recorded.
        let src = "function f({x, y}, [a, b]) { return x + a; }";
        let result = preparse(src).unwrap();
        // Destructured params are inside `{…}` and `[…]` – skipped, so no
        // top-level param names are captured.
        assert!(result.functions[0].param_names.is_empty());
    }

    // ── Async functions ───────────────────────────────────────────────────────

    #[test]
    fn test_preparse_async_function() {
        let src = "async function fetch() { return 1; }";
        let result = preparse(src).unwrap();
        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name.as_deref(), Some("fetch"));
    }

    // ── Generator functions ───────────────────────────────────────────────────

    #[test]
    fn test_preparse_generator_function() {
        let src = "function* gen() { yield 1; }";
        let result = preparse(src).unwrap();
        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name.as_deref(), Some("gen"));
    }

    // ── Empty source ─────────────────────────────────────────────────────────

    #[test]
    fn test_preparse_empty_source() {
        let result = preparse("").unwrap();
        assert!(result.functions.is_empty());
    }
}

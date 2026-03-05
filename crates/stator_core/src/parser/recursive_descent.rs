//! Minimal recursive-descent JavaScript parser.
//!
//! Converts a UTF-8 source string into a [`crate::parser::ast::Program`] using
//! the scanner from [`crate::parser::scanner`] for tokenisation.
//!
//! # Supported syntax
//!
//! This parser implements a subset of ES2025 sufficient for Phase 2 demos:
//!
//! - Variable declarations (`var`, `let`, `const`) with simple identifier
//!   bindings and optional initialisers.
//! - Expression statements with a broad expression grammar:
//!   - Numeric, string, boolean, `null` literals and identifiers.
//!   - All binary arithmetic, comparison, bitwise, and logical operators.
//!   - Assignment operators.
//!   - Conditional (`?:`) and sequence (`,`) expressions.
//!   - Unary prefix operators (`-`, `+`, `!`, `~`, `typeof`, `void`).
//!   - Postfix `++` / `--`.
//!   - Parenthesised expressions.
//! - Empty statements (`;`).
//! - Automatic Semicolon Insertion (ASI) at statement boundaries.
//!
//! Everything else (class bodies, template literals, spread, generators, etc.)
//! is rejected with an explicit `SyntaxError`.

use crate::error::{StatorError, StatorResult};
use crate::parser::ast::{
    ArrayExpr, ArrayPat, ArrowBody, ArrowExpr, AssignExpr, AssignOp, AssignPat, AssignPatProp,
    AssignTarget, AwaitExpr, BinaryExpr, BinaryOp, BlockStmt, BoolLit, BreakStmt, ClassBody,
    ClassDecl, ClassExpr, ClassMember, ContinueStmt, DebuggerStmt, DoWhileStmt, EmptyStmt,
    ExportAllDecl, ExportDefaultDecl, ExportDefaultExpr, ExportNamedDecl, ExportSpecifier, Expr,
    ExprStmt, FnDecl, FnExpr, ForInOfLeft, ForInStmt, ForInit, ForOfStmt, ForStmt, Ident, IfStmt,
    ImportDecl, ImportDefaultSpecifier, ImportNamedSpecifier, ImportNamespaceSpecifier,
    ImportSpecifier, KeyValuePatProp, LogicalExpr, LogicalOp, MethodDef, MethodKind, ModuleDecl,
    ModuleExportName, NewExpr, NullLit, NumLit, ObjectExpr, ObjectPat, ObjectPatProp, ObjectProp,
    Param, Pat, Program, ProgramItem, Prop, PropKey, PropValue, RegExpLit, RestElement, ReturnStmt,
    SequenceExpr, SourceLocation, SourceType, SpreadElement, Stmt, StringLit, SwitchCase,
    SwitchStmt, TemplateElement, TemplateLit, ThrowStmt, TryStmt, UnaryExpr, UnaryOp, UpdateExpr,
    UpdateOp, VarDecl, VarDeclarator, VarKind, WhileStmt,
};
use crate::parser::scanner::{Scanner, Span, Token, TokenKind, TokenValue};

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Recursive-descent JavaScript parser.
///
/// Wraps a [`Scanner`] and maintains one token of lookahead in `current`.
/// Maximum recursion depth for the recursive-descent parser.
///
/// Pathological inputs (e.g. `((((((…` nested thousands of levels) would
/// otherwise blow the thread stack.  256 nesting levels is far beyond any
/// reasonable real-world program.
const MAX_RECURSION_DEPTH: usize = 256;

pub struct Parser<'src> {
    scanner: Scanner<'src>,
    /// The lookahead token (already produced by the scanner).
    current: Token,
    /// When `true`, the `in` keyword is **not** treated as a binary relational
    /// operator.  This is required inside the initialiser of a `for` statement
    /// so that `for (x in obj)` is parsed as a for-in loop instead of a
    /// relational `in` expression.
    no_in: bool,
    /// Current recursion depth – incremented on entry to key recursive
    /// functions and decremented on exit.
    depth: usize,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given UTF-8 source string.
    ///
    /// Primes the lookahead by reading the first non-comment token.
    fn new(source: &'src str) -> StatorResult<Self> {
        let mut scanner = Scanner::new(source);
        let current = Self::next_significant(&mut scanner)?;
        Ok(Self {
            scanner,
            current,
            no_in: false,
            depth: 0,
        })
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Read the next token from the scanner, skipping whitespace and comments.
    fn next_significant(scanner: &mut Scanner<'_>) -> StatorResult<Token> {
        loop {
            let tok = scanner.next_token()?;
            match tok.kind {
                TokenKind::SingleLineComment | TokenKind::MultiLineComment => continue,
                _ => return Ok(tok),
            }
        }
    }

    /// Return the kind of the current lookahead token.
    fn peek_kind(&self) -> TokenKind {
        self.current.kind
    }

    /// Return the span of the current lookahead token.
    fn current_span(&self) -> Span {
        self.current.span
    }

    /// Advance past the current token, returning it.
    fn bump(&mut self) -> StatorResult<Token> {
        let next = Self::next_significant(&mut self.scanner)?;
        Ok(std::mem::replace(&mut self.current, next))
    }

    /// Consume the current token only if its kind matches `kind`.
    fn eat(&mut self, kind: TokenKind) -> StatorResult<bool> {
        if self.current.kind == kind {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Consume the current token asserting it has the given kind.
    ///
    /// Returns an error when the token does not match.
    fn expect(&mut self, kind: TokenKind) -> StatorResult<Token> {
        if self.current.kind != kind {
            let span = self.current.span;
            return Err(Self::make_error(
                span,
                &format!("expected {kind:?}, got {:?}", self.current.kind),
            ));
        }
        self.bump()
    }

    /// Build a [`StatorError::SyntaxError`] anchored at the current token.
    fn error(&self, msg: &str) -> StatorError {
        Self::make_error(self.current.span, msg)
    }

    /// Build a [`StatorError::SyntaxError`] anchored at the given span.
    fn error_at(span: Span, msg: &str) -> StatorError {
        Self::make_error(span, msg)
    }

    /// Construct a [`StatorError::SyntaxError`] with the canonical position format.
    fn make_error(span: Span, msg: &str) -> StatorError {
        StatorError::SyntaxError(format!(
            "at {}:{} \u{2014} {}",
            span.start.line, span.start.column, msg
        ))
    }

    /// Increment the recursion depth counter, returning an error if the
    /// maximum depth is exceeded.  Call [`Self::leave`] when returning from
    /// the recursive function.
    #[inline]
    fn enter(&mut self) -> StatorResult<()> {
        self.depth += 1;
        if self.depth > MAX_RECURSION_DEPTH {
            Err(self.error("maximum nesting depth exceeded"))
        } else {
            Ok(())
        }
    }

    /// Decrement the recursion depth counter.
    #[inline]
    fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Consume an optional semicolon, inserting one automatically if needed
    /// (ASI rules: end of line, right brace, or end of input).
    fn consume_semicolon(&mut self) -> StatorResult<()> {
        match self.peek_kind() {
            TokenKind::Semicolon => {
                self.bump()?;
            }
            TokenKind::RightBrace | TokenKind::Eof => {}
            _ if self.current.had_line_terminator_before => {}
            _ => {
                return Err(self.error("expected ';' after statement"));
            }
        }
        Ok(())
    }

    // ── Top-level ────────────────────────────────────────────────────────────

    /// Parse a complete source file as a [`Program`].
    ///
    /// Top-level `import` and `export` declarations are emitted as
    /// [`ProgramItem::ModuleDecl`]; everything else is a
    /// [`ProgramItem::Stmt`].  When at least one module declaration is
    /// encountered the source type is set to [`SourceType::Module`].
    fn parse_program(&mut self) -> StatorResult<Program> {
        let start = self.current_span();
        let mut body = Vec::new();
        let mut is_module = false;

        while self.peek_kind() != TokenKind::Eof {
            match self.peek_kind() {
                TokenKind::Import => {
                    is_module = true;
                    let decl = self.parse_import_decl()?;
                    body.push(ProgramItem::ModuleDecl(ModuleDecl::Import(decl)));
                }
                TokenKind::Export => {
                    is_module = true;
                    let decl = self.parse_export_decl()?;
                    body.push(ProgramItem::ModuleDecl(decl));
                }
                _ => {
                    let stmt = self.parse_stmt()?;
                    body.push(ProgramItem::Stmt(stmt));
                }
            }
        }

        let end = self.current_span();
        Ok(Program {
            loc: Self::merge_spans(start, end),
            source_type: if is_module {
                SourceType::Module
            } else {
                SourceType::Script
            },
            body,
        })
    }

    // ── Statements ───────────────────────────────────────────────────────────

    fn parse_stmt(&mut self) -> StatorResult<Stmt> {
        self.enter()?;
        let result = self.parse_stmt_inner();
        self.leave();
        result
    }

    fn parse_stmt_inner(&mut self) -> StatorResult<Stmt> {
        match self.peek_kind() {
            TokenKind::Semicolon => {
                let span = self.current_span();
                self.bump()?;
                Ok(Stmt::Empty(EmptyStmt { loc: span }))
            }
            TokenKind::LeftBrace => self.parse_block().map(Stmt::Block),
            TokenKind::Var => {
                let tok = self.bump()?;
                self.parse_var_decl(VarKind::Var, tok.span)
            }
            TokenKind::Let => {
                let tok = self.bump()?;
                self.parse_var_decl(VarKind::Let, tok.span)
            }
            TokenKind::Const => {
                let tok = self.bump()?;
                self.parse_var_decl(VarKind::Const, tok.span)
            }
            TokenKind::If => self.parse_if(),
            TokenKind::While => self.parse_while(),
            TokenKind::Do => self.parse_do_while(),
            TokenKind::For => self.parse_for(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Break => self.parse_break(),
            TokenKind::Continue => self.parse_continue(),
            TokenKind::Throw => self.parse_throw(),
            TokenKind::Try => self.parse_try(),
            TokenKind::Switch => self.parse_switch(),
            TokenKind::Debugger => {
                let span = self.current_span();
                self.bump()?;
                self.consume_semicolon()?;
                Ok(Stmt::Debugger(DebuggerStmt { loc: span }))
            }
            TokenKind::Function => {
                let tok = self.bump()?;
                self.parse_fn_decl(tok.span, false)
            }
            TokenKind::Async => {
                // `async function …` is an async function declaration.
                // Otherwise fall through to expression statement (covers
                // `async () => …` as an expression statement, etc.).
                let saved_scanner = self.scanner.clone();
                let saved_current = self.current.clone();
                let async_tok = self.bump()?; // consume `async`
                if self.peek_kind() == TokenKind::Function
                    && !self.current.had_line_terminator_before
                {
                    let fn_tok = self.bump()?; // consume `function`
                    return self
                        .parse_fn_decl(Self::merge_spans(async_tok.span, fn_tok.span), true);
                }
                // Not `async function` — restore and parse as expression statement.
                self.scanner = saved_scanner;
                self.current = saved_current;
                self.parse_expr_stmt()
            }
            TokenKind::Class => self.parse_class_decl(),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_block(&mut self) -> StatorResult<BlockStmt> {
        let start = self.current_span();
        self.expect(TokenKind::LeftBrace)?;
        let mut body = Vec::new();
        while self.peek_kind() != TokenKind::RightBrace {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input inside block"));
            }
            body.push(self.parse_stmt()?);
        }
        let end = self.current_span();
        self.bump()?; // consume '}'
        Ok(BlockStmt {
            loc: Self::merge_spans(start, end),
            body,
        })
    }

    fn parse_var_decl(&mut self, kind: VarKind, kw_span: Span) -> StatorResult<Stmt> {
        let mut declarators = Vec::new();
        loop {
            declarators.push(self.parse_var_declarator()?);
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        self.consume_semicolon()?;
        let end = declarators.last().map(|d| d.id.loc()).unwrap_or(kw_span);
        Ok(Stmt::VarDecl(VarDecl {
            loc: Self::merge_spans(kw_span, end),
            kind,
            declarators,
        }))
    }

    fn parse_var_declarator(&mut self) -> StatorResult<VarDeclarator> {
        let start = self.current_span();
        let id = self.parse_binding_pat()?;
        let id_end = id.loc();
        let init = if self.eat(TokenKind::Equal)? {
            Some(Box::new(self.parse_assignment_expr()?))
        } else {
            None
        };
        let end = init.as_ref().map(|e| e.loc()).unwrap_or(id_end);
        Ok(VarDeclarator {
            loc: Self::merge_spans(start, end),
            id,
            init,
        })
    }

    fn parse_expr_stmt(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        let expr = self.parse_expr()?;
        let end = expr.loc();
        self.consume_semicolon()?;
        Ok(Stmt::Expr(ExprStmt {
            loc: Self::merge_spans(start, end),
            expr: Box::new(expr),
        }))
    }

    fn parse_if(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'if'
        self.expect(TokenKind::LeftParen)?;
        let test = self.parse_expr()?;
        self.expect(TokenKind::RightParen)?;
        let consequent = self.parse_stmt()?;
        let alternate = if self.eat(TokenKind::Else)? {
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        let end = alternate
            .as_ref()
            .map(|a| a.loc())
            .unwrap_or_else(|| consequent.loc());
        Ok(Stmt::If(IfStmt {
            loc: Self::merge_spans(start, end),
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate,
        }))
    }

    fn parse_while(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'while'
        self.expect(TokenKind::LeftParen)?;
        let test = self.parse_expr()?;
        self.expect(TokenKind::RightParen)?;
        let body = self.parse_stmt()?;
        let end = body.loc();
        Ok(Stmt::While(WhileStmt {
            loc: Self::merge_spans(start, end),
            test: Box::new(test),
            body: Box::new(body),
        }))
    }

    fn parse_do_while(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'do'
        let body = self.parse_stmt()?;
        self.expect(TokenKind::While)?;
        self.expect(TokenKind::LeftParen)?;
        let test = self.parse_expr()?;
        let end = self.current_span();
        self.expect(TokenKind::RightParen)?;
        self.consume_semicolon()?;
        Ok(Stmt::DoWhile(DoWhileStmt {
            loc: Self::merge_spans(start, end),
            body: Box::new(body),
            test: Box::new(test),
        }))
    }

    fn parse_for(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'for'
        self.expect(TokenKind::LeftParen)?;

        // ── var / let / const  ───────────────────────────────────────────
        if matches!(
            self.peek_kind(),
            TokenKind::Var | TokenKind::Let | TokenKind::Const
        ) {
            let kind = match self.peek_kind() {
                TokenKind::Var => VarKind::Var,
                TokenKind::Let => VarKind::Let,
                TokenKind::Const => VarKind::Const,
                _ => unreachable!(),
            };
            let kw_span = self.current_span();
            self.bump()?; // consume var / let / const
            let declarator = self.parse_var_declarator()?;

            // Check for `in` or `of` → for-in / for-of.
            if self.peek_kind() == TokenKind::In || self.peek_kind() == TokenKind::Of {
                let is_of = self.peek_kind() == TokenKind::Of;
                self.bump()?; // consume `in` / `of`
                let right = self.parse_assignment_expr()?;
                self.expect(TokenKind::RightParen)?;
                let body = self.parse_stmt()?;
                let end = body.loc();

                let var_decl = VarDecl {
                    loc: Self::merge_spans(kw_span, declarator.loc),
                    kind,
                    declarators: vec![declarator],
                };
                let left = ForInOfLeft::VarDecl(var_decl);

                return if is_of {
                    Ok(Stmt::ForOf(ForOfStmt {
                        loc: Self::merge_spans(start, end),
                        is_await: false,
                        left,
                        right: Box::new(right),
                        body: Box::new(body),
                    }))
                } else {
                    Ok(Stmt::ForIn(ForInStmt {
                        loc: Self::merge_spans(start, end),
                        left,
                        right: Box::new(right),
                        body: Box::new(body),
                    }))
                };
            }

            // Otherwise, C-style for with a var init.  There may be more
            // declarators separated by commas.
            let mut declarators = vec![declarator];
            while self.eat(TokenKind::Comma)? {
                declarators.push(self.parse_var_declarator()?);
            }
            let decl_end = declarators.last().map(|d| d.id.loc()).unwrap_or(kw_span);
            let init = Some(ForInit::VarDecl(VarDecl {
                loc: Self::merge_spans(kw_span, decl_end),
                kind,
                declarators,
            }));
            return self.parse_c_style_for_rest(start, init);
        }

        // ── Empty init  (`for (; …)`) ────────────────────────────────────
        if self.peek_kind() == TokenKind::Semicolon {
            return self.parse_c_style_for_rest(start, None);
        }

        // ── Expression init (may still be for-in / for-of) ───────────────
        // Use `no_in` so that `in` is not consumed as a binary operator.
        let saved_no_in = self.no_in;
        self.no_in = true;
        let init_expr = self.parse_expr()?;
        self.no_in = saved_no_in;

        if self.peek_kind() == TokenKind::In || self.peek_kind() == TokenKind::Of {
            let is_of = self.peek_kind() == TokenKind::Of;
            self.bump()?; // consume `in` / `of`
            let right = self.parse_assignment_expr()?;
            self.expect(TokenKind::RightParen)?;
            let body = self.parse_stmt()?;
            let end = body.loc();

            let left = ForInOfLeft::Pat(self.expr_to_for_lhs(init_expr)?);

            return if is_of {
                Ok(Stmt::ForOf(ForOfStmt {
                    loc: Self::merge_spans(start, end),
                    is_await: false,
                    left,
                    right: Box::new(right),
                    body: Box::new(body),
                }))
            } else {
                Ok(Stmt::ForIn(ForInStmt {
                    loc: Self::merge_spans(start, end),
                    left,
                    right: Box::new(right),
                    body: Box::new(body),
                }))
            };
        }

        // Plain C-style for with an expression init.
        let init = Some(ForInit::Expr(Box::new(init_expr)));
        self.parse_c_style_for_rest(start, init)
    }

    /// Finish parsing a C-style `for (init; test; update) body` after the
    /// init clause has already been consumed.  The parser is expected to be
    /// sitting on the first `;`.
    fn parse_c_style_for_rest(&mut self, start: Span, init: Option<ForInit>) -> StatorResult<Stmt> {
        self.expect(TokenKind::Semicolon)?;

        // Test.
        let test = if self.peek_kind() == TokenKind::Semicolon {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(TokenKind::Semicolon)?;

        // Update.
        let update = if self.peek_kind() == TokenKind::RightParen {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(TokenKind::RightParen)?;
        let body = self.parse_stmt()?;
        let end = body.loc();
        Ok(Stmt::For(ForStmt {
            loc: Self::merge_spans(start, end),
            init,
            test,
            update,
            body: Box::new(body),
        }))
    }

    /// Convert an expression to a pattern suitable for the left-hand side of
    /// a `for-in` or `for-of` statement (currently only simple identifiers).
    fn expr_to_for_lhs(&self, expr: Expr) -> StatorResult<Pat> {
        match expr {
            Expr::Ident(id) => Ok(Pat::Ident(id)),
            Expr::Member(m) => {
                // `for (a.b in obj)` — repack as a member-access pattern.
                // For now, just wrap in a simple ident if possible.
                Err(Self::error_at(
                    m.loc,
                    "member expressions as for-in/of targets are not yet supported",
                ))
            }
            other => {
                let loc = other.loc();
                Err(Self::error_at(
                    loc,
                    "invalid left-hand side in for-in/of loop",
                ))
            }
        }
    }

    fn parse_return(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'return'
        let argument = if !self.current.had_line_terminator_before
            && self.peek_kind() != TokenKind::Semicolon
            && self.peek_kind() != TokenKind::RightBrace
            && self.peek_kind() != TokenKind::Eof
        {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let end = argument.as_ref().map(|a| a.loc()).unwrap_or(start);
        self.consume_semicolon()?;
        Ok(Stmt::Return(ReturnStmt {
            loc: Self::merge_spans(start, end),
            argument,
        }))
    }

    fn parse_break(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'break'
        let label = if !self.current.had_line_terminator_before
            && self.peek_kind() == TokenKind::Identifier
        {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };
        let end = label.as_ref().map(|l| l.loc).unwrap_or(start);
        self.consume_semicolon()?;
        Ok(Stmt::Break(BreakStmt {
            loc: Self::merge_spans(start, end),
            label,
        }))
    }

    fn parse_continue(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'continue'
        let label = if !self.current.had_line_terminator_before
            && self.peek_kind() == TokenKind::Identifier
        {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };
        let end = label.as_ref().map(|l| l.loc).unwrap_or(start);
        self.consume_semicolon()?;
        Ok(Stmt::Continue(ContinueStmt {
            loc: Self::merge_spans(start, end),
            label,
        }))
    }

    fn parse_throw(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'throw'
        if self.current.had_line_terminator_before {
            return Err(self.error("illegal newline after throw"));
        }
        let argument = self.parse_expr()?;
        let end = argument.loc();
        self.consume_semicolon()?;
        Ok(Stmt::Throw(ThrowStmt {
            loc: Self::merge_spans(start, end),
            argument: Box::new(argument),
        }))
    }

    fn parse_try(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'try'
        let block = self.parse_block()?;
        let handler = if self.peek_kind() == TokenKind::Catch {
            let catch_span = self.current_span();
            self.bump()?;
            let param = if self.eat(TokenKind::LeftParen)? {
                let pat = self.parse_binding_pat()?;
                self.expect(TokenKind::RightParen)?;
                Some(pat)
            } else {
                None
            };
            let body = self.parse_block()?;
            Some(crate::parser::ast::CatchClause {
                loc: Self::merge_spans(catch_span, body.loc),
                param,
                body,
            })
        } else {
            None
        };
        let finalizer = if self.eat(TokenKind::Finally)? {
            Some(self.parse_block()?)
        } else {
            None
        };
        if handler.is_none() && finalizer.is_none() {
            return Err(self.error("try statement must have a catch or finally clause"));
        }
        let end = finalizer
            .as_ref()
            .map(|f| f.loc)
            .or_else(|| handler.as_ref().map(|h| h.loc))
            .unwrap_or(block.loc);
        Ok(Stmt::Try(TryStmt {
            loc: Self::merge_spans(start, end),
            block,
            handler,
            finalizer,
        }))
    }

    fn parse_switch(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // 'switch'
        self.expect(TokenKind::LeftParen)?;
        let discriminant = self.parse_expr()?;
        self.expect(TokenKind::RightParen)?;
        self.expect(TokenKind::LeftBrace)?;

        let mut cases = Vec::new();
        while self.peek_kind() != TokenKind::RightBrace {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in switch statement"));
            }
            let case_start = self.current_span();
            let test = if self.eat(TokenKind::Case)? {
                Some(self.parse_expr()?)
            } else if self.eat(TokenKind::Default)? {
                None
            } else {
                return Err(self.error("expected 'case' or 'default'"));
            };
            self.expect(TokenKind::Colon)?;

            let mut consequent = Vec::new();
            while self.peek_kind() != TokenKind::Case
                && self.peek_kind() != TokenKind::Default
                && self.peek_kind() != TokenKind::RightBrace
            {
                if self.peek_kind() == TokenKind::Eof {
                    return Err(self.error("unexpected end of input in switch clause"));
                }
                consequent.push(self.parse_stmt()?);
            }

            let case_end = if let Some(last) = consequent.last() {
                last.loc()
            } else {
                case_start
            };
            cases.push(SwitchCase {
                loc: Self::merge_spans(case_start, case_end),
                test,
                consequent,
            });
        }

        let end = self.current_span();
        self.bump()?; // consume '}'
        Ok(Stmt::Switch(SwitchStmt {
            loc: Self::merge_spans(start, end),
            discriminant: Box::new(discriminant),
            cases,
        }))
    }

    fn parse_fn_decl(&mut self, fn_span: Span, is_async: bool) -> StatorResult<Stmt> {
        // [async] function [*] id ( params ) { body }
        let is_generator = self.eat(TokenKind::Star)?;
        let id = if self.peek_kind() == TokenKind::Identifier {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };
        self.expect(TokenKind::LeftParen)?;
        let params = self.parse_formal_params()?;
        let body = self.parse_block()?;
        let end = body.loc;
        Ok(Stmt::FnDecl(Box::new(FnDecl {
            loc: Self::merge_spans(fn_span, end),
            id,
            is_async,
            is_generator,
            params,
            body,
        })))
    }

    // ── Class parsing ──────────────────────────────────────────────────────

    fn parse_class_decl(&mut self) -> StatorResult<Stmt> {
        let start = self.current_span();
        self.bump()?; // consume 'class'

        let id = if self.peek_kind() == TokenKind::Identifier {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };

        let super_class = if self.eat(TokenKind::Extends)? {
            Some(Box::new(self.parse_assignment_expr()?))
        } else {
            None
        };

        let body = self.parse_class_body()?;
        let end = body.loc;
        Ok(Stmt::ClassDecl(Box::new(ClassDecl {
            loc: Self::merge_spans(start, end),
            id,
            super_class,
            body,
        })))
    }

    fn parse_class_expr(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        self.bump()?; // consume 'class'

        let id = if self.peek_kind() == TokenKind::Identifier {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };

        let super_class = if self.eat(TokenKind::Extends)? {
            Some(Box::new(self.parse_assignment_expr()?))
        } else {
            None
        };

        let body = self.parse_class_body()?;
        let end = body.loc;
        Ok(Expr::Class(Box::new(ClassExpr {
            loc: Self::merge_spans(start, end),
            id,
            super_class,
            body,
        })))
    }

    fn parse_class_body(&mut self) -> StatorResult<ClassBody> {
        let start = self.current_span();
        self.expect(TokenKind::LeftBrace)?;
        let mut members = Vec::new();

        while self.peek_kind() != TokenKind::RightBrace {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input inside class body"));
            }
            // Skip empty class elements (`;`).
            if self.peek_kind() == TokenKind::Semicolon {
                self.bump()?;
                continue;
            }

            let member_start = self.current_span();
            let is_static = self.eat(TokenKind::Static)?;

            // Check for `async` modifier on the method.
            let is_async = if self.peek_kind() == TokenKind::Async {
                // `async` is only a modifier when the next token after it is
                // NOT `(` (which would mean a method named "async") and there
                // is no line terminator between `async` and the method name.
                let saved_scanner = self.scanner.clone();
                let saved_current = self.current.clone();
                self.bump()?; // tentatively consume `async`
                if self.peek_kind() == TokenKind::LeftParen {
                    // Method literally named "async".
                    self.scanner = saved_scanner;
                    self.current = saved_current;
                    false
                } else if self.current.had_line_terminator_before {
                    self.scanner = saved_scanner;
                    self.current = saved_current;
                    false
                } else {
                    true
                }
            } else {
                false
            };

            // Determine method kind: get, set, or regular method.
            let (kind, key, is_computed) =
                if self.peek_kind() == TokenKind::Get || self.peek_kind() == TokenKind::Set {
                    let accessor_kind = if self.peek_kind() == TokenKind::Get {
                        MethodKind::Get
                    } else {
                        MethodKind::Set
                    };
                    let accessor_tok = self.bump()?;
                    // If the next token is `(`, then `get`/`set` is the
                    // method *name* (a regular method called "get"/"set").
                    if self.peek_kind() == TokenKind::LeftParen {
                        let name = format!("{:?}", accessor_tok.kind).to_lowercase();
                        (
                            MethodKind::Method,
                            PropKey::Ident(Ident {
                                loc: accessor_tok.span,
                                name,
                            }),
                            false,
                        )
                    } else {
                        let (key, is_computed) = self.parse_class_element_name()?;
                        (accessor_kind, key, is_computed)
                    }
                } else {
                    let (key, is_computed) = self.parse_class_element_name()?;
                    // Check for constructor.
                    let kind = if !is_static && !is_computed {
                        if let PropKey::Ident(ref id) = key {
                            if id.name == "constructor" {
                                MethodKind::Constructor
                            } else {
                                MethodKind::Method
                            }
                        } else {
                            MethodKind::Method
                        }
                    } else {
                        MethodKind::Method
                    };
                    (kind, key, is_computed)
                };

            // Parse (params) { body } — the method value.
            let fn_start = self.current_span();
            self.expect(TokenKind::LeftParen)?;
            let params = self.parse_formal_params()?;
            let body = self.parse_block()?;
            let fn_end = body.loc;

            let value = FnExpr {
                loc: Self::merge_spans(fn_start, fn_end),
                id: None,
                is_async,
                is_generator: false,
                params,
                body,
            };

            members.push(ClassMember::Method(MethodDef {
                loc: Self::merge_spans(member_start, fn_end),
                is_static,
                kind,
                key,
                is_computed,
                value,
            }));
        }

        let end = self.current_span();
        self.bump()?; // consume '}'
        Ok(ClassBody {
            loc: Self::merge_spans(start, end),
            body: members,
        })
    }

    /// Parse a class element name (property key).
    fn parse_class_element_name(&mut self) -> StatorResult<(PropKey, bool)> {
        match self.peek_kind() {
            TokenKind::LeftBracket => {
                self.bump()?; // consume '['
                let key_expr = self.parse_assignment_expr()?;
                self.expect(TokenKind::RightBracket)?;
                Ok((PropKey::Computed(Box::new(key_expr)), true))
            }
            TokenKind::StringLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => return Err(Self::error_at(tok.span, "invalid string token")),
                };
                Ok((
                    PropKey::Str(StringLit {
                        loc: tok.span,
                        value,
                    }),
                    false,
                ))
            }
            TokenKind::NumericLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Number(n) => n,
                    _ => return Err(Self::error_at(tok.span, "invalid numeric token")),
                };
                Ok((
                    PropKey::Num(NumLit {
                        loc: tok.span,
                        value,
                        raw: String::new(),
                    }),
                    false,
                ))
            }
            _ => {
                // Identifier or keyword used as method name.
                let tok = self.bump()?;
                let name = match &tok.value {
                    TokenValue::Str(s) => s.clone(),
                    TokenValue::None => format!("{:?}", tok.kind).to_lowercase(),
                    _ => return Err(Self::error_at(tok.span, "expected method name")),
                };
                Ok((
                    PropKey::Ident(Ident {
                        loc: tok.span,
                        name,
                    }),
                    false,
                ))
            }
        }
    }

    // ── Module declarations (import / export) ─────────────────────────────

    /// Parse `import … ;` and return an [`ImportDecl`].
    fn parse_import_decl(&mut self) -> StatorResult<ImportDecl> {
        let start = self.current_span();
        self.bump()?; // consume 'import'

        // Side-effect import: `import "module";`
        if self.peek_kind() == TokenKind::StringLiteral {
            let source = self.parse_string_lit_token()?;
            let end = source.loc;
            self.consume_semicolon()?;
            return Ok(ImportDecl {
                loc: Self::merge_spans(start, end),
                specifiers: vec![],
                source,
                attributes: vec![],
            });
        }

        let mut specifiers = Vec::new();

        if self.peek_kind() == TokenKind::Star {
            // Namespace import: `import * as ns from "module"`
            self.bump()?; // consume '*'
            self.expect(TokenKind::As)?;
            let local_tok = self.expect(TokenKind::Identifier)?;
            let local = self.ident_from_token(&local_tok)?;
            specifiers.push(ImportSpecifier::Namespace(ImportNamespaceSpecifier {
                loc: Self::merge_spans(start, local.loc),
                local,
            }));
        } else if self.peek_kind() == TokenKind::LeftBrace {
            // Named imports: `import { a, b as c } from "module"`
            self.parse_import_named_specifiers(&mut specifiers)?;
        } else if self.peek_kind() == TokenKind::Identifier {
            // Default import: `import x from "module"`
            let local_tok = self.bump()?;
            let local = self.ident_from_token(&local_tok)?;
            specifiers.push(ImportSpecifier::Default(ImportDefaultSpecifier {
                loc: local.loc,
                local,
            }));

            // Combined form: `import x, { a } from "m"` or `import x, * as ns from "m"`
            if self.eat(TokenKind::Comma)? {
                if self.peek_kind() == TokenKind::Star {
                    self.bump()?; // consume '*'
                    self.expect(TokenKind::As)?;
                    let ns_tok = self.expect(TokenKind::Identifier)?;
                    let ns_local = self.ident_from_token(&ns_tok)?;
                    specifiers.push(ImportSpecifier::Namespace(ImportNamespaceSpecifier {
                        loc: Self::merge_spans(start, ns_local.loc),
                        local: ns_local,
                    }));
                } else if self.peek_kind() == TokenKind::LeftBrace {
                    self.parse_import_named_specifiers(&mut specifiers)?;
                } else {
                    return Err(self.error("expected '{' or '*' after ',' in import"));
                }
            }
        } else {
            return Err(self.error("unexpected token after 'import'"));
        }

        // `from "source"`
        self.expect(TokenKind::From)?;
        let source = self.parse_string_lit_token()?;
        let end = source.loc;
        self.consume_semicolon()?;

        Ok(ImportDecl {
            loc: Self::merge_spans(start, end),
            specifiers,
            source,
            attributes: vec![],
        })
    }

    /// Parse the `{ a, b as c }` portion of an import and push specifiers.
    fn parse_import_named_specifiers(
        &mut self,
        specifiers: &mut Vec<ImportSpecifier>,
    ) -> StatorResult<()> {
        self.expect(TokenKind::LeftBrace)?;
        while self.peek_kind() != TokenKind::RightBrace {
            let imported_tok = self.bump()?;
            let imported = self.module_export_name_from_token(&imported_tok)?;
            let local = if self.eat(TokenKind::As)? {
                let local_tok = self.expect(TokenKind::Identifier)?;
                self.ident_from_token(&local_tok)?
            } else {
                match &imported {
                    ModuleExportName::Ident(id) => id.clone(),
                    ModuleExportName::Str(s) => {
                        return Err(Self::error_at(
                            s.loc,
                            "string import name requires 'as' alias",
                        ));
                    }
                }
            };
            specifiers.push(ImportSpecifier::Named(ImportNamedSpecifier {
                loc: Self::merge_spans(imported_tok.span, local.loc),
                imported,
                local,
            }));
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        self.expect(TokenKind::RightBrace)?;
        Ok(())
    }

    /// Parse an `export` declaration and return the appropriate [`ModuleDecl`].
    fn parse_export_decl(&mut self) -> StatorResult<ModuleDecl> {
        let start = self.current_span();
        self.bump()?; // consume 'export'

        match self.peek_kind() {
            // ── export default … ────────────────────────────────────────
            TokenKind::Default => {
                self.bump()?; // consume 'default'
                match self.peek_kind() {
                    TokenKind::Async if !self.current.had_line_terminator_before => {
                        // Speculatively check for `async function`.
                        let saved_scanner = self.scanner.clone();
                        let saved_current = self.current.clone();
                        let async_tok = self.bump()?;
                        if self.peek_kind() == TokenKind::Function
                            && !self.current.had_line_terminator_before
                        {
                            let fn_tok = self.bump()?;
                            let fn_stmt = self.parse_fn_decl(
                                Self::merge_spans(async_tok.span, fn_tok.span),
                                true,
                            )?;
                            let end = fn_stmt.loc();
                            if let Stmt::FnDecl(fd) = fn_stmt {
                                return Ok(ModuleDecl::ExportDefault(ExportDefaultDecl {
                                    loc: Self::merge_spans(start, end),
                                    declaration: ExportDefaultExpr::Fn(fd),
                                }));
                            }
                            unreachable!()
                        }
                        // Not `async function` — restore and fall through.
                        self.scanner = saved_scanner;
                        self.current = saved_current;
                        let expr = self.parse_assignment_expr()?;
                        let end = expr.loc();
                        self.consume_semicolon()?;
                        Ok(ModuleDecl::ExportDefault(ExportDefaultDecl {
                            loc: Self::merge_spans(start, end),
                            declaration: ExportDefaultExpr::Expr(Box::new(expr)),
                        }))
                    }
                    TokenKind::Function => {
                        let fn_tok = self.bump()?;
                        let fn_stmt = self.parse_fn_decl(fn_tok.span, false)?;
                        let end = fn_stmt.loc();
                        if let Stmt::FnDecl(fd) = fn_stmt {
                            Ok(ModuleDecl::ExportDefault(ExportDefaultDecl {
                                loc: Self::merge_spans(start, end),
                                declaration: ExportDefaultExpr::Fn(fd),
                            }))
                        } else {
                            unreachable!()
                        }
                    }
                    TokenKind::Class => {
                        let class_stmt = self.parse_class_decl()?;
                        let end = class_stmt.loc();
                        if let Stmt::ClassDecl(cd) = class_stmt {
                            Ok(ModuleDecl::ExportDefault(ExportDefaultDecl {
                                loc: Self::merge_spans(start, end),
                                declaration: ExportDefaultExpr::Class(cd),
                            }))
                        } else {
                            unreachable!()
                        }
                    }
                    _ => {
                        let expr = self.parse_assignment_expr()?;
                        let end = expr.loc();
                        self.consume_semicolon()?;
                        Ok(ModuleDecl::ExportDefault(ExportDefaultDecl {
                            loc: Self::merge_spans(start, end),
                            declaration: ExportDefaultExpr::Expr(Box::new(expr)),
                        }))
                    }
                }
            }

            // ── export async function … ────────────────────────────────
            TokenKind::Async => {
                let async_tok = self.bump()?; // consume `async`
                if self.peek_kind() == TokenKind::Function
                    && !self.current.had_line_terminator_before
                {
                    let fn_tok = self.bump()?;
                    let decl =
                        self.parse_fn_decl(Self::merge_spans(async_tok.span, fn_tok.span), true)?;
                    let end = decl.loc();
                    Ok(ModuleDecl::ExportNamed(ExportNamedDecl {
                        loc: Self::merge_spans(start, end),
                        specifiers: vec![],
                        source: None,
                        declaration: Some(Box::new(decl)),
                        attributes: vec![],
                    }))
                } else {
                    Err(self.error("expected 'function' after 'async' in export"))
                }
            }

            // ── export function … ──────────────────────────────────────
            TokenKind::Function => {
                let fn_tok = self.bump()?;
                let decl = self.parse_fn_decl(fn_tok.span, false)?;
                let end = decl.loc();
                Ok(ModuleDecl::ExportNamed(ExportNamedDecl {
                    loc: Self::merge_spans(start, end),
                    specifiers: vec![],
                    source: None,
                    declaration: Some(Box::new(decl)),
                    attributes: vec![],
                }))
            }

            // ── export class … ─────────────────────────────────────────
            TokenKind::Class => {
                let decl = self.parse_class_decl()?;
                let end = decl.loc();
                Ok(ModuleDecl::ExportNamed(ExportNamedDecl {
                    loc: Self::merge_spans(start, end),
                    specifiers: vec![],
                    source: None,
                    declaration: Some(Box::new(decl)),
                    attributes: vec![],
                }))
            }

            // ── export var / let / const … ─────────────────────────────
            TokenKind::Var | TokenKind::Let | TokenKind::Const => {
                let tok = self.bump()?;
                let kind = match tok.kind {
                    TokenKind::Var => VarKind::Var,
                    TokenKind::Let => VarKind::Let,
                    TokenKind::Const => VarKind::Const,
                    _ => unreachable!(),
                };
                let decl = self.parse_var_decl(kind, tok.span)?;
                let end = decl.loc();
                Ok(ModuleDecl::ExportNamed(ExportNamedDecl {
                    loc: Self::merge_spans(start, end),
                    specifiers: vec![],
                    source: None,
                    declaration: Some(Box::new(decl)),
                    attributes: vec![],
                }))
            }

            // ── export { … } [from "…"] ───────────────────────────────
            TokenKind::LeftBrace => {
                self.bump()?; // consume '{'
                let mut specifiers = Vec::new();
                while self.peek_kind() != TokenKind::RightBrace {
                    let spec_start = self.current_span();
                    let local_tok = self.bump()?;
                    let local = self.module_export_name_from_token(&local_tok)?;
                    let (exported, spec_end) = if self.eat(TokenKind::As)? {
                        let exp_tok = self.bump()?;
                        let exp = self.module_export_name_from_token(&exp_tok)?;
                        (exp, exp_tok.span)
                    } else {
                        (local.clone(), local_tok.span)
                    };
                    specifiers.push(ExportSpecifier {
                        loc: Self::merge_spans(spec_start, spec_end),
                        local,
                        exported,
                    });
                    if !self.eat(TokenKind::Comma)? {
                        break;
                    }
                }
                let rbrace = self.expect(TokenKind::RightBrace)?;

                // Optional re-export source.
                let source = if self.peek_kind() == TokenKind::From {
                    self.bump()?;
                    Some(self.parse_string_lit_token()?)
                } else {
                    None
                };

                let end = source.as_ref().map(|s| s.loc).unwrap_or(rbrace.span);
                self.consume_semicolon()?;

                Ok(ModuleDecl::ExportNamed(ExportNamedDecl {
                    loc: Self::merge_spans(start, end),
                    specifiers,
                    source,
                    declaration: None,
                    attributes: vec![],
                }))
            }

            // ── export * [as name] from "…" ────────────────────────────
            TokenKind::Star => {
                self.bump()?; // consume '*'
                let exported = if self.eat(TokenKind::As)? {
                    let tok = self.bump()?;
                    Some(self.module_export_name_from_token(&tok)?)
                } else {
                    None
                };
                self.expect(TokenKind::From)?;
                let source = self.parse_string_lit_token()?;
                let end = source.loc;
                self.consume_semicolon()?;

                Ok(ModuleDecl::ExportAll(ExportAllDecl {
                    loc: Self::merge_spans(start, end),
                    exported,
                    source,
                    attributes: vec![],
                }))
            }

            _ => Err(self.error("unexpected token after 'export'")),
        }
    }

    /// Parse a string literal token into a [`StringLit`].
    fn parse_string_lit_token(&mut self) -> StatorResult<StringLit> {
        let tok = self.expect(TokenKind::StringLiteral)?;
        let value = match &tok.value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(Self::error_at(tok.span, "expected string literal")),
        };
        Ok(StringLit {
            loc: tok.span,
            value,
        })
    }

    /// Convert a token into a [`ModuleExportName`].
    ///
    /// Accepts identifiers, string literals, and keywords (which are valid as
    /// import/export specifier names, e.g. `default`).
    fn module_export_name_from_token(&self, tok: &Token) -> StatorResult<ModuleExportName> {
        match tok.kind {
            TokenKind::Identifier => {
                let ident = self.ident_from_token(tok)?;
                Ok(ModuleExportName::Ident(ident))
            }
            TokenKind::StringLiteral => {
                let value = match &tok.value {
                    TokenValue::Str(s) => s.clone(),
                    _ => return Err(Self::error_at(tok.span, "expected string literal")),
                };
                Ok(ModuleExportName::Str(StringLit {
                    loc: tok.span,
                    value,
                }))
            }
            _ => {
                // Keywords (e.g. `default`, `from`) are valid as export/import names.
                let name = self.name_from_token(tok)?;
                Ok(ModuleExportName::Ident(Ident {
                    loc: tok.span,
                    name,
                }))
            }
        }
    }

    fn parse_formal_params(&mut self) -> StatorResult<Vec<Param>> {
        let mut params = Vec::new();
        while self.peek_kind() != TokenKind::RightParen {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in parameter list"));
            }
            let start = self.current_span();

            // Rest parameter: `...pat`
            if self.peek_kind() == TokenKind::DotDotDot {
                let rest_start = self.current_span();
                self.bump()?; // consume `...`
                let arg = self.parse_binding_pat()?;
                let rest_end = arg.loc();
                let rest_pat = Pat::Rest(Box::new(RestElement {
                    loc: Self::merge_spans(rest_start, rest_end),
                    argument: Box::new(arg),
                }));
                params.push(Param {
                    loc: Self::merge_spans(start, rest_end),
                    pat: rest_pat,
                    default: None,
                });
                // Rest must be last — eat optional trailing comma then break.
                let _ = self.eat(TokenKind::Comma)?;
                break;
            }

            let pat = self.parse_binding_pat()?;
            let pat_end = pat.loc();

            // Default value: `pat = expr`
            let default = if self.eat(TokenKind::Equal)? {
                Some(self.parse_assignment_expr()?)
            } else {
                None
            };
            let end = default.as_ref().map(|e| e.loc()).unwrap_or(pat_end);
            params.push(Param {
                loc: Self::merge_spans(start, end),
                pat,
                default,
            });
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        self.expect(TokenKind::RightParen)?;
        Ok(params)
    }

    fn parse_binding_pat(&mut self) -> StatorResult<Pat> {
        match self.peek_kind() {
            TokenKind::Identifier => {
                let tok = self.bump()?;
                let ident = self.ident_from_token(&tok)?;
                Ok(Pat::Ident(ident))
            }
            TokenKind::LeftBracket => self.parse_array_pat(),
            TokenKind::LeftBrace => self.parse_object_pat(),
            _ => Err(self.error("expected binding pattern")),
        }
    }

    /// Parse an array destructuring pattern: `[a, , b, ...rest]`.
    fn parse_array_pat(&mut self) -> StatorResult<Pat> {
        let start = self.current_span();
        self.bump()?; // consume `[`
        let mut elements: Vec<Option<Pat>> = Vec::new();

        while self.peek_kind() != TokenKind::RightBracket {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in array pattern"));
            }

            // Elision: `,` without an element.
            if self.peek_kind() == TokenKind::Comma {
                elements.push(None);
                self.bump()?; // consume `,`
                continue;
            }

            // Rest element: `...pat`
            if self.peek_kind() == TokenKind::DotDotDot {
                let rest_start = self.current_span();
                self.bump()?; // consume `...`
                let arg = self.parse_binding_pat()?;
                let rest_end = arg.loc();
                elements.push(Some(Pat::Rest(Box::new(RestElement {
                    loc: Self::merge_spans(rest_start, rest_end),
                    argument: Box::new(arg),
                }))));
                // Rest must be last element.
                let _ = self.eat(TokenKind::Comma)?;
                break;
            }

            // Normal element: pattern with optional default.
            let elem = self.parse_binding_pat()?;
            let elem = if self.peek_kind() == TokenKind::Equal {
                let eq_tok = self.bump()?; // consume `=`
                let _ = eq_tok;
                let right = self.parse_assignment_expr()?;
                let loc = Self::merge_spans(elem.loc(), right.loc());
                Pat::Assign(Box::new(AssignPat {
                    loc,
                    left: Box::new(elem),
                    right: Box::new(right),
                }))
            } else {
                elem
            };

            elements.push(Some(elem));

            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }

        let end = self.expect(TokenKind::RightBracket)?;
        Ok(Pat::Array(Box::new(ArrayPat {
            loc: Self::merge_spans(start, end.span),
            elements,
        })))
    }

    /// Parse an object destructuring pattern: `{ a, b: c, ...rest }`.
    fn parse_object_pat(&mut self) -> StatorResult<Pat> {
        let start = self.current_span();
        self.bump()?; // consume `{`
        let mut properties: Vec<ObjectPatProp> = Vec::new();

        while self.peek_kind() != TokenKind::RightBrace {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in object pattern"));
            }

            // Rest element: `...pat`
            if self.peek_kind() == TokenKind::DotDotDot {
                let rest_start = self.current_span();
                self.bump()?; // consume `...`
                let arg = self.parse_binding_pat()?;
                let rest_end = arg.loc();
                properties.push(ObjectPatProp::Rest(RestElement {
                    loc: Self::merge_spans(rest_start, rest_end),
                    argument: Box::new(arg),
                }));
                // Rest must be last.
                let _ = self.eat(TokenKind::Comma)?;
                break;
            }

            // Parse property key (reuse class-element-name logic for computed,
            // string, numeric, and identifier keys).
            let prop_start = self.current_span();
            let (key, is_computed) = self.parse_pat_property_key()?;

            // `key: value_pattern` — key-value property.
            if self.eat(TokenKind::Colon)? {
                let value = self.parse_binding_pat()?;
                // Optional default on value pattern.
                let value = if self.peek_kind() == TokenKind::Equal {
                    self.bump()?; // consume `=`
                    let right = self.parse_assignment_expr()?;
                    let loc = Self::merge_spans(value.loc(), right.loc());
                    Pat::Assign(Box::new(AssignPat {
                        loc,
                        left: Box::new(value),
                        right: Box::new(right),
                    }))
                } else {
                    value
                };
                let prop_end = value.loc();
                properties.push(ObjectPatProp::KeyValue(KeyValuePatProp {
                    loc: Self::merge_spans(prop_start, prop_end),
                    key,
                    is_computed,
                    value,
                }));
            } else {
                // Shorthand: `{ id }` or `{ id = default }`.
                // Only valid when key is an identifier.
                let ident = match key {
                    PropKey::Ident(id) => id,
                    _ => {
                        return Err(Self::error_at(
                            prop_start,
                            "expected ':' after property name in destructuring pattern",
                        ));
                    }
                };
                let default = if self.eat(TokenKind::Equal)? {
                    Some(Box::new(self.parse_assignment_expr()?))
                } else {
                    None
                };
                let prop_end = default.as_ref().map(|e| e.loc()).unwrap_or(ident.loc);
                properties.push(ObjectPatProp::Assign(AssignPatProp {
                    loc: Self::merge_spans(prop_start, prop_end),
                    key: ident,
                    value: default,
                }));
            }

            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }

        let end = self.expect(TokenKind::RightBrace)?;
        Ok(Pat::Object(Box::new(ObjectPat {
            loc: Self::merge_spans(start, end.span),
            properties,
        })))
    }

    /// Parse a property key for use in object destructuring patterns.
    fn parse_pat_property_key(&mut self) -> StatorResult<(PropKey, bool)> {
        match self.peek_kind() {
            TokenKind::LeftBracket => {
                self.bump()?; // consume `[`
                let key_expr = self.parse_assignment_expr()?;
                self.expect(TokenKind::RightBracket)?;
                Ok((PropKey::Computed(Box::new(key_expr)), true))
            }
            TokenKind::StringLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => return Err(Self::error_at(tok.span, "invalid string token")),
                };
                Ok((
                    PropKey::Str(StringLit {
                        loc: tok.span,
                        value,
                    }),
                    false,
                ))
            }
            TokenKind::NumericLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Number(n) => n,
                    _ => return Err(Self::error_at(tok.span, "invalid numeric token")),
                };
                Ok((
                    PropKey::Num(NumLit {
                        loc: tok.span,
                        value,
                        raw: String::new(),
                    }),
                    false,
                ))
            }
            _ => {
                let tok = self.bump()?;
                let name = match &tok.value {
                    TokenValue::Str(s) => s.clone(),
                    TokenValue::None => format!("{:?}", tok.kind).to_lowercase(),
                    _ => return Err(Self::error_at(tok.span, "expected property name")),
                };
                Ok((
                    PropKey::Ident(Ident {
                        loc: tok.span,
                        name,
                    }),
                    false,
                ))
            }
        }
    }

    // ── Expressions ──────────────────────────────────────────────────────────

    /// Parse a comma-separated sequence expression.
    fn parse_expr(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let first = self.parse_assignment_expr()?;
        if self.peek_kind() == TokenKind::Comma {
            let mut exprs = vec![first];
            while self.eat(TokenKind::Comma)? {
                exprs.push(self.parse_assignment_expr()?);
            }
            let end = exprs.last().map(|e| e.loc()).unwrap_or(start);
            return Ok(Expr::Sequence(Box::new(SequenceExpr {
                loc: Self::merge_spans(start, end),
                expressions: exprs,
            })));
        }
        Ok(first)
    }

    /// Parse an assignment expression (right-associative).
    ///
    /// Also handles arrow function expressions, which have assignment-level
    /// precedence: `x => x + 1`, `(x, y) => x + y`, `() => 42`.
    fn parse_assignment_expr(&mut self) -> StatorResult<Expr> {
        self.enter()?;
        let result = self.parse_assignment_expr_inner();
        self.leave();
        result
    }

    fn parse_assignment_expr_inner(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();

        // ── Async arrow function detection ───────────────────────────────
        // `async () => …`, `async (x, y) => …`, `async x => …`
        if self.peek_kind() == TokenKind::Async {
            let saved_scanner = self.scanner.clone();
            let saved_current = self.current.clone();
            let _async_tok = self.bump()?; // consume `async`

            // There must be no line terminator between `async` and the params.
            if !self.current.had_line_terminator_before {
                // `async () => body` — empty parameter list.
                if self.peek_kind() == TokenKind::LeftParen {
                    let paren_saved_scanner = self.scanner.clone();
                    let paren_saved_current = self.current.clone();
                    self.bump()?; // consume `(`
                    if self.peek_kind() == TokenKind::RightParen {
                        self.bump()?; // consume `)`
                        if self.peek_kind() == TokenKind::Arrow {
                            self.bump()?; // consume `=>`
                            let body = self.parse_arrow_body()?;
                            let end = self.arrow_body_loc(&body);
                            return Ok(Expr::Arrow(Box::new(ArrowExpr {
                                loc: Self::merge_spans(start, end),
                                is_async: true,
                                params: vec![],
                                body,
                            })));
                        }
                        // `async()` not followed by `=>` — restore to just
                        // after `async` so it parses as a call `async()`.
                        self.scanner = paren_saved_scanner;
                        self.current = paren_saved_current;
                    } else {
                        // `async(…` — restore to just after `async`.
                        self.scanner = paren_saved_scanner;
                        self.current = paren_saved_current;
                    }
                }

                // `async x => body` — single param without parens.
                if self.peek_kind() == TokenKind::Identifier {
                    let id_saved_scanner = self.scanner.clone();
                    let id_saved_current = self.current.clone();
                    let id_tok = self.bump()?; // consume identifier
                    if self.peek_kind() == TokenKind::Arrow {
                        self.bump()?; // consume `=>`
                        let ident = self.ident_from_token(&id_tok)?;
                        let params = vec![Param {
                            loc: ident.loc,
                            pat: Pat::Ident(ident),
                            default: None,
                        }];
                        let body = self.parse_arrow_body()?;
                        let end = self.arrow_body_loc(&body);
                        return Ok(Expr::Arrow(Box::new(ArrowExpr {
                            loc: Self::merge_spans(start, end),
                            is_async: true,
                            params,
                            body,
                        })));
                    }
                    // Not `async x =>` — restore to just after `async`.
                    self.scanner = id_saved_scanner;
                    self.current = id_saved_current;
                }
            }

            // Not an async arrow — restore completely so the expression
            // parser can handle `async` as an identifier or `async function`.
            self.scanner = saved_scanner;
            self.current = saved_current;
        }

        // ── Arrow function detection ─────────────────────────────────────
        // Save parser state so we can speculatively parse parenthesised
        // content and fall back if `=>` does not follow.
        let saved_scanner = self.scanner.clone();
        let saved_current = self.current.clone();

        // Case 1: `() => body` — empty parameter list.
        if self.peek_kind() == TokenKind::LeftParen {
            self.bump()?; // consume `(`
            if self.peek_kind() == TokenKind::RightParen {
                self.bump()?; // consume `)`
                if self.peek_kind() == TokenKind::Arrow {
                    self.bump()?; // consume `=>`
                    let body = self.parse_arrow_body()?;
                    let end = self.arrow_body_loc(&body);
                    return Ok(Expr::Arrow(Box::new(ArrowExpr {
                        loc: Self::merge_spans(start, end),
                        is_async: false,
                        params: vec![],
                        body,
                    })));
                }
                // `()` not followed by `=>` — restore and let normal
                // parsing produce an appropriate error.
                self.scanner = saved_scanner.clone();
                self.current = saved_current.clone();
            } else {
                // Not `()`, restore for the general path below.
                self.scanner = saved_scanner.clone();
                self.current = saved_current.clone();
            }
        }

        // General path: parse LHS as a conditional expression.  This covers
        // both plain expressions *and* parenthesised arrow-param lists
        // (via the "cover grammar" — expressions reinterpreted as params).
        let lhs = self.parse_conditional_expr()?;

        // If `=>` follows, the LHS is actually the parameter list of an
        // arrow function.
        if self.peek_kind() == TokenKind::Arrow {
            self.bump()?; // consume `=>`

            // Check if the LHS is `async(…)` — meaning an async arrow with
            // parenthesised params.  In the cover grammar this was parsed as
            // `Call(Ident("async"), args)`.
            if let Expr::Call(ref call) = lhs
                && let Expr::Ident(ref id) = *call.callee
                && id.name == "async"
            {
                let params: Vec<Param> = call
                    .arguments
                    .clone()
                    .into_iter()
                    .map(|e| self.expr_to_single_param(e))
                    .collect::<StatorResult<Vec<_>>>()?;
                let body = self.parse_arrow_body()?;
                let end = self.arrow_body_loc(&body);
                return Ok(Expr::Arrow(Box::new(ArrowExpr {
                    loc: Self::merge_spans(start, end),
                    is_async: true,
                    params,
                    body,
                })));
            }

            let params = self.expr_to_arrow_params(lhs)?;
            let body = self.parse_arrow_body()?;
            let end = self.arrow_body_loc(&body);
            return Ok(Expr::Arrow(Box::new(ArrowExpr {
                loc: Self::merge_spans(start, end),
                is_async: false,
                params,
                body,
            })));
        }

        let assign_op = match self.peek_kind() {
            TokenKind::Equal => Some(AssignOp::Assign),
            TokenKind::PlusEqual => Some(AssignOp::AddAssign),
            TokenKind::MinusEqual => Some(AssignOp::SubAssign),
            TokenKind::StarEqual => Some(AssignOp::MulAssign),
            TokenKind::StarStarEqual => Some(AssignOp::ExpAssign),
            TokenKind::SlashEqual => Some(AssignOp::DivAssign),
            TokenKind::PercentEqual => Some(AssignOp::RemAssign),
            TokenKind::LessLessEqual => Some(AssignOp::ShlAssign),
            TokenKind::GreaterGreaterEqual => Some(AssignOp::ShrAssign),
            TokenKind::GreaterGreaterGreaterEqual => Some(AssignOp::UShrAssign),
            TokenKind::AmpersandEqual => Some(AssignOp::BitAndAssign),
            TokenKind::PipeEqual => Some(AssignOp::BitOrAssign),
            TokenKind::CaretEqual => Some(AssignOp::BitXorAssign),
            TokenKind::AmpersandAmpersandEqual => Some(AssignOp::LogicalAndAssign),
            TokenKind::PipePipeEqual => Some(AssignOp::LogicalOrAssign),
            TokenKind::QuestionQuestionEqual => Some(AssignOp::NullishAssign),
            _ => None,
        };
        if let Some(op) = assign_op {
            self.bump()?;
            let rhs = self.parse_assignment_expr()?;
            let end = rhs.loc();
            return Ok(Expr::Assign(Box::new(AssignExpr {
                loc: Self::merge_spans(start, end),
                op,
                left: AssignTarget::Expr(Box::new(lhs)),
                right: Box::new(rhs),
            })));
        }
        Ok(lhs)
    }

    /// Parse a conditional (ternary) expression.
    fn parse_conditional_expr(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let test = self.parse_nullish_coalesce()?;
        if self.eat(TokenKind::Question)? {
            let consequent = self.parse_assignment_expr()?;
            self.expect(TokenKind::Colon)?;
            let alternate = self.parse_assignment_expr()?;
            let end = alternate.loc();
            return Ok(Expr::Conditional(Box::new(
                crate::parser::ast::ConditionalExpr {
                    loc: Self::merge_spans(start, end),
                    test: Box::new(test),
                    consequent: Box::new(consequent),
                    alternate: Box::new(alternate),
                },
            )));
        }
        Ok(test)
    }

    // Nullish coalescing has slightly different grouping rules with && and ||.
    // For simplicity we treat it as same level as logical OR/AND here.
    fn parse_nullish_coalesce(&mut self) -> StatorResult<Expr> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_logical_and()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::PipePipe => LogicalOp::Or,
                TokenKind::QuestionQuestion => LogicalOp::NullishCoalesce,
                _ => break,
            };
            self.bump()?;
            let right = self.parse_logical_and()?;
            let end = right.loc();
            left = Expr::Logical(Box::new(LogicalExpr {
                loc: Self::merge_spans(start, end),
                op,
                left: Box::new(left),
                right: Box::new(right),
            }));
        }
        Ok(left)
    }

    fn parse_logical_and(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_bitwise_or()?;
        while self.peek_kind() == TokenKind::AmpersandAmpersand {
            self.bump()?;
            let right = self.parse_bitwise_or()?;
            let end = right.loc();
            left = Expr::Logical(Box::new(LogicalExpr {
                loc: Self::merge_spans(start, end),
                op: LogicalOp::And,
                left: Box::new(left),
                right: Box::new(right),
            }));
        }
        Ok(left)
    }

    fn parse_bitwise_or(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_bitwise_xor,
            &[(TokenKind::Pipe, BinaryOp::BitOr)],
        )
    }

    fn parse_bitwise_xor(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_bitwise_and,
            &[(TokenKind::Caret, BinaryOp::BitXor)],
        )
    }

    fn parse_bitwise_and(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_equality,
            &[(TokenKind::Ampersand, BinaryOp::BitAnd)],
        )
    }

    fn parse_equality(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_relational,
            &[
                (TokenKind::EqualEqual, BinaryOp::Eq),
                (TokenKind::BangEqual, BinaryOp::NotEq),
                (TokenKind::EqualEqualEqual, BinaryOp::StrictEq),
                (TokenKind::BangEqualEqual, BinaryOp::StrictNotEq),
            ],
        )
    }

    fn parse_relational(&mut self) -> StatorResult<Expr> {
        if self.no_in {
            self.parse_binary_left_assoc(
                Self::parse_shift,
                &[
                    (TokenKind::Less, BinaryOp::Lt),
                    (TokenKind::LessEqual, BinaryOp::LtEq),
                    (TokenKind::Greater, BinaryOp::Gt),
                    (TokenKind::GreaterEqual, BinaryOp::GtEq),
                    (TokenKind::Instanceof, BinaryOp::Instanceof),
                ],
            )
        } else {
            self.parse_binary_left_assoc(
                Self::parse_shift,
                &[
                    (TokenKind::Less, BinaryOp::Lt),
                    (TokenKind::LessEqual, BinaryOp::LtEq),
                    (TokenKind::Greater, BinaryOp::Gt),
                    (TokenKind::GreaterEqual, BinaryOp::GtEq),
                    (TokenKind::Instanceof, BinaryOp::Instanceof),
                    (TokenKind::In, BinaryOp::In),
                ],
            )
        }
    }

    fn parse_shift(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_additive,
            &[
                (TokenKind::LessLess, BinaryOp::Shl),
                (TokenKind::GreaterGreater, BinaryOp::Shr),
                (TokenKind::GreaterGreaterGreater, BinaryOp::UShr),
            ],
        )
    }

    fn parse_additive(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_multiplicative,
            &[
                (TokenKind::Plus, BinaryOp::Add),
                (TokenKind::Minus, BinaryOp::Sub),
            ],
        )
    }

    fn parse_multiplicative(&mut self) -> StatorResult<Expr> {
        self.parse_binary_left_assoc(
            Self::parse_exponentiation,
            &[
                (TokenKind::Star, BinaryOp::Mul),
                (TokenKind::Slash, BinaryOp::Div),
                (TokenKind::Percent, BinaryOp::Rem),
            ],
        )
    }

    fn parse_exponentiation(&mut self) -> StatorResult<Expr> {
        // ** is right-associative.
        let start = self.current_span();
        let base = self.parse_unary()?;
        if self.peek_kind() == TokenKind::StarStar {
            self.bump()?;
            let exp = self.parse_exponentiation()?;
            let end = exp.loc();
            return Ok(Expr::Binary(Box::new(BinaryExpr {
                loc: Self::merge_spans(start, end),
                op: BinaryOp::Exp,
                left: Box::new(base),
                right: Box::new(exp),
            })));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let op = match self.peek_kind() {
            TokenKind::Bang => Some(UnaryOp::Not),
            TokenKind::Tilde => Some(UnaryOp::BitNot),
            TokenKind::Minus => Some(UnaryOp::Minus),
            TokenKind::Plus => Some(UnaryOp::Plus),
            TokenKind::Typeof => Some(UnaryOp::Typeof),
            TokenKind::Void => Some(UnaryOp::Void),
            TokenKind::Delete => Some(UnaryOp::Delete),
            TokenKind::Await => {
                self.bump()?;
                let argument = self.parse_unary()?;
                let end = argument.loc();
                return Ok(Expr::Await(Box::new(AwaitExpr {
                    loc: Self::merge_spans(start, end),
                    argument: Box::new(argument),
                })));
            }
            TokenKind::PlusPlus => {
                self.bump()?;
                let arg = self.parse_unary()?;
                let end = arg.loc();
                return Ok(Expr::Update(Box::new(UpdateExpr {
                    loc: Self::merge_spans(start, end),
                    op: UpdateOp::Increment,
                    prefix: true,
                    argument: Box::new(arg),
                })));
            }
            TokenKind::MinusMinus => {
                self.bump()?;
                let arg = self.parse_unary()?;
                let end = arg.loc();
                return Ok(Expr::Update(Box::new(UpdateExpr {
                    loc: Self::merge_spans(start, end),
                    op: UpdateOp::Decrement,
                    prefix: true,
                    argument: Box::new(arg),
                })));
            }
            _ => None,
        };
        if let Some(op) = op {
            self.bump()?;
            let arg = self.parse_unary()?;
            let end = arg.loc();
            return Ok(Expr::Unary(Box::new(UnaryExpr {
                loc: Self::merge_spans(start, end),
                op,
                argument: Box::new(arg),
            })));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let expr = self.parse_call_member()?;
        if !self.current.had_line_terminator_before {
            let op = match self.peek_kind() {
                TokenKind::PlusPlus => Some(UpdateOp::Increment),
                TokenKind::MinusMinus => Some(UpdateOp::Decrement),
                _ => None,
            };
            if let Some(op) = op {
                let end = self.current_span();
                self.bump()?;
                return Ok(Expr::Update(Box::new(UpdateExpr {
                    loc: Self::merge_spans(start, end),
                    op,
                    prefix: false,
                    argument: Box::new(expr),
                })));
            }
        }
        Ok(expr)
    }

    fn parse_call_member(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::Dot => {
                    self.bump()?;
                    // Allow keywords as property names.
                    let prop_tok = self.bump()?;
                    let name = self.name_from_token(&prop_tok)?;
                    let prop_ident = Ident {
                        loc: prop_tok.span,
                        name,
                    };
                    let end = prop_tok.span;
                    expr = Expr::Member(Box::new(crate::parser::ast::MemberExpr {
                        loc: Self::merge_spans(start, end),
                        object: Box::new(expr),
                        property: crate::parser::ast::MemberProp::Ident(prop_ident),
                        is_computed: false,
                    }));
                }
                TokenKind::LeftBracket => {
                    self.bump()?;
                    let prop = self.parse_expr()?;
                    let end = self.current_span();
                    self.expect(TokenKind::RightBracket)?;
                    expr = Expr::Member(Box::new(crate::parser::ast::MemberExpr {
                        loc: Self::merge_spans(start, end),
                        object: Box::new(expr),
                        property: crate::parser::ast::MemberProp::Computed(Box::new(prop)),
                        is_computed: true,
                    }));
                }
                TokenKind::LeftParen => {
                    self.bump()?;
                    let args = self.parse_call_args()?;
                    let end = self.current_span();
                    // consume ')' already eaten by parse_call_args
                    expr = Expr::Call(Box::new(crate::parser::ast::CallExpr {
                        loc: Self::merge_spans(start, end),
                        callee: Box::new(expr),
                        arguments: args,
                    }));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> StatorResult<Vec<Expr>> {
        let mut args = Vec::new();
        while self.peek_kind() != TokenKind::RightParen {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in argument list"));
            }
            if self.peek_kind() == TokenKind::DotDotDot {
                let spread_start = self.current_span();
                self.bump()?; // consume `...`
                let argument = self.parse_assignment_expr()?;
                let arg_end = argument.loc().end;
                args.push(Expr::Spread(Box::new(SpreadElement {
                    loc: SourceLocation {
                        start: spread_start.start,
                        end: arg_end,
                    },
                    argument: Box::new(argument),
                })));
            } else {
                args.push(self.parse_assignment_expr()?);
            }
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        self.expect(TokenKind::RightParen)?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> StatorResult<Expr> {
        self.enter()?;
        let result = self.parse_primary_inner();
        self.leave();
        result
    }

    fn parse_primary_inner(&mut self) -> StatorResult<Expr> {
        let span = self.current_span();
        match self.peek_kind() {
            TokenKind::NumericLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Number(n) => n,
                    _ => return Err(Self::error_at(tok.span, "invalid numeric token")),
                };
                Ok(Expr::Num(NumLit {
                    loc: tok.span,
                    value,
                    raw: String::new(),
                }))
            }
            TokenKind::StringLiteral => {
                let tok = self.bump()?;
                let value = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => return Err(Self::error_at(tok.span, "invalid string token")),
                };
                Ok(Expr::Str(StringLit {
                    loc: tok.span,
                    value,
                }))
            }
            TokenKind::RegExpLiteral => {
                let tok = self.bump()?;
                let raw = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => return Err(Self::error_at(tok.span, "invalid regexp token")),
                };
                // raw is e.g. "/abc/gi" – strip leading '/' then split at last '/'
                let body = &raw[1..];
                let closing = body
                    .rfind('/')
                    .ok_or_else(|| Self::error_at(tok.span, "malformed regexp literal"))?;
                let pattern = body[..closing].to_string();
                let flags = body[closing + 1..].to_string();
                Ok(Expr::Regexp(RegExpLit {
                    loc: tok.span,
                    pattern,
                    flags,
                }))
            }
            TokenKind::True => {
                self.bump()?;
                Ok(Expr::Bool(BoolLit {
                    loc: span,
                    value: true,
                }))
            }
            TokenKind::False => {
                self.bump()?;
                Ok(Expr::Bool(BoolLit {
                    loc: span,
                    value: false,
                }))
            }
            TokenKind::Null => {
                self.bump()?;
                Ok(Expr::Null(NullLit { loc: span }))
            }
            TokenKind::This => {
                self.bump()?;
                Ok(Expr::This(crate::parser::ast::ThisExpr { loc: span }))
            }
            TokenKind::Identifier => {
                let tok = self.bump()?;
                let name = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => return Err(Self::error_at(tok.span, "invalid identifier token")),
                };
                Ok(Expr::Ident(Ident {
                    loc: tok.span,
                    name,
                }))
            }
            TokenKind::LeftParen => {
                self.bump()?;
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RightParen)?;
                Ok(expr)
            }
            TokenKind::LeftBracket => {
                let start = self.current_span();
                self.bump()?; // consume `[`
                let mut elements: Vec<Option<Expr>> = Vec::new();
                while self.peek_kind() != TokenKind::RightBracket {
                    if self.peek_kind() == TokenKind::Eof {
                        return Err(self.error("unexpected end of input in array literal"));
                    }
                    if self.peek_kind() == TokenKind::Comma {
                        // elision
                        elements.push(None);
                        self.bump()?;
                        continue;
                    }
                    if self.peek_kind() == TokenKind::DotDotDot {
                        let spread_start = self.current_span();
                        self.bump()?; // consume `...`
                        let argument = self.parse_assignment_expr()?;
                        let arg_end = argument.loc().end;
                        elements.push(Some(Expr::Spread(Box::new(SpreadElement {
                            loc: SourceLocation {
                                start: spread_start.start,
                                end: arg_end,
                            },
                            argument: Box::new(argument),
                        }))));
                    } else {
                        elements.push(Some(self.parse_assignment_expr()?));
                    }
                    if !self.eat(TokenKind::Comma)? {
                        break;
                    }
                }
                let end = self.expect(TokenKind::RightBracket)?;
                Ok(Expr::Array(Box::new(ArrayExpr {
                    loc: SourceLocation {
                        start: start.start,
                        end: end.span.end,
                    },
                    elements,
                })))
            }
            TokenKind::LeftBrace => {
                let start = self.current_span();
                self.bump()?; // consume `{`
                let mut properties: Vec<ObjectProp> = Vec::new();
                while self.peek_kind() != TokenKind::RightBrace {
                    if self.peek_kind() == TokenKind::Eof {
                        return Err(self.error("unexpected end of input in object literal"));
                    }
                    // Spread property: `...expr`
                    if self.peek_kind() == TokenKind::DotDotDot {
                        let spread_start = self.current_span();
                        self.bump()?; // consume `...`
                        let argument = self.parse_assignment_expr()?;
                        let arg_end = argument.loc().end;
                        properties.push(ObjectProp::Spread(SpreadElement {
                            loc: SourceLocation {
                                start: spread_start.start,
                                end: arg_end,
                            },
                            argument: Box::new(argument),
                        }));
                    } else {
                        // Parse the property.
                        let prop_start = self.current_span();

                        // Check for `async method() { … }`.
                        let is_async_method = if self.peek_kind() == TokenKind::Async {
                            let saved_scanner = self.scanner.clone();
                            let saved_current = self.current.clone();
                            self.bump()?; // tentatively consume `async`
                            let is_method = !self.current.had_line_terminator_before
                                && self.peek_kind() != TokenKind::Colon
                                && self.peek_kind() != TokenKind::Comma
                                && self.peek_kind() != TokenKind::RightBrace
                                && self.peek_kind() != TokenKind::LeftParen;
                            if is_method {
                                true
                            } else {
                                self.scanner = saved_scanner;
                                self.current = saved_current;
                                false
                            }
                        } else {
                            false
                        };

                        // Check for getter/setter: `get key() { … }`, `set key(v) { … }`.
                        let getter_setter_kind = if !is_async_method
                            && (self.peek_kind() == TokenKind::Get
                                || self.peek_kind() == TokenKind::Set)
                        {
                            let saved_scanner = self.scanner.clone();
                            let saved_current = self.current.clone();
                            let accessor_kind = if self.peek_kind() == TokenKind::Get {
                                MethodKind::Get
                            } else {
                                MethodKind::Set
                            };
                            self.bump()?; // tentatively consume get/set
                            // If the next token is `:`, `,`, or `}`, then
                            // get/set is the property name, not an accessor.
                            if self.peek_kind() == TokenKind::Colon
                                || self.peek_kind() == TokenKind::Comma
                                || self.peek_kind() == TokenKind::RightBrace
                                || self.peek_kind() == TokenKind::LeftParen
                            {
                                self.scanner = saved_scanner;
                                self.current = saved_current;
                                None
                            } else {
                                Some(accessor_kind)
                            }
                        } else {
                            None
                        };

                        // Parse the property key.
                        let (key, is_computed) = match self.peek_kind() {
                            // Computed property: `[expr]`
                            TokenKind::LeftBracket => {
                                self.bump()?; // consume `[`
                                let key_expr = self.parse_assignment_expr()?;
                                self.expect(TokenKind::RightBracket)?;
                                (PropKey::Computed(Box::new(key_expr)), true)
                            }
                            // String literal key
                            TokenKind::StringLiteral => {
                                let tok = self.bump()?;
                                let value = match tok.value {
                                    TokenValue::Str(s) => s,
                                    _ => {
                                        return Err(Self::error_at(
                                            tok.span,
                                            "invalid string token",
                                        ));
                                    }
                                };
                                (
                                    PropKey::Str(StringLit {
                                        loc: tok.span,
                                        value,
                                    }),
                                    false,
                                )
                            }
                            // Numeric literal key
                            TokenKind::NumericLiteral => {
                                let tok = self.bump()?;
                                let value = match tok.value {
                                    TokenValue::Number(n) => n,
                                    _ => {
                                        return Err(Self::error_at(
                                            tok.span,
                                            "invalid numeric token",
                                        ));
                                    }
                                };
                                (
                                    PropKey::Num(NumLit {
                                        loc: tok.span,
                                        value,
                                        raw: String::new(),
                                    }),
                                    false,
                                )
                            }
                            // Identifier (or keyword used as property name)
                            _ => {
                                let tok = self.bump()?;
                                let name = match &tok.value {
                                    TokenValue::Str(s) => s.clone(),
                                    TokenValue::None => {
                                        // Keywords produce TokenValue::None;
                                        // use the debug representation lowercased.
                                        format!("{:?}", tok.kind).to_lowercase()
                                    }
                                    _ => {
                                        return Err(Self::error_at(
                                            tok.span,
                                            "expected property name",
                                        ));
                                    }
                                };
                                (
                                    PropKey::Ident(Ident {
                                        loc: tok.span,
                                        name,
                                    }),
                                    false,
                                )
                            }
                        };

                        // Determine property value kind.
                        let value = if let Some(accessor_kind) = getter_setter_kind {
                            // getter/setter: `get key() { … }` / `set key(v) { … }`
                            let fn_start = self.current_span();
                            self.expect(TokenKind::LeftParen)?;
                            let params = self.parse_formal_params()?;
                            let body = self.parse_block()?;
                            let fn_end = body.loc;
                            let fn_expr = FnExpr {
                                loc: Self::merge_spans(fn_start, fn_end),
                                id: None,
                                is_async: false,
                                is_generator: false,
                                params,
                                body,
                            };
                            match accessor_kind {
                                MethodKind::Get => PropValue::Get(fn_expr),
                                MethodKind::Set => PropValue::Set(fn_expr),
                                _ => unreachable!(),
                            }
                        } else if is_async_method || self.peek_kind() == TokenKind::LeftParen {
                            // Method shorthand: `key(params) { … }` or `async key(params) { … }`
                            let fn_start = self.current_span();
                            self.expect(TokenKind::LeftParen)?;
                            let params = self.parse_formal_params()?;
                            let body = self.parse_block()?;
                            let fn_end = body.loc;
                            PropValue::Method(FnExpr {
                                loc: Self::merge_spans(fn_start, fn_end),
                                id: None,
                                is_async: is_async_method,
                                is_generator: false,
                                params,
                                body,
                            })
                        } else if self.eat(TokenKind::Colon)? {
                            PropValue::Value(Box::new(self.parse_assignment_expr()?))
                        } else {
                            // Shorthand — only valid for identifier keys.
                            match &key {
                                PropKey::Ident(_) => PropValue::Shorthand,
                                _ => {
                                    return Err(Self::error_at(
                                        prop_start,
                                        "expected ':' after property name",
                                    ));
                                }
                            }
                        };

                        let prop_end = match &value {
                            PropValue::Value(expr) => expr.loc().end,
                            PropValue::Shorthand => prop_start.end,
                            PropValue::Get(f) | PropValue::Set(f) | PropValue::Method(f) => {
                                f.loc.end
                            }
                        };

                        properties.push(ObjectProp::Prop(Box::new(Prop {
                            loc: SourceLocation {
                                start: prop_start.start,
                                end: prop_end,
                            },
                            key,
                            is_computed,
                            value,
                        })));
                    }
                    if !self.eat(TokenKind::Comma)? {
                        break;
                    }
                }
                let end = self.expect(TokenKind::RightBrace)?;
                Ok(Expr::Object(Box::new(ObjectExpr {
                    loc: SourceLocation {
                        start: start.start,
                        end: end.span.end,
                    },
                    properties,
                })))
            }
            TokenKind::New => {
                let new_start = self.current_span();
                self.bump()?; // consume `new`
                // Parse the constructor target. Using parse_primary() allows
                // nested `new` (e.g. `new new Foo()`) while avoiding the
                // call-expression `()` being consumed by parse_call_member().
                let mut callee = self.parse_primary()?;
                // Allow member-access chains on the callee so that
                // `new Foo.Bar()` and `new a[b]()` work correctly.
                loop {
                    match self.peek_kind() {
                        TokenKind::Dot => {
                            self.bump()?;
                            let prop_tok = self.bump()?;
                            let name = self.name_from_token(&prop_tok)?;
                            let prop_ident = Ident {
                                loc: prop_tok.span,
                                name,
                            };
                            let end = prop_tok.span;
                            callee = Expr::Member(Box::new(crate::parser::ast::MemberExpr {
                                loc: Self::merge_spans(new_start, end),
                                object: Box::new(callee),
                                property: crate::parser::ast::MemberProp::Ident(prop_ident),
                                is_computed: false,
                            }));
                        }
                        TokenKind::LeftBracket => {
                            self.bump()?;
                            let prop = self.parse_expr()?;
                            let end = self.current_span();
                            self.expect(TokenKind::RightBracket)?;
                            callee = Expr::Member(Box::new(crate::parser::ast::MemberExpr {
                                loc: Self::merge_spans(new_start, end),
                                object: Box::new(callee),
                                property: crate::parser::ast::MemberProp::Computed(Box::new(prop)),
                                is_computed: true,
                            }));
                        }
                        _ => break,
                    }
                }
                // Parse optional argument list.
                let arguments = if self.peek_kind() == TokenKind::LeftParen {
                    self.bump()?;
                    self.parse_call_args()?
                } else {
                    vec![]
                };
                let end = self.current_span();
                Ok(Expr::New(Box::new(NewExpr {
                    loc: Self::merge_spans(new_start, end),
                    callee: Box::new(callee),
                    arguments,
                })))
            }
            TokenKind::Function => {
                let fn_span = self.current_span();
                self.bump()?;
                self.parse_fn_expr(fn_span, false)
            }
            TokenKind::Async => {
                // `async function …` — async function expression.
                // `async (…) => …` or `async x => …` — async arrow (handled
                // at assignment-expression level, but could also appear here
                // as a primary if it is the start of an expression).
                let saved_scanner = self.scanner.clone();
                let saved_current = self.current.clone();
                let async_tok = self.bump()?; // consume `async`
                if self.peek_kind() == TokenKind::Function
                    && !self.current.had_line_terminator_before
                {
                    let fn_tok = self.bump()?; // consume `function`
                    self.parse_fn_expr(Self::merge_spans(async_tok.span, fn_tok.span), true)
                } else {
                    // Not `async function` — restore and emit as identifier
                    // so the caller (parse_assignment_expr) can detect the
                    // `async` arrow case, or it's just a variable named
                    // `async`.
                    self.scanner = saved_scanner;
                    self.current = saved_current;
                    let tok = self.bump()?;
                    Ok(Expr::Ident(Ident {
                        loc: tok.span,
                        name: "async".into(),
                    }))
                }
            }
            TokenKind::Class => self.parse_class_expr(),
            TokenKind::NoSubstitutionTemplate => {
                let tok = self.bump()?;
                let raw = match tok.value {
                    TokenValue::Str(s) => s,
                    _ => String::new(),
                };
                Ok(Expr::Template(Box::new(TemplateLit {
                    loc: tok.span,
                    quasis: vec![TemplateElement {
                        loc: tok.span,
                        raw: raw.clone(),
                        cooked: Some(raw),
                        tail: true,
                    }],
                    expressions: vec![],
                })))
            }
            TokenKind::TemplateHead => {
                let start = self.current_span();
                let head_tok = self.bump()?;
                let head_raw = match head_tok.value {
                    TokenValue::Str(s) => s,
                    _ => String::new(),
                };
                let mut quasis = vec![TemplateElement {
                    loc: head_tok.span,
                    raw: head_raw.clone(),
                    cooked: Some(head_raw),
                    tail: false,
                }];
                let mut expressions = Vec::new();

                loop {
                    expressions.push(self.parse_expr()?);
                    // After the expression the scanner automatically produces
                    // TemplateMiddle or TemplateTail when it sees `}`.
                    match self.peek_kind() {
                        TokenKind::TemplateMiddle => {
                            let mid_tok = self.bump()?;
                            let mid_raw = match mid_tok.value {
                                TokenValue::Str(s) => s,
                                _ => String::new(),
                            };
                            quasis.push(TemplateElement {
                                loc: mid_tok.span,
                                raw: mid_raw.clone(),
                                cooked: Some(mid_raw),
                                tail: false,
                            });
                        }
                        TokenKind::TemplateTail => {
                            let tail_tok = self.bump()?;
                            let tail_raw = match tail_tok.value {
                                TokenValue::Str(s) => s,
                                _ => String::new(),
                            };
                            quasis.push(TemplateElement {
                                loc: tail_tok.span,
                                raw: tail_raw.clone(),
                                cooked: Some(tail_raw),
                                tail: true,
                            });
                            break;
                        }
                        _ => {
                            return Err(self.error("expected template continuation"));
                        }
                    }
                }

                let end = quasis.last().map(|q| q.loc).unwrap_or(start);
                Ok(Expr::Template(Box::new(TemplateLit {
                    loc: Self::merge_spans(start, end),
                    quasis,
                    expressions,
                })))
            }
            other => Err(self.error(&format!("unexpected token {:?}", other))),
        }
    }

    fn parse_fn_expr(&mut self, fn_span: Span, is_async: bool) -> StatorResult<Expr> {
        let is_generator = self.eat(TokenKind::Star)?;
        let id = if self.peek_kind() == TokenKind::Identifier {
            let tok = self.bump()?;
            Some(self.ident_from_token(&tok)?)
        } else {
            None
        };
        self.expect(TokenKind::LeftParen)?;
        let params = self.parse_formal_params()?;
        let body = self.parse_block()?;
        let end = body.loc;
        Ok(Expr::Fn(Box::new(crate::parser::ast::FnExpr {
            loc: Self::merge_spans(fn_span, end),
            id,
            is_async,
            is_generator,
            params,
            body,
        })))
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Generic left-associative binary expression parser.
    ///
    /// `next` is the parser for the higher-precedence level.
    /// `ops` maps token kinds to `BinaryOp` variants.
    fn parse_binary_left_assoc(
        &mut self,
        next: fn(&mut Self) -> StatorResult<Expr>,
        ops: &[(TokenKind, BinaryOp)],
    ) -> StatorResult<Expr> {
        let start = self.current_span();
        let mut left = next(self)?;
        loop {
            let maybe_op = ops
                .iter()
                .find(|(k, _)| *k == self.peek_kind())
                .map(|(_, op)| *op);
            if let Some(op) = maybe_op {
                self.bump()?;
                let right = next(self)?;
                let end = right.loc();
                left = Expr::Binary(Box::new(BinaryExpr {
                    loc: Self::merge_spans(start, end),
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn ident_from_token(&self, tok: &Token) -> StatorResult<Ident> {
        let name = match &tok.value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(Self::error_at(tok.span, "expected identifier")),
        };
        Ok(Ident {
            loc: tok.span,
            name,
        })
    }

    /// Accept identifiers **and** keyword tokens as property names.
    fn name_from_token(&self, tok: &Token) -> StatorResult<String> {
        match &tok.value {
            TokenValue::Str(s) => Ok(s.clone()),
            TokenValue::None => {
                // Keywords are valid property names after `.`.
                Ok(format!("{:?}", tok.kind).to_lowercase())
            }
            _ => Err(Self::error_at(tok.span, "expected property name")),
        }
    }

    // ── Arrow-function helpers ────────────────────────────────────────────

    /// Parse the body of an arrow function: either a block `{ … }` or a
    /// concise expression body.
    fn parse_arrow_body(&mut self) -> StatorResult<ArrowBody> {
        if self.peek_kind() == TokenKind::LeftBrace {
            Ok(ArrowBody::Block(self.parse_block()?))
        } else {
            Ok(ArrowBody::Expr(Box::new(self.parse_assignment_expr()?)))
        }
    }

    /// Return the source location of an arrow body (for span merging).
    fn arrow_body_loc(&self, body: &ArrowBody) -> Span {
        match body {
            ArrowBody::Block(b) => Span {
                start: b.loc.start,
                end: b.loc.end,
            },
            ArrowBody::Expr(e) => {
                let loc = e.loc();
                Span {
                    start: loc.start,
                    end: loc.end,
                }
            }
        }
    }

    /// Reinterpret a parsed expression as arrow-function parameters
    /// (the "cover grammar" conversion).
    fn expr_to_arrow_params(&self, expr: Expr) -> StatorResult<Vec<Param>> {
        match expr {
            // `x => body`  or  `(x) => body`
            Expr::Ident(id) => Ok(vec![Param {
                loc: id.loc,
                pat: Pat::Ident(id),
                default: None,
            }]),
            // `(x, y) => body`
            Expr::Sequence(seq) => seq
                .expressions
                .into_iter()
                .map(|e| self.expr_to_single_param(e))
                .collect(),
            // `(x = 1) => body`  (single param with default)
            Expr::Assign(assign) if assign.op == AssignOp::Assign => {
                Ok(vec![self.assign_expr_to_param(*assign)?])
            }
            other => {
                let loc = other.loc();
                Err(Self::error_at(
                    Span {
                        start: loc.start,
                        end: loc.end,
                    },
                    "invalid arrow-function parameter",
                ))
            }
        }
    }

    /// Convert a single expression to a single parameter.
    fn expr_to_single_param(&self, expr: Expr) -> StatorResult<Param> {
        match expr {
            Expr::Ident(id) => Ok(Param {
                loc: id.loc,
                pat: Pat::Ident(id),
                default: None,
            }),
            Expr::Assign(assign) if assign.op == AssignOp::Assign => {
                self.assign_expr_to_param(*assign)
            }
            other => {
                let loc = other.loc();
                Err(Self::error_at(
                    Span {
                        start: loc.start,
                        end: loc.end,
                    },
                    "invalid arrow-function parameter",
                ))
            }
        }
    }

    /// Convert an assignment expression (`id = default`) to a parameter with
    /// a default value.
    fn assign_expr_to_param(&self, assign: AssignExpr) -> StatorResult<Param> {
        if let AssignTarget::Expr(lhs) = assign.left
            && let Expr::Ident(id) = *lhs
        {
            return Ok(Param {
                loc: assign.loc,
                pat: Pat::Ident(id),
                default: Some(*assign.right),
            });
        }
        Err(Self::error_at(
            Span {
                start: assign.loc.start,
                end: assign.loc.end,
            },
            "invalid arrow-function parameter with default",
        ))
    }

    fn merge_spans(a: Span, b: Span) -> SourceLocation {
        let start = if a.start.offset <= b.start.offset {
            a.start
        } else {
            b.start
        };
        let end = if a.end.offset >= b.end.offset {
            a.end
        } else {
            b.end
        };
        // If a or b are empty (offset == 0 sentinel), prefer the other.
        let start = if a.start.offset == 0 && b.start.offset != 0 {
            b.start
        } else {
            start
        };
        Span {
            start,
            end: if end.offset == 0 { start } else { end },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `source` as a JavaScript script and return the root [`Program`].
///
/// # Errors
///
/// Returns a [`StatorError::SyntaxError`] if the source contains a syntax
/// error.  The message includes the source position in `at LINE:COL — REASON`
/// format.
///
/// # Example
///
/// ```
/// use stator_core::parser::parse;
/// use stator_core::parser::ast::{ProgramItem, Stmt};
///
/// let prog = parse("var x = 1 + 2;").unwrap();
/// assert_eq!(prog.body.len(), 1);
/// assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::VarDecl(_))));
/// ```
pub fn parse(source: &str) -> StatorResult<Program> {
    let mut parser = Parser::new(source)?;
    let program = parser.parse_program()?;
    Ok(program)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{ObjectPatProp, Pat, ProgramItem, PropKey, Stmt, VarKind};

    #[test]
    fn test_parse_empty_program() {
        let prog = parse("").unwrap();
        assert!(prog.body.is_empty());
    }

    #[test]
    fn test_parse_var_decl_with_number() {
        let prog = parse("var x = 42;").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::VarDecl(_))));
    }

    #[test]
    fn test_parse_var_binary_expr() {
        let prog = parse("var x = 1 + 2;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            assert_eq!(vd.declarators.len(), 1);
            assert!(vd.declarators[0].init.is_some());
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_syntax_error_var_eq() {
        let err = parse("var = ;").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("SyntaxError"),
            "expected SyntaxError in {msg:?}"
        );
        assert!(msg.contains("1:5"), "expected position 1:5 in {msg:?}");
        assert!(
            msg.contains("expected binding pattern"),
            "expected 'expected binding pattern' in {msg:?}"
        );
    }

    #[test]
    fn test_parse_expression_stmt() {
        let prog = parse("1 + 2;").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::Expr(_))));
    }

    #[test]
    fn test_parse_if_stmt() {
        let prog = parse("if (x) { y; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::If(_))));
    }

    #[test]
    fn test_parse_while_stmt() {
        let prog = parse("while (i < 10) { i = i + 1; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::While(_))));
    }

    #[test]
    fn test_parse_function_decl() {
        let prog = parse("function add(a, b) { return a + b; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::FnDecl(_))));
    }

    #[test]
    fn test_parse_return_stmt() {
        let prog = parse("function f() { return 1; }").unwrap();
        assert_eq!(prog.body.len(), 1);
    }

    #[test]
    fn test_parse_multiple_stmts() {
        let prog = parse("var a = 1; var b = 2; var c = a + b;").unwrap();
        assert_eq!(prog.body.len(), 3);
    }

    #[test]
    fn test_parse_array_literal_simple() {
        let prog = parse("var a = [1, 2, 3];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    assert_eq!(arr.elements.len(), 3);
                    assert!(arr.elements.iter().all(|e| e.is_some()));
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_array_literal_empty() {
        let prog = parse("var a = [];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    assert!(arr.elements.is_empty());
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_array_literal_trailing_comma() {
        let prog = parse("var a = [1, 2,];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    // trailing comma must not produce an extra None (elision)
                    assert_eq!(arr.elements.len(), 2);
                    assert!(arr.elements.iter().all(|e| e.is_some()));
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_array_literal_elision() {
        let prog = parse("var a = [1,,3];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    assert_eq!(arr.elements.len(), 3);
                    assert!(arr.elements[0].is_some());
                    assert!(arr.elements[1].is_none());
                    assert!(arr.elements[2].is_some());
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_array_literal_nested() {
        let prog = parse("var a = [[1], [2]];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    assert_eq!(arr.elements.len(), 2);
                    for elem in &arr.elements {
                        assert!(matches!(elem, Some(Expr::Array(_))));
                    }
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_array_literal_spread() {
        let prog = parse("var a = [...b];").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Array(arr) = init.as_ref() {
                    assert_eq!(arr.elements.len(), 1);
                    assert!(matches!(arr.elements[0], Some(Expr::Spread(_))));
                } else {
                    panic!("expected Array init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── Object literal tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_object_literal_empty() {
        let prog = parse("var o = {};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert!(obj.properties.is_empty());
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_simple() {
        let prog = parse("var o = {x: 1, y: 2};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 2);
                    // Both should be Prop variants with Value
                    for prop in &obj.properties {
                        assert!(matches!(prop, ObjectProp::Prop(_)));
                    }
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_shorthand() {
        let prog = parse("var o = {x, y};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 2);
                    for prop in &obj.properties {
                        if let ObjectProp::Prop(p) = prop {
                            assert!(matches!(p.value, PropValue::Shorthand));
                        } else {
                            panic!("expected Prop");
                        }
                    }
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_string_key() {
        let prog = parse(r#"var o = {"name": 42};"#).unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 1);
                    if let ObjectProp::Prop(p) = &obj.properties[0] {
                        assert!(matches!(&p.key, PropKey::Str(_)));
                    } else {
                        panic!("expected Prop");
                    }
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_numeric_key() {
        let prog = parse("var o = {42: true};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 1);
                    if let ObjectProp::Prop(p) = &obj.properties[0] {
                        assert!(matches!(&p.key, PropKey::Num(_)));
                    } else {
                        panic!("expected Prop");
                    }
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_computed_key() {
        let prog = parse("var o = {[x]: 1};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 1);
                    if let ObjectProp::Prop(p) = &obj.properties[0] {
                        assert!(matches!(&p.key, PropKey::Computed(_)));
                        assert!(p.is_computed);
                    } else {
                        panic!("expected Prop");
                    }
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_spread() {
        let prog = parse("var o = {...a};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 1);
                    assert!(matches!(&obj.properties[0], ObjectProp::Spread(_)));
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_trailing_comma() {
        let prog = parse("var o = {a: 1, b: 2,};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 2);
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_parse_object_literal_mixed() {
        let prog = parse("var o = {a: 1, b, [c]: 3, ...d};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 4);
                    assert!(matches!(&obj.properties[0], ObjectProp::Prop(_)));
                    // shorthand
                    if let ObjectProp::Prop(p) = &obj.properties[1] {
                        assert!(matches!(p.value, PropValue::Shorthand));
                    }
                    // computed
                    if let ObjectProp::Prop(p) = &obj.properties[2] {
                        assert!(p.is_computed);
                    }
                    // spread
                    assert!(matches!(&obj.properties[3], ObjectProp::Spread(_)));
                } else {
                    panic!("expected Object init");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── new expression tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_new_with_args() {
        let prog = parse("new Foo();").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(n) = es.expr.as_ref() {
                assert!(matches!(n.callee.as_ref(), Expr::Ident(_)));
                assert_eq!(n.arguments.len(), 0);
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_without_args() {
        let prog = parse("new Foo;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(n) = es.expr.as_ref() {
                assert!(matches!(n.callee.as_ref(), Expr::Ident(_)));
                assert_eq!(n.arguments.len(), 0);
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_with_multiple_args() {
        let prog = parse("new Foo(a, b);").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(n) = es.expr.as_ref() {
                assert!(matches!(n.callee.as_ref(), Expr::Ident(_)));
                assert_eq!(n.arguments.len(), 2);
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_member_access_after() {
        // `new Foo().bar` → MemberExpr { object: NewExpr { Foo, [] }, .bar }
        let prog = parse("new Foo().bar;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::Member(m) = es.expr.as_ref() {
                assert!(matches!(m.object.as_ref(), Expr::New(_)));
            } else {
                panic!("expected Member expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_member_callee() {
        // `new Foo.Bar()` → NewExpr { callee: MemberExpr { Foo, .Bar }, args: [] }
        let prog = parse("new Foo.Bar();").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(n) = es.expr.as_ref() {
                assert!(matches!(n.callee.as_ref(), Expr::Member(_)));
                assert_eq!(n.arguments.len(), 0);
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_nested() {
        // `new new Foo()` → NewExpr { callee: NewExpr { Foo, [] }, args: [] }
        let prog = parse("new new Foo();").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(outer) = es.expr.as_ref() {
                assert!(matches!(outer.callee.as_ref(), Expr::New(_)));
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_new_error() {
        // The Test262 harness pattern: `new Error("message")`
        let prog = parse("var e = new Error(\"fail\");").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::New(n) = init.as_ref() {
                    assert_eq!(n.arguments.len(), 1);
                } else {
                    panic!("expected New expr");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── switch statement tests ───────────────────────────────────────────

    #[test]
    fn test_parse_switch_basic() {
        let prog = parse("switch (x) { case 1: break; case 2: break; default: break; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Switch(sw)) = &prog.body[0] {
            assert_eq!(sw.cases.len(), 3);
            // First two cases have a test; default does not.
            assert!(sw.cases[0].test.is_some());
            assert!(sw.cases[1].test.is_some());
            assert!(sw.cases[2].test.is_none());
            // Each case has one `break` statement.
            assert_eq!(sw.cases[0].consequent.len(), 1);
            assert_eq!(sw.cases[1].consequent.len(), 1);
            assert_eq!(sw.cases[2].consequent.len(), 1);
        } else {
            panic!("expected Switch statement");
        }
    }

    #[test]
    fn test_parse_switch_empty_cases() {
        // Fall-through cases with no body.
        let prog = parse("switch (x) { case 1: case 2: break; }").unwrap();
        if let ProgramItem::Stmt(Stmt::Switch(sw)) = &prog.body[0] {
            assert_eq!(sw.cases.len(), 2);
            // First case has no statements (fall-through).
            assert!(sw.cases[0].consequent.is_empty());
            // Second case has the break.
            assert_eq!(sw.cases[1].consequent.len(), 1);
        } else {
            panic!("expected Switch statement");
        }
    }

    #[test]
    fn test_parse_switch_default_only() {
        let prog = parse("switch (val) { default: x = 1; }").unwrap();
        if let ProgramItem::Stmt(Stmt::Switch(sw)) = &prog.body[0] {
            assert_eq!(sw.cases.len(), 1);
            assert!(sw.cases[0].test.is_none());
            assert_eq!(sw.cases[0].consequent.len(), 1);
        } else {
            panic!("expected Switch statement");
        }
    }

    #[test]
    fn test_parse_switch_no_cases() {
        let prog = parse("switch (x) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::Switch(sw)) = &prog.body[0] {
            assert!(sw.cases.is_empty());
        } else {
            panic!("expected Switch statement");
        }
    }

    // ── Arrow function tests ─────────────────────────────────────────────

    use crate::parser::ast::{ArrowBody, ArrowExpr};

    /// Helper: parse an expression statement and return the inner `Expr`.
    fn parse_expr_stmt(src: &str) -> Expr {
        let prog = parse(src).unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::Expr(es)) = prog.body.into_iter().next().unwrap() {
            *es.expr
        } else {
            panic!("expected expression statement");
        }
    }

    #[test]
    fn test_arrow_no_params() {
        // () => 42
        let expr = parse_expr_stmt("() => 42;");
        if let Expr::Arrow(arrow) = &expr {
            assert!(!arrow.is_async);
            assert!(arrow.params.is_empty());
            assert!(matches!(&arrow.body, ArrowBody::Expr(_)));
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_single_param_no_parens() {
        // x => x + 1
        let expr = parse_expr_stmt("x => x + 1;");
        if let Expr::Arrow(arrow) = &expr {
            assert_eq!(arrow.params.len(), 1);
            if let Pat::Ident(id) = &arrow.params[0].pat {
                assert_eq!(id.name, "x");
            } else {
                panic!("expected Ident param");
            }
            assert!(matches!(&arrow.body, ArrowBody::Expr(_)));
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_single_param_with_parens() {
        // (x) => x * 2
        let expr = parse_expr_stmt("(x) => x * 2;");
        if let Expr::Arrow(arrow) = &expr {
            assert_eq!(arrow.params.len(), 1);
            if let Pat::Ident(id) = &arrow.params[0].pat {
                assert_eq!(id.name, "x");
            } else {
                panic!("expected Ident param");
            }
            assert!(matches!(&arrow.body, ArrowBody::Expr(_)));
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_multiple_params() {
        // (a, b) => a + b
        let expr = parse_expr_stmt("(a, b) => a + b;");
        if let Expr::Arrow(arrow) = &expr {
            assert_eq!(arrow.params.len(), 2);
            if let Pat::Ident(id) = &arrow.params[0].pat {
                assert_eq!(id.name, "a");
            } else {
                panic!("expected Ident param 0");
            }
            if let Pat::Ident(id) = &arrow.params[1].pat {
                assert_eq!(id.name, "b");
            } else {
                panic!("expected Ident param 1");
            }
            assert!(matches!(&arrow.body, ArrowBody::Expr(_)));
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_block_body() {
        // (x) => { return x; }
        let expr = parse_expr_stmt("(x) => { return x; };");
        if let Expr::Arrow(arrow) = &expr {
            assert_eq!(arrow.params.len(), 1);
            if let ArrowBody::Block(block) = &arrow.body {
                assert_eq!(block.body.len(), 1);
                assert!(matches!(&block.body[0], Stmt::Return(_)));
            } else {
                panic!("expected Block body");
            }
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_default_param() {
        // (x = 10) => x
        let expr = parse_expr_stmt("(x = 10) => x;");
        if let Expr::Arrow(arrow) = &expr {
            assert_eq!(arrow.params.len(), 1);
            assert!(arrow.params[0].default.is_some());
            if let Pat::Ident(id) = &arrow.params[0].pat {
                assert_eq!(id.name, "x");
            } else {
                panic!("expected Ident param");
            }
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_arrow_in_var_decl() {
        // var add = (a, b) => a + b;
        let prog = parse("var add = (a, b) => a + b;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                assert!(matches!(init.as_ref(), Expr::Arrow(_)));
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_arrow_no_params_block_body() {
        // () => { return 1; }
        let expr = parse_expr_stmt("() => { return 1; };");
        if let Expr::Arrow(arrow) = &expr {
            assert!(arrow.params.is_empty());
            assert!(matches!(&arrow.body, ArrowBody::Block(_)));
        } else {
            panic!("expected Arrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_parenthesised_expr_not_arrow() {
        // (1 + 2) should remain a binary expression, not be treated as arrow
        let expr = parse_expr_stmt("(1 + 2);");
        assert!(matches!(&expr, Expr::Binary(_)));
    }

    // ── for-in / for-of tests ────────────────────────────────────────────

    #[test]
    fn test_for_in_var() {
        let prog = parse("for (var x in obj) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForIn(fi)) = &prog.body[0] {
            assert!(matches!(
                &fi.left,
                crate::parser::ast::ForInOfLeft::VarDecl(_)
            ));
        } else {
            panic!("expected ForIn, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_for_of_var() {
        let prog = parse("for (var x of arr) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForOf(fo)) = &prog.body[0] {
            assert!(!fo.is_await);
            assert!(matches!(
                &fo.left,
                crate::parser::ast::ForInOfLeft::VarDecl(_)
            ));
        } else {
            panic!("expected ForOf, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_for_of_let() {
        let prog = parse("for (let x of arr) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForOf(fo)) = &prog.body[0] {
            if let crate::parser::ast::ForInOfLeft::VarDecl(vd) = &fo.left {
                assert_eq!(vd.kind, crate::parser::ast::VarKind::Let);
            } else {
                panic!("expected VarDecl left");
            }
        } else {
            panic!("expected ForOf, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_for_in_const() {
        let prog = parse("for (const k in obj) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForIn(fi)) = &prog.body[0] {
            if let crate::parser::ast::ForInOfLeft::VarDecl(vd) = &fi.left {
                assert_eq!(vd.kind, crate::parser::ast::VarKind::Const);
            } else {
                panic!("expected VarDecl left");
            }
        } else {
            panic!("expected ForIn, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_for_in_existing_variable() {
        // for (x in obj) — x is an existing variable, not a declaration
        let prog = parse("for (x in obj) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForIn(fi)) = &prog.body[0] {
            assert!(matches!(&fi.left, crate::parser::ast::ForInOfLeft::Pat(_)));
        } else {
            panic!("expected ForIn, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_for_of_existing_variable() {
        // for (x of arr) — x is an existing variable
        let prog = parse("for (x of arr) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForOf(fo)) = &prog.body[0] {
            assert!(matches!(&fo.left, crate::parser::ast::ForInOfLeft::Pat(_)));
        } else {
            panic!("expected ForOf, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_c_style_for_still_works() {
        let prog = parse("for (var i = 0; i < 10; i = i + 1) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::For(_))));
    }

    #[test]
    fn test_c_style_for_empty_init() {
        let prog = parse("for (;;) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::For(f)) = &prog.body[0] {
            assert!(f.init.is_none());
            assert!(f.test.is_none());
            assert!(f.update.is_none());
        } else {
            panic!("expected For, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_c_style_for_expr_init() {
        let prog = parse("for (i = 0; i < 10; i = i + 1) {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        assert!(matches!(prog.body[0], ProgramItem::Stmt(Stmt::For(_))));
    }

    #[test]
    fn test_for_in_with_body() {
        let prog = parse("for (var key in obj) { x = key; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForIn(fi)) = &prog.body[0] {
            assert!(matches!(fi.body.as_ref(), Stmt::Block(_)));
        } else {
            panic!("expected ForIn");
        }
    }

    #[test]
    fn test_for_of_with_body() {
        let prog = parse("for (let item of items) { process(item); }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ForOf(fo)) = &prog.body[0] {
            assert!(matches!(fo.body.as_ref(), Stmt::Block(_)));
        } else {
            panic!("expected ForOf");
        }
    }

    // ── Class declaration / expression tests ─────────────────────────────

    #[test]
    fn test_class_decl_empty() {
        let prog = parse("class Foo {}").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.id.as_ref().unwrap().name, "Foo");
            assert!(c.super_class.is_none());
            assert!(c.body.body.is_empty());
        } else {
            panic!("expected ClassDecl, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_class_decl_with_constructor() {
        let prog = parse("class Foo { constructor(x) { this.x = x; } }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 1);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(matches!(
                    m.kind,
                    crate::parser::ast::MethodKind::Constructor
                ));
                assert!(!m.is_static);
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_decl_with_method() {
        let prog = parse("class Foo { greet(name) { return name; } }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 1);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(matches!(m.kind, crate::parser::ast::MethodKind::Method));
                assert!(!m.is_static);
                if let crate::parser::ast::PropKey::Ident(ref id) = m.key {
                    assert_eq!(id.name, "greet");
                } else {
                    panic!("expected Ident key");
                }
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_decl_static_method() {
        let prog = parse("class Foo { static create() { return 1; } }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 1);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(m.is_static);
                assert!(matches!(m.kind, crate::parser::ast::MethodKind::Method));
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_decl_extends() {
        let prog = parse("class Bar extends Foo { constructor() {} }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.id.as_ref().unwrap().name, "Bar");
            assert!(c.super_class.is_some());
            assert_eq!(c.body.body.len(), 1);
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_decl_getter_setter() {
        let prog =
            parse("class Foo { get value() { return 1; } set value(v) { this.v = v; } }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 2);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(matches!(m.kind, crate::parser::ast::MethodKind::Get));
            } else {
                panic!("expected getter");
            }
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[1] {
                assert!(matches!(m.kind, crate::parser::ast::MethodKind::Set));
            } else {
                panic!("expected setter");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_expr_anonymous() {
        let prog = parse("var x = class {};").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let init = vd.declarators[0].init.as_ref().unwrap();
            assert!(matches!(init.as_ref(), Expr::Class(_)));
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_class_expr_named() {
        let prog = parse("var x = class MyClass { foo() {} };").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let init = vd.declarators[0].init.as_ref().unwrap();
            if let Expr::Class(c) = init.as_ref() {
                assert_eq!(c.id.as_ref().unwrap().name, "MyClass");
                assert_eq!(c.body.body.len(), 1);
            } else {
                panic!("expected Class expr");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_class_multiple_members() {
        let prog = parse(
            "class Animal {
                constructor(name) { this.name = name; }
                speak() { return this.name; }
                static create(name) { return 1; }
            }",
        )
        .unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 3);
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_semicolons_in_body() {
        // Semicolons between class members should be silently skipped.
        let prog = parse("class Foo { ; foo() {} ; bar() {} ; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 2);
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_method_named_get() {
        // A method literally named "get" (not an accessor).
        let prog = parse("class Foo { get() { return 1; } }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 1);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(matches!(m.kind, crate::parser::ast::MethodKind::Method));
                if let crate::parser::ast::PropKey::Ident(ref id) = m.key {
                    assert_eq!(id.name, "get");
                } else {
                    panic!("expected Ident key");
                }
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_class_computed_method() {
        let prog = parse("class Foo { [Symbol.iterator]() {} }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::ClassDecl(c)) = &prog.body[0] {
            assert_eq!(c.body.body.len(), 1);
            if let crate::parser::ast::ClassMember::Method(m) = &c.body.body[0] {
                assert!(m.is_computed);
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    // ── template literal tests ───────────────────────────────────────────

    #[test]
    fn test_template_no_substitution() {
        let expr = parse_expr_stmt("`hello world`");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 1);
            assert!(tpl.expressions.is_empty());
            assert_eq!(tpl.quasis[0].raw, "hello world");
            assert_eq!(tpl.quasis[0].cooked.as_deref(), Some("hello world"));
            assert!(tpl.quasis[0].tail);
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_empty() {
        let expr = parse_expr_stmt("``");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 1);
            assert!(tpl.expressions.is_empty());
            assert_eq!(tpl.quasis[0].raw, "");
            assert!(tpl.quasis[0].tail);
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_single_substitution() {
        let expr = parse_expr_stmt("`hello ${name}!`");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 2);
            assert_eq!(tpl.expressions.len(), 1);
            assert_eq!(tpl.quasis[0].raw, "hello ");
            assert!(!tpl.quasis[0].tail);
            assert_eq!(tpl.quasis[1].raw, "!");
            assert!(tpl.quasis[1].tail);
            // The expression should be an identifier `name`.
            if let Expr::Ident(id) = &tpl.expressions[0] {
                assert_eq!(id.name, "name");
            } else {
                panic!("expected Ident expression");
            }
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_multiple_substitutions() {
        let expr = parse_expr_stmt("`${a} + ${b} = ${c}`");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 4);
            assert_eq!(tpl.expressions.len(), 3);
            assert_eq!(tpl.quasis[0].raw, "");
            assert_eq!(tpl.quasis[1].raw, " + ");
            assert_eq!(tpl.quasis[2].raw, " = ");
            assert_eq!(tpl.quasis[3].raw, "");
            assert!(!tpl.quasis[0].tail);
            assert!(!tpl.quasis[1].tail);
            assert!(!tpl.quasis[2].tail);
            assert!(tpl.quasis[3].tail);
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_expression_with_binary() {
        let expr = parse_expr_stmt("`result: ${1 + 2}`");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 2);
            assert_eq!(tpl.expressions.len(), 1);
            assert_eq!(tpl.quasis[0].raw, "result: ");
            assert_eq!(tpl.quasis[1].raw, "");
            assert!(matches!(&tpl.expressions[0], Expr::Binary(_)));
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_nested() {
        // Nested template: `outer ${`inner ${x}`} end`
        let expr = parse_expr_stmt("`outer ${`inner ${x}`} end`");
        if let Expr::Template(tpl) = &expr {
            assert_eq!(tpl.quasis.len(), 2);
            assert_eq!(tpl.expressions.len(), 1);
            assert_eq!(tpl.quasis[0].raw, "outer ");
            assert_eq!(tpl.quasis[1].raw, " end");
            // The inner expression should itself be a template.
            if let Expr::Template(inner) = &tpl.expressions[0] {
                assert_eq!(inner.quasis.len(), 2);
                assert_eq!(inner.expressions.len(), 1);
                assert_eq!(inner.quasis[0].raw, "inner ");
                assert_eq!(inner.quasis[1].raw, "");
            } else {
                panic!("expected nested Template");
            }
        } else {
            panic!("expected Template, got {expr:?}");
        }
    }

    #[test]
    fn test_template_in_var_decl() {
        let prog = parse("var x = `value: ${42}`;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let init = vd.declarators[0].init.as_ref().unwrap();
            assert!(matches!(init.as_ref(), Expr::Template(_)));
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── RegExp literal tests ─────────────────────────────────────────────

    #[test]
    fn test_regexp_simple() {
        let expr = parse_expr_stmt("/abc/;");
        if let Expr::Regexp(re) = &expr {
            assert_eq!(re.pattern, "abc");
            assert_eq!(re.flags, "");
        } else {
            panic!("expected Regexp, got {expr:?}");
        }
    }

    #[test]
    fn test_regexp_with_flags() {
        let expr = parse_expr_stmt("/abc/gi;");
        if let Expr::Regexp(re) = &expr {
            assert_eq!(re.pattern, "abc");
            assert_eq!(re.flags, "gi");
        } else {
            panic!("expected Regexp, got {expr:?}");
        }
    }

    #[test]
    fn test_regexp_in_var_decl() {
        let prog = parse("var re = /test/;").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let init = vd.declarators[0].init.as_ref().unwrap();
            if let Expr::Regexp(re) = init.as_ref() {
                assert_eq!(re.pattern, "test");
                assert_eq!(re.flags, "");
            } else {
                panic!("expected Regexp, got {init:?}");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_regexp_with_char_class() {
        let expr = parse_expr_stmt("/[a-z]+/g;");
        if let Expr::Regexp(re) = &expr {
            assert_eq!(re.pattern, "[a-z]+");
            assert_eq!(re.flags, "g");
        } else {
            panic!("expected Regexp, got {expr:?}");
        }
    }

    #[test]
    fn test_regexp_with_escape() {
        let expr = parse_expr_stmt(r"/foo\/bar/;");
        if let Expr::Regexp(re) = &expr {
            assert_eq!(re.pattern, r"foo\/bar");
            assert_eq!(re.flags, "");
        } else {
            panic!("expected Regexp, got {expr:?}");
        }
    }

    #[test]
    fn test_regexp_slash_in_char_class() {
        let expr = parse_expr_stmt("/[/]/;");
        if let Expr::Regexp(re) = &expr {
            assert_eq!(re.pattern, "[/]");
            assert_eq!(re.flags, "");
        } else {
            panic!("expected Regexp, got {expr:?}");
        }
    }

    #[test]
    fn test_regexp_after_assignment() {
        // `/` after `=` should be regexp, not division
        let prog = parse("var x = /pattern/i;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let init = vd.declarators[0].init.as_ref().unwrap();
            assert!(matches!(init.as_ref(), Expr::Regexp(_)));
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── Array destructuring tests ────────────────────────────────────────

    #[test]
    fn test_array_destructuring_simple() {
        // let [a, b] = arr;
        let prog = parse("let [a, b] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            assert_eq!(vd.declarators.len(), 1);
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 2);
                assert!(matches!(ap.elements[0], Some(Pat::Ident(_))));
                assert!(matches!(ap.elements[1], Some(Pat::Ident(_))));
            } else {
                panic!("expected ArrayPat, got {:?}", d.id);
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_destructuring_elision() {
        // let [a, , b] = arr;
        let prog = parse("let [a, , b] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 3);
                assert!(matches!(ap.elements[0], Some(Pat::Ident(_))));
                assert!(ap.elements[1].is_none(), "expected elision (None)");
                assert!(matches!(ap.elements[2], Some(Pat::Ident(_))));
            } else {
                panic!("expected ArrayPat, got {:?}", d.id);
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_destructuring_rest() {
        // let [a, ...rest] = arr;
        let prog = parse("let [a, ...rest] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 2);
                assert!(matches!(ap.elements[0], Some(Pat::Ident(_))));
                assert!(matches!(ap.elements[1], Some(Pat::Rest(_))));
            } else {
                panic!("expected ArrayPat, got {:?}", d.id);
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_destructuring_default() {
        // let [a = 1] = arr;
        let prog = parse("let [a = 1] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 1);
                if let Some(Pat::Assign(assign)) = &ap.elements[0] {
                    assert!(matches!(assign.left.as_ref(), Pat::Ident(_)));
                } else {
                    panic!("expected AssignPat element");
                }
            } else {
                panic!("expected ArrayPat, got {:?}", d.id);
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_destructuring_nested() {
        // let [a, [b, c]] = arr;
        let prog = parse("let [a, [b, c]] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 2);
                assert!(matches!(ap.elements[0], Some(Pat::Ident(_))));
                assert!(matches!(ap.elements[1], Some(Pat::Array(_))));
            } else {
                panic!("expected ArrayPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_destructuring_empty() {
        // let [] = arr;
        let prog = parse("let [] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert!(ap.elements.is_empty());
            } else {
                panic!("expected ArrayPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── Object destructuring tests ───────────────────────────────────────

    #[test]
    fn test_object_destructuring_simple() {
        // let {a, b} = obj;
        let prog = parse("let {a, b} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 2);
                assert!(matches!(op.properties[0], ObjectPatProp::Assign(_)));
                assert!(matches!(op.properties[1], ObjectPatProp::Assign(_)));
            } else {
                panic!("expected ObjectPat, got {:?}", d.id);
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_rename() {
        // let {a: x} = obj;
        let prog = parse("let {a: x} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 1);
                if let ObjectPatProp::KeyValue(kv) = &op.properties[0] {
                    if let PropKey::Ident(key) = &kv.key {
                        assert_eq!(key.name, "a");
                    } else {
                        panic!("expected ident key");
                    }
                    assert!(matches!(&kv.value, Pat::Ident(id) if id.name == "x"));
                } else {
                    panic!("expected KeyValue");
                }
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_default() {
        // let {a = 1} = obj;
        let prog = parse("let {a = 1} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 1);
                if let ObjectPatProp::Assign(ap) = &op.properties[0] {
                    assert_eq!(ap.key.name, "a");
                    assert!(ap.value.is_some());
                } else {
                    panic!("expected Assign prop");
                }
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_rest() {
        // let {...rest} = obj;
        let prog = parse("let {...rest} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 1);
                assert!(matches!(op.properties[0], ObjectPatProp::Rest(_)));
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_nested() {
        // let {a: {b}} = obj;
        let prog = parse("let {a: {b}} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 1);
                if let ObjectPatProp::KeyValue(kv) = &op.properties[0] {
                    assert!(matches!(&kv.value, Pat::Object(_)));
                } else {
                    panic!("expected KeyValue");
                }
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_empty() {
        // let {} = obj;
        let prog = parse("let {} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert!(op.properties.is_empty());
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_destructuring_rename_with_default() {
        // let {a: x = 5} = obj;
        let prog = parse("let {a: x = 5} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 1);
                if let ObjectPatProp::KeyValue(kv) = &op.properties[0] {
                    assert!(matches!(&kv.value, Pat::Assign(_)));
                } else {
                    panic!("expected KeyValue");
                }
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── Destructuring in function parameters ─────────────────────────────

    #[test]
    fn test_function_param_array_destructuring() {
        // function f([a, b]) {}
        let prog = parse("function f([a, b]) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert_eq!(fd.params.len(), 1);
            assert!(matches!(&fd.params[0].pat, Pat::Array(_)));
        } else {
            panic!("expected FnDecl");
        }
    }

    #[test]
    fn test_function_param_object_destructuring() {
        // function f({a, b}) {}
        let prog = parse("function f({a, b}) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert_eq!(fd.params.len(), 1);
            assert!(matches!(&fd.params[0].pat, Pat::Object(_)));
        } else {
            panic!("expected FnDecl");
        }
    }

    #[test]
    fn test_function_param_default_value() {
        // function f(a = 1) {}
        let prog = parse("function f(a = 1) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert_eq!(fd.params.len(), 1);
            assert!(matches!(&fd.params[0].pat, Pat::Ident(_)));
            assert!(fd.params[0].default.is_some());
        } else {
            panic!("expected FnDecl");
        }
    }

    #[test]
    fn test_function_param_rest() {
        // function f(a, ...rest) {}
        let prog = parse("function f(a, ...rest) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert_eq!(fd.params.len(), 2);
            assert!(matches!(&fd.params[0].pat, Pat::Ident(_)));
            assert!(matches!(&fd.params[1].pat, Pat::Rest(_)));
        } else {
            panic!("expected FnDecl");
        }
    }

    #[test]
    fn test_function_param_destructuring_with_default() {
        // function f({a = 1, b} = {}) {}
        let prog = parse("function f({a = 1, b}) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert_eq!(fd.params.len(), 1);
            if let Pat::Object(op) = &fd.params[0].pat {
                assert_eq!(op.properties.len(), 2);
            } else {
                panic!("expected ObjectPat param");
            }
        } else {
            panic!("expected FnDecl");
        }
    }

    // ── Mixed / complex destructuring tests ──────────────────────────────

    #[test]
    fn test_const_array_destructuring() {
        // const [x, y, z] = [1, 2, 3];
        let prog = parse("const [x, y, z] = [1, 2, 3];").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            assert_eq!(vd.kind, VarKind::Const);
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 3);
            } else {
                panic!("expected ArrayPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_var_object_destructuring() {
        // var {x, y} = obj;
        let prog = parse("var {x, y} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            assert_eq!(vd.kind, VarKind::Var);
            let d = &vd.declarators[0];
            assert!(matches!(&d.id, Pat::Object(_)));
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_nested_mixed_destructuring() {
        // let {a: [b, c]} = obj;
        let prog = parse("let {a: [b, c]} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                if let ObjectPatProp::KeyValue(kv) = &op.properties[0] {
                    assert!(matches!(&kv.value, Pat::Array(_)));
                } else {
                    panic!("expected KeyValue");
                }
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_array_with_nested_object() {
        // let [{a}, {b}] = arr;
        let prog = parse("let [{a}, {b}] = arr;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Array(ap) = &d.id {
                assert_eq!(ap.elements.len(), 2);
                assert!(matches!(ap.elements[0], Some(Pat::Object(_))));
                assert!(matches!(ap.elements[1], Some(Pat::Object(_))));
            } else {
                panic!("expected ArrayPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_complex_destructuring() {
        // let {a, b: [c, ...d], ...e} = obj;
        let prog = parse("let {a, b: [c, ...d], ...e} = obj;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            let d = &vd.declarators[0];
            if let Pat::Object(op) = &d.id {
                assert_eq!(op.properties.len(), 3);
                // {a} — shorthand
                assert!(matches!(&op.properties[0], ObjectPatProp::Assign(_)));
                // {b: [c, ...d]} — key-value with nested array
                if let ObjectPatProp::KeyValue(kv) = &op.properties[1] {
                    assert!(matches!(&kv.value, Pat::Array(_)));
                } else {
                    panic!("expected KeyValue for b");
                }
                // {...e} — rest
                assert!(matches!(&op.properties[2], ObjectPatProp::Rest(_)));
            } else {
                panic!("expected ObjectPat");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_for_of_array_destructuring() {
        // for (let [a, b] of arr) {}
        let prog = parse("for (let [a, b] of arr) {}").unwrap();
        assert!(matches!(&prog.body[0], ProgramItem::Stmt(Stmt::ForOf(_))));
    }

    #[test]
    fn test_for_in_object_destructuring() {
        // for (let {a, b} in obj) {}
        let prog = parse("for (let {a, b} in obj) {}").unwrap();
        assert!(matches!(&prog.body[0], ProgramItem::Stmt(Stmt::ForIn(_))));
    }

    #[test]
    fn test_catch_array_destructuring() {
        // try {} catch ([a, b]) {}
        let prog = parse("try {} catch ([a, b]) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::Try(ts)) = &prog.body[0] {
            let handler = ts.handler.as_ref().unwrap();
            assert!(matches!(&handler.param, Some(Pat::Array(_))));
        } else {
            panic!("expected TryStmt");
        }
    }

    #[test]
    fn test_catch_object_destructuring() {
        // try {} catch ({message}) {}
        let prog = parse("try {} catch ({message}) {}").unwrap();
        if let ProgramItem::Stmt(Stmt::Try(ts)) = &prog.body[0] {
            let handler = ts.handler.as_ref().unwrap();
            assert!(matches!(&handler.param, Some(Pat::Object(_))));
        } else {
            panic!("expected TryStmt");
        }
    }

    // ── spread in call arguments ─────────────────────────────────────────

    #[test]
    fn test_spread_in_call_single() {
        // f(...args)
        let prog = parse("f(...args);").unwrap();
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::Call(c) = es.expr.as_ref() {
                assert_eq!(c.arguments.len(), 1);
                assert!(matches!(&c.arguments[0], Expr::Spread(_)));
            } else {
                panic!("expected Call expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_spread_in_call_with_leading_arg() {
        // f(a, ...args)
        let prog = parse("f(a, ...args);").unwrap();
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::Call(c) = es.expr.as_ref() {
                assert_eq!(c.arguments.len(), 2);
                assert!(matches!(&c.arguments[0], Expr::Ident(_)));
                assert!(matches!(&c.arguments[1], Expr::Spread(_)));
            } else {
                panic!("expected Call expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_spread_in_new_call() {
        // new Foo(...args)
        let prog = parse("new Foo(...args);").unwrap();
        if let ProgramItem::Stmt(Stmt::Expr(es)) = &prog.body[0] {
            if let Expr::New(n) = es.expr.as_ref() {
                assert_eq!(n.arguments.len(), 1);
                assert!(matches!(&n.arguments[0], Expr::Spread(_)));
            } else {
                panic!("expected New expr, got {:?}", es.expr);
            }
        } else {
            panic!("expected ExprStmt");
        }
    }

    #[test]
    fn test_spread_in_object_literal() {
        // var o = {...other}
        let prog = parse("var o = {...other};").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 1);
                    assert!(matches!(&obj.properties[0], ObjectProp::Spread(_)));
                } else {
                    panic!("expected Object expr");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_spread_in_object_literal_with_leading_prop() {
        // var o = {a: 1, ...other}
        let prog = parse("var o = {a: 1, ...other};").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Object(obj) = init.as_ref() {
                    assert_eq!(obj.properties.len(), 2);
                    assert!(matches!(&obj.properties[0], ObjectProp::Prop(_)));
                    assert!(matches!(&obj.properties[1], ObjectProp::Spread(_)));
                } else {
                    panic!("expected Object expr");
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    // ── Import declaration tests ─────────────────────────────────────────

    use crate::parser::ast::{
        ExportDefaultDecl, ExportDefaultExpr, ExportNamedDecl, ExportSpecifier, ImportDecl,
        ImportDefaultSpecifier, ImportNamedSpecifier, ImportNamespaceSpecifier, ImportSpecifier,
        ModuleDecl, ModuleExportName, SourceType,
    };

    #[test]
    fn test_import_default() {
        let prog = parse("import x from \"module\";").unwrap();
        assert_eq!(prog.source_type, SourceType::Module);
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            assert!(matches!(&decl.specifiers[0], ImportSpecifier::Default(_)));
            if let ImportSpecifier::Default(d) = &decl.specifiers[0] {
                assert_eq!(d.local.name, "x");
            }
            assert_eq!(decl.source.value, "\"module\"");
        } else {
            panic!("expected ImportDecl, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_import_named_single() {
        let prog = parse("import { a } from \"mod\";").unwrap();
        assert_eq!(prog.source_type, SourceType::Module);
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            if let ImportSpecifier::Named(n) = &decl.specifiers[0] {
                assert_eq!(n.local.name, "a");
                if let ModuleExportName::Ident(id) = &n.imported {
                    assert_eq!(id.name, "a");
                } else {
                    panic!("expected Ident imported name");
                }
            } else {
                panic!("expected Named specifier");
            }
            assert_eq!(decl.source.value, "\"mod\"");
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_named_multiple() {
        let prog = parse("import { a, b } from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
            assert!(matches!(&decl.specifiers[0], ImportSpecifier::Named(_)));
            assert!(matches!(&decl.specifiers[1], ImportSpecifier::Named(_)));
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_named_aliased() {
        let prog = parse("import { a as alias } from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            if let ImportSpecifier::Named(n) = &decl.specifiers[0] {
                if let ModuleExportName::Ident(id) = &n.imported {
                    assert_eq!(id.name, "a");
                } else {
                    panic!("expected Ident imported");
                }
                assert_eq!(n.local.name, "alias");
            } else {
                panic!("expected Named specifier");
            }
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_namespace() {
        let prog = parse("import * as mod_ns from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            if let ImportSpecifier::Namespace(ns) = &decl.specifiers[0] {
                assert_eq!(ns.local.name, "mod_ns");
            } else {
                panic!("expected Namespace specifier");
            }
            assert_eq!(decl.source.value, "\"mod\"");
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_side_effect() {
        let prog = parse("import \"side-effect\";").unwrap();
        assert_eq!(prog.source_type, SourceType::Module);
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert!(decl.specifiers.is_empty());
            assert_eq!(decl.source.value, "\"side-effect\"");
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_default_and_named() {
        let prog = parse("import x, { a, b } from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 3);
            assert!(matches!(&decl.specifiers[0], ImportSpecifier::Default(_)));
            assert!(matches!(&decl.specifiers[1], ImportSpecifier::Named(_)));
            assert!(matches!(&decl.specifiers[2], ImportSpecifier::Named(_)));
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_default_and_namespace() {
        let prog = parse("import x, * as ns from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
            assert!(matches!(&decl.specifiers[0], ImportSpecifier::Default(_)));
            assert!(matches!(&decl.specifiers[1], ImportSpecifier::Namespace(_)));
        } else {
            panic!("expected ImportDecl");
        }
    }

    #[test]
    fn test_import_named_trailing_comma() {
        let prog = parse("import { a, b, } from \"mod\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::Import(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
        } else {
            panic!("expected ImportDecl");
        }
    }

    // ── Export declaration tests ──────────────────────────────────────────

    #[test]
    fn test_export_default_expr() {
        let prog = parse("export default 42;").unwrap();
        assert_eq!(prog.source_type, SourceType::Module);
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(decl)) = &prog.body[0] {
            assert!(matches!(&decl.declaration, ExportDefaultExpr::Expr(_)));
        } else {
            panic!("expected ExportDefault, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_export_default_function() {
        let prog = parse("export default function foo() {}").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(decl)) = &prog.body[0] {
            if let ExportDefaultExpr::Fn(fd) = &decl.declaration {
                assert_eq!(fd.id.as_ref().unwrap().name, "foo");
            } else {
                panic!("expected Fn default export");
            }
        } else {
            panic!("expected ExportDefault");
        }
    }

    #[test]
    fn test_export_default_anonymous_function() {
        let prog = parse("export default function() {}").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(decl)) = &prog.body[0] {
            if let ExportDefaultExpr::Fn(fd) = &decl.declaration {
                assert!(fd.id.is_none());
            } else {
                panic!("expected Fn default export");
            }
        } else {
            panic!("expected ExportDefault");
        }
    }

    #[test]
    fn test_export_default_class() {
        let prog = parse("export default class Foo {}").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(decl)) = &prog.body[0] {
            if let ExportDefaultExpr::Class(cd) = &decl.declaration {
                assert_eq!(cd.id.as_ref().unwrap().name, "Foo");
            } else {
                panic!("expected Class default export");
            }
        } else {
            panic!("expected ExportDefault");
        }
    }

    #[test]
    fn test_export_function_decl() {
        let prog = parse("export function add(a, b) { return a + b; }").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert!(decl.specifiers.is_empty());
            assert!(decl.source.is_none());
            if let Some(stmt) = &decl.declaration {
                assert!(matches!(stmt.as_ref(), Stmt::FnDecl(_)));
            } else {
                panic!("expected declaration");
            }
        } else {
            panic!("expected ExportNamed, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_export_class_decl() {
        let prog = parse("export class Foo {}").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert!(decl.declaration.is_some());
            if let Some(stmt) = &decl.declaration {
                assert!(matches!(stmt.as_ref(), Stmt::ClassDecl(_)));
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_const() {
        let prog = parse("export const x = 1;").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            if let Some(stmt) = &decl.declaration {
                if let Stmt::VarDecl(vd) = stmt.as_ref() {
                    assert_eq!(vd.kind, VarKind::Const);
                } else {
                    panic!("expected VarDecl");
                }
            } else {
                panic!("expected declaration");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_let() {
        let prog = parse("export let y = 2;").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            if let Some(stmt) = &decl.declaration {
                if let Stmt::VarDecl(vd) = stmt.as_ref() {
                    assert_eq!(vd.kind, VarKind::Let);
                } else {
                    panic!("expected VarDecl");
                }
            } else {
                panic!("expected declaration");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_var() {
        let prog = parse("export var z = 3;").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            if let Some(stmt) = &decl.declaration {
                if let Stmt::VarDecl(vd) = stmt.as_ref() {
                    assert_eq!(vd.kind, VarKind::Var);
                } else {
                    panic!("expected VarDecl");
                }
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_named_specifiers() {
        let prog = parse("export { name1, name2 };").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
            assert!(decl.source.is_none());
            assert!(decl.declaration.is_none());
            if let ModuleExportName::Ident(id) = &decl.specifiers[0].local {
                assert_eq!(id.name, "name1");
            } else {
                panic!("expected Ident");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_named_aliased() {
        let prog = parse("export { name as alias };").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            if let ModuleExportName::Ident(local) = &decl.specifiers[0].local {
                assert_eq!(local.name, "name");
            }
            if let ModuleExportName::Ident(exp) = &decl.specifiers[0].exported {
                assert_eq!(exp.name, "alias");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_named_as_default() {
        let prog = parse("export { foo as default };").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 1);
            if let ModuleExportName::Ident(exp) = &decl.specifiers[0].exported {
                assert_eq!(exp.name, "default");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_named_re_export() {
        let prog = parse("export { a, b } from \"other\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
            assert_eq!(decl.source.as_ref().unwrap().value, "\"other\"");
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_all() {
        let prog = parse("export * from \"other\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportAll(decl)) = &prog.body[0] {
            assert!(decl.exported.is_none());
            assert_eq!(decl.source.value, "\"other\"");
        } else {
            panic!("expected ExportAll, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_export_all_as_namespace() {
        let prog = parse("export * as ns from \"other\";").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportAll(decl)) = &prog.body[0] {
            if let Some(ModuleExportName::Ident(id)) = &decl.exported {
                assert_eq!(id.name, "ns");
            } else {
                panic!("expected namespace alias");
            }
            assert_eq!(decl.source.value, "\"other\"");
        } else {
            panic!("expected ExportAll");
        }
    }

    #[test]
    fn test_export_named_trailing_comma() {
        let prog = parse("export { a, b, };").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            assert_eq!(decl.specifiers.len(), 2);
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_mixed_imports_and_statements() {
        let prog = parse("import x from \"mod\";\nvar a = 1;\nexport const b = 2;").unwrap();
        assert_eq!(prog.source_type, SourceType::Module);
        assert_eq!(prog.body.len(), 3);
        assert!(matches!(
            &prog.body[0],
            ProgramItem::ModuleDecl(ModuleDecl::Import(_))
        ));
        assert!(matches!(&prog.body[1], ProgramItem::Stmt(Stmt::VarDecl(_))));
        assert!(matches!(
            &prog.body[2],
            ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(_))
        ));
    }

    #[test]
    fn test_script_has_no_module_type() {
        let prog = parse("var x = 1;").unwrap();
        assert_eq!(prog.source_type, SourceType::Script);
    }

    // ── Async / await tests ──────────────────────────────────────────────

    #[test]
    fn test_async_function_declaration() {
        let prog = parse("async function fetchData() { return 1; }").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert!(fd.is_async);
            assert!(!fd.is_generator);
            assert_eq!(fd.id.as_ref().unwrap().name, "fetchData");
        } else {
            panic!("expected async FnDecl, got {:?}", prog.body[0]);
        }
    }

    #[test]
    fn test_async_function_expression() {
        let prog = parse("var f = async function() { return 1; };").unwrap();
        assert_eq!(prog.body.len(), 1);
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(init) = &vd.declarators[0].init {
                if let Expr::Fn(fe) = init.as_ref() {
                    assert!(fe.is_async);
                    assert!(fe.id.is_none());
                } else {
                    panic!("expected Fn expr, got {:?}", init);
                }
            } else {
                panic!("expected init");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_async_named_function_expression() {
        let prog = parse("var f = async function myFn() { };").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Fn(fe)) = vd.declarators[0].init.as_deref() {
                assert!(fe.is_async);
                assert_eq!(fe.id.as_ref().unwrap().name, "myFn");
            } else {
                panic!("expected async function expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_async_arrow_no_params() {
        let prog = parse("var f = async () => 42;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Arrow(arrow)) = vd.declarators[0].init.as_deref() {
                assert!(arrow.is_async);
                assert!(arrow.params.is_empty());
            } else {
                panic!("expected async arrow expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_async_arrow_single_param() {
        let prog = parse("var f = async x => x + 1;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Arrow(arrow)) = vd.declarators[0].init.as_deref() {
                assert!(arrow.is_async);
                assert_eq!(arrow.params.len(), 1);
                if let Pat::Ident(ref id) = arrow.params[0].pat {
                    assert_eq!(id.name, "x");
                } else {
                    panic!("expected ident param");
                }
            } else {
                panic!("expected async arrow expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_async_arrow_multi_params() {
        let prog = parse("var f = async (a, b) => a + b;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Arrow(arrow)) = vd.declarators[0].init.as_deref() {
                assert!(arrow.is_async);
                assert_eq!(arrow.params.len(), 2);
            } else {
                panic!("expected async arrow expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_async_arrow_block_body() {
        let prog = parse("var f = async () => { return 1; };").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Arrow(arrow)) = vd.declarators[0].init.as_deref() {
                assert!(arrow.is_async);
                assert!(matches!(arrow.body, ArrowBody::Block(_)));
            } else {
                panic!("expected async arrow expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_await_expression() {
        let prog = parse("async function f() { var x = await promise; }").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert!(fd.is_async);
            // The body should contain a var declaration whose init is an Await expr.
            if let Stmt::VarDecl(vd) = &fd.body.body[0] {
                if let Some(Expr::Await(await_expr)) = vd.declarators[0].init.as_deref() {
                    if let Expr::Ident(ref id) = *await_expr.argument {
                        assert_eq!(id.name, "promise");
                    } else {
                        panic!("expected ident in await argument");
                    }
                } else {
                    panic!("expected Await expr in var init");
                }
            } else {
                panic!("expected VarDecl in function body");
            }
        } else {
            panic!("expected async FnDecl");
        }
    }

    #[test]
    fn test_await_call_expression() {
        let prog = parse("async function f() { await fetch(); }").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert!(fd.is_async);
            if let Stmt::Expr(es) = &fd.body.body[0] {
                assert!(matches!(*es.expr, Expr::Await(_)));
            } else {
                panic!("expected ExprStmt");
            }
        } else {
            panic!("expected async FnDecl");
        }
    }

    #[test]
    fn test_async_class_method() {
        let prog = parse("class Foo { async fetchData() { return 1; } }").unwrap();
        if let ProgramItem::Stmt(Stmt::ClassDecl(cd)) = &prog.body[0] {
            if let ClassMember::Method(md) = &cd.body.body[0] {
                assert!(md.value.is_async);
                assert_eq!(md.kind, MethodKind::Method);
                if let PropKey::Ident(ref id) = md.key {
                    assert_eq!(id.name, "fetchData");
                } else {
                    panic!("expected ident key");
                }
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_async_static_class_method() {
        let prog = parse("class Foo { static async bar() { } }").unwrap();
        if let ProgramItem::Stmt(Stmt::ClassDecl(cd)) = &prog.body[0] {
            if let ClassMember::Method(md) = &cd.body.body[0] {
                assert!(md.is_static);
                assert!(md.value.is_async);
            } else {
                panic!("expected Method");
            }
        } else {
            panic!("expected ClassDecl");
        }
    }

    #[test]
    fn test_async_object_method() {
        let prog = parse("var o = { async fetch() { return 1; } };").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Object(obj)) = vd.declarators[0].init.as_deref() {
                if let ObjectProp::Prop(p) = &obj.properties[0] {
                    if let PropValue::Method(ref fe) = p.value {
                        assert!(fe.is_async);
                    } else {
                        panic!("expected Method PropValue, got {:?}", p.value);
                    }
                    if let PropKey::Ident(ref id) = p.key {
                        assert_eq!(id.name, "fetch");
                    }
                } else {
                    panic!("expected Prop");
                }
            } else {
                panic!("expected Object expr");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_object_method_shorthand() {
        let prog = parse("var o = { greet() { return 1; } };").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Object(obj)) = vd.declarators[0].init.as_deref() {
                if let ObjectProp::Prop(p) = &obj.properties[0] {
                    assert!(matches!(p.value, PropValue::Method(_)));
                    if let PropKey::Ident(ref id) = p.key {
                        assert_eq!(id.name, "greet");
                    }
                } else {
                    panic!("expected Prop");
                }
            } else {
                panic!("expected Object expr");
            }
        } else {
            panic!("expected VarDecl");
        }
    }

    #[test]
    fn test_export_async_function() {
        let prog = parse("export async function fetchData() { }").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(decl)) = &prog.body[0] {
            if let Some(decl_stmt) = &decl.declaration {
                if let Stmt::FnDecl(fd) = decl_stmt.as_ref() {
                    assert!(fd.is_async);
                    assert_eq!(fd.id.as_ref().unwrap().name, "fetchData");
                } else {
                    panic!("expected FnDecl");
                }
            } else {
                panic!("expected declaration");
            }
        } else {
            panic!("expected ExportNamed");
        }
    }

    #[test]
    fn test_export_default_async_function() {
        let prog = parse("export default async function fetchData() { }").unwrap();
        if let ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(decl)) = &prog.body[0] {
            if let ExportDefaultExpr::Fn(fd) = &decl.declaration {
                assert!(fd.is_async);
            } else {
                panic!("expected Fn in export default");
            }
        } else {
            panic!("expected ExportDefault");
        }
    }

    #[test]
    fn test_non_async_function_still_works() {
        let prog = parse("function f() { return 1; }").unwrap();
        if let ProgramItem::Stmt(Stmt::FnDecl(fd)) = &prog.body[0] {
            assert!(!fd.is_async);
        } else {
            panic!("expected FnDecl");
        }
    }

    #[test]
    fn test_non_async_arrow_still_works() {
        let prog = parse("var f = () => 42;").unwrap();
        if let ProgramItem::Stmt(Stmt::VarDecl(vd)) = &prog.body[0] {
            if let Some(Expr::Arrow(arrow)) = vd.declarators[0].init.as_deref() {
                assert!(!arrow.is_async);
            } else {
                panic!("expected arrow expression");
            }
        } else {
            panic!("expected VarDecl");
        }
    }
}

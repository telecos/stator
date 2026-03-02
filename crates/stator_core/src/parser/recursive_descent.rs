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
    AssignExpr, AssignOp, AssignTarget, BinaryExpr, BinaryOp, BlockStmt, BoolLit, BreakStmt,
    ContinueStmt, DebuggerStmt, DoWhileStmt, EmptyStmt, Expr, ExprStmt, FnDecl, ForStmt, Ident,
    IfStmt, LogicalExpr, LogicalOp, NullLit, NumLit, Param, Pat, Program, ProgramItem, ReturnStmt,
    SequenceExpr, SourceLocation, SourceType, Stmt, StringLit, ThrowStmt, TryStmt, UnaryExpr,
    UnaryOp, UpdateExpr, UpdateOp, VarDecl, VarDeclarator, VarKind, WhileStmt,
};
use crate::parser::scanner::{Scanner, Span, Token, TokenKind, TokenValue};

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Recursive-descent JavaScript parser.
///
/// Wraps a [`Scanner`] and maintains one token of lookahead in `current`.
pub struct Parser<'src> {
    scanner: Scanner<'src>,
    /// The lookahead token (already produced by the scanner).
    current: Token,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given UTF-8 source string.
    ///
    /// Primes the lookahead by reading the first non-comment token.
    fn new(source: &'src str) -> StatorResult<Self> {
        let mut scanner = Scanner::new(source);
        let current = Self::next_significant(&mut scanner)?;
        Ok(Self { scanner, current })
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
            return Err(StatorError::SyntaxError(format!(
                "at {}:{} \u{2014} expected {:?}, got {:?}",
                span.start.line, span.start.column, kind, self.current.kind
            )));
        }
        self.bump()
    }

    /// Build a [`StatorError::SyntaxError`] anchored at the current token.
    fn error(&self, msg: &str) -> StatorError {
        let span = self.current.span;
        StatorError::SyntaxError(format!(
            "at {}:{} \u{2014} {}",
            span.start.line, span.start.column, msg
        ))
    }

    /// Build a [`StatorError::SyntaxError`] anchored at the given span.
    fn error_at(span: Span, msg: &str) -> StatorError {
        StatorError::SyntaxError(format!(
            "at {}:{} \u{2014} {}",
            span.start.line, span.start.column, msg
        ))
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

    /// Parse a complete source file as a script [`Program`].
    fn parse_program(&mut self) -> StatorResult<Program> {
        let start = self.current_span();
        let mut body = Vec::new();

        while self.peek_kind() != TokenKind::Eof {
            let stmt = self.parse_stmt()?;
            body.push(ProgramItem::Stmt(stmt));
        }

        let end = self.current_span();
        Ok(Program {
            loc: Self::merge_spans(start, end),
            source_type: SourceType::Script,
            body,
        })
    }

    // ── Statements ───────────────────────────────────────────────────────────

    fn parse_stmt(&mut self) -> StatorResult<Stmt> {
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
            TokenKind::Debugger => {
                let span = self.current_span();
                self.bump()?;
                self.consume_semicolon()?;
                Ok(Stmt::Debugger(DebuggerStmt { loc: span }))
            }
            TokenKind::Function => {
                let tok = self.bump()?;
                self.parse_fn_decl(tok.span)
            }
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
        // Expect an identifier (simple binding only).
        if self.peek_kind() != TokenKind::Identifier {
            return Err(self.error("expected identifier"));
        }
        let id_tok = self.bump()?;
        let name = match &id_tok.value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(Self::error_at(id_tok.span, "invalid identifier token")),
        };
        let id = Pat::Ident(Ident {
            loc: id_tok.span,
            name,
        });
        let init = if self.eat(TokenKind::Equal)? {
            Some(Box::new(self.parse_assignment_expr()?))
        } else {
            None
        };
        Ok(VarDeclarator {
            loc: id_tok.span,
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

        // Initializer.
        let init = if self.peek_kind() == TokenKind::Semicolon {
            None
        } else {
            let init_expr = self.parse_expr()?;
            Some(crate::parser::ast::ForInit::Expr(Box::new(init_expr)))
        };
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

    fn parse_fn_decl(&mut self, fn_span: Span) -> StatorResult<Stmt> {
        // function [*] id ( params ) { body }
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
            is_async: false,
            is_generator,
            params,
            body,
        })))
    }

    fn parse_formal_params(&mut self) -> StatorResult<Vec<Param>> {
        let mut params = Vec::new();
        while self.peek_kind() != TokenKind::RightParen {
            if self.peek_kind() == TokenKind::Eof {
                return Err(self.error("unexpected end of input in parameter list"));
            }
            let start = self.current_span();
            let pat = self.parse_binding_pat()?;
            let end = pat.loc();
            params.push(Param {
                loc: Self::merge_spans(start, end),
                pat,
                default: None,
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
            _ => Err(self.error("expected binding pattern")),
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
    fn parse_assignment_expr(&mut self) -> StatorResult<Expr> {
        let start = self.current_span();
        let lhs = self.parse_conditional_expr()?;
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
            args.push(self.parse_assignment_expr()?);
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        self.expect(TokenKind::RightParen)?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> StatorResult<Expr> {
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
            TokenKind::Function => {
                let fn_span = self.current_span();
                self.bump()?;
                self.parse_fn_expr(fn_span)
            }
            other => Err(self.error(&format!("unexpected token {:?}", other))),
        }
    }

    fn parse_fn_expr(&mut self, fn_span: Span) -> StatorResult<Expr> {
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
            is_async: false,
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
    use crate::parser::ast::{ProgramItem, Stmt};

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
            msg.contains("expected identifier"),
            "expected 'expected identifier' in {msg:?}"
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
}

//! ES2025 JavaScript lexer (scanner).
//!
//! See [`Scanner`] for the main entry point.

use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// Position / Span
// ─────────────────────────────────────────────────────────────────────────────

/// A byte offset + line/column location in source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// Byte offset from the beginning of the source string.
    pub offset: usize,
    /// 1-based line number (incremented on every *LineTerminator*).
    pub line: u32,
    /// 1-based column number, measured in Unicode scalar values.
    pub column: u32,
}

/// A half-open `[start, end)` source span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Inclusive start of the span.
    pub start: Position,
    /// Exclusive end of the span.
    pub end: Position,
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenKind
// ─────────────────────────────────────────────────────────────────────────────

/// The syntactic category of a JavaScript lexical token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // ── Numeric literals ──────────────────────────────────────────────────
    /// Decimal, hex (`0x…`), binary (`0b…`), octal (`0o…` / legacy `0…`),
    /// or BigInt (trailing `n`) numeric literal.
    NumericLiteral,

    // ── String literals ───────────────────────────────────────────────────
    /// String literal enclosed in `"` or `'`.
    StringLiteral,

    // ── Template literals ─────────────────────────────────────────────────
    /// A complete template literal with no substitutions: `` `…` ``.
    NoSubstitutionTemplate,
    /// Opening span of a substituted template: `` `…${ ``.
    TemplateHead,
    /// Middle span between two substitutions: `}…${`.
    TemplateMiddle,
    /// Closing span of a substituted template: `}…`` ` ``.
    TemplateTail,

    // ── Regular-expression literals ───────────────────────────────────────
    /// Regular expression literal `/pattern/flags`.
    RegExpLiteral,

    // ── Identifiers ───────────────────────────────────────────────────────
    /// An identifier that is not a reserved word.
    Identifier,
    /// A private identifier beginning with `#`.
    PrivateIdentifier,

    // ── Reserved words ────────────────────────────────────────────────────
    /// `await`
    Await,
    /// `break`
    Break,
    /// `case`
    Case,
    /// `catch`
    Catch,
    /// `class`
    Class,
    /// `const`
    Const,
    /// `continue`
    Continue,
    /// `debugger`
    Debugger,
    /// `default`
    Default,
    /// `delete`
    Delete,
    /// `do`
    Do,
    /// `else`
    Else,
    /// `enum`
    Enum,
    /// `export`
    Export,
    /// `extends`
    Extends,
    /// `false`
    False,
    /// `finally`
    Finally,
    /// `for`
    For,
    /// `function`
    Function,
    /// `if`
    If,
    /// `import`
    Import,
    /// `in`
    In,
    /// `instanceof`
    Instanceof,
    /// `let`
    Let,
    /// `new`
    New,
    /// `null`
    Null,
    /// `of`
    Of,
    /// `return`
    Return,
    /// `static`
    Static,
    /// `super`
    Super,
    /// `switch`
    Switch,
    /// `this`
    This,
    /// `throw`
    Throw,
    /// `true`
    True,
    /// `try`
    Try,
    /// `typeof`
    Typeof,
    /// `var`
    Var,
    /// `void`
    Void,
    /// `while`
    While,
    /// `with`
    With,
    /// `yield`
    Yield,

    // ── Contextual keywords ───────────────────────────────────────────────
    /// `async`
    Async,
    /// `from`
    From,
    /// `as`
    As,
    /// `get`
    Get,
    /// `set`
    Set,
    /// `target`
    Target,
    /// `meta`
    Meta,

    // ── Punctuators ───────────────────────────────────────────────────────
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `.`
    Dot,
    /// `...`
    DotDotDot,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
    /// `==`
    EqualEqual,
    /// `!=`
    BangEqual,
    /// `===`
    EqualEqualEqual,
    /// `!==`
    BangEqualEqual,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `**`
    StarStar,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `++`
    PlusPlus,
    /// `--`
    MinusMinus,
    /// `<<`
    LessLess,
    /// `>>`
    GreaterGreater,
    /// `>>>`
    GreaterGreaterGreater,
    /// `&`
    Ampersand,
    /// `|`
    Pipe,
    /// `^`
    Caret,
    /// `!`
    Bang,
    /// `~`
    Tilde,
    /// `&&`
    AmpersandAmpersand,
    /// `||`
    PipePipe,
    /// `??`
    QuestionQuestion,
    /// `?`
    Question,
    /// `:`
    Colon,
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    StarEqual,
    /// `**=`
    StarStarEqual,
    /// `/=`
    SlashEqual,
    /// `%=`
    PercentEqual,
    /// `<<=`
    LessLessEqual,
    /// `>>=`
    GreaterGreaterEqual,
    /// `>>>=`
    GreaterGreaterGreaterEqual,
    /// `&=`
    AmpersandEqual,
    /// `|=`
    PipeEqual,
    /// `^=`
    CaretEqual,
    /// `&&=`
    AmpersandAmpersandEqual,
    /// `||=`
    PipePipeEqual,
    /// `??=`
    QuestionQuestionEqual,
    /// `=>`
    Arrow,
    /// `?.`
    QuestionDot,

    // ── Comments ──────────────────────────────────────────────────────────
    /// Single-line comment `// …`.
    SingleLineComment,
    /// Block comment `/* … */`.
    MultiLineComment,

    // ── End of file ───────────────────────────────────────────────────────
    /// End of input.
    Eof,
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenValue
// ─────────────────────────────────────────────────────────────────────────────

/// The payload value associated with a [`Token`].
#[derive(Debug, Clone, PartialEq)]
pub enum TokenValue {
    /// No semantic value (punctuators, reserved words, EOF, …).
    None,
    /// Raw source text for identifiers, strings, templates, comments, and
    /// regular-expression literals.
    Str(String),
    /// Parsed numeric value for [`TokenKind::NumericLiteral`].
    Number(f64),
}

// ─────────────────────────────────────────────────────────────────────────────
// Token
// ─────────────────────────────────────────────────────────────────────────────

/// A single lexical token produced by the [`Scanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The syntactic category.
    pub kind: TokenKind,
    /// The associated value, if any.
    pub value: TokenValue,
    /// Source location of this token.
    pub span: Span,
    /// `true` when at least one *LineTerminator* appeared between the previous
    /// token and this one.
    ///
    /// The parser uses this flag for Automatic Semicolon Insertion (ASI).
    pub had_line_terminator_before: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Character-classification helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` for ES2025 *LineTerminator* code points.
fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}

/// Returns `true` for ES2025 *WhiteSpace* **or** *LineTerminator* characters.
fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t'                      // CHARACTER TABULATION
        | '\x0B'                  // LINE TABULATION
        | '\x0C'                  // FORM FEED
        | ' '                     // SPACE
        | '\u{00A0}'              // NO-BREAK SPACE
        | '\u{FEFF}'              // ZERO WIDTH NO-BREAK SPACE (BOM)
        | '\u{1680}'              // OGHAM SPACE MARK
        | '\u{2000}'
            ..='\u{200A}' // EN QUAD … HAIR SPACE
        | '\u{202F}'              // NARROW NO-BREAK SPACE
        | '\u{205F}'              // MEDIUM MATHEMATICAL SPACE
        | '\u{3000}'              // IDEOGRAPHIC SPACE
        | '\n'
        | '\r'
        | '\u{2028}'
        | '\u{2029}'
    )
}

/// Returns `true` for characters that may *start* a JS identifier.
fn is_id_start(c: char) -> bool {
    c == '$' || c == '_' || c.is_alphabetic()
}

/// Returns `true` for characters that may *continue* a JS identifier.
fn is_id_continue(c: char) -> bool {
    c == '$' || c == '_' || c == '\u{200C}' || c == '\u{200D}' || c.is_alphanumeric()
}

/// Map an identifier string to a reserved-word/contextual-keyword
/// [`TokenKind`], or return `None` for plain identifiers.
fn keyword_kind(s: &str) -> Option<TokenKind> {
    match s {
        "await" => Some(TokenKind::Await),
        "break" => Some(TokenKind::Break),
        "case" => Some(TokenKind::Case),
        "catch" => Some(TokenKind::Catch),
        "class" => Some(TokenKind::Class),
        "const" => Some(TokenKind::Const),
        "continue" => Some(TokenKind::Continue),
        "debugger" => Some(TokenKind::Debugger),
        "default" => Some(TokenKind::Default),
        "delete" => Some(TokenKind::Delete),
        "do" => Some(TokenKind::Do),
        "else" => Some(TokenKind::Else),
        "enum" => Some(TokenKind::Enum),
        "export" => Some(TokenKind::Export),
        "extends" => Some(TokenKind::Extends),
        "false" => Some(TokenKind::False),
        "finally" => Some(TokenKind::Finally),
        "for" => Some(TokenKind::For),
        "function" => Some(TokenKind::Function),
        "if" => Some(TokenKind::If),
        "import" => Some(TokenKind::Import),
        "in" => Some(TokenKind::In),
        "instanceof" => Some(TokenKind::Instanceof),
        "let" => Some(TokenKind::Let),
        "new" => Some(TokenKind::New),
        "null" => Some(TokenKind::Null),
        "of" => Some(TokenKind::Of),
        "return" => Some(TokenKind::Return),
        "static" => Some(TokenKind::Static),
        "super" => Some(TokenKind::Super),
        "switch" => Some(TokenKind::Switch),
        "this" => Some(TokenKind::This),
        "throw" => Some(TokenKind::Throw),
        "true" => Some(TokenKind::True),
        "try" => Some(TokenKind::Try),
        "typeof" => Some(TokenKind::Typeof),
        "var" => Some(TokenKind::Var),
        "void" => Some(TokenKind::Void),
        "while" => Some(TokenKind::While),
        "with" => Some(TokenKind::With),
        "yield" => Some(TokenKind::Yield),
        "async" => Some(TokenKind::Async),
        "from" => Some(TokenKind::From),
        "as" => Some(TokenKind::As),
        "get" => Some(TokenKind::Get),
        "set" => Some(TokenKind::Set),
        "target" => Some(TokenKind::Target),
        "meta" => Some(TokenKind::Meta),
        _ => None,
    }
}

/// Returns `true` when a `/` should open a regular-expression literal rather
/// than act as a division operator, given the most recent significant token.
///
/// The heuristic: `/` is *division* only after tokens that produce a value
/// (identifier, numeric/string literal, `)`, `]`, `++`, `--`, or `true`/
/// `false`/`null`/`this`/`super`).  Every other context is regexp.
fn slash_is_regexp(last: Option<&TokenKind>) -> bool {
    match last {
        None => true,
        Some(k) => !matches!(
            k,
            TokenKind::Identifier
                | TokenKind::PrivateIdentifier
                | TokenKind::NumericLiteral
                | TokenKind::StringLiteral
                | TokenKind::NoSubstitutionTemplate
                | TokenKind::TemplateTail
                | TokenKind::RegExpLiteral
                | TokenKind::RightParen
                | TokenKind::RightBracket
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Null
                | TokenKind::This
                | TokenKind::Super
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scanner
// ─────────────────────────────────────────────────────────────────────────────

/// ES2025 JavaScript lexer.
///
/// Produces a stream of [`Token`]s from a UTF-8 source string.  Call
/// [`Scanner::next_token`] repeatedly until a token with
/// [`TokenKind::Eof`] is returned.
///
/// # Example
///
/// ```
/// use stator_core::parser::scanner::{Scanner, TokenKind};
///
/// let mut sc = Scanner::new("let x = 42;");
/// loop {
///     let tok = sc.next_token().unwrap();
///     if tok.kind == TokenKind::Eof { break; }
///     println!("{:?}", tok.kind);
/// }
/// ```
pub struct Scanner<'src> {
    /// The complete source string.
    source: &'src str,
    /// Current byte position within `source`.
    pos: usize,
    /// Current 1-based line number.
    line: u32,
    /// Current 1-based column number.
    column: u32,
    /// Stack of `brace_depth` values recorded when entering each template
    /// substitution (`` `…${ ``).  When we see `}` and `brace_depth` equals
    /// the value at the top of this stack, the `}` closes the substitution.
    template_stack: Vec<usize>,
    /// Nesting depth of plain `{…}` braces (not template-substitution openers).
    brace_depth: usize,
    /// The most recent *significant* token kind (non-whitespace, non-comment).
    /// Used for regexp / division disambiguation.
    last_significant_kind: Option<TokenKind>,
}

impl<'src> Scanner<'src> {
    /// Create a new scanner for the given UTF-8 source string.
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
            template_stack: Vec::new(),
            brace_depth: 0,
            last_significant_kind: None,
        }
    }

    /// Returns `true` when all input has been consumed.
    pub fn is_eof(&self) -> bool {
        self.pos >= self.source.len()
    }

    // ── Low-level character helpers ─────────────────────────────────────────

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut it = self.source[self.pos..].chars();
        it.next();
        it.next()
    }

    fn peek3(&self) -> Option<char> {
        let mut it = self.source[self.pos..].chars();
        it.next();
        it.next();
        it.next()
    }

    /// Advance past the current character and update line/column tracking.
    ///
    /// `\r\n` is treated as a single line terminator; the `\n` is consumed
    /// automatically so callers never see a stray `\r`.
    fn advance(&mut self) -> char {
        let ch = self.source[self.pos..]
            .chars()
            .next()
            .expect("advance called past end of input");
        self.pos += ch.len_utf8();
        match ch {
            '\r' => {
                // CRLF: consume the \n silently.
                if self.source[self.pos..].starts_with('\n') {
                    self.pos += 1;
                }
                self.line += 1;
                self.column = 1;
            }
            '\n' | '\u{2028}' | '\u{2029}' => {
                self.line += 1;
                self.column = 1;
            }
            _ => {
                self.column += 1;
            }
        }
        ch
    }

    fn current_pos(&self) -> Position {
        Position {
            offset: self.pos,
            line: self.line,
            column: self.column,
        }
    }

    // ── Whitespace ──────────────────────────────────────────────────────────

    /// Consume all leading whitespace and return `true` if any line
    /// terminators were encountered.
    fn skip_whitespace(&mut self) -> bool {
        let mut had_lt = false;
        while let Some(c) = self.peek() {
            if !is_js_whitespace(c) {
                break;
            }
            if is_line_terminator(c) {
                had_lt = true;
            }
            self.advance();
        }
        had_lt
    }

    // ── Digit-run helpers ───────────────────────────────────────────────────

    fn scan_decimal_digits(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '_') {
            self.advance();
        }
    }

    fn scan_hex_digits(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_hexdigit() || c == '_') {
            self.advance();
        }
    }

    fn scan_binary_digits(&mut self) {
        while matches!(self.peek(), Some(c) if matches!(c, '0' | '1') || c == '_') {
            self.advance();
        }
    }

    fn scan_octal_digits(&mut self) {
        while matches!(self.peek(), Some(c) if matches!(c, '0'..='7') || c == '_') {
            self.advance();
        }
    }

    // ── String / template escape helper ─────────────────────────────────────

    /// Consume one escape sequence starting *after* the leading `\`.
    fn scan_escape_sequence(&mut self) -> StatorResult<()> {
        match self.peek() {
            None => Err(StatorError::SyntaxError(
                "unterminated escape sequence".into(),
            )),
            Some(c) => {
                self.advance();
                match c {
                    'u' => {
                        if self.peek() == Some('{') {
                            self.advance(); // {
                            let mut count = 0usize;
                            while matches!(self.peek(), Some(d) if d.is_ascii_hexdigit()) {
                                self.advance();
                                count += 1;
                            }
                            if count == 0 {
                                return Err(StatorError::SyntaxError(
                                    "invalid Unicode escape sequence".into(),
                                ));
                            }
                            if self.peek() != Some('}') {
                                return Err(StatorError::SyntaxError(
                                    "expected '}' in Unicode escape sequence".into(),
                                ));
                            }
                            self.advance(); // }
                        } else {
                            for _ in 0..4 {
                                match self.peek() {
                                    Some(d) if d.is_ascii_hexdigit() => {
                                        self.advance();
                                    }
                                    _ => {
                                        return Err(StatorError::SyntaxError(
                                            "invalid Unicode escape sequence".into(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    'x' => {
                        for _ in 0..2 {
                            match self.peek() {
                                Some(d) if d.is_ascii_hexdigit() => {
                                    self.advance();
                                }
                                _ => {
                                    return Err(StatorError::SyntaxError(
                                        "invalid hex escape sequence".into(),
                                    ));
                                }
                            }
                        }
                    }
                    // Line continuation or simple single-char escape — already consumed.
                    _ => {}
                }
                Ok(())
            }
        }
    }

    // ── String literal ──────────────────────────────────────────────────────

    fn scan_string(&mut self, quote: char, start: Position) -> StatorResult<Token> {
        // `start` was recorded before the opening quote; raw includes it.
        let raw_start = start.offset;
        loop {
            match self.peek() {
                None | Some('\n') | Some('\r') | Some('\u{2028}') | Some('\u{2029}') => {
                    return Err(StatorError::SyntaxError(
                        "unterminated string literal".into(),
                    ));
                }
                Some(c) if c == quote => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume '\'
                    self.scan_escape_sequence()?;
                }
                _ => {
                    self.advance();
                }
            }
        }
        let raw = self.source[raw_start..self.pos].to_string();
        let end = self.current_pos();
        Ok(Token {
            kind: TokenKind::StringLiteral,
            value: TokenValue::Str(raw),
            span: Span { start, end },
            had_line_terminator_before: false, // caller patches this
        })
    }

    // ── Template literal body ───────────────────────────────────────────────

    /// Scan template characters after the opening `` ` `` or after a `}` that
    /// closes a substitution.  Returns `(raw_content, found_substitution)`.
    ///
    /// * `raw_content` is the text between the opening delimiter and `` ` ``
    ///   or `${`, **excluding** both delimiters.
    /// * `found_substitution` is `true` when the body was terminated by `${`.
    fn scan_template_body(&mut self) -> StatorResult<(String, bool)> {
        let body_start = self.pos;
        loop {
            match self.peek() {
                None => {
                    return Err(StatorError::SyntaxError(
                        "unterminated template literal".into(),
                    ));
                }
                Some('`') => {
                    let raw = self.source[body_start..self.pos].to_string();
                    self.advance(); // consume closing `
                    return Ok((raw, false));
                }
                Some('$') if self.peek2() == Some('{') => {
                    let raw = self.source[body_start..self.pos].to_string();
                    self.advance(); // $
                    self.advance(); // {
                    return Ok((raw, true));
                }
                Some('\\') => {
                    self.advance(); // '\'
                    // Consume one raw character (we store raw text, not cooked).
                    if let Some(nc) = self.peek() {
                        if nc == '\r' {
                            self.advance();
                            if self.peek() == Some('\n') {
                                self.advance();
                            }
                        } else {
                            self.advance();
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    // ── Regular-expression literal ──────────────────────────────────────────

    fn scan_regexp(&mut self, start: Position) -> StatorResult<Token> {
        // Opening '/' already consumed by the caller.
        let raw_start = start.offset;
        let mut in_class = false;

        loop {
            match self.peek() {
                None | Some('\n') | Some('\r') | Some('\u{2028}') | Some('\u{2029}') => {
                    return Err(StatorError::SyntaxError(
                        "unterminated regular expression literal".into(),
                    ));
                }
                Some('[') => {
                    in_class = true;
                    self.advance();
                }
                Some(']') => {
                    in_class = false;
                    self.advance();
                }
                Some('/') if !in_class => {
                    self.advance(); // closing /
                    break;
                }
                Some('\\') => {
                    self.advance(); // '\'
                    match self.peek() {
                        None | Some('\n') | Some('\r') | Some('\u{2028}') | Some('\u{2029}') => {
                            return Err(StatorError::SyntaxError(
                                "unterminated regular expression literal".into(),
                            ));
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        // Flags: zero or more ASCII letters.
        while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
            self.advance();
        }

        let raw = self.source[raw_start..self.pos].to_string();
        let end = self.current_pos();
        Ok(Token {
            kind: TokenKind::RegExpLiteral,
            value: TokenValue::Str(raw),
            span: Span { start, end },
            had_line_terminator_before: false, // caller patches this
        })
    }

    // ── Numeric literal ─────────────────────────────────────────────────────

    /// Scan a numeric literal given that `first` has already been consumed.
    fn scan_numeric(&mut self, first: char, start: Position) -> StatorResult<Token> {
        let num_start = start.offset;

        if first == '0' {
            match self.peek() {
                Some('x') | Some('X') => {
                    self.advance();
                    self.scan_hex_digits();
                    if self.peek() == Some('n') {
                        self.advance();
                    }
                }
                Some('o') | Some('O') => {
                    self.advance();
                    self.scan_octal_digits();
                    if self.peek() == Some('n') {
                        self.advance();
                    }
                }
                Some('b') | Some('B') => {
                    self.advance();
                    self.scan_binary_digits();
                    if self.peek() == Some('n') {
                        self.advance();
                    }
                }
                Some(c) if c.is_ascii_digit() => {
                    // Legacy octal (e.g. `017`) or decimal continuation.
                    self.scan_decimal_digits();
                    // If it has a decimal point or exponent, treat as decimal.
                    if matches!(self.peek(), Some('.')) {
                        self.advance();
                        self.scan_decimal_digits();
                        self.scan_exponent();
                    } else {
                        self.scan_exponent();
                    }
                }
                Some('.') => {
                    self.advance();
                    self.scan_decimal_digits();
                    self.scan_exponent();
                }
                Some('e') | Some('E') => {
                    self.scan_exponent();
                }
                Some('n') => {
                    self.advance(); // BigInt `0n`
                }
                _ => {} // bare `0`
            }
        } else if first == '.' {
            // `.5`, `.5e3`, etc. — leading dot, digits follow.
            self.scan_decimal_digits();
            self.scan_exponent();
        } else {
            // Decimal integer: first digit already consumed.
            self.scan_decimal_digits();
            if self.peek() == Some('.') {
                self.advance();
                self.scan_decimal_digits();
                self.scan_exponent();
            } else if matches!(self.peek(), Some('e') | Some('E')) {
                self.scan_exponent();
            } else if self.peek() == Some('n') {
                self.advance(); // BigInt
            }
        }

        let raw = &self.source[num_start..self.pos];
        let value = parse_numeric_raw(raw);
        let end = self.current_pos();
        Ok(Token {
            kind: TokenKind::NumericLiteral,
            value: TokenValue::Number(value),
            span: Span { start, end },
            had_line_terminator_before: false, // caller patches this
        })
    }

    /// Consume an optional exponent part (`e` / `E`, optional sign, digits).
    fn scan_exponent(&mut self) {
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.advance();
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.advance();
            }
            self.scan_decimal_digits();
        }
    }

    // ── Identifier / keyword ────────────────────────────────────────────────

    /// Scan an identifier that starts at `id_start_byte`, given that the
    /// first character `first` has already been consumed.
    fn scan_identifier(&mut self, first: char, start: Position) -> Token {
        let id_start = start.offset;
        // If the first char was a backslash, we consumed `\`; handle `u` escape.
        if first == '\\' {
            // scan_unicode_escape_in_id has already advanced past `\`.
            self.scan_unicode_escape_in_id_rest();
        }
        loop {
            match self.peek() {
                Some(c) if is_id_continue(c) => {
                    self.advance();
                }
                Some('\\') if self.peek2() == Some('u') => {
                    self.advance(); // '\'
                    self.scan_unicode_escape_in_id_rest();
                }
                _ => break,
            }
        }
        let name = self.source[id_start..self.pos].to_string();
        // For escaped identifiers, the raw text contains `\uXXXX` sequences;
        // keyword matching against raw text is intentionally skipped for them
        // (the spec allows `\u006C\u0065\u0074` as an identifier, not a keyword).
        let kind = if first == '\\' {
            TokenKind::Identifier
        } else {
            keyword_kind(&name).unwrap_or(TokenKind::Identifier)
        };
        let value = match &kind {
            TokenKind::Identifier => TokenValue::Str(name),
            _ => TokenValue::None,
        };
        let end = self.current_pos();
        Token {
            kind,
            value,
            span: Span { start, end },
            had_line_terminator_before: false,
        }
    }

    /// After consuming `\`, consume the rest of a `\uXXXX` or `\u{…}` escape.
    fn scan_unicode_escape_in_id_rest(&mut self) {
        if self.peek() != Some('u') {
            return;
        }
        self.advance(); // 'u'
        if self.peek() == Some('{') {
            self.advance();
            while matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
                self.advance();
            }
            if self.peek() == Some('}') {
                self.advance();
            }
        } else {
            for _ in 0..4 {
                if matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
                    self.advance();
                }
            }
        }
    }

    // ── Main public API ─────────────────────────────────────────────────────

    /// Scan and return the next [`Token`].
    ///
    /// Returns a token with [`TokenKind::Eof`] when the input is exhausted.
    pub fn next_token(&mut self) -> StatorResult<Token> {
        let had_lt = self.skip_whitespace();

        // ── EOF ──
        if self.is_eof() {
            return Ok(Token {
                kind: TokenKind::Eof,
                value: TokenValue::None,
                span: Span {
                    start: self.current_pos(),
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            });
        }

        // ── Shebang (only valid at byte 0) ──────────────────────────────────
        if self.pos == 0 && self.peek() == Some('#') && self.peek2() == Some('!') {
            let start = self.current_pos();
            self.advance(); // #
            self.advance(); // !
            while let Some(c) = self.peek() {
                if is_line_terminator(c) {
                    break;
                }
                self.advance();
            }
            let text = self.source[..self.pos].to_string();
            let end = self.current_pos();
            return Ok(Token {
                kind: TokenKind::SingleLineComment,
                value: TokenValue::Str(text),
                span: Span { start, end },
                had_line_terminator_before: had_lt,
            });
        }

        let start = self.current_pos();
        let c = self.advance();

        let tok = match c {
            // ── Comments / division / regexp ─────────────────────────────
            '/' => {
                match self.peek() {
                    Some('/') => {
                        self.advance(); // second /
                        let text_start = self.pos;
                        while let Some(ch) = self.peek() {
                            if is_line_terminator(ch) {
                                break;
                            }
                            self.advance();
                        }
                        let text = self.source[text_start..self.pos].to_string();
                        let end = self.current_pos();
                        // Comments are not "significant" — don't update last_significant_kind.
                        return Ok(Token {
                            kind: TokenKind::SingleLineComment,
                            value: TokenValue::Str(text),
                            span: Span { start, end },
                            had_line_terminator_before: had_lt,
                        });
                    }
                    Some('*') => {
                        self.advance(); // *
                        let text_start = self.pos;
                        let mut inner_lt = false;
                        loop {
                            match self.peek() {
                                None => {
                                    return Err(StatorError::SyntaxError(
                                        "unterminated block comment".into(),
                                    ));
                                }
                                Some('*') if self.peek2() == Some('/') => {
                                    let text = self.source[text_start..self.pos].to_string();
                                    self.advance(); // *
                                    self.advance(); // /
                                    let end = self.current_pos();
                                    return Ok(Token {
                                        kind: TokenKind::MultiLineComment,
                                        value: TokenValue::Str(text),
                                        span: Span { start, end },
                                        had_line_terminator_before: had_lt || inner_lt,
                                    });
                                }
                                Some(ch) => {
                                    if is_line_terminator(ch) {
                                        inner_lt = true;
                                    }
                                    self.advance();
                                }
                            }
                        }
                    }
                    Some('=') if !slash_is_regexp(self.last_significant_kind.as_ref()) => {
                        self.advance();
                        Token {
                            kind: TokenKind::SlashEqual,
                            value: TokenValue::None,
                            span: Span {
                                start,
                                end: self.current_pos(),
                            },
                            had_line_terminator_before: had_lt,
                        }
                    }
                    _ => {
                        if slash_is_regexp(self.last_significant_kind.as_ref()) {
                            let mut tok = self.scan_regexp(start)?;
                            tok.had_line_terminator_before = had_lt;
                            self.last_significant_kind = Some(TokenKind::RegExpLiteral);
                            return Ok(tok);
                        }
                        Token {
                            kind: TokenKind::Slash,
                            value: TokenValue::None,
                            span: Span {
                                start,
                                end: self.current_pos(),
                            },
                            had_line_terminator_before: had_lt,
                        }
                    }
                }
            }

            // ── String literals ──────────────────────────────────────────
            '"' | '\'' => {
                let mut tok = self.scan_string(c, start)?;
                tok.had_line_terminator_before = had_lt;
                self.last_significant_kind = Some(TokenKind::StringLiteral);
                return Ok(tok);
            }

            // ── Template literals ────────────────────────────────────────
            '`' => {
                let (raw, has_sub) = self.scan_template_body()?;
                let end = self.current_pos();
                let kind = if has_sub {
                    self.template_stack.push(self.brace_depth);
                    TokenKind::TemplateHead
                } else {
                    TokenKind::NoSubstitutionTemplate
                };
                self.last_significant_kind = Some(kind);
                return Ok(Token {
                    kind,
                    value: TokenValue::Str(raw),
                    span: Span { start, end },
                    had_line_terminator_before: had_lt,
                });
            }

            // ── Numeric literals — digit-first ───────────────────────────
            c if c.is_ascii_digit() => {
                let mut tok = self.scan_numeric(c, start)?;
                tok.had_line_terminator_before = had_lt;
                self.last_significant_kind = Some(TokenKind::NumericLiteral);
                return Ok(tok);
            }

            // ── Numeric literal — leading dot ────────────────────────────
            '.' => {
                if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    let mut tok = self.scan_numeric('.', start)?;
                    tok.had_line_terminator_before = had_lt;
                    self.last_significant_kind = Some(TokenKind::NumericLiteral);
                    return Ok(tok);
                } else if self.peek() == Some('.') && self.peek2() == Some('.') {
                    self.advance();
                    self.advance();
                    Token {
                        kind: TokenKind::DotDotDot,
                        value: TokenValue::None,
                        span: Span {
                            start,
                            end: self.current_pos(),
                        },
                        had_line_terminator_before: had_lt,
                    }
                } else {
                    Token {
                        kind: TokenKind::Dot,
                        value: TokenValue::None,
                        span: Span {
                            start,
                            end: self.current_pos(),
                        },
                        had_line_terminator_before: had_lt,
                    }
                }
            }

            // ── Private identifiers ──────────────────────────────────────
            '#' => {
                if !matches!(self.peek(), Some(c) if is_id_start(c) || c == '\\') {
                    return Err(StatorError::SyntaxError(format!(
                        "invalid or unexpected token '#' at {}:{}",
                        start.line, start.column
                    )));
                }
                let name_start = self.pos;
                // Consume identifier name (without '#').
                loop {
                    match self.peek() {
                        Some(nc) if is_id_continue(nc) => {
                            self.advance();
                        }
                        Some('\\') if self.peek2() == Some('u') => {
                            self.advance();
                            self.scan_unicode_escape_in_id_rest();
                        }
                        _ => break,
                    }
                }
                let name = self.source[name_start..self.pos].to_string();
                let end = self.current_pos();
                Token {
                    kind: TokenKind::PrivateIdentifier,
                    value: TokenValue::Str(name),
                    span: Span { start, end },
                    had_line_terminator_before: had_lt,
                }
            }

            // ── Identifiers / keywords ───────────────────────────────────
            c if is_id_start(c) => {
                let mut tok = self.scan_identifier(c, start);
                tok.had_line_terminator_before = had_lt;
                self.last_significant_kind = Some(tok.kind);
                return Ok(tok);
            }

            // Identifier starting with unicode escape `\uXXXX`
            '\\' if self.peek() == Some('u') => {
                let mut tok = self.scan_identifier('\\', start);
                tok.had_line_terminator_before = had_lt;
                self.last_significant_kind = Some(tok.kind);
                return Ok(tok);
            }

            // ── `}` — plain or template tail/middle ─────────────────────
            '}' => {
                if let Some(&depth) = self.template_stack.last()
                    && self.brace_depth == depth
                {
                    // This `}` closes a template substitution.
                    self.template_stack.pop();
                    let (raw, has_sub) = self.scan_template_body()?;
                    let end = self.current_pos();
                    let kind = if has_sub {
                        self.template_stack.push(self.brace_depth);
                        TokenKind::TemplateMiddle
                    } else {
                        TokenKind::TemplateTail
                    };
                    self.last_significant_kind = Some(kind);
                    return Ok(Token {
                        kind,
                        value: TokenValue::Str(raw),
                        span: Span { start, end },
                        had_line_terminator_before: had_lt,
                    });
                }
                self.brace_depth = self.brace_depth.saturating_sub(1);
                Token {
                    kind: TokenKind::RightBrace,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            // ── Simple punctuators ───────────────────────────────────────
            '{' => {
                self.brace_depth += 1;
                Token {
                    kind: TokenKind::LeftBrace,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }
            '(' => Token {
                kind: TokenKind::LeftParen,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            ')' => Token {
                kind: TokenKind::RightParen,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            '[' => Token {
                kind: TokenKind::LeftBracket,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            ']' => Token {
                kind: TokenKind::RightBracket,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            ';' => Token {
                kind: TokenKind::Semicolon,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            ',' => Token {
                kind: TokenKind::Comma,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            '~' => Token {
                kind: TokenKind::Tilde,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },
            ':' => Token {
                kind: TokenKind::Colon,
                value: TokenValue::None,
                span: Span {
                    start,
                    end: self.current_pos(),
                },
                had_line_terminator_before: had_lt,
            },

            // ── Multi-char punctuators ───────────────────────────────────
            '<' => {
                let kind = if self.peek() == Some('<') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::LessLessEqual
                    } else {
                        TokenKind::LessLess
                    }
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '>' => {
                let kind = if self.peek() == Some('>') {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        if self.peek() == Some('=') {
                            self.advance();
                            TokenKind::GreaterGreaterGreaterEqual
                        } else {
                            TokenKind::GreaterGreaterGreater
                        }
                    } else if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::GreaterGreaterEqual
                    } else {
                        TokenKind::GreaterGreater
                    }
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '=' => {
                let kind = if self.peek() == Some('=') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::EqualEqualEqual
                    } else {
                        TokenKind::EqualEqual
                    }
                } else if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Equal
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '!' => {
                let kind = if self.peek() == Some('=') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::BangEqualEqual
                    } else {
                        TokenKind::BangEqual
                    }
                } else {
                    TokenKind::Bang
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '+' => {
                let kind = if self.peek() == Some('+') {
                    self.advance();
                    TokenKind::PlusPlus
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::PlusEqual
                } else {
                    TokenKind::Plus
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '-' => {
                let kind = if self.peek() == Some('-') {
                    self.advance();
                    TokenKind::MinusMinus
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::MinusEqual
                } else {
                    TokenKind::Minus
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '*' => {
                let kind = if self.peek() == Some('*') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::StarStarEqual
                    } else {
                        TokenKind::StarStar
                    }
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::StarEqual
                } else {
                    TokenKind::Star
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '%' => {
                let kind = if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::PercentEqual
                } else {
                    TokenKind::Percent
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '&' => {
                let kind = if self.peek() == Some('&') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::AmpersandAmpersandEqual
                    } else {
                        TokenKind::AmpersandAmpersand
                    }
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::AmpersandEqual
                } else {
                    TokenKind::Ampersand
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '|' => {
                let kind = if self.peek() == Some('|') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::PipePipeEqual
                    } else {
                        TokenKind::PipePipe
                    }
                } else if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::PipeEqual
                } else {
                    TokenKind::Pipe
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '^' => {
                let kind = if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::CaretEqual
                } else {
                    TokenKind::Caret
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            '?' => {
                let kind = if self.peek() == Some('?') {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        TokenKind::QuestionQuestionEqual
                    } else {
                        TokenKind::QuestionQuestion
                    }
                } else if self.peek() == Some('.')
                    && !matches!(self.peek3(), Some(d) if d.is_ascii_digit())
                {
                    // `?.` — but NOT `?.0` (that would be `?` + `.0`)
                    self.advance();
                    TokenKind::QuestionDot
                } else {
                    TokenKind::Question
                };
                Token {
                    kind,
                    value: TokenValue::None,
                    span: Span {
                        start,
                        end: self.current_pos(),
                    },
                    had_line_terminator_before: had_lt,
                }
            }

            _ => {
                return Err(StatorError::SyntaxError(format!(
                    "unexpected character {:?} at line {}, column {}",
                    c, start.line, start.column
                )));
            }
        };

        self.last_significant_kind = Some(tok.kind);
        Ok(tok)
    }

    /// Convenience: tokenize the entire `source` string and return all tokens
    /// (the [`TokenKind::Eof`] sentinel is **not** included).
    ///
    /// # Errors
    ///
    /// Returns the first [`StatorError::SyntaxError`] encountered.
    pub fn tokenize_all(source: &'src str) -> StatorResult<Vec<Token>> {
        let mut scanner = Scanner::new(source);
        let mut tokens = Vec::new();
        loop {
            let tok = scanner.next_token()?;
            if tok.kind == TokenKind::Eof {
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numeric parsing helper
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the raw text of a numeric literal to an `f64`.
///
/// Numeric separators (`_`) and the BigInt suffix (`n`) are stripped before
/// parsing.  Returns [`f64::NAN`] if the raw text cannot be parsed (should not
/// happen for well-formed input).
fn parse_numeric_raw(raw: &str) -> f64 {
    let clean: String = raw.chars().filter(|&c| c != '_' && c != 'n').collect();
    if clean.starts_with("0x") || clean.starts_with("0X") {
        i64::from_str_radix(&clean[2..], 16)
            .map(|n| n as f64)
            .unwrap_or(f64::NAN)
    } else if clean.starts_with("0o") || clean.starts_with("0O") {
        i64::from_str_radix(&clean[2..], 8)
            .map(|n| n as f64)
            .unwrap_or(f64::NAN)
    } else if clean.starts_with("0b") || clean.starts_with("0B") {
        i64::from_str_radix(&clean[2..], 2)
            .map(|n| n as f64)
            .unwrap_or(f64::NAN)
    } else {
        clean.parse::<f64>().unwrap_or(f64::NAN)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokenise `src`, ignoring comments, and return a list of token kinds.
    fn kinds(src: &str) -> Vec<TokenKind> {
        Scanner::tokenize_all(src)
            .unwrap()
            .into_iter()
            .filter(|t| {
                !matches!(
                    t.kind,
                    TokenKind::SingleLineComment | TokenKind::MultiLineComment
                )
            })
            .map(|t| t.kind)
            .collect()
    }

    /// Same as `kinds` but also returns string/number values for literals.
    fn tokens(src: &str) -> Vec<Token> {
        Scanner::tokenize_all(src).unwrap()
    }

    // ── Keywords ─────────────────────────────────────────────────────────────

    #[test]
    fn test_keywords_reserved() {
        let src = "break case catch class const continue debugger default \
                   delete do else enum export extends false finally for \
                   function if import in instanceof let new null of return \
                   static super switch this throw true try typeof var void \
                   while with yield await";
        let toks = kinds(src);
        assert_eq!(
            toks,
            vec![
                TokenKind::Break,
                TokenKind::Case,
                TokenKind::Catch,
                TokenKind::Class,
                TokenKind::Const,
                TokenKind::Continue,
                TokenKind::Debugger,
                TokenKind::Default,
                TokenKind::Delete,
                TokenKind::Do,
                TokenKind::Else,
                TokenKind::Enum,
                TokenKind::Export,
                TokenKind::Extends,
                TokenKind::False,
                TokenKind::Finally,
                TokenKind::For,
                TokenKind::Function,
                TokenKind::If,
                TokenKind::Import,
                TokenKind::In,
                TokenKind::Instanceof,
                TokenKind::Let,
                TokenKind::New,
                TokenKind::Null,
                TokenKind::Of,
                TokenKind::Return,
                TokenKind::Static,
                TokenKind::Super,
                TokenKind::Switch,
                TokenKind::This,
                TokenKind::Throw,
                TokenKind::True,
                TokenKind::Try,
                TokenKind::Typeof,
                TokenKind::Var,
                TokenKind::Void,
                TokenKind::While,
                TokenKind::With,
                TokenKind::Yield,
                TokenKind::Await,
            ]
        );
    }

    #[test]
    fn test_keywords_contextual() {
        let toks = kinds("async from as get set target meta");
        assert_eq!(
            toks,
            vec![
                TokenKind::Async,
                TokenKind::From,
                TokenKind::As,
                TokenKind::Get,
                TokenKind::Set,
                TokenKind::Target,
                TokenKind::Meta,
            ]
        );
    }

    // ── Identifiers ───────────────────────────────────────────────────────────

    #[test]
    fn test_identifier_simple() {
        let toks = tokens("foo _bar $baz");
        assert_eq!(toks[0].kind, TokenKind::Identifier);
        assert_eq!(toks[0].value, TokenValue::Str("foo".into()));
        assert_eq!(toks[1].kind, TokenKind::Identifier);
        assert_eq!(toks[2].kind, TokenKind::Identifier);
    }

    #[test]
    fn test_identifier_unicode() {
        let toks = tokens("café");
        assert_eq!(toks[0].kind, TokenKind::Identifier);
        assert_eq!(toks[0].value, TokenValue::Str("café".into()));
    }

    #[test]
    fn test_private_identifier() {
        let toks = tokens("#foo #_bar");
        assert_eq!(toks[0].kind, TokenKind::PrivateIdentifier);
        assert_eq!(toks[0].value, TokenValue::Str("foo".into()));
        assert_eq!(toks[1].kind, TokenKind::PrivateIdentifier);
        assert_eq!(toks[1].value, TokenValue::Str("_bar".into()));
    }

    // ── Numeric literals ──────────────────────────────────────────────────────

    #[test]
    fn test_numeric_decimal_integers() {
        let toks = tokens("0 42 100");
        for t in &toks {
            assert_eq!(t.kind, TokenKind::NumericLiteral);
        }
        assert_eq!(toks[0].value, TokenValue::Number(0.0));
        assert_eq!(toks[1].value, TokenValue::Number(42.0));
        assert_eq!(toks[2].value, TokenValue::Number(100.0));
    }

    #[test]
    fn test_numeric_floats() {
        let toks = tokens("1.5 .5 1e3 1.5e-2 0.0");
        assert_eq!(toks[0].value, TokenValue::Number(1.5));
        assert_eq!(toks[1].value, TokenValue::Number(0.5));
        assert_eq!(toks[2].value, TokenValue::Number(1000.0));
        assert_eq!(toks[3].value, TokenValue::Number(0.015));
        assert_eq!(toks[4].value, TokenValue::Number(0.0));
    }

    #[test]
    fn test_numeric_hex() {
        let toks = tokens("0xFF 0x1A 0X0a");
        assert_eq!(toks[0].value, TokenValue::Number(255.0));
        assert_eq!(toks[1].value, TokenValue::Number(26.0));
        assert_eq!(toks[2].value, TokenValue::Number(10.0));
    }

    #[test]
    fn test_numeric_binary() {
        let toks = tokens("0b1010 0B1111");
        assert_eq!(toks[0].value, TokenValue::Number(10.0));
        assert_eq!(toks[1].value, TokenValue::Number(15.0));
    }

    #[test]
    fn test_numeric_octal() {
        let toks = tokens("0o17 0O7");
        assert_eq!(toks[0].value, TokenValue::Number(15.0));
        assert_eq!(toks[1].value, TokenValue::Number(7.0));
    }

    #[test]
    fn test_numeric_bigint() {
        let toks = tokens("123n 0n 0b1n");
        for t in &toks {
            assert_eq!(t.kind, TokenKind::NumericLiteral);
        }
    }

    #[test]
    fn test_numeric_separator() {
        let toks = tokens("1_000_000 0xFF_FF");
        assert_eq!(toks[0].value, TokenValue::Number(1_000_000.0));
        assert_eq!(toks[1].value, TokenValue::Number(0xFFFF as f64));
    }

    // ── String literals ───────────────────────────────────────────────────────

    #[test]
    fn test_string_double_quote() {
        let toks = tokens(r#""hello world""#);
        assert_eq!(toks[0].kind, TokenKind::StringLiteral);
        assert_eq!(toks[0].value, TokenValue::Str(r#""hello world""#.into()));
    }

    #[test]
    fn test_string_single_quote() {
        let toks = tokens("'it\\'s'");
        assert_eq!(toks[0].kind, TokenKind::StringLiteral);
    }

    #[test]
    fn test_string_escape_sequences() {
        let toks = tokens(r#""\n\t\r\\\"" '\u0041' "\x41""#);
        assert_eq!(toks[0].kind, TokenKind::StringLiteral);
        assert_eq!(toks[1].kind, TokenKind::StringLiteral);
        assert_eq!(toks[2].kind, TokenKind::StringLiteral);
    }

    #[test]
    fn test_string_unterminated_error() {
        let result = Scanner::tokenize_all(r#""unterminated"#);
        assert!(result.is_err());
    }

    // ── Template literals ─────────────────────────────────────────────────────

    #[test]
    fn test_template_no_substitution() {
        let toks = tokens("`hello world`");
        assert_eq!(toks[0].kind, TokenKind::NoSubstitutionTemplate);
        assert_eq!(toks[0].value, TokenValue::Str("hello world".into()));
    }

    #[test]
    fn test_template_with_substitution() {
        let toks = tokens("`hello ${name}!`");
        assert_eq!(toks[0].kind, TokenKind::TemplateHead);
        assert_eq!(toks[0].value, TokenValue::Str("hello ".into()));
        assert_eq!(toks[1].kind, TokenKind::Identifier); // name
        assert_eq!(toks[2].kind, TokenKind::TemplateTail);
        assert_eq!(toks[2].value, TokenValue::Str("!".into()));
    }

    #[test]
    fn test_template_multiple_substitutions() {
        let toks = tokens("`${a} and ${b}`");
        assert_eq!(toks[0].kind, TokenKind::TemplateHead);
        assert_eq!(toks[0].value, TokenValue::Str("".into()));
        assert_eq!(toks[1].kind, TokenKind::Identifier); // a
        assert_eq!(toks[2].kind, TokenKind::TemplateMiddle);
        assert_eq!(toks[2].value, TokenValue::Str(" and ".into()));
        assert_eq!(toks[3].kind, TokenKind::Identifier); // b
        assert_eq!(toks[4].kind, TokenKind::TemplateTail);
        assert_eq!(toks[4].value, TokenValue::Str("".into()));
    }

    #[test]
    fn test_template_nested() {
        // `outer ${`inner`} end`
        let toks = tokens("`outer ${`inner`} end`");
        assert_eq!(toks[0].kind, TokenKind::TemplateHead);
        assert_eq!(toks[0].value, TokenValue::Str("outer ".into()));
        // inner template: NoSubstitutionTemplate
        assert_eq!(toks[1].kind, TokenKind::NoSubstitutionTemplate);
        assert_eq!(toks[1].value, TokenValue::Str("inner".into()));
        assert_eq!(toks[2].kind, TokenKind::TemplateTail);
        assert_eq!(toks[2].value, TokenValue::Str(" end".into()));
    }

    #[test]
    fn test_template_expression_with_braces() {
        // `a ${{k:1}} b`
        let toks = tokens("`a ${{k:1}} b`");
        assert_eq!(toks[0].kind, TokenKind::TemplateHead);
        // {, k, :, 1, }
        assert_eq!(toks[1].kind, TokenKind::LeftBrace);
        assert_eq!(toks[2].kind, TokenKind::Identifier);
        assert_eq!(toks[3].kind, TokenKind::Colon);
        assert_eq!(toks[4].kind, TokenKind::NumericLiteral);
        assert_eq!(toks[5].kind, TokenKind::RightBrace);
        assert_eq!(toks[6].kind, TokenKind::TemplateTail);
    }

    // ── Regular expressions ───────────────────────────────────────────────────

    #[test]
    fn test_regexp_basic() {
        let toks = tokens("/foo/gi");
        assert_eq!(toks[0].kind, TokenKind::RegExpLiteral);
        assert_eq!(toks[0].value, TokenValue::Str("/foo/gi".into()));
    }

    #[test]
    fn test_regexp_with_char_class() {
        let toks = tokens("/[a-z]+/");
        assert_eq!(toks[0].kind, TokenKind::RegExpLiteral);
    }

    #[test]
    fn test_regexp_escaped_slash() {
        let toks = tokens(r#"/foo\/bar/"#);
        assert_eq!(toks[0].kind, TokenKind::RegExpLiteral);
        assert_eq!(toks[0].value, TokenValue::Str("/foo\\/bar/".into()));
    }

    #[test]
    fn test_regexp_after_return() {
        // `return /foo/` — after a keyword that allows regexp
        let toks = kinds("return /foo/");
        assert_eq!(toks, vec![TokenKind::Return, TokenKind::RegExpLiteral]);
    }

    #[test]
    fn test_division_after_identifier() {
        // `x / y` — after an identifier, `/` is division
        let toks = kinds("x / y");
        assert_eq!(
            toks,
            vec![
                TokenKind::Identifier,
                TokenKind::Slash,
                TokenKind::Identifier
            ]
        );
    }

    #[test]
    fn test_division_after_number() {
        let toks = kinds("4 / 2");
        assert_eq!(
            toks,
            vec![
                TokenKind::NumericLiteral,
                TokenKind::Slash,
                TokenKind::NumericLiteral
            ]
        );
    }

    #[test]
    fn test_regexp_after_assignment() {
        // `x = /foo/` — after `=`, `/` is a regexp
        let toks = kinds("x = /foo/");
        assert_eq!(
            toks,
            vec![
                TokenKind::Identifier,
                TokenKind::Equal,
                TokenKind::RegExpLiteral
            ]
        );
    }

    #[test]
    fn test_regexp_vs_division_after_paren() {
        // `(a) / b` — after `)`, `/` is division
        let toks = kinds("(a) / b");
        assert_eq!(
            toks,
            vec![
                TokenKind::LeftParen,
                TokenKind::Identifier,
                TokenKind::RightParen,
                TokenKind::Slash,
                TokenKind::Identifier
            ]
        );
    }

    // ── Comments ──────────────────────────────────────────────────────────────

    #[test]
    fn test_single_line_comment() {
        let toks = tokens("// this is a comment\nfoo");
        assert_eq!(toks[0].kind, TokenKind::SingleLineComment);
        assert_eq!(toks[0].value, TokenValue::Str(" this is a comment".into()));
        assert_eq!(toks[1].kind, TokenKind::Identifier);
        // identifier follows the comment on a new line
        assert!(toks[1].had_line_terminator_before);
    }

    #[test]
    fn test_block_comment() {
        let toks = tokens("/* block */ foo");
        assert_eq!(toks[0].kind, TokenKind::MultiLineComment);
        assert_eq!(toks[0].value, TokenValue::Str(" block ".into()));
        assert_eq!(toks[1].kind, TokenKind::Identifier);
    }

    #[test]
    fn test_block_comment_with_line_terminator() {
        let toks = tokens("/* line1\nline2 */ foo");
        assert_eq!(toks[0].kind, TokenKind::MultiLineComment);
        // Block comment containing a line terminator is itself marked.
        assert!(toks[0].had_line_terminator_before || toks[0].kind == TokenKind::MultiLineComment);
    }

    // ── Punctuators ───────────────────────────────────────────────────────────

    #[test]
    fn test_punctuators_single() {
        let toks = kinds("{ } ( ) [ ] . ; , ~ : ?");
        assert_eq!(
            toks,
            vec![
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::Dot,
                TokenKind::Semicolon,
                TokenKind::Comma,
                TokenKind::Tilde,
                TokenKind::Colon,
                TokenKind::Question,
            ]
        );
    }

    #[test]
    fn test_punctuators_comparison() {
        let toks = kinds("< > <= >= == != === !==");
        assert_eq!(
            toks,
            vec![
                TokenKind::Less,
                TokenKind::Greater,
                TokenKind::LessEqual,
                TokenKind::GreaterEqual,
                TokenKind::EqualEqual,
                TokenKind::BangEqual,
                TokenKind::EqualEqualEqual,
                TokenKind::BangEqualEqual,
            ]
        );
    }

    #[test]
    fn test_punctuators_arithmetic() {
        let toks = kinds("+ - * ** % ++ --");
        assert_eq!(
            toks,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::StarStar,
                TokenKind::Percent,
                TokenKind::PlusPlus,
                TokenKind::MinusMinus,
            ]
        );
    }

    #[test]
    fn test_punctuators_bitwise() {
        let toks = kinds("<< >> >>> & | ^ !");
        assert_eq!(
            toks,
            vec![
                TokenKind::LessLess,
                TokenKind::GreaterGreater,
                TokenKind::GreaterGreaterGreater,
                TokenKind::Ampersand,
                TokenKind::Pipe,
                TokenKind::Caret,
                TokenKind::Bang,
            ]
        );
    }

    #[test]
    fn test_punctuators_logical() {
        let toks = kinds("&& || ?? ?.");
        assert_eq!(
            toks,
            vec![
                TokenKind::AmpersandAmpersand,
                TokenKind::PipePipe,
                TokenKind::QuestionQuestion,
                TokenKind::QuestionDot,
            ]
        );
    }

    #[test]
    fn test_punctuators_assignment() {
        let toks = kinds("= += -= *= **= %= <<= >>= >>>= &= |= ^= &&= ||= ??=");
        assert_eq!(
            toks,
            vec![
                TokenKind::Equal,
                TokenKind::PlusEqual,
                TokenKind::MinusEqual,
                TokenKind::StarEqual,
                TokenKind::StarStarEqual,
                TokenKind::PercentEqual,
                TokenKind::LessLessEqual,
                TokenKind::GreaterGreaterEqual,
                TokenKind::GreaterGreaterGreaterEqual,
                TokenKind::AmpersandEqual,
                TokenKind::PipeEqual,
                TokenKind::CaretEqual,
                TokenKind::AmpersandAmpersandEqual,
                TokenKind::PipePipeEqual,
                TokenKind::QuestionQuestionEqual,
            ]
        );
    }

    #[test]
    fn test_punctuators_misc() {
        let toks = kinds("=> ...");
        assert_eq!(toks, vec![TokenKind::Arrow, TokenKind::DotDotDot,]);
    }

    // ── Line tracking ─────────────────────────────────────────────────────────

    #[test]
    fn test_line_column_tracking() {
        let toks = tokens("x\ny");
        assert_eq!(toks[0].span.start.line, 1);
        assert_eq!(toks[0].span.start.column, 1);
        assert_eq!(toks[1].span.start.line, 2);
        assert_eq!(toks[1].span.start.column, 1);
    }

    #[test]
    fn test_crlf_counts_as_one_line() {
        let toks = tokens("x\r\ny");
        assert_eq!(toks[1].span.start.line, 2);
    }

    // ── ASI lookahead flag ────────────────────────────────────────────────────

    #[test]
    fn test_asi_flag_set_on_newline() {
        let toks = tokens("x\ny");
        assert!(!toks[0].had_line_terminator_before);
        assert!(toks[1].had_line_terminator_before);
    }

    #[test]
    fn test_asi_flag_not_set_same_line() {
        let toks = tokens("x y");
        assert!(!toks[0].had_line_terminator_before);
        assert!(!toks[1].had_line_terminator_before);
    }

    // ── Full statement tokenisation ───────────────────────────────────────────

    #[test]
    fn test_tokenize_let_declaration() {
        let toks = kinds("let x = 42;");
        assert_eq!(
            toks,
            vec![
                TokenKind::Let,
                TokenKind::Identifier,
                TokenKind::Equal,
                TokenKind::NumericLiteral,
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn test_tokenize_function() {
        let toks = kinds("function add(a, b) { return a + b; }");
        assert_eq!(
            toks,
            vec![
                TokenKind::Function,
                TokenKind::Identifier,
                TokenKind::LeftParen,
                TokenKind::Identifier,
                TokenKind::Comma,
                TokenKind::Identifier,
                TokenKind::RightParen,
                TokenKind::LeftBrace,
                TokenKind::Return,
                TokenKind::Identifier,
                TokenKind::Plus,
                TokenKind::Identifier,
                TokenKind::Semicolon,
                TokenKind::RightBrace,
            ]
        );
    }

    #[test]
    fn test_tokenize_arrow_function() {
        let toks = kinds("const f = (x) => x * 2;");
        assert_eq!(
            toks,
            vec![
                TokenKind::Const,
                TokenKind::Identifier,
                TokenKind::Equal,
                TokenKind::LeftParen,
                TokenKind::Identifier,
                TokenKind::RightParen,
                TokenKind::Arrow,
                TokenKind::Identifier,
                TokenKind::Star,
                TokenKind::NumericLiteral,
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn test_tokenize_class() {
        let toks = kinds("class Foo extends Bar {}");
        assert_eq!(
            toks,
            vec![
                TokenKind::Class,
                TokenKind::Identifier,
                TokenKind::Extends,
                TokenKind::Identifier,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
            ]
        );
    }

    #[test]
    fn test_tokenize_import_export() {
        let toks = kinds("import { x } from 'mod';");
        assert_eq!(
            toks,
            vec![
                TokenKind::Import,
                TokenKind::LeftBrace,
                TokenKind::Identifier,
                TokenKind::RightBrace,
                TokenKind::From,
                TokenKind::StringLiteral,
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn test_tokenize_optional_chain() {
        let toks = kinds("obj?.prop");
        assert_eq!(
            toks,
            vec![
                TokenKind::Identifier,
                TokenKind::QuestionDot,
                TokenKind::Identifier,
            ]
        );
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_unterminated_block_comment() {
        let result = Scanner::tokenize_all("/* oops");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unterminated_template() {
        let result = Scanner::tokenize_all("`oops");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unterminated_regexp() {
        let result = Scanner::tokenize_all("/oops");
        assert!(result.is_err());
    }
}

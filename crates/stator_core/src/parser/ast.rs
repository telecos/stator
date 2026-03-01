//! ES2025 Abstract Syntax Tree node definitions.
//!
//! Every node struct carries a [`SourceLocation`] field (`loc`) that pinpoints
//! its position in the source text.  [`SourceLocation`] is a type alias for
//! [`crate::parser::scanner::Span`] so it is [`Copy`].
//!
//! # Structure
//!
//! - [`Program`] — root node.
//! - [`Stmt`] — statement nodes.
//! - [`Expr`] — expression nodes.
//! - [`Pat`] — binding/assignment pattern nodes.
//! - [`ModuleDecl`] — `import`/`export` module declarations.
//! - Literal types: [`NullLit`], [`BoolLit`], [`NumLit`], [`BigIntLit`],
//!   [`StringLit`], [`RegExpLit`], [`TemplateLit`].

use crate::parser::scanner::Span;

// ─────────────────────────────────────────────────────────────────────────────
// Source location
// ─────────────────────────────────────────────────────────────────────────────

/// Source location attached to every AST node — a half-open `[start, end)`
/// byte span in the source text.
pub type SourceLocation = Span;

// ─────────────────────────────────────────────────────────────────────────────
// Program
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the source file is a classic script or an ES module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    /// A classic `<script>` — top-level `import`/`export` are not allowed.
    Script,
    /// An ES module — top-level `import`/`export` statements are allowed.
    Module,
}

/// The root node of a parsed JavaScript source file.
#[derive(Debug, Clone)]
pub struct Program {
    /// Source location of the entire program.
    pub loc: SourceLocation,
    /// Whether the file is a script or a module.
    pub source_type: SourceType,
    /// Top-level statements and module declarations.
    pub body: Vec<ProgramItem>,
}

/// A top-level item inside a [`Program`].
#[derive(Debug, Clone)]
pub enum ProgramItem {
    /// A regular statement.
    Stmt(Stmt),
    /// A module-level `import` or `export` declaration.
    ModuleDecl(ModuleDecl),
}

// ─────────────────────────────────────────────────────────────────────────────
// Common helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript identifier (name, label, or binding).
#[derive(Debug, Clone)]
pub struct Ident {
    /// Source location.
    pub loc: SourceLocation,
    /// The raw identifier text.
    pub name: String,
}

/// A private identifier beginning with `#` (class fields/methods).
#[derive(Debug, Clone)]
pub struct PrivateIdent {
    /// Source location.
    pub loc: SourceLocation,
    /// The identifier text, **without** the leading `#`.
    pub name: String,
}

/// A function/method parameter (pattern with optional default).
#[derive(Debug, Clone)]
pub struct Param {
    /// Source location.
    pub loc: SourceLocation,
    /// The binding pattern.
    pub pat: Pat,
    /// Default value (`= expr`), if present.
    pub default: Option<Expr>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript statement.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `{ … }` block statement.
    Block(BlockStmt),
    /// `var / let / const` variable declaration.
    VarDecl(VarDecl),
    /// `function` declaration.
    FnDecl(Box<FnDecl>),
    /// `class` declaration.
    ClassDecl(Box<ClassDecl>),
    /// Expression statement (`expr ;`).
    Expr(ExprStmt),
    /// `if (test) consequent else alternate`
    If(IfStmt),
    /// `for (init; test; update) body`
    For(ForStmt),
    /// `for (left in right) body`
    ForIn(ForInStmt),
    /// `for [await] (left of right) body`
    ForOf(ForOfStmt),
    /// `while (test) body`
    While(WhileStmt),
    /// `do body while (test);`
    DoWhile(DoWhileStmt),
    /// `switch (discriminant) { cases }`
    Switch(SwitchStmt),
    /// `try { … } catch (…) { … } finally { … }`
    Try(TryStmt),
    /// `return argument?;`
    Return(ReturnStmt),
    /// `throw argument;`
    Throw(ThrowStmt),
    /// `break label?;`
    Break(BreakStmt),
    /// `continue label?;`
    Continue(ContinueStmt),
    /// `label: body`
    Labeled(LabeledStmt),
    /// `debugger;`
    Debugger(DebuggerStmt),
    /// `with (object) body`
    With(WithStmt),
    /// Empty statement `;`.
    Empty(EmptyStmt),
}

impl Stmt {
    /// Returns the source location of this statement.
    pub fn loc(&self) -> SourceLocation {
        match self {
            Stmt::Block(s) => s.loc,
            Stmt::VarDecl(s) => s.loc,
            Stmt::FnDecl(s) => s.loc,
            Stmt::ClassDecl(s) => s.loc,
            Stmt::Expr(s) => s.loc,
            Stmt::If(s) => s.loc,
            Stmt::For(s) => s.loc,
            Stmt::ForIn(s) => s.loc,
            Stmt::ForOf(s) => s.loc,
            Stmt::While(s) => s.loc,
            Stmt::DoWhile(s) => s.loc,
            Stmt::Switch(s) => s.loc,
            Stmt::Try(s) => s.loc,
            Stmt::Return(s) => s.loc,
            Stmt::Throw(s) => s.loc,
            Stmt::Break(s) => s.loc,
            Stmt::Continue(s) => s.loc,
            Stmt::Labeled(s) => s.loc,
            Stmt::Debugger(s) => s.loc,
            Stmt::With(s) => s.loc,
            Stmt::Empty(s) => s.loc,
        }
    }
}

/// `{ statements }` block statement.
#[derive(Debug, Clone)]
pub struct BlockStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Statements in the block.
    pub body: Vec<Stmt>,
}

/// Expression statement: `expr ;`
#[derive(Debug, Clone)]
pub struct ExprStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The expression.
    pub expr: Box<Expr>,
}

/// `if (test) consequent else alternate`
#[derive(Debug, Clone)]
pub struct IfStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Condition expression.
    pub test: Box<Expr>,
    /// Taken branch.
    pub consequent: Box<Stmt>,
    /// Not-taken branch, if present.
    pub alternate: Option<Box<Stmt>>,
}

/// `for (init; test; update) body`
#[derive(Debug, Clone)]
pub struct ForStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional initializer.
    pub init: Option<ForInit>,
    /// Optional loop condition.
    pub test: Option<Box<Expr>>,
    /// Optional update expression.
    pub update: Option<Box<Expr>>,
    /// Loop body.
    pub body: Box<Stmt>,
}

/// The initializer slot in a C-style `for` statement.
#[derive(Debug, Clone)]
pub enum ForInit {
    /// `var / let / const` declaration.
    VarDecl(VarDecl),
    /// Plain expression.
    Expr(Box<Expr>),
}

/// `for (left in right) body`
#[derive(Debug, Clone)]
pub struct ForInStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Left-hand side binding.
    pub left: ForInOfLeft,
    /// The object being iterated.
    pub right: Box<Expr>,
    /// Loop body.
    pub body: Box<Stmt>,
}

/// `for [await] (left of right) body`
#[derive(Debug, Clone)]
pub struct ForOfStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// `true` for `for await (…)`.
    pub is_await: bool,
    /// Left-hand side binding.
    pub left: ForInOfLeft,
    /// The iterable.
    pub right: Box<Expr>,
    /// Loop body.
    pub body: Box<Stmt>,
}

/// The left-hand side of a `for-in` or `for-of` statement.
#[derive(Debug, Clone)]
pub enum ForInOfLeft {
    /// `var / let / const` declaration.
    VarDecl(VarDecl),
    /// An assignment pattern (destructuring target).
    Pat(Pat),
}

/// `while (test) body`
#[derive(Debug, Clone)]
pub struct WhileStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Loop condition.
    pub test: Box<Expr>,
    /// Loop body.
    pub body: Box<Stmt>,
}

/// `do body while (test);`
#[derive(Debug, Clone)]
pub struct DoWhileStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Loop body.
    pub body: Box<Stmt>,
    /// Loop condition.
    pub test: Box<Expr>,
}

/// `switch (discriminant) { cases }`
#[derive(Debug, Clone)]
pub struct SwitchStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The value being switched on.
    pub discriminant: Box<Expr>,
    /// The `case` / `default` clauses.
    pub cases: Vec<SwitchCase>,
}

/// A single `case expr:` or `default:` clause in a `switch` statement.
#[derive(Debug, Clone)]
pub struct SwitchCase {
    /// Source location.
    pub loc: SourceLocation,
    /// `None` for the `default:` clause; `Some(expr)` for `case expr:`.
    pub test: Option<Expr>,
    /// Body statements for this clause.
    pub consequent: Vec<Stmt>,
}

/// `try { block } catch (param) { handler } finally { finalizer }`
#[derive(Debug, Clone)]
pub struct TryStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The `try` block.
    pub block: BlockStmt,
    /// Optional `catch` clause.
    pub handler: Option<CatchClause>,
    /// Optional `finally` block.
    pub finalizer: Option<BlockStmt>,
}

/// `catch (param) body`
#[derive(Debug, Clone)]
pub struct CatchClause {
    /// Source location.
    pub loc: SourceLocation,
    /// Binding parameter; `None` for optional-catch `catch { … }`.
    pub param: Option<Pat>,
    /// The catch block.
    pub body: BlockStmt,
}

/// `return argument?;`
#[derive(Debug, Clone)]
pub struct ReturnStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional return value.
    pub argument: Option<Box<Expr>>,
}

/// `throw argument;`
#[derive(Debug, Clone)]
pub struct ThrowStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The thrown value.
    pub argument: Box<Expr>,
}

/// `break label?;`
#[derive(Debug, Clone)]
pub struct BreakStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional target label.
    pub label: Option<Ident>,
}

/// `continue label?;`
#[derive(Debug, Clone)]
pub struct ContinueStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional target label.
    pub label: Option<Ident>,
}

/// `label: body`
#[derive(Debug, Clone)]
pub struct LabeledStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The label identifier.
    pub label: Ident,
    /// The labeled statement.
    pub body: Box<Stmt>,
}

/// `debugger;`
#[derive(Debug, Clone)]
pub struct DebuggerStmt {
    /// Source location.
    pub loc: SourceLocation,
}

/// `with (object) body`
#[derive(Debug, Clone)]
pub struct WithStmt {
    /// Source location.
    pub loc: SourceLocation,
    /// The scope object.
    pub object: Box<Expr>,
    /// The body statement.
    pub body: Box<Stmt>,
}

/// Empty statement `;`.
#[derive(Debug, Clone)]
pub struct EmptyStmt {
    /// Source location.
    pub loc: SourceLocation,
}

// ─────────────────────────────────────────────────────────────────────────────
// Variable declarations
// ─────────────────────────────────────────────────────────────────────────────

/// `var / let / const declarators`
#[derive(Debug, Clone)]
pub struct VarDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// Declaration keyword.
    pub kind: VarKind,
    /// One or more declarators.
    pub declarators: Vec<VarDeclarator>,
}

/// Whether a variable declaration uses `var`, `let`, or `const`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    /// `var` — function-scoped.
    Var,
    /// `let` — block-scoped, reassignable.
    Let,
    /// `const` — block-scoped, non-reassignable.
    Const,
}

/// A single `pattern [= initializer]` in a variable declaration.
#[derive(Debug, Clone)]
pub struct VarDeclarator {
    /// Source location.
    pub loc: SourceLocation,
    /// The binding pattern.
    pub id: Pat,
    /// Optional initializer expression.
    pub init: Option<Box<Expr>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Function declarations / expressions
// ─────────────────────────────────────────────────────────────────────────────

/// `function [id] (params) { body }`
#[derive(Debug, Clone)]
pub struct FnDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// Identifier; always `Some` for declarations, may be `None` for
    /// named-function expressions.
    pub id: Option<Ident>,
    /// `true` for `async function`.
    pub is_async: bool,
    /// `true` for generator functions (`function*`).
    pub is_generator: bool,
    /// Parameter list.
    pub params: Vec<Param>,
    /// Function body.
    pub body: BlockStmt,
}

/// `function [id] (params) { body }` used as an expression.
#[derive(Debug, Clone)]
pub struct FnExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional function name.
    pub id: Option<Ident>,
    /// `true` for `async function`.
    pub is_async: bool,
    /// `true` for generator functions (`function*`).
    pub is_generator: bool,
    /// Parameter list.
    pub params: Vec<Param>,
    /// Function body.
    pub body: BlockStmt,
}

/// `[async] (params) => body`
#[derive(Debug, Clone)]
pub struct ArrowExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// `true` for `async (…) => …`.
    pub is_async: bool,
    /// Parameter list.
    pub params: Vec<Param>,
    /// Either a block body `{ … }` or a concise expression body.
    pub body: ArrowBody,
}

/// The body of an arrow function.
#[derive(Debug, Clone)]
pub enum ArrowBody {
    /// `{ statements }` block body.
    Block(BlockStmt),
    /// Concise expression body.
    Expr(Box<Expr>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Class declarations / expressions
// ─────────────────────────────────────────────────────────────────────────────

/// `class [id] [extends superClass] { body }`
#[derive(Debug, Clone)]
pub struct ClassDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// Class name; always `Some` for declarations.
    pub id: Option<Ident>,
    /// Optional super-class expression.
    pub super_class: Option<Box<Expr>>,
    /// Class body.
    pub body: ClassBody,
}

/// `class [id] [extends superClass] { body }` used as an expression.
#[derive(Debug, Clone)]
pub struct ClassExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional class name.
    pub id: Option<Ident>,
    /// Optional super-class expression.
    pub super_class: Option<Box<Expr>>,
    /// Class body.
    pub body: ClassBody,
}

/// The `{ … }` body of a class declaration or expression.
#[derive(Debug, Clone)]
pub struct ClassBody {
    /// Source location.
    pub loc: SourceLocation,
    /// Members: methods, fields, and static blocks.
    pub body: Vec<ClassMember>,
}

/// A member inside a class body.
#[derive(Debug, Clone)]
pub enum ClassMember {
    /// A method definition (`constructor`, regular, getter, or setter).
    Method(MethodDef),
    /// A class field (public or private).
    Property(PropertyDef),
    /// A `static { … }` initialization block.
    StaticBlock(StaticBlock),
}

/// `[static] [get|set] key(params) { body }`
#[derive(Debug, Clone)]
pub struct MethodDef {
    /// Source location.
    pub loc: SourceLocation,
    /// `true` for `static` methods.
    pub is_static: bool,
    /// `constructor`, `method`, `get`, or `set`.
    pub kind: MethodKind,
    /// The property key.
    pub key: PropKey,
    /// `true` when the key is a computed expression `[expr]`.
    pub is_computed: bool,
    /// The method's function value.
    pub value: FnExpr,
}

/// The variant of a class method definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodKind {
    /// `constructor`.
    Constructor,
    /// Regular method.
    Method,
    /// Getter (`get key() { … }`).
    Get,
    /// Setter (`set key(v) { … }`).
    Set,
}

/// `[static] key [= value]` — class field definition.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    /// Source location.
    pub loc: SourceLocation,
    /// `true` for static fields.
    pub is_static: bool,
    /// The field key.
    pub key: PropKey,
    /// `true` when the key is a computed expression `[expr]`.
    pub is_computed: bool,
    /// Optional field initializer.
    pub value: Option<Box<Expr>>,
}

/// `static { … }` initialization block inside a class body.
#[derive(Debug, Clone)]
pub struct StaticBlock {
    /// Source location.
    pub loc: SourceLocation,
    /// Body statements.
    pub body: Vec<Stmt>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Module declarations
// ─────────────────────────────────────────────────────────────────────────────

/// A top-level module declaration (`import` or `export`).
#[derive(Debug, Clone)]
pub enum ModuleDecl {
    /// `import … from "source"`
    Import(ImportDecl),
    /// `export { … } [from "source"]` or `export declaration`
    ExportNamed(ExportNamedDecl),
    /// `export default …`
    ExportDefault(ExportDefaultDecl),
    /// `export * [as name] from "source"`
    ExportAll(ExportAllDecl),
}

/// `import specifiers from "source" [with { … }]`
#[derive(Debug, Clone)]
pub struct ImportDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// The imported bindings.
    pub specifiers: Vec<ImportSpecifier>,
    /// The module specifier string.
    pub source: StringLit,
    /// ES2025 import attributes (`with { key: "value" }`).
    pub attributes: Vec<ImportAttribute>,
}

/// A single binding inside an `import` declaration.
#[derive(Debug, Clone)]
pub enum ImportSpecifier {
    /// `{ imported as local }` or `{ local }`.
    Named(ImportNamedSpecifier),
    /// Default import: `defaultExport`.
    Default(ImportDefaultSpecifier),
    /// Namespace import: `* as ns`.
    Namespace(ImportNamespaceSpecifier),
}

/// `{ imported as local }` — named import specifier.
#[derive(Debug, Clone)]
pub struct ImportNamedSpecifier {
    /// Source location.
    pub loc: SourceLocation,
    /// The name as exported from the module.
    pub imported: ModuleExportName,
    /// The local binding name.
    pub local: Ident,
}

/// `defaultExport` — default import specifier.
#[derive(Debug, Clone)]
pub struct ImportDefaultSpecifier {
    /// Source location.
    pub loc: SourceLocation,
    /// The local binding for the default export.
    pub local: Ident,
}

/// `* as ns` — namespace import specifier.
#[derive(Debug, Clone)]
pub struct ImportNamespaceSpecifier {
    /// Source location.
    pub loc: SourceLocation,
    /// The local namespace binding.
    pub local: Ident,
}

/// A single key/value pair in an `import … with { … }` attribute clause.
#[derive(Debug, Clone)]
pub struct ImportAttribute {
    /// Source location.
    pub loc: SourceLocation,
    /// The attribute key.
    pub key: Ident,
    /// The attribute value (always a string literal).
    pub value: StringLit,
}

/// `export { specifiers } [from "source"] [with { … }]` or `export declaration`
#[derive(Debug, Clone)]
pub struct ExportNamedDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// Named export specifiers.
    pub specifiers: Vec<ExportSpecifier>,
    /// Re-export source, if present.
    pub source: Option<StringLit>,
    /// Inline declaration (`export function f() { … }`, etc.).
    pub declaration: Option<Box<Stmt>>,
    /// ES2025 import attributes on re-exports.
    pub attributes: Vec<ImportAttribute>,
}

/// `{ local as exported }` — named export specifier.
#[derive(Debug, Clone)]
pub struct ExportSpecifier {
    /// Source location.
    pub loc: SourceLocation,
    /// The local (or re-exported) name.
    pub local: ModuleExportName,
    /// The exported name.
    pub exported: ModuleExportName,
}

/// `export default expression | function | class`
#[derive(Debug, Clone)]
pub struct ExportDefaultDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// The exported default value.
    pub declaration: ExportDefaultExpr,
}

/// The exported item in an `export default …` declaration.
#[derive(Debug, Clone)]
pub enum ExportDefaultExpr {
    /// `export default function [id] (…) { … }`
    Fn(Box<FnDecl>),
    /// `export default class [id] { … }`
    Class(Box<ClassDecl>),
    /// `export default expr`
    Expr(Box<Expr>),
}

/// `export * [as name] from "source" [with { … }]`
#[derive(Debug, Clone)]
pub struct ExportAllDecl {
    /// Source location.
    pub loc: SourceLocation,
    /// Optional re-export alias (`as name`).
    pub exported: Option<ModuleExportName>,
    /// The source module specifier.
    pub source: StringLit,
    /// ES2025 import attributes.
    pub attributes: Vec<ImportAttribute>,
}

/// An identifier or string literal used as a module export/import name.
#[derive(Debug, Clone)]
pub enum ModuleExportName {
    /// Plain identifier.
    Ident(Ident),
    /// String literal (allows non-identifier export names).
    Str(StringLit),
}

// ─────────────────────────────────────────────────────────────────────────────
// Literals
// ─────────────────────────────────────────────────────────────────────────────

/// `null` literal.
#[derive(Debug, Clone)]
pub struct NullLit {
    /// Source location.
    pub loc: SourceLocation,
}

/// `true` or `false` literal.
#[derive(Debug, Clone)]
pub struct BoolLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The boolean value.
    pub value: bool,
}

/// Numeric literal (decimal, hex, binary, octal, or BigInt).
#[derive(Debug, Clone)]
pub struct NumLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The parsed numeric value.
    pub value: f64,
    /// The raw source text.
    pub raw: String,
}

/// BigInt literal (e.g. `42n`).
#[derive(Debug, Clone)]
pub struct BigIntLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The numeric part as a string (no trailing `n`).
    pub value: String,
}

/// String literal.
#[derive(Debug, Clone)]
pub struct StringLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The decoded string value.
    pub value: String,
}

/// Regular-expression literal `/pattern/flags`.
#[derive(Debug, Clone)]
pub struct RegExpLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The pattern string (between the slashes).
    pub pattern: String,
    /// The flag characters (after the closing slash).
    pub flags: String,
}

/// `` `quasis ${expressions} quasis` `` — template literal.
#[derive(Debug, Clone)]
pub struct TemplateLit {
    /// Source location.
    pub loc: SourceLocation,
    /// The string parts (one more than `expressions`).
    pub quasis: Vec<TemplateElement>,
    /// The interpolated expressions.
    pub expressions: Vec<Expr>,
}

/// A static string fragment inside a template literal.
#[derive(Debug, Clone)]
pub struct TemplateElement {
    /// Source location.
    pub loc: SourceLocation,
    /// Raw source text of this fragment (backslashes not interpreted).
    pub raw: String,
    /// Cooked (decoded) value; `None` if the fragment has an invalid escape.
    pub cooked: Option<String>,
    /// `true` for the final quasi (at the end of the template).
    pub tail: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────────────────────

/// A JavaScript expression.
#[derive(Debug, Clone)]
pub enum Expr {
    // ── Literals ──────────────────────────────────────────────────────────
    /// `null`
    Null(NullLit),
    /// `true` / `false`
    Bool(BoolLit),
    /// Numeric literal.
    Num(NumLit),
    /// String literal.
    Str(StringLit),
    /// BigInt literal (`42n`).
    BigInt(BigIntLit),
    /// Regular-expression literal.
    Regexp(RegExpLit),
    /// Template literal.
    Template(Box<TemplateLit>),

    // ── Primary ───────────────────────────────────────────────────────────
    /// Plain identifier.
    Ident(Ident),
    /// `this`
    This(ThisExpr),
    /// Array literal `[elements]`.
    Array(Box<ArrayExpr>),
    /// Object literal `{ properties }`.
    Object(Box<ObjectExpr>),

    // ── Function-like ─────────────────────────────────────────────────────
    /// Function expression.
    Fn(Box<FnExpr>),
    /// Arrow function expression.
    Arrow(Box<ArrowExpr>),
    /// Class expression.
    Class(Box<ClassExpr>),

    // ── Operators ─────────────────────────────────────────────────────────
    /// Unary prefix/postfix operator.
    Unary(Box<UnaryExpr>),
    /// `++` / `--` update expression.
    Update(Box<UpdateExpr>),
    /// Binary infix operator.
    Binary(Box<BinaryExpr>),
    /// Logical `&&` / `||` / `??` operator.
    Logical(Box<LogicalExpr>),
    /// `test ? consequent : alternate`
    Conditional(Box<ConditionalExpr>),
    /// Assignment expression (`=`, `+=`, …).
    Assign(Box<AssignExpr>),
    /// Comma-separated sequence `(a, b, c)`.
    Sequence(Box<SequenceExpr>),

    // ── Member / call ─────────────────────────────────────────────────────
    /// `object.property` / `object[expr]`
    Member(Box<MemberExpr>),
    /// `object?.property` / `object?.[expr]`
    OptionalMember(Box<OptionalMemberExpr>),
    /// `callee(args)`
    Call(Box<CallExpr>),
    /// `callee?.(args)`
    OptionalCall(Box<OptionalCallExpr>),
    /// `new callee(args)`
    New(Box<NewExpr>),

    // ── Template ──────────────────────────────────────────────────────────
    /// `` tag`template` ``
    TaggedTemplate(Box<TaggedTemplateExpr>),

    // ── Spread ────────────────────────────────────────────────────────────
    /// `...argument` inside an array literal or function call.
    Spread(Box<SpreadElement>),

    // ── Async / generator ─────────────────────────────────────────────────
    /// `yield [*] [argument]`
    Yield(Box<YieldExpr>),
    /// `await argument`
    Await(Box<AwaitExpr>),

    // ── Dynamic import / meta ─────────────────────────────────────────────
    /// `import(source)`
    Import(Box<ImportExpr>),
    /// `import.meta` or `new.target`
    MetaProp(MetaPropExpr),
}

impl Expr {
    /// Returns the source location of this expression.
    pub fn loc(&self) -> SourceLocation {
        match self {
            Expr::Null(e) => e.loc,
            Expr::Bool(e) => e.loc,
            Expr::Num(e) => e.loc,
            Expr::Str(e) => e.loc,
            Expr::BigInt(e) => e.loc,
            Expr::Regexp(e) => e.loc,
            Expr::Template(e) => e.loc,
            Expr::Ident(e) => e.loc,
            Expr::This(e) => e.loc,
            Expr::Array(e) => e.loc,
            Expr::Object(e) => e.loc,
            Expr::Fn(e) => e.loc,
            Expr::Arrow(e) => e.loc,
            Expr::Class(e) => e.loc,
            Expr::Unary(e) => e.loc,
            Expr::Update(e) => e.loc,
            Expr::Binary(e) => e.loc,
            Expr::Logical(e) => e.loc,
            Expr::Conditional(e) => e.loc,
            Expr::Assign(e) => e.loc,
            Expr::Sequence(e) => e.loc,
            Expr::Member(e) => e.loc,
            Expr::OptionalMember(e) => e.loc,
            Expr::Call(e) => e.loc,
            Expr::OptionalCall(e) => e.loc,
            Expr::New(e) => e.loc,
            Expr::TaggedTemplate(e) => e.loc,
            Expr::Spread(e) => e.loc,
            Expr::Yield(e) => e.loc,
            Expr::Await(e) => e.loc,
            Expr::Import(e) => e.loc,
            Expr::MetaProp(e) => e.loc,
        }
    }
}

/// `this`
#[derive(Debug, Clone)]
pub struct ThisExpr {
    /// Source location.
    pub loc: SourceLocation,
}

/// Array literal: `[elements]`.
#[derive(Debug, Clone)]
pub struct ArrayExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// Elements, where `None` represents an elision (`,`).
    pub elements: Vec<Option<Expr>>,
}

/// Object literal: `{ properties }`.
#[derive(Debug, Clone)]
pub struct ObjectExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// Property list.
    pub properties: Vec<ObjectProp>,
}

/// A single property (or spread) in an object literal.
#[derive(Debug, Clone)]
pub enum ObjectProp {
    /// `key: value`, shorthand, getter/setter, or method.
    Prop(Box<Prop>),
    /// `...expr` spread property.
    Spread(SpreadElement),
}

/// A concrete property in an object literal.
#[derive(Debug, Clone)]
pub struct Prop {
    /// Source location.
    pub loc: SourceLocation,
    /// The property key.
    pub key: PropKey,
    /// `true` when the key is a computed expression `[expr]`.
    pub is_computed: bool,
    /// The value/kind of this property.
    pub value: PropValue,
}

/// The key in an object property or class member.
#[derive(Debug, Clone)]
pub enum PropKey {
    /// Identifier key.
    Ident(Ident),
    /// Private identifier key (`#name`).
    Private(PrivateIdent),
    /// String literal key.
    Str(StringLit),
    /// Numeric literal key.
    Num(NumLit),
    /// Computed key `[expr]`.
    Computed(Box<Expr>),
}

/// The value of a property in an object literal.
#[derive(Debug, Clone)]
pub enum PropValue {
    /// `key: value` — standard property.
    Value(Box<Expr>),
    /// `{ key }` — shorthand property (key and value have the same name).
    Shorthand,
    /// `get key() { … }` — getter.
    Get(FnExpr),
    /// `set key(v) { … }` — setter.
    Set(FnExpr),
    /// `key(params) { … }` — method.
    Method(FnExpr),
}

/// `...argument` — spread element in array literals or function calls.
#[derive(Debug, Clone)]
pub struct SpreadElement {
    /// Source location.
    pub loc: SourceLocation,
    /// The spread argument.
    pub argument: Box<Expr>,
}

/// Unary expression: `op argument` (prefix) or `argument op` (postfix).
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The unary operator.
    pub op: UnaryOp,
    /// The operand.
    pub argument: Box<Expr>,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-`
    Minus,
    /// `+`
    Plus,
    /// `!`
    Not,
    /// `~`
    BitNot,
    /// `typeof`
    Typeof,
    /// `void`
    Void,
    /// `delete`
    Delete,
}

/// `++` / `--` update expression.
#[derive(Debug, Clone)]
pub struct UpdateExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// `++` or `--`.
    pub op: UpdateOp,
    /// `true` for prefix, `false` for postfix.
    pub prefix: bool,
    /// The operand (must be an l-value).
    pub argument: Box<Expr>,
}

/// The increment / decrement operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOp {
    /// `++`
    Increment,
    /// `--`
    Decrement,
}

/// Binary infix expression: `left op right`.
#[derive(Debug, Clone)]
pub struct BinaryExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The binary operator.
    pub op: BinaryOp,
    /// Left operand.
    pub left: Box<Expr>,
    /// Right operand.
    pub right: Box<Expr>,
}

/// A binary (non-assignment, non-logical) infix operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `==`
    Eq,
    /// `!=`
    NotEq,
    /// `===`
    StrictEq,
    /// `!==`
    StrictNotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `>>>`
    UShr,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `**`
    Exp,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
    /// `in`
    In,
    /// `instanceof`
    Instanceof,
}

/// Logical short-circuit expression: `left op right`.
#[derive(Debug, Clone)]
pub struct LogicalExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The logical operator.
    pub op: LogicalOp,
    /// Left operand.
    pub left: Box<Expr>,
    /// Right operand.
    pub right: Box<Expr>,
}

/// A logical (short-circuit) operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    /// `&&`
    And,
    /// `||`
    Or,
    /// `??`
    NullishCoalesce,
}

/// `test ? consequent : alternate`
#[derive(Debug, Clone)]
pub struct ConditionalExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The condition.
    pub test: Box<Expr>,
    /// Taken branch.
    pub consequent: Box<Expr>,
    /// Not-taken branch.
    pub alternate: Box<Expr>,
}

/// Assignment expression: `left op right`.
#[derive(Debug, Clone)]
pub struct AssignExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The assignment operator.
    pub op: AssignOp,
    /// Left-hand side (binding pattern or l-value expression).
    pub left: AssignTarget,
    /// Right-hand side.
    pub right: Box<Expr>,
}

/// The left-hand side of an assignment expression.
#[derive(Debug, Clone)]
pub enum AssignTarget {
    /// Simple l-value expression.
    Expr(Box<Expr>),
    /// Destructuring pattern.
    Pat(Pat),
}

/// An assignment operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
    /// `%=`
    RemAssign,
    /// `**=`
    ExpAssign,
    /// `<<=`
    ShlAssign,
    /// `>>=`
    ShrAssign,
    /// `>>>=`
    UShrAssign,
    /// `|=`
    BitOrAssign,
    /// `^=`
    BitXorAssign,
    /// `&=`
    BitAndAssign,
    /// `&&=`
    LogicalAndAssign,
    /// `||=`
    LogicalOrAssign,
    /// `??=`
    NullishAssign,
}

/// Comma-separated sequence expression: `(a, b, c)`.
#[derive(Debug, Clone)]
pub struct SequenceExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The expressions in order.
    pub expressions: Vec<Expr>,
}

/// `object.property` or `object[expr]`
#[derive(Debug, Clone)]
pub struct MemberExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The object.
    pub object: Box<Expr>,
    /// The property key.
    pub property: MemberProp,
    /// `true` for computed access `object[expr]`.
    pub is_computed: bool,
}

/// `object?.property` or `object?.[expr]`
#[derive(Debug, Clone)]
pub struct OptionalMemberExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The object.
    pub object: Box<Expr>,
    /// The property key.
    pub property: MemberProp,
    /// `true` for computed access `object?.[expr]`.
    pub is_computed: bool,
}

/// The property part of a member expression.
#[derive(Debug, Clone)]
pub enum MemberProp {
    /// Static identifier (`.name`).
    Ident(Ident),
    /// Private identifier (`#name`).
    Private(PrivateIdent),
    /// Computed expression (`[expr]`).
    Computed(Box<Expr>),
}

/// `callee(arguments)`
#[derive(Debug, Clone)]
pub struct CallExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The function being called.
    pub callee: Box<Expr>,
    /// Argument list.
    pub arguments: Vec<Expr>,
}

/// `callee?.(arguments)` — optional call expression.
#[derive(Debug, Clone)]
pub struct OptionalCallExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The function being called.
    pub callee: Box<Expr>,
    /// Argument list.
    pub arguments: Vec<Expr>,
}

/// `new callee(arguments)`
#[derive(Debug, Clone)]
pub struct NewExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The constructor.
    pub callee: Box<Expr>,
    /// Argument list.
    pub arguments: Vec<Expr>,
}

/// `` tag`template` `` — tagged template expression.
#[derive(Debug, Clone)]
pub struct TaggedTemplateExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The tag function.
    pub tag: Box<Expr>,
    /// The template literal.
    pub quasi: TemplateLit,
}

/// `yield [*] [argument]` — yield expression.
#[derive(Debug, Clone)]
pub struct YieldExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// `true` for `yield*` (delegating yield).
    pub delegate: bool,
    /// Optional value to yield.
    pub argument: Option<Box<Expr>>,
}

/// `await argument` — await expression.
#[derive(Debug, Clone)]
pub struct AwaitExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The awaited value.
    pub argument: Box<Expr>,
}

/// `import(source)` — dynamic import expression.
#[derive(Debug, Clone)]
pub struct ImportExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The module specifier.
    pub source: Box<Expr>,
    /// ES2025 import attributes (`with { … }`).
    pub options: Option<Box<Expr>>,
}

/// `import.meta` or `new.target` — meta property expression.
#[derive(Debug, Clone)]
pub struct MetaPropExpr {
    /// Source location.
    pub loc: SourceLocation,
    /// The meta object (`import` or `new`).
    pub meta: Ident,
    /// The property name (`meta` or `target`).
    pub property: Ident,
}

// ─────────────────────────────────────────────────────────────────────────────
// Patterns
// ─────────────────────────────────────────────────────────────────────────────

/// A binding or assignment pattern.
#[derive(Debug, Clone)]
pub enum Pat {
    /// Simple identifier binding.
    Ident(Ident),
    /// Array destructuring `[a, b, ...rest]`.
    Array(Box<ArrayPat>),
    /// Object destructuring `{ a, b: c, ...rest }`.
    Object(Box<ObjectPat>),
    /// Rest element `...pattern`.
    Rest(Box<RestElement>),
    /// Default-value binding `pattern = default`.
    Assign(Box<AssignPat>),
}

impl Pat {
    /// Returns the source location of this pattern.
    pub fn loc(&self) -> SourceLocation {
        match self {
            Pat::Ident(p) => p.loc,
            Pat::Array(p) => p.loc,
            Pat::Object(p) => p.loc,
            Pat::Rest(p) => p.loc,
            Pat::Assign(p) => p.loc,
        }
    }
}

/// Array destructuring pattern: `[a, , b, ...rest]`.
#[derive(Debug, Clone)]
pub struct ArrayPat {
    /// Source location.
    pub loc: SourceLocation,
    /// Elements, where `None` represents an elision.
    pub elements: Vec<Option<Pat>>,
}

/// Object destructuring pattern: `{ a, b: c, ...rest }`.
#[derive(Debug, Clone)]
pub struct ObjectPat {
    /// Source location.
    pub loc: SourceLocation,
    /// Property patterns.
    pub properties: Vec<ObjectPatProp>,
}

/// A single property inside an object destructuring pattern.
#[derive(Debug, Clone)]
pub enum ObjectPatProp {
    /// `key: pat [= default]` — key-value property.
    KeyValue(KeyValuePatProp),
    /// `id [= default]` — shorthand property (key == binding name).
    Assign(AssignPatProp),
    /// `...rest` — rest property.
    Rest(RestElement),
}

/// `key: pattern [= default]` in an object pattern.
#[derive(Debug, Clone)]
pub struct KeyValuePatProp {
    /// Source location.
    pub loc: SourceLocation,
    /// The property key.
    pub key: PropKey,
    /// `true` when the key is a computed expression.
    pub is_computed: bool,
    /// The value pattern.
    pub value: Pat,
}

/// Shorthand `{ id [= default] }` in an object pattern.
#[derive(Debug, Clone)]
pub struct AssignPatProp {
    /// Source location.
    pub loc: SourceLocation,
    /// The identifier used as both key and binding.
    pub key: Ident,
    /// Optional default value.
    pub value: Option<Box<Expr>>,
}

/// `...pattern` — rest element in array/object patterns or parameter lists.
#[derive(Debug, Clone)]
pub struct RestElement {
    /// Source location.
    pub loc: SourceLocation,
    /// The rest binding target.
    pub argument: Box<Pat>,
}

/// `pattern = default` — default-value pattern.
#[derive(Debug, Clone)]
pub struct AssignPat {
    /// Source location.
    pub loc: SourceLocation,
    /// The binding pattern.
    pub left: Box<Pat>,
    /// The default expression.
    pub right: Box<Expr>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::scanner::{Position, Span};

    fn dummy_loc() -> SourceLocation {
        Span {
            start: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
            end: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
        }
    }

    // ── SourceType ────────────────────────────────────────────────────────────

    #[test]
    fn test_source_type_variants() {
        assert_ne!(SourceType::Script, SourceType::Module);
        assert_eq!(SourceType::Script, SourceType::Script);
    }

    // ── Program ───────────────────────────────────────────────────────────────

    #[test]
    fn test_program_empty_script() {
        let prog = Program {
            loc: dummy_loc(),
            source_type: SourceType::Script,
            body: vec![],
        };
        assert!(prog.body.is_empty());
        assert_eq!(prog.source_type, SourceType::Script);
    }

    #[test]
    fn test_program_module_with_item() {
        let stmt = Stmt::Empty(EmptyStmt { loc: dummy_loc() });
        let prog = Program {
            loc: dummy_loc(),
            source_type: SourceType::Module,
            body: vec![ProgramItem::Stmt(stmt)],
        };
        assert_eq!(prog.body.len(), 1);
    }

    // ── Statements ────────────────────────────────────────────────────────────

    #[test]
    fn test_stmt_loc_block() {
        let s = Stmt::Block(BlockStmt {
            loc: dummy_loc(),
            body: vec![],
        });
        let _ = s.loc();
    }

    #[test]
    fn test_stmt_loc_all_variants() {
        let loc = dummy_loc();
        let ident = || Ident {
            loc,
            name: "x".into(),
        };
        let expr = || Box::new(Expr::Null(NullLit { loc }));
        let stmts: Vec<Stmt> = vec![
            Stmt::Block(BlockStmt { loc, body: vec![] }),
            Stmt::VarDecl(VarDecl {
                loc,
                kind: VarKind::Let,
                declarators: vec![],
            }),
            Stmt::FnDecl(Box::new(FnDecl {
                loc,
                id: Some(ident()),
                is_async: false,
                is_generator: false,
                params: vec![],
                body: BlockStmt { loc, body: vec![] },
            })),
            Stmt::ClassDecl(Box::new(ClassDecl {
                loc,
                id: Some(ident()),
                super_class: None,
                body: ClassBody { loc, body: vec![] },
            })),
            Stmt::Expr(ExprStmt { loc, expr: expr() }),
            Stmt::If(IfStmt {
                loc,
                test: expr(),
                consequent: Box::new(Stmt::Empty(EmptyStmt { loc })),
                alternate: None,
            }),
            Stmt::For(ForStmt {
                loc,
                init: None,
                test: None,
                update: None,
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::ForIn(ForInStmt {
                loc,
                left: ForInOfLeft::Pat(Pat::Ident(ident())),
                right: expr(),
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::ForOf(ForOfStmt {
                loc,
                is_await: false,
                left: ForInOfLeft::Pat(Pat::Ident(ident())),
                right: expr(),
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::While(WhileStmt {
                loc,
                test: expr(),
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::DoWhile(DoWhileStmt {
                loc,
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
                test: expr(),
            }),
            Stmt::Switch(SwitchStmt {
                loc,
                discriminant: expr(),
                cases: vec![],
            }),
            Stmt::Try(TryStmt {
                loc,
                block: BlockStmt { loc, body: vec![] },
                handler: None,
                finalizer: None,
            }),
            Stmt::Return(ReturnStmt {
                loc,
                argument: None,
            }),
            Stmt::Throw(ThrowStmt {
                loc,
                argument: expr(),
            }),
            Stmt::Break(BreakStmt { loc, label: None }),
            Stmt::Continue(ContinueStmt { loc, label: None }),
            Stmt::Labeled(LabeledStmt {
                loc,
                label: ident(),
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::Debugger(DebuggerStmt { loc }),
            Stmt::With(WithStmt {
                loc,
                object: expr(),
                body: Box::new(Stmt::Empty(EmptyStmt { loc })),
            }),
            Stmt::Empty(EmptyStmt { loc }),
        ];
        for s in &stmts {
            // Just ensure loc() doesn't panic for each variant.
            let _ = s.loc();
        }
    }

    // ── VarDecl ───────────────────────────────────────────────────────────────

    #[test]
    fn test_var_decl_kinds() {
        for kind in [VarKind::Var, VarKind::Let, VarKind::Const] {
            let d = VarDecl {
                loc: dummy_loc(),
                kind,
                declarators: vec![],
            };
            assert_eq!(d.kind, kind);
        }
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    #[test]
    fn test_expr_loc_null() {
        let e = Expr::Null(NullLit { loc: dummy_loc() });
        let _ = e.loc();
    }

    #[test]
    fn test_expr_bool_values() {
        let t = Expr::Bool(BoolLit {
            loc: dummy_loc(),
            value: true,
        });
        let f = Expr::Bool(BoolLit {
            loc: dummy_loc(),
            value: false,
        });
        if let Expr::Bool(b) = &t {
            assert!(b.value);
        }
        if let Expr::Bool(b) = &f {
            assert!(!b.value);
        }
    }

    #[test]
    fn test_expr_loc_all_variants() {
        let loc = dummy_loc();
        let ident = Ident {
            loc,
            name: "x".into(),
        };
        let null = || Box::new(Expr::Null(NullLit { loc }));
        let exprs: Vec<Expr> = vec![
            Expr::Null(NullLit { loc }),
            Expr::Bool(BoolLit { loc, value: true }),
            Expr::Num(NumLit {
                loc,
                value: 1.0,
                raw: "1".into(),
            }),
            Expr::Str(StringLit {
                loc,
                value: "s".into(),
            }),
            Expr::BigInt(BigIntLit {
                loc,
                value: "9007199254740993".into(),
            }),
            Expr::Regexp(RegExpLit {
                loc,
                pattern: "x".into(),
                flags: "g".into(),
            }),
            Expr::Template(Box::new(TemplateLit {
                loc,
                quasis: vec![],
                expressions: vec![],
            })),
            Expr::Ident(ident.clone()),
            Expr::This(ThisExpr { loc }),
            Expr::Array(Box::new(ArrayExpr {
                loc,
                elements: vec![],
            })),
            Expr::Object(Box::new(ObjectExpr {
                loc,
                properties: vec![],
            })),
            Expr::Fn(Box::new(FnExpr {
                loc,
                id: None,
                is_async: false,
                is_generator: false,
                params: vec![],
                body: BlockStmt { loc, body: vec![] },
            })),
            Expr::Arrow(Box::new(ArrowExpr {
                loc,
                is_async: false,
                params: vec![],
                body: ArrowBody::Expr(null()),
            })),
            Expr::Class(Box::new(ClassExpr {
                loc,
                id: None,
                super_class: None,
                body: ClassBody { loc, body: vec![] },
            })),
            Expr::Unary(Box::new(UnaryExpr {
                loc,
                op: UnaryOp::Not,
                argument: null(),
            })),
            Expr::Update(Box::new(UpdateExpr {
                loc,
                op: UpdateOp::Increment,
                prefix: true,
                argument: null(),
            })),
            Expr::Binary(Box::new(BinaryExpr {
                loc,
                op: BinaryOp::Add,
                left: null(),
                right: null(),
            })),
            Expr::Logical(Box::new(LogicalExpr {
                loc,
                op: LogicalOp::And,
                left: null(),
                right: null(),
            })),
            Expr::Conditional(Box::new(ConditionalExpr {
                loc,
                test: null(),
                consequent: null(),
                alternate: null(),
            })),
            Expr::Assign(Box::new(AssignExpr {
                loc,
                op: AssignOp::Assign,
                left: AssignTarget::Expr(null()),
                right: null(),
            })),
            Expr::Sequence(Box::new(SequenceExpr {
                loc,
                expressions: vec![],
            })),
            Expr::Member(Box::new(MemberExpr {
                loc,
                object: null(),
                property: MemberProp::Ident(ident.clone()),
                is_computed: false,
            })),
            Expr::OptionalMember(Box::new(OptionalMemberExpr {
                loc,
                object: null(),
                property: MemberProp::Ident(ident.clone()),
                is_computed: false,
            })),
            Expr::Call(Box::new(CallExpr {
                loc,
                callee: null(),
                arguments: vec![],
            })),
            Expr::OptionalCall(Box::new(OptionalCallExpr {
                loc,
                callee: null(),
                arguments: vec![],
            })),
            Expr::New(Box::new(NewExpr {
                loc,
                callee: null(),
                arguments: vec![],
            })),
            Expr::TaggedTemplate(Box::new(TaggedTemplateExpr {
                loc,
                tag: null(),
                quasi: TemplateLit {
                    loc,
                    quasis: vec![],
                    expressions: vec![],
                },
            })),
            Expr::Spread(Box::new(SpreadElement {
                loc,
                argument: null(),
            })),
            Expr::Yield(Box::new(YieldExpr {
                loc,
                delegate: false,
                argument: None,
            })),
            Expr::Await(Box::new(AwaitExpr {
                loc,
                argument: null(),
            })),
            Expr::Import(Box::new(ImportExpr {
                loc,
                source: null(),
                options: None,
            })),
            Expr::MetaProp(MetaPropExpr {
                loc,
                meta: ident.clone(),
                property: ident.clone(),
            }),
        ];
        for e in &exprs {
            let _ = e.loc();
        }
    }

    // ── Patterns ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pat_loc_all_variants() {
        let loc = dummy_loc();
        let ident = Ident {
            loc,
            name: "x".into(),
        };
        let pats: Vec<Pat> = vec![
            Pat::Ident(ident.clone()),
            Pat::Array(Box::new(ArrayPat {
                loc,
                elements: vec![],
            })),
            Pat::Object(Box::new(ObjectPat {
                loc,
                properties: vec![],
            })),
            Pat::Rest(Box::new(RestElement {
                loc,
                argument: Box::new(Pat::Ident(ident.clone())),
            })),
            Pat::Assign(Box::new(AssignPat {
                loc,
                left: Box::new(Pat::Ident(ident.clone())),
                right: Box::new(Expr::Null(NullLit { loc })),
            })),
        ];
        for p in &pats {
            let _ = p.loc();
        }
    }

    // ── Module declarations ───────────────────────────────────────────────────

    #[test]
    fn test_import_decl_construction() {
        let loc = dummy_loc();
        let decl = ImportDecl {
            loc,
            specifiers: vec![],
            source: StringLit {
                loc,
                value: "./foo.js".into(),
            },
            attributes: vec![],
        };
        assert_eq!(decl.source.value, "./foo.js");
    }

    #[test]
    fn test_export_all_decl() {
        let loc = dummy_loc();
        let decl = ExportAllDecl {
            loc,
            exported: None,
            source: StringLit {
                loc,
                value: "./bar.js".into(),
            },
            attributes: vec![],
        };
        assert!(decl.exported.is_none());
    }

    // ── Operators ─────────────────────────────────────────────────────────────

    #[test]
    fn test_binary_op_variants() {
        let ops = [
            BinaryOp::Eq,
            BinaryOp::NotEq,
            BinaryOp::StrictEq,
            BinaryOp::StrictNotEq,
            BinaryOp::Lt,
            BinaryOp::LtEq,
            BinaryOp::Gt,
            BinaryOp::GtEq,
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
            BinaryOp::Exp,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::BitAnd,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::UShr,
            BinaryOp::In,
            BinaryOp::Instanceof,
        ];
        assert_eq!(ops.len(), 22);
    }

    #[test]
    fn test_assign_op_variants() {
        let ops = [
            AssignOp::Assign,
            AssignOp::AddAssign,
            AssignOp::SubAssign,
            AssignOp::MulAssign,
            AssignOp::DivAssign,
            AssignOp::RemAssign,
            AssignOp::ExpAssign,
            AssignOp::ShlAssign,
            AssignOp::ShrAssign,
            AssignOp::UShrAssign,
            AssignOp::BitOrAssign,
            AssignOp::BitXorAssign,
            AssignOp::BitAndAssign,
            AssignOp::LogicalAndAssign,
            AssignOp::LogicalOrAssign,
            AssignOp::NullishAssign,
        ];
        assert_eq!(ops.len(), 16);
    }
}

//! Scope analysis and variable resolution for ES2025 JavaScript.
//!
//! # Overview
//!
//! Call [`analyze`] to build a [`ScopeTree`] from a parsed [`Program`].  The
//! tree records every [`Scope`] (with its [`ScopeKind`] and [`Binding`]
//! declarations), resolves identifier references across the entire program, and
//! reports:
//!
//! - **Duplicate bindings** — two `let`/`const`/`class` declarations for the
//!   same name in the same block scope, or a lexical binding that shadows a
//!   `var` in the same immediately-enclosing function scope.
//! - **TDZ violations** — access to a `let`/`const`/`class` binding before
//!   its declaration is reached in the source text.
//! - **Closure captures** — variables declared in an outer function scope and
//!   referenced from an inner function scope.
//!
//! # Usage flags
//!
//! Each [`Scope`] records whether the code it contains references:
//! - `uses_arguments` — the `arguments` pseudo-array.
//! - `uses_eval` — a direct call to `eval(…)`.
//! - `uses_this` — a `this` expression.
//! - `uses_super` — a `super` member or call expression.

use std::collections::{HashMap, HashSet};

use crate::parser::ast::{
    ArrowBody, AssignTarget, BlockStmt, CatchClause, ClassBody, ClassDecl, ClassExpr, ClassMember,
    Expr, FnDecl, FnExpr, ForInOfLeft, ForInit, Ident, ModuleDecl, ObjectPatProp, Param, Pat,
    Program, ProgramItem, SourceLocation, Stmt, VarDecl, VarKind,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public identifier types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a [`Scope`] in the [`ScopeTree`].
pub type ScopeId = usize;

// ─────────────────────────────────────────────────────────────────────────────
// Scope kind
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of a JavaScript scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// The implicit outermost scope of a classic `<script>`.
    Global,
    /// The scope created by a `function` declaration/expression.
    Function,
    /// A `{ … }` block scope (`if`, `for`, bare block, …).
    Block,
    /// The implicit outermost scope of an ES module.
    Module,
    /// The scope of an indirect or direct `eval(…)` call.
    Eval,
    /// The scope introduced by a `with (obj) { … }` statement.
    With,
    /// The scope created by a `catch (param) { … }` clause.
    Catch,
}

impl ScopeKind {
    /// Returns `true` if this kind creates a new *function* scope boundary
    /// (i.e. `var` declarations are hoisted here and `this`/`arguments` are
    /// fresh).
    pub fn is_function_boundary(self) -> bool {
        matches!(
            self,
            ScopeKind::Global | ScopeKind::Function | ScopeKind::Module | ScopeKind::Eval
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Binding kind
// ─────────────────────────────────────────────────────────────────────────────

/// The syntactic origin of a [`Binding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    /// `var` declaration — function-scoped, hoisted and initialised to
    /// `undefined`.
    Var,
    /// `let` declaration — block-scoped, TDZ-guarded.
    Let,
    /// `const` declaration — block-scoped, TDZ-guarded, non-reassignable.
    Const,
    /// `function` declaration — hoisted to the enclosing function scope.
    Function,
    /// `class` declaration — block-scoped, TDZ-guarded.
    Class,
    /// A formal parameter of a function.
    Param,
    /// A binding introduced by an `import` declaration.
    Import,
}

impl BindingKind {
    /// Returns `true` for bindings that are subject to the Temporal Dead Zone
    /// (`let`, `const`, `class`).
    pub fn has_tdz(self) -> bool {
        matches!(
            self,
            BindingKind::Let | BindingKind::Const | BindingKind::Class
        )
    }

    /// Returns `true` if the binding is hoisted to the enclosing *function*
    /// scope (`var` and `function`).
    pub fn is_function_scoped(self) -> bool {
        matches!(self, BindingKind::Var | BindingKind::Function)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Binding
// ─────────────────────────────────────────────────────────────────────────────

/// A single name declared in a [`Scope`].
#[derive(Debug, Clone)]
pub struct Binding {
    /// The identifier text.
    pub name: String,
    /// Syntactic origin of this binding.
    pub kind: BindingKind,
    /// Source location of the declaration.
    pub loc: SourceLocation,
    /// The scope that owns this binding.
    pub scope_id: ScopeId,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scope
// ─────────────────────────────────────────────────────────────────────────────

/// A single scope node in the [`ScopeTree`].
#[derive(Debug, Clone)]
pub struct Scope {
    /// Unique identifier within the [`ScopeTree`].
    pub id: ScopeId,
    /// The kind of this scope.
    pub kind: ScopeKind,
    /// The parent scope, or `None` for the root.
    pub parent: Option<ScopeId>,
    /// Direct child scopes.
    pub children: Vec<ScopeId>,
    /// All names declared directly in this scope.
    pub bindings: HashMap<String, Binding>,
    /// `true` if code in this scope (or a non-function-boundary descendant)
    /// references `arguments`.
    pub uses_arguments: bool,
    /// `true` if code in this scope contains a direct call to `eval(…)`.
    pub uses_eval: bool,
    /// `true` if code in this scope references `this`.
    pub uses_this: bool,
    /// `true` if code in this scope references `super`.
    pub uses_super: bool,
    /// Names declared in this scope that are captured by inner function scopes.
    pub captures: HashSet<String>,
}

impl Scope {
    fn new(id: ScopeId, kind: ScopeKind, parent: Option<ScopeId>) -> Self {
        Self {
            id,
            kind,
            parent,
            children: Vec::new(),
            bindings: HashMap::new(),
            uses_arguments: false,
            uses_eval: false,
            uses_this: false,
            uses_super: false,
            captures: HashSet::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of a scope analysis error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeErrorKind {
    /// A name was declared more than once in the same scope in a way that is
    /// not allowed (e.g. two `let`/`const`/`class` with the same name).
    DuplicateBinding {
        /// Source location of the previous declaration.
        previous: SourceLocation,
    },
    /// A `let`, `const`, or `class` binding was read or written before its
    /// declaration in the source text (Temporal Dead Zone access).
    TdzViolation,
}

/// A single diagnostic produced during scope analysis.
#[derive(Debug, Clone)]
pub struct ScopeError {
    /// What went wrong.
    pub kind: ScopeErrorKind,
    /// The offending identifier.
    pub name: String,
    /// Source location of the offending reference or re-declaration.
    pub loc: SourceLocation,
}

// ─────────────────────────────────────────────────────────────────────────────
// ScopeTree
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a complete scope analysis pass over a [`Program`].
#[derive(Debug)]
pub struct ScopeTree {
    /// All scopes, indexed by [`ScopeId`].
    pub scopes: Vec<Scope>,
    /// The root scope (Global or Module).
    pub root: ScopeId,
    /// Diagnostics collected during the analysis.
    pub errors: Vec<ScopeError>,
}

impl ScopeTree {
    /// Returns a reference to the scope with the given `id`.
    pub fn scope(&self, id: ScopeId) -> &Scope {
        &self.scopes[id]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse `program` and return the fully-populated [`ScopeTree`].
///
/// The tree is always returned even when errors are present; callers should
/// inspect [`ScopeTree::errors`] after the call.
pub fn analyze(program: &Program) -> ScopeTree {
    let mut analyzer = Analyzer::new();
    analyzer.analyze_program(program);
    ScopeTree {
        scopes: analyzer.scopes,
        root: 0,
        errors: analyzer.errors,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal analyser
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks which names in a scope are still in the Temporal Dead Zone.
type TdzSet = HashSet<String>;

struct Analyzer {
    scopes: Vec<Scope>,
    /// Stack of active scope IDs (back = innermost).
    scope_stack: Vec<ScopeId>,
    errors: Vec<ScopeError>,
    /// Per-scope TDZ sets (only populated for scopes that contain
    /// `let`/`const`/`class` declarations).
    tdz: HashMap<ScopeId, TdzSet>,
}

impl Analyzer {
    fn new() -> Self {
        Self {
            scopes: Vec::new(),
            scope_stack: Vec::new(),
            errors: Vec::new(),
            tdz: HashMap::new(),
        }
    }

    // ── Scope management ─────────────────────────────────────────────────────

    fn push_scope(&mut self, kind: ScopeKind) -> ScopeId {
        let parent = self.scope_stack.last().copied();
        let id = self.scopes.len();
        self.scopes.push(Scope::new(id, kind, parent));
        if let Some(p) = parent {
            self.scopes[p].children.push(id);
        }
        self.scope_stack.push(id);
        id
    }

    fn pop_scope(&mut self) {
        self.scope_stack.pop();
    }

    fn current_scope(&self) -> ScopeId {
        *self
            .scope_stack
            .last()
            .expect("scope stack is never empty during analysis")
    }

    /// Returns the nearest enclosing scope whose kind is a function boundary
    /// (`Global`, `Function`, `Module`, `Eval`).
    fn nearest_fn_scope(&self) -> ScopeId {
        for &id in self.scope_stack.iter().rev() {
            if self.scopes[id].kind.is_function_boundary() {
                return id;
            }
        }
        // The root is always a function boundary, so this is unreachable when
        // the stack is non-empty.
        self.scope_stack[0]
    }

    // ── Declaration helpers ───────────────────────────────────────────────────

    /// Declare `name` with `kind` at `loc` in scope `scope_id`.
    ///
    /// Duplicate-declaration checks:
    /// - A `var` redeclaring another `var` (or a function) is silently OK.
    /// - A `let`/`const`/`class` that conflicts with *any* existing binding in
    ///   the same scope is an error.
    /// - A `var` that conflicts with a `let`/`const`/`class` in the same
    ///   immediately-enclosing function scope is an error.
    fn declare(&mut self, name: &str, kind: BindingKind, loc: SourceLocation, scope_id: ScopeId) {
        if let Some(existing) = self.scopes[scope_id].bindings.get(name) {
            let prev_loc = existing.loc;
            let prev_kind = existing.kind;
            // var-on-var (and var-on-function) redeclaration is always OK.
            if kind == BindingKind::Var
                && matches!(prev_kind, BindingKind::Var | BindingKind::Function)
            {
                return;
            }
            // function-on-function in the same function scope: OK (last wins).
            if kind == BindingKind::Function && prev_kind == BindingKind::Function {
                // Update the binding to point to the latest declaration.
                let b = self.scopes[scope_id].bindings.get_mut(name).unwrap();
                b.loc = loc;
                return;
            }
            // Everything else is a duplicate.
            self.errors.push(ScopeError {
                kind: ScopeErrorKind::DuplicateBinding { previous: prev_loc },
                name: name.to_owned(),
                loc,
            });
            return;
        }

        let binding = Binding {
            name: name.to_owned(),
            kind,
            loc,
            scope_id,
        };
        self.scopes[scope_id]
            .bindings
            .insert(name.to_owned(), binding);

        // Bindings subject to TDZ start out in the dead zone.
        if kind.has_tdz() {
            self.tdz
                .entry(scope_id)
                .or_default()
                .insert(name.to_owned());
        }
    }

    /// Declare `name` in the correct scope:
    /// - `var`/`function` declarations are placed in the nearest function-boundary
    ///   scope.
    /// - All other kinds go into the current (innermost) scope.
    fn declare_auto(&mut self, name: &str, kind: BindingKind, loc: SourceLocation) {
        let target = if kind.is_function_scoped() {
            self.nearest_fn_scope()
        } else {
            self.current_scope()
        };
        self.declare(name, kind, loc, target);
    }

    /// Mark a `let`/`const`/`class` binding as past its declaration point
    /// (i.e. remove it from the TDZ).
    fn exit_tdz(&mut self, name: &str, scope_id: ScopeId) {
        if let Some(set) = self.tdz.get_mut(&scope_id) {
            set.remove(name);
        }
    }

    // ── Identifier resolution ─────────────────────────────────────────────────

    /// Resolve an identifier reference at the current position in the source.
    ///
    /// This:
    /// 1. Walks up the scope chain to find a binding.
    /// 2. Reports a TDZ violation if the binding exists but is still in the
    ///    dead zone.
    /// 3. Records a closure capture if the binding is in an outer function
    ///    scope.
    fn resolve_ref(&mut self, name: &str, loc: SourceLocation) {
        // Special built-in identifiers need no resolution.
        if name == "undefined" || name == "Infinity" || name == "NaN" {
            return;
        }

        // Walk the scope stack from innermost to outermost.
        let stack_len = self.scope_stack.len();
        for i in (0..stack_len).rev() {
            let sid = self.scope_stack[i];
            // A `With` scope makes all inner name resolution dynamic; skip
            // without reporting an error.
            if self.scopes[sid].kind == ScopeKind::With {
                return;
            }
            if self.scopes[sid].bindings.contains_key(name) {
                // Found a binding — check for TDZ.
                let in_tdz = self
                    .tdz
                    .get(&sid)
                    .map(|s| s.contains(name))
                    .unwrap_or(false);
                if in_tdz {
                    self.errors.push(ScopeError {
                        kind: ScopeErrorKind::TdzViolation,
                        name: name.to_owned(),
                        loc,
                    });
                    return;
                }

                // Check for closure capture.
                let decl_fn = self.nearest_fn_scope_from(i);
                let ref_fn = self.nearest_fn_scope();
                if decl_fn != ref_fn {
                    self.scopes[decl_fn].captures.insert(name.to_owned());
                }
                return;
            }
        }
        // Unresolved: could be a global or a forward reference — not an error
        // at the scope-analysis stage.
    }

    /// Like [`nearest_fn_scope`] but starts the search from index `from` (not
    /// from the top of the stack).
    fn nearest_fn_scope_from(&self, from: usize) -> ScopeId {
        for i in (0..=from).rev() {
            let id = self.scope_stack[i];
            if self.scopes[id].kind.is_function_boundary() {
                return id;
            }
        }
        self.scope_stack[0]
    }

    // ── Usage flag propagation ────────────────────────────────────────────────

    fn mark_uses_arguments(&mut self) {
        // `arguments` belongs to the nearest function scope (not arrow).
        let fn_scope = self.nearest_fn_scope();
        self.scopes[fn_scope].uses_arguments = true;
    }

    fn mark_uses_eval(&mut self) {
        let sid = self.current_scope();
        self.scopes[sid].uses_eval = true;
    }

    fn mark_uses_this(&mut self) {
        let sid = self.current_scope();
        self.scopes[sid].uses_this = true;
    }

    fn mark_uses_super(&mut self) {
        let sid = self.current_scope();
        self.scopes[sid].uses_super = true;
    }

    // ── Hoisting helpers ──────────────────────────────────────────────────────

    /// Pre-scan `stmts` for `var` and `function` declarations and register
    /// them in the current scope (which must be a function boundary).
    fn hoist_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.hoist_stmt(stmt);
        }
    }

    fn hoist_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl(v) if v.kind == VarKind::Var => {
                for decl in &v.declarators {
                    self.hoist_pat_bindings(&decl.id, BindingKind::Var);
                }
            }
            Stmt::FnDecl(f) => {
                if let Some(id) = &f.id {
                    self.declare_auto(&id.name, BindingKind::Function, id.loc);
                }
                // Do NOT recurse into the function body — it creates its own scope.
            }
            Stmt::Block(b) => self.hoist_stmts(&b.body),
            Stmt::If(s) => {
                self.hoist_stmt(&s.consequent);
                if let Some(alt) = &s.alternate {
                    self.hoist_stmt(alt);
                }
            }
            Stmt::For(s) => {
                if let Some(ForInit::VarDecl(v)) = &s.init
                    && v.kind == VarKind::Var
                {
                    for decl in &v.declarators {
                        self.hoist_pat_bindings(&decl.id, BindingKind::Var);
                    }
                }
                self.hoist_stmt(&s.body);
            }
            Stmt::ForIn(s) => {
                if let ForInOfLeft::VarDecl(v) = &s.left
                    && v.kind == VarKind::Var
                {
                    for decl in &v.declarators {
                        self.hoist_pat_bindings(&decl.id, BindingKind::Var);
                    }
                }
                self.hoist_stmt(&s.body);
            }
            Stmt::ForOf(s) => {
                if let ForInOfLeft::VarDecl(v) = &s.left
                    && v.kind == VarKind::Var
                {
                    for decl in &v.declarators {
                        self.hoist_pat_bindings(&decl.id, BindingKind::Var);
                    }
                }
                self.hoist_stmt(&s.body);
            }
            Stmt::While(s) => self.hoist_stmt(&s.body),
            Stmt::DoWhile(s) => self.hoist_stmt(&s.body),
            Stmt::Switch(s) => {
                for case in &s.cases {
                    self.hoist_stmts(&case.consequent);
                }
            }
            Stmt::Try(s) => {
                self.hoist_stmts(&s.block.body);
                if let Some(handler) = &s.handler {
                    self.hoist_stmts(&handler.body.body);
                }
                if let Some(fin) = &s.finalizer {
                    self.hoist_stmts(&fin.body);
                }
            }
            Stmt::Labeled(s) => self.hoist_stmt(&s.body),
            Stmt::With(s) => self.hoist_stmt(&s.body),
            _ => {}
        }
    }

    /// Walk a binding pattern and hoist all the identifiers it introduces.
    fn hoist_pat_bindings(&mut self, pat: &Pat, kind: BindingKind) {
        match pat {
            Pat::Ident(id) => self.declare_auto(&id.name, kind, id.loc),
            Pat::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.hoist_pat_bindings(el, kind);
                }
            }
            Pat::Object(o) => {
                for prop in &o.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => self.hoist_pat_bindings(&kv.value, kind),
                        ObjectPatProp::Assign(ap) => {
                            self.declare_auto(&ap.key.name, kind, ap.key.loc)
                        }
                        ObjectPatProp::Rest(r) => self.hoist_pat_bindings(&r.argument, kind),
                    }
                }
            }
            Pat::Rest(r) => self.hoist_pat_bindings(&r.argument, kind),
            Pat::Assign(a) => self.hoist_pat_bindings(&a.left, kind),
        }
    }

    // ── Program entry point ───────────────────────────────────────────────────

    fn analyze_program(&mut self, program: &crate::parser::ast::Program) {
        use crate::parser::ast::SourceType;
        let kind = if program.source_type == SourceType::Module {
            ScopeKind::Module
        } else {
            ScopeKind::Global
        };
        self.push_scope(kind);

        // Collect top-level var / function hoisted declarations first.
        let stmts: Vec<Stmt> = program
            .body
            .iter()
            .filter_map(|item| {
                if let ProgramItem::Stmt(s) = item {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
        self.hoist_stmts(&stmts);

        // Then visit each item.
        for item in &program.body {
            match item {
                ProgramItem::Stmt(s) => self.visit_stmt(s),
                ProgramItem::ModuleDecl(m) => self.visit_module_decl(m),
            }
        }

        self.pop_scope();
    }

    // ── Module declarations ───────────────────────────────────────────────────

    fn visit_module_decl(&mut self, decl: &ModuleDecl) {
        match decl {
            ModuleDecl::Import(imp) => {
                use crate::parser::ast::ImportSpecifier;
                for spec in &imp.specifiers {
                    let local: &Ident = match spec {
                        ImportSpecifier::Named(s) => &s.local,
                        ImportSpecifier::Default(s) => &s.local,
                        ImportSpecifier::Namespace(s) => &s.local,
                    };
                    self.declare_auto(&local.name, BindingKind::Import, local.loc);
                }
            }
            ModuleDecl::ExportNamed(e) => {
                if let Some(decl) = &e.declaration {
                    self.visit_stmt(decl);
                }
                // specifiers referencing already-declared names — resolve them.
                for spec in &e.specifiers {
                    use crate::parser::ast::ModuleExportName;
                    if let ModuleExportName::Ident(id) = &spec.local {
                        self.resolve_ref(&id.name, id.loc);
                    }
                }
            }
            ModuleDecl::ExportDefault(e) => {
                use crate::parser::ast::ExportDefaultExpr;
                match &e.declaration {
                    ExportDefaultExpr::Fn(f) => self.visit_fn_decl(f),
                    ExportDefaultExpr::Class(c) => self.visit_class_decl(c),
                    ExportDefaultExpr::Expr(ex) => self.visit_expr(ex),
                }
            }
            ModuleDecl::ExportAll(_) => {
                // `export * from "…"` — no bindings to declare.
            }
        }
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Block(b) => self.visit_block_stmt(b, false),
            Stmt::VarDecl(v) => self.visit_var_decl(v),
            Stmt::FnDecl(f) => self.visit_fn_decl(f),
            Stmt::ClassDecl(c) => self.visit_class_decl(c),
            Stmt::Expr(e) => self.visit_expr(&e.expr),
            Stmt::If(s) => {
                self.visit_expr(&s.test);
                self.visit_stmt(&s.consequent);
                if let Some(alt) = &s.alternate {
                    self.visit_stmt(alt);
                }
            }
            Stmt::For(s) => {
                // A for-statement with `let`/`const` init creates its own scope.
                let needs_scope = matches!(
                    &s.init,
                    Some(ForInit::VarDecl(v)) if v.kind != VarKind::Var
                );
                if needs_scope {
                    self.push_scope(ScopeKind::Block);
                }
                if let Some(init) = &s.init {
                    match init {
                        ForInit::VarDecl(v) => self.visit_var_decl(v),
                        ForInit::Expr(e) => self.visit_expr(e),
                    }
                }
                if let Some(test) = &s.test {
                    self.visit_expr(test);
                }
                if let Some(update) = &s.update {
                    self.visit_expr(update);
                }
                self.visit_stmt(&s.body);
                if needs_scope {
                    self.pop_scope();
                }
            }
            Stmt::ForIn(s) => {
                let needs_scope =
                    matches!(&s.left, ForInOfLeft::VarDecl(v) if v.kind != VarKind::Var);
                if needs_scope {
                    self.push_scope(ScopeKind::Block);
                }
                match &s.left {
                    ForInOfLeft::VarDecl(v) => self.visit_var_decl(v),
                    ForInOfLeft::Pat(p) => self.visit_pat_ref(p),
                }
                self.visit_expr(&s.right);
                self.visit_stmt(&s.body);
                if needs_scope {
                    self.pop_scope();
                }
            }
            Stmt::ForOf(s) => {
                let needs_scope =
                    matches!(&s.left, ForInOfLeft::VarDecl(v) if v.kind != VarKind::Var);
                if needs_scope {
                    self.push_scope(ScopeKind::Block);
                }
                match &s.left {
                    ForInOfLeft::VarDecl(v) => self.visit_var_decl(v),
                    ForInOfLeft::Pat(p) => self.visit_pat_ref(p),
                }
                self.visit_expr(&s.right);
                self.visit_stmt(&s.body);
                if needs_scope {
                    self.pop_scope();
                }
            }
            Stmt::While(s) => {
                self.visit_expr(&s.test);
                self.visit_stmt(&s.body);
            }
            Stmt::DoWhile(s) => {
                self.visit_stmt(&s.body);
                self.visit_expr(&s.test);
            }
            Stmt::Switch(s) => {
                self.visit_expr(&s.discriminant);
                // The switch body is a single block scope containing all cases.
                self.push_scope(ScopeKind::Block);
                for case in &s.cases {
                    if let Some(test) = &case.test {
                        self.visit_expr(test);
                    }
                    for stmt in &case.consequent {
                        self.visit_stmt(stmt);
                    }
                }
                self.pop_scope();
            }
            Stmt::Try(s) => self.visit_try_stmt(s),
            Stmt::Return(s) => {
                if let Some(arg) = &s.argument {
                    self.visit_expr(arg);
                }
            }
            Stmt::Throw(s) => self.visit_expr(&s.argument),
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Debugger(_) | Stmt::Empty(_) => {}
            Stmt::Labeled(s) => self.visit_stmt(&s.body),
            Stmt::With(s) => self.visit_with_stmt(s),
        }
    }

    fn visit_block_stmt(&mut self, block: &BlockStmt, is_fn_body: bool) {
        if !is_fn_body {
            self.push_scope(ScopeKind::Block);
            // Pre-scan for function declarations inside this block.
            self.hoist_stmts(&block.body);
        }
        for stmt in &block.body {
            self.visit_stmt(stmt);
        }
        if !is_fn_body {
            self.pop_scope();
        }
    }

    fn visit_var_decl(&mut self, v: &VarDecl) {
        for decl in &v.declarators {
            // Register the binding (var was already hoisted; let/const/class
            // are registered here for the first time).
            match v.kind {
                VarKind::Var => {
                    // Already hoisted — just visit the initialiser.
                }
                VarKind::Let => {
                    self.declare_pat_bindings(&decl.id, BindingKind::Let);
                }
                VarKind::Const => {
                    self.declare_pat_bindings(&decl.id, BindingKind::Const);
                }
            }
            if let Some(init) = &decl.init {
                self.visit_expr(init);
            }
            // After visiting the initialiser, the binding is past its
            // declaration point — lift it out of the TDZ.
            if matches!(v.kind, VarKind::Let | VarKind::Const) {
                self.exit_tdz_pat(&decl.id);
            }
        }
    }

    /// Register bindings for a pattern with the given `kind` in the *current*
    /// scope (non-hoisting).
    fn declare_pat_bindings(&mut self, pat: &Pat, kind: BindingKind) {
        match pat {
            Pat::Ident(id) => self.declare_auto(&id.name, kind, id.loc),
            Pat::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.declare_pat_bindings(el, kind);
                }
            }
            Pat::Object(o) => {
                for prop in &o.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => self.declare_pat_bindings(&kv.value, kind),
                        ObjectPatProp::Assign(ap) => {
                            self.declare_auto(&ap.key.name, kind, ap.key.loc)
                        }
                        ObjectPatProp::Rest(r) => self.declare_pat_bindings(&r.argument, kind),
                    }
                }
            }
            Pat::Rest(r) => self.declare_pat_bindings(&r.argument, kind),
            Pat::Assign(a) => self.declare_pat_bindings(&a.left, kind),
        }
    }

    /// Remove all identifiers in `pat` from the TDZ.
    fn exit_tdz_pat(&mut self, pat: &Pat) {
        let sid = self.current_scope();
        match pat {
            Pat::Ident(id) => self.exit_tdz(&id.name, sid),
            Pat::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.exit_tdz_pat(el);
                }
            }
            Pat::Object(o) => {
                for prop in &o.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => self.exit_tdz_pat(&kv.value),
                        ObjectPatProp::Assign(ap) => self.exit_tdz(&ap.key.name, sid),
                        ObjectPatProp::Rest(r) => self.exit_tdz_pat(&r.argument),
                    }
                }
            }
            Pat::Rest(r) => self.exit_tdz_pat(&r.argument),
            Pat::Assign(a) => self.exit_tdz_pat(&a.left),
        }
    }

    fn visit_fn_decl(&mut self, f: &FnDecl) {
        // The function name is already hoisted in the outer scope; here we just
        // analyse the function body.
        self.push_scope(ScopeKind::Function);
        // Declare parameters in the new function scope.
        for param in &f.params {
            self.declare_param(param);
        }
        // Hoist var/function declarations inside the body.
        self.hoist_stmts(&f.body.body);
        // Visit the body without creating an extra block scope.
        for stmt in &f.body.body {
            self.visit_stmt(stmt);
        }
        self.pop_scope();
    }

    fn visit_fn_expr(&mut self, f: &FnExpr) {
        self.push_scope(ScopeKind::Function);
        // A named function expression can self-reference by its name.
        if let Some(id) = &f.id {
            self.declare(
                &id.name,
                BindingKind::Function,
                id.loc,
                self.current_scope(),
            );
        }
        for param in &f.params {
            self.declare_param(param);
        }
        self.hoist_stmts(&f.body.body);
        for stmt in &f.body.body {
            self.visit_stmt(stmt);
        }
        self.pop_scope();
    }

    fn visit_arrow_expr(&mut self, a: &crate::parser::ast::ArrowExpr) {
        // Arrow functions share `this`/`arguments`/`super` with the enclosing
        // scope, so we still create a new scope but with `Function` kind (the
        // `is_function_boundary` predicate applies).  For simplicity we use a
        // dedicated variant — but for this implementation we re-use Function.
        self.push_scope(ScopeKind::Function);
        for param in &a.params {
            self.declare_param(param);
        }
        match &a.body {
            ArrowBody::Block(b) => {
                self.hoist_stmts(&b.body);
                for stmt in &b.body {
                    self.visit_stmt(stmt);
                }
            }
            ArrowBody::Expr(e) => self.visit_expr(e),
        }
        self.pop_scope();
    }

    fn visit_class_decl(&mut self, c: &ClassDecl) {
        // The class name binding is declared in the outer scope (block-scoped,
        // TDZ-guarded).
        if let Some(id) = &c.id {
            self.declare_auto(&id.name, BindingKind::Class, id.loc);
            self.exit_tdz(&id.name, self.current_scope());
        }
        if let Some(super_class) = &c.super_class {
            self.visit_expr(super_class);
        }
        self.visit_class_body(&c.body);
    }

    fn visit_class_expr(&mut self, c: &ClassExpr) {
        // Push a scope so the class name is only visible inside the class body.
        self.push_scope(ScopeKind::Block);
        if let Some(id) = &c.id {
            let sid = self.current_scope();
            self.declare(&id.name, BindingKind::Class, id.loc, sid);
            self.exit_tdz(&id.name, sid);
        }
        if let Some(super_class) = &c.super_class {
            self.visit_expr(super_class);
        }
        self.visit_class_body(&c.body);
        self.pop_scope();
    }

    fn visit_class_body(&mut self, body: &ClassBody) {
        for member in &body.body {
            match member {
                ClassMember::Method(m) => self.visit_fn_expr(&m.value),
                ClassMember::Property(p) => {
                    if let Some(val) = &p.value {
                        self.visit_expr(val);
                    }
                }
                ClassMember::StaticBlock(s) => {
                    self.push_scope(ScopeKind::Block);
                    self.hoist_stmts(&s.body);
                    for stmt in &s.body {
                        self.visit_stmt(stmt);
                    }
                    self.pop_scope();
                }
            }
        }
    }

    fn visit_try_stmt(&mut self, s: &crate::parser::ast::TryStmt) {
        self.visit_block_stmt(&s.block, false);
        if let Some(handler) = &s.handler {
            self.visit_catch_clause(handler);
        }
        if let Some(fin) = &s.finalizer {
            self.visit_block_stmt(fin, false);
        }
    }

    fn visit_catch_clause(&mut self, clause: &CatchClause) {
        self.push_scope(ScopeKind::Catch);
        if let Some(param) = &clause.param {
            self.declare_pat_bindings(param, BindingKind::Param);
            // Catch parameter is never in TDZ.
        }
        self.hoist_stmts(&clause.body.body);
        for stmt in &clause.body.body {
            self.visit_stmt(stmt);
        }
        self.pop_scope();
    }

    fn visit_with_stmt(&mut self, s: &crate::parser::ast::WithStmt) {
        self.visit_expr(&s.object);
        self.push_scope(ScopeKind::With);
        self.visit_stmt(&s.body);
        self.pop_scope();
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident(id) => {
                if id.name == "arguments" {
                    self.mark_uses_arguments();
                }
                self.resolve_ref(&id.name, id.loc);
            }
            Expr::This(_) => self.mark_uses_this(),
            Expr::MetaProp(m) => {
                if m.meta.name == "super" {
                    self.mark_uses_super();
                }
            }
            Expr::Null(_)
            | Expr::Bool(_)
            | Expr::Num(_)
            | Expr::Str(_)
            | Expr::BigInt(_)
            | Expr::Regexp(_) => {}
            Expr::Template(t) => {
                for e in &t.expressions {
                    self.visit_expr(e);
                }
            }
            Expr::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.visit_expr(el);
                }
            }
            Expr::Object(o) => {
                use crate::parser::ast::{ObjectProp, PropValue};
                for prop in &o.properties {
                    match prop {
                        ObjectProp::Prop(p) => match &p.value {
                            PropValue::Value(v) => self.visit_expr(v),
                            PropValue::Shorthand => {
                                // shorthand `{ x }` — resolve `x` as a reference.
                                if let crate::parser::ast::PropKey::Ident(id) = &p.key {
                                    self.resolve_ref(&id.name, id.loc);
                                }
                            }
                            PropValue::Get(f) | PropValue::Set(f) | PropValue::Method(f) => {
                                self.visit_fn_expr(f)
                            }
                        },
                        ObjectProp::Spread(s) => self.visit_expr(&s.argument),
                    }
                }
            }
            Expr::Fn(f) => self.visit_fn_expr(f),
            Expr::Arrow(a) => self.visit_arrow_expr(a),
            Expr::Class(c) => self.visit_class_expr(c),
            Expr::Unary(u) => self.visit_expr(&u.argument),
            Expr::Update(u) => self.visit_expr(&u.argument),
            Expr::Binary(b) => {
                self.visit_expr(&b.left);
                self.visit_expr(&b.right);
            }
            Expr::Logical(l) => {
                self.visit_expr(&l.left);
                self.visit_expr(&l.right);
            }
            Expr::Conditional(c) => {
                self.visit_expr(&c.test);
                self.visit_expr(&c.consequent);
                self.visit_expr(&c.alternate);
            }
            Expr::Assign(a) => {
                match &a.left {
                    AssignTarget::Pat(p) => self.visit_pat_ref(p),
                    AssignTarget::Expr(e) => self.visit_expr(e),
                }
                self.visit_expr(&a.right);
            }
            Expr::Sequence(s) => {
                for e in &s.expressions {
                    self.visit_expr(e);
                }
            }
            Expr::Member(m) => {
                self.visit_expr(&m.object);
                if m.is_computed
                    && let crate::parser::ast::MemberProp::Computed(e) = &m.property
                {
                    self.visit_expr(e);
                }
            }
            Expr::OptionalMember(m) => {
                self.visit_expr(&m.object);
                if m.is_computed
                    && let crate::parser::ast::MemberProp::Computed(e) = &m.property
                {
                    self.visit_expr(e);
                }
            }
            Expr::Call(c) => {
                // Detect direct `eval(…)` call.
                if let Expr::Ident(id) = c.callee.as_ref()
                    && id.name == "eval"
                {
                    self.mark_uses_eval();
                }
                self.visit_expr(&c.callee);
                for arg in &c.arguments {
                    self.visit_expr(arg);
                }
            }
            Expr::OptionalCall(c) => {
                self.visit_expr(&c.callee);
                for arg in &c.arguments {
                    self.visit_expr(arg);
                }
            }
            Expr::New(n) => {
                self.visit_expr(&n.callee);
                for arg in &n.arguments {
                    self.visit_expr(arg);
                }
            }
            Expr::TaggedTemplate(t) => {
                self.visit_expr(&t.tag);
                for e in &t.quasi.expressions {
                    self.visit_expr(e);
                }
            }
            Expr::Spread(s) => self.visit_expr(&s.argument),
            Expr::Yield(y) => {
                if let Some(arg) = &y.argument {
                    self.visit_expr(arg);
                }
            }
            Expr::Await(a) => self.visit_expr(&a.argument),
            Expr::Import(i) => {
                self.visit_expr(&i.source);
                if let Some(opts) = &i.options {
                    self.visit_expr(opts);
                }
            }
        }
    }

    /// Visit a pattern that appears on the *left-hand side* of an assignment
    /// (i.e. as an assignment target, not a declaration).  We resolve the
    /// identifiers as references rather than declaring new bindings.
    fn visit_pat_ref(&mut self, pat: &Pat) {
        match pat {
            Pat::Ident(id) => self.resolve_ref(&id.name, id.loc),
            Pat::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.visit_pat_ref(el);
                }
            }
            Pat::Object(o) => {
                for prop in &o.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => self.visit_pat_ref(&kv.value),
                        ObjectPatProp::Assign(ap) => self.resolve_ref(&ap.key.name, ap.key.loc),
                        ObjectPatProp::Rest(r) => self.visit_pat_ref(&r.argument),
                    }
                }
            }
            Pat::Rest(r) => self.visit_pat_ref(&r.argument),
            Pat::Assign(a) => {
                self.visit_pat_ref(&a.left);
                self.visit_expr(&a.right);
            }
        }
    }

    // ── Parameter helper ──────────────────────────────────────────────────────

    fn declare_param(&mut self, param: &Param) {
        let sid = self.current_scope();
        self.declare_param_pat(&param.pat, sid);
        if let Some(default) = &param.default {
            self.visit_expr(default);
        }
    }

    fn declare_param_pat(&mut self, pat: &Pat, scope_id: ScopeId) {
        match pat {
            Pat::Ident(id) => self.declare(&id.name, BindingKind::Param, id.loc, scope_id),
            Pat::Array(a) => {
                for el in a.elements.iter().flatten() {
                    self.declare_param_pat(el, scope_id);
                }
            }
            Pat::Object(o) => {
                for prop in &o.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => self.declare_param_pat(&kv.value, scope_id),
                        ObjectPatProp::Assign(ap) => {
                            self.declare(&ap.key.name, BindingKind::Param, ap.key.loc, scope_id)
                        }
                        ObjectPatProp::Rest(r) => self.declare_param_pat(&r.argument, scope_id),
                    }
                }
            }
            Pat::Rest(r) => self.declare_param_pat(&r.argument, scope_id),
            Pat::Assign(a) => self.declare_param_pat(&a.left, scope_id),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{
        BlockStmt, EmptyStmt, FnDecl, Ident, Param, Program, ProgramItem, SourceType, Stmt,
        VarDecl, VarDeclarator, VarKind,
    };
    use crate::parser::scanner::{Position, Span};

    fn loc() -> SourceLocation {
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

    fn ident(name: &str) -> Ident {
        Ident {
            loc: loc(),
            name: name.to_owned(),
        }
    }

    fn empty_program() -> Program {
        Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![],
        }
    }

    fn ident_expr(name: &str) -> Expr {
        Expr::Ident(ident(name))
    }

    fn var_decl(kind: VarKind, name: &str) -> VarDecl {
        VarDecl {
            loc: loc(),
            kind,
            declarators: vec![VarDeclarator {
                loc: loc(),
                id: Pat::Ident(ident(name)),
                init: None,
            }],
        }
    }

    fn var_decl_init(kind: VarKind, name: &str, init: Expr) -> VarDecl {
        VarDecl {
            loc: loc(),
            kind,
            declarators: vec![VarDeclarator {
                loc: loc(),
                id: Pat::Ident(ident(name)),
                init: Some(Box::new(init)),
            }],
        }
    }

    // ── Basic scope creation ──────────────────────────────────────────────────

    #[test]
    fn test_analyze_empty_script_creates_global_scope() {
        let prog = empty_program();
        let tree = analyze(&prog);
        assert_eq!(tree.scopes.len(), 1);
        assert_eq!(tree.scopes[0].kind, ScopeKind::Global);
        assert!(tree.errors.is_empty());
    }

    #[test]
    fn test_analyze_module_creates_module_scope() {
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Module,
            body: vec![],
        };
        let tree = analyze(&prog);
        assert_eq!(tree.scopes[0].kind, ScopeKind::Module);
    }

    // ── Var hoisting ─────────────────────────────────────────────────────────

    #[test]
    fn test_var_hoisted_to_global() {
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(Stmt::VarDecl(var_decl(
                VarKind::Var,
                "x",
            )))],
        };
        let tree = analyze(&prog);
        assert!(tree.scopes[0].bindings.contains_key("x"));
        assert_eq!(tree.scopes[0].bindings["x"].kind, BindingKind::Var);
        assert!(tree.errors.is_empty());
    }

    #[test]
    fn test_var_hoisted_out_of_block() {
        // { var x; } — x should be in the global scope, not the block scope.
        let block = Stmt::Block(BlockStmt {
            loc: loc(),
            body: vec![Stmt::VarDecl(var_decl(VarKind::Var, "x"))],
        });
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(block)],
        };
        let tree = analyze(&prog);
        // Global scope (id=0) must contain x.
        assert!(tree.scopes[0].bindings.contains_key("x"));
        // The block scope (id=1) must NOT have x.
        assert!(!tree.scopes[1].bindings.contains_key("x"));
        assert!(tree.errors.is_empty());
    }

    #[test]
    fn test_function_declaration_hoisted() {
        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: loc(),
            id: Some(ident("f")),
            is_async: false,
            is_generator: false,
            params: vec![],
            body: BlockStmt {
                loc: loc(),
                body: vec![],
            },
        }));
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(fn_decl)],
        };
        let tree = analyze(&prog);
        assert!(tree.scopes[0].bindings.contains_key("f"));
        assert_eq!(tree.scopes[0].bindings["f"].kind, BindingKind::Function);
        assert!(tree.errors.is_empty());
    }

    // ── Let / const block scoping ─────────────────────────────────────────────

    #[test]
    fn test_let_in_block_scope() {
        let block = Stmt::Block(BlockStmt {
            loc: loc(),
            body: vec![Stmt::VarDecl(var_decl(VarKind::Let, "y"))],
        });
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(block)],
        };
        let tree = analyze(&prog);
        // y must be in the inner block scope, not the global scope.
        assert!(!tree.scopes[0].bindings.contains_key("y"));
        let block_scope = tree
            .scopes
            .iter()
            .find(|s| s.kind == ScopeKind::Block)
            .unwrap();
        assert!(block_scope.bindings.contains_key("y"));
        assert!(tree.errors.is_empty());
    }

    // ── Duplicate detection ───────────────────────────────────────────────────

    #[test]
    fn test_duplicate_var_is_ok() {
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Var, "x"))),
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Var, "x"))),
            ],
        };
        let tree = analyze(&prog);
        assert!(
            tree.errors.is_empty(),
            "var redeclaration should not be an error"
        );
    }

    #[test]
    fn test_duplicate_let_is_error() {
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Let, "x"))),
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Let, "x"))),
            ],
        };
        let tree = analyze(&prog);
        assert_eq!(tree.errors.len(), 1);
        assert_eq!(tree.errors[0].name, "x");
        assert!(matches!(
            tree.errors[0].kind,
            ScopeErrorKind::DuplicateBinding { .. }
        ));
    }

    #[test]
    fn test_duplicate_const_is_error() {
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Const, "C"))),
                ProgramItem::Stmt(Stmt::VarDecl(var_decl(VarKind::Const, "C"))),
            ],
        };
        let tree = analyze(&prog);
        assert_eq!(tree.errors.len(), 1);
    }

    // ── TDZ detection ─────────────────────────────────────────────────────────

    #[test]
    fn test_tdz_let_use_before_decl() {
        // Use `x` before `let x` — TDZ violation.
        // We simulate: the init expression of `let x = x` reads x while it's in TDZ.
        let init_expr = ident_expr("x");
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(Stmt::VarDecl(var_decl_init(
                VarKind::Let,
                "x",
                init_expr,
            )))],
        };
        let tree = analyze(&prog);
        assert!(
            tree.errors
                .iter()
                .any(|e| e.name == "x" && matches!(e.kind, ScopeErrorKind::TdzViolation)),
            "expected a TDZ violation for 'x'"
        );
    }

    #[test]
    fn test_tdz_no_violation_after_decl() {
        // let x; let y = x; — no TDZ violation (x is declared before y's init).
        use crate::parser::ast::VarDeclarator;
        let x_decl = Stmt::VarDecl(VarDecl {
            loc: loc(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: loc(),
                id: Pat::Ident(ident("x")),
                init: None,
            }],
        });
        let y_decl = Stmt::VarDecl(VarDecl {
            loc: loc(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: loc(),
                id: Pat::Ident(ident("y")),
                init: Some(Box::new(ident_expr("x"))),
            }],
        });
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(x_decl), ProgramItem::Stmt(y_decl)],
        };
        let tree = analyze(&prog);
        assert!(tree.errors.is_empty(), "no TDZ error expected");
    }

    // ── Function scopes & parameters ─────────────────────────────────────────

    #[test]
    fn test_function_creates_scope_with_params() {
        let fn_stmt = Stmt::FnDecl(Box::new(FnDecl {
            loc: loc(),
            id: Some(ident("f")),
            is_async: false,
            is_generator: false,
            params: vec![Param {
                loc: loc(),
                pat: Pat::Ident(ident("a")),
                default: None,
            }],
            body: BlockStmt {
                loc: loc(),
                body: vec![],
            },
        }));
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(fn_stmt)],
        };
        let tree = analyze(&prog);
        // There should be a Function scope.
        let fn_scope = tree
            .scopes
            .iter()
            .find(|s| s.kind == ScopeKind::Function)
            .unwrap();
        assert!(fn_scope.bindings.contains_key("a"));
        assert_eq!(fn_scope.bindings["a"].kind, BindingKind::Param);
    }

    // ── Closure captures ──────────────────────────────────────────────────────

    #[test]
    fn test_closure_captures_outer_var() {
        // function outer() { var x; function inner() { return x; } }
        use crate::parser::ast::ReturnStmt;

        let inner_fn = Stmt::FnDecl(Box::new(FnDecl {
            loc: loc(),
            id: Some(ident("inner")),
            is_async: false,
            is_generator: false,
            params: vec![],
            body: BlockStmt {
                loc: loc(),
                body: vec![Stmt::Return(ReturnStmt {
                    loc: loc(),
                    argument: Some(Box::new(ident_expr("x"))),
                })],
            },
        }));
        let outer_fn = Stmt::FnDecl(Box::new(FnDecl {
            loc: loc(),
            id: Some(ident("outer")),
            is_async: false,
            is_generator: false,
            params: vec![],
            body: BlockStmt {
                loc: loc(),
                body: vec![Stmt::VarDecl(var_decl(VarKind::Var, "x")), inner_fn],
            },
        }));
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(outer_fn)],
        };
        let tree = analyze(&prog);
        // Find the outer function scope (the one that declares x).
        let outer_scope = tree
            .scopes
            .iter()
            .find(|s| s.kind == ScopeKind::Function && s.bindings.contains_key("x"))
            .expect("outer function scope not found");
        assert!(
            outer_scope.captures.contains("x"),
            "x should be captured; captures = {:?}",
            outer_scope.captures
        );
        assert!(tree.errors.is_empty());
    }

    // ── Catch scope ───────────────────────────────────────────────────────────

    #[test]
    fn test_catch_creates_catch_scope() {
        use crate::parser::ast::{CatchClause, TryStmt};

        let try_stmt = Stmt::Try(TryStmt {
            loc: loc(),
            block: BlockStmt {
                loc: loc(),
                body: vec![],
            },
            handler: Some(CatchClause {
                loc: loc(),
                param: Some(Pat::Ident(ident("e"))),
                body: BlockStmt {
                    loc: loc(),
                    body: vec![],
                },
            }),
            finalizer: None,
        });
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(try_stmt)],
        };
        let tree = analyze(&prog);
        let catch_scope = tree
            .scopes
            .iter()
            .find(|s| s.kind == ScopeKind::Catch)
            .unwrap();
        assert!(catch_scope.bindings.contains_key("e"));
        assert!(tree.errors.is_empty());
    }

    // ── With scope ───────────────────────────────────────────────────────────

    #[test]
    fn test_with_creates_with_scope() {
        use crate::parser::ast::WithStmt;

        let with_stmt = Stmt::With(WithStmt {
            loc: loc(),
            object: Box::new(ident_expr("obj")),
            body: Box::new(Stmt::Empty(EmptyStmt { loc: loc() })),
        });
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(with_stmt)],
        };
        let tree = analyze(&prog);
        assert!(tree.scopes.iter().any(|s| s.kind == ScopeKind::With));
        assert!(tree.errors.is_empty());
    }

    // ── Usage flags ───────────────────────────────────────────────────────────

    #[test]
    fn test_uses_eval_flag() {
        use crate::parser::ast::{CallExpr, ExprStmt};

        let call = Expr::Call(Box::new(CallExpr {
            loc: loc(),
            callee: Box::new(ident_expr("eval")),
            arguments: vec![Expr::Str(crate::parser::ast::StringLit {
                loc: loc(),
                value: "1+1".to_owned(),
            })],
        }));
        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(Stmt::Expr(ExprStmt {
                loc: loc(),
                expr: Box::new(call),
            }))],
        };
        let tree = analyze(&prog);
        assert!(tree.scopes[0].uses_eval);
    }

    #[test]
    fn test_uses_this_flag() {
        use crate::parser::ast::{ExprStmt, ThisExpr};

        let prog = Program {
            loc: loc(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::Stmt(Stmt::Expr(ExprStmt {
                loc: loc(),
                expr: Box::new(Expr::This(ThisExpr { loc: loc() })),
            }))],
        };
        let tree = analyze(&prog);
        assert!(tree.scopes[0].uses_this);
    }

    #[test]
    fn test_scope_kind_is_function_boundary() {
        assert!(ScopeKind::Global.is_function_boundary());
        assert!(ScopeKind::Function.is_function_boundary());
        assert!(ScopeKind::Module.is_function_boundary());
        assert!(ScopeKind::Eval.is_function_boundary());
        assert!(!ScopeKind::Block.is_function_boundary());
        assert!(!ScopeKind::Catch.is_function_boundary());
        assert!(!ScopeKind::With.is_function_boundary());
    }

    #[test]
    fn test_binding_kind_has_tdz() {
        assert!(BindingKind::Let.has_tdz());
        assert!(BindingKind::Const.has_tdz());
        assert!(BindingKind::Class.has_tdz());
        assert!(!BindingKind::Var.has_tdz());
        assert!(!BindingKind::Function.has_tdz());
        assert!(!BindingKind::Param.has_tdz());
        assert!(!BindingKind::Import.has_tdz());
    }
}

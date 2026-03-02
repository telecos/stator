//! Bytecode generator: walks an AST and emits a [`BytecodeArray`].
//!
//! # Overview
//!
//! [`BytecodeGenerator`] compiles a JavaScript [`crate::parser::ast::Program`]
//! into a top-level [`BytecodeArray`].  Nested functions are compiled
//! recursively and stored as [`ConstantPoolEntry::Function`] entries in the
//! enclosing function's constant pool, where they are referenced by the
//! [`crate::bytecode::bytecodes::Opcode::CreateClosure`] instruction.
//!
//! # Incremental phases
//!
//! 1. **Literals + arithmetic + return** — `null`, `true`/`false`, numbers,
//!    strings, binary arithmetic and comparison operators, `return`.
//! 2. **Variables + control flow** — `var`/`let`/`const` declarations,
//!    identifier references, simple assignment, `if`, `while`, `do…while`,
//!    `for`, `break`, `continue`, sequence and update expressions.
//! 3. **Functions + closures** — function declarations and expressions,
//!    arrow functions, function calls, property access, `new`, `throw`,
//!    `try`/`catch`/`finally`.
//! 4. Classes and generators are not yet implemented; they return an error.
//!
//! # Jump patching
//!
//! The generator builds a `Vec<Instruction>` with placeholder `JumpOffset(0)`
//! operands for forward jumps.  After the function body is compiled, a
//! fixed-point resolution pass computes byte-level offsets and patches every
//! jump instruction, iterating until no sizes change.

use std::collections::HashMap;

use crate::bytecode::bytecode_array::{
    BytecodeArray, ConstantPoolEntry, HandlerTableEntry, SourcePosition,
};
use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
use crate::bytecode::feedback::{FeedbackMetadata, FeedbackSlotKind};
use crate::bytecode::register::{Register, RegisterAllocator};
use crate::error::{StatorError, StatorResult};
use crate::parser::ast::{
    ArrowBody, ArrowExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, Expr, FnDecl, FnExpr,
    ForInit, ForStmt, LogicalOp, Pat, Program, ProgramItem, Stmt, UnaryOp, UpdateOp, VarDecl,
    VarDeclarator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a [`Register`] to an [`Operand::Register`].
///
/// Parameter registers (negative `i32` indices) are preserved via bit-cast to
/// `u32`; the runtime sign-extends them back when indexing the frame.
fn to_reg_op(reg: Register) -> Operand {
    Operand::Register(reg.0 as u32)
}

/// Minimum number of bytes required to encode `op` as a single operand field.
fn operand_bytes_needed(op: Operand) -> usize {
    match op {
        Operand::Register(v)
        | Operand::RegisterCount(v)
        | Operand::ConstantPoolIdx(v)
        | Operand::FeedbackSlot(v)
        | Operand::RuntimeId(v) => {
            if v <= u8::MAX as u32 {
                1
            } else if v <= u16::MAX as u32 {
                2
            } else {
                4
            }
        }
        Operand::Immediate(v) | Operand::JumpOffset(v) => {
            if (i8::MIN as i32..=i8::MAX as i32).contains(&v) {
                1
            } else if (i16::MIN as i32..=i16::MAX as i32).contains(&v) {
                2
            } else {
                4
            }
        }
        Operand::Flag(_) => 1,
    }
}

/// Byte size of one encoded instruction (prefix + opcode + operands).
fn instr_byte_size(instr: &Instruction) -> usize {
    let w = instr
        .operands
        .iter()
        .map(|&op| operand_bytes_needed(op))
        .max()
        .unwrap_or(1);
    let prefix = usize::from(w > 1);
    prefix + 1 + instr.operands.len() * w
}

/// Compute the byte offset of every instruction in `instructions`.
///
/// Returns a `Vec` of length `instructions.len() + 1`; the last entry is the
/// total byte size of the encoded stream.
fn compute_byte_offsets(instructions: &[Instruction]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(instructions.len() + 1);
    let mut pos = 0usize;
    for instr in instructions {
        offsets.push(pos);
        pos += instr_byte_size(instr);
    }
    offsets.push(pos);
    offsets
}

// ─────────────────────────────────────────────────────────────────────────────
// Label
// ─────────────────────────────────────────────────────────────────────────────

/// A bytecode label: a logical position in the instruction stream.
///
/// Labels start unbound.  When the target instruction is emitted, the label is
/// *bound* by recording its instruction index.  All jump instructions that
/// referenced the label before it was bound have their offsets resolved during
/// [`FunctionCompiler::finalize`].
struct Label {
    /// Instruction index where this label is bound, or `None` if not yet bound.
    bound_at: Option<usize>,
    /// Indices of jump instructions that jump to this label.
    refs: Vec<usize>,
}

impl Label {
    fn new() -> Self {
        Self {
            bound_at: None,
            refs: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FunctionCompiler
// ─────────────────────────────────────────────────────────────────────────────

/// Compiles a single JavaScript function body into a [`BytecodeArray`].
///
/// Create one instance per function; nested functions are compiled
/// recursively.
struct FunctionCompiler {
    /// Instructions being accumulated.
    instructions: Vec<Instruction>,
    /// Constant-pool entries (literals, nested functions, …).
    constant_pool: Vec<ConstantPoolEntry>,
    /// Virtual-register allocator.
    allocator: RegisterAllocator,
    /// Variable scope stack.  Each entry is a map from variable name to
    /// its assigned register.  The outermost scope is `scopes[0]`.
    scopes: Vec<HashMap<String, Register>>,
    /// Source-position table (populated at expression boundaries).
    source_positions: Vec<SourcePosition>,
    /// Label table.
    labels: Vec<Label>,
    /// Stack of `(continue_label, break_label)` pairs for loop statements.
    loop_stack: Vec<(usize, usize)>,
    /// Number of formal parameters.
    param_count: u32,
    /// Ordered list of feedback slot kinds allocated during compilation.
    slot_kinds: Vec<FeedbackSlotKind>,
    /// Per-function exception handler table entries.
    handler_table: Vec<HandlerTableEntry>,
}

impl FunctionCompiler {
    /// Create a new compiler for a function with the given parameter list.
    ///
    /// Each parameter that is a simple identifier binding is added to the
    /// outermost scope immediately; complex patterns are not yet supported.
    fn new(params: &[crate::parser::ast::Param]) -> StatorResult<Self> {
        let param_count = params.len() as u32;
        let mut compiler = Self {
            instructions: Vec::new(),
            constant_pool: Vec::new(),
            allocator: RegisterAllocator::new(param_count),
            scopes: vec![HashMap::new()],
            source_positions: Vec::new(),
            labels: Vec::new(),
            loop_stack: Vec::new(),
            param_count,
            slot_kinds: Vec::new(),
            handler_table: Vec::new(),
        };
        for (i, param) in params.iter().enumerate() {
            match &param.pat {
                Pat::Ident(ident) => {
                    let reg = Register::parameter(i as u32);
                    compiler.scopes[0].insert(ident.name.clone(), reg);
                }
                _ => {
                    return Err(StatorError::Internal(
                        "complex parameter patterns are not yet supported".into(),
                    ));
                }
            }
        }
        Ok(compiler)
    }

    // ── Constant pool ────────────────────────────────────────────────────────

    /// Add a `Number` entry, reusing an existing entry with the same bit
    /// pattern to avoid duplicates.
    fn add_number(&mut self, value: f64) -> u32 {
        let bits = value.to_bits();
        for (i, e) in self.constant_pool.iter().enumerate() {
            if let ConstantPoolEntry::Number(v) = e
                && v.to_bits() == bits
            {
                return i as u32;
            }
        }
        let idx = self.constant_pool.len() as u32;
        self.constant_pool.push(ConstantPoolEntry::Number(value));
        idx
    }

    /// Add a `String` entry, reusing an existing identical entry.
    fn add_string(&mut self, s: &str) -> u32 {
        for (i, e) in self.constant_pool.iter().enumerate() {
            if let ConstantPoolEntry::String(v) = e
                && v == s
            {
                return i as u32;
            }
        }
        let idx = self.constant_pool.len() as u32;
        self.constant_pool
            .push(ConstantPoolEntry::String(s.to_owned()));
        idx
    }

    /// Add a [`ConstantPoolEntry`] without deduplication.
    fn add_constant_raw(&mut self, entry: ConstantPoolEntry) -> u32 {
        let idx = self.constant_pool.len() as u32;
        self.constant_pool.push(entry);
        idx
    }

    // ── Feedback slots ───────────────────────────────────────────────────────

    /// Allocate a new feedback slot with the given [`FeedbackSlotKind`] and
    /// return its zero-based index as a [`Operand::FeedbackSlot`] operand.
    fn alloc_slot(&mut self, kind: FeedbackSlotKind) -> Operand {
        let idx = self.slot_kinds.len() as u32;
        self.slot_kinds.push(kind);
        Operand::FeedbackSlot(idx)
    }

    // ── Labels ───────────────────────────────────────────────────────────────

    /// Allocate a new unbound label and return its ID.
    fn new_label(&mut self) -> usize {
        let id = self.labels.len();
        self.labels.push(Label::new());
        id
    }

    /// Bind `label_id` to the *next* instruction to be emitted.
    fn bind_label(&mut self, label_id: usize) {
        self.labels[label_id].bound_at = Some(self.instructions.len());
    }

    // ── Instruction emission ─────────────────────────────────────────────────

    /// Append an instruction and return its index.
    fn emit(&mut self, instr: Instruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Emit a jump instruction targeting `label_id`.
    ///
    /// The offset operand is left as `JumpOffset(0)` and patched during
    /// [`finalize`].  The instruction index is recorded in the label's
    /// ref list.
    ///
    /// `opcode` must be a single-operand jump (`Jump`, `JumpIfToBooleanFalse`,
    /// etc.).
    fn emit_jump(&mut self, opcode: Opcode, label_id: usize) -> usize {
        let idx = self.instructions.len();
        self.emit(Instruction::new_unchecked(
            opcode,
            vec![Operand::JumpOffset(0)],
        ));
        self.labels[label_id].refs.push(idx);
        idx
    }

    // ── Register helpers ─────────────────────────────────────────────────────

    /// Store the accumulator into `reg`.
    fn emit_star(&mut self, reg: Register) {
        self.emit(Instruction::new_unchecked(
            Opcode::Star,
            vec![to_reg_op(reg)],
        ));
    }

    /// Load `reg` into the accumulator.
    fn emit_ldar(&mut self, reg: Register) {
        self.emit(Instruction::new_unchecked(
            Opcode::Ldar,
            vec![to_reg_op(reg)],
        ));
    }

    // ── Scope management ─────────────────────────────────────────────────────

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Define a new local variable in the innermost scope and return its
    /// register.
    fn define_local(&mut self, name: &str) -> Register {
        let reg = self.allocator.new_local();
        self.scopes.last_mut().unwrap().insert(name.to_owned(), reg);
        reg
    }

    /// Look up a variable in the scope chain (innermost first).
    ///
    /// Returns `None` if the name is not found (assumed to be global).
    fn lookup_var(&self, name: &str) -> Option<Register> {
        for scope in self.scopes.iter().rev() {
            if let Some(&reg) = scope.get(name) {
                return Some(reg);
            }
        }
        None
    }

    // ── Statement compilation ─────────────────────────────────────────────────

    /// Compile a single statement, emitting bytecode into `self.instructions`.
    fn compile_stmt(&mut self, stmt: &Stmt) -> StatorResult<()> {
        match stmt {
            Stmt::Block(s) => self.compile_block(s),
            Stmt::VarDecl(s) => self.compile_var_decl(s),
            Stmt::FnDecl(s) => self.compile_fn_decl(s),
            Stmt::Expr(s) => {
                self.compile_expr(&s.expr)?;
                Ok(())
            }
            Stmt::If(s) => self.compile_if(s),
            Stmt::While(s) => self.compile_while(s),
            Stmt::DoWhile(s) => self.compile_do_while(s),
            Stmt::For(s) => self.compile_for(s),
            Stmt::Return(s) => self.compile_return(s),
            Stmt::Throw(s) => {
                self.compile_expr(&s.argument)?;
                self.emit(Instruction::new_unchecked(Opcode::Throw, vec![]));
                Ok(())
            }
            Stmt::Try(s) => self.compile_try(s),
            Stmt::Break(s) => self.compile_break(s),
            Stmt::Continue(s) => self.compile_continue(s),
            Stmt::Labeled(s) => self.compile_stmt(&s.body),
            Stmt::Debugger(_) => {
                self.emit(Instruction::new_unchecked(Opcode::Debugger, vec![]));
                Ok(())
            }
            Stmt::Empty(_) => Ok(()),
            Stmt::Switch(s) => self.compile_switch(s),
            Stmt::ForIn(_) | Stmt::ForOf(_) | Stmt::With(_) | Stmt::ClassDecl(_) => Err(
                StatorError::Internal(format!("{} is not yet supported", stmt_kind(stmt))),
            ),
        }
    }

    /// Compile a `{ … }` block, pushing/popping a scope.
    fn compile_block(&mut self, block: &BlockStmt) -> StatorResult<()> {
        self.push_scope();
        for stmt in &block.body {
            self.compile_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    /// Compile `var` / `let` / `const` declarations.
    fn compile_var_decl(&mut self, decl: &VarDecl) -> StatorResult<()> {
        for declarator in &decl.declarators {
            self.compile_var_declarator(declarator)?;
        }
        Ok(())
    }

    fn compile_var_declarator(&mut self, declarator: &VarDeclarator) -> StatorResult<()> {
        match &declarator.id {
            Pat::Ident(ident) => {
                // Allocate the local register first.
                let reg = self.define_local(&ident.name);
                // Compile the initializer if present; otherwise load undefined.
                if let Some(init) = &declarator.init {
                    self.compile_expr(init)?;
                } else {
                    self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                }
                self.emit_star(reg);
                Ok(())
            }
            _ => Err(StatorError::Internal(
                "destructuring declarations are not yet supported".into(),
            )),
        }
    }

    /// Compile a function declaration: compile the function body, add the
    /// resulting [`BytecodeArray`] to the constant pool, emit `CreateClosure`,
    /// and bind the name to the resulting register.
    fn compile_fn_decl(&mut self, decl: &FnDecl) -> StatorResult<()> {
        if decl.is_generator || decl.is_async {
            return Err(StatorError::Internal(
                "generator and async functions are not yet supported".into(),
            ));
        }
        let func_array = compile_function(&decl.params, &decl.body)?;
        let pool_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(func_array)));
        // Emit CreateClosure: [func_idx, slot, flags]
        let slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![Operand::ConstantPoolIdx(pool_idx), slot, Operand::Flag(0)],
        ));
        // Bind the name in the current scope (accumulator holds the closure).
        if let Some(id) = &decl.id {
            let reg = self.define_local(&id.name);
            self.emit_star(reg);
        }
        Ok(())
    }

    /// Compile an `if (test) consequent [else alternate]` statement.
    fn compile_if(&mut self, s: &crate::parser::ast::IfStmt) -> StatorResult<()> {
        let else_label = self.new_label();

        self.compile_expr(&s.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, else_label);
        self.compile_stmt(&s.consequent)?;

        if let Some(alt) = &s.alternate {
            let end_label = self.new_label();
            self.emit_jump(Opcode::Jump, end_label);
            self.bind_label(else_label);
            self.compile_stmt(alt)?;
            self.bind_label(end_label);
        } else {
            self.bind_label(else_label);
        }
        Ok(())
    }

    /// Compile a `while (test) body` statement.
    fn compile_while(&mut self, s: &crate::parser::ast::WhileStmt) -> StatorResult<()> {
        let loop_start = self.new_label();
        let loop_end = self.new_label();

        self.loop_stack.push((loop_start, loop_end));

        self.bind_label(loop_start);
        self.compile_expr(&s.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, loop_end);
        self.compile_stmt(&s.body)?;
        self.emit_jump(Opcode::Jump, loop_start);
        self.bind_label(loop_end);

        self.loop_stack.pop();
        Ok(())
    }

    /// Compile a `do body while (test)` statement.
    fn compile_do_while(&mut self, s: &crate::parser::ast::DoWhileStmt) -> StatorResult<()> {
        let loop_start = self.new_label();
        let loop_end = self.new_label();
        // continue target is the condition check, placed after body
        let cond_label = self.new_label();

        self.loop_stack.push((cond_label, loop_end));

        self.bind_label(loop_start);
        self.compile_stmt(&s.body)?;
        self.bind_label(cond_label);
        self.compile_expr(&s.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanTrue, loop_start);
        self.bind_label(loop_end);

        self.loop_stack.pop();
        Ok(())
    }

    /// Compile a `for (init; test; update) body` statement.
    fn compile_for(&mut self, s: &ForStmt) -> StatorResult<()> {
        let loop_start = self.new_label();
        let loop_end = self.new_label();
        let continue_label = self.new_label();

        self.push_scope();

        // Initializer.
        if let Some(init) = &s.init {
            match init {
                ForInit::VarDecl(decl) => self.compile_var_decl(decl)?,
                ForInit::Expr(expr) => {
                    self.compile_expr(expr)?;
                }
            }
        }

        self.loop_stack.push((continue_label, loop_end));

        self.bind_label(loop_start);

        // Test.
        if let Some(test) = &s.test {
            self.compile_expr(test)?;
            self.emit_jump(Opcode::JumpIfToBooleanFalse, loop_end);
        }

        self.compile_stmt(&s.body)?;

        // Update.
        self.bind_label(continue_label);
        if let Some(update) = &s.update {
            self.compile_expr(update)?;
        }

        self.emit_jump(Opcode::Jump, loop_start);
        self.bind_label(loop_end);

        self.loop_stack.pop();
        self.pop_scope();
        Ok(())
    }

    /// Compile a `return [argument]` statement.
    fn compile_return(&mut self, s: &crate::parser::ast::ReturnStmt) -> StatorResult<()> {
        if let Some(arg) = &s.argument {
            self.compile_expr(arg)?;
        } else {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        }
        self.emit(Instruction::new_unchecked(Opcode::Return, vec![]));
        Ok(())
    }

    /// Compile a `break [label]` statement.
    ///
    /// Only unlabeled breaks are supported; labeled breaks return an error.
    fn compile_break(&mut self, s: &crate::parser::ast::BreakStmt) -> StatorResult<()> {
        if s.label.is_some() {
            return Err(StatorError::Internal(
                "labeled break is not yet supported".into(),
            ));
        }
        let (_, break_label) = *self
            .loop_stack
            .last()
            .ok_or_else(|| StatorError::Internal("break outside loop".into()))?;
        self.emit_jump(Opcode::Jump, break_label);
        Ok(())
    }

    /// Compile a `continue [label]` statement.
    fn compile_continue(&mut self, s: &crate::parser::ast::ContinueStmt) -> StatorResult<()> {
        if s.label.is_some() {
            return Err(StatorError::Internal(
                "labeled continue is not yet supported".into(),
            ));
        }
        let (continue_label, _) = *self
            .loop_stack
            .last()
            .ok_or_else(|| StatorError::Internal("continue outside loop".into()))?;
        self.emit_jump(Opcode::Jump, continue_label);
        Ok(())
    }

    /// Compile a `switch (disc) { cases }` statement.
    ///
    /// Each case is compiled as a sequence of equality-check + conditional
    /// jumps.  Fall-through is implemented by *not* emitting an end-of-case
    /// jump.
    fn compile_switch(&mut self, s: &crate::parser::ast::SwitchStmt) -> StatorResult<()> {
        let break_label = self.new_label();
        let end_label = break_label;

        // Evaluate the discriminant once and save it.
        self.compile_expr(&s.discriminant)?;
        let disc_reg = self.allocator.allocate_temporary();
        self.emit_star(disc_reg);

        // Build a label for each case clause.
        let case_labels: Vec<usize> = s.cases.iter().map(|_| self.new_label()).collect();

        // Emit the comparison chain.
        let mut default_label: Option<usize> = None;
        for (i, case) in s.cases.iter().enumerate() {
            if let Some(test) = &case.test {
                self.compile_expr(test)?;
                let slot = self.alloc_slot(FeedbackSlotKind::Compare);
                self.emit(Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![to_reg_op(disc_reg), slot],
                ));
                self.emit_jump(Opcode::JumpIfToBooleanTrue, case_labels[i]);
            } else {
                default_label = Some(case_labels[i]);
            }
        }

        // If no case matched, jump to default or skip.
        if let Some(dl) = default_label {
            self.emit_jump(Opcode::Jump, dl);
        } else {
            self.emit_jump(Opcode::Jump, end_label);
        }

        self.loop_stack.push((end_label, break_label));

        // Emit case bodies.
        for (i, case) in s.cases.iter().enumerate() {
            self.bind_label(case_labels[i]);
            for stmt in &case.consequent {
                self.compile_stmt(stmt)?;
            }
        }

        self.loop_stack.pop();

        self.bind_label(end_label);
        self.allocator
            .release_temporary(disc_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a `try { … } catch (e) { … } finally { … }` statement.
    ///
    /// Emits bytecode and registers [`HandlerTableEntry`] records so the
    /// interpreter can dispatch thrown values to the correct handler.
    ///
    /// ## Layout — try/catch
    ///
    /// ```text
    /// [try_start]  try body
    /// [try_end]    Jump(after_catch)   ← normal exit skips catch
    /// [catch_start] Star(e)            ← acc holds thrown value
    ///               catch body
    /// [after_catch] ...
    /// ```
    ///
    /// ## Layout — try/finally (no catch)
    ///
    /// The finally block is duplicated: once inline for the normal path, and
    /// once inside an exception handler that re-throws after running finally.
    ///
    /// ```text
    /// [try_start]   try body
    /// [try_end]     finally body (normal path, inlined)
    ///               Jump(after_ex_handler)
    /// [ex_handler]  Star(ex_reg)        ← save exception
    ///               finally body (exception path, inlined)
    ///               Ldar(ex_reg)
    ///               ReThrow
    /// [after_ex_handler] ...
    /// ```
    fn compile_try(&mut self, s: &crate::parser::ast::TryStmt) -> StatorResult<()> {
        let try_start = self.instructions.len() as u32;

        // Compile try block.
        self.compile_block(&s.block)?;

        let try_end = self.instructions.len() as u32;

        if let Some(handler) = &s.handler {
            // ── try / catch [ / finally ] ────────────────────────────────────
            let after_catch_label = self.new_label();

            // Normal exit: jump past catch handler.
            self.emit_jump(Opcode::Jump, after_catch_label);

            // Catch handler: accumulator holds the thrown value.
            let catch_start = self.instructions.len() as u32;
            self.push_scope();
            if let Some(param) = &handler.param {
                match param {
                    Pat::Ident(ident) => {
                        let reg = self.define_local(&ident.name);
                        self.emit_star(reg);
                    }
                    _ => {
                        return Err(StatorError::Internal(
                            "destructuring catch parameter is not yet supported".into(),
                        ));
                    }
                }
            }
            self.compile_block(&handler.body)?;
            self.pop_scope();

            self.bind_label(after_catch_label);

            // Register the catch handler entry.
            self.handler_table.push(HandlerTableEntry {
                try_start,
                try_end,
                handler: catch_start,
                is_finally: false,
            });

            // Compile finally block inline (runs on both paths).
            if let Some(finalizer) = &s.finalizer {
                self.compile_block(finalizer)?;
            }
        } else if let Some(finalizer) = &s.finalizer {
            // ── try / finally (no catch) ─────────────────────────────────────
            //
            // Normal path: inline the finally body, then jump past the
            // exception-path handler.  Exception path: save the exception,
            // run finally, then re-throw.

            // Normal path: inline finally.
            self.compile_block(finalizer)?;

            // Jump past exception-path handler.
            let after_ex_handler_label = self.new_label();
            self.emit_jump(Opcode::Jump, after_ex_handler_label);

            // Exception-path handler.
            let ex_handler_start = self.instructions.len() as u32;
            let ex_reg = self.allocator.allocate_temporary();
            self.emit_star(ex_reg); // save thrown value from acc
            self.compile_block(finalizer)?; // run finally on exception path
            self.emit_ldar(ex_reg); // reload thrown value
            self.emit(Instruction::new_unchecked(Opcode::ReThrow, vec![]));
            self.allocator
                .release_temporary(ex_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;

            self.bind_label(after_ex_handler_label);

            // Register the finally handler entry.
            self.handler_table.push(HandlerTableEntry {
                try_start,
                try_end,
                handler: ex_handler_start,
                is_finally: true,
            });
        }

        Ok(())
    }

    // ── Expression compilation ────────────────────────────────────────────────

    /// Compile an expression, leaving the result in the accumulator.
    fn compile_expr(&mut self, expr: &Expr) -> StatorResult<()> {
        match expr {
            // ── Literals ──────────────────────────────────────────────────
            Expr::Null(_) => {
                self.emit(Instruction::new_unchecked(Opcode::LdaNull, vec![]));
                Ok(())
            }
            Expr::Bool(b) => {
                let op = if b.value {
                    Opcode::LdaTrue
                } else {
                    Opcode::LdaFalse
                };
                self.emit(Instruction::new_unchecked(op, vec![]));
                Ok(())
            }
            Expr::Num(n) => {
                self.compile_number(n.value);
                Ok(())
            }
            Expr::Str(s) => {
                let idx = self.add_string(&s.value);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaConstant,
                    vec![Operand::ConstantPoolIdx(idx)],
                ));
                Ok(())
            }
            Expr::BigInt(_) => Err(StatorError::Internal(
                "BigInt literals are not yet supported".into(),
            )),
            Expr::Regexp(r) => {
                let pattern_idx = self.add_string(&r.pattern);
                // Encode flags as a bitfield: g=1, i=2, m=4, s=8, u=16, y=32
                let flags_val: u8 = r.flags.bytes().fold(0u8, |acc, b| {
                    let bit = match b {
                        b'g' => 0x01,
                        b'i' => 0x02,
                        b'm' => 0x04,
                        b's' => 0x08,
                        b'u' => 0x10,
                        b'y' => 0x20,
                        _ => 0,
                    };
                    acc | bit
                });
                let slot = self.alloc_slot(FeedbackSlotKind::Literal);
                self.emit(Instruction::new_unchecked(
                    Opcode::CreateRegExpLiteral,
                    vec![
                        Operand::ConstantPoolIdx(pattern_idx),
                        slot,
                        Operand::Flag(flags_val),
                    ],
                ));
                Ok(())
            }
            Expr::Template(t) => self.compile_template(t),

            // ── Identifier ────────────────────────────────────────────────
            Expr::Ident(id) => {
                self.compile_ident_load(&id.name);
                Ok(())
            }
            Expr::This(_) => {
                // `this` is implicitly the receiver; load from a special slot.
                // We represent it as a named global lookup for now.
                let name_idx = self.add_string("this");
                let slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaGlobal,
                    vec![Operand::ConstantPoolIdx(name_idx), slot],
                ));
                Ok(())
            }

            // ── Objects / Arrays ──────────────────────────────────────────
            Expr::Array(a) => self.compile_array(a),
            Expr::Object(o) => self.compile_object(o),

            // ── Function-like ─────────────────────────────────────────────
            Expr::Fn(f) => self.compile_fn_expr(f),
            Expr::Arrow(a) => self.compile_arrow_expr(a),
            Expr::Class(_) => Err(StatorError::Internal(
                "class expressions are not yet supported".into(),
            )),

            // ── Operators ─────────────────────────────────────────────────
            Expr::Unary(u) => self.compile_unary(u),
            Expr::Update(u) => self.compile_update(u),
            Expr::Binary(b) => self.compile_binary(b),
            Expr::Logical(l) => self.compile_logical(l),
            Expr::Conditional(c) => self.compile_conditional(c),
            Expr::Assign(a) => self.compile_assign(a),
            Expr::Sequence(s) => {
                for expr in &s.expressions {
                    self.compile_expr(expr)?;
                }
                Ok(())
            }

            // ── Member / call ─────────────────────────────────────────────
            Expr::Member(m) => self.compile_member(m),
            Expr::OptionalMember(m) => self.compile_optional_member(m),
            Expr::Call(c) => self.compile_call(c),
            Expr::OptionalCall(c) => self.compile_optional_call(c),
            Expr::New(n) => self.compile_new(n),

            // ── Async / generators ────────────────────────────────────────
            Expr::Yield(_) | Expr::Await(_) => Err(StatorError::Internal(
                "yield/await are not yet supported".into(),
            )),

            Expr::TaggedTemplate(_) => Err(StatorError::Internal(
                "tagged template literals are not yet supported".into(),
            )),
            Expr::Spread(_) => Err(StatorError::Internal(
                "spread in expression position is not yet supported".into(),
            )),
            Expr::Import(_) => Err(StatorError::Internal(
                "dynamic import() is not yet supported".into(),
            )),
            Expr::MetaProp(_) => Err(StatorError::Internal(
                "import.meta / new.target are not yet supported".into(),
            )),
        }
    }

    /// Emit the optimal load for a numeric constant.
    fn compile_number(&mut self, value: f64) {
        // Prefer LdaZero or LdaSmi when the value fits.
        if value == 0.0 && value.is_sign_positive() {
            self.emit(Instruction::new_unchecked(Opcode::LdaZero, vec![]));
        } else if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
            let smi = value as i32;
            self.emit(Instruction::new_unchecked(
                Opcode::LdaSmi,
                vec![Operand::Immediate(smi)],
            ));
        } else {
            let idx = self.add_number(value);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaConstant,
                vec![Operand::ConstantPoolIdx(idx)],
            ));
        }
    }

    /// Load a variable by name.
    ///
    /// If the name is in the scope chain, emit `Ldar`.  Otherwise assume it is
    /// a global and emit `LdaGlobal`.
    fn compile_ident_load(&mut self, name: &str) {
        if let Some(reg) = self.lookup_var(name) {
            self.emit_ldar(reg);
        } else {
            let name_idx = self.add_string(name);
            let slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
    }

    /// Store the accumulator to the variable named `name`.
    fn compile_ident_store(&mut self, name: &str) {
        if let Some(reg) = self.lookup_var(name) {
            self.emit_star(reg);
        } else {
            let name_idx = self.add_string(name);
            let slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::StaGlobal,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
    }

    /// Compile a unary expression.
    fn compile_unary(&mut self, u: &crate::parser::ast::UnaryExpr) -> StatorResult<()> {
        match u.op {
            UnaryOp::Void => {
                self.compile_expr(&u.argument)?;
                self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            }
            UnaryOp::Typeof => {
                self.compile_expr(&u.argument)?;
                let slot = self.alloc_slot(FeedbackSlotKind::TypeOf);
                self.emit(Instruction::new_unchecked(Opcode::TypeOf, vec![slot]));
            }
            UnaryOp::Not => {
                self.compile_expr(&u.argument)?;
                self.emit(Instruction::new_unchecked(
                    Opcode::ToBooleanLogicalNot,
                    vec![],
                ));
            }
            UnaryOp::Minus => {
                self.compile_expr(&u.argument)?;
                let slot = self.alloc_slot(FeedbackSlotKind::UnaryOp);
                self.emit(Instruction::new_unchecked(Opcode::Negate, vec![slot]));
            }
            UnaryOp::Plus => {
                self.compile_expr(&u.argument)?;
                let slot = self.alloc_slot(FeedbackSlotKind::UnaryOp);
                self.emit(Instruction::new_unchecked(Opcode::ToNumber, vec![slot]));
            }
            UnaryOp::BitNot => {
                self.compile_expr(&u.argument)?;
                let slot = self.alloc_slot(FeedbackSlotKind::UnaryOp);
                self.emit(Instruction::new_unchecked(Opcode::BitwiseNot, vec![slot]));
            }
            UnaryOp::Delete => {
                // Simplified: compile argument, emit sloppy-mode delete.
                match u.argument.as_ref() {
                    Expr::Member(m) => {
                        self.compile_expr(&m.object)?;
                        let obj_reg = self.allocator.allocate_temporary();
                        self.emit_star(obj_reg);
                        match &m.property {
                            crate::parser::ast::MemberProp::Computed(key) => {
                                self.compile_expr(key)?;
                            }
                            crate::parser::ast::MemberProp::Ident(id) => {
                                let idx = self.add_string(&id.name);
                                self.emit(Instruction::new_unchecked(
                                    Opcode::LdaConstant,
                                    vec![Operand::ConstantPoolIdx(idx)],
                                ));
                            }
                            crate::parser::ast::MemberProp::Private(_) => {
                                return Err(StatorError::Internal(
                                    "delete of private property is not supported".into(),
                                ));
                            }
                        }
                        self.emit(Instruction::new_unchecked(
                            Opcode::DeletePropertySloppy,
                            vec![to_reg_op(obj_reg)],
                        ));
                        self.allocator
                            .release_temporary(obj_reg)
                            .map_err(|e| StatorError::Internal(e.to_string()))?;
                    }
                    _ => {
                        // delete on non-member always returns true.
                        self.emit(Instruction::new_unchecked(Opcode::LdaTrue, vec![]));
                    }
                }
            }
        }
        Ok(())
    }

    /// Compile a `++`/`--` update expression.
    fn compile_update(&mut self, u: &crate::parser::ast::UpdateExpr) -> StatorResult<()> {
        // Load the current value.
        self.compile_expr(&u.argument)?;
        let old_reg = if !u.prefix {
            // Post: save old value.
            let r = self.allocator.allocate_temporary();
            self.emit_star(r);
            Some(r)
        } else {
            None
        };
        // Increment / decrement.
        let op = match u.op {
            UpdateOp::Increment => Opcode::Inc,
            UpdateOp::Decrement => Opcode::Dec,
        };
        let slot = self.alloc_slot(FeedbackSlotKind::BinaryOpInc);
        self.emit(Instruction::new_unchecked(op, vec![slot]));
        // Store updated value back to the target.
        match u.argument.as_ref() {
            Expr::Ident(id) => self.compile_ident_store(&id.name),
            Expr::Member(m) => self.compile_member_store(m)?,
            _ => {
                return Err(StatorError::Internal(
                    "update target must be an identifier or member expression".into(),
                ));
            }
        }
        // For post-update, reload the old value.
        if let Some(r) = old_reg {
            self.emit_ldar(r);
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        Ok(())
    }

    /// Compile a binary expression.
    ///
    /// The pattern is: evaluate RHS → temporary, evaluate LHS → accumulator,
    /// apply operator.  This works for all non-commutative ops.
    fn compile_binary(&mut self, b: &crate::parser::ast::BinaryExpr) -> StatorResult<()> {
        // Evaluate RHS first, save to temporary.
        self.compile_expr(&b.right)?;
        let rhs_reg = self.allocator.allocate_temporary();
        self.emit_star(rhs_reg);

        // Evaluate LHS into accumulator.
        self.compile_expr(&b.left)?;

        let (op, slot_kind): (Opcode, FeedbackSlotKind) = match b.op {
            BinaryOp::Add => (Opcode::Add, FeedbackSlotKind::BinaryOp),
            BinaryOp::Sub => (Opcode::Sub, FeedbackSlotKind::BinaryOp),
            BinaryOp::Mul => (Opcode::Mul, FeedbackSlotKind::BinaryOp),
            BinaryOp::Div => (Opcode::Div, FeedbackSlotKind::BinaryOp),
            BinaryOp::Rem => (Opcode::Mod, FeedbackSlotKind::BinaryOp),
            BinaryOp::Exp => (Opcode::Exp, FeedbackSlotKind::BinaryOp),
            BinaryOp::BitOr => (Opcode::BitwiseOr, FeedbackSlotKind::BinaryOp),
            BinaryOp::BitXor => (Opcode::BitwiseXor, FeedbackSlotKind::BinaryOp),
            BinaryOp::BitAnd => (Opcode::BitwiseAnd, FeedbackSlotKind::BinaryOp),
            BinaryOp::Shl => (Opcode::ShiftLeft, FeedbackSlotKind::BinaryOp),
            BinaryOp::Shr => (Opcode::ShiftRight, FeedbackSlotKind::BinaryOp),
            BinaryOp::UShr => (Opcode::ShiftRightLogical, FeedbackSlotKind::BinaryOp),
            BinaryOp::Eq => (Opcode::TestEqual, FeedbackSlotKind::Compare),
            BinaryOp::NotEq => (Opcode::TestNotEqual, FeedbackSlotKind::Compare),
            BinaryOp::StrictEq => (Opcode::TestEqualStrict, FeedbackSlotKind::Compare),
            BinaryOp::StrictNotEq => {
                // a !== b  →  !(a === b)
                let slot = self.alloc_slot(FeedbackSlotKind::Compare);
                self.emit(Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![to_reg_op(rhs_reg), slot],
                ));
                self.allocator
                    .release_temporary(rhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.emit(Instruction::new_unchecked(Opcode::LogicalNot, vec![]));
                return Ok(());
            }
            BinaryOp::Lt => (Opcode::TestLessThan, FeedbackSlotKind::Compare),
            BinaryOp::LtEq => (Opcode::TestLessThanOrEqual, FeedbackSlotKind::Compare),
            BinaryOp::Gt => (Opcode::TestGreaterThan, FeedbackSlotKind::Compare),
            BinaryOp::GtEq => (Opcode::TestGreaterThanOrEqual, FeedbackSlotKind::Compare),
            BinaryOp::In => (Opcode::TestIn, FeedbackSlotKind::BinaryOp),
            BinaryOp::Instanceof => (Opcode::TestInstanceOf, FeedbackSlotKind::InstanceOf),
        };
        let slot = self.alloc_slot(slot_kind);
        self.emit(Instruction::new_unchecked(
            op,
            vec![to_reg_op(rhs_reg), slot],
        ));
        self.allocator
            .release_temporary(rhs_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile `&&`, `||`, or `??` (short-circuit logic).
    fn compile_logical(&mut self, l: &crate::parser::ast::LogicalExpr) -> StatorResult<()> {
        let end_label = self.new_label();

        self.compile_expr(&l.left)?;

        match l.op {
            LogicalOp::And => {
                // If left is falsy, skip right (result = left).
                self.emit_jump(Opcode::JumpIfToBooleanFalse, end_label);
            }
            LogicalOp::Or => {
                // If left is truthy, skip right (result = left).
                self.emit_jump(Opcode::JumpIfToBooleanTrue, end_label);
            }
            LogicalOp::NullishCoalesce => {
                // a ?? b: if a is null or undefined, use b; otherwise keep a.
                let eval_right_label = self.new_label();
                self.compile_expr(&l.left)?;
                // Jump to right side if left is null or undefined.
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, eval_right_label);
                // Left is neither null nor undefined: already in acc, jump to end.
                self.emit_jump(Opcode::Jump, end_label);
                self.bind_label(eval_right_label);
                self.compile_expr(&l.right)?;
                self.bind_label(end_label);
                return Ok(());
            }
        }

        self.compile_expr(&l.right)?;
        self.bind_label(end_label);
        Ok(())
    }

    /// Compile `test ? consequent : alternate`.
    fn compile_conditional(&mut self, c: &crate::parser::ast::ConditionalExpr) -> StatorResult<()> {
        let else_label = self.new_label();
        let end_label = self.new_label();

        self.compile_expr(&c.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, else_label);
        self.compile_expr(&c.consequent)?;
        self.emit_jump(Opcode::Jump, end_label);
        self.bind_label(else_label);
        self.compile_expr(&c.alternate)?;
        self.bind_label(end_label);
        Ok(())
    }

    /// Compile an assignment expression.
    fn compile_assign(&mut self, a: &crate::parser::ast::AssignExpr) -> StatorResult<()> {
        // For simple assignment, compile RHS and store.
        if a.op == AssignOp::Assign {
            self.compile_expr(&a.right)?;
            self.compile_assign_target_store(&a.left)?;
            return Ok(());
        }
        // For compound assignment (+=, -=, …): load LHS, compile RHS, op, store.
        self.compile_assign_target_load(&a.left)?;
        let lhs_reg = self.allocator.allocate_temporary();
        self.emit_star(lhs_reg);

        self.compile_expr(&a.right)?;
        let rhs_reg = self.allocator.allocate_temporary();
        self.emit_star(rhs_reg);

        // Load LHS back into accumulator.
        self.emit_ldar(lhs_reg);

        let op: Opcode = match a.op {
            AssignOp::AddAssign => Opcode::Add,
            AssignOp::SubAssign => Opcode::Sub,
            AssignOp::MulAssign => Opcode::Mul,
            AssignOp::DivAssign => Opcode::Div,
            AssignOp::RemAssign => Opcode::Mod,
            AssignOp::ExpAssign => Opcode::Exp,
            AssignOp::ShlAssign => Opcode::ShiftLeft,
            AssignOp::ShrAssign => Opcode::ShiftRight,
            AssignOp::UShrAssign => Opcode::ShiftRightLogical,
            AssignOp::BitOrAssign => Opcode::BitwiseOr,
            AssignOp::BitXorAssign => Opcode::BitwiseXor,
            AssignOp::BitAndAssign => Opcode::BitwiseAnd,
            AssignOp::LogicalAndAssign => {
                // a &&= b  →  if a is falsy, keep a; else a = b
                self.allocator
                    .release_temporary(rhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(lhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                // Re-load LHS.
                self.compile_assign_target_load(&a.left)?;
                let skip = self.new_label();
                self.emit_jump(Opcode::JumpIfToBooleanFalse, skip);
                self.compile_expr(&a.right)?;
                self.compile_assign_target_store(&a.left)?;
                self.bind_label(skip);
                return Ok(());
            }
            AssignOp::LogicalOrAssign => {
                self.allocator
                    .release_temporary(rhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(lhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.compile_assign_target_load(&a.left)?;
                let skip = self.new_label();
                self.emit_jump(Opcode::JumpIfToBooleanTrue, skip);
                self.compile_expr(&a.right)?;
                self.compile_assign_target_store(&a.left)?;
                self.bind_label(skip);
                return Ok(());
            }
            AssignOp::NullishAssign => {
                self.allocator
                    .release_temporary(rhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(lhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.compile_assign_target_load(&a.left)?;
                let skip = self.new_label();
                self.emit_jump(Opcode::JumpIfNotUndefined, skip);
                let skip2 = self.new_label();
                self.emit_jump(Opcode::JumpIfNotNull, skip2);
                self.compile_expr(&a.right)?;
                self.compile_assign_target_store(&a.left)?;
                self.bind_label(skip);
                self.bind_label(skip2);
                return Ok(());
            }
            AssignOp::Assign => unreachable!("handled above"),
        };

        let slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
        self.emit(Instruction::new_unchecked(
            op,
            vec![to_reg_op(rhs_reg), slot],
        ));
        self.allocator
            .release_temporary(rhs_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(lhs_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.compile_assign_target_store(&a.left)?;
        Ok(())
    }

    /// Load an assignment target into the accumulator.
    fn compile_assign_target_load(&mut self, target: &AssignTarget) -> StatorResult<()> {
        match target {
            AssignTarget::Expr(expr) => match expr.as_ref() {
                Expr::Ident(id) => {
                    self.compile_ident_load(&id.name);
                    Ok(())
                }
                Expr::Member(m) => self.compile_member(m),
                _ => Err(StatorError::Internal(
                    "unsupported assignment target".into(),
                )),
            },
            AssignTarget::Pat(_) => Err(StatorError::Internal(
                "destructuring assignment is not yet supported".into(),
            )),
        }
    }

    /// Store the accumulator to an assignment target.
    fn compile_assign_target_store(&mut self, target: &AssignTarget) -> StatorResult<()> {
        match target {
            AssignTarget::Expr(expr) => match expr.as_ref() {
                Expr::Ident(id) => {
                    self.compile_ident_store(&id.name);
                    Ok(())
                }
                Expr::Member(m) => self.compile_member_store(m),
                _ => Err(StatorError::Internal(
                    "unsupported assignment target".into(),
                )),
            },
            AssignTarget::Pat(_) => Err(StatorError::Internal(
                "destructuring assignment is not yet supported".into(),
            )),
        }
    }

    /// Compile a member expression (`obj.prop` or `obj[key]`) as an r-value.
    fn compile_member(&mut self, m: &crate::parser::ast::MemberExpr) -> StatorResult<()> {
        self.compile_expr(&m.object)?;
        let obj_reg = self.allocator.allocate_temporary();
        self.emit_star(obj_reg);
        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                ));
            }
            crate::parser::ast::MemberProp::Computed(key) => {
                self.compile_expr(key)?;
                // Keyed property: accumulator = key_expr, obj_reg = object
                // But LdaKeyedProperty is [obj, slot], with key in accumulator.
                let slot = self.alloc_slot(FeedbackSlotKind::KeyedLoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![to_reg_op(obj_reg), slot],
                ));
            }
            crate::parser::ast::MemberProp::Private(_) => {
                return Err(StatorError::Internal(
                    "private member access is not yet supported".into(),
                ));
            }
        }
        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile an optional member expression (`obj?.prop`).
    fn compile_optional_member(
        &mut self,
        m: &crate::parser::ast::OptionalMemberExpr,
    ) -> StatorResult<()> {
        let null_label = self.new_label();
        let end_label = self.new_label();

        self.compile_expr(&m.object)?;
        let obj_reg = self.allocator.allocate_temporary();
        self.emit_star(obj_reg);

        // If the object is null or undefined, short-circuit to undefined.
        self.emit_ldar(obj_reg);
        self.emit_jump(Opcode::JumpIfUndefinedOrNull, null_label);

        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                ));
            }
            crate::parser::ast::MemberProp::Computed(key) => {
                self.compile_expr(key)?;
                let slot = self.alloc_slot(FeedbackSlotKind::KeyedLoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![to_reg_op(obj_reg), slot],
                ));
            }
            crate::parser::ast::MemberProp::Private(_) => {
                return Err(StatorError::Internal(
                    "private member access is not yet supported".into(),
                ));
            }
        }
        self.emit_jump(Opcode::Jump, end_label);

        self.bind_label(null_label);
        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        self.bind_label(end_label);

        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Store the accumulator as a property of a member expression target.
    fn compile_member_store(&mut self, m: &crate::parser::ast::MemberExpr) -> StatorResult<()> {
        // acc = value, need to: save value, eval obj, store.
        let val_reg = self.allocator.allocate_temporary();
        self.emit_star(val_reg);

        self.compile_expr(&m.object)?;
        let obj_reg = self.allocator.allocate_temporary();
        self.emit_star(obj_reg);

        // Reload value into accumulator.
        self.emit_ldar(val_reg);

        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                ));
            }
            crate::parser::ast::MemberProp::Computed(key) => {
                // Compile key expression first, then allocate register.
                self.compile_expr(key)?;
                let key_reg = self.allocator.allocate_temporary();
                self.emit_star(key_reg);
                // Reload value.
                self.emit_ldar(val_reg);
                let slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaKeyedProperty,
                    vec![to_reg_op(obj_reg), to_reg_op(key_reg), slot],
                ));
                self.allocator
                    .release_temporary(key_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            }
            crate::parser::ast::MemberProp::Private(_) => {
                return Err(StatorError::Internal(
                    "private member store is not yet supported".into(),
                ));
            }
        }
        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(val_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a call expression `callee(args…)`.
    fn compile_call(&mut self, c: &crate::parser::ast::CallExpr) -> StatorResult<()> {
        // Check for method call: `obj.method(args)`.
        if let Expr::Member(m) = c.callee.as_ref() {
            return self.compile_method_call(m, &c.arguments);
        }

        // General call with undefined receiver.
        self.compile_expr(&c.callee)?;
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        let arg_regs = self.compile_arguments(&c.arguments)?;

        self.emit_call_any_receiver(callee_reg, &arg_regs)?;

        // Release args in reverse order.
        for r in arg_regs.into_iter().rev() {
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile an optional call expression `callee?.(args)`.
    fn compile_optional_call(
        &mut self,
        c: &crate::parser::ast::OptionalCallExpr,
    ) -> StatorResult<()> {
        self.compile_expr(&c.callee)?;
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        let null_label = self.new_label();
        let end_label = self.new_label();

        self.emit_ldar(callee_reg);
        self.emit_jump(Opcode::JumpIfUndefinedOrNull, null_label);

        let arg_regs = self.compile_arguments(&c.arguments)?;
        self.emit_call_any_receiver(callee_reg, &arg_regs)?;
        for r in arg_regs.into_iter().rev() {
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.emit_jump(Opcode::Jump, end_label);

        self.bind_label(null_label);
        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        self.bind_label(end_label);

        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a method call `obj.method(args)`.
    fn compile_method_call(
        &mut self,
        m: &crate::parser::ast::MemberExpr,
        args: &[Expr],
    ) -> StatorResult<()> {
        // Load the object (receiver).
        self.compile_expr(&m.object)?;
        let recv_reg = self.allocator.allocate_temporary();
        self.emit_star(recv_reg);

        // Load the method function.
        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![
                        to_reg_op(recv_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            crate::parser::ast::MemberProp::Computed(key) => {
                self.compile_expr(key)?;
                let slot = self.alloc_slot(FeedbackSlotKind::KeyedLoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaKeyedProperty,
                    vec![to_reg_op(recv_reg), slot],
                ));
            }
            crate::parser::ast::MemberProp::Private(_) => {
                return Err(StatorError::Internal(
                    "private method calls are not yet supported".into(),
                ));
            }
        }
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        // Evaluate arguments into consecutive registers.
        let arg_regs = self.compile_arguments(args)?;

        // CallProperty: [callable, receiver, args_start_or_count, slot]
        // For arbitrary arg count use CallAnyReceiver with receiver as first arg.
        // Emit: CallProperty [callable, recv_reg, arg_count, slot]
        // args_start points to recv_reg; args follow immediately after.
        let arg_count = arg_regs.len() as u32;
        let slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::CallProperty,
            vec![
                to_reg_op(callee_reg),
                to_reg_op(recv_reg),
                Operand::RegisterCount(arg_count),
                slot,
            ],
        ));

        for r in arg_regs.into_iter().rev() {
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(recv_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Evaluate each argument into a freshly allocated temporary register and
    /// return the list of registers in order.
    fn compile_arguments(&mut self, args: &[Expr]) -> StatorResult<Vec<Register>> {
        let mut regs = Vec::with_capacity(args.len());
        for arg in args {
            let r = self.allocator.allocate_temporary();
            self.compile_expr(arg)?;
            self.emit_star(r);
            regs.push(r);
        }
        Ok(regs)
    }

    /// Emit a `CallAnyReceiver` instruction.
    fn emit_call_any_receiver(
        &mut self,
        callee_reg: Register,
        arg_regs: &[Register],
    ) -> StatorResult<()> {
        let arg_count = arg_regs.len() as u32;
        let args_start = arg_regs.first().copied().unwrap_or(callee_reg);
        let slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::CallAnyReceiver,
            vec![
                to_reg_op(callee_reg),
                to_reg_op(args_start),
                Operand::RegisterCount(arg_count),
                slot,
            ],
        ));
        Ok(())
    }

    /// Compile a `new callee(args)` expression.
    fn compile_new(&mut self, n: &crate::parser::ast::NewExpr) -> StatorResult<()> {
        self.compile_expr(&n.callee)?;
        let ctor_reg = self.allocator.allocate_temporary();
        self.emit_star(ctor_reg);

        let arg_regs = self.compile_arguments(&n.arguments)?;
        let arg_count = arg_regs.len() as u32;
        let args_start = arg_regs.first().copied().unwrap_or(ctor_reg);

        let slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::Construct,
            vec![
                to_reg_op(ctor_reg),
                to_reg_op(args_start),
                Operand::RegisterCount(arg_count),
                slot,
            ],
        ));

        for r in arg_regs.into_iter().rev() {
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.allocator
            .release_temporary(ctor_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a function expression.
    fn compile_fn_expr(&mut self, f: &FnExpr) -> StatorResult<()> {
        if f.is_generator || f.is_async {
            return Err(StatorError::Internal(
                "generator and async functions are not yet supported".into(),
            ));
        }
        let func_array = compile_function(&f.params, &f.body)?;
        let pool_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(func_array)));
        let slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![Operand::ConstantPoolIdx(pool_idx), slot, Operand::Flag(0)],
        ));
        Ok(())
    }

    /// Compile an arrow function expression.
    fn compile_arrow_expr(&mut self, a: &ArrowExpr) -> StatorResult<()> {
        if a.is_async {
            return Err(StatorError::Internal(
                "async arrow functions are not yet supported".into(),
            ));
        }
        // Build a synthetic block body if the arrow uses a concise expression.
        let body_block = match &a.body {
            ArrowBody::Block(b) => b.clone(),
            ArrowBody::Expr(expr) => {
                // Wrap `expr` in `return expr;`
                use crate::parser::ast::{BlockStmt, ReturnStmt};
                let loc = expr.loc();
                BlockStmt {
                    loc,
                    body: vec![Stmt::Return(ReturnStmt {
                        loc,
                        argument: Some(expr.clone()),
                    })],
                }
            }
        };
        let func_array = compile_function(&a.params, &body_block)?;
        let pool_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(func_array)));
        let slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![Operand::ConstantPoolIdx(pool_idx), slot, Operand::Flag(0)],
        ));
        Ok(())
    }

    /// Compile an array literal.
    fn compile_array(&mut self, a: &crate::parser::ast::ArrayExpr) -> StatorResult<()> {
        // Create an empty array, then fill each element slot.
        let arr_slot = self.alloc_slot(FeedbackSlotKind::Literal);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateEmptyArrayLiteral,
            vec![arr_slot],
        ));
        let arr_reg = self.allocator.allocate_temporary();
        self.emit_star(arr_reg);

        for (i, elem) in a.elements.iter().enumerate() {
            if let Some(elem_expr) = elem {
                // Allocate idx_reg only when an element is present.
                let idx_reg = self.allocator.allocate_temporary();
                // Load index.
                let idx_val = i as i32;
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaSmi,
                    vec![Operand::Immediate(idx_val)],
                ));
                self.emit_star(idx_reg);
                // Load element value.
                self.compile_expr(elem_expr)?;
                // StaInArrayLiteral [array_reg, index_reg, slot]
                let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![to_reg_op(arr_reg), to_reg_op(idx_reg), elem_slot],
                ));
                self.allocator
                    .release_temporary(idx_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            }
        }

        // Reload the array into accumulator.
        self.emit_ldar(arr_reg);
        self.allocator
            .release_temporary(arr_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile an object literal.
    fn compile_object(&mut self, o: &crate::parser::ast::ObjectExpr) -> StatorResult<()> {
        self.emit(Instruction::new_unchecked(
            Opcode::CreateEmptyObjectLiteral,
            vec![],
        ));
        let obj_reg = self.allocator.allocate_temporary();
        self.emit_star(obj_reg);

        for prop in &o.properties {
            match prop {
                crate::parser::ast::ObjectProp::Prop(p) => {
                    self.compile_object_prop(obj_reg, p)?;
                }
                crate::parser::ast::ObjectProp::Spread(_) => {
                    return Err(StatorError::Internal(
                        "object spread is not yet supported".into(),
                    ));
                }
            }
        }

        self.emit_ldar(obj_reg);
        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a single property definition into an object literal.
    fn compile_object_prop(
        &mut self,
        obj_reg: Register,
        p: &crate::parser::ast::Prop,
    ) -> StatorResult<()> {
        use crate::parser::ast::{PropKey, PropValue};

        // Get the key string (for static keys).
        let key_name: Option<String> = match &p.key {
            PropKey::Ident(id) => Some(id.name.clone()),
            PropKey::Str(s) => Some(s.value.clone()),
            PropKey::Num(n) => {
                // Use JS-compatible number-to-string for integer keys.
                let v = n.value;
                if v.fract() == 0.0 && v.is_finite() && v >= 0.0 {
                    Some(format!("{}", v as u64))
                } else {
                    Some(n.raw.clone())
                }
            }
            PropKey::Private(id) => Some(format!("#{}", id.name)),
            PropKey::Computed(_) => None,
        };

        match &p.value {
            PropValue::Value(expr) => {
                self.compile_expr(expr)?;
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                } else if let PropKey::Computed(key_expr) = &p.key {
                    let val_reg = self.allocator.allocate_temporary();
                    self.emit_star(val_reg);
                    self.compile_expr(key_expr)?;
                    let key_reg = self.allocator.allocate_temporary();
                    self.emit_star(key_reg);
                    self.emit_ldar(val_reg);
                    let slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedOwnProperty,
                        vec![
                            to_reg_op(obj_reg),
                            to_reg_op(key_reg),
                            Operand::Flag(0),
                            slot,
                        ],
                    ));
                    self.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    self.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            }
            PropValue::Shorthand => {
                // `{ x }` → equivalent to `{ x: x }`
                if let PropKey::Ident(id) = &p.key {
                    self.compile_ident_load(&id.name);
                    let name_idx = self.add_string(&id.name);
                    let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                }
            }
            PropValue::Method(fn_expr) => {
                self.compile_fn_expr(fn_expr)?;
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                }
            }
            PropValue::Get(fn_expr) | PropValue::Set(fn_expr) => {
                self.compile_fn_expr(fn_expr)?;
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                }
            }
        }
        Ok(())
    }

    /// Compile a template literal (basic version — no interpolation).
    fn compile_template(&mut self, t: &crate::parser::ast::TemplateLit) -> StatorResult<()> {
        if t.expressions.is_empty() {
            // No interpolation: emit a single string constant.
            let s = t
                .quasis
                .iter()
                .filter_map(|q| q.cooked.as_deref())
                .collect::<String>();
            let idx = self.add_string(&s);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaConstant,
                vec![Operand::ConstantPoolIdx(idx)],
            ));
            return Ok(());
        }
        // With interpolation: concatenate quasis and expressions via string adds.
        // Start with the first quasi.
        let first = t.quasis[0].cooked.as_deref().unwrap_or("").to_owned();
        let idx0 = self.add_string(&first);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaConstant,
            vec![Operand::ConstantPoolIdx(idx0)],
        ));
        let acc_reg = self.allocator.allocate_temporary();
        self.emit_star(acc_reg);

        for (i, expr) in t.expressions.iter().enumerate() {
            // Convert expression to string via ToString, then add.
            self.compile_expr(expr)?;
            self.emit(Instruction::new_unchecked(Opcode::ToString, vec![]));
            let expr_reg = self.allocator.allocate_temporary();
            self.emit_star(expr_reg);

            self.emit_ldar(acc_reg);
            let slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
            self.emit(Instruction::new_unchecked(
                Opcode::Add,
                vec![to_reg_op(expr_reg), slot],
            ));
            self.emit_star(acc_reg);

            self.allocator
                .release_temporary(expr_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;

            // Append the next quasi.
            let quasi = t.quasis[i + 1].cooked.as_deref().unwrap_or("").to_owned();
            let qi = self.add_string(&quasi);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaConstant,
                vec![Operand::ConstantPoolIdx(qi)],
            ));
            let q_reg = self.allocator.allocate_temporary();
            self.emit_star(q_reg);

            self.emit_ldar(acc_reg);
            let slot2 = self.alloc_slot(FeedbackSlotKind::BinaryOp);
            self.emit(Instruction::new_unchecked(
                Opcode::Add,
                vec![to_reg_op(q_reg), slot2],
            ));
            self.emit_star(acc_reg);

            self.allocator
                .release_temporary(q_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }

        self.emit_ldar(acc_reg);
        self.allocator
            .release_temporary(acc_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    // ── Finalization ─────────────────────────────────────────────────────────

    /// Resolve all jump offsets and encode the instruction stream into a
    /// [`BytecodeArray`].
    fn finalize(mut self) -> StatorResult<BytecodeArray> {
        // Ensure every function ends with an implicit `return undefined`.
        let needs_implicit_return = self
            .instructions
            .last()
            .map(|i| i.opcode != Opcode::Return)
            .unwrap_or(true);
        if needs_implicit_return {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            self.emit(Instruction::new_unchecked(Opcode::Return, vec![]));
        }

        // Resolve jump targets.
        resolve_jumps(&mut self.instructions, &self.labels)?;

        let frame_size = self.allocator.frame_size();
        let bytes = encode(&self.instructions);
        let feedback_metadata = FeedbackMetadata::new(self.slot_kinds);
        Ok(BytecodeArray::new(
            bytes,
            self.constant_pool,
            frame_size,
            self.param_count,
            self.source_positions,
            feedback_metadata,
            self.handler_table,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Jump resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve all jump operands from instruction-index–based label targets to
/// byte-offset deltas.
///
/// Iterates until the instruction sizes stabilise (handles cases where
/// changing an offset value changes the encoding width of the jump
/// instruction, which in turn changes other offsets).  Convergence is
/// guaranteed in a small, bounded number of passes for realistic functions.
fn resolve_jumps(instructions: &mut [Instruction], labels: &[Label]) -> StatorResult<()> {
    const MAX_ITERS: usize = 20;
    for _ in 0..MAX_ITERS {
        let offsets = compute_byte_offsets(instructions);
        let mut changed = false;

        for label in labels {
            // Skip labels that were allocated but never targeted by a jump.
            if label.refs.is_empty() {
                continue;
            }
            let target_instr = label.bound_at.ok_or_else(|| {
                StatorError::Internal("unbound label during jump resolution".into())
            })?;
            let target_byte = offsets[target_instr];

            for &jump_instr_idx in &label.refs {
                let instr_end_byte = offsets[jump_instr_idx + 1];
                let delta = target_byte as i64 - instr_end_byte as i64;
                let new_op = Operand::JumpOffset(delta as i32);
                if instructions[jump_instr_idx].operands[0] != new_op {
                    instructions[jump_instr_idx].operands[0] = new_op;
                    changed = true;
                }
            }
        }

        if !changed {
            return Ok(());
        }
    }
    Err(StatorError::Internal(
        "jump resolution failed to converge".into(),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a function body with the given parameter list into a
/// [`BytecodeArray`].
fn compile_function(
    params: &[crate::parser::ast::Param],
    body: &BlockStmt,
) -> StatorResult<BytecodeArray> {
    let mut compiler = FunctionCompiler::new(params)?;
    // Hoist function declarations to the top of the scope.
    for stmt in &body.body {
        if let Stmt::FnDecl(decl) = stmt {
            compiler.compile_fn_decl(decl)?;
        }
    }
    // Compile remaining statements.
    for stmt in &body.body {
        if !matches!(stmt, Stmt::FnDecl(_)) {
            compiler.compile_stmt(stmt)?;
        }
    }
    compiler.finalize()
}

/// Return a short description string for unsupported statement types.
fn stmt_kind(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::ForIn(_) => "for-in",
        Stmt::ForOf(_) => "for-of",
        Stmt::With(_) => "with",
        Stmt::ClassDecl(_) => "class declaration",
        _ => "statement",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Bytecode compiler that walks a JavaScript AST and produces a
/// [`BytecodeArray`] for the top-level script.
///
/// # Example
///
/// ```
/// use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
/// use stator_core::parser::ast::{Program, SourceType};
/// use stator_core::parser::scanner::{Position, Span};
///
/// let p = Position { offset: 0, line: 1, column: 1 };
/// let loc = Span { start: p, end: p };
/// let program = Program {
///     loc,
///     source_type: SourceType::Script,
///     body: vec![],
/// };
/// let array = BytecodeGenerator::compile_program(&program)
///     .expect("compilation should succeed");
/// assert_eq!(array.parameter_count(), 0);
/// ```
pub struct BytecodeGenerator;

impl BytecodeGenerator {
    /// Compile a top-level [`Program`] into a [`BytecodeArray`].
    ///
    /// The returned [`BytecodeArray`] represents the implicit top-level
    /// function that wraps all the program's statements.
    pub fn compile_program(program: &Program) -> StatorResult<BytecodeArray> {
        let mut compiler = FunctionCompiler::new(&[])?;

        // Hoist function declarations to the top.
        for item in &program.body {
            if let ProgramItem::Stmt(Stmt::FnDecl(decl)) = item {
                compiler.compile_fn_decl(decl)?;
            }
        }

        for item in &program.body {
            match item {
                ProgramItem::Stmt(Stmt::FnDecl(_)) => {} // already hoisted
                ProgramItem::Stmt(stmt) => compiler.compile_stmt(stmt)?,
                ProgramItem::ModuleDecl(_) => {
                    return Err(StatorError::Internal(
                        "module declarations are not yet supported".into(),
                    ));
                }
            }
        }
        compiler.finalize()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecodes::{Opcode, decode};
    use crate::parser::ast::{
        BinaryExpr, BinaryOp, BlockStmt, BoolLit, BreakStmt, CatchClause, ContinueStmt,
        DoWhileStmt, Expr, ExprStmt, FnDecl, ForStmt, Ident, IfStmt, NullLit, NumLit, Param, Pat,
        Program, ProgramItem, ReturnStmt, SourceType, Stmt, StringLit, ThrowStmt, TryStmt, VarDecl,
        VarDeclarator, VarKind, WhileStmt,
    };
    use crate::parser::scanner::{Position, Span};

    fn span() -> Span {
        let p = Position {
            offset: 0,
            line: 1,
            column: 1,
        };
        Span { start: p, end: p }
    }

    fn ident(name: &str) -> Ident {
        Ident {
            loc: span(),
            name: name.to_owned(),
        }
    }

    fn num_expr(v: f64) -> Expr {
        Expr::Num(NumLit {
            loc: span(),
            value: v,
            raw: v.to_string(),
        })
    }

    fn bool_expr(v: bool) -> Expr {
        Expr::Bool(BoolLit {
            loc: span(),
            value: v,
        })
    }

    fn str_expr(s: &str) -> Expr {
        Expr::Str(StringLit {
            loc: span(),
            value: s.to_owned(),
        })
    }

    fn null_expr() -> Expr {
        Expr::Null(NullLit { loc: span() })
    }

    fn ident_expr(name: &str) -> Expr {
        Expr::Ident(ident(name))
    }

    fn var_decl_stmt(kind: VarKind, name: &str, init: Option<Expr>) -> Stmt {
        Stmt::VarDecl(VarDecl {
            loc: span(),
            kind,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(ident(name)),
                init: init.map(Box::new),
            }],
        })
    }

    fn return_stmt(arg: Option<Expr>) -> Stmt {
        Stmt::Return(ReturnStmt {
            loc: span(),
            argument: arg.map(Box::new),
        })
    }

    fn make_program(stmts: Vec<Stmt>) -> Program {
        Program {
            loc: span(),
            source_type: SourceType::Script,
            body: stmts.into_iter().map(ProgramItem::Stmt).collect(),
        }
    }

    // ── Phase 1: literals + arithmetic + return ───────────────────────────

    #[test]
    fn test_empty_program() {
        let prog = make_program(vec![]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs.last().unwrap().opcode, Opcode::Return);
    }

    #[test]
    fn test_return_null() {
        let prog = make_program(vec![return_stmt(Some(null_expr()))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaNull);
        assert_eq!(instrs[1].opcode, Opcode::Return);
    }

    #[test]
    fn test_return_true_false() {
        let prog = make_program(vec![return_stmt(Some(bool_expr(true)))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaTrue);

        let prog2 = make_program(vec![return_stmt(Some(bool_expr(false)))]);
        let arr2 = BytecodeGenerator::compile_program(&prog2).unwrap();
        let instrs2 = arr2.instructions().unwrap();
        assert_eq!(instrs2[0].opcode, Opcode::LdaFalse);
    }

    #[test]
    fn test_return_zero() {
        let prog = make_program(vec![return_stmt(Some(num_expr(0.0)))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaZero);
        assert_eq!(instrs[1].opcode, Opcode::Return);
    }

    #[test]
    fn test_return_smi() {
        let prog = make_program(vec![return_stmt(Some(num_expr(42.0)))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaSmi);
        assert_eq!(instrs[0].operands[0], Operand::Immediate(42));
        assert_eq!(instrs[1].opcode, Opcode::Return);
    }

    #[test]
    fn test_return_float() {
        let prog = make_program(vec![return_stmt(Some(num_expr(3.14)))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaConstant);
        assert_eq!(arr.get_constant(0), Some(&ConstantPoolEntry::Number(3.14)));
    }

    #[test]
    fn test_return_string() {
        let prog = make_program(vec![return_stmt(Some(str_expr("hello")))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaConstant);
        assert_eq!(
            arr.get_constant(0),
            Some(&ConstantPoolEntry::String("hello".to_owned()))
        );
    }

    #[test]
    fn test_return_without_value() {
        let prog = make_program(vec![return_stmt(None)]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert_eq!(instrs[0].opcode, Opcode::LdaUndefined);
        assert_eq!(instrs[1].opcode, Opcode::Return);
    }

    #[test]
    fn test_arithmetic_add() {
        // return 1 + 2
        let expr = Expr::Binary(Box::new(BinaryExpr {
            loc: span(),
            op: BinaryOp::Add,
            left: Box::new(num_expr(1.0)),
            right: Box::new(num_expr(2.0)),
        }));
        let prog = make_program(vec![return_stmt(Some(expr))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Expect: LdaSmi(2), Star(r0), LdaSmi(1), Add(r0), Return
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::Add || i.opcode == Opcode::AddSmi)
        );
        assert_eq!(instrs.last().unwrap().opcode, Opcode::Return);
    }

    #[test]
    fn test_comparison_lt() {
        // return 1 < 2
        let expr = Expr::Binary(Box::new(BinaryExpr {
            loc: span(),
            op: BinaryOp::Lt,
            left: Box::new(num_expr(1.0)),
            right: Box::new(num_expr(2.0)),
        }));
        let prog = make_program(vec![return_stmt(Some(expr))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::TestLessThan));
    }

    // ── Phase 2: variables + control flow ─────────────────────────────────

    #[test]
    fn test_var_decl_and_return() {
        // let x = 5; return x;
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(5.0))),
            return_stmt(Some(ident_expr("x"))),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Star));
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Ldar));
        assert_eq!(instrs.last().unwrap().opcode, Opcode::Return);
    }

    #[test]
    fn test_frame_size() {
        // let a = 1; let b = 2; let c = 3;
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "a", Some(num_expr(1.0))),
            var_decl_stmt(VarKind::Let, "b", Some(num_expr(2.0))),
            var_decl_stmt(VarKind::Let, "c", Some(num_expr(3.0))),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        assert!(arr.frame_size() >= 3);
    }

    #[test]
    fn test_if_statement() {
        // if (true) { return 1; } else { return 2; }
        let prog = make_program(vec![Stmt::If(IfStmt {
            loc: span(),
            test: Box::new(bool_expr(true)),
            consequent: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(num_expr(1.0)))],
            })),
            alternate: Some(Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(num_expr(2.0)))],
            }))),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::JumpIfToBooleanFalse)
        );
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Jump));
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Return));
    }

    #[test]
    fn test_while_loop() {
        // let i = 0; while (i < 10) { i = i + 1; }
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget};
        let init = var_decl_stmt(VarKind::Let, "i", Some(num_expr(0.0)));
        let test_expr = Expr::Binary(Box::new(BinaryExpr {
            loc: span(),
            op: BinaryOp::Lt,
            left: Box::new(ident_expr("i")),
            right: Box::new(num_expr(10.0)),
        }));
        let incr = Expr::Assign(Box::new(AssignExpr {
            loc: span(),
            op: AssignOp::Assign,
            left: AssignTarget::Expr(Box::new(ident_expr("i"))),
            right: Box::new(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("i")),
                right: Box::new(num_expr(1.0)),
            }))),
        }));
        let prog = make_program(vec![
            init,
            Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(test_expr),
                body: Box::new(Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(incr),
                })),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Must contain a backwards jump (negative offset).
        let has_back_jump = instrs.iter().any(|i| {
            i.opcode == Opcode::Jump
                && i.operands
                    .first()
                    .map(|o| matches!(o, Operand::JumpOffset(v) if *v < 0))
                    .unwrap_or(false)
        });
        assert!(has_back_jump, "while loop must have a back-edge jump");
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must be valid");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_for_loop_break_continue() {
        // for (let i = 0; i < 3; i = i + 1) { if (i == 1) continue; if (i == 2) break; }
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget};
        let prog = make_program(vec![Stmt::For(ForStmt {
            loc: span(),
            init: Some(ForInit::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Let,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(ident("i")),
                    init: Some(Box::new(num_expr(0.0))),
                }],
            })),
            test: Some(Box::new(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Lt,
                left: Box::new(ident_expr("i")),
                right: Box::new(num_expr(3.0)),
            })))),
            update: Some(Box::new(Expr::Assign(Box::new(AssignExpr {
                loc: span(),
                op: AssignOp::Assign,
                left: AssignTarget::Expr(Box::new(ident_expr("i"))),
                right: Box::new(Expr::Binary(Box::new(BinaryExpr {
                    loc: span(),
                    op: BinaryOp::Add,
                    left: Box::new(ident_expr("i")),
                    right: Box::new(num_expr(1.0)),
                }))),
            })))),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![
                    Stmt::If(IfStmt {
                        loc: span(),
                        test: Box::new(Expr::Binary(Box::new(BinaryExpr {
                            loc: span(),
                            op: BinaryOp::Eq,
                            left: Box::new(ident_expr("i")),
                            right: Box::new(num_expr(1.0)),
                        }))),
                        consequent: Box::new(Stmt::Continue(ContinueStmt {
                            loc: span(),
                            label: None,
                        })),
                        alternate: None,
                    }),
                    Stmt::If(IfStmt {
                        loc: span(),
                        test: Box::new(Expr::Binary(Box::new(BinaryExpr {
                            loc: span(),
                            op: BinaryOp::Eq,
                            left: Box::new(ident_expr("i")),
                            right: Box::new(num_expr(2.0)),
                        }))),
                        consequent: Box::new(Stmt::Break(BreakStmt {
                            loc: span(),
                            label: None,
                        })),
                        alternate: None,
                    }),
                ],
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must decode");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_do_while_loop() {
        // let i = 0; do { i = i + 1; } while (i < 3);
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget};
        let incr = Expr::Assign(Box::new(AssignExpr {
            loc: span(),
            op: AssignOp::Assign,
            left: AssignTarget::Expr(Box::new(ident_expr("i"))),
            right: Box::new(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("i")),
                right: Box::new(num_expr(1.0)),
            }))),
        }));
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "i", Some(num_expr(0.0))),
            Stmt::DoWhile(DoWhileStmt {
                loc: span(),
                body: Box::new(Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(incr),
                })),
                test: Box::new(Expr::Binary(Box::new(BinaryExpr {
                    loc: span(),
                    op: BinaryOp::Lt,
                    left: Box::new(ident_expr("i")),
                    right: Box::new(num_expr(3.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::JumpIfToBooleanTrue)
        );
    }

    // ── Phase 3: functions + closures + calls ─────────────────────────────

    #[test]
    fn test_fn_decl_creates_closure() {
        // function add(a, b) { return a + b; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("a")),
                right: Box::new(ident_expr("b")),
            }))))],
        };
        let prog = make_program(vec![Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(ident("add")),
            is_async: false,
            is_generator: false,
            params: vec![
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("a")),
                    default: None,
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("b")),
                    default: None,
                },
            ],
            body,
        }))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::CreateClosure));
        // The nested function's BytecodeArray should be in the constant pool.
        assert!(matches!(
            arr.get_constant(0),
            Some(ConstantPoolEntry::Function(_))
        ));
    }

    #[test]
    fn test_nested_function_compiles() {
        // function add(a, b) { return a + b; }
        // Verify the inner function's bytecode is correct.
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("a")),
                right: Box::new(ident_expr("b")),
            }))))],
        };
        let inner = compile_function(
            &[
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("a")),
                    default: None,
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("b")),
                    default: None,
                },
            ],
            &body,
        )
        .unwrap();
        assert_eq!(inner.parameter_count(), 2);
        let instrs = inner.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Add));
        assert_eq!(instrs.last().unwrap().opcode, Opcode::Return);
    }

    #[test]
    fn test_call_expr() {
        // f(1, 2)
        use crate::parser::ast::CallExpr;
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Call(Box::new(CallExpr {
                loc: span(),
                callee: Box::new(ident_expr("f")),
                arguments: vec![num_expr(1.0), num_expr(2.0)],
            }))),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::CallAnyReceiver));
    }

    #[test]
    fn test_throw_statement() {
        // throw new Error("oops");
        use crate::parser::ast::{NewExpr, StringLit};
        let prog = make_program(vec![Stmt::Throw(ThrowStmt {
            loc: span(),
            argument: Box::new(Expr::New(Box::new(NewExpr {
                loc: span(),
                callee: Box::new(ident_expr("Error")),
                arguments: vec![Expr::Str(StringLit {
                    loc: span(),
                    value: "oops".to_owned(),
                })],
            }))),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Construct));
        assert!(instrs.iter().any(|i| i.opcode == Opcode::Throw));
    }

    #[test]
    fn test_try_catch() {
        // try { let x = 1; } catch (e) { }
        let prog = make_program(vec![Stmt::Try(TryStmt {
            loc: span(),
            block: BlockStmt {
                loc: span(),
                body: vec![var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0)))],
            },
            handler: Some(CatchClause {
                loc: span(),
                param: Some(Pat::Ident(ident("e"))),
                body: BlockStmt {
                    loc: span(),
                    body: vec![],
                },
            }),
            finalizer: None,
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(!instrs.is_empty());
        let bytes = arr.bytecodes();
        decode(bytes).expect("try/catch bytecode must decode");
    }

    #[test]
    fn test_bytecodes_round_trip() {
        // Compile, encode, decode — verify no corruption.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(42.0))),
            var_decl_stmt(VarKind::Let, "y", Some(num_expr(58.0))),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("y")),
            })))),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("must decode");
        assert!(!decoded.is_empty());
        assert_eq!(decoded.last().unwrap().opcode, Opcode::Return);
    }

    // ── Feedback slot allocation tests ────────────────────────────────────
    //
    // These tests verify that the compiler emits the correct FeedbackSlotKind
    // for each IC-bearing instruction category, and that the slot indices
    // embedded in the bytecode operands match the metadata.

    /// Extract a helper that filters slot kinds from compiled metadata.
    fn slot_kinds_for(prog: &Program) -> Vec<crate::bytecode::feedback::FeedbackSlotKind> {
        let arr = BytecodeGenerator::compile_program(prog).unwrap();
        arr.feedback_metadata().slot_kinds().to_vec()
    }

    #[test]
    fn test_feedback_slots_binary_add() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `return x + y` where x, y are locals → one BinaryOp slot.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0))),
            var_decl_stmt(VarKind::Let, "y", Some(num_expr(2.0))),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("y")),
            })))),
        ]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::BinaryOp),
            "expected BinaryOp slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_comparison_lt() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `x < y` → one Compare slot.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0))),
            var_decl_stmt(VarKind::Let, "y", Some(num_expr(2.0))),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Lt,
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("y")),
            })))),
        ]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::Compare),
            "expected Compare slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_global_load() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // Reference to undeclared `console` → LoadGlobal slot.
        let prog = make_program(vec![return_stmt(Some(ident_expr("console")))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::LoadGlobal),
            "expected LoadGlobal slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_global_store() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget};
        // `x = 1` where x is undeclared → StoreGlobal slot.
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                loc: span(),
                op: AssignOp::Assign,
                left: AssignTarget::Expr(Box::new(ident_expr("x"))),
                right: Box::new(num_expr(1.0)),
            }))),
        })]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::StoreGlobal),
            "expected StoreGlobal slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_function_call() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::CallExpr;
        // `f(1, 2)` → Call slot (+ LoadGlobal for `f`).
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Call(Box::new(CallExpr {
                loc: span(),
                callee: Box::new(ident_expr("f")),
                arguments: vec![num_expr(1.0), num_expr(2.0)],
            }))),
        })]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::Call),
            "expected Call slot, got {kinds:?}"
        );
        assert!(
            kinds.contains(&FeedbackSlotKind::LoadGlobal),
            "expected LoadGlobal slot for callee, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_named_property_load() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::MemberExpr;
        // `obj.name` → LoadProperty slot.
        let prog = make_program(vec![return_stmt(Some(Expr::Member(Box::new(
            MemberExpr {
                loc: span(),
                object: Box::new(ident_expr("obj")),
                property: crate::parser::ast::MemberProp::Ident(ident("name")),
                is_computed: false,
            },
        ))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::LoadProperty),
            "expected LoadProperty slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_named_property_store() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget, MemberExpr};
        // `obj.x = 1` → StoreProperty slot.
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                loc: span(),
                op: AssignOp::Assign,
                left: AssignTarget::Expr(Box::new(Expr::Member(Box::new(MemberExpr {
                    loc: span(),
                    object: Box::new(ident_expr("obj")),
                    property: crate::parser::ast::MemberProp::Ident(ident("x")),
                    is_computed: false,
                })))),
                right: Box::new(num_expr(1.0)),
            }))),
        })]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::StoreProperty),
            "expected StoreProperty slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_keyed_property_load() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::MemberExpr;
        // `obj[key]` → KeyedLoadProperty slot.
        let prog = make_program(vec![return_stmt(Some(Expr::Member(Box::new(
            MemberExpr {
                loc: span(),
                object: Box::new(ident_expr("obj")),
                property: crate::parser::ast::MemberProp::Computed(Box::new(ident_expr("key"))),
                is_computed: true,
            },
        ))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::KeyedLoadProperty),
            "expected KeyedLoadProperty slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_keyed_property_store() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget, MemberExpr};
        // `obj[key] = 1` → KeyedStoreProperty slot.
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                loc: span(),
                op: AssignOp::Assign,
                left: AssignTarget::Expr(Box::new(Expr::Member(Box::new(MemberExpr {
                    loc: span(),
                    object: Box::new(ident_expr("obj")),
                    property: crate::parser::ast::MemberProp::Computed(Box::new(ident_expr("key"))),
                    is_computed: true,
                })))),
                right: Box::new(num_expr(1.0)),
            }))),
        })]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::KeyedStoreProperty),
            "expected KeyedStoreProperty slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_fn_decl() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `function add(a, b) { return a + b; }` → CreateClosure slot.
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("a")),
                right: Box::new(ident_expr("b")),
            }))))],
        };
        let prog = make_program(vec![Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(ident("add")),
            is_async: false,
            is_generator: false,
            params: vec![
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("a")),
                    default: None,
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("b")),
                    default: None,
                },
            ],
            body,
        }))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::CreateClosure),
            "expected CreateClosure slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_typeof() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::UnaryExpr;
        // `typeof x` → TypeOf slot.
        let prog = make_program(vec![return_stmt(Some(Expr::Unary(Box::new(UnaryExpr {
            loc: span(),
            op: crate::parser::ast::UnaryOp::Typeof,
            argument: Box::new(ident_expr("x")),
        }))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::TypeOf),
            "expected TypeOf slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_unary_negate() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::UnaryExpr;
        // `-x` → UnaryOp slot.
        let prog = make_program(vec![return_stmt(Some(Expr::Unary(Box::new(UnaryExpr {
            loc: span(),
            op: crate::parser::ast::UnaryOp::Minus,
            argument: Box::new(ident_expr("x")),
        }))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::UnaryOp),
            "expected UnaryOp slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_increment() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::UpdateExpr;
        // `i++` → BinaryOpInc slot.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "i", Some(num_expr(0.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Update(Box::new(UpdateExpr {
                    loc: span(),
                    op: crate::parser::ast::UpdateOp::Increment,
                    prefix: false,
                    argument: Box::new(ident_expr("i")),
                }))),
            }),
        ]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::BinaryOpInc),
            "expected BinaryOpInc slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_decrement() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::UpdateExpr;
        // `--i` → BinaryOpInc slot (same kind as increment).
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "i", Some(num_expr(5.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Update(Box::new(UpdateExpr {
                    loc: span(),
                    op: crate::parser::ast::UpdateOp::Decrement,
                    prefix: true,
                    argument: Box::new(ident_expr("i")),
                }))),
            }),
        ]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::BinaryOpInc),
            "expected BinaryOpInc slot for decrement, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_array_literal() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        use crate::parser::ast::ArrayExpr;
        // `[1, 2, 3]` → at least one Literal slot (CreateEmptyArrayLiteral)
        // plus KeyedStoreProperty slots for StaInArrayLiteral.
        let prog = make_program(vec![return_stmt(Some(Expr::Array(Box::new(ArrayExpr {
            loc: span(),
            elements: vec![
                Some(num_expr(1.0)),
                Some(num_expr(2.0)),
                Some(num_expr(3.0)),
            ],
        }))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::Literal),
            "expected Literal slot for array creation, got {kinds:?}"
        );
        assert!(
            kinds.contains(&FeedbackSlotKind::KeyedStoreProperty),
            "expected KeyedStoreProperty slot for StaInArrayLiteral, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slots_strict_not_equal() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `a !== b` → one Compare slot (TestEqualStrict + LogicalNot).
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "a", Some(num_expr(1.0))),
            var_decl_stmt(VarKind::Let, "b", Some(num_expr(2.0))),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::StrictNotEq,
                left: Box::new(ident_expr("a")),
                right: Box::new(ident_expr("b")),
            })))),
        ]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::Compare),
            "expected Compare slot for !==, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_slot_indices_embedded_in_bytecode() {
        // Verify that the FeedbackSlot operands in the encoded bytecode are
        // consecutive indices 0, 1, 2, … and match the metadata slot count.
        use crate::bytecode::bytecodes::Operand;
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0))),
            var_decl_stmt(VarKind::Let, "y", Some(num_expr(2.0))),
            // Two binary ops → two distinct FeedbackSlot operands.
            var_decl_stmt(
                VarKind::Let,
                "a",
                Some(Expr::Binary(Box::new(BinaryExpr {
                    loc: span(),
                    op: BinaryOp::Add,
                    left: Box::new(ident_expr("x")),
                    right: Box::new(ident_expr("y")),
                }))),
            ),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Mul,
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("y")),
            })))),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instructions = arr.instructions().expect("valid bytecode");

        // Collect all FeedbackSlot operand values used in the bytecode.
        let mut slot_indices: Vec<u32> = instructions
            .iter()
            .flat_map(|instr| &instr.operands)
            .filter_map(|op| {
                if let Operand::FeedbackSlot(idx) = op {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();
        slot_indices.sort_unstable();
        slot_indices.dedup();

        let slot_count = arr.feedback_metadata().slot_count();
        // Every embedded slot index must be within [0, slot_count).
        for &idx in &slot_indices {
            assert!(
                idx < slot_count,
                "slot index {idx} >= metadata slot_count {slot_count}"
            );
        }
    }

    #[test]
    fn test_feedback_metadata_no_slots_for_constant_only_program() {
        // A program that only loads constants and returns has no IC sites,
        // so the feedback metadata should be empty.
        let prog = make_program(vec![return_stmt(Some(num_expr(42.0)))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        assert_eq!(arr.feedback_metadata().slot_count(), 0);
    }

    #[test]
    fn test_feedback_metadata_instanceof() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `x instanceof Array` → InstanceOf slot.
        let prog = make_program(vec![return_stmt(Some(Expr::Binary(Box::new(
            BinaryExpr {
                loc: span(),
                op: BinaryOp::Instanceof,
                left: Box::new(ident_expr("x")),
                right: Box::new(ident_expr("Array")),
            },
        ))))]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::InstanceOf),
            "expected InstanceOf slot, got {kinds:?}"
        );
    }

    #[test]
    fn test_feedback_vector_from_compiled_array() {
        // End-to-end: compile a program, build a FeedbackVector from the
        // resulting metadata, and verify states start Uninitialized.
        use crate::bytecode::feedback::{FeedbackVector, InlineCacheState};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0))),
            return_stmt(Some(Expr::Binary(Box::new(BinaryExpr {
                loc: span(),
                op: BinaryOp::Add,
                left: Box::new(ident_expr("x")),
                right: Box::new(num_expr(2.0)),
            })))),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let metadata = arr.feedback_metadata();
        let vector = FeedbackVector::new(metadata);
        assert_eq!(vector.slot_count(), metadata.slot_count());
        for i in 0..vector.slot_count() {
            assert_eq!(vector.get_state(i), Some(InlineCacheState::Uninitialized));
        }
    }
}

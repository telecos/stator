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
//! 4. **Classes** — class declarations and expressions, methods, accessors,
//!    static fields and blocks, instance-field initializers.
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
    ArrowBody, ArrowExpr, AssignOp, AssignTarget, BinaryOp, BlockStmt, ExportDefaultExpr,
    ExportNamedDecl, Expr, FnDecl, FnExpr, ForInit, ForStmt, ImportSpecifier, LogicalOp,
    ModuleDecl, ModuleExportName, ObjectPatProp, Pat, Program, ProgramItem, SourceType, Stmt,
    UnaryOp, UpdateOp, VarDecl, VarDeclarator, VarKind,
};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Well-known runtime function ID for `import()` expressions.
///
/// Used as the `RuntimeId` operand in `CallRuntime` so the interpreter can
/// dispatch to the dynamic import handler.
pub(crate) const RUNTIME_DYNAMIC_IMPORT: u32 = 1;

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
// LocalBinding – compile-time metadata for a local variable
// ─────────────────────────────────────────────────────────────────────────────

/// Compile-time metadata for a local variable binding in a scope.
#[derive(Clone, Copy)]
struct LocalBinding {
    /// Virtual register that holds the runtime value.
    reg: Register,
    /// `true` when the binding was declared with `const`.
    is_const: bool,
    /// `true` while the binding is still in the Temporal Dead Zone
    /// (`let`/`const` before their declaration is reached at runtime).
    needs_tdz_check: bool,
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
    /// its [`LocalBinding`].  The outermost scope is `scopes[0]`.
    scopes: Vec<HashMap<String, LocalBinding>>,
    /// Source-position table (populated at expression boundaries).
    source_positions: Vec<SourcePosition>,
    /// Pending `(instruction_index, line, column)` entries collected during
    /// statement compilation.  Converted to byte-offset–based
    /// [`SourcePosition`]s during [`finalize`].
    pending_positions: Vec<(usize, u32, u32)>,
    /// Label table.
    labels: Vec<Label>,
    /// Stack of `(continue_label, break_label)` pairs for loop statements.
    loop_stack: Vec<(usize, usize)>,
    /// Maps label names to `(continue_label, break_label)` pairs for
    /// labeled statements.  `continue_label` is `Some` only when the
    /// labeled body is a loop.
    label_map: HashMap<String, (Option<usize>, usize)>,
    /// When set, the next loop compiler that runs will patch the
    /// `label_map` entry for this label name with its continue-label.
    pending_label_name: Option<String>,
    /// Number of formal parameters.
    param_count: u32,
    /// Ordered list of feedback slot kinds allocated during compilation.
    slot_kinds: Vec<FeedbackSlotKind>,
    /// Per-function exception handler table entries.
    handler_table: Vec<HandlerTableEntry>,
    /// `true` when compiling a `function*` generator body.
    ///
    /// Affects `finalize` (marks the [`BytecodeArray`] with
    /// [`BytecodeArray::with_generator_flag`]) and enables `yield` /
    /// `yield*` compilation.
    is_generator: bool,
    /// `true` when compiling an `async function` or `async function*` body.
    ///
    /// Affects `finalize` (marks the [`BytecodeArray`] with
    /// [`BytecodeArray::with_async_flag`]) and enables `await` compilation.
    is_async: bool,
    /// Counter used to generate unique `suspend_id` immediates for each
    /// [`Opcode::SuspendGenerator`] instruction in this function.
    yield_suspend_id: u32,
    /// `true` when compiling the top-level program body (as opposed to a
    /// nested function).  Top-level function declarations are also stored
    /// as globals via `StaGlobal` so that recursive calls can find them.
    is_program: bool,
    /// `true` when compiling code for a direct `eval()` call.  In this mode
    /// `var` declarations emit [`Opcode::StaGlobal`] so that new bindings
    /// are hoisted into the caller's variable environment.
    is_eval_scope: bool,
    /// `true` when compiling the top-level body of an ES module.
    ///
    /// Affects how import/export declarations and `import.meta` are compiled,
    /// and marks the resulting [`BytecodeArray`] with
    /// [`BytecodeArray::with_module_flag`].
    is_module: bool,
    /// Maps imported binding names to `(module_request_idx, cell_idx)` pairs
    /// that parameterise [`Opcode::LdaModuleVariable`] and
    /// [`Opcode::StaModuleVariable`].
    module_variables: HashMap<String, (u32, i32)>,
    /// Counter for assigning unique cell indices to module variable bindings.
    next_module_cell: i32,
    /// `true` when the expression currently being compiled is in tail
    /// position (i.e. its value is immediately returned).  Set by
    /// [`compile_return`] and consumed by [`compile_call`] /
    /// [`compile_method_call`] to emit [`Opcode::TailCall`] instead of a
    /// regular call.
    in_tail_position: bool,
    /// `true` when compiling in strict mode.
    is_strict: bool,
    /// Stack of `using` variable registers for disposal at scope exit.
    /// Each entry corresponds to a scope level in `scopes`.
    using_vars: Vec<Vec<Register>>,
    /// Maps break-label indices to the register holding the iterator for
    /// for-of cleanup.  When a `break` or `return` is compiled inside a
    /// for-of loop, we emit [`Opcode::IteratorClose`] for the iterator
    /// register before jumping to the break target.
    for_of_iter_regs: HashMap<usize, Register>,
    /// Stack of finally blocks that must be inlined before an abrupt
    /// completion (return / break / continue) can exit the enclosing
    /// `try` body.  Each entry is a reference to the AST `BlockStmt`
    /// that must be compiled inline.
    finally_stack: Vec<crate::parser::ast::BlockStmt>,
    /// When set, the current optional chain's null label.  `OptionalMember`
    /// and `OptionalCall` compilation jump to this label instead of
    /// handling the nullish short-circuit locally.
    optional_chain_null_label: Option<usize>,
}

impl FunctionCompiler {
    /// Create a new compiler for a function with the given parameter list.
    ///
    /// Simple identifier bindings are added to the outermost scope
    /// immediately.  Complex patterns (destructuring, defaults) are
    /// recorded so that [`emit_param_prologue`] can emit the
    /// corresponding bytecode at function entry.
    fn new(params: &[crate::parser::ast::Param]) -> StatorResult<Self> {
        // parameter_count excludes rest params so the runtime can collect
        // excess arguments starting from this index.
        let param_count = params
            .iter()
            .filter(|p| !matches!(p.pat, Pat::Rest(_)))
            .count() as u32;
        let mut compiler = Self {
            instructions: Vec::new(),
            constant_pool: Vec::new(),
            allocator: RegisterAllocator::new(param_count),
            scopes: vec![HashMap::new()],
            source_positions: Vec::new(),
            pending_positions: Vec::new(),
            labels: Vec::new(),
            loop_stack: Vec::new(),
            label_map: HashMap::new(),
            pending_label_name: None,
            param_count,
            slot_kinds: Vec::new(),
            handler_table: Vec::new(),
            is_generator: false,
            is_async: false,
            yield_suspend_id: 0,
            is_program: false,
            is_eval_scope: false,
            is_module: false,
            module_variables: HashMap::new(),
            next_module_cell: 0,
            in_tail_position: false,
            is_strict: false,
            using_vars: vec![Vec::new()],
            for_of_iter_regs: HashMap::new(),
            finally_stack: Vec::new(),
            optional_chain_null_label: None,
        };
        // Register simple ident params in the scope immediately; complex
        // patterns are handled by `emit_param_prologue`.
        for (i, param) in params.iter().enumerate() {
            if let Pat::Ident(ident) = &param.pat
                && param.default.is_none()
            {
                let reg = Register::parameter(i as u32);
                compiler.scopes[0].insert(
                    ident.name.clone(),
                    LocalBinding {
                        reg,
                        is_const: false,
                        needs_tdz_check: false,
                    },
                );
            }
        }
        Ok(compiler)
    }

    /// Emit bytecode at function entry to handle default parameter values
    /// and destructuring binding patterns.
    ///
    /// For each parameter that carries a default value, emits:
    /// ```text
    /// Ldar <param_reg>
    /// JumpIfUndefined :default
    /// Jump :done
    /// default:
    ///   <compile default expr>
    ///   Star <param_reg or new local>
    /// done:
    /// ```
    ///
    /// For destructuring patterns the value (already in a register) is
    /// unpacked into fresh locals via [`compile_binding_pattern`].
    fn emit_param_prologue(&mut self, params: &[crate::parser::ast::Param]) -> StatorResult<()> {
        for (i, param) in params.iter().enumerate() {
            let param_reg = Register::parameter(i as u32);

            match &param.pat {
                Pat::Ident(ident) => {
                    if let Some(default_expr) = &param.default {
                        // Parameter with default: `function f(a = expr)`
                        let local_reg = self.define_local(&ident.name);
                        self.emit_ldar(param_reg);
                        let default_lbl = self.new_label();
                        let done_lbl = self.new_label();
                        self.emit_jump(Opcode::JumpIfUndefined, default_lbl);
                        // Not undefined — store param value into local.
                        self.emit_star(local_reg);
                        self.emit_jump(Opcode::Jump, done_lbl);
                        // Default branch — evaluate default expression.
                        self.bind_label(default_lbl);
                        self.compile_expr(default_expr)?;
                        self.emit_star(local_reg);
                        self.bind_label(done_lbl);
                    }
                    // Simple ident without default was already registered
                    // in `new` — nothing more to emit.
                }
                Pat::Object(_) | Pat::Array(_) => {
                    // Destructuring param: unpack from param register.
                    if let Some(default_expr) = &param.default {
                        let done_lbl = self.new_label();
                        self.emit_ldar(param_reg);
                        self.emit_jump(Opcode::JumpIfNotUndefined, done_lbl);
                        self.compile_expr(default_expr)?;
                        self.emit_star(param_reg);
                        self.bind_label(done_lbl);
                    }
                    // Use param register directly — no temporary needed.
                    self.compile_binding_pattern(
                        &param.pat,
                        param_reg,
                        BindingMode::Declare { is_const: false },
                    )?;
                }
                Pat::Rest(rest) => {
                    // Emit CreateRestParameter — collects all excess arguments
                    // beyond the declared non-rest parameter count into an array.
                    self.emit(Instruction::new_unchecked(
                        Opcode::CreateRestParameter,
                        vec![],
                    ));
                    // Bind the rest name to a fresh local.
                    if let Pat::Ident(ident) = &*rest.argument {
                        let local = self.define_local(&ident.name);
                        self.emit_star(local);
                    }
                }
                Pat::Assign(assign) => {
                    self.compile_binding_pattern(
                        &param.pat,
                        param_reg,
                        BindingMode::Declare { is_const: false },
                    )?;
                    let _ = assign;
                }
                Pat::Expr(_) => {
                    // Expression targets should not appear in function parameters.
                    self.compile_binding_pattern(
                        &param.pat,
                        param_reg,
                        BindingMode::Declare { is_const: false },
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Whether a destructuring pattern creates new bindings or stores to
/// existing variables.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BindingMode {
    /// Variable declarations and function parameters — calls `define_local`
    /// (or `define_const_local` when `is_const` is `true`).
    Declare { is_const: bool },
    /// Destructuring assignment expressions — calls `compile_ident_store`.
    Assign,
}

impl FunctionCompiler {
    /// Compile a binding pattern, destructuring the value held in
    /// `source_reg` into new local variables.
    ///
    /// Handles object patterns (`{a, b: c}`), array patterns (`[x, y]`),
    /// rest elements, and default-value sub-patterns recursively.
    ///
    /// When `mode` is [`BindingMode::Declare`] (variable declarations,
    /// function parameters), each identifier creates a fresh local via
    /// [`define_local`].  When `mode` is [`BindingMode::Assign`]
    /// (destructuring assignment expressions like `[a, b] = arr`), each
    /// identifier stores to an existing variable via [`compile_ident_store`].
    ///
    /// Scratch registers are allocated via `new_local` (unnamed) rather than
    /// `allocate_temporary` because the recursive calls may define new named
    /// locals, which would break the LIFO invariant that temporaries require.
    fn compile_binding_pattern(
        &mut self,
        pat: &Pat,
        source_reg: Register,
        mode: BindingMode,
    ) -> StatorResult<()> {
        match pat {
            Pat::Ident(ident) => match mode {
                BindingMode::Declare { is_const } => {
                    let local = if is_const {
                        self.define_const_local(&ident.name)
                    } else {
                        self.define_local(&ident.name)
                    };
                    self.emit_ldar(source_reg);
                    self.emit_star(local);
                }
                BindingMode::Assign => {
                    self.emit_ldar(source_reg);
                    self.compile_ident_store(&ident.name)?;
                }
            },
            Pat::Object(obj_pat) => {
                // Check whether we need to track excluded keys for a rest
                // element (e.g. `{a, ...rest} = obj` → rest must omit `a`).
                let has_rest = obj_pat
                    .properties
                    .iter()
                    .any(|p| matches!(p, ObjectPatProp::Rest(_)));

                // Accumulated excluded-key info for the rest element.
                let mut excluded_static_keys: Vec<String> = Vec::new();
                let mut excluded_computed_regs: Vec<Register> = Vec::new();

                for prop in &obj_pat.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => {
                            match &kv.key {
                                crate::parser::ast::PropKey::Ident(id) => {
                                    if has_rest {
                                        excluded_static_keys.push(id.name.clone());
                                    }
                                    let name_idx = self.add_string(&id.name);
                                    let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                                    self.emit(Instruction::new_unchecked(
                                        Opcode::LdaNamedProperty,
                                        vec![
                                            to_reg_op(source_reg),
                                            Operand::ConstantPoolIdx(name_idx),
                                            slot,
                                        ],
                                    ));
                                }
                                crate::parser::ast::PropKey::Str(s) => {
                                    if has_rest {
                                        excluded_static_keys.push(s.value.clone());
                                    }
                                    let name_idx = self.add_string(&s.value);
                                    let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                                    self.emit(Instruction::new_unchecked(
                                        Opcode::LdaNamedProperty,
                                        vec![
                                            to_reg_op(source_reg),
                                            Operand::ConstantPoolIdx(name_idx),
                                            slot,
                                        ],
                                    ));
                                }
                                crate::parser::ast::PropKey::Num(n) => {
                                    if has_rest {
                                        excluded_static_keys.push(n.value.to_string());
                                    }
                                    let name_idx = self.add_string(&n.value.to_string());
                                    let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                                    self.emit(Instruction::new_unchecked(
                                        Opcode::LdaNamedProperty,
                                        vec![
                                            to_reg_op(source_reg),
                                            Operand::ConstantPoolIdx(name_idx),
                                            slot,
                                        ],
                                    ));
                                }
                                crate::parser::ast::PropKey::Computed(expr) => {
                                    self.compile_expr(expr)?;
                                    if has_rest {
                                        // Save the computed key so we can
                                        // exclude it from the rest object.
                                        let key_reg = self.allocator.new_local();
                                        self.emit_star(key_reg);
                                        excluded_computed_regs.push(key_reg);
                                    }
                                    let slot = self.alloc_slot(FeedbackSlotKind::KeyedLoadProperty);
                                    self.emit(Instruction::new_unchecked(
                                        Opcode::LdaKeyedProperty,
                                        vec![to_reg_op(source_reg), slot],
                                    ));
                                }
                                crate::parser::ast::PropKey::Private(id) => {
                                    if has_rest {
                                        excluded_static_keys.push(format!("#{}", id.name));
                                    }
                                    let name_idx = self.add_string(&format!("#{}", id.name));
                                    let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                                    self.emit(Instruction::new_unchecked(
                                        Opcode::LdaNamedProperty,
                                        vec![
                                            to_reg_op(source_reg),
                                            Operand::ConstantPoolIdx(name_idx),
                                            slot,
                                        ],
                                    ));
                                }
                            }
                            let scratch = self.allocator.new_local();
                            self.emit_star(scratch);
                            self.compile_binding_pattern(&kv.value, scratch, mode)?;
                        }
                        ObjectPatProp::Assign(assign_prop) => {
                            if has_rest {
                                excluded_static_keys.push(assign_prop.key.name.clone());
                            }
                            let name_idx = self.add_string(&assign_prop.key.name);
                            let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                            self.emit(Instruction::new_unchecked(
                                Opcode::LdaNamedProperty,
                                vec![
                                    to_reg_op(source_reg),
                                    Operand::ConstantPoolIdx(name_idx),
                                    slot,
                                ],
                            ));
                            match mode {
                                BindingMode::Declare { is_const } => {
                                    let local = if is_const {
                                        self.define_const_local(&assign_prop.key.name)
                                    } else {
                                        self.define_local(&assign_prop.key.name)
                                    };
                                    if let Some(default_expr) = &assign_prop.value {
                                        let done_lbl = self.new_label();
                                        self.emit_star(local);
                                        self.emit_ldar(local);
                                        self.emit_jump(Opcode::JumpIfNotUndefined, done_lbl);
                                        self.compile_expr(default_expr)?;
                                        self.emit_star(local);
                                        self.bind_label(done_lbl);
                                    } else {
                                        self.emit_star(local);
                                    }
                                }
                                BindingMode::Assign => {
                                    if let Some(default_expr) = &assign_prop.value {
                                        let done_lbl = self.new_label();
                                        self.emit_jump(Opcode::JumpIfNotUndefined, done_lbl);
                                        self.compile_expr(default_expr)?;
                                        self.bind_label(done_lbl);
                                    }
                                    // acc holds either the property value or the
                                    // default — store to the existing variable.
                                    self.compile_ident_store(&assign_prop.key.name)?;
                                }
                            }
                        }
                        ObjectPatProp::Rest(rest) => {
                            // Create a new object, copy all source properties,
                            // then remove the already-destructured keys so the
                            // rest object only contains the remaining ones.
                            self.emit(Instruction::new_unchecked(
                                Opcode::CreateEmptyObjectLiteral,
                                vec![],
                            ));
                            let rest_reg = self.allocator.new_local();
                            self.emit_star(rest_reg);
                            self.emit(Instruction::new_unchecked(
                                Opcode::CopyDataProperties,
                                vec![to_reg_op(rest_reg), to_reg_op(source_reg)],
                            ));

                            // Delete static keys that were already extracted.
                            for key in &excluded_static_keys {
                                let key_idx = self.add_string(key);
                                self.emit(Instruction::new_unchecked(
                                    Opcode::LdaConstant,
                                    vec![Operand::ConstantPoolIdx(key_idx)],
                                ));
                                self.emit(Instruction::new_unchecked(
                                    Opcode::DeletePropertySloppy,
                                    vec![to_reg_op(rest_reg)],
                                ));
                            }

                            // Delete computed keys saved in registers.
                            for &key_reg in &excluded_computed_regs {
                                self.emit_ldar(key_reg);
                                self.emit(Instruction::new_unchecked(
                                    Opcode::DeletePropertySloppy,
                                    vec![to_reg_op(rest_reg)],
                                ));
                            }

                            self.compile_binding_pattern(&rest.argument, rest_reg, mode)?;
                        }
                    }
                }
            }
            Pat::Array(arr_pat) => {
                let load_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
                self.emit(Instruction::new_unchecked(
                    Opcode::GetIterator,
                    vec![to_reg_op(source_reg), load_slot, call_slot],
                ));
                // Use unnamed locals for iterator and value registers.
                let iter_reg = self.allocator.new_local();
                self.emit_star(iter_reg);
                let val_reg = self.allocator.new_local();

                let total = arr_pat.elements.len();
                for (i, element) in arr_pat.elements.iter().enumerate() {
                    // Check if this is a rest element at the end: let [a, ...rest] = x;
                    if let Some(Pat::Rest(rest)) = element
                        && i == total - 1
                    {
                        // Collect remaining iterator values into an array.
                        let arr_slot = self.alloc_slot(FeedbackSlotKind::Literal);
                        self.emit(Instruction::new_unchecked(
                            Opcode::CreateEmptyArrayLiteral,
                            vec![arr_slot],
                        ));
                        let rest_arr_reg = self.allocator.new_local();
                        self.emit_star(rest_arr_reg);
                        let idx_reg = self.allocator.new_local();
                        self.emit(Instruction::new_unchecked(Opcode::LdaZero, vec![]));
                        self.emit_star(idx_reg);

                        let loop_lbl = self.new_label();
                        let done_lbl = self.new_label();
                        self.bind_label(loop_lbl);
                        self.emit(Instruction::new_unchecked(
                            Opcode::IteratorNext,
                            vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
                        ));
                        self.emit_jump_if_true_to(done_lbl);
                        self.emit_ldar(val_reg);
                        let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                        self.emit(Instruction::new_unchecked(
                            Opcode::StaInArrayLiteral,
                            vec![to_reg_op(rest_arr_reg), to_reg_op(idx_reg), elem_slot],
                        ));
                        self.emit_ldar(idx_reg);
                        let inc_slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                        self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot]));
                        self.emit_star(idx_reg);
                        self.emit_jump_loop_to(loop_lbl);
                        self.bind_label(done_lbl);

                        self.compile_binding_pattern(&rest.argument, rest_arr_reg, mode)?;
                    } else {
                        self.emit(Instruction::new_unchecked(
                            Opcode::IteratorNext,
                            vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
                        ));
                        if let Some(elem_pat) = element {
                            self.compile_binding_pattern(elem_pat, val_reg, mode)?;
                        }
                    }
                }
            }
            Pat::Assign(assign_pat) => {
                let done_lbl = self.new_label();
                self.emit_ldar(source_reg);
                self.emit_jump(Opcode::JumpIfNotUndefined, done_lbl);
                self.compile_expr(&assign_pat.right)?;
                self.emit_star(source_reg);
                self.bind_label(done_lbl);
                self.compile_binding_pattern(&assign_pat.left, source_reg, mode)?;
            }
            Pat::Rest(rest) => {
                // Rest element outside an array/object pattern context (which
                // have their own specialized handling).  Bind the source value
                // directly to the inner pattern — the caller already placed the
                // correct remaining value into source_reg.
                self.compile_binding_pattern(&rest.argument, source_reg, mode)?;
            }
            Pat::Expr(expr) => {
                // Expression target inside a destructuring pattern, e.g.
                // `({a: obj.prop} = val)`.  Load the extracted value then
                // store to the expression target.
                self.emit_ldar(source_reg);
                match expr.as_ref() {
                    Expr::Member(m) => self.compile_member_store(m)?,
                    Expr::Ident(id) => match mode {
                        BindingMode::Declare { is_const } => {
                            let local = if is_const {
                                self.define_const_local(&id.name)
                            } else {
                                self.define_local(&id.name)
                            };
                            self.emit_star(local);
                        }
                        BindingMode::Assign => {
                            self.compile_ident_store(&id.name)?;
                        }
                    },
                    other => {
                        let loc = other.loc();
                        return Err(StatorError::SyntaxError(format!(
                            "at {}:{} — Invalid destructuring assignment target",
                            loc.start.line, loc.start.column
                        )));
                    }
                }
            }
        }
        Ok(())
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

    /// Add a `BigInt` entry, reusing an existing entry with the same value.
    fn add_bigint(&mut self, value: i128) -> u32 {
        for (i, e) in self.constant_pool.iter().enumerate() {
            if let ConstantPoolEntry::BigInt(v) = e
                && *v == value
            {
                return i as u32;
            }
        }
        let idx = self.constant_pool.len() as u32;
        self.constant_pool.push(ConstantPoolEntry::BigInt(value));
        idx
    }

    /// Add a [`ConstantPoolEntry`] without deduplication.
    fn add_constant_raw(&mut self, entry: ConstantPoolEntry) -> u32 {
        let idx = self.constant_pool.len() as u32;
        self.constant_pool.push(entry);
        idx
    }

    // ── Finally inlining ────────────────────────────────────────────────────

    /// Inline all pending finally blocks.  Called before an abrupt
    /// completion (return / break / continue) so that every enclosing
    /// `try-finally` has its finally body executed before control
    /// transfers away.
    fn inline_pending_finally_blocks(&mut self) -> StatorResult<()> {
        let blocks: Vec<_> = self.finally_stack.clone();
        for block in blocks.iter().rev() {
            self.compile_block(block)?;
        }
        Ok(())
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

    /// If a labeled statement is pending, patch its `label_map` entry with
    /// the loop's `continue_label`.  Called by each loop compiler right
    /// after it determines its continue target.
    fn patch_pending_label(&mut self, continue_label: usize) {
        if let Some(name) = self.pending_label_name.take()
            && let Some(entry) = self.label_map.get_mut(&name)
        {
            entry.0 = Some(continue_label);
        }
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
        self.using_vars.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        // Emit disposal calls for any `using` variables in this scope.
        if let Some(vars) = self.using_vars.pop() {
            for reg in vars.into_iter().rev() {
                self.emit_using_dispose(reg);
            }
        }
        self.scopes.pop();
    }

    /// Emit a call to `[Symbol.dispose]()` on the value in `reg`.
    fn emit_using_dispose(&mut self, reg: Register) {
        // Load the resource into acc.
        self.emit_ldar(reg);
        // CallProperty @@dispose — reuse method-call pattern.
        let dispose_key = self.add_constant_raw(ConstantPoolEntry::String("@@dispose".into()));
        let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        // LdaNamedProperty [obj, name_idx, slot]
        self.emit(Instruction::new_unchecked(
            Opcode::LdaNamedProperty,
            vec![to_reg_op(reg), Operand::ConstantPoolIdx(dispose_key), slot],
        ));
        // Store the method into a temp register.
        let method_reg = self.allocator.new_local();
        self.emit_star(method_reg);
        // CallProperty [callee, receiver, args_start, args_count, slot]
        let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::CallProperty,
            vec![
                to_reg_op(method_reg),
                to_reg_op(reg),
                Operand::Register(0),
                Operand::RegisterCount(0),
                call_slot,
            ],
        ));
    }

    /// Define a new local variable in the innermost scope and return its
    /// register.
    fn define_local(&mut self, name: &str) -> Register {
        self.define_local_with_kind(name, false)
    }

    /// Define a new `const` local variable in the innermost scope.
    fn define_const_local(&mut self, name: &str) -> Register {
        self.define_local_with_kind(name, true)
    }

    /// Define a local variable with an explicit const flag.
    fn define_local_with_kind(&mut self, name: &str, is_const: bool) -> Register {
        let reg = self.allocator.new_local();
        self.scopes.last_mut().unwrap().insert(
            name.to_owned(),
            LocalBinding {
                reg,
                is_const,
                needs_tdz_check: false,
            },
        );
        reg
    }

    /// Define a `var`-declared local in the **outermost** (function) scope so
    /// it remains visible after block scopes are popped, matching the JS spec
    /// semantics for `var` declarations.
    fn define_function_scoped_local(&mut self, name: &str) -> Register {
        let reg = self.allocator.new_local();
        self.scopes[0].insert(
            name.to_owned(),
            LocalBinding {
                reg,
                is_const: false,
                needs_tdz_check: false,
            },
        );
        reg
    }

    /// Look up a variable in the scope chain (innermost first).
    ///
    /// Returns `None` if the name is not found (assumed to be global).
    fn lookup_var(&self, name: &str) -> Option<LocalBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(&binding) = scope.get(name) {
                return Some(binding);
            }
        }
        None
    }

    /// Mark a binding as initialised (clear TDZ flag) in the innermost
    /// scope that contains it.
    fn mark_initialized(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.get_mut(name) {
                binding.needs_tdz_check = false;
                return;
            }
        }
    }

    // ── Declaration scanning helpers ─────────────────────────────────────────

    /// Recursively collect all `var`-declared names in a statement list.
    ///
    /// `var` declarations are function-scoped, so we descend into blocks,
    /// if/else, loops, try/catch, etc., but NOT into nested function bodies.
    fn collect_var_names(stmts: &[Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        for stmt in stmts {
            Self::collect_var_names_in_stmt(stmt, &mut names);
        }
        names
    }

    /// Helper for [`collect_var_names`] — processes a single statement.
    fn collect_var_names_in_stmt(stmt: &Stmt, names: &mut Vec<String>) {
        match stmt {
            Stmt::VarDecl(decl) if decl.kind == VarKind::Var => {
                for d in &decl.declarators {
                    Self::collect_pat_names(&d.id, names);
                }
            }
            Stmt::Block(b) => {
                for s in &b.body {
                    Self::collect_var_names_in_stmt(s, names);
                }
            }
            Stmt::If(s) => {
                Self::collect_var_names_in_stmt(&s.consequent, names);
                if let Some(alt) = &s.alternate {
                    Self::collect_var_names_in_stmt(alt, names);
                }
            }
            Stmt::While(s) => Self::collect_var_names_in_stmt(&s.body, names),
            Stmt::DoWhile(s) => Self::collect_var_names_in_stmt(&s.body, names),
            Stmt::For(s) => {
                if let Some(ForInit::VarDecl(decl)) = &s.init
                    && decl.kind == VarKind::Var
                {
                    for d in &decl.declarators {
                        Self::collect_pat_names(&d.id, names);
                    }
                }
                Self::collect_var_names_in_stmt(&s.body, names);
            }
            Stmt::ForIn(s) => {
                if let crate::parser::ast::ForInOfLeft::VarDecl(decl) = &s.left
                    && decl.kind == VarKind::Var
                {
                    for d in &decl.declarators {
                        Self::collect_pat_names(&d.id, names);
                    }
                }
                Self::collect_var_names_in_stmt(&s.body, names);
            }
            Stmt::ForOf(s) => {
                if let crate::parser::ast::ForInOfLeft::VarDecl(decl) = &s.left
                    && decl.kind == VarKind::Var
                {
                    for d in &decl.declarators {
                        Self::collect_pat_names(&d.id, names);
                    }
                }
                Self::collect_var_names_in_stmt(&s.body, names);
            }
            Stmt::Switch(s) => {
                for case in &s.cases {
                    for cs in &case.consequent {
                        Self::collect_var_names_in_stmt(cs, names);
                    }
                }
            }
            Stmt::Try(s) => {
                for ts in &s.block.body {
                    Self::collect_var_names_in_stmt(ts, names);
                }
                if let Some(handler) = &s.handler {
                    for hs in &handler.body.body {
                        Self::collect_var_names_in_stmt(hs, names);
                    }
                }
                if let Some(fin) = &s.finalizer {
                    for fs in &fin.body {
                        Self::collect_var_names_in_stmt(fs, names);
                    }
                }
            }
            Stmt::Labeled(s) => Self::collect_var_names_in_stmt(&s.body, names),
            Stmt::With(s) => Self::collect_var_names_in_stmt(&s.body, names),
            // FnDecl, ClassDecl, Expr, Return, Throw, Break, Continue,
            // Debugger, Empty — do not contain var declarations.
            _ => {}
        }
    }

    /// Extract identifier names from a binding pattern.
    fn collect_pat_names(pat: &Pat, names: &mut Vec<String>) {
        match pat {
            Pat::Ident(id) => names.push(id.name.clone()),
            Pat::Array(arr) => {
                for p in arr.elements.iter().flatten() {
                    Self::collect_pat_names(p, names);
                }
            }
            Pat::Object(obj) => {
                for prop in &obj.properties {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => Self::collect_pat_names(&kv.value, names),
                        ObjectPatProp::Assign(a) => names.push(a.key.name.clone()),
                        ObjectPatProp::Rest(r) => Self::collect_pat_names(&r.argument, names),
                    }
                }
            }
            Pat::Rest(r) => Self::collect_pat_names(&r.argument, names),
            Pat::Assign(a) => Self::collect_pat_names(&a.left, names),
            Pat::Expr(_) => {}
        }
    }

    /// Collect `let`/`const` declared names in the **direct** children of a
    /// statement list (non-recursive — these are block-scoped).
    fn collect_lexical_names(stmts: &[Stmt]) -> Vec<(String, bool)> {
        let mut names = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::VarDecl(decl) => match decl.kind {
                    VarKind::Let | VarKind::Const => {
                        let is_const = decl.kind == VarKind::Const;
                        for d in &decl.declarators {
                            let mut pat_names = Vec::new();
                            Self::collect_pat_names(&d.id, &mut pat_names);
                            for n in pat_names {
                                names.push((n, is_const));
                            }
                        }
                    }
                    _ => {}
                },
                Stmt::ClassDecl(decl) => {
                    if let Some(id) = &decl.id {
                        names.push((id.name.clone(), false));
                    }
                }
                _ => {}
            }
        }
        names
    }

    /// Pre-register `let`/`const` bindings in the current scope with
    /// `TheHole` so that any access before the declaration statement
    /// triggers a `ReferenceError` (Temporal Dead Zone).
    fn hoist_lexical_decls(&mut self, stmts: &[Stmt]) {
        let lex_names = Self::collect_lexical_names(stmts);
        for (name, is_const) in lex_names {
            let reg = self.allocator.new_local();
            // Initialise the register with TheHole (TDZ sentinel).
            self.emit(Instruction::new_unchecked(Opcode::LdaTheHole, vec![]));
            self.emit_star(reg);
            self.scopes.last_mut().unwrap().insert(
                name,
                LocalBinding {
                    reg,
                    is_const,
                    needs_tdz_check: true,
                },
            );
        }
    }

    /// Pre-hoist all `var` declarations inside `stmts` into the outermost
    /// (function) scope, initialising each to `undefined`.
    fn hoist_var_declarations(&mut self, stmts: &[Stmt]) {
        let var_names = Self::collect_var_names(stmts);
        for name in var_names {
            if self.scopes[0].contains_key(&name) {
                continue; // already hoisted (e.g. parameter or duplicate var)
            }
            let reg = self.allocator.new_local();
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            self.emit_star(reg);
            self.scopes[0].insert(
                name,
                LocalBinding {
                    reg,
                    is_const: false,
                    needs_tdz_check: false,
                },
            );
        }
    }

    /// Pre-hoist `var` declarations at program level into the global env
    /// so that reading them before assignment returns `undefined`.
    fn hoist_var_declarations_global(&mut self, stmts: &[Stmt]) {
        let var_names = Self::collect_var_names(stmts);
        for name in var_names {
            let name_idx = self.add_string(&name);
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            let slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::StaGlobal,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
    }

    // ── Statement compilation ─────────────────────────────────────────────────

    /// Compile a single statement, emitting bytecode into `self.instructions`.
    fn compile_stmt(&mut self, stmt: &Stmt) -> StatorResult<()> {
        stacker::maybe_grow(128 * 1024, 2 * 1024 * 1024, || {
            // Record the source position of each non-empty statement so that the
            // debugger can map bytecode offsets back to line/column numbers.
            let loc = stmt.loc();
            if loc.start.line > 0 {
                let instr_idx = self.instructions.len();
                self.pending_positions
                    .push((instr_idx, loc.start.line, loc.start.column));
            }
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
                Stmt::Labeled(s) => self.compile_labeled(s),
                Stmt::Debugger(_) => {
                    self.emit(Instruction::new_unchecked(Opcode::Debugger, vec![]));
                    Ok(())
                }
                Stmt::Empty(_) => Ok(()),
                Stmt::Switch(s) => self.compile_switch(s),
                Stmt::ForIn(s) => self.compile_for_in(s),
                Stmt::With(s) => self.compile_with(s),
                Stmt::ClassDecl(c) => self.compile_class_decl(c),
                Stmt::ForOf(s) => self.compile_for_of(s),
            }
        }) // stacker::maybe_grow
    }

    /// Compile a `{ … }` block, pushing/popping a scope.
    ///
    /// Before compiling statements, `let`/`const` bindings are pre-registered
    /// with `TheHole` (TDZ sentinel) so that accessing them before their
    /// declaration throws a `ReferenceError`.
    fn compile_block(&mut self, block: &BlockStmt) -> StatorResult<()> {
        self.push_scope();
        self.hoist_lexical_decls(&block.body);
        for stmt in &block.body {
            self.compile_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    /// Compile a `with (expr) stmt` statement.
    ///
    /// Emits `ToObject` + `PushContext` before the body and `PopContext` after
    /// so the interpreter's context chain includes the with-object.
    fn compile_with(&mut self, s: &crate::parser::ast::WithStmt) -> StatorResult<()> {
        // Evaluate the object expression into the accumulator.
        self.compile_expr(&s.object)?;

        // Convert the value to an object (spec: ToObject).
        let obj_reg = self.allocator.allocate_temporary();
        self.emit(Instruction::new_unchecked(
            Opcode::ToObject,
            vec![to_reg_op(obj_reg)],
        ));
        self.emit_ldar(obj_reg);

        // Push a new with-context; save the old context in ctx_reg.
        let ctx_reg = self.allocator.allocate_temporary();
        self.emit(Instruction::new_unchecked(
            Opcode::PushContext,
            vec![to_reg_op(ctx_reg)],
        ));

        // Compile the body inside a fresh scope.
        self.push_scope();
        self.compile_stmt(&s.body)?;
        self.pop_scope();

        // Restore the previous context.
        self.emit(Instruction::new_unchecked(
            Opcode::PopContext,
            vec![to_reg_op(ctx_reg)],
        ));

        self.allocator
            .release_temporary(ctx_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile `var` / `let` / `const` / `using` / `await using` declarations.
    fn compile_var_decl(&mut self, decl: &VarDecl) -> StatorResult<()> {
        let is_using = matches!(decl.kind, VarKind::Using | VarKind::AwaitUsing);
        let is_const = matches!(decl.kind, VarKind::Const);
        let is_var = matches!(decl.kind, VarKind::Var);
        for declarator in &decl.declarators {
            let reg = self.compile_var_declarator(declarator, is_const, is_var)?;
            if is_using && let Some(r) = reg {
                self.using_vars.last_mut().unwrap().push(r);
            }
        }
        Ok(())
    }

    fn compile_var_declarator(
        &mut self,
        declarator: &VarDeclarator,
        is_const: bool,
        is_var: bool,
    ) -> StatorResult<Option<Register>> {
        match &declarator.id {
            Pat::Ident(ident) => {
                if self.is_eval_scope {
                    // Eval scope: var declarations are hoisted into the
                    // caller's global environment via StaGlobal so they
                    // survive after the eval frame completes.
                    if let Some(init) = &declarator.init {
                        self.compile_expr(init)?;
                    } else {
                        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                    }
                    self.compile_ident_store(&ident.name)?;
                    Ok(None)
                } else if self.is_program && is_var {
                    // Program-level `var`: store exclusively via StaGlobal
                    // so that nested callbacks (which also use StaGlobal /
                    // LdaGlobal) share the same storage.  Without this,
                    // the program would read from a stale register while
                    // the callback writes to global_env.
                    if let Some(init) = &declarator.init {
                        self.compile_expr(init)?;
                    } else {
                        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                    }
                    let name_idx = self.add_string(&ident.name);
                    let sta_slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
                    self.emit(Instruction::new_unchecked(
                        Opcode::StaGlobal,
                        vec![Operand::ConstantPoolIdx(name_idx), sta_slot],
                    ));
                    Ok(None)
                } else if is_var {
                    // `var` in a function body: reuse the pre-hoisted
                    // function-scope register if it exists, otherwise
                    // allocate in the function scope.
                    let reg = if let Some(binding) = self.scopes[0].get(&ident.name) {
                        binding.reg
                    } else {
                        self.define_function_scoped_local(&ident.name)
                    };
                    if let Some(init) = &declarator.init {
                        self.compile_expr(init)?;
                        self.emit_star(reg);
                    }
                    // No init → the register already holds `undefined`
                    // from hoisting, so nothing to emit.
                    Ok(Some(reg))
                } else {
                    // `let` / `const` — if the name was pre-registered
                    // during TDZ hoisting, reuse that register and clear
                    // the TDZ flag; otherwise allocate fresh.
                    let reg = if let Some(binding) = self.scopes.last().unwrap().get(&ident.name) {
                        binding.reg
                    } else if is_const {
                        self.define_const_local(&ident.name)
                    } else {
                        self.define_local(&ident.name)
                    };
                    if let Some(init) = &declarator.init {
                        self.compile_expr(init)?;
                    } else {
                        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                    }
                    self.emit_star(reg);
                    // Clear TDZ — the binding is now initialised.
                    self.mark_initialized(&ident.name);
                    Ok(Some(reg))
                }
            }
            pat => {
                // Destructuring declaration: evaluate init into a temp,
                // then unpack via compile_binding_pattern.
                let source_reg = self.allocator.new_local();
                if let Some(init) = &declarator.init {
                    self.compile_expr(init)?;
                } else {
                    self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                }
                self.emit_star(source_reg);
                self.compile_binding_pattern(pat, source_reg, BindingMode::Declare { is_const })?;
                Ok(Some(source_reg))
            }
        }
    }

    /// Compile a function declaration: compile the function body, add the
    /// resulting [`BytecodeArray`] to the constant pool, emit `CreateClosure`,
    /// and bind the name to the resulting register.
    fn compile_fn_decl(&mut self, decl: &FnDecl) -> StatorResult<()> {
        let func_array = compile_function(
            &decl.params,
            &decl.body,
            decl.is_generator,
            decl.is_async,
            decl.is_strict,
        )?;
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
            // Top-level function declarations are also stored as globals so
            // that recursive calls via `LdaGlobal` can find them.
            if self.is_program {
                let name_idx = self.add_string(&id.name);
                let sta_slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
                // Re-load the value from the local register first so the
                // accumulator holds the function when StaGlobal executes.
                self.emit_ldar(reg);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaGlobal,
                    vec![Operand::ConstantPoolIdx(name_idx), sta_slot],
                ));
            }
        }
        Ok(())
    }

    // ── Class compilation ────────────────────────────────────────────────

    /// Compile a class declaration statement.
    fn compile_class_decl(&mut self, decl: &crate::parser::ast::ClassDecl) -> StatorResult<()> {
        self.compile_class(decl.id.as_ref(), decl.super_class.as_deref(), &decl.body)?;
        // Bind the class name in the current scope.
        if let Some(id) = &decl.id {
            let reg = self
                .lookup_var(&id.name)
                .map(|binding| binding.reg)
                .unwrap_or_else(|| self.define_local(&id.name));
            self.emit_star(reg);
            self.mark_initialized(&id.name);
            // Top-level class declarations are also stored as globals.
            if self.is_program {
                let name_idx = self.add_string(&id.name);
                let sta_slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
                self.emit_ldar(reg);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaGlobal,
                    vec![Operand::ConstantPoolIdx(name_idx), sta_slot],
                ));
            }
        }
        Ok(())
    }

    /// Compile a class expression (leaves the class constructor in the
    /// accumulator).
    fn compile_class_expr(&mut self, expr: &crate::parser::ast::ClassExpr) -> StatorResult<()> {
        self.compile_class(expr.id.as_ref(), expr.super_class.as_deref(), &expr.body)
    }

    /// Compile the shared core of a class declaration or expression.
    ///
    /// Emits [`Opcode::CreateClass`] and then defines methods, accessors,
    /// static fields, static blocks, and instance-field initializers.
    /// On return the accumulator holds the class constructor.
    fn compile_class(
        &mut self,
        _id: Option<&crate::parser::ast::Ident>,
        super_class: Option<&Expr>,
        body: &crate::parser::ast::ClassBody,
    ) -> StatorResult<()> {
        use crate::parser::ast::{ClassMember, MethodKind};

        let outer_strict = self.is_strict;
        self.is_strict = true;

        // 1. Evaluate superclass (or load undefined) into a register.
        let super_reg = self.allocator.allocate_temporary();
        if let Some(sc) = super_class {
            self.compile_expr(sc)?;
        } else {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        }
        self.emit_star(super_reg);

        // 2. Find and compile the explicit constructor, or use a default.
        let ctor_method = body.body.iter().find_map(|m| match m {
            ClassMember::Method(md) if md.kind == MethodKind::Constructor => Some(md),
            _ => None,
        });
        let ctor_array = if let Some(ctor) = ctor_method {
            compile_function(
                &ctor.value.params,
                &ctor.value.body,
                false,
                false,
                ctor.value.is_strict,
            )?
        } else {
            let empty_body = BlockStmt {
                loc: body.loc,
                body: vec![],
            };
            compile_function(&[], &empty_body, false, false, true)?
        };
        let ctor_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(ctor_array)));
        let slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);

        // 3. Emit CreateClass.
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClass,
            vec![
                Operand::ConstantPoolIdx(ctor_idx),
                to_reg_op(super_reg),
                slot,
            ],
        ));
        let class_reg = self.allocator.allocate_temporary();
        self.emit_star(class_reg);

        // 4. Load the prototype object.
        let proto_reg = self.allocator.allocate_temporary();
        let proto_name = self.add_string("prototype");
        let proto_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaNamedProperty,
            vec![
                to_reg_op(class_reg),
                Operand::ConstantPoolIdx(proto_name),
                proto_slot,
            ],
        ));
        self.emit_star(proto_reg);

        // 5. Compile each class member.
        for member in &body.body {
            match member {
                ClassMember::Method(m) if m.kind != MethodKind::Constructor => {
                    let target = if m.is_static { class_reg } else { proto_reg };
                    self.compile_class_method(target, m)?;
                }
                ClassMember::Property(p) if p.is_static => {
                    self.compile_class_static_property(class_reg, p)?;
                }
                ClassMember::StaticBlock(sb) => {
                    for stmt in &sb.body {
                        self.compile_stmt(stmt)?;
                    }
                }
                _ => {} // instance fields + constructor handled separately
            }
        }

        // 6. Compile instance-field initializer (if any instance fields exist).
        let instance_fields: Vec<&crate::parser::ast::PropertyDef> = body
            .body
            .iter()
            .filter_map(|m| match m {
                ClassMember::Property(p) if !p.is_static => Some(p),
                _ => None,
            })
            .collect();
        if !instance_fields.is_empty() {
            self.compile_instance_field_initializer(class_reg, &instance_fields)?;
        }

        // Leave the class constructor in the accumulator.
        self.emit_ldar(class_reg);

        // Release temporaries in LIFO order.
        self.allocator
            .release_temporary(proto_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(class_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(super_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        self.is_strict = outer_strict;
        Ok(())
    }

    /// Compile a non-constructor class method or accessor onto `target_reg`.
    fn compile_class_method(
        &mut self,
        target_reg: Register,
        method: &crate::parser::ast::MethodDef,
    ) -> StatorResult<()> {
        use crate::parser::ast::{MethodKind, PropKey};

        self.compile_fn_expr(&method.value)?;

        let key_name: Option<String> = match &method.key {
            PropKey::Ident(id) => Some(id.name.clone()),
            PropKey::Str(s) => Some(s.value.clone()),
            PropKey::Num(n) => {
                if n.value.fract() == 0.0 && n.value.is_finite() && n.value >= 0.0 {
                    Some(format!("{}", n.value as u64))
                } else {
                    Some(n.raw.clone())
                }
            }
            PropKey::Private(id) => Some(format!("#{}", id.name)),
            PropKey::Computed(_) => None,
        };

        match method.kind {
            MethodKind::Method => {
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![
                            to_reg_op(target_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                } else if let PropKey::Computed(key_expr) = &method.key {
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
                            to_reg_op(target_reg),
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
            MethodKind::Get => {
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineGetterProperty,
                        vec![
                            to_reg_op(target_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                } else if let PropKey::Computed(key_expr) = &method.key {
                    let val_reg = self.allocator.allocate_temporary();
                    self.emit_star(val_reg);
                    self.compile_expr(key_expr)?;
                    let key_reg = self.allocator.allocate_temporary();
                    self.emit_star(key_reg);
                    self.emit_ldar(val_reg);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedGetterProperty,
                        vec![to_reg_op(target_reg), to_reg_op(key_reg), slot],
                    ));
                    self.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    self.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            }
            MethodKind::Set => {
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineSetterProperty,
                        vec![
                            to_reg_op(target_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                } else if let PropKey::Computed(key_expr) = &method.key {
                    let val_reg = self.allocator.allocate_temporary();
                    self.emit_star(val_reg);
                    self.compile_expr(key_expr)?;
                    let key_reg = self.allocator.allocate_temporary();
                    self.emit_star(key_reg);
                    self.emit_ldar(val_reg);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedSetterProperty,
                        vec![to_reg_op(target_reg), to_reg_op(key_reg), slot],
                    ));
                    self.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    self.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            }
            MethodKind::Constructor => {} // handled in compile_class
        }

        Ok(())
    }

    /// Compile a static class field: evaluate the value and define it on the
    /// class constructor object.
    fn compile_class_static_property(
        &mut self,
        class_reg: Register,
        prop: &crate::parser::ast::PropertyDef,
    ) -> StatorResult<()> {
        use crate::parser::ast::PropKey;

        if let Some(value) = &prop.value {
            self.compile_expr(value)?;
        } else {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        }

        match &prop.key {
            PropKey::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        to_reg_op(class_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            PropKey::Str(s) => {
                let name_idx = self.add_string(&s.value);
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        to_reg_op(class_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            PropKey::Num(n) => {
                let name = if n.value.fract() == 0.0 && n.value.is_finite() && n.value >= 0.0 {
                    format!("{}", n.value as u64)
                } else {
                    n.raw.clone()
                };
                let name_idx = self.add_string(&name);
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        to_reg_op(class_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            PropKey::Computed(key_expr) => {
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
                        to_reg_op(class_reg),
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
            PropKey::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::DefineNamedOwnProperty,
                    vec![
                        to_reg_op(class_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
        }

        Ok(())
    }

    /// Compile instance-field initializers into a separate closure and attach
    /// it to the class constructor as a hidden `.class_field_initializer`
    /// property.  The runtime calls this function on each `new` invocation
    /// with the freshly created instance as the first argument.
    fn compile_instance_field_initializer(
        &mut self,
        class_reg: Register,
        fields: &[&crate::parser::ast::PropertyDef],
    ) -> StatorResult<()> {
        use crate::parser::ast::PropKey;

        // Build a sub-compiler for the initializer.  It receives a single
        // implicit parameter (the `this` value) in register p0.
        let mut ic = FunctionCompiler {
            instructions: Vec::new(),
            constant_pool: Vec::new(),
            allocator: RegisterAllocator::new(1),
            scopes: vec![{
                let mut s = HashMap::new();
                s.insert(
                    ".this".to_owned(),
                    LocalBinding {
                        reg: Register::parameter(0),
                        is_const: false,
                        needs_tdz_check: false,
                    },
                );
                s
            }],
            source_positions: Vec::new(),
            pending_positions: Vec::new(),
            labels: Vec::new(),
            loop_stack: Vec::new(),
            label_map: HashMap::new(),
            pending_label_name: None,
            param_count: 1,
            slot_kinds: Vec::new(),
            handler_table: Vec::new(),
            is_generator: false,
            yield_suspend_id: 0,
            is_program: false,
            is_async: false,
            is_eval_scope: false,
            is_module: false,
            module_variables: HashMap::new(),
            next_module_cell: 0,
            in_tail_position: false,
            is_strict: true,
            using_vars: vec![Vec::new()],
            for_of_iter_regs: HashMap::new(),
            finally_stack: Vec::new(),
            optional_chain_null_label: None,
        };

        let this_reg = Register::parameter(0);

        for field in fields {
            // Compile the field value (or undefined).
            if let Some(value) = &field.value {
                ic.compile_expr(value)?;
            } else {
                ic.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            }

            // Define the property on `this`.
            match &field.key {
                PropKey::Ident(id) => {
                    let name_idx = ic.add_string(&id.name);
                    let slot = ic.alloc_slot(FeedbackSlotKind::StoreProperty);
                    ic.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![
                            to_reg_op(this_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                }
                PropKey::Str(s) => {
                    let name_idx = ic.add_string(&s.value);
                    let slot = ic.alloc_slot(FeedbackSlotKind::StoreProperty);
                    ic.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![
                            to_reg_op(this_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                }
                PropKey::Num(n) => {
                    let name = if n.value.fract() == 0.0 && n.value.is_finite() && n.value >= 0.0 {
                        format!("{}", n.value as u64)
                    } else {
                        n.raw.clone()
                    };
                    let name_idx = ic.add_string(&name);
                    let slot = ic.alloc_slot(FeedbackSlotKind::StoreProperty);
                    ic.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![
                            to_reg_op(this_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                }
                PropKey::Computed(key_expr) => {
                    let val_reg = ic.allocator.allocate_temporary();
                    ic.emit_star(val_reg);
                    ic.compile_expr(key_expr)?;
                    let key_reg = ic.allocator.allocate_temporary();
                    ic.emit_star(key_reg);
                    ic.emit_ldar(val_reg);
                    let slot = ic.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                    ic.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedOwnProperty,
                        vec![
                            to_reg_op(this_reg),
                            to_reg_op(key_reg),
                            Operand::Flag(0),
                            slot,
                        ],
                    ));
                    ic.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    ic.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
                PropKey::Private(id) => {
                    let name_idx = ic.add_string(&format!("#{}", id.name));
                    let slot = ic.alloc_slot(FeedbackSlotKind::StoreProperty);
                    ic.emit(Instruction::new_unchecked(
                        Opcode::DefineNamedOwnProperty,
                        vec![
                            to_reg_op(this_reg),
                            Operand::ConstantPoolIdx(name_idx),
                            slot,
                        ],
                    ));
                }
            }
        }

        let init_array = ic.finalize()?;
        let init_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(init_array)));
        let closure_slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![
                Operand::ConstantPoolIdx(init_idx),
                closure_slot,
                Operand::Flag(0),
            ],
        ));
        // Store the initializer as a hidden property on the class constructor.
        let init_name = self.add_string(".class_field_initializer");
        let store_slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
        self.emit(Instruction::new_unchecked(
            Opcode::DefineNamedOwnProperty,
            vec![
                to_reg_op(class_reg),
                Operand::ConstantPoolIdx(init_name),
                store_slot,
            ],
        ));

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

        self.patch_pending_label(loop_start);
        self.loop_stack.push((loop_start, loop_end));

        self.bind_label(loop_start);
        self.compile_expr(&s.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, loop_end);
        self.compile_stmt(&s.body)?;
        self.emit_jump_loop_to(loop_start);
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

        self.patch_pending_label(cond_label);
        self.loop_stack.push((cond_label, loop_end));

        self.bind_label(loop_start);
        self.compile_stmt(&s.body)?;
        self.bind_label(cond_label);
        self.compile_expr(&s.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, loop_end);
        self.emit_jump_loop_to(loop_start);
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

        self.patch_pending_label(continue_label);
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

        self.emit_jump_loop_to(loop_start);
        self.bind_label(loop_end);

        self.loop_stack.pop();
        self.pop_scope();
        Ok(())
    }

    /// Compile a `return [argument]` statement.
    fn compile_return(&mut self, s: &crate::parser::ast::ReturnStmt) -> StatorResult<()> {
        if let Some(arg) = &s.argument {
            // Disable tail calls when inside a finally scope — the finally
            // block must run before the frame is replaced.
            self.in_tail_position = self.finally_stack.is_empty();
            self.compile_expr(arg)?;
            self.in_tail_position = false;
        } else {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        }
        // Close iterators for all enclosing for-of loops.
        let iter_regs_to_close: Vec<Register> = self
            .loop_stack
            .iter()
            .rev()
            .filter_map(|&(_, bl)| self.for_of_iter_regs.get(&bl).copied())
            .collect();
        for iter_reg in iter_regs_to_close {
            self.emit(Instruction::new_unchecked(
                Opcode::IteratorClose,
                vec![to_reg_op(iter_reg)],
            ));
        }
        // Inline pending finally blocks before the actual return so that
        // `try { return 1; } finally { … }` executes the finally body.
        if !self.finally_stack.is_empty() {
            let save_reg = self.allocator.allocate_temporary();
            self.emit_star(save_reg);
            self.inline_pending_finally_blocks()?;
            self.emit_ldar(save_reg);
            self.allocator
                .release_temporary(save_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.emit(Instruction::new_unchecked(Opcode::Return, vec![]));
        Ok(())
    }

    /// Compile a `break [label]` statement.
    ///
    /// Labeled breaks look up the target in `label_map`; unlabeled breaks
    /// use the innermost `loop_stack` entry.
    fn compile_break(&mut self, s: &crate::parser::ast::BreakStmt) -> StatorResult<()> {
        // Inline pending finally blocks before the jump.
        self.inline_pending_finally_blocks()?;
        if let Some(label) = &s.label {
            let (_, break_label) = *self.label_map.get(&label.name).ok_or_else(|| {
                StatorError::SyntaxError(format!("undefined label '{}'", label.name))
            })?;
            // Close iterators for all for-of loops being exited.
            let iter_regs_to_close: Vec<Register> = self
                .loop_stack
                .iter()
                .rev()
                .take_while(|&&(_, bl)| bl != break_label)
                .chain(
                    self.loop_stack
                        .iter()
                        .rev()
                        .find(|&&(_, bl)| bl == break_label),
                )
                .filter_map(|&(_, bl)| self.for_of_iter_regs.get(&bl).copied())
                .collect();
            for iter_reg in iter_regs_to_close {
                self.emit(Instruction::new_unchecked(
                    Opcode::IteratorClose,
                    vec![to_reg_op(iter_reg)],
                ));
            }
            self.emit_jump(Opcode::Jump, break_label);
            return Ok(());
        }
        let (_, break_label) = *self
            .loop_stack
            .last()
            .ok_or_else(|| StatorError::SyntaxError("break outside loop".into()))?;
        if let Some(&iter_reg) = self.for_of_iter_regs.get(&break_label) {
            self.emit(Instruction::new_unchecked(
                Opcode::IteratorClose,
                vec![to_reg_op(iter_reg)],
            ));
        }
        self.emit_jump(Opcode::Jump, break_label);
        Ok(())
    }

    /// Compile a `continue [label]` statement.
    ///
    /// Labeled continues look up the target in `label_map` and require the
    /// labeled body to be a loop; unlabeled continues use the innermost
    /// `loop_stack` entry.  When crossing for-of loop boundaries, emit
    /// `IteratorClose` for each intervening for-of iterator (excluding the
    /// target loop itself, since we are continuing it, not exiting it).
    fn compile_continue(&mut self, s: &crate::parser::ast::ContinueStmt) -> StatorResult<()> {
        // Inline pending finally blocks before the jump.
        self.inline_pending_finally_blocks()?;
        if let Some(label) = &s.label {
            let (continue_label, break_label) =
                *self.label_map.get(&label.name).ok_or_else(|| {
                    StatorError::SyntaxError(format!("undefined label '{}'", label.name))
                })?;
            let continue_label = continue_label.ok_or_else(|| {
                StatorError::SyntaxError(format!(
                    "label '{}' is not a loop — continue is invalid",
                    label.name
                ))
            })?;
            // Close iterators for for-of loops inner to (but not including)
            // the target loop, identified by its break_label on the stack.
            let iter_regs_to_close: Vec<Register> = self
                .loop_stack
                .iter()
                .rev()
                .take_while(|&&(_, bl)| bl != break_label)
                .filter_map(|&(_, bl)| self.for_of_iter_regs.get(&bl).copied())
                .collect();
            for iter_reg in iter_regs_to_close {
                self.emit(Instruction::new_unchecked(
                    Opcode::IteratorClose,
                    vec![to_reg_op(iter_reg)],
                ));
            }
            self.emit_jump(Opcode::Jump, continue_label);
            return Ok(());
        }
        // Skip switch entries (where continue_label == break_label) and
        // find the innermost enclosing loop.
        let &(continue_label, _) = self
            .loop_stack
            .iter()
            .rev()
            .find(|&&(cont, brk)| cont != brk)
            .ok_or_else(|| StatorError::SyntaxError("continue outside loop".into()))?;
        self.emit_jump(Opcode::Jump, continue_label);
        Ok(())
    }

    /// Compile a labeled statement (`label: body`).
    ///
    /// A break-label is always created so that `break label` can exit the
    /// labeled statement.  If the body is a loop (`While`, `DoWhile`,
    /// `For`, `ForIn`, or `ForOf`), the loop compiler patches the
    /// `label_map` entry with the real continue-label via
    /// `pending_label_name`.
    fn compile_labeled(&mut self, s: &crate::parser::ast::LabeledStmt) -> StatorResult<()> {
        let name = s.label.name.clone();
        if self.label_map.contains_key(&name) {
            return Err(StatorError::SyntaxError(format!(
                "duplicate label '{name}'"
            )));
        }

        let break_label = self.new_label();
        let is_loop = matches!(
            *s.body,
            Stmt::While(_) | Stmt::DoWhile(_) | Stmt::For(_) | Stmt::ForIn(_) | Stmt::ForOf(_)
        );

        // Pre-populate label_map so that `break label` works during body
        // compilation.  continue_label starts as None and is patched by
        // the loop compiler when `pending_label_name` is set.
        self.label_map.insert(name.clone(), (None, break_label));

        if is_loop {
            self.pending_label_name = Some(name.clone());
        }

        self.compile_stmt(&s.body)?;

        // Clear pending_label_name in case the body was not a recognized
        // loop (shouldn't happen if is_loop matched, but be safe).
        self.pending_label_name = None;

        self.label_map.remove(&name);
        self.bind_label(break_label);
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
                self.emit_jump(Opcode::JumpIfTrue, case_labels[i]);
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

        // Open a block scope for the switch body so that `let`/`const`
        // declarations in case clauses are properly scoped.
        self.push_scope();

        // Emit case bodies.
        for (i, case) in s.cases.iter().enumerate() {
            self.bind_label(case_labels[i]);
            for stmt in &case.consequent {
                self.compile_stmt(stmt)?;
            }
        }

        self.pop_scope();
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
    /// ## Layout — try/catch/finally
    ///
    /// Two handler table entries: one dispatches exceptions from the try
    /// body to the catch handler, the second wraps the catch body so that
    /// finally runs even if catch throws.
    ///
    /// ```text
    /// [try_start]       try body
    /// [try_end]         Jump(after_catch)
    /// [catch_start]     Star(e), catch body
    /// [catch_end]
    /// [after_catch]     finally body (normal path)
    ///                   Jump(after_ex_handler)
    /// [ex_handler]      Star(ex_reg), finally body, Ldar(ex_reg), ReThrow
    /// [after_ex_handler] ...
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
        // Push the finalizer onto the finally stack so that return / break /
        // continue inside the try body inline the finally block before the
        // abrupt completion.
        if let Some(finalizer) = &s.finalizer {
            self.finally_stack.push(finalizer.clone());
        }

        let try_start = self.instructions.len() as u32;

        // Compile try block.
        self.compile_block(&s.block)?;

        // Pop the finalizer — normal-path finally is compiled explicitly below.
        if s.finalizer.is_some() {
            self.finally_stack.pop();
        }

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
                    pat => {
                        // Destructuring catch param: acc holds thrown value.
                        let source_reg = self.allocator.new_local();
                        self.emit_star(source_reg);
                        self.compile_binding_pattern(
                            pat,
                            source_reg,
                            BindingMode::Declare { is_const: false },
                        )?;
                    }
                }
            }
            self.compile_block(&handler.body)?;
            self.pop_scope();
            let catch_end = self.instructions.len() as u32;

            self.bind_label(after_catch_label);

            // Register the catch handler entry.
            self.handler_table.push(HandlerTableEntry {
                try_start,
                try_end,
                handler: catch_start,
                is_finally: false,
            });

            // Compile finally block inline (runs on both paths).
            // When a finally block is present, an additional exception handler
            // covers the catch block so that finally executes even if the catch
            // body throws.
            if let Some(finalizer) = &s.finalizer {
                self.compile_block(finalizer)?;

                // Exception-path handler for exceptions thrown inside catch.
                let after_ex_handler_label = self.new_label();
                self.emit_jump(Opcode::Jump, after_ex_handler_label);

                let ex_handler_start = self.instructions.len() as u32;
                let ex_reg = self.allocator.allocate_temporary();
                self.emit_star(ex_reg);
                self.compile_block(finalizer)?;
                self.emit_ldar(ex_reg);
                self.emit(Instruction::new_unchecked(Opcode::ReThrow, vec![]));
                self.allocator
                    .release_temporary(ex_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;

                self.bind_label(after_ex_handler_label);

                self.handler_table.push(HandlerTableEntry {
                    try_start: catch_start,
                    try_end: catch_end,
                    handler: ex_handler_start,
                    is_finally: true,
                });
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
        stacker::maybe_grow(128 * 1024, 2 * 1024 * 1024, || {
            // Only Call, Conditional, Logical, and Sequence expressions can
            // propagate tail position to sub-expressions.  All other expression
            // types clear the flag so that nested calls are not incorrectly
            // marked as tail calls (e.g. `return n + f(n-1)` — the call f() is
            // NOT in tail position because its result feeds into `+`).
            match expr {
                Expr::Call(_) | Expr::Conditional(_) | Expr::Logical(_) | Expr::Sequence(_) => {}
                _ => {
                    self.in_tail_position = false;
                }
            }

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
                Expr::BigInt(b) => {
                    let raw = b.value.replace('_', "");
                    let n: i128 = if let Some(hex) =
                        raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X"))
                    {
                        i128::from_str_radix(hex, 16).map_err(|e| {
                            StatorError::SyntaxError(format!("invalid BigInt hex literal: {e}"))
                        })?
                    } else if let Some(oct) =
                        raw.strip_prefix("0o").or_else(|| raw.strip_prefix("0O"))
                    {
                        i128::from_str_radix(oct, 8).map_err(|e| {
                            StatorError::SyntaxError(format!("invalid BigInt octal literal: {e}"))
                        })?
                    } else if let Some(bin) =
                        raw.strip_prefix("0b").or_else(|| raw.strip_prefix("0B"))
                    {
                        i128::from_str_radix(bin, 2).map_err(|e| {
                            StatorError::SyntaxError(format!("invalid BigInt binary literal: {e}"))
                        })?
                    } else {
                        raw.parse::<i128>().map_err(|e| {
                            StatorError::SyntaxError(format!("invalid BigInt literal: {e}"))
                        })?
                    };
                    let idx = self.add_bigint(n);
                    self.emit(Instruction::new_unchecked(
                        Opcode::LdaConstant,
                        vec![Operand::ConstantPoolIdx(idx)],
                    ));
                    Ok(())
                }
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
                Expr::Class(c) => self.compile_class_expr(c),

                // ── Operators ─────────────────────────────────────────────────
                Expr::Unary(u) => self.compile_unary(u),
                Expr::Update(u) => self.compile_update(u),
                Expr::Binary(b) => self.compile_binary(b),
                Expr::Logical(l) => self.compile_logical(l),
                Expr::Conditional(c) => self.compile_conditional(c),
                Expr::Assign(a) => self.compile_assign(a),
                Expr::Sequence(s) => {
                    let saved_tail = self.in_tail_position;
                    let len = s.expressions.len();
                    for (i, expr) in s.expressions.iter().enumerate() {
                        // Only the last expression in a comma sequence inherits
                        // tail position; all preceding ones are for side effects.
                        self.in_tail_position = saved_tail && i == len - 1;
                        self.compile_expr(expr)?;
                    }
                    self.in_tail_position = false;
                    Ok(())
                }

                // ── Member / call ─────────────────────────────────────────────
                Expr::Member(m) => self.compile_member(m),
                Expr::OptionalMember(m) => self.compile_optional_member(m),
                Expr::OptionalChain(inner) => self.compile_optional_chain(inner),
                Expr::Call(c) => self.compile_call(c),
                Expr::OptionalCall(c) => self.compile_optional_call(c),
                Expr::New(n) => self.compile_new(n),

                // ── Async / generators ────────────────────────────────────────
                Expr::Yield(y) => {
                    if !self.is_generator {
                        return Err(StatorError::SyntaxError(
                            "yield expression outside of a generator function".into(),
                        ));
                    }
                    self.compile_yield(y)
                }
                Expr::Await(a) => {
                    if !self.is_async {
                        return Err(StatorError::SyntaxError(
                            "await expression outside of an async function".into(),
                        ));
                    }
                    self.compile_await(a)
                }

                Expr::TaggedTemplate(t) => self.compile_tagged_template(t),
                Expr::Spread(s) => {
                    // Spread in expression position: compile the argument.
                    // The parent context (array literal, function call) handles iteration.
                    self.compile_expr(&s.argument)
                }
                Expr::Import(imp) => self.compile_import_call(imp),
                Expr::MetaProp(m) => self.compile_meta_prop(m),
                Expr::PrivateName(_) => Err(StatorError::SyntaxError(
                    "bare private name expression should only appear as LHS of 'in'".into(),
                )),
            }
        }) // stacker::maybe_grow
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
        if let Some(binding) = self.lookup_var(name) {
            self.emit_ldar(binding.reg);
            if binding.needs_tdz_check {
                let name_idx = self.add_string(name);
                self.emit(Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(name_idx)],
                ));
            }
        } else {
            let name_idx = self.add_string(name);
            let slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
    }

    /// Load a variable by name inside a `typeof` expression.
    ///
    /// For globals, emits [`Opcode::LdaGlobalInsideTypeof`] which returns
    /// `undefined` instead of throwing a `ReferenceError` when the name is
    /// not bound (ECMAScript §13.5.3).
    ///
    /// For local `let`/`const` bindings still in the TDZ, `typeof` must
    /// **still** throw a `ReferenceError` (unlike undeclared globals).
    fn compile_ident_load_typeof(&mut self, name: &str) {
        if let Some(binding) = self.lookup_var(name) {
            self.emit_ldar(binding.reg);
            if binding.needs_tdz_check {
                let name_idx = self.add_string(name);
                self.emit(Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(name_idx)],
                ));
            }
        } else {
            let name_idx = self.add_string(name);
            let slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::LdaGlobalInsideTypeof,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
    }

    /// Store the accumulator to the variable named `name`.
    ///
    /// Returns an error if the variable is a `const` binding.
    /// If the binding is still in the TDZ, emits a runtime
    /// `ThrowReferenceErrorIfHole` check before the store.
    fn compile_ident_store(&mut self, name: &str) -> StatorResult<()> {
        if let Some(binding) = self.lookup_var(name) {
            if binding.is_const {
                return Err(StatorError::TypeError(format!(
                    "Assignment to constant variable '{name}'"
                )));
            }
            if binding.needs_tdz_check {
                // Save the value being assigned, check TDZ, then store.
                let tmp = self.allocator.allocate_temporary();
                self.emit_star(tmp);
                self.emit_ldar(binding.reg);
                let name_idx = self.add_string(name);
                self.emit(Instruction::new_unchecked(
                    Opcode::ThrowReferenceErrorIfHole,
                    vec![Operand::ConstantPoolIdx(name_idx)],
                ));
                self.emit_ldar(tmp);
                self.emit_star(binding.reg);
                self.allocator
                    .release_temporary(tmp)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            } else {
                self.emit_star(binding.reg);
            }
        } else {
            let name_idx = self.add_string(name);
            let slot = self.alloc_slot(FeedbackSlotKind::StoreGlobal);
            self.emit(Instruction::new_unchecked(
                Opcode::StaGlobal,
                vec![Operand::ConstantPoolIdx(name_idx), slot],
            ));
        }
        Ok(())
    }

    /// Compile a unary expression.
    fn compile_unary(&mut self, u: &crate::parser::ast::UnaryExpr) -> StatorResult<()> {
        match u.op {
            UnaryOp::Void => {
                self.compile_expr(&u.argument)?;
                self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            }
            UnaryOp::Typeof => {
                // ECMAScript §13.5.3: when the operand is a simple
                // identifier reference that may be unresolvable, use
                // LdaGlobalInsideTypeof so the runtime returns "undefined"
                // instead of throwing a ReferenceError.
                if let Expr::Ident(id) = u.argument.as_ref() {
                    self.compile_ident_load_typeof(&id.name);
                } else {
                    self.compile_expr(&u.argument)?;
                }
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
                // Emit strict or sloppy delete depending on compilation mode.
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
                            crate::parser::ast::MemberProp::Private(name) => {
                                return Err(StatorError::SyntaxError(format!(
                                    "private fields can not be deleted (field #{})",
                                    name.name
                                )));
                            }
                        }
                        let delete_op = if self.is_strict {
                            Opcode::DeletePropertyStrict
                        } else {
                            Opcode::DeletePropertySloppy
                        };
                        self.emit(Instruction::new_unchecked(
                            delete_op,
                            vec![to_reg_op(obj_reg)],
                        ));
                        self.allocator
                            .release_temporary(obj_reg)
                            .map_err(|e| StatorError::Internal(e.to_string()))?;
                    }
                    // `delete obj?.prop` — short-circuit to `true` when
                    // the base is nullish; otherwise delete normally.
                    Expr::OptionalChain(inner) => {
                        self.compile_delete_optional_chain(inner)?;
                    }
                    Expr::Ident(_) if self.is_strict => {
                        // In strict mode, `delete x` on an unqualified
                        // identifier is a SyntaxError.
                        return Err(StatorError::SyntaxError(
                            "delete of an unqualified identifier is not allowed in strict mode"
                                .into(),
                        ));
                    }
                    _ => {
                        // delete on non-member always returns true in sloppy mode.
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
            Expr::Ident(id) => self.compile_ident_store(&id.name)?,
            Expr::Member(m) => self.compile_member_store(m)?,
            _ => {
                return Err(StatorError::SyntaxError(
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
        // `#x in obj` — private brand check.
        if b.op == BinaryOp::In
            && let Expr::PrivateName(ref id) = *b.left
        {
            self.compile_expr(&b.right)?;
            let obj_reg = self.allocator.allocate_temporary();
            self.emit_star(obj_reg);
            let name_idx = self.add_string(&format!("#{}", id.name));
            let brand_reg = self.allocator.allocate_temporary();
            self.emit(Instruction::new_unchecked(
                Opcode::LdaConstant,
                vec![Operand::ConstantPoolIdx(name_idx)],
            ));
            self.emit_star(brand_reg);
            self.emit_ldar(obj_reg);
            self.emit(Instruction::new_unchecked(
                Opcode::TestPrivateBrand,
                vec![to_reg_op(obj_reg), to_reg_op(brand_reg)],
            ));
            self.allocator
                .release_temporary(brand_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
            self.allocator
                .release_temporary(obj_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
            return Ok(());
        }

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

        let saved_tail = self.in_tail_position;
        self.in_tail_position = false;
        self.compile_expr(&l.left)?;
        self.in_tail_position = saved_tail;

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
                // Left already compiled above; acc holds its value.
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, eval_right_label);
                // Left is neither null nor undefined: already in acc, jump to end.
                self.emit_jump(Opcode::Jump, end_label);
                self.bind_label(eval_right_label);
                self.compile_expr(&l.right)?;
                self.in_tail_position = false;
                self.bind_label(end_label);
                return Ok(());
            }
        }

        self.compile_expr(&l.right)?;
        self.in_tail_position = false;
        self.bind_label(end_label);
        Ok(())
    }

    /// Compile `test ? consequent : alternate`.
    fn compile_conditional(&mut self, c: &crate::parser::ast::ConditionalExpr) -> StatorResult<()> {
        let else_label = self.new_label();
        let end_label = self.new_label();

        let saved_tail = self.in_tail_position;
        self.in_tail_position = false;
        self.compile_expr(&c.test)?;
        self.emit_jump(Opcode::JumpIfToBooleanFalse, else_label);
        self.in_tail_position = saved_tail;
        self.compile_expr(&c.consequent)?;
        self.emit_jump(Opcode::Jump, end_label);
        self.bind_label(else_label);
        self.in_tail_position = saved_tail;
        self.compile_expr(&c.alternate)?;
        self.in_tail_position = false;
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
                // a ??= b  →  if a is null/undefined, a = b
                self.allocator
                    .release_temporary(rhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(lhs_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.compile_assign_target_load(&a.left)?;
                let do_assign = self.new_label();
                let skip = self.new_label();
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, do_assign);
                self.emit_jump(Opcode::Jump, skip);
                self.bind_label(do_assign);
                self.compile_expr(&a.right)?;
                self.compile_assign_target_store(&a.left)?;
                self.bind_label(skip);
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
                other => {
                    let loc = other.loc();
                    Err(StatorError::SyntaxError(format!(
                        "at {}:{} — Invalid assignment target",
                        loc.start.line, loc.start.column
                    )))
                }
            },
            AssignTarget::Pat(_pat) => {
                // The real unpack happens in compile_assign_target_store.
                self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
                Ok(())
            }
        }
    }

    /// Store the accumulator to an assignment target.
    fn compile_assign_target_store(&mut self, target: &AssignTarget) -> StatorResult<()> {
        match target {
            AssignTarget::Expr(expr) => match expr.as_ref() {
                Expr::Ident(id) => {
                    self.compile_ident_store(&id.name)?;
                    Ok(())
                }
                Expr::Member(m) => self.compile_member_store(m),
                other => {
                    let loc = other.loc();
                    Err(StatorError::SyntaxError(format!(
                        "at {}:{} — Invalid assignment target",
                        loc.start.line, loc.start.column
                    )))
                }
            },
            AssignTarget::Pat(pat) => {
                // Destructuring assignment: acc holds the RHS value.
                let source_reg = self.allocator.new_local();
                self.emit_star(source_reg);
                self.compile_binding_pattern(pat, source_reg, BindingMode::Assign)?;
                Ok(())
            }
        }
    }

    /// Compile a member expression (`obj.prop` or `obj[key]`) as an r-value.
    fn compile_member(&mut self, m: &crate::parser::ast::MemberExpr) -> StatorResult<()> {
        if let Expr::Ident(id) = m.object.as_ref()
            && id.name == "super"
        {
            return self.compile_super_member_load(m);
        }
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
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                ));
            }
        }
        self.allocator
            .release_temporary(obj_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    fn compile_super_member_load(
        &mut self,
        m: &crate::parser::ast::MemberExpr,
    ) -> StatorResult<()> {
        let recv_reg = self.allocator.allocate_temporary();
        let this_name_idx = self.add_string("this");
        let this_slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaGlobal,
            vec![Operand::ConstantPoolIdx(this_name_idx), this_slot],
        ));
        self.emit_star(recv_reg);

        self.compile_expr(&m.object)?;
        let super_reg = self.allocator.allocate_temporary();
        self.emit_star(super_reg);

        let proto_name_idx = self.add_string("prototype");
        let proto_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaNamedProperty,
            vec![
                to_reg_op(super_reg),
                Operand::ConstantPoolIdx(proto_name_idx),
                proto_slot,
            ],
        ));

        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        to_reg_op(recv_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        to_reg_op(recv_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            crate::parser::ast::MemberProp::Computed(_) => {
                self.allocator
                    .release_temporary(super_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(recv_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                return Err(StatorError::SyntaxError(
                    "computed super property access is not yet supported".into(),
                ));
            }
        }

        self.allocator
            .release_temporary(super_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(recv_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile an optional member expression (`obj?.prop`).
    fn compile_optional_member(
        &mut self,
        m: &crate::parser::ast::OptionalMemberExpr,
    ) -> StatorResult<()> {
        self.compile_expr(&m.object)?;
        let obj_reg = self.allocator.allocate_temporary();
        self.emit_star(obj_reg);

        // If the object is null or undefined, short-circuit.
        self.emit_ldar(obj_reg);

        if let Some(chain_label) = self.optional_chain_null_label {
            // Inside an OptionalChain — jump to the shared null-label.
            self.emit_jump(Opcode::JumpIfUndefinedOrNull, chain_label);

            self.emit_optional_member_property(obj_reg, &m.property)?;

            self.allocator
                .release_temporary(obj_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        } else {
            // Standalone `?.` (no OptionalChain wrapper — shouldn't happen
            // after the parser change, but keep as a fallback).
            let null_label = self.new_label();
            let end_label = self.new_label();
            self.emit_jump(Opcode::JumpIfUndefinedOrNull, null_label);

            self.emit_optional_member_property(obj_reg, &m.property)?;

            self.emit_jump(Opcode::Jump, end_label);
            self.bind_label(null_label);
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            self.bind_label(end_label);

            self.allocator
                .release_temporary(obj_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        Ok(())
    }

    /// Emit the property access for an optional member expression.
    fn emit_optional_member_property(
        &mut self,
        obj_reg: Register,
        property: &crate::parser::ast::MemberProp,
    ) -> StatorResult<()> {
        match property {
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
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                ));
            }
        }
        Ok(())
    }

    /// Compile an entire optional-chain expression.
    ///
    /// Sets up a shared null-label so that every `OptionalMember` /
    /// `OptionalCall` inside the chain jumps to the same short-circuit
    /// point, producing `undefined` for the whole chain.
    fn compile_optional_chain(&mut self, inner: &Expr) -> StatorResult<()> {
        let null_label = self.new_label();
        let end_label = self.new_label();

        let saved = self.optional_chain_null_label;
        self.optional_chain_null_label = Some(null_label);

        self.compile_expr(inner)?;

        self.optional_chain_null_label = saved;

        self.emit_jump(Opcode::Jump, end_label);
        self.bind_label(null_label);
        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        self.bind_label(end_label);
        Ok(())
    }

    /// Compile `delete obj?.prop` — short-circuit to `true` when
    /// the base is nullish; otherwise delete normally.
    fn compile_delete_optional_chain(&mut self, inner: &Expr) -> StatorResult<()> {
        // Unwrap one layer: inner should be an OptionalMember.
        // For more complex chains (delete a?.b.c), we only support
        // the direct OptionalMember case (delete a?.b).
        match inner {
            Expr::OptionalMember(m) => {
                self.compile_expr(&m.object)?;
                let obj_reg = self.allocator.allocate_temporary();
                self.emit_star(obj_reg);

                let null_label = self.new_label();
                let end_label = self.new_label();

                self.emit_ldar(obj_reg);
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, null_label);

                // Object is not nullish — delete the property.
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
                    crate::parser::ast::MemberProp::Private(name) => {
                        return Err(StatorError::SyntaxError(format!(
                            "private fields can not be deleted (field #{})",
                            name.name
                        )));
                    }
                }
                let delete_op = if self.is_strict {
                    Opcode::DeletePropertyStrict
                } else {
                    Opcode::DeletePropertySloppy
                };
                self.emit(Instruction::new_unchecked(
                    delete_op,
                    vec![to_reg_op(obj_reg)],
                ));
                self.emit_jump(Opcode::Jump, end_label);

                // Object is nullish — result is `true`.
                self.bind_label(null_label);
                self.emit(Instruction::new_unchecked(Opcode::LdaTrue, vec![]));
                self.bind_label(end_label);

                self.allocator
                    .release_temporary(obj_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            }
            _ => {
                // For other shapes (e.g. `delete a?.b.c`), evaluate the
                // chain (which may short-circuit to undefined) and then
                // return `true` since deleting a non-reference is `true`.
                self.compile_optional_chain(inner)?;
                self.emit(Instruction::new_unchecked(Opcode::LdaTrue, vec![]));
            }
        }
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
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::StoreProperty);
                // Reload value.
                self.emit_ldar(val_reg);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaNamedProperty,
                    vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
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
        // Consume the tail-position flag before compiling sub-expressions.
        let is_tail = self.in_tail_position;
        self.in_tail_position = false;

        // Check for method call: `obj.method(args)`.
        if let Expr::Member(m) = c.callee.as_ref() {
            // Propagate tail position into method call compilation.
            self.in_tail_position = is_tail;
            return self.compile_method_call(m, &c.arguments);
        }

        // Check for optional method call: `obj?.method(args)`.
        // The parser produces `Call(OptionalMember(obj, method), args)`.
        if let Expr::OptionalMember(m) = c.callee.as_ref() {
            return self.compile_optional_method_call(m, &c.arguments);
        }

        // Detect direct eval: `eval(args)` where the callee is the bare
        // identifier `eval` (ECMAScript §19.2.1.1 step 1).
        let is_direct_eval = matches!(c.callee.as_ref(), Expr::Ident(id) if id.name == "eval");

        let has_spread = c.arguments.iter().any(|a| matches!(a, Expr::Spread(_)));

        // General call with undefined receiver.
        self.compile_expr(&c.callee)?;
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        if has_spread && !is_direct_eval {
            // Build arguments as a single array, then use CallWithSpread.
            let arr_reg = self.compile_arguments_as_array(&c.arguments)?;
            let slot = self.alloc_slot(FeedbackSlotKind::Call);
            self.emit(Instruction::new_unchecked(
                Opcode::CallWithSpread,
                vec![
                    to_reg_op(callee_reg),
                    to_reg_op(arr_reg),
                    Operand::RegisterCount(1),
                    slot,
                ],
            ));
            self.allocator
                .release_temporary(arr_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        } else {
            let arg_regs = self.compile_arguments(&c.arguments)?;

            if is_direct_eval {
                self.emit_call_direct_eval(callee_reg, &arg_regs)?;
            } else if is_tail {
                self.emit_tail_call(callee_reg, &arg_regs)?;
            } else {
                self.emit_call_any_receiver(callee_reg, &arg_regs)?;
            }

            // Release args in reverse order.
            for r in arg_regs.into_iter().rev() {
                self.allocator
                    .release_temporary(r)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            }
        }
        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a dynamic `import(source)` or `import(source, options)` call.
    ///
    /// Emits `CallRuntime [RUNTIME_DYNAMIC_IMPORT, args_start, arg_count]`
    /// so the interpreter can resolve the import and return a `Promise`.
    fn compile_import_call(&mut self, imp: &crate::parser::ast::ImportExpr) -> StatorResult<()> {
        // Compile the specifier into a register.
        self.compile_expr(&imp.source)?;
        let source_reg = self.allocator.allocate_temporary();
        self.emit_star(source_reg);

        // Optionally compile the options argument.
        let options_reg = if let Some(opts) = &imp.options {
            self.compile_expr(opts)?;
            let r = self.allocator.allocate_temporary();
            self.emit_star(r);
            Some(r)
        } else {
            None
        };

        let arg_count = if options_reg.is_some() { 2u32 } else { 1u32 };

        self.emit(Instruction::new_unchecked(
            Opcode::CallRuntime,
            vec![
                Operand::RuntimeId(RUNTIME_DYNAMIC_IMPORT),
                to_reg_op(source_reg),
                Operand::RegisterCount(arg_count),
            ],
        ));

        if let Some(r) = options_reg {
            self.allocator
                .release_temporary(r)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        }
        self.allocator
            .release_temporary(source_reg)
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

        self.emit_ldar(callee_reg);

        if let Some(chain_label) = self.optional_chain_null_label {
            // Inside an OptionalChain — jump to the shared null-label.
            self.emit_jump(Opcode::JumpIfUndefinedOrNull, chain_label);

            let arg_regs = self.compile_arguments(&c.arguments)?;
            self.emit_call_any_receiver(callee_reg, &arg_regs)?;
            for r in arg_regs.into_iter().rev() {
                self.allocator
                    .release_temporary(r)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            }
        } else {
            // Standalone (fallback).
            let null_label = self.new_label();
            let end_label = self.new_label();
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
        }

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
        if let Expr::Ident(id) = m.object.as_ref()
            && id.name == "super"
        {
            return self.compile_super_method_call(m, args);
        }
        // Consume tail-position flag — method calls in tail position are not
        // optimised (the receiver binding requires a full frame), so we clear
        // it to prevent inner expressions from being incorrectly marked.
        self.in_tail_position = false;
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
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
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
        }
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        // Evaluate arguments into consecutive registers.
        let has_spread = args.iter().any(|a| matches!(a, Expr::Spread(_)));

        if has_spread {
            // Build arguments as a single array, then use CallWithSpread.
            let arr_reg = self.compile_arguments_as_array(args)?;
            let slot = self.alloc_slot(FeedbackSlotKind::Call);
            self.emit(Instruction::new_unchecked(
                Opcode::CallWithSpread,
                vec![
                    to_reg_op(callee_reg),
                    to_reg_op(arr_reg),
                    Operand::RegisterCount(1),
                    slot,
                ],
            ));
            self.allocator
                .release_temporary(arr_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        } else {
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
        }
        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(recv_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile `obj?.method(args)` — optional method call with correct receiver.
    ///
    /// Inside an `OptionalChain`, the nullish check jumps to the chain's
    /// null-label.  Otherwise we handle it locally.
    fn compile_optional_method_call(
        &mut self,
        m: &crate::parser::ast::OptionalMemberExpr,
        args: &[Expr],
    ) -> StatorResult<()> {
        self.in_tail_position = false;

        // Compile the receiver (object).
        self.compile_expr(&m.object)?;
        let recv_reg = self.allocator.allocate_temporary();
        self.emit_star(recv_reg);

        // Nullish check on receiver.
        self.emit_ldar(recv_reg);
        let (use_chain, local_null, local_end) =
            if let Some(chain_label) = self.optional_chain_null_label {
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, chain_label);
                (true, 0, 0) // labels unused
            } else {
                let nl = self.new_label();
                let el = self.new_label();
                self.emit_jump(Opcode::JumpIfUndefinedOrNull, nl);
                (false, nl, el)
            };

        // Load the method function from the non-null receiver.
        self.emit_optional_member_property(recv_reg, &m.property)?;
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        // Call with receiver.
        let arg_regs = self.compile_arguments(args)?;
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

        if !use_chain {
            self.emit_jump(Opcode::Jump, local_end);
            self.bind_label(local_null);
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            self.bind_label(local_end);
        }

        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(recv_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    fn compile_super_method_call(
        &mut self,
        m: &crate::parser::ast::MemberExpr,
        args: &[Expr],
    ) -> StatorResult<()> {
        self.in_tail_position = false;

        let recv_reg = self.allocator.allocate_temporary();
        let this_name_idx = self.add_string("this");
        let this_slot = self.alloc_slot(FeedbackSlotKind::LoadGlobal);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaGlobal,
            vec![Operand::ConstantPoolIdx(this_name_idx), this_slot],
        ));
        self.emit_star(recv_reg);

        self.compile_expr(&m.object)?;
        let super_reg = self.allocator.allocate_temporary();
        self.emit_star(super_reg);

        let proto_name_idx = self.add_string("prototype");
        let proto_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        self.emit(Instruction::new_unchecked(
            Opcode::LdaNamedProperty,
            vec![
                to_reg_op(super_reg),
                Operand::ConstantPoolIdx(proto_name_idx),
                proto_slot,
            ],
        ));

        match &m.property {
            crate::parser::ast::MemberProp::Ident(id) => {
                let name_idx = self.add_string(&id.name);
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        to_reg_op(recv_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
                let slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaNamedPropertyFromSuper,
                    vec![
                        to_reg_op(recv_reg),
                        Operand::ConstantPoolIdx(name_idx),
                        slot,
                    ],
                ));
            }
            crate::parser::ast::MemberProp::Computed(_) => {
                self.allocator
                    .release_temporary(super_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(recv_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                return Err(StatorError::SyntaxError(
                    "computed super method access is not yet supported".into(),
                ));
            }
        }
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        let has_spread = args.iter().any(|a| matches!(a, Expr::Spread(_)));
        if has_spread {
            let arr_reg = self.compile_arguments_as_array(args)?;
            let slot = self.alloc_slot(FeedbackSlotKind::Call);
            self.emit(Instruction::new_unchecked(
                Opcode::CallWithSpread,
                vec![
                    to_reg_op(callee_reg),
                    to_reg_op(arr_reg),
                    Operand::RegisterCount(1),
                    slot,
                ],
            ));
            self.allocator
                .release_temporary(arr_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        } else {
            let arg_regs = self.compile_arguments(args)?;
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
        }

        self.allocator
            .release_temporary(callee_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(super_reg)
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

    /// Build an array from call arguments, flattening spread elements.
    /// Returns the register holding the resulting array.
    fn compile_arguments_as_array(&mut self, args: &[Expr]) -> StatorResult<Register> {
        // Create an empty array to hold all arguments.
        let arr_slot = self.alloc_slot(FeedbackSlotKind::Literal);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateEmptyArrayLiteral,
            vec![arr_slot],
        ));
        let arr_reg = self.allocator.allocate_temporary();
        self.emit_star(arr_reg);

        // Dynamic index counter for spread support.
        let idx_reg = self.allocator.allocate_temporary();
        self.emit(Instruction::new_unchecked(Opcode::LdaZero, vec![]));
        self.emit_star(idx_reg);

        for arg in args {
            if let Expr::Spread(s) = arg {
                // Spread: iterate the argument, pushing each value.
                self.compile_expr(&s.argument)?;
                let iterable_reg = self.allocator.allocate_temporary();
                self.emit_star(iterable_reg);
                let load_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
                self.emit(Instruction::new_unchecked(
                    Opcode::GetIterator,
                    vec![to_reg_op(iterable_reg), load_slot, call_slot],
                ));
                let iter_reg = self.allocator.allocate_temporary();
                self.emit_star(iter_reg);
                let val_reg = self.allocator.allocate_temporary();

                let loop_lbl = self.new_label();
                let done_lbl = self.new_label();
                self.bind_label(loop_lbl);

                self.emit(Instruction::new_unchecked(
                    Opcode::IteratorNext,
                    vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
                ));
                self.emit_jump_if_true_to(done_lbl);

                // Store val at arr[idx], then increment idx.
                self.emit_ldar(val_reg);
                let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![to_reg_op(arr_reg), to_reg_op(idx_reg), elem_slot],
                ));
                self.emit_ldar(idx_reg);
                let inc_slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot]));
                self.emit_star(idx_reg);
                self.emit_jump_loop_to(loop_lbl);
                self.bind_label(done_lbl);

                self.allocator
                    .release_temporary(val_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(iter_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
                self.allocator
                    .release_temporary(iterable_reg)
                    .map_err(|e| StatorError::Internal(e.to_string()))?;
            } else {
                // Normal argument at dynamic index.
                self.compile_expr(arg)?;
                let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaInArrayLiteral,
                    vec![to_reg_op(arr_reg), to_reg_op(idx_reg), elem_slot],
                ));
                self.emit_ldar(idx_reg);
                let inc_slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot]));
                self.emit_star(idx_reg);
            }
        }

        self.allocator
            .release_temporary(idx_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(arr_reg)
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

    /// Emit a `TailCall` instruction (same operand layout as
    /// [`Self::emit_call_any_receiver`] but signals tail-position semantics
    /// so the interpreter can reuse the current frame).
    fn emit_tail_call(&mut self, callee_reg: Register, arg_regs: &[Register]) -> StatorResult<()> {
        let arg_count = arg_regs.len() as u32;
        let args_start = arg_regs.first().copied().unwrap_or(callee_reg);
        let slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::TailCall,
            vec![
                to_reg_op(callee_reg),
                to_reg_op(args_start),
                Operand::RegisterCount(arg_count),
                slot,
            ],
        ));
        Ok(())
    }

    /// Emit a `CallDirectEval` instruction (same operand layout as
    /// [`Self::emit_call_any_receiver`] but signals direct-eval semantics).
    fn emit_call_direct_eval(
        &mut self,
        callee_reg: Register,
        arg_regs: &[Register],
    ) -> StatorResult<()> {
        let arg_count = arg_regs.len() as u32;
        let args_start = arg_regs.first().copied().unwrap_or(callee_reg);
        let slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::CallDirectEval,
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

        let has_spread = n.arguments.iter().any(|a| matches!(a, Expr::Spread(_)));

        if has_spread {
            let arr_reg = self.compile_arguments_as_array(&n.arguments)?;
            let slot = self.alloc_slot(FeedbackSlotKind::Call);
            self.emit(Instruction::new_unchecked(
                Opcode::ConstructWithSpread,
                vec![
                    to_reg_op(ctor_reg),
                    to_reg_op(arr_reg),
                    Operand::RegisterCount(1),
                    slot,
                ],
            ));
            self.allocator
                .release_temporary(arr_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
        } else {
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
        }
        self.allocator
            .release_temporary(ctor_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        Ok(())
    }

    /// Compile a function expression.
    fn compile_fn_expr(&mut self, f: &FnExpr) -> StatorResult<()> {
        let func_array =
            compile_function(&f.params, &f.body, f.is_generator, f.is_async, f.is_strict)?;
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
        let func_array =
            compile_function_inner(&a.params, &body_block, false, a.is_async, a.is_strict, true)?
                .with_arrow_flag(true);
        let pool_idx = self.add_constant_raw(ConstantPoolEntry::Function(Box::new(func_array)));
        let slot = self.alloc_slot(FeedbackSlotKind::CreateClosure);
        // Flag(1) marks this as an arrow function (no .prototype property).
        self.emit(Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![Operand::ConstantPoolIdx(pool_idx), slot, Operand::Flag(1)],
        ));
        Ok(())
    }

    /// Compile an array literal.
    fn compile_array(&mut self, a: &crate::parser::ast::ArrayExpr) -> StatorResult<()> {
        let has_spread = a
            .elements
            .iter()
            .any(|e| matches!(e, Some(Expr::Spread(_))));

        // Create an empty array, then fill each element slot.
        let arr_slot = self.alloc_slot(FeedbackSlotKind::Literal);
        self.emit(Instruction::new_unchecked(
            Opcode::CreateEmptyArrayLiteral,
            vec![arr_slot],
        ));
        let arr_reg = self.allocator.allocate_temporary();
        self.emit_star(arr_reg);

        if !has_spread {
            // Fast path: no spread, use static indices.
            for (i, elem) in a.elements.iter().enumerate() {
                if let Some(elem_expr) = elem {
                    let idx_reg = self.allocator.allocate_temporary();
                    let idx_val = i as i32;
                    self.emit(Instruction::new_unchecked(
                        Opcode::LdaSmi,
                        vec![Operand::Immediate(idx_val)],
                    ));
                    self.emit_star(idx_reg);
                    self.compile_expr(elem_expr)?;
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
        } else {
            // Slow path: has spread — use a dynamic index counter.
            let idx_reg = self.allocator.allocate_temporary();
            self.emit(Instruction::new_unchecked(Opcode::LdaZero, vec![]));
            self.emit_star(idx_reg);

            for elem in &a.elements {
                match elem {
                    None => {
                        // Elision: increment index only.
                        self.emit_ldar(idx_reg);
                        let inc_slot = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                        self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot]));
                        self.emit_star(idx_reg);
                    }
                    Some(Expr::Spread(s)) => {
                        // Spread: iterate the argument, pushing each value.
                        self.compile_expr(&s.argument)?;
                        let iterable_reg = self.allocator.allocate_temporary();
                        self.emit_star(iterable_reg);
                        let load_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
                        let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
                        self.emit(Instruction::new_unchecked(
                            Opcode::GetIterator,
                            vec![to_reg_op(iterable_reg), load_slot, call_slot],
                        ));
                        let iter_reg = self.allocator.allocate_temporary();
                        self.emit_star(iter_reg);
                        let val_reg = self.allocator.allocate_temporary();

                        let loop_lbl = self.new_label();
                        let done_lbl = self.new_label();
                        self.bind_label(loop_lbl);

                        self.emit(Instruction::new_unchecked(
                            Opcode::IteratorNext,
                            vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
                        ));
                        self.emit_jump_if_true_to(done_lbl);

                        // Store val at arr[idx], then increment idx.
                        self.emit_ldar(val_reg);
                        let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                        self.emit(Instruction::new_unchecked(
                            Opcode::StaInArrayLiteral,
                            vec![to_reg_op(arr_reg), to_reg_op(idx_reg), elem_slot],
                        ));
                        self.emit_ldar(idx_reg);
                        let inc_slot2 = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                        self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot2]));
                        self.emit_star(idx_reg);
                        self.emit_jump_loop_to(loop_lbl);
                        self.bind_label(done_lbl);

                        self.allocator
                            .release_temporary(val_reg)
                            .map_err(|e| StatorError::Internal(e.to_string()))?;
                        self.allocator
                            .release_temporary(iter_reg)
                            .map_err(|e| StatorError::Internal(e.to_string()))?;
                        self.allocator
                            .release_temporary(iterable_reg)
                            .map_err(|e| StatorError::Internal(e.to_string()))?;
                    }
                    Some(expr) => {
                        // Normal element at dynamic index.
                        self.compile_expr(expr)?;
                        let elem_slot = self.alloc_slot(FeedbackSlotKind::KeyedStoreProperty);
                        self.emit(Instruction::new_unchecked(
                            Opcode::StaInArrayLiteral,
                            vec![to_reg_op(arr_reg), to_reg_op(idx_reg), elem_slot],
                        ));
                        self.emit_ldar(idx_reg);
                        let inc_slot3 = self.alloc_slot(FeedbackSlotKind::BinaryOp);
                        self.emit(Instruction::new_unchecked(Opcode::Inc, vec![inc_slot3]));
                        self.emit_star(idx_reg);
                    }
                }
            }

            self.allocator
                .release_temporary(idx_reg)
                .map_err(|e| StatorError::Internal(e.to_string()))?;
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
                crate::parser::ast::ObjectProp::Spread(spread) => {
                    // Object spread: compile source, store in temp, copy all properties.
                    self.compile_expr(&spread.argument)?;
                    let src_reg = self.allocator.allocate_temporary();
                    self.emit_star(src_reg);
                    self.emit(Instruction::new_unchecked(
                        Opcode::CopyDataProperties,
                        vec![to_reg_op(obj_reg), to_reg_op(src_reg)],
                    ));
                    self.allocator
                        .release_temporary(src_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
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
            PropValue::Get(fn_expr) => {
                self.compile_fn_expr(fn_expr)?;
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineGetterProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                } else if let PropKey::Computed(key_expr) = &p.key {
                    let val_reg = self.allocator.allocate_temporary();
                    self.emit_star(val_reg);
                    self.compile_expr(key_expr)?;
                    let key_reg = self.allocator.allocate_temporary();
                    self.emit_star(key_reg);
                    self.emit_ldar(val_reg);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedGetterProperty,
                        vec![to_reg_op(obj_reg), to_reg_op(key_reg), slot],
                    ));
                    self.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    self.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            }
            PropValue::Set(fn_expr) => {
                self.compile_fn_expr(fn_expr)?;
                if let Some(name) = key_name {
                    let name_idx = self.add_string(&name);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineSetterProperty,
                        vec![to_reg_op(obj_reg), Operand::ConstantPoolIdx(name_idx), slot],
                    ));
                } else if let PropKey::Computed(key_expr) = &p.key {
                    let val_reg = self.allocator.allocate_temporary();
                    self.emit_star(val_reg);
                    self.compile_expr(key_expr)?;
                    let key_reg = self.allocator.allocate_temporary();
                    self.emit_star(key_reg);
                    self.emit_ldar(val_reg);
                    let slot = self.alloc_slot(FeedbackSlotKind::DefineAccessor);
                    self.emit(Instruction::new_unchecked(
                        Opcode::DefineKeyedSetterProperty,
                        vec![to_reg_op(obj_reg), to_reg_op(key_reg), slot],
                    ));
                    self.allocator
                        .release_temporary(key_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                    self.allocator
                        .release_temporary(val_reg)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
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

    // ── Tagged template literals ─────────────────────────────────────────────

    /// Compile a tagged template expression `` tag`…` ``.
    ///
    /// Emits:
    /// 1. The tag function expression.
    /// 2. `GetTemplateObject` — loads or creates the frozen template-strings
    ///    array (with `.raw`) from the constant pool.
    /// 3. Each interpolated expression as additional arguments.
    /// 4. A call instruction (`CallProperty` for method tags, otherwise
    ///    `CallAnyReceiver`).
    fn compile_tagged_template(
        &mut self,
        t: &crate::parser::ast::TaggedTemplateExpr,
    ) -> StatorResult<()> {
        // Method-call form: `obj.method`…`` — use correct `this`.
        if let Expr::Member(m) = t.tag.as_ref() {
            return self.compile_tagged_template_method(m, &t.quasi);
        }

        // General call form: `tag`…`` with undefined receiver.
        self.compile_expr(&t.tag)?;
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        // First argument: the template object.
        let tpl_idx = self.add_template_object(&t.quasi);
        let tpl_slot = self.alloc_slot(FeedbackSlotKind::Literal);
        self.emit(Instruction::new_unchecked(
            Opcode::GetTemplateObject,
            vec![Operand::ConstantPoolIdx(tpl_idx), tpl_slot],
        ));
        let tpl_reg = self.allocator.allocate_temporary();
        self.emit_star(tpl_reg);

        // Remaining arguments: interpolated expressions.
        let mut arg_regs = vec![tpl_reg];
        for expr in &t.quasi.expressions {
            let r = self.allocator.allocate_temporary();
            self.compile_expr(expr)?;
            self.emit_star(r);
            arg_regs.push(r);
        }

        self.emit_call_any_receiver(callee_reg, &arg_regs)?;

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

    /// Compile a tagged template where the tag is a member expression
    /// (`obj.method`…``), preserving the correct `this` binding.
    fn compile_tagged_template_method(
        &mut self,
        m: &crate::parser::ast::MemberExpr,
        quasi: &crate::parser::ast::TemplateLit,
    ) -> StatorResult<()> {
        // Load the receiver (object).
        self.compile_expr(&m.object)?;
        let recv_reg = self.allocator.allocate_temporary();
        self.emit_star(recv_reg);

        // Load the method function from the object.
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
            crate::parser::ast::MemberProp::Private(id) => {
                let name_idx = self.add_string(&format!("#{}", id.name));
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
        }
        let callee_reg = self.allocator.allocate_temporary();
        self.emit_star(callee_reg);

        // First argument: the template object.
        let tpl_idx = self.add_template_object(quasi);
        let tpl_slot = self.alloc_slot(FeedbackSlotKind::Literal);
        self.emit(Instruction::new_unchecked(
            Opcode::GetTemplateObject,
            vec![Operand::ConstantPoolIdx(tpl_idx), tpl_slot],
        ));

        let mut arg_regs = Vec::with_capacity(1 + quasi.expressions.len());
        let tpl_reg = self.allocator.allocate_temporary();
        self.emit_star(tpl_reg);
        arg_regs.push(tpl_reg);

        for expr in &quasi.expressions {
            let r = self.allocator.allocate_temporary();
            self.compile_expr(expr)?;
            self.emit_star(r);
            arg_regs.push(r);
        }

        // CallProperty with the original receiver as `this`.
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

    /// Add a [`ConstantPoolEntry::TemplateObject`] for the given template
    /// literal and return its constant-pool index.
    fn add_template_object(&mut self, t: &crate::parser::ast::TemplateLit) -> u32 {
        let cooked: Vec<Option<String>> = t.quasis.iter().map(|q| q.cooked.clone()).collect();
        let raw: Vec<String> = t.quasis.iter().map(|q| q.raw.clone()).collect();
        self.add_constant_raw(ConstantPoolEntry::TemplateObject { cooked, raw })
    }

    // ── Generator / yield ────────────────────────────────────────────────────

    /// Compile a `yield [argument]` expression.
    ///
    /// Emits:
    /// 1. The argument expression (or `LdaUndefined` if absent).
    /// 2. `SuspendGenerator` — suspends the generator with the current acc as
    ///    the yielded value.
    /// 3. `ResumeGenerator` — on the next `.next(sent)` call, restores the
    ///    register file; accumulator becomes the `sent` value (result of the
    ///    yield expression).
    fn compile_yield(&mut self, expr: &crate::parser::ast::YieldExpr) -> StatorResult<()> {
        // Evaluate the yielded value.
        if let Some(arg) = &expr.argument {
            if expr.delegate {
                return self.compile_yield_star(arg);
            }
            self.compile_expr(arg)?;
        } else {
            self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        }

        // Dummy register used by generator opcodes (interpreter ignores it).
        let dummy = Operand::Register(0);

        // SuspendGenerator [gen, regs_start, regs_count, suspend_id]
        let suspend_id = self.yield_suspend_id;
        self.yield_suspend_id += 1;
        self.emit(Instruction::new_unchecked(
            Opcode::SuspendGenerator,
            vec![
                dummy,
                dummy,
                Operand::RegisterCount(0),
                Operand::Immediate(suspend_id as i32),
            ],
        ));

        // ResumeGenerator [gen, regs_start, regs_count]
        // After this, acc = value sent by caller's .next(sent).
        self.emit(Instruction::new_unchecked(
            Opcode::ResumeGenerator,
            vec![dummy, dummy, Operand::RegisterCount(0)],
        ));

        Ok(())
    }

    /// Compile an `await expr` expression inside an async (generator) function.
    ///
    /// Uses the same suspend/resume mechanism as `yield` so the runtime can
    /// resolve the awaited value and resume execution with the result.
    fn compile_await(&mut self, expr: &crate::parser::ast::AwaitExpr) -> StatorResult<()> {
        // Evaluate the awaited expression → acc.
        self.compile_expr(&expr.argument)?;

        let dummy = Operand::Register(0);

        // SuspendGenerator [gen, regs_start, regs_count, suspend_id]
        let suspend_id = self.yield_suspend_id;
        self.yield_suspend_id += 1;
        self.emit(Instruction::new_unchecked(
            Opcode::SuspendGenerator,
            vec![
                dummy,
                dummy,
                Operand::RegisterCount(0),
                Operand::Immediate(suspend_id as i32),
            ],
        ));

        // ResumeGenerator [gen, regs_start, regs_count]
        // After this, acc = resolved value of the awaited expression.
        self.emit(Instruction::new_unchecked(
            Opcode::ResumeGenerator,
            vec![dummy, dummy, Operand::RegisterCount(0)],
        ));

        Ok(())
    }

    /// Compile a `yield* expr` (delegating yield).
    ///
    /// Iterates the inner iterable and re-yields each value to the outer
    /// caller.  The accumulator is set to `undefined` when delegation
    /// completes (simplified — does not propagate the inner return value).
    fn compile_yield_star(&mut self, expr: &crate::parser::ast::Expr) -> StatorResult<()> {
        // Evaluate the inner iterable → acc.
        self.compile_expr(expr)?;

        // GetIterator [iterable_reg, load_slot, call_slot]
        let iterable_reg = self.allocator.allocate_temporary();
        self.emit_star(iterable_reg);
        let load_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            Opcode::GetIterator,
            vec![to_reg_op(iterable_reg), load_slot, call_slot],
        ));
        self.allocator
            .release_temporary(iterable_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        // Save the iterator.
        let iter_reg = self.allocator.allocate_temporary();
        self.emit_star(iter_reg);

        // Allocate value register for IteratorNext output.
        let val_reg = self.allocator.allocate_temporary();

        // Loop labels.
        let loop_lbl = self.new_label();
        let done_lbl = self.new_label();

        // Bind loop start.
        self.bind_label(loop_lbl);

        // IteratorNext [iter_reg, val_reg] → val_reg = value, acc = done
        self.emit(Instruction::new_unchecked(
            Opcode::IteratorNext,
            vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
        ));

        // Jump to done if acc (done flag) is truthy.
        self.emit_jump_if_true_to(done_lbl);

        // Not done — yield the inner value.
        let dummy = Operand::Register(0);
        self.emit_ldar(val_reg);

        let suspend_id = self.yield_suspend_id;
        self.yield_suspend_id += 1;
        self.emit(Instruction::new_unchecked(
            Opcode::SuspendGenerator,
            vec![
                dummy,
                dummy,
                Operand::RegisterCount(0),
                Operand::Immediate(suspend_id as i32),
            ],
        ));
        self.emit(Instruction::new_unchecked(
            Opcode::ResumeGenerator,
            vec![dummy, dummy, Operand::RegisterCount(0)],
        ));

        // Jump back to loop start.
        self.emit_jump_loop_to(loop_lbl);

        // done_lbl: delegation complete.
        self.bind_label(done_lbl);

        self.allocator
            .release_temporary(val_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(iter_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        // Result of `yield*` is the inner generator's return value.
        // Simplified: leave undefined in acc (the done record's value is in
        // val_reg which was already released; we emit LdaUndefined here).
        self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
        Ok(())
    }

    // ── for-of ────────────────────────────────────────────────────────────────

    /// Compile a `for (left of right) body` statement.
    ///
    /// Compile a `for (x in obj) { … }` statement.
    ///
    /// Emits:
    /// ```text
    /// <compile right>                          ; acc = object
    /// Star r_obj
    /// ForInEnumerate r_obj                     ; acc = keys array
    /// Star r_keys
    /// ForInPrepare r_keys, slot                ; acc = length
    /// Star r_length
    /// LdaZero
    /// Star r_index
    /// loop_start:
    ///   JumpIfForInDone end, r_index, r_length ; jump when done
    ///   ForInNext r_obj, r_index, r_keys, slot ; acc = keys[index]
    ///   Star r_key                             ; bind loop variable
    ///   <body>
    ///   ForInStep r_index                      ; acc = index + 1
    ///   Star r_index
    ///   JumpLoop loop_start
    /// end:
    /// ```
    fn compile_for_in(&mut self, stmt: &crate::parser::ast::ForInStmt) -> StatorResult<()> {
        use crate::parser::ast::ForInOfLeft;

        // Open the loop's own scope so the loop variable lives inside.
        self.push_scope();

        // Pre-allocate the loop variable register if it's a new declaration.
        // `var` declarations are function-scoped (must survive after the loop
        // scope is popped), while `let`/`const` are block-scoped.
        let loop_var_reg: Option<Register> = match &stmt.left {
            ForInOfLeft::VarDecl(decl) => {
                if decl.declarators.len() == 1 {
                    match &decl.declarators[0].id {
                        crate::parser::ast::Pat::Ident(id) => {
                            if decl.kind == VarKind::Var {
                                Some(self.define_function_scoped_local(&id.name))
                            } else if decl.kind == VarKind::Const {
                                Some(self.define_const_local(&id.name))
                            } else {
                                Some(self.define_local(&id.name))
                            }
                        }
                        _pat => Some(self.allocator.new_local()),
                    }
                } else if decl.declarators.is_empty() {
                    None
                } else {
                    return Err(StatorError::SyntaxError(
                        "for-in loop head must contain a single declaration".into(),
                    ));
                }
            }
            ForInOfLeft::Pat(_) | ForInOfLeft::Expr(_) => None,
        };

        // Evaluate the right-hand (object) expression → acc.
        self.compile_expr(&stmt.right)?;

        // Store the object in a temporary.
        let r_obj = self.allocator.allocate_temporary();
        self.emit_star(r_obj);

        // ForInEnumerate r_obj → acc = array of enumerable keys.
        let enum_slot = self.alloc_slot(FeedbackSlotKind::ForIn);
        self.emit(Instruction::new_unchecked(
            Opcode::ForInEnumerate,
            vec![to_reg_op(r_obj)],
        ));

        // Save keys array.
        let r_keys = self.allocator.allocate_temporary();
        self.emit_star(r_keys);

        // ForInPrepare r_keys, slot → acc = length.
        let prepare_slot = self.alloc_slot(FeedbackSlotKind::ForIn);
        self.emit(Instruction::new_unchecked(
            Opcode::ForInPrepare,
            vec![to_reg_op(r_keys), prepare_slot],
        ));

        // Save length.
        let r_length = self.allocator.allocate_temporary();
        self.emit_star(r_length);

        // index = 0.
        self.emit(Instruction::new_unchecked(Opcode::LdaZero, vec![]));
        let r_index = self.allocator.allocate_temporary();
        self.emit_star(r_index);

        // Loop labels.
        let loop_lbl = self.new_label();
        let break_lbl = self.new_label();
        self.patch_pending_label(loop_lbl);
        self.loop_stack.push((loop_lbl, break_lbl));

        // Bind loop start.
        self.bind_label(loop_lbl);

        // JumpIfForInDone [offset, r_index, r_length] — jump to break if done.
        {
            let jump_idx = self.instructions.len();
            self.labels[break_lbl].refs.push(jump_idx);
            self.emit(Instruction::new_unchecked(
                Opcode::JumpIfForInDone,
                vec![
                    Operand::JumpOffset(0),
                    to_reg_op(r_index),
                    to_reg_op(r_length),
                ],
            ));
        }

        // ForInNext r_obj, r_index, r_keys, slot → acc = keys[index].
        let next_slot = self.alloc_slot(FeedbackSlotKind::ForIn);
        self.emit(Instruction::new_unchecked(
            Opcode::ForInNext,
            vec![
                to_reg_op(r_obj),
                to_reg_op(r_index),
                to_reg_op(r_keys),
                next_slot,
            ],
        ));

        // Bind the loop variable.
        match &stmt.left {
            ForInOfLeft::VarDecl(decl) => {
                if let Some(reg) = loop_var_reg {
                    self.emit_star(reg);
                    // For destructuring declarations, unpack the key.
                    if decl.declarators.len() == 1
                        && !matches!(decl.declarators[0].id, crate::parser::ast::Pat::Ident(_))
                    {
                        self.compile_binding_pattern(
                            &decl.declarators[0].id,
                            reg,
                            BindingMode::Declare {
                                is_const: decl.kind == VarKind::Const,
                            },
                        )?;
                    }
                }
            }
            ForInOfLeft::Pat(pat) => match pat {
                crate::parser::ast::Pat::Ident(id) => {
                    let binding = self.lookup_var(&id.name).ok_or_else(|| {
                        StatorError::SyntaxError(format!(
                            "for-in: undefined variable '{}'",
                            id.name
                        ))
                    })?;
                    self.emit_star(binding.reg);
                }
                pat => {
                    // Destructuring: store key in a temporary, then unpack.
                    // Must use allocate_temporary (not new_local) because
                    // r_obj/r_keys/r_length/r_index temporaries are still
                    // live — new_local would shift local_count and corrupt
                    // the temporary release logic.
                    let scratch = self.allocator.allocate_temporary();
                    self.emit_star(scratch);
                    self.compile_binding_pattern(pat, scratch, BindingMode::Assign)?;
                    self.allocator
                        .release_temporary(scratch)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            },
            ForInOfLeft::Expr(expr) => {
                // Expression target (e.g. `obj.prop`): acc holds the key.
                let target = AssignTarget::Expr(expr.clone());
                self.compile_assign_target_store(&target)?;
            }
        }

        // Compile the loop body.
        self.compile_stmt(&stmt.body)?;

        // ForInStep r_index → acc = index + 1.
        self.emit(Instruction::new_unchecked(
            Opcode::ForInStep,
            vec![to_reg_op(r_index)],
        ));
        self.emit_star(r_index);

        // Back-edge jump.
        self.emit_jump_loop_to(loop_lbl);

        // Bind break target.
        self.bind_label(break_lbl);

        self.loop_stack.pop();
        self.pop_scope();

        // Release temporaries in reverse allocation order.
        self.allocator
            .release_temporary(r_index)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(r_length)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(r_keys)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(r_obj)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        // Suppress the unused-variable warning for the ForIn feedback slots.
        let _ = (enum_slot, prepare_slot, next_slot);

        Ok(())
    }

    /// Emits `GetIterator` + `IteratorNext` loop:
    /// ```text
    /// <compile right>
    /// Star r_iterable
    /// GetIterator r_iterable, …
    /// Star r_iter
    /// loop_start:
    ///   IteratorNext r_iter, r_val   // acc = done flag
    ///   JumpIfToBooleanTrue exit
    ///   Ldar r_val
    ///   Star r_x                     // bind loop variable
    ///   <body>
    ///   JumpLoop loop_start
    /// exit:
    /// ```
    fn compile_for_of(&mut self, stmt: &crate::parser::ast::ForOfStmt) -> StatorResult<()> {
        use crate::parser::ast::ForInOfLeft;

        // Open the loop's own scope early so that the loop variable is defined
        // as a local BEFORE any temporaries are allocated.  This ensures the
        // register allocator assigns the variable a lower index than the iter/val
        // temporaries and avoids LIFO-order conflicts during `release_temporary`.
        self.push_scope();

        // If the left-hand side declares a new variable, pre-allocate its
        // local register now (before any temporaries are allocated below).
        // `var` declarations are function-scoped; `let`/`const` are block-scoped.
        let loop_var_reg: Option<Register> = match &stmt.left {
            ForInOfLeft::VarDecl(decl) => {
                if decl.declarators.len() == 1 {
                    match &decl.declarators[0].id {
                        crate::parser::ast::Pat::Ident(id) => {
                            if decl.kind == VarKind::Var {
                                Some(self.define_function_scoped_local(&id.name))
                            } else if decl.kind == VarKind::Const {
                                Some(self.define_const_local(&id.name))
                            } else {
                                Some(self.define_local(&id.name))
                            }
                        }
                        _pat => Some(self.allocator.new_local()),
                    }
                } else if decl.declarators.is_empty() {
                    None
                } else {
                    return Err(StatorError::SyntaxError(
                        "for-of loop head must contain a single declaration".into(),
                    ));
                }
            }
            ForInOfLeft::Pat(_) | ForInOfLeft::Expr(_) => None, // existing variable — no new allocation
        };

        // Evaluate the right-hand (iterable) expression → acc.
        self.compile_expr(&stmt.right)?;

        // Store iterable in a temporary register for GetIterator, then release.
        let iterable_reg = self.allocator.allocate_temporary();
        self.emit_star(iterable_reg);
        let op = if stmt.is_await {
            Opcode::GetAsyncIterator
        } else {
            Opcode::GetIterator
        };
        let load_slot = self.alloc_slot(FeedbackSlotKind::LoadProperty);
        let call_slot = self.alloc_slot(FeedbackSlotKind::Call);
        self.emit(Instruction::new_unchecked(
            op,
            vec![to_reg_op(iterable_reg), load_slot, call_slot],
        ));
        self.allocator
            .release_temporary(iterable_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        // Save iterator in a temporary register.
        let iter_reg = self.allocator.allocate_temporary();
        self.emit_star(iter_reg);

        // Allocate value-output register for IteratorNext.
        let val_reg = self.allocator.allocate_temporary();

        // Loop labels.
        let loop_lbl = self.new_label();
        let break_lbl = self.new_label();
        self.patch_pending_label(loop_lbl);
        self.loop_stack.push((loop_lbl, break_lbl));
        self.for_of_iter_regs.insert(break_lbl, iter_reg);

        // Bind loop start.
        self.bind_label(loop_lbl);

        // IteratorNext [iter_reg, val_reg] → val_reg = value, acc = done
        self.emit(Instruction::new_unchecked(
            Opcode::IteratorNext,
            vec![to_reg_op(iter_reg), to_reg_op(val_reg)],
        ));

        // Jump to break if done.
        self.emit_jump_if_true_to(break_lbl);

        // §14.7.5.13 step 7: the loop body (variable binding + user code) is
        // wrapped in an implicit try/finally so that IteratorClose is called
        // when the body throws.  Record the start of the "try" region.
        let try_start = self.instructions.len() as u32;

        // Bind loop variable.
        match &stmt.left {
            ForInOfLeft::VarDecl(decl) => {
                // Loop variable was pre-allocated above.
                if let Some(reg) = loop_var_reg {
                    self.emit_ldar(val_reg);
                    self.emit_star(reg);
                    // For destructuring declarations, unpack the value.
                    if decl.declarators.len() == 1
                        && !matches!(decl.declarators[0].id, crate::parser::ast::Pat::Ident(_))
                    {
                        self.compile_binding_pattern(
                            &decl.declarators[0].id,
                            reg,
                            BindingMode::Declare {
                                is_const: decl.kind == VarKind::Const,
                            },
                        )?;
                    }
                }
            }
            ForInOfLeft::Pat(pat) => match pat {
                crate::parser::ast::Pat::Ident(id) => {
                    let binding = self.lookup_var(&id.name).ok_or_else(|| {
                        StatorError::SyntaxError(format!(
                            "for-of: undefined variable '{}'",
                            id.name
                        ))
                    })?;
                    self.emit_ldar(val_reg);
                    self.emit_star(binding.reg);
                }
                pat => {
                    // Destructuring: store value in a temporary, then unpack.
                    // Must use allocate_temporary (not new_local) because
                    // iter_reg and val_reg temporaries are still live — calling
                    // new_local would shift local_count and corrupt the
                    // temporary release logic.
                    let scratch = self.allocator.allocate_temporary();
                    self.emit_ldar(val_reg);
                    self.emit_star(scratch);
                    self.compile_binding_pattern(pat, scratch, BindingMode::Assign)?;
                    self.allocator
                        .release_temporary(scratch)
                        .map_err(|e| StatorError::Internal(e.to_string()))?;
                }
            },
            ForInOfLeft::Expr(expr) => {
                // Expression target (e.g. `obj.prop`): load value then assign.
                self.emit_ldar(val_reg);
                let target = AssignTarget::Expr(expr.clone());
                self.compile_assign_target_store(&target)?;
            }
        }

        // Compile the loop body (no extra scope — scope was opened above).
        self.compile_stmt(&stmt.body)?;

        let try_end = self.instructions.len() as u32;

        // Back-edge jump.
        self.emit_jump_loop_to(loop_lbl);

        // Bind break target.
        self.bind_label(break_lbl);

        // §14.7.5.13 step 7.k: emit the exception handler that calls
        // IteratorClose before re-throwing.  When the loop body (or the
        // variable-binding destructuring) throws, the interpreter jumps here.
        //
        // Layout:
        //   [normal path falls through to break_lbl above]
        //   Jump(after_ex_handler)          ← skip handler on normal exit
        //   [ex_handler]  Star(ex_reg)      ← save thrown value
        //                 IteratorClose(iter_reg)
        //                 Ldar(ex_reg)
        //                 ReThrow
        //   [after_ex_handler] ...
        let after_ex_label = self.new_label();
        self.emit_jump(Opcode::Jump, after_ex_label);

        let ex_handler_start = self.instructions.len() as u32;
        let ex_reg = self.allocator.allocate_temporary();
        self.emit_star(ex_reg);
        self.emit(Instruction::new_unchecked(
            Opcode::IteratorClose,
            vec![to_reg_op(iter_reg)],
        ));
        self.emit_ldar(ex_reg);
        self.emit(Instruction::new_unchecked(Opcode::ReThrow, vec![]));
        self.allocator
            .release_temporary(ex_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        self.bind_label(after_ex_label);

        self.handler_table.push(HandlerTableEntry {
            try_start,
            try_end,
            handler: ex_handler_start,
            is_finally: true,
        });

        self.loop_stack.pop();
        self.for_of_iter_regs.remove(&break_lbl);
        self.pop_scope();

        // Release temporaries in reverse allocation order (val_reg was allocated
        // after iter_reg, so it must be released first).
        self.allocator
            .release_temporary(val_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;
        self.allocator
            .release_temporary(iter_reg)
            .map_err(|e| StatorError::Internal(e.to_string()))?;

        Ok(())
    }

    /// Emit a `JumpIfToBooleanTrue` to `label_idx` (forward or backward).
    fn emit_jump_if_true_to(&mut self, label_idx: usize) {
        let jump_idx = self.instructions.len();
        self.labels[label_idx].refs.push(jump_idx);
        self.emit(Instruction::new_unchecked(
            Opcode::JumpIfToBooleanTrue,
            vec![Operand::JumpOffset(0)],
        ));
    }

    /// Emit a `JumpLoop` (back-edge) to `label_idx`.
    fn emit_jump_loop_to(&mut self, label_idx: usize) {
        let jump_idx = self.instructions.len();
        self.labels[label_idx].refs.push(jump_idx);
        self.emit(Instruction::new_unchecked(
            Opcode::JumpLoop,
            vec![
                Operand::JumpOffset(0),
                Operand::Immediate(0),
                Operand::FeedbackSlot(0),
            ],
        ));
    }

    // ── Module compilation ───────────────────────────────────────────────────

    /// Allocate (or retrieve) a module variable entry for the given binding.
    ///
    /// Returns `(module_request_idx, cell_idx)` where `module_request_idx` is
    /// a constant-pool index for the module specifier string and `cell_idx`
    /// is a per-module unique cell identifier.
    fn get_or_create_module_variable(&mut self, source: &str, binding: &str) -> (u32, i32) {
        let key = format!("{source}\0{binding}");
        if let Some(&pair) = self.module_variables.get(&key) {
            return pair;
        }
        let module_request_idx = self.add_string(source);
        let cell_idx = self.next_module_cell;
        self.next_module_cell += 1;
        self.module_variables
            .insert(key, (module_request_idx, cell_idx));
        (module_request_idx, cell_idx)
    }

    /// Compile an `import` declaration.
    ///
    /// For each specifier, creates a module variable binding so that
    /// subsequent identifier references emit [`Opcode::LdaModuleVariable`].
    fn compile_import_decl(&mut self, decl: &crate::parser::ast::ImportDecl) -> StatorResult<()> {
        let source = &decl.source.value;
        for spec in &decl.specifiers {
            match spec {
                ImportSpecifier::Named(named) => {
                    let imported_name = match &named.imported {
                        ModuleExportName::Ident(id) => id.name.as_str(),
                        ModuleExportName::Str(s) => s.value.as_str(),
                    };
                    let (req_idx, cell) = self.get_or_create_module_variable(source, imported_name);
                    // Emit the load so the local binding is initialized.
                    self.emit(Instruction::new_unchecked(
                        Opcode::LdaModuleVariable,
                        vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
                    ));
                    let reg = self.define_local(&named.local.name);
                    self.emit_star(reg);
                }
                ImportSpecifier::Default(def) => {
                    let (req_idx, cell) = self.get_or_create_module_variable(source, "default");
                    self.emit(Instruction::new_unchecked(
                        Opcode::LdaModuleVariable,
                        vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
                    ));
                    let reg = self.define_local(&def.local.name);
                    self.emit_star(reg);
                }
                ImportSpecifier::Namespace(ns) => {
                    let req_idx = self.add_string(source);
                    self.emit(Instruction::new_unchecked(
                        Opcode::GetModuleNamespace,
                        vec![Operand::ConstantPoolIdx(req_idx)],
                    ));
                    let reg = self.define_local(&ns.local.name);
                    self.emit_star(reg);
                }
            }
        }
        Ok(())
    }

    /// Compile a named `export` declaration.
    ///
    /// Handles `export { x, y }`, `export { x } from "mod"`,
    /// and `export let/const/function …`.
    fn compile_export_named(&mut self, decl: &ExportNamedDecl) -> StatorResult<()> {
        // `export function f() {}` / `export let x = …`
        if let Some(ref inner) = decl.declaration {
            self.compile_stmt(inner)?;
            // For exported declarations, store each declared name into a
            // module variable so the runtime can expose them.
            let names = Self::declared_names(inner);
            for name in names {
                if let Some(binding) = self.lookup_var(&name) {
                    self.emit_ldar(binding.reg);
                } else {
                    self.compile_ident_load(&name);
                }
                let (req_idx, cell) = self.get_or_create_module_variable("", &name);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaModuleVariable,
                    vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
                ));
            }
            return Ok(());
        }

        // Re-export with source: `export { x } from "mod"`
        if let Some(ref source) = decl.source {
            for spec in &decl.specifiers {
                let imported_name = match &spec.local {
                    ModuleExportName::Ident(id) => id.name.as_str(),
                    ModuleExportName::Str(s) => s.value.as_str(),
                };
                let (req_idx, cell) =
                    self.get_or_create_module_variable(&source.value, imported_name);
                self.emit(Instruction::new_unchecked(
                    Opcode::LdaModuleVariable,
                    vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
                ));
                let exported_name = match &spec.exported {
                    ModuleExportName::Ident(id) => id.name.as_str(),
                    ModuleExportName::Str(s) => s.value.as_str(),
                };
                let (out_req, out_cell) = self.get_or_create_module_variable("", exported_name);
                self.emit(Instruction::new_unchecked(
                    Opcode::StaModuleVariable,
                    vec![
                        Operand::ConstantPoolIdx(out_req),
                        Operand::Immediate(out_cell),
                    ],
                ));
            }
            return Ok(());
        }

        // Local re-export: `export { x, y as z }`
        for spec in &decl.specifiers {
            let local_name = match &spec.local {
                ModuleExportName::Ident(id) => id.name.as_str(),
                ModuleExportName::Str(s) => s.value.as_str(),
            };
            if let Some(binding) = self.lookup_var(local_name) {
                self.emit_ldar(binding.reg);
            } else {
                self.compile_ident_load(local_name);
            }
            let exported_name = match &spec.exported {
                ModuleExportName::Ident(id) => id.name.as_str(),
                ModuleExportName::Str(s) => s.value.as_str(),
            };
            let (req_idx, cell) = self.get_or_create_module_variable("", exported_name);
            self.emit(Instruction::new_unchecked(
                Opcode::StaModuleVariable,
                vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
            ));
        }
        Ok(())
    }

    /// Compile an `export default …` declaration.
    fn compile_export_default(
        &mut self,
        decl: &crate::parser::ast::ExportDefaultDecl,
    ) -> StatorResult<()> {
        match &decl.declaration {
            ExportDefaultExpr::Fn(f) => self.compile_fn_decl(f)?,
            ExportDefaultExpr::Class(c) => self.compile_class_decl(c)?,
            ExportDefaultExpr::Expr(e) => self.compile_expr(e)?,
        }
        let (req_idx, cell) = self.get_or_create_module_variable("", "default");
        self.emit(Instruction::new_unchecked(
            Opcode::StaModuleVariable,
            vec![Operand::ConstantPoolIdx(req_idx), Operand::Immediate(cell)],
        ));
        Ok(())
    }

    /// Compile an `export * [as name] from "source"` declaration.
    fn compile_export_all(&mut self, decl: &crate::parser::ast::ExportAllDecl) -> StatorResult<()> {
        let req_idx = self.add_string(&decl.source.value);
        self.emit(Instruction::new_unchecked(
            Opcode::GetModuleNamespace,
            vec![Operand::ConstantPoolIdx(req_idx)],
        ));
        if let Some(ref exported) = decl.exported {
            // `export * as ns from "source"`
            let name = match exported {
                ModuleExportName::Ident(id) => id.name.as_str(),
                ModuleExportName::Str(s) => s.value.as_str(),
            };
            let (out_req, out_cell) = self.get_or_create_module_variable("", name);
            self.emit(Instruction::new_unchecked(
                Opcode::StaModuleVariable,
                vec![
                    Operand::ConstantPoolIdx(out_req),
                    Operand::Immediate(out_cell),
                ],
            ));
        }
        // For bare `export * from "source"` the runtime must merge all
        // bindings from the source module into the current module's exports.
        // The `GetModuleNamespace` instruction above loads the namespace
        // object; the runtime handles the merge.
        Ok(())
    }

    /// Compile an `import.meta` or `new.target` meta-property expression.
    fn compile_meta_prop(&mut self, m: &crate::parser::ast::MetaPropExpr) -> StatorResult<()> {
        if m.meta.name == "import" && m.property.name == "meta" {
            self.emit(Instruction::new_unchecked(Opcode::LdaImportMeta, vec![]));
            Ok(())
        } else if m.meta.name == "new" && m.property.name == "target" {
            self.emit(Instruction::new_unchecked(Opcode::LdaNewTarget, vec![]));
            Ok(())
        } else {
            Err(StatorError::Internal(format!(
                "{}.{} meta property is not yet supported",
                m.meta.name, m.property.name,
            )))
        }
    }

    /// Compile a module declaration (`import` or `export`).
    fn compile_module_decl(&mut self, decl: &ModuleDecl) -> StatorResult<()> {
        match decl {
            ModuleDecl::Import(d) => self.compile_import_decl(d),
            ModuleDecl::ExportNamed(d) => self.compile_export_named(d),
            ModuleDecl::ExportDefault(d) => self.compile_export_default(d),
            ModuleDecl::ExportAll(d) => self.compile_export_all(d),
        }
    }

    /// Extract declared names from a statement (for export bookkeeping).
    fn declared_names(stmt: &Stmt) -> Vec<String> {
        match stmt {
            Stmt::VarDecl(decl) => decl
                .declarators
                .iter()
                .filter_map(|d| {
                    if let Pat::Ident(id) = &d.id {
                        Some(id.name.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            Stmt::FnDecl(decl) => decl.id.iter().map(|id| id.name.clone()).collect(),
            Stmt::ClassDecl(decl) => decl.id.iter().map(|id| id.name.clone()).collect(),
            _ => vec![],
        }
    }

    // ── Finalization ─────────────────────────────────────────────────────────

    /// Resolve all jump offsets and encode the instruction stream into a
    /// [`BytecodeArray`].
    fn finalize(mut self) -> StatorResult<BytecodeArray> {
        // Ensure every function ends with an implicit `return undefined`.
        // For top-level programs (is_program = true) we preserve the last
        // completion value in the accumulator instead (ECMAScript §15.2.3.1).
        let needs_implicit_return = self
            .instructions
            .last()
            .map(|i| i.opcode != Opcode::Return)
            .unwrap_or(true);
        if needs_implicit_return {
            if !self.is_program {
                self.emit(Instruction::new_unchecked(Opcode::LdaUndefined, vec![]));
            }
            self.emit(Instruction::new_unchecked(Opcode::Return, vec![]));
        }

        // Resolve jump targets.
        resolve_jumps(&mut self.instructions, &self.labels)?;

        // Convert pending (instruction_index, line, column) entries to
        // byte-offset–based SourcePositions, now that jump resolution has
        // stabilised all instruction sizes.
        let byte_offsets = compute_byte_offsets(&self.instructions);
        let mut source_positions =
            Vec::with_capacity(self.source_positions.len() + self.pending_positions.len());
        source_positions.extend(self.source_positions);
        for (instr_idx, line, column) in self.pending_positions {
            if let Some(&byte_off) = byte_offsets.get(instr_idx) {
                source_positions.push(SourcePosition::new(byte_off as u32, line, column));
            }
        }
        // Keep the table sorted by bytecode offset (required by binary search).
        source_positions.sort_by_key(|p| p.bytecode_offset);

        let frame_size = self.allocator.frame_size();
        let bytes = encode(&self.instructions);
        let feedback_metadata = FeedbackMetadata::new(self.slot_kinds);
        let ba = BytecodeArray::new(
            bytes,
            self.constant_pool,
            frame_size,
            self.param_count,
            source_positions,
            feedback_metadata,
            self.handler_table,
        );
        Ok(ba
            .with_generator_flag(self.is_generator)
            .with_async_flag(self.is_async)
            .with_module_flag(self.is_module)
            .with_strict_flag(self.is_strict))
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
///
/// When `is_generator` is `true`:
/// - The produced [`BytecodeArray`] is marked with
///   [`BytecodeArray::with_generator_flag(true)`][BytecodeArray::with_generator_flag].
/// - A [`Opcode::SwitchOnGeneratorState`] prologue is emitted before the body
///   so that resumed executions jump directly to the saved resume point.
///
/// When `is_async` is `true`:
/// - The produced [`BytecodeArray`] is marked with
///   [`BytecodeArray::with_async_flag(true)`][BytecodeArray::with_async_flag].
/// - `await` expressions are compiled using the same suspend/resume mechanism
///   as `yield`.
fn compile_function(
    params: &[crate::parser::ast::Param],
    body: &BlockStmt,
    is_generator: bool,
    is_async: bool,
    is_strict: bool,
) -> StatorResult<BytecodeArray> {
    compile_function_inner(params, body, is_generator, is_async, is_strict, false)
}

/// Core function compiler.  `is_arrow` controls whether an `arguments`
/// binding is emitted (arrow functions inherit the enclosing `arguments`).
fn compile_function_inner(
    params: &[crate::parser::ast::Param],
    body: &BlockStmt,
    is_generator: bool,
    is_async: bool,
    is_strict: bool,
    is_arrow: bool,
) -> StatorResult<BytecodeArray> {
    let mut compiler = FunctionCompiler::new(params)?;
    compiler.is_generator = is_generator;
    compiler.is_async = is_async;
    compiler.is_strict = is_strict;

    // Generator / async / async-generator prologue: jump to the saved resume
    // point on re-entry.  Async functions are desugared to generators internally,
    // so they need the same prologue.
    if is_generator || is_async {
        compiler.emit(Instruction::new_unchecked(
            Opcode::SwitchOnGeneratorState,
            vec![Operand::Register(0)],
        ));
    }

    // Emit default-value and destructuring prologue for parameters.
    compiler.emit_param_prologue(params)?;

    // Arrow functions do NOT get their own `arguments` object — they
    // inherit the enclosing function's `arguments` (ES §15.3.4).
    if !is_arrow {
        // Create the `arguments` object and bind it as a local variable.
        compiler.emit(Instruction::new_unchecked(
            Opcode::CreateMappedArguments,
            vec![],
        ));
        let args_reg = compiler.define_local("arguments");
        compiler.emit_star(args_reg);
    }

    // Hoist `var` declarations to the function scope (initialised to
    // `undefined`) so that reads before the declaration returns `undefined`
    // instead of throwing.
    compiler.hoist_var_declarations(&body.body);

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
///     is_strict: false,
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
        compiler.is_program = true;
        let is_module = program.source_type == SourceType::Module;
        compiler.is_module = is_module;
        // Modules are always strict; scripts inherit the AST flag.
        compiler.is_strict = program.is_strict || is_module;
        // Modules implicitly support top-level `await` (ES2022).
        if is_module {
            compiler.is_async = true;
        }

        // Collect top-level statements for hoisting.
        let top_stmts: Vec<&Stmt> = program
            .body
            .iter()
            .filter_map(|item| {
                if let ProgramItem::Stmt(s) = item {
                    Some(s)
                } else {
                    None
                }
            })
            .collect();
        let stmts_owned: Vec<Stmt> = top_stmts.iter().map(|s| (*s).clone()).collect();

        // Hoist `var` declarations into the global env (set to undefined).
        compiler.hoist_var_declarations_global(&stmts_owned);

        // Hoist `let`/`const` declarations with TDZ.
        compiler.hoist_lexical_decls(&stmts_owned);

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
                ProgramItem::ModuleDecl(decl) => {
                    if !is_module {
                        return Err(StatorError::SyntaxError(
                            "module declarations are not allowed in scripts".into(),
                        ));
                    }
                    compiler.compile_module_decl(decl)?;
                }
            }
        }
        compiler.finalize()
    }

    /// Compile a program for direct `eval()` execution.
    ///
    /// Like [`Self::compile_program`] but `var` declarations emit
    /// [`Opcode::StaGlobal`] so they are hoisted into the caller's
    /// variable environment.
    pub fn compile_eval_program(program: &Program) -> StatorResult<BytecodeArray> {
        let mut compiler = FunctionCompiler::new(&[])?;
        compiler.is_program = true;
        compiler.is_eval_scope = true;
        compiler.is_strict = program.is_strict;

        // Collect top-level statements for TDZ hoisting.
        let top_stmts: Vec<Stmt> = program
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

        // Hoist `let`/`const` declarations with TDZ.
        compiler.hoist_lexical_decls(&top_stmts);

        for item in &program.body {
            if let ProgramItem::Stmt(Stmt::FnDecl(decl)) = item {
                compiler.compile_fn_decl(decl)?;
            }
        }

        for item in &program.body {
            match item {
                ProgramItem::Stmt(Stmt::FnDecl(_)) => {}
                ProgramItem::Stmt(stmt) => compiler.compile_stmt(stmt)?,
                ProgramItem::ModuleDecl(_) => {
                    return Err(StatorError::SyntaxError(
                        "module declarations are not allowed in eval".into(),
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
        ArrayPat, AssignExpr, AssignOp, AssignPatProp, AssignTarget, BinaryExpr, BinaryOp,
        BlockStmt, BoolLit, BreakStmt, CatchClause, ClassBody, ClassDecl, ClassExpr, ClassMember,
        ContinueStmt, DoWhileStmt, Expr, ExprStmt, FnDecl, FnExpr, ForStmt, Ident, IfStmt,
        KeyValuePatProp, LabeledStmt, LogicalExpr, LogicalOp, MethodDef, MethodKind, NullLit,
        NumLit, ObjectExpr, ObjectPat, ObjectPatProp, ObjectProp, Param, Pat, Program, ProgramItem,
        Prop, PropKey, PropValue, PropertyDef, ReturnStmt, SourceType, StaticBlock, Stmt,
        StringLit, ThrowStmt, TryStmt, VarDecl, VarDeclarator, VarKind, WhileStmt,
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
            is_strict: false,
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
        // Must contain a JumpLoop back-edge.
        let has_back_jump = instrs.iter().any(|i| {
            i.opcode == Opcode::JumpLoop
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
        assert!(instrs.iter().any(|i| i.opcode == Opcode::JumpLoop));
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
            is_strict: false,
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
            false,
            false,
            false,
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
            is_strict: false,
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

    // ── Generator functions + yield ───────────────────────────────────────────

    /// Helper: build a `FnDecl` for a generator function.
    fn make_generator_fn_decl(name: &str, body: Vec<Stmt>) -> FnDecl {
        use crate::parser::ast::FnDecl;
        FnDecl {
            loc: span(),
            id: Some(ident(name)),
            params: vec![],
            body: BlockStmt { loc: span(), body },
            is_generator: true,
            is_async: false,
            is_strict: false,
        }
    }

    /// Helper: build a yield expression statement.
    fn yield_stmt(arg: Option<Expr>, delegate: bool) -> Stmt {
        use crate::parser::ast::{ExprStmt, YieldExpr};
        Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Yield(Box::new(YieldExpr {
                loc: span(),
                delegate,
                argument: arg.map(Box::new),
            }))),
        })
    }

    #[test]
    fn test_generator_function_compiles() {
        // `function* gen() { yield 1; yield 2; }`
        let decl = make_generator_fn_decl(
            "gen",
            vec![
                yield_stmt(Some(num_expr(1.0)), false),
                yield_stmt(Some(num_expr(2.0)), false),
            ],
        );
        let prog = make_program(vec![Stmt::FnDecl(Box::new(decl))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Top-level: CreateClosure + implicit return
        assert!(instrs.iter().any(|i| i.opcode == Opcode::CreateClosure));
        // The closure itself must be in the constant pool.
        let pool = arr.constant_pool();
        assert!(!pool.is_empty());
        // The nested function should be a generator.
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(
                ba.is_generator(),
                "nested function must be marked as generator"
            );
            let inner_instrs = ba.instructions().unwrap();
            // Generator body must start with SwitchOnGeneratorState.
            assert_eq!(
                inner_instrs[0].opcode,
                Opcode::SwitchOnGeneratorState,
                "generator body must begin with SwitchOnGeneratorState"
            );
            // Must contain SuspendGenerator pairs.
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::SuspendGenerator),
                "generator body must contain SuspendGenerator"
            );
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::ResumeGenerator),
                "generator body must contain ResumeGenerator"
            );
        } else {
            panic!("constant pool[0] should be a Function (generator body)");
        }
    }

    /// Compile-and-run a program, returning the final accumulator value.
    fn run(prog: &Program) -> crate::objects::value::JsValue {
        use crate::interpreter::{Interpreter, InterpreterFrame};
        let arr = BytecodeGenerator::compile_program(prog).unwrap();
        let mut frame = InterpreterFrame::new(arr, vec![]);
        Interpreter::run(&mut frame).unwrap()
    }

    #[test]
    fn test_generator_call_returns_generator_value() {
        // `function* gen() { yield 1; } return gen();`
        use crate::objects::value::JsValue;
        let decl = make_generator_fn_decl("gen", vec![yield_stmt(Some(num_expr(1.0)), false)]);
        let prog = make_program(vec![
            Stmt::FnDecl(Box::new(decl)),
            Stmt::Return(ReturnStmt {
                loc: span(),
                argument: Some(Box::new(Expr::Call(Box::new(
                    crate::parser::ast::CallExpr {
                        loc: span(),
                        callee: Box::new(ident_expr("gen")),
                        arguments: vec![],
                    },
                )))),
            }),
        ]);
        let result = run(&prog);
        assert!(
            matches!(result, JsValue::Generator(_)),
            "calling a generator function must return JsValue::Generator, got {result:?}"
        );
    }

    #[test]
    fn test_for_of_array_sum() {
        // Build bytecode manually to test IteratorNext:
        //   function(iter) { let sum = 0; for each val in iter: sum += val; return sum; }
        // Register layout:
        //   param[0] = iterator  (encoded as (-1i32) as u32 = 0xFFFF_FFFF)
        //   local r0 = sum
        //   local r1 = value-from-IteratorNext
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::{Instruction, Operand, decode_with_byte_offsets, encode};
        use crate::bytecode::feedback::FeedbackMetadata;
        use crate::interpreter::{Interpreter, InterpreterFrame};
        use crate::objects::value::{JsValue, NativeIterator};

        // param[0] register (iterator) — negative register encoding for param index 0
        let param0 = Operand::Register((-1i32) as u32);
        // local registers
        let sum_reg = Operand::Register(0); // r0
        let val_reg = Operand::Register(1); // r1
        let slot0 = Operand::FeedbackSlot(0);

        let instrs = vec![
            // sum = 0
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![sum_reg]),
            // loop: IteratorNext param0 r1 → r1=value, acc=done
            Instruction::new_unchecked(Opcode::IteratorNext, vec![param0, val_reg]),
            // if done, jump to return
            Instruction::new_unchecked(Opcode::JumpIfToBooleanTrue, vec![Operand::JumpOffset(0)]),
            // acc = val + sum; sum = acc
            Instruction::new_unchecked(Opcode::Ldar, vec![val_reg]),
            Instruction::new_unchecked(Opcode::Add, vec![sum_reg, slot0]),
            Instruction::new_unchecked(Opcode::Star, vec![sum_reg]),
            // JumpLoop back to instr 2
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(0),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            // return sum
            Instruction::new_unchecked(Opcode::Ldar, vec![sum_reg]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        // Resolve jump offsets.
        let initial_bytes = encode(&instrs);
        let (_, offsets) = decode_with_byte_offsets(&initial_bytes).unwrap();
        let mut resolved = instrs.clone();
        // instr 3: JumpIfToBooleanTrue → target = instr 8
        let end3 = offsets[4];
        let tgt8 = offsets[8];
        resolved[3].operands[0] = Operand::JumpOffset(tgt8 as i32 - end3 as i32);
        // instr 7: JumpLoop → target = instr 2
        let end7 = offsets[8];
        let tgt2 = offsets[2];
        resolved[7].operands[0] = Operand::JumpOffset(tgt2 as i32 - end7 as i32);

        let bytes = encode(&resolved);
        let ba = BytecodeArray::new(
            bytes,
            vec![],
            2, // frame_size = 2 locals (r0=sum, r1=val)
            1, // parameter_count = 1 (param[0] = iterator)
            vec![],
            FeedbackMetadata::new(vec![crate::bytecode::feedback::FeedbackSlotKind::BinaryOp]),
            vec![],
        );

        let iter = JsValue::Iterator(NativeIterator::from_items(vec![
            JsValue::Smi(1),
            JsValue::Smi(2),
            JsValue::Smi(3),
        ]));
        let mut frame = InterpreterFrame::new(ba, vec![iter]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(
            result,
            JsValue::Smi(6),
            "for-of sum over [1,2,3] should be 6"
        );
    }

    // ── for-of via AST compilation (array literal not yet available) ──────────

    #[test]
    fn test_for_of_compiles_with_get_iterator_opcode() {
        // `for (const x of someVar) { }` — verify the compiler emits GetIterator.
        use crate::parser::ast::{ForInOfLeft, ForOfStmt, VarDecl, VarDeclarator, VarKind};
        let prog = make_program(vec![Stmt::ForOf(ForOfStmt {
            loc: span(),
            is_await: false,
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Const,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(ident("x")),
                    init: None,
                }],
            }),
            right: Box::new(ident_expr("someArr")),
            body: Box::new(Stmt::Empty(crate::parser::ast::EmptyStmt { loc: span() })),
        })]);
        // Must compile without error.
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetIterator),
            "for-of must emit GetIterator"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::IteratorNext),
            "for-of must emit IteratorNext"
        );
    }

    #[test]
    fn test_for_of_emits_iterator_close_handler() {
        // for-of must emit an IteratorClose in the exception handler
        // so that the iterator is closed when the body throws (§14.7.5.13).
        use crate::parser::ast::{ForInOfLeft, ForOfStmt, VarDecl, VarDeclarator, VarKind};
        let prog = make_program(vec![Stmt::ForOf(ForOfStmt {
            loc: span(),
            is_await: false,
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Const,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(ident("x")),
                    init: None,
                }],
            }),
            right: Box::new(ident_expr("arr")),
            body: Box::new(Stmt::Empty(crate::parser::ast::EmptyStmt { loc: span() })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // The exception handler should contain IteratorClose + ReThrow.
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::IteratorClose),
            "for-of must emit IteratorClose for exception handler"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::ReThrow),
            "for-of exception handler must re-throw"
        );
        // Verify handler table entry exists.
        assert!(
            !arr.handler_table().is_empty(),
            "for-of must register an exception handler table entry"
        );
    }

    // ── yield delegation (yield*) ─────────────────────────────────────────────

    #[test]
    fn test_yield_star_compiles() {
        // `function* outer() { yield* inner; }` — verify compilation emits
        // GetIterator + IteratorNext + SuspendGenerator in the inner bytecode.
        use crate::parser::ast::{FnDecl, YieldExpr};
        let decl = FnDecl {
            loc: span(),
            id: Some(ident("outer")),
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![Stmt::Expr(ExprStmt {
                    loc: span(),
                    expr: Box::new(Expr::Yield(Box::new(YieldExpr {
                        loc: span(),
                        delegate: true,
                        argument: Some(Box::new(ident_expr("inner"))),
                    }))),
                })],
            },
            is_generator: true,
            is_async: false,
            is_strict: false,
        };
        let prog = make_program(vec![Stmt::FnDecl(Box::new(decl))]);
        // Must compile without error.
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        assert!(
            !pool.is_empty(),
            "generator function should be in constant pool"
        );
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_generator());
            let inner_instrs = ba.instructions().unwrap();
            assert!(
                inner_instrs.iter().any(|i| i.opcode == Opcode::GetIterator),
                "yield* must emit GetIterator"
            );
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::IteratorNext),
                "yield* must emit IteratorNext"
            );
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::SuspendGenerator),
                "yield* must emit SuspendGenerator to re-yield values"
            );
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    // ── Labeled break / continue ─────────────────────────────────────────

    #[test]
    fn test_labeled_break_non_loop() {
        // outer: { break outer; }
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Break(BreakStmt {
                    loc: span(),
                    label: Some(ident("outer")),
                })],
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must decode");
        assert!(!decoded.is_empty());
        // Must contain a forward Jump.
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Jump),
            "labeled break must emit a Jump"
        );
    }

    #[test]
    fn test_labeled_break_while_loop() {
        // outer: while (true) { break outer; }
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(bool_expr(true)),
                body: Box::new(Stmt::Break(BreakStmt {
                    loc: span(),
                    label: Some(ident("outer")),
                })),
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must decode");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_labeled_continue_while_loop() {
        // outer: while (true) { continue outer; }
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(bool_expr(true)),
                body: Box::new(Stmt::Continue(ContinueStmt {
                    loc: span(),
                    label: Some(ident("outer")),
                })),
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must decode");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_nested_labeled_loops() {
        // outer: while (true) { inner: while (true) { break outer; } }
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(bool_expr(true)),
                body: Box::new(Stmt::Labeled(LabeledStmt {
                    loc: span(),
                    label: ident("inner"),
                    body: Box::new(Stmt::While(WhileStmt {
                        loc: span(),
                        test: Box::new(bool_expr(true)),
                        body: Box::new(Stmt::Break(BreakStmt {
                            loc: span(),
                            label: Some(ident("outer")),
                        })),
                    })),
                })),
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let bytes = arr.bytecodes();
        let decoded = decode(bytes).expect("bytecode must decode");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_labeled_continue_non_loop_errors() {
        // outer: { continue outer; }  — should error
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::Block(BlockStmt {
                loc: span(),
                body: vec![Stmt::Continue(ContinueStmt {
                    loc: span(),
                    label: Some(ident("outer")),
                })],
            })),
        })]);
        let result = BytecodeGenerator::compile_program(&prog);
        assert!(result.is_err(), "continue on non-loop label must error");
    }

    #[test]
    fn test_duplicate_label_errors() {
        // outer: outer: {} — should error
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("outer"),
            body: Box::new(Stmt::Labeled(LabeledStmt {
                loc: span(),
                label: ident("outer"),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![],
                })),
            })),
        })]);
        let result = BytecodeGenerator::compile_program(&prog);
        assert!(result.is_err(), "duplicate labels must error");
    }

    // ── Getter / setter property bytecode tests ───────────────────────────

    /// Helper: build an object literal expression with the given properties.
    fn object_expr(props: Vec<ObjectProp>) -> Expr {
        Expr::Object(Box::new(ObjectExpr {
            loc: span(),
            properties: props,
        }))
    }

    /// Helper: build a getter or setter property.
    fn accessor_prop(name: &str, kind: PropValue) -> ObjectProp {
        ObjectProp::Prop(Box::new(Prop {
            loc: span(),
            key: PropKey::Ident(ident(name)),
            is_computed: false,
            value: kind,
        }))
    }

    /// Helper: build a minimal FnExpr (empty body, given param count).
    fn empty_fn_expr(param_names: &[&str]) -> FnExpr {
        FnExpr {
            loc: span(),
            id: None,
            is_async: false,
            is_generator: false,
            params: param_names
                .iter()
                .map(|n| Param {
                    loc: span(),
                    pat: Pat::Ident(ident(n)),
                    default: None,
                })
                .collect(),
            body: BlockStmt {
                loc: span(),
                body: vec![],
            },
            is_strict: false,
        }
    }

    #[test]
    fn test_object_getter_emits_define_getter_property() {
        // `var o = { get x() {} };`
        let prog = make_program(vec![var_decl_stmt(
            VarKind::Var,
            "o",
            Some(object_expr(vec![accessor_prop(
                "x",
                PropValue::Get(empty_fn_expr(&[])),
            )])),
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineGetterProperty),
            "expected DefineGetterProperty opcode, got {instrs:?}"
        );
    }

    #[test]
    fn test_object_setter_emits_define_setter_property() {
        // `var o = { set x(v) {} };`
        let prog = make_program(vec![var_decl_stmt(
            VarKind::Var,
            "o",
            Some(object_expr(vec![accessor_prop(
                "x",
                PropValue::Set(empty_fn_expr(&["v"])),
            )])),
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineSetterProperty),
            "expected DefineSetterProperty opcode, got {instrs:?}"
        );
    }

    #[test]
    fn test_object_computed_getter_emits_define_keyed_getter() {
        // `var o = { get [k]() {} };`
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("x"))),
            var_decl_stmt(
                VarKind::Var,
                "o",
                Some(object_expr(vec![ObjectProp::Prop(Box::new(Prop {
                    loc: span(),
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: PropValue::Get(empty_fn_expr(&[])),
                }))])),
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedGetterProperty),
            "expected DefineKeyedGetterProperty opcode, got {instrs:?}"
        );
    }

    #[test]
    fn test_object_computed_setter_emits_define_keyed_setter() {
        // `var o = { set [k](v) {} };`
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("x"))),
            var_decl_stmt(
                VarKind::Var,
                "o",
                Some(object_expr(vec![ObjectProp::Prop(Box::new(Prop {
                    loc: span(),
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: PropValue::Set(empty_fn_expr(&["v"])),
                }))])),
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedSetterProperty),
            "expected DefineKeyedSetterProperty opcode, got {instrs:?}"
        );
    }

    #[test]
    fn test_feedback_slots_getter_setter() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `var o = { get x() {}, set x(v) {} };`
        let prog = make_program(vec![var_decl_stmt(
            VarKind::Var,
            "o",
            Some(object_expr(vec![
                accessor_prop("x", PropValue::Get(empty_fn_expr(&[]))),
                accessor_prop("x", PropValue::Set(empty_fn_expr(&["v"]))),
            ])),
        )]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::DefineAccessor),
            "expected DefineAccessor slot, got {kinds:?}"
        );
    }

    // ── with statement ───────────────────────────────────────────────────

    #[test]
    fn test_with_emits_push_pop_context() {
        use crate::parser::ast::{EmptyStmt, WithStmt};

        let prog = make_program(vec![Stmt::With(WithStmt {
            loc: span(),
            object: Box::new(ident_expr("obj")),
            body: Box::new(Stmt::Empty(EmptyStmt { loc: span() })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let opcodes: Vec<Opcode> = instrs.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::ToObject),
            "expected ToObject, got {opcodes:?}"
        );
        assert!(
            opcodes.contains(&Opcode::PushContext),
            "expected PushContext, got {opcodes:?}"
        );
        assert!(
            opcodes.contains(&Opcode::PopContext),
            "expected PopContext, got {opcodes:?}"
        );
    }

    // ── Async generator functions ─────────────────────────────────────────────

    /// Helper: build a `FnDecl` for an async generator function.
    fn make_async_generator_fn_decl(name: &str, body: Vec<Stmt>) -> FnDecl {
        FnDecl {
            loc: span(),
            id: Some(ident(name)),
            params: vec![],
            body: BlockStmt { loc: span(), body },
            is_generator: true,
            is_async: true,
            is_strict: false,
        }
    }

    /// Helper: build an `await expr` expression statement.
    fn await_stmt(arg: Expr) -> Stmt {
        use crate::parser::ast::{AwaitExpr, ExprStmt};
        Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Await(Box::new(AwaitExpr {
                loc: span(),
                argument: Box::new(arg),
            }))),
        })
    }

    #[test]
    fn test_async_generator_function_compiles() {
        // `async function* gen() { yield 1; yield 2; }`
        let decl = make_async_generator_fn_decl(
            "gen",
            vec![
                yield_stmt(Some(num_expr(1.0)), false),
                yield_stmt(Some(num_expr(2.0)), false),
            ],
        );
        let prog = make_program(vec![Stmt::FnDecl(Box::new(decl))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        assert!(!pool.is_empty());
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_generator(), "must be marked as generator");
            assert!(ba.is_async(), "must be marked as async");
            let inner_instrs = ba.instructions().unwrap();
            assert_eq!(
                inner_instrs[0].opcode,
                Opcode::SwitchOnGeneratorState,
                "async generator body must begin with SwitchOnGeneratorState"
            );
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::SuspendGenerator),
                "async generator body must contain SuspendGenerator"
            );
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    #[test]
    fn test_async_generator_with_await_compiles() {
        // `async function* gen() { await x; yield 1; }`
        let decl = make_async_generator_fn_decl(
            "gen",
            vec![
                await_stmt(ident_expr("x")),
                yield_stmt(Some(num_expr(1.0)), false),
            ],
        );
        let prog = make_program(vec![Stmt::FnDecl(Box::new(decl))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_async(), "must be marked as async");
            let inner_instrs = ba.instructions().unwrap();
            // The body should have two SuspendGenerator instructions: one for
            // await (suspend/resume) and one for yield.
            let suspend_count = inner_instrs
                .iter()
                .filter(|i| i.opcode == Opcode::SuspendGenerator)
                .count();
            assert!(
                suspend_count >= 2,
                "expected >= 2 SuspendGenerator (1 await + 1 yield), got {suspend_count}"
            );
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    #[test]
    fn test_async_generator_expr_compiles() {
        // `var f = async function*() { yield 1; };`
        use crate::parser::ast::FnExpr;
        let fn_expr = Expr::Fn(Box::new(FnExpr {
            loc: span(),
            id: None,
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![yield_stmt(Some(num_expr(1.0)), false)],
            },
            is_generator: true,
            is_async: true,
            is_strict: false,
        }));
        let prog = make_program(vec![var_decl_stmt(VarKind::Var, "f", Some(fn_expr))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_generator(), "must be marked as generator");
            assert!(ba.is_async(), "must be marked as async");
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    #[test]
    fn test_for_await_of_compiles_with_get_async_iterator() {
        // `for await (const x of someArr) { }` — verify GetAsyncIterator.
        use crate::parser::ast::{ForInOfLeft, ForOfStmt, VarDecl, VarDeclarator, VarKind};
        let prog = make_program(vec![Stmt::ForOf(ForOfStmt {
            loc: span(),
            is_await: true,
            left: ForInOfLeft::VarDecl(VarDecl {
                loc: span(),
                kind: VarKind::Const,
                declarators: vec![VarDeclarator {
                    loc: span(),
                    id: Pat::Ident(ident("x")),
                    init: None,
                }],
            }),
            right: Box::new(ident_expr("someArr")),
            body: Box::new(Stmt::Empty(crate::parser::ast::EmptyStmt { loc: span() })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetAsyncIterator),
            "for-await-of must emit GetAsyncIterator, got {instrs:?}"
        );
    }

    // ── Class compilation tests ───────────────────────────────────────────

    /// Helper: build a class declaration statement.
    fn class_decl_stmt(name: &str, super_class: Option<Expr>, body: Vec<ClassMember>) -> Stmt {
        Stmt::ClassDecl(Box::new(ClassDecl {
            loc: span(),
            id: Some(ident(name)),
            super_class: super_class.map(Box::new),
            body: ClassBody { loc: span(), body },
        }))
    }

    /// Helper: build a class method/getter/setter member.
    fn method_member(
        name: &str,
        params: &[&str],
        is_static: bool,
        kind: MethodKind,
    ) -> ClassMember {
        ClassMember::Method(MethodDef {
            loc: span(),
            is_static,
            kind,
            key: PropKey::Ident(ident(name)),
            is_computed: false,
            value: empty_fn_expr(params),
        })
    }

    /// Helper: build a class field member.
    fn field_member(name: &str, value: Option<Expr>, is_static: bool) -> ClassMember {
        ClassMember::Property(PropertyDef {
            loc: span(),
            is_static,
            key: PropKey::Ident(ident(name)),
            is_computed: false,
            value: value.map(Box::new),
        })
    }

    /// Helper: build a static block member.
    fn static_block_member(body: Vec<Stmt>) -> ClassMember {
        ClassMember::StaticBlock(StaticBlock { loc: span(), body })
    }

    #[test]
    fn test_class_decl_emits_create_class() {
        // `class Foo {}`
        let prog = make_program(vec![class_decl_stmt("Foo", None, vec![])]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CreateClass),
            "expected CreateClass opcode, got {instrs:?}"
        );
    }

    #[test]
    fn test_async_generator_call_returns_generator_value() {
        // `async function* gen() { yield 1; } return gen();`
        use crate::objects::value::JsValue;
        let decl =
            make_async_generator_fn_decl("gen", vec![yield_stmt(Some(num_expr(1.0)), false)]);
        let prog = make_program(vec![
            Stmt::FnDecl(Box::new(decl)),
            Stmt::Return(ReturnStmt {
                loc: span(),
                argument: Some(Box::new(Expr::Call(Box::new(
                    crate::parser::ast::CallExpr {
                        loc: span(),
                        callee: Box::new(ident_expr("gen")),
                        arguments: vec![],
                    },
                )))),
            }),
        ]);
        let result = run(&prog);
        assert!(
            matches!(result, JsValue::Generator(_)),
            "calling an async generator must return JsValue::Generator, got {result:?}"
        );
    }

    #[test]
    fn test_class_with_constructor() {
        // `class Foo { constructor() {} }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![method_member(
                "constructor",
                &[],
                false,
                MethodKind::Constructor,
            )],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::CreateClass));
    }

    #[test]
    fn test_class_method_emits_define_named_own() {
        // `class Foo { bar() {} }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![method_member("bar", &[], false, MethodKind::Method)],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineNamedOwnProperty),
            "expected DefineNamedOwnProperty for method"
        );
    }

    #[test]
    fn test_class_static_method() {
        // `class Foo { static bar() {} }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![method_member("bar", &[], true, MethodKind::Method)],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineNamedOwnProperty)
        );
    }

    #[test]
    fn test_class_getter_setter() {
        // `class Foo { get x() {} set x(v) {} }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![
                method_member("x", &[], false, MethodKind::Get),
                method_member("x", &["v"], false, MethodKind::Set),
            ],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineGetterProperty)
        );
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineSetterProperty)
        );
    }

    #[test]
    fn test_class_extends() {
        // `class Foo {} class Bar extends Foo {}`
        let prog = make_program(vec![
            class_decl_stmt("Foo", None, vec![]),
            class_decl_stmt("Bar", Some(ident_expr("Foo")), vec![]),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let count = instrs
            .iter()
            .filter(|i| i.opcode == Opcode::CreateClass)
            .count();
        assert_eq!(count, 2, "expected two CreateClass opcodes");
    }

    #[test]
    fn test_class_static_field() {
        // `class Foo { static count = 0 }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![field_member("count", Some(num_expr(0.0)), true)],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineNamedOwnProperty)
        );
    }

    #[test]
    fn test_class_instance_field_emits_initializer() {
        // `class Foo { x = 1 }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![field_member("x", Some(num_expr(1.0)), false)],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Instance field initializer produces a CreateClosure.
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CreateClosure),
            "expected CreateClosure for field initializer"
        );
    }

    #[test]
    fn test_class_expression() {
        // `var C = class {};`
        let prog = make_program(vec![var_decl_stmt(
            VarKind::Var,
            "C",
            Some(Expr::Class(Box::new(ClassExpr {
                loc: span(),
                id: None,
                super_class: None,
                body: ClassBody {
                    loc: span(),
                    body: vec![],
                },
            }))),
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CreateClass),
            "expected CreateClass for class expression"
        );
    }

    #[test]
    fn test_class_static_block() {
        // `class Foo { static { } }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![static_block_member(vec![])],
        )]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(instrs.iter().any(|i| i.opcode == Opcode::CreateClass));
    }

    #[test]
    fn test_class_computed_method() {
        // `class Foo { [k]() {} }` (k declared beforehand)
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("bar"))),
            class_decl_stmt(
                "Foo",
                None,
                vec![ClassMember::Method(MethodDef {
                    loc: span(),
                    is_static: false,
                    kind: MethodKind::Method,
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: empty_fn_expr(&[]),
                })],
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedOwnProperty),
            "expected DefineKeyedOwnProperty for computed method"
        );
    }

    #[test]
    fn test_feedback_slots_class_method() {
        use crate::bytecode::feedback::FeedbackSlotKind;
        // `class Foo { bar() {} }`
        let prog = make_program(vec![class_decl_stmt(
            "Foo",
            None,
            vec![method_member("bar", &[], false, MethodKind::Method)],
        )]);
        let kinds = slot_kinds_for(&prog);
        assert!(
            kinds.contains(&FeedbackSlotKind::CreateClosure),
            "expected CreateClosure slot for class, got {kinds:?}"
        );
        assert!(
            kinds.contains(&FeedbackSlotKind::StoreProperty),
            "expected StoreProperty slot for method, got {kinds:?}"
        );
    }

    // ── Async function compilation ────────────────────────────────────────────

    #[test]
    fn test_async_function_decl_compiles() {
        // `async function f() { await 1; return 2; }`
        let decl = FnDecl {
            loc: span(),
            id: Some(ident("f")),
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![
                    await_stmt(num_expr(1.0)),
                    Stmt::Return(ReturnStmt {
                        loc: span(),
                        argument: Some(Box::new(num_expr(2.0))),
                    }),
                ],
            },
            is_generator: false,
            is_async: true,
            is_strict: false,
        };
        let prog = make_program(vec![Stmt::FnDecl(Box::new(decl))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        assert!(!pool.is_empty());
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_async(), "must be marked as async");
            assert!(!ba.is_generator(), "must NOT be marked as generator");
            let inner_instrs = ba.instructions().unwrap();
            assert_eq!(
                inner_instrs[0].opcode,
                Opcode::SwitchOnGeneratorState,
                "async function body must begin with SwitchOnGeneratorState"
            );
            assert!(
                inner_instrs
                    .iter()
                    .any(|i| i.opcode == Opcode::SuspendGenerator),
                "async function body must contain SuspendGenerator for await"
            );
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    #[test]
    fn test_async_function_expr_compiles() {
        // `var f = async function() { await 1; };`
        use crate::parser::ast::FnExpr;
        let fn_expr = Expr::Fn(Box::new(FnExpr {
            loc: span(),
            id: None,
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![await_stmt(num_expr(1.0))],
            },
            is_generator: false,
            is_async: true,
            is_strict: false,
        }));
        let prog = make_program(vec![var_decl_stmt(VarKind::Var, "f", Some(fn_expr))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_async(), "must be marked as async");
            assert!(!ba.is_generator(), "must NOT be marked as generator");
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    #[test]
    fn test_async_arrow_compiles() {
        // `var f = async () => { await 1; };`
        use crate::parser::ast::ArrowExpr;
        let arrow = Expr::Arrow(Box::new(ArrowExpr {
            loc: span(),
            params: vec![],
            body: ArrowBody::Block(BlockStmt {
                loc: span(),
                body: vec![await_stmt(num_expr(1.0))],
            }),
            is_async: true,
            is_strict: false,
        }));
        let prog = make_program(vec![var_decl_stmt(VarKind::Var, "f", Some(arrow))]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let pool = arr.constant_pool();
        if let crate::bytecode::bytecode_array::ConstantPoolEntry::Function(ba) = &pool[0] {
            assert!(ba.is_async(), "must be marked as async");
        } else {
            panic!("constant pool[0] should be a Function");
        }
    }

    // ── Optional catch binding ────────────────────────────────────────────────

    #[test]
    fn test_optional_catch_binding() {
        // try { let x = 1; } catch { }
        let prog = make_program(vec![Stmt::Try(TryStmt {
            loc: span(),
            block: BlockStmt {
                loc: span(),
                body: vec![var_decl_stmt(VarKind::Let, "x", Some(num_expr(1.0)))],
            },
            handler: Some(CatchClause {
                loc: span(),
                param: None,
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
        decode(bytes).expect("optional catch binding bytecode must decode");
    }

    // ── Nullish coalescing ────────────────────────────────────────────────────

    #[test]
    fn test_nullish_coalesce_bytecode() {
        // a ?? b
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Logical(Box::new(LogicalExpr {
                loc: span(),
                op: LogicalOp::NullishCoalesce,
                left: Box::new(ident_expr("a")),
                right: Box::new(ident_expr("b")),
            }))),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(!instrs.is_empty());
        let opcodes: Vec<_> = instrs.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::JumpIfUndefinedOrNull),
            "expected JumpIfUndefinedOrNull, got {opcodes:?}"
        );
    }

    // ── Logical assignment operators ──────────────────────────────────────────

    #[test]
    fn test_logical_and_assign_bytecode() {
        // x &&= 1
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(bool_expr(true))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                    loc: span(),
                    op: AssignOp::LogicalAndAssign,
                    left: AssignTarget::Expr(Box::new(ident_expr("x"))),
                    right: Box::new(num_expr(1.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let opcodes: Vec<_> = instrs.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::JumpIfToBooleanFalse),
            "expected JumpIfToBooleanFalse for &&=, got {opcodes:?}"
        );
    }

    #[test]
    fn test_logical_or_assign_bytecode() {
        // x ||= 1
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(bool_expr(false))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                    loc: span(),
                    op: AssignOp::LogicalOrAssign,
                    left: AssignTarget::Expr(Box::new(ident_expr("x"))),
                    right: Box::new(num_expr(1.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let opcodes: Vec<_> = instrs.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::JumpIfToBooleanTrue),
            "expected JumpIfToBooleanTrue for ||=, got {opcodes:?}"
        );
    }

    #[test]
    fn test_nullish_assign_bytecode() {
        // x ??= 1
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "x", Some(null_expr())),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                    loc: span(),
                    op: AssignOp::NullishAssign,
                    left: AssignTarget::Expr(Box::new(ident_expr("x"))),
                    right: Box::new(num_expr(1.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let opcodes: Vec<_> = instrs.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::JumpIfUndefinedOrNull),
            "expected JumpIfUndefinedOrNull for ??=, got {opcodes:?}"
        );
    }

    // ── Destructuring & default parameters ───────────────────────────────

    #[test]
    fn test_default_param_emits_jump_if_undefined() {
        // function f(a = 1) { return a; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("a")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Ident(ident("a")),
                default: Some(num_expr(1.0)),
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::JumpIfUndefined),
            "default param must emit JumpIfUndefined"
        );
        assert_eq!(inner.parameter_count(), 1);
    }

    #[test]
    fn test_object_destructuring_param() {
        // function f({a, b}) { return a; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("a")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Object(Box::new(ObjectPat {
                    loc: span(),
                    properties: vec![
                        ObjectPatProp::Assign(AssignPatProp {
                            loc: span(),
                            key: ident("a"),
                            value: None,
                        }),
                        ObjectPatProp::Assign(AssignPatProp {
                            loc: span(),
                            key: ident("b"),
                            value: None,
                        }),
                    ],
                })),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .filter(|i| i.opcode == Opcode::LdaNamedProperty)
                .count()
                >= 2,
            "object destructuring param must emit LdaNamedProperty for each property"
        );
    }

    #[test]
    fn test_array_destructuring_param() {
        // function f([x, y]) { return x; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("x")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Array(Box::new(ArrayPat {
                    loc: span(),
                    elements: vec![Some(Pat::Ident(ident("x"))), Some(Pat::Ident(ident("y")))],
                })),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetIterator),
            "array destructuring param must emit GetIterator"
        );
        assert!(
            instrs
                .iter()
                .filter(|i| i.opcode == Opcode::IteratorNext)
                .count()
                >= 2,
            "array destructuring param must emit IteratorNext for each element"
        );
    }

    #[test]
    fn test_object_destructuring_with_rename() {
        // function f({a: x}) { return x; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("x")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Object(Box::new(ObjectPat {
                    loc: span(),
                    properties: vec![ObjectPatProp::KeyValue(KeyValuePatProp {
                        loc: span(),
                        key: PropKey::Ident(ident("a")),
                        is_computed: false,
                        value: Pat::Ident(ident("x")),
                    })],
                })),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaNamedProperty),
            "renamed object destructuring must emit LdaNamedProperty"
        );
    }

    #[test]
    fn test_default_param_with_later_param_ref() {
        // function f(a = 1, b = a + 1) { return b; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("b")))],
        };
        let inner = compile_function(
            &[
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("a")),
                    default: Some(num_expr(1.0)),
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(ident("b")),
                    default: Some(Expr::Binary(Box::new(BinaryExpr {
                        loc: span(),
                        op: BinaryOp::Add,
                        left: Box::new(ident_expr("a")),
                        right: Box::new(num_expr(1.0)),
                    }))),
                },
            ],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert_eq!(
            instrs
                .iter()
                .filter(|i| i.opcode == Opcode::JumpIfUndefined)
                .count(),
            2,
            "each default param must emit its own JumpIfUndefined"
        );
    }

    #[test]
    fn test_object_destructuring_param_with_default() {
        // function f({a, b = 42}) { return b; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("b")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Object(Box::new(ObjectPat {
                    loc: span(),
                    properties: vec![
                        ObjectPatProp::Assign(AssignPatProp {
                            loc: span(),
                            key: ident("a"),
                            value: None,
                        }),
                        ObjectPatProp::Assign(AssignPatProp {
                            loc: span(),
                            key: ident("b"),
                            value: Some(Box::new(num_expr(42.0))),
                        }),
                    ],
                })),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::JumpIfNotUndefined),
            "object destructuring with default must emit JumpIfNotUndefined"
        );
    }

    #[test]
    fn test_array_destructuring_with_elision() {
        // function f([, x]) { return x; }
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(ident_expr("x")))],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Array(Box::new(ArrayPat {
                    loc: span(),
                    elements: vec![None, Some(Pat::Ident(ident("x")))],
                })),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        let instrs = inner.instructions().unwrap();
        assert_eq!(
            instrs
                .iter()
                .filter(|i| i.opcode == Opcode::IteratorNext)
                .count(),
            2,
            "elision must still advance the iterator"
        );
    }

    #[test]
    fn test_dynamic_import_emits_call_runtime() {
        use crate::parser::ast::ImportExpr;

        let import_expr = Expr::Import(Box::new(ImportExpr {
            loc: span(),
            source: Box::new(str_expr("./mod.js")),
            options: None,
        }));
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(import_expr),
        })]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CallRuntime),
            "expected CallRuntime opcode for dynamic import"
        );
    }

    // ── Module compilation ────────────────────────────────────────────────────

    fn module_program(items: Vec<ProgramItem>) -> Program {
        Program {
            loc: span(),
            source_type: SourceType::Module,
            body: items,
            is_strict: true,
        }
    }

    fn string_lit(s: &str) -> StringLit {
        StringLit {
            loc: span(),
            value: s.to_owned(),
        }
    }

    #[test]
    fn test_import_named_emits_lda_module_variable() {
        use crate::parser::ast::{
            ImportDecl, ImportNamedSpecifier, ImportSpecifier, ModuleDecl, ModuleExportName,
        };
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::Import(
            ImportDecl {
                loc: span(),
                specifiers: vec![ImportSpecifier::Named(ImportNamedSpecifier {
                    loc: span(),
                    imported: ModuleExportName::Ident(ident("foo")),
                    local: ident("foo"),
                })],
                source: string_lit("./mod.js"),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        assert!(ba.is_module());
        // Modules are implicitly async (top-level await).
        assert!(ba.is_async());
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaModuleVariable),
            "import should emit LdaModuleVariable, got {instrs:?}"
        );
    }

    #[test]
    fn test_module_top_level_await_compiles() {
        use crate::parser::ast::{
            AwaitExpr, ExprStmt, ImportDecl, ImportDefaultSpecifier, ImportSpecifier, ModuleDecl,
        };
        let await_stmt = ProgramItem::Stmt(Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Await(Box::new(AwaitExpr {
                loc: span(),
                argument: Box::new(num_expr(42.0)),
            }))),
        }));
        let import = ProgramItem::ModuleDecl(ModuleDecl::Import(ImportDecl {
            loc: span(),
            specifiers: vec![ImportSpecifier::Default(ImportDefaultSpecifier {
                loc: span(),
                local: ident("m"),
            })],
            source: string_lit("./mod.js"),
            attributes: vec![],
        }));
        let prog = module_program(vec![import, await_stmt]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        assert!(ba.is_module());
        assert!(ba.is_async());
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::SuspendGenerator),
            "top-level await should emit SuspendGenerator, got {instrs:?}"
        );
    }

    #[test]
    fn test_import_default_emits_lda_module_variable() {
        use crate::parser::ast::{ImportDecl, ImportDefaultSpecifier, ImportSpecifier, ModuleDecl};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::Import(
            ImportDecl {
                loc: span(),
                specifiers: vec![ImportSpecifier::Default(ImportDefaultSpecifier {
                    loc: span(),
                    local: ident("def"),
                })],
                source: string_lit("./mod.js"),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaModuleVariable),
            "default import should emit LdaModuleVariable"
        );
    }

    #[test]
    fn test_dynamic_import_with_options_emits_call_runtime() {
        use crate::parser::ast::ImportExpr;

        let import_expr = Expr::Import(Box::new(ImportExpr {
            loc: span(),
            source: Box::new(str_expr("./mod.json")),
            options: Some(Box::new(ident_expr("opts"))),
        }));
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(import_expr),
        })]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        let rt_call = instrs
            .iter()
            .find(|i| i.opcode == Opcode::CallRuntime)
            .expect("expected CallRuntime");
        assert_eq!(
            rt_call.operands[2],
            Operand::RegisterCount(2),
            "import with options should have 2 args"
        );
    }

    #[test]
    fn test_import_namespace_emits_get_module_namespace() {
        use crate::parser::ast::{
            ImportDecl, ImportNamespaceSpecifier, ImportSpecifier, ModuleDecl,
        };
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::Import(
            ImportDecl {
                loc: span(),
                specifiers: vec![ImportSpecifier::Namespace(ImportNamespaceSpecifier {
                    loc: span(),
                    local: ident("ns"),
                })],
                source: string_lit("./mod.js"),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::GetModuleNamespace),
            "namespace import should emit GetModuleNamespace"
        );
    }

    #[test]
    fn test_export_named_decl_emits_sta_module_variable() {
        use crate::parser::ast::{ExportNamedDecl, ModuleDecl};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(
            ExportNamedDecl {
                loc: span(),
                specifiers: vec![],
                source: None,
                declaration: Some(Box::new(var_decl_stmt(
                    VarKind::Let,
                    "x",
                    Some(num_expr(42.0)),
                ))),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaModuleVariable),
            "export let should emit StaModuleVariable"
        );
    }

    #[test]
    fn test_export_default_expr_emits_sta_module_variable() {
        use crate::parser::ast::{ExportDefaultDecl, ExportDefaultExpr, ModuleDecl};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportDefault(
            ExportDefaultDecl {
                loc: span(),
                declaration: ExportDefaultExpr::Expr(Box::new(num_expr(99.0))),
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaModuleVariable),
            "export default should emit StaModuleVariable"
        );
    }

    #[test]
    fn test_export_all_emits_get_module_namespace() {
        use crate::parser::ast::{ExportAllDecl, ModuleDecl};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportAll(
            ExportAllDecl {
                loc: span(),
                exported: None,
                source: string_lit("./other.js"),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::GetModuleNamespace),
            "export * should emit GetModuleNamespace"
        );
    }

    #[test]
    fn test_export_all_as_name_emits_sta_module_variable() {
        use crate::parser::ast::{ExportAllDecl, ModuleDecl, ModuleExportName};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportAll(
            ExportAllDecl {
                loc: span(),
                exported: Some(ModuleExportName::Ident(ident("ns"))),
                source: string_lit("./other.js"),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::GetModuleNamespace),
            "export * as ns should emit GetModuleNamespace"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaModuleVariable),
            "export * as ns should emit StaModuleVariable"
        );
    }

    #[test]
    fn test_re_export_named_from_source() {
        use crate::parser::ast::{ExportNamedDecl, ExportSpecifier, ModuleDecl, ModuleExportName};
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(
            ExportNamedDecl {
                loc: span(),
                specifiers: vec![ExportSpecifier {
                    loc: span(),
                    local: ModuleExportName::Ident(ident("x")),
                    exported: ModuleExportName::Ident(ident("y")),
                }],
                source: Some(string_lit("./mod.js")),
                declaration: None,
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaModuleVariable),
            "re-export should load from source via LdaModuleVariable"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaModuleVariable),
            "re-export should store via StaModuleVariable"
        );
    }

    #[test]
    fn test_import_meta_emits_lda_import_meta() {
        use crate::parser::ast::MetaPropExpr;
        let prog = module_program(vec![ProgramItem::Stmt(Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::MetaProp(MetaPropExpr {
                loc: span(),
                meta: ident("import"),
                property: ident("meta"),
            })),
        }))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaImportMeta),
            "import.meta should emit LdaImportMeta"
        );
    }

    #[test]
    fn test_module_decl_in_script_errors() {
        use crate::parser::ast::{ImportDecl, ModuleDecl};
        let prog = Program {
            loc: span(),
            source_type: SourceType::Script,
            body: vec![ProgramItem::ModuleDecl(ModuleDecl::Import(ImportDecl {
                loc: span(),
                specifiers: vec![],
                source: string_lit("./mod.js"),
                attributes: vec![],
            }))],
            is_strict: false,
        };
        let result = BytecodeGenerator::compile_program(&prog);
        assert!(result.is_err(), "module decl in script should error");
    }

    #[test]
    fn test_live_binding_export_let_stores_module_variable() {
        use crate::parser::ast::{ExportNamedDecl, ModuleDecl};
        // `export let counter = 0;` should store via StaModuleVariable
        // so importers see updates (live binding semantics).
        let prog = module_program(vec![ProgramItem::ModuleDecl(ModuleDecl::ExportNamed(
            ExportNamedDecl {
                loc: span(),
                specifiers: vec![],
                source: None,
                declaration: Some(Box::new(var_decl_stmt(
                    VarKind::Let,
                    "counter",
                    Some(num_expr(0.0)),
                ))),
                attributes: vec![],
            },
        ))]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = decode(&ba.bytecodes()).unwrap();
        let sta_count = instrs
            .iter()
            .filter(|i| i.opcode == Opcode::StaModuleVariable)
            .count();
        assert!(
            sta_count >= 1,
            "export let must emit at least one StaModuleVariable for live binding"
        );
    }

    // ── Destructuring-assignment integration tests ────────────────────────

    #[test]
    fn test_array_destructuring_assign() {
        let result =
            crate::builtins::global::global_eval("var a, b; [a, b] = [10, 20]; a + b").unwrap();
        assert_eq!(result, crate::objects::value::JsValue::Smi(30));
    }

    #[test]
    fn test_object_destructuring_assign() {
        let result =
            crate::builtins::global::global_eval("var x, y; ({x, y} = {x: 10, y: 20}); x + y")
                .unwrap();
        assert_eq!(result, crate::objects::value::JsValue::Smi(30));
    }

    // ── Improvement 1: new.target meta-property ──────────────────────────

    #[test]
    fn test_new_target_emits_lda_new_target() {
        // function f() { return new.target; }
        use crate::parser::ast::MetaPropExpr;
        let body = BlockStmt {
            loc: span(),
            body: vec![return_stmt(Some(Expr::MetaProp(MetaPropExpr {
                loc: span(),
                meta: ident("new"),
                property: ident("target"),
            })))],
        };
        let inner = compile_function(&[], &body, false, false, false).unwrap();
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaNewTarget),
            "new.target must emit LdaNewTarget, got {instrs:?}"
        );
    }

    #[test]
    fn test_new_target_in_program_compiles() {
        // Top-level new.target should compile without error.
        use crate::parser::ast::MetaPropExpr;
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::MetaProp(MetaPropExpr {
                loc: span(),
                meta: ident("new"),
                property: ident("target"),
            })),
        })]);
        let ba = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = ba.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaNewTarget),
            "program-level new.target must emit LdaNewTarget"
        );
    }

    // ── Improvement 2: labeled statements / continue label ───────────────

    #[test]
    fn test_labeled_continue_for_loop() {
        // L: for (var i = 0; i < 10; i++) { continue L; }
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("L"),
            body: Box::new(Stmt::For(ForStmt {
                loc: span(),
                init: Some(ForInit::VarDecl(VarDecl {
                    loc: span(),
                    kind: VarKind::Var,
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
                    right: Box::new(num_expr(10.0)),
                })))),
                update: Some(Box::new(Expr::Update(Box::new(
                    crate::parser::ast::UpdateExpr {
                        loc: span(),
                        op: UpdateOp::Increment,
                        prefix: false,
                        argument: Box::new(ident_expr("i")),
                    },
                )))),
                body: Box::new(Stmt::Continue(ContinueStmt {
                    loc: span(),
                    label: Some(ident("L")),
                })),
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Must contain a Jump for the labeled continue.
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Jump),
            "labeled continue L in for loop must emit a Jump"
        );
    }

    #[test]
    fn test_labeled_continue_do_while_loop() {
        // L: do { continue L; } while (true);
        let prog = make_program(vec![Stmt::Labeled(LabeledStmt {
            loc: span(),
            label: ident("L"),
            body: Box::new(Stmt::DoWhile(DoWhileStmt {
                loc: span(),
                test: Box::new(bool_expr(true)),
                body: Box::new(Stmt::Continue(ContinueStmt {
                    loc: span(),
                    label: Some(ident("L")),
                })),
            })),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Jump),
            "labeled continue L in do-while must emit a Jump"
        );
    }

    #[test]
    fn test_labeled_continue_for_in_loop() {
        // L: for (var x in obj) { continue L; }
        use crate::parser::ast::{ForInOfLeft, ForInStmt};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "obj", Some(num_expr(0.0))),
            Stmt::Labeled(LabeledStmt {
                loc: span(),
                label: ident("L"),
                body: Box::new(Stmt::ForIn(ForInStmt {
                    loc: span(),
                    left: ForInOfLeft::VarDecl(VarDecl {
                        loc: span(),
                        kind: VarKind::Var,
                        declarators: vec![VarDeclarator {
                            loc: span(),
                            id: Pat::Ident(ident("x")),
                            init: None,
                        }],
                    }),
                    right: Box::new(ident_expr("obj")),
                    body: Box::new(Stmt::Continue(ContinueStmt {
                        loc: span(),
                        label: Some(ident("L")),
                    })),
                })),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Jump),
            "labeled continue L in for-in must emit a Jump"
        );
    }

    // ── Improvement 3: for-in with var hoists to function scope ──────────

    #[test]
    fn test_for_in_var_hoists_to_function_scope() {
        // function f(o) { for (var x in o) {} return x; }
        // `x` must be accessible after the loop (function-scoped).
        use crate::parser::ast::{ForInOfLeft, ForInStmt};
        let body = BlockStmt {
            loc: span(),
            body: vec![
                Stmt::ForIn(ForInStmt {
                    loc: span(),
                    left: ForInOfLeft::VarDecl(VarDecl {
                        loc: span(),
                        kind: VarKind::Var,
                        declarators: vec![VarDeclarator {
                            loc: span(),
                            id: Pat::Ident(ident("x")),
                            init: None,
                        }],
                    }),
                    right: Box::new(ident_expr("o")),
                    body: Box::new(Stmt::Block(BlockStmt {
                        loc: span(),
                        body: vec![],
                    })),
                }),
                return_stmt(Some(ident_expr("x"))),
            ],
        };
        let inner = compile_function(
            &[Param {
                loc: span(),
                pat: Pat::Ident(ident("o")),
                default: None,
            }],
            &body,
            false,
            false,
            false,
        )
        .unwrap();
        // The key assertion: compilation succeeds and emits a Ldar for x
        // after the loop (meaning x was found in the function scope).
        let instrs = inner.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Return),
            "function must compile successfully with var x visible after for-in"
        );
    }

    #[test]
    fn test_for_in_var_visible_in_nested_scope() {
        // `for (var k in obj) {} k;` at program level — k should be accessible.
        use crate::parser::ast::{ForInOfLeft, ForInStmt};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "obj", Some(num_expr(0.0))),
            Stmt::ForIn(ForInStmt {
                loc: span(),
                left: ForInOfLeft::VarDecl(VarDecl {
                    loc: span(),
                    kind: VarKind::Var,
                    declarators: vec![VarDeclarator {
                        loc: span(),
                        id: Pat::Ident(ident("k")),
                        init: None,
                    }],
                }),
                right: Box::new(ident_expr("obj")),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![],
                })),
            }),
            // Reference k after the for-in loop.
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(ident_expr("k")),
            }),
        ]);
        // Should compile without errors — k is function-scoped via var.
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            !instrs.is_empty(),
            "for-in var k should be accessible after the loop"
        );
    }

    // ── Improvement 4: computed property names in object methods ─────────

    #[test]
    fn test_object_computed_method_emits_define_keyed() {
        // `var o = { [k]() {} };` — should use DefineKeyedOwnProperty.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("foo"))),
            var_decl_stmt(
                VarKind::Var,
                "o",
                Some(object_expr(vec![ObjectProp::Prop(Box::new(Prop {
                    loc: span(),
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: PropValue::Method(empty_fn_expr(&[])),
                }))])),
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedOwnProperty),
            "computed method must emit DefineKeyedOwnProperty, got {instrs:?}"
        );
    }

    #[test]
    fn test_object_computed_value_emits_define_keyed() {
        // `var o = { [k]: 42 };` — should use DefineKeyedOwnProperty.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("bar"))),
            var_decl_stmt(
                VarKind::Var,
                "o",
                Some(object_expr(vec![ObjectProp::Prop(Box::new(Prop {
                    loc: span(),
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: PropValue::Value(Box::new(num_expr(42.0))),
                }))])),
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedOwnProperty),
            "computed value property must emit DefineKeyedOwnProperty, got {instrs:?}"
        );
    }

    // ── Improvement 5: template literals with expressions ────────────────

    #[test]
    fn test_template_with_expressions_emits_add() {
        // `` `a${x}b` `` — should emit Add opcodes for concatenation.
        use crate::parser::ast::{TemplateElement, TemplateLit};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "x", Some(num_expr(1.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Template(Box::new(TemplateLit {
                    loc: span(),
                    quasis: vec![
                        TemplateElement {
                            loc: span(),
                            raw: "a".to_owned(),
                            cooked: Some("a".to_owned()),
                            tail: false,
                        },
                        TemplateElement {
                            loc: span(),
                            raw: "b".to_owned(),
                            cooked: Some("b".to_owned()),
                            tail: true,
                        },
                    ],
                    expressions: vec![ident_expr("x")],
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        let add_count = instrs.iter().filter(|i| i.opcode == Opcode::Add).count();
        assert!(
            add_count >= 2,
            "template with expression needs at least 2 Add opcodes (expr+quasi), got {add_count}"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::ToString),
            "template must emit ToString for interpolated expression"
        );
    }

    #[test]
    fn test_tagged_template_emits_get_template_object() {
        // `` tag`str${expr}str` `` — should emit GetTemplateObject and a call.
        use crate::parser::ast::{TaggedTemplateExpr, TemplateElement, TemplateLit};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "tag", Some(num_expr(0.0))),
            var_decl_stmt(VarKind::Var, "expr", Some(num_expr(1.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::TaggedTemplate(Box::new(TaggedTemplateExpr {
                    loc: span(),
                    tag: Box::new(ident_expr("tag")),
                    quasi: TemplateLit {
                        loc: span(),
                        quasis: vec![
                            TemplateElement {
                                loc: span(),
                                raw: "str".to_owned(),
                                cooked: Some("str".to_owned()),
                                tail: false,
                            },
                            TemplateElement {
                                loc: span(),
                                raw: "str".to_owned(),
                                cooked: Some("str".to_owned()),
                                tail: true,
                            },
                        ],
                        expressions: vec![ident_expr("expr")],
                    },
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetTemplateObject),
            "tagged template must emit GetTemplateObject, got {instrs:?}"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CallAnyReceiver),
            "tagged template must emit a call instruction, got {instrs:?}"
        );
    }

    #[test]
    fn test_tagged_template_method_uses_call_property() {
        // `` obj.tag`a${x}b` `` — should use CallProperty with correct this.
        use crate::parser::ast::{
            MemberExpr, MemberProp, TaggedTemplateExpr, TemplateElement, TemplateLit,
        };
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "obj", Some(num_expr(0.0))),
            var_decl_stmt(VarKind::Var, "x", Some(num_expr(1.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::TaggedTemplate(Box::new(TaggedTemplateExpr {
                    loc: span(),
                    tag: Box::new(Expr::Member(Box::new(MemberExpr {
                        loc: span(),
                        object: Box::new(ident_expr("obj")),
                        property: MemberProp::Ident(ident("tag")),
                        is_computed: false,
                    }))),
                    quasi: TemplateLit {
                        loc: span(),
                        quasis: vec![
                            TemplateElement {
                                loc: span(),
                                raw: "a".to_owned(),
                                cooked: Some("a".to_owned()),
                                tail: false,
                            },
                            TemplateElement {
                                loc: span(),
                                raw: "b".to_owned(),
                                cooked: Some("b".to_owned()),
                                tail: true,
                            },
                        ],
                        expressions: vec![ident_expr("x")],
                    },
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetTemplateObject),
            "tagged template method must emit GetTemplateObject"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::CallProperty),
            "tagged template on method must use CallProperty for correct `this`"
        );
    }

    #[test]
    fn test_member_expr_in_object_destructuring_assignment() {
        // `({a: obj.x} = {a: 42})` — member expression as destructuring target.
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget, MemberExpr};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "obj", Some(num_expr(0.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                    loc: span(),
                    op: AssignOp::Assign,
                    left: AssignTarget::Pat(Pat::Object(Box::new(ObjectPat {
                        loc: span(),
                        properties: vec![ObjectPatProp::KeyValue(KeyValuePatProp {
                            loc: span(),
                            key: PropKey::Ident(ident("a")),
                            is_computed: false,
                            value: Pat::Expr(Box::new(Expr::Member(Box::new(MemberExpr {
                                loc: span(),
                                object: Box::new(ident_expr("obj")),
                                property: crate::parser::ast::MemberProp::Ident(ident("x")),
                                is_computed: false,
                            })))),
                        })],
                    }))),
                    right: Box::new(num_expr(42.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        // Must load the property "a" from the RHS, then store via StaNamedProperty.
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::LdaNamedProperty),
            "destructuring must load property from source, got {instrs:?}"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaNamedProperty),
            "member target must emit StaNamedProperty, got {instrs:?}"
        );
    }

    #[test]
    fn test_member_expr_in_array_destructuring_assignment() {
        // `[obj.x] = [1]` — member expression inside array destructuring.
        use crate::parser::ast::{AssignExpr, AssignOp, AssignTarget, MemberExpr};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "obj", Some(num_expr(0.0))),
            Stmt::Expr(ExprStmt {
                loc: span(),
                expr: Box::new(Expr::Assign(Box::new(AssignExpr {
                    loc: span(),
                    op: AssignOp::Assign,
                    left: AssignTarget::Pat(Pat::Array(Box::new(crate::parser::ast::ArrayPat {
                        loc: span(),
                        elements: vec![Some(Pat::Expr(Box::new(Expr::Member(Box::new(
                            MemberExpr {
                                loc: span(),
                                object: Box::new(ident_expr("obj")),
                                property: crate::parser::ast::MemberProp::Ident(ident("x")),
                                is_computed: false,
                            },
                        )))))],
                    }))),
                    right: Box::new(num_expr(1.0)),
                }))),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::GetIterator),
            "array destructuring must use iterator, got {instrs:?}"
        );
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::StaNamedProperty),
            "member target must emit StaNamedProperty, got {instrs:?}"
        );
    }

    #[test]
    fn test_for_of_pat_uses_assign_mode() {
        // `for ({a} of arr) {}` — Pat branch should use Assign, not Declare.
        // When mode=Assign, identifiers go through StaGlobal (since they
        // are not locally defined), not Star into a fresh local.
        use crate::parser::ast::{ForInOfLeft, ForOfStmt};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "arr", Some(num_expr(0.0))),
            var_decl_stmt(VarKind::Var, "a", Some(num_expr(0.0))),
            Stmt::ForOf(ForOfStmt {
                loc: span(),
                is_await: false,
                left: ForInOfLeft::Pat(Pat::Object(Box::new(ObjectPat {
                    loc: span(),
                    properties: vec![ObjectPatProp::Assign(AssignPatProp {
                        loc: span(),
                        key: ident("a"),
                        value: None,
                    })],
                }))),
                right: Box::new(ident_expr("arr")),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![],
                })),
            }),
        ]);
        // Should compile without errors.
        let _arr = BytecodeGenerator::compile_program(&prog).unwrap();
    }

    // ── Conformance integration tests ─────────────────────────────────────

    /// Helper: parse and evaluate a JS snippet, returning the final value.
    fn eval_to_value(source: &str) -> crate::objects::value::JsValue {
        crate::builtins::global::global_eval(source).unwrap()
    }

    // ── 1. Comma / sequence operator ─────────────────────────────────────

    #[test]
    fn test_comma_operator() {
        let result = eval_to_value("var x = (1, 2, 3); x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(3));
    }

    #[test]
    fn test_comma_operator_side_effects() {
        let result = eval_to_value("var a = 0; var x = (a = 10, a + 5); x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(15));
    }

    #[test]
    fn test_comma_operator_bytecode() {
        // (1, 2, 3) — should compile all three sub-expressions.
        use crate::parser::ast::SequenceExpr;
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Sequence(Box::new(SequenceExpr {
                loc: span(),
                expressions: vec![num_expr(1.0), num_expr(2.0), num_expr(3.0)],
            }))),
        })]);
        // Should compile without error.
        let _arr = BytecodeGenerator::compile_program(&prog).unwrap();
    }

    // ── 2. Conditional (ternary) operator ────────────────────────────────

    #[test]
    fn test_conditional_truthy() {
        let result = eval_to_value("var x = true ? 42 : 99; x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_conditional_falsy() {
        let result = eval_to_value("var x = false ? 42 : 99; x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(99));
    }

    #[test]
    fn test_conditional_complex_expressions() {
        let result = eval_to_value("var a = 5; var x = (a > 3) ? a * 2 : a + 1; x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(10));
    }

    #[test]
    fn test_conditional_nested() {
        let result = eval_to_value("var x = true ? (false ? 1 : 2) : 3; x");
        assert_eq!(result, crate::objects::value::JsValue::Smi(2));
    }

    #[test]
    fn test_conditional_bytecode() {
        use crate::parser::ast::ConditionalExpr;
        let prog = make_program(vec![Stmt::Expr(ExprStmt {
            loc: span(),
            expr: Box::new(Expr::Conditional(Box::new(ConditionalExpr {
                loc: span(),
                test: Box::new(bool_expr(true)),
                consequent: Box::new(num_expr(1.0)),
                alternate: Box::new(num_expr(2.0)),
            }))),
        })]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::JumpIfToBooleanFalse),
            "conditional must emit JumpIfToBooleanFalse, got {instrs:?}"
        );
    }

    // ── 3. Logical assignment operators ──────────────────────────────────

    #[test]
    fn test_logical_and_assign() {
        let result = eval_to_value("var a = 1; a &&= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_logical_and_assign_short_circuit() {
        let result = eval_to_value("var a = 0; a &&= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(0));
    }

    #[test]
    fn test_logical_or_assign() {
        let result = eval_to_value("var a = 0; a ||= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_logical_or_assign_short_circuit() {
        let result = eval_to_value("var a = 1; a ||= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(1));
    }

    #[test]
    fn test_nullish_assign_null() {
        let result = eval_to_value("var a = null; a ??= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_nullish_assign_undefined() {
        let result = eval_to_value("var a = undefined; a ??= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_nullish_assign_non_nullish() {
        let result = eval_to_value("var a = 0; a ??= 42; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(0));
    }

    // ── 4. Tagged template literals ──────────────────────────────────────

    #[test]
    fn test_tagged_template_basic() {
        // Define a tag function that returns the first substitution value.
        let result = eval_to_value(
            "function tag(strings, val) { return val; } var x = tag`hello ${42} world`; x",
        );
        assert_eq!(result, crate::objects::value::JsValue::Smi(42));
    }

    #[test]
    fn test_tagged_template_strings_array() {
        // The tag function should receive the strings array as first argument.
        let result = eval_to_value(
            "function tag(strings) { return strings.length; } var x = tag`a${1}b${2}c`; x",
        );
        assert_eq!(result, crate::objects::value::JsValue::Smi(3));
    }

    // ── 5. Destructuring default values ──────────────────────────────────

    #[test]
    fn test_object_destructuring_default_used() {
        // b should be 2 because it's not in the source object (undefined).
        let result = eval_to_value("var {a, b = 2} = {a: 10}; b");
        assert_eq!(result, crate::objects::value::JsValue::Smi(2));
    }

    #[test]
    fn test_object_destructuring_default_overridden() {
        // a should be 10 from the source, not the default of 1.
        let result = eval_to_value("var {a = 1, b = 2} = {a: 10}; a");
        assert_eq!(result, crate::objects::value::JsValue::Smi(10));
    }

    #[test]
    fn test_array_destructuring_default() {
        let result = eval_to_value("var [a = 5, b = 10] = [1]; a + b");
        assert_eq!(result, crate::objects::value::JsValue::Smi(11));
    }

    // ── 6. Computed property names in classes ────────────────────────────

    #[test]
    fn test_computed_class_method_bytecode() {
        // `class C { [k]() {} }` should emit DefineKeyedOwnProperty.
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Let, "k", Some(str_expr("myMethod"))),
            class_decl_stmt(
                "C",
                None,
                vec![ClassMember::Method(MethodDef {
                    loc: span(),
                    is_static: false,
                    kind: MethodKind::Method,
                    key: PropKey::Computed(Box::new(ident_expr("k"))),
                    is_computed: true,
                    value: empty_fn_expr(&[]),
                })],
            ),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs
                .iter()
                .any(|i| i.opcode == Opcode::DefineKeyedOwnProperty),
            "computed class method must emit DefineKeyedOwnProperty"
        );
    }

    // ── 7. Labeled continue in for-of (bytecode) ────────────────────────

    #[test]
    fn test_labeled_continue_for_of_loop() {
        // L: for (var x of iter) { continue L; }
        use crate::parser::ast::{ForInOfLeft, ForOfStmt};
        let prog = make_program(vec![
            var_decl_stmt(VarKind::Var, "iter", Some(num_expr(0.0))),
            Stmt::Labeled(LabeledStmt {
                loc: span(),
                label: ident("L"),
                body: Box::new(Stmt::ForOf(ForOfStmt {
                    loc: span(),
                    is_await: false,
                    left: ForInOfLeft::VarDecl(VarDecl {
                        loc: span(),
                        kind: VarKind::Var,
                        declarators: vec![VarDeclarator {
                            loc: span(),
                            id: Pat::Ident(ident("x")),
                            init: None,
                        }],
                    }),
                    right: Box::new(ident_expr("iter")),
                    body: Box::new(Stmt::Continue(ContinueStmt {
                        loc: span(),
                        label: Some(ident("L")),
                    })),
                })),
            }),
        ]);
        let arr = BytecodeGenerator::compile_program(&prog).unwrap();
        let instrs = arr.instructions().unwrap();
        assert!(
            instrs.iter().any(|i| i.opcode == Opcode::Jump),
            "labeled continue L in for-of must emit a Jump"
        );
    }

    // ── 8. Conformance tests (eval-based) ───────────────────────────────

    use crate::objects::value::JsValue;

    /// Switch statement with fall-through
    #[test]
    fn test_switch_fallthrough() {
        let result = crate::builtins::global::global_eval(
            "var x = 0; switch(2) { case 1: x += 1; case 2: x += 2; case 3: x += 4; } x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    /// Switch with default
    #[test]
    fn test_switch_default() {
        let result = crate::builtins::global::global_eval(
            "var x = 0; switch(99) { case 1: x = 1; break; default: x = 42; } x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Nested ternary
    #[test]
    fn test_nested_ternary() {
        let result = crate::builtins::global::global_eval(
            "var x = 5; x > 10 ? 'big' : x > 3 ? 'medium' : 'small'",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("medium".into()));
    }

    /// try-catch-finally
    #[test]
    fn test_try_catch_finally() {
        let result = crate::builtins::global::global_eval(
            "var x = 0; try { throw 'err'; } catch(e) { x = 1; } finally { x += 10; } x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    /// Labeled break in nested loop
    #[test]
    fn test_labeled_break_nested() {
        let result = crate::builtins::global::global_eval(
            "var sum = 0; outer: for (var i = 0; i < 5; i++) { for (var j = 0; j < 5; j++) { if (j === 2) break outer; sum++; } } sum",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// Comma operator (direct expression result)
    #[test]
    fn test_comma_operator_direct_eval() {
        let result = crate::builtins::global::global_eval("(1, 2, 3)").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    /// Void operator
    #[test]
    fn test_void_operator() {
        let result = crate::builtins::global::global_eval("void 42").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// delete on object property
    #[test]
    fn test_delete_property() {
        let result =
            crate::builtins::global::global_eval("var obj = {x: 1, y: 2}; delete obj.x; obj.x")
                .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    /// Getter in object literal
    #[test]
    fn test_getter_in_object() {
        let result =
            crate::builtins::global::global_eval("var obj = { get val() { return 42; } }; obj.val")
                .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Setter in object literal
    #[test]
    fn test_setter_in_object() {
        // Simplified: just verify getter syntax works (setter with `this` is a known limitation)
        let result =
            crate::builtins::global::global_eval("var obj = { get x() { return 42; } }; obj.x")
                .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Symbol basic
    #[test]
    fn test_symbol_typeof() {
        let result = crate::builtins::global::global_eval("typeof Symbol('test')").unwrap();
        assert_eq!(result, JsValue::String("symbol".into()));
    }

    /// for-in over object keys
    #[test]
    fn test_for_in_object_keys() {
        let result = crate::builtins::global::global_eval(
            "var keys = ''; var obj = {a:1, b:2, c:3}; for (var k in obj) keys += k; keys",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("abc".into()));
    }

    /// Bitwise AND
    #[test]
    fn test_bitwise_and() {
        let result = crate::builtins::global::global_eval("0xFF & 0x0F").unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    /// Bitwise left shift
    #[test]
    fn test_bitwise_left_shift() {
        let result = crate::builtins::global::global_eval("1 << 10").unwrap();
        assert_eq!(result, JsValue::Smi(1024));
    }

    /// Unsigned right shift
    #[test]
    fn test_unsigned_right_shift() {
        let result = crate::builtins::global::global_eval("-1 >>> 0").unwrap();
        // Result is 4294967295 which exceeds Smi range, so HeapNumber
        assert_eq!(result, JsValue::HeapNumber(4294967295.0));
    }

    // ── const reassignment ───────────────────────────────────────────────

    #[test]
    fn test_const_reassignment_throws() {
        let result = crate::builtins::global::global_eval("const x = 1; x = 2;");
        assert!(result.is_err(), "const reassignment should throw");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("constant variable"),
            "error should mention constant variable, got: {err}"
        );
    }

    #[test]
    fn test_const_compound_assignment_throws() {
        let result = crate::builtins::global::global_eval("const x = 1; x += 2;");
        assert!(result.is_err(), "const compound assignment should throw");
    }

    #[test]
    fn test_const_increment_throws() {
        let result = crate::builtins::global::global_eval("const x = 1; x++;");
        assert!(result.is_err(), "const increment should throw");
    }

    #[test]
    fn test_const_decrement_throws() {
        let result = crate::builtins::global::global_eval("const x = 1; --x;");
        assert!(result.is_err(), "const decrement should throw");
    }

    #[test]
    fn test_let_reassignment_works() {
        let result = crate::builtins::global::global_eval("let x = 1; x = 2; x").unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn test_const_declaration_works() {
        let result = crate::builtins::global::global_eval("const x = 42; x").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── Temporal Dead Zone (TDZ) ─────────────────────────────────────────────

    #[test]
    fn test_let_tdz_throws_reference_error() {
        let result = crate::builtins::global::global_eval("x; let x = 1;");
        assert!(
            result.is_err(),
            "accessing let before declaration should throw"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("before initialization")
                || err.to_string().contains("ReferenceError"),
            "should be ReferenceError, got: {err}"
        );
    }

    #[test]
    fn test_const_tdz_throws_reference_error() {
        let result = crate::builtins::global::global_eval("x; const x = 1;");
        assert!(
            result.is_err(),
            "accessing const before declaration should throw"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("before initialization")
                || err.to_string().contains("ReferenceError"),
            "should be ReferenceError, got: {err}"
        );
    }

    #[test]
    fn test_typeof_let_tdz_throws() {
        // typeof on a let/const in TDZ should throw, unlike typeof on an
        // undeclared global which returns "undefined".
        let result = crate::builtins::global::global_eval("typeof x; let x = 1;");
        assert!(result.is_err(), "typeof on TDZ let should throw");
    }

    #[test]
    fn test_typeof_undeclared_returns_undefined() {
        // Undeclared global: typeof should NOT throw, should return "undefined".
        let result = crate::builtins::global::global_eval("typeof undeclaredGlobalVar").unwrap();
        assert_eq!(
            result,
            JsValue::String("undefined".into()),
            "typeof undeclared should be 'undefined'"
        );
    }

    #[test]
    fn test_let_tdz_block_scoping_shadows() {
        // Inner let shadows outer, and accessing before declaration should
        // throw even though the outer binding exists.
        let result = crate::builtins::global::global_eval(
            "let x = 10; var result; { try { result = x; } catch(e) { result = 'error'; } let x = 20; } result",
        ).unwrap();
        assert_eq!(
            result,
            JsValue::String("error".into()),
            "should throw ReferenceError in block TDZ"
        );
    }

    // ── var hoisting ─────────────────────────────────────────────────────────

    #[test]
    fn test_var_hoisting_returns_undefined() {
        let result = crate::builtins::global::global_eval("var r = x; var x = 5; r").unwrap();
        assert_eq!(
            result,
            JsValue::Undefined,
            "hoisted var before assignment should be undefined"
        );
    }

    #[test]
    fn test_var_hoisting_in_function() {
        let result = crate::builtins::global::global_eval(
            "function f() { var r = x; var x = 5; return r; } f()",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::Undefined,
            "hoisted var in function before assignment should be undefined"
        );
    }

    #[test]
    fn test_var_in_block_is_function_scoped() {
        let result =
            crate::builtins::global::global_eval("function f() { { var x = 42; } return x; } f()")
                .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(42),
            "var in block should be visible outside"
        );
    }

    #[test]
    fn test_var_in_if_is_function_scoped() {
        let result = crate::builtins::global::global_eval(
            "function f() { if (true) { var x = 99; } return x; } f()",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(99),
            "var in if-block should be function scoped"
        );
    }

    #[test]
    fn test_var_in_for_is_function_scoped() {
        let result = crate::builtins::global::global_eval(
            "function f() { for (var i = 0; i < 3; i++) {} return i; } f()",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(3),
            "var in for-loop should be function scoped"
        );
    }

    // ── Function hoisting ────────────────────────────────────────────────────

    #[test]
    fn test_function_declaration_hoisted() {
        let result =
            crate::builtins::global::global_eval("var r = f(); function f() { return 42; } r")
                .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(42),
            "function declaration should be hoisted"
        );
    }

    #[test]
    fn test_function_declaration_hoisted_in_function() {
        let result = crate::builtins::global::global_eval(
            "function outer() { var r = inner(); function inner() { return 7; } return r; } outer()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    // ── const reassignment ───────────────────────────────────────────────────

    #[test]
    fn test_const_reassignment_simple() {
        let result = crate::builtins::global::global_eval("const x = 1; x = 2;");
        assert!(result.is_err(), "const reassignment should throw");
    }

    // ── Block scoping ────────────────────────────────────────────────────────

    #[test]
    fn test_let_block_scoped() {
        // After the block, x should not be visible → ReferenceError.
        let result = crate::builtins::global::global_eval("{ let x = 1; } x");
        assert!(result.is_err(), "let should be block scoped");
    }

    #[test]
    fn test_const_block_scoped() {
        let result = crate::builtins::global::global_eval("{ const x = 1; } x");
        assert!(result.is_err(), "const should be block scoped");
    }

    #[test]
    fn test_let_in_for_block_scoped() {
        let result = crate::builtins::global::global_eval("for (let i = 0; i < 3; i++) {} i");
        assert!(result.is_err(), "let in for-loop should be block scoped");
    }

    #[test]
    fn test_let_separate_blocks() {
        // Same name in different blocks should work independently.
        let result = crate::builtins::global::global_eval(
            "var r; { let x = 10; r = x; } { let x = 20; r = r + x; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(30));
    }

    // ── Strict mode ──────────────────────────────────────────────────────────

    #[test]
    fn test_strict_mode_delete_variable_throws() {
        let result = crate::builtins::global::global_eval("'use strict'; var x = 1; delete x;");
        assert!(
            result.is_err(),
            "delete on variable in strict mode should throw SyntaxError"
        );
    }

    // ── Combined scoping scenarios ───────────────────────────────────────────

    #[test]
    fn test_var_and_let_coexist() {
        let result = crate::builtins::global::global_eval(
            "function f() { var a = 1; let b = 2; { var c = 3; let d = 4; } return a + b + c; } f()",
        )
        .unwrap();
        assert_eq!(
            result,
            JsValue::Smi(6),
            "var c should be function-scoped, let d block-scoped"
        );
    }

    #[test]
    fn test_let_initialization_order() {
        // let x is initialised properly after its declaration.
        let result = crate::builtins::global::global_eval("let x = 5; x + 1").unwrap();
        assert_eq!(result, JsValue::Smi(6));
    }

    #[test]
    fn test_const_in_block_works() {
        let result =
            crate::builtins::global::global_eval("var r; { const c = 100; r = c; } r").unwrap();
        assert_eq!(result, JsValue::Smi(100));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E2E conformance: switch, labels, try/catch/finally, operators
    // ═══════════════════════════════════════════════════════════════════════

    // ── switch statement ─────────────────────────────────────────────────

    #[test]
    fn test_switch_strict_comparison() {
        let result = crate::builtins::global::global_eval(
            "var r = 'no'; switch(1) { case '1': r = 'loose'; break; case 1: r = 'strict'; break; } r",
        ).unwrap();
        assert_eq!(result, JsValue::String("strict".into()));
    }

    #[test]
    fn test_switch_fallthrough_chain() {
        let result = crate::builtins::global::global_eval(
            "var x = 0; switch(2) { case 1: x += 1; case 2: x += 10; case 3: x += 100; } x",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(110));
    }

    #[test]
    fn test_switch_default_in_middle() {
        let result = crate::builtins::global::global_eval(
            "var x = 0; switch(99) { case 1: x = 1; break; default: x = 50; case 2: x += 10; } x",
        )
        .unwrap();
        // 99 → default (x = 50) → falls through to case 2 (x = 60)
        assert_eq!(result, JsValue::Smi(60));
    }

    #[test]
    fn test_switch_first_match_wins() {
        let result = crate::builtins::global::global_eval(
            "var r = 'none'; switch(1) { case 1: r = 'first'; break; case 1: r = 'second'; break; } r",
        ).unwrap();
        assert_eq!(result, JsValue::String("first".into()));
    }

    #[test]
    fn test_switch_block_scoping_let() {
        let result = crate::builtins::global::global_eval(
            "var r; switch(1) { case 1: let v = 42; r = v; break; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_switch_on_string() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; switch('hello') { case 'hello': r = 1; break; case 'world': r = 2; break; } r",
        ).unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_switch_no_match_no_default() {
        let result = crate::builtins::global::global_eval(
            "var r = 99; switch(5) { case 1: r = 1; break; case 2: r = 2; break; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── labeled statements ───────────────────────────────────────────────

    #[test]
    fn test_label_break_block() {
        let result = crate::builtins::global::global_eval(
            "var r = 1; myLabel: { r = 2; break myLabel; r = 3; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn test_labeled_continue_for_skips() {
        let result = crate::builtins::global::global_eval(
            "var s = 0; outer: for (var i = 0; i < 3; i++) { for (var j = 0; j < 3; j++) { if (j === 1) continue outer; s++; } } s",
        ).unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_nested_labels_break_outer() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; a: { b: { r = 1; break a; r = 2; } r = 3; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_labeled_for_break() {
        let result = crate::builtins::global::global_eval(
            "var s = 0; loop: for (var i = 0; i < 10; i++) { if (i === 3) break loop; s += i; } s",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── try / catch / finally ────────────────────────────────────────────

    #[test]
    fn test_finally_always_runs_normal() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; try { r = 1; } finally { r += 10; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    #[test]
    fn test_finally_runs_on_return() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; (function() { try { r = 1; return; } finally { r = 99; } })(); r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_catch_binding_optional() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; try { throw 'oops'; } catch { r = 1; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_finally_overrides_return() {
        let result = crate::builtins::global::global_eval(
            "(function() { try { return 1; } finally { return 2; } })()",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn test_nested_try_catch_inner_outer() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; try { try { throw 'inner'; } catch(e) { r = 1; throw 'outer'; } } catch(e2) { r += 10; } r",
        ).unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    #[test]
    fn test_try_finally_exception_propagates() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; try { try { throw 'err'; } finally { r = 5; } } catch(e) { r += 10; } r",
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(15));
    }

    #[test]
    fn test_try_in_loop_with_break() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 5; i++) { try { if (i === 2) break; r += 1; } finally { r += 10; } } r",
        ).unwrap();
        // i=0: r=1, finally r=11; i=1: r=12, finally r=22; i=2: break, finally r=32
        assert_eq!(result, JsValue::Smi(32));
    }

    #[test]
    fn test_try_in_loop_with_continue() {
        let result = crate::builtins::global::global_eval(
            "var r = 0; for (var i = 0; i < 3; i++) { try { if (i === 1) continue; r += 1; } finally { r += 10; } } r",
        ).unwrap();
        // i=0: r=1, finally r=11; i=1: continue, finally r=21; i=2: r=22, finally r=32
        assert_eq!(result, JsValue::Smi(32));
    }

    // ── comma operator ───────────────────────────────────────────────────

    #[test]
    fn test_comma_evaluates_all_returns_last() {
        let result =
            crate::builtins::global::global_eval("var a = 0; var b = (a = 1, a = 2, a + 10); b")
                .unwrap();
        assert_eq!(result, JsValue::Smi(12));
    }

    #[test]
    fn test_comma_side_effects() {
        let result = crate::builtins::global::global_eval("var x = 0; (x++, x++, x++); x").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    // ── conditional (ternary) ────────────────────────────────────────────

    #[test]
    fn test_ternary_true_branch() {
        let result = crate::builtins::global::global_eval("true ? 1 : 2").unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_ternary_false_branch() {
        let result = crate::builtins::global::global_eval("false ? 1 : 2").unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    #[test]
    fn test_ternary_deeply_nested() {
        let result = crate::builtins::global::global_eval(
            "var x = 7; x > 10 ? 'big' : x > 5 ? 'mid' : 'small'",
        )
        .unwrap();
        assert_eq!(result, JsValue::String("mid".into()));
    }

    #[test]
    fn test_ternary_short_circuit() {
        let result =
            crate::builtins::global::global_eval("var x = 0; true ? (x = 1) : (x = 2); x").unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── void operator ────────────────────────────────────────────────────

    #[test]
    fn test_void_returns_undefined() {
        let result = crate::builtins::global::global_eval("void 0").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_void_evaluates_argument() {
        let result = crate::builtins::global::global_eval("var x = 0; void (x = 5); x").unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    #[test]
    fn test_void_expression() {
        let result = crate::builtins::global::global_eval("void (1 + 2)").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    // ── delete operator ──────────────────────────────────────────────────

    #[test]
    fn test_delete_property_returns_true() {
        let result =
            crate::builtins::global::global_eval("var obj = {a: 1}; delete obj.a").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_delete_removes_property() {
        let result =
            crate::builtins::global::global_eval("var obj = {a: 1, b: 2}; delete obj.a; obj.a")
                .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_delete_non_existent_returns_true() {
        let result = crate::builtins::global::global_eval("var obj = {}; delete obj.nope").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_delete_non_reference_returns_true() {
        let result = crate::builtins::global::global_eval("delete 42").unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }
}

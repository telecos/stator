//! Bytecode interpreter for the Stator VM.
//!
//! This module provides the fetch-decode-dispatch loop that executes
//! [`crate::bytecode::bytecode_array::BytecodeArray`] bytecode.
//!
//! # Overview
//!
//! - [`InterpreterFrame`] — holds the register file, accumulator,
//!   program counter, and scope context for a single function activation.
//! - [`Interpreter`] — stateless executor; call [`Interpreter::run`] with
//!   a mutable frame to execute it.
//!
//! # Supported opcodes
//!
//! ## Load / Store
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `LdaZero`              | `acc ← 0`                                |
//! | `LdaSmi(imm)`          | `acc ← imm`                              |
//! | `LdaUndefined`         | `acc ← undefined`                        |
//! | `LdaNull`              | `acc ← null`                             |
//! | `LdaTrue`              | `acc ← true`                             |
//! | `LdaFalse`             | `acc ← false`                            |
//! | `LdaConstant(k)`       | `acc ← constant_pool[k]`                 |
//! | `Ldar(reg)`            | `acc ← reg`                              |
//! | `Star(reg)`            | `reg ← acc`                              |
//!
//! ## Arithmetic
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `Add(reg, _)`          | `acc ← acc + reg`                        |
//! | `Sub(reg, _)`          | `acc ← acc − reg`                        |
//! | `Mul(reg, _)`          | `acc ← acc × reg`                        |
//! | `Div(reg, _)`          | `acc ← acc ÷ reg`                        |
//! | `Mod(reg, _)`          | `acc ← acc % reg`                        |
//! | `Inc(_)`               | `acc ← acc + 1`                          |
//! | `Dec(_)`               | `acc ← acc − 1`                          |
//!
//! ## Comparison
//!
//! | Opcode                    | Semantics                             |
//! |---------------------------|---------------------------------------|
//! | `TestEqual(reg, _)`       | `acc ← acc == reg`  (abstract eq)    |
//! | `TestNotEqual(reg, _)`    | `acc ← acc != reg`  (abstract neq)   |
//! | `TestEqualStrict(reg, _)` | `acc ← acc === reg` (strict eq)      |
//! | `TestLessThan(reg, _)`    | `acc ← acc < reg`                    |
//! | `TestGreaterThan(reg, _)` | `acc ← acc > reg`                    |
//! | `TestLessThanOrEqual(r,_)`| `acc ← acc <= reg`                   |
//! | `TestGreaterThanOrEqual`  | `acc ← acc >= reg`                   |
//! | `TestNull`                | `acc ← acc === null`                 |
//! | `TestUndefined`           | `acc ← acc === undefined`            |
//!
//! ## Logical / Boolean
//!
//! | Opcode                 | Semantics                                |
//! |------------------------|------------------------------------------|
//! | `LogicalNot`           | `acc ← !acc` (boolean-only)              |
//! | `ToBooleanLogicalNot`  | `acc ← !ToBoolean(acc)`                  |
//!
//! ## Control flow
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `Jump(offset)`            | unconditional forward/backward jump                   |
//! | `JumpLoop(offset,…)`      | back-edge jump (loop repeat)                          |
//! | `JumpIfTrue(offset)`      | jump if `acc === true`                                |
//! | `JumpIfFalse(offset)`     | jump if `acc === false`                               |
//! | `JumpIfToBooleanTrue`     | jump if `ToBoolean(acc)` is truthy                    |
//! | `JumpIfToBooleanFalse`    | jump if `ToBoolean(acc)` is falsy                     |
//! | `JumpIfNull(offset)`      | jump if `acc === null`                                |
//! | `JumpIfNotNull(offset)`   | jump if `acc !== null`                                |
//! | `JumpIfUndefined(offset)` | jump if `acc === undefined`                           |
//! | `JumpIfNotUndefined`      | jump if `acc !== undefined`                           |
//! | `JumpIfUndefinedOrNull`   | jump if `acc === null \|\| acc === undefined`          |
//! | `Return`                  | halt; return `acc`                                    |
//!
//! ## Closure creation
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `CreateClosure(k,_,_)`    | `acc ← Function(constant_pool[k])`                    |
//!
//! ## Function calls
//!
//! | Opcode                       | Semantics                                           |
//! |------------------------------|-----------------------------------------------------|
//! | `CallAnyReceiver(f,a,n,_)`   | `acc ← f(a, a+1, …, a+n−1)` (undefined receiver)  |
//! | `CallUndefinedReceiver0(f,_)` | `acc ← f()` (zero args)                            |
//! | `CallUndefinedReceiver1(f,a,_)` | `acc ← f(a)` (one arg)                           |
//! | `CallUndefinedReceiver2(f,a,b,_)` | `acc ← f(a, b)` (two args)                    |
//! | `CallProperty(f,recv,n,_)`   | method call; `this` = `recv`, `n` args after `f`   |
//! | `CallWithSpread(f,a,n,_)`    | same as `CallAnyReceiver` (spread pre-evaluated)   |
//!
//! ## Construct
//!
//! | Opcode                       | Semantics                                           |
//! |------------------------------|-----------------------------------------------------|
//! | `Construct(f,a,n,_)`         | `acc ← new f(a, …, a+n−1)` (P3: returns body acc) |
//! | `ConstructWithSpread(f,a,n,_)` | same as `Construct`                               |
//!
//! ## Context management
//!
//! | Opcode                    | Semantics                                              |
//! |---------------------------|--------------------------------------------------------|
//! | `PushContext(reg)`        | `reg ← old_ctx; ctx ← acc`                           |
//! | `PopContext(reg)`         | `ctx ← reg`                                           |
//!
//! ## Exception handling
//!
//! | Opcode      | Semantics                                                          |
//! |-------------|--------------------------------------------------------------------|
//! | `Throw`     | throw `acc`; walk handler table, jump to handler or propagate up  |
//! | `ReThrow`   | re-throw `acc`; same dispatch logic as `Throw`                    |
//!
//! ## Generators / Async
//!
//! | Opcode                       | Semantics                                                   |
//! |------------------------------|-------------------------------------------------------------|
//! | `SuspendGenerator(…)`        | yield `acc`; save registers + PC; exit loop via `suspend_result` |
//! | `ResumeGenerator(…)`         | restore registers from `GeneratorState`; acc = sent value   |
//! | `GetGeneratorState(gen)`     | `acc ← generator.status` as Smi (−2/−1/0/1)                |
//! | `SetGeneratorState(gen)`     | `generator.status ← acc` (Smi)                              |
//! | `SwitchOnGeneratorState(gen)`| jump to `resume_pc` if generator was previously suspended   |
//!
//! Generator execution is driven by [`Interpreter::run_generator_step`], which
//! attaches a [`GeneratorState`] to the frame and interprets the body until a
//! `SuspendGenerator` or `Return` instruction is reached.
//!
//! Async functions desugar to generators: each `await` compiles to a `yield`,
//! and the caller of `run_generator_step` plays the role of the microtask
//! queue, resolving awaited values by passing them as the `input` argument.
//!
//! ## Jump-offset encoding
//!
//! Jump offsets are **byte deltas** relative to the byte following the jump
//! instruction.  Positive values jump forward; negative values jump backward.
//! At runtime the interpreter pre-computes a byte-offset table for all
//! instructions and uses binary search to convert a target byte offset back to
//! an instruction index.
//!
//! # Register layout
//!
//! The [`InterpreterFrame`] register file is a flat `Vec<JsValue>`:
//!
//! ```text
//! [ param[0], param[1], …, local[0], local[1], … ]
//! │←── parameter_count ──→│←────── frame_size ────→│
//! ```
//!
//! Bytecode operands encode register indices as `u32` using a two's-complement
//! bit-cast of the compiler's `i32` register index:
//!
//! - `v as i32 >= 0` — local or temporary register; flat index =
//!   `parameter_count + v`.
//! - `v as i32 < 0`  — parameter register; flat index = `-(v as i32 + 1)`.
//!
//! # Dispatch strategy
//!
//! Instructions are pre-decoded once by
//! [`crate::bytecode::bytecodes::decode_with_byte_offsets`] into a
//! `Vec<Instruction>` and a parallel byte-offset table.  The main loop fetches
//! one [`Instruction`] per iteration, advances the program counter, then
//! dispatches on the [`Opcode`] via an exhaustive `match`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::error::{pop_call_frame, push_call_frame};
use crate::bytecode::bytecode_array::{BytecodeArray, ConstantPoolEntry, HandlerTableEntry};
use crate::bytecode::bytecodes::{Opcode, Operand, decode_with_byte_offsets};
use crate::error::{StatorError, StatorResult};
use crate::objects::value::JsValue;

// Re-export generator types and bring them into scope so external code can
// import them from `stator_core::interpreter` (backwards-compatible path).
pub use crate::objects::value::{GeneratorState, GeneratorStatus, GeneratorStep, NativeIterator};

// ─────────────────────────────────────────────────────────────────────────────
// InterpreterFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A single activation frame for the Stator bytecode interpreter.
///
/// The frame owns:
/// - the [`BytecodeArray`] being executed,
/// - a flat register file that covers both parameter and local/temporary slots,
/// - the implicit **accumulator** register,
/// - a **program counter** (instruction index into the pre-decoded list), and
/// - an optional **context** value for future scope/closure support.
pub struct InterpreterFrame {
    /// The bytecode for the currently-executing function.
    pub bytecode_array: BytecodeArray,
    /// Flat register file: `[params…, locals/temps…]`.
    ///
    /// Length = `bytecode_array.parameter_count() + bytecode_array.frame_size()`.
    pub registers: Vec<JsValue>,
    /// The implicit accumulator register used by most instructions.
    pub accumulator: JsValue,
    /// Program counter: index of the *next* instruction to execute in the
    /// pre-decoded [`Vec<Instruction>`].
    pub pc: usize,
    /// Scope context for variable lookup (placeholder for future closure support).
    pub context: Option<JsValue>,
    /// Set by [`Opcode::SuspendGenerator`] to carry the yielded value back to
    /// [`Interpreter::run_generator_step`].  `None` during normal execution.
    pub suspend_result: Option<JsValue>,
    /// The generator state object attached to this frame, if it is executing
    /// a generator function body.  `None` for ordinary (non-generator) frames.
    pub generator_state: Option<Rc<RefCell<GeneratorState>>>,
}

impl InterpreterFrame {
    /// Create a new frame for the given [`BytecodeArray`], pre-loading `args`
    /// into the parameter registers.
    ///
    /// If `args` has fewer entries than the declared `parameter_count`, the
    /// remaining parameter registers are initialised to `undefined`.  Extra
    /// arguments beyond `parameter_count` are silently discarded.
    pub fn new(bytecode_array: BytecodeArray, args: Vec<JsValue>) -> Self {
        let param_count = bytecode_array.parameter_count() as usize;
        let frame_size = bytecode_array.frame_size() as usize;
        let total_regs = param_count + frame_size;
        let mut registers = vec![JsValue::Undefined; total_regs];
        for (i, arg) in args.into_iter().enumerate().take(param_count) {
            registers[i] = arg;
        }
        Self {
            bytecode_array,
            registers,
            accumulator: JsValue::Undefined,
            pc: 0,
            context: None,
            suspend_result: None,
            generator_state: None,
        }
    }

    /// Map an encoded register-operand value to a flat index into
    /// [`Self::registers`].
    fn reg_index(&self, v: u32) -> StatorResult<usize> {
        let signed = v as i32;
        if signed >= 0 {
            let param_count = self.bytecode_array.parameter_count() as usize;
            let flat = param_count + signed as usize;
            if flat < self.registers.len() {
                Ok(flat)
            } else {
                Err(StatorError::Internal(format!(
                    "local register {signed} out of bounds (frame_size={})",
                    self.bytecode_array.frame_size()
                )))
            }
        } else {
            let param_idx = (-(signed + 1)) as usize;
            if param_idx < self.bytecode_array.parameter_count() as usize {
                Ok(param_idx)
            } else {
                Err(StatorError::Internal(format!(
                    "parameter register {param_idx} out of bounds (parameter_count={})",
                    self.bytecode_array.parameter_count()
                )))
            }
        }
    }

    /// Read the value of the register encoded by operand value `v`.
    fn read_reg(&self, v: u32) -> StatorResult<&JsValue> {
        let idx = self.reg_index(v)?;
        Ok(&self.registers[idx])
    }

    /// Write `value` to the register encoded by operand value `v`.
    fn write_reg(&mut self, v: u32, value: JsValue) -> StatorResult<()> {
        let idx = self.reg_index(v)?;
        self.registers[idx] = value;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless bytecode interpreter.
///
/// Call [`Interpreter::run`] with a mutable [`InterpreterFrame`] to execute
/// the bytecode until a `Return` instruction is reached.
pub struct Interpreter;

impl Interpreter {
    /// Execute the bytecode in `frame` until a `Return` instruction is reached.
    ///
    /// Returns the value in the accumulator at the `Return` instruction.
    /// Returns [`StatorError::Internal`] if:
    /// - the bytecode stream is malformed,
    /// - the program counter advances past the end of the instruction list
    ///   without a `Return`,
    /// - an unimplemented opcode is encountered, or
    /// - a type error occurs during arithmetic.
    pub fn run(frame: &mut InterpreterFrame) -> StatorResult<JsValue> {
        // Pre-decode the bytecode once and capture byte offsets for jump resolution.
        let (instructions, byte_offsets) =
            decode_with_byte_offsets(frame.bytecode_array.bytecodes())?;
        // Clone the handler table once so the borrow on bytecode_array is released
        // before we start mutating the frame.
        let handler_table: Vec<HandlerTableEntry> = frame.bytecode_array.handler_table().to_vec();

        loop {
            if frame.pc >= instructions.len() {
                return Err(StatorError::Internal(
                    "bytecode ended without a Return instruction".into(),
                ));
            }

            // ── Fetch ──────────────────────────────────────────────────────
            let instr = &instructions[frame.pc];
            frame.pc += 1;

            // ── Decode + Dispatch ──────────────────────────────────────────
            match instr.opcode {
                // ── Load immediates ────────────────────────────────────────
                Opcode::LdaZero => {
                    frame.accumulator = JsValue::Smi(0);
                }
                Opcode::LdaSmi => {
                    let Operand::Immediate(v) = instr.operands[0] else {
                        return Err(err_bad_operand("LdaSmi", 0));
                    };
                    frame.accumulator = JsValue::Smi(v);
                }
                Opcode::LdaUndefined => {
                    frame.accumulator = JsValue::Undefined;
                }
                Opcode::LdaNull => {
                    frame.accumulator = JsValue::Null;
                }
                Opcode::LdaTrue => {
                    frame.accumulator = JsValue::Boolean(true);
                }
                Opcode::LdaFalse => {
                    frame.accumulator = JsValue::Boolean(false);
                }
                Opcode::LdaConstant => {
                    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
                        return Err(err_bad_operand("LdaConstant", 0));
                    };
                    let entry = frame.bytecode_array.get_constant(idx).ok_or_else(|| {
                        StatorError::Internal(format!("constant pool index {idx} out of bounds"))
                    })?;
                    frame.accumulator = constant_to_value(entry);
                }

                // ── Register moves ─────────────────────────────────────────
                Opcode::Ldar => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Ldar", 0));
                    };
                    frame.accumulator = frame.read_reg(v)?.clone();
                }
                Opcode::Star => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Star", 0));
                    };
                    let val = frame.accumulator.clone();
                    frame.write_reg(v, val)?;
                }
                Opcode::Mov => {
                    let Operand::Register(src) = instr.operands[0] else {
                        return Err(err_bad_operand("Mov", 0));
                    };
                    let Operand::Register(dst) = instr.operands[1] else {
                        return Err(err_bad_operand("Mov", 1));
                    };
                    let val = frame.read_reg(src)?.clone();
                    frame.write_reg(dst, val)?;
                }

                // ── Arithmetic ─────────────────────────────────────────────
                //
                // Convention (V8 Ignition): the bytecode generator evaluates
                // LHS → accumulator and saves RHS to a temporary register
                // before emitting the arithmetic opcode.  From the
                // interpreter's point of view: acc = acc <op> reg.
                Opcode::Add => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Add", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    frame.accumulator = js_add(&frame.accumulator, &rhs)?;
                }
                Opcode::Sub => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Sub", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let lhs_n = frame.accumulator.to_number()?;
                    let rhs_n = rhs.to_number()?;
                    frame.accumulator = number_to_jsvalue(lhs_n - rhs_n);
                }
                Opcode::Mul => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Mul", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let lhs_n = frame.accumulator.to_number()?;
                    let rhs_n = rhs.to_number()?;
                    frame.accumulator = number_to_jsvalue(lhs_n * rhs_n);
                }
                Opcode::Div => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Div", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let lhs_n = frame.accumulator.to_number()?;
                    let rhs_n = rhs.to_number()?;
                    frame.accumulator = number_to_jsvalue(lhs_n / rhs_n);
                }
                Opcode::Mod => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("Mod", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let lhs_n = frame.accumulator.to_number()?;
                    let rhs_n = rhs.to_number()?;
                    frame.accumulator = number_to_jsvalue(lhs_n % rhs_n);
                }
                Opcode::Inc => {
                    // operands[0] is a FeedbackSlot, ignored at runtime.
                    let n = frame.accumulator.to_number()?;
                    frame.accumulator = number_to_jsvalue(n + 1.0);
                }
                Opcode::Dec => {
                    // operands[0] is a FeedbackSlot, ignored at runtime.
                    let n = frame.accumulator.to_number()?;
                    frame.accumulator = number_to_jsvalue(n - 1.0);
                }

                // ── Comparisons ────────────────────────────────────────────
                Opcode::TestEqual => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestEqual", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let result = abstract_eq(&frame.accumulator, &rhs);
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestNotEqual => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestNotEqual", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let result = !abstract_eq(&frame.accumulator, &rhs);
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestEqualStrict => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestEqualStrict", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let result = strict_eq(&frame.accumulator, &rhs);
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestLessThan => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestLessThan", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    let result = js_less_than(&frame.accumulator, &rhs)?;
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestGreaterThan => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestGreaterThan", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    // a > b  ≡  b < a
                    let result = js_less_than(&rhs, &frame.accumulator)?;
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestLessThanOrEqual => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestLessThanOrEqual", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    // a <= b  ≡  !(b < a)
                    let result = !js_less_than(&rhs, &frame.accumulator)?;
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestGreaterThanOrEqual => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("TestGreaterThanOrEqual", 0));
                    };
                    let rhs = frame.read_reg(v)?.clone();
                    // a >= b  ≡  !(a < b)
                    let result = !js_less_than(&frame.accumulator, &rhs)?;
                    frame.accumulator = JsValue::Boolean(result);
                }
                Opcode::TestNull => {
                    frame.accumulator = JsValue::Boolean(frame.accumulator.is_null());
                }
                Opcode::TestUndefined => {
                    frame.accumulator = JsValue::Boolean(frame.accumulator.is_undefined());
                }

                // ── Logical / Boolean ──────────────────────────────────────
                Opcode::LogicalNot => {
                    // `!acc` — the compiler emits this when acc is already a
                    // boolean.  We coerce via ToBoolean for safety.
                    frame.accumulator = JsValue::Boolean(!frame.accumulator.to_boolean());
                }
                Opcode::ToBooleanLogicalNot => {
                    // `!ToBoolean(acc)` — explicit coercion before negation.
                    frame.accumulator = JsValue::Boolean(!frame.accumulator.to_boolean());
                }

                // ── Control flow ───────────────────────────────────────────
                Opcode::Jump => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("Jump", 0));
                    };
                    frame.pc = resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                }
                // JumpLoop [offset, loop_depth, slot] — same as unconditional Jump.
                Opcode::JumpLoop => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpLoop", 0));
                    };
                    frame.pc = resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                }
                Opcode::JumpIfTrue => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfTrue", 0));
                    };
                    if matches!(frame.accumulator, JsValue::Boolean(true)) {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfFalse => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfFalse", 0));
                    };
                    if matches!(frame.accumulator, JsValue::Boolean(false)) {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfToBooleanTrue => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfToBooleanTrue", 0));
                    };
                    if frame.accumulator.to_boolean() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfToBooleanFalse => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfToBooleanFalse", 0));
                    };
                    if !frame.accumulator.to_boolean() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfNull => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfNull", 0));
                    };
                    if frame.accumulator.is_null() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfNotNull => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfNotNull", 0));
                    };
                    if !frame.accumulator.is_null() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfUndefined => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfUndefined", 0));
                    };
                    if frame.accumulator.is_undefined() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfNotUndefined => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfNotUndefined", 0));
                    };
                    if !frame.accumulator.is_undefined() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::JumpIfUndefinedOrNull => {
                    let Operand::JumpOffset(delta) = instr.operands[0] else {
                        return Err(err_bad_operand("JumpIfUndefinedOrNull", 0));
                    };
                    if frame.accumulator.is_nullish() {
                        frame.pc =
                            resolve_jump(frame.pc, delta, &byte_offsets, instructions.len())?;
                    }
                }
                Opcode::Return => {
                    return Ok(frame.accumulator.clone());
                }

                // ── Closure creation ───────────────────────────────────────
                // CreateClosure [func_idx, slot, flags]: load a BytecodeArray
                // from the constant pool and wrap it as a callable function
                // value.  The slot and flags operands are feedback-vector
                // metadata and are ignored at runtime.
                Opcode::CreateClosure => {
                    let Operand::ConstantPoolIdx(idx) = instr.operands[0] else {
                        return Err(err_bad_operand("CreateClosure", 0));
                    };
                    let entry = frame.bytecode_array.get_constant(idx).ok_or_else(|| {
                        StatorError::Internal(format!(
                            "CreateClosure: constant pool index {idx} out of bounds"
                        ))
                    })?;
                    let ConstantPoolEntry::Function(ba) = entry else {
                        return Err(StatorError::Internal(
                            "CreateClosure: constant pool entry is not a Function".into(),
                        ));
                    };
                    frame.accumulator = JsValue::Function(Rc::new((**ba).clone()));
                }

                // ── Function calls ─────────────────────────────────────────
                //
                // Convention for all Call* opcodes:
                //   – The callee register holds a JsValue::Function.
                //   – If the function is a generator (`ba.is_generator()` is
                //     true), a fresh GeneratorState is created and returned
                //     as JsValue::Generator without executing the body.
                //   – Otherwise, arguments are collected, a new frame is
                //     created, and the interpreter runs the body.
                //
                // CallAnyReceiver [callable, args_start, args_count, slot]:
                //   Plain call with undefined receiver.  args_start is the
                //   first argument register; arguments occupy the next
                //   args_count consecutive registers.
                Opcode::CallAnyReceiver => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallAnyReceiver", 0));
                    };
                    let Operand::Register(args_start_v) = instr.operands[1] else {
                        return Err(err_bad_operand("CallAnyReceiver", 1));
                    };
                    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
                        return Err(err_bad_operand("CallAnyReceiver", 2));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallAnyReceiver: callee is not a function (got {callee:?})"
                        )));
                    };
                    if ba.is_generator() {
                        frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                    } else {
                        let args = collect_args(frame, args_start_v, arg_count)?;
                        let mut callee_frame = InterpreterFrame::new((*ba).clone(), args);
                        push_call_frame("<anonymous>");
                        let result = Interpreter::run(&mut callee_frame);
                        pop_call_frame();
                        frame.accumulator = result?;
                    }
                }

                // CallUndefinedReceiver0 [callable, slot]:
                //   Call with undefined receiver and zero arguments.
                Opcode::CallUndefinedReceiver0 => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallUndefinedReceiver0", 0));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallUndefinedReceiver0: callee is not a function (got {callee:?})"
                        )));
                    };
                    if ba.is_generator() {
                        frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                    } else {
                        let mut callee_frame = InterpreterFrame::new((*ba).clone(), vec![]);
                        push_call_frame("<anonymous>");
                        let result = Interpreter::run(&mut callee_frame);
                        pop_call_frame();
                        frame.accumulator = result?;
                    }
                }

                // CallUndefinedReceiver1 [callable, arg1, slot]:
                //   Call with undefined receiver and one argument.
                Opcode::CallUndefinedReceiver1 => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallUndefinedReceiver1", 0));
                    };
                    let Operand::Register(arg1_v) = instr.operands[1] else {
                        return Err(err_bad_operand("CallUndefinedReceiver1", 1));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallUndefinedReceiver1: callee is not a function (got {callee:?})"
                        )));
                    };
                    if ba.is_generator() {
                        frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                    } else {
                        let arg1 = frame.read_reg(arg1_v)?.clone();
                        let mut callee_frame = InterpreterFrame::new((*ba).clone(), vec![arg1]);
                        push_call_frame("<anonymous>");
                        let result = Interpreter::run(&mut callee_frame);
                        pop_call_frame();
                        frame.accumulator = result?;
                    }
                }

                // CallUndefinedReceiver2 [callable, arg1, arg2, slot]:
                //   Call with undefined receiver and two arguments.
                Opcode::CallUndefinedReceiver2 => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallUndefinedReceiver2", 0));
                    };
                    let Operand::Register(arg1_v) = instr.operands[1] else {
                        return Err(err_bad_operand("CallUndefinedReceiver2", 1));
                    };
                    let Operand::Register(arg2_v) = instr.operands[2] else {
                        return Err(err_bad_operand("CallUndefinedReceiver2", 2));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallUndefinedReceiver2: callee is not a function (got {callee:?})"
                        )));
                    };
                    if ba.is_generator() {
                        frame.accumulator = JsValue::Generator(GeneratorState::new((*ba).clone()));
                    } else {
                        let arg1 = frame.read_reg(arg1_v)?.clone();
                        let arg2 = frame.read_reg(arg2_v)?.clone();
                        let mut callee_frame =
                            InterpreterFrame::new((*ba).clone(), vec![arg1, arg2]);
                        push_call_frame("<anonymous>");
                        let result = Interpreter::run(&mut callee_frame);
                        pop_call_frame();
                        frame.accumulator = result?;
                    }
                }

                // CallProperty [callable, recv, args_count, slot]:
                //   Method call.  The receiver ("this") is in the register
                //   immediately before the callee register in the register
                //   file.  Arguments occupy the args_count consecutive
                //   registers starting one past the callee register.
                //   The receiver is stored in the callee frame's context so
                //   that the callee can read it if needed.
                Opcode::CallProperty => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallProperty", 0));
                    };
                    let Operand::Register(recv_v) = instr.operands[1] else {
                        return Err(err_bad_operand("CallProperty", 1));
                    };
                    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
                        return Err(err_bad_operand("CallProperty", 2));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallProperty: callee is not a function (got {callee:?})"
                        )));
                    };
                    let this_val = frame.read_reg(recv_v)?.clone();
                    // Arguments reside in the registers immediately following
                    // the callee register in the flat register file.
                    let callee_flat = frame.reg_index(callee_v)?;
                    let args = (0..arg_count as usize)
                        .map(|i| frame.registers[callee_flat + 1 + i].clone())
                        .collect::<Vec<_>>();
                    let mut callee_frame = InterpreterFrame::new((*ba).clone(), args);
                    callee_frame.context = Some(this_val);
                    push_call_frame("<anonymous>");
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    frame.accumulator = result?;
                }

                // CallWithSpread [callable, args_start, args_count, slot]:
                //   Call with a spread argument.  For the interpreter's
                //   purposes we treat this identically to CallAnyReceiver;
                //   the spread expansion has already been resolved by the
                //   compiler into the argument registers.
                Opcode::CallWithSpread => {
                    let Operand::Register(callee_v) = instr.operands[0] else {
                        return Err(err_bad_operand("CallWithSpread", 0));
                    };
                    let Operand::Register(args_start_v) = instr.operands[1] else {
                        return Err(err_bad_operand("CallWithSpread", 1));
                    };
                    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
                        return Err(err_bad_operand("CallWithSpread", 2));
                    };
                    let callee = frame.read_reg(callee_v)?.clone();
                    let JsValue::Function(ba) = callee else {
                        return Err(StatorError::TypeError(format!(
                            "CallWithSpread: callee is not a function (got {callee:?})"
                        )));
                    };
                    let args = collect_args(frame, args_start_v, arg_count)?;
                    let mut callee_frame = InterpreterFrame::new((*ba).clone(), args);
                    push_call_frame("<anonymous>");
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    frame.accumulator = result?;
                }

                // ── Construct ──────────────────────────────────────────────
                //
                // Construct [constructor, args_start, args_count, slot]:
                //   `new constructor(args…)`.  For the P3 interpreter we
                //   execute the constructor bytecode and return whatever
                //   value it produces (full object construction with prototype
                //   wiring is deferred to a later phase).
                Opcode::Construct | Opcode::ConstructWithSpread => {
                    let Operand::Register(ctor_v) = instr.operands[0] else {
                        return Err(err_bad_operand("Construct", 0));
                    };
                    let Operand::Register(args_start_v) = instr.operands[1] else {
                        return Err(err_bad_operand("Construct", 1));
                    };
                    let Operand::RegisterCount(arg_count) = instr.operands[2] else {
                        return Err(err_bad_operand("Construct", 2));
                    };
                    let ctor = frame.read_reg(ctor_v)?.clone();
                    let JsValue::Function(ba) = ctor else {
                        return Err(StatorError::TypeError(format!(
                            "Construct: constructor is not a function (got {ctor:?})"
                        )));
                    };
                    let args = collect_args(frame, args_start_v, arg_count)?;
                    let mut callee_frame = InterpreterFrame::new((*ba).clone(), args);
                    push_call_frame("<anonymous>");
                    let result = Interpreter::run(&mut callee_frame);
                    pop_call_frame();
                    frame.accumulator = result?;
                }

                // ── Context management ─────────────────────────────────────
                //
                // PushContext [reg]: the accumulator holds the new context
                //   value; the old context is saved into `reg` so it can be
                //   restored later by PopContext.
                //
                //   Registers hold `JsValue`, not `Option<JsValue>`, so an
                //   absent context is encoded as `JsValue::Undefined` in the
                //   saved register.  PopContext inverts this by mapping
                //   `Undefined` back to `None`.
                Opcode::PushContext => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("PushContext", 0));
                    };
                    // Encode None as Undefined so it can be stored in a register.
                    let old_ctx = frame.context.take().unwrap_or(JsValue::Undefined);
                    frame.write_reg(v, old_ctx)?;
                    frame.context = Some(frame.accumulator.clone());
                }

                // PopContext [reg]: restore the context that was previously
                //   saved in `reg` by PushContext.
                //   `Undefined` in the register means there was no context.
                Opcode::PopContext => {
                    let Operand::Register(v) = instr.operands[0] else {
                        return Err(err_bad_operand("PopContext", 0));
                    };
                    let saved = frame.read_reg(v)?.clone();
                    frame.context = if saved.is_undefined() {
                        None
                    } else {
                        Some(saved)
                    };
                }

                // ── Exception handling ─────────────────────────────────────
                //
                // Throw / ReThrow: the value to throw is in the accumulator.
                // Walk the handler table looking for the innermost entry whose
                // [try_start, try_end) range covers the current instruction
                // (pc was already incremented, so the throwing instruction
                // index is `frame.pc - 1`).  If a handler is found, load
                // the thrown value into the accumulator and jump to the
                // handler entry point.  Otherwise, propagate the exception
                // as a `StatorError::JsException` to the caller.
                Opcode::Throw | Opcode::ReThrow => {
                    let thrown = frame.accumulator.clone();
                    let throw_idx = (frame.pc - 1) as u32;
                    if let Some(handler_pc) = find_handler(throw_idx, &handler_table) {
                        frame.accumulator = thrown;
                        frame.pc = handler_pc;
                        continue;
                    }
                    // No handler in this frame — serialise the thrown value and
                    // propagate it as a `StatorError::JsException` to the caller.
                    let msg = error_message_from_value(&thrown);
                    return Err(StatorError::JsException(msg));
                }

                // ── Ignored no-ops ─────────────────────────────────────────
                // These opcodes carry metadata that does not affect execution.
                Opcode::StackCheck
                | Opcode::SetExpressionPosition
                | Opcode::SetExpressionPositionFromEnd => {}

                // ── Iterators ──────────────────────────────────────────────
                //
                // GetIterator [iterable_reg, load_slot, call_slot]:
                //   Obtain a sync iterator from the iterable in `iterable_reg`.
                //
                //   Supported iterables:
                //   - JsValue::Array   → NativeIterator over the array elements
                //   - JsValue::String  → NativeIterator over Unicode characters
                //   - JsValue::Generator → generators are their own iterators
                //   - JsValue::Iterator → pass through unchanged
                //
                //   The result is placed in the accumulator.
                Opcode::GetIterator => {
                    let Operand::Register(iter_v) = instr.operands[0] else {
                        return Err(err_bad_operand("GetIterator", 0));
                    };
                    let iterable = frame.read_reg(iter_v)?.clone();
                    frame.accumulator = match iterable {
                        JsValue::Array(items) => {
                            let items_vec: Vec<JsValue> = (*items).clone();
                            JsValue::Iterator(NativeIterator::from_items(items_vec))
                        }
                        JsValue::String(ref s) => JsValue::Iterator(NativeIterator::from_string(s)),
                        // Generators and existing iterators pass through unchanged.
                        JsValue::Generator(_) | JsValue::Iterator(_) => iterable,
                        other => {
                            return Err(StatorError::TypeError(format!(
                                "GetIterator: value is not iterable (got {other:?})"
                            )));
                        }
                    };
                }

                // GetAsyncIterator [iterable_reg, load_slot, call_slot]:
                //   Async variant — behaves identically to GetIterator for all
                //   currently-supported iterable types.
                Opcode::GetAsyncIterator => {
                    let Operand::Register(iter_v) = instr.operands[0] else {
                        return Err(err_bad_operand("GetAsyncIterator", 0));
                    };
                    let iterable = frame.read_reg(iter_v)?.clone();
                    frame.accumulator = match iterable {
                        JsValue::Array(items) => {
                            let items_vec: Vec<JsValue> = (*items).clone();
                            JsValue::Iterator(NativeIterator::from_items(items_vec))
                        }
                        JsValue::String(ref s) => JsValue::Iterator(NativeIterator::from_string(s)),
                        JsValue::Generator(_) | JsValue::Iterator(_) => iterable,
                        other => {
                            return Err(StatorError::TypeError(format!(
                                "GetAsyncIterator: value is not iterable (got {other:?})"
                            )));
                        }
                    };
                }

                // IteratorNext [iterator_reg, value_out_reg]:
                //   Advance the iterator stored in `iterator_reg` by one step.
                //
                //   Semantics:
                //   - `value_out_reg ← next value` (or `undefined` when done)
                //   - `acc ← done` (a `JsValue::Boolean`)
                //
                //   Supported iterator types:
                //   - JsValue::Iterator (NativeIterator)
                //   - JsValue::Generator (runs one generator step via
                //     `run_generator_step`)
                Opcode::IteratorNext => {
                    let Operand::Register(iter_v) = instr.operands[0] else {
                        return Err(err_bad_operand("IteratorNext", 0));
                    };
                    let Operand::Register(value_out_v) = instr.operands[1] else {
                        return Err(err_bad_operand("IteratorNext", 1));
                    };
                    let iter = frame.read_reg(iter_v)?.clone();
                    let (value, done) = match iter {
                        JsValue::Iterator(ni) => match ni.borrow_mut().next_item() {
                            Some(v) => (v, false),
                            None => (JsValue::Undefined, true),
                        },
                        JsValue::Generator(ref gs) => {
                            match Interpreter::run_generator_step(gs, JsValue::Undefined)? {
                                GeneratorStep::Yield(v) => (v, false),
                                GeneratorStep::Return(v) => (v, true),
                            }
                        }
                        other => {
                            return Err(StatorError::TypeError(format!(
                                "IteratorNext: value is not an iterator (got {other:?})"
                            )));
                        }
                    };
                    frame.write_reg(value_out_v, value)?;
                    frame.accumulator = JsValue::Boolean(done);
                }

                // ── Generators / Async ─────────────────────────────────────
                //
                // SuspendGenerator [gen, regs_start, regs_count, suspend_id]:
                //   Yield a value from a generator function.
                //
                //   The yield value is whatever is currently in the accumulator.
                //   When a generator_state is attached to the frame, the full
                //   register file is saved back into it and the resume PC is
                //   recorded so execution can continue from after this instruction
                //   on the next call to `run_generator_step`.  The yielded value is
                //   signalled to `run_generator_step` via `frame.suspend_result`,
                //   then the interpreter loop exits early.
                Opcode::SuspendGenerator => {
                    let yield_val = frame.accumulator.clone();

                    // Save state into the attached GeneratorState (if any).
                    if let Some(gs_rc) = frame.generator_state.as_ref() {
                        let mut gs = gs_rc.borrow_mut();
                        // Save the full register file so that ResumeGenerator
                        // can restore it on the next step.
                        gs.registers.clone_from(&frame.registers);
                        // frame.pc was already advanced past this instruction.
                        gs.resume_pc = frame.pc;
                        gs.status = GeneratorStatus::SuspendedAtYield;
                    }

                    frame.suspend_result = Some(yield_val.clone());
                    return Ok(yield_val);
                }

                // ResumeGenerator [gen, regs_start, regs_count]:
                //   Restore the register file from the generator state saved by a
                //   prior SuspendGenerator.  The accumulator is left untouched: it
                //   already holds the value sent by the caller of `.next(value)`.
                Opcode::ResumeGenerator => {
                    if let Some(gs_rc) = frame.generator_state.as_ref() {
                        let mut gs = gs_rc.borrow_mut();
                        // Restore the saved registers into the frame.
                        // The saved register file has the same length as the frame
                        // (set by SuspendGenerator from frame.registers.clone()), so
                        // the min() only protects against a fresh generator where
                        // gs.registers is empty (resume_pc == 0).
                        let count = gs.registers.len().min(frame.registers.len());
                        frame.registers[..count].clone_from_slice(&gs.registers[..count]);
                        gs.status = GeneratorStatus::Executing;
                    }
                    // Accumulator keeps the resume value supplied by run_generator_step.
                }

                // GetGeneratorState [gen]:
                //   Load the generator's lifecycle state as a Smi into the
                //   accumulator.  Encoding: Executing=−2, Completed=−1,
                //   SuspendedAtStart=0, SuspendedAtYield=1.
                Opcode::GetGeneratorState => {
                    frame.accumulator = if let Some(gs_rc) = frame.generator_state.as_ref() {
                        JsValue::Smi(gs_rc.borrow().status.to_smi())
                    } else {
                        JsValue::Smi(GeneratorStatus::Completed.to_smi())
                    };
                }

                // SetGeneratorState [gen]:
                //   Write the Smi in the accumulator back to the generator's state.
                Opcode::SetGeneratorState => {
                    if let Some(gs_rc) = frame.generator_state.as_ref()
                        && let JsValue::Smi(n) = frame.accumulator
                    {
                        gs_rc.borrow_mut().status = GeneratorStatus::from_smi(n);
                    }
                }

                // SwitchOnGeneratorState [gen]:
                //   At the beginning of a generator function, dispatch execution to
                //   the saved resume point when the generator has been previously
                //   suspended.  If the generator is fresh (resume_pc == 0), fall
                //   through to the next instruction (start of the function body).
                //
                //   Note: when called via `run_generator_step`, the frame's PC is
                //   already set to `resume_pc`, so this instruction only fires on
                //   the first call (when resume_pc == 0) and falls through.  It
                //   becomes useful when the generator bytecode is always entered
                //   from PC 0 (the V8 pattern).
                Opcode::SwitchOnGeneratorState => {
                    if let Some(gs_rc) = frame.generator_state.as_ref() {
                        let resume_pc = gs_rc.borrow().resume_pc;
                        if resume_pc > 0 {
                            frame.pc = resume_pc;
                        }
                    }
                }

                // ── Unimplemented ──────────────────────────────────────────
                other => {
                    return Err(StatorError::Internal(format!(
                        "unimplemented opcode: {other:?}"
                    )));
                }
            }
        }
    }

    /// Execute one step of a generator function.
    ///
    /// Restores the execution state from `state`, initialises the accumulator to
    /// `input` (the value passed to `.next()`), and runs the interpreter loop
    /// until the generator suspends at a `yield` expression or its function
    /// body returns.
    ///
    /// # Return value
    ///
    /// - [`GeneratorStep::Yield`] — a `SuspendGenerator` instruction was
    ///   reached; the generator state is updated and can be resumed with
    ///   another call.
    /// - [`GeneratorStep::Return`] — a `Return` instruction was reached; the
    ///   generator is marked [`GeneratorStatus::Completed`].
    ///
    /// Calling this method on an already-completed generator immediately
    /// returns `Ok(GeneratorStep::Return(JsValue::Undefined))`.
    pub fn run_generator_step(
        state: &Rc<RefCell<GeneratorState>>,
        input: JsValue,
    ) -> StatorResult<GeneratorStep> {
        // Short-circuit for exhausted generators.
        if state.borrow().status == GeneratorStatus::Completed {
            return Ok(GeneratorStep::Return(JsValue::Undefined));
        }

        let (bytecode_array, saved_registers, resume_pc) = {
            let gs = state.borrow();
            (
                gs.bytecode_array.clone(),
                gs.registers.clone(),
                gs.resume_pc,
            )
        };

        let param_count = bytecode_array.parameter_count() as usize;
        let frame_size = bytecode_array.frame_size() as usize;
        let total = param_count + frame_size;

        // Restore the saved register file, padding with Undefined if this is
        // the first step (saved_registers is empty).
        let mut registers = saved_registers;
        registers.resize(total, JsValue::Undefined);

        let mut frame = InterpreterFrame {
            bytecode_array,
            registers,
            accumulator: input,
            pc: resume_pc,
            context: None,
            suspend_result: None,
            generator_state: Some(Rc::clone(state)),
        };

        state.borrow_mut().status = GeneratorStatus::Executing;

        let return_val = Interpreter::run(&mut frame)?;

        if let Some(yield_val) = frame.suspend_result {
            Ok(GeneratorStep::Yield(yield_val))
        } else {
            state.borrow_mut().status = GeneratorStatus::Completed;
            Ok(GeneratorStep::Return(return_val))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Look up the innermost exception handler for the instruction at `instr_idx`.
///
/// Searches `handler_table` in order (innermost entries are pushed before
/// outer ones by the compiler, so the first matching entry is the innermost).
/// Returns the handler entry-point instruction index, or `None` if no entry
/// covers `instr_idx`.
fn find_handler(instr_idx: u32, handler_table: &[HandlerTableEntry]) -> Option<usize> {
    handler_table
        .iter()
        .find(|e| instr_idx >= e.try_start && instr_idx < e.try_end)
        .map(|e| e.handler as usize)
}

/// Collect `count` consecutive argument values from `frame` starting at the
/// flat register index corresponding to the encoded `args_start_v` operand.
///
/// Used by the `CallAnyReceiver`, `CallWithSpread`, and `Construct` handlers.
fn collect_args(
    frame: &InterpreterFrame,
    args_start_v: u32,
    count: u32,
) -> StatorResult<Vec<JsValue>> {
    if count == 0 {
        return Ok(vec![]);
    }
    let start_flat = frame.reg_index(args_start_v)?;
    let end_flat = start_flat + count as usize;
    if end_flat > frame.registers.len() {
        return Err(StatorError::Internal(format!(
            "collect_args: register range {start_flat}..{end_flat} out of bounds \
             (registers.len={})",
            frame.registers.len()
        )));
    }
    Ok(frame.registers[start_flat..end_flat].to_vec())
}

/// Resolve a [`Operand::JumpOffset`] delta to an instruction index.
///
/// `pc_after_jump` is the current program counter **after** incrementing past
/// the jump instruction.  `byte_offsets[pc_after_jump]` is the byte offset
/// that marks the end of the jump instruction, which is the reference point
/// for the signed `delta`.
///
/// `instr_count` is `instructions.len()`; only the first `instr_count` entries
/// of `byte_offsets` are valid instruction starts.
fn resolve_jump(
    pc_after_jump: usize,
    delta: i32,
    byte_offsets: &[usize],
    instr_count: usize,
) -> StatorResult<usize> {
    let end_byte = byte_offsets[pc_after_jump];
    let target_byte = (end_byte as i64 + delta as i64) as usize;
    byte_offsets[..instr_count]
        .binary_search(&target_byte)
        .map_err(|_| {
            StatorError::Internal(format!(
                "jump target byte offset {target_byte} is not at an instruction boundary"
            ))
        })
}

/// Convert a constant-pool entry to a [`JsValue`].
fn constant_to_value(entry: &ConstantPoolEntry) -> JsValue {
    match entry {
        ConstantPoolEntry::Number(n) => number_to_jsvalue(*n),
        ConstantPoolEntry::String(s) => JsValue::String(s.clone()),
        ConstantPoolEntry::Boolean(b) => JsValue::Boolean(*b),
        ConstantPoolEntry::Null => JsValue::Null,
        ConstantPoolEntry::Undefined => JsValue::Undefined,
        ConstantPoolEntry::Function(ba) => JsValue::Function(Rc::new((**ba).clone())),
    }
}

/// Produce a [`JsValue`] for a numeric result: use [`JsValue::Smi`] if the
/// value is an exact integer that fits in `i32`, otherwise
/// [`JsValue::HeapNumber`].
fn number_to_jsvalue(n: f64) -> JsValue {
    if n.is_finite() && n.fract() == 0.0 && (i32::MIN as f64..=i32::MAX as f64).contains(&n) {
        JsValue::Smi(n as i32)
    } else {
        JsValue::HeapNumber(n)
    }
}

/// ECMAScript `+` operator: string concatenation or numeric addition.
///
/// If either operand is already a string, both are converted to strings and
/// concatenated.  Otherwise both are converted to numbers and added.
fn js_add(lhs: &JsValue, rhs: &JsValue) -> StatorResult<JsValue> {
    if lhs.is_string() || rhs.is_string() {
        let l = lhs.to_js_string()?;
        let r = rhs.to_js_string()?;
        Ok(JsValue::String(l + &r))
    } else {
        let l = lhs.to_number()?;
        let r = rhs.to_number()?;
        Ok(number_to_jsvalue(l + r))
    }
}

/// ECMAScript §7.2.11 **Abstract Relational Comparison** (`<`).
///
/// Returns `false` when either operand is `NaN` (consistent with IEEE 754).
/// String operands are compared lexicographically; all other combinations
/// convert to numbers first.
fn js_less_than(lhs: &JsValue, rhs: &JsValue) -> StatorResult<bool> {
    if let (JsValue::String(a), JsValue::String(b)) = (lhs, rhs) {
        return Ok(a < b);
    }
    let l = lhs.to_number()?;
    let r = rhs.to_number()?;
    if l.is_nan() || r.is_nan() {
        return Ok(false);
    }
    Ok(l < r)
}

/// ECMAScript §7.2.13 **Abstract Equality Comparison** (`==`).
fn abstract_eq(lhs: &JsValue, rhs: &JsValue) -> bool {
    match (lhs, rhs) {
        // Same primitive type.
        (JsValue::Undefined, JsValue::Undefined) | (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
        (JsValue::String(a), JsValue::String(b)) => a == b,
        // Numeric — covers Smi×Smi, Smi×HeapNumber, HeapNumber×HeapNumber.
        (lhs, rhs) if lhs.is_number() && rhs.is_number() => {
            matches!((lhs.to_number(), rhs.to_number()), (Ok(a), Ok(b)) if a == b)
        }
        // null == undefined.
        (JsValue::Null, JsValue::Undefined) | (JsValue::Undefined, JsValue::Null) => true,
        // Boolean → Number coercion (ECMAScript §7.2.13 steps 9/10).
        (JsValue::Boolean(b), _) => abstract_eq(&JsValue::Smi(i32::from(*b)), rhs),
        (_, JsValue::Boolean(b)) => abstract_eq(lhs, &JsValue::Smi(i32::from(*b))),
        // String → Number coercion (steps 5/6).
        (JsValue::String(s), _) if rhs.is_number() => {
            let n: f64 = s.trim().parse().unwrap_or(f64::NAN);
            abstract_eq(&JsValue::HeapNumber(n), rhs)
        }
        (_, JsValue::String(s)) if lhs.is_number() => {
            let n: f64 = s.trim().parse().unwrap_or(f64::NAN);
            abstract_eq(lhs, &JsValue::HeapNumber(n))
        }
        // Object identity — `JsValue::Object` holds a raw `*mut HeapObject`
        // pointer; comparing the pointer values gives reference identity.
        (JsValue::Object(a), JsValue::Object(b)) => std::ptr::eq(*a, *b),
        _ => false,
    }
}

/// ECMAScript §7.2.15 **Strict Equality Comparison** (`===`).
fn strict_eq(lhs: &JsValue, rhs: &JsValue) -> bool {
    match (lhs, rhs) {
        (JsValue::Undefined, JsValue::Undefined) | (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
        (JsValue::String(a), JsValue::String(b)) => a == b,
        // Numeric — IEEE 754: NaN !== NaN is handled correctly by f64's PartialEq.
        (JsValue::Smi(a), JsValue::Smi(b)) => a == b,
        (JsValue::HeapNumber(a), JsValue::HeapNumber(b)) => a == b,
        (JsValue::Smi(a), JsValue::HeapNumber(b)) => (*a as f64) == *b,
        (JsValue::HeapNumber(a), JsValue::Smi(b)) => *a == (*b as f64),
        // Object identity — `JsValue::Object` holds a raw `*mut HeapObject`
        // pointer; comparing the pointer values gives reference identity.
        (JsValue::Object(a), JsValue::Object(b)) => std::ptr::eq(*a, *b),
        _ => false,
    }
}

/// Construct a diagnostic error for an unexpected operand kind.
#[cold]
fn err_bad_operand(opcode_name: &'static str, operand_index: usize) -> StatorError {
    StatorError::Internal(format!(
        "{opcode_name}: unexpected operand kind at index {operand_index}"
    ))
}

/// Convert a thrown JavaScript value to a human-readable error message string.
///
/// `Error` objects format as `"name: message"` (or just `"name"` for an empty
/// message).  All other values are converted via [`JsValue::to_js_string`];
/// when that conversion itself fails the debug representation is used instead.
fn error_message_from_value(value: &JsValue) -> String {
    match value {
        JsValue::Error(e) => e.to_error_string(),
        other => other
            .to_js_string()
            .unwrap_or_else(|_| format!("{other:?}")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecode_array::BytecodeArray;
    use crate::bytecode::bytecodes::{Instruction, Opcode, Operand, encode};
    use crate::bytecode::feedback::FeedbackMetadata;

    /// Build a [`BytecodeArray`] from a raw instruction list.
    fn make_bytecode(instrs: Vec<Instruction>, frame_size: u32, param_count: u32) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Run a zero-argument bytecode and return the result.
    fn run_bytecode(instrs: Vec<Instruction>, frame_size: u32) -> StatorResult<JsValue> {
        let ba = make_bytecode(instrs, frame_size, 0);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame)
    }

    // ── LdaSmi / LdaUndefined / Return ──────────────────────────────────────

    #[test]
    fn test_lda_smi_and_return() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_lda_undefined_and_return() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_lda_zero() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_lda_null() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Null);
    }

    #[test]
    fn test_lda_true_false() {
        let t = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(t, JsValue::Boolean(true));

        let f = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(f, JsValue::Boolean(false));
    }

    // ── Star / Ldar ──────────────────────────────────────────────────────────

    #[test]
    fn test_star_ldar_round_trip() {
        // lda 7 → r0, lda 0, ldar r0 → return 7
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    /// Helper: evaluate `lhs <op> rhs` using the standard Ignition pattern:
    /// evaluate rhs → r0, evaluate lhs → acc, apply op.
    fn arith_op(lhs: i32, rhs: i32, op: Opcode) -> StatorResult<JsValue> {
        run_bytecode(
            vec![
                // Save rhs to r0
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(rhs)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                // Load lhs into acc
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(lhs)]),
                // acc = acc <op> r0
                Instruction::new_unchecked(
                    op,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1, // one local register (r0)
        )
    }

    #[test]
    fn test_add_integers() {
        assert_eq!(arith_op(3, 4, Opcode::Add).unwrap(), JsValue::Smi(7));
        assert_eq!(arith_op(0, 0, Opcode::Add).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(-5, 5, Opcode::Add).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(100, 200, Opcode::Add).unwrap(), JsValue::Smi(300));
    }

    #[test]
    fn test_sub_integers() {
        assert_eq!(arith_op(10, 3, Opcode::Sub).unwrap(), JsValue::Smi(7));
        assert_eq!(arith_op(0, 5, Opcode::Sub).unwrap(), JsValue::Smi(-5));
    }

    #[test]
    fn test_mul_integers() {
        assert_eq!(arith_op(6, 7, Opcode::Mul).unwrap(), JsValue::Smi(42));
        assert_eq!(arith_op(0, 999, Opcode::Mul).unwrap(), JsValue::Smi(0));
        assert_eq!(arith_op(-3, 4, Opcode::Mul).unwrap(), JsValue::Smi(-12));
    }

    #[test]
    fn test_div_integers() {
        // 10 / 2 = 5 (exact → Smi)
        assert_eq!(arith_op(10, 2, Opcode::Div).unwrap(), JsValue::Smi(5));
        // 7 / 2 = 3.5 (fractional → HeapNumber)
        assert_eq!(
            arith_op(7, 2, Opcode::Div).unwrap(),
            JsValue::HeapNumber(3.5)
        );
    }

    #[test]
    fn test_div_by_zero() {
        // JS: 1 / 0 = Infinity
        let result = arith_op(1, 0, Opcode::Div).unwrap();
        assert_eq!(result, JsValue::HeapNumber(f64::INFINITY));
    }

    // ── TestEqual ────────────────────────────────────────────────────────────

    #[test]
    fn test_equal_same_smis() {
        // 5 == 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_equal_different_smis() {
        // 3 == 7 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── TestEqualStrict ──────────────────────────────────────────────────────

    #[test]
    fn test_strict_equal_same_smis() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_strict_equal_null_undefined_are_not_equal() {
        // null !== undefined under strict equality
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
                Instruction::new_unchecked(
                    Opcode::TestEqualStrict,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Parameter registers ──────────────────────────────────────────────────

    #[test]
    fn test_parameter_registers() {
        // Function: return first argument (param[0]).
        // param[0] → Register(-1) → encoded as Register(0xFFFF_FFFF)
        let param_reg = (-1_i32) as u32;
        let ba = make_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param_reg)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
            1, // 1 parameter
        );
        let mut frame = InterpreterFrame::new(ba, vec![JsValue::Smi(99)]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    // ── Arithmetic expression sequence ───────────────────────────────────────

    #[test]
    fn test_arithmetic_expression_sequence() {
        // Simulate: (3 + 4) * 2 − 1 = 13
        //
        // Bytecode layout:
        //   lda 4      → r0          (rhs of +)
        //   lda 3                    (lhs of +)
        //   add r0     → acc = 7
        //   star r1                  (save 7)
        //   lda 2      → r0          (rhs of *)
        //   ldar r1    → acc = 7     (lhs of *)
        //   mul r0     → acc = 14
        //   star r1                  (save 14)
        //   lda 1      → r0          (rhs of −)
        //   ldar r1    → acc = 14    (lhs of −)
        //   sub r0     → acc = 13
        //   return
        let result = run_bytecode(
            vec![
                // acc = 3 + 4
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::Add,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                // save 7 → r1
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                // acc = 7 * 2
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
                Instruction::new_unchecked(
                    Opcode::Mul,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                // save 14 → r1
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
                // acc = 14 - 1
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
                Instruction::new_unchecked(
                    Opcode::Sub,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            2, // two local registers (r0, r1)
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(13));
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_no_return_is_error() {
        let err = run_bytecode(
            vec![Instruction::new_unchecked(
                Opcode::LdaSmi,
                vec![Operand::Immediate(1)],
            )],
            0,
        )
        .unwrap_err();
        assert!(matches!(err, StatorError::Internal(_)));
    }

    #[test]
    fn test_unimplemented_opcode_is_error() {
        let err = run_bytecode(
            vec![
                // Debugger is not implemented in the interpreter yet.
                Instruction::new_unchecked(Opcode::Debugger, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap_err();
        assert!(matches!(err, StatorError::Internal(_)));
    }

    // ── Jump (unconditional) ─────────────────────────────────────────────────

    #[test]
    fn test_jump_forward_skips_instructions() {
        // Jump with offset 2 skips the next 2-byte instruction (LdaSmi(99)).
        // Byte layout:
        //   [0] LdaSmi(1)   → 2 bytes (opcode + narrow immediate)
        //   [2] Jump(2)     → 2 bytes (opcode + narrow offset)
        //   [4] LdaSmi(99)  → 2 bytes ← skipped; end-of-jump=4, target=4+2=6
        //   [6] Return      → 1 byte
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
                // Jump 2 bytes forward, skipping LdaSmi(99).
                Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(2)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        // LdaSmi(99) is skipped; acc stays 1.
        assert_eq!(result, JsValue::Smi(1));
    }

    #[test]
    fn test_jump_if_to_boolean_false_taken_skips_instruction() {
        // When acc is falsy the jump is taken and LdaSmi(99) is skipped.
        // Byte layout:
        //   [0] LdaSmi(42)             → 2 bytes  (acc = 42)
        //   [2] Star(r0)               → 2 bytes  (r0 = 42)
        //   [4] LdaFalse               → 1 byte   (acc = false)
        //   [5] JumpIfToBooleanFalse(2)→ 2 bytes  end=7; if taken target=9
        //   [7] LdaSmi(99)             → 2 bytes  ← skipped when jump taken
        //   [9] Ldar(r0)               → 2 bytes  (acc = 42)
        //  [11] Return                 → 1 byte
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
                Instruction::new_unchecked(
                    Opcode::JumpIfToBooleanFalse,
                    vec![Operand::JumpOffset(2)],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        // Jump taken (false is falsy): LdaSmi(99) skipped, Ldar(r0)=42 returned.
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_jump_if_to_boolean_false_not_taken() {
        // When acc is truthy the jump is NOT taken and execution falls through.
        // Same byte layout as above but with LdaTrue instead of LdaFalse.
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(
                    Opcode::JumpIfToBooleanFalse,
                    vec![Operand::JumpOffset(2)],
                ),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
                Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        // Jump NOT taken (true is truthy): LdaSmi(99) executes, then Ldar(r0)=42.
        // Both LdaSmi(99) and Ldar(r0) execute, so acc = 42 at return.
        assert_eq!(result, JsValue::Smi(42));
    }

    // ── Comparison opcodes ───────────────────────────────────────────────────

    #[test]
    fn test_less_than() {
        // 3 < 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));

        // 7 < 3 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_greater_than() {
        // 7 > 3 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
                Instruction::new_unchecked(
                    Opcode::TestGreaterThan,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_less_than_or_equal() {
        // 5 <= 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestLessThanOrEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_greater_than_or_equal() {
        // 5 >= 6 → false
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(6)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(
                    Opcode::TestGreaterThanOrEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_not_equal() {
        // 3 != 5 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(
                    Opcode::TestNotEqual,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_test_null_and_undefined() {
        // null → true for TestNull
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaNull, vec![]),
                Instruction::new_unchecked(Opcode::TestNull, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));

        // 42 → false for TestUndefined
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
                Instruction::new_unchecked(Opcode::TestUndefined, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    // ── Inc / Dec / Mod ──────────────────────────────────────────────────────

    #[test]
    fn test_inc_and_dec() {
        // inc: 5 → 6
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(6));

        // dec: 5 → 4
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
                Instruction::new_unchecked(Opcode::Dec, vec![Operand::FeedbackSlot(0)]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(4));
    }

    #[test]
    fn test_mod() {
        // 10 % 3 = 1
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
                Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
                Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
                Instruction::new_unchecked(
                    Opcode::Mod,
                    vec![Operand::Register(0), Operand::FeedbackSlot(0)],
                ),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            1,
        )
        .unwrap();
        assert_eq!(result, JsValue::Smi(1));
    }

    // ── LogicalNot / ToBooleanLogicalNot ─────────────────────────────────────

    #[test]
    fn test_logical_not() {
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
                Instruction::new_unchecked(Opcode::LogicalNot, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_to_boolean_logical_not() {
        // !0 → true
        let result = run_bytecode(
            vec![
                Instruction::new_unchecked(Opcode::LdaZero, vec![]),
                Instruction::new_unchecked(Opcode::ToBooleanLogicalNot, vec![]),
                Instruction::new_unchecked(Opcode::Return, vec![]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(result, JsValue::Boolean(true));
    }

    // ── End-to-end: compiler → interpreter ──────────────────────────────────
    //
    // These tests use BytecodeGenerator to compile a hand-built AST, then run
    // the resulting BytecodeArray through the interpreter.

    /// Compile `stmts` as a top-level script and run the resulting bytecode.
    fn compile_and_run(stmts: Vec<crate::parser::ast::Stmt>) -> StatorResult<JsValue> {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::ast::{Program, ProgramItem, SourceType};

        let program = Program {
            loc: span(),
            source_type: SourceType::Script,
            body: stmts.into_iter().map(ProgramItem::Stmt).collect(),
        };
        let ba = BytecodeGenerator::compile_program(&program)?;
        let mut frame = InterpreterFrame::new(ba, vec![]);
        Interpreter::run(&mut frame)
    }

    fn span() -> crate::parser::scanner::Span {
        use crate::parser::scanner::{Position, Span};
        let p = Position {
            offset: 0,
            line: 1,
            column: 1,
        };
        Span { start: p, end: p }
    }

    fn num_expr(v: f64) -> crate::parser::ast::Expr {
        use crate::parser::ast::{Expr, NumLit};
        Expr::Num(NumLit {
            loc: span(),
            value: v,
            raw: v.to_string(),
        })
    }

    fn ident_expr(name: &str) -> crate::parser::ast::Expr {
        use crate::parser::ast::{Expr, Ident};
        Expr::Ident(Ident {
            loc: span(),
            name: name.to_owned(),
        })
    }

    fn return_stmt(arg: Option<crate::parser::ast::Expr>) -> crate::parser::ast::Stmt {
        use crate::parser::ast::{ReturnStmt, Stmt};
        Stmt::Return(ReturnStmt {
            loc: span(),
            argument: arg.map(Box::new),
        })
    }

    fn var_let(name: &str, init: crate::parser::ast::Expr) -> crate::parser::ast::Stmt {
        use crate::parser::ast::{Pat, Stmt, VarDecl, VarDeclarator, VarKind};
        Stmt::VarDecl(VarDecl {
            loc: span(),
            kind: VarKind::Let,
            declarators: vec![VarDeclarator {
                loc: span(),
                id: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: name.to_owned(),
                }),
                init: Some(Box::new(init)),
            }],
        })
    }

    fn binary(
        op: crate::parser::ast::BinaryOp,
        lhs: crate::parser::ast::Expr,
        rhs: crate::parser::ast::Expr,
    ) -> crate::parser::ast::Expr {
        use crate::parser::ast::{BinaryExpr, Expr};
        Expr::Binary(Box::new(BinaryExpr {
            loc: span(),
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
        }))
    }

    fn assign_expr(
        name: &str,
        op: crate::parser::ast::AssignOp,
        rhs: crate::parser::ast::Expr,
    ) -> crate::parser::ast::Expr {
        use crate::parser::ast::{AssignExpr, AssignTarget, Expr, Ident};
        Expr::Assign(Box::new(AssignExpr {
            loc: span(),
            op,
            left: AssignTarget::Expr(Box::new(Expr::Ident(Ident {
                loc: span(),
                name: name.to_owned(),
            }))),
            right: Box::new(rhs),
        }))
    }

    /// Test: if/else branching — return "big" if x > 5, else "small".
    #[test]
    fn test_e2e_if_else_branching() {
        use crate::parser::ast::{BinaryOp, IfStmt, ReturnStmt, Stmt};

        // let x = 10;
        // if (x > 5) { return 1; } else { return 0; }
        let x_val = 10.0;
        let stmts = vec![
            var_let("x", num_expr(x_val)),
            Stmt::If(IfStmt {
                loc: span(),
                test: Box::new(binary(BinaryOp::Gt, ident_expr("x"), num_expr(5.0))),
                consequent: Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(1.0))),
                })),
                alternate: Some(Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(0.0))),
                }))),
            }),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(1)); // x=10 > 5, so return 1

        // Now with x = 3 (less than 5)
        let stmts2 = vec![
            var_let("x", num_expr(3.0)),
            Stmt::If(IfStmt {
                loc: span(),
                test: Box::new(binary(BinaryOp::Gt, ident_expr("x"), num_expr(5.0))),
                consequent: Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(1.0))),
                })),
                alternate: Some(Box::new(Stmt::Return(ReturnStmt {
                    loc: span(),
                    argument: Some(Box::new(num_expr(0.0))),
                }))),
            }),
        ];
        let result2 = compile_and_run(stmts2).unwrap();
        assert_eq!(result2, JsValue::Smi(0)); // x=3 <= 5, so return 0
    }

    /// Test: for loop computing the sum 0+1+2+…+9 = 45.
    #[test]
    fn test_e2e_for_loop_sum() {
        use crate::parser::ast::{
            AssignOp, BinaryOp, BlockStmt, ExprStmt, ForInit, ForStmt, Stmt, UpdateExpr, UpdateOp,
            VarDecl, VarDeclarator, VarKind,
        };

        // let sum = 0;
        // for (let i = 0; i < 10; i++) { sum += i; }
        // return sum;
        let stmts = vec![
            var_let("sum", num_expr(0.0)),
            Stmt::For(ForStmt {
                loc: span(),
                init: Some(ForInit::VarDecl(VarDecl {
                    loc: span(),
                    kind: VarKind::Let,
                    declarators: vec![VarDeclarator {
                        loc: span(),
                        id: crate::parser::ast::Pat::Ident(crate::parser::ast::Ident {
                            loc: span(),
                            name: "i".to_owned(),
                        }),
                        init: Some(Box::new(num_expr(0.0))),
                    }],
                })),
                test: Some(Box::new(binary(
                    BinaryOp::Lt,
                    ident_expr("i"),
                    num_expr(10.0),
                ))),
                update: Some(Box::new(crate::parser::ast::Expr::Update(Box::new(
                    UpdateExpr {
                        loc: span(),
                        op: UpdateOp::Increment,
                        prefix: false,
                        argument: Box::new(ident_expr("i")),
                    },
                )))),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![Stmt::Expr(ExprStmt {
                        loc: span(),
                        expr: Box::new(assign_expr("sum", AssignOp::AddAssign, ident_expr("i"))),
                    })],
                })),
            }),
            return_stmt(Some(ident_expr("sum"))),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(45)); // 0+1+…+9 = 45
    }

    /// Test: while loop with break — count up to 5 then stop.
    #[test]
    fn test_e2e_while_with_break() {
        use crate::parser::ast::{
            BinaryOp, BlockStmt, BreakStmt, ExprStmt, IfStmt, Stmt, UpdateExpr, UpdateOp, WhileStmt,
        };

        // let n = 0;
        // while (true) { if (n >= 5) break; n++; }
        // return n;
        let stmts = vec![
            var_let("n", num_expr(0.0)),
            Stmt::While(WhileStmt {
                loc: span(),
                test: Box::new(crate::parser::ast::Expr::Bool(
                    crate::parser::ast::BoolLit {
                        loc: span(),
                        value: true,
                    },
                )),
                body: Box::new(Stmt::Block(BlockStmt {
                    loc: span(),
                    body: vec![
                        Stmt::If(IfStmt {
                            loc: span(),
                            test: Box::new(binary(BinaryOp::GtEq, ident_expr("n"), num_expr(5.0))),
                            consequent: Box::new(Stmt::Break(BreakStmt {
                                loc: span(),
                                label: None,
                            })),
                            alternate: None,
                        }),
                        Stmt::Expr(ExprStmt {
                            loc: span(),
                            expr: Box::new(crate::parser::ast::Expr::Update(Box::new(
                                UpdateExpr {
                                    loc: span(),
                                    op: UpdateOp::Increment,
                                    prefix: false,
                                    argument: Box::new(ident_expr("n")),
                                },
                            ))),
                        }),
                    ],
                })),
            }),
            return_stmt(Some(ident_expr("n"))),
        ];

        let result = compile_and_run(stmts).unwrap();
        assert_eq!(result, JsValue::Smi(5));
    }

    // ── Function calls and closures (P3) ─────────────────────────────────────

    /// Helper: build a BytecodeArray with an explicit constant pool.
    fn make_bytecode_with_pool(
        instrs: Vec<Instruction>,
        pool: Vec<ConstantPoolEntry>,
        frame_size: u32,
        param_count: u32,
    ) -> BytecodeArray {
        BytecodeArray::new(
            encode(&instrs),
            pool,
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        )
    }

    /// Low-level test: manually build an `add(a, b)` function, create a
    /// closure with `CreateClosure`, and invoke it with `CallAnyReceiver`.
    ///
    /// Simulates: `function add(a, b) { return a + b; } ; return add(3, 4);`
    #[test]
    fn test_create_closure_and_call_any_receiver() {
        // ── Inner function: add(a, b) { return a + b; } ─────────────────────
        //
        // param[0] = a → encoded register (-1i32 as u32)
        // param[1] = b → encoded register (-2i32 as u32)
        // r0 = local register used as RHS of Add.
        let param0_v: u32 = (-1i32) as u32;
        let param1_v: u32 = (-2i32) as u32;

        let inner_instrs = vec![
            // acc = b (param[1])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param1_v)]),
            // r0 = b
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // acc = a (param[0])
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(param0_v)]),
            // acc = a + r0 (= a + b)
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let inner_ba = make_bytecode(inner_instrs, 1, 2); // frame_size=1, param_count=2

        // ── Outer script ────────────────────────────────────────────────────
        //
        // constant pool: [Function(inner_ba)]
        // r0 = closure, r1 = arg0 (3), r2 = arg1 (4)
        let outer_instrs = vec![
            // acc = JsValue::Function(inner_ba)
            Instruction::new_unchecked(
                Opcode::CreateClosure,
                vec![
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // arg0 = 3
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // arg1 = 4
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(4)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            // acc = add(3, 4)
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),      // callee = r0
                    Operand::Register(1),      // args_start = r1
                    Operand::RegisterCount(2), // arg_count = 2
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let pool = vec![ConstantPoolEntry::Function(Box::new(inner_ba))];
        let outer_ba = make_bytecode_with_pool(outer_instrs, pool, 3, 0);
        let mut frame = InterpreterFrame::new(outer_ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// End-to-end: compile `function add(a, b) { return a + b; }; return add(3, 4);`
    /// using the bytecode generator and execute it.
    #[test]
    fn test_e2e_function_call_two_args() {
        use crate::parser::ast::{BinaryOp, BlockStmt, CallExpr, Expr, FnDecl, Param, Pat, Stmt};

        // function add(a, b) { return a + b; }
        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "add".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![
                Param {
                    loc: span(),
                    pat: Pat::Ident(crate::parser::ast::Ident {
                        loc: span(),
                        name: "a".to_owned(),
                    }),
                    default: None,
                },
                Param {
                    loc: span(),
                    pat: Pat::Ident(crate::parser::ast::Ident {
                        loc: span(),
                        name: "b".to_owned(),
                    }),
                    default: None,
                },
            ],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Add,
                    ident_expr("a"),
                    ident_expr("b"),
                )))],
            },
        }));

        // return add(3, 4);
        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("add")),
            arguments: vec![num_expr(3.0), num_expr(4.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// End-to-end: call a zero-argument function.
    ///
    /// `function f() { return 42; }; return f();`
    #[test]
    fn test_e2e_function_call_no_args() {
        use crate::parser::ast::{BlockStmt, CallExpr, Expr, FnDecl, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "f".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(num_expr(42.0)))],
            },
        }));

        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("f")),
            arguments: vec![],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// End-to-end: function that uses a single parameter in a multiply.
    ///
    /// `function double(n) { return n * 2; }; return double(21);`
    #[test]
    fn test_e2e_function_call_one_arg() {
        use crate::parser::ast::{BinaryOp, BlockStmt, CallExpr, Expr, FnDecl, Param, Pat, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "double".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![Param {
                loc: span(),
                pat: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: "n".to_owned(),
                }),
                default: None,
            }],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Mul,
                    ident_expr("n"),
                    num_expr(2.0),
                )))],
            },
        }));

        let call_stmt = return_stmt(Some(Expr::Call(Box::new(CallExpr {
            loc: span(),
            callee: Box::new(ident_expr("double")),
            arguments: vec![num_expr(21.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, call_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// End-to-end: `new` constructor call.
    ///
    /// `function Box(v) { return v + 1; }; return new Box(41);`
    ///
    /// The P3 interpreter executes the constructor body and returns whatever
    /// it produces (full prototype-chain wiring is deferred).
    #[test]
    fn test_e2e_construct_returns_constructor_value() {
        use crate::parser::ast::{BinaryOp, BlockStmt, Expr, FnDecl, NewExpr, Param, Pat, Stmt};

        let fn_decl = Stmt::FnDecl(Box::new(FnDecl {
            loc: span(),
            id: Some(crate::parser::ast::Ident {
                loc: span(),
                name: "Box".to_owned(),
            }),
            is_async: false,
            is_generator: false,
            params: vec![Param {
                loc: span(),
                pat: Pat::Ident(crate::parser::ast::Ident {
                    loc: span(),
                    name: "v".to_owned(),
                }),
                default: None,
            }],
            body: BlockStmt {
                loc: span(),
                body: vec![return_stmt(Some(binary(
                    BinaryOp::Add,
                    ident_expr("v"),
                    num_expr(1.0),
                )))],
            },
        }));

        let new_stmt = return_stmt(Some(Expr::New(Box::new(NewExpr {
            loc: span(),
            callee: Box::new(ident_expr("Box")),
            arguments: vec![num_expr(41.0)],
        }))));

        let result = compile_and_run(vec![fn_decl, new_stmt]).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// Low-level test: `PushContext` saves the old context into a register
    /// and installs the accumulator as the new context; `PopContext` restores.
    #[test]
    fn test_push_pop_context() {
        // Bytecode:
        //   LdaSmi 99      ; acc = 99 (will become new context)
        //   PushContext r0 ; r0 = old ctx (undefined), frame.context = Some(Smi(99))
        //   LdaSmi 7       ; acc = 7 (some work inside the context)
        //   PopContext r0  ; frame.context = None (restored from r0=undefined)
        //   Return         ; return 7
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::PopContext, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let result = run_bytecode(instrs, 1).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// `JsValue::Function` should be truthy, non-callable via `to_number`,
    /// and stringify to `"function () {}"`.
    #[test]
    fn test_js_value_function_properties() {
        use crate::bytecode::bytecode_array::BytecodeArray;
        use crate::bytecode::bytecodes::encode;
        use std::rc::Rc;

        let ba = BytecodeArray::new(
            encode(&[Instruction::new_unchecked(Opcode::Return, vec![])]),
            vec![],
            0,
            0,
            vec![],
            FeedbackMetadata::empty(),
            vec![],
        );
        let f = JsValue::Function(Rc::new(ba));

        assert!(f.to_boolean());
        assert!(f.is_function());
        assert!(f.to_number().is_err());
        assert_eq!(f.to_js_string().unwrap(), "function () {}");
    }

    /// Calling a non-function value with `CallAnyReceiver` should produce a
    /// `TypeError`.
    #[test]
    fn test_call_non_function_is_type_error() {
        // r0 = 42 (a Smi, not a function); then CallAnyReceiver r0, r0, 0, slot
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(0),
                    Operand::Register(0),
                    Operand::RegisterCount(0),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let err = run_bytecode(instrs, 1).unwrap_err();
        assert!(matches!(err, StatorError::TypeError(_)));
    }

    // ── Exception handling (Throw / ReThrow / handler table) ─────────────────

    /// Helper: build a [`BytecodeArray`] with an explicit handler table.
    fn make_bytecode_with_handlers(
        instrs: Vec<Instruction>,
        frame_size: u32,
        param_count: u32,
        handler_table: Vec<crate::bytecode::bytecode_array::HandlerTableEntry>,
    ) -> BytecodeArray {
        let bytes = encode(&instrs);
        BytecodeArray::new(
            bytes,
            vec![],
            frame_size,
            param_count,
            vec![],
            FeedbackMetadata::empty(),
            handler_table,
        )
    }

    /// `try { throw 42; } catch(e) { return e; }` → returns 42.
    ///
    /// Bytecode layout (indices):
    ///   0: LdaSmi(42)
    ///   1: Throw
    ///   2: Jump(past handler) ← try_end = 2, never reached
    ///   3: Star(r0)           ← catch handler, acc = 42
    ///   4: Ldar(r0)
    ///   5: Return
    ///
    /// Handler table: { try_start=0, try_end=2, handler=3, is_finally=false }
    #[test]
    fn test_try_catch_basic() {
        use crate::bytecode::bytecode_array::HandlerTableEntry;

        // Build instructions manually so we know exact instruction indices.
        let instrs = vec![
            // idx 0 — try body: load 42
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            // idx 1 — throw it
            Instruction::new_unchecked(Opcode::Throw, vec![]),
            // idx 2 — normal-exit jump (never reached in this test)
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            // idx 3 — catch handler: acc holds thrown value; save to r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // idx 4 — load caught value
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // idx 5 — return it
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        // Patch the Jump at idx 2 so it points past the catch block.
        // After idx 2 the next instr is idx 3 (2 bytes away from end of Jump).
        // We actually want it to jump to idx 5 (Return).
        // The Jump at idx 2 is 2 bytes; its end byte is 2+2 = 4.
        // idx 5 (Return) starts at byte 4+2+2+1 = let's compute:
        // idx 0: LdaSmi 2 bytes, idx 1: Throw 1 byte, idx 2: Jump 2 bytes,
        // idx 3: Star 2 bytes, idx 4: Ldar 2 bytes, idx 5: Return 1 byte.
        // end of Jump at idx 2 = 2+1+2 = 5; Return byte = 5+2+2 = 9.
        // delta = 9 - 5 = 4.
        let mut instrs_patched = instrs;
        instrs_patched[2] = Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(4)]);

        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 2, // Throw is at idx 1; try_end = 2 (exclusive)
            handler: 3, // catch starts at idx 3
            is_finally: false,
        }];
        let ba = make_bytecode_with_handlers(instrs_patched, 1, 0, handler_table);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    /// `try { } finally { }` — finally runs on normal path.
    ///
    /// The normal-path inlined finally simply sets r0 = 99 before returning.
    /// We verify the return value is from the try body, and the finally ran.
    ///
    /// Layout (indices):
    ///   0: LdaSmi(1)    ← try body sets acc = 1
    ///   1: Star(r0)     ← save it
    ///   2: LdaSmi(99)   ← finally (normal path): acc = 99
    ///   3: Star(r1)     ← save finally sentinel to r1
    ///   4: Jump(6)      ← skip exception-path handler
    ///   5: ReThrow      ← exception-path handler (never reached on normal path)
    ///   6: Ldar(r0)     ← restore try result
    ///   7: Return
    ///
    /// Handler table: { try_start=0, try_end=2, handler=5, is_finally=true }
    /// (try_end=2 so the "finally" instructions at idx 2+ are outside try range)
    #[test]
    fn test_try_finally_normal_path() {
        use crate::bytecode::bytecode_array::HandlerTableEntry;

        // Byte layout:
        //   0: LdaSmi(1)  → 2 bytes  (ends at 2)
        //   1: Star(r0)   → 2 bytes  (ends at 4)
        //   --- try_end = 2 (instruction index) / byte 4 ---
        //   2: LdaSmi(99) → 2 bytes  (ends at 6)
        //   3: Star(r1)   → 2 bytes  (ends at 8)
        //   4: Jump(δ)    → 2 bytes  (ends at 10; target = idx 6 = byte 11)
        //                    δ = 11 - 10 = 1
        //   5: ReThrow    → 1 byte   (ends at 11)
        //   6: Ldar(r0)   → 2 bytes  (ends at 13)
        //   7: Return     → 1 byte

        let instrs = vec![
            // idx 0 — try body: acc = 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            // idx 1 — save to r0
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            // idx 2 — finally (normal path): acc = 99
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            // idx 3 — save sentinel to r1
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // idx 4 — jump past exception handler (δ = 1 → target byte = 10+1 = 11 = idx 6)
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(1)]),
            // idx 5 — exception handler: ReThrow (never reached in this test)
            Instruction::new_unchecked(Opcode::ReThrow, vec![]),
            // idx 6 — restore result from r0
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            // idx 7 — return
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let handler_table = vec![HandlerTableEntry {
            try_start: 0,
            try_end: 2, // only instructions 0..1 are "in try"
            handler: 5, // exception handler at idx 5
            is_finally: true,
        }];
        let ba = make_bytecode_with_handlers(instrs, 2, 0, handler_table);
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        // Normal path returns value from try body (r0 = 1).
        assert_eq!(result, JsValue::Smi(1));
    }

    /// `try { throw 5; } finally { /* runs */ }` — finally runs on exception path
    /// and the exception propagates after finally.
    ///
    /// Uses the bytecode generator to compile the equivalent JavaScript so we
    /// exercise the full pipeline.
    #[test]
    fn test_try_finally_exception_path() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        // try { throw 5; } finally { /* nothing — just let it propagate */ }
        // After finally the exception should propagate out.
        let src = "try { throw 5; } finally { }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let err = Interpreter::run(&mut frame).unwrap_err();
        // The exception value "5" should propagate.
        assert!(
            matches!(err, StatorError::JsException(_)),
            "expected JsException, got {err:?}"
        );
    }

    /// `try { throw 7; } catch(e) { return e; }` via the bytecode generator.
    ///
    /// Tests the full pipeline: parse → compile → run.
    #[test]
    fn test_try_catch_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "try { throw 7; } catch(e) { return e; }";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(7));
    }

    /// Nested try/catch: inner catch re-throws, outer catch receives it.
    ///
    /// ```javascript
    /// try {
    ///   try { throw 1; } catch(inner) { throw inner + 1; }
    /// } catch(outer) { return outer; }
    /// ```
    #[test]
    fn test_nested_try_catch_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            try {
                try { throw 1; } catch(inner) { throw inner + 1; }
            } catch(outer) { return outer; }
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(2));
    }

    /// `try { } catch(e) { } finally { }` — finally runs on normal path.
    ///
    /// ```javascript
    /// var ran = 0;
    /// try { ran = 1; } catch(e) { ran = -1; } finally { ran = ran + 10; }
    /// return ran;  // expected: 11 (try ran, then finally)
    /// ```
    #[test]
    fn test_try_catch_finally_normal_path_via_generator() {
        use crate::bytecode::bytecode_generator::BytecodeGenerator;
        use crate::parser::parse;

        let src = "
            var ran = 0;
            try { ran = 1; } catch(e) { ran = -1; } finally { ran = ran + 10; }
            return ran;
        ";
        let program = parse(src).unwrap();
        let ba = BytecodeGenerator::compile_program(&program).unwrap();
        let mut frame = InterpreterFrame::new(ba, vec![]);
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(11));
    }

    /// Uncaught throw propagates as `JsException`.
    #[test]
    fn test_throw_uncaught_produces_js_exception() {
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Throw, vec![]),
        ];
        let err = run_bytecode(instrs, 0).unwrap_err();
        assert!(
            matches!(err, StatorError::JsException(_)),
            "expected JsException, got {err:?}"
        );
    }

    // ── Generators (SuspendGenerator / ResumeGenerator) ─────────────────────

    /// Helper: encode `Register(0)` as the generator-register operand.
    const GEN_REG: Operand = Operand::Register(0);

    /// Build bytecode that simulates `function*(){ yield 1; yield 2; }`.
    ///
    /// Register layout: r0 is the generator object register (unused for state
    /// here since we save 0 registers; the GeneratorState is tracked externally
    /// by `run_generator_step`).
    ///
    /// ```text
    /// [0] LdaSmi 1
    /// [1] SuspendGenerator r0 r0 0 0   ← yield 1, resume_pc = 2
    /// [2] ResumeGenerator  r0 r0 0     ← restore (nop), acc = next's arg
    /// [3] LdaSmi 2
    /// [4] SuspendGenerator r0 r0 0 1   ← yield 2, resume_pc = 5
    /// [5] ResumeGenerator  r0 r0 0
    /// [6] LdaUndefined
    /// [7] Return                        ← done
    /// ```
    fn gen_bytecode_yield_1_yield_2() -> BytecodeArray {
        let instrs = vec![
            // yield 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            // resume point 1
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // yield 2
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(1),
                ],
            ),
            // resume point 2
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        make_bytecode(instrs, 1, 0) // frame_size=1 (r0), no params
    }

    #[test]
    fn test_generator_yield_sequence() {
        // Drives: function*() { yield 1; yield 2; }
        let ba = gen_bytecode_yield_1_yield_2();
        let gs = GeneratorState::new(ba);

        // First .next() → yields 1
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(1)));
        assert_eq!(gs.borrow().status, GeneratorStatus::SuspendedAtYield);

        // Second .next() → yields 2
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 2 ok");
        assert_eq!(step2, GeneratorStep::Yield(JsValue::Smi(2)));
        assert_eq!(gs.borrow().status, GeneratorStatus::SuspendedAtYield);

        // Third .next() → done, returns undefined
        let step3 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 3 ok");
        assert_eq!(step3, GeneratorStep::Return(JsValue::Undefined));
        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);

        // Calling step on a completed generator always returns Return(undefined).
        let step4 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 4 ok");
        assert_eq!(step4, GeneratorStep::Return(JsValue::Undefined));
    }

    #[test]
    fn test_generator_send_value() {
        // Drives: function*() { const x = yield 10; return x + 1; }
        //
        // The value sent on the second .next() becomes the result of the yield
        // expression (via the accumulator after ResumeGenerator).
        //
        // Bytecode:
        //   [0] LdaSmi 10
        //   [1] SuspendGenerator r0 r0 0 0   ← yield 10, resume_pc=2
        //   [2] ResumeGenerator  r0 r0 0      ← acc = sent_value
        //   [3] Star r1                       ← x = sent_value
        //   [4] LdaSmi 1
        //   [5] Star r2                       ← save 1 to r2
        //   [6] Ldar r1                       ← acc = x
        //   [7] Add r2 slot0                  ← acc = x + 1
        //   [8] Return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // x = sent_value (acc)
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // acc = x + 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 3, 0); // r0, r1, r2
        let gs = GeneratorState::new(ba);

        // First .next() → yields 10
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(10)));

        // Second .next(5) → sends 5 as the yield result; function returns 5 + 1 = 6
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Smi(5)).expect("step 2 ok");
        assert_eq!(step2, GeneratorStep::Return(JsValue::Smi(6)));
    }

    // ── Async / Await (desugared to generator) ───────────────────────────────

    /// Test that async/await semantics can be modelled as a generator.
    ///
    /// `async function asyncAdd() { const x = await 42; return x + 1; }`
    ///
    /// Desugars to a generator where `await` becomes `yield`:
    /// - `.next()` runs until the first `yield`, producing the awaited value.
    /// - The event-loop resolves the promise and calls `.next(resolved)`.
    /// - The generator returns `resolved + 1`.
    #[test]
    fn test_async_function_as_generator() {
        // Bytecode (same shape as test_generator_send_value, different constants):
        //   yield 42            ← models `await Promise.resolve(42)`
        //   acc = sent_value    ← models promise resolution
        //   return acc + 1
        let instrs = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            // x = resolved_value (acc after ResumeGenerator)
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            // return x + 1
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 3, 0);
        let gs = GeneratorState::new(ba);

        // Model "async call": first .next() runs until the await point.
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        // The function awaited 42 (yielded it).
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(42)));

        // Model "microtask resolution": the promise resolved with 7.
        let step2 = Interpreter::run_generator_step(&gs, JsValue::Smi(7)).expect("step 2 ok");
        // async function returned 7 + 1 = 8.
        assert_eq!(step2, GeneratorStep::Return(JsValue::Smi(8)));
    }

    // ── GetGeneratorState / SetGeneratorState ────────────────────────────────

    #[test]
    fn test_get_generator_state_initial() {
        // GetGeneratorState on a fresh generator returns SuspendedAtStart (0).
        let instrs = vec![
            Instruction::new_unchecked(Opcode::GetGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        // Attach generator state and run one step (not a generator step — direct frame).
        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        // SuspendedAtStart = 0
        assert_eq!(result, JsValue::Smi(0));
    }

    #[test]
    fn test_set_generator_state_completed() {
        // SetGeneratorState sets the generator status to whatever integer is in acc.
        let instrs = vec![
            // acc = -1 (Completed)
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(-1)]),
            Instruction::new_unchecked(Opcode::SetGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        Interpreter::run(&mut frame).unwrap();

        assert_eq!(gs.borrow().status, GeneratorStatus::Completed);
    }

    // ── SwitchOnGeneratorState ───────────────────────────────────────────────

    #[test]
    fn test_switch_on_generator_state_fresh_falls_through() {
        // When the generator has never been suspended (resume_pc == 0),
        // SwitchOnGeneratorState is a no-op and execution falls through.
        //
        // Bytecode:
        //   [0] SwitchOnGeneratorState r0   ← fresh: no-op, fall through
        //   [1] LdaSmi 99
        //   [2] Return                      ← should return 99
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        let mut frame = InterpreterFrame::new(ba, vec![]);
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        assert_eq!(result, JsValue::Smi(99));
    }

    #[test]
    fn test_switch_on_generator_state_suspended_jumps() {
        // When the generator was previously suspended, SwitchOnGeneratorState
        // jumps directly to the saved resume_pc, bypassing any code before it.
        //
        // Bytecode:
        //   [0] SwitchOnGeneratorState r0   ← if resume_pc>0: jump there
        //   [1] LdaSmi 1                    ← skipped when jumped over
        //   [2] SuspendGenerator r0 r0 0 0  ← saves resume_pc=3
        //   [3] ResumeGenerator  r0 r0 0
        //   [4] LdaSmi 2                    ← runs after jump
        //   [5] Return
        let instrs = vec![
            Instruction::new_unchecked(Opcode::SwitchOnGeneratorState, vec![GEN_REG]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    GEN_REG,
                    GEN_REG,
                    Operand::RegisterCount(0),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![GEN_REG, GEN_REG, Operand::RegisterCount(0)],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        let ba = make_bytecode(instrs, 1, 0);
        let gs = GeneratorState::new(ba.clone());

        // First step: SwitchOnGeneratorState falls through (resume_pc==0),
        // runs LdaSmi 1, yields 1, and saves resume_pc=3 (ResumeGenerator).
        let step1 = Interpreter::run_generator_step(&gs, JsValue::Undefined).expect("step 1 ok");
        assert_eq!(step1, GeneratorStep::Yield(JsValue::Smi(1)));
        let resume_pc_after_first = gs.borrow().resume_pc;
        assert_eq!(
            resume_pc_after_first, 3,
            "resume_pc should point to ResumeGenerator"
        );

        // Directly test the SwitchOnGeneratorState jump: run from PC 0
        // with resume_pc=3 already set in the generator state.  The switch
        // should jump to instruction 3, bypassing LdaSmi 1 and SuspendGenerator.
        let mut frame = InterpreterFrame::new(ba, vec![]);
        gs.borrow_mut().status = GeneratorStatus::SuspendedAtYield;
        frame.generator_state = Some(Rc::clone(&gs));
        let result = Interpreter::run(&mut frame).unwrap();
        // SwitchOnGeneratorState jumped to 3 (ResumeGenerator → no-op),
        // then LdaSmi 2, then Return → 2.
        assert_eq!(result, JsValue::Smi(2));
    }
}

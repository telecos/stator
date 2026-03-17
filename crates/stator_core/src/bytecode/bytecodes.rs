//! Bytecode instruction set matching V8 Ignition semantics.
//!
//! # Overview
//!
//! The Stator VM uses a register-based bytecode inspired by V8's Ignition
//! interpreter.  Every instruction consists of a one-byte [`Opcode`] followed
//! by zero or more [`Operand`]s whose sizes are governed by an
//! [`OperandWidth`] prefix.
//!
//! ## Operand encoding
//!
//! | Width prefix | Opcode byte | Per-operand size |
//! |---|---|---|
//! | *(none)*     | 1 byte      | 1 byte (narrow)  |
//! | [`Opcode::Wide`]      | 1 byte      | 2 bytes LE       |
//! | [`Opcode::ExtraWide`] | 1 byte      | 4 bytes LE       |
//!
//! The [`encode`] function writes the minimum-width encoding automatically.
//! [`decode`] reconstructs an [`Instruction`] sequence from raw bytes.
//!
//! ## Categories
//!
//! - **Load / Store** — accumulator ↔ register / context / global / lookup
//! - **Arithmetic** — binary ops, unary ops, increment/decrement
//! - **Comparison** — equality, ordering, type tests
//! - **Property** — named and keyed property access
//! - **Call** — function invocations and runtime calls
//! - **Construct** — `new` expressions
//! - **Control flow** — unconditional / conditional jumps, return
//! - **Exception** — throw, rethrow, pending-message
//! - **Context** — push/pop scope contexts
//! - **Generators / Async** — suspend, resume, state accessors
//! - **For-in** — enumeration helpers
//! - **Object / Array / Closure construction** — literal and closure creators
//!
//! # Example
//!
//! ```
//! use stator_core::bytecode::bytecodes::{
//!     Instruction, Operand, Opcode, encode, decode,
//! };
//!
//! // Build a tiny sequence: load 42, store to r0, return.
//! let instructions = vec![
//!     Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
//!     Instruction::new_unchecked(Opcode::Star,   vec![Operand::Register(0)]),
//!     Instruction::new_unchecked(Opcode::Return, vec![]),
//! ];
//!
//! let bytes = encode(&instructions);
//! let decoded = decode(&bytes).expect("valid bytecode");
//! assert_eq!(decoded, instructions);
//! ```

use crate::error::{StatorError, StatorResult};

// ─────────────────────────────────────────────────────────────────────────────
// OperandType
// ─────────────────────────────────────────────────────────────────────────────

/// The semantic type of a single bytecode operand.
///
/// Each [`Opcode`] carries a fixed, statically-known list of `OperandType`s
/// returned by [`Opcode::operand_types`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandType {
    /// A virtual register index.
    Register,
    /// A count of consecutive virtual registers.
    RegisterCount,
    /// A signed integer immediate value.
    Immediate,
    /// An unsigned index into the enclosing function's constant pool.
    ConstantPoolIdx,
    /// A 16-bit runtime function identifier.
    RuntimeId,
    /// An index into the inline-cache feedback vector.
    FeedbackSlot,
    /// A signed byte offset used as a jump target.
    JumpOffset,
    /// A small bit-flag (e.g. language mode, property kind, closure flags).
    Flag,
}

// ─────────────────────────────────────────────────────────────────────────────
// OperandWidth
// ─────────────────────────────────────────────────────────────────────────────

/// Bytes per operand slot, controlled by a [`Opcode::Wide`] or
/// [`Opcode::ExtraWide`] prefix opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperandWidth {
    /// 1 byte per operand (no prefix).
    Narrow = 1,
    /// 2 bytes per operand (preceded by [`Opcode::Wide`]).
    Wide = 2,
    /// 4 bytes per operand (preceded by [`Opcode::ExtraWide`]).
    ExtraWide = 4,
}

// ─────────────────────────────────────────────────────────────────────────────
// Operand
// ─────────────────────────────────────────────────────────────────────────────

/// A decoded operand value.
///
/// The variant chosen must match the [`OperandType`] declared by the
/// enclosing instruction's [`Opcode::operand_types`] slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// Virtual register index.
    Register(u32),
    /// Count of consecutive registers.
    RegisterCount(u32),
    /// Signed integer immediate.
    Immediate(i32),
    /// Constant pool index.
    ConstantPoolIdx(u32),
    /// Runtime function identifier (≤ [`u16::MAX`]).
    RuntimeId(u32),
    /// Feedback vector slot index.
    FeedbackSlot(u32),
    /// Signed jump-offset in bytes.
    JumpOffset(i32),
    /// Small flag value.
    Flag(u8),
}

impl Operand {
    /// Return the minimum [`OperandWidth`] required to encode this operand.
    fn required_width(self) -> OperandWidth {
        match self {
            Operand::Register(v)
            | Operand::RegisterCount(v)
            | Operand::ConstantPoolIdx(v)
            | Operand::FeedbackSlot(v)
            | Operand::RuntimeId(v) => {
                if v <= u8::MAX as u32 {
                    OperandWidth::Narrow
                } else if v <= u16::MAX as u32 {
                    OperandWidth::Wide
                } else {
                    OperandWidth::ExtraWide
                }
            }
            Operand::Immediate(v) | Operand::JumpOffset(v) => {
                if (i8::MIN as i32..=i8::MAX as i32).contains(&v) {
                    OperandWidth::Narrow
                } else if (i16::MIN as i32..=i16::MAX as i32).contains(&v) {
                    OperandWidth::Wide
                } else {
                    OperandWidth::ExtraWide
                }
            }
            // Flag is always one byte.
            Operand::Flag(_) => OperandWidth::Narrow,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instruction
// ─────────────────────────────────────────────────────────────────────────────

/// A fully decoded bytecode instruction: an [`Opcode`] paired with its
/// [`Operand`] list.
///
/// The number and types of operands are always determined by
/// [`Opcode::operand_types`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// The operation to perform.
    pub opcode: Opcode,
    /// Operands in the order specified by [`Opcode::operand_types`].
    pub operands: Vec<Operand>,
}

impl Instruction {
    /// Construct an instruction, verifying that the operand count matches the
    /// opcode's declared operand list.
    ///
    /// Returns an error if the wrong number of operands is supplied.
    pub fn new(opcode: Opcode, operands: Vec<Operand>) -> StatorResult<Self> {
        let expected = opcode.operand_types().len();
        if operands.len() != expected {
            return Err(StatorError::Internal(format!(
                "{opcode:?} expects {expected} operand(s), got {}",
                operands.len()
            )));
        }
        Ok(Self { opcode, operands })
    }

    /// Construct an instruction without operand-count validation.
    ///
    /// Prefer [`Instruction::new`] unless you are certain the operand count
    /// is correct (e.g. inside the decoder).
    pub fn new_unchecked(opcode: Opcode, operands: Vec<Operand>) -> Self {
        Self { opcode, operands }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Opcode
// ─────────────────────────────────────────────────────────────────────────────

/// All bytecode operations understood by the Stator VM.
///
/// The discriminants are contiguous starting at `0`; the raw byte value is
/// used directly in the encoded bytecode stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    // ── Load / Store ──────────────────────────────────────────────────────
    /// Load the integer zero into the accumulator.
    LdaZero,
    /// Load a signed-integer immediate into the accumulator. `[imm]`
    LdaSmi,
    /// Load `undefined` into the accumulator.
    LdaUndefined,
    /// Load the internal hole sentinel into the accumulator.
    ///
    /// Used to initialise `let`/`const`/`class` bindings before their
    /// declaration is reached (Temporal Dead Zone).
    LdaTheHole,
    /// Load `null` into the accumulator.
    LdaNull,
    /// Load `true` into the accumulator.
    LdaTrue,
    /// Load `false` into the accumulator.
    LdaFalse,
    /// Load a constant-pool entry into the accumulator. `[idx]`
    LdaConstant,
    /// Load a global variable into the accumulator. `[name_idx, slot]`
    LdaGlobal,
    /// Load a global variable inside a `typeof` context. `[name_idx, slot]`
    LdaGlobalInsideTypeof,
    /// Store the accumulator to a global variable. `[name_idx, slot]`
    StaGlobal,
    /// Load a context slot. `[ctx_reg, slot_idx, depth]`
    LdaContextSlot,
    /// Load an immutable context slot. `[ctx_reg, slot_idx, depth]`
    LdaImmutableContextSlot,
    /// Load the current (innermost) context slot. `[slot_idx]`
    LdaCurrentContextSlot,
    /// Load an immutable slot from the current context. `[slot_idx]`
    LdaImmutableCurrentContextSlot,
    /// Store the accumulator to a context slot. `[ctx_reg, slot_idx, depth]`
    StaContextSlot,
    /// Store the accumulator to the current context slot. `[slot_idx]`
    StaCurrentContextSlot,
    /// Dynamic lookup of a variable name. `[name_idx]`
    LdaLookupSlot,
    /// Dynamic lookup resolving to a context slot. `[name_idx, slot_idx, depth]`
    LdaLookupContextSlot,
    /// Dynamic lookup resolving to a global slot. `[name_idx, slot, depth]`
    LdaLookupGlobalSlot,
    /// Dynamic lookup inside `typeof`. `[name_idx]`
    LdaLookupSlotInsideTypeof,
    /// Dynamic lookup resolving to a context slot, inside `typeof`. `[name_idx, slot_idx, depth]`
    LdaLookupContextSlotInsideTypeof,
    /// Dynamic lookup resolving to a global slot, inside `typeof`. `[name_idx, slot, depth]`
    LdaLookupGlobalSlotInsideTypeof,
    /// Dynamic store of a variable name. `[name_idx, flags]`
    StaLookupSlot,
    /// Load a register into the accumulator. `[reg]`
    Ldar,
    /// Store the accumulator into a register. `[reg]`
    Star,
    /// Copy one register to another. `[src, dst]`
    Mov,

    // ── Property access ───────────────────────────────────────────────────
    /// Load a named property. `[obj, name_idx, slot]`
    LdaNamedProperty,
    /// Load a named property from `super`. `[obj, name_idx, slot]`
    LdaNamedPropertyFromSuper,
    /// Load a keyed property. `[obj, slot]`
    LdaKeyedProperty,
    /// Load a keyed property during `for-in` enumeration. `[obj, enum_index, cache_array]`
    LdaEnumeratedKeyedProperty,
    /// Store the accumulator as a named property. `[obj, name_idx, slot]`
    StaNamedProperty,
    /// Store the accumulator as a named own property. `[obj, name_idx, slot]`
    StaNamedOwnProperty,
    /// Store the accumulator as a keyed property. `[obj, key_reg, slot]`
    StaKeyedProperty,
    /// Define a named own property. `[obj, name_idx, slot]`
    DefineNamedOwnProperty,
    /// Define a keyed own property. `[obj, key_reg, flags, slot]`
    DefineKeyedOwnProperty,
    /// Store accumulator into an array literal. `[array, index_reg, slot]`
    StaInArrayLiteral,
    /// Define a keyed own property in a literal. `[obj, key_reg, flags, slot]`
    DefineKeyedOwnPropertyInLiteral,
    /// Set an object literal's [[Prototype]] from the accumulator. `[obj]`
    SetLiteralPrototype,
    /// Define a getter on an object (accumulator holds the getter function).
    /// `[obj, name_idx, slot]`
    DefineGetterProperty,
    /// Define a setter on an object (accumulator holds the setter function).
    /// `[obj, name_idx, slot]`
    DefineSetterProperty,
    /// Define a getter with a computed key (accumulator holds the getter function).
    /// `[obj, key_reg, slot]`
    DefineKeyedGetterProperty,
    /// Define a setter with a computed key (accumulator holds the setter function).
    /// `[obj, key_reg, slot]`
    DefineKeyedSetterProperty,
    /// Collect type-profile information for the accumulator. `[position]`
    CollectTypeProfile,

    // ── Arithmetic ────────────────────────────────────────────────────────
    /// `accumulator + reg`. `[reg, slot]`
    Add,
    /// `accumulator - reg`. `[reg, slot]`
    Sub,
    /// `accumulator * reg`. `[reg, slot]`
    Mul,
    /// `accumulator / reg`. `[reg, slot]`
    Div,
    /// `accumulator % reg`. `[reg, slot]`
    Mod,
    /// `accumulator ** reg`. `[reg, slot]`
    Exp,
    /// `accumulator | reg`. `[reg, slot]`
    BitwiseOr,
    /// `accumulator ^ reg`. `[reg, slot]`
    BitwiseXor,
    /// `accumulator & reg`. `[reg, slot]`
    BitwiseAnd,
    /// `accumulator << reg`. `[reg, slot]`
    ShiftLeft,
    /// `accumulator >> reg`. `[reg, slot]`
    ShiftRight,
    /// `accumulator >>> reg`. `[reg, slot]`
    ShiftRightLogical,
    /// `accumulator + imm`. `[imm, slot]`
    AddSmi,
    /// `accumulator - imm`. `[imm, slot]`
    SubSmi,
    /// `accumulator * imm`. `[imm, slot]`
    MulSmi,
    /// `accumulator / imm`. `[imm, slot]`
    DivSmi,
    /// `accumulator % imm`. `[imm, slot]`
    ModSmi,
    /// `accumulator ** imm`. `[imm, slot]`
    ExpSmi,
    /// `accumulator | imm`. `[imm, slot]`
    BitwiseOrSmi,
    /// `accumulator ^ imm`. `[imm, slot]`
    BitwiseXorSmi,
    /// `accumulator & imm`. `[imm, slot]`
    BitwiseAndSmi,
    /// `accumulator << imm`. `[imm, slot]`
    ShiftLeftSmi,
    /// `accumulator >> imm`. `[imm, slot]`
    ShiftRightSmi,
    /// `accumulator >>> imm`. `[imm, slot]`
    ShiftRightLogicalSmi,
    /// `++accumulator`. `[slot]`
    Inc,
    /// `--accumulator`. `[slot]`
    Dec,
    /// `-accumulator`. `[slot]`
    Negate,
    /// `~accumulator`. `[slot]`
    BitwiseNot,
    /// `!ToBoolean(accumulator)`, updating accumulator.
    ToBooleanLogicalNot,
    /// `!accumulator` (boolean accumulator only).
    LogicalNot,
    /// `typeof accumulator`. `[slot]`
    TypeOf,
    /// `delete obj[key]` in strict mode. `[reg]`
    DeletePropertyStrict,
    /// `delete obj[key]` in sloppy mode. `[reg]`
    DeletePropertySloppy,

    // ── Comparison ────────────────────────────────────────────────────────
    /// `accumulator == reg`. `[reg, slot]`
    TestEqual,
    /// `accumulator != reg`. `[reg, slot]`
    TestNotEqual,
    /// `accumulator === reg`. `[reg, slot]`
    TestEqualStrict,
    /// `accumulator < reg`. `[reg, slot]`
    TestLessThan,
    /// `accumulator > reg`. `[reg, slot]`
    TestGreaterThan,
    /// `accumulator <= reg`. `[reg, slot]`
    TestLessThanOrEqual,
    /// `accumulator >= reg`. `[reg, slot]`
    TestGreaterThanOrEqual,
    /// `accumulator === reg` (reference equality, no feedback). `[reg]`
    TestReferenceEqual,
    /// `accumulator instanceof reg`. `[reg, slot]`
    TestInstanceOf,
    /// `accumulator in reg`. `[reg, slot]`
    TestIn,
    /// True if accumulator is undetectable (e.g. `document.all`).
    TestUndetectable,
    /// True if accumulator is `null`.
    TestNull,
    /// True if accumulator is `undefined`.
    TestUndefined,
    /// Type-of test against a literal-type flag. `[flags]`
    TestTypeOf,

    // ── Casts ─────────────────────────────────────────────────────────────
    /// `ToName(accumulator)`, storing result in `dst`. `[dst]`
    ToName,
    /// `ToNumber(accumulator)`. `[slot]`
    ToNumber,
    /// `ToNumeric(accumulator)`. `[slot]`
    ToNumeric,
    /// `ToObject(accumulator)`, storing result in `dst`. `[dst]`
    ToObject,
    /// `ToString(accumulator)`.
    ToString,
    /// `ToBoolean(accumulator)`. `[slot]`
    ToBoolean,

    // ── Calls ─────────────────────────────────────────────────────────────
    /// Call with any receiver. `[callable, args_start, args_count, slot]`
    CallAnyReceiver,
    /// Call with a property receiver. `[callable, args_start, args_count, slot]`
    CallProperty,
    /// Call with property receiver, zero args. `[callable, receiver, slot]`
    CallProperty0,
    /// Call with property receiver, one arg. `[callable, receiver, arg1, slot]`
    CallProperty1,
    /// Call with property receiver, two args. `[callable, receiver, arg1, arg2, slot]`
    CallProperty2,
    /// Call with `undefined` receiver, zero args. `[callable, slot]`
    CallUndefinedReceiver0,
    /// Call with `undefined` receiver, one arg. `[callable, arg1, slot]`
    CallUndefinedReceiver1,
    /// Call with `undefined` receiver, two args. `[callable, arg1, arg2, slot]`
    CallUndefinedReceiver2,
    /// Call with spread argument. `[callable, args_start, args_count, slot]`
    CallWithSpread,
    /// Call a C++ runtime function. `[function_id, args_start, args_count]`
    CallRuntime,
    /// Call a C++ runtime function returning a pair. `[function_id, args_start, args_count, first_return]`
    CallRuntimeForPair,
    /// Call a JS runtime utility. `[context_idx, args_start, args_count]`
    CallJSRuntime,
    /// Invoke a JS intrinsic. `[function_id, args_start, args_count]`
    InvokeIntrinsic,
    /// Direct eval call (callee is the bare `eval` identifier).
    /// At runtime, if the callee is still the built-in eval function the
    /// code is executed sharing the caller's variable environment; otherwise
    /// a normal call is performed.  `[callable, args_start, args_count, slot]`
    CallDirectEval,

    /// Tail call in return position (ES2015 proper tail calls).
    /// Same operand layout as `CallAnyReceiver`:
    /// `[callable, args_start, args_count, slot]`.
    ///
    /// The interpreter reuses the current call frame instead of pushing a
    /// new one, preventing stack growth for recursive tail-position calls.
    TailCall,

    // ── Construct ─────────────────────────────────────────────────────────
    /// `new constructor(args…)`. `[constructor, args_start, args_count, slot]`
    Construct,
    /// `new constructor(...spread)`. `[constructor, args_start, args_count, slot]`
    ConstructWithSpread,
    /// `new constructor()`, forwarding all args from enclosing call.
    /// `[constructor, slot]`
    ConstructForwardAllArgs,

    // ── Iterators ─────────────────────────────────────────────────────────
    /// Obtain a sync iterator. `[iterable, load_slot, call_slot]`
    GetIterator,
    /// Obtain an async iterator. `[iterable, load_slot, call_slot]`
    GetAsyncIterator,
    /// Advance an iterator one step. `[iterator_reg, value_out_reg]`
    ///
    /// Stores the next value into `value_out_reg` and puts the `done` boolean
    /// into the accumulator.  When the iterator is exhausted, `value_out_reg`
    /// is set to `undefined` and the accumulator is `true`.
    IteratorNext,

    /// Close an iterator by calling its `.return()` method (if any).
    /// Emitted before `break` / `return` inside `for…of` loops.
    /// `[iterator_reg]`
    IteratorClose,

    /// Copy all own enumerable properties from source to target.
    /// `[target_reg, source_reg]`
    CopyDataProperties,

    // ── Control flow ──────────────────────────────────────────────────────
    /// Back-edge jump (loop). `[offset, loop_depth, slot]`
    JumpLoop,
    /// Unconditional forward jump. `[offset]`
    Jump,
    /// Unconditional jump via constant pool. `[idx]`
    JumpConstant,
    /// Jump if accumulator is `true`. `[offset]`
    JumpIfTrue,
    /// Jump if accumulator is `true` (constant pool target). `[idx]`
    JumpIfTrueConstant,
    /// Jump if accumulator is `false`. `[offset]`
    JumpIfFalse,
    /// Jump if accumulator is `false` (constant pool target). `[idx]`
    JumpIfFalseConstant,
    /// Jump if accumulator is `null`. `[offset]`
    JumpIfNull,
    /// Jump if accumulator is not `null`. `[offset]`
    JumpIfNotNull,
    /// Jump if accumulator is `undefined`. `[offset]`
    JumpIfUndefined,
    /// Jump if accumulator is not `undefined`. `[offset]`
    JumpIfNotUndefined,
    /// Jump if accumulator is `undefined` or `null`. `[offset]`
    JumpIfUndefinedOrNull,
    /// Jump if accumulator is a JS receiver (object/function). `[offset]`
    JumpIfJSReceiver,
    /// Jump if for-in index reached the cache length. `[offset, index, cache_length]`
    JumpIfForInDone,
    /// Jump if `ToBoolean(accumulator)` is `true`. `[offset]`
    JumpIfToBooleanTrue,
    /// Jump if `ToBoolean(accumulator)` is `false`. `[offset]`
    JumpIfToBooleanFalse,
    /// Jump-if-to-boolean-true, constant pool target. `[idx]`
    JumpIfToBooleanTrueConstant,
    /// Jump-if-to-boolean-false, constant pool target. `[idx]`
    JumpIfToBooleanFalseConstant,
    /// Jump if `null`, constant pool target. `[idx]`
    JumpIfNullConstant,
    /// Jump if not `null`, constant pool target. `[idx]`
    JumpIfNotNullConstant,
    /// Jump if `undefined`, constant pool target. `[idx]`
    JumpIfUndefinedConstant,
    /// Jump if not `undefined`, constant pool target. `[idx]`
    JumpIfNotUndefinedConstant,
    /// Jump if `undefined` or `null`, constant pool target. `[idx]`
    JumpIfUndefinedOrNullConstant,
    /// Jump if JS receiver, constant pool target. `[idx]`
    JumpIfJSReceiverConstant,

    // ── Return / Throw ────────────────────────────────────────────────────
    /// Return the accumulator value.
    Return,
    /// Throw a `ReferenceError` if the accumulator is the hole. `[name_idx]`
    ThrowReferenceErrorIfHole,
    /// Throw if `super()` has not yet been called.
    ThrowSuperNotCalledIfHole,
    /// Throw if `super()` has already been called.
    ThrowSuperAlreadyCalledIfNotHole,
    /// Throw the accumulator value.
    Throw,
    /// Re-throw a caught exception.
    ReThrow,
    /// Set the pending-exception message from the accumulator.
    SetPendingMessage,

    // ── Debugger ──────────────────────────────────────────────────────────
    /// Trigger a debugger breakpoint.
    Debugger,

    // ── Closure / Literal / Context construction ──────────────────────────
    /// Create a closure from a shared-function-info entry. `[func_idx, slot, flags]`
    CreateClosure,
    /// Create a block-scope context. `[scope_idx]`
    CreateBlockContext,
    /// Create a catch-scope context. `[exception_reg, scope_idx]`
    CreateCatchContext,
    /// Create a regular function context. `[scope_idx, slots]`
    CreateFunctionContext,
    /// Create an eval context. `[scope_idx, slots]`
    CreateEvalContext,
    /// Create a `with`-statement context. `[obj_reg, scope_idx]`
    CreateWithContext,
    /// Create a mapped `arguments` object.
    CreateMappedArguments,
    /// Create an unmapped `arguments` object.
    CreateUnmappedArguments,
    /// Create the rest-parameter array.
    CreateRestParameter,
    /// Create a regexp literal. `[pattern_idx, slot, flags]`
    CreateRegExpLiteral,
    /// Create an array literal. `[elements_idx, slot, flags]`
    CreateArrayLiteral,
    /// Create an array by spreading an iterable.
    CreateArrayFromIterable,
    /// Create an empty array literal. `[slot]`
    CreateEmptyArrayLiteral,
    /// Create an object literal. `[boilerplate_idx, slot, flags]`
    CreateObjectLiteral,
    /// Create an empty object literal `{}`.
    CreateEmptyObjectLiteral,
    /// Create an object by spreading an iterable.
    CreateObjectFromIterable,

    // ── Context ───────────────────────────────────────────────────────────
    /// Push a new context, storing the old one in `reg`. `[reg]`
    PushContext,
    /// Pop to the context stored in `reg`. `[reg]`
    PopContext,

    // ── For-in ────────────────────────────────────────────────────────────
    /// Enumerate own properties of `obj`. `[obj]`
    ForInEnumerate,
    /// Prepare the for-in cache. `[cache_array, slot]`
    ForInPrepare,
    /// Fetch the next for-in key. `[receiver, cache_array, cache_type, slot]`
    ForInNext,
    /// Advance the for-in index. `[index]`
    ForInStep,

    // ── Generators / Async ────────────────────────────────────────────────
    /// Load or create a template-object literal. `[template_idx, slot]`
    GetTemplateObject,
    /// Emit a stack-overflow / interrupt check.
    StackCheck,
    /// Record the current source position. `[position]`
    SetExpressionPosition,
    /// Record source position measured from the end of the source. `[position]`
    SetExpressionPositionFromEnd,
    /// Resume a suspended generator, restoring registers. `[generator, regs_start, regs_count]`
    ResumeGenerator,
    /// Read the state of a generator object. `[generator]`
    GetGeneratorState,
    /// Suspend the current generator. `[generator, regs_start, regs_count, suspend_id]`
    SuspendGenerator,
    /// Write the state of a generator object. `[generator]`
    SetGeneratorState,
    /// Dispatch to the generator's saved resume point, or fall through if
    /// the generator has not yet been suspended. `[generator]`
    SwitchOnGeneratorState,

    // ── Class ─────────────────────────────────────────────────────────────
    /// Create a class constructor with prototype chain setup.
    /// `[constructor_func_idx, super_reg, slot]`
    CreateClass,
    /// Check whether the accumulator has the private brand in `brand_reg`.
    /// `[obj, brand_reg]`
    TestPrivateBrand,
    /// Define a private brand on `obj`. `[obj]`
    DefinePrivateBrand,

    // ── Module ────────────────────────────────────────────────────────────
    /// Load a module variable (import binding) into the accumulator.
    /// `[module_request_idx, cell_idx]`
    ///
    /// `module_request_idx` is a constant-pool index for the source module
    /// specifier string and `cell_idx` identifies the binding cell inside
    /// that module's environment.  Live bindings are resolved at load time.
    LdaModuleVariable,
    /// Store the accumulator to a module variable (export binding).
    /// `[module_request_idx, cell_idx]`
    ///
    /// Used for `export let`/`export var` — writes go through the binding
    /// cell so importers see the updated value (live binding semantics).
    StaModuleVariable,
    /// Load the `import.meta` object for the current module into the accumulator.
    LdaImportMeta,
    /// Load `new.target` into the accumulator.
    ///
    /// Inside a `[[Construct]]` call this is the constructor function;
    /// in a normal call it is `undefined`.
    LdaNewTarget,
    /// Create a module namespace object (`import * as ns`) and load it into
    /// the accumulator. `[module_request_idx]`
    ///
    /// `module_request_idx` is a constant-pool index holding the module
    /// specifier string.
    GetModuleNamespace,

    // ── Encoding prefixes ─────────────────────────────────────────────────
    /// Prefix: all operands in the following instruction use 2-byte (wide) encoding.
    Wide,
    /// Prefix: all operands in the following instruction use 4-byte (extra-wide) encoding.
    ExtraWide,
    /// Illegal / unreachable trap instruction.
    Illegal,
}

/// The maximum valid opcode byte (= `Opcode::Illegal as u8`).
const MAX_OPCODE: u8 = Opcode::Illegal as u8;

impl Opcode {
    /// Convert a raw byte to an [`Opcode`], returning an error for
    /// out-of-range values.
    pub fn try_from_u8(byte: u8) -> StatorResult<Self> {
        if byte > MAX_OPCODE {
            return Err(StatorError::Internal(format!(
                "unknown opcode byte: 0x{byte:02x}"
            )));
        }
        // SAFETY: `Opcode` is `#[repr(u8)]` with contiguous discriminants
        // 0..=MAX_OPCODE, and we have verified `byte` is within that range.
        Ok(unsafe { std::mem::transmute::<u8, Opcode>(byte) })
    }

    /// Return the static list of [`OperandType`]s for this opcode.
    ///
    /// The length of the returned slice equals the number of operands that
    /// must follow this opcode in the encoded byte stream.
    pub fn operand_types(self) -> &'static [OperandType] {
        use OperandType::*;
        match self {
            // Load / Store
            Opcode::LdaZero => &[],
            Opcode::LdaSmi => &[Immediate],
            Opcode::LdaUndefined => &[],
            Opcode::LdaTheHole => &[],
            Opcode::LdaNull => &[],
            Opcode::LdaTrue => &[],
            Opcode::LdaFalse => &[],
            Opcode::LdaConstant => &[ConstantPoolIdx],
            Opcode::LdaGlobal => &[ConstantPoolIdx, FeedbackSlot],
            Opcode::LdaGlobalInsideTypeof => &[ConstantPoolIdx, FeedbackSlot],
            Opcode::StaGlobal => &[ConstantPoolIdx, FeedbackSlot],
            Opcode::LdaContextSlot => &[Register, ConstantPoolIdx, Immediate],
            Opcode::LdaImmutableContextSlot => &[Register, ConstantPoolIdx, Immediate],
            Opcode::LdaCurrentContextSlot => &[ConstantPoolIdx],
            Opcode::LdaImmutableCurrentContextSlot => &[ConstantPoolIdx],
            Opcode::StaContextSlot => &[Register, ConstantPoolIdx, Immediate],
            Opcode::StaCurrentContextSlot => &[ConstantPoolIdx],
            Opcode::LdaLookupSlot => &[ConstantPoolIdx],
            Opcode::LdaLookupContextSlot => &[ConstantPoolIdx, ConstantPoolIdx, Immediate],
            Opcode::LdaLookupGlobalSlot => &[ConstantPoolIdx, FeedbackSlot, Immediate],
            Opcode::LdaLookupSlotInsideTypeof => &[ConstantPoolIdx],
            Opcode::LdaLookupContextSlotInsideTypeof => {
                &[ConstantPoolIdx, ConstantPoolIdx, Immediate]
            }
            Opcode::LdaLookupGlobalSlotInsideTypeof => &[ConstantPoolIdx, FeedbackSlot, Immediate],
            Opcode::StaLookupSlot => &[ConstantPoolIdx, Flag],
            Opcode::Ldar => &[Register],
            Opcode::Star => &[Register],
            Opcode::Mov => &[Register, Register],

            // Property
            Opcode::LdaNamedProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::LdaNamedPropertyFromSuper => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::LdaKeyedProperty => &[Register, FeedbackSlot],
            Opcode::LdaEnumeratedKeyedProperty => &[Register, Register, Register],
            Opcode::StaNamedProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::StaNamedOwnProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::StaKeyedProperty => &[Register, Register, FeedbackSlot],
            Opcode::DefineNamedOwnProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::DefineKeyedOwnProperty => &[Register, Register, Flag, FeedbackSlot],
            Opcode::StaInArrayLiteral => &[Register, Register, FeedbackSlot],
            Opcode::DefineKeyedOwnPropertyInLiteral => &[Register, Register, Flag, FeedbackSlot],
            Opcode::SetLiteralPrototype => &[Register],
            Opcode::DefineGetterProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::DefineSetterProperty => &[Register, ConstantPoolIdx, FeedbackSlot],
            Opcode::DefineKeyedGetterProperty => &[Register, Register, FeedbackSlot],
            Opcode::DefineKeyedSetterProperty => &[Register, Register, FeedbackSlot],
            Opcode::CollectTypeProfile => &[Immediate],

            // Arithmetic
            Opcode::Add => &[Register, FeedbackSlot],
            Opcode::Sub => &[Register, FeedbackSlot],
            Opcode::Mul => &[Register, FeedbackSlot],
            Opcode::Div => &[Register, FeedbackSlot],
            Opcode::Mod => &[Register, FeedbackSlot],
            Opcode::Exp => &[Register, FeedbackSlot],
            Opcode::BitwiseOr => &[Register, FeedbackSlot],
            Opcode::BitwiseXor => &[Register, FeedbackSlot],
            Opcode::BitwiseAnd => &[Register, FeedbackSlot],
            Opcode::ShiftLeft => &[Register, FeedbackSlot],
            Opcode::ShiftRight => &[Register, FeedbackSlot],
            Opcode::ShiftRightLogical => &[Register, FeedbackSlot],
            Opcode::AddSmi => &[Immediate, FeedbackSlot],
            Opcode::SubSmi => &[Immediate, FeedbackSlot],
            Opcode::MulSmi => &[Immediate, FeedbackSlot],
            Opcode::DivSmi => &[Immediate, FeedbackSlot],
            Opcode::ModSmi => &[Immediate, FeedbackSlot],
            Opcode::ExpSmi => &[Immediate, FeedbackSlot],
            Opcode::BitwiseOrSmi => &[Immediate, FeedbackSlot],
            Opcode::BitwiseXorSmi => &[Immediate, FeedbackSlot],
            Opcode::BitwiseAndSmi => &[Immediate, FeedbackSlot],
            Opcode::ShiftLeftSmi => &[Immediate, FeedbackSlot],
            Opcode::ShiftRightSmi => &[Immediate, FeedbackSlot],
            Opcode::ShiftRightLogicalSmi => &[Immediate, FeedbackSlot],
            Opcode::Inc => &[FeedbackSlot],
            Opcode::Dec => &[FeedbackSlot],
            Opcode::Negate => &[FeedbackSlot],
            Opcode::BitwiseNot => &[FeedbackSlot],
            Opcode::ToBooleanLogicalNot => &[],
            Opcode::LogicalNot => &[],
            Opcode::TypeOf => &[FeedbackSlot],
            Opcode::DeletePropertyStrict => &[Register],
            Opcode::DeletePropertySloppy => &[Register],

            // Comparison
            Opcode::TestEqual => &[Register, FeedbackSlot],
            Opcode::TestNotEqual => &[Register, FeedbackSlot],
            Opcode::TestEqualStrict => &[Register, FeedbackSlot],
            Opcode::TestLessThan => &[Register, FeedbackSlot],
            Opcode::TestGreaterThan => &[Register, FeedbackSlot],
            Opcode::TestLessThanOrEqual => &[Register, FeedbackSlot],
            Opcode::TestGreaterThanOrEqual => &[Register, FeedbackSlot],
            Opcode::TestReferenceEqual => &[Register],
            Opcode::TestInstanceOf => &[Register, FeedbackSlot],
            Opcode::TestIn => &[Register, FeedbackSlot],
            Opcode::TestUndetectable => &[],
            Opcode::TestNull => &[],
            Opcode::TestUndefined => &[],
            Opcode::TestTypeOf => &[Flag],

            // Casts
            Opcode::ToName => &[Register],
            Opcode::ToNumber => &[FeedbackSlot],
            Opcode::ToNumeric => &[FeedbackSlot],
            Opcode::ToObject => &[Register],
            Opcode::ToString => &[],
            Opcode::ToBoolean => &[FeedbackSlot],

            // Calls
            Opcode::CallAnyReceiver => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::CallProperty => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::CallProperty0 => &[Register, Register, FeedbackSlot],
            Opcode::CallProperty1 => &[Register, Register, Register, FeedbackSlot],
            Opcode::CallProperty2 => &[Register, Register, Register, Register, FeedbackSlot],
            Opcode::CallUndefinedReceiver0 => &[Register, FeedbackSlot],
            Opcode::CallUndefinedReceiver1 => &[Register, Register, FeedbackSlot],
            Opcode::CallUndefinedReceiver2 => &[Register, Register, Register, FeedbackSlot],
            Opcode::CallWithSpread => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::CallRuntime => &[RuntimeId, Register, RegisterCount],
            Opcode::CallRuntimeForPair => &[RuntimeId, Register, RegisterCount, Register],
            Opcode::CallJSRuntime => &[ConstantPoolIdx, Register, RegisterCount],
            Opcode::InvokeIntrinsic => &[RuntimeId, Register, RegisterCount],
            Opcode::CallDirectEval => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::TailCall => &[Register, Register, RegisterCount, FeedbackSlot],

            // Construct
            Opcode::Construct => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::ConstructWithSpread => &[Register, Register, RegisterCount, FeedbackSlot],
            Opcode::ConstructForwardAllArgs => &[Register, FeedbackSlot],

            // Iterators
            Opcode::GetIterator => &[Register, FeedbackSlot, FeedbackSlot],
            Opcode::GetAsyncIterator => &[Register, FeedbackSlot, FeedbackSlot],
            Opcode::IteratorNext => &[Register, Register],
            Opcode::IteratorClose => &[Register],
            Opcode::CopyDataProperties => &[Register, Register],

            // Jumps
            Opcode::JumpLoop => &[JumpOffset, Immediate, FeedbackSlot],
            Opcode::Jump => &[JumpOffset],
            Opcode::JumpConstant => &[ConstantPoolIdx],
            Opcode::JumpIfTrue => &[JumpOffset],
            Opcode::JumpIfTrueConstant => &[ConstantPoolIdx],
            Opcode::JumpIfFalse => &[JumpOffset],
            Opcode::JumpIfFalseConstant => &[ConstantPoolIdx],
            Opcode::JumpIfNull => &[JumpOffset],
            Opcode::JumpIfNotNull => &[JumpOffset],
            Opcode::JumpIfUndefined => &[JumpOffset],
            Opcode::JumpIfNotUndefined => &[JumpOffset],
            Opcode::JumpIfUndefinedOrNull => &[JumpOffset],
            Opcode::JumpIfJSReceiver => &[JumpOffset],
            Opcode::JumpIfForInDone => &[JumpOffset, Register, Register],
            Opcode::JumpIfToBooleanTrue => &[JumpOffset],
            Opcode::JumpIfToBooleanFalse => &[JumpOffset],
            Opcode::JumpIfToBooleanTrueConstant => &[ConstantPoolIdx],
            Opcode::JumpIfToBooleanFalseConstant => &[ConstantPoolIdx],
            Opcode::JumpIfNullConstant => &[ConstantPoolIdx],
            Opcode::JumpIfNotNullConstant => &[ConstantPoolIdx],
            Opcode::JumpIfUndefinedConstant => &[ConstantPoolIdx],
            Opcode::JumpIfNotUndefinedConstant => &[ConstantPoolIdx],
            Opcode::JumpIfUndefinedOrNullConstant => &[ConstantPoolIdx],
            Opcode::JumpIfJSReceiverConstant => &[ConstantPoolIdx],

            // Return / Throw
            Opcode::Return => &[],
            Opcode::ThrowReferenceErrorIfHole => &[ConstantPoolIdx],
            Opcode::ThrowSuperNotCalledIfHole => &[],
            Opcode::ThrowSuperAlreadyCalledIfNotHole => &[],
            Opcode::Throw => &[],
            Opcode::ReThrow => &[],
            Opcode::SetPendingMessage => &[],

            // Debugger
            Opcode::Debugger => &[],

            // Construction
            Opcode::CreateClosure => &[ConstantPoolIdx, FeedbackSlot, Flag],
            Opcode::CreateBlockContext => &[ConstantPoolIdx],
            Opcode::CreateCatchContext => &[Register, ConstantPoolIdx],
            Opcode::CreateFunctionContext => &[ConstantPoolIdx, Immediate],
            Opcode::CreateEvalContext => &[ConstantPoolIdx, Immediate],
            Opcode::CreateWithContext => &[Register, ConstantPoolIdx],
            Opcode::CreateMappedArguments => &[],
            Opcode::CreateUnmappedArguments => &[],
            Opcode::CreateRestParameter => &[],
            Opcode::CreateRegExpLiteral => &[ConstantPoolIdx, FeedbackSlot, Flag],
            Opcode::CreateArrayLiteral => &[ConstantPoolIdx, FeedbackSlot, Flag],
            Opcode::CreateArrayFromIterable => &[],
            Opcode::CreateEmptyArrayLiteral => &[FeedbackSlot],
            Opcode::CreateObjectLiteral => &[ConstantPoolIdx, FeedbackSlot, Flag],
            Opcode::CreateEmptyObjectLiteral => &[],
            Opcode::CreateObjectFromIterable => &[],

            // Context
            Opcode::PushContext => &[Register],
            Opcode::PopContext => &[Register],

            // For-in
            Opcode::ForInEnumerate => &[Register],
            Opcode::ForInPrepare => &[Register, FeedbackSlot],
            Opcode::ForInNext => &[Register, Register, Register, FeedbackSlot],
            Opcode::ForInStep => &[Register],

            // Generators
            Opcode::GetTemplateObject => &[ConstantPoolIdx, FeedbackSlot],
            Opcode::StackCheck => &[],
            Opcode::SetExpressionPosition => &[Immediate],
            Opcode::SetExpressionPositionFromEnd => &[Immediate],
            Opcode::ResumeGenerator => &[Register, Register, RegisterCount],
            Opcode::GetGeneratorState => &[Register],
            Opcode::SuspendGenerator => &[Register, Register, RegisterCount, Immediate],
            Opcode::SetGeneratorState => &[Register],
            Opcode::SwitchOnGeneratorState => &[Register],

            // Class
            Opcode::CreateClass => &[ConstantPoolIdx, Register, FeedbackSlot],
            Opcode::TestPrivateBrand => &[Register, Register],
            Opcode::DefinePrivateBrand => &[Register],

            // Module
            Opcode::LdaModuleVariable => &[ConstantPoolIdx, Immediate],
            Opcode::StaModuleVariable => &[ConstantPoolIdx, Immediate],
            Opcode::LdaImportMeta => &[],
            Opcode::LdaNewTarget => &[],
            Opcode::GetModuleNamespace => &[ConstantPoolIdx],

            // Prefixes / trap — no operands of their own
            Opcode::Wide | Opcode::ExtraWide | Opcode::Illegal => &[],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoding helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Determine the minimum [`OperandWidth`] needed to encode all `operands`.
fn required_width(operands: &[Operand]) -> OperandWidth {
    operands
        .iter()
        .fold(OperandWidth::Narrow, |w, op| w.max(op.required_width()))
}

/// Append one operand to `out` using the given `width`.
fn write_operand(out: &mut Vec<u8>, operand: Operand, width: OperandWidth) {
    match width {
        OperandWidth::Narrow => {
            let byte = match operand {
                Operand::Register(v)
                | Operand::RegisterCount(v)
                | Operand::ConstantPoolIdx(v)
                | Operand::FeedbackSlot(v)
                | Operand::RuntimeId(v) => v as u8,
                Operand::Immediate(v) | Operand::JumpOffset(v) => v as i8 as u8,
                Operand::Flag(v) => v,
            };
            out.push(byte);
        }
        OperandWidth::Wide => {
            let halfword: u16 = match operand {
                Operand::Register(v)
                | Operand::RegisterCount(v)
                | Operand::ConstantPoolIdx(v)
                | Operand::FeedbackSlot(v)
                | Operand::RuntimeId(v) => v as u16,
                Operand::Immediate(v) | Operand::JumpOffset(v) => v as i16 as u16,
                Operand::Flag(v) => v as u16,
            };
            out.extend_from_slice(&halfword.to_le_bytes());
        }
        OperandWidth::ExtraWide => {
            let word: u32 = match operand {
                Operand::Register(v)
                | Operand::RegisterCount(v)
                | Operand::ConstantPoolIdx(v)
                | Operand::FeedbackSlot(v)
                | Operand::RuntimeId(v) => v,
                Operand::Immediate(v) | Operand::JumpOffset(v) => v as u32,
                Operand::Flag(v) => v as u32,
            };
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

/// Read one operand from `bytes` at `*pos`, advancing `*pos`.
fn read_operand(
    bytes: &[u8],
    pos: &mut usize,
    op_type: OperandType,
    width: OperandWidth,
) -> StatorResult<Operand> {
    let w = width as usize;
    let end = *pos + w;
    if end > bytes.len() {
        return Err(StatorError::Internal(format!(
            "truncated bytecode: need {w} byte(s) at offset {pos}",
            pos = *pos
        )));
    }
    let raw_u32 = match width {
        OperandWidth::Narrow => bytes[*pos] as u32,
        OperandWidth::Wide => {
            let arr: [u8; 2] = bytes[*pos..*pos + 2].try_into().unwrap();
            u16::from_le_bytes(arr) as u32
        }
        OperandWidth::ExtraWide => {
            let arr: [u8; 4] = bytes[*pos..*pos + 4].try_into().unwrap();
            u32::from_le_bytes(arr)
        }
    };
    *pos = end;

    let operand = match op_type {
        OperandType::Register => Operand::Register(raw_u32),
        OperandType::RegisterCount => Operand::RegisterCount(raw_u32),
        OperandType::ConstantPoolIdx => Operand::ConstantPoolIdx(raw_u32),
        OperandType::FeedbackSlot => Operand::FeedbackSlot(raw_u32),
        OperandType::RuntimeId => Operand::RuntimeId(raw_u32),
        OperandType::Immediate => {
            let signed = match width {
                OperandWidth::Narrow => raw_u32 as i8 as i32,
                OperandWidth::Wide => raw_u32 as i16 as i32,
                OperandWidth::ExtraWide => raw_u32 as i32,
            };
            Operand::Immediate(signed)
        }
        OperandType::JumpOffset => {
            let signed = match width {
                OperandWidth::Narrow => raw_u32 as i8 as i32,
                OperandWidth::Wide => raw_u32 as i16 as i32,
                OperandWidth::ExtraWide => raw_u32 as i32,
            };
            Operand::JumpOffset(signed)
        }
        OperandType::Flag => Operand::Flag(raw_u32 as u8),
    };
    Ok(operand)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public encode / decode
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a sequence of [`Instruction`]s to a compact byte stream.
///
/// The encoder automatically inserts [`Opcode::Wide`] or
/// [`Opcode::ExtraWide`] prefix bytes when operand values exceed the
/// one-byte narrow range.
pub fn encode(instructions: &[Instruction]) -> Vec<u8> {
    let mut out = Vec::new();
    for instr in instructions {
        let width = required_width(&instr.operands);
        match width {
            OperandWidth::Wide => out.push(Opcode::Wide as u8),
            OperandWidth::ExtraWide => out.push(Opcode::ExtraWide as u8),
            OperandWidth::Narrow => {}
        }
        out.push(instr.opcode as u8);
        for &operand in &instr.operands {
            write_operand(&mut out, operand, width);
        }
    }
    out
}

/// Decode a byte stream produced by [`encode`] back into [`Instruction`]s,
/// also returning the byte offset of each instruction and a trailing end-of-stream
/// offset.
///
/// The returned `byte_offsets` vector has length `instructions.len() + 1`:
/// - `byte_offsets[i]` is the byte offset (from the start of `bytes`) of
///   instruction `i` (including any preceding [`Opcode::Wide`] /
///   [`Opcode::ExtraWide`] prefix).
/// - `byte_offsets[n]` (where `n = instructions.len()`) equals `bytes.len()`.
///
/// These offsets are the same as the table produced during jump resolution in
/// the compiler, so `byte_offsets[i + 1]` gives the end byte of instruction `i`.
/// Jump handlers use this to convert a [`Operand::JumpOffset`] delta back to
/// an instruction index.
///
/// Returns an error if the stream is truncated, contains an unknown opcode,
/// or has any other structural inconsistency.
pub fn decode_with_byte_offsets(bytes: &[u8]) -> StatorResult<(Vec<Instruction>, Vec<usize>)> {
    let mut instructions = Vec::new();
    let mut byte_offsets = Vec::new();
    let mut pos = 0usize;

    while pos < bytes.len() {
        let instr_start = pos;

        // Check for a width prefix.
        let width = match Opcode::try_from_u8(bytes[pos])? {
            Opcode::Wide => {
                pos += 1;
                OperandWidth::Wide
            }
            Opcode::ExtraWide => {
                pos += 1;
                OperandWidth::ExtraWide
            }
            _ => OperandWidth::Narrow,
        };

        if pos >= bytes.len() {
            return Err(StatorError::Internal(
                "bytecode ends with a width prefix and no following opcode".into(),
            ));
        }

        let opcode = Opcode::try_from_u8(bytes[pos])?;
        pos += 1;

        // Wide/ExtraWide are only valid as prefixes; encountering them as a
        // real opcode (i.e. back-to-back) is illegal.
        if matches!(opcode, Opcode::Wide | Opcode::ExtraWide) {
            return Err(StatorError::Internal(
                "unexpected width-prefix opcode in operand position".into(),
            ));
        }

        let op_types = opcode.operand_types();
        let mut operands = Vec::with_capacity(op_types.len());
        for &op_type in op_types {
            operands.push(read_operand(bytes, &mut pos, op_type, width)?);
        }
        byte_offsets.push(instr_start);
        instructions.push(Instruction::new_unchecked(opcode, operands));
    }

    byte_offsets.push(pos); // sentinel: total byte length
    Ok((instructions, byte_offsets))
}

/// Decode a byte stream produced by [`encode`] back into [`Instruction`]s.
///
/// Returns an error if the stream is truncated, contains an unknown opcode,
/// or has any other structural inconsistency.
pub fn decode(bytes: &[u8]) -> StatorResult<Vec<Instruction>> {
    Ok(decode_with_byte_offsets(bytes)?.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────

    /// Assert that encoding then decoding `instructions` yields the original.
    fn round_trip(instructions: Vec<Instruction>) {
        let bytes = encode(&instructions);
        let decoded = decode(&bytes).expect("decode should succeed");
        assert_eq!(decoded, instructions, "round-trip mismatch");
    }

    // ── no-operand opcodes ────────────────────────────────────────────────

    #[test]
    fn test_round_trip_lda_zero() {
        round_trip(vec![Instruction::new_unchecked(Opcode::LdaZero, vec![])]);
    }

    #[test]
    fn test_round_trip_return() {
        round_trip(vec![Instruction::new_unchecked(Opcode::Return, vec![])]);
    }

    #[test]
    fn test_round_trip_lda_null_undefined_true_false() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::LdaNull, vec![]),
            Instruction::new_unchecked(Opcode::LdaUndefined, vec![]),
            Instruction::new_unchecked(Opcode::LdaTrue, vec![]),
            Instruction::new_unchecked(Opcode::LdaFalse, vec![]),
        ]);
    }

    // ── narrow immediates / registers ─────────────────────────────────────

    #[test]
    fn test_round_trip_lda_smi_narrow() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaSmi,
            vec![Operand::Immediate(42)],
        )]);
    }

    #[test]
    fn test_round_trip_lda_smi_negative_narrow() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaSmi,
            vec![Operand::Immediate(-1)],
        )]);
    }

    #[test]
    fn test_round_trip_ldar_star() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
        ]);
    }

    #[test]
    fn test_round_trip_mov() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Mov,
            vec![Operand::Register(1), Operand::Register(2)],
        )]);
    }

    // ── arithmetic ────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_add() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Add,
            vec![Operand::Register(0), Operand::FeedbackSlot(1)],
        )]);
    }

    #[test]
    fn test_round_trip_add_smi() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::AddSmi,
            vec![Operand::Immediate(10), Operand::FeedbackSlot(0)],
        )]);
    }

    #[test]
    fn test_round_trip_inc_dec() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(2)]),
            Instruction::new_unchecked(Opcode::Dec, vec![Operand::FeedbackSlot(3)]),
        ]);
    }

    #[test]
    fn test_round_trip_bitwise_ops() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::BitwiseOr,
                vec![Operand::Register(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(
                Opcode::BitwiseAnd,
                vec![Operand::Register(2), Operand::FeedbackSlot(1)],
            ),
            Instruction::new_unchecked(
                Opcode::BitwiseXor,
                vec![Operand::Register(3), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(
                Opcode::ShiftLeft,
                vec![Operand::Register(4), Operand::FeedbackSlot(3)],
            ),
            Instruction::new_unchecked(
                Opcode::ShiftRight,
                vec![Operand::Register(5), Operand::FeedbackSlot(4)],
            ),
            Instruction::new_unchecked(
                Opcode::ShiftRightLogical,
                vec![Operand::Register(6), Operand::FeedbackSlot(5)],
            ),
        ]);
    }

    // ── comparison ────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_comparisons() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::TestEqual,
                vec![Operand::Register(0), Operand::FeedbackSlot(1)],
            ),
            Instruction::new_unchecked(
                Opcode::TestEqualStrict,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(2), Operand::FeedbackSlot(3)],
            ),
            Instruction::new_unchecked(
                Opcode::TestGreaterThan,
                vec![Operand::Register(3), Operand::FeedbackSlot(4)],
            ),
            Instruction::new_unchecked(Opcode::TestNull, vec![]),
            Instruction::new_unchecked(Opcode::TestUndefined, vec![]),
        ]);
    }

    // ── property access ───────────────────────────────────────────────────

    #[test]
    fn test_round_trip_named_property() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(5),
                    Operand::FeedbackSlot(2),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::StaNamedProperty,
                vec![
                    Operand::Register(1),
                    Operand::ConstantPoolIdx(6),
                    Operand::FeedbackSlot(3),
                ],
            ),
        ]);
    }

    #[test]
    fn test_round_trip_keyed_property() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::LdaKeyedProperty,
                vec![Operand::Register(0), Operand::FeedbackSlot(1)],
            ),
            Instruction::new_unchecked(
                Opcode::StaKeyedProperty,
                vec![
                    Operand::Register(1),
                    Operand::Register(2),
                    Operand::FeedbackSlot(3),
                ],
            ),
        ]);
    }

    // ── calls ─────────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_call_property() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::CallProperty,
            vec![
                Operand::Register(0),
                Operand::Register(1),
                Operand::RegisterCount(2),
                Operand::FeedbackSlot(4),
            ],
        )]);
    }

    #[test]
    fn test_round_trip_call_runtime() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::CallRuntime,
            vec![
                Operand::RuntimeId(42),
                Operand::Register(0),
                Operand::RegisterCount(3),
            ],
        )]);
    }

    #[test]
    fn test_round_trip_construct() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Construct,
            vec![
                Operand::Register(0),
                Operand::Register(1),
                Operand::RegisterCount(2),
                Operand::FeedbackSlot(5),
            ],
        )]);
    }

    // ── control flow ──────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_jump() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Jump,
            vec![Operand::JumpOffset(8)],
        )]);
    }

    #[test]
    fn test_round_trip_jump_if_true_false() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::JumpIfTrue, vec![Operand::JumpOffset(4)]),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(-4)]),
        ]);
    }

    #[test]
    fn test_round_trip_jump_loop() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::JumpLoop,
            vec![
                Operand::JumpOffset(-20),
                Operand::Immediate(1),
                Operand::FeedbackSlot(0),
            ],
        )]);
    }

    // ── context / lookup ──────────────────────────────────────────────────

    #[test]
    fn test_round_trip_context_slot() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::LdaContextSlot,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(3),
                    Operand::Immediate(1),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::StaContextSlot,
                vec![
                    Operand::Register(0),
                    Operand::ConstantPoolIdx(4),
                    Operand::Immediate(0),
                ],
            ),
        ]);
    }

    #[test]
    fn test_round_trip_push_pop_context() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::PushContext, vec![Operand::Register(2)]),
            Instruction::new_unchecked(Opcode::PopContext, vec![Operand::Register(2)]),
        ]);
    }

    // ── exception ─────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_throw_rethrow() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::Throw, vec![]),
            Instruction::new_unchecked(Opcode::ReThrow, vec![]),
        ]);
    }

    #[test]
    fn test_round_trip_throw_reference_error_if_hole() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::ThrowReferenceErrorIfHole,
            vec![Operand::ConstantPoolIdx(7)],
        )]);
    }

    // ── generators ────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_suspend_resume_generator() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::SuspendGenerator,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(4),
                    Operand::Immediate(0),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::ResumeGenerator,
                vec![
                    Operand::Register(0),
                    Operand::Register(1),
                    Operand::RegisterCount(4),
                ],
            ),
        ]);
    }

    // ── for-in ────────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_for_in() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::ForInEnumerate, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::ForInPrepare,
                vec![Operand::Register(2), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(
                Opcode::ForInNext,
                vec![
                    Operand::Register(1),
                    Operand::Register(2),
                    Operand::Register(3),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::ForInStep, vec![Operand::Register(4)]),
        ]);
    }

    // ── closures / literals ───────────────────────────────────────────────

    #[test]
    fn test_round_trip_create_closure() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::CreateClosure,
            vec![
                Operand::ConstantPoolIdx(0),
                Operand::FeedbackSlot(1),
                Operand::Flag(0),
            ],
        )]);
    }

    #[test]
    fn test_round_trip_create_array_object_literals() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::CreateArrayLiteral,
                vec![
                    Operand::ConstantPoolIdx(2),
                    Operand::FeedbackSlot(0),
                    Operand::Flag(1),
                ],
            ),
            Instruction::new_unchecked(
                Opcode::CreateObjectLiteral,
                vec![
                    Operand::ConstantPoolIdx(3),
                    Operand::FeedbackSlot(1),
                    Operand::Flag(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::CreateEmptyObjectLiteral, vec![]),
            Instruction::new_unchecked(
                Opcode::CreateEmptyArrayLiteral,
                vec![Operand::FeedbackSlot(2)],
            ),
        ]);
    }

    // ── module opcodes ────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_lda_sta_module_variable() {
        round_trip(vec![
            Instruction::new_unchecked(
                Opcode::LdaModuleVariable,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(1)],
            ),
            Instruction::new_unchecked(
                Opcode::StaModuleVariable,
                vec![Operand::ConstantPoolIdx(0), Operand::Immediate(2)],
            ),
        ]);
    }

    #[test]
    fn test_round_trip_lda_import_meta() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaImportMeta,
            vec![],
        )]);
    }

    #[test]
    fn test_round_trip_get_module_namespace() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::GetModuleNamespace,
            vec![Operand::ConstantPoolIdx(5)],
        )]);
    }

    // ── wide encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_wide_register() {
        // Register index > 255 → should be encoded with Wide prefix.
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Ldar,
            vec![Operand::Register(256)],
        )]);
    }

    #[test]
    fn test_round_trip_wide_immediate() {
        // Immediate outside i8 range → Wide prefix.
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaSmi,
            vec![Operand::Immediate(200)],
        )]);
    }

    #[test]
    fn test_round_trip_wide_jump_offset() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Jump,
            vec![Operand::JumpOffset(1000)],
        )]);
    }

    #[test]
    fn test_round_trip_wide_constant_pool_idx() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaConstant,
            vec![Operand::ConstantPoolIdx(300)],
        )]);
    }

    // ── extra-wide encoding ───────────────────────────────────────────────

    #[test]
    fn test_round_trip_extra_wide_register() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::Star,
            vec![Operand::Register(70_000)],
        )]);
    }

    #[test]
    fn test_round_trip_extra_wide_immediate() {
        round_trip(vec![Instruction::new_unchecked(
            Opcode::LdaSmi,
            vec![Operand::Immediate(100_000)],
        )]);
    }

    // ── mixed-width sequence ──────────────────────────────────────────────

    #[test]
    fn test_round_trip_mixed_widths() {
        round_trip(vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(200)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(100_000)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(300)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ]);
    }

    // ── opcode byte encoding ──────────────────────────────────────────────

    #[test]
    fn test_opcode_try_from_u8_roundtrip() {
        // Every discriminant 0..=MAX_OPCODE must decode successfully.
        for byte in 0..=MAX_OPCODE {
            let op = Opcode::try_from_u8(byte).expect("should decode");
            assert_eq!(op as u8, byte);
        }
    }

    #[test]
    fn test_opcode_try_from_u8_out_of_range() {
        assert!(Opcode::try_from_u8(MAX_OPCODE + 1).is_err());
        assert!(Opcode::try_from_u8(255).is_err());
    }

    // ── Instruction::new validation ───────────────────────────────────────

    #[test]
    fn test_instruction_new_correct_operand_count() {
        let instr = Instruction::new(Opcode::LdaSmi, vec![Operand::Immediate(7)]);
        assert!(instr.is_ok());
    }

    #[test]
    fn test_instruction_new_wrong_operand_count() {
        let instr = Instruction::new(Opcode::LdaSmi, vec![]);
        assert!(instr.is_err());
    }

    // ── decode errors ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_empty() {
        assert_eq!(decode(&[]).unwrap(), vec![]);
    }

    #[test]
    fn test_decode_truncated_operand() {
        // LdaSmi opcode with no following byte → truncation error.
        let bytes = vec![Opcode::LdaSmi as u8];
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn test_decode_dangling_wide_prefix() {
        // Wide prefix with nothing after it → error.
        let bytes = vec![Opcode::Wide as u8];
        assert!(decode(&bytes).is_err());
    }
}

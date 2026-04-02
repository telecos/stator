//! Maglev IR node types.
//!
//! This module defines the typed intermediate representation (IR) used by the
//! Maglev optimising compiler tier.  Using typed enums eliminates runtime type
//! checks that would otherwise be required with an untyped node representation.
//!
//! # Structure
//!
//! - [`ValueNode`] — nodes that produce a value (inputs to other nodes).
//! - [`ControlNode`] — terminator instructions at the end of each block.
//! - [`BasicBlock`] — a straight-line sequence of [`ValueNode`]s followed by
//!   exactly one [`ControlNode`].
//! - [`MaglevGraph`] — the complete control-flow graph, owning all blocks.
//!
//! # Example
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//!
//! let mut graph = MaglevGraph::new(1);
//!
//! // entry block: return parameter 0
//! let mut entry = BasicBlock::new(0);
//! let param = entry.push_value(ValueNode::Parameter { index: 0 });
//! entry.set_control(ControlNode::Return { value: param });
//! graph.add_block(entry);
//! assert_eq!(graph.blocks().len(), 1);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Node IDs
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a [`ValueNode`] within a [`MaglevGraph`].
///
/// Graph-global IDs are assigned by [`MaglevGraph::add_value_node`].
/// Block-local IDs assigned by [`BasicBlock::push_value`] start at `0` per
/// block and are only unique within that block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

// ─────────────────────────────────────────────────────────────────────────────
// ValueNode
// ─────────────────────────────────────────────────────────────────────────────

/// A Maglev IR node that produces a typed value.
///
/// Each variant captures both the *kind* of computation and all data required
/// to perform it, eliminating the need for dynamic casts at compile time.
///
/// # Groupings
///
/// ## Constants
/// [`SmiConstant`](ValueNode::SmiConstant),
/// [`Float64Constant`](ValueNode::Float64Constant),
/// [`Int32Constant`](ValueNode::Int32Constant),
/// [`Uint32Constant`](ValueNode::Uint32Constant),
/// [`BigIntConstant`](ValueNode::BigIntConstant),
/// [`TrueConstant`](ValueNode::TrueConstant),
/// [`FalseConstant`](ValueNode::FalseConstant),
/// [`NullConstant`](ValueNode::NullConstant),
/// [`UndefinedConstant`](ValueNode::UndefinedConstant),
/// [`RootConstant`](ValueNode::RootConstant),
/// [`ExternalConstant`](ValueNode::ExternalConstant),
/// [`StringConstant`](ValueNode::StringConstant),
/// [`ConstantPoolEntry`](ValueNode::ConstantPoolEntry)
///
/// ## Parameters and registers
/// [`Parameter`](ValueNode::Parameter),
/// [`RegisterInput`](ValueNode::RegisterInput),
/// [`ArgumentsLength`](ValueNode::ArgumentsLength),
/// [`RestLength`](ValueNode::RestLength),
/// [`GetArgument`](ValueNode::GetArgument)
///
/// ## Arithmetic (Smi / Int32 / Uint32 / Float64 / tagged)
/// [`CheckedSmiAdd`](ValueNode::CheckedSmiAdd),
/// [`CheckedSmiSubtract`](ValueNode::CheckedSmiSubtract),
/// [`CheckedSmiMultiply`](ValueNode::CheckedSmiMultiply),
/// [`CheckedSmiDivide`](ValueNode::CheckedSmiDivide),
/// [`CheckedSmiModulus`](ValueNode::CheckedSmiModulus),
/// [`CheckedSmiIncrement`](ValueNode::CheckedSmiIncrement),
/// [`CheckedSmiDecrement`](ValueNode::CheckedSmiDecrement),
/// [`Int32Add`](ValueNode::Int32Add),
/// [`Int32Subtract`](ValueNode::Int32Subtract),
/// [`Int32Multiply`](ValueNode::Int32Multiply),
/// [`Int32Divide`](ValueNode::Int32Divide),
/// [`Int32Modulus`](ValueNode::Int32Modulus),
/// [`Int32Negate`](ValueNode::Int32Negate),
/// [`Int32Increment`](ValueNode::Int32Increment),
/// [`Int32Decrement`](ValueNode::Int32Decrement),
/// [`Int32BitwiseAnd`](ValueNode::Int32BitwiseAnd),
/// [`Int32BitwiseOr`](ValueNode::Int32BitwiseOr),
/// [`Int32BitwiseXor`](ValueNode::Int32BitwiseXor),
/// [`Int32ShiftLeft`](ValueNode::Int32ShiftLeft),
/// [`Int32ShiftRight`](ValueNode::Int32ShiftRight),
/// [`Int32ShiftRightLogical`](ValueNode::Int32ShiftRightLogical),
/// [`Uint32Add`](ValueNode::Uint32Add),
/// [`Uint32Subtract`](ValueNode::Uint32Subtract),
/// [`Uint32Multiply`](ValueNode::Uint32Multiply),
/// [`Uint32Divide`](ValueNode::Uint32Divide),
/// [`Uint32Modulus`](ValueNode::Uint32Modulus),
/// [`Float64Add`](ValueNode::Float64Add),
/// [`Float64Subtract`](ValueNode::Float64Subtract),
/// [`Float64Multiply`](ValueNode::Float64Multiply),
/// [`Float64Divide`](ValueNode::Float64Divide),
/// [`Float64Modulus`](ValueNode::Float64Modulus),
/// [`Float64Negate`](ValueNode::Float64Negate),
/// [`Float64Exponentiate`](ValueNode::Float64Exponentiate),
/// [`Float64Ieee754Unary`](ValueNode::Float64Ieee754Unary),
/// [`GenericAdd`](ValueNode::GenericAdd),
/// [`GenericSubtract`](ValueNode::GenericSubtract),
/// [`GenericMultiply`](ValueNode::GenericMultiply),
/// [`GenericDivide`](ValueNode::GenericDivide),
/// [`GenericModulus`](ValueNode::GenericModulus),
/// [`GenericExponentiate`](ValueNode::GenericExponentiate),
/// [`GenericBitwiseAnd`](ValueNode::GenericBitwiseAnd),
/// [`GenericBitwiseOr`](ValueNode::GenericBitwiseOr),
/// [`GenericBitwiseXor`](ValueNode::GenericBitwiseXor),
/// [`GenericShiftLeft`](ValueNode::GenericShiftLeft),
/// [`GenericShiftRight`](ValueNode::GenericShiftRight),
/// [`GenericShiftRightLogical`](ValueNode::GenericShiftRightLogical),
/// [`GenericBitwiseNot`](ValueNode::GenericBitwiseNot),
/// [`GenericNegate`](ValueNode::GenericNegate),
/// [`GenericIncrement`](ValueNode::GenericIncrement),
/// [`GenericDecrement`](ValueNode::GenericDecrement)
///
/// ## Comparisons
/// [`Int32Equal`](ValueNode::Int32Equal),
/// [`Int32StrictEqual`](ValueNode::Int32StrictEqual),
/// [`Int32LessThan`](ValueNode::Int32LessThan),
/// [`Int32LessThanOrEqual`](ValueNode::Int32LessThanOrEqual),
/// [`Int32GreaterThan`](ValueNode::Int32GreaterThan),
/// [`Int32GreaterThanOrEqual`](ValueNode::Int32GreaterThanOrEqual),
/// [`Float64Equal`](ValueNode::Float64Equal),
/// [`Float64LessThan`](ValueNode::Float64LessThan),
/// [`Float64LessThanOrEqual`](ValueNode::Float64LessThanOrEqual),
/// [`Float64GreaterThan`](ValueNode::Float64GreaterThan),
/// [`Float64GreaterThanOrEqual`](ValueNode::Float64GreaterThanOrEqual),
/// [`TaggedEqual`](ValueNode::TaggedEqual),
/// [`TaggedNotEqual`](ValueNode::TaggedNotEqual),
/// [`TestInstanceOf`](ValueNode::TestInstanceOf),
/// [`TestIn`](ValueNode::TestIn),
/// [`TestUndetectable`](ValueNode::TestUndetectable),
/// [`TestTypeOf`](ValueNode::TestTypeOf)
///
/// ## Type conversions
/// [`ChangeInt32ToFloat64`](ValueNode::ChangeInt32ToFloat64),
/// [`ChangeUint32ToFloat64`](ValueNode::ChangeUint32ToFloat64),
/// [`ChangeFloat64ToInt32`](ValueNode::ChangeFloat64ToInt32),
/// [`CheckedFloat64ToInt32`](ValueNode::CheckedFloat64ToInt32),
/// [`ChangeInt32ToTagged`](ValueNode::ChangeInt32ToTagged),
/// [`ChangeUint32ToTagged`](ValueNode::ChangeUint32ToTagged),
/// [`ChangeFloat64ToTagged`](ValueNode::ChangeFloat64ToTagged),
/// [`ChangeTaggedToInt32`](ValueNode::ChangeTaggedToInt32),
/// [`ChangeTaggedToUint32`](ValueNode::ChangeTaggedToUint32),
/// [`ChangeTaggedToFloat64`](ValueNode::ChangeTaggedToFloat64),
/// [`CheckedTaggedToInt32`](ValueNode::CheckedTaggedToInt32),
/// [`CheckedTaggedToFloat64`](ValueNode::CheckedTaggedToFloat64),
/// [`ToBoolean`](ValueNode::ToBoolean),
/// [`ToString`](ValueNode::ToString),
/// [`ToObject`](ValueNode::ToObject),
/// [`ToName`](ValueNode::ToName),
/// [`ToNumber`](ValueNode::ToNumber),
/// [`ToNumberOrNumeric`](ValueNode::ToNumberOrNumeric)
///
/// ## Checks / guards
/// [`CheckSmi`](ValueNode::CheckSmi),
/// [`CheckNumber`](ValueNode::CheckNumber),
/// [`CheckHeapObject`](ValueNode::CheckHeapObject),
/// [`CheckSymbol`](ValueNode::CheckSymbol),
/// [`CheckString`](ValueNode::CheckString),
/// [`CheckStringOrStringWrapper`](ValueNode::CheckStringOrStringWrapper),
/// [`CheckSeqOneByteString`](ValueNode::CheckSeqOneByteString),
/// [`CheckMaps`](ValueNode::CheckMaps),
/// [`CheckMapsWithMigration`](ValueNode::CheckMapsWithMigration),
/// [`CheckValue`](ValueNode::CheckValue),
/// [`CheckDynamicValue`](ValueNode::CheckDynamicValue),
/// [`CheckInt32IsSmi`](ValueNode::CheckInt32IsSmi),
/// [`CheckUint32IsSmi`](ValueNode::CheckUint32IsSmi),
/// [`CheckHoleyFloat64IsSmi`](ValueNode::CheckHoleyFloat64IsSmi),
/// [`CheckInt32Condition`](ValueNode::CheckInt32Condition),
/// [`CheckCacheIndicesNotCleared`](ValueNode::CheckCacheIndicesNotCleared),
/// [`CheckFloat64IsNan`](ValueNode::CheckFloat64IsNan)
///
/// ## Property / field access
/// [`LoadField`](ValueNode::LoadField),
/// [`StoreField`](ValueNode::StoreField),
/// [`LoadTaggedField`](ValueNode::LoadTaggedField),
/// [`LoadDoubleField`](ValueNode::LoadDoubleField),
/// [`LoadFixedArrayElement`](ValueNode::LoadFixedArrayElement),
/// [`LoadFixedDoubleArrayElement`](ValueNode::LoadFixedDoubleArrayElement),
/// [`LoadHoleyFixedDoubleArrayElement`](ValueNode::LoadHoleyFixedDoubleArrayElement),
/// [`StoreFixedArrayElement`](ValueNode::StoreFixedArrayElement),
/// [`StoreFixedDoubleArrayElement`](ValueNode::StoreFixedDoubleArrayElement),
/// [`LoadNamedGeneric`](ValueNode::LoadNamedGeneric),
/// [`StoreNamedGeneric`](ValueNode::StoreNamedGeneric),
/// [`LoadKeyedGeneric`](ValueNode::LoadKeyedGeneric),
/// [`StoreKeyedGeneric`](ValueNode::StoreKeyedGeneric),
/// [`LoadGlobal`](ValueNode::LoadGlobal),
/// [`StoreGlobal`](ValueNode::StoreGlobal),
/// [`LoadContextSlot`](ValueNode::LoadContextSlot),
/// [`StoreContextSlot`](ValueNode::StoreContextSlot),
/// [`LoadCurrentContextSlot`](ValueNode::LoadCurrentContextSlot),
/// [`StoreCurrentContextSlot`](ValueNode::StoreCurrentContextSlot)
///
/// ## Calls and constructs
/// [`Call`](ValueNode::Call),
/// [`CallKnownFunction`](ValueNode::CallKnownFunction),
/// [`CallBuiltin`](ValueNode::CallBuiltin),
/// [`CallRuntime`](ValueNode::CallRuntime),
/// [`CallWithSpread`](ValueNode::CallWithSpread),
/// [`Construct`](ValueNode::Construct),
/// [`ConstructWithSpread`](ValueNode::ConstructWithSpread)
///
/// ## Object / array creation
/// [`CreateObjectLiteral`](ValueNode::CreateObjectLiteral),
/// [`CreateArrayLiteral`](ValueNode::CreateArrayLiteral),
/// [`CreateShallowObjectLiteral`](ValueNode::CreateShallowObjectLiteral),
/// [`CreateShallowArrayLiteral`](ValueNode::CreateShallowArrayLiteral),
/// [`CreateFunctionContext`](ValueNode::CreateFunctionContext),
/// [`PushContext`](ValueNode::PushContext),
/// [`PopContext`](ValueNode::PopContext),
/// [`CreateBlockContext`](ValueNode::CreateBlockContext),
/// [`CreateCatchContext`](ValueNode::CreateCatchContext),
/// [`CreateWithContext`](ValueNode::CreateWithContext),
/// [`CreateClosure`](ValueNode::CreateClosure),
/// [`FastCreateClosure`](ValueNode::FastCreateClosure),
/// [`CreateEmptyObjectLiteral`](ValueNode::CreateEmptyObjectLiteral),
/// [`CreateRegExpLiteral`](ValueNode::CreateRegExpLiteral)
///
/// ## Control-value producers
/// [`Phi`](ValueNode::Phi),
/// [`ArgumentsElements`](ValueNode::ArgumentsElements),
/// [`RestElements`](ValueNode::RestElements),
/// [`VirtualObject`](ValueNode::VirtualObject)
///
/// ## Miscellaneous
/// [`GetTemplateObject`](ValueNode::GetTemplateObject),
/// [`HasInPrototypeChain`](ValueNode::HasInPrototypeChain),
/// [`DeleteProperty`](ValueNode::DeleteProperty),
/// [`ForInPrepare`](ValueNode::ForInPrepare),
/// [`ForInNext`](ValueNode::ForInNext),
/// [`LoadEnumCacheLength`](ValueNode::LoadEnumCacheLength),
/// [`StringAt`](ValueNode::StringAt),
/// [`StringLength`](ValueNode::StringLength),
/// [`StringConcat`](ValueNode::StringConcat),
/// [`StringEqual`](ValueNode::StringEqual),
/// [`NumberToString`](ValueNode::NumberToString),
/// [`TypeOf`](ValueNode::TypeOf),
/// [`Debugger`](ValueNode::Debugger),
/// [`Abort`](ValueNode::Abort)
#[derive(Debug, Clone, PartialEq)]
pub enum ValueNode {
    // ── Constants ────────────────────────────────────────────────────────────
    /// A small integer (Smi) constant.
    SmiConstant {
        /// The constant value.
        value: i32,
    },

    /// A 64-bit floating-point constant.
    Float64Constant {
        /// The constant value.
        value: f64,
    },

    /// A 32-bit signed integer constant (unboxed).
    Int32Constant {
        /// The constant value.
        value: i32,
    },

    /// A 32-bit unsigned integer constant (unboxed).
    Uint32Constant {
        /// The constant value.
        value: u32,
    },

    /// A BigInt constant represented as a decimal string.
    BigIntConstant {
        /// Decimal string representation of the BigInt value.
        value: String,
    },

    /// The boolean `true` constant.
    TrueConstant,

    /// The boolean `false` constant.
    FalseConstant,

    /// The `null` constant.
    NullConstant,

    /// The `undefined` constant.
    UndefinedConstant,

    /// A reference to a well-known engine root object (e.g. the empty string).
    RootConstant {
        /// Index into the engine root table.
        index: u32,
    },

    /// A pointer-sized external constant (e.g. a C++ function address).
    ExternalConstant {
        /// Raw pointer value stored as a `u64`.
        address: u64,
    },

    /// A string constant (interned).
    StringConstant {
        /// The string value.
        value: String,
    },

    /// A reference to a constant-pool entry by index.
    ConstantPoolEntry {
        /// Zero-based index into the enclosing function's constant pool.
        index: u32,
    },

    // ── Parameters and registers ─────────────────────────────────────────────
    /// A function parameter (zero-based).
    Parameter {
        /// Zero-based parameter index.
        index: u32,
    },

    /// A physical register input at function entry.
    RegisterInput {
        /// Physical register number.
        reg: u32,
    },

    /// The number of actual arguments passed at the call site.
    ArgumentsLength,

    /// The number of rest-parameter elements.
    RestLength,

    /// Load a single argument by index from the arguments object.
    GetArgument {
        /// The argument to retrieve.
        index: NodeId,
    },

    // ── Smi arithmetic ───────────────────────────────────────────────────────
    /// Tagged Smi addition with overflow check (deopt on overflow).
    CheckedSmiAdd {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Tagged Smi subtraction with overflow check (deopt on overflow).
    CheckedSmiSubtract {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Tagged Smi multiplication with overflow check (deopt on overflow).
    CheckedSmiMultiply {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Tagged Smi division with overflow/zero check (deopt if not exact).
    CheckedSmiDivide {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Tagged Smi modulus with overflow/zero check (deopt on non-Smi result).
    CheckedSmiModulus {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Increment a Smi value by one, deopt on overflow.
    CheckedSmiIncrement {
        /// The value to increment.
        value: NodeId,
    },

    /// Decrement a Smi value by one, deopt on overflow.
    CheckedSmiDecrement {
        /// The value to decrement.
        value: NodeId,
    },

    // ── Int32 arithmetic ─────────────────────────────────────────────────────
    /// Unboxed 32-bit integer addition (wrapping).
    Int32Add {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer subtraction (wrapping).
    Int32Subtract {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer multiplication (wrapping).
    Int32Multiply {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer division (truncating).
    Int32Divide {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Unboxed 32-bit integer modulus.
    Int32Modulus {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Unboxed 32-bit integer negation (wrapping).
    Int32Negate {
        /// The value to negate.
        value: NodeId,
    },

    /// Unboxed 32-bit integer increment by one (wrapping).
    Int32Increment {
        /// The value to increment.
        value: NodeId,
    },

    /// Unboxed 32-bit integer decrement by one (wrapping).
    Int32Decrement {
        /// The value to decrement.
        value: NodeId,
    },

    /// Unboxed 32-bit bitwise AND.
    Int32BitwiseAnd {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit bitwise OR.
    Int32BitwiseOr {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit bitwise XOR.
    Int32BitwiseXor {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit left shift.
    Int32ShiftLeft {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
    },

    /// Unboxed 32-bit arithmetic right shift.
    Int32ShiftRight {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
    },

    /// Unboxed 32-bit logical right shift.
    Int32ShiftRightLogical {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
    },

    // ── Uint32 arithmetic ────────────────────────────────────────────────────
    /// Unboxed 32-bit unsigned integer addition (wrapping).
    Uint32Add {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit unsigned integer subtraction (wrapping).
    Uint32Subtract {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit unsigned integer multiplication (wrapping).
    Uint32Multiply {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit unsigned integer division (truncating).
    Uint32Divide {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Unboxed 32-bit unsigned integer modulus.
    Uint32Modulus {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    // ── Float64 arithmetic ───────────────────────────────────────────────────
    /// Unboxed 64-bit float addition.
    Float64Add {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float subtraction.
    Float64Subtract {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float multiplication.
    Float64Multiply {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float division.
    Float64Divide {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Unboxed 64-bit float modulus (IEEE 754 remainder).
    Float64Modulus {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
    },

    /// Unboxed 64-bit float negation.
    Float64Negate {
        /// The value to negate.
        value: NodeId,
    },

    /// Unboxed 64-bit float exponentiation (`left ** right`).
    Float64Exponentiate {
        /// Base.
        left: NodeId,
        /// Exponent.
        right: NodeId,
    },

    /// Apply an IEEE 754 unary math function (e.g. `Math.sqrt`).
    Float64Ieee754Unary {
        /// The input value.
        value: NodeId,
        /// Index identifying which IEEE 754 function to apply.
        function_id: u32,
    },

    // ── Generic (slow-path) arithmetic ───────────────────────────────────────
    /// Generic (possibly slow-path) addition with feedback.
    GenericAdd {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic subtraction with feedback.
    GenericSubtract {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic multiplication with feedback.
    GenericMultiply {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic division with feedback.
    GenericDivide {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic modulus with feedback.
    GenericModulus {
        /// Dividend.
        left: NodeId,
        /// Divisor.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic exponentiation with feedback.
    GenericExponentiate {
        /// Base.
        left: NodeId,
        /// Exponent.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic bitwise AND with feedback.
    GenericBitwiseAnd {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic bitwise OR with feedback.
    GenericBitwiseOr {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic bitwise XOR with feedback.
    GenericBitwiseXor {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic left shift with feedback.
    GenericShiftLeft {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic arithmetic right shift with feedback.
    GenericShiftRight {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic logical right shift with feedback.
    GenericShiftRightLogical {
        /// Value to shift.
        left: NodeId,
        /// Shift amount.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic bitwise NOT with feedback.
    GenericBitwiseNot {
        /// The value to complement.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic negation with feedback.
    GenericNegate {
        /// The value to negate.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic increment with feedback.
    GenericIncrement {
        /// The value to increment.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic decrement with feedback.
    GenericDecrement {
        /// The value to decrement.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    // ── Int32 comparisons ────────────────────────────────────────────────────
    /// Unboxed 32-bit integer equality check.
    Int32Equal {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer strict equality check.
    Int32StrictEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer less-than check.
    Int32LessThan {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer less-than-or-equal check.
    Int32LessThanOrEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer greater-than check.
    Int32GreaterThan {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 32-bit integer greater-than-or-equal check.
    Int32GreaterThanOrEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    // ── Float64 comparisons ──────────────────────────────────────────────────
    /// Unboxed 64-bit float equality check (NaN-safe per IEEE 754).
    Float64Equal {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float less-than check.
    Float64LessThan {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float less-than-or-equal check.
    Float64LessThanOrEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float greater-than check.
    Float64GreaterThan {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    /// Unboxed 64-bit float greater-than-or-equal check.
    Float64GreaterThanOrEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
    },

    // ── Tagged comparisons ───────────────────────────────────────────────────
    /// Abstract equality (`==`) on tagged values.
    TaggedEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Abstract inequality (`!=`) on tagged values.
    TaggedNotEqual {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// `instanceof` test.
    TestInstanceOf {
        /// The object being tested.
        object: NodeId,
        /// The constructor to test against.
        callable: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// `in` operator test.
    TestIn {
        /// The property key.
        key: NodeId,
        /// The object to search.
        object: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Tests whether a value is undetectable (e.g. `document.all`).
    TestUndetectable {
        /// The value to test.
        value: NodeId,
    },

    /// Tests the typeof result against a literal type string.
    TestTypeOf {
        /// The value to test.
        value: NodeId,
        /// Index into the type-string table.
        literal_flag: u32,
    },

    // ── Type conversions ─────────────────────────────────────────────────────
    /// Lossless widening from unboxed `i32` to unboxed `f64`.
    ChangeInt32ToFloat64 {
        /// The input integer.
        input: NodeId,
    },

    /// Lossless widening from unboxed `u32` to unboxed `f64`.
    ChangeUint32ToFloat64 {
        /// The input unsigned integer.
        input: NodeId,
    },

    /// Truncating narrowing from unboxed `f64` to unboxed `i32`.
    ChangeFloat64ToInt32 {
        /// The input float.
        input: NodeId,
    },

    /// Checked narrowing from unboxed `f64` to unboxed `i32`; deopt if lossy.
    CheckedFloat64ToInt32 {
        /// The input float.
        input: NodeId,
    },

    /// Tag an unboxed `i32` as a Smi or heap number.
    ChangeInt32ToTagged {
        /// The input integer.
        input: NodeId,
    },

    /// Tag an unboxed `u32` as a Smi or heap number.
    ChangeUint32ToTagged {
        /// The input unsigned integer.
        input: NodeId,
    },

    /// Tag an unboxed `f64` as a heap number.
    ChangeFloat64ToTagged {
        /// The input float.
        input: NodeId,
    },

    /// Untag a tagged Smi to an unboxed `i32`; deopt if not Smi.
    ChangeTaggedToInt32 {
        /// The tagged value.
        input: NodeId,
    },

    /// Untag a tagged Smi to an unboxed `u32`; deopt if not Smi.
    ChangeTaggedToUint32 {
        /// The tagged value.
        input: NodeId,
    },

    /// Unbox a tagged Smi or heap number to an unboxed `f64`.
    ChangeTaggedToFloat64 {
        /// The tagged value.
        input: NodeId,
    },

    /// Checked untag to `i32`; deopt if the value cannot be represented.
    CheckedTaggedToInt32 {
        /// The tagged value.
        input: NodeId,
    },

    /// Checked untag to `f64`; deopt if the value is not numeric.
    CheckedTaggedToFloat64 {
        /// The tagged value.
        input: NodeId,
    },

    /// Convert any value to a boolean (`ToBoolean` abstract operation).
    ToBoolean {
        /// The value to convert.
        value: NodeId,
    },

    /// Convert any value to a string (`ToString` abstract operation).
    ToString {
        /// The value to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Convert any value to an object (`ToObject` abstract operation).
    ToObject {
        /// The value to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Convert any value to a property name (`ToName` abstract operation).
    ToName {
        /// The value to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Convert any value to a number (`ToNumber` abstract operation).
    ToNumber {
        /// The value to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Convert any value to a number or BigInt.
    ToNumberOrNumeric {
        /// The value to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    // ── Checks / guards ──────────────────────────────────────────────────────
    /// Guard: the value must be a Smi; deopt otherwise.
    CheckSmi {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be numeric (Smi or heap number); deopt otherwise.
    CheckNumber {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be a heap object (not a Smi); deopt otherwise.
    CheckHeapObject {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be a Symbol; deopt otherwise.
    CheckSymbol {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be a String; deopt otherwise.
    CheckString {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be a String or StringWrapper; deopt otherwise.
    CheckStringOrStringWrapper {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value must be a sequential one-byte String; deopt otherwise.
    CheckSeqOneByteString {
        /// The value to check.
        receiver: NodeId,
    },

    /// Guard: the value's map must be in the given set; deopt otherwise.
    CheckMaps {
        /// The object whose map is checked.
        receiver: NodeId,
        /// Feedback vector slot holding the expected map set.
        feedback_slot: u32,
    },

    /// Guard: the value's map must be in the given set; may migrate the object.
    CheckMapsWithMigration {
        /// The object whose map is checked.
        receiver: NodeId,
        /// Feedback vector slot holding the expected map set.
        feedback_slot: u32,
    },

    /// Guard: the value must be a specific constant; deopt otherwise.
    CheckValue {
        /// The value to check.
        receiver: NodeId,
        /// Index of the expected constant in the constant pool.
        expected: u32,
    },

    /// Guard: the value must equal another dynamic node; deopt otherwise.
    CheckDynamicValue {
        /// The value to check.
        receiver: NodeId,
        /// The expected value node.
        expected: NodeId,
    },

    /// Guard: the unboxed `i32` must fit in a Smi; deopt otherwise.
    CheckInt32IsSmi {
        /// The integer to check.
        input: NodeId,
    },

    /// Guard: the unboxed `u32` must fit in a Smi; deopt otherwise.
    CheckUint32IsSmi {
        /// The unsigned integer to check.
        input: NodeId,
    },

    /// Guard: a holey-float64 element is a valid Smi; deopt otherwise.
    CheckHoleyFloat64IsSmi {
        /// The float to check.
        input: NodeId,
    },

    /// Guard: checks a condition on two `i32` values; deopt if false.
    CheckInt32Condition {
        /// Left operand.
        left: NodeId,
        /// Right operand.
        right: NodeId,
        /// Condition code index.
        condition: u32,
    },

    /// Guard: the for-in cache indices have not been cleared; deopt otherwise.
    CheckCacheIndicesNotCleared {
        /// The for-in state object.
        receiver: NodeId,
        /// The cache indices being validated.
        indices: NodeId,
    },

    /// Guard: the float64 value is NaN; deopt otherwise.
    CheckFloat64IsNan {
        /// The float to check.
        input: NodeId,
    },

    // ── Property / field access ───────────────────────────────────────────────
    /// Load an in-object property at a fixed byte offset.
    LoadField {
        /// The object to load from.
        object: NodeId,
        /// Byte offset of the field within the object.
        offset: u32,
    },

    /// Store an in-object property at a fixed byte offset.
    StoreField {
        /// The object to store into.
        object: NodeId,
        /// Byte offset of the field within the object.
        offset: u32,
        /// The value to store.
        value: NodeId,
    },

    /// Load a tagged field (pointer-sized, GC-traced) at a fixed byte offset.
    LoadTaggedField {
        /// The object to load from.
        object: NodeId,
        /// Byte offset of the field.
        offset: u32,
    },

    /// Load a double-precision float field at a fixed byte offset.
    LoadDoubleField {
        /// The object to load from.
        object: NodeId,
        /// Byte offset of the field.
        offset: u32,
    },

    /// Load an element from a FixedArray at a known integer index.
    LoadFixedArrayElement {
        /// The FixedArray object.
        elements: NodeId,
        /// The zero-based element index.
        index: NodeId,
    },

    /// Load an element from a FixedDoubleArray at a known integer index.
    LoadFixedDoubleArrayElement {
        /// The FixedDoubleArray object.
        elements: NodeId,
        /// The zero-based element index.
        index: NodeId,
    },

    /// Load a possibly-hole element from a FixedDoubleArray.
    LoadHoleyFixedDoubleArrayElement {
        /// The FixedDoubleArray object.
        elements: NodeId,
        /// The zero-based element index.
        index: NodeId,
    },

    /// Store an element into a FixedArray at a known integer index.
    StoreFixedArrayElement {
        /// The FixedArray object.
        elements: NodeId,
        /// The zero-based element index.
        index: NodeId,
        /// The value to store.
        value: NodeId,
    },

    /// Store an element into a FixedDoubleArray at a known integer index.
    StoreFixedDoubleArrayElement {
        /// The FixedDoubleArray object.
        elements: NodeId,
        /// The zero-based element index.
        index: NodeId,
        /// The float value to store.
        value: NodeId,
    },

    /// Generic named property load with IC feedback.
    LoadNamedGeneric {
        /// The object to load from.
        object: NodeId,
        /// Index of the name in the constant pool.
        name: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic named property store with IC feedback.
    StoreNamedGeneric {
        /// The object to store into.
        object: NodeId,
        /// Index of the name in the constant pool.
        name: u32,
        /// The value to store.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic keyed property load with IC feedback.
    LoadKeyedGeneric {
        /// The object to load from.
        object: NodeId,
        /// The property key.
        key: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic keyed property store with IC feedback.
    StoreKeyedGeneric {
        /// The object to store into.
        object: NodeId,
        /// The property key.
        key: NodeId,
        /// The value to store.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Load a global variable by name.
    LoadGlobal {
        /// Index of the name in the constant pool.
        name: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Store a value into a global variable by name.
    StoreGlobal {
        /// Index of the name in the constant pool.
        name: u32,
        /// The value to store.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Load a value from a scope-context slot.
    LoadContextSlot {
        /// The context object.
        context: NodeId,
        /// Scope depth.
        depth: u32,
        /// Slot index.
        slot: u32,
    },

    /// Store a value into a scope-context slot.
    StoreContextSlot {
        /// The context object.
        context: NodeId,
        /// Scope depth.
        depth: u32,
        /// Slot index.
        slot: u32,
        /// The value to store.
        value: NodeId,
    },

    /// Load from the current (innermost) context slot.
    LoadCurrentContextSlot {
        /// Slot index in the current context.
        slot: u32,
    },

    /// Store into the current (innermost) context slot.
    StoreCurrentContextSlot {
        /// Slot index in the current context.
        slot: u32,
        /// The value to store.
        value: NodeId,
    },

    // ── Calls and constructs ─────────────────────────────────────────────────
    /// Generic function call.
    Call {
        /// The function to call.
        callee: NodeId,
        /// The receiver (`this`).
        receiver: NodeId,
        /// Ordered argument list.
        args: Vec<NodeId>,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Call to a statically-known function.
    CallKnownFunction {
        /// The function to call.
        callee: NodeId,
        /// The receiver (`this`).
        receiver: NodeId,
        /// Ordered argument list.
        args: Vec<NodeId>,
    },

    /// Call to a builtin function by ID.
    CallBuiltin {
        /// Zero-based builtin function identifier.
        builtin_id: u32,
        /// Ordered argument list.
        args: Vec<NodeId>,
    },

    /// Call to a runtime helper function by ID.
    CallRuntime {
        /// Zero-based runtime function identifier.
        function_id: u32,
        /// Ordered argument list.
        args: Vec<NodeId>,
    },

    /// Generic function call with spread (`f(...args)`).
    CallWithSpread {
        /// The function to call.
        callee: NodeId,
        /// The receiver (`this`).
        receiver: NodeId,
        /// Argument list (last element is the spread).
        args: Vec<NodeId>,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Generic `new` expression.
    Construct {
        /// The constructor function.
        constructor: NodeId,
        /// Ordered argument list.
        args: Vec<NodeId>,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// `new` expression with spread (`new F(...args)`).
    ConstructWithSpread {
        /// The constructor function.
        constructor: NodeId,
        /// Argument list (last element is the spread).
        args: Vec<NodeId>,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    // ── Object / array creation ───────────────────────────────────────────────
    /// Create an object literal from a boilerplate.
    CreateObjectLiteral {
        /// Index of the boilerplate descriptor in the constant pool.
        boilerplate_descriptor: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// Creation flags (e.g. `SHALLOW_PROPERTIES`).
        flags: u32,
    },

    /// Create an array literal from a boilerplate.
    CreateArrayLiteral {
        /// Index of the boilerplate in the constant pool.
        constant_elements: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// Creation flags.
        flags: u32,
    },

    /// Create a shallow copy of an object literal.
    CreateShallowObjectLiteral {
        /// Index of the boilerplate descriptor in the constant pool.
        boilerplate_descriptor: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// Creation flags.
        flags: u32,
    },

    /// Create a shallow copy of an array literal.
    CreateShallowArrayLiteral {
        /// Index of the boilerplate in the constant pool.
        constant_elements: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// Creation flags.
        flags: u32,
    },

    /// Allocate a new function context.
    CreateFunctionContext {
        /// Index of the scope info in the constant pool.
        scope_info: u32,
        /// Number of context slots.
        slot_count: u32,
    },

    /// Push a new context as the active closure context.
    ///
    /// Takes the new context from [`CreateFunctionContext`] or
    /// [`CreateBlockContext`] and installs it as `RT_CONTEXT`, returning
    /// the previous context so the caller can save it in a register.
    PushContext {
        /// The new context to push.
        context: NodeId,
    },

    /// Pop (restore) a previously saved context as the active closure
    /// context.
    PopContext {
        /// The saved context to restore.
        context: NodeId,
    },

    /// Allocate a new block scope context.
    CreateBlockContext {
        /// Index of the scope info in the constant pool.
        scope_info: u32,
    },

    /// Allocate a new catch scope context.
    CreateCatchContext {
        /// The caught exception value.
        exception: NodeId,
        /// Index of the scope info in the constant pool.
        scope_info: u32,
    },

    /// Allocate a new `with` scope context.
    CreateWithContext {
        /// The `with` object.
        object: NodeId,
        /// Index of the scope info in the constant pool.
        scope_info: u32,
    },

    /// Create a new closure (slow path).
    CreateClosure {
        /// Index of the shared function info in the constant pool.
        shared_function_info: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// Closure flags.
        flags: u32,
    },

    /// Create a new closure (fast path using feedback).
    FastCreateClosure {
        /// Index of the shared function info in the constant pool.
        shared_function_info: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Allocate an empty object literal (`{}`).
    CreateEmptyObjectLiteral,

    /// Allocate an empty array literal (`[]`).
    CreateEmptyArrayLiteral {
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Create a mapped `arguments` object for sloppy-mode functions.
    CreateMappedArguments,

    /// Create an unmapped `arguments` object for strict-mode functions.
    CreateUnmappedArguments,

    /// Create the rest-parameter array from surplus arguments.
    CreateRestParameter,

    /// Create a RegExp literal.
    CreateRegExpLiteral {
        /// Index of the pattern string in the constant pool.
        pattern: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
        /// RegExp flags bitmask.
        flags: u32,
    },

    // ── Control-value producers ───────────────────────────────────────────────
    /// SSA Φ-node: selects among values depending on which predecessor was
    /// taken.  The `inputs` list must have the same length as the number of
    /// predecessors of the containing block.
    Phi {
        /// One input [`NodeId`] per predecessor basic block, in predecessor
        /// order.
        inputs: Vec<NodeId>,
    },

    /// Materialise the `arguments` object as a FixedArray.
    ArgumentsElements {
        /// Type of arguments mapping (0 = mapped, 1 = unmapped).
        kind: u32,
    },

    /// Materialise the rest-parameter tail as a FixedArray.
    RestElements {
        /// Zero-based index of the first rest parameter.
        formal_parameter_count: u32,
    },

    /// A virtual (not yet allocated) object used for escape analysis.
    VirtualObject {
        /// Map index identifying the object's shape.
        map: u32,
    },

    // ── Miscellaneous ─────────────────────────────────────────────────────────
    /// Load a template object for a tagged template literal.
    GetTemplateObject {
        /// Index of the template descriptor in the constant pool.
        description: u32,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Check whether an object is present in a prototype chain.
    HasInPrototypeChain {
        /// The object to test.
        object: NodeId,
        /// The prototype to search for.
        prototype: NodeId,
    },

    /// Delete a named property (`delete obj.key`).
    DeleteProperty {
        /// The object to modify.
        object: NodeId,
        /// The property key.
        key: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Prepare a for-in enumeration (returns cache array and length).
    ForInPrepare {
        /// The object being enumerated.
        enumerator: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Produce the next key during a for-in loop.
    ForInNext {
        /// The object being enumerated.
        receiver: NodeId,
        /// The cache index.
        cache_index: NodeId,
        /// The cache array.
        cache_array: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// Load the length of the enumeration cache.
    LoadEnumCacheLength {
        /// The map object whose cache length is read.
        map: NodeId,
    },

    /// Load a character code from a string at an integer index.
    StringAt {
        /// The string object.
        string: NodeId,
        /// The character index.
        index: NodeId,
    },

    /// Load the `length` property of a string.
    StringLength {
        /// The string object.
        string: NodeId,
    },

    /// Concatenate two strings.
    StringConcat {
        /// The left string.
        left: NodeId,
        /// The right string.
        right: NodeId,
    },

    /// Test whether two strings are equal.
    StringEqual {
        /// The left string.
        left: NodeId,
        /// The right string.
        right: NodeId,
    },

    /// Convert a number to its string representation.
    NumberToString {
        /// The number to convert.
        value: NodeId,
        /// Feedback vector slot index.
        feedback_slot: u32,
    },

    /// The `typeof` operator.
    TypeOf {
        /// The value to inspect.
        value: NodeId,
    },

    /// A `debugger` statement (no-op unless a debugger is attached).
    Debugger,

    /// An unconditional abort — emitted after a deopt bailout.
    Abort {
        /// Reason code (indexes into an engine-internal reason table).
        reason: u32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// ControlNode
// ─────────────────────────────────────────────────────────────────────────────

/// A Maglev IR terminator node that ends a [`BasicBlock`].
///
/// Every basic block has exactly one `ControlNode` as its last instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlNode {
    /// Unconditional jump to `target`.
    Jump {
        /// Index of the target [`BasicBlock`] within the graph.
        target: u32,
    },

    /// Conditional branch: if `condition` is truthy jump to `if_true`,
    /// otherwise fall through to `if_false`.
    Branch {
        /// The boolean-valued condition node.
        condition: NodeId,
        /// Block index taken when the condition is true.
        if_true: u32,
        /// Block index taken when the condition is false.
        if_false: u32,
    },

    /// Trigger a deoptimisation.  Control transfers back to the interpreter at
    /// `bytecode_offset` with the given `reason`.
    Deoptimize {
        /// Byte offset in the bytecode array at which to resume interpretation.
        bytecode_offset: u32,
        /// Reason code (indexes into an engine-internal reason table).
        reason: u32,
    },

    /// Return from the function with `value` as the return value.
    Return {
        /// The value to return.
        value: NodeId,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// BasicBlock
// ─────────────────────────────────────────────────────────────────────────────

/// A Maglev basic block: a straight-line list of [`ValueNode`]s terminated by
/// a single [`ControlNode`].
///
/// # Invariants
///
/// - [`BasicBlock::control`] is `None` during construction and must be set
///   exactly once via [`BasicBlock::set_control`] before the block is
///   considered complete.
/// - Node IDs are graph-global; they are assigned by the owning
///   [`MaglevGraph`] during graph construction.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Zero-based index of this block within the graph.
    pub id: u32,

    /// The value-producing nodes of this block, in execution order.
    pub nodes: Vec<(NodeId, ValueNode)>,

    /// The terminator instruction (set exactly once).
    pub control: Option<ControlNode>,

    /// Predecessor block indices (filled in during CFG construction).
    pub predecessors: Vec<u32>,

    /// Whether this block is a loop header (has at least one back-edge
    /// predecessor).  Set by the optimizer so codegen can align it.
    pub is_loop_header: bool,
}

impl BasicBlock {
    /// Create a new, empty block with the given `id`.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            control: None,
            predecessors: Vec::new(),
            is_loop_header: false,
        }
    }

    /// Append a [`ValueNode`] to this block with a block-local [`NodeId`] and
    /// return it.
    ///
    /// IDs are assigned sequentially starting at `0` *within this block*,
    /// so they are only unique within the block itself.  When nodes from
    /// multiple blocks must be referenced (e.g. [`ValueNode::Phi`] inputs),
    /// use [`MaglevGraph::add_value_node`] instead, which issues graph-global
    /// [`NodeId`]s.
    pub fn push_value(&mut self, node: ValueNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push((id, node));
        id
    }

    /// Append a [`ValueNode`] with an explicit, caller-supplied [`NodeId`].
    ///
    /// This is the low-level primitive used by [`MaglevGraph::add_value_node`]
    /// to attach graph-global IDs.  Callers must ensure that `id` is unique
    /// across the entire graph.
    pub fn push_with_id(&mut self, id: NodeId, node: ValueNode) {
        self.nodes.push((id, node));
    }

    /// Set the terminator [`ControlNode`] for this block.
    ///
    /// # Panics
    ///
    /// Panics if a control node has already been set.
    pub fn set_control(&mut self, control: ControlNode) {
        assert!(
            self.control.is_none(),
            "BasicBlock {}: control node already set",
            self.id
        );
        self.control = Some(control);
    }

    /// Returns `true` if the block has a terminator.
    pub fn is_complete(&self) -> bool {
        self.control.is_some()
    }

    /// Returns the terminator control node, if set.
    pub fn control(&self) -> Option<&ControlNode> {
        self.control.as_ref()
    }

    /// Add a predecessor block index.
    pub fn add_predecessor(&mut self, pred: u32) {
        self.predecessors.push(pred);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MaglevGraph
// ─────────────────────────────────────────────────────────────────────────────

/// The complete Maglev control-flow graph for a single JavaScript function.
///
/// Owns all [`BasicBlock`]s and manages the graph-wide [`NodeId`] counter.
///
/// # Block indices
///
/// Block indices (`u32`) used in [`ControlNode::Jump`],
/// [`ControlNode::Branch`], and [`BasicBlock::predecessors`] are zero-based
/// indices into [`MaglevGraph::blocks`].
///
/// # Node IDs
///
/// Use [`MaglevGraph::add_value_node`] to insert nodes with graph-global
/// [`NodeId`]s when cross-block references (e.g. [`ValueNode::Phi`] inputs)
/// are needed.  [`BasicBlock::push_value`] assigns block-local IDs and is
/// sufficient for single-block graphs.
///
/// # Example
///
/// ```
/// use stator_core::compiler::maglev::ir::{
///     BasicBlock, ControlNode, MaglevGraph, ValueNode,
/// };
///
/// let mut graph = MaglevGraph::new(2);
///
/// // block 0: load param 0 and jump to block 1
/// graph.add_block(BasicBlock::new(0));
/// let _p0 = graph.add_value_node(0, ValueNode::Parameter { index: 0 });
/// graph.block_mut(0).unwrap().set_control(ControlNode::Jump { target: 1 });
///
/// // block 1: return undefined
/// graph.add_block(BasicBlock::new(1));
/// let undef = graph.add_value_node(1, ValueNode::UndefinedConstant).unwrap();
/// graph.block_mut(1).unwrap().set_control(ControlNode::Return { value: undef });
///
/// assert_eq!(graph.blocks().len(), 2);
/// assert_eq!(graph.parameter_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MaglevGraph {
    /// All basic blocks, in insertion order.
    blocks: Vec<BasicBlock>,

    /// Number of formal parameters for the compiled function.
    parameter_count: u32,

    /// Graph-wide node counter, incremented by [`MaglevGraph::add_value_node`].
    next_node_id: u32,

    /// Number of call sites identified as inlining candidates by the
    /// optimizer's inlining analysis pass.
    inline_candidates: u32,
}

impl MaglevGraph {
    /// Create an empty graph for a function with `parameter_count` formal
    /// parameters.
    pub fn new(parameter_count: u32) -> Self {
        Self {
            blocks: Vec::new(),
            parameter_count,
            next_node_id: 0,
            inline_candidates: 0,
        }
    }

    /// Append a [`BasicBlock`] to the graph.
    ///
    /// The block's `id` field should match its final index in the graph; this
    /// is not enforced but callers are expected to maintain the invariant.
    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.push(block);
    }

    /// Append a [`ValueNode`] to block `block_idx` using a graph-global
    /// [`NodeId`] and return the assigned ID.
    ///
    /// The returned [`NodeId`] is unique across the entire graph and can be
    /// safely used as an input to nodes in *other* blocks (e.g.
    /// [`ValueNode::Phi`] inputs).
    ///
    /// Returns `None` if `block_idx` is out-of-range.
    pub fn add_value_node(&mut self, block_idx: u32, node: ValueNode) -> Option<NodeId> {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let block = self.blocks.get_mut(block_idx as usize)?;
        block.push_with_id(id, node);
        Some(id)
    }

    /// Allocate a fresh graph-global [`NodeId`] without inserting a node.
    ///
    /// The caller is responsible for inserting a node with this ID into the
    /// appropriate block (e.g. via [`BasicBlock::push_with_id`] or
    /// `nodes.insert`).
    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Return an immutable slice of all blocks.
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// Return a mutable slice of all blocks.
    pub fn blocks_mut(&mut self) -> &mut [BasicBlock] {
        &mut self.blocks
    }

    /// Return the number of formal parameters.
    pub fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Look up a block by its index.  Returns `None` if the index is
    /// out-of-range.
    pub fn block(&self, index: u32) -> Option<&BasicBlock> {
        self.blocks.get(index as usize)
    }

    /// Look up a block mutably by its index.  Returns `None` if the index is
    /// out-of-range.
    pub fn block_mut(&mut self, index: u32) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(index as usize)
    }

    /// Return the entry block (block 0), if present.
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }

    /// Return `true` if the graph is *degenerate*: the entry block
    /// immediately deoptimises without performing any meaningful
    /// computation.  Compiling such a graph produces JIT code that
    /// always returns `JIT_DEOPT`, which cascades deoptimisation to
    /// callers and severely degrades performance.
    ///
    /// A graph is considered degenerate when the entry block's
    /// terminator is [`ControlNode::Deoptimize`] and the block contains
    /// no value nodes other than constants.
    pub fn is_degenerate(&self) -> bool {
        let Some(entry) = self.entry_block() else {
            return true;
        };
        let Some(ctrl) = entry.control() else {
            return false;
        };
        if !matches!(ctrl, ControlNode::Deoptimize { .. }) {
            return false;
        }
        // Entry block immediately deoptimises — check that there are
        // no real value nodes (constants don't count).
        entry.nodes.iter().all(|(_, n)| {
            matches!(
                n,
                ValueNode::SmiConstant { .. }
                    | ValueNode::Float64Constant { .. }
                    | ValueNode::UndefinedConstant
                    | ValueNode::TrueConstant
                    | ValueNode::FalseConstant
                    | ValueNode::NullConstant
            )
        })
    }

    /// Set the number of inlining candidate call sites identified by the
    /// optimizer.
    pub fn set_inline_candidates(&mut self, count: u32) {
        self.inline_candidates = count;
    }

    /// Return the number of inlining candidate call sites.
    pub fn inline_candidates(&self) -> u32 {
        self.inline_candidates
    }

    /// Look up a [`ValueNode`] by its [`NodeId`].
    ///
    /// Performs a linear scan of all blocks; intended for compile-time
    /// lookups, not hot-path code.
    pub fn node(&self, id: NodeId) -> Option<&ValueNode> {
        for block in &self.blocks {
            for (nid, node) in &block.nodes {
                if *nid == id {
                    return Some(node);
                }
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Build a minimal one-block graph that returns `undefined`.
    fn single_block_graph() -> MaglevGraph {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let undef = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: undef });
        graph.add_block(block);
        graph
    }

    // ── BasicBlock ────────────────────────────────────────────────────────────

    #[test]
    fn test_basic_block_starts_empty() {
        let block = BasicBlock::new(7);
        assert_eq!(block.id, 7);
        assert!(block.nodes.is_empty());
        assert!(block.control.is_none());
        assert!(!block.is_complete());
    }

    #[test]
    fn test_basic_block_push_assigns_sequential_ids() {
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 1 });
        let b = block.push_value(ValueNode::SmiConstant { value: 2 });
        let c = block.push_value(ValueNode::SmiConstant { value: 3 });
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
        assert_eq!(c, NodeId(2));
        assert_eq!(block.nodes.len(), 3);
    }

    #[test]
    fn test_basic_block_set_control_completes_block() {
        let mut block = BasicBlock::new(0);
        let undef = block.push_value(ValueNode::UndefinedConstant);
        assert!(!block.is_complete());
        block.set_control(ControlNode::Return { value: undef });
        assert!(block.is_complete());
    }

    #[test]
    #[should_panic(expected = "control node already set")]
    fn test_basic_block_set_control_twice_panics() {
        let mut block = BasicBlock::new(0);
        let undef = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: undef });
        // Second call must panic.
        block.set_control(ControlNode::Return { value: NodeId(0) });
    }

    #[test]
    fn test_basic_block_predecessors() {
        let mut block = BasicBlock::new(3);
        block.add_predecessor(0);
        block.add_predecessor(1);
        assert_eq!(block.predecessors, vec![0, 1]);
    }

    // ── MaglevGraph ───────────────────────────────────────────────────────────

    #[test]
    fn test_graph_new_empty() {
        let graph = MaglevGraph::new(3);
        assert_eq!(graph.parameter_count(), 3);
        assert!(graph.blocks().is_empty());
        assert!(graph.entry_block().is_none());
        assert!(graph.block(0).is_none());
    }

    #[test]
    fn test_graph_add_and_retrieve_blocks() {
        let graph = single_block_graph();
        assert_eq!(graph.blocks().len(), 1);
        let entry = graph.entry_block().unwrap();
        assert_eq!(entry.id, 0);
    }

    #[test]
    fn test_graph_block_out_of_range_returns_none() {
        let graph = single_block_graph();
        assert!(graph.block(99).is_none());
    }

    // ── ValueNode — constants ─────────────────────────────────────────────────

    #[test]
    fn test_smi_constant_node() {
        let mut block = BasicBlock::new(0);
        let id = block.push_value(ValueNode::SmiConstant { value: 42 });
        let (nid, node) = &block.nodes[0];
        assert_eq!(*nid, id);
        assert_eq!(*node, ValueNode::SmiConstant { value: 42 });
    }

    #[test]
    fn test_float64_constant_node() {
        let mut block = BasicBlock::new(0);
        let id = block.push_value(ValueNode::Float64Constant { value: 3.14 });
        let (_, node) = &block.nodes[0];
        assert_eq!(id, NodeId(0));
        if let ValueNode::Float64Constant { value } = node {
            assert!((value - 3.14).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant");
        }
    }

    #[test]
    fn test_true_false_null_undefined_constants() {
        let mut block = BasicBlock::new(0);
        block.push_value(ValueNode::TrueConstant);
        block.push_value(ValueNode::FalseConstant);
        block.push_value(ValueNode::NullConstant);
        block.push_value(ValueNode::UndefinedConstant);
        assert_eq!(block.nodes.len(), 4);
        assert_eq!(block.nodes[0].1, ValueNode::TrueConstant);
        assert_eq!(block.nodes[1].1, ValueNode::FalseConstant);
        assert_eq!(block.nodes[2].1, ValueNode::NullConstant);
        assert_eq!(block.nodes[3].1, ValueNode::UndefinedConstant);
    }

    #[test]
    fn test_string_constant_node() {
        let mut block = BasicBlock::new(0);
        block.push_value(ValueNode::StringConstant {
            value: "hello".to_string(),
        });
        if let ValueNode::StringConstant { value } = &block.nodes[0].1 {
            assert_eq!(value, "hello");
        } else {
            panic!("expected StringConstant");
        }
    }

    // ── ValueNode — parameters ────────────────────────────────────────────────

    #[test]
    fn test_parameter_node() {
        let mut block = BasicBlock::new(0);
        let id = block.push_value(ValueNode::Parameter { index: 2 });
        assert_eq!(id, NodeId(0));
        assert_eq!(block.nodes[0].1, ValueNode::Parameter { index: 2 });
    }

    // ── ValueNode — arithmetic ────────────────────────────────────────────────

    #[test]
    fn test_checked_smi_add_node() {
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 10 });
        let b = block.push_value(ValueNode::SmiConstant { value: 20 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        assert_eq!(sum, NodeId(2));
        if let ValueNode::CheckedSmiAdd { left, right } = block.nodes[2].1 {
            assert_eq!(left, a);
            assert_eq!(right, b);
        } else {
            panic!("expected CheckedSmiAdd");
        }
    }

    #[test]
    fn test_float64_arithmetic_nodes() {
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 1.0 });
        let y = block.push_value(ValueNode::Float64Constant { value: 2.0 });
        let add = block.push_value(ValueNode::Float64Add { left: x, right: y });
        let sub = block.push_value(ValueNode::Float64Subtract { left: x, right: y });
        let mul = block.push_value(ValueNode::Float64Multiply { left: x, right: y });
        let div = block.push_value(ValueNode::Float64Divide { left: x, right: y });
        assert_eq!(add, NodeId(2));
        assert_eq!(sub, NodeId(3));
        assert_eq!(mul, NodeId(4));
        assert_eq!(div, NodeId(5));
    }

    // ── ValueNode — Phi ───────────────────────────────────────────────────────

    #[test]
    fn test_phi_node() {
        // A Φ with two inputs (one per predecessor).
        let mut block = BasicBlock::new(2);
        let a = NodeId(0);
        let b = NodeId(1);
        let phi = block.push_value(ValueNode::Phi { inputs: vec![a, b] });
        assert_eq!(phi, NodeId(0));
        if let ValueNode::Phi { inputs } = &block.nodes[0].1 {
            assert_eq!(inputs, &[NodeId(0), NodeId(1)]);
        } else {
            panic!("expected Phi");
        }
    }

    // ── ValueNode — load/store ─────────────────────────────────────────────────

    #[test]
    fn test_load_field_node() {
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let field = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        assert_eq!(field, NodeId(1));
        assert_eq!(
            block.nodes[1].1,
            ValueNode::LoadField {
                object: obj,
                offset: 8
            }
        );
    }

    #[test]
    fn test_store_field_node() {
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::SmiConstant { value: 99 });
        let store = block.push_value(ValueNode::StoreField {
            object: obj,
            offset: 16,
            value: val,
        });
        assert_eq!(store, NodeId(2));
    }

    // ── ValueNode — Call ──────────────────────────────────────────────────────

    #[test]
    fn test_call_node() {
        let mut block = BasicBlock::new(0);
        let callee = block.push_value(ValueNode::Parameter { index: 0 });
        let recv = block.push_value(ValueNode::UndefinedConstant);
        let arg0 = block.push_value(ValueNode::SmiConstant { value: 1 });
        let call = block.push_value(ValueNode::Call {
            callee,
            receiver: recv,
            args: vec![arg0],
            feedback_slot: 0,
        });
        assert_eq!(call, NodeId(3));
        if let ValueNode::Call { args, .. } = &block.nodes[3].1 {
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], arg0);
        } else {
            panic!("expected Call");
        }
    }

    // ── ControlNode ───────────────────────────────────────────────────────────

    #[test]
    fn test_control_jump() {
        let mut block = BasicBlock::new(0);
        block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Jump { target: 1 });
        assert_eq!(block.control, Some(ControlNode::Jump { target: 1 }));
    }

    #[test]
    fn test_control_branch() {
        let mut block = BasicBlock::new(0);
        let cond = block.push_value(ValueNode::TrueConstant);
        block.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        if let Some(ControlNode::Branch {
            condition,
            if_true,
            if_false,
        }) = block.control
        {
            assert_eq!(condition, cond);
            assert_eq!(if_true, 1);
            assert_eq!(if_false, 2);
        } else {
            panic!("expected Branch");
        }
    }

    #[test]
    fn test_control_deoptimize() {
        let mut block = BasicBlock::new(0);
        block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Deoptimize {
            bytecode_offset: 42,
            reason: 7,
        });
        assert_eq!(
            block.control,
            Some(ControlNode::Deoptimize {
                bytecode_offset: 42,
                reason: 7,
            })
        );
    }

    #[test]
    fn test_control_return() {
        let mut block = BasicBlock::new(0);
        let undef = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: undef });
        assert_eq!(
            block.control,
            Some(ControlNode::Return { value: NodeId(0) })
        );
    }

    // ── Full graph construction ───────────────────────────────────────────────

    #[test]
    fn test_construct_return_parameter_graph() {
        // graph: entry block returns parameter 0
        let mut graph = MaglevGraph::new(1);
        let mut entry = BasicBlock::new(0);
        let p0 = entry.push_value(ValueNode::Parameter { index: 0 });
        entry.set_control(ControlNode::Return { value: p0 });
        graph.add_block(entry);

        assert_eq!(graph.blocks().len(), 1);
        let b = graph.block(0).unwrap();
        assert_eq!(b.nodes.len(), 1);
        assert!(b.is_complete());
    }

    #[test]
    fn test_construct_two_block_graph_with_jump() {
        // block 0 → jump → block 1 → return
        let mut graph = MaglevGraph::new(0);

        let mut b0 = BasicBlock::new(0);
        b0.push_value(ValueNode::UndefinedConstant);
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        let undef = b1.push_value(ValueNode::UndefinedConstant);
        b1.set_control(ControlNode::Return { value: undef });
        graph.add_block(b1);

        assert_eq!(graph.blocks().len(), 2);
        assert_eq!(
            graph.block(0).unwrap().control,
            Some(ControlNode::Jump { target: 1 })
        );
    }

    #[test]
    fn test_construct_branch_graph() {
        // block 0: branch on param0 → block 1 (true) / block 2 (false)
        // block 1: return SmiConstant(1)
        // block 2: return SmiConstant(0)
        let mut graph = MaglevGraph::new(1);

        let mut b0 = BasicBlock::new(0);
        let cond = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 1,
            if_false: 2,
        });
        graph.add_block(b0);

        let mut b1 = BasicBlock::new(1);
        let one = b1.push_value(ValueNode::SmiConstant { value: 1 });
        b1.set_control(ControlNode::Return { value: one });
        b1.add_predecessor(0);
        graph.add_block(b1);

        let mut b2 = BasicBlock::new(2);
        let zero = b2.push_value(ValueNode::SmiConstant { value: 0 });
        b2.set_control(ControlNode::Return { value: zero });
        b2.add_predecessor(0);
        graph.add_block(b2);

        assert_eq!(graph.blocks().len(), 3);
        assert_eq!(graph.block(1).unwrap().predecessors, vec![0]);
        assert_eq!(graph.block(2).unwrap().predecessors, vec![0]);
    }

    #[test]
    fn test_construct_diamond_with_phi() {
        // Classic SSA diamond using graph-global NodeIds so that the Phi inputs
        // from different blocks are unambiguous.
        //
        //   entry → if-true / if-false → merge (Φ) → return
        let mut graph = MaglevGraph::new(1);

        // block 0: entry – branch on param0
        graph.add_block(BasicBlock::new(0));
        let cond = graph
            .add_value_node(0, ValueNode::Parameter { index: 0 })
            .unwrap();
        graph
            .block_mut(0)
            .unwrap()
            .set_control(ControlNode::Branch {
                condition: cond,
                if_true: 1,
                if_false: 2,
            });

        // block 1: if-true – produce SmiConstant(1) → jump merge
        graph.add_block(BasicBlock::new(1));
        graph.block_mut(1).unwrap().add_predecessor(0);
        let one = graph
            .add_value_node(1, ValueNode::SmiConstant { value: 1 })
            .unwrap();
        graph
            .block_mut(1)
            .unwrap()
            .set_control(ControlNode::Jump { target: 3 });

        // block 2: if-false – produce SmiConstant(0) → jump merge
        graph.add_block(BasicBlock::new(2));
        graph.block_mut(2).unwrap().add_predecessor(0);
        let zero = graph
            .add_value_node(2, ValueNode::SmiConstant { value: 0 })
            .unwrap();
        graph
            .block_mut(2)
            .unwrap()
            .set_control(ControlNode::Jump { target: 3 });

        // block 3: merge – Φ([one, zero]) → return
        graph.add_block(BasicBlock::new(3));
        graph.block_mut(3).unwrap().add_predecessor(1);
        graph.block_mut(3).unwrap().add_predecessor(2);
        let phi = graph
            .add_value_node(
                3,
                ValueNode::Phi {
                    inputs: vec![one, zero],
                },
            )
            .unwrap();
        graph
            .block_mut(3)
            .unwrap()
            .set_control(ControlNode::Return { value: phi });

        // The four global IDs must be distinct.
        assert_ne!(cond, one);
        assert_ne!(cond, zero);
        assert_ne!(one, zero);
        assert_ne!(phi, one);
        assert_ne!(phi, zero);

        assert_eq!(graph.blocks().len(), 4);
        let merge = graph.block(3).unwrap();
        assert_eq!(merge.predecessors, vec![1, 2]);
        if let ValueNode::Phi { inputs } = &merge.nodes[0].1 {
            assert_eq!(inputs, &[one, zero]);
        } else {
            panic!("expected Phi in merge block");
        }
    }

    #[test]
    fn test_add_value_node_global_ids_are_unique_across_blocks() {
        // Nodes in different blocks must receive distinct graph-global IDs.
        let mut graph = MaglevGraph::new(0);
        graph.add_block(BasicBlock::new(0));
        graph.add_block(BasicBlock::new(1));

        let a = graph
            .add_value_node(0, ValueNode::UndefinedConstant)
            .unwrap();
        let b = graph
            .add_value_node(0, ValueNode::UndefinedConstant)
            .unwrap();
        let c = graph
            .add_value_node(1, ValueNode::UndefinedConstant)
            .unwrap();

        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn test_add_value_node_out_of_range_returns_none() {
        let mut graph = MaglevGraph::new(0);
        assert!(
            graph
                .add_value_node(99, ValueNode::UndefinedConstant)
                .is_none()
        );
    }

    #[test]
    fn test_graph_blocks_mut() {
        let mut graph = single_block_graph();
        graph.blocks_mut()[0].id = 99;
        assert_eq!(graph.block(0).unwrap().id, 99);
    }

    #[test]
    fn test_graph_block_mut() {
        let mut graph = single_block_graph();
        {
            let b = graph.block_mut(0).unwrap();
            b.add_predecessor(5);
        }
        assert_eq!(graph.block(0).unwrap().predecessors, vec![5]);
    }
}

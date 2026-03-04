//! Pre-CLIF type-specialisation passes for the Turbofan Cranelift backend.
//!
//! These passes operate on a [`MaglevGraph`] **before** it is lowered to
//! Cranelift CLIF, narrowing slow-path generic operations into faster
//! specialised forms.  They are composed and applied in order by
//! [`run_pre_clif_passes`].
//!
//! # Passes
//!
//! 1. **Type narrowing from feedback** ([`narrow_types_from_feedback`]) —
//!    replaces `Generic*` arithmetic/bitwise nodes with their
//!    `CheckedSmi*` / `Int32*` counterparts when the corresponding
//!    [`FeedbackVector`] slot is in the
//!    [`InlineCacheState::Monomorphic`] state.  This avoids
//!    slow JS-semantics dispatch and instead emits speculative fast-path code
//!    that deopts on type mismatch.
//!
//! 2. **Hot call-site specialisation** ([`specialize_call_sites`]) —
//!    converts [`ValueNode::Call`] nodes to [`ValueNode::CallKnownFunction`]
//!    when the call's feedback slot is [`InlineCacheState::Monomorphic`],
//!    signalling to downstream lowering that a single callee was seen and
//!    optimised dispatch is safe.
//!
//! 3. **Load/store elimination** ([`eliminate_redundant_loads`]) —
//!    within each basic block, if the same in-object field is loaded twice
//!    without an intervening store, the second load is replaced by the first
//!    load's result.  This is a local Common Sub-expression Elimination (CSE)
//!    pass for property accesses.
//!
//! 4. **Escape analysis / allocation sinking**
//!    ([`sink_non_escaping_allocations`]) — if an allocation
//!    ([`ValueNode::CreateEmptyObjectLiteral`]) is never *returned*, passed as
//!    a *call argument*, or *stored into another object*, its NodeId is
//!    replaced by a [`ValueNode::VirtualObject`] placeholder, signalling that
//!    the allocation need not be heap-allocated (it can be handled virtually
//!    by the code generator).
//!
//! # Usage
//!
//! ```
//! use stator_core::bytecode::feedback::{
//!     FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
//! };
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::turbofan::specialize::run_pre_clif_passes;
//!
//! // Build a graph with a GenericAdd at feedback slot 0.
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! let a = block.push_value(ValueNode::SmiConstant { value: 3 });
//! let b = block.push_value(ValueNode::SmiConstant { value: 4 });
//! let add = block.push_value(ValueNode::GenericAdd {
//!     left: a,
//!     right: b,
//!     feedback_slot: 0,
//! });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! // The feedback vector reports monomorphic BinaryOp at slot 0.
//! let metadata =
//!     FeedbackMetadata::new(vec![FeedbackSlotKind::BinaryOp]);
//! let mut fv = FeedbackVector::new(&metadata);
//! fv.set_state(0, InlineCacheState::Monomorphic);
//!
//! run_pre_clif_passes(&mut graph, Some(&fv));
//!
//! // The GenericAdd has been narrowed to a CheckedSmiAdd.
//! let node = &graph.blocks()[0].nodes[2].1;
//! assert!(matches!(node, ValueNode::CheckedSmiAdd { .. }));
//! ```

use std::collections::{HashMap, HashSet};

use crate::bytecode::feedback::{FeedbackVector, InlineCacheState};
use crate::compiler::maglev::ir::{BasicBlock, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Run all four pre-CLIF specialisation passes on `graph` in place.
///
/// Passes are applied in the order:
/// 1. type narrowing from feedback,
/// 2. hot call-site specialisation,
/// 3. load/store elimination,
/// 4. escape analysis / allocation sinking.
///
/// `fv` is optional — passes that require feedback silently skip when it is
/// `None`.
pub fn run_pre_clif_passes(graph: &mut MaglevGraph, fv: Option<&FeedbackVector>) {
    narrow_types_from_feedback(graph, fv);
    specialize_call_sites(graph, fv);
    eliminate_redundant_loads(graph);
    sink_non_escaping_allocations(graph);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Type narrowing from feedback
// ─────────────────────────────────────────────────────────────────────────────

/// Replace `Generic*` arithmetic/bitwise nodes with specialised equivalents
/// when the corresponding feedback slot is [`InlineCacheState::Monomorphic`].
///
/// A monomorphic slot means only one type was ever observed at that site.
/// Replacing the generic (slow-path) operation with a speculative fast-path
/// node (e.g. [`ValueNode::CheckedSmiAdd`]) allows Cranelift to emit a
/// compact integer fast-path with an overflow deopt guard instead of a full
/// JS-semantics dispatch.
///
/// The following replacements are performed:
///
/// | Generic node              | Narrowed to              |
/// |---------------------------|--------------------------|
/// | `GenericAdd`              | `CheckedSmiAdd`          |
/// | `GenericSubtract`         | `CheckedSmiSubtract`     |
/// | `GenericMultiply`         | `CheckedSmiMultiply`     |
/// | `GenericDivide`           | `CheckedSmiDivide`       |
/// | `GenericModulus`          | `CheckedSmiModulus`      |
/// | `GenericBitwiseAnd`       | `Int32BitwiseAnd`        |
/// | `GenericBitwiseOr`        | `Int32BitwiseOr`         |
/// | `GenericBitwiseXor`       | `Int32BitwiseXor`        |
/// | `GenericShiftLeft`        | `Int32ShiftLeft`         |
/// | `GenericShiftRight`       | `Int32ShiftRight`        |
/// | `GenericShiftRightLogical`| `Int32ShiftRightLogical` |
/// | `GenericNegate`           | `Int32Negate`            |
/// | `GenericIncrement`        | `CheckedSmiIncrement`    |
/// | `GenericDecrement`        | `CheckedSmiDecrement`    |
pub fn narrow_types_from_feedback(graph: &mut MaglevGraph, fv: Option<&FeedbackVector>) {
    let Some(fv) = fv else { return };

    for block in graph.blocks_mut() {
        for (_id, node) in &mut block.nodes {
            let narrowed = narrow_node(node, fv);
            if let Some(replacement) = narrowed {
                *node = replacement;
            }
        }
    }
}

/// Attempt to narrow a single `Generic*` node using `fv`.
fn narrow_node(node: &ValueNode, fv: &FeedbackVector) -> Option<ValueNode> {
    match node {
        // ── Binary ops ────────────────────────────────────────────────────────
        ValueNode::GenericAdd {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::CheckedSmiAdd {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericSubtract {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::CheckedSmiSubtract {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericMultiply {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::CheckedSmiMultiply {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericDivide {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::CheckedSmiDivide {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericModulus {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::CheckedSmiModulus {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericBitwiseAnd {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32BitwiseAnd {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericBitwiseOr {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32BitwiseOr {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericBitwiseXor {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32BitwiseXor {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericShiftLeft {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32ShiftLeft {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericShiftRight {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32ShiftRight {
            left: *left,
            right: *right,
        }),
        ValueNode::GenericShiftRightLogical {
            left,
            right,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32ShiftRightLogical {
            left: *left,
            right: *right,
        }),

        // ── Unary ops ─────────────────────────────────────────────────────────
        ValueNode::GenericNegate {
            value,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => Some(ValueNode::Int32Negate { value: *value }),
        ValueNode::GenericIncrement {
            value,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => {
            Some(ValueNode::CheckedSmiIncrement { value: *value })
        }
        ValueNode::GenericDecrement {
            value,
            feedback_slot,
        } if is_monomorphic(fv, *feedback_slot) => {
            Some(ValueNode::CheckedSmiDecrement { value: *value })
        }

        _ => None,
    }
}

/// Return `true` iff the feedback slot at `slot` is
/// [`InlineCacheState::Monomorphic`].
#[inline]
fn is_monomorphic(fv: &FeedbackVector, slot: u32) -> bool {
    fv.get_state(slot) == Some(InlineCacheState::Monomorphic)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Hot call-site specialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Convert [`ValueNode::Call`] nodes to [`ValueNode::CallKnownFunction`] when
/// the call's [`FeedbackVector`] slot is [`InlineCacheState::Monomorphic`].
///
/// A monomorphic call site has seen only one callee object.  Replacing the
/// fully-generic [`ValueNode::Call`] (which must check and dispatch at
/// runtime) with [`ValueNode::CallKnownFunction`] removes the feedback-slot
/// overhead from the hot path and gives the code generator a signal that a
/// direct-call optimisation is applicable.
pub fn specialize_call_sites(graph: &mut MaglevGraph, fv: Option<&FeedbackVector>) {
    let Some(fv) = fv else { return };

    for block in graph.blocks_mut() {
        for (_id, node) in &mut block.nodes {
            if let ValueNode::Call {
                callee,
                receiver,
                args,
                feedback_slot,
            } = node
                && is_monomorphic(fv, *feedback_slot)
            {
                *node = ValueNode::CallKnownFunction {
                    callee: *callee,
                    receiver: *receiver,
                    args: args.clone(),
                };
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Load/store elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Eliminate redundant in-object field loads within each basic block.
///
/// For each basic block the pass maintains a *load-availability table* mapping
/// `(object NodeId, field offset) → NodeId of the first load`.  When the same
/// field is loaded a second time without an intervening store to the same
/// location, the second load's [`NodeId`] is remapped to the first load's
/// result and the second load node is replaced by a cheap
/// [`ValueNode::UndefinedConstant`] placeholder (which will be removed by a
/// subsequent dead-code-elimination pass if its result is unused).
///
/// Store nodes ([`ValueNode::StoreField`]) invalidate the cached result for
/// their `(object, offset)` pair.  This is a simple intra-block analysis; no
/// inter-block load forwarding is performed.
///
/// Covered load kinds: [`ValueNode::LoadField`], [`ValueNode::LoadTaggedField`],
/// [`ValueNode::LoadDoubleField`].
pub fn eliminate_redundant_loads(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        eliminate_redundant_loads_in_block(block);
    }
}

/// Apply load/store elimination to a single basic block.
fn eliminate_redundant_loads_in_block(block: &mut BasicBlock) {
    // Maps (object_id, field_offset) → NodeId of the first available load.
    let mut available: HashMap<(NodeId, u32), NodeId> = HashMap::new();
    // Substitutions to apply after the scan: redundant_id → canonical_id.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &mut block.nodes {
        match node {
            // ── Loads ─────────────────────────────────────────────────────────
            ValueNode::LoadField { object, offset }
            | ValueNode::LoadTaggedField { object, offset }
            | ValueNode::LoadDoubleField { object, offset } => {
                let key = (*object, *offset);
                if let Some(&first_id) = available.get(&key) {
                    // Redundant load — remap this id to the first load.
                    subst.insert(*id, first_id);
                    *node = ValueNode::UndefinedConstant;
                } else {
                    available.insert(key, *id);
                }
            }

            // ── Stores — invalidate the cached load for the written field ─────
            ValueNode::StoreField { object, offset, .. } => {
                available.remove(&(*object, *offset));
            }

            _ => {}
        }
    }

    if !subst.is_empty() {
        apply_subst_to_block(block, &subst);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4 — Escape analysis / allocation sinking
// ─────────────────────────────────────────────────────────────────────────────

/// Replace non-escaping allocations with [`ValueNode::VirtualObject`].
///
/// An allocation node ([`ValueNode::CreateEmptyObjectLiteral`]) is considered
/// *non-escaping* when its result is **never**:
/// - returned from the function (appears as the operand of a
///   [`crate::compiler::maglev::ir::ControlNode::Return`]),
/// - passed as an argument or receiver to a [`ValueNode::Call`] /
///   [`ValueNode::CallKnownFunction`] / [`ValueNode::Construct`] etc., or
/// - stored *as a value* into another object
///   ([`ValueNode::StoreField`] / [`ValueNode::StoreNamedGeneric`] /
///   [`ValueNode::StoreGlobal`] / [`ValueNode::StoreContextSlot`] etc.).
///
/// Non-escaping allocations need not be heap-allocated; replacing them with
/// `VirtualObject` signals this to the Cranelift code generator.
pub fn sink_non_escaping_allocations(graph: &mut MaglevGraph) {
    // Collect the set of all allocation NodeIds.
    let alloc_ids: HashSet<NodeId> = graph
        .blocks()
        .iter()
        .flat_map(|b| b.nodes.iter())
        .filter_map(|(id, node)| {
            if matches!(node, ValueNode::CreateEmptyObjectLiteral) {
                Some(*id)
            } else {
                None
            }
        })
        .collect();

    if alloc_ids.is_empty() {
        return;
    }

    // Collect the set of allocation IDs that escape.
    let mut escaping: HashSet<NodeId> = HashSet::new();
    collect_escaping_ids(graph, &alloc_ids, &mut escaping);

    // Replace non-escaping allocations with VirtualObject.
    for block in graph.blocks_mut() {
        for (id, node) in &mut block.nodes {
            if alloc_ids.contains(id) && !escaping.contains(id) {
                *node = ValueNode::VirtualObject { map: 0 };
            }
        }
    }
}

/// Populate `escaping` with every allocation ID from `candidates` that escapes
/// (is returned, passed to a call, or stored as a value).
fn collect_escaping_ids(
    graph: &MaglevGraph,
    candidates: &HashSet<NodeId>,
    escaping: &mut HashSet<NodeId>,
) {
    for block in graph.blocks() {
        for (_id, node) in &block.nodes {
            match node {
                // Storing *as a value* into another location is an escape.
                ValueNode::StoreField { value, .. }
                | ValueNode::StoreNamedGeneric { value, .. }
                | ValueNode::StoreGlobal { value, .. }
                | ValueNode::StoreCurrentContextSlot { value, .. }
                | ValueNode::StoreContextSlot { value, .. }
                | ValueNode::StoreFixedArrayElement { value, .. }
                | ValueNode::StoreFixedDoubleArrayElement { value, .. }
                | ValueNode::StoreKeyedGeneric { value, .. } => {
                    if candidates.contains(value) {
                        escaping.insert(*value);
                    }
                }

                // Passing as argument or receiver to any call is an escape.
                ValueNode::Call {
                    callee,
                    receiver,
                    args,
                    ..
                }
                | ValueNode::CallWithSpread {
                    callee,
                    receiver,
                    args,
                    ..
                }
                | ValueNode::CallKnownFunction {
                    callee,
                    receiver,
                    args,
                } => {
                    for id in std::iter::once(callee)
                        .chain(std::iter::once(receiver))
                        .chain(args.iter())
                    {
                        if candidates.contains(id) {
                            escaping.insert(*id);
                        }
                    }
                }
                ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
                    for id in args {
                        if candidates.contains(id) {
                            escaping.insert(*id);
                        }
                    }
                }
                ValueNode::Construct {
                    constructor, args, ..
                }
                | ValueNode::ConstructWithSpread {
                    constructor, args, ..
                } => {
                    if candidates.contains(constructor) {
                        escaping.insert(*constructor);
                    }
                    for id in args {
                        if candidates.contains(id) {
                            escaping.insert(*id);
                        }
                    }
                }

                _ => {}
            }
        }

        // Returning the allocation is also an escape.
        if let Some(crate::compiler::maglev::ir::ControlNode::Return { value }) = &block.control
            && candidates.contains(value)
        {
            escaping.insert(*value);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Substitution helper (mirrors the one in maglev/optimizer.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Rewrite every [`NodeId`] operand in `block` using `subst`.
fn apply_subst_to_block(block: &mut BasicBlock, subst: &HashMap<NodeId, NodeId>) {
    let resolve = |id: NodeId| *subst.get(&id).unwrap_or(&id);

    for (_id, node) in &mut block.nodes {
        apply_subst_to_value_node(node, &resolve);
    }
    if let Some(ctrl) = &mut block.control {
        apply_subst_to_control_node(ctrl, &resolve);
    }
}

/// Apply a substitution function to every [`NodeId`] operand in a
/// [`ValueNode`].
#[allow(clippy::too_many_lines)]
fn apply_subst_to_value_node(node: &mut ValueNode, resolve: &impl Fn(NodeId) -> NodeId) {
    use ValueNode::*;
    match node {
        // ── Zero-input nodes ──────────────────────────────────────────────────
        SmiConstant { .. }
        | Float64Constant { .. }
        | Int32Constant { .. }
        | Uint32Constant { .. }
        | BigIntConstant { .. }
        | TrueConstant
        | FalseConstant
        | NullConstant
        | UndefinedConstant
        | RootConstant { .. }
        | ExternalConstant { .. }
        | StringConstant { .. }
        | ConstantPoolEntry { .. }
        | Parameter { .. }
        | RegisterInput { .. }
        | ArgumentsLength
        | RestLength
        | LoadGlobal { .. }
        | LoadCurrentContextSlot { .. }
        | ArgumentsElements { .. }
        | RestElements { .. }
        | VirtualObject { .. }
        | CreateFunctionContext { .. }
        | CreateBlockContext { .. }
        | CreateShallowObjectLiteral { .. }
        | CreateShallowArrayLiteral { .. }
        | CreateObjectLiteral { .. }
        | CreateArrayLiteral { .. }
        | CreateEmptyObjectLiteral
        | CreateRegExpLiteral { .. }
        | CreateClosure { .. }
        | FastCreateClosure { .. }
        | GetTemplateObject { .. }
        | Debugger
        | Abort { .. } => {}

        GetArgument { index } => *index = resolve(*index),

        // ── Single-input nodes ────────────────────────────────────────────────
        CheckedSmiIncrement { value }
        | CheckedSmiDecrement { value }
        | Int32Negate { value }
        | Int32Increment { value }
        | Int32Decrement { value }
        | Float64Negate { value }
        | Float64Ieee754Unary { value, .. }
        | GenericBitwiseNot { value, .. }
        | GenericNegate { value, .. }
        | GenericIncrement { value, .. }
        | GenericDecrement { value, .. }
        | ChangeInt32ToFloat64 { input: value }
        | ChangeUint32ToFloat64 { input: value }
        | ChangeFloat64ToInt32 { input: value }
        | CheckedFloat64ToInt32 { input: value }
        | ChangeInt32ToTagged { input: value }
        | ChangeUint32ToTagged { input: value }
        | ChangeFloat64ToTagged { input: value }
        | ChangeTaggedToInt32 { input: value }
        | ChangeTaggedToUint32 { input: value }
        | ChangeTaggedToFloat64 { input: value }
        | CheckedTaggedToInt32 { input: value }
        | CheckedTaggedToFloat64 { input: value }
        | ToBoolean { value }
        | ToString { value, .. }
        | ToObject { value, .. }
        | ToName { value, .. }
        | ToNumber { value, .. }
        | ToNumberOrNumeric { value, .. }
        | TypeOf { value }
        | NumberToString { value, .. }
        | TestUndetectable { value }
        | TestTypeOf { value, .. } => *value = resolve(*value),

        CheckSmi { receiver }
        | CheckNumber { receiver }
        | CheckHeapObject { receiver }
        | CheckSymbol { receiver }
        | CheckString { receiver }
        | CheckStringOrStringWrapper { receiver }
        | CheckSeqOneByteString { receiver }
        | CheckMaps { receiver, .. }
        | CheckMapsWithMigration { receiver, .. }
        | CheckValue { receiver, .. } => *receiver = resolve(*receiver),

        CheckDynamicValue { receiver, expected } => {
            *receiver = resolve(*receiver);
            *expected = resolve(*expected);
        }

        CheckInt32IsSmi { input }
        | CheckUint32IsSmi { input }
        | CheckHoleyFloat64IsSmi { input }
        | CheckFloat64IsNan { input } => *input = resolve(*input),

        LoadField { object, .. }
        | LoadTaggedField { object, .. }
        | LoadDoubleField { object, .. }
        | LoadNamedGeneric { object, .. }
        | ForInPrepare {
            enumerator: object, ..
        }
        | StringLength { string: object } => *object = resolve(*object),

        LoadEnumCacheLength { map } => *map = resolve(*map),

        LoadKeyedGeneric { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        HasInPrototypeChain { object, prototype } => {
            *object = resolve(*object);
            *prototype = resolve(*prototype);
        }

        StoreField { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        StoreCurrentContextSlot { value, .. } | StoreGlobal { value, .. } => {
            *value = resolve(*value)
        }

        LoadContextSlot { context, .. } => *context = resolve(*context),
        StoreContextSlot { context, value, .. } => {
            *context = resolve(*context);
            *value = resolve(*value);
        }

        LoadFixedArrayElement { elements, index }
        | LoadFixedDoubleArrayElement { elements, index }
        | LoadHoleyFixedDoubleArrayElement { elements, index } => {
            *elements = resolve(*elements);
            *index = resolve(*index);
        }

        StoreFixedArrayElement {
            elements,
            index,
            value,
        }
        | StoreFixedDoubleArrayElement {
            elements,
            index,
            value,
        } => {
            *elements = resolve(*elements);
            *index = resolve(*index);
            *value = resolve(*value);
        }

        StoreNamedGeneric { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        StoreKeyedGeneric {
            object, key, value, ..
        } => {
            *object = resolve(*object);
            *key = resolve(*key);
            *value = resolve(*value);
        }

        // ── Binary nodes ──────────────────────────────────────────────────────
        CheckedSmiAdd { left, right }
        | CheckedSmiSubtract { left, right }
        | CheckedSmiMultiply { left, right }
        | CheckedSmiDivide { left, right }
        | CheckedSmiModulus { left, right }
        | Int32Add { left, right }
        | Int32Subtract { left, right }
        | Int32Multiply { left, right }
        | Int32Divide { left, right }
        | Int32Modulus { left, right }
        | Int32BitwiseAnd { left, right }
        | Int32BitwiseOr { left, right }
        | Int32BitwiseXor { left, right }
        | Int32ShiftLeft { left, right }
        | Int32ShiftRight { left, right }
        | Int32ShiftRightLogical { left, right }
        | Uint32Add { left, right }
        | Uint32Subtract { left, right }
        | Uint32Multiply { left, right }
        | Uint32Divide { left, right }
        | Uint32Modulus { left, right }
        | Float64Add { left, right }
        | Float64Subtract { left, right }
        | Float64Multiply { left, right }
        | Float64Divide { left, right }
        | Float64Modulus { left, right }
        | Float64Exponentiate { left, right }
        | Int32Equal { left, right }
        | Int32StrictEqual { left, right }
        | Int32LessThan { left, right }
        | Int32LessThanOrEqual { left, right }
        | Int32GreaterThan { left, right }
        | Int32GreaterThanOrEqual { left, right }
        | Float64Equal { left, right }
        | Float64LessThan { left, right }
        | Float64LessThanOrEqual { left, right }
        | Float64GreaterThan { left, right }
        | Float64GreaterThanOrEqual { left, right }
        | StringConcat { left, right }
        | StringEqual { left, right }
        | GenericAdd { left, right, .. }
        | GenericSubtract { left, right, .. }
        | GenericMultiply { left, right, .. }
        | GenericDivide { left, right, .. }
        | GenericModulus { left, right, .. }
        | GenericExponentiate { left, right, .. }
        | GenericBitwiseAnd { left, right, .. }
        | GenericBitwiseOr { left, right, .. }
        | GenericBitwiseXor { left, right, .. }
        | GenericShiftLeft { left, right, .. }
        | GenericShiftRight { left, right, .. }
        | GenericShiftRightLogical { left, right, .. }
        | TaggedEqual { left, right, .. }
        | TaggedNotEqual { left, right, .. } => {
            *left = resolve(*left);
            *right = resolve(*right);
        }

        CheckInt32Condition { left, right, .. } => {
            *left = resolve(*left);
            *right = resolve(*right);
        }

        CheckCacheIndicesNotCleared { receiver, indices } => {
            *receiver = resolve(*receiver);
            *indices = resolve(*indices);
        }

        TestInstanceOf {
            object, callable, ..
        } => {
            *object = resolve(*object);
            *callable = resolve(*callable);
        }
        TestIn { key, object, .. } => {
            *key = resolve(*key);
            *object = resolve(*object);
        }

        StringAt { string, index } => {
            *string = resolve(*string);
            *index = resolve(*index);
        }

        ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            *receiver = resolve(*receiver);
            *cache_index = resolve(*cache_index);
            *cache_array = resolve(*cache_array);
        }

        DeleteProperty { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        CreateCatchContext { exception, .. } => *exception = resolve(*exception),
        CreateWithContext { object, .. } => *object = resolve(*object),

        Call {
            callee,
            receiver,
            args,
            ..
        }
        | CallWithSpread {
            callee,
            receiver,
            args,
            ..
        }
        | CallKnownFunction {
            callee,
            receiver,
            args,
        } => {
            *callee = resolve(*callee);
            *receiver = resolve(*receiver);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        CallBuiltin { args, .. } | CallRuntime { args, .. } => {
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        Construct {
            constructor, args, ..
        }
        | ConstructWithSpread {
            constructor, args, ..
        } => {
            *constructor = resolve(*constructor);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        Phi { inputs } => {
            for inp in inputs.iter_mut() {
                *inp = resolve(*inp);
            }
        }
    }
}

/// Apply a substitution function to every [`NodeId`] operand in a
/// [`crate::compiler::maglev::ir::ControlNode`].
fn apply_subst_to_control_node(
    ctrl: &mut crate::compiler::maglev::ir::ControlNode,
    resolve: &impl Fn(NodeId) -> NodeId,
) {
    use crate::compiler::maglev::ir::ControlNode;
    match ctrl {
        ControlNode::Return { value } => *value = resolve(*value),
        ControlNode::Branch { condition, .. } => *condition = resolve(*condition),
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::feedback::{
        FeedbackMetadata, FeedbackSlotKind, FeedbackVector, InlineCacheState,
    };
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a one-slot Monomorphic FeedbackVector.
    fn mono_fv(kind: FeedbackSlotKind) -> FeedbackVector {
        let metadata = FeedbackMetadata::new(vec![kind]);
        let mut fv = FeedbackVector::new(&metadata);
        fv.set_state(0, InlineCacheState::Monomorphic);
        fv
    }

    /// Build a one-slot Megamorphic FeedbackVector.
    fn mega_fv(kind: FeedbackSlotKind) -> FeedbackVector {
        let metadata = FeedbackMetadata::new(vec![kind]);
        let mut fv = FeedbackVector::new(&metadata);
        fv.set_state(0, InlineCacheState::Megamorphic);
        fv
    }

    // ── Pass 1: type narrowing ────────────────────────────────────────────────

    #[test]
    fn test_narrow_generic_add_monomorphic() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 3 });
        let b = block.push_value(ValueNode::SmiConstant { value: 4 });
        let add = block.push_value(ValueNode::GenericAdd {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let fv = mono_fv(FeedbackSlotKind::BinaryOp);
        narrow_types_from_feedback(&mut graph, Some(&fv));

        let node = &graph.blocks()[0].nodes[2].1;
        assert!(
            matches!(node, ValueNode::CheckedSmiAdd { .. }),
            "expected CheckedSmiAdd, got {node:?}"
        );
    }

    #[test]
    fn test_narrow_generic_add_megamorphic_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 1 });
        let b = block.push_value(ValueNode::SmiConstant { value: 2 });
        let add = block.push_value(ValueNode::GenericAdd {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let fv = mega_fv(FeedbackSlotKind::BinaryOp);
        narrow_types_from_feedback(&mut graph, Some(&fv));

        // Megamorphic — should NOT be narrowed.
        let node = &graph.blocks()[0].nodes[2].1;
        assert!(
            matches!(node, ValueNode::GenericAdd { .. }),
            "expected unchanged GenericAdd, got {node:?}"
        );
    }

    #[test]
    fn test_narrow_no_feedback_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 1 });
        let b = block.push_value(ValueNode::SmiConstant { value: 2 });
        let add = block.push_value(ValueNode::GenericAdd {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        // No feedback vector → no narrowing.
        narrow_types_from_feedback(&mut graph, None);

        let node = &graph.blocks()[0].nodes[2].1;
        assert!(matches!(node, ValueNode::GenericAdd { .. }));
    }

    #[test]
    fn test_narrow_subtract_monomorphic() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 10 });
        let b = block.push_value(ValueNode::SmiConstant { value: 3 });
        let sub = block.push_value(ValueNode::GenericSubtract {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: sub });
        graph.add_block(block);

        let fv = mono_fv(FeedbackSlotKind::BinaryOp);
        narrow_types_from_feedback(&mut graph, Some(&fv));

        let node = &graph.blocks()[0].nodes[2].1;
        assert!(matches!(node, ValueNode::CheckedSmiSubtract { .. }));
    }

    #[test]
    fn test_narrow_multiply_monomorphic() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 6 });
        let b = block.push_value(ValueNode::SmiConstant { value: 7 });
        let mul = block.push_value(ValueNode::GenericMultiply {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        let fv = mono_fv(FeedbackSlotKind::BinaryOp);
        narrow_types_from_feedback(&mut graph, Some(&fv));

        let node = &graph.blocks()[0].nodes[2].1;
        assert!(matches!(node, ValueNode::CheckedSmiMultiply { .. }));
    }

    #[test]
    fn test_narrow_bitwise_ops_monomorphic() {
        for (input_node, expected) in [
            (
                ValueNode::GenericBitwiseAnd {
                    left: NodeId(0),
                    right: NodeId(1),
                    feedback_slot: 0,
                },
                "Int32BitwiseAnd",
            ),
            (
                ValueNode::GenericBitwiseOr {
                    left: NodeId(0),
                    right: NodeId(1),
                    feedback_slot: 0,
                },
                "Int32BitwiseOr",
            ),
            (
                ValueNode::GenericBitwiseXor {
                    left: NodeId(0),
                    right: NodeId(1),
                    feedback_slot: 0,
                },
                "Int32BitwiseXor",
            ),
        ] {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let _a = block.push_value(ValueNode::SmiConstant { value: 0 });
            let _b = block.push_value(ValueNode::SmiConstant { value: 0 });
            // input_node uses NodeId(0) and NodeId(1) which match _a and _b above.
            let res = block.push_value(input_node);
            block.set_control(ControlNode::Return { value: res });
            graph.add_block(block);

            let fv = mono_fv(FeedbackSlotKind::BinaryOp);
            narrow_types_from_feedback(&mut graph, Some(&fv));

            let node = &graph.blocks()[0].nodes[2].1;
            match expected {
                "Int32BitwiseAnd" => assert!(matches!(node, ValueNode::Int32BitwiseAnd { .. })),
                "Int32BitwiseOr" => assert!(matches!(node, ValueNode::Int32BitwiseOr { .. })),
                "Int32BitwiseXor" => assert!(matches!(node, ValueNode::Int32BitwiseXor { .. })),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_narrow_unary_ops_monomorphic() {
        // GenericNegate → Int32Negate
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 5 });
            let neg = block.push_value(ValueNode::GenericNegate {
                value: a,
                feedback_slot: 0,
            });
            block.set_control(ControlNode::Return { value: neg });
            graph.add_block(block);

            let fv = mono_fv(FeedbackSlotKind::UnaryOp);
            narrow_types_from_feedback(&mut graph, Some(&fv));
            assert!(matches!(
                &graph.blocks()[0].nodes[1].1,
                ValueNode::Int32Negate { .. }
            ));
        }

        // GenericIncrement → CheckedSmiIncrement
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 5 });
            let inc = block.push_value(ValueNode::GenericIncrement {
                value: a,
                feedback_slot: 0,
            });
            block.set_control(ControlNode::Return { value: inc });
            graph.add_block(block);

            let fv = mono_fv(FeedbackSlotKind::BinaryOpInc);
            narrow_types_from_feedback(&mut graph, Some(&fv));
            assert!(matches!(
                &graph.blocks()[0].nodes[1].1,
                ValueNode::CheckedSmiIncrement { .. }
            ));
        }

        // GenericDecrement → CheckedSmiDecrement
        {
            let mut graph = MaglevGraph::new(0);
            let mut block = BasicBlock::new(0);
            let a = block.push_value(ValueNode::SmiConstant { value: 5 });
            let dec = block.push_value(ValueNode::GenericDecrement {
                value: a,
                feedback_slot: 0,
            });
            block.set_control(ControlNode::Return { value: dec });
            graph.add_block(block);

            let fv = mono_fv(FeedbackSlotKind::BinaryOpInc);
            narrow_types_from_feedback(&mut graph, Some(&fv));
            assert!(matches!(
                &graph.blocks()[0].nodes[1].1,
                ValueNode::CheckedSmiDecrement { .. }
            ));
        }
    }

    // ── Pass 2: call-site specialisation ─────────────────────────────────────

    #[test]
    fn test_specialize_call_monomorphic() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let callee = block.push_value(ValueNode::UndefinedConstant);
        let recv = block.push_value(ValueNode::UndefinedConstant);
        let call = block.push_value(ValueNode::Call {
            callee,
            receiver: recv,
            args: vec![],
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: call });
        graph.add_block(block);

        let fv = mono_fv(FeedbackSlotKind::Call);
        specialize_call_sites(&mut graph, Some(&fv));

        let node = &graph.blocks()[0].nodes[2].1;
        assert!(
            matches!(node, ValueNode::CallKnownFunction { .. }),
            "expected CallKnownFunction, got {node:?}"
        );
    }

    #[test]
    fn test_specialize_call_megamorphic_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let callee = block.push_value(ValueNode::UndefinedConstant);
        let recv = block.push_value(ValueNode::UndefinedConstant);
        let call = block.push_value(ValueNode::Call {
            callee,
            receiver: recv,
            args: vec![],
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: call });
        graph.add_block(block);

        let fv = mega_fv(FeedbackSlotKind::Call);
        specialize_call_sites(&mut graph, Some(&fv));

        // Megamorphic → unchanged.
        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Call { .. }
        ));
    }

    #[test]
    fn test_specialize_call_no_feedback_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let callee = block.push_value(ValueNode::UndefinedConstant);
        let recv = block.push_value(ValueNode::UndefinedConstant);
        let call = block.push_value(ValueNode::Call {
            callee,
            receiver: recv,
            args: vec![],
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: call });
        graph.add_block(block);

        specialize_call_sites(&mut graph, None);

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::Call { .. }
        ));
    }

    // ── Pass 3: load/store elimination ───────────────────────────────────────

    #[test]
    fn test_eliminate_duplicate_load_field() {
        // build: obj param, load1 = obj.f@8, load2 = obj.f@8, return load2
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let load1 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        let load2 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: load2 });
        graph.add_block(block);

        eliminate_redundant_loads(&mut graph);

        // load2's node should now be UndefinedConstant (replaced)
        assert!(
            matches!(&graph.blocks()[0].nodes[2].1, ValueNode::UndefinedConstant),
            "expected load2 to become UndefinedConstant"
        );
        // The Return should point to load1's NodeId (substituted)
        assert_eq!(
            graph.blocks()[0].control,
            Some(ControlNode::Return { value: load1 })
        );
    }

    #[test]
    fn test_no_elimination_after_store() {
        // load1 = obj.f@8, store obj.f@8 := v, load2 = obj.f@8
        // load2 must NOT be eliminated.
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::Int32Constant { value: 99 });
        let load1 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        let _store = block.push_value(ValueNode::StoreField {
            object: obj,
            offset: 8,
            value: val,
        });
        let load2 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: load2 });
        graph.add_block(block);

        eliminate_redundant_loads(&mut graph);

        // load2 must remain a LoadField.
        assert!(
            matches!(&graph.blocks()[0].nodes[4].1, ValueNode::LoadField { .. }),
            "load2 should NOT be eliminated after a store"
        );
        let _ = load1;
    }

    #[test]
    fn test_eliminate_different_offset_unchanged() {
        // load obj.f@8, then load obj.f@16 — different offsets, no elimination.
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let _l1 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 8,
        });
        let l2 = block.push_value(ValueNode::LoadField {
            object: obj,
            offset: 16,
        });
        block.set_control(ControlNode::Return { value: l2 });
        graph.add_block(block);

        eliminate_redundant_loads(&mut graph);

        assert!(
            matches!(&graph.blocks()[0].nodes[2].1, ValueNode::LoadField { .. }),
            "loads to different offsets should NOT be eliminated"
        );
    }

    // ── Pass 4: escape analysis ───────────────────────────────────────────────

    #[test]
    fn test_non_escaping_alloc_becomes_virtual() {
        // CreateEmptyObjectLiteral, store into it, return undefined.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let alloc = block.push_value(ValueNode::CreateEmptyObjectLiteral);
        let val = block.push_value(ValueNode::Int32Constant { value: 1 });
        // Store *into* the allocation (object = alloc) — this is not an escape.
        let _s = block.push_value(ValueNode::StoreField {
            object: alloc,
            offset: 8,
            value: val,
        });
        let undef = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: undef });
        graph.add_block(block);

        sink_non_escaping_allocations(&mut graph);

        assert!(
            matches!(
                &graph.blocks()[0].nodes[0].1,
                ValueNode::VirtualObject { .. }
            ),
            "non-escaping alloc should become VirtualObject"
        );
    }

    #[test]
    fn test_escaping_alloc_via_return_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let alloc = block.push_value(ValueNode::CreateEmptyObjectLiteral);
        block.set_control(ControlNode::Return { value: alloc });
        graph.add_block(block);

        sink_non_escaping_allocations(&mut graph);

        // Returned — must stay CreateEmptyObjectLiteral.
        assert!(
            matches!(
                &graph.blocks()[0].nodes[0].1,
                ValueNode::CreateEmptyObjectLiteral
            ),
            "escaping alloc must NOT become VirtualObject"
        );
    }

    #[test]
    fn test_escaping_alloc_via_call_arg_unchanged() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let alloc = block.push_value(ValueNode::CreateEmptyObjectLiteral);
        let callee = block.push_value(ValueNode::UndefinedConstant);
        let recv = block.push_value(ValueNode::UndefinedConstant);
        // alloc is passed as an argument to a Call → escapes.
        let call = block.push_value(ValueNode::Call {
            callee,
            receiver: recv,
            args: vec![alloc],
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: call });
        graph.add_block(block);

        sink_non_escaping_allocations(&mut graph);

        assert!(
            matches!(
                &graph.blocks()[0].nodes[0].1,
                ValueNode::CreateEmptyObjectLiteral
            ),
            "alloc passed as arg must NOT become VirtualObject"
        );
    }

    #[test]
    fn test_escaping_alloc_via_store_value_unchanged() {
        // Store the alloc *as a value* into another object → escapes.
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let alloc = block.push_value(ValueNode::CreateEmptyObjectLiteral);
        let other_obj = block.push_value(ValueNode::Parameter { index: 0 });
        let _s = block.push_value(ValueNode::StoreField {
            object: other_obj,
            offset: 8,
            value: alloc, // alloc stored as value → escapes
        });
        let undef = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: undef });
        graph.add_block(block);

        sink_non_escaping_allocations(&mut graph);

        assert!(
            matches!(
                &graph.blocks()[0].nodes[0].1,
                ValueNode::CreateEmptyObjectLiteral
            ),
            "alloc stored as value must NOT become VirtualObject"
        );
    }

    // ── run_pre_clif_passes integration ──────────────────────────────────────

    #[test]
    fn test_run_pre_clif_passes_no_feedback() {
        // Smoke test: running with no FV on a simple graph should not panic.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::Int32Constant { value: 42 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        run_pre_clif_passes(&mut graph, None);

        assert_eq!(graph.blocks()[0].nodes.len(), 1);
    }

    #[test]
    fn test_run_pre_clif_passes_with_feedback_narrows_generic_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 3 });
        let b = block.push_value(ValueNode::SmiConstant { value: 4 });
        let add = block.push_value(ValueNode::GenericAdd {
            left: a,
            right: b,
            feedback_slot: 0,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let fv = mono_fv(FeedbackSlotKind::BinaryOp);
        run_pre_clif_passes(&mut graph, Some(&fv));

        assert!(matches!(
            &graph.blocks()[0].nodes[2].1,
            ValueNode::CheckedSmiAdd { .. }
        ));
    }
}

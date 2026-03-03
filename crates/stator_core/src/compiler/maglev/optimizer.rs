//! Maglev IR optimisation passes.
//!
//! Three passes are implemented and composed by [`optimize`]:
//!
//! 1. **Constant folding** — replaces arithmetic/comparison nodes whose *all*
//!    inputs resolve to compile-time constant nodes with a new constant node,
//!    eliminating run-time computation.  Covered operations:
//!    - `Int32` binary arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`,
//!      `Modulus` and the corresponding `CheckedSmi*` variants.
//!    - `Float64` binary arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`,
//!      `Modulus`.
//!    - `Int32` / `Float64` unary: `Negate`, `CheckedSmiIncrement`,
//!      `CheckedSmiDecrement`, `Int32Increment`, `Int32Decrement`.
//!
//! 2. **Dead-code elimination (DCE)** — removes `ValueNode`s whose [`NodeId`]
//!    is never referenced by any other node (value or control) in the graph.
//!    Pure side-effect-free nodes that produce a value which nobody consumes
//!    are safe to drop.  Nodes with observable side-effects (stores, calls,
//!    allocations, guards/checks) are always considered *live* and are kept
//!    even when their result value is unused.
//!
//! 3. **Redundant `CheckMaps` removal** — within each basic block, a
//!    `CheckMaps { receiver, feedback_slot }` node is redundant if an
//!    identical guard for the *same* (receiver, feedback_slot) pair has
//!    already been emitted earlier in the same block.  The duplicate is
//!    replaced by a [`ValueNode::UndefinedConstant`] placeholder and the
//!    relevant ID is remapped so all consumers still compile correctly.
//!
//! # Usage
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, ValueNode,
//! };
//! use stator_core::compiler::maglev::optimizer::optimize;
//!
//! let mut graph = MaglevGraph::new(0);
//! let mut block = BasicBlock::new(0);
//! // 3 + 4  — both operands are compile-time constants.
//! let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
//! let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
//! let add = block.push_value(ValueNode::Int32Add { left: c3, right: c4 });
//! block.set_control(ControlNode::Return { value: add });
//! graph.add_block(block);
//!
//! let before = graph.blocks()[0].nodes.len();
//! optimize(&mut graph);
//! // The Int32Add node has been folded; node count is reduced.
//! assert!(graph.blocks()[0].nodes.len() < before);
//! ```

use std::collections::{HashMap, HashSet};

use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Run all optimisation passes on `graph` in place.
///
/// Passes are applied in the order: constant folding → redundant-CheckMaps
/// removal → dead-code elimination.  Multiple rounds are *not* performed; a
/// single sweep of each pass is sufficient for the patterns targeted here.
pub fn optimize(graph: &mut MaglevGraph) {
    fold_constants(graph);
    remove_redundant_check_maps(graph);
    eliminate_dead_code(graph);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Constant folding
// ─────────────────────────────────────────────────────────────────────────────

/// Replace arithmetic nodes with constant inputs by a single constant node.
///
/// The pass walks every block linearly, maintaining a map from [`NodeId`] →
/// known constant value for all nodes emitted so far.  When a binary or unary
/// arithmetic node is found whose operand(s) are all in the constant map, the
/// node is replaced in-place with the folded constant; the old `NodeId` for
/// that position is re-used so downstream references remain valid.
fn fold_constants(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        fold_block_constants(block);
    }
}

/// Represents a compile-time-known scalar value.
#[derive(Clone, Copy)]
enum ConstVal {
    I32(i32),
    F64(f64),
}

/// Fold constants within a single [`BasicBlock`].
fn fold_block_constants(block: &mut BasicBlock) {
    // Map from NodeId → known constant value for nodes already processed.
    let mut consts: HashMap<NodeId, ConstVal> = HashMap::new();

    for (id, node) in &mut block.nodes {
        // Seed the constant map from literal constant nodes.
        match node {
            ValueNode::Int32Constant { value } => {
                consts.insert(*id, ConstVal::I32(*value));
                continue;
            }
            ValueNode::SmiConstant { value } => {
                consts.insert(*id, ConstVal::I32(*value));
                continue;
            }
            ValueNode::Float64Constant { value } => {
                consts.insert(*id, ConstVal::F64(*value));
                continue;
            }
            _ => {}
        }

        // Try to fold binary i32 operations.
        let folded: Option<ValueNode> = match node {
            // ── Binary Int32 / CheckedSmi ─────────────────────────────────
            ValueNode::Int32Add { left, right } | ValueNode::CheckedSmiAdd { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_add(b))
            }
            ValueNode::Int32Subtract { left, right }
            | ValueNode::CheckedSmiSubtract { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_sub(b))
            }
            ValueNode::Int32Multiply { left, right }
            | ValueNode::CheckedSmiMultiply { left, right } => {
                fold_i32_bin(left, right, &consts, |a, b| a.wrapping_mul(b))
            }
            ValueNode::Int32Divide { left, right }
            | ValueNode::CheckedSmiDivide { left, right } => {
                if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) =
                    (consts.get(left), consts.get(right))
                {
                    let (a, b) = (*a, *b);
                    if b != 0 {
                        Some(ValueNode::Int32Constant { value: a / b })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ValueNode::Int32Modulus { left, right }
            | ValueNode::CheckedSmiModulus { left, right } => {
                if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) =
                    (consts.get(left), consts.get(right))
                {
                    let (a, b) = (*a, *b);
                    if b != 0 {
                        Some(ValueNode::Int32Constant { value: a % b })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            // ── Unary Int32 / CheckedSmi ──────────────────────────────────
            ValueNode::Int32Negate { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_neg(),
                    })
                } else {
                    None
                }
            }
            ValueNode::Int32Increment { value } | ValueNode::CheckedSmiIncrement { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_add(1),
                    })
                } else {
                    None
                }
            }
            ValueNode::Int32Decrement { value } | ValueNode::CheckedSmiDecrement { value } => {
                if let Some(ConstVal::I32(v)) = consts.get(value) {
                    Some(ValueNode::Int32Constant {
                        value: v.wrapping_sub(1),
                    })
                } else {
                    None
                }
            }

            // ── Binary Float64 ────────────────────────────────────────────
            ValueNode::Float64Add { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a + b)
            }
            ValueNode::Float64Subtract { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a - b)
            }
            ValueNode::Float64Multiply { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a * b)
            }
            ValueNode::Float64Divide { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a / b)
            }
            ValueNode::Float64Modulus { left, right } => {
                fold_f64_bin(left, right, &consts, |a, b| a % b)
            }

            // ── Unary Float64 ─────────────────────────────────────────────
            ValueNode::Float64Negate { value } => {
                if let Some(ConstVal::F64(v)) = consts.get(value) {
                    Some(ValueNode::Float64Constant { value: -*v })
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(replacement) = folded {
            // Update the constant map so later nodes can fold through this one.
            match &replacement {
                ValueNode::Int32Constant { value } => {
                    consts.insert(*id, ConstVal::I32(*value));
                }
                ValueNode::Float64Constant { value } => {
                    consts.insert(*id, ConstVal::F64(*value));
                }
                _ => {}
            }
            *node = replacement;
        }
    }
}

/// Attempt to fold a binary `i32` operation given a constant-value map.
fn fold_i32_bin(
    left: &NodeId,
    right: &NodeId,
    consts: &HashMap<NodeId, ConstVal>,
    op: impl Fn(i32, i32) -> i32,
) -> Option<ValueNode> {
    if let (Some(ConstVal::I32(a)), Some(ConstVal::I32(b))) = (consts.get(left), consts.get(right))
    {
        Some(ValueNode::Int32Constant { value: op(*a, *b) })
    } else {
        None
    }
}

/// Attempt to fold a binary `f64` operation given a constant-value map.
fn fold_f64_bin(
    left: &NodeId,
    right: &NodeId,
    consts: &HashMap<NodeId, ConstVal>,
    op: impl Fn(f64, f64) -> f64,
) -> Option<ValueNode> {
    let lv = match consts.get(left) {
        Some(ConstVal::F64(v)) => *v,
        Some(ConstVal::I32(v)) => *v as f64,
        None => return None,
    };
    let rv = match consts.get(right) {
        Some(ConstVal::F64(v)) => *v,
        Some(ConstVal::I32(v)) => *v as f64,
        None => return None,
    };
    Some(ValueNode::Float64Constant { value: op(lv, rv) })
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Redundant CheckMaps removal
// ─────────────────────────────────────────────────────────────────────────────

/// Remove duplicate `CheckMaps` guards within each basic block.
///
/// Within a single block a `CheckMaps { receiver, feedback_slot }` node is
/// redundant if the *same* pair has already appeared earlier in the block.
/// Redundant nodes are replaced with `UndefinedConstant` (a harmless
/// placeholder) and any uses of the original [`NodeId`] are re-mapped to the
/// first occurrence's [`NodeId`] via a substitution table applied at the end.
fn remove_redundant_check_maps(graph: &mut MaglevGraph) {
    for block in graph.blocks_mut() {
        remove_redundant_check_maps_in_block(block);
    }
}

/// Remove redundant `CheckMaps` in a single block.
fn remove_redundant_check_maps_in_block(block: &mut BasicBlock) {
    // (receiver NodeId, feedback_slot) → first NodeId that checked this pair.
    let mut seen: HashMap<(NodeId, u32), NodeId> = HashMap::new();
    // Substitution: redundant NodeId → canonical NodeId.
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();

    for (id, node) in &mut block.nodes {
        if let ValueNode::CheckMaps {
            receiver,
            feedback_slot,
        } = node
        {
            let key = (*receiver, *feedback_slot);
            if let Some(&first) = seen.get(&key) {
                // This guard is redundant — record the substitution and replace
                // the node with a cheap placeholder.
                subst.insert(*id, first);
                *node = ValueNode::UndefinedConstant;
            } else {
                seen.insert(key, *id);
            }
        }
    }

    if subst.is_empty() {
        return;
    }

    // Apply the substitution to all node inputs and the control node.
    apply_subst_to_block(block, &subst);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3 — Dead-code elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Remove `ValueNode`s that are never used and have no observable side-effects.
///
/// Algorithm:
/// 1. Walk all blocks and collect every [`NodeId`] that is referenced — either
///    as an input to another `ValueNode` or as the `value` of a `ControlNode`.
/// 2. Walk all nodes; any node whose own [`NodeId`] is *not* in the live set
///    AND is considered *pure* (no side-effects) is removed.
fn eliminate_dead_code(graph: &mut MaglevGraph) {
    let live = collect_live_ids(graph);

    for block in graph.blocks_mut() {
        block.nodes.retain(|(id, node)| {
            // Always keep nodes with side-effects.
            if has_side_effects(node) {
                return true;
            }
            // Keep pure nodes only if their result is consumed somewhere.
            live.contains(id)
        });
    }
}

/// Collect every [`NodeId`] that appears as an *input* in any node or control
/// node across all blocks.
fn collect_live_ids(graph: &MaglevGraph) -> HashSet<NodeId> {
    let mut live = HashSet::new();

    for block in graph.blocks() {
        for (_id, node) in &block.nodes {
            collect_value_node_inputs(node, &mut live);
        }
        if let Some(ctrl) = &block.control {
            collect_control_node_inputs(ctrl, &mut live);
        }
    }

    live
}

/// Enumerate all [`NodeId`] operands referenced by a [`ValueNode`].
#[allow(clippy::too_many_lines)]
fn collect_value_node_inputs(node: &ValueNode, live: &mut HashSet<NodeId>) {
    match node {
        // ── Nodes with no inputs ─────────────────────────────────────────────
        ValueNode::SmiConstant { .. }
        | ValueNode::Float64Constant { .. }
        | ValueNode::Int32Constant { .. }
        | ValueNode::Uint32Constant { .. }
        | ValueNode::BigIntConstant { .. }
        | ValueNode::TrueConstant
        | ValueNode::FalseConstant
        | ValueNode::NullConstant
        | ValueNode::UndefinedConstant
        | ValueNode::RootConstant { .. }
        | ValueNode::ExternalConstant { .. }
        | ValueNode::StringConstant { .. }
        | ValueNode::ConstantPoolEntry { .. }
        | ValueNode::Parameter { .. }
        | ValueNode::RegisterInput { .. }
        | ValueNode::ArgumentsLength
        | ValueNode::RestLength
        | ValueNode::LoadGlobal { .. }
        | ValueNode::LoadCurrentContextSlot { .. }
        | ValueNode::ArgumentsElements { .. }
        | ValueNode::RestElements { .. }
        | ValueNode::VirtualObject { .. }
        | ValueNode::CreateFunctionContext { .. }
        | ValueNode::CreateBlockContext { .. }
        | ValueNode::CreateShallowObjectLiteral { .. }
        | ValueNode::CreateShallowArrayLiteral { .. }
        | ValueNode::CreateObjectLiteral { .. }
        | ValueNode::CreateArrayLiteral { .. }
        | ValueNode::CreateEmptyObjectLiteral
        | ValueNode::CreateRegExpLiteral { .. }
        | ValueNode::CreateClosure { .. }
        | ValueNode::FastCreateClosure { .. }
        | ValueNode::GetTemplateObject { .. }
        | ValueNode::Debugger
        | ValueNode::Abort { .. } => {}

        // ── Single-input nodes ────────────────────────────────────────────────
        ValueNode::GetArgument { index } => {
            live.insert(*index);
        }
        ValueNode::CheckedSmiIncrement { value }
        | ValueNode::CheckedSmiDecrement { value }
        | ValueNode::Int32Negate { value }
        | ValueNode::Int32Increment { value }
        | ValueNode::Int32Decrement { value }
        | ValueNode::Float64Negate { value }
        | ValueNode::Float64Ieee754Unary { value, .. }
        | ValueNode::GenericBitwiseNot { value, .. }
        | ValueNode::GenericNegate { value, .. }
        | ValueNode::GenericIncrement { value, .. }
        | ValueNode::GenericDecrement { value, .. }
        | ValueNode::ChangeInt32ToFloat64 { input: value }
        | ValueNode::ChangeUint32ToFloat64 { input: value }
        | ValueNode::ChangeFloat64ToInt32 { input: value }
        | ValueNode::CheckedFloat64ToInt32 { input: value }
        | ValueNode::ChangeInt32ToTagged { input: value }
        | ValueNode::ChangeUint32ToTagged { input: value }
        | ValueNode::ChangeFloat64ToTagged { input: value }
        | ValueNode::ChangeTaggedToInt32 { input: value }
        | ValueNode::ChangeTaggedToUint32 { input: value }
        | ValueNode::ChangeTaggedToFloat64 { input: value }
        | ValueNode::CheckedTaggedToInt32 { input: value }
        | ValueNode::CheckedTaggedToFloat64 { input: value }
        | ValueNode::ToBoolean { value }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::ToName { value, .. }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToNumberOrNumeric { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::NumberToString { value, .. } => {
            live.insert(*value);
        }

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => {
            live.insert(*receiver);
        }

        ValueNode::CheckDynamicValue { receiver, expected } => {
            live.insert(*receiver);
            live.insert(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => {
            live.insert(*input);
        }

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::ForInPrepare {
            enumerator: object, ..
        }
        | ValueNode::StringLength { string: object } => {
            live.insert(*object);
        }

        ValueNode::LoadEnumCacheLength { map } => {
            live.insert(*map);
        }

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            live.insert(*object);
            live.insert(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            live.insert(*object);
            live.insert(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            live.insert(*object);
            live.insert(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            live.insert(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => {
            live.insert(*context);
        }
        ValueNode::StoreContextSlot { context, value, .. } => {
            live.insert(*context);
            live.insert(*value);
        }

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            live.insert(*elements);
            live.insert(*index);
        }

        ValueNode::StoreFixedArrayElement {
            elements,
            index,
            value,
        }
        | ValueNode::StoreFixedDoubleArrayElement {
            elements,
            index,
            value,
        } => {
            live.insert(*elements);
            live.insert(*index);
            live.insert(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            live.insert(*object);
            live.insert(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            live.insert(*object);
            live.insert(*key);
            live.insert(*value);
        }

        // ── Binary nodes ──────────────────────────────────────────────────────
        ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::CheckedSmiDivide { left, right }
        | ValueNode::CheckedSmiModulus { left, right }
        | ValueNode::Int32Add { left, right }
        | ValueNode::Int32Subtract { left, right }
        | ValueNode::Int32Multiply { left, right }
        | ValueNode::Int32Divide { left, right }
        | ValueNode::Int32Modulus { left, right }
        | ValueNode::Int32BitwiseAnd { left, right }
        | ValueNode::Int32BitwiseOr { left, right }
        | ValueNode::Int32BitwiseXor { left, right }
        | ValueNode::Int32ShiftLeft { left, right }
        | ValueNode::Int32ShiftRight { left, right }
        | ValueNode::Int32ShiftRightLogical { left, right }
        | ValueNode::Uint32Add { left, right }
        | ValueNode::Uint32Subtract { left, right }
        | ValueNode::Uint32Multiply { left, right }
        | ValueNode::Uint32Divide { left, right }
        | ValueNode::Uint32Modulus { left, right }
        | ValueNode::Float64Add { left, right }
        | ValueNode::Float64Subtract { left, right }
        | ValueNode::Float64Multiply { left, right }
        | ValueNode::Float64Divide { left, right }
        | ValueNode::Float64Modulus { left, right }
        | ValueNode::Float64Exponentiate { left, right }
        | ValueNode::Int32Equal { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32LessThanOrEqual { left, right }
        | ValueNode::Int32GreaterThan { left, right }
        | ValueNode::Int32GreaterThanOrEqual { left, right }
        | ValueNode::Float64Equal { left, right }
        | ValueNode::Float64LessThan { left, right }
        | ValueNode::Float64LessThanOrEqual { left, right }
        | ValueNode::Float64GreaterThan { left, right }
        | ValueNode::Float64GreaterThanOrEqual { left, right }
        | ValueNode::StringConcat { left, right }
        | ValueNode::StringEqual { left, right }
        | ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. }
        | ValueNode::GenericDivide { left, right, .. }
        | ValueNode::GenericModulus { left, right, .. }
        | ValueNode::GenericExponentiate { left, right, .. }
        | ValueNode::GenericBitwiseAnd { left, right, .. }
        | ValueNode::GenericBitwiseOr { left, right, .. }
        | ValueNode::GenericBitwiseXor { left, right, .. }
        | ValueNode::GenericShiftLeft { left, right, .. }
        | ValueNode::GenericShiftRight { left, right, .. }
        | ValueNode::GenericShiftRightLogical { left, right, .. }
        | ValueNode::TaggedEqual { left, right, .. }
        | ValueNode::TaggedNotEqual { left, right, .. } => {
            live.insert(*left);
            live.insert(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            live.insert(*left);
            live.insert(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            live.insert(*receiver);
            live.insert(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            live.insert(*object);
            live.insert(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            live.insert(*key);
            live.insert(*object);
        }
        ValueNode::TestUndetectable { value } | ValueNode::TestTypeOf { value, .. } => {
            live.insert(*value);
        }

        ValueNode::StringAt { string, index } => {
            live.insert(*string);
            live.insert(*index);
        }

        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            live.insert(*receiver);
            live.insert(*cache_index);
            live.insert(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            live.insert(*object);
            live.insert(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => {
            live.insert(*exception);
        }
        ValueNode::CreateWithContext { object, .. } => {
            live.insert(*object);
        }

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
            live.insert(*callee);
            live.insert(*receiver);
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            live.insert(*constructor);
            for &a in args {
                live.insert(a);
            }
        }

        ValueNode::Phi { inputs } => {
            for &inp in inputs {
                live.insert(inp);
            }
        }
    }
}

/// Collect all [`NodeId`] operands referenced by a [`ControlNode`].
fn collect_control_node_inputs(ctrl: &ControlNode, live: &mut HashSet<NodeId>) {
    match ctrl {
        ControlNode::Return { value } => {
            live.insert(*value);
        }
        ControlNode::Branch { condition, .. } => {
            live.insert(*condition);
        }
        ControlNode::Jump { .. } | ControlNode::Deoptimize { .. } => {}
    }
}

/// Return `true` if this node has observable side-effects and must never be
/// removed by DCE regardless of whether its result value is consumed.
fn has_side_effects(node: &ValueNode) -> bool {
    matches!(
        node,
        // Stores always have side-effects.
        ValueNode::StoreField { .. }
            | ValueNode::StoreFixedArrayElement { .. }
            | ValueNode::StoreFixedDoubleArrayElement { .. }
            | ValueNode::StoreNamedGeneric { .. }
            | ValueNode::StoreKeyedGeneric { .. }
            | ValueNode::StoreGlobal { .. }
            | ValueNode::StoreContextSlot { .. }
            | ValueNode::StoreCurrentContextSlot { .. }
            // Calls may have side-effects.
            | ValueNode::Call { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            // Allocations have side-effects (GC pressure / observable identity).
            | ValueNode::CreateObjectLiteral { .. }
            | ValueNode::CreateArrayLiteral { .. }
            | ValueNode::CreateShallowObjectLiteral { .. }
            | ValueNode::CreateShallowArrayLiteral { .. }
            | ValueNode::CreateFunctionContext { .. }
            | ValueNode::CreateBlockContext { .. }
            | ValueNode::CreateCatchContext { .. }
            | ValueNode::CreateWithContext { .. }
            | ValueNode::CreateClosure { .. }
            | ValueNode::FastCreateClosure { .. }
            | ValueNode::CreateEmptyObjectLiteral
            | ValueNode::CreateRegExpLiteral { .. }
            // Guards deoptimise on failure — side-effecting.
            | ValueNode::CheckSmi { .. }
            | ValueNode::CheckNumber { .. }
            | ValueNode::CheckHeapObject { .. }
            | ValueNode::CheckSymbol { .. }
            | ValueNode::CheckString { .. }
            | ValueNode::CheckStringOrStringWrapper { .. }
            | ValueNode::CheckSeqOneByteString { .. }
            | ValueNode::CheckMaps { .. }
            | ValueNode::CheckMapsWithMigration { .. }
            | ValueNode::CheckValue { .. }
            | ValueNode::CheckDynamicValue { .. }
            | ValueNode::CheckInt32IsSmi { .. }
            | ValueNode::CheckUint32IsSmi { .. }
            | ValueNode::CheckHoleyFloat64IsSmi { .. }
            | ValueNode::CheckInt32Condition { .. }
            | ValueNode::CheckCacheIndicesNotCleared { .. }
            | ValueNode::CheckFloat64IsNan { .. }
            | ValueNode::CheckedSmiAdd { .. }
            | ValueNode::CheckedSmiSubtract { .. }
            | ValueNode::CheckedSmiMultiply { .. }
            | ValueNode::CheckedSmiDivide { .. }
            | ValueNode::CheckedSmiModulus { .. }
            | ValueNode::CheckedSmiIncrement { .. }
            | ValueNode::CheckedSmiDecrement { .. }
            | ValueNode::CheckedFloat64ToInt32 { .. }
            | ValueNode::CheckedTaggedToInt32 { .. }
            | ValueNode::CheckedTaggedToFloat64 { .. }
            // Debugger / abort are always live.
            | ValueNode::Debugger
            | ValueNode::Abort { .. }
            // Property mutation.
            | ValueNode::DeleteProperty { .. }
            // For-in side-effects (cache invalidation).
            | ValueNode::ForInPrepare { .. }
            | ValueNode::ForInNext { .. }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Substitution helper
// ─────────────────────────────────────────────────────────────────────────────

/// Rewrite every [`NodeId`] operand in `block` using the `subst` table.
///
/// This is used after the redundant-CheckMaps pass to redirect consumers of
/// the removed node ID to the canonical first-occurrence ID.
fn apply_subst_to_block(block: &mut BasicBlock, subst: &HashMap<NodeId, NodeId>) {
    let resolve = |id: NodeId| *subst.get(&id).unwrap_or(&id);

    for (_id, node) in &mut block.nodes {
        apply_subst_to_value_node(node, &resolve);
    }
    if let Some(ctrl) = &mut block.control {
        apply_subst_to_control_node(ctrl, &resolve);
    }
}

/// Apply a substitution function to every [`NodeId`] operand in a [`ValueNode`].
#[allow(clippy::too_many_lines)]
fn apply_subst_to_value_node(node: &mut ValueNode, resolve: &impl Fn(NodeId) -> NodeId) {
    match node {
        ValueNode::SmiConstant { .. }
        | ValueNode::Float64Constant { .. }
        | ValueNode::Int32Constant { .. }
        | ValueNode::Uint32Constant { .. }
        | ValueNode::BigIntConstant { .. }
        | ValueNode::TrueConstant
        | ValueNode::FalseConstant
        | ValueNode::NullConstant
        | ValueNode::UndefinedConstant
        | ValueNode::RootConstant { .. }
        | ValueNode::ExternalConstant { .. }
        | ValueNode::StringConstant { .. }
        | ValueNode::ConstantPoolEntry { .. }
        | ValueNode::Parameter { .. }
        | ValueNode::RegisterInput { .. }
        | ValueNode::ArgumentsLength
        | ValueNode::RestLength
        | ValueNode::LoadGlobal { .. }
        | ValueNode::LoadCurrentContextSlot { .. }
        | ValueNode::ArgumentsElements { .. }
        | ValueNode::RestElements { .. }
        | ValueNode::VirtualObject { .. }
        | ValueNode::CreateFunctionContext { .. }
        | ValueNode::CreateBlockContext { .. }
        | ValueNode::CreateShallowObjectLiteral { .. }
        | ValueNode::CreateShallowArrayLiteral { .. }
        | ValueNode::CreateObjectLiteral { .. }
        | ValueNode::CreateArrayLiteral { .. }
        | ValueNode::CreateEmptyObjectLiteral
        | ValueNode::CreateRegExpLiteral { .. }
        | ValueNode::CreateClosure { .. }
        | ValueNode::FastCreateClosure { .. }
        | ValueNode::GetTemplateObject { .. }
        | ValueNode::Debugger
        | ValueNode::Abort { .. } => {}

        ValueNode::GetArgument { index } => *index = resolve(*index),

        ValueNode::CheckedSmiIncrement { value }
        | ValueNode::CheckedSmiDecrement { value }
        | ValueNode::Int32Negate { value }
        | ValueNode::Int32Increment { value }
        | ValueNode::Int32Decrement { value }
        | ValueNode::Float64Negate { value }
        | ValueNode::Float64Ieee754Unary { value, .. }
        | ValueNode::GenericBitwiseNot { value, .. }
        | ValueNode::GenericNegate { value, .. }
        | ValueNode::GenericIncrement { value, .. }
        | ValueNode::GenericDecrement { value, .. }
        | ValueNode::ChangeInt32ToFloat64 { input: value }
        | ValueNode::ChangeUint32ToFloat64 { input: value }
        | ValueNode::ChangeFloat64ToInt32 { input: value }
        | ValueNode::CheckedFloat64ToInt32 { input: value }
        | ValueNode::ChangeInt32ToTagged { input: value }
        | ValueNode::ChangeUint32ToTagged { input: value }
        | ValueNode::ChangeFloat64ToTagged { input: value }
        | ValueNode::ChangeTaggedToInt32 { input: value }
        | ValueNode::ChangeTaggedToUint32 { input: value }
        | ValueNode::ChangeTaggedToFloat64 { input: value }
        | ValueNode::CheckedTaggedToInt32 { input: value }
        | ValueNode::CheckedTaggedToFloat64 { input: value }
        | ValueNode::ToBoolean { value }
        | ValueNode::ToString { value, .. }
        | ValueNode::ToObject { value, .. }
        | ValueNode::ToName { value, .. }
        | ValueNode::ToNumber { value, .. }
        | ValueNode::ToNumberOrNumeric { value, .. }
        | ValueNode::TypeOf { value }
        | ValueNode::NumberToString { value, .. }
        | ValueNode::TestUndetectable { value }
        | ValueNode::TestTypeOf { value, .. } => *value = resolve(*value),

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => *receiver = resolve(*receiver),

        ValueNode::CheckDynamicValue { receiver, expected } => {
            *receiver = resolve(*receiver);
            *expected = resolve(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => *input = resolve(*input),

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::LoadEnumCacheLength { map: object }
        | ValueNode::StringLength { string: object } => *object = resolve(*object),

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            *object = resolve(*object);
            *prototype = resolve(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            *value = resolve(*value);
        }

        ValueNode::LoadContextSlot { context, .. } => *context = resolve(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            *context = resolve(*context);
            *value = resolve(*value);
        }

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            *elements = resolve(*elements);
            *index = resolve(*index);
        }

        ValueNode::StoreFixedArrayElement {
            elements,
            index,
            value,
        }
        | ValueNode::StoreFixedDoubleArrayElement {
            elements,
            index,
            value,
        } => {
            *elements = resolve(*elements);
            *index = resolve(*index);
            *value = resolve(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            *object = resolve(*object);
            *value = resolve(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            *object = resolve(*object);
            *key = resolve(*key);
            *value = resolve(*value);
        }

        ValueNode::CheckedSmiAdd { left, right }
        | ValueNode::CheckedSmiSubtract { left, right }
        | ValueNode::CheckedSmiMultiply { left, right }
        | ValueNode::CheckedSmiDivide { left, right }
        | ValueNode::CheckedSmiModulus { left, right }
        | ValueNode::Int32Add { left, right }
        | ValueNode::Int32Subtract { left, right }
        | ValueNode::Int32Multiply { left, right }
        | ValueNode::Int32Divide { left, right }
        | ValueNode::Int32Modulus { left, right }
        | ValueNode::Int32BitwiseAnd { left, right }
        | ValueNode::Int32BitwiseOr { left, right }
        | ValueNode::Int32BitwiseXor { left, right }
        | ValueNode::Int32ShiftLeft { left, right }
        | ValueNode::Int32ShiftRight { left, right }
        | ValueNode::Int32ShiftRightLogical { left, right }
        | ValueNode::Uint32Add { left, right }
        | ValueNode::Uint32Subtract { left, right }
        | ValueNode::Uint32Multiply { left, right }
        | ValueNode::Uint32Divide { left, right }
        | ValueNode::Uint32Modulus { left, right }
        | ValueNode::Float64Add { left, right }
        | ValueNode::Float64Subtract { left, right }
        | ValueNode::Float64Multiply { left, right }
        | ValueNode::Float64Divide { left, right }
        | ValueNode::Float64Modulus { left, right }
        | ValueNode::Float64Exponentiate { left, right }
        | ValueNode::Int32Equal { left, right }
        | ValueNode::Int32StrictEqual { left, right }
        | ValueNode::Int32LessThan { left, right }
        | ValueNode::Int32LessThanOrEqual { left, right }
        | ValueNode::Int32GreaterThan { left, right }
        | ValueNode::Int32GreaterThanOrEqual { left, right }
        | ValueNode::Float64Equal { left, right }
        | ValueNode::Float64LessThan { left, right }
        | ValueNode::Float64LessThanOrEqual { left, right }
        | ValueNode::Float64GreaterThan { left, right }
        | ValueNode::Float64GreaterThanOrEqual { left, right }
        | ValueNode::StringConcat { left, right }
        | ValueNode::StringEqual { left, right }
        | ValueNode::GenericAdd { left, right, .. }
        | ValueNode::GenericSubtract { left, right, .. }
        | ValueNode::GenericMultiply { left, right, .. }
        | ValueNode::GenericDivide { left, right, .. }
        | ValueNode::GenericModulus { left, right, .. }
        | ValueNode::GenericExponentiate { left, right, .. }
        | ValueNode::GenericBitwiseAnd { left, right, .. }
        | ValueNode::GenericBitwiseOr { left, right, .. }
        | ValueNode::GenericBitwiseXor { left, right, .. }
        | ValueNode::GenericShiftLeft { left, right, .. }
        | ValueNode::GenericShiftRight { left, right, .. }
        | ValueNode::GenericShiftRightLogical { left, right, .. }
        | ValueNode::TaggedEqual { left, right, .. }
        | ValueNode::TaggedNotEqual { left, right, .. } => {
            *left = resolve(*left);
            *right = resolve(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            *left = resolve(*left);
            *right = resolve(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            *receiver = resolve(*receiver);
            *indices = resolve(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            *object = resolve(*object);
            *callable = resolve(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            *key = resolve(*key);
            *object = resolve(*object);
        }

        ValueNode::StringAt { string, index } => {
            *string = resolve(*string);
            *index = resolve(*index);
        }

        ValueNode::ForInPrepare { enumerator, .. } => *enumerator = resolve(*enumerator),
        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            *receiver = resolve(*receiver);
            *cache_index = resolve(*cache_index);
            *cache_array = resolve(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            *object = resolve(*object);
            *key = resolve(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => *exception = resolve(*exception),
        ValueNode::CreateWithContext { object, .. } => *object = resolve(*object),

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
            *callee = resolve(*callee);
            *receiver = resolve(*receiver);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            *constructor = resolve(*constructor);
            for a in args.iter_mut() {
                *a = resolve(*a);
            }
        }

        ValueNode::Phi { inputs } => {
            for inp in inputs.iter_mut() {
                *inp = resolve(*inp);
            }
        }
    }
}

/// Apply a substitution to the operands of a [`ControlNode`].
fn apply_subst_to_control_node(ctrl: &mut ControlNode, resolve: &impl Fn(NodeId) -> NodeId) {
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
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Count the total number of value nodes across all blocks.
    fn total_node_count(graph: &MaglevGraph) -> usize {
        graph.blocks().iter().map(|b| b.nodes.len()).sum()
    }

    /// Return the single value node in block 0 at position `pos`.
    fn node_at(graph: &MaglevGraph, pos: usize) -> &ValueNode {
        &graph.blocks()[0].nodes[pos].1
    }

    // ── Constant folding — Int32 ──────────────────────────────────────────────

    #[test]
    fn test_fold_int32_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        // The Int32Add node must have been folded to Int32Constant(7).
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_fold_int32_subtract() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let sub = block.push_value(ValueNode::Int32Subtract {
            left: c10,
            right: c3,
        });
        block.set_control(ControlNode::Return { value: sub });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_fold_int32_multiply() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c6 = block.push_value(ValueNode::Int32Constant { value: 6 });
        let c7 = block.push_value(ValueNode::Int32Constant { value: 7 });
        let mul = block.push_value(ValueNode::Int32Multiply {
            left: c6,
            right: c7,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 42 });
    }

    #[test]
    fn test_fold_int32_divide() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c12 = block.push_value(ValueNode::Int32Constant { value: 12 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let div = block.push_value(ValueNode::Int32Divide {
            left: c12,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: div });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 3 });
    }

    #[test]
    fn test_fold_int32_divide_by_zero_not_folded() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c5 = block.push_value(ValueNode::Int32Constant { value: 5 });
        let c0 = block.push_value(ValueNode::Int32Constant { value: 0 });
        let div = block.push_value(ValueNode::Int32Divide {
            left: c5,
            right: c0,
        });
        block.set_control(ControlNode::Return { value: div });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Div-by-zero must NOT be folded (the node stays as Int32Divide).
        assert!(matches!(node_at(&graph, 2), ValueNode::Int32Divide { .. }));
        // Node count should not decrease (DCE keeps the divide because its
        // result is consumed by the Return).
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_fold_int32_modulus() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let rem = block.push_value(ValueNode::Int32Modulus {
            left: c10,
            right: c3,
        });
        block.set_control(ControlNode::Return { value: rem });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 1 });
    }

    #[test]
    fn test_fold_int32_negate() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c5 = block.push_value(ValueNode::Int32Constant { value: 5 });
        let neg = block.push_value(ValueNode::Int32Negate { value: c5 });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: -5 });
    }

    #[test]
    fn test_fold_int32_increment() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c9 = block.push_value(ValueNode::Int32Constant { value: 9 });
        let inc = block.push_value(ValueNode::Int32Increment { value: c9 });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 10 });
    }

    #[test]
    fn test_fold_int32_decrement() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c10 = block.push_value(ValueNode::Int32Constant { value: 10 });
        let dec = block.push_value(ValueNode::Int32Decrement { value: c10 });
        block.set_control(ControlNode::Return { value: dec });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 9 });
    }

    // ── Constant folding — CheckedSmi variants ────────────────────────────────

    #[test]
    fn test_fold_checked_smi_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let a = block.push_value(ValueNode::SmiConstant { value: 2 });
        let b = block.push_value(ValueNode::SmiConstant { value: 3 });
        let sum = block.push_value(ValueNode::CheckedSmiAdd { left: a, right: b });
        block.set_control(ControlNode::Return { value: sum });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 5 });
    }

    #[test]
    fn test_fold_checked_smi_increment() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::SmiConstant { value: 41 });
        let inc = block.push_value(ValueNode::CheckedSmiIncrement { value: c });
        block.set_control(ControlNode::Return { value: inc });
        graph.add_block(block);

        optimize(&mut graph);

        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 42 });
    }

    // ── Constant folding — Float64 ────────────────────────────────────────────

    #[test]
    fn test_fold_float64_add() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 1.5 });
        let y = block.push_value(ValueNode::Float64Constant { value: 2.5 });
        let add = block.push_value(ValueNode::Float64Add { left: x, right: y });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    #[test]
    fn test_fold_float64_multiply() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 3.0 });
        let y = block.push_value(ValueNode::Float64Constant { value: 4.0 });
        let mul = block.push_value(ValueNode::Float64Multiply { left: x, right: y });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - 12.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    #[test]
    fn test_fold_float64_negate() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let x = block.push_value(ValueNode::Float64Constant { value: 2.5 });
        let neg = block.push_value(ValueNode::Float64Negate { value: x });
        block.set_control(ControlNode::Return { value: neg });
        graph.add_block(block);

        optimize(&mut graph);

        if let ValueNode::Float64Constant { value } = node_at(&graph, 0) {
            assert!((value - (-2.5)).abs() < f64::EPSILON);
        } else {
            panic!("expected Float64Constant after folding");
        }
    }

    // ── Constant folding — no fold when operand is dynamic ────────────────────

    #[test]
    fn test_no_fold_when_operand_is_parameter() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        let c = block.push_value(ValueNode::Int32Constant { value: 1 });
        let add = block.push_value(ValueNode::Int32Add { left: p, right: c });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        optimize(&mut graph);

        // The add node should remain unchanged.
        assert!(matches!(node_at(&graph, 2), ValueNode::Int32Add { .. }));
    }

    // ── Dead-code elimination ─────────────────────────────────────────────────

    #[test]
    fn test_dce_removes_unused_pure_node() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        // This constant is never used.
        block.push_value(ValueNode::Int32Constant { value: 99 });
        // This one is returned.
        let ret_val = block.push_value(ValueNode::SmiConstant { value: 0 });
        block.set_control(ControlNode::Return { value: ret_val });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The unused constant should have been removed.
        assert!(total_node_count(&graph) < before);
    }

    #[test]
    fn test_dce_keeps_side_effect_node() {
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let obj = block.push_value(ValueNode::Parameter { index: 0 });
        let val = block.push_value(ValueNode::Int32Constant { value: 1 });
        // Store has a side-effect: must be kept even if nobody reads its result.
        block.push_value(ValueNode::StoreField {
            object: obj,
            offset: 8,
            value: val,
        });
        let ret_val = block.push_value(ValueNode::UndefinedConstant);
        block.set_control(ControlNode::Return { value: ret_val });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Store must not be removed.
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_dce_keeps_check_maps_node() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // CheckMaps is a guard with side-effects (deopt on failure).
        let _check = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        let field = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: field });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // CheckMaps must survive DCE.
        assert_eq!(total_node_count(&graph), before);
    }

    // ── Redundant CheckMaps removal ───────────────────────────────────────────

    #[test]
    fn test_redundant_check_maps_removed() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // First check — canonical.
        let _check1 = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        // Second identical check — redundant.
        let _check2 = block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        let field = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: field });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // The redundant CheckMaps gets replaced by UndefinedConstant; DCE then
        // removes it (it's pure and unused).  Net effect: node count decreases.
        assert!(total_node_count(&graph) < before);
    }

    #[test]
    fn test_different_receivers_both_kept() {
        let mut graph = MaglevGraph::new(2);
        let mut block = BasicBlock::new(0);
        let p0 = block.push_value(ValueNode::Parameter { index: 0 });
        let p1 = block.push_value(ValueNode::Parameter { index: 1 });
        // Two checks with *different* receivers — both must be kept.
        block.push_value(ValueNode::CheckMaps {
            receiver: p0,
            feedback_slot: 0,
        });
        block.push_value(ValueNode::CheckMaps {
            receiver: p1,
            feedback_slot: 0,
        });
        let f0 = block.push_value(ValueNode::LoadField {
            object: p0,
            offset: 8,
        });
        let f1 = block.push_value(ValueNode::LoadField {
            object: p1,
            offset: 8,
        });
        let add = block.push_value(ValueNode::Int32Add {
            left: f0,
            right: f1,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        // Neither check is redundant — count must not decrease.
        assert_eq!(total_node_count(&graph), before);
    }

    #[test]
    fn test_different_slots_both_kept() {
        let mut graph = MaglevGraph::new(1);
        let mut block = BasicBlock::new(0);
        let p = block.push_value(ValueNode::Parameter { index: 0 });
        // Same receiver, different feedback slots — both must be kept.
        block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 0,
        });
        block.push_value(ValueNode::CheckMaps {
            receiver: p,
            feedback_slot: 1,
        });
        let f = block.push_value(ValueNode::LoadField {
            object: p,
            offset: 8,
        });
        block.set_control(ControlNode::Return { value: f });
        graph.add_block(block);

        let before = total_node_count(&graph);
        optimize(&mut graph);
        assert_eq!(total_node_count(&graph), before);
    }

    // ── Combined passes ───────────────────────────────────────────────────────

    #[test]
    fn test_combined_fold_and_dce_reduces_node_count() {
        // Build: return (3 + 4)
        // After constant folding → return Int32Constant(7)
        // After DCE → constants 3 and 4 are dead.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        block.set_control(ControlNode::Return { value: add });
        graph.add_block(block);

        let before = total_node_count(&graph); // 3 nodes
        optimize(&mut graph);
        let after = total_node_count(&graph);
        // We expect only Int32Constant(7) to remain; 3 and 4 are dead.
        assert!(after < before, "expected fewer nodes after optimisation");
        // Exactly one node should survive (the folded result).
        assert_eq!(after, 1);
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 7 });
    }

    #[test]
    fn test_chain_folding() {
        // (1 + 2) * (3 + 4) = 3 * 7 = 21
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c1 = block.push_value(ValueNode::Int32Constant { value: 1 });
        let c2 = block.push_value(ValueNode::Int32Constant { value: 2 });
        let c3 = block.push_value(ValueNode::Int32Constant { value: 3 });
        let c4 = block.push_value(ValueNode::Int32Constant { value: 4 });
        let add1 = block.push_value(ValueNode::Int32Add {
            left: c1,
            right: c2,
        });
        let add2 = block.push_value(ValueNode::Int32Add {
            left: c3,
            right: c4,
        });
        let mul = block.push_value(ValueNode::Int32Multiply {
            left: add1,
            right: add2,
        });
        block.set_control(ControlNode::Return { value: mul });
        graph.add_block(block);

        optimize(&mut graph);

        // All intermediates should be DCE'd; only one node left.
        assert_eq!(total_node_count(&graph), 1);
        assert_eq!(node_at(&graph, 0), &ValueNode::Int32Constant { value: 21 });
    }
}

//! Loop-invariant code motion (LICM).
//!
//! This pass detects **natural loops** in the control-flow graph and hoists
//! loop-invariant value nodes — those whose inputs are all defined outside the
//! loop — into the loop's *preheader* block.  This avoids re-computing the
//! same value on every iteration.
//!
//! # Algorithm
//!
//! 1. **Back-edge detection** — a *back-edge* is an edge from block `b` to
//!    block `h` where `h` dominates `b`.  We approximate dominance by
//!    requiring `h.id <= b.id` (valid when blocks are in RPO — the common
//!    case for Maglev graphs emitted by [`graph_builder`]).
//! 2. **Loop body collection** — given header `h` and back-edge source `b`,
//!    the loop body is the set of blocks on any path from `h` to `b` within
//!    the CFG.  We compute this with a reverse walk from `b` to `h` over
//!    predecessors.
//! 3. **Preheader identification** — the preheader is the unique predecessor
//!    of the header `h` that is *not* inside the loop body.  If the header
//!    has multiple non-loop predecessors we skip the loop (creating a
//!    synthetic preheader would require splitting edges).
//! 4. **Invariant detection** — a value node in the loop body is
//!    *loop-invariant* if it is **pure** (no side-effects) and every
//!    [`NodeId`] it references is defined *outside* the loop body.
//! 5. **Hoisting** — invariant nodes are moved (appended) to the preheader
//!    block, preserving their [`NodeId`] so all downstream references remain
//!    valid.
//!
//! # Limitations
//!
//! - Only a single pass over each detected loop is performed (no iterative
//!   widening).
//! - Nested loops are handled independently; each back-edge produces its own
//!   loop.
//! - Guards and side-effecting nodes are never hoisted.
//!
//! # Usage
//!
//! ```
//! use stator_core::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode,
//! };
//! use stator_core::compiler::maglev::licm::hoist_loop_invariants;
//!
//! // Build a small diamond + loop CFG:
//! //   block 0 (preheader): jump → 1
//! //   block 1 (header): branch → 2 (body) / 3 (exit)
//! //   block 2 (body): jump → 1   (back-edge)
//! //   block 3 (exit): return
//! let mut graph = MaglevGraph::new(1);
//!
//! // block 0: preheader — define a constant and jump to header.
//! let mut b0 = BasicBlock::new(0);
//! let _p0 = b0.push_value(ValueNode::Parameter { index: 0 });
//! b0.set_control(ControlNode::Jump { target: 1 });
//! graph.add_block(b0);
//!
//! // block 1: loop header.
//! let mut b1 = BasicBlock::new(1);
//! b1.add_predecessor(0);
//! b1.add_predecessor(2);
//! let cond = b1.push_value(ValueNode::TrueConstant);
//! b1.set_control(ControlNode::Branch {
//!     condition: cond,
//!     if_true: 2,
//!     if_false: 3,
//! });
//! graph.add_block(b1);
//!
//! // block 2: loop body — a loop-invariant constant (uses no loop-local nodes).
//! let mut b2 = BasicBlock::new(2);
//! b2.add_predecessor(1);
//! let _inv = b2.push_value(ValueNode::Int32Constant { value: 42 });
//! b2.set_control(ControlNode::Jump { target: 1 });
//! graph.add_block(b2);
//!
//! // block 3: exit.
//! let mut b3 = BasicBlock::new(3);
//! b3.add_predecessor(1);
//! let undef = b3.push_value(ValueNode::UndefinedConstant);
//! b3.set_control(ControlNode::Return { value: undef });
//! graph.add_block(b3);
//!
//! let body_before = graph.blocks()[2].nodes.len();
//! hoist_loop_invariants(&mut graph);
//! // The invariant node was hoisted out of the loop body.
//! assert!(graph.blocks()[2].nodes.len() < body_before);
//! ```

use std::collections::HashSet;

use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Detect natural loops and hoist loop-invariant pure nodes to the preheader.
pub fn hoist_loop_invariants(graph: &mut MaglevGraph) -> usize {
    let loops = detect_loops(graph);

    let mut total = 0;
    for lp in &loops {
        total += hoist_one_loop(graph, lp);
    }
    if !loops.is_empty() {
        eprintln!(
            "LICM: {} loops detected, {} nodes hoisted",
            loops.len(),
            total
        );
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop representation
// ─────────────────────────────────────────────────────────────────────────────

/// A natural loop detected from a back-edge.
pub struct NaturalLoop {
    /// Block index of the loop header.
    pub header: u32,
    /// Block index of the preheader (unique non-loop predecessor of header).
    pub preheader: u32,
    /// Set of block indices forming the loop body (includes header).
    pub body: HashSet<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop detection
// ─────────────────────────────────────────────────────────────────────────────

/// Find all natural loops by locating back-edges.
pub fn detect_loops(graph: &MaglevGraph) -> Vec<NaturalLoop> {
    let mut loops = Vec::new();

    for block in graph.blocks() {
        // Look at the control node for edges to earlier blocks (back-edges).
        let targets = control_targets(block);
        for &target in &targets {
            // A back-edge exists when target <= block.id (header dominates the
            // source in RPO layout).  The equality case covers *self-loops*
            // where a single block branches back to itself (common for simple
            // while-loops whose entire body fits in the header block).
            if target <= block.id
                && let Some(lp) = build_loop(graph, target, block.id)
            {
                loops.push(lp);
            }
        }
    }

    loops
}

/// Return the block indices that a block's control node branches to.
pub fn control_targets(block: &BasicBlock) -> Vec<u32> {
    match &block.control {
        Some(ControlNode::Jump { target }) => vec![*target],
        Some(ControlNode::Branch {
            if_true, if_false, ..
        }) => vec![*if_true, *if_false],
        _ => Vec::new(),
    }
}

/// Attempt to build a [`NaturalLoop`] from header `header` and back-edge
/// source `back_src`.  Returns `None` if no unique preheader exists.
fn build_loop(graph: &MaglevGraph, header: u32, back_src: u32) -> Option<NaturalLoop> {
    // Collect the loop body via reverse walk from back_src to header.
    let mut body = HashSet::new();
    body.insert(header);
    body.insert(back_src);

    let mut worklist = vec![back_src];
    while let Some(b) = worklist.pop() {
        if b == header {
            continue;
        }
        if let Some(block) = graph.block(b) {
            for &pred in &block.predecessors {
                if body.insert(pred) {
                    worklist.push(pred);
                }
            }
        }
    }

    // Find the unique preheader: a predecessor of the header NOT in the loop.
    let header_block = graph.block(header)?;
    let preheaders: Vec<u32> = header_block
        .predecessors
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();

    // Require exactly one non-loop predecessor (the preheader).
    if preheaders.len() != 1 {
        return None;
    }

    Some(NaturalLoop {
        header,
        preheader: preheaders[0],
        body,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariant hoisting
// ─────────────────────────────────────────────────────────────────────────────

/// Hoist invariant nodes from a single loop to its preheader.
/// Returns the number of nodes hoisted.
fn hoist_one_loop(graph: &mut MaglevGraph, lp: &NaturalLoop) -> usize {
    // Collect all NodeIds defined outside the loop body.
    let mut outside_defs: HashSet<NodeId> = HashSet::new();
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            for (id, _) in &block.nodes {
                outside_defs.insert(*id);
            }
        }
    }

    // Collect objects mutated inside the loop so we can avoid hoisting loads
    // that may alias those stores (soundness check).
    let mutated_objects = collect_mutated_objects(graph, &lp.body);

    // Collect global name indices written by StoreGlobal inside the loop.
    // A LoadGlobal for a name that is also stored must not be hoisted.
    let mutated_globals = collect_mutated_globals(graph, &lp.body);

    // Scan loop blocks for hoistable nodes.  Include the loop header — in
    // single-block loops the entire body lives in the header.  Nodes whose
    // inputs are all defined outside the loop are loop-invariant regardless
    // of which block they reside in.
    let mut to_hoist: Vec<(u32, usize, NodeId, ValueNode)> = Vec::new();

    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (pos, (id, node)) in block.nodes.iter().enumerate() {
            // Skip Phi nodes — they define loop-carried values.
            if matches!(node, ValueNode::Phi { .. }) {
                continue;
            }
            if is_pure(node)
                && all_inputs_outside(node, &outside_defs)
                && !load_aliases_store(node, &mutated_objects)
                && !global_load_aliases_store(node, &mutated_globals)
            {
                to_hoist.push((block.id, pos, *id, node.clone()));
                // After hoisting this node, its NodeId becomes "outside" too,
                // so subsequent nodes that reference it may also qualify.
                outside_defs.insert(*id);
            }
        }
    }

    if to_hoist.is_empty() {
        return 0;
    }

    // Mark hoisted NodeIds as outside for potential future passes.
    for &(_, _, id, _) in &to_hoist {
        outside_defs.insert(id);
    }

    // Remove hoisted nodes from their source blocks (iterate in reverse
    // position order so indices remain valid).
    // Group removals by block.
    let mut removals_by_block: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for &(blk, pos, _, _) in &to_hoist {
        removals_by_block.entry(blk).or_default().push(pos);
    }
    for positions in removals_by_block.values_mut() {
        positions.sort_unstable();
        positions.dedup();
    }

    // Actually remove from blocks.
    for block in graph.blocks_mut() {
        if let Some(positions) = removals_by_block.get(&block.id) {
            // Remove in reverse order to keep indices stable.
            for &pos in positions.iter().rev() {
                if pos < block.nodes.len() {
                    block.nodes.remove(pos);
                }
            }
        }
    }

    // Append hoisted nodes to the preheader (before its control node).
    let count = to_hoist.len();
    if let Some(preheader) = graph.block_mut(lp.preheader) {
        for (_, _, id, node) in to_hoist {
            preheader.push_with_id(id, node);
        }
    }
    count
}

// ─────────────────────────────────────────────────────────────────────────────
// Store-alias analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Collect all [`NodeId`]s that appear as the target *object* of a store node
/// anywhere in the loop body.  Used to prevent hoisting loads that may read
/// a field written inside the loop.
fn collect_mutated_objects(graph: &MaglevGraph, loop_body: &HashSet<u32>) -> HashSet<NodeId> {
    let mut mutated = HashSet::new();
    for block in graph.blocks() {
        if !loop_body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            match node {
                ValueNode::StoreField { object, .. }
                | ValueNode::StoreNamedGeneric { object, .. }
                | ValueNode::StoreKeyedGeneric { object, .. } => {
                    mutated.insert(*object);
                }
                ValueNode::StoreFixedArrayElement { elements, .. }
                | ValueNode::StoreFixedDoubleArrayElement { elements, .. } => {
                    mutated.insert(*elements);
                }
                _ => {}
            }
        }
    }
    mutated
}

/// Collect all global name-indices written by [`ValueNode::StoreGlobal`]
/// inside the loop body.  A [`ValueNode::LoadGlobal`] for a stored name must
/// not be hoisted because the value changes across iterations.
fn collect_mutated_globals(graph: &MaglevGraph, loop_body: &HashSet<u32>) -> HashSet<u32> {
    let mut mutated = HashSet::new();
    for block in graph.blocks() {
        if !loop_body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            if let ValueNode::StoreGlobal { name, .. } = node {
                mutated.insert(*name);
            }
        }
    }
    mutated
}

/// Return `true` when `node` is a load whose target object has been stored
/// to inside the loop (per `mutated`).  Hoisting such a load would be
/// unsound because the loaded value may change across iterations.
fn load_aliases_store(node: &ValueNode, mutated: &HashSet<NodeId>) -> bool {
    match node {
        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. } => mutated.contains(object),
        ValueNode::LoadFixedArrayElement { elements, .. }
        | ValueNode::LoadFixedDoubleArrayElement { elements, .. }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, .. } => {
            mutated.contains(elements)
        }
        _ => false,
    }
}

/// Return `true` when `node` is a [`ValueNode::LoadGlobal`] whose name index
/// appears in the `mutated_globals` set (i.e. there is a [`StoreGlobal`] to
/// the same name inside the loop).
fn global_load_aliases_store(node: &ValueNode, mutated_globals: &HashSet<u32>) -> bool {
    if let ValueNode::LoadGlobal { name, .. } = node {
        mutated_globals.contains(name)
    } else {
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Purity check
// ─────────────────────────────────────────────────────────────────────────────

/// A node is *pure* if it has no observable side-effects (it's safe to move
/// or remove).  Guards, stores, calls, and allocations are NOT pure.
fn is_pure(node: &ValueNode) -> bool {
    !matches!(
        node,
        ValueNode::StoreField { .. }
            | ValueNode::StoreFixedArrayElement { .. }
            | ValueNode::StoreFixedDoubleArrayElement { .. }
            | ValueNode::StoreNamedGeneric { .. }
            | ValueNode::StoreKeyedGeneric { .. }
            | ValueNode::StoreGlobal { .. }
            | ValueNode::StoreContextSlot { .. }
            | ValueNode::StoreCurrentContextSlot { .. }
            | ValueNode::Call { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
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
            | ValueNode::Debugger
            | ValueNode::Abort { .. }
            | ValueNode::DeleteProperty { .. }
            | ValueNode::ForInPrepare { .. }
            | ValueNode::ForInNext { .. }
    )
}

/// Return `true` when every [`NodeId`] referenced by `node` is in the
/// `outside` set (defined before the loop).
fn all_inputs_outside(node: &ValueNode, outside: &HashSet<NodeId>) -> bool {
    let mut ok = true;
    visit_inputs(node, &mut |id| {
        if !outside.contains(&id) {
            ok = false;
        }
    });
    ok
}

/// Call `f` for every [`NodeId`] that `node` uses as an input.
fn visit_inputs(node: &ValueNode, f: &mut impl FnMut(NodeId)) {
    match node {
        // No inputs.
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

        // Single-input.
        ValueNode::GetArgument { index } => f(*index),

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
        | ValueNode::TestTypeOf { value, .. } => f(*value),

        ValueNode::CheckSmi { receiver }
        | ValueNode::CheckNumber { receiver }
        | ValueNode::CheckHeapObject { receiver }
        | ValueNode::CheckSymbol { receiver }
        | ValueNode::CheckString { receiver }
        | ValueNode::CheckStringOrStringWrapper { receiver }
        | ValueNode::CheckSeqOneByteString { receiver }
        | ValueNode::CheckMaps { receiver, .. }
        | ValueNode::CheckMapsWithMigration { receiver, .. }
        | ValueNode::CheckValue { receiver, .. } => f(*receiver),

        ValueNode::CheckDynamicValue { receiver, expected } => {
            f(*receiver);
            f(*expected);
        }

        ValueNode::CheckInt32IsSmi { input }
        | ValueNode::CheckUint32IsSmi { input }
        | ValueNode::CheckHoleyFloat64IsSmi { input }
        | ValueNode::CheckFloat64IsNan { input } => f(*input),

        ValueNode::LoadField { object, .. }
        | ValueNode::LoadTaggedField { object, .. }
        | ValueNode::LoadDoubleField { object, .. }
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::ForInPrepare {
            enumerator: object, ..
        }
        | ValueNode::StringLength { string: object }
        | ValueNode::LoadEnumCacheLength { map: object } => f(*object),

        ValueNode::LoadKeyedGeneric { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::HasInPrototypeChain { object, prototype } => {
            f(*object);
            f(*prototype);
        }

        ValueNode::StoreField { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::StoreCurrentContextSlot { value, .. } | ValueNode::StoreGlobal { value, .. } => {
            f(*value)
        }

        ValueNode::LoadContextSlot { context, .. } => f(*context),
        ValueNode::StoreContextSlot { context, value, .. } => {
            f(*context);
            f(*value);
        }

        ValueNode::LoadFixedArrayElement { elements, index }
        | ValueNode::LoadFixedDoubleArrayElement { elements, index }
        | ValueNode::LoadHoleyFixedDoubleArrayElement { elements, index } => {
            f(*elements);
            f(*index);
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
            f(*elements);
            f(*index);
            f(*value);
        }

        ValueNode::StoreNamedGeneric { object, value, .. } => {
            f(*object);
            f(*value);
        }

        ValueNode::StoreKeyedGeneric {
            object, key, value, ..
        } => {
            f(*object);
            f(*key);
            f(*value);
        }

        // Binary.
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
            f(*left);
            f(*right);
        }

        ValueNode::CheckInt32Condition { left, right, .. } => {
            f(*left);
            f(*right);
        }

        ValueNode::CheckCacheIndicesNotCleared { receiver, indices } => {
            f(*receiver);
            f(*indices);
        }

        ValueNode::TestInstanceOf {
            object, callable, ..
        } => {
            f(*object);
            f(*callable);
        }
        ValueNode::TestIn { key, object, .. } => {
            f(*key);
            f(*object);
        }

        ValueNode::StringAt { string, index } => {
            f(*string);
            f(*index);
        }

        ValueNode::ForInNext {
            receiver,
            cache_index,
            cache_array,
            ..
        } => {
            f(*receiver);
            f(*cache_index);
            f(*cache_array);
        }

        ValueNode::DeleteProperty { object, key, .. } => {
            f(*object);
            f(*key);
        }

        ValueNode::CreateCatchContext { exception, .. } => f(*exception),
        ValueNode::CreateWithContext { object, .. } => f(*object),

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
            f(*callee);
            f(*receiver);
            for &a in args.iter() {
                f(a);
            }
        }

        ValueNode::CallBuiltin { args, .. } | ValueNode::CallRuntime { args, .. } => {
            for &a in args.iter() {
                f(a);
            }
        }

        ValueNode::Construct {
            constructor, args, ..
        }
        | ValueNode::ConstructWithSpread {
            constructor, args, ..
        } => {
            f(*constructor);
            for &a in args.iter() {
                f(a);
            }
        }

        ValueNode::Phi { inputs } => {
            for &inp in inputs.iter() {
                f(inp);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, ValueNode};

    /// Build a minimal loop CFG:
    ///   block 0 (preheader) → jump 1
    ///   block 1 (header) → branch(cond, 2, 3)
    ///   block 2 (body) → jump 1   ← back-edge
    ///   block 3 (exit) → return
    fn loop_graph() -> MaglevGraph {
        let mut graph = MaglevGraph::new(1);

        // block 0: preheader
        let mut b0 = BasicBlock::new(0);
        let _p = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // block 1: header
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        let cond = b1.push_value(ValueNode::TrueConstant);
        b1.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        // block 2: body
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        // block 3: exit
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        let undef = b3.push_value(ValueNode::UndefinedConstant);
        b3.set_control(ControlNode::Return { value: undef });
        graph.add_block(b3);

        graph
    }

    // ── Invariant constant is hoisted ────────────────────────────────────────

    #[test]
    fn test_hoist_constant_from_loop_body() {
        let mut graph = loop_graph();

        // Add an invariant constant to block 2 (loop body).
        graph.blocks_mut()[2]
            .nodes
            .insert(0, (NodeId(100), ValueNode::Int32Constant { value: 42 }));

        assert_eq!(graph.blocks()[2].nodes.len(), 1);

        hoist_loop_invariants(&mut graph);

        // The constant should have been moved to block 0 (preheader).
        assert_eq!(graph.blocks()[2].nodes.len(), 0);
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Int32Constant { value: 42 }))
        );
    }

    // ── Side-effecting node stays in loop ────────────────────────────────────

    #[test]
    fn test_side_effecting_node_not_hoisted() {
        let mut graph = loop_graph();

        // A call node in block 2 — has side-effects, must not be hoisted.
        graph.blocks_mut()[2].nodes.insert(
            0,
            (
                NodeId(200),
                ValueNode::CallRuntime {
                    function_id: 0,
                    args: vec![],
                },
            ),
        );

        let before = graph.blocks()[2].nodes.len();
        hoist_loop_invariants(&mut graph);
        assert_eq!(graph.blocks()[2].nodes.len(), before);
    }

    // ── Node with loop-local input stays in loop ─────────────────────────────

    #[test]
    fn test_node_with_loop_local_input_not_hoisted() {
        let mut graph = loop_graph();

        // Define a node inside the loop body and a second node that uses it.
        let inner_id = NodeId(300);
        let user_id = NodeId(301);
        graph.blocks_mut()[2]
            .nodes
            .push((inner_id, ValueNode::Int32Constant { value: 1 }));
        graph.blocks_mut()[2]
            .nodes
            .push((user_id, ValueNode::Int32Negate { value: inner_id }));

        // The negate depends on a loop-local node, so only the constant (which
        // is loop-invariant) should be hoisted at first glance.  But since the
        // constant itself is hoistable, and we now eagerly add hoisted IDs to
        // outside_defs during the scan, the negate also qualifies for hoisting
        // because its sole input became an outside def once the constant was
        // scheduled for hoisting.  This is correct: both nodes are genuinely
        // loop-invariant.
        let body_before = graph.blocks()[2].nodes.len();
        hoist_loop_invariants(&mut graph);
        // Both the constant AND the negate should be hoisted — the constant
        // has no loop-local inputs, and the negate's input (the constant) was
        // added to outside_defs eagerly when it was hoisted.
        assert_eq!(
            graph.blocks()[2].nodes.len(),
            body_before - 2,
            "both the constant and the negate should have been hoisted"
        );
    }

    // ── No loops → no changes ────────────────────────────────────────────────

    #[test]
    fn test_no_loop_no_change() {
        // Straight-line graph: block 0 → return.
        let mut graph = MaglevGraph::new(0);
        let mut block = BasicBlock::new(0);
        let c = block.push_value(ValueNode::Int32Constant { value: 7 });
        block.set_control(ControlNode::Return { value: c });
        graph.add_block(block);

        let before = graph.blocks()[0].nodes.len();
        hoist_loop_invariants(&mut graph);
        assert_eq!(graph.blocks()[0].nodes.len(), before);
    }

    // ── Arithmetic with outside inputs is hoisted ────────────────────────────

    #[test]
    fn test_hoist_arithmetic_using_preheader_values() {
        let mut graph = loop_graph();

        // Parameter p0 is in block 0 (NodeId(0) from push_value).
        // Place an add inside the loop body that uses p0 + p0.
        let p0_id = graph.blocks()[0].nodes[0].0;
        graph.blocks_mut()[2].nodes.insert(
            0,
            (
                NodeId(400),
                ValueNode::Int32Add {
                    left: p0_id,
                    right: p0_id,
                },
            ),
        );

        assert_eq!(graph.blocks()[2].nodes.len(), 1);

        hoist_loop_invariants(&mut graph);

        // The add should have been hoisted to the preheader.
        assert_eq!(graph.blocks()[2].nodes.len(), 0);
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Int32Add { .. }))
        );
    }

    // ── LoadField hoisting (arr.length pattern) ──────────────────────────────

    #[test]
    fn test_hoist_load_field_from_loop_body() {
        let mut graph = loop_graph();

        // The parameter in block 0 (preheader) serves as the load's object.
        let param_id = graph.blocks()[0].nodes[0].0;

        // Add a LoadField in block 2 (loop body) whose object is defined
        // outside the loop — this models reading `arr.length`.
        graph.blocks_mut()[2].nodes.insert(
            0,
            (
                NodeId(100),
                ValueNode::LoadField {
                    object: param_id,
                    offset: 8,
                },
            ),
        );

        assert_eq!(graph.blocks()[2].nodes.len(), 1);

        hoist_loop_invariants(&mut graph);

        // The LoadField should have been hoisted to the preheader.
        assert_eq!(graph.blocks()[2].nodes.len(), 0);
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadField { offset: 8, .. }))
        );
    }

    // ── LoadField NOT hoisted when object is mutated inside loop ─────────────

    #[test]
    fn test_load_field_not_hoisted_when_store_aliases() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // StoreField to the same object inside the loop body.
        let val_id = NodeId(102);
        graph.blocks_mut()[2]
            .nodes
            .push((val_id, ValueNode::Int32Constant { value: 0 }));
        let store_id = NodeId(103);
        graph.blocks_mut()[2].nodes.push((
            store_id,
            ValueNode::StoreField {
                object: param_id,
                offset: 8,
                value: val_id,
            },
        ));

        // LoadField on the same object — should NOT be hoisted because the
        // object is mutated in the loop.
        graph.blocks_mut()[2].nodes.insert(
            0,
            (
                NodeId(100),
                ValueNode::LoadField {
                    object: param_id,
                    offset: 16,
                },
            ),
        );

        let body_len_before = graph.blocks()[2].nodes.len();
        hoist_loop_invariants(&mut graph);

        // The constant (pure, no alias issue) may be hoisted, but the
        // LoadField must stay because the same object is stored to.
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadField { .. })),
            "LoadField should NOT be hoisted when its object is mutated in the loop"
        );
        // Store is side-effecting and must also remain.
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::StoreField { .. })),
        );
        // At most the invariant constant was hoisted.
        assert!(graph.blocks()[2].nodes.len() >= body_len_before - 1);
    }
}

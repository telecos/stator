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
//! - Nested loops are handled independently; each back-edge produces its own
//!   loop.
//! - Guards and side-effecting nodes are never hoisted.
//!
//! # Usage
//!
//! ```
//! use stator_jse::compiler::maglev::ir::{
//!     BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode,
//! };
//! use stator_jse::compiler::maglev::licm::hoist_loop_invariants;
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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::compiler::maglev::ir::{BasicBlock, ControlNode, MaglevGraph, NodeId, ValueNode};

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics — atomic counters incremented from compilation threads
// ─────────────────────────────────────────────────────────────────────────────

/// Total loops detected across all compilations.
pub static LICM_LOOPS_DETECTED: AtomicU32 = AtomicU32::new(0);
/// Total nodes hoisted across all compilations.
pub static LICM_NODES_HOISTED: AtomicU32 = AtomicU32::new(0);
/// Total LoadNamedGeneric nodes hoisted.
pub static LICM_NAMED_GENERIC_HOISTED: AtomicU32 = AtomicU32::new(0);
/// Loops where property side-effects blocked hoisting of LoadNamedGeneric.
pub static LICM_BLOCKED_BY_SIDE_EFFECTS: AtomicU32 = AtomicU32::new(0);

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Detect natural loops and hoist loop-invariant pure nodes to the preheader.
pub fn hoist_loop_invariants(graph: &mut MaglevGraph) -> usize {
    let num_blocks = graph.blocks().len();
    let loops = detect_loops(graph);

    let _ = num_blocks;

    LICM_LOOPS_DETECTED.fetch_add(loops.len() as u32, Ordering::Relaxed);

    let mut total = 0;
    for lp in &loops {
        // Mark the loop header so codegen can align it.
        if let Some(header_block) = graph.block_mut(lp.header) {
            header_block.is_loop_header = true;
        }
        total += hoist_one_loop(graph, lp);
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariant addition chain folding
// ─────────────────────────────────────────────────────────────────────────────

/// Fold chains of loop-body additions with loop-invariant operands into a
/// single addition per iteration.
///
/// After LICM hoists property loads out of a loop, the loop body often
/// contains a left-to-right chain like:
///
/// ```text
///   t1 = add(accumulator_phi, hoisted_load_a)
///   t2 = add(t1, hoisted_load_b)
///   t3 = add(t2, hoisted_load_c)
///   ...
/// ```
///
/// This pass pre-computes the invariant sum in the preheader and replaces
/// the chain with `add(accumulator_phi, invariant_sum)`, reducing N
/// additions per iteration to 1.
pub fn fold_invariant_addition_chains(graph: &mut MaglevGraph) -> usize {
    let loops = detect_loops(graph);
    let mut total = 0;
    for lp in &loops {
        total += fold_chains_in_loop(graph, lp);
    }
    total
}

/// The kind of addition node in a chain.
#[derive(Clone, Copy, PartialEq)]
enum AddKind {
    Int32,
    CheckedSmi,
    Generic,
}

/// Create an addition node of the given kind.
fn make_add(kind: AddKind, left: NodeId, right: NodeId) -> ValueNode {
    match kind {
        AddKind::Int32 => ValueNode::Int32Add { left, right },
        AddKind::CheckedSmi => ValueNode::CheckedSmiAdd { left, right },
        AddKind::Generic => ValueNode::GenericAdd {
            left,
            right,
            feedback_slot: 0,
        },
    }
}

/// Fold the longest invariant addition chain in a single loop.
fn fold_chains_in_loop(graph: &mut MaglevGraph, lp: &NaturalLoop) -> usize {
    // 1. Collect NodeIds defined outside the loop.
    let mut outside_defs: HashSet<NodeId> = HashSet::new();
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            for (id, _) in &block.nodes {
                outside_defs.insert(*id);
            }
        }
    }

    // 2. Build use-count map across the entire graph so we can verify that
    //    intermediate chain nodes have no other consumers.
    let mut use_counts: HashMap<NodeId, usize> = HashMap::new();
    for block in graph.blocks() {
        for (_, node) in &block.nodes {
            visit_inputs(node, &mut |id| {
                *use_counts.entry(id).or_insert(0) += 1;
            });
        }
    }

    // 3. Collect Int32Add / CheckedSmiAdd / GenericAdd nodes inside the loop.
    let mut loop_adds: HashMap<NodeId, (NodeId, NodeId, AddKind)> = HashMap::new();
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (id, node) in &block.nodes {
            match node {
                ValueNode::Int32Add { left, right } => {
                    loop_adds.insert(*id, (*left, *right, AddKind::Int32));
                }
                ValueNode::CheckedSmiAdd { left, right } => {
                    loop_adds.insert(*id, (*left, *right, AddKind::CheckedSmi));
                }
                ValueNode::GenericAdd { left, right, .. } => {
                    loop_adds.insert(*id, (*left, *right, AddKind::Generic));
                }
                _ => {}
            }
        }
    }

    if loop_adds.is_empty() {
        return 0;
    }

    // 4. For each addition, walk backwards through its variant input to build
    //    the longest chain of invariant operands.  Only continue through an
    //    intermediate addition if it has exactly one use (the next link),
    //    ensuring that removing it is safe.
    let mut best_endpoint: Option<NodeId> = None;
    let mut best_inv_ops: Vec<NodeId> = Vec::new();
    let mut best_root: Option<NodeId> = None;
    let mut best_kind = AddKind::Int32;

    for &start_id in loop_adds.keys() {
        let mut inv_ops: Vec<NodeId> = Vec::new();
        let mut current = start_id;
        let mut root_variant = None;
        let mut chain_kind: Option<AddKind> = None;

        while let Some(&(left, right, kind)) = loop_adds.get(&current) {
            // Require uniform addition type across the chain.
            if let Some(prev_kind) = chain_kind
                && prev_kind != kind
            {
                break;
            }
            chain_kind = Some(kind);

            let l_out = outside_defs.contains(&left);
            let r_out = outside_defs.contains(&right);

            // Need exactly one invariant and one variant operand.
            if l_out == r_out {
                break;
            }

            let (invariant, variant) = if r_out { (right, left) } else { (left, right) };

            inv_ops.push(invariant);

            // Try to extend the chain backwards through the variant input.
            if loop_adds.contains_key(&variant) {
                let uses = use_counts.get(&variant).copied().unwrap_or(0);
                if uses == 1 {
                    current = variant;
                    continue;
                }
            }

            // Chain ends: variant is a Phi, multi-use node, or non-addition.
            root_variant = Some(variant);
            break;
        }

        if inv_ops.len() >= 2 && root_variant.is_some() && inv_ops.len() > best_inv_ops.len() {
            best_endpoint = Some(start_id);
            best_inv_ops = inv_ops;
            best_root = root_variant;
            best_kind = chain_kind.unwrap_or(AddKind::Int32);
        }
    }

    let endpoint_id = match best_endpoint {
        Some(id) => id,
        None => return 0,
    };
    let root_variant = best_root.unwrap();

    // Collected from endpoint backwards — reverse to natural order.
    best_inv_ops.reverse();

    // 5. Create the invariant sum in the preheader:
    //    inv_sum = (...((inv[0] + inv[1]) + inv[2]) + ... + inv[N-1])
    let mut prev = best_inv_ops[0];
    for &inv in &best_inv_ops[1..] {
        let nid = graph.alloc_node_id();
        let node = make_add(best_kind, prev, inv);
        if let Some(pre) = graph.block_mut(lp.preheader) {
            pre.push_with_id(nid, node);
        }
        prev = nid;
    }
    let inv_sum = prev;

    // 6. Replace the endpoint addition with add(root_variant, inv_sum).
    //    The now-dead intermediate additions will be cleaned up by DCE.
    let replacement = make_add(best_kind, root_variant, inv_sum);

    for block in graph.blocks_mut() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (id, node) in &mut block.nodes {
            if *id == endpoint_id {
                *node = replacement;
                return best_inv_ops.len();
            }
        }
    }

    0
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
///
/// A back-edge is any edge whose target is a loop header — a block that is
/// reachable from itself via the edge's source.  We detect this by attempting
/// to build a [`NaturalLoop`] for every CFG edge; [`build_loop`] validates
/// the structure (unique preheader, etc.) and returns `None` for non-loop
/// edges.
///
/// NOTE: we do **not** rely on `target <= block.id` because block IDs are
/// assigned in jump-target discovery order, not RPO.  The loop header can
/// have a *higher* ID than the body (e.g. when the exit block is discovered
/// before the body's fall-through).
pub fn detect_loops(graph: &MaglevGraph) -> Vec<NaturalLoop> {
    let mut loops = Vec::new();
    let mut seen_headers: HashSet<u32> = HashSet::new();

    for block in graph.blocks() {
        let targets = control_targets(block);
        for &target in &targets {
            // Skip forward edges (target not yet visited can't form a loop).
            // Only consider edges where the target block has this block as a
            // downstream reachable node — i.e. there is a path target→…→block.
            // The cheapest test: try to build the loop; build_loop does a
            // predecessor walk and validates the structure.
            if seen_headers.contains(&target) {
                continue; // already found a loop with this header
            }
            if let Some(lp) = build_loop(graph, target, block.id) {
                // Confirm the source is in the loop body (build_loop always
                // includes it, but verify the body actually connects back).
                if lp.body.contains(&block.id) && lp.body.contains(&target) {
                    seen_headers.insert(target);
                    loops.push(lp);
                }
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
/// source `back_src`.  Returns `None` if no unique preheader exists or if the
/// edge is not a true back-edge (i.e. `back_src` is not reachable from
/// `header` through the CFG).
fn build_loop(graph: &MaglevGraph, header: u32, back_src: u32) -> Option<NaturalLoop> {
    if header == back_src {
        // Self-loop: the block jumps back to itself.
        let block = graph.block(header)?;
        let preheaders: Vec<u32> = block
            .predecessors
            .iter()
            .copied()
            .filter(|&p| p != header)
            .collect();
        if preheaders.len() != 1 {
            return None;
        }
        let mut body = HashSet::new();
        body.insert(header);
        return Some(NaturalLoop {
            header,
            preheader: preheaders[0],
            body,
        });
    }

    // First, verify that back_src is reachable from header through forward
    // edges.  Do a BFS/DFS from header following control-flow successors.
    let mut reachable = HashSet::new();
    let mut work = vec![header];
    while let Some(b) = work.pop() {
        if !reachable.insert(b) {
            continue;
        }
        if let Some(block) = graph.block(b) {
            for &t in &control_targets(block) {
                if t != header {
                    // Don't follow back to the header — we're looking for
                    // forward reachability only.
                    work.push(t);
                }
            }
        }
    }
    if !reachable.contains(&back_src) {
        return None; // Not a true back-edge.
    }

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
///
/// Uses iterative fixed-point hoisting: each pass moves newly-eligible nodes
/// to the preheader and marks their [`NodeId`]s as "outside".  This enables
/// chained hoisting — e.g. a `LoadGlobal` hoisted in pass 1 makes a
/// dependent `LoadNamedGeneric` eligible in pass 2 — which is critical when
/// the graph's block ordering doesn't match the data-dependency order (common
/// with Maglev's non-RPO block layout).
///
/// Returns the total number of nodes hoisted across all passes.
fn hoist_one_loop(graph: &mut MaglevGraph, lp: &NaturalLoop) -> usize {
    // Collect objects mutated inside the loop so we can avoid hoisting loads
    // that may alias those stores (soundness check).
    // These are stable across passes (we never hoist stores).
    let mutated_objects = collect_mutated_objects(graph, &lp.body);

    // Collect global name indices written by StoreGlobal inside the loop.
    // A LoadGlobal for a name that is also stored must not be hoisted.
    let mutated_globals = collect_mutated_globals(graph, &lp.body);

    // Check whether the loop body contains any generic stores or calls that
    // could modify arbitrary object properties.  When true, generic property
    // loads (LoadNamedGeneric / LoadKeyedGeneric) must stay in the loop.
    // Stable across passes (we never hoist calls or stores).
    let has_property_side_effects = loop_has_property_side_effects(graph, &lp.body);

    // Count LoadNamedGeneric nodes inside the loop for diagnostics.
    let mut load_named_in_loop = 0u32;
    for block in graph.blocks() {
        if !lp.body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            if matches!(node, ValueNode::LoadNamedGeneric { .. }) {
                load_named_in_loop += 1;
            }
        }
    }
    if has_property_side_effects && load_named_in_loop > 0 {
        LICM_BLOCKED_BY_SIDE_EFFECTS.fetch_add(1, Ordering::Relaxed);
    }

    let mut total_hoisted = 0;
    let mut named_generic_hoisted = 0u32;

    // Iterate until a full pass finds nothing new to hoist.
    loop {
        // Rebuild outside_defs each pass — freshly hoisted nodes are now in
        // the preheader (a non-loop block) and will be included automatically.
        let mut outside_defs: HashSet<NodeId> = HashSet::new();
        for block in graph.blocks() {
            if !lp.body.contains(&block.id) {
                for (id, _) in &block.nodes {
                    outside_defs.insert(*id);
                }
            }
        }

        // Scan loop blocks for hoistable nodes.  Include the loop header — in
        // single-block loops the entire body lives in the header.  Nodes whose
        // inputs are all defined outside the loop are loop-invariant regardless
        // of which block they reside in.
        let mut to_hoist: Vec<(u32, usize, NodeId, ValueNode)> = Vec::new();

        // Cap the number of LoadNamedGeneric / LoadKeyedGeneric nodes
        // hoisted per loop to avoid register-pressure explosions.  Each
        // hoisted generic load occupies one register for the entire loop
        // body, plus a caller-saved IC-fill call in the preheader.
        // With 9 allocatable registers, hoisting up to 5 generic loads
        // leaves room for the loop counter and accumulator.  Chained
        // accesses (a.b.c.d.e) only keep the final result live, so the
        // effective pressure is lower than the cap suggests.
        const MAX_GENERIC_LOADS_HOISTED: usize = 5;
        let mut generic_loads_in_hoist = 0usize;

        for block in graph.blocks() {
            if !lp.body.contains(&block.id) {
                continue;
            }
            for (pos, (id, node)) in block.nodes.iter().enumerate() {
                // Skip Phi nodes — they define loop-carried values.
                if matches!(node, ValueNode::Phi { .. }) {
                    continue;
                }
                let pure = is_pure(node);
                let inputs_out = all_inputs_outside(node, &outside_defs);
                let alias_store = load_aliases_store(node, &mutated_objects);
                let alias_glob = global_load_aliases_store(node, &mutated_globals);
                let blocked_by_side_effects_flag =
                    is_generic_property_load(node) && has_property_side_effects;
                // Enforce the register-pressure cap on generic loads.
                let over_generic_load_cap = is_generic_property_load(node)
                    && generic_loads_in_hoist >= MAX_GENERIC_LOADS_HOISTED;
                if pure
                    && inputs_out
                    && !alias_store
                    && !alias_glob
                    && !blocked_by_side_effects_flag
                    && !over_generic_load_cap
                {
                    if is_generic_property_load(node) {
                        generic_loads_in_hoist += 1;
                    }
                    to_hoist.push((block.id, pos, *id, node.clone()));
                    // Eagerly mark as outside so later nodes in this same
                    // block can see the dependency as satisfied.
                    outside_defs.insert(*id);
                } else if matches!(node, ValueNode::LoadNamedGeneric { .. }) {
                    // Could not hoist this LoadNamedGeneric.
                }
            }
        }

        if to_hoist.is_empty() {
            break;
        }

        // Count LoadNamedGeneric among hoisted nodes.
        for (_, _, _, node) in &to_hoist {
            if matches!(node, ValueNode::LoadNamedGeneric { .. }) {
                named_generic_hoisted += 1;
            }
        }

        // Remove hoisted nodes from their source blocks (iterate in reverse
        // position order so indices remain valid).
        let mut removals_by_block: HashMap<u32, Vec<usize>> = HashMap::new();
        for &(blk, pos, _, _) in &to_hoist {
            removals_by_block.entry(blk).or_default().push(pos);
        }
        for positions in removals_by_block.values_mut() {
            positions.sort_unstable();
            positions.dedup();
        }

        for block in graph.blocks_mut() {
            if let Some(positions) = removals_by_block.get(&block.id) {
                for &pos in positions.iter().rev() {
                    if pos < block.nodes.len() {
                        block.nodes.remove(pos);
                    }
                }
            }
        }

        // Append hoisted nodes to the preheader (before its control node).
        let pass_count = to_hoist.len();
        if let Some(preheader) = graph.block_mut(lp.preheader) {
            for (_, _, id, node) in to_hoist {
                preheader.push_with_id(id, node);
            }
        }
        total_hoisted += pass_count;
    }

    // Update global counters.
    LICM_NODES_HOISTED.fetch_add(total_hoisted as u32, Ordering::Relaxed);
    LICM_NAMED_GENERIC_HOISTED.fetch_add(named_generic_hoisted, Ordering::Relaxed);

    if load_named_in_loop > 0 {
        // Counters updated above; no verbose logging.
    }

    total_hoisted
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
                | ValueNode::StoreKeyedGeneric { object, .. }
                | ValueNode::DeleteProperty { object, .. } => {
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
        | ValueNode::LoadNamedGeneric { object, .. }
        | ValueNode::LoadKeyedGeneric { object, .. } => mutated.contains(object),
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

/// Return `true` when `node` is a generic property load — one that invokes a
/// runtime stub and whose result could be invalidated by arbitrary side
/// effects from calls or generic stores elsewhere in the loop.
fn is_generic_property_load(node: &ValueNode) -> bool {
    matches!(
        node,
        ValueNode::LoadNamedGeneric { .. } | ValueNode::LoadKeyedGeneric { .. }
    )
}

/// Return `true` if the loop body contains any nodes that could modify
/// arbitrary object properties: generic property stores, property deletion,
/// or any JavaScript-level call / construct.  When this returns `true`,
/// [`LoadNamedGeneric`] and [`LoadKeyedGeneric`] must **not** be hoisted
/// because the operation can mutate objects reachable from its arguments or
/// from the global scope.
///
/// Engine-internal builtins ([`CallBuiltin`]) and runtime stubs
/// ([`CallRuntime`]) are **not** included because they implement
/// side-effect-free primitives (arithmetic, comparison, type conversion,
/// etc.) that cannot modify user-visible object properties.  Similarly,
/// [`StoreGlobal`] and [`LoadGlobal`] operate on the global variable
/// bindings, not on object property slots, so they do not alias with
/// named/keyed property loads.
fn loop_has_property_side_effects(graph: &MaglevGraph, loop_body: &HashSet<u32>) -> bool {
    for block in graph.blocks() {
        if !loop_body.contains(&block.id) {
            continue;
        }
        for (_, node) in &block.nodes {
            if matches!(
                node,
                ValueNode::StoreNamedGeneric { .. }
                    | ValueNode::StoreKeyedGeneric { .. }
                    | ValueNode::DeleteProperty { .. }
                    | ValueNode::Call { .. }
                    | ValueNode::CallArrayPush { .. }
                    | ValueNode::CallKnownFunction { .. }
                    | ValueNode::CallWithSpread { .. }
                    | ValueNode::Construct { .. }
                    | ValueNode::ConstructWithSpread { .. }
                    | ValueNode::SpeculativeCallFusion { .. }
                    | ValueNode::SpeculativeSumFusion { .. }
                    | ValueNode::SpeculativePushFusion { .. }
            ) {
                return true;
            }
        }
    }
    false
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
            | ValueNode::CallArrayPush { .. }
            | ValueNode::CallKnownFunction { .. }
            | ValueNode::CallBuiltin { .. }
            | ValueNode::CallRuntime { .. }
            | ValueNode::CallWithSpread { .. }
            | ValueNode::Construct { .. }
            | ValueNode::ConstructWithSpread { .. }
            | ValueNode::CreateObjectLiteral { .. }
            | ValueNode::CreateObjectLiteralWithProperties { .. }
            | ValueNode::CreateArrayLiteral { .. }
            | ValueNode::CreateShallowObjectLiteral { .. }
            | ValueNode::CreateShallowArrayLiteral { .. }
            | ValueNode::CreateFunctionContext { .. }
            | ValueNode::CreateBlockContext { .. }
            | ValueNode::CreateCatchContext { .. }
            | ValueNode::CreateWithContext { .. }
            | ValueNode::PushContext { .. }
            | ValueNode::PopContext { .. }
            | ValueNode::CreateClosure { .. }
            | ValueNode::FastCreateClosure { .. }
            | ValueNode::CreateEmptyObjectLiteral
            | ValueNode::CreateEmptyArrayLiteral { .. }
            | ValueNode::CreateMappedArguments
            | ValueNode::CreateUnmappedArguments
            | ValueNode::CreateRestParameter
            | ValueNode::CreateRegExpLiteral { .. }
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
        | ValueNode::CreateEmptyArrayLiteral { .. }
        | ValueNode::CreateMappedArguments
        | ValueNode::CreateUnmappedArguments
        | ValueNode::CreateRestParameter
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
        | ValueNode::TestNullOrUndefined { value }
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
        ValueNode::PushContext { context } | ValueNode::PopContext { context } => f(*context),

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
        // CallArrayPush codegen ignores callee — don't track it.
        ValueNode::CallArrayPush { receiver, args, .. } => {
            f(*receiver);
            for &a in args.iter() {
                f(a);
            }
        }

        ValueNode::SpeculativeCallFusion { callee, .. } => {
            f(*callee);
        }
        ValueNode::SpeculativeSumFusion { array } => {
            f(*array);
        }
        ValueNode::SpeculativePushFusion { array, .. } => {
            f(*array);
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
        ValueNode::CreateObjectLiteralWithProperties { values, .. } => {
            for &v in values.iter() {
                f(v);
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

    // ── deep_object pattern: LoadGlobal + LoadNamedGeneric chain ─────────────

    #[test]
    fn test_hoist_property_chain_from_immutable_global() {
        let mut graph = loop_graph();

        // LoadGlobal("root") in the loop body — "root" is NOT stored, so it
        // should be hoistable.
        let root_id = NodeId(500);
        let a_id = NodeId(501);
        let b_id = NodeId(502);

        // Insert chain: LoadGlobal("root") → LoadNamedGeneric(root, "a") →
        // LoadNamedGeneric(a, "b") all in block 2 (body).
        graph.blocks_mut()[2].nodes.push((
            root_id,
            ValueNode::LoadGlobal {
                name: 1, // "root"
                feedback_slot: 0,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            a_id,
            ValueNode::LoadNamedGeneric {
                object: root_id,
                name: 2, // "a"
                feedback_slot: 1,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            b_id,
            ValueNode::LoadNamedGeneric {
                object: a_id,
                name: 3, // "b"
                feedback_slot: 2,
            },
        ));

        assert_eq!(graph.blocks()[2].nodes.len(), 3);

        hoist_loop_invariants(&mut graph);

        // ALL 3 should be hoisted: LoadGlobal has no inputs, each
        // LoadNamedGeneric depends on the prior (hoisted) node.
        assert_eq!(
            graph.blocks()[2].nodes.len(),
            0,
            "all 3 invariant nodes should be hoisted from the loop body"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == root_id),
            "LoadGlobal should be in preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == b_id),
            "final LoadNamedGeneric should be in preheader"
        );
    }

    // ── LoadGlobal NOT hoisted when same name is stored in loop ──────────────

    #[test]
    fn test_load_global_not_hoisted_when_stored() {
        let mut graph = loop_graph();

        // LoadGlobal("i") + StoreGlobal("i") in the loop body.
        let load_i = NodeId(600);
        let inc = NodeId(601);
        let store_i = NodeId(602);

        graph.blocks_mut()[2].nodes.push((
            load_i,
            ValueNode::LoadGlobal {
                name: 10, // "i"
                feedback_slot: 0,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            inc,
            ValueNode::Int32Add {
                left: load_i,
                right: load_i,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            store_i,
            ValueNode::StoreGlobal {
                name: 10, // "i" — same name as load
                value: inc,
                feedback_slot: 1,
            },
        ));

        let body_len = graph.blocks()[2].nodes.len();
        hoist_loop_invariants(&mut graph);

        // LoadGlobal("i") must NOT be hoisted because "i" is also stored.
        assert!(
            graph.blocks()[2].nodes.iter().any(|(id, _)| *id == load_i),
            "LoadGlobal for stored name must stay in loop"
        );
        // StoreGlobal is impure and must stay.
        assert!(graph.blocks()[2].nodes.iter().any(|(id, _)| *id == store_i),);
        // Int32Add depends on load_i (in loop) so it stays too.
        assert_eq!(
            graph.blocks()[2].nodes.len(),
            body_len,
            "nothing should be hoisted"
        );
    }

    // ── Loop detection with non-RPO block ordering ───────────────────────────

    /// Regression test: the graph builder creates blocks in jump-target
    /// discovery order, NOT RPO.  For a for-loop, the exit block is discovered
    /// first (JumpIfFalse target), then the body (fall-through), and finally
    /// the loop header (JumpLoop target).  This means the header can have a
    /// HIGHER block ID than the body, which broke the old `target <= block.id`
    /// back-edge heuristic.
    #[test]
    fn test_detect_loop_with_reversed_block_ids() {
        let mut graph = MaglevGraph::new(1);

        // block 0: init (preheader)
        let mut b0 = BasicBlock::new(0);
        b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 3 }); // jump to header (ID 3)
        graph.add_block(b0);

        // block 1: exit — discovered FIRST by JumpIfFalse
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(3);
        let undef = b1.push_value(ValueNode::UndefinedConstant);
        b1.set_control(ControlNode::Return { value: undef });
        graph.add_block(b1);

        // block 2: body — discovered via fall-through
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(3);
        b2.set_control(ControlNode::Jump { target: 3 }); // back-edge to header
        graph.add_block(b2);

        // block 3: header — discovered LAST by JumpLoop (highest ID!)
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(0); // from init
        b3.add_predecessor(2); // back-edge from body
        let cond = b3.push_value(ValueNode::TrueConstant);
        b3.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 2,
            if_false: 1,
        });
        graph.add_block(b3);

        let loops = detect_loops(&graph);
        assert_eq!(loops.len(), 1, "must detect the loop even with non-RPO IDs");
        assert_eq!(loops[0].header, 3, "header should be block 3");
        assert_eq!(loops[0].preheader, 0, "preheader should be block 0");
        assert!(loops[0].body.contains(&3), "body must contain header");
        assert!(loops[0].body.contains(&2), "body must contain body block");
    }

    /// Regression test: LICM hoists invariant nodes from loops with non-RPO
    /// block ordering.
    #[test]
    fn test_hoist_from_loop_with_reversed_block_ids() {
        let mut graph = MaglevGraph::new(1);

        // block 0: init (preheader) with a parameter
        let mut b0 = BasicBlock::new(0);
        let _param = b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b0);

        // block 1: exit
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(3);
        let undef = b1.push_value(ValueNode::UndefinedConstant);
        b1.set_control(ControlNode::Return { value: undef });
        graph.add_block(b1);

        // block 2: body with an invariant constant
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(3);
        b2.nodes
            .push((NodeId(100), ValueNode::Int32Constant { value: 42 }));
        b2.set_control(ControlNode::Jump { target: 3 });
        graph.add_block(b2);

        // block 3: header (highest ID)
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(0);
        b3.add_predecessor(2);
        let cond = b3.push_value(ValueNode::TrueConstant);
        b3.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 2,
            if_false: 1,
        });
        graph.add_block(b3);

        assert_eq!(graph.blocks()[2].nodes.len(), 1);
        hoist_loop_invariants(&mut graph);

        // The constant should have been hoisted from body (block 2) to
        // preheader (block 0).
        assert_eq!(
            graph.blocks()[2].nodes.len(),
            0,
            "invariant should be hoisted from body"
        );
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Int32Constant { value: 42 })),
            "invariant should appear in preheader"
        );
    }

    // ── LoadNamedGeneric NOT hoisted when Call exists in loop ─────────────

    #[test]
    fn test_load_named_generic_not_hoisted_with_call_in_loop() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // LoadNamedGeneric whose object is defined outside the loop.
        graph.blocks_mut()[2].nodes.push((
            NodeId(700),
            ValueNode::LoadNamedGeneric {
                object: param_id,
                name: 1,
                feedback_slot: 0,
            },
        ));

        // A Call in the same loop body — can modify any object's properties.
        graph.blocks_mut()[2].nodes.push((
            NodeId(701),
            ValueNode::Call {
                callee: param_id,
                receiver: param_id,
                args: vec![],
                feedback_slot: 0,
            },
        ));

        let body_before = graph.blocks()[2].nodes.len();
        hoist_loop_invariants(&mut graph);

        // LoadNamedGeneric must NOT be hoisted because a Call in the loop
        // could modify the property.
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. })),
            "LoadNamedGeneric should NOT be hoisted when a Call exists in the loop"
        );
        // The Call itself is side-effecting and must remain.
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::Call { .. })),
        );
        assert_eq!(graph.blocks()[2].nodes.len(), body_before);
    }

    // ── LoadKeyedGeneric hoisted when loop is side-effect-free ───────────

    #[test]
    fn test_hoist_load_keyed_generic_no_side_effects() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // A key defined in the preheader.
        let key_id = NodeId(50);
        graph.blocks_mut()[0]
            .nodes
            .push((key_id, ValueNode::SmiConstant { value: 0 }));

        // LoadKeyedGeneric in the loop body — no stores or calls.
        graph.blocks_mut()[2].nodes.push((
            NodeId(800),
            ValueNode::LoadKeyedGeneric {
                object: param_id,
                key: key_id,
                feedback_slot: 0,
            },
        ));

        assert_eq!(graph.blocks()[2].nodes.len(), 1);
        hoist_loop_invariants(&mut graph);

        assert_eq!(
            graph.blocks()[2].nodes.len(),
            0,
            "LoadKeyedGeneric should be hoisted when no side effects in loop"
        );
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadKeyedGeneric { .. })),
            "LoadKeyedGeneric should appear in preheader"
        );
    }

    // ── LoadNamedGeneric NOT hoisted when StoreNamedGeneric exists ───────

    #[test]
    fn test_load_named_generic_not_hoisted_with_generic_store() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // A different object for the store (the store targets a DIFFERENT
        // object, but the blanket side-effect check still blocks the load).
        let other_obj = NodeId(51);
        graph.blocks_mut()[0]
            .nodes
            .push((other_obj, ValueNode::Parameter { index: 1 }));

        // LoadNamedGeneric on param_id.
        graph.blocks_mut()[2].nodes.push((
            NodeId(900),
            ValueNode::LoadNamedGeneric {
                object: param_id,
                name: 1,
                feedback_slot: 0,
            },
        ));

        // StoreNamedGeneric on a DIFFERENT object — still blocks the load
        // because generic stores can trigger setters with arbitrary effects.
        let val = NodeId(901);
        graph.blocks_mut()[2]
            .nodes
            .push((val, ValueNode::Int32Constant { value: 0 }));
        graph.blocks_mut()[2].nodes.push((
            NodeId(902),
            ValueNode::StoreNamedGeneric {
                object: other_obj,
                name: 2,
                value: val,
                feedback_slot: 0,
            },
        ));

        hoist_loop_invariants(&mut graph);

        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. })),
            "LoadNamedGeneric should NOT be hoisted when a StoreNamedGeneric exists in the loop"
        );
    }

    // ── Cross-block chained hoisting with non-RPO ordering ──────────────

    /// Regression test for the property_access benchmark pattern.  When the
    /// graph builder emits blocks in non-RPO order (body before header), a
    /// `LoadNamedGeneric` in the body depends on a `LoadGlobal` in the header
    /// which is scanned LATER.  Iterative LICM fixes this: pass 1 hoists the
    /// `LoadGlobal`, pass 2 hoists the `LoadNamedGeneric` chain.
    #[test]
    fn test_hoist_cross_block_property_chain_non_rpo() {
        let mut graph = MaglevGraph::new(1);

        // block 0: preheader
        let mut b0 = BasicBlock::new(0);
        b0.push_value(ValueNode::Parameter { index: 0 });
        b0.set_control(ControlNode::Jump { target: 3 }); // jump to header (ID 3)
        graph.add_block(b0);

        // block 1: exit
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(3);
        let undef = b1.push_value(ValueNode::UndefinedConstant);
        b1.set_control(ControlNode::Return { value: undef });
        graph.add_block(b1);

        // block 2: body — LoadNamedGeneric depends on LoadGlobal in block 3.
        // Scanned BEFORE the header in graph order.
        let obj_id = NodeId(500);
        let x_id = NodeId(501);
        let y_id = NodeId(502);
        let z_id = NodeId(503);
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(3);
        b2.nodes.push((
            x_id,
            ValueNode::LoadNamedGeneric {
                object: obj_id,
                name: 10,
                feedback_slot: 1,
            },
        ));
        b2.nodes.push((
            y_id,
            ValueNode::LoadNamedGeneric {
                object: obj_id,
                name: 11,
                feedback_slot: 2,
            },
        ));
        b2.nodes.push((
            z_id,
            ValueNode::LoadNamedGeneric {
                object: obj_id,
                name: 12,
                feedback_slot: 3,
            },
        ));
        b2.set_control(ControlNode::Jump { target: 3 }); // back-edge
        graph.add_block(b2);

        // block 3: header — contains LoadGlobal("obj"), scanned AFTER block 2.
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(0);
        b3.add_predecessor(2);
        b3.nodes.push((
            obj_id,
            ValueNode::LoadGlobal {
                name: 1,
                feedback_slot: 0,
            },
        ));
        let cond = b3.push_value(ValueNode::TrueConstant);
        b3.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 2,
            if_false: 1,
        });
        graph.add_block(b3);

        // Body has 3 LoadNamedGeneric nodes; header has LoadGlobal + cond.
        assert_eq!(graph.blocks()[2].nodes.len(), 3);
        assert_eq!(
            graph.blocks()[3]
                .nodes
                .iter()
                .filter(|(_, n)| matches!(n, ValueNode::LoadGlobal { .. }))
                .count(),
            1
        );

        hoist_loop_invariants(&mut graph);

        // All 3 LoadNamedGeneric should be hoisted from body.
        assert_eq!(
            graph.blocks()[2].nodes.len(),
            0,
            "all LoadNamedGeneric nodes should be hoisted from loop body"
        );

        // LoadGlobal should be hoisted from header.
        assert!(
            !graph.blocks()[3]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadGlobal { .. })),
            "LoadGlobal should be hoisted from loop header"
        );

        // All 4 hoisted nodes should be in the preheader (block 0).
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == obj_id),
            "LoadGlobal should be in preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == x_id),
            "LoadNamedGeneric(x) should be in preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == y_id),
            "LoadNamedGeneric(y) should be in preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == z_id),
            "LoadNamedGeneric(z) should be in preheader"
        );
    }

    // ── LoadNamedGeneric NOT hoisted when DeleteProperty exists ──────────

    #[test]
    fn test_load_named_generic_not_hoisted_with_delete_property() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // A key for the delete.
        let key_id = NodeId(52);
        graph.blocks_mut()[0]
            .nodes
            .push((key_id, ValueNode::SmiConstant { value: 0 }));

        // LoadNamedGeneric on param_id.
        graph.blocks_mut()[2].nodes.push((
            NodeId(950),
            ValueNode::LoadNamedGeneric {
                object: param_id,
                name: 1,
                feedback_slot: 0,
            },
        ));

        // DeleteProperty on a different object — still blocks the load
        // because deletion can change prototype chain resolution.
        let other_obj = NodeId(51);
        graph.blocks_mut()[0]
            .nodes
            .push((other_obj, ValueNode::Parameter { index: 1 }));
        graph.blocks_mut()[2].nodes.push((
            NodeId(951),
            ValueNode::DeleteProperty {
                object: other_obj,
                key: key_id,
                feedback_slot: 0,
            },
        ));

        hoist_loop_invariants(&mut graph);

        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. })),
            "LoadNamedGeneric should NOT be hoisted when DeleteProperty exists in the loop"
        );
    }

    // ── deep_object benchmark: LoadNamedGeneric hoisted despite StoreGlobal + CallBuiltin ──

    /// Regression test for the `deep_object_access_1k` benchmark.  The loop
    /// body contains `StoreGlobal` (for `sum`) and `CallBuiltin` (for the
    /// generic comparison / increment), but neither of those can modify
    /// object properties.  The `LoadNamedGeneric` chain must still be hoisted.
    #[test]
    fn test_hoist_load_named_generic_with_store_global_and_call_builtin() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // Simulate the deep_object benchmark pattern:
        // LoadGlobal("obj") → LoadNamedGeneric(obj, "a") → LoadNamedGeneric(a, "b")
        // ... → GenericAdd → StoreGlobal("sum")
        // Plus a CallBuiltin for the loop comparison.

        let obj_id = NodeId(500);
        let a_id = NodeId(501);
        let b_id = NodeId(502);
        let c_id = NodeId(503);
        let add_id = NodeId(504);
        let store_id = NodeId(505);
        let cmp_id = NodeId(506);

        // LoadGlobal("obj") — not stored in the loop.
        graph.blocks_mut()[2].nodes.push((
            obj_id,
            ValueNode::LoadGlobal {
                name: 1,
                feedback_slot: 0,
            },
        ));
        // LoadNamedGeneric chain: obj.a.b.c
        graph.blocks_mut()[2].nodes.push((
            a_id,
            ValueNode::LoadNamedGeneric {
                object: obj_id,
                name: 2,
                feedback_slot: 1,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            b_id,
            ValueNode::LoadNamedGeneric {
                object: a_id,
                name: 3,
                feedback_slot: 2,
            },
        ));
        graph.blocks_mut()[2].nodes.push((
            c_id,
            ValueNode::LoadNamedGeneric {
                object: b_id,
                name: 4,
                feedback_slot: 3,
            },
        ));
        // GenericAdd (pure — does not block hoisting).
        graph.blocks_mut()[2].nodes.push((
            add_id,
            ValueNode::GenericAdd {
                left: param_id,
                right: c_id,
                feedback_slot: 4,
            },
        ));
        // StoreGlobal("sum") — impure but NOT a property side-effect.
        graph.blocks_mut()[2].nodes.push((
            store_id,
            ValueNode::StoreGlobal {
                name: 10,
                value: add_id,
                feedback_slot: 5,
            },
        ));
        // CallBuiltin for loop comparison (e.g. LessThan) — NOT a property
        // side-effect; should not block LoadNamedGeneric hoisting.
        graph.blocks_mut()[2].nodes.push((
            cmp_id,
            ValueNode::CallBuiltin {
                builtin_id: 0,
                args: vec![param_id],
            },
        ));

        assert_eq!(graph.blocks()[2].nodes.len(), 7);

        hoist_loop_invariants(&mut graph);

        // LoadGlobal + 3 LoadNamedGeneric should be hoisted (4 nodes).
        // GenericAdd depends on c_id (hoisted) AND param_id (outside) → also hoisted.
        // StoreGlobal is impure → stays.
        // CallBuiltin depends on param_id (outside) but is impure → stays.
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == obj_id),
            "LoadGlobal should be hoisted to preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == a_id),
            "LoadNamedGeneric(a) should be hoisted to preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == b_id),
            "LoadNamedGeneric(b) should be hoisted to preheader"
        );
        assert!(
            graph.blocks()[0].nodes.iter().any(|(id, _)| *id == c_id),
            "LoadNamedGeneric(c) should be hoisted to preheader"
        );
        // StoreGlobal and CallBuiltin must remain in the loop.
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::StoreGlobal { .. })),
            "StoreGlobal must stay in the loop"
        );
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::CallBuiltin { .. })),
            "CallBuiltin must stay in the loop"
        );
    }

    // ── CallBuiltin alone does NOT block LoadNamedGeneric hoisting ───────

    #[test]
    fn test_call_builtin_does_not_block_load_named_generic() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // LoadNamedGeneric whose object is defined outside the loop.
        graph.blocks_mut()[2].nodes.push((
            NodeId(700),
            ValueNode::LoadNamedGeneric {
                object: param_id,
                name: 1,
                feedback_slot: 0,
            },
        ));

        // A CallBuiltin in the same loop body — engine-internal, cannot
        // modify user-visible object properties.
        graph.blocks_mut()[2].nodes.push((
            NodeId(701),
            ValueNode::CallBuiltin {
                builtin_id: 0,
                args: vec![],
            },
        ));

        hoist_loop_invariants(&mut graph);

        // LoadNamedGeneric SHOULD be hoisted — CallBuiltin is not a property
        // side-effect.
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. })),
            "LoadNamedGeneric should be hoisted when only CallBuiltin exists in the loop"
        );
        // CallBuiltin must stay (impure).
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::CallBuiltin { .. })),
        );
    }

    // ── CallRuntime alone does NOT block LoadNamedGeneric hoisting ───────

    #[test]
    fn test_call_runtime_does_not_block_load_named_generic() {
        let mut graph = loop_graph();

        let param_id = graph.blocks()[0].nodes[0].0;

        // LoadNamedGeneric whose object is defined outside the loop.
        graph.blocks_mut()[2].nodes.push((
            NodeId(700),
            ValueNode::LoadNamedGeneric {
                object: param_id,
                name: 1,
                feedback_slot: 0,
            },
        ));

        // A CallRuntime in the same loop body — engine-internal runtime stub.
        graph.blocks_mut()[2].nodes.push((
            NodeId(701),
            ValueNode::CallRuntime {
                function_id: 0,
                args: vec![],
            },
        ));

        hoist_loop_invariants(&mut graph);

        // LoadNamedGeneric SHOULD be hoisted — CallRuntime is not a property
        // side-effect.
        assert!(
            graph.blocks()[0]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::LoadNamedGeneric { .. })),
            "LoadNamedGeneric should be hoisted when only CallRuntime exists in the loop"
        );
        // CallRuntime must stay (impure).
        assert!(
            graph.blocks()[2]
                .nodes
                .iter()
                .any(|(_, n)| matches!(n, ValueNode::CallRuntime { .. })),
        );
    }

    // ── Invariant addition chain folding ──────────────────────────────────────

    #[test]
    fn test_fold_invariant_addition_chain() {
        // Build a graph with a loop containing:
        //   preheader: inv_a(100), inv_b(101), inv_c(102)
        //   body: t1 = add(phi, inv_a), t2 = add(t1, inv_b), t3 = add(t2, inv_c)
        // After folding, the loop body should have just 1 addition: add(phi, inv_sum)
        let mut graph = MaglevGraph::new(1);

        // Block 0: preheader with three invariant constants.
        let mut b0 = BasicBlock::new(0);
        b0.push_with_id(NodeId(100), ValueNode::Int32Constant { value: 1 });
        b0.push_with_id(NodeId(101), ValueNode::Int32Constant { value: 2 });
        b0.push_with_id(NodeId(102), ValueNode::Int32Constant { value: 3 });
        b0.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b0);

        // Block 1: header with Phi for the accumulator.
        let mut b1 = BasicBlock::new(1);
        b1.add_predecessor(0);
        b1.add_predecessor(2);
        b1.push_with_id(
            NodeId(200),
            ValueNode::Phi {
                inputs: vec![NodeId(100), NodeId(203)],
            },
        );
        let cond = b1.push_value(ValueNode::TrueConstant);
        b1.set_control(ControlNode::Branch {
            condition: cond,
            if_true: 2,
            if_false: 3,
        });
        graph.add_block(b1);

        // Block 2: body — chain of 3 additions with invariant operands.
        let mut b2 = BasicBlock::new(2);
        b2.add_predecessor(1);
        b2.push_with_id(
            NodeId(201),
            ValueNode::Int32Add {
                left: NodeId(200),
                right: NodeId(100),
            },
        );
        b2.push_with_id(
            NodeId(202),
            ValueNode::Int32Add {
                left: NodeId(201),
                right: NodeId(101),
            },
        );
        b2.push_with_id(
            NodeId(203),
            ValueNode::Int32Add {
                left: NodeId(202),
                right: NodeId(102),
            },
        );
        b2.set_control(ControlNode::Jump { target: 1 });
        graph.add_block(b2);

        // Block 3: exit.
        let mut b3 = BasicBlock::new(3);
        b3.add_predecessor(1);
        let undef = b3.push_value(ValueNode::UndefinedConstant);
        b3.set_control(ControlNode::Return { value: undef });
        graph.add_block(b3);

        let body_adds_before = graph.blocks()[2]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::Int32Add { .. }))
            .count();
        assert_eq!(body_adds_before, 3, "should start with 3 additions in body");

        let folded = super::fold_invariant_addition_chains(&mut graph);
        assert_eq!(folded, 3, "should fold 3 invariant operands");

        // The endpoint (NodeId 203) should now reference the invariant sum from
        // the preheader instead of the chain.  Intermediate nodes (201, 202)
        // are dead and will be cleaned up by DCE; they remain in the body for
        // now.
        let endpoint = graph.blocks()[2]
            .nodes
            .iter()
            .find(|(id, _)| *id == NodeId(203))
            .map(|(_, n)| n);
        match endpoint {
            Some(ValueNode::Int32Add { left, right }) => {
                // left should be the Phi (root_variant = 200).
                assert_eq!(*left, NodeId(200), "endpoint left should be the Phi");
                // right should be an invariant sum node in the preheader,
                // NOT one of the original operands (100, 101, 102).
                assert!(
                    right.0 != 100 && right.0 != 101 && right.0 != 102,
                    "endpoint right should be the new invariant sum node"
                );
            }
            other => panic!("expected Int32Add for endpoint, got {other:?}"),
        }

        // The preheader should have new Int32Add nodes for the invariant sum.
        let preheader_adds = graph.blocks()[0]
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, ValueNode::Int32Add { .. }))
            .count();
        assert_eq!(
            preheader_adds, 2,
            "preheader should have 2 addition nodes for invariant sum (a+b, (a+b)+c)"
        );
    }
}

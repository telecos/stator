//! Peephole fusion over decoded bytecode instructions.

// Dead-store elimination and constant folding are currently disabled (see
// `fuse_instructions`).  Allow the helper functions to remain for future use.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::bytecode::{
    bytecode_array::ConstantPoolEntry,
    bytecodes::{Instruction, Opcode, Operand, decode_with_byte_offsets, encode},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoopInfo {
    header: usize,
    back_edge: usize,
}

#[derive(Clone, Debug)]
struct HoistCandidate {
    load_idx: usize,
    star_idx: usize,
    hoisted_load: Instruction,
    hoisted_star: Instruction,
    replacement: Instruction,
}

/// Scan decoded instruction stream and fuse recognized patterns.
pub fn fuse_instructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    let _ = fuse_instructions_with_remap(instructions, byte_offsets, None);
}

pub(crate) fn fuse_instructions_with_remap(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    constant_pool: Option<&[ConstantPoolEntry]>,
) -> Vec<Option<usize>> {
    if instructions.is_empty() || byte_offsets.len() != instructions.len() + 1 {
        return Vec::new();
    }

    let original_len = instructions.len();
    let mut origins = (0..original_len).map(Some).collect::<Vec<_>>();

    // Dead-store elimination and constant folding are disabled until the
    // remaining correctness issues are resolved (423 test failures from
    // incorrect store elimination across exception-throwing paths).
    // optimize_bytecode(instructions, byte_offsets);
    hoist_loop_invariants(instructions, byte_offsets, &mut origins, constant_pool);
    eliminate_redundant_moves_with_origins(instructions, byte_offsets, &mut origins);
    fuse_superinstructions_with_origins(instructions, byte_offsets, &mut origins);

    build_old_to_new_map(&origins, original_len)
}

/// Eliminate provably redundant register-to-accumulator moves.
///
/// This pass handles three patterns, all requiring that no jump target or block
/// boundary intervenes between the two instructions:
///
/// 1. **Star rX → Ldar rX** — the `Ldar` is redundant because the accumulator
///    still holds the value just stored.  Only intermediate `Star` / `Nop`
///    instructions (which preserve the accumulator) are allowed between them.
///
/// 2. **Ldar rX → Star rX** — the `Star` is redundant because the register
///    already contains the value just loaded into the accumulator.  Only `Nop`
///    instructions are allowed between them (any other instruction might change
///    the accumulator, making the `Star` store a different value).
///
/// 3. **Star rX → Star rY** (adjacent, no intervening use of rX) — the first
///    `Star` is dead.  We only eliminate when the two `Star` instructions are
///    directly adjacent (no instructions between them) so we do not need
///    complex liveness analysis.
///
/// Eliminated instructions are replaced with `Nop` and then compacted out.
fn eliminate_redundant_moves(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
    eliminate_redundant_moves_with_origins(instructions, byte_offsets, &mut origins);
}

fn eliminate_redundant_moves_with_origins(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    origins: &mut Vec<Option<usize>>,
) {
    if instructions.len() < 2 || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    let jump_target_bytes = collect_jump_target_bytes(instructions, byte_offsets);
    let mut changed = false;

    // Pattern 1: Star rX … Ldar rX  (accumulator preserved between them)
    // Pattern 2: Ldar rX … Star rX  (no accumulator change between them)
    {
        let mut i = 0;
        while i < instructions.len() {
            // --- Pattern 1: Star rX followed (through acc-preserving ops) by Ldar rX ---
            if instructions[i].opcode == Opcode::Star
                && let Some(reg) = get_register(&instructions[i], 0)
                && let Some(ldar_idx) =
                    find_next_matching_ldar(instructions, byte_offsets, &jump_target_bytes, i, reg)
            {
                instructions[ldar_idx] = nop_instruction();
                changed = true;
                i = ldar_idx + 1;
                continue;
            }

            // --- Pattern 2: Ldar rX followed (through Nops only) by Star rX ---
            if instructions[i].opcode == Opcode::Ldar
                && let Some(reg) = get_register(&instructions[i], 0)
                && let Some(star_idx) =
                    find_next_matching_star(instructions, byte_offsets, &jump_target_bytes, i, reg)
            {
                instructions[star_idx] = nop_instruction();
                changed = true;
                i = star_idx + 1;
                continue;
            }

            i += 1;
        }
    }

    // Pattern 3: Star rX immediately followed by Star rY — first Star is dead
    // if rX is not mentioned by the second Star (rX != rY since Star takes one
    // register operand and both are different).
    {
        let mut i = 0;
        while i + 1 < instructions.len() {
            if instructions[i].opcode == Opcode::Star
                && instructions[i + 1].opcode == Opcode::Star
                && !is_jump_target(byte_offsets, &jump_target_bytes, i + 1)
            {
                // Both store the accumulator, so the first is dead unless the
                // second one somehow reads the first register (it doesn't — Star
                // only writes).  Safe to eliminate.
                instructions[i] = nop_instruction();
                changed = true;
                // Don't skip i+1; it could itself be part of another pattern.
                i += 1;
            } else {
                i += 1;
            }
        }
    }

    if changed {
        compact_nops_with_origins(instructions, byte_offsets, origins);
    }
}

/// Scan forward from a `Star rX` at `star_idx` looking for an `Ldar rX` that
/// loads the same register, with only accumulator-preserving instructions
/// (`Star`, `Nop`) in between.  Returns the index of the redundant `Ldar`.
fn find_next_matching_ldar(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    star_idx: usize,
    reg: u32,
) -> Option<usize> {
    let mut scan = star_idx + 1;
    while scan < instructions.len() {
        if is_jump_target(byte_offsets, jump_target_bytes, scan)
            || is_block_boundary(&instructions[scan])
        {
            return None;
        }

        let instr = &instructions[scan];
        if instr.opcode == Opcode::Ldar && get_register(instr, 0) == Some(reg) {
            return Some(scan);
        }

        // Only Star and Nop preserve the accumulator — anything else clobbers
        // it, so the Ldar would no longer be redundant.
        if !preserves_accumulator(instr) {
            return None;
        }

        // If an intervening Star writes to the same register, the Ldar after
        // it would still load the correct value (acc didn't change), but only
        // if nothing else wrote to that register.  An intervening Star to a
        // *different* register is fine.  An intervening Star to the *same*
        // register is also fine (it writes the same acc value back).
        scan += 1;
    }
    None
}

/// Scan forward from an `Ldar rX` at `ldar_idx` looking for a `Star rX` that
/// stores back to the same register, with only `Nop` instructions between them
/// (the accumulator must not have been modified).  Returns the index of the
/// redundant `Star`.
fn find_next_matching_star(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    ldar_idx: usize,
    reg: u32,
) -> Option<usize> {
    let mut scan = ldar_idx + 1;
    while scan < instructions.len() {
        if is_jump_target(byte_offsets, jump_target_bytes, scan)
            || is_block_boundary(&instructions[scan])
        {
            return None;
        }

        let instr = &instructions[scan];
        if instr.opcode == Opcode::Star && get_register(instr, 0) == Some(reg) {
            return Some(scan);
        }

        // Only Nop is safe here — any other instruction (including another
        // Star to a different register) might have changed the accumulator
        // value, making the Star no longer store the original Ldar'd value.
        // Actually, Star preserves the accumulator too, so we can skip over
        // Stars to *other* registers.
        if instr.opcode == Opcode::Nop {
            scan += 1;
            continue;
        }

        // A Star to a *different* register still preserves the accumulator,
        // but we must make sure it didn't write to `reg` (which would make
        // the later Star redundant for a different reason).
        if instr.opcode == Opcode::Star {
            // Star to a different register — acc is preserved, and `reg` is
            // untouched.
            if get_register(instr, 0) != Some(reg) {
                scan += 1;
                continue;
            }
            // Star to the same register — this IS the redundant store.
            return Some(scan);
        }

        return None;
    }
    None
}

fn optimize_bytecode(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    if instructions.is_empty() || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    let jump_target_bytes = collect_jump_target_bytes(instructions, byte_offsets);
    apply_local_optimizations(instructions, byte_offsets, &jump_target_bytes);
    compact_nops(instructions, byte_offsets);
}

fn apply_local_optimizations(
    instructions: &mut [Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
) {
    let mut index = 0usize;
    while index < instructions.len() {
        if let Some(folded) =
            try_fold_constant(instructions, byte_offsets, jump_target_bytes, index)
        {
            instructions[index] = folded;
            for instruction in &mut instructions[index + 1..index + 4] {
                *instruction = nop_instruction();
            }
            index += 4;
            continue;
        }

        if let Some(dead_store_index) =
            try_eliminate_dead_store(instructions, byte_offsets, jump_target_bytes, index)
        {
            instructions[dead_store_index] = nop_instruction();

            if let Some(previous_index) = dead_store_producer_to_eliminate(
                instructions,
                byte_offsets,
                jump_target_bytes,
                dead_store_index,
            ) {
                instructions[previous_index] = nop_instruction();
            }

            index += 1;
            continue;
        }

        if let Some(redundant_index) =
            try_eliminate_redundant_load(instructions, byte_offsets, jump_target_bytes, index)
        {
            instructions[redundant_index] = nop_instruction();
            index = redundant_index + 1;
            continue;
        }

        index += 1;
    }
}

fn compact_nops(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
    compact_nops_with_origins(instructions, byte_offsets, &mut origins);
}

fn compact_nops_with_origins(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    origins: &mut Vec<Option<usize>>,
) {
    let sentinel = match byte_offsets.last().copied() {
        Some(sentinel) => sentinel,
        None => return,
    };
    let mut compacted_instructions = Vec::with_capacity(instructions.len());
    let mut compacted_offsets = Vec::with_capacity(byte_offsets.len());
    let mut compacted_origins = Vec::with_capacity(origins.len());
    for (index, instruction) in instructions.iter().enumerate() {
        if instruction.opcode == Opcode::Nop {
            continue;
        }
        compacted_offsets.push(byte_offsets[index]);
        compacted_instructions.push(instruction.clone());
        compacted_origins.push(origins[index]);
    }
    compacted_offsets.push(sentinel);
    *instructions = compacted_instructions;
    *byte_offsets = compacted_offsets;
    *origins = compacted_origins;
}

fn fuse_superinstructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
    fuse_superinstructions_with_origins(instructions, byte_offsets, &mut origins);
}

fn fuse_superinstructions_with_origins(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    origins: &mut Vec<Option<usize>>,
) {
    if instructions.len() < 2 || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    // Snapshot jump targets before fusion so resolve_jump_offsets can remap
    // them after the instruction count changes.
    let old_targets = collect_jump_target_indices(instructions, byte_offsets);

    let jump_target_bytes = collect_jump_target_bytes(instructions, byte_offsets);

    let mut fused_instructions = Vec::with_capacity(instructions.len());
    let mut fused_origins = Vec::with_capacity(origins.len());
    // Track how many pre-fusion instructions each post-fusion instruction
    // consumed so we can build a complete pre→post index mapping later.
    let mut consumed: Vec<usize> = Vec::with_capacity(origins.len());
    let mut index = 0usize;

    while index < instructions.len() {
        if index + 2 < instructions.len()
            && !jump_target_bytes.contains(&byte_offsets[index + 1])
            && !jump_target_bytes.contains(&byte_offsets[index + 2])
        {
            let first = &instructions[index];
            let second = &instructions[index + 1];
            let third = &instructions[index + 2];

            match (first.opcode, second.opcode, third.opcode) {
                (Opcode::Ldar, Opcode::Add, Opcode::Star) => {
                    if let (
                        Operand::Register(src),
                        Operand::Register(add_reg),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (
                        *first.operand(0),
                        *second.operand(0),
                        *second.operand(1),
                        *third.operand(0),
                    ) {
                        fused_origins.push(origins[index]);
                        consumed.push(3);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::LdarAddStar,
                            vec![
                                Operand::Register(src),
                                Operand::Register(add_reg),
                                Operand::Register(dst),
                                Operand::FeedbackSlot(slot),
                            ],
                        ));
                        index += 3;
                        continue;
                    }
                }
                (Opcode::Ldar, Opcode::Sub, Opcode::Star) => {
                    if let (
                        Operand::Register(src),
                        Operand::Register(sub_reg),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (
                        *first.operand(0),
                        *second.operand(0),
                        *second.operand(1),
                        *third.operand(0),
                    ) {
                        fused_origins.push(origins[index]);
                        consumed.push(3);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::LdarSubStar,
                            vec![
                                Operand::Register(src),
                                Operand::Register(sub_reg),
                                Operand::Register(dst),
                                Operand::FeedbackSlot(slot),
                            ],
                        ));
                        index += 3;
                        continue;
                    }
                }
                (Opcode::Ldar, Opcode::Mul, Opcode::Star) => {
                    if let (
                        Operand::Register(src),
                        Operand::Register(mul_reg),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (
                        *first.operand(0),
                        *second.operand(0),
                        *second.operand(1),
                        *third.operand(0),
                    ) {
                        fused_origins.push(origins[index]);
                        consumed.push(3);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::LdarMulStar,
                            vec![
                                Operand::Register(src),
                                Operand::Register(mul_reg),
                                Operand::Register(dst),
                                Operand::FeedbackSlot(slot),
                            ],
                        ));
                        index += 3;
                        continue;
                    }
                }
                _ => {}
            }
        }

        if index + 1 < instructions.len() && !jump_target_bytes.contains(&byte_offsets[index + 1]) {
            let first = &instructions[index];
            let second = &instructions[index + 1];

            match (first.opcode, second.opcode) {
                (Opcode::TestLessThan, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestLessThanJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestGreaterThan, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestGreaterThanJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestEqual, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestEqualJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestNotEqual, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestNotEqualJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestEqualStrict, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestEqualStrictJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestLessThanOrEqual, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestLessThanOrEqualJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::TestGreaterThanOrEqual, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::TestGreaterThanOrEqualJump,
                            vec![
                                Operand::Register(reg),
                                Operand::FeedbackSlot(slot),
                                Operand::JumpOffset(offset),
                                Operand::Flag(is_true),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::AddSmi, Opcode::Star) => {
                    if let (
                        Operand::Immediate(imm),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::AddSmiStar,
                            vec![
                                Operand::Immediate(imm),
                                Operand::FeedbackSlot(slot),
                                Operand::Register(dst),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::SubSmi, Opcode::Star) => {
                    if let (
                        Operand::Immediate(imm),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::SubSmiStar,
                            vec![
                                Operand::Immediate(imm),
                                Operand::FeedbackSlot(slot),
                                Operand::Register(dst),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::MulSmi, Opcode::Star) => {
                    if let (
                        Operand::Immediate(imm),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::MulSmiStar,
                            vec![
                                Operand::Immediate(imm),
                                Operand::FeedbackSlot(slot),
                                Operand::Register(dst),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::Inc, Opcode::Star) => {
                    if let (Operand::FeedbackSlot(slot), Operand::Register(dst)) =
                        (*first.operand(0), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::IncStar,
                            vec![Operand::FeedbackSlot(slot), Operand::Register(dst)],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::LdaSmi, Opcode::Star) => {
                    if let (Operand::Immediate(imm), Operand::Register(dst)) =
                        (*first.operand(0), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::LdaSmiStar,
                            vec![Operand::Immediate(imm), Operand::Register(dst)],
                        ));
                        index += 2;
                        continue;
                    }
                }
                (Opcode::LdaGlobal, Opcode::Star) => {
                    if let (
                        Operand::ConstantPoolIdx(name_idx),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (*first.operand(0), *first.operand(1), *second.operand(0))
                    {
                        fused_origins.push(origins[index]);
                        consumed.push(2);
                        fused_instructions.push(Instruction::new_unchecked(
                            Opcode::LdaGlobalStar,
                            vec![
                                Operand::ConstantPoolIdx(name_idx),
                                Operand::FeedbackSlot(slot),
                                Operand::Register(dst),
                            ],
                        ));
                        index += 2;
                        continue;
                    }
                }
                _ => {}
            }
        }

        fused_instructions.push(instructions[index].clone());
        fused_origins.push(origins[index]);
        consumed.push(1);
        index += 1;
    }

    *instructions = fused_instructions;
    *origins = fused_origins;

    // Build a complete pre-fusion → post-fusion index mapping.
    // Each fused instruction consumed N pre-fusion instructions; all N
    // indices must map to the same post-fusion instruction so that jump
    // targets from any of the consumed instructions are resolved correctly.
    // Include an extra sentinel entry for jumps that target past-the-end.
    let pre_fusion_count = old_targets.len();
    let post_fusion_count = instructions.len();
    let mut pre_to_post = vec![None; pre_fusion_count + 1];
    {
        let mut old_idx = 0usize;
        for (new_idx, &count) in consumed.iter().enumerate() {
            for j in 0..count {
                if old_idx + j < pre_fusion_count {
                    pre_to_post[old_idx + j] = Some(new_idx);
                }
            }
            old_idx += count;
        }
    }
    // Map sentinel: past-end in pre-fusion → past-end in post-fusion.
    pre_to_post[pre_fusion_count] = Some(post_fusion_count);

    // Recalculate jump offsets.  Encoding sizes can change when deltas
    // change (1-byte vs 4-byte operand), so iterate until stable.
    let mut new_offsets = recompute_byte_offsets(instructions);
    for _ in 0..20 {
        let mut changed = false;
        for (old_idx, target) in old_targets.iter().enumerate() {
            let (Some(new_idx), Some(target_old)) = (pre_to_post[old_idx], *target) else {
                continue;
            };
            let Some(target_new) = pre_to_post[target_old] else {
                continue;
            };
            let delta = new_offsets[target_new] as i64 - new_offsets[new_idx + 1] as i64;
            if set_jump_offset(&mut instructions[new_idx], delta as i32) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
        new_offsets = recompute_byte_offsets(instructions);
    }
    *byte_offsets = new_offsets;
}

fn hoist_loop_invariants(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    origins: &mut Vec<Option<usize>>,
    constant_pool: Option<&[ConstantPoolEntry]>,
) {
    if instructions.is_empty()
        || byte_offsets.len() != instructions.len() + 1
        || contains_constant_pool_jump(instructions)
    {
        return;
    }

    loop {
        let mut loops = find_loops(instructions, byte_offsets);
        loops.sort_by_key(|loop_info| (loop_info.header, loop_info.back_edge));

        let mut changed = false;
        for loop_info in loops.into_iter().rev() {
            if loop_info.back_edge >= instructions.len() || loop_info.header >= loop_info.back_edge
            {
                continue;
            }

            let candidates =
                find_hoist_candidates(instructions, byte_offsets, loop_info, constant_pool);
            if candidates.is_empty() {
                continue;
            }

            apply_hoists_for_loop(instructions, byte_offsets, origins, loop_info, &candidates);
            changed = true;
            break;
        }

        if !changed {
            break;
        }
    }
}

fn find_loops(instructions: &[Instruction], byte_offsets: &[usize]) -> Vec<LoopInfo> {
    instructions
        .iter()
        .enumerate()
        .filter_map(|(index, instruction)| {
            (instruction.opcode == Opcode::JumpLoop)
                .then(|| resolve_jump_target_index(instructions, byte_offsets, index))
                .flatten()
                .filter(|&header| header < index)
                .map(|header| LoopInfo {
                    header,
                    back_edge: index,
                })
        })
        .collect()
}

fn find_hoist_candidates(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    loop_info: LoopInfo,
    constant_pool: Option<&[ConstantPoolEntry]>,
) -> Vec<HoistCandidate> {
    let jump_target_bytes = collect_jump_target_bytes(instructions, byte_offsets);
    let mut candidates = Vec::new();
    let mut reserved_registers = HashSet::new();
    let mut index = loop_info.header;

    while index < loop_info.back_edge {
        // The loop header is always a jump target (the back-edge targets it),
        // but instructions there execute on every iteration and are valid
        // hoisting candidates.  Only skip non-header jump targets.
        let at_header = index == loop_info.header;
        if (!at_header && is_jump_target(byte_offsets, &jump_target_bytes, index))
            || is_jump_target(byte_offsets, &jump_target_bytes, index + 1)
        {
            index += 1;
            continue;
        }

        let load = &instructions[index];
        let star = &instructions[index + 1];
        if star.opcode != Opcode::Star {
            index += 1;
            continue;
        }

        let Some(dst) = get_register(star, 0) else {
            index += 1;
            continue;
        };
        if reserved_registers.contains(&dst) {
            index += 1;
            continue;
        }

        let candidate = match load.opcode {
            Opcode::LdaGlobal => is_loop_invariant_global(instructions, loop_info, index, dst)
                .then(|| build_hoist_candidate(load, star, index)),
            Opcode::LdaNamedProperty => {
                is_loop_invariant_length_load(instructions, loop_info, index, dst, constant_pool)
                    .then(|| build_hoist_candidate(load, star, index))
            }
            _ => None,
        };

        if let Some(candidate) = candidate {
            reserved_registers.insert(dst);
            candidates.push(candidate);
            index += 2;
            continue;
        }

        index += 1;
    }

    candidates
}

fn build_hoist_candidate(
    load: &Instruction,
    star: &Instruction,
    load_idx: usize,
) -> HoistCandidate {
    let dst = get_register(star, 0).expect("Star must have a destination register");
    HoistCandidate {
        load_idx,
        star_idx: load_idx + 1,
        hoisted_load: load.clone(),
        hoisted_star: star.clone(),
        replacement: Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(dst)]),
    }
}

fn is_loop_invariant_global(
    instructions: &[Instruction],
    loop_info: LoopInfo,
    load_idx: usize,
    dst: u32,
) -> bool {
    let Some(name_idx) = get_constant_pool_idx(&instructions[load_idx], 0) else {
        return false;
    };

    !loop_writes_register(instructions, loop_info, dst, &[load_idx + 1])
        && !loop_invalidates_global(instructions, loop_info, load_idx, name_idx)
}

fn is_loop_invariant_length_load(
    instructions: &[Instruction],
    loop_info: LoopInfo,
    load_idx: usize,
    dst: u32,
    constant_pool: Option<&[ConstantPoolEntry]>,
) -> bool {
    let Some(obj_reg) = get_register(&instructions[load_idx], 0) else {
        return false;
    };
    let Some(name_idx) = get_constant_pool_idx(&instructions[load_idx], 1) else {
        return false;
    };

    matches!(
        constant_pool.and_then(|pool| pool.get(name_idx as usize)),
        Some(ConstantPoolEntry::String(name)) if name == "length"
    ) && !loop_writes_register(instructions, loop_info, dst, &[load_idx + 1])
        && !loop_writes_register(instructions, loop_info, obj_reg, &[])
        && !loop_invalidates_named_property(instructions, loop_info, load_idx, obj_reg)
}

fn apply_hoists_for_loop(
    instructions: &mut Vec<Instruction>,
    byte_offsets: &mut Vec<usize>,
    origins: &mut Vec<Option<usize>>,
    loop_info: LoopInfo,
    candidates: &[HoistCandidate],
) {
    let old_targets = collect_jump_target_indices(instructions, byte_offsets);
    let old_origins = origins.clone();
    let old_instructions = std::mem::take(instructions);

    let replacements = candidates
        .iter()
        .map(|candidate| (candidate.load_idx, candidate))
        .collect::<HashMap<_, _>>();
    let removed_star_indices = candidates
        .iter()
        .map(|candidate| candidate.star_idx)
        .collect::<HashSet<_>>();

    let mut new_instructions = Vec::with_capacity(old_instructions.len() + candidates.len() * 2);
    let mut new_origins = Vec::with_capacity(old_origins.len() + candidates.len() * 2);

    for (index, instruction) in old_instructions.into_iter().enumerate() {
        if index == loop_info.header {
            for candidate in candidates {
                new_instructions.push(candidate.hoisted_load.clone());
                new_origins.push(None);
                new_instructions.push(candidate.hoisted_star.clone());
                new_origins.push(None);
            }
        }

        if let Some(candidate) = replacements.get(&index) {
            new_instructions.push(candidate.replacement.clone());
            new_origins.push(old_origins[index]);
            continue;
        }

        if removed_star_indices.contains(&index) {
            continue;
        }

        new_origins.push(old_origins[index]);
        new_instructions.push(instruction);
    }

    *instructions = new_instructions;
    *origins = new_origins;
    *byte_offsets = resolve_jump_offsets(instructions, origins, &old_targets);
}

fn loop_writes_register(
    instructions: &[Instruction],
    loop_info: LoopInfo,
    register: u32,
    ignored_indices: &[usize],
) -> bool {
    (loop_info.header..=loop_info.back_edge)
        .filter(|index| !ignored_indices.contains(index))
        .any(|index| instruction_writes_register(&instructions[index], register))
}

fn loop_invalidates_global(
    instructions: &[Instruction],
    loop_info: LoopInfo,
    load_idx: usize,
    name_idx: u32,
) -> bool {
    (loop_info.header..=loop_info.back_edge)
        .filter(|&index| index != load_idx && index != load_idx + 1)
        .any(|index| {
            writes_same_global(&instructions[index], name_idx)
                || instruction_has_unknown_global_side_effect(&instructions[index])
        })
}

fn loop_invalidates_named_property(
    instructions: &[Instruction],
    loop_info: LoopInfo,
    load_idx: usize,
    obj_reg: u32,
) -> bool {
    (loop_info.header..=loop_info.back_edge)
        .filter(|&index| index != load_idx && index != load_idx + 1)
        .any(|index| {
            instruction_has_unknown_global_side_effect(&instructions[index])
                || instruction_writes_property_of_register(&instructions[index], obj_reg)
        })
}

fn instruction_writes_register(instruction: &Instruction, register: u32) -> bool {
    match instruction.opcode {
        Opcode::Star => get_register(instruction, 0) == Some(register),
        Opcode::Mov => get_register(instruction, 1) == Some(register),
        Opcode::IteratorNext => get_register(instruction, 1) == Some(register),
        Opcode::CallRuntimeForPair => get_register(instruction, 3) == Some(register),
        Opcode::ResumeGenerator => {
            if let (Some(start), Some(Operand::RegisterCount(count))) =
                (get_register(instruction, 1), instruction.operand_at(2))
            {
                register >= start && register < start + count
            } else {
                false
            }
        }
        _ => false,
    }
}

fn writes_same_global(instruction: &Instruction, name_idx: u32) -> bool {
    instruction.opcode == Opcode::StaGlobal
        && get_constant_pool_idx(instruction, 0) == Some(name_idx)
}

fn instruction_has_unknown_global_side_effect(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode,
        Opcode::StaLookupSlot
            | Opcode::DeleteLookupSlot
            | Opcode::StaContextSlot
            | Opcode::StaCurrentContextSlot
            | Opcode::StaModuleVariable
            | Opcode::StaNamedProperty
            | Opcode::StaNamedOwnProperty
            | Opcode::StaKeyedProperty
            | Opcode::DefineNamedOwnProperty
            | Opcode::DefineKeyedOwnProperty
            | Opcode::DefineKeyedOwnPropertyInLiteral
            | Opcode::DefineGetterProperty
            | Opcode::DefineSetterProperty
            | Opcode::DefineKeyedGetterProperty
            | Opcode::DefineKeyedSetterProperty
            | Opcode::DefinePrivateBrand
            | Opcode::DefineClassNamedOwnProperty
            | Opcode::DefineClassGetterProperty
            | Opcode::DefineClassSetterProperty
            | Opcode::DefineClassKeyedOwnProperty
            | Opcode::DefineClassKeyedGetterProperty
            | Opcode::DefineClassKeyedSetterProperty
            | Opcode::SetLiteralPrototype
            | Opcode::DeletePropertyStrict
            | Opcode::DeletePropertySloppy
            | Opcode::CallAnyReceiver
            | Opcode::CallProperty
            | Opcode::CallProperty0
            | Opcode::CallProperty1
            | Opcode::CallProperty2
            | Opcode::CallUndefinedReceiver0
            | Opcode::CallUndefinedReceiver1
            | Opcode::CallUndefinedReceiver2
            | Opcode::CallWithSpread
            | Opcode::CallRuntime
            | Opcode::CallRuntimeForPair
            | Opcode::CallJSRuntime
            | Opcode::InvokeIntrinsic
            | Opcode::CallDirectEval
            | Opcode::TailCall
            | Opcode::Construct
            | Opcode::ConstructWithSpread
            | Opcode::ConstructForwardAllArgs
    )
}

fn instruction_writes_property_of_register(instruction: &Instruction, register: u32) -> bool {
    match instruction.opcode {
        Opcode::StaNamedProperty
        | Opcode::StaNamedOwnProperty
        | Opcode::StaKeyedProperty
        | Opcode::DefineNamedOwnProperty
        | Opcode::DefineKeyedOwnProperty
        | Opcode::DefineKeyedOwnPropertyInLiteral
        | Opcode::DefineGetterProperty
        | Opcode::DefineSetterProperty
        | Opcode::DefineKeyedGetterProperty
        | Opcode::DefineKeyedSetterProperty
        | Opcode::SetLiteralPrototype
        | Opcode::DefinePrivateBrand
        | Opcode::DefineClassNamedOwnProperty
        | Opcode::DefineClassGetterProperty
        | Opcode::DefineClassSetterProperty
        | Opcode::DefineClassKeyedOwnProperty
        | Opcode::DefineClassKeyedGetterProperty
        | Opcode::DefineClassKeyedSetterProperty
        | Opcode::DeletePropertyStrict
        | Opcode::DeletePropertySloppy => get_register(instruction, 0) == Some(register),
        _ => false,
    }
}

fn contains_constant_pool_jump(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instruction| {
        matches!(
            instruction.opcode,
            Opcode::JumpConstant
                | Opcode::JumpIfTrueConstant
                | Opcode::JumpIfFalseConstant
                | Opcode::JumpIfToBooleanTrueConstant
                | Opcode::JumpIfToBooleanFalseConstant
                | Opcode::JumpIfNullConstant
                | Opcode::JumpIfNotNullConstant
                | Opcode::JumpIfUndefinedConstant
                | Opcode::JumpIfNotUndefinedConstant
                | Opcode::JumpIfUndefinedOrNullConstant
                | Opcode::JumpIfJSReceiverConstant
        )
    })
}

fn collect_jump_target_indices(
    instructions: &[Instruction],
    byte_offsets: &[usize],
) -> Vec<Option<usize>> {
    instructions
        .iter()
        .enumerate()
        .map(|(index, _)| resolve_jump_target_index(instructions, byte_offsets, index))
        .collect()
}

fn resolve_jump_target_index(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    index: usize,
) -> Option<usize> {
    let instruction = instructions.get(index)?;
    let delta = instruction
        .operands()
        .iter()
        .find_map(|operand| match operand {
            Operand::JumpOffset(delta) => Some(*delta),
            _ => None,
        })?;
    let end_byte = byte_offsets.get(index + 1).copied()?;
    let target_byte = end_byte as i64 + i64::from(delta);
    usize::try_from(target_byte)
        .ok()
        .and_then(|target_byte| byte_offsets.binary_search(&target_byte).ok())
}

fn resolve_jump_offsets(
    instructions: &mut [Instruction],
    origins: &[Option<usize>],
    old_targets: &[Option<usize>],
) -> Vec<usize> {
    const MAX_ITERS: usize = 20;
    let old_to_new = build_old_to_new_map(origins, old_targets.len());
    let mut offsets = recompute_byte_offsets(instructions);

    for _ in 0..MAX_ITERS {
        let mut changed = false;

        for (old_index, target) in old_targets.iter().enumerate() {
            let (Some(new_index), Some(target_old_index)) = (old_to_new[old_index], *target) else {
                continue;
            };
            let Some(target_new_index) = old_to_new[target_old_index] else {
                continue;
            };
            let delta = offsets[target_new_index] as i64 - offsets[new_index + 1] as i64;
            if set_jump_offset(&mut instructions[new_index], delta as i32) {
                changed = true;
            }
        }

        if !changed {
            return offsets;
        }

        offsets = recompute_byte_offsets(instructions);
    }

    offsets
}

fn set_jump_offset(instruction: &mut Instruction, new_delta: i32) -> bool {
    for idx in 0..instruction.operand_count() {
        if let Operand::JumpOffset(delta) = instruction.operand_mut(idx) {
            if *delta == new_delta {
                return false;
            }
            *delta = new_delta;
            return true;
        }
    }
    false
}

fn recompute_byte_offsets(instructions: &[Instruction]) -> Vec<usize> {
    let bytes = encode(instructions);
    let (_, byte_offsets) =
        decode_with_byte_offsets(&bytes).expect("encoding transformed peephole bytecode");
    byte_offsets
}

fn build_old_to_new_map(origins: &[Option<usize>], original_len: usize) -> Vec<Option<usize>> {
    let mut map = vec![None; original_len];
    for (new_index, origin) in origins.iter().enumerate() {
        if let Some(old_index) = origin {
            map[*old_index] = Some(new_index);
        }
    }
    map
}

fn try_fold_constant(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    index: usize,
) -> Option<Instruction> {
    if index + 3 >= instructions.len()
        || is_jump_target(byte_offsets, jump_target_bytes, index + 1)
        || is_jump_target(byte_offsets, jump_target_bytes, index + 2)
        || is_jump_target(byte_offsets, jump_target_bytes, index + 3)
    {
        return None;
    }

    let (Opcode::LdaSmi, Opcode::Star, Opcode::LdaSmi, Opcode::Add | Opcode::Sub | Opcode::Mul) = (
        instructions[index].opcode,
        instructions[index + 1].opcode,
        instructions[index + 2].opcode,
        instructions[index + 3].opcode,
    ) else {
        return None;
    };

    let a = get_immediate(&instructions[index], 0)?;
    let star_reg = get_register(&instructions[index + 1], 0)?;
    let b = get_immediate(&instructions[index + 2], 0)?;
    let arithmetic_reg = get_register(&instructions[index + 3], 0)?;
    if star_reg != arithmetic_reg {
        return None;
    }

    let result = match instructions[index + 3].opcode {
        Opcode::Add => a.checked_add(b)?,
        Opcode::Sub => b.checked_sub(a)?,
        Opcode::Mul => a.checked_mul(b)?,
        _ => return None,
    };

    Some(Instruction::new_unchecked(
        Opcode::LdaSmi,
        vec![Operand::Immediate(result)],
    ))
}

fn try_eliminate_dead_store(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    index: usize,
) -> Option<usize> {
    if instructions.get(index)?.opcode != Opcode::Star {
        return None;
    }

    let register = get_register(&instructions[index], 0)?;
    let mut scan = index + 1;
    while scan < instructions.len() {
        if is_jump_target(byte_offsets, jump_target_bytes, scan)
            || is_block_boundary(&instructions[scan])
        {
            return None;
        }
        if instructions[scan].opcode == Opcode::Nop {
            scan += 1;
            continue;
        }
        if instruction_mentions_register(&instructions[scan], register) {
            return (instructions[scan].opcode == Opcode::Star
                && get_register(&instructions[scan], 0) == Some(register))
            .then_some(index);
        }
        scan += 1;
    }
    None
}

fn dead_store_producer_to_eliminate(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    star_index: usize,
) -> Option<usize> {
    let previous_index = star_index.checked_sub(1)?;
    if is_jump_target(byte_offsets, jump_target_bytes, previous_index)
        || !is_pure_accumulator_load(&instructions[previous_index])
    {
        return None;
    }

    let next_index = next_non_nop_index(instructions, star_index + 1)?;
    is_pure_accumulator_load(&instructions[next_index]).then_some(previous_index)
}

fn try_eliminate_redundant_load(
    instructions: &[Instruction],
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    index: usize,
) -> Option<usize> {
    if instructions.get(index)?.opcode != Opcode::Ldar {
        return None;
    }

    let register = get_register(&instructions[index], 0)?;
    let mut scan = index + 1;
    while scan < instructions.len() {
        if is_jump_target(byte_offsets, jump_target_bytes, scan)
            || is_block_boundary(&instructions[scan])
        {
            return None;
        }

        let instruction = &instructions[scan];
        if instruction.opcode == Opcode::Nop {
            scan += 1;
            continue;
        }
        if instruction.opcode == Opcode::Ldar && get_register(instruction, 0) == Some(register) {
            return Some(scan);
        }
        if !preserves_accumulator(instruction)
            || instruction_mentions_register(instruction, register)
        {
            return None;
        }
        scan += 1;
    }
    None
}

fn nop_instruction() -> Instruction {
    Instruction::new_unchecked(Opcode::Nop, vec![])
}

fn next_non_nop_index(instructions: &[Instruction], start: usize) -> Option<usize> {
    instructions
        .iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, instruction)| (instruction.opcode != Opcode::Nop).then_some(index))
}

fn get_immediate(instruction: &Instruction, operand_index: usize) -> Option<i32> {
    match instruction.operand_at(operand_index).copied()? {
        Operand::Immediate(value) => Some(value),
        _ => None,
    }
}

fn get_register(instruction: &Instruction, operand_index: usize) -> Option<u32> {
    match instruction.operand_at(operand_index).copied()? {
        Operand::Register(register) => Some(register),
        _ => None,
    }
}

fn get_constant_pool_idx(instruction: &Instruction, operand_index: usize) -> Option<u32> {
    match instruction.operand_at(operand_index).copied()? {
        Operand::ConstantPoolIdx(index) => Some(index),
        _ => None,
    }
}

fn is_jump_target(
    byte_offsets: &[usize],
    jump_target_bytes: &HashSet<usize>,
    index: usize,
) -> bool {
    byte_offsets
        .get(index)
        .is_some_and(|byte_offset| jump_target_bytes.contains(byte_offset))
}

fn instruction_mentions_register(instruction: &Instruction, register: u32) -> bool {
    // Check explicit register operands.
    if instruction
        .operands()
        .iter()
        .any(|operand| matches!(operand, Operand::Register(value) if *value == register))
    {
        return true;
    }

    // Instructions with a (Register, RegisterCount) pair implicitly access a
    // contiguous range of registers.  Check whether `register` falls inside
    // that range so dead-store elimination doesn't incorrectly remove a store
    // to a register used as a call argument.
    if uses_register_range(instruction.opcode)
        && let Some(range_start) = get_register(instruction, 1)
        && let Some(Operand::RegisterCount(count)) = instruction.operand_at(2)
        && register >= range_start
        && register < range_start + count
    {
        return true;
    }

    false
}

/// Returns `true` for opcodes whose operand layout includes a register-range
/// pair at operand positions 1 (start register) and 2 (count).
fn uses_register_range(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::CallAnyReceiver
            | Opcode::CallProperty
            | Opcode::CallWithSpread
            | Opcode::CallRuntime
            | Opcode::CallRuntimeForPair
            | Opcode::CallJSRuntime
            | Opcode::InvokeIntrinsic
            | Opcode::CallDirectEval
            | Opcode::TailCall
            | Opcode::Construct
            | Opcode::ConstructWithSpread
            | Opcode::ResumeGenerator
            | Opcode::SuspendGenerator
    )
}

fn is_pure_accumulator_load(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode,
        Opcode::LdaZero
            | Opcode::LdaSmi
            | Opcode::LdaUndefined
            | Opcode::LdaTheHole
            | Opcode::LdaNull
            | Opcode::LdaTrue
            | Opcode::LdaFalse
            | Opcode::Ldar
    )
}

fn preserves_accumulator(instruction: &Instruction) -> bool {
    matches!(instruction.opcode, Opcode::Star | Opcode::Nop)
}

fn is_block_boundary(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode,
        Opcode::JumpLoop
            | Opcode::Jump
            | Opcode::JumpConstant
            | Opcode::JumpIfTrue
            | Opcode::JumpIfTrueConstant
            | Opcode::JumpIfFalse
            | Opcode::JumpIfFalseConstant
            | Opcode::JumpIfNull
            | Opcode::JumpIfNotNull
            | Opcode::JumpIfUndefined
            | Opcode::JumpIfNotUndefined
            | Opcode::JumpIfUndefinedOrNull
            | Opcode::JumpIfJSReceiver
            | Opcode::JumpIfForInDone
            | Opcode::JumpIfToBooleanTrue
            | Opcode::JumpIfToBooleanFalse
            | Opcode::JumpIfToBooleanTrueConstant
            | Opcode::JumpIfToBooleanFalseConstant
            | Opcode::JumpIfNullConstant
            | Opcode::JumpIfNotNullConstant
            | Opcode::JumpIfUndefinedConstant
            | Opcode::JumpIfNotUndefinedConstant
            | Opcode::JumpIfUndefinedOrNullConstant
            | Opcode::JumpIfJSReceiverConstant
            | Opcode::Return
            | Opcode::Throw
            | Opcode::ReThrow
    )
}

fn collect_jump_target_bytes(
    instructions: &[Instruction],
    byte_offsets: &[usize],
) -> HashSet<usize> {
    let mut target_bytes = HashSet::new();
    for (index, instruction) in instructions.iter().enumerate() {
        for operand in instruction.operands() {
            let Operand::JumpOffset(delta) = operand else {
                continue;
            };
            let Some(end_byte) = byte_offsets.get(index + 1).copied() else {
                continue;
            };
            let target_byte = end_byte as i64 + i64::from(*delta);
            if let Ok(target_byte) = usize::try_from(target_byte) {
                target_bytes.insert(target_byte);
            }
        }
    }
    target_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::bytecodes::encode;

    fn decode_with_offsets(instructions: &[Instruction]) -> (Vec<Instruction>, Vec<usize>) {
        let bytes = encode(instructions);
        crate::bytecode::bytecodes::decode_with_byte_offsets(&bytes).expect("valid bytecode")
    }

    fn resolve_jump_offsets_for_test(
        instructions: &mut [Instruction],
        jumps: &[(usize, usize)],
    ) -> Vec<usize> {
        let (_, offsets) = decode_with_offsets(instructions);
        for &(jump_index, target_index) in jumps {
            let target_byte = offsets[target_index];
            let jump_end_byte = offsets[jump_index + 1];
            *instructions[jump_index].operand_mut(0) =
                Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);
        }
        decode_with_offsets(instructions).1
    }

    #[test]
    fn test_fuse_recognized_patterns() {
        let original = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(4), Operand::FeedbackSlot(5)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(6), Operand::FeedbackSlot(7)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(8)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdarAddStar,
                Opcode::TestLessThanJump,
                Opcode::AddSmiStar,
                Opcode::Return,
            ]
        );
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    fn test_fuse_additional_patterns() {
        let original = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::Mul,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(3)]),
            Instruction::new_unchecked(
                Opcode::TestLessThanOrEqual,
                vec![Operand::Register(4), Operand::FeedbackSlot(5)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfTrue, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::TestGreaterThanOrEqual,
                vec![Operand::Register(6), Operand::FeedbackSlot(7)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::TestNotEqual,
                vec![Operand::Register(8), Operand::FeedbackSlot(9)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfTrue, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::MulSmi,
                vec![Operand::Immediate(10), Operand::FeedbackSlot(11)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(12)]),
            Instruction::new_unchecked(Opcode::Inc, vec![Operand::FeedbackSlot(13)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(14)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(15)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(16)]),
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(17), Operand::FeedbackSlot(18)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(19)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdarMulStar,
                Opcode::TestLessThanOrEqualJump,
                Opcode::TestGreaterThanOrEqualJump,
                Opcode::TestNotEqualJump,
                Opcode::MulSmiStar,
                Opcode::IncStar,
                Opcode::LdaSmiStar,
                Opcode::LdaGlobalStar,
                Opcode::Return,
            ]
        );
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    #[ignore = "constant folding disabled until correctness issues resolved"]
    fn test_fold_constant_add_sequence() {
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(3)]),
            Instruction::new_unchecked(
                Opcode::Add,
                vec![Operand::Register(0), Operand::FeedbackSlot(1)],
            ),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions,
            vec![Instruction::new_unchecked(
                Opcode::LdaSmi,
                vec![Operand::Immediate(8)],
            )]
        );
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    fn test_dead_store_elimination_marks_nops() {
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(2)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
        ];

        let (mut instructions, byte_offsets) = decode_with_offsets(&original);
        let jump_target_bytes = collect_jump_target_bytes(&instructions, &byte_offsets);
        apply_local_optimizations(&mut instructions, &byte_offsets, &jump_target_bytes);

        assert_eq!(
            instructions
                .iter()
                .map(|instruction| instruction.opcode)
                .collect::<Vec<_>>(),
            vec![Opcode::Nop, Opcode::Nop, Opcode::LdaSmi, Opcode::Star]
        );
        assert_eq!(get_immediate(&instructions[2], 0), Some(2));
        assert_eq!(get_register(&instructions[3], 0), Some(0));
    }

    #[test]
    fn test_redundant_load_elimination_marks_nop() {
        let original = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
        ];

        let (mut instructions, byte_offsets) = decode_with_offsets(&original);
        let jump_target_bytes = collect_jump_target_bytes(&instructions, &byte_offsets);
        apply_local_optimizations(&mut instructions, &byte_offsets, &jump_target_bytes);

        assert_eq!(
            instructions
                .iter()
                .map(|instruction| instruction.opcode)
                .collect::<Vec<_>>(),
            vec![Opcode::Ldar, Opcode::Star, Opcode::Nop]
        );
    }

    #[test]
    fn test_dead_store_not_eliminated_when_register_in_call_range() {
        // Star r1 followed by CallAnyReceiver that uses r0..r2 (RegisterCount 3).
        // The register r1 is inside the call's implicit range, so the Star must
        // NOT be eliminated even though a later Star r1 overwrites it.
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(
                Opcode::CallAnyReceiver,
                vec![
                    Operand::Register(5),
                    Operand::Register(0),
                    Operand::RegisterCount(3),
                    Operand::FeedbackSlot(0),
                ],
            ),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(99)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
        ];

        let (mut instructions, byte_offsets) = decode_with_offsets(&original);
        let jump_target_bytes = collect_jump_target_bytes(&instructions, &byte_offsets);
        apply_local_optimizations(&mut instructions, &byte_offsets, &jump_target_bytes);

        // The first Star r1 must survive — the Call reads r1 as an argument.
        assert_eq!(
            instructions
                .iter()
                .map(|instruction| instruction.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdaSmi,
                Opcode::Star,
                Opcode::CallAnyReceiver,
                Opcode::LdaSmi,
                Opcode::Star,
            ]
        );
    }

    #[test]
    fn test_does_not_fuse_across_jump_targets() {
        let unresolved = vec![
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(1), Operand::FeedbackSlot(0)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let bytes = encode(&unresolved);
        let (_, offsets) =
            crate::bytecode::bytecodes::decode_with_byte_offsets(&bytes).expect("valid bytecode");
        let mut resolved = unresolved;
        let target_byte = offsets[2];
        let jump_end_byte = offsets[1];
        *resolved[0].operand_mut(0) =
            Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&resolved);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![Opcode::Jump, Opcode::AddSmi, Opcode::Star, Opcode::Return]
        );
    }

    // ---- Move elimination tests ----

    #[test]
    fn test_star_ldar_same_register_eliminated() {
        // Star r0 → Ldar r0 should eliminate the Ldar, then LdaSmi+Star
        // fuses into LdaSmiStar.
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions.iter().map(|i| i.opcode).collect::<Vec<_>>(),
            vec![Opcode::LdaSmiStar, Opcode::Return]
        );
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    fn test_star_ldar_different_register_not_eliminated() {
        // Star r0 → Ldar r1 must NOT be eliminated.
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(42)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        // Ldar r1 should survive (different register).
        let opcodes: Vec<_> = instructions.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::Ldar),
            "Ldar to a different register must not be eliminated"
        );
    }

    #[test]
    fn test_star_ldar_with_intervening_star_eliminated() {
        // Star r0 → Star r1 → Ldar r0 — the Ldar is still redundant because
        // Star preserves the accumulator.
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(10)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        // The consecutive Star r0/Star r1 pair: first Star is dead (pattern 3),
        // then Star r1 → Ldar r0 won't match (different reg). But the original
        // Star r0 → Ldar r0 spans across the intervening Star r1 (which
        // preserves acc). The Ldar r0 should be eliminated.
        let opcodes: Vec<_> = instructions.iter().map(|i| i.opcode).collect();
        assert!(
            !opcodes.contains(&Opcode::Ldar),
            "Ldar r0 after Star r0 + Star r1 should be eliminated"
        );
    }

    #[test]
    fn test_ldar_star_same_register_eliminated() {
        // Ldar r0 → Star r0 should eliminate the Star.
        let original = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions.iter().map(|i| i.opcode).collect::<Vec<_>>(),
            vec![Opcode::Ldar, Opcode::Return]
        );
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    fn test_ldar_star_different_register_not_eliminated() {
        // Ldar r0 → Star r1 must NOT be eliminated (different register).
        let original = vec![
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        let opcodes: Vec<_> = instructions.iter().map(|i| i.opcode).collect();
        assert_eq!(opcodes, vec![Opcode::Ldar, Opcode::Star, Opcode::Return]);
    }

    #[test]
    fn test_adjacent_star_star_eliminates_first() {
        // Star r0 → Star r1 — the first Star is dead (acc unchanged).
        // After dead-store elimination, LdaSmi+Star(r1) fuses into
        // LdaSmiStar(5, r1).
        let original = vec![
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(5)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        assert_eq!(
            instructions.iter().map(|i| i.opcode).collect::<Vec<_>>(),
            vec![Opcode::LdaSmiStar, Opcode::Return]
        );
        // The LdaSmiStar target register is the second operand (r1).
        assert_eq!(get_register(&instructions[0], 1), Some(1));
        assert_eq!(byte_offsets.len(), instructions.len() + 1);
    }

    #[test]
    fn test_move_elimination_respects_jump_targets() {
        // Star r0 → [jump target] Ldar r0 — must NOT eliminate across a jump
        // target because control flow could enter at the Ldar.
        let unresolved = vec![
            Instruction::new_unchecked(Opcode::Jump, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(1)]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];

        let bytes = encode(&unresolved);
        let (_, offsets) =
            crate::bytecode::bytecodes::decode_with_byte_offsets(&bytes).expect("valid bytecode");
        // Make the jump target the Ldar instruction (index 3).
        let mut resolved = unresolved;
        let target_byte = offsets[3];
        let jump_end_byte = offsets[1];
        *resolved[0].operand_mut(0) =
            Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&resolved);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        // The Ldar must survive because it's a jump target.
        let opcodes: Vec<_> = instructions.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::Ldar),
            "Ldar at a jump target must not be eliminated"
        );
    }

    #[test]
    fn test_licm_hoists_invariant_global_load() {
        let mut original = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(1)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(1), Operand::FeedbackSlot(3)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(0),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(4),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        resolve_jump_offsets_for_test(&mut original, &[(6, 11), (10, 2)]);

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
        hoist_loop_invariants(&mut instructions, &mut byte_offsets, &mut origins, None);

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdaZero,
                Opcode::Star,
                Opcode::LdaGlobal,
                Opcode::Star,
                Opcode::Ldar,
                Opcode::Ldar,
                Opcode::TestLessThan,
                Opcode::JumpIfFalse,
                Opcode::Ldar,
                Opcode::AddSmi,
                Opcode::Star,
                Opcode::JumpLoop,
                Opcode::Return,
            ]
        );
        assert_eq!(get_register(&instructions[3], 0), Some(1));
        assert_eq!(get_register(&instructions[4], 0), Some(1));
        assert_eq!(
            resolve_jump_target_index(&instructions, &byte_offsets, 11),
            Some(4),
            "JumpLoop must still target the loop header after hoisting"
        );
    }

    #[test]
    fn test_licm_hoists_length_load_with_constant_pool() {
        let mut original = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaNamedProperty,
                vec![
                    Operand::Register(2),
                    Operand::ConstantPoolIdx(0),
                    Operand::FeedbackSlot(1),
                ],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(1), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::AddSmi,
                vec![Operand::Immediate(1), Operand::FeedbackSlot(3)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(0),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(4),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        resolve_jump_offsets_for_test(&mut original, &[(6, 11), (10, 2)]);

        let pool = vec![ConstantPoolEntry::String("length".into())];
        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
        hoist_loop_invariants(
            &mut instructions,
            &mut byte_offsets,
            &mut origins,
            Some(&pool),
        );

        assert_eq!(
            instructions
                .iter()
                .map(|instr| instr.opcode)
                .collect::<Vec<_>>(),
            vec![
                Opcode::LdaZero,
                Opcode::Star,
                Opcode::LdaNamedProperty,
                Opcode::Star,
                Opcode::Ldar,
                Opcode::Ldar,
                Opcode::TestLessThan,
                Opcode::JumpIfFalse,
                Opcode::Ldar,
                Opcode::AddSmi,
                Opcode::Star,
                Opcode::JumpLoop,
                Opcode::Return,
            ]
        );
        assert_eq!(
            resolve_jump_target_index(&instructions, &byte_offsets, 11),
            Some(4),
            "JumpLoop must target the new loop header after hoisting length"
        );
    }

    #[test]
    fn test_licm_skips_global_load_when_loop_writes_same_global() {
        let mut original = vec![
            Instruction::new_unchecked(Opcode::LdaZero, vec![]),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::LdaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(1)],
            ),
            Instruction::new_unchecked(Opcode::Star, vec![Operand::Register(1)]),
            Instruction::new_unchecked(Opcode::LdaSmi, vec![Operand::Immediate(7)]),
            Instruction::new_unchecked(
                Opcode::StaGlobal,
                vec![Operand::ConstantPoolIdx(0), Operand::FeedbackSlot(2)],
            ),
            Instruction::new_unchecked(Opcode::Ldar, vec![Operand::Register(0)]),
            Instruction::new_unchecked(
                Opcode::TestLessThan,
                vec![Operand::Register(1), Operand::FeedbackSlot(3)],
            ),
            Instruction::new_unchecked(Opcode::JumpIfFalse, vec![Operand::JumpOffset(0)]),
            Instruction::new_unchecked(
                Opcode::JumpLoop,
                vec![
                    Operand::JumpOffset(0),
                    Operand::Immediate(0),
                    Operand::FeedbackSlot(4),
                ],
            ),
            Instruction::new_unchecked(Opcode::Return, vec![]),
        ];
        resolve_jump_offsets_for_test(&mut original, &[(8, 10), (9, 2)]);

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&original);
        let mut origins = (0..instructions.len()).map(Some).collect::<Vec<_>>();
        hoist_loop_invariants(&mut instructions, &mut byte_offsets, &mut origins, None);

        assert_eq!(instructions, original);
    }
}

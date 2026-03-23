//! Peephole fusion over decoded bytecode instructions.

// Dead-store elimination and constant folding are currently disabled (see
// `fuse_instructions`).  Allow the helper functions to remain for future use.
#![allow(dead_code)]

use std::collections::HashSet;

use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};

/// Scan decoded instruction stream and fuse recognized patterns.
pub fn fuse_instructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    if instructions.is_empty() || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    // Dead-store elimination and constant folding are disabled until the
    // remaining correctness issues are resolved (423 test failures from
    // incorrect store elimination across exception-throwing paths).
    // optimize_bytecode(instructions, byte_offsets);
    eliminate_redundant_moves(instructions, byte_offsets);
    fuse_superinstructions(instructions, byte_offsets);
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
        compact_nops(instructions, byte_offsets);
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
    let sentinel = match byte_offsets.last().copied() {
        Some(sentinel) => sentinel,
        None => return,
    };
    let mut compacted_instructions = Vec::with_capacity(instructions.len());
    let mut compacted_offsets = Vec::with_capacity(byte_offsets.len());
    for (index, instruction) in instructions.iter().enumerate() {
        if instruction.opcode == Opcode::Nop {
            continue;
        }
        compacted_offsets.push(byte_offsets[index]);
        compacted_instructions.push(instruction.clone());
    }
    compacted_offsets.push(sentinel);
    *instructions = compacted_instructions;
    *byte_offsets = compacted_offsets;
}

fn fuse_superinstructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    if instructions.len() < 2 || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    let jump_target_bytes = collect_jump_target_bytes(instructions, byte_offsets);
    let sentinel = match byte_offsets.last().copied() {
        Some(sentinel) => sentinel,
        None => return,
    };

    let mut fused_instructions = Vec::with_capacity(instructions.len());
    let mut fused_offsets = Vec::with_capacity(byte_offsets.len());
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
                        first.operands[0],
                        second.operands[0],
                        second.operands[1],
                        third.operands[0],
                    ) {
                        fused_offsets.push(byte_offsets[index]);
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
                        first.operands[0],
                        second.operands[0],
                        second.operands[1],
                        third.operands[0],
                    ) {
                        fused_offsets.push(byte_offsets[index]);
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
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_offsets.push(byte_offsets[index]);
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
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_offsets.push(byte_offsets[index]);
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
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_offsets.push(byte_offsets[index]);
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
                (Opcode::TestEqualStrict, Opcode::JumpIfTrue | Opcode::JumpIfFalse) => {
                    if let (
                        Operand::Register(reg),
                        Operand::FeedbackSlot(slot),
                        Operand::JumpOffset(offset),
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        let is_true = u8::from(matches!(second.opcode, Opcode::JumpIfTrue));
                        fused_offsets.push(byte_offsets[index]);
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
                (Opcode::AddSmi, Opcode::Star) => {
                    if let (
                        Operand::Immediate(imm),
                        Operand::FeedbackSlot(slot),
                        Operand::Register(dst),
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        fused_offsets.push(byte_offsets[index]);
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
                    ) = (first.operands[0], first.operands[1], second.operands[0])
                    {
                        fused_offsets.push(byte_offsets[index]);
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
                _ => {}
            }
        }

        fused_offsets.push(byte_offsets[index]);
        fused_instructions.push(instructions[index].clone());
        index += 1;
    }

    fused_offsets.push(sentinel);
    *instructions = fused_instructions;
    *byte_offsets = fused_offsets;
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
    match instruction.operands.get(operand_index).copied()? {
        Operand::Immediate(value) => Some(value),
        _ => None,
    }
}

fn get_register(instruction: &Instruction, operand_index: usize) -> Option<u32> {
    match instruction.operands.get(operand_index).copied()? {
        Operand::Register(register) => Some(register),
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
        .operands
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
        && let Some(Operand::RegisterCount(count)) = instruction.operands.get(2)
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
        for operand in &instruction.operands {
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
        resolved[0].operands[0] = Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);

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
        // Star r0 → Ldar r0 should eliminate the Ldar.
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
            vec![Opcode::LdaSmi, Opcode::Star, Opcode::Return]
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
            vec![Opcode::LdaSmi, Opcode::Star, Opcode::Return]
        );
        // The surviving Star should target r1.
        assert_eq!(get_register(&instructions[1], 0), Some(1));
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
        resolved[0].operands[0] = Operand::JumpOffset(target_byte as i32 - jump_end_byte as i32);

        let (mut instructions, mut byte_offsets) = decode_with_offsets(&resolved);
        fuse_instructions(&mut instructions, &mut byte_offsets);

        // The Ldar must survive because it's a jump target.
        let opcodes: Vec<_> = instructions.iter().map(|i| i.opcode).collect();
        assert!(
            opcodes.contains(&Opcode::Ldar),
            "Ldar at a jump target must not be eliminated"
        );
    }
}

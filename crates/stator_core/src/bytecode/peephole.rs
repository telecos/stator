//! Peephole fusion over decoded bytecode instructions.

use std::collections::HashSet;

use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};

/// Scan decoded instruction stream and fuse recognized patterns.
pub fn fuse_instructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
    if instructions.is_empty() || byte_offsets.len() != instructions.len() + 1 {
        return;
    }

    optimize_bytecode(instructions, byte_offsets);
    fuse_superinstructions(instructions, byte_offsets);
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
    instruction
        .operands
        .iter()
        .any(|operand| matches!(operand, Operand::Register(value) if *value == register))
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
}

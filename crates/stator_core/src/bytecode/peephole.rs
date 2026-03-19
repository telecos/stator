//! Peephole fusion over decoded bytecode instructions.

use std::collections::HashSet;

use crate::bytecode::bytecodes::{Instruction, Opcode, Operand};

/// Scan decoded instruction stream and fuse recognized patterns.
pub fn fuse_instructions(instructions: &mut Vec<Instruction>, byte_offsets: &mut Vec<usize>) {
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

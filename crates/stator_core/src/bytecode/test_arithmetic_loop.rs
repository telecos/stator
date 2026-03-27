#[cfg(test)]
mod tests {
    use crate::bytecode::bytecode_generator::BytecodeGenerator;
    use crate::bytecode::bytecodes::Opcode;
    use crate::parser::recursive_descent::Parser;

    #[test]
    fn test_arithmetic_loop() {
        let code = r#"var n = 0;
for (var i = 0; i < 10000; i++) {
    n = (n + i * 3 - 1) | 0;
}
n;"#;
        
        let mut parser = Parser::new(code);
        let program = parser.parse_program().expect("parse");
        let mut gen = BytecodeGenerator::new_script();
        let array = gen.emit_program(&program).expect("codegen");
        
        println!("Bytecode instructions ({} total):", array.instructions().len());
        for instr in array.iter_instructions().take(50) {
            println!("  {:?}", instr);
        }
    }
}

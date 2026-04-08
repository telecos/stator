use stator_core::parser::recursive_descent;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;

fn main() {
    let source = r#"
        function make_counter() {
            var count = 0;
            return function() { count = count + 1; return count; };
        }
        var counter = make_counter();
        var result = 0;
        for (var i = 0; i < 1000; i++) {
            result = counter();
        }
        result;
    "#;
    
    let program = recursive_descent::parse(source).unwrap();
    let ba = BytecodeGenerator::compile_program(&program).unwrap();
    
    println!("Bytecodes for closure_counter:\n");
    ba.print_disassembly();
}

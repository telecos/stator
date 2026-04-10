use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::parser::recursive_descent;

fn main() {
    let source = r#"
        var last;
        for (var i = 0; i < 1000; i++) {
            last = { x: i, y: i + 1, z: i * 2 };
        }
        last.x + last.y + last.z;
    "#;
    
    let program = recursive_descent::parse(source).unwrap();
    let bytecode = BytecodeGenerator::compile_program(&program).unwrap();
    
    println!("=== OBJECT_CREATION BYTECODE ===");
    for (i, instr) in bytecode.bytecodes().iter().enumerate() {
        println!("{:3}: {:?}", i, instr);
    }
}

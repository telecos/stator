use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::bytecode::recursive_descent;

fn main() {
    let source1 = r#"
        var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
        }
        sum;
    "#;
    
    let source2 = r#"
        var root = { a: { b: { c: { d: { e: 99 } } } } };
        var sum = 0;
        for (var i = 0; i < 1000; i++) {
            sum = sum + root.a.b.c.d.e;
        }
        sum;
    "#;
    
    println!("=== PROPERTY_ACCESS ===");
    let program1 = recursive_descent::parse(source1).unwrap();
    let bytecode1 = BytecodeGenerator::compile_program(&program1).unwrap();
    for (i, instr) in bytecode1.bytecodes().iter().enumerate() {
        println!("{}: {:?}", i, instr);
    }
    
    println!("\n=== DEEP_OBJECT ===");
    let program2 = recursive_descent::parse(source2).unwrap();
    let bytecode2 = BytecodeGenerator::compile_program(&program2).unwrap();
    for (i, instr) in bytecode2.bytecodes().iter().enumerate() {
        println!("{}: {:?}", i, instr);
    }
}

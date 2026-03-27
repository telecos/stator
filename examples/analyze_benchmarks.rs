use std::collections::HashMap;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::bytecode::bytecodes::Opcode;
use stator_core::parser::recursive_descent;

fn count_opcodes(instructions: &[stator_core::bytecode::bytecodes::Instruction]) -> HashMap<Opcode, usize> {
    let mut counts = HashMap::new();
    for instr in instructions {
        *counts.entry(instr.opcode).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let benchmarks = vec![
        ("arithmetic_loop_1000", r#"var n = 0; for (var i = 0; i < 1000; i++) { n = (n + i) | 0; } n;"#),
        ("property_access_1k", r#"var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 }; var sum = 0; for (var i = 0; i < 1000; i++) { sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e; } sum;"#),
        ("object_creation_1k", r#"var last; for (var i = 0; i < 1000; i++) { last = { x: i, y: i + 1, z: i * 2 }; } last.x + last.y + last.z;"#),
        ("string_concat_500", r#"var s = ""; for (var i = 0; i < 500; i++) { s = s + "x"; } s.length;"#),
        ("fib_40_iterative", r#"var a = 0, b = 1; for (var i = 0; i < 40; i++) { var t = a + b; a = b; b = t; } b;"#),
        ("array_push_sum_1k", r#"var arr = []; for (var i = 0; i < 100; i++) { arr.push(i); } var sum = 0; for (var i = 0; i < arr.length; i++) { sum = sum + arr[i]; } sum;"#),
        ("deep_object_access_1k", r#"var root = { a: { b: { c: { d: { e: 99 } } } } }; var sum = 0; for (var i = 0; i < 1000; i++) { sum = sum + root.a.b.c.d.e; } sum;"#),
        ("closure_counter_100", r#"function make_counter() { var count = 0; return function() { count = count + 1; return count; }; } var counter = make_counter(); var result = 0; for (var i = 0; i < 100; i++) { result = counter(); } result;"#),
        ("sieve_primes_1k", r#"var n = 1000; var sieve = []; for (var i = 0; i <= n; i++) sieve[i] = true; sieve[0] = false; sieve[1] = false; for (var i = 2; i * i <= n; i++) { if (sieve[i]) { for (var j = i * i; j <= n; j = j + i) { sieve[j] = false; } } } var count = 0; for (var i = 0; i <= n; i++) { if (sieve[i]) count = count + 1; } count;"#),
    ];

    for (name, source) in benchmarks {
        match recursive_descent::parse(source) {
            Ok(program) => {
                match BytecodeGenerator::compile_program(&program) {
                    Ok(bytecode) => {
                        match bytecode.instructions() {
                            Ok(instructions) => {
                                let counts = count_opcodes(&instructions);
                                println!("Benchmark: {}", name);
                                println!("  Total instructions: {}", instructions.len());
                                println!("  LdaGlobal: {}", counts.get(&Opcode::LdaGlobal).unwrap_or(&0));
                                println!("  StaGlobal: {}", counts.get(&Opcode::StaGlobal).unwrap_or(&0));
                                println!("  LdaNamedProperty: {}", counts.get(&Opcode::LdaNamedProperty).unwrap_or(&0));
                                println!("  StaNamedProperty: {}", counts.get(&Opcode::StaNamedProperty).unwrap_or(&0));
                                println!("  CallProperty*: {}", 
                                    counts.get(&Opcode::CallProperty).unwrap_or(&0) +
                                    counts.get(&Opcode::CallProperty0).unwrap_or(&0) +
                                    counts.get(&Opcode::CallProperty1).unwrap_or(&0) +
                                    counts.get(&Opcode::CallProperty2).unwrap_or(&0));
                                println!("  CallUndefinedReceiver*: {}", 
                                    counts.get(&Opcode::CallUndefinedReceiver0).unwrap_or(&0) +
                                    counts.get(&Opcode::CallUndefinedReceiver1).unwrap_or(&0) +
                                    counts.get(&Opcode::CallUndefinedReceiver2).unwrap_or(&0));
                                println!("  LdaKeyedProperty: {}", counts.get(&Opcode::LdaKeyedProperty).unwrap_or(&0));
                                println!("  StaKeyedProperty: {}", counts.get(&Opcode::StaKeyedProperty).unwrap_or(&0));
                                println!();
                            }
                            Err(e) => println!("Decode error for {}: {:?}", name, e),
                        }
                    }
                    Err(e) => println!("Compile error for {}: {:?}", name, e),
                }
            }
            Err(e) => println!("Parse error for {}: {:?}", name, e),
        }
    }
}

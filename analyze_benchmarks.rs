use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::bytecode::bytecodes::Opcode;
use stator_core::parser::recursive_descent;

fn count_opcode_in_bytecode(bytecode: &[u8], target_opcode: Opcode) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < bytecode.len() {
        if bytecode[i] == target_opcode as u8 {
            count += 1;
        }
        // Skip to next instruction - rough approximation
        // Most instructions are 1-4 bytes
        i += 1;
    }
    count
}

fn main() {
    let benchmarks = vec![
        ("arithmetic_loop_1000", r#"
            var n = 0;
            for (var i = 0; i < 1000; i++) {
                n = (n + i) | 0;
            }
            n;
        "#),
        ("property_access_1k", r#"
            var obj = { a: 1, b: 2, c: 3, d: 4, e: 5 };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + obj.a + obj.b + obj.c + obj.d + obj.e;
            }
            sum;
        "#),
        ("object_creation_1k", r#"
            var last;
            for (var i = 0; i < 1000; i++) {
                last = { x: i, y: i + 1, z: i * 2 };
            }
            last.x + last.y + last.z;
        "#),
        ("string_concat_500", r#"
            var s = "";
            for (var i = 0; i < 500; i++) {
                s = s + "x";
            }
            s.length;
        "#),
        ("fib_40_iterative", r#"
            var a = 0, b = 1;
            for (var i = 0; i < 40; i++) {
                var t = a + b;
                a = b;
                b = t;
            }
            b;
        "#),
        ("array_push_sum_1k", r#"
            var arr = [];
            for (var i = 0; i < 100; i++) {
                arr.push(i);
            }
            var sum = 0;
            for (var i = 0; i < arr.length; i++) {
                sum = sum + arr[i];
            }
            sum;
        "#),
        ("deep_object_access_1k", r#"
            var root = { a: { b: { c: { d: { e: 99 } } } } };
            var sum = 0;
            for (var i = 0; i < 1000; i++) {
                sum = sum + root.a.b.c.d.e;
            }
            sum;
        "#),
        ("closure_counter_100", r#"
            function make_counter() {
                var count = 0;
                return function() { count = count + 1; return count; };
            }
            var counter = make_counter();
            var result = 0;
            for (var i = 0; i < 100; i++) {
                result = counter();
            }
            result;
        "#),
        ("sieve_primes_1k", r#"
            var n = 1000;
            var sieve = [];
            for (var i = 0; i <= n; i++) sieve[i] = true;
            sieve[0] = false;
            sieve[1] = false;
            for (var i = 2; i * i <= n; i++) {
                if (sieve[i]) {
                    for (var j = i * i; j <= n; j = j + i) {
                        sieve[j] = false;
                    }
                }
            }
            var count = 0;
            for (var i = 0; i <= n; i++) {
                if (sieve[i]) count = count + 1;
            }
            count;
        "#),
    ];

    for (name, source) in benchmarks {
        match recursive_descent::parse(source) {
            Ok(program) => {
                match BytecodeGenerator::compile_program(&program) {
                    Ok(bytecode) => {
                        let bytes = bytecode.bytes();
                        println!("Benchmark: {}", name);
                        println!("  Bytecode size: {} bytes", bytes.len());
                        println!("  Bytecode (hex): {:?}", bytes);
                    }
                    Err(e) => println!("Compile error for {}: {:?}", name, e),
                }
            }
            Err(e) => println!("Parse error for {}: {:?}", name, e),
        }
        println!();
    }
}

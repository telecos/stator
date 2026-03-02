#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_fuzz::program_from_bytes;

fuzz_target!(|data: &[u8]| {
    let program = program_from_bytes(data, 16);
    // The compiler must not panic regardless of the AST shape; errors are fine.
    let _ = BytecodeGenerator::compile_program(&program);
});

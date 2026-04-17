#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_jse::bytecode::bytecode_generator::BytecodeGenerator;
use stator_jse::bytecode::bytecodes::encode;
use stator_fuzz::program_from_bytes;

fuzz_target!(|data: &[u8]| {
    let program = program_from_bytes(data, 16);

    // Compile the program; skip if compilation fails (errors are acceptable).
    let Ok(array) = BytecodeGenerator::compile_program(&program) else {
        return;
    };

    // Roundtrip: decode the bytecode stream back to instructions, then
    // re-encode them and verify the resulting bytes are identical.
    let Ok(instructions) = array.instructions() else {
        return;
    };
    let re_encoded = encode(&instructions);
    assert_eq!(
        array.bytecodes(),
        re_encoded.as_slice(),
        "bytecode roundtrip must produce identical bytes"
    );
});

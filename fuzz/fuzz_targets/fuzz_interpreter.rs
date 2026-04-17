#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_jse::bytecode::bytecode_generator::BytecodeGenerator;
use stator_jse::interpreter::{Interpreter, InterpreterFrame};
use stator_jse::parser::parse;

fuzz_target!(|data: &[u8]| {
    // Accept arbitrary bytes as UTF-8 source (replace invalid sequences).
    let source = String::from_utf8_lossy(data);

    // Parse the source; syntax errors are acceptable.
    let Ok(program) = parse(&source) else {
        return;
    };

    // Compile to bytecode; compiler errors are acceptable.
    let Ok(bytecode) = BytecodeGenerator::compile_program(&program) else {
        return;
    };

    // Interpret; runtime errors are acceptable — panics are not.
    let mut frame = InterpreterFrame::new(bytecode, vec![]);
    let _ = Interpreter::run(&mut frame);
});

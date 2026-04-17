#![no_main]

//! Differential fuzzer: interpreter vs. baseline JIT.
//!
//! For each fuzz input we:
//! 1. Parse and compile it to bytecode.
//! 2. Run the bytecode through the **interpreter** and record the result.
//! 3. Compile the same bytecode through the **baseline JIT** and execute it.
//! 4. Assert that both execution paths produce the same `JsValue` result.
//!
//! Any divergence between interpreter and JIT is a bug.

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

    // ── Step 1: run through the interpreter ───────────────────────────────
    let interp_result = {
        let mut frame = InterpreterFrame::new(bytecode.clone(), vec![]);
        Interpreter::run(&mut frame)
    };

    // If the interpreter errors out (e.g. type error, unimplemented opcode)
    // there is nothing to compare — both sides would need to agree on the
    // error kind, which is out of scope for a basic differential fuzzer.
    let Ok(interp_val) = interp_result else {
        return;
    };

    // ── Step 2: run through the baseline JIT (x86-64 Unix only) ──────────
    //
    // On other platforms the JIT is not available and we skip the comparison.
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use stator_jse::compiler::baseline::compiler::{
            BaselineCompiler, JIT_UNDEFINED, jit_to_jsvalue,
        };

        // Compile to native code; skip if unsupported (errors are acceptable).
        let Ok(cc) = BaselineCompiler::compile(&bytecode) else {
            return;
        };

        // Build JIT arguments: parameters default to `undefined` in the JIT
        // representation (`JIT_UNDEFINED`).
        let jit_args: Vec<i64> = vec![JIT_UNDEFINED; bytecode.parameter_count() as usize];

        // Execute the JIT code.
        //
        // SAFETY: `cc.code` was produced by `BaselineCompiler::compile` and
        // contains valid x86-64 machine code emitted by the baseline compiler.
        let jit_raw = match unsafe { cc.execute(&jit_args) } {
            Ok(v) => v,
            // JIT_DEOPT or mmap failure: the JIT gracefully declined to
            // execute this input — no divergence to report.
            Err(_) => return,
        };

        // Convert the raw i64 to a JsValue; if it cannot be represented
        // (e.g. the result is a heap object) both sides are incomparable.
        let Some(jit_val) = jit_to_jsvalue(jit_raw) else {
            return;
        };

        // ── Differential assertion ─────────────────────────────────────────
        assert_eq!(
            interp_val, jit_val,
            "interpreter and JIT produced different results for the same input\n\
             interpreter: {interp_val:?}\n\
             jit:         {jit_val:?}"
        );
    }

    // On non-JIT platforms suppress the "unused variable" warning.
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    let _ = interp_val;
});

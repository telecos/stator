#![no_main]

use libfuzzer_sys::fuzz_target;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::parser::parse;

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

    // Attempt baseline JIT compilation.  This is only meaningful on x86-64
    // Unix; on all other platforms the compiler is a no-op stub and we just
    // verify that the bytecode itself is well-formed.
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        use stator_core::compiler::baseline::compiler::{
            BaselineCompiler, CompiledCode, FOOTER_SIZE, METADATA_MAGIC,
        };

        match BaselineCompiler::compile(&bytecode) {
            Err(_) => {
                // Compilation errors (unsupported opcodes, etc.) are acceptable.
            }
            Ok(cc) => {
                // The compiled code buffer must be non-empty and end with the
                // expected metadata magic number.
                assert!(
                    !cc.code.is_empty(),
                    "compiled code buffer must not be empty"
                );
                assert!(
                    cc.code.len() >= FOOTER_SIZE,
                    "compiled code must contain at least the metadata footer"
                );
                let magic_bytes = &cc.code[cc.code.len() - 4..];
                let magic = u32::from_le_bytes(magic_bytes.try_into().unwrap());
                assert_eq!(
                    magic, METADATA_MAGIC,
                    "metadata footer magic must match METADATA_MAGIC"
                );

                // The native code portion must be non-empty (at least a
                // prologue + Return + deopt epilogue).
                assert!(
                    cc.native_code_len > 0,
                    "native_code_len must be > 0 for any compiled function"
                );

                // Safepoint and deopt tables must round-trip cleanly.
                let parsed_sp = CompiledCode::parse_safepoints(&cc.code);
                assert!(
                    parsed_sp.is_some(),
                    "safepoint table must be parseable from compiled code"
                );
                let parsed_de = CompiledCode::parse_deopt_entries(&cc.code);
                assert!(
                    parsed_de.is_some(),
                    "deopt table must be parseable from compiled code"
                );

                // The in-memory vectors and the deserialized tables must agree.
                assert_eq!(
                    cc.safepoints.len(),
                    parsed_sp.unwrap().len(),
                    "in-memory and serialized safepoint counts must match"
                );
                assert_eq!(
                    cc.deopt_entries.len(),
                    parsed_de.unwrap().len(),
                    "in-memory and serialized deopt entry counts must match"
                );
            }
        }
    }

    // On non-x86_64/Unix platforms just confirm bytecode decoding succeeds.
    #[cfg(not(all(target_arch = "x86_64", unix)))]
    {
        let _ = bytecode.instructions();
    }
});

//! `st8` — Stator JavaScript shell.
//!
//! `st8` is the command-line shell for the Stator engine, analogous to
//! V8's `d8`.  It executes `.js` files, providing a `print()` global so
//! that differential tests can produce stdout output comparable with d8.
//!
//! # Usage
//!
//! ```text
//! st8 <script.js>          # Execute a JavaScript file
//! st8 -e "<code>"          # Execute a JavaScript snippet
//! ```
//!
//! Exit codes:
//! - `0` — script completed without an uncaught exception
//! - `1` — uncaught exception or engine error

use std::rc::Rc;

use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::error::StatorError;
use stator_core::interpreter::{Interpreter, InterpreterFrame, NativeFn, set_global};
use stator_core::objects::value::JsValue;
use stator_core::parser::parse;

/// Register the host-supplied globals that `st8` makes available to scripts.
///
/// Currently provides:
/// - `print(…)` — converts each argument to a string (ECMAScript ToString)
///   and writes them to stdout separated by spaces, followed by a newline.
///   Matches d8's `print()` built-in.
fn register_globals() {
    set_global(
        "print".to_string(),
        JsValue::NativeFunction(NativeFn(Rc::new(|args: &[JsValue]| {
            let parts: Vec<String> = args
                .iter()
                .map(|v| v.to_js_string().unwrap_or_else(|_| "[object]".to_string()))
                .collect();
            println!("{}", parts.join(" "));
            Ok(JsValue::Undefined)
        }))),
    );
}

/// Execute `source` as JavaScript, returning `Ok(())` on success or
/// `Err(message)` when an uncaught exception or engine error occurs.
fn run_source(source: &str) -> Result<(), String> {
    let program = parse(source).map_err(|e| match e {
        StatorError::SyntaxError(msg) => format!("SyntaxError: {msg}"),
        other => format!("{other:?}"),
    })?;

    let bytecode = BytecodeGenerator::compile_program(&program).map_err(|e| match e {
        StatorError::SyntaxError(msg) => format!("SyntaxError: {msg}"),
        other => format!("{other:?}"),
    })?;

    let mut frame = InterpreterFrame::new(bytecode, vec![]);
    Interpreter::run(&mut frame).map_err(|e| match e {
        StatorError::JsException(msg) => msg,
        StatorError::TypeError(msg) => format!("TypeError: {msg}"),
        StatorError::ReferenceError(msg) => format!("ReferenceError: {msg}"),
        StatorError::RangeError(msg) => format!("RangeError: {msg}"),
        other => format!("{other:?}"),
    })?;

    Ok(())
}

fn main() {
    register_globals();

    let args: Vec<String> = std::env::args().collect();

    let source = if args.len() >= 3 && args[1] == "-e" {
        // Inline snippet: st8 -e "<code>"
        args[2].clone()
    } else if args.len() >= 2 {
        // File execution: st8 <script.js>
        let path = &args[1];
        match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("st8: cannot read '{}': {}", path, e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Usage: st8 <script.js>");
        eprintln!("       st8 -e \"<code>\"");
        std::process::exit(1);
    };

    if let Err(msg) = run_source(&source) {
        eprintln!("{msg}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_source_arithmetic() {
        assert!(run_source("var x = 1 + 2;").is_ok());
    }

    #[test]
    fn test_run_source_syntax_error() {
        let result = run_source("var = ;");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("SyntaxError"),
            "expected SyntaxError, got: {msg}"
        );
    }

    #[test]
    fn test_run_source_throw() {
        let result = run_source("throw 42;");
        assert!(result.is_err());
    }

    #[test]
    fn test_register_globals_print_is_callable() {
        // Verify that a script calling print() does not error.
        register_globals();
        assert!(run_source("print(1 + 2);").is_ok());
    }

    #[test]
    fn test_run_source_if_else() {
        assert!(run_source("var x = 1; var r = 0; if (x > 0) { r = x; }").is_ok());
    }
}

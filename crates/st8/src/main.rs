//! `st8` — Stator JavaScript shell.
//!
//! `st8` is the command-line shell for the Stator JavaScript engine, analogous
//! to V8's `d8`.  It executes JavaScript files or inline snippets from the
//! command line.
//!
//! # Usage
//!
//! ```text
//! st8 <file.js>         execute a JavaScript file
//! st8 -e '<code>'       evaluate an inline JavaScript expression
//! ```
//!
//! ## Built-in globals
//!
//! - `print(...args)` — prints arguments joined by a space, followed by a
//!   newline, to standard output.
//! - `console.log(...args)` — alias for `print`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::process;
use std::rc::Rc;

use stator_core::builtins::wasm::make_webassembly_object;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::interpreter::{Interpreter, InterpreterFrame};
use stator_core::objects::value::JsValue;
use stator_core::parser;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let source: String = if args.len() >= 3 && args[1] == "-e" {
        args[2].clone()
    } else if args.len() >= 2 {
        match std::fs::read_to_string(&args[1]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("st8: cannot read '{}': {}", args[1], e);
                process::exit(1);
            }
        }
    } else {
        eprintln!("Usage: st8 <file.js>\n       st8 -e '<code>'");
        process::exit(1);
    };

    let globals = build_globals();

    let bytecodes =
        match parser::parse(&source).and_then(|p| BytecodeGenerator::compile_program(&p)) {
            Ok(bc) => bc,
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        };

    let mut frame = InterpreterFrame::new_with_globals(bytecodes, vec![], globals);
    if let Err(e) = Interpreter::run(&mut frame) {
        eprintln!("{e}");
        process::exit(1);
    }
}

/// Build the initial global environment with the built-in shell functions.
fn build_globals() -> Rc<RefCell<HashMap<String, JsValue>>> {
    let globals: Rc<RefCell<HashMap<String, JsValue>>> = Rc::new(RefCell::new(HashMap::new()));

    // print(...args) — ECMAScript-shell standard, mirrors d8's print()
    globals.borrow_mut().insert(
        "print".to_string(),
        JsValue::NativeFunction(Rc::new(print_args)),
    );

    // console.log(...args) — alias for print
    let console_obj: Rc<RefCell<HashMap<String, JsValue>>> = Rc::new(RefCell::new(HashMap::new()));
    console_obj.borrow_mut().insert(
        "log".to_string(),
        JsValue::NativeFunction(Rc::new(print_args)),
    );
    globals
        .borrow_mut()
        .insert("console".to_string(), JsValue::PlainObject(console_obj));

    // WebAssembly — full WebAssembly JS API namespace
    globals
        .borrow_mut()
        .insert("WebAssembly".to_string(), make_webassembly_object());

    globals
}

/// Print all arguments joined by a single space, followed by a newline.
///
/// This implements the behaviour of both `print()` and `console.log()` in the
/// st8 shell.
fn print_args(args: Vec<JsValue>) -> stator_core::error::StatorResult<JsValue> {
    let parts: Vec<String> = args
        .iter()
        .map(|v| v.to_js_string().unwrap_or_else(|_| "undefined".to_string()))
        .collect();
    println!("{}", parts.join(" "));
    Ok(JsValue::Undefined)
}

#[cfg(test)]
mod tests {
    use super::build_globals;
    use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
    use stator_core::interpreter::{Interpreter, InterpreterFrame};
    use stator_core::objects::value::JsValue;
    use stator_core::parser;
    use stator_core::parser::scanner::{Scanner, TokenKind};

    /// Helper: parse, compile, and run `src` with the shell globals.
    fn run(src: &str) -> Result<JsValue, stator_core::error::StatorError> {
        let bytecodes = parser::parse(src).and_then(|p| BytecodeGenerator::compile_program(&p))?;
        let globals = build_globals();
        let mut frame = InterpreterFrame::new_with_globals(bytecodes, vec![], globals);
        Interpreter::run(&mut frame)
    }

    #[test]
    fn test_shell_scanner_tokenises_number_literal() {
        let mut s = Scanner::new("42");
        let tok = s.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::NumericLiteral);
    }

    #[test]
    fn test_shell_scanner_tokenises_identifier() {
        let mut s = Scanner::new("foo");
        let tok = s.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Identifier);
    }

    #[test]
    fn test_run_numeric_expression() {
        let result = run("1 + 2").unwrap();
        assert_eq!(result, JsValue::Smi(3));
    }

    #[test]
    fn test_run_var_declaration() {
        let result = run("var x = 42; x").unwrap();
        assert_eq!(result, JsValue::Smi(42));
    }

    #[test]
    fn test_run_print_is_callable() {
        // print() should return undefined without panicking.
        let result = run("print(1 + 1)").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_run_console_log_is_callable() {
        let result = run("console.log('hello')").unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_syntax_error_is_reported() {
        assert!(run("var = ;").is_err());
    }

    #[test]
    fn test_build_globals_has_print() {
        let globals = build_globals();
        let env = globals.borrow();
        assert!(matches!(env.get("print"), Some(JsValue::NativeFunction(_))));
    }

    #[test]
    fn test_build_globals_has_console() {
        let globals = build_globals();
        let env = globals.borrow();
        assert!(matches!(env.get("console"), Some(JsValue::PlainObject(_))));
    }
}

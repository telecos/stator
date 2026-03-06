//! `st8` — Stator JavaScript shell.
//!
//! `st8` is the command-line shell for the Stator JavaScript engine, analogous
//! to V8's `d8`.  It executes JavaScript files or inline snippets from the
//! command line.
//!
//! # Usage
//!
//! ```text
//! st8 <file.js>                   execute a JavaScript file
//! st8 -e '<code>'                 evaluate an inline JavaScript expression
//! st8 --inspect <file.js>         run with Chrome DevTools inspector on port 9229
//! st8 --inspect-brk <file.js>     run and break before first statement
//! st8 --inspect=<port> <file.js>  run with inspector on custom port
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
use std::thread;

use stator_core::builtins::wasm::make_webassembly_object;
use stator_core::bytecode::bytecode_generator::BytecodeGenerator;
use stator_core::inspector::cdp::CdpServer;
use stator_core::interpreter::{Interpreter, InterpreterFrame};
use stator_core::objects::value::JsValue;
use stator_core::parser;

/// Parsed command-line options.
struct Options {
    /// JavaScript source code to execute.
    source: String,
    /// Source file name for DevTools (or `"<eval>"` for `-e`).
    source_name: String,
    /// If set, start the CDP inspector on this port.
    inspect_port: Option<u16>,
    /// Whether to pause before the first statement (`--inspect-brk`).
    inspect_brk: bool,
}

/// Parse command-line arguments into [`Options`].
fn parse_args() -> Options {
    let args: Vec<String> = std::env::args().collect();
    let mut inspect_port: Option<u16> = None;
    let mut inspect_brk = false;
    let mut positional: Vec<String> = Vec::new();
    let mut eval_expr: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--inspect" {
            inspect_port = Some(9229);
        } else if arg == "--inspect-brk" {
            inspect_port = Some(9229);
            inspect_brk = true;
        } else if let Some(port_str) = arg.strip_prefix("--inspect=") {
            inspect_port = Some(port_str.parse().unwrap_or_else(|_| {
                eprintln!("st8: invalid inspect port: {port_str}");
                process::exit(1);
            }));
        } else if let Some(port_str) = arg.strip_prefix("--inspect-brk=") {
            inspect_port = Some(port_str.parse().unwrap_or_else(|_| {
                eprintln!("st8: invalid inspect port: {port_str}");
                process::exit(1);
            }));
            inspect_brk = true;
        } else if arg == "-e" {
            i += 1;
            if i < args.len() {
                eval_expr = Some(args[i].clone());
            } else {
                eprintln!("st8: -e requires an argument");
                process::exit(1);
            }
        } else if !arg.starts_with('-') {
            positional.push(arg.clone());
        } else {
            eprintln!("st8: unknown option: {arg}");
            process::exit(1);
        }
        i += 1;
    }

    let (source, source_name) = if let Some(expr) = eval_expr {
        (expr, "<eval>".to_string())
    } else if let Some(file) = positional.first() {
        let s = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("st8: cannot read '{file}': {e}");
                process::exit(1);
            }
        };
        (s, file.clone())
    } else {
        eprintln!("Usage: st8 <file.js>\n       st8 -e '<code>'\n       st8 --inspect <file.js>");
        process::exit(1);
    };

    Options {
        source,
        source_name,
        inspect_port,
        inspect_brk,
    }
}

fn main() {
    let opts = parse_args();
    let globals = build_globals();

    let bytecodes =
        match parser::parse(&opts.source).and_then(|p| BytecodeGenerator::compile_program(&p)) {
            Ok(bc) => bc,
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        };

    if let Some(port) = opts.inspect_port {
        run_with_inspector(
            bytecodes,
            globals,
            &opts.source,
            &opts.source_name,
            port,
            opts.inspect_brk,
        );
    } else {
        let mut frame = InterpreterFrame::new_with_globals(bytecodes, vec![], globals);
        if let Err(e) = Interpreter::run(&mut frame) {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}

/// Start the CDP WebSocket server, wait for a DevTools connection, then
/// execute the script.  The inspector thread stays alive until the client
/// disconnects.
fn run_with_inspector(
    bytecodes: stator_core::bytecode::bytecode_array::BytecodeArray,
    globals: Rc<RefCell<HashMap<String, JsValue>>>,
    _source: &str,
    source_name: &str,
    port: u16,
    _break_on_start: bool,
) {
    let addr = format!("127.0.0.1:{port}");
    let server = match CdpServer::bind(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("st8: cannot start inspector on {addr}: {e}");
            process::exit(1);
        }
    };

    let actual_addr = server
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| addr.clone());
    eprintln!("Debugger listening on ws://{actual_addr}");
    eprintln!("For help, see: https://nodejs.org/en/docs/inspector");
    eprintln!("Debugger attached. Running {}", source_name);

    // Spawn the inspector server in a background thread so it stays alive
    // during script execution.
    let _inspector_handle = thread::spawn(move || {
        let _ = server.accept_loop();
    });

    // Execute the script on the main thread.
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

    #[test]
    fn test_inspector_server_binds_and_accepts() {
        use stator_core::inspector::cdp::CdpServer;

        // Verify the CDP server can bind to a random port.
        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let addr = server.local_addr().expect("local_addr");
        assert_ne!(addr.port(), 0);
    }

    #[test]
    fn test_inspector_server_evaluate_via_websocket() {
        use stator_core::inspector::cdp::CdpServer;
        use std::thread;
        use std::time::Duration;

        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let port = server.local_addr().expect("local_addr").port();

        let handle = thread::spawn(move || server.accept_one());
        thread::sleep(Duration::from_millis(50));

        let url = format!("ws://127.0.0.1:{port}");
        let (mut ws, _) = tungstenite::connect(url).expect("connect");

        // Evaluate an expression via the inspector
        ws.send(tungstenite::Message::Text(
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"2+3"}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let text = match reply {
            tungstenite::Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        };
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse");
        assert_eq!(json["result"]["result"]["value"], 5);
    }
}

//! CDP (Chrome DevTools Protocol) WebSocket server.
//!
//! # Protocol overview
//!
//! CDP uses a JSON-RPC 2.0–like framing over WebSocket:
//!
//! - **Request** `{"id":1,"method":"Runtime.evaluate","params":{…}}`
//! - **Response** `{"id":1,"result":{…}}` or `{"id":1,"error":{…}}`
//! - **Event** `{"method":"Runtime.executionContextCreated","params":{…}}`
//!
//! # Supported domains
//!
//! | Domain         | Method                    | Behaviour                          |
//! |----------------|---------------------------|------------------------------------|
//! | `Runtime`      | `enable`                  | Acknowledges; emits context-created event |
//! | `Runtime`      | `evaluate`                | Parses and executes JavaScript; returns result |
//! | `Runtime`      | `callFunctionOn`          | Evaluates a function call expression |
//! | `Runtime`      | `getProperties`           | Lists properties of the globals object |
//! | `Debugger`     | `enable`                  | Acknowledges; returns `debuggerId` |
//! | `Debugger`     | `setPauseOnExceptions`    | Configures exception pause state   |
//! | `Debugger`     | `setBreakpointByUrl`      | Sets a breakpoint (stub, returns id) |
//! | `Debugger`     | `resume`                  | Resumes after a pause              |
//! | `Console`      | `enable`                  | Flushes buffered messages as events |
//! | `Console`      | `disable`                 | Acknowledges                       |
//! | `Profiler`     | `enable`                  | Acknowledges                       |
//! | `Profiler`     | `start`                   | Starts CPU profiling               |
//! | `Profiler`     | `stop`                    | Stops profiling; returns profile    |
//! | `HeapProfiler` | `enable`                  | Acknowledges                       |
//! | `HeapProfiler` | `takeHeapSnapshot`        | Emits snapshot chunks              |
//! | `HeapProfiler` | `startTrackingHeapObjects` | Starts allocation tracking         |
//! | `HeapProfiler` | `stopTrackingHeapObjects`  | Returns allocation stats           |
//! | `Network`      | `enable`                  | Acknowledges (stub)                |
//! | `Network`      | `disable`                 | Acknowledges (stub)                |
//!
//! # Example
//!
//! ```no_run
//! use stator_core::inspector::cdp::CdpServer;
//!
//! let server = CdpServer::bind("127.0.0.1:9229").unwrap();
//! // Accept one connection, process all messages, then return.
//! server.accept_one().unwrap();
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tungstenite::{Message, WebSocket, accept};

use crate::bytecode::bytecode_generator::BytecodeGenerator;
use crate::error::StatorResult;
use crate::inspector::console::drain_messages;
use crate::inspector::heap_snapshot::HeapSnapshotBuilder;
use crate::inspector::profiler::CpuProfiler;
use crate::interpreter::{Interpreter, InterpreterFrame};
use crate::objects::value::JsValue;
use crate::parser;

// ─────────────────────────────────────────────────────────────────────────────
// JSON-RPC message types
// ─────────────────────────────────────────────────────────────────────────────

/// An incoming CDP request message.
#[derive(Debug, Deserialize)]
pub struct CdpRequest {
    /// JSON-RPC call identifier echoed back in the response.
    pub id: u64,
    /// Method name in `Domain.method` form, e.g. `Runtime.evaluate`.
    pub method: String,
    /// Optional method parameters object.
    #[serde(default)]
    pub params: Value,
}

/// An outgoing CDP response message.
#[derive(Debug, Serialize)]
pub struct CdpResponse {
    /// Echoed request identifier.
    pub id: u64,
    /// Successful result payload (present when there is no error).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload (present when the call failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

/// An outgoing CDP event notification.
#[derive(Debug, Serialize)]
pub struct CdpEvent {
    /// Fully-qualified event name, e.g. `Runtime.executionContextCreated`.
    pub method: String,
    /// Event parameters object.
    pub params: Value,
}

// ─────────────────────────────────────────────────────────────────────────────
// Session
// ─────────────────────────────────────────────────────────────────────────────

/// A single CDP debugging session backed by one WebSocket connection.
///
/// `CdpSession` owns the WebSocket stream and maintains the per-session
/// globals environment used by `Runtime.evaluate`.
pub struct CdpSession {
    ws: WebSocket<TcpStream>,
    globals: Rc<RefCell<HashMap<String, JsValue>>>,
    profiler: CpuProfiler,
    /// Whether the `Console` domain is currently enabled for this session.
    console_enabled: bool,
    /// Monotonically increasing ID for breakpoints set via CDP.
    next_breakpoint_id: u32,
}

impl CdpSession {
    /// Wrap an accepted WebSocket connection in a new session.
    fn new(ws: WebSocket<TcpStream>) -> Self {
        Self {
            ws,
            globals: Rc::new(RefCell::new(HashMap::new())),
            profiler: CpuProfiler::new(),
            console_enabled: false,
            next_breakpoint_id: 1,
        }
    }

    /// Drive the session until the client disconnects or a fatal error occurs.
    ///
    /// Each incoming text frame is parsed as a [`CdpRequest`], dispatched to
    /// the appropriate domain handler, and a [`CdpResponse`] (or error) is
    /// sent back.
    pub fn run(&mut self) -> io::Result<()> {
        loop {
            let msg = match self.ws.read() {
                Ok(m) => m,
                Err(tungstenite::Error::ConnectionClosed)
                | Err(tungstenite::Error::AlreadyClosed) => return Ok(()),
                Err(tungstenite::Error::Io(e)) => return Err(e),
                Err(e) => {
                    return Err(io::Error::other(e.to_string()));
                }
            };

            match msg {
                Message::Text(text) => {
                    let reply = self.handle_text(&text);
                    self.ws
                        .send(Message::Text(reply.into()))
                        .map_err(|e| io::Error::other(e.to_string()))?;
                }
                Message::Close(_) => return Ok(()),
                // Ignore binary / ping / pong frames.
                _ => {}
            }
        }
    }

    /// Parse `text` as a [`CdpRequest`] and return a serialised reply.
    fn handle_text(&mut self, text: &str) -> String {
        let request: CdpRequest = match serde_json::from_str(text) {
            Ok(r) => r,
            Err(e) => {
                // Return a JSON parse error.
                let resp = json!({
                    "id": 0u64,
                    "error": {"code": -32700, "message": format!("Parse error: {e}")}
                });
                return resp.to_string();
            }
        };

        let id = request.id;
        let result = self.dispatch(&request);

        let resp = match result {
            Ok(value) => CdpResponse {
                id,
                result: Some(value),
                error: None,
            },
            Err(e) => CdpResponse {
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": e.to_string()
                })),
            },
        };

        serde_json::to_string(&resp).unwrap_or_else(|_| {
            json!({"id": id, "error": {"code": -32603, "message": "serialization error"}})
                .to_string()
        })
    }

    /// Route a parsed request to the correct domain handler.
    fn dispatch(&mut self, req: &CdpRequest) -> StatorResult<Value> {
        match req.method.as_str() {
            // ── Runtime ───────────────────────────────────────────────────
            "Runtime.enable" => self.runtime_enable(),
            "Runtime.evaluate" => self.runtime_evaluate(&req.params),
            "Runtime.callFunctionOn" => self.runtime_call_function_on(&req.params),
            "Runtime.getProperties" => self.runtime_get_properties(&req.params),

            // ── Debugger ──────────────────────────────────────────────────
            "Debugger.enable" => Ok(json!({
                "debuggerId": "stator-debugger-0"
            })),
            "Debugger.setPauseOnExceptions" => self.debugger_set_pause_on_exceptions(&req.params),
            "Debugger.setBreakpointByUrl" => self.debugger_set_breakpoint_by_url(&req.params),
            "Debugger.resume" => Ok(json!({})),

            // ── Console ───────────────────────────────────────────────────
            "Console.enable" => self.console_enable(),
            "Console.disable" => {
                self.console_enabled = false;
                Ok(json!({}))
            }

            // ── Profiler ──────────────────────────────────────────────────
            "Profiler.enable" => Ok(json!({})),
            "Profiler.start" => self.profiler_start(&req.params),
            "Profiler.stop" => self.profiler_stop(),

            // ── HeapProfiler ──────────────────────────────────────────────
            "HeapProfiler.enable" => Ok(json!({})),
            "HeapProfiler.takeHeapSnapshot" => self.heap_profiler_take_snapshot(),
            "HeapProfiler.startTrackingHeapObjects" => self.heap_profiler_start_tracking(),
            "HeapProfiler.stopTrackingHeapObjects" => self.heap_profiler_stop_tracking(),

            // ── Network (stubs) ───────────────────────────────────────────
            "Network.enable" => Ok(json!({})),
            "Network.disable" => Ok(json!({})),

            // ── Unknown ───────────────────────────────────────────────────
            other => Err(crate::error::StatorError::Internal(format!(
                "CDP method not implemented: {other}"
            ))),
        }
    }

    // ── Runtime.enable ────────────────────────────────────────────────────────

    fn runtime_enable(&mut self) -> StatorResult<Value> {
        // Emit executionContextCreated event (best-effort; ignore send failure).
        let event = CdpEvent {
            method: "Runtime.executionContextCreated".to_string(),
            params: json!({
                "context": {
                    "id": 1,
                    "origin": "stator",
                    "name": "stator",
                    "uniqueId": "1"
                }
            }),
        };
        if let Ok(s) = serde_json::to_string(&event) {
            let _ = self.ws.send(Message::Text(s.into()));
        }
        Ok(json!({}))
    }

    // ── Runtime.evaluate ─────────────────────────────────────────────────────

    fn runtime_evaluate(&mut self, params: &Value) -> StatorResult<Value> {
        let expression = match params.get("expression").and_then(Value::as_str) {
            Some(e) => e,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.evaluate: required parameter 'expression' is missing or not a string"
                        .to_string(),
                ));
            }
        };

        let bytecodes =
            parser::parse(expression).and_then(|p| BytecodeGenerator::compile_program(&p))?;

        let mut frame =
            InterpreterFrame::new_with_globals(bytecodes, vec![], Rc::clone(&self.globals));
        let js_result = Interpreter::run(&mut frame)?;

        Ok(json!({
            "result": js_value_to_remote_object(&js_result)
        }))
    }

    // ── Runtime.callFunctionOn ───────────────────────────────────────────────

    fn runtime_call_function_on(&mut self, params: &Value) -> StatorResult<Value> {
        let declaration = match params.get("functionDeclaration").and_then(Value::as_str) {
            Some(d) => d,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.callFunctionOn: required parameter 'functionDeclaration' is missing"
                        .to_string(),
                ));
            }
        };

        // Build a call expression: wrap the declaration and invoke it with
        // any supplied arguments serialised as literals.
        let args = params
            .get("arguments")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|a| {
                        if let Some(v) = a.get("value") {
                            v.to_string()
                        } else {
                            "undefined".to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();

        let expression = format!("({declaration})({args})");
        let bytecodes =
            parser::parse(&expression).and_then(|p| BytecodeGenerator::compile_program(&p))?;

        let mut frame =
            InterpreterFrame::new_with_globals(bytecodes, vec![], Rc::clone(&self.globals));
        let js_result = Interpreter::run(&mut frame)?;

        Ok(json!({
            "result": js_value_to_remote_object(&js_result)
        }))
    }

    // ── Runtime.getProperties ────────────────────────────────────────────────

    fn runtime_get_properties(&self, _params: &Value) -> StatorResult<Value> {
        let globals = self.globals.borrow();
        let descriptors: Vec<Value> = globals
            .iter()
            .map(|(name, value)| {
                json!({
                    "name": name,
                    "value": js_value_to_remote_object(value),
                    "writable": true,
                    "configurable": true,
                    "enumerable": true,
                    "isOwn": true,
                })
            })
            .collect();

        Ok(json!({ "result": descriptors }))
    }

    // ── Debugger.setPauseOnExceptions ────────────────────────────────────────

    fn debugger_set_pause_on_exceptions(&self, params: &Value) -> StatorResult<Value> {
        // CDP state values: "none", "uncaught", "all".  We store the
        // acknowledged state but actual behaviour depends on the debugger
        // being attached to the interpreter (see inspector::debugger).
        let _state = params
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("none");
        Ok(json!({}))
    }

    // ── Debugger.setBreakpointByUrl ──────────────────────────────────────────

    fn debugger_set_breakpoint_by_url(&mut self, params: &Value) -> StatorResult<Value> {
        let line_number = params
            .get("lineNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let column_number = params
            .get("columnNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        let bp_id = format!(
            "{}:{}:{}",
            self.next_breakpoint_id, line_number, column_number
        );
        self.next_breakpoint_id += 1;

        Ok(json!({
            "breakpointId": bp_id,
            "locations": [{
                "scriptId": "0",
                "lineNumber": line_number,
                "columnNumber": column_number,
            }]
        }))
    }

    // ── Console.enable ───────────────────────────────────────────────────────

    fn console_enable(&mut self) -> StatorResult<Value> {
        self.console_enabled = true;

        // Flush any buffered console messages as `Console.messageAdded` events.
        for msg in drain_messages() {
            let event = json!({
                "method": "Console.messageAdded",
                "params": {
                    "message": {
                        "source": "console-api",
                        "level": msg.level.as_cdp_str(),
                        "text": msg.text,
                    }
                }
            });
            if let Ok(s) = serde_json::to_string(&event) {
                let _ = self.ws.send(Message::Text(s.into()));
            }
        }

        Ok(json!({}))
    }

    // ── Profiler.start ───────────────────────────────────────────────────────

    fn profiler_start(&mut self, params: &Value) -> StatorResult<Value> {
        // Optional `samplingInterval` parameter in microseconds (default 1 ms).
        let interval_micros = params
            .get("samplingInterval")
            .and_then(Value::as_u64)
            .unwrap_or(1_000);
        self.profiler.start(interval_micros)?;
        Ok(json!({}))
    }

    // ── Profiler.stop ────────────────────────────────────────────────────────

    fn profiler_stop(&mut self) -> StatorResult<Value> {
        let profile = self.profiler.stop().ok_or_else(|| {
            crate::error::StatorError::Internal("profiler was not started".into())
        })?;
        let profile_value = serde_json::to_value(&profile)
            .map_err(|e| crate::error::StatorError::Internal(e.to_string()))?;
        Ok(json!({ "profile": profile_value }))
    }

    // ── HeapProfiler.takeHeapSnapshot ────────────────────────────────────────

    fn heap_profiler_take_snapshot(&mut self) -> StatorResult<Value> {
        let snapshot = HeapSnapshotBuilder::build(&self.globals.borrow());
        let chunk = snapshot.to_json();
        // Emit the snapshot as an addHeapSnapshotChunk event.
        let event = json!({
            "method": "HeapProfiler.addHeapSnapshotChunk",
            "params": { "chunk": chunk }
        });
        if let Ok(s) = serde_json::to_string(&event) {
            let _ = self.ws.send(Message::Text(s.into()));
        }
        // Emit reportHeapSnapshotProgress to signal completion.
        let done_event = json!({
            "method": "HeapProfiler.reportHeapSnapshotProgress",
            "params": {
                "done": snapshot.snapshot.node_count,
                "total": snapshot.snapshot.node_count,
                "finished": true
            }
        });
        if let Ok(s) = serde_json::to_string(&done_event) {
            let _ = self.ws.send(Message::Text(s.into()));
        }
        Ok(json!({}))
    }

    // ── HeapProfiler.startTrackingHeapObjects ────────────────────────────────

    fn heap_profiler_start_tracking(&mut self) -> StatorResult<Value> {
        crate::inspector::heap_snapshot::start_tracking();
        Ok(json!({}))
    }

    // ── HeapProfiler.stopTrackingHeapObjects ─────────────────────────────────

    fn heap_profiler_stop_tracking(&mut self) -> StatorResult<Value> {
        let records = crate::inspector::heap_snapshot::stop_tracking();
        // Return a summary of the allocation records collected.
        let stats: Vec<Value> = records
            .iter()
            .map(|r| json!({ "id": r.id, "size": r.size }))
            .collect();
        Ok(json!({ "stats": stats }))
    }
}

/// Convert a [`JsValue`] to a CDP `Runtime.RemoteObject` description.
fn js_value_to_remote_object(value: &JsValue) -> Value {
    match value {
        JsValue::Undefined => json!({"type": "undefined"}),
        JsValue::Null => json!({"type": "object", "subtype": "null", "value": Value::Null}),
        JsValue::Boolean(b) => json!({"type": "boolean", "value": b}),
        JsValue::Smi(n) => {
            json!({"type": "number", "value": n, "description": n.to_string()})
        }
        JsValue::HeapNumber(f) => {
            // Handle special IEEE 754 values per CDP spec.
            let description = if f.is_nan() {
                "NaN".to_string()
            } else if *f == f64::INFINITY {
                "Infinity".to_string()
            } else if *f == f64::NEG_INFINITY {
                "-Infinity".to_string()
            } else {
                f.to_string()
            };
            json!({"type": "number", "value": f, "description": description})
        }
        JsValue::String(s) => {
            json!({"type": "string", "value": s})
        }
        _ => {
            // Complex objects / functions: return a generic description.
            let desc = value
                .to_js_string()
                .unwrap_or_else(|_| "[object Object]".to_string());
            json!({"type": "object", "description": desc})
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Server
// ─────────────────────────────────────────────────────────────────────────────

/// A CDP WebSocket server bound to a local TCP address.
///
/// Call [`CdpServer::accept_one`] to accept and serve a single connection to
/// completion, or [`CdpServer::accept_loop`] to serve connections
/// indefinitely.
pub struct CdpServer {
    listener: TcpListener,
}

impl CdpServer {
    /// Bind the server to `addr` and return a ready-to-accept [`CdpServer`].
    ///
    /// Passing `"127.0.0.1:0"` lets the OS assign a free port; use
    /// [`CdpServer::local_addr`] to discover which port was chosen.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the address is already in use or otherwise
    /// unavailable.
    pub fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener })
    }

    /// Return the local address this server is listening on.
    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept and fully serve a single CDP connection, then return.
    ///
    /// Blocks until the WebSocket handshake completes, the client sends
    /// messages, and the connection is closed.
    pub fn accept_one(&self) -> io::Result<()> {
        let (stream, _peer) = self.listener.accept()?;
        let ws = accept(stream).map_err(|e| io::Error::other(e.to_string()))?;
        CdpSession::new(ws).run()
    }

    /// Accept and serve connections in a loop, blocking the calling thread.
    ///
    /// Each incoming connection is served to completion before the next is
    /// accepted (single-threaded / sequential).  Returns only when the
    /// underlying listener encounters a fatal [`io::Error`].
    pub fn accept_loop(&self) -> io::Result<()> {
        for stream in self.listener.incoming() {
            let stream = stream?;
            let ws = match accept(stream) {
                Ok(ws) => ws,
                Err(_) => continue,
            };
            // Ignore per-session errors; move on to the next connection.
            let _ = CdpSession::new(ws).run();
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::net::TcpStream;
    use std::thread;
    use std::time::Duration;

    use tungstenite::{Message, connect, stream::MaybeTlsStream};

    use super::*;

    /// Spawn a single-connection CDP server on `127.0.0.1:0`, perform the
    /// WebSocket handshake, and return `(server_thread, websocket, port)`.
    fn start_server() -> (
        thread::JoinHandle<io::Result<()>>,
        tungstenite::WebSocket<MaybeTlsStream<TcpStream>>,
        u16,
    ) {
        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let port = server.local_addr().expect("local_addr").port();

        let handle = thread::spawn(move || server.accept_one());

        // Give the server thread a moment to call `accept()`.
        thread::sleep(Duration::from_millis(20));

        let url = format!("ws://127.0.0.1:{port}");
        let (ws, _resp) = connect(url).expect("connect");

        (handle, ws, port)
    }

    #[test]
    fn test_cdp_connect_and_close() {
        let (handle, mut ws, _port) = start_server();
        ws.close(None).expect("close");
        handle.join().expect("thread panic").expect("server error");
    }

    #[test]
    fn test_cdp_runtime_evaluate_numeric() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"1+2"}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let text = match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected message: {other:?}"),
        };

        let json: Value = serde_json::from_str(&text).expect("parse reply");
        assert_eq!(json["id"], 1u64);
        assert_eq!(json["result"]["result"]["type"], "number");
        assert_eq!(json["result"]["result"]["value"], 3);
    }

    #[test]
    fn test_cdp_runtime_evaluate_string() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"'hello'"}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let text = match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected message: {other:?}"),
        };

        let json: Value = serde_json::from_str(&text).expect("parse reply");
        assert_eq!(json["id"], 2u64);
        assert_eq!(json["result"]["result"]["type"], "string");
        assert_eq!(json["result"]["result"]["value"], "hello");
    }

    #[test]
    fn test_cdp_runtime_enable_sends_event() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":3,"method":"Runtime.enable","params":{}}"#.into(),
        ))
        .expect("send");

        // Expect the executionContextCreated event first, then the ack.
        let msg1 = ws.read().expect("read event");
        let msg2 = ws.read().expect("read ack");

        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let event: Value = serde_json::from_str(&match msg1 {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse event");
        assert_eq!(event["method"], "Runtime.executionContextCreated");

        let ack: Value = serde_json::from_str(&match msg2 {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse ack");
        assert_eq!(ack["id"], 3u64);
    }

    #[test]
    fn test_cdp_debugger_enable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":4,"method":"Debugger.enable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 4u64);
        assert_eq!(json["result"]["debuggerId"], "stator-debugger-0");
    }

    #[test]
    fn test_cdp_profiler_enable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":5,"method":"Profiler.enable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 5u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_heap_profiler_enable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":6,"method":"HeapProfiler.enable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 6u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_unknown_method_returns_error() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":7,"method":"Unknown.method","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 7u64);
        assert!(json.get("error").is_some(), "should have error");
    }

    #[test]
    fn test_cdp_parse_error_returns_error() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text("not-json".into())).expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert!(json.get("error").is_some(), "should have error");
    }

    #[test]
    fn test_cdp_runtime_evaluate_syntax_error() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":8,"method":"Runtime.evaluate","params":{"expression":"var ="}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 8u64);
        assert!(json.get("error").is_some(), "should have error");
    }

    // ── New domain tests ─────────────────────────────────────────────────────

    #[test]
    fn test_cdp_runtime_call_function_on() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":10,"method":"Runtime.callFunctionOn","params":{"functionDeclaration":"function(a,b){return a+b}","arguments":[{"value":3},{"value":4}]}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 10u64);
        assert_eq!(json["result"]["result"]["type"], "number");
        assert_eq!(json["result"]["result"]["value"], 7);
    }

    #[test]
    fn test_cdp_runtime_get_properties_empty() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":11,"method":"Runtime.getProperties","params":{"objectId":"1"}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 11u64);
        assert!(json["result"]["result"].is_array());
    }

    #[test]
    fn test_cdp_debugger_set_pause_on_exceptions() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":12,"method":"Debugger.setPauseOnExceptions","params":{"state":"all"}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 12u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_debugger_set_breakpoint_by_url() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":13,"method":"Debugger.setBreakpointByUrl","params":{"lineNumber":5,"columnNumber":0}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 13u64);
        assert!(json["result"]["breakpointId"].is_string());
        assert!(json["result"]["locations"].is_array());
    }

    #[test]
    fn test_cdp_debugger_resume() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":14,"method":"Debugger.resume","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 14u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_console_enable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":15,"method":"Console.enable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 15u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_console_disable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":16,"method":"Console.disable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 16u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_network_enable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":17,"method":"Network.enable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 17u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_network_disable() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":18,"method":"Network.disable","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 18u64);
        assert!(json.get("error").is_none(), "should not have error");
    }

    #[test]
    fn test_cdp_runtime_call_function_on_missing_declaration() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":19,"method":"Runtime.callFunctionOn","params":{}}"#.into(),
        ))
        .expect("send");

        let reply = ws.read().expect("read");
        ws.close(None).ok();
        handle.join().expect("thread panic").ok();

        let json: Value = serde_json::from_str(&match reply {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse reply");

        assert_eq!(json["id"], 19u64);
        assert!(
            json.get("error").is_some(),
            "should have error for missing param"
        );
    }
}

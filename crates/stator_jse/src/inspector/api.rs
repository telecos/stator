//! In-process inspector API for transport-agnostic CDP integration.
//!
//! [`InProcessInspector`] is the engine-side counterpart of the FFI
//! `StatorInspector` handle.  It owns one or more [`InProcessInspectorSession`]s,
//! each wrapping a transport-agnostic [`CdpDispatcher`].  Embedders push
//! JSON-RPC requests with [`InProcessInspectorSession::dispatch_json`] and
//! pull queued responses/events with [`InProcessInspectorSession::take_next`].
//!
//! # Threading and lifetime
//!
//! Inspector and session methods are **single-threaded**, **synchronous**, and
//! **non-reentrant**.  All calls must be issued from the isolate's owning
//! thread.  The inspector borrows an external globals environment (shared
//! with the embedder's context) and must outlive any session it produces.
//!
//! # Script registry
//!
//! [`InProcessInspector::register_script`] assigns monotonically increasing
//! non-zero script IDs and emits a `Debugger.scriptParsed` event into every
//! session whose `Debugger` domain has been enabled.

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::json;

use crate::inspector::cdp::{CdpDispatcher, DispatchOutcome};
use crate::interpreter::GlobalEnv;

/// One CDP session within an [`InProcessInspector`].
///
/// Each session owns an independent [`CdpDispatcher`] and a cache buffer
/// used by the FFI layer to return engine-owned bytes whose lifetime is
/// bounded by the next call on the same session.
pub struct InProcessInspectorSession {
    /// Embedder-supplied session identifier.  Echoed back via
    /// [`InProcessInspectorSession::id`] but otherwise opaque to the
    /// dispatcher.
    id: u32,
    dispatcher: CdpDispatcher,
    /// Buffer holding the bytes most recently returned by
    /// [`InProcessInspectorSession::take_next_bytes`].  Stored here so that
    /// the FFI layer can return a pointer into engine memory that remains
    /// valid until the next inspector call on this session.
    cached: Option<Vec<u8>>,
}

impl InProcessInspectorSession {
    /// Build a session that shares `globals` with its parent context.
    fn new(id: u32, globals: Rc<RefCell<GlobalEnv>>) -> Self {
        Self {
            id,
            dispatcher: CdpDispatcher::with_globals(globals),
            cached: None,
        }
    }

    /// The embedder-supplied session identifier.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Forward a JSON-RPC request to the dispatcher.  Returns the dispatch
    /// outcome so that callers can distinguish transport-level parse errors
    /// from in-protocol errors (which are written to the outbox).
    pub fn dispatch_json(&mut self, text: &str) -> DispatchOutcome {
        // Invalidate the cached message buffer: per the FFI contract, any
        // pointer returned by `take_next_bytes` is valid only until the
        // next inspector call on this session.
        self.cached = None;
        self.dispatcher.dispatch_json(text)
    }

    /// Number of messages currently waiting in the dispatcher's outbox.
    pub fn pending_count(&self) -> usize {
        self.dispatcher.pending_count()
    }

    /// Pop the oldest outbox message and cache its UTF-8 bytes on the
    /// session.  Returns a borrow of the cached buffer whose lifetime is
    /// bounded by the next call on this session.
    ///
    /// Returns `None` when the outbox is empty; the previously cached
    /// buffer is dropped before returning.
    pub fn take_next_bytes(&mut self) -> Option<&[u8]> {
        let next = self.dispatcher.take_next()?;
        self.cached = Some(next.into_bytes());
        self.cached.as_deref()
    }

    /// Returns `true` if this session's dispatcher has seen `Debugger.enable`.
    pub fn debugger_enabled(&self) -> bool {
        self.dispatcher.debugger_enabled()
    }

    /// Mutable access to the dispatcher.  Used by the parent inspector to
    /// emit producer events (e.g. `Debugger.scriptParsed`) into the
    /// session's outbox without re-entering the dispatcher.
    pub(crate) fn dispatcher_mut(&mut self) -> &mut CdpDispatcher {
        &mut self.dispatcher
    }
}

/// Source-only record of a script registered with the inspector.
///
/// Scripts are cached by script ID so that future slices can serve
/// `Debugger.getScriptSource` requests without holding a reference to the
/// original [`StatorScript`](crate::bytecode::bytecode_array::BytecodeArray)
/// handle.
#[derive(Debug, Clone)]
pub struct RegisteredScript {
    /// Monotonically increasing non-zero script identifier.
    pub id: u32,
    /// Cached UTF-8 source text.
    pub source: String,
}

/// Owning container for a set of CDP sessions sharing a single context.
///
/// `InProcessInspector` is the engine-side type behind the FFI
/// `StatorInspector` handle.  It owns its sessions by value and, on drop,
/// terminates them with the inspector.
pub struct InProcessInspector {
    /// Globals environment shared with the embedder's context.
    globals: Rc<RefCell<GlobalEnv>>,
    /// Active sessions, in insertion order.  Iteration is used by
    /// `register_script` for `Debugger.scriptParsed` fan-out.
    sessions: Vec<Box<InProcessInspectorSession>>,
    /// Cached script registry, keyed by monotonically increasing ID.
    scripts: Vec<RegisteredScript>,
    /// Next script ID to assign; always non-zero.
    next_script_id: u32,
}

impl InProcessInspector {
    /// Build a fresh inspector that shares `globals` with the embedder's
    /// context.
    pub fn new(globals: Rc<RefCell<GlobalEnv>>) -> Self {
        Self {
            globals,
            sessions: Vec::new(),
            scripts: Vec::new(),
            next_script_id: 1,
        }
    }

    /// Open a new session keyed by `session_id`.  The returned pointer is
    /// owned by the inspector and remains valid until either
    /// [`InProcessInspector::disconnect`] is called for it or the inspector
    /// itself is dropped.
    pub fn connect(&mut self, session_id: u32) -> &mut InProcessInspectorSession {
        let session = Box::new(InProcessInspectorSession::new(
            session_id,
            Rc::clone(&self.globals),
        ));
        self.sessions.push(session);
        // SAFETY: just pushed; the box is alive and uniquely owned.
        self.sessions.last_mut().expect("just pushed").as_mut()
    }

    /// Detach and drop the session whose handle matches `session`.  Other
    /// sessions are unaffected.  Returns `true` if a matching session was
    /// found.
    pub fn disconnect(&mut self, session: *const InProcessInspectorSession) -> bool {
        if let Some(idx) = self
            .sessions
            .iter()
            .position(|s| s.as_ref() as *const _ == session)
        {
            self.sessions.remove(idx);
            true
        } else {
            false
        }
    }

    /// Number of currently connected sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Lookup a session by its embedder-supplied identifier.  Used by
    /// tests; the FFI layer keys on raw pointers instead.
    pub fn session_by_id_mut(&mut self, id: u32) -> Option<&mut InProcessInspectorSession> {
        self.sessions
            .iter_mut()
            .find(|s| s.id() == id)
            .map(|s| s.as_mut())
    }

    /// Register `source` as the next script in the inspector's registry.
    ///
    /// Returns the freshly assigned, non-zero script ID.  Every session
    /// whose `Debugger` domain has been enabled receives a
    /// `Debugger.scriptParsed` event in its outbox.
    pub fn register_script(&mut self, source: String) -> u32 {
        let id = self.next_script_id;
        self.next_script_id = self.next_script_id.checked_add(1).unwrap_or(1);

        let line_count = source.lines().count().max(1) as u32;
        let last_line_columns = source
            .lines()
            .last()
            .map(|s| s.chars().count() as u32)
            .unwrap_or(0);

        self.scripts.push(RegisteredScript {
            id,
            source: source.clone(),
        });

        // Fan out a `Debugger.scriptParsed` event to every session that has
        // opted into the Debugger domain.  Sessions without `Debugger.enable`
        // receive no notification, matching V8 inspector semantics.
        let params = json!({
            "scriptId": id.to_string(),
            "url": "",
            "startLine": 0,
            "startColumn": 0,
            "endLine": line_count.saturating_sub(1),
            "endColumn": last_line_columns,
            "executionContextId": 1,
            "hash": "",
        });
        let event = json!({
            "method": "Debugger.scriptParsed",
            "params": params,
        });
        let serialised = event.to_string();
        for session in self.sessions.iter_mut() {
            if session.debugger_enabled() {
                session.dispatcher_mut().push_raw(serialised.clone());
            }
        }

        id
    }

    /// Returns a snapshot view of the registered scripts.  Used by tests.
    pub fn scripts(&self) -> &[RegisteredScript] {
        &self.scripts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn new_inspector() -> InProcessInspector {
        InProcessInspector::new(Rc::new(RefCell::new(GlobalEnv::new())))
    }

    fn drain_to_strings(session: &mut InProcessInspectorSession) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(bytes) = session.take_next_bytes() {
            out.push(std::str::from_utf8(bytes).unwrap().to_string());
        }
        out
    }

    #[test]
    fn runtime_enable_emits_context_created_then_ack() {
        let mut inspector = new_inspector();
        let session = inspector.connect(1);
        assert_eq!(
            session.dispatch_json(r#"{"id":7,"method":"Runtime.enable","params":{}}"#),
            DispatchOutcome::Ok
        );
        assert_eq!(session.pending_count(), 2);

        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 2);
        let event: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(event["method"], "Runtime.executionContextCreated");
        let ack: Value = serde_json::from_str(&msgs[1]).unwrap();
        assert_eq!(ack["id"], 7u64);
        assert!(ack.get("error").is_none(), "ack should not carry error");
    }

    #[test]
    fn runtime_evaluate_basic_response() {
        let mut inspector = new_inspector();
        let session = inspector.connect(2);
        let outcome = session
            .dispatch_json(r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"2+3"}}"#);
        assert_eq!(outcome, DispatchOutcome::Ok);
        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 1);
        let resp: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(resp["id"], 1u64);
        assert_eq!(resp["result"]["result"]["type"], "number");
        assert_eq!(resp["result"]["result"]["value"], 5);
    }

    #[test]
    fn unknown_method_returns_protocol_error_with_ok_outcome() {
        let mut inspector = new_inspector();
        let session = inspector.connect(3);
        let outcome = session.dispatch_json(r#"{"id":11,"method":"NoSuch.method","params":{}}"#);
        assert_eq!(outcome, DispatchOutcome::Ok);

        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 1);
        let resp: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(resp["id"], 11u64);
        assert!(resp.get("error").is_some(), "should carry error");
        assert!(
            resp["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("NoSuch.method")
        );
    }

    #[test]
    fn malformed_json_returns_parse_error_outcome_and_pushes_response() {
        let mut inspector = new_inspector();
        let session = inspector.connect(4);
        let outcome = session.dispatch_json("not-json");
        assert_eq!(outcome, DispatchOutcome::ParseError);

        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 1);
        let resp: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert!(resp.get("error").is_some(), "parse error response missing");
        assert_eq!(resp["error"]["code"], -32700);
    }

    #[test]
    fn take_next_bytes_invalidates_on_next_call() {
        let mut inspector = new_inspector();
        let session = inspector.connect(5);
        session.dispatch_json(r#"{"id":1,"method":"Runtime.enable","params":{}}"#);
        assert_eq!(session.pending_count(), 2);
        // First pop: cache the event.
        let _first = session.take_next_bytes().expect("event").to_vec();
        // Second pop: previous buffer is invalidated; cache the ack.
        let _second = session.take_next_bytes().expect("ack").to_vec();
        // Outbox now empty.
        assert!(session.take_next_bytes().is_none());
    }

    #[test]
    fn register_script_assigns_monotonic_ids_and_fans_out_to_debugger_enabled_sessions() {
        let mut inspector = new_inspector();

        // Connect two sessions.  Only the first enables `Debugger`.
        {
            let s1 = inspector.connect(10);
            assert_eq!(
                s1.dispatch_json(r#"{"id":1,"method":"Debugger.enable","params":{}}"#),
                DispatchOutcome::Ok
            );
            // Drain the Debugger.enable ack so the outbox starts empty
            // before register_script fires.
            drain_to_strings(s1);
        }
        {
            let _s2 = inspector.connect(11);
        }

        let id_a = inspector.register_script("var a = 1;\nvar b = 2;\n".to_string());
        let id_b = inspector.register_script("// second".to_string());
        assert_eq!(id_a, 1);
        assert_eq!(id_b, 2);
        assert_eq!(inspector.scripts().len(), 2);

        let s1 = inspector.session_by_id_mut(10).expect("s1");
        assert_eq!(s1.pending_count(), 2);
        let msgs = drain_to_strings(s1);
        let first: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(first["method"], "Debugger.scriptParsed");
        assert_eq!(first["params"]["scriptId"], "1");
        let second: Value = serde_json::from_str(&msgs[1]).unwrap();
        assert_eq!(second["params"]["scriptId"], "2");

        let s2 = inspector.session_by_id_mut(11).expect("s2");
        assert_eq!(
            s2.pending_count(),
            0,
            "session without Debugger.enable must not receive scriptParsed events"
        );
    }

    #[test]
    fn disconnect_drops_session() {
        let mut inspector = new_inspector();
        let s1_ptr: *const InProcessInspectorSession = inspector.connect(20);
        let _s2 = inspector.connect(21);
        assert_eq!(inspector.session_count(), 2);
        assert!(inspector.disconnect(s1_ptr));
        assert_eq!(inspector.session_count(), 1);
        assert!(
            !inspector.disconnect(s1_ptr),
            "double-disconnect is a no-op"
        );
    }
}

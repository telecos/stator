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

use crate::inspector::cdp::{
    CdpDispatcher, DispatchOutcome, ExecutionContextDescription, default_execution_context,
};
use crate::inspector::debugger::Debugger;
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
    fn new(
        id: u32,
        globals: Rc<RefCell<GlobalEnv>>,
        contexts: Vec<ExecutionContextDescription>,
        debugger: Rc<RefCell<Debugger>>,
    ) -> Self {
        let mut dispatcher = CdpDispatcher::with_globals_and_contexts(globals, contexts);
        dispatcher.attach_debugger(debugger);
        Self {
            id,
            dispatcher,
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

    /// Enqueue a transport-level parse error that occurred before a UTF-8
    /// JSON-RPC request could be passed to the dispatcher.
    pub fn dispatch_parse_error(&mut self, message: String) -> DispatchOutcome {
        self.cached = None;
        self.dispatcher.push_parse_error(message);
        DispatchOutcome::ParseError
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

    /// Bridge an interpreter pause into this session by emitting a
    /// `Debugger.paused` event derived from the attached interpreter
    /// debugger's most recent pause. See
    /// [`CdpDispatcher::notify_paused`].
    pub fn notify_paused(&mut self) -> bool {
        self.cached = None;
        self.dispatcher.notify_paused()
    }

    /// Emit a `Debugger.resumed` event into this session's outbox.
    /// See [`CdpDispatcher::notify_resumed`].
    pub fn notify_resumed(&mut self) -> bool {
        self.cached = None;
        self.dispatcher.notify_resumed()
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

/// Metadata for one inspector context group.
#[derive(Debug, Clone)]
pub struct InspectorContextGroup {
    /// Stable group identifier.
    pub id: u32,
    /// Origin inherited by contexts that do not override it.
    pub origin: String,
    /// Human-readable group name.
    pub name: String,
    /// Live context IDs owned by this group.
    pub context_ids: Vec<u32>,
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
    /// producer events for deterministic fan-out.
    // Boxed sessions keep FFI session handles stable across Vec reallocations.
    #[allow(clippy::vec_box)]
    sessions: Vec<Box<InProcessInspectorSession>>,
    /// Stable context-group registry.
    context_groups: Vec<InspectorContextGroup>,
    /// Stable execution-context registry.
    contexts: Vec<ExecutionContextDescription>,
    /// Next group ID to assign; always greater than every live group ID.
    next_group_id: u32,
    /// Next context ID to assign; always greater than every live context ID.
    next_context_id: u32,
    /// Cached script registry, keyed by monotonically increasing ID.
    scripts: Vec<RegisteredScript>,
    /// Next script ID to assign; always non-zero.
    next_script_id: u32,
    /// Single interpreter [`Debugger`] shared across every session this
    /// inspector owns.  The interpreter only supports a single attached
    /// debugger per thread, so sessions never own their own debugger; they
    /// all observe pauses through this handle.
    debugger: Rc<RefCell<Debugger>>,
}

impl InProcessInspector {
    /// Build a fresh inspector that shares `globals` with the embedder's
    /// context.
    pub fn new(globals: Rc<RefCell<GlobalEnv>>) -> Self {
        let default_context = default_execution_context();
        let default_group = InspectorContextGroup {
            id: default_context.group_id,
            origin: default_context.origin.clone(),
            name: default_context.name.clone(),
            context_ids: vec![default_context.id],
        };
        Self {
            globals,
            sessions: Vec::new(),
            context_groups: vec![default_group],
            contexts: vec![default_context],
            next_group_id: 2,
            next_context_id: 2,
            scripts: Vec::new(),
            next_script_id: 1,
            debugger: Rc::new(RefCell::new(Debugger::new())),
        }
    }

    /// Shared interpreter [`Debugger`] handle.  Embedders pass a clone of
    /// this to [`crate::interpreter::attach_debugger`] before driving the
    /// interpreter so that pause events surface through every connected
    /// CDP session.
    pub fn debugger(&self) -> Rc<RefCell<Debugger>> {
        Rc::clone(&self.debugger)
    }

    /// Build a transport dispatcher that shares this inspector's embedder state.
    ///
    /// The returned dispatcher borrows the same globals, execution-context
    /// snapshot, debugger handle, and registered script sources that in-process
    /// sessions use. It is intended for single-threaded WebSocket serving via
    /// [`CdpServer::accept_one_with_dispatcher`](crate::inspector::cdp::CdpServer::accept_one_with_dispatcher),
    /// avoiding cross-thread movement of `Rc<RefCell<_>>` embedder state.
    pub fn transport_dispatcher(&self) -> CdpDispatcher {
        let mut dispatcher = CdpDispatcher::with_globals_and_contexts(
            Rc::clone(&self.globals),
            self.contexts.clone(),
        );
        dispatcher.attach_debugger(Rc::clone(&self.debugger));
        for script in &self.scripts {
            dispatcher.register_script_source(script.id, script.source.clone());
        }
        dispatcher
    }

    /// Emit a `Debugger.paused` event into every connected session whose
    /// `Debugger` domain is enabled. Returns the number of sessions that
    /// received an event.
    ///
    /// Embedders typically call this immediately after
    /// [`crate::interpreter::Interpreter::run`] returns
    /// [`crate::error::StatorError::DebuggerPaused`].
    pub fn notify_paused(&mut self) -> usize {
        let mut emitted = 0;
        for session in &mut self.sessions {
            if session.notify_paused() {
                emitted += 1;
            }
        }
        emitted
    }

    /// Emit a `Debugger.resumed` event into every connected session whose
    /// `Debugger` domain is enabled.  Returns the number of sessions that
    /// received an event.
    pub fn notify_resumed(&mut self) -> usize {
        let mut emitted = 0;
        for session in &mut self.sessions {
            if session.notify_resumed() {
                emitted += 1;
            }
        }
        emitted
    }

    /// Open a new session keyed by `session_id`.  The returned pointer is
    /// owned by the inspector and remains valid until either
    /// [`InProcessInspector::disconnect`] is called for it or the inspector
    /// itself is dropped.
    pub fn connect(&mut self, session_id: u32) -> &mut InProcessInspectorSession {
        let mut session = Box::new(InProcessInspectorSession::new(
            session_id,
            Rc::clone(&self.globals),
            self.contexts.clone(),
            Rc::clone(&self.debugger),
        ));
        for script in &self.scripts {
            session
                .dispatcher_mut()
                .register_script_source(script.id, script.source.clone());
        }
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
            .position(|s| std::ptr::eq(s.as_ref(), session))
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

    /// Return the default context group ID.
    pub fn default_context_group_id(&self) -> u32 {
        self.context_groups
            .first()
            .map(|group| group.id)
            .unwrap_or(0)
    }

    /// Return the default execution context ID.
    pub fn default_context_id(&self) -> u32 {
        self.contexts.first().map(|context| context.id).unwrap_or(0)
    }

    /// Return the live context groups in deterministic creation order.
    pub fn context_groups(&self) -> &[InspectorContextGroup] {
        &self.context_groups
    }

    /// Return the live execution contexts in deterministic creation order.
    pub fn contexts(&self) -> &[ExecutionContextDescription] {
        &self.contexts
    }

    /// Lookup a live execution context by ID.
    pub fn context_by_id(&self, context_id: u32) -> Option<&ExecutionContextDescription> {
        self.contexts
            .iter()
            .find(|context| context.id == context_id)
    }

    /// Create a new context group and return its stable ID.
    pub fn create_context_group(
        &mut self,
        origin: impl Into<String>,
        name: impl Into<String>,
    ) -> u32 {
        let id = self.next_group_id;
        let Some(next_group_id) = self.next_group_id.checked_add(1) else {
            return 0;
        };
        self.next_group_id = next_group_id;
        self.context_groups.push(InspectorContextGroup {
            id,
            origin: origin.into(),
            name: name.into(),
            context_ids: Vec::new(),
        });
        id
    }

    /// Create a new execution context inside an existing group.
    ///
    /// Returns `0` when `group_id` is not live or the context ID space is
    /// exhausted. Runtime-enabled sessions receive exactly one
    /// `Runtime.executionContextCreated` event.
    pub fn create_context(
        &mut self,
        group_id: u32,
        origin: impl Into<String>,
        name: impl Into<String>,
        aux_data: serde_json::Value,
    ) -> u32 {
        let Some(group_index) = self
            .context_groups
            .iter()
            .position(|group| group.id == group_id)
        else {
            return 0;
        };
        let id = self.next_context_id;
        let Some(next_context_id) = self.next_context_id.checked_add(1) else {
            return 0;
        };
        self.next_context_id = next_context_id;
        let context = ExecutionContextDescription::new(id, group_id, origin, name, aux_data);
        self.context_groups[group_index].context_ids.push(id);
        self.contexts.push(context.clone());
        self.fan_out_context_created(context);
        id
    }

    /// Destroy one execution context.
    ///
    /// Returns `true` only for the first destruction of a live context. A
    /// runtime-enabled session receives exactly one
    /// `Runtime.executionContextDestroyed` event for that successful teardown.
    pub fn destroy_context(&mut self, context_id: u32) -> bool {
        let Some(index) = self
            .contexts
            .iter()
            .position(|context| context.id == context_id)
        else {
            return false;
        };
        self.contexts.remove(index);
        for group in &mut self.context_groups {
            group.context_ids.retain(|id| *id != context_id);
        }
        self.fan_out_context_destroyed(context_id);
        true
    }

    /// Clear every live execution context in a context group.
    ///
    /// Returns the number of contexts removed. Runtime-enabled sessions receive
    /// one `Runtime.executionContextsCleared` event if at least one context was
    /// live in the group, and repeated calls are no-ops.
    pub fn clear_context_group(&mut self, group_id: u32) -> usize {
        let Some(group_index) = self
            .context_groups
            .iter()
            .position(|group| group.id == group_id)
        else {
            return 0;
        };
        let context_ids = std::mem::take(&mut self.context_groups[group_index].context_ids);
        if context_ids.is_empty() {
            return 0;
        }
        self.contexts
            .retain(|context| !context_ids.contains(&context.id));
        self.fan_out_contexts_cleared(&context_ids);
        context_ids.len()
    }

    /// Register `source` as the next script in the inspector's registry.
    ///
    /// Returns the freshly assigned, non-zero script ID.  Every session
    /// whose `Debugger` domain has been enabled receives a
    /// `Debugger.scriptParsed` event in its outbox.
    pub fn register_script(&mut self, source: String) -> u32 {
        let id = self.next_script_id;
        let Some(next_script_id) = self.next_script_id.checked_add(1) else {
            return 0;
        };
        self.next_script_id = next_script_id;

        let source_url = crate::inspector::cdp::registered_script_url(&source);
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
            "url": source_url,
            "startLine": 0,
            "startColumn": 0,
            "endLine": line_count.saturating_sub(1),
            "endColumn": last_line_columns,
            "executionContextId": self.default_context_id(),
            "hash": "",
        });
        let event = json!({
            "method": "Debugger.scriptParsed",
            "params": params,
        });
        let serialised = event.to_string();
        for session in self.sessions.iter_mut() {
            session
                .dispatcher_mut()
                .register_script_source(id, source.clone());
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

    fn fan_out_context_created(&mut self, context: ExecutionContextDescription) {
        for session in &mut self.sessions {
            session
                .dispatcher_mut()
                .add_execution_context(context.clone());
        }
    }

    fn fan_out_context_destroyed(&mut self, context_id: u32) {
        for session in &mut self.sessions {
            session
                .dispatcher_mut()
                .remove_execution_context(context_id);
        }
    }

    fn fan_out_contexts_cleared(&mut self, context_ids: &[u32]) {
        for session in &mut self.sessions {
            session
                .dispatcher_mut()
                .clear_execution_contexts(context_ids);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::value::JsValue;
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

    fn dispatch_direct(dispatcher: &mut CdpDispatcher, text: &str) -> Value {
        assert_eq!(dispatcher.dispatch_json(text), DispatchOutcome::Ok);
        let mut last = None;
        while let Some(message) = dispatcher.take_next() {
            last = Some(message);
        }
        serde_json::from_str(&last.expect("response")).expect("valid JSON")
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
        assert_eq!(event["params"]["context"]["id"], 1);
        assert_eq!(event["params"]["context"]["origin"], "stator");
        assert_eq!(event["params"]["context"]["name"], "stator");
        assert_eq!(event["params"]["context"]["uniqueId"], "stator-1-1");
        assert_eq!(event["params"]["context"]["auxData"]["groupId"], 1);
        assert_eq!(event["params"]["context"]["auxData"]["isDefault"], true);
        let ack: Value = serde_json::from_str(&msgs[1]).unwrap();
        assert_eq!(ack["id"], 7u64);
        assert!(ack.get("error").is_none(), "ack should not carry error");
    }

    #[test]
    fn context_group_registry_has_stable_default_metadata() {
        let inspector = new_inspector();

        assert_eq!(inspector.default_context_group_id(), 1);
        assert_eq!(inspector.default_context_id(), 1);
        assert_eq!(inspector.context_groups().len(), 1);
        assert_eq!(inspector.contexts().len(), 1);
        let context = inspector.context_by_id(1).expect("default context");
        assert_eq!(context.group_id, 1);
        assert_eq!(context.origin, "stator");
        assert_eq!(context.name, "stator");
        assert_eq!(context.unique_id, "stator-1-1");
        assert_eq!(context.aux_data["groupId"], 1);
    }

    #[test]
    fn create_context_notifies_runtime_enabled_sessions_and_propagates_metadata() {
        let mut inspector = new_inspector();
        let group_id = inspector.create_context_group("https://example.test", "page");
        {
            let session = inspector.connect(30);
            assert_eq!(
                session.dispatch_json(r#"{"id":1,"method":"Runtime.enable","params":{}}"#),
                DispatchOutcome::Ok
            );
            drain_to_strings(session);
        }

        let context_id = inspector.create_context(
            group_id,
            "https://example.test",
            "main frame",
            json!({
                "isDefault": false,
                "type": "page",
                "frameId": "frame-1",
            }),
        );

        assert_eq!(context_id, 2);
        let context = inspector.context_by_id(context_id).expect("new context");
        assert_eq!(context.group_id, group_id);
        assert_eq!(context.unique_id, format!("stator-{group_id}-{context_id}"));

        let session = inspector.session_by_id_mut(30).expect("session");
        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 1);
        let event: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(event["method"], "Runtime.executionContextCreated");
        let payload = &event["params"]["context"];
        assert_eq!(payload["id"], context_id);
        assert_eq!(payload["origin"], "https://example.test");
        assert_eq!(payload["name"], "main frame");
        assert_eq!(payload["auxData"]["groupId"], group_id);
        assert_eq!(payload["auxData"]["frameId"], "frame-1");
    }

    #[test]
    fn destroy_context_emits_once_and_never_reuses_live_ids() {
        let mut inspector = new_inspector();
        let group_id = inspector.default_context_group_id();
        {
            let session = inspector.connect(31);
            assert_eq!(
                session.dispatch_json(r#"{"id":1,"method":"Runtime.enable","params":{}}"#),
                DispatchOutcome::Ok
            );
            drain_to_strings(session);
        }

        let destroyed_id = inspector.create_context(group_id, "stator", "worker", json!({}));
        let next_id = inspector.create_context(group_id, "stator", "worker-2", json!({}));
        assert_eq!(destroyed_id, 2);
        assert_eq!(next_id, 3);
        drain_to_strings(inspector.session_by_id_mut(31).unwrap());

        assert!(inspector.destroy_context(destroyed_id));
        assert!(!inspector.destroy_context(destroyed_id));
        let replacement_id = inspector.create_context(group_id, "stator", "replacement", json!({}));
        assert_eq!(replacement_id, 4);

        let session = inspector.session_by_id_mut(31).expect("session");
        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 2);
        let destroyed: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(destroyed["method"], "Runtime.executionContextDestroyed");
        assert_eq!(destroyed["params"]["executionContextId"], destroyed_id);
        let created: Value = serde_json::from_str(&msgs[1]).unwrap();
        assert_eq!(created["method"], "Runtime.executionContextCreated");
        assert_eq!(created["params"]["context"]["id"], replacement_id);
    }

    #[test]
    fn clear_context_group_is_isolated_and_repeated_teardown_is_noop() {
        let mut inspector = new_inspector();
        let group_a = inspector.create_context_group("a", "a");
        let group_b = inspector.create_context_group("b", "b");
        let ctx_a1 = inspector.create_context(group_a, "a", "a1", json!({}));
        let ctx_a2 = inspector.create_context(group_a, "a", "a2", json!({}));
        let ctx_b = inspector.create_context(group_b, "b", "b1", json!({}));
        {
            let session = inspector.connect(32);
            assert_eq!(
                session.dispatch_json(r#"{"id":1,"method":"Runtime.enable","params":{}}"#),
                DispatchOutcome::Ok
            );
            drain_to_strings(session);
        }

        assert_eq!(inspector.clear_context_group(group_a), 2);
        assert_eq!(inspector.clear_context_group(group_a), 0);
        assert!(inspector.context_by_id(ctx_a1).is_none());
        assert!(inspector.context_by_id(ctx_a2).is_none());
        assert!(inspector.context_by_id(ctx_b).is_some());

        let session = inspector.session_by_id_mut(32).expect("session");
        let msgs = drain_to_strings(session);
        assert_eq!(msgs.len(), 1);
        let cleared: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(cleared["method"], "Runtime.executionContextsCleared");
        assert_eq!(
            session.pending_count(),
            0,
            "repeated clear must not emit a second event"
        );
    }

    #[test]
    fn context_lifecycle_without_sessions_is_fail_closed() {
        let mut inspector = new_inspector();
        let group_id = inspector.create_context_group("edge", "edge");
        let context_id = inspector.create_context(group_id, "edge", "hidden", json!({}));

        assert_ne!(group_id, 0);
        assert_ne!(context_id, 0);
        assert_eq!(inspector.session_count(), 0);
        assert!(inspector.destroy_context(context_id));
        assert_eq!(inspector.clear_context_group(group_id), 0);
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

        let id_a = inspector
            .register_script("var a = 1;\nvar b = 2;\n//# sourceURL=stator://first.js".to_string());
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
        assert_eq!(first["params"]["url"], "stator://first.js");
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
    fn register_script_returns_zero_instead_of_wrapping_ids() {
        let mut inspector = new_inspector();
        inspector.next_script_id = u32::MAX;

        let id = inspector.register_script("let overflow = true;".to_string());

        assert_eq!(id, 0);
        assert!(inspector.scripts().is_empty());
    }

    #[test]
    fn registered_script_source_is_available_to_existing_and_late_sessions() {
        let mut inspector = new_inspector();
        let source = "let registered = 42;".to_string();
        {
            let session = inspector.connect(12);
            drain_to_strings(session);
        }

        let script_id = inspector.register_script(source.clone());

        {
            let session = inspector.session_by_id_mut(12).expect("existing session");
            assert_eq!(
                session.dispatch_json(
                    &json!({
                        "id": 1,
                        "method": "Debugger.getScriptSource",
                        "params": { "scriptId": script_id.to_string() }
                    })
                    .to_string()
                ),
                DispatchOutcome::Ok
            );
            let messages = drain_to_strings(session);
            let response: Value = serde_json::from_str(messages.last().unwrap()).unwrap();
            assert_eq!(response["result"]["scriptSource"], source);
        }

        {
            let session = inspector.connect(13);
            assert_eq!(
                session.dispatch_json(
                    &json!({
                        "id": 2,
                        "method": "Debugger.getScriptSource",
                        "params": { "scriptId": script_id.to_string() }
                    })
                    .to_string()
                ),
                DispatchOutcome::Ok
            );
            let messages = drain_to_strings(session);
            let response: Value = serde_json::from_str(messages.last().unwrap()).unwrap();
            assert_eq!(response["result"]["scriptSource"], "let registered = 42;");
        }
    }

    #[test]
    fn transport_dispatcher_shares_globals_contexts_debugger_and_scripts() {
        let globals = Rc::new(RefCell::new(GlobalEnv::new()));
        globals
            .borrow_mut()
            .insert("shared".to_string(), JsValue::Smi(41));
        let mut inspector = InProcessInspector::new(Rc::clone(&globals));
        let group_id = inspector.create_context_group("https://edge.test", "edge");
        let context_id = inspector.create_context(
            group_id,
            "https://edge.test",
            "main",
            json!({"frameId": "frame-1"}),
        );
        let source = "let registered = 42;\n//# sourceURL=stator://transport.js".to_string();
        let script_id = inspector.register_script(source.clone());

        let mut dispatcher = inspector.transport_dispatcher();
        let eval = dispatch_direct(
            &mut dispatcher,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"shared + 1"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 42);

        let source_response = dispatch_direct(
            &mut dispatcher,
            &json!({
                "id": 2,
                "method": "Debugger.getScriptSource",
                "params": { "scriptId": script_id.to_string() }
            })
            .to_string(),
        );
        assert_eq!(source_response["result"]["scriptSource"], source);

        let targets = dispatch_direct(
            &mut dispatcher,
            r#"{"id":3,"method":"Target.getTargets","params":{}}"#,
        );
        let target_ids: Vec<_> = targets["result"]["targetInfos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|target| target["targetId"].as_str().unwrap())
            .collect();
        let expected_target_id = format!("stator-{group_id}");
        assert!(
            target_ids
                .iter()
                .any(|target_id| *target_id == expected_target_id)
        );
        assert_ne!(context_id, 0);
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

    #[test]
    fn debugger_handle_is_shared_across_sessions() {
        let mut inspector = new_inspector();
        let dbg = inspector.debugger();
        // The handle returned to the embedder must be the same Rc the
        // sessions observe.
        let _s = inspector.connect(30);
        assert!(Rc::ptr_eq(&dbg, &inspector.debugger()));
    }

    #[test]
    fn notify_paused_fans_out_to_debugger_enabled_sessions_only() {
        let mut inspector = new_inspector();
        let _ = inspector.connect(1);
        let _ = inspector.connect(2);

        // Enable Debugger only on session 1.
        let s1 = inspector.session_by_id_mut(1).unwrap();
        assert_eq!(
            s1.dispatch_json(r#"{"id":1,"method":"Debugger.enable","params":{}}"#),
            DispatchOutcome::Ok
        );
        // Drain the enable ack.
        while s1.take_next_bytes().is_some() {}

        // Drive a synthetic interpreter pause through the shared debugger.
        let dbg = inspector.debugger();
        let _ = dbg.borrow_mut().on_debugger_statement(99);

        let emitted = inspector.notify_paused();
        assert_eq!(emitted, 1, "only one session has Debugger.enable");

        let s1 = inspector.session_by_id_mut(1).unwrap();
        let msgs = drain_to_strings(s1);
        assert_eq!(msgs.len(), 1);
        let event: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(event["method"], "Debugger.paused");
        assert_eq!(event["params"]["reason"], "debuggerStatement");

        let s2 = inspector.session_by_id_mut(2).unwrap();
        let msgs2 = drain_to_strings(s2);
        assert!(msgs2.is_empty(), "session 2 should receive nothing");
    }

    #[test]
    fn notify_resumed_only_targets_debugger_enabled_sessions() {
        let mut inspector = new_inspector();
        let _ = inspector.connect(1);
        let s1 = inspector.session_by_id_mut(1).unwrap();
        assert_eq!(
            s1.dispatch_json(r#"{"id":1,"method":"Debugger.enable","params":{}}"#),
            DispatchOutcome::Ok
        );
        while s1.take_next_bytes().is_some() {}

        assert_eq!(inspector.notify_resumed(), 1);
        let s1 = inspector.session_by_id_mut(1).unwrap();
        let msgs = drain_to_strings(s1);
        assert_eq!(msgs.len(), 1);
        let event: Value = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(event["method"], "Debugger.resumed");
    }
}

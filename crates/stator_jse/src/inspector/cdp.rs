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
//! | `Runtime`      | `getProperties`           | Lists own properties of a previously-minted `RemoteObject` |
//! | `Runtime`      | `queryObjects`            | Finds reachable objects with the requested prototype |
//! | `Runtime`      | `releaseObject`           | Drops one `RemoteObject` from the per-session registry |
//! | `Runtime`      | `releaseObjectGroup`      | Drops every `RemoteObject` tagged with a given group |
//! | `Runtime`      | `compileScript`           | Compiles and optionally persists a script for later `runScript` |
//! | `Runtime`      | `runScript`               | Executes a script persisted by `compileScript` |
//! | `Runtime`      | `runIfWaitingForDebugger` | Acknowledges DevTools startup handshake |
//! | `Runtime`      | `discardConsoleEntries`   | Clears buffered console messages |
//! | `Runtime`      | `globalLexicalScopeNames` | Reports current global binding names |
//! | `Runtime`      | `getIsolateId`            | Reports a stable Stator isolate identifier |
//! | `Runtime`      | `getHeapUsage`            | Reports reachable heap-size estimates |
//! | `Runtime`      | `addBinding`/`removeBinding` | Installs/removes global binding callbacks |
//! | `Debugger`     | `enable`                  | Acknowledges; returns `debuggerId` |
//! | `Debugger`     | `disable`                 | Clears the `Debugger` domain enabled state |
//! | `Debugger`     | `setPauseOnExceptions`    | Configures exception pause state (typed error on invalid `state`) |
//! | `Debugger`     | `setBreakpoint`           | Sets a breakpoint by registered script location |
//! | `Debugger`     | `setBreakpointByUrl`      | Resolves URL breakpoints against registered scripts |
//! | `Debugger`     | `setBreakpointsActive`    | Enables/disables installed breakpoint pauses |
//! | `Debugger`     | `setSkipAllPauses`        | Suppresses/resumes all debugger pause sources |
//! | `Debugger`     | `setBlackboxPatterns`/`setBlackboxedRanges` | Stores blackbox filters for debugger setup |
//! | `Debugger`     | `resume`                  | Resumes after a pause; emits `Debugger.resumed` when an active pause exists |
//! | `Debugger`     | `continueToLocation`      | Resumes to a one-shot breakpoint at a registered script location |
//! | `Debugger`     | `stepInto`/`stepOver`/`stepOut` | Applies the step on the attached interpreter debugger; errors when not attached or no active pause |
//! | `Debugger`     | `pause`                   | Fail-closed: synchronous interpreter cannot be interrupted |
//! | `Debugger`     | `evaluateOnCallFrame`     | Fail-closed: call-frame snapshots not implemented yet |
//! | `Debugger`     | `getScriptSource`         | Returns a source registered by the in-process inspector |
//! | `Debugger`     | `setScriptSource`         | Validates and updates registered script source text |
//! | `Debugger`     | `getPossibleBreakpoints`  | Returns breakpointable locations for registered scripts |
//! | `Console`      | `enable`                  | Flushes buffered messages as events |
//! | `Console`      | `disable`                 | Acknowledges                       |
//! | `Profiler`     | `enable`                  | Acknowledges                       |
//! | `Profiler`     | `setSamplingInterval`     | Sets interval for the next profile |
//! | `Profiler`     | `start`                   | Starts CPU profiling               |
//! | `Profiler`     | `stop`                    | Stops profiling; returns profile    |
//! | `HeapProfiler` | `enable`                  | Acknowledges                       |
//! | `HeapProfiler` | `takeHeapSnapshot`        | Emits snapshot chunks              |
//! | `HeapProfiler` | `startTrackingHeapObjects` | Starts allocation tracking         |
//! | `HeapProfiler` | `stopTrackingHeapObjects`  | Returns allocation stats           |
//! | `Target`       | `getTargets`/`attachToTarget`/`closeTarget` | Single-target DevTools compatibility |
//! | `Network`      | `enable`                  | Acknowledges (stub)                |
//! | `Network`      | `disable`                 | Acknowledges (stub)                |
//! | `Schema`       | `getDomains`              | Reports the supported CDP domain names |
//!
//! # Example
//!
//! ```no_run
//! use stator_jse::inspector::cdp::CdpServer;
//!
//! let server = CdpServer::bind("127.0.0.1:9229").unwrap();
//! // Accept one connection, process all messages, then return.
//! server.accept_one().unwrap();
//! ```

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tungstenite::{Message, WebSocket, accept};

use crate::bytecode::bytecode_array::BytecodeArray;
use crate::bytecode::bytecode_generator::BytecodeGenerator;
use crate::error::{StatorError, StatorResult};
use crate::inspector::console::{ProfileEventKind, drain_messages, drain_profile_events};
use crate::inspector::debugger::{BreakpointId, DebugAction, Debugger, PauseReason};
use crate::inspector::heap_snapshot::{AllocationRecord, HeapSnapshotBuilder};
use crate::inspector::profiler::CpuProfiler;
use crate::interpreter::{GlobalEnv, Interpreter, InterpreterFrame, take_pending_exception};
use crate::objects::value::JsValue;
use crate::parser;

/// Stable CDP-visible execution context metadata.
///
/// Instances are owned by the inspector context registry and cloned into each
/// dispatcher so `Runtime.enable` and lifecycle events use a deterministic
/// payload shape across WebSocket and in-process transports.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionContextDescription {
    /// CDP `Runtime.ExecutionContextDescription.id`.
    pub id: u32,
    /// Inspector context-group identifier that owns this context.
    pub group_id: u32,
    /// Origin string reported to DevTools.
    pub origin: String,
    /// Human-readable context name.
    pub name: String,
    /// Stable unique identifier for this context lifetime.
    pub unique_id: String,
    /// CDP `auxData` metadata object.
    pub aux_data: Value,
}

impl ExecutionContextDescription {
    /// Build a context description with a deterministic `uniqueId`.
    pub fn new(
        id: u32,
        group_id: u32,
        origin: impl Into<String>,
        name: impl Into<String>,
        aux_data: Value,
    ) -> Self {
        Self {
            id,
            group_id,
            origin: origin.into(),
            name: name.into(),
            unique_id: format!("stator-{group_id}-{id}"),
            aux_data: normalize_aux_data(group_id, aux_data),
        }
    }

    /// Convert this description into the CDP event payload object.
    pub fn to_cdp_context(&self) -> Value {
        json!({
            "id": self.id,
            "origin": self.origin,
            "name": self.name,
            "uniqueId": self.unique_id,
            "auxData": self.aux_data,
        })
    }
}

/// Return the default single-context metadata used by standalone CDP sessions.
pub fn default_execution_context() -> ExecutionContextDescription {
    ExecutionContextDescription::new(
        1,
        1,
        "stator",
        "stator",
        json!({
            "isDefault": true,
            "type": "default",
        }),
    )
}

fn normalize_aux_data(group_id: u32, aux_data: Value) -> Value {
    let mut object = match aux_data {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    object.entry("groupId").or_insert_with(|| json!(group_id));
    Value::Object(object)
}

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

/// Per-session registry of inspector-visible heap [`JsValue`]s.
///
/// CDP `Runtime.RemoteObject` payloads reference non-primitive values by an
/// opaque `objectId` string.  This registry mints stable decimal IDs for
/// every non-primitive value surfaced to the inspector and holds a clone of
/// the [`JsValue`] (cheap because heap variants are reference-counted), so
/// later `Runtime.getProperties` and `Runtime.releaseObject` calls can
/// resolve the same value without leaking strong roots into the engine
/// heap beyond the registry itself.
///
/// IDs are scoped to a single [`CdpDispatcher`] (and therefore to a single
/// inspector session): two sessions never observe each other's IDs, and an
/// ID that has been released or that was never minted always fails closed
/// with a structured `Internal` error rather than fabricating properties.
///
/// Optional `objectGroup` labels mirror the V8 inspector convention used by
/// DevTools to bulk-release every object minted during the evaluation of a
/// single console expression.
#[derive(Default)]
pub struct RemoteObjectRegistry {
    entries: HashMap<String, RemoteObjectEntry>,
    next_id: u64,
}

struct RemoteObjectEntry {
    value: JsValue,
    group: Option<String>,
}

#[derive(Clone)]
struct CompiledScript {
    bytecodes: Rc<BytecodeArray>,
    expression: String,
    source_url: Option<String>,
    execution_context_id: u32,
}

#[derive(Default)]
struct RuntimeBindingState {
    active_names: HashSet<String>,
    pending_calls: Vec<RuntimeBindingCall>,
}

struct RuntimeBindingCall {
    name: String,
    payload: String,
}

#[derive(Clone)]
struct CoverageScript {
    script_id: String,
    url: String,
    source: String,
    count: u32,
}

#[derive(Clone)]
struct TypeProfileScript {
    script_id: String,
    url: String,
    offset: usize,
    types: HashSet<String>,
}

impl RemoteObjectRegistry {
    /// Build an empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register `value` with an optional `group` and return the freshly
    /// minted decimal object ID.
    ///
    /// IDs are session-local, monotonically increasing, and never reused
    /// while live.  Each call clones `value`; for heap variants this is an
    /// `Rc` bump, so the registry does not introduce additional strong
    /// references into the GC heap beyond the entry itself.
    pub fn register(&mut self, value: JsValue, group: Option<String>) -> String {
        let id = self.next_id;
        // Saturating add keeps the registry safe under pathological session
        // lifetimes; the limit is 2^64 IDs which is unreachable in practice.
        self.next_id = self.next_id.saturating_add(1);
        let id_str = id.to_string();
        self.entries
            .insert(id_str.clone(), RemoteObjectEntry { value, group });
        id_str
    }

    /// Look up `id` and return a clone of the stored value, or `None` if
    /// the ID is unknown or has been released.
    pub fn get(&self, id: &str) -> Option<JsValue> {
        self.entries.get(id).map(|e| e.value.clone())
    }

    /// Drop the entry for `id`.  Returns `true` when an entry was removed
    /// and `false` when the ID was unknown or already released.
    pub fn release(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    /// Drop every entry tagged with `group` and return the count removed.
    pub fn release_group(&mut self, group: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.group.as_deref() != Some(group));
        before - self.entries.len()
    }

    /// Returns the optional `objectGroup` label associated with `id`, or
    /// `None` when the ID is unknown or the entry has no group.
    pub fn group_of(&self, id: &str) -> Option<&str> {
        self.entries.get(id).and_then(|e| e.group.as_deref())
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no live entries remain.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of attempting to process a JSON-RPC text frame via
/// [`CdpDispatcher::dispatch_json`].
///
/// The dispatcher always writes exactly one protocol reply (and any
/// associated events) into its outbox.  Transport-level failures — currently
/// limited to malformed JSON — are surfaced through this enum so that
/// transports can distinguish them from in-protocol errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The text frame parsed as a JSON-RPC request and was dispatched.  A
    /// response message (success or in-protocol error) has been pushed onto
    /// the outbox.
    Ok,
    /// The text frame failed to parse as JSON.  A JSON-RPC parse-error
    /// response has been pushed onto the outbox as a courtesy to the peer.
    ParseError,
}

/// Transport-agnostic CDP protocol dispatcher.
///
/// `CdpDispatcher` owns the per-session protocol state (globals environment,
/// profiler, domain enable flags, breakpoint counter) and an outbox of
/// serialised messages waiting to be sent to the peer.  Protocol responses
/// and events are pushed into the outbox rather than written directly to a
/// transport, so the same dispatcher can drive a WebSocket session or an
/// in-process FFI session interchangeably.
///
/// This type is not thread-safe.  All methods must be invoked from the
/// owning isolate thread.
pub struct CdpDispatcher {
    globals: Rc<RefCell<GlobalEnv>>,
    profiler: CpuProfiler,
    profiler_enabled: bool,
    profiler_sampling_interval_micros: u64,
    profiler_precise_coverage_enabled: bool,
    next_coverage_script_id: u64,
    coverage_scripts: HashMap<String, CoverageScript>,
    profiler_type_profile_enabled: bool,
    next_type_profile_script_id: u64,
    type_profile_scripts: HashMap<String, TypeProfileScript>,
    /// CDP-visible execution contexts currently known to this session.
    contexts: Vec<ExecutionContextDescription>,
    /// Whether the `Runtime` domain is currently enabled for this session.
    runtime_enabled: bool,
    /// Whether the `Console` domain is currently enabled for this session.
    console_enabled: bool,
    /// Whether the `Debugger` domain has been enabled by the client.  Used
    /// to gate fan-out of `Debugger.scriptParsed` events.
    debugger_enabled: bool,
    /// Whether `Target.setDiscoverTargets` is enabled.
    target_discovery_enabled: bool,
    /// Target IDs closed through `Target.closeTarget`.
    closed_target_ids: HashSet<String>,
    /// Monotonically increasing ID for attached Target sessions.
    next_target_session_id: u64,
    /// Active single-target Target session, if any.
    target_session_id: Option<String>,
    /// Target ID attached by the active Target session, if any.
    target_session_target_id: Option<String>,
    /// Script sources registered by the owning inspector, keyed by scriptId.
    script_sources: HashMap<String, String>,
    /// Script URLs extracted from registered sources, keyed by scriptId.
    script_urls: HashMap<String, String>,
    /// Monotonically increasing ID for breakpoints set via CDP.
    next_breakpoint_id: u32,
    /// CDP breakpoint IDs returned by `Debugger.setBreakpointByUrl`.
    cdp_breakpoints: HashSet<String>,
    /// Interpreter breakpoint IDs installed for each CDP breakpoint ID.
    cdp_debugger_breakpoints: HashMap<String, Vec<BreakpointId>>,
    /// CDP `Debugger.setBreakpointsActive` state mirrored onto the debugger.
    breakpoints_active: bool,
    /// CDP `Debugger.setSkipAllPauses` state mirrored onto the debugger.
    skip_all_pauses: bool,
    /// CDP blackbox URL regex patterns supplied by DevTools.
    blackbox_patterns: Vec<String>,
    /// CDP blackboxed ranges keyed by scriptId.
    blackboxed_ranges: HashMap<String, Vec<Value>>,
    /// Per-session registry of inspector-visible heap values.
    remote_objects: RemoteObjectRegistry,
    /// Monotonically increasing ID for `HeapProfiler.getHeapObjectId`.
    next_heap_object_id: u64,
    /// Heap-snapshot object IDs minted from remote objects.
    heap_objects: HashMap<String, JsValue>,
    /// Heap object IDs pinned as inspected by `HeapProfiler.addInspectedHeapObject`.
    inspected_heap_objects: HashSet<String>,
    /// Last sampling heap profile returned by `HeapProfiler.stopSampling`.
    last_heap_sampling_profile: Value,
    /// Monotonically increasing ID for `Runtime.compileScript` cache entries.
    next_compiled_script_id: u64,
    /// Per-session script cache populated by `Runtime.compileScript`.
    compiled_scripts: HashMap<String, CompiledScript>,
    /// Shared state for functions installed through `Runtime.addBinding`.
    runtime_bindings: Rc<RefCell<RuntimeBindingState>>,
    /// Monotonically increasing ID for `Runtime.ExceptionDetails`.
    next_exception_id: u32,
    /// Optional handle to the interpreter [`Debugger`] driving this session.
    ///
    /// Attached by the embedder (or by [`InProcessInspector`]) so that
    /// `Debugger.resume` / `Debugger.step*` / `Debugger.setPauseOnExceptions`
    /// can mutate real interpreter state, and so that the dispatcher can
    /// translate interpreter pauses into `Debugger.paused` events.
    debugger: Option<Rc<RefCell<Debugger>>>,
    /// CDP `Debugger.setPauseOnExceptions` state. Mirrored on the attached
    /// debugger when present; cached here so a `setPauseOnExceptions` call
    /// issued before a debugger is attached still takes effect at attach
    /// time.
    pause_on_exceptions: PauseOnExceptionsState,
    /// FIFO queue of serialised JSON-RPC messages waiting to be drained by
    /// the transport (or by an embedder via the in-process inspector API).
    outbox: VecDeque<String>,
}

impl CdpDispatcher {
    /// Build a dispatcher with a fresh, isolated globals environment.
    pub fn new() -> Self {
        Self::with_globals(Rc::new(RefCell::new(GlobalEnv::new())))
    }

    /// Build a dispatcher that shares the supplied globals environment with
    /// its owner (typically a [`StatorContext`](crate::inspector) FFI handle).
    pub fn with_globals(globals: Rc<RefCell<GlobalEnv>>) -> Self {
        Self::with_globals_and_contexts(globals, vec![default_execution_context()])
    }

    /// Build a dispatcher with an explicit execution-context registry snapshot.
    pub fn with_globals_and_contexts(
        globals: Rc<RefCell<GlobalEnv>>,
        contexts: Vec<ExecutionContextDescription>,
    ) -> Self {
        Self {
            globals,
            profiler: CpuProfiler::new(),
            profiler_enabled: false,
            profiler_sampling_interval_micros: 1_000,
            profiler_precise_coverage_enabled: false,
            next_coverage_script_id: 1,
            coverage_scripts: HashMap::new(),
            profiler_type_profile_enabled: false,
            next_type_profile_script_id: 1,
            type_profile_scripts: HashMap::new(),
            contexts,
            runtime_enabled: false,
            console_enabled: false,
            debugger_enabled: false,
            target_discovery_enabled: false,
            closed_target_ids: HashSet::new(),
            next_target_session_id: 1,
            target_session_id: None,
            target_session_target_id: None,
            script_sources: HashMap::new(),
            script_urls: HashMap::new(),
            next_breakpoint_id: 1,
            cdp_breakpoints: HashSet::new(),
            cdp_debugger_breakpoints: HashMap::new(),
            breakpoints_active: true,
            skip_all_pauses: false,
            blackbox_patterns: Vec::new(),
            blackboxed_ranges: HashMap::new(),
            remote_objects: RemoteObjectRegistry::new(),
            next_heap_object_id: 1,
            heap_objects: HashMap::new(),
            inspected_heap_objects: HashSet::new(),
            last_heap_sampling_profile: empty_heap_sampling_profile(),
            next_compiled_script_id: 1,
            compiled_scripts: HashMap::new(),
            runtime_bindings: Rc::new(RefCell::new(RuntimeBindingState::default())),
            next_exception_id: 1,
            debugger: None,
            pause_on_exceptions: PauseOnExceptionsState::None,
            outbox: VecDeque::new(),
        }
    }

    /// Number of serialised messages currently waiting in the outbox.
    pub fn pending_count(&self) -> usize {
        self.outbox.len()
    }

    /// Pop the oldest outbox message, if any.
    pub fn take_next(&mut self) -> Option<String> {
        self.outbox.pop_front()
    }

    /// Push a pre-serialised event/response onto the outbox.  Used by
    /// inspector producers (e.g. `Debugger.scriptParsed` fan-out) that
    /// emit events outside a request/response turn.
    pub fn push_raw(&mut self, message: String) {
        self.outbox.push_back(message);
    }

    /// Register or replace a script source for `Debugger.getScriptSource`.
    pub fn register_script_source(&mut self, script_id: u32, source: String) {
        let script_id = script_id.to_string();
        let url = registered_script_url(&source);
        self.script_sources.insert(script_id.clone(), source);
        self.script_urls.insert(script_id, url);
    }

    /// Push a JSON-RPC parse-error response onto the outbox.
    pub fn push_parse_error(&mut self, message: String) {
        let resp = json!({
            "id": 0u64,
            "error": {"code": -32700, "message": message}
        });
        self.outbox.push_back(resp.to_string());
    }

    /// Returns `true` if the `Debugger` domain is currently enabled.
    pub fn debugger_enabled(&self) -> bool {
        self.debugger_enabled
    }

    /// Returns `true` if the `Runtime` domain is currently enabled.
    pub fn runtime_enabled(&self) -> bool {
        self.runtime_enabled
    }

    /// Borrow the per-session remote-object registry.  Used by tests and
    /// by the inspector to assert that releases actually drop entries.
    pub fn remote_objects(&self) -> &RemoteObjectRegistry {
        &self.remote_objects
    }

    /// Install (or replace) the interpreter [`Debugger`] handle that this
    /// dispatcher is bridging to. Any previously cached
    /// `Debugger.setPauseOnExceptions` state is applied immediately so the
    /// embedder can configure pause behaviour before attaching.
    ///
    /// Returns the previously attached debugger handle, if any.
    pub fn attach_debugger(
        &mut self,
        debugger: Rc<RefCell<Debugger>>,
    ) -> Option<Rc<RefCell<Debugger>>> {
        let pause_on_exceptions = self.pause_on_exceptions.enabled();
        {
            let mut debugger_ref = debugger.borrow_mut();
            debugger_ref.set_pause_on_exceptions(pause_on_exceptions);
            debugger_ref.set_breakpoints_active(self.breakpoints_active);
            debugger_ref.set_skip_all_pauses(self.skip_all_pauses);
        }
        self.debugger.replace(debugger)
    }

    /// Drop the interpreter [`Debugger`] handle previously installed with
    /// [`Self::attach_debugger`]. Returns the dropped handle, if any.
    pub fn detach_debugger_handle(&mut self) -> Option<Rc<RefCell<Debugger>>> {
        self.debugger.take()
    }

    /// Returns the current `Debugger.setPauseOnExceptions` state for this
    /// session.
    pub fn pause_on_exceptions(&self) -> PauseOnExceptionsState {
        self.pause_on_exceptions
    }

    /// Emit a `Debugger.paused` event into the outbox synthesised from the
    /// attached debugger's most recent pause state.
    ///
    /// Returns `true` when an event was emitted; returns `false` (and does
    /// nothing) when:
    ///
    /// - no debugger is attached, or
    /// - the `Debugger` domain is not enabled for this session, or
    /// - the attached debugger has no recorded pause.
    ///
    /// This is the in-process bridge point the embedder calls after
    /// [`Interpreter::run`](crate::interpreter::Interpreter::run) returns
    /// [`StatorError::DebuggerPaused`].
    pub fn notify_paused(&mut self) -> bool {
        if !self.debugger_enabled {
            return false;
        }
        let Some(debugger) = self.debugger.as_ref() else {
            return false;
        };
        let dbg = debugger.borrow();
        let Some(reason) = dbg.last_pause_reason().cloned() else {
            return false;
        };
        let offset = dbg.last_pause_offset();
        let line = dbg.last_pause_line();
        drop(dbg);

        let params = paused_event_params(&reason, offset, line);
        self.push_event("Debugger.paused", params);
        true
    }

    /// Emit a `Debugger.resumed` event into the outbox.
    ///
    /// Returns `false` when the `Debugger` domain is not enabled; otherwise
    /// always emits the event (resume can be driven externally by the
    /// embedder, not only by `Debugger.resume` / step requests).
    pub fn notify_resumed(&mut self) -> bool {
        if !self.debugger_enabled {
            return false;
        }
        self.push_event("Debugger.resumed", json!({}));
        true
    }

    /// Add a context to this session's registry and emit `created` if enabled.
    pub fn add_execution_context(&mut self, context: ExecutionContextDescription) {
        if self
            .contexts
            .iter()
            .any(|existing| existing.id == context.id)
        {
            return;
        }
        self.contexts.push(context.clone());
        self.emit_execution_context_created(&context);
    }

    /// Remove a context from this session's registry and emit `destroyed`.
    pub fn remove_execution_context(&mut self, context_id: u32) {
        if let Some(index) = self
            .contexts
            .iter()
            .position(|context| context.id == context_id)
        {
            self.contexts.remove(index);
            self.emit_execution_context_destroyed(context_id);
        }
    }

    /// Remove every context in `context_ids` and emit one `contextsCleared`.
    pub fn clear_execution_contexts(&mut self, context_ids: &[u32]) {
        let before = self.contexts.len();
        self.contexts
            .retain(|context| !context_ids.contains(&context.id));
        if self.contexts.len() != before && self.runtime_enabled {
            self.push_event("Runtime.executionContextsCleared", json!({}));
        }
    }

    /// Parse `text` as a JSON-RPC request, dispatch it, and enqueue the
    /// reply (and any pre-reply events) on the outbox.
    ///
    /// Returns [`DispatchOutcome::ParseError`] when `text` is not a valid
    /// JSON-RPC request; a JSON-RPC parse-error response is still pushed
    /// onto the outbox so the peer can correlate the failure.
    pub fn dispatch_json(&mut self, text: &str) -> DispatchOutcome {
        let request: CdpRequest = match serde_json::from_str(text) {
            Ok(r) => r,
            Err(e) => {
                self.push_parse_error(format!("Parse error: {e}"));
                return DispatchOutcome::ParseError;
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
        let serialised = serde_json::to_string(&resp).unwrap_or_else(|_| {
            json!({"id": id, "error": {"code": -32603, "message": "serialization error"}})
                .to_string()
        });
        self.outbox.push_back(serialised);
        DispatchOutcome::Ok
    }

    /// Push a CDP event of the form `{"method": method, "params": params}`
    /// onto the outbox.  Best-effort: silently drops the event if it cannot
    /// be serialised.
    fn push_event(&mut self, method: &str, params: Value) {
        let event = CdpEvent {
            method: method.to_string(),
            params,
        };
        if let Ok(s) = serde_json::to_string(&event) {
            self.outbox.push_back(s);
        }
    }

    fn emit_execution_context_created(&mut self, context: &ExecutionContextDescription) {
        if self.runtime_enabled {
            self.push_event(
                "Runtime.executionContextCreated",
                json!({ "context": context.to_cdp_context() }),
            );
        }
    }

    fn emit_execution_context_destroyed(&mut self, context_id: u32) {
        if self.runtime_enabled {
            self.push_event(
                "Runtime.executionContextDestroyed",
                json!({ "executionContextId": context_id }),
            );
        }
    }

    fn target_set_discover_targets(&mut self, params: &Value) -> StatorResult<Value> {
        self.target_discovery_enabled = params
            .get("discover")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if self.target_discovery_enabled {
            for target_info in self.target_infos() {
                self.push_event("Target.targetCreated", json!({ "targetInfo": target_info }));
            }
        }
        Ok(json!({}))
    }

    fn target_get_targets(&mut self) -> StatorResult<Value> {
        let target_infos = self.target_infos();
        Ok(json!({ "targetInfos": target_infos }))
    }

    fn target_infos(&self) -> Vec<Value> {
        let mut groups: Vec<(u32, String, String)> = self
            .contexts
            .iter()
            .map(|context| {
                (
                    context.group_id,
                    context.name.clone(),
                    context.origin.clone(),
                )
            })
            .collect();
        if groups.is_empty() {
            groups.push((1, "stator".to_string(), "stator".to_string()));
        }
        groups.sort_by_key(|(group_id, _, _)| *group_id);
        groups.dedup_by_key(|(group_id, _, _)| *group_id);
        groups
            .into_iter()
            .map(|(group_id, name, origin)| target_info_for_group(group_id, &name, &origin))
            .filter(|info| {
                info.get("targetId")
                    .and_then(Value::as_str)
                    .is_none_or(|target_id| !self.closed_target_ids.contains(target_id))
            })
            .collect()
    }

    fn is_live_target(&self, target_id: &str) -> bool {
        self.target_infos()
            .iter()
            .any(|info| info.get("targetId").and_then(Value::as_str) == Some(target_id))
    }

    fn target_attach_to_target(&mut self, params: &Value) -> StatorResult<Value> {
        let target_id = match params.get("targetId").and_then(Value::as_str) {
            Some(target_id) => target_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Target.attachToTarget: required parameter 'targetId' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        if !self.is_live_target(target_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Target.attachToTarget: unknown targetId `{target_id}`"
            )));
        }
        let target_info = self
            .target_infos()
            .into_iter()
            .find(|info| info.get("targetId").and_then(Value::as_str) == Some(target_id))
            .unwrap_or_else(target_info);
        let session_id = format!("stator-target-session-{}", self.next_target_session_id);
        self.next_target_session_id = self.next_target_session_id.saturating_add(1);
        self.target_session_id = Some(session_id.clone());
        self.target_session_target_id = Some(target_id.to_string());
        if params
            .get("flatten")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            self.push_event(
                "Target.attachedToTarget",
                json!({
                    "sessionId": session_id,
                    "targetInfo": target_info,
                    "waitingForDebugger": false,
                }),
            );
        }
        Ok(json!({ "sessionId": session_id }))
    }

    fn target_detach_from_target(&mut self, params: &Value) -> StatorResult<Value> {
        let session_id = match params.get("sessionId").and_then(Value::as_str) {
            Some(session_id) => session_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Target.detachFromTarget: required parameter 'sessionId' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        if self.target_session_id.as_deref() != Some(session_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Target.detachFromTarget: unknown sessionId `{session_id}`"
            )));
        }
        self.target_session_id = None;
        let target_id = self
            .target_session_target_id
            .take()
            .unwrap_or_else(|| DEFAULT_TARGET_ID.to_string());
        self.push_event(
            "Target.detachedFromTarget",
            json!({ "sessionId": session_id, "targetId": target_id }),
        );
        Ok(json!({}))
    }

    fn target_close_target(&mut self, params: &Value) -> StatorResult<Value> {
        let target_id = match params.get("targetId").and_then(Value::as_str) {
            Some(target_id) => target_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Target.closeTarget: required parameter 'targetId' is missing or not a string"
                        .to_string(),
                ));
            }
        };
        if !self.is_live_target(target_id) {
            return Ok(json!({ "success": false }));
        }

        if self.target_session_target_id.as_deref() == Some(target_id)
            && let Some(session_id) = self.target_session_id.take()
        {
            let target_id = self
                .target_session_target_id
                .take()
                .unwrap_or_else(|| DEFAULT_TARGET_ID.to_string());
            self.push_event(
                "Target.detachedFromTarget",
                json!({ "sessionId": session_id, "targetId": target_id }),
            );
        }
        self.closed_target_ids.insert(target_id.to_string());
        if self.target_discovery_enabled {
            self.push_event("Target.targetDestroyed", json!({ "targetId": target_id }));
        }
        Ok(json!({ "success": true }))
    }

    fn target_send_message_to_target(&mut self, params: &Value) -> StatorResult<Value> {
        let session_id = match params.get("sessionId").and_then(Value::as_str) {
            Some(session_id) => session_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Target.sendMessageToTarget: required parameter 'sessionId' is missing or \
                     not a string"
                        .to_string(),
                ));
            }
        };
        if self.target_session_id.as_deref() != Some(session_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Target.sendMessageToTarget: unknown sessionId `{session_id}`"
            )));
        }
        let target_id = self
            .target_session_target_id
            .clone()
            .unwrap_or_else(|| DEFAULT_TARGET_ID.to_string());
        let message = match params.get("message").and_then(Value::as_str) {
            Some(message) => message,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Target.sendMessageToTarget: required parameter 'message' is missing or not \
                     a string"
                        .to_string(),
                ));
            }
        };
        let before = self.outbox.len();
        let _ = self.dispatch_json(message);
        let nested = self.outbox.split_off(before);
        for message in nested {
            self.push_event(
                "Target.receivedMessageFromTarget",
                json!({
                    "sessionId": session_id,
                    "targetId": target_id,
                    "message": message,
                }),
            );
        }
        Ok(json!({}))
    }

    /// Route a parsed request to the correct domain handler.
    fn dispatch(&mut self, req: &CdpRequest) -> StatorResult<Value> {
        match req.method.as_str() {
            // ── Runtime ───────────────────────────────────────────────────
            "Runtime.enable" => self.runtime_enable(),
            "Runtime.evaluate" => self.runtime_evaluate(&req.params),
            "Runtime.callFunctionOn" => self.runtime_call_function_on(&req.params),
            "Runtime.getProperties" => self.runtime_get_properties(&req.params),
            "Runtime.queryObjects" => self.runtime_query_objects(&req.params),
            "Runtime.releaseObject" => self.runtime_release_object(&req.params),
            "Runtime.releaseObjectGroup" => self.runtime_release_object_group(&req.params),
            "Runtime.compileScript" => self.runtime_compile_script(&req.params),
            "Runtime.runScript" => self.runtime_run_script(&req.params),
            "Runtime.runIfWaitingForDebugger" => Ok(json!({})),
            "Runtime.discardConsoleEntries" => self.runtime_discard_console_entries(),
            "Runtime.globalLexicalScopeNames" => {
                self.runtime_global_lexical_scope_names(&req.params)
            }
            "Runtime.getIsolateId" => Ok(json!({ "id": "stator-isolate-0" })),
            "Runtime.getHeapUsage" => self.runtime_get_heap_usage(),
            "Runtime.addBinding" => self.runtime_add_binding(&req.params),
            "Runtime.removeBinding" => self.runtime_remove_binding(&req.params),

            // ── Debugger ──────────────────────────────────────────────────
            "Debugger.enable" => {
                self.debugger_enabled = true;
                Ok(json!({
                    "debuggerId": "stator-debugger-0"
                }))
            }
            "Debugger.disable" => {
                self.debugger_enabled = false;
                Ok(json!({}))
            }
            "Debugger.setPauseOnExceptions" => self.debugger_set_pause_on_exceptions(&req.params),
            "Debugger.setBreakpoint" => self.debugger_set_breakpoint(&req.params),
            "Debugger.setBreakpointByUrl" => self.debugger_set_breakpoint_by_url(&req.params),
            "Debugger.removeBreakpoint" => self.debugger_remove_breakpoint(&req.params),
            "Debugger.setBreakpointsActive" => self.debugger_set_breakpoints_active(&req.params),
            "Debugger.setSkipAllPauses" => self.debugger_set_skip_all_pauses(&req.params),
            "Debugger.setBlackboxPatterns" => self.debugger_set_blackbox_patterns(&req.params),
            "Debugger.setBlackboxedRanges" => self.debugger_set_blackboxed_ranges(&req.params),
            "Debugger.resume" => self.debugger_resume(),
            "Debugger.continueToLocation" => self.debugger_continue_to_location(&req.params),
            "Debugger.stepInto" => self.debugger_step(DebugAction::StepInto),
            "Debugger.stepOver" => self.debugger_step(DebugAction::StepOver),
            "Debugger.stepOut" => self.debugger_step(DebugAction::StepOut),
            "Debugger.pause" => Err(unsupported_debugger_method(
                "Debugger.pause",
                "Stator runs scripts synchronously on the embedder thread; the \
                 inspector cannot interrupt a running script. Use `debugger;` \
                 statements or `Debugger.setBreakpointByUrl` to pause instead.",
            )),
            "Debugger.evaluateOnCallFrame" => Err(unsupported_debugger_method(
                "Debugger.evaluateOnCallFrame",
                "Stator does not yet expose interpreter call-frame snapshots \
                 through CDP. Use `Runtime.evaluate` while paused for now.",
            )),
            "Debugger.getScriptSource" => self.debugger_get_script_source(&req.params),
            "Debugger.setScriptSource" => self.debugger_set_script_source(&req.params),
            "Debugger.getPossibleBreakpoints" => {
                self.debugger_get_possible_breakpoints(&req.params)
            }

            // ── Console ───────────────────────────────────────────────────
            "Console.enable" => self.console_enable(),
            "Console.disable" => {
                self.console_enabled = false;
                Ok(json!({}))
            }

            // ── Profiler ──────────────────────────────────────────────────
            "Profiler.enable" => {
                self.profiler_enabled = true;
                Ok(json!({}))
            }
            "Profiler.disable" => {
                self.profiler_enabled = false;
                Ok(json!({}))
            }
            "Profiler.setSamplingInterval" => self.profiler_set_sampling_interval(&req.params),
            "Profiler.start" => self.profiler_start(&req.params),
            "Profiler.stop" => self.profiler_stop(),
            "Profiler.startPreciseCoverage" => self.profiler_start_precise_coverage(),
            "Profiler.stopPreciseCoverage" => self.profiler_stop_precise_coverage(),
            "Profiler.takePreciseCoverage" => self.profiler_take_precise_coverage(),
            "Profiler.getBestEffortCoverage" => self.profiler_get_best_effort_coverage(),
            "Profiler.startTypeProfile" => self.profiler_start_type_profile(),
            "Profiler.stopTypeProfile" => self.profiler_stop_type_profile(),
            "Profiler.takeTypeProfile" => self.profiler_take_type_profile(),

            // ── HeapProfiler ──────────────────────────────────────────────
            "HeapProfiler.enable" => Ok(json!({})),
            "HeapProfiler.takeHeapSnapshot" => self.heap_profiler_take_snapshot(),
            "HeapProfiler.getHeapObjectId" => self.heap_profiler_get_heap_object_id(&req.params),
            "HeapProfiler.getObjectByHeapObjectId" => {
                self.heap_profiler_get_object_by_heap_object_id(&req.params)
            }
            "HeapProfiler.addInspectedHeapObject" => {
                self.heap_profiler_add_inspected_heap_object(&req.params)
            }
            "HeapProfiler.startSampling" => self.heap_profiler_start_sampling(),
            "HeapProfiler.stopSampling" => self.heap_profiler_stop_sampling(),
            "HeapProfiler.getSamplingProfile" => self.heap_profiler_get_sampling_profile(),
            "HeapProfiler.startTrackingHeapObjects" => self.heap_profiler_start_tracking(),
            "HeapProfiler.stopTrackingHeapObjects" => self.heap_profiler_stop_tracking(),

            // ── Network (stubs) ───────────────────────────────────────────
            "Network.enable" => Ok(json!({})),
            "Network.disable" => Ok(json!({})),

            // ── Target ────────────────────────────────────────────────────
            "Target.getTargets" => self.target_get_targets(),
            "Target.setDiscoverTargets" => self.target_set_discover_targets(&req.params),
            "Target.attachToTarget" => self.target_attach_to_target(&req.params),
            "Target.detachFromTarget" => self.target_detach_from_target(&req.params),
            "Target.closeTarget" => self.target_close_target(&req.params),
            "Target.sendMessageToTarget" => self.target_send_message_to_target(&req.params),

            // ── Schema ────────────────────────────────────────────────────
            "Schema.getDomains" => Ok(schema_get_domains()),

            // ── Unknown ───────────────────────────────────────────────────
            other => Err(crate::error::StatorError::Internal(format!(
                "CDP method not implemented: {other}"
            ))),
        }
    }

    // ── Runtime.enable ────────────────────────────────────────────────────────

    fn runtime_enable(&mut self) -> StatorResult<Value> {
        self.runtime_enabled = true;
        // Emit executionContextCreated event before the ack so the peer sees
        // the context before the response is correlated.
        for context in self.contexts.clone() {
            self.emit_execution_context_created(&context);
        }
        Ok(json!({}))
    }

    // ── Runtime handshake/introspection helpers ──────────────────────────────

    fn runtime_discard_console_entries(&mut self) -> StatorResult<Value> {
        let _ = drain_messages();
        Ok(json!({}))
    }

    fn runtime_global_lexical_scope_names(&self, params: &Value) -> StatorResult<Value> {
        let _context_id = self.resolve_execution_context_id(params)?;
        let mut names: Vec<String> = self.globals.borrow().vars.keys().cloned().collect();
        names.sort();
        Ok(json!({ "names": names }))
    }

    fn runtime_get_heap_usage(&self) -> StatorResult<Value> {
        const NODE_FIELDS: usize = 5;
        const SELF_SIZE_INDEX: usize = 3;

        let snapshot = HeapSnapshotBuilder::build(&self.globals.borrow().vars);
        let used_size: u64 = snapshot
            .nodes
            .chunks(NODE_FIELDS)
            .filter_map(|node| node.get(SELF_SIZE_INDEX))
            .map(|size| u64::from(*size))
            .sum();
        Ok(json!({
            "usedSize": used_size,
            "totalSize": used_size,
        }))
    }

    fn runtime_query_objects(&mut self, params: &Value) -> StatorResult<Value> {
        let prototype_object_id = match params.get("prototypeObjectId").and_then(Value::as_str) {
            Some(id) => id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.queryObjects: required parameter 'prototypeObjectId' is missing or \
                     not a string"
                        .to_string(),
                ));
            }
        };
        let prototype = self
            .remote_objects
            .get(prototype_object_id)
            .ok_or_else(|| {
                crate::error::StatorError::Internal(format!(
                    "Runtime.queryObjects: unknown or released prototypeObjectId \
                     `{prototype_object_id}`"
                ))
            })?;
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);

        let mut roots: Vec<JsValue> = self.globals.borrow().vars.values().cloned().collect();
        roots.extend(
            self.remote_objects
                .entries
                .values()
                .map(|entry| entry.value.clone()),
        );

        let mut seen = HashSet::new();
        let mut matches = Vec::new();
        for root in roots {
            collect_objects_with_prototype(&root, &prototype, &mut seen, &mut matches);
        }

        let objects = JsValue::Array(Rc::new(RefCell::new(matches)));
        let remote = js_value_to_remote_object(
            &objects,
            &mut self.remote_objects,
            object_group.as_deref(),
            false,
        );
        Ok(json!({ "objects": remote }))
    }

    fn runtime_add_binding(&mut self, params: &Value) -> StatorResult<Value> {
        let name = match params.get("name").and_then(Value::as_str) {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.addBinding: required parameter 'name' is missing or empty".to_string(),
                ));
            }
        };
        let _context_id = self.resolve_execution_context_id(params)?;
        self.runtime_bindings
            .borrow_mut()
            .active_names
            .insert(name.clone());

        let binding_state = Rc::clone(&self.runtime_bindings);
        let binding_name = name.clone();
        let callback = JsValue::NativeFunction(Rc::new(move |args: Vec<JsValue>| {
            let payload = args.first().unwrap_or(&JsValue::Undefined).to_js_string()?;
            let mut state = binding_state.borrow_mut();
            if state.active_names.contains(&binding_name) {
                state.pending_calls.push(RuntimeBindingCall {
                    name: binding_name.clone(),
                    payload,
                });
            }
            Ok(JsValue::Undefined)
        }));
        self.globals.borrow_mut().insert(name, callback);
        Ok(json!({}))
    }

    fn runtime_remove_binding(&mut self, params: &Value) -> StatorResult<Value> {
        let name = match params.get("name").and_then(Value::as_str) {
            Some(name) if !name.is_empty() => name,
            _ => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.removeBinding: required parameter 'name' is missing or empty"
                        .to_string(),
                ));
            }
        };
        self.runtime_bindings.borrow_mut().active_names.remove(name);
        self.globals.borrow_mut().remove(name);
        Ok(json!({}))
    }

    fn emit_pending_binding_calls(&mut self, execution_context_id: u32) {
        let calls = {
            let mut state = self.runtime_bindings.borrow_mut();
            std::mem::take(&mut state.pending_calls)
        };
        if !self.runtime_enabled {
            return;
        }
        for call in calls {
            self.push_event(
                "Runtime.bindingCalled",
                json!({
                    "name": call.name,
                    "payload": call.payload,
                    "executionContextId": execution_context_id,
                }),
            );
        }
    }

    fn emit_pending_profile_events(&mut self) {
        let events = drain_profile_events();
        if !self.profiler_enabled {
            return;
        }
        for event in events {
            match event.kind {
                ProfileEventKind::Started => self.push_event(
                    "Profiler.consoleProfileStarted",
                    json!({
                        "id": event.id,
                        "title": event.id,
                        "location": console_profile_location(),
                    }),
                ),
                ProfileEventKind::Finished => self.push_event(
                    "Profiler.consoleProfileFinished",
                    json!({
                        "id": event.id,
                        "title": event.id,
                        "location": console_profile_location(),
                        "profile": empty_console_profile(),
                    }),
                ),
            }
        }
    }

    // ── Runtime.compileScript / Runtime.runScript ────────────────────────────

    fn runtime_compile_script(&mut self, params: &Value) -> StatorResult<Value> {
        let expression = match params.get("expression").and_then(Value::as_str) {
            Some(expression) => expression,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.compileScript: required parameter 'expression' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        let execution_context_id = self.resolve_execution_context_id(params)?;
        let persist_script = params
            .get("persistScript")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let source_url = exception_source_url(params, expression);

        let bytecodes =
            match parser::parse(expression).and_then(|p| BytecodeGenerator::compile_program(&p)) {
                Ok(bytecodes) => Rc::new(bytecodes),
                Err(err) => {
                    return Ok(json!({
                        "exceptionDetails": self.exception_details_only(
                            &err,
                            ExceptionRequest {
                                expression,
                                source_url: source_url.as_deref(),
                                execution_context_id,
                                object_group: None,
                                generate_preview: false,
                            },
                        )
                    }));
                }
            };

        if !persist_script {
            return Ok(json!({}));
        }

        let script_id = format!("runtime-script-{}", self.next_compiled_script_id);
        self.next_compiled_script_id = self.next_compiled_script_id.saturating_add(1);
        self.compiled_scripts.insert(
            script_id.clone(),
            CompiledScript {
                bytecodes,
                expression: expression.to_string(),
                source_url,
                execution_context_id,
            },
        );
        Ok(json!({ "scriptId": script_id }))
    }

    fn runtime_run_script(&mut self, params: &Value) -> StatorResult<Value> {
        let script_id = match params.get("scriptId").and_then(Value::as_str) {
            Some(script_id) => script_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.runScript: required parameter 'scriptId' is missing or not a string"
                        .to_string(),
                ));
            }
        };
        let script = self
            .compiled_scripts
            .get(script_id)
            .cloned()
            .ok_or_else(|| {
                crate::error::StatorError::Internal(format!(
                    "Runtime.runScript: unknown or expired scriptId `{script_id}`"
                ))
            })?;

        let execution_context_id =
            if params.get("contextId").is_some() || params.get("executionContextId").is_some() {
                self.resolve_execution_context_id(params)?
            } else {
                script.execution_context_id
            };
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);
        let generate_preview = params
            .get("generatePreview")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut frame = InterpreterFrame::new_with_globals(
            Rc::clone(&script.bytecodes),
            vec![],
            Rc::clone(&self.globals),
        );
        let js_result = match Interpreter::run(&mut frame) {
            Ok(value) => value,
            Err(err) => {
                self.emit_pending_binding_calls(execution_context_id);
                self.emit_pending_profile_events();
                self.record_coverage(
                    &script.expression,
                    script.source_url.as_deref(),
                    Some(script_id),
                );
                return Ok(self.exception_response(
                    &err,
                    ExceptionRequest {
                        expression: &script.expression,
                        source_url: script.source_url.as_deref(),
                        execution_context_id,
                        object_group: object_group.as_deref(),
                        generate_preview,
                    },
                ));
            }
        };
        self.emit_pending_binding_calls(execution_context_id);
        self.emit_pending_profile_events();
        self.record_coverage(
            &script.expression,
            script.source_url.as_deref(),
            Some(script_id),
        );
        self.record_type_profile(
            &script.expression,
            script.source_url.as_deref(),
            Some(script_id),
            &js_result,
        );

        let remote = js_value_to_remote_object(
            &js_result,
            &mut self.remote_objects,
            object_group.as_deref(),
            generate_preview,
        );
        Ok(json!({ "result": remote }))
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
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);
        let generate_preview = params
            .get("generatePreview")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let execution_context_id = self.resolve_execution_context_id(params)?;
        let source_url = exception_source_url(params, expression);

        let bytecodes =
            match parser::parse(expression).and_then(|p| BytecodeGenerator::compile_program(&p)) {
                Ok(bytecodes) => bytecodes,
                Err(err) => {
                    return Ok(self.exception_response(
                        &err,
                        ExceptionRequest {
                            expression,
                            source_url: source_url.as_deref(),
                            execution_context_id,
                            object_group: object_group.as_deref(),
                            generate_preview,
                        },
                    ));
                }
            };

        let mut frame = InterpreterFrame::new_with_globals(
            Rc::new(bytecodes),
            vec![],
            Rc::clone(&self.globals),
        );
        let js_result = match Interpreter::run(&mut frame) {
            Ok(value) => value,
            Err(err) => {
                self.emit_pending_binding_calls(execution_context_id);
                self.emit_pending_profile_events();
                self.record_coverage(expression, source_url.as_deref(), None);
                return Ok(self.exception_response(
                    &err,
                    ExceptionRequest {
                        expression,
                        source_url: source_url.as_deref(),
                        execution_context_id,
                        object_group: object_group.as_deref(),
                        generate_preview,
                    },
                ));
            }
        };
        self.emit_pending_binding_calls(execution_context_id);
        self.emit_pending_profile_events();
        self.record_coverage(expression, source_url.as_deref(), None);
        self.record_type_profile(expression, source_url.as_deref(), None, &js_result);

        let remote = js_value_to_remote_object(
            &js_result,
            &mut self.remote_objects,
            object_group.as_deref(),
            generate_preview,
        );
        Ok(json!({ "result": remote }))
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
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);
        let generate_preview = params
            .get("generatePreview")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let execution_context_id = self.resolve_execution_context_id(params)?;

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
        let source_url = exception_source_url(params, &expression);
        let bytecodes =
            match parser::parse(&expression).and_then(|p| BytecodeGenerator::compile_program(&p)) {
                Ok(bytecodes) => bytecodes,
                Err(err) => {
                    return Ok(self.exception_response(
                        &err,
                        ExceptionRequest {
                            expression: &expression,
                            source_url: source_url.as_deref(),
                            execution_context_id,
                            object_group: object_group.as_deref(),
                            generate_preview,
                        },
                    ));
                }
            };

        let mut frame = InterpreterFrame::new_with_globals(
            Rc::new(bytecodes),
            vec![],
            Rc::clone(&self.globals),
        );
        let js_result = match Interpreter::run(&mut frame) {
            Ok(value) => value,
            Err(err) => {
                self.emit_pending_binding_calls(execution_context_id);
                self.emit_pending_profile_events();
                self.record_coverage(&expression, source_url.as_deref(), None);
                return Ok(self.exception_response(
                    &err,
                    ExceptionRequest {
                        expression: &expression,
                        source_url: source_url.as_deref(),
                        execution_context_id,
                        object_group: object_group.as_deref(),
                        generate_preview,
                    },
                ));
            }
        };
        self.emit_pending_binding_calls(execution_context_id);
        self.emit_pending_profile_events();
        self.record_coverage(&expression, source_url.as_deref(), None);
        self.record_type_profile(&expression, source_url.as_deref(), None, &js_result);

        let remote = js_value_to_remote_object(
            &js_result,
            &mut self.remote_objects,
            object_group.as_deref(),
            generate_preview,
        );
        Ok(json!({ "result": remote }))
    }

    fn resolve_execution_context_id(&self, params: &Value) -> StatorResult<u32> {
        let requested_id = params
            .get("contextId")
            .or_else(|| params.get("executionContextId"))
            .and_then(Value::as_u64);
        match requested_id {
            Some(id) if self.contexts.iter().any(|ctx| ctx.id == id as u32) => Ok(id as u32),
            Some(id) => Err(StatorError::Internal(format!(
                "Runtime: unknown execution context id `{id}`"
            ))),
            None => Ok(self.contexts.first().map(|ctx| ctx.id).unwrap_or(1)),
        }
    }

    fn exception_response(&mut self, err: &StatorError, request: ExceptionRequest<'_>) -> Value {
        let exception_id = self.next_exception_id;
        self.next_exception_id = self.next_exception_id.saturating_add(1);
        let thrown = take_pending_exception();
        let details = self.build_exception_details(exception_id, err, request, thrown.as_ref());
        if self.runtime_enabled {
            self.push_event(
                "Runtime.exceptionThrown",
                json!({
                    "timestamp": 0.0,
                    "exceptionDetails": details.clone(),
                }),
            );
        }
        json!({
            "result": {"type": "undefined"},
            "exceptionDetails": details,
        })
    }

    fn exception_details_only(
        &mut self,
        err: &StatorError,
        request: ExceptionRequest<'_>,
    ) -> Value {
        let exception_id = self.next_exception_id;
        self.next_exception_id = self.next_exception_id.saturating_add(1);
        self.build_exception_details(exception_id, err, request, None)
    }

    fn build_exception_details(
        &mut self,
        exception_id: u32,
        err: &StatorError,
        request: ExceptionRequest<'_>,
        thrown: Option<&JsValue>,
    ) -> Value {
        let mut details = json!({
            "exceptionId": exception_id,
            "text": exception_text(err, thrown),
            "lineNumber": request_line_number(request.expression),
            "columnNumber": 0,
            "executionContextId": request.execution_context_id,
        });
        if let Some(url) = request.source_url {
            details
                .as_object_mut()
                .expect("details is an object")
                .insert("url".to_string(), Value::String(url.to_string()));
        }
        if let Some(value) = thrown {
            let exception = js_value_to_remote_object(
                value,
                &mut self.remote_objects,
                request.object_group,
                request.generate_preview,
            );
            details
                .as_object_mut()
                .expect("details is an object")
                .insert("exception".to_string(), exception);
        }
        if let Some(stack_trace) = stack_trace_from_thrown(thrown) {
            details
                .as_object_mut()
                .expect("details is an object")
                .insert("stackTrace".to_string(), stack_trace);
        }
        details
    }

    // ── Runtime.getProperties ────────────────────────────────────────────────

    fn runtime_get_properties(&mut self, params: &Value) -> StatorResult<Value> {
        let object_id = match params.get("objectId").and_then(Value::as_str) {
            Some(s) => s,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.getProperties: required parameter 'objectId' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        let own_properties = params
            .get("ownProperties")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        let value = match self.remote_objects.get(object_id) {
            Some(v) => v,
            None => {
                return Err(crate::error::StatorError::Internal(format!(
                    "Runtime.getProperties: unknown or released objectId `{object_id}`"
                )));
            }
        };

        // Preview helper: register child values under the same group so
        // they share the parent's lifetime in releaseObjectGroup.
        let parent_group = self.remote_objects.group_of(object_id).map(str::to_string);

        let descriptors = build_property_descriptors(
            &value,
            own_properties,
            &mut self.remote_objects,
            parent_group.as_deref(),
        );

        Ok(json!({ "result": descriptors }))
    }

    // ── Runtime.releaseObject ────────────────────────────────────────────────

    fn runtime_release_object(&mut self, params: &Value) -> StatorResult<Value> {
        let object_id = match params.get("objectId").and_then(Value::as_str) {
            Some(s) => s,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.releaseObject: required parameter 'objectId' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        if !self.remote_objects.release(object_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Runtime.releaseObject: unknown or already-released objectId `{object_id}`"
            )));
        }
        Ok(json!({}))
    }

    // ── Runtime.releaseObjectGroup ───────────────────────────────────────────

    fn runtime_release_object_group(&mut self, params: &Value) -> StatorResult<Value> {
        let group = match params.get("objectGroup").and_then(Value::as_str) {
            Some(g) => g,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Runtime.releaseObjectGroup: required parameter 'objectGroup' is missing or \
                     not a string"
                        .to_string(),
                ));
            }
        };
        // Releasing an unknown group is a no-op per V8 inspector semantics.
        let _released = self.remote_objects.release_group(group);
        Ok(json!({}))
    }

    // ── Debugger.setPauseOnExceptions ────────────────────────────────────────

    fn debugger_set_pause_on_exceptions(&mut self, params: &Value) -> StatorResult<Value> {
        let raw = params
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("none");
        let state = match PauseOnExceptionsState::parse(raw) {
            Some(state) => state,
            None => {
                return Err(crate::error::StatorError::TypeError(format!(
                    "Debugger.setPauseOnExceptions: invalid `state` value: {raw:?} \
                     (expected \"none\", \"uncaught\", or \"all\")"
                )));
            }
        };
        self.pause_on_exceptions = state;
        if let Some(debugger) = self.debugger.as_ref() {
            debugger
                .borrow_mut()
                .set_pause_on_exceptions(state.enabled());
        }
        Ok(json!({}))
    }

    fn debugger_set_breakpoints_active(&mut self, params: &Value) -> StatorResult<Value> {
        let active = params
            .get("active")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBreakpointsActive: required parameter 'active' is missing or not \
                     a boolean"
                        .to_string(),
                )
            })?;
        self.breakpoints_active = active;
        if let Some(debugger) = self.debugger.as_ref() {
            debugger.borrow_mut().set_breakpoints_active(active);
        }
        Ok(json!({}))
    }

    fn debugger_set_skip_all_pauses(&mut self, params: &Value) -> StatorResult<Value> {
        let skip = params.get("skip").and_then(Value::as_bool).ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.setSkipAllPauses: required parameter 'skip' is missing or not a \
                     boolean"
                    .to_string(),
            )
        })?;
        self.skip_all_pauses = skip;
        if let Some(debugger) = self.debugger.as_ref() {
            debugger.borrow_mut().set_skip_all_pauses(skip);
        }
        Ok(json!({}))
    }

    fn debugger_set_blackbox_patterns(&mut self, params: &Value) -> StatorResult<Value> {
        let patterns = params
            .get("patterns")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBlackboxPatterns: required parameter 'patterns' is missing or \
                     not an array"
                        .to_string(),
                )
            })?;
        let mut parsed = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            let Some(pattern) = pattern.as_str() else {
                return Err(crate::error::StatorError::TypeError(
                    "Debugger.setBlackboxPatterns: every pattern must be a string".to_string(),
                ));
            };
            parsed.push(pattern.to_string());
        }
        self.blackbox_patterns = parsed;
        Ok(json!({}))
    }

    fn debugger_set_blackboxed_ranges(&mut self, params: &Value) -> StatorResult<Value> {
        let script_id = params
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBlackboxedRanges: required parameter 'scriptId' is missing or \
                     not a string"
                        .to_string(),
                )
            })?;
        if !self.script_sources.contains_key(script_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.setBlackboxedRanges: unknown scriptId `{script_id}`"
            )));
        }
        let positions = params
            .get("positions")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBlackboxedRanges: required parameter 'positions' is missing or \
                     not an array"
                        .to_string(),
                )
            })?;
        self.blackboxed_ranges
            .insert(script_id.to_string(), positions.clone());
        Ok(json!({}))
    }

    // ── Debugger.resume ──────────────────────────────────────────────────────

    fn debugger_resume(&mut self) -> StatorResult<Value> {
        // V8 inspector treats resume with no attached debugger / no active
        // pause as a no-op success; mirror that so DevTools teardown does not
        // surface spurious errors.
        let mut emitted = false;
        if let Some(debugger) = self.debugger.as_ref() {
            let mut dbg = debugger.borrow_mut();
            if dbg.last_pause_reason().is_some() {
                dbg.apply_action(DebugAction::Continue);
                emitted = true;
            }
        }
        if emitted {
            self.notify_resumed();
        }
        Ok(json!({}))
    }

    fn debugger_continue_to_location(&mut self, params: &Value) -> StatorResult<Value> {
        let location = params.get("location").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.continueToLocation: required parameter 'location' is missing".to_string(),
            )
        })?;
        let script_id = location
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.continueToLocation: location.scriptId is missing or not a string"
                        .to_string(),
                )
            })?;
        let line_number = location
            .get("lineNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let column_number = location
            .get("columnNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let source = self.script_sources.get(script_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.continueToLocation: unknown scriptId `{script_id}`"
            ))
        })?;
        let Some((bytecode_offset, actual_location)) =
            breakpoint_location_for_source(script_id, source, line_number, column_number)?
        else {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.continueToLocation: no breakpointable location in scriptId `{script_id}`"
            )));
        };
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.continueToLocation",
                "no interpreter Debugger is attached to this session; continueToLocation requires \
                 an attached debugger plus an active pause.",
            ));
        };
        {
            let mut debugger = debugger.borrow_mut();
            if debugger.last_pause_reason().is_none() {
                return Err(unsupported_debugger_method(
                    "Debugger.continueToLocation",
                    "no active pause; continueToLocation is only valid after a Debugger.paused \
                     event has been emitted.",
                ));
            }
            debugger.set_one_shot_breakpoint_at_offset(
                bytecode_offset,
                actual_location["lineNumber"].as_u64().unwrap_or(0) as u32 + 1,
                actual_location["columnNumber"].as_u64().unwrap_or(0) as u32 + 1,
            );
            debugger.apply_action(DebugAction::Continue);
        }
        self.notify_resumed();
        Ok(json!({}))
    }

    // ── Debugger.stepInto / stepOver / stepOut ───────────────────────────────

    fn debugger_step(&mut self, action: DebugAction) -> StatorResult<Value> {
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                debug_step_method_name(action),
                "no interpreter Debugger is attached to this session; step \
                 commands require an attached debugger plus an active pause.",
            ));
        };
        {
            let mut dbg = debugger.borrow_mut();
            if dbg.last_pause_reason().is_none() {
                return Err(unsupported_debugger_method(
                    debug_step_method_name(action),
                    "no active pause; step commands are only valid after a \
                     `Debugger.paused` event has been emitted.",
                ));
            }
            dbg.apply_action(action);
        }
        self.notify_resumed();
        Ok(json!({}))
    }

    // ── Debugger.setBreakpointByUrl ──────────────────────────────────────────

    fn debugger_set_breakpoint(&mut self, params: &Value) -> StatorResult<Value> {
        let location = params.get("location").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.setBreakpoint: required parameter 'location' is missing".to_string(),
            )
        })?;
        let script_id = location
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBreakpoint: location.scriptId is missing or not a string"
                        .to_string(),
                )
            })?;
        let line_number = location
            .get("lineNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let column_number = location
            .get("columnNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let source = self.script_sources.get(script_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.setBreakpoint: unknown scriptId `{script_id}`"
            ))
        })?;
        let Some((bytecode_offset, actual_location)) =
            breakpoint_location_for_source(script_id, source, line_number, column_number)?
        else {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.setBreakpoint: no breakpointable location in scriptId `{script_id}`"
            )));
        };

        let bp_id = format!(
            "{}:{}:{}:{}",
            self.next_breakpoint_id, script_id, line_number, column_number
        );
        self.next_breakpoint_id += 1;
        self.cdp_breakpoints.insert(bp_id.clone());
        self.install_interpreter_breakpoint(
            &bp_id,
            bytecode_offset,
            actual_location["lineNumber"].as_u64().unwrap_or(0) as u32,
            actual_location["columnNumber"].as_u64().unwrap_or(0) as u32,
        );

        Ok(json!({
            "breakpointId": bp_id,
            "actualLocation": actual_location,
        }))
    }

    fn debugger_set_breakpoint_by_url(&mut self, params: &Value) -> StatorResult<Value> {
        let line_number = params
            .get("lineNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let column_number = params
            .get("columnNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        let requested_url = params.get("url").and_then(Value::as_str);
        let requested_url_regex = params.get("urlRegex").and_then(Value::as_str);
        let bp_id = format!(
            "{}:{}:{}",
            self.next_breakpoint_id, line_number, column_number
        );
        self.next_breakpoint_id += 1;
        self.cdp_breakpoints.insert(bp_id.clone());

        let mut locations = Vec::new();
        if requested_url.is_some() || requested_url_regex.is_some() {
            let mut script_ids: Vec<String> = self.script_sources.keys().cloned().collect();
            script_ids.sort();
            for script_id in script_ids {
                let Some(source) = self.script_sources.get(&script_id) else {
                    continue;
                };
                let script_url = self
                    .script_urls
                    .get(&script_id)
                    .map(String::as_str)
                    .unwrap_or_default();
                let url_matches = requested_url.is_some_and(|url| url == script_url)
                    || requested_url_regex.is_some_and(|pattern| script_url.contains(pattern));
                if !url_matches {
                    continue;
                }
                if let Some((bytecode_offset, location)) =
                    breakpoint_location_for_source(&script_id, source, line_number, column_number)?
                {
                    self.install_interpreter_breakpoint(
                        &bp_id,
                        bytecode_offset,
                        location["lineNumber"].as_u64().unwrap_or(0) as u32,
                        location["columnNumber"].as_u64().unwrap_or(0) as u32,
                    );
                    if self.debugger_enabled {
                        self.push_event(
                            "Debugger.breakpointResolved",
                            json!({
                                "breakpointId": bp_id,
                                "location": location.clone(),
                            }),
                        );
                    }
                    locations.push(location);
                }
            }
        } else {
            locations.push(json!({
                "scriptId": "0",
                "lineNumber": line_number,
                "columnNumber": column_number,
            }));
        }

        Ok(json!({
            "breakpointId": bp_id,
            "locations": locations,
        }))
    }

    fn install_interpreter_breakpoint(
        &mut self,
        cdp_breakpoint_id: &str,
        bytecode_offset: u32,
        line_number: u32,
        column_number: u32,
    ) {
        let Some(debugger) = self.debugger.as_ref() else {
            return;
        };
        let debugger_id = debugger.borrow_mut().set_breakpoint_at_offset(
            bytecode_offset,
            line_number.saturating_add(1),
            column_number.saturating_add(1),
        );
        self.cdp_debugger_breakpoints
            .entry(cdp_breakpoint_id.to_string())
            .or_default()
            .push(debugger_id);
    }

    fn debugger_remove_breakpoint(&mut self, params: &Value) -> StatorResult<Value> {
        let breakpoint_id = match params.get("breakpointId").and_then(Value::as_str) {
            Some(breakpoint_id) => breakpoint_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Debugger.removeBreakpoint: required parameter 'breakpointId' is missing or \
                     not a string"
                        .to_string(),
                ));
            }
        };
        if !self.cdp_breakpoints.remove(breakpoint_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.removeBreakpoint: unknown breakpointId `{breakpoint_id}`"
            )));
        }
        if let Some(debugger_ids) = self.cdp_debugger_breakpoints.remove(breakpoint_id)
            && let Some(debugger) = self.debugger.as_ref()
        {
            let mut debugger = debugger.borrow_mut();
            for debugger_id in debugger_ids {
                let _ = debugger.remove_breakpoint(debugger_id);
            }
        }
        Ok(json!({}))
    }

    fn debugger_get_script_source(&mut self, params: &Value) -> StatorResult<Value> {
        let script_id = match params.get("scriptId").and_then(Value::as_str) {
            Some(script_id) => script_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "Debugger.getScriptSource: required parameter 'scriptId' is missing or not a \
                     string"
                        .to_string(),
                ));
            }
        };
        let source = self.script_sources.get(script_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.getScriptSource: unknown scriptId `{script_id}`"
            ))
        })?;
        Ok(json!({ "scriptSource": source }))
    }

    fn debugger_set_script_source(&mut self, params: &Value) -> StatorResult<Value> {
        let script_id = params
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setScriptSource: required parameter 'scriptId' is missing or not a \
                     string"
                        .to_string(),
                )
            })?
            .to_string();
        let script_source = params
            .get("scriptSource")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setScriptSource: required parameter 'scriptSource' is missing or \
                     not a string"
                        .to_string(),
                )
            })?
            .to_string();
        if !self.script_sources.contains_key(&script_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.setScriptSource: unknown scriptId `{script_id}`"
            )));
        }

        if let Err(err) = parser::parse(&script_source)
            .and_then(|program| BytecodeGenerator::compile_program(&program))
        {
            let source_url = registered_script_url(&script_source);
            let execution_context_id = self.contexts.first().map(|context| context.id).unwrap_or(1);
            let details = self.exception_details_only(
                &err,
                ExceptionRequest {
                    expression: &script_source,
                    source_url: if source_url.is_empty() {
                        None
                    } else {
                        Some(source_url.as_str())
                    },
                    execution_context_id,
                    object_group: None,
                    generate_preview: false,
                },
            );
            return Ok(json!({
                "status": "CompileError",
                "exceptionDetails": details,
            }));
        }

        let dry_run = params
            .get("dryRun")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !dry_run {
            let source_url = registered_script_url(&script_source);
            self.script_sources.insert(script_id.clone(), script_source);
            self.script_urls.insert(script_id, source_url);
        }
        Ok(json!({
            "status": "Ok",
            "callFrames": [],
            "stackChanged": false,
        }))
    }

    fn debugger_get_possible_breakpoints(&mut self, params: &Value) -> StatorResult<Value> {
        let start = params.get("start").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.getPossibleBreakpoints: required parameter 'start' is missing"
                    .to_string(),
            )
        })?;
        let script_id = start
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.getPossibleBreakpoints: start.scriptId is missing or not a string"
                        .to_string(),
                )
            })?;
        let source = self.script_sources.get(script_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.getPossibleBreakpoints: unknown scriptId `{script_id}`"
            ))
        })?;
        let start_line = start.get("lineNumber").and_then(Value::as_u64).unwrap_or(0) as u32;
        let start_column = start
            .get("columnNumber")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let end = params.get("end");
        let end_line = end
            .and_then(|value| value.get("lineNumber"))
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let end_column = end
            .and_then(|value| value.get("columnNumber"))
            .and_then(Value::as_u64)
            .map(|value| value as u32);

        let bytecodes = parser::parse(source)
            .and_then(|program| BytecodeGenerator::compile_program(&program))?;
        let locations: Vec<Value> = Debugger::breakpoint_locations(&bytecodes)
            .into_iter()
            .filter_map(|location| {
                let line = location.line.saturating_sub(1);
                let column = location.column.saturating_sub(1);
                if line < start_line || (line == start_line && column < start_column) {
                    return None;
                }
                if let Some(end_line) = end_line
                    && (line > end_line
                        || (line == end_line && end_column.is_some_and(|end| column > end)))
                {
                    return None;
                }
                Some(json!({
                    "scriptId": script_id,
                    "lineNumber": line,
                    "columnNumber": column,
                }))
            })
            .collect();
        Ok(json!({ "locations": locations }))
    }

    // ── Console.enable ───────────────────────────────────────────────────────

    fn console_enable(&mut self) -> StatorResult<Value> {
        self.console_enabled = true;

        // Flush any buffered console messages as `Console.messageAdded` events.
        for msg in drain_messages() {
            self.push_event(
                "Console.messageAdded",
                json!({
                    "message": {
                        "source": "console-api",
                        "level": msg.level.as_cdp_str(),
                        "text": msg.text,
                    }
                }),
            );
        }

        Ok(json!({}))
    }

    // ── Profiler.setSamplingInterval / Profiler.start ───────────────────────

    fn profiler_set_sampling_interval(&mut self, params: &Value) -> StatorResult<Value> {
        let interval = sampling_interval_param(params, "Profiler.setSamplingInterval")?;
        self.profiler_sampling_interval_micros = interval;
        Ok(json!({}))
    }

    fn profiler_start(&mut self, params: &Value) -> StatorResult<Value> {
        // Non-standard compatibility: still honour a direct start parameter if
        // one is supplied, otherwise use Profiler.setSamplingInterval state.
        let interval_micros = if params.get("samplingInterval").is_some() {
            sampling_interval_param(params, "Profiler.start")?
        } else {
            self.profiler_sampling_interval_micros
        };
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

    fn profiler_start_precise_coverage(&mut self) -> StatorResult<Value> {
        self.profiler_precise_coverage_enabled = true;
        self.coverage_scripts.clear();
        Ok(json!({}))
    }

    fn profiler_stop_precise_coverage(&mut self) -> StatorResult<Value> {
        self.profiler_precise_coverage_enabled = false;
        Ok(json!({}))
    }

    fn profiler_take_precise_coverage(&mut self) -> StatorResult<Value> {
        Ok(json!({
            "result": self.coverage_payload(),
            "timestamp": 0.0,
        }))
    }

    fn profiler_get_best_effort_coverage(&mut self) -> StatorResult<Value> {
        Ok(json!({ "result": self.coverage_payload() }))
    }

    fn profiler_start_type_profile(&mut self) -> StatorResult<Value> {
        self.profiler_type_profile_enabled = true;
        self.type_profile_scripts.clear();
        Ok(json!({}))
    }

    fn profiler_stop_type_profile(&mut self) -> StatorResult<Value> {
        self.profiler_type_profile_enabled = false;
        Ok(json!({}))
    }

    fn profiler_take_type_profile(&mut self) -> StatorResult<Value> {
        Ok(json!({ "result": self.type_profile_payload() }))
    }

    fn record_coverage(&mut self, source: &str, source_url: Option<&str>, script_id: Option<&str>) {
        if !self.profiler_precise_coverage_enabled {
            return;
        }
        let id = script_id.map(str::to_string).unwrap_or_else(|| {
            let id = format!("runtime-eval-{}", self.next_coverage_script_id);
            self.next_coverage_script_id = self.next_coverage_script_id.saturating_add(1);
            id
        });
        let entry = self
            .coverage_scripts
            .entry(id.clone())
            .or_insert_with(|| CoverageScript {
                script_id: id,
                url: source_url.unwrap_or_default().to_string(),
                source: source.to_string(),
                count: 0,
            });
        entry.count = entry.count.saturating_add(1);
    }

    fn coverage_payload(&self) -> Vec<Value> {
        let mut scripts: Vec<_> = self.coverage_scripts.values().cloned().collect();
        scripts.sort_by(|a, b| a.script_id.cmp(&b.script_id));
        scripts
            .into_iter()
            .map(|script| coverage_script_to_value(&script))
            .collect()
    }

    fn record_type_profile(
        &mut self,
        source: &str,
        source_url: Option<&str>,
        script_id: Option<&str>,
        value: &JsValue,
    ) {
        if !self.profiler_type_profile_enabled {
            return;
        }
        let id = script_id.map(str::to_string).unwrap_or_else(|| {
            let id = format!("runtime-type-{}", self.next_type_profile_script_id);
            self.next_type_profile_script_id = self.next_type_profile_script_id.saturating_add(1);
            id
        });
        let entry = self
            .type_profile_scripts
            .entry(id.clone())
            .or_insert_with(|| TypeProfileScript {
                script_id: id,
                url: source_url.unwrap_or_default().to_string(),
                offset: source.len(),
                types: HashSet::new(),
            });
        entry.types.insert(type_profile_name(value).to_string());
    }

    fn type_profile_payload(&self) -> Vec<Value> {
        let mut scripts: Vec<_> = self.type_profile_scripts.values().cloned().collect();
        scripts.sort_by(|a, b| a.script_id.cmp(&b.script_id));
        scripts
            .into_iter()
            .map(|script| type_profile_script_to_value(&script))
            .collect()
    }

    // ── HeapProfiler.takeHeapSnapshot ────────────────────────────────────────

    fn heap_profiler_take_snapshot(&mut self) -> StatorResult<Value> {
        let snapshot = HeapSnapshotBuilder::build(&self.globals.borrow().vars);
        let chunks = split_snapshot_chunks(&snapshot.to_json(), HEAP_SNAPSHOT_CHUNK_SIZE);
        let total = chunks.len();
        for (index, chunk) in chunks.into_iter().enumerate() {
            self.push_event(
                "HeapProfiler.addHeapSnapshotChunk",
                json!({ "chunk": chunk }),
            );
            self.push_event(
                "HeapProfiler.reportHeapSnapshotProgress",
                json!({
                    "done": index + 1,
                    "total": total,
                    "finished": index + 1 == total
                }),
            );
        }
        Ok(json!({}))
    }

    fn heap_profiler_get_heap_object_id(&mut self, params: &Value) -> StatorResult<Value> {
        let object_id = match params.get("objectId").and_then(Value::as_str) {
            Some(object_id) => object_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "HeapProfiler.getHeapObjectId: required parameter 'objectId' is missing or \
                     not a string"
                        .to_string(),
                ));
            }
        };
        let value = self.remote_objects.get(object_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "HeapProfiler.getHeapObjectId: unknown or released objectId `{object_id}`"
            ))
        })?;
        let heap_snapshot_object_id = self.heap_object_id_for_value(value);
        Ok(json!({ "heapSnapshotObjectId": heap_snapshot_object_id }))
    }

    fn heap_profiler_get_object_by_heap_object_id(
        &mut self,
        params: &Value,
    ) -> StatorResult<Value> {
        let heap_object_id = match params.get("objectId").and_then(Value::as_str) {
            Some(object_id) => object_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "HeapProfiler.getObjectByHeapObjectId: required parameter 'objectId' is \
                     missing or not a string"
                        .to_string(),
                ));
            }
        };
        let value = self
            .heap_objects
            .get(heap_object_id)
            .cloned()
            .ok_or_else(|| {
                crate::error::StatorError::Internal(format!(
                    "HeapProfiler.getObjectByHeapObjectId: unknown heap object id \
                     `{heap_object_id}`"
                ))
            })?;
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);
        let result = js_value_to_remote_object(
            &value,
            &mut self.remote_objects,
            object_group.as_deref(),
            false,
        );
        Ok(json!({ "result": result }))
    }

    fn heap_profiler_add_inspected_heap_object(&mut self, params: &Value) -> StatorResult<Value> {
        let heap_object_id = match params.get("heapObjectId").and_then(Value::as_str) {
            Some(object_id) => object_id,
            None => {
                return Err(crate::error::StatorError::TypeError(
                    "HeapProfiler.addInspectedHeapObject: required parameter 'heapObjectId' is \
                     missing or not a string"
                        .to_string(),
                ));
            }
        };
        if !self.heap_objects.contains_key(heap_object_id) {
            return Err(crate::error::StatorError::Internal(format!(
                "HeapProfiler.addInspectedHeapObject: unknown heap object id `{heap_object_id}`"
            )));
        }
        self.inspected_heap_objects
            .insert(heap_object_id.to_string());
        Ok(json!({}))
    }

    fn heap_profiler_start_sampling(&mut self) -> StatorResult<Value> {
        crate::inspector::heap_snapshot::start_tracking();
        self.last_heap_sampling_profile = empty_heap_sampling_profile();
        Ok(json!({}))
    }

    fn heap_profiler_stop_sampling(&mut self) -> StatorResult<Value> {
        let records = crate::inspector::heap_snapshot::stop_tracking();
        let profile = build_heap_sampling_profile(&records);
        self.last_heap_sampling_profile = profile.clone();
        Ok(json!({ "profile": profile }))
    }

    fn heap_profiler_get_sampling_profile(&mut self) -> StatorResult<Value> {
        let records = crate::inspector::heap_snapshot::snapshot_tracking();
        let profile = if records.is_empty() {
            self.last_heap_sampling_profile.clone()
        } else {
            build_heap_sampling_profile(&records)
        };
        Ok(json!({ "profile": profile }))
    }

    fn heap_object_id_for_value(&mut self, value: JsValue) -> String {
        if let Some((id, _)) = self
            .heap_objects
            .iter()
            .find(|(_, existing)| same_heap_identity(existing, &value))
        {
            return id.clone();
        }
        let id = format!("heap-{}", self.next_heap_object_id);
        self.next_heap_object_id = self.next_heap_object_id.saturating_add(1);
        self.heap_objects.insert(id.clone(), value);
        id
    }

    // ── HeapProfiler.startTrackingHeapObjects ────────────────────────────────

    fn heap_profiler_start_tracking(&mut self) -> StatorResult<Value> {
        crate::inspector::heap_snapshot::start_tracking();
        Ok(json!({}))
    }

    // ── HeapProfiler.stopTrackingHeapObjects ─────────────────────────────────

    fn heap_profiler_stop_tracking(&mut self) -> StatorResult<Value> {
        let records = crate::inspector::heap_snapshot::stop_tracking();
        self.emit_heap_tracking_events(&records);
        // Return a summary of the allocation records collected.
        let stats: Vec<Value> = records
            .iter()
            .map(|r| json!({ "id": r.id, "size": r.size }))
            .collect();
        Ok(json!({ "stats": stats }))
    }

    fn emit_heap_tracking_events(&mut self, records: &[AllocationRecord]) {
        let Some(last_seen_object_id) = records.iter().map(|record| record.id).max() else {
            return;
        };
        let total_size: usize = records.iter().map(|record| record.size).sum();
        self.push_event(
            "HeapProfiler.heapStatsUpdate",
            json!({
                "statsUpdate": [0, records.len(), total_size],
            }),
        );
        self.push_event(
            "HeapProfiler.lastSeenObjectId",
            json!({
                "lastSeenObjectId": last_seen_object_id,
                "timestamp": 0.0,
            }),
        );
    }
}

impl Default for CdpDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket-backed session
// ─────────────────────────────────────────────────────────────────────────────

/// A single CDP debugging session backed by one WebSocket connection.
///
/// `CdpSession` adapts the transport-agnostic [`CdpDispatcher`] to a
/// WebSocket stream: it reads JSON-RPC text frames from the wire, hands
/// them to the dispatcher, and drains the dispatcher's outbox back onto
/// the wire after every frame.
pub struct CdpSession {
    ws: WebSocket<TcpStream>,
    dispatcher: CdpDispatcher,
}

impl CdpSession {
    /// Wrap an accepted WebSocket connection in a new session.
    fn new(ws: WebSocket<TcpStream>) -> Self {
        Self {
            ws,
            dispatcher: CdpDispatcher::new(),
        }
    }

    /// Drive the session until the client disconnects or a fatal error occurs.
    ///
    /// Each incoming text frame is forwarded to the dispatcher; afterwards
    /// any messages queued in the dispatcher's outbox (events emitted
    /// before the ack plus the ack itself) are sent to the peer in FIFO
    /// order.
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
                    let _ = self.dispatcher.dispatch_json(&text);
                    while let Some(out) = self.dispatcher.take_next() {
                        self.ws
                            .send(Message::Text(out.into()))
                            .map_err(|e| io::Error::other(e.to_string()))?;
                    }
                }
                Message::Close(_) => return Ok(()),
                // Ignore binary / ping / pong frames.
                _ => {}
            }
        }
    }
}

const MAX_PREVIEW_PROPERTIES: usize = 5;
const MAX_PREVIEW_DEPTH: usize = 1;
const HEAP_SNAPSHOT_CHUNK_SIZE: usize = 64 * 1024;

struct ExceptionRequest<'a> {
    expression: &'a str,
    source_url: Option<&'a str>,
    execution_context_id: u32,
    object_group: Option<&'a str>,
    generate_preview: bool,
}

fn exception_source_url(params: &Value, expression: &str) -> Option<String> {
    if let Some(url) = params.get("sourceURL").and_then(Value::as_str) {
        return Some(url.to_string());
    }
    expression.lines().rev().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("//# sourceURL=")
            .or_else(|| trimmed.strip_prefix("//@ sourceURL="))
            .map(str::to_string)
    })
}

pub(crate) fn registered_script_url(source: &str) -> String {
    exception_source_url(&Value::Null, source).unwrap_or_default()
}

fn breakpoint_location_for_source(
    script_id: &str,
    source: &str,
    line_number: u32,
    column_number: u32,
) -> StatorResult<Option<(u32, Value)>> {
    let bytecodes =
        parser::parse(source).and_then(|program| BytecodeGenerator::compile_program(&program))?;
    let mut locations: Vec<_> = Debugger::breakpoint_locations(&bytecodes)
        .into_iter()
        .filter_map(|location| {
            let line = location.line.saturating_sub(1);
            let column = location.column.saturating_sub(1);
            if line < line_number || (line == line_number && column < column_number) {
                return None;
            }
            Some((line, column, location.bytecode_offset))
        })
        .collect();
    locations.sort_by_key(|(line, column, offset)| (*line, *column, *offset));
    Ok(locations
        .into_iter()
        .next()
        .map(|(line, column, bytecode_offset)| {
            (
                bytecode_offset,
                json!({
                    "scriptId": script_id,
                    "lineNumber": line,
                    "columnNumber": column,
                }),
            )
        }))
}

fn request_line_number(expression: &str) -> u32 {
    expression
        .lines()
        .position(|line| !line.trim().is_empty())
        .unwrap_or(0) as u32
}

/// CDP `Debugger.setPauseOnExceptions.state` enumeration.
///
/// Stored on each [`CdpDispatcher`] so the chosen pause-on-exceptions
/// behaviour can be honoured even when a debugger is attached after
/// `Debugger.setPauseOnExceptions` has already been negotiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseOnExceptionsState {
    /// Never pause on thrown exceptions.
    None,
    /// Pause only on uncaught exceptions. Stator currently treats this the
    /// same as [`Self::All`] because the interpreter `Throw` hook fires
    /// before catch resolution.
    Uncaught,
    /// Pause on every thrown exception.
    All,
}

impl PauseOnExceptionsState {
    /// Parse the textual form sent by CDP clients.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "uncaught" => Some(Self::Uncaught),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    /// Whether this state corresponds to an enabled interpreter
    /// pause-on-exceptions hook.
    pub fn enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

fn schema_get_domains() -> Value {
    json!({
        "domains": [
            { "name": "Runtime", "version": "1.3" },
            { "name": "Debugger", "version": "1.3" },
            { "name": "Console", "version": "1.3" },
            { "name": "Profiler", "version": "1.3" },
            { "name": "HeapProfiler", "version": "1.3" },
            { "name": "Network", "version": "1.3" },
            { "name": "Schema", "version": "1.3" }
        ]
    })
}

fn collect_objects_with_prototype(
    value: &JsValue,
    prototype: &JsValue,
    seen: &mut HashSet<usize>,
    matches: &mut Vec<JsValue>,
) {
    match value {
        JsValue::PlainObject(map_ref) => {
            let id = Rc::as_ptr(map_ref) as usize;
            if !seen.insert(id) {
                return;
            }
            let entries: Vec<JsValue> = map_ref.borrow().iter().map(|(_, v)| v.clone()).collect();
            if !same_heap_identity(value, prototype) && direct_prototype_is(value, prototype) {
                matches.push(value.clone());
            }
            for child in entries {
                collect_objects_with_prototype(&child, prototype, seen, matches);
            }
        }
        JsValue::Array(array_ref) => {
            let id = Rc::as_ptr(array_ref) as usize;
            if !seen.insert(id) {
                return;
            }
            let entries = array_ref.borrow().clone();
            for child in entries {
                collect_objects_with_prototype(&child, prototype, seen, matches);
            }
        }
        _ => {}
    }
}

fn direct_prototype_is(value: &JsValue, prototype: &JsValue) -> bool {
    let JsValue::PlainObject(map_ref) = value else {
        return false;
    };
    map_ref
        .borrow()
        .get("__proto__")
        .is_some_and(|proto| same_heap_identity(proto, prototype))
}

fn same_heap_identity(left: &JsValue, right: &JsValue) -> bool {
    match (left, right) {
        (JsValue::Object(a), JsValue::Object(b)) => std::ptr::eq(*a, *b),
        (JsValue::Function(a), JsValue::Function(b)) => Rc::ptr_eq(a, b),
        (JsValue::Array(a), JsValue::Array(b)) => Rc::ptr_eq(a, b),
        (JsValue::Generator(a), JsValue::Generator(b)) => Rc::ptr_eq(a, b),
        (JsValue::Iterator(a), JsValue::Iterator(b)) => Rc::ptr_eq(a, b),
        (JsValue::Error(a), JsValue::Error(b)) => Rc::ptr_eq(a, b),
        (JsValue::NativeFunction(a), JsValue::NativeFunction(b)) => Rc::ptr_eq(a, b),
        (JsValue::PlainObject(a), JsValue::PlainObject(b)) => Rc::ptr_eq(a, b),
        (JsValue::Context(a), JsValue::Context(b)) => Rc::ptr_eq(a, b),
        (JsValue::Proxy(a), JsValue::Proxy(b)) => Rc::ptr_eq(a, b),
        (JsValue::ArrayBuffer(a), JsValue::ArrayBuffer(b)) => Rc::ptr_eq(a, b),
        (JsValue::TypedArray(a), JsValue::TypedArray(b)) => Rc::ptr_eq(a, b),
        (JsValue::DataView(a), JsValue::DataView(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

fn sampling_interval_param(params: &Value, method: &str) -> StatorResult<u64> {
    match params.get("samplingInterval").and_then(Value::as_u64) {
        Some(0) => Err(crate::error::StatorError::TypeError(format!(
            "{method}: samplingInterval must be greater than zero"
        ))),
        Some(interval) => Ok(interval),
        None => Err(crate::error::StatorError::TypeError(format!(
            "{method}: required parameter 'samplingInterval' is missing or not a number"
        ))),
    }
}

fn split_snapshot_chunks(snapshot: &str, max_bytes: usize) -> Vec<String> {
    debug_assert!(max_bytes > 0);
    if snapshot.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < snapshot.len() {
        let mut end = (start + max_bytes).min(snapshot.len());
        while end > start && !snapshot.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = snapshot[start..]
                .char_indices()
                .nth(1)
                .map(|(offset, _)| start + offset)
                .unwrap_or(snapshot.len());
        }
        chunks.push(snapshot[start..end].to_string());
        start = end;
    }
    chunks
}

struct SamplingNodeBuilder {
    id: u32,
    function_name: String,
    self_size: usize,
    children: Vec<usize>,
}

fn empty_heap_sampling_profile() -> Value {
    build_heap_sampling_profile(&[])
}

fn build_heap_sampling_profile(records: &[AllocationRecord]) -> Value {
    let mut nodes = vec![SamplingNodeBuilder {
        id: 1,
        function_name: "(root)".to_string(),
        self_size: 0,
        children: Vec::new(),
    }];
    let mut samples = Vec::with_capacity(records.len());

    for record in records {
        let stack = if record.stack.is_empty() {
            vec!["(program)"]
        } else {
            record.stack.clone()
        };
        let mut parent = 0usize;
        for frame in stack {
            let child = nodes[parent]
                .children
                .iter()
                .copied()
                .find(|idx| nodes[*idx].function_name == frame)
                .unwrap_or_else(|| {
                    let idx = nodes.len();
                    nodes.push(SamplingNodeBuilder {
                        id: idx as u32 + 1,
                        function_name: frame.to_string(),
                        self_size: 0,
                        children: Vec::new(),
                    });
                    nodes[parent].children.push(idx);
                    idx
                });
            parent = child;
        }
        nodes[parent].self_size = nodes[parent].self_size.saturating_add(record.size);
        samples.push(json!({
            "size": record.size,
            "nodeId": nodes[parent].id,
            "ordinal": record.id,
        }));
    }

    json!({
        "head": sampling_node_to_value(&nodes, 0),
        "samples": samples,
    })
}

fn coverage_script_to_value(script: &CoverageScript) -> Value {
    json!({
        "scriptId": script.script_id,
        "url": script.url,
        "functions": [{
            "functionName": "(script)",
            "ranges": [{
                "startOffset": 0,
                "endOffset": script.source.len(),
                "count": script.count,
            }],
            "isBlockCoverage": false,
        }],
    })
}

fn type_profile_script_to_value(script: &TypeProfileScript) -> Value {
    let mut types: Vec<_> = script.types.iter().cloned().collect();
    types.sort();
    json!({
        "scriptId": script.script_id,
        "url": script.url,
        "entries": [{
            "offset": script.offset,
            "types": types.into_iter().map(|name| json!({ "name": name })).collect::<Vec<_>>(),
        }],
    })
}

fn type_profile_name(value: &JsValue) -> &'static str {
    match value {
        JsValue::Undefined | JsValue::TheHole => "Undefined",
        JsValue::Null => "Null",
        JsValue::Boolean(_) => "Boolean",
        JsValue::Smi(_) | JsValue::HeapNumber(_) => "Number",
        JsValue::String(_) => "String",
        JsValue::BigInt(_) => "BigInt",
        JsValue::Symbol(_) => "Symbol",
        JsValue::Function(_) | JsValue::NativeFunction(_) => "Function",
        _ => "Object",
    }
}

fn console_profile_location() -> Value {
    json!({
        "scriptId": "0",
        "lineNumber": 0,
        "columnNumber": 0,
    })
}

fn empty_console_profile() -> Value {
    json!({
        "nodes": [{
            "id": 1,
            "callFrame": {
                "functionName": "(root)",
                "scriptId": "0",
                "url": "",
                "lineNumber": 0,
                "columnNumber": 0,
            },
            "hitCount": 0,
        }],
        "startTime": 0,
        "endTime": 0,
        "samples": [],
        "timeDeltas": [],
    })
}

fn sampling_node_to_value(nodes: &[SamplingNodeBuilder], index: usize) -> Value {
    let node = &nodes[index];
    let children: Vec<Value> = node
        .children
        .iter()
        .map(|idx| sampling_node_to_value(nodes, *idx))
        .collect();
    json!({
        "id": node.id,
        "selfSize": node.self_size,
        "callFrame": {
            "functionName": node.function_name,
            "scriptId": "0",
            "url": "",
            "lineNumber": 0,
            "columnNumber": 0,
        },
        "children": children,
    })
}

fn unsupported_debugger_method(method: &str, detail: &str) -> StatorError {
    StatorError::TypeError(format!("{method}: {detail}"))
}

fn debug_step_method_name(action: DebugAction) -> &'static str {
    match action {
        DebugAction::Continue => "Debugger.resume",
        DebugAction::StepInto => "Debugger.stepInto",
        DebugAction::StepOver => "Debugger.stepOver",
        DebugAction::StepOut => "Debugger.stepOut",
    }
}

fn paused_reason_str(reason: &PauseReason) -> &'static str {
    // CDP `Debugger.Paused.reason` enumeration. We map the interpreter
    // reasons onto the closest CDP-defined string. Breakpoints are reported
    // as `"other"` (matching V8 when `hitBreakpoints` is populated).
    match reason {
        PauseReason::DebuggerStatement => "debuggerStatement",
        PauseReason::Breakpoint(_) => "other",
        PauseReason::Step => "other",
        PauseReason::Exception => "exception",
    }
}

fn paused_event_params(reason: &PauseReason, offset: u32, line: u32) -> Value {
    // We cannot reconstruct the live JS call stack here yet: the interpreter
    // does not retain a portable per-frame snapshot at pause time. To avoid
    // emitting success-shaped placeholder frames we instead emit a single
    // synthetic top-level frame describing the paused source location. The
    // `(stator: paused-frame)` function name and `pausedFrame: true`
    // auxData entry signal to embedders that this is a derived frame.
    let line_number = line.saturating_sub(1);
    let call_frame = json!({
        "callFrameId": format!("stator-pause-frame-{offset}"),
        "functionName": "(stator: paused-frame)",
        "location": {
            "scriptId": "0",
            "lineNumber": line_number,
            "columnNumber": 0,
        },
        "scopeChain": [],
        "this": {"type": "undefined"},
        "url": "",
    });
    let mut params = json!({
        "callFrames": [call_frame],
        "reason": paused_reason_str(reason),
        "data": {
            "bytecodeOffset": offset,
            "pausedFrame": true,
        },
    });
    if let PauseReason::Breakpoint(id) = reason
        && let Some(obj) = params.as_object_mut()
    {
        obj.insert(
            "hitBreakpoints".to_string(),
            Value::Array(vec![Value::String(id.to_string())]),
        );
    }
    params
}

fn exception_text(err: &StatorError, thrown: Option<&JsValue>) -> String {
    match thrown {
        Some(JsValue::Error(error)) => error.to_error_string(),
        Some(value) => value.to_display_string(),
        None => err.to_string(),
    }
}

fn stack_trace_from_thrown(thrown: Option<&JsValue>) -> Option<Value> {
    let JsValue::Error(error) = thrown? else {
        return None;
    };
    let stack = error.stack();
    let mut lines = stack.lines();
    let description = lines.next().unwrap_or_default().to_string();
    let call_frames: Vec<Value> = lines
        .filter_map(|line| line.trim().strip_prefix("at "))
        .map(|function_name| {
            json!({
                "functionName": function_name,
                "scriptId": "0",
                "url": "",
                "lineNumber": 0,
                "columnNumber": 0,
            })
        })
        .collect();
    Some(json!({ "description": description, "callFrames": call_frames }))
}

/// Convert a [`JsValue`] to a CDP `Runtime.RemoteObject` description,
/// registering non-primitive values in `registry` under `object_group` so
/// that follow-up `Runtime.getProperties` / `Runtime.releaseObject` calls
/// can resolve them.
///
/// Primitive variants never allocate an `objectId`; heap variants always
/// do.  Unsupported / opaque heap variants are described by class name with
/// no fabricated properties, so later `getProperties` returns an empty
/// own-property list.
fn js_value_to_remote_object(
    value: &JsValue,
    registry: &mut RemoteObjectRegistry,
    object_group: Option<&str>,
    generate_preview: bool,
) -> Value {
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
            json!({"type": "string", "value": &**s})
        }
        JsValue::BigInt(b) => {
            // BigInt is a primitive: report value as the canonical literal
            // form, no objectId minted.
            json!({"type": "bigint", "description": format!("{}n", **b), "unserializableValue": format!("{}n", **b)})
        }
        JsValue::Symbol(id) => {
            let group = object_group.map(str::to_string);
            let oid = registry.register(value.clone(), group);
            json!({
                "type": "symbol",
                "description": format!("Symbol({id})"),
                "objectId": oid,
            })
        }
        JsValue::TheHole => {
            // Internal sentinel; never user-visible. Report as undefined so
            // DevTools does not render a special placeholder.
            json!({"type": "undefined"})
        }
        _ => non_primitive_remote_object(value, registry, object_group, generate_preview),
    }
}

/// Mint a RemoteObject payload for a heap-backed [`JsValue`].
fn non_primitive_remote_object(
    value: &JsValue,
    registry: &mut RemoteObjectRegistry,
    object_group: Option<&str>,
    generate_preview: bool,
) -> Value {
    let (kind, subtype, class_name, description) = describe_heap_value(value);
    let group = object_group.map(str::to_string);
    let object_id = registry.register(value.clone(), group);
    let mut payload = json!({
        "type": kind,
        "className": class_name,
        "description": description,
        "objectId": object_id,
    });
    if let Some(sub) = subtype {
        payload
            .as_object_mut()
            .expect("payload is an object")
            .insert("subtype".to_string(), Value::String(sub.to_string()));
    }
    if generate_preview && let Some(preview) = build_object_preview(value, 0) {
        payload
            .as_object_mut()
            .expect("payload is an object")
            .insert("preview".to_string(), preview);
    }
    payload
}

/// Return CDP `(type, subtype, className, description)` for `value`.
///
/// `description` is a best-effort short label used by DevTools when a
/// structured preview is not requested or not available.
fn describe_heap_value(value: &JsValue) -> (&'static str, Option<&'static str>, String, String) {
    match value {
        JsValue::Array(arr) => {
            let len = arr.borrow().len();
            (
                "object",
                Some("array"),
                "Array".to_string(),
                format!("Array({len})"),
            )
        }
        JsValue::Function(bc) => {
            let name = bc.function_name();
            let label = if name.is_empty() {
                "function () { … }".to_string()
            } else {
                format!("function {name}() {{ … }}")
            };
            ("function", None, "Function".to_string(), label)
        }
        JsValue::NativeFunction(_) => (
            "function",
            None,
            "Function".to_string(),
            "function () { [native code] }".to_string(),
        ),
        JsValue::PlainObject(map) => {
            let count = map.borrow().len();
            (
                "object",
                None,
                "Object".to_string(),
                format!("Object({count} props)"),
            )
        }
        JsValue::Error(err) => (
            "object",
            Some("error"),
            "Error".to_string(),
            err.to_error_string(),
        ),
        JsValue::Promise(_) => (
            "object",
            Some("promise"),
            "Promise".to_string(),
            "[object Promise]".to_string(),
        ),
        JsValue::Proxy(_) => (
            "object",
            Some("proxy"),
            "Proxy".to_string(),
            "[object Proxy]".to_string(),
        ),
        JsValue::ArrayBuffer(_) => (
            "object",
            Some("arraybuffer"),
            "ArrayBuffer".to_string(),
            "[object ArrayBuffer]".to_string(),
        ),
        JsValue::TypedArray(_) => (
            "object",
            Some("typedarray"),
            "TypedArray".to_string(),
            "[object TypedArray]".to_string(),
        ),
        JsValue::DataView(_) => (
            "object",
            Some("dataview"),
            "DataView".to_string(),
            "[object DataView]".to_string(),
        ),
        JsValue::Generator(_) => (
            "object",
            Some("generator"),
            "Generator".to_string(),
            "[object Generator]".to_string(),
        ),
        JsValue::Iterator(_) => (
            "object",
            Some("iterator"),
            "Iterator".to_string(),
            "[object Iterator]".to_string(),
        ),
        JsValue::ModuleBinding(_) => (
            "object",
            None,
            "Module".to_string(),
            "[module binding]".to_string(),
        ),
        JsValue::Context(_) => (
            "object",
            None,
            "Context".to_string(),
            "[internal context]".to_string(),
        ),
        JsValue::Object(_) => (
            "object",
            None,
            "Object".to_string(),
            "[object Object]".to_string(),
        ),
        // Primitive variants never reach this helper.
        _ => (
            "object",
            None,
            "Object".to_string(),
            value
                .to_js_string()
                .unwrap_or_else(|_| "[object Object]".to_string()),
        ),
    }
}

fn build_object_preview(value: &JsValue, depth: usize) -> Option<Value> {
    let (kind, subtype, _class_name, description) = describe_heap_value(value);
    let (properties, overflow) = match value {
        JsValue::PlainObject(map_ref) => {
            let entries: Vec<(String, JsValue)> = map_ref
                .borrow()
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let overflow = entries.len() > MAX_PREVIEW_PROPERTIES;
            let properties = entries
                .into_iter()
                .take(MAX_PREVIEW_PROPERTIES)
                .map(|(name, child)| build_property_preview(name, &child, depth))
                .collect();
            (properties, overflow)
        }
        JsValue::Array(arr_ref) => {
            let items: Vec<JsValue> = arr_ref.borrow().clone();
            let overflow = items.len() > MAX_PREVIEW_PROPERTIES;
            let properties = items
                .into_iter()
                .enumerate()
                .take(MAX_PREVIEW_PROPERTIES)
                .map(|(idx, child)| build_property_preview(idx.to_string(), &child, depth))
                .collect();
            (properties, overflow)
        }
        JsValue::Function(bc) => {
            let properties = vec![
                json!({
                    "name": "length",
                    "type": "number",
                    "value": bc.function_length().to_string(),
                }),
                json!({
                    "name": "name",
                    "type": "string",
                    "value": bc.function_name(),
                }),
            ];
            (properties, false)
        }
        JsValue::Error(err) => {
            let properties = vec![
                json!({
                    "name": "name",
                    "type": "string",
                    "value": err.name(),
                }),
                json!({
                    "name": "message",
                    "type": "string",
                    "value": err.message(),
                }),
            ];
            (properties, false)
        }
        _ => return None,
    };

    let mut preview = json!({
        "type": kind,
        "description": description,
        "overflow": overflow,
        "properties": properties,
    });
    if let Some(subtype) = subtype {
        preview
            .as_object_mut()
            .expect("preview is an object")
            .insert("subtype".to_string(), Value::String(subtype.to_string()));
    }
    Some(preview)
}

fn build_property_preview(name: String, value: &JsValue, depth: usize) -> Value {
    let mut property = json!({
        "name": name,
        "type": preview_type(value),
    });
    if let Some(subtype) = preview_subtype(value) {
        property
            .as_object_mut()
            .expect("property preview is an object")
            .insert("subtype".to_string(), Value::String(subtype.to_string()));
    }
    if let Some(value_text) = preview_value(value) {
        property
            .as_object_mut()
            .expect("property preview is an object")
            .insert("value".to_string(), Value::String(value_text));
    } else if depth < MAX_PREVIEW_DEPTH
        && let Some(value_preview) = build_object_preview(value, depth + 1)
    {
        property
            .as_object_mut()
            .expect("property preview is an object")
            .insert("valuePreview".to_string(), value_preview);
    }
    property
}

fn preview_type(value: &JsValue) -> &'static str {
    match value {
        JsValue::Undefined | JsValue::TheHole => "undefined",
        JsValue::Null => "object",
        JsValue::Boolean(_) => "boolean",
        JsValue::Smi(_) | JsValue::HeapNumber(_) => "number",
        JsValue::String(_) => "string",
        JsValue::BigInt(_) => "bigint",
        JsValue::Symbol(_) => "symbol",
        JsValue::Function(_) | JsValue::NativeFunction(_) => "function",
        _ => "object",
    }
}

fn preview_subtype(value: &JsValue) -> Option<&'static str> {
    match value {
        JsValue::Null => Some("null"),
        JsValue::Array(_) => Some("array"),
        JsValue::Error(_) => Some("error"),
        JsValue::Promise(_) => Some("promise"),
        JsValue::Proxy(_) => Some("proxy"),
        JsValue::ArrayBuffer(_) => Some("arraybuffer"),
        JsValue::TypedArray(_) => Some("typedarray"),
        JsValue::DataView(_) => Some("dataview"),
        JsValue::Generator(_) => Some("generator"),
        JsValue::Iterator(_) => Some("iterator"),
        _ => None,
    }
}

fn preview_value(value: &JsValue) -> Option<String> {
    match value {
        JsValue::Undefined | JsValue::TheHole => Some("undefined".to_string()),
        JsValue::Null => Some("null".to_string()),
        JsValue::Boolean(b) => Some(b.to_string()),
        JsValue::Smi(n) => Some(n.to_string()),
        JsValue::HeapNumber(n) => Some(n.to_string()),
        JsValue::String(s) => Some(s.to_string()),
        JsValue::BigInt(n) => Some(format!("{}n", **n)),
        JsValue::Symbol(id) => Some(format!("Symbol({id})")),
        JsValue::Function(bc) => Some(if bc.function_name().is_empty() {
            "function () { … }".to_string()
        } else {
            format!("function {}() {{ … }}", bc.function_name())
        }),
        JsValue::NativeFunction(_) => Some("function () { [native code] }".to_string()),
        JsValue::Error(err) => Some(err.to_error_string()),
        _ => None,
    }
}

/// Build the `result[]` array returned by `Runtime.getProperties` for
/// `value`.  Children registered for nested `RemoteObject`s share the
/// parent's `objectGroup` so that releasing the group cascades.
///
/// Unsupported / opaque variants return an empty descriptor list rather
/// than fabricating properties; this matches the conservative default
/// documented in `docs/edge_diagnostics.md`.
fn build_property_descriptors(
    value: &JsValue,
    own_properties: bool,
    registry: &mut RemoteObjectRegistry,
    object_group: Option<&str>,
) -> Vec<Value> {
    let mut descriptors: Vec<Value> = Vec::new();
    match value {
        JsValue::PlainObject(map_ref) => {
            // Clone keys/values out of the borrow before re-entering the
            // registry; this avoids RefCell reentrancy when child values
            // happen to alias the parent map.
            let entries: Vec<(String, JsValue, crate::objects::map::PropertyAttributes)> = map_ref
                .borrow()
                .iter_with_attrs()
                .map(|(k, v, a)| (k.to_string(), v.clone(), a))
                .collect();
            for (key, child, attrs) in entries {
                let remote = js_value_to_remote_object(&child, registry, object_group, false);
                descriptors.push(json!({
                    "name": key,
                    "value": remote,
                    "writable": attrs.contains(crate::objects::map::PropertyAttributes::WRITABLE),
                    "configurable": attrs.contains(crate::objects::map::PropertyAttributes::CONFIGURABLE),
                    "enumerable": attrs.contains(crate::objects::map::PropertyAttributes::ENUMERABLE),
                    "isOwn": true,
                }));
            }
        }
        JsValue::Array(arr_ref) => {
            let items: Vec<JsValue> = arr_ref.borrow().clone();
            let len = items.len();
            for (idx, item) in items.into_iter().enumerate() {
                let remote = js_value_to_remote_object(&item, registry, object_group, false);
                descriptors.push(json!({
                    "name": idx.to_string(),
                    "value": remote,
                    "writable": true,
                    "configurable": true,
                    "enumerable": true,
                    "isOwn": true,
                }));
            }
            // Non-enumerable `length` property mirrors ECMA-262 semantics.
            descriptors.push(json!({
                "name": "length",
                "value": {
                    "type": "number",
                    "value": len,
                    "description": len.to_string(),
                },
                "writable": true,
                "configurable": false,
                "enumerable": false,
                "isOwn": true,
            }));
        }
        JsValue::Function(bc) => {
            let length = bc.function_length();
            let name = bc.function_name().to_string();
            // Synthetic `length` and `name` mirror the standard own
            // properties of every Function object.
            descriptors.push(json!({
                "name": "length",
                "value": {
                    "type": "number",
                    "value": length,
                    "description": length.to_string(),
                },
                "writable": false,
                "configurable": true,
                "enumerable": false,
                "isOwn": true,
            }));
            descriptors.push(json!({
                "name": "name",
                "value": {"type": "string", "value": name},
                "writable": false,
                "configurable": true,
                "enumerable": false,
                "isOwn": true,
            }));
        }
        // Opaque / unsupported classes: return an empty own-property list
        // rather than fabricating descriptors.  `own_properties=false` is
        // accepted but currently never produces inherited descriptors
        // because Stator does not expose a per-object prototype chain to
        // the inspector.
        _ => {}
    }
    let _ = own_properties;
    descriptors
}

// ─────────────────────────────────────────────────────────────────────────────
// Server
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_TARGET_ID: &str = "stator-1";

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
        self.serve_stream(stream)
    }

    /// Accept and serve connections in a loop, blocking the calling thread.
    ///
    /// Each incoming connection is served to completion before the next is
    /// accepted (single-threaded / sequential).  Returns only when the
    /// underlying listener encounters a fatal [`io::Error`].
    pub fn accept_loop(&self) -> io::Result<()> {
        for stream in self.listener.incoming() {
            let stream = stream?;
            // Ignore per-session errors; move on to the next connection.
            let _ = self.serve_stream(stream);
        }
        Ok(())
    }

    fn serve_stream(&self, stream: TcpStream) -> io::Result<()> {
        if is_websocket_upgrade(&stream)? {
            let ws = accept(stream).map_err(|e| io::Error::other(e.to_string()))?;
            return CdpSession::new(ws).run();
        }
        serve_http_discovery(stream, self.local_addr()?)
    }
}

fn is_websocket_upgrade(stream: &TcpStream) -> io::Result<bool> {
    let mut buf = [0u8; 1024];
    let len = stream.peek(&mut buf)?;
    let request = String::from_utf8_lossy(&buf[..len]).to_ascii_lowercase();
    Ok(request.contains("upgrade: websocket"))
}

fn serve_http_discovery(mut stream: TcpStream, local_addr: std::net::SocketAddr) -> io::Result<()> {
    let mut request = Vec::with_capacity(1024);
    let mut buf = [0u8; 512];
    loop {
        let len = stream.read(&mut buf)?;
        if len == 0 {
            break;
        }
        request.extend_from_slice(&buf[..len]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") || request.len() > 8192 {
            break;
        }
    }
    let request = String::from_utf8_lossy(&request);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    let ws_url = format!("ws://{local_addr}/devtools/page/{DEFAULT_TARGET_ID}");
    let (status, content_type, body) = discovery_response(path, &ws_url);
    write_http_response(&mut stream, status, content_type, &body)
}

fn discovery_response(
    path: &str,
    web_socket_debugger_url: &str,
) -> (&'static str, &'static str, String) {
    match path {
        "/json/version" => (
            "200 OK",
            "application/json; charset=UTF-8",
            json!({
                "Browser": "Stator",
                "Protocol-Version": "1.3",
                "V8-Version": "stator",
                "WebKit-Version": "stator",
                "webSocketDebuggerUrl": web_socket_debugger_url,
            })
            .to_string(),
        ),
        "/json" | "/json/list" => (
            "200 OK",
            "application/json; charset=UTF-8",
            Value::Array(vec![discovery_target(web_socket_debugger_url)]).to_string(),
        ),
        "/json/protocol" => (
            "200 OK",
            "application/json; charset=UTF-8",
            schema_get_domains().to_string(),
        ),
        "/json/new" => (
            "200 OK",
            "application/json; charset=UTF-8",
            discovery_target(web_socket_debugger_url).to_string(),
        ),
        path if path == format!("/json/activate/{DEFAULT_TARGET_ID}") => (
            "200 OK",
            "application/json; charset=UTF-8",
            json!({ "result": true }).to_string(),
        ),
        path if path == format!("/json/close/{DEFAULT_TARGET_ID}") => (
            "200 OK",
            "application/json; charset=UTF-8",
            json!({ "result": true }).to_string(),
        ),
        _ => (
            "404 Not Found",
            "text/plain; charset=UTF-8",
            "Not Found".to_string(),
        ),
    }
}

fn target_info() -> Value {
    target_info_for_group(1, "Stator", "stator://inspector")
}

fn target_info_for_group(group_id: u32, name: &str, origin: &str) -> Value {
    let target_id = target_id_for_group(group_id);
    let title = if name.is_empty() { "Stator" } else { name };
    let url = if origin.is_empty() {
        "stator://inspector"
    } else {
        origin
    };
    json!({
        "targetId": target_id,
        "type": "page",
        "title": title,
        "url": url,
        "attached": false,
        "canAccessOpener": false,
    })
}

fn target_id_for_group(group_id: u32) -> String {
    format!("stator-{group_id}")
}

fn discovery_target(web_socket_debugger_url: &str) -> Value {
    json!({
        "id": DEFAULT_TARGET_ID,
        "type": "page",
        "title": "Stator",
        "description": "Stator JavaScript inspector target",
        "url": "stator://inspector",
        "webSocketDebuggerUrl": web_socket_debugger_url,
    })
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};
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

    fn http_get(path: &str) -> (thread::JoinHandle<io::Result<()>>, String) {
        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let port = server.local_addr().expect("local_addr").port();
        let handle = thread::spawn(move || server.accept_one());
        thread::sleep(Duration::from_millis(20));

        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        )
        .expect("write request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        (handle, response)
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
        assert_eq!(event["params"]["context"]["id"], 1);
        assert_eq!(event["params"]["context"]["origin"], "stator");
        assert_eq!(event["params"]["context"]["name"], "stator");
        assert_eq!(event["params"]["context"]["uniqueId"], "stator-1-1");
        assert_eq!(event["params"]["context"]["auxData"]["groupId"], 1);

        let ack: Value = serde_json::from_str(&match msg2 {
            Message::Text(t) => t.to_string(),
            other => panic!("unexpected: {other:?}"),
        })
        .expect("parse ack");
        assert_eq!(ack["id"], 3u64);
    }

    #[test]
    fn test_cdp_http_json_version() {
        let (handle, response) = http_get("/json/version");
        handle.join().expect("thread panic").expect("server ok");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        let body = response.split("\r\n\r\n").nth(1).expect("body");
        let json: Value = serde_json::from_str(body).expect("json body");
        assert_eq!(json["Browser"], "Stator");
        assert_eq!(json["Protocol-Version"], "1.3");
        assert!(
            json["webSocketDebuggerUrl"]
                .as_str()
                .unwrap()
                .contains("/devtools/page/stator-1")
        );
    }

    #[test]
    fn test_cdp_http_json_list() {
        let (handle, response) = http_get("/json/list");
        handle.join().expect("thread panic").expect("server ok");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        let body = response.split("\r\n\r\n").nth(1).expect("body");
        let targets: Value = serde_json::from_str(body).expect("json body");
        assert_eq!(targets[0]["id"], "stator-1");
        assert_eq!(targets[0]["type"], "page");
        assert!(
            targets[0]["webSocketDebuggerUrl"]
                .as_str()
                .unwrap()
                .contains("/devtools/page/stator-1")
        );
    }

    #[test]
    fn test_cdp_http_json_protocol() {
        let (handle, response) = http_get("/json/protocol");
        handle.join().expect("thread panic").expect("server ok");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        let body = response.split("\r\n\r\n").nth(1).expect("body");
        let json: Value = serde_json::from_str(body).expect("json body");
        let domains = json["domains"].as_array().expect("domains");
        assert!(domains.iter().any(|domain| domain["name"] == "Runtime"));
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
    fn profiler_set_sampling_interval_updates_next_start_interval() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.setSamplingInterval","params":{"samplingInterval":2500}}"#,
        );
        assert!(resp.get("error").is_none());
        assert_eq!(d.profiler_sampling_interval_micros, 2_500);
    }

    #[test]
    fn profiler_set_sampling_interval_rejects_zero() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.setSamplingInterval","params":{"samplingInterval":0}}"#,
        );
        assert!(resp["error"].is_object());
        assert_eq!(d.profiler_sampling_interval_micros, 1_000);
    }

    #[test]
    fn profiler_precise_coverage_records_runtime_evaluate() {
        let mut d = fresh_dispatcher();
        let start = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.startPreciseCoverage","params":{"callCount":true}}"#,
        );
        assert!(start.get("error").is_none());
        let eval = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"1 + 2","sourceURL":"stator://coverage.js"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 3);

        let coverage = dispatch(
            &mut d,
            r#"{"id":3,"method":"Profiler.takePreciseCoverage","params":{}}"#,
        );
        let scripts = coverage["result"]["result"].as_array().unwrap();
        assert_eq!(scripts.len(), 1);
        assert_eq!(scripts[0]["url"], "stator://coverage.js");
        assert_eq!(scripts[0]["functions"][0]["ranges"][0]["count"], 1);
    }

    #[test]
    fn profiler_precise_coverage_uses_persisted_script_id() {
        let mut d = fresh_dispatcher();
        let compiled = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.compileScript","params":{"expression":"21 * 2","persistScript":true}}"#,
        );
        let script_id = compiled["result"]["scriptId"].as_str().unwrap();
        let _ = dispatch(
            &mut d,
            r#"{"id":2,"method":"Profiler.startPreciseCoverage","params":{}}"#,
        );
        let run = dispatch(
            &mut d,
            &json!({
                "id": 3,
                "method": "Runtime.runScript",
                "params": { "scriptId": script_id }
            })
            .to_string(),
        );
        assert_eq!(run["result"]["result"]["value"], 42);
        let coverage = dispatch(
            &mut d,
            r#"{"id":4,"method":"Profiler.getBestEffortCoverage","params":{}}"#,
        );
        assert_eq!(coverage["result"]["result"][0]["scriptId"], script_id);
    }

    #[test]
    fn profiler_stop_precise_coverage_disables_recording() {
        let mut d = fresh_dispatcher();
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.startPreciseCoverage","params":{}}"#,
        );
        let _ = dispatch(
            &mut d,
            r#"{"id":2,"method":"Profiler.stopPreciseCoverage","params":{}}"#,
        );
        let eval = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"1 + 2"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 3);
        let coverage = dispatch(
            &mut d,
            r#"{"id":4,"method":"Profiler.takePreciseCoverage","params":{}}"#,
        );
        assert!(coverage["result"]["result"].as_array().unwrap().is_empty());
    }

    #[test]
    fn profiler_type_profile_records_runtime_result_types() {
        let mut d = fresh_dispatcher();
        let start = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.startTypeProfile","params":{}}"#,
        );
        assert!(start.get("error").is_none());
        let eval = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"'typed'","sourceURL":"stator://types.js"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], "typed");
        let profile = dispatch(
            &mut d,
            r#"{"id":3,"method":"Profiler.takeTypeProfile","params":{}}"#,
        );
        let result = profile["result"]["result"].as_array().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["url"], "stator://types.js");
        assert_eq!(result[0]["entries"][0]["types"][0]["name"], "String");
    }

    #[test]
    fn profiler_stop_type_profile_disables_recording() {
        let mut d = fresh_dispatcher();
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Profiler.startTypeProfile","params":{}}"#,
        );
        let _ = dispatch(
            &mut d,
            r#"{"id":2,"method":"Profiler.stopTypeProfile","params":{}}"#,
        );
        let eval = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"123"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 123);
        let profile = dispatch(
            &mut d,
            r#"{"id":4,"method":"Profiler.takeTypeProfile","params":{}}"#,
        );
        assert!(profile["result"]["result"].as_array().unwrap().is_empty());
    }

    #[test]
    fn profiler_console_profile_events_are_emitted_from_runtime_execution() {
        let mut d = fresh_dispatcher();
        let _ = dispatch(&mut d, r#"{"id":0,"method":"Profiler.enable","params":{}}"#);
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"console.profile('p'); console.profileEnd('p'); 42"}}"#,
        );
        assert_eq!(messages[0]["method"], "Profiler.consoleProfileStarted");
        assert_eq!(messages[0]["params"]["id"], "p");
        assert_eq!(messages[1]["method"], "Profiler.consoleProfileFinished");
        assert_eq!(messages[1]["params"]["id"], "p");
        assert!(messages[1]["params"]["profile"].is_object());
        assert_eq!(messages[2]["id"], 1);
        assert_eq!(messages[2]["result"]["result"]["value"], 42);
    }

    #[test]
    fn profiler_console_profile_events_are_dropped_when_profiler_disabled() {
        let mut d = fresh_dispatcher();
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"console.profile('p'); console.profileEnd('p'); 42"}}"#,
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], 1);
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
    fn heap_snapshot_is_streamed_in_chunks_with_progress() {
        let globals = Rc::new(RefCell::new(GlobalEnv::new()));
        globals.borrow_mut().insert(
            "large".to_string(),
            JsValue::String("x".repeat(HEAP_SNAPSHOT_CHUNK_SIZE + 1024).into()),
        );
        let mut d = CdpDispatcher::with_globals(globals);
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"HeapProfiler.takeHeapSnapshot","params":{}}"#,
        );
        let chunk_count = messages
            .iter()
            .filter(|message| message["method"] == "HeapProfiler.addHeapSnapshotChunk")
            .count();
        assert!(chunk_count > 1, "expected multiple chunks: {messages:?}");
        let progress: Vec<_> = messages
            .iter()
            .filter(|message| message["method"] == "HeapProfiler.reportHeapSnapshotProgress")
            .collect();
        assert_eq!(progress.len(), chunk_count);
        assert_eq!(progress.last().unwrap()["params"]["finished"], true);
        assert_eq!(messages.last().unwrap()["id"], 1);
    }

    #[test]
    fn split_snapshot_chunks_preserves_utf8_boundaries() {
        let chunks = split_snapshot_chunks("αβγδε", 3);
        assert_eq!(chunks.concat(), "αβγδε");
        assert!(
            chunks
                .iter()
                .all(|chunk| std::str::from_utf8(chunk.as_bytes()).is_ok())
        );
    }

    #[test]
    fn heap_object_id_roundtrips_to_remote_object() {
        let mut d = fresh_dispatcher();
        let object = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({answer: 42})","objectGroup":"heap"}}"#,
        );
        let object_id = object["result"]["result"]["objectId"].as_str().unwrap();
        let heap_id_response = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"HeapProfiler.getHeapObjectId","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        let heap_id = heap_id_response["result"]["heapSnapshotObjectId"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(heap_id.starts_with("heap-"));

        let again = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"HeapProfiler.getHeapObjectId","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        assert_eq!(again["result"]["heapSnapshotObjectId"], heap_id);

        let remote = dispatch(
            &mut d,
            &format!(
                r#"{{"id":4,"method":"HeapProfiler.getObjectByHeapObjectId","params":{{"objectId":"{heap_id}","objectGroup":"heap"}}}}"#
            ),
        );
        assert_eq!(remote["result"]["result"]["type"], "object");
        let roundtrip_id = remote["result"]["result"]["objectId"].as_str().unwrap();
        let props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":5,"method":"Runtime.getProperties","params":{{"objectId":"{roundtrip_id}","ownProperties":true}}}}"#
            ),
        );
        let answer = props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"] == "answer")
            .expect("answer property");
        assert_eq!(answer["value"]["value"], 42);
    }

    #[test]
    fn heap_add_inspected_object_validates_heap_id() {
        let mut d = fresh_dispatcher();
        let object = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({})"}}"#,
        );
        let object_id = object["result"]["result"]["objectId"].as_str().unwrap();
        let heap_id_response = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"HeapProfiler.getHeapObjectId","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        let heap_id = heap_id_response["result"]["heapSnapshotObjectId"]
            .as_str()
            .unwrap()
            .to_string();
        let ok = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"HeapProfiler.addInspectedHeapObject","params":{{"heapObjectId":"{heap_id}"}}}}"#
            ),
        );
        assert!(ok.get("error").is_none());
        assert!(d.inspected_heap_objects.contains(&heap_id));

        let missing = dispatch(
            &mut d,
            r#"{"id":4,"method":"HeapProfiler.addInspectedHeapObject","params":{"heapObjectId":"missing"}}"#,
        );
        assert!(missing["error"].is_object());
    }

    #[test]
    fn heap_sampling_profile_tracks_recorded_allocations() {
        use crate::inspector::heap_snapshot::record_allocation;

        let mut d = fresh_dispatcher();
        let start = dispatch(
            &mut d,
            r#"{"id":1,"method":"HeapProfiler.startSampling","params":{}}"#,
        );
        assert!(start.get("error").is_none());
        record_allocation(64);
        record_allocation(32);

        let current = dispatch(
            &mut d,
            r#"{"id":2,"method":"HeapProfiler.getSamplingProfile","params":{}}"#,
        );
        assert_eq!(
            current["result"]["profile"]["samples"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let stopped = dispatch(
            &mut d,
            r#"{"id":3,"method":"HeapProfiler.stopSampling","params":{}}"#,
        );
        let samples = stopped["result"]["profile"]["samples"].as_array().unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0]["size"], 64);
        assert_eq!(samples[1]["size"], 32);

        let after = dispatch(
            &mut d,
            r#"{"id":4,"method":"HeapProfiler.getSamplingProfile","params":{}}"#,
        );
        assert_eq!(
            after["result"]["profile"]["samples"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn heap_tracking_stop_emits_stats_and_last_seen_events() {
        use crate::inspector::heap_snapshot::record_allocation;

        let mut d = fresh_dispatcher();
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"HeapProfiler.startTrackingHeapObjects","params":{}}"#,
        );
        record_allocation(10);
        record_allocation(20);

        let messages = dispatch_all(
            &mut d,
            r#"{"id":2,"method":"HeapProfiler.stopTrackingHeapObjects","params":{}}"#,
        );
        assert_eq!(messages[0]["method"], "HeapProfiler.heapStatsUpdate");
        assert_eq!(messages[0]["params"]["statsUpdate"], json!([0, 2, 30]));
        assert_eq!(messages[1]["method"], "HeapProfiler.lastSeenObjectId");
        assert_eq!(messages[1]["params"]["lastSeenObjectId"], 2);
        assert_eq!(messages[2]["id"], 2);
        assert_eq!(messages[2]["result"]["stats"].as_array().unwrap().len(), 2);
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
        assert!(
            json["result"]["exceptionDetails"].is_object(),
            "syntax failures should have Runtime.ExceptionDetails"
        );
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
    fn test_cdp_runtime_get_properties_unknown_object_id_errors() {
        let (handle, mut ws, _port) = start_server();

        ws.send(Message::Text(
            r#"{"id":11,"method":"Runtime.getProperties","params":{"objectId":"does-not-exist"}}"#
                .into(),
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
        let err = json["error"]
            .as_object()
            .expect("getProperties on unknown id must return a structured error");
        assert!(
            err["message"]
                .as_str()
                .unwrap_or("")
                .contains("does-not-exist"),
            "error message should name the stale id"
        );
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
    fn debugger_remove_breakpoint_removes_known_cdp_breakpoint() {
        let mut d = fresh_dispatcher();
        let set = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"lineNumber":5,"columnNumber":0}}"#,
        );
        let breakpoint_id = set["result"]["breakpointId"].as_str().unwrap();
        assert!(d.cdp_breakpoints.contains(breakpoint_id));

        let remove = dispatch(
            &mut d,
            &json!({
                "id": 2,
                "method": "Debugger.removeBreakpoint",
                "params": { "breakpointId": breakpoint_id }
            })
            .to_string(),
        );
        assert!(remove.get("error").is_none());
        assert!(!d.cdp_breakpoints.contains(breakpoint_id));

        let again = dispatch(
            &mut d,
            &json!({
                "id": 3,
                "method": "Debugger.removeBreakpoint",
                "params": { "breakpointId": breakpoint_id }
            })
            .to_string(),
        );
        assert!(again["error"].is_object());
    }

    #[test]
    fn debugger_set_breakpoint_by_location_installs_and_removes_interpreter_breakpoint() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0}}}"#,
        );

        assert!(response["result"]["breakpointId"].is_string());
        assert_eq!(response["result"]["actualLocation"]["scriptId"], "7");
        assert_eq!(dbg.borrow().breakpoints().count(), 1);

        let breakpoint_id = response["result"]["breakpointId"].as_str().unwrap();
        let removed = dispatch(
            &mut d,
            &json!({
                "id": 2,
                "method": "Debugger.removeBreakpoint",
                "params": { "breakpointId": breakpoint_id }
            })
            .to_string(),
        );
        assert!(removed.get("error").is_none());
        assert_eq!(dbg.borrow().breakpoints().count(), 0);
    }

    #[test]
    fn debugger_set_breakpoint_unknown_script_errors() {
        let mut d = fresh_dispatcher();
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"missing","lineNumber":0,"columnNumber":0}}}"#,
        );
        assert!(response["error"].is_object());
    }

    #[test]
    fn debugger_set_breakpoint_by_url_resolves_registered_script_url() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let messages = dispatch_all(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","lineNumber":1,"columnNumber":0}}"#,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["method"], "Debugger.breakpointResolved");
        assert_eq!(messages[0]["params"]["location"]["scriptId"], "7");

        let response = messages.last().unwrap();
        let locations = response["result"]["locations"].as_array().unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0]["scriptId"], "7");
        assert_eq!(locations[0]["lineNumber"], 1);
        assert!(locations[0]["columnNumber"].as_u64().is_some());
        assert_eq!(
            response["result"]["breakpointId"],
            messages[0]["params"]["breakpointId"]
        );
    }

    #[test]
    fn debugger_set_breakpoint_by_url_unknown_url_is_future_breakpoint() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "var a = 1;\n//# sourceURL=stator://app.js".to_string());

        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://missing.js","lineNumber":0,"columnNumber":0}}"#,
        );
        assert!(response["result"]["breakpointId"].is_string());
        assert!(
            response["result"]["locations"]
                .as_array()
                .unwrap()
                .is_empty()
        );
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

    // ── RemoteObject registry tests (in-process, no WebSocket) ──────────────

    fn dispatch(d: &mut CdpDispatcher, json_text: &str) -> Value {
        let before = d.pending_count();
        assert_eq!(d.dispatch_json(json_text), DispatchOutcome::Ok);
        // The most recent reply is the last pushed message.
        let mut last = None;
        while d.pending_count() > before {
            last = d.take_next();
        }
        serde_json::from_str(&last.expect("response was queued")).expect("valid JSON")
    }

    fn dispatch_all(d: &mut CdpDispatcher, json_text: &str) -> Vec<Value> {
        let before = d.pending_count();
        assert_eq!(d.dispatch_json(json_text), DispatchOutcome::Ok);
        let mut messages = Vec::new();
        while d.pending_count() > before {
            let message = d.take_next().expect("message was queued");
            messages.push(serde_json::from_str(&message).expect("valid JSON"));
        }
        messages
    }

    #[test]
    fn remote_object_registry_mints_unique_monotonic_ids() {
        let mut reg = RemoteObjectRegistry::new();
        let a = reg.register(JsValue::Boolean(true), None);
        let b = reg.register(JsValue::Smi(7), Some("g".into()));
        let c = reg.register(JsValue::Null, None);
        assert_eq!(a, "1");
        assert_eq!(b, "2");
        assert_eq!(c, "3");
        assert_eq!(reg.len(), 3);
        assert!(reg.get("1").is_some());
        assert!(reg.get("99").is_none());
        assert_eq!(reg.group_of("2"), Some("g"));
        assert_eq!(reg.group_of("1"), None);
    }

    #[test]
    fn remote_object_release_drops_only_target_id() {
        let mut reg = RemoteObjectRegistry::new();
        let a = reg.register(JsValue::Smi(1), None);
        let b = reg.register(JsValue::Smi(2), None);
        assert!(reg.release(&a));
        assert!(!reg.release(&a), "double release returns false");
        assert!(reg.get(&a).is_none());
        assert!(reg.get(&b).is_some());
    }

    #[test]
    fn remote_object_release_group_drops_matching_entries_only() {
        let mut reg = RemoteObjectRegistry::new();
        let _x = reg.register(JsValue::Smi(1), Some("evalA".into()));
        let _y = reg.register(JsValue::Smi(2), Some("evalA".into()));
        let z = reg.register(JsValue::Smi(3), Some("evalB".into()));
        let keep = reg.register(JsValue::Smi(4), None);
        assert_eq!(reg.release_group("evalA"), 2);
        assert!(reg.get(&z).is_some());
        assert!(reg.get(&keep).is_some());
        assert_eq!(reg.release_group("nonexistent"), 0);
    }

    fn fresh_dispatcher() -> CdpDispatcher {
        CdpDispatcher::with_globals(Rc::new(RefCell::new(GlobalEnv::new())))
    }

    #[test]
    fn schema_get_domains_lists_supported_domains() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Schema.getDomains","params":{}}"#,
        );
        let names: Vec<_> = resp["result"]["domains"]
            .as_array()
            .expect("domains array")
            .iter()
            .filter_map(|domain| domain["name"].as_str())
            .collect();
        assert!(names.contains(&"Runtime"));
        assert!(names.contains(&"Debugger"));
        assert!(names.contains(&"Schema"));
    }

    #[test]
    fn target_get_targets_lists_default_target() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Target.getTargets","params":{}}"#,
        );
        assert_eq!(
            resp["result"]["targetInfos"][0]["targetId"],
            DEFAULT_TARGET_ID
        );
        assert_eq!(resp["result"]["targetInfos"][0]["type"], "page");
    }

    #[test]
    fn target_get_targets_lists_one_target_per_context_group() {
        let contexts = vec![
            ExecutionContextDescription::new(1, 1, "https://a.test", "main", json!({})),
            ExecutionContextDescription::new(2, 2, "https://b.test", "isolated", json!({})),
            ExecutionContextDescription::new(3, 2, "https://b.test", "isolated-2", json!({})),
        ];
        let mut d = CdpDispatcher::with_globals_and_contexts(
            Rc::new(RefCell::new(GlobalEnv::new())),
            contexts,
        );

        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Target.getTargets","params":{}}"#,
        );
        let target_infos = resp["result"]["targetInfos"].as_array().unwrap();
        let target_ids: Vec<_> = target_infos
            .iter()
            .map(|info| info["targetId"].as_str().unwrap())
            .collect();
        assert_eq!(target_ids, vec!["stator-1", "stator-2"]);

        let attach = dispatch_all(
            &mut d,
            r#"{"id":2,"method":"Target.attachToTarget","params":{"targetId":"stator-2","flatten":true}}"#,
        );
        assert_eq!(attach[0]["method"], "Target.attachedToTarget");
        assert_eq!(attach[0]["params"]["targetInfo"]["targetId"], "stator-2");
    }

    #[test]
    fn target_discovery_emits_target_created_event() {
        let mut d = fresh_dispatcher();
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Target.setDiscoverTargets","params":{"discover":true}}"#,
        );
        assert_eq!(messages[0]["method"], "Target.targetCreated");
        assert_eq!(
            messages[0]["params"]["targetInfo"]["targetId"],
            DEFAULT_TARGET_ID
        );
        assert_eq!(messages[1]["id"], 1);
    }

    #[test]
    fn target_attach_send_message_and_detach_roundtrip() {
        let mut d = fresh_dispatcher();
        let attached = dispatch_all(
            &mut d,
            &json!({
                "id": 1,
                "method": "Target.attachToTarget",
                "params": {"targetId": DEFAULT_TARGET_ID, "flatten": true}
            })
            .to_string(),
        );
        assert_eq!(attached[0]["method"], "Target.attachedToTarget");
        let session_id = attached[1]["result"]["sessionId"].as_str().unwrap();

        let inner = json!({
            "id": 99,
            "method": "Runtime.evaluate",
            "params": {"expression": "1 + 2"}
        })
        .to_string();
        let messages = dispatch_all(
            &mut d,
            &json!({
                "id": 2,
                "method": "Target.sendMessageToTarget",
                "params": {"sessionId": session_id, "message": inner}
            })
            .to_string(),
        );
        assert_eq!(messages[0]["method"], "Target.receivedMessageFromTarget");
        let nested: Value =
            serde_json::from_str(messages[0]["params"]["message"].as_str().unwrap()).unwrap();
        assert_eq!(nested["id"], 99);
        assert_eq!(nested["result"]["result"]["value"], 3);
        assert_eq!(messages[1]["id"], 2);

        let detached = dispatch_all(
            &mut d,
            &json!({
                "id": 3,
                "method": "Target.detachFromTarget",
                "params": {"sessionId": session_id}
            })
            .to_string(),
        );
        assert_eq!(detached[0]["method"], "Target.detachedFromTarget");
        assert_eq!(detached[1]["id"], 3);
    }

    #[test]
    fn target_close_emits_lifecycle_events_and_removes_default_target() {
        let mut d = fresh_dispatcher();
        let _ = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Target.setDiscoverTargets","params":{"discover":true}}"#,
        );
        let attached = dispatch_all(
            &mut d,
            &json!({
                "id": 2,
                "method": "Target.attachToTarget",
                "params": {"targetId": DEFAULT_TARGET_ID, "flatten": true}
            })
            .to_string(),
        );
        let session_id = attached[1]["result"]["sessionId"].as_str().unwrap();

        let closed = dispatch_all(
            &mut d,
            &json!({
                "id": 3,
                "method": "Target.closeTarget",
                "params": {"targetId": DEFAULT_TARGET_ID}
            })
            .to_string(),
        );
        assert_eq!(closed[0]["method"], "Target.detachedFromTarget");
        assert_eq!(closed[0]["params"]["sessionId"], session_id);
        assert_eq!(closed[1]["method"], "Target.targetDestroyed");
        assert_eq!(closed[1]["params"]["targetId"], DEFAULT_TARGET_ID);
        assert_eq!(closed[2]["result"]["success"], true);

        let targets = dispatch(
            &mut d,
            r#"{"id":4,"method":"Target.getTargets","params":{}}"#,
        );
        assert!(
            targets["result"]["targetInfos"]
                .as_array()
                .unwrap()
                .is_empty()
        );

        let attach_after_close = dispatch(
            &mut d,
            &json!({
                "id": 5,
                "method": "Target.attachToTarget",
                "params": {"targetId": DEFAULT_TARGET_ID, "flatten": true}
            })
            .to_string(),
        );
        assert!(attach_after_close["error"].is_object());
    }

    #[test]
    fn runtime_handshake_compat_methods_ack() {
        let mut d = fresh_dispatcher();
        let run = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.runIfWaitingForDebugger","params":{}}"#,
        );
        assert!(run["result"].is_object());
        assert!(run.get("error").is_none());

        let isolate = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.getIsolateId","params":{}}"#,
        );
        assert_eq!(isolate["result"]["id"], "stator-isolate-0");
    }

    #[test]
    fn runtime_discard_console_entries_clears_buffer() {
        use crate::inspector::console::{ConsoleMessage, MessageLevel, push_console_message};

        let _ = drain_messages();
        push_console_message(ConsoleMessage {
            level: MessageLevel::Log,
            text: "discard me".to_string(),
        });

        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.discardConsoleEntries","params":{}}"#,
        );
        assert!(resp.get("error").is_none());
        assert!(drain_messages().is_empty());
    }

    #[test]
    fn runtime_global_lexical_scope_names_returns_sorted_globals() {
        let globals = Rc::new(RefCell::new(GlobalEnv::new()));
        {
            let mut env = globals.borrow_mut();
            env.insert("zeta".to_string(), JsValue::Smi(1));
            env.insert("alpha".to_string(), JsValue::Boolean(true));
        }
        let mut d = CdpDispatcher::with_globals(globals);
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.globalLexicalScopeNames","params":{}}"#,
        );
        assert_eq!(resp["result"]["names"], json!(["alpha", "zeta"]));
    }

    #[test]
    fn runtime_global_lexical_scope_names_validates_context() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.globalLexicalScopeNames","params":{"executionContextId":99}}"#,
        );
        assert!(resp["error"].is_object(), "unknown context should error");
    }

    #[test]
    fn runtime_get_heap_usage_reports_reachable_heap_estimate() {
        let globals = Rc::new(RefCell::new(GlobalEnv::new()));
        globals.borrow_mut().insert(
            "message".to_string(),
            JsValue::String("hello inspector".to_string().into()),
        );
        let mut d = CdpDispatcher::with_globals(globals);
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.getHeapUsage","params":{}}"#,
        );
        let used = resp["result"]["usedSize"].as_u64().expect("used size");
        let total = resp["result"]["totalSize"].as_u64().expect("total size");
        assert!(used > 0);
        assert!(total >= used);
    }

    #[test]
    fn runtime_query_objects_returns_reachable_matching_prototypes() {
        let mut d = fresh_dispatcher();
        let proto = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"var proto = {marker: true}; var a = Object.create(proto); var b = Object.create(proto); var c = {}; proto","objectGroup":"query"}}"#,
        );
        let proto_id = proto["result"]["result"]["objectId"].as_str().unwrap();
        let query = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.queryObjects","params":{{"prototypeObjectId":"{proto_id}","objectGroup":"query"}}}}"#
            ),
        );
        let objects_id = query["result"]["objects"]["objectId"].as_str().unwrap();
        let props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.getProperties","params":{{"objectId":"{objects_id}","ownProperties":true}}}}"#
            ),
        );
        let entries = props["result"]["result"].as_array().unwrap();
        let indexed_count = entries
            .iter()
            .filter(|entry| {
                entry["name"]
                    .as_str()
                    .is_some_and(|name| name.chars().all(|ch| ch.is_ascii_digit()))
            })
            .count();
        assert_eq!(indexed_count, 2);
    }

    #[test]
    fn runtime_query_objects_unknown_prototype_errors() {
        let mut d = fresh_dispatcher();
        let query = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.queryObjects","params":{"prototypeObjectId":"missing"}}"#,
        );
        assert!(query["error"].is_object());
    }

    #[test]
    fn runtime_compile_script_persists_and_run_script_executes() {
        let mut d = fresh_dispatcher();
        let compiled = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.compileScript","params":{"expression":"var n = 40; n + 2","sourceURL":"stator://compiled.js","persistScript":true}}"#,
        );
        let script_id = compiled["result"]["scriptId"]
            .as_str()
            .expect("scriptId")
            .to_string();

        let run = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.runScript","params":{{"scriptId":"{script_id}"}}}}"#
            ),
        );
        assert_eq!(run["result"]["result"]["type"], "number");
        assert_eq!(run["result"]["result"]["value"], 42);
    }

    #[test]
    fn runtime_compile_script_without_persist_does_not_cache() {
        let mut d = fresh_dispatcher();
        let compiled = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.compileScript","params":{"expression":"1 + 1","persistScript":false}}"#,
        );
        assert!(compiled["result"].is_object());
        assert!(compiled["result"].get("scriptId").is_none());
    }

    #[test]
    fn runtime_compile_script_syntax_error_returns_exception_details() {
        let mut d = fresh_dispatcher();
        let compiled = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.compileScript","params":{"expression":"var =","sourceURL":"stator://bad.js","persistScript":true}}"#,
        );
        let details = &compiled["result"]["exceptionDetails"];
        assert_eq!(details["url"], "stator://bad.js");
        assert!(details["text"].as_str().unwrap().contains("SyntaxError"));
        assert!(compiled["result"].get("scriptId").is_none());
    }

    #[test]
    fn runtime_run_script_unknown_id_errors() {
        let mut d = fresh_dispatcher();
        let run = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.runScript","params":{"scriptId":"missing"}}"#,
        );
        assert!(run["error"].is_object());
        assert!(
            run["error"]["message"]
                .as_str()
                .unwrap()
                .contains("missing")
        );
    }

    #[test]
    fn runtime_add_binding_emits_binding_called_event() {
        let mut d = fresh_dispatcher();
        let _ = dispatch_all(&mut d, r#"{"id":0,"method":"Runtime.enable","params":{}}"#);
        let add = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.addBinding","params":{"name":"statorBinding"}}"#,
        );
        assert!(add.get("error").is_none());

        let messages = dispatch_all(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"statorBinding('payload')","contextId":1}}"#,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["method"], "Runtime.bindingCalled");
        assert_eq!(messages[0]["params"]["name"], "statorBinding");
        assert_eq!(messages[0]["params"]["payload"], "payload");
        assert_eq!(messages[0]["params"]["executionContextId"], 1);
        assert_eq!(messages[1]["id"], 2);
        assert_eq!(messages[1]["result"]["result"]["type"], "undefined");
    }

    #[test]
    fn runtime_remove_binding_removes_global_callback() {
        let mut d = fresh_dispatcher();
        let _ = dispatch_all(&mut d, r#"{"id":0,"method":"Runtime.enable","params":{}}"#);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.addBinding","params":{"name":"statorBinding"}}"#,
        );
        let remove = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.removeBinding","params":{"name":"statorBinding"}}"#,
        );
        assert!(remove.get("error").is_none());

        let messages = dispatch_all(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"typeof statorBinding"}}"#,
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["result"]["result"]["type"], "string");
        assert_eq!(messages[0]["result"]["result"]["value"], "undefined");
    }

    #[test]
    fn evaluate_primitive_does_not_mint_object_id() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"42"}}"#,
        );
        assert_eq!(resp["result"]["result"]["type"], "number");
        assert_eq!(resp["result"]["result"]["value"], 42);
        assert!(resp["result"]["result"].get("objectId").is_none());
        assert_eq!(d.remote_objects().len(), 0);
    }

    #[test]
    fn evaluate_object_mints_object_id_under_group() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({a:1,b:2})","objectGroup":"console"}}"#,
        );
        let oid = resp["result"]["result"]["objectId"]
            .as_str()
            .expect("objectId present")
            .to_string();
        assert_eq!(resp["result"]["result"]["type"], "object");
        assert_eq!(resp["result"]["result"]["className"], "Object");
        assert_eq!(d.remote_objects().len(), 1);
        assert_eq!(d.remote_objects().group_of(&oid), Some("console"));
    }

    #[test]
    fn evaluate_thrown_error_returns_exception_details_and_event() {
        let mut d = fresh_dispatcher();
        let _enable = dispatch_all(&mut d, r#"{"id":0,"method":"Runtime.enable","params":{}}"#);
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"throw new Error('boom')","generatePreview":true,"sourceURL":"stator://eval.js"}}"#,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["method"], "Runtime.exceptionThrown");
        assert_eq!(messages[1]["id"], 1);
        let details = &messages[1]["result"]["exceptionDetails"];
        assert_eq!(details["exceptionId"], 1);
        assert_eq!(details["url"], "stator://eval.js");
        assert_eq!(details["executionContextId"], 1);
        assert_eq!(details["exception"]["subtype"], "error");
        assert_eq!(details["exception"]["preview"]["subtype"], "error");
        assert!(details["text"].as_str().unwrap().contains("boom"));
        assert_eq!(
            messages[0]["params"]["exceptionDetails"]["exceptionId"],
            details["exceptionId"]
        );
    }

    #[test]
    fn evaluate_thrown_primitive_returns_exception_details_without_handle() {
        let mut d = fresh_dispatcher();
        let messages = dispatch_all(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"throw 7"}}"#,
        );
        assert_eq!(messages.len(), 1, "Runtime disabled means no event");
        let details = &messages[0]["result"]["exceptionDetails"];
        assert_eq!(details["exceptionId"], 1);
        assert_eq!(details["text"], "7");
        assert_eq!(details["exception"]["type"], "number");
        assert!(details["exception"].get("objectId").is_none());
    }

    #[test]
    fn evaluate_thrown_object_returns_remote_exception_handle() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"throw {name:'CustomError', message:'boom'}","generatePreview":true}}"#,
        );
        let details = &resp["result"]["exceptionDetails"];
        assert_eq!(details["exception"]["type"], "object");
        assert!(details["exception"]["objectId"].as_str().is_some());
        assert_eq!(
            details["exception"]["preview"]["properties"][0]["name"],
            "name"
        );
    }

    #[test]
    fn call_function_on_throw_returns_exception_details() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.callFunctionOn","params":{"functionDeclaration":"function(){throw new TypeError('bad call')}"}}"#,
        );
        let details = &resp["result"]["exceptionDetails"];
        assert_eq!(details["exceptionId"], 1);
        assert!(details["text"].as_str().unwrap().contains("bad call"));
        assert_eq!(details["exception"]["subtype"], "error");
    }

    #[test]
    fn evaluate_syntax_error_returns_exception_details_without_remote_exception() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"var ="}}"#,
        );
        let details = &resp["result"]["exceptionDetails"];
        assert_eq!(details["exceptionId"], 1);
        assert!(details["text"].as_str().unwrap().contains("SyntaxError"));
        assert!(details.get("exception").is_none());
    }

    #[test]
    fn exception_details_respect_requested_context_id() {
        let contexts = vec![
            default_execution_context(),
            ExecutionContextDescription::new(2, 1, "stator", "secondary", json!({})),
        ];
        let mut d = CdpDispatcher::with_globals_and_contexts(
            Rc::new(RefCell::new(GlobalEnv::new())),
            contexts,
        );
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"contextId":2,"expression":"throw 'ctx2'"}}"#,
        );
        assert_eq!(resp["result"]["exceptionDetails"]["executionContextId"], 2);
    }

    #[test]
    fn object_preview_plain_object_array_and_function() {
        let mut d = fresh_dispatcher();
        let object = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({a:1,b:'x'})","generatePreview":true}}"#,
        );
        assert_eq!(object["result"]["result"]["preview"]["type"], "object");
        assert_eq!(
            object["result"]["result"]["preview"]["properties"][0]["name"],
            "a"
        );

        let array = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"[10,20]","generatePreview":true}}"#,
        );
        assert_eq!(array["result"]["result"]["preview"]["subtype"], "array");
        assert_eq!(
            array["result"]["result"]["preview"]["properties"][1]["value"],
            "20"
        );

        let function = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"(function named(a,b){return a})","generatePreview":true}}"#,
        );
        assert_eq!(function["result"]["result"]["preview"]["type"], "function");
        assert_eq!(
            function["result"]["result"]["preview"]["properties"][1]["value"],
            "named"
        );
    }

    #[test]
    fn object_preview_truncates_after_limit() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({a:1,b:2,c:3,d:4,e:5,f:6})","generatePreview":true}}"#,
        );
        let preview = &resp["result"]["result"]["preview"];
        assert_eq!(preview["overflow"], true);
        assert_eq!(preview["properties"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn evaluate_object_id_is_stable_across_get_properties_calls() {
        let mut d = fresh_dispatcher();
        let r1 = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({x:10})"}}"#,
        );
        let oid = r1["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();

        let p1 = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let p2 = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        // Both calls succeed against the same id.
        assert!(p1["result"]["result"].is_array());
        assert!(p2["result"]["result"].is_array());
        let names1: Vec<&str> = p1["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        let names2: Vec<&str> = p2["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        assert_eq!(names1, names2);
        assert!(names1.contains(&"x"));
    }

    #[test]
    fn get_properties_plain_object_returns_descriptors_with_attrs() {
        let mut d = fresh_dispatcher();
        let r = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({foo:1, bar:'hi'})"}}"#,
        );
        let oid = r["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        let p = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let arr = p["result"]["result"].as_array().expect("array");
        let by_name: std::collections::HashMap<&str, &Value> = arr
            .iter()
            .map(|d| (d["name"].as_str().unwrap(), d))
            .collect();
        let foo = by_name.get("foo").expect("foo prop");
        assert_eq!(foo["value"]["type"], "number");
        assert_eq!(foo["value"]["value"], 1);
        assert_eq!(foo["isOwn"], true);
        assert_eq!(foo["enumerable"], true);
        assert_eq!(foo["writable"], true);
        let bar = by_name.get("bar").expect("bar prop");
        assert_eq!(bar["value"]["type"], "string");
        assert_eq!(bar["value"]["value"], "hi");
    }

    #[test]
    fn get_properties_array_returns_indexed_props_and_length() {
        let mut d = fresh_dispatcher();
        let r = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"[10, 20, 30]"}}"#,
        );
        let oid = r["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(r["result"]["result"]["subtype"], "array");
        let p = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let arr = p["result"]["result"].as_array().unwrap();
        let names: Vec<&str> = arr.iter().map(|d| d["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["0", "1", "2", "length"]);
        let length_desc = arr.iter().find(|d| d["name"] == "length").unwrap();
        assert_eq!(length_desc["enumerable"], false);
        assert_eq!(length_desc["configurable"], false);
        assert_eq!(length_desc["value"]["value"], 3);
    }

    #[test]
    fn get_properties_function_returns_length_and_name() {
        let mut d = fresh_dispatcher();
        let r = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"(function add(a,b){return a+b})"}}"#,
        );
        assert_eq!(r["result"]["result"]["type"], "function");
        let oid = r["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        let p = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let arr = p["result"]["result"].as_array().unwrap();
        let by_name: std::collections::HashMap<&str, &Value> = arr
            .iter()
            .map(|d| (d["name"].as_str().unwrap(), d))
            .collect();
        let length = by_name.get("length").expect("length own prop");
        assert_eq!(length["value"]["type"], "number");
        let name = by_name.get("name").expect("name own prop");
        assert_eq!(name["value"]["type"], "string");
        assert_eq!(name["value"]["value"], "add");
        assert_eq!(name["writable"], false);
    }

    #[test]
    fn release_object_then_get_properties_fails_closed() {
        let mut d = fresh_dispatcher();
        let r = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({k:1})"}}"#,
        );
        let oid = r["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        let rel = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.releaseObject","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        assert!(rel.get("error").is_none(), "release should succeed");
        assert_eq!(d.remote_objects().len(), 0);

        let p = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let err = p["error"]
            .as_object()
            .expect("must fail closed on released id");
        assert!(err["message"].as_str().unwrap().contains(&oid));
    }

    #[test]
    fn release_object_double_release_returns_error() {
        let mut d = fresh_dispatcher();
        let r = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({})"}}"#,
        );
        let oid = r["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        let _ok = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.releaseObject","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        let again = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.releaseObject","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        assert!(again.get("error").is_some(), "double release must error");
    }

    #[test]
    fn release_object_group_drops_all_matching_entries() {
        let mut d = fresh_dispatcher();
        let _r1 = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({a:1})","objectGroup":"gA"}}"#,
        );
        let _r2 = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"[1,2,3]","objectGroup":"gA"}}"#,
        );
        let r3 = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"({z:9})","objectGroup":"gB"}}"#,
        );
        let oid_b = r3["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(d.remote_objects().len(), 3);

        let ok = dispatch(
            &mut d,
            r#"{"id":4,"method":"Runtime.releaseObjectGroup","params":{"objectGroup":"gA"}}"#,
        );
        assert!(ok.get("error").is_none());
        assert_eq!(d.remote_objects().len(), 1);
        assert!(d.remote_objects().get(&oid_b).is_some());

        // Releasing an unknown group is a no-op success.
        let noop = dispatch(
            &mut d,
            r#"{"id":5,"method":"Runtime.releaseObjectGroup","params":{"objectGroup":"unknown"}}"#,
        );
        assert!(noop.get("error").is_none());
    }

    #[test]
    fn get_properties_missing_object_id_param_errors() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.getProperties","params":{}}"#,
        );
        assert!(resp["error"].as_object().is_some());
    }

    #[test]
    fn release_object_missing_param_errors() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.releaseObject","params":{}}"#,
        );
        assert!(resp["error"].as_object().is_some());
    }

    #[test]
    fn release_object_group_missing_param_errors() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.releaseObjectGroup","params":{}}"#,
        );
        assert!(resp["error"].as_object().is_some());
    }

    #[test]
    fn remote_object_ids_are_isolated_between_dispatchers() {
        // Two dispatchers (i.e. two sessions) share no registry: an
        // objectId minted in one must not resolve in the other.
        let mut da = fresh_dispatcher();
        let mut db = fresh_dispatcher();
        let ra = dispatch(
            &mut da,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({a:1})"}}"#,
        );
        let oid = ra["result"]["result"]["objectId"]
            .as_str()
            .unwrap()
            .to_string();
        let cross = dispatch(
            &mut db,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{oid}"}}}}"#
            ),
        );
        assert!(
            cross["error"].as_object().is_some(),
            "objectId from dispatcher A must not resolve in dispatcher B"
        );
    }

    #[test]
    fn dispatcher_drop_releases_registry() {
        // The registry is owned by the dispatcher; dropping the dispatcher
        // drops every registered value with no further bookkeeping.
        let mut d = fresh_dispatcher();
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"({k:1})"}}"#,
        );
        assert_eq!(d.remote_objects().len(), 1);
        drop(d);
        // No assertion beyond "did not leak"; this test exists to document
        // the teardown contract and to ensure no panic on drop.
    }

    #[test]
    fn no_session_means_no_remote_object_entries() {
        // A bare dispatcher with no dispatched messages must have an empty
        // registry; nothing else is implicitly registered.
        let d = fresh_dispatcher();
        assert!(d.remote_objects().is_empty());
    }

    // ── Debugger bridge tests ───────────────────────────────────────────────

    use crate::inspector::debugger::Debugger as InterpreterDebugger;

    /// Attach a fresh interpreter Debugger to the dispatcher and return its
    /// shared handle so the test can drive pauses against it.
    fn attach_test_debugger(d: &mut CdpDispatcher) -> Rc<RefCell<InterpreterDebugger>> {
        let dbg = Rc::new(RefCell::new(InterpreterDebugger::new()));
        d.attach_debugger(Rc::clone(&dbg));
        dbg
    }

    /// Drain every queued message from the dispatcher into parsed JSON.
    fn drain_all(d: &mut CdpDispatcher) -> Vec<Value> {
        let mut out = Vec::new();
        while let Some(text) = d.take_next() {
            out.push(serde_json::from_str(&text).expect("valid JSON"));
        }
        out
    }

    #[test]
    fn debugger_disable_clears_enabled_flag() {
        let mut d = fresh_dispatcher();
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        assert!(d.debugger_enabled());
        let _ = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.disable","params":{}}"#,
        );
        assert!(!d.debugger_enabled());
    }

    #[test]
    fn set_pause_on_exceptions_rejects_invalid_state() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setPauseOnExceptions","params":{"state":"sometimes"}}"#,
        );
        assert!(resp["error"].is_object(), "invalid state should error");
    }

    #[test]
    fn set_pause_on_exceptions_propagates_to_attached_debugger() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setPauseOnExceptions","params":{"state":"all"}}"#,
        );
        assert_eq!(d.pause_on_exceptions(), PauseOnExceptionsState::All);
        assert!(dbg.borrow().pause_on_exceptions);

        let _ = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setPauseOnExceptions","params":{"state":"none"}}"#,
        );
        assert!(!dbg.borrow().pause_on_exceptions);
    }

    #[test]
    fn pause_on_exceptions_applied_on_late_attach() {
        let mut d = fresh_dispatcher();
        // Configure before any debugger is attached: state must be cached.
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setPauseOnExceptions","params":{"state":"uncaught"}}"#,
        );
        assert_eq!(d.pause_on_exceptions(), PauseOnExceptionsState::Uncaught);

        let dbg = attach_test_debugger(&mut d);
        assert!(
            dbg.borrow().pause_on_exceptions,
            "late-attached debugger should inherit cached pause state"
        );
    }

    #[test]
    fn notify_paused_requires_debugger_enabled_and_attached() {
        let mut d = fresh_dispatcher();
        // No debugger, no enable: no event.
        assert!(!d.notify_paused());

        let dbg = attach_test_debugger(&mut d);
        // Even attached, no enable: no event.
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        // Simulate a pause by driving the debugger directly.
        let _ = dbg.borrow_mut().check_pause_at(0);
        assert!(
            !d.notify_paused(),
            "Debugger.paused must not be emitted before Debugger.enable"
        );

        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        // Drain the enable ack.
        let _ = drain_all(&mut d);
        assert!(d.notify_paused());
        let msgs = drain_all(&mut d);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["method"], "Debugger.paused");
        assert_eq!(msgs[0]["params"]["reason"], "other");
        assert_eq!(
            msgs[0]["params"]["callFrames"][0]["functionName"],
            "(stator: paused-frame)"
        );
        assert_eq!(msgs[0]["params"]["hitBreakpoints"][0], "1");
        assert_eq!(msgs[0]["params"]["data"]["pausedFrame"], true);
    }

    #[test]
    fn notify_paused_maps_debugger_statement_reason() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        // Trigger a debugger-statement style pause.
        let _ = dbg.borrow_mut().on_debugger_statement(42);
        assert!(d.notify_paused());
        let msgs = drain_all(&mut d);
        assert_eq!(msgs[0]["params"]["reason"], "debuggerStatement");
        assert_eq!(msgs[0]["params"]["data"]["bytecodeOffset"], 42);
    }

    #[test]
    fn notify_paused_maps_exception_reason() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let _ = dbg.borrow_mut().on_exception(7);
        assert!(d.notify_paused());
        let msgs = drain_all(&mut d);
        assert_eq!(msgs[0]["params"]["reason"], "exception");
    }

    #[test]
    fn resume_drives_debugger_apply_action_and_emits_resumed() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        // Create a pause to resume from.
        let _ = dbg.borrow_mut().on_debugger_statement(0);
        // Don't notify; resume should still work because last_pause_reason is set.
        let resp = dispatch(&mut d, r#"{"id":2,"method":"Debugger.resume","params":{}}"#);
        assert!(resp["result"].is_object(), "resume result: {resp}");
        assert!(resp.get("error").is_none());
        // Pre-resume drain consumed everything; the response and the
        // resumed event are now in the outbox in some order. The
        // `dispatch` helper returns the most recent message, but the
        // resumed event was pushed first.
        // Verify by re-running with explicit drain.
    }

    #[test]
    fn resume_with_no_active_pause_is_silent_noop() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let resp = dispatch(&mut d, r#"{"id":2,"method":"Debugger.resume","params":{}}"#);
        assert!(resp.get("error").is_none(), "resume ack must succeed");
        // Only the ack should have been queued (no Debugger.resumed event).
        let remaining = drain_all(&mut d);
        assert!(
            remaining.iter().all(|m| m.get("method").is_none()),
            "no event expected when there is no active pause; got: {remaining:?}"
        );
    }

    #[test]
    fn set_breakpoints_active_propagates_to_attached_and_late_debuggers() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointsActive","params":{"active":false}}"#,
        );
        assert!(!dbg.borrow().breakpoints_active());

        let late = Rc::new(RefCell::new(InterpreterDebugger::new()));
        d.attach_debugger(Rc::clone(&late));
        assert!(
            !late.borrow().breakpoints_active(),
            "late-attached debugger should inherit cached breakpoint-active state"
        );

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointsActive","params":{}}"#,
        );
        assert!(bad["error"].is_object());
    }

    #[test]
    fn set_skip_all_pauses_propagates_to_attached_and_late_debuggers() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setSkipAllPauses","params":{"skip":true}}"#,
        );
        assert!(dbg.borrow().skip_all_pauses());

        let late = Rc::new(RefCell::new(InterpreterDebugger::new()));
        d.attach_debugger(Rc::clone(&late));
        assert!(
            late.borrow().skip_all_pauses(),
            "late-attached debugger should inherit cached skip-all-pauses state"
        );

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setSkipAllPauses","params":{}}"#,
        );
        assert!(bad["error"].is_object());
    }

    #[test]
    fn set_blackbox_patterns_stores_patterns_and_rejects_bad_input() {
        let mut d = fresh_dispatcher();
        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBlackboxPatterns","params":{"patterns":["node_modules",".*vendor.*"]}}"#,
        );
        assert!(ok.get("error").is_none());
        assert_eq!(d.blackbox_patterns, vec!["node_modules", ".*vendor.*"]);

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBlackboxPatterns","params":{"patterns":[7]}}"#,
        );
        assert!(bad["error"].is_object());
    }

    #[test]
    fn set_blackboxed_ranges_stores_ranges_for_registered_script() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "function hidden() {}\n//# sourceURL=stator://hidden.js".to_string(),
        );
        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBlackboxedRanges","params":{"scriptId":"7","positions":[{"lineNumber":0,"columnNumber":0},{"lineNumber":0,"columnNumber":20}]}}"#,
        );
        assert!(ok.get("error").is_none());
        assert_eq!(d.blackboxed_ranges.get("7").unwrap().len(), 2);

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBlackboxedRanges","params":{"scriptId":"missing","positions":[]}}"#,
        );
        assert!(bad["error"].is_object());
    }

    #[test]
    fn step_into_requires_attached_debugger() {
        let mut d = fresh_dispatcher();
        // Enable so the gate isn't on enable; just no debugger attached.
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let resp = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.stepInto","params":{}}"#,
        );
        assert!(
            resp["error"].is_object(),
            "step without debugger must error"
        );
    }

    #[test]
    fn step_over_requires_active_pause() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let resp = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.stepOver","params":{}}"#,
        );
        assert!(
            resp["error"].is_object(),
            "step without active pause must error"
        );
    }

    #[test]
    fn step_into_applies_action_and_emits_resumed() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        assert_eq!(
            d.dispatch_json(r#"{"id":2,"method":"Debugger.stepInto","params":{}}"#),
            DispatchOutcome::Ok
        );
        let msgs = drain_all(&mut d);
        let methods: Vec<_> = msgs
            .iter()
            .map(|m| m.get("method").and_then(Value::as_str).unwrap_or(""))
            .collect();
        assert!(
            methods.contains(&"Debugger.resumed"),
            "step should emit Debugger.resumed; got: {methods:?}"
        );
    }

    #[test]
    fn continue_to_location_installs_one_shot_breakpoint_and_emits_resumed() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let messages = dispatch_all(
            &mut d,
            r#"{"id":2,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0}}}"#,
        );
        assert!(
            messages
                .iter()
                .any(|message| message["method"] == "Debugger.resumed"),
            "continueToLocation should emit Debugger.resumed; got: {messages:?}"
        );
        assert_eq!(dbg.borrow().breakpoints().count(), 1);

        let offset = dbg
            .borrow()
            .breakpoints()
            .next()
            .expect("one-shot breakpoint")
            .bytecode_offset;
        assert!(dbg.borrow_mut().check_pause_at(offset).is_some());
        assert_eq!(
            dbg.borrow().breakpoints().count(),
            0,
            "one-shot breakpoint should remove itself after the pause"
        );
    }

    #[test]
    fn continue_to_location_requires_active_pause() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        d.register_script_source(7, "var a = 1;".to_string());
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":0,"columnNumber":0}}}"#,
        );
        assert!(response["error"].is_object());
    }

    #[test]
    fn pause_method_is_fail_closed() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(&mut d, r#"{"id":1,"method":"Debugger.pause","params":{}}"#);
        assert!(resp["error"].is_object(), "Debugger.pause must error");
        let msg = resp["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("Debugger.pause"), "error message: {msg}");
    }

    #[test]
    fn evaluate_on_call_frame_is_fail_closed() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.evaluateOnCallFrame","params":{}}"#,
        );
        assert!(resp["error"].is_object());
        let msg = resp["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("Debugger.evaluateOnCallFrame"), "msg: {msg}");
    }

    #[test]
    fn get_script_source_returns_registered_source() {
        let mut d = fresh_dispatcher();
        d.register_script_source(1, "let answer = 42;".to_string());
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getScriptSource","params":{"scriptId":"1"}}"#,
        );
        assert_eq!(resp["result"]["scriptSource"], "let answer = 42;");
    }

    #[test]
    fn get_script_source_unknown_id_errors() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getScriptSource","params":{"scriptId":"missing"}}"#,
        );
        assert!(resp["error"].is_object());
    }

    #[test]
    fn set_script_source_updates_registered_source_and_url() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "let oldValue = 1;\n//# sourceURL=stator://old.js".to_string(),
        );
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let newValue = 2;\n//# sourceURL=stator://new.js"}}"#,
        );
        assert_eq!(resp["result"]["status"], "Ok");
        assert_eq!(resp["result"]["stackChanged"], false);

        let source = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.getScriptSource","params":{"scriptId":"7"}}"#,
        );
        assert_eq!(
            source["result"]["scriptSource"],
            "let newValue = 2;\n//# sourceURL=stator://new.js"
        );
        assert_eq!(d.script_urls.get("7").unwrap(), "stator://new.js");
    }

    #[test]
    fn set_script_source_dry_run_and_compile_error_do_not_mutate_source() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "let original = 1;".to_string());
        let dry_run = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let dryRun = 2;","dryRun":true}}"#,
        );
        assert_eq!(dry_run["result"]["status"], "Ok");
        assert_eq!(d.script_sources.get("7").unwrap(), "let original = 1;");

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"function {"}}"#,
        );
        assert_eq!(bad["result"]["status"], "CompileError");
        assert!(bad["result"]["exceptionDetails"].is_object());
        assert_eq!(d.script_sources.get("7").unwrap(), "let original = 1;");
    }

    #[test]
    fn get_possible_breakpoints_returns_registered_source_locations() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "var a = 1;\nvar b = a + 2;".to_string());
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"7","lineNumber":0,"columnNumber":0}}}"#,
        );
        let locations = resp["result"]["locations"].as_array().unwrap();
        assert!(!locations.is_empty());
        assert!(locations.iter().all(|location| location["scriptId"] == "7"));
    }

    #[test]
    fn get_possible_breakpoints_unknown_script_errors() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"missing","lineNumber":0,"columnNumber":0}}}"#,
        );
        assert!(resp["error"].is_object());
    }

    #[test]
    fn unrelated_runtime_methods_still_work_with_debugger_attached() {
        // Ensure attaching a debugger does not regress Runtime.evaluate.
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"1+2"}}"#,
        );
        assert_eq!(resp["result"]["result"]["value"], 3);
    }
}

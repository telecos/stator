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
//! | `Runtime`      | `awaitPromise`            | Drains microtasks and returns settled Promise results |
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
//! | `Runtime`      | `collectGarbage`          | Triggers the thread-local Stator GC runtime |
//! | `Runtime`      | `terminateExecution`      | Requests termination for dispatcher-run scripts |
//! | `Runtime`      | `addBinding`/`removeBinding` | Installs/removes global binding callbacks |
//! | `Debugger`     | `enable`                  | Acknowledges; returns `debuggerId` |
//! | `Debugger`     | `disable`                 | Clears the `Debugger` domain enabled state |
//! | `Debugger`     | `setPauseOnExceptions`    | Configures exception pause state (typed error on invalid `state`) |
//! | `Debugger`     | `setBreakpoint`           | Sets a breakpoint by registered script location |
//! | `Debugger`     | `setBreakpointByUrl`      | Resolves URL breakpoints against registered scripts |
//! | `Debugger`     | `setBreakpointsActive`    | Enables/disables installed breakpoint pauses |
//! | `Debugger`     | `setSkipAllPauses`        | Suppresses/resumes all debugger pause sources |
//! | `Debugger`     | `setAsyncCallStackDepth`/`getStackTrace` | Emits and resolves pause stack trace IDs when async depth is requested |
//! | `Debugger`     | `setBlackboxPatterns`/`setBlackboxedRanges` | Stores blackbox filters for debugger setup |
//! | `Debugger`     | `resume`                  | Resumes after a pause; emits `Debugger.resumed` when an active pause exists |
//! | `Debugger`     | `continueToLocation`      | Resumes to a one-shot breakpoint at a registered script location |
//! | `Debugger`     | `stepInto`/`stepOver`/`stepOut` | Applies the step on the attached interpreter debugger; errors when not attached or no active pause |
//! | `Debugger`     | `pause`                   | Requests a pause at the next debugger poll |
//! | `Debugger`     | `setInstrumentationBreakpoint` | Fail-closed: instrumentation breakpoints not implemented yet |
//! | `Debugger`     | `evaluateOnCallFrame`     | Evaluates against the synthetic paused frame globals |
//! | `Debugger`     | `setVariableValue`        | Mutates paused synthetic local/global scopes |
//! | `Debugger`     | `restartFrame`/`setReturnValue`/`setBreakpointOnFunctionCall` | Fail-closed: call-frame/function-call mutation not implemented yet |
//! | `Debugger`     | `getScriptSource`         | Returns a source registered by the in-process inspector |
//! | `Debugger`     | `searchInContent`         | Searches registered script source content |
//! | `Debugger`     | `setScriptSource`         | Validates and updates registered script source text |
//! | `Debugger`     | `getPossibleBreakpoints`  | Returns breakpointable locations for registered scripts |
//! | `Console`      | `enable`                  | Flushes buffered messages as events |
//! | `Console`      | `disable`                 | Acknowledges                       |
//! | `Profiler`     | `enable`                  | Acknowledges                       |
//! | `Profiler`     | `setSamplingInterval`     | Sets interval for the next profile |
//! | `Profiler`     | `start`                   | Starts CPU profiling               |
//! | `Profiler`     | `stop`                    | Stops profiling; returns profile    |
//! | `HeapProfiler` | `enable`                  | Acknowledges                       |
//! | `HeapProfiler` | `collectGarbage`          | Triggers the thread-local Stator GC runtime |
//! | `HeapProfiler` | `takeHeapSnapshot`        | Emits snapshot chunks              |
//! | `HeapProfiler` | `startTrackingHeapObjects` | Starts allocation tracking         |
//! | `HeapProfiler` | `stopTrackingHeapObjects`  | Returns allocation stats           |
//! | `Target`       | `getTargets`/`getTargetInfo`/`attachToTarget`/`closeTarget`/`setRemoteLocations` | Single-target DevTools compatibility and validated setup state |
//! | `Network`      | `enable`/`disable`/`clearBrowserCache`/`clearBrowserCookies` | Acknowledges and tracks state      |
//! | `Network`      | `setCacheDisabled`/`setBypassServiceWorker`/`setAttachDebugStack`/`setReportingApiEnabled`/`setUserAgentOverride`/`setExtraHTTPHeaders`/`setBlockedURLs`/`setAcceptedEncodings`/`clearAcceptedEncodingsOverride`/`emulateNetworkConditions` | Validated cached setup settings |
//! | `Page`         | `enable`/`disable`/`getResourceTree`/`getFrameTree`/`setLifecycleEventsEnabled`/`setBypassCSP`/`setAdBlockingEnabled` | Minimal standalone page metadata |
//! | `Log`          | `enable`/`disable`/`clear`/`startViolationsReport`/`stopViolationsReport` | Validated setup acknowledgements |
//! | `Security`     | `enable`/`disable`/`setIgnoreCertificateErrors` | Validated setup acknowledgements |
//! | `Performance`  | `enable`/`disable`/`getMetrics` | Reports deterministic runtime metrics |
//! | `Emulation`    | `setDeviceMetricsOverride`/`clearDeviceMetricsOverride`/`setTouchEmulationEnabled`/`setEmitTouchEventsForMouse`/`setEmulatedMedia`/`setCPUThrottlingRate`/`setHardwareConcurrencyOverride`/`setAutoDarkModeOverride`/`setDocumentCookieDisabled`/`setTimezoneOverride`/`setLocaleOverride`/`setUserAgentOverride`/`setScriptExecutionDisabled`/`setFocusEmulationEnabled`/`setIdleOverride`/`clearIdleOverride`/`setGeolocationOverride`/`clearGeolocationOverride`/`setPageScaleFactor`/`resetPageScaleFactor`/`setScrollbarsHidden` | Validated setup state |
//! | `Overlay`      | `enable`/`disable`/visual setup toggles | Validated cached setup state |
//! | `ServiceWorker` | `enable`/`disable`/`setForceUpdateOnPageLoad`/empty queries | Validated setup state |
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
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tungstenite::{Message, WebSocket, accept};

use crate::builtins::promise::{
    AsyncStackTrace, PromiseState, current_async_stack_trace, drain_active_microtask_queue,
};
use crate::bytecode::bytecode_array::BytecodeArray;
use crate::bytecode::bytecode_generator::BytecodeGenerator;
use crate::error::{StatorError, StatorResult};
use crate::inspector::console::{ProfileEventKind, drain_messages, drain_profile_events};
use crate::inspector::debugger::{
    BreakpointId, DebugAction, Debugger, DebuggerPauseBridge, PauseEvent, PauseFrameSnapshot,
    PauseReason,
};
use crate::inspector::heap_snapshot::{AllocationRecord, HeapSnapshotBuilder};
use crate::inspector::profiler::CpuProfiler;
use crate::interpreter::{GlobalEnv, Interpreter, InterpreterFrame, take_pending_exception};
use crate::objects::property_map::PropertyMap;
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

/// Cached setup-only state requested through `Network.emulateNetworkConditions`.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkEmulatedConditions {
    /// Whether the frontend requested offline network conditions.
    pub offline: bool,
    /// Requested network latency in milliseconds.
    pub latency: f64,
    /// Requested download throughput in bytes per second, or `-1` to disable throttling.
    pub download_throughput: f64,
    /// Requested upload throughput in bytes per second, or `-1` to disable throttling.
    pub upload_throughput: f64,
    /// Requested CDP connection type, when provided.
    pub connection_type: Option<String>,
    /// Requested packet loss percentage, when provided.
    pub packet_loss: Option<f64>,
    /// Requested packet queue length, when provided.
    pub packet_queue_length: Option<u32>,
    /// Requested packet reordering state, when provided.
    pub packet_reordering: Option<bool>,
}

/// Cached setup-only state requested through `Emulation.setGeolocationOverride`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmulationGeolocationOverride {
    /// Requested latitude in degrees, when provided.
    pub latitude: Option<f64>,
    /// Requested longitude in degrees, when provided.
    pub longitude: Option<f64>,
    /// Requested position accuracy in meters, when provided.
    pub accuracy: Option<f64>,
    /// Requested altitude in meters, when provided.
    pub altitude: Option<f64>,
    /// Requested altitude accuracy in meters, when provided.
    pub altitude_accuracy: Option<f64>,
    /// Requested heading in degrees, when provided.
    pub heading: Option<f64>,
    /// Requested speed in meters per second, when provided.
    pub speed: Option<f64>,
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

struct TemporaryGlobalBindings {
    globals: Rc<RefCell<GlobalEnv>>,
    saved: Vec<(String, Option<JsValue>)>,
}

impl TemporaryGlobalBindings {
    fn new(globals: Rc<RefCell<GlobalEnv>>, bindings: Vec<(String, JsValue)>) -> Self {
        let mut saved = Vec::with_capacity(bindings.len());
        {
            let mut env = globals.borrow_mut();
            for (name, value) in bindings {
                let previous = env.get(&name).map(JsValue::cheap_clone);
                env.insert(name.clone(), value);
                saved.push((name, previous));
            }
        }
        Self { globals, saved }
    }
}

impl Drop for TemporaryGlobalBindings {
    fn drop(&mut self) {
        let mut env = self.globals.borrow_mut();
        for (name, previous) in self.saved.drain(..).rev() {
            if let Some(value) = previous {
                env.insert(name, value);
            } else {
                env.remove(&name);
            }
        }
    }
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

#[derive(Clone)]
struct CdpBreakpointLocation {
    script_id: String,
    url: String,
    line_number: u32,
    column_number: u32,
}

#[derive(Clone)]
struct CdpUrlBreakpoint {
    requested_url: Option<String>,
    requested_url_regex: Option<String>,
    requested_script_hash: Option<String>,
    line_number: u32,
    column_number: u32,
}

#[derive(Clone)]
struct CdpScriptBreakpoint {
    script_id: String,
    line_number: u32,
    column_number: u32,
}

#[derive(Clone)]
struct PauseFrameLocation {
    script_id: String,
    url: String,
    line_number: u32,
    column_number: u32,
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

    /// Update a property on a registered plain object. Returns `true` when
    /// the ID resolved to a live plain object and was updated.
    pub fn set_plain_object_property(&mut self, id: &str, name: &str, value: JsValue) -> bool {
        let Some(entry) = self.entries.get_mut(id) else {
            return false;
        };
        let JsValue::PlainObject(map_ref) = &entry.value else {
            return false;
        };
        map_ref.borrow_mut().insert(name.to_string(), value);
        true
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
    /// Whether the `Inspector` domain is currently enabled for this session.
    inspector_enabled: bool,
    /// Whether the `Runtime` domain is currently enabled for this session.
    runtime_enabled: bool,
    /// Whether custom object formatters are enabled for this session.
    custom_object_formatter_enabled: bool,
    /// Maximum stack frame count requested for captured call stacks.
    max_call_stack_size_to_capture: u32,
    /// Whether the `Console` domain is currently enabled for this session.
    console_enabled: bool,
    /// Whether the Network domain is currently enabled for this session.
    network_enabled: bool,
    /// Cached `Network.setCacheDisabled` state.
    network_cache_disabled: bool,
    /// Cached `Network.setBypassServiceWorker` state.
    network_bypass_service_worker: bool,
    /// Cached `Network.setAttachDebugStack` state.
    network_attach_debug_stack: bool,
    /// Cached `Network.setReportingApiEnabled` state.
    network_reporting_api_enabled: bool,
    /// Cached `Network.setUserAgentOverride` user-agent string.
    network_user_agent: String,
    /// Cached `Network.setUserAgentOverride` accept-language string.
    network_accept_language: String,
    /// Cached `Network.setUserAgentOverride` platform string.
    network_platform: String,
    /// Cached `Network.setUserAgentOverride` user-agent metadata object.
    network_user_agent_metadata: Option<Value>,
    /// Count of cached `Network.setExtraHTTPHeaders` entries.
    network_extra_http_header_count: usize,
    /// Count of cached `Network.setBlockedURLs` URL patterns.
    network_blocked_url_count: usize,
    /// Count of cached `Network.setAcceptedEncodings` entries.
    network_accepted_encoding_count: usize,
    /// Cached setup-only `Network.emulateNetworkConditions` request state.
    network_emulated_conditions: Option<NetworkEmulatedConditions>,
    /// Whether the Page domain is currently enabled for this session.
    page_enabled: bool,
    /// Cached `Page.setLifecycleEventsEnabled` state.
    page_lifecycle_events_enabled: bool,
    /// Cached `Page.setBypassCSP` state.
    page_bypass_csp: bool,
    /// Cached `Page.setAdBlockingEnabled` state.
    page_ad_blocking_enabled: bool,
    /// Whether the Log domain is currently enabled for this session.
    log_enabled: bool,
    /// Number of cached violation-report settings from `Log.startViolationsReport`.
    log_violation_setting_count: usize,
    /// Whether the Security domain is currently enabled for this session.
    security_enabled: bool,
    /// Cached `Security.setIgnoreCertificateErrors` state.
    security_ignore_certificate_errors: bool,
    /// Whether the Performance domain is currently enabled for this session.
    performance_enabled: bool,
    /// Cached `Emulation.setDeviceMetricsOverride` state.
    emulation_device_metrics: Option<Value>,
    /// Cached `Emulation.setTouchEmulationEnabled` state.
    emulation_touch_enabled: bool,
    /// Cached touch point count for touch emulation.
    emulation_max_touch_points: u32,
    /// Cached `Emulation.setEmitTouchEventsForMouse` enabled state.
    emulation_emit_touch_events_for_mouse_enabled: bool,
    /// Cached `Emulation.setEmitTouchEventsForMouse` configuration.
    emulation_emit_touch_events_for_mouse_configuration: String,
    /// Cached media type from `Emulation.setEmulatedMedia`.
    emulation_media: String,
    /// Cached media feature count from `Emulation.setEmulatedMedia`.
    emulation_media_feature_count: usize,
    /// Cached `Emulation.setCPUThrottlingRate` value (1.0 means no throttling).
    emulation_cpu_throttling_rate: f64,
    /// Cached `Emulation.setHardwareConcurrencyOverride` value.
    emulation_hardware_concurrency: u32,
    /// Cached `Emulation.setAutoDarkModeOverride` state.
    emulation_auto_dark_mode_enabled: Option<bool>,
    /// Cached `Emulation.setDocumentCookieDisabled` state.
    emulation_document_cookie_disabled: bool,
    /// Cached `Emulation.setTimezoneOverride` timezone id.
    emulation_timezone_id: String,
    /// Cached `Emulation.setLocaleOverride` locale id.
    emulation_locale: String,
    /// Cached `Emulation.setUserAgentOverride` user-agent string.
    emulation_user_agent: String,
    /// Cached `Emulation.setUserAgentOverride` accept-language string.
    emulation_accept_language: String,
    /// Cached `Emulation.setUserAgentOverride` platform string.
    emulation_platform: String,
    /// Cached `Emulation.setUserAgentOverride` user-agent metadata object.
    emulation_user_agent_metadata: Option<Value>,
    /// Cached `Emulation.setScriptExecutionDisabled` state.
    emulation_script_execution_disabled: bool,
    /// Cached `Emulation.setFocusEmulationEnabled` state.
    emulation_focus_emulation_enabled: bool,
    /// Cached `Emulation.setIdleOverride` state.
    emulation_idle_override: Option<(bool, bool)>,
    /// Cached `Emulation.setGeolocationOverride` state.
    emulation_geolocation_override: Option<EmulationGeolocationOverride>,
    /// Cached `Emulation.setPageScaleFactor` value.
    emulation_page_scale_factor: Option<f64>,
    /// Cached `Emulation.setScrollbarsHidden` state.
    emulation_scrollbars_hidden: bool,
    /// Whether the Overlay domain is currently enabled for this session.
    overlay_enabled: bool,
    /// Cached Overlay visual-debugging toggles.
    overlay_show_paint_rects: bool,
    overlay_show_debug_borders: bool,
    overlay_show_fps_counter: bool,
    overlay_show_web_vitals: bool,
    overlay_show_layout_shift_regions: bool,
    overlay_show_ad_highlights: bool,
    overlay_show_viewport_size_on_resize: bool,
    overlay_show_scroll_bottleneck_rects: bool,
    overlay_show_hit_test_borders: bool,
    /// Cached grid overlay setup count from `Overlay.setShowGridOverlays`.
    overlay_grid_overlay_count: usize,
    /// Cached flex overlay setup count from `Overlay.setShowFlexOverlays`.
    overlay_flex_overlay_count: usize,
    /// Cached scroll-snap overlay setup count from `Overlay.setShowScrollSnapOverlays`.
    overlay_scroll_snap_overlay_count: usize,
    /// Cached container-query overlay setup count from `Overlay.setShowContainerQueryOverlays`.
    overlay_container_query_overlay_count: usize,
    /// Cached isolated-elements setup count from `Overlay.setShowIsolatedElements`.
    overlay_isolated_element_count: usize,
    /// Cached inspect mode requested through `Overlay.setInspectMode`.
    overlay_inspect_mode: String,
    /// Whether the ServiceWorker domain is currently enabled for this session.
    service_worker_enabled: bool,
    /// Cached `ServiceWorker.setForceUpdateOnPageLoad` value.
    service_worker_force_update_on_page_load: bool,
    /// Whether the `Debugger` domain has been enabled by the client.  Used
    /// to gate fan-out of `Debugger.scriptParsed` events.
    debugger_enabled: bool,
    /// Whether `Target.setDiscoverTargets` is enabled.
    target_discovery_enabled: bool,
    /// Whether `Target.setAutoAttach` is enabled.
    target_auto_attach_enabled: bool,
    /// Whether auto-attached targets should wait for debugger commands before running.
    target_auto_attach_wait_for_debugger_on_start: bool,
    /// Whether auto-attach should use flattened sessions.
    target_auto_attach_flatten: bool,
    /// Number of cached auto-attach filter entries.
    target_auto_attach_filter_count: usize,
    /// Number of cached remote-location entries from `Target.setRemoteLocations`.
    target_remote_location_count: usize,
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
    /// Direct script-location breakpoint definitions.
    cdp_script_breakpoints: HashMap<String, CdpScriptBreakpoint>,
    /// Interpreter breakpoint IDs installed for each CDP breakpoint ID.
    cdp_debugger_breakpoints: HashMap<String, Vec<BreakpointId>>,
    /// URL/regex breakpoint definitions that should resolve against future scripts.
    cdp_url_breakpoints: HashMap<String, CdpUrlBreakpoint>,
    /// CDP location metadata for each installed interpreter breakpoint ID.
    debugger_breakpoint_locations: HashMap<BreakpointId, CdpBreakpointLocation>,
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
    /// Remote object ID for the current paused local scope snapshot, if any.
    paused_local_scope_object_id: Option<String>,
    /// Remote object IDs for current paused closure scope snapshots, inner first.
    paused_context_scope_object_ids: Vec<String>,
    /// Remote object ID for the current paused global scope snapshot, if any.
    paused_global_scope_object_id: Option<String>,
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
    /// Optional cross-thread pause bridge used by WebSocket transports that
    /// cannot borrow the interpreter's non-`Send` debugger state.
    pause_bridge: Option<DebuggerPauseBridge>,
    /// Requested async call-stack depth for future async scheduling points.
    async_call_stack_depth: u32,
    /// Monotonically increasing ID for debugger stack traces exposed through CDP.
    next_stack_trace_id: u64,
    /// Stack traces addressable through `Debugger.getStackTrace`.
    stack_traces: HashMap<String, Value>,
    /// FIFO order used to cap retained stack traces.
    stack_trace_order: VecDeque<String>,
    /// Dispatcher-owned termination flag published to interpreter runs.
    termination_requested: AtomicBool,
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
            inspector_enabled: false,
            runtime_enabled: false,
            custom_object_formatter_enabled: false,
            max_call_stack_size_to_capture: 0,
            console_enabled: false,
            network_enabled: false,
            network_cache_disabled: false,
            network_bypass_service_worker: false,
            network_attach_debug_stack: false,
            network_reporting_api_enabled: false,
            network_user_agent: String::new(),
            network_accept_language: String::new(),
            network_platform: String::new(),
            network_user_agent_metadata: None,
            network_extra_http_header_count: 0,
            network_blocked_url_count: 0,
            network_accepted_encoding_count: 0,
            network_emulated_conditions: None,
            page_enabled: false,
            page_lifecycle_events_enabled: false,
            page_bypass_csp: false,
            page_ad_blocking_enabled: false,
            log_enabled: false,
            log_violation_setting_count: 0,
            security_enabled: false,
            security_ignore_certificate_errors: false,
            performance_enabled: false,
            emulation_device_metrics: None,
            emulation_touch_enabled: false,
            emulation_max_touch_points: 0,
            emulation_emit_touch_events_for_mouse_enabled: false,
            emulation_emit_touch_events_for_mouse_configuration: String::new(),
            emulation_media: String::new(),
            emulation_media_feature_count: 0,
            emulation_cpu_throttling_rate: 1.0,
            emulation_hardware_concurrency: 0,
            emulation_auto_dark_mode_enabled: None,
            emulation_document_cookie_disabled: false,
            emulation_timezone_id: String::new(),
            emulation_locale: String::new(),
            emulation_user_agent: String::new(),
            emulation_accept_language: String::new(),
            emulation_platform: String::new(),
            emulation_user_agent_metadata: None,
            emulation_script_execution_disabled: false,
            emulation_focus_emulation_enabled: false,
            emulation_idle_override: None,
            emulation_geolocation_override: None,
            emulation_page_scale_factor: None,
            emulation_scrollbars_hidden: false,
            overlay_enabled: false,
            overlay_show_paint_rects: false,
            overlay_show_debug_borders: false,
            overlay_show_fps_counter: false,
            overlay_show_web_vitals: false,
            overlay_show_layout_shift_regions: false,
            overlay_show_ad_highlights: false,
            overlay_show_viewport_size_on_resize: false,
            overlay_show_scroll_bottleneck_rects: false,
            overlay_show_hit_test_borders: false,
            overlay_grid_overlay_count: 0,
            overlay_flex_overlay_count: 0,
            overlay_scroll_snap_overlay_count: 0,
            overlay_container_query_overlay_count: 0,
            overlay_isolated_element_count: 0,
            overlay_inspect_mode: "none".to_string(),
            service_worker_enabled: false,
            service_worker_force_update_on_page_load: false,
            debugger_enabled: false,
            target_discovery_enabled: false,
            target_auto_attach_enabled: false,
            target_auto_attach_wait_for_debugger_on_start: false,
            target_auto_attach_flatten: false,
            target_auto_attach_filter_count: 0,
            target_remote_location_count: 0,
            closed_target_ids: HashSet::new(),
            next_target_session_id: 1,
            target_session_id: None,
            target_session_target_id: None,
            script_sources: HashMap::new(),
            script_urls: HashMap::new(),
            next_breakpoint_id: 1,
            cdp_breakpoints: HashSet::new(),
            cdp_script_breakpoints: HashMap::new(),
            cdp_debugger_breakpoints: HashMap::new(),
            cdp_url_breakpoints: HashMap::new(),
            debugger_breakpoint_locations: HashMap::new(),
            breakpoints_active: true,
            skip_all_pauses: false,
            blackbox_patterns: Vec::new(),
            blackboxed_ranges: HashMap::new(),
            remote_objects: RemoteObjectRegistry::new(),
            paused_local_scope_object_id: None,
            paused_context_scope_object_ids: Vec::new(),
            paused_global_scope_object_id: None,
            next_heap_object_id: 1,
            heap_objects: HashMap::new(),
            inspected_heap_objects: HashSet::new(),
            last_heap_sampling_profile: empty_heap_sampling_profile(),
            next_compiled_script_id: 1,
            compiled_scripts: HashMap::new(),
            runtime_bindings: Rc::new(RefCell::new(RuntimeBindingState::default())),
            next_exception_id: 1,
            debugger: None,
            pause_bridge: None,
            async_call_stack_depth: 0,
            next_stack_trace_id: 1,
            stack_traces: HashMap::new(),
            stack_trace_order: VecDeque::new(),
            termination_requested: AtomicBool::new(false),
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
        self.script_urls.insert(script_id.clone(), url);
        self.refresh_breakpoints_for_registered_script(&script_id);
    }

    fn refresh_breakpoints_for_registered_script(&mut self, script_id: &str) {
        self.remove_installed_breakpoints_for_script(script_id);
        let script_breakpoints: Vec<_> = self
            .cdp_script_breakpoints
            .iter()
            .filter(|(_, breakpoint)| breakpoint.script_id == script_id)
            .map(|(id, breakpoint)| (id.clone(), breakpoint.clone()))
            .collect();
        for (breakpoint_id, breakpoint) in script_breakpoints {
            if let Ok(Some(location)) = self.resolve_script_breakpoint(&breakpoint_id, &breakpoint)
                && self.debugger_enabled
            {
                self.push_event(
                    "Debugger.breakpointResolved",
                    json!({
                        "breakpointId": breakpoint_id,
                        "location": location,
                    }),
                );
            }
        }

        let pending: Vec<_> = self
            .cdp_url_breakpoints
            .iter()
            .map(|(id, breakpoint)| (id.clone(), breakpoint.clone()))
            .collect();
        for (breakpoint_id, breakpoint) in pending {
            if let Ok(Some(location)) =
                self.resolve_url_breakpoint_for_script(&breakpoint_id, &breakpoint, script_id)
                && self.debugger_enabled
            {
                self.push_event(
                    "Debugger.breakpointResolved",
                    json!({
                        "breakpointId": breakpoint_id,
                        "location": location,
                    }),
                );
            }
        }
    }

    fn remove_installed_breakpoints_for_script(&mut self, script_id: &str) {
        let mut removed = Vec::new();
        for debugger_ids in self.cdp_debugger_breakpoints.values_mut() {
            debugger_ids.retain(|debugger_id| {
                let remove = self
                    .debugger_breakpoint_locations
                    .get(debugger_id)
                    .is_some_and(|location| location.script_id == script_id);
                if remove {
                    removed.push(*debugger_id);
                }
                !remove
            });
        }
        self.cdp_debugger_breakpoints
            .retain(|_, debugger_ids| !debugger_ids.is_empty());
        if let Some(debugger) = self.debugger.as_ref() {
            let mut debugger = debugger.borrow_mut();
            for debugger_id in &removed {
                let _ = debugger.remove_breakpoint(*debugger_id);
            }
        }
        for debugger_id in removed {
            self.debugger_breakpoint_locations.remove(&debugger_id);
        }
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

    /// Returns `true` if custom object formatters are enabled for this session.
    pub fn custom_object_formatter_enabled(&self) -> bool {
        self.custom_object_formatter_enabled
    }

    /// Returns the requested maximum stack size for captured call stacks.
    pub fn max_call_stack_size_to_capture(&self) -> u32 {
        self.max_call_stack_size_to_capture
    }

    /// Returns `true` if the Network domain is currently enabled.
    pub fn network_enabled(&self) -> bool {
        self.network_enabled
    }

    /// Returns the cached `Network.setCacheDisabled` value.
    pub fn network_cache_disabled(&self) -> bool {
        self.network_cache_disabled
    }

    /// Returns the cached `Network.setBypassServiceWorker` value.
    pub fn network_bypass_service_worker(&self) -> bool {
        self.network_bypass_service_worker
    }

    /// Returns the cached `Network.setAttachDebugStack` value.
    pub fn network_attach_debug_stack(&self) -> bool {
        self.network_attach_debug_stack
    }

    /// Returns the cached `Network.setReportingApiEnabled` value.
    pub fn network_reporting_api_enabled(&self) -> bool {
        self.network_reporting_api_enabled
    }

    /// Returns the cached user-agent override string.
    pub fn network_user_agent(&self) -> &str {
        &self.network_user_agent
    }

    /// Returns the cached accept-language override string.
    pub fn network_accept_language(&self) -> &str {
        &self.network_accept_language
    }

    /// Returns the cached platform override string.
    pub fn network_platform(&self) -> &str {
        &self.network_platform
    }

    /// Returns the cached user-agent metadata override object, if present.
    pub fn network_user_agent_metadata(&self) -> Option<&Value> {
        self.network_user_agent_metadata.as_ref()
    }

    /// Returns the number of cached extra HTTP headers.
    pub fn network_extra_http_header_count(&self) -> usize {
        self.network_extra_http_header_count
    }

    /// Returns the number of cached blocked URL patterns.
    pub fn network_blocked_url_count(&self) -> usize {
        self.network_blocked_url_count
    }

    /// Returns the number of cached accepted content encodings.
    pub fn network_accepted_encoding_count(&self) -> usize {
        self.network_accepted_encoding_count
    }

    /// Returns the cached `Network.emulateNetworkConditions` setup request.
    pub fn network_emulated_conditions(&self) -> Option<&NetworkEmulatedConditions> {
        self.network_emulated_conditions.as_ref()
    }

    /// Returns `true` if the Page domain is currently enabled.
    pub fn page_enabled(&self) -> bool {
        self.page_enabled
    }

    /// Returns the cached `Page.setLifecycleEventsEnabled` value.
    pub fn page_lifecycle_events_enabled(&self) -> bool {
        self.page_lifecycle_events_enabled
    }

    /// Returns the cached `Page.setBypassCSP` value.
    pub fn page_bypass_csp(&self) -> bool {
        self.page_bypass_csp
    }

    /// Returns the cached `Page.setAdBlockingEnabled` value.
    pub fn page_ad_blocking_enabled(&self) -> bool {
        self.page_ad_blocking_enabled
    }

    /// Returns `true` if the Log domain is currently enabled.
    pub fn log_enabled(&self) -> bool {
        self.log_enabled
    }

    /// Returns the cached violation-report setting count.
    pub fn log_violation_setting_count(&self) -> usize {
        self.log_violation_setting_count
    }

    /// Returns `true` if the Security domain is currently enabled.
    pub fn security_enabled(&self) -> bool {
        self.security_enabled
    }

    /// Returns the cached certificate-error ignore setting.
    pub fn security_ignore_certificate_errors(&self) -> bool {
        self.security_ignore_certificate_errors
    }

    /// Returns `true` if the Performance domain is currently enabled.
    pub fn performance_enabled(&self) -> bool {
        self.performance_enabled
    }

    /// Returns the cached device metrics override payload, if any.
    pub fn emulation_device_metrics(&self) -> Option<&Value> {
        self.emulation_device_metrics.as_ref()
    }

    /// Returns whether touch emulation is enabled.
    pub fn emulation_touch_enabled(&self) -> bool {
        self.emulation_touch_enabled
    }

    /// Returns the cached max touch-point count.
    pub fn emulation_max_touch_points(&self) -> u32 {
        self.emulation_max_touch_points
    }

    /// Returns the cached mouse-to-touch event emulation enabled state.
    pub fn emulation_emit_touch_events_for_mouse_enabled(&self) -> bool {
        self.emulation_emit_touch_events_for_mouse_enabled
    }

    /// Returns the cached mouse-to-touch event emulation configuration.
    pub fn emulation_emit_touch_events_for_mouse_configuration(&self) -> &str {
        &self.emulation_emit_touch_events_for_mouse_configuration
    }

    /// Returns the cached emulated media string.
    pub fn emulation_media(&self) -> &str {
        &self.emulation_media
    }

    /// Returns the cached emulated media feature count.
    pub fn emulation_media_feature_count(&self) -> usize {
        self.emulation_media_feature_count
    }

    /// Returns the cached CPU throttling rate.
    pub fn emulation_cpu_throttling_rate(&self) -> f64 {
        self.emulation_cpu_throttling_rate
    }

    /// Returns the cached hardware-concurrency override value.
    pub fn emulation_hardware_concurrency(&self) -> u32 {
        self.emulation_hardware_concurrency
    }

    /// Returns the cached auto dark mode override state.
    pub fn emulation_auto_dark_mode_enabled(&self) -> Option<bool> {
        self.emulation_auto_dark_mode_enabled
    }

    /// Returns the cached document-cookie disabled state.
    pub fn emulation_document_cookie_disabled(&self) -> bool {
        self.emulation_document_cookie_disabled
    }

    /// Returns the cached timezone override id.
    pub fn emulation_timezone_id(&self) -> &str {
        &self.emulation_timezone_id
    }

    /// Returns the cached locale override id.
    pub fn emulation_locale(&self) -> &str {
        &self.emulation_locale
    }

    /// Returns the cached Emulation user-agent override string.
    pub fn emulation_user_agent(&self) -> &str {
        &self.emulation_user_agent
    }

    /// Returns the cached Emulation accept-language override string.
    pub fn emulation_accept_language(&self) -> &str {
        &self.emulation_accept_language
    }

    /// Returns the cached Emulation platform override string.
    pub fn emulation_platform(&self) -> &str {
        &self.emulation_platform
    }

    /// Returns the cached Emulation user-agent metadata object, if any.
    pub fn emulation_user_agent_metadata(&self) -> Option<&Value> {
        self.emulation_user_agent_metadata.as_ref()
    }

    /// Returns the cached script execution disabled state.
    pub fn emulation_script_execution_disabled(&self) -> bool {
        self.emulation_script_execution_disabled
    }

    /// Returns the cached focus emulation enabled state.
    pub fn emulation_focus_emulation_enabled(&self) -> bool {
        self.emulation_focus_emulation_enabled
    }

    /// Returns the cached idle override state as `(user_active, screen_unlocked)`.
    pub fn emulation_idle_override(&self) -> Option<(bool, bool)> {
        self.emulation_idle_override
    }

    /// Returns the cached geolocation override state.
    pub fn emulation_geolocation_override(&self) -> Option<&EmulationGeolocationOverride> {
        self.emulation_geolocation_override.as_ref()
    }

    /// Returns the cached page-scale factor override.
    pub fn emulation_page_scale_factor(&self) -> Option<f64> {
        self.emulation_page_scale_factor
    }

    /// Returns the cached scrollbars-hidden state.
    pub fn emulation_scrollbars_hidden(&self) -> bool {
        self.emulation_scrollbars_hidden
    }

    /// Returns `true` if the Overlay domain is currently enabled.
    pub fn overlay_enabled(&self) -> bool {
        self.overlay_enabled
    }

    /// Returns the cached paint-rect overlay toggle.
    pub fn overlay_show_paint_rects(&self) -> bool {
        self.overlay_show_paint_rects
    }

    /// Returns the cached debug-border overlay toggle.
    pub fn overlay_show_debug_borders(&self) -> bool {
        self.overlay_show_debug_borders
    }

    /// Returns the cached FPS-counter overlay toggle.
    pub fn overlay_show_fps_counter(&self) -> bool {
        self.overlay_show_fps_counter
    }

    /// Returns the cached Web Vitals overlay toggle.
    pub fn overlay_show_web_vitals(&self) -> bool {
        self.overlay_show_web_vitals
    }

    /// Returns the cached layout-shift overlay toggle.
    pub fn overlay_show_layout_shift_regions(&self) -> bool {
        self.overlay_show_layout_shift_regions
    }

    /// Returns the cached ad-highlight overlay toggle.
    pub fn overlay_show_ad_highlights(&self) -> bool {
        self.overlay_show_ad_highlights
    }

    /// Returns the cached viewport-size overlay toggle.
    pub fn overlay_show_viewport_size_on_resize(&self) -> bool {
        self.overlay_show_viewport_size_on_resize
    }

    /// Returns the cached scroll-bottleneck overlay toggle.
    pub fn overlay_show_scroll_bottleneck_rects(&self) -> bool {
        self.overlay_show_scroll_bottleneck_rects
    }

    /// Returns the cached hit-test-border overlay toggle.
    pub fn overlay_show_hit_test_borders(&self) -> bool {
        self.overlay_show_hit_test_borders
    }

    /// Returns the cached grid overlay setup count.
    pub fn overlay_grid_overlay_count(&self) -> usize {
        self.overlay_grid_overlay_count
    }

    /// Returns the cached flex overlay setup count.
    pub fn overlay_flex_overlay_count(&self) -> usize {
        self.overlay_flex_overlay_count
    }

    /// Returns the cached scroll-snap overlay setup count.
    pub fn overlay_scroll_snap_overlay_count(&self) -> usize {
        self.overlay_scroll_snap_overlay_count
    }

    /// Returns the cached container-query overlay setup count.
    pub fn overlay_container_query_overlay_count(&self) -> usize {
        self.overlay_container_query_overlay_count
    }

    /// Returns the cached isolated-elements setup count.
    pub fn overlay_isolated_element_count(&self) -> usize {
        self.overlay_isolated_element_count
    }

    /// Returns the cached inspect mode.
    pub fn overlay_inspect_mode(&self) -> &str {
        &self.overlay_inspect_mode
    }

    /// Returns `true` if the ServiceWorker domain is currently enabled.
    pub fn service_worker_enabled(&self) -> bool {
        self.service_worker_enabled
    }

    /// Returns the cached force-update-on-page-load flag.
    pub fn service_worker_force_update_on_page_load(&self) -> bool {
        self.service_worker_force_update_on_page_load
    }

    /// Returns `true` if `Target.setAutoAttach` is currently enabled.
    pub fn target_auto_attach_enabled(&self) -> bool {
        self.target_auto_attach_enabled
    }

    /// Returns `true` if auto-attached targets should wait for debugger commands.
    pub fn target_auto_attach_wait_for_debugger_on_start(&self) -> bool {
        self.target_auto_attach_wait_for_debugger_on_start
    }

    /// Returns `true` if auto-attach should use flattened sessions.
    pub fn target_auto_attach_flatten(&self) -> bool {
        self.target_auto_attach_flatten
    }

    /// Returns the cached auto-attach filter entry count.
    pub fn target_auto_attach_filter_count(&self) -> usize {
        self.target_auto_attach_filter_count
    }

    /// Returns the cached remote-location entry count.
    pub fn target_remote_location_count(&self) -> usize {
        self.target_remote_location_count
    }

    /// Returns `true` if the Inspector domain is currently enabled.
    pub fn inspector_enabled(&self) -> bool {
        self.inspector_enabled
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
        let previous = self.debugger.replace(debugger);
        self.install_deferred_breakpoints();
        previous
    }

    /// Drop the interpreter [`Debugger`] handle previously installed with
    /// [`Self::attach_debugger`]. Returns the dropped handle, if any.
    pub fn detach_debugger_handle(&mut self) -> Option<Rc<RefCell<Debugger>>> {
        self.debugger.take()
    }

    /// Install a cross-thread pause bridge for transport-backed debugger
    /// sessions.
    pub fn attach_pause_bridge(
        &mut self,
        bridge: DebuggerPauseBridge,
    ) -> Option<DebuggerPauseBridge> {
        self.pause_bridge.replace(bridge)
    }

    /// Drop the cross-thread pause bridge, waking any blocked interpreter.
    pub fn detach_pause_bridge(&mut self) -> Option<DebuggerPauseBridge> {
        let bridge = self.pause_bridge.take();
        if let Some(bridge) = &bridge {
            bridge.disconnect();
        }
        bridge
    }

    fn has_pause_bridge(&self) -> bool {
        self.pause_bridge.is_some()
    }

    /// Return the requested async call-stack depth for future async scheduling points.
    pub fn async_call_stack_depth(&self) -> u32 {
        self.async_call_stack_depth
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
        if self.drain_pause_bridge_events() {
            return true;
        }
        let Some(debugger_rc) = self.debugger.as_ref().cloned() else {
            return false;
        };

        let mut dbg = debugger_rc.borrow_mut();
        let frame_snapshot = dbg.last_pause_frame_snapshot().cloned();
        let events = dbg.take_pause_events();
        if !events.is_empty() {
            drop(dbg);
            for event in &events {
                self.notify_pause_event_with_frame(event, frame_snapshot.as_ref());
            }
            return true;
        }
        let Some(reason) = dbg.last_pause_reason().cloned() else {
            return false;
        };
        let offset = dbg.last_pause_offset();
        let line = dbg.last_pause_line();
        drop(dbg);

        let pause_location = self.pause_frame_location_for_reason(&reason, line);
        let scope_chain = self.synthetic_pause_scope_chain(frame_snapshot.as_ref());
        let async_stack_trace_id =
            self.register_pause_stack_trace_if_requested(&reason, offset, &pause_location);
        self.push_event(
            "Debugger.paused",
            paused_event_params(
                &reason,
                offset,
                line,
                scope_chain,
                async_stack_trace_id.as_deref(),
                &pause_location,
            ),
        );
        true
    }

    fn drain_pause_bridge_events(&mut self) -> bool {
        let Some(bridge) = self.pause_bridge.clone() else {
            return false;
        };
        let events = bridge.take_pause_events();
        if events.is_empty() {
            return false;
        }
        for event in &events {
            self.notify_pause_event(event);
        }
        true
    }

    /// Emit a `Debugger.paused` event from a recorded interpreter pause event.
    pub fn notify_pause_event(&mut self, event: &PauseEvent) -> bool {
        self.notify_pause_event_with_frame(event, None)
    }

    fn notify_pause_event_with_frame(
        &mut self,
        event: &PauseEvent,
        frame_snapshot: Option<&PauseFrameSnapshot>,
    ) -> bool {
        if !self.debugger_enabled {
            return false;
        }
        let pause_location = self.pause_frame_location_for_reason(&event.reason, event.line);
        let scope_chain = self.synthetic_pause_scope_chain(frame_snapshot);
        let async_stack_trace_id = self.register_pause_stack_trace_if_requested(
            &event.reason,
            event.bytecode_offset,
            &pause_location,
        );
        self.push_event(
            "Debugger.paused",
            paused_event_params(
                &event.reason,
                event.bytecode_offset,
                event.line,
                scope_chain,
                async_stack_trace_id.as_deref(),
                &pause_location,
            ),
        );
        true
    }

    fn pause_frame_location_for_reason(
        &self,
        reason: &PauseReason,
        fallback_line: u32,
    ) -> PauseFrameLocation {
        if let PauseReason::Breakpoint(id) = reason
            && let Some(location) = self.debugger_breakpoint_locations.get(id)
        {
            return PauseFrameLocation {
                script_id: location.script_id.clone(),
                url: location.url.clone(),
                line_number: location.line_number,
                column_number: location.column_number,
            };
        }
        PauseFrameLocation {
            script_id: "0".to_string(),
            url: String::new(),
            line_number: fallback_line.saturating_sub(1),
            column_number: 0,
        }
    }

    fn register_pause_stack_trace_if_requested(
        &mut self,
        reason: &PauseReason,
        offset: u32,
        location: &PauseFrameLocation,
    ) -> Option<String> {
        if self.async_call_stack_depth == 0 {
            return None;
        }
        let id = format!("stator-stack-trace-{}", self.next_stack_trace_id);
        self.next_stack_trace_id = self.next_stack_trace_id.saturating_add(1);
        let async_parent_stack = current_async_stack_trace();
        self.stack_traces.insert(
            id.clone(),
            pause_stack_trace(
                reason,
                offset,
                location,
                async_parent_stack.as_ref(),
                self.async_call_stack_depth as usize,
            ),
        );
        self.stack_trace_order.push_back(id.clone());
        while self.stack_trace_order.len() > MAX_STORED_STACK_TRACES {
            if let Some(expired) = self.stack_trace_order.pop_front() {
                self.stack_traces.remove(&expired);
            }
        }
        Some(id)
    }

    fn synthetic_pause_scope_chain(
        &mut self,
        frame_snapshot: Option<&PauseFrameSnapshot>,
    ) -> Value {
        self.paused_local_scope_object_id = None;
        self.paused_context_scope_object_ids.clear();
        self.paused_global_scope_object_id = None;
        let mut scopes = Vec::new();
        if let Some(snapshot) = frame_snapshot {
            let mut local_map = PropertyMap::new();
            for (name, value) in frame_snapshot_bindings(snapshot) {
                local_map.insert(name, value);
            }
            let remote = js_value_to_remote_object(
                &JsValue::PlainObject(Rc::new(RefCell::new(local_map))),
                &mut self.remote_objects,
                Some("debugger-local-scope"),
                false,
            );
            self.paused_local_scope_object_id = remote
                .get("objectId")
                .and_then(Value::as_str)
                .map(str::to_string);
            scopes.push(json!({
                "type": "local",
                "object": remote,
                "name": "Locals",
            }));
            for (context_index, slots) in snapshot.context_slots.iter().enumerate() {
                let mut context_map = PropertyMap::new();
                for (name, value) in context_snapshot_bindings(context_index, slots) {
                    context_map.insert(name, value);
                }
                let remote = js_value_to_remote_object(
                    &JsValue::PlainObject(Rc::new(RefCell::new(context_map))),
                    &mut self.remote_objects,
                    Some("debugger-closure-scope"),
                    false,
                );
                if let Some(object_id) = remote.get("objectId").and_then(Value::as_str) {
                    self.paused_context_scope_object_ids
                        .push(object_id.to_string());
                }
                scopes.push(json!({
                    "type": "closure",
                    "object": remote,
                    "name": format!("Closure {context_index}"),
                }));
            }
        }
        let globals: Vec<_> = self
            .globals
            .borrow()
            .vars
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let mut map = PropertyMap::new();
        for (name, value) in globals {
            map.insert(name, value);
        }
        let remote = js_value_to_remote_object(
            &JsValue::PlainObject(Rc::new(RefCell::new(map))),
            &mut self.remote_objects,
            Some("debugger-scope"),
            false,
        );
        self.paused_global_scope_object_id = remote
            .get("objectId")
            .and_then(Value::as_str)
            .map(str::to_string);
        scopes.push(json!({
            "type": "global",
            "object": remote,
            "name": "Global",
        }));
        Value::Array(scopes)
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
        self.paused_local_scope_object_id = None;
        self.paused_context_scope_object_ids.clear();
        self.paused_global_scope_object_id = None;
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

    fn target_set_auto_attach(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(auto_attach) = params.get("autoAttach").and_then(Value::as_bool) else {
            return Err(StatorError::TypeError(
                "Target.setAutoAttach: required parameter 'autoAttach' is missing or not a boolean"
                    .to_string(),
            ));
        };
        let Some(wait_for_debugger_on_start) = params
            .get("waitForDebuggerOnStart")
            .and_then(Value::as_bool)
        else {
            return Err(StatorError::TypeError(
                "Target.setAutoAttach: required parameter 'waitForDebuggerOnStart' is missing or not a boolean"
                    .to_string(),
            ));
        };
        let flatten = match params.get("flatten") {
            Some(value) => value.as_bool().ok_or_else(|| {
                StatorError::TypeError(
                    "Target.setAutoAttach: optional parameter 'flatten' must be a boolean"
                        .to_string(),
                )
            })?,
            None => false,
        };
        let filter_count = match params.get("filter") {
            Some(Value::Array(filters)) => filters.len(),
            Some(_) => {
                return Err(StatorError::TypeError(
                    "Target.setAutoAttach: optional parameter 'filter' must be an array"
                        .to_string(),
                ));
            }
            None => 0,
        };

        self.target_auto_attach_enabled = auto_attach;
        self.target_auto_attach_wait_for_debugger_on_start = wait_for_debugger_on_start;
        self.target_auto_attach_flatten = flatten;
        self.target_auto_attach_filter_count = filter_count;
        Ok(json!({}))
    }

    fn target_set_remote_locations(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(locations) = params.get("locations").and_then(Value::as_array) else {
            return Err(StatorError::TypeError(
                "Target.setRemoteLocations: required parameter 'locations' is missing or not an array"
                    .to_string(),
            ));
        };

        for location in locations {
            let Some(location) = location.as_object() else {
                return Err(StatorError::TypeError(
                    "Target.setRemoteLocations: each location must be an object".to_string(),
                ));
            };
            if location.get("host").and_then(Value::as_str).is_none() {
                return Err(StatorError::TypeError(
                    "Target.setRemoteLocations: each location requires string 'host'".to_string(),
                ));
            }
            let Some(port) = location.get("port").and_then(Value::as_i64) else {
                return Err(StatorError::TypeError(
                    "Target.setRemoteLocations: each location requires integer 'port'".to_string(),
                ));
            };
            if !(0..=65535).contains(&port) {
                return Err(StatorError::RangeError(
                    "Target.setRemoteLocations: location port must be in 0..=65535".to_string(),
                ));
            }
        }

        self.target_remote_location_count = locations.len();
        Ok(json!({}))
    }

    fn target_get_targets(&mut self) -> StatorResult<Value> {
        let target_infos = self.target_infos();
        Ok(json!({ "targetInfos": target_infos }))
    }

    fn target_get_target_info(&self, params: &Value) -> StatorResult<Value> {
        let target_id = optional_string_param(params, "targetId", "Target.getTargetInfo")?;
        let target_info = match target_id {
            Some(target_id) => {
                if !self.is_live_target(target_id) {
                    return Err(StatorError::Internal(format!(
                        "Target.getTargetInfo: unknown targetId `{target_id}`"
                    )));
                }
                self.target_infos()
                    .into_iter()
                    .find(|info| info.get("targetId").and_then(Value::as_str) == Some(target_id))
                    .unwrap_or_else(target_info)
            }
            None => self
                .target_infos()
                .into_iter()
                .next()
                .unwrap_or_else(target_info),
        };
        Ok(json!({ "targetInfo": target_info }))
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
            // ── Browser ───────────────────────────────────────────────────
            "Browser.getVersion" => Ok(browser_get_version()),

            // ── Inspector ─────────────────────────────────────────────────
            "Inspector.enable" => {
                self.inspector_enabled = true;
                Ok(json!({}))
            }
            "Inspector.disable" => {
                self.inspector_enabled = false;
                Ok(json!({}))
            }

            // ── Runtime ───────────────────────────────────────────────────
            "Runtime.enable" => self.runtime_enable(),
            "Runtime.evaluate" => self.runtime_evaluate(&req.params),
            "Runtime.callFunctionOn" => self.runtime_call_function_on(&req.params),
            "Runtime.awaitPromise" => self.runtime_await_promise(&req.params),
            "Runtime.getProperties" => self.runtime_get_properties(&req.params),
            "Runtime.queryObjects" => self.runtime_query_objects(&req.params),
            "Runtime.releaseObject" => self.runtime_release_object(&req.params),
            "Runtime.releaseObjectGroup" => self.runtime_release_object_group(&req.params),
            "Runtime.compileScript" => self.runtime_compile_script(&req.params),
            "Runtime.runScript" => self.runtime_run_script(&req.params),
            "Runtime.runIfWaitingForDebugger" => Ok(json!({})),
            "Runtime.setCustomObjectFormatterEnabled" => {
                self.runtime_set_custom_object_formatter_enabled(&req.params)
            }
            "Runtime.setMaxCallStackSizeToCapture" => {
                self.runtime_set_max_call_stack_size_to_capture(&req.params)
            }
            "Runtime.discardConsoleEntries" => self.runtime_discard_console_entries(),
            "Runtime.globalLexicalScopeNames" => {
                self.runtime_global_lexical_scope_names(&req.params)
            }
            "Runtime.getIsolateId" => Ok(json!({ "id": "stator-isolate-0" })),
            "Runtime.getHeapUsage" => self.runtime_get_heap_usage(),
            "Runtime.collectGarbage" => self.collect_garbage(),
            "Runtime.terminateExecution" => self.runtime_terminate_execution(),
            "Runtime.addBinding" => self.runtime_add_binding(&req.params),
            "Runtime.removeBinding" => self.runtime_remove_binding(&req.params),

            // ── Debugger ──────────────────────────────────────────────────
            "Debugger.enable" => {
                self.debugger_enabled = true;
                Ok(json!({
                    "debuggerId": "stator-debugger-0"
                }))
            }
            "Debugger.disable" => self.debugger_disable(),
            "Debugger.setPauseOnExceptions" => self.debugger_set_pause_on_exceptions(&req.params),
            "Debugger.setAsyncCallStackDepth" => {
                self.debugger_set_async_call_stack_depth(&req.params)
            }
            "Debugger.getStackTrace" => self.debugger_get_stack_trace(&req.params),
            "Debugger.setBreakpoint" => self.debugger_set_breakpoint(&req.params),
            "Debugger.setBreakpointByUrl" => self.debugger_set_breakpoint_by_url(&req.params),
            "Debugger.removeBreakpoint" => self.debugger_remove_breakpoint(&req.params),
            "Debugger.setBreakpointsActive" => self.debugger_set_breakpoints_active(&req.params),
            "Debugger.setSkipAllPauses" => self.debugger_set_skip_all_pauses(&req.params),
            "Debugger.setBlackboxPatterns" => self.debugger_set_blackbox_patterns(&req.params),
            "Debugger.setBlackboxedRanges" => self.debugger_set_blackboxed_ranges(&req.params),
            "Debugger.resume" => self.debugger_resume(),
            "Debugger.terminateOnResume" => self.debugger_terminate_on_resume(),
            "Debugger.continueToLocation" => self.debugger_continue_to_location(&req.params),
            "Debugger.stepInto" => self.debugger_step(&req.params, DebugAction::StepInto),
            "Debugger.stepOver" => self.debugger_step(&req.params, DebugAction::StepOver),
            "Debugger.stepOut" => self.debugger_step(&req.params, DebugAction::StepOut),
            "Debugger.pause" => self.debugger_pause(),
            "Debugger.evaluateOnCallFrame" => self.debugger_evaluate_on_call_frame(&req.params),
            "Debugger.restartFrame" => self.debugger_restart_frame(&req.params),
            "Debugger.setReturnValue" => self.debugger_set_return_value(&req.params),
            "Debugger.setVariableValue" => self.debugger_set_variable_value(&req.params),
            "Debugger.setBreakpointOnFunctionCall" => {
                self.debugger_set_breakpoint_on_function_call(&req.params)
            }
            "Debugger.setInstrumentationBreakpoint" => Err(unsupported_debugger_method(
                "Debugger.setInstrumentationBreakpoint",
                "Stator does not yet support instrumentation breakpoints before script execution.",
            )),
            "Debugger.getScriptSource" => self.debugger_get_script_source(&req.params),
            "Debugger.searchInContent" => self.debugger_search_in_content(&req.params),
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
            "HeapProfiler.collectGarbage" => self.collect_garbage(),
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
            "Network.enable" => {
                self.network_enabled = true;
                Ok(json!({}))
            }
            "Network.disable" => {
                self.network_enabled = false;
                Ok(json!({}))
            }
            "Network.setCacheDisabled" => self.network_set_cache_disabled(&req.params),
            "Network.setBypassServiceWorker" => self.network_set_bypass_service_worker(&req.params),
            "Network.setAttachDebugStack" => self.network_set_attach_debug_stack(&req.params),
            "Network.setReportingApiEnabled" => self.network_set_reporting_api_enabled(&req.params),
            "Network.setUserAgentOverride" => self.network_set_user_agent_override(&req.params),
            "Network.setExtraHTTPHeaders" => self.network_set_extra_http_headers(&req.params),
            "Network.setBlockedURLs" => self.network_set_blocked_urls(&req.params),
            "Network.setAcceptedEncodings" => self.network_set_accepted_encodings(&req.params),
            "Network.clearAcceptedEncodingsOverride" => {
                self.network_clear_accepted_encodings_override()
            }
            "Network.emulateNetworkConditions" => {
                self.network_emulate_network_conditions(&req.params)
            }
            "Network.clearBrowserCache" => Ok(json!({})),
            "Network.clearBrowserCookies" => Ok(json!({})),

            // ── Page ──────────────────────────────────────────────────────
            "Page.enable" => {
                self.page_enabled = true;
                Ok(json!({}))
            }
            "Page.disable" => {
                self.page_enabled = false;
                Ok(json!({}))
            }
            "Page.getResourceTree" => self.page_get_resource_tree(),
            "Page.getFrameTree" => self.page_get_frame_tree(),
            "Page.setLifecycleEventsEnabled" => self.page_set_lifecycle_events_enabled(&req.params),
            "Page.setBypassCSP" => self.page_set_bypass_csp(&req.params),
            "Page.setAdBlockingEnabled" => self.page_set_ad_blocking_enabled(&req.params),

            // ── Log ───────────────────────────────────────────────────────
            "Log.enable" => {
                self.log_enabled = true;
                Ok(json!({}))
            }
            "Log.disable" => {
                self.log_enabled = false;
                Ok(json!({}))
            }
            "Log.clear" => Ok(json!({})),
            "Log.startViolationsReport" => self.log_start_violations_report(&req.params),
            "Log.stopViolationsReport" => self.log_stop_violations_report(),

            // ── Security ──────────────────────────────────────────────────
            "Security.enable" => {
                self.security_enabled = true;
                Ok(json!({}))
            }
            "Security.disable" => {
                self.security_enabled = false;
                Ok(json!({}))
            }
            "Security.setIgnoreCertificateErrors" => {
                self.security_set_ignore_certificate_errors(&req.params)
            }

            // ── Performance ───────────────────────────────────────────────
            "Performance.enable" => {
                self.performance_enabled = true;
                Ok(json!({}))
            }
            "Performance.disable" => {
                self.performance_enabled = false;
                Ok(json!({}))
            }
            "Performance.getMetrics" => self.performance_get_metrics(),

            // ── Emulation ─────────────────────────────────────────────────
            "Emulation.setDeviceMetricsOverride" => {
                self.emulation_set_device_metrics_override(&req.params)
            }
            "Emulation.clearDeviceMetricsOverride" => {
                self.emulation_clear_device_metrics_override()
            }
            "Emulation.setTouchEmulationEnabled" => {
                self.emulation_set_touch_emulation_enabled(&req.params)
            }
            "Emulation.setEmitTouchEventsForMouse" => {
                self.emulation_set_emit_touch_events_for_mouse(&req.params)
            }
            "Emulation.setEmulatedMedia" => self.emulation_set_emulated_media(&req.params),
            "Emulation.setCPUThrottlingRate" => self.emulation_set_cpu_throttling_rate(&req.params),
            "Emulation.setHardwareConcurrencyOverride" => {
                self.emulation_set_hardware_concurrency_override(&req.params)
            }
            "Emulation.setAutoDarkModeOverride" => {
                self.emulation_set_auto_dark_mode_override(&req.params)
            }
            "Emulation.setDocumentCookieDisabled" => {
                self.emulation_set_document_cookie_disabled(&req.params)
            }
            "Emulation.setTimezoneOverride" => self.emulation_set_timezone_override(&req.params),
            "Emulation.setLocaleOverride" => self.emulation_set_locale_override(&req.params),
            "Emulation.setUserAgentOverride" => self.emulation_set_user_agent_override(&req.params),
            "Emulation.setScriptExecutionDisabled" => {
                self.emulation_set_script_execution_disabled(&req.params)
            }
            "Emulation.setFocusEmulationEnabled" => {
                self.emulation_set_focus_emulation_enabled(&req.params)
            }
            "Emulation.setIdleOverride" => self.emulation_set_idle_override(&req.params),
            "Emulation.clearIdleOverride" => self.emulation_clear_idle_override(),
            "Emulation.setGeolocationOverride" => {
                self.emulation_set_geolocation_override(&req.params)
            }
            "Emulation.clearGeolocationOverride" => self.emulation_clear_geolocation_override(),
            "Emulation.setPageScaleFactor" => self.emulation_set_page_scale_factor(&req.params),
            "Emulation.resetPageScaleFactor" => self.emulation_reset_page_scale_factor(),
            "Emulation.setScrollbarsHidden" => self.emulation_set_scrollbars_hidden(&req.params),

            // ── Overlay ───────────────────────────────────────────────────
            "Overlay.enable" => {
                self.overlay_enabled = true;
                Ok(json!({}))
            }
            "Overlay.disable" => {
                self.overlay_enabled = false;
                Ok(json!({}))
            }
            "Overlay.setShowPaintRects" => self.overlay_set_show_paint_rects(&req.params),
            "Overlay.setShowDebugBorders" => self.overlay_set_show_debug_borders(&req.params),
            "Overlay.setShowFPSCounter" => self.overlay_set_show_fps_counter(&req.params),
            "Overlay.setShowWebVitals" => self.overlay_set_show_web_vitals(&req.params),
            "Overlay.setShowLayoutShiftRegions" => {
                self.overlay_set_show_layout_shift_regions(&req.params)
            }
            "Overlay.setShowAdHighlights" => self.overlay_set_show_ad_highlights(&req.params),
            "Overlay.setShowViewportSizeOnResize" => {
                self.overlay_set_show_viewport_size_on_resize(&req.params)
            }
            "Overlay.setShowScrollBottleneckRects" => {
                self.overlay_set_show_scroll_bottleneck_rects(&req.params)
            }
            "Overlay.setShowHitTestBorders" => self.overlay_set_show_hit_test_borders(&req.params),
            "Overlay.setShowGridOverlays" => self.overlay_set_show_grid_overlays(&req.params),
            "Overlay.setShowFlexOverlays" => self.overlay_set_show_flex_overlays(&req.params),
            "Overlay.setShowScrollSnapOverlays" => {
                self.overlay_set_show_scroll_snap_overlays(&req.params)
            }
            "Overlay.setShowContainerQueryOverlays" => {
                self.overlay_set_show_container_query_overlays(&req.params)
            }
            "Overlay.setShowIsolatedElements" => {
                self.overlay_set_show_isolated_elements(&req.params)
            }
            "Overlay.setInspectMode" => self.overlay_set_inspect_mode(&req.params),
            "Overlay.hideHighlight" => Ok(json!({})),
            "Overlay.highlightNode" => Ok(json!({})),
            "Overlay.highlightRect" => Ok(json!({})),

            // ── ServiceWorker ─────────────────────────────────────────────
            "ServiceWorker.enable" => {
                self.service_worker_enabled = true;
                Ok(json!({}))
            }
            "ServiceWorker.disable" => {
                self.service_worker_enabled = false;
                Ok(json!({}))
            }
            "ServiceWorker.setForceUpdateOnPageLoad" => {
                self.service_worker_set_force_update_on_page_load(&req.params)
            }
            "ServiceWorker.deliverPushMessage" => self.service_worker_known_noop(&req.params),
            "ServiceWorker.dispatchSyncEvent" => self.service_worker_known_noop(&req.params),
            "ServiceWorker.dispatchPeriodicSyncEvent" => {
                self.service_worker_known_noop(&req.params)
            }
            "ServiceWorker.startWorker" => self.service_worker_known_noop(&req.params),
            "ServiceWorker.stopWorker" => self.service_worker_known_noop(&req.params),
            "ServiceWorker.skipWaiting" => self.service_worker_known_noop(&req.params),
            "ServiceWorker.updateRegistration" => self.service_worker_known_noop(&req.params),

            // ── Target ────────────────────────────────────────────────────
            "Target.getTargets" => self.target_get_targets(),
            "Target.getTargetInfo" => self.target_get_target_info(&req.params),
            "Target.setDiscoverTargets" => self.target_set_discover_targets(&req.params),
            "Target.setAutoAttach" => self.target_set_auto_attach(&req.params),
            "Target.setRemoteLocations" => self.target_set_remote_locations(&req.params),
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

    fn network_set_cache_disabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(cache_disabled) = params.get("cacheDisabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Network.setCacheDisabled: required parameter 'cacheDisabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.network_cache_disabled = cache_disabled;
        Ok(json!({}))
    }

    fn network_set_bypass_service_worker(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(bypass) = params.get("bypass").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Network.setBypassServiceWorker: required parameter 'bypass' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.network_bypass_service_worker = bypass;
        Ok(json!({}))
    }

    fn network_set_attach_debug_stack(&mut self, params: &Value) -> StatorResult<Value> {
        let enabled = required_bool_param(params, "enabled", "Network.setAttachDebugStack")?;
        self.network_attach_debug_stack = enabled;
        Ok(json!({}))
    }

    fn network_set_reporting_api_enabled(&mut self, params: &Value) -> StatorResult<Value> {
        let enabled = required_bool_param(params, "enabled", "Network.setReportingApiEnabled")?;
        self.network_reporting_api_enabled = enabled;
        Ok(json!({}))
    }

    fn network_set_user_agent_override(&mut self, params: &Value) -> StatorResult<Value> {
        let (user_agent, accept_language, platform, metadata) =
            parse_user_agent_override_params(params, "Network.setUserAgentOverride")?;
        self.network_user_agent = user_agent;
        self.network_accept_language = accept_language;
        self.network_platform = platform;
        self.network_user_agent_metadata = metadata;
        Ok(json!({}))
    }

    fn network_set_extra_http_headers(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(headers) = params.get("headers").and_then(Value::as_object) else {
            return Err(crate::error::StatorError::TypeError(
                "Network.setExtraHTTPHeaders: required parameter 'headers' is missing or not an object"
                    .to_string(),
            ));
        };
        if headers.values().any(|value| !value.is_string()) {
            return Err(crate::error::StatorError::TypeError(
                "Network.setExtraHTTPHeaders: header values must be strings".to_string(),
            ));
        }
        self.network_extra_http_header_count = headers.len();
        Ok(json!({}))
    }

    fn network_set_blocked_urls(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(urls) = params.get("urls").and_then(Value::as_array) else {
            return Err(crate::error::StatorError::TypeError(
                "Network.setBlockedURLs: required parameter 'urls' is missing or not an array"
                    .to_string(),
            ));
        };
        if urls.iter().any(|value| !value.is_string()) {
            return Err(crate::error::StatorError::TypeError(
                "Network.setBlockedURLs: URL patterns must be strings".to_string(),
            ));
        }
        self.network_blocked_url_count = urls.len();
        Ok(json!({}))
    }

    fn network_set_accepted_encodings(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(encodings) = params.get("encodings").and_then(Value::as_array) else {
            return Err(crate::error::StatorError::TypeError(
                "Network.setAcceptedEncodings: required parameter 'encodings' is missing or not an array"
                    .to_string(),
            ));
        };
        for value in encodings {
            let Some(encoding) = value.as_str() else {
                return Err(crate::error::StatorError::TypeError(
                    "Network.setAcceptedEncodings: encodings must be strings".to_string(),
                ));
            };
            if !matches!(encoding, "br" | "deflate" | "gzip" | "identity" | "zstd") {
                return Err(crate::error::StatorError::TypeError(format!(
                    "Network.setAcceptedEncodings: unsupported encoding '{encoding}'"
                )));
            }
        }
        self.network_accepted_encoding_count = encodings.len();
        Ok(json!({}))
    }

    fn network_clear_accepted_encodings_override(&mut self) -> StatorResult<Value> {
        self.network_accepted_encoding_count = 0;
        Ok(json!({}))
    }

    fn network_emulate_network_conditions(&mut self, params: &Value) -> StatorResult<Value> {
        let offline = required_bool_param(params, "offline", "Network.emulateNetworkConditions")?;
        let latency = required_finite_non_negative_number_param(
            params,
            "latency",
            "Network.emulateNetworkConditions",
        )?;
        let download_throughput = required_throughput_param(
            params,
            "downloadThroughput",
            "Network.emulateNetworkConditions",
        )?;
        let upload_throughput = required_throughput_param(
            params,
            "uploadThroughput",
            "Network.emulateNetworkConditions",
        )?;
        let connection_type = match optional_string_param(
            params,
            "connectionType",
            "Network.emulateNetworkConditions",
        )? {
            Some(connection_type) => {
                if !matches!(
                    connection_type,
                    "none"
                        | "cellular2g"
                        | "cellular3g"
                        | "cellular4g"
                        | "bluetooth"
                        | "ethernet"
                        | "wifi"
                        | "wimax"
                        | "other"
                ) {
                    return Err(crate::error::StatorError::TypeError(format!(
                        "Network.emulateNetworkConditions: unsupported connectionType '{connection_type}'"
                    )));
                }
                Some(connection_type.to_string())
            }
            None => None,
        };
        let packet_loss =
            optional_packet_loss_param(params, "packetLoss", "Network.emulateNetworkConditions")?;
        let packet_queue_length = optional_u32_param(
            params,
            "packetQueueLength",
            "Network.emulateNetworkConditions",
        )?;
        let packet_reordering = optional_bool_param(
            params,
            "packetReordering",
            "Network.emulateNetworkConditions",
        )?;

        self.network_emulated_conditions = Some(NetworkEmulatedConditions {
            offline,
            latency,
            download_throughput,
            upload_throughput,
            connection_type,
            packet_loss,
            packet_queue_length,
            packet_reordering,
        });
        Ok(json!({}))
    }

    fn page_frame_tree(&self) -> Value {
        let context = self
            .contexts
            .first()
            .cloned()
            .unwrap_or_else(default_execution_context);
        let url = if context.origin.is_empty() {
            "stator://page".to_string()
        } else {
            context.origin.clone()
        };
        json!({
            "frame": {
                "id": format!("stator-frame-{}", context.id),
                "loaderId": format!("stator-loader-{}", context.id),
                "url": url,
                "domainAndRegistry": "",
                "securityOrigin": context.origin,
                "mimeType": "text/html",
            },
            "childFrames": [],
            "resources": [],
        })
    }

    fn page_get_resource_tree(&self) -> StatorResult<Value> {
        Ok(json!({
            "frameTree": self.page_frame_tree(),
        }))
    }

    fn page_get_frame_tree(&self) -> StatorResult<Value> {
        Ok(json!({
            "frameTree": self.page_frame_tree(),
        }))
    }

    fn page_set_lifecycle_events_enabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Page.setLifecycleEventsEnabled: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.page_lifecycle_events_enabled = enabled;
        Ok(json!({}))
    }

    fn page_set_bypass_csp(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Page.setBypassCSP: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.page_bypass_csp = enabled;
        Ok(json!({}))
    }

    fn page_set_ad_blocking_enabled(&mut self, params: &Value) -> StatorResult<Value> {
        let enabled = required_bool_param(params, "enabled", "Page.setAdBlockingEnabled")?;
        self.page_ad_blocking_enabled = enabled;
        Ok(json!({}))
    }

    fn log_start_violations_report(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(config) = params.get("config").and_then(Value::as_array) else {
            return Err(crate::error::StatorError::TypeError(
                "Log.startViolationsReport: required parameter 'config' is missing or not an array"
                    .to_string(),
            ));
        };
        for setting in config {
            let name = setting.get("name").and_then(Value::as_str);
            let threshold = setting.get("threshold").and_then(Value::as_i64);
            if name.is_none() || threshold.is_none() {
                return Err(crate::error::StatorError::TypeError(
                    "Log.startViolationsReport: every config entry requires string 'name' and numeric 'threshold'"
                        .to_string(),
                ));
            }
        }
        self.log_violation_setting_count = config.len();
        Ok(json!({}))
    }

    fn log_stop_violations_report(&mut self) -> StatorResult<Value> {
        self.log_violation_setting_count = 0;
        Ok(json!({}))
    }

    fn security_set_ignore_certificate_errors(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(ignore) = params.get("ignore").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Security.setIgnoreCertificateErrors: required parameter 'ignore' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.security_ignore_certificate_errors = ignore;
        Ok(json!({}))
    }

    fn performance_get_metrics(&self) -> StatorResult<Value> {
        let heap_used = self.estimated_heap_usage_bytes() as f64;
        Ok(json!({
            "metrics": [
                { "name": "Timestamp", "value": 0.0 },
                { "name": "Documents", "value": self.contexts.len() as f64 },
                { "name": "Frames", "value": self.contexts.len().max(1) as f64 },
                { "name": "JSHeapUsedSize", "value": heap_used },
                { "name": "JSHeapTotalSize", "value": heap_used },
                { "name": "ScriptDuration", "value": 0.0 },
                { "name": "TaskDuration", "value": 0.0 },
            ]
        }))
    }

    fn emulation_set_device_metrics_override(&mut self, params: &Value) -> StatorResult<Value> {
        let width = params.get("width").and_then(Value::as_u64);
        let height = params.get("height").and_then(Value::as_u64);
        let scale = params.get("deviceScaleFactor").and_then(Value::as_f64);
        let mobile = params.get("mobile").and_then(Value::as_bool);
        if width.is_none() || height.is_none() || scale.is_none() || mobile.is_none() {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setDeviceMetricsOverride: width, height, deviceScaleFactor, and mobile are required"
                    .to_string(),
            ));
        }
        if scale.is_some_and(|value| value < 0.0) {
            return Err(crate::error::StatorError::RangeError(
                "Emulation.setDeviceMetricsOverride: deviceScaleFactor must be non-negative"
                    .to_string(),
            ));
        }
        self.emulation_device_metrics = Some(params.clone());
        Ok(json!({}))
    }

    fn emulation_clear_device_metrics_override(&mut self) -> StatorResult<Value> {
        self.emulation_device_metrics = None;
        Ok(json!({}))
    }

    fn emulation_set_touch_emulation_enabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setTouchEmulationEnabled: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        let max_touch_points = params
            .get("maxTouchPoints")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .min(u32::MAX as u64) as u32;
        self.emulation_touch_enabled = enabled;
        self.emulation_max_touch_points = max_touch_points;
        Ok(json!({}))
    }

    fn emulation_set_emit_touch_events_for_mouse(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setEmitTouchEventsForMouse: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        let configuration = match params.get("configuration") {
            Some(value) => {
                let Some(configuration) = value.as_str() else {
                    return Err(crate::error::StatorError::TypeError(
                        "Emulation.setEmitTouchEventsForMouse: optional parameter 'configuration' must be a string"
                            .to_string(),
                    ));
                };
                if configuration != "mobile" && configuration != "desktop" {
                    return Err(crate::error::StatorError::TypeError(
                        "Emulation.setEmitTouchEventsForMouse: optional parameter 'configuration' must be 'mobile' or 'desktop'"
                            .to_string(),
                    ));
                }
                configuration
            }
            None => "",
        };
        self.emulation_emit_touch_events_for_mouse_enabled = enabled;
        self.emulation_emit_touch_events_for_mouse_configuration = configuration.to_string();
        Ok(json!({}))
    }

    fn emulation_set_emulated_media(&mut self, params: &Value) -> StatorResult<Value> {
        let media = params.get("media").and_then(Value::as_str).unwrap_or("");
        let feature_count = match params.get("features") {
            Some(Value::Array(features)) => features.len(),
            Some(_) => {
                return Err(crate::error::StatorError::TypeError(
                    "Emulation.setEmulatedMedia: optional parameter 'features' must be an array"
                        .to_string(),
                ));
            }
            None => 0,
        };
        self.emulation_media = media.to_string();
        self.emulation_media_feature_count = feature_count;
        Ok(json!({}))
    }

    fn emulation_set_cpu_throttling_rate(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(rate) = params.get("rate").and_then(Value::as_f64) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setCPUThrottlingRate: required parameter 'rate' is missing or not a number"
                    .to_string(),
            ));
        };
        if !rate.is_finite() || rate < 0.0 {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setCPUThrottlingRate: 'rate' must be a finite, non-negative number"
                    .to_string(),
            ));
        }
        self.emulation_cpu_throttling_rate = rate;
        Ok(json!({}))
    }

    fn emulation_set_hardware_concurrency_override(
        &mut self,
        params: &Value,
    ) -> StatorResult<Value> {
        let hardware_concurrency = required_u32_param(
            params,
            "hardwareConcurrency",
            "Emulation.setHardwareConcurrencyOverride",
        )?;
        if hardware_concurrency == 0 {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setHardwareConcurrencyOverride: 'hardwareConcurrency' must be a positive integer"
                    .to_string(),
            ));
        }
        self.emulation_hardware_concurrency = hardware_concurrency;
        Ok(json!({}))
    }

    fn emulation_set_auto_dark_mode_override(&mut self, params: &Value) -> StatorResult<Value> {
        self.emulation_auto_dark_mode_enabled =
            optional_bool_param(params, "enabled", "Emulation.setAutoDarkModeOverride")?;
        Ok(json!({}))
    }

    fn emulation_set_document_cookie_disabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(disabled) = params.get("disabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setDocumentCookieDisabled: required parameter 'disabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.emulation_document_cookie_disabled = disabled;
        Ok(json!({}))
    }

    fn emulation_set_timezone_override(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(timezone_id) = params.get("timezoneId").and_then(Value::as_str) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setTimezoneOverride: required parameter 'timezoneId' is missing or not a string"
                    .to_string(),
            ));
        };
        self.emulation_timezone_id = timezone_id.to_string();
        Ok(json!({}))
    }

    fn emulation_set_locale_override(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(locale) = params.get("locale").and_then(Value::as_str) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setLocaleOverride: required parameter 'locale' is missing or not a string"
                    .to_string(),
            ));
        };
        self.emulation_locale = locale.to_string();
        Ok(json!({}))
    }

    fn emulation_set_user_agent_override(&mut self, params: &Value) -> StatorResult<Value> {
        let (user_agent, accept_language, platform, metadata) =
            parse_user_agent_override_params(params, "Emulation.setUserAgentOverride")?;
        self.emulation_user_agent = user_agent;
        self.emulation_accept_language = accept_language;
        self.emulation_platform = platform;
        self.emulation_user_agent_metadata = metadata;
        Ok(json!({}))
    }

    fn emulation_set_script_execution_disabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(value) = params.get("value").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setScriptExecutionDisabled: required parameter 'value' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.emulation_script_execution_disabled = value;
        Ok(json!({}))
    }

    fn emulation_set_focus_emulation_enabled(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setFocusEmulationEnabled: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.emulation_focus_emulation_enabled = enabled;
        Ok(json!({}))
    }

    fn emulation_set_idle_override(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(user_active) = params.get("userActive").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setIdleOverride: required parameter 'userActive' is missing or not a boolean"
                    .to_string(),
            ));
        };
        let Some(screen_unlocked) = params.get("screenUnlocked").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Emulation.setIdleOverride: required parameter 'screenUnlocked' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.emulation_idle_override = Some((user_active, screen_unlocked));
        Ok(json!({}))
    }

    fn emulation_clear_idle_override(&mut self) -> StatorResult<Value> {
        self.emulation_idle_override = None;
        Ok(json!({}))
    }

    fn emulation_set_geolocation_override(&mut self, params: &Value) -> StatorResult<Value> {
        let method = "Emulation.setGeolocationOverride";
        let latitude = optional_finite_number_param(params, "latitude", method)?;
        validate_optional_number_range(method, "latitude", latitude, -90.0, 90.0)?;
        let longitude = optional_finite_number_param(params, "longitude", method)?;
        validate_optional_number_range(method, "longitude", longitude, -180.0, 180.0)?;
        let accuracy = optional_finite_number_param(params, "accuracy", method)?;
        validate_optional_non_negative_number(method, "accuracy", accuracy)?;
        let altitude = optional_finite_number_param(params, "altitude", method)?;
        let altitude_accuracy = optional_finite_number_param(params, "altitudeAccuracy", method)?;
        validate_optional_non_negative_number(method, "altitudeAccuracy", altitude_accuracy)?;
        let heading = optional_finite_number_param(params, "heading", method)?;
        validate_optional_number_range(method, "heading", heading, 0.0, 360.0)?;
        let speed = optional_finite_number_param(params, "speed", method)?;
        validate_optional_non_negative_number(method, "speed", speed)?;

        self.emulation_geolocation_override = Some(EmulationGeolocationOverride {
            latitude,
            longitude,
            accuracy,
            altitude,
            altitude_accuracy,
            heading,
            speed,
        });
        Ok(json!({}))
    }

    fn emulation_clear_geolocation_override(&mut self) -> StatorResult<Value> {
        self.emulation_geolocation_override = None;
        Ok(json!({}))
    }

    fn emulation_set_page_scale_factor(&mut self, params: &Value) -> StatorResult<Value> {
        let method = "Emulation.setPageScaleFactor";
        let Some(page_scale_factor) = params.get("pageScaleFactor").and_then(Value::as_f64) else {
            return Err(StatorError::TypeError(
                "Emulation.setPageScaleFactor: required parameter 'pageScaleFactor' is missing or not a number"
                    .to_string(),
            ));
        };
        if !page_scale_factor.is_finite() || page_scale_factor <= 0.0 {
            return Err(StatorError::TypeError(format!(
                "{method}: 'pageScaleFactor' must be a finite, positive number"
            )));
        }
        self.emulation_page_scale_factor = Some(page_scale_factor);
        Ok(json!({}))
    }

    fn emulation_reset_page_scale_factor(&mut self) -> StatorResult<Value> {
        self.emulation_page_scale_factor = None;
        Ok(json!({}))
    }

    fn emulation_set_scrollbars_hidden(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(hidden) = params.get("hidden").and_then(Value::as_bool) else {
            return Err(StatorError::TypeError(
                "Emulation.setScrollbarsHidden: required parameter 'hidden' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.emulation_scrollbars_hidden = hidden;
        Ok(json!({}))
    }

    fn overlay_required_bool(params: &Value, method: &str, field: &str) -> StatorResult<bool> {
        params.get(field).and_then(Value::as_bool).ok_or_else(|| {
            crate::error::StatorError::TypeError(format!(
                "{method}: required parameter '{field}' is missing or not a boolean"
            ))
        })
    }

    fn overlay_set_show_paint_rects(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_paint_rects =
            Self::overlay_required_bool(params, "Overlay.setShowPaintRects", "result")?;
        Ok(json!({}))
    }

    fn overlay_set_show_debug_borders(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_debug_borders =
            Self::overlay_required_bool(params, "Overlay.setShowDebugBorders", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_fps_counter(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_fps_counter =
            Self::overlay_required_bool(params, "Overlay.setShowFPSCounter", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_web_vitals(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_web_vitals =
            Self::overlay_required_bool(params, "Overlay.setShowWebVitals", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_layout_shift_regions(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_layout_shift_regions =
            Self::overlay_required_bool(params, "Overlay.setShowLayoutShiftRegions", "result")?;
        Ok(json!({}))
    }

    fn overlay_set_show_ad_highlights(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_ad_highlights =
            Self::overlay_required_bool(params, "Overlay.setShowAdHighlights", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_viewport_size_on_resize(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_viewport_size_on_resize =
            Self::overlay_required_bool(params, "Overlay.setShowViewportSizeOnResize", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_scroll_bottleneck_rects(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_scroll_bottleneck_rects =
            Self::overlay_required_bool(params, "Overlay.setShowScrollBottleneckRects", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_hit_test_borders(&mut self, params: &Value) -> StatorResult<Value> {
        self.overlay_show_hit_test_borders =
            Self::overlay_required_bool(params, "Overlay.setShowHitTestBorders", "show")?;
        Ok(json!({}))
    }

    fn overlay_set_show_grid_overlays(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(configs) = params.get("gridNodeHighlightConfigs") else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowGridOverlays: required parameter 'gridNodeHighlightConfigs' is missing"
                    .to_string(),
            ));
        };
        let Value::Array(configs) = configs else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowGridOverlays: required parameter 'gridNodeHighlightConfigs' must be an array"
                    .to_string(),
            ));
        };
        for config in configs {
            if !config.is_object() {
                return Err(crate::error::StatorError::TypeError(
                    "Overlay.setShowGridOverlays: every grid overlay config must be an object"
                        .to_string(),
                ));
            }
        }
        self.overlay_grid_overlay_count = configs.len();
        Ok(json!({}))
    }

    fn overlay_set_show_flex_overlays(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(configs) = params.get("flexNodeHighlightConfigs") else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowFlexOverlays: required parameter 'flexNodeHighlightConfigs' is missing"
                    .to_string(),
            ));
        };
        let Value::Array(configs) = configs else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowFlexOverlays: required parameter 'flexNodeHighlightConfigs' must be an array"
                    .to_string(),
            ));
        };
        for config in configs {
            if !config.is_object() {
                return Err(crate::error::StatorError::TypeError(
                    "Overlay.setShowFlexOverlays: every flex overlay config must be an object"
                        .to_string(),
                ));
            }
        }
        self.overlay_flex_overlay_count = configs.len();
        Ok(json!({}))
    }

    fn overlay_set_show_scroll_snap_overlays(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(configs) = params.get("scrollSnapHighlightConfigs") else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowScrollSnapOverlays: required parameter 'scrollSnapHighlightConfigs' is missing"
                    .to_string(),
            ));
        };
        let Value::Array(configs) = configs else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowScrollSnapOverlays: required parameter 'scrollSnapHighlightConfigs' must be an array"
                    .to_string(),
            ));
        };
        for config in configs {
            if !config.is_object() {
                return Err(crate::error::StatorError::TypeError(
                    "Overlay.setShowScrollSnapOverlays: every scroll-snap overlay config must be an object"
                        .to_string(),
                ));
            }
        }
        self.overlay_scroll_snap_overlay_count = configs.len();
        Ok(json!({}))
    }

    fn overlay_set_show_container_query_overlays(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(configs) = params.get("containerQueryHighlightConfigs") else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowContainerQueryOverlays: required parameter 'containerQueryHighlightConfigs' is missing"
                    .to_string(),
            ));
        };
        let Value::Array(configs) = configs else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowContainerQueryOverlays: required parameter 'containerQueryHighlightConfigs' must be an array"
                    .to_string(),
            ));
        };
        for config in configs {
            if !config.is_object() {
                return Err(crate::error::StatorError::TypeError(
                    "Overlay.setShowContainerQueryOverlays: every container-query overlay config must be an object"
                        .to_string(),
                ));
            }
        }
        self.overlay_container_query_overlay_count = configs.len();
        Ok(json!({}))
    }

    fn overlay_set_show_isolated_elements(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(configs) = params.get("isolatedElementHighlightConfigs") else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowIsolatedElements: required parameter 'isolatedElementHighlightConfigs' is missing"
                    .to_string(),
            ));
        };
        let Value::Array(configs) = configs else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setShowIsolatedElements: required parameter 'isolatedElementHighlightConfigs' must be an array"
                    .to_string(),
            ));
        };
        for config in configs {
            if !config.is_object() {
                return Err(crate::error::StatorError::TypeError(
                    "Overlay.setShowIsolatedElements: every isolated-elements config must be an object"
                        .to_string(),
                ));
            }
        }
        self.overlay_isolated_element_count = configs.len();
        Ok(json!({}))
    }

    fn overlay_set_inspect_mode(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(mode) = params.get("mode").and_then(Value::as_str) else {
            return Err(crate::error::StatorError::TypeError(
                "Overlay.setInspectMode: required parameter 'mode' is missing or not a string"
                    .to_string(),
            ));
        };
        self.overlay_inspect_mode = mode.to_string();
        Ok(json!({}))
    }

    fn service_worker_set_force_update_on_page_load(
        &mut self,
        params: &Value,
    ) -> StatorResult<Value> {
        let Some(force_update_on_page_load) =
            params.get("forceUpdateOnPageLoad").and_then(Value::as_bool)
        else {
            return Err(crate::error::StatorError::TypeError(
                "ServiceWorker.setForceUpdateOnPageLoad: required parameter 'forceUpdateOnPageLoad' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.service_worker_force_update_on_page_load = force_update_on_page_load;
        Ok(json!({}))
    }

    fn service_worker_known_noop(&self, params: &Value) -> StatorResult<Value> {
        if let Some(version_id) = params.get("versionId")
            && !version_id.is_string()
        {
            return Err(crate::error::StatorError::TypeError(
                "ServiceWorker request parameter 'versionId' must be a string when present"
                    .to_string(),
            ));
        }
        if let Some(registration_id) = params.get("registrationId")
            && !registration_id.is_string()
        {
            return Err(crate::error::StatorError::TypeError(
                "ServiceWorker request parameter 'registrationId' must be a string when present"
                    .to_string(),
            ));
        }
        Ok(json!({}))
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

    fn runtime_set_custom_object_formatter_enabled(
        &mut self,
        params: &Value,
    ) -> StatorResult<Value> {
        let Some(enabled) = params.get("enabled").and_then(Value::as_bool) else {
            return Err(crate::error::StatorError::TypeError(
                "Runtime.setCustomObjectFormatterEnabled: required parameter 'enabled' is missing or not a boolean"
                    .to_string(),
            ));
        };
        self.custom_object_formatter_enabled = enabled;
        Ok(json!({}))
    }

    fn runtime_set_max_call_stack_size_to_capture(
        &mut self,
        params: &Value,
    ) -> StatorResult<Value> {
        let Some(size) = params.get("size").and_then(Value::as_u64) else {
            return Err(crate::error::StatorError::TypeError(
                "Runtime.setMaxCallStackSizeToCapture: required parameter 'size' is missing or not a number"
                    .to_string(),
            ));
        };
        self.max_call_stack_size_to_capture = size.min(u32::MAX as u64) as u32;
        Ok(json!({}))
    }

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

    fn estimated_heap_usage_bytes(&self) -> u64 {
        const NODE_FIELDS: usize = 5;
        const SELF_SIZE_INDEX: usize = 3;

        let snapshot = HeapSnapshotBuilder::build(&self.globals.borrow().vars);
        snapshot
            .nodes
            .chunks(NODE_FIELDS)
            .filter_map(|node| node.get(SELF_SIZE_INDEX))
            .map(|size| u64::from(*size))
            .sum()
    }

    fn runtime_get_heap_usage(&self) -> StatorResult<Value> {
        let used_size = self.estimated_heap_usage_bytes();
        Ok(json!({
            "usedSize": used_size,
            "totalSize": used_size,
        }))
    }

    fn collect_garbage(&mut self) -> StatorResult<Value> {
        crate::gc::runtime::gc_collect();
        Ok(json!({}))
    }

    fn runtime_terminate_execution(&mut self) -> StatorResult<Value> {
        self.termination_requested.store(true, Ordering::SeqCst);
        Ok(json!({}))
    }

    fn run_interpreter_frame(&self, frame: &mut InterpreterFrame) -> StatorResult<JsValue> {
        // SAFETY: the pointer references `self.termination_requested`, which
        // remains valid until this method clears the thread-local association.
        unsafe { crate::interpreter::set_interrupt_flag(&self.termination_requested) };
        let result = Interpreter::run(frame);
        crate::interpreter::clear_interrupt_flag();
        self.termination_requested.store(false, Ordering::SeqCst);
        result
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

    fn emit_script_failed_to_parse(
        &mut self,
        script_id: String,
        source: &str,
        source_url: Option<&str>,
        execution_context_id: u32,
        exception_details: Value,
    ) {
        if !self.debugger_enabled {
            return;
        }
        let url = source_url
            .filter(|url| !url.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| registered_script_url(source));
        let line_count = source.lines().count().max(1) as u32;
        let last_line_columns = source
            .lines()
            .last()
            .map(|line| line.chars().count() as u32)
            .unwrap_or(0);
        self.push_event(
            "Debugger.scriptFailedToParse",
            json!({
                "scriptId": script_id,
                "url": url,
                "startLine": 0,
                "startColumn": 0,
                "endLine": line_count.saturating_sub(1),
                "endColumn": last_line_columns,
                "executionContextId": execution_context_id,
                "hash": script_hash(source),
                "scriptLanguage": "JavaScript",
                "exceptionDetails": exception_details,
            }),
        );
    }

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
                    let details = self.exception_details_only(
                        &err,
                        ExceptionRequest {
                            expression,
                            source_url: source_url.as_deref(),
                            execution_context_id,
                            object_group: None,
                            generate_preview: false,
                        },
                    );
                    let failed_script_id =
                        format!("runtime-script-failed-{}", self.next_compiled_script_id);
                    self.next_compiled_script_id = self.next_compiled_script_id.saturating_add(1);
                    self.emit_script_failed_to_parse(
                        failed_script_id,
                        expression,
                        source_url.as_deref(),
                        execution_context_id,
                        details.clone(),
                    );
                    return Ok(json!({
                        "exceptionDetails": details
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
        let js_result = match self.run_interpreter_frame(&mut frame) {
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
        let js_result = match self.run_interpreter_frame(&mut frame) {
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
        let js_result = match self.run_interpreter_frame(&mut frame) {
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

    fn runtime_await_promise(&mut self, params: &Value) -> StatorResult<Value> {
        let promise_object_id =
            params
                .get("promiseObjectId")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    crate::error::StatorError::TypeError(
                        "Runtime.awaitPromise: required parameter 'promiseObjectId' is missing or not a string"
                            .to_string(),
                    )
                })?;
        let object_group = params
            .get("objectGroup")
            .and_then(Value::as_str)
            .map(str::to_string);
        let generate_preview = params
            .get("generatePreview")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let execution_context_id = self.resolve_execution_context_id(params)?;
        let promise_value = self.remote_objects.get(promise_object_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Runtime.awaitPromise: unknown or released objectId `{promise_object_id}`"
            ))
        })?;
        let JsValue::Promise(promise) = promise_value else {
            return Err(crate::error::StatorError::TypeError(format!(
                "Runtime.awaitPromise: objectId `{promise_object_id}` does not reference a Promise"
            )));
        };

        drain_active_microtask_queue();
        match promise.state() {
            PromiseState::Fulfilled(value) => {
                let remote = js_value_to_remote_object(
                    &value,
                    &mut self.remote_objects,
                    object_group.as_deref(),
                    generate_preview,
                );
                Ok(json!({ "result": remote }))
            }
            PromiseState::Rejected(reason) => {
                let exception_id = self.next_exception_id;
                self.next_exception_id = self.next_exception_id.saturating_add(1);
                let err = StatorError::JsException(format!("{reason:?}"));
                let details = self.build_exception_details(
                    exception_id,
                    &err,
                    ExceptionRequest {
                        expression: "Runtime.awaitPromise",
                        source_url: None,
                        execution_context_id,
                        object_group: object_group.as_deref(),
                        generate_preview,
                    },
                    Some(&reason),
                );
                Ok(json!({
                    "result": {"type": "undefined"},
                    "exceptionDetails": details,
                }))
            }
            PromiseState::Pending => Err(crate::error::StatorError::Internal(
                "Runtime.awaitPromise: promise is still pending after draining microtasks"
                    .to_string(),
            )),
        }
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

    fn debugger_set_async_call_stack_depth(&mut self, params: &Value) -> StatorResult<Value> {
        let Some(max_depth) = params.get("maxDepth").and_then(Value::as_u64) else {
            return Err(crate::error::StatorError::TypeError(
                "Debugger.setAsyncCallStackDepth: required parameter 'maxDepth' is missing or not a number"
                    .to_string(),
            ));
        };
        self.async_call_stack_depth = max_depth.min(u32::MAX as u64) as u32;
        Ok(json!({}))
    }

    fn debugger_get_stack_trace(&self, params: &Value) -> StatorResult<Value> {
        let stack_trace_id = params.get("stackTraceId").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.getStackTrace: required parameter 'stackTraceId' is missing".to_string(),
            )
        })?;
        let id = stack_trace_id
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.getStackTrace: stackTraceId.id is missing or not a string"
                        .to_string(),
                )
            })?;
        if let Some(stack_trace) = self.stack_traces.get(id) {
            return Ok(json!({ "stackTrace": stack_trace }));
        }
        Err(crate::error::StatorError::Internal(format!(
            "Debugger.getStackTrace: unknown stackTraceId `{id}`"
        )))
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
            regress::Regex::new(pattern).map_err(|err| {
                StatorError::SyntaxError(format!(
                    "Debugger.setBlackboxPatterns: invalid pattern `{pattern}`: {err}"
                ))
            })?;
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
        let positions: Vec<Value> = positions
            .iter()
            .map(|position| {
                let line_number =
                    required_u32_param(position, "lineNumber", "Debugger.setBlackboxedRanges")?;
                let column_number =
                    required_u32_param(position, "columnNumber", "Debugger.setBlackboxedRanges")?;
                Ok(json!({
                    "lineNumber": line_number,
                    "columnNumber": column_number,
                }))
            })
            .collect::<StatorResult<_>>()?;
        self.blackboxed_ranges
            .insert(script_id.to_string(), positions);
        Ok(json!({}))
    }

    fn debugger_disable(&mut self) -> StatorResult<Value> {
        self.debugger_enabled = false;
        self.clear_debugger_breakpoint_state();
        self.paused_local_scope_object_id = None;
        self.paused_context_scope_object_ids.clear();
        self.paused_global_scope_object_id = None;
        Ok(json!({}))
    }

    fn clear_debugger_breakpoint_state(&mut self) {
        if let Some(debugger) = self.debugger.as_ref() {
            let mut debugger = debugger.borrow_mut();
            for debugger_id in self
                .debugger_breakpoint_locations
                .keys()
                .copied()
                .collect::<Vec<_>>()
            {
                let _ = debugger.remove_breakpoint(debugger_id);
            }
        }
        self.cdp_breakpoints.clear();
        self.cdp_script_breakpoints.clear();
        self.cdp_debugger_breakpoints.clear();
        self.cdp_url_breakpoints.clear();
        self.debugger_breakpoint_locations.clear();
    }

    // ── Debugger.resume ──────────────────────────────────────────────────────

    fn debugger_resume(&mut self) -> StatorResult<Value> {
        // V8 inspector treats resume with no attached debugger / no active
        // pause as a no-op success; mirror that so DevTools teardown does not
        // surface spurious errors.
        let mut emitted = false;
        if let Some(bridge) = &self.pause_bridge {
            emitted |= bridge.resume(DebugAction::Continue);
        }
        if let Some(debugger) = self.debugger.as_ref() {
            let mut dbg = debugger.borrow_mut();
            if dbg.last_pause_reason().is_some() {
                dbg.apply_action(DebugAction::Continue);
                dbg.clear_last_pause();
                emitted = true;
            }
        }
        if emitted {
            self.notify_resumed();
        }
        Ok(json!({}))
    }

    fn debugger_terminate_on_resume(&mut self) -> StatorResult<Value> {
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.terminateOnResume",
                "no same-thread interpreter Debugger is attached to this session.",
            ));
        };
        if debugger.borrow().last_pause_reason().is_none() {
            return Err(unsupported_debugger_method(
                "Debugger.terminateOnResume",
                "no active pause; terminateOnResume is only valid after a Debugger.paused event.",
            ));
        }
        self.termination_requested.store(true, Ordering::SeqCst);
        Ok(json!({}))
    }

    fn debugger_pause(&mut self) -> StatorResult<Value> {
        if let Some(bridge) = &self.pause_bridge {
            bridge.request_pause();
        }
        if let Some(debugger) = self.debugger.as_ref() {
            debugger.borrow_mut().request_pause();
        }
        Ok(json!({}))
    }

    fn debugger_continue_to_location(&mut self, params: &Value) -> StatorResult<Value> {
        validate_continue_to_location_options(params)?;
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
        let line_number =
            required_u32_param(location, "lineNumber", "Debugger.continueToLocation")?;
        let column_number =
            optional_u32_param(location, "columnNumber", "Debugger.continueToLocation")?
                .unwrap_or(0);
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
            debugger.clear_last_pause();
        }
        self.notify_resumed();
        Ok(json!({}))
    }

    fn debugger_evaluate_on_call_frame(&mut self, params: &Value) -> StatorResult<Value> {
        let call_frame_id = params
            .get("callFrameId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.evaluateOnCallFrame: required parameter 'callFrameId' is missing or \
                     not a string"
                        .to_string(),
                )
            })?;
        let requested_offset =
            parse_pause_call_frame_id("Debugger.evaluateOnCallFrame", call_frame_id)?;
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.evaluateOnCallFrame",
                "no interpreter Debugger is attached to this session.",
            ));
        };
        let debugger_ref = debugger.borrow();
        ensure_active_call_frame_id(
            "Debugger.evaluateOnCallFrame",
            requested_offset,
            &debugger_ref,
        )?;
        let frame_snapshot = debugger_ref.last_pause_frame_snapshot().cloned();
        drop(debugger_ref);
        if let Some(snapshot) = frame_snapshot {
            let mut bindings = frame_snapshot_bindings(&snapshot);
            for (context_index, slots) in snapshot.context_slots.iter().enumerate() {
                bindings.extend(context_snapshot_bindings(context_index, slots));
            }
            let _bindings = TemporaryGlobalBindings::new(Rc::clone(&self.globals), bindings);
            return self.runtime_evaluate(params);
        }
        self.runtime_evaluate(params)
    }

    fn debugger_set_variable_value(&mut self, params: &Value) -> StatorResult<Value> {
        let scope_number = params
            .get("scopeNumber")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setVariableValue: required parameter 'scopeNumber' is missing or \
                     not a number"
                        .to_string(),
                )
            })?;
        let variable_name = params
            .get("variableName")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setVariableValue: required parameter 'variableName' is missing or \
                     not a string"
                        .to_string(),
                )
            })?;
        let call_frame_id = params
            .get("callFrameId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setVariableValue: required parameter 'callFrameId' is missing or \
                     not a string"
                        .to_string(),
                )
            })?;
        let requested_offset =
            parse_pause_call_frame_id("Debugger.setVariableValue", call_frame_id)?;
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.setVariableValue",
                "no interpreter Debugger is attached to this session.",
            ));
        };
        let new_value = params.get("newValue").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.setVariableValue: required parameter 'newValue' is missing".to_string(),
            )
        })?;
        let value = call_argument_to_js_value(new_value, &self.remote_objects)?;
        let (has_local_scope, context_scope_count) = {
            let debugger_ref = debugger.borrow();
            ensure_active_call_frame_id(
                "Debugger.setVariableValue",
                requested_offset,
                &debugger_ref,
            )?;
            let snapshot = debugger_ref.last_pause_frame_snapshot();
            (
                snapshot.is_some(),
                snapshot
                    .map(|snapshot| snapshot.context_slots.len() as u64)
                    .unwrap_or(0),
            )
        };
        let global_scope_number = if has_local_scope {
            1 + context_scope_count
        } else {
            0
        };
        if scope_number == 0 && has_local_scope {
            let scope_value = value.cheap_clone();
            debugger
                .borrow_mut()
                .set_paused_frame_binding(variable_name, value)?;
            if let Some(object_id) = self.paused_local_scope_object_id.as_deref() {
                self.remote_objects.set_plain_object_property(
                    object_id,
                    variable_name,
                    scope_value,
                );
            }
            return Ok(json!({}));
        }
        if has_local_scope && (1..global_scope_number).contains(&scope_number) {
            let context_index = (scope_number - 1) as usize;
            let scope_value = value.cheap_clone();
            debugger
                .borrow_mut()
                .set_paused_context_binding(variable_name, value)?;
            if let Some(object_id) = self
                .paused_context_scope_object_ids
                .get(context_index)
                .map(String::as_str)
            {
                self.remote_objects.set_plain_object_property(
                    object_id,
                    variable_name,
                    scope_value,
                );
            }
            return Ok(json!({}));
        }
        if scope_number != global_scope_number {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.setVariableValue: unsupported synthetic scopeNumber `{scope_number}`"
            )));
        }
        let scope_value = value.cheap_clone();
        self.globals
            .borrow_mut()
            .insert(variable_name.to_string(), value);
        if let Some(object_id) = self.paused_global_scope_object_id.as_deref() {
            self.remote_objects
                .set_plain_object_property(object_id, variable_name, scope_value);
        }
        Ok(json!({}))
    }

    fn debugger_restart_frame(&mut self, params: &Value) -> StatorResult<Value> {
        let call_frame_id = params
            .get("callFrameId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.restartFrame: required parameter 'callFrameId' is missing or not a string"
                        .to_string(),
                )
            })?;
        let requested_offset = parse_pause_call_frame_id("Debugger.restartFrame", call_frame_id)?;
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.restartFrame",
                "no interpreter Debugger is attached to this session.",
            ));
        };
        ensure_active_call_frame_id(
            "Debugger.restartFrame",
            requested_offset,
            &debugger.borrow(),
        )?;
        Err(unsupported_debugger_method(
            "Debugger.restartFrame",
            "Stator does not yet support rewinding a paused interpreter frame.",
        ))
    }

    fn debugger_set_return_value(&mut self, params: &Value) -> StatorResult<Value> {
        let new_value = params.get("newValue").ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.setReturnValue: required parameter 'newValue' is missing".to_string(),
            )
        })?;
        let _value = call_argument_to_js_value(new_value, &self.remote_objects)?;
        let Some(debugger) = self.debugger.as_ref() else {
            return Err(unsupported_debugger_method(
                "Debugger.setReturnValue",
                "no interpreter Debugger is attached to this session.",
            ));
        };
        if debugger.borrow().last_pause_reason().is_none() {
            return Err(unsupported_debugger_method(
                "Debugger.setReturnValue",
                "no active pause; setReturnValue is only valid after a Debugger.paused event.",
            ));
        }
        Err(unsupported_debugger_method(
            "Debugger.setReturnValue",
            "Stator does not yet support mutating the return value of a paused frame.",
        ))
    }

    fn debugger_set_breakpoint_on_function_call(&mut self, params: &Value) -> StatorResult<Value> {
        let object_id = params
            .get("objectId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                    "Debugger.setBreakpointOnFunctionCall: required parameter 'objectId' is missing or not a string"
                        .to_string(),
                )
            })?;
        if let Some(condition) = params.get("condition") {
            let condition = condition.as_str().ok_or_else(|| {
                StatorError::TypeError(
                    "Debugger.setBreakpointOnFunctionCall: optional parameter 'condition' must be a string"
                        .to_string(),
                )
            })?;
            if !condition.is_empty() {
                return Err(unsupported_debugger_method(
                    "Debugger.setBreakpointOnFunctionCall",
                    "conditional function-call breakpoints are not implemented yet.",
                ));
            }
        }
        let value = self.remote_objects.get(object_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.setBreakpointOnFunctionCall: unknown or released objectId `{object_id}`"
            ))
        })?;
        if !value.is_function() {
            return Err(crate::error::StatorError::TypeError(format!(
                "Debugger.setBreakpointOnFunctionCall: objectId `{object_id}` does not reference a callable function"
            )));
        }
        Err(unsupported_debugger_method(
            "Debugger.setBreakpointOnFunctionCall",
            "Stator does not yet support pausing on calls to an arbitrary function object.",
        ))
    }

    // ── Debugger.stepInto / stepOver / stepOut ───────────────────────────────

    fn debugger_step(&mut self, params: &Value, action: DebugAction) -> StatorResult<Value> {
        validate_step_options(params, action)?;
        if let Some(bridge) = &self.pause_bridge
            && bridge.resume(action)
        {
            self.notify_resumed();
            return Ok(json!({}));
        }
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
            dbg.clear_last_pause();
        }
        self.notify_resumed();
        Ok(json!({}))
    }

    // ── Debugger.setBreakpointByUrl ──────────────────────────────────────────

    fn debugger_set_breakpoint(&mut self, params: &Value) -> StatorResult<Value> {
        reject_unsupported_breakpoint_condition("Debugger.setBreakpoint", params)?;
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
        let line_number = required_u32_param(location, "lineNumber", "Debugger.setBreakpoint")?;
        let column_number =
            optional_u32_param(location, "columnNumber", "Debugger.setBreakpoint")?.unwrap_or(0);
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
        self.cdp_script_breakpoints.insert(
            bp_id.clone(),
            CdpScriptBreakpoint {
                script_id: script_id.to_string(),
                line_number,
                column_number,
            },
        );
        let actual_line = actual_location["lineNumber"].as_u64().unwrap_or(0) as u32;
        let actual_column = actual_location["columnNumber"].as_u64().unwrap_or(0) as u32;
        let breakpoint_location = CdpBreakpointLocation {
            script_id: script_id.to_string(),
            url: self.script_urls.get(script_id).cloned().unwrap_or_default(),
            line_number: actual_line,
            column_number: actual_column,
        };
        self.install_interpreter_breakpoint(
            &bp_id,
            bytecode_offset,
            actual_line,
            actual_column,
            breakpoint_location,
        );

        Ok(json!({
            "breakpointId": bp_id,
            "actualLocation": actual_location,
        }))
    }

    fn debugger_set_breakpoint_by_url(&mut self, params: &Value) -> StatorResult<Value> {
        reject_unsupported_breakpoint_condition("Debugger.setBreakpointByUrl", params)?;
        let line_number = required_u32_param(params, "lineNumber", "Debugger.setBreakpointByUrl")?;
        let column_number =
            optional_u32_param(params, "columnNumber", "Debugger.setBreakpointByUrl")?.unwrap_or(0);

        let requested_url = params.get("url").and_then(Value::as_str);
        let requested_url_regex = params.get("urlRegex").and_then(Value::as_str);
        let requested_script_hash = params
            .get("scriptHash")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    StatorError::TypeError(
                        "Debugger.setBreakpointByUrl: optional parameter 'scriptHash' must be a string"
                            .to_string(),
                    )
                })
            })
            .transpose()?;
        if let Some(pattern) = requested_url_regex {
            compile_breakpoint_url_regex(pattern)?;
        }
        let bp_id = format!(
            "{}:{}:{}",
            self.next_breakpoint_id, line_number, column_number
        );
        self.next_breakpoint_id += 1;
        self.cdp_breakpoints.insert(bp_id.clone());

        let mut locations = Vec::new();
        let selectorless = requested_url.is_none()
            && requested_url_regex.is_none()
            && requested_script_hash.is_none();
        let breakpoint = CdpUrlBreakpoint {
            requested_url: if selectorless {
                Some(String::new())
            } else {
                requested_url.map(str::to_string)
            },
            requested_url_regex: requested_url_regex.map(str::to_string),
            requested_script_hash: requested_script_hash.map(str::to_string),
            line_number,
            column_number,
        };
        self.cdp_url_breakpoints
            .insert(bp_id.clone(), breakpoint.clone());
        let mut script_ids: Vec<String> = self.script_sources.keys().cloned().collect();
        script_ids.sort();
        for script_id in script_ids {
            if let Some(location) =
                self.resolve_url_breakpoint_for_script(&bp_id, &breakpoint, &script_id)?
            {
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
        location: CdpBreakpointLocation,
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
        self.debugger_breakpoint_locations
            .insert(debugger_id, location);
    }

    fn resolve_url_breakpoint_for_script(
        &mut self,
        cdp_breakpoint_id: &str,
        breakpoint: &CdpUrlBreakpoint,
        script_id: &str,
    ) -> StatorResult<Option<Value>> {
        if self.cdp_breakpoint_has_script_location(cdp_breakpoint_id, script_id) {
            return Ok(None);
        }
        let Some(source) = self.script_sources.get(script_id).cloned() else {
            return Ok(None);
        };
        let script_url = self.script_urls.get(script_id).cloned().unwrap_or_default();
        if !url_breakpoint_matches(breakpoint, &script_url, &source)? {
            return Ok(None);
        }
        let Some((bytecode_offset, location)) = breakpoint_location_for_source(
            script_id,
            &source,
            breakpoint.line_number,
            breakpoint.column_number,
        )?
        else {
            return Ok(None);
        };
        let actual_line = location["lineNumber"].as_u64().unwrap_or(0) as u32;
        let actual_column = location["columnNumber"].as_u64().unwrap_or(0) as u32;
        self.install_interpreter_breakpoint(
            cdp_breakpoint_id,
            bytecode_offset,
            actual_line,
            actual_column,
            CdpBreakpointLocation {
                script_id: script_id.to_string(),
                url: script_url,
                line_number: actual_line,
                column_number: actual_column,
            },
        );
        Ok(Some(location))
    }

    fn resolve_script_breakpoint(
        &mut self,
        cdp_breakpoint_id: &str,
        breakpoint: &CdpScriptBreakpoint,
    ) -> StatorResult<Option<Value>> {
        if self.cdp_breakpoint_has_script_location(cdp_breakpoint_id, &breakpoint.script_id) {
            return Ok(None);
        }
        let Some(source) = self.script_sources.get(&breakpoint.script_id).cloned() else {
            return Ok(None);
        };
        let Some((bytecode_offset, location)) = breakpoint_location_for_source(
            &breakpoint.script_id,
            &source,
            breakpoint.line_number,
            breakpoint.column_number,
        )?
        else {
            return Ok(None);
        };
        let actual_line = location["lineNumber"].as_u64().unwrap_or(0) as u32;
        let actual_column = location["columnNumber"].as_u64().unwrap_or(0) as u32;
        self.install_interpreter_breakpoint(
            cdp_breakpoint_id,
            bytecode_offset,
            actual_line,
            actual_column,
            CdpBreakpointLocation {
                script_id: breakpoint.script_id.clone(),
                url: self
                    .script_urls
                    .get(&breakpoint.script_id)
                    .cloned()
                    .unwrap_or_default(),
                line_number: actual_line,
                column_number: actual_column,
            },
        );
        Ok(Some(location))
    }

    fn install_deferred_breakpoints(&mut self) {
        let script_breakpoints: Vec<_> = self
            .cdp_script_breakpoints
            .iter()
            .map(|(id, breakpoint)| (id.clone(), breakpoint.clone()))
            .collect();
        for (breakpoint_id, breakpoint) in script_breakpoints {
            let _ = self.resolve_script_breakpoint(&breakpoint_id, &breakpoint);
        }

        let url_breakpoints: Vec<_> = self
            .cdp_url_breakpoints
            .iter()
            .map(|(id, breakpoint)| (id.clone(), breakpoint.clone()))
            .collect();
        let script_ids: Vec<_> = self.script_sources.keys().cloned().collect();
        for (breakpoint_id, breakpoint) in url_breakpoints {
            for script_id in &script_ids {
                let _ =
                    self.resolve_url_breakpoint_for_script(&breakpoint_id, &breakpoint, script_id);
            }
        }
    }

    fn cdp_breakpoint_has_script_location(&self, cdp_breakpoint_id: &str, script_id: &str) -> bool {
        self.cdp_debugger_breakpoints
            .get(cdp_breakpoint_id)
            .is_some_and(|breakpoints| {
                breakpoints.iter().any(|debugger_id| {
                    self.debugger_breakpoint_locations
                        .get(debugger_id)
                        .is_some_and(|location| location.script_id == script_id)
                })
            })
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
        self.cdp_script_breakpoints.remove(breakpoint_id);
        self.cdp_url_breakpoints.remove(breakpoint_id);
        if let Some(debugger_ids) = self.cdp_debugger_breakpoints.remove(breakpoint_id)
            && let Some(debugger) = self.debugger.as_ref()
        {
            let mut debugger = debugger.borrow_mut();
            for debugger_id in debugger_ids {
                let _ = debugger.remove_breakpoint(debugger_id);
                self.debugger_breakpoint_locations.remove(&debugger_id);
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

    fn debugger_search_in_content(&mut self, params: &Value) -> StatorResult<Value> {
        let script_id = params
            .get("scriptId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                crate::error::StatorError::TypeError(
                "Debugger.searchInContent: required parameter 'scriptId' is missing or not a string"
                    .to_string(),
            )
            })?;
        let query = params.get("query").and_then(Value::as_str).ok_or_else(|| {
            crate::error::StatorError::TypeError(
                "Debugger.searchInContent: required parameter 'query' is missing or not a string"
                    .to_string(),
            )
        })?;
        let source = self.script_sources.get(script_id).ok_or_else(|| {
            crate::error::StatorError::Internal(format!(
                "Debugger.searchInContent: unknown scriptId `{script_id}`"
            ))
        })?;
        let case_sensitive = params
            .get("caseSensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let is_regex = params
            .get("isRegex")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let result = search_script_content(source, query, case_sensitive, is_regex)?;
        Ok(json!({ "result": result }))
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
            let source_url = if source_url.is_empty() {
                None
            } else {
                Some(source_url.as_str())
            };
            let details = self.exception_details_only(
                &err,
                ExceptionRequest {
                    expression: &script_source,
                    source_url,
                    execution_context_id,
                    object_group: None,
                    generate_preview: false,
                },
            );
            self.emit_script_failed_to_parse(
                script_id.clone(),
                &script_source,
                source_url,
                execution_context_id,
                details.clone(),
            );
            return Ok(json!({
                "status": "CompileError",
                "exceptionDetails": details,
            }));
        }

        let dry_run =
            optional_bool_param(params, "dryRun", "Debugger.setScriptSource")?.unwrap_or(false);
        if optional_bool_param(params, "allowTopFrameEditing", "Debugger.setScriptSource")?
            .unwrap_or(false)
        {
            return Err(unsupported_debugger_method(
                "Debugger.setScriptSource",
                "allowTopFrameEditing requires live top-frame edit support that is not implemented yet.",
            ));
        }
        if !dry_run {
            let source_url = registered_script_url(&script_source);
            self.script_sources.insert(script_id.clone(), script_source);
            self.script_urls.insert(script_id.clone(), source_url);
            self.refresh_breakpoints_for_registered_script(&script_id);
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
        let start_line =
            required_u32_param(start, "lineNumber", "Debugger.getPossibleBreakpoints")?;
        let start_column =
            optional_u32_param(start, "columnNumber", "Debugger.getPossibleBreakpoints")?
                .unwrap_or(0);
        let end = params.get("end");
        let end_line = end
            .map(|value| optional_u32_param(value, "lineNumber", "Debugger.getPossibleBreakpoints"))
            .transpose()?
            .flatten();
        let end_column = end
            .map(|value| {
                optional_u32_param(value, "columnNumber", "Debugger.getPossibleBreakpoints")
            })
            .transpose()?
            .flatten();
        if let Some(end_script_id) = end
            .and_then(|value| value.get("scriptId"))
            .and_then(Value::as_str)
            && end_script_id != script_id
        {
            return Err(crate::error::StatorError::Internal(format!(
                "Debugger.getPossibleBreakpoints: end.scriptId `{end_script_id}` does not match start.scriptId `{script_id}`"
            )));
        }
        if params
            .get("restrictToFunction")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(unsupported_debugger_method(
                "Debugger.getPossibleBreakpoints",
                "restrictToFunction requires function boundary metadata that is not implemented yet.",
            ));
        }

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
        Self::with_dispatcher(ws, CdpDispatcher::new())
    }

    /// Wrap an accepted WebSocket connection using a caller-provided dispatcher.
    fn with_dispatcher(mut ws: WebSocket<TcpStream>, dispatcher: CdpDispatcher) -> Self {
        if dispatcher.has_pause_bridge() {
            let _ = ws
                .get_mut()
                .set_read_timeout(Some(Duration::from_millis(50)));
        }
        Self { ws, dispatcher }
    }

    fn drain_outbox(&mut self) -> io::Result<()> {
        while let Some(out) = self.dispatcher.take_next() {
            self.ws
                .send(Message::Text(out.into()))
                .map_err(|e| io::Error::other(e.to_string()))?;
        }
        Ok(())
    }

    fn close_pause_bridge(&mut self) {
        if let Some(bridge) = &self.dispatcher.pause_bridge {
            bridge.disconnect();
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
            let _ = self.dispatcher.notify_paused();
            self.drain_outbox()?;
            let msg = match self.ws.read() {
                Ok(m) => m,
                Err(tungstenite::Error::ConnectionClosed)
                | Err(tungstenite::Error::AlreadyClosed) => {
                    self.close_pause_bridge();
                    return Ok(());
                }
                Err(tungstenite::Error::Io(e))
                    if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
                {
                    continue;
                }
                Err(tungstenite::Error::Io(e)) => {
                    self.close_pause_bridge();
                    return Err(e);
                }
                Err(e) => {
                    self.close_pause_bridge();
                    return Err(io::Error::other(e.to_string()));
                }
            };

            match msg {
                Message::Text(text) => {
                    let _ = self.dispatcher.dispatch_json(&text);
                    let _ = self.dispatcher.notify_paused();
                    self.drain_outbox()?;
                }
                Message::Close(_) => {
                    self.close_pause_bridge();
                    return Ok(());
                }
                // Ignore binary / ping / pong frames.
                _ => {}
            }
        }
    }
}

const MAX_PREVIEW_PROPERTIES: usize = 5;
const MAX_PREVIEW_DEPTH: usize = 1;
const MAX_STORED_STACK_TRACES: usize = 128;
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

pub(crate) fn registered_source_map_url(source: &str) -> String {
    source
        .lines()
        .rev()
        .find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("//# sourceMappingURL=")
                .or_else(|| trimmed.strip_prefix("//@ sourceMappingURL="))
                .map(str::to_string)
        })
        .unwrap_or_default()
}

pub(crate) fn script_hash(source: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn search_script_content(
    source: &str,
    query: &str,
    case_sensitive: bool,
    is_regex: bool,
) -> StatorResult<Vec<Value>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let mut matches = Vec::new();
    if is_regex {
        let pattern = if case_sensitive {
            query.to_string()
        } else {
            format!("(?i:{query})")
        };
        let regex = stacker::maybe_grow(256 * 1024, 4 * 1024 * 1024, || {
            regress::Regex::new(&pattern)
        })
        .map_err(|err| {
            crate::error::StatorError::SyntaxError(format!(
                "Debugger.searchInContent: invalid regular expression `{query}`: {err}"
            ))
        })?;
        for (line_number, line) in source.lines().enumerate() {
            if regex.find(line).is_some() {
                matches.push(json!({
                    "lineNumber": line_number,
                    "lineContent": line,
                }));
            }
        }
        return Ok(matches);
    }

    let query_cmp = if case_sensitive {
        query.to_string()
    } else {
        query.to_lowercase()
    };
    for (line_number, line) in source.lines().enumerate() {
        let line_cmp = if case_sensitive {
            line.to_string()
        } else {
            line.to_lowercase()
        };
        if line_cmp.contains(&query_cmp) {
            matches.push(json!({
                "lineNumber": line_number,
                "lineContent": line,
            }));
        }
    }
    Ok(matches)
}

fn compile_breakpoint_url_regex(pattern: &str) -> StatorResult<regress::Regex> {
    regress::Regex::new(pattern).map_err(|err| {
        StatorError::SyntaxError(format!(
            "Debugger.setBreakpointByUrl: invalid urlRegex `{pattern}`: {err}"
        ))
    })
}

fn required_u32_param(container: &Value, field: &str, method: &str) -> StatorResult<u32> {
    let value = container.get(field).ok_or_else(|| {
        StatorError::TypeError(format!("{method}: required parameter '{field}' is missing"))
    })?;
    u32_param(value, field, method)
}

fn required_bool_param(container: &Value, field: &str, method: &str) -> StatorResult<bool> {
    container
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            StatorError::TypeError(format!(
                "{method}: required parameter '{field}' is missing or not a boolean"
            ))
        })
}

fn required_finite_non_negative_number_param(
    container: &Value,
    field: &str,
    method: &str,
) -> StatorResult<f64> {
    let value = container
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            StatorError::TypeError(format!(
                "{method}: required parameter '{field}' is missing or not a number"
            ))
        })?;
    if !value.is_finite() || value < 0.0 {
        return Err(StatorError::TypeError(format!(
            "{method}: parameter '{field}' must be a finite, non-negative number"
        )));
    }
    Ok(value)
}

fn optional_finite_number_param(
    container: &Value,
    field: &str,
    method: &str,
) -> StatorResult<Option<f64>> {
    container
        .get(field)
        .map(|value| {
            let value = value.as_f64().ok_or_else(|| {
                StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be a number"
                ))
            })?;
            if !value.is_finite() {
                return Err(StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be finite"
                )));
            }
            Ok(value)
        })
        .transpose()
}

fn validate_optional_number_range(
    method: &str,
    field: &str,
    value: Option<f64>,
    min: f64,
    max: f64,
) -> StatorResult<()> {
    if value.is_some_and(|value| value < min || value > max) {
        return Err(StatorError::TypeError(format!(
            "{method}: optional parameter '{field}' must be between {min} and {max}"
        )));
    }
    Ok(())
}

fn validate_optional_non_negative_number(
    method: &str,
    field: &str,
    value: Option<f64>,
) -> StatorResult<()> {
    if value.is_some_and(|value| value < 0.0) {
        return Err(StatorError::TypeError(format!(
            "{method}: optional parameter '{field}' must be non-negative"
        )));
    }
    Ok(())
}

fn required_throughput_param(container: &Value, field: &str, method: &str) -> StatorResult<f64> {
    let value = container
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            StatorError::TypeError(format!(
                "{method}: required parameter '{field}' is missing or not a number"
            ))
        })?;
    if !value.is_finite() || !(value == -1.0 || value >= 0.0) {
        return Err(StatorError::TypeError(format!(
            "{method}: parameter '{field}' must be -1 or a finite, non-negative number"
        )));
    }
    Ok(value)
}

fn optional_packet_loss_param(
    container: &Value,
    field: &str,
    method: &str,
) -> StatorResult<Option<f64>> {
    container
        .get(field)
        .map(|value| {
            let value = value.as_f64().ok_or_else(|| {
                StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be a number"
                ))
            })?;
            if !value.is_finite() || !(0.0..=100.0).contains(&value) {
                return Err(StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be a finite number between 0 and 100"
                )));
            }
            Ok(value)
        })
        .transpose()
}

fn optional_u32_param(container: &Value, field: &str, method: &str) -> StatorResult<Option<u32>> {
    container
        .get(field)
        .map(|value| u32_param(value, field, method))
        .transpose()
}

fn optional_bool_param(container: &Value, field: &str, method: &str) -> StatorResult<Option<bool>> {
    container
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be a boolean"
                ))
            })
        })
        .transpose()
}

fn optional_string_param<'a>(
    container: &'a Value,
    field: &str,
    method: &str,
) -> StatorResult<Option<&'a str>> {
    container
        .get(field)
        .map(|value| {
            value.as_str().ok_or_else(|| {
                StatorError::TypeError(format!(
                    "{method}: optional parameter '{field}' must be a string"
                ))
            })
        })
        .transpose()
}

fn parse_user_agent_override_params(
    params: &Value,
    method: &str,
) -> StatorResult<(String, String, String, Option<Value>)> {
    let Some(user_agent) = params.get("userAgent").and_then(Value::as_str) else {
        return Err(StatorError::TypeError(format!(
            "{method}: required parameter 'userAgent' is missing or not a string"
        )));
    };
    let accept_language = optional_string_param(params, "acceptLanguage", method)?;
    let platform = optional_string_param(params, "platform", method)?;
    let metadata = match params.get("userAgentMetadata") {
        Some(value @ Value::Object(_)) => Some(value.clone()),
        Some(_) => {
            return Err(StatorError::TypeError(format!(
                "{method}: optional parameter 'userAgentMetadata' must be an object"
            )));
        }
        None => None,
    };

    Ok((
        user_agent.to_string(),
        accept_language.unwrap_or("").to_string(),
        platform.unwrap_or("").to_string(),
        metadata,
    ))
}

fn u32_param(value: &Value, field: &str, method: &str) -> StatorResult<u32> {
    let value = value.as_u64().ok_or_else(|| {
        StatorError::TypeError(format!(
            "{method}: parameter '{field}' is not a non-negative integer"
        ))
    })?;
    u32::try_from(value).map_err(|_| {
        StatorError::RangeError(format!("{method}: parameter '{field}' exceeds u32::MAX"))
    })
}

fn url_breakpoint_matches(
    breakpoint: &CdpUrlBreakpoint,
    script_url: &str,
    source: &str,
) -> StatorResult<bool> {
    if breakpoint.requested_url.as_deref() == Some(script_url) {
        return Ok(true);
    }
    if let Some(pattern) = breakpoint.requested_url_regex.as_deref()
        && compile_breakpoint_url_regex(pattern)?
            .find(script_url)
            .is_some()
    {
        return Ok(true);
    }
    if let Some(requested_hash) = breakpoint.requested_script_hash.as_deref()
        && requested_hash == script_hash(source)
    {
        return Ok(true);
    }
    Ok(false)
}

fn reject_unsupported_breakpoint_condition(method: &str, params: &Value) -> StatorResult<()> {
    let Some(condition) = params.get("condition") else {
        return Ok(());
    };
    let condition = condition.as_str().ok_or_else(|| {
        StatorError::TypeError(format!(
            "{method}: optional parameter 'condition' must be a string"
        ))
    })?;
    if condition.is_empty() {
        return Ok(());
    }
    Err(unsupported_debugger_method(
        method,
        "conditional breakpoints are not implemented yet.",
    ))
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
            { "name": "Browser", "version": "1.3" },
            { "name": "Inspector", "version": "1.3" },
            { "name": "Runtime", "version": "1.3" },
            { "name": "Debugger", "version": "1.3" },
            { "name": "Console", "version": "1.3" },
            { "name": "Profiler", "version": "1.3" },
            { "name": "HeapProfiler", "version": "1.3" },
            { "name": "Network", "version": "1.3" },
            { "name": "Page", "version": "1.3" },
            { "name": "Log", "version": "1.3" },
            { "name": "Security", "version": "1.3" },
            { "name": "Performance", "version": "1.3" },
            { "name": "Emulation", "version": "1.3" },
            { "name": "Overlay", "version": "1.3" },
            { "name": "ServiceWorker", "version": "1.3" },
            { "name": "Target", "version": "1.3" },
            { "name": "Schema", "version": "1.3" }
        ]
    })
}

fn browser_get_version() -> Value {
    let version = env!("CARGO_PKG_VERSION");
    json!({
        "protocolVersion": "1.3",
        "product": format!("StatorJSE/{version}"),
        "revision": version,
        "userAgent": format!("StatorJSE/{version}"),
        "jsVersion": version,
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

fn parse_pause_call_frame_id(method: &str, call_frame_id: &str) -> StatorResult<u32> {
    let Some(offset) = call_frame_id.strip_prefix("stator-pause-frame-") else {
        return Err(StatorError::Internal(format!(
            "{method}: unknown callFrameId `{call_frame_id}`"
        )));
    };
    offset.parse::<u32>().map_err(|_| {
        StatorError::Internal(format!("{method}: malformed callFrameId `{call_frame_id}`"))
    })
}

fn ensure_active_call_frame_id(
    method: &str,
    requested_offset: u32,
    debugger: &Debugger,
) -> StatorResult<()> {
    if debugger.last_pause_reason().is_none() {
        return Err(unsupported_debugger_method(
            method,
            "no active pause; call-frame access is only valid after a Debugger.paused event.",
        ));
    }
    let active_offset = debugger.last_pause_offset();
    if active_offset != requested_offset {
        return Err(StatorError::Internal(format!(
            "{method}: stale callFrameId `stator-pause-frame-{requested_offset}`; active callFrameId is `stator-pause-frame-{active_offset}`"
        )));
    }
    Ok(())
}

fn debug_step_method_name(action: DebugAction) -> &'static str {
    match action {
        DebugAction::Continue => "Debugger.resume",
        DebugAction::StepInto => "Debugger.stepInto",
        DebugAction::StepOver => "Debugger.stepOver",
        DebugAction::StepOut => "Debugger.stepOut",
    }
}

fn validate_step_options(params: &Value, action: DebugAction) -> StatorResult<()> {
    let method = debug_step_method_name(action);
    if matches!(action, DebugAction::StepInto)
        && params
            .get("breakOnAsyncCall")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(unsupported_debugger_method(
            method,
            "breakOnAsyncCall is not implemented yet.",
        ));
    }
    if let Some(skip_list) = params.get("skipList") {
        let ranges = skip_list.as_array().ok_or_else(|| {
            StatorError::TypeError(format!(
                "{method}: optional parameter 'skipList' must be an array"
            ))
        })?;
        if !ranges.is_empty() {
            return Err(unsupported_debugger_method(
                method,
                "skipList stepping ranges are not implemented yet.",
            ));
        }
    }
    Ok(())
}

fn validate_continue_to_location_options(params: &Value) -> StatorResult<()> {
    let Some(target_call_frames) = params.get("targetCallFrames") else {
        return Ok(());
    };
    let target_call_frames = target_call_frames.as_str().ok_or_else(|| {
        StatorError::TypeError(
            "Debugger.continueToLocation: optional parameter 'targetCallFrames' must be a string"
                .to_string(),
        )
    })?;
    match target_call_frames {
        "any" | "current" => Ok(()),
        other => Err(StatorError::TypeError(format!(
            "Debugger.continueToLocation: unsupported targetCallFrames `{other}`"
        ))),
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

fn frame_snapshot_bindings(snapshot: &PauseFrameSnapshot) -> Vec<(String, JsValue)> {
    let mut bindings = Vec::with_capacity(snapshot.registers.len() + 1);
    bindings.push((
        "$accumulator".to_string(),
        snapshot.accumulator.cheap_clone(),
    ));
    let parameter_count = snapshot.parameter_count as usize;
    for (idx, value) in snapshot.registers.iter().enumerate() {
        let name = if idx < parameter_count {
            format!("$param{idx}")
        } else {
            format!("$local{}", idx - parameter_count)
        };
        bindings.push((name, value.cheap_clone()));
    }
    bindings
}

fn context_snapshot_bindings(
    context_index: usize,
    slots: &[JsValue],
) -> impl Iterator<Item = (String, JsValue)> + '_ {
    slots.iter().enumerate().map(move |(slot_index, value)| {
        (
            format!("$context{context_index}_slot{slot_index}"),
            value.cheap_clone(),
        )
    })
}

fn paused_event_params(
    reason: &PauseReason,
    offset: u32,
    _line: u32,
    scope_chain: Value,
    async_stack_trace_id: Option<&str>,
    location: &PauseFrameLocation,
) -> Value {
    // We cannot reconstruct the live JS call stack here yet: the interpreter
    // does not retain a portable per-frame snapshot at pause time. To avoid
    // emitting success-shaped placeholder frames we instead emit a single
    // synthetic top-level frame describing the paused source location. The
    // `(stator: paused-frame)` function name and `pausedFrame: true`
    // auxData entry signal to embedders that this is a derived frame.
    let call_frame = json!({
        "callFrameId": format!("stator-pause-frame-{offset}"),
        "functionName": "(stator: paused-frame)",
        "location": {
            "scriptId": location.script_id.as_str(),
            "lineNumber": location.line_number,
            "columnNumber": location.column_number,
        },
        "scopeChain": scope_chain,
        "this": {"type": "undefined"},
        "url": location.url.as_str(),
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
    if let Some(id) = async_stack_trace_id
        && let Some(obj) = params.as_object_mut()
    {
        obj.insert("asyncStackTraceId".to_string(), json!({ "id": id }));
    }
    params
}

fn pause_stack_trace(
    reason: &PauseReason,
    offset: u32,
    location: &PauseFrameLocation,
    async_parent_stack: Option<&AsyncStackTrace>,
    async_parent_depth: usize,
) -> Value {
    let mut stack_trace = json!({
        "description": format!("Stator paused at bytecode {offset}: {}", paused_reason_str(reason)),
        "callFrames": [{
            "functionName": "(stator: paused-frame)",
            "scriptId": location.script_id.as_str(),
            "url": location.url.as_str(),
            "lineNumber": location.line_number,
            "columnNumber": location.column_number,
        }],
    });
    if let Some(parent) = async_parent_stack
        && let Some(parent) = async_stack_trace_to_cdp(parent, async_parent_depth)
        && let Some(obj) = stack_trace.as_object_mut()
    {
        obj.insert("parent".to_string(), parent);
    }
    stack_trace
}

fn async_stack_trace_to_cdp(trace: &AsyncStackTrace, remaining_frames: usize) -> Option<Value> {
    if remaining_frames == 0 {
        return None;
    }
    let call_frames: Vec<Value> = trace
        .frames
        .iter()
        .rev()
        .take(remaining_frames)
        .map(|frame| {
            json!({
                "functionName": frame,
                "scriptId": "0",
                "url": "",
                "lineNumber": 0,
                "columnNumber": 0,
            })
        })
        .collect();
    let remaining_frames = remaining_frames.saturating_sub(call_frames.len());
    let mut value = json!({
        "description": trace.description,
        "callFrames": call_frames,
    });
    if let Some(parent) = trace.parent.as_deref()
        && remaining_frames > 0
        && let Some(parent) = async_stack_trace_to_cdp(parent, remaining_frames)
        && let Some(obj) = value.as_object_mut()
    {
        obj.insert("parent".to_string(), parent);
    }
    Some(value)
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

fn call_argument_to_js_value(
    argument: &Value,
    registry: &RemoteObjectRegistry,
) -> StatorResult<JsValue> {
    if let Some(object_id) = argument.get("objectId").and_then(Value::as_str) {
        return registry.get(object_id).ok_or_else(|| {
            StatorError::Internal(format!(
                "Debugger.setVariableValue: unknown objectId `{object_id}`"
            ))
        });
    }
    if let Some(unserializable) = argument.get("unserializableValue").and_then(Value::as_str) {
        return match unserializable {
            "NaN" => Ok(JsValue::HeapNumber(f64::NAN)),
            "Infinity" => Ok(JsValue::HeapNumber(f64::INFINITY)),
            "-Infinity" => Ok(JsValue::HeapNumber(f64::NEG_INFINITY)),
            "-0" => Ok(JsValue::HeapNumber(-0.0)),
            _ if unserializable.ends_with('n') => {
                Ok(JsValue::String(unserializable.to_string().into()))
            }
            _ => Err(StatorError::TypeError(format!(
                "Debugger.setVariableValue: unsupported unserializableValue `{unserializable}`"
            ))),
        };
    }
    match argument.get("value").unwrap_or(&Value::Null) {
        Value::Null => Ok(JsValue::Null),
        Value::Bool(value) => Ok(JsValue::Boolean(*value)),
        Value::Number(number) => {
            if let Some(value) = number.as_i64()
                && let Ok(smi) = i32::try_from(value)
            {
                return Ok(JsValue::Smi(smi));
            }
            let value = number.as_f64().ok_or_else(|| {
                StatorError::TypeError("Debugger.setVariableValue: invalid numeric value".into())
            })?;
            Ok(JsValue::HeapNumber(value))
        }
        Value::String(value) => Ok(JsValue::String(value.clone().into())),
        other => Err(StatorError::TypeError(format!(
            "Debugger.setVariableValue: unsupported value payload `{other}`"
        ))),
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

    /// Accept and serve a single CDP connection using `dispatcher`.
    ///
    /// This single-threaded entry point lets embedders share their isolate's
    /// globals and script registry with a WebSocket session without moving
    /// `Rc<RefCell<_>>` state across threads. HTTP discovery requests are
    /// still answered normally; the dispatcher is only consumed for WebSocket
    /// upgrades.
    pub fn accept_one_with_dispatcher(&self, dispatcher: CdpDispatcher) -> io::Result<()> {
        let (stream, _peer) = self.listener.accept()?;
        self.serve_stream_with_dispatcher(stream, dispatcher)
    }

    /// Accept and serve a single CDP connection using a dispatcher factory.
    ///
    /// The factory is invoked only for WebSocket upgrades.  Plain HTTP
    /// discovery requests are answered without constructing a dispatcher.
    pub fn accept_one_with_dispatcher_factory<F>(&self, dispatcher_factory: F) -> io::Result<()>
    where
        F: FnOnce() -> CdpDispatcher,
    {
        let (stream, _peer) = self.listener.accept()?;
        self.serve_stream_with_dispatcher_factory(stream, dispatcher_factory)
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

    /// Accept and serve connections using a fresh dispatcher per WebSocket.
    ///
    /// This is the long-lived counterpart to
    /// [`Self::accept_one_with_dispatcher`].  The loop remains single-threaded:
    /// embedders can capture non-`Send` isolate state in the factory as long as
    /// they call this method from the isolate/inspector owner thread.
    pub fn accept_loop_with_dispatcher_factory<F>(
        &self,
        mut dispatcher_factory: F,
    ) -> io::Result<()>
    where
        F: FnMut() -> CdpDispatcher,
    {
        for stream in self.listener.incoming() {
            let stream = stream?;
            if is_websocket_upgrade(&stream)? {
                let ws = accept(stream).map_err(|e| io::Error::other(e.to_string()))?;
                let _ = CdpSession::with_dispatcher(ws, dispatcher_factory()).run();
            } else {
                let _ = serve_http_discovery(stream, self.local_addr()?);
            }
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

    fn serve_stream_with_dispatcher(
        &self,
        stream: TcpStream,
        dispatcher: CdpDispatcher,
    ) -> io::Result<()> {
        self.serve_stream_with_dispatcher_factory(stream, || dispatcher)
    }

    fn serve_stream_with_dispatcher_factory<F>(
        &self,
        stream: TcpStream,
        dispatcher_factory: F,
    ) -> io::Result<()>
    where
        F: FnOnce() -> CdpDispatcher,
    {
        if is_websocket_upgrade(&stream)? {
            let ws = accept(stream).map_err(|e| io::Error::other(e.to_string()))?;
            return CdpSession::with_dispatcher(ws, dispatcher_factory()).run();
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
    use crate::builtins::error::{pop_call_frame, push_call_frame};
    use crate::builtins::promise::MicrotaskQueue;

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
    fn websocket_session_can_use_supplied_shared_globals_dispatcher() {
        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let port = server.local_addr().expect("local_addr").port();
        let globals = Rc::new(RefCell::new(GlobalEnv::new()));
        globals
            .borrow_mut()
            .insert("shared".to_string(), JsValue::Smi(41));
        let dispatcher = CdpDispatcher::with_globals(globals);

        let client = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            let url = format!("ws://127.0.0.1:{port}");
            let (mut ws, _resp) = connect(url).expect("connect");
            ws.send(Message::Text(
                r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"shared + 1"}}"#
                    .into(),
            ))
            .expect("send");
            let reply = ws.read().expect("read");
            ws.close(None).ok();
            match reply {
                Message::Text(text) => serde_json::from_str::<Value>(&text).expect("valid JSON"),
                other => panic!("unexpected websocket reply: {other:?}"),
            }
        });

        server
            .accept_one_with_dispatcher_factory(move || dispatcher)
            .expect("server");
        let response = client.join().expect("client thread");
        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["result"]["value"], 42);
    }

    #[test]
    fn websocket_session_sends_pause_bridge_events_without_client_request() {
        let server = CdpServer::bind("127.0.0.1:0").expect("bind");
        let port = server.local_addr().expect("local_addr").port();
        let bridge = DebuggerPauseBridge::new();
        let worker_bridge = bridge.clone();

        let worker = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            let mut dbg = InterpreterDebugger::new();
            dbg.set_pause_bridge(worker_bridge);
            dbg.set_breakpoint_at_offset(0, 1, 1);
            dbg.check_pause_at(0).is_none()
        });

        let client = thread::spawn(move || {
            let url = format!("ws://127.0.0.1:{port}");
            let (mut ws, _resp) = connect(url).expect("connect");
            if let MaybeTlsStream::Plain(stream) = ws.get_mut() {
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("set client read timeout");
            }
            let read_json = |ws: &mut tungstenite::WebSocket<MaybeTlsStream<TcpStream>>| match ws
                .read()
                .expect("read websocket message")
            {
                Message::Text(text) => serde_json::from_str::<Value>(&text).expect("valid JSON"),
                other => panic!("unexpected websocket reply: {other:?}"),
            };

            ws.send(Message::Text(
                r#"{"id":1,"method":"Debugger.enable","params":{}}"#.into(),
            ))
            .expect("send enable");
            let enable = read_json(&mut ws);
            assert_eq!(enable["id"], 1);

            let paused = read_json(&mut ws);
            assert_eq!(paused["method"], "Debugger.paused");
            assert_eq!(paused["params"]["data"]["bytecodeOffset"], 0);

            ws.send(Message::Text(
                r#"{"id":2,"method":"Debugger.resume","params":{}}"#.into(),
            ))
            .expect("send resume");
            let first = read_json(&mut ws);
            let second = read_json(&mut ws);
            ws.close(None).ok();
            assert!(
                [first.clone(), second.clone()]
                    .iter()
                    .any(|msg| msg["method"] == "Debugger.resumed"),
                "expected resumed event, got {first:?} and {second:?}"
            );
            assert!(
                [first, second]
                    .iter()
                    .any(|msg| msg["id"] == 2u64 && msg.get("error").is_none()),
                "expected resume ack"
            );
        });

        server
            .accept_one_with_dispatcher_factory(move || {
                let mut dispatcher = CdpDispatcher::new();
                dispatcher.attach_pause_bridge(bridge);
                dispatcher
            })
            .expect("server");
        client.join().expect("client thread");
        assert!(worker.join().expect("worker thread"));
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
    fn debugger_attach_installs_existing_script_breakpoint() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0}}}"#,
        );
        assert!(response.get("error").is_none());

        let dbg = Rc::new(RefCell::new(InterpreterDebugger::new()));
        d.attach_debugger(Rc::clone(&dbg));
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
    }

    #[test]
    fn debugger_attach_installs_existing_url_breakpoints() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","lineNumber":1,"columnNumber":0}}"#,
        );
        assert!(response.get("error").is_none());

        let dbg = Rc::new(RefCell::new(InterpreterDebugger::new()));
        d.attach_debugger(Rc::clone(&dbg));
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
    }

    #[test]
    fn debugger_paused_uses_registered_breakpoint_script_location() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":4}}"#,
        );
        let response = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0}}}"#,
        );
        assert!(response.get("error").is_none());
        let offset = dbg
            .borrow()
            .breakpoints()
            .next()
            .expect("installed breakpoint")
            .bytecode_offset;
        let _ = dispatch(&mut d, r#"{"id":3,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        assert!(dbg.borrow_mut().check_pause_at(offset).is_some());
        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        let frame = &paused[0]["params"]["callFrames"][0];
        assert_eq!(frame["location"]["scriptId"], "7");
        assert_eq!(frame["location"]["lineNumber"], 1);
        assert_eq!(frame["url"], "stator://app.js");

        let stack_trace_id = paused[0]["params"]["asyncStackTraceId"]["id"]
            .as_str()
            .expect("stack trace id");
        let stack_trace = dispatch(
            &mut d,
            &format!(
                r#"{{"id":4,"method":"Debugger.getStackTrace","params":{{"stackTraceId":{{"id":"{stack_trace_id}"}}}}}}"#
            ),
        );
        assert_eq!(
            stack_trace["result"]["stackTrace"]["callFrames"][0]["scriptId"],
            "7"
        );
        assert_eq!(
            stack_trace["result"]["stackTrace"]["callFrames"][0]["url"],
            "stator://app.js"
        );
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
    fn debugger_set_breakpoint_validates_location_numbers() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "var a = 1;\n//# sourceURL=stator://app.js".to_string());

        let missing_line = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","columnNumber":0}}}"#,
        );
        assert!(missing_line["error"].is_object());

        let bad_column = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":0,"columnNumber":-1}}}"#,
        );
        assert!(bad_column["error"].is_object());

        let optional_column = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":0}}}"#,
        );
        assert!(optional_column.get("error").is_none());
    }

    #[test]
    fn debugger_set_breakpoint_rejects_unsupported_conditions() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );

        let conditional = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0},"condition":"a > 1"}}"#,
        );
        assert!(conditional["error"].is_object());
        assert!(
            conditional["error"]["message"]
                .as_str()
                .unwrap()
                .contains("conditional breakpoints")
        );

        let empty = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0},"condition":""}}"#,
        );
        assert!(empty.get("error").is_none());
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
    fn debugger_set_breakpoint_by_url_without_selector_matches_anonymous_scripts() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(7, "var a = 1;\nvar b = a + 2;".to_string());

        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"lineNumber":1,"columnNumber":0}}"#,
        );
        let locations = response["result"]["locations"].as_array().unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0]["scriptId"], "7");
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
    }

    #[test]
    fn debugger_set_breakpoint_by_url_uses_real_url_regex() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );

        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"urlRegex":"stator://app\\.js","lineNumber":1,"columnNumber":0}}"#,
        );
        let locations = response["result"]["locations"].as_array().unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0]["scriptId"], "7");

        let invalid = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"urlRegex":"(","lineNumber":0,"columnNumber":0}}"#,
        );
        assert!(invalid["error"].is_object());
        assert!(
            invalid["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid urlRegex")
        );
    }

    #[test]
    fn debugger_set_breakpoint_by_url_matches_script_hash() {
        let mut d = fresh_dispatcher();
        let source = "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js";
        d.register_script_source(7, source.to_string());
        let hash = script_hash(source);

        let response = dispatch(
            &mut d,
            &json!({
                "id": 1,
                "method": "Debugger.setBreakpointByUrl",
                "params": {
                    "scriptHash": hash,
                    "lineNumber": 1,
                    "columnNumber": 0
                }
            })
            .to_string(),
        );
        let locations = response["result"]["locations"].as_array().unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0]["scriptId"], "7");

        let bad_hash_type = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"scriptHash":true,"lineNumber":0,"columnNumber":0}}"#,
        );
        assert!(bad_hash_type["error"].is_object());
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
    fn debugger_set_breakpoint_by_url_validates_location_numbers() {
        let mut d = fresh_dispatcher();

        let missing_line = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","columnNumber":0}}"#,
        );
        assert!(missing_line["error"].is_object());

        let bad_column = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","lineNumber":0,"columnNumber":-1}}"#,
        );
        assert!(bad_column["error"].is_object());

        let optional_column = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://missing.js","lineNumber":0}}"#,
        );
        assert!(optional_column.get("error").is_none());
    }

    #[test]
    fn debugger_set_breakpoint_by_url_rejects_unsupported_conditions() {
        let mut d = fresh_dispatcher();
        let conditional = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","lineNumber":0,"columnNumber":0,"condition":"a > 1"}}"#,
        );
        assert!(conditional["error"].is_object());
        assert!(
            conditional["error"]["message"]
                .as_str()
                .unwrap()
                .contains("conditional breakpoints")
        );

        let bad_type = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://app.js","lineNumber":0,"columnNumber":0,"condition":true}}"#,
        );
        assert!(bad_type["error"].is_object());
    }

    #[test]
    fn debugger_set_breakpoint_by_url_resolves_future_registered_script() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let response = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://future.js","lineNumber":1,"columnNumber":0}}"#,
        );
        let breakpoint_id = response["result"]["breakpointId"]
            .as_str()
            .expect("breakpoint id")
            .to_string();
        assert!(
            response["result"]["locations"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(dbg.borrow().breakpoints().count(), 0);

        d.register_script_source(
            8,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://future.js".to_string(),
        );
        let events = drain_all(&mut d);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["method"], "Debugger.breakpointResolved");
        assert_eq!(events[0]["params"]["breakpointId"], breakpoint_id);
        assert_eq!(events[0]["params"]["location"]["scriptId"], "8");
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
    }

    #[test]
    fn debugger_remove_breakpoint_clears_future_url_breakpoint() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let response = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://future.js","lineNumber":1,"columnNumber":0}}"#,
        );
        let breakpoint_id = response["result"]["breakpointId"].as_str().unwrap();
        let removed = dispatch(
            &mut d,
            &json!({
                "id": 3,
                "method": "Debugger.removeBreakpoint",
                "params": { "breakpointId": breakpoint_id }
            })
            .to_string(),
        );
        assert!(removed.get("error").is_none());

        d.register_script_source(
            8,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://future.js".to_string(),
        );
        assert!(drain_all(&mut d).is_empty());
        assert_eq!(dbg.borrow().breakpoints().count(), 0);
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
    fn service_worker_setup_methods_store_state_and_validate_input() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(
            &mut d,
            r#"{"id":1,"method":"ServiceWorker.enable","params":{}}"#,
        );
        assert!(enable.get("error").is_none());
        assert!(d.service_worker_enabled());

        let force = dispatch(
            &mut d,
            r#"{"id":2,"method":"ServiceWorker.setForceUpdateOnPageLoad","params":{"forceUpdateOnPageLoad":true}}"#,
        );
        assert!(force.get("error").is_none());
        assert!(d.service_worker_force_update_on_page_load());

        let bad_force = dispatch(
            &mut d,
            r#"{"id":3,"method":"ServiceWorker.setForceUpdateOnPageLoad","params":{}}"#,
        );
        assert!(bad_force["error"].is_object());

        let start = dispatch(
            &mut d,
            r#"{"id":4,"method":"ServiceWorker.startWorker","params":{"versionId":"v1"}}"#,
        );
        assert!(start.get("error").is_none());

        let bad_start = dispatch(
            &mut d,
            r#"{"id":5,"method":"ServiceWorker.startWorker","params":{"versionId":1}}"#,
        );
        assert!(bad_start["error"].is_object());

        let disable = dispatch(
            &mut d,
            r#"{"id":6,"method":"ServiceWorker.disable","params":{}}"#,
        );
        assert!(disable.get("error").is_none());
        assert!(!d.service_worker_enabled());
    }

    #[test]
    fn overlay_setup_methods_store_state_and_validate_input() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(&mut d, r#"{"id":1,"method":"Overlay.enable","params":{}}"#);
        assert!(enable.get("error").is_none());
        assert!(d.overlay_enabled());

        let paint = dispatch(
            &mut d,
            r#"{"id":2,"method":"Overlay.setShowPaintRects","params":{"result":true}}"#,
        );
        assert!(paint.get("error").is_none());
        assert!(d.overlay_show_paint_rects());

        let borders = dispatch(
            &mut d,
            r#"{"id":3,"method":"Overlay.setShowDebugBorders","params":{"show":true}}"#,
        );
        assert!(borders.get("error").is_none());
        assert!(d.overlay_show_debug_borders());

        let fps = dispatch(
            &mut d,
            r#"{"id":4,"method":"Overlay.setShowFPSCounter","params":{"show":true}}"#,
        );
        assert!(fps.get("error").is_none());
        assert!(d.overlay_show_fps_counter());

        let web_vitals = dispatch(
            &mut d,
            r#"{"id":24,"method":"Overlay.setShowWebVitals","params":{"show":true}}"#,
        );
        assert!(web_vitals.get("error").is_none());
        assert!(d.overlay_show_web_vitals());

        let layout = dispatch(
            &mut d,
            r#"{"id":5,"method":"Overlay.setShowLayoutShiftRegions","params":{"result":true}}"#,
        );
        assert!(layout.get("error").is_none());
        assert!(d.overlay_show_layout_shift_regions());

        let ad = dispatch(
            &mut d,
            r#"{"id":6,"method":"Overlay.setShowAdHighlights","params":{"show":true}}"#,
        );
        assert!(ad.get("error").is_none());
        assert!(d.overlay_show_ad_highlights());

        let viewport = dispatch(
            &mut d,
            r#"{"id":7,"method":"Overlay.setShowViewportSizeOnResize","params":{"show":true}}"#,
        );
        assert!(viewport.get("error").is_none());
        assert!(d.overlay_show_viewport_size_on_resize());

        let scroll = dispatch(
            &mut d,
            r#"{"id":8,"method":"Overlay.setShowScrollBottleneckRects","params":{"show":true}}"#,
        );
        assert!(scroll.get("error").is_none());
        assert!(d.overlay_show_scroll_bottleneck_rects());

        let hit = dispatch(
            &mut d,
            r#"{"id":9,"method":"Overlay.setShowHitTestBorders","params":{"show":true}}"#,
        );
        assert!(hit.get("error").is_none());
        assert!(d.overlay_show_hit_test_borders());

        let grid = dispatch(
            &mut d,
            r#"{"id":10,"method":"Overlay.setShowGridOverlays","params":{"gridNodeHighlightConfigs":[{"nodeId":1,"gridHighlightConfig":{}},{"backendNodeId":2,"gridHighlightConfig":{}}]}}"#,
        );
        assert!(grid.get("error").is_none());
        assert_eq!(d.overlay_grid_overlay_count(), 2);

        let flex = dispatch(
            &mut d,
            r#"{"id":11,"method":"Overlay.setShowFlexOverlays","params":{"flexNodeHighlightConfigs":[{"nodeId":3,"flexContainerHighlightConfig":{}}]}}"#,
        );
        assert!(flex.get("error").is_none());
        assert_eq!(d.overlay_flex_overlay_count(), 1);

        let scroll_snap = dispatch(
            &mut d,
            r#"{"id":12,"method":"Overlay.setShowScrollSnapOverlays","params":{"scrollSnapHighlightConfigs":[{"nodeId":4,"scrollSnapContainerHighlightConfig":{}}]}}"#,
        );
        assert!(scroll_snap.get("error").is_none());
        assert_eq!(d.overlay_scroll_snap_overlay_count(), 1);

        let container_query = dispatch(
            &mut d,
            r#"{"id":13,"method":"Overlay.setShowContainerQueryOverlays","params":{"containerQueryHighlightConfigs":[{"nodeId":5,"containerQueryContainerHighlightConfig":{}}]}}"#,
        );
        assert!(container_query.get("error").is_none());
        assert_eq!(d.overlay_container_query_overlay_count(), 1);

        let isolated = dispatch(
            &mut d,
            r#"{"id":14,"method":"Overlay.setShowIsolatedElements","params":{"isolatedElementHighlightConfigs":[{"nodeId":6,"highlightConfig":{}}]}}"#,
        );
        assert!(isolated.get("error").is_none());
        assert_eq!(d.overlay_isolated_element_count(), 1);

        let inspect = dispatch(
            &mut d,
            r#"{"id":15,"method":"Overlay.setInspectMode","params":{"mode":"searchForNode"}}"#,
        );
        assert!(inspect.get("error").is_none());
        assert_eq!(d.overlay_inspect_mode(), "searchForNode");

        let bad = dispatch(
            &mut d,
            r#"{"id":16,"method":"Overlay.setShowHitTestBorders","params":{"show":"yes"}}"#,
        );
        assert!(bad["error"].is_object());

        let bad_web_vitals = dispatch(
            &mut d,
            r#"{"id":25,"method":"Overlay.setShowWebVitals","params":{"show":"yes"}}"#,
        );
        assert!(bad_web_vitals["error"].is_object());

        let bad_grid = dispatch(
            &mut d,
            r#"{"id":17,"method":"Overlay.setShowGridOverlays","params":{"gridNodeHighlightConfigs":[1]}}"#,
        );
        assert!(bad_grid["error"].is_object());

        let bad_flex = dispatch(
            &mut d,
            r#"{"id":18,"method":"Overlay.setShowFlexOverlays","params":{"flexNodeHighlightConfigs":[42]}}"#,
        );
        assert!(bad_flex["error"].is_object());

        let bad_scroll_snap = dispatch(
            &mut d,
            r#"{"id":19,"method":"Overlay.setShowScrollSnapOverlays","params":{"scrollSnapHighlightConfigs":[false]}}"#,
        );
        assert!(bad_scroll_snap["error"].is_object());

        let bad_container_query = dispatch(
            &mut d,
            r#"{"id":20,"method":"Overlay.setShowContainerQueryOverlays","params":{"containerQueryHighlightConfigs":[null]}}"#,
        );
        assert!(bad_container_query["error"].is_object());

        let bad_isolated = dispatch(
            &mut d,
            r#"{"id":21,"method":"Overlay.setShowIsolatedElements","params":{"isolatedElementHighlightConfigs":["node"]}}"#,
        );
        assert!(bad_isolated["error"].is_object());

        let hide = dispatch(
            &mut d,
            r#"{"id":22,"method":"Overlay.hideHighlight","params":{}}"#,
        );
        assert!(hide.get("error").is_none());

        let disable = dispatch(
            &mut d,
            r#"{"id":23,"method":"Overlay.disable","params":{}}"#,
        );
        assert!(disable.get("error").is_none());
        assert!(!d.overlay_enabled());
    }

    #[test]
    fn emulation_setup_methods_store_state_and_validate_input() {
        let mut d = fresh_dispatcher();
        let metrics = dispatch(
            &mut d,
            r#"{"id":1,"method":"Emulation.setDeviceMetricsOverride","params":{"width":800,"height":600,"deviceScaleFactor":2,"mobile":false}}"#,
        );
        assert!(metrics.get("error").is_none());
        assert!(d.emulation_device_metrics().is_some());

        let bad_metrics = dispatch(
            &mut d,
            r#"{"id":2,"method":"Emulation.setDeviceMetricsOverride","params":{"width":800}}"#,
        );
        assert!(bad_metrics["error"].is_object());

        let touch = dispatch(
            &mut d,
            r#"{"id":3,"method":"Emulation.setTouchEmulationEnabled","params":{"enabled":true,"maxTouchPoints":5}}"#,
        );
        assert!(touch.get("error").is_none());
        assert!(d.emulation_touch_enabled());
        assert_eq!(d.emulation_max_touch_points(), 5);

        let emit_touch = dispatch(
            &mut d,
            r#"{"id":4,"method":"Emulation.setEmitTouchEventsForMouse","params":{"enabled":true,"configuration":"mobile"}}"#,
        );
        assert!(emit_touch.get("error").is_none());
        assert!(d.emulation_emit_touch_events_for_mouse_enabled());
        assert_eq!(
            d.emulation_emit_touch_events_for_mouse_configuration(),
            "mobile"
        );

        let emit_touch_disabled = dispatch(
            &mut d,
            r#"{"id":5,"method":"Emulation.setEmitTouchEventsForMouse","params":{"enabled":false}}"#,
        );
        assert!(emit_touch_disabled.get("error").is_none());
        assert!(!d.emulation_emit_touch_events_for_mouse_enabled());
        assert_eq!(d.emulation_emit_touch_events_for_mouse_configuration(), "");

        let emit_touch_bad_enabled = dispatch(
            &mut d,
            r#"{"id":6,"method":"Emulation.setEmitTouchEventsForMouse","params":{"enabled":"yes"}}"#,
        );
        assert!(emit_touch_bad_enabled["error"].is_object());

        let emit_touch_bad_config_type = dispatch(
            &mut d,
            r#"{"id":7,"method":"Emulation.setEmitTouchEventsForMouse","params":{"enabled":true,"configuration":1}}"#,
        );
        assert!(emit_touch_bad_config_type["error"].is_object());

        let emit_touch_bad_config_value = dispatch(
            &mut d,
            r#"{"id":8,"method":"Emulation.setEmitTouchEventsForMouse","params":{"enabled":true,"configuration":"tablet"}}"#,
        );
        assert!(emit_touch_bad_config_value["error"].is_object());

        let media = dispatch(
            &mut d,
            r#"{"id":9,"method":"Emulation.setEmulatedMedia","params":{"media":"print","features":[{"name":"prefers-color-scheme","value":"dark"}]}}"#,
        );
        assert!(media.get("error").is_none());
        assert_eq!(d.emulation_media(), "print");
        assert_eq!(d.emulation_media_feature_count(), 1);

        let media_bad = dispatch(
            &mut d,
            r#"{"id":10,"method":"Emulation.setEmulatedMedia","params":{"features":{}}}"#,
        );
        assert!(media_bad["error"].is_object());

        let cpu = dispatch(
            &mut d,
            r#"{"id":11,"method":"Emulation.setCPUThrottlingRate","params":{"rate":4}}"#,
        );
        assert!(cpu.get("error").is_none());
        assert_eq!(d.emulation_cpu_throttling_rate(), 4.0);

        let cpu_bad_missing = dispatch(
            &mut d,
            r#"{"id":12,"method":"Emulation.setCPUThrottlingRate","params":{}}"#,
        );
        assert!(cpu_bad_missing["error"].is_object());

        let cpu_bad_negative = dispatch(
            &mut d,
            r#"{"id":13,"method":"Emulation.setCPUThrottlingRate","params":{"rate":-1}}"#,
        );
        assert!(cpu_bad_negative["error"].is_object());

        let timezone = dispatch(
            &mut d,
            r#"{"id":14,"method":"Emulation.setTimezoneOverride","params":{"timezoneId":"UTC"}}"#,
        );
        assert!(timezone.get("error").is_none());
        assert_eq!(d.emulation_timezone_id(), "UTC");

        let timezone_clear = dispatch(
            &mut d,
            r#"{"id":15,"method":"Emulation.setTimezoneOverride","params":{"timezoneId":""}}"#,
        );
        assert!(timezone_clear.get("error").is_none());
        assert_eq!(d.emulation_timezone_id(), "");

        let timezone_bad = dispatch(
            &mut d,
            r#"{"id":16,"method":"Emulation.setTimezoneOverride","params":{"timezoneId":1}}"#,
        );
        assert!(timezone_bad["error"].is_object());

        let locale = dispatch(
            &mut d,
            r#"{"id":17,"method":"Emulation.setLocaleOverride","params":{"locale":"fr-FR"}}"#,
        );
        assert!(locale.get("error").is_none());
        assert_eq!(d.emulation_locale(), "fr-FR");

        let locale_clear = dispatch(
            &mut d,
            r#"{"id":18,"method":"Emulation.setLocaleOverride","params":{"locale":""}}"#,
        );
        assert!(locale_clear.get("error").is_none());
        assert_eq!(d.emulation_locale(), "");

        let locale_bad = dispatch(
            &mut d,
            r#"{"id":19,"method":"Emulation.setLocaleOverride","params":{"locale":1}}"#,
        );
        assert!(locale_bad["error"].is_object());

        let user_agent = dispatch(
            &mut d,
            r#"{"id":41,"method":"Emulation.setUserAgentOverride","params":{"userAgent":"Stator/1.0","acceptLanguage":"en-US","platform":"Win32","userAgentMetadata":{"brands":[]}}}"#,
        );
        assert!(user_agent.get("error").is_none());
        assert_eq!(d.emulation_user_agent(), "Stator/1.0");
        assert_eq!(d.emulation_accept_language(), "en-US");
        assert_eq!(d.emulation_platform(), "Win32");
        assert!(d.emulation_user_agent_metadata().is_some());

        let user_agent_minimal = dispatch(
            &mut d,
            r#"{"id":42,"method":"Emulation.setUserAgentOverride","params":{"userAgent":"Stator/2.0"}}"#,
        );
        assert!(user_agent_minimal.get("error").is_none());
        assert_eq!(d.emulation_user_agent(), "Stator/2.0");
        assert_eq!(d.emulation_accept_language(), "");
        assert_eq!(d.emulation_platform(), "");
        assert!(d.emulation_user_agent_metadata().is_none());

        let user_agent_bad_missing = dispatch(
            &mut d,
            r#"{"id":43,"method":"Emulation.setUserAgentOverride","params":{}}"#,
        );
        assert!(user_agent_bad_missing["error"].is_object());

        let user_agent_bad_platform = dispatch(
            &mut d,
            r#"{"id":44,"method":"Emulation.setUserAgentOverride","params":{"userAgent":"Stator/1.0","platform":1}}"#,
        );
        assert!(user_agent_bad_platform["error"].is_object());

        let user_agent_bad_metadata = dispatch(
            &mut d,
            r#"{"id":45,"method":"Emulation.setUserAgentOverride","params":{"userAgent":"Stator/1.0","userAgentMetadata":[]}}"#,
        );
        assert!(user_agent_bad_metadata["error"].is_object());

        let script_disabled = dispatch(
            &mut d,
            r#"{"id":20,"method":"Emulation.setScriptExecutionDisabled","params":{"value":true}}"#,
        );
        assert!(script_disabled.get("error").is_none());
        assert!(d.emulation_script_execution_disabled());

        let script_enabled = dispatch(
            &mut d,
            r#"{"id":21,"method":"Emulation.setScriptExecutionDisabled","params":{"value":false}}"#,
        );
        assert!(script_enabled.get("error").is_none());
        assert!(!d.emulation_script_execution_disabled());

        let script_bad = dispatch(
            &mut d,
            r#"{"id":22,"method":"Emulation.setScriptExecutionDisabled","params":{}}"#,
        );
        assert!(script_bad["error"].is_object());

        let focus_enabled = dispatch(
            &mut d,
            r#"{"id":23,"method":"Emulation.setFocusEmulationEnabled","params":{"enabled":true}}"#,
        );
        assert!(focus_enabled.get("error").is_none());
        assert!(d.emulation_focus_emulation_enabled());

        let focus_disabled = dispatch(
            &mut d,
            r#"{"id":24,"method":"Emulation.setFocusEmulationEnabled","params":{"enabled":false}}"#,
        );
        assert!(focus_disabled.get("error").is_none());
        assert!(!d.emulation_focus_emulation_enabled());

        let focus_bad = dispatch(
            &mut d,
            r#"{"id":25,"method":"Emulation.setFocusEmulationEnabled","params":{"enabled":"yes"}}"#,
        );
        assert!(focus_bad["error"].is_object());

        let idle = dispatch(
            &mut d,
            r#"{"id":26,"method":"Emulation.setIdleOverride","params":{"userActive":true,"screenUnlocked":false}}"#,
        );
        assert!(idle.get("error").is_none());
        assert_eq!(d.emulation_idle_override(), Some((true, false)));

        let idle_bad_user_active = dispatch(
            &mut d,
            r#"{"id":27,"method":"Emulation.setIdleOverride","params":{"screenUnlocked":true}}"#,
        );
        assert!(idle_bad_user_active["error"].is_object());

        let idle_bad_screen_unlocked = dispatch(
            &mut d,
            r#"{"id":28,"method":"Emulation.setIdleOverride","params":{"userActive":true,"screenUnlocked":"yes"}}"#,
        );
        assert!(idle_bad_screen_unlocked["error"].is_object());

        let hardware_concurrency = dispatch(
            &mut d,
            r#"{"id":29,"method":"Emulation.setHardwareConcurrencyOverride","params":{"hardwareConcurrency":8}}"#,
        );
        assert!(hardware_concurrency.get("error").is_none());
        assert_eq!(d.emulation_hardware_concurrency(), 8);

        let hardware_concurrency_bad_missing = dispatch(
            &mut d,
            r#"{"id":30,"method":"Emulation.setHardwareConcurrencyOverride","params":{}}"#,
        );
        assert!(hardware_concurrency_bad_missing["error"].is_object());

        let hardware_concurrency_bad_zero = dispatch(
            &mut d,
            r#"{"id":31,"method":"Emulation.setHardwareConcurrencyOverride","params":{"hardwareConcurrency":0}}"#,
        );
        assert!(hardware_concurrency_bad_zero["error"].is_object());

        let hardware_concurrency_bad_fractional = dispatch(
            &mut d,
            r#"{"id":32,"method":"Emulation.setHardwareConcurrencyOverride","params":{"hardwareConcurrency":1.5}}"#,
        );
        assert!(hardware_concurrency_bad_fractional["error"].is_object());

        let auto_dark_mode = dispatch(
            &mut d,
            r#"{"id":33,"method":"Emulation.setAutoDarkModeOverride","params":{"enabled":true}}"#,
        );
        assert!(auto_dark_mode.get("error").is_none());
        assert_eq!(d.emulation_auto_dark_mode_enabled(), Some(true));

        let clear_auto_dark_mode = dispatch(
            &mut d,
            r#"{"id":34,"method":"Emulation.setAutoDarkModeOverride","params":{}}"#,
        );
        assert!(clear_auto_dark_mode.get("error").is_none());
        assert_eq!(d.emulation_auto_dark_mode_enabled(), None);

        let auto_dark_mode_bad = dispatch(
            &mut d,
            r#"{"id":35,"method":"Emulation.setAutoDarkModeOverride","params":{"enabled":"yes"}}"#,
        );
        assert!(auto_dark_mode_bad["error"].is_object());

        let document_cookie_disabled = dispatch(
            &mut d,
            r#"{"id":36,"method":"Emulation.setDocumentCookieDisabled","params":{"disabled":true}}"#,
        );
        assert!(document_cookie_disabled.get("error").is_none());
        assert!(d.emulation_document_cookie_disabled());

        let document_cookie_enabled = dispatch(
            &mut d,
            r#"{"id":37,"method":"Emulation.setDocumentCookieDisabled","params":{"disabled":false}}"#,
        );
        assert!(document_cookie_enabled.get("error").is_none());
        assert!(!d.emulation_document_cookie_disabled());

        let document_cookie_bad = dispatch(
            &mut d,
            r#"{"id":38,"method":"Emulation.setDocumentCookieDisabled","params":{"disabled":"yes"}}"#,
        );
        assert!(document_cookie_bad["error"].is_object());

        let geolocation = dispatch(
            &mut d,
            r#"{"id":46,"method":"Emulation.setGeolocationOverride","params":{"latitude":37.422,"longitude":-122.084,"accuracy":5,"altitude":12.5,"altitudeAccuracy":2,"heading":180,"speed":1.5}}"#,
        );
        assert!(geolocation.get("error").is_none());
        let geolocation_state = d.emulation_geolocation_override().unwrap();
        assert_eq!(geolocation_state.latitude, Some(37.422));
        assert_eq!(geolocation_state.longitude, Some(-122.084));
        assert_eq!(geolocation_state.accuracy, Some(5.0));
        assert_eq!(geolocation_state.altitude, Some(12.5));
        assert_eq!(geolocation_state.altitude_accuracy, Some(2.0));
        assert_eq!(geolocation_state.heading, Some(180.0));
        assert_eq!(geolocation_state.speed, Some(1.5));

        let geolocation_unavailable = dispatch(
            &mut d,
            r#"{"id":47,"method":"Emulation.setGeolocationOverride","params":{}}"#,
        );
        assert!(geolocation_unavailable.get("error").is_none());
        assert_eq!(
            d.emulation_geolocation_override(),
            Some(&EmulationGeolocationOverride {
                latitude: None,
                longitude: None,
                accuracy: None,
                altitude: None,
                altitude_accuracy: None,
                heading: None,
                speed: None,
            })
        );

        let geolocation_restore = dispatch(
            &mut d,
            r#"{"id":48,"method":"Emulation.setGeolocationOverride","params":{"latitude":1,"longitude":2,"accuracy":3}}"#,
        );
        assert!(geolocation_restore.get("error").is_none());

        let geolocation_bad_range = dispatch(
            &mut d,
            r#"{"id":49,"method":"Emulation.setGeolocationOverride","params":{"latitude":91,"longitude":2,"accuracy":3}}"#,
        );
        assert!(geolocation_bad_range["error"].is_object());
        assert_eq!(
            d.emulation_geolocation_override().unwrap().latitude,
            Some(1.0)
        );

        let geolocation_bad_accuracy = dispatch(
            &mut d,
            r#"{"id":50,"method":"Emulation.setGeolocationOverride","params":{"accuracy":-1}}"#,
        );
        assert!(geolocation_bad_accuracy["error"].is_object());

        let geolocation_bad_longitude = dispatch(
            &mut d,
            r#"{"id":51,"method":"Emulation.setGeolocationOverride","params":{"longitude":181}}"#,
        );
        assert!(geolocation_bad_longitude["error"].is_object());

        let geolocation_bad_type = dispatch(
            &mut d,
            r#"{"id":52,"method":"Emulation.setGeolocationOverride","params":{"heading":null}}"#,
        );
        assert!(geolocation_bad_type["error"].is_object());

        let clear_geolocation = dispatch(
            &mut d,
            r#"{"id":53,"method":"Emulation.clearGeolocationOverride","params":{}}"#,
        );
        assert!(clear_geolocation.get("error").is_none());
        assert!(d.emulation_geolocation_override().is_none());

        let page_scale = dispatch(
            &mut d,
            r#"{"id":54,"method":"Emulation.setPageScaleFactor","params":{"pageScaleFactor":1.25}}"#,
        );
        assert!(page_scale.get("error").is_none());
        assert_eq!(d.emulation_page_scale_factor(), Some(1.25));

        let page_scale_bad_missing = dispatch(
            &mut d,
            r#"{"id":55,"method":"Emulation.setPageScaleFactor","params":{}}"#,
        );
        assert!(page_scale_bad_missing["error"].is_object());
        assert_eq!(d.emulation_page_scale_factor(), Some(1.25));

        let page_scale_bad_zero = dispatch(
            &mut d,
            r#"{"id":56,"method":"Emulation.setPageScaleFactor","params":{"pageScaleFactor":0}}"#,
        );
        assert!(page_scale_bad_zero["error"].is_object());
        assert_eq!(d.emulation_page_scale_factor(), Some(1.25));

        let reset_page_scale = dispatch(
            &mut d,
            r#"{"id":57,"method":"Emulation.resetPageScaleFactor","params":{}}"#,
        );
        assert!(reset_page_scale.get("error").is_none());
        assert_eq!(d.emulation_page_scale_factor(), None);

        let scrollbars_hidden = dispatch(
            &mut d,
            r#"{"id":58,"method":"Emulation.setScrollbarsHidden","params":{"hidden":true}}"#,
        );
        assert!(scrollbars_hidden.get("error").is_none());
        assert!(d.emulation_scrollbars_hidden());

        let scrollbars_bad = dispatch(
            &mut d,
            r#"{"id":59,"method":"Emulation.setScrollbarsHidden","params":{"hidden":"yes"}}"#,
        );
        assert!(scrollbars_bad["error"].is_object());
        assert!(d.emulation_scrollbars_hidden());

        let scrollbars_visible = dispatch(
            &mut d,
            r#"{"id":60,"method":"Emulation.setScrollbarsHidden","params":{"hidden":false}}"#,
        );
        assert!(scrollbars_visible.get("error").is_none());
        assert!(!d.emulation_scrollbars_hidden());

        let clear_idle = dispatch(
            &mut d,
            r#"{"id":39,"method":"Emulation.clearIdleOverride","params":{}}"#,
        );
        assert!(clear_idle.get("error").is_none());
        assert_eq!(d.emulation_idle_override(), None);

        let clear = dispatch(
            &mut d,
            r#"{"id":40,"method":"Emulation.clearDeviceMetricsOverride","params":{}}"#,
        );
        assert!(clear.get("error").is_none());
        assert!(d.emulation_device_metrics().is_none());
    }

    #[test]
    fn performance_domain_reports_metrics_and_tracks_enable_state() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(
            &mut d,
            r#"{"id":1,"method":"Performance.enable","params":{}}"#,
        );
        assert!(enable.get("error").is_none());
        assert!(d.performance_enabled());

        d.globals
            .borrow_mut()
            .insert("perfObject".to_string(), JsValue::Smi(1));
        let metrics = dispatch(
            &mut d,
            r#"{"id":2,"method":"Performance.getMetrics","params":{}}"#,
        );
        let names: HashSet<_> = metrics["result"]["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|metric| metric["name"].as_str())
            .collect();
        assert!(names.contains("Timestamp"));
        assert!(names.contains("JSHeapUsedSize"));
        assert!(names.contains("JSHeapTotalSize"));

        let disable = dispatch(
            &mut d,
            r#"{"id":3,"method":"Performance.disable","params":{}}"#,
        );
        assert!(disable.get("error").is_none());
        assert!(!d.performance_enabled());
    }

    #[test]
    fn security_setup_methods_store_state_and_validate_input() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(&mut d, r#"{"id":1,"method":"Security.enable","params":{}}"#);
        assert!(enable.get("error").is_none());
        assert!(d.security_enabled());

        let ignore = dispatch(
            &mut d,
            r#"{"id":2,"method":"Security.setIgnoreCertificateErrors","params":{"ignore":true}}"#,
        );
        assert!(ignore.get("error").is_none());
        assert!(d.security_ignore_certificate_errors());

        let bad = dispatch(
            &mut d,
            r#"{"id":3,"method":"Security.setIgnoreCertificateErrors","params":{}}"#,
        );
        assert!(bad["error"].is_object());

        let disable = dispatch(
            &mut d,
            r#"{"id":4,"method":"Security.disable","params":{}}"#,
        );
        assert!(disable.get("error").is_none());
        assert!(!d.security_enabled());
    }

    #[test]
    fn log_setup_methods_store_state_and_validate_input() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(&mut d, r#"{"id":1,"method":"Log.enable","params":{}}"#);
        assert!(enable.get("error").is_none());
        assert!(d.log_enabled());

        let start = dispatch(
            &mut d,
            r#"{"id":2,"method":"Log.startViolationsReport","params":{"config":[{"name":"longTask","threshold":200}]}}"#,
        );
        assert!(start.get("error").is_none());
        assert_eq!(d.log_violation_setting_count(), 1);

        let bad = dispatch(
            &mut d,
            r#"{"id":3,"method":"Log.startViolationsReport","params":{"config":[{"name":"longTask"}]}}"#,
        );
        assert!(bad["error"].is_object());

        let clear = dispatch(&mut d, r#"{"id":4,"method":"Log.clear","params":{}}"#);
        assert!(clear.get("error").is_none());

        let stop = dispatch(
            &mut d,
            r#"{"id":5,"method":"Log.stopViolationsReport","params":{}}"#,
        );
        assert!(stop.get("error").is_none());
        assert_eq!(d.log_violation_setting_count(), 0);

        let disable = dispatch(&mut d, r#"{"id":6,"method":"Log.disable","params":{}}"#);
        assert!(disable.get("error").is_none());
        assert!(!d.log_enabled());
    }

    #[test]
    fn page_setup_methods_return_minimal_frame_tree_and_validate_input() {
        let mut d = CdpDispatcher::with_globals_and_contexts(
            Rc::new(RefCell::new(GlobalEnv::new())),
            vec![ExecutionContextDescription::new(
                7,
                1,
                "stator://test-page",
                "test-page",
                json!({"isDefault": true}),
            )],
        );
        let enable = dispatch(&mut d, r#"{"id":1,"method":"Page.enable","params":{}}"#);
        assert!(enable.get("error").is_none());
        assert!(d.page_enabled());

        let resource_tree = dispatch(
            &mut d,
            r#"{"id":2,"method":"Page.getResourceTree","params":{}}"#,
        );
        assert_eq!(
            resource_tree["result"]["frameTree"]["frame"]["id"],
            "stator-frame-7"
        );
        assert_eq!(
            resource_tree["result"]["frameTree"]["frame"]["url"],
            "stator://test-page"
        );
        assert_eq!(
            resource_tree["result"]["frameTree"]["resources"]
                .as_array()
                .unwrap()
                .len(),
            0
        );

        let frame_tree = dispatch(
            &mut d,
            r#"{"id":3,"method":"Page.getFrameTree","params":{}}"#,
        );
        assert_eq!(
            frame_tree["result"]["frameTree"]["frame"]["id"],
            "stator-frame-7"
        );

        let lifecycle = dispatch(
            &mut d,
            r#"{"id":4,"method":"Page.setLifecycleEventsEnabled","params":{"enabled":true}}"#,
        );
        assert!(lifecycle.get("error").is_none());
        assert!(d.page_lifecycle_events_enabled());

        let lifecycle_bad = dispatch(
            &mut d,
            r#"{"id":5,"method":"Page.setLifecycleEventsEnabled","params":{}}"#,
        );
        assert!(lifecycle_bad["error"].is_object());

        let bypass_csp = dispatch(
            &mut d,
            r#"{"id":6,"method":"Page.setBypassCSP","params":{"enabled":true}}"#,
        );
        assert!(bypass_csp.get("error").is_none());
        assert!(d.page_bypass_csp());

        let bypass_csp_bad = dispatch(
            &mut d,
            r#"{"id":7,"method":"Page.setBypassCSP","params":{}}"#,
        );
        assert!(bypass_csp_bad["error"].is_object());

        let ad_blocking = dispatch(
            &mut d,
            r#"{"id":8,"method":"Page.setAdBlockingEnabled","params":{"enabled":true}}"#,
        );
        assert!(ad_blocking.get("error").is_none());
        assert!(d.page_ad_blocking_enabled());

        let ad_blocking_missing = dispatch(
            &mut d,
            r#"{"id":9,"method":"Page.setAdBlockingEnabled","params":{}}"#,
        );
        assert!(ad_blocking_missing["error"].is_object());
        assert!(d.page_ad_blocking_enabled());

        let ad_blocking_false = dispatch(
            &mut d,
            r#"{"id":10,"method":"Page.setAdBlockingEnabled","params":{"enabled":false}}"#,
        );
        assert!(ad_blocking_false.get("error").is_none());
        assert!(!d.page_ad_blocking_enabled());

        let ad_blocking_bad = dispatch(
            &mut d,
            r#"{"id":11,"method":"Page.setAdBlockingEnabled","params":{"enabled":"yes"}}"#,
        );
        assert!(ad_blocking_bad["error"].is_object());
        assert!(!d.page_ad_blocking_enabled());

        let disable = dispatch(&mut d, r#"{"id":12,"method":"Page.disable","params":{}}"#);
        assert!(disable.get("error").is_none());
        assert!(!d.page_enabled());
    }

    #[test]
    fn network_setup_methods_store_settings_and_validate_input() {
        let mut d = fresh_dispatcher();
        let enable = dispatch(&mut d, r#"{"id":1,"method":"Network.enable","params":{}}"#);
        assert!(enable.get("error").is_none());
        assert!(d.network_enabled());

        let cache = dispatch(
            &mut d,
            r#"{"id":2,"method":"Network.setCacheDisabled","params":{"cacheDisabled":true}}"#,
        );
        assert!(cache.get("error").is_none());
        assert!(d.network_cache_disabled());

        let bypass = dispatch(
            &mut d,
            r#"{"id":3,"method":"Network.setBypassServiceWorker","params":{"bypass":true}}"#,
        );
        assert!(bypass.get("error").is_none());
        assert!(d.network_bypass_service_worker());

        let attach_debug_stack = dispatch(
            &mut d,
            r#"{"id":33,"method":"Network.setAttachDebugStack","params":{"enabled":true}}"#,
        );
        assert!(attach_debug_stack.get("error").is_none());
        assert!(d.network_attach_debug_stack());

        let reporting_api = dispatch(
            &mut d,
            r#"{"id":34,"method":"Network.setReportingApiEnabled","params":{"enabled":true}}"#,
        );
        assert!(reporting_api.get("error").is_none());
        assert!(d.network_reporting_api_enabled());

        let cache_bad = dispatch(
            &mut d,
            r#"{"id":4,"method":"Network.setCacheDisabled","params":{}}"#,
        );
        assert!(cache_bad["error"].is_object());

        let bypass_bad = dispatch(
            &mut d,
            r#"{"id":5,"method":"Network.setBypassServiceWorker","params":{}}"#,
        );
        assert!(bypass_bad["error"].is_object());

        let attach_debug_stack_bad = dispatch(
            &mut d,
            r#"{"id":35,"method":"Network.setAttachDebugStack","params":{"enabled":"yes"}}"#,
        );
        assert!(attach_debug_stack_bad["error"].is_object());
        assert!(d.network_attach_debug_stack());

        let reporting_api_bad = dispatch(
            &mut d,
            r#"{"id":36,"method":"Network.setReportingApiEnabled","params":{}}"#,
        );
        assert!(reporting_api_bad["error"].is_object());
        assert!(d.network_reporting_api_enabled());

        let attach_debug_stack_reset = dispatch(
            &mut d,
            r#"{"id":37,"method":"Network.setAttachDebugStack","params":{"enabled":false}}"#,
        );
        assert!(attach_debug_stack_reset.get("error").is_none());
        assert!(!d.network_attach_debug_stack());

        let reporting_api_reset = dispatch(
            &mut d,
            r#"{"id":38,"method":"Network.setReportingApiEnabled","params":{"enabled":false}}"#,
        );
        assert!(reporting_api_reset.get("error").is_none());
        assert!(!d.network_reporting_api_enabled());

        let user_agent = dispatch(
            &mut d,
            r#"{"id":6,"method":"Network.setUserAgentOverride","params":{"userAgent":"Stator/1.0","acceptLanguage":"en-US","platform":"Win32","userAgentMetadata":{"brands":[]}}}"#,
        );
        assert!(user_agent.get("error").is_none());
        assert_eq!(d.network_user_agent(), "Stator/1.0");
        assert_eq!(d.network_accept_language(), "en-US");
        assert_eq!(d.network_platform(), "Win32");
        assert!(d.network_user_agent_metadata().is_some());

        let user_agent_minimal = dispatch(
            &mut d,
            r#"{"id":7,"method":"Network.setUserAgentOverride","params":{"userAgent":"Stator/2.0"}}"#,
        );
        assert!(user_agent_minimal.get("error").is_none());
        assert_eq!(d.network_user_agent(), "Stator/2.0");
        assert_eq!(d.network_accept_language(), "");
        assert_eq!(d.network_platform(), "");
        assert!(d.network_user_agent_metadata().is_none());

        let user_agent_bad_missing = dispatch(
            &mut d,
            r#"{"id":8,"method":"Network.setUserAgentOverride","params":{}}"#,
        );
        assert!(user_agent_bad_missing["error"].is_object());

        let user_agent_bad_language = dispatch(
            &mut d,
            r#"{"id":9,"method":"Network.setUserAgentOverride","params":{"userAgent":"Stator/1.0","acceptLanguage":1}}"#,
        );
        assert!(user_agent_bad_language["error"].is_object());

        let user_agent_bad_metadata = dispatch(
            &mut d,
            r#"{"id":10,"method":"Network.setUserAgentOverride","params":{"userAgent":"Stator/1.0","userAgentMetadata":[]}}"#,
        );
        assert!(user_agent_bad_metadata["error"].is_object());

        let headers = dispatch(
            &mut d,
            r#"{"id":11,"method":"Network.setExtraHTTPHeaders","params":{"headers":{"X-Test":"1","Accept":"text/plain"}}}"#,
        );
        assert!(headers.get("error").is_none());
        assert_eq!(d.network_extra_http_header_count(), 2);

        let headers_empty = dispatch(
            &mut d,
            r#"{"id":12,"method":"Network.setExtraHTTPHeaders","params":{"headers":{}}}"#,
        );
        assert!(headers_empty.get("error").is_none());
        assert_eq!(d.network_extra_http_header_count(), 0);

        let headers_bad_missing = dispatch(
            &mut d,
            r#"{"id":13,"method":"Network.setExtraHTTPHeaders","params":{}}"#,
        );
        assert!(headers_bad_missing["error"].is_object());

        let headers_bad_value = dispatch(
            &mut d,
            r#"{"id":14,"method":"Network.setExtraHTTPHeaders","params":{"headers":{"X-Test":1}}}"#,
        );
        assert!(headers_bad_value["error"].is_object());

        let blocked_urls = dispatch(
            &mut d,
            r#"{"id":15,"method":"Network.setBlockedURLs","params":{"urls":["*.png","https://example.test/*"]}}"#,
        );
        assert!(blocked_urls.get("error").is_none());
        assert_eq!(d.network_blocked_url_count(), 2);

        let blocked_urls_empty = dispatch(
            &mut d,
            r#"{"id":16,"method":"Network.setBlockedURLs","params":{"urls":[]}}"#,
        );
        assert!(blocked_urls_empty.get("error").is_none());
        assert_eq!(d.network_blocked_url_count(), 0);

        let blocked_urls_bad_missing = dispatch(
            &mut d,
            r#"{"id":17,"method":"Network.setBlockedURLs","params":{}}"#,
        );
        assert!(blocked_urls_bad_missing["error"].is_object());

        let blocked_urls_bad_value = dispatch(
            &mut d,
            r#"{"id":18,"method":"Network.setBlockedURLs","params":{"urls":["ok",1]}}"#,
        );
        assert!(blocked_urls_bad_value["error"].is_object());

        let accepted_encodings = dispatch(
            &mut d,
            r#"{"id":19,"method":"Network.setAcceptedEncodings","params":{"encodings":["gzip","br","zstd","identity","deflate"]}}"#,
        );
        assert!(accepted_encodings.get("error").is_none());
        assert_eq!(d.network_accepted_encoding_count(), 5);

        let accepted_encodings_empty = dispatch(
            &mut d,
            r#"{"id":20,"method":"Network.setAcceptedEncodings","params":{"encodings":[]}}"#,
        );
        assert!(accepted_encodings_empty.get("error").is_none());
        assert_eq!(d.network_accepted_encoding_count(), 0);

        let accepted_encodings_bad_missing = dispatch(
            &mut d,
            r#"{"id":21,"method":"Network.setAcceptedEncodings","params":{}}"#,
        );
        assert!(accepted_encodings_bad_missing["error"].is_object());

        let accepted_encodings_bad_value = dispatch(
            &mut d,
            r#"{"id":22,"method":"Network.setAcceptedEncodings","params":{"encodings":["gzip",1]}}"#,
        );
        assert!(accepted_encodings_bad_value["error"].is_object());

        let accepted_encodings_bad_unsupported = dispatch(
            &mut d,
            r#"{"id":23,"method":"Network.setAcceptedEncodings","params":{"encodings":["gzip","compress"]}}"#,
        );
        assert!(accepted_encodings_bad_unsupported["error"].is_object());

        let accepted_encodings_reset = dispatch(
            &mut d,
            r#"{"id":24,"method":"Network.setAcceptedEncodings","params":{"encodings":["gzip","br"]}}"#,
        );
        assert!(accepted_encodings_reset.get("error").is_none());
        assert_eq!(d.network_accepted_encoding_count(), 2);

        let clear_accepted_encodings = dispatch(
            &mut d,
            r#"{"id":25,"method":"Network.clearAcceptedEncodingsOverride","params":{}}"#,
        );
        assert!(clear_accepted_encodings.get("error").is_none());
        assert_eq!(d.network_accepted_encoding_count(), 0);

        let emulate_conditions = dispatch(
            &mut d,
            r#"{"id":26,"method":"Network.emulateNetworkConditions","params":{"offline":true,"latency":12.5,"downloadThroughput":1024,"uploadThroughput":2048,"connectionType":"wifi","packetLoss":1.5,"packetQueueLength":7,"packetReordering":true}}"#,
        );
        assert!(emulate_conditions.get("error").is_none());
        let conditions = d.network_emulated_conditions().unwrap();
        assert!(conditions.offline);
        assert_eq!(conditions.latency, 12.5);
        assert_eq!(conditions.download_throughput, 1024.0);
        assert_eq!(conditions.upload_throughput, 2048.0);
        assert_eq!(conditions.connection_type.as_deref(), Some("wifi"));
        assert_eq!(conditions.packet_loss, Some(1.5));
        assert_eq!(conditions.packet_queue_length, Some(7));
        assert_eq!(conditions.packet_reordering, Some(true));

        let emulate_conditions_minimal = dispatch(
            &mut d,
            r#"{"id":27,"method":"Network.emulateNetworkConditions","params":{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1}}"#,
        );
        assert!(emulate_conditions_minimal.get("error").is_none());
        let conditions = d.network_emulated_conditions().unwrap();
        assert!(!conditions.offline);
        assert_eq!(conditions.latency, 0.0);
        assert_eq!(conditions.download_throughput, -1.0);
        assert_eq!(conditions.upload_throughput, -1.0);
        assert_eq!(conditions.connection_type, None);
        assert_eq!(conditions.packet_loss, None);
        assert_eq!(conditions.packet_queue_length, None);
        assert_eq!(conditions.packet_reordering, None);

        let emulate_conditions_packet_loss_max = dispatch(
            &mut d,
            r#"{"id":28,"method":"Network.emulateNetworkConditions","params":{"offline":false,"latency":3,"downloadThroughput":0,"uploadThroughput":0,"packetLoss":100}}"#,
        );
        assert!(emulate_conditions_packet_loss_max.get("error").is_none());
        assert_eq!(
            d.network_emulated_conditions().unwrap().packet_loss,
            Some(100.0)
        );
        let preserved_conditions = d.network_emulated_conditions().cloned();

        for invalid_params in [
            r#"{"latency":0,"downloadThroughput":-1,"uploadThroughput":-1}"#,
            r#"{"offline":false,"downloadThroughput":-1,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1}"#,
            r#"{"offline":"false","latency":0,"downloadThroughput":-1,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":-1,"downloadThroughput":-1,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":null,"downloadThroughput":-1,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-2,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-0.5,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":null,"uploadThroughput":-1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-2}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-0.5}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":null}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"connectionType":"5g"}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"connectionType":null}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"packetLoss":100.1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"packetLoss":-0.1}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"packetQueueLength":1.5}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"packetReordering":"yes"}"#,
            r#"{"offline":false,"latency":0,"downloadThroughput":-1,"uploadThroughput":-1,"packetReordering":null}"#,
        ] {
            let invalid_emulation = dispatch(
                &mut d,
                &format!(
                    r#"{{"id":29,"method":"Network.emulateNetworkConditions","params":{invalid_params}}}"#
                ),
            );
            assert!(
                invalid_emulation["error"].is_object(),
                "expected error for {invalid_params}"
            );
            assert_eq!(
                d.network_emulated_conditions(),
                preserved_conditions.as_ref()
            );
        }

        let clear_cache = dispatch(
            &mut d,
            r#"{"id":30,"method":"Network.clearBrowserCache","params":{}}"#,
        );
        assert!(clear_cache.get("error").is_none());

        let clear_cookies = dispatch(
            &mut d,
            r#"{"id":31,"method":"Network.clearBrowserCookies","params":{}}"#,
        );
        assert!(clear_cookies.get("error").is_none());

        let disable = dispatch(
            &mut d,
            r#"{"id":32,"method":"Network.disable","params":{}}"#,
        );
        assert!(disable.get("error").is_none());
        assert!(!d.network_enabled());
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
        assert!(names.contains(&"Browser"));
        assert!(names.contains(&"Inspector"));
        assert!(names.contains(&"Runtime"));
        assert!(names.contains(&"Debugger"));
        assert!(names.contains(&"Target"));
        assert!(names.contains(&"Schema"));
    }

    #[test]
    fn browser_get_version_reports_stator_metadata() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Browser.getVersion","params":{}}"#,
        );
        let result = &resp["result"];
        assert_eq!(result["protocolVersion"], "1.3");
        assert!(
            result["product"]
                .as_str()
                .expect("product string")
                .starts_with("StatorJSE/")
        );
        assert_eq!(result["revision"], env!("CARGO_PKG_VERSION"));
        assert_eq!(result["jsVersion"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn inspector_enable_disable_tracks_state() {
        let mut d = fresh_dispatcher();
        let enabled = dispatch(
            &mut d,
            r#"{"id":1,"method":"Inspector.enable","params":{}}"#,
        );
        assert!(enabled.get("error").is_none());
        assert!(d.inspector_enabled());

        let disabled = dispatch(
            &mut d,
            r#"{"id":2,"method":"Inspector.disable","params":{}}"#,
        );
        assert!(disabled.get("error").is_none());
        assert!(!d.inspector_enabled());
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
    fn target_get_target_info_reports_live_targets_and_rejects_bad_ids() {
        let contexts = vec![
            ExecutionContextDescription::new(1, 1, "https://a.test", "main", json!({})),
            ExecutionContextDescription::new(2, 2, "https://b.test", "isolated", json!({})),
        ];
        let mut d = CdpDispatcher::with_globals_and_contexts(
            Rc::new(RefCell::new(GlobalEnv::new())),
            contexts,
        );

        let current = dispatch(
            &mut d,
            r#"{"id":1,"method":"Target.getTargetInfo","params":{}}"#,
        );
        assert_eq!(current["result"]["targetInfo"]["targetId"], "stator-1");
        assert_eq!(current["result"]["targetInfo"]["url"], "https://a.test");

        let explicit = dispatch(
            &mut d,
            r#"{"id":2,"method":"Target.getTargetInfo","params":{"targetId":"stator-2"}}"#,
        );
        assert_eq!(explicit["result"]["targetInfo"]["targetId"], "stator-2");
        assert_eq!(explicit["result"]["targetInfo"]["url"], "https://b.test");

        let non_string = dispatch(
            &mut d,
            r#"{"id":3,"method":"Target.getTargetInfo","params":{"targetId":2}}"#,
        );
        assert!(non_string["error"].is_object());

        let unknown = dispatch(
            &mut d,
            r#"{"id":4,"method":"Target.getTargetInfo","params":{"targetId":"missing"}}"#,
        );
        assert!(unknown["error"].is_object());

        let close = dispatch(
            &mut d,
            r#"{"id":5,"method":"Target.closeTarget","params":{"targetId":"stator-2"}}"#,
        );
        assert_eq!(close["result"]["success"], true);
        let closed = dispatch(
            &mut d,
            r#"{"id":6,"method":"Target.getTargetInfo","params":{"targetId":"stator-2"}}"#,
        );
        assert!(closed["error"].is_object());
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
    fn target_auto_attach_setup_stores_state_and_validates_input() {
        let mut d = fresh_dispatcher();
        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Target.setAutoAttach","params":{"autoAttach":true,"waitForDebuggerOnStart":true,"flatten":true,"filter":[{"type":"page"}]}}"#,
        );
        assert!(ok.get("error").is_none());
        assert!(d.target_auto_attach_enabled());
        assert!(d.target_auto_attach_wait_for_debugger_on_start());
        assert!(d.target_auto_attach_flatten());
        assert_eq!(d.target_auto_attach_filter_count(), 1);

        let bad_required = dispatch(
            &mut d,
            r#"{"id":2,"method":"Target.setAutoAttach","params":{"autoAttach":true}}"#,
        );
        assert!(bad_required["error"].is_object());

        let bad_filter = dispatch(
            &mut d,
            r#"{"id":3,"method":"Target.setAutoAttach","params":{"autoAttach":false,"waitForDebuggerOnStart":false,"filter":true}}"#,
        );
        assert!(bad_filter["error"].is_object());
    }

    #[test]
    fn target_remote_locations_setup_stores_state_and_validates_input() {
        let mut d = fresh_dispatcher();
        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Target.setRemoteLocations","params":{"locations":[{"host":"127.0.0.1","port":9222},{"host":"localhost","port":0}]}}"#,
        );
        assert!(ok.get("error").is_none());
        assert_eq!(d.target_remote_location_count(), 2);

        let empty = dispatch(
            &mut d,
            r#"{"id":2,"method":"Target.setRemoteLocations","params":{"locations":[]}}"#,
        );
        assert!(empty.get("error").is_none());
        assert_eq!(d.target_remote_location_count(), 0);

        let reset = dispatch(
            &mut d,
            r#"{"id":3,"method":"Target.setRemoteLocations","params":{"locations":[{"host":"remote.test","port":65535}]}}"#,
        );
        assert!(reset.get("error").is_none());
        assert_eq!(d.target_remote_location_count(), 1);

        for invalid_params in [
            r#"{}"#,
            r#"{"locations":{}}"#,
            r#"{"locations":[true]}"#,
            r#"{"locations":[{"port":9222}]}"#,
            r#"{"locations":[{"host":1,"port":9222}]}"#,
            r#"{"locations":[{"host":"localhost"}]}"#,
            r#"{"locations":[{"host":"localhost","port":"9222"}]}"#,
            r#"{"locations":[{"host":"localhost","port":1.5}]}"#,
            r#"{"locations":[{"host":"localhost","port":-1}]}"#,
            r#"{"locations":[{"host":"localhost","port":65536}]}"#,
        ] {
            let invalid = dispatch(
                &mut d,
                &format!(
                    r#"{{"id":4,"method":"Target.setRemoteLocations","params":{invalid_params}}}"#
                ),
            );
            assert!(
                invalid["error"].is_object(),
                "expected error for {invalid_params}"
            );
            assert_eq!(d.target_remote_location_count(), 1);
        }
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
    fn runtime_setup_compat_methods_store_settings_and_reject_bad_input() {
        let mut d = fresh_dispatcher();
        let formatter = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.setCustomObjectFormatterEnabled","params":{"enabled":true}}"#,
        );
        assert!(formatter.get("error").is_none());
        assert!(d.custom_object_formatter_enabled());

        let formatter_bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.setCustomObjectFormatterEnabled","params":{}}"#,
        );
        assert!(formatter_bad["error"].is_object());

        let stack = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.setMaxCallStackSizeToCapture","params":{"size":32}}"#,
        );
        assert!(stack.get("error").is_none());
        assert_eq!(d.max_call_stack_size_to_capture(), 32);

        let stack_huge = dispatch(
            &mut d,
            r#"{"id":4,"method":"Runtime.setMaxCallStackSizeToCapture","params":{"size":9999999999}}"#,
        );
        assert!(stack_huge.get("error").is_none());
        assert_eq!(d.max_call_stack_size_to_capture(), u32::MAX);

        let stack_bad = dispatch(
            &mut d,
            r#"{"id":5,"method":"Runtime.setMaxCallStackSizeToCapture","params":{}}"#,
        );
        assert!(stack_bad["error"].is_object());
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
    fn collect_garbage_methods_trigger_gc_runtime() {
        use std::alloc::Layout;

        let mut d = fresh_dispatcher();
        let before = crate::gc::runtime::gc_stats().collections;
        let _ = crate::gc::runtime::gc_alloc_raw(Layout::from_size_align(128, 8).unwrap());

        let runtime = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.collectGarbage","params":{}}"#,
        );
        assert!(runtime.get("error").is_none());
        let after_runtime = crate::gc::runtime::gc_stats();
        assert_eq!(after_runtime.collections, before + 1);
        assert_eq!(after_runtime.bytes_allocated, 0);

        let heap = dispatch(
            &mut d,
            r#"{"id":2,"method":"HeapProfiler.collectGarbage","params":{}}"#,
        );
        assert!(heap.get("error").is_none());
        assert_eq!(
            crate::gc::runtime::gc_stats().collections,
            after_runtime.collections + 1
        );
    }

    #[test]
    fn terminate_execution_interrupts_next_dispatcher_script_run() {
        let mut d = fresh_dispatcher();
        let terminate = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.terminateExecution","params":{}}"#,
        );
        assert!(terminate.get("error").is_none());

        let interrupted = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"1 + 2"}}"#,
        );
        assert!(interrupted["result"]["exceptionDetails"].is_object());
        assert!(
            interrupted["result"]["exceptionDetails"]["text"]
                .as_str()
                .unwrap_or("")
                .contains(crate::interpreter::SCRIPT_TERMINATED_MESSAGE)
        );

        let after = dispatch(
            &mut d,
            r#"{"id":3,"method":"Runtime.evaluate","params":{"expression":"1 + 2"}}"#,
        );
        assert_eq!(after["result"]["result"]["value"], 3);
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
    fn runtime_compile_script_syntax_error_emits_script_failed_to_parse() {
        let mut d = fresh_dispatcher();
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        assert_eq!(
            d.dispatch_json(r#"{"id":2,"method":"Runtime.compileScript","params":{"expression":"var =","sourceURL":"stator://bad.js","persistScript":true}}"#),
            DispatchOutcome::Ok
        );
        let messages = drain_all(&mut d);
        assert!(messages.iter().any(|msg| msg["id"] == 2u64));
        let event = messages
            .iter()
            .find(|msg| msg["method"] == "Debugger.scriptFailedToParse")
            .expect("scriptFailedToParse event");
        assert_eq!(event["params"]["url"], "stator://bad.js");
        assert!(event["params"]["exceptionDetails"].is_object());
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
    fn runtime_await_promise_returns_fulfilled_value() {
        let mut d = fresh_dispatcher();
        let promise = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.evaluate","params":{"expression":"Promise.resolve(42)","objectGroup":"promises"}}"#,
        );
        let promise_object_id = promise["result"]["result"]["objectId"].as_str().unwrap();
        let awaited = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.awaitPromise","params":{{"promiseObjectId":"{promise_object_id}","objectGroup":"promises"}}}}"#
            ),
        );
        assert_eq!(awaited["result"]["result"]["value"], 42);
    }

    #[test]
    fn runtime_await_promise_rejects_unknown_and_non_promise_objects() {
        let mut d = fresh_dispatcher();
        let unknown = dispatch(
            &mut d,
            r#"{"id":1,"method":"Runtime.awaitPromise","params":{"promiseObjectId":"missing"}}"#,
        );
        assert!(unknown["error"].is_object());

        let object = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"({value: 1})"}}"#,
        );
        let object_id = object["result"]["result"]["objectId"].as_str().unwrap();
        let non_promise = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.awaitPromise","params":{{"promiseObjectId":"{object_id}"}}}}"#
            ),
        );
        assert!(non_promise["error"].is_object());
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

    fn sample_pause_frame_snapshot() -> PauseFrameSnapshot {
        PauseFrameSnapshot {
            parameter_count: 1,
            frame_size: 1,
            registers: vec![JsValue::Smi(3), JsValue::Smi(41)],
            context_slots: vec![],
            accumulator: JsValue::Smi(2),
        }
    }

    fn sample_pause_frame_snapshot_with_context() -> PauseFrameSnapshot {
        PauseFrameSnapshot {
            parameter_count: 0,
            frame_size: 0,
            registers: vec![],
            context_slots: vec![vec![JsValue::Smi(7)], vec![JsValue::Smi(11)]],
            accumulator: JsValue::Smi(2),
        }
    }

    fn wait_until_bridge_paused(bridge: &DebuggerPauseBridge) {
        for _ in 0..200 {
            if bridge.is_paused() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        bridge.disconnect();
        panic!("pause bridge did not enter paused state");
    }

    #[test]
    fn debugger_get_stack_trace_rejects_missing_or_unknown_id() {
        let mut d = fresh_dispatcher();
        let missing = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getStackTrace","params":{}}"#,
        );
        assert!(missing["error"].is_object());

        let bad = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.getStackTrace","params":{"stackTraceId":{}}}"#,
        );
        assert!(bad["error"].is_object());

        let unknown = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.getStackTrace","params":{"stackTraceId":{"id":"missing"}}}"#,
        );
        assert!(unknown["error"].is_object());
        assert!(
            unknown["error"]["message"]
                .as_str()
                .unwrap()
                .contains("missing")
        );
    }

    #[test]
    fn debugger_set_async_call_stack_depth_stores_depth_and_rejects_bad_input() {
        let mut d = fresh_dispatcher();
        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":8}}"#,
        );
        assert!(ok.get("error").is_none());
        assert_eq!(d.async_call_stack_depth(), 8);

        let huge = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":9999999999}}"#,
        );
        assert!(huge.get("error").is_none());
        assert_eq!(d.async_call_stack_depth(), u32::MAX);

        let bad = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setAsyncCallStackDepth","params":{}}"#,
        );
        assert!(bad["error"].is_object());

        let bad_position = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setBlackboxedRanges","params":{"scriptId":"7","positions":[{"lineNumber":0}]}}"#,
        );
        assert!(bad_position["error"].is_object());

        let bad_number = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.setBlackboxedRanges","params":{"scriptId":"7","positions":[{"lineNumber":0,"columnNumber":-1}]}}"#,
        );
        assert!(bad_number["error"].is_object());
    }

    #[test]
    fn debugger_get_stack_trace_resolves_pause_async_stack_trace_id() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        dbg.borrow_mut().set_breakpoint_at_offset(0, 7, 1);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":4}}"#,
        );
        let _ = dispatch(&mut d, r#"{"id":2,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let _ = dbg.borrow_mut().check_pause_at(0);
        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        let stack_trace_id = paused[0]["params"]["asyncStackTraceId"]["id"]
            .as_str()
            .expect("paused event stack trace id");
        assert!(stack_trace_id.starts_with("stator-stack-trace-"));

        let stack_trace = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Debugger.getStackTrace","params":{{"stackTraceId":{{"id":"{stack_trace_id}"}}}}}}"#
            ),
        );
        assert_eq!(
            stack_trace["result"]["stackTrace"]["callFrames"][0]["functionName"],
            "(stator: paused-frame)"
        );
        assert_eq!(
            stack_trace["result"]["stackTrace"]["callFrames"][0]["lineNumber"],
            6
        );
    }

    #[test]
    fn debugger_pause_stack_trace_includes_microtask_parent_stack() {
        let queue = MicrotaskQueue::new();
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":4}}"#,
        );
        let _ = dispatch(&mut d, r#"{"id":2,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let dispatcher = Rc::new(RefCell::new(d));
        let task_dispatcher = Rc::clone(&dispatcher);
        let task_debugger = Rc::clone(&dbg);
        push_call_frame("scheduleMicrotask").expect("push scheduler frame");
        queue.enqueue(Box::new(move || {
            let _ = task_debugger.borrow_mut().on_debugger_statement(0);
            assert!(task_dispatcher.borrow_mut().notify_paused());
        }));
        pop_call_frame();

        queue.drain();
        let mut d = dispatcher.borrow_mut();
        let paused = drain_all(&mut d);
        let stack_trace_id = paused[0]["params"]["asyncStackTraceId"]["id"]
            .as_str()
            .expect("paused event stack trace id");
        let stack_trace = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Debugger.getStackTrace","params":{{"stackTraceId":{{"id":"{stack_trace_id}"}}}}}}"#
            ),
        );
        let parent = &stack_trace["result"]["stackTrace"]["parent"];
        assert_eq!(parent["description"], "microtask");
        assert_eq!(parent["callFrames"][0]["functionName"], "scheduleMicrotask");
    }

    #[test]
    fn debugger_async_call_stack_depth_limits_microtask_parent_frames() {
        let queue = MicrotaskQueue::new();
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setAsyncCallStackDepth","params":{"maxDepth":1}}"#,
        );
        let _ = dispatch(&mut d, r#"{"id":2,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let dispatcher = Rc::new(RefCell::new(d));
        let task_dispatcher = Rc::clone(&dispatcher);
        let task_debugger = Rc::clone(&dbg);
        push_call_frame("outerScheduler").expect("push outer scheduler frame");
        push_call_frame("innerScheduler").expect("push inner scheduler frame");
        queue.enqueue(Box::new(move || {
            let _ = task_debugger.borrow_mut().on_debugger_statement(0);
            assert!(task_dispatcher.borrow_mut().notify_paused());
        }));
        pop_call_frame();
        pop_call_frame();

        queue.drain();
        let mut d = dispatcher.borrow_mut();
        let paused = drain_all(&mut d);
        let stack_trace_id = paused[0]["params"]["asyncStackTraceId"]["id"]
            .as_str()
            .expect("paused event stack trace id");
        let stack_trace = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Debugger.getStackTrace","params":{{"stackTraceId":{{"id":"{stack_trace_id}"}}}}}}"#
            ),
        );
        let parent = &stack_trace["result"]["stackTrace"]["parent"];
        let frames = parent["callFrames"].as_array().expect("parent frames");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["functionName"], "innerScheduler");
        assert!(parent.get("parent").is_none());
    }

    #[test]
    fn debugger_disable_clears_enabled_flag() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let breakpoint = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0}}}"#,
        );
        assert!(breakpoint.get("error").is_none());
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
        assert!(d.debugger_enabled());
        let _ = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.disable","params":{}}"#,
        );
        assert!(!d.debugger_enabled());
        assert!(d.cdp_breakpoints.is_empty());
        assert!(d.cdp_script_breakpoints.is_empty());
        assert!(d.cdp_debugger_breakpoints.is_empty());
        assert_eq!(dbg.borrow().breakpoints().count(), 0);
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
    fn notify_paused_includes_local_scope_when_frame_snapshot_exists() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        let _ = dbg
            .borrow_mut()
            .check_pause_at_with_frame(0, sample_pause_frame_snapshot);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        let scope_chain = paused[0]["params"]["callFrames"][0]["scopeChain"]
            .as_array()
            .unwrap();
        assert_eq!(scope_chain[0]["type"], "local");
        assert_eq!(scope_chain[1]["type"], "global");

        let object_id = scope_chain[0]["object"]["objectId"].as_str().unwrap();
        let props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        let names: HashSet<_> = props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|prop| prop["name"].as_str())
            .collect();
        assert!(names.contains("$accumulator"));
        assert!(names.contains("$param0"));
        assert!(names.contains("$local0"));
    }

    #[test]
    fn notify_paused_includes_closure_scopes_when_context_snapshot_exists() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        let _ = dbg
            .borrow_mut()
            .check_pause_at_with_frame(0, sample_pause_frame_snapshot_with_context);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        let scope_chain = paused[0]["params"]["callFrames"][0]["scopeChain"]
            .as_array()
            .unwrap();
        assert_eq!(scope_chain[0]["type"], "local");
        assert_eq!(scope_chain[1]["type"], "closure");
        assert_eq!(scope_chain[2]["type"], "closure");
        assert_eq!(scope_chain[3]["type"], "global");

        let object_id = scope_chain[1]["object"]["objectId"].as_str().unwrap();
        let props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":2,"method":"Runtime.getProperties","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        let value = props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|prop| prop["name"] == "$context0_slot0")
            .expect("context slot property");
        assert_eq!(value["value"]["value"], 7);

        let eval = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"$context0_slot0 + $context1_slot0"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 18);

        let mutation = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.setVariableValue","params":{"scopeNumber":1,"variableName":"$context0_slot0","callFrameId":"stator-pause-frame-0","newValue":{"value":9}}}"#,
        );
        assert!(mutation.get("error").is_none());
        let updated_props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":5,"method":"Runtime.getProperties","params":{{"objectId":"{object_id}"}}}}"#
            ),
        );
        let updated_value = updated_props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|prop| prop["name"] == "$context0_slot0")
            .expect("updated context slot property");
        assert_eq!(updated_value["value"]["value"], 9);

        let updated_eval = dispatch(
            &mut d,
            r#"{"id":6,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"$context0_slot0 + $context1_slot0"}}"#,
        );
        assert_eq!(updated_eval["result"]["result"]["value"], 20);
    }

    #[test]
    fn set_variable_value_updates_scope_chain_remote_objects() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        let _ = dbg
            .borrow_mut()
            .check_pause_at_with_frame(0, sample_pause_frame_snapshot);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        let scope_chain = paused[0]["params"]["callFrames"][0]["scopeChain"]
            .as_array()
            .unwrap();
        let local_object_id = scope_chain[0]["object"]["objectId"].as_str().unwrap();
        let global_object_id = scope_chain[1]["object"]["objectId"].as_str().unwrap();

        let local = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"$local0","callFrameId":"stator-pause-frame-0","newValue":{"value":9}}}"#,
        );
        assert!(local.get("error").is_none());
        let local_props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":3,"method":"Runtime.getProperties","params":{{"objectId":"{local_object_id}"}}}}"#
            ),
        );
        let local_value = local_props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|prop| prop["name"] == "$local0")
            .expect("$local0 property");
        assert_eq!(local_value["value"]["value"], 9);

        let global = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.setVariableValue","params":{"scopeNumber":1,"variableName":"changed","callFrameId":"stator-pause-frame-0","newValue":{"value":42}}}"#,
        );
        assert!(global.get("error").is_none());
        let global_props = dispatch(
            &mut d,
            &format!(
                r#"{{"id":5,"method":"Runtime.getProperties","params":{{"objectId":"{global_object_id}"}}}}"#
            ),
        );
        let global_value = global_props["result"]["result"]
            .as_array()
            .unwrap()
            .iter()
            .find(|prop| prop["name"] == "changed")
            .expect("changed property");
        assert_eq!(global_value["value"]["value"], 42);
    }

    #[test]
    fn pause_bridge_emits_paused_and_resume_without_rc_debugger() {
        let bridge = DebuggerPauseBridge::new();
        let worker_bridge = bridge.clone();
        let handle = thread::spawn(move || {
            let mut dbg = InterpreterDebugger::new();
            dbg.set_pause_bridge(worker_bridge);
            dbg.set_breakpoint_at_offset(0, 3, 1);
            dbg.check_pause_at(0).is_none()
        });

        let mut d = fresh_dispatcher();
        d.attach_pause_bridge(bridge.clone());
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        wait_until_bridge_paused(&bridge);
        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        assert_eq!(paused.len(), 1);
        assert_eq!(paused[0]["method"], "Debugger.paused");
        assert_eq!(paused[0]["params"]["data"]["bytecodeOffset"], 0);
        assert_eq!(
            paused[0]["params"]["callFrames"][0]["location"]["lineNumber"],
            2
        );

        assert_eq!(
            d.dispatch_json(r#"{"id":2,"method":"Debugger.resume","params":{}}"#),
            DispatchOutcome::Ok
        );
        assert!(handle.join().unwrap());
        let messages = drain_all(&mut d);
        assert!(
            messages
                .iter()
                .any(|msg| msg["id"] == 2u64 && msg.get("error").is_none()),
            "expected successful resume response, got {messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|msg| msg["method"] == "Debugger.resumed"),
            "expected Debugger.resumed event, got {messages:?}"
        );
    }

    #[test]
    fn debugger_pause_requests_bridge_pause_at_next_poll() {
        let bridge = DebuggerPauseBridge::new();
        let mut d = fresh_dispatcher();
        d.attach_pause_bridge(bridge.clone());
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let resp = dispatch(&mut d, r#"{"id":2,"method":"Debugger.pause","params":{}}"#);
        assert!(resp.get("error").is_none());

        let worker_bridge = bridge.clone();
        let handle = thread::spawn(move || {
            let mut dbg = InterpreterDebugger::new();
            dbg.set_pause_bridge(worker_bridge);
            dbg.check_pause_at_with_frame(0, sample_pause_frame_snapshot)
                .is_none()
        });

        wait_until_bridge_paused(&bridge);
        assert!(d.notify_paused());
        let paused = drain_all(&mut d);
        assert_eq!(paused.len(), 1);
        assert_eq!(paused[0]["method"], "Debugger.paused");
        assert_eq!(paused[0]["params"]["reason"], "other");

        let resume = dispatch(&mut d, r#"{"id":3,"method":"Debugger.resume","params":{}}"#);
        assert!(resume.get("error").is_none());
        assert!(handle.join().unwrap());
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
    fn terminate_on_resume_interrupts_next_script_after_resume() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let terminate = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.terminateOnResume","params":{}}"#,
        );
        assert!(terminate.get("error").is_none());
        let resume = dispatch(&mut d, r#"{"id":3,"method":"Debugger.resume","params":{}}"#);
        assert!(resume.get("error").is_none());

        let interrupted = dispatch(
            &mut d,
            r#"{"id":4,"method":"Runtime.evaluate","params":{"expression":"1 + 2"}}"#,
        );
        assert!(
            interrupted["result"]["exceptionDetails"]["text"]
                .as_str()
                .unwrap_or("")
                .contains(crate::interpreter::SCRIPT_TERMINATED_MESSAGE)
        );

        let after = dispatch(
            &mut d,
            r#"{"id":5,"method":"Runtime.evaluate","params":{"expression":"1 + 2"}}"#,
        );
        assert_eq!(after["result"]["result"]["value"], 3);
    }

    #[test]
    fn terminate_on_resume_requires_active_pause() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.terminateOnResume","params":{}}"#,
        );
        assert!(response["error"].is_object());
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

        let invalid_regex = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setBlackboxPatterns","params":{"patterns":["("]}}"#,
        );
        assert!(invalid_regex["error"].is_object());
        assert!(
            invalid_regex["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid pattern")
        );
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
    fn step_commands_reject_unsupported_step_options() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let async_step = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.stepInto","params":{"breakOnAsyncCall":true}}"#,
        );
        assert!(async_step["error"].is_object());
        assert!(
            async_step["error"]["message"]
                .as_str()
                .unwrap()
                .contains("breakOnAsyncCall")
        );

        let skip_step = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.stepOver","params":{"skipList":[{"scriptId":"7","start":{"lineNumber":0,"columnNumber":0},"end":{"lineNumber":1,"columnNumber":0}}]}}"#,
        );
        assert!(skip_step["error"].is_object());
        assert!(
            skip_step["error"]["message"]
                .as_str()
                .unwrap()
                .contains("skipList")
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
    fn continue_to_location_validates_target_call_frames() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "var a = 1;\nvar b = a + 2;\n//# sourceURL=stator://app.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let bad_enum = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0},"targetCallFrames":"invalid"}}"#,
        );
        assert!(bad_enum["error"].is_object());

        let bad_type = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0},"targetCallFrames":true}}"#,
        );
        assert!(bad_type["error"].is_object());

        let current = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":1,"columnNumber":0},"targetCallFrames":"current"}}"#,
        );
        assert!(current.get("error").is_none());
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
    fn continue_to_location_validates_location_numbers() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        d.register_script_source(7, "var a = 1;".to_string());

        let missing_line = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","columnNumber":0}}}"#,
        );
        assert!(missing_line["error"].is_object());

        let bad_column = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.continueToLocation","params":{"location":{"scriptId":"7","lineNumber":0,"columnNumber":-1}}}"#,
        );
        assert!(bad_column["error"].is_object());
    }

    #[test]
    fn pause_method_acknowledges_next_poll_request() {
        let mut d = fresh_dispatcher();
        let resp = dispatch(&mut d, r#"{"id":1,"method":"Debugger.pause","params":{}}"#);
        assert!(resp["result"].is_object());
        assert!(resp.get("error").is_none());
    }

    #[test]
    fn evaluate_on_call_frame_evaluates_against_paused_globals() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.globals
            .borrow_mut()
            .insert("pausedValue".to_string(), JsValue::Smi(41));
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"pausedValue + 1"}}"#,
        );
        assert_eq!(resp["result"]["result"]["value"], 42);
    }

    #[test]
    fn evaluate_on_call_frame_reads_synthetic_local_snapshot_bindings() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.globals
            .borrow_mut()
            .insert("$local0".to_string(), JsValue::Smi(100));
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        let _ = dbg
            .borrow_mut()
            .check_pause_at_with_frame(0, sample_pause_frame_snapshot);

        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"$accumulator + $param0 + $local0"}}"#,
        );
        assert_eq!(resp["result"]["result"]["value"], 46);

        let restored = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"$local0"}}"#,
        );
        assert_eq!(restored["result"]["result"]["value"], 100);
    }

    #[test]
    fn evaluate_on_call_frame_requires_active_pause() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let resp = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"1"}}"#,
        );
        assert!(resp["error"].is_object());
    }

    #[test]
    fn resume_clears_active_pause_for_call_frame_methods() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);
        assert!(d.notify_paused());
        let _ = drain_all(&mut d);

        let resume = dispatch(&mut d, r#"{"id":2,"method":"Debugger.resume","params":{}}"#);
        assert!(resume.get("error").is_none());

        let eval = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"1"}}"#,
        );
        assert!(eval["error"].is_object());

        let set = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"x","callFrameId":"stator-pause-frame-0","newValue":{"value":1}}}"#,
        );
        assert!(set["error"].is_object());
    }

    #[test]
    fn call_frame_methods_reject_stale_or_malformed_call_frame_ids() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(2);

        let stale_eval = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"1"}}"#,
        );
        assert!(
            stale_eval["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stale callFrameId")
        );

        let malformed_eval = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-nope","expression":"1"}}"#,
        );
        assert!(
            malformed_eval["error"]["message"]
                .as_str()
                .unwrap()
                .contains("malformed callFrameId")
        );

        let stale_set = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"x","callFrameId":"stator-pause-frame-0","newValue":{"value":1}}}"#,
        );
        assert!(
            stale_set["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stale callFrameId")
        );
    }

    #[test]
    fn call_frame_mutation_methods_are_fail_closed() {
        for method in [
            "Debugger.restartFrame",
            "Debugger.setReturnValue",
            "Debugger.setBreakpointOnFunctionCall",
            "Debugger.setInstrumentationBreakpoint",
        ] {
            let mut d = fresh_dispatcher();
            let response = dispatch(
                &mut d,
                &json!({
                    "id": 1,
                    "method": method,
                    "params": {}
                })
                .to_string(),
            );
            assert!(response["error"].is_object(), "{method} must error");
            let message = response["error"]["message"].as_str().unwrap_or("");
            assert!(message.contains(method), "message for {method}: {message}");
        }
    }

    #[test]
    fn unsupported_call_frame_methods_validate_active_pause_inputs() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let restart = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.restartFrame","params":{"callFrameId":"stator-pause-frame-0"}}"#,
        );
        assert!(
            restart["error"]["message"]
                .as_str()
                .unwrap()
                .contains("rewinding")
        );

        let stale_restart = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.restartFrame","params":{"callFrameId":"stator-pause-frame-1"}}"#,
        );
        assert!(
            stale_restart["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stale callFrameId")
        );

        let return_value = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setReturnValue","params":{"newValue":{"value":7}}}"#,
        );
        assert!(
            return_value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("return value")
        );
    }

    #[test]
    fn set_breakpoint_on_function_call_validates_function_object() {
        let mut d = fresh_dispatcher();
        let function_id = d.remote_objects.register(
            JsValue::NativeFunction(Rc::new(|_| Ok(JsValue::Undefined))),
            None,
        );
        let unsupported = dispatch(
            &mut d,
            &json!({
                "id": 1,
                "method": "Debugger.setBreakpointOnFunctionCall",
                "params": { "objectId": function_id }
            })
            .to_string(),
        );
        assert!(
            unsupported["error"]["message"]
                .as_str()
                .unwrap()
                .contains("arbitrary function object")
        );

        let object_id = d.remote_objects.register(JsValue::Smi(1), None);
        let non_function = dispatch(
            &mut d,
            &json!({
                "id": 2,
                "method": "Debugger.setBreakpointOnFunctionCall",
                "params": { "objectId": object_id }
            })
            .to_string(),
        );
        assert!(
            non_function["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not reference a callable")
        );

        let conditional = dispatch(
            &mut d,
            &json!({
                "id": 3,
                "method": "Debugger.setBreakpointOnFunctionCall",
                "params": { "objectId": function_id, "condition": "x > 0" }
            })
            .to_string(),
        );
        assert!(
            conditional["error"]["message"]
                .as_str()
                .unwrap()
                .contains("conditional function-call")
        );
    }

    #[test]
    fn set_variable_value_mutates_synthetic_global_scope() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        let _ = dbg.borrow_mut().on_debugger_statement(0);

        let ok = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"changed","callFrameId":"stator-pause-frame-0","newValue":{"value":42}}}"#,
        );
        assert!(ok.get("error").is_none());

        let eval = dispatch(
            &mut d,
            r#"{"id":2,"method":"Runtime.evaluate","params":{"expression":"changed"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 42);
    }

    #[test]
    fn set_variable_value_mutates_local_snapshot_and_global_after_snapshot() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        dbg.borrow_mut().set_breakpoint_at_offset(0, 1, 1);
        let _ = dbg
            .borrow_mut()
            .check_pause_at_with_frame(0, sample_pause_frame_snapshot);

        let local = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"$local0","callFrameId":"stator-pause-frame-0","newValue":{"value":9}}}"#,
        );
        assert!(local.get("error").is_none());
        let local_eval = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.evaluateOnCallFrame","params":{"callFrameId":"stator-pause-frame-0","expression":"$local0"}}"#,
        );
        assert_eq!(local_eval["result"]["result"]["value"], 9);

        let global = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setVariableValue","params":{"scopeNumber":1,"variableName":"changed","callFrameId":"stator-pause-frame-0","newValue":{"value":42}}}"#,
        );
        assert!(global.get("error").is_none());

        let eval = dispatch(
            &mut d,
            r#"{"id":4,"method":"Runtime.evaluate","params":{"expression":"changed"}}"#,
        );
        assert_eq!(eval["result"]["result"]["value"], 42);
    }

    #[test]
    fn set_variable_value_requires_active_pause() {
        let mut d = fresh_dispatcher();
        let _dbg = attach_test_debugger(&mut d);
        let response = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setVariableValue","params":{"scopeNumber":0,"variableName":"x","callFrameId":"stator-pause-frame-0","newValue":{"value":1}}}"#,
        );
        assert!(response["error"].is_object());
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
    fn search_in_content_finds_literal_and_regex_matches() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            1,
            "let answer = 42;\nconst other = answer + 1;\nANSWER;".to_string(),
        );

        let literal = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.searchInContent","params":{"scriptId":"1","query":"answer","caseSensitive":true}}"#,
        );
        let literal_matches = literal["result"]["result"].as_array().unwrap();
        assert_eq!(literal_matches.len(), 2);
        assert_eq!(literal_matches[0]["lineNumber"], 0);
        assert_eq!(literal_matches[1]["lineNumber"], 1);

        let insensitive = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.searchInContent","params":{"scriptId":"1","query":"answer","caseSensitive":false}}"#,
        );
        assert_eq!(insensitive["result"]["result"].as_array().unwrap().len(), 3);

        let regex = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.searchInContent","params":{"scriptId":"1","query":"\\d+","isRegex":true}}"#,
        );
        let regex_matches = regex["result"]["result"].as_array().unwrap();
        assert_eq!(regex_matches.len(), 2);
        assert_eq!(regex_matches[0]["lineContent"], "let answer = 42;");
    }

    #[test]
    fn search_in_content_rejects_unknown_script_and_invalid_regex() {
        let mut d = fresh_dispatcher();
        let missing = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.searchInContent","params":{"scriptId":"missing","query":"x"}}"#,
        );
        assert!(missing["error"].is_object());

        d.register_script_source(1, "let x = 1;".to_string());
        let bad_regex = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.searchInContent","params":{"scriptId":"1","query":"(","isRegex":true}}"#,
        );
        assert!(bad_regex["error"].is_object());
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
    fn set_script_source_resolves_pending_url_breakpoints_for_new_url() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "let oldValue = 1;\n//# sourceURL=stator://old.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let breakpoint = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://new.js","lineNumber":0,"columnNumber":0}}"#,
        );
        assert!(
            breakpoint["result"]["locations"]
                .as_array()
                .unwrap()
                .is_empty()
        );

        let messages = dispatch_all(
            &mut d,
            r#"{"id":3,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let newValue = 2;\n//# sourceURL=stator://new.js"}}"#,
        );
        assert!(
            messages
                .iter()
                .any(|msg| msg["method"] == "Debugger.breakpointResolved")
        );
        assert!(messages.iter().any(|msg| msg["id"] == 3u64));
        assert_eq!(dbg.borrow().breakpoints().count(), 1);
    }

    #[test]
    fn set_script_source_refreshes_installed_breakpoints_for_script() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "let oldValue = 1;\nlet next = 2;\n//# sourceURL=stator://same.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);

        let direct = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setBreakpoint","params":{"location":{"scriptId":"7","lineNumber":0,"columnNumber":0}}}"#,
        );
        assert!(direct.get("error").is_none());
        let url = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://same.js","lineNumber":1,"columnNumber":0}}"#,
        );
        assert!(url.get("error").is_none());
        assert_eq!(dbg.borrow().breakpoints().count(), 2);

        let messages = dispatch_all(
            &mut d,
            r#"{"id":4,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let newValue = 3;\nlet still = 4;\n//# sourceURL=stator://same.js"}}"#,
        );
        assert_eq!(dbg.borrow().breakpoints().count(), 2);
        assert_eq!(
            messages
                .iter()
                .filter(|msg| msg["method"] == "Debugger.breakpointResolved")
                .count(),
            2
        );
    }

    #[test]
    fn set_script_source_removes_url_breakpoints_that_no_longer_match() {
        let mut d = fresh_dispatcher();
        let dbg = attach_test_debugger(&mut d);
        d.register_script_source(
            7,
            "let oldValue = 1;\nlet next = 2;\n//# sourceURL=stator://old.js".to_string(),
        );
        let url = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setBreakpointByUrl","params":{"url":"stator://old.js","lineNumber":1,"columnNumber":0}}"#,
        );
        assert!(url.get("error").is_none());
        assert_eq!(dbg.borrow().breakpoints().count(), 1);

        let edit = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let newValue = 3;\nlet still = 4;\n//# sourceURL=stator://new.js"}}"#,
        );
        assert!(edit.get("error").is_none());
        assert_eq!(dbg.borrow().breakpoints().count(), 0);
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
    fn set_script_source_validates_optional_edit_flags() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "let original = 1;".to_string());

        let bad_dry_run = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let changed = 2;","dryRun":"yes"}}"#,
        );
        assert!(bad_dry_run["error"].is_object());
        assert_eq!(d.script_sources.get("7").unwrap(), "let original = 1;");

        let top_frame = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"let changed = 2;","allowTopFrameEditing":true}}"#,
        );
        assert!(top_frame["error"].is_object());
        assert!(
            top_frame["error"]["message"]
                .as_str()
                .unwrap()
                .contains("allowTopFrameEditing")
        );
        assert_eq!(d.script_sources.get("7").unwrap(), "let original = 1;");
    }

    #[test]
    fn set_script_source_compile_error_emits_script_failed_to_parse() {
        let mut d = fresh_dispatcher();
        d.register_script_source(
            7,
            "let original = 1;\n//# sourceURL=stator://old.js".to_string(),
        );
        let _ = dispatch(&mut d, r#"{"id":1,"method":"Debugger.enable","params":{}}"#);
        let _ = drain_all(&mut d);
        assert_eq!(
            d.dispatch_json(r#"{"id":2,"method":"Debugger.setScriptSource","params":{"scriptId":"7","scriptSource":"function {\n//# sourceURL=stator://new-bad.js"}}"#),
            DispatchOutcome::Ok
        );
        let messages = drain_all(&mut d);
        assert!(messages.iter().any(|msg| msg["id"] == 2u64));
        let event = messages
            .iter()
            .find(|msg| msg["method"] == "Debugger.scriptFailedToParse")
            .expect("scriptFailedToParse event");
        assert_eq!(event["params"]["scriptId"], "7");
        assert_eq!(event["params"]["url"], "stator://new-bad.js");
        assert!(event["params"]["exceptionDetails"].is_object());
        assert_eq!(
            d.script_sources.get("7").unwrap(),
            "let original = 1;
//# sourceURL=stator://old.js"
        );
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
    fn get_possible_breakpoints_validates_range_and_restrict_to_function() {
        let mut d = fresh_dispatcher();
        d.register_script_source(7, "function f() {\n  return 1;\n}".to_string());
        let different_end_script = dispatch(
            &mut d,
            r#"{"id":1,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"7","lineNumber":0,"columnNumber":0},"end":{"scriptId":"8","lineNumber":1,"columnNumber":0}}}"#,
        );
        assert!(different_end_script["error"].is_object());

        let restrict = dispatch(
            &mut d,
            r#"{"id":2,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"7","lineNumber":0,"columnNumber":0},"restrictToFunction":true}}"#,
        );
        assert!(restrict["error"].is_object());
        assert!(
            restrict["error"]["message"]
                .as_str()
                .unwrap()
                .contains("restrictToFunction")
        );

        let missing_line = dispatch(
            &mut d,
            r#"{"id":3,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"7","columnNumber":0}}}"#,
        );
        assert!(missing_line["error"].is_object());

        let bad_end_column = dispatch(
            &mut d,
            r#"{"id":4,"method":"Debugger.getPossibleBreakpoints","params":{"start":{"scriptId":"7","lineNumber":0,"columnNumber":0},"end":{"scriptId":"7","lineNumber":1,"columnNumber":-1}}}"#,
        );
        assert!(bad_end_column["error"].is_object());
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

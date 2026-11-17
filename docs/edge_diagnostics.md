# Edge performance diagnostics

This document describes the always-on, release-safe diagnostic counters
that Stator exposes for Edge performance proof runs. All counters
described below are intended to be enabled in release builds: each
update is a handful of relaxed atomic operations and (where timing is
involved) a single `Instant::now()` pair per public entry point.

## Inspector context groups and lifecycle events

The in-process inspector used by Edge DevTools exposes a stable CDP
execution-context registry. Each inspector starts with context group `1` and
execution context `1` (`origin = "stator"`, `name = "stator"`), and additional
groups/contexts receive monotonically increasing IDs that are not reused while
live. `Runtime.enable` deterministically emits `Runtime.executionContextCreated`
for every live context before its acknowledgement.

Embedders can query the default group/context IDs through
`stator_inspector_context_group_id` and `stator_inspector_context_id`, create
new groups/contexts with `stator_inspector_create_context_group` and
`stator_inspector_create_context`, and tear contexts down with
`stator_inspector_destroy_context` or `stator_inspector_clear_context_group`.
Runtime-enabled sessions observe one `Runtime.executionContextCreated`,
`Runtime.executionContextDestroyed`, or `Runtime.executionContextsCleared`
event per successful lifecycle transition; repeated teardown is a no-op.
Creating or destroying contexts with no attached inspector session is
fail-closed and only updates the registry.

## Inspector RemoteObject registry

Each inspector session owns a per-session `RemoteObjectRegistry` that mints
stable decimal `objectId` strings (`"1"`, `"2"`, ...) for every non-primitive
[`JsValue`] surfaced through CDP. Primitive values (`undefined`, `null`,
booleans, numbers, strings, BigInt, the internal `TheHole` sentinel) are
serialised by value and **never** mint an `objectId`. Heap values
(`PlainObject`, `Array`, `Function`, `NativeFunction`, `Error`, `Promise`,
`Proxy`, `ArrayBuffer`, `TypedArray`, `DataView`, `Generator`, `Iterator`,
`ModuleBinding`, raw `Object`, `Symbol`, `Context`) always do.

`Runtime.evaluate` and `Runtime.callFunctionOn` honour the optional
`objectGroup` parameter; every `RemoteObject` minted while producing the
result inherits that group. `Runtime.getProperties` registers nested child
values under the parent's group so that `Runtime.releaseObjectGroup`
cascades.

### Lifetime and identity

- IDs are scoped to one CDP session. Two sessions never observe each other's
  IDs, and an ID minted in session A always fails closed when used in
  session B with a structured error.
- IDs are monotonically increasing and never reused while live.
- `Runtime.releaseObject` with an unknown or already-released ID returns a
  structured error rather than silently succeeding, so embedders catch
  double-release bugs early.
- `Runtime.releaseObjectGroup` with an unknown group is a no-op success,
  matching V8 inspector semantics.
- Dropping the inspector session (and therefore the dispatcher) drops the
  registry; no out-of-band cleanup is required.
- The registry holds `JsValue` clones, which for heap variants is an `Rc`
  bump — it does not introduce additional GC strong roots beyond the entry
  itself.

### `Runtime.getProperties` and preview coverage

Stator's `Runtime.getProperties` returns own properties for the following
shapes:

| `JsValue` variant | Properties returned                                       |
|-------------------|-----------------------------------------------------------|
| `PlainObject`     | Every entry in insertion order with the engine's `writable` / `enumerable` / `configurable` attributes |
| `Array`           | Indexed elements `0..n` followed by a non-enumerable, non-configurable `length` |
| `Function`        | Synthetic `length` (parameter count) and `name`           |
| `Error`           | No lazy own properties yet; preview exposes derived `name` / `message` |
| Other heap classes (Promise, Proxy, ArrayBuffer, TypedArray, DataView, Generator, Iterator, NativeFunction, ModuleBinding, raw `Object`, `Context`) | Empty own-property list — DevTools renders the `className`/`description` from the RemoteObject payload |

When `Runtime.evaluate` or `Runtime.callFunctionOn` receives
`generatePreview: true`, Stator emits CDP `preview` objects for
`PlainObject`, `Array`, `Function`, and `Error` results/exceptions. Previews
are intentionally shallow: at most five properties are included, nested
object previews stop after one level, and unsupported host/native values do
not fabricate preview data. `customPreview` is not emitted.

### `Runtime.ExceptionDetails` coverage

`Runtime.evaluate` and `Runtime.callFunctionOn` surface parse/compile/runtime
failures as successful CDP replies containing `exceptionDetails` plus an
`undefined` result, matching DevTools' expected Runtime shape. Details use
per-session monotonically increasing `exceptionId` values, include the
requested `executionContextId`, report `lineNumber` / `columnNumber` as `0`
unless source locations are available, and carry `url` when the request
supplies `sourceURL` or a trailing `//# sourceURL=` comment.

For uncaught JavaScript `throw` completions, the original thrown `JsValue` is
also exposed as `exception` with the same RemoteObject/preview rules and
object-group lifetime as normal results. Engine/parser errors that do not
have a thrown JavaScript value omit `exception` rather than inventing one.
Error stack strings are exposed as `stackTrace.description`; call frames are
included only from the currently captured JavaScript call stack.

Runtime-enabled sessions receive `Runtime.exceptionThrown` before the
corresponding response. Sessions that have not sent `Runtime.enable` still
receive `exceptionDetails` in the response but no event is queued. Stator has
no exception-clearing lifecycle today, so `Runtime.exceptionRevoked` is not
emitted.

### CDP method coverage

| Method                       | Behaviour                                          |
|------------------------------|---------------------------------------------------|
| `Runtime.evaluate`           | Returns `result`; failures return `exceptionDetails`. |
| `Runtime.callFunctionOn`     | Returns `result`; failures return `exceptionDetails`. |
| `Runtime.getProperties`      | Required `objectId`. Unknown/stale ID → error.    |
| `Runtime.releaseObject`      | Required `objectId`. Unknown/stale ID → error.    |
| `Runtime.releaseObjectGroup` | Required `objectGroup`. Unknown group → no-op OK. |
| `Debugger.enable`/`disable`  | Toggles per-session domain state.                 |
| `Debugger.setPauseOnExceptions` | `"none"`/`"uncaught"`/`"all"`; other values → typed error. Forwarded to the attached interpreter debugger (and cached so a debugger attached later inherits the chosen state). |
| `Debugger.setBreakpointByUrl`| Returns a per-session breakpoint id with a single location. |
| `Debugger.resume`            | Applies `Continue` to the attached debugger when there is an active pause; emits `Debugger.resumed`. Otherwise a silent no-op success so DevTools teardown does not error. |
| `Debugger.stepInto`/`stepOver`/`stepOut` | Applies the matching step on the attached debugger and emits `Debugger.resumed`; typed error when no debugger is attached or no pause is active. |
| `Debugger.pause`             | Fail-closed: synchronous interpreter cannot be interrupted. Use `debugger;` or `setBreakpointByUrl` instead. |
| `Debugger.evaluateOnCallFrame` | Fail-closed: call-frame snapshots are not exposed through CDP yet. |
| `Debugger.getScriptSource`/`getPossibleBreakpoints` | Fail-closed: not bridged through the dispatcher yet. |

## Bridging interpreter Debugger pauses into CDP

The in-process inspector owns a single `inspector::debugger::Debugger`
instance — exposed to the embedder via `InProcessInspector::debugger()` and
shared with every session's dispatcher. Embedders attach it to the
interpreter with `interpreter::attach_debugger(inspector.debugger())` before
calling `Interpreter::run`. When `Interpreter::run` returns
`StatorError::DebuggerPaused`, the embedder calls
`InProcessInspector::notify_paused()`, which emits one `Debugger.paused`
event into every session that has sent `Debugger.enable`. Sessions without
`Debugger.enable` receive nothing.

The emitted `Debugger.paused` payload is intentionally conservative:

- `reason` maps interpreter pause reasons onto CDP strings:
  `DebuggerStatement` → `"debuggerStatement"`,
  `Breakpoint` → `"other"` plus a populated `hitBreakpoints` array,
  `Step` → `"other"`,
  `Exception` → `"exception"`.
- `callFrames` contains exactly one synthetic frame named
  `(stator: paused-frame)` carrying the paused source line and bytecode
  offset under `data.bytecodeOffset`/`data.pausedFrame`. The interpreter
  does not retain a portable per-frame snapshot at pause time yet, so we
  deliberately avoid fabricating multi-frame stacks or scope chains.
- `Debugger.evaluateOnCallFrame` is fail-closed for the same reason; use
  `Runtime.evaluate` while paused instead.

`Debugger.resume`/`stepInto`/`stepOver`/`stepOut` drive the same shared
debugger through `Debugger::apply_action`, and emit a matching
`Debugger.resumed` event into the calling session. Embedders can also push
`Debugger.resumed` explicitly through
`InProcessInspector::notify_resumed()` after the embedder resumes
execution out-of-band.

No new FFI symbols are required — CDP requests/responses flow through the
existing `stator_inspector_dispatch` and `stator_inspector_next_message`
entry points.

## Tier-promotion latency counters

### Scope

The `stator_jse::compiler::tier_latency` module records, per JIT tier:

- **`requested`** — every time a synchronous tier-promotion entry point
  (`force_tier_sync` and the underlying `force_baseline_sync`,
  `force_maglev_sync`, `force_turbofan_sync`) is invoked.
- **`succeeded`** — promotions that left the target tier observable as
  ready when the call returned, plus the elapsed wall time per sample.
- **`failed`** — promotions that performed real work but did not leave
  the target tier ready (graph-build failure, degenerate graph, codegen
  failure, executable-allocation failure), plus the elapsed wall time
  per sample.
- **`success_buckets[…]`** — log-spaced histogram of successful
  promotion durations from ≤ 1 µs up to ≤ 10 s, with an overflow bucket
  for anything slower.

Short-circuit paths that do not consume any compile time
(`AlreadyReady`, `DeoptBlocked`) are counted as requests but explicitly
**not** recorded as successes or failures — they would otherwise
pollute the latency histogram with near-zero samples. `JitDisabled` and
`UnsupportedTier` outcomes return before the request is recorded at
all, so they are excluded from these counters.

### Scope limitations

Counters are **process-global aggregates**. Per-script attribution is
not exposed today: `BytecodeArray` does not carry a stable script hash
through the tier-promotion entry points, and adding one is outside the
scope of this diagnostic surface. Edge consumers should diff successive
snapshots across the workload boundary they care about.

### Rust API

```rust
use stator_jse::compiler::tier_latency::{self, PromotionTier};

tier_latency::reset();
// ... run workload ...
let snap = tier_latency::snapshot();
let maglev = snap.for_tier(PromotionTier::Maglev);
println!(
    "maglev: {} req, {} ok, {} fail, mean_ok_ns = {}",
    maglev.requested,
    maglev.succeeded,
    maglev.failed,
    maglev.mean_success_ns(),
);
```

### C FFI

```c
StatorTierLatencyStats stats;
stator_isolate_reset_tier_latency_stats(NULL);
/* ... run workload ... */
stator_isolate_get_tier_latency_stats(NULL, &stats);
printf("baseline ok=%llu mean_ns=%llu max_ns=%llu\n",
       (unsigned long long) stats.baseline.succeeded,
       (unsigned long long)
           (stats.baseline.succeeded
                ? stats.baseline.success_total_ns / stats.baseline.succeeded
                : 0),
       (unsigned long long) stats.baseline.success_max_ns);
```

Both FFI calls accept a null `StatorIsolate*` — the counters are
process-global, not heap-owned — matching the existing
`stator_isolate_get_tiering_stats` / `stator_isolate_reset_tiering_stats`
shape.

The histogram bucket count is exposed as
`STATOR_TIER_LATENCY_BUCKET_COUNT` and **must match**
`stator_jse::compiler::tier_latency::NUM_HISTOGRAM_BUCKETS`. An FFI
contract test enforces this. Bumping it is a breaking ABI change.

### Interaction with deterministic force-tier APIs

`stator_script_force_tier` and `stator_script_force_maglev_compile`
both reach `force_tier_sync` under the hood, so calls to those entry
points are reflected directly in the tier-latency counters. This makes
the counters useful for deterministic Edge proof runs that force a
specific tier rather than relying on the heuristic tier-up path.

## JIT code memory counters

`stator_jse::compiler::jit_memory` exposes release-safe aggregate
counters for native-code footprint:

| Field | Meaning |
| --- | --- |
| `code_bytes_emitted` | Native instruction bytes emitted by the compiler tier. |
| `executable_bytes_committed` | Bytes successfully copied into executable memory. |
| `executable_bytes_freed` | Executable bytes observed by teardown/drop hooks. |
| `executable_pages_committed` | Page-rounded executable allocations. |
| `executable_pages_reserved` | Page-rounded reservations; currently equal to committed pages for Stator's allocation APIs. |
| `executable_pages_freed` | Page-rounded executable releases. |
| `live_code_bytes` | Current live executable-code bytes after observed frees. |
| `code_cache_artifact_bytes` | Native code-cache artifact bytes produced by this tier. |

The stable tier rows are `baseline`, `maglev`, `turbofan`, and
`cranelift`.  The current top-tier compiler is Cranelift-backed, so
Cranelift rows receive emitted-code samples from that backend and the
plain `turbofan` row remains zero until a distinct Turbofan emitter is
wired in.  Native code-cache artifacts are not produced today; those
fields are stable zeroes until `baseline-code` / `jit-code` artifacts
are implemented.  Script/module bytecode cache blobs are intentionally
not counted as native-code artifacts.

### Scope limitations

Counters are process-global aggregates, not true per-isolate rows:
executable-memory allocation and code-cache ownership currently happen
below the FFI isolate boundary.  The FFI snapshot therefore reports
`isolate_scoped = false`; Edge should reset or diff snapshots around the
proof workload being measured.  Executable page counts use Stator's
observed allocation sizes rounded to 4 KiB pages, matching current Edge
proof targets; they are not OS working-set measurements.

Temporary `ExecutableMemory::new` allocations are not attributed unless
the caller uses `ExecutableMemory::new_for_tier`.  Persistent Baseline
and Maglev executable caches are attributed and update live/freed bytes
on drop.  Cranelift code bytes are emitted-code counters only today
because Cranelift's internal code memory is managed by `cranelift-jit`
without a stable teardown hook in Stator.

### Rust API

```rust
use stator_jse::compiler::jit_memory::{self, JitMemoryTier};

jit_memory::reset();
// ... run workload ...
let snap = jit_memory::snapshot();
let maglev = snap.for_tier(JitMemoryTier::Maglev);
println!(
    "maglev emitted={} live_exec={}",
    maglev.code_bytes_emitted,
    maglev.live_code_bytes,
);
```

### C FFI

```c
StatorJitMemoryStats stats = {0};
stator_isolate_reset_jit_memory_stats(NULL);
/* ... run workload ... */
stator_isolate_get_jit_memory_stats(NULL, &stats);
printf("maglev emitted=%llu live=%llu isolate_scoped=%d\n",
       (unsigned long long) stats.tiers[1].code_bytes_emitted,
       (unsigned long long) stats.tiers[1].live_code_bytes,
       (int) stats.isolate_scoped);
```

`STATOR_JIT_MEMORY_TIER_COUNT` mirrors the Rust tier count and is
enforced by FFI tests.  The FFI ABI minor version was bumped when this
surface was added.

## Deopt reason histogram counters

The `stator_jse::compiler::deopt_counters` module records release-safe
histograms of real deopt, bailout, and runtime-fallback events by tier:
`baseline`, `maglev`, and `turbofan` (the Cranelift-backed top tier).
The reason schema is intentionally small and stable:

| Field | Meaning today |
| --- | --- |
| `unsupported_opcode_or_runtime_fallback` | The tier hit unsupported bytecode/codegen and returned the generic `JIT_DEOPT` fallback sentinel. |
| `arithmetic_overflow` | A checked small-integer arithmetic fast path overflowed. |
| `runtime_stub_fallback` | A runtime stub could not satisfy the specialised path and returned a stub deopt sentinel. |
| `global_load_fallback` | A promoted global access fast path failed. |
| `termination_interrupt` | Embedder termination or interrupt polling requested unwind. This is not treated as code-quality instability. |
| `division_by_zero` | Integer division by zero required lower-tier/runtime semantics. |
| `internal_error` | Unknown, internal, or out-of-range deopt sentinel. |

### Current tier limitations

Only Maglev execution is active in the current runtime, so Maglev deopt
sentinels are the only real increment source. Baseline JIT and
Turbofan/Cranelift execution paths are currently disabled due known codegen
correctness/safety issues; their histogram rows are still exposed and reset but
remain zero. Do not interpret zero baseline/Turbofan counts as proof that those
tiers are stable until execution is re-enabled.

### Rust API

```rust
use stator_jse::compiler::deopt_counters::{self, DeoptReason, DeoptTier};

deopt_counters::reset();
// ... run workload ...
let snap = deopt_counters::snapshot();
let maglev = snap.for_tier(DeoptTier::Maglev);
println!(
    "maglev stub fallback deopts = {}",
    maglev.count(DeoptReason::RuntimeStubFallback),
);
```

### C FFI

```c
StatorDeoptHistogramStats stats;
stator_isolate_reset_deopt_histogram_stats(NULL);
/* ... run workload ... */
stator_isolate_get_deopt_histogram_stats(NULL, &stats);
printf("maglev stub fallback deopts=%llu\n",
       (unsigned long long) stats.maglev.runtime_stub_fallback);
```

Both FFI calls accept a null `StatorIsolate*` because the histogram is
process-global. `STATOR_DEOPT_TIER_COUNT` and `STATOR_DEOPT_REASON_COUNT`
mirror the Rust tier/reason enum counts and are enforced by FFI tests.

## Inline-cache counters

Stator records release-safe per-tier inline-cache (IC) probe / hit / miss
/ transition counters so that Edge proof runs can attribute slow paths to
specific operation kinds.  The counter table is keyed by
`(IcTier, IcOp, IcEvent)`:

| Dimension | Variants |
| --------- | -------- |
| `IcTier`  | `Interpreter`, `Baseline`, `Maglev`, `Turbofan` |
| `IcOp`    | `NamedLoad`, `NamedStore`, `IndexedLoad`, `IndexedStore`, `Call` |
| `IcEvent` | `Probe`, `Hit`, `Miss`, `Transition` |

The full snapshot has shape `4 × 5 × 4 = 80` cells.

### Current tier limitations

Only the `Interpreter` row is populated today. The interpreter is the only
consistently active IC source in this build: the Baseline JIT is gated off
in default workloads, Turbofan/Cranelift execution is disabled while
correctness issues are being fixed, and Baseline's `jit_runtime_*` IC stubs
do not yet feed these global counters.  The `Baseline`, `Maglev`, and
`Turbofan` rows are wired into the schema so embedders can rely on a
stable layout, but they read zero until the corresponding JIT IC stubs are
instrumented in a follow-up.

For named-property load and store, an IC site is considered "eligible" and
counted when the bytecode carries a non-sentinel feedback slot and the
receiver is a `PlainObject` / `Array` / `Function` (load) or `PlainObject`
(store).  For keyed/indexed access, every Smi-key receiver is probed.

### Rust API

```rust
use stator_jse::ic::counters::{self, IcEvent, IcOp, IcTier};

counters::reset();
// ... run workload ...
let snap = counters::snapshot();
let interp_named_load = snap
    .for_tier(IcTier::Interpreter)
    .for_op(IcOp::NamedLoad);
let hit_rate = interp_named_load.count(IcEvent::Hit) as f64
    / interp_named_load.count(IcEvent::Probe).max(1) as f64;
println!("interpreter named-load IC hit rate = {hit_rate:.3}");
```

### C FFI

```c
StatorIcCountersStats stats = {0};
stator_isolate_reset_ic_counters_stats(NULL);
/* ... run workload ... */
stator_isolate_get_ic_counters_stats(NULL, &stats);
printf("interp named-load hits=%llu miss=%llu transitions=%llu\n",
       (unsigned long long) stats.interpreter.named_load.hit,
       (unsigned long long) stats.interpreter.named_load.miss,
       (unsigned long long) stats.interpreter.named_load.transition);
```

Both FFI calls accept a null `StatorIsolate*` because the counters are
process-global. `STATOR_IC_TIER_COUNT`, `STATOR_IC_OP_COUNT`, and
`STATOR_IC_EVENT_COUNT` mirror the Rust enum counts and are enforced by
FFI tests.


## OSR entry/exit counters

Stator exposes a stable release-safe OSR diagnostics schema through
`stator_jse::compiler::osr_counters` and the C FFI calls
`stator_isolate_get_osr_counters_stats` /
`stator_isolate_reset_osr_counters_stats`.

### Current tier limitations

The current runtime **does not support true mid-frame OSR**. Hot-loop
back-edges may request Baseline, Maglev, or Turbofan compilation, but the
active interpreter dispatch paths deliberately avoid jumping into JIT code
mid-loop: compiled code would re-run the function from entry rather than
restore the live interpreter frame. Those compile requests are tiering
activity, not OSR events, and they are covered by the compile and
promotion-latency counters above.

Because there is no true OSR entry/exit machinery wired in today,
`true_osr_supported` is `false`, `per_script_row_count` is `0`, and every
entry/exit counter remains zero. Edge consumers should treat this as
"OSR unsupported in this build", not as proof that loop OSR was stable.
Future real OSR support must increment only genuine mid-execution handoff
attempts/successes/failures and exits; it must not count force-tier APIs,
background compilation, or next-call tier-up as OSR.

### Schema

Entry counters are split by source tier and target tier:

```text
entries.from_<source>.<target>.attempts
entries.from_<source>.<target>.successes
entries.from_<source>.<target>.failures
```

The tier dimensions are `interpreter`, `baseline`, `maglev`, and
`turbofan`. Exit counters are split by the tier entered through OSR and a
stable reason set:

| Field | Meaning |
| --- | --- |
| `normal_return` | OSR-entered code returned normally. |
| `deopt` | OSR-entered code exited through a deopt/bailout path. |
| `exception` | OSR-entered code exited by throwing an exception. |
| `termination_interrupt` | OSR-entered code exited due to embedder termination or interrupt polling. |

### Per-script attribution

Per-script OSR attribution is not available today. The exported schema is
aggregate-only and reports `per_script_row_count == 0`; no script hash rows
are emitted until bytecode/script metadata is carried through a real OSR
entry path.

### Rust API

```rust
use stator_jse::compiler::osr_counters::{self, OsrTier};

osr_counters::reset();
// ... run workload ...
let snap = osr_counters::snapshot();
assert!(!snap.true_osr_supported);
let interp_to_baseline = snap.entry(OsrTier::Interpreter, OsrTier::Baseline);
println!("OSR attempts = {}", interp_to_baseline.attempts);
```

### C FFI

```c
StatorOsrCountersStats stats = {0};
stator_isolate_reset_osr_counters_stats(NULL);
/* ... run workload ... */
stator_isolate_get_osr_counters_stats(NULL, &stats);
if (!stats.true_osr_supported) {
    /* Current Stator build has no true mid-frame OSR. */
}
printf("interp->baseline OSR attempts=%llu\n",
       (unsigned long long) stats.entries.from_interpreter.baseline.attempts);
```

Both FFI calls accept a null `StatorIsolate*` because the counters are
process-global. `STATOR_OSR_TIER_COUNT` and
`STATOR_OSR_EXIT_REASON_COUNT` mirror the Rust enum counts and are enforced
by FFI tests.

### Edge usage notes

For loop-tiering instability in current Edge proof runs, combine this OSR
snapshot with:

- tier-latency counters to see deterministic force-tier and promotion
  request outcomes;
- compile counters to confirm whether Baseline/Maglev/Turbofan compilation
  actually ran;
- deopt histograms to explain optimized-tier exits once execution reaches a
  JIT tier.

If OSR counters are all zero while `true_osr_supported == false`, report the
build as "no true OSR support" rather than "zero OSR churn".

## JIT Control-Flow Guard / CET shadow-stack diagnostics

### Scope

Chromium/Edge ship with Windows Control-Flow Guard (CFG) enabled and
rely on hardware CET shadow stacks plus Indirect Branch Tracking
(`ENDBR64` prologues) on capable CPUs.  Any JIT-emitted executable
region must comply with both mitigations: every valid indirect call
target must be registered with the CFG bitmap
(`SetProcessValidCallTargets`), and every indirect-call target must
begin with `ENDBR64` so that CET does not trip on the first instruction.

The `stator_jse::jit_mitigations` module exposes a release-safe,
fail-closed diagnostic surface that reports — per JIT tier — whether
the build currently has the metadata needed to perform either
registration, and a process-level probe for the OS-side mitigation
policy (`GetProcessMitigationPolicy`).

The module deliberately does **not** call `SetProcessValidCallTargets`
on behalf of any tier today: handing the OS an unverified target list
would *claim* mitigation coverage we have not actually produced.  When
a tier learns to emit a verified target list and CET-compatible
prologues, flip [`JitMitigationsTier::cfg_supported`] /
[`JitMitigationsTier::cet_compatible`] for that arm and route the
registration through a new helper that performs the real OS call.

### Current tier support matrix

| Tier      | `cfg_supported` | `cet_compatible` | Notes |
| --------- | --------------- | ---------------- | ----- |
| Baseline  | **no**          | **no**           | Handcrafted `masm_x64` prologue/epilogue, no `ENDBR64`, no exported target list. |
| Maglev    | **no**          | **no**           | Same in-tree assembler. |
| Turbofan  | **no**          | **no**           | Reserved row; current Stator native backend is Cranelift. |
| Cranelift | **no**          | **no**           | Standalone Cranelift code shares the JIT pool; targets are not enumerated and `ENDBR64` prologue emission is not yet wired in. |

Until a tier flips an arm, [`jit_mitigations::record_cfg_registration`]
returns `MitigationError::UnsupportedTier` (or
`MitigationError::UnsupportedPlatform` off Windows) and bumps the
fail-closed `cfg_unsupported_tier_attempts` counter.
[`jit_mitigations::record_cet_page`] called with `compatible = true`
still lands in the **incompatible** bucket while every tier reports
`cet_compatible == false`.

### Counters

Per tier (Baseline / Maglev / Turbofan / Cranelift):

| Field | Meaning |
| ----- | ------- |
| `cfg_supported` | Mirrors `JitMitigationsTier::cfg_supported`.  Currently always `false`. |
| `cet_compatible` | Mirrors `JitMitigationsTier::cet_compatible`.  Currently always `false`. |
| `cfg_register_attempts` | Every call into `record_cfg_registration`. |
| `cfg_register_successes` | Calls that successfully delivered the target list to the OS (zero while every tier is `cfg_supported == false`). |
| `cfg_register_failures` | Empty region / OS-rejected outcomes after the supported-tier check passed. |
| `cfg_unsupported_tier_attempts` | Fail-closed sentinel: the platform or tier had no real registration backend. |
| `cfg_targets_registered` | Sum of individual targets successfully registered. |
| `cet_pages_marked_compatible` | JIT pages observed and verified CET-compatible. |
| `cet_pages_marked_incompatible` | JIT pages observed CET-incompatible (also the fail-closed bucket for unverified claims). |

Aggregate process-level fields:

| Field | Meaning |
| ----- | ------- |
| `platform_supported` | Whether the build target is a Windows target.  Off Windows, every other status is `UnsupportedPlatform`. |
| `process_cfg_status` | `GetProcessMitigationPolicy(ProcessControlFlowGuardPolicy)` result encoded as `STATOR_MITIGATION_STATUS_*`. |
| `process_cet_shadow_stack_status` | `ProcessUserShadowStackPolicy`'s `EnableUserShadowStack` bit. |
| `process_cet_user_shadow_stack_strict_status` | `ProcessUserShadowStackPolicy`'s strict-mode bit (only `Enabled` when both the base policy and strict mode are on). |

The `STATOR_MITIGATION_STATUS_*` encoding (stable across the ABI
minor):

| Value | Constant | Meaning |
| ----- | -------- | ------- |
| 0 | `STATOR_MITIGATION_STATUS_UNSUPPORTED_PLATFORM` | Non-Windows build; mitigation does not apply. |
| 1 | `STATOR_MITIGATION_STATUS_DISABLED` | OS confirms mitigation is off for this process. |
| 2 | `STATOR_MITIGATION_STATUS_ENABLED` | OS confirms mitigation is on for this process. |
| 3 | `STATOR_MITIGATION_STATUS_UNKNOWN` | Probe failed (e.g. pre-Win8 host or `GetProcessMitigationPolicy` error). |

### Rust API

```rust
use stator_jse::jit_mitigations::{self, JitMitigationsTier, MitigationStatus};

let snapshot = jit_mitigations::snapshot();
assert!(!snapshot.tier(JitMitigationsTier::Baseline).cfg_supported);
let unsupported = snapshot.total_cfg_unsupported_tier_attempts();
let cfg_on = matches!(snapshot.process_cfg_status, MitigationStatus::Enabled);
```

### C FFI

```c
StatorJitMitigationsStats stats = {0};
stator_isolate_reset_jit_mitigations_stats(NULL);
/* ... run workload ... */
stator_isolate_get_jit_mitigations_stats(NULL, &stats);
if (!stats.platform_supported) {
    /* Non-Windows build — CFG/CET do not apply. */
}
if (stats.process_cfg_status == STATOR_MITIGATION_STATUS_ENABLED &&
    !stats.tiers[/*Baseline=*/0].cfg_supported) {
    /* OS has CFG on but Stator's baseline tier does not yet register
       its call targets — flag this in Edge crash diagnostics. */
}
```

Both FFI calls accept a null `StatorIsolate*` because the counters are
process-global.  Adding the symbol surface bumps the ABI minor version
to 19 (additive, backwards-compatible).

### Edge usage notes

* Pair this snapshot with `stator_isolate_get_jit_unwind_stats` and
  `stator_isolate_get_jit_memory_stats` to build a complete "is the
  JIT compatible with Chromium/Windows exploit mitigations?" report.
* A `process_cfg_status == Enabled` plus a tier with
  `cfg_supported == false` is *not* a regression; it is the documented
  state until a tier emits a CFG target list.  Edge crash triage
  should treat it as "Stator JIT runs without CFG coverage" rather
  than "CFG broke".
* A monotonically growing `cfg_register_failures` while
  `cfg_supported == true` for some tier indicates the OS is rejecting
  registrations — escalate as an ABI/OS bug.
* `cet_pages_marked_incompatible` will be non-zero on every workload
  that hits the JIT today; that is expected until a tier emits
  `ENDBR64` prologues and audited returns.

## Win64 JIT unwind registration counters

### Scope

Edge crash reporting, ETW profiling, and the Visual Studio debugger walk
Win64 stacks through the OS structured-exception-handling table.  JIT
code that is not covered by a registered `RUNTIME_FUNCTION` array is
invisible to that machinery: a sampling profiler attempting to climb out
of a JIT frame will either truncate or, worse, produce a bogus walk.

The `stator_jse::jit_unwind` module exposes the registration plumbing
(`RtlAddFunctionTable` / `RtlDeleteFunctionTable` on Windows, no-op
stubs elsewhere) plus a release-safe per-tier counter snapshot that
reports, fail-closed, which tiers actually emit unwind metadata today.

### Current tier support

| Tier      | Emits Win64 unwind records | Notes |
| --------- | -------------------------- | ----- |
| Baseline  | **no**                     | Handcrafted `masm_x64` prologue/epilogue. |
| Maglev    | **no**                     | Same in-tree assembler. |
| Turbofan  | **no**                     | Cranelift can emit `.pdata`, but the in-process pipeline does not yet thread it through. |
| Cranelift | **no**                     | Standalone Cranelift functions share executable memory but are not yet registered. |

[`jit_unwind::register_for_tier`] therefore always returns
`UnwindError::UnsupportedTier` (or `UnsupportedPlatform` off Windows) and
bumps the `unsupported_tier_attempts` counter rather than fabricating
bogus unwind info.  When a tier learns to emit `.pdata`, flip
`JitTier::emits_unwind_info` for that arm and route the registration
through `jit_unwind::register_runtime_functions` with caller-owned
records.  The returned `JitUnwindRegistration` is RAII: drop
deregisters via `RtlDeleteFunctionTable` with the same pointer that was
registered.

### Counters

Per tier (Baseline / Maglev / Turbofan / Cranelift):

| Field | Meaning |
| ----- | ------- |
| `unwind_supported` | Mirrors `JitTier::emits_unwind_info` for the tier. Currently always `false`. |
| `register_attempts` | Every call into `register_for_tier` / `register_runtime_functions`. |
| `register_successes` | Calls that returned a live `JitUnwindRegistration`. |
| `register_failures` | Empty region / OS-rejected outcomes after the supported-tier check passed. |
| `unsupported_tier_attempts` | Fail-closed sentinel — the platform or tier had no real registration backend. |
| `deregistrations` | `JitUnwindRegistration` handles dropped (table removed from the OS). |

Aggregate flag `platform_supported` reports whether this build is
`x86_64-pc-windows-*` and therefore *could* register unwind tables at
all.

### Rust API

```rust
use stator_jse::jit_unwind::{self, JitTier};

let snapshot = jit_unwind::snapshot();
assert!(!snapshot.tier(JitTier::Baseline).unwind_supported);
let unsupported = snapshot.total_unsupported_tier_attempts();
let live = snapshot.total_currently_registered();
```

### C FFI

```c
StatorJitUnwindStats stats = {0};
stator_isolate_reset_jit_unwind_stats(NULL);
/* ... run workload ... */
stator_isolate_get_jit_unwind_stats(NULL, &stats);
if (!stats.platform_supported) {
    /* Non-Windows or non-x86_64 build — Win64 unwind not applicable. */
}
if (!stats.tiers[/*Baseline=*/0].unwind_supported) {
    /* Edge profilers cannot unwind through Baseline JIT frames yet. */
}
```

Both FFI calls accept a null `StatorIsolate*` because the counters are
process-global.  Adding the symbol surface bumps the ABI minor version
to 14 (additive, backwards-compatible).

### Edge usage notes

When Edge crash reports show truncated stacks that bottom out in a JIT
code region, query this snapshot at the time of the crash:

* If `platform_supported == false`, this is a non-Windows build — Win64
  unwind is structurally inapplicable.
* If `unwind_supported == false` for the tier hosting the truncated
  frame, the truncation is expected; treat it as "no unwind info
  emitted" rather than a regression.
* If `register_failures` is non-zero, the OS rejected a tier that
  *does* emit unwind records — escalate as an OS/ABI bug.
* `currently_registered` (= `register_successes - deregistrations`)
  should stay bounded by the live JIT code-cache size; an ever-growing
  gap indicates a teardown leak.

## Related counters

- **`stator_jse::compiler::compile_counters`** — per-tier
  attempt/success/failure counters that also cover the source →
  bytecode interpreter tier. Useful for the "did the compile ever
  run?" question; the tier-latency counters answer the "how long did
  the promotion take?" question.
- **`stator_isolate_get_tiering_stats`** — per-isolate-snapshot of code
  sizes, Maglev deopt categories, stub call counts, and other
  steady-state JIT diagnostics.
- **`stator_jse::gc::pause_metrics`** — GC pause-time histogram. Same
  always-on / release-safe instrumentation philosophy.

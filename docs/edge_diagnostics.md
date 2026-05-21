# Edge performance diagnostics

This document describes the always-on, release-safe diagnostic counters
that Stator exposes for Edge performance proof runs. All counters
described below are intended to be enabled in release builds: each
update is a handful of relaxed atomic operations and (where timing is
involved) a single `Instant::now()` pair per public entry point.

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

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

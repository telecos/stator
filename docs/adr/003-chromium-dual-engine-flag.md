# ADR 003 — Chromium Dual-Engine Flag: `--enable-features=StatorEngine`

| Field    | Value                |
|----------|----------------------|
| Status   | Accepted             |
| Date     | 2026-03-04           |
| Deciders | Stator core team     |

---

## Context

Stator is designed to be a drop-in JavaScript engine replacement inside
Chromium renderer processes.  The existing GN build arg `js_engine = "stator"`
(introduced with the `gin/BUILD.gn` integration work) switches the engine at
**compile time**.  This means that any testing or gradual rollout of Stator
requires separate build artefacts, which is impractical in a production browser.

A production-ready rollout strategy requires:

1. **A runtime feature flag** so that the same browser binary can switch
   between V8 and Stator without a rebuild.
2. **A phased rollout order** that minimises user-visible risk by enabling the
   new engine for low-risk contexts first (workers) before rolling out to the
   highest-risk context (main-frame renderer with full DOM access).
3. **A differential-testing mechanism** so that V8 and Stator can run
   side-by-side on the same page and their outputs can be compared
   automatically, enabling regression detection before Stator is the sole
   engine in any context type.

---

## Decision

Implement three components:

### 1 — Chrome Feature Flags (`chromium/features/`)

Introduce five `base::Feature` entries (all **disabled by default**):

| Feature name                 | Description                                            |
|------------------------------|--------------------------------------------------------|
| `StatorEngine`               | Umbrella flag — enables Stator for all context types   |
| `StatorEngineWorkers`        | Stator for dedicated Web Workers only                  |
| `StatorEngineServiceWorkers` | Stator for Service Workers only                        |
| `StatorEngineMainFrame`      | Stator for main-frame renderer contexts                |
| `StatorDualEngine`           | Differential-testing mode (2× execution, compare DOM)  |

The umbrella flag `StatorEngine` maps to the documented user-facing switch:

```
chrome --enable-features=StatorEngine
```

The sub-flags allow field experiments to enable only a subset of context types,
independent of each other.  For example, a Finch experiment could enable
`StatorEngineWorkers` for 1 % of canary users to collect stability data before
rolling out to service workers or main frames.

### 2 — Engine Selector (`chromium/content/stator_engine_selector.h`)

A single function `content::IsStatorEnabledFor(RendererType)` encapsulates
the flag-check logic so that every renderer-context creation point in the
content layer can call one function instead of duplicating feature-list checks.

Phased rollout order (lowest risk → highest risk):

| Phase | Context type      | Feature flag                   | Rationale                                 |
|-------|-------------------|--------------------------------|-------------------------------------------|
| 1     | Dedicated workers | `StatorEngineWorkers`          | No DOM access; isolated V8 context        |
| 2     | Service workers   | `StatorEngineServiceWorkers`   | No DOM; controls network interception     |
| 3     | Main frame        | `StatorEngineMainFrame`        | Full DOM; highest-risk for regressions    |

Workers are switched first because:

* They have no access to the DOM, so any engine discrepancy cannot affect
  page rendering or user-visible state.
* Worker scripts are typically authored for performance (heavy computation),
  so Stator's JIT tiers are exercised early under realistic load.
* Worker crashes are isolated and do not take down the renderer process.

Service workers come second because they intercept network requests (a
correctness-critical path) but still have no DOM access.

Main-frame renderer contexts come last because a bug in Stator there would
directly break page rendering for the user.

### 3 — Differential-Testing Harness (`chromium/content/stator_dual_engine.h`)

When `StatorDualEngine` is enabled, both V8 and Stator execute each script
evaluation in a renderer context.  Only the V8 execution's side-effects are
applied to the real DOM; the Stator execution runs on an isolated snapshot.
After both complete, the harness compares the enumerable own string properties
of the global object and reports any discrepancy.

The same component also exposes a benchmark mode for Chromium integration
experiments. Chromium supplies a primary-engine runner that prepares and runs a
V8 snippet; the harness runs the matching Stator shadow snippet, reports median,
mean, and minimum timings for both engines, and emits JSON rows compatible with
automation and DevTools logging.

Discrepancies are reported via a new DevTools Protocol event:

```json
{
  "method": "Stator.dualEngineDiscrepancy",
  "params": {
    "statesMatch": false,
    "discrepancy": "property \"counter\": V8=42 Stator=41",
    "v8State":     { "counter": 42, "result": "\"ok\"" },
    "statorState": { "counter": 41, "result": "\"ok\"" }
  }
}
```

This enables automated regression detection in Chrome canary/dev channel
without shipping Stator to production users.

---

## GN Integration

The three components integrate into the existing Chromium build graph as
follows:

```
//third_party/stator/chromium/features:stator_features
    ↑
//third_party/stator/chromium/content:stator_engine_selector
//third_party/stator/chromium/content:stator_dual_engine
    ↑
//third_party/stator/chromium/gin:gin   (always depends on stator_features)
```

The `gin` source_set already depends on `stator_features` so that feature-flag
checks compile in for both `js_engine = "v8"` and `js_engine = "stator"`
builds.  This is intentional: the runtime flag and the build-time arg are
independent mechanisms and both must be available at the same time.

---

## Alternatives Considered

### A — Build-time arg only (`js_engine = "stator"`)

**Rejected** as the sole mechanism because it requires separate browser builds
for each engine configuration, which is incompatible with A/B field experiments
and phased production rollout.

### B — Single monolithic flag (`StatorEngine` only, no sub-flags)

**Rejected** because it forces an all-or-nothing rollout.  Historical engine
migration experience (e.g. V8's migration from the full-codegen compiler to
Ignition) shows that sub-flag granularity is essential to isolate regressions
by context type.

### C — Differential testing via a separate binary / test runner

**Rejected** as insufficient for production regression detection.  Running a
separate binary cannot replicate the exact sequence of script evaluations that
a real browser page produces.  In-process `StatorDualEngine` mode is the only
way to compare the two engines under identical conditions.

---

## Consequences

* **Positive:** The same binary can participate in a Finch field experiment that
  gradually rolls out Stator — first to workers (Phase 1), then service workers
  (Phase 2), then main frame (Phase 3) — without any recompilation.
* **Positive:** `StatorDualEngine` mode provides continuous regression
  detection in canary/dev channel at the cost of 2× script execution time.
* **Positive:** `IsStatorEnabledFor()` is a single call site for all engine
  selection logic; future context types (shared workers, worklets, etc.) can be
  added by extending the `RendererType` enum without touching call sites.
* **Negative:** The `StatorDualEngine` harness's global-property comparison is
  limited to enumerable own string properties.  Full DOM diffing would require
  a more sophisticated serialisation layer (e.g. a Mutable Document Source
  comparison).  This is accepted as a first-iteration trade-off.
* **Negative:** Non-deterministic scripts (e.g. those calling `Date.now()` or
  `Math.random()`) will always produce discrepancies in `StatorDualEngine`
  mode.  Callers are responsible for filtering expected discrepancies.
* **Neutral:** The `js_engine` GN arg and the runtime feature flags are
  independent.  When `js_engine = "stator"` the engine is Stator at compile
  time; when `js_engine = "v8"` the runtime flag can still select Stator per
  context type.  This dual-mode setup is intentional and required for
  incremental integration.

---

## Implementation Files

| File                                                 | Purpose                                         |
|------------------------------------------------------|-------------------------------------------------|
| `chromium/features/stator_features.h`                | `BASE_DECLARE_FEATURE` for all five flags       |
| `chromium/features/stator_features.cc`               | `BASE_FEATURE` definitions (disabled by default)|
| `chromium/features/BUILD.gn`                         | GN `source_set("stator_features")`              |
| `chromium/content/stator_engine_selector.h`          | `RendererType` enum + `IsStatorEnabledFor()`    |
| `chromium/content/stator_engine_selector.cc`         | Phased rollout flag-check logic                 |
| `chromium/content/stator_dual_engine.h`              | `DualEngineResult`, `RunDualEngine()`, and dual-engine benchmark API |
| `chromium/content/stator_dual_engine.cc`             | Differential-testing and benchmark harness implementation |
| `chromium/content/BUILD.gn`                          | GN source_sets for selector + dual-engine       |
| `chromium/gin/BUILD.gn`                              | Updated to depend on `stator_features`          |

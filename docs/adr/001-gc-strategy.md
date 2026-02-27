# ADR 001 — GC Backend Strategy: MMTk vs Custom

| Field    | Value                |
|----------|----------------------|
| Status   | Accepted             |
| Date     | 2026-02-27           |
| Deciders | Stator core team     |

---

## Context

Stator is an experimental JavaScript engine written in Rust, designed to be
embedded in browser-like environments (e.g., Chromium's content layer) through
a stable C FFI.  A correct and efficient garbage collector is a hard
requirement for any production-grade JavaScript engine.

At the time this decision was made the repository already contains a
hand-written GC skeleton (`crates/stator_core/src/gc/`):

* **`heap.rs`** — two-generation heap (8 MiB nursery + 64 MiB tenured space)
  backed by bump-pointer allocation.
* **`trace.rs`** — `Trace` trait and `Tracer` mark-stack for future
  mark-and-sweep traversal.
* **`handle.rs`** — V8-style `HandleScope` / `Local<'scope, T>` for
  compile-time-safe roots.
* **`objects/tagged.rs`** — NaN-boxing-style `TaggedValue` (Smi + heap
  pointer) aligned with V8's tagging scheme.

The team evaluated two paths forward from this starting point.

---

## Options Considered

### Option A — MMTk (`mmtk-core`)

[MMTk](https://www.mmtk.io/) is a production-grade, language-agnostic memory
management toolkit written in Rust.  It ships multiple GC plans
(`MarkSweep`, `SemiSpace`, `GenCopy`, `Immix`, …) behind a stable Rust API and
is already used by OpenJDK, CRuby, Julia, and several other VMs.

**Pros**

* Battle-tested GC algorithms available out of the box; no need to implement
  mark-and-sweep, Immix, or generational GC from scratch.
* `mmtk-core` supports concurrent and parallel collection, which is
  necessary for production throughput targets.
* A growing ecosystem of language bindings and documented integration patterns.
* The Rust API is stable enough that upstream regressions are rare and quickly
  fixed.

**Cons**

* **Integration cost is high.** Adopting MMTk requires implementing the
  `ObjectModel`, `ReferenceGlue`, `ActivePlan`, `Collection`, and scanning
  callbacks (roughly 500–1 000 lines of binding glue) before the engine can
  even allocate its first object through the framework.
* **Tightly couples the object model to MMTk's requirements.**  MMTk mandates
  specific layout guarantees (object start, forwarding-pointer storage, VO-bit
  tables) that diverge from the V8-inspired `HeapObject` / `TaggedValue` design
  already in place.
* **Large transitive dependency.**  `mmtk-core` pulls in a significant
  dependency graph (crossbeam, atomic, …) and makes cross-compilation (e.g.,
  targeting Android's Chromium build) non-trivial.
* **Runtime configuration overhead.**  MMTk is initialised as a global
  singleton; reconciling this with Stator's per-`Isolate` heap model requires
  careful design.
* Not yet stable on 32-bit targets.

### Option B — Custom GC (current trajectory)

Continue evolving the hand-written GC skeleton already present in the
repository.

**Pros**

* **No integration friction.**  The existing `MemoryRegion` / `HeapObject` /
  `HandleScope` / `Trace` abstractions are already aligned with the V8-like
  object model (tagged pointers, handle scopes, nursery promotion).  The
  full generational copying GC can be built incrementally on top.
* **Full control over object layout.**  Stator can adopt the exact same
  pointer-tagging and in-object metadata layout as V8/SpiderMonkey, which
  simplifies future JIT-compiler work (hidden classes, inline caches, etc.).
* **Minimal dependencies.**  The GC stays inside `stator_core` with no
  additional crates, keeping the dependency graph small and cross-compilation
  straightforward.
* **Incremental risk.**  Each GC phase (minor collection, major
  mark-and-sweep, write barriers, card tables, concurrent marking) can be
  added and tested independently without changing external API contracts.
* **Engine-lifecycle alignment.**  Per-isolate heap creation / destruction maps
  directly to `Heap::new()` / `Drop`, with no global initialisation required.

**Cons**

* **High implementation effort.**  A production-quality generational GC with
  concurrent marking, write barriers, and weak references represents a
  significant engineering investment (months, not weeks).
* **Risk of correctness bugs.**  Subtle GC bugs (premature collection,
  mis-scanning, missed write barriers) are hard to find and can cause
  non-deterministic crashes in production.
* **No free algorithmic improvements.**  Each major GC technique
  (incremental marking, compaction, concurrent sweeping) must be designed,
  implemented, and tuned in-house.

---

## Decision

**Option B — Custom GC** is adopted for the initial implementation.

The primary drivers are:

1. **Object model compatibility.**  The V8-inspired `TaggedValue`, `HeapObject`,
   and `HandleScope` design is already deeply embedded in the early codebase.
   Retrofitting MMTk's object-model requirements on top of this would require
   either discarding the existing skeleton or maintaining a translation layer —
   both of which add more complexity than they remove at this stage.

2. **Complexity-vs-benefit ratio.**  MMTk integration has a large fixed cost
   (the binding layer) that is only justified when the engine reaches a stage
   where GC throughput is a measurable bottleneck.  At the current experimental
   stage the custom path delivers more value per engineering hour.

3. **Dependency discipline.**  Stator's embedding story (stable C ABI,
   Chromium integration) benefits from a minimal dependency footprint.  Adding
   `mmtk-core` and its transitive dependencies at this stage would complicate
   the build for all downstream embedders.

4. **Incremental delivery.**  The custom GC can ship a correct minor collector
   quickly (completing the existing `heap.rs` skeleton) and then evolve toward
   a concurrent Immix-style collector over successive milestones.

### MMTk Re-evaluation Trigger

This decision should be re-evaluated if **any** of the following conditions
arise:

* GC throughput or pause-time benchmarks show that the custom collector cannot
  meet the engine's performance targets after reasonable tuning effort.
* The team decides to invest in a parallel / concurrent collector and
  estimates that implementing one from scratch would take longer than writing
  MMTk bindings.
* MMTk ships a stable, documented binding API that reduces integration cost to
  < 2 engineer-weeks.

---

## Implementation Roadmap (Custom GC)

The following milestones complete the custom GC from the current skeleton:

| Milestone | Deliverable |
|-----------|-------------|
| M1 | Complete minor (Cheney semi-space) collector: copying evacuation, forwarding pointers, root scanning via `HandleScope` |
| M2 | Write-barrier infrastructure: card table or remembered set for old→young pointers |
| M3 | Major (mark-and-sweep) collector for old space: tri-colour marking, sweep, free lists |
| M4 | Weak references and finalisation callbacks |
| M5 | Incremental / concurrent marking to reduce STW pauses |
| M6 | Benchmark against V8 and SpiderMonkey on Octane / Speedometer; revisit MMTk if targets are not met |

---

## Consequences

* **Positive:** The engine can ship a working minor GC quickly, unblocking
  parser and VM work without waiting for a full MMTk integration.
* **Positive:** The object model and GC remain under full team control,
  enabling future JIT-compiler optimisations (pointer compression, hidden
  classes) without external constraints.
* **Negative:** The team accepts responsibility for long-term GC correctness
  and performance; a dedicated GC engineer (or significant reviewer bandwidth)
  will be needed as the engine matures.
* **Neutral:** The MMTk option remains viable as a future migration path;
  the `Trace` trait and `HandleScope` abstractions are designed to remain
  compatible with an eventual MMTk binding layer if the re-evaluation trigger
  conditions are met.

# Stator release/gap matrix

Last updated: 2026-06-08.

This matrix is the checked-in release-readiness record for the Stator surfaces
that currently block an Edge/Stator release candidate. It intentionally points
to existing contracts and source evidence instead of duplicating every detail.
A release candidate is not ready until every row is either marked release-ready
or has a documented, approved exception.

## Status legend

| Status | Meaning |
|---|---|
| Release-ready | Implemented and covered by an explicit release gate. |
| Partial | Implementation exists, but compatibility, coverage, or release validation is incomplete. |
| Blocked | Missing runtime capability or fail-closed placeholder remains. |

## Matrix

| Area | Status | Current state | Remaining blockers | Release gate / exit criteria | Source evidence |
|---|---|---|---|---|---|
| FFI identity and handle lifetime | Partial | The C ABI has explicit version markers and ABI-symbol tests; module namespace objects are cached to preserve ES module namespace identity. Persistent, weak, and traced handles are implemented in the FFI isolate state, but the browser-grade handle taxonomy still documents identity invariants and moving-GC requirements as the Edge contract. | Prove stable handle-slot identity across GC compaction and slot reuse; enforce isolate/thread affinity and generation checks on all long-lived handles; verify weak-callback ordering and traced-reference visitor phases with Edge/Blink-style wrapper tests; audit `StatorValue`/`StatorObject` wrappers for pointer aliasing and stale-handle failure modes. | `cargo test -p stator_jse_ffi --test abi_contract`; generated `crates\stator_ffi\include\stator.h` must be clean after build; add release smoke covering persistent, weak, traced, DOM wrapper, and module-namespace identity through the FFI path. | `crates\stator_ffi\README.md`, `docs\handles.md`, `crates\stator_ffi\src\lib.rs` |
| Wasm imports, memory, and table | Blocked | Wasmtime-backed module support can compile/instantiate modules and the FFI module graph records function imports, primitive globals, and shared-memory imports. The JavaScript `WebAssembly` namespace remains function-export focused: `importObject` is ignored, `Memory`/`Table`/`Global` constructors fail closed, and non-function exports are rejected. Tables still have no safe cross-store bridge. | Implement JS import-object binding for functions/globals/memory/table; expose `WebAssembly.Memory` through an ArrayBuffer-compatible bridge; expose table/reference objects and table imports/exports; define non-shared memory, memory64, and cross-store ownership semantics; add conformance tests for JS API and FFI module-graph import resolution. | Wasm JS API tests must cover imports, memory growth/views, table get/set/call, global mutation, and failure cases; FFI tests must cover Wasm-to-Wasm function, global, shared-memory, and table diagnostics; unsupported shapes must continue to fail closed. | `crates\stator_jse\src\builtins\wasm.rs`, `crates\stator_jse\src\wasm\mod.rs`, `crates\stator_ffi\src\lib.rs` |
| Snapshots and code-cache | Blocked | Legacy `STSS` globals snapshots, strict serialization helpers, STSM/STWC envelopes, module code-cache metadata, canonical code-cache keys, release-manifest validation, and native-code cache header validation exist. Classic-script cache APIs are still proposed, native tier artifacts validate only, and warm-context snapshots still do not fully restore realms, intrinsics, prototypes, host callbacks, or module/wasm state. | Finish full warm-context restore with realm/intrinsics/prototype identity; make host callback/template reinstall fail closed; implement classic-script bytecode cache production/restore APIs; define accepted payload telemetry for script/module cache hits; keep native baseline/JIT cache as validate-only until a safe serializer/loader exists; integrate snapshot identity into code-cache acceptance. | `cargo test -p stator_jse_ffi --test release_manifest`; cache restore tests must exercise schema/source/origin/platform/snapshot mismatches; `scripts\prepublish_edge_gate.ps1` must report runtime payloads available only when real manifest-backed artifacts are supplied; no release may report native cache hits while `unsupported_native_code` is the only safe outcome. | `docs\snapshot.md`, `docs\code_cache.md`, `crates\stator_ffi\src\code_cache_key.rs`, `crates\stator_ffi\src\native_code_cache.rs`, `scripts\prepublish_edge_gate.ps1` |
| Inspector CDP and DevTools | Partial | `st8` can start a CDP WebSocket server and the inspector implements broad Runtime/Debugger/Console/Profiler/HeapProfiler/Target setup coverage. RemoteObject identity is per session and fail-closed on stale IDs. Several domains are minimal compatibility shims, and some debugger operations remain synthetic or fail-closed. | Validate against Chrome/Edge DevTools smoke flows; finish source-map and console integration needed by browser embedding; expose real paused call-frame/scope snapshots instead of synthetic frames where required; prove multi-context and multi-session lifecycle behavior through FFI/in-process transport; define fail-closed behavior for unimplemented CDP domains. | Add an automated CDP smoke that attaches DevTools-compatible client(s), evaluates code, inspects objects, sets breakpoints, steps, reads script source, captures console/profiler/heap output, and verifies stale object IDs fail closed. | `docs\edge_diagnostics.md`, `crates\stator_jse\src\inspector\cdp.rs`, `crates\st8\src\main.rs` |
| Windows tiering | Blocked | The executable-memory abstraction has a Windows x86-64 W^X path and smoke tests; deterministic Baseline and Maglev tier-control code is compiled for x86-64 Unix or Windows. Turbofan/Cranelift tiering is still Unix-only, and repository CI runs release tests and benchmarks on Ubuntu only. | Finish and validate Win64 JIT entry ABI, stack alignment, unwind/SEH, safepoints, deopt, termination polling, and executable-memory teardown under Windows; decide whether Turbofan/Cranelift is unsupported or ported for the release; add Windows CI for tier-control and FFI release artifacts. | CI must include Windows x86-64 `cargo test --workspace`, release build, ABI/header checks, and deterministic force-tier tests for Baseline/Maglev; Turbofan requests on Windows must have a documented release policy and tested diagnostic. | `crates\stator_jse\src\executable_memory.rs`, `crates\stator_jse\src\interpreter\mod.rs`, `.github\workflows\ci.yml` |
| Performance and diagnostics | Blocked | Microbenchmark and FFI benchmark workflows exist, and release-safe counters cover tier latency, JIT memory, deopts, ICs, and OSR. The performance analysis still identifies large gaps versus V8: 24-byte `JsValue`, clone traffic, HashMap-heavy frames/globals, O(n²) indexed insertion, stop-the-world GC, limited JIT coverage, process-global counters, no true OSR, and incomplete tier rows. | Replace microbenchmark-only confidence with workload-level Edge proof runs; establish pass/fail thresholds for latency, throughput, memory, GC pauses, deopts, IC hit rate, tier-promotion latency, and code size; add per-isolate/per-script attribution where current counters are process-global; complete JIT/OSR coverage or gate unsupported paths explicitly. | Bench workflow plus Edge perfproof must pass thresholded gates; `scripts\prepublish_edge_gate.ps1` must collect perfproof JSON for the release candidate; diagnostics schema must remain ABI-stable and privacy-reviewed. | `docs\performance_analysis.md`, `docs\edge_diagnostics.md`, `.github\workflows\bench.yml`, `scripts\prepublish_edge_gate.ps1` |
| Release gates and packaging | Partial | CI covers fmt, clippy, debug/release tests, differential tests, mini-browser, fuzz, sanitizers, Miri, Test262, coverage, and benchmarks on Linux. `scripts\prepublish_edge_gate.ps1` enforces clean `main`, mandatory Rust checks, release build, ABI/header checks, package/publish dry runs, optional Edge validation, and release metadata output. | Make Edge validation non-optional for real release candidates; require Windows tiering CI before advertising Windows artifacts; require signed/digested release manifests and real code-cache artifact availability; align README claims with actual gate results; publish a single pass/fail checklist that names every required workflow and script invocation. | No release unless `git diff --check`, mandatory Rust checks, release build, ABI/header tests, package/publish dry-runs, Test262 gate, benchmark gate, Edge conformance smoke, Edge perfproof smoke, and artifact-manifest validation all pass without skip flags. | `.github\workflows\ci.yml`, `.github\workflows\test262.yml`, `.github\workflows\bench.yml`, `scripts\prepublish_edge_gate.ps1` |

## Release-candidate validation checklist

Run these gates in order for a release candidate. Documentation-only changes may
use the narrower validation noted in the pull request, but the release candidate
itself must satisfy every item.

1. `git diff --check`
2. `cargo fmt --all`
3. `cargo clippy --workspace -- -D warnings`
4. `cargo build --workspace`
5. `cargo test --workspace`
6. `cargo build --workspace --release`
7. `cargo build -p stator_jse_ffi`
8. `git diff --exit-code -- crates\stator_ffi\include\stator.h`
9. `cargo test -p stator_jse_ffi --test abi_contract`
10. `cargo test -p stator_jse_ffi --test release_manifest`
11. `cargo run --release --bin stator_jse_test262` or the current CI Test262 gate
12. Benchmark gate from `.github\workflows\bench.yml`
13. `scripts\prepublish_edge_gate.ps1` without `-SkipMandatoryChecks` and, for a real release, without `-SkipEdgeValidation`
14. Windows x86-64 tiering/FFI CI once Windows artifacts are in scope

## Update rules

- Update this matrix in the same change that closes or adds a release blocker.
- Do not downgrade a blocker to release-ready unless the release gate is checked
  in or named above.
- If a surface intentionally remains unsupported for a release, document the
  fail-closed diagnostic and the release exception in that row.

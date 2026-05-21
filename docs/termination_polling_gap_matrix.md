# Termination polling gap matrix

This matrix records where `stator_isolate_terminate_execution` is observed during mid-execution work. "Bound" is the expected polling interval when execution is already inside that surface; "risk" names the remaining unbounded area.

| Surface / file | Function or path | Polling exists? | Bound / risk |
|---|---|---:|---|
| FFI entry point (`crates\stator_ffi\src\lib.rs`) | `stator_isolate_terminate_execution` / cancel / query | Yes | Atomic flag set is immediate and sticky until cancelled; run entry points reject new scripts while set. |
| FFI script/module execution (`crates\stator_ffi\src\lib.rs`) | `run_script_inner`, `run_script_no_result_inner`, module evaluation | Yes | Publishes the isolate flag to the interpreter thread for the duration of the run and checks before entry. |
| Interpreter (`crates\stator_jse\src\interpreter\dispatch.rs`, `mod.rs`) | `JumpLoop`, `Interpreter::run_inner`, SMI fast loop | Yes | Backward branches and function-call boundaries; worst case is one bytecode loop iteration or one interpreted call. |
| Maglev JIT (`crates\stator_jse\src\compiler\maglev\codegen.rs`) | `emit_loop_header_termination_poll` | Yes | Poll at each generated loop header via `stator_jit_poll_terminated`; deopts to interpreter termination handling. |
| Baseline JIT (`crates\stator_jse\src\compiler\baseline\compiler.rs`) | Generated loop bodies | No | Only observes termination after returning/deopting to interpreter. Infinite baseline-resident loops remain unbounded. |
| Turbofan / Cranelift (`crates\stator_jse\src\compiler\turbofan\mod.rs`) | Generated loop bodies | No | No Cranelift-side interrupt check. Infinite Turbofan-resident loops remain unbounded. |
| Wasm (`crates\stator_jse\src\wasm\mod.rs`) | JS→Wasm entry, host callbacks, epoch interruption | Yes | Entry/callback checks plus process-wide Wasmtime epoch bump, gated by the running thread's published Stator flag. |
| Promise microtasks (`crates\stator_jse\src\builtins\promise.rs`) | `MicrotaskQueue::drain` | Yes | Polls between microtasks; one currently-running microtask must return before the drain stops. |
| JSON parser/stringifier (`crates\stator_jse\src\builtins\json.rs`) | `json_parse`, `json_stringify`, `json_stringify_js_value` | Yes | Polls at value boundaries and during long strings/numbers/whitespace at 1,024-character intervals. |
| structuredClone (`crates\stator_jse\src\builtins\install_globals.rs`) | `structured_clone*` helpers | Partial | Polls at clone entry and per property/array element; large ArrayBuffer byte copies are not interruptible mid-copy. |
| RegExp wrapper (`crates\stator_jse\src\objects\regexp.rs`) | global match/replace/split/matchAll loops | Partial | Wrapper loops poll the published interrupt flag at every iteration via the `try_symbol_*` methods used by the interpreter and the `RegExp` prototype installer; the underlying `regress` engine call (single `find_from` invocation) is *not* interruptible, so a pathological pattern that backtracks inside one call remains unbounded until that call returns. Tracked by todo `stator-engine-regress-uninterruptible-backtracking`. |

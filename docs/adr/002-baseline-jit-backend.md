# ADR 002 — Baseline JIT Backend: Custom MacroAssembler vs Cranelift SimpleJIT

| Field    | Value                |
|----------|----------------------|
| Status   | Accepted             |
| Date     | 2026-03-03           |
| Deciders | Stator core team     |

---

## Context

Stator currently executes JavaScript through a fetch-decode-dispatch interpreter
(`crates/stator_jse/src/interpreter/mod.rs`).  The interpreter is correct and
complete but pays a per-instruction overhead for opcode dispatch, register-file
indirection, and the Rust function-call frame around every bytecode handler.

The next performance tier — as seen in V8's **Sparkplug** compiler — is a
*baseline JIT*: a compiler that makes a single, linear pass over a
[`BytecodeArray`][bca] and emits native machine code with a **1:1 mapping**
from bytecode instruction to machine-code sequence.  There is no intermediate
representation, no register allocation beyond the simple "each virtual register
maps to a fixed stack slot" model, and no optimisation.  The sole goal is to
eliminate interpreter dispatch overhead while reusing all existing
runtime-support functions (type coercion, property IC stubs, etc.) through
direct `call` instructions.

The Stator bytecode format is well-suited to this approach:

* The [`Opcode`][op] enum covers a rich but finite instruction set with
  well-defined operand signatures (`operand_types()`).
* [`BytecodeArray`][bca] exposes a decoded `Vec<Instruction>` through
  `instructions()`.
* The inline-cache (`crates/stator_jse/src/ic/`) and feedback-vector
  infrastructure is already in place to back IC stubs at JIT call sites.
* The GC `HandleScope` and `TaggedValue` tagging scheme (ADR 001) provides the
  object-model assumptions the JIT must respect.

At the time this decision was made two backend paths were evaluated.

[bca]: ../../crates/stator_jse/src/bytecode/bytecode_array.rs
[op]:  ../../crates/stator_jse/src/bytecode/bytecodes.rs

---

## Options Considered

### Option A — Custom MacroAssembler

Implement a thin `MacroAssembler` abstraction in Rust (analogous to V8's or
JavaScriptCore's) that wraps a `Vec<u8>` code buffer and exposes typed
instruction-emit methods (`mov_reg_imm64`, `call_rel32`, `jmp_short`, etc.).
The JIT compiler struct walks the decoded `Vec<Instruction>` and calls the
appropriate assembler method(s) for each opcode.  The resulting code buffer is
mmap'd as RWX (or RX after sealing), and a function pointer is stored in a
`CompiledCode` wrapper attached to the `BytecodeArray`.

**Pros**

* **Zero new dependencies.**  The entire assembler lives inside `stator_jse`
  and adds no entries to `Cargo.toml`, preserving the minimal-dependency story
  that guided ADR 001.
* **Full architecture control.**  The team chooses exactly which native calling
  convention to use (e.g. a custom `JitFrame` matching the interpreter's
  `InterpreterFrame` layout), which platform registers hold the accumulator and
  frame pointer, and when to flush the instruction cache.
* **Incremental risk.**  A minimal viable assembler covering only the handful of
  opcodes exercised by the interpreter's hot paths (load/store, arithmetic,
  conditional jumps, calls) can be shipped first.  Coverage grows opcode-by-
  opcode, each addition being independently testable.
* **No IR translation overhead.**  Because the assembler works directly from
  decoded `Instruction` objects (already available via `BytecodeArray::
  instructions()`), the compilation pipeline adds no parse or lowering step.
* **Cross-compilation friendly.**  The assembler can target multiple
  architectures (x86-64, aarch64) by switching enum arms; no build-system
  complexity is introduced.

**Cons**

* **Non-trivial engineering effort.**  A correct, portable assembler that
  handles all Stator opcodes — including exception-handler patching, IC stubs,
  and safepoints for GC — is hundreds to thousands of lines of low-level Rust.
* **Correctness risk.**  Encoding bugs (wrong ModRM byte, incorrect REX prefix,
  off-by-one in jump offsets) produce silent crashes rather than compile errors.
  Thorough disassembler-based tests are mandatory.
* **Platform-specific code paths.**  x86-64 and AArch64 instruction encoding
  differ substantially; `#[cfg(target_arch)]` branches must be maintained.

### Option B — Cranelift `SimpleJIT`

Use [Cranelift](https://cranelift.dev/) (the code-generation backend powering
Wasmtime and rustc's experimental backend) through its high-level
`cranelift-jit` crate.  Each Stator bytecode function is translated into
Cranelift IR (`cranelift_codegen::ir`), and `SimpleJIT::finalize_function`
produces and loads native code.

**Pros**

* **Production-quality code generation out of the box.**  Cranelift handles
  register allocation, instruction selection, stack-frame layout, and calling-
  convention ABI for x86-64, AArch64, s390x, and RISC-V; the team does not
  write a single architecture-specific byte.
* **Correctness guarantees.**  Cranelift is extensively fuzz-tested and used in
  production by Wasmtime; the probability of silent instruction-encoding bugs is
  far lower than for a hand-written assembler.
* **Future optimisation pathway.**  Adding basic-block passes (constant
  folding, dead-code elimination) is straightforward once the IR translation
  layer exists.
* **Safepoints and stack maps.**  Cranelift's emerging GC support can produce
  stack maps for precise GC root scanning, which will be required once the GC
  matures past the skeleton stage.

**Cons**

* **Significant dependency footprint.**  `cranelift-jit` and its transitive
  dependencies (`cranelift-codegen`, `cranelift-frontend`, `cranelift-module`,
  `regalloc2`, …) add roughly 150 000 lines of external Rust to the build.
  This complicates cross-compilation (especially to Android/Chromium targets)
  and increases binary size.
* **Cranelift IR is not a 1:1 bytecode mapping.**  Each Stator opcode must be
  lowered into multiple Cranelift IR instructions via a `FunctionBuilder`.  This
  adds an IR construction layer, a register-allocator run, and an instruction-
  selection pass on every compilation — all overhead the baseline tier tries to
  avoid.
* **Tighter coupling to Cranelift's object model.**  Data types, calling
  conventions, and memory layouts must be expressed in Cranelift's type system
  (`I64`, `R64`, `F64`, `Ref`).  Adapting Stator's `JsValue` NaN-boxing scheme
  requires non-trivial mapping work and careful use of `cranelift_codegen::ir::
  Type::int(64)` for opaque tagged-value words.
* **Slower compilation latency.**  Even SimpleJIT's unoptimised tier runs a
  full linear-scan register allocator; for the short functions typical of JS
  baselines the regalloc overhead is measurable.
* **API churn.**  Cranelift's public API changed several times between 0.90 and
  0.110; pinning a version risks missing security or correctness fixes, while
  unpinning risks API breakage.

---

## Decision

**Option A — Custom MacroAssembler** is adopted for the initial baseline JIT
implementation.

The primary drivers are:

1. **Dependency discipline.**  As established in ADR 001, Stator targets a
   minimal dependency footprint for embedding in Chromium and similar
   environments.  Adding Cranelift's ~150 KLOC transitive graph at this stage
   is disproportionate to the benefit, especially when the baseline tier
   intentionally avoids optimisation.

2. **True 1:1 mapping.**  The issue specification requires a *Sparkplug-
   equivalent* design: one bytecode instruction → one fixed machine-code
   sequence, with no IR, no register allocator, and no optimisation passes.
   A custom assembler implements this directly; Cranelift IR translation cannot
   avoid the intermediate representation step.

3. **Object-model alignment.**  The custom assembler can be written to assume
   the V8-inspired `TaggedValue`/`JsValue` layout already in place, using the
   same calling conventions as the existing `Interpreter::run` function.  No
   type-system adapter layer is required.

4. **Incremental deliverability.**  A minimal assembler covering only the
   simplest opcodes (`LdaSmi`, `Star`, `Ldar`, `Add`, `Sub`, `Return`, and the
   most common conditional jumps) can run existing test-suite programs through
   the JIT path in a single milestone, unblocking performance measurement and
   IC integration work without waiting for full opcode coverage.

### Cranelift Re-evaluation Trigger

This decision should be revisited if **any** of the following conditions arise:

* The custom assembler accumulates more than ~2 000 lines of
  architecture-specific encoding logic and maintenance burden becomes
  prohibitive.
* The team decides to invest in an optimising JIT tier (equivalent to V8's
  Maglev or TurboFan) where Cranelift's IR passes, loop detection, and
  optimisation infrastructure would provide direct value.
* Cranelift ships a stable `cranelift-jit` API under a semver guarantee with
  documented integration patterns for NaN-boxing runtimes, reducing integration
  cost to < 1 engineer-week.
* Cross-compilation complexity introduced by the custom assembler's
  `#[cfg(target_arch)]` branches exceeds that of a Cranelift integration.

---

## Architecture

### Pipeline overview

```
BytecodeArray
     │
     ▼  (decode once, cache Vec<Instruction>)
  Vec<Instruction>
     │
     ▼  BaselineCompiler::compile()
  MacroAssembler  ─── emits ──▶  Vec<u8> code buffer
     │                                 │
     │  (patch jump targets)           │ mmap RX
     ▼                                 ▼
  JumpPatchList                  CompiledCode { fn_ptr, code }
```

### Register convention

| Purpose             | x86-64    | AArch64   |
|---------------------|-----------|-----------|
| Accumulator         | `rax`     | `x0`      |
| Frame pointer (JIT) | `rbp`     | `x29`     |
| Stack pointer       | `rsp`     | `sp`      |
| Scratch 1           | `r10`     | `x16`     |
| Scratch 2           | `r11`     | `x17`     |
| Callee-saved base   | `r12`     | `x19`     |

Virtual registers (`BytecodeArray::frame_size()` slots) are spilled to the
native stack at frame entry and accessed via `[rbp - 8*(reg+1)]`
(x86-64) / `[x29, #-8*(reg+1)]` (AArch64).  This matches the interpreter's
conceptual register file, making it straightforward to switch between
interpreted and compiled execution of the same `BytecodeArray`.

### Compilation pass

`BaselineCompiler::compile` performs a **two-pass** walk over the decoded
instruction list:

1. **Emit pass** — iterate instructions in order; for each opcode call the
   corresponding `MacroAssembler::emit_*` method.  Forward jump targets are
   recorded in a `JumpPatchList` with the byte offset of the 32-bit
   displacement field to be back-patched.
2. **Patch pass** — resolve all entries in the `JumpPatchList`, writing the
   correct signed 32-bit displacement into each jump instruction now that all
   label positions are known.

The compiler keeps a `bytecode_to_pc: Vec<u32>` map (instruction index →
code-buffer offset) built during the emit pass; the patch pass uses this map to
resolve labels.

### Runtime support calls

Every opcode that cannot be handled with a few inline instructions (property
loads, `instanceof`, coercions, IC stubs, etc.) emits a `call` to an existing
Rust runtime function with the `extern "C"` ABI.  This keeps the JIT code
small and reuses the interpreter's runtime helpers without duplication.

### Exception handling

The existing `HandlerTableEntry` from `BytecodeArray::handler_table()` is
preserved in the `CompiledCode` struct.  When a runtime call returns a
`StatorError`, the compiled frame's unwind code walks the handler table and
transfers control (via a `jmp`) to the handler's native PC (looked up in
`bytecode_to_pc`).

---

## Implementation Roadmap

| Milestone | Deliverable |
|-----------|-------------|
| M1 | `MacroAssembler` struct with x86-64 encoding: `mov`, `push`/`pop`, `add`/`sub`, conditional/unconditional `jmp`, `call`, `ret`. Unit tests compare emission against known hex. |
| M2 | `BaselineCompiler` translating the simplest hot opcodes: `LdaZero`, `LdaSmi`, `LdaConstant` (number/SMI fast-path), `Ldar`, `Star`, `Mov`, `Add`, `Sub`, `Mul`, `Div`, `Return`. End-to-end test: compile a simple arithmetic function and execute via fn pointer. |
| M3 | Control-flow opcodes: `Jump`, `JumpLoop`, `JumpIfTrue`, `JumpIfFalse`, `JumpIfToBooleanTrue`, `JumpIfToBooleanFalse`. Two-pass label resolution. |
| M4 | Call opcodes (`CallUndefinedReceiver0/1/2`, `CallProperty0/1/2`) via `extern "C"` runtime stubs. Inline-cache slot integration. |
| M5 | Remaining load/store opcodes (globals, context slots, named/keyed properties) via runtime stubs. |
| M6 | Exception-handler support: `Throw`, `ReThrow`, handler-table-driven unwind in compiled frames. |
| M7 | AArch64 backend: duplicate `MacroAssembler` encoding paths behind `#[cfg(target_arch = "aarch64")]`. |
| M8 | Tiering trigger: interpreter counts bytecode executions via a per-`BytecodeArray` counter; functions that exceed the tier-up threshold are compiled on the next call. |
| M9 | Benchmark against the interpreter on Octane / SunSpider micro-benchmarks; verify ≥ 2× throughput improvement on arithmetic-heavy loops. |

---

## Consequences

* **Positive:** Eliminates interpreter dispatch overhead for all JIT-compiled
  functions, improving throughput on arithmetic-intensive JavaScript.
* **Positive:** The custom assembler stays inside `stator_jse` with no
  additional Cargo dependencies, preserving the minimal-footprint embedding
  story.
* **Positive:** The 1:1 bytecode→native mapping makes deoptimisation trivial:
  execution can fall back to the interpreter at any bytecode boundary because
  the virtual register layout on the stack mirrors the interpreter's frame.
* **Negative:** The team accepts responsibility for assembler correctness and
  must maintain architecture-specific encoding paths.  A disassembler-based
  test harness (using the `capstone` crate or equivalent, added only to `dev-
  dependencies`) is strongly recommended to catch encoding regressions.
* **Negative:** Generator functions (`BytecodeArray::is_generator() == true`)
  and async functions require `SuspendGenerator`/`ResumeGenerator` support
  that is non-trivial to implement in a native frame; these may be left on the
  interpreter until M6+.
* **Neutral:** The `CompiledCode` type introduced here is designed to be
  forward-compatible with a future optimising tier (Maglev/TurboFan equivalent)
  that would replace the baseline-compiled code of hot functions.

# Stator ↔ V8 Functional Parity Report

**Date:** 2026-03-04  
**Stator commit:** 909ed95 (main)  
**Methodology:** Source analysis, `cargo test`, live `st8` execution, V8 API mapping, **Test262 conformance suite (50,734 tests)**

---

## Executive Summary

Stator is a **68,581-line Rust JavaScript engine** (61,638 in `stator_core`) with 2,053 passing tests, a 4-tier compilation pipeline (interpreter → baseline JIT → Maglev → Turbofan/Cranelift), generational GC, and a 112-function C-ABI FFI surface with a C++ `v8::` compatibility shim.

**Overall parity with V8: ~15–20% of functional behavior, ~30–35% of architectural scaffolding.**  
**Test262 pass rate: 6.58% (1,977 / 30,031 attempted; 20,703 skipped)**

The engine successfully executes basic JavaScript (arithmetic, strings, functions, closures*, if/else, while/for loops, try/catch, ternary, recursion) but has critical gaps in the parser (no array/object literals, no classes, no arrow functions, no for-in/of), ~103 unimplemented interpreter opcodes (out of ~175 real opcodes), and builtins (Math, JSON, Array, Object, etc.) that exist as Rust code but are **not wired to the runtime global environment**. The Test262 suite confirms 0% pass rate on built-in tests (builtins unreachable from JS), with strongest results in keywords (100%), punctuators (91%), block-scope (54%), identifiers (57%), and literals (53%).

---

## 1. End-to-End JavaScript Execution (st8 CLI)

### Working

| Feature | Example | Output | Status |
|---------|---------|--------|--------|
| Integer arithmetic | `print(2 + 3 * 4)` | `14` | **PASS** |
| String concatenation | `print("hello" + " " + "world")` | `hello world` | **PASS** |
| Function declaration + call | `function add(a,b){return a+b} print(add(10,20))` | `30` | **PASS** |
| Recursive functions | `function fib(n){if(n<=1) return n; return fib(n-1)+fib(n-2)} print(fib(10))` | `55` | **PASS** |
| If/else | `var x=10; if(x>5){print("big")}else{print("small")}` | `big` | **PASS** |
| While loop | `var i=0; var s=0; while(i<5){s=s+i; i=i+1} print(s)` | `10` | **PASS** |
| For loop (expr init) | `var i; for(i=0;i<10;i=i+1){...}` | `45` | **PASS** |
| Ternary operator | `var x=5; print(x>3 ? "yes" : "no")` | `yes` | **PASS** |
| Try/catch/throw | `try{throw "oops"}catch(e){print("caught: "+e)}` | `caught: oops` | **PASS** |
| console.log / print | both work | | **PASS** |

### Broken / Not Working

| Feature | Example | Result | Root Cause |
|---------|---------|--------|------------|
| Array literals | `var a = [1,2,3]` | `SyntaxError: unexpected token LeftBracket` | **Parser** doesn't parse array expressions |
| Object literals | `var o = {x:1}` | `SyntaxError: unexpected token LeftBrace` | **Parser** doesn't parse object expressions |
| Class syntax | `class Foo{...}` | `SyntaxError: unexpected token Class` | **Parser** doesn't parse class declarations |
| for(var init) | `for(var i=0;...)` | `SyntaxError: unexpected token Var` | **Parser** doesn't allow var-decl in for-init |
| typeof operator | `typeof 42` | `unimplemented opcode: TypeOf` | **Interpreter** missing opcode handler |
| Closures (variable capture) | `counter()` pattern | Returns `NaN` | **Interpreter** closure variable mutation broken |
| string.length | `"hello".length` | `undefined` | **Property access** on primitives not implemented |
| Math.max() | `Math.max(3,7)` | `TypeError: callee is not a function` | **Math** builtin not registered in global env |
| Arrow functions | `var f = (x) => x+1` | Not tested (parser lacks support) | **Parser** |
| Template literals | `` `hello ${name}` `` | Not tested | **Parser** lacks backtick support at expression level |
| Destructuring | `var {a,b} = obj` | Not supported | **Parser** |
| Spread/rest | `...args` | Not supported | **Parser** |
| for-in / for-of | `for(x in obj)` | Not supported | **Parser** |
| Modules (import/export) | `import x from 'y'` | Not supported | **Parser** |
| async/await | | `await is not yet supported` | **BytecodeGen** error |
| Generators (yield) | | Partial (in bytecode gen + interpreter) | Needs testing |

---

## 2. Subsystem Parity Assessment

### 2.1 Parser (recursive_descent.rs — 1,203 lines)

**V8 equivalent:** `src/parsing/` (~50,000+ lines)  
**Parity: ~20%**

| Feature | Status | Notes |
|---------|--------|-------|
| Numeric/string/bool/null literals | **Done** | |
| Identifiers, `this` | **Done** | |
| Binary/unary operators (all) | **Done** | Full precedence chain |
| Conditional (?:), sequence (,) | **Done** | |
| Assignment operators (=, +=, etc.) | **Done** | |
| Parenthesized expressions | **Done** | |
| var/let/const declarations | **Done** | Simple identifier bindings only |
| Function declarations + expressions | **Done** | Including generators (`*`) |
| if/else, while, do-while, for | **Done** | for-init excludes var declarations |
| return, break, continue, throw | **Done** | |
| try/catch/finally | **Done** | |
| Block statements ({}) | **Done** | |
| Empty statements (;) | **Done** | |
| ASI (automatic semicolon insertion) | **Done** | |
| Member expressions (a.b, a[b]) | **Done** | In call/member chain |
| Optional chaining (?.) | **Done** | |
| **Array literals** | **Missing** | AST type exists, bytecode gen ready |
| **Object literals** | **Missing** | AST type exists, bytecode gen ready |
| **Arrow functions** | **Missing** | AST type exists, bytecode gen ready |
| **Class declarations/expressions** | **Missing** | AST type exists |
| **Template literals** | **Partial** | AST exists, bytecode gen ready, parser missing |
| **for-in / for-of** | **Missing** | AST types exist |
| **switch** | **Missing** | AST type exists, bytecode gen ready |
| **Destructuring** | **Missing** | |
| **Spread / rest parameters** | **Missing** | |
| **Computed property names** | **Missing** | |
| **import / export** | **Missing** | AST types exist |
| **Regular expression literals** | **Missing in parser** | AST + bytecode gen ready |
| **Labeled statements** | **Missing** | |

**Key finding:** The AST has 106 node types and the bytecode generator has handlers for arrays, objects, arrows, templates, regexp, switch, and more — but the **parser is the bottleneck**, unable to produce these AST nodes from source text.

### 2.2 Scanner/Lexer (scanner.rs — 2,390 lines)

**Parity: ~70%**

Tokenizes: identifiers, keywords (var/let/const/if/else/while/for/function/return/class/etc.), numeric literals (int, float, hex, octal, binary), string literals (single/double-quoted with escapes), operators (all arithmetic/comparison/bitwise/logical/assignment), punctuation, comments (single/multi-line), template literals, regexp literals.

Missing: some edge cases in Unicode identifiers, private identifiers (#x), optional chaining vs decimal ambiguity.

### 2.3 Bytecode Compiler (bytecode_generator.rs — 3,830 lines)

**V8 equivalent:** `src/interpreter/bytecode-generator.cc` (~7,000 lines)  
**Parity: ~45%**

Compiles: literals, identifiers, binary/unary/update ops, conditionals, logical ops, assignments (including compound +=, -=, etc.), member expressions (computed + named), method calls, function calls, new expressions, function expressions, arrow expressions, generators (yield), array literals, object literals, template literals, switch statements, try/catch/finally, for/while/do-while loops, var declarations, closures (CreateClosure), context management, debugger statements.

Missing: class compilation, for-in/for-of iteration, destructuring patterns, spread args, async functions, dynamic import, labeled break/continue, with statements.

### 2.4 Bytecode Instruction Set (bytecodes.rs — 1,576 lines, 198 opcodes defined)

**V8 equivalent:** ~174 opcodes in Ignition  
**Parity: ~100% definition, ~40% execution**

All 198 opcodes are defined with proper operand types. However, only **70 opcodes are handled** in the interpreter dispatch loop. **103 opcodes** hit the fallthrough `unimplemented opcode` error at runtime.

**Implemented in interpreter (70):** LdaZero, LdaSmi, LdaUndefined, LdaNull, LdaTrue, LdaFalse, LdaConstant, Ldar, Star, Add, Sub, Mul, Div, Mod, Inc, Dec, TestEqual, TestNotEqual, TestEqualStrict, TestLessThan, TestGreaterThan, TestLessThanOrEqual, TestGreaterThanOrEqual, TestNull, TestUndefined, LogicalNot, ToBooleanLogicalNot, Jump, JumpLoop, JumpIfTrue, JumpIfFalse, JumpIfToBooleanTrue, JumpIfToBooleanFalse, JumpIfNull, JumpIfNotNull, JumpIfUndefined, JumpIfNotUndefined, JumpIfUndefinedOrNull, Return, CreateClosure, CallAnyReceiver, CallUndefinedReceiver0/1/2, CallProperty, CallWithSpread, Construct, ConstructWithSpread, PushContext, PopContext, Throw, ReThrow, SuspendGenerator, ResumeGenerator, GetGeneratorState, SetGeneratorState, SwitchOnGeneratorState, LdaGlobal, StaGlobal, LdaNamedProperty, StaNamedProperty, GetKeyedProperty, SetKeyedProperty, Debugger.

**Critical missing opcodes:** TypeOf, ToString, ToNumber, ToBoolean, ToObject, TestInstanceOf, TestIn, CreateArrayLiteral, CreateObjectLiteral, CreateRegExpLiteral, CreateEmptyObjectLiteral, CreateBlockContext, LdaContextSlot, StaContextSlot (closure variables!), StaKeyedProperty, BitwiseAnd/Or/Xor, ShiftLeft/Right, ForIn*, LdaKeyedProperty, all *Smi variants, Negate, etc.

### 2.5 Interpreter (mod.rs — 4,855 lines)

**V8 equivalent:** `src/interpreter/interpreter.cc` + handlers (~15,000+ lines)  
**Parity: ~25–30%**

Working: basic arithmetic, comparisons, control flow jumps, function calls (4 call variants), construct (new), closure creation, generator suspend/resume, global variable load/store, named property access, exception handling (throw/catch with handler table), debugger integration, JIT tiering (baseline → Maglev → Turbofan).

**Critical gap: Context slots (LdaContextSlot/StaContextSlot) are not implemented**, which is why closures that capture mutable variables return `NaN`. The interpreter uses a `HashMap<String, JsValue>` global environment model rather than V8's context chain.

### 2.6 Baseline JIT (compiler.rs — 1,829 lines, masm_x64.rs — 1,099 lines)

**V8 equivalent:** Sparkplug (~5,000 lines)  
**Parity: ~30%**

x86-64 only. Compiles bytecode to native code with:
- JIT value representation (i64 with sentinel tagging)
- Register conventions (R12=acc, R14=reg base, R11=scratch)
- Safepoint tables for GC
- Deoptimization tables for fallback to interpreter
- MacroAssembler with MOV, ADD, SUB, MUL, CMP, JMP, CALL, RET, etc.
- Handles: LdaZero, LdaSmi, LdaTrue/False/Undefined/Null, Add, Sub, Mul, Div, Mod, comparisons, jumps, Star/Ldar, Return

Missing: function calls, property access, object operations, string ops.

### 2.7 Maglev JIT (4 files, ~7,840 lines total)

**V8 equivalent:** Maglev mid-tier optimizer (~40,000+ lines)  
**Parity: ~15–20% (architectural)**

Has: IR node types (2,402 lines), graph builder from bytecode (2,371 lines), optimization passes (1,640 lines: constant folding, dead code elimination, loop-invariant code motion, range analysis, escape analysis, load elimination, strength reduction, check elimination, phi resolution), x86-64 codegen (1,427 lines), register allocator, deopt support.

Background compilation with thread safety. Full pipeline works for simple integer-heavy bytecode.

### 2.8 Turbofan/Cranelift (3 files, ~2,943 lines total)

**V8 equivalent:** Turbofan top-tier optimizer (~100,000+ lines)  
**Parity: ~5–10% (proof of concept)**

Uses Cranelift as the code generator backend. Has: type specialization (1,426 lines — speculates Smi types), deoptimization support, Cranelift CLIF IR generation. Background compilation thread.

### 2.9 Garbage Collector (6 files, 2,985 lines)

**V8 equivalent:** Oilpan/V8 GC (~50,000+ lines)  
**Parity: ~20%**

| Component | Status |
|-----------|--------|
| Generational heap (young + old) | **Done** — 4MB semi-spaces + 64MB old space |
| Bump-pointer allocation | **Done** |
| Cheney semi-space scavenger | **Done** — copies live objects, swaps spaces |
| Mark-Sweep-Compact | **Done** — mark phase, sweep, compaction |
| Write barriers | **Done** — card table, remembered set |
| Handle scopes | **Done** — nested scopes with automatic cleanup |
| Tracing (Trace trait) | **Done** — implemented for all JsValue variants |
| Large Object Space | **Done** |
| Incremental marking | **Missing** |
| Concurrent marking | **Missing** |
| Concurrent sweeping | **Missing** |
| Weak references integration | **Partial** |

### 2.10 Built-in Functions

**V8 equivalent:** `src/builtins/` (~200,000+ lines)  
**Parity: ~35% code exists, ~5% runtime-accessible**

Extensive Rust implementations exist as tested library functions, but **almost none are wired to the JavaScript global environment**:

| Builtin | Lines | Tests | Wired to runtime? |
|---------|-------|-------|-------------------|
| Array (push, pop, map, filter, reduce, splice, etc.) | 1,512 | 81 | **No** |
| String (charAt, slice, indexOf, replace, split, etc.) | 1,575 | 112 | **No** |
| JSON (parse, stringify) | 1,670 | 62 | **No** |
| Promise (resolve, reject, then, all, race, etc.) | 1,371 | 31 | **No** |
| Proxy (construct, handlers, traps) | 1,193 | 37 | **No** |
| Math (abs, floor, ceil, max, min, sin, cos, etc.) | 988 | 68 | **No** |
| Object (keys, values, entries, assign, freeze, etc.) | 822 | 41 | **No** |
| Number (parseInt, isInteger, toFixed, etc.) | 636 | 43 | **No** |
| RegExp (test, exec, match, replace, etc.) | 1,033 | 44 | **No** |
| Error (TypeError, RangeError, etc.) | 529 | 16 | **No** |
| Map | 471 | 15 | **No** |
| Set | 333 | 11 | **No** |
| WeakMap | 363 | 13 | **No** |
| WeakSet | 304 | 11 | **No** |
| Reflect | 575 | 28 | **No** |
| Iterator | 470 | 16 | **No** |
| Global (isNaN, parseInt, parseFloat, eval, etc.) | 780 | 48 | **No** |
| WebAssembly | 554 | 60 | **Yes** (st8 only) |

**Total built-in code: ~13,179 lines, 737 tests — but inaccessible from JS runtime.**

### 2.11 Object Model (objects/ — 6 files, ~4,050 lines)

**Parity: ~25%**

| Component | Status |
|-----------|--------|
| JsValue enum (17 variants) | **Done** — Undefined, Null, Boolean, Smi, HeapNumber, String, Object, Array, Function, NativeFunction, PlainObject, Generator, Iterator, Error, RegExp, Map, Set |
| Tagged pointer representation | **Done** — NaN-boxing with Smi 31-bit inline |
| HeapObject (GC-managed) | **Done** |
| Hidden class / Map (shapes) | **Partial** — 314 lines, transition tracking |
| JsObject (property storage) | **Done** — 1,074 lines, named + indexed properties |
| JsArray | **Done** — 425 lines |
| JsFunction | **Done** — 490 lines, bytecode + closure env |
| JsString | **Done** — 490 lines, rope/flat representations |
| RegExp objects | **Done** — 1,033 lines, `regress` crate backend |
| Prototype chain | **Missing** |
| Property descriptors (get/set) | **Missing** |
| Symbol type | **Missing** |
| WeakRef | **Missing** |
| FinalizationRegistry | **Missing** |
| Proxy target/handler model | **Partial** (in builtins) |

### 2.12 FFI / Chromium Integration Layer

**Parity: ~20–25% of V8 public API surface**

| Metric | Stator | V8 |
|--------|--------|-----|
| C-ABI functions | 112 | ~310 |
| C++ wrapper classes | 8 | 75+ |
| Header files | stator.h (1,597L) + v8_compat.h (513L) | ~65 headers |

**v8_compat.h classes implemented:** `v8::Isolate`, `v8::Context`, `v8::HandleScope`, `v8::Local<T>`, `v8::MaybeLocal<T>`, `v8::Value`, `v8::String`, `v8::FunctionTemplate`, `v8::Script`

**v8:: classes missing:** `v8::Object`, `v8::Array`, `v8::Number`, `v8::Integer`, `v8::Boolean`, `v8::Function`, `v8::Promise`, `v8::ArrayBuffer`, `v8::TypedArray`, `v8::Map`, `v8::Set`, `v8::Date`, `v8::RegExp`, `v8::Proxy`, `v8::Symbol`, `v8::BigInt`, `v8::Module`, `v8::SharedArrayBuffer`, `v8::DataView`, `v8::External`, `v8::ObjectTemplate`, `v8::Persistent<T>`, `v8::Global<T>`, `v8::EscapableHandleScope`, `v8::TryCatch`, `v8::Message`, `v8::StackTrace`, `v8::StackFrame`, `v8::Platform`, `v8::Inspector`, `v8::CpuProfiler`, `v8::HeapProfiler`, `v8::Snapshot`, etc.

### 2.13 Inspector / DevTools

**Parity: ~10% (proof of concept)**

- CDP protocol handler (679 lines)
- Debugger with breakpoints, stepping, pause (914 lines)  
- Heap snapshot (666 lines)
- CPU profiler (687 lines)
- Interpreter integration via thread-local debugger attachment

Missing: WebSocket transport, full CDP domain coverage, source map support, console API integration.

### 2.14 WebAssembly

**Parity: ~15%**

- Wasmtime backend (554 lines)
- Module compile, instantiate, export listing, function calls
- `WebAssembly` JS API object wired in st8
- 60 tests passing

Missing: streaming compilation, JS↔Wasm value marshaling for complex types, memory/table/global imports, WASI.

### 2.15 Inline Caches (ic/mod.rs — 951 lines)

**Parity: ~15%**

Feedback vector and IC state tracking exists. Monomorphic/polymorphic/megamorphic state transitions implemented in code. Not fully integrated with interpreter property access paths.

---

## 3. Infrastructure Assessment

| Component | Status |
|-----------|--------|
| CI (cargo fmt + clippy + test) | **Done** — 8 workflow files |
| ASAN | **Done** — asan.yml |
| TSAN | **Done** — tsan.yml |
| Miri | **Done** — miri.yml |
| Fuzz testing (24 targets) | **Done** — fuzz.yml + nightly |
| Benchmarks | **Done** — bench.yml |
| Code coverage | **Done** — coverage.yml |
| ADRs | **Done** — 2 architectural decision records |
| Copilot agent rules | **Done** — .github/copilot-instructions.md |
| Test262 harness | **Done** — full runner (1,001 lines), CI gate (test262.yml), YAML frontmatter parser, feature skip-list |

---

## 3a. Test262 Conformance Results

**Full suite:** 50,734 tests | **Attempted:** 30,031 | **Skipped:** 20,703 | **Pass:** 1,977 | **Fail:** 28,054  
**Overall pass rate: 6.58%**

### By Category (Top-Level)

| Category | Total | Attempted | Pass | Fail | Skip | Pass Rate |
|----------|-------|-----------|------|------|------|-----------|
| language/ | 23,605 | ~9,500 | ~1,970 | ~7,500 | ~14,100 | ~20.7% |
| built-ins/ | 22,980 | 16,798 | 0 | 16,798 | 6,182 | **0.00%** |
| annexB/ | 1,079 | 1,011 | 7 | 1,004 | 68 | 0.69% |
| intl402/ | 1,566 | 1,449 | 0 | 1,449 | 117 | 0.00% |
| staging/ | 1,636 | 1,548 | 0 | 1,548 | 88 | 0.00% |

### Language Subcategory Breakdown

| Subcategory | Pass Rate | Pass/Attempted |
|-------------|-----------|----------------|
| **keywords** | **100.00%** | 25/25 |
| **punctuators** | **90.91%** | 10/11 |
| **return** | **62.50%** | 10/16 |
| **block** | **58.82%** | 10/17 |
| **identifiers** | **56.86%** | 116/204 |
| **assignment** | **55.40%** | 390/704 |
| **for-in** | **54.24%** | 64/118 |
| **block-scope** | **54.26%** | 51/94 |
| **literals** | **52.93%** | 226/427 |
| **break** | **52.63%** | 10/19 |
| **if** | **48.28%** | 28/58 |
| **comments** | **47.46%** | 28/59 |
| **continue** | **45.45%** | 10/22 |
| **reserved-words** | **46.15%** | 12/26 |
| **do-while** | **42.42%** | 14/33 |
| line-terminators | 41.46% | 17/41 |
| **while** | **37.14%** | 13/35 |
| asi | 34.31% | 35/102 |
| expression statements | 33.33% | 1/3 |
| variable | 30.41% | 45/148 |
| for | 20.40% | 194/951 |
| statements (agg) | 19.89% | 600/3,016 |
| expressions (agg) | 18.99% | 786/4,140 |
| function expressions | 16.19% | 34/210 |
| try | 14.29% | 26/182 |
| types | 10.09% | 11/109 |
| directive-prologue | 9.68% | 6/62 |
| conditional | 9.09% | 2/22 |
| white-space | 8.96% | 6/67 |
| global-code | 5.79% | 11/190 |
| eval-code | 0.42% | 3/710 |
| function-code | 0.00% | 0/376 |
| statementList | 0.00% | 0/60 |
| source-text | 0.00% | 0/1 |

### Key Observations

1. **Built-ins are the biggest failure category** — 22,980 tests, 0% pass rate. Builtins exist as Rust code (13,000+ lines, 737 unit tests) but are not registered in the global environment. Wiring them would immediately unlock thousands of passing tests.

2. **Parser is the #1 bottleneck for language tests** — Most failures cite `SyntaxError: unexpected token`. The harness itself (sta.js) uses `new Error()`, `new Test262Error()` which require parsing `new` expressions with constructor calls and the `Error` global — explaining why even trivially correct tests fail.

3. **Best areas** — Keywords (100%), punctuators (91%), and identifiers (57%) are the strongest categories, confirming the scanner/lexer is solid. Assignment (55%) and block-scope (54%) show the core variable/control-flow pipeline works for basic cases.

4. **Skipped tests** — 20,703 tests (41%) skipped due to unsupported features (async, generators, classes, Symbols, Proxy, BigInt, modules, Intl, etc.). As these features are implemented, the attempted count will grow.

5. **Execution speed** — The full 50,734-test suite completes in **1.2 seconds** (release build), demonstrating good engine startup and per-test throughput.

---

## 4. Quantitative Summary

| Metric | Stator | V8 (approximate) | Ratio |
|--------|--------|-------------------|-------|
| Lines of Rust/C++ | 68,581 | ~4,000,000 | 1.7% |
| Tests | 2,053 | ~100,000+ | 2% |
| Bytecode opcodes defined | 198 | ~174 | 114% |
| Bytecode opcodes **executed** | 70 | ~174 | 40% |
| Parser syntax coverage | ~30% ES2025 | ~99% ES2025 | 30% |
| Built-in runtime functions | ~737 test-covered | ~5,000+ | 15% |
| Built-ins accessible at runtime | ~3 (print, console.log, WebAssembly) | all | <1% |
| FFI C functions | 112 | ~310 | 36% |
| v8:: compat classes | 8 | 75+ | 11% |
| JIT tiers | 3 (baseline, Maglev, Turbofan/CL) | 3 (Sparkplug, Maglev, Turbofan) | 100% |
| GC generations | 2 (young + old) | 2 (young + old) | 100% |
| Test262 pass rate | **6.58%** (1,977/30,031 attempted) | ~99% | 6.6% |

---

## 5. Critical Path to Chromium Integration

### Phase A: Make JavaScript Actually Work (Priority: CRITICAL)

1. **Fix parser** — Add parsing for: array literals, object literals, arrow functions, class declarations, for-in/for-of, switch, destructuring, template literals, regexp literals, spread/rest, for(var...) in for-init. The AST types and bytecode generator already support most of these; only the parser is missing.

2. **Implement missing interpreter opcodes** — Focus on the 103 unhandled opcodes, prioritizing:
   - Context slots (LdaContextSlot, StaContextSlot, etc.) — **fixes closures**
   - TypeOf, ToString, ToNumber, ToBoolean, ToObject — **basic type coercion**
   - CreateArrayLiteral, CreateObjectLiteral, CreateRegExpLiteral
   - Bitwise/shift operators
   - Property keyed access (StaKeyedProperty, LdaKeyedProperty)
   - ForIn* opcodes
   - TestInstanceOf, TestIn

3. **Wire builtins to global environment** — Create a `install_globals()` function that registers Math, JSON, Array, Object, String, Number, Promise, Map, Set, RegExp, Error, parseInt, parseFloat, isNaN, isFinite, etc. into the interpreter's global HashMap. ~13,000 lines of builtin code is sitting unused.

### Phase B: Expand V8 API Surface (Priority: HIGH)

4. **Add v8:: wrapper classes** — Object, Array, Number, Function, Promise, TryCatch, ObjectTemplate at minimum for Blink integration.
5. **Expand FFI** — From 112 to ~200+ functions covering the most-used V8 API entry points.
6. **Implement prototype chains** — Essential for `instanceof`, method resolution, and standard library inheritance.

### Phase C: Spec Conformance (Priority: MEDIUM)

7. **Improve Test262 pass rate** — Test262 harness is operational (6.58%); target 25% by fixing parser + wiring builtins, 50% with full opcode coverage.
8. **Module system** — import/export for Chromium's module loading.
9. **Async/await** — Required for modern web applications.
10. **Full error types** — TypeError, RangeError, ReferenceError with proper messages.

### Phase D: Performance & Production (Priority: LOWER)

11. **Incremental/concurrent GC** — Required for production latency.
12. **Complete JIT coverage** — Currently JIT only handles numeric Smi operations.
13. **Inspector WebSocket transport** — Required for DevTools integration in Chromium.
14. **Snapshot/startup** — V8's snapshot mechanism for fast startup.

---

## 6. Honest Assessment

**What Stator has built well:**
- Sound architectural foundation matching V8's compilation pipeline
- Real, tested GC with generational collection
- Working 3-tier JIT on x86-64 (unique for a project of this maturity)
- Comprehensive FFI design with cbindgen automation
- Solid CI/fuzzing infrastructure
- Extensive builtin implementations with good test coverage

**What prevents it from being a V8 replacement today:**
- The parser is the single biggest bottleneck — it can't parse ~70% of JS syntax that the rest of the engine is prepared to handle
- Closures are broken (no context slot support)
- Not a single standard library object (Math, JSON, Array, etc.) is accessible from JavaScript code
- Prototype chain doesn't exist, so no method inheritance
- Only ~40% of defined opcodes actually execute
- The engine is ~1.7% of V8's code size; closing the gap requires substantial sustained effort

**Recommended immediate actions (highest ROI):**
1. Fix the parser (unblocks ~45% of the bytecode generator)
2. Implement LdaContextSlot/StaContextSlot (fixes closures)
3. Wire builtins to global env (makes 13,000 lines of code useful)
4. Implement TypeOf and type coercion opcodes

These four actions would approximately **double** the engine's functional parity.

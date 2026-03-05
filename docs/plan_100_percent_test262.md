# Stator: Plan to 100% Test262 + Beating V8 Performance

**Date:** 2026-03-05  
**Baseline:** 5.28% (1,583 / 29,997 attempted), 20,737 skipped, 79K LoR  
**Target:** 100% Test262 (50,734 tests), faster than V8 on all major benchmarks  

---

## Table of Contents

1. [Philosophy & Constraints](#1-philosophy--constraints)  
2. [Architectural Prerequisites](#2-architectural-prerequisites)  
3. [Phase 1 — Core Language (5% → 40%)](#3-phase-1--core-language-5--40)  
4. [Phase 2 — Standard Library (40% → 70%)](#4-phase-2--standard-library-40--70)  
5. [Phase 3 — Advanced Features (70% → 90%)](#5-phase-3--advanced-features-70--90)  
6. [Phase 4 — Edge Cases & Full Spec (90% → 100%)](#6-phase-4--edge-cases--full-spec-90--100)  
7. [Performance Architecture: Beating V8](#7-performance-architecture-beating-v8)  
8. [Estimated Effort & Milestones](#8-estimated-effort--milestones)  
9. [Risk Register](#9-risk-register)  

---

## 1. Philosophy & Constraints

### Why this is possible in Rust

V8 carries 4M lines of C++ accumulated over 16 years. Much of that is compatibility glue, platform abstraction, and technical debt. A clean-room Rust implementation can:

- **Eliminate undefined behavior** — no GC-during-compile data races, no use-after-free.
- **Use algebraic types** — `enum JsValue` is exhaustive; V8's tagged-pointer union is not.
- **Leverage Cranelift** — mature, audited code generator vs. V8's hand-rolled Turbofan backend.
- **Ship with `#[no_std]` core** — zero libc dependency in hot paths.

### Key constraint: spec-correctness first, then optimize

Every optimization must be gated behind `debug_assert!` equivalence checks against a reference interpreter path. We never sacrifice correctness for speed.

### Metric definitions

| Metric | Definition |
|--------|-----------|
| **Test262 pass rate** | `pass / (pass + fail)` — skipped tests don't count |
| **Attempted rate** | `(pass + fail) / total` — measures feature coverage |
| **Performance** | Geometric mean of Octane, JetStream2, Speedometer subtests vs. V8 baseline |

---

## 2. Architectural Prerequisites

These foundational changes must land before or during Phase 1. They affect everything downstream.

### 2.1 Object Model Rewrite (`objects/`)

**Current:** `JsValue` is a 19-variant Rust enum with `Rc<RefCell<HashMap<String, JsValue>>>` for objects. This is:
- 5-10× slower than V8's tagged-pointer + hidden-class model
- Incompatible with GC (Rc cycles leak)
- Unable to represent property descriptors, getters/setters, or prototype chains correctly

**Target architecture:**

```
┌────────────────────────────────────────────────────────────┐
│  JsValue: NaN-boxed 64-bit                                │
│  ┌──────────────┬─────────────────────────────────────────┐│
│  │ Smi (31-bit) │ HeapPtr (47-bit + tag in NaN payload)   ││
│  └──────────────┴─────────────────────────────────────────┘│
│                                                            │
│  HeapObject layout:                                        │
│  ┌──────┬──────────┬───────────┬──────────────────────────┐│
│  │ Map* │ hash/len │ elements* │ properties (in-object)   ││
│  └──────┴──────────┴───────────┴──────────────────────────┘│
│                                                            │
│  Map (Hidden Class):                                       │
│  ┌──────────────┬──────────────┬──────────────────────────┐│
│  │ instance_type│ bit_field    │ transition_table*        ││
│  │ prototype*   │ descriptors* │ back_pointer*            ││
│  └──────────────┴──────────────┴──────────────────────────┘│
│                                                            │
│  DescriptorArray:                                          │
│  ┌──────┬───────────┬──────────────┬─────────────────────┐│
│  │ key  │ details   │ field_index  │ constness           ││
│  └──────┴───────────┴──────────────┴─────────────────────┘│
└────────────────────────────────────────────────────────────┘
```

**Steps:**
1. Implement `NanBoxedValue` — 64-bit representation, Smi in lower 31 bits, heap pointer in payload with type tag in upper bits. No `enum` in hot path.
2. Implement `HeapObject` with `Map*` (hidden class pointer) as first word.
3. Implement `Map` (hidden class) with `DescriptorArray`, `TransitionTable`, prototype pointer, instance size, bit fields.
4. Implement in-object property storage (fixed slots after header) + overflow `PropertyArray`.
5. Implement `Elements` backing store (dense array: contiguous `JsValue[]`, sparse: `HashMap`).
6. All allocations through GC heap — no more `Rc<RefCell<>>`.

**Lines of code:** ~4,000  
**Unlocks:** Property descriptors, getters/setters, prototype chains, `Object.defineProperty`, `in` operator correctness, `for-in` enumeration order, hidden-class transitions (IC performance).

### 2.2 GC Integration

**Current:** GC infrastructure exists (2,985 lines) but objects use `Rc<RefCell<>>`, bypassing GC entirely.

**Target:**
1. All `JsValue::Object/Function/Array/String` become `HeapPtr<T>` — a GC-traced pointer.
2. Replace `Rc<RefCell<HashMap>>` with `GcPtr<JsObject>`.
3. Implement precise stack scanning (register map per call frame).
4. Implement write barrier on every `StaNamedProperty` / `StaKeyedProperty`.
5. Integrate generational collection: young objects in semi-space, tenured in mark-sweep-compact.

**Lines of code:** ~2,000 (integration) + refactoring existing 2,985 into live use.

### 2.3 String Representation

**Current:** `JsValue::String(String)` — heap-allocated, UTF-8, no rope, no interning.

**Target:**
- **SeqOneByteString** / **SeqTwoByteString** — Latin-1 for ASCII-only strings (majority of real workloads), UTF-16 otherwise.  
- **ConsString** — concatenation without copy (O(1) `+` operator).
- **SlicedString** — `substring()` without copy.  
- **InternalizedString** — identity-comparison interned strings for property keys.  
- **ThinString** — forwarding pointer after internalization.

**Lines of code:** ~2,500  
**Performance impact:** 3-5× on string-heavy benchmarks (DOM manipulation, JSON parsing).

---

## 3. Phase 1 — Core Language (5% → 40%)

**Goal:** Parse and execute all core ES2025 syntax. Remove `class`, `generators`, `Symbol.iterator`, `async-functions` from the skip list. This is the highest-ROI phase.

### 3.1 Parser Completions

| Feature | Test262 tests unlocked | Effort |
|---------|----------------------|--------|
| `class` declarations & expressions | ~4,200 (class + class-fields) | L |
| `for-of` with iterator protocol | ~800 | M |
| `switch` statements | ~400 | S |
| `new.target` in constructors | ~100 | S |
| Labeled `break` / `continue` | ~200 | S |
| `with` statement (sloppy mode) | ~150 | S |
| Computed property names (full) | ~300 | M |
| Property shorthand `{x}` → `{x: x}` | ~200 | S |
| Getter/setter `get x(){}` / `set x(){}` | ~400 | M |
| `import` / `export` declarations | ~800 (module-code) | L |
| `import()` dynamic import | ~200 | M |
| Optional catch binding `catch {}` | ~50 | S |
| Nullish coalescing `??` | ~50 | S |
| Logical assignment `??=` `||=` `&&=` | ~80 | S |
| Tagged templates | ~100 | S |
| Async functions + await | ~600 | L |
| Async generators | ~300 | L |
| Private class fields `#x` | ~500 | M |
| Static class blocks `static {}` | ~200 | M |

**Total parser work:** ~15 features, ~9,000 tests unlocked.

### 3.2 Bytecode Generator Completions

| Feature | Notes |
|---------|-------|
| Class compilation | `CreateClass`, field initializers, static blocks, private brand checks |
| For-of compilation | `GetIterator`, `IteratorNext`, `IteratorClose`, `IteratorValue` |
| Switch compilation | Jump table for dense `case` values, chained comparisons for sparse |
| Module compilation | `LdaModuleVariable`, `StaModuleVariable`, `import.meta` |
| Async/await compilation | `SuspendGenerator` / `ResumeGenerator` with Promise wiring |
| Destructuring in params | Generate binding pattern bytecodes for function parameters |
| Default parameters | Conditional assignment at function entry |
| Labeled jumps | Track label → break/continue target pairs |

### 3.3 Remaining Interpreter Opcodes (56 unhandled)

Priority-ordered remaining opcodes:

| Priority | Opcodes | Test262 impact |
|----------|---------|----------------|
| **P0** | `DeletePropertyStrict`, `DeletePropertySloppy` | ~200 tests |
| **P0** | `CreateRestParameter`, `CreateMappedArguments`, `CreateUnmappedArguments` | ~400 tests |
| **P0** | `ThrowReferenceErrorIfHole`, `ThrowSuperNotCalledIfHole`, `ThrowSuperAlreadyCalledIfNotHole` | ~300 tests (class constructors) |
| **P0** | `CallProperty0`, `CallProperty1`, `CallProperty2`, `CallRuntime` | ~500 tests |
| **P0** | `StaNamedOwnProperty`, `StaLookupSlot` | ~300 tests |
| **P1** | `LdaLookupSlot`, `LdaLookupSlotInsideTypeof`, `LdaLookupContextSlot`, `LdaLookupContextSlotInsideTypeof` | ~200 tests (with/eval scoping) |
| **P1** | `LdaLookupGlobalSlot`, `LdaLookupGlobalSlotInsideTypeof` | ~100 tests |
| **P1** | `LdaNamedPropertyFromSuper` | ~200 tests (class) |
| **P1** | `LdaEnumeratedKeyedProperty` | ~50 tests |
| **P1** | `GetTemplateObject` | ~100 tests |
| **P1** | `SetPendingMessage`, `TestReferenceEqual`, `TestUndetectable` | ~100 tests |
| **P1** | `JumpIfJSReceiver`, `JumpIfJSReceiverConstant` | ~100 tests |
| **P1** | `ToNumeric` | ~50 tests |
| **P2** | `Constant` jump variants (all `*Constant` opcodes) | Performance — same semantics as non-constant, just wider operand |
| **P2** | `CallJSRuntime`, `InvokeIntrinsic`, `CallRuntimeForPair` | Host integration |
| **P2** | `ConstructForwardAllArgs` | ~50 tests |
| **P2** | `CollectTypeProfile` | No-op for correctness, needed for feedback |
| **P2** | `CreateObjectFromIterable` | ~50 tests |
| **P2** | `Wide`, `ExtraWide` | Prefix opcodes — required when operands exceed 8/16 bits |

### 3.4 Spec-Correct Type Coercion

Many Test262 failures come from incorrect coercion in operators. Each operator must call the correct Abstract Operation:

| Operation | Spec algorithm | Current status |
|-----------|---------------|----------------|
| `ToPrimitive(input, hint)` | OrdinaryToPrimitive → `valueOf`/`toString` | **Missing** — no `valueOf`/`toString` dispatch |
| `ToNumber(arg)` | Handles undefined→NaN, null→0, bool→0/1, string→parse, object→ToPrimitive | **Partial** — missing object→ToPrimitive |
| `ToString(arg)` | object→ToPrimitive(hint:string) | **Partial** |
| `ToPropertyKey(arg)` | ToPrimitive(hint:string), then ToString unless Symbol | **Missing** |
| `ToObject(arg)` | Wraps primitives in Boolean/Number/String wrapper objects | **Missing** — wrapper objects not implemented |
| `ToInt32`, `ToUint32`, `ToInt16`, etc. | Modular truncation after ToNumber | **Partial** — missing edge cases |
| `SameValue`, `SameValueZero` | Distinguished `-0` vs `+0`, `NaN === NaN` | **Missing** |
| `IsLooselyEqual` (==) | Full 10-step algorithm including ToPrimitive, ToNumber | **Incomplete** — missing object coercion path |
| `IsStrictlyEqual` (===) | Type check then value comparison | **Mostly correct** |

### 3.5 Milestone gate

**Phase 1 exit criteria:**
- Skip list reduced to: `Proxy`, `Reflect`, `BigInt`, `Atomics`, `SharedArrayBuffer`, `WeakRef`, `FinalizationRegistry`, `Intl`, `tail-call-optimization`
- All `class`, `async`, `generator`, `Symbol`, `for-of`, `switch`, `module` tests attempted
- Test262 pass rate ≥ 40% (≥ 18,000 / ~45,000 attempted)
- All 2,587 existing unit tests still pass

---

## 4. Phase 2 — Standard Library (40% → 70%)

**Goal:** Spec-complete implementations of all ES2025 built-in objects. This phase is where the largest absolute number of Test262 tests live (~23,000 in `built-ins/`).

### 4.1 Prototype Chain Infrastructure

Before any builtin can pass Test262, the prototype chain must work correctly:

1. **Every constructor has `.prototype`** with correct `constructor` back-reference
2. **`Object.getPrototypeOf` / `Object.setPrototypeOf`** work
3. **`instanceof`** uses `[Symbol.hasInstance]` fallback → `OrdinaryHasInstance`
4. **Property lookup** walks `__proto__` chain (not just own properties)
5. **`Object.create(proto)`** sets `[[Prototype]]` correctly
6. **`.toString()` / `.valueOf()`** dispatch through prototype chain

### 4.2 Built-in Object Spec Compliance

Each built-in needs to be rewritten to match the spec algorithm step-by-step. Current implementations are "close enough for unit tests" but fail Test262 because they skip edge cases.

| Built-in | Test262 tests | Key gaps |
|----------|--------------|----------|
| **Object** | ~2,400 | `defineProperty` descriptor validation, `preventExtensions`/`seal`/`freeze`, `getOwnPropertyDescriptor` with accessors, property enumeration order (insertion order + integer indices first) |
| **Array** | ~3,200 | Species pattern (`Symbol.species`), generic methods (work on array-likes), `length` as exotic property, `Array.from`/`Array.of`, correct `holes` handling, stable `sort` |
| **String** | ~2,100 | `String.prototype[Symbol.iterator]` (codepoint iteration), `matchAll`, `replaceAll`, `trimStart`/`trimEnd`, `padStart`/`padEnd`, correct Unicode normalization |
| **Number** | ~800 | `Number.isFinite`/`isNaN`/`isInteger`/`isSafeInteger`, `toFixed`/`toPrecision`/`toExponential` edge cases, `Number.parseFloat`/`parseInt` spec compliance |
| **Math** | ~400 | All 43 static methods with correct edge cases (Math.log(-1)→NaN, Math.pow(-∞, 0.5)→+∞, etc.) |
| **RegExp** | ~1,800 | Named groups, lookbehind, dotAll flag, Unicode property escapes, `matchAll` (RegExpStringIterator), sticky flag semantics, `Symbol.match`/`replace`/`search`/`split` |
| **JSON** | ~400 | `JSON.stringify` replacer + space + toJSON, `JSON.parse` reviver, `Infinity`/`NaN`/`undefined` handling |
| **Map / Set** | ~600 | Insertion-order iteration, `forEach`, `entries`/`keys`/`values` iterators, `Symbol.iterator`, correct key comparison (SameValueZero) |
| **Promise** | ~800 | `then` chaining, microtask queue, `Promise.all`/`allSettled`/`any`/`race`, `finally`, unhandled rejection tracking |
| **Error** | ~300 | `Error.prototype.stack` (non-standard but needed), `.cause` property, `AggregateError`, correct `name`/`message` inheritance |
| **TypedArray** | ~2,500 | All 9 typed array types, `ArrayBuffer`, `DataView`, byte-order, `.slice`, `.subarray`, detached buffer checks |
| **Date** | ~800 | Full `Date.parse`, `Date.UTC`, all getter/setters, `toISOString`/`toJSON`, `Symbol.toPrimitive` |
| **Function** | ~500 | `.bind`, `.call`, `.apply`, `.name`, `.length`, `Function.prototype[Symbol.hasInstance]` |
| **Proxy** | ~1,400 | All 13 internal method traps, invariant enforcement, revocable proxies |
| **Reflect** | ~400 | Mirror of Proxy traps as functions |
| **WeakMap / WeakSet** | ~200 | GC-weak references, correct key constraints (objects only) |
| **WeakRef / FinalizationRegistry** | ~150 | GC integration, cleanup callbacks |
| **Symbol** | ~400 | Well-known symbols, `Symbol.for`/`Symbol.keyFor`, `Symbol.toPrimitive` |
| **BigInt** | ~400 | Arbitrary precision, operators, no implicit coercion with Number |
| **Atomics / SharedArrayBuffer** | ~300 | `wait`/`notify`, atomic RMW ops, requires threading model |
| **Iterator helpers** | ~200 | `Iterator.prototype.map/filter/take/drop/flatMap/reduce/toArray/forEach/some/every/find` |
| **Global functions** | ~300 | `eval` (direct/indirect), `isNaN`, `isFinite`, `parseInt`, `parseFloat`, `encodeURI`/`decodeURI`, `encodeURIComponent`/`decodeURIComponent` |

### 4.3 Microtask Queue

Required for Promise correctness. Implementation:

```
struct MicrotaskQueue {
    queue: VecDeque<Box<dyn FnOnce()>>,
}

impl MicrotaskQueue {
    fn enqueue(&mut self, task: Box<dyn FnOnce()>);
    fn drain(&mut self); // Called after each script/module evaluation
}
```

Every `then`/`catch`/`finally` enqueues a microtask. `await` expression suspends to a pending microtask.

### 4.4 Milestone gate

**Phase 2 exit criteria:**
- Skip list reduced to: `Atomics`, `SharedArrayBuffer`, `Intl`, `tail-call-optimization`
- All `Proxy`, `Reflect`, `BigInt`, `WeakRef`, `FinalizationRegistry` attempted
- Test262 pass rate ≥ 70% (≥ 33,000 / ~47,000 attempted)
- Benchmark: Octane score within 2× of V8 on interpreter-only (no JIT)

---

## 5. Phase 3 — Advanced Features (70% → 90%)

### 5.1 Complete Module System

1. **Source Text Module Records** — `import`/`export` with circular dependency resolution
2. **`import.meta`** — environment-specific metadata object
3. **Dynamic `import()`** — returns Promise, resolves module specifier at runtime
4. **Namespace exotic objects** — `import * as ns from 'mod'`
5. **`export * from`** — re-export with possible name conflicts
6. **Module environment records** — lexical bindings, live bindings for `export let`

### 5.2 Direct & Indirect `eval()`

Critical for many Test262 tests:
- **Direct eval:** Creates new scope in calling context
- **Indirect eval:** `(0, eval)('code')` runs in global scope
- **Strict mode eval:** Gets its own variable environment
- Variable hoisting through eval requires dynamic scope chain lookup (the `LdaLookupSlot*` opcodes)

### 5.3 `with` Statement

Sloppy-mode only, but ~150 Test262 tests depend on it:
- Creates `ObjectEnvironmentRecord` that intercepts property lookups
- Requires `has` trap awareness for Proxy interop

### 5.4 Full Proxy/Reflect

All 13 MOP (Meta-Object Protocol) traps:
`getPrototypeOf`, `setPrototypeOf`, `isExtensible`, `preventExtensions`, `getOwnPropertyDescriptor`, `defineOwnProperty`, `has`, `get`, `set`, `deleteProperty`, `ownKeys`, `apply`, `construct`

Each trap must enforce the invariant checks (e.g. non-configurable property can't be reported as non-existent by `has` trap).

### 5.5 Complete RegExp

- **Named capture groups:** `(?<name>pattern)` → `match.groups.name`
- **Lookbehind assertions:** `(?<=)` and `(?<!)`
- **`/s` (dotAll) flag**
- **Unicode property escapes:** `\p{Letter}`, `\P{Number}`
- **`/v` flag** (Unicode sets)
- **`match indices`** (`/d` flag) → `match.indices`

The `regress` crate handles most of this. Gap is in the JS-side integration (creating match objects with the right shape).

### 5.6 Intl (Internationalization)

~1,566 Test262 tests. Requires ICU bindings:
- `Intl.Collator`, `Intl.DateTimeFormat`, `Intl.NumberFormat`, `Intl.PluralRules`, `Intl.RelativeTimeFormat`, `Intl.ListFormat`, `Intl.Segmenter`, `Intl.DisplayNames`, `Intl.Locale`
- Bind to `icu4x` (Rust-native ICU) rather than system ICU4C for deterministic behavior

### 5.7 SharedArrayBuffer & Atomics

Requires a threading model:
- `SharedArrayBuffer` — shared memory between agents (threads/workers)
- `Atomics.wait` / `Atomics.notify` — futex-style synchronization
- `Atomics.load/store/add/sub/and/or/xor/exchange/compareExchange`
- Memory ordering: `SeqCst` for all atomic ops

### 5.8 Milestone gate

**Phase 3 exit criteria:**
- Skip list: `tail-call-optimization` only (35 tests — spec feature only Safari implements)
- Test262 pass rate ≥ 90% (≥ 45,000 / ~50,700 attempted)

---

## 6. Phase 4 — Edge Cases & Full Spec (90% → 100%)

### 6.1 Annex B (Web Legacy)

~1,079 tests for legacy web behaviors:
- `__proto__` property
- `String.prototype.substr` (non-standard)
- Labelled function declarations in sloppy mode
- `RegExp` legacy static properties (`RegExp.$1`, etc.)
- HTML comment syntax in scripts (`<!-- -->`)
- `escape()` / `unescape()`
- Block-scoped function declarations in sloppy mode

### 6.2 Tail Call Optimization (35 tests)

Currently skipped. Only Safari implements this. Options:
1. **Implement TCO** — rewrite call frames in-place when tail position detected. ~500 lines.
2. **Skip permanently** — legitimate spec-optional feature.

Recommendation: Implement it. It's a competitive advantage and only ~500 lines. Mark tail positions in bytecode generator, reuse frame in interpreter.

### 6.3 Staging Tests

~1,636 tests for stage-3/4 proposals not yet in ES2025:
- Explicit resource management (`using` / `await using`)
- Decorator metadata
- Array grouping (`Object.groupBy`, `Map.groupBy`)
- `Set` methods (union, intersection, difference, etc.)
- `Promise.withResolvers`
- `RegExp` modifiers

### 6.4 Edge-Case Hardening

The last ~5% is the hardest. Common failure patterns:
- **Unicode edge cases:** Surrogate pairs in identifiers, string length vs codepoint count
- **`arguments` object:** Mapped vs unmapped, `callee`/`caller` in strict mode
- **Numeric precision:** `Number.MAX_SAFE_INTEGER + 1 === Number.MAX_SAFE_INTEGER + 2`
- **Property enumeration order:** Integer indices sorted numerically, then string keys in insertion order, then symbols
- **`eval` + closures:** Variable declaration hoisting through eval in nested closures
- **Proxy invariant violations:** Proper TypeError throwing for every violation
- **Generator `return()` / `throw()`:** Correct abrupt completion handling during iteration
- **Module circular dependencies:** Live bindings, TDZ for uninitialized imports
- **Strict mode edge cases:** `delete` on unresolvable reference, `arguments` restrictions, `eval` restrictions

### 6.5 Milestone gate

**Phase 4 exit criteria:**
- Skip list: empty (even TCO implemented)
- Test262 pass rate = 100% (50,734 / 50,734)
- Zero regressions on full suite across 3 consecutive runs

---

## 7. Performance Architecture: Beating V8

### 7.1 Where V8 Is Slow (and Stator Can Win)

| V8 weakness | Stator opportunity |
|-------------|-------------------|
| **Startup time** — V8's snapshot is ~1.5MB, parsing is deferred but still UTF-16 | Stator: native Rust binary, startup snapshot from `serde_bincode`, UTF-8 → UTF-16 lazy conversion |
| **Memory overhead** — V8 uses ~2MB per isolate baseline | Stator: `bumpalo` arena for compiler temps, zero-copy string interning, smaller object headers |
| **GC pauses** — V8's concurrent marker can't eliminate all pauses | Stator: Immix-style GC with region-based evacuation, no global safepoints for minor GC |
| **IC megamorphism** — V8 falls back to runtime after 4 IC states | Stator: PIC (polymorphic inline cache) with 8 entries, Bloom-filter fast path for megamorphic |
| **String concatenation** — V8 creates ConsString but flatten is expensive | Stator: Rope-based strings with O(1) concatenation, lazy materialization |
| **Cold function compilation** — Sparkplug compiles all functions, even once-executed | Stator: profile-guided lazy compilation, only compile after 2nd call |

### 7.2 JIT Compilation Pipeline (4 tiers)

```
                    ┌─────────────────────────────────────────┐
                    │              Source Text                 │
                    └─────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────┐
                    │         Parser + BytecodeGen             │
                    │    (< 1ms per function, single-pass)     │
                    └─────────────┬───────────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │        Tier 0: Interpreter               │
              │  • 0 compilation overhead                 │
              │  • Collects type feedback (FeedbackVector)│
              │  • ~20× slower than native               │
              └───────────────────┬──────────────────────┘
                                  │ (call_count > 6)
              ┌───────────────────▼──────────────────────┐
              │     Tier 1: Sparkplug (Baseline JIT)     │
              │  • < 0.1ms per function                  │
              │  • 1:1 bytecode → native translation     │
              │  • No optimization — just removes         │
              │    dispatch overhead                      │
              │  • ~5× slower than optimal native        │
              └───────────────────┬──────────────────────┘
                                  │ (call_count > 100)
              ┌───────────────────▼──────────────────────┐
              │     Tier 2: Maglev (Mid-Tier JIT)        │
              │  • < 1ms per function                    │
              │  • SSA-based IR, type-specialization      │
              │  • Range analysis, LICM, dead code elim   │
              │  • Inline caches, type guards             │
              │  • ~2× slower than optimal native        │
              └───────────────────┬──────────────────────┘
                                  │ (call_count > 10000 + hot loop)
              ┌───────────────────▼──────────────────────┐
              │    Tier 3: Cranelift (Top-Tier JIT)      │
              │  • 1-10ms per function                   │
              │  • Full SSA optimization suite            │
              │  • GVN, loop unrolling, register coalesce │
              │  • Speculative specialization w/ deopt    │
              │  • Target: 0.9-1.1× optimal native       │
              └──────────────────────────────────────────┘
```

### 7.3 Critical Optimizations for Beating V8

#### 7.3.1 Hidden Class Transitions (Shapes)

V8's hidden class system is its single biggest performance feature. We must match and exceed it:

```rust
struct Shape {
    parent: Option<ShapeId>,       // Transition from parent shape
    property: InternedString,       // Key added in this transition
    offset: u32,                    // In-object slot index
    attrs: PropertyAttributes,      // writable, enumerable, configurable
    transition_table: SmallVec<[(InternedString, ShapeId); 4]>,
    prototype: Option<HeapPtr<JsObject>>,
    instance_size: u16,
    n_in_object_properties: u8,
}
```

**Optimization over V8:** Use a global shape table with `ShapeId` (u32 index) instead of heap pointers. Shape lookups are array indexing, not pointer chasing. Shapes are never GC'd (they're in a side arena).

#### 7.3.2 Inline Caches (ICs)

```rust
struct LoadIC {
    map_id: ShapeId,          // Expected shape
    offset: u32,              // Slot offset if monomorphic
    state: ICState,           // Uninit → Mono → Poly(4) → Mega
}

// Hot path: 2 instructions (cmp + load)
fn load_named_property_ic(obj: &JsObject, ic: &mut LoadIC) -> JsValue {
    if obj.shape_id() == ic.map_id {
        return obj.in_object_slot(ic.offset); // FAST PATH
    }
    load_named_property_miss(obj, ic) // Updates IC, returns value
}
```

#### 7.3.3 Allocation Fast Path

```rust
// Bump allocation: 1 add + 1 compare + 1 store
fn allocate<T: HeapObject>(heap: &Heap, size: usize) -> *mut T {
    let addr = heap.young.alloc_ptr;
    let new_ptr = addr + size;
    if new_ptr <= heap.young.limit {
        heap.young.alloc_ptr = new_ptr;
        addr as *mut T
    } else {
        allocate_slow(heap, size)
    }
}
```

#### 7.3.4 Interpreter Dispatch: Computed Goto

Current interpreter uses `match` on opcode enum. Replace with computed-goto dispatch table:

```rust
// Instead of:  match opcode { ... }
// Use:
static DISPATCH_TABLE: [fn(&mut Frame); 198] = [
    handle_lda_zero,
    handle_lda_smi,
    handle_lda_undefined,
    // ...
];

// Hot loop:
loop {
    let opcode = bytecode[pc] as usize;
    DISPATCH_TABLE[opcode](&mut frame);
}
```

This eliminates branch prediction misses from the central `match`. V8 uses this technique; Stator currently doesn't.

#### 7.3.5 Deoptimization Framework

For speculative optimizations (type guards, bounds check elimination):

```rust
struct DeoptEntry {
    bytecode_offset: u32,     // Where to resume in interpreter
    register_map: Vec<ValueRecovery>,  // How to reconstruct frame state
    reason: DeoptReason,
}

enum ValueRecovery {
    InRegister(PhysReg),
    OnStack(i32),
    Constant(JsValue),
    Materialized,  // Object that was scalar-replaced
}
```

Every speculative optimization records a deopt entry. When a type guard fails, the JIT code calls `Deoptimizer::deopt()` which reconstructs the interpreter frame and resumes.

#### 7.3.6 Background Compilation

```rust
// Main thread:
fn on_tier_up_request(function_id: FunctionId, tier: Tier) {
    compilation_queue.push(CompileJob { function_id, tier });
}

// Background thread pool (N = num_cpus - 1):
fn compilation_worker(queue: &CompileQueue) {
    loop {
        let job = queue.pop_blocking();
        let ir = build_ir(job.function_id, job.tier);
        let code = generate_code(ir);
        // Atomic swap of code pointer happens on main thread
        install_queue.push(InstallJob { function_id, code });
    }
}
```

### 7.4 Memory Layout Optimizations

#### 7.4.1 Object Sizes

| | V8 | Stator target |
|--|---|--------------|
| Empty `{}` | 56 bytes | 32 bytes (shape_id:4 + hash:4 + elements_ptr:8 + 2 in-object slots:16) |
| Small array `[1,2,3]` | 80 bytes | 48 bytes (header:16 + length:4 + pad:4 + elements-inline:24) |
| Function | 72 bytes | 48 bytes (header:16 + bytecode_ptr:8 + context_ptr:8 + feedback_ptr:8 + flags:8) |

#### 7.4.2 NaN-Boxing

```
 Smi (31-bit signed):  xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxx1  (bit 0 = 1)
 HeapPtr:              0TTTAAAA AAAAAAAA AAAAAAAA AAAAAAA0  (bit 0 = 0, T = type tag)
 Double:               stored separately (when needed) or as HeapNumber in heap

 Where T (3-bit type tag):
   000 = Object
   001 = String
   010 = Symbol  
   011 = BigInt
   100 = Function
   101 = Array
   110 = (reserved)
   111 = (reserved — special values)
   
 Special values encoded in tag:
   false     = 0x06  (0...0 0000 0110)
   true      = 0x16  (0...0 0001 0110)  
   null      = 0x26  (0...0 0010 0110)
   undefined = 0x36  (0...0 0011 0110)
   hole      = 0x46  (   the "empty" sentinel)
```

This fits in 64 bits with no boxing for Smis, booleans, null, undefined. Doubles promote to HeapNumber only when needed. This beats V8's tagged representation on 64-bit (V8 uses 64-bit pointers with Smi in upper 32 bits + shift, requiring more ALU ops).

### 7.5 GC Design for Performance

#### Immix-Style Regional GC

Instead of V8's semi-space scavenger:

```
┌─────────────────────────────────────────────────┐
│  Young Generation (Immix)                        │
│  ┌──────┬──────┬──────┬──────┬──────┬──────────┐│
│  │Block │Block │Block │Block │Block │   ...    ││
│  │ 32KB │ 32KB │ 32KB │ 32KB │ 32KB │          ││
│  │ bump │ bump │ FULL │ bump │ FULL │          ││
│  └──────┴──────┴──────┴──────┴──────┴──────────┘│
│                                                  │
│  Collection: mark lines (128B) → evacuate only   │
│  blocks with < 50% live data. No full copy.      │
│  Average pause: 0.2ms (V8 scavenger: 0.5-2ms)   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Old Generation (Mark-Region)                    │
│  ┌──────────────────────────────────────────────┐│
│  │  Same block structure, concurrent marking     ││
│  │  Compaction: incremental, region-at-a-time    ││
│  │  Average pause: 0.1ms (V8 MSC: 0.5-5ms)      ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

**Advantage over V8:** Immix eliminates the 50% space overhead of semi-space copying. Same or better throughput with half the memory. Concurrent marking with snapshot-at-the-beginning barrier (Dijkstra-style).

#### 7.5.1 Allocation Buffer per Thread

```rust
thread_local! {
    static TLAB: Cell<AllocBuffer> = Cell::new(AllocBuffer::empty());
}

struct AllocBuffer {
    cursor: *mut u8,
    limit: *mut u8,  // end of 32KB block
}

fn alloc(size: usize) -> *mut u8 {
    TLAB.with(|tlab| {
        let buf = tlab.get();
        let new_cursor = buf.cursor.add(size);
        if new_cursor <= buf.limit {
            let ptr = buf.cursor;
            tlab.set(AllocBuffer { cursor: new_cursor, ..buf });
            ptr
        } else {
            alloc_slow(size) // Get new block from global heap
        }
    })
}
```

---

## 8. Estimated Effort & Milestones

### Lines of Code Estimate

| Component | New LoR | Refactor existing |
|-----------|---------|-------------------|
| Object model rewrite (NaN-boxing + shapes) | 4,000 | 3,000 |
| GC integration (precise roots, write barriers) | 2,000 | 2,000 |
| String representation (ropes, internalization) | 2,500 | 1,000 |
| Parser completions (class, modules, etc.) | 3,000 | 500 |
| Bytecode generator (class, for-of, etc.) | 2,000 | 500 |
| Interpreter opcodes (56 remaining) | 1,500 | 200 |
| Type coercion (spec-correct ToPrimitive, etc.) | 1,000 | 500 |
| Built-in objects (spec-complete rewrite) | 15,000 | 8,000 |
| Microtask queue + Promise integration | 800 | 200 |
| Module system | 2,000 | 0 |
| eval() direct/indirect | 1,000 | 200 |
| Proxy/Reflect (13 traps + invariants) | 2,000 | 1,200 |
| BigInt | 1,500 | 0 |
| SharedArrayBuffer/Atomics | 1,500 | 0 |
| Intl (icu4x bindings) | 4,000 | 0 |
| WeakRef/FinalizationRegistry | 500 | 0 |
| Annex B legacy | 800 | 0 |
| TCO | 500 | 200 |
| IC framework (shapes + PIC) | 2,000 | 1,000 |
| Interpreter dispatch optimization | 500 | 500 |
| Baseline JIT full opcode coverage | 3,000 | 1,800 |
| Maglev full coverage | 4,000 | 2,400 |
| Cranelift tier full coverage | 3,000 | 1,400 |
| Deoptimization framework | 1,500 | 500 |
| Background compilation | 1,000 | 200 |
| Test262 harness improvements | 500 | 300 |
| **Total** | **~60,000** | **~26,000** |

**Total new + refactored: ~86,000 lines** on top of existing 79,000 → ~165,000 lines final.  
(V8 is ~4,000,000 lines. Stator at ~165K is 4% of V8 for 100% conformance.)

### Timeline (Sequential, Single-Team)

| Phase | Duration | Test262 target | Performance target |
|-------|----------|---------------|-------------------|
| **Phase 0: Object model** | 6 weeks | No change (refactor) | — |
| **Phase 1: Core language** | 10 weeks | 40% | — |
| **Phase 2: Standard library** | 14 weeks | 70% | 0.5× V8 (interp only) |
| **Phase 3: Advanced features** | 10 weeks | 90% | 0.8× V8 (with Maglev) |
| **Phase 4: Edge cases + polish** | 8 weeks | 100% | 1.0× V8 (with Cranelift) |
| **Phase 5: Performance sprint** | 8 weeks | 100% | 1.2× V8 (target) |
| **Total** | **~56 weeks** | **100%** | **>1.0× V8** |

### Parallelizable Tracks

With a team, these can run concurrently:

```
Track A: Correctness (1 engineer)     Track B: Performance (1 engineer)
─────────────────────────────         ──────────────────────────────────
Phase 0: Object model                 Phase 0: NaN-boxing implementation
Phase 1: Parser + bytegen             Phase 1: IC framework + dispatch
Phase 2: Built-ins                    Phase 2: Baseline JIT full coverage
Phase 3: Proxy/Modules/Intl           Phase 3: Maglev full coverage
Phase 4: Edge cases                   Phase 4: Cranelift + deopt framework
                                      Phase 5: Benchmark-driven tuning

Track C: GC (1 engineer)
──────────────────────────
Phase 0: Immix young-gen
Phase 1: Mark-region old-gen
Phase 2: Concurrent marking
Phase 3: Incremental compaction
Phase 4: WeakRef/FinReg integration
```

With 3 engineers: **~24 weeks** (6 months).

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Object model rewrite breaks all existing tests | High | High | Feature-flag new path, run both in parallel, switch when Green |
| Spec edge cases in builtins take 3× estimated time | High | Medium | Use Test262 as TDD — implement one test at a time, don't gold-plate |
| Cranelift can't match Turbofan's peak throughput | Medium | Medium | Cranelift is within 10-20% of LLVM; aggressive inlining + deopt compensates |
| Intl (ICU) adds 5MB+ binary size | Medium | Low | Use `icu4x` with data slicing — only include locales needed for Test262 |
| GC rework causes memory leaks during transition | High | Medium | Leak detector in debug builds, `#[cfg(debug_assertions)]` ref-count auditing |
| `eval()` + closures interaction too complex | Medium | Medium | Implement conservatively (slow path always), optimize later |
| Proxy invariant checks are combinatorially complex | Medium | Medium | Port V8's invariant check logic — well-tested, spec-mapped |
| Thread safety for SharedArrayBuffer | Medium | High | Use Rust's `Send`/`Sync` + `parking_lot` — type system prevents data races |

---

## Summary: The Critical Path

```
       NOW (5.28%)
        │
        ▼
   ┌──────────────────────┐
   │  OBJECT MODEL REWRITE │ ← Prerequisite for everything
   │  (NaN-box + Shapes)   │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  CLASS + FOR-OF +     │ ← Unlocks 5,000+ skipped tests
   │  SWITCH + MODULES     │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  SPEC-CORRECT BUILTINS│ ← 23,000 built-in tests
   │  (prototype chain OK) │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  PROXY + BIGINT +    │ ← 2,500+ tests
   │  WEAKREF + INTL      │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │  EDGE CASES + TCO +  │ ← Last 5%
   │  ANNEX B + STAGING   │
   └──────────┬───────────┘
              │
              ▼
        100% (50,734)
```

The single most impactful action is the **object model rewrite** — without proper shapes, prototype chains, and property descriptors, no amount of parser or builtin work will move the needle past ~40%. Everything builds on this foundation.

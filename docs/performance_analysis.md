# Stator Performance Analysis & Optimization Plan

**Date:** 2025-07-19
**Branch:** `feature/ws5-performance`
**Engine version:** post Phase-A conformance sprint (~5.28% Test262 pass rate)

---

## 1. Executive Summary

This document catalogues the top performance bottlenecks in the Stator
interpreter measured against V8's architecture, ranks them by estimated
impact, and proposes a phased optimization plan. The analysis is based on
code inspection of the dispatch loop, property lookup paths, value
representation, frame management, and GC subsystem.

**Key finding:** The interpreter is estimated to be **10–50× slower than
V8's Ignition interpreter** on typical workloads. Most of the gap comes from
five sources that are individually addressable:

| Rank | Bottleneck | Estimated Impact | Effort |
|------|-----------|-----------------|--------|
| 1 | JsValue size (24 B vs 8 B) | 2–3× cache pressure | Large |
| 2 | Excessive `.clone()` in dispatch hot paths | 2–3× on object-heavy code | Medium |
| 3 | Frame initialization allocates 5+ HashMaps | 3–5% overall allocation | Small |
| 4 | Integer-indexed property insertion O(n²) | 10× on large indexed objects | Medium |
| 5 | Global variable access through `Rc<RefCell<HashMap>>` | 10–15% on global-heavy scripts | Medium |

---

## 2. Architecture Overview

### 2.1 Dispatch Loop (`interpreter/dispatch.rs`)

Stator already uses a **static function-pointer dispatch table**
(`DISPATCH_TABLE: [OpcodeHandler; 192]`) indexed by opcode discriminant.
Each handler is a standalone `fn(&mut DispatchContext, &Instruction) ->
StatorResult<DispatchAction>` invoked via indirect call. This is materially
better than an exhaustive `match` — the CPU branch predictor sees a
consistent indirect-call site.

**Gap vs V8:** V8's Ignition uses hand-written assembly with
computed-`goto` (GNU extension) or tail-call trampolines per platform. That
eliminates the function-call overhead entirely (no prologue/epilogue, no
`StatorResult` return-value wrapping). Estimated cost: **5–10% overhead per
instruction**.

### 2.2 Value Representation (`objects/value.rs`)

`JsValue` is a 23-variant Rust `enum`:

```text
JsValue size = 24 bytes  (discriminant + largest payload: BigInt i128)
```

V8's `TaggedValue` is **8 bytes** (NaN-boxed or pointer-tagged). Stator's
`TaggedValue` also exists (`objects/tagged.rs`) but is only used in the GC
heap layer, not in the interpreter register file.

**Consequence:** The interpreter register file is 3× larger than optimal.
A frame with 32 registers occupies 768 B instead of 256 B, increasing L1
cache pressure on every instruction that touches registers.

### 2.3 Property Lookup (`objects/property_map.rs`)

`PropertyMap` is a well-designed structure:

- **Flat `Vec<JsValue>` storage** — cache-friendly sequential access.
- **4-entry FNV-1a inline cache** — avoids `HashMap` probing on repeated
  hot-property access. Cache hit is O(1).
- **`HashMap<String, usize>` index** — O(1) amortized for cold lookups.
- **Shape ID** — monotonic stamp for inline-cache invalidation.

The inline cache and shape-based offset API (`offset_of` / `get_by_offset`)
are the right primitives. The interpreter dispatch already has monomorphic
and polymorphic load caches in the `InterpreterFrame` that consult
`shape_id` + `offset`.

**Gap:** Integer-indexed insertion (`spec_insert_pos`) uses a linear scan
of all existing keys to find the correct sorted position, then
`index_shift_right` iterates the entire `HashMap` to bump indices. This
makes `obj[i] = v` for indices 0..N an **O(N²)** operation.

### 2.4 Frame Management (`interpreter/mod.rs`)

`InterpreterFrame::new()` allocates:

| Field | Type | Cost |
|-------|------|------|
| `registers` | `Vec<JsValue>` | 1 heap alloc |
| `global_env` | `Rc<RefCell<HashMap>>` | 1 alloc (top-level) / shared |
| `template_cache` | `HashMap<u32, JsValue>` | 1 alloc |
| `mono_load_cache` | `HashMap<u32, (usize, JsValue)>` | 1 alloc |
| `poly_load_cache` | `HashMap<u32, Vec<…>>` | 1 alloc |
| `shape_load_ic` | `HashMap<u32, PropertyIc>` | 1 alloc |
| `shape_store_ic` | `HashMap<u32, PropertyIc>` | 1 alloc |
| `string_cache` | `HashMap<u32, Rc<str>>` | 1 alloc |

Every function **call** creates a new frame. For a call-heavy workload
(1 million calls), this is **5–8 million HashMap allocations**, most of
which are never populated for leaf functions.

### 2.5 Garbage Collection (`gc/`)

| Component | Algorithm | Status |
|-----------|-----------|--------|
| Young generation | Cheney semi-space scavenger | Implemented |
| Old generation | Mark-Sweep-Compact | Implemented |
| Write barrier | Store-buffer remembered set | Implemented |
| Promotion threshold | 3 scavenge cycles | Configured |

**Mark set** uses `HashSet<usize>` — approximately 32 bytes per marked
object. For a 100 MB heap with 1 M small objects, the mark set alone
consumes ~32 MB. A bitmap (1 bit per 8-byte-aligned word) would use
~1.5 MB.

**Pause model:** All phases are stop-the-world. No incremental or
concurrent marking. Acceptable for current workloads but will become the
dominant pause source once interpreter throughput improves.

---

## 3. Hot-Path Bottleneck Detail

### 3.1 Clone Traffic in the Dispatch Loop

The dispatch handlers clone `JsValue` from registers on nearly every
non-trivial operation:

```rust
// dispatch.rs — function call argument marshalling
let arg1 = ctx.frame.read_reg(arg1_v)?.clone();
```

Because `JsValue` is an `enum` with `Rc`-wrapped heap variants,
`.clone()` performs an atomic reference-count increment for every
`String`, `Function`, `Array`, `PlainObject`, `Generator`, etc. On
`Object(*mut HeapObject)` the clone is a raw pointer copy (cheap), but
for the `Rc`-wrapped variants the refcount traffic is significant.

**Measurement opportunity:** The new `jsvalue_clone_*` benchmarks
(added in this branch) isolate the per-variant clone cost.

### 3.2 Polymorphic Inline Cache Thrashing

The interpreter maintains per-feedback-slot polymorphic caches:

```rust
pub poly_load_cache: HashMap<u32, Vec<(usize, JsValue)>>
```

Each slot holds up to **4 entries** searched linearly. When a 5th
distinct object shape accesses the same property slot, the cache
transitions to megamorphic (full `HashMap` lookup on every access).

V8 uses 4 entries too, but transitions to a stub that performs a
generic dictionary lookup — Stator falls back to `proto_lookup` which
walks the prototype chain each time.

### 3.3 Global Variable Access

Every `LdaGlobal` / `StaGlobal`:

```rust
ctx.frame.global_env.borrow().get(name).cloned()
ctx.frame.global_env.borrow_mut().insert(name, val);
```

This requires:
1. `Rc::deref` → pointer chase to `RefCell`
2. `RefCell::borrow()` → runtime borrow-flag check
3. `HashMap::get` → hash + probe
4. `.cloned()` → JsValue clone (see §3.1)

V8 uses **global property cells** with direct pointer loads — one memory
access, no hashing, no borrow checks.

### 3.4 Integer-Indexed Property Insertion

```rust
fn spec_insert_pos(&self, key: &str) -> usize {
    // Linear scan of all existing keys to find sorted position
    while pos < self.keys.len() {
        if let Some(existing) = parse_integer_index(&self.keys[pos]) {
            if existing > n { return pos; }
            pos += 1;
        } else { return pos; }
    }
}
```

Followed by `index_shift_right` which iterates all `HashMap` values.
For `obj[0] = a; obj[1] = b; ... obj[999] = z;` the total work is
**O(N²)** — roughly 500K iterations for 1000 properties.

---

## 4. Optimization Plan

### Phase 1: Quick Wins (1–2 days, est. 2–3× improvement on micro-benchmarks)

#### 4.1 Lazy HashMap Initialization in InterpreterFrame

Replace eager HashMap allocation with `Option<HashMap>` or `Option<Box<HashMap>>`:

```rust
pub mono_load_cache: Option<HashMap<u32, (usize, JsValue)>>,
pub poly_load_cache: Option<HashMap<u32, Vec<(usize, JsValue)>>>,
pub shape_load_ic: Option<HashMap<u32, PropertyIc>>,
pub shape_store_ic: Option<HashMap<u32, PropertyIc>>,
pub string_cache: Option<HashMap<u32, Rc<str>>>,
pub template_cache: Option<HashMap<u32, JsValue>>,
```

Only allocate on first use. Leaf functions that never access named
properties pay zero allocation cost.

**Expected impact:** Eliminates 5 HashMap allocations per function call.

#### 4.2 Reduce Clone in Hot Dispatch Paths

For argument marshalling (`CallUndefinedReceiver0/1/2`), use `SmallVec<[JsValue; 4]>`
(already a workspace dependency) to avoid heap allocation for ≤4 arguments.

For register reads that only need an immutable reference, add a
`read_reg_ref(&self, reg) -> &JsValue` path that avoids cloning.

**Expected impact:** 20–40% fewer refcount operations in call-heavy code.

#### 4.3 Fix Integer-Index Insertion to O(N log N)

Maintain a running count of integer-indexed keys and use binary search
(or a separate sorted `Vec<u32>` of indices) instead of linear scan.
Replace `index_shift_right` with a single re-index pass.

**Expected impact:** `obj[i] = v` in a loop goes from O(N²) to O(N log N).

### Phase 2: Medium-Term (1–2 weeks, est. 3–5× cumulative)

#### 4.4 Global Property Cells

Replace `Rc<RefCell<HashMap<String, JsValue>>>` with a flat `Vec<JsValue>`
indexed by global-variable slot numbers assigned at compile time. The
compiler emits `LdaGlobal(slot_index)` instead of `LdaGlobal("name")`.

**Expected impact:** Global access becomes a single array index instead
of hash + borrow + clone.

#### 4.5 Bitmap Mark Set for GC

Replace `HashSet<usize>` in `MarkSweepCompactor` with a bit-vector
(one bit per aligned heap word). Memory usage drops from ~32 B/object to
~0.125 B/object. Mark and test operations become single bit operations.

**Expected impact:** Major GC pause reduction; lower memory overhead.

#### 4.6 Increase Polymorphic IC to 8 Entries

Widen `poly_load_cache` from 4 to 8 entries per slot. This keeps more
call sites monomorphic/polymorphic before falling to megamorphic.

### Phase 3: Large Architectural Changes (2–4 weeks, est. 5–10× cumulative)

#### 4.7 NaN-Boxing for JsValue

Implement NaN-boxing to reduce `JsValue` from 24 bytes to 8 bytes:

```text
Encoding (64-bit):
  Double:      any bit pattern that is a valid IEEE 754 double
  Integer:     0xFFFF_0000_XXXX_XXXX  (32-bit signed in low bits)
  Pointer:     0x0001_XXXX_XXXX_XXXX  (48-bit pointer in low bits)
  Undefined:   0x7FF8_0000_0000_0001
  Null:        0x7FF8_0000_0000_0002
  Boolean:     0x7FF8_0000_0000_000(3|4)
  TheHole:     0x7FF8_0000_0000_0005
```

This requires a complete audit of every `JsValue` pattern match in the
codebase (~300+ match sites) but provides the single largest throughput
improvement: 3× smaller register files, 3× better cache utilization,
zero-cost "clone" (it's just a u64 copy).

**Dependency:** Must decide how to handle `BigInt(i128)` — likely box it
behind a tagged pointer since BigInt is rare.

#### 4.8 Baseline JIT Compilation

As specified in ADR-002, the baseline JIT compiler maps bytecode 1:1 to
native code, eliminating:
- Function-pointer dispatch overhead (~5–10% per instruction)
- `StatorResult` wrapping on every handler return
- Instruction decoding

The MacroAssembler architecture (x86-64 + AArch64) is designed and has a
9-milestone roadmap. Estimated 3–5× speedup over the interpreter for
compute-bound code.

### Phase 4: V8-Parity Features (ongoing)

#### 4.9 Hidden Classes / Shapes

Replace per-object `PropertyMap` with shared hidden-class chains (V8's
"Map" objects). Objects with the same property sequence share a single
shape descriptor; property access becomes a shape-check + fixed-offset
load.

#### 4.10 Inline Cache Stubs

Generate per-site machine-code stubs that encode the expected shape and
offset, falling back to a generic handler on miss. This is the standard
technique for making property access as fast as a struct field load.

#### 4.11 Concurrent / Incremental GC

Implement incremental marking (budget-based per allocation) and
concurrent sweeping to reduce GC pause times from stop-the-world to
<1 ms on typical heaps.

---

## 5. Benchmark Coverage

The following benchmarks have been added to
`crates/stator_js/benches/engine_benchmarks.rs` to track progress on
the optimizations above:

### Infrastructure Benchmarks (existing)

| Benchmark | What it measures |
|-----------|-----------------|
| `heap_allocate_single` | Steady-state bump-pointer allocation |
| `heap_allocate_burst_1000` | Burst allocation throughput |
| `tagged_smi_round_trip` | TaggedValue SMI encode/decode |
| `tagged_heap_ptr_round_trip` | TaggedValue heap-pointer encode/decode |
| `handle_scope_create_destroy` | HandleScope lifecycle cost |
| `handle_scope_create_100_locals` | Local handle allocation throughput |

### PropertyMap Benchmarks (new)

| Benchmark | What it measures | Optimization target |
|-----------|-----------------|-------------------|
| `property_map_insert_100` | Named property insertion throughput | §4.3 |
| `property_map_lookup_hit` | Inline-cache hit path | §4.6 |
| `property_map_lookup_miss` | HashMap fallback path | §4.6 |
| `property_map_integer_index_insert_1000` | Integer-index O(N²) hotspot | §4.3 |
| `property_map_get_by_offset` | Shape-based offset load (IC simulation) | §4.9 |

### JsValue Benchmarks (new)

| Benchmark | What it measures | Optimization target |
|-----------|-----------------|-------------------|
| `jsvalue_clone_smi` | Inline variant clone (baseline) | §4.7 |
| `jsvalue_clone_string` | Rc refcount bump cost | §4.2 |
| `jsvalue_clone_plain_object` | Rc<RefCell<PropertyMap>> clone | §4.2 |
| `jsvalue_clone_array_100` | Rc<RefCell<Vec>> clone | §4.2 |

### End-to-End JS Benchmarks (new)

| Benchmark | What it measures | Optimization target |
|-----------|-----------------|-------------------|
| `js_property_access_1000` | Named property load in tight loop | §4.6, §4.9, §4.10 |
| `js_function_call_1000` | Function call overhead | §4.1, §4.2 |
| `js_string_concat_500` | String concatenation (Rc<str> traffic) | §4.2, §4.7 |
| `js_array_push_pop_1000` | Array mutation throughput | §4.2 |
| `js_object_creation_500` | Object literal allocation rate | §4.1, §4.9 |
| `js_keyed_property_access_200` | Computed-key property access | §4.3 |
| `js_prototype_chain_lookup_1000` | Prototype chain walk cost | §4.6, §4.9 |
| `js_arithmetic_loop_10000` | Pure integer arithmetic (dispatch bound) | §4.7, §4.8 |
| `js_closure_capture_1000` | Closure variable access overhead | §4.4 |

---

## 6. Comparison with V8 Architecture

| Feature | V8 (Ignition) | Stator | Gap |
|---------|--------------|--------|-----|
| Value representation | 8 B (tagged pointer) | 24 B (Rust enum) | 3× memory |
| Dispatch | Computed goto / asm | Function-pointer table | ~5–10% overhead |
| Property access | Hidden classes + IC stubs | PropertyMap + shape ID | No shared shapes |
| Global access | Property cells (direct load) | `Rc<RefCell<HashMap>>` | 3–4× slower |
| Frame init | Pre-sized from SharedFunctionInfo | 5+ HashMap allocs | Allocation heavy |
| GC mark set | Bitmap (1 bit/word) | `HashSet<usize>` | ~200× memory |
| String intern | String table dedup | Rc<str> (no dedup) | Memory waste |
| JIT tiers | Sparkplug → Maglev → Turbofan | Baseline → Maglev → Turbofan (planned) | Not yet compiled |

---

## 7. Priority Ranking

Optimizations ranked by (estimated speedup × inverse effort):

1. **Lazy HashMap init** — trivial change, immediate 3–5% on call-heavy code
2. **SmallVec for call args** — smallvec already in deps, 2–3% on calls
3. **Integer-index insertion fix** — medium effort, 10× on indexed objects
4. **Global property cells** — medium effort, 10–15% on global-heavy code
5. **Bitmap mark set** — medium effort, GC pause + memory reduction
6. **NaN-boxing** — large effort, 2–3× across the board
7. **Baseline JIT** — large effort, 3–5× on compute-bound code

Items 1–2 can be shipped in a single PR. Items 3–5 are independent and
parallelizable. Items 6–7 are architectural and should be sequenced (NaN-boxing
first, since the JIT can then assume 8-byte values).

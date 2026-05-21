# Startup Snapshot Contract — Warm-Context Design for Edge

| Field    | Value                              |
|----------|------------------------------------|
| Status   | Draft (design only — no code yet)  |
| Audience | Stator engine + Edge embedder team |
| Scope    | `crates/stator_jse/src/snapshot/`, `crates/st8`, future `stator_ffi` snapshot FFI |

This document specifies the **warm-context startup snapshot** that Stator must
provide to support Edge's mksnapshot-equivalent embedding flow.  It describes
what exists today, what must change, and the rules a future implementation must
respect.  Nothing in this document is yet implemented; concrete code work is
tracked by separate slices/todos referenced at the end.

---

## 1. Current capability and limitations

The shipping snapshot subsystem (`crates/stator_jse/src/snapshot/mod.rs`,
binary format `STSS` v2) provides:

- A `StartupSnapshot` opaque blob with `from_bytes` / `as_bytes`,
  `write_to_file` / `read_from_file`, and a cheap `validate()` that checks
  magic, version, and FNV-1a-32 footer checksum.
- `serialize_globals(&HashMap<String, JsValue>) -> StartupSnapshot` and
  `deserialize_globals(&[u8]) -> StatorResult<HashMap<String, JsValue>>`.
- A tagged JsValue encoding covering Undefined, Null, Boolean, Smi,
  HeapNumber, String, Symbol (as bare `u64`), BigInt (`i128`), Function
  (full `BytecodeArray` including constant pool, source positions,
  feedback metadata, handler table, generator flag), Array, PlainObject,
  Error (`ErrorKind` + message), and a `NativeFunction` *placeholder*
  (string name only).
- `DefineRef` / `BackRef` tags that preserve shared and circular Rc
  identity for `PlainObject`, `Array`, `Function`, `Error`.
- `serialize_bytecode_array` / `deserialize_bytecode_array` helpers used
  by the FFI bytecode-cache surface (`stator_ffi`).

The CLI driver (`crates/st8/src/main.rs`) exposes the flow as
`st8 --emit-snapshot=<path>` (writes the bootstrapped global map) and
`st8 --snapshot=<path> file.js` (loads the map and skips bootstrap).

### Hard limitations today

1. **Globals-only.** Only the contents of the
   `GlobalEnv` `HashMap<String, JsValue>` round-trip.  There is no concept
   of a saved *Context*, *Realm*, intrinsics table, prototype chain
   topology, hidden class / shape registry, builtins object identity, or
   module cache.
2. **`JsValue::Object` (raw GC pointer), `Generator`, and `Iterator`
   silently degrade to `Undefined`.**  Round-tripping the bootstrapped
   global object is therefore lossy — any builtin currently represented
   as `Object` is lost, and on reload would have to be re-bootstrapped.
3. **Native callbacks are erased.** `NativeFunction` values become
   no-op closures returning `Undefined`; the embedder is expected to
   re-install host bindings by name post-load.  There is no
   well-defined "name → callback" registry, no template/accessor
   restoration, and no validation that an embedder actually reinstalled
   every required slot.
4. **No version metadata beyond `u32` format version.**  There is no
   engine build id, no feature-flag fingerprint, no toolchain id, no
   architecture/endianness/pointer-width guard, and no embedder
   isolate-data version.  A snapshot built for one Stator build can be
   loaded into a structurally incompatible one as long as the format
   version matches.
5. **No FFI surface.** `stator_ffi` does not expose any
   snapshot-create / snapshot-load entry points; Edge cannot consume the
   current capability without linking Rust directly.
6. **No determinism guarantees.** Serialization iterates a
   `HashMap`, so byte output is not stable across runs.  This blocks
   reproducible-build verification, deterministic golden tests, and
   content-addressed packaging.

These limitations are why a new *warm-context* contract is needed before
Edge can ship Stator behind a snapshot boundary equivalent to V8
`mksnapshot` / `SnapshotCreator`.

> **Implemented today (legacy `STSS`):** the engine now exposes
> `snapshot::serialize_globals_strict`, a strict-mode counterpart to the
> legacy lossy `serialize_globals`.  It fails closed with the structured
> [`StatorError::SnapshotUnsupportedValue { class, path, reason }`]
> diagnostic the first time it encounters any heap value class the v2
> binary format cannot losslessly capture (`NativeFunction`, raw
> `Object` GC pointers, `Generator`, `Iterator`, `Promise`, `Context`,
> `Proxy`, `ArrayBuffer`, `TypedArray`, `DataView`, `TheHole`), including
> when reached transitively through `Array`, `PlainObject`, or
> `ModuleBinding` values.  This matches the rejection policy specified
> for the future `STWC` warm-context format and is the migration target
> for embedders that must never silently downgrade unsupported state.

---

## 2. Goals and non-goals

### Goals (v1 warm-context snapshot)

- Serialize and restore a *Context* (≈ realm) such that, after load,
  scripts observe the same set of intrinsics, global properties,
  prototypes, and identity relationships they would observe after a
  cold bootstrap of the same engine build.
- Preserve **identity invariants** required by language semantics:
  e.g. `Array.prototype === Object.getPrototypeOf([])`, exactly one
  `%TypedArrayPrototype%`, the canonical `%ThrowTypeError%` poisoned
  function, well-known symbols (`@@iterator` etc.) with stable
  identity.
- Provide a deterministic byte output: identical inputs on identical
  engine builds produce byte-identical snapshots, so the result can be
  shipped as a reproducible build artefact and content-hashed.
- Provide a strong **compatibility envelope** so loading a snapshot
  produced by a different engine build, target arch, or feature
  configuration fails closed rather than producing UB.
- Provide a **callback reinstall contract**: a documented, enumerable
  set of host slots whose absence after load is a fatal error rather
  than a silent no-op.
- Expose the create/load flow through `stator_ffi` so Edge can drive
  it without depending on Rust.

### Non-goals (v1)

- **Snapshotting a running heap** (i.e. mid-execution, with stacks /
  generators / async tasks / pending promise jobs).  Snapshots are
  taken at well-defined quiescent points: end of bootstrap and (later)
  end of a user-provided "warm-up" script.
- **Cross-build portability.**  A snapshot is bound to its producing
  engine build by an opaque fingerprint; there is no compatibility
  story across builds.  Edge will rebuild snapshots as part of its
  build, exactly like V8 `mksnapshot`.
- **Live mutation of an already-loaded snapshot.** Snapshots are
  immutable inputs to `Isolate::new_with_snapshot(...)`.
- **Wasm module / compiled-code snapshotting.** Out of scope for v1;
  bytecode-cache (already wired through `stator_ffi`) covers the
  compiled-script use-case separately.
- **Snapshotting host-owned objects** (DOM wrappers, Edge bindings).
  Host objects are stripped at snapshot time and must be re-attached
  by the embedder using the callback reinstall contract.

---

## 3. What is serializable vs. rejected

The new contract is strict: anything the serializer cannot faithfully
round-trip is **rejected** (returns an error from
`create_snapshot(...)`), not silently degraded.

### Serializable (must round-trip with identity preserved where noted)

- Primitives: `Undefined`, `Null`, `Boolean`, `Smi`, `HeapNumber`,
  `String`, `BigInt`.
- `Symbol` — identity-preserving for the well-known symbol set, plus
  user-created symbols reachable from the global object.  Encoded by a
  realm-local symbol table, not raw `u64`.
- `PlainObject`, `Array`, sealed/frozen variants — including shape /
  hidden class identity (see §5 builtins determinism).
- `Function` with a bytecode body (`BytecodeArray`), preserving
  constant pool, source positions, feedback metadata, handler table,
  generator flag.
- `Error` instances created during bootstrap (kind + message + stack
  string only; live stack frames are not captured).
- Cross-references and cycles via `DefineRef` / `BackRef`.
- The realm's intrinsics table and the global object, as a single
  reachable graph rooted at the Context.

### Rejected at serialize time (snapshot creation fails)

- `JsValue::Object` raw GC pointers that are not also reachable as a
  typed object (i.e. host-opaque heap cells).
- Live `Generator` and `Iterator` instances with pending state.
- `NativeFunction` values whose `name` is not registered in the
  embedder's *callback manifest* (see §4).
- Any value transitively reachable from a `Proxy` whose handler is a
  native function not on the manifest.
- WeakRef / WeakMap / WeakSet entries — v1 serializes the collection
  shell empty and rejects if non-empty.
- FinalizationRegistry entries — same rule.
- Pending microtasks, promise jobs, unresolved Promises.
- Module records (resolved or unresolved) — module cache must be empty
  at snapshot time in v1.
- Wasm module / instance / memory / table / global.
- ArrayBuffer with non-zero byte length whose backing store is
  embedder-provided (detached or external buffers).  Plain
  internally-allocated buffers may be supported in a later revision;
  v1 rejects.

Rejection produces a `StatorError::Snapshot` with a path describing the
offending object (`"global.foo.bar[3] -> NativeFunction(\"unknown\")"`)
so embedders can fix their bootstrap script.

---

## 4. Native callback / template reinstall plan

The single biggest gap today is that `NativeFunction` values become
silent no-ops on load.  The v1 contract replaces this with an
explicit, fail-closed protocol.

### Callback manifest

An embedder constructs a `SnapshotCallbackManifest` *before* calling
`create_snapshot`.  The manifest is an ordered list of stable string
ids; for each id the embedder also registers a native callback pointer
+ optional data pointer.  Example (conceptual Rust):

```text
manifest.register("edge.console.log",   console_log_thunk,   null);
manifest.register("edge.fetch",         fetch_thunk,         null);
manifest.register("edge.crypto.random", crypto_random_thunk, null);
```

During serialization every `NativeFunction` is replaced by its
manifest id (a `str32`).  Serialization fails if any reachable
`NativeFunction` has no manifest entry.

During load the embedder provides a *matching* manifest.  The loader:

1. Verifies that **every** id present in the snapshot is registered in
   the load-time manifest.  Missing ids = fatal load error.
2. Replaces each placeholder with a fresh `JsValue::NativeFunction`
   pointing at the load-time callback + data pointer.
3. Verifies that the load-time manifest registers **no extra** ids
   beyond what the snapshot uses, unless explicitly marked
   `allow_extra = true` (Edge expects to allow this for forward
   compatibility).

### Templates / accessors / interceptors

Future work — once Stator gains a V8-style `FunctionTemplate` /
`ObjectTemplate` surface (currently it does not), the manifest will be
extended with template ids using the same fail-closed model.  v1
explicitly rejects accessor properties whose getter/setter is a
`NativeFunction` unless both are on the manifest.

### Internal fields

Objects with internal fields (none exist in `stator_jse` today, but
the FFI exposes the concept via wrapper objects) are rejected by v1
unless every internal field is either (a) `Undefined` or (b) a value
representable by the manifest as an *external reference id* — same
mechanism as callbacks, but for raw `*mut c_void` pointers owned by
the embedder.

---

## 5. Builtins and global-object determinism

v1 requires **deterministic byte output** for identical
(engine-build, bootstrap-script, manifest) inputs.  The following
sources of nondeterminism in the current code must be eliminated:

1. **HashMap iteration order.**  Replace ad-hoc `HashMap` walks with a
   canonical traversal: own properties enumerated in insertion order
   (the language-observable order), intrinsics enumerated in a
   hard-coded order matching the bootstrap sequence.
2. **Rc address-based ref ids.**  Ref ids today are assigned by
   first-encounter during a `HashMap` walk.  They must instead be
   assigned by first-encounter during the canonical traversal so the
   same graph always gets the same numbering.
3. **Symbol identity.**  Allocate a per-realm canonical symbol table
   that lists well-known symbols first (in spec order) and then any
   user-created symbols in creation order.  Encode by index.
4. **Hidden class / shape identity.**  Shapes are part of the
   identity-preserving graph; the serializer walks the shape tree and
   assigns deterministic ids in DFS order from the empty shape.
5. **Floating-point payload bytes.**  `HeapNumber` is written as raw
   little-endian IEEE-754; NaN payloads must be canonicalised to a
   single bit pattern at snapshot time so that any non-canonical NaNs
   produced by the JIT do not leak into the artefact.
6. **String interning.**  All `String` values share a per-snapshot
   string table indexed by deduplicated content; the table is written
   first, then values reference table indices.  This both reduces
   size and removes any nondeterminism from per-Rc string sharing.
7. **Bootstrap script reproducibility.**  The bootstrap script and
   any embedder warm-up script must themselves be deterministic
   (no `Date.now`, `Math.random` seeding from time, no environment
   reads).  v1 documents this as an embedder responsibility; v2 may
   add a "deterministic-mode" Isolate flag that poisons these APIs
   during snapshot capture.

A reproducibility test (§9) asserts byte equality across two
back-to-back snapshots of the same input.

---

## 6. Security and sandbox boundaries

A snapshot is *executable data*: a malicious or corrupted blob loaded
into the engine can corrupt the heap.  The contract:

- **Snapshots are trust-boundary inputs only when the embedder vouches
  for them.**  The default load entry point is named
  `load_snapshot_trusted(...)` to make this explicit at every call
  site.
- **Integrity:** every snapshot carries a 32-byte BLAKE3 digest of all
  bytes preceding the digest, replacing the existing FNV-1a-32 footer
  for v1.  FNV is preserved for the legacy globals-only format.
- **Build binding:** every snapshot carries a `BuildId` (32 bytes,
  derived at engine build time from a hash of `cargo build`
  inputs that affect ABI: `stator_jse` source rev, `rustc -vV`,
  target triple, feature flags, `STATOR_FFI_ABI_VERSION`).  Loading
  refuses to proceed unless the embedder's `BuildId` matches byte for
  byte.
- **Bounds-checking deserializer:** every length-prefixed field is
  bounded against remaining-bytes before allocation; no length is
  used as a `Vec::with_capacity` argument without a sanity ceiling
  (configurable, default 256 MiB total snapshot size, 16 MiB per
  string, 1 M entries per collection).
- **No code execution during load.**  Deserialization is data-only;
  the loader never invokes user-visible JS, never calls user-supplied
  callbacks (the callback manifest is consulted by id only — the
  pointer is installed but not called).  This rule is enforced by an
  assertion in the test suite.
- **Pointer provenance:** raw pointers (callback function pointers,
  external-reference data pointers) live exclusively in the
  load-time manifest, never in the snapshot bytes.
- **Manifest hash:** the snapshot stores a hash of the *id set* used
  by the manifest at create time; load refuses to proceed if the
  load-time manifest's id set hash differs and `allow_extra` is not
  set.
- **Untrusted-blob path (post-v1):** if an embedder must accept
  snapshots over the network, a future `load_snapshot_untrusted` API
  will run the bytes through an additional structural validator + a
  per-object size cap and reject anything that escapes the JS object
  shape grammar.  Not in v1 scope.

---

## 7. Compatibility / version metadata

Snapshot compatibility is **fail-closed**.  A loader must reject the
blob before allocating payload storage, creating JS objects, installing
callbacks, or running any bytecode unless every compatibility key below
matches the current engine and embedder environment exactly.  There is
no best-effort or warning-only mode for production loads.

The v1 binary header is extended (new magic `STWC` — *Stator Warm
Context* — to keep the legacy `STSS` loader available unchanged):

```text
Header (fixed):
  magic               : [u8; 4]   = b"STWC"
  snapshot_format_ver : u32 LE    = 1
  bytecode_format_ver : u32 LE    (current BytecodeArray encoding)
  engine_crate_ver    : str32     (stator_jse Cargo.toml version)
  ffi_crate_ver       : str32     (stator_ffi Cargo.toml version)
  commit_id           : str32     (git SHA or "unknown")
  build_id            : [u8; 32]  (BLAKE3 fingerprint of engine build)
  ffi_abi_version     : u32 LE    (mirrors STATOR_FFI_ABI_VERSION)
  target_triple       : str32     (rustc target triple)
  os                  : str16     (target OS component)
  arch                : str16     (target arch component)
  pointer_width       : u8        (4 or 8)
  endianness          : u8        (1 = little, 2 = big)
  cargo_profile       : str16     (dev, release, custom profile name)
  build_features_hash : [u8; 32]  (BLAKE3 of sorted enabled Cargo cfg/features)
  jit_tiering_hash    : [u8; 32]  (BLAKE3 of enabled JIT/tiering modes)
  cpu_features_hash   : [u8; 32]  (BLAKE3 of required CPU feature set)
  manifest_hash       : [u8; 32]  (BLAKE3 of sorted native callback id list)
  edge_release_hash   : [u8; 32]  (BLAKE3 of Edge vendored metadata; zero if unused)
  payload_len         : u64 LE
Payload:
  …deterministic encoding of the context graph…
Footer:
  digest          : [u8; 32]  (BLAKE3 of header + payload)
```

### Field-specific rejection rules

Every field above is checked by the loader before any allocation beyond
the header itself.  Every mismatch returns a typed diagnostic that names
the field, the snapshot value, and the runtime value.  The diagnostic is
also surfaced through the C ABI status detail string so Edge telemetry
can bucket failures without parsing human prose.

| Field(s) | Match rule | Required diagnostic |
|----------|------------|---------------------|
| `magic` | Must be `STWC`; `STSS` is accepted only by the legacy globals-only loader. | `SnapshotError::MagicMismatch { found, expected }` |
| `snapshot_format_ver` | Must equal the loader's warm-context snapshot format version. | `SnapshotError::SnapshotFormatMismatch { found, expected }` |
| `bytecode_format_ver` | Must equal the engine bytecode encoding version used by `BytecodeArray`. | `SnapshotError::BytecodeFormatMismatch { found, expected }` |
| `engine_crate_ver`, `ffi_crate_ver` | Must equal the linked `stator_jse` and `stator_ffi` crate versions. | `SnapshotError::CrateVersionMismatch { crate_name, found, expected }` |
| `commit_id`, `build_id` | `build_id` must match byte-for-byte; `commit_id` is diagnostic metadata and must match unless the runtime was built with an explicit local-development override. | `SnapshotError::BuildMismatch { found_build_id, expected_build_id, found_commit, expected_commit }` |
| `ffi_abi_version` | Must equal `STATOR_FFI_ABI_VERSION`; any ABI bump invalidates existing snapshots. | `SnapshotError::AbiMismatch { found, expected }` |
| `target_triple`, `os`, `arch`, `pointer_width`, `endianness` | Must equal the current target environment exactly. | `SnapshotError::TargetMismatch { field, found, expected }` |
| `cargo_profile` | Must equal the profile that built the runtime.  Release and debug snapshots are not interchangeable because assertions, layout choices, and JIT settings may differ. | `SnapshotError::CargoProfileMismatch { found, expected }` |
| `build_features_hash` | Must equal the hash of enabled Cargo features and relevant `cfg` values (`jit`, `wasm`, `intl`, `inspector`, GC strategy, allocator, panic mode). | `SnapshotError::BuildFeaturesMismatch { found, expected }` |
| `jit_tiering_hash` | Must equal the configured JIT/tiering feature set, including interpreter-only mode, baseline JIT, optimizing JIT, inline-cache policy, and codegen backend. | `SnapshotError::JitTieringMismatch { found, expected }` |
| `cpu_features_hash` | Must be exactly equal for snapshots containing native/JIT-sensitive metadata.  A future portable-interpreter-only mode may define a weaker subset check, but v1 does not. | `SnapshotError::CpuFeaturesMismatch { found, expected }` |
| `manifest_hash` | Must equal the BLAKE3 hash of the sorted native callback manifest ids used at load time, unless `allow_extra` is explicitly set; missing ids are always fatal. | `SnapshotError::ManifestMismatch { missing_ids, extra_ids, found_hash, expected_hash }` |
| `edge_release_hash` | If non-zero, must equal the Edge vendored release metadata hash supplied by the embedder. | `SnapshotError::EdgeReleaseMismatch { found, expected }` |
| `payload_len`, `digest` | Length must fit configured caps; digest must verify over the exact header and payload bytes. | `SnapshotError::PayloadLengthExceeded { found, max }` or `SnapshotError::DigestMismatch { found, expected }` |

The legacy `STSS` v2 format (today's globals-only snapshot) is kept
working for `st8`'s existing flow; new embedders MUST use `STWC`.

### Edge vendored release metadata

Edge consumes Stator as a vendored component, so the compatibility
envelope also binds snapshots to the release metadata Edge already
records for reproducibility.  The Edge-side package must provide a
canonical metadata document containing at least: Stator repository URL,
vendored commit, crate versions, generated `stator.h` version,
`STATOR_FFI_ABI_VERSION`, Rust toolchain, target triple, Cargo profile,
enabled Cargo features, JIT/tiering configuration, CPU baseline, and the
native callback manifest hash.  The snapshot header stores
`edge_release_hash = BLAKE3(canonical-json(metadata))`.

During load, Edge passes the same canonical metadata to the FFI loader.
If the snapshot header contains a non-zero `edge_release_hash`, the
loader rejects any mismatch with `SnapshotError::EdgeReleaseMismatch`.
If the field is zero, only non-Edge embedders may continue; Edge release
and prepublish gates must reject zero-valued release hashes so a shipped
snapshot is always traceable to the exact vendored Stator artefact that
created it.

---

## 8. Proposed FFI / API shape (high level)

Final names are pending; this section fixes the *shape* the
implementation slices must hit.

### Rust (engine-internal)

> **v1 implementation status (in tree):** the in-process Rust API
> below is implemented today in `crates/stator_jse/src/snapshot/`:
> [`SnapshotCallbackManifest`](../crates/stator_jse/src/snapshot/manifest.rs)
> plus
> [`serialize_globals_with_manifest`](../crates/stator_jse/src/snapshot/mod.rs)
> and
> [`reinstall_globals_with_manifest`](../crates/stator_jse/src/snapshot/mod.rs).
> The implemented blob uses the magic `STSM` and embeds a 32-byte
> deterministic manifest digest (four domain-separated FNV-1a-64
> folds) plus the sorted id list directly in the header.  Load is
> strictly fail-closed: any mismatch — missing id, extra id, renamed
> id, tampered digest — produces
> [`StatorError::SnapshotManifestMismatch`] before any callback is
> reinstalled.  An `allow_extra` mode is not implemented in v1.  The
> Edge release-bundle BLAKE3 digests described in §7 and §10 layer
> on top of this in-process digest at the file/bundle level.
>
> The full `Isolate` / `Context` constructor surface described in
> the pseudo-Rust block below remains a future work item.

```text
pub struct SnapshotCallbackManifest { … }
impl SnapshotCallbackManifest {
    pub fn new() -> Self;
    pub fn register(&mut self, id: &str, cb: NativeCallbackPtr, data: *mut c_void);
    pub fn id_set_hash(&self) -> [u8; 32];
}

pub fn create_snapshot(
    isolate: &Isolate,
    context: &Context,
    manifest: &SnapshotCallbackManifest,
) -> Result<WarmContextSnapshot, SnapshotError>;

pub struct WarmContextSnapshot { /* opaque, owns Vec<u8> */ }
impl WarmContextSnapshot {
    pub fn as_bytes(&self) -> &[u8];
    pub fn write_to_file(&self, path: &Path) -> Result<(), SnapshotError>;
    pub fn read_from_file(path: &Path) -> Result<Self, SnapshotError>;
    pub fn validate_header(&self) -> Result<SnapshotHeader, SnapshotError>;
}

impl Isolate {
    pub fn new_with_snapshot(
        snapshot: &WarmContextSnapshot,
        manifest: &SnapshotCallbackManifest,
    ) -> Result<Self, SnapshotError>;
}
```

### C ABI (`stator_ffi`)

Mirroring the conventions in `crates/stator_ffi/src/lib.rs`:

```text
typedef struct StatorSnapshotManifest StatorSnapshotManifest;
typedef struct StatorSnapshot         StatorSnapshot;

StatorSnapshotManifest* stator_snapshot_manifest_create(void);
void                    stator_snapshot_manifest_destroy(StatorSnapshotManifest*);
StatorStatus            stator_snapshot_manifest_register(
                            StatorSnapshotManifest*,
                            const char* id,
                            StatorNativeCallback cb,
                            void* data);

StatorStatus            stator_snapshot_create(
                            StatorIsolate*,
                            StatorContext*,
                            const StatorSnapshotManifest*,
                            StatorSnapshot** out);
void                    stator_snapshot_destroy(StatorSnapshot*);

const uint8_t*          stator_snapshot_bytes(const StatorSnapshot*, size_t* out_len);
StatorStatus            stator_snapshot_from_bytes(const uint8_t*, size_t, StatorSnapshot** out);

StatorStatus            stator_isolate_create_with_snapshot(
                            const StatorSnapshot*,
                            const StatorSnapshotManifest*,
                            StatorIsolate** out);
```

All names are prefixed `stator_`, all opaque types use
`#[repr(C)] _opaque: [u8; 0]`, the header is regenerated by
`cbindgen`, and `STATOR_FFI_ABI_VERSION` is bumped exactly once when
this surface lands.  No FFI symbol added here may be called from
the load path before header validation succeeds.

### CLI (`st8`)

Add `st8 --emit-warm-snapshot=<path> [--warmup=<file.js>]` and
`st8 --warm-snapshot=<path> <file.js>`.  Existing
`--emit-snapshot` / `--snapshot` continue to drive the legacy
globals-only path so existing scripts and golden tests are
untouched.

---

## 9. Test plan

The implementation slices must land tests in `crates/stator_jse`,
`crates/stator_ffi`, and `crates/st8` covering:

1. **Round-trip parity:** for every JS expression class
   (primitive, object, array, function, error, symbol, bigint, map,
   set, typed array, well-known symbol), create a snapshot, load it,
   evaluate `assertDeepEqual(restored, original)` in the loaded
   context.  Driven by a fixture catalogue under
   `crates/stator_jse/tests/snapshot_fixtures/`.
2. **Identity preservation:** assert
   `Array.prototype === Object.getPrototypeOf([])` after load,
   `Symbol.iterator === restored.Symbol.iterator`, single
   `%ThrowTypeError%`, prototype-chain pointer equality.
3. **Determinism:** create two snapshots back-to-back from the same
   Isolate; assert byte equality of the entire blob (header +
   payload + digest).  Add a third run that mutates the bootstrap
   script and assert digest *changes*.
4. **Reject paths:** for every rejected category in §3 (live
   generator, unregistered native callback, non-empty WeakMap,
   pending module, non-empty microtask queue, …) assert
   `create_snapshot` returns the correct typed error with a path
   string pointing at the offending value.
5. **Header / fingerprint validation:** load attempts with a
   mutated `build_id`, `ffi_abi_version`, `arch`, `pointer_width`,
   `endianness`, `feature_flags`, `manifest_hash`, or `digest` byte
   each produce the matching `SnapshotError::*Mismatch` variant.
6. **Fuzzing:** add a `fuzz/` target that mutates a valid `STWC`
   blob and asserts `Isolate::new_with_snapshot` either loads
   cleanly or returns a typed error — never panics, never UB,
   never executes user JS.  Run with `cargo +nightly fuzz run
   snapshot_load` in CI on the schedule already used by the other
   fuzz targets.
7. **No-code-execution invariant:** instrument the loader with a
   debug-only assertion that no JS callback is invoked during
   load; flip the assertion under `cfg(test)` for the test run.
8. **FFI surface:** mirror the Rust round-trip test in C from
   `examples/mini_browser/` so the published header is exercised.

---

## 10. Release artefact expectations

Edge consumes Stator as a vendored component, so every tagged Stator
release that ships a warm-context snapshot MUST publish a fully
self-describing artefact bundle that Edge's vendoring tooling can
ingest without out-of-band coordination.  The bundle is the *only*
supported source of warm-context snapshots used in shipping Edge
builds; ad-hoc snapshots produced on developer machines MUST NOT be
checked into Edge's vendored tree.

### 10.1 Artefact bundle layout

A release tag `vX.Y.Z` produces a single bundle published as a GitHub
release asset and mirrored to Edge's internal package store.  Per
supported target triple the bundle is named:

```text
stator-warm-snapshot-<crate_ver>-<target_triple>-<profile>.tar.zst
```

where `<crate_ver>` is the `stator_jse` Cargo version (which must
equal `stator_ffi`'s version at release time), `<target_triple>` is a
rustc target triple (e.g. `x86_64-pc-windows-msvc`,
`aarch64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`,
`aarch64-apple-darwin`), and `<profile>` is `release` or
`release-edge` (the Edge-tuned Cargo profile defined in the workspace
`Cargo.toml`).  Bundles for `dev`/`debug` profiles MUST NOT be
published; they are rejected by the prepublish gate.

The decompressed bundle has a fixed directory layout:

```text
stator-warm-snapshot-<crate_ver>-<target_triple>-<profile>/
├── manifest.json                  # canonical metadata, see §10.2
├── manifest.json.blake3           # BLAKE3 digest of manifest.json (lower-hex)
├── manifest.json.sig              # detached signature, see §10.4
├── stator.h                       # cbindgen-generated header for this release
├── stator.h.blake3
├── CHANGELOG.md                   # release notes section for this tag
├── snapshots/
│   ├── edge-bootstrap.stwc        # canonical warm-context snapshot
│   ├── edge-bootstrap.stwc.blake3
│   ├── edge-bootstrap.stwc.sig
│   ├── edge-bootstrap.manifest.json  # callback manifest source (sorted)
│   └── edge-bootstrap.warmup.js   # exact bootstrap script used
├── callback-manifest/
│   ├── manifest.toml              # human-editable manifest source
│   └── manifest.blake3            # mirrors header `manifest_hash` value
└── validation/
    ├── validate.ps1               # Windows verification entry-point
    ├── validate.sh                # POSIX verification entry-point
    └── expected-header.json       # decoded header expected by validators
```

All filenames are stable across releases.  Tooling that needs to
locate a specific file MUST do so by path, not by globbing, so future
additions cannot silently shadow expected files.

Bundles for additional snapshot variants (for example, an extra
warm-context produced from a different bootstrap script) are placed
side-by-side in `snapshots/<variant>.stwc` with matching `.blake3`,
`.sig`, `.manifest.json`, and `.warmup.js` siblings.  Every variant
is independently listed in `manifest.json` (see §10.2).

### 10.2 `manifest.json` fields

`manifest.json` is the single source of truth for the bundle.  It is
canonicalised as JCS (RFC 8785) before hashing so the BLAKE3 digest
and signature are reproducible.  Required top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | Always `"stator.warm-snapshot.manifest/v1"`. |
| `release_tag` | string | Git tag, e.g. `"v0.4.0"`. |
| `released_at` | string | RFC 3339 UTC timestamp of the build. |
| `repository_url` | string | Canonical Stator repo URL. |
| `commit_id` | string | Full 40-char git SHA of the release commit. |
| `crate_versions.stator_jse` | string | semver of `stator_jse`. |
| `crate_versions.stator_ffi` | string | semver of `stator_ffi`. |
| `crate_versions.st8` | string | semver of `st8` at this tag. |
| `ffi_abi_version` | integer | Decimal value of `STATOR_FFI_ABI_VERSION`. |
| `snapshot_format_version` | integer | `STWC` header `snapshot_format_ver`. |
| `bytecode_format_version` | integer | `STWC` header `bytecode_format_ver`. |
| `rust_toolchain` | string | Output of `rustc -V` used for the build. |
| `cargo_profile` | string | `release` or `release-edge`. |
| `target_triple` | string | rustc target triple matching the bundle name. |
| `os` | string | Target OS component (`windows`, `linux`, `macos`). |
| `arch` | string | Target arch component (`x86_64`, `aarch64`). |
| `pointer_width` | integer | 4 or 8. |
| `endianness` | string | `"little"` or `"big"`. |
| `build_id` | string | Lower-hex BLAKE3 fingerprint of the engine build (mirrors `STWC` header `build_id`). |
| `build_features_hash` | string | Lower-hex BLAKE3, mirrors `STWC` header. |
| `jit_tiering_hash` | string | Lower-hex BLAKE3, mirrors `STWC` header. |
| `cpu_features_hash` | string | Lower-hex BLAKE3, mirrors `STWC` header. |
| `cpu_baseline` | object | Human-readable enumeration of required CPU features (e.g. `{"x86_64": ["sse4.2", "popcnt"]}`); informational, hashed into `cpu_features_hash`. |
| `enabled_features` | array of strings | Sorted Cargo features enabled at build time. |
| `enabled_cfgs` | array of strings | Sorted non-default `rustc --cfg` keys (e.g. `gc=marksweep`, `inspector`). |
| `callback_manifest.hash` | string | Lower-hex BLAKE3 mirroring `STWC` header `manifest_hash`. |
| `callback_manifest.ids` | array of strings | Sorted callback ids included in the manifest. |
| `callback_manifest.source_path` | string | Path within the bundle to the manifest source file. |
| `snapshots[]` | array | One entry per `.stwc` file in `snapshots/`. |
| `snapshots[].name` | string | File basename, e.g. `"edge-bootstrap.stwc"`. |
| `snapshots[].purpose` | string | Short slug, e.g. `"edge-bootstrap"`. |
| `snapshots[].bytes` | integer | File size in bytes. |
| `snapshots[].blake3` | string | Lower-hex BLAKE3 of the file. |
| `snapshots[].header_digest` | string | Lower-hex BLAKE3 of the snapshot's footer `digest` field (mirrors the `STWC` footer). |
| `snapshots[].warmup_script` | string | Path within the bundle to the exact `.js` source replayed to create the snapshot. |
| `snapshots[].warmup_script_blake3` | string | BLAKE3 of the warmup script. |
| `header_digest` | string | Lower-hex BLAKE3 of the canonicalised manifest *excluding* the `header_digest`, `signatures`, and `edge_release_hash` fields.  This is the value Edge feeds back into the loader as the embedder-supplied release metadata hash and is what `STWC` header `edge_release_hash` MUST equal. |
| `edge_release_hash` | string | Equal to `header_digest`; published as a separate field so Edge tooling can lift it without re-hashing. |
| `signatures[]` | array | Detached signatures over `manifest.json`'s canonical bytes, see §10.4. |
| `revoked` | boolean | `false` at publish time.  Flipped to `true` by a follow-up patch release if the bundle is recalled (see §10.6). |

`manifest.json` MUST NOT contain any field not listed above; unknown
fields cause the prepublish gate and Edge ingestion to reject the
bundle.

### 10.3 Validation commands

Every bundle is self-validating.  After extracting, an Edge build
agent runs (Windows):

```powershell
# 1. Recompute and verify the manifest digest and signature.
.\validation\validate.ps1 -BundleRoot .

# 2. Verify every per-file digest listed in manifest.json.
.\validation\validate.ps1 -BundleRoot . -CheckFiles

# 3. Load the reference snapshot into a freshly built engine and
#    assert the typed-no-op invariant (no JS executed during load).
.\validation\validate.ps1 -BundleRoot . -Smoke `
    -StatorLibDir $env:STATOR_LIB_DIR
```

POSIX agents run the equivalent `validation/validate.sh`.  Both
scripts:

1. Recompute BLAKE3 of `manifest.json` and compare with
   `manifest.json.blake3`.
2. Verify `manifest.json.sig` against the published Stator release
   public key set (see §10.4).
3. Iterate `snapshots[]`, recompute BLAKE3 for each file and compare
   with both `manifest.json` and the `<file>.blake3` sibling.
4. Decode each `.stwc` header, compare every field with the
   corresponding `manifest.json` value, and emit a typed diagnostic
   identifying the first mismatch (mirroring §7 error variants).
5. (`-Smoke` only) invoke `st8 --warm-snapshot=<path> --no-eval` to
   exercise `stator_snapshot_from_bytes` +
   `stator_isolate_create_with_snapshot` against the just-built FFI,
   asserting that loading succeeds and that no user JS callback was
   invoked during load.
6. Exit with status `0` only when every check passes; any failure
   exits non-zero with the matching `SnapshotError::*` variant name
   on the last stderr line so CI can bucket failures.

The scripts MUST be hermetic — no network calls, no environment
inference beyond `STATOR_LIB_DIR` for the smoke step — so any
embedder can reproduce the same verification offline.

### 10.4 Digests, signing, and key management

- Every artefact (`manifest.json`, `stator.h`, every `.stwc`, every
  `.warmup.js`, every callback `manifest.toml`) ships with a
  `<name>.blake3` sibling containing the lower-hex BLAKE3 digest of
  the file.  Digests are computed over the raw file bytes.
- `manifest.json` and every `.stwc` are additionally signed with
  detached Ed25519 signatures stored alongside as `<name>.sig`.  The
  same signatures are duplicated inside `manifest.json`'s
  `signatures[]` array (so the manifest carries its own provenance
  even when distributed without sidecar files) using objects of the
  form `{ "key_id": "<hex>", "algorithm": "ed25519", "target":
  "<relative-path>", "signature": "<base64>" }`.
- Signing keys are owned by the Stator release team and rotated
  yearly.  The public key set is checked into the Stator repo at
  `docs/release-keys/` and mirrored to Edge's vendored tree so
  offline validation never depends on network access.
- The prepublish gate refuses to upload a bundle whose manifest
  references a `key_id` not present in `docs/release-keys/`.

### 10.5 Revendor flow

Edge revendors Stator using the following deterministic procedure;
it is the only supported way to consume a new warm-context snapshot:

1. Download the bundle for every supported target triple from the
   Stator release page (or the mirror).
2. Verify each bundle with `validation/validate.sh -Smoke`.  Any
   failure aborts the revendor with a non-zero exit code; partial
   updates are not allowed.
3. Replace the entire `third_party/stator/` directory in Edge's
   source tree with the new bundle contents; never merge files
   from two different releases.
4. Update Edge's component metadata (`third_party/stator/README.chromium`
   plus the Edge-internal `stator_release.gni`) with the manifest's
   `release_tag`, `commit_id`, `crate_versions`, `ffi_abi_version`,
   `snapshot_format_version`, `build_id`, `manifest_hash`, every
   per-target `header_digest`, and the bundle's BLAKE3.
5. Pin Edge's build system to the new `STATOR_FFI_ABI_VERSION`
   (also emitted as a compile-time `static_assert` in
   `examples/mini_browser/` and in Edge's wrapper) so a stale
   header cannot link against a refreshed library.
6. Run Edge's snapshot-load CI step, which calls
   `validate.ps1 -Smoke` against the freshly built Edge binary —
   not just the Stator release library — to confirm the snapshot
   loads under the exact Edge link configuration.
7. Land the revendor as a single commit whose message records the
   `release_tag`, `commit_id`, `STATOR_FFI_ABI_VERSION`, and
   `edge_release_hash` of the new bundle, so bisection can map an
   Edge regression to an exact Stator artefact without consulting
   external systems.

A revendor MUST NOT cherry-pick a subset of bundles (e.g. update
only the Windows artefact); doing so would diverge `build_id` and
`edge_release_hash` across targets, breaking the loader's
fail-closed envelope on the unupdated platforms.

### 10.6 Cache invalidation and rollback

Edge persists loaded warm-context snapshots under a content-addressed
cache keyed by the bundle's `edge_release_hash`.  The following
events invalidate cache entries and force a re-fetch:

- A revendor that changes any of `crate_versions`, `ffi_abi_version`,
  `snapshot_format_version`, `bytecode_format_version`, `build_id`,
  `build_features_hash`, `jit_tiering_hash`, `cpu_features_hash`,
  `manifest_hash`, or `edge_release_hash`.  Because all of these
  participate in `header_digest`, *any* substantive change flips
  `edge_release_hash` and therefore the cache key.
- A bundle being marked `"revoked": true` in a follow-up patch
  release (see below).  Edge's update channel publishes the revoked
  hash list; on next launch the cache evicts matching entries
  before any load attempt.
- A loader rejection at runtime (any `SnapshotError::*` variant from
  §7).  The offending cache entry is quarantined and the loader
  falls back to cold start; the rejection diagnostic is reported
  through Edge telemetry bucketed by the `SnapshotError` variant
  name surfaced via the FFI status detail string.

Rollback procedure when a shipped bundle is found defective:

1. The Stator release team publishes a patch release `vX.Y.Z+1` that
   re-uses the prior known-good bundle contents but bumps
   `release_tag` and `released_at` and lists the defective bundle's
   `edge_release_hash` in a new `revoked_hashes[]` array of the
   release notes.  A separate `revocations.json` (signed with the
   same key set) is published alongside for tooling consumption.
2. Edge's update channel propagates `revocations.json`.  On next
   launch every Edge instance evicts cache entries matching the
   listed hashes and refuses to load any snapshot whose
   `edge_release_hash` appears in the revocation list — even if the
   bundle is still physically present in the vendored tree.
3. The Edge revendor for `vX.Y.Z+1` follows §10.5 verbatim.
4. The defective Stator release is left in place (not deleted) so
   bisection and post-mortem can still inspect it, but the
   prepublish gate marks it `revoked: true` so any accidental
   re-ingestion is rejected.

### 10.7 Prepublish gate

Before a Stator tag is allowed to publish artefacts the prepublish
gate (extending the workflow introduced in commit `aa8f1b2f`) MUST:

1. Build every supported `(target_triple, profile)` pair.
2. Regenerate `stator.h` and assert it matches the committed copy
   byte-for-byte.
3. Produce the bundle exactly as described in §10.1 from build
   outputs only — no manual file copies.
4. Run `validation/validate.sh -Smoke` (or `.ps1` on Windows) on
   every bundle in a freshly provisioned runner that matches the
   bundle's target.
5. Verify `STATOR_FFI_ABI_VERSION` was bumped iff the FFI surface or
   `STWC` header changed since the previous tag; refuse to publish
   on mismatch.
6. Cross-check every `snapshots[].header_digest` value in
   `manifest.json` against the actual footer digest of each `.stwc`.
7. Refuse to publish if any bundle's `cargo_profile` is `dev` or
   `debug`, if `revoked` is `true`, or if `edge_release_hash` is
   absent or all-zero.

The same gate runs in dry-run mode on every PR that touches the
snapshot subsystem, so regressions in artefact packaging are caught
before the tag is cut.

---

## 11. Out-of-scope follow-ups (tracked as separate todos)

This document is design only.  Concrete code work is split into
follow-up slices, recorded in the session todo store at the time
this doc landed.  Implementation must not begin until each slice is
scheduled.

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

When the implementation slices ship, a tagged release MUST include:

- `stator-snapshot-spec-v1.md` (frozen copy of this document, header
  bumped to *Status: Frozen*).
- The regenerated `stator.h` exposing the §8 FFI surface.
- A `snapshots/` directory in the release tarball containing a
  reference warm-context snapshot produced from the canonical
  bootstrap script, alongside its `BuildId`, `manifest_hash`, and
  BLAKE3 digest in a sidecar `manifest.json`.
- An updated `RELEASES.md` entry that records the bumped
  `STATOR_FFI_ABI_VERSION` and lists the new `SnapshotError` variants
  as a stability checkpoint.
- The Edge prepublish gate (see commit `aa8f1b2f`) extended to
  verify that the shipped snapshot loads cleanly into a freshly
  built engine with the matching manifest.

---

## 11. Out-of-scope follow-ups (tracked as separate todos)

This document is design only.  Concrete code work is split into
follow-up slices, recorded in the session todo store at the time
this doc landed.  Implementation must not begin until each slice is
scheduled.

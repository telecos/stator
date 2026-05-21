# Stator Code Cache Key Schema

This document defines the complete invalidation key schema Edge must use before accepting any Stator-produced cached code artifact. The key is intentionally larger than the current module cache metadata so stale bytecode, tiered native code, snapshot references, and vendored release artifacts fail closed instead of executing under incompatible engine, source, build, or platform conditions.

## Cache artifact types

Every artifact key begins with an `artifact_type` discriminator:

| Artifact type | Description | Restore behavior |
|---|---|---|
| `script-bytecode` | Classic-script parser and bytecode output, including line/column origin data and source map metadata. | May be restored into interpreter/baseline input only. |
| `module-bytecode` | Module parser metadata, dependency/export records, import attributes, and bytecode. | May be restored only when module graph identity and compile options match. |
| `baseline-code` | Baseline native code plus relocation/deopt/IC metadata for one function or script body. | May be restored only on exact platform, CPU, build, and tiering compatibility. |
| `jit-code` | Optimizing-tier native code (`maglev`, `turbofan`, or future tiers) plus guards, feedback dependencies, deopt metadata, and code-range layout constraints. | May be restored only when feedback schema, compiler flags, and CPU feature set match exactly. |
| `snapshot-reference` | A reference from code cache entries to a startup or context snapshot blob. The key stores the referenced snapshot identity, not a copy of the snapshot. | May be used only if the referenced snapshot blob is present and its own format/build key matches. |

## Proposed C ABI: classic-script bytecode cache

The existing C ABI exposes module cache entry points only:
`stator_module_create_code_cache` and `stator_module_compile_cached`. Classic
scripts currently compile with `stator_script_compile`, then callers can attach
origin metadata afterward with `stator_script_set_origin`; that is too late for
a bytecode cache key because the origin, source-map URL, and host options must
be validated before accepting cached bytecode. The script cache ABI should
therefore use a compile-options struct that is passed to both production and
restore paths.

### Types

```c
typedef enum StatorScriptCacheStatus {
  /* Operation succeeded: a blob was produced or bytecode was restored. */
  StatorScriptCacheStatusOk = 0,
  /*
   * Cache was well-formed enough to inspect but did not match the requested
   * source, origin, options, engine, or format identity. compile_cached falls
   * back to a normal source compile and returns that script handle.
   */
  StatorScriptCacheStatusRejected = 1,
  /*
   * A required pointer/length pair, options struct, output argument, or cache
   * envelope was invalid. compile_cached falls back to source compile when the
   * source/options inputs are still usable; otherwise it returns an errored
   * script handle.
   */
  StatorScriptCacheStatusInvalid = 2,
  /*
   * This engine build or option combination cannot serialize/restore script
   * bytecode. compile_cached falls back to source compile.
   */
  StatorScriptCacheStatusUnsupported = 3,
  /*
   * Producing bytecode, restoring bytecode, or compiling the fallback source
   * raised a parser/compiler exception. The returned script is an errored
   * StatorScript and no cache bytes are accepted.
   */
  StatorScriptCacheStatusException = 4,
} StatorScriptCacheStatus;

typedef enum StatorScriptCacheDiagnostic {
  StatorScriptCacheDiagnosticNone = 0,
  StatorScriptCacheDiagnosticSchemaVersion = 1,
  StatorScriptCacheDiagnosticArtifactType = 2,
  StatorScriptCacheDiagnosticEngineVersion = 3,
  StatorScriptCacheDiagnosticFormatVersion = 4,
  StatorScriptCacheDiagnosticSourceIdentity = 5,
  StatorScriptCacheDiagnosticOrigin = 6,
  StatorScriptCacheDiagnosticSourceMap = 7,
  StatorScriptCacheDiagnosticHostOptions = 8,
  StatorScriptCacheDiagnosticParserFlags = 9,
  StatorScriptCacheDiagnosticCompilerFlags = 10,
  StatorScriptCacheDiagnosticPlatform = 11,
  StatorScriptCacheDiagnosticBuildFeatures = 12,
  StatorScriptCacheDiagnosticSnapshot = 13,
  StatorScriptCacheDiagnosticCorruptPayload = 14,
  StatorScriptCacheDiagnosticUnsupportedFeature = 15,
  StatorScriptCacheDiagnosticCompileException = 16,
} StatorScriptCacheDiagnostic;

typedef struct StatorScriptCompileOptions {
  /* Stable script URL/resource name; copied by Stator during the call. */
  const char *resource_name;
  size_t resource_name_len;
  /* Serialized origin or opaque-origin token supplied by the embedder. */
  const char *source_origin;
  size_t source_origin_len;
  /* Optional //# sourceURL= override when it differs from resource_name. */
  const char *source_url;
  size_t source_url_len;
  int32_t line_offset;
  int32_t column_offset;
  const char *source_map_url;
  size_t source_map_url_len;
  /*
   * Embedder-defined compile-affecting options. Stator treats the byte sequence
   * as opaque key material and stores only its digest in diagnostics.
   */
  const uint8_t *host_defined_options;
  size_t host_defined_options_len;
  uint64_t parser_feature_bits;
  uint64_t bytecode_feature_bits;
  uint32_t strict_mode_policy;
} StatorScriptCompileOptions;

typedef struct StatorScriptCacheTelemetry {
  StatorScriptCacheStatus status;
  StatorScriptCacheDiagnostic diagnostic;
  uint32_t cache_schema_version;
  uint32_t script_cache_format_version;
  uint64_t source_length_bytes;
  uint64_t cache_length_bytes;
  uint64_t restored_bytecode_length_bytes;
  bool fallback_compile_attempted;
  bool fallback_compile_succeeded;
} StatorScriptCacheTelemetry;
```

Telemetry fields are low-cardinality by design. They must not contain full URLs,
source text, raw hashes, source-map URLs, or host-defined option bytes. If Edge
needs URL/source attribution, it should join these counters to a privacy-reviewed
browser-side cache key bucket rather than logging Stator inputs directly.

### Entry points

```c
StatorString *stator_script_create_code_cache(
    const StatorScript *script,
    const char *source,
    size_t source_len,
    const StatorScriptCompileOptions *options,
    StatorScriptCacheStatus *out_status,
    StatorScriptCacheTelemetry *out_telemetry);

StatorScript *stator_script_compile_cached(
    StatorContext *ctx,
    const char *source,
    size_t source_len,
    const char *cache_data,
    size_t cache_len,
    const StatorScriptCompileOptions *options,
    StatorScriptCacheStatus *out_status,
    StatorScriptCacheTelemetry *out_telemetry);
```

`stator_script_create_code_cache` accepts only a successfully compiled
`StatorScript` plus the exact source and compile options used to create it. It
returns an engine-owned `StatorString` containing a versioned `script-bytecode`
blob when `out_status` is `StatorScriptCacheStatusOk`; callers read bytes with
`stator_string_data` / `stator_string_len`, copy them into their persistent
store if desired, and release the handle with `stator_string_free`. Null script
handles, errored scripts, invalid source/options, or unsupported bytecode shapes
return null and set `out_status` to `Invalid`, `Exception`, or `Unsupported`.

`stator_script_compile_cached` validates the blob magic, schema, script cache
format, engine/ABI identity, source hash/length/encoding, origin fields,
source-map URL, host-defined options, parser flags, bytecode flags, platform,
build features, and snapshot references before any bytecode is restored. On
`Ok`, the returned `StatorScript` owns restored bytecode and has the supplied
origin/options installed before callers can execute it. On `Rejected`,
`Invalid`, or `Unsupported`, the API falls back to compiling `source` with the
same options so embedders can use a single call on cache miss; telemetry records
whether fallback was attempted and succeeded. On `Exception`, the returned
script is an errored `StatorScript` whose existing error accessors expose the
parser/compiler failure.

All pointer/length inputs are borrowed for the duration of the call and copied
or hashed before return. `out_status` and `out_telemetry` are optional; when
present each must be valid for one write. Cache bytes are untrusted input and
must fail closed: accepted blobs may skip parsing and bytecode generation, but
must never weaken source, origin, option, or version validation.

### Parity and differences versus module cache

Script and module caches share the same key schema, manifest invalidation rules,
cache blob ownership model, privacy constraints, coarse status contract, and
diagnostic vocabulary. Both production APIs return `StatorString` blobs owned by
Stator, both restore APIs validate source/options before use, and both must
reject corrupt payloads without executing cached code.

Script cache behavior intentionally differs where classic scripts differ from
modules:

- `artifact_type` is `script-bytecode`, `artifact_scope` is `classic-script`,
  and `parse_goal` is Script; there is no module request graph, import/export
  shape, import attributes hash, top-level-await state, or `import.meta.url`.
- Script origin/options are inputs to cache production and restore. The current
  post-compile `stator_script_set_origin` remains useful for non-cached scripts
  but is not sufficient for cacheable scripts because origin affects the key.
- `stator_script_compile_cached` falls back to normal source compilation on
  rejected, invalid, or unsupported cache input. Module cache restore currently
  returns an errored module on rejection because module graph/linking state has
  stricter fail-closed behavior.
- Classic-script blobs restore interpreter bytecode only. Native baseline/JIT
  artifacts, if added later, must use separate `baseline-code` or `jit-code`
  artifact types and stricter platform/sandbox validation.

## Canonical serialization

Keys are serialized as deterministic UTF-8 records before hashing or embedding in cache payloads. Producers and consumers must use the same order below.

1. Encode the ASCII magic string `stator-code-cache-key-v1`.
2. Append sections in the order listed in [Required key fields](#required-key-fields).
3. Within each section, append fields in the table order shown below.
4. Encode each field as `field_name`, a single NUL byte, a type tag, an unsigned little-endian length, the canonical value bytes, and a trailing NUL byte.
5. Encode absent optional strings as a present `null` value, not an empty string.
6. Normalize URLs with Edge's canonical URL serializer before hashing; do not lower-case or otherwise transform opaque origins, credentials mode names, import attribute values, or source map URLs after serialization.
7. Sort unordered maps lexicographically by UTF-8 key bytes. Duplicate keys are invalid and must reject the artifact.
8. Hash the serialized record with the cache manifest's configured key hash algorithm. The first supported algorithm is `sha256`; non-cryptographic checksums may be stored inside payloads for corruption detection but are not valid key hashes.

The same canonical record must also be stored, or be reproducible from stored metadata, so rejection diagnostics can identify the first mismatched field.

## Required key fields

### 1. Artifact identity

| Field | Type | Notes |
|---|---|---|
| `artifact_type` | enum | One of the artifact types above. |
| `artifact_scope` | enum | `classic-script`, `module`, `function`, `eval`, `wasm-module`, or `snapshot`. |
| `artifact_subtype` | string/null | Tier name (`baseline`, `maglev`, `turbofan`) or module type (`javascript`, `json`, `css`, `wasm`). |
| `cache_producer` | string | `stator_jse`, `stator_jse_ffi`, or Edge build component that emitted the artifact. |
| `cache_schema_version` | u32 | Version of this key schema. Start at `1`; bump on any incompatible key layout change. |

### 2. Version and format identity

| Field | Type | Notes |
|---|---|---|
| `stator_jse_crate_version` | semver | Engine internals version. |
| `stator_jse_ffi_crate_version` | semver | FFI crate version used by the embedder. |
| `stator_ffi_abi_version` | u32 | Exact `STATOR_FFI_ABI_VERSION` packed value. |
| `bytecode_format_version` | u32 | Interpreter bytecode format and operand encoding version. |
| `module_cache_format_version` | u32/null | Existing module code-cache payload format, when applicable. |
| `script_cache_format_version` | u32/null | Classic-script payload format, when applicable. |
| `baseline_code_format_version` | u32/null | Baseline native-code serialization/relocation format. |
| `jit_code_format_version` | u32/null | Optimizing-tier native-code serialization/relocation format. |
| `snapshot_format_version` | u32/null | Startup/context snapshot blob format. |
| `parser_ast_format_version` | u32 | Parser metadata and AST lowering version. |
| `compiler_ir_format_version` | u32 | Baseline/Maglev/Turbofan IR and feedback schema version. |
| `c_header_generation_id` | string/null | Edge-vendored generated `stator.h` identifier or digest, if available. |

### 3. Source identity and origin

| Field | Type | Notes |
|---|---|---|
| `source_hash_algorithm` | enum | `sha256` for Edge-managed keys; legacy FNV checksums are payload integrity only. |
| `source_hash` | bytes | Hash of the exact UTF-8/UTF-16 source bytes supplied to Stator after network decoding but before parser normalization. |
| `source_length_bytes` | u64 | Exact byte length of source input. |
| `source_encoding` | enum | `utf8`, `utf16le`, `utf16be`, or `latin1`. |
| `resource_url` | string/null | Canonical script/module URL. |
| `source_url` | string/null | `//# sourceURL=` override, if distinct from `resource_url`. |
| `source_origin` | string/null | Serialized origin or opaque-origin token supplied by Edge. |
| `base_url` | string/null | Base URL used for relative module specifier resolution. |
| `referrer_url` | string/null | Referrer used by fetch/compile policy. |
| `integrity_metadata` | string/null | Subresource Integrity metadata string. |
| `credentials_mode` | enum/null | `omit`, `same-origin`, or `include`. |
| `referrer_policy` | enum/null | Edge/Blink referrer policy used at fetch time. |
| `line_offset` | u32 | Initial line offset for diagnostics and source positions. |
| `column_offset` | u32 | Initial column offset. |
| `source_map_url` | string/null | `//# sourceMappingURL=` or embedder-provided source map URL. |
| `host_defined_options_hash` | bytes/null | Hash of host-defined compile options that can affect semantics. |

### 4. Module and import metadata

| Field | Type | Notes |
|---|---|---|
| `module_type` | enum/null | `javascript`, `json`, `css`, `wasm`, or future module type. |
| `module_request_count` | u32/null | Number of static module requests encoded in the artifact. |
| `module_requests_hash` | bytes/null | Hash of canonical specifier, assertion, and attribute records. |
| `import_attributes_hash` | bytes/null | Hash of top-level import attributes supplied by Edge. |
| `import_map_epoch` | string/null | Edge-provided import map/version token if resolution can vary. |
| `resolution_base_url` | string/null | Base URL used by the resolver after import map processing. |

### 5. Parser and compiler flags

| Field | Type | Notes |
|---|---|---|
| `strict_mode_policy` | enum | `source`, `force-strict`, or `force-sloppy`. |
| `script_kind` | enum | `classic`, `module`, `worker`, `worklet`, `extension`, or `internal`. |
| `language_mode` | enum | Parser language mode after embedder policy. |
| `parse_goal` | enum | Script, module, JSON module, CSS module, or Wasm module. |
| `enable_top_level_await` | bool | Required for module bytecode compatibility. |
| `enable_import_meta` | bool | Required for module bytecode compatibility. |
| `parser_feature_bits` | u64 | Bitset for parser gates and experimental syntax. |
| `bytecode_feature_bits` | u64 | Bitset for bytecode lowering choices. |
| `compiler_feature_bits` | u64 | Bitset for optimization passes and lowering choices. |
| `jit_enabled` | bool | Whether native tiers are allowed. |
| `tiering_mode` | enum | `interpreter-only`, `baseline`, `maglev`, `turbofan`, or `adaptive`. |
| `optimization_level` | enum/u32 | Build/runtime optimization policy. |
| `debug_instrumentation` | bool | Debug hooks, coverage, breakpoints, or inspector instrumentation enabled. |
| `profiling_instrumentation` | bool | CPU/heap profiling instrumentation enabled. |
| `sandbox_mode` | enum | Edge sandbox/JIT-write-protection mode affecting code generation. |

### 6. Platform and build identity

| Field | Type | Notes |
|---|---|---|
| `target_arch` | string | Rust/LLVM target architecture, for example `x86_64` or `aarch64`. |
| `target_os` | string | Rust target OS, for example `windows`. |
| `target_env` | string/null | MSVC/GNU/other ABI environment. |
| `target_pointer_width` | u32 | Usually `64` for Edge targets. |
| `endianness` | enum | `little` or `big`. |
| `cpu_vendor` | string/null | Vendor used for code generation decisions. |
| `cpu_family_model_stepping` | string/null | Stable CPU identity when native code depends on model-specific features. |
| `cpu_feature_set` | sorted string list | Exact enabled CPU feature list, for example `sse4.2`, `avx2`, `bmi2`. |
| `rustc_version` | string | Compiler version used to build Stator. |
| `llvm_version` | string/null | LLVM backend version if exposed by the build. |
| `cargo_profile` | enum | `debug`, `release`, `release-lto`, or Edge-defined profile. |
| `build_feature_set` | sorted string list | Cargo features, `cfg` flags, allocator mode, GC mode, Wasm backend, and inspector feature gates. |
| `link_time_optimization` | bool | Whether LTO changed code layout/inlining decisions. |
| `panic_strategy` | enum | `unwind` or `abort`. |
| `edge_channel` | enum/null | `canary`, `dev`, `beta`, `stable`, or internal release channel. |
| `edge_build_id` | string/null | Edge build/version used to produce or vendor the artifact. |

### 7. Snapshot references

| Field | Type | Notes |
|---|---|---|
| `snapshot_digest` | bytes/null | Digest of the referenced snapshot blob. |
| `snapshot_build_id` | string/null | Snapshot producer build identifier. |
| `snapshot_feature_set` | sorted string list/null | Snapshot build/runtime features. |
| `snapshot_context_kind` | enum/null | `startup`, `main-world`, `isolated-world`, `worker`, or `worklet`. |

## Rejection diagnostics and telemetry codes

Restore APIs must return a coarse status plus a structured code. Telemetry must not log full URLs, source text, hashes that can identify private content, or import attribute values unless Edge privacy review explicitly allows it.

| Code | Meaning |
|---|---|
| `accepted` | Artifact restored and executed under a matching key. |
| `miss_not_found` | No artifact was available for the requested key. |
| `rejected_schema_version` | Key schema version unsupported. |
| `rejected_artifact_type` | Artifact type/subtype not valid for this consumer. |
| `rejected_engine_version` | Engine crate, FFI crate, or ABI version mismatch. |
| `rejected_format_version` | Bytecode, native-code, module-cache, script-cache, or snapshot format mismatch. |
| `rejected_source_identity` | Source hash, length, encoding, URL, origin, or source map metadata mismatch. |
| `rejected_embedder_policy` | Credentials, referrer policy, integrity metadata, sandbox, or host-defined options mismatch. |
| `rejected_parser_flags` | Parser feature, parse goal, language mode, or compile option mismatch. |
| `rejected_compiler_flags` | Compiler feature, optimization, instrumentation, or tiering mode mismatch. |
| `rejected_platform` | OS, architecture, pointer width, endianness, CPU feature set, or target ABI mismatch. |
| `rejected_build_features` | Cargo/build features, profile, LTO, allocator, GC, Wasm backend, or panic strategy mismatch. |
| `rejected_snapshot` | Referenced snapshot missing or incompatible. |
| `rejected_release_artifact` | Vendored manifest, signing, channel, or Edge build identity mismatch. |
| `corrupt_payload` | Payload checksum, structural decode, relocation, or metadata validation failed. |
| `unsupported_native_code` | Native-code artifact was valid but cannot be mapped safely on this platform or sandbox. |

Counters should aggregate by artifact type, subtype/tier, accepted/miss/rejected status, and rejection code. High-cardinality fields must be bucketed or omitted.

## Edge release and vendoring expectations

Each Edge vendored Stator drop must ship a manifest next to `stator.h` and the Stator libraries. The manifest must include:

- Stator crate versions and exact `STATOR_FFI_ABI_VERSION`.
- Generated header digest and generation command.
- Supported cache schema, bytecode, module-cache, script-cache, native-code, and snapshot format versions.
- Supported target triples, CPU feature policies, build features, sandbox modes, and tiering modes.
- Edge channel/build compatibility window and explicit cache eviction policy for roll-forward and rollback.
- Optional prebuilt snapshot/code-cache artifact names, sizes, digests, and signatures.
- Privacy-approved telemetry code list and field allowlist.

Edge must clear or partition persisted cache storage when the manifest changes in any key field. Release automation should reject vendoring if the manifest and generated header disagree on ABI or cache format versions.

## Test plan

1. Unit-test canonical serialization with golden key records for each artifact type and prove field ordering is stable.
2. Unit-test every required field by mutating one field at a time and asserting the expected rejection diagnostic code.
3. Round-trip script and module bytecode caches across matching inputs; reject on source hash, length, URL/origin, source map URL, import attributes, parser flags, and format version changes.
4. Round-trip baseline/JIT artifacts only on matching target triples, CPU feature sets, compiler flags, sandbox modes, and tiering modes; reject downgraded CPU or changed optimization settings.
5. Validate snapshot references by accepting only present matching snapshot digests and rejecting missing, stale, or wrong-context snapshots.
6. Exercise corruption paths with truncated payloads, bad checksums, duplicate map keys, invalid enum values, and unsupported future schema versions.
7. Add Edge integration tests that simulate browser cache persistence across Stator revendor, Edge channel upgrade, rollback, CPU-feature change, and sandbox policy change.
8. Verify telemetry emits only approved low-cardinality rejection codes and no raw source text, full URLs, or private import attribute values.

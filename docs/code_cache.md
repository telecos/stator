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
  /* Canonical script URL/resource name; copied by Stator during the call. */
  const char *resource_url;
  size_t resource_url_len;
  /* Serialized origin or opaque-origin token supplied by the embedder. */
  const char *source_origin;
  size_t source_origin_len;
  /* Optional //# sourceURL= override when it differs from resource_url. */
  const char *source_url;
  size_t source_url_len;
  const char *base_url;
  size_t base_url_len;
  const char *referrer_url;
  size_t referrer_url_len;
  const char *integrity_metadata;
  size_t integrity_metadata_len;
  uint32_t credentials_mode;
  uint32_t referrer_policy;
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
  const uint8_t *compile_options;
  size_t compile_options_len;
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

Canonical values are encoded as follows:

- Strings are UTF-8 byte strings after Edge canonicalization. URL fields use the
  browser's canonical URL serializer, including punycode, percent-encoding, and
  opaque-origin serialization. `source_url` and `source_map_url` comments are
  trimmed exactly as the parser recognizes the directive, then serialized as
  strings without additional path resolution unless Edge has already resolved
  them.
- Enums are lowercase ASCII tokens. Booleans are one byte (`0` or `1`). Integers
  are unsigned little-endian except `line_offset` and `column_offset`, which are
  signed little-endian because embedders may represent wrapper offsets relative
  to generated source.
- Byte fields store the digest bytes, not hex text. Raw source, raw URLs, raw
  integrity metadata, import-attribute values, and host compile option bytes must
  not appear in telemetry.
- Sorted string lists are length-prefixed item sequences sorted by UTF-8 bytes.
  Maps are length-prefixed key/value item sequences sorted by key bytes; duplicate
  keys, invalid UTF-8, invalid enum tokens, or unknown non-nullable values reject
  the artifact as `corrupt_payload`.

Optional fields are never omitted from the canonical record. An absent pointer,
missing browser concept, or intentionally inapplicable field is encoded as
`null`. An empty string, empty list, empty map, or zero-length digest is distinct
from `null` and must match exactly. Producers must reject inputs that cannot be
canonicalized instead of silently omitting them. Consumers that read an older
payload missing a field required by the active schema must report the field's
specific `rejected_*` code when the payload is otherwise well-formed, or
`corrupt_payload` when the record cannot be decoded.

Script-bytecode and module-bytecode caches use the same source-metadata contract:
the key must include source origin, line/column offsets, resource URL, base URL,
referrer URL, integrity metadata, credentials mode, referrer policy, import
attributes/policy, source map URL, sourceURL comment, and all parser/compiler
compile options that can affect observable behavior or diagnostics. Classic
scripts encode module-only fields as `null`; modules encode script-only wrapper
fields as `null` only when the host genuinely has no value.

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
| `resource_url` | string/null | Canonical script/module URL used as the resource name. Required for network-backed script and module caches; `null` only for anonymous/eval/internal sources. |
| `source_url` | string/null | `//# sourceURL=` comment after parser directive handling. Distinct from `resource_url`; encode `null` when no directive is present. |
| `source_origin` | string/null | Serialized origin or opaque-origin token supplied by Edge. Must preserve opaque-origin identity and must not be derived from `resource_url` by Stator. |
| `base_url` | string/null | Base URL used for relative script metadata or module specifier resolution before import-map processing. |
| `referrer_url` | string/null | Referrer used by fetch/compile policy after browser policy trimming. |
| `integrity_metadata` | string/null | Canonical Subresource Integrity metadata string after browser parsing, or `null` when no integrity constraint applied. |
| `credentials_mode` | enum/null | `omit`, `same-origin`, or `include`; `null` only when no fetch credential policy exists for the artifact type. |
| `referrer_policy` | enum/null | Edge/Blink referrer policy used at fetch time. |
| `line_offset` | i32 | Initial line offset for diagnostics, stack traces, source positions, and source map lookup. |
| `column_offset` | i32 | Initial column offset. |
| `source_map_url` | string/null | `//# sourceMappingURL=` comment or embedder-provided source map URL after directive parsing. Must be keyed even when source text and bytecode are unchanged because diagnostics/devtools behavior changes. |
| `host_defined_options_hash` | bytes/null | Hash of opaque host-defined compile options that can affect semantics, diagnostics, instrumentation, or embedder policy. |
| `compile_options_hash` | bytes/null | Hash of structured Stator/Edge compile options not represented elsewhere, including parser goal overrides, code-generation mode, inspector/coverage/debug settings, source wrapping, and future flags. |

### 4. Module and import metadata

| Field | Type | Notes |
|---|---|---|
| `module_type` | enum/null | `javascript`, `json`, `css`, `wasm`, or future module type. |
| `module_request_count` | u32/null | Number of static module requests encoded in the artifact. |
| `module_requests_hash` | bytes/null | Hash of canonical specifier, assertion, and attribute records. |
| `import_attributes_hash` | bytes/null | Hash of top-level import attributes supplied by Edge. |
| `import_policy_hash` | bytes/null | Hash of import policy inputs that can change resolution or validation, including assertion policy, import-attributes feature mode, import maps, CSP/module policy tokens, and host allow/deny decisions. |
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

When a cache payload is decodable but cannot be accepted, restore must compare
the canonical record in section order and emit the telemetry code for the first
mismatched field. Corruption, duplicate fields, duplicate map keys, invalid enum
tokens, bad lengths, or non-canonical encodings use `corrupt_payload` instead of
a mismatch code.

| Field | Telemetry code |
|---|---|
| `artifact_type` | `rejected_artifact_type` |
| `artifact_scope` | `rejected_artifact_type` |
| `artifact_subtype` | `rejected_artifact_type` |
| `cache_producer` | `rejected_release_artifact` |
| `cache_schema_version` | `rejected_schema_version` |
| `stator_jse_crate_version` | `rejected_engine_version` |
| `stator_jse_ffi_crate_version` | `rejected_engine_version` |
| `stator_ffi_abi_version` | `rejected_engine_version` |
| `bytecode_format_version` | `rejected_format_version` |
| `module_cache_format_version` | `rejected_format_version` |
| `script_cache_format_version` | `rejected_format_version` |
| `baseline_code_format_version` | `rejected_format_version` |
| `jit_code_format_version` | `rejected_format_version` |
| `snapshot_format_version` | `rejected_format_version` |
| `parser_ast_format_version` | `rejected_format_version` |
| `compiler_ir_format_version` | `rejected_format_version` |
| `c_header_generation_id` | `rejected_release_artifact` |
| `source_hash_algorithm` | `rejected_source_identity` |
| `source_hash` | `rejected_source_identity` |
| `source_length_bytes` | `rejected_source_identity` |
| `source_encoding` | `rejected_source_identity` |
| `resource_url` | `rejected_source_identity` |
| `source_url` | `rejected_source_identity` |
| `source_origin` | `rejected_source_identity` |
| `base_url` | `rejected_source_identity` |
| `referrer_url` | `rejected_embedder_policy` |
| `integrity_metadata` | `rejected_embedder_policy` |
| `credentials_mode` | `rejected_embedder_policy` |
| `referrer_policy` | `rejected_embedder_policy` |
| `line_offset` | `rejected_source_identity` |
| `column_offset` | `rejected_source_identity` |
| `source_map_url` | `rejected_source_identity` |
| `host_defined_options_hash` | `rejected_embedder_policy` |
| `compile_options_hash` | `rejected_compiler_flags` |
| `module_type` | `rejected_parser_flags` |
| `module_request_count` | `rejected_source_identity` |
| `module_requests_hash` | `rejected_source_identity` |
| `import_attributes_hash` | `rejected_embedder_policy` |
| `import_policy_hash` | `rejected_embedder_policy` |
| `import_map_epoch` | `rejected_embedder_policy` |
| `resolution_base_url` | `rejected_source_identity` |
| `strict_mode_policy` | `rejected_parser_flags` |
| `script_kind` | `rejected_parser_flags` |
| `language_mode` | `rejected_parser_flags` |
| `parse_goal` | `rejected_parser_flags` |
| `enable_top_level_await` | `rejected_parser_flags` |
| `enable_import_meta` | `rejected_parser_flags` |
| `parser_feature_bits` | `rejected_parser_flags` |
| `bytecode_feature_bits` | `rejected_compiler_flags` |
| `compiler_feature_bits` | `rejected_compiler_flags` |
| `jit_enabled` | `rejected_compiler_flags` |
| `tiering_mode` | `rejected_compiler_flags` |
| `optimization_level` | `rejected_compiler_flags` |
| `debug_instrumentation` | `rejected_compiler_flags` |
| `profiling_instrumentation` | `rejected_compiler_flags` |
| `sandbox_mode` | `rejected_embedder_policy` |
| `target_arch` | `rejected_platform` |
| `target_os` | `rejected_platform` |
| `target_env` | `rejected_platform` |
| `target_pointer_width` | `rejected_platform` |
| `endianness` | `rejected_platform` |
| `cpu_vendor` | `rejected_platform` |
| `cpu_family_model_stepping` | `rejected_platform` |
| `cpu_feature_set` | `rejected_platform` |
| `rustc_version` | `rejected_build_features` |
| `llvm_version` | `rejected_build_features` |
| `cargo_profile` | `rejected_build_features` |
| `build_feature_set` | `rejected_build_features` |
| `link_time_optimization` | `rejected_build_features` |
| `panic_strategy` | `rejected_build_features` |
| `edge_channel` | `rejected_release_artifact` |
| `edge_build_id` | `rejected_release_artifact` |
| `snapshot_digest` | `rejected_snapshot` |
| `snapshot_build_id` | `rejected_snapshot` |
| `snapshot_feature_set` | `rejected_snapshot` |
| `snapshot_context_kind` | `rejected_snapshot` |

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

## Release manifest validation contract

Release automation MUST validate every Edge code-cache release manifest with a
schema gate that fails closed on any missing or mismatched field. The
authoritative validator and its mutation test matrix live in
`crates/stator_ffi/tests/release_manifest.rs` and run as part of
`cargo test --workspace`. The schema is:

- `schema_id` (string): must equal `stator-edge-code-cache-release-manifest`.
- `schema_version` (uint): must equal `1`. Older or newer schema versions
  fail closed; bump the constant and add a migration when the schema evolves.
- `generated_at_utc` (string): non-empty ISO-8601 UTC timestamp.
- `stator.commit` (string): non-empty Stator source commit hash.
- `stator.crates[]`: must contain at least entries for `stator_jse`,
  `stator_jse_ffi`, and `st8`, each with a non-empty `version` string.
- `ffi_abi.{major,minor,patch,packed}` (uint): present, non-negative, and
  internally consistent (`packed == (major<<16) | (minor<<8) | patch`).
- `ffi_abi.c_header_generation_id` (string): non-empty digest or build id
  identifying the vendored `stator.h`.
- `cache_formats.{cache_schema_version, script_cache_format_version,
  module_cache_format_version, bytecode_format_version,
  baseline_code_format_version, jit_code_format_version,
  snapshot_format_version}` (uint): all required.
- `key_schema.key_schema_version` (uint) and
  `key_schema.canonical_key_hash_algorithm` (enum: `sha256`).
- `artifacts[]`: at least one entry. Each entry requires `artifact_type`
  (one of `script-bytecode`, `module-bytecode`, `baseline-code`,
  `jit-code`, `snapshot-reference`), `target_triple`, `target_os`,
  `target_arch`, `cargo_profile`, `size_bytes`, `digest_algorithm`
  (`sha256`/`sha384`/`sha512`), `digest_hex` (even-length lower-case hex),
  and a `signature` object with non-empty `algorithm` and lower-case hex
  `value_hex`.
- `telemetry.diagnostic_codes[]`: must include every code listed in
  "Cache restore telemetry" above so Edge privacy review and counter
  aggregation stay in sync with the engine's emitted codes.
- `telemetry.field_allowlist[]`: required array describing the privacy-
  approved low-cardinality fields permitted in telemetry payloads.

The validator collects every error rather than failing on the first
mismatch so release automation can surface a single, actionable report.
Any change to the manifest schema MUST be accompanied by a corresponding
change to the constants and mutation tests in
`crates/stator_ffi/tests/release_manifest.rs`.

## Test plan

1. Unit-test canonical serialization with golden key records for each artifact type and prove field ordering is stable.
2. Unit-test every required field by mutating one field at a time and asserting the expected rejection diagnostic code.
3. Round-trip script and module bytecode caches across matching inputs; reject on source hash, length, URL/origin, source map URL, import attributes, parser flags, and format version changes.
4. Round-trip baseline/JIT artifacts only on matching target triples, CPU feature sets, compiler flags, sandbox modes, and tiering modes; reject downgraded CPU or changed optimization settings.
5. Validate snapshot references by accepting only present matching snapshot digests and rejecting missing, stale, or wrong-context snapshots.
6. Exercise corruption paths with truncated payloads, bad checksums, duplicate map keys, invalid enum values, and unsupported future schema versions.
7. Add Edge integration tests that simulate browser cache persistence across Stator revendor, Edge channel upgrade, rollback, CPU-feature change, and sandbox policy change.
8. Verify telemetry emits only approved low-cardinality rejection codes and no raw source text, full URLs, or private import attribute values.

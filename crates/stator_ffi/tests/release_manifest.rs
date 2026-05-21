//! Validation for the Edge code-cache release/vendoring manifest.
//!
//! `docs/code_cache.md` ("Edge release and vendoring expectations" and
//! "Release manifest validation contract") specifies that every vendored
//! Stator drop ships a manifest next to `stator.h` describing the engine
//! identity, ABI version, cache/format versions, target/profile, key
//! schema, signing/digest fields, and the approved telemetry mapping.
//! Edge must fail closed if any required field is missing, malformed, or
//! mismatched.
//!
//! This file codifies the manifest schema as a pure validator and
//! exercises every required field with a one-field-at-a-time mutation
//! matrix, mirroring the unit-test guidance in the code-cache test plan.
//! The validator is intentionally self-contained: it has no `unsafe`,
//! takes no I/O, and depends only on `serde_json`, so it can run as part
//! of the existing `cargo test --workspace` release-train gate.

use serde_json::{Map, Value, json};

/// Canonical manifest schema identifier. Every release manifest must
/// embed this string verbatim under the top-level `schema_id` field.
const SCHEMA_ID: &str = "stator-edge-code-cache-release-manifest";

/// Currently supported manifest schema version. The validator only
/// accepts manifests whose `schema_version` exactly matches; older or
/// newer schema versions must fail closed at the vendoring boundary.
const SCHEMA_VERSION: u64 = 1;

/// Crate names that must appear in `stator.crates`. The order does not
/// matter; the validator enforces presence and well-formed non-empty
/// version strings only.
const REQUIRED_CRATES: &[&str] = &["stator_jse", "stator_jse_ffi", "st8"];

/// Cache/format version fields that must appear under `cache_formats`.
/// Each field must be a non-negative integer; absence or non-integer
/// values must fail closed because they are part of the key schema.
const REQUIRED_CACHE_FORMAT_FIELDS: &[&str] = &[
    "cache_schema_version",
    "script_cache_format_version",
    "module_cache_format_version",
    "bytecode_format_version",
    "baseline_code_format_version",
    "jit_code_format_version",
    "snapshot_format_version",
];

/// Artifact-type discriminator values defined by `docs/code_cache.md`
/// ("Cache artifact types"). Manifests must only describe artifacts
/// whose type matches one of these tokens.
const VALID_ARTIFACT_TYPES: &[&str] = &[
    "script-bytecode",
    "module-bytecode",
    "baseline-code",
    "jit-code",
    "snapshot-reference",
];

/// Telemetry diagnostic codes that the manifest must enumerate so that
/// downstream consumers can verify every code path documented in
/// `docs/code_cache.md` ("Cache restore telemetry") is mapped. Missing
/// codes must fail closed because Edge requires the full set for
/// privacy review and aggregation.
const REQUIRED_DIAGNOSTIC_CODES: &[&str] = &[
    "accepted",
    "miss_not_found",
    "rejected_schema_version",
    "rejected_artifact_type",
    "rejected_engine_version",
    "rejected_format_version",
    "rejected_source_identity",
    "rejected_embedder_policy",
    "rejected_parser_flags",
    "rejected_compiler_flags",
    "rejected_platform",
    "rejected_build_features",
    "rejected_snapshot",
    "rejected_release_artifact",
    "corrupt_payload",
    "unsupported_native_code",
];

/// Canonical mapping from every key-schema field documented in
/// `docs/code_cache.md` ("Rejection diagnostics and telemetry codes") to
/// the telemetry diagnostic code that restore APIs must emit when that
/// field is the first mismatch. Edge release manifests and the runtime
/// cache restore paths must agree on this exact table; corruption,
/// duplicate fields, duplicate map keys, invalid enum tokens, bad
/// lengths, or non-canonical encodings must use `corrupt_payload`
/// instead of any of the codes below.
///
/// This list is the authoritative compile-time contract: any change
/// must be made here and in `docs/code_cache.md` together, and the
/// `test_field_diagnostic_map_matches_docs` test enforces parity.
const FIELD_TO_DIAGNOSTIC_CODE: &[(&str, &str)] = &[
    ("artifact_type", "rejected_artifact_type"),
    ("artifact_scope", "rejected_artifact_type"),
    ("artifact_subtype", "rejected_artifact_type"),
    ("cache_producer", "rejected_release_artifact"),
    ("cache_schema_version", "rejected_schema_version"),
    ("stator_jse_crate_version", "rejected_engine_version"),
    ("stator_jse_ffi_crate_version", "rejected_engine_version"),
    ("stator_ffi_abi_version", "rejected_engine_version"),
    ("bytecode_format_version", "rejected_format_version"),
    ("module_cache_format_version", "rejected_format_version"),
    ("script_cache_format_version", "rejected_format_version"),
    ("baseline_code_format_version", "rejected_format_version"),
    ("jit_code_format_version", "rejected_format_version"),
    ("snapshot_format_version", "rejected_format_version"),
    ("parser_ast_format_version", "rejected_format_version"),
    ("compiler_ir_format_version", "rejected_format_version"),
    ("c_header_generation_id", "rejected_release_artifact"),
    ("source_hash_algorithm", "rejected_source_identity"),
    ("source_hash", "rejected_source_identity"),
    ("source_length_bytes", "rejected_source_identity"),
    ("source_encoding", "rejected_source_identity"),
    ("resource_url", "rejected_source_identity"),
    ("source_url", "rejected_source_identity"),
    ("source_origin", "rejected_source_identity"),
    ("base_url", "rejected_source_identity"),
    ("referrer_url", "rejected_embedder_policy"),
    ("integrity_metadata", "rejected_embedder_policy"),
    ("credentials_mode", "rejected_embedder_policy"),
    ("referrer_policy", "rejected_embedder_policy"),
    ("line_offset", "rejected_source_identity"),
    ("column_offset", "rejected_source_identity"),
    ("source_map_url", "rejected_source_identity"),
    ("host_defined_options_hash", "rejected_embedder_policy"),
    ("compile_options_hash", "rejected_compiler_flags"),
    ("module_type", "rejected_parser_flags"),
    ("module_request_count", "rejected_source_identity"),
    ("module_requests_hash", "rejected_source_identity"),
    ("import_attributes_hash", "rejected_embedder_policy"),
    ("import_policy_hash", "rejected_embedder_policy"),
    ("import_map_epoch", "rejected_embedder_policy"),
    ("resolution_base_url", "rejected_source_identity"),
    ("strict_mode_policy", "rejected_parser_flags"),
    ("script_kind", "rejected_parser_flags"),
    ("language_mode", "rejected_parser_flags"),
    ("parse_goal", "rejected_parser_flags"),
    ("enable_top_level_await", "rejected_parser_flags"),
    ("enable_import_meta", "rejected_parser_flags"),
    ("parser_feature_bits", "rejected_parser_flags"),
    ("bytecode_feature_bits", "rejected_compiler_flags"),
    ("compiler_feature_bits", "rejected_compiler_flags"),
    ("jit_enabled", "rejected_compiler_flags"),
    ("tiering_mode", "rejected_compiler_flags"),
    ("optimization_level", "rejected_compiler_flags"),
    ("debug_instrumentation", "rejected_compiler_flags"),
    ("profiling_instrumentation", "rejected_compiler_flags"),
    ("sandbox_mode", "rejected_embedder_policy"),
    ("target_arch", "rejected_platform"),
    ("target_os", "rejected_platform"),
    ("target_env", "rejected_platform"),
    ("target_pointer_width", "rejected_platform"),
    ("endianness", "rejected_platform"),
    ("cpu_vendor", "rejected_platform"),
    ("cpu_family_model_stepping", "rejected_platform"),
    ("cpu_feature_set", "rejected_platform"),
    ("rustc_version", "rejected_build_features"),
    ("llvm_version", "rejected_build_features"),
    ("cargo_profile", "rejected_build_features"),
    ("build_feature_set", "rejected_build_features"),
    ("link_time_optimization", "rejected_build_features"),
    ("panic_strategy", "rejected_build_features"),
    ("edge_channel", "rejected_release_artifact"),
    ("edge_build_id", "rejected_release_artifact"),
    ("manifest_id", "rejected_release_artifact"),
    ("min_compatible_manifest_id", "rejected_release_artifact"),
    ("revoked_manifest_ids", "rejected_release_artifact"),
    ("eviction_policy", "rejected_release_artifact"),
    ("edge_channel_window", "rejected_release_artifact"),
    ("previous_manifest_id", "rejected_release_artifact"),
    ("rollback_supported", "rejected_release_artifact"),
    ("artifact_path", "rejected_release_artifact"),
    ("root_relative_to_manifest", "rejected_release_artifact"),
    ("artifact_subdir", "rejected_release_artifact"),
    ("snapshot_digest", "rejected_snapshot"),
    ("snapshot_build_id", "rejected_snapshot"),
    ("snapshot_feature_set", "rejected_snapshot"),
    ("snapshot_context_kind", "rejected_snapshot"),
];

/// Hash algorithms accepted for canonical key hashing. Per
/// `docs/code_cache.md` Section "Canonical serialization", the first
/// supported algorithm is `sha256`; non-cryptographic checksums are not
/// valid key hashes and must be rejected here too.
const VALID_KEY_HASH_ALGORITHMS: &[&str] = &["sha256"];

/// Digest algorithms accepted for per-artifact integrity fields.
const VALID_DIGEST_ALGORITHMS: &[&str] = &["sha256", "sha384", "sha512"];

/// Cache eviction policies a manifest is allowed to declare. Anything
/// outside this set must fail closed because Edge would not know how to
/// honor a previously-cached partition under an unknown policy.
const VALID_EVICTION_POLICIES: &[&str] = &["clear-all", "partition-by-manifest-id"];

/// Validate a parsed manifest value and return every detected error.
/// The validator collects all errors rather than failing on the first
/// problem so release automation can show actionable diagnostics in one
/// shot. An empty error vector means the manifest is accepted.
fn validate(manifest: &Value) -> Vec<String> {
    let mut errs: Vec<String> = Vec::new();
    let root = match manifest.as_object() {
        Some(o) => o,
        None => {
            errs.push("manifest root must be a JSON object".to_string());
            return errs;
        }
    };

    require_string_eq(root, "schema_id", SCHEMA_ID, &mut errs);
    require_uint_eq(root, "schema_version", SCHEMA_VERSION, &mut errs);
    require_nonempty_string(root, "generated_at_utc", &mut errs);

    if let Some(stator) = require_object(root, "stator", &mut errs) {
        require_nonempty_string(stator, "commit", &mut errs);
        validate_crates(stator, &mut errs);
    }

    if let Some(abi) = require_object(root, "ffi_abi", &mut errs) {
        let major = require_uint(abi, "major", &mut errs);
        let minor = require_uint(abi, "minor", &mut errs);
        let patch = require_uint(abi, "patch", &mut errs);
        let packed = require_uint(abi, "packed", &mut errs);
        if let (Some(ma), Some(mi), Some(pa), Some(pk)) = (major, minor, patch, packed) {
            let expected = (ma << 16) | (mi << 8) | pa;
            if expected != pk {
                errs.push(format!(
                    "ffi_abi.packed ({pk}) does not match major/minor/patch ({expected})"
                ));
            }
        }
        require_nonempty_string(abi, "c_header_generation_id", &mut errs);
    }

    if let Some(formats) = require_object(root, "cache_formats", &mut errs) {
        for field in REQUIRED_CACHE_FORMAT_FIELDS {
            require_uint(formats, field, &mut errs);
        }
    }

    if let Some(key_schema) = require_object(root, "key_schema", &mut errs) {
        require_uint(key_schema, "key_schema_version", &mut errs);
        if let Some(algo) =
            require_nonempty_string(key_schema, "canonical_key_hash_algorithm", &mut errs)
            && !VALID_KEY_HASH_ALGORITHMS.contains(&algo.as_str())
        {
            errs.push(format!(
                "key_schema.canonical_key_hash_algorithm '{algo}' is not an accepted algorithm \
                 (expected one of {VALID_KEY_HASH_ALGORITHMS:?})"
            ));
        }
    }

    validate_artifacts(root, &mut errs);
    validate_telemetry(root, &mut errs);
    validate_packaging_metadata(root, &mut errs);

    errs
}

/// Validate the packaging/invalidation/rollback metadata block. These
/// fields are what lets Edge invalidate, partition, or revoke a
/// previously vendored drop without re-deriving identity from the
/// per-artifact records. Anything missing or malformed must fail closed
/// at the vendoring boundary, because the cost of acting on a stale
/// `manifest_id` or unknown eviction policy is loading bytecode under
/// the wrong engine identity.
fn validate_packaging_metadata(root: &Map<String, Value>, errs: &mut Vec<String>) {
    require_nonempty_string(root, "manifest_id", errs);

    if let Some(compat) = require_object(root, "compatibility", errs) {
        require_nonempty_string(compat, "min_compatible_manifest_id", errs);
        if let Some(policy) = require_nonempty_string(compat, "eviction_policy", errs)
            && !VALID_EVICTION_POLICIES.contains(&policy.as_str())
        {
            errs.push(format!(
                "compatibility.eviction_policy '{policy}' is not an accepted policy \
                 (expected one of {VALID_EVICTION_POLICIES:?})"
            ));
        }
        match compat.get("revoked_manifest_ids") {
            Some(Value::Array(items)) => {
                let mut seen: Vec<&str> = Vec::new();
                for (idx, item) in items.iter().enumerate() {
                    match item.as_str() {
                        Some(s) if !s.is_empty() => {
                            if seen.contains(&s) {
                                errs.push(format!(
                                    "compatibility.revoked_manifest_ids[{idx}] duplicates '{s}'"
                                ));
                            }
                            seen.push(s);
                        }
                        _ => errs.push(format!(
                            "compatibility.revoked_manifest_ids[{idx}] must be a non-empty string"
                        )),
                    }
                }
                if let Some(mid) = root.get("manifest_id").and_then(Value::as_str)
                    && seen.contains(&mid)
                {
                    errs.push(format!(
                        "compatibility.revoked_manifest_ids must not contain this manifest's own \
                         manifest_id '{mid}'"
                    ));
                }
            }
            Some(_) => {
                errs.push("compatibility.revoked_manifest_ids must be a JSON array".to_string())
            }
            None => errs.push("'revoked_manifest_ids' is required".to_string()),
        }
        match compat.get("edge_channel_window") {
            Some(Value::Array(items)) if !items.is_empty() => {
                for (idx, item) in items.iter().enumerate() {
                    if item.as_str().is_none_or(str::is_empty) {
                        errs.push(format!(
                            "compatibility.edge_channel_window[{idx}] must be a non-empty string"
                        ));
                    }
                }
            }
            Some(Value::Array(_)) => errs.push(
                "compatibility.edge_channel_window must contain at least one entry".to_string(),
            ),
            Some(_) => {
                errs.push("compatibility.edge_channel_window must be a JSON array".to_string())
            }
            None => errs.push("'edge_channel_window' is required".to_string()),
        }
    }

    if let Some(rb) = require_object(root, "rollback", errs) {
        match rb.get("rollback_supported") {
            Some(Value::Bool(_)) => {}
            Some(_) => errs.push("rollback.rollback_supported must be a boolean".to_string()),
            None => errs.push("'rollback_supported' is required".to_string()),
        }
        match rb.get("previous_manifest_id") {
            Some(Value::Null) => {}
            Some(Value::String(s)) if !s.is_empty() => {}
            Some(Value::String(_)) => errs.push(
                "rollback.previous_manifest_id must be a non-empty string or null".to_string(),
            ),
            Some(_) => {
                errs.push("rollback.previous_manifest_id must be a string or null".to_string())
            }
            None => errs.push("'previous_manifest_id' is required".to_string()),
        }
    }

    if let Some(layout) = require_object(root, "package_layout", errs) {
        require_nonempty_string(layout, "root_relative_to_manifest", errs);
        require_nonempty_string(layout, "artifact_subdir", errs);
    }
}

fn validate_crates(stator: &Map<String, Value>, errs: &mut Vec<String>) {
    let crates = match stator.get("crates").and_then(Value::as_array) {
        Some(c) => c,
        None => {
            errs.push("stator.crates must be a JSON array".to_string());
            return;
        }
    };
    for required in REQUIRED_CRATES {
        let found = crates
            .iter()
            .filter_map(Value::as_object)
            .find(|c| c.get("name").and_then(Value::as_str) == Some(*required));
        match found {
            None => errs.push(format!(
                "stator.crates is missing required crate '{required}'"
            )),
            Some(c) => {
                if require_nonempty_string(c, "version", errs).is_none() {
                    errs.push(format!("stator.crates['{required}'].version must be set"));
                }
            }
        }
    }
}

fn validate_artifacts(root: &Map<String, Value>, errs: &mut Vec<String>) {
    let artifacts = match root.get("artifacts").and_then(Value::as_array) {
        Some(a) if !a.is_empty() => a,
        Some(_) => {
            errs.push("artifacts must contain at least one entry".to_string());
            return;
        }
        None => {
            errs.push("artifacts is required and must be a JSON array".to_string());
            return;
        }
    };
    for (idx, artifact) in artifacts.iter().enumerate() {
        let Some(a) = artifact.as_object() else {
            errs.push(format!("artifacts[{idx}] must be an object"));
            continue;
        };
        if let Some(ty) = require_nonempty_string(a, "artifact_type", errs)
            && !VALID_ARTIFACT_TYPES.contains(&ty.as_str())
        {
            errs.push(format!(
                "artifacts[{idx}].artifact_type '{ty}' is not a recognized type"
            ));
        }
        require_nonempty_string(a, "target_triple", errs);
        require_nonempty_string(a, "target_os", errs);
        require_nonempty_string(a, "target_arch", errs);
        require_nonempty_string(a, "cargo_profile", errs);
        require_uint(a, "size_bytes", errs);
        if let Some(path) = require_nonempty_string(a, "artifact_path", errs)
            && let Some(reason) = invalid_artifact_path_reason(&path)
        {
            errs.push(format!(
                "artifacts[{idx}].artifact_path is invalid: {reason}"
            ));
        }
        if let Some(algo) = require_nonempty_string(a, "digest_algorithm", errs)
            && !VALID_DIGEST_ALGORITHMS.contains(&algo.as_str())
        {
            errs.push(format!(
                "artifacts[{idx}].digest_algorithm '{algo}' is not an accepted algorithm \
                 (expected one of {VALID_DIGEST_ALGORITHMS:?})"
            ));
        }
        if let Some(hex) = require_nonempty_string(a, "digest_hex", errs)
            && !is_lower_hex(&hex)
        {
            errs.push(format!(
                "artifacts[{idx}].digest_hex must be lower-case hexadecimal"
            ));
        }
        if let Some(sig) = a.get("signature") {
            let Some(sig) = sig.as_object() else {
                errs.push(format!("artifacts[{idx}].signature must be an object"));
                continue;
            };
            require_nonempty_string(sig, "algorithm", errs);
            if let Some(hex) = require_nonempty_string(sig, "value_hex", errs)
                && !is_lower_hex(&hex)
            {
                errs.push(format!(
                    "artifacts[{idx}].signature.value_hex must be lower-case hexadecimal"
                ));
            }
        } else {
            errs.push(format!("artifacts[{idx}].signature is required"));
        }
    }
}

fn validate_telemetry(root: &Map<String, Value>, errs: &mut Vec<String>) {
    let Some(t) = require_object(root, "telemetry", errs) else {
        return;
    };
    let codes = match t.get("diagnostic_codes").and_then(Value::as_array) {
        Some(c) => c,
        None => {
            errs.push("telemetry.diagnostic_codes must be a JSON array".to_string());
            return;
        }
    };
    let present: Vec<&str> = codes.iter().filter_map(Value::as_str).collect();
    for required in REQUIRED_DIAGNOSTIC_CODES {
        if !present.contains(required) {
            errs.push(format!(
                "telemetry.diagnostic_codes is missing required code '{required}'"
            ));
        }
    }
    if !t
        .get("field_allowlist")
        .map(|v| v.is_array())
        .unwrap_or(false)
    {
        errs.push("telemetry.field_allowlist must be a JSON array".to_string());
    }
}

fn require_object<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    errs: &mut Vec<String>,
) -> Option<&'a Map<String, Value>> {
    match obj.get(key) {
        Some(Value::Object(o)) => Some(o),
        Some(_) => {
            errs.push(format!("'{key}' must be a JSON object"));
            None
        }
        None => {
            errs.push(format!("'{key}' is required"));
            None
        }
    }
}

fn require_uint(obj: &Map<String, Value>, key: &str, errs: &mut Vec<String>) -> Option<u64> {
    match obj.get(key) {
        Some(v) => match v.as_u64() {
            Some(n) => Some(n),
            None => {
                errs.push(format!("'{key}' must be a non-negative integer"));
                None
            }
        },
        None => {
            errs.push(format!("'{key}' is required"));
            None
        }
    }
}

fn require_uint_eq(obj: &Map<String, Value>, key: &str, expected: u64, errs: &mut Vec<String>) {
    if let Some(n) = require_uint(obj, key, errs)
        && n != expected
    {
        errs.push(format!("'{key}' must equal {expected}, got {n}"));
    }
}

fn require_nonempty_string(
    obj: &Map<String, Value>,
    key: &str,
    errs: &mut Vec<String>,
) -> Option<String> {
    match obj.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::String(_)) => {
            errs.push(format!("'{key}' must be a non-empty string"));
            None
        }
        Some(_) => {
            errs.push(format!("'{key}' must be a string"));
            None
        }
        None => {
            errs.push(format!("'{key}' is required"));
            None
        }
    }
}

fn require_string_eq(obj: &Map<String, Value>, key: &str, expected: &str, errs: &mut Vec<String>) {
    if let Some(s) = require_nonempty_string(obj, key, errs)
        && s != expected
    {
        errs.push(format!("'{key}' must equal '{expected}', got '{s}'"));
    }
}

fn is_lower_hex(s: &str) -> bool {
    !s.is_empty()
        && s.len().is_multiple_of(2)
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Reject artifact paths that would let a vendored manifest escape its
/// packaging root, alias an absolute system path, or rely on
/// platform-specific separators. The packaging validator joins these
/// paths against the on-disk manifest root, so any traversal segment
/// (`..`), absolute prefix (`/`, drive letter), backslash separator, or
/// embedded NUL must fail closed before any I/O happens. Returns `None`
/// when the path is acceptable, or `Some(reason)` with a short
/// human-readable diagnostic otherwise.
fn invalid_artifact_path_reason(path: &str) -> Option<&'static str> {
    if path.is_empty() {
        return Some("must not be empty");
    }
    if path.contains('\\') {
        return Some("must use forward slashes (POSIX-style relative path)");
    }
    if path.contains('\0') {
        return Some("must not contain a NUL byte");
    }
    if path.starts_with('/') {
        return Some("must not start with '/'");
    }
    // Reject Windows-style drive prefixes like `C:` so the manifest
    // cannot be interpreted as an absolute path on Windows hosts.
    if path.len() >= 2 && path.as_bytes()[1] == b':' && path.as_bytes()[0].is_ascii_alphabetic() {
        return Some("must not contain a drive letter prefix");
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            return Some("must not contain empty path segments");
        }
        if segment == "." || segment == ".." {
            return Some("must not contain '.' or '..' segments");
        }
    }
    None
}

/// Validate that the on-disk Edge code-cache package layout under
/// `artifact_root` matches the manifest. This is the fail-closed
/// packaging hook: every artifact declared in the manifest must exist
/// as a regular file at `artifact_root.join(package_layout.artifact_subdir).join(artifact_path)`,
/// its size must match `size_bytes`, and (when `digest_algorithm` is
/// `sha256`) its on-disk content must hash to `digest_hex`. Missing
/// payloads, length mismatches, or digest mismatches are reported with
/// the `rejected_release_artifact` diagnostic vocabulary so vendoring
/// automation cannot publish a drop whose runtime artifact payloads are
/// absent or tampered with. Manifests whose schema is itself invalid
/// (as reported by `validate`) must be rejected before calling this
/// helper.
fn validate_packaging_layout(manifest: &Value, artifact_root: &std::path::Path) -> Vec<String> {
    use sha2::{Digest, Sha256};

    let mut errs = Vec::new();
    let Some(root) = manifest.as_object() else {
        errs.push("manifest root must be a JSON object".to_string());
        return errs;
    };
    let subdir = root
        .get("package_layout")
        .and_then(Value::as_object)
        .and_then(|l| l.get("artifact_subdir"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if subdir.is_empty() || invalid_artifact_path_reason(subdir).is_some() {
        errs.push(
            "rejected_release_artifact: package_layout.artifact_subdir is missing or invalid"
                .to_string(),
        );
        return errs;
    }
    let Some(artifacts) = root.get("artifacts").and_then(Value::as_array) else {
        errs.push("rejected_release_artifact: artifacts is missing".to_string());
        return errs;
    };
    for (idx, artifact) in artifacts.iter().enumerate() {
        let Some(a) = artifact.as_object() else {
            errs.push(format!(
                "rejected_release_artifact: artifacts[{idx}] must be an object"
            ));
            continue;
        };
        let Some(rel) = a.get("artifact_path").and_then(Value::as_str) else {
            errs.push(format!(
                "rejected_release_artifact: artifacts[{idx}].artifact_path is missing"
            ));
            continue;
        };
        if let Some(reason) = invalid_artifact_path_reason(rel) {
            errs.push(format!(
                "rejected_release_artifact: artifacts[{idx}].artifact_path is invalid: {reason}"
            ));
            continue;
        }
        let on_disk = artifact_root.join(subdir).join(rel);
        let bytes = match std::fs::read(&on_disk) {
            Ok(b) => b,
            Err(e) => {
                errs.push(format!(
                    "rejected_release_artifact: artifacts[{idx}] payload at {} is unreadable: {e}",
                    on_disk.display()
                ));
                continue;
            }
        };
        if let Some(expected_size) = a.get("size_bytes").and_then(Value::as_u64)
            && bytes.len() as u64 != expected_size
        {
            errs.push(format!(
                "rejected_release_artifact: artifacts[{idx}] size mismatch (manifest {}, on disk {})",
                expected_size,
                bytes.len()
            ));
        }
        let algo = a
            .get("digest_algorithm")
            .and_then(Value::as_str)
            .unwrap_or("");
        let expected_hex = a.get("digest_hex").and_then(Value::as_str).unwrap_or("");
        if algo == "sha256" && !expected_hex.is_empty() {
            let digest = Sha256::digest(&bytes);
            let actual_hex: String = digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            if actual_hex != expected_hex {
                errs.push(format!(
                    "rejected_release_artifact: artifacts[{idx}] sha256 digest mismatch \
                     (manifest {expected_hex}, on disk {actual_hex})"
                ));
            }
        }
    }
    errs
}

fn baseline_manifest() -> Value {
    json!({
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": "2026-05-21T00:00:00Z",
        "stator": {
            "commit": "0000000000000000000000000000000000000000",
            "crates": [
                {"name": "stator_jse", "version": "0.3.5"},
                {"name": "stator_jse_ffi", "version": "0.3.7"},
                {"name": "st8", "version": "0.2.3"}
            ]
        },
        "ffi_abi": {
            "major": 1,
            "minor": 0,
            "patch": 0,
            "packed": 65536,
            "c_header_generation_id": "stator.h@0000000"
        },
        "cache_formats": {
            "cache_schema_version": 1,
            "script_cache_format_version": 1,
            "module_cache_format_version": 1,
            "bytecode_format_version": 1,
            "baseline_code_format_version": 1,
            "jit_code_format_version": 1,
            "snapshot_format_version": 1
        },
        "key_schema": {
            "key_schema_version": 1,
            "canonical_key_hash_algorithm": "sha256"
        },
        "artifacts": [
            {
                "artifact_type": "script-bytecode",
                "target_triple": "x86_64-pc-windows-msvc",
                "target_os": "windows",
                "target_arch": "x86_64",
                "cargo_profile": "release",
                "size_bytes": 1024,
                "artifact_path": "script-bytecode/example.scbc",
                "digest_algorithm": "sha256",
                "digest_hex": "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
                "signature": {
                    "algorithm": "ed25519",
                    "value_hex": "deadbeef"
                }
            }
        ],
        "manifest_id": "stator-edge-cache-2026-05-21-0000",
        "compatibility": {
            "min_compatible_manifest_id": "stator-edge-cache-2026-05-21-0000",
            "revoked_manifest_ids": [],
            "eviction_policy": "partition-by-manifest-id",
            "edge_channel_window": ["canary", "dev", "beta", "stable"]
        },
        "rollback": {
            "rollback_supported": true,
            "previous_manifest_id": null
        },
        "package_layout": {
            "root_relative_to_manifest": ".",
            "artifact_subdir": "cache"
        },
        "telemetry": {
            "diagnostic_codes": [
                "accepted",
                "miss_not_found",
                "rejected_schema_version",
                "rejected_artifact_type",
                "rejected_engine_version",
                "rejected_format_version",
                "rejected_source_identity",
                "rejected_embedder_policy",
                "rejected_parser_flags",
                "rejected_compiler_flags",
                "rejected_platform",
                "rejected_build_features",
                "rejected_snapshot",
                "rejected_release_artifact",
                "corrupt_payload",
                "unsupported_native_code"
            ],
            "field_allowlist": [
                "artifact_type",
                "cache_schema_version",
                "stator_ffi_abi_version"
            ]
        }
    })
}

fn assert_ok(m: &Value) {
    let errs = validate(m);
    assert!(
        errs.is_empty(),
        "expected manifest to validate, got: {errs:#?}"
    );
}

fn assert_err_contains(m: &Value, needle: &str) {
    let errs = validate(m);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "expected error containing '{needle}', got: {errs:#?}"
    );
}

#[test]
fn test_validate_baseline_accepts() {
    assert_ok(&baseline_manifest());
}

#[test]
fn test_validate_root_must_be_object() {
    let m = json!([]);
    assert_err_contains(&m, "manifest root must be a JSON object");
}

#[test]
fn test_validate_rejects_missing_schema_id() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("schema_id");
    assert_err_contains(&m, "'schema_id' is required");
}

#[test]
fn test_validate_rejects_wrong_schema_id() {
    let mut m = baseline_manifest();
    m["schema_id"] = json!("other-schema");
    assert_err_contains(&m, "must equal 'stator-edge-code-cache-release-manifest'");
}

#[test]
fn test_validate_rejects_wrong_schema_version() {
    let mut m = baseline_manifest();
    m["schema_version"] = json!(2);
    assert_err_contains(&m, "'schema_version' must equal 1");
}

#[test]
fn test_validate_rejects_missing_generated_at() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("generated_at_utc");
    assert_err_contains(&m, "'generated_at_utc' is required");
}

#[test]
fn test_validate_rejects_missing_commit() {
    let mut m = baseline_manifest();
    m["stator"].as_object_mut().unwrap().remove("commit");
    assert_err_contains(&m, "'commit' is required");
}

#[test]
fn test_validate_rejects_missing_required_crate() {
    let mut m = baseline_manifest();
    let crates = m["stator"]["crates"].as_array_mut().unwrap();
    crates.retain(|c| c["name"] != "stator_jse_ffi");
    assert_err_contains(&m, "missing required crate 'stator_jse_ffi'");
}

#[test]
fn test_validate_rejects_crate_without_version() {
    let mut m = baseline_manifest();
    let crates = m["stator"]["crates"].as_array_mut().unwrap();
    for c in crates {
        if c["name"] == "st8" {
            c.as_object_mut().unwrap().remove("version");
        }
    }
    assert_err_contains(&m, "'version' is required");
}

#[test]
fn test_validate_rejects_missing_ffi_abi() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("ffi_abi");
    assert_err_contains(&m, "'ffi_abi' is required");
}

#[test]
fn test_validate_rejects_inconsistent_ffi_abi_packed() {
    let mut m = baseline_manifest();
    m["ffi_abi"]["packed"] = json!(1);
    assert_err_contains(&m, "ffi_abi.packed");
}

#[test]
fn test_validate_rejects_missing_c_header_generation_id() {
    let mut m = baseline_manifest();
    m["ffi_abi"]
        .as_object_mut()
        .unwrap()
        .remove("c_header_generation_id");
    assert_err_contains(&m, "'c_header_generation_id' is required");
}

#[test]
fn test_validate_rejects_each_missing_cache_format_field() {
    for field in REQUIRED_CACHE_FORMAT_FIELDS {
        let mut m = baseline_manifest();
        m["cache_formats"].as_object_mut().unwrap().remove(*field);
        assert_err_contains(&m, &format!("'{field}' is required"));
    }
}

#[test]
fn test_validate_rejects_non_integer_cache_format_field() {
    let mut m = baseline_manifest();
    m["cache_formats"]["bytecode_format_version"] = json!("v1");
    assert_err_contains(&m, "must be a non-negative integer");
}

#[test]
fn test_validate_rejects_missing_key_schema_version() {
    let mut m = baseline_manifest();
    m["key_schema"]
        .as_object_mut()
        .unwrap()
        .remove("key_schema_version");
    assert_err_contains(&m, "'key_schema_version' is required");
}

#[test]
fn test_validate_rejects_unsupported_key_hash_algorithm() {
    let mut m = baseline_manifest();
    m["key_schema"]["canonical_key_hash_algorithm"] = json!("md5");
    assert_err_contains(&m, "not an accepted algorithm");
}

#[test]
fn test_validate_rejects_empty_artifacts() {
    let mut m = baseline_manifest();
    m["artifacts"] = json!([]);
    assert_err_contains(&m, "artifacts must contain at least one entry");
}

#[test]
fn test_validate_rejects_unknown_artifact_type() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_type"] = json!("turbofan-blob");
    assert_err_contains(&m, "is not a recognized type");
}

#[test]
fn test_validate_rejects_missing_target_triple() {
    let mut m = baseline_manifest();
    m["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_triple");
    assert_err_contains(&m, "'target_triple' is required");
}

#[test]
fn test_validate_rejects_missing_target_os_arch_profile() {
    for field in ["target_os", "target_arch", "cargo_profile"] {
        let mut m = baseline_manifest();
        m["artifacts"][0].as_object_mut().unwrap().remove(field);
        assert_err_contains(&m, &format!("'{field}' is required"));
    }
}

#[test]
fn test_validate_rejects_unsupported_digest_algorithm() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["digest_algorithm"] = json!("crc32");
    assert_err_contains(&m, "not an accepted algorithm");
}

#[test]
fn test_validate_rejects_non_hex_digest() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["digest_hex"] = json!("not-hex!");
    assert_err_contains(&m, "must be lower-case hexadecimal");
}

#[test]
fn test_validate_rejects_missing_signature() {
    let mut m = baseline_manifest();
    m["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("signature");
    assert_err_contains(&m, "signature is required");
}

#[test]
fn test_validate_rejects_missing_telemetry_codes() {
    for code in REQUIRED_DIAGNOSTIC_CODES {
        let mut m = baseline_manifest();
        let arr = m["telemetry"]["diagnostic_codes"].as_array_mut().unwrap();
        arr.retain(|v| v.as_str() != Some(*code));
        assert_err_contains(&m, &format!("missing required code '{code}'"));
    }
}

#[test]
fn test_validate_rejects_missing_field_allowlist() {
    let mut m = baseline_manifest();
    m["telemetry"]
        .as_object_mut()
        .unwrap()
        .remove("field_allowlist");
    assert_err_contains(&m, "telemetry.field_allowlist");
}

// ---------------------------------------------------------------------------
// Diagnostics contract: ensure the validator's required code list and the
// field-to-code map stay synchronized with docs/code_cache.md, and that the
// fail-closed digest/signature checks reject every malformed shape Edge can
// produce (uppercase hex, odd-length hex, non-hex bytes, missing signature
// value).
// ---------------------------------------------------------------------------

/// Path to `docs/code_cache.md` relative to this crate's `CARGO_MANIFEST_DIR`.
const CODE_CACHE_DOC: &str = "../../docs/code_cache.md";

fn read_code_cache_doc() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir).join(CODE_CACHE_DOC);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

#[test]
fn test_field_diagnostic_map_codes_are_all_required() {
    for (field, code) in FIELD_TO_DIAGNOSTIC_CODE {
        assert!(
            REQUIRED_DIAGNOSTIC_CODES.contains(code),
            "field '{field}' maps to code '{code}' which is not in REQUIRED_DIAGNOSTIC_CODES",
        );
    }
}

#[test]
fn test_field_diagnostic_map_is_unique() {
    for (i, (field, _)) in FIELD_TO_DIAGNOSTIC_CODE.iter().enumerate() {
        for (j, (other, _)) in FIELD_TO_DIAGNOSTIC_CODE.iter().enumerate() {
            if i != j {
                assert_ne!(
                    field, other,
                    "field '{field}' appears more than once in FIELD_TO_DIAGNOSTIC_CODE",
                );
            }
        }
    }
}

#[test]
fn test_required_diagnostic_codes_match_docs() {
    let doc = read_code_cache_doc();
    for code in REQUIRED_DIAGNOSTIC_CODES {
        let needle = format!("`{code}`");
        assert!(
            doc.contains(&needle),
            "REQUIRED_DIAGNOSTIC_CODES entry '{code}' is not documented in docs/code_cache.md",
        );
    }
}

#[test]
fn test_field_diagnostic_map_matches_docs() {
    let doc = read_code_cache_doc();
    for (field, code) in FIELD_TO_DIAGNOSTIC_CODE {
        let field_tok = format!("`{field}`");
        let code_tok = format!("`{code}`");
        let row = doc.lines().find(|line| {
            line.starts_with('|') && line.contains(&field_tok) && line.contains(&code_tok)
        });
        assert!(
            row.is_some(),
            "docs/code_cache.md is missing the field-to-code row for ('{field}', '{code}')",
        );
    }
}

#[test]
fn test_validate_rejects_odd_length_hex_digest() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["digest_hex"] = json!("abc");
    assert_err_contains(&m, "must be lower-case hexadecimal");
}

#[test]
fn test_validate_rejects_uppercase_hex_digest() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["digest_hex"] = json!("AA");
    assert_err_contains(&m, "must be lower-case hexadecimal");
}

#[test]
fn test_validate_rejects_non_hex_signature_value() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["signature"]["value_hex"] = json!("not-hex!");
    assert_err_contains(&m, "signature.value_hex must be lower-case hexadecimal");
}

#[test]
fn test_validate_rejects_missing_signature_value() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["signature"]
        .as_object_mut()
        .unwrap()
        .remove("value_hex");
    assert_err_contains(&m, "'value_hex' is required");
}

#[test]
fn test_validate_rejects_missing_signature_algorithm() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["signature"]
        .as_object_mut()
        .unwrap()
        .remove("algorithm");
    assert_err_contains(&m, "'algorithm' is required");
}

#[test]
fn test_validate_rejects_missing_size_bytes() {
    let mut m = baseline_manifest();
    m["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("size_bytes");
    assert_err_contains(&m, "'size_bytes' is required");
}

// ---------------------------------------------------------------------------
// Packaging/invalidation/rollback contract: every manifest must declare a
// stable manifest identity, an Edge channel compatibility window, an
// explicit cache eviction policy, an unambiguous revocation list, the
// rollback contract, and a packaging layout that points at on-disk
// artifact payloads. Anything missing, malformed, or escaping the
// packaging root must fail closed so vendoring cannot publish a drop
// whose runtime cache artifacts are absent or tampered with.
// ---------------------------------------------------------------------------

#[test]
fn test_validate_rejects_missing_manifest_id() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("manifest_id");
    assert_err_contains(&m, "'manifest_id' is required");
}

#[test]
fn test_validate_rejects_empty_manifest_id() {
    let mut m = baseline_manifest();
    m["manifest_id"] = json!("");
    assert_err_contains(&m, "'manifest_id' must be a non-empty string");
}

#[test]
fn test_validate_rejects_missing_compatibility_block() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("compatibility");
    assert_err_contains(&m, "'compatibility' is required");
}

#[test]
fn test_validate_rejects_missing_min_compatible_manifest_id() {
    let mut m = baseline_manifest();
    m["compatibility"]
        .as_object_mut()
        .unwrap()
        .remove("min_compatible_manifest_id");
    assert_err_contains(&m, "'min_compatible_manifest_id' is required");
}

#[test]
fn test_validate_rejects_unknown_eviction_policy() {
    let mut m = baseline_manifest();
    m["compatibility"]["eviction_policy"] = json!("never");
    assert_err_contains(&m, "not an accepted policy");
}

#[test]
fn test_validate_rejects_missing_revoked_list() {
    let mut m = baseline_manifest();
    m["compatibility"]
        .as_object_mut()
        .unwrap()
        .remove("revoked_manifest_ids");
    assert_err_contains(&m, "'revoked_manifest_ids' is required");
}

#[test]
fn test_validate_rejects_duplicate_revoked_manifest_id() {
    let mut m = baseline_manifest();
    m["compatibility"]["revoked_manifest_ids"] = json!(["old-1", "old-1"]);
    assert_err_contains(&m, "duplicates 'old-1'");
}

#[test]
fn test_validate_rejects_self_revocation() {
    let mut m = baseline_manifest();
    let id = m["manifest_id"].as_str().unwrap().to_string();
    m["compatibility"]["revoked_manifest_ids"] = json!([id]);
    assert_err_contains(&m, "must not contain this manifest's own manifest_id");
}

#[test]
fn test_validate_rejects_non_string_revoked_entry() {
    let mut m = baseline_manifest();
    m["compatibility"]["revoked_manifest_ids"] = json!([42]);
    assert_err_contains(&m, "must be a non-empty string");
}

#[test]
fn test_validate_rejects_empty_channel_window() {
    let mut m = baseline_manifest();
    m["compatibility"]["edge_channel_window"] = json!([]);
    assert_err_contains(&m, "must contain at least one entry");
}

#[test]
fn test_validate_rejects_missing_rollback_block() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("rollback");
    assert_err_contains(&m, "'rollback' is required");
}

#[test]
fn test_validate_rejects_missing_rollback_supported() {
    let mut m = baseline_manifest();
    m["rollback"]
        .as_object_mut()
        .unwrap()
        .remove("rollback_supported");
    assert_err_contains(&m, "'rollback_supported' is required");
}

#[test]
fn test_validate_rejects_non_bool_rollback_supported() {
    let mut m = baseline_manifest();
    m["rollback"]["rollback_supported"] = json!("yes");
    assert_err_contains(&m, "must be a boolean");
}

#[test]
fn test_validate_rejects_missing_previous_manifest_id_field() {
    let mut m = baseline_manifest();
    m["rollback"]
        .as_object_mut()
        .unwrap()
        .remove("previous_manifest_id");
    assert_err_contains(&m, "'previous_manifest_id' is required");
}

#[test]
fn test_validate_accepts_null_previous_manifest_id() {
    let mut m = baseline_manifest();
    m["rollback"]["previous_manifest_id"] = json!(null);
    assert_ok(&m);
}

#[test]
fn test_validate_rejects_missing_package_layout() {
    let mut m = baseline_manifest();
    m.as_object_mut().unwrap().remove("package_layout");
    assert_err_contains(&m, "'package_layout' is required");
}

#[test]
fn test_validate_rejects_missing_artifact_subdir() {
    let mut m = baseline_manifest();
    m["package_layout"]
        .as_object_mut()
        .unwrap()
        .remove("artifact_subdir");
    assert_err_contains(&m, "'artifact_subdir' is required");
}

#[test]
fn test_validate_rejects_missing_artifact_path() {
    let mut m = baseline_manifest();
    m["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("artifact_path");
    assert_err_contains(&m, "'artifact_path' is required");
}

#[test]
fn test_validate_rejects_traversal_artifact_path() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_path"] = json!("../escape.bin");
    assert_err_contains(&m, "must not contain '.' or '..' segments");
}

#[test]
fn test_validate_rejects_absolute_artifact_path() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_path"] = json!("/etc/passwd");
    assert_err_contains(&m, "must not start with '/'");
}

#[test]
fn test_validate_rejects_backslash_artifact_path() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_path"] = json!("cache\\foo.bin");
    assert_err_contains(&m, "must use forward slashes");
}

#[test]
fn test_validate_rejects_drive_letter_artifact_path() {
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_path"] = json!("C:/cache/foo.bin");
    assert_err_contains(&m, "must not contain a drive letter prefix");
}

// ---------------------------------------------------------------------------
// Fail-closed on-disk packaging validator: missing/short/tampered payloads
// must reject under the `rejected_release_artifact` vocabulary instead of
// silently accepting an Edge drop whose runtime cache files are absent or
// corrupted. Manifests in this section synthesize their digests from the
// payload bytes written to disk so the test is hermetic.
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn packaging_tmpdir(label: &str) -> std::path::PathBuf {
    // Cargo sets CARGO_TARGET_TMPDIR for integration tests; fall back to
    // OUT_DIR if for some reason it is not set. This keeps test artifacts
    // inside the workspace target directory and never touches /tmp.
    let base = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("OUT_DIR").map(std::path::PathBuf::from))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = base.join(format!("release_manifest_pkg_{label}_{pid}_{nanos}"));
    std::fs::create_dir_all(&dir).expect("create packaging tmpdir");
    dir
}

fn write_payload(root: &std::path::Path, subdir: &str, rel: &str, bytes: &[u8]) {
    let path = root.join(subdir).join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir packaging subdir");
    std::fs::write(&path, bytes).expect("write packaging payload");
}

#[test]
fn test_packaging_layout_accepts_matching_payload() {
    let root = packaging_tmpdir("accept");
    let payload = b"hello-bytecode-payload";
    let rel = "script-bytecode/example.scbc";
    write_payload(&root, "cache", rel, payload);

    let mut m = baseline_manifest();
    m["artifacts"][0]["size_bytes"] = json!(payload.len() as u64);
    m["artifacts"][0]["digest_hex"] = json!(sha256_hex(payload));
    m["artifacts"][0]["artifact_path"] = json!(rel);

    let errs = validate_packaging_layout(&m, &root);
    assert!(
        errs.is_empty(),
        "expected no packaging errors, got {errs:?}"
    );
}

#[test]
fn test_packaging_layout_rejects_missing_payload() {
    let root = packaging_tmpdir("missing");
    let m = baseline_manifest();
    let errs = validate_packaging_layout(&m, &root);
    assert!(
        errs.iter()
            .any(|e| e.contains("rejected_release_artifact") && e.contains("unreadable")),
        "expected missing-payload rejection, got {errs:?}"
    );
}

#[test]
fn test_packaging_layout_rejects_size_mismatch() {
    let root = packaging_tmpdir("size");
    let payload = b"short";
    let rel = "script-bytecode/example.scbc";
    write_payload(&root, "cache", rel, payload);

    let mut m = baseline_manifest();
    m["artifacts"][0]["size_bytes"] = json!(9999u64);
    m["artifacts"][0]["digest_hex"] = json!(sha256_hex(payload));
    m["artifacts"][0]["artifact_path"] = json!(rel);

    let errs = validate_packaging_layout(&m, &root);
    assert!(
        errs.iter()
            .any(|e| e.contains("rejected_release_artifact") && e.contains("size mismatch")),
        "expected size-mismatch rejection, got {errs:?}"
    );
}

#[test]
fn test_packaging_layout_rejects_digest_mismatch() {
    let root = packaging_tmpdir("digest");
    let payload = b"hello-payload";
    let rel = "script-bytecode/example.scbc";
    write_payload(&root, "cache", rel, payload);

    let mut m = baseline_manifest();
    m["artifacts"][0]["size_bytes"] = json!(payload.len() as u64);
    m["artifacts"][0]["digest_hex"] =
        json!("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
    m["artifacts"][0]["artifact_path"] = json!(rel);

    let errs = validate_packaging_layout(&m, &root);
    assert!(
        errs.iter().any(
            |e| e.contains("rejected_release_artifact") && e.contains("sha256 digest mismatch")
        ),
        "expected digest-mismatch rejection, got {errs:?}"
    );
}

#[test]
fn test_packaging_layout_rejects_traversal_path_without_io() {
    // Even when the on-disk file would exist, the validator must refuse to
    // touch a traversal path so a malicious manifest can never read outside
    // its packaging root.
    let root = packaging_tmpdir("traversal");
    let mut m = baseline_manifest();
    m["artifacts"][0]["artifact_path"] = json!("../escape.bin");
    let errs = validate_packaging_layout(&m, &root);
    assert!(
        errs.iter()
            .any(|e| e.contains("rejected_release_artifact") && e.contains("invalid")),
        "expected traversal rejection, got {errs:?}"
    );
}

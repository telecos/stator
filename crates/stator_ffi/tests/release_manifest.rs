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

/// Hash algorithms accepted for canonical key hashing. Per
/// `docs/code_cache.md` Section "Canonical serialization", the first
/// supported algorithm is `sha256`; non-cryptographic checksums are not
/// valid key hashes and must be rejected here too.
const VALID_KEY_HASH_ALGORITHMS: &[&str] = &["sha256"];

/// Digest algorithms accepted for per-artifact integrity fields.
const VALID_DIGEST_ALGORITHMS: &[&str] = &["sha256", "sha384", "sha512"];

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

    errs
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
                "digest_algorithm": "sha256",
                "digest_hex": "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
                "signature": {
                    "algorithm": "ed25519",
                    "value_hex": "deadbeef"
                }
            }
        ],
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

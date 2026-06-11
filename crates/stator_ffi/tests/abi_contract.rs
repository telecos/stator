//! ABI contract gate for `stator_jse_ffi`.
//!
//! These tests are the first concrete release-train check for the Stator
//! C ABI surface.  They validate three things:
//!
//! 1. The packed ABI version constant exposed by the generated C header
//!    matches the value returned by the exported `stator_ffi_abi_version`
//!    function, so embedders cannot silently link a header against a
//!    differently versioned library.
//! 2. The generated header still contains the `STATOR_FFI_ABI_VERSION*`
//!    markers and the `stator_ffi_abi_version` function declaration, so a
//!    future cbindgen/export change can not accidentally drop the version
//!    contract.
//! 3. The exported `stator_*` function surface, as observed in the
//!    generated `include/stator.h`, matches a checked-in baseline at
//!    `tests/abi_symbols.baseline.txt`.  Any removal or rename of an
//!    exported function will fail this test; additions also fail to force
//!    a deliberate baseline refresh as part of the release-train workflow.
//!
//! When intentionally adding/removing exported functions, regenerate the
//! baseline by running the test once with the environment variable
//! `STATOR_FFI_ABI_UPDATE_BASELINE=1` set.  The test will rewrite
//! `tests/abi_symbols.baseline.txt` from the current header and pass.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Path to the cbindgen-generated public C header, resolved relative to the
/// crate manifest directory.
fn header_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("include")
        .join("stator.h")
}

/// Path to the checked-in ABI baseline file containing one exported
/// `stator_*` function name per line, sorted alphabetically.
fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("abi_symbols.baseline.txt")
}

/// Extract every `stator_*` function name that appears as a function
/// declaration in the generated header.  The header is produced by cbindgen
/// in a predictable shape: each declaration sits on a single line that
/// starts with the return type, is followed by the function name, and ends
/// with an open parenthesis.
fn parse_exported_functions(header: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in header.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with('*') || trimmed.starts_with("/*") {
            continue;
        }
        let Some(paren) = trimmed.find('(') else {
            continue;
        };
        let head = &trimmed[..paren];
        let Some(name_start) = head.rfind(|c: char| c.is_whitespace() || c == '*') else {
            continue;
        };
        let name = head[name_start + 1..].trim();
        if name.starts_with("stator_")
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            out.insert(name.to_string());
        }
    }
    out
}

#[test]
fn test_parse_exported_functions_handles_pointer_returns_and_noise() {
    let header = r#"
        uint32_t stator_ffi_abi_version(void);
        const char *stator_native_code_cache_diagnostic_name(StatorNativeCodeCacheDiagnostic diagnostic);
        // void stator_commented_out(void);
        #define stator_not_a_function 1
        typedef void (*StatorCallback)(void);
    "#;
    let parsed = parse_exported_functions(header);
    assert_eq!(
        parsed,
        BTreeSet::from([
            "stator_ffi_abi_version".to_string(),
            "stator_native_code_cache_diagnostic_name".to_string(),
        ])
    );
}

#[test]
fn test_abi_version_function_matches_constant() {
    let runtime = stator_jse_ffi::stator_ffi_abi_version();
    assert_eq!(
        runtime,
        stator_jse_ffi::STATOR_FFI_ABI_VERSION,
        "stator_ffi_abi_version() must equal STATOR_FFI_ABI_VERSION"
    );

    let major = stator_jse_ffi::STATOR_FFI_ABI_VERSION_MAJOR;
    let minor = stator_jse_ffi::STATOR_FFI_ABI_VERSION_MINOR;
    let patch = stator_jse_ffi::STATOR_FFI_ABI_VERSION_PATCH;
    let packed = (major << 16) | (minor << 8) | patch;
    assert_eq!(
        runtime, packed,
        "packed ABI version must equal (major << 16) | (minor << 8) | patch"
    );

    assert_eq!(stator_jse_ffi::stator_ffi_abi_version_major(), major);
    assert_eq!(stator_jse_ffi::stator_ffi_abi_version_minor(), minor);
    assert_eq!(stator_jse_ffi::stator_ffi_abi_version_patch(), patch);
}

/// Parse the integer value of a `#define <name> <int-literal>` line from the
/// generated header, returning `None` if the macro is missing or its value is
/// not a bare decimal integer.  Only the simple `#define NAME N` shape used
/// for the ABI version components is recognised; the packed
/// `STATOR_FFI_ABI_VERSION` macro uses an expression and is handled
/// separately.
fn parse_define_u32(header: &str, name: &str) -> Option<u32> {
    for line in header.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("#define") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(after_name) = rest.strip_prefix(name) else {
            continue;
        };
        let value = after_name.trim();
        if value.is_empty() {
            continue;
        }
        if let Ok(v) = value.parse::<u32>() {
            return Some(v);
        }
    }
    None
}

#[test]
fn test_header_abi_version_macros_match_rust_constants() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");

    let header_major = parse_define_u32(&header, "STATOR_FFI_ABI_VERSION_MAJOR")
        .expect("generated stator.h must define STATOR_FFI_ABI_VERSION_MAJOR as a decimal integer");
    let header_minor = parse_define_u32(&header, "STATOR_FFI_ABI_VERSION_MINOR")
        .expect("generated stator.h must define STATOR_FFI_ABI_VERSION_MINOR as a decimal integer");
    let header_patch = parse_define_u32(&header, "STATOR_FFI_ABI_VERSION_PATCH")
        .expect("generated stator.h must define STATOR_FFI_ABI_VERSION_PATCH as a decimal integer");

    assert_eq!(
        header_major,
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_MAJOR,
        "stator.h STATOR_FFI_ABI_VERSION_MAJOR ({header_major}) drifted from Rust constant ({}); regenerate the header by running `cargo build -p stator_jse_ffi`",
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_MAJOR,
    );
    assert_eq!(
        header_minor,
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_MINOR,
        "stator.h STATOR_FFI_ABI_VERSION_MINOR ({header_minor}) drifted from Rust constant ({}); regenerate the header by running `cargo build -p stator_jse_ffi`",
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_MINOR,
    );
    assert_eq!(
        header_patch,
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_PATCH,
        "stator.h STATOR_FFI_ABI_VERSION_PATCH ({header_patch}) drifted from Rust constant ({}); regenerate the header by running `cargo build -p stator_jse_ffi`",
        stator_jse_ffi::STATOR_FFI_ABI_VERSION_PATCH,
    );
}

#[test]
fn test_header_native_code_cache_constants_and_diagnostics_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");

    assert_eq!(
        parse_define_u32(&header, "STATOR_NATIVE_CODE_CACHE_DIGEST_LEN"),
        Some(stator_jse_ffi::native_code_cache::STATOR_NATIVE_CODE_CACHE_DIGEST_LEN as u32),
        "generated stator.h native code-cache digest length drifted from Rust constant"
    );
    assert_eq!(
        parse_define_u32(&header, "STATOR_NATIVE_CODE_CACHE_HEADER_SIZE"),
        Some(stator_jse_ffi::native_code_cache::STATOR_NATIVE_CODE_CACHE_HEADER_SIZE as u32),
        "generated stator.h native code-cache header size drifted from Rust constant"
    );

    for (name, discriminant) in [
        ("StatorNativeCodeCacheDiagnosticAccepted", 0),
        ("StatorNativeCodeCacheDiagnosticInvalidArgument", 1),
        ("StatorNativeCodeCacheDiagnosticCorruptPayload", 2),
        ("StatorNativeCodeCacheDiagnosticRejectedArtifactType", 3),
        ("StatorNativeCodeCacheDiagnosticRejectedEngineVersion", 4),
        ("StatorNativeCodeCacheDiagnosticRejectedFormatVersion", 5),
        ("StatorNativeCodeCacheDiagnosticRejectedSourceIdentity", 6),
        ("StatorNativeCodeCacheDiagnosticRejectedPlatform", 7),
        ("StatorNativeCodeCacheDiagnosticRejectedBuildFeatures", 8),
        ("StatorNativeCodeCacheDiagnosticRejectedCompilerFlags", 9),
        ("StatorNativeCodeCacheDiagnosticUnsupportedNativeCode", 10),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable native code-cache diagnostic `{marker}`"
        );
    }
}

#[test]
fn test_header_native_code_cache_function_signatures_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "size_t stator_native_code_cache_header_size(void);",
        "const char *stator_native_code_cache_diagnostic_name(enum StatorNativeCodeCacheDiagnostic diagnostic);",
        "enum StatorNativeCodeCacheDiagnostic stator_native_code_cache_classify_header(const uint8_t *bytes,\n                                                                              size_t len,\n                                                                              struct StatorNativeCodeCacheHeaderInfo *out_info);",
        "enum StatorNativeCodeCacheDiagnostic stator_native_code_cache_validate_header(const uint8_t *bytes,\n                                                                              size_t len,\n                                                                              const struct StatorNativeCodeCacheCompatibility *expected,\n                                                                              struct StatorNativeCodeCacheHeaderInfo *out_info);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h native code-cache signature drifted:\n{signature}"
        );
    }
}

#[test]
fn test_header_contains_abi_version_markers() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for marker in [
        "STATOR_FFI_ABI_VERSION_MAJOR",
        "STATOR_FFI_ABI_VERSION_MINOR",
        "STATOR_FFI_ABI_VERSION_PATCH",
        "STATOR_FFI_ABI_VERSION",
        "stator_ffi_abi_version(",
        "stator_ffi_abi_version_major(",
        "stator_ffi_abi_version_minor(",
        "stator_ffi_abi_version_patch(",
    ] {
        assert!(
            header.contains(marker),
            "generated stator.h is missing ABI marker `{marker}`; cbindgen export list may have regressed"
        );
    }
    for signature in [
        "uint32_t stator_ffi_abi_version(void);",
        "uint32_t stator_ffi_abi_version_major(void);",
        "uint32_t stator_ffi_abi_version_minor(void);",
        "uint32_t stator_ffi_abi_version_patch(void);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h ABI version signature drifted:\n{signature}"
        );
    }
}

#[test]
fn test_exported_symbol_surface_matches_baseline() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    let current = parse_exported_functions(&header);

    // Defensive sanity check: parsing must at least find representative
    // scalar-return and pointer-return accessors. If it doesn't, the parser
    // regressed rather than the ABI.
    for required in [
        "stator_ffi_abi_version",
        "stator_native_code_cache_diagnostic_name",
    ] {
        assert!(
            current.contains(required),
            "header parser failed to find {required} — parser regressed"
        );
    }

    if std::env::var_os("STATOR_FFI_ABI_UPDATE_BASELINE").is_some() {
        let mut body = String::new();
        for name in &current {
            body.push_str(name);
            body.push('\n');
        }
        fs::write(baseline_path(), body).expect("failed to write updated baseline");
        return;
    }

    let baseline_raw = fs::read_to_string(baseline_path()).expect("ABI baseline file must exist");
    let baseline: BTreeSet<String> = baseline_raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();

    let added: Vec<&String> = current.difference(&baseline).collect();
    let removed: Vec<&String> = baseline.difference(&current).collect();

    if !added.is_empty() || !removed.is_empty() {
        let mut msg = String::from(
            "Stator FFI ABI surface drifted from the checked-in baseline.\n\
             Run `STATOR_FFI_ABI_UPDATE_BASELINE=1 cargo test -p stator_jse_ffi --test abi_contract` \
             to refresh tests/abi_symbols.baseline.txt after an intentional ABI change, \
             and bump STATOR_FFI_ABI_VERSION_{MAJOR,MINOR} accordingly.\n",
        );
        if !added.is_empty() {
            msg.push_str("\nAdded exports:\n");
            for name in &added {
                msg.push_str("  + ");
                msg.push_str(name);
                msg.push('\n');
            }
        }
        if !removed.is_empty() {
            msg.push_str("\nRemoved exports:\n");
            for name in &removed {
                msg.push_str("  - ");
                msg.push_str(name);
                msg.push('\n');
            }
        }
        panic!("{msg}");
    }
}

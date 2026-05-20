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
}

#[test]
fn test_exported_symbol_surface_matches_baseline() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    let current = parse_exported_functions(&header);

    // Defensive sanity check: parsing must at least find the ABI version
    // accessor.  If it doesn't, the parser regressed rather than the ABI.
    assert!(
        current.contains("stator_ffi_abi_version"),
        "header parser failed to find stator_ffi_abi_version — parser regressed"
    );

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

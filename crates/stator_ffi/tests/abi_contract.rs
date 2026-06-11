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
fn test_header_stats_constants_match_abi_and_use_c_friendly_docs() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, value) in [
        (
            "STATOR_DEOPT_TIER_COUNT",
            stator_jse_ffi::STATOR_DEOPT_TIER_COUNT as u32,
        ),
        (
            "STATOR_DEOPT_REASON_COUNT",
            stator_jse_ffi::STATOR_DEOPT_REASON_COUNT as u32,
        ),
        (
            "STATOR_IC_TIER_COUNT",
            stator_jse_ffi::STATOR_IC_TIER_COUNT as u32,
        ),
        (
            "STATOR_IC_OP_COUNT",
            stator_jse_ffi::STATOR_IC_OP_COUNT as u32,
        ),
        (
            "STATOR_IC_EVENT_COUNT",
            stator_jse_ffi::STATOR_IC_EVENT_COUNT as u32,
        ),
        (
            "STATOR_TIER_LATENCY_BUCKET_COUNT",
            stator_jse_ffi::STATOR_TIER_LATENCY_BUCKET_COUNT as u32,
        ),
        (
            "STATOR_OSR_TIER_COUNT",
            stator_jse_ffi::STATOR_OSR_TIER_COUNT as u32,
        ),
        (
            "STATOR_OSR_EXIT_REASON_COUNT",
            stator_jse_ffi::STATOR_OSR_EXIT_REASON_COUNT as u32,
        ),
        (
            "STATOR_JIT_UNWIND_TIER_COUNT",
            stator_jse_ffi::STATOR_JIT_UNWIND_TIER_COUNT as u32,
        ),
        (
            "STATOR_JIT_MEMORY_TIER_COUNT",
            stator_jse_ffi::STATOR_JIT_MEMORY_TIER_COUNT as u32,
        ),
        (
            "STATOR_JIT_MITIGATIONS_TIER_COUNT",
            stator_jse_ffi::STATOR_JIT_MITIGATIONS_TIER_COUNT as u32,
        ),
        (
            "STATOR_MITIGATION_STATUS_UNSUPPORTED_PLATFORM",
            stator_jse_ffi::STATOR_MITIGATION_STATUS_UNSUPPORTED_PLATFORM,
        ),
        (
            "STATOR_MITIGATION_STATUS_DISABLED",
            stator_jse_ffi::STATOR_MITIGATION_STATUS_DISABLED,
        ),
        (
            "STATOR_MITIGATION_STATUS_ENABLED",
            stator_jse_ffi::STATOR_MITIGATION_STATUS_ENABLED,
        ),
        (
            "STATOR_MITIGATION_STATUS_UNKNOWN",
            stator_jse_ffi::STATOR_MITIGATION_STATUS_UNKNOWN,
        ),
    ] {
        assert_eq!(
            parse_define_u32(&header, name),
            Some(value as u32),
            "generated stator.h `{name}` drifted from Rust ABI constant"
        );
    }
    for marker in [
        "[`StatorDeoptHistogramStats`]",
        "[`StatorDeoptReasonCounts`]",
        "[`StatorIcCountersStats`]",
        "[`StatorIcTierCounters`]",
        "[`StatorIcOpCounters`]",
        "[`StatorTierLatencyTier`]",
        "[`stator_jse::compiler::tier_latency::NUM_HISTOGRAM_BUCKETS`]",
        "stator_jse::compiler::tier_latency::",
        "[`StatorOsrCountersStats`]",
        "[`StatorOsrExitReasonCounts`]",
        "[`StatorJitUnwindStats`]",
        "stator_jse::jit_unwind::",
        "[`StatorJitMemoryStats`]",
        "[`StatorJitMitigationsStats`]",
        "stator_jse::jit_mitigations::",
        "MitigationStatus::",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_dom_handler_flags_match_abi_and_use_c_friendly_docs() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for define in [
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_NONE 0",
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_ALL_CAN_READ (1 << 0)",
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_NON_MASKING (1 << 1)",
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_ONLY_INTERCEPT_STRINGS (1 << 2)",
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_INTERCEPT_SYMBOLS (1 << 3)",
        "#define STATOR_DOM_NAMED_HANDLER_FLAG_HAS_NO_SIDE_EFFECT (1 << 4)",
        "#define STATOR_DOM_INDEXED_HANDLER_FLAG_NONE 0",
        "#define STATOR_DOM_INDEXED_HANDLER_FLAG_ALL_CAN_READ (1 << 0)",
        "#define STATOR_DOM_INDEXED_HANDLER_FLAG_NON_MASKING (1 << 1)",
        "#define STATOR_DOM_INDEXED_HANDLER_FLAG_HAS_NO_SIDE_EFFECT (1 << 4)",
    ] {
        assert!(
            header.contains(define),
            "generated stator.h is missing DOM handler flag define `{define}`"
        );
    }
    for signature in [
        "enum StatorStatus stator_dom_object_wrap_set_named_handler_flags(struct StatorDomObjectWrap *wrap,\n                                                                 uint32_t flags);",
        "enum StatorStatus stator_dom_object_wrap_get_named_handler_flags(const struct StatorDomObjectWrap *wrap,\n                                                                 uint32_t *out_flags);",
        "enum StatorStatus stator_dom_object_wrap_set_indexed_handler_flags(struct StatorDomObjectWrap *wrap,\n                                                                   uint32_t flags);",
        "enum StatorStatus stator_dom_object_wrap_get_indexed_handler_flags(const struct StatorDomObjectWrap *wrap,\n                                                                   uint32_t *out_flags);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h DOM handler flag signature drifted:\n{signature}"
        );
    }
    for marker in [
        "[`stator_dom_object_wrap_set_named_handler_flags`]",
        "[`STATOR_DOM_NAMED_HANDLER_FLAG_ALL`]",
        "[`stator_dom_object_wrap_install_named_handler`]",
        "[`STATOR_DOM_INDEXED_HANDLER_FLAG_ALL`]",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_dom_symbol_buffer_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "struct StatorDomSymbolBuffer *stator_dom_symbol_buffer_new(void);",
        "void stator_dom_symbol_buffer_destroy(struct StatorDomSymbolBuffer *buf);",
        "size_t stator_dom_symbol_buffer_len(const struct StatorDomSymbolBuffer *buf);",
        "enum StatorStatus stator_dom_symbol_buffer_push(struct StatorDomSymbolBuffer *buf,\n                                                uint64_t symbol_id,\n                                                const char *description_utf8,\n                                                size_t description_len);",
        "enum StatorStatus stator_dom_symbol_buffer_get(const struct StatorDomSymbolBuffer *buf,\n                                               size_t index,\n                                               struct StatorDomSymbolKey *out_key);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h DOM symbol-buffer signature drifted:\n{signature}"
        );
    }

    let key_start = header
        .find("POD descriptor for a symbol property key")
        .expect("generated stator.h should document StatorDomSymbolKey");
    let key_end = header
        .find("POD bundle of symbol-keyed named-property interceptors")
        .expect("generated stator.h should document symbol handler APIs");
    let symbol_key_docs = &header[key_start..key_end];
    for marker in [
        "[`SymbolKey`][stator_jse::dom::SymbolKey]",
        "stator_jse::dom::SymbolKey",
    ] {
        assert!(
            !symbol_key_docs.contains(marker),
            "generated stator.h DOM symbol-key docs should not expose Rust-specific marker `{marker}`"
        );
    }

    let push_start = header
        .find("Append a symbol identity to a `StatorDomSymbolBuffer`")
        .expect("generated stator.h should document symbol-buffer push");
    let push_end = header
        .find("Install an aggregated set of symbol-keyed named-property interceptors")
        .expect("generated stator.h should document symbol handler install APIs");
    let push_docs = &header[push_start..push_end];
    let buffer_start = header
        .find("Allocate a fresh, empty `StatorDomSymbolBuffer`")
        .expect("generated stator.h should document StatorDomSymbolBuffer allocation");
    let buffer_end = header
        .find("Install additive named/symbol definer")
        .expect("generated stator.h should document symbol definer APIs");
    let buffer_docs = &header[buffer_start..buffer_end];
    for marker in ["StatorStatus::StatorStatus"] {
        assert!(
            !push_docs.contains(marker) && !buffer_docs.contains(marker),
            "generated stator.h DOM symbol-buffer docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_dom_index_and_name_buffer_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "struct StatorDomIndexBuffer *stator_dom_index_buffer_new(void);",
        "void stator_dom_index_buffer_destroy(struct StatorDomIndexBuffer *buf);",
        "size_t stator_dom_index_buffer_len(const struct StatorDomIndexBuffer *buf);",
        "enum StatorStatus stator_dom_index_buffer_get(const struct StatorDomIndexBuffer *buf,\n                                              size_t index,\n                                              uint32_t *out_index);",
        "enum StatorStatus stator_dom_index_buffer_push(struct StatorDomIndexBuffer *buf, uint32_t index);",
        "enum StatorStatus stator_dom_name_buffer_push(struct StatorDomNameBuffer *buf,\n                                              const char *name_utf8,\n                                              size_t name_len);",
        "enum StatorStatus stator_dom_object_wrap_invoke_indexed_deleter(struct StatorDomObjectWrap *wrap,\n                                                                uint32_t index,\n                                                                bool *out_deleted);",
        "enum StatorStatus stator_dom_object_wrap_invoke_indexed_enumerate_into(struct StatorDomObjectWrap *wrap,\n                                                                       struct StatorDomIndexBuffer *buf);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h DOM index/name buffer signature drifted:\n{signature}"
        );
    }

    let typedef_start = header
        .find("Opaque index buffer passed to a `StatorDomIndexedEnumeratorCb`")
        .expect("generated stator.h should document StatorDomIndexBuffer");
    let typedef_end = header
        .find("An opaque handle to a DOM object wrapper.")
        .expect("generated stator.h should document DOM wrappers after buffers");
    let typedef_docs = &header[typedef_start..typedef_end];
    let api_start = header
        .find("Allocate a fresh, empty `StatorDomIndexBuffer`")
        .expect("generated stator.h should document index-buffer allocation");
    let api_end = header
        .find("Install an aggregated set of named-property interceptors on `wrap`.")
        .expect("generated stator.h should document named handler APIs after name buffers");
    let api_docs = &header[api_start..api_end];
    let invoke_start = header
        .find("Invoke the wrapper's indexed-property **deleter** path.")
        .expect("generated stator.h should document indexed deleter invocation");
    let invoke_end = header
        .find("Collect the indices reported by the wrapper's indexed-property")
        .expect("generated stator.h should document indexed enumerate invocation");
    let deleter_docs = &header[invoke_start..invoke_end];
    let enumerate_start = invoke_end;
    let invoke_end = header
        .find("Allocate a fresh, empty `StatorDomSymbolBuffer`")
        .expect("generated stator.h should document symbol buffers after indexed invocation");
    let invoke_docs = &header[enumerate_start..invoke_end];

    for marker in [
        "[`StatorDomIndexBuffer`]",
        "[`StatorDomNameBuffer`]",
        "[`stator_dom_index_buffer_push`]",
        "[`stator_dom_index_buffer_new`]",
        "[`stator_dom_index_buffer_destroy`]",
        "[`stator_dom_index_buffer_get`]",
        "[`stator_dom_name_buffer_push`]",
        "StatorStatus::StatorStatus",
        "*mut u32",
        "*mut bool",
        "stator_jse::dom::IndexedDeleterResult",
        "[`StatorDomObjectWrap`]",
    ] {
        assert!(
            !typedef_docs.contains(marker)
                && !api_docs.contains(marker)
                && !deleter_docs.contains(marker)
                && !invoke_docs.contains(marker),
            "generated stator.h DOM index/name buffer docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_dom_callable_callback_typedefs_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "typedef enum StatorStatus (*StatorDomCallAsFunctionCb)(const struct StatorValue *receiver,\n                                                       const struct StatorValue *const *args,\n                                                       size_t arg_count,\n                                                       void *data,\n                                                       struct StatorValue **out);",
        "typedef enum StatorStatus (*StatorDomConstructCb)(const struct StatorValue *new_target,\n                                                  const struct StatorValue *const *args,\n                                                  size_t arg_count,\n                                                  void *data,\n                                                  struct StatorValue **out);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h DOM callable callback typedef drifted:\n{signature}"
        );
    }

    let start = header
        .find("DOM wrapper call-as-function callback.")
        .expect("generated stator.h should document DOM callable callback");
    let end = header
        .find("V2 named-property **getter** callback.")
        .expect(
            "generated stator.h should document named getter callback after callable callbacks",
        );
    let docs = &header[start..end];
    for marker in [
        "[`StatorValue`]",
        "StatorStatus::StatorStatus",
        "[`StatorDomCallAsFunctionCb`]",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h DOM callable callback docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_dom_v2_callback_docs_use_c_friendly_names() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    let start = header
        .find("V2 named-property **getter** callback.")
        .expect("generated stator.h should document DOM V2 callbacks");
    let end = start
        + header[start..]
            .find("Return the packed Stator FFI ABI version compiled into this library.")
            .expect("generated stator.h should document ABI version APIs after callback typedefs");
    let docs = &header[start..end];
    for marker in [
        "StatorStatus::",
        "[`StatorValue`]",
        "[`stator_isolate_throw_exception`]",
        "[`stator_dom_name_buffer_push`]",
        "[`stator_dom_index_buffer_push`]",
        "[`StatorDomNamedGetterCbV2`]",
        "[`StatorDomNamedSetterCbV2`]",
        "[`StatorDomNamedQueryCb`]",
        "stator_jse::dom::IndexedDeleterResult",
        "*mut ",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h DOM V2 callback docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_status_discriminants_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorStatusOk", 0),
        ("StatorStatusFalse", 1),
        ("StatorStatusException", 2),
        ("StatorStatusInvalidArg", 3),
        ("StatorStatusUnsupported", 4),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable status discriminant `{marker}`"
        );
    }
    for marker in [
        "[`StatorStatusOk`][Self::StatorStatusOk]",
        "[`StatorStatusFalse`][Self::StatorStatusFalse]",
        "[`StatorStatusException`][Self::StatorStatusException]",
        "[`StatorStatusInvalidArg`][Self::StatorStatusInvalidArg]",
        "[`StatorStatusUnsupported`][Self::StatorStatusUnsupported]",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_object_property_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "enum StatorStatus stator_object_get_property(const struct StatorObject *obj,\n                                             const char *key,\n                                             size_t key_len,\n                                             struct StatorValue **out_val);",
        "enum StatorStatus stator_object_set_property(struct StatorObject *obj,\n                                             const char *key,\n                                             size_t key_len,\n                                             const struct StatorValue *val);",
        "enum StatorStatus stator_object_has_property(const struct StatorObject *obj,\n                                             const char *key,\n                                             size_t key_len,\n                                             bool *out);",
        "enum StatorStatus stator_object_delete_property(struct StatorObject *obj,\n                                                const char *key,\n                                                size_t key_len,\n                                                bool *out);",
        "enum StatorStatus stator_value_call(struct StatorContext *ctx,\n                                    const struct StatorValue *callable,\n                                    const struct StatorValue *recv,\n                                    int32_t argc,\n                                    const struct StatorValue *const *args,\n                                    struct StatorValue **out_val);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h object property/value call signature drifted:\n{signature}"
        );
    }

    let start = header
        .find("Read the property named `(key, key_len)` from `obj`")
        .expect("generated stator.h should document object property reads");
    let has_start = header
        .find("Test whether `obj` has a property named `(key, key_len)`")
        .expect("generated stator.h should document object property lookup");
    let get_set_docs = &header[start..has_start];
    for marker in [
        "StatorStatus::StatorStatus",
        "StatorMessageKind::",
        "[`StatorObject`]",
        "[`StatorValue`]",
        "[`StatorMessage`]",
        "[`stator_value_destroy`]",
        "*mut StatorValue",
    ] {
        assert!(
            !get_set_docs.contains(marker),
            "generated stator.h object get/set docs should not expose Rust-specific marker `{marker}`"
        );
    }

    let call_start = header
        .find("Invoke a callable `StatorValue` with `argc` arguments")
        .expect("generated stator.h should document value calls after object property APIs");
    let property_docs = &header[has_start..call_start];
    for marker in ["StatorStatus::StatorStatus", "[`StatorObject`]"] {
        assert!(
            !property_docs.contains(marker),
            "generated stator.h object property docs should not expose Rust-specific marker `{marker}`"
        );
    }

    let call_end = call_start
        + header[call_start..]
            .find("Register a native function named `name` on `ctx`.")
            .unwrap_or_else(|| {
                panic!("generated stator.h should document native registration after value call")
            });
    let call_docs = &header[call_start..call_end];
    for marker in [
        "[`StatorValue`]",
        "[`NativeFn`]",
        "StatorValueInner::",
        "StatorStatus::StatorStatus",
        "*const StatorValue",
        "*mut StatorValue",
    ] {
        assert!(
            !call_docs.contains(marker),
            "generated stator.h value call docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_value_numeric_getter_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "enum StatorStatus stator_value_get_boolean(const struct StatorValue *val, bool *out);",
        "enum StatorStatus stator_value_get_number(const struct StatorValue *val, double *out);",
        "enum StatorStatus stator_value_get_int32(const struct StatorValue *val, int32_t *out);",
        "enum StatorStatus stator_value_get_uint32(const struct StatorValue *val, uint32_t *out);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h value getter signature drifted:\n{signature}"
        );
    }

    let start = header
        .find("Read the boolean value of `val` into `*out`.")
        .expect("generated stator.h should document value boolean getter");
    let end = header
        .find("Read the UTF-8 byte length of the string stored in `val` into `*out`.")
        .expect("generated stator.h should document string getter after numeric getters");
    let docs = &header[start..end];
    for marker in [
        "StatorStatus::StatorStatus",
        "[`StatorValue`]",
        "[`stator_value_to_int32`]",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h value getter docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_value_string_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "enum StatorStatus stator_value_get_string_utf8_length(const struct StatorValue *val, size_t *out);",
        "enum StatorStatus stator_value_write_string_utf8(const struct StatorValue *val,\n                                                 char *buf,\n                                                 size_t buf_size,\n                                                 size_t *written);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h value string signature drifted:\n{signature}"
        );
    }

    let start = header
        .find("Read the UTF-8 byte length of the string stored in `val` into `*out`.")
        .expect("generated stator.h should document string length getter");
    let end = header
        .find("Wrap a `StatorObject` handle as a fresh `StatorValue` handle")
        .expect("generated stator.h should document object/value bridge after string APIs");
    let docs = &header[start..end];
    for marker in [
        "StatorStatus::StatorStatus",
        "[`StatorValue`]",
        "[`stator_value_write_string_utf8`]",
        "[`stator_value_get_string_utf8_length`]",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h value string docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_object_value_bridge_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "struct StatorValue *stator_object_as_value(const struct StatorObject *obj);",
        "struct StatorObject *stator_value_as_object(const struct StatorValue *val);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h object/value bridge signature drifted:\n{signature}"
        );
    }

    let start = header
        .find("Wrap a `StatorObject` handle as a fresh `StatorValue` handle")
        .expect("generated stator.h should document object/value bridge");
    let end = header
        .find("Read the property named `(key, key_len)` from `obj`")
        .expect("generated stator.h should document object property reads after bridge APIs");
    let docs = &header[start..end];
    for marker in [
        "[`StatorObject`]",
        "[`StatorValue`]",
        "[`stator_object_as_value`]",
        "[`stator_value_as_object`]",
        "[`stator_value_destroy`]",
        "[`stator_object_destroy`]",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h object/value bridge docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_promise_state_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorPromiseStatePending", 0),
        ("StatorPromiseStateFulfilled", 1),
        ("StatorPromiseStateRejected", 2),
        ("StatorPromiseStateInvalid", 3),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable Promise state discriminant `{marker}`"
        );
    }
    for signature in [
        "enum StatorPromiseState stator_promise_state(const struct StatorValue *promise);",
        "struct StatorValue *stator_promise_result(const struct StatorValue *promise);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h Promise signature drifted:\n{signature}"
        );
    }
    for marker in ["[`StatorPromiseState::StatorPromiseStateInvalid`]"] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_persistent_handle_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "struct StatorPersistent *stator_persistent_new(struct StatorContext *ctx,\n                                               struct StatorValue *value);",
        "void stator_persistent_reset(struct StatorPersistent *persistent);",
        "bool stator_persistent_is_empty(const struct StatorPersistent *persistent);",
        "struct StatorValue *stator_persistent_get(const struct StatorPersistent *persistent);",
        "void stator_persistent_dispose(struct StatorPersistent *persistent);",
        "struct StatorWeak *stator_persistent_make_weak(struct StatorPersistent *persistent,\n                                               void *parameter,\n                                               void *internal_field0,\n                                               void *internal_field1,\n                                               StatorWeakCallback cb,\n                                               enum StatorWeakParameterKind parameter_kind);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h persistent-handle signature drifted:\n{signature}"
        );
    }
    for marker in [
        "[`StatorPersistentSlot`]",
        "[`stator_jse::gc::handle::PersistentRoots`]",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_weak_parameter_kind_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [("Opaque", 0), ("InternalFields", 1)] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable weak parameter-kind discriminant `{marker}`"
        );
    }
    for signature in [
        "struct StatorWeak *stator_weak_new(struct StatorContext *ctx,\n                                   struct StatorValue *value,\n                                   void *parameter,\n                                   void *internal_field0,\n                                   void *internal_field1,\n                                   StatorWeakCallback cb,\n                                   enum StatorWeakParameterKind parameter_kind);",
        "struct StatorWeak *stator_persistent_make_weak(struct StatorPersistent *persistent,\n                                               void *parameter,\n                                               void *internal_field0,\n                                               void *internal_field1,\n                                               StatorWeakCallback cb,\n                                               enum StatorWeakParameterKind parameter_kind);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h weak-handle signature drifted:\n{signature}"
        );
    }
    for marker in [
        "[`StatorWeakCallbackInfo`]",
        "[`StatorWeakCallbackInfo::internal_fields`]",
        "[`StatorWeakParameterKind::InternalFields`]",
        "[`StatorWeakParameterKind::Opaque`]",
        "StatorWeakParameterKind::InternalFields",
        "StatorWeakParameterKind::Opaque",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_wasm_value_kind_discriminants_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorWasmValueKindI32", 0),
        ("StatorWasmValueKindI64", 1),
        ("StatorWasmValueKindF32", 2),
        ("StatorWasmValueKindF64", 3),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable Wasm value-kind discriminant `{marker}`"
        );
    }
    for marker in ["[`StatorWasmValue`]", "[`StatorWasmValue::kind`]"] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_context_cdp_debug_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "struct StatorContext *stator_context_new(struct StatorIsolate *isolate);",
        "void stator_context_destroy(struct StatorContext *ctx);",
        "struct StatorCdpServer *stator_cdp_server_create(uint16_t port);",
        "void stator_cdp_server_destroy(struct StatorCdpServer *server);",
        "struct StatorDebugSession *stator_debug_session_create(const struct StatorScript *script,\n                                                       struct StatorContext *ctx);",
        "bool stator_debug_session_run(struct StatorDebugSession *session);",
        "int32_t stator_debug_session_get_global_string(const struct StatorDebugSession *session,\n                                               const char *name,\n                                               char *buf,\n                                               size_t buf_len);",
        "bool stator_debug_session_resume(struct StatorDebugSession *session);",
        "struct StatorValue *stator_debug_session_result(const struct StatorDebugSession *session);",
        "void stator_debug_session_destroy(struct StatorDebugSession *session);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h context/CDP/debug signature drifted:\n{signature}"
        );
    }

    for typedef in [
        "typedef struct StatorContext StatorContext;",
        "typedef struct StatorCdpServer StatorCdpServer;",
        "typedef struct StatorDebugSession StatorDebugSession;",
    ] {
        let end = header
            .find(typedef)
            .unwrap_or_else(|| panic!("generated stator.h missing `{typedef}`"));
        let start = end.saturating_sub(700);
        let doc_window = &header[start..end];
        assert!(
            !doc_window.contains("[`"),
            "generated stator.h opaque-handle docs should not expose Rust intra-doc links near `{typedef}`:\n{doc_window}"
        );
    }
}

#[test]
fn test_header_isolate_safety_docs_use_c_friendly_names() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    assert!(
        !header.contains("[`StatorIsolate`]"),
        "generated stator.h should not expose Rust intra-doc links to StatorIsolate"
    );
}

#[test]
fn test_header_module_host_callback_signatures_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "bool stator_context_set_module_url_resolver(struct StatorContext *ctx,\n                                            enum StatorResolveStatus (*callback)(struct StatorContext *ctx,\n                                                                                 void *user_data,\n                                                                                 const struct StatorModule *referrer,\n                                                                                 const struct StatorModuleOrigin *origin,\n                                                                                 const char *specifier,\n                                                                                 size_t specifier_len,\n                                                                                 const struct StatorImportAttribute *attributes,\n                                                                                 size_t attributes_len,\n                                                                                 struct StatorString **out_resolved_url,\n                                                                                 struct StatorString **out_error),\n                                            void *user_data,\n                                            void (*free_user_data)(void *user_data));",
        "bool stator_context_set_import_meta_populator(struct StatorContext *ctx,\n                                              enum StatorResolveStatus (*callback)(struct StatorContext *ctx,\n                                                                                   void *user_data,\n                                                                                   const struct StatorModule *referrer,\n                                                                                   const struct StatorModuleOrigin *origin,\n                                                                                   struct StatorImportMetaProperties *out_meta,\n                                                                                   struct StatorString **out_error),\n                                              void *user_data,\n                                              void (*free_user_data)(void *user_data));",
        "bool stator_context_set_dynamic_import_resolver(struct StatorContext *ctx,\n                                                struct Option_StatorDynamicImportCallback callback,\n                                                void *user_data,\n                                                void (*free_user_data)(void *user_data));",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h module host-callback signature drifted:\n{signature}"
        );
    }

    let start = header
        .find("Register, replace, or clear the module URL resolver callback for `ctx`.")
        .expect("generated stator.h should document module URL resolver registration");
    let end = header
        .find("Create a new number value.")
        .expect("generated stator.h should document value constructors after module callbacks");
    let docs = &header[start..end];
    for marker in [
        "[`StatorResolveStatus`]",
        "[`StatorDynamicImportRequest`]",
        "StatorResolveStatus::",
        "[`stator_context_set_module_resolver`]",
        "[`stator_dynamic_import_request_resolve_module`]",
        "[`stator_dynamic_import_request_reject`]",
    ] {
        assert!(
            !docs.contains(marker),
            "generated stator.h module host-callback docs should not expose Rust-specific marker `{marker}`"
        );
    }
}

#[test]
fn test_header_module_cache_status_discriminants_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorModuleCacheStatusProducedMetadata", 0),
        ("StatorModuleCacheStatusAcceptedValidatedRecompiled", 1),
        ("StatorModuleCacheStatusInvalidArgument", 2),
        ("StatorModuleCacheStatusCompileError", 3),
        ("StatorModuleCacheStatusRejected", 4),
        ("StatorModuleCacheStatusUnsupported", 5),
        ("StatorModuleCacheStatusAcceptedBytecodeRestored", 6),
        ("StatorModuleCacheStatusCorruptPayload", 7),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable module cache status discriminant `{marker}`"
        );
    }
    let signature = "const char *stator_module_cache_status_name_u32(uint32_t status);";
    assert!(
        header.contains(signature),
        "generated stator.h module cache status name signature drifted:\n{signature}"
    );
}

#[test]
fn test_header_module_status_discriminants_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorModuleStatusErrored", -1),
        ("StatorModuleStatusUnlinked", 0),
        ("StatorModuleStatusLinking", 1),
        ("StatorModuleStatusLinked", 2),
        ("StatorModuleStatusEvaluating", 3),
        ("StatorModuleStatusEvaluated", 4),
        ("StatorModuleStatusPendingAsyncEvaluation", 5),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable module status discriminant `{marker}`"
        );
    }
    for marker in [
        "[`stator_module_resume_evaluation`]",
        "[`stator_module_pending_evaluation_fulfill`]",
        "[`stator_module_pending_evaluation_reject`]",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_snapshot_build_hash_discriminants_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorSnapshotBuildHashBuildId", 0),
        ("StatorSnapshotBuildHashBuildFeatures", 1),
        ("StatorSnapshotBuildHashJitTiering", 2),
        ("StatorSnapshotBuildHashCpuFeatures", 3),
        ("StatorSnapshotBuildHashEdgeRelease", 4),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable snapshot build hash discriminant `{marker}`"
        );
    }
    assert!(
        !header.contains("[`StatorSnapshotBuildBinding`]"),
        "generated stator.h should not expose Rust intra-doc links for snapshot build binding"
    );
}

#[test]
fn test_header_script_cache_discriminants_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorScriptCacheStatusProducedMetadata", 0),
        ("StatorScriptCacheStatusAcceptedBytecodeRestored", 1),
        ("StatorScriptCacheStatusAcceptedValidatedRecompiled", 2),
        ("StatorScriptCacheStatusInvalidArgument", 3),
        ("StatorScriptCacheStatusCompileError", 4),
        ("StatorScriptCacheStatusRejected", 5),
        ("StatorScriptCacheStatusUnsupported", 6),
        ("StatorScriptCacheStatusCorruptPayload", 7),
        ("StatorScriptCacheDiagnosticNone", 0),
        ("StatorScriptCacheDiagnosticMagicMismatch", 1),
        ("StatorScriptCacheDiagnosticVersionMismatch", 2),
        ("StatorScriptCacheDiagnosticSourceMismatch", 3),
        ("StatorScriptCacheDiagnosticOptionsMismatch", 4),
        ("StatorScriptCacheDiagnosticCorruptPayload", 5),
        ("StatorScriptCacheDiagnosticCompileError", 6),
        ("StatorScriptCacheDiagnosticUnsupported", 7),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable script cache discriminant `{marker}`"
        );
    }
    for signature in [
        "const char *stator_script_cache_status_name_u32(uint32_t status);",
        "const char *stator_script_cache_diagnostic_name_u32(uint32_t diagnostic);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h script cache telemetry name signature drifted:\n{signature}"
        );
    }
}

#[test]
fn test_header_jit_tier_discriminants_and_signatures_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [("Baseline", 1), ("Maglev", 2), ("Turbofan", 3)] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable JIT tier discriminant `{marker}`"
        );
    }
    for (name, discriminant) in [
        ("AlreadyReady", 0),
        ("Compiled", 1),
        ("Pending", 2),
        ("UnsupportedTier", 3),
        ("JitDisabled", 4),
        ("DeoptBlocked", 5),
        ("GraphBuildFailed", 6),
        ("DegenerateGraph", 7),
        ("CompileFailed", 8),
        ("ExecutableAllocationFailed", 9),
        ("TimedOut", 10),
        ("InvalidScript", 11),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable tier-request status discriminant `{marker}`"
        );
    }
    for signature in [
        "bool stator_script_force_tier(struct StatorScript *script,\n                              enum StatorJitTier tier,\n                              struct StatorTierRequestResult *result);",
        "bool stator_script_observe_tier(const struct StatorScript *script,\n                                enum StatorJitTier tier,\n                                struct StatorTierRequestResult *result);",
        "bool stator_script_wait_for_tier(const struct StatorScript *script,\n                                 enum StatorJitTier tier,\n                                 uint64_t timeout_ms,\n                                 struct StatorTierRequestResult *result);",
    ] {
        assert!(
            header.contains(signature),
            "generated stator.h deterministic tier signature drifted:\n{signature}"
        );
    }
}

#[test]
fn test_header_message_kind_discriminants_and_docs_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for (name, discriminant) in [
        ("StatorMessageKindUnknown", 0),
        ("StatorMessageKindSyntax", 1),
        ("StatorMessageKindType", 2),
        ("StatorMessageKindRange", 3),
        ("StatorMessageKindReference", 4),
        ("StatorMessageKindURI", 5),
        ("StatorMessageKindWasm", 6),
        ("StatorMessageKindInternal", 7),
        ("StatorMessageKindTermination", 8),
        ("StatorMessageKindJsException", 9),
        ("StatorMessageKindOutOfMemory", 10),
        ("StatorMessageKindSandboxViolation", 11),
        ("StatorMessageKindUnsupportedModuleType", 12),
    ] {
        let marker = format!("{name} = {discriminant}");
        assert!(
            header.contains(&marker),
            "generated stator.h is missing stable message-kind discriminant `{marker}`"
        );
    }
    for marker in [
        "[`StatorMessageKind::StatorMessageKindUnknown`]",
        "[`StatorMessageKind::StatorMessageKindTermination`]",
        "stator_jse::error::StatorError::",
    ] {
        assert!(
            !header.contains(marker),
            "generated stator.h should not expose Rust intra-doc link `{marker}`"
        );
    }
}

#[test]
fn test_header_native_code_cache_function_signatures_match_abi() {
    let header = fs::read_to_string(header_path()).expect("generated stator.h must exist");
    for signature in [
        "size_t stator_native_code_cache_header_size(void);",
        "const char *stator_native_code_cache_diagnostic_name(enum StatorNativeCodeCacheDiagnostic diagnostic);",
        "const char *stator_native_code_cache_diagnostic_name_u32(uint32_t diagnostic);",
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

//! Fail-closed native-tier code-cache artifact header validation.
//!
//! Stator does not deserialize or execute persisted native code today. This
//! module defines the fixed native artifact header that Edge may package for
//! baseline, Maglev, and Turbofan tiers, then exposes conservative FFI helpers
//! that classify headers, validate compatibility fields, and return stable
//! diagnostics without ever reporting a runtime cache hit.

use std::ffi::c_char;

use sha2::{Digest, Sha256};

/// Magic bytes at the start of every native-tier artifact header.
pub const STATOR_NATIVE_CODE_CACHE_HEADER_MAGIC: [u8; 8] = *b"STNCACH1";
/// Current native-tier artifact header format version.
pub const STATOR_NATIVE_CODE_CACHE_HEADER_VERSION: u32 = 1;
/// SHA-256 digest length used by all fixed header digest fields.
pub const STATOR_NATIVE_CODE_CACHE_DIGEST_LEN: usize = 32;
/// Fixed byte length of a native-tier artifact header.
pub const STATOR_NATIVE_CODE_CACHE_HEADER_SIZE: usize = 256;

const MAGIC_LEN: usize = 8;
const U32_LEN: usize = 4;
const U64_LEN: usize = 8;
const DIGEST_LEN: usize = STATOR_NATIVE_CODE_CACHE_DIGEST_LEN;

/// Native code-cache tier encoded in a native artifact header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatorNativeCodeCacheTier {
    /// Unknown or unrecognized tier value.
    StatorNativeCodeCacheTierUnknown = 0,
    /// Baseline native-code tier.
    StatorNativeCodeCacheTierBaseline = 1,
    /// Maglev optimizing tier.
    StatorNativeCodeCacheTierMaglev = 2,
    /// Turbofan optimizing tier.
    StatorNativeCodeCacheTierTurbofan = 3,
}

impl StatorNativeCodeCacheTier {
    fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::StatorNativeCodeCacheTierBaseline,
            2 => Self::StatorNativeCodeCacheTierMaglev,
            3 => Self::StatorNativeCodeCacheTierTurbofan,
            _ => Self::StatorNativeCodeCacheTierUnknown,
        }
    }

    fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Stable native artifact compatibility diagnostic.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatorNativeCodeCacheDiagnostic {
    /// Header classification accepted the fixed header shape.
    StatorNativeCodeCacheDiagnosticAccepted = 0,
    /// A required input pointer/length pair or compatibility pointer was invalid.
    StatorNativeCodeCacheDiagnosticInvalidArgument = 1,
    /// Header structure, payload length, or payload digest was invalid.
    StatorNativeCodeCacheDiagnosticCorruptPayload = 2,
    /// The target tier or artifact type was not accepted by this consumer.
    StatorNativeCodeCacheDiagnosticRejectedArtifactType = 3,
    /// Engine crate or FFI ABI identity did not match.
    StatorNativeCodeCacheDiagnosticRejectedEngineVersion = 4,
    /// Native artifact format version did not match.
    ///
    /// Unsupported fixed-header versions are treated as corrupt payloads because
    /// the decoder only accepts `STATOR_NATIVE_CODE_CACHE_HEADER_VERSION`.
    StatorNativeCodeCacheDiagnosticRejectedFormatVersion = 5,
    /// Source key digest did not match the requested source artifact key.
    StatorNativeCodeCacheDiagnosticRejectedSourceIdentity = 6,
    /// Target triple or CPU feature policy did not match.
    StatorNativeCodeCacheDiagnosticRejectedPlatform = 7,
    /// Compiler version/build identity did not match.
    StatorNativeCodeCacheDiagnosticRejectedBuildFeatures = 8,
    /// JIT flags, sandbox mode, or tiering policy did not match.
    StatorNativeCodeCacheDiagnosticRejectedCompilerFlags = 9,
    /// Header is compatible, but this build must not load native bytes.
    StatorNativeCodeCacheDiagnosticUnsupportedNativeCode = 10,
}

/// Expected native-tier artifact compatibility fields supplied by the embedder.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatorNativeCodeCacheCompatibility {
    /// Required target tier.
    pub tier: StatorNativeCodeCacheTier,
    /// Required baseline/JIT native artifact format version.
    pub native_format_version: u32,
    /// Required packed `STATOR_FFI_ABI_VERSION` value.
    pub ffi_abi_version: u32,
    /// SHA-256 digest of the canonical Stator engine version string.
    pub engine_version_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical compiler/backend version string.
    pub compiler_version_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical target triple string.
    pub target_triple_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the sorted canonical CPU feature set string.
    pub cpu_feature_set_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of canonical tiering/JIT/sandbox flags.
    pub jit_flags_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical source code-cache key record.
    pub source_key_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
}

/// Decoded native-tier artifact header information.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatorNativeCodeCacheHeaderInfo {
    /// Header format version.
    pub header_version: u32,
    /// Target tier encoded by the artifact.
    pub tier: StatorNativeCodeCacheTier,
    /// Baseline/JIT native artifact format version.
    pub native_format_version: u32,
    /// Packed `STATOR_FFI_ABI_VERSION` value encoded by the producer.
    pub ffi_abi_version: u32,
    /// SHA-256 digest of the canonical Stator engine version string.
    pub engine_version_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical compiler/backend version string.
    pub compiler_version_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical target triple string.
    pub target_triple_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the sorted canonical CPU feature set string.
    pub cpu_feature_set_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of canonical tiering/JIT/sandbox flags.
    pub jit_flags_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the canonical source code-cache key record.
    pub source_key_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// SHA-256 digest of the native payload bytes following this header.
    pub payload_digest: [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN],
    /// Exact byte length of the native payload following this header.
    pub payload_length_bytes: u64,
}

/// Return the fixed byte length of native-tier artifact headers.
#[unsafe(no_mangle)]
pub extern "C" fn stator_native_code_cache_header_size() -> usize {
    STATOR_NATIVE_CODE_CACHE_HEADER_SIZE
}

/// Return a stable low-cardinality telemetry code string for a diagnostic.
///
/// The returned pointer refers to a process-static, NUL-terminated string owned
/// by Stator. Embedders must not free or mutate it; it remains valid for the
/// lifetime of the process.
#[unsafe(no_mangle)]
pub extern "C" fn stator_native_code_cache_diagnostic_name(
    diagnostic: StatorNativeCodeCacheDiagnostic,
) -> *const c_char {
    diagnostic_name_bytes(diagnostic).as_ptr().cast()
}

/// Decode a native-tier artifact header without checking runtime compatibility.
///
/// This helper accepts only the fixed header shape. It never validates or loads
/// payload bytes and never reports that native code can be executed.
///
/// # Safety
///
/// When `bytes` is non-null and `len > 0`, it must point to `len` readable
/// bytes for the duration of the call. Null `bytes` or zero `len` returns
/// [`StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument`]
/// without dereferencing `bytes`. When `out_info` is non-null, it must be valid
/// for one write; null `out_info` is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_native_code_cache_classify_header(
    bytes: *const u8,
    len: usize,
    out_info: *mut StatorNativeCodeCacheHeaderInfo,
) -> StatorNativeCodeCacheDiagnostic {
    let Some(header_bytes) = non_null_slice(bytes, len) else {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument;
    };
    match parse_header(header_bytes) {
        Some(info) => {
            write_info(out_info, info);
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticAccepted
        }
        None => StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload,
    }
}

/// Validate a full native-tier artifact against expected compatibility fields.
///
/// On a fully compatible header and payload this still returns
/// `UnsupportedNativeCode`: current Stator builds do not deserialize or execute
/// native bytes from code-cache storage. Consumers must treat every non-mismatch
/// result as a cache miss/recompile path, not as a native cache hit.
///
/// # Safety
///
/// When `bytes` is non-null and `len > 0`, it must point to `len` readable
/// bytes for the duration of the call. `expected` must either be null or point
/// to a valid compatibility struct. Null `bytes`, zero `len`, or null
/// `expected` returns
/// [`StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument`]
/// without dereferencing invalid inputs. When `out_info` is non-null, it must be
/// valid for one write; null `out_info` is accepted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stator_native_code_cache_validate_header(
    bytes: *const u8,
    len: usize,
    expected: *const StatorNativeCodeCacheCompatibility,
    out_info: *mut StatorNativeCodeCacheHeaderInfo,
) -> StatorNativeCodeCacheDiagnostic {
    let Some(artifact) = non_null_slice(bytes, len) else {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument;
    };
    if expected.is_null() {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument;
    }
    // SAFETY: `expected` was checked non-null and is borrowed for this call only.
    let expected = unsafe { &*expected };
    let Some(info) = parse_header(artifact) else {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload;
    };
    write_info(out_info, info);

    if info.tier == StatorNativeCodeCacheTier::StatorNativeCodeCacheTierUnknown
        || info.tier != expected.tier
    {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedArtifactType;
    }
    if info.header_version != STATOR_NATIVE_CODE_CACHE_HEADER_VERSION
        || info.native_format_version != expected.native_format_version
    {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedFormatVersion;
    }
    if info.ffi_abi_version != expected.ffi_abi_version
        || info.engine_version_digest != expected.engine_version_digest
    {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedEngineVersion;
    }
    if info.compiler_version_digest != expected.compiler_version_digest {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedBuildFeatures;
    }
    if info.target_triple_digest != expected.target_triple_digest
        || info.cpu_feature_set_digest != expected.cpu_feature_set_digest
    {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedPlatform;
    }
    if info.jit_flags_digest != expected.jit_flags_digest {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedCompilerFlags;
    }
    if info.source_key_digest != expected.source_key_digest {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedSourceIdentity;
    }

    let payload_len = usize::try_from(info.payload_length_bytes).ok();
    let Some(payload_len) = payload_len else {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload;
    };
    let Some(expected_total) = STATOR_NATIVE_CODE_CACHE_HEADER_SIZE.checked_add(payload_len) else {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload;
    };
    if artifact.len() != expected_total {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload;
    }
    let payload = &artifact[STATOR_NATIVE_CODE_CACHE_HEADER_SIZE..];
    if sha256_digest(payload) != info.payload_digest {
        return StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload;
    }

    StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticUnsupportedNativeCode
}

/// Build a fixed native-tier artifact header for tests and release tooling.
pub fn build_native_code_cache_header(
    expected: &StatorNativeCodeCacheCompatibility,
    payload: &[u8],
) -> [u8; STATOR_NATIVE_CODE_CACHE_HEADER_SIZE] {
    let mut out = [0u8; STATOR_NATIVE_CODE_CACHE_HEADER_SIZE];
    out[..MAGIC_LEN].copy_from_slice(&STATOR_NATIVE_CODE_CACHE_HEADER_MAGIC);
    let mut offset = MAGIC_LEN;
    write_u32(
        &mut out,
        &mut offset,
        STATOR_NATIVE_CODE_CACHE_HEADER_VERSION,
    );
    write_u32(&mut out, &mut offset, expected.tier.as_u32());
    write_u32(&mut out, &mut offset, expected.native_format_version);
    write_u32(&mut out, &mut offset, expected.ffi_abi_version);
    write_digest(&mut out, &mut offset, &expected.engine_version_digest);
    write_digest(&mut out, &mut offset, &expected.compiler_version_digest);
    write_digest(&mut out, &mut offset, &expected.target_triple_digest);
    write_digest(&mut out, &mut offset, &expected.cpu_feature_set_digest);
    write_digest(&mut out, &mut offset, &expected.jit_flags_digest);
    write_digest(&mut out, &mut offset, &expected.source_key_digest);
    write_digest(&mut out, &mut offset, &sha256_digest(payload));
    write_u64(&mut out, &mut offset, payload.len() as u64);
    debug_assert_eq!(offset, STATOR_NATIVE_CODE_CACHE_HEADER_SIZE);
    out
}

/// Return the SHA-256 digest for native compatibility fields.
pub fn native_sha256_digest(bytes: &[u8]) -> [u8; STATOR_NATIVE_CODE_CACHE_DIGEST_LEN] {
    sha256_digest(bytes)
}

fn parse_header(bytes: &[u8]) -> Option<StatorNativeCodeCacheHeaderInfo> {
    if bytes.len() < STATOR_NATIVE_CODE_CACHE_HEADER_SIZE {
        return None;
    }
    if bytes.get(..MAGIC_LEN)? != STATOR_NATIVE_CODE_CACHE_HEADER_MAGIC {
        return None;
    }
    let mut offset = MAGIC_LEN;
    let header_version = read_u32(bytes, &mut offset)?;
    if header_version != STATOR_NATIVE_CODE_CACHE_HEADER_VERSION {
        return None;
    }
    let tier = StatorNativeCodeCacheTier::from_u32(read_u32(bytes, &mut offset)?);
    if tier == StatorNativeCodeCacheTier::StatorNativeCodeCacheTierUnknown {
        return None;
    }
    let native_format_version = read_u32(bytes, &mut offset)?;
    let ffi_abi_version = read_u32(bytes, &mut offset)?;
    let engine_version_digest = read_digest(bytes, &mut offset)?;
    let compiler_version_digest = read_digest(bytes, &mut offset)?;
    let target_triple_digest = read_digest(bytes, &mut offset)?;
    let cpu_feature_set_digest = read_digest(bytes, &mut offset)?;
    let jit_flags_digest = read_digest(bytes, &mut offset)?;
    let source_key_digest = read_digest(bytes, &mut offset)?;
    let payload_digest = read_digest(bytes, &mut offset)?;
    let payload_length_bytes = read_u64(bytes, &mut offset)?;
    if offset != STATOR_NATIVE_CODE_CACHE_HEADER_SIZE {
        return None;
    }
    Some(StatorNativeCodeCacheHeaderInfo {
        header_version,
        tier,
        native_format_version,
        ffi_abi_version,
        engine_version_digest,
        compiler_version_digest,
        target_triple_digest,
        cpu_feature_set_digest,
        jit_flags_digest,
        source_key_digest,
        payload_digest,
        payload_length_bytes,
    })
}

fn non_null_slice<'a>(bytes: *const u8, len: usize) -> Option<&'a [u8]> {
    if bytes.is_null() || len == 0 {
        return None;
    }
    // SAFETY: Caller owns `bytes..bytes+len` for the duration of the FFI call;
    // this helper is only used after null and zero-length checks.
    Some(unsafe { std::slice::from_raw_parts(bytes, len) })
}

fn write_info(
    out_info: *mut StatorNativeCodeCacheHeaderInfo,
    info: StatorNativeCodeCacheHeaderInfo,
) {
    if !out_info.is_null() {
        // SAFETY: Optional out pointer is checked non-null and written once.
        unsafe { out_info.write(info) };
    }
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    let end = offset.checked_add(U32_LEN)?;
    let value = u32::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    let end = offset.checked_add(U64_LEN)?;
    let value = u64::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

fn read_digest(bytes: &[u8], offset: &mut usize) -> Option<[u8; DIGEST_LEN]> {
    let end = offset.checked_add(DIGEST_LEN)?;
    let value = bytes.get(*offset..end)?.try_into().ok()?;
    *offset = end;
    Some(value)
}

fn write_u32(out: &mut [u8], offset: &mut usize, value: u32) {
    let end = *offset + U32_LEN;
    out[*offset..end].copy_from_slice(&value.to_le_bytes());
    *offset = end;
}

fn write_u64(out: &mut [u8], offset: &mut usize, value: u64) {
    let end = *offset + U64_LEN;
    out[*offset..end].copy_from_slice(&value.to_le_bytes());
    *offset = end;
}

fn write_digest(out: &mut [u8], offset: &mut usize, value: &[u8; DIGEST_LEN]) {
    let end = *offset + DIGEST_LEN;
    out[*offset..end].copy_from_slice(value);
    *offset = end;
}

fn sha256_digest(bytes: &[u8]) -> [u8; DIGEST_LEN] {
    Sha256::digest(bytes).into()
}

fn diagnostic_name_bytes(diagnostic: StatorNativeCodeCacheDiagnostic) -> &'static [u8] {
    match diagnostic {
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticAccepted => b"accepted\0",
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument => {
            b"invalid_argument\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload => {
            b"corrupt_payload\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedArtifactType => {
            b"rejected_artifact_type\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedEngineVersion => {
            b"rejected_engine_version\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedFormatVersion => {
            b"rejected_format_version\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedSourceIdentity => {
            b"rejected_source_identity\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedPlatform => {
            b"rejected_platform\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedBuildFeatures => {
            b"rejected_build_features\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedCompilerFlags => {
            b"rejected_compiler_flags\0"
        }
        StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticUnsupportedNativeCode => {
            b"unsupported_native_code\0"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compatibility() -> StatorNativeCodeCacheCompatibility {
        StatorNativeCodeCacheCompatibility {
            tier: StatorNativeCodeCacheTier::StatorNativeCodeCacheTierBaseline,
            native_format_version: 1,
            ffi_abi_version: crate::STATOR_FFI_ABI_VERSION,
            engine_version_digest: native_sha256_digest(b"stator_jse 0.3.5"),
            compiler_version_digest: native_sha256_digest(b"rustc-test llvm-test"),
            target_triple_digest: native_sha256_digest(b"x86_64-pc-windows-gnu"),
            cpu_feature_set_digest: native_sha256_digest(b"avx2,sse4.2"),
            jit_flags_digest: native_sha256_digest(b"baseline;jit-write-protect"),
            source_key_digest: native_sha256_digest(b"canonical-source-key"),
        }
    }

    fn artifact(payload: &[u8]) -> Vec<u8> {
        let header = build_native_code_cache_header(&compatibility(), payload);
        let mut artifact = header.to_vec();
        artifact.extend_from_slice(payload);
        artifact
    }

    fn sentinel_info() -> StatorNativeCodeCacheHeaderInfo {
        StatorNativeCodeCacheHeaderInfo {
            header_version: u32::MAX,
            tier: StatorNativeCodeCacheTier::StatorNativeCodeCacheTierUnknown,
            native_format_version: u32::MAX,
            ffi_abi_version: u32::MAX,
            engine_version_digest: [0xAA; DIGEST_LEN],
            compiler_version_digest: [0xBB; DIGEST_LEN],
            target_triple_digest: [0xCC; DIGEST_LEN],
            cpu_feature_set_digest: [0xDD; DIGEST_LEN],
            jit_flags_digest: [0xEE; DIGEST_LEN],
            source_key_digest: [0x11; DIGEST_LEN],
            payload_digest: [0x22; DIGEST_LEN],
            payload_length_bytes: u64::MAX,
        }
    }

    #[test]
    fn test_classify_header_decodes_canonical_fields() {
        let payload = b"native payload bytes are never executed";
        let artifact = artifact(payload);
        let mut info = StatorNativeCodeCacheHeaderInfo {
            header_version: 0,
            tier: StatorNativeCodeCacheTier::StatorNativeCodeCacheTierUnknown,
            native_format_version: 0,
            ffi_abi_version: 0,
            engine_version_digest: [0; DIGEST_LEN],
            compiler_version_digest: [0; DIGEST_LEN],
            target_triple_digest: [0; DIGEST_LEN],
            cpu_feature_set_digest: [0; DIGEST_LEN],
            jit_flags_digest: [0; DIGEST_LEN],
            source_key_digest: [0; DIGEST_LEN],
            payload_digest: [0; DIGEST_LEN],
            payload_length_bytes: 0,
        };
        // SAFETY: The artifact vector is alive for the duration of the call and out pointer is valid.
        let diagnostic = unsafe {
            stator_native_code_cache_classify_header(artifact.as_ptr(), artifact.len(), &mut info)
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticAccepted
        );
        let expected = compatibility();
        assert_eq!(info.header_version, STATOR_NATIVE_CODE_CACHE_HEADER_VERSION);
        assert_eq!(info.tier, expected.tier);
        assert_eq!(info.native_format_version, expected.native_format_version);
        assert_eq!(info.ffi_abi_version, expected.ffi_abi_version);
        assert_eq!(info.engine_version_digest, expected.engine_version_digest);
        assert_eq!(
            info.compiler_version_digest,
            expected.compiler_version_digest
        );
        assert_eq!(info.target_triple_digest, expected.target_triple_digest);
        assert_eq!(info.cpu_feature_set_digest, expected.cpu_feature_set_digest);
        assert_eq!(info.jit_flags_digest, expected.jit_flags_digest);
        assert_eq!(info.source_key_digest, expected.source_key_digest);
        assert_eq!(info.payload_digest, native_sha256_digest(payload));
        assert_eq!(info.payload_length_bytes, payload.len() as u64);
    }

    #[test]
    fn test_classify_rejects_null_and_empty_input() {
        let mut info = sentinel_info();
        let original = info;
        // SAFETY: null bytes are explicitly rejected before dereference; out pointer is valid.
        let diagnostic =
            unsafe { stator_native_code_cache_classify_header(std::ptr::null(), 1, &mut info) };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument
        );
        assert_eq!(info, original);

        let bytes = b"";
        // SAFETY: zero length is rejected before dereference; out pointer is valid.
        let diagnostic =
            unsafe { stator_native_code_cache_classify_header(bytes.as_ptr(), 0, &mut info) };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument
        );
        assert_eq!(info, original);
    }

    #[test]
    fn test_classify_rejects_short_and_bad_magic_header() {
        let mut info = sentinel_info();
        let original = info;

        let short = [0u8; STATOR_NATIVE_CODE_CACHE_HEADER_SIZE - 1];
        // SAFETY: short buffer pointer is valid for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_classify_header(short.as_ptr(), short.len(), &mut info)
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload
        );
        assert_eq!(info, original);

        let mut bad_magic = artifact(b"payload");
        bad_magic[0] ^= 0xFF;
        // SAFETY: artifact buffer pointer is valid for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_classify_header(bad_magic.as_ptr(), bad_magic.len(), &mut info)
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload
        );
        assert_eq!(info, original);
    }

    #[test]
    fn test_classify_accepts_null_out_info() {
        let artifact = artifact(b"payload");
        // SAFETY: artifact bytes are valid for the duration of the call; null
        // out_info is documented as accepted.
        let diagnostic = unsafe {
            stator_native_code_cache_classify_header(
                artifact.as_ptr(),
                artifact.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticAccepted
        );
    }

    #[test]
    fn test_validate_rejects_null_expected() {
        let payload = b"payload";
        let artifact = artifact(payload);
        let mut info = sentinel_info();
        let original = info;
        // SAFETY: artifact bytes are valid; null expected is explicitly rejected before dereference.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(
                artifact.as_ptr(),
                artifact.len(),
                std::ptr::null(),
                &mut info,
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument
        );
        assert_eq!(info, original);
    }

    #[test]
    fn test_validate_rejects_null_and_empty_input() {
        let expected = compatibility();
        let mut info = sentinel_info();
        let original = info;
        // SAFETY: null bytes are explicitly rejected before dereference; pointers are otherwise valid.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(std::ptr::null(), 1, &expected, &mut info)
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument
        );
        assert_eq!(info, original);

        let bytes = b"";
        // SAFETY: zero length is explicitly rejected before dereference; pointers are otherwise valid.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(bytes.as_ptr(), 0, &expected, &mut info)
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument
        );
        assert_eq!(info, original);
    }

    #[test]
    fn test_validate_treats_header_version_mismatch_as_corrupt_payload() {
        let expected = compatibility();
        let mut artifact = artifact(b"payload");
        artifact[8..12]
            .copy_from_slice(&(STATOR_NATIVE_CODE_CACHE_HEADER_VERSION + 1).to_le_bytes());

        // SAFETY: artifact and expected pointers are valid for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(
                artifact.as_ptr(),
                artifact.len(),
                &expected,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload
        );
    }

    #[test]
    fn test_validate_rejects_payload_digest_and_length_mismatch() {
        let payload = b"payload";
        let expected = compatibility();
        let mut tampered = artifact(payload);
        *tampered.last_mut().unwrap() ^= 0x55;
        // SAFETY: Pointers reference live values for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(
                tampered.as_ptr(),
                tampered.len(),
                &expected,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload
        );

        let truncated =
            &artifact(payload)[..STATOR_NATIVE_CODE_CACHE_HEADER_SIZE + payload.len() - 1];
        // SAFETY: Pointers reference live values for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(
                truncated.as_ptr(),
                truncated.len(),
                &expected,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload
        );
    }

    #[test]
    fn test_validate_rejects_target_abi_compiler_and_jit_mismatch() {
        let payload = b"payload";
        let artifact = artifact(payload);
        let cases: &[(fn(&mut StatorNativeCodeCacheCompatibility), StatorNativeCodeCacheDiagnostic)] = &[
            (
                |c| c.tier = StatorNativeCodeCacheTier::StatorNativeCodeCacheTierMaglev,
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedArtifactType,
            ),
            (
                |c| c.native_format_version += 1,
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedFormatVersion,
            ),
            (
                |c| c.ffi_abi_version += 1,
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedEngineVersion,
            ),
            (
                |c| c.compiler_version_digest = native_sha256_digest(b"other compiler"),
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedBuildFeatures,
            ),
            (
                |c| c.target_triple_digest = native_sha256_digest(b"aarch64-pc-windows-msvc"),
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedPlatform,
            ),
            (
                |c| c.cpu_feature_set_digest = native_sha256_digest(b"sse4.2"),
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedPlatform,
            ),
            (
                |c| c.jit_flags_digest = native_sha256_digest(b"jitless"),
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedCompilerFlags,
            ),
            (
                |c| c.source_key_digest = native_sha256_digest(b"other source"),
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedSourceIdentity,
            ),
        ];
        for (mutate, expected_diag) in cases {
            let mut expected = compatibility();
            mutate(&mut expected);
            // SAFETY: Pointers reference live values for the duration of the call.
            let diagnostic = unsafe {
                stator_native_code_cache_validate_header(
                    artifact.as_ptr(),
                    artifact.len(),
                    &expected,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(&diagnostic, expected_diag);
        }
    }

    #[test]
    fn test_validate_never_reports_runtime_cache_hit() {
        let payload = b"valid payload";
        let artifact = artifact(payload);
        let expected = compatibility();
        // SAFETY: Pointers reference live values for the duration of the call.
        let diagnostic = unsafe {
            stator_native_code_cache_validate_header(
                artifact.as_ptr(),
                artifact.len(),
                &expected,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            diagnostic,
            StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticUnsupportedNativeCode
        );
    }

    #[test]
    fn test_stable_diagnostic_name_mapping() {
        let cases = [
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticAccepted,
                "accepted",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticInvalidArgument,
                "invalid_argument",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticCorruptPayload,
                "corrupt_payload",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedArtifactType,
                "rejected_artifact_type",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedEngineVersion,
                "rejected_engine_version",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedFormatVersion,
                "rejected_format_version",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedSourceIdentity,
                "rejected_source_identity",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedPlatform,
                "rejected_platform",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedBuildFeatures,
                "rejected_build_features",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticRejectedCompilerFlags,
                "rejected_compiler_flags",
            ),
            (
                StatorNativeCodeCacheDiagnostic::StatorNativeCodeCacheDiagnosticUnsupportedNativeCode,
                "unsupported_native_code",
            ),
        ];
        for (diagnostic, expected) in cases {
            let ptr = stator_native_code_cache_diagnostic_name(diagnostic);
            assert!(
                !ptr.is_null(),
                "diagnostic name pointer must be non-null for {diagnostic:?}"
            );
            // SAFETY: The function returns static NUL-terminated strings.
            let actual = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
            assert_eq!(actual, expected);
        }
    }
}

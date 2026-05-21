//! Canonical code-cache key construction for Edge-managed cache artifacts.
//!
//! The serialization implemented here follows `docs/code_cache.md`: all required
//! fields are emitted in schema order, absent optional values are encoded as
//! canonical `null` values, unordered string collections are sorted by UTF-8
//! bytes, and the resulting record is hashed with SHA-256. This module only
//! constructs and validates keys; it does not load, accept, or fake runtime cache
//! hits.

use sha2::{Digest, Sha256};

use crate::STATOR_FFI_ABI_VERSION;

const MAGIC: &[u8] = b"stator-code-cache-key-v1";
const TAG_NULL: u8 = 0;
const TAG_STRING: u8 = 1;
const TAG_ENUM: u8 = 2;
const TAG_U32: u8 = 3;
const TAG_U64: u8 = 4;
const TAG_I32: u8 = 5;
const TAG_BOOL: u8 = 6;
const TAG_BYTES: u8 = 7;
const TAG_STRING_LIST: u8 = 8;

/// Current canonical key schema version from `docs/code_cache.md`.
pub(crate) const CACHE_KEY_SCHEMA_VERSION: u32 = 1;
/// Current module cache payload format version used by the FFI module cache.
pub(crate) const MODULE_CACHE_FORMAT_VERSION: u32 = 6;
/// Current script-cache payload format placeholder for schema construction.
pub(crate) const SCRIPT_CACHE_FORMAT_VERSION: u32 = 1;
/// Current bytecode encoding version used for cache-key invalidation.
pub(crate) const BYTECODE_FORMAT_VERSION: u32 = 1;
/// Current baseline native-code serialization version placeholder.
pub(crate) const BASELINE_CODE_FORMAT_VERSION: u32 = 1;
/// Current optimizing JIT native-code serialization version placeholder.
pub(crate) const JIT_CODE_FORMAT_VERSION: u32 = 1;
/// Current startup/context snapshot format version.
pub(crate) const SNAPSHOT_FORMAT_VERSION: u32 = 2;
/// Current parser AST metadata format version.
pub(crate) const PARSER_AST_FORMAT_VERSION: u32 = 1;
/// Current compiler IR and feedback schema format version.
pub(crate) const COMPILER_IR_FORMAT_VERSION: u32 = 1;
/// Length in bytes of a SHA-256 code-cache key hash.
pub(crate) const CODE_CACHE_KEY_HASH_LEN: usize = 32;

/// Required canonical key field order from `docs/code_cache.md`.
pub const REQUIRED_KEY_FIELDS: &[&str] = &[
    "artifact_type",
    "artifact_scope",
    "artifact_subtype",
    "cache_producer",
    "cache_schema_version",
    "stator_jse_crate_version",
    "stator_jse_ffi_crate_version",
    "stator_ffi_abi_version",
    "bytecode_format_version",
    "module_cache_format_version",
    "script_cache_format_version",
    "baseline_code_format_version",
    "jit_code_format_version",
    "snapshot_format_version",
    "parser_ast_format_version",
    "compiler_ir_format_version",
    "c_header_generation_id",
    "source_hash_algorithm",
    "source_hash",
    "source_length_bytes",
    "source_encoding",
    "resource_url",
    "source_url",
    "source_origin",
    "base_url",
    "referrer_url",
    "integrity_metadata",
    "credentials_mode",
    "referrer_policy",
    "line_offset",
    "column_offset",
    "source_map_url",
    "host_defined_options_hash",
    "compile_options_hash",
    "module_type",
    "module_request_count",
    "module_requests_hash",
    "import_attributes_hash",
    "import_policy_hash",
    "import_map_epoch",
    "resolution_base_url",
    "strict_mode_policy",
    "script_kind",
    "language_mode",
    "parse_goal",
    "enable_top_level_await",
    "enable_import_meta",
    "parser_feature_bits",
    "bytecode_feature_bits",
    "compiler_feature_bits",
    "jit_enabled",
    "tiering_mode",
    "optimization_level",
    "debug_instrumentation",
    "profiling_instrumentation",
    "sandbox_mode",
    "target_arch",
    "target_os",
    "target_env",
    "target_pointer_width",
    "endianness",
    "cpu_vendor",
    "cpu_family_model_stepping",
    "cpu_feature_set",
    "rustc_version",
    "llvm_version",
    "cargo_profile",
    "build_feature_set",
    "link_time_optimization",
    "panic_strategy",
    "edge_channel",
    "edge_build_id",
    "snapshot_digest",
    "snapshot_build_id",
    "snapshot_feature_set",
    "snapshot_context_kind",
];

/// A finalized canonical code-cache key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeCacheKey {
    /// SHA-256 digest of the canonical record.
    pub hash: [u8; CODE_CACHE_KEY_HASH_LEN],
    /// Canonical byte record that was hashed.
    pub record: Vec<u8>,
}

/// Errors returned while constructing a canonical code-cache key.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CodeCacheKeyError {
    /// A canonical field value exceeded the supported u32 length envelope.
    #[error("code-cache key field `{field}` is too large to encode canonically")]
    FieldTooLarge {
        /// Field whose encoded value exceeded the length envelope.
        field: &'static str,
    },
    /// Top-level import attributes contained the same key more than once.
    #[error("duplicate import attribute key `{key}`")]
    DuplicateImportAttribute {
        /// Duplicate import attribute key.
        key: String,
    },
}

macro_rules! token_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident { $($(#[$vmeta:meta])* $variant:ident => $token:literal,)+ }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)+
        }
        impl $name {
            fn token(self) -> &'static str {
                match self {
                    $(Self::$variant => $token,)+
                }
            }
        }
    };
}

token_enum! { /// Artifact type discriminator.
    pub enum ArtifactType {
        /// Classic-script parser and bytecode artifact.
        ScriptBytecode => "script-bytecode",
        /// Module parser metadata and bytecode artifact.
        ModuleBytecode => "module-bytecode",
        /// Baseline native code artifact.
        BaselineCode => "baseline-code",
        /// Optimizing JIT native code artifact.
        JitCode => "jit-code",
        /// Reference to a startup or context snapshot blob.
        SnapshotReference => "snapshot-reference",
    }
}

token_enum! { /// Artifact scope discriminator.
    pub enum ArtifactScope {
        /// Classic script scope.
        ClassicScript => "classic-script",
        /// ECMAScript module scope.
        Module => "module",
        /// Function body scope.
        Function => "function",
        /// Eval source scope.
        Eval => "eval",
        /// WebAssembly module scope.
        WasmModule => "wasm-module",
        /// Snapshot scope.
        Snapshot => "snapshot",
    }
}

token_enum! { /// Source byte encoding.
    pub enum SourceEncoding {
        /// UTF-8 source bytes.
        Utf8 => "utf8",
        /// UTF-16 little-endian source bytes.
        Utf16Le => "utf16le",
        /// UTF-16 big-endian source bytes.
        Utf16Be => "utf16be",
        /// Latin-1 source bytes.
        Latin1 => "latin1",
    }
}

token_enum! { /// Fetch credentials mode.
    pub enum CredentialsMode {
        /// Do not include credentials.
        Omit => "omit",
        /// Include credentials for same-origin requests.
        SameOrigin => "same-origin",
        /// Include credentials.
        Include => "include",
    }
}

token_enum! { /// Module type token.
    pub enum ModuleType {
        /// JavaScript module.
        JavaScript => "javascript",
        /// JSON module.
        Json => "json",
        /// CSS module.
        Css => "css",
        /// WebAssembly module.
        Wasm => "wasm",
    }
}

token_enum! { /// Strict-mode policy.
    pub enum StrictModePolicy {
        /// Respect the source's strictness directives.
        Source => "source",
        /// Force strict mode.
        ForceStrict => "force-strict",
        /// Force sloppy mode.
        ForceSloppy => "force-sloppy",
    }
}

token_enum! { /// Script kind token.
    pub enum ScriptKind {
        /// Classic script.
        Classic => "classic",
        /// ECMAScript module script.
        Module => "module",
        /// Worker script.
        Worker => "worker",
        /// Worklet script.
        Worklet => "worklet",
        /// Extension script.
        Extension => "extension",
        /// Internal engine or embedder script.
        Internal => "internal",
    }
}

token_enum! { /// Parser language mode.
    pub enum LanguageMode {
        /// Sloppy language mode.
        Sloppy => "sloppy",
        /// Strict language mode.
        Strict => "strict",
    }
}

token_enum! { /// Parser goal.
    pub enum ParseGoal {
        /// Classic script parse goal.
        Script => "script",
        /// ECMAScript module parse goal.
        Module => "module",
        /// JSON module parse goal.
        JsonModule => "json-module",
        /// CSS module parse goal.
        CssModule => "css-module",
        /// WebAssembly module parse goal.
        WasmModule => "wasm-module",
    }
}

token_enum! { /// JIT tiering mode.
    pub enum TieringMode {
        /// Interpreter only.
        InterpreterOnly => "interpreter-only",
        /// Baseline tier enabled.
        Baseline => "baseline",
        /// Maglev tier enabled.
        Maglev => "maglev",
        /// Turbofan tier enabled.
        Turbofan => "turbofan",
        /// Adaptive tiering policy.
        Adaptive => "adaptive",
    }
}

token_enum! { /// Target endianness.
    pub enum Endianness {
        /// Little-endian target.
        Little => "little",
        /// Big-endian target.
        Big => "big",
    }
}

token_enum! { /// Cargo or Edge build profile.
    pub enum CargoProfile {
        /// Debug profile.
        Debug => "debug",
        /// Release profile.
        Release => "release",
        /// Release with link-time optimization profile.
        ReleaseLto => "release-lto",
    }
}

token_enum! { /// Rust panic strategy.
    pub enum PanicStrategy {
        /// Stack-unwinding panic strategy.
        Unwind => "unwind",
        /// Abort-on-panic strategy.
        Abort => "abort",
    }
}

token_enum! { /// Edge release channel.
    pub enum EdgeChannel {
        /// Canary channel.
        Canary => "canary",
        /// Dev channel.
        Dev => "dev",
        /// Beta channel.
        Beta => "beta",
        /// Stable channel.
        Stable => "stable",
    }
}

token_enum! { /// Snapshot context kind.
    pub enum SnapshotContextKind {
        /// Startup snapshot context.
        Startup => "startup",
        /// Main-world context snapshot.
        MainWorld => "main-world",
        /// Isolated-world context snapshot.
        IsolatedWorld => "isolated-world",
        /// Worker context snapshot.
        Worker => "worker",
        /// Worklet context snapshot.
        Worklet => "worklet",
    }
}

/// Top-level import attribute key/value pair supplied by the embedder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportAttributeInput {
    /// Attribute key.
    pub key: String,
    /// Attribute value.
    pub value: String,
}

/// Artifact identity fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactIdentity {
    /// Artifact type discriminator.
    pub artifact_type: ArtifactType,
    /// Artifact scope discriminator.
    pub artifact_scope: ArtifactScope,
    /// Optional subtype or tier token.
    pub artifact_subtype: Option<String>,
    /// Producer component token.
    pub cache_producer: String,
}

/// Version and format identity fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VersionIdentity {
    /// Stator engine crate version.
    pub stator_jse_crate_version: String,
    /// Stator FFI crate version.
    pub stator_jse_ffi_crate_version: String,
    /// Packed FFI ABI version.
    pub stator_ffi_abi_version: u32,
    /// Bytecode format version.
    pub bytecode_format_version: u32,
    /// Module-cache format version when applicable.
    pub module_cache_format_version: Option<u32>,
    /// Script-cache format version when applicable.
    pub script_cache_format_version: Option<u32>,
    /// Baseline native-code format version when applicable.
    pub baseline_code_format_version: Option<u32>,
    /// Optimizing JIT format version when applicable.
    pub jit_code_format_version: Option<u32>,
    /// Snapshot blob format version when applicable.
    pub snapshot_format_version: Option<u32>,
    /// Parser AST metadata version.
    pub parser_ast_format_version: u32,
    /// Compiler IR and feedback schema version.
    pub compiler_ir_format_version: u32,
    /// Generated header digest or build id when available.
    pub c_header_generation_id: Option<String>,
}

impl VersionIdentity {
    /// Return current crate, ABI, and format versions for the FFI crate.
    pub fn current() -> Self {
        Self {
            stator_jse_crate_version: env!("STATOR_JSE_CRATE_VERSION").to_owned(),
            stator_jse_ffi_crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            stator_ffi_abi_version: STATOR_FFI_ABI_VERSION,
            bytecode_format_version: BYTECODE_FORMAT_VERSION,
            module_cache_format_version: Some(MODULE_CACHE_FORMAT_VERSION),
            script_cache_format_version: Some(SCRIPT_CACHE_FORMAT_VERSION),
            baseline_code_format_version: Some(BASELINE_CODE_FORMAT_VERSION),
            jit_code_format_version: Some(JIT_CODE_FORMAT_VERSION),
            snapshot_format_version: Some(SNAPSHOT_FORMAT_VERSION),
            parser_ast_format_version: PARSER_AST_FORMAT_VERSION,
            compiler_ir_format_version: COMPILER_IR_FORMAT_VERSION,
            c_header_generation_id: None,
        }
    }
}

/// Source identity, origin, source-map, and compile-option fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceIdentity {
    /// SHA-256 or other configured source hash bytes.
    pub source_hash: Vec<u8>,
    /// Source length in bytes.
    pub source_length_bytes: u64,
    /// Source byte encoding.
    pub source_encoding: SourceEncoding,
    /// Canonical resource URL, or null for anonymous/internal sources.
    pub resource_url: Option<String>,
    /// SourceURL directive URL, when present.
    pub source_url: Option<String>,
    /// Serialized source origin or opaque-origin token.
    pub source_origin: Option<String>,
    /// Canonical base URL.
    pub base_url: Option<String>,
    /// Canonical referrer URL.
    pub referrer_url: Option<String>,
    /// Canonical Subresource Integrity metadata.
    pub integrity_metadata: Option<String>,
    /// Fetch credentials mode.
    pub credentials_mode: Option<CredentialsMode>,
    /// Referrer policy token.
    pub referrer_policy: Option<String>,
    /// Initial diagnostic line offset.
    pub line_offset: i32,
    /// Initial diagnostic column offset.
    pub column_offset: i32,
    /// Source map URL after directive parsing.
    pub source_map_url: Option<String>,
    /// Hash of opaque host-defined compile options.
    pub host_defined_options_hash: Option<Vec<u8>>,
    /// Hash of structured compile options not represented elsewhere.
    pub compile_options_hash: Option<Vec<u8>>,
}

/// Module request, import-attribute, and resolution metadata fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleImportMetadata {
    /// Module type token.
    pub module_type: Option<ModuleType>,
    /// Static module request count.
    pub module_request_count: Option<u32>,
    /// Hash of canonical module request records.
    pub module_requests_hash: Option<Vec<u8>>,
    /// Top-level import attributes to hash canonically by sorted key.
    pub import_attributes: Option<Vec<ImportAttributeInput>>,
    /// Import policy input hash.
    pub import_policy_hash: Option<Vec<u8>>,
    /// Import map version token.
    pub import_map_epoch: Option<String>,
    /// Resolver base URL after import map processing.
    pub resolution_base_url: Option<String>,
}

/// Parser and compiler compatibility fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParserCompilerFlags {
    /// Strict-mode policy.
    pub strict_mode_policy: StrictModePolicy,
    /// Script kind.
    pub script_kind: ScriptKind,
    /// Parser language mode.
    pub language_mode: LanguageMode,
    /// Parser goal.
    pub parse_goal: ParseGoal,
    /// Whether top-level await is enabled.
    pub enable_top_level_await: bool,
    /// Whether import.meta is enabled.
    pub enable_import_meta: bool,
    /// Parser feature bitset.
    pub parser_feature_bits: u64,
    /// Bytecode lowering feature bitset.
    pub bytecode_feature_bits: u64,
    /// Compiler feature bitset.
    pub compiler_feature_bits: u64,
    /// Whether native JIT tiers are allowed.
    pub jit_enabled: bool,
    /// Tiering mode.
    pub tiering_mode: TieringMode,
    /// Optimization level or policy value.
    pub optimization_level: u32,
    /// Whether debug instrumentation is enabled.
    pub debug_instrumentation: bool,
    /// Whether profiling instrumentation is enabled.
    pub profiling_instrumentation: bool,
    /// Sandbox or JIT-write-protection mode token.
    pub sandbox_mode: String,
}

/// Platform, build, and Edge release identity fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildIdentity {
    /// Rust target architecture.
    pub target_arch: String,
    /// Rust target operating system.
    pub target_os: String,
    /// Rust target ABI environment.
    pub target_env: Option<String>,
    /// Target pointer width.
    pub target_pointer_width: u32,
    /// Target endianness.
    pub endianness: Endianness,
    /// CPU vendor when code generation depends on it.
    pub cpu_vendor: Option<String>,
    /// CPU family/model/stepping when native code depends on it.
    pub cpu_family_model_stepping: Option<String>,
    /// Enabled CPU features.
    pub cpu_feature_set: Vec<String>,
    /// Rust compiler version string.
    pub rustc_version: String,
    /// LLVM backend version string when available.
    pub llvm_version: Option<String>,
    /// Cargo or Edge build profile.
    pub cargo_profile: CargoProfile,
    /// Build features, cfgs, allocator mode, GC mode, and feature gates.
    pub build_feature_set: Vec<String>,
    /// Whether link-time optimization was enabled.
    pub link_time_optimization: bool,
    /// Rust panic strategy.
    pub panic_strategy: PanicStrategy,
    /// Edge channel when applicable.
    pub edge_channel: Option<EdgeChannel>,
    /// Edge build identifier when applicable.
    pub edge_build_id: Option<String>,
}

/// Snapshot reference fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SnapshotReferenceIdentity {
    /// Referenced snapshot digest.
    pub snapshot_digest: Option<Vec<u8>>,
    /// Snapshot producer build identifier.
    pub snapshot_build_id: Option<String>,
    /// Snapshot feature set.
    pub snapshot_feature_set: Option<Vec<String>>,
    /// Snapshot context kind.
    pub snapshot_context_kind: Option<SnapshotContextKind>,
}

/// Complete typed input for a canonical code-cache key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeCacheKeyInput {
    /// Artifact identity section.
    pub artifact: ArtifactIdentity,
    /// Version and format identity section.
    pub versions: VersionIdentity,
    /// Source identity and origin section.
    pub source: SourceIdentity,
    /// Module and import metadata section.
    pub module: ModuleImportMetadata,
    /// Parser and compiler flags section.
    pub flags: ParserCompilerFlags,
    /// Platform and build identity section.
    pub build: BuildIdentity,
    /// Snapshot reference section.
    pub snapshot: SnapshotReferenceIdentity,
}

/// Return the SHA-256 digest bytes for arbitrary input.
pub fn sha256_bytes(bytes: &[u8]) -> [u8; CODE_CACHE_KEY_HASH_LEN] {
    Sha256::digest(bytes).into()
}

/// Hash top-level import attributes with canonical UTF-8 key ordering.
pub fn canonical_import_attributes_hash(
    attributes: &[ImportAttributeInput],
) -> Result<[u8; CODE_CACHE_KEY_HASH_LEN], CodeCacheKeyError> {
    let mut sorted = attributes.to_vec();
    sorted.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
    for pair in sorted.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(CodeCacheKeyError::DuplicateImportAttribute {
                key: pair[0].key.clone(),
            });
        }
    }
    let mut record = Vec::new();
    push_len(&mut record, attributes.len(), "import_attributes_hash")?;
    for attr in &sorted {
        push_len(&mut record, attr.key.len(), "import_attributes_hash")?;
        record.extend_from_slice(attr.key.as_bytes());
        push_len(&mut record, attr.value.len(), "import_attributes_hash")?;
        record.extend_from_slice(attr.value.as_bytes());
    }
    Ok(sha256_bytes(&record))
}

/// Build a canonical code-cache key from typed input.
pub fn build_code_cache_key(input: &CodeCacheKeyInput) -> Result<CodeCacheKey, CodeCacheKeyError> {
    let import_attributes_hash = input
        .module
        .import_attributes
        .as_deref()
        .map(canonical_import_attributes_hash)
        .transpose()?
        .map(|hash| hash.to_vec());

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_enum(
        &mut out,
        "artifact_type",
        input.artifact.artifact_type.token(),
    )?;
    push_enum(
        &mut out,
        "artifact_scope",
        input.artifact.artifact_scope.token(),
    )?;
    push_optional_string(
        &mut out,
        "artifact_subtype",
        input.artifact.artifact_subtype.as_deref(),
    )?;
    push_string(&mut out, "cache_producer", &input.artifact.cache_producer)?;
    push_u32(&mut out, "cache_schema_version", CACHE_KEY_SCHEMA_VERSION)?;
    push_string(
        &mut out,
        "stator_jse_crate_version",
        &input.versions.stator_jse_crate_version,
    )?;
    push_string(
        &mut out,
        "stator_jse_ffi_crate_version",
        &input.versions.stator_jse_ffi_crate_version,
    )?;
    push_u32(
        &mut out,
        "stator_ffi_abi_version",
        input.versions.stator_ffi_abi_version,
    )?;
    push_u32(
        &mut out,
        "bytecode_format_version",
        input.versions.bytecode_format_version,
    )?;
    push_optional_u32(
        &mut out,
        "module_cache_format_version",
        input.versions.module_cache_format_version,
    )?;
    push_optional_u32(
        &mut out,
        "script_cache_format_version",
        input.versions.script_cache_format_version,
    )?;
    push_optional_u32(
        &mut out,
        "baseline_code_format_version",
        input.versions.baseline_code_format_version,
    )?;
    push_optional_u32(
        &mut out,
        "jit_code_format_version",
        input.versions.jit_code_format_version,
    )?;
    push_optional_u32(
        &mut out,
        "snapshot_format_version",
        input.versions.snapshot_format_version,
    )?;
    push_u32(
        &mut out,
        "parser_ast_format_version",
        input.versions.parser_ast_format_version,
    )?;
    push_u32(
        &mut out,
        "compiler_ir_format_version",
        input.versions.compiler_ir_format_version,
    )?;
    push_optional_string(
        &mut out,
        "c_header_generation_id",
        input.versions.c_header_generation_id.as_deref(),
    )?;
    push_enum(&mut out, "source_hash_algorithm", "sha256")?;
    push_bytes(&mut out, "source_hash", &input.source.source_hash)?;
    push_u64(
        &mut out,
        "source_length_bytes",
        input.source.source_length_bytes,
    )?;
    push_enum(
        &mut out,
        "source_encoding",
        input.source.source_encoding.token(),
    )?;
    push_optional_string(
        &mut out,
        "resource_url",
        input.source.resource_url.as_deref(),
    )?;
    push_optional_string(&mut out, "source_url", input.source.source_url.as_deref())?;
    push_optional_string(
        &mut out,
        "source_origin",
        input.source.source_origin.as_deref(),
    )?;
    push_optional_string(&mut out, "base_url", input.source.base_url.as_deref())?;
    push_optional_string(
        &mut out,
        "referrer_url",
        input.source.referrer_url.as_deref(),
    )?;
    push_optional_string(
        &mut out,
        "integrity_metadata",
        input.source.integrity_metadata.as_deref(),
    )?;
    push_optional_enum(
        &mut out,
        "credentials_mode",
        input.source.credentials_mode.map(CredentialsMode::token),
    )?;
    push_optional_string(
        &mut out,
        "referrer_policy",
        input.source.referrer_policy.as_deref(),
    )?;
    push_i32(&mut out, "line_offset", input.source.line_offset)?;
    push_i32(&mut out, "column_offset", input.source.column_offset)?;
    push_optional_string(
        &mut out,
        "source_map_url",
        input.source.source_map_url.as_deref(),
    )?;
    push_optional_bytes(
        &mut out,
        "host_defined_options_hash",
        input.source.host_defined_options_hash.as_deref(),
    )?;
    push_optional_bytes(
        &mut out,
        "compile_options_hash",
        input.source.compile_options_hash.as_deref(),
    )?;
    push_optional_enum(
        &mut out,
        "module_type",
        input.module.module_type.map(ModuleType::token),
    )?;
    push_optional_u32(
        &mut out,
        "module_request_count",
        input.module.module_request_count,
    )?;
    push_optional_bytes(
        &mut out,
        "module_requests_hash",
        input.module.module_requests_hash.as_deref(),
    )?;
    push_optional_bytes(
        &mut out,
        "import_attributes_hash",
        import_attributes_hash.as_deref(),
    )?;
    push_optional_bytes(
        &mut out,
        "import_policy_hash",
        input.module.import_policy_hash.as_deref(),
    )?;
    push_optional_string(
        &mut out,
        "import_map_epoch",
        input.module.import_map_epoch.as_deref(),
    )?;
    push_optional_string(
        &mut out,
        "resolution_base_url",
        input.module.resolution_base_url.as_deref(),
    )?;
    push_enum(
        &mut out,
        "strict_mode_policy",
        input.flags.strict_mode_policy.token(),
    )?;
    push_enum(&mut out, "script_kind", input.flags.script_kind.token())?;
    push_enum(&mut out, "language_mode", input.flags.language_mode.token())?;
    push_enum(&mut out, "parse_goal", input.flags.parse_goal.token())?;
    push_bool(
        &mut out,
        "enable_top_level_await",
        input.flags.enable_top_level_await,
    )?;
    push_bool(
        &mut out,
        "enable_import_meta",
        input.flags.enable_import_meta,
    )?;
    push_u64(
        &mut out,
        "parser_feature_bits",
        input.flags.parser_feature_bits,
    )?;
    push_u64(
        &mut out,
        "bytecode_feature_bits",
        input.flags.bytecode_feature_bits,
    )?;
    push_u64(
        &mut out,
        "compiler_feature_bits",
        input.flags.compiler_feature_bits,
    )?;
    push_bool(&mut out, "jit_enabled", input.flags.jit_enabled)?;
    push_enum(&mut out, "tiering_mode", input.flags.tiering_mode.token())?;
    push_u32(
        &mut out,
        "optimization_level",
        input.flags.optimization_level,
    )?;
    push_bool(
        &mut out,
        "debug_instrumentation",
        input.flags.debug_instrumentation,
    )?;
    push_bool(
        &mut out,
        "profiling_instrumentation",
        input.flags.profiling_instrumentation,
    )?;
    push_enum(&mut out, "sandbox_mode", &input.flags.sandbox_mode)?;
    push_string(&mut out, "target_arch", &input.build.target_arch)?;
    push_string(&mut out, "target_os", &input.build.target_os)?;
    push_optional_string(&mut out, "target_env", input.build.target_env.as_deref())?;
    push_u32(
        &mut out,
        "target_pointer_width",
        input.build.target_pointer_width,
    )?;
    push_enum(&mut out, "endianness", input.build.endianness.token())?;
    push_optional_string(&mut out, "cpu_vendor", input.build.cpu_vendor.as_deref())?;
    push_optional_string(
        &mut out,
        "cpu_family_model_stepping",
        input.build.cpu_family_model_stepping.as_deref(),
    )?;
    push_string_list(&mut out, "cpu_feature_set", &input.build.cpu_feature_set)?;
    push_string(&mut out, "rustc_version", &input.build.rustc_version)?;
    push_optional_string(
        &mut out,
        "llvm_version",
        input.build.llvm_version.as_deref(),
    )?;
    push_enum(&mut out, "cargo_profile", input.build.cargo_profile.token())?;
    push_string_list(
        &mut out,
        "build_feature_set",
        &input.build.build_feature_set,
    )?;
    push_bool(
        &mut out,
        "link_time_optimization",
        input.build.link_time_optimization,
    )?;
    push_enum(
        &mut out,
        "panic_strategy",
        input.build.panic_strategy.token(),
    )?;
    push_optional_enum(
        &mut out,
        "edge_channel",
        input.build.edge_channel.map(EdgeChannel::token),
    )?;
    push_optional_string(
        &mut out,
        "edge_build_id",
        input.build.edge_build_id.as_deref(),
    )?;
    push_optional_bytes(
        &mut out,
        "snapshot_digest",
        input.snapshot.snapshot_digest.as_deref(),
    )?;
    push_optional_string(
        &mut out,
        "snapshot_build_id",
        input.snapshot.snapshot_build_id.as_deref(),
    )?;
    push_optional_string_list(
        &mut out,
        "snapshot_feature_set",
        input.snapshot.snapshot_feature_set.as_deref(),
    )?;
    push_optional_enum(
        &mut out,
        "snapshot_context_kind",
        input
            .snapshot
            .snapshot_context_kind
            .map(SnapshotContextKind::token),
    )?;

    let hash = sha256_bytes(&out);
    Ok(CodeCacheKey { hash, record: out })
}

fn push_len(out: &mut Vec<u8>, len: usize, field: &'static str) -> Result<(), CodeCacheKeyError> {
    let len = u32::try_from(len).map_err(|_| CodeCacheKeyError::FieldTooLarge { field })?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn push_field(
    out: &mut Vec<u8>,
    field: &'static str,
    tag: u8,
    value: &[u8],
) -> Result<(), CodeCacheKeyError> {
    out.extend_from_slice(field.as_bytes());
    out.push(0);
    out.push(tag);
    push_len(out, value.len(), field)?;
    out.extend_from_slice(value);
    out.push(0);
    Ok(())
}

fn push_null(out: &mut Vec<u8>, field: &'static str) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_NULL, &[])
}
fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_STRING, value.as_bytes())
}
fn push_optional_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: Option<&str>,
) -> Result<(), CodeCacheKeyError> {
    match value {
        Some(value) => push_string(out, field, value),
        None => push_null(out, field),
    }
}
fn push_enum(out: &mut Vec<u8>, field: &'static str, value: &str) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_ENUM, value.as_bytes())
}
fn push_optional_enum(
    out: &mut Vec<u8>,
    field: &'static str,
    value: Option<&str>,
) -> Result<(), CodeCacheKeyError> {
    match value {
        Some(value) => push_enum(out, field, value),
        None => push_null(out, field),
    }
}
fn push_u32(out: &mut Vec<u8>, field: &'static str, value: u32) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_U32, &value.to_le_bytes())
}
fn push_optional_u32(
    out: &mut Vec<u8>,
    field: &'static str,
    value: Option<u32>,
) -> Result<(), CodeCacheKeyError> {
    match value {
        Some(value) => push_u32(out, field, value),
        None => push_null(out, field),
    }
}
fn push_u64(out: &mut Vec<u8>, field: &'static str, value: u64) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_U64, &value.to_le_bytes())
}
fn push_i32(out: &mut Vec<u8>, field: &'static str, value: i32) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_I32, &value.to_le_bytes())
}
fn push_bool(out: &mut Vec<u8>, field: &'static str, value: bool) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_BOOL, &[u8::from(value)])
}
fn push_bytes(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &[u8],
) -> Result<(), CodeCacheKeyError> {
    push_field(out, field, TAG_BYTES, value)
}
fn push_optional_bytes(
    out: &mut Vec<u8>,
    field: &'static str,
    value: Option<&[u8]>,
) -> Result<(), CodeCacheKeyError> {
    match value {
        Some(value) => push_bytes(out, field, value),
        None => push_null(out, field),
    }
}

fn push_string_list(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &[String],
) -> Result<(), CodeCacheKeyError> {
    let mut sorted = value.to_vec();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    let mut encoded = Vec::new();
    push_len(&mut encoded, sorted.len(), field)?;
    for item in &sorted {
        push_len(&mut encoded, item.len(), field)?;
        encoded.extend_from_slice(item.as_bytes());
    }
    push_field(out, field, TAG_STRING_LIST, &encoded)
}

fn push_optional_string_list(
    out: &mut Vec<u8>,
    field: &'static str,
    value: Option<&[String]>,
) -> Result<(), CodeCacheKeyError> {
    match value {
        Some(value) => push_string_list(out, field, value),
        None => push_null(out, field),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> CodeCacheKeyInput {
        CodeCacheKeyInput {
            artifact: ArtifactIdentity {
                artifact_type: ArtifactType::ModuleBytecode,
                artifact_scope: ArtifactScope::Module,
                artifact_subtype: Some("javascript".to_owned()),
                cache_producer: "stator_jse_ffi".to_owned(),
            },
            versions: VersionIdentity::current(),
            source: SourceIdentity {
                source_hash: sha256_bytes(b"import value from './dep.js';").to_vec(),
                source_length_bytes: 29,
                source_encoding: SourceEncoding::Utf8,
                resource_url: Some("https://example.test/app.js".to_owned()),
                source_url: None,
                source_origin: Some("https://example.test".to_owned()),
                base_url: Some("https://example.test/".to_owned()),
                referrer_url: Some("https://referrer.test/".to_owned()),
                integrity_metadata: None,
                credentials_mode: Some(CredentialsMode::SameOrigin),
                referrer_policy: Some("strict-origin-when-cross-origin".to_owned()),
                line_offset: 0,
                column_offset: 0,
                source_map_url: Some("app.js.map".to_owned()),
                host_defined_options_hash: Some(sha256_bytes(b"host-options").to_vec()),
                compile_options_hash: Some(sha256_bytes(b"compile-options").to_vec()),
            },
            module: ModuleImportMetadata {
                module_type: Some(ModuleType::JavaScript),
                module_request_count: Some(1),
                module_requests_hash: Some(sha256_bytes(b"./dep.js").to_vec()),
                import_attributes: Some(vec![
                    ImportAttributeInput {
                        key: "type".to_owned(),
                        value: "json".to_owned(),
                    },
                    ImportAttributeInput {
                        key: "integrity".to_owned(),
                        value: "sha256-abc".to_owned(),
                    },
                ]),
                import_policy_hash: Some(sha256_bytes(b"policy").to_vec()),
                import_map_epoch: Some("epoch-7".to_owned()),
                resolution_base_url: Some("https://example.test/".to_owned()),
            },
            flags: ParserCompilerFlags {
                strict_mode_policy: StrictModePolicy::Source,
                script_kind: ScriptKind::Module,
                language_mode: LanguageMode::Strict,
                parse_goal: ParseGoal::Module,
                enable_top_level_await: true,
                enable_import_meta: true,
                parser_feature_bits: 1,
                bytecode_feature_bits: 2,
                compiler_feature_bits: 3,
                jit_enabled: false,
                tiering_mode: TieringMode::InterpreterOnly,
                optimization_level: 0,
                debug_instrumentation: false,
                profiling_instrumentation: false,
                sandbox_mode: "jitless".to_owned(),
            },
            build: BuildIdentity {
                target_arch: "x86_64".to_owned(),
                target_os: "windows".to_owned(),
                target_env: Some("gnu".to_owned()),
                target_pointer_width: 64,
                endianness: Endianness::Little,
                cpu_vendor: Some("generic".to_owned()),
                cpu_family_model_stepping: None,
                cpu_feature_set: vec!["sse4.2".to_owned(), "avx2".to_owned()],
                rustc_version: "rustc-test".to_owned(),
                llvm_version: None,
                cargo_profile: CargoProfile::Release,
                build_feature_set: vec!["gc-immix".to_owned(), "wasm".to_owned()],
                link_time_optimization: false,
                panic_strategy: PanicStrategy::Unwind,
                edge_channel: Some(EdgeChannel::Canary),
                edge_build_id: Some("edge-123".to_owned()),
            },
            snapshot: SnapshotReferenceIdentity {
                snapshot_digest: Some(sha256_bytes(b"snapshot").to_vec()),
                snapshot_build_id: Some("snap-1".to_owned()),
                snapshot_feature_set: Some(vec!["warm-context".to_owned()]),
                snapshot_context_kind: Some(SnapshotContextKind::Startup),
            },
        }
    }

    fn field_names(record: &[u8]) -> Vec<String> {
        let mut names = Vec::new();
        let mut offset = MAGIC.len();
        while offset < record.len() {
            let name_end = record[offset..]
                .iter()
                .position(|b| *b == 0)
                .expect("field name terminator")
                + offset;
            names.push(String::from_utf8(record[offset..name_end].to_vec()).unwrap());
            offset = name_end + 1;
            offset += 1;
            let len = u32::from_le_bytes(record[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4 + len + 1;
        }
        names
    }

    #[test]
    fn test_deterministic_output_and_field_order() {
        let input = sample_input();
        let a = build_code_cache_key(&input).unwrap();
        let b = build_code_cache_key(&input).unwrap();
        assert_eq!(a, b);
        assert_eq!(field_names(&a.record), REQUIRED_KEY_FIELDS);
        assert!(a.record.starts_with(MAGIC));
    }

    #[test]
    fn test_null_and_empty_string_are_distinct() {
        let mut absent = sample_input();
        absent.source.source_url = None;
        let mut empty = absent.clone();
        empty.source.source_url = Some(String::new());
        assert_ne!(
            build_code_cache_key(&absent).unwrap().hash,
            build_code_cache_key(&empty).unwrap().hash
        );
    }

    #[test]
    fn test_mismatch_relevant_fields_change_hash() {
        let input = sample_input();
        let base = build_code_cache_key(&input).unwrap().hash;
        let mut source_map_changed = input.clone();
        source_map_changed.source.source_map_url = Some("other.map".to_owned());
        assert_ne!(
            base,
            build_code_cache_key(&source_map_changed).unwrap().hash
        );
        let mut abi_changed = input.clone();
        abi_changed.versions.stator_ffi_abi_version += 1;
        assert_ne!(base, build_code_cache_key(&abi_changed).unwrap().hash);
    }

    #[test]
    fn test_import_attribute_order_is_normalized() {
        let mut a = sample_input();
        let mut b = a.clone();
        b.module.import_attributes.as_mut().unwrap().reverse();
        assert_eq!(
            build_code_cache_key(&a).unwrap().hash,
            build_code_cache_key(&b).unwrap().hash
        );
        a.module
            .import_attributes
            .as_mut()
            .unwrap()
            .push(ImportAttributeInput {
                key: "type".to_owned(),
                value: "json".to_owned(),
            });
        assert!(
            matches!(build_code_cache_key(&a), Err(CodeCacheKeyError::DuplicateImportAttribute { key }) if key == "type")
        );
    }

    #[test]
    fn test_docs_required_fields_are_represented() {
        let key = build_code_cache_key(&sample_input()).unwrap();
        let names = field_names(&key.record);
        for required in REQUIRED_KEY_FIELDS {
            assert!(
                names.iter().any(|name| name == required),
                "missing {required}"
            );
        }
        assert_eq!(names.len(), REQUIRED_KEY_FIELDS.len());
    }

    #[test]
    fn test_sorted_string_lists_are_canonical() {
        let mut a = sample_input();
        let mut b = a.clone();
        a.build.cpu_feature_set = vec!["sse4.2".to_owned(), "avx2".to_owned()];
        b.build.cpu_feature_set = vec!["avx2".to_owned(), "sse4.2".to_owned()];
        assert_eq!(
            build_code_cache_key(&a).unwrap().hash,
            build_code_cache_key(&b).unwrap().hash
        );
    }

    fn assert_field_mutation_changes_hash<F>(label: &str, mutate: F)
    where
        F: FnOnce(&mut CodeCacheKeyInput),
    {
        let base = sample_input();
        let base_hash = build_code_cache_key(&base).unwrap().hash;
        let mut mutated = base;
        mutate(&mut mutated);
        let mutated_hash = build_code_cache_key(&mutated).unwrap().hash;
        assert_ne!(base_hash, mutated_hash, "{label} did not change key hash");
    }

    fn assert_null_distinct_from_empty<F, S>(label: &str, mut set_to_empty: F, mut set_to_null: S)
    where
        F: FnMut(&mut CodeCacheKeyInput),
        S: FnMut(&mut CodeCacheKeyInput),
    {
        let mut empty_input = sample_input();
        set_to_empty(&mut empty_input);
        let mut null_input = sample_input();
        set_to_null(&mut null_input);
        assert_ne!(
            build_code_cache_key(&empty_input).unwrap().hash,
            build_code_cache_key(&null_input).unwrap().hash,
            "{label} null must differ from empty"
        );
    }

    #[test]
    fn test_resource_url_changes_key() {
        assert_field_mutation_changes_hash("resource_url", |input| {
            input.source.resource_url = Some("https://example.test/other.js".to_owned());
        });
        assert_null_distinct_from_empty(
            "resource_url",
            |input| input.source.resource_url = Some(String::new()),
            |input| input.source.resource_url = None,
        );
    }

    #[test]
    fn test_source_url_directive_changes_key_distinct_from_resource_url() {
        let mut input = sample_input();
        input.source.resource_url = Some("https://example.test/app.js".to_owned());
        input.source.source_url = None;
        let base = build_code_cache_key(&input).unwrap().hash;
        let mut with_directive = input.clone();
        with_directive.source.source_url = Some("https://example.test/app.js".to_owned());
        assert_ne!(
            base,
            build_code_cache_key(&with_directive).unwrap().hash,
            "//# sourceURL= directive must affect key even when equal to resource_url"
        );
        let mut other_directive = input;
        other_directive.source.source_url = Some("webpack:///app.js".to_owned());
        assert_ne!(
            build_code_cache_key(&with_directive).unwrap().hash,
            build_code_cache_key(&other_directive).unwrap().hash
        );
    }

    #[test]
    fn test_source_origin_changes_key_independent_of_resource_url() {
        assert_field_mutation_changes_hash("source_origin", |input| {
            input.source.source_origin = Some("https://other.test".to_owned());
        });
        assert_field_mutation_changes_hash("opaque source_origin", |input| {
            input.source.source_origin = Some("opaque://abc-123".to_owned());
        });
        assert_null_distinct_from_empty(
            "source_origin",
            |input| input.source.source_origin = Some(String::new()),
            |input| input.source.source_origin = None,
        );
    }

    #[test]
    fn test_base_url_changes_key() {
        assert_field_mutation_changes_hash("base_url", |input| {
            input.source.base_url = Some("https://example.test/sub/".to_owned());
        });
        assert_null_distinct_from_empty(
            "base_url",
            |input| input.source.base_url = Some(String::new()),
            |input| input.source.base_url = None,
        );
    }

    #[test]
    fn test_referrer_url_changes_key() {
        assert_field_mutation_changes_hash("referrer_url", |input| {
            input.source.referrer_url = Some("https://other-referrer.test/".to_owned());
        });
        assert_null_distinct_from_empty(
            "referrer_url",
            |input| input.source.referrer_url = Some(String::new()),
            |input| input.source.referrer_url = None,
        );
    }

    #[test]
    fn test_integrity_metadata_changes_key() {
        assert_field_mutation_changes_hash("integrity_metadata", |input| {
            input.source.integrity_metadata = Some("sha384-xyz".to_owned());
        });
        assert_null_distinct_from_empty(
            "integrity_metadata",
            |input| input.source.integrity_metadata = Some(String::new()),
            |input| input.source.integrity_metadata = None,
        );
    }

    #[test]
    fn test_credentials_mode_changes_key() {
        for mode in [
            CredentialsMode::Omit,
            CredentialsMode::SameOrigin,
            CredentialsMode::Include,
        ] {
            let mut a = sample_input();
            a.source.credentials_mode = Some(mode);
            let mut b = sample_input();
            b.source.credentials_mode = None;
            assert_ne!(
                build_code_cache_key(&a).unwrap().hash,
                build_code_cache_key(&b).unwrap().hash,
                "credentials_mode {} must differ from null",
                mode.token()
            );
        }
        let mut omit = sample_input();
        omit.source.credentials_mode = Some(CredentialsMode::Omit);
        let mut include = sample_input();
        include.source.credentials_mode = Some(CredentialsMode::Include);
        assert_ne!(
            build_code_cache_key(&omit).unwrap().hash,
            build_code_cache_key(&include).unwrap().hash
        );
    }

    #[test]
    fn test_referrer_policy_changes_key() {
        assert_field_mutation_changes_hash("referrer_policy", |input| {
            input.source.referrer_policy = Some("no-referrer".to_owned());
        });
        assert_null_distinct_from_empty(
            "referrer_policy",
            |input| input.source.referrer_policy = Some(String::new()),
            |input| input.source.referrer_policy = None,
        );
    }

    #[test]
    fn test_line_and_column_offsets_change_key_including_negative() {
        assert_field_mutation_changes_hash("line_offset positive", |input| {
            input.source.line_offset = 5;
        });
        assert_field_mutation_changes_hash("line_offset negative", |input| {
            input.source.line_offset = -3;
        });
        assert_field_mutation_changes_hash("column_offset positive", |input| {
            input.source.column_offset = 12;
        });
        assert_field_mutation_changes_hash("column_offset negative", |input| {
            input.source.column_offset = -1;
        });
        let mut neg = sample_input();
        neg.source.line_offset = -1;
        let mut pos = sample_input();
        pos.source.line_offset = 1;
        assert_ne!(
            build_code_cache_key(&neg).unwrap().hash,
            build_code_cache_key(&pos).unwrap().hash,
            "signed line_offset must distinguish sign"
        );
    }

    #[test]
    fn test_source_map_url_changes_key_even_when_source_unchanged() {
        let base_input = sample_input();
        let base_hash = build_code_cache_key(&base_input).unwrap().hash;
        let mut changed = base_input.clone();
        changed.source.source_map_url = Some("https://example.test/other.map".to_owned());
        assert_ne!(base_hash, build_code_cache_key(&changed).unwrap().hash);
        assert_null_distinct_from_empty(
            "source_map_url",
            |input| input.source.source_map_url = Some(String::new()),
            |input| input.source.source_map_url = None,
        );
    }

    #[test]
    fn test_host_defined_and_compile_options_change_key() {
        assert_field_mutation_changes_hash("host_defined_options_hash", |input| {
            input.source.host_defined_options_hash = Some(sha256_bytes(b"other-host").to_vec());
        });
        assert_field_mutation_changes_hash("compile_options_hash", |input| {
            input.source.compile_options_hash = Some(sha256_bytes(b"other-compile").to_vec());
        });
        // Null must differ from a present zero-length digest.
        let mut empty_digest = sample_input();
        empty_digest.source.compile_options_hash = Some(Vec::new());
        let mut null_digest = sample_input();
        null_digest.source.compile_options_hash = None;
        assert_ne!(
            build_code_cache_key(&empty_digest).unwrap().hash,
            build_code_cache_key(&null_digest).unwrap().hash,
            "compile_options_hash zero-length digest must differ from null"
        );
        let mut empty_host = sample_input();
        empty_host.source.host_defined_options_hash = Some(Vec::new());
        let mut null_host = sample_input();
        null_host.source.host_defined_options_hash = None;
        assert_ne!(
            build_code_cache_key(&empty_host).unwrap().hash,
            build_code_cache_key(&null_host).unwrap().hash
        );
    }

    #[test]
    fn test_import_policy_and_map_epoch_change_key() {
        assert_field_mutation_changes_hash("import_policy_hash", |input| {
            input.module.import_policy_hash = Some(sha256_bytes(b"other-policy").to_vec());
        });
        assert_field_mutation_changes_hash("import_map_epoch", |input| {
            input.module.import_map_epoch = Some("epoch-8".to_owned());
        });
        assert_field_mutation_changes_hash("resolution_base_url", |input| {
            input.module.resolution_base_url = Some("https://example.test/v2/".to_owned());
        });
    }

    #[test]
    fn test_import_attributes_value_change_affects_key() {
        let base = sample_input();
        let base_hash = build_code_cache_key(&base).unwrap().hash;
        let mut changed_value = base.clone();
        changed_value
            .module
            .import_attributes
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|a| a.key == "type")
            .unwrap()
            .value = "css".to_owned();
        assert_ne!(
            base_hash,
            build_code_cache_key(&changed_value).unwrap().hash,
            "import attribute value change must alter key"
        );
        let mut added = base;
        added
            .module
            .import_attributes
            .as_mut()
            .unwrap()
            .push(ImportAttributeInput {
                key: "extra".to_owned(),
                value: "v".to_owned(),
            });
        assert_ne!(base_hash, build_code_cache_key(&added).unwrap().hash);
    }

    #[test]
    fn test_import_attributes_present_empty_distinct_from_null() {
        let mut empty = sample_input();
        empty.module.import_attributes = Some(Vec::new());
        let mut null = sample_input();
        null.module.import_attributes = None;
        assert_ne!(
            build_code_cache_key(&empty).unwrap().hash,
            build_code_cache_key(&null).unwrap().hash,
            "present empty import attributes list must differ from null"
        );
    }
}

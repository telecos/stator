//! Host module integration hooks consulted by the interpreter.
//!
//! The interpreter has no direct knowledge of the embedder's module
//! loader.  Embedders publish a [`HostModuleLoader`] (and optional
//! current-module URL) onto the active thread before driving the
//! interpreter; the dynamic `import()` runtime call and the
//! `import.meta` accessor consult these thread-local hooks.
//!
//! When no loader is installed the runtime falls back to the
//! pre-existing fail-closed behaviour: dynamic `import()` rejects with
//! a `TypeError`, `import.meta.url` is the empty string, and
//! `import.meta.resolve` throws.  This keeps the slice safe for
//! embedders that have not opted in.

#![allow(clippy::result_large_err)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::error::JsError;
use crate::builtins::promise::{JsPromise, MicrotaskQueue};
use crate::objects::value::JsValue;

/// A host-visible ECMAScript import attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostImportAttribute {
    /// Attribute key (for example, `type`).
    pub key: String,
    /// Attribute value.
    pub value: String,
}

/// Host request object for dynamic `import()`.
///
/// The request owns the promise returned to JavaScript. Hosts may settle it
/// synchronously inside [`HostModuleLoader::dynamic_import`] or retain a clone
/// and settle it later on the same thread after fetch/compile/evaluate work
/// completes.
#[derive(Clone)]
pub struct HostDynamicImportRequest {
    specifier: String,
    referrer: Option<String>,
    attributes: Vec<HostImportAttribute>,
    source_metadata: Option<HostModuleSourceMetadata>,
    promise: JsPromise,
    queue: MicrotaskQueue,
}

/// Host-visible module source identity and cache metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostModuleSourceMetadata {
    /// Optional source URL / sourceURL directive.
    pub source_url: Option<String>,
    /// Optional browser origin URL.
    pub origin_url: Option<String>,
    /// Optional referrer URL.
    pub referrer_url: Option<String>,
    /// Optional source map URL.
    pub source_map_url: Option<String>,
    /// Optional source map digest.
    pub source_map_digest: Option<String>,
    /// Optional browser cache-policy token.
    pub cache_policy: Option<String>,
    /// Optional opaque Edge cache metadata, encoded by the host.
    pub edge_cache_metadata: Option<String>,
}

impl HostDynamicImportRequest {
    /// Create a host dynamic-import request around a pending promise.
    pub fn new(
        specifier: String,
        referrer: Option<String>,
        attributes: Vec<HostImportAttribute>,
        promise: JsPromise,
        queue: MicrotaskQueue,
    ) -> Self {
        Self {
            specifier,
            referrer,
            attributes,
            source_metadata: None,
            promise,
            queue,
        }
    }

    /// Attach source metadata to this dynamic import request.
    pub fn with_source_metadata(mut self, metadata: HostModuleSourceMetadata) -> Self {
        self.source_metadata = Some(metadata);
        self
    }

    /// Requested module specifier after `ToString`.
    pub fn specifier(&self) -> &str {
        &self.specifier
    }

    /// URL of the referrer module, when known.
    pub fn referrer(&self) -> Option<&str> {
        self.referrer.as_deref()
    }

    /// Import attributes supplied by `import(specifier, { with: ... })`.
    pub fn attributes(&self) -> &[HostImportAttribute] {
        &self.attributes
    }

    /// Source/cache metadata inherited from the referrer and URL resolver.
    pub fn source_metadata(&self) -> Option<&HostModuleSourceMetadata> {
        self.source_metadata.as_ref()
    }

    /// Promise returned to JavaScript for this dynamic import.
    pub fn promise(&self) -> JsPromise {
        self.promise.clone()
    }

    /// Fulfil the dynamic-import promise.
    pub fn resolve(&self, namespace: JsValue) {
        self.promise.resolve(namespace, &self.queue);
    }

    /// Reject the dynamic-import promise with a structured JavaScript error.
    pub fn reject(&self, error: JsError) {
        self.promise
            .reject(JsValue::Error(Rc::new(error)), &self.queue);
    }
}

/// Host-populated fields for an `import.meta` object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostImportMeta {
    /// Public `import.meta.url` value.
    pub url: String,
    /// Optional host origin/base identity.
    pub origin: Option<String>,
    /// Optional module source kind metadata.
    pub source_type: Option<String>,
    /// Optional browser base URL metadata.
    pub base_url: Option<String>,
    /// Optional Subresource Integrity metadata.
    pub integrity_metadata: Option<String>,
    /// Optional browser credentials-mode metadata.
    pub credentials_mode: Option<String>,
    /// Optional browser referrer-policy metadata.
    pub referrer_policy: Option<String>,
    /// Optional HTML parser-metadata classification.
    pub parser_metadata: Option<String>,
    /// Optional source URL / sourceURL directive.
    pub source_url: Option<String>,
    /// Optional browser origin URL.
    pub origin_url: Option<String>,
    /// Optional referrer URL.
    pub referrer_url: Option<String>,
    /// Optional source map URL.
    pub source_map_url: Option<String>,
    /// Optional source map digest.
    pub source_map_digest: Option<String>,
    /// Optional browser cache-policy token.
    pub cache_policy: Option<String>,
    /// Optional opaque Edge cache metadata.
    pub edge_cache_metadata: Option<String>,
}

/// Embedder hook for resolving dynamic `import()` and `import.meta.resolve`.
///
/// Dynamic import is intentionally start/settle split: the interpreter creates
/// the JavaScript promise and passes a [`HostDynamicImportRequest`] to the host.
/// Hosts may settle the request immediately or retain a clone and settle it
/// later after browser fetch/compile/evaluate lifecycle completes.
pub trait HostModuleLoader {
    /// Start host processing for dynamic `import()`.
    ///
    /// Returning `Err` rejects the promise immediately. Returning `Ok(())`
    /// means the host has accepted the request and is responsible for calling
    /// [`HostDynamicImportRequest::resolve`] or
    /// [`HostDynamicImportRequest::reject`]. If it does neither, the promise
    /// remains pending rather than silently falling back to fake resolution.
    fn dynamic_import(&self, request: HostDynamicImportRequest) -> Result<(), JsError>;

    /// Resolve `specifier` to a URL string for `import.meta.resolve`.
    ///
    /// `referrer` is the URL of the importing module, or `None` when
    /// no module URL is currently published.  Returning `Err` causes
    /// the call to throw.
    fn resolve(&self, specifier: &str, referrer: Option<&str>) -> Result<String, JsError>;

    /// Populate host fields for the current module's `import.meta` object.
    ///
    /// The default implementation preserves Stator's built-in metadata. Hosts
    /// may override URL/policy fields or return `Err` to fail module evaluation
    /// closed before any partially-populated `import.meta` object is exposed.
    fn populate_import_meta(&self, defaults: HostImportMeta) -> Result<HostImportMeta, JsError> {
        Ok(defaults)
    }
}

thread_local! {
    static HOST_LOADER: RefCell<Option<Rc<dyn HostModuleLoader>>> = const { RefCell::new(None) };
    static CURRENT_MODULE_URL: RefCell<Option<Rc<str>>> = const { RefCell::new(None) };
}

/// RAII guard that installs a [`HostModuleLoader`] for the active
/// thread and restores the previous loader (and module URL) on drop.
pub struct HostScope {
    previous_loader: Option<Rc<dyn HostModuleLoader>>,
    previous_url: Option<Rc<str>>,
}

impl HostScope {
    /// Publish `loader` and `module_url` for the duration of the guard.
    ///
    /// Either argument may be `None`.  When `loader` is `None` the
    /// dynamic-import runtime falls back to the host-less rejection
    /// path; when `module_url` is `None` `import.meta.url` reports an
    /// empty string.
    pub fn install(loader: Option<Rc<dyn HostModuleLoader>>, module_url: Option<&str>) -> Self {
        let previous_loader = HOST_LOADER.with(|cell| cell.replace(loader));
        let url_rc = module_url.map(Rc::<str>::from);
        let previous_url = CURRENT_MODULE_URL.with(|cell| cell.replace(url_rc));
        Self {
            previous_loader,
            previous_url,
        }
    }
}

impl Drop for HostScope {
    fn drop(&mut self) {
        HOST_LOADER.with(|cell| *cell.borrow_mut() = self.previous_loader.take());
        CURRENT_MODULE_URL.with(|cell| *cell.borrow_mut() = self.previous_url.take());
    }
}

/// Return the loader currently installed on this thread, if any.
pub fn current_loader() -> Option<Rc<dyn HostModuleLoader>> {
    HOST_LOADER.with(|cell| cell.borrow().clone())
}

/// Return the module URL currently published on this thread, if any.
pub fn current_module_url() -> Option<Rc<str>> {
    CURRENT_MODULE_URL.with(|cell| cell.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::error::ErrorKind;

    struct StubLoader;

    impl HostModuleLoader for StubLoader {
        fn dynamic_import(&self, request: HostDynamicImportRequest) -> Result<(), JsError> {
            request.resolve(JsValue::String(request.specifier().to_string().into()));
            Ok(())
        }

        fn resolve(&self, specifier: &str, _referrer: Option<&str>) -> Result<String, JsError> {
            Err(JsError::new(ErrorKind::TypeError, specifier.to_string()))
        }
    }

    #[test]
    fn test_host_scope_installs_loader_and_url() {
        assert!(current_loader().is_none());
        assert!(current_module_url().is_none());
        let loader: Rc<dyn HostModuleLoader> = Rc::new(StubLoader);
        let _guard = HostScope::install(Some(Rc::clone(&loader)), Some("https://example/m.js"));
        assert!(current_loader().is_some());
        assert_eq!(
            current_module_url().as_deref(),
            Some("https://example/m.js")
        );
    }

    #[test]
    fn test_host_scope_restores_previous_state_on_drop() {
        assert!(current_loader().is_none());
        {
            let loader: Rc<dyn HostModuleLoader> = Rc::new(StubLoader);
            let _guard = HostScope::install(Some(loader), Some("a"));
            assert_eq!(current_module_url().as_deref(), Some("a"));
        }
        assert!(current_loader().is_none());
        assert!(current_module_url().is_none());
    }

    #[test]
    fn test_host_scope_nests() {
        let loader: Rc<dyn HostModuleLoader> = Rc::new(StubLoader);
        let _outer = HostScope::install(Some(Rc::clone(&loader)), Some("outer"));
        {
            let _inner = HostScope::install(Some(Rc::clone(&loader)), Some("inner"));
            assert_eq!(current_module_url().as_deref(), Some("inner"));
        }
        assert_eq!(current_module_url().as_deref(), Some("outer"));
    }
}

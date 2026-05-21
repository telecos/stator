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
use crate::objects::value::JsValue;

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
}

/// Embedder hook for resolving dynamic `import()` and `import.meta.resolve`.
///
/// Implementations are intentionally synchronous: the interpreter
/// fulfils or rejects the returned dynamic-import `Promise` immediately
/// based on the result.  Embedders that need to defer work should keep
/// their own scheduler and resolve their own pending state inside
/// [`Self::dynamic_import`] only once the work is ready.
pub trait HostModuleLoader {
    /// Resolve and evaluate `specifier` on behalf of dynamic `import()`.
    ///
    /// `referrer` is the URL of the module that issued the import, or
    /// `None` if the import came from a script or a module without
    /// origin metadata.  The returned [`JsValue`] is used to fulfil the
    /// dynamic-import `Promise`; an `Err` rejects it.
    fn dynamic_import(&self, specifier: &str, referrer: Option<&str>) -> Result<JsValue, JsError>;

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
        fn dynamic_import(
            &self,
            specifier: &str,
            _referrer: Option<&str>,
        ) -> Result<JsValue, JsError> {
            Ok(JsValue::String(specifier.to_string().into()))
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

/**
 * v8_compat.h — V8-compatible C++ wrapper around the Stator C API (Phase 4).
 *
 * Provides a `v8::` namespace with types and functions that closely mirror the
 * V8 embedding API.  Code written for V8 can be migrated to Stator with
 * minimal changes by including this header instead of `v8.h`.
 *
 * **Simplified model**
 * - `v8::Local<T>` is a plain owning smart-pointer (no scope-linked GC).
 * - `v8::HandleScope` wraps a `StatorHandleScope` for RAII lifetime.
 * - `v8::FunctionCallback` matches `StatorFunctionTemplateCallback` directly;
 *   use `v8::FunctionCallbackInfo` for convenient argument access.
 *
 * Link against `libstator_ffi.a` (or the dynamic variant).
 */

#ifndef V8_COMPAT_H
#define V8_COMPAT_H

#include "stator.h"
#include <cstring>

namespace v8 {

/* -------------------------------------------------------------------------
 * Forward declarations
 * ------------------------------------------------------------------------- */

class Isolate;
class Context;
class Script;
class FunctionTemplate;

/* -------------------------------------------------------------------------
 * FunctionCallbackInfo
 *
 * Thin wrapper around StatorFunctionCallbackInfo.  Passed by pointer to
 * FunctionTemplate callbacks.
 * ------------------------------------------------------------------------- */

/**
 * Provides access to the arguments and isolate for a native function call.
 *
 * An instance of this class is only valid for the duration of the native
 * callback; do not store a pointer to it.
 */
class FunctionCallbackInfo {
    const StatorFunctionCallbackInfo *info_;

public:
    /** Wrap a raw call-info pointer (engine use only). */
    explicit FunctionCallbackInfo(const StatorFunctionCallbackInfo *info)
        : info_(info)
    {}

    /** Return the number of arguments supplied by JavaScript. */
    int Length() const {
        return stator_function_callback_info_length(info_);
    }

    /**
     * Return the argument at position @p idx as a non-owning StatorValue*.
     *
     * The returned pointer is valid only for the duration of the callback.
     * Returns NULL when @p idx is out of range.
     */
    const StatorValue *operator[](int idx) const {
        return stator_function_callback_info_get(info_, idx);
    }

    /** Return the isolate on which the call is executing. */
    StatorIsolate *GetIsolate() const {
        return stator_function_callback_info_get_isolate(info_);
    }
};

/**
 * Native-function callback type used with FunctionTemplate::New().
 *
 * Return a new StatorValue* (the engine takes ownership) or nullptr to return
 * `undefined` to JavaScript.  Use stator_value_new_*() to allocate the
 * return value.
 */
using FunctionCallback = StatorFunctionTemplateCallback;

/* -------------------------------------------------------------------------
 * Isolate
 * ------------------------------------------------------------------------- */

/**
 * An isolated JavaScript engine instance with its own heap and root set.
 *
 * Create with Isolate::New() and release with Dispose().
 */
class Isolate {
    StatorIsolate *raw_;

    Isolate() = default;

public:
    /** Parameters for Isolate::New() (currently unused; reserved for future). */
    struct CreateParams {};

    /**
     * Create a new isolate.
     *
     * The caller is responsible for calling Dispose() when done.
     */
    static Isolate *New(const CreateParams & = CreateParams{}) {
        Isolate *iso = new Isolate();
        iso->raw_    = stator_isolate_new();
        return iso;
    }

    /**
     * Destroy the isolate and release all associated resources.
     *
     * The pointer is invalid after this call.
     */
    void Dispose() {
        stator_isolate_dispose(raw_);
        delete this;
    }

    /** Return the underlying raw pointer (engine / compat-layer use only). */
    StatorIsolate *GetRaw() const { return raw_; }
};

/* -------------------------------------------------------------------------
 * HandleScope
 * ------------------------------------------------------------------------- */

/**
 * RAII wrapper that opens a StatorHandleScope on construction and closes it
 * on destruction.
 *
 * Any StatorValue handles created while this scope is open on @p isolate are
 * owned by the scope and freed automatically when it goes out of scope.
 */
class HandleScope {
    StatorHandleScope *scope_;

public:
    explicit HandleScope(Isolate *isolate)
        : scope_(stator_handle_scope_new(isolate->GetRaw()))
    {}

    ~HandleScope() { stator_handle_scope_close(scope_); }

    /* Non-copyable, non-movable. */
    HandleScope(const HandleScope &)            = delete;
    HandleScope &operator=(const HandleScope &) = delete;
};

/* -------------------------------------------------------------------------
 * Context
 * ------------------------------------------------------------------------- */

/**
 * An execution context associated with an Isolate.
 *
 * Create with Context::New() and destroy with delete (the destructor calls
 * stator_context_destroy()).
 */
class Context {
    StatorContext *raw_;

    Context() = default;

public:
    /**
     * Create a new context on @p isolate.
     *
     * The caller owns the returned pointer and must eventually `delete` it.
     */
    static Context *New(Isolate *isolate) {
        Context *ctx = new Context();
        ctx->raw_    = stator_context_new(isolate->GetRaw());
        return ctx;
    }

    ~Context() { stator_context_destroy(raw_); }

    /* Non-copyable, non-movable. */
    Context(const Context &)            = delete;
    Context &operator=(const Context &) = delete;

    /** Return the underlying raw pointer (engine / compat-layer use only). */
    StatorContext *GetRaw() const { return raw_; }
};

/* -------------------------------------------------------------------------
 * Script
 * ------------------------------------------------------------------------- */

/**
 * A compiled JavaScript script.
 *
 * Compile with Script::Compile().  The Script object owns the compiled
 * bytecode and frees it on destruction.
 */
class Script {
    StatorScript *raw_;

    explicit Script(StatorScript *s) : raw_(s) {}

public:
    ~Script() { stator_script_free(raw_); }

    /* Non-copyable, non-movable. */
    Script(const Script &)            = delete;
    Script &operator=(const Script &) = delete;

    /**
     * Compile @p source and return a new Script*, or nullptr on parse error.
     *
     * The caller owns the returned pointer and must eventually `delete` it.
     *
     * @param ctx     The context to associate with the script.
     * @param source  Null-terminated JavaScript source string.
     */
    static Script *Compile(Context *ctx, const char *source) {
        StatorScript *s =
            stator_script_compile(ctx->GetRaw(), source, std::strlen(source));
        if (!s) return nullptr;
        if (stator_script_get_error(s)) {
            stator_script_free(s);
            return nullptr;
        }
        return new Script(s);
    }

    /**
     * Execute the script and return the completion value.
     *
     * Returns a new StatorValue* that the caller must pass to
     * stator_value_destroy() when finished, or nullptr on an uncaught
     * exception or internal error.
     *
     * @param ctx  The execution context.
     */
    StatorValue *Run(Context *ctx) {
        return stator_script_run(raw_, ctx->GetRaw());
    }
};

/* -------------------------------------------------------------------------
 * FunctionTemplate
 * ------------------------------------------------------------------------- */

/**
 * Associates a native C++ callback with an isolate so that it can be
 * installed as a JavaScript global function.
 *
 * Usage:
 * @code
 *   auto *tmpl = v8::FunctionTemplate::New(iso, my_callback);
 *   tmpl->Install(ctx, "myFunction");   // JS can now call myFunction(...)
 *   delete tmpl;
 * @endcode
 */
class FunctionTemplate {
    StatorFunctionTemplate *tmpl_;
    Isolate                *iso_;

    FunctionTemplate() = default;

public:
    /**
     * Create a new function template wrapping @p callback on @p isolate.
     *
     * The caller owns the returned pointer and must eventually `delete` it.
     *
     * @param isolate   The isolate on which the function will be used.
     * @param callback  The native function to invoke from JavaScript.
     */
    static FunctionTemplate *New(Isolate *isolate, FunctionCallback callback) {
        FunctionTemplate *t = new FunctionTemplate();
        t->iso_  = isolate;
        t->tmpl_ = stator_function_template_new(isolate->GetRaw(), callback);
        return t;
    }

    ~FunctionTemplate() { stator_function_template_destroy(tmpl_); }

    /* Non-copyable, non-movable. */
    FunctionTemplate(const FunctionTemplate &)            = delete;
    FunctionTemplate &operator=(const FunctionTemplate &) = delete;

    /**
     * Produce a callable value and install it as @p name in @p ctx's global
     * environment.
     *
     * After this call, JavaScript running in @p ctx can invoke @p name as a
     * function.
     *
     * An inner HandleScope is opened and closed around the temporary function
     * value so that callers do not have to worry about manual lifetime
     * management regardless of whether an outer scope is active.
     *
     * @param ctx   The context whose global scope should receive the function.
     * @param name  Null-terminated name (may use dot notation, e.g.
     *              "document.getElementById").
     */
    void Install(Context *ctx, const char *name) {
        /* Open an inner scope to own fn so we never double-free it, even
         * when an outer HandleScope is already active on the isolate. */
        StatorHandleScope *inner =
            stator_handle_scope_new(iso_->GetRaw());
        StatorValue *fn =
            stator_function_template_get_function(tmpl_, ctx->GetRaw());
        stator_context_global_set(ctx->GetRaw(), name, fn);
        /* Closing the inner scope frees fn (it is scope-owned). */
        stator_handle_scope_close(inner);
    }
};

} /* namespace v8 */

#endif /* V8_COMPAT_H */

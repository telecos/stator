/**
 * v8_compat.h — C++17 drop-in compatibility shim for the Stator engine.
 *
 * Provides v8:: namespace wrappers around the stator.h C API:
 *   v8::Isolate, v8::Context, v8::HandleScope, v8::Local<T>,
 *   v8::MaybeLocal<T>, v8::Value, v8::String, v8::FunctionTemplate,
 *   v8::Script
 *
 * Intended as a drop-in replacement for v8/include/v8.h for simple
 * embedder use-cases.
 *
 * Memory model
 * ------------
 * All wrapper objects are reference-counted through Local<T> (backed by
 * std::shared_ptr).  A HandleScope is provided for structural compatibility
 * but does **not** interact with Stator's internal handle-scope mechanism;
 * value lifetimes are governed by Local<T> reference counts instead.
 *
 * Specifically:
 *   - Values created via factory functions (String::NewFromUtf8, etc.) are
 *     owned by the returned Local<T> and freed via stator_value_destroy when
 *     the last copy goes out of scope.
 *   - Script objects are freed via stator_script_free on last-copy destruction.
 *   - FunctionTemplate objects are freed via stator_function_template_destroy.
 *   - Context objects created with Context::New are freed via
 *     stator_context_destroy on last-copy destruction.  Non-owning wrappers
 *     (e.g. returned by Isolate::GetCurrentContext) do not call destroy.
 *
 * Limitations
 * -----------
 *   - Persistent<T> / Global<T> are not provided.
 *   - EscapableHandleScope is not provided.
 *   - Only the eight classes listed above are implemented.
 */

#ifndef V8_COMPAT_H
#define V8_COMPAT_H

#include "stator.h"

#include <cstring>
#include <memory>
#include <string>

namespace v8 {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class Context;
class FunctionTemplate;
class Isolate;
class ObjectTemplate;
class Script;
class String;
class Value;
class Number;
class Integer;
class Boolean;
class Object;
class Array;
template <typename T>
class Local;
template <typename T>
class MaybeLocal;

// ---------------------------------------------------------------------------
// Local<T> — reference-counted heap handle
//
// Mirrors the v8::Local<T> API surface for the supported types.  Internally
// backed by std::shared_ptr<T> so copies are cheap and the wrapped object is
// destroyed exactly once.
// ---------------------------------------------------------------------------
template <typename T>
class Local {
public:
    Local() = default;

    /// Construct from a raw heap-allocated pointer; takes ownership.
    explicit Local(T *raw) : ptr_(raw) {}

    bool              IsEmpty()         const noexcept { return !ptr_; }
    T                *operator->()      const noexcept { return ptr_.get(); }
    T                &operator*()       const noexcept { return *ptr_; }
    explicit          operator bool()   const noexcept { return static_cast<bool>(ptr_); }

    /// Static reinterpret-style cast (mirrors v8::Local<T>::Cast).
    template <typename U>
    static Local<T> Cast(Local<U> other) noexcept {
        return Local<T>(std::static_pointer_cast<T>(other.ptr_));
    }

private:
    std::shared_ptr<T> ptr_;

    explicit Local(std::shared_ptr<T> p) noexcept : ptr_(std::move(p)) {}

    template <typename U>
    friend class Local;
    template <typename U>
    friend class MaybeLocal;
};

// ---------------------------------------------------------------------------
// MaybeLocal<T> — an optional Local<T>
// ---------------------------------------------------------------------------
template <typename T>
class MaybeLocal {
public:
    MaybeLocal() = default;

    explicit MaybeLocal(Local<T> local)
        : local_(std::move(local)), has_value_(true) {}

    bool IsEmpty() const noexcept { return !has_value_; }

    bool ToLocal(Local<T> *out) const {
        if (has_value_) {
            *out = local_;
            return true;
        }
        return false;
    }

    /// Return the value, assuming non-empty (caller's responsibility).
    Local<T> ToLocalChecked() const noexcept { return local_; }

private:
    Local<T> local_;
    bool     has_value_{false};
};

// ---------------------------------------------------------------------------
// Value — wraps StatorValue*
//
// When owned == true (the default) the destructor calls stator_value_destroy.
// Pass owned == false for non-owning aliases (e.g. arguments received inside
// a native callback that are owned by the engine for the callback's duration).
// ---------------------------------------------------------------------------
class Value {
public:
    explicit Value(StatorValue *raw, bool owned = true) noexcept
        : raw_(raw), owned_(owned) {}

    Value(const Value &)             = delete;
    Value &operator=(const Value &)  = delete;

    virtual ~Value() noexcept {
        if (owned_ && raw_)
            stator_value_destroy(raw_);
    }

    /// Underlying C handle; valid for the lifetime of this object.
    StatorValue *raw() const noexcept { return raw_; }

    bool IsUndefined() const noexcept { return stator_value_is_undefined(raw_); }
    bool IsNull()      const noexcept { return stator_value_is_null(raw_); }
    bool IsString()    const noexcept { return stator_value_is_string(raw_); }
    bool IsNumber()    const noexcept { return stator_value_is_number(raw_); }
    bool IsBoolean()   const noexcept { return stator_value_is_boolean(raw_); }
    bool IsObject()    const noexcept { return stator_value_is_object(raw_); }
    bool IsFunction()  const noexcept { return stator_value_is_function(raw_); }
    bool IsArray()     const noexcept { return stator_value_is_array(raw_); }

    /// Coerce to double (ECMAScript ToNumber).
    /// The context parameter is accepted for V8 API compatibility but is not
    /// used by the Stator back-end.
    double NumberValue(Local<Context> /*ctx - V8 compat*/) const noexcept {
        return stator_value_as_number(raw_);
    }

    /// Coerce to bool (ECMAScript ToBoolean).
    /// The isolate parameter is accepted for V8 API compatibility but is not
    /// used by the Stator back-end.
    bool BooleanValue(Isolate * /*iso - V8 compat*/) const noexcept {
        return stator_value_to_boolean(raw_);
    }

protected:
    StatorValue *raw_;
    bool         owned_;
};

// ---------------------------------------------------------------------------
// String — inherits Value, adds UTF-8 I/O helpers
// ---------------------------------------------------------------------------
class String : public Value {
public:
    explicit String(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned) {}

    /// Return the number of UTF-8 bytes (excluding any NUL terminator).
    /// The isolate parameter is accepted for V8 API compatibility but is not
    /// used by the Stator back-end.
    int Utf8Length(Isolate * /*isolate - V8 compat*/) const noexcept {
        return static_cast<int>(stator_string_utf8_length(raw_));
    }

    /// Write at most capacity bytes of UTF-8 into buf (not NUL-terminated).
    /// Returns the number of bytes written.  If nchars is non-null it is set
    /// to the same value.
    /// The isolate and flags parameters are accepted for V8 API compatibility
    /// but are not used by the Stator back-end.
    int WriteUtf8(Isolate * /*isolate - V8 compat*/, char *buf, int capacity,
                  int *nchars = nullptr, int /*flags - V8 compat*/ = 0) const noexcept {
        size_t written = 0;
        size_t n = stator_string_write_utf8(
            raw_, buf, static_cast<size_t>(capacity), &written);
        if (nchars)
            *nchars = static_cast<int>(written);
        return static_cast<int>(n);
    }

    /// Create a String from a UTF-8 C string.  Pass length == -1 to infer
    /// the length via strlen.
    static MaybeLocal<String> NewFromUtf8(Isolate *   isolate,
                                           const char *data,
                                           int         length = -1);

    /// RAII helper that copies a Value's string content into an owned buffer.
    ///
    /// Usage:
    ///   v8::String::Utf8Value utf8(isolate, value);
    ///   const char* s = *utf8;
    class Utf8Value {
    public:
        Utf8Value(Isolate *isolate, Local<Value> val);

        const char *operator*() const noexcept { return buf_.c_str(); }
        int         length()    const noexcept {
            return static_cast<int>(buf_.size());
        }

    private:
        std::string buf_;
    };
};

// ---------------------------------------------------------------------------
// Number — inherits Value, wraps a numeric StatorValue
// ---------------------------------------------------------------------------
class Number : public Value {
public:
    explicit Number(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned) {}

    /// Return the double value.
    double Value_() const noexcept { return stator_value_as_number(raw_); }

    /// Create a Number from a double.
    static Local<Number> New(Isolate *isolate, double value);
};

// ---------------------------------------------------------------------------
// Integer — inherits Value, wraps an integer StatorValue
// ---------------------------------------------------------------------------
class Integer : public Value {
public:
    explicit Integer(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned) {}

    /// Return the int64 value.
    int64_t Value_() const noexcept {
        return static_cast<int64_t>(stator_value_as_number(raw_));
    }

    /// Create an Integer from a signed 32-bit value.
    static Local<Integer> New(Isolate *isolate, int32_t value);

    /// Create an Integer from an unsigned 32-bit value.
    static Local<Integer> NewFromUnsigned(Isolate *isolate, uint32_t value);
};

// ---------------------------------------------------------------------------
// Boolean — inherits Value
// ---------------------------------------------------------------------------
class Boolean : public Value {
public:
    explicit Boolean(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned) {}

    bool Value_() const noexcept { return stator_value_to_boolean(raw_); }

    /// Create a Boolean value.
    static Local<Boolean> New(Isolate *isolate, bool value);
};

// ---------------------------------------------------------------------------
// Object — wraps StatorObject* with V8-compatible API
// ---------------------------------------------------------------------------
class Object : public Value {
public:
    explicit Object(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned), obj_(nullptr), obj_owned_(false) {}

    Object(StatorValue *raw, StatorObject *obj, bool owned, bool obj_owned) noexcept
        : Value(raw, owned), obj_(obj), obj_owned_(obj_owned) {}

    ~Object() noexcept override {
        if (obj_owned_ && obj_)
            stator_object_destroy(obj_);
    }

    /// Create a new empty Object.
    static Local<Object> New(Isolate *isolate);

    /// Set a named property.  Returns true on success.
    bool Set(Local<Context> /*ctx*/, Local<Value> key, Local<Value> value);

    /// Get a named property.
    MaybeLocal<Value> Get(Local<Context> /*ctx*/, Local<Value> key);

    /// Get the number of internal fields (always 0 for plain objects).
    int InternalFieldCount() const noexcept { return 0; }

    StatorObject *obj() const noexcept { return obj_; }

private:
    StatorObject *obj_;
    bool          obj_owned_;
};

// ---------------------------------------------------------------------------
// Array — inherits Object
// ---------------------------------------------------------------------------
class Array : public Value {
public:
    explicit Array(StatorValue *raw, bool owned = true) noexcept
        : Value(raw, owned), length_(0) {}

    Array(StatorValue *raw, uint32_t len, bool owned = true) noexcept
        : Value(raw, owned), length_(len) {}

    /// Create a new Array with the given length.
    static Local<Array> New(Isolate *isolate, int length = 0);

    /// Return the array's length.
    uint32_t Length() const noexcept { return length_; }

private:
    uint32_t length_;
};

// ---------------------------------------------------------------------------
// Isolate — wraps StatorIsolate*
//
// Not copyable or movable (matches v8::Isolate semantics).
// Create with Isolate::New(); destroy with isolate->Dispose().
// ---------------------------------------------------------------------------
class Isolate {
public:
    struct CreateParams {};

    /// Create a new Isolate.  The caller must eventually call Dispose().
    /// Returns nullptr on allocation failure.
    static Isolate *New(const CreateParams & = CreateParams()) {
        StatorIsolate *raw = stator_isolate_new();
        if (!raw)
            return nullptr;
        return new Isolate(raw);
    }

    /// Destroy the Isolate and release all associated resources.
    void Dispose() {
        stator_isolate_dispose(raw_);
        delete this;
    }

    void Enter() noexcept { stator_isolate_enter(raw_); }
    void Exit()  noexcept { stator_isolate_exit(raw_); }

    void  SetData(uint32_t slot, void *data) noexcept {
        stator_isolate_set_data(raw_, slot, data);
    }
    void *GetData(uint32_t slot) const noexcept {
        return stator_isolate_get_data(raw_, slot);
    }

    void ThrowException(Local<Value> exception) noexcept {
        stator_isolate_throw_exception(raw_, exception->raw());
    }

    /// Return the context most recently entered on this isolate, or an empty
    /// Local if none is current.  The returned Local is non-owning.
    Local<Context> GetCurrentContext();

    /// Underlying C handle; valid for the lifetime of this object.
    StatorIsolate *raw() const noexcept { return raw_; }

    Isolate(const Isolate &)            = delete;
    Isolate &operator=(const Isolate &) = delete;

    /// RAII scope: calls Enter on construction and Exit on destruction.
    class Scope {
    public:
        explicit Scope(Isolate *iso) noexcept : iso_(iso) { iso_->Enter(); }
        ~Scope() noexcept { iso_->Exit(); }

        Scope(const Scope &)            = delete;
        Scope &operator=(const Scope &) = delete;

    private:
        Isolate *iso_;
    };

private:
    explicit Isolate(StatorIsolate *raw) noexcept : raw_(raw) {}
    StatorIsolate *raw_;
};

// ---------------------------------------------------------------------------
// Context — wraps StatorContext*
//
// Owned contexts (created with Context::New) call stator_context_destroy when
// the last Local<Context> copy goes out of scope.  Non-owning aliases (e.g.
// returned by Isolate::GetCurrentContext) do not.
// ---------------------------------------------------------------------------
class Context {
public:
    /// Create a new Context associated with isolate.  Returns an empty
    /// Local<Context> if isolate is null.
    static Local<Context> New(Isolate *isolate) {
        StatorContext *raw = stator_context_new(isolate->raw());
        if (!raw)
            return Local<Context>();
        return Local<Context>(new Context(raw, isolate, /*owned=*/true));
    }

    void Enter() noexcept { stator_context_enter(raw_); }
    void Exit()  noexcept { stator_context_exit(raw_); }

    Isolate        *GetIsolate() const noexcept { return isolate_; }
    StatorContext  *raw()        const noexcept { return raw_; }

    Context(const Context &)            = delete;
    Context &operator=(const Context &) = delete;

    ~Context() noexcept {
        if (owned_ && raw_)
            stator_context_destroy(raw_);
    }

    /// RAII scope: calls Enter on construction and Exit on destruction.
    class Scope {
    public:
        explicit Scope(Local<Context> ctx) noexcept : ctx_(std::move(ctx)) {
            ctx_->Enter();
        }
        ~Scope() noexcept { ctx_->Exit(); }

        Scope(const Scope &)            = delete;
        Scope &operator=(const Scope &) = delete;

    private:
        Local<Context> ctx_;
    };

private:
    Context(StatorContext *raw, Isolate *isolate, bool owned) noexcept
        : raw_(raw), isolate_(isolate), owned_(owned) {}

    StatorContext *raw_;
    Isolate       *isolate_;
    bool           owned_;

    friend class Isolate;
};

// Deferred definition of Isolate::GetCurrentContext (needs complete Context).
inline Local<Context> Isolate::GetCurrentContext() {
    StatorContext *ctx = stator_isolate_get_current_context(raw_);
    if (!ctx)
        return Local<Context>();
    // Non-owning: caller does not destroy the context.
    return Local<Context>(new Context(ctx, this, /*owned=*/false));
}

// ---------------------------------------------------------------------------
// HandleScope — structural API compatibility
//
// In this shim value lifetimes are managed by Local<T> reference counts
// rather than by Stator's internal handle scope, so HandleScope is a no-op.
// It is provided so that code written against the v8 API compiles unchanged.
// The Isolate* constructor parameter is accepted for V8 API compatibility
// but is not used.
// ---------------------------------------------------------------------------
class HandleScope {
public:
    /// Accepted for V8 API compatibility; does not open a Stator handle scope.
    explicit HandleScope(Isolate * /*isolate - V8 compat*/) noexcept {}
    ~HandleScope() noexcept = default;

    HandleScope(const HandleScope &)            = delete;
    HandleScope &operator=(const HandleScope &) = delete;
};

// ---------------------------------------------------------------------------
// EscapableHandleScope — allows one value to escape to the outer scope
//
// Like HandleScope, this is structural compatibility.  In the reference-counted
// Local<T> model used by this shim, escaping is a no-op — the Local<T> copy
// already extends the value's lifetime.
// ---------------------------------------------------------------------------
class EscapableHandleScope {
public:
    explicit EscapableHandleScope(Isolate * /*isolate - V8 compat*/) noexcept {}
    ~EscapableHandleScope() noexcept = default;

    /// "Escape" a value from this scope.  In the reference-counted model this
    /// simply returns the same Local<T> unchanged.
    template <typename T>
    Local<T> Escape(Local<T> value) noexcept { return value; }

    EscapableHandleScope(const EscapableHandleScope &)            = delete;
    EscapableHandleScope &operator=(const EscapableHandleScope &) = delete;
};

// ---------------------------------------------------------------------------
// TryCatch — exception handling across the FFI boundary
//
// Mirrors v8::TryCatch RAII semantics.  While a TryCatch is active, any
// pending exception thrown on the isolate is captured and can be inspected.
// ---------------------------------------------------------------------------
class TryCatch {
public:
    explicit TryCatch(Isolate *isolate) noexcept
        : isolate_(isolate),
          tc_(stator_try_catch_new(isolate ? isolate->raw() : nullptr)) {}

    ~TryCatch() noexcept {
        if (tc_)
            stator_try_catch_destroy(tc_);
    }

    /// Return true if an exception was caught.
    bool HasCaught() noexcept {
        return tc_ ? stator_try_catch_has_caught(tc_) : false;
    }

    /// Return the caught exception value, or an empty Local if none.
    Local<Value> Exception() noexcept {
        if (!tc_)
            return Local<Value>();
        StatorValue *exc = stator_try_catch_exception(tc_);
        if (!exc)
            return Local<Value>();
        // Non-owning: the exception is owned by the TryCatch scope.
        return Local<Value>(new Value(exc, /*owned=*/false));
    }

    /// Reset the TryCatch, clearing any caught exception.
    void Reset() noexcept {
        if (tc_)
            stator_try_catch_reset(tc_);
    }

    TryCatch(const TryCatch &)            = delete;
    TryCatch &operator=(const TryCatch &) = delete;

private:
    Isolate       *isolate_;
    StatorTryCatch *tc_;
};

// ---------------------------------------------------------------------------
// ObjectTemplate — wraps StatorObjectTemplate*
//
// Describes the shape of objects that will be created from it.  Used by Blink
// to set up DOM prototype objects.
// ---------------------------------------------------------------------------
class ObjectTemplate {
public:
    /// Create a new, empty ObjectTemplate.
    static Local<ObjectTemplate> New(Isolate *isolate) {
        StatorObjectTemplate *raw =
            stator_object_template_new(isolate ? isolate->raw() : nullptr);
        if (!raw)
            return Local<ObjectTemplate>();
        return Local<ObjectTemplate>(new ObjectTemplate(raw));
    }

    /// Set a named property on the template.
    void Set(Local<String> name, Local<Value> value) {
        if (!name || !value || !raw_)
            return;
        // Extract the key as a UTF-8 string.
        size_t len = stator_string_utf8_length(name->raw());
        std::string key(len + 1, '\0');
        if (len > 0)
            stator_string_write_utf8(name->raw(), &key[0], len, nullptr);
        key[len] = '\0';
        stator_object_template_set(raw_, key.c_str(), value->raw());
    }

    /// Set the number of internal fields.
    void SetInternalFieldCount(int count) noexcept {
        if (raw_)
            stator_object_template_set_internal_field_count(raw_, count);
    }

    /// Get the internal field count.
    int InternalFieldCount() const noexcept {
        return raw_ ? stator_object_template_internal_field_count(raw_) : 0;
    }

    /// Create a new object instance from this template.
    MaybeLocal<Object> NewInstance(Local<Context> context);

    StatorObjectTemplate *raw() const noexcept { return raw_; }

    ObjectTemplate(const ObjectTemplate &)            = delete;
    ObjectTemplate &operator=(const ObjectTemplate &) = delete;

    ~ObjectTemplate() noexcept {
        if (raw_)
            stator_object_template_destroy(raw_);
    }

private:
    explicit ObjectTemplate(StatorObjectTemplate *raw) noexcept : raw_(raw) {}
    StatorObjectTemplate *raw_;
};

// ---------------------------------------------------------------------------
// FunctionTemplate — wraps StatorFunctionTemplate*
//
// Freed via stator_function_template_destroy when the last Local copy is
// released.
// ---------------------------------------------------------------------------
class FunctionTemplate {
public:
    using Callback = StatorFunctionTemplateCallback;

    /// Create a new FunctionTemplate with an optional native callback.
    static Local<FunctionTemplate> New(Isolate  *isolate,
                                       Callback  callback = nullptr) {
        StatorFunctionTemplate *raw =
            stator_function_template_new(isolate->raw(), callback);
        if (!raw)
            return Local<FunctionTemplate>();
        return Local<FunctionTemplate>(new FunctionTemplate(raw));
    }

    /// Obtain a callable Value from this template, suitable for installing
    /// into a context via stator_context_global_set.
    Local<Value> GetFunction(Local<Context> context) const {
        StatorValue *v =
            stator_function_template_get_function(raw_, context->raw());
        if (!v)
            return Local<Value>();
        return Local<Value>(new Value(v, /*owned=*/true));
    }

    StatorFunctionTemplate *raw() const noexcept { return raw_; }

    FunctionTemplate(const FunctionTemplate &)            = delete;
    FunctionTemplate &operator=(const FunctionTemplate &) = delete;

    ~FunctionTemplate() noexcept {
        if (raw_)
            stator_function_template_destroy(raw_);
    }

private:
    explicit FunctionTemplate(StatorFunctionTemplate *raw) noexcept : raw_(raw) {}
    StatorFunctionTemplate *raw_;
};

// ---------------------------------------------------------------------------
// Script — wraps StatorScript*
//
// Freed via stator_script_free when the last Local<Script> copy is released.
// ---------------------------------------------------------------------------
class Script {
public:
    /// Compile JavaScript source text.  Returns empty on compile error.
    static MaybeLocal<Script> Compile(Local<Context> context,
                                      Local<String>  source) {
        if (!context || !source)
            return MaybeLocal<Script>();

        // Copy the source string into a temporary buffer for stator_script_compile.
        size_t len = stator_string_utf8_length(source->raw());
        std::string buf(len, '\0');
        if (len > 0)
            stator_string_write_utf8(source->raw(), &buf[0], len, nullptr);

        StatorScript *s =
            stator_script_compile(context->raw(), buf.data(), len);
        if (!s)
            return MaybeLocal<Script>();

        if (stator_script_get_error(s) != nullptr) {
            stator_script_free(s);
            return MaybeLocal<Script>();
        }
        return MaybeLocal<Script>(Local<Script>(new Script(s)));
    }

    /// Execute the compiled script in context.  Returns empty on uncaught
    /// exception or when the script produces no result.
    MaybeLocal<Value> Run(Local<Context> context) const {
        StatorValue *result = stator_script_run(raw_, context->raw());
        if (!result)
            return MaybeLocal<Value>();
        return MaybeLocal<Value>(Local<Value>(new Value(result, /*owned=*/true)));
    }

    StatorScript *raw() const noexcept { return raw_; }

    Script(const Script &)            = delete;
    Script &operator=(const Script &) = delete;

    ~Script() noexcept {
        if (raw_)
            stator_script_free(raw_);
    }

private:
    explicit Script(StatorScript *raw) noexcept : raw_(raw) {}
    StatorScript *raw_;
};

// ---------------------------------------------------------------------------
// Deferred String method definitions (require complete Isolate)
// ---------------------------------------------------------------------------

inline MaybeLocal<String> String::NewFromUtf8(Isolate *   isolate,
                                               const char *data,
                                               int         length) {
    size_t len = (length < 0) ? std::strlen(data)
                              : static_cast<size_t>(length);
    StatorValue *v = stator_string_new_from_utf8(isolate->raw(), data, len);
    if (!v)
        return MaybeLocal<String>();
    return MaybeLocal<String>(Local<String>(new String(v, /*owned=*/true)));
}

inline String::Utf8Value::Utf8Value(Isolate * /*isolate*/, Local<Value> val) {
    if (!val || !val->raw())
        return;
    size_t len = stator_string_utf8_length(val->raw());
    if (len == 0)
        return;
    buf_.resize(len);
    stator_string_write_utf8(val->raw(), &buf_[0], len, nullptr);
}

// ---------------------------------------------------------------------------
// Deferred typed-value factory definitions (require complete Isolate)
// ---------------------------------------------------------------------------

inline Local<Number> Number::New(Isolate *isolate, double value) {
    StatorValue *v = stator_number_new(isolate->raw(), value);
    if (!v)
        return Local<Number>();
    return Local<Number>(new Number(v, /*owned=*/true));
}

inline Local<Integer> Integer::New(Isolate *isolate, int32_t value) {
    StatorValue *v = stator_integer_new(isolate->raw(), static_cast<int64_t>(value));
    if (!v)
        return Local<Integer>();
    return Local<Integer>(new Integer(v, /*owned=*/true));
}

inline Local<Integer> Integer::NewFromUnsigned(Isolate *isolate, uint32_t value) {
    StatorValue *v = stator_number_new(isolate->raw(), static_cast<double>(value));
    if (!v)
        return Local<Integer>();
    return Local<Integer>(new Integer(v, /*owned=*/true));
}

inline Local<Boolean> Boolean::New(Isolate *isolate, bool value) {
    StatorValue *v = stator_boolean_new(isolate->raw(), value);
    if (!v)
        return Local<Boolean>();
    return Local<Boolean>(new Boolean(v, /*owned=*/true));
}

inline Local<Object> Object::New(Isolate *isolate) {
    StatorValue *v = stator_value_new_object(isolate->raw());
    if (!v)
        return Local<Object>();
    return Local<Object>(new Object(v, /*owned=*/true));
}

inline bool Object::Set(Local<Context> /*ctx*/, Local<Value> key, Local<Value> value) {
    // Object::Set requires a StatorObject; without one, this is a stub.
    // Blink bindings typically call this on objects returned from
    // ObjectTemplate::NewInstance which have a backing StatorObject.
    (void)key;
    (void)value;
    return false;
}

inline MaybeLocal<Value> Object::Get(Local<Context> /*ctx*/, Local<Value> key) {
    (void)key;
    return MaybeLocal<Value>();
}

inline Local<Array> Array::New(Isolate *isolate, int length) {
    StatorValue *v = stator_value_new_array_tag(isolate->raw());
    if (!v)
        return Local<Array>();
    return Local<Array>(new Array(v, static_cast<uint32_t>(length >= 0 ? length : 0),
                                  /*owned=*/true));
}

// ---------------------------------------------------------------------------
// Deferred ObjectTemplate::NewInstance (requires complete Object)
// ---------------------------------------------------------------------------

inline MaybeLocal<Object> ObjectTemplate::NewInstance(Local<Context> /*context*/) {
    if (!raw_)
        return MaybeLocal<Object>();
    StatorObject *obj = stator_object_template_new_instance(raw_);
    if (!obj)
        return MaybeLocal<Object>();
    // Create an Object wrapper with a backing StatorObject but no StatorValue.
    // The StatorObject is owned by this wrapper.
    return MaybeLocal<Object>(
        Local<Object>(new Object(nullptr, obj, /*owned=*/false, /*obj_owned=*/true)));
}

} // namespace v8

#endif // V8_COMPAT_H

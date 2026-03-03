// gin_js_engine_test.cc — Smoke test that the selected JS engine links and
// executes JavaScript correctly.
//
// When built with js_engine="stator" (GIN_ENGINE_STATOR=1) the test exercises
// Stator through the v8_compat.h shim's v8:: namespace API.
//
// When built with js_engine="v8" (GIN_ENGINE_V8=1) the test exercises V8
// directly through the same v8:: API surface, so the source is identical for
// both backends.
//
// Acceptance: `gn gen out/stator --args='js_engine="stator"'` followed by
// `ninja -C out/stator gin_js_engine_test` must succeed, and the resulting
// binary must exit with status 0.

#if defined(GIN_ENGINE_STATOR)
#include "v8_compat.h"   // stator_bridge — v8:: over stator.h
#else
#include "v8.h"
#include "libplatform/libplatform.h"
#endif

#include <cassert>
#include <cstdio>

int main() {
    v8::Isolate::CreateParams params;
    v8::Isolate *isolate = v8::Isolate::New(params);
    assert(isolate != nullptr);

    {
        v8::Isolate::Scope iso_scope(isolate);
        v8::HandleScope  handle_scope(isolate);

        auto context = v8::Context::New(isolate);
        assert(!context.IsEmpty());

        v8::Context::Scope ctx_scope(context);

        // Compile and run a trivial arithmetic expression.
        auto maybe_src =
            v8::String::NewFromUtf8(isolate, "1 + 2");
        assert(!maybe_src.IsEmpty());
        auto src = maybe_src.ToLocalChecked();

        auto maybe_script = v8::Script::Compile(context, src);
        assert(!maybe_script.IsEmpty());

        auto maybe_result = maybe_script.ToLocalChecked()->Run(context);
        assert(!maybe_result.IsEmpty());

        // v8_compat.h's NumberValue returns a plain double (unlike real V8's
        // Maybe<double>), so no further unwrapping is needed here.
        double val = maybe_result.ToLocalChecked()->NumberValue(context);
        assert(val == 3.0);

        std::printf(
            "gin_js_engine_test PASSED  engine=%s  1+2=%.0f\n",
#if defined(GIN_ENGINE_STATOR)
            "stator",
#else
            "v8",
#endif
            val);
    }

    isolate->Dispose();
    return 0;
}

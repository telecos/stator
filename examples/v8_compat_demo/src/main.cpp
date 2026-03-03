/**
 * v8_compat_demo — Demonstrates the v8_compat.h compatibility shim.
 *
 * This example uses the v8:: namespace API provided by v8_compat.h to run
 * JavaScript through the Stator engine.  It mirrors the typical V8 embedder
 * pattern:
 *
 *   1. Isolate::New        — create an engine instance
 *   2. HandleScope         — establish a local handle scope
 *   3. Context::New        — create a JavaScript execution context
 *   4. String::NewFromUtf8 — wrap a C string as a v8::String
 *   5. Script::Compile     — compile the source to bytecode
 *   6. script->Run         — execute the bytecode
 *   7. result->NumberValue — extract the numeric result
 *   8. isolate->Dispose    — tear everything down
 *
 * Expected output (exit code 0):
 *   21 + 21 = 42
 *   hello world
 *   fibonacci(10) = 55
 *   v8_compat_demo: OK
 */

#include "v8_compat.h"

#include <cassert>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Helper: run a JS snippet and return its numeric result (-1 on error).
// ---------------------------------------------------------------------------
static double run_number(v8::Isolate *isolate,
                         v8::Local<v8::Context> context,
                         const char *source)
{
    v8::Local<v8::String> src =
        v8::String::NewFromUtf8(isolate, source).ToLocalChecked();

    v8::MaybeLocal<v8::Script> maybe = v8::Script::Compile(context, src);
    if (maybe.IsEmpty())
        return -1.0;

    v8::MaybeLocal<v8::Value> result = maybe.ToLocalChecked()->Run(context);
    if (result.IsEmpty())
        return -1.0;

    return result.ToLocalChecked()->NumberValue(context);
}

// ---------------------------------------------------------------------------
// Helper: run a JS snippet and return its string result (empty on error).
// ---------------------------------------------------------------------------
static std::string run_string(v8::Isolate *isolate,
                               v8::Local<v8::Context> context,
                               const char *source)
{
    v8::Local<v8::String> src =
        v8::String::NewFromUtf8(isolate, source).ToLocalChecked();

    v8::MaybeLocal<v8::Script> maybe = v8::Script::Compile(context, src);
    if (maybe.IsEmpty())
        return {};

    v8::MaybeLocal<v8::Value> result = maybe.ToLocalChecked()->Run(context);
    if (result.IsEmpty())
        return {};

    v8::Local<v8::Value> val = result.ToLocalChecked();
    v8::String::Utf8Value utf8(isolate, val);
    return std::string(*utf8, static_cast<std::size_t>(utf8.length()));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    /* 1. Create the isolate. */
    v8::Isolate::CreateParams params;
    v8::Isolate *isolate = v8::Isolate::New(params);
    assert(isolate != nullptr);

    {
        /* 2. Enter the isolate and establish a handle scope. */
        v8::Isolate::Scope  isolate_scope(isolate);
        v8::HandleScope     handle_scope(isolate);

        /* 3. Create a JavaScript execution context. */
        v8::Local<v8::Context> context = v8::Context::New(isolate);
        assert(!context.IsEmpty());

        v8::Context::Scope context_scope(context);

        /* ── Test 1: simple arithmetic ─────────────────────────────────── */
        double result = run_number(isolate, context, "21 + 21");
        assert(result == 42.0);
        std::printf("21 + 21 = %g\n", result);

        /* ── Test 2: string result ──────────────────────────────────────── */
        std::string s = run_string(isolate, context, "'hello' + ' ' + 'world'");
        assert(s == "hello world");
        std::printf("%s\n", s.c_str());

        /* ── Test 3: function definition + call ─────────────────────────── */
        double fib = run_number(isolate, context,
            "function fibonacci(n) {"
            "  if (n <= 1) return n;"
            "  return fibonacci(n - 1) + fibonacci(n - 2);"
            "}"
            "fibonacci(10);");
        assert(fib == 55.0);
        std::printf("fibonacci(10) = %g\n", fib);

        /* ── Test 4: Local<T> copy semantics ────────────────────────────── */
        {
            v8::Local<v8::String> a =
                v8::String::NewFromUtf8(isolate, "copy test").ToLocalChecked();
            v8::Local<v8::String> b = a; // shared ownership
            assert(!a.IsEmpty() && !b.IsEmpty());
            assert(a->raw() == b->raw()); // same underlying StatorValue*
        }

        /* ── Test 5: compile error returns empty ────────────────────────── */
        {
            v8::Local<v8::String> bad =
                v8::String::NewFromUtf8(isolate, "var = ;").ToLocalChecked();
            v8::MaybeLocal<v8::Script> maybe =
                v8::Script::Compile(context, bad);
            assert(maybe.IsEmpty());
        }

        /* ── Test 6: Value type predicates ─────────────────────────────── */
        {
            double n = run_number(isolate, context, "42");
            assert(n == 42.0);

            v8::Local<v8::String> num_src =
                v8::String::NewFromUtf8(isolate, "42").ToLocalChecked();
            v8::Local<v8::Script> scr =
                v8::Script::Compile(context, num_src).ToLocalChecked();
            v8::Local<v8::Value> val = scr->Run(context).ToLocalChecked();
            assert(val->IsNumber());
            assert(!val->IsString());
        }

        /* ── Test 7: Isolate::Scope / Context::Scope ────────────────────── */
        {
            v8::Local<v8::Context> ctx2 = v8::Context::New(isolate);
            v8::Context::Scope     cs2(ctx2);
            double r = run_number(isolate, ctx2, "7 * 6");
            assert(r == 42.0);
        }
    }

    /* 8. Tear down the isolate. */
    isolate->Dispose();

    std::puts("v8_compat_demo: OK");
    return 0;
}

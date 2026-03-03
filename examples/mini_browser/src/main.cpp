/**
 * mini_browser — Demonstrates Stator JavaScript parsing, bytecode
 * compilation, and end-to-end execution (Phase 3).
 *
 * This sample exercises the Phase 1 object model, the Phase 2 compilation
 * pipeline, and the new Phase 3 execution layer:
 *
 *   Phase 1 — GC / object model:
 *     1. Create an isolate and a JS context.
 *     2. Allocate a number value, a string value, and an object with
 *        properties.
 *     3. Print their types and values via FFI.
 *     4. Trigger GC while handles are held — objects survive.
 *     5. Release all handles, trigger GC — objects are reclaimed.
 *     6. Print final heap statistics.
 *
 *   Phase 2 — JavaScript parsing + bytecode compilation:
 *     7. Simulate extracting inline <script> content from an HTML page.
 *     8. Compile each script fragment with stator_script_compile().
 *     9. Report success (bytecode count) or failure (error message).
 *    10. Dump the bytecode listing for a simple arithmetic script.
 *    11. Demonstrate error reporting for malformed JavaScript.
 *
 *   Phase 3 — JavaScript execution (new):
 *    12. Register a native console.log function.
 *    13. Execute inline scripts — console.log actually prints output.
 *    14. Execute a Fibonacci and factorial benchmark.
 *    15. Demonstrate error handling: uncaught exceptions are reported.
 *    16. Show execution timing.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stator.h"

/* -------------------------------------------------------------------------
 * Global context pointer (used by native callbacks)
 * ------------------------------------------------------------------------- */

static StatorContext *g_ctx = nullptr;

/* -------------------------------------------------------------------------
 * Native console.log implementation
 * ------------------------------------------------------------------------- */

/**
 * Native implementation of console.log.
 *
 * Prints all arguments separated by spaces, followed by a newline.
 * String values are printed without surrounding quotes; numbers use %g.
 */
static StatorValue *native_console_log(StatorContext * /*ctx*/,
                                       const StatorValue *const *args,
                                       int argc)
{
    for (int i = 0; i < argc; ++i) {
        if (i > 0) std::fputs(" ", stdout);
        if (!args[i]) {
            std::fputs("undefined", stdout);
            continue;
        }
        const char *type = stator_value_type(args[i]);
        if (std::strcmp(type, "string") == 0) {
            std::fputs(stator_value_as_string(args[i]), stdout);
        } else {
            /* Number or other: use to_string_utf8 for a consistent format. */
            char buf[64];
            int n = stator_value_to_string_utf8(args[i], buf, sizeof(buf));
            if (n >= 0)
                std::fwrite(buf, 1, static_cast<std::size_t>(n), stdout);
        }
    }
    std::fputc('\n', stdout);
    return nullptr; /* returns undefined */
}

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/**
 * Compile one JavaScript snippet, print a pass/fail summary, optionally dump
 * bytecodes, and then free the script handle.
 *
 * @param ctx        The active Stator context (may be NULL).
 * @param source     Null-terminated JavaScript source string.
 * @param dump_code  When non-zero, print the bytecode listing on success.
 */
static void compile_and_report(StatorContext *ctx,
                                const char   *source,
                                int           dump_code)
{
    std::printf("[tab] compiling: %s\n", source);

    StatorScript *script = stator_script_compile(ctx, source,
                                                 std::strlen(source));
    if (!script) {
        std::printf("[tab] ERROR: internal allocation failure\n");
        return;
    }

    const char *err = stator_script_get_error(script);
    if (err) {
        std::printf("[tab] ERROR: %s\n", err);
    } else {
        size_t count = stator_script_bytecode_count(script);
        std::printf("[tab] OK -- %zu bytecode(s) generated\n", count);
        if (dump_code) {
            std::printf("[tab] bytecodes:\n");
            // Flush C's stdio buffer before calling into Rust so that all
            // preceding printf output is written to fd 1 before the Rust
            // println! calls inside stator_bytecode_dump write to the same fd.
            std::fflush(nullptr);
            stator_bytecode_dump(script);
        }
    }

    stator_script_free(script);
}

/**
 * Compile and execute a JavaScript snippet.
 *
 * Prints "[tab] executing: <source>" then runs the script.  If the script
 * produces a non-undefined result it is printed as "[tab] result: <val>".
 * Uncaught exceptions are printed as "[tab] UNCAUGHT: <msg>".
 * When @p time_it is non-zero, the elapsed time is appended to the result
 * line.
 *
 * @param ctx      The active Stator context.
 * @param source   Null-terminated JavaScript source string.
 * @param time_it  When non-zero, measure and print execution time.
 */
static void execute_script(StatorContext *ctx,
                            const char   *source,
                            int           time_it)
{
    std::printf("[tab] executing: %s\n", source);

    StatorScript *script = stator_script_compile(ctx, source,
                                                 std::strlen(source));
    if (!script) {
        std::printf("[tab] ERROR: internal allocation failure\n");
        return;
    }

    const char *err = stator_script_get_error(script);
    if (err) {
        std::printf("[tab] ERROR (compile): %s\n", err);
        stator_script_free(script);
        return;
    }

    /* Run the script and measure time. */
    auto t0 = std::chrono::high_resolution_clock::now();
    StatorValue *result = stator_script_run(script, ctx);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!result) {
        /* NULL means either undefined or an uncaught exception.
         * We can't distinguish here without richer error propagation,
         * so treat NULL as "uncaught exception / void". */
        std::printf("[tab] UNCAUGHT: (exception)\n");
    } else {
        const char *type = stator_value_type(result);
        /* Only print the result if it is not undefined. */
        if (std::strcmp(type, "undefined") != 0) {
            char buf[256] = {};
            stator_value_to_string_utf8(result, buf, sizeof(buf));
            if (time_it) {
                std::printf("[tab] result: %s  (%.2fms)\n", buf, elapsed_ms);
            } else {
                std::printf("[tab] result: %s\n", buf);
            }
        } else if (time_it) {
            std::printf("[tab] result: undefined  (%.2fms)\n", elapsed_ms);
        }
        stator_value_destroy(result);
    }

    stator_script_free(script);
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main() {
    StatorIsolate *isolate = stator_isolate_create();
    if (!isolate) {
        std::fprintf(stderr, "ERROR: failed to create isolate\n");
        return 1;
    }

    /* ── Phase 1: GC / object model ─────────────────────────────────────── */

    /* 1. Create a context ------------------------------------------------- */
    StatorContext *ctx = stator_context_new(isolate);
    g_ctx = ctx;
    std::printf("[tab] created context\n");

    /* 2. Allocate values -------------------------------------------------- */
    StatorValue  *num = stator_value_new_number(isolate, 42.0);
    StatorValue  *str = stator_value_new_string(isolate, "hello", 5);
    StatorObject *obj = stator_object_new(isolate);

    /* Set object properties x = 1, y = 2 (temporaries destroyed right away). */
    {
        StatorValue *xv = stator_value_new_number(isolate, 1.0);
        StatorValue *yv = stator_value_new_number(isolate, 2.0);
        stator_object_set(obj, "x", xv);
        stator_object_set(obj, "y", yv);
        stator_value_destroy(xv);
        stator_value_destroy(yv);
    }

    /* 3. Print types and values ------------------------------------------- */
    {
        StatorValue *xv = stator_object_get(obj, "x");
        StatorValue *yv = stator_object_get(obj, "y");
        std::printf(
            "[tab] allocated: %s(%g), %s(\"%s\"), object{x: %g, y: %g}\n",
            stator_value_type(num),   stator_value_as_number(num),
            stator_value_type(str),   stator_value_as_string(str),
            stator_value_as_number(xv), stator_value_as_number(yv));
        stator_value_destroy(xv);
        stator_value_destroy(yv);
    }

    /* 4. GC with handles held — 3 live objects survive ------------------- */
    stator_gc_collect(isolate);
    std::printf("[tab] GC: %zu objects survived (held by handles)\n",
                stator_live_object_count(isolate));

    /* 5. Release handles, GC — all objects reclaimed --------------------- */
    stator_value_destroy(num);
    stator_value_destroy(str);
    stator_object_destroy(obj);
    stator_gc_collect(isolate);
    std::printf("[tab] released handles, GC: %zu objects (all reclaimed)\n",
                stator_live_object_count(isolate));

    /* 6. Print heap stats ------------------------------------------------- */
    std::printf("[tab] heap: %zu bytes used / %zu bytes capacity\n",
                stator_heap_used(isolate), stator_heap_capacity(isolate));

    /* ── Phase 2: JavaScript parsing + bytecode compilation ─────────────── */

    std::printf("\n[tab] --- Phase 2: JavaScript parsing demo ---\n\n");

    /*
     * Simulate inline <script> fragments extracted from an HTML document:
     *
     *   <script>var x = 1 + 2;</script>
     *   <script>var = ;</script>          <!-- intentional syntax error -->
     */

    /* 7 & 8 & 9 & 10. Compile a valid arithmetic expression and dump its
     *                  bytecodes.  --------------------------------------- */
    compile_and_report(ctx, "var x = 1 + 2;", /* dump_code= */ 1);

    std::printf("\n");

    /* 11. Demonstrate error reporting for malformed JavaScript. ----------- */
    compile_and_report(ctx, "var = ;", /* dump_code= */ 0);

    /* ── Phase 3: JavaScript execution ──────────────────────────────────── */

    std::printf("\n[tab] --- Phase 3: JavaScript execution demo ---\n\n");

    /* 12. Register native console.log ------------------------------------ */
    stator_register_native_function(ctx, "console.log", native_console_log);
    std::printf("[tab] registered native console.log\n\n");

    /* 13. Execute inline scripts from HTML pages -------------------------- */
    execute_script(ctx, "console.log('page A loaded');",      /* time= */ 0);
    execute_script(ctx, "const x = 1 + 2; console.log('x =', x);",
                   /* time= */ 0);
    std::printf("\n");

    /* 14. Fibonacci benchmark -------------------------------------------- */
    execute_script(ctx,
        "function fibonacci(n) {"
        "  if (n <= 1) return n;"
        "  return fibonacci(n-1) + fibonacci(n-2);"
        "}"
        "fibonacci(10);",
        /* time= */ 1);

    execute_script(ctx,
        "function factorial(n) {"
        "  if (n <= 1) return 1;"
        "  return n * factorial(n - 1);"
        "}"
        "factorial(10);",
        /* time= */ 1);
    std::printf("\n");

    /* 15. Error handling demo -------------------------------------------- */
    execute_script(ctx,
        "var caught = false;"
        "try { var x = 1; } catch(e) { caught = true; }"
        "caught;",
        /* time= */ 0);

    /* Uncaught exception: throw propagates to the host as a NULL result. */
    execute_script(ctx, "throw 'uncaught error';", /* time= */ 0);

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    stator_context_destroy(ctx);
    stator_isolate_destroy(isolate);
    return 0;
}

/**
 * mini_browser — Demonstrates Stator JavaScript parsing, bytecode
 * compilation, and end-to-end execution (Phase 9).
 *
 * This sample exercises the Phase 1 object model, the Phase 2 compilation
 * pipeline, Phase 3 execution, Phase 4 v8-compatible API, Phase 5 JIT,
 * Phase 8 WebAssembly, and the new Phase 9 Chrome DevTools Protocol layer:
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
 *   Phase 3 — JavaScript execution:
 *    12. Register a native console.log function.
 *    13. Execute inline scripts — console.log actually prints output.
 *    14. Execute a Fibonacci and factorial benchmark.
 *    15. Demonstrate error handling: uncaught exceptions are reported.
 *    16. Show execution timing.
 *
 *   Phase 9 — Chrome DevTools Protocol inspector (new):
 *    17. Start a CDP WebSocket server on port 9229 and print the DevTools URL.
 *    18. Set a breakpoint at source line 3 of a debug script.
 *    19. Run the script; execution pauses at the breakpoint.
 *    20. Inspect global variables at the pause point.
 *    21. Resume execution to completion.
 *    22. Pass --inspect to keep the server alive for real DevTools connections.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stator.h"
#include "v8_compat.h"

/* -------------------------------------------------------------------------
 * Global context pointer (used by native callbacks)
 * ------------------------------------------------------------------------- */

static StatorContext *g_ctx = nullptr;

/* -------------------------------------------------------------------------
 * Phase 4 globals: mock DOM state for getElementById demo
 * ------------------------------------------------------------------------- */

/** Last element ID requested via document.getElementById() from JS. */
static char g_dom_element_id[64] = {};

/**
 * Native implementation of document.getElementById (Phase 4 v8-compat demo).
 *
 * Stores the requested element ID in g_dom_element_id and returns a plain
 * object that JavaScript can set properties on (e.g. el.textContent).
 */
static StatorValue *get_element_by_id_cb(const StatorFunctionCallbackInfo *info)
{
    int argc = stator_function_callback_info_length(info);
    if (argc > 0) {
        const StatorValue *arg = stator_function_callback_info_get(info, 0);
        const char *id = stator_value_as_string(arg);
        std::strncpy(g_dom_element_id, id,
                     sizeof(g_dom_element_id) - 1);
        g_dom_element_id[sizeof(g_dom_element_id) - 1] = '\0';
    }
    /* Return an empty plain object so JS can set textContent on it. */
    StatorIsolate *iso = stator_function_callback_info_get_isolate(info);
    return stator_value_new_object(iso);
}

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

int main(int argc, char **argv) {
    /* Check for --inspect flag */
    bool inspect_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--inspect") == 0) {
            inspect_mode = true;
        }
    }

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

    /* ── Phase 4: v8-compatible API demo ─────────────────────────────────── */

    std::printf("\n[tab] --- Phase 4: v8-compatible API demo ---\n\n");
    std::printf("[tab] using v8-compatible API\n");

    /* Create isolate and context through the v8:: compatibility layer.
     * The HandleScope is kept in a nested block so it is destroyed (and its
     * owned values freed) before v8iso->Dispose() tears down the isolate. */
    v8::Isolate *v8iso = v8::Isolate::New();
    {
        v8::HandleScope v8hs(v8iso);
        v8::Context    *v8ctx = v8::Context::New(v8iso);
        std::printf("[tab] created v8::Isolate, v8::Context\n");

        /* Register document.getElementById via FunctionTemplate::New(). */
        v8::FunctionTemplate *elem_tmpl =
            v8::FunctionTemplate::New(v8iso, get_element_by_id_cb);
        elem_tmpl->Install(v8ctx, "document.getElementById");
        std::printf("[tab] registered document.getElementById\n");

        /* Execute the mock DOM script.
         *
         * el = document.getElementById('title')  → C++ stores "title" in
         *                                           g_dom_element_id and returns
         *                                           an empty plain object.
         * el.textContent = 'Hello!'              → stores 'Hello!' on the object;
         *                                           the assignment expression is
         *                                           the program's completion value.
         *
         * After stator_script_run() returns 'Hello!' we combine it with the
         * captured element ID to print the DOM update message, demonstrating
         * the full JS → C++ → JS callback round-trip.
         */
        const char *dom_source =
            "var el = document.getElementById('title');"
            " el.textContent = 'Hello!';";
        std::printf("[tab] executing: %s\n", dom_source);

        g_dom_element_id[0] = '\0';
        v8::Script *dom_script = v8::Script::Compile(v8ctx, dom_source);
        if (dom_script) {
            StatorValue *result = dom_script->Run(v8ctx);
            if (result && g_dom_element_id[0] != '\0' &&
                std::strcmp(stator_value_type(result), "string") == 0) {
                const char *text = stator_value_as_string(result);
                std::printf("[tab] DOM update: #%s.textContent = '%s'\n",
                            g_dom_element_id, text);
            }
            if (result) stator_value_destroy(result);
            delete dom_script;
        }

        /* Release Phase 4 resources (context and template before scope). */
        delete elem_tmpl;
        delete v8ctx;
        /* v8hs goes out of scope here, closing the handle scope cleanly. */
    }
    v8iso->Dispose();

    /* ── Phase 5: JIT compilation and tiering demo ───────────────────────────── */

    std::printf("\n[tab] --- Phase 5: JIT compilation / tiering demo ---\n\n");

    /*
     * Compile a script that defines and calls sum(1..1000000).
     * The inner loop iterates 1 000 000 times, so the OSR back-edge counter
     * exceeds OSR_LOOP_THRESHOLD (1 000) within the very first call.  After
     * that call the baseline JIT compiler caches machine code for sum(), and
     * every subsequent call executes via native code.
     */
    const char *sum_src =
        "function sum(n) {"
        "  var s = 0;"
        "  var i = 1;"
        "  while (i <= n) { s = s + i; i = i + 1; }"
        "  return s;"
        "}"
        "sum(1000000);";

    StatorScript *sum_script =
        stator_script_compile(ctx, sum_src, std::strlen(sum_src));
    if (!sum_script || stator_script_get_error(sum_script)) {
        std::printf("[tab] ERROR: failed to compile sum script\n");
        if (sum_script) stator_script_free(sum_script);
    } else {
        /* Snapshot stats before the first run. */
        StatorCompilationStats stats_before = {};
        stator_isolate_get_stats(isolate, &stats_before);

        /* Iteration 1 — interpreted (triggers OSR → JIT compiles sum). */
        auto t0 = std::chrono::high_resolution_clock::now();
        StatorValue *r1 = stator_script_run(sum_script, ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms1 =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("[tab] running sum(1..1000000) — iteration 1 (interpreted)"
                    ": %.1fms\n", ms1);

        /* Check whether tiering fired during the first run. */
        StatorCompilationStats stats_after = {};
        stator_isolate_get_stats(isolate, &stats_after);
        uint32_t new_fns =
            stats_after.jit_functions_compiled -
            stats_before.jit_functions_compiled;
        size_t new_bytes =
            stats_after.jit_code_bytes - stats_before.jit_code_bytes;
        bool jit_available = (new_fns > 0);
        if (jit_available) {
            std::printf("[tab] [tier-up] sum() compiled to baseline JIT"
                        " (code: %zu bytes)\n", new_bytes);
        }

        /* Iteration 2 — executes via JIT on supported platforms. */
        auto t2 = std::chrono::high_resolution_clock::now();
        StatorValue *r2 = stator_script_run(sum_script, ctx);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms2 =
            std::chrono::duration<double, std::milli>(t3 - t2).count();

        const char *tier2 = jit_available ? "baseline JIT" : "interpreted";
        std::printf("[tab] running sum(1..1000000) — iteration 2 (%s)"
                    ": %.1fms\n", tier2, ms2);

        if (jit_available && ms2 > 0.0) {
            std::printf("[tab] speedup: %.1fx\n", ms1 / ms2);
        }

        StatorCompilationStats stats_final = {};
        stator_isolate_get_stats(isolate, &stats_final);
        std::printf("[tab] stats: %u function(s) JIT-compiled,"
                    " %zu bytes machine code\n",
                    stats_final.jit_functions_compiled,
                    stats_final.jit_code_bytes);

        if (r1) stator_value_destroy(r1);
        if (r2) stator_value_destroy(r2);
        stator_script_free(sum_script);
    }

    /* ── Phase 8: WebAssembly execution demo ────────────────────────────── */

    std::printf("\n[tab] --- Phase 8: WebAssembly execution demo ---\n\n");

    /*
     * add.wasm — pre-compiled binary for:
     *
     *   (module
     *     (func (export "add") (param i32 i32) (result i32)
     *       local.get 0
     *       local.get 1
     *       i32.add))
     *
     * Generated with: wat2wasm (or the equivalent wasmtime encoding).
     */
    static const uint8_t add_wasm[] = {
        0x00, 0x61, 0x73, 0x6d, /* magic: \0asm               */
        0x01, 0x00, 0x00, 0x00, /* version: 1                  */
        /* type section: (i32 i32) -> i32 */
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        /* function section: func[0] uses type[0] */
        0x03, 0x02, 0x01, 0x00,
        /* export section: "add" -> func 0 */
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        /* code section: body of func 0 */
        0x0a, 0x09, 0x01, 0x07, 0x00,
        0x20, 0x00, /* local.get 0 */
        0x20, 0x01, /* local.get 1 */
        0x6a,       /* i32.add     */
        0x0b        /* end         */
    };
    static const std::size_t add_wasm_size = sizeof(add_wasm);

    std::printf("[tab] loaded add.wasm (%zu bytes)\n", add_wasm_size);

    /* Compile the .wasm binary. */
    StatorWasmModule *wasm_module =
        stator_wasm_compile(isolate, add_wasm, add_wasm_size);
    if (!wasm_module) {
        std::printf("[tab] ERROR: stator_wasm_compile failed\n");
    } else {
        /* Instantiate the module. */
        StatorWasmInstance *wasm_instance =
            stator_wasm_instantiate(wasm_module, ctx, nullptr);
        if (!wasm_instance) {
            std::printf("[tab] ERROR: stator_wasm_instantiate failed\n");
        } else {
            std::printf("[tab] WebAssembly.instantiate: OK\n");

            /* ── JS call path ─────────────────────────────────────────────
             * Register a native JS function "wasmAdd" backed by the live
             * Wasm instance so that JavaScript can call it.  We use a
             * global pointer to pass the instance to the callback.
             */
            static StatorWasmInstance *g_wasm_instance = nullptr;
            static StatorIsolate      *g_wasm_isolate   = nullptr;
            g_wasm_instance = wasm_instance;
            g_wasm_isolate  = isolate;

            /* Register wasmAdd(a, b) as a native function. */
            stator_register_native_function(
                ctx, "wasmAdd",
                [](StatorContext * /*c*/,
                   const StatorValue *const *args, int argc) -> StatorValue * {
                    /* Forward JS arguments to the Wasm "add" export. */
                    const StatorValue *wasm_args[2] = {
                        argc > 0 ? args[0] : nullptr,
                        argc > 1 ? args[1] : nullptr
                    };
                    std::size_t nargs = static_cast<std::size_t>(
                        (argc < 2) ? argc : 2);
                    return stator_wasm_instance_call(
                        g_wasm_instance, g_wasm_isolate,
                        "add", wasm_args, nargs);
                });

            /* Execute the JS call: wasmAdd(3, 4) and print the result. */
            const char *js_src = "wasmAdd(3, 4);";
            StatorScript *js_script =
                stator_script_compile(ctx, js_src, std::strlen(js_src));
            if (js_script && !stator_script_get_error(js_script)) {
                StatorValue *js_result = stator_script_run(js_script, ctx);
                if (js_result) {
                    char buf[32] = {};
                    stator_value_to_string_utf8(js_result, buf, sizeof(buf));
                    std::printf("[tab] JS: add(3, 4) = %s\n", buf);
                    stator_value_destroy(js_result);
                }
            }
            if (js_script) stator_script_free(js_script);

            /* ── C++ direct call path ─────────────────────────────────────
             * Call the Wasm "add" export directly from C++ via the FFI,
             * without going through the JS interpreter.
             */
            StatorValue *arg3 = stator_value_new_number(isolate, 3.0);
            StatorValue *arg4 = stator_value_new_number(isolate, 4.0);
            const StatorValue *cpp_args[2] = {arg3, arg4};
            StatorValue *cpp_result =
                stator_wasm_instance_call(wasm_instance, isolate,
                                          "add", cpp_args, 2);
            if (cpp_result) {
                char buf[32] = {};
                stator_value_to_string_utf8(cpp_result, buf, sizeof(buf));
                std::printf("[tab] C++ direct: stator_wasm_call(add, 3, 4) = %s\n",
                            buf);
                stator_value_destroy(cpp_result);
            }
            stator_value_destroy(arg3);
            stator_value_destroy(arg4);

            stator_wasm_instance_destroy(wasm_instance);
        }
        stator_wasm_module_destroy(wasm_module);
    }

    /* ── Phase 9: Chrome DevTools Protocol inspector demo ───────────────────── */

    std::printf("\n[tab] --- Phase 9: Chrome DevTools Protocol demo ---\n\n");

    /*
     * Start a CDP WebSocket server on port 9229 (the Node.js / V8 default).
     * In --inspect mode the server runs in a background thread so that a real
     * Chrome DevTools frontend can connect while the process stays alive.
     * Otherwise we start it, print the connection URL, and then let it go
     * (the process will exit shortly).
     */
    uint16_t cdp_port = 9229;
    StatorCdpServer *cdp_server = stator_cdp_server_create(cdp_port);

    if (!cdp_server) {
        /* Port 9229 may already be in use; fall back to an OS-assigned port. */
        cdp_server = stator_cdp_server_create(0);
    }

    if (cdp_server) {
        cdp_port = stator_cdp_server_local_port(cdp_server);
        std::printf("[tab] inspector listening on ws://127.0.0.1:%u\n",
                    static_cast<unsigned>(cdp_port));
        std::printf("[tab] DevTools URL: devtools://devtools/bundled/js_app.html?ws=127.0.0.1:%u\n",
                    static_cast<unsigned>(cdp_port));

        if (inspect_mode) {
            /* Transfer server ownership to a background thread. */
            stator_cdp_server_run_background(cdp_server);
            cdp_server = nullptr; /* consumed — do not destroy */
            std::printf("[tab] (CDP server running in background — "
                        "connect with Chrome DevTools)\n");
        } else {
            stator_cdp_server_destroy(cdp_server);
            cdp_server = nullptr;
        }
    } else {
        std::printf("[tab] WARNING: could not start CDP server\n");
    }

    /*
     * Debugger demo — set a breakpoint, run a script, inspect variables,
     * then resume execution.
     *
     * Script (uses implicit globals so that the debugger can read them back
     * from the shared global environment):
     *   Line 1:  x = 42;
     *   Line 2:  y = 'hello';
     *   Line 3:  x;           <-- breakpoint
     *
     * When the breakpoint fires we print both globals, then resume.
     */
    const char *debug_src =
        "x = 42;\n"
        "y = 'hello';\n"
        "x;";

    StatorScript *debug_script =
        stator_script_compile(ctx, debug_src, std::strlen(debug_src));

    if (debug_script && !stator_script_get_error(debug_script)) {
        StatorDebugSession *sess =
            stator_debug_session_create(debug_script, ctx);

        if (sess) {
            /* Install a breakpoint at source line 3. */
            stator_debug_session_set_breakpoint_at_line(sess, 3);

            /* Run until the breakpoint is hit. */
            bool paused = stator_debug_session_run(sess);

            if (paused) {
                uint32_t line = stator_debug_session_pause_line(sess);
                std::printf("[tab] executing script (paused at breakpoint"
                            " line %u)\n", line);

                /* Inspect global variables. */
                char vx[64] = {};
                char vy[64] = {};
                stator_debug_session_get_global_string(sess, "x", vx, sizeof(vx));
                stator_debug_session_get_global_string(sess, "y", vy, sizeof(vy));
                std::printf("[tab] variables: x = %s, y = '%s'\n", vx, vy);

                /* Resume to completion. */
                stator_debug_session_resume(sess);
                std::printf("[tab] resumed execution\n");
            } else {
                /* Breakpoint mapping may have failed — still print something. */
                std::printf("[tab] executing script (no breakpoint hit)\n");
            }

            /* Optionally print the final result. */
            StatorValue *dbg_result = stator_debug_session_result(sess);
            if (dbg_result) {
                char buf[64] = {};
                stator_value_to_string_utf8(dbg_result, buf, sizeof(buf));
                std::printf("[tab] script result: %s\n", buf);
                stator_value_destroy(dbg_result);
            }

            stator_debug_session_destroy(sess);
        }
    }
    if (debug_script) stator_script_free(debug_script);

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    stator_context_destroy(ctx);
    stator_isolate_destroy(isolate);
    return 0;
}

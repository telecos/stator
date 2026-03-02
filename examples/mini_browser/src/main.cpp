/**
 * mini_browser — Demonstrates Stator JavaScript parsing and bytecode
 * compilation (Phase 2).
 *
 * This sample exercises both the Phase 1 object model and the Phase 2
 * compilation pipeline:
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
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stator.h"

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
            stator_bytecode_dump(script);
        }
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

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    stator_context_destroy(ctx);
    stator_isolate_destroy(isolate);
    return 0;
}

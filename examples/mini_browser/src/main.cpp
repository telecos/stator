/**
 * mini_browser — Demonstrates Stator object allocation and GC.
 *
 * This sample exercises the Phase 1 GC and object model:
 *
 *   1. Create an isolate and a JS context.
 *   2. Allocate a number value, a string value, and an object with properties.
 *   3. Print their types and values via FFI.
 *   4. Trigger GC while handles are held — objects survive.
 *   5. Release all handles, trigger GC — objects are reclaimed.
 *   6. Print final heap statistics.
 */

#include <cstdio>
#include <cstdlib>

#include "stator.h"

int main() {
    StatorIsolate *isolate = stator_isolate_create();
    if (!isolate) {
        std::fprintf(stderr, "ERROR: failed to create isolate\n");
        return 1;
    }

    /* 1. Create a context ------------------------------------------------- */
    /* The context is the execution environment for JS code.  In this Phase 1
     * demo it is not yet wired to a script evaluator, but creating it via the
     * FFI ensures the lifecycle API (create / destroy) is exercised. */
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

    stator_context_destroy(ctx);
    stator_isolate_destroy(isolate);
    return 0;
}


/**
 * stator.h — Public C API for the Stator JavaScript engine.
 *
 * This header mirrors the symbols exported by `crates/stator_ffi`.  Link
 * against `libstator_ffi.a` (static) or `libstator_ffi.so` / `stator_ffi.dll`
 * (dynamic) to use this API.
 *
 * Ownership model
 * ---------------
 * All opaque handle types are heap-allocated by Stator and returned as raw
 * pointers.  The caller is responsible for passing them to the corresponding
 * `_destroy` function when they are no longer needed.  Double-freeing or using
 * a handle after destruction is undefined behaviour.
 */

#ifndef STATOR_H
#define STATOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Opaque types
 * ------------------------------------------------------------------------- */

/** An isolated JavaScript engine instance with its own heap and root set. */
typedef struct StatorIsolate StatorIsolate;

/* -------------------------------------------------------------------------
 * Isolate lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a new Stator isolate.
 *
 * Returns a pointer to a freshly initialised isolate.  The caller owns the
 * returned pointer and must eventually pass it to stator_isolate_destroy().
 *
 * Returns NULL on allocation failure (extremely rare in practice).
 */
StatorIsolate *stator_isolate_create(void);

/**
 * Destroy an isolate and release all associated resources.
 *
 * @param isolate  A non-NULL pointer previously returned by
 *                 stator_isolate_create().  Must not be used after this call.
 */
void stator_isolate_destroy(StatorIsolate *isolate);

/* -------------------------------------------------------------------------
 * Heap / GC
 * ------------------------------------------------------------------------- */

/**
 * Trigger a minor (young-generation) garbage collection on the isolate heap.
 *
 * Safe to call at any point when no JavaScript code is currently executing
 * on the isolate.
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 */
void stator_isolate_gc(StatorIsolate *isolate);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STATOR_H */

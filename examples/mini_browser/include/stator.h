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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Opaque types
 * ------------------------------------------------------------------------- */

/** An isolated JavaScript engine instance with its own heap and root set. */
typedef struct StatorIsolate StatorIsolate;

/** An execution context associated with an isolate. */
typedef struct StatorContext StatorContext;

/** A handle to a JavaScript value (number or string). */
typedef struct StatorValue StatorValue;

/** A handle to a JavaScript object with named properties. */
typedef struct StatorObject StatorObject;

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

/**
 * Trigger a minor (young-generation) GC on the isolate heap.
 *
 * Preferred spelling for embedders using the Phase 1 object model.
 * Equivalent to stator_isolate_gc().
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 */
void stator_gc_collect(StatorIsolate *isolate);

/**
 * Return the number of live embedder-owned value/object handles on the
 * isolate.  After a GC cycle this reflects how many objects survive because
 * they are still held by live handles.
 *
 * @param isolate  A valid isolate pointer, or NULL (returns 0).
 */
size_t stator_live_object_count(const StatorIsolate *isolate);

/**
 * Return bytes currently allocated in the young-generation from-space.
 * This is 0 immediately after a GC cycle.
 *
 * @param isolate  A valid isolate pointer, or NULL (returns 0).
 */
size_t stator_heap_used(const StatorIsolate *isolate);

/**
 * Return the total capacity (both halves) of the young-generation semi-space
 * in bytes.
 *
 * @param isolate  A valid isolate pointer, or NULL (returns 0).
 */
size_t stator_heap_capacity(const StatorIsolate *isolate);

/* -------------------------------------------------------------------------
 * Context lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a new execution context associated with `isolate`.
 *
 * Returns NULL if `isolate` is NULL.  The caller must eventually pass the
 * returned pointer to stator_context_destroy().
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 */
StatorContext *stator_context_new(StatorIsolate *isolate);

/**
 * Destroy a context previously created with stator_context_new().
 *
 * @param ctx  A non-NULL pointer returned by stator_context_new().
 */
void stator_context_destroy(StatorContext *ctx);

/* -------------------------------------------------------------------------
 * Value lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a new number value.
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 * @param val      The double-precision floating-point number to wrap.
 * @return         A new StatorValue*, or NULL on failure.
 */
StatorValue *stator_value_new_number(StatorIsolate *isolate, double val);

/**
 * Create a new string value from a buffer of `len` bytes.
 *
 * The buffer need not be null-terminated; `len` bytes are copied.
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 * @param data     Pointer to the first byte of the string data.
 * @param len      Number of bytes to copy.
 * @return         A new StatorValue*, or NULL on failure.
 */
StatorValue *stator_value_new_string(StatorIsolate *isolate, const char *data, size_t len);

/**
 * Destroy a value and decrement the isolate's live-object counter.
 *
 * @param val  A non-NULL pointer returned by stator_value_new_number(),
 *             stator_value_new_string(), or stator_object_get().
 */
void stator_value_destroy(StatorValue *val);

/**
 * Return a static C string describing the type of `val`: "number" or
 * "string".  Returns "undefined" when `val` is NULL.
 *
 * @param val  A valid StatorValue pointer, or NULL.
 */
const char *stator_value_type(const StatorValue *val);

/**
 * Return the numeric value stored in `val`, or NaN if `val` is NULL or not
 * a number.
 *
 * @param val  A valid StatorValue pointer, or NULL.
 */
double stator_value_as_number(const StatorValue *val);

/**
 * Return a null-terminated C string for the string stored in `val`.
 *
 * Returns a pointer to an empty string when `val` is NULL or not a string.
 * The pointer is valid only as long as `val` is alive.
 *
 * @param val  A valid StatorValue pointer, or NULL.
 */
const char *stator_value_as_string(const StatorValue *val);

/* -------------------------------------------------------------------------
 * Object lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a new, empty JavaScript object.
 *
 * @param isolate  A valid, non-NULL isolate pointer.
 * @return         A new StatorObject*, or NULL on failure.
 */
StatorObject *stator_object_new(StatorIsolate *isolate);

/**
 * Destroy an object and decrement the isolate's live-object counter.
 *
 * @param obj  A non-NULL pointer returned by stator_object_new().
 */
void stator_object_destroy(StatorObject *obj);

/**
 * Set (or overwrite) the named property `key` on `obj` to `val`.
 *
 * The value is copied into the object; the caller retains ownership of `val`.
 *
 * @param obj  A valid, non-NULL StatorObject pointer.
 * @param key  A null-terminated property name string.
 * @param val  A valid, non-NULL StatorValue pointer.
 */
void stator_object_set(StatorObject *obj, const char *key, const StatorValue *val);

/**
 * Get the named property `key` from `obj` as a new StatorValue.
 *
 * Returns NULL if `obj` or `key` is NULL or the property does not exist.
 * The caller owns the returned pointer and must pass it to
 * stator_value_destroy().
 *
 * @param obj  A valid, non-NULL StatorObject pointer.
 * @param key  A null-terminated property name string.
 */
StatorValue *stator_object_get(const StatorObject *obj, const char *key);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STATOR_H */

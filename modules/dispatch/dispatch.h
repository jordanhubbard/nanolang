/* dispatch.h — NanoLang dispatch module C API
 *
 * Thin wrappers over libdispatch. Exposed to NanoLang via opaque types.
 * All callback parameters use C function pointer type `void (*)(void)` so
 * that NanoLang's generated `FnType_N` typedefs (which are C function
 * pointers) can be passed directly without any Blocks extension on the
 * caller side.  dispatch.c wraps them in Clang Blocks before handing off
 * to the GCD API (requires -fblocks, macOS/Linux libdispatch).
 */
#ifndef NL_DISPATCH_H
#define NL_DISPATCH_H

#include <stdint.h>

/* NanoLang opaque type typedefs — must match transpiler.nano's is_opaque_type list */
#ifndef NL_DISPATCH_TYPES_DECLARED
#define NL_DISPATCH_TYPES_DECLARED
typedef void* NlQueue;
typedef void* NlQueueGroup;
#endif

/* C function pointer type used for all callbacks */
typedef void (*nl_dispatch_fn)(void);

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Queue lifecycle ---------- */

/** Create a serial queue (FIFO, one task at a time). */
void* nl_queue_serial(const char* label);

/** Create a concurrent queue (tasks may run in parallel). */
void* nl_queue_concurrent(const char* label);

/** Destroy (drain + release) a queue. */
void nl_queue_destroy(void* q);

/* ---------- Dispatch primitives ---------- */

/** Enqueue fn asynchronously (fire-and-forget). */
void nl_queue_async(void* q, nl_dispatch_fn fn);

/** Enqueue fn synchronously — blocks caller until fn completes. */
void nl_queue_sync(void* q, nl_dispatch_fn fn);

/** Enqueue a barrier async. */
void nl_queue_barrier_async(void* q, nl_dispatch_fn fn);

/** Enqueue after a delay (nanoseconds). */
void nl_queue_after_ns(void* q, int64_t ns, nl_dispatch_fn fn);

/* ---------- Group operations ---------- */

void* nl_group_create(void);
void  nl_group_destroy(void* g);

/** Submit fn to q, tracked by group g. */
void nl_group_async(void* g, void* q, nl_dispatch_fn fn);

/** Run fn on q when all tasks in group have completed. */
void nl_group_notify(void* g, void* q, nl_dispatch_fn fn);

/** Block caller until all tasks in group complete (or timeout_ns elapses).
 *  Returns 0 on success, non-zero on timeout. */
int nl_group_wait_ns(void* g, int64_t timeout_ns);

#ifdef __cplusplus
}
#endif

#endif /* NL_DISPATCH_H */

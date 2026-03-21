/* dispatch.h — NanoLang dispatch module C API
 *
 * Thin wrappers over libdispatch. Exposed to NanoLang via dispatch.nano opaque types.
 * All handles are malloc'd structs containing a single dispatch_queue_t / dispatch_group_t.
 */
#ifndef NL_DISPATCH_H
#define NL_DISPATCH_H

#ifdef __APPLE__
#  include <dispatch/dispatch.h>
#else
#  include <dispatch/dispatch.h>
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Opaque handles ---------- */

typedef struct { dispatch_queue_t q; }  NlQueue;
typedef struct { dispatch_group_t g; }  NlQueueGroup;

/* ---------- Queue lifecycle ---------- */

/** Create a serial queue (FIFO, one task at a time). */
NlQueue* nl_queue_serial(const char* label);

/** Create a concurrent queue (tasks may run in parallel). */
NlQueue* nl_queue_concurrent(const char* label);

/** Destroy (drain + release) a queue created with nl_queue_serial/concurrent.
 *  Must not be called from inside a block submitted to this queue. */
void nl_queue_destroy(NlQueue* q);

/* ---------- Dispatch primitives ---------- */

/** Enqueue a no-arg NanoLang callback (void fn(void*) / block) asynchronously.
 *  The NanoLang transpiler calls this with a Clang block: ^{ ... }            */
void nl_queue_async(NlQueue* q, dispatch_block_t blk);

/** Enqueue synchronously — blocks caller until block completes. */
void nl_queue_sync(NlQueue* q, dispatch_block_t blk);

/** Enqueue a barrier async: waits for all prior items, runs alone, then lets later items proceed. */
void nl_queue_barrier_async(NlQueue* q, dispatch_block_t blk);

/** Enqueue after a delay (nanoseconds). */
void nl_queue_after_ns(NlQueue* q, int64_t ns, dispatch_block_t blk);

/* ---------- Group operations ---------- */

NlQueueGroup* nl_group_create(void);
void          nl_group_destroy(NlQueueGroup* g);

/** Submit blk to q, tracked by group g. */
void nl_group_async(NlQueueGroup* g, NlQueue* q, dispatch_block_t blk);

/** Run blk on q when all tasks in group have completed. */
void nl_group_notify(NlQueueGroup* g, NlQueue* q, dispatch_block_t blk);

/** Block caller until all tasks in group complete (or timeout_ns elapses).
 *  Returns 0 on success, non-zero on timeout. */
int nl_group_wait_ns(NlQueueGroup* g, int64_t timeout_ns);

#ifdef __cplusplus
}
#endif

#endif /* NL_DISPATCH_H */

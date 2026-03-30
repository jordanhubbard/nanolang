/* dispatch.c — NanoLang dispatch module implementation
 *
 * On Apple platforms: thin wrappers over libdispatch (Grand Central Dispatch).
 * The public API accepts plain C function pointers so NanoLang's generated
 * FnType_N typedefs integrate without Blocks on the caller side.  Internally
 * each wrapper converts to a Clang Block before passing to GCD (-fblocks is
 * applied via cflags_macos in module.json).
 *
 * On non-Apple platforms: stub implementations that compile cleanly but abort
 * at runtime (GCD / libdispatch is not available without special installation).
 */
#include "dispatch.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef __APPLE__

#include <dispatch/dispatch.h>

/* Internal struct types — heap-allocated, passed as void* to NanoLang */
typedef struct { dispatch_queue_t q; }  _NlQueueInternal;
typedef struct { dispatch_group_t g; }  _NlGroupInternal;

/* ── Queue lifecycle ─────────────────────────────────────────────────────── */

void* nl_queue_serial(const char* label) {
    _NlQueueInternal* q = (_NlQueueInternal*)malloc(sizeof(_NlQueueInternal));
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_SERIAL);
    return (void*)q;
}

void* nl_queue_concurrent(const char* label) {
    _NlQueueInternal* q = (_NlQueueInternal*)malloc(sizeof(_NlQueueInternal));
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_CONCURRENT);
    return (void*)q;
}

void nl_queue_destroy(void* qv) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    if (!q) return;
    dispatch_barrier_sync(q->q, ^{});
    dispatch_release(q->q);
    free(q);
}

/* ── Dispatch primitives ─────────────────────────────────────────────────── */

void nl_queue_async(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_async(q->q, ^{ fp(); });
}

void nl_queue_sync(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_sync(q->q, ^{ fp(); });
}

void nl_queue_barrier_async(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_barrier_async(q->q, ^{ fp(); });
}

void nl_queue_after_ns(void* qv, int64_t ns, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_time_t when = dispatch_time(DISPATCH_TIME_NOW, ns);
    dispatch_after(when, q->q, ^{ fp(); });
}

/* ── Group operations ────────────────────────────────────────────────────── */

void* nl_group_create(void) {
    _NlGroupInternal* g = (_NlGroupInternal*)malloc(sizeof(_NlGroupInternal));
    g->g = dispatch_group_create();
    return (void*)g;
}

void nl_group_destroy(void* gv) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    if (!g) return;
    dispatch_release(g->g);
    free(g);
}

void nl_group_async(void* gv, void* qv, nl_dispatch_fn fp) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_group_async(g->g, q->q, ^{ fp(); });
}

void nl_group_notify(void* gv, void* qv, nl_dispatch_fn fp) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_group_notify(g->g, q->q, ^{ fp(); });
}

int nl_group_wait_ns(void* gv, int64_t timeout_ns) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    dispatch_time_t t = (timeout_ns < 0)
        ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
    return (int)dispatch_group_wait(g->g, t);
}

#else /* non-Apple stub implementations */

#define DISPATCH_UNAVAILABLE(fn) \
    fprintf(stderr, "dispatch: " #fn " is not available on this platform (macOS only)\n"); \
    abort();

void* nl_queue_serial(const char* label)            { (void)label; DISPATCH_UNAVAILABLE(nl_queue_serial); return NULL; }
void* nl_queue_concurrent(const char* label)        { (void)label; DISPATCH_UNAVAILABLE(nl_queue_concurrent); return NULL; }
void  nl_queue_destroy(void* qv)                    { (void)qv; DISPATCH_UNAVAILABLE(nl_queue_destroy); }
void  nl_queue_async(void* qv, nl_dispatch_fn fp)   { (void)qv; (void)fp; DISPATCH_UNAVAILABLE(nl_queue_async); }
void  nl_queue_sync(void* qv, nl_dispatch_fn fp)    { (void)qv; (void)fp; DISPATCH_UNAVAILABLE(nl_queue_sync); }
void  nl_queue_barrier_async(void* qv, nl_dispatch_fn fp) { (void)qv; (void)fp; DISPATCH_UNAVAILABLE(nl_queue_barrier_async); }
void  nl_queue_after_ns(void* qv, int64_t ns, nl_dispatch_fn fp) { (void)qv; (void)ns; (void)fp; DISPATCH_UNAVAILABLE(nl_queue_after_ns); }
void* nl_group_create(void)                         { DISPATCH_UNAVAILABLE(nl_group_create); return NULL; }
void  nl_group_destroy(void* gv)                    { (void)gv; DISPATCH_UNAVAILABLE(nl_group_destroy); }
void  nl_group_async(void* gv, void* qv, nl_dispatch_fn fp) { (void)gv; (void)qv; (void)fp; DISPATCH_UNAVAILABLE(nl_group_async); }
void  nl_group_notify(void* gv, void* qv, nl_dispatch_fn fp) { (void)gv; (void)qv; (void)fp; DISPATCH_UNAVAILABLE(nl_group_notify); }
int   nl_group_wait_ns(void* gv, int64_t timeout_ns) { (void)gv; (void)timeout_ns; DISPATCH_UNAVAILABLE(nl_group_wait_ns); return -1; }

#endif /* __APPLE__ */

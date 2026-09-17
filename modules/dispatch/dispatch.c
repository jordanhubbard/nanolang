/* dispatch.c — NanoLang dispatch module implementation
 *
 * On Apple platforms: thin wrappers over libdispatch (Grand Central Dispatch).
 * My native API accepts plain C function pointers so generated
 * FnType_N typedefs integrate without Blocks on the caller side.  Internally
 * each wrapper converts to a Clang Block before passing to GCD (-fblocks is
 * applied via cflags_macos in module.json). My bytecode adapters use retained
 * handles and explicit owner/worker policies from the same manifest.
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
typedef struct { dispatch_queue_t q; dispatch_group_t pending; } _NlQueueInternal;
typedef struct { dispatch_group_t g; }  _NlGroupInternal;

/* ── Queue lifecycle ─────────────────────────────────────────────────────── */

void* nl_queue_serial(const char* label) {
    _NlQueueInternal* q = (_NlQueueInternal*)malloc(sizeof(_NlQueueInternal));
    if (!q) return NULL;
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_SERIAL);
    q->pending = dispatch_group_create();
    if (!q->q || !q->pending) {
        if (q->q) dispatch_release(q->q);
        if (q->pending) dispatch_release(q->pending);
        free(q);
        return NULL;
    }
    return (void*)q;
}

void* nl_queue_concurrent(const char* label) {
    _NlQueueInternal* q = (_NlQueueInternal*)malloc(sizeof(_NlQueueInternal));
    if (!q) return NULL;
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_CONCURRENT);
    q->pending = dispatch_group_create();
    if (!q->q || !q->pending) {
        if (q->q) dispatch_release(q->q);
        if (q->pending) dispatch_release(q->pending);
        free(q);
        return NULL;
    }
    return (void*)q;
}

void nl_queue_destroy(void* qv) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    if (!q) return;
    dispatch_group_wait(q->pending, DISPATCH_TIME_FOREVER);
    dispatch_barrier_sync(q->q, ^{});
    dispatch_release(q->pending);
    dispatch_release(q->q);
    free(q);
}

/* ── Dispatch primitives ─────────────────────────────────────────────────── */

void nl_queue_async(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_group_async(q->pending, q->q, ^{ fp(); });
}

void nl_queue_sync(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_sync(q->q, ^{ fp(); });
}

void nl_queue_barrier_async(void* qv, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_group_t pending = q->pending;
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_barrier_async(q->q, ^{ fp(); dispatch_group_leave(pending); dispatch_release(pending); });
}

void nl_queue_after_ns(void* qv, int64_t ns, nl_dispatch_fn fp) {
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_time_t when = dispatch_time(DISPATCH_TIME_NOW, ns);
    dispatch_group_t pending = q->pending;
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_after(when, q->q, ^{ fp(); dispatch_group_leave(pending); dispatch_release(pending); });
}

/* ── Group operations ────────────────────────────────────────────────────── */

void* nl_group_create(void) {
    _NlGroupInternal* g = (_NlGroupInternal*)malloc(sizeof(_NlGroupInternal));
    if (!g) return NULL;
    g->g = dispatch_group_create();
    if (!g->g) { free(g); return NULL; }
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
    dispatch_group_t pending = q->pending;
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_group_async(g->g, q->q, ^{ fp(); dispatch_group_leave(pending); dispatch_release(pending); });
}

void nl_group_notify(void* gv, void* qv, nl_dispatch_fn fp) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    _NlQueueInternal* q = (_NlQueueInternal*)qv;
    dispatch_group_t pending = q->pending;
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_group_notify(g->g, q->q, ^{ fp(); dispatch_group_leave(pending); dispatch_release(pending); });
}

int nl_group_wait_ns(void* gv, int64_t timeout_ns) {
    _NlGroupInternal* g = (_NlGroupInternal*)gv;
    dispatch_time_t t = (timeout_ns < 0)
        ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
    return (int)dispatch_group_wait(g->g, t);
}

int nl_dispatch_available(void) {
    return 1;
}

/* I release the publication reference even when VM shutdown cancels execution.
 * VM errors are latched by the handle bridge and checked by its owner. */
static void invoke_retained(NanoCallbackV1 *callback) {
    NanoCallbackValue result;
    (void)callback->invoke(callback, NULL, 0, &result);
    callback->release(callback);
}

void nl_queue_async_retained(void *qv, NanoCallbackV1 *callback) {
    _NlQueueInternal *q = qv;
    callback->retain(callback);
    dispatch_group_async(q->pending, q->q, ^{ invoke_retained(callback); });
}

void nl_queue_sync_retained(void *qv, NanoCallbackV1 *callback) {
    _NlQueueInternal *q = qv;
    callback->retain(callback);
    dispatch_sync(q->q, ^{ invoke_retained(callback); });
}

void nl_queue_barrier_async_retained(void *qv, NanoCallbackV1 *callback) {
    _NlQueueInternal *q = qv;
    dispatch_group_t pending = q->pending;
    callback->retain(callback);
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_barrier_async(q->q, ^{
        invoke_retained(callback);
        dispatch_group_leave(pending);
        dispatch_release(pending);
    });
}

void nl_queue_after_ns_retained(void *qv, int64_t ns, NanoCallbackV1 *callback) {
    _NlQueueInternal *q = qv;
    dispatch_group_t pending = q->pending;
    callback->retain(callback);
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, ns), q->q, ^{
        invoke_retained(callback);
        dispatch_group_leave(pending);
        dispatch_release(pending);
    });
}

void nl_group_async_retained(void *gv, void *qv, NanoCallbackV1 *callback) {
    _NlGroupInternal *g = gv;
    _NlQueueInternal *q = qv;
    dispatch_group_t pending = q->pending;
    callback->retain(callback);
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_group_async(g->g, q->q, ^{
        invoke_retained(callback);
        dispatch_group_leave(pending);
        dispatch_release(pending);
    });
}

void nl_group_notify_retained(void *gv, void *qv, NanoCallbackV1 *callback) {
    _NlGroupInternal *g = gv;
    _NlQueueInternal *q = qv;
    dispatch_group_t pending = q->pending;
    callback->retain(callback);
    dispatch_retain(pending);
    dispatch_group_enter(pending);
    dispatch_group_notify(g->g, q->q, ^{
        invoke_retained(callback);
        dispatch_group_leave(pending);
        dispatch_release(pending);
    });
}

int64_t nl_group_wait_ns_retained(void *g, int64_t ns) {
    return nl_group_wait_ns(g, ns);
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
int   nl_dispatch_available(void) { return 0; }

void nl_queue_async_retained(void *q, NanoCallbackV1 *c) { (void)q; (void)c; DISPATCH_UNAVAILABLE(nl_queue_async); }
void nl_queue_sync_retained(void *q, NanoCallbackV1 *c) { (void)q; (void)c; DISPATCH_UNAVAILABLE(nl_queue_sync); }
void nl_queue_barrier_async_retained(void *q, NanoCallbackV1 *c) { (void)q; (void)c; DISPATCH_UNAVAILABLE(nl_queue_barrier_async); }
void nl_queue_after_ns_retained(void *q, int64_t ns, NanoCallbackV1 *c) { (void)q; (void)ns; (void)c; DISPATCH_UNAVAILABLE(nl_queue_after_ns); }
void nl_group_async_retained(void *g, void *q, NanoCallbackV1 *c) { (void)g; (void)q; (void)c; DISPATCH_UNAVAILABLE(nl_group_async); }
void nl_group_notify_retained(void *g, void *q, NanoCallbackV1 *c) { (void)g; (void)q; (void)c; DISPATCH_UNAVAILABLE(nl_group_notify); }
int64_t nl_group_wait_ns_retained(void *g, int64_t ns) { (void)g; (void)ns; DISPATCH_UNAVAILABLE(nl_group_wait_ns); return -1; }

#endif /* __APPLE__ */

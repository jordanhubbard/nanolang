/* dispatch.c — NanoLang dispatch module implementation
 *
 * Thin wrappers over libdispatch. Every function allocates/frees a small
 * heap struct so NanoLang's opaque-type FFI can pass it as a pointer.
 */
#include "dispatch.h"
#include <stdlib.h>

/* ── Queue lifecycle ─────────────────────────────────────────────────────── */

NlQueue* nl_queue_serial(const char* label) {
    NlQueue* q = (NlQueue*)malloc(sizeof(NlQueue));
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_SERIAL);
    return q;
}

NlQueue* nl_queue_concurrent(const char* label) {
    NlQueue* q = (NlQueue*)malloc(sizeof(NlQueue));
    q->q = dispatch_queue_create(label, DISPATCH_QUEUE_CONCURRENT);
    return q;
}

void nl_queue_destroy(NlQueue* q) {
    if (!q) return;
    /* Drain then release */
    dispatch_barrier_sync(q->q, ^{});
    dispatch_release(q->q);
    free(q);
}

/* ── Dispatch primitives ─────────────────────────────────────────────────── */

void nl_queue_async(NlQueue* q, dispatch_block_t blk) {
    dispatch_async(q->q, blk);
}

void nl_queue_sync(NlQueue* q, dispatch_block_t blk) {
    dispatch_sync(q->q, blk);
}

void nl_queue_barrier_async(NlQueue* q, dispatch_block_t blk) {
    dispatch_barrier_async(q->q, blk);
}

void nl_queue_after_ns(NlQueue* q, int64_t ns, dispatch_block_t blk) {
    dispatch_time_t when = dispatch_time(DISPATCH_TIME_NOW, ns);
    dispatch_after(when, q->q, blk);
}

/* ── Group operations ────────────────────────────────────────────────────── */

NlQueueGroup* nl_group_create(void) {
    NlQueueGroup* g = (NlQueueGroup*)malloc(sizeof(NlQueueGroup));
    g->g = dispatch_group_create();
    return g;
}

void nl_group_destroy(NlQueueGroup* g) {
    if (!g) return;
    dispatch_release(g->g);
    free(g);
}

void nl_group_async(NlQueueGroup* g, NlQueue* q, dispatch_block_t blk) {
    dispatch_group_async(g->g, q->q, blk);
}

void nl_group_notify(NlQueueGroup* g, NlQueue* q, dispatch_block_t blk) {
    dispatch_group_notify(g->g, q->q, blk);
}

int nl_group_wait_ns(NlQueueGroup* g, int64_t timeout_ns) {
    dispatch_time_t t = (timeout_ns < 0)
        ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
    return (int)dispatch_group_wait(g->g, t);
}

/* I inspect my private queue only to make cancellation races deterministic. */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

static int allocations_until_failure = -1;
static void *checked_calloc(size_t count, size_t size) {
    if (allocations_until_failure == 0) return NULL;
    if (allocations_until_failure > 0) allocations_until_failure--;
    return calloc(count, size);
}
#define calloc checked_calloc
#include "../../src/runtime/callback_runtime.c"
#undef calloc

static unsigned drops;
static NanoCallbackStatus never_execute(void *p, const NanoCallbackValue *args,
                                       uint32_t count, NanoCallbackValue *out) {
    (void)p; (void)args; (void)count; (void)out;
    abort();
}
static void count_drop(void *p) {
    (void)p;
    drops++;
}
static void *cancelled_worker(void *p) {
    NanoCallbackV1 *cb = p;
    NanoCallbackValue out;
    assert(cb->invoke(cb, NULL, 0, &out) == NANO_CALLBACK_CANCELLED);
    assert(out.tag == NANO_CALLBACK_VOID);
    cb->release(cb);
    return NULL;
}

static void test_allocation_failure(void) {
    allocations_until_failure = 0;
    assert(!nano_callback_runtime_create());
    allocations_until_failure = -1;
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    allocations_until_failure = 0;
    NanoCallbackSignature sig = {0};
    assert(!nano_callback_create(rt, &sig, never_execute, count_drop, NULL));
    assert(!rt->handles && !rt->callbacks && !drops);
    allocations_until_failure = -1;
    NanoCallbackV1 *cb = nano_callback_create(rt, &sig, never_execute, count_drop, NULL);
    assert(cb && rt->handles == 1);
    allocations_until_failure = 0;
    assert(!nano_callback_create(rt, &sig, never_execute, count_drop, NULL));
    assert(rt->handles == 1 && !drops);
    cb->release(cb);
    nano_callback_collect(rt);
    assert(!rt->handles && !rt->callbacks && drops == 1);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
    allocations_until_failure = -1;
}

static void test_queued_cancellation(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    NanoCallbackSignature sig = {0};
    NanoCallbackV1 *cb = nano_callback_create(rt, &sig, never_execute, count_drop, NULL);
    assert(cb);
    cb->retain(cb);
    pthread_t thread;
    assert(pthread_create(&thread, NULL, cancelled_worker, cb) == 0);
    pthread_mutex_lock(&rt->mutex);
    while (!rt->first) pthread_cond_wait(&rt->changed, &rt->mutex);
    pthread_mutex_unlock(&rt->mutex);
    /* I know the foreign thread has enqueued, rather than merely started. */
    unsigned previous = drops;
    assert(nano_callback_close(rt) == NANO_CALLBACK_OK);
    assert(drops == previous + 1 && !rt->first && !rt->last);
    assert(pthread_join(thread, NULL) == 0);
    assert(rt->handles == 1);
    cb->release(cb);
    assert(!rt->handles && !rt->callbacks);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
}

static NanoCallbackStatus echo(void *p, const NanoCallbackValue *args,
                               uint32_t count, NanoCallbackValue *out) {
    (void)p;
    assert(count == 1);
    *out = args[0];
    return NANO_CALLBACK_OK;
}

static void test_scalars(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    for (uint32_t tag = NANO_CALLBACK_INT; tag <= NANO_CALLBACK_POINTER; tag++) {
        NanoCallbackSignature sig = { .argument_count = 1, .result_tag = tag,
                                      .argument_tags = { tag } };
        NanoCallbackV1 *cb = nano_callback_create(rt, &sig, echo, NULL, NULL);
        assert(cb);
        NanoCallbackValue arg = { .tag = tag }, out;
        switch (tag) {
            case NANO_CALLBACK_INT: arg.as.integer = INT64_MIN; break;
            case NANO_CALLBACK_FLOAT: arg.as.number = 12.5; break;
            case NANO_CALLBACK_BOOL: arg.as.byte = 1; break;
            case NANO_CALLBACK_BYTE: arg.as.byte = 255; break;
            case NANO_CALLBACK_POINTER: arg.as.pointer = rt; break;
        }
        assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_OK);
        assert(out.tag == tag && memcmp(&out.as, &arg.as, sizeof(arg.as)) == 0);
        if (tag == NANO_CALLBACK_BOOL) {
            arg.as.byte = 2;
            assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_TYPE_ERROR);
            assert(out.tag == NANO_CALLBACK_VOID);
        }
        cb->release(cb);
        nano_callback_collect(rt);
    }
    assert(!rt->handles);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
}

int main(void) {
    test_allocation_failure();
    test_scalars();
    for (unsigned i = 0; i < 100; i++) test_queued_cancellation();
    puts("I passed callback allocation, scalar, and deterministic cancellation checks.");
    return 0;
}

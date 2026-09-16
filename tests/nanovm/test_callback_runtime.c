#include "../../src/runtime/callback_runtime.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

enum { THREADS = 8, CALLS = 500 };
typedef struct {
    NanoCallbackRuntime *runtime;
    pthread_t owner;
    unsigned executed, dropped;
    bool bad_result, fail, release_in_call;
    NanoCallbackV1 *nested, *self;
} Payload;

static NanoCallbackStatus run(void *opaque, const NanoCallbackValue *args,
                              uint32_t count, NanoCallbackValue *result) {
    Payload *p = opaque;
    assert(pthread_equal(pthread_self(), p->owner));
    assert(nano_callback_close(p->runtime) == NANO_CALLBACK_BUSY);
    assert(nano_callback_runtime_destroy(p->runtime) == NANO_CALLBACK_BUSY);
    assert(count == 1);
    p->executed++;
    if (p->release_in_call) {
        p->release_in_call = false;
        p->self->release(p->self);
        nano_callback_collect(p->runtime);
        assert(!p->dropped);
    }
    if (p->nested) {
        assert(p->nested->invoke(p->nested, args, count, result) == NANO_CALLBACK_OK);
        return NANO_CALLBACK_OK;
    }
    result->tag = p->bad_result ? NANO_CALLBACK_FLOAT : NANO_CALLBACK_INT;
    result->as.integer = args[0].as.integer + 1;
    return p->fail ? NANO_CALLBACK_EXECUTION_ERROR : NANO_CALLBACK_OK;
}

static void drop(void *opaque) {
    Payload *p = opaque;
    assert(pthread_equal(pthread_self(), p->owner));
    assert(nano_callback_runtime_destroy(p->runtime) == NANO_CALLBACK_BUSY);
    p->dropped++;
}

static NanoCallbackV1 *publish(NanoCallbackRuntime *rt, Payload *p) {
    p->owner = pthread_self();
    p->runtime = rt;
    NanoCallbackSignature sig = { .argument_count = 1,
        .argument_tags = { NANO_CALLBACK_INT }, .result_tag = NANO_CALLBACK_INT };
    NanoCallbackV1 *cb = nano_callback_create(rt, &sig, run, drop, p);
    assert(cb && cb->abi_version == NANO_CALLBACK_ABI_V1);
    p->self = cb;
    return cb;
}

static NanoCallbackValue integer(int64_t n) {
    NanoCallbackValue v = { .tag = NANO_CALLBACK_INT, .as.integer = n };
    return v;
}

static void test_types_and_reentrancy(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    Payload p = {0}, q = {0};
    NanoCallbackV1 *cb = publish(rt, &p);
    NanoCallbackV1 *nested = publish(rt, &q);
    NanoCallbackValue arg = integer(41), out;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_OK);
    assert(out.tag == NANO_CALLBACK_INT && out.as.integer == 42);
    assert(cb->invoke(cb, &arg, 1, &arg) == NANO_CALLBACK_OK);
    assert(arg.as.integer == 42);
    assert(cb->invoke(cb, &arg, 0, &out) == NANO_CALLBACK_TYPE_ERROR);
    assert(cb->invoke(cb, NULL, 1, &out) == NANO_CALLBACK_TYPE_ERROR);
    assert(cb->invoke(cb, &arg, UINT32_MAX, &out) == NANO_CALLBACK_TYPE_ERROR);
    arg.tag = NANO_CALLBACK_FLOAT;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_TYPE_ERROR);
    assert(out.tag == NANO_CALLBACK_VOID);
    arg = integer(2);
    p.bad_result = true;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_TYPE_ERROR);
    assert(out.tag == NANO_CALLBACK_VOID);
    p.bad_result = false;
    p.fail = true;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_EXECUTION_ERROR);
    assert(out.tag == NANO_CALLBACK_VOID);
    p.fail = false;
    p.nested = nested;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_OK);
    assert(out.as.integer == 3 && q.executed == 1);
    p.nested = NULL;
    p.release_in_call = true;
    assert(cb->invoke(cb, &arg, 1, NULL) == NANO_CALLBACK_OK);
    nano_callback_collect(rt);
    assert(p.dropped == 1);
    nested->release(nested);
    nano_callback_collect(rt);
    assert(q.dropped == 1);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
}

static void *call_many(void *opaque) {
    NanoCallbackV1 *cb = opaque;
    for (unsigned i = 0; i < CALLS; i++) {
        NanoCallbackValue arg = integer(i), out;
        assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_OK);
        assert(out.tag == NANO_CALLBACK_INT && out.as.integer == i + 1);
    }
    cb->release(cb);
    return NULL;
}

static void test_concurrent(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    Payload p = {0};
    NanoCallbackV1 *cb = publish(rt, &p);
    pthread_t threads[THREADS];
    for (unsigned i = 0; i < THREADS; i++) {
        cb->retain(cb);
        assert(pthread_create(&threads[i], NULL, call_many, cb) == 0);
    }
    cb->release(cb);
    while (p.executed < THREADS * CALLS) assert(nano_callback_pump(rt, true) >= 0);
    for (unsigned i = 0; i < THREADS; i++) assert(pthread_join(threads[i], NULL) == 0);
    nano_callback_collect(rt);
    assert(p.dropped == 1);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
}

static void *call_cancelled(void *opaque) {
    NanoCallbackV1 *cb = opaque;
    NanoCallbackValue arg = integer(42), out;
    assert(cb->invoke(cb, &arg, 1, &out) == NANO_CALLBACK_CANCELLED);
    assert(out.tag == NANO_CALLBACK_VOID);
    cb->release(cb);
    return NULL;
}

static void test_shutdown(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    Payload p = {0};
    NanoCallbackV1 *cb = publish(rt, &p);
    pthread_t threads[THREADS];
    for (unsigned i = 0; i < THREADS; i++) {
        cb->retain(cb);
        assert(pthread_create(&threads[i], NULL, call_cancelled, cb) == 0);
    }
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
    assert(p.dropped == 1 && p.executed == 0);
    for (unsigned i = 0; i < THREADS; i++) assert(pthread_join(threads[i], NULL) == 0);
    /* I retain the public handle beyond destruction of the owning runtime. */
    assert(pthread_create(&threads[0], NULL, call_cancelled, cb) == 0);
    assert(pthread_join(threads[0], NULL) == 0);
}

static void *wrong_thread(void *opaque) {
    NanoCallbackRuntime *rt = opaque;
    assert(nano_callback_pump(rt, false) == -1);
    assert(nano_callback_close(rt) == NANO_CALLBACK_WRONG_THREAD);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_WRONG_THREAD);
    NanoCallbackSignature sig = {0};
    assert(!nano_callback_create(rt, &sig, run, NULL, NULL));
    nano_callback_collect(rt);
    nano_callback_wake(rt);
    return NULL;
}

static void test_publication_and_owner(void) {
    NanoCallbackRuntime *rt = nano_callback_runtime_create();
    assert(rt);
    Payload p = {0};
    NanoCallbackSignature sig = { .argument_count = NANO_CALLBACK_MAX_ARGS + 1 };
    assert(!nano_callback_create(rt, &sig, run, drop, &p));
    sig.argument_count = 1;
    assert(!nano_callback_create(rt, &sig, run, drop, &p));
    sig.argument_tags[0] = 99;
    assert(!nano_callback_create(rt, &sig, run, drop, &p));
    sig.argument_tags[0] = NANO_CALLBACK_INT;
    sig.result_tag = 99;
    assert(!nano_callback_create(rt, &sig, run, drop, &p));
    assert(!p.dropped);
    pthread_t t;
    assert(pthread_create(&t, NULL, wrong_thread, rt) == 0);
    assert(pthread_join(t, NULL) == 0);
    /* I must not lose a wake delivered before entering my wait. */
    assert(nano_callback_pump(rt, true) == 0);
    assert(nano_callback_close(rt) == NANO_CALLBACK_OK);
    sig.result_tag = NANO_CALLBACK_INT;
    assert(!nano_callback_create(rt, &sig, run, drop, &p));
    assert(!p.dropped);
    assert(nano_callback_pump(rt, true) == 0);
    assert(nano_callback_runtime_destroy(rt) == NANO_CALLBACK_OK);
}

int main(void) {
    test_types_and_reentrancy();
    test_publication_and_owner();
    for (unsigned i = 0; i < 10; i++) {
        test_concurrent();
        test_shutdown();
    }
    puts("I passed callback lifecycle tests, including 40,000 cross-thread invocations.");
    return 0;
}

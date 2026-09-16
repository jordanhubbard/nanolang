#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "../../modules/dispatch/dispatch.h"
#include "../../src/runtime/callback_runtime.h"

#ifdef __APPLE__
typedef struct { pthread_t owner; int calls, drops; } Payload;
static NanoCallbackStatus execute(void *opaque, const NanoCallbackValue *args,
                                  uint32_t count, NanoCallbackValue *result) {
    (void)args;
    Payload *payload = opaque;
    assert(!count && pthread_equal(payload->owner, pthread_self()));
    payload->calls++;
    *result = (NanoCallbackValue){.tag = NANO_CALLBACK_VOID};
    return NANO_CALLBACK_OK;
}
static void drop(void *opaque) {
    Payload *payload = opaque;
    assert(pthread_equal(payload->owner, pthread_self()));
    payload->drops++;
}
typedef struct {
    NanoCallbackRuntime *runtime;
    void *queue;
    NanoCallbackV1 *callback;
    pthread_mutex_t mutex;
    bool done;
} Wait;
static void *wait_worker(void *opaque) {
    Wait *wait = opaque;
    if (wait->callback) nl_queue_sync_retained(wait->queue, wait->callback);
    else nl_queue_destroy(wait->queue);
    pthread_mutex_lock(&wait->mutex);
    wait->done = true;
    pthread_mutex_unlock(&wait->mutex);
    nano_callback_wake(wait->runtime);
    return NULL;
}
static void pump_wait(NanoCallbackRuntime *runtime, void *queue, NanoCallbackV1 *callback) {
    Wait wait = {.runtime = runtime, .queue = queue, .callback = callback};
    assert(!pthread_mutex_init(&wait.mutex, NULL));
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, wait_worker, &wait));
    for (;;) {
        pthread_mutex_lock(&wait.mutex);
        bool done = wait.done;
        pthread_mutex_unlock(&wait.mutex);
        if (done) break;
        assert(nano_callback_pump(runtime, true) >= 0);
    }
    assert(!pthread_join(thread, NULL));
    assert(!pthread_mutex_destroy(&wait.mutex));
    nano_callback_collect(runtime);
}

int main(void) {
    alarm(20);
    NanoCallbackRuntime *runtime = nano_callback_runtime_create();
    assert(runtime);
    Payload payload = {.owner = pthread_self()};
    NanoCallbackSignature signature = {.result_tag = NANO_CALLBACK_VOID};
    NanoCallbackV1 *callback = nano_callback_create(runtime, &signature, execute, drop, &payload);
    assert(callback);
    void *queue = nl_queue_concurrent("nano.retained.test");
    void *group = nl_group_create();
    assert(queue && group);
    pump_wait(runtime, queue, callback);
    assert(payload.calls == 1);
    for (int i = 0; i < 128; i++) nl_group_async_retained(group, queue, callback);
    nl_group_notify_retained(group, queue, callback);
    nl_queue_async_retained(queue, callback);
    nl_queue_barrier_async_retained(queue, callback);
    nl_queue_after_ns_retained(queue, 10000000, callback);
    callback->release(callback);
    /* I must execute the timer before destruction returns, not just drain
     * the tasks that happened to be enqueued before its deadline. */
    pump_wait(runtime, queue, NULL);
    assert(payload.calls == 133 && payload.drops == 1);
    assert(nl_group_wait_ns_retained(group, 0) == 0);
    nl_group_destroy(group);
    assert(nano_callback_runtime_destroy(runtime) == NANO_CALLBACK_OK);

    runtime = nano_callback_runtime_create();
    assert(runtime);
    Payload cancelled = {.owner = pthread_self()};
    callback = nano_callback_create(runtime, &signature, execute, drop, &cancelled);
    assert(callback);
    queue = nl_queue_serial("nano.retained.cancel");
    assert(queue);
    nl_queue_after_ns_retained(queue, 10000000, callback);
    callback->release(callback);
    assert(nano_callback_runtime_destroy(runtime) == NANO_CALLBACK_OK);
    assert(cancelled.drops == 1);
    nl_queue_destroy(queue);
    assert(cancelled.calls == 0);
    alarm(0);
    puts("I passed retained dispatch sync, concurrent groups, notification, barrier, timer drain and late cancellation checks.");
    return 0;
}
#else
int main(void) {
    assert(!nl_dispatch_available());
    puts("I verified the non-Apple unavailable backend; Apple dispatch execution is not tested here.");
    return 0;
}
#endif

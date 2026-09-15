#include "callback_runtime.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct Callback Callback;
typedef struct Request Request;
struct Callback {
    NanoCallbackV1 abi;
    NanoCallbackRuntime *runtime;
    Callback *next;
    uint64_t references;
    bool rooted;
    NanoCallbackExecute execute;
    NanoCallbackDrop drop;
    void *payload;
};
struct Request {
    Request *next;
    Callback *callback;
    const NanoCallbackValue *arguments;
    uint32_t count;
    NanoCallbackValue result;
    NanoCallbackStatus status;
    bool done;
};
struct NanoCallbackRuntime {
    pthread_t owner;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    Callback *callbacks;
    Request *first, *last;
    uint64_t handles;
    uint32_t executing;
    bool closed, host_live, wake_pending;
};

static bool owner(NanoCallbackRuntime *rt) {
    return pthread_equal(rt->owner, pthread_self()) != 0;
}

static void free_runtime(NanoCallbackRuntime *rt) {
    pthread_cond_destroy(&rt->changed);
    pthread_mutex_destroy(&rt->mutex);
    free(rt);
}

static void unlink_callback(NanoCallbackRuntime *rt, Callback *cb) {
    Callback **slot = &rt->callbacks;
    while (*slot != cb) slot = &(*slot)->next;
    *slot = cb->next;
    rt->handles--;
}

static void retain(NanoCallbackV1 *abi) {
    Callback *cb = (Callback *)abi;
    NanoCallbackRuntime *rt = cb->runtime;
    pthread_mutex_lock(&rt->mutex);
    /* I cannot recover from invalid ownership or counter overflow. */
    if (!cb->references || cb->references == UINT64_MAX) abort();
    cb->references++;
    pthread_mutex_unlock(&rt->mutex);
}

static void release(NanoCallbackV1 *abi) {
    Callback *cb = (Callback *)abi;
    NanoCallbackRuntime *rt = cb->runtime;
    bool dispose = false, dispose_rt = false;
    pthread_mutex_lock(&rt->mutex);
    if (!cb->references) abort();
    if (--cb->references == 0) {
        if (!cb->rooted) {
            unlink_callback(rt, cb);
            dispose = true;
            dispose_rt = !rt->host_live && !rt->handles;
        }
        rt->wake_pending = true;
        pthread_cond_broadcast(&rt->changed);
    }
    pthread_mutex_unlock(&rt->mutex);
    if (dispose) free(cb);
    if (dispose_rt) free_runtime(rt);
}

static bool valid_value(NanoCallbackValue v, uint32_t tag) {
    return v.tag == tag && (tag != NANO_CALLBACK_BOOL || v.as.byte <= 1);
}

static NanoCallbackStatus execute_request(NanoCallbackRuntime *rt, Request *r) {
    Callback *cb = r->callback;
    rt->executing++;
    NanoCallbackStatus status = cb->execute(cb->payload, r->arguments,
                                           r->count, &r->result);
    rt->executing--;
    if (status == NANO_CALLBACK_OK &&
        !valid_value(r->result, cb->abi.signature.result_tag))
        status = NANO_CALLBACK_TYPE_ERROR;
    if (status != NANO_CALLBACK_OK) memset(&r->result, 0, sizeof(r->result));
    return status;
}

static NanoCallbackStatus invoke(NanoCallbackV1 *abi,
    const NanoCallbackValue *arguments, uint32_t count,
    NanoCallbackValue *result) {
    Callback *cb = (Callback *)abi;
    NanoCallbackRuntime *rt = cb->runtime;
    /* I snapshot before clearing result: it may alias an argument. */
    NanoCallbackValue copied[NANO_CALLBACK_MAX_ARGS];
    bool valid = count == abi->signature.argument_count && (!count || arguments);
    if (valid) {
        for (uint32_t i = 0; i < count; i++) {
            copied[i] = arguments[i];
            if (!valid_value(copied[i], abi->signature.argument_tags[i])) valid = false;
        }
    }
    if (result) memset(result, 0, sizeof(*result));
    if (!valid) return NANO_CALLBACK_TYPE_ERROR;
    retain(abi);
    Request r = { .callback = cb, .arguments = copied, .count = count };
    pthread_mutex_lock(&rt->mutex);
    if (rt->closed) {
        pthread_mutex_unlock(&rt->mutex);
        release(abi);
        return NANO_CALLBACK_CANCELLED;
    }
    if (owner(rt)) {
        pthread_mutex_unlock(&rt->mutex);
        r.status = execute_request(rt, &r);
    } else {
        if (rt->last) rt->last->next = &r;
        else rt->first = &r;
        rt->last = &r;
        pthread_cond_broadcast(&rt->changed);
        while (!r.done) pthread_cond_wait(&rt->changed, &rt->mutex);
        pthread_mutex_unlock(&rt->mutex);
    }
    if (result) *result = r.result;
    NanoCallbackStatus status = r.status;
    release(abi);
    return status;
}

NanoCallbackRuntime *nano_callback_runtime_create(void) {
    NanoCallbackRuntime *rt = calloc(1, sizeof(*rt));
    if (!rt) return NULL;
    if (pthread_mutex_init(&rt->mutex, NULL) != 0) { free(rt); return NULL; }
    if (pthread_cond_init(&rt->changed, NULL) != 0) {
        pthread_mutex_destroy(&rt->mutex);
        free(rt);
        return NULL;
    }
    rt->owner = pthread_self();
    rt->host_live = true;
    return rt;
}

NanoCallbackV1 *nano_callback_create(NanoCallbackRuntime *rt,
    const NanoCallbackSignature *signature, NanoCallbackExecute execute,
    NanoCallbackDrop drop, void *payload) {
    if (!rt || !owner(rt) || rt->closed || !signature || !execute ||
        signature->argument_count > NANO_CALLBACK_MAX_ARGS ||
        signature->result_tag > NANO_CALLBACK_POINTER) return NULL;
    for (uint32_t i = 0; i < signature->argument_count; i++)
        if (signature->argument_tags[i] == NANO_CALLBACK_VOID ||
            signature->argument_tags[i] > NANO_CALLBACK_POINTER) return NULL;
    Callback *cb = calloc(1, sizeof(*cb));
    if (!cb) return NULL;
    cb->abi.abi_version = NANO_CALLBACK_ABI_V1;
    cb->abi.signature = *signature;
    cb->abi.retain = retain;
    cb->abi.release = release;
    cb->abi.invoke = invoke;
    cb->runtime = rt;
    cb->references = 1;
    cb->rooted = true;
    cb->execute = execute;
    cb->drop = drop;
    cb->payload = payload;
    pthread_mutex_lock(&rt->mutex);
    cb->next = rt->callbacks;
    rt->callbacks = cb;
    rt->handles++;
    pthread_mutex_unlock(&rt->mutex);
    return &cb->abi;
}

void nano_callback_collect(NanoCallbackRuntime *rt) {
    if (!rt || !owner(rt)) return;
    for (;;) {
        pthread_mutex_lock(&rt->mutex);
        Callback *cb = rt->callbacks;
        while (cb && cb->references) cb = cb->next;
        if (!cb) { pthread_mutex_unlock(&rt->mutex); return; }
        unlink_callback(rt, cb);
        pthread_mutex_unlock(&rt->mutex);
        if (cb->rooted && cb->drop) {
            rt->executing++;
            cb->drop(cb->payload);
            rt->executing--;
        }
        free(cb);
    }
}

int nano_callback_pump(NanoCallbackRuntime *rt, bool wait) {
    if (!rt || !owner(rt)) return -1;
    nano_callback_collect(rt);
    pthread_mutex_lock(&rt->mutex);
    if (wait && !rt->first && !rt->closed && !rt->wake_pending)
        pthread_cond_wait(&rt->changed, &rt->mutex);
    rt->wake_pending = false;
    Request *r = rt->first;
    if (!r) { pthread_mutex_unlock(&rt->mutex); return 0; }
    rt->first = r->next;
    if (!rt->first) rt->last = NULL;
    pthread_mutex_unlock(&rt->mutex);
    NanoCallbackStatus status = execute_request(rt, r);
    pthread_mutex_lock(&rt->mutex);
    r->status = status;
    r->done = true;
    pthread_cond_broadcast(&rt->changed);
    pthread_mutex_unlock(&rt->mutex);
    return 1;
}

void nano_callback_wake(NanoCallbackRuntime *rt) {
    pthread_mutex_lock(&rt->mutex);
    rt->wake_pending = true;
    pthread_cond_broadcast(&rt->changed);
    pthread_mutex_unlock(&rt->mutex);
}

NanoCallbackStatus nano_callback_close(NanoCallbackRuntime *rt) {
    if (!rt || !owner(rt)) return NANO_CALLBACK_WRONG_THREAD;
    if (rt->executing) return NANO_CALLBACK_BUSY;
    pthread_mutex_lock(&rt->mutex);
    rt->closed = true;
    for (Request *r = rt->first; r; r = r->next) {
        r->status = NANO_CALLBACK_CANCELLED;
        r->done = true;
    }
    rt->first = rt->last = NULL;
    pthread_cond_broadcast(&rt->changed);
    pthread_mutex_unlock(&rt->mutex);
    /* I rescan under the lock: native releases may remove detached handles. */
    for (;;) {
        pthread_mutex_lock(&rt->mutex);
        Callback *cb = rt->callbacks;
        while (cb && !cb->rooted) cb = cb->next;
        if (!cb) { pthread_mutex_unlock(&rt->mutex); break; }
        NanoCallbackDrop drop = cb->drop;
        void *payload = cb->payload;
        cb->payload = NULL;
        cb->rooted = false;
        bool dispose = !cb->references;
        if (dispose) unlink_callback(rt, cb);
        pthread_mutex_unlock(&rt->mutex);
        if (drop) {
            rt->executing++;
            drop(payload);
            rt->executing--;
        }
        if (dispose) free(cb);
    }
    return NANO_CALLBACK_OK;
}

NanoCallbackStatus nano_callback_runtime_destroy(NanoCallbackRuntime *rt) {
    NanoCallbackStatus status = nano_callback_close(rt);
    if (status != NANO_CALLBACK_OK) return status;
    pthread_mutex_lock(&rt->mutex);
    rt->host_live = false;
    bool dispose = !rt->handles;
    pthread_mutex_unlock(&rt->mutex);
    if (dispose) free_runtime(rt);
    return NANO_CALLBACK_OK;
}

#include "runtime/nano_callback.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

static pthread_once_t string_once = PTHREAD_ONCE_INIT;
static pthread_key_t string_key;
static void create_string_key(void) {
    assert(pthread_key_create(&string_key, free) == 0);
}

/* I return storage freed by the worker's TLS destructor, not static memory. */
const char *retained_string(NanoCallbackV1 *callback, const char *text, void *original) {
    assert(text && text != original);
    if (!*text) return NULL;
    NanoCallbackValue arg = {.tag = NANO_CALLBACK_INT, .as.integer = 41}, result;
    if (callback->invoke(callback, &arg, 1, &result) != NANO_CALLBACK_OK) return NULL;
    assert(pthread_once(&string_once, create_string_key) == 0);
    char *buffer = malloc(strlen(text) + 32);
    assert(buffer && pthread_setspecific(string_key, buffer) == 0);
    sprintf(buffer, "%s:%lld", text, (long long)result.as.integer);
    return buffer;
}

const char *retained_string_identity(const char *text, void *original) {
    assert(text && text != original);
    return text;
}

int64_t retained_call(NanoCallbackV1 *callback, int64_t value) {
    NanoCallbackValue arg = {.tag = NANO_CALLBACK_INT, .as.integer = value}, result;
    if (callback->invoke(callback, &arg, 1, &result) != NANO_CALLBACK_OK) return -1;
    return result.as.integer;
}

double retained_mix(NanoCallbackV1 *callback, double fraction, bool enabled,
                    uint8_t byte, void *pointer) {
    return retained_call(callback, 40) + fraction + (enabled ? byte : 0) + (pointer ? 1000 : 0);
}

typedef struct {
    NanoCallbackV1 *callback;
    pthread_t thread;
    int64_t value;
} Pending;

static void *pending_run(void *opaque) {
    Pending *pending = opaque;
    pending->value = retained_call(pending->callback, 41);
    pending->callback->release(pending->callback);
    return NULL;
}

void *retained_start(NanoCallbackV1 *callback) {
    Pending *pending = calloc(1, sizeof(*pending));
    if (!pending) return NULL;
    callback->retain(callback);
    pending->callback = callback;
    if (pthread_create(&pending->thread, NULL, pending_run, pending)) {
        callback->release(callback);
        free(pending);
        return NULL;
    }
    return pending;
}

int64_t retained_wait(void *opaque) {
    Pending *pending = opaque;
    if (!pending) return -1;
    if (pthread_join(pending->thread, NULL)) abort();
    int64_t value = pending->value;
    free(pending);
    return value;
}

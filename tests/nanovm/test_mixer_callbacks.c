#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "../../src/runtime/callback_runtime.h"

static pthread_mutex_t audio_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t clear_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t clear_changed = PTHREAD_COND_INITIALIZER;
static bool clear_entered;
typedef void (SDLCALL *TestMixCallback)(void *, Uint8 *, int);
static TestMixCallback hook;
static void *hook_data;
static int closes;
static bool fail_allocation;
static void test_set_post_mix(TestMixCallback callback, void *userdata) {
    if (!callback) {
        pthread_mutex_lock(&clear_lock);
        clear_entered = true;
        pthread_cond_signal(&clear_changed);
        pthread_mutex_unlock(&clear_lock);
    }
    pthread_mutex_lock(&audio_lock);
    hook = callback;
    hook_data = userdata;
    pthread_mutex_unlock(&audio_lock);
}
static void test_close_audio(void) { closes++; }
static void *test_malloc(size_t size) { return fail_allocation ? NULL : malloc(size); }
#define Mix_SetPostMix test_set_post_mix
#define Mix_CloseAudio test_close_audio
#define malloc test_malloc
#include "../../modules/sdl_mixer/sdl_mixer_callbacks.c"
#undef malloc
#undef Mix_CloseAudio
#undef Mix_SetPostMix

typedef struct { pthread_t owner; int calls, drops; } Payload;
static NanoCallbackStatus execute(void *opaque, const NanoCallbackValue *args,
                                  uint32_t count, NanoCallbackValue *result) {
    Payload *payload = opaque;
    assert(pthread_equal(payload->owner, pthread_self()));
    assert(count == 3 && args[0].as.pointer == payload && args[2].as.integer == 8);
    assert(args[1].as.pointer);
    ((Uint8 *)args[1].as.pointer)[0] = 42;
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
    pthread_mutex_t mutex;
    bool entered, done;
    Uint8 buffer[8];
} AudioCall;
static void *audio_worker(void *opaque) {
    AudioCall *call = opaque;
    pthread_mutex_lock(&audio_lock);
    pthread_mutex_lock(&call->mutex);
    call->entered = true;
    pthread_mutex_unlock(&call->mutex);
    nano_callback_wake(call->runtime);
    if (hook) hook(hook_data, call->buffer, 8);
    pthread_mutex_unlock(&audio_lock);
    pthread_mutex_lock(&call->mutex);
    call->done = true;
    pthread_mutex_unlock(&call->mutex);
    nano_callback_wake(call->runtime);
    return NULL;
}
static void *clear_worker(void *opaque) { (void)opaque; nl_mix_clear_post_mix(); return NULL; }
static void native_callback(void *opaque, void *buffer, int64_t length) {
    int *count = opaque;
    assert(length == 8);
    ((Uint8 *)buffer)[0] = 17;
    (*count)++;
}

int main(void) {
    alarm(20);
    NanoCallbackRuntime *runtime = nano_callback_runtime_create();
    assert(runtime);
    Payload payload = {.owner = pthread_self()};
    NanoCallbackSignature signature = {.argument_count = 3, .result_tag = NANO_CALLBACK_VOID,
        .argument_tags = {NANO_CALLBACK_POINTER, NANO_CALLBACK_POINTER, NANO_CALLBACK_INT}};
    NanoCallbackV1 *callback = nano_callback_create(runtime, &signature, execute, drop, &payload);
    assert(callback);
    assert(nl_mix_set_post_mix_retained(NULL, &payload) == -1);
    assert(nl_mix_set_post_mix_retained(callback, &payload) == 0);
    PostMix *original = registered;
    fail_allocation = true;
    assert(nl_mix_set_post_mix_retained(callback, &payload) == -1 && registered == original);
    assert(nl_mix_set_post_mix(native_callback, &payload) == -1 && registered == original);
    fail_allocation = false;
    assert(nl_mix_set_post_mix_retained(callback, &payload) == 0);
    callback->release(callback);
    AudioCall call = {.runtime = runtime};
    assert(!pthread_mutex_init(&call.mutex, NULL));
    pthread_t audio, clearer;
    assert(!pthread_create(&audio, NULL, audio_worker, &call));
    /* I start clearing while the old callback still holds the audio lock. */
    for (;;) {
        pthread_mutex_lock(&call.mutex);
        bool entered = call.entered;
        pthread_mutex_unlock(&call.mutex);
        if (entered) break;
        sched_yield();
    }
    assert(!pthread_create(&clearer, NULL, clear_worker, NULL));
    pthread_mutex_lock(&clear_lock);
    while (!clear_entered) pthread_cond_wait(&clear_changed, &clear_lock);
    pthread_mutex_unlock(&clear_lock);
    for (;;) {
        pthread_mutex_lock(&call.mutex);
        bool done = call.done;
        pthread_mutex_unlock(&call.mutex);
        if (done) break;
        assert(nano_callback_pump(runtime, true) >= 0);
    }
    assert(!pthread_join(audio, NULL) && !pthread_join(clearer, NULL));
    assert(call.buffer[0] == 42 && payload.calls == 1 && !registered && !hook);
    nano_callback_collect(runtime);
    assert(payload.drops == 1);
    assert(!pthread_mutex_destroy(&call.mutex));

    int native_calls = 0;
    assert(nl_mix_set_post_mix(native_callback, &native_calls) == 0);
    Uint8 buffer[8] = {0};
    hook(hook_data, buffer, 8);
    assert(native_calls == 1 && buffer[0] == 17);
    Payload late = {.owner = pthread_self()};
    callback = nano_callback_create(runtime, &signature, execute, drop, &late);
    assert(callback && nl_mix_set_post_mix_retained(callback, &late) == 0);
    callback->release(callback);
    assert(nano_callback_runtime_destroy(runtime) == NANO_CALLBACK_OK);
    assert(late.drops == 1);
    buffer[0] = 0;
    hook(hook_data, buffer, 8);
    assert(buffer[0] == 0 && late.calls == 0);
    nl_mix_close_audio_retained();
    assert(closes == 1 && !registered && !hook);
    alarm(0);
    puts("I passed post-mix retention, replacement, allocation failure, audio-lock clearing, buffer mutation and late cancellation checks.");
    return 0;
}

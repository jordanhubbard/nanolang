#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>
#include <pthread.h>
#include <stdlib.h>
#include "../../src/runtime/nano_callback.h"
#define NL_MIX_IMPLEMENTATION
#include "sdl_mixer_callbacks.h"

typedef struct {
    NanoCallbackV1 *callback;
    NlPostMixCallback native;
    void *userdata;
} PostMix;

/* I serialize registration changes, not audio execution. Mix_SetPostMix's
 * audio lock establishes quiescence before I release the replaced context. */
static pthread_mutex_t registration_lock = PTHREAD_MUTEX_INITIALIZER;
static PostMix *registered;

static void SDLCALL post_mix(void *opaque, Uint8 *stream, int length) {
    PostMix *context = opaque;
    if (context->native) {
        context->native(context->userdata, stream, length);
        return;
    }
    NanoCallbackValue args[] = {
        {.tag = NANO_CALLBACK_POINTER, .as.pointer = context->userdata},
        {.tag = NANO_CALLBACK_POINTER, .as.pointer = stream},
        {.tag = NANO_CALLBACK_INT, .as.integer = length}
    }, result;
    (void)context->callback->invoke(context->callback, args, 3, &result);
}

static void replace_locked(PostMix *replacement) {
    PostMix *previous = registered;
    Mix_SetPostMix(replacement ? post_mix : NULL, replacement);
    registered = replacement;
    if (previous) {
        if (previous->callback) previous->callback->release(previous->callback);
        free(previous);
    }
}

/* I report publication failure without losing the previous registration. */
int64_t nl_mix_set_post_mix_retained(NanoCallbackV1 *callback, void *userdata) {
    if (!callback || callback->abi_version != NANO_CALLBACK_ABI_V1 ||
        callback->signature.argument_count != 3 ||
        callback->signature.result_tag != NANO_CALLBACK_VOID ||
        callback->signature.argument_tags[0] != NANO_CALLBACK_POINTER ||
        callback->signature.argument_tags[1] != NANO_CALLBACK_POINTER ||
        callback->signature.argument_tags[2] != NANO_CALLBACK_INT) return -1;
    PostMix *context = malloc(sizeof(*context));
    if (!context) return -1;
    callback->retain(callback);
    *context = (PostMix){.callback = callback, .userdata = userdata};
    pthread_mutex_lock(&registration_lock);
    replace_locked(context);
    pthread_mutex_unlock(&registration_lock);
    return 0;
}

int64_t nl_mix_set_post_mix(NlPostMixCallback callback, void *userdata) {
    if (!callback) return -1;
    PostMix *context = malloc(sizeof(*context));
    if (!context) return -1;
    *context = (PostMix){.native = callback, .userdata = userdata};
    pthread_mutex_lock(&registration_lock);
    replace_locked(context);
    pthread_mutex_unlock(&registration_lock);
    return 0;
}

void nl_mix_clear_post_mix(void) {
    pthread_mutex_lock(&registration_lock);
    replace_locked(NULL);
    pthread_mutex_unlock(&registration_lock);
}

void nl_mix_close_audio_retained(void) {
    pthread_mutex_lock(&registration_lock);
    replace_locked(NULL);
    Mix_CloseAudio();
    pthread_mutex_unlock(&registration_lock);
}

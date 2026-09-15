#ifndef NL_SDL_MIXER_CALLBACKS_H
#define NL_SDL_MIXER_CALLBACKS_H
#include <SDL2/SDL_mixer.h>
#include <stdint.h>

/* I use a 64-bit byte count at my native language boundary as well. */
typedef void (*NlPostMixCallback)(void *userdata, void *buffer, int64_t length);
int64_t nl_mix_set_post_mix(NlPostMixCallback callback, void *userdata);
void nl_mix_clear_post_mix(void);
void nl_mix_close_audio_retained(void);

/* I clear adapter-owned registrations on native close as on VM close. */
#ifndef NL_MIX_IMPLEMENTATION
#define Mix_CloseAudio nl_mix_close_audio_retained
#endif
#endif

#include <SDL2/SDL.h>
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include "../../modules/sdl_mixer/sdl_mixer_callbacks.h"
#include "../../modules/sdl_mixer/sdl_mixer_operations.h"

static void post_mix(void *userdata, void *buffer, int64_t length) {
    assert(buffer && length > 0);
    SDL_AtomicAdd(userdata, 1);
}
static void *missing_file(void *unused) {
    (void)unused;
    assert(!Mix_LoadWAV("examples/audio/nanolang-test-tone.wav/missing"));
    return NULL;
}
int main(void) {
    alarm(20);
    assert(SDL_Init(SDL_INIT_AUDIO) == 0);
    assert(Mix_OpenAudio(22050, AUDIO_S16SYS, 1, 128) == 0);
    SDL_atomic_t calls = {0};
    assert(nl_mix_set_post_mix(post_mix, &calls) == 0);
    while (SDL_AtomicGet(&calls) < 8) SDL_Delay(1);
    assert(Mix_AllocateChannels(16) == 16);
    assert(Mix_GetNumChannels() == 16);
    assert(Mix_PlayChannel(-1, NULL, 0) == -1);
    assert(strlen(Mix_GetError()) > 0);
    assert(Mix_ClearError() == 0 && !*Mix_GetError());
    assert(Mix_AllocateChannels(INT64_MAX) == -1);
    assert(strstr(Mix_GetError(), "SDK range"));
    pthread_t worker;
    assert(!pthread_create(&worker, NULL, missing_file, NULL));
    assert(!pthread_join(worker, NULL));
    assert(strlen(Mix_GetError()) > 0);
    Mix_Chunk *chunk = Mix_LoadWAV("examples/audio/nanolang-test-tone.wav");
    assert(chunk);
    assert(Mix_PlayChannel(0, chunk, -1) == 0);
    assert(Mix_Playing(0) > 0);
    assert(Mix_FreeChunk(chunk) == 0 && Mix_Playing(0) == 0);
    Mix_Music *music = Mix_LoadMUS("examples/audio/nanolang-test-tone.wav");
    assert(music && Mix_PlayMusic(music, -1) == 0);
    Mix_PauseMusic();
    assert(Mix_PausedMusic());
    Mix_ResumeMusic();
    assert(!Mix_PausedMusic());
    assert(Mix_RewindMusic() == 0);
    Mix_FreeMusic(music);
    assert(!Mix_PlayingMusic());
    Mix_CloseAudio();
    int stopped = SDL_AtomicGet(&calls);
    SDL_Delay(10);
    assert(SDL_AtomicGet(&calls) == stopped);
    Mix_Quit();
    SDL_Quit();
    return 0;
}

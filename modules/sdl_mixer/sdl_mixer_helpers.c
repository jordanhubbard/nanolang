// SDL_mixer Helper Functions for Nanolang FFI
// Provides wrappers for SDL_mixer audio functions

#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>
#include <stdint.h>
#include <stdlib.h>

// === Initialization ===

int64_t Mix_Init(int64_t flags) {
    return (int64_t)Mix_Init((int)flags);
}

int64_t Mix_Quit(void) {
    Mix_Quit();
    return 0;
}

int64_t Mix_OpenAudio(int64_t frequency, int64_t format, int64_t channels, int64_t chunksize) {
    return (int64_t)Mix_OpenAudio((int)frequency, (uint16_t)format, (int)channels, (int)chunksize);
}

int64_t Mix_CloseAudio(void) {
    Mix_CloseAudio();
    return 0;
}

// === Channel Allocation ===

int64_t Mix_AllocateChannels(int64_t numchans) {
    return (int64_t)Mix_AllocateChannels((int)numchans);
}

// === Sound Effect Functions ===

int64_t Mix_LoadWAV(const char* file) {
    Mix_Chunk* chunk = Mix_LoadWAV(file);
    return (int64_t)chunk;
}

int64_t Mix_FreeChunk(int64_t chunk) {
    Mix_FreeChunk((Mix_Chunk*)chunk);
    return 0;
}

int64_t Mix_PlayChannel(int64_t channel, int64_t chunk, int64_t loops) {
    return (int64_t)Mix_PlayChannel((int)channel, (Mix_Chunk*)chunk, (int)loops);
}

int64_t Mix_PlayChannelTimed(int64_t channel, int64_t chunk, int64_t loops, int64_t ticks) {
    return (int64_t)Mix_PlayChannelTimed((int)channel, (Mix_Chunk*)chunk, (int)loops, (int)ticks);
}

int64_t Mix_FadeInChannel(int64_t channel, int64_t chunk, int64_t loops, int64_t ms) {
    return (int64_t)Mix_FadeInChannel((int)channel, (Mix_Chunk*)chunk, (int)loops, (int)ms);
}

int64_t Mix_HaltChannel(int64_t channel) {
    return (int64_t)Mix_HaltChannel((int)channel);
}

int64_t Mix_FadeOutChannel(int64_t channel, int64_t ms) {
    return (int64_t)Mix_FadeOutChannel((int)channel, (int)ms);
}

// === Volume Control ===

int64_t Mix_Volume(int64_t channel, int64_t volume) {
    return (int64_t)Mix_Volume((int)channel, (int)volume);
}

int64_t Mix_VolumeChunk(int64_t chunk, int64_t volume) {
    return (int64_t)Mix_VolumeChunk((Mix_Chunk*)chunk, (int)volume);
}

// === Music Functions ===

int64_t Mix_LoadMUS(const char* file) {
    Mix_Music* music = Mix_LoadMUS(file);
    return (int64_t)music;
}

int64_t Mix_FreeMusic(int64_t music) {
    Mix_FreeMusic((Mix_Music*)music);
    return 0;
}

int64_t Mix_PlayMusic(int64_t music, int64_t loops) {
    return (int64_t)Mix_PlayMusic((Mix_Music*)music, (int)loops);
}

int64_t Mix_FadeInMusic(int64_t music, int64_t loops, int64_t ms) {
    return (int64_t)Mix_FadeInMusic((Mix_Music*)music, (int)loops, (int)ms);
}

int64_t Mix_FadeInMusicPos(int64_t music, int64_t loops, int64_t ms, double position) {
    return (int64_t)Mix_FadeInMusicPos((Mix_Music*)music, (int)loops, (int)ms, position);
}

int64_t Mix_HaltMusic(void) {
    return (int64_t)Mix_HaltMusic();
}

int64_t Mix_FadeOutMusic(int64_t ms) {
    return (int64_t)Mix_FadeOutMusic((int)ms);
}

int64_t Mix_RewindMusic(void) {
    Mix_RewindMusic();
    return 0;
}

int64_t Mix_PauseMusic(void) {
    Mix_PauseMusic();
    return 0;
}

int64_t Mix_ResumeMusic(void) {
    Mix_ResumeMusic();
    return 0;
}

int64_t Mix_VolumeMusic(int64_t volume) {
    return (int64_t)Mix_VolumeMusic((int)volume);
}

// === Status Queries ===

int64_t Mix_Playing(int64_t channel) {
    return (int64_t)Mix_Playing((int)channel);
}

int64_t Mix_Paused(int64_t channel) {
    return (int64_t)Mix_Paused((int)channel);
}

int64_t Mix_PlayingMusic(void) {
    return (int64_t)Mix_PlayingMusic();
}

int64_t Mix_PausedMusic(void) {
    return (int64_t)Mix_PausedMusic();
}

// === Error Handling ===

const char* Mix_GetError(void) {
    return Mix_GetError();
}

int64_t Mix_ClearError(void) {
    SDL_ClearError();
    return 0;
}


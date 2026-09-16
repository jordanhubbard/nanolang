#define NL_MIX_IMPLEMENTATION
#include "sdl_mixer_operations.h"
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

/* I keep the last completed operation's error with SDL's process-global mixer.
 * I copy at most 1023 bytes; readers receive a per-thread snapshot. */
static pthread_mutex_t error_lock = PTHREAD_MUTEX_INITIALIZER;
static char operation_error[1024];
static __thread char error_snapshot[1024];

void nl_mix_record_error(const char *message) {
    pthread_mutex_lock(&error_lock);
    snprintf(operation_error, sizeof operation_error, "%s", message ? message : "");
    pthread_mutex_unlock(&error_lock);
}
void nl_mix_begin_operation(void) { SDL_ClearError(); }
void nl_mix_finish_operation(void) { nl_mix_record_error(SDL_GetError()); }
const char *nl_mix_GetError(void) {
    pthread_mutex_lock(&error_lock);
    memcpy(error_snapshot, operation_error, sizeof error_snapshot);
    pthread_mutex_unlock(&error_lock);
    return error_snapshot;
}
int64_t nl_mix_ClearError(void) {
    SDL_ClearError();
    nl_mix_record_error("");
    return 0;
}

int64_t nl_mix_Init(int64_t flags) {
    nl_mix_begin_operation();
    if (flags < INT_MIN || flags > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_Init((int)flags);
    nl_mix_finish_operation();
    return result;
}

void nl_mix_Quit(void) {
    nl_mix_begin_operation();
    Mix_Quit();
    nl_mix_finish_operation();
}

int64_t nl_mix_OpenAudio(int64_t frequency, int64_t format, int64_t channels, int64_t chunksize) {
    nl_mix_begin_operation();
    if (frequency < INT_MIN || frequency > INT_MAX || format < 0 || format > UINT16_MAX || channels < INT_MIN || channels > INT_MAX || chunksize < INT_MIN || chunksize > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_OpenAudio((int)frequency, (Uint16)format, (int)channels, (int)chunksize);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_AllocateChannels(int64_t count) {
    nl_mix_begin_operation();
    if (count < INT_MIN || count > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_AllocateChannels((int)count);
    nl_mix_finish_operation();
    return result;
}

Mix_Chunk * nl_mix_LoadWAV(const char * file) {
    nl_mix_begin_operation();
    Mix_Chunk * result = Mix_LoadWAV(file);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FreeChunk(Mix_Chunk * chunk) {
    nl_mix_begin_operation();
    Mix_FreeChunk(chunk);
    nl_mix_finish_operation();
    return 0;
}

int64_t nl_mix_PlayChannel(int64_t channel, Mix_Chunk * chunk, int64_t loops) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX || loops < INT_MIN || loops > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_PlayChannel((int)channel, chunk, (int)loops);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_PlayChannelTimed(int64_t channel, Mix_Chunk * chunk, int64_t loops, int64_t ticks) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX || loops < INT_MIN || loops > INT_MAX || ticks < INT_MIN || ticks > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_PlayChannelTimed((int)channel, chunk, (int)loops, (int)ticks);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FadeInChannel(int64_t channel, Mix_Chunk * chunk, int64_t loops, int64_t ms) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX || loops < INT_MIN || loops > INT_MAX || ms < INT_MIN || ms > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_FadeInChannel((int)channel, chunk, (int)loops, (int)ms);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_HaltChannel(int64_t channel) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_HaltChannel((int)channel);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FadeOutChannel(int64_t channel, int64_t ms) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX || ms < INT_MIN || ms > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_FadeOutChannel((int)channel, (int)ms);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_Volume(int64_t channel, int64_t volume) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX || volume < INT_MIN || volume > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_Volume((int)channel, (int)volume);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_VolumeChunk(Mix_Chunk * chunk, int64_t volume) {
    nl_mix_begin_operation();
    if (volume < INT_MIN || volume > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_VolumeChunk(chunk, (int)volume);
    nl_mix_finish_operation();
    return result;
}

Mix_Music * nl_mix_LoadMUS(const char * file) {
    nl_mix_begin_operation();
    Mix_Music * result = Mix_LoadMUS(file);
    nl_mix_finish_operation();
    return result;
}

void nl_mix_FreeMusic(Mix_Music * music) {
    nl_mix_begin_operation();
    Mix_FreeMusic(music);
    nl_mix_finish_operation();
}

int64_t nl_mix_PlayMusic(Mix_Music * music, int64_t loops) {
    nl_mix_begin_operation();
    if (loops < INT_MIN || loops > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_PlayMusic(music, (int)loops);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FadeInMusic(Mix_Music * music, int64_t loops, int64_t ms) {
    nl_mix_begin_operation();
    if (loops < INT_MIN || loops > INT_MAX || ms < INT_MIN || ms > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_FadeInMusic(music, (int)loops, (int)ms);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FadeInMusicPos(Mix_Music * music, int64_t loops, int64_t ms, double position) {
    nl_mix_begin_operation();
    if (loops < INT_MIN || loops > INT_MAX || ms < INT_MIN || ms > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_FadeInMusicPos(music, (int)loops, (int)ms, position);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_HaltMusic(void) {
    nl_mix_begin_operation();
    int64_t result = Mix_HaltMusic();
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_FadeOutMusic(int64_t ms) {
    nl_mix_begin_operation();
    if (ms < INT_MIN || ms > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_FadeOutMusic((int)ms);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_RewindMusic(void) {
    nl_mix_begin_operation();
    Mix_RewindMusic();
    nl_mix_finish_operation();
    return 0;
}

void nl_mix_PauseMusic(void) {
    nl_mix_begin_operation();
    Mix_PauseMusic();
    nl_mix_finish_operation();
}

void nl_mix_ResumeMusic(void) {
    nl_mix_begin_operation();
    Mix_ResumeMusic();
    nl_mix_finish_operation();
}

int64_t nl_mix_VolumeMusic(int64_t volume) {
    nl_mix_begin_operation();
    if (volume < INT_MIN || volume > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_VolumeMusic((int)volume);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_Playing(int64_t channel) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_Playing((int)channel);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_Paused(int64_t channel) {
    nl_mix_begin_operation();
    if (channel < INT_MIN || channel > INT_MAX) {
        nl_mix_record_error("I require mixer integer arguments within the SDK range");
        return -1;
    }
    int64_t result = Mix_Paused((int)channel);
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_PlayingMusic(void) {
    nl_mix_begin_operation();
    int64_t result = Mix_PlayingMusic();
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_PausedMusic(void) {
    nl_mix_begin_operation();
    int64_t result = Mix_PausedMusic();
    nl_mix_finish_operation();
    return result;
}

int64_t nl_mix_GetNumChannels(void) {
    nl_mix_begin_operation();
    int64_t result = Mix_AllocateChannels(-1);
    nl_mix_finish_operation();
    return result;
}

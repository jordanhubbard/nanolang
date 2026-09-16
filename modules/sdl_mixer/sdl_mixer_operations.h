#ifndef NL_SDL_MIXER_OPERATIONS_H
#define NL_SDL_MIXER_OPERATIONS_H
#include <SDL2/SDL_mixer.h>
#include <stdint.h>

void nl_mix_begin_operation(void);
void nl_mix_finish_operation(void);
void nl_mix_record_error(const char *message);
const char *nl_mix_GetError(void);
int64_t nl_mix_ClearError(void);
int64_t nl_mix_Init(int64_t flags);
void nl_mix_Quit(void);
int64_t nl_mix_OpenAudio(int64_t frequency, int64_t format, int64_t channels, int64_t chunksize);
int64_t nl_mix_AllocateChannels(int64_t count);
Mix_Chunk * nl_mix_LoadWAV(const char * file);
int64_t nl_mix_FreeChunk(Mix_Chunk * chunk);
int64_t nl_mix_PlayChannel(int64_t channel, Mix_Chunk * chunk, int64_t loops);
int64_t nl_mix_PlayChannelTimed(int64_t channel, Mix_Chunk * chunk, int64_t loops, int64_t ticks);
int64_t nl_mix_FadeInChannel(int64_t channel, Mix_Chunk * chunk, int64_t loops, int64_t ms);
int64_t nl_mix_HaltChannel(int64_t channel);
int64_t nl_mix_FadeOutChannel(int64_t channel, int64_t ms);
int64_t nl_mix_Volume(int64_t channel, int64_t volume);
int64_t nl_mix_VolumeChunk(Mix_Chunk * chunk, int64_t volume);
Mix_Music * nl_mix_LoadMUS(const char * file);
void nl_mix_FreeMusic(Mix_Music * music);
int64_t nl_mix_PlayMusic(Mix_Music * music, int64_t loops);
int64_t nl_mix_FadeInMusic(Mix_Music * music, int64_t loops, int64_t ms);
int64_t nl_mix_FadeInMusicPos(Mix_Music * music, int64_t loops, int64_t ms, double position);
int64_t nl_mix_HaltMusic(void);
int64_t nl_mix_FadeOutMusic(int64_t ms);
int64_t nl_mix_RewindMusic(void);
void nl_mix_PauseMusic(void);
void nl_mix_ResumeMusic(void);
int64_t nl_mix_VolumeMusic(int64_t volume);
int64_t nl_mix_Playing(int64_t channel);
int64_t nl_mix_Paused(int64_t channel);
int64_t nl_mix_PlayingMusic(void);
int64_t nl_mix_PausedMusic(void);
int64_t nl_mix_GetNumChannels(void);

#ifndef NL_MIX_IMPLEMENTATION
#undef Mix_Init
#define Mix_Init nl_mix_Init
#undef Mix_Quit
#define Mix_Quit nl_mix_Quit
#undef Mix_OpenAudio
#define Mix_OpenAudio nl_mix_OpenAudio
#undef Mix_AllocateChannels
#define Mix_AllocateChannels nl_mix_AllocateChannels
#undef Mix_LoadWAV
#define Mix_LoadWAV nl_mix_LoadWAV
#undef Mix_FreeChunk
#define Mix_FreeChunk nl_mix_FreeChunk
#undef Mix_PlayChannel
#define Mix_PlayChannel nl_mix_PlayChannel
#undef Mix_PlayChannelTimed
#define Mix_PlayChannelTimed nl_mix_PlayChannelTimed
#undef Mix_FadeInChannel
#define Mix_FadeInChannel nl_mix_FadeInChannel
#undef Mix_HaltChannel
#define Mix_HaltChannel nl_mix_HaltChannel
#undef Mix_FadeOutChannel
#define Mix_FadeOutChannel nl_mix_FadeOutChannel
#undef Mix_Volume
#define Mix_Volume nl_mix_Volume
#undef Mix_VolumeChunk
#define Mix_VolumeChunk nl_mix_VolumeChunk
#undef Mix_LoadMUS
#define Mix_LoadMUS nl_mix_LoadMUS
#undef Mix_FreeMusic
#define Mix_FreeMusic nl_mix_FreeMusic
#undef Mix_PlayMusic
#define Mix_PlayMusic nl_mix_PlayMusic
#undef Mix_FadeInMusic
#define Mix_FadeInMusic nl_mix_FadeInMusic
#undef Mix_FadeInMusicPos
#define Mix_FadeInMusicPos nl_mix_FadeInMusicPos
#undef Mix_HaltMusic
#define Mix_HaltMusic nl_mix_HaltMusic
#undef Mix_FadeOutMusic
#define Mix_FadeOutMusic nl_mix_FadeOutMusic
#undef Mix_RewindMusic
#define Mix_RewindMusic nl_mix_RewindMusic
#undef Mix_PauseMusic
#define Mix_PauseMusic nl_mix_PauseMusic
#undef Mix_ResumeMusic
#define Mix_ResumeMusic nl_mix_ResumeMusic
#undef Mix_VolumeMusic
#define Mix_VolumeMusic nl_mix_VolumeMusic
#undef Mix_Playing
#define Mix_Playing nl_mix_Playing
#undef Mix_Paused
#define Mix_Paused nl_mix_Paused
#undef Mix_PlayingMusic
#define Mix_PlayingMusic nl_mix_PlayingMusic
#undef Mix_PausedMusic
#define Mix_PausedMusic nl_mix_PausedMusic
#undef Mix_GetNumChannels
#define Mix_GetNumChannels nl_mix_GetNumChannels
#undef Mix_GetError
#define Mix_GetError nl_mix_GetError
#undef Mix_ClearError
#define Mix_ClearError nl_mix_ClearError
#endif
#endif

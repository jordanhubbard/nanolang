# 16.3 SDL_mixer - Audio Playback

**Play music and sound effects with mixing, fading, and volume control.**

SDL_mixer is a multi-channel audio mixer library built on top of SDL. It handles both short sound effects (chunks) and longer background music tracks, with built-in support for WAV, OGG, MP3, FLAC, MIDI, and more.

NanoLang exposes SDL_mixer through `modules/sdl_mixer/sdl_mixer.nano`. Sound effects use the `Mix_Chunk` opaque type; music uses `Mix_Music`.

## Overview

SDL_mixer provides:

- Up to N mixing channels for simultaneous sound effects
- Separate music channel for background tracks
- Volume control per-channel and globally
- Fade-in and fade-out effects
- Loop support (repeat N times, or infinitely with `-1`)
- Playback state queries (is a channel playing? paused?)
- Supported formats: WAV, OGG, MP3, FLAC, MID, MOD, Opus

## Initialization

SDL_mixer requires `SDL_INIT_AUDIO` from SDL, plus its own `Mix_Init` call for format support.

```nano
from "modules/sdl/sdl.nano" import SDL_Init, SDL_Quit, SDL_INIT_AUDIO
from "modules/sdl_mixer/sdl_mixer.nano" import
    Mix_Init, Mix_Quit,
    Mix_OpenAudio, Mix_CloseAudio,
    Mix_AllocateChannels,
    Mix_GetError,
    MIX_INIT_OGG, MIX_INIT_MP3, MIX_DEFAULT_FORMAT

fn init_audio() -> void {
    (SDL_Init SDL_INIT_AUDIO)

    # Initialize format decoders
    let formats: int = (+ MIX_INIT_OGG MIX_INIT_MP3)
    (Mix_Init formats)

    # Open audio device: 44100 Hz, default format, stereo, 2048-byte chunks
    let result: int = (Mix_OpenAudio 44100 MIX_DEFAULT_FORMAT 2 2048)
    if (!= result 0) {
        (print (Mix_GetError))
        return
    }

    # Allocate 16 mixing channels for sound effects
    (Mix_AllocateChannels 16)
}

shadow init_audio { assert true }
```

**Format flags for `Mix_Init`:**

| Constant | Format |
|----------|--------|
| `MIX_INIT_FLAC` | FLAC lossless audio |
| `MIX_INIT_MOD` | Tracker music (MOD, XM, etc.) |
| `MIX_INIT_MP3` | MP3 |
| `MIX_INIT_OGG` | Ogg Vorbis |
| `MIX_INIT_MID` | MIDI |
| `MIX_INIT_OPUS` | Opus |

**Mix_OpenAudio parameters:**

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| frequency | 44100 | Sample rate in Hz |
| format | `MIX_DEFAULT_FORMAT` | Audio format (signed 16-bit LE) |
| channels | 2 | 1=mono, 2=stereo |
| chunksize | 2048 | Buffer size in bytes |

## Sound Effects (Chunks)

### Loading

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_LoadWAV, Mix_FreeChunk, Mix_Chunk

let explosion: Mix_Chunk = (Mix_LoadWAV "sounds/explosion.wav")
let jump: Mix_Chunk = (Mix_LoadWAV "sounds/jump.ogg")
```

`Mix_LoadWAV` accepts WAV files and, if OGG/MP3 support was initialized, other formats too — despite the name.

### Playing

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_PlayChannel

# Play once on first available channel (-1 = auto-select)
(Mix_PlayChannel -1 explosion 0)

# Play 3 times on channel 5
(Mix_PlayChannel 5 jump 2)

# Loop forever on channel 0
(Mix_PlayChannel 0 jump -1)
```

`Mix_PlayChannel` returns the channel number used, or -1 on error.

### Fade-In

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_FadeInChannel

# Fade in over 500ms, play once on auto channel
(Mix_FadeInChannel -1 explosion 0 500)
```

### Stopping

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_HaltChannel, Mix_FadeOutChannel

# Stop immediately
(Mix_HaltChannel 0)

# Fade out over 1 second
(Mix_FadeOutChannel 0 1000)

# Stop all channels
(Mix_HaltChannel -1)
```

### Volume

Volume ranges 0–128 (`MIX_MAX_VOLUME = 128`).

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_Volume, Mix_VolumeChunk, MIX_MAX_VOLUME

# Set channel 0 to 50% volume
(Mix_Volume 0 64)

# Set all channels to max
(Mix_Volume -1 MIX_MAX_VOLUME)

# Set volume for a specific chunk (affects all plays of that chunk)
(Mix_VolumeChunk jump 96)

# Query current volume (pass -1 as volume to just read)
let current: int = (Mix_Volume 0 -1)
```

### Playback State

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_Playing, Mix_Paused

let is_playing: int = (Mix_Playing 0)   # 1 if channel 0 is active
let is_paused: int = (Mix_Paused 0)

# Check all channels
let any_playing: int = (Mix_Playing -1)
```

### Freeing Chunks

```nano
(Mix_FreeChunk explosion)
(Mix_FreeChunk jump)
```

## Background Music

Music is handled separately from sound effect channels. Only one music track plays at a time.

### Loading

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_LoadMUS, Mix_FreeMusic, Mix_Music

let bgm: Mix_Music = (Mix_LoadMUS "music/theme.ogg")
```

Supported music formats include OGG, MP3, FLAC, WAV, MID, MOD, and Opus (depending on what `Mix_Init` enabled).

### Playing

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_PlayMusic, Mix_FadeInMusic

# Play forever (loops = -1)
(Mix_PlayMusic bgm -1)

# Play twice
(Mix_PlayMusic bgm 1)

# Fade in over 2 seconds, loop forever
(Mix_FadeInMusic bgm -1 2000)
```

### Pausing and Resuming

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import
    Mix_PauseMusic, Mix_ResumeMusic, Mix_RewindMusic,
    Mix_PlayingMusic, Mix_PausedMusic

(Mix_PauseMusic)
(Mix_ResumeMusic)
(Mix_RewindMusic)   # restart from beginning

let playing: int = (Mix_PlayingMusic)
let paused: int = (Mix_PausedMusic)
```

### Stopping and Fading Out

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_HaltMusic, Mix_FadeOutMusic

(Mix_HaltMusic)            # immediate stop
(Mix_FadeOutMusic 2000)    # fade out over 2 seconds
```

### Music Volume

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_VolumeMusic, MIX_MAX_VOLUME

(Mix_VolumeMusic 64)                # 50%
(Mix_VolumeMusic MIX_MAX_VOLUME)    # 100%

let vol: int = (Mix_VolumeMusic -1)  # query current
```

### Freeing Music

```nano
(Mix_FreeMusic bgm)
```

## Channel Count

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_GetNumChannels

let num: int = (Mix_GetNumChannels)
```

## Shutdown

```nano
from "modules/sdl_mixer/sdl_mixer.nano" import Mix_CloseAudio, Mix_Quit

(Mix_CloseAudio)
(Mix_Quit)
(SDL_Quit)
```

## Complete Example: Game Audio

A complete audio setup for a simple game with sound effects and music:

```nano
from "modules/sdl/sdl.nano" import SDL_Init, SDL_Quit, SDL_INIT_AUDIO, SDL_Delay
from "modules/sdl_mixer/sdl_mixer.nano" import
    Mix_Init, Mix_Quit,
    Mix_OpenAudio, Mix_CloseAudio,
    Mix_AllocateChannels,
    Mix_LoadWAV, Mix_FreeChunk,
    Mix_LoadMUS, Mix_FreeMusic,
    Mix_PlayChannel, Mix_HaltChannel,
    Mix_PlayMusic, Mix_FadeInMusic, Mix_HaltMusic,
    Mix_Volume, Mix_VolumeMusic,
    Mix_Chunk, Mix_Music,
    MIX_INIT_OGG, MIX_DEFAULT_FORMAT, MIX_MAX_VOLUME

fn main() -> void {
    (SDL_Init SDL_INIT_AUDIO)
    (Mix_Init MIX_INIT_OGG)
    (Mix_OpenAudio 44100 MIX_DEFAULT_FORMAT 2 2048)
    (Mix_AllocateChannels 8)

    # Load assets
    let sfx_coin: Mix_Chunk = (Mix_LoadWAV "sounds/coin.wav")
    let sfx_hit: Mix_Chunk = (Mix_LoadWAV "sounds/hit.ogg")
    let music: Mix_Music = (Mix_LoadMUS "music/level1.ogg")

    # Set volumes
    (Mix_Volume -1 96)              # sound effects at 75%
    (Mix_VolumeMusic MIX_MAX_VOLUME) # music at max

    # Start music with 1s fade-in
    (Mix_FadeInMusic music -1 1000)

    # Simulate game loop events
    (Mix_PlayChannel -1 sfx_coin 0)   # player collected a coin
    (SDL_Delay 500)
    (Mix_PlayChannel -1 sfx_hit 0)    # player took damage

    (SDL_Delay 2000)

    # Cleanup
    (Mix_HaltChannel -1)
    (Mix_HaltMusic)
    (Mix_FreeChunk sfx_coin)
    (Mix_FreeChunk sfx_hit)
    (Mix_FreeMusic music)
    (Mix_CloseAudio)
    (Mix_Quit)
    (SDL_Quit)
}

shadow main { assert true }
```

## Tips

- **Channel -1** in `Mix_PlayChannel` selects the first available channel automatically. This is the most common usage.
- **Loop value** is the number of _additional_ plays after the first. So `0` = play once, `1` = play twice, `-1` = infinite.
- **Music vs. chunks**: Use `Mix_Music` for long tracks (BGM); use `Mix_Chunk` for short sounds (SFX). Mixing multiple music tracks simultaneously is not supported — use additional `Mix_Chunk` channels instead.
- **Error recovery**: Always check `Mix_GetError()` when a load returns null/0.

## API Summary

| Function | Description |
|----------|-------------|
| `Mix_Init(flags)` | Initialize format decoders |
| `Mix_Quit()` | Shutdown SDL_mixer |
| `Mix_OpenAudio(freq, fmt, ch, chunk)` | Open audio device |
| `Mix_CloseAudio()` | Close audio device |
| `Mix_AllocateChannels(n)` | Set number of SFX channels |
| `Mix_GetNumChannels()` | Get current channel count |
| `Mix_GetError()` | Return last error string |
| `Mix_LoadWAV(file)` | Load sound chunk from file |
| `Mix_FreeChunk(chunk)` | Free a sound chunk |
| `Mix_PlayChannel(ch, chunk, loops)` | Play chunk on channel |
| `Mix_FadeInChannel(ch, chunk, loops, ms)` | Play with fade-in |
| `Mix_HaltChannel(ch)` | Stop channel immediately |
| `Mix_FadeOutChannel(ch, ms)` | Fade out and stop channel |
| `Mix_Volume(ch, vol)` | Set/query channel volume |
| `Mix_VolumeChunk(chunk, vol)` | Set chunk volume |
| `Mix_Playing(ch)` | 1 if channel is playing |
| `Mix_Paused(ch)` | 1 if channel is paused |
| `Mix_LoadMUS(file)` | Load music from file |
| `Mix_FreeMusic(music)` | Free music |
| `Mix_PlayMusic(music, loops)` | Play music |
| `Mix_FadeInMusic(music, loops, ms)` | Play music with fade-in |
| `Mix_HaltMusic()` | Stop music immediately |
| `Mix_FadeOutMusic(ms)` | Fade out and stop music |
| `Mix_PauseMusic()` | Pause music |
| `Mix_ResumeMusic()` | Resume music |
| `Mix_RewindMusic()` | Restart music from beginning |
| `Mix_VolumeMusic(vol)` | Set/query music volume |
| `Mix_PlayingMusic()` | 1 if music is playing |
| `Mix_PausedMusic()` | 1 if music is paused |

---

**Previous:** [16.2 SDL_image](sdl_image.html)
**Next:** [16.4 SDL_ttf](sdl_ttf.html)

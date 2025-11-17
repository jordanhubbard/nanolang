# SDL_mixer Module for Nanolang

Complete audio playback module supporting music files (MP3, OGG, WAV, FLAC, MOD) and sound effects.

## Features

- ✅ Music playback (MP3, OGG, WAV, FLAC, MOD, MIDI)
- ✅ Sound effects (WAV format)
- ✅ Volume control (per-channel and global)
- ✅ Fade in/out effects
- ✅ Multiple mixing channels (up to 256)
- ✅ Pause/resume functionality
- ✅ Loop control

## Installation

### macOS
```bash
brew install sdl2_mixer
```

### Ubuntu/Debian
```bash
sudo apt-get install libsdl2-mixer-dev
```

### Windows
Download SDL2_mixer development libraries from [libsdl.org](https://www.libsdl.org/projects/SDL_mixer/)

## Quick Start

### Basic Music Playback

```nano
import "modules/sdl_mixer/sdl_mixer.nano"

fn main() -> int {
    # Initialize
    (Mix_Init 2)  # MOD support
    (Mix_OpenAudio 44100 32784 2 2048)
    
    # Load and play
    let music: int = (Mix_LoadMUS "song.mp3")
    (Mix_PlayMusic music -1)  # Loop forever
    
    # Wait a bit
    (SDL_Delay 5000)
    
    # Cleanup
    (Mix_FreeMusic music)
    (Mix_CloseAudio)
    (Mix_Quit)
    return 0
}
```

### Using Helper Functions

```nano
import "modules/sdl_mixer/sdl_mixer_helpers.nano"

fn main() -> int {
    # Easy initialization
    (init_audio)
    
    # Load and play with helpers
    let music: int = (Mix_LoadMUS "song.ogg")
    (play_music_looped music -1)
    
    # Volume control
    (set_music_volume 64)  # 50% volume
    
    # Cleanup
    (shutdown_audio)
    return 0
}
```

## Module Structure

### Core Module (`sdl_mixer.nano`)
- Direct FFI bindings to SDL_mixer C library
- Low-level control
- All SDL_mixer functions exposed

### Helper Module (`sdl_mixer_helpers.nano`)
- Higher-level convenience functions
- Sensible defaults
- Simplified API

### C Helpers (`sdl_mixer_helpers.c`)
- Type conversion (int64_t ↔ SDL types)
- Pointer management
- FFI compatibility layer

## API Reference

### Initialization

```nano
# Initialize with format support
let flags: int = (+ MIX_INIT_MP3 MIX_INIT_OGG)
(Mix_Init flags)

# Open audio device
(Mix_OpenAudio 44100 MIX_DEFAULT_FORMAT 2 2048)
```

**Parameters:**
- `frequency`: Sample rate (44100 recommended)
- `format`: Audio format (use `MIX_DEFAULT_FORMAT`)
- `channels`: 1 = mono, 2 = stereo
- `chunksize`: Buffer size (2048 recommended)

### Music Functions

```nano
# Load music file
let music: int = (Mix_LoadMUS "song.mp3")

# Play once
(Mix_PlayMusic music 0)

# Loop forever
(Mix_PlayMusic music -1)

# Loop N times (plays N+1 times total)
(Mix_PlayMusic music 5)

# Fade in (3 seconds)
(Mix_FadeInMusic music -1 3000)

# Stop with fade out
(Mix_FadeOutMusic 2000)

# Pause/Resume
(Mix_PauseMusic)
(Mix_ResumeMusic)

# Rewind to start
(Mix_RewindMusic)

# Stop immediately
(Mix_HaltMusic)

# Free when done
(Mix_FreeMusic music)
```

### Sound Effects

```nano
# Load sound
let sound: int = (Mix_LoadWAV "explosion.wav")

# Play on any channel
let channel: int = (Mix_PlayChannel -1 sound 0)

# Play on specific channel
(Mix_PlayChannel 0 sound 0)

# Loop forever
(Mix_PlayChannel -1 sound -1)

# Fade in
(Mix_FadeInChannel -1 sound 0 1000)

# Stop channel
(Mix_HaltChannel channel)

# Free when done
(Mix_FreeChunk sound)
```

### Volume Control

```nano
# Music volume (0-128)
(Mix_VolumeMusic 64)

# Channel volume (0-128)
(Mix_Volume 0 100)  # Channel 0
(Mix_Volume -1 80)  # All channels

# Chunk volume
(Mix_VolumeChunk sound 96)
```

### Status Queries

```nano
# Check if music playing
let playing: int = (Mix_PlayingMusic)
if (== playing 1) {
    (println "Music is playing")
} else {}

# Check if channel playing
let ch_playing: int = (Mix_Playing 0)

# Check if paused
let paused: int = (Mix_PausedMusic)
```

## Supported Formats

### Music Formats
- **MP3**: MPEG audio (requires MIX_INIT_MP3)
- **OGG**: Ogg Vorbis (requires MIX_INIT_OGG)
- **FLAC**: Lossless audio (requires MIX_INIT_FLAC)
- **MOD**: ProTracker modules (requires MIX_INIT_MOD)
- **MIDI**: MIDI files (requires MIX_INIT_MID)
- **OPUS**: Opus codec (requires MIX_INIT_OPUS)
- **WAV**: Always supported (no flag needed)

### Sound Effect Formats
- **WAV**: Only format supported for Mix_Chunk

## Constants

```nano
# Init flags
let MIX_INIT_FLAC: int = 1
let MIX_INIT_MOD: int = 2
let MIX_INIT_MP3: int = 8
let MIX_INIT_OGG: int = 16
let MIX_INIT_MID: int = 32
let MIX_INIT_OPUS: int = 64

# Audio format
let MIX_DEFAULT_FORMAT: int = 32784

# Volume range
let MIX_MAX_VOLUME: int = 128
```

## Complete Example

See `examples/audio_player_sdl.nano` for a full-featured audio player with:
- Music playback
- Sound effects
- Volume control
- Fade effects
- Keyboard controls
- Visual feedback

## Error Handling

```nano
let music: int = (Mix_LoadMUS "song.mp3")
if (== music 0) {
    (println "Failed to load music:")
    (println (Mix_GetError))
} else {
    (Mix_PlayMusic music -1)
}
```

## Best Practices

### 1. Initialize Once
```nano
# Do this once at startup
(init_audio)
```

### 2. Check File Existence
```nano
import "modules/stdio/stdio.nano"

if (file_exists "music.mp3") {
    let music: int = (Mix_LoadMUS "music.mp3")
} else {}
```

### 3. Always Cleanup
```nano
# Free resources
(Mix_FreeMusic music)
(Mix_FreeChunk sound)

# Shutdown
(shutdown_audio)
```

### 4. Handle Errors
```nano
let result: int = (Mix_OpenAudio 44100 MIX_DEFAULT_FORMAT 2 2048)
if (!= result 0) {
    (println (Mix_GetError))
    return 1
} else {}
```

## Common Issues

### "No such audio device"
- Ensure SDL2_mixer is installed
- Check audio device is available
- Try different audio settings

### "File format not supported"
- Initialize with correct flags (MIX_INIT_MP3, etc.)
- Verify file format matches extension
- Check file is not corrupted

### "No channels available"
- Allocate more channels: `(Mix_AllocateChannels 32)`
- Stop unused channels: `(Mix_HaltChannel channel)`

## Performance Tips

- Use OGG or MP3 for music (compressed)
- Use WAV for short sound effects (uncompressed)
- Preallocate channels: `(Mix_AllocateChannels 16)`
- Load frequently-used sounds once
- Use fade effects sparingly (CPU intensive)

## Platform Notes

### macOS
- Uses CoreAudio backend
- Excellent format support
- No additional configuration needed

### Linux
- Uses ALSA or PulseAudio
- May need pulseaudio-utils package
- Check `SDL_AUDIODRIVER` environment variable

### Windows
- Uses DirectSound or WASAPI
- Requires SDL2_mixer.dll
- Place DLL next to executable

## References

- [SDL_mixer Documentation](https://www.libsdl.org/projects/SDL_mixer/docs/)
- [Nanolang Module Tutorial](../../docs/MODULE_CREATION_TUTORIAL.md)
- [SDL Module](../sdl/README.md)

## License

SDL_mixer is licensed under the zlib license. This module provides FFI bindings only.

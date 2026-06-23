# SDL Extension Modules

This document describes the SDL extension modules available in nanolang for building graphical and audio applications.

## Overview

SDL (Simple DirectMedia Layer) is a cross-platform UI engine that provides access to graphics, audio, input, and more. It works natively on macOS, X11/Linux, Windows, and other platforms.

Nanolang provides three SDL-related module families:

1. **SDL Core** (`modules/sdl/`) - Base SDL functionality (windowing, rendering, events, input)
2. **SDL_ttf** (`modules/sdl_ttf/`) - TrueType font rendering
3. **SDL_mixer** (`modules/sdl_mixer/`) - Audio playback and mixing

## Installation Requirements

The nanolang module system will automatically detect and use SDL libraries if they're installed. Install the development packages for your platform:

### Linux (Debian/Ubuntu/Mint)

```bash
# Install SDL2 core (required)
sudo apt-get update
sudo apt-get install libsdl2-dev

# Install SDL2_ttf for text rendering (optional)
sudo apt-get install libsdl2-ttf-dev

# Install SDL2_mixer for audio (optional)
sudo apt-get install libsdl2-mixer-dev

# Install pkg-config (required for module system)
sudo apt-get install pkg-config
```

### Linux (Fedora/RHEL/CentOS)

```bash
# Install SDL2 core (required)
sudo dnf install SDL2-devel

# Install SDL2_ttf for text rendering (optional)
sudo dnf install SDL2_ttf-devel

# Install SDL2_mixer for audio (optional)
sudo dnf install SDL2_mixer-devel

# Install pkg-config (required for module system)
sudo dnf install pkgconf-pkg-config
```

### Linux (Arch/Manjaro)

```bash
# Install SDL2 core (required)
sudo pacman -S sdl2

# Install SDL2_ttf for text rendering (optional)
sudo pacman -S sdl2_ttf

# Install SDL2_mixer for audio (optional)
sudo pacman -S sdl2_mixer

# pkg-config is usually pre-installed
sudo pacman -S pkg-config
```

### macOS (using Homebrew)

```bash
# Install Homebrew if not already installed:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SDL2 core (required)
brew install sdl2

# Install SDL2_ttf for text rendering (optional)
brew install sdl2_ttf

# Install SDL2_mixer for audio (optional)
brew install sdl2_mixer

# pkg-config is usually installed with Homebrew
brew install pkg-config
```

### Verifying Installation

After installation, verify SDL is available:

```bash
# Check SDL2 (should show version and flags)
pkg-config --modversion sdl2
pkg-config --cflags sdl2
pkg-config --libs sdl2

# Check SDL2_ttf
pkg-config --modversion SDL2_ttf

# Check SDL2_mixer
pkg-config --modversion SDL2_mixer
```

If `pkg-config` reports "Package not found", the development package is not installed correctly.

## Module Usage

### SDL Core (`modules/sdl/sdl.nano`)

Basic SDL functionality for creating windows, rendering graphics, and handling events.

```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_helpers/sdl_helpers.nano"

fn main() -> int {
    # Initialize SDL
    (SDL_Init SDL_INIT_VIDEO)
    
    # Create window
    let window: int = (SDL_CreateWindow "My App" 100 100 800 600 SDL_WINDOW_SHOWN)
    let renderer: int = (SDL_CreateRenderer window (- 1) SDL_RENDERER_ACCELERATED)
    
    # Main loop
    let mut running: bool = true
    while running {
        let quit: int = (nl_sdl_poll_event_quit)
        if (== quit 1) {
            set running false
        } else {}
        
        # Clear screen
        (SDL_SetRenderDrawColor renderer 0 0 0 255)
        (SDL_RenderClear renderer)
        
        # Draw something
        (SDL_SetRenderDrawColor renderer 255 0 0 255)
        (nl_sdl_render_fill_rect renderer 100 100 200 200)
        
        # Present
        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
    }
    
    # Cleanup
    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (SDL_Quit)
    
    return 0
}
```

### SDL_ttf (`modules/sdl_ttf/sdl_ttf.nano`)

TrueType font rendering for professional-looking text in SDL applications.

```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_ttf/sdl_ttf.nano"
import "modules/sdl_ttf/sdl_ttf_helpers.nano"

fn main() -> int {
    # Initialize SDL and TTF
    (SDL_Init SDL_INIT_VIDEO)
    (TTF_Init)
    
    let window: int = (SDL_CreateWindow "Text Demo" 100 100 800 600 SDL_WINDOW_SHOWN)
    let renderer: int = (SDL_CreateRenderer window (- 1) SDL_RENDERER_ACCELERATED)
    
    # Load font
    let font: int = (TTF_OpenFont "/System/Library/Fonts/Helvetica.ttc" 24)
    
    # Main loop
    let mut running: bool = true
    while running {
        let quit: int = (nl_sdl_poll_event_quit)
        if (== quit 1) {
            set running false
        } else {}
        
        # Clear screen
        (SDL_SetRenderDrawColor renderer 255 255 255 255)
        (SDL_RenderClear renderer)
        
        # Draw text (black color)
        (draw_text_blended renderer font "Hello, World!" 100 100 0 0 0 255)
        
        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
    }
    
    # Cleanup
    (TTF_CloseFont font)
    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (TTF_Quit)
    (SDL_Quit)
    
    return 0
}
```

**Key Functions:**
- `TTF_Init()` - Initialize TTF library
- `TTF_OpenFont(path, size)` - Load a font
- `TTF_RenderText_Solid()` - Fast, blocky text
- `TTF_RenderText_Blended()` - Smooth, anti-aliased text
- `draw_text_blended()` - Helper to render text directly to screen

**Common Font Paths:**

Linux:
- `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
- `/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf`
- `/usr/share/fonts/TTF/DejaVuSans.ttf` (Arch)
- `~/.local/share/fonts/` (user fonts)

macOS:
- `/System/Library/Fonts/Helvetica.ttc`
- `/Library/Fonts/Arial.ttf`
- `~/Library/Fonts/` (user fonts)

Windows:
- `C:\\Windows\\Fonts\\arial.ttf`
- `C:\\Windows\\Fonts\\calibri.ttf`

Finding fonts on your system:
```bash
# Linux
fc-list | grep -i "ttf"

# macOS
ls /System/Library/Fonts/*.ttc
ls /Library/Fonts/*.ttf

# List available fonts with fc-list (Linux/macOS)
fc-list : family style | sort | uniq
```

### SDL_mixer (`modules/sdl_mixer/sdl_mixer.nano`)

Audio playback and mixing for sound effects and music.

```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_mixer/sdl_mixer.nano"
import "modules/sdl_mixer/sdl_mixer_helpers.nano"

fn main() -> int {
    # Initialize SDL and audio
    (SDL_Init (+ SDL_INIT_VIDEO SDL_INIT_AUDIO))
    (init_audio)
    
    # Load audio files
    let beep: int = (Mix_LoadWAV "beep.wav")
    let music: int = (Mix_LoadMUS "background.ogg")
    
    # Play sound effect
    (play_sound beep)
    
    # Play music looped
    (play_music_looped music (- 1))
    
    # Set volumes (0-128)
    (set_master_volume 64)
    (set_music_volume 32)
    
    # Wait a bit
    (SDL_Delay 5000)
    
    # Cleanup
    (Mix_FreeChunk beep)
    (Mix_FreeMusic music)
    (shutdown_audio)
    (SDL_Quit)
    
    return 0
}
```

**Key Functions:**
- `init_audio()` - Initialize with common defaults
- `Mix_LoadWAV(path)` - Load sound effect (WAV, OGG, FLAC)
- `Mix_LoadMUS(path)` - Load music (MP3, OGG, FLAC, MOD, MIDI)
- `play_sound()` - Play sound effect once
- `play_music_looped()` - Play music looped
- `set_master_volume()` - Set sound effect volume
- `set_music_volume()` - Set music volume

**Supported Formats:**
- Sound Effects: WAV, OGG, FLAC
- Music: MP3, OGG, FLAC, MOD, MIDI, etc.

## Building Examples with SDL Extensions

**With the module system (automatic - recommended):**

The nanolang compiler automatically builds and links modules:

```bash
# Set module path (if not in default location)
export NANO_MODULE_PATH=modules

# Just compile - modules build automatically!
nanoc examples/your_program.nano -o bin/your_program

# Run it
./bin/your_program
```

The compiler will:
1. Detect imported SDL modules
2. Compile C helpers if needed (cached for speed)
3. Link with system libraries automatically
4. Generate the binary

**Manual building (if needed):**

If you need to manually build for debugging:

```bash
# Compile C helpers
gcc -c modules/sdl_ttf/sdl_ttf_helpers.c -o sdl_ttf_helpers.o $(pkg-config --cflags SDL2_ttf)
gcc -c modules/sdl_mixer/sdl_mixer_helpers.c -o sdl_mixer_helpers.o $(pkg-config --cflags SDL2_mixer)

# Transpile nanolang to C
nanoc your_program.nano -o your_program.c --keep-c

# Link everything
gcc your_program.c sdl_ttf_helpers.o sdl_mixer_helpers.o -o your_program \
    $(pkg-config --libs sdl2 SDL2_ttf SDL2_mixer)
```

## Example Applications

See the `examples/` directory for complete working examples:

- **checkers.nano** - Full checkers game with SDL rendering
- **boids_sdl.nano** - Flocking simulation with vector graphics
- **music_sequencer_sdl.nano** - ProTracker-style music sequencer (can use SDL_mixer for playback)
- **falling_sand_sdl.nano** - Particle physics sandbox
- **terrain_explorer_sdl.nano** - Procedural terrain with fog of war
- **particles_sdl.nano** - Particle explosion demo
- **life_sdl.nano** - Conway's Game of Life
- **snake_sdl.nano** - Snake game with AI
- **maze_sdl.nano** - Maze generation and solving

## Troubleshooting

### "SDL2_ttf/SDL_ttf.h: No such file or directory"

Install SDL2_ttf development package:
```bash
# Linux (Debian/Ubuntu)
sudo apt-get install libsdl2-ttf-dev

# Linux (Fedora/RHEL)
sudo dnf install SDL2_ttf-devel

# macOS
brew install sdl2_ttf
```

### "Package 'SDL2_ttf' was not found"

pkg-config cannot find the SDL2_ttf package. Install it:
```bash
# Linux (Debian/Ubuntu)
sudo apt-get install libsdl2-ttf-dev pkg-config

# Linux (Fedora/RHEL)
sudo dnf install SDL2_ttf-devel pkgconf-pkg-config

# macOS
brew install sdl2_ttf pkg-config
```

### "undefined reference to TTF_Init" or "undefined symbol: TTF_Init"

The linker cannot find SDL2_ttf. Verify it's installed:
```bash
pkg-config --libs SDL2_ttf
```

If the command works, rebuild your program to pick up the correct link flags.

### "Mix_OpenAudio failed"

Audio device issues:
- Check that your audio device is working
- Ensure no other application is using the audio device exclusively
- On Linux, verify PulseAudio or ALSA is working:
  ```bash
  speaker-test -t sine -f 1000 -l 1
  ```

### Font file not found

Use absolute paths to fonts or check common directories:
```bash
# Find fonts on Linux
fc-list | grep -i dejavu

# Find fonts on macOS  
ls /System/Library/Fonts/*.ttc

# Test if font exists
ls -l /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

### Module not rebuilding automatically

Clear the build cache:
```bash
rm -rf modules/*/.build/
```

### Verbose build output for debugging

```bash
NANO_VERBOSE_BUILD=1 nanoc my_program.nano -o my_program
```

This shows all compilation and linking commands.

## Performance Tips

1. **Cache rendered text** - Render text to textures once, reuse them
2. **Use solid rendering for HUD** - Faster than blended, good for simple text
3. **Limit audio channels** - 16 channels is usually sufficient
4. **Preload audio files** - Load all sounds at startup, not during gameplay
5. **Use appropriate audio formats** - OGG for music, WAV for short sound effects

## API Reference

For complete API documentation, see:
- `modules/sdl_ttf/sdl_ttf.nano` - All SDL_ttf FFI functions
- `modules/sdl_mixer/sdl_mixer.nano` - All SDL_mixer FFI functions
- `modules/sdl_ttf/sdl_ttf_helpers.nano` - Convenience wrappers for text rendering
- `modules/sdl_mixer/sdl_mixer_helpers.nano` - Convenience wrappers for audio playback


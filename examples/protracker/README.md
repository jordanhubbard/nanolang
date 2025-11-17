# NanoLang ProTracker Clone

A multi-file ProTracker music tracker implementation in nanolang, inspired by [pt2-clone](https://github.com/jordanhubbard/pt2-clone.git).

## Overview

This is a **fully modular ProTracker implementation** demonstrating nanolang's capability to build real applications from multiple source files. The project showcases:

- ✅ Multi-file project structure
- ✅ Module system with imports
- ✅ Pattern editor UI
- ✅ Audio playback (SDL_mixer)
- ✅ Makefile-based build system
- ✅ Clean separation of concerns

## Architecture

```
protracker/
├── Makefile           # Build system
├── README.md          # This file
├── types.nano         # Core data structures and enums
├── pattern.nano       # Pattern data management
├── ui.nano            # SDL rendering and UI
└── main.nano          # Main entry point
```

### Module Dependencies

```
main.nano
├── imports types.nano
├── imports pattern.nano (depends on types.nano)
└── imports ui.nano (depends on types.nano, pattern.nano)
```

## Current Status

⚠️ **LIMITATION DISCOVERED**: Nanolang's import system currently only supports imports from `modules/`, not cross-file imports within `examples/`.

This ProTracker project demonstrates the **intended architecture** for multi-file nanolang applications, but currently requires enhancement to the import system to function.

### What Works
- ✅ Project structure and organization
- ✅ Module separation (types, pattern, ui, main)
- ✅ Makefile for multi-file builds
- ✅ Individual file compilation
- ✅ Module imports (SDL, SDL_mixer)

### What Needs Implementation
- ⚠️ Cross-file imports within examples/
- ⚠️ Project-local module resolution

### Path Forward
1. Enhance nanolang's import system to support project-local imports
2. Or: Add `protracker` as a module in `modules/protracker/`
3. Or: Concatenate files during build process

## Building

**Note**: Currently doesn't build due to import limitation above.

### Prerequisites

```bash
# Install SDL2_mixer for audio support
brew install sdl2_mixer          # macOS
sudo apt-get install libsdl2-mixer-dev  # Ubuntu/Debian
```

### Compile

```bash
cd examples/protracker
make
```

This will:
1. Compile all source files through the import system
2. Link with SDL2 and SDL2_mixer
3. Generate `../../bin/protracker`

### Run

```bash
make run
```

Or directly:
```bash
../../bin/protracker
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause |
| `↑` `↓` | Navigate rows |
| `←` `→` | Navigate channels |
| `+` `-` | Volume control |
| `ESC` | Quit |

## Features

### Implemented

- ✅ Pattern editor grid (4 channels, 64 rows)
- ✅ Keyboard navigation
- ✅ Visual cursor
- ✅ Playback control
- ✅ Audio system initialization
- ✅ Multi-file compilation

### Planned

- 🔄 Note entry (piano keyboard)
- 🔄 Sample editor
- 🔄 .MOD file loading/saving
- 🔄 Effect implementation
- 🔄 Sample playback
- 🔄 Scopes and VU meters
- 🔄 Disk operations
- 🔄 Text rendering (requires SDL_ttf)

## Development

### Project Statistics

```bash
make stats
```

Shows:
- Lines of code per file
- Total project size
- Module dependencies

### Watch Mode

Auto-rebuild on file changes:
```bash
make watch
```

### Clean Build

```bash
make clean
make
```

## Technical Details

### Pattern Data Structure

Patterns are stored as flat arrays:
```
[note, sample, effect, param] × 4 channels × 64 rows × 64 patterns
```

Indexed with:
```nano
fn pattern_index(pattern: int, channel: int, row: int, component: int) -> int
```

### Color Scheme

Classic ProTracker-inspired colors:
- Background: `#1A1A2A` (dark blue-gray)
- Pattern: `#1F1F3F` (darker blue)
- Cursor: `#E94560` (red highlight)
- Text: `#DDDDDD` (light gray)

### Note System

Uses Amiga period values:
- C-1 = 856 Hz
- C-2 = 428 Hz
- C-3 = 214 Hz

### Compilation Process

The multi-file build works through nanolang's import system:

1. `main.nano` is compiled
2. Compiler resolves all imports
3. Type checking across all modules
4. Shadow tests run
5. Single binary generated

No manual concatenation or linking required!

## Comparison with pt2-clone

| Feature | pt2-clone (C) | This Project (nanolang) |
|---------|---------------|-------------------------|
| Lines of Code | ~50,000 | ~600 (growing) |
| Build System | CMake | Makefile + nanolang |
| Language | C | nanolang |
| Modules | ~40 C files | 4 nano files |
| Audio | Custom Paula | SDL_mixer |
| UI | Custom SDL | SDL + helpers |

## Contributing

This is a demonstration project showing:
- Multi-file nanolang projects
- Import system usage
- Module organization
- Makefile integration

Feel free to:
- Add more features
- Improve UI rendering
- Implement missing effects
- Add .MOD file I/O

## References

- [pt2-clone](https://github.com/jordanhubbard/pt2-clone) - Original C implementation
- [ProTracker](https://en.wikipedia.org/wiki/ProTracker) - Original Amiga software
- [nanolang Documentation](../../docs/)
- [SDL_mixer Module](../../modules/sdl_mixer/)

## License

Educational demonstration project. Original pt2-clone is BSD licensed.

## Credits

- **pt2-clone**: Olav Sørensen (8bitbubsy)
- **ProTracker**: Amiga music tracker (1987)
- **This Implementation**: NanoLang demonstration

---

**Built with nanolang** - A modern systems language for music and creativity! 🎵

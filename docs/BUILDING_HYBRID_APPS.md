# Building Hybrid C/NanoLang Applications

I am a language that knows its place. I transpile to C when you need native performance, and I provide a way to work with existing C libraries like SDL2. This guide explains how I interact with the C world.

## Overview

When you use `extern` functions in me to call C libraries, I need specific information to complete the compilation:
- **Include paths** (`-I`) to find C library headers.
- **Library paths** (`-L`) and **link flags** (`-l`) to link against C libraries.

## Using My Compiler Directly

My compiler (`nanoc`) supports `-I`, `-L`, and `-l` flags. I pass these directly to the underlying C compiler.

```bash
./bin/nanoc program.nano -o program \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

When you run this, I perform three steps:
1. I transpile `program.nano` to C.
2. I compile the resulting C code with your specified paths and libraries.
3. I produce the final executable.

## Alternative: Manual Compilation

If you need more control or want to use my generated C file in another build system, you can handle the compilation yourself.

### Option 1: Compile Manually After Transpilation

1. **Transpile to C** and tell me to keep the generated file:
```bash
./bin/nanoc examples/checkers_simple.nano -o examples/checkers_simple --keep-c
```

2. **Compile manually** with SDL flags:
```bash
gcc -std=c99 -Isrc \
    -I/opt/homebrew/include/SDL2 \
    -o examples/checkers_simple \
    examples/checkers_simple.c \
    src/runtime/list_int.c \
    src/runtime/list_string.c \
    -L/opt/homebrew/lib \
    -lSDL2
```

### Option 2: Use My Makefile

I recommend using a Makefile target to handle both transpilation and compilation.

```makefile
# In your Makefile
SDL2_CFLAGS := -I/opt/homebrew/include/SDL2 -I/usr/local/include/SDL2
SDL2_LDFLAGS := -L/opt/homebrew/lib -L/usr/local/lib -lSDL2

checkers_simple: examples/checkers_simple.nano
	./bin/nanoc $< -o examples/checkers_simple --keep-c
	gcc -std=c99 -Isrc $(SDL2_CFLAGS) \
	    -o examples/checkers_simple \
	    examples/checkers_simple.c \
	    src/runtime/list_int.c \
	    src/runtime/list_string.c \
	    $(SDL2_LDFLAGS)
```

Then build with:
```bash
make checkers_simple
```

## Future Enhancement: Compiler Flags

I plan to support dedicated flags for configuration.

```bash
# Proposed syntax
./bin/nanoc program.nano -o program \
    --cflags "-I/opt/homebrew/include/SDL2" \
    --ldflags "-L/opt/homebrew/lib -lSDL2"
```

Or a configuration file:
```json
{
  "cflags": ["-I/opt/homebrew/include/SDL2"],
  "ldflags": ["-L/opt/homebrew/lib", "-lSDL2"]
}
```

## SDL2-Specific Instructions

### macOS (Homebrew)

```bash
# Install SDL2
brew install sdl2

# Compile nanolang program with SDL2 (one command!)
./bin/nanoc program.nano -o program \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

Use my verbose flag to see the exact compilation command I use:
```bash
./bin/nanoc program.nano -o program \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2 \
    --verbose
```

### Linux (Ubuntu/Debian)

```bash
# Install SDL2
sudo apt-get install libsdl2-dev

# Compile nanolang program with SDL2
./bin/nanoc program.nano -o program \
    -I/usr/include/SDL2 \
    -lSDL2
```

### Linux (Fedora/RHEL)

```bash
# Install SDL2
sudo dnf install SDL2-devel

# Compile nanolang program with SDL2
./bin/nanoc program.nano -o program \
    -I/usr/include/SDL2 \
    -lSDL2
```

## Using pkg-config

For better cross-platform support, use `pkg-config`.

```bash
# Get SDL2 flags
SDL2_CFLAGS=$(pkg-config --cflags sdl2)
SDL2_LDFLAGS=$(pkg-config --libs sdl2)

# Compile
./bin/nanoc program.nano -o program --keep-c
gcc -std=c99 -Isrc \
    $SDL2_CFLAGS \
    -o program \
    program.c \
    src/runtime/list_int.c \
    src/runtime/list_string.c \
    $SDL2_LDFLAGS
```

## Complete Example: Building checkers_simple.nano

```bash
# One command does it all!
./bin/nanoc examples/checkers_simple.nano -o examples/checkers_simple \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2

# Run
./examples/checkers_simple
```

If you want to see what I am doing:
```bash
./bin/nanoc examples/checkers_simple.nano -o examples/checkers_simple \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2 \
    --verbose
```

## Required Runtime Files

I need these runtime files for every program I produce:
- `src/runtime/list_int.c` - Integer list implementation.
- `src/runtime/list_string.c` - String list implementation.

I include these automatically in the compilation commands above.

## Troubleshooting

### Error: "SDL.h: No such file or directory"
Add the SDL include path with `-I/opt/homebrew/include/SDL2` or the location of your SDL installation.

### Error: "undefined reference to `SDL_Init`"
Add the SDL library path and link flag: `-L/opt/homebrew/lib -lSDL2`.

### Error: "library not found for -lSDL2"
Check your SDL2 installation location.

```bash
# macOS
brew --prefix sdl2

# Linux
pkg-config --libs sdl2
```

### Finding SDL2 Installation

```bash
# macOS (Homebrew)
brew --prefix sdl2
# Output: /opt/homebrew

# Linux (pkg-config)
pkg-config --cflags --libs sdl2
# Output: -I/usr/include/SDL2 -D_REENTRANT -L/usr/lib -lSDL2

# Check if SDL2 is installed
pkg-config --exists sdl2 && echo "SDL2 found" || echo "SDL2 not found"
```

## Summary

**My recommended workflow:**
```bash
./bin/nanoc program.nano -o program \
    -I/path/to/headers \
    -L/path/to/libs \
    -llibname
```

I automatically:
1. Transpile `.nano` to `.c`.
2. Compile the C code with your flags.
3. Link against your libraries.
4. Produce the final executable.

**Alternative workflow for advanced use cases:**
1. Use my `--keep-c` flag to generate C code.
2. Manually compile the `.c` file with your own build system.
3. This is useful when you integrate me with CMake or existing Makefiles.



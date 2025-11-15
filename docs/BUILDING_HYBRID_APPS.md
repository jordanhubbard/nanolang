# Building Hybrid C/nanolang Applications

This guide explains how to build nanolang applications that use external C libraries like SDL2.

## Overview

nanolang programs that use `extern` functions to call C libraries (like SDL2) need special compilation flags:
- **Include paths** (`-I`) to find C library headers
- **Library paths** (`-L`) and **link flags** (`-l`) to link against C libraries

## Using the Compiler Directly (Recommended)

The nanolang compiler (`nanoc`) now supports `-I`, `-L`, and `-l` flags that are passed through to the C compiler:

```bash
./bin/nanoc program.nano -o program \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

This will:
1. Transpile `program.nano` to C
2. Compile the C code with the specified include paths, library paths, and libraries
3. Produce the final executable

## Alternative: Manual Compilation

If you need more control or want to use the generated C file in another build system, you can compile manually:

### Option 1: Compile Manually After Transpilation

1. **Transpile to C** (keep the generated C file):
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

### Option 2: Use Makefile (Recommended)

Create a Makefile target that handles both transpilation and compilation:

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

A better long-term solution would be to add flags to `nanoc`:

```bash
# Proposed syntax
./bin/nanoc program.nano -o program \
    --cflags "-I/opt/homebrew/include/SDL2" \
    --ldflags "-L/opt/homebrew/lib -lSDL2"
```

Or use a configuration file:
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

Or with verbose output to see the compilation command:
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

## Using pkg-config (Cross-Platform)

For better cross-platform support, use `pkg-config`:

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

Or if you want to see what's happening:
```bash
./bin/nanoc examples/checkers_simple.nano -o examples/checkers_simple \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2 \
    --verbose
```

## Required Runtime Files

All nanolang programs need these runtime files:
- `src/runtime/list_int.c` - Integer list implementation
- `src/runtime/list_string.c` - String list implementation

These are automatically included in the compilation command above.

## Troubleshooting

### Error: "SDL.h: No such file or directory"
- **Solution**: Add SDL include path with `-I/opt/homebrew/include/SDL2` (or your SDL install location)

### Error: "undefined reference to `SDL_Init`"
- **Solution**: Add SDL library path and link flag: `-L/opt/homebrew/lib -lSDL2`

### Error: "library not found for -lSDL2"
- **Solution**: Check SDL2 installation location:
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

**Recommended workflow:**
```bash
./bin/nanoc program.nano -o program \
    -I/path/to/headers \
    -L/path/to/libs \
    -llibname
```

The compiler automatically:
1. Transpiles `.nano` → `.c`
2. Compiles the C code with your specified flags
3. Links against your specified libraries
4. Produces the final executable

**Alternative workflow** (for advanced use cases):
1. Use `nanoc` with `--keep-c` to generate C code
2. Manually compile the `.c` file with your own build system
3. This is useful for integration with CMake, Makefiles, etc.


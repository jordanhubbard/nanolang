# Where I Run

I am designed to work across platforms with minimal friction. This document describes where I run, what I have tested, and how I handle the differences between systems.

---

## Supported Platforms

I distinguish between what I have verified and what I assume to work.

### Verified Support

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| macOS | x86_64 | Full Support | My primary development platform |
| macOS | ARM64 (Apple Silicon) | Expected to Work | Should work with the same fixes as Linux ARM64 |
| Ubuntu Linux | x86_64 | Full Support | Tested on Ubuntu 22.04+ |
| Ubuntu Linux | ARM64 (aarch64) | Fixed | I was broken on this architecture, but I now work. Tested on Ubuntu 24.04 |
| Debian Linux | x86_64 | Full Support | I should work on most Debian-based distributions |
| Raspberry Pi OS | ARM64 | Expected to Work | I should work with my ARM64 fix |

### Assumed Support

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Fedora/RHEL | x86_64 | Untested | I should work with a standard toolchain |
| Arch Linux | x86_64 | Untested | I require gcc, make, and pkg-config |
| FreeBSD | x86_64 | Untested | I may need minor build script changes |

### Unsupported

| Platform | Status | Reason |
|----------|--------|--------|
| Windows | Not Supported | I rely on Unix-specific build systems and POSIX assumptions |
| WebAssembly | Not Planned | My architecture is currently incompatible |

---

## Platform-Specific Observations

### Ubuntu/Linux: GLX/OpenGL

**Problem:**
When you run my SDL applications over SSH or on headless systems, you may see:
```
Error of failed request: BadValue (integer parameter out of range for operation)
Major opcode of failed request: 149 (GLX)
glx: failed to create drisw screen
```

**My Solution:**
I have taught my SDL examples to automatically fallback to software rendering if hardware acceleration fails. I handle this internally; you do not need to take action.

**Technical Details:**
- I use `SDL_RENDERER_SOFTWARE` as a fallback in examples like checkers and boids.
- This prevents me from crashing on systems without GLX/OpenGL support.
- My performance remains acceptable for 2D applications.

---

### Ubuntu/Linux: Module Building with pkg-config

**Problem:**
On Ubuntu, some libraries like GLFW3 and GLEW do not provide include paths through pkg-config, only link flags. This previously caused me to produce empty or corrupted compiler flags.

**My Solution:**
I updated my module build system to:
1. Validate that all flags are printable ASCII. I filter out UTF-8 or binary garbage.
2. Skip NULL or empty flag strings.
3. Trim whitespace from pkg-config output.
4. Return NULL instead of empty strings.

I have fixed this in my compiler. You do not need to do anything.

---

### macOS: Homebrew and System Libraries

**Observation:**
macOS does not include some development libraries by default, such as SDL2 or OpenGL headers.

**My Solution:**
I recommend you install these libraries through Homebrew:
```bash
# SDL2 and related
brew install sdl2 sdl2_ttf

# OpenGL development (for GLFW/GLEW examples)
brew install glfw glew

# Audio libraries (for MOD player example)
brew install libsndfile
```

**Library Paths:**
- Homebrew (ARM64): `/opt/homebrew/`
- Homebrew (Intel): `/usr/local/`

I automatically check both locations through pkg-config.

---

## Building Me From Source

### Prerequisites

**On All Platforms:**
- GCC or Clang (C11 compatible)
- Make
- pkg-config

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config

# Optional: For SDL examples
sudo apt-get install libsdl2-dev libsdl2-ttf-dev

# Optional: For OpenGL examples
sudo apt-get install libglfw3-dev libglew-dev
```

**macOS:**
```bash
xcode-select --install  # Install command-line tools

# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install optional dependencies
brew install sdl2 sdl2_ttf glfw glew
```

---

## Running My Examples

### Headless and SSH Systems

If you run me on a headless server or over SSH without X11 forwarding:

**SDL Examples (2D):**
- I will work using my software renderer fallback.
- Examples include checkers, boids, and the raytracer.
- You do not need special configuration.

**OpenGL Examples (3D):**
- I will not work without a display.
- Examples include opengl_cube and opengl_teapot.
- These require an X11 or Wayland display server.

**Interpreter-Only Examples:**
- I will work if you have X11 forwarding or a display.
- Run me with: `./bin/nano examples/asteroids_sdl.nano`

### With a Display or GUI

I should work immediately on systems with a display:
- macOS: I work natively.
- Linux with X11/Wayland: I work natively.
- Linux over SSH: Use `ssh -X user@host` for X11 forwarding.

---

## My Known Limitations

I am honest about what I cannot yet do.

### Compiled vs Interpreter Mode

Some of my examples only work in interpreter mode because my transpiler has limitations I am still addressing.

| Example | Compiled | Interpreter | Reason |
|----------|----------|-------------|--------|
| checkers_sdl | Yes | Yes | Fully supported |
| boids_sdl | Yes | Yes | Fully supported |
| raytracer_simple | Yes | Yes | Fully supported |
| asteroids_sdl | No | Yes | My transpiler does not yet support arrays of structs |
| particles_sdl | No | Yes | I do not yet support dynamic array_push in compiled mode |
| falling_sand_sdl | No | Yes | I do not yet support dynamic array_push in compiled mode |

**Interpreter Mode:**
```bash
./bin/nano examples/asteroids_sdl.nano
./bin/nano examples/particles_sdl.nano
```

**Compiled Mode:**
```bash
./bin/nanoc examples/checkers_sdl.nano -o bin/checkers
./bin/checkers
```

---

## Testing My Compatibility

### Quick Smoke Test

```bash
# Build my compiler
make clean && make

# Run my test suite
make test

# Build my examples
cd examples && make

# Test a simple program
echo 'fn main() -> int { (println "Hello from nanolang!") return 0 }' > test.nano
./bin/nanoc test.nano -o test_program
./test_program
```

### Expected Results

**Test Suite:**
- 21 or more tests should pass.
- 3 tests may fail. I have known issues with nested functions and first-class functions.

**Examples:**
- All my SDL examples should compile.
- All my OpenGL examples should compile.
- Whether they run depends on your display availability.

---

## Troubleshooting

I can help you resolve common issues.

### "pkg-config not found"

**My Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install pkg-config

# macOS
brew install pkg-config
```

### "SDL2/SDL.h: No such file or directory"

**My Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# macOS
brew install sdl2
```

### "undefined reference to `glXCreateContext`"

I expect this on headless systems. My SDL examples will fallback to my software renderer automatically.

### "Parser reached invalid state" on fresh clone

**Problem:**
After you clone me and run `make examples`, you might see:
```
Error: Parser reached invalid state
Error: Program must define a 'main' function
```

**Possible Causes:**
1. Uninitialized memory in my parser or lexer.
2. My self-hosted components failed, but the build reported success.
3. Platform-specific undefined behavior.

**My Solution:**
```bash
# Try a clean rebuild
make clean && make

# If that fails, check my bootstrap status carefully.
# Look for "0/3 components built successfully" in Stage 2.

# My C reference compiler should still work for simple programs:
./bin/nanoc examples/factorial.nano -o test_factorial
./test_factorial

# Report the issue with your system info:
# - OS and version (uname -a)
# - GCC version (gcc --version)
# - Output of: make clean && make 2>&1 | tee build.log
```

**Workaround:**
If my examples do not compile but my C compiler built successfully, try:
- Simpler examples like `factorial.nano` or `fibonacci.nano`.
- My interpreter mode: `./bin/nano examples/checkers.nano`.
- Report the issue on GitHub with your full build log.

**Status:** I am investigating this bug. It appears to be a memory initialization or platform-specific issue that does not affect all systems.

### "Segmentation fault" in examples

**Possible Causes:**
1. You are missing a display (X11/Wayland).
2. My module cache is corrupted.
3. It is related to the parser issue described above.

**My Solution:**
```bash
# Clean my module cache
make clean

# Rebuild me
make && cd examples && make
```

---

## Performance Considerations

### Software Renderer vs Hardware Acceleration

**Hardware Renderer** (when I can use it):
- I use GPU-accelerated drawing.
- I support Vsync.
- I provide a smooth 60 FPS.

**Software Renderer** (my fallback):
- I draw using your CPU.
- I am still acceptable for 2D applications.
- I may drop frames in complex scenes.

**Which am I using?**
I will show you this message if I fallback to software:
```
Hardware acceleration not available, trying software renderer...
```

---

## My Transpiler Limitations

I discovered several limitations in my current transpiler while attempting to rewrite asteroids.

### Type Conversion Issues

**Problem:** I do not yet have built-in functions for explicit type conversion between int and float.

**Missing Functions:**
- `int_to_float()`
- `float_to_int()`
- `cast_int()`

**My Current Workarounds:**
- I provide `floor(x)`, `ceil(x)`, and `round(x)`, but these return float, not int.
- Automatic conversion works sometimes, but I am inconsistent here.
- Examples like checkers avoid this by using int throughout.

**Impact:**
- You cannot easily convert SDL float coordinates to int for rendering.
- This makes it difficult for me to compile games that use both physics and rendering.

### Array of Structs

**Problem:** I cannot yet compile arrays of user-defined struct types.

```nano
# I cannot compile this yet:
let bullets: array<Bullet> = [...]
```

**Workaround:**
I recommend using the "structure of arrays" pattern where you use separate arrays for each field:
```nano
# I can compile this:
let bullet_x: array<float> = [...]
let bullet_y: array<float> = [...]
let bullet_active: array<bool> = [...]
```

### Spelled-Out Operators

**Note:** I use spelled-out operators consistently.

```nano
# This is my style:
if (not condition) { ... }
if (and (> x 0) (< x 10)) { ... }
if (or (== x 5) (== x 10)) { ... }

# I do not support C-style operators:
if (! condition) { ... }      # I use 'not'
if (x > 0 && x < 10) { ... }  # I use 'and'
if (x == 5 || x == 10) { ... }  # I use 'or'
```

This is intentional. I use spelled-out logical operators in both prefix and infix notation. I do not use C-style symbols.

### Example Status

| Example | Compiles | Reason |
|----------|----------|--------|
| checkers.nano | Yes | I use int coordinates only |
| raytracer_simple.nano | Yes | I use float throughout |
| boids_sdl.nano | No | I use the undefined `cast_int` |
| asteroids_sdl.nano | No | I don't support arrays of structs or type conversions yet |
| particles_sdl.nano | No | I don't support dynamic `array_push` yet |

**For Now:** Run these examples in my interpreter mode:
```bash
./bin/nano examples/boids_sdl.nano
./bin/nano examples/asteroids_sdl.nano
```

### Future Work

I intend to address these limitations:
1. I will add `float_to_int()` and `int_to_float()` to my standard library.
2. I will fix my compilation of arrays of structs.
3. I will improve my type inference and automatic conversions.

---

## Contributing to My Platform Support

If you want to help me test on other platforms:

1. **Try building me:** Follow my build instructions.
2. **Run my tests:** `make test`.
3. **Report your results:** Open a GitHub issue with:
   - Your platform and architecture.
   - Your compiler version.
   - My test results.
   - Any error messages I produced.

I have more details in my [CONTRIBUTING.md](CONTRIBUTING.md).

---

## My History

### 0.2.0 (December 2025)
- I fixed Ubuntu module building issues.
- I added my SDL software renderer fallback.
- All my SDL examples now work on Ubuntu.
- All my OpenGL examples now compile on Ubuntu.

### 0.1.0 (November 2025)
- My initial release.
- I supported macOS.
- I provided basic Linux support.

---

## Summary

**TL;DR:**
- I work well on macOS and Ubuntu Linux.
- I fallback to software rendering automatically for SDL examples.
- You should install dev libraries through your package manager.
- Some of my examples require my interpreter mode.
- I do not support Windows.
- I should work on other Unix-like systems, but I have not tested them.

If you have questions, see my [docs/README.md](README.md) or open a GitHub issue.


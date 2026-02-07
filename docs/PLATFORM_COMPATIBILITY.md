# Platform Compatibility Guide

**Last Updated:** December 7, 2025  
**Version:** 0.2.0

---

## Overview

Nanolang is designed to work cross-platform with minimal friction. This document describes platform-specific considerations, known issues, and solutions.

## Supported Platforms

### ✅ Fully Tested

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| macOS | x86_64 | ✅ Full Support | Primary development platform |
| macOS | ARM64 (Apple Silicon) | ⚠️ Expected to Work | Should work (same fix as Linux ARM64) |
| Ubuntu Linux | x86_64 | ✅ Full Support | Tested on Ubuntu 22.04+ |
| Ubuntu Linux | ARM64 (aarch64) | ✅ **FIXED!** | Was broken, now works (tested on Ubuntu 24.04) |
| Debian Linux | x86_64 | ✅ Full Support | Should work on most Debian-based distros |
| Raspberry Pi OS | ARM64 | ✅ Expected to Work | Should work with ARM64 fix

### 🟡 Expected to Work

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Fedora/RHEL | x86_64 | 🟡 Untested | Should work with standard toolchain |
| Arch Linux | x86_64 | 🟡 Untested | Requires gcc, make, pkg-config |
| FreeBSD | x86_64 | 🟡 Untested | May need minor build script changes |

### ❌ Not Supported

| Platform | Status | Reason |
|----------|--------|--------|
| Windows | ❌ Not Supported | Unix-specific build system, POSIX assumptions |
| WebAssembly | ❌ Not Planned | Architecture incompatibility |

---

## Platform-Specific Issues & Solutions

### Ubuntu/Linux: GLX/OpenGL Errors

**Problem:**
When running SDL applications over SSH or on headless systems, you may see:
```
Error of failed request: BadValue (integer parameter out of range for operation)
Major opcode of failed request: 149 (GLX)
glx: failed to create drisw screen
```

**Solution:**
SDL examples now automatically fallback to software rendering if hardware acceleration fails. No action needed - this is handled by the compiler.

**Technical Details:**
- Checkers, boids, and other SDL apps use `SDL_RENDERER_SOFTWARE` as fallback
- Prevents crashes on systems without GLX/OpenGL support
- Performance is still acceptable for 2D applications

---

### Ubuntu/Linux: Module Building with pkg-config

**Problem:**
On Ubuntu, some libraries (like GLFW3, GLEW) don't provide include paths via pkg-config, only link flags. This previously caused empty or corrupted compiler flags.

**Solution:**
The module build system now:
1. Validates all flags are printable ASCII (filters UTF-8/binary garbage)
2. Skips NULL or empty flag strings
3. Trims whitespace from pkg-config output
4. Returns NULL instead of empty strings

**No user action required** - this is fixed in the compiler.

---

### macOS: Homebrew vs System Libraries

**Issue:**
macOS doesn't include some development libraries by default (SDL2, OpenGL headers, etc.)

**Solution:**
Install libraries via Homebrew:
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

The compiler automatically checks both locations via pkg-config.

---

## Building from Source

### Prerequisites

**All Platforms:**
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

## Running Examples

### Headless/SSH Systems

If you're running on a headless server or over SSH without X11 forwarding:

**SDL Examples (2D):**
- ✅ **Will work** with software renderer fallback
- Examples: checkers, boids, raytracer
- No special configuration needed

**OpenGL Examples (3D):**
- ❌ **Will not work** without display
- Examples: opengl_cube, opengl_teapot
- Requires X11/Wayland display server

**Interpreter-Only Examples:**
- ✅ **Will work** if you have X11 forwarding or display
- Run with: `./bin/nano examples/asteroids_sdl.nano`

### With Display/GUI

All examples should work out of the box on systems with a display:
- macOS: Works natively
- Linux with X11/Wayland: Works natively
- Linux over SSH: Use `ssh -X user@host` for X11 forwarding

---

## Known Limitations

### Compiled vs Interpreter Mode

Some examples only work in **interpreter mode** due to transpiler limitations:

| Example | Compiled | Interpreter | Reason |
|---------|----------|-------------|--------|
| checkers_sdl | ✅ | ✅ | Fully supported |
| boids_sdl | ✅ | ✅ | Fully supported |
| raytracer_simple | ✅ | ✅ | Fully supported |
| asteroids_sdl | ❌ | ✅ | Arrays of structs not supported in transpiler |
| particles_sdl | ❌ | ✅ | Uses dynamic array_push |
| falling_sand_sdl | ❌ | ✅ | Uses dynamic array_push |

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

## Testing Platform Compatibility

### Quick Smoke Test

```bash
# Build compiler
make clean && make

# Run test suite
make test

# Build examples
cd examples && make

# Test a simple program
echo 'fn main() -> int { (println "Hello from nanolang!") return 0 }' > test.nano
./bin/nanoc test.nano -o test_program
./test_program
```

### Expected Results

**Test Suite:**
- 21+ tests should pass
- 3 tests may fail (known issues with nested functions, first-class functions)

**Examples:**
- All SDL examples should compile
- All OpenGL examples should compile
- Runtime success depends on display availability

---

## Troubleshooting

### "pkg-config not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install pkg-config

# macOS
brew install pkg-config
```

### "SDL2/SDL.h: No such file or directory"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# macOS
brew install sdl2
```

### "undefined reference to `glXCreateContext`"

This is expected on headless systems. SDL examples will fallback to software renderer automatically.

### "Parser reached invalid state" on fresh clone

**Problem:**
After `git clone` and `make examples`, you get:
```
Error: Parser reached invalid state
Error: Program must define a 'main' function
```

**Possible causes:**
1. Uninitialized memory in parser/lexer (compiler-specific)
2. Self-hosted components failed but build claimed success
3. Platform-specific undefined behavior

**Solution:**
```bash
# Try a clean rebuild
make clean && make

# If that fails, check the bootstrap status carefully
# Look for "0/3 components built successfully" in Stage 2

# The C reference compiler should still work for simple programs:
./bin/nanoc examples/factorial.nano -o test_factorial
./test_factorial

# Report the issue with your system info:
# - OS and version (uname -a)
# - GCC version (gcc --version)
# - Output of: make clean && make 2>&1 | tee build.log
```

**Workaround:**
If examples don't compile but the C compiler built successfully, try:
- Simpler examples: `factorial.nano`, `fibonacci.nano`
- Interpreter mode: `./bin/nano examples/checkers.nano`
- Report the issue on GitHub with full build log

**Status:** This is a known bug being investigated. It appears to be a memory
initialization or platform-specific issue that doesn't affect all systems.

### "Segmentation fault" in examples

**Possible causes:**
1. Missing display (X11/Wayland)
2. Corrupted module cache
3. Related to above parser issue

**Solution:**
```bash
# Clean module cache
make clean

# Rebuild
make && cd examples && make
```

---

## Performance Considerations

### Software Renderer vs Hardware Acceleration

**Hardware Renderer** (when available):
- GPU-accelerated drawing
- Vsync support
- Smooth 60 FPS

**Software Renderer** (fallback):
- CPU-based drawing
- Still acceptable for 2D applications
- May have slight frame drops on complex scenes

**Which is being used?**
You'll see this message if falling back to software:
```
Hardware acceleration not available, trying software renderer...
```

---

## Transpiler Limitations (December 2025)

The current transpiler has several limitations discovered during the asteroids rewrite attempt:

### Type Conversion Issues

**Problem:** No built-in functions for explicit type conversion between int and float.

**Missing Functions:**
- `int_to_float()` - Convert integer to float
- `float_to_int()` - Convert float to integer  
- `cast_int()` - Used in boids_sdl but doesn't exist

**Current Workarounds:**
- Use `floor(x)`, `ceil(x)`, `round(x)` - but these return float, not int
- Automatic conversion sometimes works, but is inconsistent
- Examples like checkers avoid the issue by using int throughout

**Impact:**
- Can't easily convert SDL float coordinates to int for rendering
- Makes compiled games with physics (float) + rendering (int) difficult

### Array of Structs

**Problem:** Arrays of user-defined struct types don't compile.

```nano
# This doesn't work in compiled mode:
let bullets: array<Bullet> = [...]
```

**Workaround:**  
Use "structure of arrays" pattern (separate arrays for each field):
```nano
# This works:
let bullet_x: array<float> = [...]
let bullet_y: array<float> = [...]
let bullet_active: array<bool> = [...]
```

### Spelled-Out Operators (By Design)

**Note:** Nanolang uses spelled-out operators consistently:

```nano
# Correct nanolang style:
if (not condition) { ... }
if (and (> x 0) (< x 10)) { ... }
if (or (== x 5) (== x 10)) { ... }

# NOT valid (C-style operators not supported):
if (! condition) { ... }      # Wrong - use 'not'
if (x > 0 && x < 10) { ... }  # Wrong - use 'and'
if (x == 5 || x == 10) { ... }  # Wrong - use 'or'
```

This is intentional - NanoLang uses spelled-out logical operators (`and`, `or`, `not`) in both prefix `(and a b)` and infix `a and b` notation, rather than C-style symbols (`&&`, `||`, `!`).

### Examples Status

| Example | Compiles | Reason |
|---------|----------|--------|
| checkers.nano | ✅ Yes | Uses int coordinates only |
| raytracer_simple.nano | ✅ Yes | Uses float throughout |
| boids_sdl.nano | ❌ No | Uses undefined `cast_int` |
| asteroids_sdl.nano | ❌ No | Arrays of structs + type conversions |
| particles_sdl.nano | ❌ No | Dynamic `array_push` |

**For Now:** Run these examples in interpreter mode:
```bash
./bin/nano examples/boids_sdl.nano
./bin/nano examples/asteroids_sdl.nano
```

### Future Work

These limitations should be addressed in future releases:
1. Add `float_to_int()` and `int_to_float()` standard library functions
2. Fix arrays of structs compilation  
3. Improve type inference and automatic conversions

---

## Contributing Platform Support

Want to help test nanolang on other platforms?

1. **Try building:** Follow the build instructions above
2. **Run tests:** `make test`
3. **Report results:** Open a GitHub issue with:
   - Platform and architecture
   - Compiler version (`gcc --version`)
   - Test results
   - Any error messages

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## Version History

### 0.2.0 (December 2025)
- ✅ Fixed Ubuntu module building issues (empty pkg-config, UTF-8 corruption)
- ✅ Added SDL software renderer fallback
- ✅ All SDL examples work on Ubuntu
- ✅ All OpenGL examples compile on Ubuntu

### 0.1.0 (November 2025)
- Initial release
- macOS support
- Basic Linux support

---

## Summary

**TL;DR:**
- ✅ nanolang works great on macOS and Ubuntu Linux
- ✅ SDL examples fallback to software rendering automatically
- ✅ Install dev libraries via package manager (`apt` or `brew`)
- ✅ Some examples require interpreter mode (arrays of structs limitation)
- ❌ Windows not supported
- 🟡 Other Unix-like systems should work but are untested

Questions? See [docs/README.md](README.md) or open a GitHub issue.

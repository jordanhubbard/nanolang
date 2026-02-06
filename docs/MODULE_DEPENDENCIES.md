# Module Dependency Management

This document explains how NanoLang handles module dependencies and provides guidance for building with optional system libraries.

## Overview

NanoLang has a modular architecture where some modules require external system libraries (SDL2, OpenGL, SQLite, etc.). The build system provides tools to validate these dependencies and offers two build modes:

1. **Strict Mode** (`make examples`): Fails if any module dependencies are missing
2. **Graceful Mode** (`make examples-available`): Skips examples with missing dependencies

## Checking Module Dependencies

### Validate All Modules

```bash
make -B modules
```

This command:
- ✅ Shows which modules have all required dependencies
- ❌ Shows which modules have missing dependencies
- 📦 Provides installation commands for missing packages
- ⚠️  Exits with error code 1 if any dependencies are missing

### Example Output

```
Module Validation
========================================

Checking system dependencies for modules...

  ✓ vector2d
  ✓ math_ext
  ✗ sdl (missing: sdl2,sdl2_mixer)
  ✗ readline (missing: readline)

Summary
========================================
Total modules with dependencies: 16
Available: 2
Missing dependencies: 14
```

## Building Examples

### Strict Mode (Recommended for Development)

```bash
make examples
```

**Behavior:**
- Validates module dependencies first (`make modules`)
- Fails immediately if any dependencies are missing
- Ensures you know exactly what's needed
- Best for CI/CD and production builds

**Use when:**
- Setting up a new development environment
- Preparing for a release
- Running CI/CD pipelines
- You want explicit dependency management

### Graceful Mode (Convenient for Quick Iteration)

```bash
make examples-available
```

**Behavior:**
- Builds all examples that can be built
- Skips examples with missing dependencies
- Always succeeds (exit code 0)
- Useful for partial builds

**Use when:**
- Quick testing without installing all dependencies
- Working on core features that don't need graphics/audio
- Building on minimal systems
- You know which examples you need

## Installing Dependencies

### All Dependencies (Full Build)

For a complete build with all modules on Ubuntu/WSL2:

```bash
sudo apt-get install -y \
  pkg-config \
  libsdl2-dev \
  libsdl2-image-dev \
  libsdl2-mixer-dev \
  libsdl2-ttf-dev \
  libgl1-mesa-dev \
  libglew-dev \
  libglfw3-dev \
  freeglut3-dev \
  libncurses-dev \
  libreadline-dev \
  libsqlite3-dev \
  libcurl4-openssl-dev \
  libuv1-dev \
  libbullet-dev
```

### Minimal Dependencies (Core Only)

For just the core compiler and standard library:

```bash
sudo apt-get install -y build-essential python3
make build
make test
```

### Module-Specific Dependencies

The `make modules` command shows exactly what's needed for each module:

```bash
$ make -B modules

  • readline
    Install: sudo apt-get install libreadline-dev
  • sdl
    Install: sudo apt-get install libsdl2-dev libsdl2-mixer-dev
```

## Module Categories

### No External Dependencies

These modules work everywhere:
- `std` - Standard library (files, process, env)
- `math_ext` - Extended math functions
- `vector2d` - 2D vector math
- `stdlib` - Legacy stdlib utilities
- And more...

### Graphics & Windowing

Require system graphics libraries:
- `sdl` - SDL2 window/graphics/input
- `sdl_image`, `sdl_mixer`, `sdl_ttf` - SDL2 extensions
- `opengl` - OpenGL 3D rendering
- `glfw` - OpenGL window management
- `glew`, `glut` - OpenGL utilities

### Terminal UI

- `ncurses` - Terminal UI with colors/cursor control
- `readline` - Line editing with history

### Databases & Networking

- `sqlite` - Embedded SQL database
- `curl` - HTTP client
- `http_server`, `uv` - HTTP server and async I/O

### Physics & Specialized

- `bullet` - Bullet physics engine
- `pybridge` - Python interop

## Workflow Examples

### Setting Up a New Development Environment

```bash
# 1. Check what's needed
make -B modules

# 2. Install dependencies
sudo apt-get install -y [packages from above]

# 3. Build everything
make build
make test
make examples
```

### Working Without Graphics Libraries

```bash
# Build core compiler
make build

# Run tests (most will pass)
make test

# Build only available examples
make examples-available
```

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Install dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential python3 pkg-config
    # Add other dependencies as needed

- name: Validate modules
  run: make -B modules

- name: Build examples (strict)
  run: make examples
```

## Troubleshooting

### "make modules" shows dependencies missing

**Solution:** Install the packages shown in the output, then run `make -B modules` again.

### "make examples" fails

**Option 1:** Install missing dependencies
```bash
# Follow the commands shown by 'make modules'
sudo apt-get install [missing-packages]
```

**Option 2:** Use graceful mode
```bash
make examples-available
```

### Module build fails even with packages installed

1. Verify pkg-config can find the package:
   ```bash
   pkg-config --exists sdl2 && echo "Found" || echo "Not found"
   ```

2. Check package names:
   ```bash
   dpkg -l | grep libsdl2
   ```

3. Reinstall pkg-config:
   ```bash
   sudo apt-get install --reinstall pkg-config
   ```

## Platform-Specific Notes

### WSL2

- Graphics modules require WSLg (Windows 11) or X11 forwarding (Windows 10)
- See [BUILDING_WSL2.md](BUILDING_WSL2.md) for detailed setup

### macOS

- Use Homebrew for dependencies: `brew install sdl2 readline ...`
- Some packages have different names (e.g., `freeglut` vs `glut`)

### FreeBSD

- Use pkg: `pkg install sdl2 readline ...`
- Package names may differ slightly from Linux

## See Also

- [BUILDING_WSL2.md](BUILDING_WSL2.md) - WSL2-specific build guide
- [MODULE_SYSTEM.md](MODULE_SYSTEM.md) - Module system architecture
- [PLATFORM_COMPATIBILITY.md](PLATFORM_COMPATIBILITY.md) - Cross-platform notes

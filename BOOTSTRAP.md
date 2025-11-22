# Bootstrap Guide for nanolang

This document explains how to set up nanolang's dependencies on different platforms.

## macOS - Automatic Bootstrap

On macOS, the build system automatically handles dependency installation:

```bash
make examples
```

This will:
1. Automatically detect if Homebrew is installed
2. Run the bootstrap script if needed to install Homebrew and SDL2
3. Build all examples

The bootstrap process is **fully idempotent** - you can run it multiple times safely, and it will only install what's missing.

### First-Time Setup on macOS

Simply run:

```bash
make examples
```

The system will automatically:
- Check for Homebrew (install if missing)
- Check for SDL2 (install if missing)
- Build all examples

### Manual Bootstrap

You can also run the bootstrap script directly:

```bash
./scripts/bootstrap-macos.sh
```

This will:
- Install Homebrew (if needed)
- Install SDL2 for graphics examples (if needed)
- Optionally install GLFW and GLEW for OpenGL examples
- Verify all installations

The script is idempotent and can be run multiple times safely.

## Linux

On Linux, install dependencies using your package manager:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential libsdl2-dev pkg-config
```

### Fedora
```bash
sudo dnf install gcc make SDL2-devel pkgconfig
```

### Arch Linux
```bash
sudo pacman -S base-devel sdl2 pkgconf
```

## Building Examples

After dependencies are installed:

```bash
# Build compiler and interpreter
make

# Build all examples
make examples

# Build only SDL examples
make -C examples sdl

# Run an example
./bin/checkers_sdl
./bin/boids_sdl
```

## Verifying Installation

Check that dependencies are installed:

```bash
# Check SDL2
pkg-config --modversion sdl2

# Check Homebrew (macOS)
brew --version

# Check compiler
gcc --version
make --version
```

## Troubleshooting

### macOS: "brew: command not found"
The bootstrap script will handle this automatically. If you need to install Homebrew manually:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then run `make examples` again.

### Linux: "SDL.h file not found"
Install SDL2 development headers:
```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# Fedora
sudo dnf install SDL2-devel
```

### "pkg-config not found"
Install pkg-config:
```bash
# macOS
brew install pkg-config

# Ubuntu/Debian
sudo apt-get install pkg-config

# Fedora
sudo dnf install pkgconf
```

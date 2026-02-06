# Building NanoLang on WSL2

This guide covers building NanoLang on Windows Subsystem for Linux 2 (WSL2).

## Quick Start

```bash
# 1. Install build essentials
sudo apt-get update
sudo apt-get install -y build-essential python3

# 2. Install pkg-config (required for dependency detection)
sudo apt-get install -y pkg-config

# 3. Build the compiler
make build

# 4. Run tests
make test

# 5. Check which modules can be built
make modules
```

## Installing All Module Dependencies

To build **all** modules and examples (including graphics, audio, and database modules), install these packages:

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

Then rebuild:

```bash
make clean
make build
make examples
```

## Minimal Build (No External Dependencies)

If you don't need graphics, audio, or database modules, you can build just the core compiler and standard library modules:

```bash
# Only need build essentials
sudo apt-get install -y build-essential python3

# Build core compiler
make build

# Run tests (most will pass, some graphics tests may be skipped)
make test

# Build examples (examples requiring unavailable modules will be skipped)
make examples
```

## Validating Module Dependencies

Use `make modules` to see which modules can be built with your current system packages:

```bash
make modules
```

This will:
- Show which modules are available (✓)
- Show which modules have missing dependencies (✗)
- Provide installation commands for missing dependencies

## Graphics and Audio on WSL2

WSL2 supports:
- **X11 forwarding**: Set up an X server (like VcXsrv or X410) on Windows
- **WSLg**: Built-in GUI support on Windows 11

For graphics examples:
1. Windows 11: WSLg is enabled by default
2. Windows 10: Install VcXsrv and set `DISPLAY`:
   ```bash
   export DISPLAY=:0
   ```

## Platform Differences

NanoLang is fully cross-platform. The same code builds on:
- WSL2 (Ubuntu, Debian)
- Native Linux (Ubuntu, Fedora, Arch, etc.)
- macOS (via Homebrew)
- FreeBSD (via pkg)

The build system automatically detects your platform and adjusts accordingly.

## Troubleshooting

### sudo requires password

If you see messages about sudo requiring a password, you can either:

1. **Run sudo manually**: Install packages yourself using the commands shown
2. **Configure passwordless sudo**: Add your user to sudoers (not recommended for security)
3. **Build without those modules**: Examples requiring unavailable modules will be skipped

### Permission denied errors

Make sure all scripts are executable:

```bash
chmod +x scripts/*.sh
```

### Python not found

Install Python 3:

```bash
sudo apt-get install -y python3
```

## See Also

- [Main README](../README.md)
- [Platform Compatibility](PLATFORM_COMPATIBILITY.md)
- [Module System](MODULE_SYSTEM.md)

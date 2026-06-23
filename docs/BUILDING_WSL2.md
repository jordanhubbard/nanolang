# Building me on WSL2

I provide this guide for those building me on the Windows Subsystem for Linux 2 (WSL2).

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

To build all my modules and examples, including graphics, audio, and database modules, I require these packages:

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

After you install these, rebuild my components:

```bash
make clean
make build
make examples
```

## Minimal Build (No External Dependencies)

If you do not need graphics, audio, or database modules, you can build my core compiler and standard library modules alone.

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

Use `make modules` to see which of my modules can be built with your current system packages:

```bash
make modules
```

This command will:
- Show which modules are available (✓)
- Show which modules have missing dependencies (✗)
- Provide installation commands for missing dependencies

## Graphics and Audio on WSL2

I support graphics and audio on WSL2 through:
- **X11 forwarding**: Set up an X server like VcXsrv or X410 on Windows
- **WSLg**: Built-in GUI support on Windows 11

For my graphics examples:
1. Windows 11: WSLg is enabled by default.
2. Windows 10: Install VcXsrv and set `DISPLAY`:
   ```bash
   export DISPLAY=:0
   ```

## Platform Differences

I am fully cross-platform. The same code builds on:
- WSL2 (Ubuntu, Debian)
- Native Linux (Ubuntu, Fedora, Arch, etc.)
- macOS (via Homebrew)
- FreeBSD (via pkg)

My build system detects your platform and adjusts itself.

## Troubleshooting

### sudo requires password

If you see messages about sudo requiring a password, you can:

1. **Run sudo manually**: Install packages yourself using the commands I show.
2. **Configure passwordless sudo**: Add your user to sudoers. I do not recommend this for security.
3. **Build without those modules**: I will skip examples that require unavailable modules.

### Permission denied errors

Ensure all my scripts are executable:

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

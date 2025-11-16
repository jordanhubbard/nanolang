# nanolang Module Packaging System

This directory contains the module packaging and distribution system for nanolang.

## Directory Structure

```
modules/
├── MODULE_FORMAT.md          # Module package format specification
├── README.md                 # This file
├── sdl/                      # SDL2 module source
│   ├── module.json          # Module metadata
│   └── sdl.nano             # Module source
├── sdl_helpers/              # SDL helpers module source
│   ├── module.json          # Module metadata
│   ├── sdl_helpers.nano     # Module source
│   └── sdl_helpers.c        # Runtime implementation
└── tools/                    # Build and installation tools
    ├── build_module.sh       # Build a module package
    └── install_module.sh     # Install a module package
```

## Quick Start

### Building a Module

```bash
cd modules
./tools/build_module.sh sdl
```

This will:
1. Check for system dependencies (brew/apt packages)
2. Compile the module (if it has implementation)
3. Create a `.nano.tar.zst` package

### Installing a Module

```bash
./tools/install_module.sh sdl/sdl.nano.tar.zst
```

This installs the module to `~/.nanolang/modules/` (or `NANO_MODULE_PATH` if set).

### Using Modules

Set `NANO_MODULE_PATH`:
```bash
export NANO_MODULE_PATH=~/.nanolang/modules
```

Then import in your nanolang code:
```nano
import "sdl"
```

The runtime will automatically:
- Find the module in `NANO_MODULE_PATH`
- Unpack it to a temporary directory
- Load it dynamically (interpreter) or statically link it (compiler)

## Module Format

See `MODULE_FORMAT.md` for complete specification.

## Distribution Conundrum

**Current Limitation**: Modules depend on system libraries (e.g., SDL2 installed via brew/apt). When shipping applications:

- ✅ **Works**: Build and run on the same host
- ❌ **Doesn't work**: Ship binary to another host without the same libraries

**Future Consideration**: Static linking (like Go) to create self-contained binaries.

## Module Search Order

1. Current directory (relative to source file)
2. `NANO_MODULE_PATH` directories (in order)
3. `~/.nanolang/modules` (default)


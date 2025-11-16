# nanolang Module Package Format

## Overview

nanolang modules are distributed as compressed archives containing:
- Module metadata (JSON)
- Compiled object files (.o)
- Source files (.nano) for reference
- Build metadata

## Archive Format

- **Format**: tar archive compressed with zstd
- **Extension**: `.nano.tar.zst`
- **Tool**: `tar -I zstd` (supported by GNU tar and BSD tar)

## Directory Structure

```
module-name.nano.tar.zst
├── module.json          # Module metadata
├── module.o             # Compiled object file (if has implementation)
├── module.nano          # Source file (for reference/documentation)
└── build.json           # Build metadata (compiler version, flags, etc.)
```

## Module Metadata (module.json)

```json
{
  "name": "sdl",
  "version": "1.0.0",
  "description": "SDL2 library bindings for nanolang",
  "type": "ffi",
  "dependencies": {
    "system": {
      "macos": {
        "brew": ["sdl2"]
      },
      "linux": {
        "apt": ["libsdl2-dev"]
      }
    },
    "nanolang": []
  },
  "compilation": {
    "include_paths": ["/opt/homebrew/include/SDL2"],
    "library_paths": ["/opt/homebrew/lib"],
    "libraries": ["SDL2"]
  },
  "exports": {
    "functions": [
      {
        "name": "SDL_Init",
        "signature": "extern fn SDL_Init(flags: int) -> int"
      }
    ]
  },
  "build_info": {
    "compiler_version": "1.0.0",
    "build_date": "2025-11-15",
    "platform": "darwin-arm64"
  }
}
```

## Build Metadata (build.json)

```json
{
  "source_file": "sdl.nano",
  "object_file": "sdl.o",
  "has_implementation": false,
  "is_ffi_only": true,
  "compiler_flags": {
    "include_paths": ["-I/opt/homebrew/include/SDL2"],
    "library_paths": ["-L/opt/homebrew/lib"],
    "libraries": ["-lSDL2"]
  }
}
```

## Module Types

1. **ffi** - FFI-only module (only extern declarations, no implementation)
2. **implementation** - Module with nanolang implementations
3. **hybrid** - Module with both FFI declarations and implementations

## Installation

Modules are installed to a directory specified by `NANO_MODULE_PATH`:
- Default: `~/.nanolang/modules/`
- Can be overridden with `NANO_MODULE_PATH` environment variable
- Multiple paths supported (colon-separated on Unix, semicolon on Windows)

## Module Search

When importing a module:
1. Check current directory
2. Check `NANO_MODULE_PATH` directories (in order)
3. For each directory, look for `module-name.nano.tar.zst`
4. If found, unpack to temp directory and load


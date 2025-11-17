# Nanolang Module System

This document describes the module build system, including automatic C compilation, dependency management, and linking.

## Overview

Nanolang modules are self-contained packages that can include:
- Nanolang source code (`.nano` files)
- C FFI implementations (`.c` files)
- Module metadata (`module.json`)

When you import a module, the compiler automatically:
1. Compiles any C sources (if needed)
2. Caches compiled objects
3. Tracks dependencies
4. Generates correct link commands

## Module Structure

A typical module directory:

```
modules/sdl_ttf/
├── module.json              # Module metadata (required if C sources exist)
├── sdl_ttf.nano             # Main module file (required)
├── sdl_ttf_helpers.c        # C implementation (optional)
└── sdl_ttf_helpers.nano     # Additional nanolang helpers (optional)
```

## module.json Specification

The `module.json` file describes how to build and link the module.

### Basic Structure

```json
{
  "name": "module_name",
  "version": "1.0.0",
  "description": "Short description of the module",
  
  "c_sources": [
    "helper.c",
    "another_file.c"
  ],
  
  "system_libs": [
    "SDL2_ttf",
    "m"
  ],
  
  "pkg_config": [
    "sdl2_ttf"
  ],
  
  "include_dirs": [
    "/usr/local/include"
  ],
  
  "cflags": [
    "-O2",
    "-Wall"
  ],
  
  "ldflags": [
    "-L/usr/local/lib"
  ],
  
  "dependencies": [
    "sdl"
  ]
}
```

### Fields

#### Required Fields

- **`name`** (string): Module name (should match directory name)

#### Optional Fields

- **`version`** (string): Semantic version (e.g., "1.0.0")
- **`description`** (string): Brief description of module purpose
- **`c_sources`** (array of strings): C source files to compile (relative to module directory)
- **`system_libs`** (array of strings): System libraries to link (e.g., "SDL2", "m", "pthread")
- **`pkg_config`** (array of strings): pkg-config packages for compile and link flags
- **`include_dirs`** (array of strings): Additional include directories
- **`cflags`** (array of strings): Additional compiler flags
- **`ldflags`** (array of strings): Additional linker flags
- **`dependencies`** (array of strings): Other nanolang modules this module depends on

### Field Priority

When determining build flags, the following priority applies (later overrides earlier):

1. Default system flags
2. `pkg_config` flags
3. `include_dirs` → `-I` flags
4. `cflags` (custom compile flags)
5. `ldflags` (custom link flags)
6. `system_libs` → `-l` flags

## Build Process

### When a Module is Imported

1. **Parse module.json** (if it exists)
2. **Check dependencies** - recursively process dependent modules
3. **Check build cache**:
   - Module build directory: `modules/module_name/.build/`
   - Object file: `modules/module_name/.build/module_name.o`
   - Metadata: `modules/module_name/.build/.build_info.json`
4. **Rebuild if needed**:
   - Any C source is newer than object file
   - module.json changed
   - Dependency changed
5. **Track for linking**:
   - Add object file to link list
   - Add system_libs to link list
   - Add ldflags to link list

### Build Cache

Each module with C sources gets a `.build/` directory:

```
modules/sdl_ttf/
├── .build/
│   ├── sdl_ttf.o              # Compiled object file
│   ├── .build_info.json       # Build metadata
│   └── .compile_commands.json # Compile commands (optional)
```

**`.build_info.json`**:
```json
{
  "sources": {
    "sdl_ttf_helpers.c": {
      "mtime": 1234567890,
      "size": 12345
    }
  },
  "module_json_mtime": 1234567890,
  "compile_command": "gcc ...",
  "timestamp": "2025-11-17T12:00:00Z"
}
```

### Incremental Compilation

The build system only recompiles when:
- Source file modified (mtime or size changed)
- `module.json` modified
- Object file missing
- Dependency rebuilt

### Parallel Builds

Modules with no interdependencies can be compiled in parallel (future enhancement).

## Using Modules

### Pure Nanolang Modules (No C)

No `module.json` needed - just import:

```nano
import "modules/my_pure_module/my_module.nano"
```

### Modules with C FFI

Automatic compilation and linking:

```nano
import "modules/sdl_ttf/sdl_ttf.nano"

fn main() -> int {
    (TTF_Init)  # C function automatically available
    return 0
}
```

Compile:
```bash
nanoc my_app.nano -o my_app
```

The compiler automatically:
1. Detects `sdl_ttf` module import
2. Reads `modules/sdl_ttf/module.json`
3. Compiles `sdl_ttf_helpers.c` (if needed)
4. Links with SDL2_ttf system library
5. Produces working binary

### Module Dependencies

If module A depends on module B:

**modules/my_module/module.json**:
```json
{
  "name": "my_module",
  "dependencies": ["sdl", "sdl_helpers"],
  "c_sources": ["my_module.c"]
}
```

When you import `my_module`, the compiler automatically:
1. Processes `sdl` module first
2. Processes `sdl_helpers` module
3. Processes `my_module`
4. Links all three modules' objects

## Environment Variables

### NANO_MODULE_PATH

Semicolon-separated list of directories to search for modules:

```bash
export NANO_MODULE_PATH="/usr/local/lib/nano/modules:./modules:~/nano/modules"
```

Default: `modules` (relative to current directory)

### NANO_BUILD_CACHE

Directory for module build cache:

```bash
export NANO_BUILD_CACHE="/tmp/nano_build_cache"
```

Default: `.build/` in each module directory

### NANO_CC

C compiler to use:

```bash
export NANO_CC=clang
```

Default: `gcc`

### NANO_VERBOSE_BUILD

Enable verbose build output:

```bash
export NANO_VERBOSE_BUILD=1
nanoc my_app.nano -o my_app
```

## Examples

### Example 1: SDL Window Module

**modules/sdl/module.json**:
```json
{
  "name": "sdl",
  "version": "1.0.0",
  "description": "SDL2 windowing, rendering, and events",
  "pkg_config": ["sdl2"]
}
```

No C sources - uses system SDL2 library directly.

### Example 2: SDL Helpers with C Code

**modules/sdl_helpers/module.json**:
```json
{
  "name": "sdl_helpers",
  "version": "1.0.0",
  "description": "Helper functions for SDL rendering",
  "c_sources": ["sdl_helpers.c"],
  "pkg_config": ["sdl2"],
  "dependencies": ["sdl"]
}
```

Has C implementation and depends on SDL module.

### Example 3: SDL_ttf Extension

**modules/sdl_ttf/module.json**:
```json
{
  "name": "sdl_ttf",
  "version": "1.0.0",
  "description": "TrueType font rendering for SDL",
  "c_sources": ["sdl_ttf_helpers.c"],
  "pkg_config": ["sdl2_ttf"],
  "dependencies": ["sdl"]
}
```

## Troubleshooting

### "pkg-config: command not found"

Install pkg-config:
- macOS: `brew install pkg-config`
- Linux: `sudo apt-get install pkg-config`

### "No package 'sdl2_ttf' found"

Install the development package:
- macOS: `brew install sdl2_ttf`
- Linux: `sudo apt-get install libsdl2-ttf-dev`

### Module not rebuilding

Clear the build cache:
```bash
rm -rf modules/*/build/
```

Or touch the source file:
```bash
touch modules/sdl_ttf/sdl_ttf_helpers.c
```

### Verbose build output

```bash
NANO_VERBOSE_BUILD=1 nanoc my_app.nano -o my_app
```

## Best Practices

1. **Keep module.json minimal** - Only specify what's needed
2. **Use pkg-config when available** - Better than hardcoded paths
3. **Version your modules** - Use semantic versioning
4. **Document dependencies** - List all nanolang module dependencies
5. **Test incremental builds** - Ensure cache works correctly
6. **Provide examples** - Show how to use the module

## Future Enhancements

- [ ] Parallel module compilation
- [ ] Cross-compilation support
- [ ] Module registry/package manager
- [ ] Automatic dependency installation
- [ ] Binary module distribution (.a/.so files)
- [ ] Module versioning and compatibility checking


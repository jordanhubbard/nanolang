# nanolang Module Format

## Overview

nanolang modules are directories containing:
- Module metadata (`module.json`) - **Required if module has C sources**
- Nanolang source files (`.nano`) - **Required**
- C implementation files (`.c`) - Optional
- C header files (`.h`) - Optional

**Note:** This document describes the **current implementation**. For future archive-based distribution format, see the roadmap.

## Module Structure

A typical module directory:

```
modules/sdl_ttf/
├── module.json              # Module metadata (required if C sources exist)
├── sdl_ttf.nano             # Main module file (required)
├── sdl_ttf_helpers.c        # C implementation (optional)
├── sdl_ttf_helpers.nano     # Additional nanolang helpers (optional)
└── sdl_ttf_helpers.h        # C header (optional)
```

## module.json Format

The `module.json` file describes how to build and link the module.

### Standard Format

```json
{
  "name": "module_name",
  "version": "1.0.0",
  "description": "Short description of the module",
  
  "c_sources": [
    "helper.c",
    "another_file.c"
  ],
  
  "headers": [
    "helper.h"
  ],
  
  "pkg_config": [
    "sdl2_ttf"
  ],
  
  "cflags": [
    "-O2",
    "-Wall",
    "-Imodules/module_name"
  ],
  
  "dependencies": [
    "sdl"
  ]
}
```

### Field Reference

#### Required Fields

- **`name`** (string): Module name (should match directory name)

#### Optional Fields

- **`version`** (string): Semantic version (e.g., "1.0.0")
- **`description`** (string): Brief description of module purpose
- **`c_sources`** (array of strings): C source files to compile (relative to module directory)
- **`headers`** (array of strings): C header files (for include paths)
- **`pkg_config`** (array of strings): pkg-config packages for compile and link flags
- **`cflags`** (array of strings): Additional compiler flags (including `-I` for include dirs)
- **`dependencies`** (array of strings): Other nanolang modules this module depends on

### Field Usage Notes

1. **`c_sources`**: List all `.c` files that need compilation
2. **`headers`**: List header files (used for include path detection)
3. **`pkg_config`**: Preferred method for system library detection
4. **`cflags`**: Use for custom compiler flags, include paths (`-I`), etc.
5. **`dependencies`**: List other nanolang modules this depends on

### Build Process

When a module is imported:

1. **Parse module.json** (if it exists)
2. **Check dependencies** - recursively process dependent modules
3. **Check build cache**:
   - Module build directory: `modules/module_name/.build/`
   - Object files compiled from `c_sources`
4. **Rebuild if needed**:
   - Any C source is newer than object file
   - `module.json` changed
   - Dependency changed
5. **Track for linking**:
   - Add object files to link list
   - Add pkg-config libraries to link list
   - Add cflags to compile flags

### Examples

#### Simple FFI Module (No C Sources)

```json
{
  "name": "vector2d",
  "version": "1.0.0",
  "description": "2D vector mathematics"
}
```

#### Module with C Implementation

```json
{
  "name": "sdl_helpers",
  "version": "1.0.0",
  "description": "Helper functions for SDL rendering",
  "c_sources": ["sdl_helpers.c"],
  "headers": ["sdl_helpers.h"],
  "cflags": ["-Imodules/sdl_helpers"],
  "pkg_config": ["sdl2"],
  "dependencies": ["sdl"]
}
```

#### Module with System Libraries

```json
{
  "name": "math_ext",
  "version": "1.0.0",
  "description": "Extended math functions",
  "cflags": ["-lm"]
}
```

## Module Search

When importing a module:

1. Check current directory
2. Check `NANO_MODULE_PATH` directories (in order, colon-separated)
3. For each directory, look for `module-name/module-name.nano` or `module-name.nano`
4. Load `module.json` if it exists in the module directory

## Pure Nanolang Modules

Modules with no C sources don't need `module.json`:

```
modules/my_pure_module/
└── my_module.nano
```

Just import:
```nano
import "modules/my_pure_module/my_module.nano"
```

## Future: Archive Format

**Note:** Archive-based distribution (`.nano.tar.zst`) is planned for future releases but not yet implemented. The current system uses directory-based modules as described above.

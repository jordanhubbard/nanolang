# nanolang module.json Schema Reference

## Overview

This document provides a complete reference for the `module.json` schema used by nanolang modules.

## Schema Version

Current schema version: **1.0.0**

## Field Reference

### Required Fields

#### `name` (string)
- **Required**: Yes
- **Description**: Module name (should match directory name)
- **Example**: `"sdl"`, `"sdl_helpers"`

### Standard Fields

#### `version` (string)
- **Required**: No (recommended)
- **Description**: Semantic version (e.g., "1.0.0")
- **Example**: `"1.0.0"`

#### `description` (string)
- **Required**: No (recommended)
- **Description**: Brief description of module purpose
- **Example**: `"SDL2 windowing, rendering, events, and input"`

#### `c_sources` (array of strings)
- **Required**: No (required if module has C implementation)
- **Description**: C source files to compile (relative to module directory)
- **Example**: `["sdl_helpers.c", "another_file.c"]`

#### `headers` (array of strings)
- **Required**: No
- **Description**: C header files (used for include path detection)
- **Example**: `["sdl_helpers.h"]`

#### `pkg_config` (array of strings)
- **Required**: No
- **Description**: pkg-config packages for compile and link flags
- **Example**: `["sdl2", "sdl2_ttf"]`
- **Note**: Preferred method for system library detection

#### `cflags` (array of strings)
- **Required**: No
- **Description**: Additional compiler flags
- **Example**: `["-O2", "-Wall", "-Imodules/module_name"]`
- **Note**: Use `-I` for include directories, `-D` for defines, etc.

#### `ldflags` (array of strings)
- **Required**: No
- **Description**: Additional linker flags
- **Example**: `["-L/usr/local/lib", "-lm"]`
- **Note**: Use `-L` for library paths, `-l` for libraries

#### `system_libs` (array of strings)
- **Required**: No
- **Description**: System libraries to link (alternative to pkg-config)
- **Example**: `["SDL2", "m", "pthread"]`
- **Note**: Prefer `pkg_config` when available

#### `include_dirs` (array of strings)
- **Required**: No
- **Description**: Additional include directories (alternative to `-I` in cflags)
- **Example**: `["/usr/local/include"]`
- **Note**: Prefer `-I` in `cflags` for consistency

#### `dependencies` (array of strings)
- **Required**: No
- **Description**: Other nanolang modules this module depends on
- **Example**: `["sdl"]`
- **Note**: Dependencies are processed recursively

### Platform-Specific Fields

#### `frameworks` (array of strings)
- **Required**: No
- **Description**: macOS frameworks to link (macOS only)
- **Example**: `["OpenGL", "Cocoa"]`
- **Note**: Ignored on non-macOS platforms

#### `header_priority` (integer)
- **Required**: No
- **Description**: Header include priority (higher = included first)
- **Default**: `0`
- **Example**: `1000`
- **Note**: Used for header ordering when multiple modules provide headers

### Documentation Fields

These fields are **not used by the build system** but are useful for documentation and tooling:

#### `install` (object)
- **Required**: No
- **Description**: Installation instructions for different platforms
- **Example**:
  ```json
  {
    "macos": {
      "brew": "sdl2",
      "command": "brew install sdl2"
    },
    "linux": {
      "apt": "libsdl2-dev",
      "command": "sudo apt install libsdl2-dev"
    }
  }
  ```

#### `notes` (string)
- **Required**: No
- **Description**: Additional notes about the module
- **Example**: `"GLEW's Homebrew pkg-config on macOS doesn't include the OpenGL framework dependency"`

#### `author` (string)
- **Required**: No
- **Description**: Module author or maintainer
- **Example**: `"nanolang"`

## Field Priority

When determining build flags, the following priority applies (later overrides earlier):

1. Default system flags
2. `pkg_config` flags
3. `include_dirs` → `-I` flags
4. `cflags` (custom compile flags)
5. `ldflags` (custom link flags)
6. `system_libs` → `-l` flags
7. `frameworks` → `-framework` flags (macOS only)

## Examples

### Minimal Module (Pure Nanolang)

```json
{
  "name": "vector2d",
  "version": "1.0.0",
  "description": "2D vector mathematics"
}
```

### FFI-Only Module (No C Sources)

```json
{
  "name": "sdl",
  "version": "1.0.0",
  "description": "SDL2 library bindings",
  "headers": ["SDL.h"],
  "pkg_config": ["sdl2"]
}
```

### Module with C Implementation

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

### Module with System Libraries

```json
{
  "name": "math_ext",
  "version": "1.0.0",
  "description": "Extended math functions",
  "cflags": [],
  "ldflags": ["-lm"]
}
```

### Module with Platform-Specific Features

```json
{
  "name": "glew",
  "version": "1.0.0",
  "description": "GLEW - OpenGL Extension Wrangler",
  "headers": ["GL/glew.h"],
  "header_priority": 1000,
  "pkg_config": ["glew"],
  "frameworks": ["OpenGL"],
  "install": {
    "macos": {
      "brew": "glew",
      "command": "brew install glew"
    },
    "linux": {
      "apt": "libglew-dev",
      "command": "sudo apt install libglew-dev"
    }
  },
  "notes": "GLEW's Homebrew pkg-config on macOS doesn't include the OpenGL framework dependency"
}
```

## Validation

The module builder validates:
- JSON syntax
- Required `name` field
- Array types for list fields
- String types for text fields

**Note**: Unknown fields are ignored (allows for documentation fields and future extensions).

## Migration Guide

If you have an older module.json with non-standard fields:

### Old → New Field Names

- `source_files` → `c_sources`
- `compile_flags` → `cflags`
- `link_flags` → `ldflags`
- `system_libs` → `system_libs` (unchanged, but prefer `pkg_config`)

### Removed Fields

- `type` - Not used by build system
- `exports` - Not used by build system (documentation only)

## Best Practices

1. **Always include** `name`, `version`, and `description`
2. **Prefer `pkg_config`** over `system_libs` when available
3. **Use `cflags`** for include directories (`-I`) rather than `include_dirs`
4. **Document dependencies** in `dependencies` array
5. **Add `install`** field for modules requiring system libraries
6. **Keep it simple** - only include fields you actually need

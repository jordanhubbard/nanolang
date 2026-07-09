# Multi-File Projects

## Overview

I now support complete multi-file project development. I handle project-relative imports, module caching, constant sharing, and automatic dependency resolution.

## Features

### Project-Relative Imports

You can import files from anywhere in your project using project-relative paths.

```nano
// Import from examples directory
import "examples/myproject/types.nano"
import "examples/myproject/utils.nano"

// Import from src directory
import "src/core/parser.nano"
import "src/core/lexer.nano"
```

My compiler automatically performs these steps:
1. I detect imports starting with `examples/` or `src/`.
2. I walk up the directory tree to find the project root.
3. I resolve paths relative to the project root.
4. I work regardless of where you invoke compilation.

### Module Caching and Deduplication

I prevent duplicate imports and circular dependencies.

```nano
// types.nano - base types
enum Status { Ok, Error }

// parser.nano
import "examples/myproject/types.nano"  // First load

// main.nano  
import "examples/myproject/types.nano"  // Cached, skips
import "examples/myproject/parser.nano" // Loads, types already cached
```

**My Cache Lifecycle:**
- I clear the cache at the start of compilation.
- I cache modules as I load them during import processing.
- Each `.nano` to `.o` compilation gets an isolated cache.
- I do this to prevent "already defined" errors.

### Constant Export and Inlining

I automatically export and inline your top-level immutable constants.

```nano
// config.nano
let MAX_CONNECTIONS: int = 100
let PI: float = 3.14159
let DEBUG: bool = true

// server.nano
import "examples/myproject/config.nano"

fn init_server() -> int {
    let pool_size = MAX_CONNECTIONS * 2  // Constant inlined!
    // ...
}
```

**I transpile this to:**
```c
int64_t pool_size = (100 * 2LL);  // Constant value inlined
```

### Recursive Import Processing

My modules can import other modules transitively.

```
types.nano (base types)
  ↑
parser.nano (imports types)
  ↑
compiler.nano (imports parser, gets types transitively)
```

## Example: Project Structure

This is how I expect a complete multi-file project to look.

```
examples/myproject/
├── types.nano     - Base types, enums, constants
├── utils.nano     - Helper functions
├── core.nano      - Core logic
├── main.nano      - Entry point
├── Makefile
└── README.md
```

**Import Structure:**
```nano
// types.nano - Base definitions
enum Status { Ok, Error, Pending }
let MAX_RETRIES: int = 3
let TIMEOUT: float = 30.0

// utils.nano - Uses types
import "examples/myproject/types.nano"
let retry_count = MAX_RETRIES * 2  // Constants work!

// core.nano - Uses types and utils
import "examples/myproject/types.nano"
import "examples/myproject/utils.nano"

// main.nano - Uses everything
import "examples/myproject/types.nano"
import "examples/myproject/utils.nano"
import "examples/myproject/core.nano"
```

**Compilation:**
```bash
cd examples/myproject
make
# Compiles main.nano with all imports
# Each module compiled to .o file
# Linked together
```

## Generic Module System

I use metadata-driven header inclusion for my modules.

### Module Schema (module.json)

```json
{
  "name": "sdl_mixer",
  "version": "1.0.0",
  "description": "Audio mixing for SDL",
  "headers": ["SDL_mixer.h"],          // ← C headers to include
  "c_sources": [],
  "pkg_config": ["SDL2_mixer"],        // ← Compile/link flags
  "dependencies": ["sdl"]              // ← Module dependencies
}
```

### Supported Libraries

I can wrap any C library.

**SDL Ecosystem:**
- SDL2: `"headers": ["SDL.h"]`
- SDL2_mixer: `"headers": ["SDL_mixer.h"]`
- SDL2_ttf: `"headers": ["SDL_ttf.h"]`
- SDL2_image: `"headers": ["SDL_image.h"]`

**Graphics:**
- CUDA: `"headers": ["cuda.h", "cuda_runtime.h"]`
- OpenGL: `"headers": ["GL/gl.h"]`
- Vulkan: `"headers": ["vulkan/vulkan.h"]`

**UI Frameworks:**
- GTK: `"headers": ["gtk/gtk.h"]`, `"pkg_config": ["gtk+-3.0"]`
- Qt: Custom headers and flags

**Utilities:**
- curl: `"headers": ["curl/curl.h"]`
- zlib: `"headers": ["zlib.h"]`
- libpng: `"headers": ["png.h"]`

## Architecture

### 1. Module Resolution (module.c)

```c
char *resolve_module_path(const char *module_path, const char *current_file) {
    // Detect project-relative paths
    if (strncmp(module_path, "examples/", 9) == 0 || 
        strncmp(module_path, "src/", 4) == 0) {
        // Walk up directory tree to find project root
        // Resolve path relative to root
    }
    // ...
}
```

### 2. Module Cache

```c
typedef struct {
    char **loaded_paths;
    int count;
    int capacity;
} ModuleCache;

// Global cache cleared at compilation start
static ModuleCache *module_cache = NULL;
```

### 3. Constant Evaluation

```c
// In process_imports() - module.c
if (module_item->type == AST_LET && !module_item->as.let.is_mut) {
    // Evaluate literal values
    if (value_node->type == AST_NUMBER) {
        val = create_int(value_node->as.number);
    }
    // Store in environment
    env_define_var(env, name, type, false, val);
}
```

### 4. Constant Inlining

```c
// In transpile_expression() - transpiler.c
case AST_IDENTIFIER: {
    // Check if immutable constant
    if (var_index >= 0 && !env->symbols[var_index].is_mut) {
        if (val.type == VAL_INT) {
            // Inline the literal value
            snprintf(num_buf, sizeof(num_buf), "%lld", val.as.int_val);
            sb_append(sb, num_buf);
        }
    }
}
```

## Benefits

| Before | After |
|--------|-------|
| Single-file only | Multi-file projects |
| No code reuse | Shared constants and types |
| Monolithic programs | Modular architecture |
| Hardcoded SDL | Generic library support |

## Future Enhancements

1. **Module Visibility**
   - I will add public and private exports.
   - I will support selective imports: `import { function_name } from "module"`.

2. **Package Management**
   - I plan to have a package registry.
   - I will support semantic versioning.
   - I will handle dependency resolution.

3. **Expression Evaluation**
   - I will support compile-time computed constants.
   - I will implement constant folding optimization.

4. **Type Exports**
   - I will allow sharing of struct, enum, and union definitions.
   - I will support type aliases across modules.

## Example: Creating a CUDA Module

```json
// modules/cuda/module.json
{
  "name": "cuda",
  "version": "1.0.0",
  "description": "NVIDIA CUDA GPU computing",
  "headers": [
    "cuda.h",
    "cuda_runtime.h"
  ],
  "c_sources": ["cuda_helpers.c"],
  "system_libs": ["cuda", "cudart"],
  "include_dirs": ["/usr/local/cuda/include"],
  "ldflags": ["-L/usr/local/cuda/lib64"]
}
```

```nano
// modules/cuda/cuda.nano
extern fn cudaMalloc(ptr: int, size: int) -> int
extern fn cudaMemcpy(dst: int, src: int, size: int, kind: int) -> int
extern fn cudaFree(ptr: int) -> int

// Usage in your program
import "modules/cuda/cuda.nano"

fn main() -> int {
    let device_ptr: int = 0
    cudaMalloc(device_ptr, 1024)
    // ...
}
```

## Conclusion

My support for multi-file projects enables:
- Professional software architecture.
- Large-scale application development.
- Team collaboration on codebases.
- Reusable module libraries.
- Integration with any C library.

My future for development is clear.

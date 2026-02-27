# Creating Modules

I show you how to create custom modules to integrate C libraries with me.

## My Module Structure

Every module I use lives in `modules/<name>/` and contains:

```
modules/mymodule/
├── module.json          # Module metadata and build configuration
├── mymodule.nano        # My FFI declarations
├── mymodule_helpers.c   # Optional: C helper functions
└── mymodule_helpers.nano # Optional: My helper function declarations
```

## Step 1: Create Module Directory

```bash
mkdir modules/mymodule
```

## Step 2: Create `module.json`

The `module.json` file tells my build system how to compile and link your module.

### Example: Pure FFI Module (no C code)

```json
{
  "name": "stdio",
  "c_sources": [],
  "dependencies": []
}
```

### Example: Module with C Helpers

```json
{
  "name": "sdl_helpers",
  "c_sources": ["sdl_helpers.c"],
  "dependencies": ["sdl"]
}
```

### Example: Module with External Library

```json
{
  "name": "sdl_mixer",
  "pkg_config": ["SDL2_mixer"],
  "dependencies": ["sdl"]
}
```

### Full Configuration Options

```json
{
  "name": "mymodule",           // Module name (matches directory)
  "c_sources": [                // C files to compile (optional)
    "mymodule_helpers.c"
  ],
  "pkg_config": [                // System libraries via pkg-config (optional)
    "libname"
  ],
  "link_flags": [                // Manual linker flags (optional)
    "-framework CoreAudio"
  ],
  "dependencies": [              // Other modules I require (optional)
    "sdl"
  ]
}
```

## Step 3: Create FFI Declarations (`mymodule.nano`)

Declare external C functions with `extern fn`:

```nano
# Simple functions
extern fn c_function_name(arg1: int, arg2: string) -> int

# Void return
extern fn do_something(value: float) -> void

# Pointer types (use int)
extern fn get_pointer() -> int
extern fn use_pointer(ptr: int) -> void

# Constants
let MY_CONSTANT: int = 42
let MY_FLAG_A: int = 1
let MY_FLAG_B: int = 2
```

### Type Mapping

| My Type | C Equivalent |
|----------|--------------|
| `int` | `int64_t` |
| `float` | `double` |
| `string` | `const char*` |
| `bool` | `bool` (0 or 1) |
| `void` | `void` |
| (pointers) | Cast to/from `int` |

## Step 4: Optional C Helpers

If you need custom C code, create `mymodule_helpers.c`:

```c
#include <stdint.h>
#include <stdbool.h>

// Helper function I can call
int64_t nl_mymodule_helper(int64_t arg1, const char* arg2) {
    // Your C code here
    return 42;
}
```

Then declare it in `mymodule_helpers.nano`:

```nano
extern fn nl_mymodule_helper(arg1: int, arg2: string) -> int
```

### Naming Convention

- Prefix C functions with `nl_` to avoid name collisions.
- Use descriptive names: `nl_sdl_render_fill_rect` not `nl_rfr`.

## Step 5: Using Your Module

In the code you write for me:

```nano
import "modules/mymodule/mymodule.nano"

fn main() -> int {
    let result: int = (c_function_name 10 "hello")
    return 0
}
```

## Complete Example: stdio Module

### `modules/stdio/module.json`

```json
{
  "name": "stdio",
  "c_sources": [],
  "dependencies": []
}
```

### `modules/stdio/stdio.nano`

```nano
# File operations
extern fn fopen(filename: string, mode: string) -> int
extern fn fclose(file: int) -> int
extern fn fread(ptr: int, size: int, count: int, file: int) -> int
extern fn fseek(file: int, offset: int, whence: int) -> int
extern fn ftell(file: int) -> int

# Constants
let SEEK_SET: int = 0
let SEEK_CUR: int = 1
let SEEK_END: int = 2
let FILE_MODE_READ: string = "rb"

# Helper function in my syntax
fn file_size(filename: string) -> int {
    let file: int = (fopen filename FILE_MODE_READ)
    if (== file 0) {
        return -1
    } else {
        (fseek file 0 SEEK_END)
        let size: int = (ftell file)
        (fclose file)
        return size
    }
}

# I require shadow tests for my functions
shadow file_size {
    assert (>= (file_size "/nonexistent") -1)
}
```

### Usage

```nano
import "modules/stdio/stdio.nano"

fn main() -> int {
    let size: int = (file_size "myfile.txt")
    (println size)
    return 0
}
```

## Complete Example: SDL Helpers

### `modules/sdl_helpers/module.json`

```json
{
  "name": "sdl_helpers",
  "c_sources": ["sdl_helpers.c"],
  "dependencies": ["sdl"]
}
```

### `modules/sdl_helpers/sdl_helpers.c`

```c
#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdbool.h>

// Fill rectangle helper
void nl_sdl_render_fill_rect(SDL_Renderer* renderer, 
                              int64_t x, int64_t y, 
                              int64_t w, int64_t h) {
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    SDL_RenderFillRect(renderer, &rect);
}

// Check if window should close
bool nl_sdl_window_should_close() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            return true;
        }
    }
    return false;
}
```

### `modules/sdl_helpers/sdl_helpers.nano`

```nano
extern fn nl_sdl_render_fill_rect(renderer: int, x: int, y: int, w: int, h: int) -> void
extern fn nl_sdl_window_should_close() -> bool
```

### Usage

```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_helpers/sdl_helpers.nano"

fn main() -> int {
    (SDL_Init 32)
    let window: int = (SDL_CreateWindow "Test" 100 100 800 600 4)
    let renderer: int = (SDL_CreateRenderer window -1 6)
    
    let mut running: bool = true
    while running {
        (SDL_SetRenderDrawColor renderer 255 0 0 255)
        (SDL_RenderClear renderer)
        
        (nl_sdl_render_fill_rect renderer 100 100 200 200)
        
        (SDL_RenderPresent renderer)
        (SDL_Delay 16)
        
        if (nl_sdl_window_should_close) {
            set running false
        } else {}
    }
    
    (SDL_DestroyRenderer renderer)
    (SDL_DestroyWindow window)
    (SDL_Quit)
    return 0
}
```

## Building and Testing

### Automatic Building

I build modules automatically when you use them:

```bash
# Compile your program
./bin/nanoc examples/myprogram.nano -o myprogram

# I compile modules as needed
```

### Manual Testing

```bash
# Test with my interpreter (no C compilation)
./bin/nano examples/myprogram.nano

# Compile to native binary
./bin/nanoc examples/myprogram.nano -o myprogram
./myprogram
```

## Best Practices

### 1. Start Simple

Begin with pure FFI bindings:

```nano
extern fn existing_c_function(arg: int) -> int
```

### 2. Add Helpers as Needed

Write C helpers only when:
- The C API is too complex for direct FFI.
- You must marshal data structures.
- Performance requires a C implementation.

### 3. Use Descriptive Names

```nano
# Good
extern fn nl_sdl_render_filled_rectangle(...)

# Bad
extern fn rfr(...)
```

### 4. Document Your Module

Add comments explaining what each function does.

```nano
# Load audio file from disk
# Returns: audio handle (int) or 0 on failure
# Example: let audio: int = (load_audio "sound.wav")
extern fn load_audio(filename: string) -> int
```

### 5. Add Shadow Tests

I require you to test your helper functions.

```nano
fn clamp(value: int, min_val: int, max_val: int) -> int {
    if (< value min_val) {
        return min_val
    } else {
        if (> value max_val) {
            return max_val
        } else {
            return value
        }
    }
}

shadow clamp {
    assert (== (clamp 5 0 10) 5)
    assert (== (clamp -5 0 10) 0)
    assert (== (clamp 15 0 10) 10)
}
```

## Common Patterns

### Pattern 1: Opaque Handles

C libraries often return pointers. I treat them as `int`.

```nano
# SDL_Window* becomes int
extern fn SDL_CreateWindow(...) -> int

# Use the handle
let window: int = (SDL_CreateWindow "Title" 0 0 800 600 4)
if (== window 0) {
    (println "Failed to create window")
} else {
    # Use window
    (SDL_DestroyWindow window)
}
```

### Pattern 2: Flags and Constants

```nano
# Define flags as constants
let SDL_INIT_VIDEO: int = 32
let SDL_INIT_AUDIO: int = 16
let SDL_INIT_EVERYTHING: int = 62097

# Combine with bitwise OR (use + for now)
let flags: int = (+ SDL_INIT_VIDEO SDL_INIT_AUDIO)
(SDL_Init flags)
```

### Pattern 3: Error Checking

```nano
extern fn SDL_GetError() -> string

fn init_sdl() -> bool {
    let result: int = (SDL_Init 32)
    if (!= result 0) {
        (println "SDL_Init failed:")
        (println (SDL_GetError))
        return false
    } else {
        return true
    }
}
```

### Pattern 4: Resource Management

```nano
fn load_and_process_file(filename: string) -> bool {
    let file: int = (fopen filename "rb")
    if (== file 0) {
        return false
    } else {
        # Process file
        # ...
        
        # Always cleanup
        (fclose file)
        return true
    }
}
```

## Troubleshooting

### Module Not Found

```
Error: Cannot find module 'mymodule'
```

Check that:
1. Directory exists: `modules/mymodule/`.
2. `module.json` exists.
3. `mymodule.nano` exists.
4. Import path is correct: `import "modules/mymodule/mymodule.nano"`.

### Linking Errors

```
Undefined symbols: _my_function
```

1. Add function to `c_sources` in `module.json`.
2. Add library to `pkg_config`.
3. Add to `link_flags`.

### Type Mismatches

```
Error: Type mismatch in function call
```

Check my type mapping:
- C `int` to my `int` (int64_t).
- C `double` to my `float`.
- C `char*` to my `string`.
- C pointers to my `int`.

## Examples

I provide working modules here:
- `modules/sdl/` - SDL2 bindings.
- `modules/sdl_helpers/` - SDL2 with C helpers.
- `modules/sdl_mixer/` - External library (SDL2_mixer).
- `modules/stdio/` - Standard C library bindings.
- `modules/math_ext/` - Math extensions.

All examples in my `examples/` directory demonstrate how I use modules.

## Next Steps

1. Browse my existing modules for examples.
2. Create your first simple FFI module.
3. Add C helpers if you need them.
4. Document your module.
5. Share it.

## Reference

- [Module Format / System](MODULE_SYSTEM.md)
- [SDL Extensions Guide](../modules/SDL_EXTENSIONS.md)
- [Standard Library Reference](STDLIB.md)
- [FFI Best Practices](../planning/MODULE_FFI_IMPLEMENTATION.md)

---

I wish you success with your modules.

# FFI Guide: Creating C Modules for Nanolang

This guide teaches you how to create C modules that integrate seamlessly with nanolang, enabling access to C libraries, system APIs, and high-performance code.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Module Structure](#module-structure)
4. [Type Marshaling](#type-marshaling)
5. [Memory Management](#memory-management)
6. [Advanced Topics](#advanced-topics)
7. [Best Practices](#best-practices)

## Overview

### What is FFI?

FFI (Foreign Function Interface) allows nanolang to call C functions. This enables:
- Access to C libraries (SDL, SQLite, curl, etc.)
- System API calls (file I/O, networking, etc.)
- Performance-critical code in C
- Reuse of existing C codebases

### How It Works

```
┌─────────────┐
│ .nano file  │  Nanolang code imports module
│ import "X"  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ module.json │  Build metadata (headers, libs, etc.)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ .nano API   │  Nanolang function signatures
│ extern fns  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ .c/.h files │  C implementation
│ Wrappers    │
└─────────────┘
```

**Compiler Path**: nanoc generates C code that includes module headers and links against compiled modules

**Interpreter Path**: nano loads `.so`/`.dylib` shared libraries and calls functions via dlsym()

## Quick Start

### Example: Simple Math Module

Let's create a module that exposes C math functions to nanolang.

**1. Create module directory**:
```bash
mkdir -p modules/mymath
cd modules/mymath
```

**2. Create `module.json`**:
```json
{
  "name": "mymath",
  "version": "1.0.0",
  "description": "Custom math utilities",
  "c_sources": ["mymath.c"],
  "system_libs": ["m"]
}
```

**3. Create `mymath.c`**:
```c
#include <math.h>
#include <stdint.h>

// Power function
double nl_pow(double base, double exponent) {
    return pow(base, exponent);
}

// Absolute value
int64_t nl_abs(int64_t x) {
    return (x < 0) ? -x : x;
}
```

**4. Create `mymath.nano`**:
```nano
// Function declarations (extern = implemented in C)
extern fn nl_pow(base: float, exponent: float) -> float
extern fn nl_abs(x: int) -> int

// Optional: Convenience wrappers
pub fn power(base: float, exp: float) -> float {
    return (nl_pow base exp)
}

pub fn absolute(x: int) -> int {
    return (nl_abs x)
}
```

**5. Use the module**:
```nano
import "modules/mymath/mymath.nano" as Math

fn main() -> int {
    let result: float = (Math.power 2.0 8.0)  // 256.0
    let abs_val: int = (Math.absolute (- 42))  // 42
    (println (float_to_string result))
    return 0
}
```

**6. Compile and run**:
```bash
cd ../..  # Back to project root
./bin/nanoc examples/use_mymath.nano -o test
./test
```

That's it! The module is automatically built and linked.

## Module Structure

### Required Files

```
modules/mymath/
├── module.json       # Build metadata (REQUIRED)
├── mymath.nano       # Nanolang API (REQUIRED)
├── mymath.c          # C implementation
├── mymath.h          # C headers (optional)
└── README.md         # Documentation (recommended)
```

### module.json Format

```json
{
  "name": "module_name",
  "version": "1.0.0",
  "description": "What this module does",
  
  // C compilation
  "c_sources": ["file1.c", "file2.c"],
  "headers": ["module.h"],
  "include_dirs": ["include/"],
  "cflags": ["-O3", "-Wall"],
  
  // Linking
  "system_libs": ["pthread", "m"],
  "ldflags": ["-L/usr/local/lib"],
  "frameworks": ["CoreFoundation"],  // macOS only
  
  // Dependencies
  "pkg_config": ["sdl2", "gtk+-3.0"],
  "dependencies": ["other_module"],
  
  // Package management
  "apt_packages": ["libsdl2-dev"],
  "brew_packages": ["sdl2"],
  "dnf_packages": ["SDL2-devel"]
}
```

### .nano API File

This declares the interface between nanolang and C:

```nano
// Extern functions (implemented in C)
extern fn nl_open_file(path: string, mode: string) -> opaque
extern fn nl_read_line(file: opaque) -> string
extern fn nl_close_file(file: opaque) -> int

// Opaque types (C pointers)
// Use 'opaque' for FILE*, SDL_Window*, etc.

// Optional: Nanolang wrappers
pub fn open_for_reading(path: string) -> opaque {
    return (nl_open_file path "r")
}
```

## Type Marshaling

### Supported Types

| Nanolang Type | C Type | Notes |
|---------------|--------|-------|
| `int` | `int64_t` | Always 64-bit |
| `float` | `double` | Always 64-bit |
| `bool` | `bool` | C99 bool |
| `string` | `const char*` | Null-terminated |
| `opaque` | `void*` | C pointers (FILE*, etc.) |

### Type Mapping Examples

**Integers**:
```c
// Nanolang: fn add(a: int, b: int) -> int
int64_t nl_add(int64_t a, int64_t b) {
    return a + b;
}
```

**Floats**:
```c
// Nanolang: fn sqrt(x: float) -> float
double nl_sqrt(double x) {
    return sqrt(x);
}
```

**Booleans**:
```c
// Nanolang: fn is_positive(x: int) -> bool
bool nl_is_positive(int64_t x) {
    return x > 0;
}
```

**Strings**:
```c
// Nanolang: fn string_length(s: string) -> int
int64_t nl_string_length(const char* s) {
    return (int64_t)strlen(s);
}
```

**Opaque Pointers**:
```c
// Nanolang: fn fopen(path: string, mode: string) -> opaque
void* nl_fopen(const char* path, const char* mode) {
    return fopen(path, mode);
}

// Nanolang: fn fclose(file: opaque) -> int
int64_t nl_fclose(void* file) {
    return (int64_t)fclose((FILE*)file);
}
```

### Arrays (Advanced)

Arrays use the `DynArray` type:

```c
#include "runtime/dyn_array.h"

// Nanolang: fn sum_array(arr: array<int>) -> int
int64_t nl_sum_array(DynArray* arr) {
    int64_t sum = 0;
    for (int64_t i = 0; i < arr->length; i++) {
        sum += dyn_array_get_int(arr, i);
    }
    return sum;
}

// Nanolang: fn create_range(n: int) -> array<int>
DynArray* nl_create_range(int64_t n) {
    DynArray* arr = dyn_array_new(ELEM_INT);
    for (int64_t i = 0; i < n; i++) {
        arr = dyn_array_push_int(arr, i);
    }
    return arr;
}
```

## Memory Management

### Rules

1. **Strings**: Nanolang manages memory. C receives `const char*` — don't free!
2. **Return strings**: Allocate with `gc_alloc_string()` (preferred) or `strdup()` — nanolang's GC will manage the result
3. **Opaque pointers**: C allocates with `malloc()`. If the module's metadata includes `requires_manual_free=true` and a cleanup function, ARC wrapping handles deallocation automatically. Otherwise, manual free is required.
4. **Arrays**: Use GC for DynArray allocation

### String Memory

**✅ Correct: Return GC-allocated string**
```c
#include "runtime/gc.h"

const char* nl_get_name() {
    return gc_alloc_string("Alice");  // GC-managed, no manual free needed
}
```

**Also acceptable (legacy): Return strdup'd string**
```c
const char* nl_get_name() {
    return strdup("Alice");  // Works but gc_alloc_string() preferred
}
```

**❌ Wrong: Return stack string**
```c
const char* nl_get_name() {
    char name[] = "Alice";
    return name;  // DANGLING POINTER!
}
```

**✅ Correct: Receive string as const**
```c
void nl_print(const char* str) {
    printf("%s\n", str);
    // Don't free str!
}
```

### Opaque Pointer Memory

**✅ Correct: Caller frees**
```c
// Open: allocates
void* nl_open_file(const char* path) {
    return fopen(path, "r");
}

// Close: frees
int64_t nl_close_file(void* file) {
    return fclose((FILE*)file);
}
```

**Nanolang usage**:
```nano
let file: opaque = (nl_open_file "data.txt")
# Use file...
(nl_close_file file)  # Must explicitly close
```

### ARC Wrapping (Automatic)

If a module's `module.json` metadata marks a function with `requires_manual_free: true` and provides a `cleanup_function`, the runtime automatically wraps the returned pointer in a GC envelope. When the variable goes out of scope, the cleanup function is called.

This works transparently for types like Regex and HashMap. The C code uses plain `malloc`/`free` — the compiler inserts wrapping at call boundaries.

**Memory semantics metadata** (auto-generated from return types):
- `returns_gc_managed: true` — String returns are GC-tracked
- `requires_manual_free: true` — Opaque returns need cleanup
- `cleanup_function: "regex_free"` — Inferred from function name prefix
- `returns_borrowed: true` — Return is a borrowed reference (no wrapping)

### GC Integration

For nanolang-managed memory:

```c
#include "runtime/gc.h"

DynArray* nl_create_array() {
    // GC-managed allocation
    DynArray* arr = gc_alloc(sizeof(DynArray), GC_TYPE_ARRAY);
    arr->length = 0;
    arr->capacity = 10;
    arr->elem_type = ELEM_INT;
    arr->data = gc_alloc(10 * sizeof(int64_t), GC_TYPE_ARRAY);
    return arr;  // Nanolang's GC will handle cleanup
}
```

## Advanced Topics

### Platform-Specific Code

```c
#ifdef __APPLE__
    #include <CoreFoundation/CoreFoundation.h>
    // macOS implementation
#elif defined(__linux__)
    #include <linux/limits.h>
    // Linux implementation
#elif defined(_WIN32)
    #include <windows.h>
    // Windows implementation
#endif
```

**module.json**:
```json
{
  "frameworks": ["CoreFoundation"],  // macOS only
  "system_libs": ["pthread"]         // Cross-platform
}
```

### pkg-config Integration

For libraries with pkg-config:

```json
{
  "pkg_config": ["sdl2", "sdl2_ttf", "sdl2_mixer"]
}
```

Nanolang automatically runs:
```bash
pkg-config --cflags sdl2
pkg-config --libs sdl2
```

### Multiple Source Files

```json
{
  "c_sources": [
    "module_main.c",
    "helpers.c",
    "platform_linux.c"
  ],
  "headers": ["module.h", "helpers.h"]
}
```

### Wrapper Functions

Create nanolang-friendly wrappers around C APIs:

```c
// Raw SDL API: complex
SDL_Rect rect = {x, y, w, h};
SDL_RenderFillRect(renderer, &rect);

// Wrapper: simplified
int64_t nl_sdl_render_fill_rect(void* renderer, int64_t x, int64_t y, 
                                  int64_t w, int64_t h) {
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    return SDL_RenderFillRect((SDL_Renderer*)renderer, &rect);
}
```

### Error Handling

**Pattern 1: Return error code**
```c
// Returns 0 on success, -1 on error
int64_t nl_write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    
    fputs(content, f);
    fclose(f);
    return 0;
}
```

**Pattern 2: Return opaque (NULL on error)**
```c
// Returns pointer or NULL
void* nl_open_database(const char* path) {
    sqlite3* db;
    if (sqlite3_open(path, &db) != SQLITE_OK) {
        return NULL;
    }
    return db;
}
```

Check in nanolang:
```nano
let db: opaque = (nl_open_database "test.db")
if (== db (cast 0 opaque)) {
    (println "Failed to open database")
    return 1
}
```

## Best Practices

### 1. Prefix All Functions

```c
// ✅ Good: Prefixed
int64_t nl_mymodule_function();

// ❌ Bad: Name collision risk
int64_t function();
```

### 2. Use Standard Types

```c
// ✅ Good: Portable
#include <stdint.h>
int64_t nl_add(int64_t a, int64_t b);

// ❌ Bad: Platform-specific
long nl_add(long a, long b);
```

### 3. Document Memory Ownership

```c
/**
 * Open a file for reading.
 * 
 * @param path File path
 * @return File handle (caller must close with nl_close_file)
 *         or NULL on error
 */
void* nl_open_file(const char* path);

/**
 * Close a file.
 * 
 * @param file File handle from nl_open_file (will be freed)
 * @return 0 on success, -1 on error
 */
int64_t nl_close_file(void* file);
```

### 4. Handle NULL Pointers

```c
int64_t nl_string_length(const char* str) {
    if (!str) return 0;  // Guard against NULL
    return (int64_t)strlen(str);
}
```

### 5. Test Your Module

**test_mymodule.nano**:
```nano
import "modules/mymodule/mymodule.nano" as M

fn test_basic() -> int {
    let result: int = (M.some_function 42)
    assert (== result 42)
    return 0
}

shadow test_basic {
    assert (== (test_basic) 0)
}
```

### 6. Provide Examples

**examples/mymodule_demo.nano**:
```nano
import "modules/mymodule/mymodule.nano" as M

fn main() -> int {
    // Show typical usage
    let result: int = (M.function1 arg1)
    (println (int_to_string result))
    return 0
}
```

## Real-World Example: SQLite Module

See `modules/sqlite/` for a complete example:

**module.json**:
```json
{
  "name": "sqlite",
  "c_sources": ["sqlite_helpers.c"],
  "system_libs": ["sqlite3"]
}
```

**sqlite.nano** (excerpt):
```nano
extern fn nl_sqlite3_open(filename: string) -> opaque
extern fn nl_sqlite3_close(db: opaque) -> int
extern fn nl_sqlite3_exec(db: opaque, sql: string) -> int
extern fn nl_sqlite3_prepare_v2(db: opaque, sql: string) -> opaque
// ... more functions
```

**sqlite_helpers.c** (excerpt):
```c
#include <sqlite3.h>

void* nl_sqlite3_open(const char* filename) {
    sqlite3* db;
    if (sqlite3_open(filename, &db) != SQLITE_OK) {
        return NULL;
    }
    return db;
}

int64_t nl_sqlite3_close(void* db) {
    return sqlite3_close((sqlite3*)db);
}
```

## Debugging FFI Modules

### Compilation Issues

```bash
# Verbose build
NANO_VERBOSE_BUILD=1 ./bin/nanoc program.nano

# Check module compilation
ls -la modules/mymodule/.build/
```

### Runtime Issues

**Compiler mode**:
```bash
# Check linker output
./bin/nanoc program.nano -o test -v
```

**Interpreter mode**:
```bash
# Enable FFI verbose mode
./bin/nano program.nano  # Shows module loading
```

### Common Errors

**Error: Undefined reference to `nl_function`**
- Function not in .c file
- Function name mismatch
- Module not in module.json c_sources

**Error: Cannot find module header**
- Check module.json "headers" field
- Verify file exists
- Check include_dirs

**Segmentation fault**
- NULL pointer dereference
- Memory corruption
- Type mismatch (int64_t vs int)
- String not null-terminated

## Further Reading

- [Module Reference](MODULES.md) - Available modules
- [Memory Management](MEMORY_MANAGEMENT.md) - GC details
- [SQLite Example](../modules/sqlite/) - Complete module
- [SDL Example](../modules/sdl/) - Graphics module

## Getting Help

- Check `modules/` for examples
- Read existing module source code
- Ask on GitHub Discussions
- Report bugs on GitHub Issues

Happy module building! 🛠️


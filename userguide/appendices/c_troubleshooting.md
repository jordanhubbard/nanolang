# Appendix C: Troubleshooting Guide

**Solutions and workarounds for common problems.**

## C.1 Common Compile Errors

### "Undefined function X"

**Symptom:** Compiler reports `undefined function 'foo'` or `unknown identifier 'foo'`.

**Causes:**
1. Function not imported from module
2. Typo in function name
3. Function defined after it's called (NanoLang requires forward declaration or definition before use)

**Solutions:**

```nano
# Problem: Using a module function without importing
fn main() -> int {
    (println (json_encode data))  # Error: undefined function 'json_encode'
    return 0
}

# Fix: Add the import
from "modules/json/json.nano" import json_encode

fn main() -> int {
    (println (json_encode data))  # Now works
    return 0
}
```

**Checklist:**
- ✅ Check spelling of function name
- ✅ Verify the module is imported with correct path
- ✅ Ensure function is defined before it's called
- ✅ Check that function is marked `pub` if in another module

### "Type mismatch"

**Symptom:** Compiler reports type mismatch between expected and actual types.

**Common Causes:**

```nano
# Problem 1: Wrong return type
fn get_count() -> string {  # Declared as string
    return 42              # Error: returning int
}

# Fix: Match types
fn get_count() -> int {
    return 42
}

# Problem 2: Wrong argument type
fn greet(name: string) -> void {
    (println name)
}
fn main() -> int {
    (greet 123)  # Error: passing int to string parameter
    return 0
}

# Fix: Pass correct type
fn main() -> int {
    (greet "World")
    return 0
}
```

**Checklist:**
- ✅ Verify function return type matches actual return value
- ✅ Check that arguments match parameter types
- ✅ Use explicit type conversions: `(int_to_string n)`, `(string_to_int s)`
- ✅ For arrays, ensure element type matches: `array<int>` vs `array<string>`

### "Missing shadow test"

**Symptom:** Compiler warns or errors about missing shadow test for a function.

**Cause:** Every function in NanoLang (except `extern` functions) MUST have a shadow test.

```nano
# Problem: Function without shadow test
fn add(a: int, b: int) -> int {
    return (+ a b)
}
# Error: Missing shadow test for 'add'

# Fix: Add shadow test block
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

**Notes:**
- Shadow tests run at compile time
- They verify your function works before the binary is created
- The `main` function can have a trivial shadow test: `shadow main { assert true }`

### "Infinite loop during compilation"

**Symptom:** Compiler hangs, CPU spins at 100%, no output.

**Causes:**
1. Recursive type definition
2. Circular imports
3. Compiler bug (rare)

**Solutions:**

```bash
# Always compile with a timeout
perl -e 'alarm 30; exec @ARGV' ./bin/nanoc file.nano -o bin/output

# If it hangs, check for:
# 1. Circular imports
# 2. Self-referential types without indirection
# 3. Large generated code (string literals > 1MB)
```

**Debugging Steps:**
1. Simplify your code - remove functions until it compiles
2. Check import graph for cycles
3. Try `./bin/nanoc --verbose file.nano` for more output
4. Report persistent hangs as bugs

## C.2 Runtime Issues

### Segmentation Faults

**Symptom:** Program crashes with "Segmentation fault" or "Bus error".

**Common Causes:**

1. **Array out of bounds:**
```nano
fn bad_access() -> int {
    let arr: array<int> = [1, 2, 3]
    return (at arr 10)  # Crash: index 10 out of bounds (size 3)
}

# Fix: Check bounds first
fn safe_access(arr: array<int>, idx: int) -> int {
    if (>= idx (array_length arr)) {
        return -1  # Default value
    }
    return (at arr idx)
}
```

2. **Use after free (with extern/FFI):**
```nano
unsafe {
    let ptr: Opaque = (some_c_alloc)
    (some_c_free ptr)
    (some_c_use ptr)  # Crash: ptr already freed
}
```

3. **Null pointer from C library:**
```nano
unsafe {
    let result: string = (ffi_function_that_returns_null)
    (println result)  # Crash if result is NULL
}
```

**Prevention:**
- Always bounds-check array access
- Track resource lifecycles carefully in unsafe blocks
- Check return values from FFI functions

### Memory Leaks

**Symptom:** Program memory usage grows over time, system slows down.

**Common Causes:**

1. **Not freeing C resources:**
```nano
unsafe {
    let window: SDL_Window = (SDL_CreateWindow "Test" 0 0 800 600 0)
    # ... use window ...
    # Forgot: (SDL_DestroyWindow window)
}

# Fix: Always pair create/destroy
unsafe {
    let window: SDL_Window = (SDL_CreateWindow "Test" 0 0 800 600 0)
    # ... use window ...
    (SDL_DestroyWindow window)  # Clean up
}
```

2. **Accumulating data in loops:**
```nano
# Problem: Strings accumulate
fn bad_loop() -> string {
    let mut result: string = ""
    for i in (range 0 1000000) {
        set result (+ result "x")  # Creates new string each iteration
    }
    return result
}

# Fix: Use StringBuilder for large string building
from "stdlib/StringBuilder.nano" import sb_new, sb_append, sb_to_string

fn good_loop() -> string {
    let sb: StringBuilder = (sb_new)
    for i in (range 0 1000000) {
        (sb_append sb "x")
    }
    return (sb_to_string sb)
}
```

### Unexpected Behavior

**Symptom:** Program runs but produces wrong results.

**Debugging Steps:**

1. **Add debug prints:**
```nano
fn mysterious_function(x: int) -> int {
    (println (+ "Input: " (int_to_string x)))
    let result: int = (* x 2)
    (println (+ "Result: " (int_to_string result)))
    return result
}
```

2. **Use structured logging:**
```nano
from "stdlib/log.nano" import log_debug, set_log_level, LOG_LEVEL_DEBUG

fn main() -> int {
    (set_log_level LOG_LEVEL_DEBUG)
    (log_debug "mymodule" "Starting processing")
    # ... your code ...
    return 0
}
```

3. **Add more shadow test cases:**
```nano
shadow my_function {
    # Test edge cases
    assert (== (my_function 0) expected_0)
    assert (== (my_function -1) expected_neg)
    assert (== (my_function 999999) expected_large)
}
```

## C.3 Module Installation

### Module Not Found

**Symptom:** `Error: cannot find module 'modules/foo/foo.nano'`

**Solutions:**

1. **Check the path:**
```bash
# Modules live in the modules/ directory
ls modules/
# Should see: sdl/, json/, ncurses/, etc.

# Correct import path
from "modules/sdl/sdl.nano" import SDL_Init
```

2. **Check if module is built:**
```bash
# Build all modules
make modules

# Or build specific module
make -C modules/sdl
```

3. **Check module dependencies:**
```bash
# Some modules require system libraries
# SDL requires: brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer
# SQLite requires: brew install sqlite3
# ncurses is usually pre-installed
```

### Dependency Conflicts

**Symptom:** Module compiles but crashes, or linker errors.

**Solutions:**

1. **Version mismatch:**
```bash
# Check installed version
pkg-config --modversion sdl2

# Reinstall to get consistent version
brew reinstall sdl2
```

2. **Multiple library versions:**
```bash
# On macOS, check for conflicts
brew doctor

# Clean up old versions
brew cleanup
```

### Platform-Specific Issues

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install common dependencies
brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer
brew install sqlite3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install build-essential
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev
sudo apt-get install libsqlite3-dev
sudo apt-get install libncurses5-dev
```

## C.4 Platform-Specific Issues

### macOS

**Issue: "Library not loaded" error**
```bash
# Fix: Set library path
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

**Issue: Gatekeeper blocks binary**
```bash
# If macOS says "cannot be opened because the developer cannot be verified"
xattr -d com.apple.quarantine ./bin/your_program
```

**Issue: OpenGL deprecated warnings**
```
# macOS marks OpenGL as deprecated but it still works
# Add to suppress warnings in your code:
# These are informational only - OpenGL functions still work
```

**Issue: Apple Silicon (M1/M2) compatibility**
```bash
# NanoLang compiles natively on ARM64
# If you have Rosetta issues:
arch -arm64 make clean
arch -arm64 make
```

### Linux

**Issue: Missing libGL**
```bash
# Install OpenGL libraries
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev
```

**Issue: Permission denied on /dev/fb0**
```bash
# For framebuffer access (some graphics)
sudo usermod -a -G video $USER
# Then log out and back in
```

**Issue: ncurses terminal issues**
```bash
# Set terminal type
export TERM=xterm-256color
```

**Issue: pkg-config not finding libraries**
```bash
# Add library paths
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### Windows

**Issue: NanoLang requires WSL2**

NanoLang currently requires a Unix-like environment. On Windows:

```powershell
# Install WSL2
wsl --install

# Choose Ubuntu as your distribution
# Then follow Linux instructions inside WSL
```

**Issue: File path differences**
```nano
# In WSL, Windows drives are mounted at /mnt/
# C:\Users\me\project becomes:
let path: string = "/mnt/c/Users/me/project"
```

**Issue: GUI applications in WSL**
```bash
# WSL2 supports GUI apps natively in Windows 11
# For Windows 10, install an X server:
# 1. Install VcXsrv or Xming
# 2. In WSL: export DISPLAY=:0
```

---

**Previous:** [Appendix B: Quick Reference](b_quick_reference.html)  
**Next:** [Appendix D: Glossary](d_glossary.html)

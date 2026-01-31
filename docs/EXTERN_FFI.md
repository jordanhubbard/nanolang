# External C Function Interface (FFI)

**Version:** 2.0  
**Date:** November 15, 2025  
**Status:** ✅ First-Class Feature

---

## Overview

nanolang supports calling external C functions through the `extern` keyword. This is a **first-class feature** that allows nanolang to seamlessly integrate with the C ecosystem, enabling hybrid C/nanolang applications.

**Key Principle: First-Class Integration**

Extern functions are treated as first-class citizens in nanolang:
- **Cannot be shadowed or redefined** - Extern functions are immutable declarations
- **Cannot have shadow tests** - They are C functions, not nanolang functions
- **Type-checked at compile time** - Full type safety for extern function calls
- **Seamless integration** - Use C libraries directly without wrappers

This design allows nanolang to work cooperatively with the C ecosystem without attempting to wrap everything. You can use any C library (SDL2, OpenGL, system libraries, etc.) directly from nanolang.

---

## Syntax

### Declaring External Functions

```nano
extern fn function_name(param1: type1, param2: type2) -> return_type
```

**Key Points:**
- Use the `extern` keyword before `fn`
- Provide the exact C function signature matching the C library
- No function body - just the declaration
- **No shadow test required** - Extern functions cannot have shadow tests
- **Cannot be redefined** - Extern functions are immutable

### Example

```nano
# Declare external C math functions
extern fn sqrt(x: float) -> float
extern fn pow(x: float, y: float) -> float
extern fn sin(x: float) -> float

# Use them in nanolang code
fn calculate_hypotenuse(a: float, b: float) -> float {
    let a_squared: float = (pow a 2.0)
    let b_squared: float = (pow b 2.0)
    let sum: float = (+ a_squared b_squared)
    return (sqrt sum)
}
```

---

## First-Class Extern Function Rules

### 1. Extern Functions Cannot Be Shadowed

Extern functions are **immutable declarations** that cannot be redefined or shadowed by regular nanolang functions:

```nano
extern fn SDL_Init(flags: int) -> int

# ERROR: Cannot shadow extern function
fn SDL_Init(flags: int) -> int {
    return 0
}
```

**Error Message:**
```
Error: Function 'SDL_Init' cannot shadow extern function
  Extern functions are first-class and cannot be shadowed
  Choose a different function name
```

### 2. Extern Functions Cannot Have Shadow Tests

Extern functions are C functions that cannot be tested in the nanolang interpreter:

```nano
extern fn SDL_Init(flags: int) -> int

# ERROR: Shadow test cannot be attached to extern function
shadow SDL_Init {
    assert (== (SDL_Init 0) 0)
}
```

**Error Message:**
```
Error: Shadow test cannot be attached to extern function 'SDL_Init'
  Extern functions are C functions and cannot be tested in the interpreter
  Remove the shadow test or test a wrapper function instead
```

### 3. Functions Using Extern Functions Don't Require Shadow Tests

Regular nanolang functions that call extern functions don't require shadow tests (they're automatically skipped):

```nano
extern fn SDL_Init(flags: int) -> int

fn main() -> int {
    let result: int = (SDL_Init 32)
    return result
}

# Shadow test is optional - will be skipped if present
shadow main {
    assert (== (main) 0)  # Skipped at runtime (uses extern functions)
}
```

### 4. Extern Functions Are Type-Checked

All extern function calls are fully type-checked at compile time:

```nano
extern fn SDL_Init(flags: int) -> int

fn main() -> int {
    # ERROR: Wrong argument type
    let result: int = (SDL_Init "invalid")
    return result
}
```

---

## Type Mapping

### nanolang → C Type Mapping

| nanolang Type | C Type     | Notes                          |
|---------------|------------|--------------------------------|
| `int`         | `int64_t`  | 64-bit signed integer          |
| `float`       | `double`   | Double-precision floating point|
| `bool`        | `bool`     | C99 boolean (0 or 1)           |
| `string`      | `const char*` | Null-terminated C string    |
| `void`        | `void`     | No return value                |

**Important:**
- nanolang `int` is always 64-bit, C `int` is platform-dependent
- nanolang `float` is always double precision
- Strings are immutable in nanolang, passed as `const char*` to C
- **Pointer types** (like `SDL_Window*`) are represented as `int` in nanolang and cast appropriately by the transpiler

---

## Safe Function Categories

### Category 1: Math Functions (100% Safe)

All C math library functions are safe because they:
- Operate on scalar values (no pointers)
- Have no side effects (except errno)
- Cannot cause memory issues

**Example:**

```nano
extern fn asin(x: float) -> float
extern fn acos(x: float) -> float
extern fn atan(x: float) -> float
extern fn exp(x: float) -> float
extern fn log(x: float) -> float
extern fn cbrt(x: float) -> float
extern fn hypot(x: float, y: float) -> float
extern fn trunc(x: float) -> float
extern fn fmod(x: float, y: float) -> float
```

See `examples/21_extern_math.nano` for comprehensive examples.

### Category 2: Character Classification (100% Safe)

Character functions operate on single integers (ASCII values):

```nano
extern fn isalpha(c: int) -> int
extern fn isdigit(c: int) -> int
extern fn isalnum(c: int) -> int
extern fn isspace(c: int) -> int
extern fn isupper(c: int) -> int
extern fn islower(c: int) -> int
extern fn toupper(c: int) -> int
extern fn tolower(c: int) -> int
```

See `examples/22_extern_char.nano` for comprehensive examples.

### Category 3: String Read-Only (Safe with Caution)

Read-only string functions are safe if strings are null-terminated:

```nano
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int
extern fn strncmp(s1: string, s2: string, n: int) -> int
```

**Safety Notes:**
- nanolang guarantees all strings are null-terminated
- These functions don't modify memory
- `strncmp` is safer than `strcmp` for bounded comparisons

See `examples/23_extern_string.nano` for comprehensive examples.

---

## Functions NOT to Expose

### ❌ Unsafe String Manipulation

**Never expose these:**

```c
// NO - No bounds checking
strcpy(char* dest, const char* src);
strcat(char* dest, const char* src);
sprintf(char* str, const char* format, ...);
gets(char* str);  // Deprecated for a reason!
```

**Why?** These functions can cause buffer overflows because they don't take buffer size parameters.

**Alternative:** If you need string manipulation, implement safe wrappers in the nanolang runtime:

```c
// src/runtime/string_safe.c
const char* nl_safe_strdup(const char* s) {
    if (!s) return "";
    size_t len = strlen(s);
    char* copy = malloc(len + 1);
    if (!copy) return "";
    memcpy(copy, s, len + 1);
    return copy;
}
```

### ❌ Raw Memory Operations

**Be very careful with:**

```c
malloc(size_t size);
free(void* ptr);
memcpy(void* dest, const void* src, size_t n);
```

**Why?** These require manual memory management and can easily cause:
- Memory leaks
- Double-free errors
- Use-after-free vulnerabilities
- Buffer overflows

**Alternative:** Use nanolang's built-in data structures (lists, arrays, strings).

---

## How It Works

### Compilation Process

1. **Parsing:** `extern fn` declarations are recognized by the parser
2. **Type Checking:** Extern functions are registered but not type-checked (no body)
3. **Transpilation:** Generates C forward declarations:
   ```c
   extern double sqrt(double);
   ```
4. **Linking:** The C compiler links against the standard library (`-lm` for math)

### Code Generation

**nanolang:**
```nano
extern fn sqrt(x: float) -> float

fn main() -> int {
    let result: float = (sqrt 16.0)
    (println result)
    return 0
}
```

**Generated C:**
```c
/* External declarations */
extern double sqrt(double);

/* Function implementations */
int main() {
    double result = sqrt(16.0);
    nl_println_float(result);
    return 0;
}
```

**Note:** Extern functions are called directly with their original names (no `nl_` prefix).

---

## Best Practices

### 1. Declare Only What You Need

Don't declare all possible C functions. Only declare the ones you actually use.

```nano
# Good - specific declarations
extern fn sqrt(x: float) -> float
extern fn pow(x: float, y: float) -> float

# Bad - unused declarations
extern fn sqrt(x: float) -> float
extern fn sin(x: float) -> float
extern fn cos(x: float) -> float
extern fn tan(x: float) -> float
# ... 50 more functions you never use
```

### 2. Group Related Declarations

Organize extern declarations by category:

```nano
# Math operations
extern fn sqrt(x: float) -> float
extern fn pow(x: float, y: float) -> float
extern fn abs(x: int) -> int

# String operations
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int

# Character classification
extern fn isdigit(c: int) -> int
extern fn toupper(c: int) -> int
```

### 3. Document Preconditions

Add comments explaining any preconditions:

```nano
# sqrt: Returns square root of x
# Precondition: x >= 0 (negative values produce NaN)
extern fn sqrt(x: float) -> float

# log: Returns natural logarithm of x
# Precondition: x > 0 (zero/negative produce NaN or -inf)
extern fn log(x: float) -> float
```

### 4. Check Return Values

For functions that can fail, check return values:

```nano
extern fn strcmp(s1: string, s2: string) -> int

fn strings_equal(s1: string, s2: string) -> bool {
    let result: int = (strcmp s1 s2)
    return (== result 0)
}
```

### 5. Use Bounded Functions When Available

Prefer bounded functions over unbounded:

```nano
# Prefer this:
extern fn strncmp(s1: string, s2: string, n: int) -> int

# Over this:
extern fn strcmp(s1: string, s2: string) -> int
```

---

## Limitations

### Current Limitations

1. **No Structs:** Cannot pass nanolang structs to C functions (yet)
2. **No Arrays:** Cannot pass nanolang arrays to C functions directly
3. **No Pointers:** No explicit pointer manipulation
4. **No Callbacks:** Cannot pass nanolang functions to C functions
5. **No Variadic Functions:** Cannot declare functions like `printf(...)`

### Workarounds

**For Arrays:**
- Use individual elements
- Create wrapper functions that convert arrays

**For Structs:**
- Pass individual fields
- Create C wrappers that construct structs

**For Output Parameters:**
- Use return values instead
- Create wrapper functions in the nanolang runtime

---

## Complete List of Recommended Safe Functions

### Math Functions (31 functions)

**Trigonometric:**
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`, `atan2`
- `sinh`, `cosh`, `tanh`
- `asinh`, `acosh`, `atanh`

**Exponential/Logarithmic:**
- `exp`, `exp2`, `expm1`
- `log`, `log10`, `log2`, `log1p`

**Power:**
- `pow`, `sqrt`, `cbrt`, `hypot`

**Rounding:**
- `ceil`, `floor`, `round`, `trunc`
- `rint`, `nearbyint`

**Other:**
- `fabs`, `fmod`, `remainder`, `copysign`

### Character Functions (14 functions)

**Classification:**
- `isalpha`, `isdigit`, `isalnum`, `isspace`
- `isupper`, `islower`, `isprint`, `ispunct`
- `iscntrl`, `isxdigit`, `isblank`, `isgraph`

**Conversion:**
- `toupper`, `tolower`

### String Functions (4 functions)

**Read-Only:**
- `strlen`, `strcmp`, `strncmp`, `strchr`

**Total: 49 safe functions ready to use!**

---

## Examples

See the following files for comprehensive examples:

- **`examples/21_extern_math.nano`** - Math library functions (31 functions)
- **`examples/22_extern_char.nano`** - Character classification (14 functions)
- **`examples/23_extern_string.nano`** - String operations (4 functions)

---

## Implementation Details

### Parser Changes

- Added `TOKEN_EXTERN` keyword
- `parse_function()` now accepts `is_extern` parameter
- Extern functions have no body (skipped during parsing)

### Type Checker Changes

- Extern functions skip body type checking
- Extern functions don't require shadow tests
- Function signature is still validated

### Transpiler Changes

- Generates `extern` declarations in C code
- Calls extern functions without `nl_` prefix
- Links directly to C standard library

### Files Modified

- `src/nanolang.h` - Added `is_extern` fields to ASTNode and Function
- `src/lexer.c` - Recognize `extern` keyword
- `src/parser.c` - Parse `extern fn` declarations
- `src/typechecker.c` - Skip extern function body checks
- `src/transpiler.c` - Generate extern declarations and calls

---

## Future Enhancements

Potential future improvements:

1. **Custom Headers:** `extern "mylib.h" fn ...`
2. **Link Libraries:** `extern "libmylib.so" fn ...`
3. **Struct Support:** Pass structs to C functions
4. **Callback Support:** Pass nanolang functions to C
5. **Generic Pointers:** `extern fn func(ptr: *void) -> int`
6. **Error Handling:** `extern fn func() -> Result<int, Error>`

---

## Security Checklist

Before adding any extern function:

- [ ] ✅ Takes explicit length parameters for buffers?
- [ ] ✅ Returns bounded results?
- [ ] ✅ No pointer arithmetic required?
- [ ] ✅ No complex types (structs, unions)?
- [ ] ✅ Safe with nanolang's type system?
- [ ] ✅ Can't cause buffer overflow?
- [ ] ✅ Can't cause use-after-free?
- [ ] ✅ Clear error handling?

**If any answer is NO, don't expose it directly!**

Create a safe wrapper in the nanolang runtime instead.

---

## Hybrid C/nanolang Applications

nanolang is designed to work seamlessly with C libraries. This enables **hybrid applications** where:

- **Core logic** is written in nanolang (type-safe, shadow-tested)
- **System integration** uses C libraries (SDL2, OpenGL, system APIs)
- **No wrappers needed** - call C functions directly

### Example: SDL2 Application

```nano
# Declare SDL2 extern functions
extern fn SDL_Init(flags: int) -> int
extern fn SDL_CreateWindow(title: string, x: int, y: int, w: int, h: int, flags: int) -> int
extern fn SDL_Quit() -> void

fn main() -> int {
    let result: int = (SDL_Init 32)
    if (< result 0) {
        return 1
    } else {
        let window: int = (SDL_CreateWindow "My App" 0 0 640 480 4)
        if (== window 0) {
            (SDL_Quit)
            return 1
        } else {
            # ... application logic ...
            (SDL_Quit)
            return 0
        }
    }
}
```

**Compilation:**
```bash
./bin/nanoc app.nano -o app \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

See [Building Hybrid Apps](BUILDING_HYBRID_APPS.md) for complete details.

---

## Conclusion

**Extern functions are a first-class feature** that enables nanolang to seamlessly integrate with the C ecosystem:

✅ **Cannot be shadowed** - Immutable declarations  
✅ **Cannot have shadow tests** - C functions, not nanolang  
✅ **Fully type-checked** - Type safety at compile time  
✅ **No wrappers needed** - Direct C library integration  
✅ **Hybrid applications** - Mix nanolang and C libraries  

This design allows nanolang to work cooperatively with C without attempting to wrap everything. Use any C library directly from nanolang code.

**Remember:** When in doubt, create a safe wrapper!


---

## FFI Safety Guidelines

### Safety Promise

**CRITICAL:** `extern` functions in NanoLang are **trusted** by the compiler. The compiler assumes they are memory-safe and correct. **YOU** are responsible for ensuring safety.

### The Trust Boundary

```nano
# This is a TRUST BOUNDARY
extern fn strcpy(dest: string, src: string) -> string  # ⚠️ UNSAFE!

# The compiler trusts you that strcpy:
# - Won't overflow dest buffer
# - Won't access invalid memory
# - Won't cause undefined behavior

# If you're wrong, your program will crash or have security vulnerabilities!
```

---

## Safety Checklist

Before declaring any `extern fn`, verify ALL of these:

### Buffer Overflows ✅

- [ ] Function does NOT take unchecked string pointers
- [ ] All buffer operations have explicit length parameters
- [ ] Uses `strncpy` not `strcpy`, `snprintf` not `sprintf`
- [ ] Array parameters include length/bounds information

### Memory Safety ✅

- [ ] Function does NOT return pointers to stack memory
- [ ] All allocated memory has clear ownership
- [ ] No dangling pointers after function returns
- [ ] No double-free bugs

### Type Safety ✅

- [ ] No unsafe casts between incompatible types
- [ ] Pointer types match exactly
- [ ] Integer sizes match (int64_t, not int)
- [ ] No pointer arithmetic exposed

### Input Validation ✅

- [ ] Function validates all inputs
- [ ] Returns errors instead of crashing
- [ ] Handles edge cases (NULL, 0, negative values)
- [ ] Documents preconditions clearly

---

## Unsafe C Functions - DO NOT USE

### String Functions ❌

| Unsafe Function | Why Unsafe | Safe Alternative |
|----------------|------------|------------------|
| `strcpy(dest, src)` | No bounds checking | `strncpy(dest, src, n)` |
| `strcat(dest, src)` | No bounds checking | `strncat(dest, src, n)` |
| `sprintf(buf, fmt, ...)` | No bounds checking | `snprintf(buf, n, fmt, ...)` |
| `gets(buf)` | No bounds checking | `fgets(buf, n, stdin)` |
| `scanf("%s", buf)` | No bounds checking | `scanf("%100s", buf)` |

### Memory Functions ❌

| Unsafe Function | Why Unsafe | Safe Alternative |
|----------------|------------|------------------|
| `memcpy(dst, src, n)` | No overlap check | `memmove(dst, src, n)` |
| `alloca(n)` | Stack overflow | `malloc(n)` + `free()` |

**Example of unsafe function:**
```nano
# ❌ NEVER DO THIS
extern fn strcpy(dest: string, src: string) -> string

fn copy_string(source: string) -> string {
    let buffer: string = (c_malloc 10)  # Only 10 bytes!
    (strcpy buffer source)  # Buffer overflow if source > 9 chars!
    return buffer
}
```

---

## Safe FFI Patterns

### Pattern 1: Length-Bounded Operations

✅ **Safe:**
```nano
# C function signature:
# char* strncpy(char* dest, const char* src, size_t n);

extern fn strncpy(dest: string, src: string, n: int) -> string

fn safe_copy(src: string) -> string {
    let len: int = (str_length src)
    let buffer: string = (c_malloc (+ len 1))  # +1 for null terminator
    (strncpy buffer src len)
    return buffer
}
```

### Pattern 2: Error Return Values

✅ **Safe:**
```nano
# C returns -1 on error
extern fn safe_operation(x: int) -> int

fn checked_operation(x: int) -> Result<int, string> {
    let result: int = (safe_operation x)
    if (< result 0) {
        return Result.Err { error: "Operation failed" }
    }
    return Result.Ok { value: result }
}
```

### Pattern 3: Opaque Pointers

✅ **Safe:**
```nano
# Use opaque type for C pointers
extern fn fopen(path: string, mode: string) -> opaque
extern fn fclose(file: opaque) -> int

fn safe_file_operation(path: string) -> bool {
    let file: opaque = (fopen path "r")
    if (== file (null_opaque)) {
        return false
    }
    
    # Use file...
    
    (fclose file)
    return true
}
```

---

## Common Vulnerabilities

### 1. Buffer Overflow

**Vulnerable:**
```nano
extern fn unsafe_read(buffer: string, size: int) -> int

fn read_data() -> string {
    let buf: string = (c_malloc 100)
    (unsafe_read buf 200)  # ❌ Reads 200 bytes into 100-byte buffer!
    return buf
}
```

**Fixed:**
```nano
extern fn safe_read(buffer: string, size: int, max_size: int) -> int

fn read_data() -> string {
    let buf: string = (c_malloc 100)
    (safe_read buf 0 100)  # ✅ Respects buffer size
    return buf
}
```

### 2. Use-After-Free

**Vulnerable:**
```nano
extern fn get_temp_buffer() -> string

fn process() -> string {
    let buf: string = (get_temp_buffer)
    # C function may free buf after returning!
    return buf  # ❌ Use-after-free!
}
```

**Fixed:**
```nano
extern fn get_temp_buffer() -> string

fn process() -> string {
    let buf: string = (get_temp_buffer)
    let copy: string = (str_concat "" buf)  # ✅ Make a copy
    return copy
}
```

### 3. NULL Pointer Dereference

**Vulnerable:**
```nano
extern fn may_return_null(x: int) -> string

fn use_result(x: int) -> int {
    let s: string = (may_return_null x)
    return (str_length s)  # ❌ Crash if s is NULL!
}
```

**Fixed:**
```nano
extern fn may_return_null(x: int) -> opaque

fn use_result(x: int) -> int {
    let s: opaque = (may_return_null x)
    if (== s (null_opaque)) {
        return 0  # ✅ Check for NULL
    }
    let str: string = (opaque_to_string s)
    return (str_length str)
}
```

---

## Auditing FFI Bindings

### Manual Audit Checklist

For each `extern fn` declaration:

1. **Find C documentation** - Read man pages or header files
2. **Check return values** - What does -1, NULL, 0 mean?
3. **Check error conditions** - How are errors reported?
4. **Check buffer requirements** - Are lengths specified?
5. **Check thread safety** - Can it be called from multiple threads?
6. **Check side effects** - Does it modify global state?

### Automated Scanning

Create a script to detect unsafe patterns:

```bash
#!/bin/bash
# scripts/audit_ffi_safety.sh

echo "Scanning for unsafe FFI patterns..."

# Check for known unsafe functions
grep -rn "extern fn strcpy\|extern fn strcat\|extern fn sprintf\|extern fn gets" . --include="*.nano"

# Check for missing length parameters
grep -rn "extern fn.*string.*string" . --include="*.nano" | grep -v "length\|size\|n"

echo "Audit complete."
```

---

## Testing FFI Bindings

### Test for Buffer Overflows

```nano
# Test with deliberately long inputs
extern fn my_ffi_func(buf: string, len: int) -> int

shadow my_ffi_func {
    # Test with maximum safe size
    let buf: string = (c_malloc 1000)
    assert (>= (my_ffi_func buf 1000) 0)
    
    # Test with empty string
    assert (>= (my_ffi_func "" 0) 0)
}
```

### Test for NULL Handling

```nano
extern fn process_string(s: string) -> int

shadow process_string {
    # Test with NULL (if FFI accepts it)
    assert (>= (process_string (null_string)) 0)
    
    # Test with empty string
    assert (>= (process_string "") 0)
}
```

---

## Documentation Requirements

Every `extern fn` should be documented:

```nano
# Opens a file for reading.
#
# Safety:
# - Returns NULL on failure (check with null_opaque)
# - Caller must call fclose() on success
# - Path must be valid UTF-8
# - Mode must be "r", "w", "a", "rb", "wb", or "ab"
#
# Example:
#   let f: opaque = (fopen "data.txt" "r")
#   if (!= f (null_opaque)) {
#       # Use file
#       (fclose f)
#   }
extern fn fopen(path: string, mode: string) -> opaque
```

---

## Unsafe Block (Future Feature)

**Planned for v1.0:**

```nano
# Explicitly mark unsafe FFI code
unsafe {
    extern fn dangerous_c_function(ptr: opaque) -> int
    
    fn wrapper(ptr: opaque) -> int {
        # All code in unsafe block is marked as requiring careful review
        return (dangerous_c_function ptr)
    }
}

# Safe code outside unsafe block
fn safe_function() -> int {
    return 42
}
```

**Benefits:**
- Makes trust boundaries explicit
- Easy to audit (grep for `unsafe {`)
- Requires code review for unsafe sections

---

## Security Best Practices

1. **Minimize FFI surface** - Use NanoLang stdlib when possible
2. **Wrapper functions** - Wrap unsafe C in safe NanoLang APIs
3. **Input validation** - Validate ALL inputs before passing to C
4. **Error handling** - Check ALL return values
5. **Documentation** - Document safety requirements
6. **Testing** - Test with edge cases and fuzzing
7. **Code review** - Have FFI code reviewed by another developer
8. **Regular audits** - Re-audit FFI bindings periodically

---

## Advanced Safety Topics

### Memory Ownership Patterns

FFI introduces complex ownership semantics. Understanding who owns memory is critical.

#### Ownership Model 1: Caller-Owned (NanoLang Manages)

✅ **Pattern: Pass buffer to C, NanoLang owns it**

```nano
fn safe_file_read(path: string) -> string {
    # NanoLang allocates and owns buffer
    let buffer: string = (string_new 4096)

    # C function fills buffer (read-only operation on path)
    extern fn read_file_into_buffer(path: string, buf: string, size: int) -> int
    let bytes_read: int = (read_file_into_buffer path buffer 4096)

    if (< bytes_read 0) {
        return ""  # Error
    }

    # NanoLang still owns buffer, automatic cleanup
    return buffer
}
```

**Safety guarantees:**
- NanoLang GC handles deallocation
- No dangling pointers
- Buffer size known at NanoLang level

#### Ownership Model 2: Callee-Owned (C Allocates)

⚠️ **Pattern: C allocates, NanoLang must free**

```nano
# C function: char* strdup(const char* s);  // Allocates new string
extern fn c_strdup(s: string) -> string
extern fn c_free(ptr: string) -> void

fn duplicate_string(source: string) -> string {
    let copy: string = (c_strdup source)

    # ⚠️ CRITICAL: Must manually free later!
    # Store copy somewhere and ensure c_free is called

    return copy  # Caller now responsible for freeing!
}

# Better: RAII wrapper
fn safe_duplicate(source: string) -> string {
    let copy: string = (c_strdup source)
    let result: string = (string_copy copy)  # Copy to GC-managed memory
    (c_free copy)  # Immediately free C-allocated memory
    return result  # Return GC-managed copy
}
```

**Safety rules:**
- Document ownership in comments
- Use RAII wrappers to minimize manual memory management
- Never return C-owned pointers directly to users

#### Ownership Model 3: Borrowed (No Transfer)

✅ **Pattern: C function borrows, doesn't own**

```nano
# C function: size_t strlen(const char* s);  // Only reads, doesn't own
extern fn strlen(s: string) -> int

fn get_length(text: string) -> int {
    # strlen borrows 'text', doesn't take ownership
    return (strlen text)
    # text still valid after function returns
}
```

**Safety guarantees:**
- No ownership transfer
- Original owner still responsible for cleanup
- Read-only access

---

### RAII (Resource Acquisition Is Initialization) Patterns

RAII ensures resources are cleaned up even on error paths.

#### Pattern: File Handle Wrapper

```nano
# C file operations
extern fn fopen(path: string, mode: string) -> int  # Returns FILE* as int
extern fn fclose(file: int) -> int
extern fn fread(ptr: string, size: int, count: int, file: int) -> int

struct File {
    handle: int
    is_open: bool
}

fn file_open(path: string, mode: string) -> File {
    let handle: int = (fopen path mode)
    return File { handle: handle, is_open: (!= handle 0) }
}

fn file_close(f: mut File) -> void {
    if f.is_open {
        (fclose f.handle)
        set f.is_open false
        set f.handle 0
    }
}

fn safe_read_file(path: string) -> string {
    let f: File = (file_open path "r")
    if (not f.is_open) {
        return ""  # Error: couldn't open
    }

    let buffer: string = (string_new 4096)
    let bytes: int = (fread buffer 1 4096 f.handle)

    # RAII: Always close, even on error
    (file_close f)

    if (< bytes 0) {
        return ""  # Error: read failed
    }

    return buffer
}
```

**Benefits:**
- Guaranteed cleanup
- No resource leaks
- Clear ownership semantics

---

### Thread Safety Considerations

FFI introduces thread safety concerns not present in pure NanoLang.

#### Rule 1: Non-Reentrant C Functions

Many C standard library functions are NOT thread-safe:

```nano
# ❌ UNSAFE in multi-threaded context
extern fn strtok(str: string, delim: string) -> string  # Uses global state!

# ✅ SAFE: Thread-safe alternative
extern fn strtok_r(str: string, delim: string, saveptr: int) -> string
```

#### Rule 2: Global State

```nano
# ❌ UNSAFE: C library with global state
extern fn set_error_callback(callback: fn(string) -> void) -> void

# Problem: Only one callback globally!
# Multiple NanoLang components can't each have their own callback
```

**Solutions:**
- Use thread-local storage when available
- Add locking around non-reentrant functions
- Document thread-safety requirements clearly

#### Rule 3: Callbacks from C to NanoLang

```nano
# C library calls this function
fn my_callback(data: string) -> void {
    # ⚠️ Called from C thread!
    # Avoid accessing NanoLang GC-managed objects
    # Be careful with global state
    (println data)  # Simple operations OK
}

extern fn register_callback(cb: fn(string) -> void) -> void
```

**Thread safety rules:**
- Callbacks run in C's thread context
- Minimize shared state access
- Use message passing instead of shared memory

---

### Error Handling Across FFI Boundary

C and NanoLang have different error handling conventions.

#### Pattern 1: Return Code Translation

```nano
# C convention: 0 = success, non-zero = error
extern fn c_operation(arg: string) -> int

fn safe_operation(arg: string) -> Result<void, string> {
    let status: int = (c_operation arg)

    if (== status 0) {
        return Ok(void)
    } else {
        # Translate C error code to message
        let msg: string = (error_code_to_string status)
        return Err(msg)
    }
}

fn error_code_to_string(code: int) -> string {
    if (== code 1) {
        return "Invalid argument"
    } else {
        if (== code 2) {
            return "File not found"
        } else {
            return "Unknown error"
        }
    }
}
```

#### Pattern 2: errno Handling

```nano
extern fn c_errno() -> int  # Get errno value
extern fn c_strerror(errnum: int) -> string  # Error message

fn safe_file_operation(path: string) -> Result<void, string> {
    # C function sets errno on error
    extern fn remove_file(path: string) -> int

    let result: int = (remove_file path)

    if (!= result 0) {
        let err_num: int = (c_errno)
        let err_msg: string = (c_strerror err_num)
        return Err(err_msg)
    }

    return Ok(void)
}
```

#### Pattern 3: NULL Pointer Returns

```nano
# C function: FILE* fopen(const char* path, const char* mode);
# Returns NULL on error
extern fn fopen(path: string, mode: string) -> int

fn safe_open_file(path: string) -> Result<int, string> {
    let handle: int = (fopen path "r")

    if (== handle 0) {  # NULL pointer
        return Err("Failed to open file")
    }

    return Ok(handle)
}
```

---

### Input Validation and Sanitization

Always validate inputs before passing to C functions.

#### String Validation

```nano
fn safe_path_operation(user_path: string) -> Result<void, string> {
    # Validate: No null bytes (C string terminator)
    if (str_contains user_path "\0") {
        return Err("Path contains null byte")
    }

    # Validate: Length check
    if (> (str_length user_path) 1024) {
        return Err("Path too long")
    }

    # Validate: No directory traversal
    if (str_contains user_path "..") {
        return Err("Path traversal not allowed")
    }

    # Safe to pass to C
    extern fn c_open_file(path: string) -> int
    let handle: int = (c_open_file user_path)

    if (== handle 0) {
        return Err("Failed to open file")
    }

    return Ok(void)
}
```

#### Integer Validation

```nano
fn safe_buffer_operation(size: int) -> Result<string, string> {
    # Validate: Positive size
    if (<= size 0) {
        return Err("Size must be positive")
    }

    # Validate: Reasonable limit (prevent allocation attacks)
    if (> size 10485760) {  # 10MB max
        return Err("Size too large")
    }

    # Safe to allocate
    extern fn c_malloc(size: int) -> string
    let buffer: string = (c_malloc size)

    if (== buffer "") {
        return Err("Allocation failed")
    }

    return Ok(buffer)
}
```

#### Array Bounds

```nano
fn safe_array_access(arr: array<int>, index: int) -> Result<int, string> {
    # Validate: Index in bounds
    let len: int = (array_length arr)
    if (or (< index 0) (>= index len)) {
        return Err("Index out of bounds")
    }

    # Safe to access
    return Ok((at arr index))
}
```

---

### Security: Preventing Injection Attacks

FFI opens vectors for security vulnerabilities.

#### Command Injection

```nano
# ❌ DANGEROUS: Direct command execution
fn unsafe_run_command(user_input: string) -> int {
    extern fn system(cmd: string) -> int

    # VULNERABILITY: user_input = "ls; rm -rf /"
    let cmd: string = (str_concat "ls " user_input)
    return (system cmd)
}

# ✅ SAFE: Whitelist or escape
fn safe_run_command(filename: string) -> Result<int, string> {
    # Validate: Only alphanumeric and underscore
    if (not (is_safe_filename filename)) {
        return Err("Invalid filename")
    }

    extern fn system(cmd: string) -> int
    let cmd: string = (str_concat "ls " filename)
    return Ok((system cmd))
}

fn is_safe_filename(s: string) -> bool {
    let len: int = (str_length s)
    let mut i: int = 0
    while (< i len) {
        let c: int = (char_at s i)
        if (not (or (is_alnum c) (== c 95))) {  # 95 = '_'
            return false
        }
        set i (+ i 1)
    }
    return true
}
```

#### SQL Injection (If Using SQL FFI)

```nano
# ❌ DANGEROUS: String concatenation
fn unsafe_query(user_input: string) -> string {
    extern fn sql_query(query: string) -> string

    # VULNERABILITY: user_input = "'; DROP TABLE users; --"
    let query: string = (str_concat "SELECT * FROM users WHERE name = '"
                                     (str_concat user_input "'"))
    return (sql_query query)
}

# ✅ SAFE: Prepared statements (requires proper SQL FFI wrapper)
fn safe_query(user_input: string) -> string {
    extern fn sql_prepare(query: string) -> int
    extern fn sql_bind_string(stmt: int, index: int, value: string) -> void
    extern fn sql_execute(stmt: int) -> string
    extern fn sql_finalize(stmt: int) -> void

    let stmt: int = (sql_prepare "SELECT * FROM users WHERE name = ?")
    (sql_bind_string stmt 0 user_input)  # Safely bind parameter
    let result: string = (sql_execute stmt)
    (sql_finalize stmt)

    return result
}
```

---

### Performance Considerations

FFI calls have overhead. Optimize hot paths.

#### Batching Operations

```nano
# ❌ SLOW: Call FFI in tight loop
fn process_slow(items: array<string>) -> void {
    extern fn c_process(item: string) -> void

    for item in items {
        (c_process item)  # FFI call per iteration!
    }
}

# ✅ FAST: Batch operations
fn process_fast(items: array<string>) -> void {
    extern fn c_process_batch(items: array<string>, count: int) -> void

    # Single FFI call for all items
    (c_process_batch items (array_length items))
}
```

#### Caching Results

```nano
# Cache expensive FFI calls
fn get_system_info() -> string {
    static mut cached: string = ""
    static mut initialized: bool = false

    if (not initialized) {
        extern fn c_get_system_info() -> string
        set cached (c_get_system_info)
        set initialized true
    }

    return cached
}
```

---

## Related Documentation

- [MODULE_SYSTEM.md](MODULE_SYSTEM.md) - Module creation with C sources
- [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) - Debugging FFI issues
- [SPECIFICATION.md](SPECIFICATION.md) - extern fn specification
- [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) - Memory model

---

**Last Updated:** January 25, 2026
**FFI Safety:** CRITICAL - Read and understand before using extern
**Version:** 0.2.0+

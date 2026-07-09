# External C Function Interface (FFI)

I support calling external C functions through the `extern` keyword. This is a first-class feature that allows me to integrate with the C ecosystem. I do not attempt to wrap everything. I allow you to use what exists.

## Overview

I treat extern functions as first-class citizens:
- I do not allow them to be shadowed or redefined. They are immutable declarations.
- I do not allow shadow tests for them. They are C functions, and I cannot test what I cannot see.
- I type-check them at compile time. I enforce safety even at my borders.
- I integrate them seamlessly. You can use C libraries directly without wrappers.

This design allows me to work with the C ecosystem. You can use any C library, such as SDL2, OpenGL, or system libraries, directly from my code.

---

## Syntax

### Declaring External Functions

```nano
extern fn function_name(param1: type1, param2: type2) -> return_type
```

**Key Points:**
- Use the `extern` keyword before `fn`.
- Provide the exact C function signature matching the C library.
- No function body is permitted. Provide only the declaration.
- I do not require or allow shadow tests for these functions.
- I do not allow them to be redefined. They are immutable.

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

### 1. I Do Not Allow Shadowing Extern Functions

Extern functions are immutable declarations. I do not let you redefine or shadow them with regular functions.

```nano
extern fn SDL_Init(flags: int) -> int

# ERROR: Cannot shadow extern function
fn SDL_Init(flags: int) -> int {
    return 0
}
```

**My Error Message:**
```
Error: Function 'SDL_Init' cannot shadow extern function
  Extern functions are first-class and cannot be shadowed
  Choose a different function name
```

### 2. I Do Not Allow Shadow Tests for Extern Functions

Extern functions are C functions. I cannot run them in my interpreter, so I do not allow shadow tests for them.

```nano
extern fn SDL_Init(flags: int) -> int

# ERROR: Shadow test cannot be attached to extern function
shadow SDL_Init {
    assert (== (SDL_Init 0) 0)
}
```

**My Error Message:**
```
Error: Shadow test cannot be attached to extern function 'SDL_Init'
  Extern functions are C functions and cannot be tested in the interpreter
  Remove the shadow test or test a wrapper function instead
```

### 3. I Skip Shadow Tests for Functions Using Externs

If a regular function of mine calls an extern function, I make the shadow test optional. I skip it during execution because I cannot verify the behavior of the external world.

```nano
extern fn SDL_Init(flags: int) -> int

fn main() -> int {
    let result: int = (SDL_Init 32)
    return result
}

# Shadow test is optional - I will skip it at runtime
shadow main {
    assert (== (main) 0)
}
```

### 4. I Type-Check Every Extern Call

I fully type-check all extern function calls at compile time. I do not let you wander into type mismatches.

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

### My Types to C Types

| My Type | C Type | Notes |
|---------|--------|-------|
| `int` | `int64_t` | 64-bit signed integer |
| `float` | `double` | Double-precision floating point |
| `bool` | `bool` | C99 boolean (0 or 1) |
| `string` | `const char*` | Null-terminated C string |
| `void` | `void` | No return value |

**Important:**
- My `int` is always 64-bit. C `int` is platform-dependent.
- My `float` is always double precision.
- My strings are immutable. I pass them as `const char*` to C.
- I represent pointer types like `SDL_Window*` as `int`. My transpiler casts them appropriately.

---

## Safe Function Categories

### Category 1: Math Functions

I consider C math library functions safe. They operate on scalar values, have no side effects beyond errno, and cannot cause memory issues.

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

I provide comprehensive examples in `examples/21_extern_math.nano`.

### Category 2: Character Classification

These functions are safe because they operate on single integers.

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

I provide comprehensive examples in `examples/22_extern_char.nano`.

### Category 3: Read-Only String Functions

These are safe if the strings are null-terminated. I guarantee that all my strings are null-terminated.

```nano
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int
extern fn strncmp(s1: string, s2: string, n: int) -> int
```

**My Safety Notes:**
- I guarantee all my strings are null-terminated.
- These functions do not modify memory.
- I recommend `strncmp` over `strcmp` for bounded comparisons.

I provide comprehensive examples in `examples/23_extern_string.nano`.

---

## Functions I Refuse to Support Directly

### Unsafe String Manipulation

I do not want you to use these functions. They are dangerous.

```c
// NO - No bounds checking
strcpy(char* dest, const char* src);
strcat(char* dest, const char* src);
sprintf(char* str, const char* format, ...);
gets(char* str);
```

These functions cause buffer overflows because they do not respect buffer sizes. If you need string manipulation, use my runtime or implement safe wrappers.

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

### Raw Memory Operations

I am cautious about these:

```c
malloc(size_t size);
free(void* ptr);
memcpy(void* dest, const void* src, size_t n);
```

They require manual memory management and cause leaks, double-free errors, use-after-free vulnerabilities, and buffer overflows. Use my built-in data structures instead. I handle the memory so you do not have to.

---

## How I Work

### My Compilation Process

1. **Parsing:** I recognize `extern fn` declarations.
2. **Type Checking:** I register these functions but do not check for a body.
3. **Transpilation:** I generate C forward declarations like `extern double sqrt(double);`.
4. **Linking:** The C compiler links against the standard library.

### My Code Generation

**In my syntax:**
```nano
extern fn sqrt(x: float) -> float

fn main() -> int {
    let result: float = (sqrt 16.0)
    (println result)
    return 0
}
```

**The generated C:**
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

I call extern functions directly with their original names. I do not add my `nl_` prefix to them.

---

## My Best Practices

### 1. Declare Only What You Use

I recommend you only declare the C functions you actually need.

```nano
# Good - specific declarations
extern fn sqrt(x: float) -> float
extern fn pow(x: float, y: float) -> float

# Bad - unused declarations
extern fn sqrt(x: float) -> float
extern fn sin(x: float) -> float
extern fn cos(x: float) -> float
extern fn tan(x: float) -> float
```

### 2. Group Your Declarations

I find it helpful to organize extern declarations by category.

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

### 3. Document Your Preconditions

I suggest you add comments explaining what the C function expects.

```nano
# sqrt: Returns square root of x
# Precondition: x >= 0 (negative values produce NaN)
extern fn sqrt(x: float) -> float

# log: Returns natural logarithm of x
# Precondition: x > 0 (zero/negative produce NaN or -inf)
extern fn log(x: float) -> float
```

### 4. Check Every Return Value

For functions that can fail, I expect you to check the return values.

```nano
extern fn strcmp(s1: string, s2: string) -> int

fn strings_equal(s1: string, s2: string) -> bool {
    let result: int = (strcmp s1 s2)
    return (== result 0)
}
```

### 5. Use Bounded Functions

I prefer bounded functions over unbounded ones.

```nano
# I prefer this:
extern fn strncmp(s1: string, s2: string, n: int) -> int

# I do not recommend this:
extern fn strcmp(s1: string, s2: string) -> int
```

---

## My Limitations

### What I Cannot Do Yet

1. I cannot pass my structs to C functions yet.
2. I cannot pass my arrays to C functions directly.
3. I do not allow explicit pointer manipulation.
4. I do not support passing my functions to C as callbacks.
5. I do not support variadic functions like `printf(...)`.

### How to Work Around Them

**For Arrays:**
Use individual elements or create wrapper functions that convert my arrays.

**For Structs:**
Pass individual fields or create C wrappers that construct the structs.

**For Output Parameters:**
Use return values or create wrapper functions in my runtime.

---

## Safe Functions I Recommend

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

I have 49 safe functions ready for you to use.

---

## Examples

I provide comprehensive examples in these files:

- `examples/21_extern_math.nano`
- `examples/22_extern_char.nano`
- `examples/23_extern_string.nano`

---

## My Implementation Details

### Changes to My Parser

- I added the `TOKEN_EXTERN` keyword.
- My `parse_function()` accepts an `is_extern` parameter.
- I skip the body of extern functions during parsing.

### Changes to My Type Checker

- I skip body type checking for extern functions.
- I do not require shadow tests for them.
- I still validate the function signature.

### Changes to My Transpiler

- I generate `extern` declarations in the C code.
- I call extern functions without my `nl_` prefix.
- I link directly to the C standard library.

### Files I Modified

- `src/nanolang.h` - I added `is_extern` fields to ASTNode and Function.
- `src/lexer.c` - I recognize the `extern` keyword.
- `src/parser.c` - I parse `extern fn` declarations.
- `src/typechecker.c` - I skip body checks for extern functions.
- `src/transpiler.c` - I generate the declarations and calls.

---

## Enhancements I Plan to Add

I am considering these improvements:

1. Custom Headers: `extern "mylib.h" fn ...`
2. Linking Libraries: `extern "libmylib.so" fn ...`
3. Struct Support: Passing my structs to C functions.
4. Callback Support: Passing my functions to C.
5. Generic Pointers: `extern fn func(ptr: *void) -> int`
6. Error Handling: `extern fn func() -> Result<int, Error>`

---

## My Security Checklist

Before you add an extern function, answer these:

- Does it take explicit length parameters for buffers?
- Does it return bounded results?
- Does it avoid pointer arithmetic?
- Does it avoid complex types like structs or unions?
- Is it safe with my type system?
- Can it avoid causing a buffer overflow?
- Can it avoid causing a use-after-free error?
- Does it have clear error handling?

If you answer NO to any of these, do not expose the function directly. Create a safe wrapper in my runtime instead.

---

## Hybrid Applications

I am designed to work with C libraries. This allows you to build hybrid applications:

- You write your core logic in my syntax. It is type-safe and shadow-tested.
- You use C libraries like SDL2 or OpenGL for system integration.
- You do not need wrappers. You call the C functions directly.

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

**How to compile:**
```bash
./bin/nanoc app.nano -o app \
    -I/opt/homebrew/include/SDL2 \
    -L/opt/homebrew/lib \
    -lSDL2
```

I explain this further in `docs/BUILDING_HYBRID_APPS.md`.

---

## Conclusion

Extern functions allow me to integrate with the C ecosystem. I treat them as first-class features. I do not let you shadow them. I do not let you test them in my interpreter. I type-check every call. I do not require wrappers.

I work with C. I do not try to hide it. But I also do not let you be reckless. When you are in doubt, create a safe wrapper.

---

## My FFI Safety Guidelines

### My Safety Promise

I trust the `extern` functions you declare. I assume they are memory-safe and correct. This means you are responsible for my safety at this boundary.

### My Trust Boundary

```nano
# This is where I trust you
extern fn strcpy(dest: string, src: string) -> string  # DANGEROUS

# I trust you that strcpy:
# - Will not overflow the destination buffer
# - Will not access invalid memory
# - Will not cause undefined behavior

# If you are wrong, I will crash or become vulnerable.
```

---

## My Safety Checklist

Before you declare an `extern fn`, verify these:

### Buffer Overflows

- The function must not take unchecked string pointers.
- All buffer operations must have explicit length parameters.
- Use `strncpy` instead of `strcpy`, and `snprintf` instead of `sprintf`.
- Array parameters must include length and bounds information.

### Memory Safety

- The function must not return pointers to stack memory.
- All allocated memory must have clear ownership.
- There must be no dangling pointers after the function returns.
- There must be no double-free bugs.

### Type Safety

- Do not use unsafe casts between incompatible types.
- Pointer types must match exactly.
- Integer sizes must match. My `int` is `int64_t`.
- Do not expose pointer arithmetic.

### Input Validation

- The function must validate all inputs.
- It must return errors instead of crashing.
- It must handle edge cases like NULL, 0, or negative values.
- You must document preconditions clearly.

---

## C Functions I Forbid

### Unsafe String Functions

| Unsafe Function | Why it is Unsafe | My Safe Alternative |
|----------------|------------------|---------------------|
| `strcpy(dest, src)` | No bounds checking | `strncpy(dest, src, n)` |
| `strcat(dest, src)` | No bounds checking | `strncat(dest, src, n)` |
| `sprintf(buf, fmt, ...)` | No bounds checking | `snprintf(buf, n, fmt, ...)` |
| `gets(buf)` | No bounds checking | `fgets(buf, n, stdin)` |
| `scanf("%s", buf)` | No bounds checking | `scanf("%100s", buf)` |

**Example of what I do not allow:**
```nano
# I DO NOT WANT YOU TO DO THIS
extern fn strcpy(dest: string, src: string) -> string

fn copy_string(source: string) -> string {
    let buffer: string = (c_malloc 10)  # Only 10 bytes
    (strcpy buffer source)  # I will overflow if the source is too long
    return buffer
}
```

### Unsafe Memory Functions

| Unsafe Function | Why it is Unsafe | My Safe Alternative |
|----------------|------------------|---------------------|
| `memcpy(dst, src, n)` | No overlap check | `memmove(dst, src, n)` |
| `alloca(n)` | Stack overflow | `malloc(n)` and `free()` |

---

## Patterns I Consider Safe

### Pattern 1: Length-Bounded Operations

```nano
# C function signature:
# char* strncpy(char* dest, const char* src, size_t n);

extern fn strncpy(dest: string, src: string, n: int) -> string

fn safe_copy(src: string) -> string {
    let len: int = (str_length src)
    let buffer: string = (c_malloc (+ len 1))  # I add 1 for the null terminator
    (strncpy buffer src len)
    return buffer
}
```

### Pattern 2: Error Return Values

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

```nano
# I use an opaque type for C pointers
extern fn fopen(path: string, mode: string) -> opaque
extern fn fclose(file: opaque) -> int

fn safe_file_operation(path: string) -> bool {
    let file: opaque = (fopen path "r")
    if (== file (null_opaque)) {
        return false
    }
    
    # Use the file...
    
    (fclose file)
    return true
}
```

---

## Vulnerabilities I Want You to Avoid

### 1. Buffer Overflow

**Vulnerable Code:**
```nano
extern fn unsafe_read(buffer: string, size: int) -> int

fn read_data() -> string {
    let buf: string = (c_malloc 100)
    (unsafe_read buf 200)  # I am reading too much into the buffer
    return buf
}
```

**My Fixed Version:**
```nano
extern fn safe_read(buffer: string, size: int, max_size: int) -> int

fn read_data() -> string {
    let buf: string = (c_malloc 100)
    (safe_read buf 0 100)  # I respect the buffer size
    return buf
}
```

### 2. Use-After-Free

**Vulnerable Code:**
```nano
extern fn get_temp_buffer() -> string

fn process() -> string {
    let buf: string = (get_temp_buffer)
    # C might free the buffer here
    return buf  # I am using freed memory
}
```

**My Fixed Version:**
```nano
extern fn get_temp_buffer() -> string

fn process() -> string {
    let buf: string = (get_temp_buffer)
    let copy: string = (str_concat "" buf)  # I make my own copy
    return copy
}
```

### 3. NULL Pointer Dereference

**Vulnerable Code:**
```nano
extern fn may_return_null(x: int) -> string

fn use_result(x: int) -> int {
    let s: string = (may_return_null x)
    return (str_length s)  # I will crash if s is NULL
}
```

**My Fixed Version:**
```nano
extern fn may_return_null(x: int) -> opaque

fn use_result(x: int) -> int {
    let s: opaque = (may_return_null x)
    if (== s (null_opaque)) {
        return 0  # I check for NULL
    }
    let str: string = (opaque_to_string s)
    return (str_length str)
}
```

---

## How to Audit My FFI Bindings

### My Manual Audit Checklist

For every `extern fn` you declare:

1. Find the C documentation. Read the man pages or headers.
2. Check the return values. Know what -1, NULL, or 0 means.
3. Check the error conditions. Know how errors are reported.
4. Check the buffer requirements. Ensure lengths are specified.
5. Check thread safety. Can I call this from multiple threads?
6. Check for side effects. Does it modify global state?

### Automated Scanning

I suggest you create a script to find unsafe patterns.

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

## How to Test My FFI Bindings

### Testing for Buffer Overflows

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

### Testing for NULL Handling

```nano
extern fn process_string(s: string) -> int

shadow process_string {
    # Test with NULL
    assert (>= (process_string (null_string)) 0)
    
    # Test with empty string
    assert (>= (process_string "") 0)
}
```

---

## Documentation I Require

I expect every `extern fn` to be documented like this:

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
#       # Use the file...
#       (fclose f)
#   }
extern fn fopen(path: string, mode: string) -> opaque
```

---

## My Plan for Unsafe Blocks

I plan to add explicit `unsafe` blocks.

```nano
# I will mark unsafe FFI code
unsafe {
    extern fn dangerous_c_function(ptr: opaque) -> int
    
    fn wrapper(ptr: opaque) -> int {
        # I require careful review for everything in this block
        return (dangerous_c_function ptr)
    }
}

# My safe code remains outside
fn safe_function() -> int {
    return 42
}
```

This will make the trust boundaries clear. You will be able to find them by searching for `unsafe {`.

---

## My Security Best Practices

1. Minimize your FFI surface. Use my standard library when you can.
2. Use wrapper functions. Wrap unsafe C in my safe APIs.
3. Validate all inputs before passing them to C.
4. Check all return values.
5. Document your safety requirements.
6. Test your code with edge cases and fuzzing.
7. Have your FFI code reviewed.
8. Perform regular audits of your bindings.

---

## Advanced Safety Topics

### How I Handle Memory Ownership

I have complex ownership semantics when I talk to C. You must understand who owns the memory.

#### Caller-Owned (I Manage It)

This is my preferred pattern. I allocate and own the buffer.

```nano
fn safe_file_read(path: string) -> string {
    # I allocate and own this buffer
    let buffer: string = (string_new 4096)

    # The C function fills my buffer
    extern fn read_file_into_buffer(path: string, buf: string, size: int) -> int
    let bytes_read: int = (read_file_into_buffer path buffer 4096)

    if (< bytes_read 0) {
        return ""
    }

    # I still own the buffer. I will clean it up automatically.
    return buffer
}
```

My garbage collector handles the deallocation. There are no dangling pointers.

#### Callee-Owned (C Allocates It)

When C allocates memory, I require you to free it.

```nano
# C allocates a new string
extern fn c_strdup(s: string) -> string
extern fn c_free(ptr: string) -> void

fn duplicate_string(source: string) -> string {
    let copy: string = (c_strdup source)

    # I require you to call c_free later
    return copy
}

# I prefer this RAII wrapper:
fn safe_duplicate(source: string) -> string {
    let copy: string = (c_strdup source)
    let result: string = (string_copy copy)  # I copy it to my managed memory
    (c_free copy)  # I free the C memory immediately
    return result
}
```

I do not want you to return C-owned pointers directly to users.

#### Borrowed (No Transfer)

C borrows the memory and does not own it. This is safe.

```nano
# strlen only reads, it does not own
extern fn strlen(s: string) -> int

fn get_length(text: string) -> int {
    # I let strlen borrow 'text'
    return (strlen text)
}
```

---

### My RAII Patterns

I use RAII to ensure resources are cleaned up even when errors occur.

#### File Handle Wrapper

```nano
# C file operations
extern fn fopen(path: string, mode: string) -> int
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
        return ""
    }

    let buffer: string = (string_new 4096)
    let bytes: int = (fread buffer 1 4096 f.handle)

    # I always close the file
    (file_close f)

    if (< bytes 0) {
        return ""
    }

    return buffer
}
```

---

### My Thread Safety Considerations

I am concerned about thread safety when you use FFI.

#### Rule 1: Avoid Non-Reentrant C Functions

Many C functions are not thread-safe. I want you to use safe alternatives.

```nano
# I do not want you to use this. It uses global state.
extern fn strtok(str: string, delim: string) -> string

# Use this thread-safe alternative instead
extern fn strtok_r(str: string, delim: string, saveptr: int) -> string
```

#### Rule 2: Beware of Global State

If a C library has global state, I may have trouble if multiple components try to use it at once. I suggest you use locking or thread-local storage.

#### Rule 3: Callbacks from C

If C calls my functions, they run in C's thread context. I expect you to minimize shared state access and use message passing.

---

### Error Handling Across My Boundary

I have different conventions than C. I expect you to translate between them.

#### Pattern 1: Return Code Translation

```nano
# C returns 0 for success
extern fn c_operation(arg: string) -> int

fn safe_operation(arg: string) -> Result<void, string> {
    let status: int = (c_operation arg)

    if (== status 0) {
        return Ok(void)
    } else {
        let msg: string = (error_code_to_string status)
        return Err(msg)
    }
}
```

#### Pattern 2: errno Handling

```nano
extern fn c_errno() -> int
extern fn c_strerror(errnum: int) -> string

fn safe_file_operation(path: string) -> Result<void, string> {
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

---

### How I Validate Inputs

I always validate inputs before I pass them to C.

#### String Validation

I check for null bytes, length limits, and directory traversal.

```nano
fn safe_path_operation(user_path: string) -> Result<void, string> {
    if (str_contains user_path "\0") {
        return Err("Path contains null byte")
    }

    if (> (str_length user_path) 1024) {
        return Err("Path too long")
    }

    if (str_contains user_path "..") {
        return Err("Path traversal not allowed")
    }

    extern fn c_open_file(path: string) -> int
    let handle: int = (c_open_file user_path)

    if (== handle 0) {
        return Err("Failed to open file")
    }

    return Ok(void)
}
```

#### Integer Validation

I check for positive sizes and reasonable limits.

```nano
fn safe_buffer_operation(size: int) -> Result<string, string> {
    if (<= size 0) {
        return Err("Size must be positive")
    }

    if (> size 10485760) {
        return Err("Size too large")
    }

    extern fn c_malloc(size: int) -> string
    let buffer: string = (c_malloc size)

    if (== buffer "") {
        return Err("Allocation failed")
    }

    return Ok(buffer)
}
```

---

### Security: How I Prevent Injection

I want to prevent command injection and SQL injection.

```nano
# I do not allow direct command execution with user input
fn safe_run_command(filename: string) -> Result<int, string> {
    if (not (is_safe_filename filename)) {
        return Err("Invalid filename")
    }

    extern fn system(cmd: string) -> int
    let cmd: string = (str_concat "ls " filename)
    return Ok((system cmd))
}
```

I prefer prepared statements for SQL.

---

### My Performance Considerations

I know that FFI calls have overhead. I want you to optimize them.

#### Batch Your Operations

Instead of calling FFI in a tight loop, I recommend batching.

```nano
# I prefer this batch operation
fn process_fast(items: array<string>) -> void {
    extern fn c_process_batch(items: array<string>, count: int) -> void
    (c_process_batch items (array_length items))
}
```

#### Cache Your Results

I find it useful to cache the results of expensive FFI calls.

---

## My Related Documentation

- `docs/MODULE_SYSTEM.md`
- `docs/DEBUGGING_GUIDE.md`
- `docs/SPECIFICATION.md`
- `docs/MEMORY_MANAGEMENT.md`


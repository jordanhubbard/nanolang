# External C Function Interface (FFI)

**Version:** 1.0  
**Date:** November 12, 2025  
**Status:** ✅ Implemented

---

## Overview

nanolang supports calling external C functions through the `extern` keyword. This allows you to leverage existing C standard library functions directly from nanolang code without reimplementing them.

**Key Principle: Safety First**

We only expose **safe** C functions that:
- Take explicit length parameters for buffers
- Have no side effects on global state (except math errno)
- Cannot cause buffer overflows
- Are well-documented and standardized

> **Note:** For a detailed reference of safe C functions, see [Safe C FFI Functions Reference](SAFE_C_FFI_FUNCTIONS.md).

---

## Syntax

### Declaring External Functions

```nano
extern fn function_name(param1: type1, param2: type2) -> return_type
```

**Key Points:**
- Use the `extern` keyword before `fn`
- Provide the exact C function signature
- No function body - just the declaration
- No shadow test required for extern functions

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

## Conclusion

The extern FFI provides safe, efficient access to C standard library functions while maintaining nanolang's safety guarantees. By carefully selecting only safe functions and following best practices, you can leverage decades of C library development without compromising security.

**Remember:** When in doubt, create a safe wrapper!


# Safe C FFI Functions Reference

**Date:** November 12, 2025  
**Purpose:** Document safe C standard library functions for nanolang FFI

---

## Safety Guidelines

### String Functions

**❌ UNSAFE - Never expose these:**
- `strcpy()` - No bounds checking
- `strcat()` - No bounds checking
- `sprintf()` - No bounds checking
- `gets()` - Deprecated, always unsafe

**✅ SAFE - Use these instead:**
- `strncpy()` - Takes buffer size
- `strncat()` - Takes max characters
- `snprintf()` - Takes buffer size
- `strnlen()` - Takes max length
- `strncmp()` - Takes max characters

### Memory Functions

**✅ SAFE - Read-only operations:**
- `strlen()` - Safe on null-terminated strings
- `strcmp()` - Safe if both strings null-terminated
- `memcmp()` - Safe with correct length
- `memchr()` - Safe with length parameter

**⚠️ CAREFUL - Need wrappers:**
- `malloc()` - Must check return value
- `free()` - Must check not already freed
- `realloc()` - Must check return value

---

## Recommended Functions to Expose

### Category 1: Math (Always Safe)

```nano
# Trigonometric
extern fn asin(x: float) -> float
extern fn acos(x: float) -> float
extern fn atan(x: float) -> float
extern fn atan2(y: float, x: float) -> float

# Exponential/Logarithmic
extern fn exp(x: float) -> float
extern fn exp2(x: float) -> float
extern fn log(x: float) -> float
extern fn log10(x: float) -> float
extern fn log2(x: float) -> float

# Power
extern fn hypot(x: float, y: float) -> float
extern fn cbrt(x: float) -> float

# Rounding
extern fn trunc(x: float) -> float
extern fn rint(x: float) -> float
extern fn nearbyint(x: float) -> float

# Remainder
extern fn fmod(x: float, y: float) -> float
extern fn remainder(x: float, y: float) -> float
```

### Category 2: String (Read-Only, Safe)

```nano
# Length (safe - reads until null terminator)
extern fn strlen(s: string) -> int

# Comparison (safe if null-terminated)
extern fn strcmp(s1: string, s2: string) -> int
extern fn strncmp(s1: string, s2: string, n: int) -> int

# Search (safe - returns pointer or NULL)
extern fn strchr(s: string, c: int) -> string
extern fn strrchr(s: string, c: int) -> string
extern fn strstr(haystack: string, needle: string) -> string

# Character classification (safe - single char)
extern fn isalpha(c: int) -> int
extern fn isdigit(c: int) -> int
extern fn isalnum(c: int) -> int
extern fn isspace(c: int) -> int
extern fn isupper(c: int) -> int
extern fn islower(c: int) -> int
extern fn isprint(c: int) -> int

# Character conversion (safe - single char)
extern fn toupper(c: int) -> int
extern fn tolower(c: int) -> int
```

### Category 3: File I/O (Safe with Proper Usage)

```nano
# File operations (using opaque int for FILE*)
extern fn fopen(path: string, mode: string) -> int
extern fn fclose(file: int) -> int
extern fn fflush(file: int) -> int

# Character I/O (safe - single char)
extern fn fgetc(file: int) -> int
extern fn fputc(c: int, file: int) -> int
extern fn ungetc(c: int, file: int) -> int

# EOF checking
extern fn feof(file: int) -> int
extern fn ferror(file: int) -> int

# Position (safe)
extern fn ftell(file: int) -> int
extern fn fseek(file: int, offset: int, whence: int) -> int
extern fn rewind(file: int) -> void
```

### Category 4: Memory (Read-Only, Safe)

```nano
# Comparison (safe with correct length)
extern fn memcmp(s1: string, s2: string, n: int) -> int

# Search (safe with length)
extern fn memchr(s: string, c: int, n: int) -> int
```

---

## Functions to Wrap (Not Expose Directly)

These need nanolang wrappers for safety:

### String Copy/Manipulation

**Don't expose directly:**
```c
// UNSAFE in C
char* strcpy(char* dest, const char* src);
char* strcat(char* dest, const char* src);
```

**Instead, create safe wrappers in nanolang runtime:**
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

int64_t nl_safe_strcmp(const char* s1, const char* s2) {
    if (!s1 || !s2) return 0;
    return strcmp(s1, s2);
}
```

Then expose in nanolang:
```nano
extern fn nl_safe_strdup(s: string) -> string
extern fn nl_safe_strcmp(s1: string, s2: string) -> int
```

---

## Safe Usage Patterns

### Pattern 1: String Length Before Operations

```nano
extern fn strlen(s: string) -> int

fn safe_substring(s: string, start: int, len: int) -> string {
    let slen: int = (strlen s)
    
    # Bounds checking
    if (< start 0) { return "" }
    if (>= start slen) { return "" }
    if (<= len 0) { return "" }
    
    # Clamp length
    let remaining: int = (- slen start)
    let actual_len: int = (min len remaining)
    
    return (str_substring s start actual_len)
}
```

### Pattern 2: File Handle Validation

```nano
extern fn fopen(path: string, mode: string) -> int
extern fn fclose(file: int) -> int

fn with_file(path: string, mode: string, action: fn(int) -> int) -> int {
    let file: int = (fopen path mode)
    
    # Check for NULL (0)
    if (== file 0) {
        return -1
    }
    
    let result: int = (action file)
    (fclose file)
    
    return result
}
```

### Pattern 3: String Comparison with Null Checks

```nano
extern fn strcmp(s1: string, s2: string) -> int
extern fn strlen(s1: string) -> int

fn safe_equals(s1: string, s2: string) -> bool {
    # Both strings must be non-empty (null-terminated)
    if (== (strlen s1) 0) { return (== (strlen s2) 0) }
    if (== (strlen s2) 0) { return false }
    
    return (== (strcmp s1 s2) 0)
}
```

---

## Complete Safe Function List

### Always Safe (No Wrappers Needed)

**Math Functions (31):**
```nano
# Trigonometric (6)
extern fn sin(x: float) -> float
extern fn cos(x: float) -> float
extern fn tan(x: float) -> float
extern fn asin(x: float) -> float
extern fn acos(x: float) -> float
extern fn atan(x: float) -> float
extern fn atan2(y: float, x: float) -> float

# Hyperbolic (6)
extern fn sinh(x: float) -> float
extern fn cosh(x: float) -> float
extern fn tanh(x: float) -> float
extern fn asinh(x: float) -> float
extern fn acosh(x: float) -> float
extern fn atanh(x: float) -> float

# Exponential (5)
extern fn exp(x: float) -> float
extern fn exp2(x: float) -> float
extern fn expm1(x: float) -> float
extern fn log(x: float) -> float
extern fn log10(x: float) -> float
extern fn log2(x: float) -> float
extern fn log1p(x: float) -> float

# Power (3)
extern fn pow(x: float, y: float) -> float
extern fn sqrt(x: float) -> float
extern fn cbrt(x: float) -> float
extern fn hypot(x: float, y: float) -> float

# Rounding (7)
extern fn ceil(x: float) -> float
extern fn floor(x: float) -> float
extern fn round(x: float) -> float
extern fn trunc(x: float) -> float
extern fn rint(x: float) -> float
extern fn nearbyint(x: float) -> float

# Other (4)
extern fn fabs(x: float) -> float
extern fn fmod(x: float, y: float) -> float
extern fn remainder(x: float, y: float) -> float
extern fn copysign(x: float, y: float) -> float
```

**Character Classification (12):**
```nano
extern fn isalpha(c: int) -> int
extern fn isdigit(c: int) -> int
extern fn isalnum(c: int) -> int
extern fn isspace(c: int) -> int
extern fn isupper(c: int) -> int
extern fn islower(c: int) -> int
extern fn isprint(c: int) -> int
extern fn ispunct(c: int) -> int
extern fn iscntrl(c: int) -> int
extern fn isxdigit(c: int) -> int
extern fn isblank(c: int) -> int
extern fn isgraph(c: int) -> int

extern fn toupper(c: int) -> int
extern fn tolower(c: int) -> int
```

**String Read-Only (4):**
```nano
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int
extern fn strncmp(s1: string, s2: string, n: int) -> int
extern fn strchr(s: string, c: int) -> string
```

---

## Security Checklist

Before exposing any C function:

- [ ] ✅ Takes explicit length parameters for buffers?
- [ ] ✅ Returns bounded results?
- [ ] ✅ No pointer arithmetic required?
- [ ] ✅ No complex types (structs, unions)?
- [ ] ✅ Safe with nanolang's type system?
- [ ] ✅ Can't cause buffer overflow?
- [ ] ✅ Can't cause use-after-free?
- [ ] ✅ Clear error handling?

---

## Recommended Initial Set (Version 1)

Start with these 50 safe functions:

**Math (31):** All trigonometric, exponential, power, rounding  
**Character (14):** All classification + toupper/tolower  
**String (4):** strlen, strcmp, strncmp, strchr  
**File I/O (0):** Wait for careful wrapper design

**Total: 49 functions** - All completely safe!

---

## Future: Wrapper Library

Create `src/runtime/safe_wrappers.h`:

```c
// Safe string operations
const char* nl_safe_strdup(const char* s);
int64_t nl_safe_strncpy(char* dest, const char* src, size_t n);
int64_t nl_safe_snprintf(char* buf, size_t size, const char* fmt, ...);

// Safe file operations with error checking
int64_t nl_safe_fopen(const char* path, const char* mode);
int64_t nl_safe_fclose(int64_t file_ptr);
int64_t nl_safe_fgetc(int64_t file_ptr);

// Safe memory operations
int64_t nl_safe_malloc(size_t size);
void nl_safe_free(int64_t ptr);
```

---

## Conclusion

**Safety-First Approach:**

1. ✅ Expose only inherently safe functions (math, char classification)
2. ✅ Use length-bounded string functions (strncmp, not strcmp for writes)
3. ✅ Create safe wrappers for dangerous operations
4. ✅ Document safe usage patterns
5. ✅ Provide high-level nanolang helpers

**Recommended Implementation Order:**

1. **Week 1:** Math functions (31) - completely safe
2. **Week 2:** Character functions (14) - completely safe
3. **Week 3:** String read-only (4) - safe with null termination
4. **Week 4:** Safe wrapper library + file I/O

This gives us ~50 immediately safe functions while we carefully design wrappers for more complex operations.


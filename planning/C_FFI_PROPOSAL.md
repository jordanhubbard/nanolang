# C FFI (Foreign Function Interface) Proposal

**Date:** November 12, 2025  
**Purpose:** Enable direct calls to C standard library functions from nanolang

---

## Motivation

nanolang already transpiles to C, so we can leverage the entire C standard library without:
- Reimplementing everything as builtins in the interpreter
- Maintaining duplicate implementations
- Missing out on optimized C stdlib implementations

Since nanolang's type system maps cleanly to C's basic types, we can call most C functions directly.

---

## Type Compatibility

### nanolang → C Type Mapping

| nanolang Type | C Type | Notes |
|---------------|--------|-------|
| `int` | `int64_t` | 64-bit signed |
| `float` | `double` | 64-bit float |
| `bool` | `bool` | C99 stdbool |
| `string` | `const char*` | Null-terminated |
| `void` | `void` | No return value |

### Opaque Handles
For C pointers we don't need to dereference (like `FILE*`), use `int`:
```nano
extern fn fopen(path: string, mode: string) -> int  # Returns FILE* as int64_t
extern fn fclose(file: int) -> int                   # Takes FILE* as int64_t
```

**Rationale:** Cast pointer to `int64_t` for storage, cast back for C calls.

---

## Proposed Syntax

### Option 1: `extern` Keyword (Recommended)

```nano
extern fn c_function_name(param: type, ...) -> return_type
```

**Examples:**
```nano
# String functions
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int
extern fn strdup(s: string) -> string

# Math functions
extern fn log(x: float) -> float
extern fn exp(x: float) -> float
extern fn atan2(y: float, x: float) -> float

# File I/O (using opaque handles)
extern fn fopen(path: string, mode: string) -> int
extern fn fclose(file: int) -> int
extern fn fgetc(file: int) -> int
extern fn fputc(c: int, file: int) -> int

# Memory (advanced use)
extern fn malloc(size: int) -> int
extern fn free(ptr: int) -> void
```

**Properties:**
- ✅ No function body required
- ✅ No shadow test required (can't test external functions)
- ✅ Clear intent: this is an external C function
- ✅ Compatible with C `extern` semantics

### Option 2: `cfn` Keyword (Alternative)

```nano
cfn strlen(s: string) -> int
```

**Pros:** Shorter, more distinct  
**Cons:** New keyword, less conventional

### Option 3: Explicit C Name (If Different)

```nano
extern "c_name" fn nano_name(params) -> return_type
```

**Example:**
```nano
extern "strtoll" fn parse_long(s: string, base: int) -> int
```

**Use case:** When you want a different name in nanolang than C.

---

## Implementation Plan

### Phase 1: Parser Changes

**File:** `src/parser.c`

Add `extern` keyword and allow function declarations without body:

```c
// In parse_function_definition():
bool is_extern = false;
if (current_token(p)->type == TOKEN_IDENTIFIER && 
    strcmp(current_token(p)->value, "extern") == 0) {
    is_extern = true;
    advance(p);
}

// ... parse function signature ...

if (is_extern) {
    // No body expected
    item->as.function.is_extern = true;
    item->as.function.body = NULL;
} else {
    // Regular function, expect body
    item->as.function.body = parse_block(p);
}
```

**AST Changes:**
```c
typedef struct {
    // ... existing fields ...
    bool is_extern;  // NEW: Mark external C functions
} FunctionDef;
```

### Phase 2: Type Checker Changes

**File:** `src/typechecker.c`

Skip body and shadow test requirements for extern functions:

```c
case AST_FUNCTION: {
    // ... existing signature checks ...
    
    if (!item->as.function.is_extern) {
        // Regular function - check body and require shadow test
        check_statement(item->as.function.body, tc->env);
        
        if (!item->as.function.shadow_test) {
            fprintf(stderr, "Error: Function '%s' missing shadow test\n", 
                    item->as.function.name);
            tc->has_error = true;
        }
    } else {
        // Extern function - no body or shadow test needed
        if (item->as.function.body) {
            fprintf(stderr, "Error: Extern function '%s' cannot have a body\n",
                    item->as.function.name);
            tc->has_error = true;
        }
    }
}
```

### Phase 3: Transpiler Changes

**File:** `src/transpiler.c`

Generate function declarations instead of definitions:

```c
// In transpile_function():
if (func->is_extern) {
    // Generate extern declaration
    sb_appendf(sb, "// Extern C function: %s\n", func->name);
    // Don't generate function body - will use C's version
    return;
}

// In transpile_function_call():
if (function_is_extern(func_name, env)) {
    // Call C function directly, no nl_ prefix
    sb_appendf(sb, "%s(", func_name);
} else {
    // Call nanolang function with prefix
    sb_appendf(sb, "%s(", get_c_func_name(func_name));
}
```

### Phase 4: Interpreter Changes

**File:** `src/eval.c`

**Option A: Hardcoded Table (Simple)**
```c
static Value call_extern_function(const char *name, Value *args, int arg_count) {
    // Hardcoded common functions
    if (strcmp(name, "strlen") == 0) {
        return create_int(strlen(args[0].as.string_val));
    }
    if (strcmp(name, "log") == 0) {
        return create_float(log(args[0].as.float_val));
    }
    // ... more functions ...
    
    fprintf(stderr, "Error: Extern function '%s' not available in interpreter\n", name);
    return create_void();
}
```

**Option B: Error (Compiler-Only)**
```c
if (func->is_extern) {
    fprintf(stderr, "Error: Extern functions only supported in compiler mode\n");
    fprintf(stderr, "Use './nanoc' instead of './nano'\n");
    exit(1);
}
```

**Recommendation:** Start with Option A for common functions, fall back to error for others.

---

## Usage Examples

### Example 1: Extended Math Library

```nano
# Declare external math functions
extern fn log(x: float) -> float
extern fn log10(x: float) -> float
extern fn exp(x: float) -> float
extern fn atan2(y: float, x: float) -> float
extern fn hypot(x: float, y: float) -> float

fn calculate_entropy(p: float) -> float {
    if (<= p 0.0) {
        return 0.0
    }
    return (* (- 0.0 p) (log p))
}

fn main() -> int {
    let entropy: float = (calculate_entropy 0.5)
    print "Entropy: "
    print entropy
    return 0
}
```

### Example 2: File I/O with FILE* Handles

```nano
# Declare file operations (FILE* as int)
extern fn fopen(path: string, mode: string) -> int
extern fn fclose(file: int) -> int
extern fn fgetc(file: int) -> int
extern fn fputc(c: int, file: int) -> int
extern fn feof(file: int) -> int

fn read_first_char(path: string) -> int {
    let file: int = (fopen path "r")
    if (== file 0) {
        return -1  # Error
    }
    
    let c: int = (fgetc file)
    (fclose file)
    return c
}

fn main() -> int {
    let first: int = (read_first_char "README.md")
    print "First character code: "
    print first
    return 0
}
```

### Example 3: String Processing

```nano
# Declare string functions
extern fn strlen(s: string) -> int
extern fn strcmp(s1: string, s2: string) -> int
extern fn strstr(haystack: string, needle: string) -> string

fn find_substring(text: string, pattern: string) -> bool {
    let result: string = (strstr text pattern)
    # strstr returns NULL (empty string) if not found
    return (> (strlen result) 0)
}

fn main() -> int {
    if (find_substring "Hello World" "World") {
        print "Found!"
    }
    return 0
}
```

---

## Safety Considerations

### Type Safety

✅ **Safe:**
- All parameters type-checked at compile time
- Return types verified
- No void* or complex types exposed

⚠️ **Unsafe (But Manageable):**
- Opaque handles (`FILE*` as `int`) require care
- Must match C function signatures exactly
- Segfaults possible if used incorrectly

**Mitigation:**
- Document each extern function carefully
- Provide safe wrappers in nanolang stdlib
- Error checking in nanolang code

### Memory Safety

**Rules:**
1. Strings passed to C are read-only (`const char*`)
2. Strings returned from C must be copied (use `strdup` pattern)
3. Opaque handles must be closed/freed properly
4. No manual memory management in safe code

**Example Safe Pattern:**
```nano
# SAFE: Wrapper provides cleanup
fn with_file(path: string, mode: string, handler: fn(int) -> int) -> int {
    let file: int = (fopen path mode)
    if (== file 0) {
        return -1
    }
    
    let result: int = (handler file)
    (fclose file)
    return result
}
```

---

## Benefits

### For Users
1. **More Functions:** Access entire C standard library
2. **Better Performance:** Optimized C implementations
3. **Less Reimplementation:** Don't duplicate effort
4. **Flexibility:** Call any C function that fits the type system

### For Implementation
1. **Simpler Builtins:** Less code in interpreter
2. **Better Transpiler:** More powerful C generation
3. **Maintainability:** Let C stdlib handle edge cases
4. **Compatibility:** Easy to add new functions

### For Self-Hosting
1. **More Tools:** Access to file I/O, string ops, etc.
2. **Better Compiler:** Can use C's lexing/parsing helpers
3. **Performance:** Fast stdlib functions
4. **Portability:** C stdlib works everywhere

---

## Limitations

### Not Supported
- ❌ Complex C types (structs passed by value)
- ❌ Variadic functions (except manually wrapped)
- ❌ Function pointers (except as opaque int)
- ❌ Macros (need function wrappers)
- ❌ Complex pointer arithmetic

### Workarounds
Most limitations can be addressed with thin wrapper functions in C:

```c
// nanolang_wrappers.h
int64_t nl_printf_int(const char *fmt, int64_t value) {
    return printf(fmt, value);
}

int64_t nl_printf_string(const char *fmt, const char *value) {
    return printf(fmt, value);
}
```

Then declare in nanolang:
```nano
extern fn nl_printf_int(fmt: string, value: int) -> int
extern fn nl_printf_string(fmt: string, value: string) -> int
```

---

## Migration Path

### Phase 1: Core Functions (Week 1)
- Math: `log`, `exp`, `log10`, `atan2`, `hypot`
- String: `strlen`, `strcmp`, `strstr`, `strchr`
- I/O: `fopen`, `fclose`, `fgetc`, `fputc`

### Phase 2: Extended Functions (Week 2)
- More string: `strncmp`, `strcpy`, `strcat`
- More I/O: `fread`, `fwrite`, `fseek`, `ftell`
- Memory: `malloc`, `free`, `realloc` (advanced)

### Phase 3: Helper Wrappers (Week 3)
- Printf family: `nl_printf_*` wrappers
- Scanf family: `nl_scanf_*` wrappers
- Advanced I/O: buffered operations

---

## Alternative: Inline C Blocks

**Future extension** - Allow inline C code:

```nano
fn unsafe_c_operation(x: int) -> int {
    inline_c """
        return x * 2 + 1;
    """
}
```

**Pros:** Maximum flexibility  
**Cons:** Breaks type safety, hard to implement  
**Recommendation:** Start with `extern` functions first

---

## Recommendation

**Start with `extern` keyword approach:**

1. ✅ Clean syntax
2. ✅ Type-safe
3. ✅ Conventional (like C, Rust, etc.)
4. ✅ Easy to implement
5. ✅ Solves immediate needs

**Implementation order:**
1. Parser + AST (1 day)
2. Type checker (1 day)
3. Transpiler (1 day)
4. Interpreter (optional, 1-2 days)
5. Documentation + examples (1 day)

**Total time:** 3-5 days for full implementation

---

## Example: Self-Hosting Use Case

```nano
# Lexer using C string functions for performance
extern fn isalpha(c: int) -> int
extern fn isdigit(c: int) -> int
extern fn isalnum(c: int) -> int

fn tokenize_identifier(source: string, start: int) -> string {
    let mut end: int = start
    let len: int = (str_length source)
    
    while (< end len) {
        let c: int = (char_at source end)
        if (== (isalnum c) 0) {
            return (str_substring source start (- end start))
        }
        set end (+ end 1)
    }
    
    return (str_substring source start (- end start))
}
```

---

## Conclusion

**Recommendation:** Implement `extern fn` syntax for C FFI.

**Why:**
- Leverages existing C transpilation
- Type-safe and explicit
- Enables self-hosting compiler
- Simple to implement
- Huge functionality gain

**Next Steps:**
1. Approve syntax
2. Implement parser changes
3. Update type checker
4. Modify transpiler
5. Add documentation
6. Create examples

This would be a major feature that makes nanolang significantly more powerful! 🚀


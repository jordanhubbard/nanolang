# Advanced String Operations Implementation Plan

**Date:** November 12, 2025  
**Status:** 🚧 IN PROGRESS  
**Goal:** Complete final feature for self-hosting (6/6 = 100%)

---

## Overview

String operations are the **final missing piece** for self-hosting. The compiler needs to:
- Parse source code character by character
- Build strings dynamically (for C code generation)
- Classify characters (digits, letters, whitespace)
- Convert between strings and integers

**Timeline:** 1-2 weeks  
**Current Progress:** 5/6 features (83%) → Target: 6/6 (100%)

---

## Required Operations

### Priority 1: Character Access (Critical)

```nano
# Get ASCII value of character at index
fn char_at(s: string, index: int) -> int

# Get character as single-character string
fn char_at_str(s: string, index: int) -> string
```

**Why Critical:** The lexer needs to examine each character of source code.

**Example Usage:**
```nano
let source: string = "let x = 42"
let c: int = (char_at source 0)  # Returns 108 (ASCII 'l')
if (== c 108) {
    print "Found 'let' keyword"
}
```

### Priority 2: Character Classification (Critical)

```nano
fn is_digit(c: int) -> bool      # '0'-'9'
fn is_alpha(c: int) -> bool      # 'a'-'z', 'A'-'Z'
fn is_alnum(c: int) -> bool      # digit or alpha
fn is_whitespace(c: int) -> bool # ' ', '\t', '\n', '\r'
fn is_upper(c: int) -> bool      # 'A'-'Z'
fn is_lower(c: int) -> bool      # 'a'-'z'
```

**Why Critical:** The lexer needs to classify characters for tokenization.

**Example Usage:**
```nano
fn parse_number(source: string, start: int) -> int {
    let c: int = (char_at source start)
    if (is_digit c) {
        # Parse the number
        return (digit_value c)
    }
    return -1
}
```

### Priority 3: String Building (Critical)

```nano
fn string_from_char(c: int) -> string      # Create string from ASCII
fn string_append(s1: string, s2: string) -> string  # Already exists!
fn string_repeat(s: string, count: int) -> string
```

**Why Critical:** Building C code output requires dynamic string construction.

**Example Usage:**
```nano
fn generate_c_code(name: string, value: int) -> string {
    let code: string = "int "
    set code (str_concat code name)
    set code (str_concat code " = ")
    set code (str_concat code (int_to_string value))
    set code (str_concat code ";\n")
    return code
}
```

### Priority 4: Conversion Functions (Critical)

```nano
fn int_to_string(n: int) -> string
fn string_to_int(s: string) -> int
fn digit_value(c: int) -> int       # '5' -> 5
fn char_to_lower(c: int) -> int     # 'A' -> 'a'
fn char_to_upper(c: int) -> int     # 'a' -> 'A'
```

**Why Critical:** Converting between types during parsing and code generation.

### Priority 5: Utility Functions (Nice to Have)

```nano
fn string_split(s: string, delim: string) -> list_string
fn string_trim(s: string) -> string
fn string_starts_with(s: string, prefix: string) -> bool
fn string_ends_with(s: string, suffix: string) -> bool
fn string_replace(s: string, old: string, new: string) -> string
```

**Why Nice:** Makes string manipulation easier but can be implemented in nanolang later.

---

## Implementation Strategy

### Phase 1: Character Access (Day 1)

**Add to `src/eval.c` and `src/typechecker.c`:**

```c
/* char_at(s: string, index: int) -> int */
static Value builtin_char_at(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_INT) {
        fprintf(stderr, "Error: char_at requires string and int\n");
        return create_void();
    }
    
    const char *str = args[0].as.string_val;
    long long index = args[1].as.int_val;
    int len = strlen(str);
    
    if (index < 0 || index >= len) {
        fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", index, len);
        return create_void();
    }
    
    return create_int((unsigned char)str[index]);
}

/* string_from_char(c: int) -> string */
static Value builtin_string_from_char(Value *args) {
    if (args[0].type != VAL_INT) {
        fprintf(stderr, "Error: string_from_char requires int\n");
        return create_void();
    }
    
    char buffer[2];
    buffer[0] = (char)args[0].as.int_val;
    buffer[1] = '\0';
    return create_string(buffer);
}
```

### Phase 2: Character Classification (Day 2)

**Add to `src/eval.c`:**

```c
/* is_digit(c: int) -> bool */
static Value builtin_is_digit(Value *args) {
    if (args[0].type != VAL_INT) return create_bool(false);
    int c = (int)args[0].as.int_val;
    return create_bool(c >= '0' && c <= '9');
}

/* is_alpha(c: int) -> bool */
static Value builtin_is_alpha(Value *args) {
    if (args[0].type != VAL_INT) return create_bool(false);
    int c = (int)args[0].as.int_val;
    return create_bool((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
}

/* is_alnum(c: int) -> bool */
static Value builtin_is_alnum(Value *args) {
    if (args[0].type != VAL_INT) return create_bool(false);
    int c = (int)args[0].as.int_val;
    return create_bool((c >= '0' && c <= '9') || 
                       (c >= 'a' && c <= 'z') || 
                       (c >= 'A' && c <= 'Z'));
}

/* is_whitespace(c: int) -> bool */
static Value builtin_is_whitespace(Value *args) {
    if (args[0].type != VAL_INT) return create_bool(false);
    int c = (int)args[0].as.int_val;
    return create_bool(c == ' ' || c == '\t' || c == '\n' || c == '\r');
}
```

### Phase 3: Conversions (Day 3)

**Add to `src/eval.c`:**

```c
/* int_to_string(n: int) -> string */
static Value builtin_int_to_string(Value *args) {
    if (args[0].type != VAL_INT) {
        return create_string("0");
    }
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%lld", args[0].as.int_val);
    return create_string(buffer);
}

/* string_to_int(s: string) -> int */
static Value builtin_string_to_int(Value *args) {
    if (args[0].type != VAL_STRING) {
        return create_int(0);
    }
    
    long long result = strtoll(args[0].as.string_val, NULL, 10);
    return create_int(result);
}

/* digit_value(c: int) -> int */
static Value builtin_digit_value(Value *args) {
    if (args[0].type != VAL_INT) return create_int(-1);
    int c = (int)args[0].as.int_val;
    if (c >= '0' && c <= '9') {
        return create_int(c - '0');
    }
    return create_int(-1);
}
```

### Phase 4: Registration (Day 4)

**Update `src/typechecker.c`:**

1. Add function names to `builtin_function_names[]`
2. Register signatures in `register_builtin_functions()`

**Update transpiler:**
- Add function names to `get_c_func_name()` to prevent prefixing
- Generate inline implementations or include string helper header

### Phase 5: Testing (Day 5)

**Create `examples/20_string_advanced_test.nano`:**

```nano
fn test_char_access() -> int {
    let s: string = "Hello"
    
    assert (== (char_at s 0) 72)   # 'H'
    assert (== (char_at s 1) 101)  # 'e'
    assert (== (char_at s 4) 111)  # 'o'
    
    return 0
}

fn test_char_classification() -> int {
    let digit: int = 53        # '5'
    let letter: int = 65       # 'A'
    let space: int = 32        # ' '
    
    assert (== (is_digit digit) true)
    assert (== (is_digit letter) false)
    
    assert (== (is_alpha letter) true)
    assert (== (is_alpha digit) false)
    
    assert (== (is_whitespace space) true)
    
    return 0
}

fn test_conversions() -> int {
    assert (== (int_to_string 42) "42")
    assert (== (int_to_string -10) "-10")
    
    assert (== (string_to_int "123") 123)
    assert (== (string_to_int "-456") -456)
    
    assert (== (digit_value 53) 5)  # '5' -> 5
    
    return 0
}

fn test_string_building() -> int {
    let c: int = 65  # 'A'
    let s: string = (string_from_char c)
    assert (== s "A")
    
    return 0
}
```

---

## C Code Generation Approach

### Option 1: Inline in Transpiled Code

Generate helper functions directly in the C output:

```c
// Generated at top of transpiled C file
static int64_t nl_char_at(const char* s, int64_t index) {
    return (unsigned char)s[index];
}

static const char* nl_string_from_char(int64_t c) {
    static char buffer[2];
    buffer[0] = (char)c;
    buffer[1] = '\0';
    return buffer;
}
```

**Pros:** Self-contained C output  
**Cons:** Duplicated code in every transpiled file

### Option 2: Runtime Header

Create `src/runtime/string_ops.h` and link it:

```c
// src/runtime/string_ops.h
int64_t char_at(const char* s, int64_t index);
const char* string_from_char(int64_t c);
bool is_digit(int64_t c);
// ...
```

**Pros:** No duplication, cleaner  
**Cons:** Another runtime dependency

**Recommendation:** Option 1 for simplicity (inline helpers)

---

## Implementation Checklist

### Day 1: Character Access
- [ ] Implement `char_at` in eval.c
- [ ] Implement `string_from_char` in eval.c
- [ ] Register in typechecker
- [ ] Add transpiler support
- [ ] Test in interpreter

### Day 2: Classification
- [ ] Implement `is_digit`, `is_alpha`, `is_alnum`, `is_whitespace`
- [ ] Register in typechecker
- [ ] Add transpiler support
- [ ] Test all classification functions

### Day 3: Conversions
- [ ] Implement `int_to_string`, `string_to_int`
- [ ] Implement `digit_value`
- [ ] Implement `char_to_lower`, `char_to_upper`
- [ ] Register and test

### Day 4: Integration
- [ ] Update builtin list
- [ ] Update transpiler code generation
- [ ] Ensure all functions work in both modes

### Day 5: Testing
- [ ] Write comprehensive test file
- [ ] Test all operations in interpreter
- [ ] Test all operations in compiler
- [ ] Memory safety checks
- [ ] Edge case testing

### Day 6-7: Documentation
- [ ] Update STDLIB.md
- [ ] Create usage examples
- [ ] Write STRING_OPERATIONS_COMPLETE.md
- [ ] Update SELF_HOSTING_REQUIREMENTS.md

---

## Success Criteria

✅ **Feature Complete When:**
- All 15+ string functions implemented
- Work in both interpreter and compiler
- All tests passing (100%)
- Documentation complete
- Self-hosting features: **6/6 (100%)**

✅ **Ready for Compiler Rewrite When:**
- Can parse character-by-character
- Can classify tokens
- Can build strings dynamically
- Can convert types
- All shadow tests passing

---

## Timeline

**Week 1:**
- Days 1-3: Implementation
- Days 4-5: Testing and bug fixes
- Days 6-7: Documentation

**Week 2 (Optional):**
- Advanced features (split, trim, etc.)
- Performance optimization
- Additional utility functions

---

## Next Steps After String Operations

Once string operations are complete (6/6 features = 100%):

### Phase 1: Lexer in nanolang (2 weeks)
```nano
fn tokenize(source: string) -> list_token {
    let mut tokens: list_token = (list_token_new)
    let mut pos: int = 0
    let len: int = (str_length source)
    
    while (< pos len) {
        let c: int = (char_at source pos)
        
        if (is_whitespace c) {
            set pos (+ pos 1)
        } else if (is_digit c) {
            # Parse number token
            let tok: Token = (parse_number source pos)
            (list_token_push tokens tok)
            # Update pos
        } else if (is_alpha c) {
            # Parse identifier
            # ...
        }
    }
    
    return tokens
}
```

### Phase 2: Parser in nanolang (3 weeks)
### Phase 3: Type Checker in nanolang (3 weeks)
### Phase 4: Transpiler in nanolang (3 weeks)
### Phase 5: Bootstrap! (1 week)

**Total Timeline to Self-Hosting:** 3-4 months from now

---

## Conclusion

String operations are the **final piece** of the puzzle. After this, nanolang will have all the features needed to implement its own compiler.

**Current Status:** 5/6 (83%)  
**After String Ops:** 6/6 (100%) ✅  
**Next Milestone:** Begin Compiler Rewrite

Let's implement string operations and complete the foundation for self-hosting! 🚀


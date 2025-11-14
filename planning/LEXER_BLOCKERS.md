# Lexer Self-Hosting Blockers

## Critical Blocker #1: Enum Variant Access Not Transpiling

### Issue
Enum variant access using dot notation (e.g., `TokenType.FN`) doesn't properly transpile to C.

**Example:**
```nano
enum TokenType {
    FN = 19
}

fn test() -> int {
    return TokenType.FN  /* Should return 19 */
}
```

**Expected C Output:**
```c
typedef enum {
    TokenType_FN = 19
} TokenType;

int64_t test() {
    return TokenType_FN;  /* or just: return 19; */
}
```

**Actual C Output:**
```c
int64_t test() {
    return FN;  /* ERROR: undeclared identifier */
}
```

### Impact
- Cannot use clean enum variant syntax in nanolang code
- Must use magic numbers instead (error-prone, not maintainable)
- Blocks proper lexer implementation

### Workaround
Use integer literals directly:
```nano
/* Instead of: return TokenType.FN */
return 19  /* TokenType.FN */
```

### Fix Required
Update transpiler (`src/transpiler.c`) to properly handle enum variant access in `AST_FIELD_ACCESS` when the object is an enum type.

---

## Critical Blocker #2: No Dynamic Array/List for Structs

### Issue
Cannot build dynamic arrays of custom structs (like `Token`).

**Available:**
- `List_int` - dynamic list of integers
- `List_string` - dynamic list of strings
- `array<T>` - fixed-size arrays only

**Missing:**
- `List<Token>` or `List_token`
- Generic `List<T>` support
- `array_push()` / `array_pop()` for dynamic arrays

### Impact
- Cannot collect unbounded number of tokens in lexer
- Would need to pre-allocate huge fixed-size array (wasteful, limiting)
- Blocks completion of lexer `lex()` function

### Workarounds

**Option A**: Pre-allocate large fixed array (limiting)
```nano
let mut tokens: array<Token> = []  /* How to size this? */
```

**Option B**: Use parallel arrays (ugly, error-prone)
```nano
let mut token_types: array<int> = []
let mut token_values: array<string> = []
let mut token_lines: array<int> = []
let mut token_columns: array<int> = []
```

**Option C**: Implement `List_token` in C (best short-term)
- Copy `src/runtime/list_int.c` to `src/runtime/list_token.c`
- Replace `int` with `Token*` or embed token data
- Add builtin functions: `list_token_new()`, `list_token_push()`, etc.

### Fix Required
Either:
1. Add `List_token` runtime type (fastest)
2. Implement generic `List<T>` (better long-term)
3. Add `array_push()` builtin for dynamic arrays

---

## Minor Issue #3: Break Statement

### Issue
Parser expects `break` keyword in loops but it may not be fully supported.

**Code:**
```nano
while (< i len) {
    if (condition) {
        break  /* Exit loop early */
    } else {}
    set i (+ i 1)
}
```

### Impact
- Minor - can work around with flags or restructure logic
- Not blocking for lexer implementation

### Workaround
Use boolean flags:
```nano
let mut done: bool = false
while (and (< i len) (not done)) {
    if (condition) {
        set done true
    } else {}
    set i (+ i 1)
}
```

---

## Priority

1. **HIGH**: Fix enum variant access transpilation
   - Affects code quality and maintainability
   - Workaround exists but is ugly
   - Should be fixed soon

2. **HIGH**: Add dynamic array/list support
   - Completely blocks lexer completion
   - `List_token` is fastest solution
   - Generic `List<T>` is better long-term

3. **LOW**: Break statement
   - Easy workarounds available
   - Not blocking

---

## Recommendations

### Immediate (Week 1)
1. Fix enum variant access in transpiler
   - Look at how struct field access works
   - Extend to handle enum type checking
   - Generate proper C enum names

### Short-term (Week 2-3)
2. Implement `List_token` runtime type
   - Clone `list_int.c` structure
   - Adapt for Token struct
   - Add to build system
   - Register builtin functions

3. Complete lexer implementation
   - Use `List_token` for token storage
   - Finish all operator tokenization
   - Add comprehensive tests

### Medium-term (Month 1-2)
4. Design and implement generic `List<T>`
   - Consider code generation approach
   - Or runtime type erasure
   - Replace `List_int`, `List_string`, `List_token` with generic version

---

## Testing Plan

Once blockers are resolved:

1. **Unit tests** for lexer helper functions (already have shadow tests ✅)
2. **Integration test**: Compare nano lexer vs C lexer output
3. **Corpus testing**: Run on all example programs
4. **Performance benchmarking**: Compare speeds
5. **Bootstrapping**: Use nano lexer in compiler pipeline

---

## Current Status

**Lexer Implementation**: 60% complete
- ✅ Token enum and struct
- ✅ Character classification
- ✅ String utilities
- ✅ Keyword classification
- ✅ Shadow tests
- ❌ Complete tokenization (blocked by dynamic arrays)
- ❌ Enum variant access (blocked by transpiler bug)

**Can Proceed With**: Further design, testing infrastructure, documentation
**Cannot Proceed With**: Actual token collection and complete implementation

**ETA**: 2-3 weeks after blockers resolved


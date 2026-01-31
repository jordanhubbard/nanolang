# Known Issues in NanoLang Compiler

This document tracks known bugs and limitations in the NanoLang compiler that need to be addressed.

## Critical Bugs

### 1. Expression Statements Not Validated (Parser/Typechecker Gap)

**Severity:** High  
**Component:** Parser/Typechecker  
**Discovered:** 2026-01-14

**Description:**
The compiler allows standalone expressions as statements without validating that they have side effects or are meaningful. This means syntactically invalid code can compile successfully.

**Example - Should FAIL but compiles with only warnings:**
```nano
fn main() -> int {
    let x: int = 5
    let y: int = 10
    
    # Invalid: just identifiers doing nothing
    x y
    
    return 0
}
```

**What happens:**
- NanoLang compiler: Compiles with C compiler warnings about "unused value"
- Should be: **Compilation error** - standalone expressions without side effects are not valid statements

**Root Cause:**
The parser allows any expression to be used as a statement without validation. The typechecker doesn't verify that expression statements:
1. Are function calls (which may have side effects)
2. Are assignments/mutations
3. Have any meaningful purpose

**Impact:**
- LLM-generated code with syntax errors (like deleted function names) can compile
- Makes debugging harder because invalid code silently compiles
- Reduces compiler's ability to catch programmer errors
- False sense of correctness when code "compiles successfully"

**Files Involved:**
- `src/parser.c` line ~2550: "Try to parse as expression statement"
- `src/typechecker.c` line ~2919: "Expression statement" handling
- `src/transpiler.c`: Expression statement transpilation

**Potential Fix:**
Add validation in typechecker for expression statements:
1. If expression is a function call -> allow (may have side effects)
2. If expression is assignment/set -> allow (mutation)
3. If expression is pure identifier/literal -> **ERROR** "Expression statement has no effect"
4. For other operators, could warn or error

**Related Issues:**
- Self-hosted typechecker also has gaps (see tests/selfhost/test_function_arg_type_errors.nano)
- Need comprehensive expression statement validation rules

**Priority:** High - this masks real bugs and reduces code quality

---

---

### 3. For-In Loops Not Transpiled (P0) - ✅ FIXED

**Severity:** P0 - Critical  
**Component:** Transpiler (transpiler_iterative_v3_twopass.c)  
**Discovered:** 2026-01-14  
**Fixed:** 2026-01-14

**Description:**
The `for x in (range ...)` loop construct was not properly transpiled to C. Instead of generating actual loop code, the transpiler emitted `/* unsupported expr type 14 */`, causing loop bodies to be completely skipped.

**Fix Applied:**
Added AST_FOR case to `build_stmt()` in `src/transpiler_iterative_v3_twopass.c` at line 2554. The transpiler now properly converts `for i in (range start end)` to standard C for loops: `for (int64_t i = start; i < end; i++)`.

**Before (broken):**
```c
static int64_t nl_sum_range(int64_t n) {
    int64_t sum = 0LL;
    /* unsupported expr type 14 */;  // Loop missing!
    return sum;  // Always returns 0
}
```

**After (working):**
```c
static int64_t nl_sum_range(int64_t n) {
    int64_t sum = 0LL;
    for (int64_t i = 0LL; i < n; i++) {
        sum = (sum + i);
    }
    return sum;  // Correctly computes sum
}
```

**Verification:**
- ✅ All 148 tests pass
- ✅ Created comprehensive `test_for_in_loops.nano` with 5 test functions
- ✅ Tests cover: simple ranges, nested loops, array iteration, custom start/end
- ✅ Runtime verification: sum_range(5)=10, nested_loops(3)=18, range_start_end()=35

**Files Modified:**
- `src/transpiler_iterative_v3_twopass.c`: Added AST_FOR case with proper C for-loop generation
- `tests/test_for_in_loops.nano`: Comprehensive regression test suite

**Status:** RESOLVED in v2.0.5

---

### 4. Missing Parentheses in Boolean Expression Transpilation (P0) - ✅ FIXED

**Severity:** P0 - Critical  
**Component:** Transpiler  
**Discovered:** 2026-01-14  
**Fixed:** 2026-01-14

**Description:**
When transpiling nested `and`/`or` expressions, the transpiler didn't add parentheses to preserve the intended precedence, causing C compiler warnings and potential logic errors.

**Fix Applied:**
Modified `src/transpiler_iterative_v3_twopass.c` line 702-703 to add `TOKEN_AND` and `TOKEN_OR` to the `needs_parens` check, ensuring all boolean operators are wrapped in parentheses when nested.

**Verification:**
- ✅ All 148 tests pass (including new `test_boolean_precedence.nano`)
- ✅ Zero `-Wlogical-op-parentheses` warnings (was 4)
- ✅ XOR simulation and complex logic tests work correctly
- ✅ Generated C code now has proper parentheses: `((a && b) || c)`

**Files Modified:**
- `src/transpiler_iterative_v3_twopass.c`: Added `TOKEN_AND` and `TOKEN_OR` to parentheses logic
- `tests/test_boolean_precedence.nano`: Comprehensive regression test

**Status:** RESOLVED in v2.0.5

---

## Self-Hosted Compiler Gaps

### 2. Self-Hosted Typechecker Misses Function Argument Type Errors

**Severity:** Medium  
**Component:** Self-hosted typechecker (src_nano/)  
**Status:** Test disabled in CI

**Description:**
The self-hosted compiler's typechecker doesn't properly validate function argument types, allowing type mismatches that the C reference compiler correctly rejects.

**Example:**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn main() -> int {
    let label: string = "oops"
    return (add label 10)  # Should fail: passing string to int parameter
}
```

**What happens:**
- C reference compiler: Correctly rejects with type error
- Self-hosted compiler: Compiles (incorrectly)

**Workaround:**
Test `test_function_arg_type_errors.nano` disabled in CI until fixed

**Priority:** Medium - self-hosted compiler is not primary yet

---

## Documentation

Last updated: 2026-01-14
Maintainer: NanoLang Team

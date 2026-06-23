# Known Issues in NanoLang Compiler

This document tracks known bugs and limitations in the NanoLang compiler that need to be addressed.

## Critical Bugs

### 1. Expression Statements Not Validated (Parser/Typechecker Gap) - FIXED

**Severity:** High  
**Component:** Parser/Typechecker  
**Discovered:** 2026-01-14
**Fixed:** 2026-02-21

**Description:**
The compiler allowed standalone expressions as statements without validating that they have side effects or are meaningful. This meant syntactically invalid code could compile successfully.

**Fix Applied:**
Added validation in `src/typechecker.c` `check_statement()`:
- `AST_PREFIX_OP` case: operator expressions used as statements now emit "EXPRESSION HAS NO EFFECT" error
- `default` case: literals, identifiers, field access, tuple literals, etc. used as statements now emit typed error messages (e.g., "This numeric literal is used as a statement...")
- `AST_MODULE_QUALIFIED_CALL` case: added as explicitly valid (function calls have side effects)

**Verification:**
- All 186 tests pass
- Negative test `tests/negative/type_errors/expression_statement_no_effect.nano` added
- Self-hosted compiler compiles cleanly (no false positives)

**Status:** RESOLVED

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

### 2. Self-Hosted Typechecker Misses Function Argument Type Errors - FIXED

**Severity:** Medium  
**Component:** Self-hosted typechecker (src_nano/)  
**Fixed:** 2026-02-21

**Description:**
The self-hosted compiler's typechecker didn't validate function argument types, allowing type mismatches that the C reference compiler correctly rejects.

**Fix Applied:**
Added argument type validation in `src_nano/typecheck.nano` `check_expr_node()` PNODE_CALL case:
- After resolving the function symbol, iterates arguments and compares each type against the parameter type using `types_equal`
- Emits diagnostic "E0010" with message like "Argument 1 to 'add': expected int, got string"

**Verification:**
- `tests/selfhost/test_function_arg_type_errors.nano` re-enabled in CI and passes
- All 12 selfhost tests pass
- 3-stage bootstrap validates successfully

**Status:** RESOLVED

---

## Documentation

Last updated: 2026-02-21
Maintainer: NanoLang Team

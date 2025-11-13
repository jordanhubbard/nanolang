# Stage 1.5 Implementation Issues

**Date:** November 13, 2025  
**Status:** Blocked - Transpiler Bugs Discovered

---

## Summary

Attempting to build Stage 1.5 (nanolang lexer + C rest) revealed critical bugs in the **transpiler** when generating C code from complex nanolang programs. The nanolang lexer itself works correctly (all shadow tests pass), but the generated C code has multiple compilation errors.

---

## Issues Discovered

### 1. Enum Redefinition (CRITICAL)
**Error:**
```
obj/lexer_nano.tmp.c:471:3: error: typedef redefinition with different types ('enum TokenType' vs 'enum TokenType')
  471 | } TokenType;
```

**Cause:**
- Lexer defines `enum TokenType { ... }` in nanolang
- Transpiler generates C enum definition
- C header `nanolang.h` already has `TokenType` enum
- Conflict when compiling generated C code

**Root Cause:**
- Transpiler doesn't check if enum is already defined in runtime
- Should either:
  - Not transpile enums that match runtime types
  - Use different names for nanolang-defined enums
  - Include #ifndef guards

**Impact:** ❌ **BLOCKER** - Cannot compile generated code

---

### 2. String Comparison Bug (CRITICAL)
**Error:**
```
obj/lexer_nano.tmp.c:511:12: warning: result of comparison against a string literal is unspecified (use an explicit string comparison function instead) [-Wstring-compare]
  511 |     if ((s == "extern")) {
```

**Cause:**
- Nanolang code: `if (== s "extern")`
- Transpiler generates: `if ((s == "extern"))`
- C requires: `if (strcmp(s, "extern") == 0)`

**Root Cause:**
- Transpiler doesn't properly handle string equality comparison
- `==` operator on strings should become `strcmp(s1, s2) == 0`
- Currently generates pointer comparison (undefined behavior)

**Impact:** ❌ **BLOCKER** - Generated code has undefined behavior

**Fix Required:**
In `transpiler.c`, when transpiling `AST_PREFIX_OP` with `TOKEN_EQ`:
```c
if (op == TOKEN_EQ) {
    // Check if operands are strings
    if (is_string_type(arg1_type) && is_string_type(arg2_type)) {
        sb_append(sb, "(strcmp(");
        transpile_expression(sb, args[0], env);
        sb_append(sb, ", ");
        transpile_expression(sb, args[1], env);
        sb_append(sb, ") == 0)");
        return;
    }
}
```

---

### 3. Struct vs. Typedef Inconsistency (CRITICAL)
**Error:**
```
obj/lexer_nano.tmp.c:731:33: error: passing 'struct Token' to parameter of incompatible type 'Token'
  731 |         list_token_push(tokens, tok);
```

**Cause:**
- C Runtime: `void list_token_push(List_token *list, Token token);`
- Generated C: Passes `struct Token tok`
- Type mismatch: `struct Token` vs `Token`

**Root Cause:**
- Transpiler generates `struct Token` for struct types
- Runtime uses `typedef struct { ... } Token;`
- Inconsistent naming between generated code and runtime

**Impact:** ❌ **BLOCKER** - Type mismatch prevents compilation

**Fix Required:**
- Either: Always use `Token` (typedef) instead of `struct Token`
- Or: Ensure runtime and generated code use same naming convention

---

### 4. Unused Variables (Minor)
**Warnings:**
```
Warning at line 271, column 5: Unused variable 'start'
Warning at line 299, column 5: Unused variable 'start'
Warning at line 520, column 5: Unused variable 'line_start'
Warning at line 524, column 9: Unused variable 'new_pos'
```

**Cause:**
- Variables declared but not used
- Likely leftover from refactoring

**Impact:** ⚠️ **WARNING** - Non-critical, but should be fixed

---

## Root Cause Analysis

The fundamental issue is that **the transpiler was not designed to handle self-hosting**. It has bugs that only appear when:

1. **Complex nanolang programs** are transpiled (not just simple examples)
2. **Runtime type names** conflict with user-defined types
3. **String operations** are used extensively
4. **Struct types** from runtime are used in nanolang code

The transpiler works fine for simple examples, but breaks down for compiler components.

---

## Implications for Self-Hosting

### Stage 1.5 is Blocked
- Cannot proceed with hybrid compiler until transpiler is fixed
- Need to fix critical bugs (#1, #2, #3) before retrying

### Stage 2 is Also Blocked
- Full self-hosting has same transpiler issues
- Even with union support, these bugs would prevent compilation

### What This Means
Self-hosting requires **fixing the transpiler** before we can proceed with either Stage 1.5 or Stage 2.

---

## Required Fixes

### Priority 1: Critical Bugs (Required for Stage 1.5)
1. **String Comparison**: `(== string1 string2)` → `strcmp(s1, s2) == 0`
2. **Enum Redefinition**: Don't re-generate runtime-provided enums
3. **Struct Naming**: Consistent `Token` vs `struct Token` usage

### Priority 2: Type System (Required for Stage 2)
4. **Type Checking for Operators**: Know when operands are strings
5. **Runtime Type Registry**: Track which types are provided by runtime
6. **Namespace Management**: Prevent user types from conflicting with runtime

### Priority 3: Code Quality
7. **Unused Variable Warnings**: Remove or use all declared variables
8. **Similar Function Name Warnings**: Suppress for intentionally similar names

---

## Recommended Next Steps

### Option A: Fix Transpiler First (2-3 weeks)
1. Implement proper string comparison transpilation
2. Add runtime type registry to prevent conflicts
3. Fix struct naming consistency
4. Test with lexer_main.nano
5. Retry Stage 1.5 build

**Pros:**
- Enables Stage 1.5
- Improves transpiler for all code
- Required for self-hosting anyway

**Cons:**
- Delays Stage 1.5 by 2-3 weeks
- Requires significant transpiler changes

### Option B: Workaround in Lexer (1 week)
1. Rename `TokenType` enum in lexer to `LexerTokenType`
2. Add explicit `strcmp` calls instead of `==` for strings
3. Adjust types to match runtime expectations
4. Retry Stage 1.5 build

**Pros:**
- Faster path to Stage 1.5
- Validates overall approach

**Cons:**
- Doesn't fix root cause
- Same issues for Stage 2
- Makes lexer less idiomatic

### Option C: Hybrid Approach (1-2 weeks)
1. Fix string comparison in transpiler (essential)
2. Workaround enum conflict in lexer (temporary)
3. Fix struct naming in transpiler (essential)
4. Complete Stage 1.5
5. Plan full transpiler fixes for Stage 2

**Pros:**
- Balances progress and quality
- Fixes most critical issues
- Enables Stage 1.5

**Cons:**
- Still has some technical debt
- Will need more fixes for Stage 2

---

## Recommendation

**Proceed with Option C: Hybrid Approach**

1. **Week 1:** Fix string comparison and struct naming in transpiler
   - These are critical bugs that affect all code
   - Relatively localized fixes
   - High impact

2. **Week 2:** Workaround enum conflict, complete Stage 1.5
   - Rename TokenType in lexer temporarily
   - Build and test Stage 1.5 hybrid compiler
   - Validate nanolang lexer works in production

3. **Future:** Plan comprehensive transpiler improvements
   - Type registry for runtime types
   - Better namespace management
   - Full Stage 2 support

This approach gets us to Stage 1.5 quickly while fixing the most critical transpiler bugs.

---

## Current Status

- ❌ **Stage 1.5:** BLOCKED - Transpiler bugs prevent compilation
- ❌ **Stage 2:** BLOCKED - Same transpiler bugs
- ✅ **Nanolang Lexer:** Works correctly (shadow tests pass)
- ✅ **Stage 0:** Still fully functional

---

**Last Updated:** 2025-11-13  
**Next Action:** Decide on Option A, B, or C


# Lexer Enum Access Limitation

## Issue

When updating `src_nano/lexer_v2.nano` to use enum variants like `TokenType.FN`, shadow tests fail because the interpreter doesn't have access to enum definitions during shadow test execution.

## Root Cause

1. **Type Checking Phase**: Enum definitions are registered in the environment during typechecking
2. **Shadow Test Execution**: Shadow tests run in the interpreter with a separate environment
3. **Missing Registration**: The interpreter's environment doesn't include enum definitions
4. **Eval Handling**: `AST_ENUM_DEF` in `eval_statement` just returns void (line 1787 in eval.c)

```c
case AST_ENUM_DEF:
case AST_FUNCTION:
case AST_SHADOW:
    /* Enum definitions are handled at program level */
    /* Just return void if encountered during execution */
    return create_void();
```

## Test Case

```nano
enum TokenType {
    IDENTIFIER = 4,
    FN = 19
}

fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") { 
        return TokenType.FN  /* Works in transpiled code */
    } else {}
    return TokenType.IDENTIFIER
}

shadow classify_keyword {
    /* This fails: TokenType not in interpreter environment */
    assert (== (classify_keyword "fn") TokenType.FN)  
}
```

**Error**: Assertion fails because `TokenType.FN` evaluates to something other than 19 in the interpreter.

## Current Solution

**Hybrid Approach**: Use enum variants in function bodies (which transpile correctly), but use literals in shadow tests (which run in interpreter):

```nano
fn classify_keyword(word: string) -> int {
    /* ✓ TokenType.FN works here - gets transpiled to C */
    if (strings_equal word "fn") { return TokenType.FN } else {}
    return TokenType.IDENTIFIER
}

shadow classify_keyword {
    /* ✓ Literal works in shadow test - interpreter evaluates it */
    assert (== (classify_keyword "fn") 19)  /* TokenType.FN */
}
```

## Long-Term Fix

To fully support enum access in shadow tests, we would need to:

1. **Update Interpreter Initialization**: Before running shadow tests, process all enum definitions
2. **Register Enums**: Call `env_register_enum()` in the interpreter's environment
3. **Handle AST_ENUM_DEF**: Properly evaluate enum definitions in `eval_statement`

### Proposed Implementation

```c
/* In eval.c */
case AST_ENUM_DEF: {
    /* Register enum in interpreter environment */
    const char *enum_name = stmt->as.enum_def.name;
    int variant_count = stmt->as.enum_def.variant_count;
    char **variant_names = stmt->as.enum_def.variant_names;
    int *variant_values = stmt->as.enum_def.variant_values;
    
    env_register_enum(env, enum_name, variant_names, variant_values, variant_count);
    return create_void();
}
```

**Effort**: 1-2 hours to implement properly

**Priority**: Low - current workaround is acceptable

## Impact

### Current Impact
- **Minor inconvenience**: Shadow tests use magic numbers with comments
- **Function bodies are clean**: Enum variants work in actual code
- **Transpiled code is correct**: C code uses proper enum values

### If Fixed
- **Consistent syntax**: Enum variants everywhere
- **Better test readability**: No magic numbers in tests
- **Type safety**: Enums work consistently

## Workaround Quality

**Rating**: ⭐⭐⭐⭐ (4/5 stars)

**Pros**:
- Simple to implement
- Clear comments explain intent
- Transpiled code uses proper enums
- No performance impact

**Cons**:
- Shadow tests less readable
- Requires discipline to keep comments in sync

## Recommendation

**Keep current workaround** for now:
- Enum variants in function bodies ✓
- Magic numbers + comments in shadow tests ✓
- Implement proper fix when time permits

## Example: lexer_v2.nano

```nano
/* ============================================
 * Current approach (works well):
 * ============================================ */

fn classify_keyword(word: string) -> int {
    /* Use clear enum names */
    if (strings_equal word "fn") { return TokenType.FN } else {}
    if (strings_equal word "let") { return TokenType.LET } else {}
    return TokenType.IDENTIFIER
}

shadow classify_keyword {
    /* Note: Using literals because TokenType conflicts with runtime */
    assert (== (classify_keyword "fn") 19)  /* TokenType.FN */
    assert (== (classify_keyword "let") 20)  /* TokenType.LET */
    assert (== (classify_keyword "hello") 4)  /* TokenType.IDENTIFIER */
}
```

**Result**: Clean function code, working shadow tests, proper transpilation!

---

*Status: Documented workaround*  
*Priority: Low (fix when improving interpreter)*  
*Date: November 14, 2025*


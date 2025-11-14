# Interpreter If/Else Bug

## Issue

The interpreter does not correctly handle the pattern `if (cond) { return value } else {}` when the else block is empty.

## Root Cause

When evaluating an if/else statement in the interpreter where:
1. The if block contains a `return` statement
2. The else block is empty `else {}`  
3. The condition is true

The interpreter appears to not properly handle the return, causing execution to fall through to subsequent code.

## Test Case

```nano
enum TokenType {
    IDENTIFIER = 4,
    FN = 19
}

extern fn str_equals(s1: string, s2: string) -> bool

fn strings_equal(s1: string, s2: string) -> bool {
    return (str_equals s1 s2)
}

fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") { return TokenType.FN } else {}  /* ❌ Bug! */
    return TokenType.IDENTIFIER
}

shadow classify_keyword {
    /* This fails: returns 4 instead of 19 */
    assert (== (classify_keyword "fn") 19)
}
```

**Expected**: Returns `19` (TokenType.FN)  
**Actual**: Returns `4` (TokenType.IDENTIFIER)

## Workaround

Use proper if/else structure with explicit return in else block:

```nano
fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") {
        return TokenType.FN
    } else {
        return TokenType.IDENTIFIER
    }
}
```

Or use nested if/else:

```nano
fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") {
        return TokenType.FN
    } else if (strings_equal word "let") {
        return TokenType.LET
    } else {
        return TokenType.IDENTIFIER
    }
}
```

## Impact

### Current Impact
- **lexer_v2.nano cannot be compiled** with enum values in classify_keyword
- **Workaround required**: Use numeric literals instead of enum names
- **Affects**: Any function with multiple early-return if statements

### Files Affected
- `src_nano/lexer_v2.nano` - classify_keyword function

## Technical Details

The pattern `if (cond) { return value } else {}` is valid nanolang syntax:
- **Type checker**: ✅ Passes (both branches present)
- **Transpiler**: ✅ Generates correct C code
- **Interpreter**: ❌ Incorrectly evaluates

## Example: Working vs. Broken

### ❌ Broken Pattern
```nano
fn test() -> int {
    if (== 1 1) { return 10 } else {}
    return 20
}
/* Returns: 20 (wrong!) */
```

### ✅ Working Pattern
```nano
fn test() -> int {
    if (== 1 1) {
        return 10
    } else {
        return 20
    }
}
/* Returns: 10 (correct!) */
```

## Root Cause Analysis

Likely issue in `eval.c` when handling `AST_IF` with return statements. The interpreter may be:
1. Evaluating the if body correctly
2. Seeing the return value
3. BUT then continuing to evaluate the else block
4. OR not properly propagating the return value

## Fix Required

In `src/eval.c`, the `AST_IF` handling needs to:
1. Check if the if-body returned a value
2. If so, return immediately without evaluating else block
3. Properly handle empty else blocks

### Estimated Effort
1-2 hours to diagnose and fix in `eval.c`

### Priority
**Medium** - Affects self-hosted code development

## Current Status

**Workaround in place**: lexer_v2.nano uses integer literals instead of enum values

**Long-term fix**: Update interpreter to handle this pattern correctly

---

*Discovered: November 14, 2025*  
*Status: Documented, workaround applied*  
*Priority: Medium (fix when improving interpreter)*


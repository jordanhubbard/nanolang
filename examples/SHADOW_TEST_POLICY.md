# Shadow Test Policy for Examples

## ⚠️ Shadow Tests Are MANDATORY in Examples ⚠️

**All functions in example code MUST have shadow tests.**

This is not optional. Shadow tests are a core design principle of NanoLang and apply to **all code**, including examples.

## Why Examples Need Shadow Tests

### 1. **Examples Are Teaching Tools**

Examples show users how to write NanoLang code. If examples don't have shadow tests, users will think they're optional.

**Bad Example (Don't do this):**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}
/* NO SHADOW TEST - User learns bad habits! */
```

**Good Example (Do this):**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add -1 1) 0)
}
/* Shows proper NanoLang style! */
```

### 2. **Examples Demonstrate Language Features**

Shadow tests in examples show:
- How to test edge cases
- How to handle boundary conditions  
- How to document expected behavior
- Best practices for test coverage

### 3. **Examples Are Validated Code**

Shadow tests ensure example code actually works. Without them:
- Examples might have bugs
- Refactoring could break examples silently
- Users copy broken code

### 4. **LLM Training Data**

Examples are used to train LLMs (like this one) on NanoLang. If examples lack shadow tests, LLMs will learn to omit them.

## Current Status

**⚠️ Technical Debt Alert ⚠️**

Many example files currently show "missing shadow test" warnings. This is **technical debt**, not the intended design.

**These warnings are NOT false positives. They indicate missing tests that should be added.**

## Action Items

### For New Examples
**ALWAYS include shadow tests for every function.**

### For Existing Examples
Gradually add shadow tests:
1. Start with most-used examples
2. Add tests when modifying examples
3. Treat "missing shadow test" as a TODO item, not a false positive

### For Contributors
When you see:
```
Warning: Function 'my_function' is missing a shadow test
```

This means: **"Please add a shadow test for my_function"** - not "ignore this warning".

## Exceptions

**ONLY these functions are exempt from shadow tests:**

1. **`extern` functions** (C FFI) - tested in C
2. **`main` function** when it uses `extern` functions that can't be mocked

**All other functions need shadow tests. No exceptions.**

## Implementation Notes

### For Utility Functions

Even simple helper functions need tests:

```nano
fn square(x: int) -> int {
    return (* x x)
}

shadow square {
    assert (== (square 0) 0)
    assert (== (square 5) 25)
    assert (== (square -3) 9)
}
```

### For Graphics/SDL Functions

Functions that wrap SDL need tests:

```nano
fn draw_circle(x: int, y: int, radius: int) -> bool {
    /* SDL drawing code */
    return true
}

shadow draw_circle {
    /* Test returns true for valid input */
    assert (draw_circle 100 100 50)
    /* Test handles edge cases */
    assert (draw_circle 0 0 1)
}
```

## Summary

**Shadow tests are mandatory in examples.**

This is not negotiable - it's part of NanoLang's design philosophy. Examples without shadow tests are incomplete and should be treated as technical debt.

When writing or reviewing example code, always ask: **"Does every function have a shadow test?"**

If not, add them.


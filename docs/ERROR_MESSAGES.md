# NanoLang Error Messages Guide

Complete guide to understanding and fixing NanoLang compiler errors.

## Table of Contents

1. [Reading Error Messages](#reading-error-messages)
2. [Type Errors](#type-errors)
3. [Syntax Errors](#syntax-errors)
4. [Shadow Test Errors](#shadow-test-errors)
5. [Scope and Binding Errors](#scope-and-binding-errors)
6. [Control Flow Errors](#control-flow-errors)
7. [Module and Import Errors](#module-and-import-errors)
8. [Common Mistakes](#common-mistakes)

---

## Reading Error Messages

NanoLang error messages follow this format:

```
error: <error type>
  --> <file>:<line>:<column>
   |
<line#> | <source code line>
   | <visual indicator>
   |
   = help: <helpful message>
```

### Example Error

```nano
let x: int = "hello"
```

**Error:**
```
error: type mismatch
  --> example.nano:1:14
   |
 1 | let x: int = "hello"
   |              ^^^^^^^ expected `int`, found `string`
   |
   = help: Did you mean to use `string` type?
```

### Parts of an Error

- **error type**: Category of error (type mismatch, undefined symbol, etc.)
- **location**: File, line number, column number
- **source context**: The problematic line of code
- **visual indicator**: Points to the exact problem
- **help message**: Suggestions for fixing (when available)

---

## Type Errors

### Type Mismatch

**Error:**
```
error: type mismatch
  --> code.nano:5:18
   |
 5 | let answer: int = 3.14
   |                  ^^^^ expected `int`, found `float`
```

**Fix:**
```nano
# Option 1: Use correct type annotation
let answer: float = 3.14

# Option 2: Cast to int (explicit conversion)
let answer: int = (cast_int 3.14)
```

### Incompatible Operation Types

**Error:**
```
error: type error in binary operation
  --> code.nano:3:18
   |
 3 | let sum: int = (+ 5 "hello")
   |                  ^ cannot add `int` and `string`
   |
   = help: Both operands of `+` must have the same type (int or float)
```

**Fix:**
```nano
# Convert string to int first
let sum: int = (+ 5 (string_to_int "10"))

# Or use string concatenation
let text: string = (str_concat (int_to_string 5) "hello")
```

### Return Type Mismatch

**Error:**
```
error: return type mismatch
  --> code.nano:7:12
   |
 6 | fn get_name() -> string {
 7 |     return 42
   |            ^^ expected `string`, found `int`
```

**Fix:**
```nano
fn get_name() -> string {
    return "Alice"  # Return correct type
}

# Or change function signature
fn get_age() -> int {
    return 42
}
```

### Comparison of Incompatible Types

**Error:**
```
error: type error in comparison
  --> code.nano:2:9
   |
 2 | if (== 5 "5") {
   |        ^^^^^^ cannot compare `int` and `string`
```

**Fix:**
```nano
# Convert types to match
if (== 5 (string_to_int "5")) {
    # ...
}

# Or compare as strings
if (str_equals (int_to_string 5) "5") {
    # ...
}
```

---

## Syntax Errors

### Missing Return Type

**Error:**
```
error: missing return type
  --> code.nano:1:13
   |
 1 | fn compute(x: int) {
   |                    ^ expected `->`
   |
   = help: All functions must declare a return type, use `-> void` if no return value
```

**Fix:**
```nano
fn compute(x: int) -> int {
    return (* x 2)
}

# Or for functions with no return value
fn print_hello() -> void {
    (println "Hello")
}
```

### Missing Type Annotation

**Error:**
```
error: missing type annotation
  --> code.nano:1:9
   |
 1 | let x = 42
   |         ^^ type annotation required
   |
   = help: Add `: type` after variable name: let x: int = 42
```

**Fix:**
```nano
let x: int = 42
let name: string = "Alice"
let flag: bool = true
```

### Unbalanced Parentheses

**Error:**
```
error: syntax error: unbalanced parentheses
  --> code.nano:2:22
   |
 2 | let result: int = (+ 1 2
   |                          ^ expected `)`
```

**Fix:**
```nano
let result: int = (+ 1 2)  # Add closing paren
```

### Unbalanced Grouping in Infix Expressions

**Error:**
```
error: syntax error: unbalanced parentheses
  --> code.nano:1:30
   |
 1 | let sum: int = a * (b + c
   |                              ^ expected `)`
```

**Fix:**
```nano
let sum: int = a * (b + c)  # Close the grouping parentheses
```

**Note:** NanoLang supports both prefix `(+ 1 2)` and infix `1 + 2` notation. All infix operators have equal precedence and evaluate left-to-right, so use parentheses to control grouping: `a * (b + c)`.

---

## Shadow Test Errors

### Missing Shadow Test

**Error:**
```
error: missing shadow test
  --> code.nano:3:1
   |
 1 | fn double(x: int) -> int {
 2 |     return (* x 2)
 3 | }
   | ^ function `double` requires a shadow test
   |
   = help: Add a shadow block: shadow double { assert ... }
```

**Fix:**
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
    assert (== (double 0) 0)
    assert (== (double -3) -6)
}
```

### Shadow Test Assertion Failed

**Error:**
```
error: shadow test failed
  --> code.nano:8:5
   |
 8 |     assert (== (add 2 3) 6)
   |            ^^^^^^^^^^^^^^^^ assertion failed
   |
   = note: Expected (add 2 3) to equal 6, but got 5
   = help: Check the implementation of `add` function
```

**Fix:**
```nano
# Either fix the test:
shadow add {
    assert (== (add 2 3) 5)  # Correct expected value
}

# Or fix the implementation:
fn add(a: int, b: int) -> int {
    return (+ a b)  # Was: return a (missing +)
}
```

---

## Scope and Binding Errors

### Undefined Variable

**Error:**
```
error: undefined variable
  --> code.nano:5:18
   |
 5 | let sum: int = (+ x y)
   |                   ^ undefined variable `x`
```

**Fix:**
```nano
fn calculate() -> int {
    let x: int = 5  # Define before use
    let y: int = 3
    let sum: int = (+ x y)
    return sum
}
```

### Undefined Function

**Error:**
```
error: undefined function
  --> code.nano:2:18
   |
 2 | let result = (compute 42)
   |               ^^^^^^^ function `compute` not found
   |
   = help: Did you forget to define or import this function?
```

**Fix:**
```nano
# Option 1: Define the function
fn compute(x: int) -> int {
    return (* x 2)
}

# Option 2: Import from module
from "./math.nano" import compute
```

### Variable Used Before Declaration

**Error:**
```
error: use before declaration
  --> code.nano:2:14
   |
 2 | (println x)
   |          ^ variable `x` used before declaration
 3 | let x: int = 42
```

**Fix:**
```nano
let x: int = 42
(println x)  # Use after declaration
```

### Assignment to Immutable Variable

**Error:**
```
error: cannot assign to immutable variable
  --> code.nano:3:5
   |
 2 | let x: int = 10
 3 | set x 20
   |     ^ variable `x` is immutable
   |
   = help: Declare as mutable: let mut x: int = 10
```

**Fix:**
```nano
let mut x: int = 10  # Add `mut` keyword
set x 20             # Now assignment is allowed
```

---

## Control Flow Errors

### If Without Else

**Error:**
```
error: if expression requires else branch
  --> code.nano:2:1
   |
 2 | if (> x 0) {
 3 |     return 1
 4 | }
   | ^ missing else branch
   |
   = help: if expressions must have both branches: if { ... } else { ... }
```

**Fix:**
```nano
if (> x 0) {
    return 1
} else {
    return 0
}
```

### Missing Return on All Paths

**Error:**
```
error: not all code paths return a value
  --> code.nano:4:1
   |
 1 | fn get_sign(x: int) -> int {
 2 |     if (> x 0) {
 3 |         return 1
 4 |     }
   |     ^ missing return for this branch
```

**Fix:**
```nano
fn get_sign(x: int) -> int {
    if (> x 0) {
        return 1
    } else {
        if (< x 0) {
            return -1
        } else {
            return 0
        }
    }
}  # All paths now return
```

### Condition Must Be Bool

**Error:**
```
error: condition must have type `bool`
  --> code.nano:2:5
   |
 2 | if (x) {
   |     ^ expected `bool`, found `int`
```

**Fix:**
```nano
if (!= x 0) {  # Explicit comparison
    # ...
}
```

---

## Module and Import Errors

### Module Not Found

**Error:**
```
error: module not found
  --> code.nano:1:6
   |
 1 | from "./missing.nano" import func
   |      ^^^^^^^^^^^^^^^^ file not found
   |
   = help: Check the file path and ensure the module exists
```

**Fix:**
```nano
# Check file exists:
# ls ./missing.nano

# Correct the path:
from "./existing.nano" import func
```

### Symbol Not Exported

**Error:**
```
error: symbol not exported
  --> code.nano:1:35
   |
 1 | from "./utils.nano" import internal_helper
   |                            ^^^^^^^^^^^^^^^ symbol `internal_helper` is private
   |
   = help: Only `pub` symbols can be imported. Check the module definition.
```

**Fix:**
```nano
# In utils.nano: Mark as public
pub fn internal_helper() -> int {
    return 42
}
```

### Circular Import

**Error:**
```
error: circular import detected
  --> code.nano:1:1
   |
 1 | import "./b.nano"
   | ^^^^^^^^^^^^^^^^^ circular dependency
   |
   = note: a.nano -> b.nano -> a.nano
   = help: Refactor shared code into a third module
```

**Fix:**
```nano
# Create shared.nano with common code
# a.nano and b.nano both import shared.nano
```

---

## Common Mistakes

### Note: Both Notations Work

Both prefix and infix are valid:
```nano
let x: int = (+ 5 3)   # Prefix notation
let x: int = 5 + 3     # Infix notation (also valid!)
```

**Watch out for precedence!** All infix operators have equal precedence (left-to-right, no PEMDAS):
```nano
# This evaluates as (5 + 3) * 2, NOT 5 + (3 * 2)
let x: int = 5 + 3 * 2     # Result: 16

# Use parentheses to get the grouping you want:
let x: int = 5 + (3 * 2)   # Result: 11
```

### Mistake: Missing Parentheses in Function Call

❌ **Wrong:**
```nano
(println add 2 3)  # Tries to print the function `add`
```

✅ **Correct:**
```nano
(println (add 2 3))  # Call add first, then print result
```

### Mistake: Forgetting mut for Mutable Variables

❌ **Wrong:**
```nano
let counter: int = 0
set counter (+ counter 1)  # Error: counter is immutable
```

✅ **Correct:**
```nano
let mut counter: int = 0
set counter (+ counter 1)  # OK: counter is mutable
```

### Mistake: Using == for String Comparison

❌ **Wrong:**
```nano
if (== name "Alice") {  # May not work as expected
    # ...
}
```

✅ **Correct:**
```nano
if (str_equals name "Alice") {  # Use str_equals
    # ...
}
```

---

## Error Recovery Tips

1. **Read the error message carefully** - It usually tells you exactly what's wrong
2. **Check the line and column** - The error is at or near this position
3. **Look at the help message** - Suggestions for fixing
4. **Start from the first error** - Later errors may be cascading
5. **Check parentheses balance** - Use editor matching
6. **Verify types match** - Add explicit type annotations
7. **Run shadow tests frequently** - Catch errors early

---

## Getting Help

If you're stuck:

1. **Check examples/** - Find similar code that works
2. **Read SPECIFICATION.md** - Full language reference
3. **Search GitHub issues** - Someone may have hit the same error
4. **Ask in discussions** - Community can help
5. **Report a bug** - If error message is unclear or wrong

---

## Related Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Tutorial
- [SPECIFICATION.md](SPECIFICATION.md) - Language reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Syntax cheat sheet
- [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) - Debugging techniques

---

**Last Updated:** January 25, 2026
**Status:** Complete
**Version:** 0.2.0+

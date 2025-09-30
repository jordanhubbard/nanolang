# nanolang Language Specification v0.1

## Table of Contents

1. [Introduction](#introduction)
2. [Lexical Structure](#lexical-structure)
3. [Types](#types)
4. [Expressions](#expressions)
5. [Statements](#statements)
6. [Functions](#functions)
7. [Shadow-Tests](#shadow-tests)
8. [Semantics](#semantics)
9. [Compilation Model](#compilation-model)

## 1. Introduction

nanolang is a minimal, statically-typed programming language designed for clarity and LLM-friendliness. Its design prioritizes:

- **Unambiguity**: One syntax for each semantic concept
- **Explicitness**: No implicit conversions or hidden behavior
- **Testability**: Mandatory shadow-tests for all functions
- **Simplicity**: Minimal feature set with clear semantics

## 2. Lexical Structure

### 2.1 Comments

```nano
# This is a single-line comment
# There are no multi-line comments
```

### 2.2 Identifiers

Identifiers must start with a letter or underscore, followed by letters, digits, or underscores:

```
identifier = (letter | "_") { letter | digit | "_" }
```

Examples: `x`, `my_var`, `count2`, `_internal`

### 2.3 Keywords

Reserved keywords that cannot be used as identifiers:

```
fn       let      mut      set      if       else
while    for      in       return   assert   shadow
int      float    bool     string   void     
true     false    print    and      or       not
```

### 2.4 Literals

**Integer Literals**: Sequence of digits, optionally with leading `-`
```nano
42
-17
0
```

**Float Literals**: Digits with decimal point
```nano
3.14
-0.5
2.0
```

**String Literals**: UTF-8 text in double quotes
```nano
"Hello, World!"
"nanolang"
""
```

**Boolean Literals**: 
```nano
true
false
```

### 2.5 Operators and Delimiters

```
(  )  {  }  ,  :  =  ->
```

### 2.6 Whitespace

Whitespace (spaces, tabs, newlines) separates tokens but is otherwise insignificant.

## 3. Types

### 3.1 Built-in Types

| Type     | Description                    | Example Values    |
|----------|--------------------------------|-------------------|
| `int`    | 64-bit signed integer          | `42`, `-17`, `0`  |
| `float`  | 64-bit floating point          | `3.14`, `-0.5`    |
| `bool`   | Boolean value                  | `true`, `false`   |
| `string` | UTF-8 encoded text             | `"hello"`         |
| `void`   | Absence of value (return only) | N/A               |

### 3.2 Type Annotations

All variables and function parameters must have explicit type annotations:

```nano
let x: int = 42
let name: string = "Alice"

fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

### 3.3 Type Checking

nanolang is statically typed. All type errors are caught at compile time:

```nano
let x: int = 42
let y: string = "hello"
# let z: int = y        # ERROR: Type mismatch
# let w: int = (+ x y)  # ERROR: Cannot add int and string
```

## 4. Expressions

### 4.1 Literals

Literals evaluate to their corresponding values:

```nano
42          # int
3.14        # float
"hello"     # string
true        # bool
```

### 4.2 Variables

Identifiers evaluate to the value of the named variable:

```nano
let x: int = 42
let y: int = x  # y = 42
```

### 4.3 Prefix Operations

All operations use prefix notation (S-expressions) to eliminate ambiguity:

```nano
(+ 2 3)              # Addition: 2 + 3 = 5
(* (+ 2 3) 4)        # Multiplication: (2 + 3) * 4 = 20
(== x 5)             # Comparison: x == 5
(and (> x 0) (< x 10))  # Logical: x > 0 && x < 10
```

### 4.4 Arithmetic Operations

| Operator | Description    | Type Signature     |
|----------|----------------|--------------------|
| `+`      | Addition       | `(int, int) -> int` or `(float, float) -> float` |
| `-`      | Subtraction    | `(int, int) -> int` or `(float, float) -> float` |
| `*`      | Multiplication | `(int, int) -> int` or `(float, float) -> float` |
| `/`      | Division       | `(int, int) -> int` or `(float, float) -> float` |
| `%`      | Modulo         | `(int, int) -> int` |

### 4.5 Comparison Operations

All comparison operations return `bool`:

| Operator | Description      | Type Signature     |
|----------|------------------|--------------------|
| `==`     | Equal            | `(T, T) -> bool`   |
| `!=`     | Not equal        | `(T, T) -> bool`   |
| `<`      | Less than        | `(int, int) -> bool` or `(float, float) -> bool` |
| `<=`     | Less or equal    | `(int, int) -> bool` or `(float, float) -> bool` |
| `>`      | Greater than     | `(int, int) -> bool` or `(float, float) -> bool` |
| `>=`     | Greater or equal | `(int, int) -> bool` or `(float, float) -> bool` |

### 4.6 Logical Operations

| Operator | Description  | Type Signature          |
|----------|--------------|-------------------------|
| `and`    | Logical AND  | `(bool, bool) -> bool`  |
| `or`     | Logical OR   | `(bool, bool) -> bool`  |
| `not`    | Logical NOT  | `(bool) -> bool`        |

### 4.7 Function Calls

Functions are called using prefix notation:

```nano
(add 2 3)
(multiply (add 1 2) 4)
(is_prime 17)
```

### 4.8 If Expressions

`if` is an expression that returns a value:

```nano
let x: int = if (> a 0) {
    42
} else {
    -1
}
```

Both branches must return the same type. Both branches are required (no optional `else`).

### 4.9 Evaluation Order

Expressions are evaluated left-to-right within each prefix operation:

```nano
(+ (f x) (g y))  # f(x) is evaluated before g(y)
```

## 5. Statements

### 5.1 Variable Declaration

Variables are declared with `let`:

```nano
let x: int = 42
let mut counter: int = 0  # Mutable variable
```

Variables are immutable by default. Use `mut` for mutable variables.

### 5.2 Assignment

Only mutable variables can be reassigned using `set`:

```nano
let mut x: int = 0
set x (+ x 1)
# let y: int = 0
# set y 1  # ERROR: y is not mutable
```

### 5.3 While Loop

```nano
while condition {
    # body
}
```

The condition must be a `bool` expression. The loop executes while the condition is `true`.

```nano
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

### 5.4 For Loop

```nano
for identifier in expression {
    # body
}
```

The `for` loop is syntactic sugar for iterating over a range:

```nano
for i in (range 0 10) {
    print i
}

# Equivalent to:
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

### 5.5 Return Statement

```nano
return expression
```

Returns a value from a function. The expression type must match the function's return type.

### 5.6 Expression Statement

Any expression can be used as a statement:

```nano
print "hello"
(add 2 3)  # Result is discarded
```

## 6. Functions

### 6.1 Function Definition

```nano
fn name(param1: type1, param2: type2) -> return_type {
    # body
}
```

Functions must:
1. Have explicit parameter types
2. Have an explicit return type
3. Return a value if return type is not `void`
4. Have a corresponding shadow-test

### 6.2 Parameters

Parameters are passed by value. They are immutable within the function:

```nano
fn increment(x: int) -> int {
    # set x (+ x 1)  # ERROR: Parameters are immutable
    return (+ x 1)
}
```

### 6.3 Return Type

Functions must specify a return type:

- Non-`void` functions must return a value on all code paths
- `void` functions may use `return` without a value or omit `return`

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
}  # OK: All paths return a value

fn greet() -> void {
    print "Hello"
    # No return needed
}
```

## 7. Shadow-Tests

### 7.1 Purpose

Shadow-tests are mandatory tests that:
- Run during compilation
- Fail compilation if any assertion fails
- Are stripped from production builds
- Document expected behavior
- Ensure correctness

### 7.2 Syntax

```nano
shadow function_name {
    # test body with assertions
}
```

Each function must have exactly one shadow-test block. The shadow-test is defined after the function it tests.

### 7.3 Assertions

Shadow-tests use `assert` to verify behavior:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -5 3) -2)
}
```

### 7.4 Assertion Semantics

`assert` takes a boolean expression:
- If `true`: Test passes, continue
- If `false`: Compilation fails with error message

### 7.5 Coverage Requirements

Shadow-tests should cover:
- Normal cases
- Edge cases (0, negative numbers, empty strings, etc.)
- Boundary conditions
- Error conditions (where applicable)

### 7.6 Execution Order

Shadow-tests run immediately after their function is defined during compilation. This ensures that functions are tested as soon as they're available.

## 8. Semantics

### 8.1 Static Scoping

nanolang uses static (lexical) scoping. Variables are resolved at compile time:

```nano
let x: int = 1

fn f() -> int {
    return x  # Refers to the global x
}

fn g() -> int {
    let x: int = 2
    return (f)  # Returns 1, not 2
}

shadow f {
    assert (== (f) 1)
}

shadow g {
    assert (== (g) 1)
}
```

### 8.2 Variable Shadowing

Inner scopes can shadow outer variables:

```nano
let x: int = 1
{
    let x: int = 2  # Shadows outer x
    print x         # Prints 2
}
print x            # Prints 1
```

### 8.3 Type Equivalence

Types are equivalent if they have the same name. There is no structural typing:

```nano
# int and int are the same type
# int and float are different types
```

### 8.4 No Implicit Conversions

All type conversions must be explicit:

```nano
let x: int = 42
# let y: float = x  # ERROR: No implicit conversion
```

### 8.5 Short-Circuit Evaluation

Logical operators `and` and `or` use short-circuit evaluation:

```nano
(and false (expensive_computation))  # expensive_computation not called
(or true (expensive_computation))    # expensive_computation not called
```

## 9. Compilation Model

### 9.1 Phases

1. **Lexing**: Source text → Tokens
2. **Parsing**: Tokens → AST
3. **Type Checking**: Verify types, shadow-tests, return paths
4. **Shadow-Test Execution**: Run all shadow-tests
5. **Transpilation**: AST → C code
6. **C Compilation**: C code → Native binary

### 9.2 Shadow-Test Compilation

Shadow-tests are:
1. Extracted during parsing
2. Checked for type correctness
3. Executed during compilation
4. Removed from the final output

If any shadow-test fails, compilation stops with an error.

### 9.3 C Transpilation

nanolang compiles to clean, readable C:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

Transpiles to:

```c
int64_t add(int64_t a, int64_t b) {
    return a + b;
}
```

### 9.4 Entry Point

Programs must define a `main` function:

```nano
fn main() -> int {
    # program logic
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### 9.5 Built-in Functions

Built-in functions are provided by the runtime:

- `print`: Output to stdout (polymorphic over printable types)
- `assert`: Runtime assertion (used in shadow-tests)
- `range`: Generate integer range for iteration

## 10. Example Programs

### 10.1 Fibonacci

```nano
fn fib(n: int) -> int {
    if (<= n 1) {
        return n
    } else {
        return (+ (fib (- n 1)) (fib (- n 2)))
    }
}

shadow fib {
    assert (== (fib 0) 0)
    assert (== (fib 1) 1)
    assert (== (fib 2) 1)
    assert (== (fib 5) 5)
    assert (== (fib 10) 55)
}

fn main() -> int {
    print (fib 10)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### 10.2 Prime Number Checker

```nano
fn is_prime(n: int) -> bool {
    if (< n 2) {
        return false
    }
    let mut i: int = 2
    while (< i n) {
        if (== (% n i) 0) {
            return false
        }
        set i (+ i 1)
    }
    return true
}

shadow is_prime {
    assert (== (is_prime 1) false)
    assert (== (is_prime 2) true)
    assert (== (is_prime 3) true)
    assert (== (is_prime 4) false)
    assert (== (is_prime 17) true)
    assert (== (is_prime 100) false)
}

fn main() -> int {
    for n in (range 1 20) {
        if (is_prime n) {
            print n
        }
    }
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## 11. Design Rationale

### 11.1 Why Prefix Notation?

Traditional infix notation requires memorizing operator precedence:

```
a + b * c    # Is this (a + b) * c or a + (b * c)?
```

Prefix notation makes nesting explicit:

```nano
(+ a (* b c))  # Unambiguous
```

This is especially valuable for LLMs, which may not consistently apply precedence rules.

### 11.2 Why Mandatory Shadow-Tests?

1. **Quality**: Untested code doesn't compile
2. **Documentation**: Tests show how to use functions
3. **Confidence**: Tests prove correctness
4. **LLM-friendly**: Forces test generation

### 11.3 Why Static Typing?

Static typing catches errors at compile time:
- No type errors at runtime
- Better tooling support
- Clearer semantics
- LLM-friendly (types guide generation)

### 11.4 Why C Transpilation?

- **Performance**: Native speed
- **Portability**: C runs everywhere
- **Interop**: Easy FFI
- **Self-hosting**: nanolang can eventually compile itself
- **Tooling**: Leverage mature C ecosystem

## 12. Future Extensions

Potential future additions (not in v0.1):

- Arrays/lists
- Structs/records
- Generics/templates
- Modules/imports
- Error handling
- Memory management primitives
- Standard library expansion

All extensions must maintain the core principles: minimal, unambiguous, LLM-friendly, and test-driven.

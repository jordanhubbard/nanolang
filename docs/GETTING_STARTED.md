# Getting Started with nanolang

Welcome to nanolang! This guide will help you understand and start writing programs in this minimal, LLM-friendly language.

## What is nanolang?

nanolang is a programming language designed with these goals:

- **Simple**: Small set of features, easy to learn
- **Clear**: Every construct has exactly one meaning
- **Safe**: Static typing catches errors at compile time
- **Tested**: All code must have shadow-tests
- **LLM-friendly**: Syntax optimized for AI code generation

## Your First Program

Let's start with the classic "Hello, World!":

```nano
fn main() -> int {
    print "Hello, World!"
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### Breaking it down:

1. `fn main() -> int` - Define a function named `main` that returns an `int`
2. `print "Hello, World!"` - Print the string
3. `return 0` - Return success code
4. `shadow main { ... }` - Required tests for the `main` function

## Core Concepts

### 1. Dual Notation: Prefix and Infix

nanolang supports **both** prefix notation (like Lisp) and conventional infix notation for operators:

```nano
# Prefix notation (S-expression style):
(+ 1 2)              # 3
(* 3 4)              # 12
(== x 5)             # x == 5
(and true false)     # false

# Infix notation (conventional style):
1 + 2                # 3
3 * 4                # 12
x == 5               # x == 5
true and false       # false

# Both styles work - use whichever is clearer!
```

**Important:** All infix operators have **equal precedence** and evaluate **left-to-right** (no PEMDAS). Use parentheses to control grouping:

```nano
# Without parens: evaluated left-to-right
a + b * c            # means (a + b) * c, NOT a + (b * c)

# Use parens for explicit grouping
a * (b + c)          # multiply a by the sum of b and c
```

**Note:** Function calls still use prefix notation: `(println "hello")`, `(add 2 3)`. Unary operators `not` and `-` work without parens: `not flag`, `-x`.

### 2. Explicit Types

Every variable must declare its type:

```nano
let x: int = 42
let name: string = "Alice"
let flag: bool = true
let pi: float = 3.14
```

Types available:
- `int` - 64-bit integer
- `float` - 64-bit floating point
- `bool` - true or false
- `string` - UTF-8 text
- `void` - no value (for functions only)

### 3. Functions

Functions are defined with `fn`:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

**Rules:**
- Parameters must have types: `a: int`
- Must specify return type: `-> int`
- Must return a value (unless `void`)
- Must have a shadow-test

### 4. Shadow-Tests

Every function needs tests:

```nano
shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

**Key points:**
- Tests run during compilation
- Failed tests = failed compilation
- Tests document expected behavior
- Tests are removed from production builds

### 5. Immutable by Default

Variables are immutable unless marked with `mut`:

```nano
let x: int = 10
# set x 20  # ERROR: x is immutable

let mut y: int = 10
set y 20     # OK: y is mutable
```

## Common Patterns

### Conditional Logic

```nano
fn abs(n: int) -> int {
    if (< n 0) {
        return (- 0 n)
    } else {
        return n
    }
}

shadow abs {
    assert (== (abs 5) 5)
    assert (== (abs -5) 5)
    assert (== (abs 0) 0)
}
```

**Note:** Both `if` and `else` branches are required.

### Loops

**While loop:**
```nano
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

**For loop:**
```nano
for i in (range 0 10) {
    print i
}
```

### Recursion

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```

## Quick Reference

### Arithmetic Operators
```nano
# Prefix         # Infix
(+ a b)          # a + b     Addition
(- a b)          # a - b     Subtraction
(* a b)          # a * b     Multiplication
(/ a b)          # a / b     Division
(% a b)          # a % b     Modulo
```

### Comparison Operators
```nano
# Prefix         # Infix
(== a b)         # a == b    Equal
(!= a b)         # a != b    Not equal
(< a b)          # a < b     Less than
(<= a b)         # a <= b    Less or equal
(> a b)          # a > b     Greater than
(>= a b)         # a >= b    Greater or equal
```

### Logical Operators
```nano
# Prefix         # Infix
(and a b)        # a and b   Logical AND
(or a b)         # a or b    Logical OR
(not a)          # not a     Logical NOT (unary, no parens needed)
```

### Keywords
```
fn       let      mut      set      if       else
while    for      in       return   assert   shadow
int      float    bool     string   void
true     false    print    and      or       not
```

## Common Mistakes

### Both notations work for operators
```nano
let sum: int = (+ a b)    # Prefix notation
let sum: int = a + b       # Infix notation (also valid!)
```

**Note:** All infix operators have equal precedence (left-to-right). Use parentheses to group:
```nano
let result: int = a * (b + c)   # Parens needed: no PEMDAS
```

---

### ❌ Wrong: Missing type annotation
```nano
let x = 42
```

### ✅ Correct: Explicit type
```nano
let x: int = 42
```

---

### ❌ Wrong: Missing shadow-test
```nano
fn double(x: int) -> int {
    return (* x 2)
}
```

### ✅ Correct: Include shadow-test
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
    assert (== (double 0) 0)
}
```

---

### ❌ Wrong: Mutating immutable variable
```nano
let x: int = 10
set x 20
```

### ✅ Correct: Declare as mutable
```nano
let mut x: int = 10
set x 20
```

---

### ❌ Wrong: If without else
```nano
if (> x 0) {
    return 1
}
```

### ✅ Correct: Include else branch
```nano
if (> x 0) {
    return 1
} else {
    return -1
}
```

## Next Steps

1. **Read the examples** - Check out the `examples/` directory
2. **Try writing code** - Start with simple functions
3. **Write shadow-tests** - Practice test-driven development
4. **Read the spec** - See `SPECIFICATION.md` for details

## Learning Resources

- `README.md` - Overview and philosophy
- `SPECIFICATION.md` - Complete language reference
- `examples/` - Sample programs
- `examples/README.md` - Example walkthrough

## Philosophy

nanolang is designed to be:

1. **Minimal** - Small language, big capabilities
2. **Unambiguous** - One way to do things
3. **Safe** - Catch errors at compile time
4. **Tested** - Tests are mandatory, not optional
5. **Clear** - Readable by humans and LLMs

## Questions?

The language specification (`SPECIFICATION.md`) covers everything in detail. For specific examples, check the `examples/` directory.

Happy coding! 🚀

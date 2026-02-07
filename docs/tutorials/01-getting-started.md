# Tutorial 1: Getting Started with Nanolang

Welcome to nanolang! This tutorial will get you up and running in minutes.

## What is Nanolang?

Nanolang is a minimal, LLM-friendly programming language that:
- Compiles to native code via C
- Has an interpreter for rapid development
- Features mandatory testing (shadow tests)
- Supports both prefix `(+ a b)` and infix `a + b` notation for operators
- Supports modern features (generics, pattern matching, FFI)

## Installation

### Prerequisites

- GCC or Clang compiler
- Make build system
- Git

### Build from Source

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make
```

This builds:
- `bin/nanoc` - The compiler (transpiles to C)
- `bin/nano` - The interpreter (direct execution)

### Verify Installation

```bash
./bin/nano --version
./bin/nanoc --version
```

## Your First Program

Create `hello.nano`:

```nano
fn greet(name: string) -> void {
    (println (string_concat "Hello, " name))
}

fn main() -> int {
    (greet "World")
    return 0
}

shadow greet {
    assert (== (greet "Test") void)
}
```

### Running with the Interpreter

```bash
./bin/nano hello.nano
```

Output:
```
Hello, World
```

### Compiling to Native Code

```bash
./bin/nanoc hello.nano -o hello
./hello
```

Same output, but now it's a native executable!

## Understanding the Syntax

### Prefix and Infix Notation

Nanolang supports both prefix notation (like Lisp) and infix notation for operators:

```nano
// Prefix notation (original style)
(+ 2 3)          // 5
(* (+ 1 2) 4)    // 12

// Infix notation (also valid!)
2 + 3             // 5
(1 + 2) * 4      // 12

// Comparison (both styles work)
(< 5 10)         // true
5 < 10           // true
(== x y)         // equality check
x == y           // equality check

// Function calls always use prefix
(println "Hello")
(string_concat "Hello" " World")
```

**Note:** All infix operators have equal precedence and are evaluated left-to-right. Use parentheses to group: `a * (b + c)`. Unary `not` and `-` work without parens: `not flag`, `-x`.

### Type System

Nanolang is statically typed:

```nano
let x: int = 42
let y: float = 3.14
let name: string = "Alice"
let flag: bool = true
```

Type inference works for simple cases:

```nano
let z = (+ 1 2)  // inferred as int
```

### Functions

Functions are defined with `fn`:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn greet(name: string) -> void {
    (println name)
}
```

**Return types are mandatory** - even `void` for functions with no return value.

## Shadow Tests (Mandatory Testing)

Every function can have a `shadow` test block:

```nano
fn multiply(a: int, b: int) -> int {
    return (* a b)
}

shadow multiply {
    assert (== (multiply 2 3) 6)
    assert (== (multiply 0 5) 0)
    assert (== (multiply -2 4) (- 8))
}
```

Shadow tests:
- Run automatically during compilation
- Ensure code correctness
- Are self-documenting
- Catch regressions early

## Common Patterns

### Variables

```nano
let x: int = 10           // Immutable by default
let mut y: int = 5        // Mutable variable
let y = (+ y 1)           // Shadowing (creates new binding)
set y (+ y 1)             // Mutation (requires 'mut')
```

### Conditionals

```nano
fn abs(x: int) -> int {
    if (< x 0) {
        return (- x)
    } else {
        return x
    }
}
```

### Loops

```nano
fn count_to_n(n: int) -> void {
    let mut i: int = 0
    while (< i n) {
        (println (int_to_string i))
        set i (+ i 1)
    }
}
```

### Arrays

```nano
let numbers: array<int> = [1, 2, 3, 4, 5]
let first: int = (get numbers 0)
let length: int = (len numbers)

let mut arr: array<int> = []
let arr = (push arr 42)  // Arrays are immutable - returns new array
```

## Next Steps

Now that you understand the basics:

1. **[Tutorial 2: Language Fundamentals](02-language-fundamentals.md)** - Deeper dive into types, 
control flow, and data structures
2. **[Tutorial 3: Module System](03-modules.md)** - Organizing code and using external libraries
3. **[Examples](../../examples/)** - Browse working examples

## Quick Reference

### Built-in Functions

```nano
// Console I/O
(println str)              // Print with newline
(print str)                // Print without newline

// String operations
(string_concat a b)        // Concatenate strings
(string_length str)        // Get length
(substring str start len)  // Extract substring

// Type conversions
(int_to_string n)          // Convert int to string
(string_to_int str)        // Convert string to int
(float_to_string f)        // Convert float to string

// Array operations
(len array)                // Get array length
(get array index)          // Get element
(push array value)         // Append element (returns new array)
```

### Compilation Options

```bash
# Compile to executable
./bin/nanoc program.nano -o program

# Compile with verbose output
./bin/nanoc program.nano -o program -v

# Run with interpreter
./bin/nano program.nano

# Compile and run all tests
make test
```

## Getting Help

- **Documentation**: `docs/FEATURES.md` - Complete language reference
- **Examples**: `examples/` - Working code samples
- **Issues**: GitHub Issues for questions and bug reports
- **Modules**: `docs/MODULES.md` - Available libraries

## Common Pitfalls

❌ **Wrong: Missing type annotation**
```nano
let x = 2 + 3  // ERROR: Missing type annotation
```

✅ **Correct: Both prefix and infix work with type annotations**
```nano
let x: int = (+ 2 3)   // prefix notation
let y: int = 2 + 3     // infix notation (also valid!)
```

---

❌ **Wrong: Missing return type**
```nano
fn greet(name: string) {  // ERROR: No return type
    (println name)
}
```

✅ **Correct: Explicit return type**
```nano
fn greet(name: string) -> void {
    (println name)
}
```

---

❌ **Wrong: Mutating immutable variable**
```nano
let x: int = 5
set x 10  // ERROR: x is not mutable
```

✅ **Correct: Declare as mutable**
```nano
let mut x: int = 5
set x 10  // OK
```

## What's Next?

You now know enough to write simple nanolang programs! Continue to 
[Tutorial 2: Language Fundamentals](02-language-fundamentals.md) to learn about:

- Structs and enums
- Generic types
- Pattern matching
- Higher-order functions
- Error handling with union types

Happy coding! 🚀


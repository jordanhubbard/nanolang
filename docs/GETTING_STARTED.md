# Learning My Ways

I am NanoLang. This guide exists to help you understand how I function and how to write programs that I will accept. I am a minimal language designed to be written by machines and read by humans without ambiguity.

## Who I Am

I was designed with specific convictions:

- **Simple**: I have a small set of features. I do not provide multiple ways to do the same thing.
- **Clear**: Every construct I possess has exactly one meaning.
- **Safe**: My static type system catches errors at compile time so they do not happen at runtime.
- **Tested**: I require shadow tests for every function. If you do not test your code, I will not compile it.
- **LLM-friendly**: My syntax is optimized for AI code generation.

## Your First Program

This is how I expect a basic program to look.

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

1. `fn main() -> int` - You define a function named `main` that returns an `int`.
2. `print "Hello, World!"` - I output the string.
3. `return 0` - You return a success code.
4. `shadow main { ... }` - You provide the mandatory tests for your `main` function.

## Core Concepts

### 1. Dual Notation: Prefix and Infix

I support both prefix notation and conventional infix notation for operators. You may use whichever you find clearer in a given context.

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

**Important**: I do not have operator precedence. All my infix operators have equal precedence and I evaluate them strictly from left to right. I do not follow PEMDAS. You must use parentheses to control grouping.

```nano
# Without parens: evaluated left-to-right
a + b * c            # means (a + b) * c, NOT a + (b * c)

# Use parens for explicit grouping
a * (b + c)          # multiply a by the sum of b and c
```

My function calls always use prefix notation: `(println "hello")`, `(add 2 3)`. My unary operators `not` and `-` work without parentheses: `not flag`, `-x`.

### 2. Explicit Types

I require you to declare the type of every variable. I do not like guessing what you meant.

```nano
let x: int = 42
let name: string = "Alice"
let flag: bool = true
let pi: float = 3.14
```

The types I provide are:
- `int` - 64-bit integer
- `float` - 64-bit floating point
- `bool` - true or false
- `string` - UTF-8 text
- `void` - no value (used only for functions)

### 3. Functions

You define functions using the `fn` keyword.

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

My rules for functions are strict:
- You must provide types for all parameters.
- You must specify a return type.
- You must return a value unless the return type is `void`.
- You must provide a shadow test.

### 4. Shadow Tests

I require a test for every function you write.

```nano
shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

What you should know about my tests:
- I run them during compilation.
- If a test fails, I will refuse to compile your program.
- They serve as documentation for how your code should behave.
- I remove them from the final production binary.

### 5. Immutable by Default

My variables are immutable by default. I find it safer that way. If you intend to change a value, you must mark it with `mut`.

```nano
let x: int = 10
# set x 20  # I will report an error here because x is immutable

let mut y: int = 10
set y 20     # This is allowed because y is mutable
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

I require both the `if` and the `else` branch. I do not allow ambiguity in control flow.

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

## Things I Will Not Let You Do

I am very particular about how I am written. Here are several common ways to make me stop compiling your code.

### Both notations work for operators
```nano
let sum: int = (+ a b)    # Prefix notation
let sum: int = a + b       # Infix notation (also valid)
```

Remember that I evaluate all infix operators from left to right. I do not have precedence rules.

```nano
let result: int = a * (b + c)   # You must use parentheses here
```

---

### Incorrect: Missing type annotation
```nano
let x = 42
```

### Correct: Explicit type
```nano
let x: int = 42
```

---

### Incorrect: Missing shadow test
```nano
fn double(x: int) -> int {
    return (* x 2)
}
```

### Correct: Include shadow test
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

### Incorrect: Mutating an immutable variable
```nano
let x: int = 10
set x 20
```

### Correct: Declare as mutable
```nano
let mut x: int = 10
set x 20
```

---

### Incorrect: If without else
```nano
if (> x 0) {
    return 1
}
```

### Correct: Include else branch
```nano
if (> x 0) {
    return 1
} else {
    return -1
}
```

## Alternative: My Virtual Machine

I can transpile to C, but I can also compile to my own virtual machine.

```bash
# Build the VM backend
make vm

# Compile and run directly in the VM
./bin/nano_virt hello.nano --run

# Or compile to a native binary that embeds the VM
./bin/nano_virt hello.nano -o hello
./hello

# Or emit bytecode and run it separately
./bin/nano_virt hello.nano --emit-nvm -o hello.nvm
./bin/nano_vm hello.nvm
```

Why you might use my VM backend:
- **Sandboxed execution**: I isolate all external function calls in a separate co-process. If an FFI call crashes, my VM survives.
- **No C compiler needed**: My VM runs bytecode directly.
- **Formally grounded**: My core semantics are verified in Coq. My VM behavior can be validated against a reference interpreter that I have proved correct.

Both my backends produce identical results for the same program. You can find more details in docs/NANOISA.md.

## Next Steps

1. **Read my examples**: You can find many programs in the `examples/` directory.
2. **Try writing code**: Start with simple functions.
3. **Write shadow tests**: You must practice this, as I will not let you skip it.
4. **Read my specification**: I have documented every detail in `SPECIFICATION.md`.

## Learning Resources

- `README.md` - My overview and convictions.
- `SPECIFICATION.md` - My complete technical reference.
- `examples/` - Programs that I can compile.
- `examples/README.md` - A guide to my examples.

## My Philosophy

I am governed by these principles:

1. **Minimal**: I am a small language with significant capabilities.
2. **Unambiguous**: I provide exactly one way to do things.
3. **Safe**: I catch errors before they happen.
4. **Tested**: My tests are mandatory.
5. **Clear**: I am readable by both humans and machines.

## Questions?

My specification covers everything in detail. If you need to see how I handle specific tasks, look at my examples. I usually have an answer there.

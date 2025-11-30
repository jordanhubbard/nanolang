# nanolang Features Overview

**Version:** 0.2.0  
**Status:** Alpha - Feature-Complete, Production-Ready Approaching

---

## Core Features

### ✅ Prefix Notation (S-Expressions)

All operations use prefix notation for unambiguous parsing:

```nano
(+ a b)           # Addition
(* (+ 2 3) 4)     # (2 + 3) * 4
(and (> x 0) (< x 10))  # x > 0 && x < 10
```

**Benefits:**
- No operator precedence ambiguity
- LLM-friendly syntax
- Consistent function call syntax

---

### ✅ Static Type System

All variables and parameters must have explicit type annotations:

```nano
let x: int = 42              # Explicit type required
let mut y: float = 3.14      # Mutable variable
```

**Type Safety:**
- No implicit conversions
- No type inference
- Compile-time type checking
- Type errors caught before runtime

---

### ✅ Mandatory Shadow-Tests

Every function must have a `shadow` block with assertions:

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

**Benefits:**
- 100% test coverage guaranteed
- Tests run at compile time
- Living documentation
- Catch bugs before runtime

---

## Type System Features

### ✅ Primitive Types

| Type     | Description | Size | Example |
|----------|-------------|------|---------|
| `int`    | 64-bit signed integer | 8 bytes | `42`, `-17` |
| `float`  | 64-bit floating point | 8 bytes | `3.14`, `-0.5` |
| `bool`   | Boolean | 1 byte | `true`, `false` |
| `string` | UTF-8 text | 8 bytes (pointer) | `"Hello"` |
| `void`   | No value (return only) | 0 bytes | - |

---

### ✅ Structs (Product Types)

Structs group related data:

```nano
struct Point {
    x: int,
    y: int
}

let p: Point = Point { x: 10, y: 20 }
let x_coord: int = p.x
```

**Features:**
- Named fields
- Field access with `.` operator
- Stack-allocated by default
- GC-managed heap allocation available

---

### ✅ Enums (Enumerated Types)

Enums define named integer constants:

```nano
enum Status {
    Pending = 0,
    Active = 1,
    Complete = 2
}

let s: int = Status.Active  # Enums are integers
```

**Features:**
- Explicit integer values
- Compile-time constants
- Zero runtime overhead

---

### ✅ Unions (Tagged Unions / Sum Types)

Unions represent a value that can be one of several variants:

```nano
union Result {
    Ok { value: int },
    Error { code: int, message: string }
}

fn divide(a: int, b: int) -> Result {
    if (== b 0) {
        return Result.Error { code: 1, message: "Division by zero" }
    } else {
        return Result.Ok { value: (/ a b) }
    }
}

shadow divide {
    match (divide 10 2) {
        Ok(v) => assert (== v.value 5),
        Error(e) => assert false
    }
}
```

**Features:**
- Type-safe variant handling
- Pattern matching with `match` expressions
- Named fields per variant
- Compile-time exhaustiveness checking

---

### ✅ Pattern Matching

Match expressions destructure unions safely:

```nano
match result {
    Ok(v) => (println v.value),
    Error(e) => (println e.message)
}
```

**Features:**
- Exhaustive pattern checking
- Variable binding for each variant
- Type-safe field access
- Expression-based (returns a value)

---

### ✅ Generics (Monomorphization)

Generic types enable reusable, type-safe code:

```nano
# Built-in generic: List<T>
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)
(List_int_push numbers 17)

let len: int = (List_int_length numbers)
let first: int = (List_int_get numbers 0)

# Generics with user-defined types
struct Point { x: int, y: int }
let points: List<Point> = (List_Point_new)
(List_Point_push points (Point { x: 1, y: 2 }))
```

**Implementation:**
- **Monomorphization:** Each concrete type generates specialized code
- `List<int>` → `List_int_new`, `List_int_push`, etc.
- `List<Point>` → `List_Point_new`, `List_Point_push`, etc.
- Zero runtime overhead
- Type-safe at compile time

**Generic Functions Available:**
- `List_<T>_new()` - Create empty list
- `List_<T>_push(list, value)` - Push element
- `List_<T>_length(list)` - Get length
- `List_<T>_get(list, index)` - Get element

---

### ✅ First-Class Functions

Functions can be passed as parameters, returned from functions, and assigned to variables:

```nano
fn double(x: int) -> int {
    return (* x 2)
}

# Assign function to variable
let f: fn(int) -> int = double
let result: int = (f 7)  # result = 14

# Function as parameter
fn apply_twice(op: fn(int) -> int, x: int) -> int {
    return (op (op x))
}

let y: int = (apply_twice double 5)  # y = 20

# Function as return value
fn get_operation(choice: int) -> fn(int) -> int {
    if (== choice 0) {
        return double
    } else {
        return triple
    }
}
```

**Function Type Syntax:**
```nano
fn(param_type1, param_type2) -> return_type
```

**Features:**
- Functions are values
- No exposed function pointers
- Type-safe function variables
- No dereferencing needed

---

### ⏳ Tuples (In Development)

Tuples allow returning multiple values:

```nano
fn divide_with_remainder(a: int, b: int) -> (int, int) {
    return ((/ a b), (% a b))
}

let result: (int, int) = (divide_with_remainder 10 3)
let quotient: int = result.0
let remainder: int = result.1
```

**Status:** Type system complete, parser implementation pending

---

## Control Flow

### ✅ If Expressions

Both branches required:

```nano
if (> x 0) {
    (println "Positive")
} else {
    (println "Non-positive")
}
```

---

### ✅ While Loops

```nano
let mut i: int = 0
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

---

### ✅ For Loops

```nano
for i in (range 0 10) {
    (println i)
}
```

---

## Mutability

### ✅ Immutable by Default

Variables are immutable unless declared with `mut`:

```nano
let x: int = 10
# set x 20  # ERROR: x is immutable

let mut y: int = 10
set y 20  # OK: y is mutable
```

**Benefits:**
- Safer code by default
- Explicit mutability tracking
- Easier to reason about

---

## Standard Library

### ✅ Comprehensive Built-ins (37 Functions)

**Core I/O (3):**
- `print`, `println`, `assert`

**Math Operations (11):**
- `abs`, `min`, `max`, `sqrt`, `pow`
- `floor`, `ceil`, `round`
- `sin`, `cos`, `tan`

**String Operations (18):**
- `str_length`, `str_concat`, `str_substring`
- `str_contains`, `str_equals`
- `char_at`, `string_from_char`
- `is_digit`, `is_alpha`, `is_alnum`
- `is_whitespace`, `is_upper`, `is_lower`
- `int_to_string`, `string_to_int`, `digit_value`
- `char_to_lower`, `char_to_upper`

**Array Operations (4):**
- `at`, `array_length`, `array_new`, `array_set`

**OS/System (3):**
- `getcwd`, `getenv`, `range`

**Generics (Dynamic per type):**
- `List_<T>_new`, `List_<T>_push`, `List_<T>_length`, `List_<T>_get`

See [`STDLIB.md`](STDLIB.md) for full reference.

---

## Compilation & Tooling

### ✅ C Transpilation

nanolang transpiles to C99:

```bash
nanoc program.nano -o program
./program
```

**Features:**
- Readable C output (with `--keep-c`)
- Zero-overhead abstractions
- Compatible with C toolchain
- Easy FFI with C libraries

---

### ✅ Namespacing

All user-defined types are prefixed with `nl_` in generated C code:

```nano
struct Point { x: int, y: int }
enum Status { Active = 1, Pending = 0 }
union Result { Ok { value: int }, Error { code: int } }
```

**Generated C:**
```c
typedef struct nl_Point { int64_t x; int64_t y; } nl_Point;
typedef enum { nl_Status_Active = 1, nl_Status_Pending = 0 } nl_Status;
typedef struct nl_Result { /* tagged union */ } nl_Result;
```

**Benefits:**
- Prevents name collisions with C runtime
- Clean C interop
- Enables calling nanolang from C

---

### ✅ Interpreter with Tracing

nanolang includes a fast interpreter for development:

```bash
nano program.nano
```

**Tracing Flags:**
- `--trace-all` - Trace everything
- `--trace-function=<name>` - Trace specific function
- `--trace-var=<name>` - Trace variable operations
- `--trace-scope=<name>` - Trace function scope
- `--trace-regex=<pattern>` - Trace by regex

**Benefits:**
- Fast iteration during development
- Detailed execution traces
- No compilation step for testing
- Shadow-tests run automatically

---

## Safety Features

### ✅ Memory Safety

- Static type checking
- Bounds-checked array access
- No manual memory management
- GC for dynamic data structures

---

### ✅ Type Safety

- No implicit conversions
- No null pointers
- Tagged unions for error handling
- Exhaustive pattern matching

---

### ✅ Test-Driven

- Mandatory shadow-tests
- 100% function coverage
- Compile-time test execution
- Living documentation

---

## FFI (Foreign Function Interface)

### ✅ Extern Functions

Call C functions from nanolang:

```nano
extern fn sqrt(x: float) -> float
extern fn sin(x: float) -> float

fn pythagorean(a: float, b: float) -> float {
    return (sqrt (+ (* a a) (* b b)))
}
```

**Features:**
- Direct C function calls
- Type-safe bindings
- No overhead
- Easy integration with C libraries

---

## Development Status

**Completed Features:**
- ✅ Core language (types, expressions, statements)
- ✅ Structs, enums, unions
- ✅ Generics with monomorphization
- ✅ First-class functions
- ✅ Pattern matching
- ✅ Standard library (37 functions)
- ✅ C transpilation with namespacing
- ✅ Interpreter with tracing
- ✅ Shadow-test system
- ✅ FFI support
- ✅ Zero compiler warnings

**In Development:**
- ⏳ Tuple types (type system done, parser pending)
- ⏳ Self-hosted compiler (nanolang-in-nanolang)

**Planned:**
- 📋 Module system
- 📋 More generic types (Map<K,V>, Set<T>)
- 📋 Package manager
- 📋 Standard library expansion

---

## Getting Started

**Install:**
```bash
git clone <repository>
cd nanolang
make
```

**Hello World:**
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

**Compile:**
```bash
nanoc hello.nano -o hello
./hello
```

**Run with interpreter:**
```bash
nano hello.nano
```

---

## Resources

- **Quick Reference:** [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)
- **Language Spec:** [`SPECIFICATION.md`](SPECIFICATION.md)
- **Getting Started:** [`GETTING_STARTED.md`](GETTING_STARTED.md)
- **Testing Guide:** [`SHADOW_TESTS.md`](SHADOW_TESTS.md)
- **Standard Library:** [`STDLIB.md`](STDLIB.md)
- **Examples:** [`../examples/`](../examples/)

---

**nanolang** - Minimal, LLM-friendly, test-driven programming for the modern age! 🚀


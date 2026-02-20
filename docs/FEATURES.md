# Features

**Version:** 0.2.0  
**Status:** Alpha - I am feature-complete and approaching production-readiness.

---

## Core Features

### Automatic Memory Management (ARC)

I provide automatic reference counting (ARC) to manage memory without manual intervention.

```nano
from "modules/std/json/json.nano" import Json, parse, get_string

fn extract_data(json_text: string) -> string {
    let root: Json = (parse json_text)         # Owned - auto-freed
    let name: string = (get_string root "name") # Borrowed - no overhead
    return name
    # No free() needed - ARC handles everything!
}
```

I eliminate manual memory management and the need for `free` calls. My cleanup is deterministic: I free objects as soon as their last reference disappears. I detect borrowed references to ensure accessors have no overhead, and I handle circular references automatically. I compile to C with minimal overhead to maintain performance.

See [Automatic Memory Management Guide](../userguide/03_basic_types.md#automatic-memory-management) for details.

---

### Dual Notation: Prefix and Infix

I support both prefix (S-expression) and infix notation for operators.

```nano
# Prefix notation (S-expression style):
(+ a b)                      # Addition
(* (+ 2 3) 4)                # (2 + 3) * 4
(and (> x 0) (< x 10))      # x > 0 && x < 10

# Infix notation (conventional style):
a + b                        # Addition
(2 + 3) * 4                  # Use parens to group
x > 0 and x < 10            # Logical operators too
```

My infix operators include: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `and`, `or`.

My rules for infix notation are:
- All infix operators have equal precedence and evaluate left-to-right. I do not use PEMDAS.
- You must use parentheses to control grouping: `a * (b + c)`.
- Unary `not` and `-` work without parentheses: `not flag`, `-x`.
- My function calls always remain prefix: `(println "hello")`.

I offer these choices so you can use the clearest notation for each situation. My prefix style eliminates all ambiguity, while my infix style reads naturally for simple expressions. I designed this syntax to be friendly to both humans and machines.

---

### Static Type System

I require explicit type annotations for all variables and parameters.

```nano
let x: int = 42              # Explicit type required
let mut y: float = 3.14      # Mutable variable
```

My type system ensures safety by disallowing implicit conversions and type inference. I check all types at compile time so that errors are caught before your program runs.

---

### Mandatory Shadow-Tests

I require every function to have a `shadow` block containing assertions.

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

I enforce this to guarantee test coverage. I run these tests at compile time, providing you with living documentation and catching bugs before runtime.

---

## Type System Features

### Primitive Types

| Type     | Description | Size | Example |
|----------|-------------|------|---------|
| `int`    | 64-bit signed integer | 8 bytes | `42`, `-17` |
| `float`  | 64-bit floating point | 8 bytes | `3.14`, `-0.5` |
| `bool`   | Boolean | 1 byte | `true`, `false` |
| `string` | UTF-8 text | 8 bytes (pointer) | `"Hello"` |
| `void`   | No value (return only) | 0 bytes | - |

---

### Structs (Product Types)

I use structs to group related data.

```nano
struct Point {
    x: int,
    y: int
}

let p: Point = Point { x: 10, y: 20 }
let x_coord: int = p.x
```

I support named fields and field access via the `.` operator. I allocate structs on the stack by default, but I also provide GC-managed heap allocation when needed.

---

### Enums (Enumerated Types)

I define enums as named integer constants.

```nano
enum Status {
    Pending = 0,
    Active = 1,
    Complete = 2
}

let s: int = Status.Active  # Enums are integers
```

My enums use explicit integer values and function as compile-time constants with zero runtime overhead.

---

### Unions (Tagged Unions / Sum Types)

I represent values that can be one of several variants using unions.

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

I provide type-safe variant handling and pattern matching with `match` expressions. Each variant can have named fields. I check for exhaustiveness at compile time.

### Generic Unions

I allow unions to be generic over type parameters.

```nano
union Result<T, E> {
    Ok { value: T },
    Err { error: E }
}

fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    return Result.Ok { value: (/ a b) }
}

shadow divide {
    let r1: Result<int, string> = (divide 10 2)
    match r1 {
        Ok(v) => assert (== v.value 5),
        Err(e) => assert false
    }
    
    let r2: Result<int, string> = (divide 10 0)
    match r2 {
        Ok(v) => assert false,
        Err(e) => assert (str_equals e.error "Division by zero")
    }
}
```

I support generic type parameters like `<T, E>` and use monomorphization to generate concrete types at compile time. This works with primitives, structs, and other generics. My standard library includes `Result<T,E>`, and I plan to add helper functions once I support generic functions.

**Standard Library Usage:**
```nano
fn main() -> int {
    let result: Result<int, string> = (divide 10 2)

    match result {
        Ok(v) => (println v.value),
        Err(e) => (println e.error)
    }

    return 0
}
```

---

### Pattern Matching

I use match expressions to destructure unions safely.

```nano
match result {
    Ok(v) => (println v.value),
    Error(e) => (println e.message)
}
```

I ensure pattern checking is exhaustive and provide variable binding for each variant. This allows for type-safe field access within an expression that returns a value.

---

### Generics (Monomorphization)

I use generic types to enable the creation of reusable, type-safe code.

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

My implementation uses monomorphization, meaning I generate specialized code for each concrete type, such as `List_int_new` or `List_Point_new`. This approach ensures zero runtime overhead and maintains type safety at compile time.

My available generic functions include:
- `List_<T>_new()` - Create empty list
- `List_<T>_push(list, value)` - Push element
- `List_<T>_length(list)` - Get length
- `List_<T>_get(list, index)` - Get element

---

### First-Class Functions

I treat functions as values. You can pass them as parameters, return them from other functions, and assign them to variables.

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

My syntax for function types is `fn(param_types) -> return_type`. I do not expose function pointers; I use type-safe function variables that do not require dereferencing.

---

### Tuples

Planned: I am adding tuples to allow returning multiple values.

```nano
fn divide_with_remainder(a: int, b: int) -> (int, int) {
    return ((/ a b), (% a b))
}

let result: (int, int) = (divide_with_remainder 10 3)
let quotient: int = result.0
let remainder: int = result.1
```

Development status: My type system support is complete, and I am currently implementing the parser.

---

## Control Flow

### If Expressions

I require both branches to be present.

```nano
if (> x 0) {
    (println "Positive")
} else {
    (println "Non-positive")
}
```

---

### While Loops

```nano
let mut i: int = 0
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

---

### For Loops

```nano
for i in (range 0 10) {
    (println i)
}
```

---

## Mutability

### Immutable by Default

My variables are immutable unless you declare them with `mut`.

```nano
let x: int = 10
# set x 20  # ERROR: x is immutable

let mut y: int = 10
set y 20  # OK: y is mutable
```

I enforce this to make your code safer and easier to reason about through explicit mutability tracking.

---

## Standard Library

### Comprehensive Built-ins (72 Functions)

I provide 72 built-in functions covering several areas.

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

## Compilation and Tooling

### Dual Compilation: C Transpilation and NanoISA Virtual Machine

I offer two compilation backends.

**C Transpilation** (default, maximum performance):
```bash
nanoc program.nano -o program    # Transpile to C, compile to native binary
./program
```

**NanoISA Virtual Machine** (sandboxed, portable):
```bash
nano_virt program.nano --run          # Compile to bytecode + execute in VM
nano_virt program.nano -o program     # Compile to native binary (embeds VM)
nano_virt program.nano --emit-nvm -o program.nvm  # Emit raw bytecode
nano_vm program.nvm                   # Execute bytecode
nano_vm --isolate-ffi program.nvm     # Execute with FFI in separate process
```

Comparison of my backends:

| | C Backend (`nanoc`) | VM Backend (`nano_virt`) |
|-|---------------------|--------------------------|
| **Performance** | Native speed | Interpreted |
| **FFI safety** | In-process | Process-isolated (crash-safe) |
| **Dependencies** | Needs gcc/clang | Self-contained |
| **Debugging** | GDB/LLDB on C output | Source-line debug info in bytecode |
| **Formal grounding** | Semantics verified in Coq | Semantics verified in Coq + differential testing |

**My VM Architecture:**
- I use a 178-opcode stack machine with reference-counted GC.
- I isolate external function calls in a separate `nano_cop` process via RPC, so FFI crashes cannot take down my VM.
- I offer an optional daemon mode (`nano_vmd`) to reduce startup latency.
- I use a trap model that separates my pure-compute core from I/O, allowing for potential FPGA acceleration.

See [NanoISA Architecture Reference](NANOISA.md) for the complete specification.

---

### Formally Verified Semantics

I have mechanically verified my core language (NanoCore) in the Rocq Prover (Coq).

- **Type Soundness**: I have proved that well-typed programs do not go wrong (preservation and progress).
- **Determinism**: I have proved that evaluation is a partial function.
- **Semantic Equivalence**: I have proved that my big-step and small-step semantics agree.
- **Computable Evaluator**: I have an axiom-free soundness proof for my fuel-based reference interpreter.

I have completed 6,170 lines of Coq proof and 193 theorems using zero axioms.

See [formal/README.md](../formal/README.md) for the full proof suite.

---

### C Transpilation

I transpile to C99.

```bash
nanoc program.nano -o program
./program
```

I produce readable C output when you use the `--keep-c` flag. I use zero-overhead abstractions and remain compatible with the C toolchain for easy FFI integration.

---

### Namespacing

I prefix all user-defined types with `nl_` in my generated C code.

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

I do this to prevent name collisions with the C runtime and to provide clean interop for calling me from C.

---

### Interpreter with Tracing

I include a fast interpreter for development.

```bash
nano program.nano
```

I provide several tracing flags:
- `--trace-all` - Trace everything
- `--trace-function=<name>` - Trace specific function
- `--trace-var=<name>` - Trace variable operations
- `--trace-scope=<name>` - Trace function scope
- `--trace-regex=<pattern>` - Trace by regex

I use this to allow for fast iteration and detailed execution traces without a compilation step. I run shadow-tests automatically when using the interpreter.

---

## Safety Features

### Memory Safety

I ensure memory safety through static type checking, bounds-checked array access, and the elimination of manual memory management. I use GC for dynamic data structures.

---

### Type Safety

I maintain type safety by disallowing implicit conversions and null pointers. I use tagged unions for error handling and require exhaustive pattern matching.

---

### Test-Driven

I am test-driven. I require shadow-tests for 100% function coverage and execute these tests at compile time.

---

## FFI (Foreign Function Interface)

### Extern Functions

I allow you to call C functions.

```nano
extern fn sqrt(x: float) -> float
extern fn sin(x: float) -> float

fn pythagorean(a: float, b: float) -> float {
    return (sqrt (+ (* a a) (* b b)))
}
```

I provide type-safe bindings for direct C function calls with no overhead, facilitating easy integration with C libraries.

---

## Development Status

What I have completed:
- Complete: Core language (types, expressions, statements)
- Complete: Structs, enums, unions
- Complete: Generics with monomorphization
- Complete: First-class functions
- Complete: Pattern matching
- Complete: Standard library (72 functions)
- Complete: C transpilation with namespacing
- Complete: Interpreter with tracing
- Complete: Shadow-test system
- Complete: FFI support
- Complete: Zero compiler warnings

What I am currently developing:
- In development: Tuple types (type system complete, parser implementation pending)
- In development: Self-hosted compiler (I am writing myself in myself)

What I have planned:
- Planned: Module system
- Planned: More generic types (Map<K,V>, Set<T>)
- Planned: Package manager
- Planned: Standard library expansion

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



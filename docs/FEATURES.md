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

I support tuple destructuring for binding multiple return values.

```nano
fn divide_with_remainder(a: int, b: int) -> (int, int) {
    return ((/ a b), (% a b))
}

let (quotient, remainder): (int, int) = (divide_with_remainder 10 3)
```

Full first-class tuple types (as standalone function return types across all backends) are partially implemented.

---

## Control Flow

### If Statements

The `else` branch is optional when `if` is used as a statement.

```nano
if (> x 0) {
    (println "Positive")
} else {
    (println "Non-positive")
}

# else is optional for early returns or void branches:
if (< x 0) {
    return 0
}
```

When `if` is used as an expression (producing a value), both branches are required and must return the same type.

```nano
let label: string = if (> x 0) { "positive" } else { "non-positive" }
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

### Multi-Backend Compilation

I offer five compilation backends.

**C Transpilation** (default, maximum performance):
```bash
nanoc program.nano -o program         # Transpile to C, compile to native binary
nanoc program.nano --keep-c -o prog   # Keep generated C source
nanoc program.nano -g -o program      # Include DWARF debug info
```

**WebAssembly** (portable, sandboxed):
```bash
nanoc program.nano --target wasm -o program.wasm
# Also emits program.wasm.map — source map sidecar
nanoc sign   program.wasm             # Sign with Ed25519
nanoc verify program.wasm             # Verify signature
```

**LLVM IR** (LLVM-based optimization and codegen):
```bash
nanoc program.nano --target llvm -o program.ll
```

**PTX / CUDA** (GPU kernels):
```bash
nanoc program.nano --target ptx -o program.ptx
```

**RISC-V Assembly**:
```bash
nanoc program.nano --target riscv -o program.s
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

| | C (`nanoc`) | WASM | LLVM | PTX | RISC-V | NanoISA VM |
|-|-------------|------|------|-----|--------|-----------|
| **Performance** | Native | Near-native (JIT) | Optimized native | GPU | Native | Interpreted |
| **FFI safety** | In-process | Sandboxed | In-process | Device only | In-process | Process-isolated |
| **Debug info** | DWARF (`-g`) | Source map | DWARF (`-g`) | — | DWARF (`-g`) | Line info in bytecode |
| **Signing** | — | Ed25519 | — | — | — | — |

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

## Parallel Hints

### Par Blocks

I provide `par { }` blocks as an explicit independence annotation. When you write a `par` block, you declare that the enclosed statements have no ordering dependency between them. The runtime may evaluate them concurrently, in any order, or sequentially.

```nano
par {
    let a: int = (expensive_compute_a)
    let b: int = (expensive_compute_b)
    let c: int = (expensive_compute_c)
}
# a, b, c are all available here
let result: int = (+ a (+ b c))
```

**Important:** You are asserting independence. If statements in a `par` block actually depend on each other, behavior is undefined.

### Par-Let (Parallel Binding)

`par-let` binds multiple variables with a declared independence between their initializers. Both `x` and `y` are bound simultaneously and are in scope after the `in` keyword.

```nano
par-let x = 1 + 2  y = 3 * 4  in (println (+ x y))
```

This is equivalent to a `par` block followed by a `let`, but expressed as an inline expression.

---

## Advanced Language Features

### Match Guards

Pattern arms may carry a guard condition after `if`. The guard must evaluate to `bool`.

```nano
match result {
    Ok(v) if v.value > 0 => (println "positive result"),
    Ok(v)                => (println "non-positive result"),
    Err(e)               => (println e.message)
}
```

### Or-Patterns

Multiple patterns in a single arm are separated by `|`.

```nano
match status {
    | Active | Pending => (println "not done"),
    Complete           => (println "done")
}
```

---

## Algebraic Effects

I support algebraic effects as a typed, resumable side-effect mechanism.

```nano
effect Log {
    log : string -> void
}

fn greet(name: string) -> void {
    perform Log.log(f"Hello, {name}!")
}

shadow greet { assert true }

fn main() -> int {
    handle (greet "World") with {
        Log.log(msg) -> { (println msg) }
    }
    return 0
}

shadow main { assert (== (main) 0) }
```

I declare effects with `effect EffectName { op : input_type -> output_type }`. I perform them with `perform EffectName.op(arg)`. I install handlers with `handle expr with { EffectName.op(param) -> body }`.

---

## Async / Await

I support async/await through a CPS (continuation-passing style) transformation applied at compile time.

```nano
async fn fetch_data(url: string) -> string {
    let response: string = await (http_get url)
    return response
}

shadow fetch_data { assert true }
```

I lower `async fn` to a state machine and `await` to a CPS callback. The transformation is transparent to the caller.

---

## F-String Interpolation

I support f-string interpolation with automatic type conversion.

```nano
let name: string = "World"
let count: int = 42
let msg: string = f"Hello, {name}! Count: {count}"
```

I desugar `f"..."` to a sequence of `str_concat` calls at compile time. Embedded expressions may be any typed expression.

---

## Pipe Operator

I support the pipe operator `|>` for readable left-to-right function chains.

```nano
let result: int = x |> double |> increment |> negate
# equivalent to: (negate (increment (double x)))
```

---

## Documentation Export

I can export triple-slash doc comments as GFM Markdown.

```nano
/// Adds two integers and returns the sum.
/// @param a  First operand
/// @param b  Second operand
fn add(a: int, b: int) -> int { return (+ a b) }
```

```bash
nanoc program.nano --doc-md -o program.md
```

---

## WASM Module Signing

I can sign compiled WASM binaries with an Ed25519 key pair. The signature is embedded as a `agentos.signature` custom section.

```bash
nanoc program.nano --target wasm -o program.wasm
nanoc sign program.wasm          # Signs with ~/.nanoc/signing.key
nanoc verify program.wasm        # Verifies the embedded signature
```

I generate the key pair automatically on first use. Re-signing an already-signed WASM is idempotent. Tampered bytes cause `nanoc verify` to return a non-zero exit code.

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

**Status: Production-ready.** I am self-hosting, formally verified, and ship multiple compilation backends.

### Completed

**Core Language:**
- Complete: Core language (types, expressions, statements)
- Complete: Structs, enums, unions with generics
- Complete: Generic types with monomorphization
- Complete: First-class functions
- Complete: Pattern matching with match guards (`Val(u) if u.n > 0 =>`) and or-patterns (`| A | B =>`)
- Complete: Algebraic effects (`effect`, `perform`, `handle`)
- Complete: Async/await with CPS transformation
- Complete: Coroutines
- Complete: Par-let (`par-let x=e1 y=e2 in body`) for parallel evaluation hints
- Complete: Local type inference (`let x = 42` — annotation optional when unambiguous)
- Complete: Hindley-Milner type inference
- Complete: F-string interpolation (`f"Hello {name}!"`)
- Complete: Pipe operator (`x |> f |> g`)
- Complete: Anonymous functions (`fn(x: int) -> int { return (* x 2) }`)
- Complete: Tuple destructuring (`let (q, r) = (divmod 17 5)`)
- Complete: Wildcard `_` in match arms

**Compilation Backends:**
- Complete: C transpilation (default, native performance)
- Complete: NanoISA Virtual Machine (178 opcodes, process-isolated FFI)
- Complete: WebAssembly binary emit (`--target wasm`)
- Complete: LLVM IR backend (`--target llvm`)
- Complete: PTX (CUDA) backend (`--target ptx`)
- Complete: RISC-V assembly backend (`--target riscv`)
- Complete: DWARF debug info emission (`--debug` / `-g` flag)
- Complete: Source map sidecar for WASM (`program.wasm.map`)
- Complete: Ed25519 WASM module signing (`nanoc sign / nanoc verify`)

**Standard Library:**
- Complete: Standard library (72+ built-in functions)
- Complete: Module system with `module`, `pub`, `from … import`, `use`
- Complete: Package registry / manager
- Complete: 30+ FFI modules (SDL, ncurses, OpenGL, curl, readline, Python bridge, JSON, regex, PEG2…)

**Tooling:**
- Complete: Self-hosted compiler (100% bootstrap — stage1 → stage2 → stage3)
- Complete: Interpreter with tracing (`--trace-all`, `--trace-function=<name>`, …)
- Complete: Shadow-test system (compile-time execution, 100% function coverage required)
- Complete: Interactive REPL with history, hot-reload (`:load`, `:save`, `:reload`)
- Complete: Language Server (`bin/nanolang-lsp`) — hover, go-to-definition, completion, diagnostics
- Complete: Debug Adapter Protocol server (`bin/nanolang-dap`) — breakpoints, step-through, variable inspection
- Complete: VS Code extension (`editors/vscode/`) — LSP + DAP wired automatically, format-on-save
- Complete: Web playground (CodeMirror 6 editor, share permalink, AgentFS hosting — port 8792)
- Complete: `--doc-md` flag for GFM Markdown documentation export from triple-slash comments
- Complete: `--emit-typed-ast-json` for tooling integration
- Complete: DWARF debug info (`-g`) for LLVM IR and RISC-V backends
- Complete: Constant folding and dead-code elimination (AST-level optimizer passes)
- Complete: Formally verified semantics (6,170 lines of Coq, zero axioms)

### What I Am Not Yet Doing

- Tuple types are implemented for destructuring but not as first-class function return types in all backends.
- Map<K,V> and Set<T> generic containers are not yet in the standard library.

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



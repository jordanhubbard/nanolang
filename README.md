# nanolang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-20%2F20%20passing-brightgreen.svg)
![Examples](https://img.shields.io/badge/examples-25%2F28%20working-green.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange.svg)

**Status**: ✅ Production Ready - 20/20 tests passing + 25/28 examples working, **49+ stdlib functions**, arrays with bounds checking, **self-hosting foundation 100% complete** ✅

A minimal, LLM-friendly programming language designed for AI-assisted programming with strict, unambiguous syntax, mandatory shadow-tests, and a path to self-hosting via C transpilation.

## Documentation 📚

**New to nanolang?** Start here:
- 🚀 [Quick Start](#quick-start) (scroll down)
- 📖 [Getting Started Guide](docs/GETTING_STARTED.md) - Learn the basics
- ⚡ [Quick Reference](docs/QUICK_REFERENCE.md) - Syntax cheat sheet
- 📘 [Language Specification](docs/SPECIFICATION.md) - Complete reference
- 📋 [Documentation Index](docs/DOCS_INDEX.md) - All documentation

**Contributing:**
- 🤝 [Contributing Guide](docs/CONTRIBUTING.md)
- 🐛 [Report a Bug](https://github.com/jordanhubbard/nanolang/issues/new?template=bug_report.md)
- 💡 [Request a Feature](https://github.com/jordanhubbard/nanolang/issues/new?template=feature_request.md)

## Quick Start

### Prerequisites

**Required:**
- C compiler (GCC or Clang)
- Make

**Optional Modules** (only needed if using specific features):

> **Note:** The core nanolang compiler and interpreter work without any of these libraries. They're only needed if you import the corresponding modules in your code.

#### Graphics & Game Development

**SDL2** - Window management, 2D graphics, input
```bash
# macOS
brew install sdl2

# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# Fedora
sudo dnf install SDL2-devel
```

**SDL2_mixer** - Audio playback (music, sound effects)
```bash
# macOS
brew install sdl2_mixer

# Ubuntu/Debian
sudo apt-get install libsdl2-mixer-dev
```

**SDL2_ttf** - TrueType font rendering
```bash
# macOS
brew install sdl2_ttf

# Ubuntu/Debian
sudo apt-get install libsdl2-ttf-dev
```

#### OpenGL Development

**GLFW** - Modern OpenGL window and input
```bash
# macOS
brew install glfw

# Ubuntu/Debian
sudo apt install libglfw3-dev
```

**GLEW** - OpenGL Extension Wrangler
```bash
# macOS
brew install glew

# Ubuntu/Debian
sudo apt install libglew-dev
```

#### AI/ML

**ONNX Runtime** - Neural network inference
```bash
# macOS
brew install onnxruntime

# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev
```

See [docs/AI_ML_GUIDE.md](docs/AI_ML_GUIDE.md) for ONNX usage.

#### Networking & HTTP

**libcurl** - HTTP/HTTPS client for web requests and REST APIs
```bash
# macOS
brew install curl

# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev
```

#### Async I/O & Event Loops

**libevent** - Asynchronous event notification (for servers, high-performance I/O)
```bash
# macOS
brew install libevent

# Ubuntu/Debian
sudo apt install libevent-dev
```

**libuv** - Cross-platform async I/O (powers Node.js)
```bash
# macOS
brew install libuv

# Ubuntu/Debian
sudo apt install libuv1-dev
```

#### Database

**SQLite3** - Embedded SQL database
```bash
# macOS
brew install sqlite

# Ubuntu/Debian
sudo apt install libsqlite3-dev
```

#### Module Reference Table

| Module | Description | Install Command (macOS) | Use Cases |
|--------|-------------|------------------------|-----------|
| **Graphics & Games** ||||
| `sdl` | 2D graphics, windowing, input | `brew install sdl2` | Games, GUI apps, graphics |
| `sdl_mixer` | Audio playback | `brew install sdl2_mixer` | Music, sound effects |
| `sdl_ttf` | Font rendering | `brew install sdl2_ttf` | Text display in games/apps |
| **OpenGL** ||||
| `glfw` | OpenGL windowing | `brew install glfw` | 3D graphics, OpenGL apps |
| `glew` | OpenGL extensions | `brew install glew` | Modern OpenGL features |
| **Networking** ||||
| `curl` | HTTP/HTTPS client | `brew install curl` | REST APIs, web requests, downloads |
| **Async I/O** ||||
| `event` | Event notification (libevent) | `brew install libevent` | Network servers, async I/O |
| `uv` | Cross-platform async (libuv) | `brew install libuv` | Node.js-style event loop |
| **Database** ||||
| `sqlite` | Embedded SQL database | `brew install sqlite` | Local data storage, SQL queries |
| **AI/ML** ||||
| `onnx` | Neural network inference | `brew install onnxruntime` | PyTorch/TF models, ML inference |

See [docs/MODULE_SYSTEM.md](docs/MODULE_SYSTEM.md) for creating your own modules.

---

**Don't have any modules installed?** No problem! Try the basic examples first:
```bash
./bin/nano examples/hello.nano
./bin/nano examples/factorial.nano
./bin/nano examples/calculator.nano
./bin/nano examples/primes.nano
```

### Building nanolang

```bash
# Clone the repository
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang

# Build the compiler and interpreter
make

# Option 1: Run with the interpreter (instant execution)
./bin/nano examples/hello.nano

# Option 2: Compile to native binary
./bin/nanoc examples/hello.nano -o hello
./hello

# Run the test suite
make test

# Build all examples (requires SDL2 for some examples)
make examples
```

## Philosophy

**nanolang** is built on five core principles:

1. **LLM-friendly**: Syntax optimized for large language model understanding and generation
2. **Minimal**: Small core with clear semantics - easy to learn, hard to misuse
3. **Unambiguous**: Every construct has exactly one meaning, no operator precedence surprises
4. **Self-hosting**: Designed to eventually compile itself via C transpilation
5. **Test-driven**: Mandatory shadow-tests ensure correctness at every level

## Implementation Status

- ✅ **Lexer**: 100% complete with column tracking
- ✅ **Parser**: 100% complete (all features working)
- ✅ **Type Checker**: 100% complete with warnings
- ✅ **Shadow-Test Runner**: 100% complete
- ✅ **C Transpiler**: 100% complete with optimizations
- ✅ **CLI Tool**: Full-featured with version support

**Working Examples**: 25/28 (89%) ⭐  
**Critical Bugs**: 0  
**Standard Library**: **49+ functions** (11 math, 18 string, 12 bstring, 4 arrays, 3 I/O, OS stdlib)  
**Data Structures**: Arrays, Structs, Enums, Unions, Dynamic Lists ✅  
**Advanced Features**: Module system, FFI, Pattern matching, First-class functions, Generics ✅  
**Self-Hosting Foundation**: **100% Complete** (6/6 features) 🎉  
**Quality**: CI/CD, sanitizers, coverage, linting

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/FEATURES.md](docs/FEATURES.md) for details.

## What Makes nanolang Different?

nanolang is designed for the AI age - optimized for both human readability and LLM code generation:

1. **No Operator Precedence** - Prefix notation `(+ a b)` eliminates ambiguity
2. **Mandatory Tests** - Every function requires a `shadow` test block
3. **Explicit Everything** - No implicit conversions, no hidden behavior
4. **Rich Type System** - Structs, enums, unions, generics, and first-class functions
5. **Modern Tooling** - Dual execution (interpreter + compiler), module system, AI/ML support

## Quick Example

```nano
# Define a function with its shadow-test
fn add(a: int, b: int) -> int {
    return (+ a b)
}

# Shadow-test: automatically runs when function is defined
shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}

# Entry point
fn main() -> int {
    print (add 5 7)
    return 0
}
```

**Key Points:**
- Prefix notation: `(+ a b)` instead of `a + b`
- Every function needs a shadow test (enforced at compile time)
- Types are always explicit: `a: int, b: int`
- Simple, readable syntax optimized for LLMs

## Language Features

### 1. Explicit Everything

- No implicit conversions
- No operator precedence (prefix notation eliminates ambiguity)
- All types must be declared
- All functions must have return types

### 2. Shadow-Tests

Every function must have a corresponding `shadow` block that tests it. Shadow-tests:
- Run immediately after function definition during compilation
- Fail compilation if tests don't pass
- Are stripped from production builds
- Serve as executable documentation

### 3. Prefix Notation (S-expressions)

To eliminate precedence ambiguity, all operations use prefix notation:

```nano
# Instead of: a + b * c (ambiguous without memorizing precedence)
# Write: (+ a (* b c))

# Comparisons
(== x 5)
(> a b)
(<= x y)

# Boolean logic
(and (> x 0) (< x 10))
(or (== y 1) (== y 2))
(not flag)
```

### 4. Rich Type System

```nano
# Primitive types
int      # Signed integer (64-bit)
float    # Floating point (64-bit)
bool     # Boolean (true/false)
string   # C-style string (null-terminated)
bstring  # Binary string (length-explicit, UTF-8 aware)
void     # No return value

# Composite types
struct   # Product types (record types)
enum     # Named integer constants
union    # Tagged unions (sum types)

# Generic types
List<T>  # Dynamic lists with type parameter

# Type syntax is always: name: type
let x: int = 42
let name: string = "nano"
let flag: bool = true
let data: bstring = (bstr_new "Hello 世界")
```

### 5. Functions

```nano
# Function definition
fn function_name(param1: type1, param2: type2) -> return_type {
    # body
}

# Functions must:
# 1. Have explicit parameter types
# 2. Have explicit return type
# 3. Have a shadow-test block
# 4. Return a value (unless return type is void)

fn multiply(x: int, y: int) -> int {
    return (* x y)
}

shadow multiply {
    assert (== (multiply 2 3) 6)
    assert (== (multiply 0 5) 0)
    assert (== (multiply -2 3) -6)
}
```

### 6. Variables

```nano
# Immutable by default
let x: int = 10

# Mutable (requires 'mut' keyword)
let mut counter: int = 0
set counter (+ counter 1)
```

### 7. Control Flow

```nano
# If expression (always returns a value)
let result: int = if (> x 0) {
    42
} else {
    -1
}

# While loop
while (< i 10) {
    print i
    set i (+ i 1)
}

# For loop (sugar for while)
for i in (range 0 10) {
    print i
}
```

### 8. Built-in Functions (49+ total)

**Core I/O (3):**
```nano
print       # Print to stdout
println     # Print with newline
assert      # Runtime assertion (used in shadow-tests)
```

**String Operations (18):**
```nano
# Basic string ops
str_length str_concat str_substring str_contains str_equals

# Character access & classification
char_at string_from_char is_digit is_alpha is_alnum
is_whitespace is_upper is_lower

# Type conversions
int_to_string string_to_int digit_value
char_to_lower char_to_upper
```

**Binary String Operations (12):**
```nano
# Binary strings - length-explicit, UTF-8 aware
bstr_new bstr_new_binary bstr_length bstr_concat
bstr_substring bstr_equals bstr_byte_at bstr_to_cstr
bstr_validate_utf8 bstr_utf8_length bstr_utf8_char_at bstr_free
```

**Math Operations (11):**
```nano
abs min max sqrt pow floor ceil round sin cos tan
```

**Array Operations (4):**
```nano
at array_length array_new array_set
```

**Dynamic Lists (per type):**
```nano
List_<T>_new List_<T>_push List_<T>_pop List_<T>_get
List_<T>_length List_<T>_clear
# Available for: int, string, and user-defined types
```

**OS/System (3):**
```nano
getcwd getenv range
```

See [docs/STDLIB.md](docs/STDLIB.md) for complete documentation.

### 9. Operators (All Prefix)

```nano
# Arithmetic
(+ a b)     # Addition
(- a b)     # Subtraction
(* a b)     # Multiplication
(/ a b)     # Division
(% a b)     # Modulo

# Comparison
(== a b)    # Equal
(!= a b)    # Not equal
(< a b)     # Less than
(<= a b)    # Less or equal
(> a b)     # Greater than
(>= a b)    # Greater or equal

# Logical
(and a b)   # Logical AND
(or a b)    # Logical OR
(not a)     # Logical NOT
```

### 10. Module System & Imports

nanolang supports a module system for organizing code across multiple files and integrating with C libraries.

```nano
# Import a nanolang module
import "modules/sdl/sdl.nano"

# Use functions from the module
fn main() -> int {
    let window: int = (SDL_CreateWindow "Game" 100 100 800 600 0)
    (SDL_Quit)
    return 0
}

shadow main {
    # Skip extern-based tests
}
```

**Module Features:**
- Automatic C compilation for FFI modules
- Dependency management via `module.json`
- pkg-config integration
- Cached compilation

See [docs/MODULE_SYSTEM.md](docs/MODULE_SYSTEM.md) for details.

### 11. Structs, Enums, and Unions

**Structs (Product Types):**
```nano
struct Point {
    x: int,
    y: int
}

let p: Point = Point { x: 10, y: 20 }
let x_val: int = p.x
```

**Enums (Named Constants):**
```nano
enum Status {
    Pending = 0,
    Active = 1,
    Complete = 2
}

let s: int = Status.Active
```

**Unions (Tagged Unions / Sum Types):**
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

# Pattern matching on unions
match result {
    Ok(v) => (println v.value),
    Error(e) => (println e.message)
}
```

### 12. Generics & First-Class Functions

**Generic Lists with Monomorphization:**
```nano
# Create typed lists
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)
let first: int = (List_int_get numbers 0)

# Lists work with user-defined types
struct Point { x: int, y: int }
let points: List<Point> = (List_Point_new)
```

**First-Class Functions:**
```nano
fn double(x: int) -> int {
    return (* x 2)
}

# Assign function to variable
let f: fn(int) -> int = double

# Pass function as parameter
fn apply_twice(op: fn(int) -> int, x: int) -> int {
    return (op (op x))
}

let result: int = (apply_twice double 5)  # result = 20
```

### 13. AI/ML Support with ONNX

nanolang can run neural network models via ONNX Runtime:

```nano
import "modules/onnx/onnx.nano"

fn main() -> int {
    let model: int = (onnx_load_model "resnet50.onnx")
    if (< model 0) {
        (println "Failed to load model")
        return 1
    }
    
    # Run inference (see AI_ML_GUIDE.md for details)
    (onnx_free_model model)
    return 0
}

shadow main {
    # Skip extern tests
}
```

**Supported:**
- PyTorch, TensorFlow, scikit-learn models (converted to ONNX)
- Image classification, NLP, object detection
- CPU-only (no GPU required)

See [docs/AI_ML_GUIDE.md](docs/AI_ML_GUIDE.md) for complete guide.

## Complete Example

```nano
# Calculate factorial with shadow-tests
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}

# Check if number is prime
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
    assert (== (is_prime 2) true)
    assert (== (is_prime 3) true)
    assert (== (is_prime 4) false)
    assert (== (is_prime 17) true)
    assert (== (is_prime 1) false)
}

fn main() -> int {
    print "Factorial of 6:"
    print (factorial 6)
    
    print "Is 13 prime?"
    print (is_prime 13)
    
    return 0
}

shadow main {
    # Main doesn't need extensive tests,
    # but shadow blocks are still required
    assert (== (main) 0)
}
```

## Grammar (EBNF-like)

```
program        = { function | shadow_test }

function       = "fn" identifier "(" params ")" "->" type block

params         = [ param { "," param } ]
param          = identifier ":" type

shadow_test    = "shadow" identifier block

block          = "{" { statement } "}"

statement      = let_stmt
               | set_stmt
               | while_stmt
               | for_stmt
               | return_stmt
               | expr_stmt

let_stmt       = "let" ["mut"] identifier ":" type "=" expression
set_stmt       = "set" identifier expression
while_stmt     = "while" expression block
for_stmt       = "for" identifier "in" expression block
return_stmt    = "return" expression
expr_stmt      = expression

expression     = literal
               | identifier
               | if_expr
               | call_expr
               | prefix_op

if_expr        = "if" expression block "else" block
call_expr      = "(" operator_or_fn { expression } ")"
prefix_op      = "(" operator expression expression ")"

literal        = integer | float | string | bool
type           = "int" | "float" | "bool" | "string" | "void"
operator       = "+" | "-" | "*" | "/" | "%" 
               | "==" | "!=" | "<" | "<=" | ">" | ">="
               | "and" | "or" | "not"
```

## Compilation Model

nanolang compiles to C for performance and portability:

1. **Parse**: Source → AST
2. **Validate**: Check types, shadow-tests, return paths
3. **Test**: Run shadow-tests during compilation
4. **Transpile**: AST → C code
5. **Compile**: C code → Native binary

### Why C Transpilation?

- **Portability**: C runs everywhere
- **Performance**: Native speed
- **Self-hosting**: Eventually, nanolang can compile itself
- **Interop**: Easy to call C libraries
- **Tooling**: Leverage mature C toolchains

## Design Review

**Independent Analysis:** nanolang achieves an **8.5/10 (A-)** in high-level language design for LLM-friendly code generation.

**Key Strengths:**
- ✅ Prefix notation eliminates operator precedence errors (10/10)
- ✅ Mandatory shadow-tests enforce compile-time correctness (10/10)
- ✅ Dual execution model (interpreter + transpiler) is innovative (10/10)
- ✅ Minimal syntax reduces LLM confusion (9/10)

**Critical Issues - NOW FIXED:** ✅
- ✅ **FIXED:** Duplicate function detection now prevents namespace collisions
- ✅ **FIXED:** Built-in shadowing prevention protects 44 standard library functions
- ✅ **ADDED:** Similar name warnings catch typos (Levenshtein distance ≤ 2)

See [Namespace Fixes Document](docs/NAMESPACE_FIXES.md) for details.

📊 **See [Design Review Summary](docs/REVIEW_SUMMARY.md)** for executive summary  
📖 **See [Full Design Review](docs/LANGUAGE_DESIGN_REVIEW.md)** for detailed analysis

## Architectural Elegance Review

**Assessment:** nanolang maintains architectural elegance while adding self-hosting features. **Grade: B+ (8.2/10)**

**Key Finding:** The language has grown appropriately from "nano" to "small" but maintains core principles:
- ✅ Still LLM-friendly (core spec fits in ~10KB)
- ✅ Still minimal (18 keywords vs 32 in C, 25 in Go)
- ✅ Shadow tests still mandatory
- ✅ No pointers, immutability by default maintained
- ✅ Can write a compiler in ~5,000 lines

📊 **See [Architecture Analysis](docs/ARCHITECTURE_ANALYSIS.md)** for detailed architecture review

## Design Rationale

### Why Prefix Notation?

Prefix notation eliminates operator precedence confusion:

```
# Infix (requires precedence knowledge):
a + b * c        # Is this (a + b) * c or a + (b * c)?
x == y && z      # Precedence of == vs &&?

# Prefix (crystal clear):
(+ a (* b c))    # Unambiguous
(and (== x y) z) # Clear nesting
```

### Why Shadow-Tests?

1. **Immediate Feedback**: Tests run during compilation
2. **Documentation**: Tests show expected behavior
3. **Confidence**: Code without tests doesn't compile
4. **LLM-friendly**: Forces LLMs to think about test cases

### Why Minimal?

A small language is:
- Easier to learn
- Easier to implement
- Easier for LLMs to generate correctly
- Easier to reason about
- Less prone to bugs

## Building nanolang

### Build Commands

```bash
# Build compiler and interpreter (default)
make

# Run test suite
make test

# Build all examples
make examples

# Build with memory sanitizers (for development)
make sanitize

# Build with coverage instrumentation
make coverage

# Generate HTML coverage report
make coverage-report

# Run valgrind memory checks
make valgrind

# Run static analysis
make lint

# Quick check (build + test)
make check

# Install to system (default: /usr/local/bin)
make install

# Uninstall from system
make uninstall

# Clean build artifacts
make clean

# Show all available targets
make help
```

### Interpreter vs Compiler

nanolang provides two execution modes:

**Interpreter (`bin/nano`)**:
- Instant execution, no compilation step
- Great for development and debugging
- Built-in tracing support (`--trace-all`, `--trace-function=name`, etc.)
- Runs shadow-tests automatically
- Use for rapid iteration

**Compiler (`bin/nanoc`)**:
- Transpiles to C, then compiles to native binary
- Maximum performance
- Produces standalone executables
- Runs shadow-tests during compilation
- Use for production builds

## Getting Started

### 1. Install nanolang

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make
```

This builds two executables:
- `bin/nano` - Interpreter (instant execution)
- `bin/nanoc` - Compiler (generates native binaries)

### 2. Write Your First Program

Create `hello.nano`:

```nano
fn greet(name: string) -> void {
    (println "Hello, ")
    (println name)
}

shadow greet {
    # For void functions, just verify they don't crash
    greet "World"
    greet "nanolang"
}

fn main() -> int {
    greet "World"
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### 3. Run It Two Ways

**Option A: Interpreter (instant execution, great for development)**

```bash
./bin/nano hello.nano
```

**Option B: Compile to native binary (maximum performance)**

```bash
./bin/nanoc hello.nano -o hello
./hello
```

Both produce the same output:
```
Hello, 
World
```

### 4. Next Steps

- **Learn the syntax**: Read [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Explore examples**: Check out `examples/` directory
- **Try the tutorial**: Follow [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- **Build something**: See `examples/game_of_life.nano` or `examples/calculator.nano`

## Project Status

**Current Status**: ✅ Production Ready - Core compiler complete

### Roadmap

- [x] Language specification
- [x] Lexer implementation
- [x] Parser implementation
- [x] Type checker
- [x] Shadow-test runner
- [x] C transpiler
- [x] Standard library (24 functions)
- [x] Command-line tools (nanoc + nano)
- [x] **Self-hosting foundation** - ✅ 6/6 essential features complete
- [ ] **Self-hosting compiler** - Phase 2: Rewrite compiler components in nanolang

**Next major milestone:** Self-hosting Phase 2 (estimated 13-18 weeks)

See [planning/SELF_HOSTING_IMPLEMENTATION_PLAN.md](planning/SELF_HOSTING_IMPLEMENTATION_PLAN.md) for the roadmap to implementing nanolang in nanolang itself. All 6 essential features are complete: structs, enums, dynamic lists, file I/O, advanced string operations, and system execution. See [docs/SELF_HOSTING_CHECKLIST.md](docs/SELF_HOSTING_CHECKLIST.md) for status details.

## Learning Path

**Complete Beginner?**
1. Start with [Quick Start](#quick-start) above
2. Read [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - basics tutorial
3. Try examples: `hello.nano`, `factorial.nano`, `calculator.nano`

**Know Programming?**
1. Skim [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - syntax cheat sheet
2. Review [Language Features](#language-features) section above
3. Explore advanced examples: `game_of_life.nano`, `boids_complete.nano`

**Want Advanced Features?**
- **Modules**: [docs/MODULE_SYSTEM.md](docs/MODULE_SYSTEM.md)
- **FFI/C Integration**: [docs/EXTERN_FFI.md](docs/EXTERN_FFI.md)
- **AI/ML**: [docs/AI_ML_GUIDE.md](docs/AI_ML_GUIDE.md)
- **Type System**: [docs/SPECIFICATION.md](docs/SPECIFICATION.md)

## Contributing

nanolang welcomes contributions! The language is designed to be:
- **Simple enough** that an LLM can implement features
- **Clear enough** that humans can review implementations
- **Complete enough** to be useful

**Ways to contribute:**
- Add examples or tutorials
- Improve documentation
- Report bugs or suggest features
- Implement new stdlib functions
- Write modules (SDL, OpenGL, etc.)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - See LICENSE file for details.

## Why "nanolang"?

Because it's nano-sized, designed for the modern (AI) age, and every good minimal language needs a memorable name! 

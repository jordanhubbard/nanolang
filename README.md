# nanolang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-100%25%20self--hosting-success.svg)
![Type System](https://img.shields.io/badge/type%20system-100%25%20functional-success.svg)
![Language](https://img.shields.io/badge/language-compiled-success.svg)

**A minimal, LLM-friendly programming language with mandatory testing and unambiguous syntax.**

NanoLang transpiles to C for native performance while providing a clean, modern syntax optimized for both human readability and AI code generation.

> **Self-hosting:** NanoLang supports true self-hosting via a Stage 0 → Stage 1 → Stage 2 bootstrap (`make bootstrap`); see [planning/SELF_HOSTING.md](planning/SELF_HOSTING.md).

## Quick Start

### Install

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build
```

**Note for FreeBSD/BSD users:** Use `gmake` instead of `make` (requires GNU Make).

This builds the compiler:
- `bin/nanoc` - NanoLang compiler (transpiles to C)

### Hello World

Create `hello.nano`:

```nano
fn greet(name: string) -> string {
    return (+ "Hello, " name)
}

shadow greet {
    assert (str_equals (greet "World") "Hello, World")
}

fn main() -> int {
    (println (greet "World"))
    return 0
}

shadow main {
    assert true
}
```

Run it:

```bash
# Compile to native binary
./bin/nanoc hello.nano -o hello
./hello
```

## Interactive Development 🎮

NanoLang includes **two interactive development tools** for learning and experimentation:

### 1. Web Playground (Recommended for Beginners)

Browser-based playground inspired by Swift Playgrounds:

```bash
# Build and start the playground server
./bin/nanoc examples/playground/playground_server.nano -o bin/playground
./bin/playground

# Open in your browser
open http://localhost:8080
```

**Features:**
- 📝 Interactive code editor
- 📚 10+ example programs
- ⚡ Real-time syntax validation
- 📋 Copy/download functionality
- 🎨 Beautiful modern UI

See **[examples/playground/README.md](examples/playground/README.md)** for full documentation.

### 2. Terminal REPL

Full-featured command-line REPL:

```bash
# Build the REPL
./bin/nanoc examples/language/full_repl.nano -o bin/repl

# Launch it
./bin/repl
```

### Features

- ✅ **Persistent variables** - Define variables that persist across evaluations
- ✅ **Function definitions** - Define functions with support for recursion
- ✅ **Module imports** - Import and use modules interactively
- ✅ **Multi-line input** - Smart continuation prompts for complex code
- ✅ **Multi-type support** - Evaluate int, float, string, and bool expressions
- ✅ **Session management** - Commands to inspect and manage your session

### Example Session

```nano
$ ./bin/repl

NanoLang Full-Featured REPL
============================
Variables: let x: int = 42
Functions: fn double(x: int) -> int { return (* x 2) }
Imports: from "std/math" import sqrt
Types: :int, :float, :string, :bool
Commands: :vars, :funcs, :imports, :clear, :quit

nano> let x: int = 42
Defined: x

nano> let y: float = 3.14159
Defined: y

nano> (+ x 10)
=> 52

nano> :float (* y 2.0)
=> 6.28318

nano> fn factorial(n: int) -> int {
....>     if (<= n 1) {
....>         return 1
....>     } else {
....>         return (* n (factorial (- n 1)))
....>     }
....> }
Defined: factorial(n: int) -> int

nano> (factorial 5)
=> 120

nano> :vars
Defined variables: x, y

nano> :funcs
Defined functions: factorial(n: int) -> int

nano> :quit
Goodbye!
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `:vars` | List all defined variables |
| `:funcs` | List all defined functions |
| `:imports` | List all imported modules |
| `:clear` | Clear entire session (variables, functions, imports) |
| `:quit` | Exit REPL (or press Ctrl-D) |

### Type-Specific Evaluation

By default, expressions are evaluated as integers. Use type prefixes for other types:

```nano
nano> (+ 1 2)           # Default: int
=> 3

nano> :float (* 3.14 2.0)
=> 6.28

nano> :string (+ "Hello, " "World")
=> Hello, World

nano> :bool (> 5 3)
=> true
```

### Multi-Line Input

The REPL automatically detects incomplete input and shows a continuation prompt:

```nano
nano> fn double(x: int) -> int {
....>     return (* x 2)
....> }
Defined: double(x: int) -> int

nano> if (> x 10) {
....>     (println "big")
....> } else {
....>     (println "small")
....> }
=> ...
```

### Use Cases

- **Learning NanoLang** - Try syntax and features interactively
- **Quick calculations** - Use as a calculator with variables
- **Prototyping** - Test ideas before writing full programs
- **Debugging** - Experiment with expressions and functions
- **Teaching** - Demonstrate language features live

See `examples/language/` for REPL source code and implementation details.

## Platform Support

### Tier 1: Fully Supported ✅
NanoLang is actively tested and supported on:

- **Ubuntu 22.04+** (x86_64)
- **Ubuntu 24.04** (ARM64) - Raspberry Pi, AWS Graviton, etc.
- **macOS 14+** (ARM64/Apple Silicon)
- **FreeBSD**

### Tier 2: Windows via WSL 🪟
**Windows 10/11 users:** NanoLang runs perfectly on Windows via WSL2 (Windows Subsystem for Linux).

#### Install WSL2:
```powershell
# In PowerShell (as Administrator)
wsl --install -d Ubuntu
```

After installation, restart your computer, then:

```bash
# Inside WSL Ubuntu terminal
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make
./bin/nanoc examples/language/nl_hello.nano -o hello
./hello
```

**Why WSL?** NanoLang's dependencies (SDL2, ncurses, pkg-config) are Unix/POSIX libraries. WSL2 provides a full Linux environment with near-native performance on Windows.

**Note:** Native Windows binaries (`.exe`) are not currently supported, but may be added in a future release via cross-compilation.

### Tier 3: Experimental 🧪
These platforms should work but are not actively tested in CI:

- macOS Intel (via Rosetta 2 on Apple Silicon, or native on older Macs)
- Other Linux distributions (Arch, Fedora, Debian, etc.)
- OpenBSD (requires manual dependency installation)

## Key Features

- **Prefix Notation** - No operator precedence: `(+ a (* b c))` is always clear
- **Mandatory Testing** - Every function requires a `shadow` test block
- **Static Typing** - Catch errors at compile time
- **Generic Types** - Generic unions like `Result<T, E>` for error handling
- **Compiled Language** - Transpiles to C for native performance
- **Immutable by Default** - Use `let mut` for mutability
- **C Interop** - Easy FFI via modules with automatic package management
- **Module System** - Automatic dependency installation via `module.json`
- **Standard Library** - Growing stdlib with `Result<T,E>`, string ops, math, and more

## Documentation

### Learning Path

0. **[User Guide (HTML)](https://jordanhubbard.github.io/nanolang/)** - Progressive tutorial + executable snippets
1. **[Getting Started](docs/GETTING_STARTED.md)** - 15-minute tutorial
2. **[Quick Reference](docs/QUICK_REFERENCE.md)** - Syntax cheat sheet  
3. **[Language Specification](docs/SPECIFICATION.md)** - Complete reference
4. **[Examples](examples/README.md)** - Working examples (all runnable)

### Key Topics

- **[Standard Library](docs/STDLIB.md)** - Built-in functions
- **[Type Inference](docs/TYPE_INFERENCE.md)** - What can/cannot be inferred
- **[Module System](docs/MODULE_SYSTEM.md)** - Creating and using modules
- **[FFI Guide](docs/EXTERN_FFI.md)** - Calling C functions
- **[Shadow Tests](docs/SHADOW_TESTS.md)** - Testing philosophy
- **[Code Coverage](docs/COVERAGE.md)** - Coverage reporting
- **[All Documentation](docs/DOCS_INDEX.md)** - Complete index

## Language Overview

### Syntax Basics

```nano
# Variables (immutable by default)
let x: int = 42
let mut counter: int = 0

# Functions with mandatory tests
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add -1 1) 0)
}

# Control flow
if (> x 0) {
    (println "positive")
} else {
    (println "negative or zero")
}

# Loops
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

### Why Prefix Notation?

No operator precedence to remember:

```nano
# Crystal clear - no ambiguity
(+ a (* b c))           # a + (b * c)
(and (> x 0) (< x 10))  # x > 0 && x < 10
(/ (+ a b) (- c d))     # (a + b) / (c - d)
```

### Type System

```nano
# Primitives
int, float, bool, string, void

# Composite types
struct Point { x: int, y: int }
enum Status { Pending = 0, Active = 1, Complete = 2 }

# Generic lists
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)

# First-class functions
fn double(x: int) -> int { return (* x 2) }
let f: fn(int) -> int = double

# Generic unions (NEW!)
union Result<T, E> {
    Ok { value: T },
    Err { error: E }
}

let success: Result<int, string> = Result.Ok { value: 42 }
let failure: Result<int, string> = Result.Err { error: "oops" }
```

### Standard Library

NanoLang includes a growing standard library:

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

fn main() -> int {
    let result: Result<int, string> = (divide 10 2)

    /* Note: Result helper functions (is_ok/unwrap/etc) are planned once
     * generic functions are supported. For now, use match.
     */
    match result {
        Ok(v) => (println v.value),
        Err(e) => (println e.error)
    }

    return 0
}
```

## Examples

### Core Examples

- **[hello.nano](examples/language/nl_hello.nano)** - Basic structure
- **[calculator.nano](examples/language/nl_calculator.nano)** - Arithmetic operations
- **[factorial.nano](examples/language/nl_factorial.nano)** - Recursion
- **[fibonacci.nano](examples/language/nl_fibonacci.nano)** - Multiple algorithms
- **[primes.nano](examples/language/nl_primes.nano)** - Prime number sieve

### Game Examples

- **[snake_ncurses.nano](examples/terminal/ncurses_snake.nano)** - Classic snake with NCurses UI
- **[game_of_life_ncurses.nano](examples/terminal/ncurses_game_of_life.nano)** - Conway's Game of Life
- **[asteroids_complete.nano](examples/games/sdl_asteroids.nano)** - Full Asteroids game (SDL)
- **[checkers.nano](examples/games/sdl_checkers.nano)** - Checkers with AI (SDL)
- **[boids_sdl.nano](examples/graphics/sdl_boids.nano)** - Flocking simulation (SDL)

See **[examples/README.md](examples/README.md)** for the complete list.

## Modules

NanoLang includes several modules with **automatic dependency management**:

### Graphics & Games
- **ncurses** - Terminal UI (interactive games, text interfaces)
- **sdl** - 2D graphics, windows, input (`brew install sdl2`)
- **sdl_mixer** - Audio playback (`brew install sdl2_mixer`)
- **sdl_ttf** - Font rendering (`brew install sdl2_ttf`)
- **glfw** - OpenGL window management (`brew install glfw`)

Modules automatically install dependencies via package managers (Homebrew, apt, etc.) when first used. See **[docs/MODULE_SYSTEM.md](docs/MODULE_SYSTEM.md)** for details.

## Building & Testing

```bash
# Build (3-stage component bootstrap)
make build

# Run full test suite
make test

# Quick test (language tests only)
make test-quick

# Build all examples
make examples

# Launch the examples browser
make examples-launcher

# Generate code coverage report (requires: brew install lcov)
make coverage-report

# Validate user guide snippets (extract → compile → run)
make userguide-check

# Build static HTML for the user guide
make userguide-html
# Options:
#   CMD_TIMEOUT=600              # per-command timeout (seconds)
#   USERGUIDE_TIMEOUT=600        # build timeout (seconds)
#   USERGUIDE_BUILD_API_DOCS=1   # regenerate API reference
#   NANO_USERGUIDE_HIGHLIGHT=0   # disable highlighting (CI default)

# Serve the user guide locally (dev)
make -C userguide serve

# Clean build
make clean

# Install to /usr/local/bin (override with PREFIX=...)
sudo make install
```

On BSD systems (FreeBSD/OpenBSD/NetBSD), use GNU make: `gmake build`, `gmake test`, etc.

## Teaching LLMs NanoLang

NanoLang is designed to be LLM-friendly with unambiguous syntax and mandatory testing. To teach an AI system to code in NanoLang:

### For LLM Training

- **[MEMORY.md](MEMORY.md)** - Complete LLM training reference with patterns, idioms, debugging workflows, and common errors
- **[spec.json](spec.json)** - Formal language specification (types, stdlib, syntax, operations)
- **[Examples](examples/README.md)** - Runnable examples demonstrating all features

### Quick LLM Bootstrap

1. Read `MEMORY.md` first - covers syntax, patterns, testing, debugging
2. Reference `spec.json` for stdlib functions and type details  
3. Study examples for idiomatic usage patterns

The combination of MEMORY.md (practical guidance) + spec.json (formal reference) provides complete coverage for code generation and understanding.

## Contributing

We welcome contributions! Areas where you can help:

- Add examples and tutorials
- Improve documentation
- Report bugs or suggest features
- Create new modules
- Implement standard library functions

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

## Project Status

**Current**: Production-ready compiler with full self-hosting support.

### Completed Features

- ✅ Complete language implementation (lexer, parser, typechecker, transpiler)
- ✅ Compiled language (transpiles to C for native performance)
- ✅ Static typing with inference
- ✅ Structs, enums, unions, generics
- ✅ Module system with auto-dependency management
- ✅ 72 standard library functions (see spec.json)
- ✅ 90+ working examples
- ✅ Shadow-test framework
- ✅ FFI support for C libraries
- ✅ Memory safety features

See **[docs/ROADMAP.md](docs/ROADMAP.md)** for future plans.

## Why NanoLang?

NanoLang solves three problems:

1. **LLM Code Generation** - Unambiguous syntax reduces AI errors
2. **Testing Discipline** - Mandatory tests improve code quality
3. **Simple & Fast** - Minimal syntax, native performance

**Design Philosophy:**
- Minimal syntax (18 keywords vs 32 in C)
- One obvious way to do things
- Tests are part of the language, not an afterthought
- Transpile to C for maximum compatibility

## License

Apache License 2.0 - See LICENSE file for details.

## Links

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/jordanhubbard/nanolang/issues)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

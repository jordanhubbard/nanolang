# nanolang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-100%25%20self--hosting-success.svg)
![Type System](https://img.shields.io/badge/type%20system-100%25%20functional-success.svg)

**A minimal, LLM-friendly programming language with mandatory testing and unambiguous syntax.**

NanoLang transpiles to C for native performance while providing a clean, modern syntax optimized for both human readability and AI code generation.

> **🎉 NEW**: NanoLang has achieved **TRUE 100% SELF-HOSTING** with a fully functional type system! The compiler is now written entirely in NanoLang and successfully compiles itself through multiple bootstrap stages. See [BOOTSTRAP_VICTORY.md](BOOTSTRAP_VICTORY.md) for details.

## Quick Start

### Install

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make
```

This builds two tools:
- `bin/nano` - Interactive interpreter
- `bin/nanoc` - Compiler (transpiles to C)

### Hello World

Create `hello.nano`:

```nano
fn greet(name: string) -> void {
    (println (str_concat "Hello, " name))
}

shadow greet {
    greet "World"
    greet "NanoLang"
}

fn main() -> int {
    greet "World"
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Run it:

```bash
# Option 1: Interpret (instant execution)
./bin/nano hello.nano

# Option 2: Compile to native binary
./bin/nanoc hello.nano -o hello
./hello
```

## Key Features

- **Prefix Notation** - No operator precedence: `(+ a (* b c))` is always clear
- **Mandatory Testing** - Every function requires a `shadow` test block
- **Static Typing** - Catch errors at compile time
- **Immutable by Default** - Use `let mut` for mutability
- **C Interop** - Easy FFI via modules with automatic package management
- **Module System** - Automatic dependency installation via `module.json`

## Documentation

### Learning Path

1. **[Getting Started](docs/GETTING_STARTED.md)** - 15-minute tutorial
2. **[Quick Reference](docs/QUICK_REFERENCE.md)** - Syntax cheat sheet  
3. **[Language Specification](docs/SPECIFICATION.md)** - Complete reference
4. **[Examples](examples/)** - 90+ working examples

### Key Topics

- **[Standard Library](docs/STDLIB.md)** - 49+ built-in functions
- **[Module System](docs/MODULE_SYSTEM.md)** - Creating and using modules
- **[FFI Guide](docs/EXTERN_FFI.md)** - Calling C functions
- **[Shadow Tests](docs/SHADOW_TESTS.md)** - Testing philosophy
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
```

## Examples

### Core Examples

- **[hello.nano](examples/hello.nano)** - Basic structure
- **[calculator.nano](examples/calculator.nano)** - Arithmetic operations
- **[factorial.nano](examples/factorial.nano)** - Recursion
- **[fibonacci.nano](examples/fibonacci.nano)** - Multiple algorithms
- **[primes.nano](examples/primes.nano)** - Prime number sieve

### Game Examples

- **[snake_ncurses.nano](examples/snake_ncurses.nano)** - Classic snake with NCurses UI
- **[game_of_life_ncurses.nano](examples/game_of_life_ncurses.nano)** - Conway's Game of Life
- **[asteroids_complete.nano](examples/asteroids_complete.nano)** - Full Asteroids game (SDL)
- **[checkers.nano](examples/checkers.nano)** - Checkers with AI (SDL)
- **[boids_sdl.nano](examples/boids_sdl.nano)** - Flocking simulation (SDL)

See **[examples/README.md](examples/README.md)** for the complete list.

## Modules

NanoLang includes several modules with **automatic dependency management**:

### Graphics & Games
- **ncurses** - Terminal UI (interactive games, text interfaces)
- **sdl** - 2D graphics, windows, input (`brew install sdl2`)
- **sdl_mixer** - Audio playback (`brew install sdl2_mixer`)
- **sdl_ttf** - Font rendering (`brew install sdl2_ttf`)
- **glfw** - OpenGL window management (`brew install glfw`)

### AI/ML
- **onnx** - Neural network inference (`brew install onnxruntime`)

Modules automatically install dependencies via package managers (Homebrew, apt, etc.) when first used. See **[docs/MODULE_SYSTEM.md](docs/MODULE_SYSTEM.md)** for details.

## Building & Testing

```bash
# Build compiler and interpreter
make

# Run test suite
make test

# Build all examples
make examples

# Clean build
make clean

# Install to /usr/local/bin
sudo make install
```

## Teaching LLMs NanoLang

NanoLang is designed to be LLM-friendly with unambiguous syntax and mandatory testing. To teach an AI system to code in NanoLang:

### For LLM Training

- **[MEMORY.md](MEMORY.md)** - Complete LLM training reference with patterns, idioms, debugging workflows, and common errors
- **[spec.json](spec.json)** - Formal language specification (types, stdlib, syntax, operations)
- **[Examples](examples/)** - 90+ working examples demonstrating all features

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
- ✅ Dual execution (interpreter + compiler)
- ✅ Static typing with inference
- ✅ Structs, enums, unions, generics
- ✅ Module system with auto-dependency management
- ✅ 49+ standard library functions
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

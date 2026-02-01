# NanoLang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-100%25%20self--hosting-success.svg)

**A minimal, LLM-friendly programming language with mandatory testing and unambiguous syntax.**

NanoLang transpiles to C for native performance while providing a clean, modern syntax optimized for both human readability and AI code generation.

## Documentation

**[User Guide](https://jordanhubbard.github.io/nanolang/)** - Comprehensive tutorial with executable examples

Quick references:
- [Getting Started](docs/GETTING_STARTED.md) - 15-minute tutorial
- [Quick Reference](docs/QUICK_REFERENCE.md) - Syntax cheat sheet
- [Language Specification](docs/SPECIFICATION.md) - Complete reference
- [All Documentation](docs/DOCS_INDEX.md) - Full index

## Quick Start

```bash
# Clone and build
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build

# Create hello.nano
cat > hello.nano << 'EOF'
fn greet(name: string) -> string {
    return (+ "Hello, " name)
}

shadow greet {
    assert (== (greet "World") "Hello, World")
}

fn main() -> int {
    (println (greet "World"))
    return 0
}

shadow main { assert true }
EOF

# Compile and run
./bin/nanoc hello.nano -o hello
./hello
```

**BSD users:** Use `gmake` instead of `make`.

## Key Features

- **Prefix Notation** - No operator precedence ambiguity: `(+ a (* b c))`
- **Mandatory Testing** - Every function requires a `shadow` test block
- **Static Typing** - Catch errors at compile time
- **C Interop** - Easy FFI via modules with automatic dependency installation
- **Native Performance** - Transpiles to C

## Language Overview

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
}

# Control flow
if (> x 0) {
    (println "positive")
}

# Structs and enums
struct Point { x: int, y: int }
enum Status { Pending = 0, Active = 1 }

# Generic types
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)
```

## Building & Testing

```bash
make build          # Build compiler (bin/nanoc)
make test           # Run full test suite
make test-quick     # Quick language tests only
make examples       # Build all examples
```

## Examples

Run the examples launcher to browse and run all examples (requires SDL2):

```bash
cd examples && make launcher
```

Or compile individual examples:

```bash
./bin/nanoc examples/language/nl_fibonacci.nano -o fib
./fib
```

See **[examples/README.md](examples/README.md)** for the complete list including games (Snake, Asteroids, Checkers) and graphics demos.

## Interactive Development

### Web Playground

Browser-based code editor that compiles and runs NanoLang code:

```bash
# Build and run the playground server
./bin/nanoc examples/playground/playground_server.nano -o bin/playground
./bin/playground
# Open http://localhost:8080
```

Features: Interactive editor, 10+ examples, real-time compilation and execution.

### REPL (Experimental)

Basic expression evaluator:

```bash
./bin/nanoc examples/language/full_repl.nano -o bin/repl
./bin/repl
```

Note: The REPL is experimental. For reliable development, use the compiler directly.

## Platform Support

**Fully supported:**
- Ubuntu 22.04+ (x86_64, ARM64)
- macOS 14+ (Apple Silicon)
- FreeBSD

**Windows:** Use WSL2 with Ubuntu.

## For LLM Training

NanoLang is designed for AI code generation:

- **[MEMORY.md](MEMORY.md)** - LLM training reference with patterns and idioms
- **[spec.json](spec.json)** - Formal language specification

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

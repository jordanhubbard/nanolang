# NanoLang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-100%25%20self--hosting-success.svg)

**A minimal, LLM-friendly programming language with mandatory testing, unambiguous syntax, and formally verified semantics.**

NanoLang transpiles to C for native performance while also providing a custom virtual machine backend (NanoISA) with process-isolated FFI. Its core semantics are mechanically verified in Coq with zero axioms.

## 📖 Documentation

**→ [User Guide](https://jordanhubbard.github.io/nanolang/) ←** - **Start here!** Comprehensive tutorial with executable examples

**Additional Resources:**
- [Getting Started](docs/GETTING_STARTED.md) - 15-minute quick start
- [Quick Reference](docs/QUICK_REFERENCE.md) - Syntax cheat sheet
- [Language Specification](docs/SPECIFICATION.md) - Complete language reference
- [NanoISA VM Architecture](docs/NANOISA.md) - Virtual machine reference
- [Formal Verification](formal/README.md) - Coq proof suite
- [All Documentation](docs/DOCS_INDEX.md) - Full documentation index

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

- **Formally Verified Semantics** ⭐ - Type soundness, progress, determinism, and semantic equivalence proved in Coq (0 axioms)
- **NanoISA Virtual Machine** ⭐ - Custom stack-based VM with 178 opcodes, co-process FFI isolation, and optional daemon execution
- **Automatic Memory Management (ARC)** - Zero-overhead reference counting, no manual free() calls
- **LLM-Powered Autonomous Optimization** - Continuous profiling and automatic optimization loop
- **Dual Compilation** - Transpile to C for native performance, or compile to NanoISA bytecode for sandboxed execution
- **Dual Notation** - Both prefix `(+ a b)` and infix `a + b` operators supported
- **Mandatory Testing** - Every function requires a `shadow` test block
- **Static Typing** with type inference
- **C Interop** - Easy FFI via modules, with optional process isolation

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

## NanoISA Virtual Machine

NanoLang includes a complete virtual machine backend as an alternative to C transpilation:

```bash
# Compile to NanoISA bytecode and run
./bin/nano_virt hello.nano --run

# Compile to native binary (embeds VM + bytecode)
./bin/nano_virt hello.nano -o hello

# Emit raw .nvm bytecode, then execute separately
./bin/nano_virt hello.nano --emit-nvm -o hello.nvm
./bin/nano_vm hello.nvm

# Run with FFI isolation (external calls in separate process)
./bin/nano_vm --isolate-ffi hello.nvm
```

**Architecture:**
- **178 opcodes** - Stack machine with RISC/CISC hybrid instruction set
- **Co-process FFI** (`nano_cop`) - External function calls isolated in a separate process via RPC, so FFI crashes cannot take down the VM
- **VM daemon** (`nano_vmd`) - Optional persistent VM process for reduced startup latency
- **Trap model** - Pure-compute core separated from I/O operations, enabling potential FPGA acceleration
- **Reference-counted GC** - Deterministic memory management with scope-based auto-release

See [docs/NANOISA.md](docs/NANOISA.md) for the complete architecture reference.

## Formal Verification

NanoLang's core semantics (NanoCore) are mechanically verified in the Rocq Prover (Coq) with **zero axioms**:

- **Type Soundness** - Well-typed programs don't go wrong (preservation + progress)
- **Determinism** - Evaluation is a partial function
- **Semantic Equivalence** - Big-step and small-step semantics agree

The verified subset covers integers, booleans, strings, arrays, records, variants with pattern matching, closures, recursive functions, and mutable variables. See [formal/README.md](formal/README.md) for details.

```bash
cd formal/ && make    # Build all proofs (requires Rocq Prover >= 9.0)
```

## Building & Testing

```bash
make build          # Build compiler (bin/nanoc)
make nano_virt      # Build VM compiler (bin/nano_virt)
make nano_vm        # Build VM executor (bin/nano_vm)
make test           # Run full test suite
make test-quick     # Quick language tests only
make examples       # Build all examples
```

## Examples & Interactive Tools

**Web Playground** (recommended for learning):
```bash
./bin/nanoc examples/playground/playground_server.nano -o bin/playground
./bin/playground  # Open http://localhost:8080
```

**Examples Browser** (requires SDL2):
```bash
cd examples && make launcher
```

**Individual examples:**
```bash
./bin/nanoc examples/language/nl_fibonacci.nano -o fib && ./fib
```

See **[examples/README.md](examples/README.md)** for the complete catalog including games (Snake, Asteroids, Checkers) and graphics demos.

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

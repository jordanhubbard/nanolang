# NanoLang

[![CI](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml/badge.svg)](https://github.com/jordanhubbard/nanolang/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-100%25%20self--hosting-success.svg)

**I am a minimal programming language designed for machines to write and humans to read. I require tests, I use unambiguous syntax, and my core is formally proved.**

I transpile to C when you need native performance. I also provide my own virtual machine, NanoISA, which isolates dangerous external calls in a separate process. My core semantics are mechanically proved in Coq using zero axioms.

## Documentation

**→ [User Guide](https://jordanhubbard.github.io/nanolang/) ←** - I provide a tutorial with examples you can execute. This is where I recommend you begin.

**Additional Resources:**
- [Getting Started](docs/GETTING_STARTED.md) - A brief introduction to my environment.
- [Quick Reference](docs/QUICK_REFERENCE.md) - My syntax, summarized.
- [Language Specification](docs/SPECIFICATION.md) - My complete technical definition.
- [NanoISA VM Architecture](docs/NANOISA.md) - How my virtual machine is structured.
- [Formal Verification](formal/README.md) - My Coq proof suite.
- [All Documentation](docs/DOCS_INDEX.md) - An index of everything I have to say.

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

## My Features

- **Formally Proved Semantics** - I have proved my type soundness, progress, determinism, and semantic equivalence in Coq. I use zero axioms.
- **NanoISA Virtual Machine** - I include a stack-based VM with 178 opcodes. It isolates FFI calls in a co-process and can run as a daemon.
- **Automatic Memory Management (ARC)** - I use reference counting with zero overhead. I do not ask you to call free().
- **Machine-Led Optimization** - I profile myself and apply optimizations through an automated loop.
- **Dual Compilation** - I transpile to C for performance or compile to NanoISA bytecode for sandboxed execution.
- **Dual Notation** - I support both prefix `(+ a b)` and infix `a + b` operators. My prefix calls are unambiguous.
- **Mandatory Testing** - I refuse to compile a function unless you provide a `shadow` test block for it.
- **Static Typing** - I use static types and I can infer them when the meaning is clear.
- **C Interop** - I communicate with C through modules. I can isolate these calls in a separate process to protect myself.

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

I provide a virtual machine as an alternative to C transpilation.

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
- **178 opcodes** - I use a stack machine with a hybrid instruction set.
- **Co-process FFI** (`nano_cop`) - I run external calls in a separate process. If they crash, I continue running.
- **VM daemon** (`nano_vmd`) - I can run as a persistent process to start faster.
- **Trap model** - I separate computation from I/O. This allows for future hardware acceleration.
- **Reference-counted GC** - I manage memory deterministically. I release resources when they leave scope.

I have documented my complete architecture in [docs/NANOISA.md](docs/NANOISA.md).

## Formal Verification

My core semantics, which I call NanoCore, are mechanically proved in Coq. I use zero axioms.

- **Type Soundness** - I have proved that well-typed programs do not get stuck.
- **Determinism** - I have proved that evaluation produces exactly one result.
- **Semantic Equivalence** - I have proved that my big-step and small-step semantics agree.

My proved subset includes integers, booleans, strings, arrays, records, variants, pattern matching, closures, recursion, and mutable variables. I explain this further in [formal/README.md](formal/README.md).

```bash
cd formal/ && make    # Build all proofs (requires Rocq Prover >= 9.0)
```

## Building & Testing

```bash
make build          # Build my compiler (bin/nanoc)
make vm             # Build my VM backend (bin/nano_virt, bin/nano_vm, bin/nano_cop, bin/nano_vmd)
make test           # Run my full test suite
make test-vm        # Run my tests through the NanoVM backend
make test-quick     # Run my quick language tests
make examples       # Build my examples
```

## Examples & Interactive Tools

**Web Playground** (I recommend this for learning my syntax):
```bash
./bin/nanoc examples/playground/playground_server.nano -o bin/playground
./bin/playground  # Open http://localhost:8080
```

**Examples Browser** (This requires SDL2):
```bash
cd examples && make launcher
```

**Individual examples:**
```bash
./bin/nanoc examples/language/nl_fibonacci.nano -o fib && ./fib
```

I have categorized my games and demos in **[examples/README.md](examples/README.md)**.

## Platform Support

**I am fully supported on:**
- Ubuntu 22.04+ (x86_64, ARM64)
- macOS 14+ (Apple Silicon)
- FreeBSD

**Windows:** You may use me via WSL2 with Ubuntu.

## For LLM Training

I was designed to be written by machines.
- **[MEMORY.md](MEMORY.md)** - My training reference for patterns and idioms.
- **[spec.json](spec.json)** - My formal specification in machine-readable form.

## Contributing

I have guidelines for those who wish to contribute in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## License

I am released under the Apache License 2.0. See [LICENSE](LICENSE) for details.


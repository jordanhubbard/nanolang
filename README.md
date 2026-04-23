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
- **Machine-Led Optimization** - I profile myself and apply optimizations through an automated loop. I also run constant folding and dead-code elimination on the AST before code generation.
- **Multi-Target Compilation** - I transpile to C for native performance, emit WebAssembly (`--target wasm`), LLVM IR (`--target llvm`), PTX/CUDA (`--target ptx`), or RISC-V assembly (`--target riscv`). Each WASM output gets a source-map sidecar and can be signed with Ed25519.
- **Algebraic Effects** - I support typed, resumable effects with `effect`, `perform`, and `handle`. Side effects are explicit and composable.
- **Async / Await** - I lower `async fn` and `await` to a CPS state machine at compile time.
- **Dual Notation** - I support both prefix `(+ a b)` and infix `a + b` operators. My prefix calls are unambiguous.
- **Rich Pattern Matching** - I support match guards (`Ok(v) if v > 0 =>`), or-patterns (`| A | B =>`), wildcard `_`, and exhaustiveness checking.
- **Mandatory Testing** - I refuse to compile a function unless you provide a `shadow` test block for it.
- **Type Inference** - I use Hindley-Milner inference. Explicit annotations are optional when the type is unambiguous (`let x = 42`).
- **F-Strings and Pipes** - I support `f"Hello, {name}!"` string interpolation and `x |> f |> g` pipeline syntax.
- **C Interop** - I communicate with C through modules. I can isolate these calls in a separate process to protect myself.
- **VS Code Extension** - I ship a Language Server, Debug Adapter Protocol server, and a packaged `.vsix` extension with semantic tokens, format-on-save, and task integration.
- **Web Playground** - I include a browser-based CodeMirror 6 editor with share permalink and live evaluation.

## Language Overview

```nano
# Variables — immutable by default, type annotation optional when inferrable
let x: int = 42
let y = "hello"            # type inferred as string
let mut counter: int = 0

# Functions with mandatory shadow tests
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}

# F-string interpolation
let msg = f"Result: {(add 2 3)}"

# Pipe operator
let result = 5 |> add 3 |> double   # equivalent to double(add(3, 5))

# Control flow
if (> x 0) {
    (println "positive")
}

# Pattern matching with guards and or-patterns
union Shape { Circle { r: float }, Square { side: float }, Point {} }
match shape {
    Circle(c) if c.r > 0.0 => (println "circle"),
    | Square(_) | Point(_) => (println "other")
}

# Structs, enums, and generic types
struct Point { x: int, y: int }
enum Status { Pending = 0, Active = 1 }
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)

# Algebraic effects
effect Log { log : string -> void }
handle (perform Log.log "hi") with {
    Log.log(msg) -> { (println msg) }
}

# Parallel binding hint
par-let a = (compute_x)  b = (compute_y)  in (println (+ a b))
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

# Strip source-map debug info for production .nvm output
./bin/nano_virt hello.nano --emit-nvm --strip-debug -o hello.prod.nvm

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

## IDE & Debugger Support

I ship a Language Server (`bin/nanolang-lsp`) and a Debug Adapter (`bin/nanolang-dap`) for IDE integration.

```bash
make lsp   # Build bin/nanolang-lsp  (hover, go-to-definition, completion, diagnostics)
make dap   # Build bin/nanolang-dap  (breakpoints, step-through, variable inspection)
```

A VS Code extension is provided in `editors/vscode/`. It wires the LSP and DAP servers automatically.

```bash
# Compile to native C (default)
./bin/nanoc program.nano -o program

# Compile to WebAssembly (emits program.wasm + program.wasm.map source map)
./bin/nanoc program.nano --target wasm -o program.wasm

# Other backends
./bin/nanoc program.nano --target llvm  -o program.ll   # LLVM IR
./bin/nanoc program.nano --target ptx   -o program.ptx  # CUDA PTX
./bin/nanoc program.nano --target riscv -o program.s    # RISC-V assembly

# Sign and verify WASM modules
./bin/nanoc sign   program.wasm   # Signs with ~/.nanoc/signing.key
./bin/nanoc verify program.wasm   # Verifies embedded Ed25519 signature

# Export documentation from triple-slash comments
./bin/nanoc program.nano --doc-md -o program.md
```

## Building & Testing

```bash
make build          # Build my compiler (bin/nanoc)
make lsp            # Build my language server (bin/nanolang-lsp)
make dap            # Build my debugger (bin/nanolang-dap)
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

## The Totally True and Not At All Embellished History of NanoLang

### The continuing adventures of Jordan Hubbard and Sir Reginald von Fluffington III

> *Part 3 of an ongoing chronicle.  [← Part 2: sheme](https://github.com/jordanhubbard/sheme#the-totally-true-and-not-at-all-embellished-history-of-sheme) | [Part 4: Aviation →](https://github.com/jordanhubbard/Aviation#the-totally-true-and-not-at-all-embellished-history-of-aviation)*
> *Sir Reginald von Fluffington III appears throughout.  He does not endorse any of it.*

The programmer had, by this point, written a text editor in bash and a Scheme interpreter in bash, and had grown accustomed to things that arguably should not exist.  He was sitting at his desk — Sir Reginald von Fluffington III occupying his preferred position on the keyboard, which was all of it — when the programmer had what he described as "a thought" and Sir Reginald later categorized, by aggressively ignoring it, as "a cry for help."

"The problem," the programmer announced, to the room, to Sir Reginald, to the seventeen browser tabs he had open about formal verification, "is ambiguity.  Every language I use is ambiguous.  You write `f(x)` and nobody knows if it's a function call or a multiplication.  You write `1 + 2 * 3` and apparently this is a source of *controversy*.  And don't get me started on implicit type coercion, which should be a war crime."

Sir Reginald blinked once.  He had heard variations of this speech before.  They usually ended with something in the repository.

"I need a language," the programmer said, "that machines can write.  That has no ambiguity.  That requires tests.  That can be formally proved correct."  He paused, with the expression of a man who has just located a loophole in reality.  "And since no such language exists, I will build one."

Sir Reginald pushed a pen off the desk.  Slowly.  Deliberately.  With full awareness of the implications.

What followed was a period the programmer later called "necessary" and Sir Reginald filed under "this again."  A grammar was designed that distinguished prefix from infix without argument.  A type system was specified with the kind of precision usually reserved for bridge construction.  A virtual machine materialized, with 178 opcodes, which the programmer described as "minimal" and which the number 178 suggests was anything but.

"It needs formal proofs," the programmer said, sometime around the fourth week.  "In Coq.  Zero axioms."

Sir Reginald, who had been sleeping on the formal specification documents, shifted his weight slightly to cover the section on semantic equivalence.  He had learned that if he stayed on the important papers, progress was slowed.  He had also learned that this did not stop progress.  It merely made it slightly damp.

The mandatory test blocks — `shadow` functions that must accompany every function definition, without exception, or the compiler refuses to proceed — were added because the programmer was, as he put it, "tired of code that was never tested before it was written and therefore was never tested at all."  When asked whether `shadow` was a strange name for a test block, the programmer replied that it was "evocative."  Sir Reginald expressed no opinion.  His opinion on keywords was that they were all equally irrelevant to the procurement of tuna.

The language was named NanoLang.  It was minimal in the way that a Swiss watch is minimal: every component was necessary, the whole was smaller than it had any right to be, and explaining how it worked required considerably more time than most people had.

The language was also given a voice.  "It speaks in first person," the programmer noted, reviewing the README in which the language described its own features, "because it was designed to be written by machines, and machines should be able to parse its documentation without encountering ambiguity."  This was a reasonable position.  It did not make it less unsettling.

Sir Reginald walked across the keyboard.  Fourteen characters were appended to the type specification.  They were later identified as `jjjjjjjjjkkkk` and removed.

The formal proofs passed.  The VM ran.  The compiler compiled itself.  The programmer stared at this for a long moment in the way one stares at something one made that works, and experienced the specific species of pride that comes not from elegance but from completion.

"It's done," he said.

Sir Reginald knocked the coffee off the desk.  Not out of malice.  Out of a principled refusal to allow the programmer an uncontested moment.

As of this writing, NanoLang has been used in production by exactly one person, who also wrote it.  Sir Reginald continues to withhold his endorsement across all four projects, citing "procedural concerns," "insufficient tuna," "a general atmosphere of hubris," and, most recently, "aviation."


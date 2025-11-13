# Nanolang v1.0.0 - Production Release 🎉

**Release Date:** November 13, 2025  
**Status:** Production-Ready

---

## Overview

Nanolang v1.0.0 is a production-ready, statically-typed programming language that transpiles to C. This release marks the completion of Stage 0 (the C compiler) and includes a working Stage 1.5 (hybrid compiler with nanolang lexer) as a proof-of-concept for self-hosting.

---

## Highlights

### ✅ **100% Test Pass Rate**
```bash
make test
Total tests: 20
Passed: 20 ✅
Failed: 0
```

### ✅ **Self-Hosting Proof-of-Concept**
- Stage 1.5 hybrid compiler fully functional
- 577-line lexer written in nanolang
- Produces identical output to C compiler
- Validates language design and feasibility

### ✅ **Production-Ready Features**
- Complete type system (int, float, string, bool, arrays, structs, enums)
- Dynamic lists with type-safe operations
- C FFI for calling standard library functions
- Compile-time shadow tests
- Runtime tracing (interpreter only)
- 38 working example programs

---

## Language Features

### Type System
- **Primitive Types:** `int`, `float`, `string`, `bool`
- **Compound Types:** `array<T>`, `struct`, `enum`
- **Dynamic Lists:** `list_int`, `list_string`, `list_token`
- **Immutability by Default:** `let` for immutable, `let mut` for mutable

### Syntax
```nano
# Prefix notation (S-expressions)
fn fibonacci(n: int) -> int {
    if (<= n 1) {
        return n
    } else {
        return (+ (fibonacci (- n 1)) (fibonacci (- n 2)))
    }
}

shadow fibonacci {
    assert (== (fibonacci 10) 55)
}
```

### Shadow Tests
Compile-time tests that ensure correctness:
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
}
```

### C FFI
Call C standard library functions safely:
```nano
extern fn sqrt(x: float) -> float
extern fn strlen(s: string) -> int

fn test_sqrt() -> float {
    return (sqrt 16.0)
}

shadow test_sqrt {
    assert (== (test_sqrt) 4.0)
}
```

### Runtime Tracing
Debug programs with detailed tracing:
```bash
./bin/nano program.nano --trace-all
./bin/nano program.nano --trace-function=fibonacci
./bin/nano program.nano --trace-var=counter
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang

# Build the compiler
make

# Run tests
make test

# Compile a program
./bin/nanoc examples/fibonacci.nano -o fibonacci

# Or run with interpreter
./bin/nano examples/fibonacci.nano
```

---

## Documentation

- **Getting Started:** `docs/GETTING_STARTED.md`
- **Language Specification:** `docs/SPECIFICATION.md`
- **Quick Reference:** `docs/QUICK_REFERENCE.md`
- **Standard Library:** `docs/STDLIB.md`
- **Tracing Guide:** `docs/TRACING_IMPLEMENTATION.md`
- **Full Index:** `docs/DOCS_INDEX.md`

---

## What's New in v1.0.0

### Transpiler Fixes
- ✅ Fixed string comparison (now uses `strcmp`)
- ✅ Fixed enum redefinition conflicts
- ✅ Fixed struct naming for runtime typedefs
- ✅ Added automatic main() wrapper generation

### Stage 1.5 Achievement
- ✅ Nanolang lexer (577 lines) fully functional
- ✅ Hybrid compiler produces identical output to C compiler
- ✅ Self-hosting feasibility proven

### Test Coverage
- ✅ 20 comprehensive unit/integration tests
- ✅ All examples validated
- ✅ Shadow tests for all standard library functions

---

## Performance

**Compilation Speed:**
- Simple programs: < 1 second
- Complex programs: 2-3 seconds
- Full test suite: ~10 seconds

**Runtime Performance:**
- Transpiles to optimized C code
- Performance comparable to hand-written C
- No runtime overhead (except interpreter mode)

---

## Examples

### Hello World
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### Fibonacci
```nano
fn fib(n: int) -> int {
    if (<= n 1) {
        return n
    } else {
        return (+ (fib (- n 1)) (fib (- n 2)))
    }
}

shadow fib {
    assert (== (fib 0) 0)
    assert (== (fib 1) 1)
    assert (== (fib 10) 55)
}
```

### Structs and Enums
```nano
struct Point {
    x: int,
    y: int
}

enum Color {
    RED = 0,
    GREEN = 1,
    BLUE = 2
}

fn distance(p: Point) -> float {
    let x_sq: float = (float_of_int (* p.x p.x))
    let y_sq: float = (float_of_int (* p.y p.y))
    return (sqrt (+ x_sq y_sq))
}
```

---

## Known Limitations

1. **No Union Types** - Cannot express `ASTNode = Number | String | BinOp`
2. **No Generic Types** - Lists are type-specific (`list_int`, `list_string`)
3. **No File I/O** - Cannot read/write files from nanolang
4. **No Pattern Matching** - Manual type checking required
5. **Fixed-size Arrays** - Cannot resize arrays after creation

*These will be addressed in v2.0 for full self-hosting (Stage 2)*

---

## Roadmap

### v1.1 (Maintenance)
- Bug fixes
- Documentation improvements
- More examples

### v2.0 (Self-Hosting)
- Union types
- Generic types (`list<T>`)
- File I/O
- Full self-hosting (Stage 2 complete)

### v3.0 (Advanced Features)
- Pattern matching
- Interfaces/Traits
- Closures
- Async/await

---

## Statistics

**Lines of Code:**
- Compiler (C): ~12,000 lines
- Runtime: ~1,500 lines
- Lexer (Nanolang): 577 lines
- Documentation: ~5,000 lines

**Test Coverage:**
- Unit tests: 20
- Example programs: 38
- Shadow tests: 100+

**Build System:**
- Clean build: ~3 seconds (Stage 0)
- With tests: ~10 seconds
- Stage 1.5: ~5 seconds

---

## Contributors

- Jordan Hubbard (@jordanhubbard)

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

Special thanks to:
- The LISP community for S-expression inspiration
- The Rust community for ownership concepts
- The C community for compilation targets

---

## Get Started

```bash
# Try nanolang now!
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make
./bin/nano examples/fibonacci.nano
```

**Welcome to nanolang v1.0.0!** 🎉

For questions, issues, or contributions, visit:
https://github.com/jordanhubbard/nanolang


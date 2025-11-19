# nanolang Documentation Index

> **Note:** This page has been reorganized. For the complete documentation guide, see **[docs/README.md](README.md)**.

## Quick Navigation

### 🚀 Getting Started
- **[Getting Started Guide](GETTING_STARTED.md)** - Learn nanolang in 15 minutes
- **[Quick Reference](QUICK_REFERENCE.md)** - Syntax cheat sheet
- **[Examples](../examples/README.md)** - 70+ example programs

### 📖 Core Documentation
- **[Language Specification](SPECIFICATION.md)** - Complete language reference
- **[Features Guide](FEATURES.md)** - All language features
- **[Standard Library](STDLIB.md)** - 37 built-in functions
- **[Shadow Tests](SHADOW_TESTS.md)** - Testing philosophy

### 🏗️ Advanced
- **[Module System](MODULE_SYSTEM.md)** - Creating modules
- **[Extern FFI](EXTERN_FFI.md)** - Calling C functions
- **[AI & Machine Learning](AI_ML_GUIDE.md)** - Neural network inference with ONNX
- **[Architecture Analysis](ARCHITECTURE_ANALYSIS.md)** - System design

### 🤝 Contributing
- **[Contributing Guide](CONTRIBUTING.md)** - How to help
- **[Roadmap](ROADMAP.md)** - Future plans

---

## What is nanolang?

nanolang is a minimal, LLM-friendly programming language with:

- **Prefix notation** - No operator precedence confusion
- **Shadow-tests** - Mandatory compile-time testing
- **Static typing** - Catch errors early
- **C transpilation** - Native performance
- **Clear semantics** - Unambiguous and simple

## Documentation by Skill Level

**🟢 Beginner:** Start with [Getting Started](GETTING_STARTED.md) → [Examples](../examples/README.md) → [Quick Reference](QUICK_REFERENCE.md)

**🟡 Intermediate:** [Language Specification](SPECIFICATION.md) → [Features](FEATURES.md) → [Standard Library](STDLIB.md)

**🔴 Advanced:** [Architecture](ARCHITECTURE_ANALYSIS.md) → [Module System](MODULE_SYSTEM.md) → [Language Design](LANGUAGE_DESIGN_REVIEW.md)

---

📚 **[View Complete Documentation Map →](README.md)**
3. **Simplicity over features** - Minimal but complete
4. **LLM-friendly** - Optimized for AI understanding
5. **Self-hosting** - Eventually compiles itself

## Core Features

### Prefix Notation

```nano
# Clear nesting, no precedence rules
(+ a (* b c))
(and (> x 0) (< x 10))
```

### Shadow-Tests

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

### Static Typing

```nano
let x: int = 42
let name: string = "Alice"
let flag: bool = true
```

### Immutable by Default

```nano
let x: int = 10        # Immutable
let mut y: int = 20    # Mutable
```

## Learning Path

### Day 1: Basics
1. Read [README.md](README.md) for overview
2. Follow [GETTING_STARTED.md](GETTING_STARTED.md) tutorial
3. Run through `examples/hello.nano`

### Day 2: Language Features
1. Study [SPECIFICATION.md](SPECIFICATION.md) sections 1-6
2. Work through `examples/calculator.nano`
3. Try writing your own functions

### Day 3: Advanced Topics
1. Read [SHADOW_TESTS.md](SHADOW_TESTS.md) in detail
2. Study `examples/factorial.nano` and `examples/fibonacci.nano`
3. Practice writing comprehensive tests

### Day 4: Complex Programs
1. Analyze `examples/primes.nano`
2. Write a larger program
3. Review [CONTRIBUTING.md](CONTRIBUTING.md)

## Example Programs

### Tutorial Examples
| Example | Difficulty | Demonstrates |
|---------|-----------|--------------|
| [hello.nano](../examples/hello.nano) | Beginner | Basic structure |
| [calculator.nano](../examples/calculator.nano) | Beginner | Prefix notation |
| [factorial.nano](../examples/factorial.nano) | Intermediate | Recursion, loops |
| [fibonacci.nano](../examples/fibonacci.nano) | Intermediate | Multiple recursion |
| [primes.nano](../examples/primes.nano) | Advanced | Complex logic |

### Game Examples (NEW!)
| Game | Features | Status |
|------|----------|--------|
| [game_of_life.nano](../examples/game_of_life.nano) | Cellular automata, 40x20 grid | ✅ Working |
| [snake.nano](../examples/snake.nano) | AI pathfinding, collision detection | ✅ Working |
| [boids_complete.nano](../examples/boids_complete.nano) | Flocking simulation, vector math | ✅ Working |
| [maze.nano](../examples/maze.nano) | Procedural generation, pathfinding | ✅ Working |
| [particles_sdl.nano](../examples/particles_sdl.nano) | Physics, gravity, SDL rendering | ✅ Working |
| [checkers.nano](../examples/checkers.nano) | Full SDL game, AI, king pieces | ✅ Working |
| [boids_sdl.nano](../examples/boids_sdl.nano) | Visual flocking with SDL | ✅ Working |

## Key Concepts

### Syntax
- Prefix notation for all operations
- Explicit type annotations
- Clear block structure with `{}`
- Keywords: `fn`, `let`, `if`, `while`, `for`, `return`

### Types
- `int` - 64-bit integer
- `float` - 64-bit floating point
- `bool` - Boolean (true/false)
- `string` - UTF-8 text
- `void` - No return value
- `struct` - Composite data types (product types)
- `enum` - Named constants
- `union` - Tagged unions/sum types
- `List<T>` - Generic lists (monomorphized at compile time)
- `fn(T1, T2) -> R` - First-class function types
- `(T1, T2)` - Tuple types (in development)

### Operations
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `and`, `or`, `not`

### Shadow-Tests
- Mandatory for all functions
- Run at compile time
- Use `assert` for verification
- Stripped from production builds

## Project Status

**Current Phase**: ✅ Production Ready

### Completed
- ✅ Language design and specification
- ✅ Lexer with full tokenization
- ✅ Parser with AST generation
- ✅ Type checker with inference
- ✅ Shadow-test runner (interpreter mode)
- ✅ C transpiler for native compilation
- ✅ Interpreter (`nano`) - runs shadow tests with tracing
- ✅ Compiler (`nanoc`) - transpiles to C99 and compiles
- ✅ Standard library (37 functions)
  - 3 I/O, 11 Math, 18 String, 4 Array, 3 OS, Dynamic Generics
- ✅ Array support with bounds checking
- ✅ 24 example programs
- ✅ Test suite (17/17 tests passing)
- ✅ CI/CD with GitHub Actions
- ✅ Comprehensive documentation

### Recent Improvements (November 2025)
- ✅ **Zero Compiler Warnings** - Clean build with no warnings
- ✅ **Generics (Monomorphization)** - `List<T>` with compile-time specialization
- ✅ **First-Class Functions** - Functions as values, parameters, and return types
- ✅ **Union Types** - Tagged unions/sum types with pattern matching
- ✅ **Pattern Matching** - `match` expressions with exhaustive checking
- ✅ **Namespacing** - `nl_` prefix for all user-defined types in C output
- ✅ **Garbage Collection** - Universal GC with reference counting and cycle detection
- ✅ **Dynamic Arrays** - `array_push`, `array_pop`, `array_remove_at` with automatic memory management
- ✅ **Type Casting** - Explicit type conversions: `cast_int()`, `cast_float()`, `cast_bool()`, `cast_string()`
- ✅ **Top-Level Constants** - Global immutable constants at module scope
- ✅ **Unary Operators** - Unary minus and logical not
- ✅ **Float Comparison Fix** - Critical bug fixed: float comparisons now return bool (was returning void)
- ✅ **6 Game Examples** - Conway's Life, Snake, Boids, Maze, Particles, Checkers
- ✅ **Vector Math Module** - 2D vector operations for game development
- ✅ **SDL Integration** - Full SDL2 support with text rendering (SDL_ttf)
- ✅ Multi-line comment support (`/* */`)
- ✅ Fixed critical infinite loop bug in return propagation
- ✅ Added array operations with safety guarantees
- ✅ Expanded math library (sqrt, pow, trig functions)
- ✅ Added string operations (length, concat, substring, etc.)
- ✅ Memory sanitizers (AddressSanitizer, UBSan)
- ✅ Column tracking for precise error messages
- ✅ Unused variable warnings

### Future Enhancements
- ⏳ More array operations (map, filter, reduce, slice)
- ⏳ File I/O functions
- ⏳ More string operations (split, join, case conversion)
- ⏳ Advanced math (logarithms, inverse trig)

## Community

### How to Help

1. **Review documentation** - Report unclear sections
2. **Write examples** - Show interesting use cases
3. **Implement compiler** - Help build the toolchain
4. **Spread the word** - Share nanolang with others

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Pull request process
- Testing requirements
- Documentation standards

## FAQ

**Q: Why prefix notation?**  
A: Eliminates operator precedence confusion. `(+ a (* b c))` is unambiguous.

**Q: Why mandatory shadow-tests?**  
A: Ensures all code is tested. Untested code doesn't compile.

**Q: Is nanolang production-ready?**  
A: Yes! The compiler, interpreter, and standard library are complete. 17/17 tests pass, 23/24 examples work.

**Q: Can I use nanolang for my project?**  
A: Absolutely! It's open source (Apache 2.0). See [GETTING_STARTED.md](GETTING_STARTED.md).

**Q: How do I compile nanolang programs?**  
A: Use `nanoc program.nano` to compile and test, or `nano program.nano` to interpret.

**Q: Why target C?**  
A: Portability, performance, and path to self-hosting.

**Q: What's the standard library like?**  
A: 37 functions covering I/O, math, strings, arrays, OS operations, and generic lists. See [STDLIB.md](STDLIB.md).

## Resources

### Documentation
- [README.md](../README.md) - Overview
- [FEATURES.md](FEATURES.md) - Complete feature list with examples
- [SPECIFICATION.md](SPECIFICATION.md) - Language reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page syntax guide
- [STDLIB.md](STDLIB.md) - Standard library reference (37 functions)
- [GETTING_STARTED.md](GETTING_STARTED.md) - Tutorial
- [SHADOW_TESTS.md](SHADOW_TESTS.md) - Testing guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guide
- [ARRAY_DESIGN.md](ARRAY_DESIGN.md) - Array implementation details
- [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) - System architecture
- [LANGUAGE_DESIGN_REVIEW.md](LANGUAGE_DESIGN_REVIEW.md) - Comprehensive design review (8.5/10)
- [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md) - Executive summary of review findings
- [NAMESPACE_FIXES.md](NAMESPACE_FIXES.md) - Namespace management bug fixes ✅
- [SELF_HOSTING_REQUIREMENTS.md](SELF_HOSTING_REQUIREMENTS.md) - Features needed for self-hosting (detailed)
- [SELF_HOSTING_FEATURE_GAP.md](SELF_HOSTING_FEATURE_GAP.md) - Gap analysis with visuals
- [SELF_HOSTING_CHECKLIST.md](SELF_HOSTING_CHECKLIST.md) - Implementation tracking

**User Guides:**
- [MUTABILITY_GUIDE.md](MUTABILITY_GUIDE.md) - Complete guide to variable mutability (`let` vs `let mut`)
- [TRACING_IMPLEMENTATION.md](TRACING_IMPLEMENTATION.md) - Debugging with the interpreter tracing system
- [GC_FEATURES.md](GC_FEATURES.md) - Garbage collection and dynamic arrays guide
- [FEATURE_TESTING_SUMMARY.md](FEATURE_TESTING_SUMMARY.md) - Summary of tested features and capabilities
- [EXTERN_FFI.md](EXTERN_FFI.md) - How to use external C functions
- [SAFE_C_FFI_FUNCTIONS.md](SAFE_C_FFI_FUNCTIONS.md) - Reference of safe C functions for FFI
- [AI_ML_GUIDE.md](AI_ML_GUIDE.md) - AI and machine learning with ONNX Runtime
- [MODULES.md](MODULES.md) - Module system guide (importing, packaging, distribution)
- [BUILDING_HYBRID_APPS.md](BUILDING_HYBRID_APPS.md) - Building hybrid C/nanolang applications

**Feature Designs (Historical Reference):**
- [STRUCTS_DESIGN.md](STRUCTS_DESIGN.md) - Struct design documentation (✅ Implemented)
- [ENUMS_DESIGN.md](ENUMS_DESIGN.md) - Enum design documentation (✅ Implemented)
- [LISTS_DESIGN.md](LISTS_DESIGN.md) - Dynamic list design documentation (✅ Implemented)
- [ARRAY_DESIGN.md](ARRAY_DESIGN.md) - Array design documentation (✅ Implemented)
- [STDLIB_ADDITIONS_DESIGN.md](STDLIB_ADDITIONS_DESIGN.md) - Standard library additions design (✅ Implemented)

### Examples
- [examples/README.md](examples/README.md) - Example guide
- [examples/*.nano](examples/) - Sample programs

### Project Files
- [LICENSE](LICENSE) - Apache 2.0 license
- [.gitignore](.gitignore) - Git exclusions

## Design Rationale

### Why LLM-Friendly?

Large Language Models are becoming important tools for software development. nanolang is designed to be:

1. **Predictable** - Clear patterns for LLMs to learn
2. **Unambiguous** - One way to do things reduces errors
3. **Testable** - Forces LLMs to think about test cases
4. **Minimal** - Smaller surface area for LLMs to master

### Why Minimal?

A minimal language is:
- Easier to learn
- Easier to implement correctly
- Easier to reason about
- Less prone to bugs
- More maintainable

### Why Shadow-Tests?

Traditional testing is optional. Shadow-tests make testing:
- Mandatory (can't compile without tests)
- Integrated (tests with code)
- Immediate (run during compilation)
- Documented (tests show usage)

### Why C Transpilation?

C as a target provides:
- **Performance** - Compiled to native code
- **Portability** - C runs everywhere
- **Interoperability** - Easy FFI with C libraries
- **Tooling** - Leverage mature C ecosystem
- **Self-hosting** - Path to compiling nanolang in nanolang

## Conclusion

nanolang is designed to be a minimal, unambiguous, testable language that both humans and LLMs can work with effectively. We believe that by making testing mandatory and syntax unambiguous, we can build more reliable software.

Start with [GETTING_STARTED.md](GETTING_STARTED.md) and enjoy coding in nanolang!

---

**License**: Apache 2.0  
**Project**: https://github.com/jordanhubbard/nanolang

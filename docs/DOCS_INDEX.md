# nanolang Documentation Index

Welcome to nanolang! This index will help you navigate the documentation.

## Quick Links

- **New to nanolang?** Start with [Getting Started](GETTING_STARTED.md)
- **Want details?** Read the [Language Specification](SPECIFICATION.md)
- **Need a quick reference?** See [Quick Reference](QUICK_REFERENCE.md)
- **Standard library?** Browse [Standard Library Reference](STDLIB.md) (24 functions)
- **Learn about shadow-tests?** See [Shadow-Tests Guide](SHADOW_TESTS.md)
- **Ready to contribute?** Check [Contributing Guidelines](CONTRIBUTING.md)
- **Want examples?** Browse the [examples/](../examples/) directory
- **Using C FFI?** See [External C Function Interface](EXTERN_FFI.md) and [Safe C FFI Functions Reference](SAFE_C_FFI_FUNCTIONS.md)
- **Variable mutability?** See [Mutability Guide](MUTABILITY_GUIDE.md)
- **Debugging with tracing?** See [Tracing System](TRACING_IMPLEMENTATION.md)

## Documentation Structure

### For Beginners

1. **[README.md](README.md)** - Project overview and quick introduction
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Your first nanolang program
3. **[examples/](examples/)** - Sample code to learn from

### For Developers

1. **[SPECIFICATION.md](SPECIFICATION.md)** - Complete language reference
2. **[SHADOW_TESTS.md](SHADOW_TESTS.md)** - Deep dive into testing
3. **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute

## What is nanolang?

nanolang is a minimal, LLM-friendly programming language with:

- **Prefix notation** - No operator precedence confusion
- **Shadow-tests** - Mandatory compile-time testing
- **Static typing** - Catch errors early
- **C transpilation** - Native performance
- **Clear semantics** - Unambiguous and simple

## Philosophy

nanolang prioritizes:

1. **Clarity over brevity** - Explicit is better than implicit
2. **Testing over trust** - All code must be tested
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

| Example | Difficulty | Demonstrates |
|---------|-----------|--------------|
| [hello.nano](examples/hello.nano) | Beginner | Basic structure |
| [calculator.nano](examples/calculator.nano) | Beginner | Prefix notation |
| [factorial.nano](examples/factorial.nano) | Intermediate | Recursion, loops |
| [fibonacci.nano](examples/fibonacci.nano) | Intermediate | Multiple recursion |
| [primes.nano](examples/primes.nano) | Advanced | Complex logic |

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
- ✅ Interpreter (`nano`) - runs shadow tests
- ✅ Compiler (`nanoc`) - transpiles to C and compiles
- ✅ Standard library (24 functions)
  - 3 I/O, 11 Math, 5 String, 4 Array, 3 OS
- ✅ Array support with bounds checking
- ✅ 24 example programs
- ✅ Test suite (17/17 tests passing)
- ✅ CI/CD with GitHub Actions
- ✅ Comprehensive documentation

### Recent Improvements
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
A: 24 functions covering I/O, math, strings, arrays, and OS operations. See [STDLIB.md](STDLIB.md).

## Resources

### Documentation
- [README.md](../README.md) - Overview
- [SPECIFICATION.md](SPECIFICATION.md) - Language reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page syntax guide
- [STDLIB.md](STDLIB.md) - Standard library reference (24 functions)
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
- [EXTERN_FFI.md](EXTERN_FFI.md) - How to use external C functions
- [SAFE_C_FFI_FUNCTIONS.md](SAFE_C_FFI_FUNCTIONS.md) - Reference of safe C functions for FFI

**Feature Designs (Ready to Implement):**
- [STRUCTS_DESIGN.md](STRUCTS_DESIGN.md) - Complete struct design (Priority #1)
- [ENUMS_DESIGN.md](ENUMS_DESIGN.md) - Complete enum design (Priority #2)
- [LISTS_DESIGN.md](LISTS_DESIGN.md) - Complete dynamic list design (Priority #3)
- [STDLIB_ADDITIONS_DESIGN.md](STDLIB_ADDITIONS_DESIGN.md) - File I/O, string ops, system execution

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

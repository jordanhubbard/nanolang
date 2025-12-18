# Nanolang Tutorials

Comprehensive step-by-step guides to master nanolang from beginner to advanced.

## Learning Path

### 🌟 Beginner (Start Here!)

**[1. Getting Started](01-getting-started.md)** ⏱️ 30 mins
- Installation and setup
- Your first program
- Understanding prefix notation
- Basic syntax and types
- Shadow tests introduction

**[2. Language Fundamentals](02-language-fundamentals.md)** ⏱️ 1-2 hours
- Structs and enums
- Pattern matching
- Generic types
- Higher-order functions
- Error handling with unions

**[3. Module System](03-modules.md)** ⏱️ 45 mins
- Importing modules
- Standard library overview
- Creating your own modules
- Module organization
- Visibility and encapsulation

### 🔧 Intermediate

**[4. FFI Integration](04-ffi-integration.md)** ⏱️ 1 hour  
*Coming soon - See [FFI Guide](../FFI_GUIDE.md)*
- Calling C libraries
- Type marshaling
- Creating C modules
- Memory management in FFI

**[5. Building Applications](05-building-applications.md)** ⏱️ 2-3 hours  
*Coming soon*
- Project structure
- Testing strategies
- Database integration
- File I/O patterns
- Error handling patterns

### 🚀 Advanced

**[6. Performance Optimization](06-performance.md)** ⏱️ 1-2 hours  
*Coming soon*
- Benchmarking
- Memory optimization
- Compilation flags
- Profiling tools

**[7. Advanced Type System](07-advanced-types.md)** ⏱️ 1 hour  
*Coming soon*
- Complex generic constraints
- Variance and type bounds
- Phantom types
- Type-level programming

## Quick Start for Impatient Developers

Already know programming? Start here:

```bash
# Install
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang && make

# Hello World
echo 'fn main() -> int {
    (println "Hello, World!")
    return 0
}' > hello.nano

./bin/nano hello.nano
./bin/nanoc hello.nano -o hello && ./hello

# Learn by example
cd examples && ls *.nano
```

Key differences from other languages:
- **Prefix notation**: `(+ 1 2)` not `1 + 2`
- **Mandatory testing**: Shadow tests required
- **Explicit types**: Return types always specified
- **Immutable by default**: Use `mut` for mutation

## Tutorial Topics

### Core Language
- ✅ Syntax and semantics
- ✅ Type system fundamentals
- ✅ Control flow
- ✅ Functions and closures
- ✅ Data structures (structs, enums, unions)
- ✅ Pattern matching
- ✅ Generic programming

### Standard Library
- ✅ Console I/O (stdio)
- ✅ String operations
- ✅ Collections (arrays, lists)
- ✅ Math utilities
- 🔄 File system operations
- 🔄 JSON parsing
- 🔄 HTTP client/server
- 🔄 Regular expressions

### Tooling
- ✅ Compiler vs Interpreter
- ✅ Module system
- 🔄 REPL usage
- 🔄 IDE integration
- 🔄 Debugging techniques
- 🔄 Performance profiling

### Advanced Topics
- 🔄 FFI (Foreign Function Interface)
- 🔄 Memory management
- 🔄 Concurrency patterns
- 🔄 Optimization techniques
- 🔄 Building libraries
- 🔄 Package management

Legend: ✅ Complete | 🔄 In Progress | 📋 Planned

## Example Programs

After completing the tutorials, explore these examples:

### Basic Examples
- `examples/nl_hello.nano` - Hello World
- `examples/nl_factorial.nano` - Recursion
- `examples/nl_fibonacci.nano` - Iteration and recursion
- `examples/nl_struct.nano` - Struct basics
- `examples/nl_enum.nano` - Enum basics

### Intermediate Examples
- `examples/nl_union_types.nano` - Union types and pattern matching
- `examples/nl_first_class_functions.nano` - Higher-order functions
- `examples/nl_arrays_test.nano` - Array operations
- `examples/nl_map_reduce.nano` - Functional programming

### Advanced Examples
- `examples/nl_data_analytics.nano` - Data processing
- `examples/sqlite_simple.nano` - Database operations
- `examples/sdl_pong.nano` - Game development
- `examples/sdl_raytracer.nano` - Graphics programming

## Additional Resources

### Documentation
- [Feature List](../FEATURES.md) - Complete language reference
- [Module API Reference](../MODULES.md) - Available libraries
- [Memory Management](../MEMORY_MANAGEMENT.md) - GC and ownership

### Development
- [Contributing Guide](../../CONTRIBUTING.md) - How to contribute
- [Project Structure](../project-structure.md) - Codebase organization
- [Testing Standards](../../.cursor/rules/PROJECT_RULES.md) - Quality guidelines

### Community
- [GitHub Issues](https://github.com/jordanhubbard/nanolang/issues) - Report bugs, request features
- [Examples Directory](../../examples/) - Real-world code samples
- [Discussions](https://github.com/jordanhubbard/nanolang/discussions) - Community help

## Learning Tips

### 1. Type Along

Don't just read - type the examples! Muscle memory helps learning stick.

### 2. Experiment

Modify the examples. Break things. See what error messages you get. This builds intuition.

### 3. Use Shadow Tests

Write tests as you learn. They'll catch mistakes and reinforce concepts.

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
    assert (== (double 0) 0)
}
```

### 4. Read Examples

The `examples/` directory has production-quality code. Reading it teaches idiomatic patterns.

### 5. Start Small

Build simple programs first. Gradually add complexity.

**Learning progression:**
1. Calculator (arithmetic, functions)
2. Todo list (arrays, structs, file I/O)
3. HTTP server (networking, modules)
4. Game (SDL, event handling, state management)

## Common Questions

**Q: Do I need to know C?**  
A: No! Nanolang hides C details. But knowing C helps with FFI and performance tuning.

**Q: Can I use nanolang for production?**  
A: Yes! It compiles to native code with excellent performance. Used for games, tools, and applications.

**Q: How do I install packages?**  
A: Currently manual. Copy modules to `modules/` directory. Package manager coming soon!

**Q: Is there an IDE?**  
A: Syntax highlighting available for major editors. LSP server in development for full IDE support.

**Q: How fast is nanolang?**  
A: Same as C (it compiles to C). Typical benchmarks show 90-100% of hand-written C performance.

## Next Steps

1. **Complete Tutorial 1** - Get comfortable with basics
2. **Try examples** - Run programs in `examples/`
3. **Build something** - Start a small project
4. **Read MODULES.md** - Explore available libraries
5. **Join the community** - Share your creations!

Happy learning! 🚀


# NanoLang User Guide

**Welcome to NanoLang by Example** - A comprehensive guide to writing NanoLang programs.

## The Problem: LLMs and Existing Languages

Large Language Models struggle with existing programming languages because they offer too many ways to do the same thing:

**Python:**
```python
# String formatting - 4 different ways
"Hello " + name                    # concatenation
"Hello %s" % name                  # old style
"Hello {}".format(name)            # format method
f"Hello {name}"                    # f-strings
```

**JavaScript:**
```javascript
// Function definitions - 5 different ways
function add(a, b) { return a + b; }          // function declaration
const add = function(a, b) { return a + b; }  // function expression
const add = (a, b) => { return a + b; }       // arrow function
const add = (a, b) => a + b;                  // implicit return
class Math { static add(a, b) { return a + b; } }  // class method
```

**Rust:**
```rust
// Implicit vs explicit returns - both valid
fn add1(a: i32, b: i32) -> i32 { a + b }        // implicit
fn add2(a: i32, b: i32) -> i32 { return a + b; } // explicit
```

This ambiguity creates problems for LLM code generation:
- **Inconsistent outputs** - Same prompt produces different styles
- **Mixing styles** - Code generated in multiple sessions has no coherent style
- **Hard to validate** - Multiple valid forms make correctness harder to verify
- **Context pollution** - LLMs must track multiple equivalent patterns

## The Solution: NanoLang's Canonical Syntax

NanoLang provides **exactly ONE canonical way** to write each construct:

```nano
# String concatenation - ONE way only
(str_concat "Hello " name)

# Function definition - ONE way only
fn add(a: int, b: int) -> int {
    return (+ a b)
}

# All operations use prefix notation
(+ a b)           # Addition
(== x 5)          # Comparison
(println "Hi")    # Function call
```

**Benefits for LLM code generation:**
- ✅ **Consistent output** - Same prompt always produces same style
- ✅ **Predictable structure** - Prefix notation is unambiguous
- ✅ **Easy to validate** - Only one correct form
- ✅ **Better training** - Less noise in training data
- ✅ **Composable** - Uniform syntax makes code generation reliable

## What is NanoLang?

NanoLang is a compiled systems programming language designed specifically for LLM code generation. It transpiles to C for native performance while maintaining a simple, consistent syntax.

**Key Features:**
- **Automatic Memory Management (ARC):** Zero-overhead reference counting, no manual free() calls ⭐ NEW in v2.3.0
- **LLM-Powered Autonomous Optimization:** Continuous profiling and automatic code optimization ⭐ NEW in v2.3.0
- **LLM-First Design:** Exactly ONE canonical way to write each construct
- **Prefix Notation:** All operations use `(f x y)` form
- **Explicit Types:** Always annotate types, minimal inference
- **Shadow Tests:** Every function has compile-time tests (mandatory)
- **C Interop:** Full FFI for calling C libraries
- **Zero Runtime Overhead:** Compiles to native C code

## How to Use This Guide

This guide is organized into four main parts:

### Part I: Language Fundamentals
Learn the core NanoLang language, one concept at a time. Start here if you're new to NanoLang.

- **Chapter 1:** [Getting Started](01_getting_started.md)
- **Chapter 2:** [Control Flow](02_control_flow.md)
- **Chapter 3:** [Basic Types](03_basic_types.md)
- **Chapter 4:** [Higher-Level Patterns](04_higher_level_patterns.md)
- **Chapter 5:** [Modules](05_modules.md)
- **Chapter 6:** [Canonical Syntax](06_canonical_syntax.md)
- **Chapter 7:** [Examples](07_examples.md)
- **Chapter 8:** [LLM-Powered Profiling](08_profiling.md) ⭐ NEW

### Part II: Standard Library
The batteries included with NanoLang - built-in utilities available in every program.

- **Chapter 9:** Core Utilities
- **Chapter 10:** Collections Library
- **Chapter 11:** I/O & Filesystem
- **Chapter 12:** System & Runtime

### Part III: External Modules
Powerful extensions for graphics, networking, databases, and more.

- **Chapter 13:** Text Processing (regex, log, StringBuilder)
- **Chapter 14:** Data Formats (JSON, SQLite)
- **Chapter 15:** Web & Networking (curl, http_server, uv)
- **Chapter 16:** Graphics Fundamentals (SDL modules)
- **Chapter 17:** OpenGL Graphics
- **Chapter 18:** Game Development
- **Chapter 19:** Terminal UI (ncurses)
- **Chapter 20:** Testing & Quality
- **Chapter 21:** Configuration

### Part IV: Advanced Topics
Deep dives into best practices, patterns, and performance.

- **Chapter 22:** Canonical Style Guide
- **Chapter 23:** Higher-Level Patterns
- **Chapter 24:** Performance & Optimization
- **Chapter 25:** Contributing & Extending
- **Chapter 26:** LLM Code Generation
- **Chapter 27:** Self-Hosting

### Appendices
Quick references, troubleshooting, and comprehensive examples.

- **Appendix A:** Examples Gallery
- **Appendix B:** Quick Reference
- **Appendix C:** Troubleshooting Guide
- **Appendix D:** Glossary
- **Appendix E:** Error Reference
- **Appendix F:** Migration Guide

## Learning Path

**Absolute Beginners:** Start with Part I and work through sequentially.

**Experienced Programmers:** Skim Chapters 1-5, focus on NanoLang-specific features (shadow tests, prefix notation, modules).

**Looking for Specific Features:** Use the Quick Reference (Appendix B) or search functionality.

**Working on a Project:** Part III shows you how to use external libraries for graphics, networking, databases, and more.

## Example-Driven Documentation

Every function, every feature has working examples with shadow tests. You can copy and run any code snippet in this guide.

**Conventions:**
- ✅ **Try This:** Working example you can run
- ⚠️ **Watch Out:** Common pitfalls and warnings
- 💡 **Pro Tip:** Best practices and optimization hints
- ❌ **Don't Do This:** Anti-patterns to avoid

## Get Started

Ready to write your first NanoLang program? Jump to [Chapter 1: Getting Started](part1_fundamentals/01_getting_started.md).

Already familiar with the basics? Explore the [Examples Gallery](appendices/a_examples_gallery.md) for real-world programs.

---

**NanoLang Version:** 0.5+ (current development version)  
**Last Updated:** 2026-01-20

# My Documentation

I organize my documentation to help you understand my structure, my syntax, and my proofs. I describe what I am and how I function. I do not use marketing language.

## Table of Contents

### Getting Started

I recommend starting here if you are new to my environment:

1. **[Getting Started Guide](GETTING_STARTED.md)** - My first steps
   - Installation and setup
   - Your first program
   - My core concepts
   - Common patterns and mistakes

2. **[Quick Reference](QUICK_REFERENCE.md)** - Syntax cheat sheet
   - My operators at a glance
   - My type system
   - Common patterns
   - Quick examples

3. **[Examples Directory](../examples/README.md)** - Learning by example
   - 70+ programs I can run
   - From "hello world" to games
   - Progressive complexity
   - All with shadow tests

### Language Reference

The details of my design:

4. **[Language Specification](SPECIFICATION.md)** - Complete language reference
   - Formal syntax and semantics
   - Type system details
   - Standard library reference
   - Compilation model

5. **[Features Guide](FEATURES.md)** - My features
   - Data structures (structs, enums, unions, tuples)
   - Generics and type parameters
   - First-class functions
   - Module system

5.5. **Type System Deep Dives**
   - [Type Inference Rules](TYPE_INFERENCE.md) - What I can and cannot infer
   - [Generics Deep Dive](GENERICS_DEEP_DIVE.md) - How I handle monomorphization
   - [Error Messages Guide](ERROR_MESSAGES.md) - How to read my compiler errors

5.6. **Strings and Text**
   - [Unicode Support](UNICODE.md) - UTF-8, string vs bstring, limitations

5.7. **Performance**
   - [Performance Characteristics](PERFORMANCE.md) - Speed, memory, optimization

5.8. **Arrays and Memory**
   - [Array Safety](ARRAY_SAFETY.md)
   - [Dynamic Arrays](DYNAMIC_ARRAYS.md)
   - [Memory Management](MEMORY_MANAGEMENT.md)

6. **[Standard Library](STDLIB.md)** - My built-in functions

### Testing and Quality

My philosophy on honesty and verification:

7. **[Shadow Tests](SHADOW_TESTS.md)** - Mandatory compile-time testing
   - What shadow tests are
   - Why I require them
   - Writing good tests
   - Best practices

8. **[Code Coverage](COVERAGE.md)** - Coverage reporting and tracking
   - Generating coverage reports
   - Coverage targets by component
   - Interpreting results
   - [Coverage Status](COVERAGE_STATUS.md)

9. **[Feature Coverage](../tests/FEATURE_COVERAGE.md)** - My test suite

### Architecture and Design

How I am built:

10. **[NanoISA Virtual Machine](NANOISA.md)** - My VM backend
    - 178-opcode stack machine ISA
    - .nvm binary format specification
    - Co-process FFI isolation protocol
    - VM daemon for distributed execution
    - Trap model (pure-compute core and I/O handlers)
    - Native binary generation from bytecode

10.5. **[Formal Verification](../formal/README.md)** - My proofs in Coq
    - Type soundness (preservation and progress)
    - Determinism of evaluation
    - Big-step / small-step semantic equivalence
    - Computable reference interpreter with soundness proof
    - 6,170 lines of Coq, 193 theorems, 0 axioms

11. **Design and implementation notes**
    - Notes for those maintaining me live in [planning/](../planning/)

12. **[Language Design Review](LANGUAGE_DESIGN_REVIEW.md)** - My design philosophy
    - Why I use prefix notation for calls and dual notation for operators
    - My LLM-friendly design
    - How I compare to other languages
    - My design trade-offs

### Advanced Topics

For experienced users:

12. **[Module System](MODULE_SYSTEM.md)** - Creating and using modules
    - Module structure
    - FFI (Foreign Function Interface)
    - Building C extensions
    - Automatic module building
    - [Namespace Usage](NAMESPACE_USAGE.md) - Import patterns and best practices

13. **[Platform Compatibility](PLATFORM_COMPATIBILITY.md)** - Cross-platform guide
    - macOS and Linux support
    - SDL software renderer fallback
    - Troubleshooting platform issues
    - Known limitations

14. **[Extern FFI](EXTERN_FFI.md)** - Calling C functions from me
    - FFI safety guidelines
    - Common vulnerabilities and fixes

15. **[Building Hybrid Apps](BUILDING_HYBRID_APPS.md)** - Combining me with C

### Roadmap and Status

My status and plans:

16. **[Roadmap](ROADMAP.md)** - Future development plans
17. **[RFC Process](RFC_PROCESS.md)** - How I evolve
18. **[Package Manager Design](PACKAGE_MANAGER_DESIGN.md)** - Package manager proposal
19. **[Spec / Coverage Audit](../SPEC_AUDIT.md)** - Gaps I track for parity
20. **[Self-Hosting Checklist](../planning/SELF_HOSTING_CHECKLIST.md)** - My path to self-hosting

### Contributing

How to help me:

19. **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
    - Code contributions
    - Documentation improvements
    - Bug reports
    - Feature requests

20. **[Contributors List](../CONTRIBUTORS.md)** - Those who have helped me

## Documentation by Topic

### By Skill Level

**I am new to you**
- [Getting Started Guide](GETTING_STARTED.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Examples](../examples/README.md)

**You know my basics**
- [Language Specification](SPECIFICATION.md)
- [Features Guide](FEATURES.md)
- [Standard Library](STDLIB.md)
- [Shadow Tests](SHADOW_TESTS.md)

**You seek deep understanding**
- [NanoISA Virtual Machine](NANOISA.md)
- [Formal Verification](../formal/README.md)
- [Planning / design notes](../planning/README.md)
- [Module System](MODULE_SYSTEM.md)
- [Language Design Review](LANGUAGE_DESIGN_REVIEW.md)

### By Interest

**I want to learn the language:**
[Getting Started](GETTING_STARTED.md) -> [Examples](../examples/README.md) -> [Quick Reference](QUICK_REFERENCE.md)

**I want to understand the design:**
[Language Design Review](LANGUAGE_DESIGN_REVIEW.md) -> [NanoISA VM](NANOISA.md) -> [Formal Proofs](../formal/README.md) -> [planning/](../planning/)

**I want to contribute:**
[Contributing Guide](CONTRIBUTING.md) -> [planning/](../planning/) -> [Roadmap](ROADMAP.md)

**I want to build something:**
[Examples](../examples/README.md) -> [Standard Library](STDLIB.md) -> [Module System](MODULE_SYSTEM.md)

**I want to use C libraries:**
[Extern FFI](EXTERN_FFI.md) -> [Module System](MODULE_SYSTEM.md) -> [Building Hybrid Apps](BUILDING_HYBRID_APPS.md)

**I am having platform or compatibility issues:**
[Platform Compatibility](PLATFORM_COMPATIBILITY.md) -> [Module System](MODULE_SYSTEM.md)

## Planning Documents

I keep historical design documents and implementation plans in the [planning/](../planning/) directory. I intend these for those interested in how I have evolved.

## Quick Links

- [Main README](../README.md)
- [Report a Bug](https://github.com/jordanhubbard/nanolang/issues/new?template=bug_report.md)
- [Request a Feature](https://github.com/jordanhubbard/nanolang/issues/new?template=feature_request.md)
- [Discussions](https://github.com/jordanhubbard/nanolang/discussions)

## My Documentation Style

When you write documentation for me, follow these principles:
- **Be concise.** Respect the reader's time.
- **Use examples.** Show what I do.
- **Be clear.** Avoid jargon.
- **Be accurate.** Keep my docs in sync with my code.
- **Be helpful.** Provide what is needed.
- **Use my voice.** Speak as me, in the first person.

## Search for Content

- Check the [Quick Reference](QUICK_REFERENCE.md) for syntax questions.
- Browse the [Examples](../examples/README.md) for code patterns.
- Search the [Issues](https://github.com/jordanhubbard/nanolang/issues) for discussions.
- Ask in [Discussions](https://github.com/jordanhubbard/nanolang/discussions).


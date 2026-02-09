# Nanolang Documentation

Welcome to the nanolang documentation! This guide will help you navigate all available resources.

## 📚 Table of Contents

### 🚀 Getting Started (Start Here!)

Perfect for newcomers to nanolang:

1. **[Getting Started Guide](GETTING_STARTED.md)** - Your first steps with nanolang
   - Installation and setup
   - Your first program
   - Core concepts explained
   - Common patterns and mistakes

2. **[Quick Reference](QUICK_REFERENCE.md)** - Handy syntax cheat sheet
   - All operators at a glance
   - Type system overview
   - Common patterns
   - Quick examples

3. **[Examples Directory](../examples/README.md)** - Learn by example
   - 70+ example programs
   - From "hello world" to games
   - Progressive complexity
   - All with shadow-tests

### 📖 Language Reference

Complete language documentation:

4. **[Language Specification](SPECIFICATION.md)** - Complete language reference
   - Formal syntax and semantics
   - Type system details
   - Standard library reference
   - Compilation model

5. **[Features Guide](FEATURES.md)** - Feature-by-feature breakdown
   - Data structures (structs, enums, unions, tuples)
   - Generics and type parameters
   - First-class functions
   - Module system

5.5. **Type System Deep Dives**
   - [Type Inference Rules](TYPE_INFERENCE.md) - What can/cannot be inferred
   - [Generics Deep Dive](GENERICS_DEEP_DIVE.md) - Monomorphization explained
   - [Error Messages Guide](ERROR_MESSAGES.md) - Understanding compiler errors

5.6. **Strings & Text**
   - [Unicode Support](UNICODE.md) - UTF-8, string vs bstring, limitations

5.7. **Performance**
   - [Performance Characteristics](PERFORMANCE.md) - Speed, memory, optimization

5.8. **Arrays & Memory**
   - [Array Safety](ARRAY_SAFETY.md)
   - [Dynamic Arrays](DYNAMIC_ARRAYS.md)
   - [Memory Management](MEMORY_MANAGEMENT.md)

6. **[Standard Library](STDLIB.md)** - Built-in functions

### 🧪 Testing & Quality

Understanding nanolang's testing philosophy:

7. **[Shadow Tests](SHADOW_TESTS.md)** - Mandatory compile-time testing
   - What are shadow-tests?
   - Why they're mandatory
   - Writing good tests
   - Best practices

8. **[Code Coverage](COVERAGE.md)** - Coverage reporting and tracking
   - Generating coverage reports
   - Coverage targets by component
   - Interpreting results
   - [Coverage Status](COVERAGE_STATUS.md)

9. **[Feature Coverage](../tests/FEATURE_COVERAGE.md)** - Test suite overview

### 🏗️ Architecture & Design

For contributors and language designers:

10. **[NanoISA Virtual Machine](NANOISA.md)** - Complete VM backend
    - 178-opcode stack machine ISA
    - .nvm binary format specification
    - Co-process FFI isolation protocol
    - VM daemon for distributed execution
    - Trap model (pure-compute core + I/O handlers)
    - Native binary generation from bytecode

10.5. **[Formal Verification](../formal/README.md)** - Mechanized proofs in Coq
    - Type soundness (preservation + progress)
    - Determinism of evaluation
    - Big-step / small-step semantic equivalence
    - Computable reference interpreter with soundness proof
    - ~6,170 lines of Coq, 193 theorems, 0 axioms

11. **Design + implementation notes**
    - Maintainer-facing design docs live in [planning/](../planning/)

12. **[Language Design Review](LANGUAGE_DESIGN_REVIEW.md)** - Design philosophy
    - Why prefix notation for function calls? Why dual notation for operators?
    - LLM-friendly design
    - Comparison to other languages
    - Design trade-offs

### 🔧 Advanced Topics

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

14. **[Extern FFI](EXTERN_FFI.md)** - Calling C functions from nanolang
    - FFI safety guidelines
    - Common vulnerabilities and fixes

15. **[Building Hybrid Apps](BUILDING_HYBRID_APPS.md)** - Combining nanolang with C

### 📊 Roadmap & Status

Project status and future plans:

16. **[Roadmap](ROADMAP.md)** - Future development plans
17. **[RFC Process](RFC_PROCESS.md)** - Language evolution process
18. **[Package Manager Design](PACKAGE_MANAGER_DESIGN.md)** - Package manager proposal
19. **[Spec / Coverage Audit](../SPEC_AUDIT.md)** - Gaps tracked for LLM/implementation parity
20. **[Self-Hosting Checklist](../planning/SELF_HOSTING_CHECKLIST.md)** - Path to self-hosting

### 🤝 Contributing

Help make nanolang better:

19. **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
    - Code contributions
    - Documentation improvements
    - Bug reports
    - Feature requests

20. **[Contributors List](../CONTRIBUTORS.md)** - Thank you!

## 📋 Documentation by Topic

### By Skill Level

**🟢 Beginner (New to nanolang)**
- [Getting Started Guide](GETTING_STARTED.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Examples](../examples/README.md)

**🟡 Intermediate (Know the basics)**
- [Language Specification](SPECIFICATION.md)
- [Features Guide](FEATURES.md)
- [Standard Library](STDLIB.md)
- [Shadow Tests](SHADOW_TESTS.md)

**🔴 Advanced (Deep understanding)**
- [NanoISA Virtual Machine](NANOISA.md)
- [Formal Verification](../formal/README.md)
- [Planning / design notes](../planning/README.md)
- [Module System](MODULE_SYSTEM.md)
- [Language Design Review](LANGUAGE_DESIGN_REVIEW.md)

### By Interest

**I want to learn the language:**
→ [Getting Started](GETTING_STARTED.md) → [Examples](../examples/README.md) → [Quick Reference](QUICK_REFERENCE.md)

**I want to understand the design:**
→ [Language Design Review](LANGUAGE_DESIGN_REVIEW.md) → [NanoISA VM](NANOISA.md) → [Formal Proofs](../formal/README.md) → [planning/](../planning/)

**I want to contribute:**
→ [Contributing Guide](CONTRIBUTING.md) → [planning/](../planning/) → [Roadmap](ROADMAP.md)

**I want to build something:**
→ [Examples](../examples/README.md) → [Standard Library](STDLIB.md) → [Module System](MODULE_SYSTEM.md)

**I want to use C libraries:**
→ [Extern FFI](EXTERN_FFI.md) → [Module System](MODULE_SYSTEM.md) → [Building Hybrid Apps](BUILDING_HYBRID_APPS.md)

**I'm having platform/compatibility issues:**
→ [Platform Compatibility](PLATFORM_COMPATIBILITY.md) → [Module System](MODULE_SYSTEM.md)

## 🗂️ Planning Documents

Historical design documents and implementation plans are in the [planning/](../planning/) directory. These are primarily for maintainers and contributors interested in the project's evolution.

## 💡 Quick Links

- 🏠 [Main README](../README.md)
- 🐛 [Report a Bug](https://github.com/jordanhubbard/nanolang/issues/new?template=bug_report.md)
- 💡 [Request a Feature](https://github.com/jordanhubbard/nanolang/issues/new?template=feature_request.md)
- 🤝 [Discussions](https://github.com/jordanhubbard/nanolang/discussions)

## 📝 Documentation Style Guide

When contributing to documentation:
- **Be concise** - Respect the reader's time
- **Use examples** - Show, don't just tell
- **Be clear** - Avoid jargon where possible
- **Be accurate** - Keep docs in sync with code
- **Be helpful** - Think about what readers need

## 🔍 Can't Find What You Need?

- Check the [Quick Reference](QUICK_REFERENCE.md) for syntax questions
- Browse the [Examples](../examples/README.md) for code patterns
- Search the [Issues](https://github.com/jordanhubbard/nanolang/issues) for discussions
- Ask in [Discussions](https://github.com/jordanhubbard/nanolang/discussions)

---

**Happy coding with nanolang!** 🚀

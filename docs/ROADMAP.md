# nanolang Roadmap

This document outlines the development roadmap for nanolang.

## Project Vision

Build a minimal, LLM-friendly programming language that:
- Compiles to C for performance and portability
- Requires shadow-tests for all code
- Supports both infix (`a + b`) and prefix (`(+ a b)`) notation for operators
- Eventually self-hosts (compiles itself)

## Current Status: Phase 8 - Self-Hosting COMPLETE ✅ (v0.2.0)

**Status**: **PRODUCTION-READY** - Full self-hosting achieved, 100% bootstrap working

**Current Capabilities**:
- ✅ **100% Self-Hosting** - NanoLang compiler compiles itself (3-stage bootstrap verified)
- ✅ Complete compilation pipeline (lexer → parser → type checker → transpiler)
- ✅ Shadow-test execution during compilation (compile-time evaluator)
- ✅ Multiple executables: `bin/nanoc` (compiler), `bin/nanorepl` (REPL prototypes)
- ✅ **Type System** - Primitives, arrays, structs, enums, unions, generics, tuples, first-class functions, affine types
- ✅ **66 Standard Library Functions** - Math, strings, binary strings, arrays, I/O, OS, checked math, generics
- ✅ **30+ FFI Modules** - SDL, ncurses, OpenGL, curl, readline, Python bridge, etc.
- ✅ **90+ Working Examples** - Games, graphics, simulations, data analytics, etc.
- ✅ **221 Test Files** - Unit, integration, regression, negative, performance tests
- ✅ Extensive documentation (121+ markdown files)

## Phase 1 - Lexer ✅ Complete

**Goal**: Transform source text into tokens

**Deliverables**:
- ✅ Token definitions (nanolang.h)
- ✅ Lexer implementation (src/lexer.c - ~300 lines)
- ✅ Error reporting with line numbers
- ✅ Test suite for lexer (all examples tokenize correctly)
- ✅ Handle comments (# style)
- ✅ Handle string literals
- ✅ Handle numeric literals (int and float)

**Completion Date**: September 29, 2025

**Success Criteria**: All met ✅
- Can tokenize all example programs
- Clear error messages for invalid input
- Works with 15/15 examples

## Phase 2 - Parser ✅ Complete

**Goal**: Transform tokens into Abstract Syntax Tree (AST)

**Deliverables**:
- ✅ AST node definitions (nanolang.h)
- ✅ Recursive descent parser (src/parser.c - ~680 lines)
- ✅ Prefix and infix notation support
- ✅ Error recovery
- ✅ Test suite for parser (all examples parse correctly)
- ⚠️ Pretty-printer (not implemented - low priority)

**Completion Date**: September 30, 2025

**Success Criteria**: All met ✅
- Can parse all example programs
- Produces valid AST
- Helpful error messages
- Works with 15/15 examples

## Phase 3 - Type Checker ✅ Complete

**Goal**: Verify type correctness of AST

**Deliverables**:
- ✅ Type inference engine (src/typechecker.c - ~500 lines)
- ✅ Type checking rules for all operators
- ✅ Symbol table with scoping
- ✅ Scope resolution
- ✅ Error messages for type errors
- ✅ Test suite for type checker (all examples type-check correctly)

**Completion Date**: September 30, 2025

**Success Criteria**: All met ✅
- Catches all type errors
- Rejects invalid programs
- Accepts valid programs
- Clear error messages

## Phase 4 - Shadow-Test Runner & Interpreter ✅ Complete

**Goal**: Execute shadow-tests during compilation and provide full interpretation

**Deliverables**:
- ✅ Test extraction from AST
- ✅ Complete interpreter for shadow-tests and programs (src/eval.c - ~450 lines)
- ✅ Assertion checking
- ✅ Test result reporting
- ✅ Function call interface
- ✅ Test suite for interpreter (15/15 examples pass)

**Completion Date**: September 30, 2025

**Success Criteria**: All met ✅
- Executes all shadow-tests
- Reports failures clearly
- Full program interpretation support
- Fast execution

## Phase 5 - C Transpiler ✅ Complete

**Goal**: Transform AST to C code

**Deliverables**:
- ✅ C code generation (src/transpiler.c - ~380 lines)
- ✅ Runtime library integration
- ✅ Built-in function implementations
- ✅ Memory management (C standard library)
- ✅ Test suite for transpiler (15/15 examples compile and run)
- ⚠️ C code formatter (basic formatting, could be improved)

**Completion Date**: September 30, 2025

**Success Criteria**: All met ✅
- Generates valid C code
- Compiles with standard C compiler (gcc)
- Matches nanolang semantics
- Produces working binaries

## Phase 6 - Standard Library (Minimal - ⚠️ In Progress)

**Goal**: Provide common functionality

**Deliverables**:
- ⚠️ String operations (basic print only)
- ✅ I/O functions (print)
- ⚠️ Math functions (basic operators only, no advanced functions)
- ⏳ Data structures (arrays, lists - not yet implemented)
- ⚠️ Documentation (basic)
- ✅ Shadow-tests for built-in functions

**Current Status**: Basic functionality only

**Next Steps**:
- Add more math functions (sin, cos, sqrt, etc.)
- Implement arrays
- Add string manipulation functions
- Expand I/O (file operations)

## Phase 7 - Command-Line Tools ✅ Complete

**Goal**: User-friendly compiler and interpreter interfaces

**Deliverables**:
- ✅ `bin/nanoc` compiler command (src/main.c - ~190 lines)
- ✅ `bin/nano` interpreter command (src/interpreter_main.c - ~180 lines)
- ✅ Command-line options (-o, --verbose, --keep-c, --call)
- ✅ Help system (--help)
- ✅ Version information (--version)
- ✅ Error formatting with line numbers
- ✅ Makefile for building both tools
- ✅ Documentation

**Completion Date**: September 30, 2025

**Success Criteria**: All met ✅
- Easy to use
- Clear error messages
- Good help text
- Follows Unix conventions
- Both compilation and interpretation supported

## Phase 8 - Self-Hosting ✅ COMPLETE

**Completion Date**: January 2026

**Goal**: Compile nanolang compiler in nanolang - **ACHIEVED**

**Documentation**: See [planning/SELF_HOSTING.md](../planning/SELF_HOSTING.md) for detailed analysis

**Required Features** (6 essential) - ALL COMPLETE:
1. ✅ Structs - Represent tokens, AST nodes, symbols (November 2025)
2. ✅ Enums - Token types, AST node types (November 2025)
3. ✅ Dynamic Lists - Store collections of tokens/nodes (November 2025)
4. ✅ File I/O - Read source files, write C output (November 2025)
5. ✅ Advanced String Operations - Character access, parsing, formatting (November 2025)
6. ✅ System Execution - Invoke gcc on generated code (November 2025)

**Bootstrap Implementation**:
- [x] ✅ Implemented lexer in nanolang (December 2025)
- [x] ✅ Implemented parser in nanolang (December 2025)
- [x] ✅ Implemented type checker in nanolang (December 2025)
- [x] ✅ Implemented transpiler in nanolang (December 2025)
- [x] ✅ **3-Stage Bootstrap** working perfectly (January 2026):
  - Stage 0: C-based nanoc_c compiles Stage 1
  - Stage 1: Self-hosted components (parser, typecheck, transpiler)
  - Stage 2: Stage 1 recompiles itself
  - Stage 3: Verification (Stage 1 output == Stage 2 output)
- [x] ✅ Performance optimization (within 2-3x of C)
- [x] ✅ Documentation complete
- [x] ✅ Full test suite passing (221 tests)

**Success Criteria**: ALL MET ✅
- ✅ nanolang compiler (written in nanolang) compiles itself
- ✅ Bootstrapping process works reliably (`make bootstrap`)
- ✅ Output binaries functionally equivalent (verified via Stage 3)
- ✅ Performance acceptable (native C performance via transpilation)
- ✅ All tests pass (shadow tests + examples + 221 test files)
- ✅ Documentation complete (121+ docs)

## Phase 9 - Ecosystem & Polish (Current - v0.3.0 target)

**Goal**: Polish the project for 1.0 release and build ecosystem

**Status**: In Progress

**High Priority**:
- [ ] Complete STDLIB.md documentation (41 missing functions)
- [x] Add code coverage metrics (gcov/lcov integration) - ✅ Completed
- [x] Create ERROR_MESSAGES.md with examples - ✅ Completed
- [x] Document memory management model (MEMORY_MANAGEMENT.md) - ✅ Completed
- [ ] Expand FFI safety documentation
- [x] Create GENERICS_DEEP_DIVE.md - ✅ Completed
- [x] Add missing NAMESPACE_USAGE.md - ✅ Completed
- [x] Fix eval.c size (split into modules) - ✅ Completed (Jan 2026)
- [x] Add performance benchmarks - ✅ Completed (CI integration)
- [x] Integrate fuzzing (AFL++/libFuzzer) - ✅ Completed (Jan 2026)

**Medium Priority**:
- [ ] VS Code extension (syntax highlighting)
- [ ] Add --profile flag for performance profiling
- [ ] Create LEARNING_PATH.md for examples
- [ ] Document error handling philosophy
- [ ] Add build modes (--debug / --release)
- [ ] Unicode support planning
- [x] Expand negative test coverage - ✅ Completed (20 → 36 tests, Jan 2026)

**Low Priority**:
- [x] RFC process for language evolution - ✅ Completed (Jan 2026)
- [ ] Package manager prototype (nanopkg)
- [ ] Concurrency model documentation
- [ ] Formal grammar specification

**Target Completion**: Q1 2026

## Completed Language Features

### Core Data Types ✅
- [x] ✅ **Arrays** - Dynamic arrays with bounds checking (November 2025)
- [x] ✅ **Structs** - User-defined composite types (November 2025)
- [x] ✅ **Enums** - Enumerated types with named constants (November 2025)
- [x] ✅ **Unions** - Tagged unions/sum types with pattern matching (December 2025)
- [x] ✅ **Generics** - Monomorphized generic types (December 2025)
- [x] ✅ **Tuples** - Heterogeneous tuples (December 2025)
- [x] ✅ **First-Class Functions** - Functions as values (December 2025)
- [x] ✅ **Affine Types** - Resource management (December 2025)

## Future Enhancements

These features may be added after self-hosting:

### Language Features
- [ ] Dynamic Lists/Slices - Resizable collections
- [ ] Generics/templates
- [ ] Pattern matching
- [ ] Modules/imports
- [ ] Error handling (Result type)
- [ ] Algebraic data types
- [ ] Tuples

### Tooling
- [ ] REPL (Read-Eval-Print Loop)
- [ ] Language server (LSP)
- [ ] Debugger
- [ ] Package manager
- [ ] Build system
- [ ] Documentation generator

### Optimizations
- [ ] Tail call optimization
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Inlining
- [ ] LLVM backend (alternative to C)

### Ecosystem
- [ ] VS Code extension
- [ ] Vim plugin
- [ ] Emacs mode
- [ ] Online playground
- [ ] Tutorial website
- [ ] Community forum

## Timeline Actual vs Estimated

| Phase | Original Estimate | Actual Time | Status |
|-------|------------------|-------------|---------|
| Phase 0: Specification | - | 1 day | ✅ Complete |
| Phase 1: Lexer | 2-3 weeks | 1 day | ✅ Complete |
| Phase 2: Parser | 3-4 weeks | 1 day | ✅ Complete |
| Phase 3: Type Checker | 3-4 weeks | 1 day | ✅ Complete |
| Phase 4: Shadow-Test Runner | 2-3 weeks | 1 day | ✅ Complete |
| Phase 5: C Transpiler | 4-5 weeks | 1 day | ✅ Complete |
| Phase 6: Standard Library | 3-4 weeks | - | ⚠️ Minimal |
| Phase 7: CLI Tools | 2 weeks | 1 day | ✅ Complete |
| Phase 8: Self-Hosting | 8-12 weeks | 3 months | ✅ Complete (Jan 2026) |

**Total Actual Time (Phases 0-7)**: 2 days (September 29-30, 2025)
**Efficiency**: Much faster than estimated due to focused development and AI assistance

## Milestones

### Milestone 1: First Compilation (Phase 1-5) ✅ ACHIEVED
**Completion Date**: September 30, 2025
- ✅ Can compile simple nanolang programs
- ✅ Generates working C code
- ✅ Shadow-tests execute
- ✅ All 15 examples working

### Milestone 2: Usable Compiler (Phase 6-7) ✅ MOSTLY ACHIEVED
**Completion Date**: September 30, 2025
- ⚠️ Standard library minimal (basic functionality only)
- ✅ Command-line tools polished (compiler + interpreter)
- ✅ Documentation complete
- ✅ Ready for simple projects

### Milestone 3: Self-Hosting (Phase 8)
**Target**: nanolang compiles itself
- Compiler rewritten in nanolang
- Bootstrap process working
- Full test suite passing

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Current Focus**: Implementation planning

**Most Needed**:
1. Feedback on specification
2. Additional example programs
3. Test cases
4. Implementation volunteers

## Success Metrics

### Technical
- All example programs compile and run
- Shadow-tests catch bugs
- Generated C code is readable
- Compilation is fast
- Self-hosting works

### Community
- Clear documentation
- Active contributors
- Growing example library
- Positive feedback

### Adoption
- Real projects using nanolang
- LLMs can generate correct code
- Teaching material available
- Community resources

## Risks and Mitigations

### Risk: Specification Changes
**Mitigation**: Community review before implementation starts

### Risk: Implementation Complexity
**Mitigation**: Incremental development, extensive testing

### Risk: Performance Issues
**Mitigation**: C transpilation provides good baseline performance

### Risk: Limited Contributors
**Mitigation**: Keep codebase simple and well-documented

### Risk: LLM Generation Quality
**Mitigation**: Iterate on language design based on LLM testing

## Communication

### Updates
- Commit messages
- Release notes
- GitHub issues/PRs

### Discussion
- GitHub Discussions (when available)
- Issue tracker for bugs/features

### Documentation
- Keep docs in sync with code
- Update examples regularly
- Maintain changelog

## Versioning

Following semantic versioning (semver):

- **0.x.y**: Pre-1.0 development
- **1.0.0**: First stable release (after self-hosting)
- **1.x.0**: New features (backwards compatible)
- **x.0.0**: Breaking changes

## Release Strategy

### Pre-1.0 Releases
- 0.1.0: Lexer complete
- 0.2.0: Parser complete
- 0.3.0: Type checker complete
- 0.4.0: Shadow-test runner complete
- 0.5.0: C transpiler complete
- 0.6.0: Standard library complete
- 0.7.0: CLI tool complete
- 0.9.0: Self-hosting beta

### 1.0 Release Criteria
- Self-hosting works
- All examples compile
- Documentation complete
- Test suite passes
- Performance acceptable
- Breaking changes unlikely

## Long-Term Vision

nanolang aims to be:

1. **Reference implementation** for LLM-friendly language design
2. **Teaching tool** for programming language concepts
3. **Practical language** for systems programming
4. **Proof of concept** for shadow-test methodology
5. **Community project** with active contributors

## Questions?

For questions about the roadmap:
1. Check [SPECIFICATION.md](SPECIFICATION.md) for language details
2. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help
3. Open an issue for discussion

---

**Last Updated**: January 25, 2026 (Post-Self-Hosting Update)
**Current Phase**: Phase 9 - Ecosystem & Polish
**Next Major Milestone**: v1.0 Release (target: Q3 2026)
**Next Review**: After Phase 9 completion

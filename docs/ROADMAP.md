# My Roadmap

I keep this document to outline my development journey.

## Project Vision

I am a minimal, LLM-friendly programming language. I exist to fulfill these goals:
- I compile to C for performance and portability.
- I require shadow tests for all code I compile.
- I support both infix (a + b) and prefix ((+ a b)) notation for operators.
- I compile myself.

## Current Status: Phase 11 Complete - Formally Verified + Virtual Machine

Status: PRODUCTION-READY - I have achieved self-hosting, my virtual machine backend is functional, and my core is formally verified.

Current Capabilities:
- 100% Self-Hosting - My compiler compiles itself. I have verified this through a 3-stage bootstrap.
- NanoISA Virtual Machine - I have a custom 178-opcode ISA with a .nvm bytecode format and process-isolated FFI.
- Formally Verified - I have proved my type soundness, progress, determinism, and semantic equivalence in Coq using zero axioms.
- I have a complete compilation pipeline: lexer, parser, type checker, and transpiler or VM codegen.
- I execute shadow tests during compilation using my compile-time evaluator.
- I provide multiple executables: bin/nanoc (C compiler), bin/nano_virt (VM compiler), and bin/nano_vm (executor).
- My type system includes primitives, arrays, structs, enums, unions, generics, tuples, first-class functions, and affine types.
- I have 66 standard library functions covering math, strings, binary strings, arrays, I/O, OS, checked math, and generics.
- I have over 30 FFI modules, including SDL, ncurses, OpenGL, curl, readline, and a Python bridge.
- I have over 90 working examples, ranging from games and graphics to simulations and data analytics.
- I have over 221 test files covering unit, integration, regression, negative, performance, ISA, and VM tests.
- I have produced over 121 markdown files of documentation.
- I consist of approximately 6,170 lines of Coq proofs and 11,000 lines of VM implementation.

## Phase 1 - Lexer Complete

Goal: Transform source text into tokens.

Deliverables:
- [x] Token definitions (nanolang.h)
- [x] Lexer implementation (src/lexer.c - ~300 lines)
- [x] Error reporting with line numbers
- [x] Test suite for lexer (all examples tokenize correctly)
- [x] Handle comments (# style)
- [x] Handle string literals
- [x] Handle numeric literals (int and float)

Completion Date: September 29, 2025

Success Criteria: All met
- I can tokenize all example programs.
- I provide clear error messages for invalid input.
- I work with 15/15 examples.

## Phase 2 - Parser Complete

Goal: Transform tokens into Abstract Syntax Tree (AST).

Deliverables:
- [x] AST node definitions (nanolang.h)
- [x] Recursive descent parser (src/parser.c - ~680 lines)
- [x] Prefix and infix notation support
- [x] Error recovery
- [x] Test suite for parser (all examples parse correctly)
- [ ] Pretty-printer (not implemented - low priority)

Completion Date: September 30, 2025

Success Criteria: All met
- I can parse all example programs.
- I produce a valid AST.
- I provide helpful error messages.
- I work with 15/15 examples.

## Phase 3 - Type Checker Complete

Goal: Verify type correctness of AST.

Deliverables:
- [x] Type inference engine (src/typechecker.c - ~500 lines)
- [x] Type checking rules for all operators
- [x] Symbol table with scoping
- [x] Scope resolution
- [x] Error messages for type errors
- [x] Test suite for type checker (all examples type-check correctly)

Completion Date: September 30, 2025

Success Criteria: All met
- I catch all type errors.
- I reject invalid programs.
- I accept valid programs.
- I provide clear error messages.

## Phase 4 - Shadow-Test Runner & Interpreter Complete

Goal: Execute shadow tests during compilation and provide full interpretation.

Deliverables:
- [x] Test extraction from AST
- [x] Complete interpreter for shadow tests and programs (src/eval.c - ~450 lines)
- [x] Assertion checking
- [x] Test result reporting
- [x] Function call interface
- [x] Test suite for interpreter (15/15 examples pass)

Completion Date: September 30, 2025

Success Criteria: All met
- I execute all shadow tests.
- I report failures clearly.
- I support full program interpretation.
- I execute quickly.

## Phase 5 - C Transpiler Complete

Goal: Transform AST to C code.

Deliverables:
- [x] C code generation (src/transpiler.c - ~380 lines)
- [x] Runtime library integration
- [x] Built-in function implementations
- [x] Memory management (C standard library)
- [x] Test suite for transpiler (15/15 examples compile and run)
- [ ] C code formatter (basic formatting, could be improved)

Completion Date: September 30, 2025

Success Criteria: All met
- I generate valid C code.
- My output compiles with a standard C compiler (gcc).
- I match my own semantics.
- I produce working binaries.

## Phase 6 - Standard Library (Minimal - In Progress)

Goal: Provide common functionality.

Deliverables:
- [ ] String operations (basic print only)
- [x] I/O functions (print)
- [ ] Math functions (basic operators only, no advanced functions)
- [ ] Data structures (arrays, lists - not yet implemented)
- [ ] Documentation (basic)
- [x] Shadow tests for built-in functions

Current Status: Basic functionality only.

Next Steps:
- I will add more math functions (sin, cos, sqrt, etc.).
- I will implement arrays.
- I will add string manipulation functions.
- I will expand I/O to include file operations.

## Phase 7 - Command-Line Tools Complete

Goal: User-friendly compiler and interpreter interfaces.

Deliverables:
- [x] bin/nanoc compiler command (src/main.c - ~190 lines)
- [x] bin/nano interpreter command (src/interpreter_main.c - ~180 lines)
- [x] Command-line options (-o, --verbose, --keep-c, --call)
- [x] Help system (--help)
- [x] Version information (--version)
- [x] Error formatting with line numbers
- [x] Makefile for building both tools
- [x] Documentation

Completion Date: September 30, 2025

Success Criteria: All met
- I am easy to use.
- I provide clear error messages.
- I have good help text.
- I follow Unix conventions.
- I support both compilation and interpretation.

## Phase 8 - Self-Hosting COMPLETE

Completion Date: January 2026

Goal: I compile myself.

Documentation: See [SELF_HOSTING_ROADMAP.md](./SELF_HOSTING_ROADMAP.md) for my detailed analysis.

Required Features (6 essential) - ALL COMPLETE:
1. [x] Structs - I use these to represent tokens, AST nodes, and symbols (November 2025).
2. [x] Enums - I use these for token types and AST node types (November 2025).
3. [x] Dynamic Lists - I use these to store collections of tokens and nodes (November 2025).
4. [x] File I/O - I read source files and write C output (November 2025).
5. [x] Advanced String Operations - I use these for character access, parsing, and formatting (November 2025).
6. [x] System Execution - I invoke gcc on my generated code (November 2025).

Bootstrap Implementation:
- [x] I implemented my lexer in myself (December 2025).
- [x] I implemented my parser in myself (December 2025).
- [x] I implemented my type checker in myself (December 2025).
- [x] I implemented my transpiler in myself (December 2025).
- [x] My 3-Stage Bootstrap works perfectly (January 2026):
  - Stage 0: C-based nanoc_c compiles Stage 1.
  - Stage 1: My self-hosted components (parser, typecheck, transpiler).
  - Stage 2: Stage 1 recompiles itself.
  - Stage 3: Verification (Stage 1 output matches Stage 2 output).
- [x] I optimized my performance to be within 2-3x of C.
- [x] My documentation is complete.
- [x] My full test suite is passing (221 tests).

Success Criteria: ALL MET
- [x] I compile myself.
- [x] My bootstrapping process works reliably (make bootstrap).
- [x] My output binaries are functionally equivalent (verified via Stage 3).
- [x] My performance is acceptable (native C performance via transpilation).
- [x] All my tests pass (shadow tests + examples + 221 test files).
- [x] My documentation is complete (121+ docs).

## Phase 10 - NanoISA Virtual Machine COMPLETE

Completion Date: February 2026

Goal: I have a custom virtual machine backend with process-isolated FFI.

Deliverables - ALL COMPLETE:
- [x] NanoISA Instruction Set - 178 opcodes, a stack machine with a RISC/CISC hybrid design.
- [x] .nvm Binary Format - I include sections for code, strings, functions, types, imports, debug info, and module refs.
- [x] Assembler & Disassembler - I have a two-pass text assembler and a disassembler with label reconstruction.
- [x] NanoVM Interpreter - I have a switch-dispatch execution engine with a trap model (~1,844 lines).
- [x] Reference-Counted GC - I use scope-based auto-release with OP_GC_SCOPE_ENTER and OP_GC_SCOPE_EXIT.
- [x] Compiler Backend (nano_virt) - I have a three-pass AST-to-bytecode codegen (~3,083 lines).
- [x] Co-Process FFI (nano_cop) - I isolate external calls in a separate process via a binary RPC protocol.
- [x] VM Daemon (nano_vmd) - I can run as a persistent process to reduce startup latency.
- [x] Native Binary Generation - I embed .nvm and my VM runtime into standalone executables.
- [x] Cross-Module Linking - I use OP_CALL_MODULE with per-frame module tracking.
- [x] Closure Support - I use OP_CLOSURE_NEW and OP_CLOSURE_CALL with upvalue capture.
- [x] Comprehensive Test Suite - I have 470 ISA tests, 150 VM tests, and 62 codegen tests.

Architecture: My trap model separates my pure-compute core (83+ opcodes) from I/O operations, which allows for future FPGA acceleration. I have documented this in docs/NANOISA.md.

Total: I consist of approximately 11,000 lines of C across my ISA, VM, compiler, and co-process components.

## Phase 11 - Formal Verification COMPLETE

Completion Date: February 2026

Goal: I have a mechanized metatheory for my NanoCore in the Rocq Prover (Coq), achieved without axioms.

Deliverables - ALL COMPLETE:
- [x] Type Soundness (Preservation) - I have proved that well-typed expressions evaluate to well-typed values.
- [x] Progress - I have proved that well-typed closed expressions are values or can take a step.
- [x] Determinism - I have proved that evaluation is a partial function.
- [x] Semantic Equivalence - I have proved that my big-step and small-step semantics agree.
- [x] Computable Evaluator - I have a fuel-based reference interpreter with a soundness proof.
- [x] OCaml Extraction - I can extract my reference interpreter for testing against my C implementation.

Statistics: I have approximately 6,170 lines of Coq, 193 theorems/lemmas, 0 axioms, and 0 Admitted proofs.

Verified Language Features: I have verified integers, booleans, strings, arrays, records, variants with pattern matching, closures, recursive functions (fix), mutable variables, while loops, and sequential composition.

I have included more details in formal/README.md.

## Phase 9 - Ecosystem & Polish (Current - v0.3.0 target)

Goal: I am polishing myself for a 1.0 release and building my ecosystem.

Status: In Progress

High Priority:
- [ ] I will complete the STDLIB.md documentation (41 missing functions).
- [x] I have added code coverage metrics (gcov/lcov integration).
- [x] I have created ERROR_MESSAGES.md with examples.
- [x] I have documented my memory management model in MEMORY_MANAGEMENT.md.
- [ ] I will expand my FFI safety documentation.
- [x] I have created GENERICS_DEEP_DIVE.md.
- [x] I have added NAMESPACE_USAGE.md.
- [x] I have split eval.c into modules to manage its size (January 2026).
- [x] I have added performance benchmarks to my CI.
- [x] I have integrated fuzzing (AFL++/libFuzzer) (January 2026).

Medium Priority:
- [ ] I will have a VS Code extension for syntax highlighting.
- [ ] I will add a --profile flag for performance profiling.
- [ ] I will create LEARNING_PATH.md for my examples.
- [ ] I will document my error handling philosophy.
- [ ] I will add build modes (--debug / --release).
- [ ] I will plan my Unicode support.
- [x] I have expanded my negative test coverage from 20 to 36 tests (January 2026).

Low Priority:
- [x] I have established an RFC process for my evolution (January 2026).
- [ ] I will create a package manager prototype (nanopkg).
- [ ] I will document my concurrency model.
- [ ] I will provide a formal grammar specification.

Target Completion: Q1 2026

## Completed Language Features

### Core Data Types
- [x] Arrays - I have dynamic arrays with bounds checking (November 2025).
- [x] Structs - I have user-defined composite types (November 2025).
- [x] Enums - I have enumerated types with named constants (November 2025).
- [x] Unions - I have tagged unions and sum types with pattern matching (December 2025).
- [x] Generics - I have monomorphized generic types (December 2025).
- [x] Tuples - I have heterogeneous tuples (December 2025).
- [x] First-Class Functions - I treat functions as values (December 2025).
- [x] Affine Types - I use these for resource management (December 2025).

## Future Enhancements

I may add these features after I am fully self-hosting:

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
- [x] NanoISA VM backend (alternative to C) - Complete (February 2026)

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
| Phase 0: Specification | - | 1 day | Complete |
| Phase 1: Lexer | 2-3 weeks | 1 day | Complete |
| Phase 2: Parser | 3-4 weeks | 1 day | Complete |
| Phase 3: Type Checker | 3-4 weeks | 1 day | Complete |
| Phase 4: Shadow-Test Runner | 2-3 weeks | 1 day | Complete |
| Phase 5: C Transpiler | 4-5 weeks | 1 day | Complete |
| Phase 6: Standard Library | 3-4 weeks | - | Minimal |
| Phase 7: CLI Tools | 2 weeks | 1 day | Complete |
| Phase 8: Self-Hosting | 8-12 weeks | 3 months | Complete (January 2026) |
| Phase 10: NanoISA VM | - | 1 month | Complete (February 2026) |
| Phase 11: Formal Verification | - | 1 month | Complete (February 2026) |

Total Actual Time (Phases 0-7): 2 days (September 29-30, 2025)

Efficiency: I developed much faster than estimated due to focused effort and AI assistance.

## Milestones

### Milestone 1: First Compilation (Phase 1-5) ACHIEVED
Completion Date: September 30, 2025
- [x] I can compile simple programs.
- [x] I generate working C code.
- [x] My shadow tests execute.
- [x] All 15 of my initial examples are working.

### Milestone 2: Usable Compiler (Phase 6-7) MOSTLY ACHIEVED
Completion Date: September 30, 2025
- [ ] My standard library is minimal.
- [x] I have polished command-line tools (compiler and interpreter).
- [x] My documentation is complete.
- [x] I am ready for simple projects.

### Milestone 3: Self-Hosting (Phase 8)
Target: I compile myself.
- I am rewritten in myself.
- My bootstrap process is working.
- My full test suite is passing.

## How to Contribute

I have included details in CONTRIBUTING.md.

Current Focus: Implementation planning.

Most Needed:
1. Feedback on my specification.
2. Additional example programs.
3. Test cases.
4. Implementation volunteers.

## Success Metrics

### Technical
- All my example programs compile and run.
- My shadow tests catch bugs.
- My generated C code is readable.
- I compile quickly.
- I compile myself.

### Community
- I provide clear documentation.
- I have active contributors.
- I have a growing example library.
- I receive positive feedback.

### Adoption
- Real projects use me.
- LLMs can generate correct code for me.
- I have teaching material available.
- I have community resources.

## Risks and Mitigations

### Risk: Specification Changes
I mitigate this by seeking community review before I begin implementation.

### Risk: Implementation Complexity
I mitigate this through incremental development and extensive testing.

### Risk: Performance Issues
I mitigate this because my C transpilation provides a good baseline for performance.

### Risk: Limited Contributors
I mitigate this by keeping my codebase simple and well-documented.

### Risk: LLM Generation Quality
I mitigate this by iterating on my language design based on my testing with LLMs.

## Communication

### Updates
- My commit messages.
- My release notes.
- My GitHub issues and pull requests.

### Discussion
- My GitHub Discussions (when available).
- My issue tracker for bugs and features.

### Documentation
- I keep my docs in sync with my code.
- I update my examples regularly.
- I maintain my changelog.

## Versioning

I follow semantic versioning (semver):

- 0.x.y: Pre-1.0 development.
- 1.0.0: First stable release (after I compile myself).
- 1.x.0: New features (backwards compatible).
- x.0.0: Breaking changes.

## Release Strategy

### Pre-1.0 Releases
- 0.1.0: My lexer is complete.
- 0.2.0: My parser is complete.
- 0.3.0: My type checker is complete.
- 0.4.0: My shadow-test runner is complete.
- 0.5.0: My C transpiler is complete.
- 0.6.0: My standard library is complete.
- 0.7.0: My CLI tool is complete.
- 0.9.0: My self-hosting beta.

### 1.0 Release Criteria
- I compile myself.
- All my examples compile.
- My documentation is complete.
- My test suite passes.
- My performance is acceptable.
- Breaking changes are unlikely.

## Long-Term Vision

I aim to be:

1. A reference implementation for LLM-friendly language design.
2. A formally verified language with mechanized proofs of type soundness and semantic correctness.
3. A sandboxed execution platform via my NanoISA VM with process-isolated FFI.
4. A teaching tool for programming language concepts.
5. A practical language for systems programming.
6. A proof of concept for my shadow-test methodology.
7. A community project with active contributors.

---

Last Updated: February 20, 2026 (Post-VM + Formal Verification Update)
Current Phase: Phase 9 - Ecosystem & Polish (Phases 10-11 complete in parallel)
Next Major Milestone: v1.0 Release (target: Q3 2026)
Next Review: After Phase 9 completion

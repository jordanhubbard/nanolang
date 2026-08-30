# My Roadmap

I keep this document to outline my development journey.

I execute active work from top to bottom. Before implementation begins, I add
the work here as checkable items, including its tests and documentation. I mark
an item complete only after I have verified it. MAC tasks track ownership and
execution; this document records product direction and order.

## Active Execution Queue

### Phase 12 - NanoISA v2

Goal: I will make NanoISA a regular, compositional, verified instruction set
for NanoLang, Forth, and future frontends. My portable bytecode will remain
readable. Verification and instantiation may translate it into a faster private
form, including measured superinstructions.

Workflow and evidence:
- [x] I wrote the initial Forth-on-NanoISA architecture contract in `docs/superpowers/specs/2026-08-30-ans-forth-nanoisa-design.md`.
- [x] I added persistent `vm_invoke`, latest-function lookup, and incremental function verification on PR #115.
- [x] I measured static NanoVirt opcode and instruction-sequence frequencies across the repository.
- [x] I added a reproducible NanoISA benchmark and profiling harness before changing execution architecture.
- [x] I record opcode, pair, and triple frequencies, retired instructions, branches, call kinds, stack depths, traps, retain/release traffic, allocations, and FFI byte/latency counters.
- [ ] I benchmark NanoLang execution, allocation, direct and indirect calls, FFI, and the current Forth interpreter; compiled Forth, its compiler, and Forth exceptions remain blocked on the Phase 13 runtime.
- [x] I publish benchmark summaries with hardware, OS, compiler, commit, distributions, retired instructions, and normalized costs.
- [x] I require semantic equivalence and full quality gates for every accepted optimization in `docs/NANOISA_OPTIMIZATION_POLICY.md`.

Portable ISA design:
- [x] I defined `spec/nanoisa.yaml` as the source of truth and generate the active opcode metadata plus v2 stack and ownership metadata from it.
- [x] I use a regular local/stack hybrid: indexed locals for named state and an operand stack for expression evaluation.
- [x] I removed fictional architectural registers from the active NanoISA documentation and v2 schema.
- [ ] I will give each portable instruction one comprehensible meaning and keep operand forms symmetric.
- [ ] NanoVirt emits explicit signed integer, floating, comparison, and boolean scalar operations; unsigned, bitwise, and dynamic-value separation remains.
- [x] I added signed and unsigned division, remainder, comparison, shifts, carry, borrow, and wide multiplication primitives required by Forth double cells.
- [ ] I will add coherent indexed stack operations rather than one-off stack permutations.
- [ ] I will add byte-addressed memory loads and stores at 8, 16, 32, and 64 bits with explicit alignment behavior.
- [ ] I will replace language-specific aggregate opcodes with regular layout-driven construct, get, set, and tag operations.
- [ ] I will separate direct function references and closures so a callable has one unambiguous representation.
- [ ] I will regularize direct, indirect, tail, imported, and linked calls around verified signatures.
- [ ] I will resolve separate-module calls to callable handles during linking rather than carry module/function pairs through dispatch.
- [ ] I will replace special print, assert, and host operations with typed traps where that improves composition.
- [ ] I will move trimming, case conversion, splitting, replacement, formatting, parsing, and collection algorithms from the ISA into runtime libraries.
- [ ] I will retain only primitive string and aggregate operations justified by representation or measured cost.
- [ ] I will add compact constants, short local forms, and compact general operands without making assembly irregular.
- [ ] I will define a clean extended-opcode space without treating an opcode value as an instruction count.

Execution architecture:
- [ ] I will separate compact serialized bytecode, verified instruction IR, and optimized dispatch IR.
- [ ] I will decode and resolve each function once rather than call the generic decoder for every retired instruction.
- [ ] I will build instruction-boundary maps and resolve branches, calls, layouts, constants, globals, and imports during instantiation.
- [ ] I will provide computed-goto dispatch where supported and retain a portable switch fallback.
- [x] I moved generated source locations entirely to side tables and removed executable `DEBUG_LINE` instructions from NanoVirt output.
- [x] I made `--strip-debug` remove all generated runtime debug cost by stripping the side table from code that contains no debug opcodes.
- [ ] I will remove generated `PUSH_VOID; POP`, unreachable `RET; JMP`, and other administrative sequences in lowering before adding fusions.
- [ ] I will add tail-call lowering and execution.
- [ ] I will add profile-selected private superinstructions without exposing frontend bookkeeping as portable opcodes.
- [ ] I will initially evaluate local-field load, local increment, compare-branch, union-tag branch, and tail-call fusions.
- [ ] I will accept a fusion only when maintained NanoLang or Forth workloads justify it.

Runtime representation:
- [ ] I will measure and evaluate split payload/tag operand stacks and globals.
- [ ] I will dynamically size globals from serialized declarations instead of embedding 4,096 values in every VM.
- [ ] I will preinstantiate module constants so string literals do not allocate and search the intern table on every execution.
- [ ] I will replace linear transient-string interning or stop interning transient values.
- [ ] I will consistently use stored string lengths and preserve embedded zero bytes.
- [ ] I will add unboxed homogeneous arrays for integer, float, boolean, and byte elements.
- [ ] I will simplify array mutator stack effects and remove the two-result `ARR_POP` convention.
- [ ] I will replace chained hash-map entries with a measured contiguous implementation.
- [ ] I will fix reference ownership for array removal, closure calls, FFI trap arguments, and marshalled arrays.
- [ ] I will choose and document tracing collection or enforceable cycle restrictions for heap graphs.

Verifier and safety:
- [ ] I will implement a control-flow verifier with instruction-boundary validation.
- [ ] I will infer stack height and types through every basic block and require compatible merge states.
- [ ] I will verify call arity, result shape, aggregate counts, local/global/upvalue bounds, type tags, and import signatures.
- [ ] I will verify return shape, maximum operand depth, frame depth, ownership effects, and explicit termination.
- [ ] I will verify linked-module calls and every opcode family rather than selected operands only.
- [ ] I will eliminate integer-overflow and overlap gaps in code-range and section validation.
- [ ] I will rewrite verified operations to unchecked private handlers where the proof permits it.
- [ ] I will add malformed-bytecode tests and fuzz the decoder, loader, verifier, assembler, disassembler, and co-process protocol.

Module format and tools:
- [ ] I will design a NanoISA v2 module header with format version, ISA version, feature bits, total size, and bounded section directory.
- [ ] I will serialize required code, constants, signatures, globals, imports, layouts, links, metadata, and optional debug sections.
- [ ] I will reject duplicate singleton sections, overlaps, partial records, trailing data, and arithmetic overflow.
- [ ] I will make symbolic functions, imports, fields, types, constants, and labels first-class assembler operands.
- [ ] I will make canonical disassembly lossless and byte-length aware.
- [ ] I will validate complete operand consumption and verify every assembled module.
- [ ] I will correct disassembler import annotations, branch operand roles, label construction, and binary-string handling.
- [ ] I will add an exhaustive coverage check tying every opcode to schema, VM behavior, verifier rules, assembly, disassembly, and tests.
- [ ] I will remove or justify opcodes that no frontend emits, including no-op GC scopes and duplicated closure/string operations.
- [ ] I will choose one coherent flattened or separately linked module model and test it end to end.

FFI and traps:
- [ ] I will resolve imports once into typed call descriptors.
- [ ] I will use generated typed stubs or a general ABI layer for mixed integer and floating signatures.
- [ ] I will make argument limits consistent across imports, traps, direct FFI, and co-process calls.
- [ ] I will pass trap stack ranges instead of copying a fixed array of tagged values where measurement supports it.
- [ ] I will make co-process serialization explicitly little-endian and restore a tested large-payload path.
- [ ] I will measure cold startup, warm calls, scalar calls, strings, arrays, crashes, restarts, and batching.
- [ ] I will batch high-frequency host work rather than cross the process boundary for each element.

Documentation and acceptance:
- [ ] I will replace stale NanoISA opcode counts and architecture claims in `docs/NANOISA.md` and `docs/ROADMAP.md`.
- [ ] I will document the portable ISA separately from verified and optimized runtime representations.
- [ ] I will provide readable symbolic assembly examples for NanoLang and Forth.
- [ ] I will record why every public instruction belongs in the ISA rather than a runtime library.
- [ ] I will demonstrate performance changes with distributions, not single timing claims.

### Phase 13 - Forth 2012 on NanoISA

Goal: I will implement a standards-oriented Forth system whose colon words are
verified NanoISA functions and whose typed library words use the same import and
co-process machinery as NanoLang.

Foundation:
- [x] I selected Forth 2012 Core and every optional word set as the target.
- [x] I established the persistent NanoVM invocation boundary required by an interactive compiler.
- [x] I added and verified `examples/language/forth/pi.fs` under Gforth for 0, 1, 10, and 50 places.
- [ ] I will pin the exact maintained-standard and test-suite revisions.
- [ ] I will confirm licensing before vendoring third-party conformance files.
- [ ] I will add differential runs against a pinned Gforth release.
- [ ] I will document cells, characters, addresses, division, floats, files, terminals, blocks, limits, and ambiguous-condition behavior.

Compiler and runtime:
- [ ] I will create one mutable `NvmModule` and persistent `VmState` per Forth session.
- [ ] I will add VM-owned data, return, floating-point, and control-flow stacks.
- [ ] I will add a byte-addressable virtual Forth address space with validated allocation and file handles.
- [ ] I will implement dictionary headers, execution tokens, name tokens, early binding, immediacy, and word lists.
- [ ] I will implement nested terminal, evaluated-string, included-file, and block input sources with `SOURCE` and `>IN` restoration.
- [ ] I will compile each colon definition privately to NanoISA, verify it, then publish it atomically.
- [ ] I will compile calls to earlier definitions as stable `OP_CALL` references and `RECURSE` to the reserved current definition.
- [ ] I will compile structured control flow with a checked compile-control stack and branch patching.
- [ ] I will implement `CATCH` and `THROW` by restoring Forth stacks, locals, input sources, and NanoVM invocation state.
- [ ] I will implement typed Forth import declarations that lower to `NvmImportEntry` and `OP_CALL_EXTERN`.
- [ ] I will reject FFI signatures the active ABI cannot call correctly instead of guessing.
- [ ] I will restart an isolated FFI co-process after dynamic import-table mutation.
- [ ] I will make `SEE` disassemble the actual compiled NanoISA function and describe imported words.

Standard word sets, in dependency order:
- [ ] I will implement and test Core.
- [ ] I will implement and test Core Extensions.
- [ ] I will implement and test Exception and Exception Extensions.
- [ ] I will implement genuine double-cell arithmetic and test Double Number and its extensions.
- [ ] I will implement and test String and String Extensions.
- [ ] I will implement and test Search Order and Search Order Extensions.
- [ ] I will implement and test File Access and File Access Extensions.
- [ ] I will implement and test Memory Allocation.
- [ ] I will implement recursive, reentrant Locals and Locals Extensions.
- [ ] I will implement and test Facility and Facility Extensions.
- [ ] I will implement and test Programming Tools and Programming Tools Extensions.
- [ ] I will implement an IEEE binary64 floating stack and test Floating Point and its extensions.
- [ ] I will implement UTF-8 Extended Character and Extended Character Extensions.
- [ ] I will implement Block and Block Extensions against an explicitly disposable image.

Tests, examples, and SDL IDE:
- [ ] I will retain the existing 280 cases as regression tests while replacing their nonstandard harness assumptions.
- [ ] I will run pinned committee Core and optional-word-set tests.
- [ ] I will run licensed Forth-2012 tests and record unsupported or manual cases separately.
- [ ] I will add malformed definitions, multiline definitions, early binding, immediate words, execution tokens, overflow, unsigned output, loop boundaries, exceptions, source nesting, and UTF-8 tests.
- [ ] I will make `pi.fs` pass under my Memory-Allocation and Exception implementations with the exact 50-place output.
- [ ] I will update every file in `examples/language/forth/` to standard behavior.
- [ ] I will update `sdl_forth_ide` to launch the NanoISA-backed Forth executable.
- [ ] I will keep the SDL IDE as a PTY client rather than create a second Forth implementation.
- [ ] I will add build, PTY, file-loading, interpreter-liveness, and graphical smoke coverage.
- [ ] I will publish the precise standard-system label only after tests and required documentation support it.

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
- [x] String operations (concat, split, trim, length, format, etc.)
- [x] I/O functions (print, println, file_read, file_write, file_exists)
- [x] Math functions (abs, sqrt, pow, sin, cos, floor, ceil, round, etc.)
- [x] Data structures (arrays with bounds checking, dynamic array_push, HashMap)
- [x] Documentation (STDLIB.md)
- [x] Shadow tests for built-in functions

Current Status: Comprehensive stdlib with 40+ built-in functions.

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

Documentation: See [planning/](../planning/) for my implementation design notes.

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
- [x] I have completed my STDLIB.md documentation. Every builtin in `src/builtins_registry.c` now has an entry, and `tests/check_stdlib_docs.sh` (wired into `make test-quick` and `make test`) fails the build if the two ever drift apart again.
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
- [x] I have a VS Code extension with DAP debug support (editors/vscode/).
- [x] I have added a --profile flag and --profile-output for structured benchmark JSON.
- [x] I have created LEARNING_PATH.md for my examples (docs/LEARNING_PATH.md).
- [x] I have documented my error handling philosophy (docs/ERROR_HANDLING.md).
- [ ] I will add build modes (--debug / --release).
- [x] I have planned my Unicode support (docs/UNICODE.md).
- [x] I have expanded my negative test coverage from 20 to 36 tests (January 2026).

Low Priority:
- [x] I have established an RFC process for my evolution (January 2026).
- [x] I have a package manager (scripts/nano-pkg.sh, packages.json, docs/PACKAGE_MANAGER.md).
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
- [x] Dynamic arrays/slices — `array_push`, bounds checking
- [x] Generics/templates — monomorphized generic types
- [x] Pattern matching — `match` statement with enum/union dispatch
- [x] Modules/imports — native and FFI modules via module.json
- [x] Error handling (Result type) — `Result<T, E>` in stdlib
- [x] Algebraic data types — tagged unions with `union` keyword
- [x] Tuples — heterogeneous tuples
- [x] Parallel independence blocks — `par { }` annotation
- [x] WASM backend — `--target wasm` emits WebAssembly binary
- [ ] Explicit type conversions (`float_to_int`, `int_to_float`) in compiled mode
- [ ] Arrays of structs in compiled mode

### Tooling
- [ ] REPL (Read-Eval-Print Loop)
- [x] Language server (LSP) — `bin/nanolang-lsp` (hover, definition, completion, diagnostics)
- [x] Debugger — DAP server `bin/nanolang-dap` (breakpoints, step, inspect via VS Code)
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
- [x] VS Code extension (editors/vscode/ — syntax highlighting, LSP, DAP debug)
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

# Consolidated TODOs

**Created:** 2026-02-20
**Source:** Extracted from internal planning docs during cleanup audit.

---

## Self-Hosting and Compiler

### Bootstrap Block Bug (from BOOTSTRAP_BLOCK_BUG.md)
Store statements directly in AST blocks instead of using `statement_start` index.
- Change `ASTBlock` to hold `statements: array<ASTStmtRef>` directly.
- Estimated effort: 6-8 hours.

### Self-Host Struct Access Bug (from BUG_SELFHOST_STRUCT_ACCESS.md)
Self-hosted compiler generates wrong C for struct field access.
- Compare C output from `nanoc` vs `nanoc_v06` for the same input.
- Check `generate_expression` in `src_nano/transpiler.nano` for `PNODE_FIELD_ACCESS`.
- Verify struct field offset calculation.

### Self-Hosting Roadmap (from SELF_HOSTING_ROADMAP.md)
- Fix lexer column tracking bug in `src_nano/compiler/lexer.nano` (line 272-280).
- Begin incremental parser refactoring (Phases 1-4).
- Long-term: full modular architecture for self-hosted compiler.

### Parser Refactoring (from PARSER_REFACTOR_PLAN.md)
Split 6,743-line `parser.nano` into maintainable, testable modules (147 functions in a single file).

### Driver Integration (from DRIVER_INTEGRATION.md)
Integrate `src_nano/driver.nano` with existing compiler phases for a fully self-hosted compiler driver. Phase 1 (interface documentation) is complete.

### Session Summary Node ID Bug (from SESSION_SUMMARY.md)
Parser node ID assignment bug: `last_expr_node_id` vs actual list indices mismatch.
- Add debug output to track ID assignment.
- Compare C parser and self-hosted parser AST structures.

### Module Visibility / Transitive Imports (from MODULE_VISIBILITY_INVESTIGATION.md)
Issue nanolang-zmwa: transitive module imports do not work in self-hosted compiler (191 errors when compiling `nanoc_v06.nano`).
- Implement `ModuleCache`, `resolve_module_path()`, `extract_type_symbols()`, `process_imports()`.
- Add circular dependency detection.
- Create `src_nano/compiler/module_loader.nano` and `src_nano/compiler/symbol_extractor.nano`.

---

## Type System and Language Features

### Affine Types (from AFFINE_TYPES_DESIGN.md)
Full implementation plan for `resource struct` keyword with compile-time use-tracking.
- Parser: add `resource` keyword, parse `resource struct`.
- Type system: track resource usage (unused/used/consumed states).
- Control flow: ensure resources consumed on all branches.
- Transpiler: emit as regular structs with warnings.
- Stdlib: define FileHandle, Socket as resource types.

### Linear Types (from LINEAR_TYPES_DESIGN.md)
Design phase. Extends affine types with must-use semantics.

### Dependent Types and Contracts (from DEPENDENT_TYPES_DESIGN.md)
Design phase (P1, bead nanolang-o6fn). Incremental approach to contract-based verification.

### Result Type (from RESULT_TYPE_DESIGN.md)
Draft design for `Result<T, E>` structured error handling. ~38 implementation items.

### Module Namespace Design (from MODULE_NAMESPACE_DESIGN.md)
Draft design for `::` namespace system. 11-16 week implementation roadmap with parser, typechecker, transpiler, stdlib restructuring, and migration phases.

### Anonymous Struct Literals (from ANONYMOUS_STRUCT_LITERALS.md)
Partially implemented. Remaining:
- Function call arguments: `(draw { x: 5, y: 10 })` not yet supported.
- Arrays of structs: transpiler does not support `array<UserStruct>` yet.

### Control Flow: cond (from CONTROL_FLOW_IMPROVEMENTS.md)
Design proposal for `cond` expression. Note: COND_IMPLEMENTATION_SUMMARY.md says `cond` is implemented and deployed. This doc may describe additional improvements beyond the basic `cond`.

---

## Testing and Quality

### Shadow Test Audit (from SHADOW_TEST_AUDIT_SUMMARY.md)
Priority 1 (Critical): Add shadows to `std/datetime/` (28), `stdlib/lalr.nano` (24), `std/json/` (23), `std/regex/` (13).
Priority 2 (High): Add shadows to `modules/std/json/` (27), `modules/vector3d/` (18), `modules/std/collections/` (13), `modules/std/math/` (12).
Priority 3 (Medium): Add shadows to `parser.nano` (75), `transpiler.nano` (51), `typecheck.nano` (22).

### Shadow Scope Analysis (from SHADOW_SCOPE_ANALYSIS.md)
Variable scoping issue in shadow tests. Options B or C proposed for `src_nano/typecheck.nano`.

### Shadow Test Execution (from SHADOW_TEST_CRISIS.md)
May be resolved (CI_STATUS.md says tests now work). Needs verification.

### Test Coverage Audit (from TEST_COVERAGE_AUDIT.md)
- Update MEMORY.md with affine types and driver architecture.
- Add integration tests: `test_driver_integration.nano`, `test_affine_integration.nano`, `test_error_messages.nano`, `test_multi_module_complex.nano`.
- Update `spec.json` features_complete list.

---

## String Operations

### String Integration Phase 3 (from STRING_INTEGRATION.md)
Deferred: integrate `nl_string_t` into the transpiler. 2-3 weeks effort.

### str_concat Refactoring (from STRING_PLUS_REFACTORING_EXAMPLES.md, CODEBASE_REFACTORING_2025_12_31.md)
~30 `str_concat` calls remain in codebase (mostly in `src_nano/nanoc_integrated.nano`). Replace with `+` operator.

---

## Struct Metadata

### Self-Hosted Compiler Integration (from STRUCT_METADATA_STATUS.md)
Phase 2 in progress (60% complete).
- Add reflection function calls to `typecheck.nano`.
- Replace `init_struct_metadata()` with reflection-based lookup.
- Test full self-hosted compiler compilation.

### Reflection Roadmap (from CHANGELOG_REFLECTION.md)
Short-term: macro system for auto-declaring extern functions, helper library, perfect hashing.
Medium-term: attribute-based reflection `@derive(Reflect)`, metadata caching, cross-module support.
Long-term: runtime field value access, generic metadata structs, reflection for enums/unions/functions.

---

## Error Handling and Messages

### Error Handling Patterns (from ERROR_HANDLING.md)
User-facing reference doc with error handling patterns. Some items may describe aspirational features.

### Elm-Style Error Messages (from ERROR_MESSAGES_IMPROVEMENT.md)
P1 (bead nanolang-v7tu). Design and implementation for improved error messages.

---

## Modules and Integration

### Module Examples (from MODULE_EXAMPLES_PLAN.md)
Create practical module examples: stdio_file_processor, math_ext_scientific_calc, vector2d_physics_demo, curl_weather_api, sqlite_contact_manager, audio_waveform_synth, pt2_mod_player, unicode_i18n_demo.

### HTTP Callback Design (from HTTP_CALLBACK_DESIGN.md)
Enable NanoLang functions as HTTP route handlers.

### JSON Usage Opportunities (from JSON_USAGE_OPPORTUNITIES.md)
~12 items for integrating JSON module across the project.

### Module System Redesign (from MODULE_SYSTEM_REDESIGN.md)
Awaiting architectural decision: should Module be runtime type or compile-time only? 5 implementation phases.

---

## User Guide and Documentation

### Sidebar Generation (from SIDEBAR_GENERATION_SPEC.md)
Spec for sidebar navigation in HTML user guide. Implementation pending.

### Userguide Restructure (from USERGUIDE_RESTRUCTURE_PLAN.md)
Proposal awaiting approval. Professional technical writing restructure with "NanoLang by Example" theme.

### Profiling Case Study (from PROFILING_CASE_STUDY_USERGUIDE.md)
- Add StringBuilder to all string concatenation loops.
- Add GC batch mode (C runtime change).
- Optimize syntax highlighter.
- Design and implement arena allocation API.

---

## Showcase and Examples

### Advanced Examples (from ADVANCED_EXAMPLES_PLAN.md)
Create: nl_csv_processor.nano, nl_log_analyzer.nano, nl_sales_pipeline.nano, nl_ast_analyzer.nano.

### Showcase Applications (from SHOWCASE_APPLICATIONS.md)
NanoAmp: error handling, playlist management, more audio formats.
AST Demo: type checking, code generation, optimization passes.
New tools: code analysis tool (linter), documentation generator, custom DSL example.

### Autonomous Agents (from AUTONOMOUS_AGENTS.md)
Future: streaming responses, webhook integration, vector database, multi-repo coordination, automated deployment.

---

## Compiler Optimizations (from planning/COMPILER_OPTIMIZATIONS_DESIGN.md)
Full optimization pass framework. No phases implemented.
- Phase 1: Optimization pass infrastructure and `--optimize=N` flag.
- Phase 2: Constant folding.
- Phase 3: Dead code elimination.
- Phase 4: Tail call optimization.
- Phase 5: CSE, loop invariant code motion, inline expansion, loop unrolling.

## Compiler Module Boundaries (from planning/compiler_module_boundaries.md)
Immutable phase-boundary contracts.
- Each phase module must expose `fn run(input) -> PhaseOutput`.
- `nanoc_integrated.nano` must become a coordinator chaining phase outputs.
- C refactors must include generated headers.

## Code Coverage Infrastructure (from planning/BEAD_FIXES_SUMMARY.md)
- Add `make coverage` target with gcov/lcov integration.
- Generate HTML coverage reports.
- Add CI coverage badge and thresholds.

## Stdlib Reorganization (from planning/STDLIB_REORGANIZATION_PLAN.md, planning/EVAL_REFACTORING_PLAN.md)
- Create `src/stdlib/` directory structure.
- Reorganize 72 stdlib functions into focused modules.
- Complete documentation for ~41 undocumented stdlib functions.
- Add per-module tests.
- Update CONTRIBUTING.md with new structure.

## Generics (from planning/GENERICS_DESIGN.md, planning/GENERICS_EXTENDED_DESIGN.md)
- Parse `List<T>` syntax for generic types and functions.
- Type checker: track and create generic instantiations.
- Transpiler: generate concrete monomorphized types.
- Multiple type parameters (`Map<K,V>`), nested generics.
- `Result<T>` and `Option<T>` union types with pattern matching.
- Add `TYPE_LIST_GENERIC` to type enum, `ListInstantiation` tracking.

## Garbage Collection (from planning/GC_DESIGN.md, planning/GC_UNIVERSAL_OBJECTS.md)
- `src/runtime/gc.c`: GC allocator with reference counting and free list.
- String GC integration.
- Transpiler: generate `gc_retain()`/`gc_release()` calls.
- Mark-and-sweep cycle detection (`gc_collect_cycles()`).
- New stdlib: `array_capacity`, `array_reserve`, `array_filter`, `array_map`.
- Extend GC to structs, unions, lists (`GC_TYPE_STRUCT`, `GC_TYPE_UNION`, `GC_TYPE_LIST`).
- Language syntax for heap allocation, parser/type checker GC inference.

## First-Class Functions (from planning/FIRST_CLASS_FUNCTIONS_DESIGN.md)
- Phase B4: User guide and spec documentation updates.
- Phase B5: Code audit to refactor `src_nano/*` with first-class function patterns (parser dispatch tables, token classification callbacks, AST transformation callbacks).

## Known Bugs (from planning/KNOWN_ISSUES.md)
- (High) Expression statements not validated: parser/typechecker allows standalone pure expressions as statements without error.
- (Medium) Self-hosted typechecker misses function argument type errors: `test_function_arg_type_errors.nano` disabled in CI.

## Dep Locator (from planning/DEP_LOCATOR_NANO_REQUIREMENTS.md)
- Fix generic type inference for `at(array<T>, index)` when array is a function parameter.
- Improve generic type representation in `Type` struct (`src/nanolang.h`).
- Enhance `env_lookup_variable()` to return full type info including generics.

## LSP Server (from planning/LSP_DESIGN.md)
Full Language Server Protocol implementation (not yet started).
- Phase 1: Basic LSP (textDocument/didOpen, completion, hover, definition).
- Phase 2: Navigation (references, rename, formatting, code actions).
- Phase 3: Semantic tokens, inlay hints, workspace symbols.
- VS Code extension (`editors/vscode/nanolang/`).

## REPL Improvements (from planning/REPL_IMPLEMENTATION_PLAN.md, planning/REPL_PROGRESS.md, planning/REPL_DESIGN.md)
- Stateful session manager with persistent variables.
- Function definitions in REPL.
- Module imports in REPL.
- Tab completion, help, save/load.
- Variable reassignment (`set` statement support).
- C-native `nanorepl` binary (alternative to nano-based REPL).
- Multiline input support.
- Multi-type eval commands (`:int`, `:float`, `:string`, `:bool`).

## Package Manager (from planning/PACKAGE_MANAGER_DESIGN.md)
Full `nano pkg` CLI and ecosystem (not yet started).
- Phase 1: CLI tool and `nano.toml` format.
- Phase 2: Registry HTTP API and package signing.
- Phase 3: Lock files and vulnerability scanning.
- Phase 4: Web UI.

## Module FFI (from planning/MODULE_FFI_IMPLEMENTATION.md, planning/MODULE_SYSTEM_ANALYSIS.md)
- Complete metadata deserialization (read from `.o` files, binary-only module import).
- Improve C type mapping (pointers, C structs, C enums, function pointers).
- Module namespacing (`as` keyword qualified access).
- Module versioning.
- Complete FFI tool: C header parser in `nanoc-ffi`.

## Examples Cleanup (from planning/EXAMPLES_AUDIT_REPORT.md, planning/EXAMPLES_CONSOLIDATION_AUDIT.md, planning/EXAMPLES_MODERNIZATION.md)
- Add header comments to ~48 undocumented examples.
- Modernize 6 game examples with enums and structs (checkers, snake, maze, particles, boids, boids_complete).
- Delete 15 redundant/superseded example files.
- Create 11 new merged comprehensive examples.
- Reorganize examples into subdirectories (beginner/, language_features/, games/, modules/, advanced/).

## Test Cleanup (from planning/TEST_AUDIT_REPORT.md)
- Delete 25-30 duplicate test files.
- Move 6 tutorial tests to `examples/language_features/`.
- Consolidate generic union and array tests.
- Add header comments to all tests.

## Tracing Infrastructure (from planning/TRACING_DESIGN.md)
Full tracing system (not yet started).
- Create `src/tracing.h` and `src/tracing.c`.
- Add hooks in `eval.c` and `typechecker.c`.
- Implement command-line tracing flags.

## Transpiler Improvements (from planning/TRANSPILER_ENUM_ISSUE.md, planning/transpiler_refactoring_plan.md, planning/TUPLE_RETURN_IMPLEMENTATION.md)
- Fix proper enum field handling (currently using `int` workaround).
- Add `GeneratedTypes` tracking struct to `transpiler.c`.
- Complete transpiler refactoring (58% remaining): stdlib to separate file, function declarations, function implementations.
- Integrate `TupleTypeRegistry` into `transpile_program()` (95% done, 4-5 integration points remain).

## Self-Hosted Compiler Improvements (from planning/SRC_NANO_IMPROVEMENTS.md, planning/STAGE2_STATUS.md, planning/UNION_TYPES_AUDIT.md)
- Replace magic numbers in `lexer_v2.nano` with enum variants.
- Generic List integration (`List<Token>`).
- AST union type refactor in `ast_types.nano`.
- Fix match arm bindings in typechecker scope.
- Generic union instantiation parsing (`Result<int, string>` syntax).
- Tuple types in self-hosted compiler.
- First-class functions in self-hosted compiler.

## Showcase Games (from planning/SHOWCASE_GAMES_ROADMAP.md)
- Boids simulation.
- Procedural terrain explorer.
- Falling sand (physics sandbox).
- Music sequencer.
- Shared infrastructure modules (vector2d, spatial, noise, audio, cellular, ui).

## Web Playground (from planning/WEB_PLAYGROUND_DESIGN.md)
- WASM build of compiler.
- Share via URL, embed mode, download.
- Code formatting, keyboard shortcuts, mobile support.
- Multi-file projects, module support.
- Debugging and profiling integration.
- Backend services (sharing API, PostgreSQL, Redis).

## Safety and Quality (from planning/safety_guidelines.md)
- Add lint rules / codegen checks to forbid recursion beyond configurable depth.
- Property-based tests (QuickCheck-style) for parser/type checker invariants.
- Extend schema/contracts to cover module imports, IR transforms, and runtime list helpers.

## Interpreter (from planning/LEXER_ENUM_ACCESS_LIMITATION.md, planning/INTERPRETER_ONLY_EXAMPLES_ANALYSIS.md)
- Fix interpreter to register enum definitions before running shadow tests (currently requires magic numbers).
- Systematically test all `nl_*` examples for compilation.
- Update Makefile to compile all working examples.
- Fix remaining transpiler `print` to `nl_print_*` prefix bug.

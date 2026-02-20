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

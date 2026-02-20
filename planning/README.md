# Planning Documents

I keep my design documents and incomplete implementation plans here. Completed work has been removed. All remaining TODOs from deleted documents are consolidated in [docs/CONSOLIDATED_TODOS.md](../docs/CONSOLIDATED_TODOS.md).

---

## Active Documents

### Compiler

- [COMPILER_OPTIMIZATIONS_DESIGN.md](COMPILER_OPTIMIZATIONS_DESIGN.md) -- Optimization pass framework
- [compiler_module_boundaries.md](compiler_module_boundaries.md) -- Phase-boundary contracts
- [EVAL_REFACTORING_PLAN.md](EVAL_REFACTORING_PLAN.md) -- eval.c follow-up work
- [TRANSPILER_ENUM_ISSUE.md](TRANSPILER_ENUM_ISSUE.md) -- Enum field workaround (proper fix pending)
- [transpiler_refactoring_plan.md](transpiler_refactoring_plan.md) -- Transpiler restructuring (42% done)
- [TUPLE_RETURN_IMPLEMENTATION.md](TUPLE_RETURN_IMPLEMENTATION.md) -- Tuple return integration (95% done)
- [SRC_NANO_IMPROVEMENTS.md](SRC_NANO_IMPROVEMENTS.md) -- Self-hosted compiler improvements
- [STAGE2_STATUS.md](STAGE2_STATUS.md) -- Self-hosted compiler remaining features
- [UNION_TYPES_AUDIT.md](UNION_TYPES_AUDIT.md) -- AST union type refactor
- [DEP_LOCATOR_NANO_REQUIREMENTS.md](DEP_LOCATOR_NANO_REQUIREMENTS.md) -- Generic type inference bug

### Language Features

- [FIRST_CLASS_FUNCTIONS_DESIGN.md](FIRST_CLASS_FUNCTIONS_DESIGN.md) -- Documentation and code audit remaining
- [GENERICS_DESIGN.md](GENERICS_DESIGN.md) -- Generic types and functions
- [GENERICS_EXTENDED_DESIGN.md](GENERICS_EXTENDED_DESIGN.md) -- Extended generics (List of UserType)
- [ENUM_IMPLEMENTATION_PLAN.md](ENUM_IMPLEMENTATION_PLAN.md) -- Verify enum completeness
- [GC_DESIGN.md](GC_DESIGN.md) -- Garbage collection system
- [GC_UNIVERSAL_OBJECTS.md](GC_UNIVERSAL_OBJECTS.md) -- GC for structs, unions, lists
- [PACKAGE_MANAGER_DESIGN.md](PACKAGE_MANAGER_DESIGN.md) -- Package manager design

### Module System

- [MODULE_SYSTEM_ANALYSIS.md](MODULE_SYSTEM_ANALYSIS.md) -- Binary module import and FFI tool
- [MODULE_FFI_IMPLEMENTATION.md](MODULE_FFI_IMPLEMENTATION.md) -- FFI metadata and type mapping

### Tooling

- [LSP_DESIGN.md](LSP_DESIGN.md) -- Language Server Protocol
- [TRACING_DESIGN.md](TRACING_DESIGN.md) -- Tracing infrastructure
- [WEB_PLAYGROUND_DESIGN.md](WEB_PLAYGROUND_DESIGN.md) -- Web playground Phase 2+
- [REPL_DESIGN.md](REPL_DESIGN.md) -- C-native REPL binary
- [REPL_IMPLEMENTATION_PLAN.md](REPL_IMPLEMENTATION_PLAN.md) -- REPL feature plan
- [REPL_PROGRESS.md](REPL_PROGRESS.md) -- REPL progress tracker
- [MULTILINE_REPL_DESIGN.md](MULTILINE_REPL_DESIGN.md) -- Multiline REPL input
- [MULTI_TYPE_REPL_DESIGN.md](MULTI_TYPE_REPL_DESIGN.md) -- Multi-type eval in REPL

### Examples and Tests

- [EXAMPLES_AUDIT_PROGRESS.md](EXAMPLES_AUDIT_PROGRESS.md) -- Examples modernization progress
- [EXAMPLES_AUDIT_REPORT.md](EXAMPLES_AUDIT_REPORT.md) -- Examples audit findings
- [EXAMPLES_CONSOLIDATION_AUDIT.md](EXAMPLES_CONSOLIDATION_AUDIT.md) -- Redundant examples cleanup
- [EXAMPLES_MODERNIZATION.md](EXAMPLES_MODERNIZATION.md) -- Example modernization plan
- [SHOWCASE_GAMES_ROADMAP.md](SHOWCASE_GAMES_ROADMAP.md) -- Showcase game examples
- [INTERPRETER_ONLY_EXAMPLES_ANALYSIS.md](INTERPRETER_ONLY_EXAMPLES_ANALYSIS.md) -- Compilation testing for examples
- [TEST_AUDIT_REPORT.md](TEST_AUDIT_REPORT.md) -- Test file cleanup and consolidation

### Stdlib and Runtime

- [STDLIB_REORGANIZATION_PLAN.md](STDLIB_REORGANIZATION_PLAN.md) -- Stdlib directory restructure
- [STRING_OPERATIONS_PLAN.md](STRING_OPERATIONS_PLAN.md) -- String operation tasks

### Quality and Safety

- [safety_guidelines.md](safety_guidelines.md) -- Safety lint rules and property tests
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) -- Active bugs

### Status and Tracking

- [BEAD_FIXES_SUMMARY.md](BEAD_FIXES_SUMMARY.md) -- Coverage, stdlib, eval.c follow-ups
- [TODO.md](TODO.md) -- Legacy TODO (stale, see CONSOLIDATED_TODOS.md)
- [REMAINING_TODOS.md](REMAINING_TODOS.md) -- Legacy remaining items (stale, see CONSOLIDATED_TODOS.md)

---

For user-facing documentation, see [docs/](../docs/).
For the consolidated list of all remaining work, see [docs/CONSOLIDATED_TODOS.md](../docs/CONSOLIDATED_TODOS.md).

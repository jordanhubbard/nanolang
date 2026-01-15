# Changelog

All notable changes to NanoLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

[0;34mℹ️  Generating changelog from v2.0.5 to HEAD...[0m
## [2.0.6] - 2026-01-14

### Added
- Implement working exec_command() in process module
- Add automated release system with make release targets

### Fixed
- Fix awk error when updating CHANGELOG with multi-line entries
- Fix three P0 test failures across platforms
- Only define _POSIX_C_SOURCE if not already defined
- Use $(MAKE) instead of 'make' in version check
- Split Makefile to properly detect BSD make
- Accept reality - BSD make detection impossible
- Add proper GNU make detection for BSD systems
- Remove broken GNU make guard, add clear documentation instead
- Add unistd.h include for getpid()
- Remove Clang-only warning flags for GCC compatibility
- Fix Ubuntu build and eliminate all C compiler warnings
- Add stub implementations for beads module dependencies

## [2.0.5] - 2026-01-14

### Added
- **Beads module** (`stdlib/beads.nano`) - Programmatic issue tracking API
  - Type-safe wrappers for bd command-line tool
  - Query beads by status, priority, type
  - Create and close issues from code
  - Get project statistics
  - **Killer feature**: `assert_with_bead()` - automatically create bugs from failing assertions
  - Enhanced `assert_with_bead_context()` with file/line information
  - Complete API: `bd_list`, `bd_open`, `bd_ready`, `bd_by_priority`, `bd_create`, `bd_close`, `bd_stats`
- **Process module** (`stdlib/process.nano`) - Command execution helpers
- Comprehensive beads module documentation (`stdlib/README_BEADS.md`)
- Three practical examples:
  - `examples/advanced/beads_basic_usage.nano` - Basic querying and creation
  - `examples/advanced/beads_assert_with_bead.nano` - Automatic issue creation
  - `examples/advanced/beads_workflow_automation.nano` - Workflow automation and triage
- Complete test suite (`tests/test_beads_module.nano`) with 10 test functions

### Fixed
- **P0: For-in loop transpilation** - Loops now properly transpile to C
  - Fixed missing `AST_FOR` case in transpiler that caused loops to be replaced with `/* unsupported expr type 14 */`
  - Added proper C for-loop generation: `for (int64_t i = start; i < end; i++)`
  - Created comprehensive test suite (`tests/test_for_in_loops.nano`) with 5 test functions
  - All 148 tests pass with working loops
  - Examples: sum_range, nested loops, array iteration, custom ranges
- **P0: Boolean operator parentheses** (from v2.0.4 session continuation)
  - Added `TOKEN_AND` and `TOKEN_OR` to parentheses logic in transpiler
  - Eliminates `-Wlogical-op-parentheses` warnings
  - Ensures correct precedence: `((a && b) || c)` instead of `a && b || c`

### Changed
- Closed 2 completed in-progress beads (examples organization, match arm scoping verification)
- Updated KNOWN_ISSUES.md to mark both P0 transpiler bugs as FIXED

### Testing
- All 149 tests passing (148 existing + 1 new for-in loop test)
- Zero P0 bugs remaining
- Build system clean with `-Werror`

### Documentation
- Complete beads module documentation with API reference and examples
- Updated KNOWN_ISSUES.md with resolution details for P0 bugs

## [2.0.4] - 2026-01-14

### Added
- **GC-managed strings**: All string allocations now use garbage collection to prevent memory leaks
  - Added `gc_alloc_string()` helper function for GC-tracked string allocation
  - Updated all string producers: `nl_str_concat`, `nl_str_substring`, `int_to_string`, `float_to_string`, `string_from_char`, `nl_string_from_bytes`
  - Added `test_gc_string_leak.nano` regression test for memory management
  - Prevents unbounded memory growth in long-running programs

### Changed
- **Code modernization**: Replaced deprecated `str_concat` with canonical `+` operator throughout codebase
  - 970 lines modernized across 17 files
  - Affects compiler sources, standard library, and test files
  - Improves code consistency and follows NanoLang canonical syntax

### Fixed
- Fixed `test_module_introspection_import.nano` to use if statements instead of assert in expression context
- Fixed 12 autotest compilation failures (P4 issues)
- Removed unused variables and dummy workarounds in examples
- Suppressed unused function warnings in generated runtime code

### Documentation
- Added P0 transpiler bugs to KNOWN_ISSUES.md
- Documented critical compiler bug - expression statements not validated

### Testing
- All 147 tests pass (6 language + 133 application + 8 unit)
- Self-hosted compiler tests: 10 passed
- Build system clean with `-Werror` enabled

## [2.0.3] - 2026-01-13

### Added
- Warning suppression system to make `-Werror` practical
- Comprehensive debugging infrastructure

### Fixed
- Multiple warning suppressions in generated runtime code
- Example launcher syntax highlighting restoration

## [2.0.2] - 2026-01-12

### Added
- OPL examples directory with Makefile and sample programs
- Interactive bouncy balls demo
- Unsafe block type-checking support

### Fixed
- Example organization and build system
- Audio and physics examples
- Ball pit physics (preventing tunneling through bottom)
- Disabled failing self-hosted typechecker test until fixed

## [2.0.1] - 2026-01-11

### Added
- Break and continue statements fully implemented
- Module introspection capabilities
- Module-qualified struct names (Module.StructName)

### Fixed
- Union variant constructor compilation
- Module system qualified calls
- Heap-buffer-overflow in env_define_function
- CI workflow paths for new example structure

### Changed
- Examples organized into category-based subdirectories (language/, graphics/, terminal/, etc.)
- Root directory cleanup

## [2.0.0] - 2026-01-10

### Added
- **Complete module system redesign** with four phases:
  - Phase 1: Module metadata collection
  - Phase 2: FFI tracking and introspection
  - Phase 3: Graduated warning system for module safety
  - Phase 4: Module-qualified calls (Module.function syntax)
- **Comprehensive debugging infrastructure**:
  - Structured logging API with 6 levels (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
  - Coverage tracking and tracing APIs
  - Property-based testing framework foundation
- **Self-validating code generation**:
  - Enhanced shadow test failure messages with variable context
  - Comprehensive debugging guides
  - Property testing documentation

### Changed
- Module compilation now generates unique objects for each imported .nano file
- AST extended with MODULE_QUALIFIED_CALL node type
- Struct metadata includes field names and types

### Fixed
- Module function compilation blocker resolved
- Shadow test variable scoping issues
- Module introspection fully functional
- Union variant construction in AST_STRUCT_LITERAL

## [1.0.0] - 2025-12-XX

### Added
- Initial functionally complete release
- Core language features: functions, structs, enums, unions, generics
- Module system with imports
- Shadow testing framework
- SDL, NCurses, and other module bindings
- Self-hosting capability (partial)
- Comprehensive standard library

### Language Features
- Prefix notation syntax
- Type annotations
- Pattern matching
- Generic types (List<T>, Result<T,E>, Option<T>)
- Tuple support
- Array operations
- String operations
- File I/O

### Tooling
- Compiler (nanoc)
- Interpreter (nano)
- REPL support
- Module builder
- Example launcher

---

## Release Links

- [2.0.4]: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.4
- [2.0.3]: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.3
- [2.0.2]: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.2
- [2.0.1]: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.1
- [2.0.0]: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.0
- [1.0.0]: https://github.com/jordanhubbard/nanolang/releases/tag/v1.0.0

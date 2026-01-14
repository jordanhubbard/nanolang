# Changelog

All notable changes to NanoLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

# Changelog

All notable changes to NanoLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.1] - 2026-02-05

### Fixed
- GC refcount bug causing use-after-free on aliased string variables
- correct sidebar sorting and chapter numbering in user guide

## [2.0.25] - 2026-02-03

## [2.0.24] - 2026-02-03

### Changed
- simplify key handling with or expressions
- modernize more language examples
- more example modernization with cond and helpers
- modernize examples with cond, helpers, shadow tests
- modernize sdl_example_launcher.nano

### Fixed
- pi calculator algorithm and readline dependency check
- comprehensive ASan memory leak and UAF fixes
- add markdown table support and fix inline code link parsing

## [2.0.24] - 2026-02-02

### Changed
- Modernize examples with cond expressions, helper functions, and shadow tests
- Simplify key handling with or expressions in SDL examples
- Add rgb() helper for color packing in physics/SDL examples
- Extract is_out_of_bounds() helper for grid bounds checking
- Modernize sdl_example_launcher.nano with struct-based Example type

### Fixed
- Fix pi calculator algorithm and readline dependency check
- Comprehensive ASan memory leak and UAF fixes
- Add markdown table support and fix inline code link parsing in user guide

## [2.0.23] - 2026-02-02

## [2.0.22] - 2026-02-02

## [2.0.21] - 2026-02-02

### Fixed
- Rebuild TOC with correct page_dir for nested pages

## [2.0.20] - 2026-02-02

### Added
- Add bootstrap-profile target for self-analysis profiling

### Fixed
- macOS sample no longer requires sudo
- Fix broken links and clean up duplicate chapters
- Prefer built-in generic HashMap<K,V> over user-defined map_*

## [2.0.19] - 2026-02-01

### Added
- Complete WSL2 port with full module dependency support
  - Added WSL2 detection in module builder
  - Fixed module validation to check pkg_config names correctly
  - Fixed install-deps.sh to discover all module dependencies
  - All 20 modules with dependencies now available on WSL2
  - Created comprehensive WSL2 documentation

### Fixed
- Fixed bullet module C++ linking for Linux/WSL2 (lstdc++ vs lc++)
- Repaired bullet_soft_body_beads example to use modern SDL/Bullet API
- Fixed module validation pkg-config name mismatches (SDL2_mixer vs sdl2_mixer)

### Changed
- Updated install-deps.sh to check both index.json and individual module.json files

## [2.0.18] - 2026-02-01

### Added
- Add server-side code execution

## [2.0.17] - 2026-01-31

### Fixed
- increase release timeout to 40 minutes
- avoid running make test twice during release

## [2.0.16] - 2026-01-31

### Added
- add Linux profiling test infrastructure for ubuntu.local
- major v2 preparation - contracts, modules, docs, examples
- add contracts to StringBuilder, Matrix4, and Timing modules
- add contracts to stdlib math library
- complete contract system with better errors, static analysis, and docs
- implement requires/ensures contract clauses

### Fixed
- handle Homebrew keg-only packages for readline
- exclude broken examples from build
- prevent nested profiling recursion
- correct gprofng command syntax and output parsing
- avoid always-true comparison warning in test
- use intermediate variables for nested match bindings in binary.nano
- disable ASan leak detection in all sanitizer CI steps
- free module ASTs in clear_module_cache to fix memory leaks
- update link checker to accept .html links with .md sources
- improve physics examples with visual feedback and controls

## [2.0.15] - 2026-01-30

### Added
- add pkg-config dependency checking with auto-install
- Auto-invoke sudo for macOS profiling with helpful message
- Cross-platform profiling with fork/exec wrapper
- Auto-run gprof analysis with LLM-friendly JSON output
- Add -pg profiling flag and optimize syntax highlighting

### Fixed
- Skip syntax highlighting for large files in example launcher
- Cache syntax-highlighted source code in example launcher
- prevent icon clicks from triggering launch
- keep release logs out of changelog

## [2.0.14] - 2026-01-28

### Fixed
- deep-copy struct string fields in interpreter
- avoid sudo prompts in non-interactive installs
- add FreeBSD pkg mappings for SDL2
- use unsafe srand in random sentence shadow test
- map sdl2_image dependencies
- expose strptime in datetime impl
- add utf8proc package mapping
- include stdint in sqlite helpers
- remove llabs from ffi tutorial
- add select headers for readline
- add pkg-config for readline
- support pkg installs and timing fallback
- stabilize example builds and module linkage
- add pi cli override and quiet playground banner

## [2.0.13] - 2026-01-26

### Added
- Implement postcondition injection for ensures clauses
- Add ensures clause parsing and playground accessibility improvements
- Add NanoLang Playground - Interactive web-based REPL

### Fixed
- stabilize coverage arrays and add shadow tests

## [2.0.12] - 2026-01-26

### Added
- Complete Full-Featured REPL with functions and imports
- Add multi-line input support to REPL
- Add multi-type REPL with support for int, float, string, bool
- Add REPL implementations with persistent variable support
- Add exhaustiveness checking for match expressions
- Add color support to error messages
- Add VS Code extension for NanoLang
- Add fuzzing infrastructure for compiler robustness
- Add readline module and REPL examples

### Changed
- Modularize interpreter and enhance test infrastructure
- Consolidate REPL into single full-featured example
- Update simple_repl to use readline module

### Fixed
- stabilize pi example timing shadow test
- improve pi timing precision and test

## [2.0.11] - 2026-01-23

### Added
- Add stdlib/timing module and high-precision π calculator example

## [2.0.10] - 2026-01-23

### Fixed
- Make utility functions public and fix transpiler string concat
- Make parser_new public for module_loader access
- Force userguide tools to use C compiler and add module caching
- Add missing dependency for compiler_schema.h

## [2.0.9] - 2026-01-21

### Added
- Implement recursive import processing for type visibility
- Add module-qualified call support
- Sync C compiler with module-qualified call support
- Add module-qualified call support (part 1)
- add StringBuilder pattern for O(n) string accumulation
- BREAKTHROUGH - Stage1 compiles real programs!
- add builtin function mapping to transpiler
- Add shadow test for operator_to_string (transpiler)
- Add shadow tests for array/expr utility functions (2 functions)
- Add shadow tests for typechecker helper functions (5 functions)
- Add shadow tests for module/path resolution functions (13 functions)
- Add shadow tests for type and list helper functions (11 functions)
- Add shadow tests for string utility functions (11 functions)
- Add shadow tests for hashset/genenv/CLI utils (8 functions)
- OPL AST builder functions for JSON-based AST construction
- add syntax highlighting to inline examples on Examples page

### Fixed
- Update Parser struct initializations for module-qualified calls
- Add module_qualified_calls fields to all Parser struct initializations
- Simplify module-qualified call parsing
- Increase BOOTSTRAP2_TIMEOUT to 3600s (1 hour)
- Correct broken markdown link in TECHNICAL_WRITING_STYLE.md
- function calls now return return_type not function_type
- self-hosted typechecker now infers types for binary operators
- proper contrast for code blocks (explicit light color for pre code)
- userguide examples page code visibility (CSS color inheritance)

## [2.0.8] - 2026-01-19

### Added
- add batch mode to release script
- add process management to example launcher (PID tracking, force quit)
- NanoLang API doc generator + regenerated docs
- add core HashMap<K,V> type and map_* builtins
- add auto-generated API documentation for all modules
- add module reflection API (--reflect flag)
- add header and footer navigation to userguide pages
- use == for string comparison (replace str_equals)
- implement nano_tools eval
- use nano_tools formatting in launcher
- expand tooling and module MVPs
- add user guide hero styling
- add nanolang highlighter for user guide
- publish HTML via GitHub Pages
- add executable user guide + snippet checks

### Changed
- rename hashset param
- adopt HashMap helpers in selfhost

### Fixed
- for-in-range loop variable scope in nested contexts
- auto-install deps, snake game loop, and launcher UX
- stream make examples output instead of silently capturing it
- bootstrap env ordering
- autobeads for make examples
- stage2 typecheck build
- selfhost if syntax in typecheck
- fast inline only for API refs in CI
- speed userguide HTML in CI
- tuple and fn types in selfhost
- generic union typing in selfhost
- remove Python API doc generator, file bead for NanoLang rewrite
- stabilize userguide asset links
- userguide tooling and local serve
- avoid redundant schema rebuilds
- avoid cjson duplication in pybridge matplotlib
- stabilize module builds and userguide tooling
- make selfhost tests /bin/sh compatible
- export runtime symbols for module dlopen
- make GLUT portable (no -framework on Linux)

## [2.0.7] - 2026-01-15

### Added
- modern GL wrappers + LearnOpenGL demos
- add advanced OpenGL demos + launcher entries
- bouncy balls absolute gravity slider
- bouncy balls obstacles + gravity slider
- add pybridge helper-process Python interop module

### Fixed
- restore source viewer scrolling in SDL launcher
- repair launcher build; remove generated .genC artifacts
- Use __builtin_unreachable() in exhaustive match default case

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

# Changelog

All notable changes to NanoLang are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [3.3.6] - 2026-04-02

### Added
- source-mapped stack trace tests + multi-frame coverage

### Fixed
- sdl_forth_ide native binary path + nl_forth_interpreter TTY REPL

## [3.3.5] - 2026-04-02

### Added
- forth REPL + release sync fix
- add FIG-Forth interpreter example (token-threaded, ~2500 lines)
- generic function monomorphization (type-variable T in fn signatures)
- --doc-md flag — GFM Markdown doc export from triple-slash comments
- CodeMirror 6 editor, share permalink, AgentFS hosting
- nanolang interactive playground (browser editor + eval server, port 8792)
- complete installable .vsix — semantic tokens, format-on-save, tasks, packaging
- VS Code extension for nanolang (LSP, syntax highlighting, format-on-save)
- hot-reload — :load, :save, :reload commands
- REPL hot-reload (:reload <module>) — live module reloading without restart

### Changed
- make release script non-interactive by default

### Fixed
- effects tests — mut variable and missing shadow stubs

## [3.3.4] - 2026-04-02

### Added
- add FIG-Forth interpreter example (token-threaded, ~2500 lines)
- generic function monomorphization (type-variable T in fn signatures)
- or-patterns in match, stdlib docs site, match completeness tests
- DWARF debug info emission (--debug/-g flag for LLVM IR and RISC-V backends)
- nanolang benchmark suite — optimizer comparison + regression tracking
- REPL persistent history and session save/load
- WASM SIMD128 auto-vectorization — emit v128 opcodes for numeric patterns
- WASM SIMD128 auto-vectorization for numeric array loops
- property-based test oracle (@property, --proptest, QuickCheck-style shrinking)
- property-based test oracle (@property, --proptest, QuickCheck-style shrinking)
- REPL scripting — nano --script <file> and nano -e '<expr>' modes
- RISC-V assembly backend (riscv_backend.c)
- nano-docs static site generator — make docs builds HTML from userguide/**/*.md
- nano-bench micro-benchmark harness (--bench mode)
- nano-docs — documentation search CLI for nanolang
- typechecker generics — per-call type variable unification with consistency checking
- nano-fmt — canonical code formatter with LSP textDocument/formatting support
- agentOS typed shared-memory ringbuffer library (ringbuf.h)
- nano-to-C transpiler backend (--target c)
- nano-to-C transpiler — nanoc --target c emits clean C99 for seL4 PD embedding
- nano-to-C transpiler backend (c_backend.c)
- standard library — Option, Result, List, Map, Set, Iterator, String
- LLVM IR backend — nanoc --llvm emits .ll for ARM64/x86-64 native code
- LSP semantic tokens (textDocument/semanticTokens/full)
- LSP hover types show row-polymorphic records and type-scheme generalization
- hover displays row-polymorphic HM types
- nano package registry — HTTP server, nanoc pkg CLI, semver resolution, lockfile
- nano package registry + nanoc install/publish subcommands
- generational GC — tri-color mark-sweep cycle collection (Bacon-Rajan)
- PGO (profile-guided inlining) pass — nanoc --pgo <file>.nano.prof
- nanodoc — mdoc-style HTML doc generator for .nano modules
- cooperative scheduler runtime for async/await
- async/await syntax — CPS transform pass, cooperative async functions
- WASM reference-counting GC (refcount_gc.h) + transpiler integration
- algebraic effect system — effect declarations, handlers, effect polymorphism
- coroutine runtime — cooperative scheduler for async/await continuations
- async/await syntax — CPS transform pass, cooperative async functions
- row-polymorphic records — fix build compatibility with main branch
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels (#50)
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels
- WASM runtime profiler — --profile-runtime emits flamegraph collapsed-stack .nano.prof
- WASM runtime profiler — --profile-runtime emits flamegraph collapsed-stack .nano.prof (#51)
- row-polymorphic records — row var unification, spread, open patterns
- f-string string interpolation — comprehensive tests and userguide docs (#49)
- coverage tracking — --coverage flag, 80% CI threshold, coverage-check target (#48)
- add format/string interpolation builtin to stdlib (closes wq-API-1774937552439)
- coroutine runtime — cooperative scheduler for async/await continuations
- async/await syntax — CPS transform pass, cooperative async functions
- algebraic effect system — effect declarations, handle expressions, type checking
- add package manager (nanoc-pkg) with lockfile support
- structured output formatter — JUnit XML and TAP for CI
- NanoLang interpreter compiled to WASM via Emscripten
- WASM source maps — .wasm.map JSON + sourceMappingURL custom section
- add shadow tests for missing functions; fix stdlib redefinitions
- add WASM binary emit backend (--target wasm)
- DAP-compatible debugger server (breakpoints, step, inspect)
- nanolang Language Server Protocol server (hover, definition, completion, diagnostics)
- add --profile-output flag for structured benchmark JSON
- implement par { } blocks
- improve compiler error messages with structured Note/Hint format
- add guard clauses to match expressions (#20)
- complete stdlib expansion — 7 new builtins + bootstrap rename (#23)
- update extension to v0.2.0 for NanoLang v3.3
- implement module system – circular import detection, private visibility, shadow tests
- add 8 new string stdlib builtins (#23)
- implement ? error propagation operator in C compiler

### Changed
- make release script non-interactive by default

### Fixed
- effects tests — mut variable and missing shadow stubs
- remove duplicate AST_EFFECT_OP cases and guard unfinished SIMD block
- resolve CI compile errors from overnight branch merges
- remove stale validate-topology dependency from bench job
- merge-branches.sh use git branch -r for existence check
- resolve main build failures — missing sources, riscv bug, tidy/format collision
- wire algebraic effects syntax — comma parsing, handle forms, perform keyword
- wire full coroutine scheduler for PR #47 rebase
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- add libssl-dev to all apt jobs + -lcrypto to sanitizer/coverage LDFLAGS
- resolve CI test failures — missing test sources + f-string float formatting
- resolve merge conflicts and restore real pass implementations
- row-poly spread syntax, typechecker usage tracking, test_optimizer stub
- cross-platform random seed in sign.c for macOS/FreeBSD
- remove invalid sprint2 test; add macOS OpenSSL paths for sign.c
- add -lcrypto to sanitizer and coverage LDFLAGS for OpenSSL sign.c
- float_to_string preserves decimal point for whole-number floats
- restore missing source files and fix build for algebraic effects integration
- add fn main stubs to ug_fstring_basic and ug_fstring_exprs snippets
- move nl-snippet markers outside [?2004h)0[1;24r[m(B[4l[?7h[?25l[H[J[22;35H[0;7m(B[ New File ][m(B[?25h[24;1H[?2004lnano fences instead of before them. The userguide_snippets_check scanner does not track fence state, so it found these markers inside fenced blocks and then hit the bare nano and <!--nl-snippet ...--> lines so the marker precedes the fence, as required by the checker.
- add AST_ASYNC_FN/AST_AWAIT cases to type_infer.c
- cross-platform random seed in sign.c for macOS/FreeBSD
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- add TOKEN_ASYNC/AWAIT and PNODE_ASYNC_FN to schema so they survive schema regeneration
- tco_pass.c — suppress unused-function warning on make_return helper
- add stub sources and macOS OpenSSL path so main builds
- build on Linux/GCC — strncpy/strncat warnings and NULL guards
- add checks:write permission and continue-on-error for test reporter
- rename 'handle' in pybridge test; add main to stdlib strings test
- rename 'handle' identifier to avoid collision with new keyword
- cross-platform random seed in sign.c for macOS/FreeBSD
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- regenerate compiler schema — add TOKEN_BAR (shifts EFFECT/HANDLE/WITH/RESUME by 1)
- remove test_str_split from main() accumulator — returns array_length (3), not pass/fail count
- rename local str_index_of_from to avoid conflict with new builtin (#46)
- unblock CI after stdlib expansion (#43)
- forward-declare g_typecheck_error_count before first use in typechecker.c
- add TOKEN_ASYNC/AWAIT and PNODE_ASYNC_FN to schema so they survive schema regeneration
- remove string-returning functions from main() accumulator in test_stdlib_strings
- add missing comma after AST_PAR_BLOCK in ASTNodeType enum
- resolve CI breakages blocking PR merges (#31)
- tco_pass.c — suppress unused-function warning on make_return helper
- remove local str_trim/str_starts_with from vars_repl.nano
- fix examples-full build failures on all platforms
- remove redundant str_trim from simple_repl.nano
- add _POSIX_C_SOURCE=200809L for module compilation on Linux/FreeBSD
- add non-Apple sequential fallback in nano_dispatch.h
- guard Blocks/GCD code with #ifdef __APPLE__, add stubs for Linux/FreeBSD
- initialize env->profile in create_environment(); add platform-specific cflags support
- correct smoke test assertions to match actual REPL output
- use public fs.read/write/exists instead of private file_* variants (#38)
- use GITHUB_REPOSITORY env var instead of context.repo for PR comments
- remove userguide local redefinitions of stdlib builtins (str_starts_with, str_ends_with, str_index_of)
- repair test_stdlib_strings.nano — dedupe fn main()
- remove remaining local redefinitions of str_starts_with/str_ends_with builtins
- resolve two CI breakages on main
- resolve two CI breakages blocking PR merges (#31)

## [3.3.4] - 2026-03-30

### Added
- WebAssembly binary emit backend: `./bin/nanoc program.nano --target wasm -o program.wasm`
  supports int/float/bool types, arithmetic, comparisons, function calls, if/else, and recursion
- DAP-compatible debugger server (`bin/nanolang-dap`, `make dap`): breakpoints, step/next/stepIn/stepOut,
  variable inspection, and stack traces over JSON-RPC stdio — compatible with VS Code and any DAP client
- VS Code extension updated with DAP debug launch configuration
- Shadow tests added for all previously-untested functions across
  nl_forth_interpreter.nano, transpiler.nano, module_loader.nano, and nanoc_v06.nano
- Enable previously-disabled shadow tests for nl_path_dirname and parse_import_path_from_line in nanoc_v06.nano

### Fixed
- Remove local `str_trim` definition in example_discovery.nano (stdlib builtin conflict)
- Remove local `str_starts_with` definition in nl_forth_interpreter.nano (stdlib builtin conflict)

## [3.3.3] - 2026-03-28

### Added
- Fill in all 27 placeholder user guide pages in Part 3 (text processing,
  data formats, web/networking, graphics, OpenGL, game dev, terminal UI,
  testing, configuration) with full API references, examples, and best practices
- Fill in 3 previously-empty API reference pages (StringBuilder, coverage,
  vector2d) by marking their public functions as pub fn

### Fixed
- Reconcile user guide and API doc generator with v3.3.2 stdlib refactor:
  update stringbuilder.md to new sb_* API, update regex.md import paths,
  fix generate_all_api_docs.sh module paths
- Fix pre-existing StringBuilder_to_string shadow test (discarded append
  return value under immutable semantics)
- Fix userguide_build_html HashMap: module pub fn unreachable from regular
  functions; replace with inline get_theme_color() lookup

## [3.3.2] - 2026-03-28

### Changed
- remove redundant stdlib/std, std/json, and unused stdlib files
- retire stdlib/regex.nano in favour of std/regex/regex.nano
- modernize all test files to current nanolang syntax

### Fixed
- mark public APIs as pub fn so API reference generator picks them up
- resolve three CI failures introduced by modernize-tests commit
- restore explicit type annotations for array_pop calls in test_dynamic_arrays.nano
- resolve stage3 bootstrap blockers
- make libdispatch-dev install optional in Concurrency CI job
- add out_path null guard in ffi_loader_find_library
- use memcpy instead of snprintf in nl_walkdir_rec to avoid format-truncation
- suppress GCC format-truncation false positive in eval_io.c
- typechecker null-guard and regex test private API usage
- update .nano files to use public fs.nano API
- correct CI failures — typechecker exit code and module.c truncation
- resolve module paths correctly when cwd is not repo root

## [3.3.1] - 2026-03-27

### Added
- @pure/@associative annotations, frozen let, par blocks — C compiler + .nano sources
- f-string interpolation with automatic type conversion
- Option A dispatch wrapping for module-scope let mut primitives

### Fixed
- array_remove_at return type void→array in builtins registry
- resolve -Werror build failures (fread/system/fgets unused return, strncpy truncation)
- resolve all CI failures
- add explicit Makefile rule for nl_forth_interpreter_vm

## [3.3.1] - 2026-03-27

### Added
- @pure/@associative annotations, frozen let, par blocks — C compiler + .nano sources
- f-string interpolation with automatic type conversion
- Option A dispatch wrapping for module-scope let mut primitives

### Fixed
- resolve -Werror build failures (fread/system/fgets unused return, strncpy truncation)
- resolve all CI failures
- add explicit Makefile rule for nl_forth_interpreter_vm

## [3.3.0] - 2026-03-23

### Added
- complete roadmap features 1-11 with self-hosting pipeline
- dispatch module + parallel physics in sdl_boids and sdl_falling_sand
- add automatic zero-boilerplate concurrency via libdispatch
- add intrinsic PEG grammar support (std/peg2)

### Fixed
- make test use nanoc_c; remove debugging artifact test
- pre-populate module-level float arrays with 0.0 to set ELEM_FLOAT type
- resolve peg2 crashes, union match bindings, and example build failures

## [3.2.0] - 2026-03-11

### Added
- Local type inference: `let x = 42` infers type from RHS (no annotation required)
- Pipe operator `|>`: `x |> f |> g` desugars to `(g (f x))` for readable chains
- String interpolation `f"..."`: `f"Hello {name}!"` desugars to str_concat at compile time
- Tuple destructuring: `let (q, r) = (divmod 17 5)` binds each element directly
- Wildcard `_` in match: catch-all arm `_ => { ... }` for exhaustive pattern matching
- Anonymous functions (lambdas): `fn(x: int) -> int { return (* x 2) }` as expressions
- `for x in List<T>`: iterate List<int>, List<string>, and List<struct> directly
- `--emit-typed-ast-json`: compiler flag emitting type-annotated AST as JSON for tooling
- All 8 features implemented in both C reference compiler and NanoLang self-hosted compiler

## [3.1.12] - 2026-03-07

### Fixed
- harden module dependency auto-install across platforms
- replace static buffers in fs.c path functions; fix gc_mark for ELEM_STRUCT arrays
- make examples now builds SDL/NCurses/network examples correctly
- correct array broadcast misidentification of float literals as identifiers

## [3.1.11] - 2026-03-04

### Added
- add module metadata support to stage1 compiler

### Fixed
- resolve make examples hang and black-on-black docs styling
- work around two stage1 transpiler bugs in examples and launcher
- restore original PNG icons for example launcher
- resolve 3 pre-existing stage1 transpiler failures
- eliminate 152+ -Wparentheses-equality warnings in self-hosted transpiler

## [3.1.10] - 2026-03-01

### Fixed
- resolve make examples build failures

## [3.1.9] - 2026-03-01

## [3.1.8] - 2026-03-01

## [3.1.6] - 2026-02-25

### Fixed
- eliminate const-qualifier warnings across build pipeline

## [3.1.5] - 2026-02-23

### Added
- add inline source editor with text input and syntax highlighting

## [3.1.4] - 2026-02-23

### Added
- add default args support for examples

## [3.1.3] - 2026-02-23

### Fixed
- run from repo root so icons and source code resolve

## [3.1.2] - 2026-02-23

### Changed
- gate verbose/debug output behind --verbose flag

## [3.1.1] - 2026-02-22

### Added
- infer anonymous struct literal names from function parameter types

### Changed
- rewrite SDL launcher with modular architecture
- replace str_concat with + operator in stdlib/timing.nano, update stale TODOs

### Fixed
- resolve conflicting types and missing nl_get_time_ms in stdlib
- eliminate all -Wdiscarded-qualifiers warnings from clean build and bootstrap
- resolve remaining TODOs in self-hosted compiler
- implement outstanding TODOs across compiler and examples
- reject pure expression statements, validate function arg types in self-hosted typechecker

## [3.1.0] - 2026-01-31

### Added
- shadow test audit: add ~167 missing shadow tests across compiler files
- generic union type support in test suite

### Fixed
- reject pure expression statements in self-hosted typechecker
- validate function argument types in self-hosted typechecker
- implement outstanding TODOs across compiler and examples
- resolve remaining TODOs in self-hosted compiler

### Changed
- replace str_concat with + operator in stdlib/timing.nano
- consolidate generic_union tests

## [3.0.2] - 2025-12

### Fixed
- bootstrap stability improvements

## [3.0.1] - 2025-12

### Fixed
- self-hosting bootstrap fixes

## [3.0.0] - 2025-12

### Added
- true 100% self-hosting achieved
- complete 3-stage bootstrap verified
- NanoLang compiler written entirely in NanoLang

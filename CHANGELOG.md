# Changelog

All notable changes to NanoLang are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

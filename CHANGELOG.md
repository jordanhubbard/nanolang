# Changelog

All notable changes to NanoLang are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

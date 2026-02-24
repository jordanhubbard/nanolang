# Changelog

All notable changes to NanoLang are documented here.

## [Unreleased]

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

## [3.1.1] - 2026-02-21

### Added
- infer anonymous struct literal names from function parameter types

### Changed
- replace str_concat with + operator in stdlib/timing.nano, update stale TODOs

### Fixed
- eliminate all -Wdiscarded-qualifiers warnings from clean build and bootstrap
- resolve remaining TODOs in self-hosted compiler
- implement outstanding TODOs across compiler and examples
- reject pure expression statements, validate function arg types in self-hosted typechecker

## [3.1.0] - 2026-01-31

### Added
- Shadow test audit: add ~167 missing shadow tests across compiler files
- Generic union type support in test suite

### Fixed
- Reject pure expression statements in self-hosted typechecker
- Validate function argument types in self-hosted typechecker
- Implement outstanding TODOs across compiler and examples
- Resolve remaining TODOs in self-hosted compiler

### Changed
- Replace str_concat with + operator in stdlib/timing.nano
- Consolidate generic_union tests

## [3.0.2] - 2025-12

### Fixed
- Bootstrap stability improvements

## [3.0.1] - 2025-12

### Fixed
- Self-hosting bootstrap fixes

## [3.0.0] - 2025-12

### Added
- True 100% self-hosting achieved
- Complete 3-stage bootstrap verified
- NanoLang compiler written entirely in NanoLang

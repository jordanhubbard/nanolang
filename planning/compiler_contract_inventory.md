# Compiler Contract Inventory

This note captures the structures that historically drifted between the C reference implementation and the Nano self-hosted compiler. Each section lists the authoritative source in the new schema, the legacy locations that previously diverged, and outstanding consumers.

## Tokens / Lexer

| Component | Legacy location(s) | Schema fields |
| --- | --- | --- |
| Token enum ordering | `src/nanolang.h`, `src_nano/compiler/lexer.nano`, `src_nano/parser.nano` | `schema.compiler_schema.json -> tokens[]` (auto-generated into `src/generated/compiler_schema.h` & `src_nano/generated/compiler_schema.nano`) |
| Token struct payload | `Token` in C runtime vs. `LexerToken` in Nano | `token.fields` section (type-safe mirroring in both outputs) |
| Keyword maps | Multiple hand-written switch ladders per module | Consumption now via generated enum values + shared lexer module |

## Parse nodes / AST arenas

| Component | Legacy location(s) | Schema replacement |
| --- | --- | --- |
| `ParseNodeType` enum | `src/parser.c`, `src/nano/parser.*`, `nanoc_integrated.nano` | `parse_nodes[]` list; emitted into both languages |
| Struct layouts (`ASTNumber`, `ASTFunction`, etc.) | Copied across `parser.nano`, `transpiler.nano`, `typecheck.nano`, `nanoc_integrated.nano` | `nano_structs[]` definitions emitted via `src_nano/generated/compiler_ast.nano` |
| Parser storage (`Parser` struct, tuple literal arenas, counters) | Duplicated per module | `Parser` struct entry in schema; all modules import from generated file |

## Type / Environment contracts

| Component | Legacy location(s) | Schema replacement |
| --- | --- | --- |
| `Type` struct + `TypeKind` enum | `typecheck.nano`, `nanoc_integrated.nano`, C headers | Generated from schema (now part of `compiler_ast.nano`) |
| `TypeEnvironment` / `TypeCheckState` | Divergent definitions across C + Nano | `type_environment` section; consumed everywhere via generated files |
| Symbol tables | Locally defined (still Nano-specific) | Shared `Type` definitions ensure symbol payloads remain consistent |

## Diagnostics & Phase Outputs

| Component | Legacy location(s) | Schema replacement |
| --- | --- | --- |
| Ad-hoc error structs | `stderr` strings sprinkled per phase | `CompilerDiagnostic`, `CompilerSourceLocation`, `DiagnosticSeverity`, `CompilerPhase` |
| Phase hand-offs | `compile_file` passed raw lists/structs | `LexPhaseOutput`, `ParsePhaseOutput`, `TypecheckPhaseOutput`, `TranspilePhaseOutput` now define immutable contracts |

## Runtime list helpers

| Component | Legacy location(s) | Current status |
| --- | --- | --- |
| `List<Token>` glue (`list_token.*` vs `List<LexerToken>`) | C runtime + Nano generics | Both sides consume the enum/structs emitted by the schema; list helpers now operate on canonical `LexerToken` |

## Ownership / Sources of Truth

1. **Schema**: `schema/compiler_schema.json` is the only place new fields/enums are added.
2. **Generator**: `scripts/gen_compiler_schema.py` emits Nano + C bindings (`src_nano/generated/*.nano`, `src/generated/compiler_schema.h`).
3. **Build integration**: `Makefile` target `schema` plus the `test-impl` prerequisite `scripts/check_compiler_schema.sh` guarantee regenerated artifacts match the repo.

This inventory satisfies `nanolang-0phd.1` by documenting every contract that previously diverged and pointing to the canonical replacements.

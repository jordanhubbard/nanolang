# Nano compiler module boundaries

## Goals

1. Eliminate ad-hoc structs between compiler stages by publishing canonical outputs.
2. Keep all phase data immutable so different implementations (C, Nano) can reason about the same contracts.
3. Ensure diagnostic reporting uses shared metadata for CI and editor tooling.

## Canonical contracts

All contract types now live in `schema/compiler_schema.json` and are generated into:

- `src_nano/generated/compiler_schema.nano` – tokens, parse node ids, environments.
- `src_nano/generated/compiler_ast.nano` – AST nodes, parser storage, and phase interfaces.
- `src/generated/compiler_schema.h` – C mirror definitions.

Key shared enums:

| Enum | Purpose |
| --- | --- |
| `TypeKind` | Type checker classification for `Type`. |
| `DiagnosticSeverity` | Severity for diagnostics (info/warn/error). |
| `CompilerPhase` | Source of diagnostics (lexer/parser/typecheck/transpiler/runtime). |

Shared structs added by this task:

- `CompilerSourceLocation` – normalized file/line/column triple.
- `CompilerDiagnostic` – phase + severity + code + human message + location.
- `LexPhaseOutput`, `ParsePhaseOutput`, `TypecheckPhaseOutput`, `TranspilePhaseOutput` – immutable snapshots used at phase boundaries.

## Phase boundaries

### Lexer → Parser

```
struct LexPhaseOutput {
    tokens: List<LexerToken>
    token_count: int
    diagnostics: List<CompilerDiagnostic>
    had_error: bool
}
```

* Single source of truth for token buffers and lexer errors.
* Parser only depends on `List<LexerToken>` and `token_count`; CLI/reporting inspects `diagnostics` without reinterpreting lexer state.

### Parser → Type checker

```
struct ParsePhaseOutput {
    parser: Parser
    diagnostics: List<CompilerDiagnostic>
    had_error: bool
}
```

* `Parser` already stores every AST node arena plus counters; this wrapper keeps the phase immutable and attaches diagnostics that the CLI can surface even if parsing failed.

### Type checker → Transpiler

```
struct TypecheckPhaseOutput {
    environment: TypeEnvironment
    diagnostics: List<CompilerDiagnostic>
    had_error: bool
}
```

* `TypeEnvironment` replaces former `TypeCheckState` across both C and Nano builds.
* Transpiler reads the environment for type metadata (structs, functions) without re-running semantic passes.

### Transpiler → Runtime/build

```
struct TranspilePhaseOutput {
    c_source: string
    diagnostics: List<CompilerDiagnostic>
    had_error: bool
    output_path: string
}
```

* Records the generated C translation unit and filesystem destination so build scripts can run gcc or cache artifacts deterministically.

## Usage plan

1. Each phase module will expose `fn run(input) -> PhaseOutput` using these structs.
2. `nanoc_integrated.nano` will become a coordinator that chains `LexPhaseOutput`, `ParsePhaseOutput`, etc., and aggregates diagnostics for CLI display.
3. Future C refactors will include the same header (`src/generated/compiler_schema.h`) to guarantee matching layouts.

This specification closes `nanolang-prih.2` by establishing immutable boundary contracts that every implementation will import instead of mutating bespoke structs.

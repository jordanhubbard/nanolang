# Functional compiler architecture research

Goal: capture patterns from functional/immutable compiler pipelines that inform Nano’s new phase contracts.

## References surveyed

| Compiler | Key references |
| --- | --- |
| GHC (Haskell) | GHC Commentary on the pipeline (Core/HIR/STG), focus on explicitly-typed IR transitions and immutable `ModGuts` records. |
| rustc | Rustc-dev-guide sections on queries + typed arenas for HIR/MIR; emphasizes returning `TyCtxt` handles rather than mutating globals. |
| Sorbet / Ruby 3 YJIT | Phase diagrams showing `ResolveResult`, `TypedResult`, etc. with shared diagnostic structs. |

## Takeaways applied to Nano

1. **Immutable phase outputs**: Each compiler produces a snapshot struct (e.g., GHC’s `ModGuts`, rustc’s `TyCtxt`) that downstream passes treat as read-only. Nano now mirrors that with `LexPhaseOutput`, `ParsePhaseOutput`, `TypecheckPhaseOutput`, `TranspilePhaseOutput`.
2. **Shared diagnostic envelope**: All surveyed compilers annotate errors with phase+severity+location. `CompilerDiagnostic` + `CompilerSourceLocation` replicate that pattern.
3. **Code generation as pure function**: rustc’s HIR→MIR and GHC’s Core→STG transformations are pure functions returning new arenas. Nano’s transpiler now exposes `transpile_phase(parser, output_path)` returning `TranspilePhaseOutput` without mutating global buffers.
4. **Schema-first data contracts**: GHC stores interface files describing exported types. Our JSON schema + generator play the same role: single source of truth for tokens/AST/type env.
5. **Composable orchestration**: Query-based compilers treat each phase as a function `PhaseInput -> (PhaseOutput, Diagnostics)`. `src_nano/compiler_modular.nano` now composes phases exactly this way.

Remaining opportunities: adopt persistent arenas (like rustc’s typed arenas) for AST storage and explore query-style memoization for modules/imports. This document fulfills `nanolang-prih.1`’s research summary requirement.

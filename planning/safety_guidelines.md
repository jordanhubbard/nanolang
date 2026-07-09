# NanoLang Safety & Reliability Guidelines

This checklist maps NanoLang compiler/runtime goals to practices inspired by safety-critical standards (JSF‑AV, MISRA) and functional languages like Haskell.

## Deterministic semantics
- Ban unspecified behavior: no implicit conversions, no uninitialized variables, and no hidden global mutations in compiler modules.
- Require every function to return a value explicitly; forbid fall-through control flow (already enforced by Nano’s structured syntax).

## Exhaustive handling
- All `match`/`if` trees over enums (e.g., `ParseNodeType`) must either enumerate every case or provide a diagnostic for unknown variants.
- Schema-driven generation ensures adding a token/AST node automatically surfaces compile breaks in unhandled code.

## Pure phase contracts
- Each compiler phase returns an immutable `*PhaseOutput` snapshot with diagnostics and `had_error`, mirroring Haskell’s pure functions.
- No phase mutates global state; downstream code treats outputs as read-only data.

## Error propagation
- Diagnostics use `CompilerDiagnostic` with phase/severity/code/location so every failure is traceable.
- New helper structs (e.g., `OptionType`) replace sentinel values, forcing callers to acknowledge missing data.

## Tooling enforcement
- `scripts/check_compiler_schema.sh` must run before tests/commits to guarantee schema & generated bindings sync across Nano and C.
- Future CI steps should rebuild both C reference and self-hosted compilers, rejecting schema drift automatically.

## Future work toward JSF/MISRA alignment
1. Add lint rules (or codegen checks) to forbid recursion beyond a configurable depth inside the compiler.
2. Provide property-based tests (QuickCheck-style) for parser/type checker invariants.
3. Extend schema/contracts to cover module imports, IR transforms, and runtime list helpers so no structural duplication remains.

These guidelines make NanoLang’s compiler pipeline more fault-intolerant and pave the way toward formal compliance with conservative safety standards while retaining functional-programming ergonomics.

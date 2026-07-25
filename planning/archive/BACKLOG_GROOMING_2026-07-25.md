# Nanolang Backlog Grooming — Fan-in Integration Record (2026-07-25)

**Purpose:** Record the cooperative backlog-grooming pass for nanolang and the
prioritized next-step backlog it produced. This note also documents that the
integrated base already contains the completed child work fanned in from the
dependent tasks, so the grooming was performed against the up-to-date tree.

## Integration state verified

The working tree used for grooming already contains the completed child work:

- Match-arm binding scope fix and `nl_control_*` core test coverage
  (`tests/nl_control_flow.nano`, `tests/nl_control_match.nano`,
  `tests/nl_control_while.nano`).
- Tuple-return typedef / `TupleTypeRegistry` integration and tuple tests
  (`tests/nl_types_tuple.nano`, `planning/TUPLE_RETURN_IMPLEMENTATION.md`).
- Array element-type propagation and `tests/nl_functions_array_param.nano`.
- `formal/Equivalence.v` is Admitted-free (verified: no `Admitted` tokens).
- Module-metadata embedding re-enabled
  (`src/module_metadata.c`, `tests/nl_functions_module_metadata.nano`,
  `tests/test_modules/metadata_probe.nano`).
- Proof-status and README/FEATURES/PLATFORM_COMPATIBILITY doc reconciliation.

Verification of the combined tree:

- `make build` — 3-stage bootstrap completed (Stage 1/2/3 all green).
- `make test-quick` — 12 passed, 0 failed, 0 skipped (all child-added tests
  included).

## Prioritized backlog (next 5 steps)

Each item is genuinely open (not part of the already-completed child work) and
is grounded in current source. Priority is highest-leverage-correctness-first,
then breadth of language/backend coverage, then ergonomics.

1. **Serialize nested/aggregate TypeInfo in module metadata.**
   `src/module_metadata.c` still emits `.structs = NULL`, `.enums = NULL`,
   `.unions = NULL` and leaves recursive `TypeInfo` (parameters, return_fn_sig,
   element_type, type_params) unhandled (`TODO` at lines ~18, 127, 143, 157,
   227, 475-481). Complete serialization so imported modules expose full type
   metadata; add a test extending `tests/nl_functions_module_metadata.nano`
   that asserts struct/enum type info round-trips.

2. **Broaden LLVM backend AST coverage.**
   `src/llvm_backend.c` fails loudly (error → xfail) on strings, structs,
   arrays, tuples, field access, `match`, effects, and `par` (see the
   `default:` case ~line 348 and the header note ~line 19). Implement at least
   arrays/tuples/field-access and `match` lowering so the cross-backend driver
   stops classifying them as compile errors; done when the LLVM backend runs
   the corresponding `tests/nl_*` cases without hitting the unsupported-node
   error path.

3. **Make stdlib list combinators element-type-generic.**
   `stdlib/list.nano` combinators (`map`, `filter`, `fold`, `zip`, `nth`, ...)
   are hardcoded to `int` element types. Generalize them over a generic element
   type using the existing generics support, keeping the current int-based API
   working. Done when a new `tests/nl_functions_*` test exercises `map`/`filter`/
   `fold` over a non-`int` element type and passes under `make test-quick`.

4. **Implement `unsafe module name { ... }` declarations.**
   `src/parser.c` (~line 5349) still has `unsafe module name {...} - declaration
   (TODO: implement)`. Implement parsing + downstream handling for the unsafe
   module declaration form and add a focused parser/transpiler test. Done when
   an `unsafe module` declaration parses, transpiles, and is covered by a test.

5. **Add ergonomic `Result` error-handling helpers.**
   `stdlib/result.nano` / `modules/std/result.nano` exist but the ergonomics are
   flagged partial in `planning/REMAINING_TODOS.md`. Add combinators such as
   `map`, `map_err`, `and_then`, `unwrap_or`, and a `?`-style propagation helper
   where feasible, with a new `tests/nl_*` test covering the happy and error
   paths.

## Rationale

Ordered highest-leverage-correctness-first: metadata serialization and LLVM
coverage close real functional gaps (silent NULLs / xfail'd constructs), then
combinator genericity and the unsafe-module parser fill language-completeness
gaps, and Result ergonomics improves day-to-day usability last.

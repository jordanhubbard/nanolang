# Shared match policy: analysis preservation and NanoCore boundary

I qualified the first implementation prerequisites for my shared match policy
without changing match dispatch or no-success behavior.

## What I changed

- I visit every present match guard during effect analysis, CPS validation and
  HM type inference. HM guards unify with `bool`.
- I clone, substitute and recursively inspect guards during PGO inlining.
- I keep each match payload binding in lexical scope while PGO substitutes an
  outer formal and while HM infers the guard and body. A same-named payload is
  not rewritten to the call argument or typed from the outer formal.
- I resolve identifiers that occur only inside guards in my LSP hover path.
- I reject guarded matches at my NanoCore subset and exporter boundaries. My
  trust report says `I do not model match guards in NanoCore`; I do not erase a
  guard to obtain a formal expression.
- I zero-initialize the LSP document singleton so additions to its analysis
  state do not make strict builds depend on positional initializer updates.

## Bounded checks

On Darwin, from the isolated `feat/shared-match-policy-short-paths` checkout
rebased onto canonical `d2b6c771600e422aeb7e9e5238b5138f57a465bf`, I passed:

```text
make -j8 test-pgo-pass
make -j8 test-effects
make -j8 test-type-infer
make -j8 test-opt-passes
make -j8 test-nanocore-unit
make -j8 test-lsp-match-guard
make -j8 test-parser test-typechecker
```

The focused regressions prove a cloned-and-substituted guard, a guard-only
unknown effect, a non-BOOL HM guard refusal plus BOOL control, a guard-only
invalid `await`, an exact NanoCore trust-report refusal plus unguarded control,
and a guard-only LSP hover lookup. The adjacent parser and typechecker suites
also pass.

After independent review found that the first PGO/HM fixtures used only
literal or unshadowed guards, I added paired lexical-scope controls. On Darwin,
Apple clang 21 compiled the corrected PGO and HM paths under
`-Wall -Wextra -Werror -std=c99`; all 10 PGO methods and the full HM unit suite
pass at tree `8a17dee9ebe671f0973c0ffd15657ddd773c89fb`. The retained log is
`/private/tmp/nanolang-match-policy-lexical-scope.log`, SHA-256
`e6d30c909ecd0ae18288355a1f84dbe6a01c9e6c30339628b4c3ac5881c4bcb1`.

I then repeated the complete focused gate from a fresh detached checkout at
exact source `780d85136e0d5866896d4ec362d094526bcb3ddf`, tree
`92f2fe6ba7c94f7fdd3103634e83b59786b83abe`. Before and after inventories
record the same clean source archive SHA-256
`40a3c41e001361ecbf0796746833089a26803a34d24cc554d9c5afc168d77231`
and the same compiler, make, Python and Git identities. The eight requested
targets pass in 27 seconds. I retain the raw log and status under
`/private/tmp/nanolang-pr786-final.G39y1a/evidence`; `SHA256SUMS` has SHA-256
`4ec6a35db273d4362407580010db686985b2d0c27019a03cf47a7beff836fc59`.

## Preserved failing prerequisites

The aggregate `make test-nanocore` does not pass on this Darwin toolchain for
two failures outside these source changes:

- `tests.test_reference_eval_transport` reaches status 17 rather than the
  expected status 3 on its missing-program/PATH control, then LeakSanitizer
  reports the harness's 10-byte early-return leak. I recorded
  `task_c564d9e9089845ab9566b004a7828643`.
- `tests/test_nanocore_export_buffer.c` redefines the active SDK's fortified
  `vsnprintf` macro under `-Werror`. I recorded
  `task_43dea95525b24546b4b3e259a3148205`.

I did not disable LeakSanitizer, suppress the SDK warning or relabel the direct
NanoCore unit result as an aggregate pass.

## Boundary

This evidence does not qualify lexical first-success dispatch, wildcard
ordering, static totality, terminal no-success behavior, a product candidate or
a release. Tasks `task_477bdd430a1442e7bc19cbacdbac0bde` and
`task_70c5a56802e44142af0f19da2469f654` remain open.

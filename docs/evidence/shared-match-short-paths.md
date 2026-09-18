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
- I keep that HM payload opaque. I do not have exact variant-record facts in
  this pass, so a direct payload guard is refused rather than inferred as
  `bool`. A distinct outer `bool` name remains usable in the same arm.
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

I first repeated the complete focused gate from a fresh detached checkout at
exact source `780d85136e0d5866896d4ec362d094526bcb3ddf`, tree
`92f2fe6ba7c94f7fdd3103634e83b59786b83abe`. That retained seal predates the
opaque HM payload correction and is superseded for that narrow claim.

I then recorded the narrowing contract before code, pushed production
checkpoint `ec9a4e53b3bce415a68af9723917931601887598`, tree
`46bcd9ae62adf88e3d04bf3de03076a4b8a4a6d5`, and ran no qualification until
after that checkpoint. From a new clean detached checkout, the same eight
targets passed in 30 seconds. The paired HM method refuses a same-named arm
payload used directly as a guard even though the enclosing formal is `bool`;
renaming the arm payload makes that distinct enclosing `bool` visible and the
control passes. The paired PGO methods preserve the same-named arm identifiers
and substitute the unmatched outer formal.

Before and after inventories record the same clean source archive SHA-256
`67323d8c8978569f0259dc6d4bd18590471b82428f180cdad7e1310a38e5dd6b`,
source/tree identity and compiler, make, Python and Git hashes. I retain the raw
log, status and inventories under
`/private/tmp/nanolang-pr786-hm-final.gRmxNx/evidence`. Their SHA-256 values are:

```text
b7b47edc0f4c04306c8101b25a8e4b59c4cbd8a52dbe5d6a42cd4f861dca16da  inventory-before.log
4d4a73d5858f884121e113caddd9e2ebe85605882f8f71bd73bd7973169e38ba  focused.log
38cab61a162afe8451e6d29ff531c6ebd666447f89dda20bb09896a4e2d51349  focused.status
b0dd404c1091d89bb5135c8250bfbb03f996498d66a7c16e6d335647f9943cc9  inventory-after.log
```

The retained `SHA256SUMS` manifest has SHA-256
`9e3e2c00065131a502e9e0fd64f5041155f28f49f6145ff98a59138d5bb33306`.

The original inventories identify `/usr/bin/cc`, `/usr/bin/make` and
`/usr/bin/git` with the same SHA-256 because these are Apple tool dispatchers.
That hash is not the selected Clang executable's hash. At supplement time,
`xcrun --find clang` selects
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
a native arm64 executable with SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`
and version `Apple clang 21.0.0 (clang-2100.3.34.2)`. That version matches the
original `/usr/bin/cc --version` output. I did not record the `xcrun` path or
selected binary hash before the gate, so the original inventory does not prove
that exact compiler-binary identity at gate time. I retain this limitation and
the current lookup in `compiler-identity-supplement.log`, SHA-256
`a5869ae2701758620315e7151966f84cfbb950300de3c477e3b59792d3bba1f5`.

This is deliberately narrower than exact guarded payload typing. Exact variant
field facts remain a prerequisite before HM can admit payload-derived guards.

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

# My record/union source repair evidence

I retain initial failures and corrected terminals for this source repair.

- `raw-tests.log`: the paired gate exposed three C constructor type-checking failures.
- `instrumented-first-failure.log`: LeakSanitizer found the untransferred nested-array leaf name.
- `compound-first-failure.log`: conditional block emission lost its value and an inline generic match lost concrete payload context.
- `instrumented-tests.log`: all five focused methods pass with scoped parser/checker/codegen instrumentation and instrumented generated native consumers. Other compiler objects are ordinary objects; this is not a fully instrumented compiler.
- `bootstrap-extern-failure.log`: fresh Stage 2 refuses `List<CompilerDiagnostic>` after the shared shape helper incorrectly excludes declared extern records. The corrected bootstrap below passes with the new shadow checks.

I compress exact original log bytes. `logs.json` records their uncompressed hashes and sizes. These intermediate results do not qualify #522 for merging or release.

My current C seed passes all 31 callback-boundary/function-value methods and all 23 retained compatibility methods. Fresh Stage 1 passes 20 of the 23 compatibility methods; only the three aggregate/integer-selector match-result methods fail. The complete fresh-stage matrix is recorded below.

The corrected `make bootstrap nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump`
completes successfully. All 44 generic identity/selected-pattern/selected-ownership
methods and all 21 canonical-publication/emitter-driver methods pass. The latter
retains exact unsupported-array-of-union diagnostics and prior-output checks,
and executes the formerly refused nested record-array result through both
producers, verification, VM and native C.

The unchanged 23-method compatibility matrix passes 20 methods across all three
compilers. Its remaining three match-result methods fail at Stage 1; a separate
Stage 2 run reproduces those same three failures. I retain both terminals. The
C-only run passes all 23. This is a reduction from the retained 16-failure baseline,
not completion of the compatibility requirement.

All five `tests.test_source_record_unions` methods pass with rebuilt normal tools in 76.015 seconds, including separate self-hosted shadow-module emission and generated native ASan/UBSan/leak checks. The scoped compiler-instrumentation terminal remains separately identified above.

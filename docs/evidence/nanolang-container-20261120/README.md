# My first NanoLang translator component

At `ac163f5d1` I implement the v2 binary container boundary in
`src_nano/nvm2c/container.nano`. All decoding, bounds checks, overlap/duplicate
checks and CRC32 arithmetic are NanoLang code. I do not call a C decoder,
verifier, checksum helper or translator to make these decisions.

My first complete `make test-nanolang-translator-container` passes in3.335s:
165 independently constructed real byte files,117 explicit expected decisions,
and all165 decisions compared with the existing C reference through direct
C-seed native execution, C-produced NanoISA/NanoVM and sanitizer-instrumented
C AOT. This is container validation, not whole-module or executable admission.

I then build the full NanoLang driver from `src_nano/nanoc_v06.nano` with the
current C seed, without changing the existing shadow deadline or selection.
The build succeeds. With that exact executable I run:

```sh
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/nvm2c-full-source-compiler" \
  python3 -m unittest -v tests.test_nanolang_translator_container
```

All165 decisions match across five execution routes in4.749s. The additional
routes are actual output from that NanoLang driver, executed in NanoVM and
translated to sanitizer-instrumented native C. My seven useful function shadows
remain in the imported library and default dependency selection is retained.
The native instrumented products use the existing ordinary AOT runtime object;
I do not call this full runtime sanitizer qualification.

My embedded-fixture gate passes all five checker controls and scans3275 static
fragments with zero reserved names. Its133 lexer-refused fragments remain
reported with their owning executable tests. Python compilation and
`git diff --check` pass.

I preserve all first/corrected terminals. The first test driver lacks the CLI
extern declarations, then places unsafe expressions where the C backend cannot
emit the expected values; the corrected fixture uses declared calls in an
ordinary unsafe statement block. The raw single-file emitter refuses the
imported `wire.validate` call as expected; I use the full module driver. One
premature full-driver invocation reports missing output while its build is
still live. I poll that original build to successful completion and do not
restart it. These are fixture/command setup results, not source-compiler
regressions or reasons to weaken admission.

`manifest.json` records the implementation pin and post-run source/tool hashes.
The unit test's temporary modules/native images are deleted on success, so this
is reproducible component evidence, not an archived exact-candidate product or
complete SDK receipt. I retain full compressed build/test/setup logs.

My [translator contract](../../NANOLANG_NANOISA_C_TRANSLATOR.md) maps the
remaining payload, module/instruction verification, representation, emitter,
service, driver/bootstrap and cutover work. #1142 and #976 remain open.

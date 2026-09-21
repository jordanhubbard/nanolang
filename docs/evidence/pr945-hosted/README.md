# My retained PR945 hosted failures

I keep PR945 at exact head `061e8a5fbcd00124076d6cdae79bfeb2be2dac45` while its checks run. These reports belong to CI run `35618552172`; they do not replace my qualified private generated-backend evidence.

My coverage job `106395496115` fails three `ArtifactImports` native links with unresolved `__gcov_init`, `__gcov_exit` and `__gcov_merge_add`. The merged native helper reads `LDFLAGS` from its environment, but my Make recipe relies on automatic export after `override LDFLAGS += ...`. My retained non-product Make probe shows the Make value still contains the requested flags while the child environment has no `LDFLAGS`. I record task `task_96a578e4e4d6f645c396c0a7c453a677` before changing the owning recipe. I will pass exact Make `CC` and `LDFLAGS` explicitly to the Python suite, preserving the higher-priority `NANO_NATIVE_TEST_CC` selector and all assertions. I require the actual owning Make invocation with ordinary, coverage and sanitizer providers.

The adjacent flat-record and tuple fixtures use their existing explicit native selector and compile runtime source with their own sanitizer settings; they do not read this `LDFLAGS` variable. The shadow fixture does not perform the artifact native links. I do not infer a global export policy from this bounded correction.

My Linux x64 job `106395496224` and ARM job `106395496132` each select 176 verifier programs, verify 174 and fail two. Both actual 524-member diagnostic archives retain `test_u8_basic` rejecting the declared byte value type and `token_value_bytes` missing `list_LexerToken_insert`. These are my existing open tasks `task_c6b2a040c1434fc784a9d46c02a4981e` and `task_7b805000dfda4da386b55d4691e8c647`; I do not duplicate or close them.

My Pages run `35618552096` was cancelled after bootstrap. Its repository-wide `userguide-pages` concurrency group cancels earlier runs, and a newer root-branch run started before this cancellation. The retained log contains no preceding product failure; I do not count a cancelled guide build as passing.

My manifest records exact stored and raw bytes. Gzip files decompress losslessly; ZIP files are the actual downloaded GitHub artifacts. No failing product workload is replayed during this diagnosis.

## Final hosted state

All checks are terminal at my unchanged head. The jobs actually check out GitHub's synthetic merge of `061e8a5fb` into `87e0cef0e`, recorded in their checkout logs. My sanitizer job `106395495847` fails the same three native links with missing ASan/UBSan runtime symbols; this is another manifestation of task96a, not an executed sanitizer report. My macOS job `106395496291` also reports 176 selected, 174 verified and the same two compiler refusals. Its third actual 524-member diagnostic archive is retained alongside Linux's archives.

My other completed checks pass: strict examples on both architectures, docs links, API docs, concurrency, code quality, performance/benchmark checks, formal proofs and C/PTX/RISC-V backend checks. Pages remains cancelled. I neither rerun nor bypass these outcomes, and my full release criteria remain open.

The reviewed one-line correction is source `585a6bc12` on this separate branch. My fresh actual Make gate supplies `CC` and `LDFLAGS` only as command-line assignments, removing inherited values first. Its [separate qualification](../pr945-make-flags/README.md) passes the original 90 methods in ordinary, coverage and explicitly mixed sanitizer-runtime configurations, while retaining the fully instrumented preparation timeout. These results do not relabel this hosted run. This correction does not modify runtime or fixture assertions.

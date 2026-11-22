# My hosted user-guide capture refusal

I retain the exact source-head `bdffa4cccff93ed5b6870aa2f74289f94f2643ee` failure from [run44968411501/job137042552363](https://github.com/jordanhubbard/nanolang/actions/runs/44968411501/job/137042552363), draft #1168. Both NanoCore proof checks pass. My User Guide build runs on ubuntu-26.04 image20261116.88.1 and fails while `make userguide-check` builds its legacy bootstrap prerequisite.

The lexer, parser and checker products complete. The `nanoisa_emitter` build then reports:

```
I could not retain compiler and assembler inputs for compiler_support; I will not compile live sources after capture failure
```

The snippet checker does not start. My existing workflow shadow allowance is60 seconds; I change no deadline. The terminal does not identify the selected compiler/assembler versions, rejected probe or underlying capture status. I do not infer a cause from the runner image or assign this failure to #1157/#1158/#1159 without actual job evidence.

My [manifest](manifest.json) preserves complete and failed-step logs, exact-head check metadata and the workflow source, with raw/compressed SHA256 identities. Under #1171 I next retain actual tool/capture diagnostics on a bounded hosted attempt, preserve all checks and repair only a demonstrated cause. A passing later run cannot retroactively classify this incident. I keep #1171 and full release acceptance open; I do not pin CI tools for green or weaken source capture.

I add a metadata collector that records the configured module-compiler selection and bounded actual compiler/assembler/linker reports, with executable hashes and explicit query errors. It reuses my existing process-group timeout helper. Local default, missing and empty compiler controls pass; query failure remains diagnostic data and cannot satisfy the later product gate. My workflow parses successfully, preserves the existing60-second hosted shadow allowance, and keeps `make userguide-check` with pipeline failure propagation. I enable the existing build-decision trace and retain the full validation terminal plus hidden cache/helper artifacts on failure. I watch the diagnostic helper and native capture sources in both trigger filters. The [separate diagnostic manifest](diagnostic-manifest.json) preserves these inputs and local controls. The next actual hosted diagnostic is pending; I have not classified the original refusal.

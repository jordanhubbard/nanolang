# My hosted user-guide capture refusal

I retain the exact source-head `bdffa4cccff93ed5b6870aa2f74289f94f2643ee` failure from [run44968411501/job137042552363](https://github.com/jordanhubbard/nanolang/actions/runs/44968411501/job/137042552363), draft #1168. Both NanoCore proof checks pass. My User Guide build runs on ubuntu-26.04 image20261116.88.1 and fails while `make userguide-check` builds its legacy bootstrap prerequisite.

The lexer, parser and checker products complete. The `nanoisa_emitter` build then reports:

```
I could not retain compiler and assembler inputs for compiler_support; I will not compile live sources after capture failure
```

The snippet checker does not start. My existing workflow shadow allowance is60 seconds; I change no deadline. The terminal does not identify the selected compiler/assembler versions, rejected probe or underlying capture status. I do not infer a cause from the runner image or assign this failure to #1157/#1158/#1159 without actual job evidence.

My [manifest](manifest.json) preserves complete and failed-step logs, exact-head check metadata and the workflow source, with raw/compressed SHA256 identities. Under #1171 I next retain actual tool/capture diagnostics on a bounded hosted attempt, preserve all checks and repair only a demonstrated cause. A passing later run cannot retroactively classify this incident. I keep #1171 and full release acceptance open; I do not pin CI tools for green or weaken source capture.

I add a metadata collector that records the configured module-compiler selection and bounded actual compiler/assembler/linker reports, with executable hashes and explicit query errors. It reuses my existing process-group timeout helper. Local default, missing and empty compiler controls pass; query failure remains diagnostic data and cannot satisfy the later product gate. My workflow parses successfully, preserves the existing60-second hosted shadow allowance, and keeps `make userguide-check` with pipeline failure propagation. I enable the existing build-decision trace and retain the full validation terminal plus hidden cache/helper artifacts on failure. I watch the diagnostic helper and native capture sources in both trigger filters. The [separate diagnostic manifest](diagnostic-manifest.json) preserves these inputs and local controls. My next actual hosted diagnostic is retained below; I do not retrospectively establish the earliest return in the original untraced refusal.


My diagnostic [run44971375929/job137051691372](https://github.com/jordanhubbard/nanolang/actions/runs/44971375929/job/137051691372) also fails before snippet validation; both NanoCore proof gates pass. Its synthetic PR merge48bdbfe6f1402c91f8f51b72f9b6c18e8d60e453 has the same Git tree581726062b41d4a128941b764b2134b824d99760 as my source head c90da7166439297e3c56032be8c0f6076e3a0852. Actual bounded tool reports establish GCC15.2.0-16ubuntu1 selecting /usr/bin/x86_64-linux-gnu-as, GNU binutils2.46.

My unchanged production module_assembler_version_supported predicate accepts only qualified2.40/2.42 banners. I compile a probe that includes the actual complete production translation unit and calls that predicate with the actual collected system2.42 and hosted2.46 banners. It exits0 and prints1 then0. This demonstrates an unqualified-assembler barrier in the observed hosted configuration. It does not establish successful2.46 capture, the earliest failing branch of either job, or an upstream assembler defect causing this terminal. Failed private staging directories are removed by the builder, so the uploaded artifact does not contain the failed capture records.

My [hosted manifest](hosted-manifest.json) retains the downloaded hidden failure artifact, tool reports, full validation terminal, native products/helper, actual production source and bounded predicate probe with raw/compressed hashes verified. I keep #1171/#1157 open. I preserve capture assertions, flags and deadlines and do not add a2.46 admission token or select another CI compiler to obtain a green result.

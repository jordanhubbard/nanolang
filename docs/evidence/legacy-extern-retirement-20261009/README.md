# Legacy extern regression retirement

I retire `src_nano/transpiler.nano` under [#983](https://github.com/jordanhubbard/nanolang/issues/983). I freeze the declaration-regression helper closure as test data, not a product backend. Its 31 original function bodies and their shadows preserve historical prototype selection, name handling and type formatting. I retain the five required declared globals, exact source provenance and every original `main`/`shadow main` assertion. I move the assertion source into an explicit fixture joined by the Python test runner.

My unchanged assertions check runtime declaration selection, the absence of accidental prefix matches, the three custom extern prototypes, `bool` argument/result spelling and direct Boolean type conversion. This historical C-text regression does not claim that arbitrary extern symbols are executable through my closed host-import allowlist.

My initial extracted-source probe passes all 18 commands: checked generation, verification, VM execution, native translation, strict sanitizer compilation and native execution through each of NanoVirt, Stage 1 and Stage 2. After removing the product emitter, the final regression and adjacent phase/bootstrap controls pass all 20 methods in 146.904 seconds. VM and native runs produce the exact expected declaration-success message; ASan/UBSan and leak detection remain enabled.

`make test-nanoisa-extern-declarations` is part of `test-units`; the old `test-transpiler-externs` name delegates to it. Its prerequisites require the current bootstrap. I have launched that full Make gate after the source deletions; its fresh fixed-point and installed-fixture receipt remains pending. The preceding bootstrap belongs to the earlier source inventory and is not relabeled as this deletion's qualification. Linux and full release-candidate checks remain open.

I retain text sources, logs and artifact hashes. Generated modules, objects and executables are not committed.

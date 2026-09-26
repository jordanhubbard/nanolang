# My aggregate-global qualification

I retain the original failures and local repairs here. My base commit and final source/product hashes are in `checkpoint.json`. These checks run on Darwin; they do not establish final hosted, Linux or full LLVM/Wasm qualification.

I give each exact record global stable value storage, preserve variant and recursive field facts, copy records on load, and trace managed fields through my existing global roots. Saved copies retain nested strings, arrays and maps after replacement. I continue to refuse incompatible aggregate/scalar global representations and resource-bearing global ownership.

My C seed now resolves nominal union/enum globals, emits their exact declared types, rejects mismatched union initializers, and snapshots map string results before replacement or removal can invalidate them. The original comparator failure did not identify the execution phase; I do not infer that phase from its assertion alone.

My broader unchanged aggregate-initialization fixture exposed another refusal: an integer global is carried as a tagged value at a map store. I pass that carrier to the existing runtime map setter, which checks the exact destination value tag. Added raw controls cover integer/string success and wrong or missing tags.

## Retained checkpoints

- `pr522-globals-source.log`: three original C-seed comparison failures; canonical native/VM paths passed.
- `pr522-globals-expanded.log`: integer-key map rejection in four canonical paths. C-seed comparison passes. This remains separate required work under `task_f85d854f0e0b4c4dbe4bb1cb8a3083a0`; the C-seed-only regression does not establish canonical integer-key support.
- `pr522-globals-translator.log`, `pr522-globals-translator-corrected.log`, `pr522-globals-translator-final.log`: retained intermediate projected-global inference and incompatible-store failures.
- `pr522-globals-translator-isolated.log` and `pr522-globals-translator-sanitized.log`: corrected pre-tagged-map translator passes 2,641 assertions and 1,614 shape checks. Sanitizers use fresh private objects with ASan/UBSan, leak detection and stack-use-after-return detection.
- `pr522-globals-bootstrap-gates.log`: fresh compiler bootstrap, C-seed typechecker and transpiler gates pass. This bootstrap used the aggregate repair before the last negative-only shape constraint and tagged-map correction. Later translator builds qualify those changes independently; this is not a final compiler fixed-point claim.
- `pr522-globals-neighbors.log`: 74 ordinary methods pass, including enum globals and map fields retained inside copied records.
- `pr522-globals-source-sanitized-final.log`: 21 source methods pass with instrumented generated products, including resource-global refusal controls. This is not full linked compiler/VM instrumentation.
- `pr522-globals-emitter.log`: 86 bytecode comparisons and 90/91 Python methods pass; the added native execution of the unchanged aggregate-initialization fixture refuses the tagged integer map value.
- `pr522-globals-tagged-target.log`: that original fixture passes after the tagged-map correction, through both emitted modules and both VM/native execution routes.

## Final tagged-map correction

- `pr522-globals-tagged-ordinary.log`: fresh isolated translator passes 2,657 assertions and 1,614 shape checks, including checked integer/string map stores and wrong/missing tagged values.
- `pr522-globals-neighbors-final.log`: all 74 ordinary methods pass in 72.270 seconds.
- `pr522-globals-source-tagged-sanitized.log`: all 21 source methods pass with generated-product ASan/UBSan/leak/UAR in 35.308 seconds.

- `pr522-globals-tagged-sanitized.log`: fresh private translator and shape objects pass 2,657 assertions and 1,614 shape checks with ASan/UBSan/leak/UAR; the driver confirms actual object instrumentation.

- `pr522-globals-emitter-final.log`: all 86 bytecode comparisons and 91 Python methods pass in 361.388 seconds. The unchanged aggregate initialization fixture now also executes natively for both emitted modules.

 MAC rejects direct open-to-completed transitions for the C-seed and tagged-map tasks; I retain those responses and do not bypass review. The aggregate-global parent remains open for full-compiler and hosted qualification.

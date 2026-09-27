# My reference aggregate-formatting parity repair

I retain and execute the three original shadow programs from `../aggregate-format-ownership`: enum-valued record fields, whole-float arrays and nested arrays. I do not change their expected output or bypass their shadows.

My interpreter now uses the scalar float formatter for both static and dynamic array elements, recursively formats static nested-array Values, and consults declared field type names when formatting enums inside records and ordinary unions. Direct enum conversion remains numeric, matching the existing C-native rule.

The mixed enum/union/nested-float regression exposed a separate constructor-checker defect: a declared enum payload remained classified as struct. I resolve that nominal payload and use the existing exact-type comparison. Negative controls reject an unrelated enum with the same numeric value and a raw integer, preserving prior output. This ordinary enum contract is tracked by the task in `pr522-format-parity-enum-task.json`; generic/resource-bearing union work remains separate.

## Retained checkpoints

- `pr522-format-parity-before.log`: the original shadow disagreements and mixed case fail before correction.
- `pr522-format-parity-after.log`: all three archived shadow programs pass after the formatter repair; the mixed case exposes enum-versus-struct constructor refusal.
- `pr522-format-parity-corrected.log`: the first nominal constructor correction still rejects the valid enum due to an unnormalized expected type.
- `pr522-format-parity-final-source.log`: the valid mixed case passes, but two negative controls expose overly permissive enum compatibility.
- `pr522-format-parity-compiler-instrumented.log`: a fresh instrumented C seed reproduces those two negative-control failures. The other methods pass with instrumented compiler internals and generated products.
- `pr522-format-parity-exact-enum.log`: the exact enum correction passes all five focused methods ordinarily.

`pr522-format-parity-instrumented.py` builds privately owned fresh compiler objects and verifies ASan/UBSan symbols in eval/typechecker. It retains leak and use-after-return detection, runs original shadows, instruments generated products, and removes its private build directory after completion. The final fresh run (`pr522-format-parity-compiler-instrumented-final.log`) passes all five methods in 20.602 seconds with compiler internals and generated products instrumented. `pr522-format-parity-bootstrap-gates.log` passes fresh Stage 1/Stage 2 bootstrap, smoke checks, the complete typechecker suite and the transpiler gate. `pr522-format-parity-neighbors.log` passes all 61 ordinary formatting, generic-function/signature, global-resource-boundary, aggregate-global and scalar-string methods through the rebuilt stages in 46.834 seconds.

Canonical aggregate string lowering, full platform qualification and the wider PR522 acceptance criteria remain open under `task_fe7abd6028d14ae387e70e0e83b8885e` and their existing parent tasks.

MAC rejects a direct open-to-completed transition for the locally verified enum task. I retain `enum-task-close.txt` and its updated evidence without forcing lifecycle completion. The formatting parent remains open.

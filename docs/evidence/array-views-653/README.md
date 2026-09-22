# My full checker and evaluator qualification at 653

I qualify reviewed source `653af65401d8fde9253f5671c96ccc785fae340e`
with fresh stage1 providers and the full original `test-typechecker` and
`test-eval` Make targets. I select no fixture subset and change no original
assertion. The checker includes52 new arithmetic result-view controls.

| Host/configuration | Whole checker | Whole evaluator |
| --- | ---: | ---: |
| Linux GCC ordinary | 0.635s PASS | 2.797s PASS |
| Linux GCC ASan/UBSan/LSan | 0.858s PASS | 4.283s PASS |
| Darwin Apple Clang ordinary | 0.927s PASS | 1.607s PASS |
| Darwin Homebrew Clang ASan/UBSan/LSan | 1.149s PASS | 2.753s PASS |

All four fresh builds and eight test phases pass within unchanged1200s bounds.
I keep leak detection enabled and use no suppression. Actual retained sanitized
executables expose ASan and UBSan symbols on both hosts. Every existing provider
path, source file and compiler/tool identity agrees at its endpoints; all
supervised groups are gone. I retain all eight test executables.

`seal.json` and `seal.py` bind104 report members and1136 rehashed durable local
CAS objects (396640893bytes). The four Git bundles carry reports; CAS object
paths are local evidence rather than a claim that binaries are published in Git.
`run.py` is the actual driver. The earlier full610 refusal remains separately
sealed in `../eval-610`.

This establishes these full C fixture gates on this source. It does not establish
Nano compiler shadow deadlines, the full producer/backend matrix or release
readiness. Actual byte-array arithmetic and nested-array source admission remain
open: refusing a complete destination view does not fix coarse expression
admission in every context.

# I preserve strict C condition rendering

Hosted run 35934003714 rejects 18 scalar reconstruction methods because C
conditions render as `if ((a == b))`. I reproduce the pure-loop comparison
failure locally after building the missing nvm2hl prerequisite. The first
unprepared invocation is retained separately; missing tooling is not the
comparison defect.

I remove only the binary expression's outer grouping when the C if/while
statement supplies that grouping. Operand parentheses, evaluation snapshots,
typed region analysis and NanoLang rendering remain unchanged. I retain
-Wall -Wextra -Werror and ASan/UBSan in the generated-C harness.

The unchanged complete `make test-scalar-reconstruction` gate passes all
60 methods in 205.665 seconds after fresh bootstrap. This includes VM/native,
reconstructed C and reconstructed NanoLang execution through the C seed and
both native stages, with the route-specific sanitizer checks defined by the
harness. Native bootstrap binaries differ; smoke tests pass. This does not
establish canonical bytecode fixed points or hosted platform acceptance.

The separate shared-purity reproduction still fails its stage-2 function
reference case. Its callback local accepts only a declared identifier, while
existing computed-call lowering already resolves exact selector functions.
That repair, resource/generic callbacks, instrumented compiler deadlines and
final release gates remain open. #522 stays draft.

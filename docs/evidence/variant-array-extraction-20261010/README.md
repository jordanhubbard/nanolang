# My finite variant array extraction repair

My [finite integer-array carrier contract](../../NATIVE_VARIANT_ARRAY_CARRIERS.md), implemented by `b78b3a2c7`, explicitly admits scalar and exact `array<int>` members in one variant payload set. The old union-padding test still expected integer arrays to be refused. I replace that obsolete expectation with executed tag-only acceptance and preserve output-retaining refusal for string-array payloads. I do not extend the admitted set.

Running the contract's existing execution tests exposed a real regression: five subcases failed when direct `ARR_GET` constrained the whole finite variant payload to an exact array. The retained `before.log` includes both producer orders and alias/churn failures.

I now derive a separate exact integer-array shape for checked extraction. The producer's finite payload set remains intact. I apply that rule to reads, pop, removal and slicing; slicing returns a new exact array rather than copying the input's scalar alternatives. Runtime tag, integer storage-kind and presence checks remain unchanged.

## My evidence

- I pass 23 focused methods across union padding, scalar/array carriers and optional array reads, with VM execution and LLVM ASan/UBSan/LSan generated C execution.
- My added operation control slices, removes, pops and reads absent elements through the retained carrier. My wrong-member controls verify that reading, popping, removing or slicing a scalar member fails in both VM and native execution; native execution reaches an invariant rather than an AddressSanitizer memory failure.
- The existing scalar/array source fixture passes the current C producer and both installed Stage 1/Stage 2 producers, module verification, VM and sanitized C execution. I compare module bytes before and after consuming each artifact. This does not refresh or qualify the installed compilers for newer language changes.
- I pass the complete 2,443-check C AOT target with 379 callable and 3,098 shape constraints and its Python prerequisites. I then add the finite-carrier and union-padding targets as prerequisites of `test-nvm2c` so that this contract cannot silently fall outside that gate. I forward the selected C compiler to both test targets. Their explicit Make invocations pass all 13 carrier methods and three padding methods, plus the 3,098 shape constraints.

Callable arrays in the legacy evaluator/native backend remain open, as do full bootstrap/platform qualification and my complete 5.1 release scope. GitHub API access failed; my issue update is retained locally.

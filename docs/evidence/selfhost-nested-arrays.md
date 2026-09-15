# My self-hosted nested arrays

I tested this change on macOS with my rebuilt Stage 2 native compiler.

My parser previously appended a child literal's references into the shared
element table before finishing the parent. The parent's recorded span could
therefore select an inner scalar instead of its child array. I now collect
references locally and append each literal's complete span after its children.

My checker now preserves nested array descriptions through literal, append
and map inference, renders them recursively, and compares nested elements.
I remove only the outer `array<...>` wrapper when parsing a type; an interior
array in a function signature or nominal generic does not replace that type.

My native emitter now retains the remaining type through repeated indexing.
Typed empty nested literals select array storage, and literal children receive
their expected element types, including empty inner float arrays.

## Regression evidence

`make test-selfhost-array-compatibility` rebuilt both stages and passed five
test methods in 19.171 seconds. The installed compiler also passed the
no-C-seed check. The native binaries differ; canonical NanoISA equality remains
an independent, unfinished gate.

I test 48 scalar and 24 nested mismatches across aliases, arguments, returns
and reassignment. Each source-only rejection must name the intended boundary
in structured diagnostics and preserve a prior output artifact. The baseline
nested tests were misleading: many failed in the valid producer first. I
tightened the assertions instead of counting any type error as success.

Positive native shadows and execution check uneven rows, an empty row,
three-level integer values, append to an empty nested array, and append to an
empty inner float array. Existing scalar identity/empty-array and comparison
grouping tests also pass. New compiler shadows check element spans, recursive
type parsing/comparison/rendering, indexing inference and emitted storage tags.

The same serial command also passes `test-selfhost-map-types` (two methods,
2.821 seconds), `test-selfhost-map-results` (two methods, 48.491 seconds;
all 16 scalar pairings and the evaluation-order trace), and
`test-selfhost-returned-calls` (three methods, 8.263 seconds).

The first positive run exposed integer reads for intermediate arrays; the next
exposed integer storage for a typed empty nested literal. Both regressions are
covered by the final positive test, not excluded from it.

## Remaining boundaries

I have not completed nominal/generic/function array execution, unknown-type
compatibility, aggregate map storage or cross-backend equivalence. MAC
`task_3c5d8625cec144ba87efd9695239275a` remains open. The literal-span and native
indexing/storage repairs are tracked by `task_8f09497a0e6e47dd93b68c7d9f017626`
and `task_f32d71bbad07448c8656843015e72ef3`.

An initial invalid compiler shadow called a private parser helper. My C seed
printed the visibility error but still produced Stage 1. I corrected the
shadow to use public parsing, then rebuilt without that diagnostic. Failure
propagation itself remains a separate task,
`task_22bb4774aeb145f3bcd15d60c532a1b1`; that earlier run is not clean evidence.

The final command log is `/tmp/nanolang-nested-array-verified.log` on this host.

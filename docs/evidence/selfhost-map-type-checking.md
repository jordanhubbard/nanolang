# My self-hosted map signature checks

I add `check_map_call` before generic builtin handling. I require two
arguments, an array source, a unary function transform and a known non-void
result. For known input element types I compare the source array type with
the transform parameter's array type. I return the callback-derived array
type so local inference can backfill it into the AST for native emission.

Direct map initializers also check their declared result type without the
general array-compatibility wildcard. I do not claim this fixes that wildcard
for aliases, function arguments or returns. Complete recursive/nominal type
metadata, unknown-element arrays and aggregate runtime parity remain open.

## Evidence on Darwin, 2026-09-15

Before the change, `python3 -m unittest tests.test_selfhost_map_types` exposes
eight invalid programs accepted by `--target c`, replacing prior artifacts.
Log: `/tmp/nanolang-map-types-before.log`.

`make test-selfhost-map-types test-selfhost-map-results` rebuilds both stages
and passes their smoke/no-C-seed checks. The type gate passes two methods in
2.835 seconds: eight negative cases and one inferred-result execution.
The scalar matrix and execution trace pass two methods in 46.648 seconds.
Log: `/tmp/nanolang-map-type-gates.log`.

`make test-selfhost-returned-calls` also passes its three methods in
8.309 seconds. Log: `/tmp/nanolang-map-types-returned.log`.

The negative cases are wrong map arity, non-array source, non-function
transform, two-parameter transform, void transform, incompatible input for
named and returned transforms, and an incompatible result annotation.
Each requires its specific diagnostic, positive failure status, and exact
preservation of a preexisting output file. Source-only mode never invokes
the native C compiler, so C rejection cannot satisfy this test.

The positive case infers a float array from a returned transform, checks
length and indexed values in both shadows and the published executable.

The broader map task remains `task_75b340982b6cf797f29b38c1a188aab3`.
The shared array wildcard is separately tracked as
`task_850ac4914d9b4a9cbc34f7f16dd1902c`. I have not run the full release suite
or established complete type soundness with this checkpoint.

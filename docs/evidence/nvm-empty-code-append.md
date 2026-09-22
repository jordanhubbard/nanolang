# I preserve empty code appends under UBSan

I qualify cfddc48df on Linux from canonical d2af2d98d. My one-line early return
makes a zero-length append return the current code offset without pointer
arithmetic, allocation, or copying. Nonempty behavior is unchanged. The original
PR939 UBSan null-source report remains in this bundle; I did not rerun unfixed code.

Fresh GCC 13 and Clang provider builds both pass the original `test-nanoisa`
and `test-verifier` targets with strict warnings, O3, undefined-behavior sanitizer,
`-fno-sanitize-recover=all`, and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
Each run passes 2988 ISA checks, 96 verifier controls, verifier allocation cleanup,
and 33 schema tests. The append control checks empty NULL and nonnull inputs,
exact offsets, unchanged capacity/pointer/bytes and subsequent nonempty append.

The GCC command takes 14.101 seconds; Clang takes 8.438 seconds. Both leaders are
reaped, both groups disappear, and neither times out. All 1320 selected tracked
inputs and five selected tool identities match afterward. Exact compile/link
commands, raw streams, provider hashes and the unchanged input maps are retained
in the [manifest](nvm-empty-code-append/manifest.json).

I relocate the retained object directories losslessly into the report tree and
record every original and retained path. The [archive](nvm-empty-code-append/archive.json)
retains those products. Passing test executables removed by the original Make
recipes are not reconstructed. These are scoped Linux UBSan gates; I do not claim
Darwin qualification, a complete sanitizer CI run, bootstrap or release acceptance.

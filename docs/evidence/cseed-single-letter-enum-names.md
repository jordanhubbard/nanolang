# My declared single-letter enum names

I extend my native emission context to exact non-foreign enum declarations
(MAC `task_fc3046bc294543ca837fcb4ca21298c3`). My enum definition emitter uses
the common native name helper, whose free-variable fallback previously selected
`void*` for any unrecognized single-uppercase name.

I now include declared enums in the same invocation-scoped bitset as declared
unions. Exact declared names take precedence over that fallback. I preserve
existing enum-as-integer checking and do not change type compatibility. I skip
foreign enums when extending the snapshot, preserve runtime-name handling, and
retain the prior snapshot after every inner emitter return. Unbound generic
variables keep their prior representation.

At production checkpoint `b1ad8b12`:

- Seven focused C-seed methods pass with GCC in 7.642 seconds and strict Clang
  in 7.506 seconds. I execute enum values and locals, declared T function
  annotations, imported definitions, `Box<T>` payload formals beside enum T,
  and an unrelated unbound function variable U. Unknown variants and wrong
  scalar values preserve the previous artifact.
- The same-process nominal context harness passes enum-to-fresh-generic
  transitions, successive union/generic emissions, and restoration of an
  enclosing context after success and early refusal.
- Twenty-five adjacent union naming, imported-union and callback-signature
  methods pass in 25.370 seconds.

My fresh `make -j4 bootstrap test-parser test-typechecker test-transpiler`
passes. Six declared-enum methods pass through fresh Stage1 in 13.408 seconds
and Stage2 in 12.961 seconds. I select these explicitly without replaying the
known selfhost unbound-function-generic refusal.

An additional negative precedence method passes across C-seed, Stage1 and
Stage2 in 0.018 seconds: an actual declared enum T function parameter refuses
a boolean argument instead of treating T as an implicit generic variable.
It also passes in the Clang-configured C-seed run in 0.002 seconds; refusal
occurs before C compilation.

My existing selfhost implicit-function-generic task
`task_0198b105373a4cc2b3ebbbb0e9336af8` remains open; I do not count its known
refusal as shared frontend success. No ownership checker or module signature
registration behavior changes in this repair.

I retain logs under `/tmp/nanolang-single-letter-enum-*.log`. These are ordinary
post-repair checks, separate from historical product abort evidence and task dd74.

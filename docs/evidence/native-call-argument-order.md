# Native call argument order

I evaluate ordinary named-call and module-qualified arguments once, from left
to right, before invoking the target. I bind each argument to a scoped native
temporary. I copy the mapped callee name before lowering arguments because
nested calls can reuse the name mapper's buffer. I avoid temporary names that
collide with known source variables.

I preserve my existing foreign array ABI checks, opaque-pointer wrapping,
effect dispatch and computed-call lowering. This change covers ordinary named
and module-qualified calls; it does not claim that every specialized builtin
has completed the broader argument-order audit.

My regression requires trace `123` for ordinary and qualified calls, trace
`12312` for nested calls in both directions, and a captured value before later
arguments mutate that value. I execute it with my interpreter, NanoVM and
C-seed native compiler. `make test-native-call-argument-order` is part of
`test-units` and selects `bin/nanoc_c` explicitly.

I also run the transpiler tests, 15 native-effect/callback/native-array-ABI
boundary tests, and the existing computed-callee regression, which prints
`callee` before `argument`. These pass on Linux ARM64. My three shared effect execution tests and
`make bootstrap` also pass; Stage 1 and Stage 2 both compile and run the
bootstrap smoke program. This record does not
claim a new x86 or Darwin run.

MAC task: `task_c897ac40d20b43669817b766dbe1c5a3`.

Review added two native regressions. I capture a mutable function-valued callee
before a match-expression argument replaces it: the current call uses the old
function and the next call uses the replacement. I also execute nested calls
with 64 source variables named like my generated argument/callee temporaries.
Both pass. The release compiler fails the mutable-callee assertion. Separate
VM/interpreter fixes are tracked by `task_8555af61281944eb9ac4ca9043849a94`;
this native test does not claim those backends pass that new case yet.

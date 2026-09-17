# Production classifier shape constraints

I now link the shape graph into the translator and attach persistent variables
to classifier values, locals, arguments, results and field/element projections.
Calls, returns, array updates and branch joins unify the related variables.
Declared return representations are checked at `RET`, retaining that diagnostic
boundary. All graph storage is released on translation success or failure.

I build these constraints during the final classification pass, after existing
flat representation inference converges. Binding provisional kinds earlier
incorrectly constrained legacy array constructors; the first regression run
exposed that. Legacy integer-tagged constructors leave their element shape
unconstrained until contents or typed operations establish it. Instruction
variables are scoped per function, not solely by bytecode offset, so functions
sharing a code range do not accidentally share parameter shapes.

## Verification

`make -j1 test-nvm2c` passes 1,048 AOT checks and 952 graph checks. New tests
reject conflicting record fields inserted through aliases of the same array,
execute compatible alias mutation, and compile two functions sharing one code
range with different record-field representations. Existing legacy array,
branch, recursive-call, wide-record and scalar-array-field tests remain passing.
`git diff --check` passes.

I explicitly recompiled the test harness, `nvm2c.c` and `nvm2c_shape.c` with
`-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g`. That run passes
the same 1,048 AOT checks without a sanitizer report. Other linked objects and
generated executables were not instrumented; I do not claim whole-runtime
sanitizer coverage.

`make -j1 test-one-ir-compiler` still fails at function 20 with
`AGG_PACK field requires unsupported nested aggregate shape facts`.

## Remaining work

The emitter still consumes flat field-kind vectors. Nested record/record-array
fields remain disabled, and the new constraints are not yet the source of
representation inference. I must connect resolved nested shapes to emitted
storage and extraction before claiming compiler acceptance. MAC
`task_9c850e94e5a74b6f8941622e2872af23` remains open.

I also found that `test-nvm2c-sanitizers` changes compiler flags without forcing
existing objects to rebuild. I filed warm-cache instrumentation isolation as
`task_f3df199b025042e0b1d83484cd104ed3`; a passing warm invocation of that target
alone does not establish translator instrumentation.

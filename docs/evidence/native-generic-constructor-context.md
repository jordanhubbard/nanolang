# Native generic constructor context

I retain owned concrete TypeInfo for C-seed union constructors at globals,
inline arguments, returns, mutable assignment, and nested payload boundaries.
I register nested specialization dependencies and substitute their field types.
My generic instance registry frees its retained copies at environment teardown.

I execute `tests/unit/test_native_generic_contexts.nano` with the C seed. Its
shadows and main inspect distinct payload values for every boundary, including
an empty record array followed by append. Ten matching constructor controls
execute; eight wrong nominal contexts are refused by both C and bytecode
frontends while preserving a prior artifact.

My fresh three-stage native bootstrap passes again after integration with
PR452 ownership metadata. Sixteen constructor, nested-generic and instantiated
ownership methods pass on that integrated build. The existing nested generic,
nominal ordering and record-array suites remain adjacent checks. Forty repeated
parser/typechecker/environment/AST teardown checks pass ASan and UBSan with
leak detection disabled; this does not establish leak freedom.

The same stronger fixture fails under both self-hosted native stages: nested
`Envelope<Plain>` emits `nl_Box_T`, and a union global gets an invalid aggregate
`= 0` initializer. I preserve that failure evidence in
`/tmp/nanolang-constructor-adjacent.log` and track its repair as
`task_85a8db6e186440eaad80442bfc133dd8`. The new executable context gate is scoped
to the C seed until that separate repair lands; existing paired gates retain
both self-hosted stages.

Task: `task_633f2402ec5944cfba0911a56a9f4eb1`.
Logs: `/tmp/nanolang-constructor-integrated-bootstrap.log`,
`/tmp/nanolang-constructor-integrated-asan.log`, and
`/tmp/nanolang-constructor-final-integrated.log`.
I do not claim the NanoISA bootstrap or bytecode fixed point from these checks.

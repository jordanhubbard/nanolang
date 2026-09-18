# My borrowed-source lexical scalar acceptance

MAC `task_d3fdac5a43784608851272407743977b`.

I implement my [bounded lexical contract](../NANOISA_SOURCE_BORROW_LEXICAL_SCALARS.md)
in the C and selfhost borrowed producers. My original source checkpoint was
`d921b162` on main `948f98a6`; I then restacked onto `5193858f`, including the
separate native record-array scalar-tag repair. That integration changed no
source compiler code. My verifier and runtime authority are unchanged.

My fresh default bootstrap passed. The full source gate passed all 17 methods
in 167.637 seconds before that restack. It compared C-seed, C-built selfhost,
Stage1-built and Stage2-built raw modules and selected-shadow modules, plus
canonical Stage1/Stage2 publication. Exact dumps include code, ownership,
layouts and advisory names. VM and generated native execution passed; native
fixtures used ASan/UBSan with leak detection. Stripping advisory names retained
execution. My scalar analysis gate passed 441 ordinary and 751 allocation
checks; my assertion gate passed 959 lifecycle checks.

My new fixture exercises int/bool inferred and explicit locals in helper,
entry and selected-shadow branches/loops. Initializers refer to outer names;
inner mutable/immutable scope changes do not leak. Both branch outcomes,
zero/multiple iterations, nested scopes and borrowed calls execute. Distinct
slots retain lexical name intervals even for a zero-iteration body, and
sibling/nested name intervals close before leaving their bodies. Hidden
constructor/destructure temporaries remain unnamed. Ordinary out-of-scope and
self-initializer refusals preserve existing output, as do the retained
resource/destructuring/move/early-exit and false-shadow controls.

I retain branch-local resource moves and path-specific early returns as
separate work. I do not infer complete affine source control flow from this
scalar-local acceptance.

Logs:

- `/tmp/nanolang-borrow-lexical-bootstrap.log`
- `/tmp/nanolang-borrow-lexical-gates.log`
- `/tmp/nanolang-borrow-lexical-integrated.log` (final integrated run)

My final integrated gate on `5193858f` plus this slice passed all 17 source
methods in 169.205 seconds, including explicit zero-iteration and reused-name
interval assertions. The same invocation passed 441 ordinary and 751
allocation-instrumented affine bytecode checks, 959 owned assertion lifecycle
checks and 123 lexical local-name codec checks. I used
`make -j8 -o bootstrap test-source-borrow-emission test-affine-bytecode test-owned-assertions`
after the already completed fresh bootstrap; this avoids rebuilding unchanged
source compiler inputs and skips no test method.

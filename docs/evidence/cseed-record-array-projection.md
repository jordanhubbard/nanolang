# My direct record-array projections

I retain nominal element identity through `at` and `array_get`, including
record-array fields, returned arrays and `array_push` results. My native
accessors share the same typed lowering and parenthesize the dereferenced
record before a subsequent field selection. I borrow declaration metadata;
I do not allocate a new partial type object for each query.

My original direct-projection fixture failed typechecking with "Cannot determine
struct type for field access". The focused repaired fixture checks scalar and
nested record fields, aliases after pushes, returned arrays and nested push/access
expressions. Both accessors compile and execute through my C seed and NanoVM.
Six wrong-field cases reject with E004 and preserve prior output. My typechecker
and transpiler gates and fresh three-stage bootstrap also pass, including the
installed compiler check with my C seed removed.

This is a bounded nominal projection repair, not complete record-array support.
The richer existing compiler record-array fixture revealed three independent
boundaries: native nonempty record-array literal construction (task_a5fc558cbfa34f14b4d580923de4209c),
user/schema `NSType` name collision (task_1d90fd4257bb43c599a79ab11cfa7aec),
and a VM shadow failure after direct projection before a nested aggregate read
(task_911461243652466fb0af1fb706d4ac01). I keep those open on my roadmap.

Local evidence: `/tmp/nanolang-record-projection-baseline.log`,
`/tmp/nanolang-record-projection-gates2.log`, and
`/tmp/nanolang-record-projection-bootstrap.log`. An initial test assertion
expected different diagnostic capitalization; the corrected test requires the
actual stable E004 category and the same refusal/output-preservation behavior.

# I retain nested native specialization identity

My C frontend registration and native emitters now share the existing owned
TypeInfo specialization-name helper. It recursively names generic arguments,
so `Box<Result<int,string>>` consistently names its complete payload type.
My first shared-helper revision omitted the HashMap base case; bootstrap caught
that error and the corrected helper retains generic-name handling across kinds.

My self-hosted native emitter visits actual nested type nodes when collecting
union instances. A substring match previously treated the enclosing Box as a
Result instance and read a missing argument. I also reject an inconsistent
native-definition argument count before indexing payload arguments.

Fresh three-stage bootstrap passes. The paired executable fixture passes C,
Stage1 and Stage2, exercising both Result variants, an ordinary outer copy,
parameters and nested payload reads. Mandatory shadows remain enabled.

This closes task_b25eef03a0874cf8b0c7f0fa23c05289 and a prerequisite of
C native task_633f2402ec5944cfba0911a56a9f4eb1. Inline constructor arguments,
mutable reassignment, nested literal context and globals remain under633.

Evidence: /tmp/nanolang-nested-generic-baseline.log,
/tmp/nanolang-nested-generic-after.log,
/tmp/nanolang-nested-generic-bootstrap-r2.log and
/tmp/nanolang-nested-generic-paired.log.

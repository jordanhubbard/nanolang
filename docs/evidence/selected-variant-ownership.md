# Selected nongeneric payload ownership — incomplete checkpoint

I move an exhaustive unguarded nongeneric union scrutinee before splitting arm
flows. Each arm receives its own variant identity and resource obligation. A
complete qualified pattern transfers every field, and my existing lexical
exit, branch-join and loop-edge checks resolve the resulting owners. Empty and
ordinary sibling variants do not acquire another variant's obligations.

My C pass retains the selected name only for the lifetime of its arm flow.
My self-hosted pass uses declaration-indexed variant identities and resolves
field owners through the selected declaration. I still reject generic,
collection, wildcard and guarded ownership boundaries that lack this lowering.
I do not establish borrow support or NanoISA ownership metadata here.

On September 17, 2026, my fresh three-stage bootstrap passed. All nineteen
C-seed pattern/ownership methods passed in 8.628 seconds. They execute multiple
resource fields, nested resource records, ordinary and empty sibling arms,
exactly-once scrutinee calls and compatible joins. Ten ownership rejection
methods pass across all three compilers, retaining the prior artifact for
ignored payloads/fields, repeated consumption, post-move use, partial moves,
incompatible joins, wildcard hiding, ignored binders and loop reconsumption.
My 44 adjacent paired affine/generic/collection methods passed in 75.143 seconds.

My complete paired positive gate is **not passing**. Both self-hosted stages
reach native emission, but my emitter writes union payload typedefs before
by-value records: `nl_Handle` and `nl_Pair` are not yet defined. I preserve that
failure in `/tmp/nanolang-selected-ownership-paired.log`. The prerequisite is
`task_68a6b53f768245bfacfc79b6db78b621`; globally reversing declaration groups
would fail records containing unions. I keep selected-transfer task
`task_c17b55115379414980609a5d867ccad1` open until dependency-ordered emission
and the full paired gate pass.

Other logs: `/tmp/nanolang-selected-ownership-bootstrap.log`,
`/tmp/nanolang-selected-ownership-c-expanded.log`,
`/tmp/nanolang-selected-ownership-expanded-negative.log`, and
`/tmp/nanolang-selected-ownership-adjacent.log`.

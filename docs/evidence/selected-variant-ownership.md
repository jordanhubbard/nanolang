# Selected nongeneric payload ownership

I move an exhaustive unguarded nongeneric union scrutinee before splitting arm
flows. Each arm receives its own variant identity and resource obligation. A
complete qualified pattern transfers every field, and my existing lexical
exit, branch-join and loop-edge checks resolve the resulting owners. Empty and
ordinary sibling variants do not acquire another variant's obligations.

My C pass retains the selected name only for the lifetime of its arm flow.
My self-hosted pass uses declaration-indexed variant identities and resolves
field owners through the selected declaration. I still reject generic,
collection, wildcard and guarded ownership boundaries that lack this lowering.
My C frontend rejects owned guards during ownership checking; both self-hosted
stages reject their syntax during parsing. The negative regression checks each
diagnostic and preservation of the previous output.

On September 17, 2026, I integrated main `a16b2d8a` at source checkpoint
`71eb4b16`. A fresh three-stage bootstrap passed. My final 21 paired
pattern/ownership methods passed in 55.485 seconds, including the added guard
regression. Each method checks the C seed and both self-hosted stages.
My 58 adjacent affine, generic, resource-collection, union literal-context and
native nominal-layout methods passed in 160.123 seconds.
Positive controls execute multiple resource fields, nested resource records,
ordinary and empty sibling arms, exactly-once owned scrutinee calls, ordinary
call scrutinees in expression and statement matches, and compatible joins.
Rejection controls preserve previous artifacts for ignored payloads/fields,
repeated consumption, post-move use, partial moves, incompatible joins,
wildcard hiding, ignored binders, guards and loop reconsumption.

Two prerequisites are now implemented. Mixed complete-value declaration
ordering (`task_68a6b53f768245bfacfc79b6db78b621`, PR436) makes record payloads
available before their union definitions. Direct-call match type inference
(`task_7eb80289f352425aa2abd26af55d0ffc`) retains declared union return types
without adding another call evaluation. Earlier failures remain in
`/tmp/nanolang-selected-ownership-paired.log` and
`/tmp/nanolang-selected-ownership-final-paired.log`; I did not weaken those
positive controls to obtain acceptance.

My isolated ASan/UBSan parser, typechecker and teardown harness completed 280
checks: fourteen ownership fixtures repeated twenty times. That run used C
ownership source at `17b50354`, unchanged by the later match-emission repair.
Leak detection was disabled because the separate metadata-leak task remains
open (`task_00c47a5d65d04c48914864ec0de553d6`). This is not a leak-freedom claim.

I do not establish general generic owned unions, concrete generic resource
substitution, known-empty generic payload obligations, borrow support or
NanoISA ownership metadata. Global resource lifetime remains unverified
(`task_8afaef937f934a6e9919e41b91b7a41c`): the separate duplicate-use probe
reaches native emission before failing global-record initialization, and does
not demonstrate an executable double consumption. The complete affine
contract and full-roadmap release remain unfinished.

Logs: `/tmp/nanolang-selected-ownership-integrated-bootstrap.log`,
`/tmp/nanolang-selected-ownership-final21-paired.log`,
`/tmp/nanolang-selected-ownership-integrated-adjacent.log`,
`/tmp/nanolang-selected-ownership-asan.log`.

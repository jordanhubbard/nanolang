# I retain instantiated ownership metadata

My C ownership bindings retain complete immutable `TypeInfo` trees borrowed
from the function AST or retained environment. Those owners outlive the flow
and its branch clones. Typed aliases and declared-function call-result lookup preserve the trees;
classification substitutes union payload fields into temporary owned trees and
frees each temporary after inspection. I do not attach temporary trees to
bindings or free borrowed metadata when a branch ends. Complete first-class
function signature metadata remains a separate prerequisite (`task_e05a42e2e09b47cc9c53fa6923eeeaef`);
my C `FunctionSignature` cannot yet represent complete generic result trees.

My self-hosted ownership identities retain declaration indices and recursively
qualified concrete arguments. Ordinary generic match payloads keep the same
arguments on their selected-variant identities; field lookup substitutes the
selected declaration's fields before resolving their ownership identity.
Resource classification follows stored payload fields, so an unused resource
type argument does not itself create an owner.

Generic owned matching remains rejected. Resource collections remain rejected.
For C payload shapes whose embedded formals are not yet substituted, I preserve
the prior conservative resource-argument guard instead of treating a copied
metadata tree as fully resolved. My self-hosted classifier recognizes resource
leaves in tuple spelling and rejects them before native emission. This does not
implement tuple ownership transfer, general row/function substitution, global
resource lifetime or a NanoISA ownership contract.

The isolated parser prerequisite is merged as PR446; native nested type spelling
is merged as PR450. I integrated both at source checkpoint `1f4a6727`. The
combined checkpoint passed a fresh three-stage bootstrap. All twelve new paired
methods passed in 31.330 seconds, and all eighty adjacent methods passed in
225.485 seconds. The new methods compile and execute ordinary direct/nested
`Result` and `Box` values and unused-resource-argument markers, while checking
resource, collection, generic-match and tuple rejection across the C seed and
both self-hosted stages. Negative cases retain the previous output artifact.

An isolated ASan/UBSan harness passed 240 checks on this combined source: twelve
ordinary/negative parser and typechecker cases repeated twenty times with AST
and environment teardown. Leak detection was disabled for the existing metadata
leak task (`task_00c47a5d65d04c48914864ec0de553d6`); this does not establish leak freedom. The harness includes ordinary
nested types independently of native emission, and is not a substitute for the
executable positive gate.

Tasks: `task_d329989e8acf43149c38d6a998bd730f` (instantiated identity),
`task_bcd773ad3c084ce099a3da5aef682fef` (unsupported tuple diagnostics).
Ordinary generic selected-pattern validation remains
`task_bbda7f126bda403aa74034a762930f24` before generic owned transfer.
Global ownership remains `task_8afaef937f934a6e9919e41b91b7a41c`; broader generic
constructor contexts remain `task_633f2402ec5944cfba0911a56a9f4eb1`.

Logs: `/tmp/nanolang-instantiated-integrated-bootstrap.log`,
`/tmp/nanolang-instantiated-integrated-new.log`,
`/tmp/nanolang-instantiated-integrated-adjacent.log`,
`/tmp/nanolang-instantiated-ownership-integrated-asan.log`.

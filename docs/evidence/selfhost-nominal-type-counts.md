# My self-hosted nominal type bounds

My aggregate operands already use parser record indices. I now declare the
parser's complete record, enum and union counts in `.types` before `.entry`.
Unused declarations remain in these index spaces; executable reachability
only selects functions and does not renumber nominal declarations.

The regression has an unused record before the constructed record, two enums
and one unused union. Both C-seed and self-hosted output retain `.types 2 2 1`
and the constructed record uses ID 1. Their selected function instructions
match, and both artifacts verify and execute in my VM and native translator.
Reducing the record count to one fails assembly verification and preserves
the previously published artifact. My native proof requirements are unchanged.

`make test-nanoisa-src-nano` passes 86 comparison checks and all 63 Python
methods. The log is `/tmp/nanolang-type-counts-gate.log`. These are bounded
metadata and execution checks; they do not establish full compiler bootstrap
equality. The complete compiler artifact is the separate native acceptance
for `task_250092bed54749ad988f06af5b88c228`.

I track this emitter companion as `task_14dca63e4c1146e59ae0a1649ba29060`.

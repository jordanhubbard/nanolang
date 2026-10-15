# Nested-function source names

I reproduced missing lambda parameter/local names in the C producer while the
fresh native selfhosted compiler passed the same fixture. I enable the existing
local-name recording inside nested functions, begin parameter names at entry,
and close child lexical intervals before restoring parent state.

I pass all nine paired metadata methods with strict native C compilation and
AddressSanitizer/UndefinedBehaviorSanitizer, with leak detection enabled. The
nested fixture includes a grandparent capture, a captured array, a record local,
two closure parameter lists, and a shadowed child local. It checks exact names,
function ownership, entry/lexical intervals, original-wire canonical equality,
second-cycle equality, VM/native execution and metadata-stripped equivalence.
I also pass all 90 bytecode checks and the broader lexical capture/refusal
fixture in VM and sanitized native execution.

I use the fresh native selfhost product from
`/private/tmp/nl51-localnames/compiler`; this change only alters the C emitter,
so it does not require rebuilding the unchanged selfhosted source/schema.
Upvalue names remain open: an upvalue index is not a local slot. Effects and
other producer families remain part of the full #976 scope.

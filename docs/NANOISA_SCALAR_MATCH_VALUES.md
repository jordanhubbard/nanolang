# My scalar expression-match contract

I extend my exact nongeneric scalar union path with exhaustive expression
matches. Every variant appears once by its declared name. Every arm is a
non-block expression with the same exact `int`, `bool`, `float` or `string`
result type. I retain nominal union identity and declared payload positions.
I do not admit wildcard, guarded/or patterns, generic unions, resource/nested
payloads, aggregate results or value-yielding block arms in this slice.

I evaluate the scrutinee once. The selected arm consumes its retained union
operand and leaves one result; all reaching arms join with that same scalar
type. The existing impossible-tag assertion remains. Statement-block match
lowering keeps its return/fallthrough behavior.

During result inference I use isolated copies of my lexical binding tables,
add only the current payload binding, and restore the original tables before
returning. Inference does not publish local slots or names. During emission
I allocate a scoped payload slot and remove its lookup name before leaving
its arm, while retaining allocated slot types for final function metadata.
Nested matches must restore each outer binding exactly.

Acceptance includes both raw emitter hosts and C-seed NanoVirt plus both
canonical selfhost stages, VM and sanitized native execution, selected shadows,
all four result types, calls/returns/initializers, same-name lexical restoration,
nested matches and once-only scrutinee/selected-arm effects. Refusal controls
must preserve prior output and establish the intended lowering/type boundary.
I retain all selected shadows or refuse publication. No runtime or ownership
authority changes.

MAC: `task_b43095db71f74c0b8418c27530f1686a`; lexical prerequisites are PR652.

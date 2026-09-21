# My self-hosted array facts

I track this prerequisite under task_f284f62c42a8405baa281e8d6eea6834, within
the original full record-list acceptance. I preserve f54 successful bootstrap
and scoped instrumentation, plus the later both-host first refusal failure.
Stage1 publishes output for `array<array<LocalItem>> = [[], foreign_items]`.
I do not execute that output or replay known incorrect acceptance for diagnosis.

## What my current source establishes

My literal checker takes the first element's type; `types_equal` deliberately
permits unknown array elements for empty-literal inference. This drops later
known imported identity when the first element is empty. My contextual helper
checks byte-array literals, while ordinary typed literals still rely on the
permissive equality path. My field-only checker fixes that boundary but does
not establish every local/call/return destination.

My direct array_push fast path returns a known receiver without checking the
value against it; array_set's builtin result is VOID without that agreement.
array_slice has no complete result inference. The IF expression path checks
only its then tail, unlike statement IF checking. Iteration reconstructs a few
scalar/record kinds and otherwise falls back to INT. These are static findings,
not additional executed failures. I audit map/filter callback and match/block
paths with this same complete contract before qualification.

## Literal facts and contextual destinations

I keep global types_equal unchanged. I unify literal member facts recursively,
retaining canonical nominal names supplied by nb_rewrite. A missing type is an
empty hole only when the actual AST is a genuine empty literal (or recursively
contains only those holes). A later known member fills compatible holes; two
known incompatible nominal leaves cannot unify. Unknown facts from a call,
identifier, unsupported expression or incompatible nonempty literal are not
empty evidence. I retain the existing legitimate empty-literal policy without
turning every unknown array value into a destination type.

I extend destination checking across let/set/return/direct and indirect calls,
record fields and union payload contexts. A typed array literal visits every
member against its declared element type, recursively including nested empties.
Only accepted literal subtrees receive element annotations. A failure rejects
the containing value; I do not claim rollback of previously valid child
annotations. Existing array variables keep their actual complete type.

I reuse root's scalar/byte contextual conversion policy. I do not report a
permitted explicit byte-literal conversion as an irreversible raw inference
error before its destination is available. Inferred literals with incompatible
nonempty members must instead produce a checker diagnostic. Neither an inferred
local nor apply_return_type_hint may erase a known nominal mismatch.

## Actual producers, consumers and branches

I select intrinsics only from the exact bound builtin name with no visible
ordinary/foreign/function-value declaration. I preserve qualification and local
precedence rather than classifying by suffix. Existing argument visits stay
static checks; this change adds no runtime evaluations or reorders them.

I retain array_new's reviewed checked fill inference. Array push/set check
arity, actual receiver element facts and the value destination even when their
result is unused. Slice returns the complete receiver type after index checks.
Get/at and iteration derive the complete nested element annotation rather than
defaulting another array, union or callable to INT. A real empty constructor
can use an explicit checked context; an existing unknown array cannot.

Map uses its actual receiver and unary callback: parameter agreement is exact
where storage/nominal identity requires it, and output carries the complete
callback result. Filter additionally requires BOOL and retains the receiver's
complete element type. I preserve declared same-named functions and the already
checked reduce contract. Unsupported facts diagnose; they do not invent owners.

I use the existing check_block_value/check_statement_list machinery for lexical
locals and statements in expression blocks. IF/COND and match values inspect
all reachable alternatives using their own bindings and existing enclosing
return/control-flow rules. Agreement retains complete array/nominal facts,
including contextual empty literals, without choosing only the first branch.
I do not replace ownership analysis, source evaluation or backend authority.

## Acceptance and attribution

I preserve all fifteen source methods and add helper shadows for first-empty
then-known inference, wrong later nominal leaves, heterogeneous nonempty arrays,
typed and inferred destinations, every changed intrinsic, ignored mutation
returns, callback/iteration facts and declared-call precedence. Positive cases
include matching nested empty branches and block-local values; negative cases
require checker diagnostics and output sentinels before any native execution.

Root reviews the complete source/fixture checkpoint before corrected gates.
Fresh full bootstrap retains the original ten-second shadow deadline and
outer bounds. The full15 matrix, relevant neighbors and unchanged Make remain
required. Exact C-seed/selfhost/emitter integration and enum list parity remain
separate full5.1 obligations; no scoped pass closes them.

## My source checkpoint

I carry `CheckedBlock.ok`, complete value type, AST-proven empty holes and a
separate terminating-control flag. My statement worker takes an explicit
`value_required` flag: an ordinary statement IF need not produce a common
value, while IF/COND/MATCH used as a value must check every alternative. A
checked terminal branch contributes no value to a join. At a declared
unreachable destination it satisfies that destination without creating an empty
array fact or publishing a literal annotation. I retain warning diagnostics
without counting them as type errors.

My recursive literal facts merge later complete members into actual empty
holes, including arrays inside tuple members. Existing identifiers/calls need
complete facts and cannot provide holes. I publish inferred annotations only
when complete; an unresolved genuinely empty initializer retains its AST-based
inference rather than turning into a declared unknown array on a later visit.
My common destination helper checks every array/tuple member, invokes concrete
union constructor checking where required, and preserves existing scalar/byte
conversion policy. Failed existing arrays retain their actual type plus an
error; permissive global equality does not suppress that error.

I retain exact unshadowed dispatch for array operations and preserve complete
nested results through get, slice, iteration and callback signatures. The
actual `array_new` path still requires complete checked fill facts, including
nested fills; I do not add an unknown-fill constructor admission. Explicit
empty literal syntax receives a checked destination. I also retain the full
List annotation when it is itself an array element, rather than serializing
its empty outer name. Qualified declared callables check all parameters and
arity, including function-valued declarations, without selecting array
intrinsics from a qualified suffix.

I add helper shadows and retain all prior source methods and assertions.
Static delimiter balance, added-helper shadow inventory and diff whitespace
checks are preparation checks only. I have not built or executed this
checkpoint. The complete paired source fixtures and original full bootstrap,
fifteen-method corpus, instrumentation and remaining Make acceptance are still
required after independent source review and capacity preflight.

## My paired fixture checkpoint

I add one source method without changing the prior fifteen. Its positive
program runs nested first-empty inference, direct/indirect arguments, record
and union payloads, lexical conditional blocks, integer match alternatives,
terminal return branches, complete nested iteration, slice and push/set through
the existing C-seed, Stage1, Stage2, evaluator and NanoISA/VM routes. Each helper
has a meaningful shadow. I add thirteen refusal programs for later nominal or
scalar mismatches, typed/inferred destinations, calls, payloads, branches,
iteration, ignored mutations, BOOL filter results and return destinations.
They require actual checker diagnostics and preserve output sentinels.

The runner remains unchanged, including retained products, command terminals,
process-group cleanup and deadlines. Static Python AST inspection finds sixteen
methods. I have not imported the runner or executed any fixture at this pin.

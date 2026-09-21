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

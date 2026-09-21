# My concrete union payload boundary

I retain task_7e994b0b89aa40629002347160d82ef6 and the first 744 Linux
focused assertion. The checker accepts `Box<Item> = Box.Value{item:true}`.
The constructor branch explicitly skips compatibility for every STRUCT or
GENERIC field when the declaration has any formal parameters. The earlier
context visitor handles arrays, lists and callables but leaves scalar/record
leaves to that incomplete branch. No generated program was executed.

I will replace that bypass with complete selected-field checking. I select the
actual union declaration and unique variant, validate the field count, and
match each supplied field exactly once. Count equality alone cannot excuse a
duplicate field or an omitted sibling. I retain the original missing/unknown
variant and field diagnostics and add the duplicate-field refusal.

Before recursive child checking I prepare owned field views. Each view starts
from the declaration's complete field annotation and declaring owner, with the
constructor's actual argument annotation/owner/context as its substitution link.
The existing retained constructor view is authority when present. An explicit
constructor annotation supplies its actual owner when no retained view exists;
a nongeneric constructor uses its declared fields directly. I copy the complete
view and substitution chain before any recursive checker may grow declaration
or symbol arrays. No retained pointer into those growable arrays survives a
recursive call. I discard every prepared field view on success or failure.

I use the existing contextual argument machinery for base compatibility and
literal children, and retain complete original views for nominal comparisons.
Materialized names are emitter metadata, never substituted proof. Fixed record
leaves retain the definition owner while formal leaves retain the actual owner.
Nested list, array, tuple, union and callable fields keep that distinction.
Existing checked scalar conversions remain the destination policy; I do not
introduce enum/U8 compatibility or a new enum representation in this repair.

The field checker must not recursively re-enter the same constructor node while
preparing its own context. It may recurse only into child payload expressions.
Destination visitors still establish let/set/return/call/record-field context
before the constructor is checked. I audit all such entry routes. Generic
constructors without a complete explicit or destination context require a
separate inference decision: I will inspect the existing accepted inference and
emission routes before changing their behavior. I do not use an unresolved
formal as a wildcard or silently label that audit complete.

I retain all original controls and the newly exposed bool-to-record refusal.
Additive cases cover explicit and destination-derived arguments, fixed/formal
same-spelling records from different owners, nested constructors/containers,
callable fields, missing/unknown/duplicate payloads, and each destination route.
Checked allocation sweeps must cover prepared field views and preserve original
AST/previous cache entries on failure with fresh recovery. The existing binder
allocation domain remains separately attributed. Source and fixtures receive
review before corrected execution; full eighteen source/eight native methods,
both hosts and unchanged bootstrap deadlines remain required.


## My source checkpoint

Both AST_UNION_CONSTRUCT and the legacy dotted struct-literal constructor route
now prepare owned field views before checking children. Unknown/duplicate fields
retain E004; missing variant/arity checks remain at their original entry sites.
Preparation leaves its output pointer unchanged on failure and frees every
partially prepared view. Existing outer expression/program transactions retain
previous checker proofs and native array/tuple rows if child checking fails.

The actual allocation domain measures preparation separately from recursive
checker behavior: a definition-owned Item and caller-owned CallerItem, formal
payload, nested callback and array fields exercise deep context copying. Every
measured transient/persistent prefix preserves source annotation bytes, caller
output and earlier cache counts; independent recovery follows. Six parsed
destination routes retain positive, wrong-tag and distinct-record controls,
with additional duplicate/missing/unknown/fixed-record payload refusals. The
original constructor, nested callback and eighteen/eight corpus controls remain.

A generic constructor lacking both retained destination proof and complete
explicit type arguments now refuses preparation. I found no payload-driven
inference in this native constructor emitter: it uses the node annotation or a
function return annotation. This is not full inference acceptance; I keep that
remaining audit and the whole unchanged source/native matrix open. No execution
has occurred for this checkpoint.


## My explicit scalar policy amendment

Source review found that the shared contextual helper is not scalar-policy
neutral. `types_match` admits INT/U8, INT/ENUM, ENUM/U8 and UNKNOWN. Its later
indirect argument check compares complete metadata only when the actual value
has it. That can make acceptance depend on expression metadata rather than
source type. The c2dc checkpoint has not executed.

I choose the existing AST_UNION_CONSTRUCT strict rule for this payload boundary,
using the resolved declaration tag rather than an unresolved parser placeholder:
INT accepts INT, U8 accepts U8, ENUM accepts ENUM, and the remaining scalar tags
accept their own tag only. UNKNOWN always refuses. I align the legacy dotted
struct-literal route with this rule; its former `types_match` broad acceptance
is not retained. Both source spellings already normalize to the same constructor
representation in the full binder path. This is an explicit alignment, not a
claim that c2dc preserved both formerly inconsistent paths.

No INT/ENUM, INT/U8 or ENUM/U8 payload conversion is introduced here. Numeric
conversion at other existing destinations remains unchanged. The broader
computed-U8 and enum numeric-destination contracts still require integration
and actual lowering/evaluator parity; this checkpoint cannot close them by
rejecting their cases or calling them accepted. Composite payloads retain the
full contextual owner-aware checks. Scalar checks execute the checker once,
reject UNKNOWN and compare resolved tags without metadata-dependent fallback.
I add the complete INT/U8/ENUM/UNKNOWN matrix with and without stored primitive
annotations, plus actual parsed nongeneric scalar-union cases in both direct
and explicit-generic source forms before corrected qualification.

# My ordinary heap-bearing record authority prerequisite

I execute task_15f955fae5cf402d92bf88794122e9a2 after descriptor733 and private
storage737. Those checkpoints describe shapes and retain runtime children; they
do not certify an ordinary source definition. This contract is non-admitting.
I leave aggregate488, managed51da and nominal execution open.

## My audited gap

The ordinary Cseed code generator registers per-kind record indices and publishes
counts, but does not emit retained field layouts or ownership declarations.
The ordinary selfhost emitter likewise publishes `.types`. Only the specialized
borrow producers currently retain both complete layouts and ownership contracts.
Count-only placeholder records cannot supply ordinary authority.

Existing version1/version2 ownership flags distinguish COMPLETE1 and RESOURCE2.
COMPLETE without RESOURCE explicitly declares a complete ordinary definition;
absence, flags0 and incomplete definitions remain UNKNOWN. The current validator
accepts COMPLETE only for finite numeric/bool record trees. It rejects strings
and arrays, even for explicitly ordinary declarations. I must not weaken the
existing resource/reference eligibility while adding ordinary heap-bearing facts.

## My first checked transport checkpoint

I reuse the existing wire encoding and flags. A complete ordinary record may
contain existing numeric/bool leaves, STRING with no nested index, and earlier
complete ordinary records. RESOURCE records retain exactly their current finite
numeric/bool tree constraint, including transitive ordinary children: an ordinary
string-bearing child cannot make a resource parent eligible. I compute that
transitive scalar-only property during validation; I do not infer it from a
missing RESOURCE bit on an incomplete child. Existing borrow modes, path tables,
function/local counts, signature tags and reserved checks remain unchanged.
Old readers may reject the newly described ordinary heap shape; no bytes or
opcode numbers are reinterpreted. Existing accepted scalar/resource contracts
retain their validation and execution decisions.

A checked read-only authority query distinguishes UNKNOWN, ORDINARY and RESOURCE
from validated declarations. It leaves output untouched on failure and reports
allocation errors through the existing metadata result convention. A descriptive
record plan can carry ORDINARY only when every described record has validated
complete ordinary authority. Mixed resource/unknown tables stay unresolved for
that shared plan; no new executable profile consumes it in this checkpoint.

I test canonical byte roundtrips, explicit empty records, same-shaped distinct
identities, nested ordinary strings, incomplete declarations, unchanged scalar
resource/borrow controls and refusal of direct or indirect heap-bearing resource
claims. Validation does not prove instruction field effects; field provenance
and runtime admission remain separate prerequisites.

## My paired ordinary producer checkpoint

I initially publish complete facts only for a closed module whose record
schema is fully supported: local non-generic, non-resource definitions with
numeric/bool/string fields or earlier supported records. I preserve each
producer's existing per-kind record order and map it to exact retained global
indices. I retain names as advisory only. Prior-record order matches the current
closed retained-layout encoding without remapping executable nominal identities.

I derive classification from checked declarations/full field annotations and
existing resource classification. A missing definition, unsupported annotation,
resource or ambiguous identity never becomes ordinary by name or by absence of
metadata. If this first producer cannot describe the complete module honestly,
it omits the optional authority/layout publication and preserves existing source
acceptance and its UNKNOWN status. I do not publish fake empty placeholders.

For a qualified module I serialize existing per-function metadata with exact
function/local counts and parameter/result tags. Unproved local/result nominal
identity remains NO_INDEX; declaration authority does not supply field-flow or
call-result provenance. I preserve all existing local-name/debug/shadow metadata.
Specialized borrowed source emission is unchanged. Entry and shadow producers
both receive truthful declaration facts, without skipping selected shadows.

Acceptance requires Cseed and fresh selfhost-stage source publication controls,
normal VM behavior, matching record/field authority projections, canonical
transport stability and unchanged LLVM/Wasm nominal output-preserving refusals.
Unsupported source controls must remain accepted where already supported while
reporting absent authority, rather than gaining an ad hoc source rejection.
No changes to test_llvm_managed_* are needed while the portable harness lane owns
those files.

## My remaining required authority coverage

The first checkpoint does not close full15f. Ordinary arrays of resources cannot
be certified from a bare ARRAY tag: retained element ownership/provenance must
connect the complete declared element type and producer classification before
array-bearing record authority. I separately retain generic substitutions,
forward record ordering, imported/qualified definitions, union/tuple/map/callable
fields and recursive nominal declarations as required authority coverage under
15f/full aggregate scope. No new wire representation is chosen here for those
cases. Each extension needs a contract that preserves exact nominal identity
and cannot hide an affine child behind an incomplete or unknown declaration.

After the checked transport and paired producer boundary is qualified, I design
field-origin/effect analysis across aliases, calls/globals and joins. Neither
these declaration flags nor private737 storage substitutes for that analysis or
its generated ownership/safe-point acceptance.

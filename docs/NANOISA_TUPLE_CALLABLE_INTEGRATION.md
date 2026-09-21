# My tuple and callable annotation integration

I compare my 7e27bbf62 callable-provenance lane with the SDK lane's
3b4989e8a source. SDK qualification is incomplete. This is a source integration
contract, not a passing backend or lifetime claim.

## One representation

I retain the SDK d1d2d5556 representation: TYPE_TUPLE stores complete owned
children in type_params, with type_param_count equal to tuple_element_count.
The existing tuple_types and tuple_type_names are independently owned duplicate
views and must agree with those children. Legacy tuples without full children
remain valid only where their flat facts describe the complete element type.
I use type_info_tuple_element, type_info_tuple_valid and
 type_info_tuple_refresh; I do not introduce a second tuple schema.

I also retain 4170979a3's TOKEN_FN fallback. When the caller requests TypeInfo
but no separate signature, the parser owns the signature through that tree.
This already repairs the SDK typed-array route; my older branch lacks it.
Explicit constructor auxiliary-output retention from a43 and nested callable
return merging from 87cc remain necessary. Their ownership rules are compatible:
borrowed merged views are copied before publication, and duplicate owners are
never made aliases.

## Required adapters

My contextual equality must validate duplicate tuple facts, select complete
children through the SDK accessor, and compare each under its actual owner and
substitution context. It must not compare full children and then reconstruct
lossy flat callable leaves, nor require identical legacy/full encoding when
both describe the same complete type. Unknown or unresolved nested facts still
refuse. No same-pointer or spelling-only shortcut is authority.

My contextual materializer must copy/materialize complete tuple children before
refreshing the duplicated flat fields. It must not reconstruct nested callable,
array or tuple annotations from flat tags. On allocation failure the unpublished
copy is discarded and source/output owners stay unchanged. No-context
materialization retains its existing checked-copy behavior.

Both nominal_array_matches_context and contextual_argument_matches must use
complete tuple children for literal destinations. SDK literal snapshots remain
owned annotation storage; materialized names do not replace my original
NominalView owner/context proof. Existing indirect coarse compatibility checks
must remain conjoined with exact nominal checks.

AST_TUPLE_INDEX requires a checked in-range projection from the enclosing owned
NominalView, retaining its original owner and substitution chain. SDK's plain
try_get_expr_type_info child pointer alone is insufficient for a tuple reached
through a mixed-owner selected union payload. Nested index, field, callable
alias, call, return, assignment and branch consumers must use the resulting
owned projection.

An inferred tuple literal can combine children with different original owners
or substitution chains. A single flattened tuple annotation and one owner
cannot establish those facts. Before accepting that case I must retain each
child's complete proof or establish an equivalent reviewed representation.
A destination-checked tuple may retain the destination proof only after every
actual child has passed its exact contextual comparison. I keep inferred and
contextual cases distinct; I do not silently drop inferred tuples from scope.

## Ownership and acceptance

I reconcile the SDK ordinary/fatal copy paths with my checked signature graph
copier and Environment registry. Full tuple children are recursively owned once;
flat duplicate strings and callable duplicate signatures have independent
owners. Parser cleanup, metadata encoding, substitution, native declaration
ordering and both transpiler registries remain in the integration audit.

I require parsed tuples containing arrays/callbacks/nested tuples, selected
payload tuple index to callback alias/call/pass/return/set, mixed definition and
caller nominal leaves, inferred literals and destination literals, swapped-owner
refusals, conflicting duplicate metadata, and every checked allocation prefix
with recovery. Original 18 source methods and 8 native methods remain required.
Source adapters need review before execution, and unchanged capacity/deadline
guards apply. A clean textual SDK merge does not complete these consumers.


## Owned child proof checkpoint

I extend only the private NominalView proof, not SDK TypeInfo. A composed tuple
owns one child view per element; an array composed from such a value owns one
element view. Each child owns its original annotation, owner and substitution
chain and may itself contain child views. Empty tuples use an explicitly
present zero-child proof. The outer TypeInfo contains independent duplicate
annotation storage for compatibility; its flattened owner never authorizes the
children. I validate count/kind/storage agreement before consuming a composed
view. Ordinary declaration-derived views keep their existing single-context
representation.

Clone, comparison, materialization and projection recurse through owned child
views when present and use declaration context projection otherwise. Array
wrapping moves the complete child proof only after allocating independent outer
storage. Child cloning and aggregate publication commit only after every
allocation succeeds. Teardown recursively destroys each proof once. Depth and
count/size overflow bounds remain explicit. Branch equality, array
homogeneity, mutation, map/filter/reduce, literal destination checking, inferred
binding publication and assignment must call proof-aware helpers; none may
fall back to the composed outer owner. Checked metadata copies are emission
artifacts, not replacements for child proofs.

I reuse the SDK parser/tuple helper foundation as an attributed source subset
on this lane before the eventual complete branch integration. I retain the
same public tuple APIs and representation, including complete-child validation
and duplicated-view refresh. The final merge must preserve both implementations'
additional checker/opaque authority and code-generation consumers.

## Source checkpoint and remaining native closure

I reuse the SDK d1d2/417 parser and tuple accessor representation at this source
checkpoint. My checked copier validates duplicate tuple facts; substitution and
materialization refresh the independent flat view after changing full children.
Private composed proofs retain each child's original owner/context through
clone, literal construction, nested tuple indexing, homogeneous arrays,
mutation and callback consumers. I add actual selected-payload tuple projection
and mixed-owner inferred tuple controls, plus separate allocation-prefix and
transient sweeps for construction/publication and clone/projection/materialization.
Those controls have not executed at this checkpoint.

My current native tuple registry still compares flat element tags and emits
fields through its older type spelling path. The SDK complete derived-type
registry, temporary tuple ownership, forward declarations and nested callable
field emission are required integration dependencies, not proved by these
checker changes. I must reconcile that existing SDK implementation rather than
invent a second tuple layout or naming scheme. Native tuple/callback producer
coverage and all original 18 source plus 8 native methods remain open; I make
no build, runtime or native-parity claim from this source review checkpoint.

## Native integration dependency order

I preserve e456006ad as the separately reviewed checker checkpoint. The SDK
native graph depends on its earlier canonical opaque owner keys, per-emission
name projections, complete annotation snapshots, tuple/array expression
bindings and derived declaration ordering. I integrate the tuple expression
binding and registry ownership first with the same SDK API and representation,
using this lane's checked graph copier. The registry borrows Environment-owned
literal annotations and explicitly owns only temporary copies; it compares
complete children rather than flat tags. The checker remains the authority:
these expression bindings are emission artifacts after contextual validation,
not permission to resolve owners from flattened names.

I then reconcile the SDK semantic-key/name and derived declaration closure,
including dependencies through fixed record fields and generic union payloads.
Its current activation predicate covers opaque carriers and array-derived
values; ordinary nested tuples/callbacks also need dependency ordering for my
full scope. No partial registry checkpoint authorizes native qualification.
I preserve existing list specialization selection and do not import unrelated
peer changes merely because the branch has newer canonical ancestry.

I retain the exact 568dd718c opaque identity/key source dependency plus d1d2's
complete tuple key traversal in a separate unqualified checkpoint. Conflict
resolution preserves my record declaration-owner resolver, checked callable
and tuple comparisons, stable assignment destination snapshot and compact List
C spelling. SDK opaque value checks are additive to those boundaries. The
paired Nano nominal owner tables and native name projections remain SDK code;
I do not introduce a second key format. Existing Make owners name the added
include prerequisites. The tuple expression API uses my existing checked
copier and remains defined once. The derived declaration graph, current array
carriers and subsequent SDK corrections still follow before affected gates.

## Emission snapshot publication during SDK graph integration

I retain SDK 417/8559/0b870/8a090/190b/b19 declaration and carrier machinery,
but its plain annotation inference cannot replace my contextual checker views.
My native metadata adapter materializes the complete retained view and keeps
that independent snapshot in the Environment. A callable API returns a checked
owning signature copy. Neither API grants source compatibility from emitted
names. Actual registry-created builtin identity remains authoritative.

I publish literal snapshots only after the selected destination comparison
succeeds, or from the final inferred view. I do not eagerly register an INT
literal tuple before a contextual U8 destination is checked. Complete child
annotations and substitution owners survive until materialization; independent
emission storage is not a new provenance proof. A later sibling failure still
rejects its outer value; this is not rollback of previously accepted child
annotations. Allocation failure records a checker error, retains original
proofs and releases unpublished copies.

The graph must cover ordinary nested tuples and callbacks as well as opaque
leaves. The SDK's initial opaque-only activation is an integration prerequisite,
not my full tuple acceptance boundary. I preserve list forward declarations,
initializer discovery, checked operation evaluation order and optional helper
annotations while reconciling both native producers.

My final literal-lowering audit finds two remaining old paths: the C tuple
emitter selects a typedef from flat tags (or invents INT fields), and its
compound initializer does not impose left-to-right child evaluation. I replace
that selection with the checked complete tuple snapshot and stage each child
once in source order before assembling the typed value. Empty unit tuples keep
the existing dummy representation. These are required corrected-only native
controls; the old paths are not replayed to produce invalid output.

### Source checkpoint boundary

I reconcile SDK 4170979a3, 8559c2273, 0b8709511, 8a090a035,
190b00ad7, b19cfa561, 8ce60bc45, 1a9052254 and 5f3e0f4e1 over my
40ece44a9 canonical-key checkpoint. I preserve my exact builtin registry
identity instead of adopting a bodyless-function heuristic. My checker adapter
uses retained views instead of importing the SDK's separate plain callback
comparison and first-element tuple inference. The Nano merger keeps this
lane's MergeResult schema; I add the opaque-declaration preservation itself
and its original owner/source-line assertions.

I have not built or executed this integrated source. The independent e456 and
73b focused results cover only their stated parser/checker/lifetime revisions.
The complete native fixture additions, both-host bootstrap and unchanged
18 plus 8 method matrices remain required. SDK packaging and full SDK
acceptance are separate from these source dependencies.

## Failed-check emission cache transaction

Root review of unexecuted 1e2e862dd finds that my recursive emission publisher
appends a root row before later children can fail. A failed checker entry can
then free its AST while my Environment still retains that address as a cache
key. MAC task_d3d5a17bad0f48ae92c3fa4dc4d306d5 owns this correction.

I checkpoint the tuple and array row counts at each complete recursive
publication, emitted-snapshot API, direct expression check, program check,
module check, root shadow check and imported shadow scope. A failed operation
frees and zeros only newly appended owned snapshots and restores those counts.
Rows are append-only: existing keys either compare equal without mutation or
refuse. Prior snapshot pointers stay stable even if the backing vector grew.
The vector itself is private and has no external borrowed row pointers. I need
no new allocation to roll back. A newly empty vector is released and nulled.

Nested calls use stack-local checkpoints; there is no ambient cache transaction
or reuse across Environments. A failed outer entry also removes successful child
publications made since its checkpoint. I include failure after binding but
before final Environment-owned output registration. Direct expression failure
includes UNKNOWN, a newly reported checker error, or a new metadata-failure
flag. Main's later shadow failure destroys the Environment with the program;
the failed shadow entry still removes its newly appended rows. Previously
accepted ASTs remain borrowed under their existing Environment/cache lifetime
contract; this transaction does not extend that lifetime.

I audit every early return in these entry wrappers and test corrected-only
initial/growth allocation prefixes, recursive late-child failure, malformed
later siblings, preservation of prior owned rows, and fresh success. I do not
execute a stale-key reproduction on 1e2e862dd.

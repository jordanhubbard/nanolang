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

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

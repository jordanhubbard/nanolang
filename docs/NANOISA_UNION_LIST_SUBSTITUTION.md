# My compact list payload substitution

I retain both `83370647b` first native discovery failures under
`task_40aa248a7326409cba8fcadc4bddecdd`. Fresh bootstrap and all five focused
configurations passed. The complete first native method passed, then C-seed
refused the unchanged `Held<T>.Items { values: List<T> }` vector instantiated as
`Held<Item>`. Neither refused output was executed; the eighteen-method suite
remains unreached.

My compact annotation encodes `List<T>` as `TYPE_LIST_GENERIC` with the element
spelling `T` in `generic_name`. It is not a whole-type formal reference. Two
existing consumers confuse these meanings: `nominal_substitute_annotation`
rejects that container when its element spelling matches a formal; the legacy
`payload_substitute` would replace the whole list with the concrete record.
The first refusal is observed. The second container-loss defect follows from
source inspection, not a separate runtime experiment.

I will keep whole-type substitution limited to actual formal leaf encodings.
Compact list containers must reach existing list normalization, which resolves
their element under the explicit substitution context. Concrete materialization
must preserve the list container and substitute only its element, consistently
with the explicit `List` plus one type-argument representation. I will audit
constructor validation, owned annotation copies, match payload field facts,
list operation result typing and both native provider discoveries. I will not
repair this by treating an unresolved list as an arbitrary record or changing
the source fixture.

A definition-site fixed record leaf keeps its declaration owner; a substituted
formal keeps the concrete argument's owner, including nested context. I will
use the existing owner-aware view/identity machinery and preserve exact
non-null identity, same-spelled-owner refusal, scalar list catalog boundaries,
and the explicit intermediate enum-list refusal. No current-name fallback is
permitted. If materialized type spelling cannot retain that provenance alone,
I must retain the original template/context view at the consuming boundary.

My additive controls will distinguish compact and explicit encodings, preserve
container type through constructor and match projection, cover direct/nested
payloads and definition-site/import-site record identities, and reject wrong
or unresolved concrete elements. The original discovery source and assertions
remain unchanged. Full corrected source review precedes execution. Fresh
changed provider qualification must then reach every original eighteen and
additive eight native methods; passing annotation tests alone does not close
native list acceptance or enum parity.

My consumer audit also finds that C native generic-union payload emission sends
`TYPE_LIST_GENERIC` to `type_to_c`, whose empty result requires a special path.
I will emit the concrete list pointer from its complete annotation, retaining
existing provider-forward ordering. This is a static prerequisite revealed by
preserving the container, not a newly executed invalid C program. Compact list
name conversion must agree with explicit `List` plus one argument; signatures
and provider keys must use the same result.

My source checkpoint retains an Environment-owned checked scrutinee annotation
copy for each expression/statement match binding when its nominal view resolves.
A zero-initialized checker-only Symbol bit records successful provenance; unknown
views do not become root-owner authority. Legacy non-nominal union behavior is
unchanged, while nominal payload consumers require that bit and revalidate the
original union identity, exact variant, field template and argument substitution.
Fixed list elements use the union declaration owner; formal list elements use
the retained argument owner. This metadata owns no runtime Value and creates no
AST/schema field. Ordinary symbol insertion and header-constant insertion zero
initialize the bit; existing Symbol copies stay within their Environment.
The existing shallow checker owner registration and legacy payload copier retain
their documented fatal allocation boundary; the new checked annotation copy
returns failure without publishing a provenance bit.

## My nested projection amendment

The 3587 review identifies a remaining gap: a synthetic payload binding has
STRUCT expression type but a union annotation, and a single copied owner cannot
represent a nested field whose fixed leaves come from the definition while its
formal arguments come from the caller. I will retain a private owned nominal
view, not flatten those owners into the consuming module.

The private view keeps its original annotation, declaration owner and an owned
substitution-context chain. Each context owns copies of formal names, concrete
argument annotation and argument owner, so it retains neither a stack pointer
nor a movable UnionDef address. A payload view also records its exact selected
variant. Field projection constructs a new context from that original template;
whole-formal substitution moves to the corresponding argument owner/context.
Synthetic payload identifiers and aliases copy this view, and nested matches
select their own variant without treating the payload as an ordinary StructDef.

I will pass the context through view equality, array element/wrap operations,
branch agreement, mutation checks, inference and iteration. Runtime/codegen
annotations are separately materialized from the same view; they are not an
owner authority. The Symbol's private checker view is Environment-owned through
an explicit checked destructor registration. Temporary views free their complete
owned closure; retained views are registered once and freed once at Environment
teardown. Failed copy/registration does not publish a view. Existing ordinary
symbol constructors initialize the pointer to NULL, and same-Environment Symbol
copies borrow the retained view. No runtime Value or AST/schema field owns it.

My additive controls must exercise both annotation encodings, fixed versus
substituted same-spelled imported records, actual direct constructor/call/field
scrutinees, payload aliasing and nested matches. I will preserve all prior source
vectors and refuse unresolved or contradictory facts; no corrected execution
precedes the complete source/fixture review.

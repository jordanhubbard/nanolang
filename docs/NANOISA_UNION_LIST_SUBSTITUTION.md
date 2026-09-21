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

My superseded 3587 source checkpoint retains an Environment-owned checked scrutinee annotation
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

My nested source checkpoint replaces the earlier bit with an Environment-owned
proof pointer. I copy contexts before the projection stack unwinds and materialize
an independent annotation for emitted storage. I preserve that proof through
both match binders, inferred aliases, repeated binding checks, assignments and
array iteration, including the native/NanoISA match and shared loop-binding restorations. Same-Environment
Symbol copies borrow it; no copy transfers ownership. Both ordinary Symbol
construction and module header constant insertion initialize their whole storage.

The new registry-node operation returns failure without taking ownership. If the
materialized annotation is registered before the later proof registration fails,
that unpublished copy remains Environment-owned until teardown; I do not claim
allocation rollback of the registry itself. The output Symbol is unchanged.
The allocation fixture measures every attempted allocation in the actual
checker/env copy, materialization and publication path, denies each prefix and
single position, verifies unchanged output, tears down retained intermediates,
and repeats successful recovery. Legacy fatal copiers outside this checked path
retain their prior explicit limit.

My direct identity fixture checks copied nested contexts after changing the
original arguments, compact/explicit key agreement, exact variant refusal,
payload aliases, unresolved leaves and depth refusal. The native source fixture
adds constructor, direct and qualified call, record field and nested match
scrutinees with caller and definition-site records. I preserve the original
discovery program and all prior methods. Only static Python parsing and
`git diff --check` have run at this checkpoint; compiler and runtime acceptance
remain pending source review.

## My callable consumer completion

The cd7 review finds that my old callable view copies a materialized signature
and a single owner. That cannot represent a selected payload signature with both
fixed definition-site records and substituted caller records. Its ordinary-record
field lookup also misses selected payload fields. A retained proof that never
reaches the consumer does not establish provenance.

I will use the same owned NominalView for callable values. Selecting a call target
and projecting its result are separate operations: a direct declaration supplies
its declaration signature, a function expression or symbol supplies its retained
view, and result projection copies the return annotation under that view's exact
context. Function-valued returns repeat this operation without flattening owners.
Branch representatives compare whole context-bearing annotations. Materialized
signatures remain emitted/ABI metadata and cannot authorize identities.

Indirect arguments preserve the existing coarse type/arity check and compare
nominal leaves against the original contextual parameter annotation, recursively
through literals and complete annotations. Scalar conversions retain their existing
rules. Callable pass/return/assignment, inferred aliases, map/filter callback
parameters and results, and repeated checks must consume the same owned view.
No unresolved or swapped nominal leaf is accepted by matching spelling.

I will add owning-path controls that retain a mixed-owner callable payload,
project it into an alias, select it as a call target, compare branches and project
nested callable returns. Nested tuple/list/function annotations and every measured
new checked allocation prefix remain required. This amendment does not qualify
cd7 or drop any original 18+8 method. Complete source and fixture review precedes
all execution.

## My synthesized constructor owner boundary

The callable completion audit also finds that destination application installs a
materialized annotation on AST_UNION_CONSTRUCT, while a later value view reads it
under the current module. That is not a valid origin for fixed declaration-site
leaves. I will retain an Environment-owned expression proof keyed by the exact
constructor AST, in addition to Symbol proofs. It borrows only the AST key, owns
the full view, and is destroyed with its Environment. Its destructor never
dereferences the borrowed AST key. Failed public checks unlink newly published
keys before their fresh ASTs can be freed; successful keys are consulted only
while their checked source AST remains live.
Only constructor paths consult this registry; unrelated expressions have no cache.

I will validate original explicit arguments and nested field values against the
original expected owner/context before publishing a proof. Rechecking the same
constructor must agree with its retained complete view; a conflicting destination
cannot overwrite it. Nested constructor proofs can remain registered if a later
sibling fails, but the complete expression remains rejected. This is the same
explicit non-transactional annotation boundary already documented for child facts.
Materialization can change emitted metadata only after that check; later views
clone the retained proof rather than treating the copy as a new declaration.
Registry allocation failure refuses before proof publication. No AST/schema field
or runtime value owns this registry, and no proof crosses Environments.

## My emitted local binding boundary

My secondary-checker audit finds that native and NanoISA local declaration
restoration copies emitted TypeInfo but drops the retained callable context. I
will restore the original checked declaration's proof and owner fields using
its exact name, source file and location before later checks consume that local.
The proof stays owned by the same Environment; emitted spelling does not become
new authority. I copy metadata before symbol storage can grow. Parameter
annotations retain their original declaration owner; runtime value bindings do
not acquire checker proof ownership.

My branch-value comparison excludes arms that the ordinary control-flow checker
identifies as definite function returns. Those arms have their own return
contract and do not supply a callable/array join value. A branch with no surviving
value cannot manufacture a nominal view. This preserves lexical control flow
while requiring exact agreement among every surviving value arm.

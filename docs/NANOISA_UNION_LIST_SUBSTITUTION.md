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

## My callable completion source checkpoint

I now select callable targets and project their results through owned NominalView
contexts. Inferred symbols retain that proof; pass/return/set, indirect arguments,
map/filter and exact reduce callback comparisons consume original annotations.
Indirect calls publish copied ABI signatures only after their contextual argument
checks succeed. Full callable returns also populate the duplicate ABI signature
from the checked materialized result. The legacy declaration-signature copier
retains its disclosed fatal-allocation boundary; my new checked-copy fault domain
does not claim to make that old API recoverable.

I retain original constructor context before emitted annotation replacement,
refuse conflicting destination rechecks, and unlink new registry keys on failed
public program/module/shadow checks. The registered graph allocations still live
until Environment teardown. I restore same-declaration local proofs in both
emitters before secondary checks; those consumers cannot derive identity from
materialized spelling. Existing source nominal binding supplies emitted record
names; full contextual views remain the checker authority.

My additive C controls retain selected payload callbacks, alias them, compare
fixed Definitions.Item against formal Caller.Item, check indirect arguments and
nested callable results, and exercise map/filter input/result identities. Separate
fault controls cover owned callable tuple/list/nested-function materialization
and constructor registry publication in every measured prefix/transient position.
A parsed failed-module control requires registry rollback and destroys the AST
before Environment cleanup. These are fixture assertions, not observed passes.

My native imported-source vector adds field-to-alias calls, pass/return/set, cond
and match selection, an early function-return arm, and nested callable returns.
Two C-seed-only swapped-owner source refusals preserve prior output and require a
type diagnostic before C compilation. The original 18 methods and eight native
methods remain, with no original source/assertion removed. Aggregate map/filter
fixtures establish only checker obligations; aggregate runtime parity stays open.
Static Python parsing and diff whitespace checks pass. I have not built or run
this checkpoint; independent complete source/fixture review precedes execution.

## My assignment destination lifetime boundary

Independent source review finds that AST_SET borrows a Symbol vector entry across
recursive RHS checking. Branch bindings can grow that vector; a later read of the
entry is invalid even though its declaration remains live. I will snapshot the
selected destination metadata before every RHS checker, including contextual
constructor application and callable comparison. The snapshot retains the exact
declaration already selected, rather than looking its name up after branch scopes
change. Its TypeInfo, owner strings and retained proof belong to the checked AST
or Environment and remain live through this synchronous check; the checker does
not replace or free the selected declaration while checking its RHS.

Borrowed-field assignment also borrows a StructDef vector entry. I will snapshot
the selected field's type, annotation, name and owner before recursion. Registered
record declarations append to a growable vector; their separately owned field
arrays/annotations remain live until Environment/provider teardown. RHS checking
does not tear down those owners. I will exercise corrected assignments with
branch-local bindings that cross a symbol capacity boundary and reuse the
destination name, preserving the original destination proof and refusal policy.
No unfixed memory-fault path will execute.

My correction snapshots the selected Symbol and borrowed field metadata before
any recursive RHS check. Explicit callable comparison follows ordinary RHS
checking so its lexical branch bindings exist; contextual constructor preparation
still precedes that check. Four new C controls cross the actual symbol capacity
boundary while adding a same-named Boolean local: retained-proof and explicit
callable destinations each accept matching owners and refuse swapped owners.
They assert the original indexed declaration/proof remains unchanged. This is
a corrected-only fixture checkpoint; no compiler or fixture has executed.


### 5.1 Explicit constructor argument annotations (144 first terminal)

I retain both 144 fresh build/bootstrap/provider passes. My first Linux
GCC-ordinary focused group passes the storage and scheduler methods, then
refuses the valid Box<Item>.Value constructor in constructor_failure_rollback
before its assertion at line 461 (outer status 1, 4.886935102 seconds;
log SHA256 6aeace5b3990f3d9c20b3bc978432a909c7b6c4bb92a5d9fec9019f6e99f89e4).
Inputs remain equal and the process group is reaped. Darwin preparation passes;
I hold its matching matrix and all dependent source/native gates.

Static inspection identifies parse_generic_type_args passing NULL for the
record-name and callable-signature outputs of parse_type_with_element. A plain
Item argument consequently becomes TYPE_STRUCT without its declaration name;
the complete checker correctly refuses that missing fact. This parser boundary
also discards a direct callable argument's signature. I preserve both outputs
in an owned TypeInfo, keep complete nested annotations, and clean partial
arguments transactionally. I retain exact identity refusal and the original
constructor rollback assertion.

- [ ] Preserve complete explicit constructor argument annotations and check
  parsed record, callable, nested and malformed argument controls.
- [ ] Review the source checkpoint before corrected full qualification;
  original 18 methods, native 8 methods and all allocation gates remain required.


### 5.1 Nested parser annotation losses remain required

While tracing the 144 constructor refusal, I also find two earlier parser
boundaries: array element parsing passes no callable-signature output, and tuple
parsing obtains child TypeInfo but stores only flat tags/names. My explicit
constructor output repair does not establish either deeper syntax path. Full
nested callable/container parity remains open. I coordinate with the SDK
carrier lane's d1d2d5556 owned tuple-child implementation before changing the
shared annotation representation; I do not introduce a competing schema.

- [ ] Reconcile owned tuple child annotations, all copy/free/equality/materialize
  consumers and nested callable metadata with the reviewed SDK implementation.
- [ ] Preserve callable annotations at array element parsing with exact owned
  output/cleanup semantics and source-level nested controls.
- [ ] Qualify complete nested shapes after source review; narrower constructor
  acceptance cannot close the full generic-list or callable parent.


The SDK review refines my array finding: the local a43 TOKEN_FN branch loses a
signature without fn_sig_out, but the SDK lane already repairs that boundary in
4170979a3. At its current 4205973e0, a complete array request supplies nested
TypeInfo and TOKEN_FN publishes the owned signature through that carrier.
Thus I must integrate and qualify this existing carrier repair, not duplicate
an array-specific change or call the SDK path defective merely because its
separate signature output is NULL. Its tuple foundation remains d1d2d5556.

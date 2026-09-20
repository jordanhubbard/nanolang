# My array element declaration provenance

I record this prerequisite under task_1a4cba8a04804aa287e5b9d6eefb98f7 before
changing my checker. My frozen 9f4e Darwin build compiled the parser and then
rejected `let state: ResourceState = (at states i)` at typecheck.nano:525:36.
The ordinary checker already infers STRUCT from the array, but my new exact
nominal resolver has no array-index path. I preserve that first terminal.

I do not repair this by resolving the returned spelling in the current module.
An array annotation and its declaring owner travel together. I introduce an
internal expression annotation view containing a borrowed TypeInfo and its
owner, with explicit success separate from a NULL owner (the root module is a
valid owner). The view has a bounded depth of 128 and never changes source facts.
For synthesized views I use checked owned copies and a single cleanup operation;
no pointer to temporary signature storage escapes.

I resolve identifiers from their symbol annotation and retained nominal owner;
record fields from the resolved record declaration and that declaration's owner;
direct and qualified call returns from their registered signature owner; and
indirect returns from the checked callable signature and its paired owner. I
preserve local callable and declared function precedence before builtin logic.
Indexing descends exactly one ARRAY element subtree, so nested indexing retains
the original declaration owner. Missing element facts refuse nominal use.

I require agreement across every IF, COND or MATCH result using my existing
recursive owner-aware annotation comparison, retaining the selected annotation
with its own owner. Blocks and returns forward their result. Array literals
must agree across every element; an empty literal has no standalone nominal
identity but retains its existing contextual destination typing. Array-preserving
builtins (push, slice and filter) forward their source facts only when the actual
builtin is selected; array_new uses its initializer and map uses the actual
callback return signature. These operations do not manufacture record identity
from a bare name. Their original bounds and base type checks still apply.

I retain annotation ownership for inferred array bindings, including repeated
checker passes, rather than relabeling copied imported annotations as local.
For explicitly annotated bindings, parameters, fields and returns, destination
facts retain their declared owner. I audit array-record boundary consumers so a
raw array_record_name/env_get_struct fallback cannot bypass the exact identity
that indexing now requires. Enum numeric conversion and formal union handling
remain their existing separate policies; I do not admit enum list operations.

My fixtures cover local and imported indexing, nested arrays, fields, direct
and indirect returned arrays, inferred aliases, branch agreement, array-producing
builtins and user/local callable name precedence. Same-spelling distinct records
and unresolved element leaves refuse. I preserve the unchanged typecheck graph
and full compiler build as acceptance, not a reduced source substitute.

I submit implementation and fixture checkpoints before running any new compiler
or qualification command. The root-index timeout work remains open until all
original required graphs pass; parser progress alone does not close it.

My iteration metadata needs to outlive a temporary annotation view. Before
implementing that publication, I extend my private checker allocation registry
with an explicit destructor selector: existing allocations still use shallow
free, while a new checked registration accepts only a newly owned complete
TypeInfo tree and destroys that tree at Environment teardown. Registration
failure transfers nothing. Symbols remain borrowers and never register the same
tree twice. I register only after the iterable view succeeds, and do not publish
the loop symbol on registration failure. This adds no Environment layout field
or runtime value ownership change. Inferred LET views instead transfer their
tree to the existing AST annotation owner.

My union-payload visitor must not assign one owner to a mixed substituted tree.
I retain the original declaration template and an explicit, stack-local
substitution context containing the declared formal names, concrete argument
annotations, and the argument owner's identity. Recursive annotation comparison
switches owner only when it reaches that exact declared formal; fixed leaves
keep their definition owner. Resolved argument subtrees are compared without
reapplying the template substitution. The old public-internal comparison wrapper
uses NULL contexts and retains its existing behavior. I carry both comparison
sides' contexts through nested callable signatures and duplicated signature
facts. This is comparison metadata, not a new source type or runtime layout.

My source checkpoint uses `src/typechecker_nominal_arrays.inc` for checked owned
views, and the existing bounded signature snapshot implementation for checked
TypeInfo copying. Failed views leave their output untouched and discard only
new trees. The typechecker object explicitly depends on the new include.

My array-boundary wrapper now receives the destination owner at parameters,
fields, locals, assignment, returns and both module passes. The original union
constructor materializer is separate from a recursive origin validator. Nested
union templates retain a linked stack of substitution contexts; a formal switches
to its argument's original context, never back into its own declaration template.
The public-internal equality wrappers supply no contexts, preserving ordinary
comparison behavior. Contextual empty arrays validate their destination facts
before accepting zero elements.

I retain inferred array trees under the AST owner and register synthesized loop
annotation trees under the explicit checker-tree destructor. This is cumulative
checker metadata until Environment teardown, not expression-bounded reclamation.
I have not run this checkpoint. Fresh full build/bootstrap and all previously
required source, allocation and lifetime gates remain pending.

## My static identity correction before execution

Root review of 245710e1c identified two invalid shortcuts before any run. A NULL
function body does not identify a builtin. My actual builtin declarations are
static cache objects populated from BUILTIN_LANG registry rows. I will expose a
predicate for those exact objects and use it in both array intrinsic predicates;
registered externs and copied Function values do not acquire builtin identity.
The existing reserved-name lookup order remains explicit and unchanged.

My array-identity requirement also used ownerless union/opaque lookups and a
single-capital-letter heuristic to bypass checks. I remove those shortcuts.
An exact ordinary record declaration takes precedence; only an exact enum or
union declaration in the annotation's owner can use its existing separate type
policy. Declared union formals are resolved through the explicit substitution
context. Missing declarations do not become formals because of their spelling.
OpaqueTypeDef currently represents a global foreign C-pointer namespace and has
no module-owner field. Its presence cannot exempt an ordinary-record array from
identity comparison; I do not invent module provenance for that legacy table.
I retain enum-list parity and all original gates as open requirements.

## My complete fixture checkpoint

I retain the prior nine list/lifetime methods and the actual builtin identity
method. Three additional methods exercise array source paths/producers, rejected
boundary/owner mismatches, and mixed fixed-definition/substituted generic-union
leaves. Each source route keeps C seed, Stage1, Stage2, actual evaluator and
NanoVirt/NanoVM controls; imports include repeated aliases of one declaration
and same-spelling distinct records. Local and imported arrays cross fields,
parameters, direct/qualified/indirect returns, inferred aliases, branches,
nested literals and iteration. Empty arrays need destination facts. I keep all
original compiler shadow deadlines and the unchanged LexerToken input.

A separate two-TU fault binary includes my actual env.c and typechecker.c under
malloc/calloc/realloc/strdup/free hooks. It excludes exactly env.o/typechecker.o
from the prepared provider closure. My old evaluator fault binary and its three
hooked TUs are unchanged. I measure and sweep every allocation of the checked
TypeInfo graph copy and selected array views, both persistent-prefix and single
transient failures, with fresh recovery and output/retained-input checks. A
transient branch probe may recompute a complete valid view; I check its exact
result depth/name/owner and cleanup rather than claiming that every injected
failure must terminate the operation. I separately fail the one registry-node
allocation, retain the caller-owned tree, then transfer it once and check full
Environment teardown. This is not a whole-compiler OOM recovery claim:
nominal_callable_view still uses legacy fatal signature construction/copying;
map/filter/callable source controls do not turn that policy into checked failure.

While authoring those source controls, I identify a further static prerequisite:
ignored-result array_push and array_set bypass the destination-view check. I
retain exact wrong-record controls and a valid array_set control in this
checkpoint. Their production check requires review before execution; I do not
call the source draft qualified or weaken those required refusals.

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

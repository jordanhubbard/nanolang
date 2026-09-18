# My non-admitting nested-array provenance contract

I record `task_078055bd21d343ee97c5b1cf9130fea4` after graph-core PR724 on
main `55208c47`. Private storage can now retain and collect array graphs, but
my shared managed profile still rejects them. Parent aggregate488 requires
both truthful child provenance and generated lifetime/collection before that
admission changes. This child addresses provenance only.

## My API boundary

I preserve `nvm_analyze_managed_arrays` and its existing leaf report/decision
contract. It remains the function called by `nvm_verify_profile`. I add a
separate `nvm_analyze_managed_array_graphs` query with its own report/free pair,
reusing the existing ELIGIBLE/UNRESOLVED/INVALID/LIMIT/MEMORY result statuses.
Neither query mutates its borrowed module; failed queries leave output pointers
untouched. There is no new wire metadata, profile flag or translator dispatch.

A graph report contains the existing per-origin declaration/storage/content-tag
facts plus per-origin possible child origins and an explicit unknown flag. I
keep the leaf report ABI unchanged rather than silently teaching old consumers
that a successful leaf query permits nested arrays. Internal code may share
transfers with an explicit mode; shared profile behavior must remain identical.

## My finite graph domain

Each abstract value retains possible tags, a uint64 origin set and unknown
provenance. Each boxed origin additionally retains a weak joined child value,
including child tags and child-origin set. Packed origins retain their exact
fixed storage kind and the existing finite portable coercion matrix; they own
no heap children. Graph-mode boxed children may contain the existing scalar/
string leaves and ARRAY7. Array values must retain authoritative possible
origins; an ARRAY tag without a resolved origin is not evidence of eligibility.

I admit ARRAY7 as a boxed declaration only to this graph query. Other nominal,
map, callable, imported or unproved heap kinds remain UNRESOLVED. I do not add
new instruction transfers beyond those already implemented in the leaf query.
The existing explicit instruction/function/stack/global/state/origin caps remain
in force. Allocation and limit failure retain their distinct statuses.

## My transfers and fixed point

Append/set/literal writes weakly join complete child values into every possible
boxed receiver origin. Each possible packed receiver still requires every
possible incoming tag to satisfy its storage matrix; I never select a favorable
alternative at a join. Unknown or unsupported written children remain unresolved.

GET and POP return VOID or the joined complete child value from every possible
receiver origin. A successful POP may remove a concrete edge at runtime, but
static summaries never narrow after removal. Nested reads therefore preserve
inner identities for later alias mutation instead of returning an ARRAY tag
without origins. Scalar/string contents preserve existing exact tag behavior.

A slice allocates fresh outer origins keyed by site and possible declaration;
it weakly copies the complete child summaries, including nested child identities.
It does not copy children deeply or substitute the receiver's outer origin for
its result. Repeated executions at one allocation site weakly summarize all
surviving instances, including conservative self-edges and cycles. I do not
infer ownership uniqueness, graph depth or concrete acyclicity from that
abstraction.

CFG/local joins, direct calls, recursive SCCs, results, globals and repeated
entry use finite monotone unions until convergence. Mutating an inner array
through any containing array reaches the same possible inner origins. Effects
from a later call/global write remain visible to earlier reads conservatively.
Unknown unused parameters/escapes remain explicit; absence of a call does not
prove an empty graph. No runtime root, reachability or collection decision is
made from this overapproximate graph.

## My acceptance and remaining lifetime dependency

I require ordinary VM controls and independent report assertions for nested
read/write aliases, shallow copies, duplicate edges, transferred POP, self/mutual
cycles, heterogeneous child tags, mixed packed/boxed outer joins, repeated
allocation sites, recursive calls, and globals/reentry. I verify unknown origins,
unsupported packed writes, caps and deterministic allocation failure. Native
analysis sanitizers, unchanged canonical module bytes, existing leaf reports,
unchanged backend refusal/output preservation, and all current profile/runtime
regressions remain gates. Passing graph analysis alone cannot admit execution.

I separately record `task_4070da262082405280b5f65abf9581c1` for generated
collection safe points and final nested lifetime admission. The next contract
must pin complete live owners at completed instruction/frame/module boundaries,
first-error preservation, temporary-root ownership, persistent globals and an
allocation-pressure strategy. It must keep collection outside an uncommitted
mutation and qualify failure/recovery plus bounded-live cycles in repeated
native LLVM/Wasm entries. An explicit private collector called only by tests
cannot satisfy those generated-code obligations. Full aggregate488/managed51da,
nominal/map/callable graphs and release scope remain open.

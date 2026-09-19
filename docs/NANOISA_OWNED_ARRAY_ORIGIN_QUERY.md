# I prove private owner ARRAY field origins before admission

I refine the second checkpoint of [my runtime plan](NANOISA_OWNED_FLOAT_ARRAY_RUNTIME_PLAN.md), task430220. My first query PR824 describes paths without element facts. This addendum proposes a separate private instruction query. I hold production and fixtures for independent review. No public verifier, selector, runtime, source producer or existing mixed query changes in this checkpoint.

## My API owns facts, not runtime authority

I propose an opaque `NvmOwnedArrayOrigins`, created by
`nvm_analyze_owned_array_origins(const NvmModule *, NvmOwnedArrayOrigins **)`
and released by `nvm_owned_array_origins_free`. Its result carries a status
(PROVED, UNRESOLVED, INVALID, LIMIT or MEMORY), function index, instruction PC
and a static diagnostic. PROVED means the bounded origin obligations below,
never affine liveness, service authority, complete scalar verification or
permission to execute. A null module/output refuses; every failure preserves
`*out`. I borrow immutable module bytes only during analysis and own all returned
facts, including the first query's exact layout/ownership snapshot.

Indexed getters copy a complete fact only after checking every index. Invalid
indices leave outputs unchanged. I expose counts and copies for concrete sites,
function input/result paths, field-origin facts and scalar obligations; no
mutable internal arrays or caller-built proof imports. A site identifies
function/PC and allocation opcode, not a runtime allocation identity. A path
identifies parameter/result ordinal, original global layout and the exact
sequence of field indices from the first query. I preserve source/global maps
and RESOURCE flags, never compacting an owner into an ordinary managed row.

## I bound representation and work before allocation

I retain eight functions, an acyclic complete call graph, 256 locals and256 stack
cells per function,4,096 total decoded instructions,64 concrete ARRAY-producing
sites,256 symbolic ARRAY input paths and256 ARRAY result paths per function.
Each origin set has five64-bit words:64 concrete module sites and256 formal
paths belonging to the currently analyzed function. Tags, exact nominal layout,
uninitialized and unknown states are separate from these bits. No empty set
stands for an unknown origin or successful proof.

I cap the sum of materialized instruction-state local/stack cells at1,048,576,
stored field-fact cells at65,536 and dequeued worklist visits at262,144. These
budgets include intermediate joins, summary instantiations and nested path
facts, not merely the final exported arrays. I check every count/product before
allocation and reject before exceeding a budget. Per-instruction scratch storage
is bounded by those same counts and discarded before the next transfer. The
existing256-layout,65,536 descriptor-path,32-depth/path and16MiB ownership limits
remain. I do not expand every descriptor path into every instruction state
without charging the field-fact budget.

## I interpret every accepted transfer explicitly

I decode complete bodies and validate instruction boundaries and branch/call
indices before following them. I analyze every function, including uncalled
helpers and shadows. Unsupported opcodes and signatures refuse; a default
transfer never fabricates tags or origins. The accepted scalar/control subset
must be enumerated from the existing mixed FLOAT analysis in the production
checkpoint, with explicit additions for STRING transport and owner field facts.
Exact existing numeric rules are preserved; this contract does not admit new
STRING arithmetic, FLOAT owner fields, references or service instructions.

ARRAY construction requires the exact existing FLOAT element declaration and
initialization obligations; constants and typed operations propagate their
actual tags. ARRAY literals check every initializer. Copies, stack permutations,
local stores/loads and branch joins retain origin sets and exact initializedness.
An uninitialized alternative remains uninitialized after a join. Loops reach a
finite monotone fixed point; visit exhaustion refuses. ARRAY reads remain
FLOAT|VOID, length remains INT, and writes/appends require FLOAT for every
possible receiver origin. Every supported alias-producing array operation must
retain or create its explicitly defined origin; unmodeled slices/copies refuse
rather than inherit an assumed FLOAT origin.

OWN_PACK consumes the exact declared field ordering and checks each scalar tag,
STRING tag, nested owner nominal and ARRAY origin set. It builds path facts by
prepending each field index. OWN_MOVE and owner local transfers preserve the
entire path map. Owner projection yields the selected ARRAY's origin set while
retaining a relation to its original shell path; nested owner observations may
not become an independently movable owner merely through this analysis.
OWN_UNPACK_LOCAL distributes exact path suffixes to children and separately
rooted ARRAY/STRING values. Scalar/STRING siblings never acquire array origins.
The query records shape facts only: the later independent affine authority must
check consumes, duplicate moves, initialization and unique shell identity.

## My call summaries prove all origin alternatives

I process callees before callers. Each owner parameter's ARRAY leaves receive
unique formal path symbols. A callee's operations on a formal ARRAY generate
FLOAT element/write obligations for that symbol; formal symbols are conditional
requirements, never concrete allocation evidence. Each call substitutes the
actual owner's complete path sets into all callee result facts and obligations.
No callee-local formal symbol may escape substitution. Concrete allocation sites
retain their module-wide indices; repeated calls and loop iterations at one site
remain distinct runtime objects even though they share a possible-origin fact.

All continuing returns join exact result layouts/path facts; incompatible or
unknown/uninitialized alternatives refuse. Nested returned owners and relay
calls use the same substitution. Arguments are checked in their existing stack
order without inventing source evaluation. Entry parameters with ARRAY leaves
have no closed concrete origin and refuse. Bare ARRAY parameters/results remain
outside this profile. Uncalled helpers still require valid symbolic transfers,
but cannot lend unsupported assumptions to a reachable call. Before PROVED,
every reachable ARRAY consumer resolves to at least one checked concrete FLOAT
site and every substituted scalar obligation is satisfied; unresolved formal
bits at a closed entry are a refusal.

I do not derive array element safety from ownership bytes, nominal layouts,
ARRAY tags alone or an earlier successful proof. Required-service presence
refuses first, including partial metadata. Whole-root borrows through ARRAY or
STRING siblings, linked modules and unmodeled external calls stay refused. The
existing ordinary Samples query and its callers remain untouched.

## I qualify the query before considering runtime use

After production review I require query-only controls for local pack/projection/
unpack, nested owners, direct and relay owner returns, two formal fields with
shared actual origins, multiple possible concrete origins, loops and branch
initialization joins, uncalled helper bodies, STRING siblings, empty arrays and
reordered observation. I require checked refusals for mixed element writes,
unknown or absent origins, missing result paths, incompatible nominals, recursive
calls, unsupported borrowing/services, bounds and allocation failures. Each
failure preserves result/getter outputs and leaves module bytes unchanged.

I test summaries using integer identities and exact path/origin expectations;
I do not execute pending owner ARRAY modules. GCC/Clang strict sanitizer query
gates and existing retained/mixed/provider adjacency precede any later authority
integration. Runtime mutable aliases, ownership cleanup, public admission and
unchanged Bundle/PREFIX source acceptance remain separate required checkpoints.

# I prove a finite cyclic File state graph without execution

I refine task_243a9a5809bf422caab0bddafa447098 under72556/6931 after
[my design sequence](NANOISA_FILE_CONTROL_CALL_EXTENSION.md). This is the next
production contract proposed for review, not authorization to change admission.
I leave dfa149's acyclic public API, ABI and installed packaging untouched.

## I choose a separate, closed query

I propose opaque `NvmFileCyclicReport` and private C API
`nvm_file_cyclic_analyze(const NvmModule *, NvmFileCyclicReport **)`, returning
existing `NvmFileFlowStatus`. Input is immutable during the call; output and
input storage are disjoint. Failure preserves the prior output pointer. Success
owns copied declarations, decoded instructions and facts, borrowing nothing.
`nvm_file_cyclic_free` releases every report allocation. External serialization
remains the existing private-query precondition. This is module analysis, not
serialized hosted preparation, target readiness or a grant.

I retain the exact catalog, nominal/global mappings, operand validation, all
function/dead-instruction checking, old scalar opcode inventory and direct
acyclic call graph. CALL_REF still has exactly one borrowed formal. I add only
intraprocedural cyclic successor analysis. Unsupported dead opcodes still refuse;
no unreachable instruction is silently exempt from structural validation. I
analyze every declared function from its exact parameter state even if uncalled.

I separate syntactic successor construction from the existing DAG requirement:
the new private path may retain cycles, while the old preparation still applies
its identical topological rejection. No old report changes type or meaning.
Tarjan or equivalent SCC discovery uses bounded iterative storage, never native
recursion proportional to instructions. I check direct callees in the existing
callee-first order; recursive call SCCs remain UNRESOLVED.

## I define one concrete abstract state

Each state records the original function index; initialized flags and exact
`NvmFileFlowDeclaration` for every declared local and occupied stack slot;
stack height; exact Result arm NONE/UNKNOWN/OK/ERROR; owner-equality labels;
ordered active region stack; and all256 reference entries with live/formal,
original local or parameter origin, owner label, epoch label and region depth.
Unused fields are zeroed before comparison. Nominal identity includes original
global index and catalog ordinal; tags alone never identify File or Results.

INT/BOOL payloads have no constant domain. Conditional scalar branches explore
both successors; only the existing exact Result-arm refinement removes an
impossible arm. Each File/OpenResult root owns one label; two owning slots with
the same label are invalid. References observe an owner but never become roots.
Borrowed parameter anchors are distinct from local owning labels and cannot be
renamed into them. VOID is initialized VOID, not an unknown or empty slot.

I store no accumulated execution count, allocation serial or historical drop
count in the abstract state. Runtime generation and borrow epochs are absent
from the proof representation: they remain mandatory runtime checks. Original
flow `next_identity` and `cleanup_obligations` cannot be reused unchanged as
loop state. No code may reset those counters in the old public flow objects.
The new adapter uses a private bounded label allocator and read-only declaration
facts; any shared transfer factoring must preserve old query behavior exactly.

## I canonicalize only a checked live graph

After each atomic instruction transfer and before interning a successor:

1. I validate that every live local owner, operand owner, reference and region
   has a well-formed origin. Each local reference must point to the same live
   File local it observes; each formal refers to its immutable parameter anchor.
   A dangling reference, duplicate exclusive observer, duplicate root or
   inconsistent origin refuses before normalization.
2. I scan owning locals by original local index, then operand slots bottom to
   top. I assign dense labels1..N in that order, maintaining a bijection from
   the transfer's temporary labels. Every reference to an owner follows that
   same map. External formal anchors remain a separate parameter-index domain.
3. I map each live region to its depth in the ordered region stack. I map local
   borrow epochs to reference-slot index; formal aliases use their parameter
   anchor. I first verify uniqueness and exact region membership, so this is
   renaming a live relation, not merging two epochs or ending a borrow.
4. I clear unused storage and compare semantic fields explicitly, never struct
   padding. Deterministic hashes may accelerate lookup but equality must compare
   every field. Hash collision never proves equality.

There are no hidden pending owners at instruction boundaries: CALL and service
transfers are atomic abstract operations with explicit failure/root obligations.
The query does not infer their runtime completion. A future target must stage
those owners exactly as the existing matched runtime does.

A fresh owner replacing a consumed owner at the same local can normalize to the
same abstract label only after transfer proved the old root absent and every
old observation ended. The report proves a location/lifetime relation, not
physical identity across iterations. An unchanged live owner may retain its
location relation through a backedge. Any operation that moves, overwrites,
closes or drops a held owner refuses before this renaming step.

## I join by finite alternatives, not optimistic merging

At each instruction I retain a set of at most16 distinct canonical input states.
Join is exact set union with deduplication. No initialization bit, owner
occupancy, Result arm or reference origin is widened away. The entry state is
inserted before any backedge; zero-iteration paths therefore remain present.
Every alternative must transfer successfully. A malformed state/operation gives
INVALID; unsupported exact combinations give UNRESOLVED. A17th alternative or
budget exhaustion gives LIMIT and no report. A known-unreachable Result edge
contributes nothing. Scalar constants never justify dropping an alternative.

Different stack heights, initialized locals or region shapes may exist as
separate alternatives in this non-admitting report; a later runtime/hosted
consumer must understand all variants or refuse. This does not retrofit one
arbitrary input_stack value into an old single-state body fact. A LOAD reached
with an uninitialized alternative fails even if another alternative is valid.

I enqueue each newly interned (instruction,state) pair once, FIFO, instructions
and successor edges in their original stable order. I process to exhaustion;
only then is the function checked. Every reachable RET must satisfy exact
result/owner/reference/region exit rules. I retain the existing requirement of
at least one reachable valid RET per function; no proof of eventual return is
claimed. Nontermination requires the later matched runtime fuel contract.

## I give finite bounds and a termination argument

| Quantity | Exact first checkpoint bound |
| --- | ---: |
| Functions / locals / stack / references / live owners | Existing64 /256 /256 /256 /256 |
| Instructions per function / total / CODE bytes | Existing256 /4096 /65536 |
| Canonical alternatives per instruction |16 |
| Interned pairs and transfer executions per function |4096 |
| Interned pairs and transfer executions over the module |65536 |
| Propagated successor alternatives over the module |131072 |
| Pending queue entries per function |4096 |
| Accounted live query storage |16MiB, including copied inputs/facts, states, queue, hash tables, report and transient overlap |

I check multiplication, addition and index bounds before allocation or mutation.
Allocation failure returns MEMORY; an exceeded bound returns LIMIT. The16MiB
ceiling may refuse far below the combinatorial limits; it is not a promise that
all maximal dimensions fit simultaneously. No allocation is made for a maximum
Cartesian state universe. Each unique pair consumes a bounded slot and is
processed once. A duplicate adds no work; a new alternative consumes one of the
finite slots. Thus work ends after at most65536 transfers or an earlier checked
refusal. Each transfer, canonical scan and equality comparison is bounded by
the fixed dimensions and at most16 candidates at one instruction. Calls use
already completed summaries and cannot recurse through analysis.

Cleanup/obligation facts are monotone per instruction and input-variant, not
path-history counters. I retain the exact existing service/call requirement
mask, target and outcome semantics for each variant; cleanup kinds include
local/stack drop, assertion, call, service and return. Repeated visits do not
append another dynamic event. Peak roots/stack/regions are maxima over states;
this report gives no finite total number of runtime closes or cleanup failures.
Such counts remain checked dynamic runtime data, not a statically reset counter.

## I publish an immutable, distinct report ABI

The header exposes revision1, opaque report/free and copying accessors for
summary, original function/local facts, decoded instruction/successors,
variant count, variant input-state entries and output edge variants, plus exact
pending masks/cleanup facts. Accessors use original function and instruction
indices; variant ordinals are stable for the deterministic traversal, never
nominal IDs. Out-of-range access leaves outputs unchanged. No mutable internal
pointer or old `NvmFileBodyReport *` conversion is provided.

Summary fields include revision, function/instruction/variant counts, processed
transfer count, storage peak and `runtime_admitted=false`. A variant records
input/output stack and root/reference/region counts, original cleanup local,
exit/refinement flags, exact obligations and edge-to-variant indices. Copying
accessors expose full canonical slot/reference facts, not only a digest or
counts. All branch refinements and every successful edge are retained. The
publication pass validates bounds and every referenced variant index before
setting `*out`; partial graphs are never published. No wire format, installed
ABI or native-runtime semantic ABI revision changes at this query checkpoint.

## I require concrete proof controls before runtime work

| Program shape | Required query result and reason |
| --- | --- |
| Zero iterations, x initialized only inside loop then LOAD x | INVALID from retained uninitialized entry alternative. |
| Zero/many iterations with x initialized before header | Prepared; scalar values are abstract, both edges checked. |
| Each iteration creates OpenResult, consumes either arm, closes Ok File, ends all borrows before header | Prepared owner-empty invariant; no growing historical cleanup count. |
| Header File in local2, consume it then acquire replacement and return to local2 on Ok; Error exits after empty cleanup | Prepared only if old owner/observers are absent and every exit is valid; new physical owner is not claimed equal. |
| Same replacement while a local or forwarded reference to old File is held | INVALID before normalization; no relabeling escape. |
| Header File and balanced exclusive reference stay live through loop and are ended/closed on exit | Prepared if transfer preserves exact held relation and each loop service requires that epoch; runtime liveness remains pending. |
| Region BEGIN each iteration without matching END | LIMIT or earlier exact transfer refusal; never prepared by erasing region depth. |
| Branch preserves File on one edge and consumes on another, then unconditional use | INVALID on empty alternative, not a merged live owner. |
| Distinct simultaneous owners swap locations | Each transfer obeys empty-destination/no-held-owner rules; canonical labels preserve two roots, never one. |
| Result-arm loop, nested SCC, multiple exits, lower-index direct callee | All alternatives/exits and callee facts checked; original indices retained. |
| Cyclic call graph, indirect call, multi-borrow encoding, unknown/dead opcode | Existing structural INVALID/UNRESOLVED distinction retained; no new admission. |

Fixtures must exercise16-versus17 alternatives, exact work/byte boundaries via
bounded hooks where raw modules cannot reach a limit, allocation prefixes and
single transient faults, immutable input/prior output and complete cleanup of
partial reports. I compare acyclic query outcomes and old report bytes/fields
where meaningful without requiring alternative numbering to equal old IDs.
Both Linux and Darwin private query gates retain old hosted/body/flow/refusal
neighbors. No newly cyclic module executes. Production then fixtures receive
separate review before any gate; public/source/VM/native work remains pending.

## My first unexecuted production checkpoint

I define the concrete copying-accessor ABI in `src/nanoisa/file_cyclic.h` before
implementation. The summary, state counts and per-variant body/edge facts are
distinct from old body reports. Complete input locals/stack/references/regions
and exact nominal/import maps remain queryable. Original successor indices pair
with variant ordinals; an absent refined edge carries NO_VARIANT.

`file_cyclic.inc` owns bounded canonical states and FIFO pairs. It uses iterative
reachability bitsets to name SCCs by the least original instruction index. Each
transfer reconstructs a private scratch flow state and a separately allocated
copy of declaration facts; only that copy has a bounded temporary-label counter.
The immutable plan's declaration counters and every preexisting query state stay
unchanged. Existing transfer routines perform one instruction, while their
per-step cleanup/obligation facts are copied into the new variant report.

Live owners use labels1..256; formal anchors257..512 remain fixed; canonical
regions513..768 and local reference epochs769..1024 occupy disjoint domains.
The single-step scratch allocator starts at1025, with complete post-transfer
validation before bijective normalization. Fresh physical generations are never
reported as these labels. Function bodies reuse existing checked transfers with
a private callee-checked table and retain acyclic calls.

The only existing preparation change factors a private `acyclic` argument:
`nvm_file_code_prepare` always supplies true, preserving its original DAG
refusal. The new query alone supplies false after the same syntactic preflight.
Old selectors, runtime/hosted plans, grants, emitters and producer sources do not
consume the report. I add the include to existing file_flow and its explicit
Make prerequisites; no new link-provider or wire format is introduced.

All report, retained-site/node, workspace and scratch-declaration allocations
are accounted before materialization under16MiB, including conservative old-plan
storage and transient overlap. Partial failure frees only its owned storage;
the scratch borrows its report's nominal plan and does not release that borrow.
Publication validates every processed variant and edge reference. This
checkpoint is static-only: no compilation, fixtures or cyclic execution has run.
I request source review before preparing the meaningful fixture checkpoint.

## My unexecuted query fixture checkpoint

I retain and call the entire existing acyclic body fixture unchanged before
my cyclic cases. New instrumented and separately linked C fixtures inspect
actual reports only; no pending bytecode or File service executes. They cover
entry-versus-backedge initialization, zero-iteration LOAD refusal, owner
replacement after consume-close, held-owner replacement refusal, a live
exclusive borrow across a loop with CALL_REF, balanced per-iteration borrow/drop
and owner-empty headers, lower-index callees, nested SCCs and unbalanced exits.
Both nominal permutations retain original catalog/global mappings.

My16/17 boundary uses real decoded branches and distinct initialization masks
at one join, without changing the production bound. I compare each deterministic
report's state fields, every reference/region and edge variant, retain old DAG
refusals, compare acyclic body facts, and prove copied reports survive destruction
of original CODE/ownership input. An independently live old flow state retains
its historical cleanup counter and original declaration counter.

Query-only white-box controls additionally test bijective two-owner renaming,
duplicate roots, dangling/duplicate observations, exact storage arithmetic,
queue/module-pair ceilings and preflight size bounds. I label these separately
from constructible maximum wire programs. Every allocation prefix and each
single transient allocation failure requires unchanged output, zero tracked
query allocations and a fresh successful analysis. I add an opt-in allocator
hook in the inherited fixture; its default behavior and assertions stay intact.

The new runner selects compiler/flags explicitly, keeps every binary and raw
stdout/stderr, records command/status, and bounds child process-group cleanup
after timeout. Ordinary versus sanitizer flags are explicit gate configuration;
no sanitizer claim follows from default ordinary flags. Scope is query/provider
instrumentation, not whole-runtime instrumentation. I have not run this fixture,
a compiler or its runner; I request review before frozen gates.

### I correct the unexecuted fixture after review

Root's static review of1ffa identifies a runner gap: returning from communicate
after TERM could leave descendants alive, and Popen failure lacked a terminal
record. I preserve that unexecuted checkpoint and correct only fixtures. The
runner now writes stdout/stderr directly to retained files, records launch and
wait errors, probes the process group independently of leader status, sends TERM
then KILL when needed with separate5second bounds, and records leader reaping
and group disappearance. The same cleanup runs after ordinary leader exit. A
surviving group or unreaped leader is a failing terminal, never assumed gone.

I add an actual decoded loop swapping two File roots through a third empty
local. All six move/store instructions retain exactly two roots; stack staging,
empty temporary and distinct canonical roots are inspected. A separate otherwise
identical held-reference case refuses before moving the observed File. I also
exercise the exact transfer and edge counter guards with real decoded analysis
and white-box counter initialization, preserving precise maximum values and
cleaning the retained partial node. No production changes or executions occur.

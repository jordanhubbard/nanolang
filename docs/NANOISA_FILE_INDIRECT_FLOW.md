# I compose every indirect target with File ownership

I continue task_2c135a488bd61576caf83debb2786270 after the actual merge of
PR900, `8b84b9cd012d3db595bcb461701e76f84e488cbb`. My first target query
retains complete local same-module target sets and exact signatures. It does
not prove owner movement or cleanup. This next checkpoint proposes a separate
private ownership report; no old hosted or runtime entry will accept it.

## My shared proof and distinct report

I propose opaque `NvmFileIndirectFlow` with transactional analyze/free and copied
summary, declaration, decoded-instruction, variant, input-value/reference/region,
and indirect-call getters. Its summary explicitly says runtime_admitted=false.
A new call fact contains the original function/PC, complete candidate bitset,
checked-candidate bitset and common ownership obligation. That obligation's
single-target field is NO_INDEX: consumers must use the complete candidate set.
I do not expose an inner NvmFileCyclicReport pointer or manufacture an old
single-target hosted certificate. Wrong getters preserve caller outputs.

I first prepare the existing complete target query. Its immutable input
precondition extends through the second checked decode/declaration preparation.
The new report owns both results; no caller-supplied target report or code pointer
can substitute for that conjunction. Complete original tables, unused bodies,
branch targets and original function/PC identities are checked. Existing query,
acyclic and cyclic entrypoints retain their old declaration/opcode modes.

I reuse cyclic owner canonicalization and transition machinery through an
explicit internal callable mode. The ordinary cyclic entry passes no callable
context; its allocation accounting and exposed facts stay unchanged. The new
mode adds exactly FUNCREF and CALL_INDIRECT transfers after the target query
has accepted the whole program. FUNCTION values are initialized, non-owning,
non-borrowed local/stack facts. FUNCREF checks the original function index before
pushing that category. LOAD/STORE/DUP/POP use the same checked category transfers.
The target query has already refused arithmetic, aggregate packing, callable
parameters/results and unsupported callable sources. I do not reinterpret an
integer value or relax the public scalar push API.

I do not need to discard owner alternatives or correlate a favorable callable
with one owner state. At every incoming ownership variant I check *all* targets
in the query's conservative site set, even if a more precise correlation could
exclude one. This may refuse a program requiring relational target precision;
it cannot grant authority by selecting only the first matching function.

## My complete candidate transition

I build the combined direct/indirect graph from checked original sites and all
candidate bits. The existing target query refuses recursion; I independently
schedule all functions in callee-before-caller order under that complete graph.
Every candidate body must already have completed ownership analysis before a
call site is accepted. Uncalled functions are still analyzed. A missing checked
candidate or incomplete ordering refuses rather than defaulting to a direct
callee or generic verifier.

At an indirect call the top value must be initialized FUNCTION with no owner,
reference mode or Result arm. I consume that callable in private staging only.
Every candidate receives a separate copy of the identical pre-argument flow
state, including locals, operand stack, regions, references, outstanding cleanup
and identity counters. I apply the existing checked File call transition with
the exact candidate declaration and all-NO_REFERENCE vector. The target query's
first profile permits no borrowed indirect formal; richer explicit reference
transport remains required later work. Argument consumption is checked for every
candidate and cannot consume an owner still held by a borrow.

Candidate transitions use fixed private scratch state. I reset scratch declaration
identity counters from the same snapshot before each candidate so an earlier
candidate cannot alter a later result. I canonicalize each successful result
with the existing owner/region/reference relation machinery, then compare all
observable locals/stack initialization, categories, Result arms, owner relations,
regions, formal references and cleanup obligations. The common call obligation
must agree on all fields except the original target identity. I retain all
candidate identities separately and never erase a target-specific obligation
merely because result counts match. If compatible canonical poststates or
obligations differ, this first composition returns UNRESOLVED; general joining
of distinct poststates is a later explicit precision extension.

Only after every candidate succeeds do I publish that variant's common output
and complete checked-target bitset into the private report. A failing candidate
publishes nothing and releases every report allocation. All ordinary File
service, Result refinement, cleanup, borrow and return checks remain active.
Every body must retain a reachable checked exit. No query executes host effects.

## My explicit work and storage bounds

I preserve target-query bounds, 16 alternatives per decoded ownership site,
4096 function instruction/variant pairs, 65536 total pairs and131072 edges.
Candidate applications add a separately counted maximum262144 across the whole
query; reaching it returns LIMIT before another application. This is an explicit
precision/work ceiling, not runtime fuel or a proof that every size-valid input
will fit. All scalar branches and initial zero-iteration paths participate.

The new composition has a32MiB total peak budget. I conservatively charge the
complete target-query storage bound, including its already-freed preparation
workspace, plus all second-plan/ownership-report allocations and fixed candidate
scratch. The ownership side retains its own16MiB ceiling, with added scratch and
per-variant call facts included before allocation. This overcounts the target
workspace rather than dropping it from a peak claim. Checked multiplication and
addition precede allocation. There is no per-candidate heap clone or allocation
inside the transfer loop. I do not raise any existing query's16MiB limit.

## My acceptance and remaining executable boundary

I review complete production before fixtures, then review fixtures before runs.
The synthetic corpus must include scalar and owned File/OpenResult arguments
and results, two compatible targets with distinct valid bodies, branch and late
backedge target discovery, an unused invalid candidate body, same-signature
borrow violations, owner already held by a caller borrow, missing cleanup,
Result-arm alternatives and zero-iteration uninitialized callables. Every
candidate must be checked; candidate order/permuted tables must preserve the
same semantic facts. Failure sentinels, input-destroyed lifetime, deterministic
bounds and all allocating preparation prefixes remain required. Original target,
cyclic, hosted and public refusal suites remain separate neighbors.

After this report qualifies, a separate hosted conjunction must combine it with
scalar/startup facts. Matched VM/native runtime transport must preserve exact
module/index identity, check candidate membership before frame transfer, carry
fuel through CALL/RET, unwind every failed candidate setup and retain first-error
cleanup. Native code must dispatch directly to generated functions, not an
interpreter or foreign-call fallback. Callable arguments/results, richer borrowed
calls, public grants/installed archives and paired source with full mandatory
shadows remain required within5.1. This report alone closes none of those gates.

## My preproduction review refinements

The independent cyclic-carrier review found no scoped design blocker. I copy
and reset the entire obligation array/count and cleanup state, not merely the
owner fields or next-identity counter. Equality compares semantic fields rather
than C padding. The emitted instruction fact records the original indirect-call
input stack including its callable; the candidate's direct-call transition sees
the staged stack after that callable is consumed. Output counts refer to the
actual post-call stack. I account for every additional scratch/fact allocation
before it occurs under both ownership and combined bounds.

## My first production checkpoint

I implement the separate report in `file_indirect_flow.h/.inc` within the existing
File flow translation unit. An internal cyclic extension adds the accounted
fixed scratch, complete callee graph and transfer callback; the old entry passes
NULL and keeps its original report/node/workspace sizes and opcode mode. I
check the second plan's original caller, instruction index and PC against the
owned first query before applying a candidate set. Copied component, nominal
layout and import getters complete the report without exposing its inner plan.

Every candidate starts with the same full transfer state and declaration identity
counter. I normalize only the call obligation's target field, compare canonical
owner/borrow relations and all obligation metadata, then retain every checked
candidate. My new scratch is charged before allocation within the existing
ownership16MiB ceiling and the combined32MiB bound. No transfer allocates.

This checkpoint is source for independent review. I have not compiled it or run
new fixtures; the qualification checkbox remains open.

## My fixture review checkpoint

The independent review of2bf4840fc found no execution blocker. I supplement its
successful two-candidate transfer with an explicit zero-allocation budget and
unchanged allocation count, alongside the application-limit refusal. Owned File
and OpenResult copied call/stack/type facts survive complete source destruction.
The corpus also retains caller/candidate borrow violations, unused-body cleanup,
late backedge candidates, both catalog permutations and target branch orderings.
I test selected invalid getters; I do not claim every accessor's complete index
space is exhaustively exercised. Production remains5687cf13c unchanged.

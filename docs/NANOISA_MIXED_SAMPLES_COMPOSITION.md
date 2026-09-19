# I compose mixed shape and ownership before execution

I record task_1bf5b051b828444c989382f8cc7702b6 under
`task_4be28fef163f42069064357639b3b5cc` at canonical47ad402f.
PR802 supplies my descriptive view; PR805 supplies my closed FLOAT-array shape
query. I reconciled only a2e469 from PR805's actual merge. The ownership,
managed-value and product parents remain open.

This contract proposes the next private composition checkpoint and states the
later public-admission dependencies. I change no production or execution route
until review. A private positive result is not executable authority.

## My unchanged source boundary

I retain the complete PREFIX and source of
`tests/test_owned_record_patterns.py::test_ordinary_inferred_field_and_alias`:
Handle owns one INT field; close consumes Handle; shadow close constructs and
consumes a Handle; Samples contains array<float>; main constructs Samples,
projects values, aliases it and compares `(at alias 1)` with2.5; shadow main calls
main. I keep every selected shadow in its original selection and order. I do not
split the file into separate owner and ordinary modules, omit close's shadow,
change source assertions, or declare the original positive case inapplicable.

My first composition retains the existing closed entry0, at most eight functions,
acyclic direct-value-call graph. All declarations and unused instruction bodies
are checked. Calls retain exact scalar or consuming scalar-leaf owner signatures;
ordinary managed parameters/results, CALL_REF, callbacks, links, external calls,
globals, initializers and captured values remain outside this mixed checkpoint.
I preserve the shape query's instruction/state/origin/work budgets. Unsupported
operations fail closed, including an unsupported operation beside an owner.

I separate ordinary Samples in a module with Handle from managed fields inside an
owner. Bundle containing Handle plus array<float> or STRING is not admitted by
proving an ordinary child. Native managed-owner fields have a separate owner and
contract; this work neither duplicates nor depends on their admission.

## My first checkpoint composes queries only

I propose an internal `nvm_analyze_mixed_samples` query. It borrows one immutable
module and internally constructs its own descriptive view and shape certificate.
It never trusts a caller-built or mutable certificate. It runs a separate complete
owner/scalar transfer analysis and returns an owned report only after every pass
succeeds. The report distinguishes checked shape, checked affine obligations,
scalar operations requiring runtime checks, and absent runtime admission.

The report retains original function/byte-PC positions, exact nominal mappings,
all required runtime scalar checks and the bounded maximum stack requirements.
Failure leaves the previous output unchanged and releases all temporary storage.
Malformed facts, unsupported forms, work limits and allocation failure remain
distinct. A positive report cannot be passed to an old selector as if it were an
ordinary NvmRecordPlan. No public verifier, source route, VM dispatch, native
emitter or shared authority query changes in this first checkpoint.

I do not call public nvm_verify from either query. The old managed-array analysis
calls public verification; it is not a composable foundation without an explicit
nonrecursive extraction. Each independent pass inspects every instruction and
validates operands, branch targets, stack counts, signatures and unused bodies.
Future shared structural factoring must retain all old acceptance and diagnostics
outside the new explicit composition route.

## My independent affine proof retains real categories

Today nvm_affine_state_create calls the old ownership validator and rejects
collection locals. affine_bytecode treats LOAD_LOCAL STRUCT as an owner
observation and STORE_LOCAL as scalar-only. These are real prerequisites, not
checks I can bypass with a successful shape report.

I propose a private checked facts-construction path fed only by the internally
validated immutable mixed view and complete positive shape result. It retains
all original layout/local indices and an explicit category for RESOURCE, ordinary
managed and unavailable facts. It does not clear resource flags, rewrite metadata,
substitute ordinary slots with authoritative VOID, or construct a filtered fake
module. The old public initializer and ownership validator keep their decisions.
Existing owner transitions may inspect only positively RESOURCE slots and rows;
generic STRUCT put/take must never gain authority over an ordinary record.

I reuse existing owner transitions only behind that category check. An independent
transfer pass checks live/dead ownership, exact nominal moves, complete constructor
fields, destructive consumption, observation roots, pending stack owners, call
actuals and helper exits. It requires source-ordered argument preparation, once-only
transfer and exact positional signatures. Every reaching join and loop backedge
retains exact owner/observation/region state; all normal exits consume required
owners. This checkpoint has no reference opcodes, so it proves their absence
rather than claiming a new mixed borrow contract. Existing reference profiles and
regression controls remain unchanged.

Ordinary values have their own tag/origin/initialization state. Their joins retain
all reaching alternatives, including uninitialized/VOID paths, independently of
exact owner-state joins. Ordinary copies and discards are not owner moves or
consumption. Every record/array transfer must agree with the complete shape facts;
no unknown branch is erased because another analysis knows a desired result type.
The affine pass cannot accept the shape query's opaque owner bookkeeping as proof
of consumption. It must reject deliberate owner leaks/copies/moves independently.

## My scalar obligations stay explicit

ARR_GET from a proved flat FLOAT array produces FLOAT|VOID, including for the
literal in Samples. I add no bounds proof, implicit cast, invented zero or changed
missing-index semantics. A typed F64 consumer retains a check for exact FLOAT at
that byte PC and operand position. Its abstract successful continuation may have
the opcode's result type, but a runtime type-error edge remains required.

The composition pass independently checks all scalar transfers. It distinguishes
statically exact inputs, supported dynamic FLOAT|VOID inputs requiring explicit
checks, and unsupported/unknown alternatives. It records checked runtime-error
edges rather than labelling every possible error a verifier failure. Only the
narrow FLOAT|VOID case is proposed; arbitrary unknown tags, owners, observations
or heap handles cannot satisfy typed F64 operations. Generic comparisons require
their actual VM tag semantics; I do not replace them with a typed predicate or
assume a matching tag absent at the emission position.

Public admission must later account for every required check with an audited VM
handler and native implementation. Both check tags before reading scalar union
storage and preserve the existing error category and cleanup. A negative/empty/
out-of-range read followed by typed F64 must report the ordinary checked error and
release live ordinary roots and owners. This is runtime semantics, not a claim
that shape or affine analysis proves the program cannot fail.

## My identities remain unchanged

The descriptive view keeps source per-kind STRUCT ordinal and original global
layout index, including RESOURCE, pending and unknown rows. A separate certified
ordinary map may assign compact managed ordinals only after positive per-field
proof. Its inverse retains global identity; bytecode operands never become compact
indices. Pending classes in the descriptive view remain pending. RESOURCE and
unknown rows never enter the ordinary map.

Every AGG_PACK/GET resolution checks source ordinal, global nominal identity,
field number and every receiver/field origin alternative. A same-shaped different
record cannot satisfy the declared child. ARRAY/NO_INDEX alone proves no element
type. Constructor and alias writes must retain the complete closed FLOAT origin
proof. The compact map must not reuse an all-record record_count bound for a
filtered table. Ordinary-only descriptor binding reuses qualified managed storage
only after proving this mapping; it does not mark Handle ordinary.

## My later public conjunction requires runtime support

After the private query is qualified, I require separately reviewed VM/native
checkpoints before selecting mixed execution. The final conjunction is structural
and instruction validity, descriptive transport validity, closed managed shape,
independent affine obligations, scalar checks accounted for by the implementation,
and an exact supported runtime/native profile. No one term substitutes for another.
Shared old validators keep their original scope; a new explicit composed route
must own the mixed metadata checks rather than turning old failure into fallback.
Every public API independently establishes the conjunction. Any invocation-local
reuse obeys the existing immutable-module/lifetime and callback/yield boundaries;
no persistent module trust or mutable certificate cache is introduced.

VM ordinary arrays/records use existing heap objects and exact tags, with roots
held through all allocating operations and frame transitions. Ordinary LOAD_LOCAL,
AGG_GET and alias copies retain managed identity while owner observations remain
separate. Stores release replaced ordinary roots only after the new value is safe;
POP releases an ordinary managed root. Owner cleanup still requires explicit
consumption on successful exits. Error unwind releases ordinary values and actual
owners exactly once, including prepared arguments and a failed helper activation.

Native mixed carriers require an explicit scalar/ordinary-handle/owner distinction.
I propose reusing qualified NmsRuntime flat arrays and ordinary records in one
invocation context shared by helper frames, retaining separate affine owner and
reference machinery. Every stack/local/temporary/pending/partial-constructor root
must have an explicit retain/move/release rule. Cleanup includes scratch operands,
not only the existing t/l/pending arrays. Frame roots die before their runtime
context; copy aliases and replacing stores retain before release. The finite
ordinary graph is acyclic; this is no general collector claim. Ordinary managed
call arguments/results remain refused, so STRUCT result does not silently change
from the existing owned-return contract.

Only after VM/native qualification do C-seed and both selfhost stages admit the
source subset. They keep complete source checks, exact descriptor classes/indices,
local metadata/name intervals, declaration-order evaluation and all selected
shadows. Failed lowering, verification or a false shadow preserves prior output.
Raw all-definition emission and selected publication have separately explicit
coverage; neither obtains an ordinary fallback after a mixed-route failure.

## My ordered acceptance

1. I qualify the private composition without executing pending modules: unchanged
   Samples/PREFIX-shaped complete graph, owner use in selected shadows, lower-index
   calls, exact maps, aliases, all receiver alternatives, repeated allocation-site
   weak unions and zero-iteration joins. I require independent refusals for owner
   leaks/copies/mismatched joins/nominals and unknown or uninitialized managed
   alternatives. Runtime F64 obligations remain visible. Fault sweeps reach success,
   preserve outputs and free temporary analyses. Existing authority APIs retain
   their decisions. I send this production checkpoint for review before tests.
2. I record and review VM/native lifecycle and checked-scalar changes separately.
   Their ordinary valid-module controls cover success, FLOAT|VOID checked errors,
   false assertions, helper failure, prepared values, aliases surviving replacement,
   all four public APIs, repeated invocation and bounded heap/native allocation
   faults. No refused module executes. Native sanitizers and exact live-allocation
   accounting qualify cleanup; fault coverage names actual allocator boundaries.
3. I review the public composed selector and paired source changes only after those
   runtime controls pass. I compile the unchanged original test with PREFIX and all
   shadows using C-seed, Stage1, Stage2 and NanoVirt, compare exact retained nominal/
   ownership/name facts and canonical outputs where contracted, verify artifacts,
   and execute VM/native pairs. A false close shadow and a false main shadow each
   block publication. Missing/unknown constructor fields and wrong nominal actuals
   remain checked refusals with prior output retained. Mixed Bundle fields remain
   outside this child and are not reclassified as ordinary to pass these gates.

I keep historical950f and every earlier qualified artifact immutable. Completion
of this private composition child does not close public Samples execution or any
full ownership, managed-value or release parent. The later checkpoints require
explicit ledger/roadmap contracts before their production edits.

## My approved first implementation boundary

I keep the checked facts constructor static inside affine_state.c's private
composition include; no caller can request an unchecked public state. The private
walker explicitly guards every reused record-fields, pack/unpack, put/take, exit
and observation transition by original RESOURCE classification. Ordinary locals
keep their original descriptors and a separate ordinary value state; they are
never rewritten as VOID or passed into owner transitions.

My first generic EQ/NE accepts exact matching scalar tags and the explicit
FLOAT-or-VOID read alternatives with existing value equality semantics. Generic
ordering requires exact matching supported non-VOID scalar tags. Mixed numeric
tags and possible-VOID ordering remain unresolved. Neither owner tokens nor
ordinary heap handles participate in generic comparisons. This is deliberately
narrower than the general VM, without changing its comparison behavior.

## My first production checkpoint

I compile `mixed_samples.inc` inside affine_state.c solely to keep its checked
Facts constructor static. The existing public state constructor and all existing
transition bodies are unchanged. My new header exposes only a report/free query,
not the private state. The report owns the earlier shape report and a separate
certified ordinary identity map; `runtime_admitted` stays false.

| Reused operation | My private call boundary |
| --- | --- |
| state clone/initialization meet | I clone one function's real facts; ordinary local values have their own state, owner liveness and stack observations join exactly. |
| record_fields | I first require the exact global RESOURCE row with only INT/BOOL/U8 leaves; pending/ordinary rows cannot reach it. |
| OWN_PACK | I use that checked field list and exact scalar operands, then create one stack owner token. The local-normalized pack API is not used. |
| OWN_UNPACK_LOCAL/take_local | I first require a RESOURCE local, no outstanding stack observation and a live exact owner, then publish each checked scalar field. The local-normalized unpack API is not used. |
| put_local | I require a RESOURCE destination and an exact matching stack owner; the existing availability check prevents overwriting a live owner. |
| local_info/scalar_field | I invoke these only for a positively RESOURCE local/observation with matching original nominal identity. Ordinary projections use all closed origin alternatives instead. |
| scalar_define | I call it only for declared numeric/Boolean scalar locals; ordinary ARRAY/STRUCT initialization never uses it. |
| can_exit_type | I first check the exact declared result/count and the stack token category; the shared exit check then independently rejects every remaining live resource local. |
| calls | I inspect every exact positional signature before consuming any prepared stack value; every callee is independently analyzed, including unused helpers. |

My concrete opcode inventory is PUSH_I64/F64/U8/BOOL/VOID; NOP/DUP/POP/SWAP/ROT3;
LOAD/STORE_LOCAL; I64_ADD/SUB/MUL/DIV_S/REM_S/NEG and comparisons;
F64_ADD/SUB/MUL/DIV/NEG and comparisons; BOOL_AND/OR/NOT; scalar EQ/NE/LT/LE/GT/GE;
JMP/JMP_TRUE/JMP_FALSE; CALL/RET/ASSERT; OWN_MOVE_LOCAL/STORE_LOCAL/PACK/UNPACK_LOCAL;
ordinary STRUCT AGG_PACK/AGG_GET; and FLOAT ARR_NEW/LITERAL/PUSH/SET/GET/LEN.
Every other opcode is unresolved. The earlier positive shape query and the new
independent pass must both accept the entire graph; an absent transition is not
an invitation to ordinary execution.

I retain a separate bounded deduplicated worklist with262,144 visits, at most
4,096 instructions,256 locals/stack slots and1,048,576 stored value cells. I cap
scalar records at two per instruction, retaining each operand's actual tags and
policy. These records identify typed FLOAT-or-VOID checks and generic value
comparison semantics; they do not claim to enumerate ordinary division-by-zero,
allocation or other existing runtime failure conditions. My first checkpoint has
not been compiled or executed; I hold fixtures and gates for independent review.

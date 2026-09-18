# My owned and void value-call result contract

I track `task_a2797653878d4acba42423a689909ebf` as the next runtime prerequisite under
`task_c4351c720aee424ea9b90187e51a08f2`, after canonical PR725.

I extend only the reviewed acyclic value-only owned graph after its canonical
merge. I retain at most eight functions/frames and 0..8 exact mode-zero value
parameters. Entry0 still returns one INT/BOOL/U8; other functions may return
one exact complete scalar-leaf resource STRUCT, one existing scalar, or zero values for
VOID. My borrowed CALL_REF profile remains unchanged. Strings, printing,
reference/value mixtures, recursion, external calls and source admission stay
outside this prerequisite.

I do not introduce new resource-field shapes; nested return paths mean nested
calls/forwarding with existing complete scalar-leaf returned layouts. Existing
parameter layouts remain subject to their prior rules.

My ownership result descriptor supplies the exact nominal/layout identity;
the ordinary result tag alone does not authorize a returned owner. My existing
ownership schema already carries the result tag/mode/layout descriptor, and my
function signature carries result count/tag. I first validate that both formats
already represent each admitted result without a wire revision, then require
binary/text roundtrip equality; if they do not, I split a transport prerequisite. I expose a
checked descriptor query and validate its consistency with function result
count/tag. Every return path has exactly the declared stack result count. A
resource result must be one owned stack token obtained by explicit construction
or move; observations and references cannot escape. Every other local owner is
consumed, all regions end, and reaching-edge facts remain exact. Void returns
require an empty operand stack and the same complete local consumption.

At CALL I retain whole-argument preflight and once-only transfer. Analysis
pushes the declared exact owned token, scalar, or nothing, only after verifying
the callee's complete contract. A returned owner must be consumed or moved by
the caller under existing rules; it cannot be silently discarded, duplicated,
reinterpreted by same-shaped nominal layout or overwritten.

My VM separates the returned value from callee-local cleanup, validates exact
runtime result tag and resource layout before caller publication, and retains
that value as an actual cleanup root if any subsequent step can fail. The
existing scalar return path remains unchanged where possible. I audit caller
result capacity and error unwinding explicitly rather than assuming a popped
frame guarantees a safe publication path.

My native helper ABI returns status separately from a full nown_value carrier.
All incoming carriers are cleared before an output can overlap them. On RET I
move the result to a dedicated temporary and clear its former stack carrier;
I drain remaining actual roots and publish the temporary only on successful
completion. On failure I release any pending result exactly once and do not
write a successful result to the caller. Void calls supply no stack operand
and no fabricated scalar. Entry's external scalar ABI remains intact.

My normal acceptance includes zero-argument owner construction, owned identity
and nested call/forwarding return paths, chained return/forward/consume paths, a void consumer,
scalar callers retaining unrelated holds across returned-owner allocation, conditional early returns and exact
nominal differences. I compare all four VM APIs and native sanitized output
with repeated entry and explicit owner allocation failures before/after return
preparation. Static controls retain refusal of missing/extra results, live
unconsumed locals, observation/reference escape, wrong nominal results and
ignored returned owners. Existing scalar graph and borrowed gates remain
required. These tests use corrected ordinary modules; I do not replay frozen
product failures or change the affine example.

## My inspected descriptor and transport boundary

I base this contract on canonical11d64547, containing reviewed PR725 and the
independent managed-record foundation. I do not change the managed backend
admission rules in this slice.

My current ownership formats1 and2 already encode each function as local count
u16, parameter count u16, then one result descriptor and the local descriptors.
Each descriptor is tag u8, mode u8, reserved u16 and layout index u32.
`ownership_contracts.c` already requires a zero result mode, zero reserved bits,
valid exact layout indexes, and agreement with the function's result tag (or
VOID for zero results). Resource layout flags establish the ownership boundary;
shape equality alone does not establish nominal identity. My affine Facts
already retain this descriptor. `nvm_affine_can_exit_type` already checks an
exact moved stack owner and absence of other live resource locals; the scalar
exit check already represents VOID. I reuse these obligations instead of
inventing a weaker return rule.

I add a checked result-descriptor query with unchanged outputs on failure,
using the existing exact declaration facts/transport. It distinguishes VOID,
INT, BOOL, U8 and an owned STRUCT with complete/resource flags and only
INT/BOOL/U8 leaf fields. Returned nested-resource fields remain refused;
existing nested parameter/local shapes keep their current independent rules.
My function entry must report result_count0/result_tagVOID for VOID and
result_count1 with the exact declared tag otherwise. Entry0 remains one scalar.
The complete graph checker, runtime admission and native stack-effect analysis
must agree with this descriptor; CALL pushes zero or one actual result.

No opcode, tag, ownership version, feature bit or binary section changes are
needed. I require exact ownership/layout bytes and result count/tag after
execution-module to NVMv2 serialization/deserialization and canonical NASM
`.ownership`/`.layouts` disassembly/reassembly. I also preserve canonical text
on a second roundtrip. Earlier execution consumers continue to refuse these
result profiles through their existing scalar-only admission checks; transport
alone never authorizes execution. If these representations fail an ordinary
roundtrip, I record a separate transport prerequisite before widening runtime
admission.

## My publication and lifetime boundary

At VM return I validate count/tag and, for an owned result, non-null record,
exact VmStruct.def_idx and declared field count before removing it from the
callee stack. Invalid or unavailable declaration facts fail while all values
remain rooted for existing terminal cleanup. I reserve/prove the final caller
stack capacity before destructive frame cleanup and handle any failure before
publication. A moved pending result remains owned by exactly one internal
cleanup location. I audit both explicit RET and their common completion path;
affine verification continues requiring explicit RET. I do not alter ordinary
non-owned return behavior merely to admit this profile.

My native value-graph helper result changes from int64_t output to nown_value
output. Entry's external int64_t ABI and borrowed helper ABI stay unchanged.
A private pending carrier owns a result until callee cleanup succeeds. Exact
native result-layout checks require retaining the constructor's layout identity
in nown_record (an internal generated-C representation, not a wire change).
All owned constructors initialize it; scalar carriers remain unchanged. I clear
all incoming argument carriers before publishing into any overlapping output
slot, move and clear the result stack carrier before cleanup, release a pending
result on error, and clear the pending carrier after successful transfer.
VOID contributes no stack operand and never writes a fabricated scalar result.

An unrelated caller hold remains valid across returned-owner allocation and
callee cleanup: its root, frame origin and invocation generation do not change.
A returned owner carries no borrowed descriptor or callee reference context.
Each callee still ends its regions and clears its own reference context at
return. Reusing a frame/local slot in a sibling call still creates a fresh
generation. I retain PR725's public-invocation proof limits; new return checks
are live checks and are never skipped by that proof.

## My acceptance and sequencing

I first qualify exact descriptor/query and ordinary wire/text roundtrips, then
shared verifier/VM/native execution in one bounded runtime PR. Source producers
remain guarded until a separate reviewed admission change. I require:

1. Owner factories, consuming identity/forwarding helpers, an owned result
   forwarded through several graph depths, and a VOID consumer, including
   interleaved scalar/resource parameters and an eight-frame route.
2. Exact returned nominal identity for two distinct same-shaped layouts;
   static refusals for wrong/missing/extra results, reference/observation escape,
   unconsumed locals, discarded/duplicated/overwritten result owners, unsupported
   returned nested fields, and owned/void entry0. I never execute refused modules.
3. Caller-local observations before and after a disjoint returned-owner
   allocation, repeated sibling calls, balanced regions, explicit consumption,
   true/false assertions after result preparation, and conditional early returns.
4. All four public VM APIs, repeated entry, exact scalar final results,
   generation/context/stack/root cleanup, generated-native strict GCC/Clang
   execution and ASan/UBSan/LSan. Owner allocation faults and any separately
   injectable declaration/frame-reservation faults are reported by actual
   failure site rather than claimed as exhaustive allocation coverage.
5. Existing scalar graph, consuming, borrowed CALL_REF and authority gates.
   New fully instrumented qualification has explicit bounds and retains every
   case/assertion; any incomplete attempt is recorded independently.

My full c435 example, string/PRINT requirements, mixed borrowed/value graphs,
owned returns from the public entry, recursive graphs, source admission and
full normative ownership acceptance remain open. I do not modify or rerun any
preserved failing product artifact to qualify this prerequisite.

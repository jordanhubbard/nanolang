# I bind File flow facts to decoded instructions

I record `task_546cf4241d4050fae080fd25b5602b8c` under72556/6931, dependent
on the private transfer-state foundation673932/PR839. This is a preimplementation
contract. I introduce no opcode, decoder change, profile admission or execution
in this commit. Each production stage below requires independent review.

My first milestone covers bounded acyclic CODE/CFG and direct call graphs only.
Loops, indirect calls, matched VM/native cleanup and complete paired source plus
all selected shadows remain required work on the parent. They are not waived
release criteria. The current successful state API calls are not certificates.

## My encoding audit and proposed boundary

I inspected `spec/nanoisa.yaml`, `src/nanoisa/isa.h`, the generated
`src/nanoisa/generated_schema.h`, its generator, assembler/disassembler and
`src_nano/nanoisa*` registries at the current preimplementation base. Primary
bytes0x91 through0x96 are unassigned in the normative inventory and active enum
and have no independent producer spelling. The active C metadata is generated
from `legacy_opcodes`; assembler/disassembler resolve through that metadata.
I recheck all those inventories before the implementation commit so concurrent
ISA work cannot silently reuse a proposed byte.

The extension plane is empty. `DecodedInstruction.opcode` remains uint8 and has
no independent extension-plane identity. I do not pretend that metadata alone
makes this a ready alternative or change that representation in this slice.
I propose these six unused primary identifiers, with exact LE operands:

| Proposed byte/name | Operands | Exact logical effect |
| --- | --- | --- |
|0x91 FILE_SERVICE|U32 import, U16 reference|Resolve exact five-method nominal import. Temp uses no stack input; close consumes one File; exclusive write consumes one INT and a live reference; rewind/read use only a live reference. Push one exact Result. Nonborrow methods require reference65535.|
|0x92 FILE_RESULT_BRANCH|U16 local, I32 error displacement|Observe an initialized exact Result local without copying it. Fallthrough is Ok; opcode-PC-relative target is Error. Publish separately refined successor states.|
|0x93 FILE_RESULT_TAKE|U16 local, U8 arm|Arm0 means Ok; arm1 means Error. Consume an exactly matching refined Result local and push its exact catalog payload. Other arm bytes refuse.|
|0x94 FILE_DROP_LOCAL|U16 local|Consume one unheld File/OpenResult local and record required cleanup.|
|0x95 FILE_DROP_STACK|none|Consume one File/OpenResult stack owner and record required cleanup.|
|0x96 FILE_END_BORROW|U16 reference|End a local exclusive File reference; borrowed formals cannot be ended by the callee.|

These names describe the exact File catalog, not arbitrary services/resources.
The branch creates no ordinary BOOL/int fact and generic equality never refines
an owner. Stack metadata is dynamic only where actual catalog method effects
require it. The private verifier derives those effects itself.

I retain ISA schema version2 and existing primary encodings. Existing required
service bit9 and the exact nominal120-byte version2 metadata remain necessary;
required bits1/7/8/9 and cross-section checks do not change. No new service
payload version or import-kind reinterpretation follows from these opcodes.
Old tools reject unknown primary bytes, and old executable consumers already
refuse service claims. New raw decoding must not weaken either fact. New File
opcodes without a service claim also refuse at every execution/conversion route.
Version1 service metadata remains non-executable. An in-memory NvmModule lacks
a serialized feature envelope; a private query does not claim that envelope was
validated. Later public entry must independently validate the actual container.

My first encoding checkpoint updates enum/schema/generated metadata together,
exact encoder/decoder/text roundtrip support and explicit non-admitting consumer
guards. I audit generic verification, owned/mixed/managed selection, direct VM
entry, native/LLVM/Wasm emitters, reconstruction, facade and wrapper paths for
known-opcode default acceptance or lossy conversion. No raw decoder or assembler
success is runtime authority. Fixtures that construct unverified modules use the
established unverified assembly API explicitly, then test public refusal.

## My private analysis input and output

I accept the actual immutable NvmModule, never a caller list of logical sites,
accepted body summaries, desired rights, alternative catalog or a permission
boolean. I prepare the qualified exact nominal/declaration object, then decode
all function CODE buffers with checked offsets and instruction boundaries.
Logical sites are actual `(function index, byte offset)` identities. The existing
per-function state site ID is that checked byte offset, not a caller's arbitrary
number. An import/call event at the site must match the decoded opcode/operands.

The query owns its analysis result, including copied code and exact relevant
nominal/function/local facts. Output is published only after complete success;
failure preserves the caller's sentinel and frees every partial allocation.
No retained caller buffers or mutable certificate survive the query. A later
execution selector must consume its own freshly prepared result for that same
immutable module, not accept an externally supplied opaque pointer as permission.

I distinguish INVALID malformed encoding/indices/contradictory declarations,
UNRESOLVED unsupported operations/shapes/cycles/joins, LIMIT checked budgets,
and MEMORY allocation failure. I do not infer MEMORY from an ambiguous legacy
boolean error. I use the allocation-free scalar decoder where possible and
stage my own bounded graph/state arrays; any reused allocating decoder needs a
reviewed status-preserving boundary first.

## My bounded instruction transfer table

Every function uses its exact retained parameter/local/result declarations;
nonparameters start uninitialized. I support the existing exact INT/BOOL/VOID
push/load/store/copy/drop operations, NOP, exact integer predicates/arithmetic,
BOOL predicates, ASSERT, ordinary conditional/unconditional branches and RET.
I enumerate the actual supported scalar opcodes in production review; an
unmodeled transfer is UNRESOLVED, never an inferred VOID or generic scalar.
Scalar transfers do not discharge host byte-domain or rights checks.

OWN_MOVE_LOCAL means the established local-to-stack transfer; OWN_STORE_LOCAL
means stack-to-empty-local transfer. I map their existing operand semantics to
the qualified state operations without changing other profiles. REGION_BEGIN,
REGION_END and BORROW_LOCAL_EXCLUSIVE retain existing operand order
(reference slot, local). Local borrow epochs never become copyable values.
Shared/path/reborrow operations and reference field reads/writes remain outside
this first File family.

AGG_PACK/UNION_CONSTRUCT and scalar projections are accepted only for exact
copyable FileError, ReadByte and scalar Result constructors/fields. I map existing
source/per-kind indices through the nominal plan before using a global identity.
No structural tag or matching field shape can manufacture File/OpenResult.
Generic UNION_TAG/FIELD or AGG_TAG/GET cannot observe or extract an affine Result.
New explicit branch/take operations carry that authority.

CALL requires mode0 parameters and consumes the exact ordered stack suffix.
CALL_REF supports exactly one exclusive File formal; its explicit reference slot
fills that formal and remaining mode0 arguments consume the ordered value suffix.
Zero/multiple borrowed formals under CALL_REF are unresolved in this milestone;
the richer private state API is not an undocumented wire encoding. Borrowed
formals remain caller-owned, cannot escape a result or be closed/dropped, and
cannot alias another exclusive actual. Owned File/OpenResult helper arguments
and results preserve the logical linear obligations already reviewed.

Globals, upvalues/captures, callbacks, CALL_MODULE, CALL_EXTERN, tail calls,
CALL_INDIRECT, FUNCREF/closures, foreign dispatch, noncatalog shapes and unsupported
opcodes are unresolved. FILE_SERVICE is the sole proposed exact service opcode;
I do not reuse generic RPC/FFI CALL_EXTERN for verified File ownership.

## My CFG and complete body checks

I validate every function extent, decode every byte exactly once, reject truncated
operands and require every branch target/fallthrough to name an instruction in
the same function. Relative displacements are based on the opcode PC, with widened
signed arithmetic checked before indexing. No branch may land inside an operand,
leave a function, or fall off its end. RET has no fallthrough.

I build the CFG before state propagation and reject any syntactic cycle in this
first milestone, even if a constant condition might make it unreachable. I use
topological processing, not an iterative algorithm falsely claiming convergence
from fresh symbolic IDs. Shared predecessors clone one entry family; joins retain
exact owner/reference placement and compatible initialized facts, widen only
Result arms and union compatible site obligations. Independently allocated owners
cannot acquire matching IDs merely because their shapes/sites look similar.
An incompatible live-owner join remains explicitly unresolved.

Both branches of an unknown Result or ordinary BOOL condition are checked.
Known opposite Result arms can be unreachable, but all bytes still receive
structural decoding, operand bounds and supported-opcode checks. Every reachable
exit satisfies exact result count/type, no leftover stack values/owned locals,
no local borrow/region escapes and preserved borrowed formal ownership.
Owner drops record cleanup even when the function later exits successfully.

I inspect every declared function, including uncalled helpers and selected shadow
functions already present in the module. I do not partition or omit inconvenient
functions. I derive all direct edges from CODE, reject self or mutual recursion,
and reject indirect calls rather than inventing their targets. Callees are
analyzed in reverse topological order. A call-site body obligation is discharged
only by the complete actual callee result, never its signature alone. Unreachable
calls still require bounded valid callee identity and belong to the conservative
cycle inventory. The public entry boundary remains scalar-only and nonescaping;
internal owner returns do not authorize File/Result escape from an invocation.

## My obligations and allocation bounds

A private successful result distinguishes discharged structural/type/owner/body
facts from pending checked-runtime obligations at each actual site. Binding,
invocation/liveness, rights, byte range, borrow epoch, result publication and
first/secondary cleanup status remain explicit runtime requirements. Static
success does not grant host authority or change close's consumed-on-error rule.
No obligation disappears at a branch join, helper call or successful RET.

I retain64 functions,256 locals/stack/owners/references and256 distinct pending
sites per function from the state API. I additionally bound total CODE to64KiB,
256 decoded instructions per function and4096 per module. I check code products,
graph arrays, copied declarations and all simultaneously live cloned states
against a16MiB total analysis-owned storage budget before each allocation.
If exact opaque state size is needed, I add a private size query using the actual
state extent rather than duplicating a guessed struct size. Reaching an instruction
count limit does not promise it fits the byte budget. Failure remains atomic.

## My staged acceptance and required follow-ons

1. Review exact schema/encoding/decoder plus all-consumer refusal production.
   Qualify golden LE bytes, every truncated operand, index/displacement extremes,
   exact text roundtrip, old unknown-opcode behavior, absent/malformed service
   claims and ordinary-neighbor verdicts. No service instruction executes.
2. Review the private complete CODE/CFG query before fixtures. Qualify actual
   bytecode for both service Result outcomes, internal owned/borrowed helper
   flows, exact entry/local/result identities, branches, multiple predecessors,
   uncalled invalid helpers, self/mutual recursion, mid-operand/out-of-function
   branches, stale refinements, borrowed escapes, owner loss/duplication, all
   allocation prefixes and unchanged failure outputs. Boundaries are tested on
   Linux/Darwin normal and strict sanitizer builds with actual provider closure.
3. Preserve parent requirements for backedges/loops (including repeated bounded-live
   acquire/close and cleanup), indirect-call target/body coverage and multi-borrow
   call encoding where needed by the complete source boundary. Each needs an
   explicit later reviewed contract; none is accepted by this acyclic milestone.
4. Review the matched VM/native lifetime, host-grant and cleanup conjunction before
   public execution. It must consume all pending site obligations, preserve first
   and secondary errors, and refuse public File/owned Result escape. Paired source
   bindings, generated ownership metadata and complete original source/shadows
   follow with actual temporary-file lifecycle acceptance. These remain72556/6931,
   d03c/ed702 and full-release obligations after the private query lands.

## My first-checkpoint implementation audit

Before code I distinguish metadata presence from execution refusal. I add an
allocation-free instruction-family scan plus a separate pending-execution query;
I do not redefine `service_bindings_present` or infer required feature bits from
arbitrary immediate bytes. I scan declared function instruction boundaries and
recognize a new opcode before reading its operands, so a truncated new opcode
still refuses. Malformed ranges/earlier undecodable bytes remain the existing
structural decoder's responsibility, never positive File authority. Metadata
validation requires exact nominal version2 when a new instruction is present;
bare/v1 instruction modules cannot pass the non-executing bridge/container path.

Module disassembly/reconstruction that drops service authority continues refusing.
Raw instruction/function disassembly without a module remains available for exact
text controls and conveys no ownership metadata. Its common branch formatting
currently adds signed32 offsets directly; I record that static arithmetic finding
before widening target arithmetic for this checkpoint. I execute no pre-fix
boundary fixture. Raw `isa_decode` retains its existing partial-output-on-truncation
contract; staged immutable query outputs are a separate later API guarantee.

I keep the legacy serializer's new instruction scan in the format/ISA providers,
not the full service catalog library. Its standalone pool-allocation fixture now
links the ISA provider explicitly; module manifests already include ISA. A separate
existing `forth_see` manifest gap is recorded on the roadmap before any repair.
The current checkpoint changes no Forth manifest or service execution path.

## My wrapper-publication prerequisite

During static fixture preparation after06120, I found that both wrapper APIs
enter `build_wrapper` without checking the supplied module or serialized blob
before staging and linking. I record task89b8 before correction. This is an
unsupported publication path, not demonstrated host service execution: generated
runtime loaders and execution guards remain independent. I ran no such wrapper.

I require a guard before path/object discovery or output staging. For the normal
API I check the supplied module's pending service/File claim. For both APIs I
load the actual embedded bytes through my existing version-aware loader, refuse
load failure or a pending service/File claim, and free the temporary module on
every outcome. A clean supplied module cannot hide a claimed blob; a clean blob
cannot hide a claimed supplied module. Daemon generation has only the blob. I do
not introduce a new parser, relax the retired-v1 boundary, or grant execution
from successful decoding. Existing ordinary wrappers must retain their outcomes.

My fixtures require preserved destination contents and absence of staging for
normal/daemon rejection, including malformed bytes and each mismatched clean/
claimed representation. Any early allocation failure refuses before publication.
I qualify this correction only after independent source review.

## My next reviewable preparation boundary

After actual PR841 merge I first prepare an independently owned, non-certifying
CODE plan. I retain the copied private declarations, copy CODE, decode all
function extents and retain exact instructions, source-to-catalog constructor
identities, successor indices and topological orders. I reject overlapping,
unclaimed or empty function extents, so every CODE byte belongs to exactly one
function. Function-table order need not equal CODE order. The public entry has
only mode0 INT/BOOL/VOID parameters and scalar/no result; internal declarations
retain the existing exact owner/borrow facts.

I check operand-local/callee/reference/constructor/import identities even in
unreachable instructions, and reject syntactic CFG and direct-call cycles.
Preparation does not check stack types, owner/refinement/exit flow, branch
reachability, or discharge any body/runtime obligation. Its successful status
must never be consumed as an admission certificate. A dependent reviewed state
propagation checkpoint performs those checks before private acceptance.

My exact preparation inventory is NOP; PUSH_I64/BOOL/VOID; DUP/POP;
LOAD_LOCAL/STORE_LOCAL; OWN_MOVE_LOCAL/OWN_STORE_LOCAL; REGION_BEGIN/END;
BORROW_LOCAL_EXCLUSIVE; the six File opcodes; CALL/CALL_REF; JMP/JMP_TRUE/JMP_FALSE;
RET; ASSERT; ADD/SUB/MUL/DIV/MOD/NEG and I64_ADD/SUB/MUL/DIV_S/REM_S/NEG;
EQ/NE/LT/LE/GT/GE and I64_EQ/NE/LT_S/LE_S/GT_S/GE_S; AND/OR/NOT;
AGG_PACK/UNION_CONSTRUCT; AGG_GET/UNION_FIELD; and AGG_TAG/UNION_TAG.
Generic scalar operators will require exact scalar facts during transfer;
constructor and projection shape facts are not inferred from their opcodes.
Unlisted operations are UNRESOLVED, including SWAP/ROT3 and indirect calls.

I implement this private plan in an include owned by the existing file_flow
translation unit, where actual declaration/state extents and the immutable
nominal map are available. I check the worst-case nominal allocation plus exact
declaration, plan, CODE and instruction allocations and bounded preparation
scratch against16MiB before allocating. Later cloned state peaks must separately
use the actual state extent and remaining budget; no future flow budget is
claimed from preparation. Every failure frees partial ownership and preserves
caller output. No public verifier, converter, VM or backend calls this API.

## My logical entry is not a hosted entry contract

The private plan treats `header.entry_point` as a bounded logical root with
scalar-only declarations. It does not require HAS_MAIN, choose an invocation
ABI, or interpret function names as initializer dispatch. Every declared
function remains in its decoding/cycle inventory, including names such as
`__init__`, but no implicit host call edge is claimed.

The existing VM's `vm_execute` first requires HAS_MAIN, then invokes the first
function named `__init__` with no arguments, then invokes the entry with no
arguments. Existing closed backend verification separately requires a zero-arg
integer/bool entry and a zero-arg initializer. A future File public conjunction
must explicitly validate flags, entry invocation signature, initializer identity,
selection/order and cleanup/results against the actual module before dispatch.
It cannot infer those facts from this private plan, or lose an initializer's
owned result because ordinary startup ignores it. Logical helper returns and
whole-function decoding grant neither public escape nor hosted authority.

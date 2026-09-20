# I execute checked File loops with finite fuel

I propose task_15a92c930af7433e9e25b41c7c5c761f under72556/6931 after
PR894 (`5d1d3d39ffff55035440074680437e74af915acd`). My
[cyclic query](NANOISA_FILE_CYCLIC_QUERY.md) is qualified and non-admitting.
This contract requests review before implementation. No cyclic service executes
under this document alone. My public acyclic dfa149 lane remains independently
owned; I preserve its byte APIs, native semantic ABI1 and runtime report.

## I consume the whole checked graph

I add a distinct opaque `NvmFileCyclicHostedPlan`, prepared from immutable
serialized v2 bytes by `nvm_file_cyclic_hosted_prepare(bytes,size,out)`. It returns
existing File flow statuses, owns every retained fact and leaves output unchanged
on failure. It retains the current16MiB input ceiling and64MiB hosted ceiling.
I reuse syntactic envelope/codec checks only with identical old outcomes; I do
not cast a cyclic report into an acyclic body report or choose variant0 as a
substitute for all alternatives. No failure falls back to an old plan.

Preparation runs the complete cyclic query on the exact decoded module, checks
copied CODE equality and original nominal/global/import identities, and applies
the existing initializer/entry rules: zero-argument VOID initializer, then
zero-argument INT/BOOL entry. Unsupported globals, links, callbacks, passive
sections, opcodes and recursive calls retain their refusal. Every function and
unused instruction receives structural and target coverage checks. The query's
`runtime_admitted=false` remains unchanged; only the distinct composed plan
records matched target coverage after that checkpoint is implemented.

Copying accessors expose startup, functions, locals, instructions and ALL
variant facts/edges, plus original type/import maps. Preparation validates edge
indices and the complete pending/discharged check inventory independently of
the dispatcher. It preserves malformed versus unsupported versus limit outcomes;
legacy allocating codec failures that do not distinguish OOM remain UNRESOLVED,
not asserted MEMORY. Allocation of my own data reports MEMORY atomically.

For each function I take operand/reference/region/root maxima over all input
and output variants. I compute staging and call storage over every call variant
in the existing callee-first nonrecursive order. A loop reuses one frame; it
never multiplies storage by iteration count. Declared max_stack is checked
against the maximum, not the first alternative. I retain64 functions/frames,
256 locals/operands/references/live owners and the existing instruction/byte
bounds,16 alternatives and query work limits. I account copied envelope, bridge,
query peak/report, plan tables and transient overlap before allocation under
64MiB; runtime carrier storage remains separately bounded by64MiB. I publish
separate phase and simultaneous-overlap bounds, not a misleading single64MiB
whole-process claim. No allocation scales with an execution fuel value.

## I keep abstract relations separate from physical lifetimes

The carrier has an explicit acyclic/cyclic plan kind. Existing creation and
getters retain acyclic behavior; a cyclic plan cannot escape through the old
single-fact plan getter. A private cyclic creation path borrows no input bytes.
Shared factoring must retain the original acyclic corpus and ABI1 behavior.
The source checkpoint supplies exact structs/accessors before executable changes.

Each active cyclic frame records original function/instruction and a checked
input variant ordinal. Entry selects the query's retained parameter entry state,
not an arbitrary compatible state. At each boundary I validate initialized
local/operand categories, original nominal IDs, actual Result arm compatibility,
owner uniqueness, reference origins/formal anchors and ordered region membership
against that variant. An abstract UNKNOWN Result arm permits either valid
runtime arm; it is not an initialized unknown value. Empty abstract slots must
be physically empty. Frame staging must be empty outside atomic transfers.

I build a bounded temporary bijection from the report's canonical owner labels
to actual live carrier owners, including forwarded formal anchors. This witness
is internal and exposes no raw File token through the existing public view.
Every reference checks its actual origin, physical generation, invocation and
borrow epoch as well as the canonical relation. I never rewrite a physical
generation or epoch to the query's canonical IDs. Unknown/dangling/duplicate
relations refuse; equal slot numbers alone do not establish identity.

The selected decoded successor and the current variant determine the exact next
variant through the report's edge table. An absent refined edge refuses. I
validate the resulting physical state against that exact target; I do not search
for an unrelated alternative that happens to fit. Scalar branch predicates and
Result arms still come from actual values. Two abstract alternatives with the
same counts but different ownership/initialization remain distinct.

A CALL validates and stages arguments before moving ownership, suspends its
caller at the exact variant and enters the callee's parameter state. A RET
validates its variant, scalar/nominal result and every owner/reference/region
exit obligation before publishing to the caller. Only then does the caller
advance through its recorded successor and revalidate its restored state.
Caller suffix, staged prefix and pending return remain roots throughout failure.
CALL_REF retains exactly one borrowed formal; forwarded aliases never own the
originating epoch. No indirect or recursive call is added.

Repeated acquisition must recycle released physical slots while generations
remain monotonic and checked. More than64 total acquisitions is required with
bounded simultaneous live roots; reaching the physical live-slot limit remains
a limit failure. Generation/region/epoch overflow refuses without wrap. The
checkpoint must audit the existing host/core reuse logic rather than assuming
query canonicalization solves runtime exhaustion.

## I charge one shared invocation budget

I propose private revision1 types, separate from existing runtime/public structs:

```c
typedef struct { uint32_t revision; uint64_t instruction_limit; }
    NvmFileCyclicOptions;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmFileCyclicExecutionReport;
```

I propose the macro-private VM symbol `nvm_file_vm_cyclic_execute(bytes,size,
options,out)` returning `NvmFileCyclicExecutionReport`, and private emitter
`nvm2c_file_cyclic_private_emit(bytes,size,out,err,err_size)` returning existing
`NvmFileRuntimeStatus`. Its generated `nvm_file_native_cyclic_execute(options,out)`
returns the same cyclic execution report. Emission has no host effects and does
not bake an unchangeable fuel value into a program. The distinct runtime ABI
query `nvm_file_runtime_cyclic_abi(revision,options_size,report_size,view_size,
frame_size)` checks the implementation, not just the generated header. These
names are private checkpoint proposals, not installed API commitments.

The options pointer is required, revision must equal1, limit ranges0..1000000.
A convenience initializer supplies100000; there is no silent unlimited mode.
Invalid options refuse before context acquisition. The private VM entry and
generated native entry take identical options and disjoint scalar output.
A later public API choice requires separate review after dfa149 lands.

One context-owned budget covers initializer, entry and all helper frames.
Immediately before each decoded instruction's effects, after checking its site
and variant, I compare started with limit. Equality records LIMIT at that
instruction's original function/byte offset, sets fuel_exhausted and performs
no effects from that instruction. Otherwise I increment once and execute it.
NOP, branches, CALL, CALL_REF and RET each cost one; callee instructions cost
individually. Caller resumption, frame bookkeeping, validation and cleanup cost
zero. No helper can reset or privately copy the budget. An attempted instruction
that fails after the charge remains counted. A program needing N instructions
succeeds at N and fails before instruction N at N-1. Limit0 can prepare/begin,
then fails at the first root instruction without issuing a service operation.

All execution paths use one checked charge primitive. LIMIT for another reason
leaves fuel_exhausted false. Counters cannot overflow under the maximum limit;
separate physical generation counters retain their own overflow checks. The
report includes the first failing site and the original runtime cleanup report.
The old NvmFileRuntimeReport layout and native ABI1 query stay unchanged.

Fuel bounds decoded instruction work, not wall-clock time or hostile host I/O.
Service calls remain synchronous and can block inside the host. I retain bounded
external test-process supervision and make no sandbox/cancellation guarantee.
Cleanup is never fuel-limited: it walks bounded live roots/frames/references and
finishes even when no instruction fuel remains. A secondary close failure cannot
replace the primary ASSERT/fuel/host/state failure. All failure paths leave prior
scalar output untouched; successful scalar publication occurs only after clean
finish and destruction, with no live File or Result escape.

## I match VM and genuine native functions

The private VM consumes the composed plan and explicit frame cursor. Native C
retains real generated functions, labels, direct calls and ordinary C operators;
I do not embed a bytecode dispatch interpreter. A generated label checks its
original function/instruction, obtains the current checked variant, charges
fuel, performs its generated operation and advances through the checked edge.
It must support all admitted variants at that label or emission refuses.

The emitter retains its128MiB output ceiling and atomic malloc-owned output.
Generated startup compares serialized bytes, catalog/nominal/local/call facts,
all variant states/edges and storage facts before begin. Digest equality alone
is insufficient. I add a DISTINCT cyclic semantic ABI1 query and symbol family
with explicit options/report extents and fuel/variant semantics; I do not reuse
or redefine `NVM_FILE_NATIVE_ABI 1`. Missing/mismatched cyclic helpers refuse
before host acquisition. The explicit linked runtime is qualified with no VM
dispatch object in the native driver. Build/provider/header dependencies and
strict C11 O0/O2 requirements are part of the source checkpoint.

## I qualify each dependency before public selection

1. I implement only the copied cyclic hosted plan and bounds, with no runtime
   or public selector change; source review then fixture review precede gates.
   I cover full variants, wrong declared stack, original indices, malformed
   unused instructions, input destruction, allocation/output and exact bounds.
2. I implement the private carrier variant cursor, physical witness checks,
   fuel/report and frame transfers. Source review precedes adversarial fixtures
   for wrong variant/edge/fact/ABI, held references and cleanup. Old acyclic
   creation/report/ABI behavior remains an explicit adjacency gate.
3. I implement matched private VM and real native emission with an exact opcode
   coverage table and all-variant startup checks. Complete source and fixture
   review precedes newly cyclic service execution.
4. I qualify the SAME serialized corpus on fresh Linux and Darwin providers,
   ordinary supported compilers and strict supported ASan/UBSan/LSan scopes,
   VM and native O0/O2, retaining all bytes and first unexpected terminals.
   Shared ordinary objects and instrumented providers are labeled separately.
5. After actual dfa149 integration and private seals, I propose a separate public
   conjunction: explicit cyclic opt-in/options, fresh complete preparation,
   existing grant/gate ownership and archive/CLI packaging, output atomicity and
   installed tests. Old acyclic calls continue refusing cycles until that
   reviewed public contract says otherwise; failed cyclic admission never falls
   back. No public fuel field, ABI revision or CLI flag is reserved here.

My complete runtime corpus includes zero/one/many/nested iterations, scalar
break/continue/early RET, multiple exits, entry and initializer fuel sharing,
lower-index direct helpers, live File/reference across a backedge, two-owner
swaps, both Result outcomes, repeated acquire/use/close beyond64 acquisitions,
consume-and-replace and balanced per-iteration borrows. I assert per-iteration
live slots/streams before terminal disposal. Rejected zero-iteration loads,
held-owner moves/replacements, unbalanced regions, recursion, indirect/multiple
borrow calls and unreachable unsupported opcodes retain their checked refusal.

For each finite successful path I compare exact VM/native started counts and
scalar/service trace at limitN, and first site/output/cleanup at0,N-1. I include
a data-dependent nonterminating path in a module with a structurally reachable
valid RET, including live owner/reference roots when fuel exhausts. Failure
prefixes cover later call arguments, staged owner returns, ASSERT, host errors,
secondary cleanup failure and allocations before/after preparation. Repeated
same-context finish/destroy and fresh invocation recovery must not double-close
or leak. Invalid options/facts/ABI require zero host acquisitions. Raw outputs,
provider identities, runner bounds and pre-disposal root counters are retained;
passing after disposal alone is insufficient.

I keep the full original acyclic carrier/frame/VM/native corpus and public dfa149
positive/refusal matrix as adjacency after integration. This bounded child does
not close closed-indirect target sets, richer borrowed transport, paired source
loops/imports/helpers/mandatory shadows, LLVM/Wasm/linked support, installed
full-product acceptance or full72556/6931. Their ordered obligations remain in
[my parent control/call contract](NANOISA_FILE_CONTROL_CALL_EXTENSION.md).

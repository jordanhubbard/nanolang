# My private generated-native File checkpoint

I record this design before implementation. Task
`task_9a23711b4d9afb8b3408ccc6deff609c` continues runtime parent
`task_82ffef0affec2b23c4390931370f6774`. My
[original runtime contract](NANOISA_FILE_PRIVATE_RUNTIME.md) already requires
actual generated C functions/labels, exact embedded serialized bytes, a fresh
plan at invocation, and explicit runtime/core linking. My checked carrier and
frame milestones are merged through PR858/879. The separately owned private VM
checkpoint is being qualified from production `bdc8d29f3` and corrected fixture
`cfb2a9903`; I do not call that pending seal canonical acceptance.

This is a design-only checkpoint on canonical `8c959439d`. I do not implement,
compile or execute a new native service fixture before independent source and
fixture reviews. I integrate the qualified private VM dependency in a fresh
implementation tree when it lands. I preserve its failed first expectation and
corrected qualification; I do not replay historical failed binaries.

## My private entry and link boundary

I propose a separate `nvm2c_file_private.h/.c`, compiled only with
`NVM_FILE_NATIVE_PRIVATE`. A private emitter takes immutable serialized-v2 bytes,
a length, and disjoint `char **out` plus diagnostic storage. It returns a checked
status and publishes a malloc-owned C string only on complete success. A failed
query, allocation, extent, opcode or emission leaves the prior output untouched.
The ordinary `nvm2c_emit` does not call it. Macro-off builds introduce no callable
adapter or unresolved default provider dependency.

The emitted translation unit exposes one private callable entry returning
`NvmFileRuntimeReport` and conditionally publishing `NvmFileRuntimeView`. A fixture
supplies `main`; failure is observable without terminating the test process.
This entry takes no mutable module, caller proof, capability, service object or
preexisting context. It creates its own invocation from its exact immutable
embedded bytes, in `NVM_FILE_RUNTIME_NATIVE` mode. Caller output must be disjoint;
external serialization remains the existing private runtime requirement.

I explicitly link the checked File runtime/frame, File-values/service/capability,
hosted/body/flow/code/nominal and common codec/query provider closure. The exact
source/object/link list and transitive system dependencies are reviewed and
recorded before qualification. Generated C includes the private checked headers;
it performs no source-path lookup, compiler invocation or runtime-file lookup.
This first private artifact is not the installed standalone-C product contract.
I do not alter public nvm2c, VM, verifier, converter, wrapper, linked, LLVM, Wasm
or source-profile service refusals. No dynamic loader, co-process or public VM
entry is needed to execute it.

## My generated program and fresh agreement

I lower each prepared function to a distinct static C function. Each prepared
instruction has a static C label. Direct bytecode calls become direct generated
C calls, and checked branches become C predicates and direct gotos. No emitted
opcode switch, program-counter dispatch loop, generic instruction executor or
call to the private VM interpreter performs the program. A private scalar helper
may implement one fixed arithmetic operation; it does not decode an opcode.

I embed the serialized module as immutable data for fresh preparation, not for
instruction interpretation. Both emission and every invocation check the entire
plan: every function and instruction, including unused functions and unreachable
opcode coverage, exact operands/successors, body reachability, cleanup/refinement/
exit facts, obligation targets/rights and exact pending/discharged masks. I also
retain and compare the startup, function/local declarations, exact nominal and
import maps used by generated instructions. Unknown bits or a changed provider
interpretation refuse before begin/acquisition, even if a weaker query accepts.
I compare meaningful fields explicitly, not padding bytes or pointer identity.

The implementation checkpoint defines the concrete private runtime ABI/version
agreement and generated fact table before review. It must cover the carrier,
frame, catalog and opcode-policy assumptions used by generated code; a compiler
header match or successful link alone is insufficient. I do not introduce a new
wire-format feature or globally widen shared authority for this agreement.
Fresh context allocation and complete agreement precede `begin`. Failure drains
only resources actually acquired by this invocation and preserves prior output.

## My roots, calls and control flow

I use the existing NATIVE whole-frame arena. Every owner, Result owner, formal
reference and service result remains in a checked carrier root. C automatic
variables contain only passive views, scalar values, root indices, statuses and
bounded temporary index arrays. I do not create a second owner stack or release
File values through an unrelated allocator.

At each label I verify the expected current function/instruction and use the
existing checked site/frame accessors. After an ordinary operation I complete
its exact physical stack transition with `frame_next` and take the statically
selected label. The branch predicate and Result arm come from actual live values,
not the query's possible-arm set. I preserve nominal/global/catalog identities
without filtering or renumbering caller layouts.

At CALL/CALL_REF I validate/stage through `frame_call`, invoke the exact generated
callee, and resume only at the prepared continuation. The callee performs checked
`frame_return`; a C return transports status, never an unrooted File/Result.
Overlapping argument slots, borrowed formals, forwarding, nested borrows, staged
owner returns and failure after partial transfer retain the existing frame
semantics. An error unwinds passive C frames; one outer finish/destroy drains the
still-rooted arena. A helper never destroys its caller's invocation.

The supported acyclic graph has the existing checked depth bound (at most 64).
I do not promise constant native machine-stack usage or count compiler/libc stack
bytes inside the project arena. I retain the checked native value/reference/
region/frame storage bound and allocate project storage before begin. No
post-begin project allocation is added. Any execution-step overflow guard reports
LIMIT, never successful truncation; no practical timeout proof follows from it.

The initializer executes completely before entry in the same context. Its VOID
completion and cleanup snapshot must be clean before entry acquisition. Entry
publishes only its exact clean INT/BOOL result after finish. First error/site
survives subsequent cleanup errors; secondary cleanup diagnostics remain visible.

## My supported operations and refusal boundary

I match the reviewed private VM inventory, not every opcode in the ISA:

- Exact INT/BOOL/VOID constants, checked local/stack operations and owner moves;
  modulo-2^64 integer add/subtract/multiply/negate, total division/remainder,
  matching-tag INT/BOOL equality, typed integer comparisons and BOOL logic.
- Real conditional/unconditional edges, ASSERT, direct CALL/CALL_REF and RET;
  exact region/exclusive-borrow/end-reference operations.
- Exact catalog passive record/variant construction, tags and projections;
  File Result branch/take, explicit drops and five-method File service calls.

I preserve signed minima, zero divisors and MIN/-1 semantics without signed C
undefined behavior or duplicate operand evaluation. The production checkpoint
includes an explicit opcode-to-generated-operation and obligation-discharge
table. Reachable pending masks must match handlers exactly; unsupported opcodes
in unused code still refuse before any host effect. FLOAT, strings, ordinary
arrays, richer owner graphs, indirect calls, loops and arbitrary externs are not
silently added.

A service output is first rooted in the existing final staging scratch slot.
Only successful checked publication moves it into the emptied operand position.
Generation overflow, stale references, invalid byte values, direction failures,
owner-drop errors and failed return publication keep a cleanup root. I reuse
`nsi_file_values` and its real File service, not a second service implementation.

## My ordered implementation and acceptance

1. I send the complete private emitter, generated ABI/fact agreement and exact
   build/provider closure for static review. Public guards remain unchanged.
2. I send the complete fixture and runner before any native service execution.
   The same serialized corpus runs through the qualified private VM and actual
   generated C. I preserve old carrier/frame/VM assertions and all first failures.
3. I compile generated C with strict C11 warnings at O0/O2 and execute both on
   Linux and Darwin. I identify actual GCC/Clang, Apple/Homebrew compiler, SDK,
   include/library dependencies and executable hashes. Homebrew sanitizer gates
   explicitly enable ASan/UBSan and leak detection. I report the actual rebuilt
   instrumented provider closure; ordinary reused objects do not become covered
   merely because the final link uses sanitizer flags.
4. I verify real temporary acquisition, byte I/O/rewind/EOF, both Result arms,
   all direction/domain failures, cleanup and initializer/entry order. The full
   corpus includes numeric boundaries, both conditional edges, nominal
   permutations, overlapping calls, nested/formal aliases, owning File/Result
   returns, explicit and automatic drops, ASSERT and post-acquisition failures.
   I compare exact scalar results, statuses/sites and cleanup observations with
   VM expectations, including the existing owner-module result 37.
5. I qualify allocation/output sentinels before preparation, allocation failures
   across preparation/emission/context construction, and a post-begin project
   allocation ban. I test generation-limited call/return/scratch publication,
   modeled I/O errors following real I/O, close failure plus secondary cleanup,
   and repeated invocation/context recovery. Real host counters and pre-disposal
   root checks are explicit; disposal does not mask leaked roots. Fault hooks
   remain private fixture mechanisms with separately labeled instrumented and
   ordinary linked routes.
6. I test malformed/truncated modules, unsupported dead instructions, changed
   expected facts/ABI and unknown obligation masks before acquisition, unchanged
   public refusal/output sentinels, and absence of loader/fork activity. Source
   and symbol inspection confirms actual generated functions/labels and absence
   of a VM/interpreter dependency. If a corrupted-fact control cannot be exposed
   through serialized input, I label its copied-fact unit boundary explicitly.
7. I seal source/tool/provider maps, generated C, all retained binaries, logs,
   commands and archives, and verify advertised report paths against committed
   Git blobs. Independent review precedes merge and bounded task reconciliation.

No result here closes public/source File acceptance, richer control-flow/borrow
clauses, full runtime82ff or the complete 5.1 release merely from a private native
gate. I review any needed shared helper change before implementation and preserve
the separate private VM owner's qualified tree and evidence.

## My first production checkpoint

I integrate the approved contract into fresh canonical PR885 `d416b2456`, which
also includes PR884's checker registry/common-provider changes. My original
qualified VM trees remain untouched. Native qualification must build its own
current providers; old common objects are not current integration evidence.

My concrete private API is `nvm2c_file_private_emit(bytes,size,out,err,err_size)`.
I bound generated source, including the terminator, to 128MiB and preserve output
on every failure. This is an output bound, not a claim that transient realloc
storage plus the hosted plan fits in 128MiB. Both the emitter and generated entry
exist only under `NVM_FILE_NATIVE_PRIVATE`; no default Make/provider or public
selector changes are needed. Private fixtures will explicitly compile the new
emitter and flagged runtime with their checked dependencies.

My semantic ABI revision is 1. Generated C fixes that literal revision and checks
it against the linked carrier TU through `nvm_file_runtime_native_abi`, including
view/frame/report sizes. The owning carrier TU exports the check only under the
private macro. Any incompatible carrier/frame/catalog/operation semantics must
bump the revision and repeat qualification; sizes alone are not semantic proof.
Generated prebegin agreement then compares every meaningful startup/function/
local/type/instruction/body/obligation field and each used import identity with
the actual newly prepared plan. Exact serialized bytes stay embedded and owned;
no padding or hash collision is used as authority. Labels validate the current
NATIVE function/instruction and expected input stack before any operation.

| Opcode family | Generated execution | Pending runtime obligation |
| --- | --- | --- |
| NOP, JMP | Checked next plus static goto | Physical stack/selected successor |
| PUSH_I64/BOOL/VOID | Reserve exact empty root, scalar construct | Exact tag/domain |
| DUP, POP, LOAD_LOCAL | Copy/drop after carrier category checks | No implicit owner or formal copy/loss |
| STORE_LOCAL, OWN_STORE_LOCAL | Existing checked frame store then next | Exact declared local and occupied-root policy |
| OWN_MOVE_LOCAL | Exact owner observation then carrier move | Unique owner/live source |
| REGION_BEGIN/END, BORROW_LOCAL_EXCLUSIVE, FILE_END_BORROW | Existing checked frame transitions | Region floor, live exclusive origin, balanced formals |
| FILE_DROP_LOCAL/STACK | Carrier drop at exact root | First error and secondary cleanup |
| FILE_RESULT_BRANCH/TAKE | Actual live arm, checked take, direct edge | Arm refinement and generation/root validity |
| FILE_SERVICE | Exact import; checked reference/input; scratch then output move | Binding, invocation, liveness, rights, borrow, byte, Result and cleanup |
| CALL/CALL_REF | Frame staging, direct generated callee, static continuation | Exact parameter/Result identity and aliases; partial transfer roots |
| RET | Frame return, C status return | Exit obligations and rooted return publication |
| ASSERT, JMP_TRUE/FALSE | Exact BOOL, drop predicate, C conditional | Real assertion/selected edge |
| ADD/SUB/MUL/NEG and typed I64 equivalents | Unsigned arithmetic then memcpy bit conversion | Matching INT, no signed overflow |
| DIV/MOD and typed I64 equivalents | Explicit zero and MIN/-1 branches | Existing total integer policy |
| EQ/NE, signed relational and typed I64 comparisons | Fixed C scalar expression | Exact INT or BOOL only for generic EQ/NE |
| AND/OR/NOT | Fixed normalized BOOL expression | Exact BOOL inputs |
| AGG_PACK/UNION_CONSTRUCT | Ordered root indices, exact catalog constructor | Passive fields, ordinal/arm identity |
| AGG_GET/UNION_FIELD, AGG_TAG/UNION_TAG | Carrier projection or actual scalar-Result arm | Exact passive category; no owner copying |

I compare all unused function and dead instruction facts before begin, but a dead
label body refuses if reached unexpectedly. Such labels are syntactically
referenced through constant-false branches to keep strict unused-label checks;
they are not admitted execution paths. All generated functions have prototypes,
and direct static references retain complete unused-function coverage without
an interpreter dispatch table. Integer operations are selected at emission time.

I have not built the emitter or executed generated C at this checkpoint. The
complete source review precedes fixtures and their independent review; no test
result is claimed by this source inspection.

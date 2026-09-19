# I bind File bodies to a hosted invocation before selecting execution

I record `task_6fc967db5ee6921f84d1619ca5fd084c` under File72556 after the
private acyclic CODE/body milestone546cf. This is a preimplementation contract.
I request review of the first private hosted-plan checkpoint only. Matched
runtime carriers/handlers and public selection require separate source review
and qualification. No implementation, host operation or admission accompanies
this document.

## My actual boundaries

`nvm_file_body_analyze` owns copied declarations/CODE and checks every bounded
acyclic body. It retains pending binding, invocation, liveness, rights, borrow,
byte-domain, result-publication and cleanup requirements. It does not check the
container's required-feature envelope, hosted flags, ignored initializer result,
or declared v2 max_stack. These are real dependencies, not permissions implied
by a successful private report.

`vm_execute` requires HAS_MAIN, selects `header.entry_point`, calls the first
function whose name compares equal to `__init__` with zero arguments, then calls
the entry with zero arguments. The existing invocation/owner route rejects File
claims before dispatch. I keep those guards until the complete matched runtime
conjunction is qualified. Direct vm_call_function/vm_invoke/core, generated
native entry, facade and wrapper routes must not bypass that conjunction.

The v2 FUNCTIONS table carries max_stack; the bridged NvmModule does not retain
that declaration. A pair of independently supplied module and depth arrays
cannot prove identity. The required service bits1/9, exact120-byte nominal
catalog, ownership descriptors, original global/per-kind indices and immutable
catalog identity must remain tied to the same input as CODE and signatures.

## My first private plan consumes actual v2 bytes

I propose an opaque private hosted plan built from a serialized v2 byte span.
The caller supplies valid memory and keeps it immutable during the call; output
storage is disjoint. I validate the actual container/version/required bits and
cross-section rules with existing checked readers, retaining exact bytes or
owned facts needed by the plan. I then bridge that same decoded module and run
fresh private body analysis. No caller-supplied report or capability bit grants
permission. Version1 service metadata, unknown required features, mismatched
section/import/catalog identities and malformed partial claims remain refused.

Before allocation I bound the serialized span and validate section extents and
count products. My first limit is16MiB for input and64MiB for the combined peak
of copied input, decoded tables, bridge, body query and hosted report. The
body query retains its own existing16MiB cap. The production checkpoint must
show checked actual-size accounting for every simultaneously owned allocation,
including nested decoder arrays and temporary bridge copies; a raw input-size
check alone is insufficient. Unsupported excess returns LIMIT before the
relevant allocation. The result owns its inputs; failure publishes nothing and
frees all partial ownership. INVALID, UNRESOLVED, LIMIT and MEMORY remain distinct
where the underlying API can distinguish them; legacy ambiguous decoder errors
must not be relabeled as a proven allocation diagnosis.

I allow only the existing exact File nominal family and private opcode/call/CFG
inventory. I refuse globals, links, callbacks, passive effects, captures, generic
foreign imports and coprocess dispatch in this first hosted slice. These are
explicit initial boundaries, not a new claim that other release parents are
complete. Every function remains checked, including uncalled and later-named
initializer functions. All catalog methods remain exact builtin service indices;
FFI bit1 does not grant generic dispatch.

I copy and validate hosted facts from that same input:

- The v2 entry index must exist and bridge to HAS_MAIN. The entry has zero
  parameters, no captures and exactly one mode0 INT or BOOL result with NO_INDEX.
  File, affine OpenResult, scalar nominal Results and VOID do not escape this
  first public entry contract. Internal exact owner returns remain supported.
- I select the first `__init__` exactly in function-table order. All function
  names used for selection are valid bounded strings without embedded NUL;
  this avoids a different C-string alias for a wire name. I retain its exact
  function index, or explicit absence, in the plan.
- A selected initializer has zero parameters, no captures and zero VOID results.
  This first slice refuses even a scalar ignored initializer result, and always
  refuses an owned ignored result. It may perform checked internal helper/service
  work later, but its successful body must close/drop all its owners and end
  local borrows/regions. The entry cannot also be the initializer because the
  result contracts differ. Other functions named `__init__` remain ordinary
  checked internal bodies; they are not implicitly called.
- I record startup order explicitly: one invocation context, optional initializer,
  then entry only on initializer success, then complete cleanup. Context lifetime
  spans both calls. I never dispose a successful context before an owned output
  has been accounted for, and this initial host ABI publishes only scalar output.

## My stack and call storage facts are derived

For each function I derive operand peak from reachable body input/output stack
counts, plus explicit instruction transient requirements. I retain decoded
callee identity/parameter counts and exact CALL_REF transport facts; borrowed
arguments are references, not invented operand values. I compute checked call
chain/frame extents in the existing callee-first DAG order, including local
storage, caller survivors, parameters, return staging and cleanup temporaries.
Initializer and entry are sequential roots rather than a fabricated call edge.

A nonzero wire max_stack smaller than the derived operand peak is invalid.
Zero remains an undeclared value, as existing bridge documentation states; it
is never a zero-allocation instruction. Larger declarations do not enlarge
runtime allocations by themselves. The owned report retains derived exact
bounds independently, and a later runtime/emitter uses those bounds with its
actual carrier size and overflow checks. The production review must compare
VM call-frame layout and native argument/result staging against this calculation;
logical stack height alone does not certify either allocation ABI.

My first hosted-plan query remains non-executing even after these facts pass.
Public direct in-memory routes must obtain an equivalent privately generated
plan from their actual module and envelope, or refuse; a caller cannot attest
that its old private report belongs to modified metadata. Mutable module/proof
caching needs an explicit lifetime and invalidation contract before reuse.

## My later runtime conjunction cannot silently discharge obligations

| Pending logical requirement | Later matched runtime requirement |
| --- | --- |
| Binding | Exact immutable five-method catalog/version and function/global/per-kind/runtime identity maps, without dynamic library symbol lookup. |
| Invocation | One fresh checked private File-values context, initializer plus entry ordering, explicit acquired/no-acquisition cleanup and actual synchronized entry exclusion or a documented enforced serialized host API. |
| Liveness | Invocation/slot/generation checks on each File/OpenResult use; moves clear the source and stale C/value copies confer no ownership. |
| Rights | The qualified service adapter checks required/acquired rights before host access; source/structural tags cannot grant them. |
| Borrow | Exact live exclusive reference/epoch tied to owner and region; move/close/drop while held is refused, and unwind revokes references before owner cleanup. |
| Byte domain | write_byte validates INT0..255 before narrowing or stream access; refused input retains File. Read Ok yields0..255 or canonical0 at EOF. |
| Result | Stage real scalar/affine Result output, validate selected arm and publish only after all fallible preparation; publication failure drops the new owned result exactly once. |
| Cleanup | Track actual frame/local/operand/pending-argument/result owners on all normal/error/ASSERT/allocation exits; accepted close consumes File on both Ok and Error; preserve first execution error and secondary close/cleanup statuses. |

Private `nsi_file_values` already supplies checked value lifetimes, but existing
NanoValue/native owned carriers do not acquire a File finalizer from that fact.
I require separately reviewed carrier representation and source/global/runtime
identity mapping. TAG_OPAQUE remains the coprocess proxy tag. Ordinary structural
pack/get/dup cannot manufacture, copy or project opaque File ownership.

Every boundary must retain real temporary roots across CALL/RET, Result takes,
service output preparation and stack/frame growth. The failed instruction owns
its precise consumed/unconsumed inputs. Cleanup of partial arguments, selected
Result payloads and prior outputs cannot be delegated to an untracked C copy.
Finish runs individual root cleanup before disposal, preserves first and
secondary errors, and refuses a clean overall result when cleanup failed.
Reentry and disposal controls cover both acquired and no-acquisition outcomes.

## My ordered review and acceptance

1. I implement and qualify only the private hosted-plan/envelope/startup/bounds
   query after review. Controls cover actual bytes/features/catalog mismatches,
   zero/multiple initializer names and selection order, initializer signature
   and owner-result refusal, malformed/undersized max_stack, zero undeclared
   depth, exact call/frame bounds, all output/allocation/input-lifetime failures,
   and unchanged public consumer refusal. No File handler executes.
2. I separately review matched VM/native carrier and service handlers, complete
   pending-obligation coverage, and common cleanup before any public selector
   changes. Private target controls then use actual temporary-file lifecycle,
   byte write/position/read, both update-stream direction guards, consuming close,
   descriptor observations, errors, fault allocation prefixes, stale/cross-context
   identities, borrowed/moved inputs, initializer failure, reentry and disposal.
3. Only a combined fresh selector plus matching checked runtime route may admit
   this exact family. All direct APIs and emitted paths retain File priority and
   refuse unsupported claims before output or host effects. Ordinary/owned/mixed
   neighbor verdicts and output preservation remain adjacent gates.
4. Paired C/Nano generated NSI declarations, full source and mandatory shadows,
   actual installed routes, loops, indirect-call policy and richer borrowed calls
   remain concrete required File parent work. This bounded acyclic hosted slice
   does not silently defer or close those clauses. Socket/GPU are separate handle
   families; this exact File catalog does not authorize either.

## My first implementation accounting details

I preflight every section's count/row and nested field extents before invoking
allocating readers. I retain prior-only nested indices, already required by
nominal-v2 validation, so no forward-graph decoder scratch is entered. My64MiB
bound conservatively sums simultaneously possible phases: copied envelope,
decoded rows/fields, retained-layout decoder replay, bridge initial/grown
capacities with old+new replacement peaks, string/tag payload copies, retained
layout/ownership/service bytes, nominal-v2 temporary pointer/function/layout
views and its exact nominal allocation, and the full existing16MiB body budget.
The whole reader's internal validation bridge and my later owned bridge are
sequential; neither is omitted from that common peak bound. Missing or malformed
extents fail before allocation. Legacy TRUNCATED ambiguity stays UNRESOLVED.

Derived storage is an upper bound in abstract value/reference/region/frame slots,
not an existing File runtime carrier byte ABI. A VM-style suffix calculation
subtracts consumed ordinary operand arguments while retaining caller locals and
staged arguments/result; a native-style calculation retains the full caller
frame during the callee. Each frame reserves one result and its largest direct
call's argument staging. Reference and region arrays conservatively retain the
existing256-slot per-frame limit. Wire zero depth remains undeclared, and actual
runtime carrier sizes, heap payload extents and error cleanup remain pending
before any public allocation or service dispatch can consume these facts.

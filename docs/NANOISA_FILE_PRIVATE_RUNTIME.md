# I bind File carriers to one checked hosted invocation

I record private runtime child `task_82ffef0affec2b23c4390931370f6774` under
hosted `task_6fc967db5ee6921f84d1619ca5fd084c`. This is a preimplementation
contract, extending [my hosted conjunction](NANOISA_FILE_HOSTED_CONJUNCTION.md).
I request review of the first carrier/context checkpoint before implementation.
VM dispatch and native lowering need their own complete source checkpoints.
No service execution, public admission or source change accompanies this text.

## My qualified inputs and remaining work

PR853 merged at `1e64676e16761936facc0c63da825257124a8184`. I attach actual merge
and ancestry evidence to hosted6fc without closing it. Its private plan accepts
actual serialized v2 bytes, exact nominal/ownership facts, acyclic bodies and
startup signatures. It owns copied decoded instructions and declarations,
derives abstract storage bounds, and preserves pending runtime obligations.
It does not install a File finalizer into NanoValue or generated owned values.

PR854's shared bridge repair and combined hosted/provider qualification remain
separately pinned. This contract is based on canonical854
`1ad52d2fc73c9f7fe6f157f48fb3410fd9e11d1b`; I keep the
private exact-CODE containment. A successful plan remains necessary but is not
itself a host grant or runtime cleanup certificate.

My current lifetime implementation is `nsi_file_values.c/.h`, backed by the
qualified `nsi_file.c` adapter and private capability table. It has 64 live
File/OpenResult slots, invocation/generation identities, exclusive borrow epochs,
consuming close on both outcomes, and terminal cleanup reporting. It requires
external serialization, including creation counters in both layers. A copied C
handle is not another owner.

My ordinary VM and native carrier do not understand these identities. File
service claims are still rejected by common/general/function/linked verification,
VM route readiness, nvm2c/nvm2llvm, facades, full disassembly and both wrapper
inputs. I retain all those guards in this child. The private target adapters
are explicitly selected test/build APIs, not a fallback in public dispatch.

## My exact first runtime boundary

I implement the existing eight-type, five-method immutable catalog only. The
entry remains zero-argument INT or BOOL; a selected first initializer remains
zero-argument VOID. Internal exact owner returns and supported CALL_REF remain
available. I retain the qualified acyclic CFG/call limits, all-body checks,
no-global/no-callback/no-link/no-capture boundaries and the complete required-v2
feature envelope. Version1, unknown catalog/nominal identities, bare opcodes,
and mismatched service declarations do not reach context creation or host calls.

I do not add an int extern wrapper, generic symbol lookup, COP proxy or caller
function pointer as a File binding. Exact import indices in the plan select the
immutable catalog ordinal, which selects a compiled call to the corresponding
`nl_file_value_*` operation. FileError fields and structural tags grant no rights.

A private invocation constructor takes serialized bytes, builds a fresh hosted
plan, and owns that plan until all execution storage is destroyed. Caller bytes
need remain immutable only during construction. No independently supplied
module, proof, depth array or host token can replace the checked input. No
mutable plan cache or cross-module proof reuse enters this checkpoint.

## My carrier and nominal values

I propose a private `NvmFileRuntimeValue` with initialized/empty state, copied
exact declaration identity, and one of these payloads:

| Category | Private payload and operations |
| --- | --- |
| VOID/INT/BOOL | Exact scalar value; ordinary copy/dup permitted only for initialized mode0 scalar values. |
| File | One `NlFileValue` owner; only checked move, borrow, consume-close or drop. |
| Affine OpenResult | One `NlFileValue` owner, including Error arm; inspect through `nl_file_open_view`, consume through the exact take API. |
| FileError/ReadByte | Fixed inline scalar fields in catalog order, with original global identity; no heap pointer. |
| Write/Position/Read/Close Result | Explicit arm plus fixed inline scalar/record payload and exact nominal identity. |

At most seven scalar fields are needed by the existing catalog. A result's
payload is bounded by FileError's seven fields; ReadByte has two. I validate
field kinds/counts and exact declaration identity before publication. I do not
coerce generic INT into BOOL, use VOID as unknown, or interpret an arbitrary
record as an opaque File. Runtime Error fields preserve status, errno,
cleanup_errno, progress, eof, consumed and cleanup_failed without narrowing.
The byte service has at most one byte progress; I still check size_t-to-INT
conversion before publishing an error record. Source-constructed passive
FileError values may contain ordinary INT fields; they remain passive data.

Ordinary AGG_PACK/UNION_CONSTRUCT can build only the passive catalog shapes
already permitted by body analysis. They cannot manufacture File or OpenResult.
Projection/tag/dup use the exact passive shape and arm. FILE_RESULT_BRANCH
observes the actual arm; FILE_RESULT_TAKE checks the selected arm and consumes
the local. Unknown arms are never treated as an assumed Ok. Empty and VOID are
distinct states. Uninitialized loads trap before reading storage.

A mode2 File formal stores a checked reference alias, never an owned File
payload. Clearing that formal does not drop File or end the caller borrow.

A VM private frame uses these carriers as a companion arena, not TAG_OPAQUE or
an unqualified NanoHeap object. A generated native frame uses the same carrier
contract. I retain separate VM suffix-frame and native whole-frame layout
requirements; matching a C struct does not erase those different bounds.
Only the final scalar crosses into the host-visible output representation.

## My allocation and invocation lifecycle

I allocate the complete frame/value/reference/region arenas and staging storage
before acquiring any stream. Sizes come from the same hosted plan and checked
actual sizeof products. I use its VM or native bound according to the adapter,
not a wire max_stack allocation or an assumed common frame layout. Zero
max_stack means undeclared; the derived bound still applies.

My project-owned memory ceiling for this slice is 64MiB for the conservative
hosted-plan bound plus concrete runtime arenas, context, cleanup records and
all fixed File-values/service/capability storage. I add private nonallocating
storage-size queries in the owning translation units if needed so opaque
structs are counted with actual sizeof, including the full capability table
rather than only its 64 privately usable slots. I check every product and sum
before each affected allocation. Old/new storage overlaps are counted; the
initial design needs no growing carrier arena. A bound refusal precedes stream
acquisition and leaves output untouched.

This is a bound on project-owned storage, not libc's internal FILE allocation
or process RSS. Host resources remain separately bounded by the existing 64
slots. Exhausting those slots is a checked runtime failure with complete
cleanup, not a claim that every accepted body must succeed under every resource
budget. No successful service output depends on a subsequent heap allocation:
I reserve its empty result root before calling the service.

The private API explicitly requires external serialization across all calls to
these contexts and the underlying private File/capability APIs, including
creation/destruction. I do not advertise thread safety. A context state rejects
same-thread nested entry and use after terminal finish; that state is not a
substitute for a mutex. General concurrent public use needs separately reviewed
synchronization before later admission. There are no callback/reentrant host
service bindings in this slice.

Creation stages the plan and arenas before publishing a context. Failure owns
and frees only what it acquired. Run has an explicit acquired flag and phases:
ready, active initializer, active entry, cleanup, terminal. Entry begins only if
the initializer succeeds. Both share one fresh `NlFileValues` context. A
no-acquisition refusal does not finish/dispose another active invocation. After
terminal finish a new invocation uses a fresh File-values identity; I do not
revive an old handle by reusing its context address.

## My instruction, call and reference ownership

Every slot that may own File/OpenResult participates in cleanup, including
locals, operand values, pending arguments and pending returns. I never use an
untracked temporary C copy as an owner. Checked moves go through
`nl_file_value_move`: destination is empty, source is cleared only on success,
and old generations become stale. Scalar/passive copies do not alter owners.

Before CALL I validate the exact callee, reserve its frame/staging and transport
all arguments in source stack order. Each owner moved out of a caller root is
immediately rooted in staging or the callee. If a later transfer fails, cleanup
owns the already moved prefix and the still-live caller suffix; I do not invent
rollback generations. On RET I stage the exact result before draining callee
roots, then move it into an empty caller result slot. A failed final transfer
still leaves exactly one tracked root. No owner escapes the scalar host ABI.

Borrow descriptors store the exact live `NlFileValueBorrow`, owner location,
region and formal/local origin. A local exclusive borrow has one releasable
epoch owner. CALL_REF passes a checked reference alias to the callee formal;
it does not mint a second borrow or allow the callee to end the caller's epoch.
Callee return/unwind clears formal aliases; caller cleanup ends its actual
borrow. Duplicate borrowed roots remain refused by the existing call proof.
Reference slots and region depth use the hosted frame bounds and actual runtime
checks. Move, close, drop and overwrite while held are refused before mutation.
REGION_END ends its local references before removing the region. Unwind clears
callee formal aliases, ends originating borrows, then drops owner roots.

FILE_SERVICE's five operations preserve their qualified lifetime contracts:

- temp publishes an affine OpenResult only after capacity/slot preparation.
  A checked core failure publishes nothing; a host failure is its real Error arm.
- write_byte checks INT0..255 before narrowing or stream access; range Error
  retains File. rewind/read also require the exact live exclusive epoch and
  rights. Both update-stream direction changes still require successful rewind.
- read Ok preserves the actual byte/eof pair and canonical zero at EOF.
- close consumes File on both Ok and Error after an accepted close attempt.
  Rejected stale/borrowed/type inputs do not close any resource. No second close
  follows an accepted-close Error.

The runtime result distinguishes the first execution failure (including function
and instruction site) from core status and the full cleanup report. Assertion,
capacity, malformed runtime identity and allocation errors never disappear into
a later successful close. I drain every tracked root before terminal File-values
finish/destroy, then publish a scalar result only if execution and cleanup are
both clean. The core deliberately counts an accepted close Error in cleanup
failures even when the source handles its Error arm; I preserve that behavior.
Secondary cleanup errno and failure counts survive every later cleanup step.

The body query permits generic and typed integer arithmetic only at exact INT
operands. Both targets must implement the existing modulo-2^64 add/sub/mul/neg,
total division/remainder (zero divisor and MIN/-1 cases), exact INT/BOOL equality,
comparison, boolean operations and assertion semantics without C signed-overflow
UB or duplicate operand evaluation. This is existing opcode semantics, not a
new numeric policy. No FLOAT, string, ordinary array or owner-shell managed field
is silently added to this File execution path.

## My matched private targets

1. **Carrier/context checkpoint.** I add the private runtime value/context and
   fixed-storage accounting, exact scalar/passive construction, owner moves,
   reference/formal handling, staged service results and common failure cleanup.
   It uses `nsi_file_values`, not a second File service implementation. I send
   the complete source and allocation/ownership table for review before any new
   fixture can execute a service. The default public dispatch remains unchanged.
2. **VM checkpoint.** I add an explicitly macro-gated private NanoVM adapter
   over the owned hosted plan. It dispatches copied decoded instructions and
   checked successor indices using bounded VM frames; it does not reinterpret
   a caller's mutable NvmModule or reuse a stale public readiness proof. The
   private engine is identified separately from vm_execute/vm_invoke admission.
   All File handlers and integer/control/call paths are reviewed before target
   qualification. Unsupported operations refuse before context/host effects.
3. **Native checkpoint.** I add a private C lowering from the same fresh plan,
   with generated functions/labels and explicit helper calls for each supported
   operation, not a wrapper that calls the private VM interpreter. Generated
   code embeds the exact serialized module and verifies its fresh plan/runtime
   ABI at invocation, then uses the separately bounded native frame arena.
   It links the reviewed runtime/core objects explicitly. This first private
   link contract does not claim installed standalone C, LLVM or Wasm admission.
   Those existing public entrypoints retain their service refusals.
4. **Target acceptance and later public conjunction.** After independent complete
   production/fixture review I qualify actual private VM and compiled native
   execution with the same serialized fixtures on Linux/Darwin. Public selection
   is a separate subsequent contract requiring all pending obligations to be
   discharged by the matching invocation, not by the old query's success bit.

I keep runtime references out of normal `file_flow`/ISA query objects until a
reviewed provider change is needed. Private build targets explicitly link the
runtime, File-values, File service and capability sources with their query
providers. Macro-gated adapters must not introduce unresolved default-build
symbols. If normal provider manifests change, I audit Make, module manifests,
wrapper object closure and Forth closure together; current manifest success is
not evidence for a changed link set.

## My qualification obligations

I retain the complete old hosted/flow/body/transport/refusal fixtures. New private
gates use fresh source/tools and first-terminal preservation, strict GCC/Clang
normal/sanitizer builds, explicit Darwin compiler/SDK selection and real native
binaries. They cover:

- Actual temporary acquisition, write/rewind/read/EOF, both direction errors,
  consume-close and automatic temporary-file removal observations, exact catalog
  Result/error fields and host descriptor closure. I do not invent a pathname
  or claim a general remove API from tmpfile's lifecycle.
- Initializer-before-entry ordering, no entry after initializer failure, shared
  invocation identity, sequential max storage bounds and scalar publication only
  after all cleanup. No initializer or owner result is silently discarded.
- Owner moves through locals, stack, nested helpers, RETURN and partial CALL
  staging; both OpenResult arms and scalar Result arms; alias/stale/cross-context
  rejection; local/reference/formal lifetimes and full region unwinding.
- Every allocation prefix and single transient failure across preparation and
  runtime creation, no stream before storage readiness, result-publication and
  acquired/no-acquisition paths, failed open/seek/read/write/close, partial progress,
  first plus secondary error preservation, final disposal and fresh reentry.
- Multiple live streams/slot reuse, exact resource ceiling, root counts and
  unrelated sentinel descriptors that neither stale cleanup nor duplicate close
  may affect. Deterministic fault hooks supplement actual linked libc behavior;
  neither substitutes for the other.
- Exact integer extremes/division/boolean/control behavior, reached ASSERT
  failure, malformed actual carrier categories, bounded call frames and every
  common error exit. These are fresh private fixtures, never historical replay.
- Ordinary/owned/mixed neighbors and every public File refusal before output,
  library loading or host access. Native generated output preservation remains
  checked on emitter failure.

Only this private matched runtime child can close from those gates. Public File
selection, paired NSI source declarations/producers, complete mandatory shadows,
installed routes, loops, indirect calls and richer borrowed calls remain concrete
required parent work. I do not close File72556/6931, hosted6fc, d03c or the release
because a private carrier and two private targets pass.

## My first carrier checkpoint ownership table

I implement carrier primitives before an instruction dispatcher. Their caller
must later enforce the checked CFG, frame layout, local declaration placement
and each call's no-duplicate-borrow argument rule. These primitives do not claim
to discharge those obligations. Nested formal aliases can refer to one originating
borrow; they never own its epoch. The matched call adapter must distinguish valid
forwarding from duplicate arguments in one call.

| Allocation/root | Acquisition and owner | Release/publication |
| --- | --- | --- |
| Hosted plan | Fresh serialized prepare; context owns it after construction | Every partial failure frees it; final context destruction frees it. |
| Context | Exact sizeof, including cached nominal facts and result staging | Published only after all arenas and maps succeed. |
| Value arena | Actual carrier sizeof times selected VM/native derived slots | Each owner is moved into an empty counted slot; cleanup attempts every live root. |
| Reference arena | Actual reference sizeof times reported reference bound | Formal aliases clear first, then original epochs end. |
| Region/frame arenas | uint64 region IDs and concrete frame bookkeeping extents | Fixed allocations; no growing/unaccounted staging storage. |
| File-values/service/capability storage | Owning-TU nonallocating checked queries, nested bound counted once | Created at begin after all arenas; terminal destroy follows individual root cleanup. |
| Service output | Empty arena root reserved before the core call | No allocation follows host acquisition; accepted close invalidates input even on Error. |
| Host scalar output | Context's passive result staging | Published only after successful entry and clean terminal cleanup. |

Before implementing initializer completion I need a read-only private snapshot
of the existing File-values cleanup report. Otherwise a handled close Error in
the initializer could be noticed only at terminal finish, after entry had begun.
The snapshot copies existing status/counts without finishing or changing the
context; initializer completion refuses pending cleanup failure before selecting
entry. This preserves one context across both roots and first/secondary errors.

Qualification will count attempted host acquisition/library loading, not merely
successful opens or unchanged loader state. The inherited fclose fault hook
really closes and then injects a reporting error; I retain that exact modeled
scope instead of claiming arbitrary libc close failures were reproduced.

The first context already owns its single frame arena. Later adapters receive
checked frame-index accessors to this arena; they must not allocate another
frame/argument/result arena after begin. Locals, operand survivors and every
pending argument/return are indices in the existing value arena. The VM adapter
must demonstrate its suffix overlap calculation and the native adapter its
whole-frame recurrence against the retained bounds before either is qualified.
If the concrete frame bookkeeping changes, runtime creation uses its actual
new sizeof and the same checked total; no earlier ABI/storage acceptance is
silently reused. A native generated function's temporary C scalars may not hold
an owning File/OpenResult outside a counted root. These access/layout adapters
are explicitly absent from the first carrier checkpoint.

# I retain every possible File call target

I scope `task_2c135a488bd61576caf83debb2786270` under72556/6931 and my
[control and call contract](NANOISA_FILE_CONTROL_CALL_EXTENSION.md), at
canonical `5f98a27a0565da92539dfc7bd242aa01f6c16d8e`. This proposal adds no
executable admission. The cyclic carrier/fuel and paired source lanes continue
separately. I review the complete declaration/transfer design before production.

## My actual callable representation

`isa.h` already defines `FUNCREF` with an original u32 function-table index,
and `CALL_INDIRECT` with u16 argument/result counts. In `vm.c`, arguments lie
below the callable at stack top. `FUNCREF` produces TAG_FUNCTION with the
current module identity; the ordinary VM also accepts captured closures and
cross-module callable identities. Those broader routes supply no File authority.
Its ordinary arity check alone does not prove File parameter modes or cleanup.

My File declaration reader currently accepts scalar INT/BOOL/VOID and exact
catalog nominals, and refuses FUNCTION. `file_code.inc` refuses both callable
opcodes. My cyclic query relies on those exact declarations and decoded rules.
I will not change an opcode number, reinterpret an integer as a function, clear
module identity, or make a new declaration fact select either existing runner.

## My first bounded query

I propose a private copied target report. It describes which original functions
may be selected at each indirect call; it does not certify File ownership or
prepare an executable hosted plan. It validates the complete File nominal and
service envelope, all function boundaries/operands and declaration tables first.
Every decoded body remains in the report, including uncalled helpers. Unknown
opcodes and unsupported callable sources refuse rather than being skipped.

The new declaration mode permits only mode0 TAG_FUNCTION with NO_INDEX in local
slots, including transient operand stack values. Existing public/acyclic/cyclic
query entries retain their old declaration/opcode choices. First-query function
parameters and results cannot themselves be callable. Callable arguments/results
and whole-program propagation are mandatory later stages, not inferred from a
local-slot success. Captured closures, globals, foreign/module imports, generic
extern dispatch, callback contracts and arithmetic/conversion on callables remain
unresolved in this first report. Other File categories retain their exact tags.

A target fact is a uint64 bitset over at most64 original functions, plus explicit
initialization and category. `FUNCREF f` contributes exactly bitf after bounds and
same-module checks. LOAD/STORE/DUP/POP preserve or clear the appropriate fact;
ordinary moves of File/OpenResult still require their separate ownership proof.
No payload bit pattern creates a callable. A join unions target bits only for
compatible initialized FUNCTION values with equal stack shape. A missing or
uninitialized predecessor never gains initialization from another predecessor.
Scalar/File/category disagreement refuses; UNKNOWN is never the empty target set.

Each function begins with its exact declared scalar/nominal parameter categories
and uninitialized remaining locals. Both successors of a branch participate;
constant conditions do not discard an inconvenient candidate. Backedge unions
iterate until stable, including the initial zero-iteration edge. Work queues are
finite: existing64 functions/256 locals and stack/256 instructions per function,
4096 instructions total and64KiB CODE; at most262144 transfer visits, with a
16MiB total retained/transient query budget. Checked arithmetic and budget checks
precede allocation. A deterministic increasing instruction-order worklist and
monotone target-bit additions terminate or explicitly return LIMIT. These are
query budgets; they do not replace runtime fuel.

The query uses actual operand-stack transfers for every supported instruction,
including branch/refinement, calls, File service/results and explicit cleanup.
Its concrete source checkpoint must enumerate each transfer and its category
facts. It does not use the generic verifier's advisory type pass as a proof and
does not make a second permissive File body selector. Shared decoding/declaration
helpers may be factored with explicit mode entrypoints; old entry behavior must
remain byte-for-byte equivalent on its original corpus.

At CALL_INDIRECT, the top value must be an initialized FUNCTION with a nonempty
set. Every candidate must have the declared arity/result count and the same exact
parameter modes, nominal layout/catalog identities and result declaration.
Borrowed formals refuse here: existing CALL_INDIRECT carries no reference-slot
vector. Multiple borrowed indirect calls require the separate richer-borrow
transport, not guessed operands. The callable is checked before its arguments
are removed. Every argument's exact category must match every candidate; the
remaining owner transition is a later complete File-flow proof.

I combine direct edges with every possible indirect candidate edge and reject
any recursive call-graph component, including an unused function's self-edge.
No first-target shortcut or name-based deduplication is permitted. The report
retains the original function, decoded instruction index and byte PC, complete
candidate bitset, exact common signature and original candidate identities.
Unsupported parameter/return callable flow returns UNRESOLVED, not a fabricated
empty set or an assumed all-functions set.

## My proposed ownership and API boundary

The first source checkpoint freezes an opaque `NvmFileIndirectTargets` report
and result status DESCRIBED/INVALID/UNRESOLVED/LIMIT/MEMORY, plus original
function/PC or NO_INDEX and a static message. I propose
`nvm_file_indirect_targets(const NvmModule *, NvmFileIndirectTargets **)`, free,
summary and copied per-call-site getters. DESCRIBED alone publishes a new report;
all failures and invalid getters preserve output. No input pointers, library
handles or executable-proof flags survive. Free(NULL) is harmless. Caller output
must be valid and disjoint; input remains immutable throughout preparation.

The report owns copied declarations, code/decoded identity, targets and call graph.
All direct allocation prefixes are transactional. Any shared verifier failure
classification is reported precisely rather than relabeled as a recoverable
MEMORY guarantee. Complete structures, field widths, checked allocation counts
and getters are reviewed with the implementation before fixtures run.

## I compose ownership before execution

A separate checkpoint composes this complete report with the cyclic File flow
engine. Each possible target receives the identical pre-call state and ordered
arguments. Every candidate's body/result/cleanup obligation must succeed; compatible
poststates join without dropping an owner, Result arm, pending obligation or
borrow relation. Per-candidate obligations replace the current single-target call
fact only in the new reviewed report, with explicit limits and exact fact matching.
All callees and unused bodies remain checked. Target-query success cannot be
passed to an old direct-call plan as a substitute certificate.

Matched private VM/native carriers then retain same-plan module identity plus
original function index, independently check set membership and exact signature
before frame/argument transfer, and clear callable slots without inventing resource
ownership. Native lowering emits bounded dispatch to real generated functions.
There is no interpreter/FFI fallback. The invocation-wide cyclic fuel account,
first-error handling, caller suffix/callee prefix unwind, result publication and
root cleanup apply identically to direct and indirect calls. Partial setup must
not duplicate or lose File/OpenResult ownership.

## My required qualification and continuation

I first review and qualify only target facts: two different callees selected by
branches and loops, zero-iteration initialization, local copies/overwrites,
heterogeneous signatures and nominal identities, wrong counts, unknown/forged
callables, all-candidate and unused-body refusal, direct/indirect recursion,
module identity, exact bounds, allocation cleanup, input-destroyed lifetime and
unchanged destinations. Existing direct/cyclic query and public refusal suites
remain required neighbors. No service executes at this stage.

Then reviewed flow/hosted/runtime integration must execute the same serialized
corpus in VM and native O0/O2 on Linux/Darwin, including every target, owner
arguments/results, result-arm alternatives, repeated calls, host/allocation/fuel
failures, sentinel isolation and recovery. Public grants/installed archive and
paired C-seed/Stage1/Stage2 callable source with full mandatory shadows follow.
Callable parameter/result propagation, richer borrowed targets and their source
routes remain required within5.1; the first bounded local-target report closes
none of72556/6931, full compiler bootstrap or release publication.

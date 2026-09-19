# I check optional FLOAT operands before private owner ARRAY execution

I continue task430220 after canonical PR830 at
`9d254504a69a9badbd39ab06bf7827687817b51e`. My complete private authority query is
qualified; my public selectors still refuse this profile. I preserve the
qualified source/tools in the authority tree. This addendum orders a query-only
operand checkpoint before a separately reviewed private VM/native runtime
checkpoint. Public activation and paired source acceptance remain separate.

## I preserve facts and attach exact operand obligations

The original Bundle/PREFIX source compares `(at samples 0)` with FLOAT1.5.
C `borrow_codegen.inc` numeric comparison and Nano `nb_binary` choose F64_EQ;
my current `oq_scalar` and `la_scalar` transfers require exact FLOAT, whereas
ARR_GET correctly produces FLOAT|VOID. I do not replace that union with FLOAT,
invent an absent value or change array bounds semantics.

My first checkpoint extends only these typed consumers:
`F64_EQ`, `F64_NE`, `F64_LT`, `F64_LE`, `F64_GT`, `F64_GE`.
Each accepts a known nonempty subset of FLOAT|VOID containing FLOAT, independently
for the left and right operand. Exact FLOAT is already admitted. Exact VOID,
UNKNOWN, other tags, owner observations and managed handles remain refused.
Optional operands originate in the existing checked value lattice; branch joins
may retain the union, but no unmodeled opcode acquires a scalar transfer. Numeric
arithmetic, negation, casts, bit transport, signatures and exact local stores
keep their existing exact-tag restrictions. Generic equality/order semantics do
not change.

I add bounded per-operand facts to `NvmOwnerOriginObligation`: two actual tag
sets, two required tag sets, an operand count and a runtime-check bit mask.
Index0 is the lower/left operand; index1 is the top/right operand. The existing
aggregate actual/required/read fields remain descriptive unions for existing
callers. I initialize location and all new fields for every decoded instruction,
including RET and instructions without obligations. The instruction budget
remains4,096; the extra arrays have fixed size2. Counts above2 are not silently
truncated. Only the six comparisons set the new optional-FLOAT check bits.

At a repeated control-flow visit I merge actual alternatives monotonically at
the same operand index and preserve required FLOAT. An exact path cannot erase
a check required by an optional path. A later incompatible alternative refuses
the analysis. The independent authority pass derives its own per-operand facts
from its own states and verifies compatibility with the fresh origin rows,
including function/PC/opcode/arity; it never treats an origin check bit as proof
that an independently incompatible operand is safe. A prepared plan retains
these unsatisfied runtime obligations, not an already performed check.

On execution I check both live operand tags before reading either FLOAT payload,
before comparison or result publication. VOID produces the existing type-error
category and follows complete root cleanup. A successful comparison uses the
existing six IEEE relational operations, including ordinary NaN behavior and
signed zero, without modifying input representation. These checks do not promise
new floating-environment behavior. Every actual consumed value is evaluated
once; there is no second array read or recomputation to obtain the tag.

## I model observable output separately and narrowly

Private failure controls need an observable successful prefix. I separately
model PRINT and PRINTLN for exact INT operands in both origin and authority
transfers: consume one ordinary exact INT, produce no value and no owner/origin
change. I reuse existing INT rendering/newline behavior. I do not add STRING,
FLOAT, owner, ARRAY, VOID or ambiguous-print admission in this checkpoint.
Other required output forms must receive their own transfer review; I do not
infer source-wide printing support from an INT prefix fixture.

The query checkpoint first qualifies left-only, right-only and both optional
operands, both branch arms, joins that acquire a later optional alternative,
wrong-tag/unknown refusals, retained exact arithmetic/cast/store restrictions,
all six comparison rows, INT print stack effects, exact locations and unchanged
output/module bytes on allocation failures. No pending module executes here.
I preserve prior strict optional-consumer refusal evidence and explicitly migrate
only its six-comparison expectation; unrelated negative assertions remain.
The complete authority suite and origin/provider adjacency run before the next
runtime implementation checkpoint.

## I keep the private runtime entry separate from public selection

I propose test-build-only adapters in the VM and native emitter translation
units, declared in private fixture headers under a dedicated build macro. Normal
objects export neither adapter and acquire no new public selector branch.
Each adapter receives the original immutable module, prepares a fresh complete
plan internally and owns it until completion. Neither accepts a supplied plan,
trusted bool, forged mixed proof or rewritten descriptors. A refusal precedes
execution/emission and preserves publication outputs. No shared validator's
ARRAY rejection is relaxed.

The VM adapter admits only a fresh, idle root invocation at function0, no links,
callbacks, tracing, active references or prepopulated caller values; entry has
no parameters and one INT/BOOL/U8 result. It checks the same constants/header and
common structural facts as the existing invocation path. It supplies a distinct
internal owner-array invocation category to the shared execution core, with exact
signatures, locals, original nominal maps and obligations. Internal CALL uses
this one fresh invocation context; public helper-entry or continuation APIs
cannot manufacture or retain that category. This checkpoint does not expose
owner-array public resume/call/link behavior. A later activation review must
cover every such API independently.

I factor only the execution/emission mechanisms needed to consume the distinct
checked plan. I do not cast an owned-array plan into `NvmMixedSamplesPlan` or
fabricate a nonempty ordinary-record proof. Existing Samples/STRING/scalar paths
retain their original preparation and signatures. The native private adapter
stages its full generated result and returns it only after complete success;
failed translation preserves caller output and normal CLI refusal behavior.
Both backends refuse any origin opcode whose runtime transfer is not explicitly
implemented. Query preparation alone never selects a runtime handler.

## I preserve unique shells and shared mutable roots

VM ARRAY roots remain retained NanoValue arrays inside unique owner records.
STRING roots keep their existing separate retention. Owner load observations
are nonescaping until projection; extracting an ordinary ARRAY or STRING stages
a checked retain before publishing an alias. Unique child records are moved,
never copied into ordinary aliases. Unpack transfers fields once and consumes
the shell. Owner calls/results retain exact nested nominal identity and consume
argument roots once. Locals, operand stack and pending results remain included
in cleanup on every error, including a failed optional-FLOAT check.

Native unique shells remain `nown_record` with category1. ARRAY values use
category2 and the exact `(NmsRuntime *, NmsHandle)` identity, including inside
nested fields. STRING uses `nown_string`, not an NmsHandle. The root invocation
owns one NmsRuntime until helpers, stack, locals, temporary operands and pending
results are drained. A shared internal declaration accessor may select either
existing mixed facts or owner-array facts, but it must retain the distinct
prepared-plan types and reject an absent category rather than guess.

Owner layouts never enter the compact ordinary record map. This profile can
have zero ordinary records: native generation emits no zero-length C array and
binds `(NULL,0)` with `nms_bind_records`. Actual ARRAY allocation needs no fake
ordinary descriptor. This stage does not newly admit ordinary record operations
inside owners; existing origin refusals remain. Original RESOURCE flags and
source/global identities survive every transfer and query.

Pack allocates the shell before consuming input roots. Retain failure publishes
no alias. Partial allocation/retain cleanup releases exactly the committed
prefix. Unpack, pending result and call return transfer rather than duplicate
roots. All error exits drain roots, then finish/dispose the managed runtime;
cleanup errors cannot turn a failed operation into success. Each mutable alias
observes writes and append growth through the same identity. Failed growth
preserves length, content and all aliases. Stale/cross-runtime handles and
counter/generation exhaustion retain checked failure. There is no int fallback
for handles and no retry that loses an unknown cleanup outcome.

## I qualify each stage before the next one

1. I implement only operand/INT-output query transfers and reviewed fixed-size
   getter extensions, send production for review, freeze fresh query fixtures
   and retain first terminal results. Public selectors remain unchanged.
2. After that evidence is reviewed, I implement the private VM/native adapters
   and exact transfer/failure paths, then send the complete production checkpoint
   before any pending-module execution. I verify normal-object symbol/routing
   absence and preserve ordinary public refusal controls.
3. Fresh private execution covers two fields/two shells sharing an ARRAY, nested
   factories/relays, reverse unpack, alias survival, mutation/growth, empty arrays
   and empty owners, STRING siblings, calls/loops/branches and both optional
   comparison operands. Wrong optional tags trap before comparison and drain all
   roots. NaN/sign/zero controls use integer bit observers where applicable;
   no floating equality is used to prove bit preservation.
4. Allocation-prefix fixtures check exact successful INT stdout prefixes,
   status, live root/object/byte counts, input preservation when promised and
   later recovery. Rejected query modules never execute. GCC/Clang strict C and
   sanitizers identify their actual instrumentation scope; VM dispatch variants
   and native use the same frozen module bytes. Existing scalar/nested/STRING/
   mixed/provider suites remain adjacent controls. I record any inability to
   inject a failure rather than claim it was tested.
5. Only after private runtime qualification do I propose public activation
   across verifier/converter/VM APIs/native and explicit LLVM/Wasm/service/link
   refusals. Unchanged Bundle/PREFIX plus all selected shadows and paired Linux/
   Darwin source acceptance remain required later. I do not close430220 here.

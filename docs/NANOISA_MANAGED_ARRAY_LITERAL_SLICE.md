# My bounded leaf-array literal and slice contract

I audited main fdec4ffd after mutable-array PR713. I continue aggregate488 and
managed51da with leaf arrays only. This contract records VM prerequisite1e89
and managed childb702 before implementation. My existing profiles remain
unchanged until each new operation has matched checked lowering. Nested arrays,
cycles, nominal children and host inputs remain separately required work.

## My existing VM behavior

ARR_LITERAL carries an element tag and uint16 count. Its count inputs are in
source order on the stack. The VM allocates before popping, chooses packed
int/float/bool/U8 versus boxed storage from the declared tag, and creates
capacity `max(8, count)`. It transfers boxed input owners directly into the new
array; packed inputs use the existing packing rules. The instruction preflight
checks the complete input count before any transfer. Allocation failure leaves
those stack roots available to terminal cleanup.

ARR_SLICE pops end, start, then receiver. For an integer bound, the VM converts
the complete signed payload to uint32 modulo 2^32. A non-integer start means 0;
a non-integer end means the source length. This is the bytecode endpoint
contract, distinct from the source array_slice start/length convention tracked
by efa11. I do not change that frontend convention in this child.

The heap helper clips start and end to source length. End <= start produces a
fresh empty array. Otherwise it copies [start,end) into a fresh array with
capacity `max(8, end-start)`. The result keeps the source's declared kind and
packed/boxed representation. Packed bytes are copied exactly. Boxed children
are retained, so changing the array structure is independent while immutable
string values may share ownership. The source is borrowed by the heap helper.

The VM handler currently omits release of popped start/end owners and does not
check the helper's NULL result before publication. The helper already checks
its allocation. My separate prerequisite1e89 releases all popped values after
their last use, including receiver-type refusal, and reports MEMORY before
publishing a result. I inspect and repair that source defensively, then test
ordinary corrected lifecycle/allocation behavior. I do not replay a failed
artifact or construct a crash demonstration.

## My literal allocation and transfer

I extend the private VM-policy constructor with an explicitly checked initial
capacity; existing NEW keeps capacity8. Literal capacity is max8/count, not the
next power of two and not repeated append growth. I validate width, widened
byte multiplication, size_t conversion and accounting before allocation. I
publish neither the handle nor a partially initialized child list on failure.

I use only the existing finite packed coercion matrix and boxed leaf tags
VOID/int/U8/float/bool/string/enum. A boxed declaration does not constrain all
children to its own tag. Unsupported packed writes are unresolved eligibility,
not new runtime coercions. I preserve source order and exact scalar payloads;
matching float payload bits remain intact.

My private preparation borrows input values until its array is complete,
retaining each boxed string edge. On any failure it releases only prepared
edges and storage, leaves caller inputs and output unchanged, and returns the
existing precise status. Generated code then consumes the input owners once
on success or error through an explicit counted-root transfer. It cannot use
FrameOutput's three-operand bookkeeping for a uint16 literal count. I retain
unprocessed values as stack roots, clear each transferred slot exactly once,
and keep the new array owned until publication or cleanup. No private array
handle escapes while its contents are incomplete.

The C/LLVM boundary continues to use scalar arguments/output pointers. If batch
input storage is needed, I pass separate uint64 payload and uint32 tag arrays
with checked count/alignment, not an assumed native/Wasm aggregate ABI. A
staged internal builder is acceptable only with the same private ownership and
rollback boundary. An initial allocation failure must leave original stack
roots intact for frame cleanup.

## My slice ownership and bounds

I add a borrowed-source private slice primitive. It validates the live array
kind and output pointer, computes uint32 endpoint fallback/wrapping in the
instruction adapter, and clips using subtraction after ordering checks. Slice
bounds never introduce BOUNDS7: reversed, negative-after-wrapping and excessive
endpoints retain their existing empty/clipped behavior. A non-array receiver
produces TYPE; allocation failure produces MEMORY; lifecycle/state statuses
retain their existing numeric ABI. Output remains untouched on private failure.

The result is a distinct VM-policy handle with exact capacity and declaration.
For packed values I copy bytes; for boxed leaves I retain each string child
before the new descriptor becomes visible. I keep source owners alive across
byte allocation and descriptor-table relocation and do not cache table pointers
across growth. Partial retention/allocation failure rolls back all prepared
children without changing source contents, aliases, capacities or references.

The emitted slice helper consumes receiver/start/end on every path, including
non-integer heap-bearing bounds that select fallback endpoints. It publishes
one result owner only after preparation succeeds. Its three operand transfers
fit FrameOutput, and existing status/error cleanup runs before the next edge.
Caller globals and prior alias writes survive later failure. Terminal disposal
reclaims both source and copied-array roots and their retained leaf edges.

## My shared shape prerequisite

Both new operations trigger the same managed array preflight used for mutation;
read-only modules without these operations retain prior admission. Newly
qualified modules consistently select prepared boxed SPLIT storage. I extend
stack analysis with ARR_LITERAL's dynamic count and check every input against
its declared storage before recording weak boxed contents.

A slice result is a fresh allocation abstraction, not an alias of its receiver.
I derive origins by slice site and possible declared storage kind, retaining
all possibilities across joins. Each derived boxed summary weakly includes
all possible source child tags. Repeated execution at one site must not reset
an older surviving instance's summary. Distinct packed/boxed declarations at
a join must remain distinct alternatives; I never pick one favorable branch.

I preserve finite origin/function/stack/state caps and report LIMIT/MEMORY or
UNRESOLVED explicitly. Calls, recursive summaries, globals and reentry retain
the existing conservative fixed point. Unknown array origins or unsupported
children cannot become a guessed portable shape. Successful shape analysis
still does not remove runtime receiver, lifecycle, allocation or ownership
checks. LLVM and Wasm consume the same predicate.

## My dependency order and acceptance

1. Correct and qualify VM slice operand cleanup/result allocation (task1e89).
2. Qualify private exact-capacity literal/slice primitives, scalar ABI,
   transactional preparation and retained children, without opcode admission.
3. Extend shape analysis for count effects and derived copy origins; retain
   existing reports/refusals and explicit limits.
4. Add consuming LLVM lowering and deliberate shared admission, then actual
   VM/native LLVM/import-free Wasm gates (taskb702).

My ordinary controls cover empty and nonempty literals, counts around8/16,
source order, every portable packed pair and boxed leaf tag, exact float bits,
NUL/high-byte string children, duplicate child aliases, and rollback after a
partially retained batch. Slice controls include default non-integer bounds,
modulo2^32 endpoints, empty/reversed/clipped ranges, exact initial capacity,
mutation independence in both directions, and children surviving source/result
teardown in either order. I include mixed receiver-origin joins, repeated-site
copies, calls/globals/reentry and refusal of unsupported nested/packed shapes.

Deterministic failure controls follow defensive checks and cover storage,
descriptor growth and child retention without publishing output. Public entry
status tests require complete frame cleanup and retained prior globals. Native
sanitizers, finite-memory Wasm reclamation, zero imports, packaged ABI checks,
existing mutable/string/core/profile tests and old-output refusal remain
required. This bounded result cannot close either full parent or release scope.

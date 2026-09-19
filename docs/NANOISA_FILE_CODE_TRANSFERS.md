# I check File instruction flow before granting body facts

I continue task546cf after my private declaration/decode/DAG preparation. This
is a design checkpoint for independent source review before implementation.
My [parent contract](NANOISA_FILE_CODE_AUTHORITY.md) retains loops, indirect
calls, richer borrowed calls, hosted entry/initializers, matched VM/native
cleanup, generated source and full File acceptance as required later work.
No selector or service handler changes in this design.

## My query and staged result

I introduce a separate private query taking the actual immutable NvmModule.
It calls preparation itself, never accepts a caller's prepared pointer as body
permission, and analyzes all declared functions in callee-before-caller order.
The result owns that preparation and copied per-function/per-instruction facts.
A successful result checks only the documented acyclic logical bodies. It keeps
runtime obligations explicit and is not hosted execution authority.

Per instruction I retain whether it is reachable, exact input/output stack
counts, decoded site identity, service/call obligations where present, and
cleanup/refinement/exit requirements. Every reachable return has a checked
logical exit. A call's callee-body fact is discharged only after the actual
callee analysis succeeds. I preserve binding, invocation, liveness, rights,
byte-domain, borrow, result-publication and cleanup requirements. I distinguish
checked body facts from remaining runtime masks rather than silently dropping
an obligation. Drops and ASSERT/trap cleanup get explicit site records even
though the older transfer API's drop count is not a location map.

Query failure leaves the caller's output untouched. No report borrows caller
CODE, metadata, name buffers or transient states. INVALID identifies malformed
or contradictory flow; UNRESOLVED identifies unsupported transfers or joins;
LIMIT and MEMORY remain distinct. No shared verifier recursion is introduced.

## My forward propagation

I start each function with the qualified declaration state's initialized formal
parameters and uninitialized remaining locals. A function's initial state defines
one identity family; all successor states are clones of that family. I process
its complete prepared topological instruction order once. Unreachable instructions
retain the preparation's structural/opcode/operand checks but do not acquire
runtime obligations or fabricate initialized values.

Each incoming edge is complete before its destination is processed. The first
incoming state becomes the destination state; later predecessors use the existing
transactional conservative join. Owner IDs/locations, borrow regions/references,
stack height and initialized facts must agree. Only compatible Result-arm facts
widen. Distinct fresh owners and incompatible cleanup histories remain unresolved;
I do not rename owners to make a join pass. I free a processed state after sending
its checked successor states, while retaining only copied report facts.

JMP forwards one state. Ordinary conditional branches consume exact BOOL and
check both successors; this milestone does not infer scalar constants. The
File Result branch uses the qualified two-output refinement operation, preserving
its allocation-failure atomicity. Known opposite arms remain unreachable. Both
CFG and conservative direct-call cycles are already refused by preparation,
including cycles in unreachable code; no bounded iteration substitutes for a
fixed-point proof.

## My exact transfers

The reviewed preparation inventory is an upper bound, not automatic typed
acceptance. Push/load/store/dup/pop and explicit owner move/store/drop, regions,
exclusive borrows, service calls and Result takes delegate to the qualified
state operations with actual function-relative byte offsets as sites.

Integer arithmetic and ordered comparisons require exact INT operands. Generic
EQ/NE accept equal exact INT or BOOL operands; logical operations require BOOL.
Signed typed I64 operators preserve those requirements. ASSERT consumes BOOL;
its normal successor is checked, while trap cleanup stays a pending runtime
requirement. I do not evaluate arithmetic or claim bounds/rights from scalar
facts. An unsupported scalar combination is not an inferred VOID fact.

Constructors use preparation's exact catalog identity, encoded variant and field
count, then the existing copyable constructor transition. File and OpenResult
cannot be manufactured structurally. Record AGG_GET uses the exact FileError or
ReadByte fields. UNION_FIELD and variant AGG_GET require a scalar Result, field0
and a known arm so its exact copyable payload is known. Unknown-arm projection
is unresolved in this milestone. UNION_TAG and AGG_TAG accept only scalar Results
and yield INT without refining another value. This matches the existing VM's
variant-only AGG_TAG boundary; ordinary records are not treated as variants.
Generic aggregate operations cannot inspect an affine OpenResult or File.

CALL uses mode0 parameters and the ordered stack suffix. CALL_REF fills exactly
one exclusive File formal from its explicit reference slot and uses values for
remaining mode0 parameters. I construct the exact per-parameter reference vector
and delegate alias/hold/consumption checks to the existing call transfer. Every
callee body is checked even if no reachable caller uses it. A returned owner may
receive a fresh symbolic caller identity; this is conservative and does not
claim a new runtime resource or erase actual liveness/generation requirements.

At RET I require exact declared result count/type, no leftover stack owners or
owned locals, no local reference/region escape, and preserved borrowed formal
ownership. Internal owner returns remain possible; the logical entry stays
scalar-only. Hosted flags, invocation ABI and the VM's implicit first-__init__
call require a later explicit conjunction over the actual module. This query
cannot dispose a context after publishing an owned public result.

## My storage and review gates

I extend preparation's accounting with exact report arrays, incoming-state
pointers, all simultaneously live states and transient clone/refinement peaks.
I use sizeof of the actual private state in the same translation unit, not a
copied estimate. Before refinement I reserve room for both clones, and before
every other state/report allocation I check the total16MiB bound. Failure frees
all predecessors, working states, report storage and preparation references.
Identity exhaustion and reference-count bounds retain the qualified API results.

Before execution I send the complete production diff for review. Fresh private
fixtures will cover real bytecode for both service outcomes; correct and wrong
helper bodies; borrows, moves and exact nominal values; known-arm unreachable
paths; same-family and divergent-owner joins; all returns and explicit cleanup
sites; allocation prefixes including second-refinement-clone failure; limits;
input/output lifetime; and unchanged public refusal. Actual bytecode is analyzed,
not executed. Existing File declaration/CODE preparation/provider controls remain
adjacent gates. Runtime, source, loops and indirect-call work stay open afterward.

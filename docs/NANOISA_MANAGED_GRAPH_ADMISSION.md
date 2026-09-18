# My generated managed array graph boundary

I continue task4070da262 after private graph core724, graph-origin query727,
and prepared runtime729 on canonical main `23294b37`. My prepared runtime
childc01 is complete; this contract covers the remaining generated integration.
I keep full aggregate488/managed51da open beyond this bounded array graph lane.

## My eligibility conjunction

I retain ordinary verification, the existing closed managed opcode/signature
whitelist, finite global/index checks, and refusal of imports, nominal layouts,
owned/passive contracts, captures and unsupported heap tags. No opcode allocation,
wire metadata, source frontend or host capability changes are part of this work.

For modules containing mutable array operations, I first run the existing leaf
query. Eligible leaf modules retain the existing entry/finish lowering and do not
pay for prepared graph workspace. Only a leaf UNRESOLVED result may fall back
to the graph query. INVALID, LIMIT and MEMORY remain failures; I do not retry a
failed resource budget through another analysis. The graph query must return
ELIGIBLE with its existing conservative calls/globals/reentry/alias summaries.
Unknown writes, deferred opcodes and incompatible packed coercions still refuse.
A shared selector returns leaf versus graph mode only on successful analysis,
so verifier and emitter cannot independently widen different boundaries. The
existing public leaf and non-admitting graph queries retain their meanings.

Graph mode admits only boxed arrays with already admitted scalar/string/array
children, including cycles, alongside fixed packed scalar arrays with the
existing finite coercion matrix. It combines possible-shape analysis with the
qualified counted-owner runtime and the safe-point discipline below. Origins
alone establish neither liveness nor memory safety. LLVM and Wasm select the
same mode and lifetime protocol; old closed scalar/literal profiles stay closed.

## My live owner audit

| Generated path | Owner at a possible collection boundary |
| --- | --- |
| Operand stack and locals | Every dynamic value has one counted owner; LOAD retains, STORE transfers, replaced slot releases. |
| DUP/SWAP/POP | DUP retains its second owner; SWAP transfers both; POP releases. |
| Persistent globals | LOAD retains; STORE transfers and releases the previous owner. Completed writes survive later errors and reentry. |
| CALL arguments | Caller pops without releasing and immediately transfers each owner into callee parameter locals. There is no collector between pop and parameter publication. Suspended caller roots remain counted. |
| CALL result | Callee removes its result owner before clearing locals. The returned SSA value remains owned until pushed in the caller; no collector runs during this transfer. Failed calls return no heap owner. |
| GET and POP result | GET retains the child; POP transfers its edge owner. This owner survives releasing the receiver before publication. |
| PUSH/SET/SLICE/string helpers | Synchronous helper invocation owns or borrows according to its existing adapter contract. No collection runs inside preparation, retain/publication/release, or while result ownership is pending. |
| ARR_LITERAL | All counted operands stay on the stack while scratch vectors borrow their bits; successful preparation creates retained edges before consuming originals. Failure leaves stack owners for common cleanup. |
| Return/error | Explicit and implicit return use existing common cleanup. Error cleanup releases all remaining stack/local owners; any returned temporary is released on failure. |
| Module initializer and entry | Initializer result and final scalar entry result are released before graph finish. Globals remain owned during finish collection. |

The collector computes external roots as real counts minus internal edges. It
therefore sees temporary and suspended-frame owners without compiler stack maps.
The table above is an implementation obligation, tested by actual generated
programs; the shape report is not a replacement for this audit.

## My generated safe points

For graph mode only, I emit a checked collection at the start of each potentially
allocating instruction, before any operands are popped or scratch is borrowed:

- ARR_NEW, ARR_PUSH, ARR_SET, ARR_LITERAL and ARR_SLICE;
- STR_SPLIT, STR_CONCAT, STR_SUBSTR, STR_REPLACE, STR_TRIM,
  STR_TO_LOWER, STR_TO_UPPER, STR_FROM_INT and STR_FROM_FLOAT;
- CAST_STRING and generic ADD, if otherwise eligible under the existing query.

Some listed operations are currently deferred by graph analysis; listing their
allocation behavior does not grant admission. I maintain an explicit opcode
predicate alongside lowering, including conservative cases such as prepared
boxed SET. I do not silently extend it when admitting a new allocating opcode.
Constants, retained reads, parsing and scalar arithmetic do not allocate managed
objects. Direct CALL transfers roots; callee allocation instructions collect.

Collection returns through `nms_module_graph_collect`; nonzero status branches
to the existing function error cleanup before executing the instruction. The
operand stack and locals still own every original input at that point. No
collection runs inside an allocator callback or after partially publishing an
operation. Prepared scratch makes collection allocation-free; subsequent object,
slot-table and replacement-workspace allocation may still fail transactionally
when live storage plus required old/new peak cannot fit.

Graph finish collects after frame cleanup even if there is no later allocation.
The initial deterministic pre-instruction schedule provides bounded-live cycle
reclamation across repeated allocating loops. I do not claim identical VM
collector timing or a throughput improvement.

## My entry acquisition and first-error protocol

The existing leaf wrapper treats every failed begin as unacquired. I cannot copy
that control flow for graph begin: preparation may return MEMORY after ordinary
begin has acquired an active instance. This is a required integration distinction,
not a reason to change leaf behavior.

For a valid graph instance, graph begin has these outcomes:

- OK: active entry and prepared workspace; run initializer and entry.
- MEMORY: active entry acquired, workspace preparation failed; call graph finish
  before returning the packed error, without invoking any generated function.
- BUSY or DISPOSED: no new entry acquired; return refusal without finishing or
  changing a suspended caller's status, roots or workspace.

I document and test that adapter protocol explicitly. Generated code does not
infer entry ownership merely from `active`, since BUSY also observes an active
caller. Initial preparation is the only allocation in this begin step and is
the only new post-acquisition failure. Existing first-error accumulation survives
all cleanup/collection calls. Initializer failure and ordinary function failure
both reach graph finish after their frame/result owners have been released.
Public trapping entry wrappers trap only after this cleanup; `nano_try_entry`
returns status without trapping. Disposal releases globals and the runtime only
when inactive, and stays terminal. Existing persistent global writes are not
rolled back by an error.

## My ordered implementation and acceptance

1. Add shared leaf/graph selection and direct verifier controls, preserving
   ordinary/leaf query results and all non-graph refusals. Keep graph mode
   unexposed by the public profile until its matching lowering is in place.
2. Connect checked pre-instruction collection, graph begin acquisition handling
   and graph finish to the existing counted-owner lowering. Review the combined
   production delta before claiming public graph acceptance.
3. Use ordinary valid bytecode through normal verification and public translation
   for paired VM/native LLVM/import-free Wasm results: nested aliases, duplicate
   children, self/mutual cycles, shallow slices, literal order, GET/POP retained
   results, nested mutation, calls/returned temporaries, branches/loops and
   persistent globals. Compare observable values and identity, not timing.
4. Qualify actual generated lifetimes under native sanitizers and finite-memory
   Wasm: many more allocated cycle bytes than memory capacity with bounded live
   roots, mixed strings/packed children, repeated entries/fresh instances,
   final-frame-only garbage and global-root survival. Exercise first preparation
   failure, descriptor/workspace growth failure, object/child-copy allocation
   failure, ordinary ASSERT/type/bounds failure and later successful reentry.
   Preserve the first status and verify cleanup/live counts without altering
   emitted eligibility or bypassing normal verification.
5. Rerun existing leaf runtime/target/profile and graph provenance controls,
   package generation/hash/import checks, and refusal/output-preservation tests
   for unknown or unsupported shapes. Update the former intentional nested-array
   refusal fixtures only for the now-admitted exact cases; preserve adjacent
   packed mismatch/unknown/nominal/import refusals.

I retain actual native/Wasm artifacts and source/tool pins. No historical crash
artifact is replayed. Runtime/private harness acceptance from729 supports this
work but does not stand in for the generated tests above. Completion of this
contract remains a bounded array graph acceptance, not full heap/aggregate,
platform, compiler-product or release completion.

# I execute checked ordinary records with counted managed owners

I propose `task_55677fd9ebc341a685590197ee621ed7` under aggregate488 and
managed51da. My baseline is canonical `7ebc9e9c` through779. This contract
precedes production and needs independent review. I leave the frozen forward
producer qualification tree unchanged. Forward authority240f does not extend
this prior-order runtime contract merely by transporting more declarations.

## My existing obligations and implementation points

The original aggregate488 task requires authoritative nominal layouts, shared
mutation/alias identity, heap-bearing fields, calls/globals/returns and repeated
instances with explicit lifetime/cycle policy. Its accepted private record2dc
checkpoint supplies storage and traversal, not generated execution. My merged
field-origin queryf2cf supplies finite exact field/origin facts, not lifetime.
This child connects those prerequisites to an actual executable subset; it
closes neither the full aggregate task nor the managed/runtime release parent.

`nvm_describe_managed_records` supplies owned layouts, ORDINARY authority and
record-ordinal/global-layout maps. `nvm_analyze_managed_records` already checks
normal verification, exact constructor counts, all possible receiver fields,
nominal writes and weak interprocedural/global/repeated-site origins. Its
record report retains numeric identities and owned field summaries. I consume
its existing bounded verdict without broadening the analysis to unknown heaps.

`managed_strings.c` already implements private `nms_bind_records`,
`nms_record_create`, retained GET, borrowed SET, stable record slot identity,
iterative release and mixed-edge collection. In contrast, generated module
retain/release currently recognize only strings and arrays; no generated entry
binds record descriptors. `verifier.c` refuses nominal metadata in all closed
profiles, and `nvm2llvm.c` selects array lifetime and lowers no record operations.
These are the concrete connections I implement after review.

## My shared selection and exact boundary

I add one checked heap-selection/plan API beside the existing analysis APIs,
used by both managed-profile verification and LLVM emission before output.
It has distinct ordinary leaf, array-graph and checked ordinary-record modes.
Modules without record facts retain the existing array selector and verdicts.
For record-bearing modules I require successful normal verification,
DESCRIBED **and ORDINARY** descriptor authority, and successful
`nvm_analyze_managed_records`. I do not retry weaker modes after any record
failure. Existing leaf/graph query APIs and their public report layouts remain
unchanged. The selected plan owns any record descriptor/report state; every
failure leaves caller output unchanged and frees private allocations.

I preserve all existing analysis caps and distinguish its INVALID, UNRESOLVED,
LIMIT and MEMORY outcomes. Legacy ambiguous authority-allocation failure stays
UNRESOLVED. A successful origin report is one necessary conjunct, alongside
supported opcode/signature checks, descriptors and the matched runtime path.
I call ordinary `nvm_verify`, never recursively call profile verification from
analysis. No storage permission follows from a tag, shape or missing metadata.

Only the managed closed profile gains the selected record lane. Scalar and
literal profiles continue refusing nominal modules. I permit STRUCT helper
parameters/results only in the selected record mode; entry remains INT/BOOL,
zero-argument, and initializers/calls retain the existing signature/arity rules.
No captures, host/import/module references, passive or affine/reference
contracts become eligible. VOID local metadata is never an unknown substitute.

The declarations remain COMPLETE ordinary int/U8/float/bool/string/prior-record
fields. Explicit empty records are distinct from missing layouts. Same-shaped
records keep distinct identities. Missing/UNKNOWN/resource/foreign/generic/
forward authority, array-valued fields, wrong nominal writes and unresolved
receivers remain refused. Existing arrays can coexist only when the record
query establishes their existing portable leaf/nested-array behavior without
record children. I preserve its prohibition on both mixed edge directions.
Union/tuple operations remain unsupported. Interleaved retained enum entries
must not confuse record ordinals with global layout indices; enum scalar
behavior remains the already admitted contract.

## My immutable descriptors and module lifecycle

I emit an immutable descriptor table in record-ordinal order, each entry holding
the exact global layout index and field count from the checked plan. I do not
sort or renumber executable ordinals. Its lifetime spans the generated module
instance through terminal disposal. I assert the two-u32 descriptor layout and
TAG_STRUCT/NMS_RECORD_TAG8 identity in native and wasm32 packaging.

A record-aware begin adapter initializes a fresh module runtime, binds the
immutable table **while inactive and before any dynamic allocation**, then
acquires the entry and prepares the existing collection workspace. Calling
`nms_bind_records` after ordinary begin would return BUSY; I do not do that.
Binding occurs exactly once; later entries reuse the same table and persistent
globals. Rejected nested begin leaves the active outer entry and first error
unchanged. DISPOSED never reinitializes the instance.

Pre-acquisition binding/lifecycle refusal returns without finish. Successful
acquisition followed by workspace MEMORY follows the existing acquired-error
finish path. Generated initialization and entry do not run after any failed
begin. The adapter exposes an unambiguous acquired flag/status contract rather
than assuming every nonzero result is unacquired. I preserve numeric public
status values and the packed `nano_try_entry` result convention.

I extend module retain/release to exact record handles. All existing stack,
local, global, argument/result and pending FrameOutput roots then retain or
transfer the same stable object identity. No C stack aggregate or copied record
stands in for VM shared storage. Descriptor storage is immutable data, not a
counted heap root. Cleanup, error return, initializer results and terminal
global disposal all release record owners under the same first-error policy.

## My instruction and adapter semantics

| Instruction | Matched operation and ownership |
| --- | --- |
| STRUCT_NEW | Resolve its per-kind record ordinal. Only an explicitly empty declared record is eligible because the VM creates zero fields. Allocate a fresh handle; no default fields are invented. |
| STRUCT_LITERAL / AGG_PACK | Admit only AGG_RECORD with neutral variant and exact retained field count. Borrow ordered operands while they remain counted on the stack, create retained field edges, then consume each original exactly once. Failure leaves original stack owners for common cleanup. |
| STRUCT_GET / AGG_GET | Check receiver family and field index, retain the returned scalar/string/record value before consuming its receiver, and publish only a complete tagged result. |
| STRUCT_SET / AGG_SET | Mutate the existing receiver. Retain the new field before publishing it and releasing the old edge; consume the incoming value's original owner once, then transfer the unchanged receiver owner to the result. Same-object/same-field aliases remain valid. |

Constructor field vectors use the existing bounded payload/tag scratch pattern,
sized by verified stack depth, rather than FrameOutput's three temporary slots.
The module adapter accepts parallel scalar arrays and builds a bounded private
`NmsValue` vector; it never assumes LLVM `%V` and C struct padding are identical.
The 256-slot analysis cap bounds this staging. I use scalar handles/status and
out-parameters across embedded target IR, avoiding native/wasm aggregate-return
ABI assumptions. No scratch points into a relocatable slot table. Count-zero
construction requires no dereference of absent field data.

The core retains input edges transactionally. On constructor failure I release
any privately retained prefix and unpublished storage, preserve outputs and
leave caller-owned operands intact. If release/status fails after a result is
created, I release that result and enter common cleanup rather than pushing it.
GET's retained temporary survives receiver destruction. SET retains before
replacing an aliased child; after success only the receiver is returned, not an
extra child owner. SET performs no allocation in this fixed-field subset.

VM error distinctions matter: STRUCT_GET/SET use TYPE for a non-record receiver
and BOUNDS for an absent field; AGG_GET/SET use BOUNDS for an unavailable field,
including a non-aggregate receiver. My adapters preserve that distinction rather
than forwarding private TYPE indiscriminately. Known invalid field alternatives
are already ineligible, but admitted scalar/record joins still need runtime
receiver checks. Allocation failure is MEMORY; first error survives subsequent
operand cleanup. No helper traps while counted roots remain live.

## My common operations and collection audit

Record admission also reaches existing scalar/generic consumers. I audit and
qualify them, not just the new switch cases: DUP/LOAD retain; STORE replaces;
POP, type tests, truthiness and comparisons release consumed owners; CALL/RET
transfer once. EQ/NE compare live record identity, same-tag ordering returns the
VM's zero comparison result, mixed-tag ordering follows tag order, and records
are truthy. CAST_INT/FLOAT yield zero and CAST_STRING yields an empty string as
the VM specifies, while consuming the record owner. Existing array-only cast
release/sanitization branches must include records. Wrong numeric/string
operations retain their ordinary checked runtime error and complete cleanup.

I use prepared graph lifetime mode for every admitted record module, even a
record-only module whose prior-order field schema cannot form a record cycle.
This keeps coexisting qualified nested arrays collectible and reuses the proven
counted-root protocol. Add STRUCT_NEW, STRUCT_LITERAL and AGG_RECORD PACK to the
explicit pre-allocation safe-point predicate. Collection runs before popping
operands or borrowing literal scratch, never during a partially committed write.
Suspended caller frames, argument vectors, returned temporaries and globals are
real counted owners. Finish collects only after frame cleanup and preserves
live globals across successful or failed entry. Existing module disposal is
terminal and removes all roots.

This schedule reclaims dead storage during repeated bounded-live programs; it
does not claim VM-identical collection timing. Allocation/table/replacement-
workspace failures retain transactional state, including old-plus-new peak
requirements. I reuse the reclaiming import-free wasm allocator and native
allocator, with no hidden host imports or grow-only arena.

## My review and acceptance order

1. Independently review private module descriptor/record adapters and root-tag
   extension, still without public record admission. Regenerate native64 and
   wasm32 embedded IR with source/generator hashes and check target ABI, no
   imports and no test hooks. Audit explicit source/link lists; prefer existing
   translation units. Qualify private adapters, begin acquisition and errors.
2. Review shared record selection and matching lowering together. No public
   intermediate step merely relaxes nominal metadata or STRUCT signature guards.
   Preserve unknown/resource/forward/array-field and all old profile refusals.
3. Freeze ordinary positive VM/native LLVM/import-free Wasm fixtures: scalar and
   string leaves, exact float bits, empty records, constructors beyond three
   operands, distinct same-shaped/interleaved identities, nested aliases,
   GET after temporary-owner cleanup, SET replacement/alias visibility, helper
   argument reordering/returns, branches, loops, globals and repeated entries.
   Include independent qualified arrays without mixed record edges. Use actual
   generated output with Node/Wasmtime and native sanitizers, not only core APIs.
4. Qualify deterministic allocation/table/workspace failures through the shared
   portable allocation harness, retained globals across failure/reentry, returned
   temporaries, first-error cleanup, nested-entry rejection and terminal disposal.
   Distinguish fresh modules from repeated entry on one instance. Require bounded
   live-storage churn with collector safe points and preserved output files on
   translation refusal. Do not execute old failed artifacts.
5. Run affected descriptor/field-origin, existing leaf/graph analysis, private
   runtime/package and generated managed regression gates; actual module host
   linkage if source lists change. Record frozen pins and first outcomes, seek
   independent production/evidence review, and integrate canonically before this
   child closes. No new source bootstrap claim follows from runtime-only checks.

Mixed record-array field/element authority and provenance, forward/generic/
imported nominal families, further aggregates and declared host capabilities
remain required subsequent work under the existing parents. This bounded first
record execution path neither drops those clauses nor invents new parent
completion gates. Darwin parser/evaluator historical incidents retain their
separate task/evidence status.

## My first private adapter checkpoint

`nms_module_record_begin` returns low32 status and high32 acquisition (0 or1),
distinct from public try-entry's low32 result/high32 status. It initializes and
binds descriptors before begin, preserves nested-call first errors, and requires
identical immutable descriptor/literal table pointers and counts on reentry.
Failed inactive binding can be corrected before any allocation; preparation
failure after begin returns acquired1 for graph finish. No selector calls this
adapter yet.

`nms_module_record_literal` borrows parallel payload/tag arrays into at most256
private `NmsValue` entries. Empty NEW uses count0. The get/set adapters borrow
receiver and value owners; GET publishes retained scalar outputs only on success.
The aggregate selector is exactly0 (STRUCT) or1 (AGG). Receiver validation maps
wrong-kind TYPE to BOUNDS only for AGG; a later value-retain error is not
misclassified as a receiver error. Module retain/release now count record tags.

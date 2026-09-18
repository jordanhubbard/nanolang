# My non-admitting ordinary record field origins

I propose task_f2cf9288ef53481782240698b3073048 under aggregate488, following
record descriptor/storage2dc and the bounded ordinary authority15f checkpoints.
My baseline is canonical `e5f2f6dc687135dcb82730ef3f15f86261b370c3` through772.
I change no production until this contract is reviewed. This child supplies
shape/origin evidence only; profile selection, runtime allocation, generated
ownership and record opcode admission are later reviewed work.

## My existing architecture

`src/nanoisa/managed_array_shapes.c` already implements a finite monotone
interprocedural analysis. Its `Value` contains a tag set, allocation-origin
bitset and explicit unknown. `Function` stores instruction-entry locals/stack,
return summaries and a work queue; `Analysis` stores weak global and array-child
summaries. `seed`, `join`, `walk` and `final_check` propagate calls and recursive
cycles to a fixed point. Repeated allocations at one site share a weak summary.
Unused functions receive unknown arguments, so absence of a known caller is
not permission to infer a parameter's identity. Globals retain all observed
values, including initial VOID, across entry/initializer effects and reentry.

The existing APIs reject nominal/layout/ownership modules before walking them.
Their separate leaf/graph reports and `nvm_select_managed_array_mode` are public
consumers of the current narrow verdicts. I keep those APIs, output layouts,
limits and execution selection unchanged. I add a separately named record
analysis query/report using the same internal transfer engine in an explicit
new mode; I do not remove the old preflight merely to make records pass.

`nvm_describe_managed_records` already owns retained layouts plus explicit
record-ordinal/global-layout maps and supplies an authority enum. I require
DESCRIBED **and ORDINARY**, not descriptive UNKNOWN. The underlying
`nvm_ownership_contracts_validate` must accept the complete payload without
requiring affine/reference execution. Resource/ref declarations are unresolved
for this child even if one particular record looks ordinary. No new wire data,
producer metadata, flags or opcode allocation is needed.

## My bounded declarations and transfers

I use the already qualified COMPLETE ordinary scalar/string/prior-record field
schema. Every record field must have an exact supported scalar/string tag or an
authoritative nested record identity. I preserve distinct same-shaped nominal
records. Unsupported array-field authority, generic/import/foreign/forward
nominal families stay unresolved; source producer expansion remains15f's work.
I do not infer missing identity from a spelling, shape or VOID placeholder.

I retain the old closed-module constraints: normal verifier first, zero-argument
entry and first initializer, no imports/module refs/captures/passive contracts,
at most one result, explicit supported opcode whitelist. The new query permits
the checked ordinary ownership/layout data needed for records. It reuses the
existing scalar/string and array transfers; unrelated unknown opcodes remain
unresolved. Array and record origins share one internal disjoint tagged site
space in this mode, so their numeric bits can never be confused. Old leaf/graph
report numbering remains unchanged outside this mode.

My added transfers are limited to:

| Instruction | Analysis obligation |
| --- | --- |
| STRUCT_NEW | The VM calls `vm_struct_new(def_idx, 0)`: it creates **zero fields**, not declared defaults. I resolve the record ordinal and accept only an explicitly empty retained record in this first query. Nonempty declarations remain unresolved; I never invent VOID fields. |
| STRUCT_LITERAL | Resolve its per-kind record ordinal, require the encoded field count to equal the retained count, and read that many ordered stack values. No fixed three-operand shortcut. |
| AGG_PACK | Only AGG_RECORD, with the existing per-kind record ordinal, neutral record variant and exact retained field count. Variant/tuple packing remains unresolved. |
| STRUCT_GET / AGG_GET | For every possible successful record receiver, resolve the immediate field against its actual record identity and join that field's full value summary. AGG_GET does not silently admit tuple/union receivers. |
| STRUCT_SET / AGG_SET | Check the incoming value for each possible record receiver's exact field contract, weakly add its tags/origins/unknown to that field, and return the unchanged receiver origin. Runtime SET is shared mutation, not a copied record. |

I explicitly require field-index validity for every possible record origin in
this bounded query; an out-of-range origin is UNRESOLVED, not a fabricated
successful read. This is an analysis eligibility boundary, not a change to VM
error behavior. Non-record receiver alternatives are recorded as runtime tag
obligations only when all possible successful record alternatives are known;
unknown heap identity remains unresolved. A subsequent lowering contract still
needs the exact existing STRUCT versus AGG failure/status behavior.

Constructor values and SET writes must match the declared scalar/string tag or
the nested record's exact global layout identity for every possible origin.
Unknown values or missing record origins remain unresolved. I model no implicit
coercion, copied value, narrowed phi or erased write. New record constructors
begin with their actual ordered operands, not hypothetical defaults. Empty
record construction contributes no field values. Wrong-tag source programs are
not made valid by metadata; I conservatively refuse their eligibility.

Existing array operations may coexist in the module with existing portable
scalar/string/nested-array contents. I keep record values in array writes and
array values in record fields unresolved in this first child, preserving their
explicit later aggregate/authority obligations. This bounds the first consumer
without weakening the full required record-array integration. Arrays' own
origins, copied slice origins and packed coercion matrix stay unchanged.

## My lattice and publication

Each record allocation origin stores its function/byte offset, record ordinal,
global layout index and ordered per-field `Value` summaries. A GET retains the
full nested-origin alternatives, not only TAG_STRUCT. Aliased SET, helper calls,
returned records and global storage propagate into the same summaries. I weakly
join all instances from one allocation site, including repeated loop iterations
and recursive calls; a later write never erases an earlier possible child.

I keep monotone tag/origin/unknown unions, function result summaries, caller
argument seeding, instruction states and global effects. An initially bottom
constructor/GET result must not fabricate an origin or unknown before a valid
incoming state arrives. I run all propagation to convergence before checking
field writes and publishing results. This establishes bounded eligibility, not
path-sensitive ownership, object lifetime, absence of runtime failure or a
proof of program behavior.

I retain existing caps of256 functions,256 locals/stack/globals,65536 decoded
instructions and1048576 abstract cells. This mode permits at most64 combined
array/record allocation origins; I check before shifts. I cap the sum of
per-origin field-summary cells at65536 and include instruction and field storage
in widened allocation/size checks. Limits return LIMIT, never truncated facts.
The descriptor plan's own256-layout/65536-field limits remain unchanged.

The new report owns its origin descriptions and flattened field summaries;
it retains numeric nominal identities without borrowed source pointers. It
requires a dedicated free function. All invalid/unresolved/limit/allocation
exits free the plan, decoded functions and temporary summaries, preserve the
borrowed module and leave the caller's output pointer unchanged. Report allocation
is the final publication step. I preserve the existing descriptor validator's
conservative UNRESOLVED result where its legacy error code cannot distinguish
invalid authority from allocation failure; I do not falsely label that MEMORY.
New allocations made by this analysis report MEMORY precisely.

## My acceptance order

1. Review the production checkpoint without changing verifier profiles or
   LLVM/Wasm lowering. Audit every explicit build source list if a new host
   translation unit is introduced; keeping the engine in its existing unit
   avoids an unnecessary new linkage boundary.
2. Qualify ordinary verified records with exact constructor counts, empty
   STRUCT_NEW, interleaved global/per-kind layouts, distinct nominal identities,
   nested GET/SET aliases, reordered helper arguments, returns, globals/reentry,
   branch joins, loops and recursive call summaries. Repeated allocation sites
   must retain all possible field origins. Use fresh ordinary VM controls to
   anchor successful read/mutation results; no old failed-artifact execution.
3. Qualify UNKNOWN/resource/foreign/unsupported authority, unused unknown
   parameters, wrong nominal writes, count/index boundaries, deferred transfers,
   origin/field/state limits and deterministic allocation failures. Preserve
   output/module state. Run analysis controls under GCC/Clang sanitizers.
4. Require unchanged old leaf/graph verdicts, selector results and managed array
   suites. Actual LLVM/Wasm record emission still refuses and preserves prior
   output; a successful new report must not widen the public profile.
5. Record exact evidence and canonical integration before this child closes.
   Draft a separate matched record lowering/lifetime contract next: counted
   literal roots, retained GET/shared SET, actual adapters, collection safe
   points and native/Wasm execution. Mixed record-array fields/contents and
   broader15f authority remain explicit later requirements, not closed here.

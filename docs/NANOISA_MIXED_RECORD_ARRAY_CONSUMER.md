# I prove record-field array origins before mixed managed execution

I continue task_f64151e8a1c14585909bc44e801f5a4b under f36b/15f/488 from
canonical7bba8c798, actual PR916. This is a design checkpoint, not executable
acceptance. My complete mixed declaration projection31c is qualified; it does
not establish instruction flow or lifetime. I preserve all full5.1 obligations.

## My next concrete dependency

My smallest missing consumer is exact ARRAY-field provenance in the managed
record origin analysis. I first add a separate non-admitting query, then qualify
the existing counted runtime with those field shapes. I do not enable a public
selector by replacing its declaration decoder alone.

My current `managed_array_shapes.c` has three independent barriers:
`prepare_records` requires the old ordinary declaration validator and excludes
ARRAY fields; `record_value_matches` knows scalar/string and exact record
origins only; `analyze` calls the old verifier and rejects every union table.
`nvm_select_managed_heap` feeds the closed-profile verifier and `nvm2llvm`.
Changing that shared selector would affect generated native LLVM and Wasm,
not merely an internal query.

My `managed_strings.c` counted core already validates ARRAY-valued NmsValues,
retains them in `nms_record_create`, retains GET results and retains replacement
values before releasing old fields in SET. Its collector traverses records and
boxed arrays. These are relevant existing primitives, not evidence that new
generated graphs have correct roots. Its record descriptors contain ordinal,
global-layout identity and field count, not element contracts. Declaration and
flow proof must precede execution; the runtime cannot recover them from shape.

## My first query and owned report

I propose source-private `managed_record_array_origins.h`, an opaque
`NvmRecordArrayOrigins`, and:

```
NvmArrayEligibilityResult nvm_analyze_record_array_origins(
    const NvmModule *, NvmRecordArrayOrigins **);
void nvm_record_array_origins_free(NvmRecordArrayOrigins *);
```

I reuse existing status meanings: ELIGIBLE is flow evidence only; UNRESOLVED,
INVALID, LIMIT and MEMORY never publish a report. Exact accessor C structs and
signatures must be included in the production checkpoint for review. Getters
copy counts, origin identities, field value summaries and required element tags;
none returns a source pointer or a public executable-eligible flag. The report
owns the declaration plan, global-layout/per-kind maps and numeric origin facts.
It survives destruction of the module, strings, instructions and all input
metadata. Free(NULL) is harmless; every failed getter preserves all output bytes.
Old array, graph and record APIs retain their modes, layouts and decisions.

I prepare the actual complete mixed declaration plan once. Every union and array
extension is validated before any flow facts escape. Record maps preserve exact
global layout indices even when scalar unions precede/interleave records.
Equal-shaped nominal records are not interchangeable. My first query permits
validated scalar/string union declarations to coexist, but no executed union
value, constructor, MATCH_TAG, union projection or union signature. Every function,
including unused helpers, must satisfy this instruction boundary. Full union
flow and generated union lifetime remain the next required mixed-graph step.

I initially support ordinary record DAG fields containing INT/U8/FLOAT/BOOL/
STRING, exact nested ordinary records, and flat arrays of those five element
tags. Resource flags, borrowed/reference descriptors, service/passive/imported
contracts, unknown/incomplete authority and nested ARRAY/STRUCT element rows
remain unresolved. I never infer ordinary status from a missing RESOURCE bit.
Service presence retains priority at every eventual public admission boundary.

## My instruction validation prerequisite

The old public verifier intentionally refuses these ARRAY_FIELDS envelopes. I
cannot call it, ignore its result, strip ownership bytes, clear union counts, or
verify a rewritten module and call that proof of the original.

Before implementing the new flow query, I require a reviewed source checkpoint
that factors the existing structural instruction/function validation into a
private entry with an explicit declaration-plan mode. That entry receives only
a freshly prepared plan for the same immutable module, checks the original
header, sections, strings, code ranges, decoded operands, call/global/local
indices, CFG successors, stack depths/maxima, arities and signature tags, and
returns copied decoded/stack facts. It must preserve all old public entry
behavior. It cannot accept a caller-provided trusted bit or recursively call
the new origin query. A plan is not rebound to another module by pointer or
shape equality. Its borrow ends with this synchronous preparation.

The first production review must enumerate exactly which existing checks move
and how each old wrapper invokes them; no blanket verifier bypass is authorized
by this design. Unknown instructions and unsupported indirect/captured calls
refuse. General/linked/function/bridge verification and serialization continue
their old refusals until the later matched admission checkpoint.

## My finite origin model and exact element constraints

I reuse the existing monotone Value lattice: tag bits, a64-bit allocation-origin
set and unknown. Origins identify function plus instruction byte offset, not a
particular dynamic allocation. Record origins carry exact global layout and
record ordinal. Array origins carry declared element tag, packed policy and full
content summary. Each record-origin field has a weak Value summary; repeated
instances at one site and all writes join rather than erase previous facts.

I preserve the existing explicit scalar/array transfer whitelist plus record
STRUCT_NEW/LITERAL/GET/SET and AGG_RECORD PACK/GET/SET. STRUCT_NEW remains accepted
only for zero-field records. Constructors require exact field count and source
stack order; a record PACK requires neutral variant. Unknown aggregate kinds
refuse. LOAD/DUP preserve identity; STORE replaces the abstract slot while weak
heap/global summaries retain all relevant prior instances. Direct call arguments,
returns, globals, branches and backedges participate in the same fixed point.
Unused helpers are seeded with unknown arguments after reachable convergence;
unknown cannot become a fabricated scalar or accepted empty origin set.

For an ARRAY field of declared element T, every possible successful constructor
or field replacement must have exact ARRAY tag, nonempty known array origins,
and each origin's declared element T. I collect a required-element mask on each
such origin. More than one distinct required element tag refuses. Every write
through every alias to that origin, including an alias obtained before field
installation, must carry exactly T. I check ARR_LITERAL initial values, ARR_PUSH
and ARR_SET; STR_SPLIT establishes exact STRING. Empty ARR_NEW requires explicit
T and is not inferred from later uses. Boxed contents must have no unknown or
heap-child alternatives. A packed representation's numeric coercion permission
is not proof of a different source element declaration: the new constrained
origin requires exact tags even where the old unconstrained array profile can
perform a qualified coercion. Old modes retain their coercion matrix.

I compute all summaries and field-derived constraints to convergence before
checking writes. I do not validate only the constructor-time snapshot. Field GET
exports all array origins. SET returns the original record origin; replacement
adds the new possible field origins without invalidating existing aliases to the
old array. Slice/copy receives a distinct allocation origin with the same declared
element contract and copied contents; subsequent mutations of the source do not
become writes to the copy. Repeated copies at one site remain weakly summarized.
GET/POP retain VOID for bounds outcomes. Optional values cannot silently satisfy
an exact field or element tag. Actual wrong receiver/index and bounds failures
remain runtime semantics; every successful heap receiver alternative must have
known origins, and the report records the remaining runtime tag obligations.

I retain256 functions/locals/stack/globals,65536 decoded instructions,64 combined
array/record origins,65536 field-summary cells and1048576 total abstract cells.
The declaration plan retains its own16MiB/1048576-step limits. The new query has
a64MiB simultaneous allocation ceiling including that plan, decoded functions,
queues, states, fields, temporary/report overlap and all explicit scratch. I
charge checked arithmetic before allocation. I cap cumulative transfers plus
visited origin/field/cell operations at16777216; no uncharged repeated scan is
allowed. Finite bit-union joins terminate; the work cap gives an independent
operational bound. Exhaustion returns LIMIT, never a partial report. New
allocation failures return MEMORY; ambiguous legacy errors keep their existing
conservative classification. Each failure frees every prefix.

## My counted runtime acceptance before activation

I first use the existing private core/module ABI without broadening public
admission. A record R retains array A while outside aliases remain live. I mutate
A through both the field and outside alias, replace R's field with distinct B,
then independently mutate/grow A and B and consume every root. Nested records,
calls/results, globals across repeated entry, distinct equal-shaped declarations
and copies must preserve identity. Flat element constraints do not permit
record-array cycles yet; I test existing nested-array collector neighbors without
claiming the new schema accepts recursive graphs.

Every live stack/local/global/argument/result and temporary owns a counted root.
Constructors collect before removing operands; SET acquires the new edge before
releasing the old; GET publishes a retained result. Return moves the result
before frame cleanup. Failure preserves the first error and releases every
partial staging owner; global roots survive until their documented disposal.
Collection never runs inside a partially committed retain/publication/release
transaction. Descriptor tables outlive all records and cannot be rebound during
an active instance. Busy/reentry and output-sentinel behavior remain unchanged.

I require allocation-prefix failures with fresh recovery, descriptor/workspace
growth, retained old field on failed replacement, root accounting before and
after collection, finite-memory repeated creation/destruction, assertion/bounds
failure cleanup and terminal disposal. Native sanitizer and import-free wasm32
private harnesses must exercise the actual existing core adapters. If a primitive
fails, I record and separately review its repair rather than expanding this
query patch. No VM/native/Wasm executable admission follows from a harness pass.

## My later shared admission and cross-consumer checklist

| Consumer | Required separately reviewed action |
|---|---|
| Public ownership/layout/variant accessors | Preserve current mixed refusal until a complete-plan consumer replaces it; never expose one half before validating the other. |
| General/function/linked verifier and max-stack/serialization | Use original-module complete facts plus flow, preserve service-first decisions and failure-atomic output; malformed/limit/memory failures cannot fall back. |
| VM and generated native C | Check exact admitted field origins and existing aggregate/array behavior; retain general VM numeric semantics outside this profile and paired failure/lifetime evidence. |
| Closed managed profile and nvm2llvm | Select one fresh complete owned plan; add no independent guessed schema. Emit exact descriptor maps, roots and allocating-instruction safe points. |
| Native LLVM and Wasm | Qualify generated program behavior before/after LLVM optimization and import-free Wasm instance cleanup, with exact runtime package/provider closure. |
| Converter/linker/bridges | Preserve metadata bytes/nominal remaps or retain checked refusal; no lossy v3 downgrade. |
| C seed/Stage1/Stage2/NanoVirt | Later paired publication from resolved source type facts, original program plus all selected shadows; no literal-content authority or new parser spelling in this child. |

The activation proposal must audit every changed call path before any old refusal
is relaxed. It must enumerate supported opcodes/signatures and exact status/output
behavior. Fresh declaration+flow+lifetime+consumer coverage is a conjunction;
none is a replacement for another. There is no new wire version or runtime ABI
reservation in this checkpoint.

## My ordered review and qualification

1. Review this design and the private structural-validation factoring contract.
2. Review complete query source and exact immutable accessor ABI, then fixtures.
3. Qualify all five element tags, forward records with interleaved unused unions,
   every alias/call/global/join/repeated-site case and independent malformed halves.
   Add wrong nominal origins, conflicting field constraints, optional writes,
   slice identity, unknown unused helpers and all old-mode controls. Exercise
   actual boundary/next counts, work/memory caps, every measured allocation prefix,
   copied-input destruction and complete sentinel preservation on both hosts.
4. Review and qualify existing counted core/module native+Wasm harnesses above.
5. Review the full shared admission/lowering delta and generated VM/native-C/
   LLVM/Wasm differential fixtures before any new executable selection. Preserve
   original program/shadow corpora and installed package/consumer refusal gates.
6. Separately review paired source publication, fresh bootstrap and full unchanged
   product acceptance. No parent closes merely because this query or harness passes.

My required continuation remains executable scalar/string unions with exact
variant/value refinement, nested arrays and arrays of exact records, recursive
nominal graphs and cycle collection, generic/imported nominal identity, mixed
resource/ordinary restrictions, tuples/maps/callables and richer applicable
language shapes. These remain full5.1 requirements under15f/488. File source and
peer affine source lanes remain separate; this design changes neither.

## My prerequisite extraction checkpoint

I first extract three existing static helpers without adding a query entry or
changing any caller's policy. This small checkpoint does not implement the new
mode, budget or report yet. I keep the original function bodies, diagnostics,
allocation and output behavior intact, including the existing max-stack write
before the type pass; the future new report must stage that output privately.

| Existing check | Extracted helper and unchanged caller order |
|---|---|
| NULL/service/code pointer; main index; function code ranges, overlap, names, result/parameter tags, result count and arity/local relation | `verify_module_ranges`, called first by `verify_structure_checked` after its original output initialization. |
| Complete ownership validation, affine analysis/owned admission, retained layout validation | Remain in `verify_structure_checked`, between ranges and contracts, with the existing `mixed_composed` private callers unchanged. |
| Passive and callback contracts; import names/kind/path/signature/parameter bounds | `verify_module_contracts_and_imports`, after retained-layout validation. |
| Decode/boundaries; every operand switch case including owned-transfer refusal, branch/handler/match targets, direct/tail/linked calls, closures/function refs, strings/imports/locals, aggregate/type operands and primary-plane guard; stack height/ownership balance; type pass; decoded cleanup | `verify_function_body`, called only after unchanged service/owner-array/Samples routing, module validation, function-index validation and admitted-owner shortcut. |

All general/function/max-stack/linked wrappers retain `verify_function_impl`;
affine/owned paths retain `verify_structure`; Samples and owner-array private
preparations retain their existing `verify_structure_checked` calls. No external
header or caller can invoke the extracted static helpers. The historical
`mixed_composed` bool is not a new permission mechanism.

The next checkpoint must introduce a synchronous original-module preparation,
constructing its own fresh declaration plan and bounded decoded/stack report.
It cannot export or accept that bool. It must reject unsupported transfers before
calling a common body that could invoke affine analysis, account decode capacity
and byte-boundary arrays plus stack/type scratch and publication overlap, and
include all scan/work counters. That complete private entry still requires
production review and fixture review before execution. This extraction alone
is not sufficient to call the new grammar structurally prepared.

I integrate actual PR919 canonicalca3779e08 in this working branch after the
extraction checkpoint. Only additive roadmap tails conflict; both histories
remain. My verifier source is byte-identical to242ebc8cf, and the incoming
constructor/service transaction and selector evidence stay independent. No
new build or executable acceptance is claimed by this integration.

## My complete private production checkpoint

I now prepare the new path without invoking an old executable selector. The
public source-private header `managed_record_array_origins.h` defines the opaque
report, exact copied count/origin/field/element-mask structs and declaration
getters. The separate internal structure header is construction machinery, not
an installed API: the sole origin entry creates a zeroed budget, calls preparation
on the original module, and never accepts a previously returned structure or
caller declaration plan. No reusable trust bit is introduced.

Preparation checks closed-host boundaries before dispatch, validates counted
string storage, in-memory header, metadata, function/range/signature limits and
all instruction opcodes. It calls the extracted original ranges and contract/
import checks. Its own fresh complete declaration plan replaces only the old
ownership/layout eligibility route for this private preparation. It does not
strip or edit any module field. It then invokes the same decoded operand switch,
stack-height/ownership-balance walk and type rules on every original function.
Decoded records carry no source pointers for the supported direct-call subset;
they transfer into analysis once, then are freed before report publication ends.
No public verifier, profile, converter, emitter or VM calls the new query.

The old function body passes NULL accounting and no kept-decoded output, preserving
its output order and optional type-pass allocation behavior. Only the new bounded
type entry turns a type-workspace allocation failure into MEMORY. Both entries
share the same type rules and transfer/join loop. The private preflight refuses
unsupported opcodes before shared operand validation could reach affine analysis.
No foreign, indirect, union-value, effect, owned or borrowed opcode is admitted.

I found an inherited static empty-body precondition: the old height walk skips
an empty terminal node, while the managed origin walk expects a value for a
nonvoid return. I record task_f4d020187bf34faca52e761cbb299dde without claiming a
runtime observation. My new mode explicitly rejects empty nonvoid functions;
I also check exact depth against the declared result count at each implicit and
explicit result consumption in the new origin walk. This covers nonempty
fallthrough as well as empty bodies before reading a result slot. I leave wider
old behavior for separate review and require negative result-boundary fixtures
here. This is not a silent change to the old verifier.

### My accounting and cleanup

I count conservative work units and reserved live capacities, not measured
allocator traffic. The following reservations precede work/allocation:

| Component | Reservation / work |
|---|---|
| Fixed explicit automatic scratch | 64KiB and65536 work units; implementation/compiler call-frame overhead and allocator metadata are not claimed measured. |
| Whole declaration plan and staging | Full16MiB and1048576 units, retained conservatively even when the concrete plan is smaller. Its independently reviewed internal caps remain enforced. |
| Each decoded function | Five bytes per code byte plus sentinel for boundaries/indices; three times geometric instruction capacity times `sizeof(VmDecodedInstruction)` for realloc overlap; code and instruction passes charged before decode. |
| Height work | Three32-bit vectors per instruction plus terminal;32 units per entry, released before type-work allocation. |
| Type work | `(instructions+1)*(max_stack+sizeof(u16)+2*sizeof(bool)+sizeof(u32))`; initial cells charged, each worklist visit charges `4*max_stack+32`, then reservation released. |
| Origin analysis | Exact allocated structure, state/queue/depth/seen/origin arrays, copied layouts/maps/element constraints, weak field summaries and final report overlap; allocation initialization and explicit scans charged. |
| Fixed-point work |2048 units per transfer for fixed state/seed scratch; joins charge actual cell counts; array/record helpers charge their64-origin loops separately, including each constructor/literal value; slices charge64x64 search bound; outer sweeps and final checks are charged. |

No allocation or scan follows a refused reservation. Cleanup is bounded by already
established function/layout/origin counts and must still execute after exhaustion.
Borrowed input storage is not counted as owned report memory. The absolute64MiB
and16777216-unit ceilings can reject a combination below individual table maxima;
I do not claim every advertised individual maximum fits simultaneously.

The copied plan recreates exact global/per-kind record maps without removing
interleaved unused union declarations. I retain full declaration facts in the
published report. `all_writes` is a monotone per-array-origin summary even for
packed arrays; existing old modes do not create or consult it. After convergence,
record constructors/replacements collect field element obligations, then every
such origin's complete write summary must fit its one exact element tag. The
ordinary packed-coercion matrix remains unchanged outside these new constraints.
GET/POP preserve VOID alternatives, and known parameter/result signatures retain
exact tags. Existing weak global VOID alternatives are not erased to force
acceptance; record-field access can retain runtime receiver checks, while an
optional direct value cannot satisfy an exact constructor field.

A partial structure owns all decoded functions and its fresh declarations.
After decoded functions transfer to analysis their old slots are zeroed. Analysis
owns copied record maps, field constraints and lattice allocations. A successful
report receives the declaration owner only after every copied origin/field
allocation succeeds; otherwise cleanup frees every prefix and leaves `*out`
unchanged. The report contains only numeric facts and owned arrays. All old
query wrappers select the old mode; new budget/fact behavior is conditional on
the private structure. Old selectors still refuse ARRAY_FIELDS as before.

The shared source lists already link verifier, verifier_types, managed_array_shapes
and ownership providers together, including the File public archive closure.
I add no translation unit or installed header. Explicit Make prerequisites cover
both new includes and the private/public source headers. Source review, fixture
review and fresh selected gates are still required; no build or module execution
has occurred for this checkpoint.

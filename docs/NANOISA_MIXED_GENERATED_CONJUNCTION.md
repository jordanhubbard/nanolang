# My generated mixed record-array conjunction

I continue from actual PR928 merge `ad0080ebee76c26012ef24ba47c690a5400d9d37`.
My complete declarations/origin query and counted adapter acceptance are evidence
for preparation and storage, not generated execution. This design uses existing
parents `task_f36b179a0f2b4a1b99c29ccd2af66f99`,
`task_15f955fae5cf402d92bf88794122e9a2` and
`task_488a05eb5e2a417caf83a8353363a30d`. Their full requirements remain mandatory.
I change no source in this checkpoint.

## The next complete vertical dependency

I carry the already qualified record-array graph through **all** VM, generated
C, native LLVM and Wasm consumers before public selection, then through both
source producers. I do not declare one successful backend an executable profile.
This first vertical dependency inherits the exact query domain: ordinary record
DAGs, exact nominal field identities, and flat INT/U8/FLOAT/BOOL/STRING arrays,
including mutation through aliases, nested records, loops, direct calls/results,
globals, repeated sites and copies. Validated unused scalar-union declarations
may interleave global layout indices. Executed union values remain a required
extension below, not an alternate interpretation of this query's evidence.

I retain all eligible query opcodes and argument/result flows that have matching
existing scalar/string semantics. If a consumer lacks one, I implement and
qualify it or keep this entire new public route closed; I do not silently narrow
the query until a small example passes. Unsupported opcodes retain explicit
refusal during private development. Raw CAST_U8 and peer affine/File changes
need their own coordinated consumer coverage before joining this domain.

## Actual current barriers

| Current path | Required change and retained boundary |
| --- | --- |
| `nvm_select_managed_heap` | Calls old `nvm_verify` and constructs old record facts; it cannot serve the new plan through a recursive public admission call. Old wrapper decisions remain unchanged during private qualification. |
| `nvm2llvm_emit_target` | Chooses old closed profiles, constructs a second heap plan and re-verifies each function after writing runtime IR. The new path must finish original-module preparation, max-stack and complete target coverage before its first output byte. |
| `nvm2c_emit` | Service-first, owned-array and Samples routing precede its ordinary typed representations. A separate checked ordinary mixed path must not reinterpret those representations or steal their selection. |
| VM | Existing NanoValue/heap records and arrays have their own lifetime implementation. A private adapter uses those real instructions and roots with the new checked original-module certificate; no public verifier bypass or NmsValue/NanoValue bit cast. |
| LLVM/module adapters | Existing descriptor binding, counted roots, record GET/SET and graph safe points are useful. PR928 did not exercise their generated control flow. |
| Wire/link/bridges and C/Nano producers | Whole extension tables, exact nominal remapping and all selected function/shadow graphs must survive. Missing metadata cannot become empty ordinary authority. |

## Dependency 1: a copied non-admitting execution plan

My smallest source checkpoint is preparation only, in a new source-private
`managed_record_array_execution.h`. Proposed opaque API:

```
typedef struct NvmRecordArrayExecutionPlan NvmRecordArrayExecutionPlan;
NvmArrayEligibilityResult nvm_prepare_record_array_execution(
    const NvmModule *, NvmRecordArrayExecutionPlan **);
void nvm_record_array_execution_free(NvmRecordArrayExecutionPlan *);
```

I retain ELIGIBLE/UNRESOLVED/INVALID/LIMIT/MEMORY meanings, but ELIGIBLE here means
copied preparation facts, never public executable selection. Failure preserves
`*out`. The production checkpoint supplies exact C count/function/instruction/
edge/descriptor/accessor structs and numeric semantics before fixtures. All
getters copy defined fields, preserve output on failure, and expose no mutable
source pointer or reusable caller-supplied trust bit.

I bound and copy every execution-relevant original module input, including exact
code, strings/counts, function flags/signatures/sidecars, initializer/entry identity,
layouts, ownership extensions, globals and imports. I neither erase an unsupported
section nor rewrite tags to satisfy a verifier. The immutable borrowed input must
remain stable during this synchronous operation. Preparation runs fresh complete
structural/declaration/origin validation against that exact owned snapshot.
Independent comparison with the original checks every copied defined field and
byte before publication. A content hash may label evidence, not replace semantic
comparison. Destroying all original inputs must leave every plan fact usable.

The plan owns decoded instructions, exact next/branch edges, function maxima,
global extent, record ordinal/global-layout/field maps and the full origin report.
Each instruction gets an explicit operation classification, stack effect,
receiver/tag obligations, allocation/safe-point classification, and root-transfer
recipe. Coverage includes every declared function and decoded instruction,
including unused functions; unreachable instructions cannot hide an unsupported
operation. Explicit and implicit returns both require exact result depth and tag
obligations. Unknown signatures/receiver origins are not fabricated proofs.

I derive descriptors from the plan's complete declarations, never a second
name/shape reader. Exact nominal global identities remain separate from compact
runtime descriptor ordinals. ARRAY field declarations and their constrained
origin alternatives remain attached even though the runtime descriptor itself
contains only layout identity and field count.

I retain query bounds (256 functions/locals/stack/globals, 65,536 instructions,
64 origins, 65,536 field-summary cells, 1,048,576 abstract cells). My wrapper
reserves at most 128 MiB simultaneous requested memory: conservatively reserve
64 MiB for the complete query's own documented peak, and at most 64 MiB for
snapshot, decoded/root/coverage tables, all other temporary capacity and final
report overlap. I reserve before allocation with checked products/sums; freeing
one stage may release its reservation only after all ownership is gone. The
wrapper's additional copy/compare/decode/coverage operations have a separate
16,777,216-step ceiling, giving a maximum combined 33,554,432 charged steps with
the query. Each byte/row/cell traversal and repeated helper scan is charged;
no unbounded strlen, recursive graph walk or hidden rescan sits outside it.
Actual boundary/next and allocation-prefix fixtures must measure both domains.
No executable code or public guard changes in this dependency.

## Generated root and failure semantics

I use an explicit operation table shared by the consumers, not inferred cleanup
from emitted text. Its production review must enumerate every numeric opcode
accepted by `nvm_record_array_opcode_supported` at the reviewed pin, including
all scalar/string entries supplied through `fixed_result`, with a total switch
and refusal default. No capability bit can substitute for implementing an opcode.

| Operation family | Owning roots and transaction boundary |
| --- | --- |
| Constants, scalar arithmetic/comparison/casts | Scalar values own no heap edge; counted STRING results acquire one root. Runtime wrong-tag and arithmetic failure follow existing VM semantics; first failure wins. |
| LOAD/DUP | Retain the exact referenced object before publishing a second root. Retain failure publishes nothing. |
| STORE local/global | Incoming root is acquired/staged before old destination release; overwrite moves one root. Old aliases remain valid. Failure leaves destination/operand ownership defined and cleanup complete. |
| POP, condition/assert and scalar consumers | Remove one operand root exactly once, including STRING/record/array operands rejected by a runtime tag check. |
| Record/array literals | All source-order operands stay rooted until allocation and child retains succeed. Partial child retains roll back; only then consume inputs and publish one owner. |
| Record GET / ARRAY GET | Publish retained child before releasing receiver. Optional array bounds result remains VOID. Nominal/index/tag errors are not conflated with an absent element. |
| Record SET / ARRAY SET/PUSH | Retain new edge before releasing old edge; receiver/result identity follows exact opcode semantics. Failure cannot publish a half mutation or lose an outside alias. |
| POP/slice/copy | Popped child transfers/retains according to the original opcode; copy has distinct handle/storage with independently retained children. It is never represented as the original allocation site at runtime. |
| CALL/RET | Stage actual arguments in source stack order, transfer each exactly once to the callee's locals, and move result before callee cleanup. A failing callee leaves no unowned staged arguments. |
| Branch/backedge | Every live stack/local slot remains an owning root; incoming abstract alternatives do not collapse physical aliases or create duplicated releases. |

Every managed temporary is an owned slot until explicitly transferred or released.
C/native LLVM use the existing checked counted core/module ABI and stable handles;
VM uses its native value/heap ownership operations. No raw handle crosses between
these two representations. Both must preserve program-observable identity and
value bits; different allocators need not have equal internal allocation counts.

Collection happens before an allocating instruction while all operands and
caller roots remain published, or after complete cleanup at entry finish. Never
collect midway through a constructor, replacement, call transfer or return.
If growing collector workspace is required, prepare it before changing roots;
failure is an ordinary recorded allocation failure. A suspended caller's locals,
operand stack and staged arguments stay visible across every callee safe point.
Global roots survive repeated entries and release exactly once at disposal.
Descriptor tables and runtime package live until terminal disposal.

An entry wrapper validates complete plan/consumer/runtime ABI agreement before
runtime acquisition or output mutation. An acquired-but-failed initialization
still needs finish; BUSY does not acquire and does not inspect/mutate active
state. Initializer and root share the same instance and first-error state.
Nested invocation, recursive direct calls and frame exhaustion have explicit
bounded failure cleanup. I cap the new invocation at256 simultaneously active
frames (initializer and root are sequential, with each counted when active).
Each callee checks/reserves its frame before argument transfer; exhausting this
limit records a stack-limit failure and unwinds all staged/caller roots. C/LLVM
use explicit bounded frame storage if256 host calls cannot be shown safe on the
selected targets; no unchecked C recursion overflow is acceptable. VM enforces
the same new-route limit, preserving old-route limits. This is a new checked
profile execution limit, not a proof of recursive termination. No new implicit
execution fuel is introduced into ordinary language semantics.

## Dependency 2: private matched consumers

I add distinct private entry functions, gated for fixtures, taking only freshly
prepared owned plans. A consumer that also accepts a module must freshly compare
all relevant original facts before effects. There is no public boolean bypass.
The source checkpoint freezes exact adapter/report ABI and root accounting.

VM uses its actual decoded instruction handlers, both true switch and computed
goto, with normal runtime tag/bounds behavior. Generated C uses real functions or
explicit checked frames, labels and direct operators calling counted primitives;
it embeds no VM/interpreter. Native LLVM and Wasm emit real LLVM functions/blocks
against the existing packaged runtime. I reuse existing operation emission only
where its root transaction matches the table; factoring retains old entry behavior.
All use complete descriptor and stack facts from the same plan, not old public
verification called recursively from the new private verifier.

C/LLVM/Wasm output is staged privately until every function, edge, runtime ABI,
export name and required operation is covered. Allocation/validation failures
preserve both output pointers and preexisting CLI files. Stream-writing APIs
prepare fully before output; I document that an actual caller-stream I/O error
cannot retroactively restore bytes, while file publication uses atomic staging.
Native isolated linkage proves no VM-dispatch dependency. Wasm remains import-free
for this ordinary closed domain with explicit maximum memory and stable repeated
instance behavior. Existing entry/result ABI forms remain exact; adding a new
heap-result export or changing scalar status truncation requires separate review.

## Dependency 3: generated acceptance before admission

I translate the same complete original modules through all four private routes,
not hand-replace generated code with the PR928 adapter sequence. Corpus coverage
includes five element tags, counted strings with embedded NUL/UTF-8, exact FLOAT
bits (signed zero/infinities/NaN payloads), INT extrema, U8 endpoints, nominally
distinct equal-shaped records and interleaved unused unions. I require field and
outside aliases surviving replacement, independent growth, shallow copy identity,
constructor order, branch joins, repeated allocation sites/backedges, direct
calls/results, initializer/global replacement and 1,024 repeated entry/disposal.

I retain normal and assertion/bounds/tag failures after successful verification.
Native allocation hooks exercise measured one-shot and persistent positions with
fresh recovery; VM allocator hooks use its own measured positions. I compare
observable results, error class, field/array identity and cleanup obligations,
not coincidental allocator numbering. Targeted collection/safe-point schedules
cover live ancestors, staged calls, partial construction and global-only roots.
O0/O2 generated C and original/O2-optimized LLVM execute under supported native
sanitizers; LLVM verifies before/after optimization. Both Wasmtime and Node execute
import-free finite-memory Wasm before/after optimization on both hosts. Runtime
packaged artifacts must be test-hook-free and match the exact generator sources.

Negative controls independently alter declarations, origin constraints, a later
instruction/edge, stack/result shape, descriptor ordinal, ABI size/revision and
runtime package identity. Every refusal happens before acquisition/publication.
Fault and finite-work/storage sweeps prove sentinel preservation and cleanup.
Old managed strings, records, graph arrays/cycles, owned-array/Samples, scalar
unions and File/service-first refusal neighbors remain intact. Whole parent
acceptance is not replaced by these generated fixtures.

## Dependency 4: public conjunction and transport

Only after all preceding evidence is reviewed do I add one service-first checked
route decision: NOT_SELECTED for unrelated old profiles, SELECTED for the exact
new ordinary array-field envelope, INVALID for malformed attempted selection.
SELECTED preparation failure is terminal; no fallback to old ordinary, owner-array,
Samples, scalar or unverified emission. Existing service/owned paths keep priority
and identity. The public selector performs fresh original-module preparation;
a previously obtained private report cannot be passed as a trust certificate.

I review all general/function/max-stack/linked verifier and serializer callers,
VM direct/CLI wrappers, nvm2c, closed managed profiles, nvm2llvm/nvm2wasm and
converter/linker/bridge consumers together. Each either uses the complete new
plan or retains checked refusal until its matching implementation is qualified.
No released route may accept metadata and then misinterpret field identities.
Serialization and linking preserve both extension kinds and exact type/global
nominal remapping; incomplete remapping fails atomically, never drops ARRAY_FIELDS.
I include real save/load/link and installed header/runtime/archive controls.
A broadened public ownership helper cannot expose half a mixed declaration table.

## Dependency 5: paired source and the rest of the full graph

Both C and Nano producers derive array element authority from resolved declared
types, including exact module/declaration/recursive generic arguments. Literal
contents, alias spelling and storage representation are never authority. They
publish complete mutually consistent tables atomically, preserving source
acceptance with honest absent optional metadata where unsupported. Initializers,
callees and all selected original shadows remain in the graph. I review source
emission and lowering before fresh bootstrap, paired bytecode/source behavior,
full unchanged ordinary/affine/mixed corpora and installed acceptance.

The next required graph extensions remain ordered work, not exclusions from5.1:

1. Executed scalar/string union values with exact variant/value refinement and
   retained payload roots, composed with ARRAY_FIELDS and all nominal joins.
2. Recursive element declarations, nested arrays and arrays of exact ordinary
   records: full origin alternatives, runtime child traversal and rooted/dead
   cycles across records/arrays, including alias-preserving string-array promotion.
3. Forward/recursive and imported/generic nominal graphs, then the applicable
   tuples/maps/callables and mixed ordinary/resource boundaries. Resource edges
   require their actual affine ownership proof, never ordinary counting by default.
4. Each extension repeats query + all-consumer root/lifetime + public transport
   + paired-source conjunction and its original program/shadow acceptance.

I preserve the original parent descriptions and histories. The first complete
vertical dependency does not close f36/15f/488 or authorize a5.1 release. It
establishes reusable exact plans and generated root semantics needed by those
full obligations instead of substituting a restricted language for them.

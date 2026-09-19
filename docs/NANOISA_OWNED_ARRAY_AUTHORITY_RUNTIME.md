# I compose owner ARRAY authority before runtime admission

I continue task430220 after descriptive PR824 and origin-query PR829. Their
success describes layouts and conditional FLOAT origins; it does not establish
unique owner lifetimes, scalar obligations or runtime permission. This contract
is preimplementation only. I preserve both qualified trees and all original
Bundle/PREFIX source and selected shadows.

## My current boundaries are concrete

- `ownership_contracts.c:check_layouts` rejects ARRAY fields. Its retained
  descriptors preserve RESOURCE and strictly prior owner edges. I do not remove
  that shared rejection merely because my private origin query succeeds.
- `affine_state.c:nested_result_tree`, value-result queries and ordinary state
  construction admit INT/BOOL/U8/STRING owner leaves, not ARRAY leaves. FLOAT
  locals from PR804 do not imply FLOAT owner fields or signatures.
- `mixed_samples.inc:mc_checked_state` privately constructs exact original Facts
  after a fresh shape query, but `mc_owned_layout` accepts only direct scalar
  leaves. Its independent stack/local owner joins and complete exit checks are
  useful mechanisms, not existing nested ARRAY-owner authority.
- `mixed_samples_prepare.inc:mp_declaration/mp_value_signature` excludes STRING,
  and preparation requires a nonempty ordinary-record map. A Bundle containing
  Handle and ARRAY can have no ordinary record at all. Its existing candidate
  already sees any ARRAY field plus RESOURCE, including such a Bundle; failed
  existing preparation must not become ordinary fallback.
- `vm.c:vm_mixed_invocation_prepare` consumes checked mixed signatures and
  original source/global/managed mappings. Public execute/call/resume/link entry
  guards recompute or check that invocation proof. Those paths cannot accept an
  origin-only object in place of their complete conjunction.
- `nvm2c_owned.h` already distinguishes owner category1 and ordinary category2
  values. Category2 retains both NmsRuntime and NmsHandle; recursive shell
  cleanup releases managed children. Existing mixed output assumes at least one
  generated ordinary-record descriptor. I must not emit a zero-length C array
  for Bundle-only modules. `managed_strings.c:nms_bind_records` already permits
  count0 with NULL descriptors and still records that binding occurred.
- `nvm_v2_convert.c`, general/function/max-stack/linked verification and native
  emission route through existing mixed candidate/admit APIs. LLVM/Wasm closed
  profiles refuse mixed ownership. All remain unchanged in my first checkpoint.

## I first build a private complete authority query

I propose an opaque `NvmOwnedArrayPlan` prepared only from the immutable original
NvmModule. No API accepts a caller-built origin proof, trusted flag, rewritten
layout table or cached nominal certificate. Preparation recomputes the retained
layout/origin queries, independent owner/scalar analysis and complete common
structure/metadata checks from the same bytes. A success owns exact transport,
checked signatures, original local/global/source indices, per-function stack
bounds and runtime scalar obligations. Failure preserves the caller's output.

My first production checkpoint stays query-only and non-admitting. Its checked
Facts constructor lives inside affine_state.c, like the existing private mixed
constructor, and cannot be called through a public unchecked constructor. I keep
ordinary ARRAY/STRING locals as their real tags and exact owner records as
RESOURCE; I never rewrite either to VOID or clear ownership flags. Shared public
ownership validation, reference semantics and existing mixed Samples queries
retain their old boundaries.

I reuse exact move/put/clone/owner-state equality/exit machinery where its
preconditions hold. New private descriptor/result helpers admit only the
reviewed strictly prior nested owner DAG with INT/BOOL/U8/STRING and proved ARRAY
leaves, retaining original depth/field limits. They are not general heap-valued
owner helpers. All nested fields, including result paths, must match the fresh
origin plan and exact declared nominal identity. Ordinary record fields inside
owners, FLOAT owner fields, generic/union owners and borrowed roots remain
outside this checkpoint.

Independent state tracks each owner local's initialized/live status, each
operand's unique owner token or nonescaping observation, and ordinary value
initialization/tags. ARRAY aliases may copy; unique shells may not. Ordinary
stores cannot overwrite owners. Move/unpack consumes a live unobserved shell;
pack consumes exact child owner tokens in declaration order, while transferring
ordinary roots. Calls transfer exactly declared owner arguments and establish
only their checked return category. Every return leaves no unreturned live owner
in locals or operand stack. Branch joins require compatible owner state before
merging ordinary alternatives; loops cannot duplicate a token or resurrect a
consumed owner. An origin-proof success with missing cleanup must fail this
independent analysis. STRING initialization meets remain distinct from owner
joins and retain their existing checked semantics.

Common structural checks retain function/header/name/range/import/metadata and
retained-layout checks. Any private ownership delegation is enabled only after
fresh origin and independent lifetime/scalar success, within the verifier's
internal preparation path. It cannot be requested by an external bool argument
or exposed as an executable authorization object. Service presence is rejected
first, including partial metadata. Linked modules, callbacks, reference modes,
initializer roots and unsupported entry/signature forms remain refused.

I retain origin-query bounds:8 acyclic functions,8 parameters,256 locals/stack,
4,096 instructions,64 sites,256 formal/result ARRAY paths, bounded descriptor
DAGs and1,048,576 state/65,536 field cells. Independent authority frames and work
queues receive their own explicit checked budgets before allocation. Public
entry execution will retain the existing INT/BOOL/U8 result profile; an
origin-only VOID-entry or STRING-result summary does not silently widen it.

## I audit scalar and source coverage before selecting an executable profile

The current origin whitelist deliberately refuses unmodeled operations. Before
runtime publication I compare the exact paired producer opcodes for unchanged
Bundle/PREFIX and selected shadows with both origin and independent authority
transfers. I preserve FLOAT-or-VOID array reads. A typed numeric consumer must
have either exact established operands or an explicit checked runtime obligation
consistent with the already qualified mixed policy; I do not invent a FLOAT
default for absent elements. Any required origin-query extension gets its own
reviewed production checkpoint and fresh query tests before activation.

STRING sibling transport must coexist with arrays without becoming an NmsHandle
or array origin. Existing nonmixed STRING-owner programs stay on their qualified
path. I include exact PRINT/PRINTLN prefix semantics needed by failure controls
only after matching origin/scalar/runtime transfer review; no silent opcode
fallback or blanket STRING/arithmetic widening follows. Numeric helper semantics,
NaN/rounding policy and stored bits remain unchanged.

## I then qualify a private VM/native runtime path

I stage runtime changes separately from the authority query and public routing.
A private test adapter may prepare a fresh complete plan and enter the reviewed
handler path; it cannot execute a rejected plan or import a forged proof.
After that stage qualifies, I review public activation in a separate checkpoint.

VM arrays remain ordinary retained NanoValue roots within unique owner shells.
Shell transfer is affine; observing an ARRAY creates a separately retained
ordinary alias whose lifetime can outlast shell consumption. Unpack transfers
field roots exactly once. Calls, pending results, locals and operand stack keep
all ordinary roots reachable until transfer or cleanup. STRING siblings retain
their existing independent root mechanism. A trap or refused continuation drains
all owned and ordinary roots without relying on a successful return.

Native arrays retain `(NmsRuntime *, NmsHandle)` identity through nested
`nown_record` fields and helpers; the root invocation owns the runtime until
all locals, stack values, pending results and nested shell children are drained.
Owner rows never enter the compact ordinary-record map. For zero ordinary rows,
I bind `(NULL,0)` and emit no zero-length descriptor array; ARRAY allocation is
still authorized only by exact origin/complete plan facts. Existing Samples
ordinary record maps and indices remain unchanged.

Pack allocates its shell before consuming input roots; allocation failure leaves
all inputs available for cleanup. Observation stages checked retains before
publishing an alias. Unpack and return transfer roots without an extra retain or
double release. Failure after any partial retain releases exactly that prefix
and preserves other inputs. Mutable aliases see writes and append growth through
any root. Failed growth preserves prior length/storage/content and every alias.
Cross-runtime, stale handle and counter/generation exhaustion remain checked;
no fallback treats a handle as an integer or a unique owner as a managed record.

## I review activation as a separate all-consumer change

A new owner-ARRAY routing hint must take precedence over the broader existing
mixed Samples candidate when an owner contains transitive ARRAY leaves. It is
allocation-free and conservative; it grants no authority. A positive candidate
whose complete plan fails must refuse, never fall back to old owned, ordinary,
managed or FFI handling. Existing ordinary Samples plus scalar-owner programs
continue through their existing branch. Service presence wins before both.

Activation covers general/function/max-stack verification, v2 conversion both
ways, zero/nonzero linked cases, every VM execute/run/call/continuation API,
public helper-entry refusal and native module/function emission. LLVM/Wasm and
other closed profiles retain explicit refusals until a separate backend contract.
All executable consumers freshly prepare from the current immutable module;
no stale proof survives a different invocation/module or changed bytes. Native
refusal preserves pre-existing output. I do not update only one CLI while
leaving alternate public entry paths permissive.

## My acceptance stays staged and observable

Private query controls first pair positive origin facts with independent
lifetime refusals: omitted destruction, duplicate move, incompatible branch
owners, overwritten live locals, observation across consumption, uninitialized
STRING/ARRAY joins and wrong call/result categories. I retain exact maps and
output sentinels through every allocation failure and qualify existing
scalar/nested-result/STRING/mixed Samples/refusal/provider suites.

Private runtime controls then require two fields sharing one ARRAY, two shells
sharing it, nested/relay return transfers, alias survival after consumption,
reverse unpack, mutation and capacity growth, empty arrays/owners, STRING
siblings, loop/branch cleanup, traps and pending-return failure. I check actual
root/object/byte counts, no double releases and later complete recovery. Native
fault runs preserve exact successful stdout prefixes as well as terminal status;
I never weaken a prefix assertion to obtain cleanup acceptance. GCC/Clang strict
sanitizers and actual VM/native integer/array observations qualify these paths.

After separately reviewed activation and both source producers, I run the exact
unchanged `test_ordinary_array_field_keeps_element_type` Bundle/PREFIX with all
selected close/main shadows, including inline Handle construction, reverse
binding order, FLOAT read1.5 and close7. Linux/Darwin and adjacent existing source
families remain the full-source acceptance requirement. None of the earlier
private checkpoints closes430220 or the managed/affine/product parents.

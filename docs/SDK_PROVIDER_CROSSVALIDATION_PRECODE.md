# I connect complete provider facts to an executable contract

This is my precode contract after private typed profile895. It proposes the
next validation and lifetime representation; it does not activate a feature,
load an artifact, change wire revision1, or grant execution. I keep complete
source/installed SDK acceptance required, including callbacks and COP. A
transport-only success is not a substitute for any of those gates.

## I keep one type graph and exact declaration identity

I consume the owned V2 module snapshot, its original signatures/selectors,
the shared typed declaration plan, and the raw provider rows. I do not convert
through legacy NvmModule to establish identity. I retain every unused signature,
layout, nominal, binding and policy row, validating unused rows too.

I compare names by their actual STRING constant byte lengths and bytes, never
by string-pool index, a short suffix, C spelling, or a guessed import prefix.
Identifiers and artifact symbols cannot contain an embedded NUL. The actual
unnamed-root owner may be empty; I do not invent an owner to make it match an
import. A nominal key is `(owner, original name, nominal kind, ordered complete
generic arguments)`. RECORD, UNION, ENUM and OPAQUE remain different kinds.

Each nonopaque key maps to its exact original layout ordinal of matching kind;
opaque keys have NO_LAYOUT. Duplicate alias rows may repeat the same full key
and ordinal. One key selecting different ordinals, or different keys claiming
one nominal ordinal, refuses. Anonymous tuples compare structurally and do not
acquire a nominal owner from a containing record. Every used record/union/enum
layout and every referenced opaque row needs a complete key. Unused nominal
rows must satisfy the same uniqueness and storage conditions.

I compare complete types through the retained shared rows: scalar tags,
ARRAY child, anonymous tuple fields, exact nominal key, or exact FUNCTION
signature detail. Signature details must agree with every coarse tag/count,
while functions/imports keep their original signature_idx (including duplicate
coarse rows). Equality of coarse signatures cannot equate different opaque
owners. I use a bounded worklist and visited pairs for recursive reference
shapes; no recursive C stack or uncharged pairwise scan establishes equality.
The existing by-value layout DAG and64-node ARRAY chain constraints remain.

Bindings have one actual subject. Import and function bindings have NO_SLOT;
field bindings use `(original layout ordinal, actual field index)`. Duplicate
subjects refuse even if their payloads agree. Every function/import signature
requiring complete composite facts needs exactly one detail binding. Every
FUNCTION/OPAQUE coarse field needs one exact shared type binding. ARRAY field
bindings must agree with the already retained ARRAY_FIELDS element facts.
Optional redundant scalar/nominal field facts must agree with the existing
layout tag/referent. Union fields use actual flattened field ordinals and the
checked variant partitions; I do not derive them from variant names alone.

Provider requirements match the selected module/artifact owner, ABI revision,
target ABI, generation digest and artifact digest. Artifact import paths are
resolved through their exact generation manifest entry; they are not mistaken
for logical module names. I compare the import's real symbol and selected
signature against that owner's generated adapter row. All digest/manifest
checks precede invocation. These checks identify trusted native input; they do
not turn a manifest into a trust signature or sandbox arbitrary native code.

## I make recursive lifetime policy explicit

The earlier16-byte top-level slot proposal cannot describe mixed nested leaves.
I replace it in proposed provider revision2 with32-byte policy nodes while
keeping the proposed40-byte header and existing five tables. Header offset4
counts24-byte call-policy rows; offset28 counts32-byte policy nodes; offsets32
and36 remain zero. Original revision1 remains exactly32-byte-header transport.
No new recursive type table is added.

A24-byte call-policy row has parameter-first/count, result-first/count,
call execution selector and reserved zero, all u32. The execution selector
references an existing NO_PARAMETER callback/wait contract or is NO_INDEX. The two slices select
policy nodes directly and exactly match the selected signature's ordered
parameters/results. An import binding's existing reserved offset20 selects
its call-policy row in revision2 only. Function/field bindings retain zero.

A32-byte policy node contains these u32 fields in order:

| Offset | Meaning |
| --- | --- |
| 0 | Existing shared complete type index |
| 4 | Lifetime/mutation mode |
| 8 | Owning argument index, or NO_INDEX |
| 12 | Exact provider hook-set symbol STRING index, or NO_INDEX |
| 16 | First child policy node |
| 20 | Child policy count |
| 24 | Callback ABI in low8 bits, execution in next8, upper16 zero; zero for noncallback |
| 28 | Reserved zero |

Node children are ordered by existing type facts: record/tuple field order,
union flattened fields under existing variant partitions, one ARRAY element,
and FUNCTION parameters then results. Scalars/string/opaque have no children.
Each child node's type must equal the corresponding complete type. Parent
COPY never supplies a missing child lifetime. Node slices may share rows;
cycle checks follow type edges, permitting reference recursion only where the
existing type graph permits it. All nodes, including unused nodes, validate.
Counts stay bounded at65,536 call policies and65,536 policy nodes, with the
whole section16MiB. I charge all worklists, comparisons, copied rows and scratch
against the same transactional32MiB/1,048,576-step generation budget.

I propose mode values0 through8 in the following order, checked against position
and tag:

- VALUE copies scalars by exact width and preserves enum identity.
- BORROW_CALL gives the provider a read-only, call-duration view. The adapter
  cannot retain it; nested leaves have their own explicit policies.
- BORROW_MUTABLE_CALL stages permitted changes and publishes all argument
  updates together only after successful result validation, following the
  existing FFI array frame's prepare/commit/dispose contract.
- SNAPSHOT_RESULT copies result storage into caller-owned runtime storage.
  STRING snapshots use existing copy-before-release behavior. Structural
  children follow their individual modes. A hook-set, if specified, drops the
  original provider result exactly once after successful copying or refusal.
- BORROW_ARGUMENT_RESULT identifies an exact argument owner. The prepared call
  retains that owner through every returned borrowed leaf; it cannot infer an
  owner from pointer equality alone or grant ownership to an external input.
- OPAQUE_PIN uses the exact provider hook-set to pin and unpin logical object
  lifetime. OPAQUE_PROVIDER_LIFETIME instead requires an explicit generated
  provider promise that the object survives the generation lease. Neither mode
  derives lifetime from an opaque COP token or a loaded image alone.
- CALLBACK_CALL and CALLBACK_RETAINED reuse the exact retained callback ABI,
  execution policy, signature and owner shutdown rules. A retained handle owns
  its module/generation and value roots until explicit provider release and
  runtime retirement. No raw code pointer crosses the boundary.

I reject modes applied to the wrong tag/direction, owner indices outside actual
arguments, incompatible hook sets, an existing top-level callback contract for another import/slot,
or mismatched callback signatures/execution policies before provider effects.
Top-level FUNCTION policy must agree with the actual existing CALLBACKS row for
that `(import, parameter)` and its exact selected signature. Nested FUNCTION
leaves cannot select that table by invented parameter indices: its sorted unique
`(import, parameter)` key has no field path. Instead each nested policy node
carries its own ABI/execution byte pair and existing complete FUNCTION type;
its containing import, exact policy path and generated hook-set identify the
adapter. This is explicit additional metadata, not permission to ignore a
conflicting top-level contract. Shared callback validators must check the same
ABI/execution enums. RetainedV1 currently accepts only its supported signature
shapes; callback parameters/results needing complete aggregates require a new
generated typed callback adapter ABI and their own reviewed runtime entry. They
must refuse under V1 until that entry is implemented and qualified. I retain this
as required SDK work, not an accepted callback-scope reduction.

Every unused call-policy row still supplies its own argument count and roots for
context validation. Unreachable policy nodes may not carry an owning-argument
index or callback execution claim: without a call context I cannot establish
either. They still retain and validate their exact type, mode and children.

A provider-owned aggregate hook drops its own graph once; child release hooks
must be absent unless the generated schema explicitly declares independent
owners. This prevents both inherited ownership and double release.

## I require the actual generated C adapter to agree

I propose one versioned generated SDK adapter table per actual provider image,
resolved through `ffi_loader_resolve_module`/retained image ownership, never
RTLD_DEFAULT fallback. Each table row advertises the original declaration key,
complete signature and policy schema, target/ABI and its actual typed adapter.
A schema digest is only an acceleration hint; I compare the complete schema.
The table and all cleanup hooks must belong to that same resolved provider.

A generated adapter includes the provider's real headers, calls its real C
prototype and translates each aggregate with the compiler-known C layout.
Records/tuples/unions are not dispatched through an int64 function-pointer cast
or an assumed universal struct ABI. The neutral invocation interface carries
checked runtime values plus exact schema indices; it adds a value transport,
not another source type system. The same generated adapter implementation is
selected by VM and nvm2c; host ABI, padding, discriminants, scalar widths and
callback calling conventions are checked at adapter compilation and runtime
schema attachment. Strict-prototype/sanitizer fixtures exercise real providers.

Preparation owns all argument temporaries and rollback actions before effects.
I validate source values and reserve output/commit bookkeeping before invoking
native code. After effects I validate the actual variant, extents and ownership
claims before publishing output or mutable arguments. A failure preserves
caller output/argument publication and runs staged cleanup once. It cannot
promise to undo arbitrary provider-side effects; retries are never automatic.
The owning frame preserves repeated-reference identity, including aliases in
nested arrays, and uses a bounded visited value graph instead of double-freeing
shared leaves. Cyclic references require the same explicit identity graph on
copy/cleanup; I do not silently flatten aliases or claim a tree-only success as
full aggregate support.

## I map the same contract onto COP deliberately

Current scalar opaque support is separate reviewed native work: owner generation
plus explicit issued membership, worker pointer slots, pre-call reserve and
result capture before fallible commit. It establishes transport identity only.
I will integrate its qualified source separately before using it. Nested opaque
and callback transfer are currently refused; the new schema must implement those
paths before claiming complete COP SDK acceptance.

Scalar opaque leaves reuse exact owner/worker token checks. Nested opaque leaves
use those same checks at every policy node, including same-owner aliases and
stale/foreign/unissued token refusal. No pointer bytes cross pipe/mailbox frames.
Before foreign entry I reserve bounded nested opaque capture and cleanup
capacity. Every actual provider pointer acquired by the call remains captured
even if later graph decoding or publication fails; a failed output is not
permission to lose its retirement action. Existing scalar slot-hole refusal
controls remain required.

Logical pin/drop hooks run in the owning worker; token table disposal alone
never calls an inferred provider destructor. Worker shutdown must drain/reject
outstanding calls and settle logical lifetime actions before image teardown.

Aggregates use a typed value-graph envelope carrying node IDs and original type
indices, with variant and extent validation before allocation/publication.
Repeated references use node IDs, preserving alias identity. Existing channel
frame/mailbox limits and bounded decoding remain authoritative; transport
chunking, if needed for already required SDK values, needs its own bounded
framing review rather than a hidden larger limit.

CALLBACK leaves need an explicit owner-side callback registry and correlated
worker callback request/reply messages. The existing callback ABI owns execution
and retirement on the original VM thread; the worker sees a generated typed
trampoline, never a VM pointer. Parent wait logic pumps permitted callback
requests with call/generation IDs so a synchronous callback cannot deadlock the
blocked call. Callback requests also require explicit issued registry membership plus exact
generation, call, policy path and complete type agreement; IDs or a high-water
mark do not establish authority. Pumping alone does not solve a callback that
synchronously re-enters FFI on the same busy worker: I must implement a bounded
nested-call protocol or refuse that re-entry before its provider effects.
Retained callbacks need explicit revoke/drain/release acknowledgments. Unsupported execution modes, worker restarts and late requests refuse
before dereferencing retired owners. This protocol extension needs native
owner review and actual pipe/mailbox round trips; it is not already supplied
by `vm_callback_create`, which currently refuses isolated FFI.

I make no claim that these adapters establish general multithreaded no-exec
fork safety. The separate worker launcher/loader contract still governs entry.

## I publish a new generation only after complete validation

The next private crossvalidation API stages an independent owned module snapshot,
typed declaration plan, provider transport and normalized comparison/policy
facts under one copied budget. It publishes an immutable description generation
only after every row succeeds. No old module/cache pointer, descriptor cache or
active borrow changes. Description preparation does not load code.

A separate adapter-attachment phase verifies real provider images/schemas and
owns their leases; it still returns a new unpublished generation. Execution
requires that attached generation plus the existing ordinary module verifier.
The caller may publish only an exclusively owned, unpublished pointer. Partial
attachment failure drops newly acquired leases/plans and leaves original output,
budget and old generation unchanged. There is no live module swap or in-place
cache repair. Existing modules without the new feature keep their current paths.

Before feature/section activation I require exact-key conflicts and alias cases,
all retained selectors/unused rows, recursive mixed policies, boundary budgets,
every allocation/attachment prefix, same-image hook refusals, output sentinels,
real VM/nvm2c provider mutation and cleanup, callback lifetime/reentrancy/shutdown,
and all COP pipe/mailbox variants with stale/foreign/nested tokens. Then I run
original installed source-hidden/read-only/output-failure SDK acceptance on both
hosts and the canonical producer/backend matrix. Unknown or incomplete metadata
must refuse consistently in every reader/writer/verifier/converter/VM/AOT path.

## I checkpoint the private lifetime wire before semantic crossvalidation

I implement the reviewed40-byte revision2 header, original five tables,24-byte
call-policy rows and32-byte policy nodes through the same provider shape and
reserved/slice validator. Existing encode/decode/budget entry points still select
revision1. New explicit lifetime entry points select revision2; shared by-value
accessors retain original declaration indices and expose separate policy rows.
All count dimensions are bounded before arithmetic, allocation or traversal;
the caller budget includes every table visit and owned transport bytes. Output
and budget publish together. No codec entry validates provider lifetime claims.

The additive raw fixture retains every original assertion, checks both revisions'
refusal boundaries, exact counts/row bytes, a truly unused policy, all slice/
reserved/profile limits, exact/minus-one budgets, repeated allocator refusal
and recovery, and independent data after source destruction. Strict production
C99 and fixture C11 syntax checks pass. Runtime remains held for source review.
Complete type/key/subject/policy-context crossvalidation is the next required
source unit, followed by actual adapter/image attachment and execution gates.

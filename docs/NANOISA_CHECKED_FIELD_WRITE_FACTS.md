# My checked exclusive field-write fact contract

I own this contract-only #998 continuation from `d67295b3d` on
`design/5.1-checked-field-write-facts`. I make no production or schema changes.

## Existing acceptance obligation

Release scope row10407 requires preserving purity, affine, generic, effect and
exhaustiveness frontend facts. Issue #998 requires checker provenance, applicable
paired producers, transport/lookup, advisory independence and fresh/platform
qualification. After checked call-borrow and field-read projections, dropping
accepted exclusive-write projections leaves the same affine field-access
obligation incomplete. This is an actual existing checker decision, not a new
CFG/join certificate or extension of accepted executable profiles.

`tests/nanoisa/fixtures/source_borrow_exclusive.nano` has `update(view: &mut
Counter)` assigning `view.value` and `view.active`. Both producers already accept
and execute it. I require exactly two write sites for that emitted function,
separate from its three read sites. Source spelling describes a place; exact
checked owner/site identity establishes provenance.

## Actual capture and emission boundary

C `own_node` AST_SET (`src/resource_flow.c`) checks an existing, unmoved exclusive
borrow and conflicts on the constructed field path, then traverses the RHS. Nano
`resource_flow_node_context` PNODE_SET performs the same check from the actual
checked ResourceState. I capture only after both destination checks AND that
single existing RHS traversal succeed with no incoming/new diagnostics. I never
revisit/evaluate the RHS to collect metadata. Final successful function completion
and final successful checking still gate publication; a later failure discards
all prospective facts. Whole-binding assignment is outside this slice.

C resource lowering resolves the actual field, lowers the RHS once, validates its
scalar tag and emits REF_SET. Nano does the corresponding work in its statement
lowerer. I stage a descriptor only at successful REF_SET for that original AST_SET
or parser set index, never from scanning field syntax or RHS reads. If a later
lowering fails, no module text/output is published. No change to opcode order,
RHS evaluation count, field type checks or native ownership transfer is authorized.

## Invocation-owned storage and interfaces

I reuse the existing invocation ledger, not AST-owned pointers. A separate write
event snapshots original function and assignment identity, completed epoch,
assignment coordinates, target and field bytes, RHS node identity/type and copied
source path/root/projection. C snapshots borrowed AST pointers but destruction
frees only owned bytes, never dereferences the AST. Nano retains parser indices
and the existing owner token with its epoch; C-hosted compilation-lifetime string
limitations remain explicit. The assignment has no root identifier expression,
so I must not fabricate a root node or infer category from spelling. Capture uses
the checked exclusive binding; lookup defensively checks unchanged assignment
shape, target, field, RHS identity and owner completion.

Proposed private provider roles are `checked_field_write_capture(env, function,
assignment, root, projection)` and `checked_field_write_lookup(env, function,
assignment)`. Nano uses corresponding capture/lookup within the existing final
checker context and private checked-emission bridge. Public lowering may return
only untrusted bounded site descriptors, never accept a trusted context. Existing
raw wrappers pass absent context; generic/synthetic/rebuilt/stale owners omit.
Repeated checks remove old events without refunding cumulative work budgets.

The shared 4096-event, 1 MiB copied-byte, 131072-work budgets cover call, read AND
write vectors; root/projection length and depth bounds remain unchanged. Optional
limit exhaustion disables all three families consistently before serialization;
C clears storage while Nano may retain bounded inaccessible storage. Allocation
failure is a hard error preserving prior output. Sealed lookup does not mutate
or invalidate already returned snapshots. Shared emitter staging caps also cover
all three vectors; final append is transactional across all families.

## Independent wire representation

I propose `nano.field-write.v1`, independently versioned rather than reinterpreting
read category bits. Its fixed 28-byte header contains LEu32 function, line, column;
mode byte12=1 (checked exclusive field write), coordinate byte13=1 original C or2
flattened Nano, zero bytes14/15; LEu32 source/root/projection lengths16/20/24.
Counted UTF8 bytes follow, all nonempty without NUL, projection starts with `.`.
Coordinates are positive. Invalid source encoding omits optional capture without
changing source acceptance. Overflow-safe lengths and total byte limits apply.
Duplicate identity is emitted function+coordinates+coordinate provenance+source
path, ignoring descriptive root/projection/mode. Known malformed/duplicate lookup
fails closed; unknown versions remain opaque. Function reorder must remap/drop.

Copy, canonical assembly and lookup retain the distinct key. Runtime/verifier/
native consumers never authorize writes from it. I append the new write suffix
after existing read records, leaving modules with no writes unchanged. Comparison
may remove only validated exclusively owned trailing new-key bytes before older
normalization, retaining all surviving string/CODE/authority references.

## Required focused acceptance

- The existing exclusive fixture proves exact two writes and unchanged three
  reads in both producers, transport, VM and sanitized native products.
- An accepted side-effecting RHS fixture proves one evaluation and preserved
  ordering/result. A later RHS, later statement or runtime-shadow failure leaves
  prior output untouched; an invalid/shared/moved destination never gains a fact.
- Recheck, raw, generic, clone/rebuilt owner, wrong site, stale epoch, target/field/
  RHS mutation, program/shadow reset and failed invocation controls reject stale
  or fabricated provenance. Capture needs no synthetic identifier root.
- Limits in either direction omit all three optional families; allocation denial,
  retry and expired-AST Environment teardown cover the new vector and growth.
- Codec framing/UTF8/reserved/mode/duplicate/index/overflow controls, full copy and
  roundtrip, and strip/forge/native-byte invariance remain required. Adversarial
  shared suffix references and changed authority must fail normalization.
- No-write modules retain exact previous bytes. Fresh raw Stage1/Stage2, fixed
  point and exact Linux/Darwin checks remain necessary after new record shapes.

I own ledger/codec/checker/resource emitter/tests/docs only after interface review.
The coordinator owns aggregate build/workflow integration. No build or production
change is authorized by this document alone; blocked primitive files stay untouched.

## Reviewed implementation checkpoint

I implement the approved bounded interfaces on `fix/5.1-checked-field-write-facts`.
My first direct ledger, actual Environment, 1504 codec checks, five comparator
controls and 14 strict GCC/Clang syntax checks pass. Changed producer execution
and C-hosted Nano qualification remain pending the coordinated build slot.
The admitted RHS order fixture uses reads, ADD and MUL followed by two REF_SETs;
I do not require a foreign/helper side-effect profile that is not admitted.

My [final local receipt](evidence/checked-field-write-20261019/README.md) now
qualifies the changed C and C-hosted Nano producers, exact admitted RHS order,
all-three-family lifetime/limits, codec and final standalone raw comparisons.
Fresh aggregate generations/fixed point and exact platforms remain pending.

# My checked call-borrow place facts contract

I propose this bounded #998 continuation from `8161bfc1f` on
`design/5.1-call-borrow-facts`. `workspace_repairs` owns this contract and the
acceptance map. This checkpoint changes no schema, checker, emitter or runtime.
The blocked primitive work remains untouched.

## Existing obligation and bounded result

My [affine-use contract](NANOISA_AFFINE_USE_FACTS.md) explicitly leaves borrowed
projections and broader path information open. I already retain root identifier
OBSERVE/CONSUME sites. I now propose retaining one additional result the checker
actually computes: a particular explicit call argument acquired a checked shared
or exclusive borrow of a named root and field chain during that call.

For example, my existing `source_borrow_nested.nano` passes
`&mut root.branch.left`, `&mut root.branch.right` and `&root.other` to `change`.
The checker distinguishes these places and modes before accepting the call. My
current root OBSERVE records do not carry that distinction. This does not imply
lost runtime authority: executable BORROW_PATH and ownership contracts remain
separate and authoritative.

I do not introduce CFG edges, join certificates, loop invariants, loan lifetime
proofs, arbitrary projections, inferred generic-instance facts or borrow escape.
My branch checkers compare transient moved states but do not produce persistent
join certificates. I do not invent those to satisfy this slice.

## Exact source/checker mapping

### C producer

An explicit address expression is an `AST_CALL` with `as.call.borrow_mode` 1 or 2
and one place argument. The enclosing direct or module-qualified call visits it
in `own_node` (`src/resource_flow.c:706–740`). `own_place` computes the root and
field chain; `OwnBinding` identifies the live lexical owner. `own_conflict` checks
existing holds, and `OwnHold` records the accepted path and exclusive bit until
the call finishes. That path is freed when the call restores its prior holds.

I must capture only a successful event after the actual place traversal, binding
and overlap checks. The capture also requires a clean incoming diagnostic state
and no new diagnostics. A later failed argument, callee/effect check, or function
exit prevents publication through the existing completed-function/epoch gate.
I do not treat syntax `borrow_mode`, a field chain alone or successful lowering
as evidence of checker acceptance.

The proposed storage is an optional event on the explicit address AST_CALL,
containing its exact source-site pointer, enclosing source function pointer,
check epoch, enclosing call/site and argument ordinal, accepted mode, and an
owned immutable snapshot of root spelling and field-chain bytes. The snapshot
is descriptive; the pointers/epoch identify the checked source event. A fresh
check clears the prior event before traversing that address. Parser-created
nodes begin with no event. Saturated epochs permanently disable publication.

### Nano producer

The address is an `ASTCall` list entry. `resource_flow_node` obtains the actual
parameter context, computes `resource_place_root` / `resource_place_path`, checks
the root state and shared-to-exclusive/overlap rules, then calls
`resource_place_hold` (`src_nano/typecheck.nano:1585–1607`). I capture at this
successful checker boundary, with the diagnostic delta and completed-function
conditions above. Its held-place strings are transient flow state, not an
independently retained certificate.

The proposed ASTCall fields carry the exact function owner token/current epoch,
original call-list index, enclosing call identity/argument ordinal, checked mode
and immutable root/path snapshots. An accessor accepts the original parser site
and exact original function, requiring positive current/completed epoch and
matching owner token; it must not accept caller-supplied trusted facts. A rebuilt
or synthetic call starts UNKNOWN. A shallow copy of the same unchanged parser
site may refer to that same event; a copied/rebound graph or new index may not.

Current ASTCall is eight scalar fields in `schema/compiler_schema.json`. Adding
these fields requires regenerating its C/Nano/list ABI and updating constructors
at `parser_store_call`, explicit borrow parsing, and `nisa_get_call`'s qualified
call adapter. The adapter is synthetic and must not copy checked events.
Generic-active/shared-template emission remains UNKNOWN even if a source
snapshot exists. No process-global event cache is proposed.

## Lifetime and rewrite prerequisites

C must never retain the temporary `OwnHold.path` pointer. A capture allocates a
single owned immutable payload for the root/path snapshot; it is released on
reset and AST_CALL destruction. Allocation failure reports the existing compiler
error and prevents publication; it is not relabeled an unsupported UNKNOWN.
All borrowed owner/site pointers are valid only while their source AST is alive.

I must audit PGO's bitwise AST clone (`src/pgo_pass.c:211`): a clone clears the new
payload and identity fields rather than sharing owned storage. DCE's retained
node move and TCO's node transfer must move ownership exactly once; a relocated
site cannot retain its old self-identity as checked evidence. Any rewrite of a
borrow argument, mode, enclosing call or field chain must invalidate that event
or its function epoch before publication. Emit-time comparison against the
snapshot is an additional guard, not a substitute for this mutation audit.

Nano snapshots must outlive temporary resource-flow strings across actual seed,
VM and native list/FFI paths. The generated list currently assigns structs by
value; that alone does not establish retention of newly added string/token
fields. I must map and qualify the existing compiler-schema marshalling/lifetime
rules, including replacement, copied parser state and collection, before schema
implementation is considered sound. If that ownership cannot be established,
implementation stops for a revised storage contract rather than adding a dangling
foreign pointer. Fresh raw Stage 1/Stage 2 bootstrap is mandatory after this ABI
change; a C-hosted compiler is not sufficient evidence.

## Actual emission routes

C borrowed lowering sees the original explicit argument AST_CALL in its CALL_REF
argument loop (`src/nanovirt/borrow_codegen.inc:915–939`). Nano sees the matching
original call-list index in the borrowed helper argument loop of
`src_nano/compiler/nanoisa_borrows.nano`. I collect a record only for that exact
emitted argument after ordinary lowering checks, using its retained checker event
and the current original source-function completion gate. I append advisory data
after executable constants, preserving their indices.

I do not reconstruct a checked event from `BorrowPlace`/`NBPlace`, numeric runtime
paths or `BORROW_PATH_*` opcodes. Those are independent lowering/runtime data.
Unsupported producer routes simply lack this advisory record. There is no claim
of whole-module collection completeness.

## Proposed independent transport

I propose a new `nano.borrow-place.v1` key; `nano.affine-use.v1` is unchanged.
The versioned payload identifies emitted function index, enclosing call and
argument source coordinates plus argument ordinal, coordinate provenance,
checked mode, source-path byte count, root-spelling byte count and projection
byte count, followed by counted immutable bytes. Exact byte offsets will be
fixed in the codec review before implementation. No runtime layout index,
capability, borrow duration or executable authority is encoded.

Mode 1 means checked shared and mode 2 checked exclusive. UNKNOWN is explicit
absence for raw/generic/unproved events, never a proved negative or permission;
lookup must distinguish absence from malformed/ambiguous known-version records.
If an explicit UNKNOWN record is later needed, it requires a documented codec
choice rather than silently reusing a positive mode. C coordinates identify the
original source; Nano coordinates identify the flattened compilation unit. Their
paths are not interchangeable nominal owners.

Root-only borrows use an empty projection; field chains use the existing checked
dot-separated field spelling. I retain at most the existing source profile's
32-field projection depth and propose a 4096-byte combined root/projection
snapshot bound. Exceeding advisory limits omits the optional fact without
changing accepted execution. Codec length/overflow/enum/reserved-byte checks are
independent of producer bounds. Unknown versions remain opaque transport data.
Malformed/duplicate site entries fail lookup without modifying its output.

Canonical assembly, load/save, module copy and assembly roundtrip must preserve
all bytes. A function-index/code-coordinate rewriter must remap or discard facts.
Stripping, duplication, malformed or forged advisory entries cannot authorize
execution, affect verifier acceptance or alter generated native code. I keep
existing typed ownership proof independent.

## Ordered acceptance before completion

1. Complete constructor/clone/reset/free and Nano foreign-list lifetime mapping;
   specify exact v1 codec and failure categories before code changes.
2. Prove paired checker capture on existing root, nested shared and disjoint
   exclusive call arguments. Reject overlap and shared-to-exclusive upgrade;
   preserve prior artifacts. Check both import orders and same-spelled owners.
3. Exercise raw, failed/repeated check, stale epoch, copied/rebound AST, generic
   and synthetic UNKNOWN; later-argument failure must invalidate the entire
   uncompleted function's prospective facts. Check snapshot lifetime independently
   of original temporary hold storage and borrowed parser copies.
4. Qualify codec truncation/overflow/duplicate/unknown version and allocation
   failure/recovery, copied module lifetime and canonical byte roundtrip. Force
   string/GC lifetime stress where supported; do not infer native success from
   a mock or C-only helper.
5. Run actual original VM and sanitized native fixtures; strip/forge facts and
   require unchanged executable behavior/native code. Preserve existing affine
   payload bytes and the unchanged ordinary producer subset.
6. Integrate reviewed changes, then run fresh raw generations/fixed point and
   exact Linux/Darwin gates. Full #998 stays open until its complete acceptance
   is met; this bounded borrow-event slice does not invent or close join facts.

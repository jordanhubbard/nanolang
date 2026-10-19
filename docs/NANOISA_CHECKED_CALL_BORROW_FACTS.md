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

I retain events in an invocation-owned ledger, never in owning AST_CALL fields.
Each entry identifies the original source function and explicit address node,
enclosing call, argument ordinal and check epoch. It owns copied immutable source
path/root/projection bytes. The synchronous checked producer owns and destroys
this ledger on every success/error exit before freeing the source AST. No caller
can supply a trusted completed ledger. A fresh function check invalidates its
previous segment; nested checks have independent function/epoch segments.

### Nano producer

The address is an `ASTCall` list entry. `resource_flow_node` obtains the actual
parameter context, computes `resource_place_root` / `resource_place_path`, checks
the root state and shared-to-exclusive/overlap rules, then calls
`resource_place_hold` (`src_nano/typecheck.nano:1585–1607`). I capture at this
successful checker boundary, with the diagnostic delta and completed-function
conditions above. Its held-place strings are transient flow state, not an
independently retained certificate.

I retain events in the private checked-emission invocation, keyed by exact
function owner token/epoch, original call-list index, enclosing call index and
argument ordinal. I do not add ASTCall fields. Only the final resource-check
phase (`typecheck.nano:7717`) collects; the earlier standalone function check
(`:6717`) keeps absent context. Program and shadow emission share this completed
context synchronously; the public result contains diagnostics/text only.
Generic-active, synthetic, rebuilt and raw emission have absent context.

## Lifetime and rewrite findings

C's `OwnHold.path` is temporary. The ledger copies it after the successful hold,
then frees its own bytes independently. Per-call ownership was rejected because
PGO clones by `*dst=*src` and overwrites calls/substitutions
(`pgo_pass.c:210,304,432`); DCE moves retained nodes (`dce_pass.c:244`), and TCO
moves nodes (`tco_pass.c:317,490,496`). Parser cloning reconstructs calls while
`free_ast` frees them. Invocation storage avoids introducing double ownership
at these sites. Exact identity/epoch validation still excludes transformed sites;
no optimization or semantic mutation is permitted between completion and use.

The managed VM route lowers generated list operations to ARR operations
(`nanovirt/codegen.c:2178–2284`), not foreign struct-list calls. STRUCT_LITERAL
transfers field owners; ARR_PUSH retains the record; ARR_SET publishes its new
edge before releasing the old record. Struct destruction recursively releases
fields (`nanovm/vm.c:4069,4123,4270`, `nanovm/heap.c:296,637`). Native translated
record-array roots trace every record, its string fields and array owner fields
before sweeping (`nanoisa/nvm2c_map_roots.inc:65–104`).

The C-hosted route is distinct. Generated `list_ASTIdentifier.c` shallow-copies
records and frees only list storage. `stdlib_runtime.c:413` builds concatenated
strings with `gc_alloc_string`; C transpiler scope cleanup deliberately excludes
STRING and array locals (`transpiler_iterative_v3_twopass.c:377–400`). Native
compatibility `gc_collect_all` does not collect cycles (`runtime/gc.c:311–320`).
I rely on bounded compilation-lifetime string/token storage here, and make no
per-event reclamation claim. I do not add frees for these borrowed strings.
Managed routes must independently qualify collection with a live ledger.
Fresh raw Stage 1/Stage 2 and fixed-point qualification remains required for new
checker-result/event record shapes even without an ASTCall ABI change.

## Bounds and failure policy

I allow at most 4096 events, 4096 combined root/projection bytes per event,
32 field projections, 1 MiB cumulative copied source-path/root/projection bytes,
and 131072 charged capture/path visits per invocation. I check additions,
multiplications and lengths before allocation. All capture and path traversal
work is charged; counters never refund on recheck or invalidation. Reaching a
limit disables and clears only this optional ledger; checker execution and
existing facts continue unchanged. Allocation failure is a hard compiler error,
not optional absence, and cannot replace a prior output artifact.

A later failed argument, function exit or final phase diagnostic discards every
prospective fact. Repeated checks invalidate earlier function segments and
emission validates the exact completed current epoch. No process-global collector
or reusable caller-supplied trusted context is permitted.

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
The fixed header is 40 bytes: offsets 0/4/8/12/16/20 are six LE u32 values
(function index, enclosing-call line/column, argument line/column, argument
ordinal); offsets 24/25 are mode and coordinate provenance; offsets 26/27 must
be zero; offsets 28/32/36 are LE u32 lengths of source path, root and projection.
The three counted byte strings follow in that order, with exact checked total
length, UTF-8 and no embedded NUL. Source path and root must be nonempty;
projection may be empty for a root borrow. A producer omits optional events
whose OS source-path bytes are not valid UTF-8, rather than rejecting an
otherwise accepted program. Decoder malformed-data refusal remains independent.
No runtime authority is encoded.

Both source lexers start at line/column 1. Explicit borrowed calls retain their
ampersand token coordinates (`parser.c:1429`, `parser.nano:2896`). All four
coordinates must be positive and fit u32. Unknown/zero/synthetic sites omit the
record; the decoder rejects zero in this positive-event version. Duplicate site
identity is function + both coordinate pairs + ordinal + provenance + source
path. Mode/root/projection are descriptive and cannot distinguish duplicate
sites. A duplicate with changed descriptive bytes remains ambiguous.

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

1. Review explicit C collector threading and the Nano final-phase bridge before
   production changes; retain the lifetime distinctions and codec above.
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

## Proposed implementation ownership and remaining interface review

I own the ledger/capture and advisory codec slice only after interface approval:
C `resource_flow.c`, `resource_tracking.h`, `typechecker.c` and its declaration
header, `nanovirt/main.c`, `nanovirt/codegen.h`/`codegen.c` and
`borrow_codegen.inc`; Nano `typecheck.nano`, `nanoisa_codegen.nano` and
`nanoisa_borrows.nano`; dedicated codec files/tests and this contract. I do not
change ASTCall schema, parser constructors, runtime borrow authority or the
blocked primitive files. Generated record/list providers are touched only if
actual new record generation requires them and are separately reviewed.

Proposed C shape: opaque invocation collector create/destroy, checked entry
wrappers taking a private optional collector, and capture into the current
function-check segment. Existing public type-check/codegen entrypoints remain
absent-context wrappers. The exact API threading through nested checks,
module checks, semantic shadows and runtime shadow execution remains the next
read-only review checkpoint; these names are interface roles, not an approved
signature patch. No global mutable collector is an acceptable shortcut.

Proposed Nano shape: a private call-borrow context in the existing checked
emission bridge; final resource checks accept an optional collector, while
existing standalone wrappers pass absent context. Internal emission accepts the
completed context and performs original identity/epoch validation. Public
`nanoisa_check_and_emit` continues returning only diagnostics and text; raw
wrappers preserve existing no-global and affine-use behavior. The actual
recursive ResourceFlow threading signature is still to be reviewed before code.

### Concrete C routing proposal for final review

`nanovirt/main.c:179–243` owns one Environment from before import loading through
root type checking, semantic shadows, runtime shadows and codegen.
`module.c:726` checks imported modules using that same Environment while setting
and restoring the actual cached module source identity. Capturing only the root
checker would miss those imported events.

I propose an opaque `CheckedBorrowLedger` owned by this Environment solely during
that synchronous invocation. Proposed private interfaces are:

```c
bool checked_borrow_begin(Environment *env); /* refuses an already active scope */
void checked_borrow_finish_capture(Environment *env); /* before runtime shadows */
bool checked_borrow_seal(Environment *env);  /* after all checks/shadows succeed */
void checked_borrow_end(Environment *env);   /* frees owned copies; sets NULL */
```

The Environment owns the scope while its nullable opaque pointer is attached;
this field is initialized absent by calloc and is never inherited as an owning
pointer.
`free_environment` is the error-exit backstop that destroys it. Success explicitly
ends the scope immediately after checked emission. This is neither process-global
nor persistent AST storage. No event-creation API accepts caller-asserted trust.
The capture functions stay checker-internal; lookup stays checked-emitter-internal.
Raw `codegen_compile` remains absent-context; a private checked variant accepts
only this sealed scope for that exact Environment/source invocation. Existing
non-NanoVirt callers never start a scope and preserve their current behavior.

`check_function_ownership` obtains the active unsealed collector into OwnFlow,
begins a segment for its exact function/epoch and records successful holds.
Nested function checks share the invocation budget but have distinct segments.
Capture is disabled immediately after semantic shadow checks, before runtime
shadow execution or lowering. Sealing is a separate step after runtime shadows
succeed; it does not resume collection.
`type_check`, `type_check_module` and semantic shadow signatures need not change.
After sealing, further checker calls cannot add facts: changed epochs invalidate
old events. Runtime shadow failure never seals/publishes. Import failure destroys
the ledger before cached AST teardown; failure copies remain separately owned.

This scoped Environment-pointer choice still requires coordinator approval; it
avoids broad extra parameters across import-loader/checker APIs but adds explicit
Environment lifecycle ownership. It is not implemented by this contract.

### Concrete Nano routing proposal for final review

I propose `BorrowFactEvent` plus `BorrowFactContext` internal records: events,
monotonic budgets/disabled/error status, and exact function owner/epoch/site data.
The optional context is an empty-or-one-element array, so legacy wrappers can
pass `[]`. Capture mutates only its owned event/budget arrays, not Parser fields.
`resource_check_function_context(..., context)` owns segment begin/completion;
existing `resource_check_function(...)` delegates with `[]`. Internal resource
flow helpers thread this optional context recursively; their existing wrappers
and checker shadows preserve absent-context behavior. No function-local global
collector is used.

The final `typecheck_advisory_core` resource pass supplies this context; earlier
`check_function` resource checks do not. On any final diagnostic it returns no
events. `CheckedAdvisoryOutput` carries the completed internal context through
`nanoisa_check_and_emit` into its private program/shadow emitters; public
`NanoisaCheckedText` still exposes only diagnostics/text. Lookup validates exact
original function token/completed epoch and original call/argument indices, and
never synthesizes facts for the qualified-call adapter or a generic clone.
A standalone/raw emitter passes absent context while preserving all existing
no-global effects and affine-use facts.

Additional proposed C ownership is `src/nanolang.h` (opaque pointer declaration)
and `src/env.c` (lifecycle backstop), plus a dedicated private ledger module/header;
this replaces the earlier broad typechecker signature changes. Module loader
behavior and cache ownership remain unchanged. These file/interface choices are
pending final review, not production authorization.

### Environment lifetime audit and cleanup order

The actual C Environment (`nanolang.h`, ending at line 930) has no parent pointer.
`create_environment` uses calloc (`env.c:228`); targeted allocation, whole-struct
assignment and memcpy inspection found no shallow Environment clone. Imported
checks share the same pointer, not an inherited owning field. Separate REPL and
other driver environments start with absent collection. No parent lookup or
implicit copying of the scope is permitted.

The private begin/finish-capture/seal/end operations operate only on that exact
Environment. Begin fails rather than replacing an active scope. End first clears
the pointer, then frees ledger-owned bytes; it never dereferences AST identities.
`free_environment` calls end before other teardown as a backstop for every early
import/typecheck/shadow/codegen failure. Some existing driver branches free the
AST before Environment, so destructor independence from AST is mandatory.
Success ends collection immediately after checked codegen, before package/save
or actual VM execution. The existing late VM verifier exit at main.c:411 skips
Environment teardown, but has no live ledger under this ordering; I do not expand
this slice into unrelated driver cleanup.

Cached imported ASTs not actually checked during this invocation have no matching
ledger entries. They remain absent, never inferred from a stale completed epoch.
The current driver clears its module cache before processing imports, while
reusable library callers retain this conservative missing-entry behavior.

### Final recursive/interface outline

The private C header declares the lifecycle signatures above and:

```c
CodegenResult codegen_compile_checked(ASTNode *program, Environment *env,
                                     ModuleList *modules, const char *input_file);
```

This entry obtains only the sealed ledger attached to the exact Environment;
the shared internal codegen routine gets an optional borrowed `const` ledger.
Existing `codegen_compile`, `codegen_compile_shadows` and
`codegen_compile_shadow_scope` pass NULL. Runtime shadow emission therefore
cannot publish these events. The checked production emitter validates the
source function/call/epoch again at lookup. The emitter cannot create an event.
Private checker helpers begin/end function segments and capture holds into
`OwnFlow`'s borrowed active collector. Internal module/source identities are
captured while the actual checker has them selected.

For Nano I propose internal records `BorrowFactEvent` (exact owner token, epoch,
function index, enclosing/address call indices, ordinal, mode, coordinates,
source/root/projection strings) and `BorrowFactContext` (event array plus a small
shared integer state array for enabled/hard-error/counters). No Symbol or Parser
is stored. Empty `array<BorrowFactContext>` means absent collection.

I split these three existing recursive entrypoints into `_context` internal
implementations with one appended `context: array<BorrowFactContext>` parameter:

- `resource_flow_block` (currently line 1131)
- `resource_flow_value` (currently line 1275)
- `resource_flow_node` (currently line 1346)

Their existing signatures remain wrappers passing `[]`; all recursive internal
calls pass the same context. `ResourceFlow` and its state/next semantics remain
unchanged. The existing `resource_check_function` signature remains an absent
wrapper around `resource_check_function_context(parser, func, diags, file_name,
context)`; only the final phase calls the contextual variant. Branch copies may
share the ledger because it records static checked sites, not runtime branch
execution. No branch joins or path summaries are inferred from ledger contents.

`CheckedAdvisoryOutput` gains an internal optional borrow context. Every empty or
failed result sets it absent; successful final diagnostics return it with the
existing owner/mask arrays. The current synchronous program/shadow internal
emission context carries it to borrowed lowering. Public/raw wrappers supply
absence and keep prior facts unchanged. Capture limits disable/clear only this
context; allocation failure creates a diagnostic and yields no assembly. This
record-shape addition requires fresh raw stages after local paired qualification.

### My module-boundary correction

My first compiler build rejects cross-module calls to private helper functions.
I do not fix that by exporting an emitter accepting caller-supplied checked
context. My borrowed lowerer instead exports `nanoisa_borrow_lower(parser,
shadows, first_shadow, collect_sites)`, returning assembly and untrusted source
site descriptors with the actual unchanged emitted-to-source function map.
Each descriptor contains source/emitted function indices, enclosing/address call
indices and argument ordinal. The lowerer never sees a checked context. My raw
wrapper disables collection. I bound staging to 4096 sites per invocation;
overflow clears all optional descriptors, while normal lowering continues.
Failure returns no usable descriptors or assembly. Synthetic shadow entries do
not receive descriptors.

Only my private checked codegen bridge consumes this result and the context
from its own successful synchronous checker invocation. I validate emitted
function range and exact source mapping, exclude generic/synthetic sites, then
revalidate owner, completed epoch and actual call/argument identity before
appending the new trailing advisory records. No public entrypoint accepts both
caller text and trusted summaries. This preserves existing owned/general
lowering selection without introducing another checker invocation.

My C ledger additionally snapshots the callee expression identity and bounded
callee spelling; these copied bytes count toward its 1 MiB cumulative capture
budget. Lookup uses a separate bounded per-query comparison, never clears
sealed storage, and never allocates a replacement projection. The Nano lookup
also compares the retained projection against existing field bytes without
concatenating compilation-lifetime strings. These per-query bounds do not claim
a cumulative bound on all downstream assembly text allocation.

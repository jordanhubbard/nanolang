# I stage private FLOAT-array facts inside unique owners

I refine task_430220ce190946518d404088533531b6 under my existing
[owned-array contract](NANOISA_OWNED_FLOAT_ARRAY_FIELDS.md). My audit pin is
canonical118cc44711b1cb1b49a1503d948c4e29c428b9a7, including mixed runtime819,
STRING runtime813 and service guard821. This is a preimplementation plan: no
producer, shared validator, public selector, runtime or fixture is changed.
Samples source child e64a2673 remains in qualification and precedes complete
source acceptance; it does not prevent private descriptor planning.

## I identify the remaining boundaries precisely

- mixed_layout_view.inc currently rejects RESOURCE rows that are not scalar
  trees, including STRING and ARRAY leaves. Its classes deliberately confer no
  execution authority. I cannot clear RESOURCE to reuse its ordinary map.
- mixed_float_proof.c represents owner values as opaque nominal identities.
  mf_scalar_layout admits only INT/BOOL/U8 leaves; OWN_PACK checks scalar tags,
  OWN_UNPACK_LOCAL recreates only tags, and CALL returns a nominal owner without
  field origins. Those transfers cannot prove owner ARRAY elements or aliases.
- mixed_samples_prepare.inc currently requires owner+array origins and a
  nonempty ordinary managed-record mapping. Bundle plus Handle need not contain
  an ordinary record at all; a zero-record managed runtime must remain distinct
  from absent ARRAY provenance. Its signature and scalar policy also exclude
  STRING from this mixed route, while the separate owned STRING route is already
  qualified. I preserve that route until explicit coexistence qualification.
- ownership_contracts.c and affine_state.c retain shared strict field/local
  rules. Existing STRING support is not permission to treat ARRAY as a scalar.
  Whole-root reference checks must inspect transitive managed leaves, not merely
  the selected scalar path. Bare ARRAY descriptors do not prove FLOAT.
- The VM owner shell holds NanoValue fields and transfers them during pack/unpack.
  The generated nown_record shell holds nown_value fields; its mixed category2
  already retains/releases NmsRuntime handles recursively. These are useful
  mechanisms, not current admission evidence for arrays inside owners. A managed
  handle must stay attached to its invocation runtime through calls and returns.

## My first production checkpoint is a private descriptive query only

I propose new internal owned-array layout query files. I reuse retained layout
byte preflight/decoding, but leave nvm_describe_mixed_layouts and all its existing
callers unchanged. A new opaque descriptive result owns decoded fields, exact
ownership bytes and numeric maps; no caller-supplied descriptor or proof can be
converted into runtime authority. A failure leaves the caller's result pointer
and every query output unchanged.

I require explicit COMPLETE flags with RESOURCE identity preserved. Owner graphs
may contain INT/BOOL/U8, retained STRING, flat ARRAY/NO_INDEX leaves and strictly
prior owner-record children. ARRAY is described as pending origin proof, never
as proved FLOAT. FLOAT owner fields, ordinary-record children inside owners,
array element descriptors inferred from nominal absence, unknown flags, generic
or union owner layouts remain refused in this first query. Existing ordinary
records retain their own exact global/source/compact-managed identities. Only
ordinary rows enter a compact managed-record map; owner rows never do.

I retain the256-layout and65,536 direct-field limits and16MiB ownership transport
limit. Prior-index owner edges establish a finite DAG; depth is at most32.
I expand retained leaf paths with a checked65,536-path total budget and at most32
field indices per path, checking addition/multiplication before storage. Shared
DAG subgraphs may appear under distinct paths but cannot cause unbounded work:
I reject before exceeding path/storage budgets. Every row receives separate
has_array and has_string transitive facts, exact source/global identity and its
original authority flags. A scalar sibling path under an ARRAY-bearing root is
still marked unsuitable for whole-root borrowing. No layout fact states that a
runtime owner is live or that an array element has any particular tag.

This first checkpoint does not interpret instructions, instantiate call
summaries, alter shared ownership validation, allocate runtime objects, update
public candidates, or emit code. I return its production for independent review
before adding query fixtures or executing any new input. Later private query
qualification covers original maps, shared DAG/depth bounds, STRING coexistence,
unknown/forward/cyclic refusals and allocation-failure output preservation.

## My second checkpoint proves closed field origins across the whole module

Before executing owners with arrays I require a separate reviewed instruction
analysis over immutable module bytes. I retain8 acyclic functions,256 locals and
stack cells per function,4,096 instructions,64 concrete allocation sites,
1,048,576 state cells,65,536 field facts and262,144 worklist visits. Every body,
including unused helpers and selected shadows, participates. Limit exhaustion
is a checked refusal, not acceptance with partial summaries.

For each function I allow at most256 symbolic input ARRAY-leaf paths across its
owner parameters and at most256 ARRAY-leaf result paths. Each field fact uses a
finite set of64 concrete sites plus256 formal-path symbols (five64-bit words),
with explicit unknown/uninitialized state and exact tags/nominals. Result facts
are summarized bottom-up in the acyclic call graph. At a call, formal symbols
are substituted with the actual owner's exact path facts; concrete callee
allocation sites retain their static identity. Summaries never assert that two
runtime allocations at one static site are the same object. They prove possible
FLOAT-producing origins, while the independent affine analysis proves unique
shell transfer. Budgeted joins union all alternatives; no nominal-only result
or argument assertion substitutes for field facts.

OWN_PACK records each exact field fact, OWN_MOVE/STORE transfers the shell's
provenance, observation retains a nonescaping shell relation while yielding the
array origin set, and unpack distributes field facts to separately rooted
values. Nested owner paths and returned owners keep these facts. Unknown or
uninitialized origins refuse; every possible origin must be a checked FLOAT
ARR_NEW/LITERAL and every mutation through every alias requires FLOAT. ARR_GET
remains FLOAT|VOID. STRING siblings retain their distinct tag/lifetime without
being converted into managed-array origins. No bare ordinary array parameters
or results are added by these owner-parameter/result summaries.

## My later runtime checkpoint preserves distinct ownership mechanisms

Only after private proof qualification do I review the fresh complete
structural+affine+origin+scalar conjunction and its explicitly selected runtime
path. I do not globally relax the shared ownership descriptor validator. Any
internal checked delegation must be created by fresh analysis of the same
immutable module, preserve original RESOURCE/layout bytes, and be inaccessible
as an unchecked caller-built state constructor. Existing public selectors remain
unchanged until a separate full admission review.

The VM retains ordinary array NanoValue roots independently from the unique
owner shell. Native mixed nown_value fields retain NmsRuntime+NmsHandle identity;
the root invocation owns that runtime until stack, local, nested shell, callee
and pending-return roots are drained. I stage all fallible retains before
publication, unwind completed retains on failure, and transfer pack/unpack roots
exactly once. An external alias survives consuming its shell. Shared mutable
ARRAY identity remains observable after writes/append growth through any alias;
failed growth preserves old storage and all aliases. Runtime array access keeps
current bounds and FLOAT-or-VOID behavior; no fabricated default is introduced.

Required execution controls include two fields sharing one array, multiple
shells retaining one ordinary array, nested owner moves/calls/results, observation
before consumption, reversed unpack, mutation/growth, trap/return/frame cleanup,
allocation/retain failure with exact prior output prefixes and later recovery.
Those controls are prospective and require a reviewed runtime checkpoint. The
unchanged original Bundle/PREFIX source and all shadows remain later required;
private local-only success cannot close calls, nested results or source430220.

## I preserve every consumer boundary until its own review

I audit and then qualify ownership_contracts, affine_state/affine_bytecode,
reference_places, mixed_layout_view, mixed_float_proof, mixed_samples composition
and preparation, general/function/max-stack/linked/closed verification,
nvm_v2_convert, all VM entry/continuation/call APIs and generated nvm2c_owned
carrier paths. managed_array_shapes and nvm2llvm/LLVM/Wasm profiles retain their
current refusals. Both source producers and the ordinary/owner selector retain
current boundaries until separate source review. No new opcode, wire layout,
service import, external Result ABI or source FLOAT-owner-field admission follows.

Required-service presence rejection remains before candidate/proof/runtime
selection, including malformed or partial metadata; neither owner ARRAY facts
nor an ordinary companion permits service fallback. Whole ARRAY-bearing borrowed
roots (including scalar siblings), linked graphs and unsupported closed profiles
remain refused. Existing nonmixed STRING-owner and mixed ordinary Samples paths
must preserve their qualified behavior. Private planning does not narrow the
original430220 or managed/affine/full-product parents.

## My first private production checkpoint

I add owned_array_layouts.h/.inc through retained_layouts.c so I reuse its exact
allocation-free byte preflight and existing ownership transport reader without
changing either or their callers. The new opaque query owns decoded fields,
original layout/ownership bytes, source/global/ordinary-managed maps, per-row
flags and pending ARRAY/STRING facts, plus bounded expanded owner leaf paths.
Passive non-record rows keep no record/managed identity; complete ordinary rows
with pending ARRAY leaves also keep NO_INDEX until a later proof. Empty owner
rows remain describable. Depth and path length are independently bounded32,
with65,536 total expanded leaf paths; ordinary nested rows remain prior-only.

I reject service claims before description and preserve the existing whole-root
reference validation. No mixed/owned verifier, selector, VM, native emitter or
source producer calls the new query. Existing dependency-file generation tracks
the added include/header through the modified retained-layout translation unit.
I have inspected source and whitespace only; query fixtures and compilation await
independent review. This checkpoint establishes neither element origins nor
owner liveness and does not close430220.

# My checked resource field-read projection design

I record a contract-only continuation of #998 on
`design/5.1-checked-field-read-facts` from `c56b6ef67`. I change no schema,
production source or tests and run no builds at this checkpoint.

## Existing requirement and current qualification

My release scope row10407 says: **Preserve purity, affine, generic, effect, and
exhaustiveness frontend facts in module metadata.** Issue #998 requires a mapped
checker origin, versioned representation, transport, lookup and unchanged
semantics after stripping or forging advisory records. It requires applicable
facts through both fresh producers, including imported/generic identities and
negative cases, rather than assuming absence is a proved negative.

My `NANOISA_AFFINE_USE_FACTS.md` explicitly leaves broader projection/path
information open. My prior call-borrow component now retains checked argument
place/mode. Aggregate `c56b6ef67` records its completed 17-step fresh bootstrap,
identical raw Stage1/Stage2 bytes and metadata neighbors. Exact release-platform
qualification remains separate. I do not repeat the obsolete claim that these
new compiler records still lack any fresh-stage evidence.

My implemented families include declared purity, bounded inferred/global effects,
generic specialization provenance, checked match coverage, affine completion,
identifier OBSERVE/CONSUME and call-borrow place/mode. Generic template UNKNOWN,
callback UNKNOWN and bounded-analysis UNKNOWN do not establish a missing proved
fact. I retain the imported call-borrow Nano helper-profile refusal explicitly;
C-only acceptance cannot establish paired imported coverage.

## Concrete information still discarded

C `src/resource_flow.c`, `own_node`'s `AST_FIELD_ACCESS` branch, calls `own_place`,
looks up the lexical owner, checks capture/restriction/move/conflicting holds,
and frees the path. After success it records only root identifier OBSERVE.
Nano `src_nano/typecheck.nano`, `resource_flow_node_context`'s field-access branch,
checks `resource_place_path` against held places and similarly retains only the
root identifier fact. These are actual successful checker events, not new
inferences from syntax or executable ownership tables.

My existing accepted fixtures provide exact examples:

- `source_borrow_shared.nano`: `read(view: &Counter)` reads `view.value`.
- `source_borrow_exclusive.nano`: `update(view: &mut Counter)` reads
  `view.value`, `view.active`, then `view.value`; main reads owned `owner.active`.
- `source_borrow_nested.nano`: main reads owned `root.branch.left.value` and
  `root.branch.right.value`; the helper reads borrowed leaf fields.

`tests/test_affine_use_metadata.py::test_borrowed_resource_roots_observe_without_consuming`
already checks shared/exclusive helper OBSERVE sites with VM/native execution.
The nested source corpus already executes its deeper paths. New assertions must
prove each emitted read's retained projection and root category independently;
existing root counts do not prove that new metadata.

## Bounded result, not expanded runtime authority

I propose positive checked FIELD_READ facts only for those existing accepted
resource paths. Root category distinguishes owned resource, shared reference and
exclusive reference; an exclusive-reference read is still a read, never mutation
or consume. Root and field spellings are descriptive. They do not identify a
nominal type, authorize a field offset or create a loan.

I exclude assignment destinations, destructuring-generated projections, union
payloads, ordinary non-resource records, arbitrary indexing, borrowed escapes,
generic-instance inference and new executable profiles. Assignment RHS reads
are eligible existing field-read occurrences. I do not relabel writes as reads.
Nested reference types remain absent unless the existing observation classifier
actually accepts them. I add no join certificates or control-flow summaries.

## Proposed capture and lifetime interfaces

I reuse the invocation-owned scope and its begin/disable/seal/end lifecycle,
not AST-owned strings or a persistent parser cache. I add a separately typed
read-event vector to that scope; call-event identity and existing payload bytes
remain unchanged. Shared total event, snapshot-byte and capture-work accounting
must cover both vectors (4096 events, 1 MiB copied bytes, 131072 work, path depth32,
root-plus-projection4096 bytes). Exhaustion clears/disables optional publication
for the invocation; allocation failure remains a hard error preserving outputs.
Sealed lookup never clears or reallocates retained event storage.

Proposed private C roles:

- `checked_field_read_capture(env, function, field_node, root_node, root_kind,
  root, projection)` is called only after every existing field-read check
  succeeds. It copies owned strings and snapshots exact source pointers,
  function epoch, coordinates and path topology. It does not infer success.
- `checked_field_read_lookup(env, function, field_node)` requires a sealed scope,
  exact current/completed owner epoch and matching current source shape. It
  returns a borrowed immutable event valid until scope end. Destruction frees
  only owned snapshots and never dereferences AST nodes.

Nano adds a separate typed read-event array to the existing invocation context,
using original parser field/root indices and the exact one-cell owner token plus
completed epoch. The final successful checker phase owns capture. Existing raw
wrappers still receive absent context. Rechecking invalidates old facts; failed
later checks discard all prospective facts. Generic/rebuilt/synthetic functions
cannot borrow another source owner's completion. C-hosted strings retain the
existing bounded-per-invocation compilation-lifetime limitation; I claim no
per-event reclamation.

## Exact emitter occurrence

C `borrow_expr` handles `AST_FIELD_ACCESS`: ordinary and union paths are separate;
resource paths emit owner `AGG_GET` or inherited/temporary-reference `REF_GET`.
I stage a read descriptor only at the actual successful resource read lowering
for the original field node, never by scanning a whole function or its root.
Failure of later lowering prevents the whole module's publication. Nested place
resolution is not several executed reads merely because its syntax has several
field nodes; only checker events matched to actual emitted occurrences qualify.

Nano `nb_expr` has the corresponding field branch. Its public lowerer may return
bounded untrusted read descriptors (original source-function/field indices and
emitted function index) beside the existing call descriptors. It never accepts
trusted context. Private `nisa_checked_borrow_lower` validates source/emitted
mapping and looks up the actual checked event before appending metadata.
Raw lowering disables staging; descriptor overflow omits optional records.
Program/shadow reset replaces staging arrays and must not invalidate a previously
returned descriptor result. Synthetic shadow entry records remain absent.

I must audit any additional ordinary emitter route before claiming it covered;
this contract qualifies the established resource lowering route first.

## Proposed independent wire contract for review

I propose `nano.field-read.v1`, without changing `nano.affine-use.v1` or
`nano.borrow-place.v1`. A 28-byte fixed header holds LEu32 emitted function,
field line, field column; one root-category byte (1 owned, 2 shared, 3 exclusive),
one coordinate byte (1 original C, 2 flattened Nano), two zero reserved bytes;
then three LEu32 byte lengths for source path, root and projection. Counted UTF-8
bytes follow. All three strings are nonempty, no embedded NUL; projection starts
with a field separator and contains a checked bounded field chain. Coordinates
are positive. Invalid source path encoding omits optional producer facts rather
than rejecting otherwise accepted source.

Duplicate site identity is function, coordinates, coordinate provenance and
source path, independent of category/root/projection. Consumer lookup fails
closed on malformed known-version or duplicate entries; unknown versions remain
opaque. Reordering functions must remap or drop these records. Copy and canonical
transport retain complete counted bytes; no hidden string-pool references exist.
The serialized descriptor never replaces exact internal source identity.

## Required acceptance before implementation is complete

I first review these interfaces and root-category classification. Then I require
paired existing-fixture facts for owned/shared/exclusive roots, nested reads,
multiple same-spelled bindings and actual branch-local read occurrences. I test
raw/generic/synthetic absence, failed checks, repeated checks, stale/copied owners,
later lowering failure and prior-output preservation. I do not infer imported
parity from an executable profile that one frontend refuses.

I require ledger allocation-denial/recovery, active-scope teardown, shared total
budget limits, copied descriptor lifetime across program/shadow reset, exact
codec overflow/duplicate/encoding controls, full canonical transport and actual
VM/sanitized-native behavior. Strip/forge/duplicate records cannot change native
code or acceptance. Existing old metadata and executable authority remain byte
stable except independently proved exclusive new advisory pool suffixes.
New result shapes require fresh raw generations/fixed point and exact Linux/Darwin
qualification. Full #998 remains open until its complete mapped acceptance passes.

## Proposed file ownership

After review I would own the ledger providers and codec, resource-checker capture,
resource emitter descriptors/private bridge, focused tests, contract and receipt.
The coordinator owns aggregate workflow wiring and bootstrap. This design grants
no production-edit or build authorization by itself; blocked primitive files
remain untouched.

## Reviewed implementation checkpoint

I now implement the approved interfaces on `fix/5.1-checked-field-read-facts`.
C physically clears both event vectors on optional capture overflow. Nano disables
both lookups through shared state while retaining already bounded invocation
storage; I do not claim physical per-event reclamation there. Both emitters stage
all descriptors before serializing either family, and a shared staging overflow
omits both. My initial C ledger, 1377 codec checks, 14 strict GCC/Clang checks and
three-fixture C source method pass. My final paired local qualification now passes;
I retain [the final receipt](evidence/checked-field-read-20261019/README.md).
Fresh aggregate raw generations and exact platform gates remain pending.

# My paired File companion resolution plan

I begin from actual PR927 merge `8ed0a0ff6e30969b9721aa0cf89db011ec4f4506`.
My parser/retention and bounded cleanup acceptance is recorded under
`task_8bbc1cf5295b4b59b314640ef57c725f`; that full source task remains open.
This is the next preimplementation checkpoint of
[NANOISA_FILE_SOURCE_IMPLEMENTATION.md](NANOISA_FILE_SOURCE_IMPLEMENTATION.md),
not permission to execute generated File source. I retain parents6fc/72556/6931/d03c
and the required cyclic, indirect, richer-borrow and mixed source conjunctions.

## My observed integration points

The actual publisher emits one service declaration followed by five behavior
shadows. It does not emit eight ordinary type definitions or five externs.
Resolution must introduce the exact eight catalog types and five methods; it
cannot discover authority from coincidentally named ordinary declarations.
`nl_file_binding_prepare` validates complete immutable document bytes and renders
canonical interface/source bytes. `nl_file_source_plan_build` and independent
Nano `compiler/file_source_plan.nano` require complete already checked namespace
records. Neither establishes a source module's origin on its caller's behalf.

C `load_module_internal` currently parses, recursively processes imports and can
subsequently prepare module build/header metadata. `process_imports` deliberately
refuses service nodes. Nano `merge_with_imports` retains canonical `files` and an
exact `source_files/source_lines` row for each retained merged line;
`bind_merged_functions` then rewrites names through the ordinary binding tables.
I must associate service declarations before those name changes. The last
`file_starts` boundary alone is not an original-file line or a source identity.

I factor source read/lex/parse and import graph collection from C's effectful
module preparation, using the existing parser and import resolver. I add the
matching collection stage to the actual Nano driver, using its existing parser
and resolver. There is no new grammar parser or C implementation of Nano name
resolution. Ordinary driver behavior remains unchanged. For service graphs the
new preparation stage may read declared compiler inputs, but it does not invoke
module Make commands, load foreign libraries, run shadows or acquire File grants.
Existing execution/export refusals remain after preparation until later reviewed
lowering and driver conjunction. A successful descriptive report cannot bypass
those guards.

## My invocation-owned graph and origin identity

I retain every selected source module once by its canonical absolute path under
the existing stable-ancestor filesystem precondition. Multiple imports of that
path identify one original module; equal basenames, equal explicit module names
or hard-linked paths do not merge different canonical paths. Path aliases resolved
by realpath refer to the same module. I retain the source bytes actually parsed
and their original path/line/column mapping; a later read cannot replace them.
Input providers remain stable throughout synchronous collection. Concurrent source
mutation is outside that precondition, not made safe by an after-the-fact hash.

I use invocation-local origin ordinals sorted by unsigned canonical pathname
bytes, independently in C and Nano. Zero is invalid; AST `origin_index` stores
the corresponding zero-based index and remains -1 until association. A published
resolution owns its origin table; an index without that owning table is never
reusable authority. C annotates only this invocation's owned parsed/cloned ASTs,
not cached module ASTs. A reused clone must be rebound against its new table.
Nano constructs updated ASTServiceDecl records and a complete retained Parser;
all ordinary Parser copies preserve the service list as qualified by PR927.

For every service node Nano checks its merged line against both source map arrays,
requires a unique matching canonical path in `files`, and retains the original
line and column. C attaches that same original path before flattening imports.
I refuse missing, ambiguous or contradictory mappings. I retain original import
edges and owner paths separately from generated function/type names. A module's
textual declaration name and import alias are visibility inputs, not its identity.

I preserve existing import-cycle policy, dependency selection and source limits.
The new resolver additionally caps the selected graph at 5,000 modules and
50,000 retained lines, with 20,000 lines per file; a lower configured existing
limit remains effective. It inherits the descriptive plan's 16 service requests,
64 service aliases, 256 ordinary namespace rows and 1MiB copied-text bound.
These are explicit preparation refusals, not full 5.1 acceptance exclusions.
I use checked linear/indexed passes or charge bounded repeated comparisons;
source review must pin the actual allocation/work table before fixtures.

## My companion snapshot API and ownership

I propose `src/file_companion_snapshot.h/.c`, with the following owning C surface:

```
typedef struct NlFileCompanionSet NlFileCompanionSet;
NlFileCompanionStatus nl_file_companions_prepare(
    const NlFileCompanionRequest *, size_t, NlFileCompanionSet **out);
void nl_file_companions_free(NlFileCompanionSet *);
```

Each request contains counted canonical declaring-module path, counted relative
companion path, counted interface ID, catalog version and original location.
It contains no declaration IDs, AST pointer, visible alias or lowering decision.
Statuses distinguish INVALID, LIMIT, MEMORY, UNRESOLVED and IO from OK; the report
retains first errno/stage/request and secondary close failure separately.
Failure preserves the caller's output. All getters copy fixed facts or return
counted immutable views with explicitly set lifetime until set destruction.
No getter hands out a mutable NSI tree or runtime capability.

I anchor relative companion lookup at the declaring source's directory, not the
root source or process working directory. I reject absolute paths, empty path
components, dot/dot-dot components, embedded NUL, malformed UTF-8 and excessive
counted paths before opening. I permit bounded nested relative components and
retain the stable-ancestor precondition; the final component uses O_NOFOLLOW.
An opened companion must be regular. I read at most the qualified strict 1MiB
limit, explicitly check EOF beyond the boundary, reject short/inconsistent reads,
and close once. A read or close failure never publishes a successful snapshot.
I do not use pathname existence checks as authorization for a later open.

Each accepted snapshot retains exact original bytes, canonical validated interface
bytes and immutable full catalog facts. I call `nl_file_binding_prepare` once per
request; its complete strict validation is mandatory. Static catalog getters do
not turn an unvalidated document into a catalog. The interface/version must agree
with the parsed declaration. I never reopen the companion during checking or
lowering. Identical document contents in two modules do not unify their types.

I cap simultaneous project-requested snapshot heap storage at64MiB. Before each open
I reserve a fixed input buffer of MAX_BYTES+1 and descriptor/path bookkeeping;
there is no unaccounted realloc overlap. Before strict preparation I reserve its
nonallocating `nl_file_binding_allocation_bound` result alongside all retained
prior snapshots and live input/path buffers. That nested bound already contains
the strict plan's peak: I count it once. Once preparation returns, I replace its
reservation with `nl_file_binding_storage_size` while retaining exact source
bytes and all copied catalog/path storage. Any additional copy is charged before
allocation while both old/new storage coexist. I report named automatic buffers
separately; actual compiler stack
frames, nested strict-reader stack, libc/kernel/allocator overhead are not
project heap measurement.
A following request can fail LIMIT without changing a previously published set,
because publication occurs only once after all requests and close operations.

The Nano bridge exposes this data-only set with a checked owning opaque handle,
size/count and copied getters, plus explicit destroy. Its actual ABI/manifest,
handle validity and all provider closures require source review. It neither reads
Nano ASTs nor allocates declaration IDs, resolves aliases, chooses types or emits
instructions. Nano independently builds its own `FileSourceRequest`/alias/ordinary
records and calls its own plan implementation. C calls the existing C plan.
Nano allocation failure remains process-fatal where its runtime is fatal; I do
not claim C-style transactional recoverable OOM equivalence. No output artifact
is created by this checkpoint. Later publication requires the supervised staging
and cleanup contract already specified in the parent design.

## My complete namespace and nominal identities

I collect all ordinary declarations and import edges before registering service
names. Each service declaration introduces the catalog's exact13 bindings in its
original module, in catalog order. I refuse duplicate service declarations in one
module, missing catalog fields, unsupported versions and collisions with ordinary
values/types in the relevant existing namespace. Shadows are behavior declarations
attached to their target; their expected method names are not duplicate methods.
I do not count a declaration twice because a module was imported twice.

My canonical declaration key is `(origin, declaration kind, local declaration
identity)`. A service binding adds `(service declaration, catalog kind, ordinal)`.
I allocate positive bounded IDs deterministically from sorted original module
paths and original declaration order, with the13 synthetic service entries in
catalog order. These are invocation IDs, not wire layout/import indices.
Normalized paired reports include full original keys; equality is not inferred
from equal integers allocated by two independent producers.

Aliases map a visible `(importing origin, namespace, spelling)` to one original
key. I follow actual selective/wildcard/module-alias visibility and export rules,
retain complete original import edges, and reject ambiguity or cycles rather
than resolving by first match. Alias chains terminate in an original declaration;
aliases never acquire their own nominal type. Reordered import traversal must
produce the same normalized original keys. A private unexported declaration does
not become visible because a catalog spelling matches it. The generated service
bindings have the module-public visibility promised by the parser contract.

Both producers submit complete checked namespaces to their existing descriptive
plans. The plan cannot be called on an early partial import list and later treated
as final. Its global_layout/import_index fields remain NO_INDEX. I retain eight
exact type identities, five method identities and method input/result relationships;
I do not classify arbitrary opaque/union types as File or ordinary FFI.

The next nominal propagation source checkpoint, reviewed separately, adds an
explicit service-origin/declaration/kind/ordinal identity to C TypeInfo/resolved
expression/signature facts and equivalent Nano checker tables. Ordinary types
keep an invalid sentinel. Its audit must enumerate all constructors, deep copies,
function signatures, equality/joins, generic/import transport and exports. Same
catalog ordinal in different declaring modules is unequal; two aliases of one
original key are equal. Primitive payload fields remain ordinary INT/BOOL;
FileError/ReadByte and every Result retain exact owning type identities and member
order. No type propagation reaches executable admission until both independent
lowerers and matched byte validation are implemented and reviewed.

## My staged source and acceptance checkpoints

1. I implement bounded data-only snapshots and actual paired graph/origin
   association, with complete namespace construction and descriptive report
   ownership. I review the full production allocation table and all bridge/driver
   failure paths before fixtures. Existing service execution refusals remain.
2. I add fixtures from the actual publisher, then review before execution:
   transitive/reordered imports, multiple aliases, equal basenames/module names,
   wrong-root companion decoys, shared document bytes in distinct modules,
   namespace collisions, forged lookalikes and all catalog field mutations.
3. I cover open/read/EINTR/close/allocation failure prefixes, symlink/nonregular
   files, limit boundaries, source/document mutation after snapshot, input free,
   getter lifetimes and untouched output sentinels. Faults are precisely labeled;
   no generic malformed/OOM ambiguity becomes a fabricated INVALID verdict.
4. I compare independently normalized C/Nano reports and retain every helper and
   module shadow selected by actual fresh C seed/Stage1/Stage2. Both Linux and
   Darwin use exact tool/source/provider inventories and supported selected
   sanitizers. No generated service behavior shadow runs yet; its eventual
   execution remains mandatory, not replaced by these helper shadows.
5. I review and qualify nominal propagation, both independent lowerers, selected
   initializer/shadow graph, explicit per-invocation grants and installed staged
   publication in dependency order. No report flag authorizes execution; public
   byte APIs continue their own fresh catalog/flow/runtime checks.

I retain shared v3 ordinary union/array coexistence requirements. This resolver
assigns no new envelope kind/version, collapses no ordinary authority, and selects
no source-visible cyclic fuel or indirect/richer-borrow route. The complete
paired source and full5.1 requirements remain open after this prerequisite.

## My first owning snapshot source checkpoint

I pin `file_companion_snapshot.h/.c` as the first production subcheckpoint.
It does not yet install the paired graph collector, bridge or namespace resolver;
those remain the next required parts of this same reviewed dependency. The
standalone `file-companion-plan` Make target compiles seven actual providers with
caller CPPFLAGS/CFLAGS. No default compiler/provider list or public guard changes.
The opaque set API fixes counted requests/views and status/stage/request/errno,
secondary close error, heap peak/work and named automatic buffer report fields. It returns reports by value
and leaves the output set pointer unchanged on all failures.

| Owning allocation/reservation | Lifetime and accounting |
| --- | --- |
| Fixed set,16 zeroed rows and32KiB catalog view | Checked fixed size before calloc; charged once; owns all partial rows |
| Three request strings per row, each at most4095 bytes plus NUL | Validated and copied before any filesystem operation; charged including terminators |
| MAX_BYTES+1 input buffer per request | Reserved before any open; retained at full capacity, never shrunk or reallocated |
| One strict binding preparation | Full nonallocating bound reserved alongside all existing rows; replaced by exact retained owning size after successful return |
| Named automatic buffers, outside heap cap | Parent array4096 bytes; owning-TU query reports sizeof(CatalogText)+32 for formatter buffers. These are not total C stack bounds |
| Views | Borrow only completed immutable set storage; no allocation or source reopening |

I cap conservative project-requested heap reservations at64MiB. Automatic
objects, compiler stack frames and allocator/kernel/libc storage are excluded. My wrapper charges every counted validation,
copy, path scan and observed read, plus bounded fixed catalog/row bookkeeping.
For each strict preparation I reserve
`8 * MAX_TOKENS * MAX_LEXEME * (MAX_DEPTH + MAX_MEMBERS + MAX_ELEMENTS + 16)`
work units. This conservatively covers its two bounded decodes, duplicate-key and
typed-array lookups, complete catalog checks and repeated rendering scans. The
combined cap is2^41 work units; all16 maximum reservations fit. These are charged
upper bounds, not CPU instruction measurements or wall-clock guarantees. The
existing strict reader's individual byte/token/depth/member/array limits remain
unchanged. Source review and later allocation/failure fixtures must verify this
accounting; no measured peak claim exists yet.

I reject noncanonical lexical module paths and relative paths containing empty,
dot or dot-dot components; actual canonical origin provenance remains the paired
graph collector's responsibility. Final opening uses O_NOFOLLOW and O_NONBLOCK
before fstat, so a FIFO cannot block waiting for a writer. I allow64 EINTR retries
across the request's open/stat/read sequence; interruption65 terminates. I do not
retry close. First operation failure and first secondary close error remain
separate. A successful read checks its actual count against initial size and
final descriptor identity/size/coarse timestamps. Those checks detect selected
changes; they do not replace the documented stable-input/ancestor precondition.
A successful set retains original bytes even after the caller deletes the files.

## My ordinary namespace completeness inventory before graph code

| Existing producer input | Required resolver treatment |
| --- | --- |
| C AST_PROGRAM declarations and Nano Parser tables | Include ordinary functions/externs, records/resources, unions/enums, opaque/type declarations and module globals in their actual namespaces, before service names |
| AST imports, aliases and selective/wildcard exports | Resolve every edge under the original importing path; use original declaration identities before alias/mangling passes |
| C `load_module_internal` module.json `headers` and `parse_c_header_constants` | Header-generated constants contribute names; a parse-only graph cannot silently omit them. Snapshot and independently reproduce the relevant declaration facts, or return UNRESOLVED before claiming completeness |
| Explicit FFI declarations and module build metadata | Declared extern names are ordinary namespace members, not service methods. Missing metadata/unsupported generated declaration sources refuse complete preparation; no dlopen, compiler, pkg-config or module Make invocation supplies hidden authority |
| Imported package extraction and cached metadata/AST | Existing package/cache paths are not presumed immutable source-origin evidence. A complete owned source mapping must be established before use; unsupported package preparation refuses explicitly |
| Generic/list helpers, lambdas and compiler-generated declarations | Reserve and check generated identities in their established namespace before a complete plan is published; a later specialization that changes that namespace invalidates preparation and must rebuild/refuse, never reuse a partial report |
| Nano merge stripping module/opaque source lines and binding-table rewrites | Retain the original parsed declaration inventory before stripping/mangling. Absence from merged text is not evidence that no declaration exists |

I have not yet claimed complete namespaces for any graph. Both producer adapters
must resolve these rows or return UNRESOLVED; the old ordinary driver continues
its existing behavior. Supporting all required full source graphs remains parent
work rather than redefining an incomplete graph as complete.

## My explicit-relative import prerequisite

At `cec2788d0` my existing C `resolve_module_path` returns `./...` unchanged;
subsequent input opening and canonicalization therefore use process CWD. My Nano
`resolve_import_path` joins both `./...` and `../...` to the importing source
directory. I observed this in source, not an executed reproducer. I record it
as `task_e2ac2553c26a4640b00f6aa296f6da34` before changing either implementation.

I propose the same explicit-relative rule for both producers: when I have an
importing source path, I join `./...` and `../...` to its directory before
canonicalization; a basename-only importer uses its CWD directory. With no
importing source path I retain the supplied relative path. I do not change
absolute paths, bare-name/project/module search precedence, package policy or
ordinary source visibility. I make the small C resolver correction as a reviewed
shared prerequisite, not a special File-only resolver. I preserve existing Nano
semantics. Paired root and transitive imports must select the same actual file
with a conflicting CWD decoy, including symlinked importer canonicalization,
missing inputs and the unchanged search modes. Ordinary import tests remain
required. No graph report can claim paired origin parity before this gate.

### My first actual graph-retention source delta

I align the shared C resolver's explicit `./` and `../` branch with my actual
Nano collector: the declaring source is canonicalized before joining its
directory. A symlinked root or dependency therefore resolves beside its physical
source, not beside the symlink. A normal basename importer resolves in CWD; with
no importer I retain the supplied relative path. Failed canonicalization of a
supplied importer refuses rather than searching a decoy. The actual source must
exist and remain stable, as required by graph collection. I do not change the
absolute/bare/project/module search branches. I check join-size arithmetic and
free the canonical temporary on every return. This is a shared producer change,
not File spelling authority.

My Nano `CollectResult` now carries source strings aligned with completed DFS
order. Every return and recursive call preserves that alignment; duplicate
canonical inputs reuse their retained string. `MergeResult.original_sources`
retains these unstripped strings, and merging reads that array instead of reopening
source files. This retains original module/opaque declarations and visibility
for the upcoming real-parser namespace inventory. Existing traversal and
configured limits remain unchanged. This does not yet make the line-based
collector a complete AST import inventory, bind service origins, bound all Nano
allocator storage, or authorize source execution. Later introspection readers
and namespace preparation still require the retained-byte audit before the full
production checkpoint. The existing helper shadows now check aligned retention
and duplicate-path reuse after an input mutation; none have executed.

### My explicit visibility tuples and descriptive namespace keys

My resolution owns `(importing origin, qualifier, member, original target)` rows.
For unqualified rows the descriptive plan's module key remains the canonical
absolute origin. For qualified rows I use `N<origin-bytes>:<origin><qualifier-bytes>:<qualifier>`
with canonical unsigned decimal lengths and no leading zeros. The domains are
disjoint because supported absolute origins begin with `/`, never `N`. Lengths
count bytes, not Unicode characters. I validate the complete key against the
old plan's4096-byte counted module limit before allocation; qualifier/member
identifiers retain the existing128-byte bound. Both producers independently
construct the same key and charge every copied byte and comparison. Normalized
paired reports retain and compare the original tuple and target, not just this
internal encoding. The encoded key alone grants no identity or authority.

My actual Nano declaration nodes currently omit public modifiers, and the let
list includes locals. I therefore propose an optional capture wrapper around
the existing parse-program loop, retaining each real parse-definition token
interval, public modifier and before/after node-list extents. Legacy parsing
disables capture and keeps the current Parser/schema ABI. Opted-in graph
preparation owns the captured records alongside the original parsed module; it
does not infer visibility from stripped merged text or a second grammar.

The actual `--allow-temporary-files` preparatory branch performs this source
work before module builds, FFI loading or shadows. During this checkpoint it
creates no runtime grant and publishes no service output; existing lowering and
consumer refusal remains. A no-service graph returns to ordinary compilation,
with ordinary shadow and output behavior preserved. Default paths do not opt in.

### My inherited string-allocation failure boundary

My source audit finds that both existing generated `nl_str_concat` helpers return
an empty string on allocation failure; other Nano runtime allocations can stop
the process. I do not describe all Nano allocation failures as process-fatal or
claim recoverable-OOM equivalence with the C transactional APIs. Before releasing
a bridge token I compare each copied source/document/catalog byte count with its
immutable borrowed count. A mismatch refuses conservatively as UNRESOLVED. An
empty successful input remains distinguishable by its reported zero count. My
process publishes no service output at this checkpoint; later publication must
stage and clean its own artifacts across process failure.

## My complete preparatory source checkpoint

I connect the actual C `compile_file` and Nano `compile_program` drivers through
`--allow-temporary-files`. C calls `nl_file_source_resolve`; Nano selects
`collect_files_ast`, retains `ParsedDeclarations` with each first-read source,
then calls its independent `file_resolution_prepare`. The Nano collector follows
only the imports produced by its actual parser. It does not reconstruct imports
or public visibility with a second grammar. Both collectors canonicalize physical
source paths, deduplicate completed inputs, refuse active cycles, and retain
original bytes. My Nano legacy collector still scans lines and now retains
first-read text; recognizing the existing C `pub use` form is an explicit paired
language prerequisite, not a File-only interpretation. My new parser branch
reuses `parse_import` and marks its single actual import. Bare `use` stays invalid.
The parser capture wrapper leaves the existing Parser/generated-schema ABI intact.

My no-service result is `NONE`; the actual drivers then run ordinary compilation,
including its existing shadows and output rules. Preparatory source input limits
still apply to an explicitly opted-in invocation: unsupported packages, malformed
source, active cycles or exceeded configured limits fail rather than silently
bypassing preparation. Without the option I retain the ordinary selection path.
A service graph returns `PREPARED` or a refusal, and both CLI drivers return a
nonzero terminal without output publication, module preparation, dynamic loading,
FFI calls or shadow execution. `PREPARED` is descriptive success, never admission.
The separate byte APIs and their actual per-invocation runtime grants are unchanged.

I assign original IDs by canonical path byte order and actual declaration order.
Each service introduces all13 checked catalog names; ordinary declarations share
the collision domain before a plan can succeed. The exact returned top-level let
node identifies a global; a let inside its initializer does not become a global.
Shadow locals are not declarations. Public import rows propagate original target
identities through aliases. A qualifier is now a sequence of dot-separated
identifier components, each1..128 ASCII bytes; its complete encoded namespace
must fit4096 bytes. Importing an exported namespace preserves all of its qualified
descendants. Selectively renaming that namespace replaces only the first qualifier
component, preserving its original target tuple. I sort alias rows before final
IDs, then remap namespace-alias target IDs by their retained original origin/name.
Service aliases always point to original catalog IDs. Sorting C rows never leaves
duplicate owning qualifier pointers on a work-limit exit: I reserve each insertion
pass before moving rows. No encoded key grants authority on its own.

I refuse a complete namespace when metadata could contribute header/FFI-generated
names, package extraction is needed, anonymous/hoisted declarations have unknown
original identities, destructuring produces generated globals, or generic unions
require generated declarations. I also refuse Nano-only permissive public
modifiers outside my paired declaration/import forms. These are explicit
`UNRESOLVED` boundaries, not completion of full source acceptance. The required
metadata/packages/generated-name, complete language/shadow, nominal propagation,
independent lowering, cyclic/indirect/richer-borrow and runtime/publication gates
remain open under the original parents.

### My concrete storage and work accounting

| Allocation or retained data | Actual owner and bound |
| --- | --- |
| C graph and namespace tables | One zeroed `sizeof(NlFileResolution)` allocation contains5000 module slots,50000 import edges,528 visibility rows with4097-byte encoded-key arrays, sorted/postorder tables and all16x13 request/binding slots. Allocation failure preserves the output sentinel. |
| C retained canonical paths | At most5000 separately owned `strlen(path)+1` allocations, each at most4096 bytes; completed duplicate paths reuse the first module. `realpath`/shared resolver temporary allocation is inherited libc/resolver storage, not this retained-byte report. |
| C and Nano original input payload | At most64MiB in the sum of `(complete source bytes+1)` across unique modules. The common data-only reader checks file size before allocation, regularity, complete EOF, NUL and RFC3629 UTF-8; it closes once. Nano copies while its source token is live and verifies the exact count before destroying it. |
| C qualifier strings | One owned `strlen(qualifier)+1` per successful visibility row, at most528x4097 bytes, freed once after any failure or successful resolution destruction. Name/target spans borrow retained AST/path/static catalog storage. |
| Strict companion set | Its existing independent64MiB requested-heap cap counts all retained snapshots plus the full nested strict-parser/binding reservation. `peak_heap_bytes_reserved` remains a snapshot-only bound. It overlaps the retained graph and later plan; I never label that64MiB as a total compiler cap. |
| C descriptive plan | One transactional allocation from the existing plan builder, retaining its existing16 requests/64 aliases/256 ordinary and1MiB text limits. `plan_bytes` reports its exact owning size. It coexists with graph and companion storage until resolution destruction. |
| Nano retained graph and resolution values | Arrays retain original Parser/list handles and source strings; immutable row records retain original tuple/target strings. Snapshot copying reserves a separate64MiB logical sum of all four copied document/catalog spans before each copy. Encoded keys are checked before construction. Visibility logical text has a2300000-byte ceiling; the old plan still enforces its own unchanged1MiB budget. These are logical payload bounds, not a measured Nano allocator peak or recoverable-OOM guarantee. |
| Data bridge | One externally serialized positive token, never reused, owns either a source copy or a strict set. Its bounded200000-byte wire contains at most16 request records. The fixed `CompanionTransport` automatic array is reported separately, alongside named catalog/parent buffers. A busy/exhausted bridge returns0 and does not replace its owner. Failed preparation still requires token destruction. |
| Inherited parser/runtime storage | C lexer/token/AST allocations, Nano list/string/array allocator overhead and lifetimes, ordinary resolver/libc temporaries, stack frames and kernel storage are outside the snapshot and retained-graph caps. I make no total compiler heap or stack claim and no new recoverable parser OOM claim. |

On a successful C result, `input_bytes` is the fixed graph size plus retained
path/source/qualifier bytes; `source_bytes` is its raw-source subset. A conservative
upper bound on that report is `sizeof(NlFileResolution) + 64MiB + 5000*4096 +
528*4097`. The snapshot peak plus that retained graph and `plan_bytes` conservatively
covers the named owning heap overlaps, excluding the inherited domains above.
Failure reports preserve the actual first boundary and snapshot/reader report;
I do not reinterpret every inherited parser/resolver failure as MEMORY.
C free destroys the plan, strict set, qualifier allocations, original ASTs,
source and paths. Getters return borrowed immutable views until destruction.

The graph's own work budget is separately2^41 charged units. I bound modules,
edges, captured declarations and visibility before table growth. C charges input
passes, path/name comparisons and sorting before their destructive steps. Nano
reserves repeated path-table comparisons, raw-byte bookkeeping and actual sorting
comparisons, including the C-seed `char_at` helper's repeated length scans; row
validation/comparisons and alias remapping consume the same finite budget. The
bridge reports raw newline-count+1 as data, avoiding a second source-size-squared
Nano scan merely to enforce line limits. Both adapters preserve smaller configured
file/line limits and cap maxima at5000 files,50000 edges/lines and20000 lines per
file. The owning strict snapshot independently retains its existing2^41 work cap.
These are implementation-specific conservative counters, not equal CPU counts,
wall-clock guarantees or bounds on inherited parser, system path resolution or
ordinary compiler execution. Paired normalized reports compare original origins,
visibility/targets, catalog/plan facts and bytes, not these differing counters.

### My source review and later qualification boundary

This checkpoint changes no service AST/schema layout, nominal type propagation,
wire authority or runtime admission. The C driver links a distinct sorted
`FILE_SOURCE_COMPILER_OBJECTS` closure; I do not add overlapping NSI objects to
unsorted common VM/interpreter links. The Nano data module names its nine actual
C providers and canonical header. Full fresh C-seed/Stage1/Stage2 bootstrap and
all selected module/helper shadows remain required after the production and
fixture reviews. Static diff checks do not stand in for those gates.

My upcoming fixtures must exercise actual drivers, real publisher bytes and
strict companion documents; ordinary no-service opt-in/default behavior;
first-read mutation and original parser retention; symlink/CWD import selection;
public/private/selective/wildcard/nested re-export visibility; namespace and
catalog collisions; metadata/package/generated-namespace refusals; UTF-8/NUL and
configured/counted limits; C allocation prefixes and every owning cleanup root;
bridge busy/stale/destroy/copy lifetime; exact paired tuple/target/plan reports;
no module/FFI/shadow/output attempts for preparatory service graphs; and unchanged
public byte-consumer refusal/grant controls. I have not executed these fixtures.

My final static provider audit identifies an additional prerequisite before this
source checkpoint is a qualification candidate: repeated canonical C providers
and shared basename object paths in the actual Nano native module collector.
I record task_28cf8f795b2a410d8bd015d2a4545018 and the concrete
[native provider closure correction](NANOISA_FILE_NATIVE_PROVIDER_CLOSURE.md).
The graph/namespace/bridge source is reviewable, but I hold provider implementation
for that plan review and all builds/fixtures for complete source review. I do not
claim the currently repeated manifests already link successfully.

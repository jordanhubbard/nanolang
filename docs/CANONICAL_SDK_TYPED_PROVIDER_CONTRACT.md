# I preserve exact SDK types through canonical products

I propose this contract at candidate `35c523a81`. It is a source design, not
execution admission. My default `.nvm`, explicit native and `--target c` routes
continue through verified NanoISA. My internal source-native helper preserves
regression evidence only. Every original SDK, File, provider, callback and
installed acceptance requirement remains part of 5.1.

## I reuse actual interfaces with their current boundaries

| Existing interface | I can reuse | I must not infer |
| --- | --- | --- |
| `nvm_format_v2.h`, `nvm_v2_sections.h` | checked section framing, feature/section agreement, typed signature and layout indices, constants, round-trip ownership | coarse tags or layout names do not establish declaration identity |
| `NvmImportEntry`, `nvm_add_import` | stable import ordinal, kind, module/artifact identity, coarse parameter/result tags | TAG_OPAQUE does not identify the declared opaque type; TAG_ARRAY does not describe children |
| `NvmCallDescriptor`, `vm_ffi_load_import`, `vm_ffi_call_vm` | one resolved dispatch cache per module/import and existing selected provider loader | a cached symbol is not a validated complete signature or lifetime policy |
| `NvmCallbackContract`, VM retained callback scheduler | exact supported callback shape, execution thread policy, registration/release adapters and owning VM lifetime | arbitrary closure escape, isolated-process callback transport or an unspecified native callback ABI |
| retained layouts and `ownership_contracts` | existing ordinary/resource authority and complete supported aggregate/ownership proofs | layout codec success does not admit executable operations |
| `service_bindings_module`, File nominal plans | checked immutable cross-reference plans and unchanged-output failure conventions | the fixed File catalog is not a general SDK extern registry; all existing execution holds remain |
| `nvm2c` artifact adapters | exact supported scalar/string/array adapters with their existing snapshot/release rules | an arbitrary symbol or matching short spelling is not an admitted adapter |
| `NativeSdk.runtime_root`, native provider collector, module builder | verified SDK generation, actual metadata include closure, canonical source deduplication, ordered flags and private objects | manifest self-consistency is not a trust signature; user flags remain trusted input, not sandboxed execution |

My current canonical emitter explicitly rejects tuple children outside its
bounded scalar/nested-tuple set and rejects externs outside metadata, artifact or
its fixed host ABI. Adding provider objects to a link command cannot repair those
missing type and execution proofs.

## I carry an exact typed descriptor beside existing imports

I propose one separately versioned **typed provider contract** section and its
own feature bit, assigned centrally with the v2 format owner. I do not change
v1 interpretation, repurpose File catalog ordinals, put authority in advisory
metadata, or silently reinterpret existing import bytes. Unknown readers refuse
its feature. The exact numeric assignment and codec budgets require a source
checkpoint before implementation.

The payload has three ordered tables:

1. Complete type nodes: scalar kind, array child, tuple children, callable
   parameter/result children, and nominal declaration references. A nominal key
   retains canonical declaring origin, declaration kind and original name; a
   generic instance retains its complete ordered argument keys. Aliases point to
   the same actual declaration. Distinct origins remain distinct even with the
   same public name. C symbol spelling is a separate field of the provider ABI,
   never semantic identity. Nodes also reference existing retained layouts where
   those layouts express the full storage shape.
2. Provider requirements: logical declaring module, exact supported native ABI
   revision, target data model/ABI requirements, provider identity and the
   selected contract revision. SDK generation references use installed relative
   roles rather than embedding a source checkout path. User provider resolution
   retains its actual declaring metadata origin and canonical artifact identity.
   Provider identity and manifest byte identity are not authorization signatures.
3. Import contracts: exact import ordinal, provider ordinal, symbol spelling,
   complete parameter/result node indices, and explicit argument/result lifetime
   policy. Every coarse import tag must match its full node. Existing callback
   contract indices are cross-checked for the same import and parameter; there
   is one authoritative policy, not two competing callback declarations.

I intern by full structural key, with unambiguous framing and original nominal
identity. A hash may select a bucket but equality compares the complete key.
Publication is transactional: failed copies, validation, interning or graph
construction publish no row or changed module. Decoder views remain borrowed
only for their documented buffer lifetime; execution plans independently own
all retained facts. Declaration-reference cycles are distinct from recursive
by-value storage. I reuse the derived declaration graph's declaration-versus-
complete-layout distinction; a callback referring to its enclosing record does
not become a recursive by-value layout.

I keep existing format/count limits and checked size arithmetic. Before coding,
I inventory each codec's count widths, module size budgets and accepted compiler
fixture maxima, then publish inclusive limits for new nodes/edges/depth/imports.
I must not choose a smaller unexplained limit that excludes an already accepted
SDK program. All traversals are bounded and fail before provider effects.

## I admit operations through one checked execution plan

Both producers derive nodes from resolved declaration authority and retained
complete annotations. Source namespace visibility is checked before descriptor
construction. Lexical callable and real declaration precedence stay ahead of
intrinsic handling. No emitter recovers owners from a unique short name or
strips qualifiers to manufacture identity.

A shared plan validates the complete payload, all import/layout/callback
cross-references and the instruction uses that consume these types. It projects
one immutable plan for VM and nvm2c. Deserialization and transport validation
alone confer no dispatch authority. Duplicate/conflicting contracts, unresolved
provider ABI, unsupported storage/lifetime shapes and stale ABI1 requirements
refuse before loading a provider or entering native code. Both executing routes
must consume the same validated facts. Re-serialization preserves them, and
readers/writers/pretty-printers/linkers cannot silently drop the feature.

The smallest first implementation unit is this codec plus transactional attach,
plan and refusal integration across all consumers. The next units add the actual
SDK types and operations already required by the unchanged acceptance corpus;
the first unit is not a release-complete subset. Existing scalar artifact
adapters remain valid without the new feature and do not acquire wider authority.

## I preserve representation and lifetime at the ABI boundary

Canonical runtime values are not native C records or DynArray elements. I use
explicit typed adapters; I do not reinterpret NanoValue storage as an arbitrary
C struct or function pointer. A native adapter uses the actual generated C type,
`sizeof` and alignment, including the actual callback typedef. Complete tuple,
record, union and nested array children follow the descriptor plan.

For native DynArray ABI2, tuple/callback elements retain exact-width ELEM_STRUCT
copies, opaque values retain their existing borrowed identity, and nested arrays
retain ELEM_ARRAY. Loads snapshot a value before any callback can relocate its
source; absent elements refuse before copying. Stores evaluate/stage source and
index once. Existing empty-width-zero/pop and terminal allocation semantics stay
unchanged. Canonical array storage preserves aliasing and mutation without
assuming native DynArray layout; conversion wrappers must state which side owns
each buffer and release every staged child on failure.

Borrowed opaque handles remain non-owning unless an existing declared provider
contract supplies the owning operation. The descriptor establishes type identity,
not a new general borrow lifetime. Provider/cache-generation leases and VM or
Environment leases remain live through calls, copied returned views and retained
callbacks. I preserve scheduler active-callback latches, owned result clone/drop,
owner-thread rules and teardown preflight before cache/AST destruction. A
process-local pointer is never serialized as a portable opaque value or passed
through an isolated-process route that lacks an admitted handle protocol.

Where a provider returns transient strings or arrays, the existing named
snapshot/release contract remains exact. General native aggregate returns and
foreign callbacks need a concrete adapter declaration, not libffi guesses from
coarse tags. Different hosts may use different C layouts while preserving the
same semantic key; target ABI mismatch is refused before entry.

## I resolve providers and install complete canonical products

Canonical preparation consumes the existing ordered provider registry and actual
metadata dependency closure after type/plan validation. Identical canonical
sources compile once; same-basename different sources remain distinct. ABI,
compiler/language/profile conflicts refuse before final publication. Existing
trusted flags, checked quoting, private indexed objects, first-failure retention
and cleanup remain. I preserve the static public owner/private dynamic dependency
rules for the full 36-provider corpus and both import orders.

The public canonical driver currently selects `bin/nano_vm`, `bin/nvm2c` and
`bin/nano_aot_runtime.o`. The installed SDK generator already enumerates VM and
translator executable roles, but it does not enumerate the AOT runtime object.
I must add the actual AOT closure as a distinct object role, its clean-install
Make prerequisite and exact manifest validation. I must also replace the current
blanket executable-mode test for every `bin/` row with explicit role checks so an
object is not falsely required to be executable. I will regenerate all inventory
outputs only after the final source/role set is complete. Invalid explicit SDK or
tool selection must fail before provider preparation; installed generations stay
read-only and invocation work remains privately writable.

A produced `.nvm` must retain a usable provider requirement rather than a
short-lived compiler object path. Its runtime resolver acquires the verified
installed generation or exact user artifact with the appropriate lifetime; AOT
translation uses the same plan and stable provider closure. The actual relocation
and resolver contract must be reviewed with the owning loader before execution.
No source-checkout assistance may substitute for installed-only acceptance.

## I qualify the complete boundary

I retain the original source-native fixtures under their explicit internal route
and independently run their required canonical behaviors with all three producer
roles. I require both hosts, fresh bootstrap, clean install, moved source, spaced
prefix, read-only generation, concurrency and owned uninstall. Original Json.Json,
504-byte records, complete tuple/callback arrays, global-only empty/nonempty
initializers, source-order staging, same-name lexical declarations and all provider
ownership/ABI controls remain unchanged.

Additive controls cover exact type/alias round-trip, same-basename distinct
owners, graph/layout/callback cross-reference disagreement, stale ABI before entry,
unknown feature refusal, failure rollback and every published limit. Independent
VM and nvm2c caller/callee combinations must preserve tags, widths and complete
identity. Callback/array mutation and borrowed-result lifetime controls must use
the existing approved lifetime policy. Original File service execution/lowering
holds are closed only by their own complete reviewed runtime/source acceptance,
never by SDK descriptor transport success. I preserve first terminals and output
sentinels, stop dependent gates on failure and claim only the actual measured
producer/backend/instrumentation scope.

## I refine the wire proposal around retained indices

This amendment follows the actual `ownership_array_fields.inc` and
`ownership_declaration_projection` inventory. I do not add an independent
recursive type graph to the provider section. The retained ARRAY_FIELDS type
pool already contains scalar, string, array-child and record-layout references;
its declaration projection owns copied rows and bindings. Existing limits are
4096 type rows,65536 field bindings,64 array levels,1048576 charged steps and
16MiB owned declaration storage. The private layout profile separately limits
256 layouts and65536 fields. Its current grammar does not admit opaque, tuple,
union or callable rows; that is an actual extension requirement, not permission
to treat a rejected row as unknown.

I propose a new shared ARRAY_FIELDS extension revision, selected only with the
new typed-provider feature. Existing revision1 bytes and acceptance remain
unchanged. Shared rows retain their current8-byte tag/reserved/referent encoding:
scalar/string referents stay NO_INDEX, arrays reference another retained type
row, records reference existing LAYOUTS. Added tuple/union rows reference the
same retained layout table. Opaque rows reference exact nominal declarations;
function rows reference exact signature-detail rows. These last two referents
are checked against the provider contract after both sections have been decoded.
The common declaration projection exposes the resulting complete rows; SDK
consumers must not parse a competing private type table.

The provider section then contains only facts missing from current sections:
nominal identity records, provider requirements, exact signature details and
bindings to actual imports/functions/fields. Every layout uses its existing
index. Every exact signature detail records its existing coarse SIGNATURES index
plus retained type-row indices for parameters/results. Coarse signatures remain
deduplicated by tags; different opaque owners may share that coarse index while
requiring distinct exact detail rows. I compare exact detail identity for typed
indirect calls, not coarse signature equality alone. Layout-field bindings add
exact child facts only where the existing layout tag/referent is insufficient;
they must agree with all existing layout/ownership fields. A nominal record row
maps its original owner/name/kind and ordered generic arguments to its retained
layout index, rather than creating a second layout. Opaque rows have NO_LAYOUT.

I propose feature bit11 (`0x00000800`) and section type`0x10`, the next unused
values in the audited candidate. The known-feature mask becomes`0x00000fff` and
the section-type maximum becomes`0x10`; the existing16-section in-memory bound
is sufficient for all16 defined section kinds. These assignments require owner
review and a fresh collision check against the final candidate before coding.
The provider codec revision is1. The shared ownership extension uses an explicit
new revision/version gate; no existing ownership-v3 ARRAY_FIELDS revision1 row
is reinterpreted. Unknown sections/features/revisions refuse at the normal
container/ownership boundary.

Proposed inclusive new-section limits are4096 nominal rows,4096 providers,
4096 exact signature-detail rows,65536 bindings and65536 total parameter/result/
generic-argument references. Counts are u32; per-signature arity stays within the
existing u16 field. The existing retained callback ABI's16-argument limit remains
specific to that ABI, not an invented general extern cap. The new section is at
most16MiB; complete preparation charges its bytes plus the owned shared
projection against a32MiB combined bound, and all new graph/cross-reference work
against1048576 steps. Existing shared declaration preparation keeps its own
16MiB and step bounds. Allocation products are checked with subtraction-form
remaining budgets before allocation; zero counts allocate nothing. These are
new descriptor-profile limits, not blanket limits for modules without the
feature. Before admission I must verify all original SDK fixture maxima fit and
publish any necessary reviewed extension; a codec-limit refusal cannot replace
required SDK acceptance.

Array/storage edge depth retains64. Complete by-value layout cycles refuse;
callback-to-enclosing-record declaration edges use a visited worklist and may
cycle without consuming artificial infinite depth. Every edge examination,
identity comparison and interning collision comparison is charged; deduplication
cannot hide unbounded quadratic work. The first revision may conservatively
reach its published work limit, but must not claim linear complexity without
measurement. Wire indices are validated before dereference and exact table
extents/reserved zeros are checked before publishing an owned projection.

## I attach atomically and refuse incomplete consumers first

I propose separate raw decode/encode, module validation, owned plan preparation
and attach entry points. Raw codec success means transport only. Each API has
explicit INVALID/LIMIT/MEMORY outcomes; legacy TRUNCATED-to-memory ambiguity is
not copied into the new API. Failure preserves caller outputs. Attach prepares
all section/type/signature/string changes and validation in temporary owned
storage, then swaps the complete module state once; failure frees only staged
storage. Identical valid attachment is a no-op, conflicting attachment refuses.
Call-descriptor caches are invalidated only after successful publication, with
no active generation/dispatch leases. I do not mutate a module while executing
or invalidate borrowed string/type pointers under a live call.

The first source checkpoint must enumerate every reader/writer and executing or
dropping consumer from actual callers of v2 loading, bridging, serialization and
module validation. Deserialization, assembly/disassembly, pretty-printing,
linking and v1 bridging either preserve the complete feature or refuse it;
lossy fallback is forbidden. VM, nvm2c, nvm2hl, LLVM/Wasm backends, daemon/cop and
packaging tools refuse unsupported typed-provider execution before provider
loading or output publication until their exact adapter is implemented. Presence
includes malformed partial claims, as in existing File/capture guards. This
initial refusal checkpoint is a dependency, not completion: the full required
VM/nvm2c and installed SDK execution adapters must follow before final release.

Static and corrected-only controls cover feature/payload agreement, exact row
bounds, full referenced type/signature/layout/nominal identity, empty/boundary
payloads, same-tag distinct owners, declaration cycles versus storage cycles,
unchanged-output allocation failure, cache/lease-safe publication and exact
round-trip preservation. Actual provider entry markers, stale ABI and source-
hidden installed executions remain later reviewed acceptance. No arbitrary
module-generation or foreign-call execution follows from transport tests.

## I fix the first raw codec row bytes

My private `sdk_provider_codec` implementation is a transport prerequisite. It
is not linked into product consumers and does not yet activate bit11 or section
0x10. Current product readers therefore continue refusing those unknown claims.
I require the shared projection revision, module cross-validation, atomic attach
and complete consumer guards before admitting any new module. Full ABI/provider
execution still follows that reviewed transport checkpoint.

All integers below are little-endian u32. Tables occur in this exact order;
there are no padding bytes or trailing data. My32-byte header contains revision1
at0, reserved-zero flags at4, nominal/provider/detail/binding/reference counts at
8/12/16/20/24, and reserved zero at28. Count products are bounded before loops or
allocation. Row widths and fields are:

| Table | Bytes | Fields by byte offset |
| --- | --- | --- |
| Nominal |32| owner-string0, original-name-string4, kind8, layout12, generic-reference first16/count20, zeros24/28 |
| Provider |32| module-string0, ABI-string4, target-string8, artifact-digest-string12, generation-digest-string16, library-string20, zeros24/28 |
| Exact signature |24| coarse-signature0, parameter-reference first4/count8, result-reference first12/count16, zero20 |
| Binding |24| kind0, actual subject4, slot8, exact detail12, provider16, zero20 |
| Reference |4| existing shared ARRAY_FIELDS type index0 |

Nominal kinds0/1/2/3 mean record/union/opaque/enum. Only opaque carries NO_LAYOUT.
Enums retain existing TAG_ENUM and LAYOUT_ENUM identity; plain INT cannot replace
them. The shared revision must bind enum rows to their existing enum layout and
its exact owner/name nominal declaration.
Binding kinds0/1/2 mean import/function/layout-field. Import/function slots are
NO_INDEX and details select exact signature rows. Imports require a provider
row; functions require NO_PROVIDER. Layout-field slots are u16-range field
ordinals, details select shared type rows and provider is NO_PROVIDER. I check
these internal slices and reserved bytes in the raw codec. Actual string
validity, digest syntax, target ABI, layout kind/field/coarse-signature agreement,
complete owner identity, duplicate claims and referenced shared types remain
module-plan validation obligations; raw success must never stand in for them.

Decode copies the complete immutable payload into one owned plan after shape
validation; accessors return values. Encode accepts readable borrowed row arrays,
allocates one staged zeroed payload, writes canonical reserved bytes and validates
its result before publishing. Both preserve outputs on INVALID/LIMIT/MEMORY.
The decoder's supplied limit includes its plan header plus payload; encoder's
limit covers its single payload. No recursive graph work occurs in this slice.
Neither API touches module strings, call-descriptor caches or generation leases.
The earlier graph/combined-plan limits still apply to the forthcoming module plan.

## I inventory consumer integration before admission

I traced actual `nvm_serialize`/`nvm_deserialize`, v2 conversion and
`nanoisa_load_file`/save callers. The next integration checkpoint must preserve or
explicitly refuse this complete feature at each owning boundary:

- `nvm_format.c`: module allocation/free, legacy bridge selection, descriptor
  cache lifetime and serialization; `nvm_v2_module.c` and `nvm_v2_convert.c`:
  section planning, required-feature agreement, complete owned copy/free and
  signature/layout remapping. `modules/nanoisa/nanoisa.c` owns public loading,
  saving and assembly bridge publication.
- `ownership_array_fields.inc`, `ownership_contracts.c`,
  `ownership_declaration_projection`: explicit new revision grammar, shared
  type rows, complete graph checks and owned projection. Existing revision1
  acceptance is unchanged. Verifier and `verifier_types.c` must refuse a new
  required SDK claim before treating coarse tags as sufficient proof.
- Assembly/disassembly and introspection: `disassembler.c`,
  `modules/nanoisa/dump_main.c`, `modules/forth_see/forth_see.c`,
  `hl_facts_main.c`, and generated Nano metadata/assembly consumers. No textual
  round-trip may silently erase descriptors.
- Executing/provider consumers: `nanovm/main.c`, `vm.c`, every `vm_ffi.c`
  resolution/dispatch entry, `cop_main.c`, `vmd_server.c`,
  `nanovirt/main.c`/`wrapper_gen.c`, `nvm2c.c`/`nvm2c_main.c`,
  `nvm2llvm.c`/`nvm2llvm_main.c`, and canonical nvm2hl/Wasm routing. Direct
  in-memory modules require the same early refusal as binary input.
- File hosted/cyclic/indirect adapters and portable/managed/ordinary ownership
  plans must not treat SDK declarations as File authority or ordinary supported
  layouts. Packaging/linking paths must retain the feature or refuse before
  writing output. Installed AOT/provider execution stays blocked until exact
  ABI adapters and source-hidden acceptance qualify.

This inventory is a source-work queue, not a claim that these adapters are
implemented or qualified. I keep the feature unavailable while closing each
boundary and review the complete change before product execution.

My standalone raw-codec fixture includes same-name distinct-owner rows, an enum declaration, complete by-value round-trip, input mutation/release after decode, every truncated length, reserved-byte and count/slice refusals, exact allocation-byte boundary, both allocating API failures with unchanged outputs and independent recovery, and empty payload transport. It includes the real codec with allocation hooks; no module/provider execution occurs. Strict syntax-only compilation passes; execution awaits independent fixture review.

## I prepare a new immutable module generation

My actual NvmModule has no complete active-VM lease counter. An empty call cache
cannot establish exclusive ownership. I therefore replace the proposed in-place
attachment with construction of a separate fully owned immutable generation.
The original module, runtime call cache and active borrowers remain untouched.
All new module bytes, tables and descriptor plans are staged and validated before
publication; failure leaves the caller output and old generation unchanged.
There are no interior pointer aliases into the old generation. Publication may
replace only an exclusively owned unpublished pointer, never a live module.
An identical existing contract may return an explicit unchanged result without
transferring an aliased owner; conflicting identity refuses. Combined bounds
charge actual clone and plan bytes. Existing accepted maxima must be audited
before imposing any smaller clone profile.

The legacy bridge rebuilds coarse signatures from function/import/callback tags
and discards the original table indices. Typed generations must retain the
canonical SIGNATURES section and each subject's selected signature index,
validate all subject shapes against those rows, and preserve unused but
referenced callable signature rows. Exact SDK details still distinguish same-tag
opaque owners. Ordinary modules keep the existing interning path. The exact
owned representation and complete conversion/free/serialization consumers need
source review before execution; I do not copy stale indices into a rebuilt table.

My next private helper, `sdk_signature_snapshot`, copies all original coarse
signatures (including unused/duplicate rows), their complete tag arrays, and
function/import/callback/link indices into independently owned storage. It
publishes only after every allocation succeeds, reports exact owned bytes,
charges both validation and copy work, and exposes indices without reinterning.
Only callback/link indices may be NO_INDEX at this transport layer; their actual
semantic use remains module validation. No old module pointers survive. The
32MiB/1048576 bounds apply to this typed-profile helper and caller remaining
budgets; ordinary conversion is unchanged. It is still unlinked, and does not
substitute for complete generation copy or module/ABI validation.

My signature snapshot controls retain duplicate and unused rows with exact subject indices, then overwrite/free all borrowed inputs before querying the owned result. Both one-shot and persistent failures cover all seven measured allocating sites, each followed by independent recovery; exact byte-budget and invalid-selector/tag/count refusals preserve outputs without allocation. Empty snapshots remain valid transport. These controls remain source-only pending review; no full module or ABI authority follows.

## I copy complete module generations without hidden legacy conversion

My next bounded implementation retains the complete NvmV2Module representation,
not a lossy NvmModule bridge. The existing direct V2 serializer preserves table
indices, but its sizing path calls nvm_v2_to_nvm_module whenever ownership,
passive, capture or service data is present. That path performs additional
allocation and profile validation. I therefore cannot use an unbudgeted
serialize/deserialize roundtrip as my generation-copy primitive or silently
exclude those sections.

I propose a private owned module snapshot with these explicit copies:

- I retain every signature row/tag and exact selector through the already
  qualified signature snapshot, including duplicate and unused rows.
- I copy metadata, constants, functions, globals, imports, callbacks, links and
  debug row arrays with their original counts and ordinal order.
- I copy each layout header and every field row, retaining name/nested indices,
  kind and field count exactly; no projection or spelling-derived substitution.
- I copy constant payload bytes by length, including physical zero bytes, code
  by its uint64 size, and capture/ownership/passive/service bytes by their actual
  sizes. Shared source payloads may have separate owned copies; no old pointer
  is retained as generation storage.
- I preserve isa_version, entry_point, has_debug and extra_features. owned_tags
  is an allocation convention, not semantic data: the signature snapshot owns
  the new tag bytes independently of the source's owned_tags allocation.

Existing public V2 decoders bound uint32 table counts against encoded remaining
bytes; ordinary layout fields have uint16 counts. They do not impose my raw SDK
4096-row limit on all module tables. I preserve that distinction. The existing
32MiB generation allocation budget and 1048576 work budget are explicit private
SDK preparation limits, not new global module validity rules. I return LIMIT
before allocation if this preparation cannot represent an otherwise valid large
module; no ordinary loader or execution admission is changed by this primitive.

Before allocation I validate nonzero-count pointer presence, multiplication,
addition and uint64-to-size_t conversions and charge every owned table, layout
field, payload byte and snapshot object, including the signature snapshot's
storage. Work counts every visited table/field/tag/payload unit according to a
single monotonic bound; I do not first scan an unbounded table to calculate it.
The source and reachable storage must remain readable and unchanged throughout
preparation. I do not promise safety for arbitrary invalid C pointers.

I publish only a complete private snapshot. Failure leaves the source and caller
output unchanged and frees every partial allocation. Its read-only view follows
the same contractual read-only nested-pointer rule as the signature transport;
no deep-const enforcement or execution authority is claimed. Destruction uses
explicit ownership of each copied allocation and never invokes old module/cache
teardown. Later descriptor attachment/crossvalidation must charge this snapshot
and its additional plans together before publishing an exclusively owned new
generation. All current sections remain required in that future validation.

Controls must retain all table counts and unused rows, binary constant/code and
auxiliary bytes, empty-but-present DEBUG, selector sentinels, source mutation
independence, old-generation continued readability, negative structural inputs,
exact budgets and every allocation prefix with recovery. This private primitive
still grants no descriptor, VM, nvm2c or installed SDK admission.

## I checkpoint the complete private snapshot implementation

I reuse signature validation through one allocation-free measure function and
one shared internal plan. Its work result charges validation plus copying; the
module reserves three passes (external measurement, preparation validation,
copying) before entering that signature scan. The signature measure accepts the
remaining work bound explicitly, so preflight cannot consume a second hidden
million-step budget after other tables have already consumed it. The existing
standalone signature preparation retains its original allocation sites and
budget. Failure leaves measure outputs unchanged too.

The module snapshot owns separate flat tables, layout-field allocations, the
signature snapshot, and one complete payload pool. One nonempty payload copy is
a work operation; actual copied bytes are charged to the byte bound. Row and
field visits are charged before traversal. Every allocation including both plan
objects is counted; no caller module pointer or cache enters destruction.

The private fixture carries every table and all five code/auxiliary byte ranges,
physical zero bytes, duplicate and unused signatures, callback sentinel selectors,
and empty-present DEBUG. It checks exact byte budget, malformed dimensions,
work refusal, every discovered allocation prefix in persistent/one-shot modes,
and recovery. A second generation stays readable after source mutation and first
generation destruction. This is deliberately transport data; the fixture does
not assert cross-section semantic validity or admission. Original signature
controls remain unchanged and must rerun with the shared planning refactor.

## I make the shared declaration reader consume retained V2 facts directly

My complete snapshot now qualifies as independent storage. It still cannot feed
nvm_prepare_ownership_declarations: that interface accepts NvmModule, and its
shared reader consumes legacy function tags, union counts and string indices.
Calling the legacy bridge would discard exact signature selectors. I will add a
private V2 entry into the same reader rather than construct a second SDK grammar.

I first factor the reader's borrowed module facts behind a bounded internal view:
function count, locals, exact selected signature parameter/result tags, layout
kind counts, named string constant validity, and ownership bytes. The legacy
adapter reads its existing fields without reinterning or allocation. The V2
adapter reads each function's actual signature_idx, bounds it before access,
and checks complete parameter/result extents; no inferred signature index or
first matching coarse row is substituted. Its layouts remain their original
ordinal order. Layout names/field names/variant names must reference actual
STRING constants, not merely a number below constants.count. Both adapters
reuse the same descriptor, extension framing, union-fact and array-row readers.
No public consumer obtains wider admission from this refactor.

I retain a separate profile parameter selected only by the private typed SDK
preparer. Legacy/public callers continue revision1 and existing UNKNOWN/refusal
policy, including unsupported imports/capture/passive/service conjunctions.
The private declaration reader may copy facts from a module carrying those
sections without claiming their validation: full module preparation must check
every section before dispatch. Presence of an unrelated section cannot be
silently dropped, nor may a declaration query stand in for its checker.

I then add revision2 ARRAY_FIELDS to that private profile using the reviewed
8-byte shared row encoding, and carry its revision through extension framing.
Opaque referents select exact provider nominal rows; function referents select
exact provider signature details; tuple/union/enum referents select actual layout
kinds. Legacy revision1 bytes keep their original meaning. The shared projection
copies these numeric facts but reports unresolved foreign referents until the
provider cross-validator proves them. No executable plan is published with
unresolved identity, layout, signature or lifetime facts.

Before exposing the new entry, I require unchanged legacy declaration/ownership
fixtures, direct V2 duplicate/unused signature selectors, mixed constant tags,
all layout kinds, array cycles/depth, malformed references, shared byte/work
budgets, every allocation prefix, and old/new generation lifetime independence.
New output publication remains atomic. The final consumer audit must cover
VM, nvm2c, V2 serializer/deserializer, assembly/disassembly, linker and legacy
bridge before the feature can be admitted; this reader alone closes none of
those gates or the original installed SDK acceptance corpus.

My first direct-reader checkpoint adds nvm_prepare_ownership_declarations_v2
without enabling ARRAY_FIELDS revision2. A borrowed OwnershipModuleFacts view
feeds the same ownership descriptor and union readers; common declaration checks
and final allocation/publication are shared with legacy preparation. The legacy
adapter retains its absent-parameter TAG_VOID behavior and original auxiliary
section refusal. The V2 adapter uses the selected signature row and STRING
constant kind directly. Other V2 sections remain stored by the independent
module snapshot but are outside this declaration-only query.

To reuse the exact reviewed structural layout grammar, I encode only the bounded
layout table and invoke the existing private mixed-layout decoder. This is not a
module conversion and touches no signature/constant/auxiliary table. I preflight
all layout/field extents, conservatively charge visits before selector traversal,
and account for wire plus decoder copy/DAG workspace peak; I free the temporary
wire before array and final union-plan allocations. Existing16MiB/1M private
limits remain. Combined caller-budget projection and revision2 cross-validation
are still required before provider attachment.

Additive controls reuse the unchanged mixed/ordinary fixture bodies and append
V2 selected/duplicate/unused signature inputs, wrong selector/tag/name-kind
refusals, revision2 refusal, dimension limits, source independence and every new
allocation position under persistent/one-shot failure with recovery. Strict
syntax checks pass for production and both fixture compilation modes; runtime
qualification awaits this source checkpoint review.

## I close the remaining shared-profile and lifetime gaps before admission

My qualified direct V2 reader remains revision1-only. The next source unit adds
an explicitly selected private typed profile to the existing layout/descriptor/
union/ARRAY_FIELDS readers. The legacy and public wrappers select their existing
profile. Revision2 rows keep the shared8-byte encoding: STRUCT/TUPLE/UNION/ENUM
referents must name the matching actual layout kind; ARRAY names a shared type;
OPAQUE names a provider nominal declaration; FUNCTION names an exact provider
signature detail. Scalar/string referents remain NO_INDEX. Foreign referents
stay unresolved until the provider cross-validator checks the complete table.
A copied declaration plan with unresolved facts cannot become a call plan.

The layout reader must validate all tables, not only tables with forward edges.
By-value STRUCT/TUPLE/UNION field edges form one bounded acyclic storage graph;
ARRAY/FUNCTION/OPAQUE fields use NO_INDEX in the coarse layout and obtain exact
children from shared/provider field bindings. They are declaration/reference
edges, not inline storage cycles. Array child chains retain64 levels. Complete
nominal identity requires one consistent declaration key for each nominal layout,
including enums; duplicate aliases may repeat the same full key but cannot give
one layout conflicting owners or one exact key conflicting storage.

I must replace unconditional private limits with a caller-accounted preparation
context before composing module copy, declarations and provider validation. The
context carries remaining allocation bytes and work steps; each preparation
works on a local copy and publishes both output and consumed budget only on
success. Failure preserves the original budget and output. I charge temporary
layout encoding, decoder DAG workspace, copied signatures/tables, provider raw
bytes, graph worklists and final plans, including repeated passes. Freed scratch
still counts toward this preparation's allocation budget, giving a conservative
monotonic bound. Existing unbounded-by-caller APIs wrap the same implementation
with their current16MiB or32MiB/1M limits, preserving existing fixture behavior.
No allocator retry receives a fresh hidden budget. A completed owned generation
must fit32MiB and1,048,576 steps across the complete preparation, not per helper.

The current provider revision1 codec has no lifetime table. Existing imports
carry only module/symbol/signature/kind, while CALLBACKS describes callback ABI
and execution. Neither establishes argument/result aggregate ownership. I keep
revision1 transport-only and propose an explicit provider revision2 format,
with separately reviewed row bytes before implementation:

- I retain the five original table encodings and add call-policy and policy-slot
  tables. The revision2 header is40 bytes: revision at0, call-policy count at4,
  original five counts at8..24, policy-slot count at28, reserved zeros at32/36.
  Revision1's32-byte header and reserved-zero rules remain unchanged.
- Each24-byte call-policy row contains parameter-first/count, result-first/count,
  callback-contract profile and reserved zero (all u32). Each16-byte slot stores
  mode, owning-parameter index or NO_INDEX, release-symbol constant or NO_INDEX,
  and reserved flags. Import binding offset20 names its policy row in revision2;
  function/field bindings retain zero there. No revision1 reserved word changes.
- Call-policy rows are bounded by65,536 and total policy slots by65,536; the
  complete provider section remains16MiB. Parameter/result counts must exactly
  match the selected exact and coarse signatures, not merely fit within them.
  Every string index must select a STRING constant and release metadata must
  resolve through the same provider owner before a plan is admitted.

The mode contract must distinguish call-duration borrowing, copied results,
borrowed opaque identity and explicitly retained callbacks. A structural policy
row is not permission to execute an arbitrary C ABI. The actual generated typed
adapter must advertise an exact schema agreeing with complete types, provider
ABI, slot policies and existing callback contracts. Aggregate conversion copies
only the children permitted by that schema; nested opaque/callable leaves cannot
acquire ownership by inheritance from a container's copied-result mode. Unknown
modes, missing child policies, inconsistent owner/release metadata, an unsupported
COP transfer or absent generated adapter refuse before provider effects. I will
review the concrete recursive slot-policy/schema mapping with the existing
callback/provider owners before implementing this wire extension.

I require original revision1/legacy controls alongside revision2 type/graph/
identity/refusal cases, combined exact-budget and budget-minus-one controls,
allocation-prefix rollback with unchanged caller budgets, and all consuming
reader/writer/VM/nvm2c refusal paths. Full source/installed SDK behavior remains
required; none of these private profiles substitute for it.

## I checkpoint the shared preparation budget before typed-profile widening

My preparation context contains remaining bytes and steps, capped at32MiB and
1,048,576. New budgeted signature, module, raw-provider and direct-V2 declaration
entry points stage a local copy and publish it only with their successful owner.
Existing entry points retain their original limits and acceptance profiles.
Module preparation charges signature measurement plus validation/copy passes;
it does not charge a hidden independent allowance. Raw-provider preparation
bounds all counts before reserving validation/copy work. It remains transport.

Declaration preparation charges its layout wire, ordinary owner, all copied
layout/field storage and a conservative full DAG workspace reservation before
allocation; it subsequently charges shared type/binding arrays and final union
plan storage against that same local context. Freed layout scratch remains
charged. Existing coarse whole-graph preflight and each later shared array-reader
step consume the same work allowance. No prepared plan retains a pointer to the
caller's budget or a stack-local context. A caller composing several owners must
stage its own context until the whole generation succeeds; the additive fixture
demonstrates that transaction and releases all earlier owners on later failure.

Original private fixtures remain, with additive combined module/declaration/raw
transport success at the measured exact cumulative budget, byte-minus-one and
step-minus-one refusal, zero-budget unchanged output, and every combined
allocation prefix under persistent/one-shot failure followed by fresh recovery.
This source checkpoint does not implement revision2 rows, lifetime wire, provider
cross-validation or consumer admission. Those remain the next required units.

## I checkpoint the private typed declaration profile

I add `nvm_prepare_ownership_declarations_typed_v2(module, budget, out)` as an
explicit private transport query. Existing legacy, retained-V2, and budgeted-V2
wrappers select their original profile. I use the same bounded layout decoder,
ownership descriptor reader, extension framing, union partition reader, and
ARRAY_FIELDS reader. Only the new entry accepts ARRAY_FIELDS revision2; its
count/type/binding encoding stays byte-for-byte the existing8/12-byte format.
Union partition revision remains1 and retains its original ordinal, offset,
unique-name, exact coverage and name-kind checks.

The typed layout path validates every table, even without a forward edge, with
the existing iterative graph walk. STRUCT/TUPLE/UNION/ENUM tags name matching
layout kinds. Scalar/string and ARRAY/FUNCTION/OPAQUE fields have NO_INDEX in
coarse storage. Every ARRAY field, including tuple or union fields, needs one
ordered shared element binding. Reference/declaration children do not become
inline storage edges. Resource modes remain unsupported by this new transport
profile. Array chains retain the existing64-node depth bound and cycle refusal.

Typed ARRAY_FIELDS rows admit exact layout referents and bounded foreign indices
below65,536. FUNCTION/OPAQUE rows and coarse fields/descriptors set the copied
`foreign_unresolved` fact. I have not resolved those indices against provider
nominal/signature tables; that fact is not a runtime capability or proof that
all other provider obligations are complete. Even when false, the plan remains
non-admitting. Exact declaration keys/owners/generic arguments, function/opaque
field bindings, full callback signatures, imported bindings, generated ABI and
lifetime policies still require the separate provider cross-validator.

Shared caller accounting covers the full typed graph workspace and repeated
union passes using the existing preflight. Success publishes budget and copied
plan together; failure preserves both. Original linked/instrumented controls
remain, with additive mixed record/tuple/union/enum rows, unused foreign rows,
exact signature selector, wrong-kind/cycle/reserved/depth controls, legacy
revision2 refusal, independent output lifetime and all allocation-prefix modes.
Strict production and both fixture-mode syntax checks pass. I have not executed
this new profile before source review.

The production consumer inventory still selects legacy/public wrappers:
`verifier.c`, `affine_bytecode.c`, `affine_state.c`, `managed_array_shapes.c`,
`record_array_structure.inc`, `nvm_v2_convert.c`, `nvm2c.c`, and `nvm2c_owned.h`.
The new API is called only by the private fixture. No module loader, VM, FFI,
COP, AOT emitter, conversion, feature bit or installed provider route selects it.
Whole-consumer acceptance/refusal qualification remains required before any
future execution activation; this checkpoint changes no such authority.

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

# My ordinary record array-element authority design

I propose the next required15f/aggregate488 dependency at canonical5d1d3d39f.
This is a design checkpoint, not executable admission. I preserve generic,
imported/qualified, mixed nominal, nested-array and recursive declaration scope
as required subsequent work. I do not duplicate the peer's scalar generic-union
implementation or infer its completion.

## My current boundaries

`src/nanoisa/nvm_v2_sections.h` defines twelve-byte retained field rows with a
runtime tag, nested nominal layout index and advisory name. There is no ARRAY
layout kind. An ARRAY tag alone has no declared element ownership information.
`ownership_contracts.c:check_layouts` accepts complete scalar/string/record
fields; it does not admit complete ARRAY fields. Its current versions1/2 and
COMPLETE/RESOURCE flags must retain their meaning. Current resource STRING and
specialized owner-array support must not be regressed to historical scalar-only
rules described in the original15f document.

`src/nanovirt/ordinary_authority.inc` omits publication for element_type/type
parameters. `nisa_ordinary_tag` and `nisa_ordinary_publication` in
`src_nano/compiler/nanoisa_codegen.nano` likewise lack array declaration facts.
The optional paired publisher must preserve source acceptance when it cannot
truthfully publish the whole supported table.

`managed_array_shapes.c:prepare_records` requires explicit ordinary authority
and its scalar/string/record field schema. `record_value_matches` currently
checks exact tags and nominal record origins, not array element contracts.
Its weak allocation-site summaries, array child summaries and record field
summaries provide a foundation, not the missing proof. `nvm_select_managed_heap`
and the LLVM/Wasm consumers must not acquire array-field admission merely
because declaration decoding succeeds. Private record storage/graph collection
already exists; its existence does not establish generated mixed graph cleanup.

## My proposed compatible metadata checkpoint

I propose one shared ownership extension envelope for the array and peer PR893
union-variant lanes. The shared decision is now frozen at peer90e6fe7b and implemented in
8fadd11fd6280738540bab24d242ccfe652c5c6a. Version3 is the common envelope;
my dependent branch extends its checked reader without defining another grammar. I allocate
no opcode, layout kind, public runtime tag or outer container section. Existing
LAYOUTS rows remain byte-compatible: an ARRAY field uses TAG_ARRAY and NO_INDEX,
never an invented record index. Existing v1/v2 payloads and accepted decisions
remain byte-identical. Older validators reject version3 with FORMAT_VERSION;
I do not advertise backward execution compatibility. The v1/v2 raw container
transport must preserve the opaque ownership bytes through roundtrip; any
semantic converter that rejects v3 continues to reject until separately reviewed.

The proposed version3 retains the v1 layout-count/flags/function-descriptor
prefix, then stores `path_bytes:u32`, exactly that many bytes containing the
existing v2 path-count/rows/alignment encoding, followed by `extension_count:u32`.
Even an empty path table encodes its zero count (four bytes). Existing paths,
indices, modes, maximum256 paths/depth32 and resource restrictions do not change.
The checked common reader passes a bounded path subcursor to existing path
validation; its exact-end check applies to that subcursor. Version2 still checks
the end of its original whole payload. No new paths arise from array metadata.

Each extension is `kind:u16, revision:u16, payload_bytes:u32`, followed by exactly
that many payload bytes and zero alignment to four. Kinds are strictly ordered,
unique and mandatory-understanding: unknown kind/revision refuses, never skips
potential ownership obligations. The agreed kinds are UNION_VARIANTS=1 and ARRAY_FIELDS=2, both
revision1, with at most two extensions. ARRAY_FIELDS revision1 contains the type and binding tables below.
UNION_VARIANTS must carry the peer's exact per-variant identities/field contracts;
this document does not invent or approve their payload. Coexistence is valid only
when the common validator validates both extensions and cross-references; neither
consumer can authorize a mixed table from just the extension it understands.

All integers use existing little-endian encoding. A type row is eight bytes: tag u8, three zero bytes, referent u32.
Scalar INT/U8/FLOAT/BOOL/STRING rows require NO_INDEX. An ARRAY row references an
element type row; a STRUCT row references an exact global retained layout.
A binding row is twelve bytes: global record layout u32, field ordinal u16,
zero u16, element type row u32. Both tables begin with u32 counts. Bindings are
strictly ordered by layout/field, unique, and cover every ARRAY field of every
COMPLETE ordinary declaration exactly once. Non-ARRAY bindings and trailing
bytes are invalid. Advisory names never supply identity.

My first production checkpoint is a **private non-admitting** reader/query for
this proposed payload. It accepts only complete ordinary scalar/string/record
DAG declarations with flat arrays whose element row is INT/U8/FLOAT/BOOL/STRING.
ARRAY/STRUCT type rows are reserved in this version's proposed grammar but are
UNRESOLVED in the first query, not silently flattened or certified. No resource
flag, borrowed descriptor, import, foreign or passive authority is admitted.
Existing complete v1/v2 declarations can be described unchanged; an ARRAY field
without the new exact binding is UNKNOWN, never ordinary by omission.

I factor one checked internal ownership reader, used by existing public validation
and the private query. It validates common descriptors, paths, extension spans and
all cross-table facts before projecting either lane's declarations. Public v1/v2
entry behavior remains unchanged; the public entry retains the peer's accepted union-only version3 behavior
and explicitly refuses ARRAY_FIELDS until a separate consumer review. My private
reader mode alone admits the new declaration grammar. The private query
requests a reviewed grammar version, not a trusted/skip-validation boolean.
Resource flags remain authoritative: ordinary parents cannot hide resource
children, including through element rows. Resource-bearing layout tables retain
their existing prior-order envelope restriction. Array metadata grants no owned,
borrowed or reference-path eligibility. Unsupported resource/union combinations
return UNKNOWN in this ordinary query after complete structural validation. The query owns all returned data and survives input destruction.
Statuses are DESCRIBED, UNKNOWN, INVALID, LIMIT, MEMORY; failures preserve the
entire caller output. Null output is invalid. Getter failure preserves outputs;
no getter exposes source pointers. My private query and report use the existing managed descriptor ceiling
NVM_RECORD_PLAN_MAX_LAYOUTS=256, not4096. Common wire counts remain u32; their
representability is not target eligibility. A structurally valid larger table
returns LIMIT before private report allocation. Other independent limits:
65536 total fields,4096 type rows,65536 bindings,64 type depth,16MiB total owned
and transient memory,1048576 traversal steps. Checked arithmetic precedes every
allocation/index. Visited iterative graph walks prevent exponential DAG work.
All limit exhaustion refuses; it never degrades into ordinary authority.

The joint envelope review is complete. I read the full common reader and Nano
producer delta at8fadd11f; the peer records the exact union payload in
`AFFINE_SCALAR_UNION_INSTANCE_CONTRACT.md`. I retain the current union-only public
behavior. My first array reader uses an explicit private mode of that common
validator and complete cross-table validation. No separate array-only envelope
reader or public classification bypass is introduced.

### My frozen proposed private query API

I propose `ordinary_array_authority.h` with an opaque owned report and these
names; implementation remains subject to the shared-envelope review:

```c
typedef struct NvmOrdinaryArrayAuthority NvmOrdinaryArrayAuthority;
typedef enum {
    NVM_OAA_DESCRIBED, NVM_OAA_UNKNOWN, NVM_OAA_INVALID,
    NVM_OAA_LIMIT, NVM_OAA_MEMORY
} NvmOrdinaryArrayStatus;
typedef struct {
    NvmOrdinaryArrayStatus status;
    uint32_t layout, field; /* NO_INDEX when no exact location applies. */
    const char *message;   /* Static storage, never input-owned. */
} NvmOrdinaryArrayResult;
typedef struct { uint32_t layouts, types, bindings; } NvmOrdinaryArrayCounts;
typedef struct { uint8_t tag; uint32_t referent; } NvmOrdinaryArrayType;
typedef struct {
    uint32_t layout;
    uint16_t field;
    uint32_t element_type;
} NvmOrdinaryArrayBinding;
NvmOrdinaryArrayResult nvm_describe_ordinary_array_authority(
    const NvmModule *, NvmOrdinaryArrayAuthority **);
void nvm_ordinary_array_authority_free(NvmOrdinaryArrayAuthority *);
bool nvm_ordinary_array_authority_counts(
    const NvmOrdinaryArrayAuthority *, NvmOrdinaryArrayCounts *);
bool nvm_ordinary_array_authority_type(
    const NvmOrdinaryArrayAuthority *, uint32_t, NvmOrdinaryArrayType *);
bool nvm_ordinary_array_authority_binding(
    const NvmOrdinaryArrayAuthority *, uint32_t, NvmOrdinaryArrayBinding *);
```

The opaque report owns a retained numeric layout copy, exact ordinary flags/maps,
type rows and bindings. It owns no AST/module/string/library reference. DESCRIBED
alone publishes a newly allocated report; UNKNOWN and every error leave *out
unchanged and free all staging. Legacy supported ordinary tables can describe
zero bindings; missing authority remains UNKNOWN. Free(NULL) is harmless. Getters
copy one complete value only on success and reject null outputs/out-of-range
indices unchanged. Structure padding is not serialized or used for equality.
These are declaration facts only; there is no executable-proof boolean.

## My separate source and provenance checkpoints

After private codec qualification, I propose shared validation and paired
publication with a separate consumer audit. Both producers derive the complete
element tree from checked TypeInfo / resolved selfhost annotation facts, not
array literal contents, spelling prefixes or runtime representation. They retain
exact global/per-kind layout maps, declaration flags, functions, initializers
and every selected shadow. A plain supported record with array<float> gains
truthful optional facts; arrays of resource-bearing/unknown types do not.
Allocation failure cannot publish one half of layout/ownership metadata.
Existing unsupported source acceptance and UNKNOWN publication stay unchanged.

Before any LLVM/Wasm execution, a separately reviewed origin query must prove
that every constructor, field replacement and aliased array mutation satisfies
the declared element contract. Record field GET exports the full array origin
set. An alias obtained before the array enters a field still constrains later
writes; checking only the original constructor is insufficient. Weak summaries
must include calls/results, globals, joins, repeated sites, slices and copies.
Unknown origins, uncalled-helper unknown parameters and unsupported writes refuse.
Packed representation compatibility is distinct from exact source element type:
a VM packed numeric coercion must not silently establish a different declaration.
Optional out-of-range GET values retain their actual tag alternatives.

This proof is neither affine authority nor lifetime permission. Ordinary aliases
retain shared roots; owner shells remain unique. Mixed owner/ordinary tables use
their separately qualified complete authority and are not routed into a closed
ordinary plan by clearing RESOURCE. File pending/service checks retain precedence.
Generated LLVM and import-free Wasm require separate matched constructor/read/
write/call/global roots, failure cleanup and graph safe-point qualification.

## My full required continuation

The bounded leaf-array checkpoint does not close15f or488. I retain these ordered
requirements: nested arrays and arrays of exact ordinary records; record-array
cycles and recursive nominal declarations; generic concrete-instance identities
and substitutions; imported/qualified definitions and link identity; mixed
resource/ordinary graphs with checked affine element restrictions; tuple, union,
map and callable element/field authority. Shared type graph representation must
be reviewed before admitting cycles or new kinds. A future rejected/unknown
stage is a temporary checked boundary, not removal from5.1 scope.

## My acceptance sequence

I first qualify private metadata decoding/query only: exact old payload stability,
new roundtrip transport or explicit unchanged semantic-converter refusal,
malformed/truncated/reserved/duplicate/missing bindings, distinct equal-shaped
nominal identities, resource/unknown element refusal, deep/shared graph limits,
allocation-prefix cleanup, unchanged outputs and input-destroyed plan lifetime.
No module executes from this query checkpoint.

Next I qualify paired source metadata on fresh C seed/Stage1/Stage2 with helper
shadows, normal entry/initializer/all selected shadows and unchanged old profiles.
Then separate provenance and execution reviews require alias writes before/after
field installation, replacement retaining old aliases, distinct mutable arrays,
call/global/branch flows, source numeric tags, first failure/output atomicity,
collector/reentry and allocation-failure cleanup. Linux/Darwin native LLVM and
Wasm acceptance must retain exact compiler/runtime/provider provenance. Tests
and production each require review before execution. No gate has run here.

## My shared-envelope implementation boundary

I base this dependent branch on8fadd11f, not on an independently assigned v3.
The common reader will accept an internal profile enum. Its ordinary public
entry keeps ARRAY_FIELDS refused; the new private query enables kind2 structural
validation, then narrows to its declared ordinary flat-array profile. Unknown
kinds/revisions, missing bindings and invalid cross-references still refuse the
whole table before any projection. Union facts must validate even when the
private ordinary query subsequently returns UNKNOWN for the mixed table.

My public layout classifier therefore cannot silently label new array-bearing
records ORDINARY before the separate executable-consumer audit. Public v1/v2
and peer union-only v3 outcomes remain controls. I retain every nested/generic/
recursive/import/mixed and final execution requirement above. This is a design
reconciliation only; no new array code or qualification exists at this checkpoint.

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

I propose ownership payload version3 for ordinary array declarations. I allocate
no opcode, layout kind, public runtime tag or outer container section. Existing
LAYOUTS rows remain byte-compatible: an ARRAY field uses TAG_ARRAY and NO_INDEX,
never an invented record index. Existing v1/v2 payloads and accepted decisions
remain byte-identical. Older validators reject version3 with FORMAT_VERSION;
I do not advertise backward execution compatibility. The v1/v2 raw container
transport must preserve the opaque ownership bytes through roundtrip; any
semantic converter that rejects v3 continues to reject until separately reviewed.

Version3 starts with the exact v1 layout-count/flags/function-descriptor encoding,
then appends an element-type table and field-binding table. It has no v2 path
suffix in this first ordinary-only form. All integers use existing little-endian
encoding. A type row is eight bytes: tag u8, three zero bytes, referent u32.
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

The private reader must independently validate the full common envelope and
all function/local/signature descriptors. It must not bypass the old validator
with a caller-supplied trusted flag or make the public validator accept v3.
Implementation review must pin the exact internal factoring before modifying a
shared reader. The query owns all returned data and survives input destruction.
Statuses are DESCRIBED, UNKNOWN, INVALID, LIMIT, MEMORY; failures preserve the
entire caller output. Null output is invalid. Getter failure preserves outputs;
no getter exposes source pointers. Proposed independent limits:4096 layouts,
65536 total fields,4096 type rows,65536 bindings,64 type depth,16MiB total owned
and transient memory,1048576 traversal steps. Checked arithmetic precedes every
allocation/index. Visited iterative graph walks prevent exponential DAG work.
All limit exhaustion refuses; it never degrades into ordinary authority.

Before coding, review must settle the version number against current allocations
and freeze the exact private API names/ownership table. This proposal does not
reserve version3 by itself.

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

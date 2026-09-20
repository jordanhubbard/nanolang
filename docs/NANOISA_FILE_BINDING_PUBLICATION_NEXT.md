# I publish exact File bindings before enabling paired source execution

I continue `task_8bbc1cf5295b4b59b314640ef57c725f` under6fc/72556/6931/d03c from
actual PR901 merge `ec51690f7028fc788145d4d84e630627a07df668`.
[My original source contract](NANOISA_FILE_SOURCE_PUBLICATION.md) remains binding.
The merged slice independently describes explicit C/Nano requests. It does not
read source declarations, establish their complete namespace, publish bindings,
lower File bodies or execute their shadows. This document is design only.

## My measured milestone and next boundary

Root independently checked892 original and293 integration report Git blobs,
1352+698 retained artifact objects,26+8 equal input pairs and18 complete producer
outputs with exactly75 selected shadows each. I retain all first failures and
the original32ade bootstrap,701 sanitizer andaf68 ordinary integration scopes.
Task8bbc stays open. I add no competing task for its next dependent slice.

My next first production checkpoint is a strict byte-snapshot/document plan and
pure binding renderer. A separately reviewed checkpoint adds no-replacement
publication and its explicit command. Paired parser/schema/checker work remains
held until the shared ownership envelope is agreed. Neither initial checkpoint
changes compiler selection or executes File services.

## My actual implementation inventory

| Existing source | Consequence for the next checkpoint |
| --- | --- |
| `src/nsi.c:nl_nsi_load_path` | Path-only loader, size from seek, allocating cJSON parse and generic NSI validation. I cannot reopen it after validating a different snapshot. |
| `src/nsi.c:keys_allowed` | Membership check does not reject repeated JSON keys. This is insufficient for the promised strict document boundary. |
| `src/cJSON.c:cJSON_Parse`, `cJSON_ParseWithLengthOpts` | The legacy entry does not demand complete input consumption; decoded string extents are not retained in `NlNsi`. I must reject raw/decoded NUL before comparing C strings and require a complete bounded document. |
| `src/nsi_file_plan.c:nl_file_plan_build` | Sole existing exact catalog shape validator; I reuse it after strict decoding. It checks eight types, five methods, one error and one capability. |
| `src/nsi_gen.c:nl_nsi_gen_nanolang` | Demonstration functions map RESOURCE/VARIANT to int. I leave this API intact and never advertise it as verified File generation. |
| `tests/fixtures/nsi_file_plan.json` | Existing exact temporary-file catalog document. The legacy `schema/nsi/modules/filesystem.nsi.json` instead describes open(path); it must refuse this mode despite sharing an interface ID. |
| `src/nanoisa/file_source_plan.c`, `src_nano/compiler/file_source_plan.nano` | Independent descriptive mappings; neither validates raw NSI bytes or a complete parsed source graph. |
| `src/main.c`, `src/nanovirt/main.c`, `src_nano/nanoc_v06.nano` | Distinct actual C-seed, C NanoISA and selfhost driver routes. Native C-seed output is not proof of C NanoISA lowering. All require explicit later integration. |
| `schema/compiler_schema.json`, `scripts/gen_compiler_schema.py` and `.nano` | Primary token/AST/Parser definitions and paired generators; no new numeric assignment is reserved here. |
| `src/generated/compiler_schema.h`, `src_nano/generated/compiler_schema.nano`, `compiler_ast.nano` | Generated consumers must agree with primary definitions and the peer's ownership envelope. |
| `src/parser.c`, `src/nanolang.h`, `src/env.c`, `src/module.c`, `src/emit_typed_ast.c` | C parse/name/type/module/copy/diagnostic facts must retain declaration identity and spans. |
| `src_nano/parser.nano`, `typecheck.nano`, `compiler/module_loader.nano`, `module_bindings.nano` | Nano Parser reconstruction, imported name binding and module union merging must preserve the added facts; a side table lost by a reconstruction is not sufficient. |
| `src/nanovirt/codegen.c`, `borrow_codegen.inc`, Nano `nanoisa_codegen.nano`/`nanoisa_borrows.nano` | Later independent AST-to-instruction lowering; no hidden delegation of Nano lowering to C. |

I record the loader's boundedness, complete-consumption, repeated-key and string
extent gaps before implementation. These are static findings, not measured
malformed-document executions or a general change to legacy NSI acceptance.

## My strict snapshot and pure rendering API

I propose separate `src/nsi_file_binding.h/.c` with opaque
`NlFileBindingPlan`, statuses `NL_FILE_BINDING_OK`, `INVALID`, `LIMIT`, `MEMORY`,
`UNRESOLVED`, `IO`, `EXISTS`, `UNSUPPORTED`, and the following first API:

```c
NlFileBindingStatus nl_file_binding_prepare(
    const unsigned char *bytes, size_t size, NlFileBindingPlan **out);
void nl_file_binding_free(NlFileBindingPlan *plan);
const unsigned char *nl_file_binding_interface_bytes(
    const NlFileBindingPlan *plan, size_t *size);
const unsigned char *nl_file_binding_source_bytes(
    const NlFileBindingPlan *plan, size_t *size);
```

The input is valid readable immutable storage for the call; input and output
storage are disjoint. I cap input at1MiB, JSON depth at64 and lexical tokens at8192
before allocating a decoder tree. An exact lexical preflight checks complete
UTF-8, JSON string escape boundaries and raw/decoded U+0000 rejection. Escaped
backslashes are interpreted correctly; I do not reject a literal backslash
followed by text as though it were an escape. The length-aware decoder must
consume exactly the input except legal JSON whitespace. I reject repeated
object keys after decoded-name comparison, including repeated allowed keys.
Unknown fields retain existing NSI v0 refusal; missing required fields do too.

I factor the existing NSI object-to-`NlNsi` validation into an internal reusable
helper rather than duplicate its enum/reference rules. The existing path loader
retains its compatibility behavior in this slice. The new strict path validates
the full decoded tree before using the shared helper and exact
`nl_file_plan_build`. A future global legacy-loader hardening change requires
separate review. I do not use a temporary pathname and reopen it to achieve the
byte API. Existing allocating parser helpers conflate malformed input and OOM;
where I cannot distinguish them I return UNRESOLVED, not invented precise MEMORY.
My own allocations return MEMORY. Every failure frees complete partial trees,
NSI/descriptor objects and buffers, preserving `*out`.

After exact validation I render canonical current-schema JSON and source bytes
from the same immutable facts. All eight types, every field/arm, five methods,
parameters, modes, rights-relevant catalog identity, error and capability remain
represented; the generator does not delete facts absent from a convenience
getter. Canonical JSON roundtrips through the strict path and the existing exact
validator. Catalog name/ordinal ordering is deterministic. No source IDs,
module identity, generated provenance hash or filename becomes runtime authority.
I bound each rendered file at1MiB and account for input, decoder, transient and
published buffers with checked products before source review. The plan owns
both immutable outputs independently of the input lifetime; getters borrow only
until free. A null plan or null size pointer returns NULL without writing size;
a valid getter writes the exact count and returns the immutable bytes. No `FILE *` streaming success is reported before a
complete render exists.

The source artifact uses the previously proposed declaration exactly:

```text
service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"
```

It contains no integer token constructors, fake extern implementations or `main`.
The declaration supplies all eight public nominal types and five callables.
I emit five real behavioral shadows, one named for each declared operation.
They acquire via temp and inspect both arms, assert exact successful progress,
write/rewind/read bytes and EOF as appropriate, and consume any acquired File.
An acquisition or cleanup error fails a positive lifecycle shadow after retaining
its error facts; it is not silently accepted as a successful host test. Fault
fixtures separately check Error semantics, byte range and close consumption.
Every helper emitted by a later template revision has its own shadow and enters
the exact selection manifest. Result matching and empty Ok arms use the reviewed
paired grammar; I do not fabricate an int payload for unit.

Before that parser is implemented, this is deliberately forward source text:
old compilers must refuse it. Pure render/publication tests do not compile away
its declaration or count sample text matches as executed behavioral shadows.
The full milestone still requires actual generated-module imports and all
selected dependency/root shadows through the paired producers.

## My no-replacement directory publication

I propose `nl_file_binding_publish(const NlFileBindingPlan *, const char *dir,
NlFileBindingPublishReport *)` as a separate source checkpoint. Its report records
status, saved first errno, secondary cleanup errno, `published`, and `durable`.
A well-formed invocation initializes the report explicitly; a malformed pointer
call remains outside valid-memory preconditions. Status/error precedence and
all partial stages are specified in its source review.

The caller supplies a stable trusted destination parent directory; this API is
not a sandbox against another process replacing ancestor directories. I resolve
and open that parent once, validate the final single component, and use anchored
directory descriptors thereafter. I create a unique owned sibling staging
directory with exclusive creation. Only `interface.nsi.json` and `binding.nano`
are created inside it, with exclusive/no-follow opens and checked complete
writes, fsync and closes. Relative `from` always names the exact companion.
I never execute a shell, derive filenames from document strings, follow an
existing final symlink, or overwrite an existing destination.

Publication uses the host's atomic exclusive directory rename: Linux
`renameat2(..., RENAME_NOREPLACE)` and Darwin `renameatx_np(..., RENAME_EXCL)`.
Unsupported hosts/filesystems return UNSUPPORTED; there is no check-then-rename
or replacing-rename fallback. Concurrent publishers may race, but at most one
can create the final name, and the loser cleans only its own staging. Existing
files, directories and symlinks are preserved, including an empty directory.
Before rename, failure removes only the two known owned children and owned
staging directory; no recursive deletion traverses an arbitrary supplied path.
I verify ownership assumptions before cleanup and retain cleanup errors.

After successful rename, `published=true` is irrevocable for this operation.
A later parent fsync/close failure returns a distinct reported durability/IO
failure with `published=true`; I do not delete the committed destination or
pretend it was never published. Successful data/directory sync establishes the
specified POSIX durability attempt, not a universal power-loss guarantee. I
capture each failing errno before any cleanup operation can replace it.
A killed process may leave an uncommitted owned staging directory; supervised
CLI/process cleanup is a later measured control, never an assertion of automatic
recovery after arbitrary SIGKILL or machine failure.

My initial explicit tool is `bin/nsi-file-binding INPUT --file-binding-dir DIR`,
built by its own Make target. No existing generic language generator changes
meaning; incompatible options/languages refuse. The tool reads one bounded
regular-file snapshot with checked open/read/close, then prepares and publishes
that exact snapshot. Path replacements after the read cannot change it. Output
is two files in one directory; neither stdout nor two independent renames is an
alternate publication path. Installed packaging follows its separate reviewed
source-consumer checkpoint, not this first private tool.

## My paired parser, checker and lowering plan

I retain the source spelling above, exact decoded token byte counts and source
spans. `service` and `catalog` are proposed contextual top-level words; I do not
reassign existing token/AST enum ordinals here. Only the complete declaration
may produce an immutable request; comments, module filenames, JSON sidecars and
ordinary types cannot insert requests into a compile. Relative document paths
resolve against the declaring module; one owned validated snapshot survives all
later passes. The parser must reject embedded NUL using original decoded counts,
not infer completeness from strlen or a later prefix slice.

A new service declaration representation carries interface/catalog identity,
owned document facts, source module identity/span and thirteen declaration IDs.
It is distinct from ordinary opaque/record/union definitions. The checker
resolves imported aliases to those IDs and copies exact File/OpenResult versus
scalar Result categories from validated plans. Source IDs are assigned after
whole-module collision checks and mapped explicitly to final global layouts,
STRUCT/UNION slots and SERVICE imports; reorderings are tested. No runtime
capability lives in the AST. Ordinary union matches cannot acquire File rules
because their arm names happen to be Ok/Error.

Before editing schema-bearing code I require the shared peer893 decision covering
ordinary scalar-union and record/array authority. I will inventory primary and
generated schema definitions, every Parser constructor/reconstruction, module
merge/clone, typed-AST export/import and native/selfhost consumer. I reserve no
v3 spelling, numeric node slot or competing payload envelope. An old consumer
must refuse required service facts; an ignored extension field cannot preserve
authority. Unsupported coexistence refuses the complete source graph instead of
dropping declarations. This is a concrete implementation hold, not removal of
paired source acceptance from5.1.

C-seed native output, C `nano_virt` source-to-NanoISA, and Nano Stage1/2 source
paths must converge on equivalent exact File modules through independent
lowering. The C/Nano descriptive plan is reused for identities only. The Nano
catalog/document bridge, if needed, may return bounded immutable NSI facts; it
may not return lowered code, fabricated stack maps or asserted body validity.
A shared C strict document parser is data ingestion, not permission to delegate
Nano AST lowering. Its allocation/process failure boundary remains explicit.

The first lowering profile stays acyclic with existing exact direct calls and
single exclusive borrow. It lowers actual source argument order, affine Result
branches/takes, File moves/drops and borrow ends through the qualified six FILE
instructions and checked call/return staging. I retain every function body,
including uncalled helpers, and publish complete version2 service120 metadata,
required bits1/9, nominal ownership flags, signatures and truthful local/stack/
reference maxima. Final bytes pass the actual hosted conjunction. Generic C
opaque/extern/evaluator or ordinary union fallbacks remain refusals.

I place File source selection after complete parse/module/type/identity checking
but before the C-seed `compile_modules`/FFI library loading and
`check_interpreted_shadows` path in `src/main.c`. File source then bypasses
`transpile_to_c` only through the reviewed checked-byte/direct-native route,
not by pretending that generic C resource emission implements File. The C
`nano_virt` front end branches before ordinary `codegen_compile`/wrapper
publication; the Nano `compile_program` path branches before ordinary shadow
assembly/`--check-shadows`. Typed-AST export must preserve all required service
facts under the agreed envelope or explicitly refuse; diagnostic output is not
an excuse to emit an incomplete reusable representation. No rejecting branch
falls back to ordinary compilation. These changes remain later source-review
checkpoints and do not belong to the pure generator implementation.

Driver surfaces use existing `--emit-nvm`, `--target c` and native output modes
where actually supported; C-seed native and C NanoISA option tables are reviewed
separately rather than assuming identical flags. The explicit new
`--allow-temporary-files` is required before File shadow execution on every
relevant driver. Each selected dependency/root shadow receives a fresh grant
and context, exact original selection identity, all helper bodies and VOID-only
startup. Missing grant, unsupported initializer/global or failed cleanup prevents
entry and publication. The legacy `nano_vm --check-shadows` File refusal stays;
new supervised scalar-entry invocation uses the already qualified granted API.
Generated native functions link the explicit package with collision-free entry
identifiers and no VM engine fallback. No source output is published before all
selected shadows and final validation succeed.

## My ordered implementation and acceptance

1. Independently review this design; implement only strict byte ingestion and
   pure immutable binding rendering, with full allocation/cleanup table. Review
   source and fixtures before any new qualification.
2. Qualify complete JSON consumption, decoded-key uniqueness, NUL/UTF-8/extents,
   input/count/depth/output budgets, all catalog mutations, object-key reorder acceptance and catalog-array ordering
   verdicts matching the existing exact validator,
   exact canonical roundtrip, allocation prefixes/transient failure, copied
   lifetime, output sentinels and unchanged legacy generation. No service runs.
3. Review then implement the publisher/explicit tool. Test every open/read/write/
   flush/sync/close/rename cleanup edge, existing file/directory/symlink sentinels,
   concurrent no-replacement winners/losers, arbitrary filename bytes/quoting,
   saved first/secondary errno and postcommit durability reporting on Linux/puck.
   Retain attempted operations, not merely successful file counts.
4. Settle peer schema coexistence; review paired parser/checker/clone/transport
   source. Qualify exact C-seed/Stage1/Stage2 declaration/alias/type/span facts,
   every helper shadow, malformed whole-program refusal and allocation/process
   failure containment. A forward-generated file that old compilers refuse does
   not satisfy this step.
5. Review independent lowering and final byte transport, then supervised source
   shadows/publication. Run the full generated binding lifecycle corpus, all
   helpers/shadows/initializers, C-seed/fresh Stage1/fresh Stage2, actual VM and
   direct native O0/O2, package/install/outside-tree consumer and supported strict
   sanitizer controls on both hosts. Preserve existing full conversion/core/
   bootstrap/product requirements and every first terminal.
6. Retain mandatory cyclic/fuel, finite indirect and richer/multiple-borrow source
   acceptance under their original parents. The separately owned cyclic carrier
   is still a private prerequisite; no source-visible fuel option or broadened
   public ABI is selected by this design. Passing the bounded acyclic source
   corpus cannot close the full File/Socket/GPU parent.

## I pin the first decoder allocation and work budget

Before allocation I scan the complete JSON grammar with a64-container recursion
limit and at most8192 tokens,256 objects,64 members per object,256 array elements
and4096 raw bytes per string/number. Decoded strings cannot exceed their raw
extent. With B=1MiB and T=8192, cJSON allocates at most(T+1) nodes, B+2T bytes of
retained decoded key/value strings, and4097 bytes of one numeric scratch buffer.
My own terminated snapshot costs B+1. NSI copies charge B+2T string bytes,
sizeof(NlNsi),256 times the sum of every NSI element struct size, and257 ID
pointers for uniqueness scratch. Each NSI element corresponds to a distinct
validated JSON object; fields are copied once, not once per reference. I add the
actual owning File-plan size from its nonallocating owning-TU query.

The published binding is one allocation of its header plus the two bounded
outputs and terminators. I free the first decoder/NSI/descriptor phase before
canonical validation; the peak upper bound is this entire published allocation
plus the conservative decoder bound above. I check every sum/product and refuse
above16MiB. The bound excludes caller input, C stack, allocator overhead and
libc internals; it includes every project-requested dynamic allocation.
Getters expose both exact published allocation and conservative peak bound.

My recursive lexical/tree/cJSON/delete paths are bounded by64 containers, not
input length. Duplicate-key checks are at most64*63/2 comparisons per object,
at most256 objects, and at most4097 bytes per comparison. Existing NSI ID checks
are at most257*256/2 bounded-string comparisons, with other reference lookups
bounded by the same object cap. I retain these finite work limits rather than
claim linear-time parsing. No custom global cJSON allocator hook is installed;
callers must externally serialize against cJSON global hook mutation, and this
API does not establish cJSON thread safety. Allocating cJSON/shared decoder
failure remains conservative UNRESOLVED; owned snapshot/plan/descriptor OOM is
precise MEMORY. Canonical validation is one lower-level decode call, with no
recursive preparation/rendering.

Root reports the peer's reviewed-in-progress envelope acknowledgement at
`8fadd11fd6280738540bab24d242ccfe652c5c6a`, design90e6fe7b: v3 path_bytes
contains an exact v2 substream and unique ordered TLVs, kind1 UNION_VARIANTS
revision1, reserved/refused kind2 ARRAY_FIELDS revision1. I do not treat that
unmerged checkpoint as canonical source acceptance. Later paired parser/schema
work must integrate the actually reviewed producer/consumer rules and required
unknown-kind refusal; this strict NSI checkpoint assigns no shared numeric slot.

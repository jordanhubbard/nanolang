# My explicit C-seed provider owners for companion preparation

I record the measured10b74 both-host Stage1 link failure under task28cf8 before
changing manifests or module imports. My ordinary-record correction passed source
type checking. The C seed then linked multiple module aggregate objects containing
nsi_file_plan, file_source_catalog, cJSON and UTF-8 definitions. I preserve those
terminals and stop dependent checks. Nano's native preparer has not run yet and
cannot repair this earlier link.

## My existing implementation boundary

`module_builder.c` compiles all `c_sources` into each module's exported aggregate
object. It separately compiles `shared_c_sources` with hidden visibility and adds
them only to that module's shared library. `module.c::compile_modules` deduplicates
physical module generations, not arbitrary C sources inside different aggregates.
I retain these semantics, cache identity, native-unit evidence and cleanup. I do
not add symbol suppression, first-provider-wins linking, archive-order selection,
new runtime admission or changes to the peer-owned module loader.

Moving a repeated provider only to `shared_c_sources` is insufficient: its static
consumer would lose that provider when compiled alone. I instead declare one
physical public provider owner and an explicit real Nano import edge from every
affected consumer. The existing collector then includes that owner's metadata and
object for standalone and combined programs. Shared libraries keep private copies
of required dependencies, so interpreted shadows and dynamic wrappers remain
self-contained. Those copies use the exact existing source, not forked catalogs.

## My concrete ownership and import table

I introduce two ownership-only module directories, each with an ordinary module
manifest and a `.nano` file containing only its module declaration and explanatory
comment. They add no functions, callable shadows, service or source authority.
Their real import edge exists to include the selected canonical provider; no
synthetic name lookup or module spelling bypass substitutes for the edge.

| Canonical provider | Public static owner | Consumer changes |
| --- | --- | --- |
| src/cJSON.c | modules/cjson_provider, one c_sources unit | compiler_support, std/json and file_companion import the owner; their duplicate cJSON unit moves to shared_c_sources. |
| src/nsi_file_plan.c | modules/nsi_file_catalog_provider, one c_sources unit | nanoisa, forth_see, file_source_catalog and file_companion import the owner; duplicate plan units move to shared_c_sources. |
| src/nanoisa/file_source_catalog.c | existing file_source_catalog module | file_companion imports that existing module; its duplicate catalog unit moves to shared_c_sources. |
| src/utf8.c | actual final C/Nano runtime list | file_companion moves its duplicate unit to shared_c_sources, matching the existing compiler_support private UTF-8 dependency. |

I retain each consumer's other original c_sources in order. I append newly private
units in their previous relative order, retain existing private units once, and
retain all headers, flags, package queries, platform settings and public wrapper
signatures. Explicit imports use unambiguous qualified aliases and add no wrapper
calls. Empty ownership modules must pass both real parsers/module collectors; I
will not replace a failed import with an assumed link flag.

The companion's public aggregate retains its bridge, source reader, snapshots,
strict binding and NSI decoder. Its shared library privately retains plan, catalog,
cJSON and UTF-8 implementation dependencies. Its imports provide the exact public
static owners instead. File catalog's public aggregate retains the catalog TU;
its plan implementation is private dynamically and separately owned statically.
The new plan owner needs no NSI decoder: the actual plan TU uses its own checked
catalog and libc, with no external NSI function calls.

The C seed's provider compile contexts remain per owning module. Factoring changes
which module owns compilation of the public shared unit, so I require fresh source,
metadata, flags, compiler and resulting object identities; I do not claim bitwise
identity with an arbitrary old consumer's copy. There is exactly one canonical
owner for these four provider families in the affected manifests. The Nano native
preparer still checks all canonical requests against its final ordered profile,
including requests in shared_c_sources. Its actual runtime already owns cJSON and
UTF-8; those requests share that owner only after exact context agreement.

## My complete qualification before acceptance

I review source before execution and extend the existing companion fixture gate.
Both hosts require fresh C-seed/Stage1/Stage2 bootstrap and all original shadows.
I compile/link ordinary standalone users of each affected data module and combined
users with different import orders, including compiler_support plus JSON plus
companion/catalog/NanoISA. Actual static link inputs and symbol inventories must
show one public provider owner; no missing provider may be masked by another
unrelated test executable. Existing C-seed interpreter/module shared-library tests
must still exercise private dependency closure. I inspect dynamic export tables
for the moved private definitions, preserve public wrapper exports, and retain
actual shared library dependency identities. Shared copies are not claimed to
share allocator-hook or global mutable state with the public owner.

I retain the provider preparer's compiler/profile conflict, required-failure,
concurrent invocation, exact final runtime alias and known-object cleanup controls.
I retain existing Forth SEE, JSON, compiler-support/module metadata and wrapper
neighbors because their manifests participate in the correction. I preserve the
strict catalog/reader/publisher/source-plan fixtures and all File public refusals.
No library load or provider build becomes permitted in descriptive service graph
preparation; that path still stops earlier without effects or output publication.

Other overlapping NanoISA/Forth implementation families already listed in both
manifests are a separate general co-import closure concern, not silently proven
by this four-family correction. I do not weaken full5.1 or general standalone
module requirements: any measured failure there remains a recorded blocker with
its own complete source review. Full File source nominal typing, independent
lowering, execution shadows, grants and publication remain open under full8bbc.

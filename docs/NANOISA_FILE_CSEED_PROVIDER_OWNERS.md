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

## My prepared source checkpoint

I add the two ownership-only manifests and module declarations, eight explicit
consumer import edges, and the exact six-consumer public/private manifest split.
[NANOISA_FILE_CSEED_PROVIDER_OWNER_DIFF.json](NANOISA_FILE_CSEED_PROVIDER_OWNER_DIFF.json)
retains the old/new ordered provider lists and unchanged import targets. Static
canonical-path enumeration of all module manifests finds one public cJSON owner,
one public nsi_file_plan owner, the existing single catalog owner, and no module
public UTF-8 owner (the actual final runtime supplies it). Every unrelated manifest
field is byte-value equal before and after JSON reformatting. Existing private
providers remain first, followed by moved providers in their original order.

I add no new C allocation, runtime struct, module-loader branch, metadata grammar,
provider flag or wrapper signature. Existing module preparation allocates its
normal records/objects for two additional selected physical modules and imports;
those costs belong to compiler/module preparation, not my snapshot64MiB bound.
Shared-library copies retain existing hidden compilation and cleanup paths.
The complete standalone/static/dynamic/combined fixture supplement remains required
before any corrected bootstrap or module execution.

## My remaining Forth co-import prerequisite

I record an additional static closure defect before attempting a predictable link:
the Forth and NanoISA manifests both publicly compile the same 34 canonical C
sources. This includes the NanoISA C wrapper, ISA/verifier/assembler/disassembler,
ownership/format providers and selected VM value/heap/dispatch providers. The exact
ordered intersection and proposed ownership lists are in
[NANOISA_FORTH_PROVIDER_OWNER_PLAN.json](NANOISA_FORTH_PROVIDER_OWNER_PLAN.json).
This is manifest evidence, not a newly executed failure. My original10b74 failures
and the four-family correction remain separately retained.

I propose the existing NanoISA module as the sole public static owner of those
34 units. I retain its manifest and wrapper declarations unchanged. Forth keeps
only forth_see.c as its public unit, moves the 34 units to its private shared list
in original order after the existing private plan dependency, and imports the
actual NanoISA module through the qualified ForthNanoisaProvider alias. Its
existing explicit plan-owner import remains. No new provider module, spelling
exception, module-loader suppression or source copy is needed. The NanoISA-only
file_flow.c remains in its existing owner; selecting that module adds its existing
closure to a standalone Forth static link rather than omitting part of that owner.

I retain every other Forth manifest value and its actual extern declaration.
Forth's shared library still links its original complete C dependency set, now
with those dependencies hidden rather than exported. Its nl_forth_see export must
remain callable. A separately loaded NanoISA library keeps its own wrapper exports.
I do not promise shared mutable state between private copies. The Nano native
preparer continues to compare canonical source, language/compiler and final
ordered profile before deduplication; the new import cannot override a conflict.

Standalone Forth now selects NanoISA's actual source import closure and mandatory
shadows, including its existing filesystem module dependency. I require that
complete selected-shadow multiset in all three real producers. These are ordinary
assembler/file fixtures, not File service source admission. Descriptive opted-in
service graphs must still refuse before module preparation, shadow execution or
output publication; my existing no-compiler-attempt controls remain unchanged.

Before qualification I require source review of this extension, followed by the
complete fixture supplement: standalone affected modules; NanoISA plus Forth in
both import orders; the full compiler-support/JSON/companion/catalog/NanoISA/Forth
co-import in both orders; actual wrapper behavior; exact public object owner and
dynamic export checks; original provider conflict/concurrency controls; fresh
C-seed/Stage1/Stage2 bootstrap and all original companion gates on both hosts.
The Forth shared-library wrapper check must work in its own process without first
loading NanoISA or another test library that could mask a missing dependency.
Linux builds remain held for capacity. I do not execute a known-conflicting old
combined program as a substitute for fixing this statically demonstrated defect.

I implement the reviewed extension with exactly the proposed ordered lists and
one actual import. Before treating the private dependencies as complete, I compare
all 130 exported names from the retained ordinary10b74 file_flow object against
the 34 moved provider TUs and their complete 96-file quoted project include
closure. None refers to a file_flow export. The retained object's entire quoted
source closure is byte-equal to my current source. I retain the hashes, exact nm
output and name inventory in
[NANOISA_FORTH_FLOW_REFERENCE_AUDIT.json](NANOISA_FORTH_FLOW_REFERENCE_AUDIT.json).
This conservative textual reference check does not execute a binary, model system
headers or prove dynamic linking. Isolated real dynamic loading/wrapper behavior
remains required. No file_flow private copy is needed by the observed project
references; the public NanoISA owner keeps its existing copy and source closure.

## My examples Forth library recipe prerequisite

I also inspect the required existing examples Forth SEE neighbor before running it.
Its independent libforth_see.so recipe retains an older explicit provider list,
omitting six units already required by the current Forth manifest: mixed float
proof, service bindings, module service bindings, File nominal identity/plan and
the strict NSI File plan. Listed format/ownership providers reference these
families. I record the exact old rule and complete proposed canonical list in
[NANOISA_FORTH_EXAMPLES_CLOSURE_PLAN.json](NANOISA_FORTH_EXAMPLES_CLOSURE_PLAN.json).
I have not executed this known incomplete recipe or relabeled it as a measured
failure.

I propose a named FORTH_SEE_C_SOURCES list in examples/Makefile containing exactly
the current manifest public-plus-private source union, once each and in that order.
Both the library prerequisites and compile command consume that same variable.
Existing headers, flags, output path, test executable rule, compiler selection and
test-forth-see behavior remain unchanged. This independent examples library is a
complete shared artifact, not one of the aggregate static module owners; I do not
change its visibility policy or use it to mask the isolated manifest-library tests.
The fixture checks exact recipe/manifest source-set agreement and actual library
load/disassembly, in addition to running the existing test-forth-see target.
This narrow build closure correction requires review before implementation.

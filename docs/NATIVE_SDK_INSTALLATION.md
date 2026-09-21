# My installed native compiler and runtime closure

I keep this full5.1 prerequisite under task8bbc. This is a design checkpoint,
not installed acceptance. My File companion qualification at65b remains separate;
I change no qualified source, provider or fixture tree here.

## My observed gap

My current Make install copies compiler/VM/emitter binaries and the explicit File
package. It omits the general runtime headers, C sources, standard modules,
module manifests and generator inputs consumed by native compilation. My source
bytecode CLI additionally needs the exact prebuilt obj/ closure named by
wrapper_gen.c for ordinary and daemon wrappers. I install that checked object
inventory in the same generation, retaining its bytes/modes in package identity. My C seed
infers a root two directories above argv[0]; PATH-only invocation can fall back
to cwd. My Nano driver finds a source repository by searching cwd/input ancestors.
Neither establishes an installed runtime outside a source checkout.

I also observe two necessary subordinate boundaries. My module cache defaults to
module_dir/.build, which cannot serve a read-only installed SDK. My C module list
generator invokes ./scripts/generate_list.sh relative to cwd and writes generated
list products under shared /tmp paths. I must address those actual paths; shipping
headers alone cannot make the installed compiler/module closure complete.

## My installed generation and inventory

I propose one immutable package generation at
`PREFIX/lib/nanolang/sdk/<identity>/`, with `bin/`, `src/`, `src_nano/`, `modules/`,
`stdlib/`, `scripts/` and `tools/` retaining their relative layouts. Public
`PREFIX/bin/nanoc` and `PREFIX/bin/nanoc_c` select the generation's corresponding
real binaries through relative symlinks. The generation retains Stage1 as an
internal qualification/debug binary; it does not add a public Stage1 command.
Existing VM/emitter commands retain their public paths through generation-relative
symlinks too; source-capable nano_virt must discover the same generation. Explicit
File package commands/headers retain their paths.

A checked, committed relative-path inventory selects SDK inputs, including source,
headers, generated schema, module metadata, script/template inputs and module data
actually required by those modules. I generate it from repository-owned inputs,
then require every runtime-provider, header, metadata dependency and script path
used by the actual compiler to appear in it. Make install uses this inventory,
not a recursive copy of a developer's working directory or cache. Release source
archives carry the inventory, so installation does not require Git. A verification
tool rejects absolute paths, traversal, duplicate paths, unexpected symlink inputs,
missing files and input/output overlap before copying.

The generation identity is SHA256 over ordered canonical manifest rows containing
relative path bytes, file lengths, SHA256 content digests and modes, including
rows for the exact C-seed/Stage1/Stage2 binaries. The
manifest records native array ABI2 and every copied byte identity. The manifest
itself is a derived output, excluded from its own digest. This is package identity,
not a signature or an authority grant. Installed files are trusted compiler inputs;
I do not claim protection against a party modifying that trusted installation.

I stage a complete generation privately under the destination parent, verify the
copied inventory, then rename it into its final identity directory without
replacing a different generation. An existing same-identity directory is usable
only after exact inventory verification, including file modes and the complete
set of owned directories. Unexpected files, directories and symlinks refuse reuse. Only then do I atomically replace each
public compiler symlink. An already running compiler retains its original real
executable path and SDK generation. I keep old generations until explicit uninstall
or a separately authorized cleanup; I do not delete them during upgrade.
Uninstall follows the install-owned manifest and removes only those files/links,
then empty owned directories. Unrelated prefix sentinels and user caches survive.
I retain the first failure, known staged paths and post-commit status explicitly;
I do not describe a partial install as an atomic multi-command transaction.

## My paired discovery boundary

I factor OS executable discovery into a data-only C helper, with a bounded caller
buffer, explicit status and no output write on failure. Linux reads /proc/self/exe;
Darwin uses _NSGetExecutablePath followed by canonicalization. I retain the actual
executable's containing directory, so symlink/PATH invocation selects the matching
installed generation instead of the caller's cwd. Source-tree binaries continue
to resolve their adjacent source root. I validate the expected source/layout
markers and ABI revision before invoking a provider compiler.

I expose that immutable path/status to the Nano driver through its existing
compiler-support bridge; the bridge does not resolve language declarations,
parse imports, lower source or confer File authority. Both producers own their
ordinary source/import logic. I separate runtime-root discovery from source-project
search: explicit ./ and ../ imports remain relative to the declaring module;
local project imports retain their current ordering. The installed SDK is the
standard-library/runtime fallback, not a replacement for local declaring origins.
Original canonical module identities and qualifier/member tuples remain unchanged.

I provide an explicit NANOLANG_SDK_ROOT override for controlled embedding/tests.
A present invalid, mismatched or incomplete override fails before provider
compilation and preserves output; it does not silently fall back to cwd. Without
an override, discovery selects the executable-adjacent generation/source root.
The helper returns owned/copied counted path data with explicit lifetime. I retain
ordinary compiler entry/refusal behavior and all existing File opt-in/public guards.

## My writable products and runtime ABI

Installed SDK inputs may be read-only. Native provider objects remain in the
existing invocation-private native directory. C module generation uses the
existing checked generation/cache protocol in a writable user-selected cache.
An explicit NANO_BUILD_CACHE retains precedence. Otherwise an installed compiler
creates an invocation-private cache beneath the selected temporary directory,
records ownership and releases it after compiler/module-loader shutdown. Source
checkouts retain their current default manifest cache policy. Generated C module
objects use a separate invocation-private objects directory on both installed and
source routes: the old shared obj/nano_modules names cannot satisfy concurrent
compiler isolation. I retain these objects until the final native link completes. I never create .build inside
an installed SDK or silently share an unowned /tmp basename across invocations.

I inventory both module-level and final-compilation list-generation call sites.
They invoke the SDK-owned script by a checked quoted absolute path, write to an
invocation-private generated directory, propagate required failures and pass that
directory to the actual compile/link closure. I preserve source type identity and
existing list semantics; this is a location/lifetime correction, not a new list
representation or a spelling-based provider suppression. All generated products
and loaded module generations remain live through their actual consumers.

Every installed native module compiles against the generation's canonical
DynArray header and sources. ABI2 remains explicit. Actual ABI1/missing/wrong-image
array-bearing exports must refuse before entry in C-seed, Nano native, VM and
standalone emitted paths. Scalar exports that internally use DynArray require
fresh/revalidated header-dependent provider compilation too; entry guards alone
cannot establish their layout. I preserve existing transitive depfile/preprocessing
cache checks and final ordered native provider profiles.

## My implementation and acceptance order

1. I inventory all copied inputs, discovery call sites, generated-list paths and
   cache ownership. I implement the checked installer/discovery/data bridge and
   private writable-product lifecycle, then submit complete production/allocation
   and failure-path source for review before new fixture execution.
2. I add meaningful installer and paired compiler fixtures. They cover missing or
   stale SDK/ABI, partial install, existing generation, symlink/PATH and spaces in
   prefix, exact explicit override, read-only SDK, concurrent isolated compilers,
   required generator/provider failure and preserved output. Exact installed
   inventories and unrelated-prefix sentinels are checked before and after uninstall.
3. I fresh-bootstrap C-seed/Stage1/Stage2 with the new discovery closure on Linux
   and Puck. I install into isolated prefixes, then run from directories with no
   source checkout ancestor, clear development-root/module-path overrides and make
   the original checkout inaccessible to the test process's path selection. I
   retain actual compiler/provider argv proving every selected SDK input is under
   the installed generation or the explicit user project, not the source checkout.
4. I execute ordinary scalar/string/large-record and module programs, all mandatory
   selected shadows, explicit provider owner/co-import combinations in both orders,
   real dynamic wrappers and stale-ABI-before-entry cases through all three
   producers. I retain normal and supported sanitizer scope precisely. The existing
   full File companion/parser/schema/provider/public-refusal neighbors remain
   required, with actual installed binaries and rebuilt SDK module artifacts.

This checkpoint cannot close paired File nominal resolution/lowering, actual
source File execution, startup/shadow effects, cyclic/indirect/richer-borrow work
or full release acceptance. I keep those original requirements open.

## My first production checkpoint

I place discovery in the existing module_build_dir owning translation unit. Its
installed checks consume sdk.inputs and every named regular file before returning
an output path; the path and installed flag remain untouched on failure. The
C seed and nano_virt call preparation before ordinary compilation. My Nano driver
calls the same data bridge before import preparation; the bridge retains one
process/thread-local result and returns a copied Nano string. The ABI query is an
explicit exception: it reports the compiled constant without SDK preparation and
performs no provider work. It exists so installation can validate fresh binaries.

I keep project-origin import search separate from the runtime root. My native
metadata collector now selects project and SDK modules in original selected-file
order. Each module's metadata-root-relative flags and dependencies use its owning
root; identical roots retain the original behavior. I neither suppress providers
by name nor delegate Nano source resolution to the bridge.

| Boundary | My owned state and checked extent | My failure/lifetime rule |
| --- | --- | --- |
| Installation inventory | At most8192 files, at most1GiB total file bytes, each relative path at most2048 UTF-8 bytes, manifest at most4MiB | I preflight before hashing/copying and refuse changed copy lengths. Python objects and allocator overhead are not a1GiB heap guarantee. |
| Installation streams | One1MiB hashing/copy buffer, counted file bytes; no whole binary copy in memory | I verify copied bytes/modes and exact file/directory sets before exclusive publication. |
| Manifest verification | Python manifest at most4MiB plus bounded rows/path sets; C4096-byte row,2049-byte previous path,4096-byte resolved path and8192-byte hash buffer | C consumes one row/file at a time, checks ordered unique rows and the generation digest. Libc/OpenSSL allocation is outside an exact project-heap bound. |
| Root discovery | Three4096-byte automatic paths; caller capacity checked before publication | OS executable identity selects source or installed root. Source markers and compiled ABI must agree. An invalid override refuses without fallback. |
| Private work | One4096-byte process path, creating PID, retention flag; per-call private objects directories | Only the creating process removes work, after loader shutdown. Requested diagnostics retain the known path. Cleanup failure reports that path and does not overwrite an earlier result. |
| Generated lists | C main creates one private directory; module compilation uses its existing private directory; identifier lengths at most127 | I execute the exact script with argv, check its terminal, and include generated files from their actual directory. Parent compiler/shadow supervision remains the time bound; I add no independent general supervisor. |
| Uninstall | Manifest-owned files and their exact ancestor set | I validate all surviving owned files before deletion, preserve modified inputs and unknown files/empty directories, and remove only exact owned public links. |

These extents bound selected package work, not total compiler memory. Existing
parser, code generator, Python, OpenSSL and process-allocation failure semantics
remain explicit. I make no paired recoverable-OOM claim. The installed C verifier
checks recorded bytes and executable bits; write-bit removal for a read-only SDK
is allowed during execution. Installer reuse and uninstall require exact recorded
modes, so qualification restores its own read-only mode changes before uninstall.

I treat the installation prefix and selected compiler inputs as trusted, stable
administrative state while installation runs. I reject observed symlinks and
unknown entries, use an authoritative no-replace generation rename, and retain a
stage whose root inode no longer matches mine. I do not claim hostile concurrent
ancestor-substitution protection. Generation publication and each command link
are distinct commit points; a later failure reports whether my new generation
committed. Completed links and generations are retained for explicit recovery.
The existing explicit File archive/header installation remains a separate Make
step with its own failure status.

My current source-only checks are C99 strict syntax for the five affected C
translation units, Python syntax, all1161 selected input paths, and diff checks.
They do not establish linking, installation, bootstrap, generated-source behavior
or shadows. The121 wrapper objects must be freshly built, checked against the
actual wrapper inventory, and installed with the compiler generation. I require
the complete fixture checkpoint before executing this implementation.

## My required installed input closure

I do not accept a merely self-consistent partial manifest. My owning runtime
compiles the exact sorted path table generated from the1161 committed source/data
inputs,121 wrapper objects and nine compiler/VM/emitter roles. Linux also requires
the assembler capture helper. I require all1292 Linux or1291 Darwin rows in that
exact order; missing/additional paths refuse before discovery publishes a root.
Every bin role and the list generator must retain executable bits. I separately
require exactly one canonical DynArray ABI declaration matching the owning
binary's compiled ABI2. Installation queries all three actual compiler binaries
for their compiled ABI before copying them. I do not infer arbitrary binary
semantics from a manifest or turn package identity into a signature.

scripts/generate_native_sdk_inventory.py derives both the compiled path table and
Make's object prerequisite list. I keep those generated files committed and check
full-byte equality before installation. Changing either committed JSON inventory
requires regenerating both files; stale output refuses installation. Header
prerequisites include the runtime table. A clean make install now requires the
actual bootstrap, VM/emitter targets and all121 explicit object targets before
its installer runs. I preserve caller compiler/link flags through existing target
recipes rather than building an undocumented second provider closure.

The generated path table is immutable static data; validation adds no per-row
project allocation and scans it alongside each manifest row. The fixed row/file
limits remain unchanged. My fixture checkpoint must remove a required source,
object and each compiler role while recomputing a self-consistent generation,
then observe root/output refusal and zero provider invocations. Wrong/duplicate
ABI declarations and non-executable roles remain separate required controls.

## My unexecuted fixture checkpoint

I run actual `make install` from a fresh source tree with no bin/obj/lib products,
retaining the empty preinstall inventory and all build output. That target owns
fresh bootstrap and the wrapper closure; I do not describe a prebuilt binary copy
as clean installation. Reinstall must select the same generation. The ordinary
installed generation is shared by the host's fresh selected discovery probes:
GCC/Clang ordinary and ASan+UBSan on Linux, Apple/Homebrew ordinary and Homebrew
ASan+UBSan on Darwin. Those probes compile only module_build_dir.c and their C
harness under the selected instrumentation. Installed compilers/providers remain
ordinary; full-provider sanitizer acceptance is not implied.

My fixture runs the three installed producers with the source checkout renamed
out of its original path, from a separate user directory, and checks complete
selected shadows plus compiler argv. It covers read-only generations,504-byte
record arrays, local imports plus SDK JSON, simultaneous native invocations,
all36 provider ownership/co-import cases, real Forth bytecode, standalone and
daemon wrappers using installed objects, both dependency-header import orders,
and missing/ABI1/ABI2/unknown native foreign providers before entry. Foreign
incompatibility is deliberately tested outside shadows; the existing abort
terminal and absent entry marker remain exact assertions.

I retain self-consistent missing-source/object/each-compiler-role manifests,
wrong/duplicate ABI declarations, removed executable modes and invalid overrides.
All three installed compilers must preserve existing output and make no observed
compiler invocation when root discovery refuses. Required user-provider compiler
failure and required list-generator failure are explicitly modeled real tool
terminals, with final output preservation. My partial-install control injects a
first regular-file fsync failure through the actual Python installer and checks
its wrapped error/cause, no committed generation and retained user sentinel.
This labels the injection; it does not claim a measured storage-device failure.

FIFO manifests, unknown generation entries and a symlinked public bin ancestor
remain negative controls. Actual Make uninstall must remove every owned row,
command, File archive/header and manifest while retaining unrelated prefix and
in-generation file/directory sentinels. All commands use durable output, bounded
process-group cleanup and retained first terminal. The complete source File
companion/parser/schema/refusal matrix remains required independently after SDK
qualification; no File source execution or release completion follows from this
fixture preparation.

# My installed native compiler and runtime closure

I keep this full5.1 prerequisite under task8bbc. This is a design checkpoint,
not installed acceptance. My File companion qualification at65b remains separate;
I change no qualified source, provider or fixture tree here.

## My observed gap

My current Make install copies compiler/VM/emitter binaries and the explicit File
package. It omits the general runtime headers, C sources, standard modules,
module manifests and generator inputs consumed by native compilation. My C seed
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
Existing VM/emitter and explicit File package commands/headers retain their paths.

A checked, committed relative-path inventory selects SDK inputs, including source,
headers, generated schema, module metadata, script/template inputs and module data
actually required by those modules. I generate it from repository-owned inputs,
then require every runtime-provider, header, metadata dependency and script path
used by the actual compiler to appear in it. Make install uses this inventory,
not a recursive copy of a developer's working directory or cache. Release source
archives carry the inventory, so installation does not require Git. A verification
tool rejects absolute paths, traversal, duplicate paths, unexpected symlink inputs,
missing files and input/output overlap before copying.

The generation identity is SHA256 over ordered counted path bytes, lengths and
SHA256 content digests, plus the exact C-seed/Stage1/Stage2 binary bytes. The
manifest records native array ABI2 and every copied byte identity. The manifest
itself is a derived output, excluded from its own digest. This is package identity,
not a signature or an authority grant. Installed files are trusted compiler inputs;
I do not claim protection against a party modifying that trusted installation.

I stage a complete generation privately under the destination parent, verify the
copied inventory, then rename it into its final identity directory without
replacing a different generation. An existing same-identity directory is usable
only after exact inventory verification. Only then do I atomically replace each
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
checkouts retain their current default cache policy. I never create .build inside
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

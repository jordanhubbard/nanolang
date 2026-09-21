# My native provider closure prerequisite for File source preparation

I record task_28cf8f795b2a410d8bd015d2a4545018 under full8bbc before changing
provider selection. My new File compiler imports expose existing native-driver
closure gaps: `nanoisa`, `file_source_catalog` and `file_companion` repeat canonical
C providers; the driver appends basename-only object paths; each provider sees
only the flags accumulated before its module; compile failures warn and continue.
My native runtime also already compiles `utf8.c`, not only the specially excluded
`cJSON.c`. These are static findings. I have not executed a failed build.

## My concrete corrective boundary

I change only the actual Nano native driver's module preparation and its immediate
owned staging cleanup. I retain standalone module manifests and C-seed module
builder behavior. File service graph preparation still returns before this path;
this correction is necessary to build the compiler and ordinary standalone
programs that use its data modules. It adds no File execution, provider spelling
authority or alternate lowering path. Root reviews this plan before production.

I split `collect_module_build_flags` into two ordered phases. First I collect all
module metadata and ordered pkg-config, explicit C flags, include paths, framework
and linker contributions using the existing traversal and precedence. I retain
provider requests as `(canonical source, selected compiler command, language)`.
The default selector is the existing `resolve_cc_binary`; explicit `c_compiler:
"c++"` retains `c++` and C++11. Default providers retain C99. Unsupported nonempty
selectors refuse instead of silently selecting another compiler. Missing source
canonicalization or a missing required metadata/provider fact is a failure.

After metadata collection, every provider sees the documented final whole-program
C flag profile, in its exact existing order. This intentionally replaces the old
accumulated-prefix profile; I do not pretend the old and new command strings are
equal. I neither sort flags nor silently deduplicate repeated `-D`, `-U`, include
or linker options. The generated program uses the same final profile. Provider
identity is the canonical source plus exact selected compiler command, language
standard and ordered effective flag string. Identical repeated requests share one
owner and one link argument. One canonical source requested with conflicting
compiler/language/profile refuses; first/last request never wins. Compiler command
selectors are compared exactly, not treated as equivalent merely because two
PATH spellings might resolve to one executable. Tool/metadata providers are stable
for an invocation; this is not a hostile PATH mutation guarantee.

I factor the existing actual native-runtime source list into one ordered helper
used both to append runtime sources to the main/shadow commands and to initialize
explicit already-linked owners. This preserves the exact existing list/order,
including conditional peg2, cJSON and UTF-8. A module request matching one of those
canonical files must agree with its actual C99/default-compiler final profile;
otherwise it refuses. I do not suppress arbitrary same-basename files or assume
that a symbol with the same spelling is already provided. This removes the old
cJSON-only exception without introducing a second unsynchronized runtime list.
Required runtime includes and existing link flags remain explicit and ordered.

## My actual object lifetime and failure API

`compile_program` already owns a fresh `mktemp_dir("nano_native_")` directory before
module compilation. I pass that directory to preparation and place each unique
provider at `<directory>/provider_<registry-index>.o`. Index identity is safe only
inside this invocation-private directory; I create no reusable object under
shared `obj/nano_modules`. Canonical-source identity, not basename, selects the
registry record. Distinct same-basename sources therefore cannot overwrite each
other, and concurrent invocations/profiles do not share object paths. I shell-quote
owned/source paths using the existing native quote helper; compiler selectors and
metadata flag fragments retain their existing trusted-command precondition.

`ModuleBuildFlags` gains an explicit `ok` status and an array of owned object paths.
I record each path before starting its compile so partial compiler output is
owned too. A required provider's nonzero compiler terminal returns failure; no
warning-only continuation, test executable or final link follows it. The caller
removes those exact objects and then applies the existing generated-C cleanup
policy. All normal, shadow-failure and link-failure exits after preparation invoke
the same provider cleanup. `--keep-c` may retain `program.c` as before, but does not
retain unreported shared provider objects. I preserve the first failure if cleanup
also fails and report failed cleanup paths; I do not recursively delete arbitrary
external paths or claim arbitrary compiler-created side products are captured.
Existing subprocess supervision limitations remain explicit; outer qualification
uses bounded process-group cleanup and retained file-backed terminals.

Provider/preparation failure occurs before final output linking and therefore
preserves a pre-existing output sentinel. I do not broaden that fact into an
atomic-publication claim for every existing native linker failure. The old direct
final-link publication boundary is separate from this provider correction.
The registry and command strings use checked finite counts/byte budgets before
append: at most5000 selected module directories,50000 provider requests and1MiB
per retained command/profile string. Existing configured import limits remain
in force. Nano allocation/runtime failures retain their established limitations;
this plan adds no general transactional allocator retrofit.

## My required qualification before acceptance

I review the full corrected source and then fixtures before execution. I require
fresh actual C-seed/Stage1/Stage2 compiler closure on both hosts; existing default
native/standalone module behavior; the full File graph/data-module helper shadows;
duplicate canonical provider requests compiled/linked once; distinct same-basename
sources preserving both functions; exact ordered flag/probe behavior; conflicting
compiler/language/profile refusal; required compilation failure with a pre-existing
output sentinel and complete known-object cleanup; and concurrent invocations with
distinct private directories and profiles. I retain actual compiler commands and
provider/source/artifact maps. Passing this prerequisite does not complete File
source lowering or any of its required generated behavior shadows/runtime grants.

## My prepared production and storage boundary

I prepare this source without running a compiler or fixture. My single runtime
helper retains the old54 unconditional paths in exact order and the old conditional
peg2 path. Static extraction against be24 confirms that ordered identity. The
same helper supplies actual final commands and canonical registry owners.

I now visit previously unselected manifest dependencies immediately after their
owner, in declared order; already selected directories retain their order. This
intentionally replaces the old dependency-only pkg-config/link subset with full
metadata and required-provider preparation. I append `shared_c_sources` after
`c_sources` and subject both to the same canonical identity checks. Thus shared
UTF-8/cJSON requests agree with actual runtime owners rather than disappearing
because of their spelling. Missing dependency metadata refuses. A selected source
module without a manifest retains the existing no-manifest behavior. My inherited
metadata string extractor is not a new strict JSON validator; manifests and flag
fragments remain trusted, stable compiler inputs.

| Storage or work | Owner, overlap and limit |
| --- | --- |
| Module queue | At most5000 directories; dependency insertion temporarily retains old queue, pending queue and replacement queue. Exact string references/copies follow the existing Nano runtime. |
| Provider requests | At most50000 rows, including repeated and shared requests; retained throughout registry preparation and compilation. |
| Registry | At most50128 rows, covering all requests plus54/55 actual runtime sources. Exact-source scans are bounded by request count times registry count. Each row retains compiler, language and final profile facts. |
| Metadata | Each inherited `file_read` result is checked at1MiB after reading. Decoded arrays, package deduplication, dependency scans and queue rebuilding remain bounded by the selected manifests and their byte/count caps; this is not a preallocation or total-heap guarantee. |
| Strings | Final profiles, object paths, object link arguments and compiler commands have a1MiB checked append limit. Appends temporarily overlap old and new strings; runtime allocator/object headers and inherited path/quote helpers are outside a total-memory claim. |
| Objects | Known paths enter the returned owner list before compiler invocation; successful and partial objects are removed on every subsequent normal/error exit. Private directory ownership prevents cross-invocation registry-index collisions. |
| Failure | `ok=false` and `first_status=1` describe preparation refusal; a measured provider compiler failure retains its actual nonzero status. Cleanup does not replace it. Successful final linking followed by failed object cleanup returns failure with output already present. |

I capture pkg-config output with the existing bounded capture helper, appending a
success marker only after a zero shell status. I require that exact trailing marker
and fewer than65535 captured bytes; legitimate empty flags remain valid. This is
framing for a trusted command, not an adversarial subprocess protocol. Inherited
capture ignores the raw child terminal and has no internal deadline; a missing or
truncated marker refuses, and external qualification retains bounded group cleanup.
I do not claim that this change supervises arbitrary compiler children or recovers
all Nano allocation failures. Exact append-count checks detect failed copies at
those boundaries; inherited constructors, path helpers, metadata reads and arrays
retain their existing allocation behavior. No snapshot64MiB claim applies here.

My cleanup checks known object existence before deletion using the existing file
API. Stable accessible invocation-owned directories and cooperative compiler
behavior remain preconditions; I do not claim hostile filesystem identity safety
or distinguish every possible lookup error from absence. Existing generated-C
cleanup remains separate. Complete producer, standalone, conflict, failure,
concurrency and ordered-profile fixtures still precede acceptance.

I append a canonical runtime source to the final command vector only when its
request inserts a new registry row. Distinct inventory paths resolving to the same
canonical file therefore share the first owner and its original order, just like
module requests. My fixtures must supply an actual canonical runtime alias and
observe one final compiler input; a registry-only duplicate unit control is not
sufficient for this corrected boundary.

## My generated C module dependency headers

The3bc actual public Forth wrapper exposed a distinct C-seed boundary: isolated
module parsing recursively loads its dependencies and transpilation emits their
metadata headers, while the module-object compiler receives only the wrapper's
own directory and flags. I preserve the first missing-nanoisa.h terminals on both
hosts. The source file and actual command are retained; no failed binary ran.

I prepare the include closure from that isolated parser's actual module cache,
while its canonical paths and ASTs remain live. I add each selected dependency's
canonical declaring directory and its existing metadata include_dirs through the
same checked path quoting used by ordinary compilation. I preserve first encounter
order and deduplicate canonical directories. This is the same selected dependency
namespace already used for header collection, not a new resolver, import grammar,
provider owner or spelling-based path injection. I do not merge arbitrary
transitive macros/link flags or compile a second provider copy.

I load metadata through the existing module metadata parser, distinguish absent
module.json from a present malformed/unreadable record, and fail before invoking
the C compiler on required metadata/path/argument construction failure. Ordinary
metadata origin rules remain the authority for its include directories. The
bounded include buffer and final command must account for every appended quoted
argument; overflow is a precise refusal, never silent omission. All temporary
metadata/path ownership ends after copying the checked arguments, before the
isolated AST/cache is released. I preserve the first failure and object-publication
behavior.

I require a nested ordinary module with an actual imported local header, paths
with spaces, reversed independent import order, and compilation from a cwd that
contains none of those headers. The existing complete owner/co-import/wrapper
corpus remains unchanged. Installed SDK qualification repeats this boundary with
SDK-owned and user-project origins. I retain the existing external compiler's
header search rules; this does not claim a general C preprocessor namespace proof.

My C implementation bounds the actual isolated-cache walk and total declaring-
directory/metadata include requests to1024, each input/canonical path to4095bytes,
retained canonical strings to65537bytes, and quoted arguments plus final command
to65536bytes excluding the terminator. Canonical dedup preserves first encounter
order. The owned workspace is one closure containing1024pointers/two size fields
and65537argument bytes, plus at most65537retained path bytes and one transient
quoted path allocation of at most16383bytes. Automatic path/manifest buffers and
the final command/error buffers are separate. Metadata parsing retains its
existing allocation behavior; I do not claim a total parser heap bound or new
recoverable-OOM semantics for its inherited constructors. A returned metadata
failure, invalid requested directory, missing required path, or overflowing
argument refuses before the selected module's C compiler invocation. Earlier
module preparation/shadow effects are not described as absent.

I distinguish absent metadata (ENOENT at its directory entry) from a dangling
link, directory, FIFO or other present unreadable/malformed record. Ordinary
symlinks to regular manifests retain the existing canonical declaring origin.
Stable source/metadata files are the existing compilation precondition; this is
not a hostile concurrent-filesystem protocol. The narrow helper probe covers
these cases, exact request-cap endpoints and quoted-argument capacity without
running a compiler. Its repeated borrowed cache rows are labeled a unit control,
not a real parser graph. The paired integration corpus compiles and executes both
import orders of actual nested wrappers from an unrelated cwd, with spaced
module/include paths and duplicate canonical metadata include requests, retains
actual compiler argv, and checks the exact three selected shadows. The original
complete provider-owner corpus remains mandatory. SDK-owned/user-project origins
are additionally required by my installed SDK fixture checkpoint.

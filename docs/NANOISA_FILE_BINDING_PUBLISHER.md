# I publish an already validated File binding directory exclusively

I continue `task_8bbc1cf5295b4b59b314640ef57c725f` and the open6fc/72556/6931/d03c
parents. My base is actual PR905 merge `af8809b32850454d06d9c1881c3a27b43f4c9d9c`;
its reviewed92726 integration and pure binding seal remain separate. This is a design
checkpoint, not implementation or measured publication. I reuse
`nl_file_binding_prepare` and its immutable output views. I add no second NSI
parser, catalog validator or source renderer.

## My exact surface and storage

I add separate `src/nsi_file_publish.h/.c`, plus `src/nsi_file_binding_main.c`.
The public publisher header includes the existing binding header and defines:

```c
typedef enum {
    NL_FILE_PUBLISH_NONE, NL_FILE_PUBLISH_VALIDATE,
    NL_FILE_PUBLISH_PARENT, NL_FILE_PUBLISH_STAGE,
    NL_FILE_PUBLISH_INTERFACE, NL_FILE_PUBLISH_SOURCE,
    NL_FILE_PUBLISH_STAGE_SYNC, NL_FILE_PUBLISH_RENAME,
    NL_FILE_PUBLISH_PARENT_SYNC, NL_FILE_PUBLISH_CLOSE
} NlFilePublishStage;
typedef struct {
    NlFileBindingStatus status;
    NlFilePublishStage failed_stage;
    int first_errno;
    int cleanup_errno;
    bool published;
    bool durable;
    bool cleanup_pending;
    char staging_name[96];
} NlFileBindingPublishReport;
NlFileBindingStatus nl_file_binding_publish(
    const NlFileBindingPlan *plan, const char *directory,
    NlFileBindingPublishReport *report);
```

A writable report is required and disjoint from all input storage. Null report
returns INVALID without filesystem calls. Otherwise I initialize the entire
report before validation; null plan/path and invalid components return INVALID.
The plan and path remain immutable for the call, with ordinary C lifetime and
valid readable-string preconditions. I accept only a plan from the qualified
constructor, not a fabricated memory object. I retain no caller pointers.

Publisher storage is fixed automatic storage: a4096-byte parent path buffer,
a256-byte final component, a96-byte stage component,16 entropy bytes, fixed
stat/descriptor/ownership records and the report. I make no project heap
allocation. The two borrowed output lengths must each be nonzero and at most
`NL_FILE_BINDING_MAX_BYTES`; both views are checked before filesystem access.
Checked lengths, no concatenated document-dependent paths, and fixed syscall
argument records precede source review. I preserve the plan and its bytes on
all success/failure paths. No shell or recursive cleanup is used.

## My directory and authority preconditions

The destination path is at most4095 bytes excluding its terminator; final
component is1..255 bytes, is neither `.` nor `..`, and contains no slash.
Trailing slash is invalid. Relative single-component destinations use parent
`.`; `/name` uses parent `/`. I accept arbitrary non-NUL filename bytes,
including spaces and shell metacharacters; no UTF-8 promise is needed for paths.
Longer paths/components return LIMIT before filesystem mutation. A filesystem
with a smaller component limit may still return its ordinary checked IO error.

The caller supplies a trusted stable parent and trusted ancestor resolution.
I do not claim sandboxing against a malicious same-user actor replacing names.
I open the parent once with directory, close-on-exec and no-follow flags, fstat
it as a directory, and retain its device/inode identity. Thus an existing
symlink in the final parent component refuses; earlier ancestors are trusted
resolution, not a claim that all ancestors were symlink-free. Thereafter all
creation, identity checks, removal and rename use this directory descriptor.
A later path rename cannot redirect these descriptor-relative operations.
The externally named destination is meaningful only while the caller's stable-
parent precondition holds; I do not infer pathname stability from an inode check.

Competing cooperative publishers of the same final component are supported.
They must not alter another operation's private staging directory or children.
There is no preflight existence check that grants overwrite permission. The
exclusive rename itself decides the winner; existing regular files, empty or
nonempty directories, symlinks and dangling symlinks all remain intact.

## My ordered filesystem transaction

1. Validate the complete plan/views/path and platform capability. Unsupported
   compiled platforms return UNSUPPORTED before creating anything.
2. Open and identify the parent. Obtain16 entropy bytes with checked host
   `getentropy`; a failure is IO before staging. Encode a fixed private prefix
   plus32 hexadecimal digits. Try at most64 exclusive `mkdirat(...,0700)`
   candidates; only EEXIST advances to another candidate. Exhaustion is IO
   with saved EEXIST, not the final-destination EXISTS verdict.
3. Open the newly created directory using `openat` with directory/no-follow/
   close-on-exec flags. Record descriptor and anchored entry identity. Keep its
   descriptor until after rename or rollback. The restrictive directory is
   intentionally retained as0700 after publication; files are0600 subject to
   umask. I do not silently loosen caller permissions.
4. Create only `interface.nsi.json`, then `binding.nano`, relative to the owned
   staging descriptor using exclusive/no-follow/close-on-exec writable opens.
   Track each successful creation immediately, and its device/inode after
   fstat, before any subsequent fallible operation. Write exact counted bytes
   in progress-checked loops, then fsync and close each descriptor once.
   No NUL terminator is written. A zero write is IO/EIO. EINTR retries on
   read/write/fsync permit at most64 EINTR retries per operation; the65th
   interruption is terminal. The counter does not reset after partial progress. Close
   is never retried because its descriptor disposition may be ambiguous.
5. Fsync the staging directory after both file closes succeed. Confirm the
   anchored staging entry is still the recorded directory before rename.
   No previously reported data/write/sync/close error permits publication.
6. Rename relative to the same parent descriptor using Linux
   `renameat2(...,RENAME_NOREPLACE)` or Darwin
   `renameatx_np(...,RENAME_EXCL)`. I provide no replacing-rename, check-then-
   rename, link/unlink, or two-file publication fallback. ENOSYS/ENOTSUP/
   EOPNOTSUPP and unsupported-flag EINVAL return UNSUPPORTED; EEXIST/ENOTEMPTY
   return EXISTS. Other errors are IO with the first errno retained. A syscall
   interface or filesystem limitation is a refusal, not permission to weaken it.
7. On successful rename set `published=true` immediately and irrevocably.
   Fsync the parent and set `durable=true` only when that sync also succeeds.
   Close retained staging and parent descriptors once each. Never unlink the
   final destination after successful rename, including later failures.

I do not retry rename after an error or try to infer commit from a final name
that another publisher may own. The supported local filesystem syscall contract
must give an authoritative success/failure result; uncertain remote filesystem
outcomes are outside this first qualification and are not advertised as atomic
remote publication. Sync success records the OS durability attempt, not a
universal guarantee against hardware or power failure.

## My first error, cleanup and recovery states

Status/failed_stage/first_errno capture the first failure before cleanup.
Logical validation uses errno0; syscall failures retain their actual errno.
Cleanup cannot overwrite the first status or stage. Its first separate failure
sets cleanup_errno, and cleanup continues over every remaining owned resource.
Descriptor close errors are recorded and never retried; tests distinguish a
modeled close-then-error from arbitrary OS close disposition.

Before rename, rollback attempts only the two known children and the owned
staging directory. Before unlinking a name I compare no-follow anchored entry
identity with the recorded owned inode. Missing names count as already absent;
a mismatched name or unavailable identity is left untouched and reported as
cleanup pending, with ESTALE or the actual identity error. Open descriptors
are still closed once. I do not delete an unidentified directory merely because
its random name was selected by this operation. This explicitly includes a
failure between mkdir/open/fstat or open-file/fstat: I retain the stage name and
refuse unsafe deletion if ownership cannot be re-established. The fixture must
cover these narrow partial states, not only ordinary write failures.

`staging_name` records the chosen name once mkdir succeeds. `cleanup_pending`
means an owned/unresolved staging entry or descriptor cleanup could not be
confirmed complete. It is false on clean rollback and success. After commit,
staging_name is cleared and no automatic retry or rollback is permitted.
A postcommit parent fsync failure returns IO with published=true,durable=false;
a subsequent close failure after all syncs returns IO with published=true,
durable=true. Cleanup errno retains a later separate failure. CLI failure does
not imply absence: the report must be consulted before a caller retries.

A new invocation against an already published destination returns EXISTS and
preserves it. I provide no automatic scavenging command in this slice. Process
termination may leave private staging; its bounded name is diagnostic evidence,
not authority for recursive deletion. Arbitrary SIGKILL/crash recovery remains
an explicit limitation; supervised failure tests may show only measured cleanup.

## My explicit command and packaging boundary

I build `bin/nsi-file-binding` only through explicit Make target
`nsi-file-binding`, preserving caller CC/CPPFLAGS/CFLAGS/LDFLAGS and ordinary
C99 compatibility with owning-TU platform feature macros. It links only this
publisher, strict binding/descriptor/shared NSI, cJSON and UTF-8 providers.
It does not join default compiler/provider/package/install/uninstall lists yet.
I invoke exactly:

```text
nsi-file-binding INPUT --file-binding-dir DIRECTORY
```

Only a sole `--help` is another accepted form. Wrong arity, duplicate/unknown
options and stdout-output modes refuse before opening anything. A positional
INPUT beginning with `-` is still a literal filename in the exact three-argument
form; no stdin alias exists. DIRECTORY is validated before any staging mutation.
The command emits a fixed machine-readable JSON report to stderr, never rendered
source to stdout. Path/staging bytes use a JSON byte representation: ASCII printable bytes retain
their spelling except quoted escapes, and every control/non-ASCII byte is
`\u00XX`. A consumer recovers bytes from code points0..255, not by UTF-8
encoding them. Thus arbitrary filename bytes never emit invalid UTF-8. First status/errno,
published/durable/cleanup_pending are present. Exit0 means statusOK; exit2 is
usage, and exit1 is any preparation/publication/IO failure. No exit code encodes
absence or authorizes an automatic overwrite retry.

Input uses one no-follow/close-on-exec/nonblocking open, requires a regular file,
and reads at most1MiB+1 into a single bounded owned buffer. Nonblocking open
avoids hanging on an attacker-supplied FIFO before fstat refusal. I reject the
extra byte as LIMIT, check read errors and close exactly once, then prepare from
that retained byte span and free the input buffer before publication. A failed
input close prevents publication. Concurrent writers are outside a coherent
snapshot guarantee: I validate exactly the retained bytes, never reopen the
path or claim read() produced one atomic filesystem version. Input symlinks,
directories and special files refuse without staging; original files remain.

I account for that buffer plus `nl_file_binding_allocation_bound`, fixed command
state and publisher stack explicitly. The input buffer overlaps preparation but
not publisher execution. Allocation failures free the buffer/plan and emit no
output directory. Report-write failure cannot undo a committed publication;
CLI returns failure while the publisher report's commit facts remain unchanged.

## My review and acceptance order

1. Review this contract and platform/status/storage inventory; then review the
   complete publisher and CLI production diff before preparing execution gates.
2. Review fixtures observing every attempted open/read/write/sync/close/rename,
   descriptor identity and partial ownership; no counter of successes stands
   for attempted effects. Inject first/secondary failures at every stage, short
   writes/EINTR/zero progress, identity mismatch and allocation failure. Keep
   unexpected first terminals before corrections. Model ambiguous close errors
   accurately and assert no close retry or unrelated descriptor closure.
3. On Linux and puck, publish actual directories and compare both complete files
   to qualified plan bytes. Test existing file/empty-directory/nonempty-directory/
   symlink sentinels, arbitrary path bytes, parent symlink refusal, renamed parent
   anchoring under the stated precondition boundary, and competing real processes
   with exactly one winner. Verify precommit rollback, postcommit sync/close
   reports, retry EXISTS, no unknown-child deletion and retained cleanup errors.
4. Run actual explicit Make-built CLI and C API ordinary/supported sanitizers,
   bounds and strict decoder/renderer/legacy neighbors with fresh identities and
   artifacts. Generated forward shadows still do not execute. Installed paired
   source imports, schema/AST/lowering/selected shadows, fresh grants, startup,
   cyclic/indirect/richer-borrow source and complete platform/product requirements
   remain mandatory later stages under the original parents.

## My first production checkpoint ownership table

I implement the approved standalone publisher and explicit command only. The
strict decoder/renderer/catalog files and existing default/installed surfaces
are unchanged. Both filesystem paths and stderr JSON byte representation are
bounded: the CLI refuses input/destination path strings above4095 bytes before
opening, representing refused oversized paths as JSON null. Redundant trailing
parent separators are trimmed before the no-follow directory open, so a trailing
slash cannot accidentally bypass that final-component check.

| Owner / storage | Acquisition and release / failure rule |
| --- | --- |
| caller plan and two counted views | Borrowed through publication; never modified or freed by publisher. |
| publisher parent/final buffers |4096+256 automatic bytes; checked before filesystem calls. |
| `FpTransaction` | Automatic owning state: two directory descriptors, two child descriptors and exact stat identities/creation flags. It starts with every descriptor -1. |
| staging selection helper | Automatic96-byte candidate and16-byte entropy; at most64 mkdir attempts. Report owns96 diagnostic bytes; no heap allocation. |
| stat/write/sync helpers | Fixed stat/scalar automatic scratch, no recursion, no document-sized automatic arrays or allocations. |
| parent directory fd | One open and fstat; anchored operations until one terminal close. |
| staging directory fd/name | Created flag recorded immediately after mkdir; identity only after fstat. Unknown identity survives cleanup as pending, never guessed-owned. |
| each file fd/name | Created flag recorded immediately after exclusive open; identity after fstat. Exact write/fsync/close, or close-once plus checked known-child unlink. |
| CLI input buffer | Exactly1MiB+1 requested bytes after regular-file fstat; freed after prepare and before publisher. Read never requests beyond remaining capacity. |
| strict binding preparation | Existing queried conservative bound plus input buffer is checked against16MiB before opening. CLI retains one returned plan through publication, then frees it. |
| output report/JSON | Fixed report; path byte escaping streams bounded input names to stderr. Reporting failure returns nonzero and does not undo commit. |

The exact publisher automatic object sizes are owning-TU `sizeof(FpTransaction)`
plus the listed buffers, stat/scalar scratch and report supplied by the caller;
compiler stack-frame layout/ABI overhead is not a claimed heap allocation or
portable byte-exact stack guarantee. There is no allocation after publication
begins. Platform/libc internals remain outside project-requested heap accounting.

I capture first errors before rollback and preserve later cleanup errors. A file
close failure is a first CLOSE failure; a rollback close failure is secondary.
The stage and parent are closed even after an unknown-child identity prevents
unlink, and no descriptor slot can be closed twice. Parent sync success precedes
durable=true; later close errors preserve that fact and published=true.
No new fixture, compiler build, syscall publication or generated shadow has run
at this source checkpoint. Full source review precedes fixture preparation.

I isolate all seven explicit-tool objects under `obj/file-binding-publisher`
(the configured OBJ_DIR remains authoritative). Its scoped pattern recipe
passes CC/CPPFLAGS/CFLAGS and its link passes those plus LDFLAGS; existing generic
provider recipes are unchanged. Each qualification configuration must use a
fresh object directory/tree or remove only this owned directory before rebuilding,
because ordinary Make timestamps do not encode compiler/flag changes. Header
dependencies cover the complete seven-provider closure. Fixture review must retain
actual compile commands demonstrating CPPFLAGS at compile time, not only link time.
My read/write/fsync tests distinguish64 interrupted retries followed by progress
from65 interruptions causing failure; close-once remains unchanged.

## My fixture checkpoint before execution

I prepare one C API fixture in linked and included-production instrumented forms.
The latter wraps actual open/openat/mkdir/stat/write/sync/close/entropy/rename/
unlink operations, keeps live descriptor identities, and sweeps every operation
of a successful publication with one injected first failure. Each retained line
identifies the failed operation, stage, commit/durability and cleanup state.
I also fail each cleanup unlink/close ordinal, combine postcommit sync failure
with secondary close failure, preserve unknown child/inode states, exercise the
64-collision budget, and temporarily rename the parent to observe descriptor
anchoring (without claiming the original pathname stays stable). Short writes,
zero write, exact64/65 interruption boundaries and unchanged plan bytes remain
assertions. The close hook actually closes before injecting its error, explicitly
modeling reporting failure rather than arbitrary OS close disposition.

The same instrumented translation unit invokes the actual CLI main with checked
input open/stat/malloc/read/close hooks, first-operation failures, short reads and
64/65 read interruptions. The decoder/renderer providers remain freshly linked;
the existing strict suite separately supplies their complete allocation sweeps.
No fixture changes the publisher's production source or conditionally admits a
new operation. Fixture cleanup owns its isolated test directory, uses only known
fixture children, and does not turn production unknown-identity cleanup into an
unreported success.

The Python runner builds all seven providers through the actual explicit Make
recipe into a fresh per-configuration directory. A header forced by CFLAGS
requires a macro supplied only through CPPFLAGS, proving compile-time forwarding.
Linked tests use those objects; instrumented tests include publisher/CLI source
and link the same five strict providers. Strict warnings remain errors. Each
configuration also reruns the unchanged strict binding and legacy neighbors.

The actual CLI subprocess corpus compares both complete output files with
independent goldens, permissions, repeat EXISTS, existing regular/empty/nonempty/
symlink/dangling sentinels, parent symlink refusal, regular input restrictions,
byte-valued filenames and JSON byte recovery, oversized inputs, usage and eight
concurrent publishers with exactly one winner. A read-only stderr descriptor
checks nonzero reporting failure after successful publication without removing
the result. Every subprocess has durable file-backed output/status; children
inherit the outer retained process group for bounded terminal cleanup. These
are proposed controls, not measured results. All new fixtures require review
before any compilation or filesystem publication.

My planned frozen matrix is Linux GCC/Clang ordinary and strict ASan/UBSan, plus
puck Apple/Homebrew ordinary and Homebrew strict ASan/UBSan. Each uses explicit
selected tools, fresh scoped Make/provider objects, source/tool before/after
maps, actual binaries and first terminals. Leak detection stays enabled with no
inherited LSAN suppression. This does not qualify installed compiler routing,
generated shadows, service execution or broader source acceptance.

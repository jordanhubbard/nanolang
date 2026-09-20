# My private affine-state variant storage

I preserve task_0924394710ff4dbc9a762b26b1eebf34 and both first integrated51d mixed-admission failures. Both hosts returned Make status2 when the unchanged admission fixture received SIGSEGV; no assertion identified a phase. My one root-authorized bounded Linux debugger run then located the crash in `nvm_affine_state_clone`, copying `s->variants`, from `mc_frame_clone` during the initial `nvm_verify` at fixture main line9. This is before its admission-allocation fault switch is enabled. The raw debugger log and binary/provider before/after hashes are retained at `/tmp/nanolang-service-mixed-51d-debug`; its shell sequence did not separately preserve GDB's process exit code, so I use the recorded signal/backtrace rather than claiming a diagnostic pass. No full gate was replayed.

Canonical cfd73dfdc7 added variant storage to the ordinary affine constructor, clone and destructor. My mixed `mc_checked_state` and owner ARRAY `la_checked_state` constructors still allocate only locals/live/reference storage. Their zeroed state therefore has no variants allocation although the shared clone now requires it. The mixed path is observed; the analogous owner ARRAY gap is established by source inspection and has not been separately executed. This is a constructor invariant defect, not an allocator-hook failure or a measured infrastructure/ABI problem.

Before code I audit every state allocation site in src: the public constructor and clone are in affine_state.c; the only other constructors are those two private includes in the same owning translation unit. There is no state-resize/realloc path in that family. The public constructor already allocates at least one variant cell and initializes each declared local to NVM_AFFINE_UNKNOWN_VARIANT. Clone allocates its own capacity and copies declared cells; destructor frees variants independently of other partial allocations. State equality/meet and union define/refine/move use that storage under checked local counts; no other pointer replacement or storage transfer was found.

I repair only the two private constructors: allocate at least one uint16_t variant cell using their already bounded local count, include that allocation in the existing failure cleanup, and initialize every declared local to NVM_AFFINE_UNKNOWN_VARIANT before returning. I do not admit union values to mixed or owner ARRAY profiles, change verifier authority, alter clone behavior to tolerate missing storage, weaken assertions, or skip allocation failures. The existing destructor releases the new allocation on every partial-construction failure. Local counts are uint16_t, so the capacity multiplication is representable on supported size_t targets.

I submit the tiny production diff for independent review before fixtures/runs. Acceptance must cover construction, clone independence/unknown initialization, cleanup and failure at the new allocation for both private paths, alongside normal constructor/clone and existing mixed/owner authority controls. Fresh corrected gates retain the first failures and original successful51d/1f9 phases, with actual changed affine-provider attribution. Full roadmap/Darwin timeout work remains separate.

## My focused fixture boundary

Before writing fixtures, I select direct constructor controls in one test-only
translation unit that includes the unchanged affine implementation. I construct
ordinary scalar-local metadata with one resource layout, obtain real copied mixed
and owner-layout descriptions, and call the two private constructors with those
inputs. These are constructor invariant controls, not independently admitted
mixed/ARRAY programs. Existing complete mixed and owner authority/admission
fixtures remain separate acceptance.

I check zero, three and 256 locals, every UNKNOWN variant cell, clone storage
independence, survival after the original state is freed, and the public
constructor as an adjacent control. Test-only malloc storage is poisoned before
production initialization. I instrument the affine TU and layout decoder, count
retained allocations, and sweep every observed constructor and clone allocation
with permanent and single-failure budgets. Every failure must preserve an existing
state, release partial storage, and permit fresh recovery. I identify the variant
allocation by its returned pointer on the successful path, then require its exact
index to fail during both sweeps. Other providers remain ordinary; sanitizers
cover these two rebuilt TUs and the fixture, not the complete provider closure.
I submit the fixture and retained runner before executing any new gate.

Independent fixture review accepted c3875585b before execution and found one
Make forwarding omission: my target passed the compiler and link inputs but not
`AFFINE_VARIANTS_CFLAGS`. I record and correct that fixture-only omission before
qualification, forwarding the caller's exact CFLAGS without altering assertions
or production. No c387 gate ran. I now qualify fresh constructor and existing
service-admission controls with explicit ordinary/sanitizer compiler selections,
then complete actual mixed/owner/shared-authority neighbors and the unchanged
imported-emitter deadline. Original51d failures remain frozen.

My first660 Linux preparation passed, but the constructor link failed before any
fixture execution: the external driver supplied the larger VM/compiler closure,
whose eval/CLI objects require g_argc/g_argv. I retain command/status/log and input
endpoints at `/tmp/nanolang-affine-variants-660-linux`. This is a demonstrated
external object-selection error; the reviewed Make target is already correct.
Before continuation I make the external Make fragment print that target's exact
NANOISA_OBJECTS minus affine_state/nvm_v2_layouts plus NANOISA_UTF8. I retain the
larger closure for the VM fixture only. No production or fixture changes, no
failed binary replay, and no claim that the original link passed. I verify the
frozen660 selected provider identities before corrected commands. Darwin had no
fixture command before this correction.

The corrected660 five-configuration constructor/admission matrix passes. Both
hosts pass actual mixed admission and complete mixed query; Linux also passes
owner authority. Darwin's old owner-authority runner then fails before execution
because its hardcoded `-lm -lcrypto` drops the prepared OpenSSL search directory.
I retain `/tmp/nanolang-affine-variants-660-puck-neighbors/owner-authority.log` and
its Make2 status. Before correction I specify an optional
`OWNED_ARRAY_AUTHORITY_LDFLAGS` override, defaulting to the exact old flags, and
Make forwarding of its actual LDFLAGS. This changes build selection only; I keep
all allocation, status and output assertions. Corrected Darwin continuation gets
separate fixture/input attribution and retained products. No Linux replay is
needed for this link-selection correction.

## My canonical914/915 integration

I integrate actual main99f390264b7e860ffad659a3852c9b66cbd73399 in a separate
ready tree after the sealed660/033 acceptance. The sole textual conflict is the
additive roadmap tail; I retain both histories. My constructor/admission
production and fixtures retain their qualified bytes. Incoming914 adds explicit
public cyclic File engines, opt-in CLI branches, package providers and header
compatibility changes. Ordinary CLI selection remains the existing branch.
Incoming915 changes the assembler-capture Make recipe to forward real caller
flags, with its documented separate external-assembler sanitizer boundary; I use
that actual canonical recipe, not an older copied recipe.

I prepare fresh providers and CLIs on both hosts, run ordinary constructor and
switch/computed-goto admission controls, actual callback/FFI/mixed/owner/shared
neighbors, public cyclic package controls and the unchanged full imported emitter.
I retain earlier sanitizer acceptance at660; unchanged source identity does not
relabel it as integration-pin sanitizer acceptance. Full-suite qualification waits
for the separately reviewed projected/tuple native-compiler-selector correction;
this scoped integration neither closes nor weakens that requirement.

Before any52b gate, I also integrate actual916 main7bba8c798eeb19027d596539a9c86f4db6cb0840.
The additive Make/roadmap conflicts retain both targets and histories. This changes
shared ownership/layout provider code: existing public calls pass NULL projection,
and the old public/private layout routes pass false for mixed unions. The new
complete declaration query is separate and non-admitting. I preserve my qualified
constructor/admission source and fixture bytes, build these current providers
fresh, and add the actual declaration-projection target to ordinary neighbors.
The unused52b checkouts contain no qualification runs and are not evidence of a
pass. Public package checks use a separate integration tree because their actual
Make install deliberately rebuilds providers; my ordinary gate tree stays frozen.

Both bc45 ordinary integrations pass, including the new declaration query and
original emitter. Separate public-package setup then stops before fixtures on
both hosts: my external environment omitted NMS_RUNTIME_CLANG/NMS_RUNTIME_OPT and
the known Linux NMS_NATIVE_CLANG_FLAGS GCC13 selection. Linux's managed IR builder
therefore reports the strict Clang GCC14-selection warning; Darwin reports missing
opt. I retain both Make2 setup logs and source/tool/product maps at
`/tmp/nanolang-admission-public-bc45-{linux,puck}`. I correct launch configuration
only, explicitly inventory the selected optimizer, and continue separate public
trees. There is no product/fixture change or ordinary-gate repetition. I record
that some setup products already exist and may be completed/rebuilt; only source
and tool immutability is claimed during these actual setup/install phases.

My first corrected Darwin launch then fails in external preflight before any
command: SSH shell quoting removed the JSON quotes around CARRIER_EXTRA_TOOLS.
I retain its exact environment and JSONDecodeError, with observed SSH status1,
at the corrected-puck report/log paths. Linux's corrected setup is unaffected.
I replace shell construction with a retained Python launcher that uses
json.dumps and an explicit environment mapping. This changes no source, fixture,
tool selection or deadline; it makes the already reviewed configuration exact.

My final review catches unrelated Make comment damage from my merge-marker
cleanup: replacing the separator substring also shortened separator comments and
joined following comment lines. I restore those exact comment-only hunks from
canonical7bba, verifying every noncomment/nonblank line remains identical to the
qualified bc45 recipe. No source, target, flags or command order changes. I keep
all frozen bc45 files and their original hashes; this ready-tree comment-only
correction does not relabel qualification and does not justify a runtime replay.

Linux's corrected public two-method corpus/package passes. Darwin's selected
setup passes, but its first instrumented method stops while compiling vm_ffi.c:
ffi.h is absent from the compiler's default header search. The external launch
had left NANO_FILE_RUNTIME_CFLAGS empty instead of forwarding actual Make CFLAGS.
I retain that selected-puck method terminal and its partial provider products.
Before continuation I set the selector from the already retained frozenbc45
Make-emitted cflags.txt, including its exact Xcode26.2 SDK ffi and Homebrew OpenSSL
include paths, with all strict warning flags intact. No production/fixture edit
is needed. I rerun only read-only configuration/discovery and the Darwin public
corpus; its completed setup and Linux complete pass remain attributed separately.

After bc45 qualification and before final publication I integrate actual main918
merge367476f0caa83b847d1856b248a70289f80a3aa9. Its only non-document changes are the
two reviewed native test compiler selectors in flat-record and tuple fixtures.
They are exact3ef bytes, not part of the earlier ordinary/package gate selection.
No production, provider recipe or qualified fixture changes; I retain the bc45
attribution. Fresh complete90-method canonical acceptance remains required after
actual merge, with supported selected native compilers and unchanged deadlines.

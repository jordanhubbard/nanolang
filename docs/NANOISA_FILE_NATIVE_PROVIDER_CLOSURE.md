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

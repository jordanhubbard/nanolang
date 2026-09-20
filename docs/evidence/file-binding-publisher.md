# I qualify explicit exclusive File binding publication

I qualify the standalone publisher/API and explicit `nsi-file-binding` command.
I do not install it into compiler routing, execute generated source/shadows or
run File services. Task8bbc and its original source/5.1 parents remain open.

| Host/compiler | Frozen pin | Ordinary | Strict ASan/UBSan with leak detection |
| --- | --- | --- | --- |
| Linux GCC | `a22cc0c0a` | PASS | PASS |
| Linux Clang | `a22cc0c0a` | PASS | PASS |
| puck Apple Clang | `4e828f288` | PASS | not requested/unsupported configuration |
| puck Homebrew Clang | `4e828f288` | PASS | PASS |

Each configuration builds all seven providers through the actual explicit Make
recipe into fresh scoped objects. A header forced by CFLAGS requires a macro
supplied only by CPPFLAGS, establishing compile-time forwarding. The linked API
uses those objects. The instrumented fixture includes the actual publisher and
CLI source with syscall/input-allocation hooks and links the same five strict
providers. Sanitizers cover this newly built closure, fixtures and unchanged
strict/legacy neighbors; I do not claim complete libc/toolchain instrumentation.
I clear LSAN_OPTIONS and retain detect_leaks=1 and strict warning flags.

Every linked API run reports43 checks. Instrumented runs report3,447 on Linux
and3,422 on Darwin; the Linux syscall adapter additionally checks its syscall
number. Full first-operation failure sweeps retain the actual operation/stage,
first/secondary errno, published/durable and cleanup state. All tracked
successfully opened descriptors close once, including modeled close-then-error
injection. This model does not prove arbitrary libc close failure disposition.
Partial identity failures retain unknown staging rather than deleting it;
secondary cleanup failures, unknown children,64 stage collisions, renamed-parent
anchoring, short/zero progress, exact64/65 interruptions and interleaved progress
remain checked. Descriptor anchoring does not promise pathname stability after
the trusted-parent precondition is deliberately disturbed by the fixture.

Actual CLI runs report27 JSON outcomes per original Linux configuration and28
per corrected puck configuration, plus separately retained report-output failure
status. Eight competing processes produce exactly one published winner; every
loser reports EXISTS. Both complete files match independent canonical JSON and
forward-source goldens. Existing regular/empty/nonempty/symlink/dangling targets,
parent symlinks (including redundant separators), regular-file input restrictions,
existing-target/dangling input symlinks, permissions, maximum255-byte components,
usage, oversized inputs and byte diagnostics remain checked. Reporting to a
read-only stderr descriptor fails after commit without deleting the result.

Each configuration separately runs the unchanged strict binding suite:704 cases,
26,881,859 instrumented checks,7,855 linked checks, complete canonical goldens,
all allocation prefix/transient outcomes and NSI/generator/File-plan neighbors.
No valid injected-failure plan unexpectedly recovers. I retain each actual
status rather than labeling malformed decoder outcomes as precise MEMORY.

## I preserve both first terminals and exact correction scopes

- Original puck `a22cc0c0a` fails in actual Make before fixture execution because
  getentropy lacks its Darwin header. Read-only SDK inspection locates the
  declaration in sys/random.h. After precode3b70 and review, `5fff9b63d` adds only
  Apple-specific includes to production/fixture. Linux's preprocessor branch
  and passing a22 matrix remain separately retained.
- Puck `5fff9b63d` builds successfully and reaches the CLI raw-byte pathname
  assertion. Rename reports IO/EILSEQ(92), published=false,durable=false and
  clean rollback, with exact escaped filename bytes. The fixture incorrectly
  expects universal filesystem acceptance. After precodec1d and review,
  `4e828f288` keeps a valid-UTF8 quoted/metacharacter positive and probes mkdir
  of the exact invalid-byte component in the same parent. All three fresh puck
  probes report EILSEQ92; publication must return the same exact RENAME/EILSEQ,
  preserve JSON byte identity and leave no final/staging entry. No arbitrary IO
  is accepted. Original Linux raw-byte publication succeeds and remains at a22;
  its logs are not relabeled as executions of the later probe/UTF8 assertions.

Final production differs from original reviewed74eb only by the Apple header;
4e828 changes fixture/documentation only. All24 source/tool before/after pairs
match, including failed phases. Each map covers16,218 tracked source files.
Current read-only checks match all four retained trees: Linux and original puck
with12 selected tools each; header-corrected and final puck with13 each, including
sys/random.h. Original prephase maps are unchanged. Tool scope includes explicit
compilers, owning Apple driver/SDK selection and selected dependencies, not a
complete transitive system toolchain inventory.

## I retain the seal and its boundaries

[The report manifest](file-binding-publisher/report-sha256.json) covers1,944
reports. [The artifact index](file-binding-publisher/artifact-index.json) records
1,298 unique objects (1,376,171,921 bytes), with6,095 references in
`/tmp/nanolang-file-publisher-artifacts`. These retain actual Make/fixture/CLI
binaries, objects, generated corpus/output bytes, source archives and outer
drivers. File-backed command/status reports preserve normal/failed terminals
and bounded child/process-group cleanup. No passing gate was rerun merely to
manufacture a missing prephase inventory.

The retained Linux root is
`/home/jkh/Src/nanolang-file-publisher-qualified-a22`, reports
`/tmp/nanolang-file-publisher-a22-linux`. Configured `ssh puck.local` reaches:

- `/tmp/nanolang-file-publisher-qualified-a22`, reports `/tmp/nanolang-file-publisher-a22-puck`.
- `/tmp/nanolang-file-publisher-corrected-5fff`, reports `/tmp/nanolang-file-publisher-5fff-puck`.
- `/tmp/nanolang-file-publisher-corrected-4e828`, reports `/tmp/nanolang-file-publisher-4e828-puck`.

I retain no arbitrary-crash recovery, hostile same-user isolation, remote
filesystem atomicity or universal power-loss guarantee. The tested directory
transaction and OS durability attempts have the contract's trusted-parent/local-
filesystem scope. Independent seal review and actual merge remain required;
paired parser/schema/lowering, all selected generated shadows, installed source
consumers and full File runtime/source acceptance are still mandatory.

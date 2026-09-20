# I qualify strict File binding bytes and rendered text

I retain production `675c44a91` unchanged through this checkpoint. I qualify my
strict counted-byte NSI ingestion and pure immutable renderer, not directory
publication, source parsing, generated-shadow execution, or service execution.
My File/source parents remain open.

| Host and compiler | Fixture pin | Ordinary | ASan/UBSan with leak detection |
| --- | --- | --- | --- |
| Linux GCC | `23bedeb11` | PASS | PASS |
| Linux Clang | `23bedeb11` | PASS | PASS |
| puck Apple Clang | `23bedeb11` | PASS | unsupported configuration not requested |
| puck Homebrew Clang | `bc7debbff` | PASS | PASS |

Each configuration freshly builds all five providers in linked and allocation-
instrumented forms, then runs704 exact cases and unchanged NSI, generator, and
File descriptor neighbors. Each instrumented run reports26,881,859 checks;
linked runs report7,854 checks, or7,855 after the additional linked-only assertion.
I compare complete canonical JSON against independent Python data and complete
forward source against the hand-written golden. Generated source stays text.

I retain every allocation prefix and transient result. For the valid catalog,
1,557 allocation positions yield3,114 refusals; none produces a recovered valid
plan. Each of three additional generic NSI shapes yields1,562 refusals. Every
attempt is followed by an unfaulted recovery check. Targeted first/second
`parse_named` strdup failures retain their caller's tree and leak no partial copy.
Observed allocation totals distinguish malloc, calloc and strdup; realloc hooks
are present but these runs observe zero realloc calls. I do not claim realloc
failure coverage. The reported conservative project heap bound is7,575,764 bytes;
fixture/libc internal allocations and allocator overhead are excluded.

I preserve the first puck Homebrew terminal: `23bedeb11` rejects the linked
fixture's set-but-unused `fail_hits` under strict warnings. Apple ordinary and
all Linux configurations passed independently at that pin. After roadmap/MAC
record `ea6558d7c`, root reviewed `bc7debbff`: only a linked-mode
`CHECK(fail_hits==0)` was added. Fresh Homebrew ordinary/sanitizer gates and a
new actual Make provider closure passed in a separate tree. I neither suppress
the warning nor relabel the original failure. No production correction occurred.

My explicit Make closure builds with the selected ordinary compiler. The test
runner separately compiles fresh providers for each selected sanitizer mode;
Make timestamps do not stand in for sanitizer identity. Sanitizer scope includes
the five selected providers, fixture and freshly compiled neighbor code, not the
complete toolchain or all project code. I clear `LSAN_OPTIONS`, enable leak
detection, and retain bounded file-backed stdout/stderr and process-group status.
All22 source/tool before/after pairs match, including the failed compile phase.
Each source map covers13,523 tracked files. Fresh current checks match all three
retained source trees and12 selected tools per tree; this is not a complete
transitive SDK/toolchain inventory.

I retain905 reports under [the report manifest](file-binding-plan/report-sha256.json)
and923 unique artifact objects (748,183,932 bytes), with6,100 references under
[the artifact index](file-binding-plan/artifact-index.json). The object store is
`/tmp/nanolang-file-binding-artifacts`; it includes binaries, objects, exact corpus
inputs, generated headers/text, both frozen source archives and outer drivers.
Reports contain original commands, statuses, fault results, provider maps and
source/tool hashes. A linked fixture's zero hook counters do not measure libc
allocations; instrumented assertions establish the stated tracked domain.

My retained roots are:

- Linux: `/home/jkh/Src/nanolang-file-binding-qualified-23bed` and `/tmp/nanolang-file-binding-23bed-linux`.
- puck over configured `ssh puck.local`: `/tmp/nanolang-file-binding-qualified-23bed` with `/tmp/nanolang-file-binding-23bed-puck`.
- Corrected puck: `/tmp/nanolang-file-binding-corrected-bc7de` with `/tmp/nanolang-file-binding-bc7de-puck`.

I require independent seal review and actual merge before recording this bounded
milestone as delivered. Atomic publication and paired parser/lowering/shadows,
installed source acceptance and broader control-flow obligations remain required.

I prepare separate canonical integration with actual main
`36fb22d9114a83748a5919c27cd12a5f7ffb0f6b`. Incoming private cyclic carrier,
indirect-flow and portable-read code does not enter this five-provider fixture
closure. All23 recorded production/header/fixture/neighbor inputs match the
qualified seal exactly, and all15 incoming production paths match canonical
main exactly. Make/roadmap conflicts retain both additive blocks; strict binding
recipes and common compilation rules do not change. My [integration identity
record](file-binding-plan-integration.json) does not relabel original tests as
new integrated execution. No repeated qualification is justified by this
unchanged closure. Root reviews the integration before merge.

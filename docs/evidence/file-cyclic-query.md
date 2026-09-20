# I qualify cyclic File query facts without execution

I qualify task_243a9a5809bf422caab0bddafa447098 under72556/6931 and
[my reviewed cyclic query contract](../NANOISA_FILE_CYCLIC_QUERY.md).
Production is `f6eeecee3e8f8f88b795bed83dc38916d3e5f198`; corrected fixture
`04d58ca3a4514ea1ed3c71a469c9784224bd67c5` changes only one local identifier
from the first frozen486ba fixture, plus its ledger text. No selector, hosted
plan or VM/native target consumes this new report. Its runtime_admitted field
remains false. I execute query fixtures, never pending cyclic File bytecode.

## I retain the measured platform matrix

| Phase | Linux seconds | puck seconds |
| --- | ---: | ---: |
| Fresh common-provider setup |20.814 PASS|9.891 PASS|
| Configuration capture |0.114 PASS|0.127 PASS|
| Ordinary query, GCC / Apple Clang respectively |2.269 PASS|2.431 PASS|
| Ordinary query, Clang / Homebrew Clang |2.770 PASS|2.380 PASS|
| GCC / Homebrew Clang ASan+UBSan+LSan query |8.485 PASS|4.838 PASS|
| Linux Clang ASan+UBSan+LSan query |6.583 PASS|Not selected|
| Unchanged CODE query neighbor |5.978 PASS|4.334 PASS|
| Unchanged body query neighbor |6.079 PASS|4.238 PASS|
| Unchanged flow query neighbor |5.477 PASS|4.141 PASS|
| Unchanged hosted query neighbor |20.066 PASS|9.810 PASS|
| Unchanged opcode/public-refusal neighbor |3.222 PASS|3.963 PASS|

Every query phase runs the instrumented and separately linked methods. Each
instrumented method reports124,339 checks, including47 allocation prefixes and47
single transient allocation failures with recovery. Each linked method reports
108,837 checks. These totals include repeated state/accessor/allocator assertions;
they are not independent source programs. The complete inherited body fixture
runs first and reports9,508 instrumented/3,663 linked checks as part of the totals.

The fixtures inspect exact canonical owner/reference relations through backedges,
zero-iteration initialization alternatives, owner replacement, held-reference
refusal, owner-empty repeated drops, lower-index callees, nested SCCs, all six
decoded transfers in a two-owner swap and real16/17-alternative joins. Complete
report states/edges are compared across fresh analyses and remain usable after
original CODE/ownership storage is destroyed. Old DAG refusals and acyclic body
facts remain checked; independent old flow histories are unchanged. White-box
canonicalization, storage and otherwise unreachable counter maxima are explicitly
labeled controls, not maximal wire-program acceptance.

Sanitizer phases retain detect_leaks=1 and strict warnings. They instrument the
new query and nominal preparation when rebuilt directly/included by the fixture,
plus the fixture itself. Shared ordinary NanoISA/compiler/VM providers remain
ordinary objects with recorded hashes. This is not whole-program sanitizer
coverage. The linked method uses separately compiled production query providers;
its allocations are outside the fixture's injected allocator counter.

## I preserve the first terminal

The first frozen486ba Linux setup passes21.065 seconds and configuration0.064.
Its ordinary fixture compile then fails0.265 seconds: nested_cycles local `exit`
shadows libc exit invoked by CHECK. No fixture binary executes. Raw compiler
stderr, status and all source/tool/input/artifact maps are retained under
`/tmp/nanolang-file-cyclic-486ba-linux` and `/tmp/nano-file-cyclic-3u3k0oad`.
Task_1f0d53da650d42aba327852e74a9f3f7 records the bounded fixture correction.
The approved rename to `outer_exit` preserves every program/assertion and all
production. Corrected Linux uses an entirely fresh setup, not reused old objects;
Darwin starts only on the corrected fixture. Historical output is not replayed.

Earlier runner concerns were static review findings at1ffa, corrected at486ba
before the first run. File-backed raw logs, launch/wait errors, bounded process
group TERM/KILL cleanup, leader reaping and group disappearance are retained.
This qualification encounters no timeout or remaining process group. It does not
claim these normal terminals exercise every OS timeout failure mode.

## I retain exact source, provider and artifact identities

- Linux frozen trees: `/home/jkh/Src/nanolang-file-cyclic-486ba` and
  `/home/jkh/Src/nanolang-file-cyclic-04d58`.
- puck frozen tree: `/tmp/nanolang-file-cyclic-04d58` (resolved `/private/tmp`).
  Its source archive and10,204-file manifest were checked before setup.
- Each phase records10,204 source hashes and9 Linux/12 Darwin tool labels.
  Labels may refer to the same executable. All24 source/tool before-after pairs
  match. Current source/tool hashes were rechecked on all three frozen trees.
- Existing164 Linux/163 Darwin prepared inputs remain identical after setup and
  across subsequent phases; setup creates its providers and is not described as
  unchanged output. Current prepared inputs were separately rehashed on both hosts.
- Explicit selected compiler paths, flags, SDK and tools are retained. Linux
  Clang uses its GCC13 install selection; puck ordinary uses Xcode Apple Clang
  and sanitizer uses Homebrew LLVM. The SDK path/version and selected ffi header
  are recorded, not a claim that every system header/library is inventoried.

[My report manifest](file-cyclic-query/report-sha256.json) hashes247 retained
reports. [My summary](file-cyclic-query/qualification-summary.json) records
24 phase pairs and all statuses. [My artifact index](file-cyclic-query/artifact-store.json)
contains782 unique files referenced7,875 times, with actual bytes retained at
`/tmp/nanolang-file-cyclic-artifacts`. The combined100,713,084-byte archive is
`/tmp/nanolang-file-cyclic-artifacts.tar.gz`, SHA256
`87422725e6eef927a9a66fac1f4647e5b87bb6a44cbab1038d13291ff8e4c0a6`.
Remote report and artifact archives have separately verified hashes recorded in
`nanolang-file-cyclic-puck-packaging.json`. Raw evidence bytes are unchanged.

I request independent seal review and actual merge before completing only this
bounded query milestone and its fixture repair. Matched cyclic hosted/VM/native
fuel/lifetime execution, public conjunction, indirect calls, richer borrows,
paired source/full shadows and full72556/6931 remain open. Public acyclic dfa149
continues independently; neither private query success nor this seal grants it
new control-flow authority.

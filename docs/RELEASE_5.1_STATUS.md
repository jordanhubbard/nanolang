# My 5.1 requirement inventory — 2026-10-11

## Current execution and readiness — 2026-11-26

I retain all69 headline entries in [my scope contract](RELEASE_5.1_SCOPE.md).
The dated inventories below remain historical observations. I do not infer
completion from their counts or from a private component checkpoint.

My [current GitHub scope inventory](evidence/release-readiness-20261126/README.md)
captures134 open issues and22 open PRs at integration source
`4a1d4e24eeeb134addfc362411ef3bec844e743e`. I classify132 issues and all22 PRs
as release-scoped after reviewing body text and release links. I preserve
#1089's explicit legacy AST-to-C non-gate and #1195's separate fleet migration.
The original69-entry contract and every optional/deferred decision are unchanged.
No APPROVED review record is present across the22 PRs. Eighteen heads, including
the integration head itself, are ancestors of integration; none is an ancestor
of observed main. I close or merge no item from that ancestry. Readiness fails.

My [published exact04cee Linux compiler receipt](evidence/darwin-current-source-20261124/diagnostics/typechecker-vector-prefix-cold-bootstrap-linux-20261126/README.md)
now records the original17-step bootstrap, byte-identical1064460-byte raw
Stage1/Stage2 modules, both native hellos, installation and no-C-seed execution.
The original native36 and emitter12 suites pass with their original limits.
[#1255](https://github.com/jordanhubbard/nanolang/issues/1255) is closed for that
Linux acceptance. The earlier cbb/16b/3021 failures below remain retained;
these later results do not turn them into passes or establish the final paired
release candidate.

My private pure translator now includes the graph/directed solver, bounded
constant reachability and shared effect-arm collector, each with its published
Linux dependency receipt. My [whole callable collector/linker draft](NANOLANG_CALLABLE_CONSTRAINTS.md)
has989 matching diagnostic packets on C-seed native and both producer VM routes,
plus505 original C assertions and26 captured original module inputs. That
initial full native gate failed under
[#1262](https://github.com/jordanhubbard/nanolang/issues/1262). I retain that
failure. The owner has since handed me both C repairs and their regressions.
My subsequent product6 callable and solver component gates pass through all
five producer/execution routes; the final combined candidate still requires
its fresh complete consumer gates and all release acceptance below.

I still require complete callable/native acceptance, representation and body/
runtime emission, module/profile/ownership composition, the original private/
public consumers, paired Linux/Darwin products and installed pure-translator
cutover. The current C-reference bootstrap remains a development prerequisite
for that architecture. Full exact-candidate CI, LITAI-001 acceptance, source/
service/ownership/reconstruction requirements, release prose, tag and publication
remain open under [#976](https://github.com/jordanhubbard/nanolang/issues/976).

My restored approved Puck connection now has a
[published fresh Darwin factory failure](evidence/darwin-current-source-20261124/diagnostics/darwin-stage1-generation-timeout-20261126/README.md)
at source d907f264d1. Cold tools pass214.086792s, but original Stage1 VM generation
times out at1800.018317s; Make bootstrap3 exits2 after1951.020760s. Only seed and
seed verification complete. No Darwin Stage1/Stage2 native product, raw equality,
installed/no-C-seed or native36/emitter12 acceptance follows. Full guards and
cleanup pass. I track bounded localization and repair under
[#1265](https://github.com/jordanhubbard/nanolang/issues/1265), preserve the failed
root and all original budgets, and make no unchanged full retry or historical
#1247/infrastructure-cause claim. The subsequent explicit handoff authorizes
my C repairs; frozen inputs, fleet restrictions and all acceptance remain in force.

My subsequent [bounded phase diagnostic](evidence/darwin-current-source-20261124/diagnostics/darwin-stage1-phase-diagnostic-20261126/README.md)
records imports37.433/lex47.444/parse61.992/typecheck69.447→294.712/program-lowering
294.828s before its300s cutoff. Typechecking completes; lowering has no terminal
outcome. Source/tool/host/frozen-input guards and cleanup pass. This localizes
phase boundaries without establishing the original timeout cause or qualifying
any compiler. My [subsequent reversible lowering probe](evidence/darwin-current-source-20261124/diagnostics/darwin-lowering-window-diagnostic-20261126/README.md)
reaches index/global setup by4.489s and queue indices0/32/64 at4.598/21.551/115.356s
within its requested120s lowering window. It records no whole lowering return;
the observer stops the VM. Source reversal, all guards and cleanup pass, but
probe timings do not establish the original timeout cause or a repair. I retain
the per-function localization and every original acceptance gate under #1265.


My [finer Darwin observation stopped under fleet policy](evidence/darwin-current-source-20261124/diagnostics/darwin-lowering-detail-policy-stop-20261126/README.md)
retains the explicit owned-process SIGTERM at2026-11-26T21:06:44Z. The VM exits-15
after251.386299s during typechecking, before lowering. Its controller records
diagnostic_valid=false; source/tool/host/frozen-input guards and cleanup pass.
The reader checks40 new records and the complete parent chain. I start no
replacement unattended Puck loop. Under the updated fleet policy, sustained
measurement and builds belong on sparky or madmax; Puck remains in the fleet
on light duty. I preserve its originals and all original Darwin acceptance.


My [subsequent Linux finer-lowering observation](evidence/darwin-current-source-20261124/diagnostics/linux-lowering-detail-diagnostic-20261126/README.md)
on sparky reaches program-lowering-done112.060742s after lowering begins. Entry41→42
(`publish_service_source`) takes25.898833s and49→50 (`assemble_text_save`) takes
6.537230s. The33-record reader re-verifies4683 Git blobs and the7088-record Linux
prerequisite; all guards and cleanup pass. I stop the VM at that return marker,
before complete compiler acceptance. These intervals include observer/loop costs;
helper attribution, a demonstrated repair and original Darwin acceptance remain
open. I change no production source, C ownership, original assertion or deadline.


My [Linux helper and artifact-boundary observations](evidence/darwin-current-source-20261124/diagnostics/linux-lowering-helper-diagnostic-20261126/README.md)
place the two intervals inside body emission25.373321s/6.365149s. An unchanged
module-artifact witness reproduces25.427272/25.503232s for first/repeated
file_product lookup and6.839779/6.692734s for nanoisa, returning the pinned prefix
library paths/bytes. I preserve all observer failures, including a raw watcher
cleanup failure reconciled by an exact-identity post-terminal audit. Built-in
tracing passes with3 cache reuses/1 miss and163ms across ten timed tool queries.
My151-record readers verify4683 source blobs per case plus33/7088 parent records.
Further retained-input profiling, C-owner coordination and a demonstrated repair
remain; no original Darwin timeout cause or compiler/release acceptance follows.


My [retained-input compiler profile](evidence/darwin-current-source-20261124/diagnostics/retained-compiler-profile-linux-20261126/README.md)
records4486 events for2243 real compiler invocations. Each file_product cache hit
invokes cc331 times, including85 assembly emissions taking21.422708/21.178067s.
All four positive lookups reuse their pinned prefix paths/bytes. The66-record
reader replays all spans and source mappings, checks4683 Git blobs and the151-record
parent chain, and verifies four observer/direct-compiler controls. The wrapper
changes compiler identity and adds overhead; I do not equate its full lookup
timings with production timings. All guards and cleanup pass. A reviewed C-owner
repair or handoff, the original Darwin failure, #1262 and full69 acceptance remain
open; no production source or qualification deadline changes.


## My C repair ownership — 2026-11-26

I own #1262 and #1265 under the explicit handoff, including their regressions.
This supersedes the earlier C-owner waiting boundaries in my retained history.
I keep all69 requirements and every original qualification deadline.

I repair unknown array-valued writes, numeric/tagged join destination storage,
checked shared-record array views and present slice results. My permanent
controls cover80 positive array writes,28 numeric join paths,16 static refusals,
250 expected runtime traps and40 raw/sliced record return cases. I retain the
fresh consumer failures that exposed slice representation and the callback
countercase that requires checked tagged storage. My guarded final candidate passes the eleven original private/public/cache,
solver and callable methods. Three companion methods pass separately after a
text-stream observer correction; the original enclosing observer remains failed.
My expanded array controls add60 nonempty content cases and36 prior-output
refusals. I publish the exact component archive and independent reader in
[my repair evidence](evidence/c-repairs-linux-20261126/README.md).

I keep every compiler invocation in cache validation and run independent
retained C-to-assembly jobs with at most eight workers. My real-process controls
cover overlap, limits, tail batches, shared/non-C inputs, failure draining,
spawn failures, signals, EINTR and an unrelated child. The complete original
bytecode-shadow/cache/source-snapshot/link target passes284 reported tests
with34 existing platform skips. My same-input assembly comparison preserves
all output bytes:81 file-product units take17.367427s serial/7.606510s parallel;
42 NanoISA units take2.618612s/0.816396s. Wrapper/direct-module measurements do
not establish an original Darwin VM-bootstrap improvement.

My broad native target reports2456 structured-C passes and two failing old
CAST_STRING refusal assertions. Those conflict with the integrated #1193
VM-compatible empty-text contract and its newer regression suite. I retain the
failed gate and a test-only reconciliation proposal; I do not call it green.

I still require broad native acceptance, fresh original bootstrap/raw
equality/installed/no-C-seed/native36/emitter12 acceptance, paired Darwin
qualification, the installed pure translator and exact-candidate CI. I publish
no tag or release from these repairs alone.

At `5d0d1dcaa4` my hosted Linux and Darwin workers both pass all17 original
bootstrap steps, raw equality, installation and no-C-seed execution. Linux
also passes native36 and the complete original284-test cache target with34
existing skips, worker3 and array5. The native gates remain failed: Linux
Stage2 exceeds the original300s emitter-setup command, while Darwin native36
has one extra duplicate-math-library warning in its exact stderr check.

My followup candidate reduces legacy record copying by sharing mutually
exclusive pointer arms while retaining separate scalar bits. A reversible
compiler-image experiment passes all12 original emitter methods in234.859s;
its emitted628200-byte module is exactly equal to the retained original once
Make's original cache directory is restored. I preserve the first comparison's
three differing host paths. The proposed generator passes all broad prerequisites,
2456 structured-C assertions and a new sanitized mixed-pointer/scalar-bit control;
the same two frozen CAST_STRING assertions still fail. I also omit only my
generated Darwin math flag, preserving caller flags and all original shadows.
Current-source native/bootstrap/paired qualification and the full release remain
open; these experimental and preflight results do not replace them.


## Current scope reconciliation — 2026-11-24

I retain **69 headline entries** in [my release contract](RELEASE_5.1_SCOPE.md):
the original 67 entries, LITAI-001 and NANOISA-NANOLANG-C-001. My original
67-row inventory and dated observations below remain historical evidence.
Their counts do not omit either later requirement from the current release.
Existing optional and deferred classifications remain unchanged; a headline
count is not a count of completed requirements.

| Additional entry | Acceptance still required |
| --- | --- |
| LITAI-001, [#1039](https://github.com/jordanhubbard/nanolang/issues/1039) | I adopt real native unit authority, locks, lifecycle parity and current independent receipts under my [ordered adoption contract](literate-ai/ADOPTION.md). Metadata or wrappers alone do not complete adoption. |
| NANOISA-NANOLANG-C-001, [#1142](https://github.com/jordanhubbard/nanolang/issues/1142) | I decode, admit, analyze and emit C in NanoLang, translate the complete required corpus and compiler products, and pass paired Linux/Darwin raw-equality and installed-product gates. An FFI call to my C translator does not implement this requirement. |

I retain my [pure translator implementation boundary](NANOLANG_NANOISA_C_TRANSLATOR.md)
and every unfinished payload, verifier, ownership, analysis, emitter, service,
driver, bootstrap and platform requirement. The C-reference translator remains
a development oracle and bootstrap tool. Its successful bootstrap does not
establish the required NanoLang translator or authorize its product cutover.

My [exact-cbb Linux checkpoint](evidence/darwin-current-source-20261124/diagnostics/core-supervisor-bootstrap-linux-20261124/README.md)
records all 17 original Make bootstrap steps, byte-identical raw Stage1/Stage2,
both native hellos and installed/no-C-seed execution. The same root's original
native-shadow gate fails five of eighteen methods; its emitter companion does
not run. My [independent localization](evidence/darwin-current-source-20261124/diagnostics/native-shadow-localization-linux-20261124/README.md)
records four pre-execution no-main refusals and a missing C-only notice.
These are recorded results at `cbbfdcb70c`, not final-candidate acceptance.

I track the entry/notice and declared foreign-helper contracts under
[#1253](https://github.com/jordanhubbard/nanolang/issues/1253). My
[unsuccessful ordinary development derivation](evidence/darwin-current-source-20261124/diagnostics/shadow-entry-development-linux-20261124/README.md)
retains its original600/default10 timeout, no compiler product and the separate
retained-image setup refusal. I admit no repair from those observations.
My [separate cold source16b checkpoint](evidence/darwin-current-source-20261124/diagnostics/shadow-entry-cold-bootstrap-linux-20261124/README.md)
passes all17 original bootstrap steps in1951.352s, raw1049944-byte equality,
both native hellos and installed/no-C-seed execution. Original native24 then
FAILS two foreign controls; all six new entry/initialization/notice methods pass.
Its emitter12 companion does not run. A separate old-pin exact artifact API
observation passes its native marker/refusal/alarm/early-exit controls, while
all five C-reference unknown-artifact translations still refuse and preserve
prior output. My reader verifies6543 records. Full native acceptance remains
open; shared-library derivatives do not establish arbitrary CFLAGS equivalence
or generic C admission.

The [original sanitizer directory-cache failure](evidence/darwin-current-source-20261124/diagnostics/core-supervisor-owned-names-linux-20261124/README.md)
remains open under [#1252](https://github.com/jordanhubbard/nanolang/issues/1252).
I retain the full ordinary-source, complete sanitizer, current-source paired,
installed closure, corpus, integration/CI and publication obligations. I keep
[#976](https://github.com/jordanhubbard/nanolang/issues/976) open until the
complete contract and exact release-candidate gates pass; this reconciliation
changes no requirement's acceptance state.


## Additional required 5.1 scope, 2026-10-28

**Required (LITAI-001):** I adopt Literate AI under [#1039](https://github.com/jordanhubbard/nanolang/issues/1039)
and the ordered [roadmap](ROADMAP.md#litai-001-my-51-literate-ai-adoption).
My scope is now the 67 original entries plus this additional requirement (68 headline entries).
I preserve every original status, identifier and explicit deferral. I require real native unit authority,
locks, lifecycle parity and current independent receipts; wrapper or original-source metadata alone is incomplete.
Compiler-contract, sanitizer, retirement and exact Linux/Darwin gates stay required as previously scoped.
My [adoption record](literate-ai/ADOPTION.md) distinguishes completed metadata from unqualified execution.


I inspect implementation at `bfab9d50a`, the release branch tip before this
reconciliation. I preserve all requirements in RELEASE_5.1_SCOPE.md and the
detailed Phase 20 roadmap. I count contractual entries, not estimated days,
bugs, independent tasks, or percentage complete.

| Headline state | Entries | Meaning |
| --- | ---: | --- |
| Recorded done | 18 | Existing bounded completion claims; not rerun at this pin |
| Implemented | 6 | Implementation evidence exists; final candidate acceptance remains |
| Required / unclosed | 31 | Full acceptance is not established; includes partial implementation |
| Optional (demoted 2026-10-13) | 4 | LLVM and Wasm rows 10661, 10670, 10671 and 10674 moved out of the release gate |
| Optional (demoted 2026-10-14) | 2 | Portable read-text rows 10663 and 10664 moved out of the release gate |
| Explicitly deferred | 5 | Existing GPU/exploratory-target deferrals, unchanged |
| Obsolete duplicate | 1 | Existing duplicate, unchanged |
| Total | 67 | Every headline scope entry appears below |

The 37 unclosed entries comprise **27 functional acceptance requirements and
10 cross-target/integration/release gates**. These overlap: a single platform
run may address multiple rows, while one broad row may require several changes.
The six implemented entries still need release acceptance too. The detailed
roadmap includes additional child defects, stale unchecked checkpoints and
historical duplicates; these headline counts do not replace that contract.

## What the evidence changes

- My compiler already emits NanoISA and uses `nvm2c` plus `cc` for native
  products. `src_nano/transpiler.nano` is absent; historical helper bodies live
  in test data. The phase enum is already renamed with a separate backend
  phase. I correct those stale implementation labels without closing their
  exact-candidate gates.
- My portable read-text and byte adapters, source execution and relocated
  native/Node packages already exist. The byte-read evidence explicitly lacks
  generated Wasmtime execution. Aggregate results, remaining compiler host
  capabilities and full platform qualification remain required.
- File callable source publication and public VM/native execution have real
  evidence. Mixed File/TCP/WebSocket source publication also exists, but its
  required successful live-peer test stops at sandbox bind `EPERM`. This is
  unqualified execution, not evidence that the entire service layer is absent.
- Native HALT and scalar effect argument/resume gaps are repaired through
  `bfab9d50a`. Resource-bearing effects, imported/nominal/generic signature
  completeness and actual foreign callback reentry remain open.
- Local names, including captured upvalues, and reconstruction have bounded
  implementations. All producer facts and full reconstruction are not
  established by those bounded tests. Managed records/arrays likewise have admitted subsets and
  private foundations, not complete applicable-target coverage.

My subsequent [declared-purity checkpoint](evidence/function-purity-20261011/README.md)
preserves source declaration flags through both producers and repairs duplicate
assembly symbols for imported functions. The broader frontend-fact requirement
remains open; these changes do not alter the headline counts.

## Concrete next acceptance blocks

| Block | Remaining evidence or implementation |
| --- | --- |
| Ownership throughout One IR | Complete facts through linking/reconstruction/all shipped consumers; full affine parity |
| Services | Complete shared source/product/platform matrix and successful live mixed network lifecycles |
| Frontend and native effects | Resource transport/cleanup, complete signature identity, real foreign reentry |
| Producer metadata | Missing storage/name conventions, purity/affine/generic/effect/exhaustiveness coverage |
| Reconstruction and profiles | Full structured language/CFG coverage and verifier-enforced applicable feature sets |
| Managed aggregate authority | Remaining arrays, generic/imported/nominal records, mutation and cycle obligations |
| Portable host capabilities | Generated Wasmtime read execution; aggregate and remaining compiler/module capabilities |
| LLVM/Wasm coverage | Applicable language beyond the currently admitted subsets |
| Candidate qualification | One exact source/module corpus, clean Linux/Darwin builds, raw Stage1/Stage2 equality, product dependency checks |
| Integration and publication | Merge/CI, issue reconciliation, release notes, tag and release |

I prioritize a contract-by-contract implementation audit within these blocks,
rather than interpreting the next unchecked historical line as missing code.
The immediate portable-host distinction is actionable: run the existing
Wasmtime acceptance if available, then implement the documented remaining
capabilities; do not recreate the already implemented read adapters.

## External state observed

GitHub reports PR #974 OPEN, head `bfab9d50a36ed5cd17d543e66c925ae74b740c97`,
base `main`, merge state BLOCKED. That field does not identify a particular
blocking check. `gh release view v5.1.0` reports `release not found`.
Issue #976 is OPEN; its body still names historical pin `7b9d9ecb7`, so I do
not use that paragraph as current implementation state. This is an observation,
not a claim that GitHub stayed reachable afterward.

My local follow-through finds Wasmtime CLI 49.0.2, but no Python `wasmtime`
binding. The checked adapter explicitly requires version 43.0.0. I find no
cached matching wheel under the task temporary directory or pip cache. Fetching
the repository-pinned Darwin wheel fails at DNS resolution before download; I
do not substitute CLI 49 for that embedding acceptance or claim a test pass.

## Addendum — 2026-10-13, branch tip `416b6e9d8`

**Scope decision, 2026-10-13 (supersedes the counts in the first table above):** my user demoted LLVM and Wasm to optional for 5.1. The first table is updated: 33 unclosed, 4 optional, total unchanged at 67. Row 10680 narrows to NanoVM and C AOT. I made no change to any other row.

I add observations since the 2026-10-11 inventory. **No headline row changes state and the
counts above stand: 18 recorded done, 6 implemented, 37 unclosed, 5 deferred, 1 obsolete.**
`gh release view v5.1.0` still reports no release and I have not tagged one.

| Rows | New evidence or gate | What remains |
| --- | --- | --- |
| 10686 | Darwin clean-clone `make bootstrap` passes at `5633d0da7` with byte-identical raw Stage 1 and Stage 2 modules ([checkpoint](evidence/bootstrap-darwin-881e49c9c-20261012/README.md)); hosted `bootstrap3` also passes on Linux x64, Linux ARM, macOS and coverage | Exact release revision, Linux Stage 2 equality, hosted CI green |
| 10685, 10688 | `make test-one-ir-product-invariants` (VM-symbol-free native product; no `transpiler.nano`) | One small program only; Linux and hosted CI |
| 10684 | `make test-selfhost-product-derivation` (`--target c` equals `nvm2c` of the `--emit-nvm` module) | `--target native` and the full compiler source; Linux and hosted CI |
| 10494 | `make test-eval` passes 127 cases and passes under ASan/UBSan on Darwin; the original SIGSEGV does not reproduce | Cause unidentified (original binary lost); Linux leak-detection run |
| 10419 | `make test-verifier-profiles test-verifier` pass on Darwin (97 assertions plus the profile gate) | Linux; the complete applicable-language coverage (10674) |
| 10687 | None this session | Pinned VM and C AOT equivalence suite at the release revision |
| closeout-1 | I repaired a module-builder buffer overflow, GCC `maybe-uninitialized` locals, a generated-struct regression of mine, and two test fixtures, and fitted CI time budgets to measured runner speeds | Hosted CI is not green: Memory Sanitizers Stage 1 exceeds 7200 seconds and is unexplained, and the Linux and macOS test suites on the newest commit had not finished when I wrote this |

**Hosted results at `b384c4f7b` (run 38428174998, finished).** For the first time the bootstrap, the
File service checks and the full `Run test suite` all passed on every hosted platform: Linux x64
(suite about 90 minutes), Linux ARM (about 106) and macOS. The Linux x64 log records the raw Stage 1
and Stage 2 modules as byte-identical. The jobs still failed later: Linux x64 and macOS at
`Validate user guide snippets` (four snippets; the self-hosted `str_equals`, `to_string` and
top-level `std/` fixes since then address three f-string snippets and the import resolution, but the
logging snippet and two regex tests need host-ABI support for `nl_get_t…` and `nl_regex_match`,
which row 10280 deliberately refuses); Linux ARM ended during the extra JUnit step with no recorded
conclusion, which I cannot explain; Code Coverage's tests passed and it failed only its 40 percent
threshold at 38.1 percent; Memory Sanitizers' Stage 1 exceeded its 7200-second bound for the third
time. This does not change a headline count: closeout-1 still requires one exact candidate with
all gates green, and the platform and corpus rows still need their own evidence.

**Hosted results at `e4ab9d3d0` (run 38507082795, still running when I wrote this).** The first fully
green Build and Test job: macOS (macos-14) passed every step, including the bootstrap, the File service
checks, `Run test suite` (about 96 minutes), the JUnit step and `Validate user guide snippets`, which
needs the self-hosted `str_equals`, `to_string` and host clock-read fixes and the 1800-second command
bound. All nine light jobs passed (fast gate, Strict examples on x64 and arm64, Docs link check, API
Docs, Code Quality, Concurrency, Performance Benchmarks, Benchmark check). Linux x64 and ARM had passed
`Run test suite` and were in the JUnit step; their user-guide result was pending. Two jobs failed:
Memory Sanitizers (Stage 1 bound, fifth occurrence) and Code Coverage in `Run tests`, where the legacy
Stage 2 rebuild reached its 600-second bound; `e74a4d21e` raises that bound and has not run yet. The
regex tests and the host-ABI scope are unchanged. This is one platform on one commit, not exact-candidate
qualification: closeout-1 still needs every gate green on one revision.

**Final result for run 38507082795 at `e4ab9d3d0`.** All three Build and Test jobs passed every step on
the same commit: Linux x64, Linux ARM and macOS, including the suite, the JUnit step and the user-guide
check. Of the eleven checks that the repository ruleset `sanity` requires, ten passed (the three Build and
Test jobs, both Strict examples jobs, Docs link check, API Docs, Code Quality, Concurrency and the
Benchmark check). The eleventh, Code Coverage, failed in `Run tests` on the legacy Stage 2 rebuild's
600-second bound, which `e74a4d21e` raises and which has not run; its earlier failures were at the 40
percent threshold (38.1 percent), so the threshold is the remaining required red once that bound clears.
Memory Sanitizers also failed (Stage 1 bound), but it is not a required check. This is one commit
and one run; it is not release qualification, because rows 10686 and 10687 and the release gates still
need their own exact-candidate evidence, and 33 headline rows remain unclosed.

**Code Coverage at `e74a4d21e` (run 38519541445).** With `CMD_TIMEOUT=1800` the job's `Run tests`
step, the coverage report and the HTML report all succeeded, and the job failed only at `Enforce coverage
threshold` (`Coverage 38.1% is below threshold 40.0%`, `.github/workflows/ci.yml` default
`COVERAGE_THRESHOLD` 40.0). The threshold's original value was 80 percent (commit `bda238ad1`) and it
has been lowered since. My user has challenged this gate as a poor trade for an agentic harness; I
have proposed making it report-only and have not changed it without a decision. The three Build and Test
jobs of this run were still finishing when I wrote this.

**Reproduced at `e74a4d21e` (run 38519541445, finished).** The same ten of eleven required checks passed
on a second commit: all three Build and Test jobs (Linux x64, Linux ARM and macOS, each through the suite,
the JUnit step and the user-guide check), both Strict examples jobs, Docs link check, API Docs, Code
Quality, Concurrency and the Benchmark check. Code Coverage again failed only at its threshold
(38.1 percent against 40.0), and Memory Sanitizers again failed at its Stage 1 bound (not a required
check). The result is therefore reproducible across two commits and not a single favorable run. It remains
platform and CI evidence only: no headline row changes state, 33 remain unclosed, and no exact-candidate
release qualification exists.

**Evidence toward row 10686 at `e74a4d21e`.** I read the finished logs of the three Build and Test jobs
of run 38519541445. Each platform's log records `I compared raw Stage 1 and Stage 2 modules: they are
byte-identical in this build.` twice (the bootstrap runs more than once per job): Linux x64, Linux ARM and
macOS. A regular expression of mine also flagged one line per log as a mismatch; that line is a long
listing of hidden file names containing the word "differ", not a comparison result. This is hosted,
three-platform evidence at one commit. Row 10686 asks for the release revision, which is not fixed, so it
stays unclosed.

**Rows 10685 and 10688 enforced in hosted CI at `675c555d8`.** The Code Coverage job of run 38552250441
ran `make test-quick`, whose new prerequisite `test-one-ir-product-invariants` executed both tests on a
hosted Linux runner with the job's coverage instrumentation and reported them `ok`: a program compiled
through `nano_virt`, `nvm2c` and `cc` runs and exports no NanoVM symbols, and no `transpiler.nano` exists
under `src_nano`. The job again failed only at its 40 percent threshold (38.1 percent). This shows the
gate holds in one instrumented configuration on one platform; the other jobs of that run had not finished
when I wrote this. It enforces a regression check for those two rows; it does not close them, because they
ask for the release revision and for the compiler product as a whole, and this covers one small program.

**Third reproduction at `675c555d8` (run 38552250441, finished).** Ten of the eleven checks that the
ruleset `sanity` requires passed again: all three Build and Test jobs (Linux x64, Linux ARM and macOS,
each through the suite with the product-invariants gate, the JUnit step and the user-guide check), both
Strict examples jobs, Docs link check, API Docs, Code Quality, Concurrency and the Benchmark check.
Code Coverage failed only at its threshold (38.1 percent against 40.0) and Memory Sanitizers failed at its
Stage 1 bound (not required). The same result now holds at `e4ab9d3d0`, `e74a4d21e` and `675c555d8`. It is
platform and CI evidence; no headline row changes state, 33 remain unclosed, and no release candidate
has been qualified.

Not touched this session: the service, managed-aggregate, producer-metadata, reconstruction,
portable-host and LLVM/Wasm blocks (rows 9538–9575, 10260–10409, 10618–10674 and 10680).
`make test-one-ir-compiler` cannot pass on Darwin arm64 because LeakSanitizer is unsupported
there; it requires a leak-capable Linux run.

## All headline entries

“Recorded” below retains the scope's existing bounded completion claim.
An evidence link identifies a checkpoint or remaining contract, not proof that
an unclosed requirement is satisfied at the audited pin. Exact-candidate gates
remain open even when an older bootstrap or focused suite passed.

| ID | State | Requirement | Evidence / boundary |
| --- | --- | --- | --- |
| 9511 | Recorded done | Replace the C seed ownership prototype with path-sensitive analysis; both frontends need the same affine rules. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9538 | Unclosed | Preserve verified ownership facts through the module, linker, reconstruction, and every shipped translator. | [Checkpoint; full acceptance open](ORIGINAL_ACCEPTANCE_RECONCILIATION.md) |
| 9547 | Unclosed | Connect generated File and Result descriptors to a verified service-call boundary; integer wrappers are not enough. | [Checkpoint; full acceptance open](evidence/file-callable-source-20261009/README.md) |
| 9549 | Unclosed | Integrate the reviewed five-method File descriptor query before executable service publication. | [Checkpoint; full acceptance open](evidence/nsi-file-plan.md) |
| 9551 | Unclosed | Qualify private File provenance and Result transfer and cleanup before public admission. | [Checkpoint; full acceptance open](evidence/file-indirect-public-20261009/README.md) |
| 9558 | Unclosed | Complete File loops, indirect calls, multi-borrow calls, cleanup, source publication, and selected shadows. | [Checkpoint; full acceptance open](evidence/file-callable-source-20261009/README.md) |
| 9559 | Unclosed | Review matched File and Result flow through VM and native dispatch before changing public selection. | [Checkpoint; full acceptance open](evidence/file-indirect-public-20261009/README.md) |
| 9560 | Unclosed | Qualify paired File generation and all shadows through the C seed, Stage 1, and Stage 2 on Linux and Darwin. | [Checkpoint; full acceptance open](evidence/bootstrap-file-integration-20261009/README.md) |
| 9561 | Unclosed | Admit File publicly only after verifier, VM, native, generated-binding, and real-file equivalence passes. | [Checkpoint; full acceptance open](evidence/file-indirect-public-20261009/README.md) |
| 9563 | Recorded done | Qualify one real local Socket lifecycle; service-handle migration requires a tested socket owner. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9564 | Recorded done | Review the private Socket API and close/error policy before host operations. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9565 | Recorded done | Qualify Socket lifecycle, refusal, fault, and adjacent File/capability controls on both hosts. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9566 | Unclosed | Complete public network/WebSocket bindings and paired verified execution; the local pair alone is insufficient. | [Checkpoint; full acceptance open](evidence/mixed-websocket-source-20261010/README.md) |
| 9567 | Deferred-to-later | Qualify a real CUDA GPU buffer lifecycle; this needs explicit GPU hardware and does not block the portable One IR cut. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9568 | Deferred-to-later | Review CUDA driver/context restoration on qualified hardware as part of the GPU lifecycle. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9569 | Deferred-to-later | Run real GPU allocation, copy, release, context, and fault gates on qualified hardware. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9570 | Deferred-to-later | Extend GPU acceptance to other platforms and public bindings after the portable 5.1 release. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9572 | Unclosed | Migrate real File, Socket, and capability handles after ownership and IR enforcement; GPU follows its deferred hardware gate. | [Checkpoint; full acceptance open](evidence/mixed-websocket-source-20261010/README.md) |
| 9575 | Unclosed | Pass one affine matrix across both frontends, NanoISA, NanoVM, and AOT C. | [Checkpoint; full acceptance open](ORIGINAL_ACCEPTANCE_RECONCILIATION.md) |
| 9707 | Recorded done | Infer C-seed loop element metadata for inline and computed arrays; compiler-source lowering depends on valid loops. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9879 | Recorded done | Propagate transitive foreign header paths through C-seed module compilation. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9882 | Recorded done | Reclaim shared static arrays and owned elements at an alias-safe interpreter boundary. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9902 | Recorded done | Define Boolean List syntax and backend parity before the compiler product claims list coverage. `List<bool>` now parses and the interpreter, native C and NanoVM backends share its boolean element contract (`tests/nl_types_bool_list.nano`). | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9941 | Recorded done | Reconcile the legacy `string_from_char` alias so VM and AOT resolve the same host contract; NanoVM exports the alias and `tests/test_one_ir_compiler.py` executes both names in both backends. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 9978 | Recorded done | Prevent ordinary record names from colliding with native runtime helper names. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10055 | Implemented | Carry record and map globals through standalone AOT with correct runtime ownership. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| 10068 | Implemented | Bind imported globals and selective or qualified aliases in the standalone product. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| 10153 | Implemented | Make `--emit-nvm` the self-hosted compiler's only backend and make binary output an explicit `nvm2c` plus `cc` pipeline. | [Checkpoint; full acceptance open](evidence/native-driver-cutover-20261007/README.md) |
| 10260 | Unclosed | Finish the self-hosted NanoISA lowering dual for the full compiler subset. | [Checkpoint; full acceptance open](evidence/native-effects-20261011/README.md) |
| 10262 | Recorded done | Compile the compiler to `.nvm` through the C seed and then the self-hosted emitter. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10264 | Recorded done | Compare Stage 1 and Stage 2 module bytes; native binary comparison remains a translator test. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10266 | Implemented | Freeze and remove `transpiler.nano` from the product after the compiler module builds itself. | [Checkpoint; full acceptance open](evidence/legacy-extern-retirement-20261009/README.md) |
| 10269 | Implemented | Rename the transpiler compiler phase so the language pipeline ends at NanoISA. | [Checkpoint; full acceptance open](evidence/compiler-phases-20261008/README.md) |
| 10272 | Recorded done | Make C11 the canonical AOT consumer of NanoISA without a VM process dependency. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10278 | Recorded done | Cover compiler functions, records, control flow, arrays, strings, modules, and declared host ABI in `nvm2c`; `STR_EQ` is now native (`make test-nvm2c`, 2,431 passed). | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10280 | Recorded done | Lower `CALL_EXTERN` only through the declared host ABI and refuse unsupported imports; `make test-nvm2c` (2,431 passed) covers the builtin and artifact adapters and the refusal path. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10282 | Recorded done | Keep `wrapper_gen` explicitly packaged-interpreter-only in the CLI and documentation; the `-o` help and verbose output name a packaged interpreter, never native AOT (`make test-wrapper-packaged-interpreter`). | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10400 | Unclosed | Store local names in module metadata for useful reconstruction and diagnostics. | [Checkpoint; full acceptance open](evidence/upvalue-names-20261011/README.md) |
| 10407 | Unclosed | Preserve purity, affine, generic, effect, and exhaustiveness frontend facts in module metadata. | [Checkpoint; full acceptance open](NANOISA_ONLY.md) |
| 10409 | Unclosed | Recover structured `if`, `while`, and `return` from verified control flow for reconstruction. | [Checkpoint; full acceptance open](evidence/reconstruction-feasibility-audit.md) |
| 10419 | Unclosed | Define verifier-enforced general and restricted compute profiles. | [Checkpoint; full acceptance open](evidence/nanoisa-verifier-profiles.md) |
| 10494 | Unclosed | Resolve the retained evaluator shadow failure before complete compiler-product acceptance. | [Checkpoint; full acceptance open](ROADMAP.md) |
| 10618 | Unclosed | Finish managed string-array ownership and target execution. | [Checkpoint; full acceptance open](evidence/managed-split-arrays.md) |
| 10623 | Unclosed | Preserve aggregate and collection identity, mutation, and an explicit cycle policy on the managed runtime. | [Checkpoint; full acceptance open](evidence/managed-graph-safepoints.md) |
| 10643 | Unclosed | Establish explicit ordinary heap-bearing record authority before nominal target admission. | [Checkpoint; full acceptance open](evidence/ordinary-record-producers.md) |
| 10648 | Unclosed | Cover array elements, generics, imports, forward order, and remaining nominal record authority. | [Checkpoint; full acceptance open](evidence/managed-forward-records.md) |
| 10656 | Obsolete | The shared checked record selection row duplicates completed prior-order and forward-record execution rows; remaining authority is tracked at 10643 and 10648. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10661 | Optional (demoted 2026-10-13) | Implement exact declared host/module capability linkage and owned results for LLVM and Wasm. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10663 | Optional (demoted 2026-10-14) | Review the portable read-text linkage proposal as the first concrete portable host capability. | [Checkpoint; full acceptance open](NANOISA_PORTABLE_READ_TEXT_EXECUTION.md) |
| 10664 | Optional (demoted 2026-10-14) | Qualify the read-text query, adapters, opt-in lowering, source publication, and installed links in ordered checkpoints. | [Checkpoint; full acceptance open](evidence/portable-read-execution-20261010/README.md) |
| 10669 | Implemented | Implement the private allowlisted read-text adapters and managed-copy ownership checkpoint. | [Checkpoint; full acceptance open](evidence/portable-read-execution-20261010/README.md) |
| 10670 | Optional (demoted 2026-10-13) | Qualify native, LLVM, Wasmtime, and Node adapters with real files and denial/fault cleanup. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10671 | Optional (demoted 2026-10-13) | Extend portable linkage to byte and aggregate results and the remaining compiler capabilities. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10672 | Recorded done | Complete LLVM IR translation from NanoISA rather than from the NanoLang AST. I admit `ROT3` in the shared closed profile and lower it in `nvm2llvm`; VM/C/LLVM parity tests pass. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10673 | Recorded done | Complete WebAssembly translation from NanoISA rather than from the NanoLang AST. I translate verified NanoISA through the shared `nvm2llvm` closed profile, including `ROT3`; VM/C/LLVM/Wasmtime/import-free Node parity tests pass. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10674 | Optional (demoted 2026-10-13) | Cover the applicable language in both translators before exposing them as release targets. | [Checkpoint; full acceptance open](NANOISA_ONLY.md) |
| 10676 | Deferred-to-later | Evaluate JVM, SPIR-V, PTX, OpenCL, and Metal translators; 5.1 does not ship every evaluated optional target. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10680 | Unclosed (narrowed) | Run one pinned module corpus through NanoVM and C AOT and compare semantics; LLVM and Wasm are an optional cross-check. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| 10684 | Unclosed | Verify that `src_nano` emits `.nvm` as its only compiler product. | [Checkpoint; full acceptance open](evidence/native-driver-cutover-20261007/README.md) |
| 10685 | Unclosed | Verify that `nvm2c` products do not link `nano_vm`. | [Checkpoint; full acceptance open](evidence/native-effects-20261011/README.md) |
| 10686 | Unclosed | Verify byte-identical Stage 1 and Stage 2 `.nvm` products at the release revision. | [Checkpoint; full acceptance open](evidence/structural-bootstrap-0c501dc4b/README.md) |
| 10687 | Unclosed | Verify the pinned NanoVM and C AOT equivalence suite at the release revision. | [Checkpoint; full acceptance open](evidence/effect-byte-enum-20261011/README.md) |
| 10688 | Unclosed | Verify that `transpiler.nano` is absent from the product compiler. | [Checkpoint; full acceptance open](evidence/legacy-extern-retirement-20261009/README.md) |
| 1033 | Unclosed | Reconcile completed roadmap rows with merged commits and authoritative GitHub issue states. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-1 | Unclosed | Keep main CI green and run the clean release/platform gates on the exact candidate revision. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-2 | Unclosed | Draft `RELEASE_5.1.md` after the required implementation rows close, using `RELEASE_5.0.md` as the format. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-3 | Unclosed | Tag and publish `v5.1.0` after the complete scope and exact-candidate release gates pass. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |

## Addendum 2026-10-14: coverage is report-only

By owner decision, the `Code Coverage` job no longer enforces a line-coverage floor. It still builds, runs the test suite and publishes the percentage (job summary, PR comment, artifact), so the required check fails only when the instrumented tests fail. Rationale: a percentage floor steers an agentic harness toward unit tests that do not guard behaviour. Behaviour gates remain the product-invariants test, bootstrap identity and the strict examples. `make coverage-check` keeps its local threshold for developers.

## Addendum 2026-10-14: rows 10663 and 10664 demoted

By owner decision, rows 10663 and 10664 (portable read-text) are optional for 5.1. Unclosed headline rows: 31. Optional: 6. Total unchanged at 67. The checkpoint evidence stays linked; it is not a release gate.

## Addendum 2026-10-14: regex host ABI and bootstrap scope

- **Regex host symbols.** By owner decision I added the nine `nl_regex_*` C-backed externs to the declared host ABI in `src_nano/compiler/nanoisa_codegen.nano` (opaque `Regex` handle, `array<int>`/`array<string>`/`string`/`int`/`void` results, module-path import `./std/regex/regex.nano`, as the seed emits). `tests/test_split2.nano` and `tests/test_std_regex_groups_uaf.nano` now compile on the self-hosted compiler and run on NanoVM. `tests/test_selfhost_regex_host_abi.py` gates both and runs in `test-quick`. Not verified on hosted Linux yet, and the C AOT (`nvm2c`) path for these imports is not tested.
- **Bootstrap scope.** By owner decision, 5.1 does not require an instrumented full bootstrap; the sanitizer Stage 1 job stays non-required.

## My current Linux native component — 2026-11-27

At6e09c07a9f I pass the original bootstrap17/native36/emitter12 sequence with
matching source/tool/head/index/six-host guards. The raw modules are identical
1064904 bytes, SHA8920ee0546671f9fa13ea98f7aaf5030b235d6d7349b277cdb875b4a5aa46833.
The installed image and original no-C-seed probe pass. All12 emitter methods
complete256.998s with the original300s setup commands. I retain and independently
verify [my current Linux component receipt](evidence/c-repairs-linux-20261126/native-current-linux-20261127/README.md).
The hosted Linux cache284/34 existing skips, workers3 and combined array/pointer6
also pass on that code revision. Current Darwin, broad CAST_STRING reconciliation,
required exact-candidate CI, pure-translator cutover and all69 release requirements
remain open. This component result accepts no complete release.


My [current hosted Linux original-gate receipt](evidence/c-repairs-linux-20261126/hosted-current-linux-45970514720/README.md) independently verifies source6e09c07a9f: all17 bootstrap steps, raw1064512-byte equality, actual generated index, five compiler products, six host libraries, installed/no-C-seed, native36 and emitter12. Both exact GitHub artifact digests and source/tool/image guards match. The complete cache284/34 existing skips, workers3 and array/pointer6 pass. This adds hosted Linux component evidence; current Darwin, the active current-producer affine/solver/callable rerun, frozen CAST_STRING decision, exact-candidate required CI, pure-translator cutover and all69 release requirements remain open.

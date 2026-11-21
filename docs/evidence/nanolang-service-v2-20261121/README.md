# My rich-service implementation and incomplete paired qualification

I retain implementation source at `a9f831f7e`, following `88de21768`, on
`feat/5.1-nanolang-translator` / draft #1144. I implement normal rich-v2
service-table adaptation in `service_v2.nano`. I preserve legacy signature-index
ordering, nominal preflight, prior layout edges, typed constants and signature
storage, metadata presence and bounded File classification. Ordinary shared
transport and the private WebSocket adapter remain distinct. I call no C
validator from this NanoLang implementation.

My current fresh-source qualification is incomplete:

| Check | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Fresh development compiler, default dependency/root shadows | Published | Two creations refuse at the original ten-second shadow deadline |
| Combined component methods at a9f | 22 pass in 323.443s | Never starts: no fresh compiler image |
| Rich reports per execution route | 5,130 through five routes | Earlier 88de run passes five routes; separate pin/image |
| Independently collected synthetic carriers and actual C reports | 5,122 | 5,122; complete carriers/reports match Linux |
| Original structured-C / shape / callable checks | 2,458 / 3,890 / 505 pass | 2,458 / 3,890 / 505 pass |
| Original checker / type inference / module metadata / evaluator | Pass | Pass |
| Owner context / checker metadata ownership / focused owner ASan+UBSan | Pass | Pass |
| Original shared-service methods | Three pass after explicit archive-order correction | Three pass |

My synthetic corpus contains 4,722 explicit expected statuses. The remaining
mutation cases compare complete actual-C status/classification reports. I check
constant/result tags, signature references and absent storage, feature/import
presence, legacy ordering, prior-edge/preflight ordering, all 256 opcode bytes,
unsigned function spans, layout/import bounds and 64 instances. Each complete
Linux route adds eight actual File/TCP/WebSocket/mixed producer modules.

My five routes are C-seed native execution, C-producer NanoVM and instrumented C
AOT, and genuine NanoLang-producer NanoVM and instrumented C AOT. I build the
Linux development source compiler from `src_nano/nanoc_v06.nano`, with default
selected shadows, and configure the existing artifact-capture helper explicitly.
These are development products, not installed Stage 1/2 qualification. Generated
C drivers use ASan/UBSan and an ordinary AOT runtime. The separate actual-C oracle
instruments the service/scanner/nominal/raw/layout-adapter providers. My focused
owner sanitizer image instruments `nominal_types.c` and its fixture; other common
objects are ordinary. I make no full-interpreter sanitizer claim.

I retain the constructor/return/conditional/record-join failures that this work
exposes under #1145, #1153 and #1154. My Linux component gate executes three
positive constructor/container/branch fixtures through all five routes and seven
same-shape nominal refusals through all three producers, preserving prior output.
The branch fixture checks selected values, early return, lexical cleanup and
once-only side effects. The native record-join matrix checks int/bool/float/string
payload equality and present/missing tags in both branch orders, plus incompatible
exact/aggregate field refusals. I preserve the unchanged lifetime witness, its
initial join refusal, the intermediate unresolved-pack refusal and its corrected
VM/native result 42. These passing Linux controls do not close paired acceptance.

My first Darwin creation has no shadow trace and cannot establish its timeout
phase. My separate observed attempt records 956 starts, 955 completions, zero
assertion failures and 9.099963 summed body seconds. SIGALRM interrupts
`nisa_call_host_symbol`; dispatch/report/completion never finish. I retain the
validated monotonic phase/wait report and the original ten-second deadline. A
one-second parent sample observes module snapshot reads before shadow dispatch;
it does not explain the original deadline. My child profile
retains its own prerequisite, attempt and terminal evidence. Its actual eight-second
sample contains a 657-sample call graph, but the stream parser rejects the
interrupted final JSON record and cleanup terminates the parent. I preserve
that refusal, the partial trace and the copied C-seed image. I audit these
completed observations offline and do not repeat that sampling plan for green. I do not turn a
later success or incomplete sample into reliable source creation.

My partial audit checks 972 source hashes against Git `a9f831f7e` and both host
manifests. Tool hashes are post-run observations, not invented before-run
identities. My additional permanent Linux producer qualification retains eight
modules, their rich carriers, all sources and 40 file byte strings; all five
actual routes agree for every product. Ordinary unittest temporary products were
deleted by their harness. I retain these additional products separately rather
than claim those deleted bytes remain available. No fresh Darwin product archive
exists at this checkpoint.

I retain reporting/setup refusals, the obsolete compiler-entry attempt, baseline
array-diagnostic drift, the misplaced intermediate checker edit, both native join
refusals and the Float payload correction. I preserve the original GNU ld static
archive-order failure and its explicit correction. The later Make repair and
sibling runtime qualification belong to #1155 at `eb76f0e9b`; those do not alter
this source pin. The exact-head user-guide macro failure is separately tracked
under #1156 and corrected in `e5542a999`.

`manifest.json` maps retained files to compressed bytes and SHA-256 values. I
retain the collectors, terminal records, synthetic inputs/reports, source/tool
manifests, implementation snapshots, source/module/generated-C witnesses and
original test logs. I retain the profiler's copied C-seed executable; other tools are identified
by hash. My archive builder reads tracked sources from the immutable code pin.

I keep paired fresh-source qualification, all new component roadmap rows and
#1142 open. Whole-module profile conversion, instruction/flow/ownership/host
admission, native emission, complete pinned corpora, raw fixed points, installed
products, integration and every original 5.1 release requirement remain open.
Metadata success grants none of that authority.

# My 5.1 requirement inventory — 2026-10-11

I inspect implementation at `bfab9d50a`, the release branch tip before this
reconciliation. I preserve all requirements in RELEASE_5.1_SCOPE.md and the
detailed Phase 20 roadmap. I count contractual entries, not estimated days,
bugs, independent tasks, or percentage complete.

| Headline state | Entries | Meaning |
| --- | ---: | --- |
| Recorded done | 18 | Existing bounded completion claims; not rerun at this pin |
| Implemented | 6 | Implementation evidence exists; final candidate acceptance remains |
| Required / unclosed | 37 | Full acceptance is not established; includes partial implementation |
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
| 10661 | Unclosed | Implement exact declared host/module capability linkage and owned results for LLVM and Wasm. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10663 | Unclosed | Review the portable read-text linkage proposal as the first concrete portable host capability. | [Checkpoint; full acceptance open](NANOISA_PORTABLE_READ_TEXT_EXECUTION.md) |
| 10664 | Unclosed | Qualify the read-text query, adapters, opt-in lowering, source publication, and installed links in ordered checkpoints. | [Checkpoint; full acceptance open](evidence/portable-read-execution-20261010/README.md) |
| 10669 | Implemented | Implement the private allowlisted read-text adapters and managed-copy ownership checkpoint. | [Checkpoint; full acceptance open](evidence/portable-read-execution-20261010/README.md) |
| 10670 | Unclosed | Qualify native, LLVM, Wasmtime, and Node adapters with real files and denial/fault cleanup. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10671 | Unclosed | Extend portable linkage to byte and aggregate results and the remaining compiler capabilities. | [Checkpoint; full acceptance open](evidence/portable-byte-execution-20261010/README.md) |
| 10672 | Recorded done | Complete LLVM IR translation from NanoISA rather than from the NanoLang AST. I admit `ROT3` in the shared closed profile and lower it in `nvm2llvm`; VM/C/LLVM parity tests pass. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10673 | Recorded done | Complete WebAssembly translation from NanoISA rather than from the NanoLang AST. I translate verified NanoISA through the shared `nvm2llvm` closed profile, including `ROT3`; VM/C/LLVM/Wasmtime/import-free Node parity tests pass. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10674 | Unclosed | Cover the applicable language in both translators before exposing them as release targets. | [Checkpoint; full acceptance open](NANOISA_ONLY.md) |
| 10676 | Deferred-to-later | Evaluate JVM, SPIR-V, PTX, OpenCL, and Metal translators; 5.1 does not ship every evaluated optional target. | [Existing scope classification](RELEASE_5.1_SCOPE.md) |
| 10680 | Unclosed | Run one pinned module corpus through NanoVM, C AOT, LLVM, and Wasm and compare semantics. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| 10684 | Unclosed | Verify that `src_nano` emits `.nvm` as its only compiler product. | [Checkpoint; full acceptance open](evidence/native-driver-cutover-20261007/README.md) |
| 10685 | Unclosed | Verify that `nvm2c` products do not link `nano_vm`. | [Checkpoint; full acceptance open](evidence/native-effects-20261011/README.md) |
| 10686 | Unclosed | Verify byte-identical Stage 1 and Stage 2 `.nvm` products at the release revision. | [Checkpoint; full acceptance open](evidence/structural-bootstrap-0c501dc4b/README.md) |
| 10687 | Unclosed | Verify the pinned NanoVM and C AOT equivalence suite at the release revision. | [Checkpoint; full acceptance open](evidence/effect-byte-enum-20261011/README.md) |
| 10688 | Unclosed | Verify that `transpiler.nano` is absent from the product compiler. | [Checkpoint; full acceptance open](evidence/legacy-extern-retirement-20261009/README.md) |
| 1033 | Unclosed | Reconcile completed roadmap rows with merged commits and authoritative GitHub issue states. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-1 | Unclosed | Keep main CI green and run the clean release/platform gates on the exact candidate revision. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-2 | Unclosed | Draft `RELEASE_5.1.md` after the required implementation rows close, using `RELEASE_5.0.md` as the format. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |
| closeout-3 | Unclosed | Tag and publish `v5.1.0` after the complete scope and exact-candidate release gates pass. | [Checkpoint; full acceptance open](RELEASE_5.1_SCOPE.md) |

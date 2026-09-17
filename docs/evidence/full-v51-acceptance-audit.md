# My remaining full-v5.1 acceptance contract

I audited source and documentation at `7e27be20` (PR496), merged-PR metadata,
and live MAC task states on 2026-09-17. This is a bounded acceptance audit,
not a test run or release approval. I preserve the publication hold. My parent
contract is `task_7bad6bb81bdc3eef2e9a8bf0ba52f2ff`; its stopped dispatch state
does not cancel the user's full-roadmap requirement. Phase 22 / 6.0 stays separate.

My authority is the full-roadmap queue and Phase 20 in [ROADMAP](../ROADMAP.md),
[NanoISA-only acceptance](../NANOISA_ONLY.md#acceptance), and the normative
[affine contract](../AFFINE_TYPES_DESIGN.md). The earlier local
`/tmp/nanolang-full-roadmap-audit.{md,json}` recorded 47 checklist obligations
against `a06cbc11`, then `e0d01acb`. Those were not 47 independently reproduced
bugs. I retain that historical inventory without treating its old failure
counts, paths or classifications as current acceptance evidence.

## My dependency order

| Order | Required result and next bounded slice | Existing ownership / evidence |
| --- | --- | --- |
| 1 | I finish canonical mandatory VM shadows and their complete compiler closure, then execute the full compiler as standalone AOT against its declared host ABI. Initializer frames are merged; range/float lowering and filesystem ABI are active dependencies. I require an executable final-compiler product, not successful translation alone. | Root and VM work already assigned. PR491/494/495 establish selected-module emission, supervision and initializer frames. `vm-bytecode-fixedpoint.md` explicitly separates the unfinished native compiler route. |
| 1, independent | I add paired call-scoped `&T` and `&mut T` syntax, annotation retention and ownership semantics. My first slice must prove repeated shared reads then consumption, exclusive mutation then consumption, rejection of consuming/mutating shared borrows, overlapping exclusive arguments, use after move, and escaping/storing borrows. I keep unsupported contexts rejected rather than claiming general references. | `AFFINE_TYPES_DESIGN.md` rules 2/5 and its matrix require this; current boundary explicitly excludes borrow support. `src/resource_flow.c` tracks transfers but has no borrow-state implementation. Existing C umbrella `task_c4e2f078cef8c4e461f0de3711c8a2b9` is cancelled; a scoped continuation must precede code. |
| 2 | I reconcile every affine matrix row across C seed, Stage 1 and Stage 2, including error propagation with unrelated live owners, nested aggregate transfer, loop/early-return joins and borrowed projections. I distinguish supported ownership transfer from conservative guards. | `generic-selected-ownership.md` proves bounded selected transfer; PR475 rejects unsupported global resources and PR489 rejects generic resource callback signatures. Neither proves those broader lifetimes/transfers. `task_195cac35e7704e56805932977512ae02` tracks safe early-return result transfer. |
| 3 | I encode ownership facts with stable declaration/place identity, verify transfers and joins, preserve them through serialization/linking/reconstruction, then compare VM/AOT behavior before migrating real handles. | Blocked live tasks `task_ed70242ac4d83be7b2327da7ece387ad`, `task_d03c232dc067e75cbc2fb2b7fb84ee46`, `task_28f2fb4b1f3c8a5ce93df628bb569d76`. Their cancelled/completed frontend dependencies need truthful reconciliation, not automatic closure. |
| 1–3, independent | I connect the closed purity checker to frontend `par`/`flow` eligibility emission with deterministic serial semantics. Start with the already verified constant/scalar record, then add only independently proved immutable inputs and exact trusted-call identities. Compare the emitted module on VM and AOT. | `passive-scalar-metadata.md` states that no frontend emits these records yet. External inputs and calls remain excluded. Open tasks `task_bf571298c10d4cc5a387b9f233ff3c40` and `task_20f6cb36fbf24bba987b4ea503529438`; completed definition task `task_90b123edcc301b464a031c55e4ba1a11` is not implementation acceptance. |
| 4 | After the complete VM-shadow and standalone compiler gates, I route product compilation exclusively through verified `.nvm` and `nvm2c`, remove product dependence on `transpiler.nano`, rename the obsolete transpiler phase and repeat the raw bytecode fixed point on that exact route. | `src_nano/nanoc_v06.nano` still imports `transpiler.nano` and has separate native/`emit_nvm` branches. I do not delete the fallback before its replacement passes. |
| 5 | I retain local names and frontend purity/affine/generic/effect/exhaustiveness facts; establish verifier-enforced general/restricted profiles; recover structured control and run the second high-level-surface spike from the same module. | Phase 20 module-richness/reconstruction rows remain unchecked. `NANOISA_HL_ROUNDTRIP.md` exists, but its integer-only inventory is historical and must be refreshed from measured current capability. Disassembly roundtrip alone is insufficient. |
| 6 | I reconcile translator scope explicitly, then run one pinned module corpus on VM, AOT C and every shipped translator. AST PTX/OpenCL product routes must become translators or leave the product. | `NANOISA_ONLY.md` makes LLVM/Wasm optional if absent, while roadmap implementation rows are unconditional. I cannot silently choose either interpretation: preserve these rows for an explicit contract reconciliation. GPU/JVM evaluation is required; shipping every evaluated target is not. |
| 7 | I complete the separately named full-roadmap audits and fresh platform release evidence on the final revision. | Language claims, formal-model correspondence, component-shadow execution, native service lifecycle, typed FFI authority, measured authoring/repair ergonomics, and enforceable release evidence remain explicit audits. `task_cffdafd16e641ac417ccfddb962534b9` owns the latter; the stopped Darwin audit is `task_07c456728ae3b2b4dab1c42744f8c049`. Historical green release CI does not cover the new contract. |

## My bounded reconciliation

I found these stale or duplicated claims; I did not edit their checkboxes:

- PR460 (`ef275046`) completed nested generic payload substitution and union
  global initialization, task `task_85a8db6e186440eaad80442bfc133dd8`.
  Its Phase 20 unchecked failure note is stale within that tested scope.
- PR492 (`ba55b5fa`) supplies qualified and indirect callback signature checks.
  The unchecked qualified-call mismatch note is stale. Parent
  `task_e05a42e2e09b47cc9c53fa6923eeeaef` remains open and is not closed by
  metadata serialization or guarded generic resource callbacks alone.
- PR485 (`2de449f7`) records two verified VM-executed generations with identical
  raw 352,236-byte outputs at clean `1277bce2`; its permanent gate also runs a
  final compiler product. I can reconcile duplicate fixed-point checkboxes to
  that dated result, but not claim the later NanoISA-only product cutover:
  those generations still generated and ran native C shadows.
- The affine design's current-boundary statement that general generic selected
  union transfer is absent predates PR469 (`c2769a39`). Its bounded selected
  transfer evidence supersedes that sentence, not the whole contract.
- The self-host affine task `task_20048de825616195b9f2bc492231a851` is completed,
  while its broad Phase 20 row remains unchecked. I use the paired evidence
  documents to delimit progress; neither the checkbox nor terminal task state
  proves all normative rows. The same distinction applies to passive purity.

## My followup boundary

I do not turn every discovered repair into a new release criterion. For example,
legacy raw `string_from_char` alias compatibility, `List<bool>` syntax, a GCC
sanitizer configuration diagnostic and historical capture incidents need their
own declared acceptance or demonstrated dependency before they block this
contract. Reproducible failures of supported pinned behavior still block their
own gate; I do not dismiss them because they were discovered incidentally.

Imported globals (`task_2713a842846b417fbfd6aa4b8059d0dd`) and standalone record/map
globals (`task_95796f5f49564ed4a911fd05a1aac5b4`) are concrete unresolved product
capabilities to include in applicable-language parity. Explicit `map_free`
NanoISA semantics (`task_2f848b73acf847a79df68418b9213637`) and borrowed-result
cleanup need an agreed lifetime contract, not an inference from PR493/496.
Conservative rejection of resource-bearing collections is required by the
current affine design; supporting resource collections is not a missing feature.

I have not rerun compiler, platform or release gates in this documentation audit.
I checked the cited merge ancestry and local evidence paths. My next implementation
slice is paired call-scoped borrows; the full publication hold remains in force.

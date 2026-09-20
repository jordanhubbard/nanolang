# I reconcile bounded owned and managed acceptance

I read the current MAC descriptions for430220,4be and e8d and their original
contracts after canonical PR876. Those descriptions still state the initial
proposal; they do not describe the merged implementations. This audit proposes
bounded reconciliation for review. It does not itself complete a parent or
replace the original contracts with narrower acceptance.

## My owner-array contract430220

The numbered criteria below come from
[NANOISA_OWNED_FLOAT_ARRAY_FIELDS.md](NANOISA_OWNED_FLOAT_ARRAY_FIELDS.md).

| Original criterion | Merged evidence | Boundary retained |
| --- | --- | --- |
|1: reviewed private field provenance/authority, joins/initialization/all bodies, calls/results, atomic publication before admission | PR824 `0c7756c2`,829 `e119cda8`,830 `9d254504`,832 `3b0f3151`; [layout](evidence/owned-array-layout-query.md), [origins](evidence/owned-array-origin-query.md), [authority](evidence/owned-array-authority-query.md), [operand obligations](evidence/owned-array-operand-query.md) | Private success is not public runtime permission; original nominal/global identity and exact flat FLOAT proof remain required. |
|2: reviewed VM/native lifecycle and unchanged other-profile refusals before source | PR836 `2cb8728a`,838 `f88af33f`,842 `8953110f`; [private runtime](evidence/private-owned-array-runtime.md), [true switch](evidence/private-owned-array-true-switch.md), [public activation](evidence/owned-array-public-activation.md) | Fresh complete admission and service-first refusal; no mixed-selector fallback, nonzero links, helper/closure entry or LLVM/Wasm admission. |
|3: aliases across pack/observe/unpack/move/call/return, writes/growth, empty/nested owners, overwrite, assertions/repeated calls | Above runtime, PR847 `a52d990d`,849 `ec8f5b7a`,876 `4fbca357`; [mutation](evidence/owned-array-mutation-runtime.md), [Darwin mutation](evidence/owned-array-mutation-darwin.md), [distinct overwrite](evidence/owned-array-overwrite/README.md) | The last missing distinct A-to-B occupied ARRAY-local overwrite retains owner/outside aliases, independent B/C values and A growth. It proves VM/native replacement, not source binding assignment. |
|4: allocation/growth/publication failure, sentinels/surviving aliases/recovery, signed/extreme indices and exact FLOAT bits with native sanitizers | Public authority/runtime sweeps above; PR852 `c4529256` [bits/extreme indices](evidence/owned-array-bits-boundaries.md), plus876 overwrite sweeps | Exact partial instrumentation and allocation categories remain as sealed. No floating equality substitutes for uint64 bit observations; no disposal sweep hides live roots; no every-system-allocation claim. |
|5: separately reviewed paired source, original Bundle/PREFIX/shadows, canonical/VM/native/installed routes, order/inference/false shadows/refusals on both hosts | PR846 `3660228d` [source](evidence/owned-array-source-final.md); PR861 `59ffcecc` [mutation/source final](evidence/owned-array-mutation-final/README.md) | Exact builtin mutation/length is admitted. Bare ARRAY signatures, managed-root borrowing, owner-field assignment and managed binding reassignment remain outside that reviewed source slice. Original Bundle and full12 pattern suite pass without changes. |
|6: integrated source/tool/command/status/artifact seals; bounded child only | All named seals and canonical merge ancestry; proposed PR878 adds fresh851-source full33 script on bothhosts | Source mutation acceptance includes fresh bootstrap, full6 identity/12 mutation/12 patterns and evaluator123/totality8 at its frozen pin. Later current-product/fixed-point/release acceptance is separate. |

The independent native-effects audit identifies no additional unmeasured runtime
criterion after mutation, exact bits/extreme indices and distinct overwrite.
I keep the source reassignment refusal visible because runtime STORE alone cannot
prove source admission. The original contract's alias-after-local-reassignment
lifetime requirement has actual runtime evidence; I do not silently turn it into
a claim that all source assignment syntax or broader managed profiles work.

## My mixed managed contract4be

I retain [NANOISA_MIXED_OWNED_MANAGED_VALUES.md](NANOISA_MIXED_OWNED_MANAGED_VALUES.md)
and distinguish its five dependency stages and its lifetime/source requirements.

| Original requirement | Merged evidence | Limit |
| --- | --- | --- |
|1: binary64 scalar carrier and exact generic/typed semantics | PR798 `5079d92e`; [binary64](evidence/owned-binary64.md) | Outer public admission was separately corrected; inner-analysis support alone was never permission. |
|2: explicit owner/ordinary/observation categories and closed shape transport, exact nominal maps | PR802 `4fc20434`,805 `47ad402f`,812 `b7040dcd`; [view](evidence/mixed-layout-view.md), [shape](evidence/mixed-float-proof.md), [composition](evidence/mixed-samples-composition.md) | No fake VOID/RESOURCE clearing, no wire-bit reinterpretation; raw GET retains FLOAT-or-VOID obligations. |
|3: runtime/native ordinary flat FLOAT arrays/records alongside owners, rooted aliases and checked operations | PR819 `d0de3d23`,820 `9ac3b046`; [runtime](evidence/mixed-samples-runtime.md), [integration](evidence/mixed-samples-runtime-integration.md), [Darwin link qualification](evidence/mixed-samples-linkargs-darwin.md) | Existing scalar/owned signatures and closed finite graph only; per-report VM/native instrumentation and heap/admission fault boundaries stay explicit. |
|4: paired source descriptors, inferred aliases, original Samples/PREFIX/all selected shadows, roundtrip/parity/refusals on bothhosts | PR828 `6a993be5`; [Linux seal](evidence/mixed-samples-source-linux/manifest.json), [Darwin seal](evidence/mixed-samples-source-darwin/manifest.json), source-provider integrations; PR861 full patterns | One graph retains original close and main shadows. No source partition or declaration omission supplies acceptance. |
|5: separate managed fields inside unique owners, moves/unpack/partial construction/aliases | STRING PR813 `1cbe8c6f` and [source evidence](evidence/owned-string-fields-source.md); owner ARRAY430220 matrix above; transitive scalar wrappers PR801 `f156daba7` | Distinct STRING, ordinary Samples and owner ARRAY profiles are separately qualified. No claim of arbitrary profile composition, ordinary managed CALL/result expansion, cycles or imported/linked managed execution. |
|Lifetime and refusal acceptance: exact roots across locals/stack/projection, typed F64 checks, prepared/pending owner calls, all four VM APIs, failure/recovery, wrong shapes/nominals/duplicates/unknowns | Private composition sweeps; runtime12 cases and heap/admission injection; source62-method Linux/Darwin qualification plus owned graph/refusal adjacencies and later full original patterns | The runtime report explicitly distinguishes heap injection, first admission allocation injection and existing graph/result preflight controls; it does not claim new mixed-frame allocation injection or every transitive system allocation. |

The four named original source families now have distinct evidence: Samples,
Bundle with STRING, Bundle with FLOAT ARRAY, and Connection's transitive wrapper.
This supports review of the bounded4be contract; it does not establish general
mutable collections, cycles, callbacks, imports or complete LLVM/Wasm targets.
Those remain required in the original51da/488a/28f2 and full-product parents.

## My installed-product e8d boundary remains open

I retain the original950f full-quick status2/1734.273s and55 checked subcase
failures. PR878's unchanged33-method script passes at3c728/851 after fresh
bootstrap on bothhosts; its authority-specific public-C refusals are retained,
not treated as missing callable/resource-union ABIs. I do not replay950f binaries.

The actual make target additionally runs nine module-identity and sixteen
generic-identity methods. The nine-method supplement is authorized under
[AFFINE_REMAINING_ACCEPTANCE.md](AFFINE_REMAINING_ACCEPTANCE.md). Generic16 waits
for coordinated a18 union/first-success integration. Complete current installed
full-quick and remaining fixed-point/product gates are still absent. Thus e8d
and full d76 remain open even if all bounded430220/4be clauses are reconciled.

## My proposed ledger action

I replace stale initial-proposal wording with this exact evidence matrix while
preserving historical entries and exclusions. I request review before using the
actual-merge helper to close430220 or4be. At this checkpoint both remain open;
e8d and all broader parents remain open. No new source ARRAY assignment code is
part of this audit.

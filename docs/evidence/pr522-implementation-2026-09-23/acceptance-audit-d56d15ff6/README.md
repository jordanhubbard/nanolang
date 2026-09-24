# My acceptance audit after the final-source fixed points

I audit the seven requirements in `docs/NANOISA_ONLY.md` without treating a
passing subset as full release acceptance. My compiler-source pin is
`d56d15ff6934dcd4872aa0f90bfe7ea828cf79fa`; documentation checkpoint `b70537583`
changes no compiler source.

| Requirement | Current evidence and limit |
| --- | --- |
| 1. Canonical `.nvm` compiler product | My canonical output/module-fact suites pass 20 methods with the selected compiler. These exercise publication, source preservation, dependency shadows, module facts and VM/native products. They are bounded evidence; final platform qualification remains required. |
| 2. Native C11 product without NanoVM linkage | My retained `native-fixedpoint-d56d15ff6` gate builds three standalone compiler generations with strict C11 flags and checks their dynamic dependencies. All succeed without `libnanovm`. The separate reconstruction report retains its structured-region limits. |
| 3. Raw Stage 1/Stage 2 equality | Both VM and standalone-native routes pass independently on this source pin. Their exact bytes, hashes, commands and host closures are retained in the sibling fixed-point archives. |
| 4. Pinned VM/AOT answers | My 32-method instrumented One IR suite and these canonical-product controls pass. Full hosted platform, coverage and sanitizer acceptance is still live and incomplete. |
| 5. No legacy transpiler in product closure | `test_canonical_source_closure_excludes_legacy_emission` passes after recursively checking the canonical source imports for `transpiler.nano` and `module_introspection.nano`. Retained bootstrap/reference source is a separate boundary. |
| 6. Full applicable LLVM/Wasm coverage | **Incomplete.** The ledger keeps managed aggregate identity (`task_488a05eb5e2a417caf83a8353363a30d`) and portable host/module linkage (`task_2d2e9eb552394f6e84e90f5aa08484e2`) open. Both MAC records were read during this audit. Their bounded implemented subsets do not close the required full coverage or equivalence. |
| 7. Evidence-based reconstruction finding | `docs/NANOISA_HL_ROUNDTRIP.md` explicitly distinguishes sufficient closed scalar regions from insufficient full reconstruction and links the original five-deliverable feasibility matrix. It does not claim arbitrary high-level reconstruction. |

My first canonical-suite invocation passes 17 methods and fails three before
program semantics because Apple ASan rejects `detect_leaks=1`. The fixtures
already honor `NANO_NATIVE_TEST_CC`; I omitted that selector. I retain this
terminal and rerun the same 20 methods with
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`. All pass with leak
detection unchanged. This is an invocation correction, not a product repair.

My hosted snapshot records run `35997348910`; running or queued jobs are not
qualified. That run and the newer documentation-head runs must be inspected
again before any completion claim. Even all-green CI does not discharge
requirement 6. My next scope audit must follow the existing aggregate and host
linkage contracts and current implementation evidence, preserving their full
parent requirements. I do not mark the release or PR complete.

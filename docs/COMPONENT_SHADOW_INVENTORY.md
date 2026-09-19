# My component shadow inventory contract

I continue the component audit under `task_3b21e8577b1f42f7a7654add7c2f6b6b`, after PR691 added explicit driver entry assertions. I inspect current parser, checker and transpiler declarations, their driver assertions, and the selection/execution routes in the C seed and self-hosted compiler. I record exact source identities.

My inventory distinguishes source declarations, selected tests, completed shadows and driver entry assertions. A source count is not an execution count. I inspect current ordinary source only and do not replay historical failed artifacts. I publish concrete remaining evidence requirements without closing broader task56a or changing compiler policy.

## My current findings

I inspect canonical source `3373f01b` through PR709 (the complete object ID is in my [inventory](evidence/component-shadow-inventory.json)). My lexical inventory masks strings and comments while preserving line numbers, then collects line-leading shadow declarations. It finds 240 parser, 148 checker and 165 transpiler declarations: 553 total. I retain each name, line and source SHA-256. These counts exclude transitive dependencies and are not parser-AST or execution counts.

My three stage-three drivers now execute meaningful entry assertions. Parser checks tokenization, the function/body/return AST and one parsed shadow. Checker distinguishes a valid integer return from a boolean mismatch with diagnostics. Transpiler checks generated body text, selected shadow-call text and invalid selection refusal. PR691 and the frozen-candidate gates provide bounded execution evidence for these drivers. Their assertions do not invoke every component shadow.

My current selection policy differs from the historical task56a description:

- C seed `src/main.c` defaults `test_imports` to true. Its ordinary interpreted route calls `run_shadow_tests_scope`; `src/eval.c` iterates selected imported modules and then the root, executing every explicit shadow. The former foreign-call exemption is absent. A callback-adapter module instead selects `src/nanovirt/shadow_runner.c` and its VM path.
- Self-hosted `src_nano/nanoc_v06.nano` also defaults `test_imports` to true. It starts at shadow zero unless `--root-shadows-only` selects the root suffix. Native output runs the generated shadow executable before publication; bytecode output assembles/verifies and runs the selected shadow module before publication. Explicit C-text emission reports that it has not executed shadows.
- Stage two invokes the currently selected `bin/nanoc` without a root-only override. Stage three executes each driver binary and reports entry assertions. Its component logs do not by themselves enumerate every selected/completed shadow, and the selected compiler identity must be retained with an execution claim.

I therefore do not repeat the obsolete blanket claim that current C seed skips extern-dependent shadows. Historical evidence remains dated to its original source. I also do not infer a 553-test pass from three entry checks or from this static selection review.

## My remaining acceptance

Broader task56a remains open for exact-source, selected-compiler execution evidence: retain a dependency-aware shadow inventory, selected/completed counts or traces for each component, final completion status and source/tool identities. Report root-only selection and explicit C-text emission separately. Any current-source refusal or failed assertion must be preserved and entered in the roadmap before a repair. Existing historical failed artifacts remain untouched.

This inventory is documentation-only. I check its source hashes and lexical counts, compare the routes above with their full implementations, and run `git diff --check`. I make no new runtime claim and do not close compiler correctness or release obligations.

## My later completion evidence

My [canonical execution report](CANONICAL_COMPONENT_SHADOW_COMPLETION.md) supplies the remaining exact-source selected/completed evidence at product e9a5f55f. Together these reports complete the original component-execution task56a. Earlier open-status statements above describe the evidence available when this report was written; later source changes and full release acceptance remain separate.

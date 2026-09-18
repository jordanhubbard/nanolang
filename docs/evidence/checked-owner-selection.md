# My checked owner-free selection qualification

I qualify task_fa08b9ffba9c40f0bb4ce22a63321589 at integrated source `9b87b18ca6656e5cde472d9cf56872c881a1be5b` (PR #788). My [contract](../NANOISA_CHECKED_OWNER_DEPENDENCY_SELECTION.md) preceded production. The merge includes main `df51f4cf`; only the roadmap needed conflict resolution. The later array-slice component is outside this pin and needs combined product qualification separately.

I select my ordinary program/shadow queue only after a positive bounded scalar-closure analysis. Complete source checks, original function/lexical identities, initializer ordering, all selected shadows, and raw full-module/C-seed NanoVirt behavior remain. Unknown or selected ownership keeps specialized lowering. I add no foreign, managed-field or ownership authority.

## My final integrated checks

| Check at 9b87b18c | Result | Elapsed |
| --- | --- | --- |
| Fresh isolated bootstrap | PASS, both stages and installed smoke | 259.259s |
| Required tools | PASS | 0.368s |
| Checked selection | PASS, three methods | 154.731s |
| Complete source-borrow gate | PASS, all 46 methods | 404.057s make / 376.325s unittest |
| Owned-call graph gate | PASS | 17.153s |

The checked-selection method reuses all 36 original declaration-probe sources unchanged: 14 accepted and 22 refused. Each runs through a driver compiled by C-seed, Stage1 and Stage2 that invokes full-source `typecheck_phase_with_shadows` before emission. Accepted assembly is identical across those producers, contains no unselected `consume_handle`, and verifies/executes in NanoVM and strict sanitized native C. The 14 accepted sources also run through actual Stage1/Stage2 `--emit-nvm` publication and VM/native execution. These C-seed-produced canonical checks are distinct from the separate all-definition public C exporter.

Additional controls cover initializer-before-binding, restoration of an outer binding, exact callable shadowing, lower-index transitive calls, recursion refusal, shadow-only owner/extern dependencies, complete/suffix shadow selection, invalid unused ownership/types, prior-output preservation and retained raw-module refusal. The existing ownership gate retains its metadata/name checks, false-shadow and cleanup controls, the unchanged affine example, and all selected-owner/global/full-module refusals. Three exact old owner-free shadow refusal expectations became verified positives; a false owning-shadow control still prevents publication.

The graph gate reports 1847 graph checks, 338 preflight checks, 529 invocation-proof checks and 69 verification-reuse checks, plus its VM/native Python acceptance. These are the existing gate's scopes, not a new claim about every allocation or authority path.

My runner records a clean fixed head and 1708 tracked source/test/build inputs unchanged before/after. All compiler/runtime/spec sources remain frozen during this run. The five tools present after bootstrap remain unchanged through the final gate; five additional tools are built, leaving 10 recorded tool hashes. The checked driver hashes are recorded in the focused log; Stage1/Stage2 driver hashes agree. GCC native executions retain `-Wall -Wextra -Werror`, ASan/UBSan and leak detection through the existing paired harness.

## My preserved earlier outcomes

| Pin | First outcome retained |
| --- | --- |
| 32b91d52 | Fresh bootstrap PASS 257.456s; tools stop on nonexistent `bin/nano_asm` target before selection tests. |
| 29133945 | Tools stop on nonexistent `nano_asm`; source inspection establishes `nanoisa_dump` / `bin/nanoisa asm`. |
| 6ae02fa1 | Tools PASS; unintended imported TestCase discovery runs 69 methods, ending with 14 failures and 105 missing-prerequisite errors in 324.480s. Nine failures are new fixture PARSE errors; three are old owner-free shadow refusal expectations; two are separate existing public C-profile limitations. |
| 0989576c | Corrected intended three methods PASS 154.282s, unchanged compiler/stage hashes. |
| f6d2eb4e | Complete 46-method run: 45 methods pass, one method has two stale Stage1/Stage2 no-transfer subcase expectations; terminal failure retained (405.034s make). |
| 4abe5e94 | Reviewed affected method PASS 167.049s and graphs PASS 16.859s, unchanged compiler/stage hashes. |

I preserve the two extra public C diagnostics exactly in the first log: indirect callable values are outside that C profile, and the owned-union return payload is outside its scalar result profile. These are existing reviewed profile limitations, recorded under installed-product task_e8d860a16da0464891dd32e91c42bef1, not new ABI requirements or defects established by this selector. I neither relabel any failed run as passing nor execute its refused output. The final integrated full 46 pass is a separate later result.

I retain [all raw reports and hashes](checked-owner-selection/reports.sha256), their [derived summary](checked-owner-selection/summary.json), exact runners, manifests, tool hashes, and source inventories. Original `/tmp/nanolang-checked-owner-selection-*` directories remain; the earlier qualified worktree/tools remain at `evidence/owner-selection-qualified-4abe5e94`.

Full product-affine e8d860 and normative ownership parents remain open. In particular, ordinary transitive owned wrappers and mixed owned/string/array fields are separate prerequisites. Even an owner-free Samples program has an owning PREFIX shadow in its original fixture; this checkpoint does not partition or omit that shadow, widen mixed runtime support, or claim full 55-case product closure. Canonical ancestry precedes MAC completion.

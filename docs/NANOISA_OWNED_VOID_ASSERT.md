# I retain exact VOID helpers and ordered assertions

I advance #999 from `9071eadb9` with the existing `tests/nanoisa/fixtures/source_owned_value_graph.nano`. Both retained producers accept, original-verify and execute that unmodified fixture with result zero. Reconstruction currently refuses its VOID consumer; existing INT-result helpers also contain ASSERT. My retained mapping is evidence of an existing language path, not a proposed language feature.

## My unchanged authority boundary

I require original nvm_verify admission before reconstruction facts. Existing affine consistency checks and the acyclic direct value-call graph remain mandatory but confer no independent execution authority. I keep an argument-free INT entry, exact mode-zero INT/BOOL/resource parameters, scalar-field resource layouts, exact numeric nominal identities, existing budgets and atomic publication. VOID is permitted only as a helper result, never as a parameter, local, operand value or entry result. Borrowed/indirect/recursive/module/extern calls remain outside this slice.

The existing nvm_affine_value_result API already checks VOID against result_count zero and exact no-layout identity. I expose and check result counts explicitly: zero precisely for VOID, one for the existing value results. I do not weaken that API or the verifier.

## My call, return and assertion semantics

I evaluate arguments in the retained instruction order and consume each owned argument exactly once. A zero-result CALL becomes an explicit ordered call statement, pushes no operand and creates no fake value or result temporary. A VOID RET requires an empty operand stack and no live owners, then emits a bare return. Value-return behavior is unchanged. Branches and loops keep the shared region tree and existing ownership state rules; a successful VOID return is terminal in the same sense as a successful value return.

ASSERT pops exactly one BOOL and becomes an ordered statement. Its operand is evaluated once at its original position. I retain existing scalar temporary materialization and do not move an assertion across calls, ownership transfers, reads, stores or another assertion. Native failure must remain active under NDEBUG; I will use an explicit failure branch rather than the C assert macro. Nano emits its language assert statement. This slice retains assertion failure as a trap, not as a successful return or proof of owner cleanup. I do not clear live-owner state or permit an otherwise-invalid successful exit merely because a predicate may be false. All successful exits still consume owners. An actual failing assertion may stop before later cleanup; that is the source program's trap behavior, and no cleanup-success claim follows.

I require semantic failure on false assertions in original VM and reconstructed outputs, but do not invent equal process exit codes across VM/native engines. I retain exact actual statuses and diagnostics, including native signal exits. I preserve original files on any reconstruction refusal.

## My acceptance plan

I qualify the unmodified source_owned_value_graph fixture from retained C and changed-source C-hosted Nano producers. I compare original VM, structured C at O0/O2 with ASan/UBSan, reconstructed Nano through both producers, and its VM/nvm2c native execution. I check exact nominal layouts, parameter types, result tags/counts, zero-result call behavior and absence of VM/goto/dispatch output.

I add source-driven zero-result calls in structured branches/loops, explicit VOID returns, and assertions interleaved with observable scalar/owned helper results so order and once-only evaluation matter. An executed false assertion before later owner cleanup must fail in VM/native, including native NDEBUG builds, with exact diagnostics/status retained; its generated success-path ownership obligations stay checked. I retain negative controls for VOID parameters/entry, inconsistent result counts, spurious return/call operands, missing owner consumption, non-BOOL assertions, corrupt authority and atomic output preservation. CRC-valid mutations of actual producer modules exercise original admission; supplemental analyzer tests do not replace it.

Original ownership/local/call/branch/loop suites remain required. Fresh aggregate stages, borrowed paths, nested resource/union/array layouts, broader loop conditions and full #999 remain open. The integration owner approved this contract before production changes and reviewed the resulting narrow production diff without a blocking finding.

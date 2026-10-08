# My legacy global-name retirement

At `1321b8bdf`, my direct `typecheck_phase` probe accepts `module_aliases`,
`module_alias_targets`, `func_aliases`, and `func_alias_targets` without source
declarations. It rejects an ordinary undeclared-name control. I retain that
probe and output, plus the failing executable component regression.

I remove the four injected symbols. Their ordinary declarations still enter
my existing global-symbol pass. My first correction rejects the undeclared
names but exposes a zero environment error count despite error diagnostics.
I retain that failure. I now include diagnostic errors in the reported count,
while preserving larger existing failed-check counts. Warnings do not count.

My checker component tests each name both undeclared and declared, asserting
error status, error count and diagnostic presence. Its shadows execute those
assertions too. The existing accepted integer and rejected boolean-return
controls remain. Both the C-seed and retained self-hosted producer compile the
changed checker component; both modules pass in NanoVM and strict C11 native
execution under ASan/UBSan/LSan. All ten recorded steps exit zero.

I qualify the changed checker through a temporary driver whose import selects
the sparse worktree's exact checker source; other dependencies come from the
unchanged `1321b8bdf` primary tree. My manifest records every command, status
and duration; `inputs.json` records changed-source and tool hashes. This is
component evidence, not a claim that the new product compiler bootstraps.
My clean sparse-checkout raw bootstrap at `b76b33a53` now passes in
670.100 seconds, with unchanged HEAD and clean tracked source. Stage1 and
Stage2 are byte-identical: 492,432 bytes, SHA256
`a729c1d528ecd8dc29621583fa2da855df6a4ff0f3433817512c03657213642c`.
I retain the full gate, runner, terminal manifest and bootstrap receipt.
Integration and final release-candidate qualification remain open.

My broader retirement audit still finds direct legacy emitter imports in
`compiler_modular.nano`, `driver.nano`, `transpiler_driver.nano`, the extern
selection test, the old shadow-emitter fixture, the File parser fixture and
the unchecked-match backstop fixture. Source-text tests also inspect lexical
environments and array-slice helpers in the old emitter. I must migrate their
behavioral coverage before removing that source; I do not delete acceptance
coverage to make retirement pass.

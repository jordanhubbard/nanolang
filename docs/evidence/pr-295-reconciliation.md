# My PR #295 reconciliation

I merge main `436a2caa0f76d761192b2b947078d628384a0259` into integration
`1e152e06`. The reviewed worker head
`64835ab0dd6c396d7e8ec5547a1e123e772165d2` has the same tree as that main
commit; I also retain it as an ancestor without changing source again.

The implicit-exit `free(owed)` fix is already present from `b770cd03`.
The normal merge retains my additional unknown-effect cleanup and adds the
worker's rejection regression after two pushes and balanced retain/release.
I keep that test, but clarify that rejection alone does not measure leaks.

I extend `test_verifier_cleanup.c` with the same decoded instruction path.
The allocation-counted probe requires failure, an unchanged caller depth
output, and zero outstanding verifier allocations. Existing allocation-failure,
unknown-effect, successful-return and simple implicit-exit checks remain.

`make schema-check test-verifier` exits zero. The allocation-counted probe
passes and the verifier reports 95 tests passed. The final host-local log is
`/tmp/nanolang-pr295-reconciliation-final.log`. This is not a full leak
sanitizer run or an ownership proof; balanced counts do not establish object
identity. Full release acceptance remains open.

MAC `task_69dab2ee6f1a4c96845ba7139cfc360f` is completed when inspected.
I attach integration evidence without changing its ownership or status.

# I qualify owned operand-stack binary64 execution

I qualify native task `task_cb823e6c37e6465e8d6ad10f3b726364` and public admission
companion `task_77952f5657c94f0181354a8193d0d5ec` at source
`2e3b649b46afcd512cc1f1ae5c5304aa6fc1ada9`. My native production is d8212ebd;
my separately reviewed outer opcode whitelist is a62a6baf. I do not claim that
inner affine analysis previously established public execution admission. My
[corrected contract](../NANOISA_OWNED_NATIVE_BINARY64.md) preserves float local,
parameter, result and resource-field refusals.

| Gate | Outcome |
| --- | --- |
| Fresh focused GCC | PASS, 3.975 seconds; 1,098 C checks, four public VM APIs and four declaration-refusal controls. |
| Pinned Clang 23 focused | PASS, 4.976 seconds; same corpus and assertions. |
| Combined verifier/affine/reference/owned/native regression | PASS, 204.326 seconds. |

My focused modules exercise 588 generic/typed comparison combinations over both
signed zeros, finite values, infinities and NaNs; arithmetic/nonfinite operations,
stack copies, branches and loop-carried floats; a scalar-result helper; real live
owners; and assertion failure cleanup. Generic NaN LE/GE retain the VM three-way
ordering result, distinct from typed F64 predicates. The generated native helper
also checks nine transported bit patterns and five shared arithmetic policy
results. I do not present those helper bit checks as public float-result admission.

For each of four valid modules I qualify all four VM APIs twice, then native
strict C11 compilation, ASan/UBSan with leak detection enabled, eight repeated
invocations and an owner-allocation failure sweep reaching a non-memory terminal
outcome. I preserve exact cleanup counts and output sentinels. My VM fixture uses
an ordinary C build; the generated native programs are instrumented. This does
not claim a new fully instrumented VM corpus or frame-allocation injection.

The combined gate includes 96 verifier cases, 751 affine checks, 1,548 caller
reference checks with 43/55 allocation controls, 959 assertion lifecycle checks,
1,847 value-graph checks, 338 graph preflight, 529 invocation-proof and 69 reuse
checks. Owned results pass 1,552 checks, 3,485 allocation checks covering 312
budgets/272 actual faults, and 117 return-preflight checks. String controls pass
600 allocation and 150 proof checks with nine admissions. The native suite passes
2,422 checks; its shape, opcode-inventory and sanitizer-driver prerequisites pass.

My [39-report manifest](owned-binary64/reports.sha256) seals raw logs, statuses,
source manifests, the runner, compiler identities and the exact Clang wrapper.
All 1,596 tracked native source/test/build inputs agree before/after each final
gate. My three core tool hashes were captured during the regression and verified
unchanged after its terminal outcome; I do not claim that capture preceded the
focused run. The qualified tree and tools remain at
`/home/jkh/Src/nanolang-owned-native-binary64`.

I preserve these earlier outcomes without relabeling them:

| Pin | First terminal |
| --- | --- |
| cd8c7f6c | Strict compilation refused misleading indentation in the new fixture; no fixture ran. |
| fe190992 | The new FLOAT local was refused by the outer profile before execution. I corrected the contract/test to operand-stack values. |
| e7cb4fce | The outer runtime opcode whitelist refused F64 before execution. I recorded and reviewed the explicit admission companion. |
| 35f03637 | Declaration refusals passed; the new API selector expected stack publication from the wrong wrapper. Static review also fixed its arithmetic expected-index selector. |
| 29dc1d21 | Corrected focused gate passed, 3.926 seconds, before adding explicit nonfinite arithmetic controls. |

I keep mixed managed Samples/PREFIX execution, managed fields inside affine
Bundle, source float admission and full ownership/release parents open. No
historical product artifact was replayed, and no refused module was executed.
Canonical ancestry precedes bounded MAC task completion.


I restack in a separate tool-free ready tree at merge c3ba0fdd over canonical
main dd0fed57 (PR797). The only changed qualification input is the unrelated
four-line export-buffer test portability correction; all production files and my
owned-binary64 tests/build target are unchanged. I verify the other 1,595 input
hashes and all 39 sealed reports. I do not claim a repeated native gate at the
restacked pin. My original qualified tree/tools remain preserved; this final
restack changes no relevant executable implementation.

# My explicit global ownership boundary

I reject global declarations whose concrete type contains a resource. This is
an unsupported-global-lifetime diagnostic, not a global ownership model. I do
not assign the same global owner independently to every function's local flow.

Both C global registration paths call my ownership classifier after checking
the initializer. My self-hosted pass identifies nonlocal lets using the same
local/parameter/block mask as global registration, then classifies the declared
and available initializer identity. Resource-bearing empty union arms are
conservatively rejected because the declared global type can hold an owner.
An unused resource generic argument does not make an ordinary payload affine.

My paired driver tests cover mutable/immutable and inferred globals, nested
records, fixed/generic unions, empty resource union arms, collections and an
imported global. Rejections require my explicit diagnostic and preserve the
previous output artifact. Ordinary scalar/string/union globals and function
local owners remain executable controls.

## My initial checkpoint

At `80a61649`, I completed a fresh full bootstrap,12 C methods (4.462 seconds),
12 Stage1 methods (9.773 seconds), and44 paired global/affine/selected-ownership
methods (125.839 seconds). An isolated ASan/UBSan harness repeated11 nonimport
parser/typechecker cases20 times:220 lifetime checks passed. Imported global
behavior is covered by the driver gate. Leak detection remains disabled under
`task_00c47a5d65d04c48914864ec0de553d6`; this is not a leak-freedom claim.

Local logs are under `/tmp/nanolang-global-boundary-audit/`: `bootstrap.log`,
`c-tests.log`, `stage1-tests.log`, `paired-adjacent.log`, and `asan.log`.
Integration with generic selected ownership remains the next acceptance gate.

My original probes also found native record-global emission failures. That
work remains separate under `task_95796f5f49564ed4a911fd05a1aac5b4`. Ordinary
generic union globals provided an executable control for this guard; I do not
claim that all ordinary global aggregate lowering is repaired.

MAC: `task_8afaef937f934a6e9919e41b91b7a41c`.

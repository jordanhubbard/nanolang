# Ordinary HALT baseline before private union execution design

Source/provider head: preparation checkpoint `5b899da28d4b8abd245203f700a448bea78f5cb2`. No union executor or public admission changes. The parent approved this bounded baseline after aggregate integration. All existing preparation sources/products remain unchanged.

The test-only C harness assembles nine ordinary modules without union authority, independently calls `nvm_verify`, and executes only verifier-successful modules. Eight use actual public `vm_execute`; one is explicitly a manually established core frame calling `vm_core_execute`. Every module verified. Each first invocation is printed before any teardown; there is no dirty-state retry. Exact NASM and serialized module bytes are retained.

| Case | First observed outcome | Stack / frames |
|---|---|---|
| ordinary return | VM_OK, INT7 | 1 / 0 |
| root HALT with local | VM_OK, local9 and operand11 remain | 2 / 1 |
| CALL helper HALT | VM_OK, INT12 | 1 / 2 |
| root TAIL helper HALT | VM_OK, INT13 | 1 / 1 |
| true ASSERT then HALT | VM_OK, INT14 | 1 / 1 |
| false ASSERT | error8, exact `Assertion failed` | 0 / 1 |
| initializer HALT then entry | VM_OK; entry writes global99; INT16 and INT7 remain | 2 / 1 |
| normal INT initializer then entry | VM_OK; global99; INT16 and INT7 remain | 2 / 0 |
| core-only HALT | TRAP_HALT4, INT17; not a public invocation | 1 / 1 |

All cases finish `vm_destroy` with zero heap objects, zero intern count and zero frames, and equal allocated/freed heap payload statistics. This is live heap-owner accounting, not an independent allocator leak oracle. ASan leak detection is disabled under the existing macOS gate convention; address/undefined checks remain fatal. No sanitizer diagnostic occurred.

Actual compile0.638s, execution0.427s, using Homebrew Clang23 O1 ASan/UBSan. `build.argv.json` pins the complete original permanent linked closure. Only the new harness was compiled; no provider/frontend/bootstrap rebuild or installed product replacement. `pins-before.json` and `pins-after.json` match exactly. The compiler executable and every input object/archive/source are hashed. The retained original provider command is adapted by replacing only the harness, removing its unused allocation support TU, and changing the executable destination.

Source interpretation: `OP_HALT` clears references and yields TRAP_HALT; ordinary call supervision collapses HALT to VM_OK without normal-return checks. The scoped wrapper unwinds only errors, so HALT retains frames. Ordinary `vm_execute` proceeds to entry after initializer VM_OK. The private union supervisor must therefore explicitly specify its selected outcome/cleanup policy rather than claim ordinary HALT already behaves as a clean return or whole-invocation stop. Existing map supervisors reject HALT, which also is not a qualified union policy.

Health before run:46GiB free (98% used),62% memory free,22% CPU idle. The small serial harness used existing frozen providers; cleanup inspection found151MiB of retained preparation evidence and no deletion was performed. First read-only lookup errors for guessed log/temp paths were corrected; compilation and execution succeeded on their first attempts.

# My explicit scalar artifact consumers

I add explicit `.import_kind index declared_scalar_artifact` transport and
verification, exact-handle VM resolution, and standalone native libffi adapters.
Every declaration carries a bounded signature. Native string results are copied
before caller arguments can be released; the same-image companion receives the
original result once, including null and failed-copy paths. Kind 2 retains its
named native adapter boundary. Generated C requires libffi and has no NanoVM
runtime dependency.

On Darwin I pass:

- 24 ordinary VM / instrumented generated-native artifact methods: all 11 prior
  kind-2 tests and 13 kind-4 tests. These include mixed 16-argument calls, scalar
  results, void side effects, signed 64-bit enum round trips, borrowed aliases,
  same-image ownership, cleanup counts and copy allocation failure.
- Two additional paired controls for dump/reassembly fidelity, missing symbols
  and null borrowed results.
- 913 bridge checks, 98 verifier tests and the complete 2,582-assertion native
  translator gate. The owning Make target now includes the new Python suite.

The Python runs select Homebrew LLVM with a temporary `cc` symlink. Generated
native executables use ASan/UBSan and leak detection. The VM, providers and full
translator gate are ordinary builds; this is not full sanitizer qualification.

I retain the intermediate failures: an incorrect Make target left the assembler
stale; Apple's SDK needs `ffi/ffi.h`; my initial byte extraction lost its tag,
and the first correction named the tagged-storage member incorrectly. The
corrected byte/enum cases and all paired checks pass without weakened assertions.

Source emission, libffi linkage through the actual compiler driver and installed
package, unchanged native module-linking acceptance, Linux execution and full
instrumented attribution remain required. This checkpoint does not complete
`task_b8838417bbc54fb98a4c49eea1b0885a` or #522. The public LLVM/Wasm host contract
remains separate and incomplete.

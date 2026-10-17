# My FFI service-scan diagnosis

I investigate #982 on top of `6fc4bca180569bcd23bb55cf8fd9ceea265dd8e2`.
Hosted run38690873098 times out in instrumented Stage1 after7200 seconds.
Its retained Stage1 log is empty; it contains no sanitizer finding.

My five-second Darwin sample of the unchanged optimized Stage1 VM attributes
422 of443 samples to `nvm_file_instructions_present` from `vm_ffi_call_vm`.
Every foreign call scans every function in the compiler module. I deliberately
interrupt this diagnostic baseline after sampling; it is not a terminal build.

I reuse the VM's existing invocation-local immutable File-opcode fact at the
immediate FFI boundary. I refresh mutable bindings before dispatch and recheck
them inside the private entry. Public FFI entry, absent facts and mismatched
module identity retain the full query. No fact is retained across public
invocations or used as ownership or callback authority.

The corrected five-second Stage1 sample has no File-opcode scan samples;
402 of407 samples are in the VM execution loop. These are sampling observations,
not an end-to-end speedup claim or hosted sanitizer acceptance.

## Checks

- All29 actual FFI tests pass, including fresh mutable bindings, changed code
  at public entry and mismatched private fact refusal.
- Existing switch/threaded admission tests pass with ASan/UBSan in the actual
  VM translation unit. Their host hooks are modeled; they do not replace the
  actual FFI tests.
- Strict Clang and GCC16 compiles pass:222 sources each.
- Fresh bootstrap and hosted sanitizer qualification remain pending.

I do not change sanitizer settings, shadow deadlines, generation equality,
service refusal rules or release scope.

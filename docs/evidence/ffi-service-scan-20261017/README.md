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

## Active module mutation review

My first candidate completes Stage1 in528.83 seconds, verifies its module,
builds its native compiler and passes native hello compilation/execution.
I interrupt Stage2 when review finds an exception to the assumed immutable
code boundary: `vm_sync_new_functions` explicitly supports active callers,
and `vm_rebuild_module` can replace decoded code during a host call.

I now bind the cached opcode fact to the VM mutation revision plus code/table
identity and lengths. Invalidate, rebuild and append APIs advance that revision;
a saturated revision forces fresh scans. Mutable service bindings remain fresh.
Direct byte writes must follow the documented invalidation/rebuild API; a fresh
public invocation always obtains new facts regardless. The shared classification
comment now permits the immediate boundary check and forbids retaining its facts
across host or bytecode execution.

My actual native FFI regression reenters a helper, then either invalidates and
rebuilds existing code or appends a File instruction through the supported active
append API. Both refuse before guest execution resumes. Temporarily disabling
revision advancement makes the real rebuild control fail; restoring it passes
all30 actual FFI tests. Expanded switch/threaded modeled-host controls cover
both mutations with and without reentry, plus revision saturation, and pass
under ASan/UBSan. I retain an initial observer-count fixture failure: rebuilding
also invokes the observed predicate, so the corrected count baseline is captured
after the rebuild rather than before it. Assertions still require a new scan.

A fresh full bootstrap of the corrected revision guard is running. Neither the
interrupted first candidate nor the focused controls replace its terminal or
hosted sanitizer acceptance.

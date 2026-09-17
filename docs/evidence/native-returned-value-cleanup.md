# Returned native-value cleanup

I keep returned record-array handles alive for the generated program's lifetime,
then free each allocation once after my entry function returns. An intrusive
ownership link belongs to the allocated container, so copying an alias does not
register a second owner. This follows my existing primitive-array cleanup policy;
it is not a new tracing collector or a bound on live program memory.

I allocate `string_from_char` results through my existing owned-string allocator.
I emit that storage and cleanup even when character conversion is the program's
only string operation. The byte-conversion and retained-result semantics stay the
same for both accepted AOT import names.

My five regression cases cover a returned record list without setters, alias
mutation, a discarded character string with no other string operations, and
repeated conversion through each AOT alias. They compile strict C with ASan and
UBSan; Linux explicitly enables LeakSanitizer. An allocation/free probe also
requires zero outstanding allocations after generated main returns, preventing
an owner list from merely hiding a leak through reachability.

Before this change, both record-array cases leak 932,360 bytes across 65 calls;
the three string cases leak 2, 130, and 130 bytes. All five repaired cases pass.
Four canonical cases also pass NanoVM. The legacy unprefixed `string_from_char`
raw import is currently AOT-only: I recorded the VM availability difference as
`task_dbe0c69106984f22b59c241f3be08919` without claiming cross-backend parity for it.

Local logs: `/tmp/nanolang-native-cleanup-baseline.log`,
`/tmp/nanolang-native-cleanup-targeted.log`, and
`/tmp/nanolang-native-cleanup-gate.log`.

My integrated native gate passes 2,215 checks and 1,092 shape checks. The five
cleanup cases and unchanged full compiler bytecode-to-native-to-program test
pass together in 84.842 seconds. I distinguish this native bridge from the
still-open NanoISA-only bootstrap fixed point. Integrated logs:
`/tmp/nanolang-native-cleanup-integrated-gate.log` and
`/tmp/nanolang-native-cleanup-integrated-compiler.log`.

After rebasing onto main `4aa0edb6`, I rebuilt the compiler/runtime and reran
both tests successfully in 84.260 seconds. Final log:
`/tmp/nanolang-native-cleanup-final-compiler.log`.

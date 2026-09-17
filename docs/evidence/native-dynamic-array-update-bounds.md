# Tagged native array-update bounds

I validate the complete signed 64-bit index before narrowing a tagged array
update. Negative and out-of-range writes stop execution instead of silently
returning the array or wrapping a large index into a valid slot. Valid writes
retain the same handle and remain visible through aliases.

My 21 regression cases cover integer, boolean and string arrays with indices
zero, minus one, length, 99, 2^32, INT64_MAX and INT64_MIN. All 21 bytecode
fixtures agree between NanoVM and native output: three valid alias updates and
18 rejected writes. I preserve the existing separate array-read behavior; its
index/tag semantics remain a recorded follow-up.

My native gate passes 2,257 checks and 1,092 shape checks. A fresh build also
passes the unchanged compiler bytecode-to-native-to-program test in 83.922
seconds. I do not infer NanoISA-only bootstrap completion from this bridge.
Logs: `/tmp/nanolang-native-dynamic-bounds-gate.log`,
`/tmp/nanolang-native-dynamic-bounds/parity.log`, and
`/tmp/nanolang-native-dynamic-bounds-compiler.log`.

After rebasing onto main `4aa0edb6`, I rebuilt and repeated the unchanged
full compiler bridge successfully in 82.324 seconds; final log:
`/tmp/nanolang-native-dynamic-bounds-final-compiler.log`.

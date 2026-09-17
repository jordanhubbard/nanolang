# Projected string call storage

My current compiler previously stopped before native C emission because
`check_expr_node` passes a projected `NSType.name` into `tc_is_opaque_name`.
The projection can retain tagged storage, while flat call facts described a
string. I equated the parameter with the producer too early, then rejected the
later optional representation as an exact-string conflict.

I now seed inferred string reads with directed storage facts and convert
string call arguments into parameter storage. I preserve exact constructor,
array and map payload constraints. My shape solver and its exact-destination
rejection guard are unchanged. Once graph facts converge, existing emission
selects tagged parameter storage and checks string consumption at runtime.

A small nested-record projection into a string-length helper reproduces the
old failure. Twenty-seven new native cases cover caller and function order,
ordinary and tail calls, typed optional and dynamically tagged sources,
unchanged concrete string-array producers, taken/untaken wrong-tag paths, and
an incompatible exact optional payload that must remain rejected.

My native gate passes 2,386 checks and 1,092 shape checks. Twenty-six actual
VM/native cases agree: 21 complete successfully, including ASan/UBSan with leak
detection, and five reject invalid executed string consumption. The unchanged
full compiler bytecode-to-native-to-program test passes after a fresh build in
85.398 seconds. This restores the native bridge; it does not prove a
NanoISA-only bootstrap fixed point.

Evidence is retained under `/tmp/nanolang-native-shape-convergence/`:
`minimal.nasm`, `minimal-baseline.log`, `gate.log`, `parity.log`, and
`full-compiler.log`. The original full-module provenance trace is
`/tmp/nanolang-shape-convergence-trace3.log`; temporary diagnostics were removed.

After rebasing onto main `0072c05a`, including canonical artifact imports, I
rebuilt the compiler/runtime and repeated the unchanged full native compiler
gate successfully in 89.096 seconds. Final log:
`/tmp/nanolang-native-shape-convergence/integrated-full-compiler.log`.

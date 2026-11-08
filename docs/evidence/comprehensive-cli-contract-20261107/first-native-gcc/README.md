# My first original checker CLI execution

Actual instrumented GCC run 42453097157 at b7e4bf3667f0850f1318b1105b5955273239cae4 compiles the unchanged original resource checker through fresh C-seed and NanoVirt providers. Its C-seed-native program passes all four exact contracts: invalid arity 2, accepted input 0, ownership refusal 1 with its ownership diagnostic, and extra-argument refusal 2.

My VM invocation omitted the required -- guest separator. NanoVM correctly refuses the supplied input path as a second module, and the native control fails instead of accepting a wrong ownership diagnostic. I retain every original command, status and diagnostic, the full compile logs, exact source tree and unchanged providers. Complete original binaries remain in the private CRC-checked archive identified by archive.json. I correct both the comprehensive runner and native control with the required separator, retaining all expected statuses and source assertions. The fresh actual VM and complete four-route reports remain required under #1088/#1081.

Native checker compilation uses the existing shared host-build budget and process-group supervisor in the correction, preserving partial output if compilation times out. Guest execution keeps its separate original total deadline.

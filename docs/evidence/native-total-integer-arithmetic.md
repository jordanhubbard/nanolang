# I preserve total native integer arithmetic

At production source `b6e5ed8d`, my ordinary standalone C emitter performs
integer add/sub/mul/neg on unsigned bits, then reconstructs the signed result
using only representable signed conversions. Division and remainder guard both
zero and the minimum-integer/negative-one case before evaluating the C operator.
I retain operand checks, comparisons and Boolean operations. The separate owned
emitter already uses wrapping intermediates and guarded division; it is unchanged.

My new gate checks 38 semantic boundary cases across typed and generic integer
opcodes on VM and generated C. GCC and strict Clang pass at O0 and O2 with UBSan
and immediate failure on any sanitizer finding. The shared typed boundary method
also passes VM, C, LLVM interpretation, optimized LLVM, native LLVM and freestanding
Wasm. The new and shared methods pass in 0.562 seconds; the Clang method passes in
0.549 seconds. My full native gate passes 2,422 checks and 1,092 shape checks.
The repeatable focused target is `make test-native-total-arithmetic`.

I found this gap by reading generated-C emission logic; I did not execute an old
failing artifact to reproduce signed overflow. My first edit script stopped at
an unmatched insertion anchor before writing source; the subsequent patch is
the tested production change. Logs are `/tmp/nanolang-total-arithmetic-*.log`.
This does not implement generic floating-point promotion or complete arithmetic
parent `task_66a6dd8ca51d415f9efb0f2904f85b49`, and is not a release gate result.

# My native array operand order

Hosted x64 run `35974675846`, job `107552245345`, fails the original
`order == 123` setter assertion. My [retained first failure](../setter-order-hosted/)
and generated call show three effectful arguments passed directly to a C
helper. C does not specify their evaluation order.

I use my existing ordered-call snapshot helper for ordinary scalar/nested-array
reads and writes and for record-array accesses. I bind each operand once, in
source order, before the helper call. A record setter passes the address of
its captured value, including when that value comes from a function. The
existing empty nested-array replacement retains its runtime element tag.

All seven setter methods pass against my C seed and both existing native
stages (35.017 seconds), including five new scalar/record cases checking
`array_set`, `at` and `array_get`. All nine adjacent call-order, array-ABI and
callee-snapshot methods pass (42.190 seconds).

A fresh private unoptimized Clang ASan/UBSan C seed and instrumented generated
programs pass all seven setter methods (49.058 seconds), with leak detection,
use-after-return detection and sanitizer recovery disabled. The existing
per-compile, per-program and shadow deadlines remain. The runner restores
the ordinary C seed byte-for-byte; settings and hashes are retained. This is
not a claim that every host runtime object is instrumented.

Fresh ordinary bootstrap passes Stage 1, Stage 2 and installed/no-C-seed smoke
checks. All seven setter methods then pass against those fresh stages
(33.714 seconds). This is a native bootstrap result, not a final canonical
bytecode fixed point.

In the isolated Linux ARM64 d48 checkout, strict GCC builds the changed
transpiler object and links it with the existing C-seed objects. The baseline
passes 13 of the 14 exact source fixtures and rejects the new record setter
because `&` requires an lvalue. The corrected compiler passes all 14. I retain
setup failures from parsing make's conditional recipe and from initially
placing the private compiler outside its expected runtime directory layout;
neither is a product failure. The final runner preserves that layout with
source/runtime symlinks and leaves the checkout source unchanged.

Final hosted x64 acceptance and final-source release qualification remain
open. I do not infer the x64 result from my ARM64 control.

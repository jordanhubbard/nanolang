# My scalar float lowering checkpoint

I retain float literal text through `PUSH_F64`, lower matching float operands to
strict F64 arithmetic/comparison instructions, and transport scalar float locals,
parameters, results and globals. My parser records unary-minus provenance so
`(- 0.0)` remains distinct from `(- 0.0 0.0)`; my native self-hosted transpiler
consumes the same distinction. I refuse mixed numeric operands, implicit integer
initializers for float locals, float remainder/logical operations and float
aggregate fields/arrays outside this slice.

My focused fixture compares twelve functions against C-seed bytecode: 26 checks
pass. Both modules verify and execute in NanoVM. A fresh native three-stage
bootstrap succeeds, and my C seed, Stage 1 and Stage 2 compile and execute the
fixture (one method, three compiler cases, 7.908 seconds). The fixture checks
precision, prefix/bare unary minus, signed zero versus binary subtraction,
arithmetic, six comparisons, calls and mutable global transport.

My earlier broader checkpoint passes 86 existing comparisons and 78 of 79 Python
methods. The remaining scalar-fixture method reaches native translation and
refuses `F64_NEG`; native task `task_fd4c63cf9f3e46f09ece380ce00c7a58` owns typed
F64 instructions and required scalar transport. I keep this test enabled and do
not claim complete native parity before that companion passes it.

I separately record `task_29976241a36244f7b0ce4ad75cb10b3f`: my C reference
interpreter/native `float_to_string` appends `.0` for whole floats, while my VM
cast and current self-hosted native path omit that suffix. My signed-zero test
observes the leading sign, without claiming full formatting parity. Repairing
the existing formatting contract is the immediate follow-up.

Local evidence: `/tmp/nanolang-float-gate2.log`,
`/tmp/nanolang-float-exact.log`, `/tmp/nanolang-selfhost-float-bootstrap.log`
(the bootstrap passes; its original format assertions expose the separate
suffix discrepancy), and `/tmp/nanolang-selfhost-float-values2.log`.

After adding direct formatting returns and call-return type refusal guards, eight
focused refusal/shadow methods pass in 16.822 seconds. My C-seed-built and
Stage2-built emitters produce identical scalar-fixture assembly. The additive
restack onto `ae4aa585` also passes a fresh native three-stage bootstrap and the
three-compiler fixture; the later return-type guard is covered by the focused
negative tests.

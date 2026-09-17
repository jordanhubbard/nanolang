# My scalar float lowering

I retain float literal text through `PUSH_F64`, lower matching float operands to
strict F64 arithmetic/comparison instructions, and transport scalar float locals,
parameters, results and globals. My parser records unary-minus provenance so
`(- 0.0)` remains distinct from `(- 0.0 0.0)`; my native self-hosted transpiler
consumes the same distinction. I refuse mixed numeric operands, implicit integer
initializers for float locals, float remainder/logical operations and float
aggregate fields/arrays outside this slice. Direct builtin returns remain inline;
float call results must match their declared return types.

On integrated base `5ea3adac` (including shared-borrow schema PR508 and native
float PR513), my full paired emitter gate passes 86 existing comparisons and
81 Python methods in 95.112 seconds, with native checks enabled throughout.
My focused fixture compares twelve functions against C-seed bytecode: 26 checks
pass. Both modules verify and execute in NanoVM and standalone native code.
The fixture checks precision, prefix/bare unary minus, signed zero versus binary
subtraction, arithmetic, six comparisons, calls, direct formatting returns and
mutable global transport. Refusal tests retain prior accepted output.

My fresh integrated native three-stage bootstrap passes at the default shadow
deadline. C seed, Stage 1 and Stage 2 compile and execute the fixture (one method,
three compiler cases, 8.158 seconds). Schema regeneration preserves both
`ASTCall.borrow_mode` and `ASTBinaryOp.is_unary`; `make schema-check` passes.
Earlier C-seed-built and Stage2-built emitters also produced identical fixture
assembly; native binary equality is not claimed.

I separately record `task_29976241a36244f7b0ce4ad75cb10b3f`: my C reference
interpreter/native `float_to_string` appends `.0` for whole floats, while my VM
cast and previous self-hosted native path omit that suffix. This slice observes
the leading sign without claiming full formatting parity; PR511 repairs that
contract as the immediate follow-up.

Tasks: `task_8bc58d33e73f4e24a474d3724d862c94` and
`task_ef26778894c441c2b1002128fd0a8c37`. Local evidence:
`/tmp/nanolang-float-full-integrated.log`, `/tmp/nanolang-float-exact.log`,
`/tmp/nanolang-float-bootstrap-final.log`, and
`/tmp/nanolang-float-schema-integrated2.log`.

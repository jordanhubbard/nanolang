# My string concatenation builtin lowering

I lower the existing two-string `str_concat` builtin to `STR_CONCAT`. I evaluate
its operands once, left to right. A direct return emits the operation followed
by `RET`. A declared function keeps its own call target; I refuse a local
indirect target rather than treating its spelling as a builtin.

My focused fixture records operand calls as the integer trace `12`, checks the
result `leftright`, and checks an empty-string operand. I compare its five
functions, including initialization, against my C-seed bytecode and execute
both modules in my VM and generated native code. Malformed argument counts
and non-string operands refuse output.

My C-seed frontend intentionally refuses definitions named `str_concat`.
The raw parser entrypoint preserves such an explicit function's identity and
executes its body in VM/native code; this is not canonical source acceptance.
The initially filed task `task_a7db79fc3318472094eb32dfdaa6fab6` was an invalid
bug report, resolved by the complete redefinition diagnostic retained in
`/tmp/nanolang-string-concat-gate3.log`.

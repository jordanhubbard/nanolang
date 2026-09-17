# Projected global-store facts

I collect nested record shape constraints before deciding whether a global-store
operand has a supported representation. Previously, my final classification pass
rejected an unknown projected field before the shape solver and nominal-field
refinement ran. The compiler's `ASTFunction.return_type` store exposed this order.

I now defer only unknown operands during classification. Known unsupported
representations still fail there. My generated-store emitter still requires an
integer, boolean, string, primitive array, or existing tagged value after graph
resolution; it does not cast an unresolved operand or accept arbitrary records.

My minimal two-level projection fails before this change at function 0 offset 9.
I test both function orders with integer, boolean, string, and each primitive
array kind. Unsupported record, record-array, and float projections are refused.
The twelve valid cases also pass identical VM/native programs and ASan+UBSan
execution. `make test-nvm2c` passes 2,173 native checks and 1,092 shape
checks; I rebased onto integrated main `b73f227f` afterward without an AOT
source conflict. I retain the original compiler-bytecode-to-native-to-program gate.

Fresh compiler bytecode advances to a separate local-storage conflict in
`nisa_emit_module_nasm`, function 602 offset 336: a concrete loop index and a
tagged global-array read join at local 14. I recorded
`task_9d72bb9f29074e51ad826bc26e87b13a`; full native compiler execution remains
open.

Local evidence: `/tmp/nanolang-projected-global/baseline.log`, `gate.log`,
`parity.log`, `compiler.nvm`, `compiler.nasm`, and `compiler-translate.log`.

# My VM effect dispatch

I lower handlers into scoped dynamic registrations in NanoISA. `HANDLER_PUSH`
records the operation name, a verified arm entry, and its lexical parameter
slots. `PERFORM` evaluates its arguments before entering the nearest matching
handler. `EFFECT_RESUME` delivers the arm's final value to the suspended
perform; an explicit `return` unwinds to the handler's lexical function.
`HANDLER_POP` removes registrations when the handled expression completes.

I alias outer locals into the handler activation. Parameters and arm temporaries
belong to that activation, so recursive performs do not overwrite them. Frame
unwinding releases owned values and removes registrations. I bound active
registrations and call frames independently and trap unhandled operations.

My verifier checks operation indices, local ranges, arm instruction boundaries,
argument stack depth, and the single-value resumption shape. It walks handler
arms as separate empty-stack entries, including arms unreachable through ordinary
control flow. I reject outstanding explicit retain obligations at registration:
a lexical return must not abandon them.

I test helper-call dispatch, lexical exits through intervening calls, resumed
values, ordered multiple and zero arguments, name shadowing, outer-local
mutation, nested handlers, recursive activation parameters, float signatures,
and owned string results in `tests/nanovirt/test_codegen.c`. I test malformed
handler operands and arm stacks in `tests/nanoisa/test_verifier.c`.

I run `make test-nanovirt test-nanovm test-nanoisa test-verifier schema-check`
and `python3 -m unittest tests.test_nvm2c_opcode_coverage`. These tests establish
these execution and rejection cases, not semantic equivalence or a proof of
all effect programs.

My native C source backend has its own handler implementation. My `nvm2c` AOT
translator rejects these effect opcodes; I do not claim AOT handler support.
I do not unwind an effect across an externally entered callback activation.
A handler arm can loop locally, but cannot break or continue a suspended
outer loop. Those transfers need their own specified unwinding contract.

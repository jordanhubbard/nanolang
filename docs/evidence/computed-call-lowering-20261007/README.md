# My computed-call lowering checkpoint

I lower named function references, function parameters/results and computed
callees to `FUNCREF` and `CALL_INDIRECT`. I preserve nested signature spellings,
carry function tags and parameter tags in the module, and save the callee before
evaluating arguments from left to right. A bound function in tail position uses
an indirect call followed by return. Void calls retain zero-result stack behavior.

My C-seed bytecode build passes dependency shadows, then `nvm2c` and strict C11
produce the compiler used here. Eight VM function-value methods pass, including
the original three returned-call execution/refusal assertions through an explicitly
named VM launcher. Additional cases cover global callee mutation during argument
evaluation, nested/void signatures, byte/array/record arguments, imported aliases,
and wrong-signature/non-callable refusals preserving prior output. All 24 adjacent
CLI/product methods pass. I retain a representative verified module and its VM
execution, source and compiler hashes, and the exact test logs.

My first new signature shadow rejects a valid void result because I used local
value admissibility; I admit void only in result position. My first execution
run also finds the bound tail-call lookup still selecting the direct-call path.
I retain those terminals and the corrected checks.

This does not complete native computed calls. The unchanged three-method native
suite now reaches translation and fails all three methods: `nvm2c` does not
implement `FUNCREF` or `CALL_INDIRECT` and reports stack underflow while inferring
the module. I require native callable identity, target/signature handling,
argument/result transport, ownership and VM/native parity. I retain the native
gate in Make alongside the new VM cases; I neither skip it nor restore the old
AST-to-C product backend. Complete compiler and final-release gates must be
repeated after the repair.

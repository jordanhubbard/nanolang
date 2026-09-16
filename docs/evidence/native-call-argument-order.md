# Native call argument order

I track this audit as `task_8bee9e1410ad44c2ab714b8dab98fdfa`.

My interpreter and VM evaluate call arguments from left to right. My C seed used
ordinary C argument lists for named and module-qualified calls, so the generated
program inherited C's unspecified argument order. The earlier array-search fix
covered two builtins, not this general call path.

I now bind every ordinary named-call argument to a scoped temporary in source
order, then call the target with those temporaries. I apply the same rule to
module-qualified calls. Each argument executes exactly once.

Computed calls have a separate contract: `tests/selfhost/test_returned_function_calls.nano`
checks that the computed callee runs before its argument. I leave that lowering
unchanged rather than treating its existing test as evidence for named calls.

`tests/test_native_call_argument_order.py` executes three side-effecting
arguments through an ordinary call and a module-qualified call. It requires the
trace `123` under my interpreter, NanoVM, and a native executable emitted by my
C seed.

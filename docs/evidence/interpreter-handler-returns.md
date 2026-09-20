# My interpreter handler-return boundary

I make `return` in an effect handler leave the active call that installed the
handler. The handler's final expression, without `return`, supplies the result
of `perform` and lets evaluation continue.

The interpreter carries a lexical return destination in each active call and
handler frame. It preserves that destination through ordinary and qualified
calls, prefix operators, conditionals, aggregates, field access, match subjects
and guards, loops, assertions, higher-order array callbacks, and the supported
synchronous async-call path. Partial array construction and higher-order output
are discarded before a return escapes. String and record values that survive
the abandoned call are copied or retained before local cleanup.

I did not broaden the handwritten `coro_spawn`, `coro_done`, or `coro_result`
paths. They are not registered as source builtins and remain part of the task
lifecycle work. I also do not use this interpreter result to claim native or VM
coverage. Native execution separately refuses nonlocal handler return across a
foreign callback boundary; ordinary handler-value resumption across that
boundary remains supported.

## Current-main qualification

I tested canonical main
`50777b766e124de9db716b2ff3cb633cd449f1cd` on Darwin with:

```text
make -j8 test-eval test-effects
```

The command passed 123 evaluator tests and 35 effect-system tests. Eight focused
evaluator cases cover lexical destination, final-expression resumption,
expression order, nested and string returns, recursive activation, partial
literal cleanup, static and dynamic higher-order callbacks, and 100 repeated
synchronous async executions. No handler frame remains installed after those
cases.

The retained combined log is
`/private/tmp/nanolang-handler-return-current-main.log`, SHA-256
`31ae4c02e3479c831dfa59984896276d8cdfc5bef0104c9d34bf848f8015172b`.
The selected C compiler was Apple Clang 21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.

This closes only MAC
`task_67e5e620d75a413b99753c7cdbde1f48`. My broader effect dispatch,
task-lifecycle, product, and release acceptance rows remain open.

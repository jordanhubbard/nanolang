# My exact-INT generic arithmetic reconstruction

I record `task_fe0395a88f4b4740aa8fe2898859e44c` before implementation.
I admit generic ADD/SUB/MUL/DIV/MOD/NEG only for statically exact INT
operands/results, reusing unchanged total typed integer helpers. Addition,
subtraction, multiplication and negation wrap modulo 2^64. Division truncates
toward zero; division/remainder by zero return zero; MIN/-1 returns MIN
for division and zero for remainder. I retain immutable operand snapshots
and existing pure-loop restrictions.

I do not admit BOOL, U8, FLOAT, ENUM, heap values or dynamic promotion.
I require ordinary small same-module VM/native/reconstructed-C/Nano checks,
call/local evaluation snapshots, loops and output-preserving other-tag
refusal. Historical carry679 artifacts and compiler selections remain excluded.
Full reconstruction `task_4bd034f6029b7458201db74e2c3aeb32` remains open.

My compiler snapshot is detached `f38b6409` in
`/home/jkh/Src/nanolang-reconstruction-tool-pin`. Cseed/Stage1/Stage2 were
copied byte-for-byte from the successful product checkout and verified
against its recorded SHA-256 values. These tools were not rebuilt from this
generator tree. Stage2 retains three absolute host-library cache paths in
the original product checkout. I copied those libraries separately, retain
the original paths and verify their hashes before/after gates. This is not
hermetic relocation. Local downstream translator/runtime tools are built
and hashed separately; process system libraries remain host dependencies.

My initial new fixture used unsupported `.param` syntax and was refused by the assembler before any module/compiler execution. I preserve `/tmp/nanolang-generic-reconstruction-first.log`; A first spelling correction to `.params` was also refused (`/tmp/nanolang-generic-reconstruction-corrected.log`). I inspected the existing paired fixture and used its `.parameters identity int` directive; neither setup attempt executed a module.

## My measured acceptance

At production `3c699f84`, three focused methods pass GCC in 14.320 seconds.
Those three plus the adjacent typed multiplication-loop and mocked compiler
diagnostic methods pass Clang in 15.351 seconds. The focused cases include
12 ordinary total-arithmetic cases in four small modules, call/local/loop
snapshots, and 18 other-tag inputs refused before either output is published.
I execute only correctly typed modules. Positive modules compare VM, native
C, reconstructed sanitized C and NanoLang compiled by all three copied
producers; Cseed Nano-C also has UBSan enabled.

I verified generator/local-tool/copied-compiler hashes and both original
and copied host libraries unchanged after the gates. The exact pins,
paths and logs are in [my manifest](evidence/reconstruction-generic-integer.json).
I use no live product compiler binaries after making the snapshot.
This does not establish a hermetic relocation or fresh compiler bootstrap.

My additive main integration through `3e4e5a89` preserves generator and focused
test source byte-for-byte. It changes separately admitted owned-helper and
managed-runtime paths; reconstruction refuses retained ownership metadata.
I preserve the stated original local tool pin and make no rebuilt-main claim.

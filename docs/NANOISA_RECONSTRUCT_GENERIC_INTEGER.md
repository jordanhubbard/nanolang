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

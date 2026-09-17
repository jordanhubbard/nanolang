# My unsafe blocks and local scopes

On 2026-09-17 I lowered unsafe statement blocks through the existing statement
and host-call paths. Nested returns, loop break/continue, and unreachable
statements retain their control flow. My C seed now includes unsafe blocks in
its termination analysis, avoiding previously emitted unreachable branches and
returns that prevented exact opcode comparisons.

I also repaired lexical name lookup exposed by this slice. I resolve the most
recent active local and hide block-local names at scope exit, while retaining
monotonically allocated slots. Nested unsafe, if and while scopes can shadow a
name and still mutate outer bindings. A local that escapes its block is refused.

`make -j8 test-nanovirt test-nanoisa-src-nano` checks my C-seed VM compiler and
self-hosted emitter: 89 C-seed checks, 86 emitter checks, and 32 integration
methods pass. The new fixture adds twelve C-seed opcode comparisons and
executes both modules under NanoVM and strict C11 AOT. It covers nested returns,
loop control, a supported host lookup, repeated shadowing, and outer mutation.
Refusal tests retain unknown-host rejection and reject a loop exit without a
loop. I add no new host ABI and do not bypass canonical frontend unsafe checks.

The initial shadowing reproducer failed its VM assertion; the repaired fixture
checks that inner references resolve to the inner value and that the outer value
survives each block. The C-seed termination defect was separately recorded as
`task_ed2788c9071249d2a8c8ce3fbec37a74`; lexical binding repair is
`task_64cc83f059264bf983a449c3312cde4f`; unsafe lowering is
`task_750341a5ccb04cffa9b2e0cc92e1f7d6`.

A fresh C-seed-hosted canonical compiler advances to enum member
`TOKEN_DOUBLE_COLON`, then refuses it as a non-record field receiver. The
actual debugger probe is `/tmp/nanolang-canonical-after-unsafe-field.log`;
`task_3ef1df55630e4199b825ff5d8a0043f1` records AST-backed enum lowering. I do
not publish a compiler `.nvm` or claim a successful bytecode bootstrap here.

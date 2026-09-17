# My initializer temporary frame

A global `(array_new 3 7)` previously emitted temporary `STORE_LOCAL`
instructions inside `__init__`, whose header always declared zero locals.
Assembly verification rejected slot 0. I now start initializer local state
independently, declare the actual capacity after lowering every initializer,
and clear that local state before ordinary function lowering. Global order,
type metadata, and verifier checks remain intact.

My regression uses three filled arrays (int, string and bool), twelve
initializer slots, and observable count/fill evaluation order. It compares
C-seed bytecode against self-hosted emitter output and executes both modules
in NanoVM and through native AOT. Its shadows check constructor helper return
values and effects, restore global state, and execute main. The shadow-module
fixture also contains a filled-array global.

Validation:

- The full emitter gate passes 86 bytecode comparisons and 70 Python methods
  in 87.926 seconds, including exact initializer/function comparisons and
  VM/native execution of the new fixture.
- C-seed and Stage 2-built emitters produce identical assembly for the fixture:
  SHA256 `cec87c2b128c1e367aba7e9ed3762fa27dd7b359abd2bf7279ab956a6b0db09e`.
  This is fixture parity, not a whole-compiler fixed-point claim.
- All five shadow-module methods, including a temporary-using initializer,
  pass with C-seed and Stage 2-built harnesses (17.149 and 9.906 seconds).
- After strengthening fixture shadows, C-seed VM compilation and execution
  pass; both emitters retain identical assembly and its prior hash.

Logs remain at `/tmp/nanolang-initializer-frame-gate.log`,
`/tmp/nanolang-initializer-stage2-build.log`,
`/tmp/nanolang-initializer-shadow-cseed.log`,
`/tmp/nanolang-initializer-shadow-stage2.log`, and
`/tmp/nanolang-initializer-final-fixture.log`. The earlier refusal is retained
as `/tmp/nanolang-global-filled-shadow.nano` and its `.nasm` output.
This repairs `task_3ed43e842d704748af03f82883793652`; the full compiler's range
for-loop shadow continuation remains separate.

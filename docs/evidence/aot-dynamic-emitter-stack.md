# Dynamic emitter operand storage

I replace the C emitter's 64-value operand arrays with checked allocations
bounded by function bytecode length, matching the classifier's bound. Each
branch snapshot owns independent slot and kind arrays. Restoring a snapshot
copies its live operands into the working arrays while preserving temporary
high-water counts. I free working storage and every snapshot on both success
and failure. Empty snapshots need no allocation or memory copy.

Array literals use dynamic scratch storage and incremental C output. I remove
both their 64-element translation limit and their 768-byte expression buffer.
This does not remove generated array-runtime capacities, the 256-temporary
limit, or the eight-field scalar/string aggregate restriction.

## Verification

`make -j1 test-nvm2c` passes 936 checks on Darwin. New generated executables
exercise both paths of a branch carrying 75 operands and merge distinct top
values before consuming the complete stack. Integer and string literals with
200 elements compile and execute; their distinct last values check ordering.
Existing malformed-join, record, array and tail-call tests remain passing.
`git diff --check` passes.

`make -j1 test-one-ir-compiler` still fails at function 20 with
`AGG_PACK has too many fields`. That is the next aggregate boundary, not a
regression in operand storage. I have not established full compiler execution
or release readiness. MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71` remains open;
the hub still refuses my claim with `agent_status_unavailable`.

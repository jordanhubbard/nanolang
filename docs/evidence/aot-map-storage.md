# Native map storage foundation

I added a self-contained C source fragment for the compiler's first native map
representations: string keys with integer or string values. It is currently
exercised by generated-C tests, not emitted for NanoISA hashmap instructions.
`HM_NEW` remains unsupported until classification, type propagation and emission
are connected. This is a storage prerequisite, not completed hashmap support.

The map uses separate collision chains and grows its bucket array with checked
size arithmetic. Keys and inserted string values are copied. Map aliases share
mutations. A successful lookup copies its string result, which the caller must
release; that result survives replacement, deletion and destruction of the map.
Missing lookup returns a void-tagged value, distinct from an empty string.
Destruction frees entries, keys, values and buckets without recursive traversal.

I reject unsupported value kinds and impossible growth instead of silently
dropping an update. Allocation failure aborts this native runtime; it is not a
recoverable allocation-error API. I do not claim adversarial hash-flood resistance
or support for arbitrary key/value representations.

## Tests

Generated C exercises 4,096 integer entries through growth and retrieval,
deletion/reinsertion, both int64 bounds, shared aliases, copied mutable input
buffers, replacement, empty keys and values, missing keys, retained string
lookups after map destruction, and deliberate collision chains. Separate child
executions reject wrong value kinds, unsupported map kinds and overflow-sized
growth. Each test compiles the same source fragment intended for later emission.

Ownership integration, map-shaped constraints, locals, calls, returns, branches
and hashmap opcode emission remain required before compiler acceptance.
Parent MAC task: `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.

## Verification

Final-tree normal and fresh ASan/UBSan runs each pass 1,106 AOT checks and 965
shape checks, plus opcode-coverage and sanitizer-driver tests. The map test
executes its growth/collision/ownership assertions inside generated C; its four
outer checks are not a count of every assertion. `git diff --check` passes.
Leak detection remains disabled by the broader sanitizer target, so I do not
claim a leak-checking run.

`make -j1 test-one-ir-compiler` still rejects `HM_NEW` in function 267 at offset
zero. Claiming the parent MAC task still returns `agent_status_unavailable`.

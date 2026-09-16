# Explicit AOT opcode coverage

I no longer let the classifier's default switch branch silently ignore an
instruction. It reports the unsupported opcode, function and byte offset.
Unconditional jumps have an explicit case; their existing control-flow handling
still records the target and stack state.

My new source-level test compares the explicit opcode cases in classification
and emission. This catches adding support on one side without updating the
other. It does not establish correct stack effects or semantic equivalence.
The broader verifier/transfer audit remains open.

Runtime diagnostic tests encode eleven unsupported instruction families through
ISA metadata and check rejection at offset zero, rather than a later stack
underflow or unrelated inference conflict. They cover hashmap operations,
floating-point and void constants, globals, float casts, string trimming,
indirect calls and stack rotation. Existing supported-opcode execution tests
remain part of the same target.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,102 AOT
checks, 965 graph checks, the opcode case-parity test and three sanitizer-driver
tests. The sanitizer driver verifies fresh ASan/UBSan object instrumentation.
`git diff --check` passes. Compiler acceptance still fails as described below.

## Compiler boundary

Compiler acceptance now reports `HM_NEW (0x78)` in function 267 at offset zero,
instead of an operand-stack underflow. Hashmap emission is still missing.
The retained compiler artifact contains three `HM_NEW 5 1` and three
`HM_NEW 5 5` instructions, eighteen `HM_SET`, six `HM_GET` and fifteen `HM_HAS`
instructions: string keys with integer or string values. This is an inventory
of that artifact, not a claim that other map forms are unnecessary for the
language.

MAC task: `task_f90db79b0f464637a18486c44262c4d3`. Claiming remains unavailable;
the semantic audit and hashmap implementation remain open.

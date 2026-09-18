# My character-access operand lifetime

I record `task_62caf894db9649cd904ce6faf3c37ffb` before changing my VM.
Static inspection found that STR_CHAR_AT pops its index operand but releases
only its string operand. I correct both the normal and source-type-error paths
so each popped owner is released once, after its last use.

I preserve the existing operation: an integer index selects an unsigned stored
byte; a non-integer index uses zero; a negative or out-of-range index returns
integer -1. I do not decode Unicode or allocate a character string. My stale
ISA header comment must describe this actual integer result.

I verify only fresh ordinary programs against the corrected source: embedded
NUL and unsigned bytes, ordinary range cases, existing non-integer fallback,
retained caller aliases, repeated invocation and source-type refusal cleanup.
The focused lifecycle harness and full VM regression suite must pass. I do not
run a historical failed artifact or use a pre-fix failure as an acceptance gate.

This prerequisite does not admit STR_CHAR_AT to LLVM/Wasm. A separate managed
contract must define its borrowed views, two-operand cleanup and exact byte
result before that admission. Full runtime51da, Darwin7ba and evaluator791a
remain open.

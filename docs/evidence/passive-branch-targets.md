# My passive branch target compatibility

I calculate relative branch destinations from the instruction start, matching
my ordinary verifier and execution contract. My passive metadata validator
previously added the instruction length a second time.

I test ordinary `JMP`, `JMP_TRUE`, and `JMP_FALSE` entry into a complete passive
block, followed by a normal jump after the block. Each module prints `42` in
NanoVM and in native output compiled with `-Wall -Wextra -Werror`. My existing
flow graph also retains its VM/native output.

The original calculation rejects all three valid branch fixtures at verification
and v2 conversion (six failed assertions). The corrected calculation passes
249 metadata checks, the four paired executions, and all 96 adjacent verifier
cases. I used ordinary positive control flow for this regression.

This repairs existing metadata acceptance. I do not yet emit `par`/`flow` records
from either frontend. Immutable external input proof remains
`task_bf571298c10d4cc5a387b9f233ff3c40`; trusted foreign intrinsic identity remains
`task_20f6cb36fbf24bba987b4ea503529438`.

# My exclusive multi-borrow File calls

I retain legacy `CALL_REF` for one borrowed formal. `FILE_CALL_REFS` (`0x97`)
encodes a little-endian `u32` callee and `u32` constant-pool index. The constant
uses the existing counted string storage as a byte container, not C-string
text. Its length is exactly twice the callee's declared parameter count.
Each little-endian `u16` is the caller's reference slot for that parameter,
or `0xffff` for an ordinary/owned value parameter. At least one parameter must
be borrowed. Every borrowed slot is below the existing 256-reference bound.

I copy the complete validated map into instruction facts before releasing the
input module. Both acyclic and cyclic analysis check the actual referenced
File types and owners. Distinct exclusive arguments cannot alias one owner.
A malformed map in an uncalled function also refuses preparation. Decoding the
opcode alone grants no File authority to generic consumers.

My VM and native frame entry validate every mapped reference, its live owner,
the declared parameter type and pairwise exclusive owner separation. I stage
all ordinary/owned values before installing child locals, then bind each formal
from its own mapped source. Existing return, loan-ending, assertion failure and
whole-runtime cleanup rules remain in force. Native agreement checks include
the copied map alongside decoded operands and complete call obligations.

My C and Nano lowerers derive the parameter map independently. Nano emits its
unique binary maps before ordinary constants and offsets every string index
accordingly; C uses its counted, deduplicating constant pool. Scalar operands
retain source evaluation order, and terminal operands still end pending loans
and owned temporaries. Neither frontend delegates source semantics to the host
publication bridge.

The source gate uses three forwarded and reordered references with a scalar
between them, repeated calls in a loop, real writes/rewinds/reads/closes and
all generated shadows. Separate malformed-map controls preserve failed outputs
and check copied facts after input mutation. Shared references and indirect
calls remain separate release requirements.

# Boolean scalar tags

I now distinguish boolean and integer representation facts even though both
use C integer temporary storage. My shape graph also keeps the two kinds
distinct. Boolean constants, comparisons, predicates, map presence checks and
type checks produce boolean facts. Locals, stack transfers, direct and tail
calls, returns and record fields preserve them.

Generic equality observes the operand tags instead of forcing unknown or
mixed operands into string storage. Boolean `true` is not integer `1`.
Comparisons with tagged map lookups can now box known integer and boolean
operands with their actual tags. Missing values remain distinct from false
and zero. My homogeneous integer-map emitter rejects boolean insertion instead
of silently storing it as an integer; broader map value kinds remain work.

`CAST_INT` explicitly converts a boolean to zero or one. Boolean `CAST_STRING`
and printing produce `true` or `false`. `TYPE_CHECK` can inspect known scalar
tags as well as optional lookup tags. Integer-only and boolean-only operations
and declared return types do not implicitly substitute the other kind.

## Initialization boundary

The VM initializes non-parameter locals to void. My current typed native
locals cannot represent that state. A control-flow worklist now checks definite
assignment before emission: parameters begin initialized, stores establish a
local, branch joins intersect incoming facts, and backedges converge before
reads are checked. Unreachable reads do not invalidate a function.

This is a rejection boundary, not implementation of void-valued local reads.
I track the latter as `task_288b833c300d40d18f6204c0cccba24f`. Without the guard,
tag inspection could report integer for a local whose VM value was still void.
The analysis is conservative and does not prune branches by constant values.

## Tests

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,191
AOT checks and 994 shape checks. The sanitizer driver verifies fresh ASan/UBSan
translator and graph objects. Opcode case-parity, driver tests and
`git diff --check` pass.

My producer matrix checks the tags of integer comparisons, boolean operations,
string predicates and `TYPE_CHECK`. Execution tests cover boolean locals,
branch stack transfers, direct and tail calls, record fields, printed output,
explicit casts and equality against integers and tagged lookups. Negative
tests cross function boundaries to exercise native inference rather than
being rejected first by the assembler's typed-opcode verifier.

Several old fixtures returned boolean values from functions declared to return
integers, or compared a boolean foreign result with integer `1`. The VM's
`result_tag_matches` requires exact boolean/integer return tags. I corrected
the fixtures to use boolean constants or explicit `CAST_INT`, retaining their
original output expectations, and added rejection regressions for implicit
conversions. No production language source was changed to hide a mismatch.

Initialization regressions reject direct, conditional and first-iteration
reads before assignment; accept assignments on both branch paths through loop
backedges; and ignore dead reads after return. Graph tests reject bool/int
unification.

Full compiler acceptance still stops at unsupported `LOAD_GLOBAL` in function
323 at offset 21. Native globals, general tagged scalar joins and void-capable
local storage remain unfinished. MAC still refuses task claims with
`agent_status_unavailable`.

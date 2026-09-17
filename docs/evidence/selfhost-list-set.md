# My typed list replacement

On 2026-09-17 I lowered supported `list_T_set` and `array_set` calls through
`ARR_SET` followed by `POP`, preserving their zero-result source contract.
I validate the receiver element type, integer index, setter type and replacement
value before emitting arguments once in source order.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 37 integration methods
after rebasing onto PR412. My fixture adds fourteen C-seed opcode comparisons and executes both modules
under NanoVM and strict C11 AOT. It checks receiver/index/value effects in order,
shared list identity, and old record/string aliases surviving replacement.
Malformed arity, index, receiver, value and typed-setter mismatches are refused.
Bounds cases exercise negative, end and 2^32 indices under both backends using
the separately merged VM full-width index repair.

Unsuppressed ASan/UBSan native execution exposed a returned-record-array leak:
28,680 bytes in the full fixture. A minimized program without any setter leaks
14,344 bytes from `nrarr_new` through a returned `List<LeakRow>`. Task
`task_0916ab0afb014b5984d69fbb11b0432d` tracks that existing AOT cleanup defect;
`/tmp/nanolang-returned-record-array-leak.log` retains the measured failure.
I do not claim sanitizer success or repair native lifetime tracking here.

A fresh C-seed-hosted canonical compiler advances to
`undefined function string_from_char`, recorded as
`task_292f60fa2cfa431990494d079cc8630c`. The actual probe is
`/tmp/nanolang-canonical-after-list-set-probe.log`. Full compiler emission and
matching bytecode bootstrap remain unfinished.

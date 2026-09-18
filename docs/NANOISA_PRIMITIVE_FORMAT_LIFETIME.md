# My primitive numeric formatting lifetime

I record `task_4ba402d50b864467bda1e8acf7dd601d` before changing my VM.
STR_FROM_INT/FLOAT currently pop a value but do not release its owner or check
the final string allocation. Their heap helpers also cast snprintf's returned
length without first checking status and buffer capacity. Static inspection
establishes these missing guards; I do not require a pre-fix failure replay.

I preserve exact-tag conversion: STR_FROM_INT reads only TAG_INT, otherwise
formats integer zero; STR_FROM_FLOAT reads only TAG_FLOAT, otherwise formats
floating zero. Existing decimal and C-locale/default-rounding %g bytes remain.
I check snprintf status/capacity before constructing a string, release the
popped operand after its last use, and reject a missing result before stack
publication. Existing string interning and first-error behavior remain.

My fresh corrected-source controls cover ordinary signed limits, float signed
zero/fraction, nonmatching numeric/owned-string fallback, unchanged caller
aliases and deterministic result allocation failure followed by recovery.
The formatter status/capacity checks receive bounded test doubles, not hostile
input. I run focused sanitizers and full VM acceptance. A separate managed
admission contract will reuse the reviewed portable formatters. Parent51da,
Darwin7ba and historical evaluator791a remain open.

# My primitive formatting lifetime evidence

I implement `task_4ba402d50b864467bda1e8acf7dd601d` at frozen production
`83e602e4`, after pre-code contract `6a33edfa`. My integer/float heap formatting
helpers now reject negative or truncated snprintf lengths before the uint32
cast. STR_FROM_INT/FLOAT release the popped operand after its last use and
refuse a missing output before publishing a string value.

I preserve the existing exact-tag behavior: only int or float respectively is
formatted; other tags use zero. My ordinary fixture covers signed integer
limits, -42, float -0/1.25, bool/U8/void/owned-string fallback, aliases and
allocation failure followed by repeated successful entry. Bounded formatter
return test doubles check negative and capacity-equal results without reading
uninitialized output. Both heap helpers and VM cleanup retain the caller's
single string owner and restore the heap baseline after release.

I passed the focused lifecycle target normally and with
`SUBSTRING_TEST_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -O1'`.
The focused flags instrument VM/heap/value/cycle sources, not every linked
library. Full `make test-nanovm` passed all 274493 checks plus required
allocation/callback/stack companions. Independent parent source review found
no scoped blocker. Logs: `/tmp/nanolang-primitive-format-focused.log`,
`/tmp/nanolang-primitive-format-sanitized.log`,
`/tmp/nanolang-primitive-format-full.log`.

I did not replay historical failed artifacts. I retain VM interning and its
existing formatting/locale behavior. A separate managed admission must stay
within the established portable C-locale/default-rounding formatter contract;
these checks do not prove arbitrary host locale agreement. Full runtime51da,
Darwin7ba and evaluator791a remain open.

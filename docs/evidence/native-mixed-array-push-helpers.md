# My mixed-array push helpers

I conservatively emit integer and string push helpers when a module contains
both array kinds and any push operation. I declare those helpers `static inline`,
as I already do for array storage helpers, so a module that pushes only one
kind remains valid under strict C11 warnings. I do not change their bodies,
mutation behavior, runtime checks, or compiler warning flags.

Two focused assembly regressions cover both directions: an integer literal with
only string pushes, and a string literal with only integer pushes. Both compile
with `-Wall -Wextra -Werror` and execute their length assertion.

On Linux ARM64, `make -j8 test-nvm2c` passes 1,769 checks; shape constraints pass
1,076 checks. Task `task_2dedfc5f181f41d4951eed8927648306` records this fix.

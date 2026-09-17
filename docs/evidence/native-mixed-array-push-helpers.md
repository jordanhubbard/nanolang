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
1,076 checks. Task `task_2dedfc5f181f41d4951eed8927648306` records that fix.

Clang still diagnoses an unused `static inline` helper. I therefore reference
both emitted scalar push helpers from generated `main`, following the same
portable convention I use for other optional helpers. The mixed-array cases
require those references in addition to compiling and executing with strict
warnings. Task `task_4c32ccf895f5435998cd4982c7dd4087` records this Darwin
portability correction. On Darwin ARM64, `make -j8 test-nvm2c` passes 1,773
translator checks and 1,076 shape checks.

# My parser compiler deadline evidence

Hosted Darwin job 117906341666 on `28e50b802` passes its C parser allocation
corpus and the paired token/schema generator corpus, then times out the full
parser fixture's Stage1 compiler invocation at its hardcoded 900-second limit.
The retained artifact 12167861413 records empty stdout/stderr and no temporary
products. I cannot establish whether frontend execution or native compilation
consumed that time. Cleanup reports no errors and the whole group is absent.

I connect these explicitly identified compiler invocations to the existing
`CMD_TIMEOUT` policy (1800 in the qualification job), preserving their 900-second
default if unset. I do not change guest execution, per-shadow deadlines,
assertions, selected shadows, fixture inputs or provider lists. The enclosing
parser step allows 90 minutes so both self-hosted invocations can use their
explicit compiler budgets without the old 40-minute step truncating them. My retained
runner now records the actual deadline and elapsed time and takes a bounded
2-second snapshot of only that process group's command names on timeout.
Diagnostic failure does not interfere with mandatory process-group cleanup.

Six focused controls pass in 5.987 seconds: existing compiler-budget/flags/descendant
checks, preserved 900-second default and explicit override, actual short guest timeout
under a 1800-second compiler environment, retained group snapshot and cleanup,
and unavailable-diagnostic cleanup. The initial control run accidentally
imported an unrelated TestCase into discovery; I retain the resulting missing
provider failures and import the module instead, without changing its tests.

The full parser corpus with fresh combined tools and the exact hosted candidate
must still pass. A larger explicit deadline alone is not evidence of compiler
correctness or proof that the original phase was merely slow.

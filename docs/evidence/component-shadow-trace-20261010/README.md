# Isolated component-shadow trace

I retained the original general-build failure in my
[callable loop evidence](../callable-array-loops-20261010/build.log).
I then ran `bin/nanoc src_nano/typecheck_driver.nano -o /private/tmp/nl51-typecheck-driver`
with `NANO_SHADOW_TRACE=1`, without a shadow-deadline override. At this run,
`bin/nanoc` selected `nanoc_c`; this is C-seed evidence, not a self-hosted gate.

The isolated compile returned zero in 96.632 seconds (`run.json`). All 563
selected shadows completed with zero failures; their measured durations sum
to 4.756716 seconds (`trace.log`, `timings.json`). The two slowest individual
shadows took 0.808889 and 0.791397 seconds. The emitted executable also passed
its checker entry assertions (`entry.log`). I did not change the default
ten-second shadow deadline.

This trace did not reproduce the historical timeout. It does not establish
its cause, qualify the whole general build, or refresh the bootstrap receipt.
I retain the original incident as open evidence while continuing reproducible
implementation gaps. I ran this trace before the callable-operation changes
in this checkpoint; its source baseline is b6f5dd6ea.

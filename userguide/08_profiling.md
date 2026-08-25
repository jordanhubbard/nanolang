# Performance Monitoring

This page used to be a long marketing write-up of `-pg`. The published session
is now [Performance Profiling](guide/07_performance_profiling.md). The
repository canonical page is
[docs/PERFORMANCE_MONITORING.md](../docs/PERFORMANCE_MONITORING.md).

I wrap native `main` as `_nl_run_with_profiling`. JSON goes to **stdout** and
to `--profile-output`. Collectors are **gprofng**, **xctrace**, and **sample**.
`--pgo` does not read that JSON.

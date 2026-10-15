# Tracking migration CI terminals

PR #977 remains open and blocked. I retrieved the original failed job logs
from run 37837249695, attempt 1. Memory Sanitizers reaches compiler bootstrap
and stops at its 60-second shadow deadline. Code Coverage stops during package
installation after three attempts, ending with exit 137. I preserve these as
distinct failures and do not weaken the shadow deadline or infer an OOM cause.

I checked active jobs before attempting a single coverage-job rerun. GitHub
connectivity prevented submission; the later authoritative run query still
reported attempt 1, completed. No successful rerun is claimed. Main CI and
migration merge remain open under #976 and #975.

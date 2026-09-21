# My retained generated-C artifact integrity

I independently read and hash all 946 retained reports and 28,899 content-addressed
objects (6,765,906,543 bytes) in my combined generated-C qualification seal. Every
recorded size and SHA-256 matches. The read-only audit takes 20.590 seconds.

I preserve the [audit result](audit.json) and [driver](audit.py). This establishes
retained artifact integrity. It does not replace semantic coverage review or
fresh integration acceptance against current main; those remain open.

My [independent coverage audit](coverage-audit.json) checks the six final retained
Linux/Darwin configurations. Each has 1,590 successful child statuses, 292
linked/observed O0/O2 product build and run/baseline checks over all 73 cases,
and 146 complete, contiguous allocation-failure coverage records totaling
18,976 recoveries. Source and tool maps match before and after each phase.
My first audit lookup used `run` for observed products; their retained execution
records are named `baseline`. I preserve that initial audit script and correct
only the lookup. Fresh current-main integration remains required.

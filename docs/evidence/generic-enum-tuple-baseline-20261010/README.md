# My enum and tuple generic parity baseline

I run these sources against the C producer and the fresh full compiler seed at
0c501dc4b while preserving bootstrap sources and host-library generations.
My self-hosted enum and tuple identity programs compile with mandatory shadows,
verify, and execute in NanoVM and sanitized generated C; every recorded step
returns zero. My repeated-variable control using distinct enum declarations
refuses during self-hosted type checking and preserves its prior output.

My C producer rejects the enum identity at generic specialization. It also
rejects the distinct-enum control there rather than diagnosing conflicting
nominal identities during checking. I require exact enum identity retention,
not simply permission for another value tag. My C tuple identity loses its
result element metadata: destructuring reports a zero-element tuple before
emission. All C refusals preserve prior output.

I retain results.json and execution.json with explicit exit codes; empty logs
alone do not establish success. These cases identify the next C-producer parity
work, not complete enum/tuple generic acceptance. I require nested/returned and
repeated-variable controls alongside the retained positive execution cases.

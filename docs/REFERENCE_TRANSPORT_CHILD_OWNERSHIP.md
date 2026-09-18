# I check the children my reference transport owns

I correct task_c564d9e9089845ab9566b004a7828643 in my test fixture, after static review and fresh Darwin diagnostics. Production remains unchanged.

My original fixture asserts that process-wide `waitpid(-1, WNOHANG)` finds no child. A library owns only its successfully spawned evaluator and writer. It must also preserve a caller-owned child. I record that distinction before changing the fixture.

Fresh Darwin diagnostics preserve the original missing-PATH status 17. One observed wait returns zero; one millisecond later it returns ECHILD without a positive PID being reaped by the fixture. A separate instrumented spawn returns ENOENT with its initialized PID still -1. A diagnostic baseline wait before evaluation changes the outcome. Symbolizer warnings accompany these runs, but I have not established the transient child's identity or attributed a production defect to it. I do not use a delay, repeated wait or baseline warmup as the acceptance correction.

I will wrap only the exporter test translation unit's posix_spawnp and fork calls. Allocation-free bounded bookkeeping records positive PIDs only on successful creation in the parent. Immediately after every evaluation I require each recorded PID to return -1/ECHILD from waitpid with WNOHANG. I reset bookkeeping only after those assertions. Failed spawn output is never inspected for ownership. I add a separately created caller child with a readiness handshake: it remains live and unreaped across evaluation, and the fixture alone terminates and reaps it during cleanup. No process-wide wait may consume unrelated children.

I retain existing direct argv/pipe transport, output, descriptor, signal, repeated-call, closed-standard-descriptor, allocation-failure and missing-program controls. Every fixture exit closes its files/descriptors, frees input/result and cleans its own sentinel without hiding the original status. Strict warnings and ASan/UBSan with detect_leaks=1 remain enabled. Fresh Linux GCC/Clang and Darwin Homebrew Clang transport suites must pass; no broad NanoCore or release acceptance follows from this fixture repair.

# Co-process lifecycle gate isolation

I qualified this bounded gate at production commit
`cb0bf298b74b2ec2cd12a0b1cc96c9e77ba94d4f` (tree
`98f1cf5e79440b09b0fda98d00789f4685780095`). The checkout was clean before
the final runs. Its base was canonical `db68fd52678ff6eb6d03b82ec51716431c773d31`;
later main integrations are not part of this evidence.

The native probe starts with no co-process, calls the C `abs` import twice and
observes one stable owned PID, sends that PID `SIGKILL`, observes its exit with
`waitid(..., WNOWAIT)`, and makes another call. That call reaps the dead child
and launches a distinct worker. Explicit stop leaves `waitpid` reporting
`ECHILD`. A separate child blocked on a private pipe remains alive through the
crash and cleanup checks.

The shell runner now records only children it starts. Each daemon uses a socket
under its private temporary directory. The EXIT and signal paths stop and wait
only for recorded PIDs; there is no process-name discovery or killing. Compile,
execution, output-comparison, daemon-start and client failures remain visible.
The fake-tool harness proves that normal and injected-failure exits preserve an
unrelated process whose executable path contains `nano_cop`.

The first fake-tool run found that an empty indexed array under `set -u` is not
portable to macOS `/bin/bash` 3.2. I replaced that bookkeeping with a scalar
space-delimited PID list. The corrected harness passed; I did not change a
runtime deadline or lifecycle assertion.

## Frozen results

- `make -j8 test-cop-lifecycle`: 13 passed, 0 failed in 1.23 seconds. Log
  SHA-256 `1c985ede944c714d5e5e8aeedd42267b8ad62ef0f92e071c5157455b0d680df2`.
- `python3 -m unittest -v tests.test_cop_lifecycle_gate`: 3 passed in 1.355
  seconds. It covers the normal gate, an injected compile failure and an
  injected second-client failure. Log SHA-256
  `431ff093d08f51b56713b7d712388143749a5bce3bed29a21c8958a088efbda4`.
- `make -j8 test-cop-protocol`: all 35 adjacent protocol tests passed in 1.07
  seconds. Log SHA-256
  `1659687a422da59d9fd1cc84ee68d4e35b90680dbfdc7d117f81eac118d8ba5b`.
- `bash -n scripts/test_cop_lifecycle.sh` passed.

The build selected Apple Clang 21.0.0
(`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`),
GNU Make 3.81 and Python 3.14.6. The strict native build retained
`-Wall -Wextra -Werror`.

This closes the lifecycle *gate* defect. It does not prove every concurrent FFI
schedule, every worker allocation failure, or the held 5.1 product acceptance.

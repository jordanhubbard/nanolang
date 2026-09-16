# My daemon gate owns its endpoint and children

I replace `scripts/test_nanovm_daemon.sh` with an entry point to
`tests/daemon_integration.py`. I retain its eight selected source programs.
Every compilation must succeed and produce nonempty bytecode. Standalone
and daemon execution must both exit zero and produce identical byte output.
Matching failed executions do not pass.

I create a mode-0700 temporary directory under `/tmp`, start a foreground
daemon in an owned process group, and probe the protocol with a ping/pong.
I check daemon liveness before and after executions, bound commands and
startup, and terminate/reap my daemon before removing my directory. I do not
kill by process name or remove the user's default socket/PID file. I print
bounded failure-log tails before removing temporary artifacts.

`NANOVMD_SOCKET` selects an explicit socket for both daemon and client; the
corresponding PID filename adds `.pid`. The default paths are unchanged.
Oversized paths fail rather than aliasing a truncated path.
`NANOVMD_NO_AUTOSTART=1` makes a failed connection fail without launching a
replacement. I set both variables in the test's child environment. An
explicit path is configuration, not a sandbox: its caller must own and
protect the containing directory.

## Evidence on Darwin, 2026-09-15

- `make test-nanovm-daemon` rebuilds affected binaries and passes the existing
  three socket-boundary cases, native endpoint/no-autostart checks, injected
  gate checks, and all eight real standalone/daemon comparisons: zero skips.
  Log: `/tmp/nanolang-daemon-fixed.log`.
- The final injected gate covers 14 modes, plus rejection of an empty corpus:
  success, a daemon ignoring SIGTERM, compilation failure, missing artifact,
  identical failed VMs, daemon-only failure, output mismatch, daemon death,
  startup failure, startup hang, invalid pong, compiler hang, standalone hang,
  and daemon-client hang. It checks that recorded children are reaped, private
  directories are removed, and an unrelated owned process remains alive.
  Log: `/tmp/nanolang-daemon-injection-final.log`.
- The first strict real run failed on the first client. A separate private
  ping reproduced daemon `SIGBUS`. The crash report
  `nano_vmd-2026-09-15-114500.ips` identifies `___chkstk_darwin` in
  `client_thread`, a 544 KiB thread stack, and a stack-size-exceeded error.
  I moved `VmState` to a checked heap allocation owned by each execution,
  with cleanup on output-stream failure and normal completion. The positive
  gate above passes after rebuilding. Logs before the fix:
  `/tmp/nanolang-daemon-real.log`, `/tmp/nanolang-daemon-final.log`.

I initially misread `sun_family` and suggested it was missing. Source review
corrected that diagnosis before any such change; it was already `AF_UNIX`.

MAC rejected my daemon-gate claim with `agent_status_unavailable`. Code
verification does not establish ledger closure. Tasks:
`task_999bf1a1ab96472294660aa2b19cae9a` and
`task_ed9fc711786a459abb5bafb69ea33f3e`.

The separate co-process lifecycle script still kills ambient processes and
must be repaired before running that gate:
`task_a02a66101b184e6eaa3e61480079f300`. This checkpoint does not verify all
daemon exit-value semantics, concurrent clients, foreign calls, Linux, or
the full release suite.

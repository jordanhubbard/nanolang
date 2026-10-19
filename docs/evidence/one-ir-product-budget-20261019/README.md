# My full compiler product build budget

My hosted Darwin managed suite (run 39205176925, job 117906341664) reached the
native compiler full-source self-rebuild after native C compilation. The rebuild
failed at a literal 900-second limit despite the job's explicit CMD_TIMEOUT=1800.
I retain the exact failure excerpt and original log hash; I do not label that
historical run successful or infer that 1,800 seconds guarantees completion.

I apply the existing native_compile_timeout(default=900) selector only to this
full-source compiler product build. The default remains 900 seconds; an explicit
positive CMD_TIMEOUT is honored. Ordinary guests retain their original limits,
including 10-second hello/help executions and the unchanged verifier assertions.
The existing run_captured supervisor and process-group cleanup are unchanged.

I qualify selection and cleanup with tests.test_native_process: default900,
explicit1800, rejected invalid settings, independently short guest limits, and
actual descendant termination/reaping. This is harness evidence, not proof of a
completed compiler rebuild. The actual full product remains to be run with
pinned providers when the coordinator's worker slot is available; I do not start
a competing build during the aggregate bootstrap.

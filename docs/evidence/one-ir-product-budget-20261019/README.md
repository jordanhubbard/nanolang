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

## Distinct Linux capture prerequisite

Linux job117906341352 built bin/nano_as_capture.so, then failed the native
compiler's full source rebuild at file_source_inputs module capture. My first
OneIR product test did not set NANO_AS_CAPTURE_HELPER. module_builder.c defaults
to a helper beside /proc/self/exe, so the relocated temporary compiler searched
its temporary directory. The second product test and bootstrap already supplied
the explicit path for selected calls.

I share that existing Linux selector locally and propagate its environment to
all relocated compiler invocations and their child test suites. I change no
capture product code, helper build rules or live-source fallback policy.
Five focused tests cover real supervisor/descendant environment propagation
and the existing four budget/cleanup tests. I removed a newly proposed brittle
call-count assertion during review; the actual Linux product gate remains required. Platform selection is simulated only
for the environment-mapping test; no native Linux success is claimed.

Existing tests.test_assembler_capture_records requires Linux and independently
checks incomplete captures/replay refusals; tests.test_module_builder_self_capture
retains GCC fail-closed PCH controls. Neither is replaced by this harness evidence.
Actual Linux helper/capture and full native compiler product qualification remain
pending the hosted/aggregate run. My local host is Darwin.

My final reviewed five-test run passes in 5.354 seconds; `capture-environment-final.log` supersedes the earlier six-test wiring run. No Linux native product was run locally.

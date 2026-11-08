# My nonblocking JUnit failures

I retain the full JUnit step and all 64 compilation failure records from Linux
x64 job 128477711690 in run 42397278342. Requested head is
`435dae34cc2ce74d5210806d26ef1ebb9f9e4b7a`; actual PR checkout is
`a6ba0d210667fcc65d91778fcfcccaa8b2cfb917`, with complete Git tree equality already
recorded in `../hosted-linux-qualification-20261107/checkout-tree-identity.json`.

My [original job state](job.json) has a successful blocking quick step, a failed
JUnit step and successful overall conclusion. The workflow explicitly marks the
JUnit command `continue-on-error: true`. My [full step](junit-step.log.gz) and
[machine-readable summary](summary.json) retain 237 tests: 173 passed, 64 failed,
zero skipped. All 64 observed failures occur during compilation. I retain the
complete original job log privately under
`~/.worktrees/nanolang/audits/2026-11-07/junit-1081/`, with its SHA-256 here.
ANSI colors are removed from the retained step; timestamps/diagnostics remain.

I trace one selection difference directly: `make test-quick` invokes
`tests/run_all_tests.sh --lang`, while `make test-junit` invokes the same runner
with `--format=junit --output=test-results.xml` and selects language, application
and unit categories. Both default to backend `c` and compiler `./bin/nanoc`.
The extra corpus contains existing legacy syntax and refused host imports as
well as concrete lowering/typechecking failures. I have not classified each
failure against the unchanged 5.1 scope or reproduced a common cause. #1081
tracks the required mapping and repairs. I preserve the tests and assertions.

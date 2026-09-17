# My bootstrap dependency branch reconciliation

I reconcile worker head `d152181d70bdf2caea6332f8f08685e9bac207fe`
against integration `e8d2fbb4`. My ancestor `d665d144` already tracks linked
runtime inputs and directory membership through `SELFHOST_SOURCES`.
I retain the current recursive Make rules and direct dependency on the
self-hosted stage stamp rather than attach all inputs to C-seed readiness.

The worker branch adds prerequisite-list inspection tests. My existing
`tests/test_bootstrap_source_dependencies.py` instead queries the real Make
rules in isolated fixtures with controlled timestamps. It checks current
artifacts, source and runtime edits, additions/removals, missing compiler
binaries, and exclusion of unrelated programs, documentation and artifacts.
It also checks that hidden cache payload changes do not invalidate stages.
I retain this behavioral suite rather than add a duplicate structural suite.
The current traversal conservatively includes source-like files in visible
directories; I do not adopt the branch's pruning of all directories named
`build`, `cache` or `obj`.

I retain the branch's useful quick-test integration by adding the named
`test-bootstrap-dependencies` target for the existing suite and invoking it
from both `test-quick` and `test-impl`.

`make test-bootstrap-dependencies` exits zero: eight tests pass in 17.126
seconds. The host-local log is
`/tmp/nanolang-bootstrap-branch-reconciliation.log`.
I inspected both entry-point recipes; I did not run the entire quick or full
suite in this reconciliation. This tests rebuild scheduling, not a new
bootstrap, content-addressed caching, or compiler-output equivalence.

MAC task `task_85191b435c95480d9db52e3671d1e740` is stopped and unowned
when inspected. I attach evidence without claiming a ledger closure. Full
branch reconciliation and release acceptance remain open.

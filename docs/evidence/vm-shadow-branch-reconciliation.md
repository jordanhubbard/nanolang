# My VM-shadow branch reconciliation

I compare worker `a13993621554766f9adfd75371663af29c7efa33` with
integration `0afd074f`. Its 19 bytecode-shadow test methods are all present
in the current 39-method module. Current source also retains its typed
absolute value, single-evaluation min/max lowering, local metadata restoration,
array literal annotation checks and initialized nested function signatures.

I retain the current dependency-shadow default, graph-wide checking in each
module's authority context, separate test bytecode, and foreign binding
before verification. I do not restore root-only selection or fixed shadow
function capacity. The full pre-change bytecode-shadow module passes:
39 methods in 55.856 seconds. That result does not establish all examples
or full backend equivalence.

## Publication defect found during review

My VM supervisor checked exit status and deadline but did not require the
test entry to finish. A foreign `exit(0)` or `_exit(0)` therefore published
output before reaching a failing assertion. Both new regression cases fail
against the previous compiler.

I add a private pipe with close-on-exec descriptors and nonblocking parent
reads. The child writes completion only after successful VM execution and
teardown. The parent requires both completion and a successful exit status;
its existing deadline and signal diagnostics remain. Setup failures close
descriptors and reject publication. This is not a sandbox: foreign code still
has host authority, and the channel does not defend against deliberate
in-process descriptor tampering.

The new regression preserves prior bytecode after both early-exit routes.
The existing suite covers success, traps, dependency selection, foreign calls,
timeouts and production/test separation. After rebuilding `nano_virt`,
`python3 tests/test_bytecode_shadows.py` exits zero: all 40 methods pass in
54.173 seconds. This is not the larger cache/snapshot gate or full release
suite. Logs on this macOS host:

- `/tmp/nanolang-vm-shadow-exit-before.log`
- `/tmp/nanolang-vm-shadow-handshake-build.log`
- `/tmp/nanolang-vm-shadow-reconciliation-final.log`

MAC defect `task_9d9eefa909be4990be0151bed7439953` was auto-assigned to
Rocky; I sent an AgentBus handoff and do not alter that ownership. Branch
scope is tracked by `task_4c6ff6e9986a49d6a01701a66b8842d6`. Full release
acceptance remains open.

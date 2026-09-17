# Serial calls around passive nodes

I checked `task_eccab63d320946d283342bb8fc8eb18f` at production/test checkpoint
`29689d55607eee25f10743cae63b19c59ce04640`, restacked on main `ea63b76d`.

I remove only the requirement that every direct CALL in an owner belongs to a
node. Calls inside nodes still require the unchanged whole-callee scalar proof.
The ordinary verifier validates direct targets and actual arity/result stack
effects. NanoVM reserves a separate callee frame; my positive assembly test
writes callee slots 0–2 and retains the caller parameter and all node results.
Calls before, between and after par/flow blocks print 10, 20 and 30 in that order.
An effectful callee owning a pure block still cannot serve as a pure node call.

My C-seed and self-hosted producers now both publish the ordinary multi-block
source program. Both bytecode artifacts execute the same values in NanoVM and
strict standalone native C. Other owner opcode restrictions remain; I do not
claim foreign/indirect/tail owner calls, broader captures or a parallel scheduler.
Version 1 is unchanged. The node/result/read/branch checks are unchanged.

## Checked gates

- `make test-passive-metadata test-disasm-roundtrip test-verifier`: passed;
  22 Python methods, 210 roundtrip checks and 96 verifier checks.
  Log: `/tmp/nanolang-passive-owner-calls-gates.log`.
- Fresh `make bootstrap nanoisa_emit`: both native compiler stages and configured
  smoke checks passed. The subsequent frontend test found my fixture used the
  reserved function name `effect`; I corrected it to `emit_value`. I retained
  `/tmp/nanolang-passive-owner-calls-frontends.log` with that test-authoring error.
- `make test-passive-flow-frontends test-passive-par-frontends`: six methods
  passed in 40.473 seconds and five passed in 37.380 seconds after the correction.
  Log: `/tmp/nanolang-passive-owner-calls-frontends-r2.log`.
- After the additive native record-array-global restack, 22 passive methods
  passed in 2.099 seconds and the focused both-producer owner-call method passed
  in 0.110 seconds. Log: `/tmp/nanolang-passive-owner-calls-restack.log`.

These are bounded semantic checks, not a new compiler fixed-point claim.

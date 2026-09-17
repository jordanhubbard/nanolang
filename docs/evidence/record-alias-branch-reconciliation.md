# My interpreted record alias reconciliation

I compare worker head `9698ce08e10edadddbde25d8f0d0325d9de7877c`
with integration `7cf7667b`. The worker introduces `owner_count` and scans
environment symbols to discover aliases. That scan does not account for
references held inside other records. It also adds parameter-specific copies.

I retain the current binding policy: `env_define_var_with_type_info` copies
records, and `env_set_var` copies before releasing the previous binding.
`create_struct` copies strings and recursively copies nested records. Both
interpreter call routes bind parameters through these environment helpers.
This is not complete aggregate reclamation: other field kinds still need
their own ownership contracts. I do not claim that copying records implements
full affine ownership or identity-preserving mutable captures.

The existing `eval_record_alias_reassignment` shadow checks nested records,
string fields, self-assignment, callee replacement and aliases surviving
replacement. I retain the worker's additional integer-pair scenarios as
`eval_record_alias_across_direct_calls`, checking both result tags and values
from host `call_function`, then executing their shadows. A helper shadow also
checks `read_pair` directly.

Verification command:

```sh
make test-eval test-env-scoping
```

The command exits zero. Its prerequisites rebuild both self-hosted stages
and pass bootstrap smoke checks, including execution without the C seed.
All evaluator tests pass, including both alias regressions; the environment
suite reports 32 passed and zero failed. These are ordinary host tests,
not allocation-failure injection, leak sanitization or full release gates.
The host-local log is
`/tmp/nanolang-record-branch-reconciliation.log`.
MAC `task_1ac1c2fd6bc04b5e81932553901b0d4b` was stopped and unowned
when inspected. Full branch reconciliation and release acceptance remain open.

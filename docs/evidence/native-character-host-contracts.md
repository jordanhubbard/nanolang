# Native character host contracts

I accept eight exact builtin imports emitted by my bytecode frontend:
`vm_is_digit`, `vm_is_alpha`, `vm_is_alnum`, `vm_is_space`, `vm_is_upper`,
`vm_is_lower`, `vm_is_whitespace`, and `vm_digit_value`. Each takes one integer;
the classifiers return a boolean and `digit_value` returns an integer.

I require the empty builtin namespace, FFI import kind, exact arity, exact
parameter and result tags, and strings without hidden suffix bytes. I retain
rejection for arbitrary libraries and malformed signatures.

My generated helpers preserve the current VM contracts. The six shared ASCII
classifiers first narrow to C `int`. `is_space` then tests its unsigned byte
for positive inputs; `is_whitespace` instead compares the full integer to space,
tab, newline, or carriage return. I preserve these existing differences rather
than silently changing the language contract in an adapter.

Validation on main `4a555b5d` plus this change:

- `make test-nvm2c`: 2,146 native checks and 1,092 shape checks pass.
- Eight native fixtures each assert 29 input boundaries, including vertical tab,
  form feed, non-ASCII bytes, byte and integer narrowing, and both int64 limits.
- The same 232 assertions pass NanoVM and strict generated C.
- Sixty-four malformed import variants are rejected.

Fresh compiler bytecode passes import validation and the integrated array
parameter repair. It now reaches the recorded projected global-store blocker in
`nisa_emit_function`, function 598 offset 58
(`task_3636ea1587cd41a88e4660abe94acb53`). Full native compiler execution and the
bootstrap fixed-point gate remain open.

Local logs: `/tmp/nanolang-native-character-integrated-gate.log`,
`/tmp/nanolang-native-character-parity.log`, and
`/tmp/nanolang-native-character-integrated-fullcompiler.log`.

# Boxed primitive-array parameters

I preserve an array's handle and element kind when a call receives both a
concrete primitive array and a tagged value, such as a global load. The original
compiler failure joined `env.names` and global `nisa_names` at `mb_resolve`.
Both are string arrays; the latter uses tagged global storage.

I widen only call-parameter storage. My separate record-field merge contract
remains unchanged. Directed shape conversion puts the concrete array inside the
tagged payload without erasing its element constraints. Integer, boolean, and
string arrays retain their distinct storage kinds. I preserve an absent value
until the callee checks or consumes it. I do not add record-array boxing.

My regression matrix exercises both function declaration orders and both caller
orders for all three primitive array kinds. It checks ordinary and tail calls,
absent values, shared mutation, and a surviving local alias after replacing the
global. Populated incompatible payloads are rejected statically; unknown global
values with wrong receiver or element tags are rejected when consumed.

Validation:

- `make test-nvm2c`: 1,834 native checks and 1,092 shape checks pass.

- Twelve identical programs pass NanoVM and strict generated C.
- All twelve generated programs pass ASan and UBSan with default runtime checks.
- My shape tests retain later element facts inside a tagged wrapper and reject
  conflicting payload types.

My full compiler gate remains open. At baseline `c477f068`, fresh compiler
bytecode reaches an unbound `vm_is_alnum` host import
(`task_9e4e686f52bc426bbe4d2e694fa565bc`). The retained compiler bytecode that
originally failed at `mb_resolve` now advances to an unresolved projected global
store in `nisa_emit_function` (`task_3636ea1587cd41a88e4660abe94acb53`). Neither
advance establishes native compiler execution or a bootstrap fixed point.

I also recorded the existing tagged-array update bounds gap separately
(`task_40b4cf1f065f49609cec599123f06557`); this parameter conversion does not
repair that helper's narrowing or ignored out-of-range updates.

Local logs: `/tmp/nanolang-array-parameter-final-gate.log`,
`/tmp/nanolang-array-parameter-shapes.log`,
`/tmp/nanolang-array-parameter-parity.log`,
`/tmp/nanolang-array-parameter-sanitized.log`,
`/tmp/nanolang-array-parameter-fullcompiler.log`, and
`/tmp/nanolang-array-parameter-retained.log`.

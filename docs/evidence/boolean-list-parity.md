# My boolean List syntax and backend parity

My C seed parser rejected `List<bool>`: the `List<...>` element switch knew
`int`, `u8`, `string` and named types but not the `bool` keyword, so it refused
the type token. I gave the boolean list its own element kind (`TYPE_LIST_BOOL`)
and a full runtime (`src/runtime/list_bool.{h,c}`) instead of lowering it
through an unrelated integer list.

I now accept `List<bool>` in my seed parser and register the complete
`list_bool_*` builtin family. The interpreter (`src/eval.c`) and my native C
transpiler use the same `List_bool` runtime, my NanoVM codegen tags boolean
list elements with `TAG_BOOL`, and my self-hosted codegen accepts `List<bool>`
as a local, field and result type. The self-hosted emitter publishes
`ARR_NEW 4` for `list_bool_new`, so every frontend and backend agrees that a
boolean list element is boolean, not integer.

The dedicated runtime covers construction, `with_capacity`, `push`, `pop`,
`get`, `set`, `insert`, `remove`, `length`, `capacity`, `is_empty`, `clear` and
`free`. `tests/test_runtime_lists.c` exercises that contract directly.
`tests/nl_types_bool_list.nano` is the end-to-end regression: it constructs
boolean lists, reads elements as booleans in conditions, mutates them,
iterates with `for`, and passes and returns `List<bool>` across functions.

Validation:

- `make build` completes the three-stage bootstrap with the C seed and both
  self-hosted component entry checks.
- `make -j1 test-runtime-lists` passes 33 AST list types plus seven non-AST
  list types, including `List_bool`.
- `./bin/nanoc tests/nl_types_bool_list.nano` passes its shadow tests and the
  produced binary exits 0.
- `./bin/nano_virt tests/nl_types_bool_list.nano --emit-nvm` plus
  `./bin/nano_vm` executes the same program and exits 0.
- `./bin/nanoisa_emit` accepts a `List<bool>` program that uses my
  cross-frontend list API and publishes `ARR_NEW 4`; the four
  `test-nanoisa-src-nano` unittest modules pass (90 tests) when LeakSanitizer
  is disabled, because LSAN cannot run under the authoring sandbox's ptrace.

The C-seed-only operations (`with_capacity`, `capacity`, `is_empty`, `clear`,
`free`) remain available in the interpreter and native backend. My NanoVM list
surface keeps the same narrowing it already has for `List<int>` and
`List<string>`, so boolean lists do not claim more than the shipped list
contract.

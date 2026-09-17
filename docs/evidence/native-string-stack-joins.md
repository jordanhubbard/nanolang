# My projected string joins

I reproduce the fresh compiler failure on main `b3f79449`: function 357,
`check_let_statement`, joins a tagged projected field with the string returned
by `type_to_string`. Both the unchanged translator and the map-global repair
reject the same retained compiler bytecode. The mismatch is storage kind 8
versus 1 after byte offset 363.

I give a forward string/tagged join its own storage shape. I convert branch
values into that storage without rewriting a producer's exact shape. I retain
those shape IDs until emission, box strings on the taken edge, and keep the
original conditional fallthrough stack unchanged. My existing parallel join
assignments still handle overlapping temporary slots. I propagate temporary
high-water marks and free retained join metadata on success and failure.

My focused tests check both branch orders and directions, present and missing
map lookups, lower live operands, returned strings and copied text surviving
20,000 temporary-map allocations. Positive native runs use ASan, UBSan and
leak detection, with at most 32 peak owned allocations. The conditional-edge
case also proves that boxing does not rewrite the fallthrough operand stack.

I keep these remaining boundaries explicit:

- Widening an already classified backward stack edge needs a fixed point for
  its consumers. I retain a VM-positive two-iteration fixture and an explicit
  native refusal under `task_ea3c8acd272a49669bd6ae6aa75cdf49`.
- The complete selfhost-emitted compiler is a different artifact from the
  Cseed-seeded bridge. My native probe reaches an integer/string parameter
  conflict at `purity_node` calling `purity_call`, recorded as
  `task_04376d3e430c478d968af69e26543a0f`. I retain that input at
  `/tmp/nanolang-selfhost-compiler-native-input.nvm`, SHA256
  `dc517d54b0417b4934907ea565d34bdb23110da7116bef3f5713faa48746d4ee`.
  I have not established whether its cause is emitter metadata or native
  inference, and I do not call native execution of that artifact complete.

The forward repair is MAC `task_55002ea4e4c64f80a6ba70b7f147ebef`.

I measured 2,386 native checks and 1,092 shape checks passing. My initial three
join methods plus the existing one-IR suite pass 31 methods in 321.753 seconds,
including the freshly generated native compiler's explicit `--emit-nvm` hello
product executed by both VM and native output. The final four focused methods,
including the retained loop refusal, pass in 6.993 seconds. Instrumented
translator and shape-solver objects also translate the retained full compiler
and reject the loop fixture with ASan, UBSan and leak detection enabled.

I keep logs at `/tmp/nanolang-native-string-join-{full,integration}.log`,
`/tmp/nanolang-native-string-joins-final-focused.log`, and
`/tmp/nanolang-native-string-join-asan/{positive,negative}.log`.

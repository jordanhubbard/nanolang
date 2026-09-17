# My string-valued map lowering

I retain `HashMap<string,string>` alongside my existing string/integer map
shape. Constructor context selects `HM_NEW 5 5`; lookup result inference
follows the receiver's value type. Map updates require matching key/value
types, and void updates retain the existing `POP` stack effect.

My parity fixture uses local construction and carries maps through direct
calls, returns, aliases, and record fields. Eighteen C-seed comparisons pass;
both modules run in my VM and generated native code. A saved lookup string
survives a subsequent overwrite. An empty string value remains distinct from
an absent key through `map_has`.

I separately execute expected constructor contexts through self-hosted
returns, arguments and record fields in VM/native code. Global construction
passes in the VM. My C seed currently refuses a bare constructor in a typed
return (`task_f4e1871af407805219770d7620d58349`); I do not claim cross-frontend
constructor parity until that repair lands.

Standalone native map globals remain unsupported, as already tracked by
`task_95796f5f49564ed4a911fd05a1aac5b4`. The complete failing module and the
successful variant without the global are retained under
`/tmp/nanolang-map-context-repro/`. I preserve VM global coverage separately
from native-supported constructor contexts; full compiler/native acceptance
still requires the open AOT task.

The typed gate passes 86 comparison checks and all 62 methods
(`/tmp/nanolang-string-maps-typed-gate.log`). I also refuse mismatched map
returns, local bindings and arguments. My C seed currently accepts those
four probes, separately tracked as `task_d0438e26b84147cdb9fd16b654c44a6a`;
`/tmp/nanolang-cseed-map-mismatches/` retains their source and diagnostics.
Constructor propagation does not establish assignment-type checking.

Both native map fixtures pass ASan/UBSan with leak detection enabled
(`/tmp/nanolang-string-maps-sanitizers.log`). This covers the supported local,
return, argument and record paths, including the retained overwritten value;
it does not establish native global-map support.

The fresh canonical whole-compiler probe now reaches assembly publication
without hitting the emitter failure breakpoint. Publication rejects
`Expected quoted string after .string`; no compiler module is produced.
I retain `/tmp/nanolang-maps-fullcompiler-probe.log` and track the next
quoting/publication boundary as `task_c77ac0644fda463a8a2d0ae7dd735908`. This is progress through lowering,
not a complete bytecode compiler or bootstrap fixed point.

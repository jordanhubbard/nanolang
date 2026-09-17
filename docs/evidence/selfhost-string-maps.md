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

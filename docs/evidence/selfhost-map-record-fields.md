# My string/int map fields

On 2026-09-16 I admitted `HashMap<string,int>` fields in supported finite
records. I reuse my existing map opcodes and record transport; no new ABI is
introduced. The `CollectResult` fixture checks construction, calls, returns,
string-array and boolean siblings, and shared map mutation through an alias.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 30 integration methods.
Ten new C-seed bytecode comparisons pass, and both emitted modules execute under
NanoVM and strict C11 AOT. Wrong map field shapes and scalar field values remain
refused. A separate self-hosted fixture variant exercises direct field receivers
for `map_put`, `map_get` and `map_has` under VM and native execution.

The cross-compiler fixture uses explicit typed map locals. My C seed loses the
generic field arguments in direct map receiver inference, tracked separately as
`task_160826784e8a4aa4ac9d5e589a54c814`; I do not claim to repair that path here.
Map/record global AOT transport remains a separate boundary. Full compiler
emission and matching bytecode bootstrap remain unfinished.

A fresh C-seed-hosted canonical driver reaches the next refusal,
`statement outside the pinned subset`, after this change. The debugger log
`/tmp/nanolang-canonical-after-map-fields-probe.log` records the actual path;
no compiler `.nvm` is published. This is progress through compiler lowering,
not a successful bytecode compiler bootstrap.

The statement is `PNODE_UNSAFE_BLOCK` (kind 37, node 25), confirmed by
`/tmp/nanolang-canonical-after-map-fields-kind.log`;
`task_750341a5ccb04cffa9b2e0cc92e1f7d6` tracks its scoped lowering.

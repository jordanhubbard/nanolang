# My bounded native scalar ROT3 contract

I record `task_18ca25c8fc294c81b1a662fd22fc411b` before native edits.
A fresh ordinary reconstruction fixture was refused at opcode ROT3,
function0 offset50; `/tmp/nanolang-reconstruct-rot3-gcc.log` retains that
translation result. No native binary was produced or executed.

I admit only exact INT/BOOL operands in this child. Bottom-to-top `a b c`
becomes `c a b`. Classification permutes complete simulation slots, including
origin, exact shape, record metadata and scalar provenance. Emission permutes
existing immutable slot IDs and their kinds; other metadata remains indexed
by those IDs. I allocate no value storage and change no ownership/root lifetime.
I preserve underflow guards and refuse other kinds before publication.

I require small ordinary VM/native GCC/Clang sanitizer parity for distinct
values, mixed tags, calls/locals and loops; underflow and excluded-kind
translation controls preserve previous outputs without executing refusals.
Full native ROT3 for other kinds and reconstruction parent4bd remain separate.

My first compile caught an incorrect return of the void diagnostic helper
(`/tmp/nanolang-native-rot3-build.log`). I separate the diagnostic call from
`return 0`; this was a build failure before any test execution. The task and
roadmap prerequisite were already recorded in the reconstruction branch
before native edits; this branch carries the same task contract.

My first focused invocation also discovered five imported unittest methods,
which passed; I switched to a module import to keep the gate scoped. Two new
positive controls passed, then the string-refusal fixture was rejected by
assembly because PUSH_STR requires a pool reference, not inline text.
I retain `/tmp/nanolang-native-rot3-gcc.log` and correct the fixture with
an explicit string-pool declaration before rerunning the focused gate.

## My bounded acceptance

At production `e8e34083`, four focused GCC methods pass in 0.168 seconds.
The same four plus the existing total-arithmetic method pass Clang in
1.192 seconds. Positive cases execute the same verified module through VM
and standalone generated C with ASan/UBSan; total arithmetic retains its
O0/O2 UBSan checks. Four excluded kinds and three underflow counts preserve
previous output without executing rejected operations. I do not claim a new
full native suite, heap rotation coverage or compiler bootstrap from these
focused gates. Reconstruction qualification consumes this exact translator
separately.

I retain `/tmp/nanolang-native-rot3-corrected-gcc.log` and
`/tmp/nanolang-native-rot3-clang.log`. My measured source/tool hashes are:

```text
3dedd55a768a3b2e23d9924940ff2138690aab657444f03a48c36b1616999f02  src/nanoisa/nvm2c.c
72ea51c2896da10f92cec08cbb14c0a074944999a60adc9397e4f215203f75de  bin/nvm2c
bc03e4085b16ce563cb2e4e0607152652b66a4ecd8b254e266eca001cae7d035  bin/nanoisa
3af7fc1f1c42ebcdbbbe3408f6c7a8da5c3d8338502d80105b1ea66bd143b1d4  bin/nano_vm
```

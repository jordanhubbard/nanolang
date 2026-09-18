# My scalar expression-match evidence

I lower exhaustive distinct named expression arms over exact nongeneric scalar
unions. Every arm produces the same int/bool/float/string type. I evaluate the
scrutinee once, preserve one selected result, isolate inference bindings and
restore emitted arm names. My statement-block path remains shared and retains
its return/fallthrough behavior.

Production checkpoint after canonical integration:
`f607b97e12df8e0ef7c73f1eb68e05896281d476`, based on main `9a5d6938`, including
native terminal-assertion PR656 and current functional-array additions.

Fresh integrated `make -j8 bootstrap nanoisa_emit nano_virt nano_vm nvm2c
nanoisa_dump` passes. Log:
`/tmp/nanolang-scalar-match-values-integrated-bootstrap.log`.

The final own-tool gate, with no native translator override, passes all
**12 source methods in 44.184 seconds** and all **4 native prerequisite methods
in 0.361 seconds**. Log:
`/tmp/nanolang-scalar-match-values-integrated-paired.log`.
The earlier focused NanoVirt/typechecker gate also passes all 89 NanoVirt
checks and all typechecker tests at the reviewed source checkpoint; log
`/tmp/nanolang-scalar-match-values-adjacent.log`.

My twelve-method suite contains the seven existing scalar-union methods and
five expression methods. It covers both raw emitter hosts with exact text
comparison, C-seed NanoVirt, both canonical stages, separately emitted selected
shadows, VM and ASan/UBSan native execution. Positive controls include four
scalar result types, signed zero, call/return/initializer contexts, nested
payload shadowing and once-only scrutinee/selected-arm effects. Unknown and
mismatched nested arms refuse while retaining previous output; false or
unsupported selected shadows still block publication.

My first gate retained `/tmp/nanolang-scalar-match-values-paired.log`: VM
execution passed all producers for the full four-scalar fixture, but native
translation refused its unreachable empty-stack HALT after a false assertion
in float-returning helpers. I recorded and separately repaired only terminal
same-block false assertions in PR656; reachable HALT policy is unchanged.
The unchanged twelve-method suite then passed in **44.167 seconds** with the
explicit companion translator, recorded at
`/tmp/nanolang-scalar-match-values-native-companion.log`.

Generic/resource/nested union payloads, aggregate results, wildcard/guard/or
patterns and value-yielding block arms remain outside this bounded admission.
I do not change runtime or ownership authority or infer full release readiness.

MAC: `task_b43095db71f74c0b8418c27530f1686a`.

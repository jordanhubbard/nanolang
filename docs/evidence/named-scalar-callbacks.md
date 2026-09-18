# My named scalar callback checkpoint

I qualify task_fd15bbaf8acd4e448347ce3cce4e7a1d, exact checker prerequisite
65b8b96f489b4e268a0a79b0052cbe61 and global signature retention prerequisite
cc6eebea1f7c4d2880c8517ffff501a3 on Linux AArch64. My contracts are
[NANOISA_NAMED_SCALAR_CALLBACKS](../NANOISA_NAMED_SCALAR_CALLBACKS.md) and
[NANOISA_CSEED_REDUCE_CHECKER](../NANOISA_CSEED_REDUCE_CHECKER.md).

I replace FUNCREF/CALL_INDIRECT only for an exact, original same-module
named int/float/bool callback without a visible value binding or capture.
I preserve source/initializer ordering and the captured array length.
Computed/local/global callback values retain their indirect execution path;
my native backend still refuses that path before replacing output.
I check reduce as array<E>, A, fn(A,E)->A with exact retained identities,
and retain explicit global function annotations using the existing AST owner.

## My frozen acceptance

| Source | Actual check | Result |
|---|---|---|
| 2c68c01c | Fresh bootstrap: Stage1, Stage2, installed compiler and C-seed independence | Pass |
| 2c68c01c | Full checker units plus main/module global identity controls | Pass |
| 2c68c01c | NanoVirt controls | 90 pass |
| 2c68c01c | Named/scalar-reduce/source-arithmetic suite | 19 pass, 166.850s |
| 2c68c01c | Clang ASan/UBSan named suite | 8 pass, 109.991s |
| 8a751c1e | Integrated C/runtime tools, checker units, NanoVirt | Pass; 90 NanoVirt |
| 8a751c1e | Integrated paired suite | 19 pass, 167.274s |
| 8a751c1e | Integrated Clang ASan/UBSan named suite | 8 pass, 110.807s |

My final integration includes main8b0fa1fd through owned-string PR750 and
public-C PR757. My reviewed codegen and checker production are unchanged.
All 17 preintegration and 24 integrated source/tool hashes match their
before/after observations. Stage1/2 hashes remain those of the actual
2c68c01c bootstrap; I do not claim a fresh final-integration bootstrap.

My named suite compiles through Cseed and three raw emitters built by
Cseed/Stage1/Stage2, then verifies and executes ordinary VM and generated
native C. Named native cases use ASan/UBSan under GCC and Clang. The new
binary64 fixture also executes interpreter and all three legacy producers.
I compare exact NaN/zero/result and unchanged-input bits, int/bool cross-kind
map output, typed empties, alias growth/order, visible/computed callbacks and
prior-output refusals. Adjacent scalar arithmetic separately exercises
LLVM, optimized LLVM, Wasmtime and import-free Node execution; that is not a
claim that LLVM accepts these callback array modules.

## My preserved first outcomes

I retain the original seed native array-return refusal under task_baca01f.
The corrected ordering fixture uses a local array passed through parameters
and still tests alias growth and source/initializer/callback order.
My initial 23224572 suite had five passes and three failures: an annotated
U8 literal reached a U8-return shadow as INT (task_b5d82), Cseed rejects
reserved builtin declarations while selfhost resolves them, and the old
Cseed reduce checker accepted a wrong signature without executing it.
I preserve the U8 boundary and qualify actual packed-U8 fallback separately.
I retain each route's real builtin-declaration outcome.

My 6a9265f9 checker/bootstrap passed, but paired qualification found the
missing explicit global signature wrapper. I record and repair that metadata
prerequisite. The adjacent scalar method also found missing nvm2llvm, then
nvm2wasm, in my setup. I retain both logs, build the complete tool set, and
qualify the corrected suite. These setup errors are not compiler defects.

I separately track missing tuple-literal and bare-function accumulator
expression facts under task_a2cb46decf4443ffa5b3bdbc946c0e5e. Explicit retained
bindings remain the supported route. No unknown identity becomes compatible.
I do not close scalar5009 or the release. I audit the original d099 acceptance
separately; general function-value support is not an invented condition on
that original named callback task. Its failed-worker history remains intact.

My [manifest](named-scalar-callbacks.json) seals 25 reports and the 24 current
source/tool hashes. Full temporary artifacts remain at the paths in those
reports. I have not replayed the historical failed-worker artifact.

# I qualify the bounded owner-array source slice on Linux

I freeze production at c9fb07ed55329f59e6b67ef0bf3f9dc756d6aa88:
reviewed source99fdb4804 plus qualified activationc510a9358. The isolated tree is
`/home/jkh/Src/nanolang-owned-array-source-integrated`. Later corrected fixtures
run externally from `nanolang-owned-array-source-refusal-phase`; the frozen tree,
compiler stages, raw emitters and shadow drivers remain unchanged.

I pass fresh bootstrap271.579s and tool/probe setup26.523s. The five additional
source emitter/shadow driver builds take177.671s. I retain separate inventories
at these boundaries rather than claiming setup-generated cache files were inputs.

My first source run preserves its actual failure: five methods365.579s, four
passed, then optional-local raw assembly succeeded where the harness expected a
lowering refusal. The four passes include the original unchanged Bundle/PREFIX
and every shadow through all four public drivers; canonical dumps and complete
shadow modules; VM/native ASan+UBSan+LSan; integer minimum/wrap/zero/range/Boolean
controls; nested returns/prepared roots/empty arrays/aliases; and both-sided
optional FLOAT runtime checks and failure cleanup. No refused module executes.

My reviewed fixture-only phase correction requires final assembler refusal for
optional-local/add/negate. Its fresh affected run passes those groups and bound-at,
then preserves a second failure: reserved `not` cannot form a source variable
binding. That one-method run takes0.233s. Both lexers reserve TOKEN_NOT; I change
only the expected diagnostic phase, retaining the exact source and output sentinel.

My final reviewed run passes two selected methods40.480s (runner40.978s).
It executes bound-not plus seven later refusal groups, then the previously unrun
Samples/STRING/scalar/borrow adjacency. Four earlier optional/at groups are
explicitly omitted and attributed to their prior evidence. I do not call either
failed original run successful or claim a new complete six-method single run.

All2265 source maps, seven host tools,799 post-setup inputs and13 retained
compiler/emitter/driver binaries are unchanged across the relevant assertions.
Native C uses explicit `/usr/bin/gcc`, strict C11/O2, address/undefined sanitizers,
`-lm`, and `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`.

The [summary](owned-array-source-linux/summary.json) and
[report hashes](owned-array-source-linux/report-sha256.json) retain65 reports and
428 artifact files. The complete retained archive is
`/tmp/nanolang-owner-array-source-linux-artifacts.tar.gz`, SHA256
`051db33537396ff81928e39ada3b941e035be9f3c8b578c9bb624f9d29d5dbe5`.
The seal script checks before/after maps and uses checked sequential packaging;
its archive, report manifest and per-file artifact inventory are separate.

I make no Darwin, current-parser/File canonical integration, mutation set/push/
length, full430220/4be, product or release acceptance claim. Source18731 and the
fixture child remain open until their remaining acceptance and actual merge.

# My fresh owned-source installed-product attempt

I freeze `cd72dab32f03f0817cffcf1527e75e8a3482f534` in
`/home/jkh/Src/nanolang-product-owned-integration`. This new tree integrates
preserved productfa14d95f and current main through owned-string PR761,
aggregate policyPR762 and checked match-binding scopePR765. I did not modify
or execute olde9 failure artifacts, compilers or libraries. The reviewed
[build plan](../PRODUCT_OWNED_INTEGRATION_PLAN.md) remains separate from this
measured result. Full product parentd76 remains open.

My first Linux run starts with no bin or object/cache directory. Fresh
`make -j2 bootstrap` passes in215.658seconds, then the six requested tool
targets pass in25.454seconds. I explicitly select this tree's VM, translator,
AOT runtime and module cache. I verify installed Stage2 selection before the
tool and quick phases and separately during actual parser component
compilation. Toolchain paths/hashes and the exact runner are retained.

My unchanged `make test-quick` exits2 after1724.690seconds. All three canonical
component compilations and entry assertions pass; all17 core language cases
pass. Parenthesized parser, extern selection, module introspection and the
preceding dependency/toolchain gates pass. Every one of the unchanged244
eligible examples compiles for the VM (248 on disk, four existing exclusions).
This includes the unchanged affine-resource example and its mandatory shadows.
The example source hash matches the preserved product source. Combined with
PR761's C-seed/both-stage exact VM/native output and original-shadow controls,
this satisfies c435's repaired-example return to installed-product coverage;
it does not turn the complete quick gate green.

The next target, `check-stdlib-docs`, fails:158 registered builtins versus157
documented, missing `float_to_bits`, and the conversion section heading says11
while12 entries exist. I record task_dc5edb6c84ec4a809f9b98900c2e4a81 before
repair. No later quick phases run, and I do not exclude this documentation gate.
The full log SHA-256 is
`80de94a0c569baca934bf656e1bb81af9e56fc6e22fee00e21d21f8bab0ac70f`.

All1674 recorded tracked source/test/script hashes are unchanged, as are the
Git head and clean tracked tree. Make intentionally rebuilds tools during the
quick phase: the existing Stage1 executable hash changes; the other previously
recorded tool/cache artifacts do not. New test binaries and libraries appear.
I retain complete before/after artifact manifests and do not claim every tool
was immutable. The source tree, tools and libraries stay preserved at the
qualified pin; this evidence commit lives in a separate worktree.

My [sealed reports](product-quick-cd72dab3/) include phase logs/status/timing,
compiler-selection observations, source and artifact manifests, runner,
toolchain and unchanged-example source identity. `report-sha256.json` seals
their exact bytes. The parent's Darwin qualification, future fixed points and
full release/ownership/reconstruction gates remain separate. Older passing
fixed points do not qualify this new source.

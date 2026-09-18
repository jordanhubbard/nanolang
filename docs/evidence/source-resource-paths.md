# My explicit source resource-path acceptance

MAC `task_d74d8a4fb4a048a786666195eaa4e8d5`.

I implement my [bounded resource-path contract](../NANOISA_SOURCE_RESOURCE_PATHS.md)
on main `1114aba9`. Source checkpoint `09e71bc1` passed a fresh bootstrap and
all 21 paired source methods in 244.068 seconds. Parent review then identified
an effectful ownership comparison nested inside Boolean `and`. I replaced
that guard with explicit nested statements in `44445be0`: a returning loop
body never invokes the reaching-edge comparison. I retain the first passing
run as evidence of its tested native compiler configuration, not evidence for
every Boolean lowering strategy. The corrected source receives a fresh
bootstrap and final acceptance below.

I compare incoming owner liveness and pending-disposal flags only across
reaching branches, select a sole reaching arm, and require the incoming state
on every loop backedge. Each lexical resource local must be explicitly
consumed. Cleanup drains only holders marked by complete destructive patterns;
I also refuse live-source-owner exits in raw program/shadow lowering rather
than depend on the canonical checker to reject them. Existing verifier,
reference runtime, nominal layout and wire authority are unchanged.

Eight positive variants cover both branch orders, nested construction and
reordered complete patterns, whole-owner moves, reused loop slots,
zero-iteration bodies, one-return arms in both orders, and entered/zero loops
which consume an incoming owner on their terminal path. Both producers and
canonical Stage1/Stage2 compare exact code, layout, ownership/path and advisory
name metadata, including all selected shadows. The same modules execute in
VM and ASan/UBSan native output with leak detection; stripping advisory names
retains execution.

Nine dedicated refusal cases cover local leaks, mismatched reaching-owner
states in both arm orders, changed loop ownership, moved-value use, resource
assignment, partial moves, helper-owned locals and wrong nominal moves.
Existing source refusal controls and authority tests remain required. I do not
admit resource assignment, partial field moves, break/continue, helper-owned
locals or deeper call graphs in this slice.

Logs:

- `/tmp/nanolang-resource-path-bootstrap.log`
- `/tmp/nanolang-resource-path-gates.log`
- `/tmp/nanolang-resource-path-sequenced-bootstrap.log`
- `/tmp/nanolang-resource-path-final-gates.log`

My corrected fresh bootstrap passed. Final acceptance passed all 21 source
methods in 242.035 seconds. The same invocation passed 314 ordinary/343
allocation affine state checks, 441 ordinary/751 allocation affine bytecode
checks, 959 assertion lifecycle checks, 1,662 nested-reference checks plus 63
allocation checks, and 2,004 multi-caller checks plus 93 atomic-binding and 89
owner-allocation checks. The local-name codec also passed 123 checks.

I used `make -j8 -o bootstrap test-source-borrow-emission test-affine-state test-affine-bytecode test-owned-assertions test-nested-references test-multi-caller-references`
after the corrected fresh bootstrap. No test method was skipped. Source stayed
on main `1114aba9` plus this slice throughout the final run; later map changes
on main were not copied into the active tree. No verifier/runtime source was
changed by this PR.

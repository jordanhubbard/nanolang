# My consumed-owner reassignment acceptance

MAC `task_227b7b89a64d4e768705a9bd2c56430e`; [bounded contract](../NANOISA_SOURCE_OWNER_REASSIGNMENT.md).

I began on canonical `e4706bf5`. The first implementation `62d6889d` passed a
fresh three-stage bootstrap. Review identified that a pending-disposal holder
is logically consumed even while physically live. I added an explicit source
provenance refusal in both producers at `b4fd2db3`; I do not rely on synthetic
names to prevent its reuse. The updated fresh bootstrap also passed.

My positive acceptance retains exact C-seed/raw selfhost/Stage1/Stage2 code,
layout, ownership/path and advisory name metadata. VM and ASan/UBSan native
execution, including stripped names and every selected shadow, pass for
nested owner roundtrips, reassignment after explicit destructuring, both
branch orders and incoming-owner restoration across zero/entered loops. The
destination keeps its original name interval and physical slot. Runtime and
verifier source are unchanged.

The full source gate ran 23 methods in 274.858 seconds. Its 22 positive and
existing methods passed. My new refusal method stopped on a fixture diagnostic
assertion: its borrowed-target case first declared an unsupported helper-owned
local and correctly received the earlier helper-local refusal. I retain that
log. The corrected fixture assigns directly to the borrowed formal and
requires ownership/type/emitter diagnostics while excluding parse errors.
The corrected method passes in 150.248 seconds, covering all nine refusal
cases across C-seed, canonical Stage1/Stage2 and three raw emitters. Production
and the other 22 methods remain unchanged. This is one retained 22/23 run plus
a passing focused correction, not a claim that the original run was green.

Independent authority gates pass: 314 ordinary/343 allocation affine-state
checks; 441 ordinary/751 allocation bytecode checks; 959 assertion lifecycle
checks; 1,662 nested-reference plus 63 allocation checks; 2,004 multi-caller
plus 93 atomic-binding and 89 owner-allocation checks. The name codec passes
123 checks and 20 marker-allocation boundaries.

Logs:

- `/tmp/nanolang-owner-reassignment-bootstrap.log`
- `/tmp/nanolang-owner-reassignment-paired.log`
- `/tmp/nanolang-owner-reassignment-refusal-corrected.log`
- `/tmp/nanolang-owner-reassignment-authority.log`

Commands: `make bootstrap`, then the corrected source's
`make test-source-borrow-emission`; independent unchanged-tool authority gates
used `make -j6 -o bootstrap -o nano_vm -o nvm2c test-affine-state test-affine-bytecode test-owned-assertions test-nested-references test-multi-caller-references`.

This slice does not admit constructor assignment, partial field assignment,
helper-owned locals, break/continue or broader owned call graphs. It does not
execute or claim acceptance for the separately frozen product artifacts.

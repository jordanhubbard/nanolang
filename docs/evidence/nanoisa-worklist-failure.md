# Failed NanoISA worklists

I stop declaration registration, global initializer traversal and function worklists after the first lowering failure. Function dependency collection leaves its queue unchanged once failure is recorded. A failed function body returns before creating a successful function header or passive parameter metadata. Global initializer failure still clears its temporary local-name/type frame before returning empty. Statement lowering retains lexical cleanup from PR #558.

My module result remains empty on refusal. Shadow entry text built before a dependency fails stays internal and is discarded before final prelude/result publication. The existing fresh-entry reset remains responsible for the next compilation.

The independent contract is MAC `task_a919486cc4b749b8b47cae9bee986abf`, recorded before implementation. I make no attribution or completion claim for historical export-shadow incident `task_dd74b033c3984805bc27ce5017096c3c`. I did not replay its preserved binaries or cases.

At source `8c51a1c9`, ordinary state shadows verify unsupported declaration, failed global and failed function-body behavior: no later string literal interning, retained first diagnostics, global temporary cleanup, unchanged failed-state dependency queue, and exact successful assembly after reset. `make test-nanoisa-src-nano` passed 86 comparison checks and 87 paired methods in 116.117 seconds. These paired checks retain existing VM/native successful behavior and refusal contracts.

The first authoring build was deliberately stopped after review caught an overbroad dependency-search loop edit; the corrected entry guard and its queue-preservation shadow are in the tested source. I retain `/tmp/nanolang-worklist-failure-gates.log` as interrupted evidence and `/tmp/nanolang-worklist-failure-final.log` for the corrected complete gate.

The same command completed fresh native bootstrap through Stage2 and installed-compiler smoke checks, exit 0. I then rebased additively through PRs #567/#569 to `7402cd72`; my emitter file is byte-identical to the original tested source (SHA-256 `ac8e3bca9f87bf2cc0eb27b61379e07ab0d3aa810bc63c86e75157ab068034e2`). No compiler-source bootstrap repeat is required for that restack.

After restack, I rebuilt the emitter (including ordinary shadows) and current VM/native tools, then passed the two inferred-local success/refusal methods with the normal comparator rebuilt. Logs: `/tmp/nanolang-worklist-restack-final.log` (tool build plus the first focused invocation, which lacked the comparator removed by the complete gate) and `/tmp/nanolang-worklist-restack-focused.log` (correct prerequisite plus both passing methods). A prior misspelled make target stopped before building, retained in `/tmp/nanolang-worklist-restack.log`. These setup corrections are explicit, not unexplained compiler failures.

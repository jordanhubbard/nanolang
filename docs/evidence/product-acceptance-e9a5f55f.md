# My installed-canonical candidate acceptance

I freeze production `e9a5f55f9b82b3de8c03b3df240665865fd25e5c`, integrating canonical main through PR712. This includes the qualified consuming runtime/source calls, managed mutable arrays, exact binary64 facts/text and native optional-map-key repair. Later main changes, including VM slice cleanup PR715, are outside this pin. I do not release this candidate or close the full roadmap.

The failed63c Linux gate and its independently passing focused/fixed-point/Darwin evidence remain in [their original record](product-acceptance-63c26ecd.md). The optional-key repair passes a fresh unchanged63c transpiler compile with a corrected translator and subsequent native entry assertions; that bounded repair does not substitute for this combined product gate.

I use isolated detached Linux worktrees for the full quick gate, VM fixed point and native fixed point. Each records exact source and tool identities. The quick runner explicitly performs bootstrap and tool builds before `make test-quick`, asserts that `bin/nanoc` resolves to Stage2, and retains compiler identities before each phase. Its bootstrap and tool builds pass. The full quick gate subsequently fails at the affine-resource example shadow boundary recorded below. My Darwin peer has completed bootstrap and the requested tool build at the same pin, recorded installed Stage2 selection, and reached canonical component compilation under the full PTY quick gate. Both platform quick gates are terminal and fail at the same affine-resource example boundary; their prior component compilation and entry checks pass.

The fixed-point generations retain their prior per-stage bounds: VM1200 seconds and native1800 seconds, 48 GiB owned RSS and 32 GiB minimum host-available memory. Each route has independent embedded library paths; no cross-run byte-equality claim is implied. Both complete generation routes now pass independently, as recorded below.

Local manifests/logs are under `/tmp/nanolang-product-quick-e9a5f55f`, `/tmp/nanolang-product-vm-fixedpoint-e9a5f55f`, and `/tmp/nanolang-product-native-fixedpoint-e9a5f55f`. I seal terminal results without changing the tested checkouts. Broader ownership, reconstruction, managed runtime, formal correspondence, clean full tests and documentation gates remain open. Subsequent source integrations need fresh combined acceptance.

My separate canonical component-shadow report (`task_14c8ecbd8aaa484ea5e73f2aa43fa48b`) has a pushed pre-execution contract at `4cbffa9e`. Its own fresh tools complete all three supervised shadow modules and driver entries. Parser315, checker513 and transpiler505 ordered calls pass with unchanged recorded tools and source. PR722 retains the report; this separate run does not modify these acceptance trees.

## My Linux full-gate failure

My exact installed-canonical quick gate exits 2 after 1611.6 seconds. All three component compilations and entry assertions pass, as do the 17 core examples. VM example coverage compiles 243 of 244 eligible examples, then reports `language/nl_affine_resource_demo.nano` shadow compilation failure at line 28: `I require one borrowed helper in my source borrow profile`. I keep this example eligible and retain all shadow requirements. Later quick-gate phases do not run.

I record repair `task_c4351c720aee424ea9b90187e51a08f2` and preserve the [full log](product-quick-e9a5f55f-failed.log), SHA-256 `dfc9e4f5816fc09f7b9be8da1bc86009f6fa0084b65913da464c3a86501b4191`, and [manifest](product-quick-e9a5f55f-failed.json). The head is unchanged and tracked source remains clean. Compiler-selection evidence precedes component compilation; it does not claim that internal gate bootstrap leaves every tool binary unchanged. The independent VM/native fixed-point and component-shadow runs subsequently pass at their own immutable pins.

## My completed VM fixed point

My fresh VM run exits zero. Initial, Stage1 and Stage2 compiler modules each contain 408,836 bytes and share SHA-256 `27e281811a8fab6dabddd311d6fef36bbe8bac6905e14bdfb1cdb3478c7c403d`. The two complete generations take 910.103 and 912.126 seconds under their unchanged 1200-second bounds. Verification and Stage2 hello compilation/execution pass. The three embedded host-library paths and hashes match across generations.

My [manifest](product-vm-fixedpoint-e9a5f55f.json) and [integrity report](product-vm-fixedpoint-e9a5f55f-integrity.json) retain clean unchanged source plus unchanged capture helper, translator and recorded host libraries. The runner does not record a before/after VM executable hash, so I do not extend the integrity claim to that binary. This raw fixed point does not establish full compiler correctness, supersede the failed Linux quick gate, or qualify later main changes. The completed native qualification is recorded below.

## My completed native fixed point

My independent native run exits zero. Initial, Stage1 and Stage2 compiler modules each contain 408,848 bytes and share SHA-256 `0a64dec9f362b6b572c931d5ba51035f584175a221fce7b9cf7fae807f306759`. Complete generations take 1194.313 and 1118.534 seconds under unchanged 1800-second bounds. Module verification, native translation/build/help, and Stage2 hello compilation/execution pass. The three embedded host-library paths and hashes match across generations.

My [manifest](product-native-fixedpoint-e9a5f55f.json) and [integrity report](product-native-fixedpoint-e9a5f55f-integrity.json) retain clean unchanged source and unchanged recorded helper, translator, initial native compiler and host libraries. This route has different embedded paths from the VM run, so I claim equality within each route only. Both fixed points pass while the full Linux quick gate remains failed; this candidate remains held.

## My matching Darwin full-gate failure

My peer completes exact-e9 installed-canonical bootstrap in 227.302 seconds and
the requested tool build in 62.630 seconds, then runs `make test-quick` under PTY
without timeout overrides. All three canonical component compilations/entries
and 17 core examples pass. The gate exits two after 1874.353 seconds on the same
`nl_affine_resource_demo.nano` line-28 shadow-profile refusal as Linux. The
other 243 eligible examples compile; no example is excluded and later quick
suites do not run. Repair c435 remains the implementation owner; the peer's
separate report task6042 is complete and duplicate defect61f0 is being reconciled.

I independently copy and SHA-256 verify the [sealed report](product-e9a5f55f-installed-darwin-evidence.txt),
[full log](product-e9a5f55f-installed-darwin-test-quick.log),
[bootstrap log](product-e9a5f55f-installed-darwin-bootstrap.log) and
[tool log](product-e9a5f55f-installed-darwin-tools.log). Their hashes match the
peer's report. I independently match the source Git tree
`f6d52efa59298c58f981e1b3d06c1a5b362bf724` and Makefile SHA-256
`a72854943fb12f8a32e05bb70742c5022585f714d6767a2106975d0583b1a08a`.
The report preserves compiler selection during actual component compilation
separately from pre-gate and final binaries. The detached peer checkout remains
clean. Both platform failures block this candidate despite passing fixed points.

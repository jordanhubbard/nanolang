# My installed-canonical candidate acceptance

I freeze production `e9a5f55f9b82b3de8c03b3df240665865fd25e5c`, integrating canonical main through PR712. This includes the qualified consuming runtime/source calls, managed mutable arrays, exact binary64 facts/text and native optional-map-key repair. Later main changes, including VM slice cleanup PR715, are outside this pin. I do not release this candidate or close the full roadmap.

The failed63c Linux gate and its independently passing focused/fixed-point/Darwin evidence remain in [their original record](product-acceptance-63c26ecd.md). The optional-key repair passes a fresh unchanged63c transpiler compile with a corrected translator and subsequent native entry assertions; that bounded repair does not substitute for this combined product gate.

I use isolated detached Linux worktrees for the full quick gate, VM fixed point and native fixed point. Each records exact source and tool identities. The quick runner explicitly performs bootstrap and tool builds before `make test-quick`, asserts that `bin/nanoc` resolves to Stage2, and retains compiler identities before each phase. Its bootstrap and tool builds pass; full acceptance is running. My Darwin peer has completed bootstrap and the requested tool build at the same pin, recorded installed Stage2 selection, and reached canonical component compilation under the full PTY quick gate. Both platform gates remain active with unchanged budgets.

The fixed-point generations retain their prior per-stage bounds: VM1200 seconds and native1800 seconds, 48 GiB owned RSS and 32 GiB minimum host-available memory. Each route has independent embedded library paths; no cross-run byte-equality claim is implied. Initial seeds build successfully; generation qualification is running.

Local manifests/logs are under `/tmp/nanolang-product-quick-e9a5f55f`, `/tmp/nanolang-product-vm-fixedpoint-e9a5f55f`, and `/tmp/nanolang-product-native-fixedpoint-e9a5f55f`. I will seal terminal results without modifying active checkouts. Broader ownership, reconstruction, managed runtime, formal correspondence, component-shadow, clean full tests and documentation gates remain open.

My separate canonical component-shadow report (`task_14c8ecbd8aaa484ea5e73f2aa43fa48b`) has a pushed pre-execution contract at `4cbffa9e`. It uses its own checkout and fresh tools, captures the ordinary verified shadow modules through the supported VM hook, and preserves normal shadow supervision. It does not modify these acceptance runs.

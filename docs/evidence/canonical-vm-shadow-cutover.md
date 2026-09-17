# My canonical bytecode shadow cutover

My `--emit-nvm` route lowers one checked, bound parser into production and selected-shadow modules. I verify the staged shadow module, invoke `nano_vm --check-shadows`, clean temporary storage, and publish verified production bytes only after successful completion. This route returns before legacy C transpilation. My default native product remains separate architecture work.

`NANO_VM` selects an explicit runner; otherwise I use `bin/nano_vm` under my discovered repository root. I quote runner and module paths and preserve the runner's parent deadline. All results below use the unchanged default ten-second shadow budget.

## My focused and bootstrap evidence

A fresh native bootstrap passes. All 14 canonical publication and VM-shadow methods pass through Stage 1 in 11.424 seconds and Stage 2 in 14.978 seconds. They cover shared global state, ordered shadows, ordinary main, dependency/root-only selection, assertions and deadlines, previous-output preservation, missing runner, no-shadow compilation, runner paths with spaces, artifact identity and supported scalar floats. The focused bytecode route accepts a rejecting native-compiler stub. Logs: `/tmp/nanolang-canonical-current-{bootstrap,stage1,stage2}.log`.

My complete compiler VM fixed-point gate passes at source **e35d8f55a20477d1917ed26abe3cd586409130c9** in **1067.128 seconds**. Both self-hosted generations produce identical raw **365,976-byte** modules:

```text
5523c1217027790d49112d6e3dc1539b4e68f65780eef4b6f99d3f4eac32e543
```

Generation 1 takes 524.756 seconds; generation 2 takes 526.455 seconds. Both modules pass verification. The second-generation compiler compiles hello, whose output verifies and executes successfully. Source, capture helper and the exact host-library paths/hashes remain unchanged.

I retain the complete manifest, modules, disassembly and stage logs in `/tmp/nanolang-vm-fixedpoint-identity/`; the gate log is `/tmp/nanolang-vm-fixedpoint-identity.log`.

## My native host boundary

The same compiler wrapper identity is present before seed generation and through both VM generations. During VM generations the guard rejects NanoLang-generated C compilation while allowing declared native host-artifact inputs, their private cache snapshots and compiler identity/cache queries. The passing run records **140 host queries/preprocessing probes** and **380 native host-artifact calls**, with **zero rejected or NanoLang-generated C compilation calls**. This is instrumentation of the selected bootstrap path, not a security sandbox or a claim that no host C compiler runs.

My end-to-end guard test compiles an explicitly declared native host input, refuses a generated-product input, and admits the compiler identity queries used by the cache. Earlier harness attempts incorrectly denied native host work or identity queries. Their retained directories are `/tmp/nanolang-vm-fixedpoint-{c7e06a36,36e70d53,1aba81ce,host-allowlist}/`; they are failed/stopped harness evidence, not successful product gates. The last of those completed its first generation but selected different host-cache paths after a denied version query. I corrected the harness without relaxing raw module equality or exact host-closure checks.

This closes the bounded VM-shadow cutover task `task_c5a7a4835d364b50b747018c794a07d0` at the recorded pin. It does not establish native full-source self-compilation, removal of the default native C backend, current future-source reproducibility, semantic correctness or full release acceptance.

After integrating main through PR525 at `7e341377`, a fresh default-budget native bootstrap passes. Stage 1 passes all 14 canonical publication/shadow methods in 17.294 seconds; Stage 2 passes those methods plus the guard test (15 methods) in 16.647 seconds. Logs: `/tmp/nanolang-canonical-integrated-main-{bootstrap,stage1,stage2}.log`. The full bytecode fixed-point measurement above remains explicitly pinned to `e35d8f55`; I do not relabel it as a measurement of later source.

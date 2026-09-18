# My NanoISA-only product draft

I route explicit native and C-source products through the same verified NanoISA lowering as bytecode. My driver no longer imports `transpiler.nano`; native and bytecode products execute selected shadows through NanoVM. Source-only C emission keeps its existing no-execution behavior. I stage native products beside their destination and publish only after translation and host compilation succeed.

My C-seed-built driver passes all 14 canonical publication and VM-shadow methods in 8.358 seconds. Two new product methods pass in 0.299 seconds: native and source routes invoke `nvm2c`, rebuilt C executes, paths with spaces work, temporary staging is cleaned, and failures of the VM runner, translator, runtime lookup or C compiler preserve prior output. Logs: `/tmp/nanolang-only-product-{existing,routes}.log`.

My fresh `make bootstrap` builds and executes Stage 1. Stage 2 reaches verified module translation but stops because the translator scans all generated C for `nano_vm`, including the compiler's legitimate shadow-runner path string. I retain `/tmp/nanolang-only-product-bootstrap.log` and task `task_b4a32a35810544f4a0234ab5828b46b4`.

This draft is not cutover acceptance. Module introspection must move from removed C helper generation into canonical NanoISA facts/lowering (`task_faa47ec22a2545348aa9c9d705580321`). Default no-output CLI policy, canonical phase naming, ownership/reference IR coverage, full native self-compilation resource practicality and the final current-source bytecode fixed point remain open. I do not publish a release from these focused results.

With the wrapper-string checker correction, my next bounded bootstrap creates Stage 2 successfully. Its ordinary `examples/language/nl_hello.nano` smoke invocation then aborts without a compiler diagnostic. I retain `/tmp/nanolang-only-product-bootstrap-wrapper-fixed.log`, the Stage 2 executable, and task `task_57eb1d541d084ec4930b8764c591bad3`. This supersedes the translation blocker for that attempt, but does not pass native product acceptance.


## My integrated product gate after typed projection repair

At clean pin `599d75585521b6734f4227e66ae3049eefde9dbb`, I combined the module-facts, default-output and lowering-phase drafts with main through PR538 and the PR540 host-manifest repair. Fresh tools built; Stage 1 built and passed hello. Stage 2 translation stopped with `I cannot yet store an aggregate or unresolved global in function 125 at offset 4`. My new `array<ModuleIntrospection>` global requires record-array global transport that native lowering currently refuses. Task `task_1e569db4d8f1486abdd7d5ed3ca00bc1` records this before repair.

The log is `/tmp/nanolang-product-integrated-bootstrap.log`. This is an explicit unsupported product boundary before the later native startup check, not evidence that the original abort has recurred or been resolved. The original Stage2 binary is preserved separately at `/tmp/nanolang-product-startup-original-stage2`, SHA-256 `24f6036049c1376b3d2f709e1af799d74ebc70d8985539d07a95ba0ada876b51`. I have not run it again.


At product source `1fae65efabc0598ea66f031f94882fdba7c4f2f7`, after merged PR558, I pass fresh bootstrap, both hello stages and installed execution without my C seed. My ordinary 27-method product gate passes 26 methods in 14.692 seconds. Existing export-shadow compilation still terminates with SIGABRT and no diagnostic. The inference and failed-lowering-state repairs do not establish a cause or resolution. I retain the new compiler, logs and hashes at `/tmp/nanolang-product-exports-1fae65ef`; I do not replay the old binaries. Task `task_dd74b033c3984805bc27ce5017096c3c` and product publication remain open.

At `bd4a2428bd3d8b98d1f1e6d2cb064bde3eb5f456`, after projected-global, implicit-return and worklist repairs, fresh bootstrap still passes. The full ordinary gate again passes 26 of 27 methods in 13.588 seconds; export-shadow compilation still aborts without a diagnostic. I retain this run separately at `/tmp/nanolang-product-exports-bd4a2428`. These independently validated repairs do not resolve or explain dd74. I have not started a new product fixed-point run while this gate remains blocked.

My independent full-source VM fixed-point gate passes at `ae63b248`: initial and both generations match raw 259,428-byte modules, with verified second-generation hello and unchanged host closure. [Pinned evidence](product-vm-fixedpoint-ae63b248.md). The export-shadow product gate remains separate and unresolved.

## My current integration checkpoint

I integrated main through PR592 at `d332ea1f`, preserving the pinned product VM fixed-point evidence and the export-shadow publication hold. The ordinary C-seed, NanoVirt, VM, native translator and NanoISA CLI build passed (`/tmp/nanolang-product-current-integration-build.log`). This is compilation evidence only; I did not rerun historical aborting artifacts or claim new full product acceptance. Draft PR584 and task dd74 remain unresolved.

At integrated product `9808acf6` (main through PR597), a fresh ordinary build
and Stage1 hello pass. Stage2 translation explicitly refuses aggregate storage
`optional` to `int` at nodes 225121/8641. I retain
`/tmp/nanolang-product-through597-bootstrap.log` and record
`task_497e1ba5b9544b81b3614ec37da90e98` before repair. The focused optional-array
checks did not establish full compiler storage acceptance. This refusal is
separate from historical export-shadow aborts, which I did not replay.

# My NanoISA-only product draft

I route explicit native and C-source products through the same verified NanoISA lowering as bytecode. My driver no longer imports `transpiler.nano`; native and bytecode products execute selected shadows through NanoVM. Source-only C emission keeps its existing no-execution behavior. I stage native products beside their destination and publish only after translation and host compilation succeed.

My C-seed-built driver passes all 14 canonical publication and VM-shadow methods in 8.358 seconds. Two new product methods pass in 0.299 seconds: native and source routes invoke `nvm2c`, rebuilt C executes, paths with spaces work, temporary staging is cleaned, and failures of the VM runner, translator, runtime lookup or C compiler preserve prior output. Logs: `/tmp/nanolang-only-product-{existing,routes}.log`.

My fresh `make bootstrap` builds and executes Stage 1. Stage 2 reaches verified module translation but stops because the translator scans all generated C for `nano_vm`, including the compiler's legitimate shadow-runner path string. I retain `/tmp/nanolang-only-product-bootstrap.log` and task `task_b4a32a35810544f4a0234ab5828b46b4`.

This draft is not cutover acceptance. Module introspection must move from removed C helper generation into canonical NanoISA facts/lowering (`task_faa47ec22a2545348aa9c9d705580321`). Default no-output CLI policy, canonical phase naming, ownership/reference IR coverage, full native self-compilation resource practicality and the final current-source bytecode fixed point remain open. I do not publish a release from these focused results.

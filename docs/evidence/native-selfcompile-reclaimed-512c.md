# My native self-compilation attempt after string reclamation

I keep MAC `task_fc43d8d1923b40ebb343ae56da535dfc` open. I rebuilt my canonical native compiler through the repaired translator and passed its help and hello-bytecode controls. Its one full-source attempt did not finish within my recorded memory budget. I neither retry this attempt nor count it as a correctness failure or fixed point.

## My exact inputs and passing prerequisites

My clean source pin was `ca12067c78a03f4fff54c66f7bd6ae01fdb86175`, a roadmap-only commit over code `512c2ec6f957765be50b27988cb184f58cab0163`, in `/home/jkh/Src/nanolang-native-selfcompile-reclaimed`. My prerequisite build passed for `bin/nanoc_c nano_virt nvm2c nanoisa_dump nano_vm nvm2c-runtime`. I removed `NANO_SHADOW_TIMEOUT_SECONDS`, retained normal C shadows and used that checkout's capture helper and module directory. The later NanoISA-only product and VM-shadow cutover are separate code and acceptance pins.

| Command stage | Observed seconds | Result |
| --- | ---: | --- |
| C-seed build of canonical frontend | 31.874 | passed |
| Canonical frontend emits full compiler module | 142.484 | passed |
| Verify initial compiler module | 0.225 | passed |
| Repaired `nvm2c` translates initial module | 4.224 | passed |
| Strict C11 host build of native compiler | 126.642 | passed |
| Native compiler help | 0.003 | passed |
| Native compiler emits hello bytecode | 1.719 | passed |
| Verify and execute hello in VM | 0.006 combined | exact greeting |

My initial module is 366,532 bytes with SHA-256 `2f30759d8c55a60a5081cd84d5cc59fa10d247aaef34432a21fa34f630e1d43b`. Its generated native executable is `/tmp/nanolang-native-selfcompile-reclaimed-512c/initial-native`, SHA-256 `b3ace0c8f33366380d320e170062d0b02cde0ed7c8e88bb731df04d12f96f865`. The manifest retains generated C, translator, capture helper and all three immutable host library paths/hashes too. These checks do not establish full native self-compilation.

## My one full-source attempt

With the same helper/module environment and working directory, I ran:

```sh
/tmp/nanolang-native-selfcompile-reclaimed-512c/initial-native \
  /home/jkh/Src/nanolang-native-selfcompile-reclaimed/src_nano/nanoc_v06.nano \
  --emit-nvm -o /tmp/nanolang-native-selfcompile-reclaimed-512c/stage1.nvm
```

I retained an 1800-second stage budget and sampled the owned process tree every five seconds. My external guard set a 48 GiB owned-RSS threshold and a 32 GiB minimum host-available reserve. At **1680.187 seconds**, a sample reached **50,832,364 KiB** (48.48 GiB); host available memory was still **53,534,968 KiB** (51.06 GiB). The sampling guard sent SIGTERM to the owned process group. Termination completed at **1688.995 seconds** (28 minutes 9 seconds), status `-15`. The sampled threshold can be crossed between samples; this was not an operating-system hard allocation cap.

I produced no Stage 1 artifact and the compiler log remained empty. Verification, a second native generation and raw equality comparison therefore did not run. I verified after stopping that the source commit and clean tree, initial compiler/module/C, capture helper, translator and immutable host library hashes were unchanged.

This attempt is incomplete resource-budget evidence. My small concat regression establishes bounded temporary-string pool retention; it does not attribute this full compiler's RSS or establish a whole-program memory bound. Static record/array owner retention is separately recorded under `task_7ff98f74f0ee40afba708605c01fad52`; host-result adoption remains `task_d5f899966241452a900422938fff3265`. I make no before/after full-compiler speed comparison across different source pins.

## My retained evidence

`/tmp/nanolang-native-selfcompile-reclaimed-512c/` contains `manifest.json`, `memory-samples.jsonl`, `post-stop-integrity.json`, each stage log, initial module/assembly/C/executable and hello bytecode. The external runner is `/tmp/nanolang-native-selfcompile-reclaimed-run.py`, its log is `/tmp/nanolang-native-selfcompile-reclaimed-run.log`, and my build log is `/tmp/nanolang-native-selfcompile-reclaimed-build.log`. No historical fault fixture or diagnostic retry was executed.

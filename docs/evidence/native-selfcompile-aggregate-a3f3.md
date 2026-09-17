# My native acceptance prerequisite after aggregate reclamation

I attempted one fresh acceptance setup at merged code `a3f321a62446f68b6110029009eb1e564e361f52`, with roadmap-only source commit `29fd7ef0`. The clean isolated checkout was `/home/jkh/Src/nanolang-native-selfcompile-aggregate`. My tool build passed for `bin/nanoc_c nano_virt nvm2c nanoisa_dump nano_vm nvm2c-runtime`.

My first prerequisite command was:

```sh
bin/nanoc_c src_nano/nanoc_v06.nano -o /tmp/nanolang-native-selfcompile-aggregate-a3f3/seed
```

It exited 1 after 31.622 seconds, with sampled peak owned RSS 167540 KiB. Native linking reports missing `nvm_layouts_have_facts`, `nvm_retained_layouts_valid` and `nvm_retain_layouts` from the v2-module and verifier objects. The nanoisa facade manifest includes those dependent sources but omits the new retained-layout implementation. MAC `task_8b0906d2885749b88bc486ee5a7c7d5d` records this closure defect before repair.

I did not reach initial compiler bytecode, native help/hello, or either full-source generation. I did not retry. This prerequisite failure establishes neither a full compiler correctness failure nor resource acceptance. The full native acceptance task `task_fc43d8d1923b40ebb343ae56da535dfc` remains open. This pin includes VM shadows, so I make no speed comparison with older C-shadow source.

I retained the runner, build log and `/tmp/nanolang-native-selfcompile-aggregate-a3f3/` manifest, memory samples, seed-build log and post-stop integrity checks. Source stayed clean and unchanged; helper and translator hashes match the initial manifest, and no initial/stage1 module exists. No host-library closure was produced to hash. The planned 48 GiB owned-RSS threshold, 32 GiB host reserve and 1800-second generation caps never became a limiting condition.

## My manifest closure repair

In a separate fresh checkout, I added `src/nanoisa/retained_layouts.c` to the nanoisa facade's declared C sources. I changed no ABI, symbol-resolution rule or shadow policy. A fresh C-seed build and the same canonical source native compilation now succeed. The resulting seed prints help and publishes hello bytecode through normal VM shadows; verification and execution succeed with `Hello from NanoLang!`.

My first extra hello invocation in this repair checkout lacked `bin/nano_vm` because I had built only the seed target; it correctly refused publication. I retained that log, built the required VM tool, and then performed the successful hello check. This prerequisite correction changes no product source or assertion. Repair logs use `/tmp/nanolang-retained-layout-*`; the original isolated acceptance evidence is untouched. I have not started a further full-source acceptance run in this repair checkout.

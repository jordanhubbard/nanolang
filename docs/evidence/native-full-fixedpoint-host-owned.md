# My native full-source fixed point after host ownership repair

I tested task `task_fc43d8d1923b40ebb343ae56da535dfc` at frozen source `2f50d7a02c8d78f4eefb05d398d52db0c86d936e`, based on merged code `61b0aedc8ccb5b3e8817325568018153ffd43e8d`. The source-only difference records the acceptance contract. My checkout remained clean throughout this single attempt.

I built my C-seed-hosted canonical compiler, emitted its complete compiler module with normal VM shadows, translated that module through `nvm2c`, and linked a native compiler. After its help and verified hello prerequisites passed, that native compiler compiled the complete compiler source to generation one. I verified and translated generation one, linked its native compiler, checked help, and compiled the same complete source to generation two. Both generations completed within their original 1,800-second budgets. I made no retry or budget extension.

| Stage | Wall seconds | Peak monitored process-group RSS (KiB) | Status |
| --- | ---: | ---: | --- |
| seed-build | 33.627 | 320744 | 0 |
| initial-emission | 140.079 | 6625960 | 0 |
| initial-translation | 4.386 | 600 | 0 |
| initial-native-build | 136.193 | 2290048 | 0 |
| stage1 | 1272.849 | 3612392 | 0 |
| stage1-translation | 4.386 | 1476 | 0 |
| stage1-native-build | 136.684 | 2287000 | 0 |
| stage2 | 1367.801 | 4017228 | 0 |

The monitor retained a 48 GiB owned-RSS limit and 32 GiB host-available reserve. It sampled the process group; these measurements are not instantaneous allocation peaks. Neither limit stopped the run. I make no speed attribution against earlier source pins.

My initial, generation-one and generation-two modules are all **378144 bytes**, with SHA-256:

```
616ab7dd3fca6c0a5d9a7a4598e1c7042e70fe1d849d9dbbfed8e81d8c9d4bb2
```

I checked raw generation-one/generation-two byte equality, verified both modules, and checked all three immutable imported libraries for equal paths and hashes. My final hello compilation, verification and execution passed with `Hello from NanoLang!`. The post-run integrity check confirmed unchanged source, capture helper, translator, initial native executable and host libraries.

## My retained evidence

The immutable checkout is `/home/jkh/Src/nanolang-native-selfcompile-host-owned`. My runner is `/tmp/nanolang-native-selfcompile-host-owned-run.py`; its log is `/tmp/nanolang-native-selfcompile-host-owned-run.log`. I retain artifacts, exact argument vectors, exit codes, timestamps and measured resource peaks under `/tmp/nanolang-native-selfcompile-host-owned-61b0/`: `manifest.json`, `memory-samples.jsonl`, `post-run-integrity.json`, per-stage logs and all generated modules/C/executables. These local paths are retained evidence, not portable build inputs.

My exact full-generation commands use the same source checkout:

```sh
NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so" NANO_MODULE_PATH="$PWD/modules" \
  /tmp/nanolang-native-selfcompile-host-owned-61b0/initial-native \
  "$PWD/src_nano/nanoc_v06.nano" --emit-nvm -o /tmp/nanolang-native-selfcompile-host-owned-61b0/stage1.nvm
NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so" NANO_MODULE_PATH="$PWD/modules" \
  /tmp/nanolang-native-selfcompile-host-owned-61b0/stage1-native \
  "$PWD/src_nano/nanoc_v06.nano" --emit-nvm -o /tmp/nanolang-native-selfcompile-host-owned-61b0/stage2.nvm
```

The external runner enforces the bounds above; I do not disable shadows or increase their default deadline. My [retained manifest and final integrity snapshot](native-full-fixedpoint-host-owned.json) are authoritative for every invocation.

| Immutable component | SHA-256 |
| --- | --- |
| Capture helper | `51d8fe2da58d680c31bfe37172e79045eea256df80be38bb31aebff84ca7f4c6` |
| Translator | `b5d863e60a1363d6c2944e07d0d87b2d54bf8684bb0d7811ad5ae15991ad7bdd` |
| Initial native compiler | `6c3457efab5d2d5e565afc85909aa3b35b28d856acf15a60a4224c751cfcf99a` |
| `/home/jkh/Src/nanolang-native-selfcompile-host-owned/modules/compiler_support/.build/.nano-gen-US9pCp/libcompiler_support.so` | `f34f77c929a25241724b8d369ffdc9a579ef168400d124fb8dfbfaa2f99cbee5` |
| `/home/jkh/Src/nanolang-native-selfcompile-host-owned/modules/nanoisa/.build/.nano-gen-Kf6c76/libnanoisa.so` | `567888b0c4af5fc8a3caf27199b2cbba48b576d93578ad3ef0f76ee7a83b52bd` |
| `/home/jkh/Src/nanolang-native-selfcompile-host-owned/modules/std/.build/.nano-gen-kdehIQ/libstd.so` | `238983bcf2d0188b358517bdd99a8a6d3b4a179b3128aeaa8e45a9c73ef6f283` |

This is a tested native full-source compilation and raw fixed point for this pinned compiler and host closure. Absolute immutable library paths are embedded in the modules, so I do not claim a cross-checkout reproducible hash. I do not close the separate default-product cutover, full LLVM/Wasm profile, passive-language or reference ownership release requirements with this result. Earlier capped attempts remain recorded as incomplete resource/time outcomes.

# My bounded native self-compilation attempt

I keep MAC `task_fc43d8d1923b40ebb343ae56da535dfc` open. I did not finish full native self-compilation within my recorded memory budget. This result is neither a correctness failure nor a passing compiler acceptance result.

## My immutable inputs

I ran source checkout `62c9d758f414a62655eb8f363042e84b77be5b0d`, a roadmap-only commit over code `8666cb095c9da23a115f41c6d83aa0dafe5be579`, in `/home/jkh/Src/nanolang-native-selfcompile`. My retained canonical native compiler came from the successful ordinary compiler-product gate at that code pin:

- Executable: `/tmp/nanolang-native-readiness-8666/nano-selfhost-native-product-it_bqyi8/compiler`
- Executable SHA-256: `5ef0dafc4b977eaa40bc2b15c9c8f5ad51b6c254c651aa1d8e85ee5d05f3873e`
- Compiler module SHA-256: `4935fb3c3f4c01795a38ebe2efa65431e529f64a9e39046de40d8542a9ce9113`

I successfully built `bin/nanoc_c nano_virt nvm2c nanoisa_dump nano_vm nvm2c-runtime` before this attempt. I kept normal shadows enabled, removed the shadow-timeout override, and used the checkout's capture helper and module directory. I ran once:

```sh
NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so" \
NANO_MODULE_PATH="$PWD/modules" \
/tmp/nanolang-native-readiness-8666/nano-selfhost-native-product-it_bqyi8/compiler \
  src_nano/nanoc_v06.nano --emit-nvm \
  -o /tmp/nanolang-native-selfcompile-8666/stage1.nvm
```

## My measured boundary

I retained an 1800-second stage budget and monitored the owned process tree every five seconds. My external guard capped owned resident memory at 48 GiB and reserved at least 32 GiB of host available memory for other work.

The owned RSS cap stopped Stage 1 with SIGTERM after **1419.248 seconds** (23 minutes 39 seconds). At the triggering sample, the process tree used **50,370,432 KiB** (48.04 GiB); the host still reported **59,355,044 KiB** (56.61 GiB) available. The runner recorded signal status `-15`. I terminated only the owned test process group.

I produced no Stage 1 module, and the compiler log remained empty. I therefore did not attempt verification, Stage 2, raw byte comparison or a generated Stage 1 product check. I did not rerun the workload or infer a leak, deadlock or correctness defect from this measurement. Full native self-compilation practicality remains unmet. Canonical VM bootstrap and the pending VM-shadow cutover have separate evidence and ownership.

After stopping, I verified that my source commit and clean checkout, initial executable, capture helper and three immutable host libraries were unchanged. My retained manifest records their exact paths and hashes.

## My retained evidence

My host-local evidence is under `/tmp/nanolang-native-selfcompile-8666/`: `manifest.json`, `memory-policy.json`, `memory-samples.jsonl`, `resource-budget-stop.json`, `post-stop-integrity.json`, `initial.nasm` and the empty `stage1.log`. The external runner and monitor are `/tmp/nanolang-native-selfcompile-run.py` and `/tmp/nanolang-native-selfcompile-memory-guard.py`; their logs use the same names with `.log`. My prerequisite build log is `/tmp/nanolang-native-selfcompile-build.log`.

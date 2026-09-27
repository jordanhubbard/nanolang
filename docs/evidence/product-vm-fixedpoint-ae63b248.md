# My product compiler VM fixed point

At source `ae63b24817313e21412b0aefc4242a7f7f196188` on Linux ARM64, I compile my full canonical compiler through two consecutive NanoVM generations. My seed-produced module and both VM-produced modules are byte-identical: **259,428 bytes**, SHA-256 **b02452f60a3058158120499f7eabfeb31fbc0768b3cddf4c9727aecc21efb05a**. An independent hash check agrees with the runner.

| Step | Seconds | Peak owned RSS, KiB |
| --- | ---: | ---: |
| C seed builds canonical driver | 31.772 | 84,428 |
| Driver emits initial compiler module | 104.889 | 4,117,172 |
| First VM compiler generation | 400.406 | 138,292 |
| Second VM compiler generation | 403.119 | 98,688 |

Every command exits zero. All three modules verify. The second VM generation compiles my unchanged hello example; that module verifies and prints `Hello from NanoLang!` with its expected newline.

Each VM generation has an explicit 1,200-second limit. The runner retains a 48 GiB owned-process RSS cap and 32 GiB host-memory reserve. I do not retry, normalize paths or canonicalize bytes. The manifest's top-level 1,800-second setting is its generic runner default; the per-stage 1,200-second budgets govern these VM commands.

My initial and generated modules retain identical absolute host-library paths and content hashes. My source commit, clean tree, capture helper, translator and host libraries remain unchanged through the run. Absolute host paths mean this raw hash does not establish cross-checkout reproducibility. Host artifact compilation uses the ordinary toolchain; this run does not repeat PR497's compiler-call instrumentation.

I retain the measured commands, budgets, hashes and integrity check in [the manifest](product-vm-fixedpoint-ae63b248.json). Original logs and artifacts remain in `/tmp/nanolang-product-vm-fixedpoint-ae63b248`; the runner is `/tmp/nanolang-product-vm-fixedpoint-ae63b248.py`.

This is a pinned VM fixed point, not native self-compilation at this source or proof of compiler semantics. PR563 records native fixed-point evidence at its separate source pin. Product export-shadow acceptance still fails at the preceding `bd4a2428` ordinary gate (task `task_dd74b033c3984805bc27ce5017096c3c`); I have not replayed that case here. PR522, its stacked task closures and full release publication remain held.

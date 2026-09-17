# My bounded native compiler acceptance at b09a16a8

I tested merged code `b09a16a869e52e423e7221ce870af207abce39e8`, including aggregate reclamation, retained-layout facade closure and typed integer projections. My roadmap-only immutable source pin was `31a5dda2424ce80a9660804a8ed0d0e850e3f7b9`, in `/home/jkh/Src/nanolang-native-selfcompile-closure`.

## My result

One native full-source generation reached the unchanged 1800-second deadline. I stopped its process group with SIGTERM; it exited -15 and finished reaping at 1800.415 seconds. No stage-one artifact or diagnostic was produced. I did not retry, extend the budget, start generation two or claim raw equality. Full native acceptance `task_fc43d8d1923b40ebb343ae56da535dfc` remains open.

Peak sampled owned RSS was 4,367,660 KiB (about 4.17 GiB). The final sample was 2,329,892 KiB, with host available memory 110,009,048 KiB. Neither my 48 GiB owned-memory threshold nor 32 GiB host reserve triggered. These are sampled process-tree RSS figures, not exact allocation ownership. This result is incomplete time-budget acceptance, not a demonstrated compiler correctness failure. The source now uses VM shadows; I make no speed attribution against earlier C-shadow pins.

## My passed prerequisites

| Step | Seconds |
| --- | ---: |
| Fresh canonical native seed | 32.373 |
| Full compiler NanoISA emission | 133.936 |
| Compiler bytecode verification | 0.214 |
| Native translation | 4.325 |
| Strict C11 native compiler build | 130.130 |
| Native compiler help | 0.002 |
| Native compiler hello emission | 0.064 |
| Hello verification | 0.002 |
| Hello VM execution | 0.004 |

The hello output was exactly `Hello from NanoLang!` followed by a newline. The initial module contains 369508 bytes. I used normal shadows and no shadow-timeout override. My actual generation-one command was:

```sh
/tmp/nanolang-native-selfcompile-closure-b09a/initial-native \
  /home/jkh/Src/nanolang-native-selfcompile-closure/src_nano/nanoc_v06.nano \
  --emit-nvm -o /tmp/nanolang-native-selfcompile-closure-b09a/stage1.nvm
```

## My retained identities

| Artifact | SHA-256 |
| --- | --- |
| Initial module | `fbe1c5fc61d032cc754229b30022bff53c3e9e4db543a0c207c05e21c7ef9349` |
| Generated C | `dd040891ef7e4a80d708e1f4779fea3a1b6adbbf984f20b1186dca499c300f1d` |
| Native compiler | `2242d5dcbd7d866ec2079a4afb8e7a7fc0e4d54f80b4ecb26fc56e1c1b1e70b8` |
| Capture helper | `51d8fe2da58d680c31bfe37172e79045eea256df80be38bb31aebff84ca7f4c6` |
| Translator | `19255f14354dd3c2f4a00f25bec6794e914bce40f0ab052db569edd5c9d9c5a2` |

My manifest also retains exact immutable paths and hashes for compiler_support, nanoisa and std host libraries. Post-stop checks confirm unchanged source/clean checkout, helper, translator, module, generated C, native binary and all three libraries. Local evidence is `/tmp/nanolang-native-selfcompile-closure-b09a/`, including `manifest.json`, `memory-samples.jsonl`, every stage log and `post-stop-integrity.json`; the runner is `/tmp/nanolang-native-selfcompile-closure-run.py`. I retain both the original missing-manifest prerequisite and this corrected attempt separately.

MAC `task_c7931f1c22db473682d077b119b9c87d` records the next bounded CPU-cost investigation. I will inspect ordinary allocation/traversal and compiler operations, measure isolated workloads, and distinguish a demonstrated local cost from an unproved attribution of the full-source timeout.

## My bounded CPU-cost candidate

I inspected the retained generated C without rerunning full compilation. `nmap_owned_get` marks allocation debt whenever it creates an owned copied string; `nmap_owned_new` does the same for a map. `nmap_collect_if_needed` treats this debt as a Boolean, so one such allocation causes a complete reachable graph trace at the next eligible safe point. The newer temporary-string and aggregate pools instead use byte budgets. This is a candidate for measured batching, not evidence that collection is unnecessary.

I generated a small ordinary module with a record array and string map, verified/executed its bytecode, and instrumented only counters in its emitted runtime. My bounded harness retains the array and map, performs 500 successful copied-string reads, runs the existing safe-point collector, checks retained values and final zero live owners, and measures CPU time at `-O0`:

| Reachable records | Reads | Collections | Root visits | CPU seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 500 | 500 | 501500 | 0.032586 |
| 5000 | 500 | 500 | 2501500 | 0.168035 |
| 10000 | 500 | 500 | 5001500 | 0.374575 |

The local cost scales with reads multiplied by the reachable graph. I retain assembly, bytecode, unmodified generated C, instrumented harness and output in `/tmp/nanolang-native-cpu-probe/`. This normal-value workload executes in under one second here and is not a compiler failure reproduction. The stopped full compiler has no CPU profile, so I do not attribute its timeout to this path alone. Repeated whole-string `strlen` in `nstr_char_at`/`nstr_substr`, invoked from lexer loops, is another static candidate without an isolated measurement here.

The next implementable slice is checked allocation-byte accounting for owned map headers and copied map results, with a justified collection budget at existing published-root safe points. It must retain forced collection, graph closure, aliases and bounded garbage; it must measure both allocation churn and a large live graph before acceptance. Any pooling or cross-pool budget change needs explicit retention evidence. I have not changed collector policy in this diagnostic.

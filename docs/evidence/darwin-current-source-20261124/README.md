# I retain three passing Darwin profiles and a failed full-source gate

Under #1121/#1116/#1247 I qualify exact `e6a7c3bf634e15b5eb13a2e98feeb93797ef6f27` in new host-local Darwin ARM64 roots. My compiler/runtime inputs equal Linux-qualified `8949313b0a56c0c3bd52509fb363edb4eefba3e4`; the only selected test change exposes Darwin declarations in the rollback fixture. I preserve the first fixture-failure root and every older qualification. My normal source checkout has a new isolated Git store; sanitizer and production checkouts are detached worktrees in that new store. No existing Puck checkout, common Git store or NAS path is used.

| Profile | Complete fresh build | Original focused targets | Private REPL cohort |
| --- | ---: | ---: | --- |
| normal O1 | 211.084s | 114.809s | 41 methods, one platform skip, PASS |
| ASan/UBSan O1 | 422.424s | 124.550s | 41 methods, one platform skip, PASS |
| production O3 | 114.098s | 53.034s | 41 methods, one platform skip, PASS |

Each begins with absent `obj`, `lib` and `bin`, builds my actual C seed, VM, assembler and C translator with serial original Make recipes, and preserves the original1800-second outer build limit. Focused targets retain their600-second outer limit and cover original GC/fault refusal, environment60, lexical10, source-C20/API, checker context, module rollback and callable/state/session REPL methods. The normal companion-tool control passes twenty original commands. Each existing externally enabled private collector runs58 actual REPL calls at the unchanged30-second command deadline; the original Linux descriptor-fault observer is skipped on Darwin. I change no assertion, optimization, compiler pin or permission to obtain these passes.

My actual installed tools are Apple Clang21.0.0 and GNU Make3.81, with the original SDK/OpenSSL/libffi flags. Make3.81 lacks `--trace`, so I retain its original recipes/output and a separate variable plan. Darwin keeps its original `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`; these passes do not establish LeakSanitizer acceptance.

After the profiles, three CPU-share rounds pass90.5/87.0/83.5percent before the original ordinary compiler-source command. It reaches the unchanged **600-second outer timeout** in600.073064s, exit-9. There are **zero shadow-start, shadow-complete or supervisor-phase events**. None of the1,306 shadow cases is established as executed. Child accounting retains373.254user and149.183system seconds. Sampled descendants are Clang processes; I do not infer the timeout cause from these observations. No final seed is published. The private full-source companion is **NOT RUN**.

This failure differs from #1229's historical terminal after1,306 completions and #1234's two missing-tool assertions. I preserve the complete unsuccessful controller result and all successful predecessors, source/tool guards, generated module/cache files, native providers, actual tool images, original commands/logs, private-case stdin/output and every file present at terminal. Original test/driver cleanup can remove intermediate files; I do not claim bytes already removed by that cleanup. I do not archive every SDK/system header or claim a hermetic operating-system closure.

My independent reader verifies **17,144 records,3,520 selected Git sources and86 distinct native ARM64 images**, all declared successes and the failed full-source boundary. The image count describes archived inputs/products, not executions. It checks8MiB content chunks and their compressed identities, actual source bytes against Git, original profile flags and limits, private58-call cohorts, unchanged production inputs, absent final seed and the unrun companion. A passing reader means archive integrity and accurate result classification; the full qualification remains unsuccessful.

```sh
python3 docs/evidence/darwin-current-source-20261124/read_archive.py
```

I keep the frozen qualification at `/private/tmp/nanolang-source-qual-20261124-tgs3q7n7` and independently copied archive on local ext4. Publication uses a separate local worktree, and my shared source checkout excludes this new archive path while the NAS copy pause remains. I reclaim nothing. I release the Darwin native reservation and perform no unchanged full retry. #1247 retains bounded pre-shadow diagnosis and unchanged ordinary/private acceptance. Default ownership activation, final paired source/raw/bootstrap/installed acceptance and all69 release requirements remain open.

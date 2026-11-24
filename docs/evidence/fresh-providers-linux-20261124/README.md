# I retain complete fresh Linux providers and checker ownership repairs

I qualify source `8949313b0a56c0c3bd52509fb363edb4eefba3e4` in three separate detached Linux AArch64 worktrees. Each begins with absent `obj`, `lib` and `bin` directories and uses the original serial Make build with its default output paths, strict C99 warnings, GCC and an 1800-second outer build limit. My profiles are normal O1, ASan/UBSan O1 with leak detection and immediate refusal, and production O3 with vectorization.

| Profile | Fresh build | Focused targets | Result |
| --- | ---: | ---: | --- |
| normal | 78.088s | 24.041s | PASS |
| san | 198.123s | 28.575s | PASS |
| production | 130.619s | 26.132s | PASS |

My original focused targets cover GC shutdown and allocation refusal, 60 environment checks, ten lexical methods, the source graph API and twenty source-C methods, the new checker ownership API, module-rejection rollback, and 34 callable/state/session REPL methods. They preserve their original assertions and per-command limits. The additional focused Make invocation has a 600-second outer bound. I do not include the separate seven-method default-ownership admission cohort in this result.

I preserve the earlier `7c0b9b27621918587eb96320d8241ba73cae8ea4` complete normal and sanitizer builds. The original sanitizer source-C cohort fails on an 11-byte declared-module context and a 31-byte overwritten match-binding name. The first cohort cleans its automatic temporary cases; I retain its exact logs and templates, and separately reproduce both failures in new explicitly retained case directories. I do not claim those new directories are the original deleted temporary files.

My API witness also reproduces an invalid free of a caller-owned module cursor and both expression/statement match-name leaks against the earlier sanitizer providers. The correction borrows stable environment-owned module names, retires only loader-owned temporary stems, restores prior borrowed REPL context before rollback, and frees an inherited owned nominal spelling before replacement. I preserve caller, AST and value ownership and the public header layout. An incremental relink passes the three API controls and twenty original source-C methods; I distinguish that localization from the subsequent complete fresh builds.

My archive retains all five completed build closures, their generated objects, dependency files, static archives, executable products, actual build executions, original Make output, source cases retained by the test facility, failure/localization products and tool identities. My independent reader verifies 221 compiler translation units and 169 final compiler object/archive inputs per profile. Long shell recipe strings are truncated by the original 8192-byte strace setting; the actual compiler command arguments are complete and checked separately. I verify 3520 before/after selected source guards on the corrected profiles, all tracked-source diffs, and additional project dependencies against their Git blobs. I retain executed tools and project dependencies; I do not claim a hermetic copy of every operating-system header.

My independent reader passes 14,263 records, 3,520 selected Git inputs and 93 distinct native images. Both failed roots and all three passing roots remain intact.

Run my reader from any working directory:

```bash
python3 docs/evidence/fresh-providers-linux-20261124/read_archive.py
```

My manifest deduplicates unchanged content through the prior committed evidence chain. The reader needs the repository and both named Git commits. All old roots remain intact. I make no full compiler-source, paired Darwin, default ownership, historical timeout cause, performance gain or whole-release acceptance claim. #1234, #1116, #1121, #1229 and the full #976 release contract retain their wider acceptance requirements. I perform no NAS, TCC or migration action.

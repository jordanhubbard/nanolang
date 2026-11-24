# I retain fresh Linux providers and an incomplete source suite

I qualify exact production source `5bf50072752f6e88bad274058663e6e7effc1f89` after removing unused global list generation. I create three clean host-local worktrees and build my C seed plus VM, assembler and C translator with the original flags, tools and 1,800-second build limit.

| Profile | Complete build seconds | Original focused targets plus module-list regression seconds | Private collector cohort |
| --- | ---: | ---: | --- |
| Normal | 72.094 | 22.306 | 41 pass, zero skips, 62 native calls |
| Sanitizer | 133.938 | 31.126 | 41 pass, zero skips, 62 native calls |
| Production | 136.336 | 27.804 | 41 pass, zero skips, 62 native calls |

I retain Linux `detect_leaks=1:halt_on_error=1`, UBSan halt behavior and every original assertion/deadline. All selected source/tool guards match. Private collector images remain qualification-only; I do not enable ordinary ownership by default.

My original complete ordinary compiler-source command exits **1 after 128.806 seconds**, below its unchanged 600-second outer limit. It records 1,071 shadow starts and 1,070 completions, with zero assertion failures among completed shadows. The last started case is `nisa_emit_flow`. The supervisor retains signal 14, `parent_timeout=true` and its original ten-second deadline, then refuses output publication. No seed exists. Private full source is **NOT RUN**. This is a failed qualification under #1229/#1247, distinct from the older 1,306-completion terminal and Darwin's pre-shadow timeout. I infer no historical cause or performance gain.

I preserve the initial missing-observer setup failure, its zero native executions and the observer recovered from the independently verified Darwin archive. The corrected follow-up executes the original private tests and source command without changing their selections or limits.

Run `python3 read_archive.py`. I verify 18,103 file records, every compressed/raw chunk, parent bindings, actual selected Git source identity, all three native provider closures, tools, original Make commands, private inputs/calls and the complete failed source/supervisor streams. My manifest SHA256 is `edbdb38eab6c9679f36a3c04ad7e4af38768f8972f1882c28e2ebe96204fe6f6`. Reused chunks refer to earlier receipts. This is not a hermetic archive of every operating-system header.

I preserve all old roots and keep source/platform/default/raw/bootstrap and the complete 5.1 release open. No unchanged full retry or deadline increase follows this failure.

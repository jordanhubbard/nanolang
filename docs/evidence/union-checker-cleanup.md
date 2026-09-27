# My focused union checker cleanup

I qualified the corrected source at `46b2faad1` with the original two focused
methods. Linux GCC ordinary/GCC sanitizer/Clang sanitizer pass in
15.254/47.028/31.738 seconds. Darwin Apple ordinary/Homebrew sanitizer pass in
8.482/25.949 seconds. Leak detection remains enabled in sanitizer configurations.
My provider preparation is fresh on each host. These controls rebuild the
owning fixture translation units; remaining selected providers are ordinary.

I preserve three earlier terminals:85a reports1,786 bytes/136 allocations on
both sanitizer hosts;09e Linux reports120 bytes/1 allocation;45c reaches the
next allocation executable and reports8,532 bytes/711 allocations on both
hosts. Each corrected source was independently reviewed before execution.
Puck09e was held after the Linux failure and was never executed.

My audits rehash the retained immutable artifacts and check successful process
reaping, absent child groups and unchanged selected inputs at phase endpoints.
They are [85a](union-checker-cleanup/tuple-85a-focused-terminal-audit.json),
[09e](union-checker-cleanup/tuple-09e-focused-terminal-audit.json),
[45c](union-checker-cleanup/tuple-45c-focused-terminal-audit.json), and
[46b](union-checker-cleanup/tuple-46b-focused-terminal-audit.json).
The audit files point to persistent local qualification roots, including
locally copied Puck reports. Raw logs, products and content-addressed objects
remain there; this is a focused status checkpoint, not a complete evidence
archive or independent final seal.

I have started fresh complete build/bootstrap preparation on both hosts at
46b. Its terminals, original18+8 source/native corpus, imported graph deadline,
whole Make and canonical integration remain required and unclaimed here.

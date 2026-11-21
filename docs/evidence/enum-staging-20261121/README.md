# My repeated enum-registration ownership control

Under #1165/#1149 I retain a bounded driver using my original enum-value test.
I extend that test to execute its useful shadows three times and preserve the
canonical registered declaration, all four variants and their values. My fresh
GCC13 ASan/UBSan providers report432 leaked bytes in32 allocations after the
control completes. Checking already registered the enum, but my evaluator
allocates another row before env_define_enum rejects that duplicate. Each
repeated shadow pass repeats the staging allocations.

I check for the existing enum before constructing that duplicate, as my
registration API already does when deciding whether to keep a row. I preserve
the existing lookup and first declaration. I change no borrowed module-name
ownership or public registration API. The exact same bounded driver passes
with leak detection after recompiling only the changed evaluator provider;
all other instrumented providers and assertions remain the baseline's. I
retain both actual drivers, changed provider images, source snapshots, commands
and complete terminals. The repaired bounded run takes13.201 seconds including
rebuild/link; the baseline fresh-provider build/control takes45.210 seconds.

My complete ordinary evaluator passes all134 named methods. The bounded enum
control and original environment-scoping gate pass, including ten Python scope
methods. I derive and execute a separately retained complete driver with the
same source/providers/flags after the original successful Make recipe removes
its executable. I retain its matching rebuilt native FFI fixture and another
complete134-method passing terminal.

I also rerun the complete original sanitizer driver with its matching rebuilt
native fixture. It still fails with13096 leaked bytes in364 allocations; the
last visible original case remains the terminal-match refusal control. No
remaining stack comes from the enum-registration allocations. I do not claim
all134 methods complete under sanitizers. I do not compare this full total as
a matched ownership delta: the enum method now executes three additional
shadow passes. Other callable, AST, record/array/binding owners remain open
under #1149/#1140/#1116. I suppress no leak and change no refusal assertion.

My manifest retains65 logical inputs and142433321 raw bytes with checked raw
and compressed SHA-256 identities. This closes only the bounded enum-staging
repair; complete ownership and release acceptance remain separate.

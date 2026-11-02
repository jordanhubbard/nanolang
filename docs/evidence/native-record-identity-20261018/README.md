# My native record identity prerequisite

I give already-admitted ordinary record/flat-array modules a canonical traced record owner. Copies and snapshots retain its field-vector identity. Other native profiles retain their existing representation. I continue to refuse AGG_SET; the direct generated-C alias probe qualifies representation, not source or bytecode field mutation.

My final Darwin public gate passes in 30.737 seconds: ordinary and instrumented public VM/transport checks, all 4,720 execution allocation prefixes, and 24 sanitized native O0/O2 executions. The native controls include actual early collection before the first record store, a branch-unassigned record local, canonical snapshots without extra owners, constructor failure and recovery, retained old children across replacement, nested record roots, and collection to zero owners before shutdown.

Review found that statically typed record locals enter the root map before their first assignment. An actual helper-call collection boundary reproduced an abort. I now skip only a fully empty zero-initialized wrapper; nonempty wrappers without identity still fail. I retain the original failure and corrected full gate. Independent review found the corrected guard covers every current wrapper member.

My neighboring native suites passed 21 methods in the retained combined run; seven global-type methods could not run because that worktree lacked nanoc_c. After building that prerequisite, all seven passed in the separate retained final run. I do not report the earlier run as wholly green. My strict gate passes 222 Clang and 222 GCC compiles.

I retain source hashes with this receipt. This is Darwin development evidence; fresh combined compiler stages and Linux/Darwin hosted qualification remain required. Source authority publication, generic schemas, AGG_SET admission and the broader #1001 acceptance remain open.

# My checked empty-record allocation evidence

I qualified task_66983a5c5f9a4c869fc726824cf6a773 at source/test pin
`51a6f326656bfc07445e5ec01d28c23792055b83` on Linux aarch64. Production is the one-line NULL guard reviewed
at d2f1ad6a: I report VM_ERR_MEMORY before publishing a failed STRUCT_NEW result.
I preserve definition identity, empty-record behavior and existing profiles.

I passed full `make -j6 test-nanovm`: 274541 checks, zero failures, followed by
heap and stack allocation recovery. Clang ASan/UBSan with leak detection passed
`test-vm-heap-allocation-sanitizers` with both computed-goto and switch dispatch.
I used `--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13` and
`-DNANO_NO_COMPUTED_GOTO` for the switch run. These are Linux gates, not Darwin
or managed-record execution acceptance.

My fresh verified module returns record definition1 with zero fields. After the
guard exists, a bounded header allocation refusal leaves no returned record,
clean invocation frames/stack and unchanged heap counters. A later invocation
succeeds even when the legal zero-sized field allocation returns NULL. Releasing
the result restores object count and live bytes; another ordinary invocation
also succeeds. I do not execute a pre-fix fault case.

I preserve my initial fixture failures: the first full VM suite passed before
its new fixture failed to compile because I misspelled the cycle-drain API;
the corrected fixture then failed its successful teardown assertion because
I compared cumulative allocated bytes with the initial value. Static inspection
of heap allocation/release accounting established that allocated and freed are
separate cumulative counters. I retained exact object-count checks and changed
the success check to allocated-minus-freed. The failure path still checks
unchanged raw counters. No production change followed the independent review.

My [artifact manifest](vm-struct-new-allocation-artifacts.json) hashes all retained
logs and final source. Private record storage/traversal2dc, ordinary authority15f,
managed aggregate488, managed51da and release acceptance remain open.

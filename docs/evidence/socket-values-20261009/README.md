# I qualify Socket value lifetimes before source admission

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I extend
`576f4975a` with the [Socket value contract](../../NSI_SOCKET_VALUES.md).
My private ConnectResult and connection carriers retain one adapter owner across
moves, exclusive borrows, host Result errors, drop and terminal cleanup. They do
not admit source calls, equate private Socket with public Conn, or complete the
required compiler/VM/native/WebSocket integration.

My first sanitizer Make batch passes without correction: Socket values in both
linkage modes, existing TCP/local Socket tests, File values, File adapter and
capability tests. I retain the complete log in
`initial-sanitizers-and-adjacency.log`. File values pass1,989 instrumented and1,289
linked checks; File adapter passes2,634 checks with395 real opens/closes; all
seven capability tests pass. The Socket adapter passes9,878 checks with514 tracked
real opens/closes. I use LLVM ASan/UBSan, `ASAN_OPTIONS=detect_leaks=1`, strict
warnings and no sanitizer recovery. The capability Make target deletes its test
executable; I retain hashes for the other eight produced executables.

I then extend only the new value test with host send/receive error canonicalization,
rights refusal and an unaccepted-close control. My final12 compile/execution
steps pass in4.599 seconds on macOS26.7.1 arm64:

| Compiler/configuration | Included-source checks | Separately linked checks |
| --- | ---: | ---: |
| LLVM ASan/UBSan/leak detection | 3,630 | 261 |
| Apple Clang21, strict O2 | 3,630 | 260 |
| GCC16.2, strict O2 | 3,629 | 261 |

The small count variation comes from checked nonblocking polls before readiness.
Each instrumented run accounts for81 tracked real opens and81 closes. One
deliberately unclosed fault descriptor is recovered by the harness, never claimed
as successful adapter cleanup. Listener and accepted peer fixtures close their
own descriptors outside these adapter counters.

`runs.json` retains exact commands, return codes, timings and six binary hashes.
I retain actual compiler versions/hashes,19 input hashes, all logs and the runner;
the runner verifies unchanged inputs at the end. Its absolute paths describe this
host's invocation. Existing `test-units` now includes `test-nsi-socket-values`;
I do not claim a fresh Linux run from this Darwin evidence.

My real cases exercise both IP families through Result extraction, move,
exclusive borrow, completion, NUL send,255 receive, EOF, close, unhandled Ok drop
and terminal cleanup with a live borrow. Other controls cover both Result arms,
copy-stale and cross-invocation refusal, reused borrow epochs, source/output
overlap, all three allocation prefixes, identity/generation/epoch exhaustion,
full value/adapter capacity, pending and failed moves, canonical host error
Results, exact rights, invalid terminal status recovery and unknown-close reports.
I retain the first execution status across repeated finish calls; no repeated
finish performs host cleanup.

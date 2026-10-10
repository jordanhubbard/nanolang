# I qualify private TCP ownership on Darwin

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I extend
`77880c27a` with numeric IPv4/IPv6 acquisition and explicit nonblocking completion.
My [contract](../../NSI_TCP_SOCKET_SERVICE.md) keeps private Socket and public Conn
distinct. Public source/VM/native bindings, DNS policy, WebSocket and exact
Linux/Darwin candidate acceptance remain open.

I pass all13 steps in `runs.json` on macOS26.7.1 arm64, totaling10.658 seconds:
included-source and separately linked compile/execution with Apple Clang21,
GCC16.2 and Homebrew LLVM under ASan/UBSan with leak detection, followed by File
and capability sanitizer adjacency. Each instrumented Socket run passes9,878
checks with514 tracked real opens/514 closes. Two deliberately unclosed fault
descriptors require explicit harness recovery. Listener/accepted peer fixtures
are separately closed and are not included in the adapter descriptor counters.
File passes2,634 checks,395 opens/closes and no live resources; its linked control
and all seven capability tests also pass.

I retain full logs, command arguments, durations, binary hashes, actual tool
versions/hashes and14 source hashes. `qualify.py` records the run and checks source
equality afterward. Its absolute paths identify this host's invocation, not a
relocatable release artifact. Existing CI's sanitizer `test-units` dependency
includes these Socket tests; this local receipt does not claim a Linux result.

My real TCP cases cover IPv4/IPv6 connection, transfer, bidirectional NUL/255,
empty read, half-close EOF, connection refusal and cleanup through both linkage
modes. Fault controls cover every descriptor/configuration step, synchronous
connect errors, zero-time polling, interrupted observation, malformed SO_ERROR,
sticky host errors, immediate/pending publication, rights/context/identity,
capacity/generation exhaustion, transfer rollback and ambiguous close. Injected
readiness is a state-machine control, not real traffic evidence. Existing local
pair and SIGPIPE tests remain intact.

## I retain the first refusal-fixture failure

`first-refusal-timeout.log` fails at the bounded connection wait. Diagnostic
instrumentation in `refusal-diagnostic.log` shows the real IPv4 connection became
writable; the second connect, to a reserved but non-listening socket, remained
pending. An isolated Python `socket`/`select.poll` check reproduced this:

```text
bound but not listening ('127.0.0.1', 64021) 36
poll [] error 0
after close poll [(4, 16)] error 61
```

Darwin kept the attempt pending while the non-listener stayed bound. I corrected
the fixture to close that socket immediately after initiating the connection,
then require ECONNREFUSED within the original bounded wait. I removed temporary
diagnostic prints. I changed neither production code nor the refusal assertions
to obtain the corrected pass. I subsequently added immediate-success and full
TCP-table transfer controls before the final13-step qualification.

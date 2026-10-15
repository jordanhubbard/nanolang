# I qualify the TCP catalog and Endpoint bridge

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I extend
`0f1e04f3c` with the [exact TCP catalog](../../NSI_TCP_BINDING_CATALOG.md), shared
descriptive catalog shapes/equality, and checked numeric Endpoint conversion into
the actual TCP value adapter. The File catalog retains its original methods,
types, constants and behavior. Conn is the exact public nominal resource;
private Socket substitution and legacy string-connect documents are refused.

My first focused Make sanitizer batch passes without a production correction:
Socket plan, File plan and Socket values, each included-source and separately
linked. I then add an assertion that invalid Endpoint tests make zero socket
acquisition calls. The final matrix covers the resulting sources; I retain the
initial log as `first-sanitizer-batch.log`.

All36 final compile/execution steps pass in16.630 seconds on macOS26.7.1 arm64:

| Subject | Included-source checks | Separately linked checks |
| --- | ---: | ---: |
| TCP catalog, each compiler | 1,647 | 1,136 |
| File catalog, each compiler | 1,319 | 905 |
| Endpoint/Socket values, LLVM sanitizers | 3,823 | 320 |
| Endpoint/Socket values, Apple Clang21 O2 | 3,822 | 318 |
| Endpoint/Socket values, GCC16.2 O2 | 3,823 | 320 |

Socket counts vary with checked nonblocking readiness observations. Every
instrumented value run accounts for81 tracked real opens/closes, with one
explicit harness recovery for the intentionally unclosed fault. Real IPv4/IPv6
connection, NUL/byte traffic, EOF and cleanup now enter through
`nl_socket_values_begin_connect` and its numeric Endpoint conversion.

I check exact method/type/member/order/ownership/lifetime identities, altered
declarations, cross-catalog refusal, allocation failure without output mutation
and query lifetime after the NSI document is freed. Endpoint controls include
word byte order, all unsigned32-bit boundaries, family/port domains, IPv4 unused
words/scope, overlap refusal and canonical Error publication without host access.

My File binding compatibility suite passes both methods in14.239 seconds under
LLVM ASan/UBSan/leak detection. Both linkage modes traverse704 corpus cases and
reproduce the retained interface and generated binding bytes. The instrumented
mode passes26,881,859 checks including allocation failures; the linked mode
passes7,855. The NSI and generator neighbors pass20 and3 tests respectively,
and their File descriptor control passes905 checks. `file-binding/` retains
commands, statuses, stdout/stderr, expected/actual bytes and a manifest of the
original813 artifacts. Large generated corpus inputs and binaries remain in
the recorded temporary directory; they are hashed rather than copied here.

`runs.json` retains exact matrix commands, durations and18 executable hashes.
I retain actual compiler versions/hashes and27 unchanged source/fixture hashes.
`qualify.py` checks input equality after its final step. Absolute paths describe
this host's invocation. The Make unit suite includes the new TCP catalog gate.

These are descriptive catalog and host-value results, not source or execution
admission. Paired C/Nano resolution/lowering, nominal wire metadata, selected
shadows, verified VM/native dispatch, DNS/public string connect, WebSocket and
exact release-candidate Linux/Darwin gates remain required. I do not infer a
Linux result or full compiler fixed point from this local matrix.

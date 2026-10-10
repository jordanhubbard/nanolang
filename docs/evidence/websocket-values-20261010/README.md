# My WebSocket invocation values

I add private invocation-owned Connection and ConnectResult values after
`684e1ee81` under #990. My copied host policy, 64-slot ownership table,
move generations, exclusive borrow epochs and consuming cleanup prepare the
transport for checked runtime integration. I retain caller-owned message bytes
across value/context destruction.

My injected-host probe checks allocation refusal, copied resolver policy,
cross-invocation and stale token refusal, ConnectResult arm extraction, moves,
exclusive borrow conflicts, stale borrow epochs, signed deadline bounds,
slot/generation/epoch/identity exhaustion, denied network access, connection
failure, cleanup failure reports, cached finish, invalid finish status and
cleanup of borrowed/unhandled values. It runs real value code against a small
counted fake transport so I can assert host call counts and outstanding owners.

My four real-peer methods cover counted binary traffic through borrows, moves,
stale token rejection, successful close, consuming close with a negative timeout,
and invocation destruction with a live borrow or an unhandled ConnectResult.
Message bytes survive the destroyed context. The hostname case uses my installed
resolver helper selected by the peer harness.

| Gate | Result |
| --- | --- |
| `make -f Makefile.gnu test-websocket-values CC=/opt/homebrew/opt/llvm/bin/clang` | injected-host probe and 4 peer methods pass (`clang.log`) |
| Same target with `CC=/opt/homebrew/bin/gcc-16 OBJ_DIR=/private/tmp/nl51-websocket-values-gcc-obj` | injected-host probe and 4 peer methods pass (`gcc.log`) |
| LLVM ASan/UBSan with leak detection, real peers | 4 methods pass (`sanitized.log`) |
| Final injected-host probe, LLVM ASan/UBSan with leak detection | exit 0, no diagnostics (`unit-sanitized.log`) |
| Final injected-host probe, strict GCC | exit 0, no diagnostics (`unit-gcc.log`) |

The final injected probe adds a failed-connect cleanup-report assertion after
the Make runs. I compile it with `-D_GNU_SOURCE -std=c11 -Wall -Wextra -Werror`;
my LLVM run also uses `-fsanitize=address,undefined -fno-omit-frame-pointer`
and executes with `ASAN_OPTIONS=detect_leaks=1`. My real-peer sanitizer command
is `python3 -m unittest -v tests.test_websocket_values` with
`NANO_WEBSOCKET_CC=/opt/homebrew/opt/llvm/bin/clang`, the same sanitizer flags in
`NANO_WEBSOCKET_CFLAGS`, and leak detection enabled.

I have tested this private carrier on Darwin. It assumes serialized native
callers; token validation is not isolation against arbitrary C memory access.
I have not added public WebSocket runtime dispatch or compiler admission.
String-bearing records/results, runtime storage accounting and trusted policy
integration, verified dispatch, paired frontend lowering and final Linux/Darwin
release qualification remain required.

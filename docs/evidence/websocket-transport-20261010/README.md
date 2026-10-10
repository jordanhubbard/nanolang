# My shared counted WebSocket transport

I extract the host transport from my legacy wrapper so my owned service can use
counted text/binary messages without routing them through borrowed C strings.
This checkpoint follows `2a89a5bfa` under #990.

I test both the linked implementation and an included-source allocation-failure
probe against real local peers. Each method runs numeric and hostname connections.
My six methods cover embedded-zero text, binary and empty messages, storage retained
across later receives and close, retry after failed message-copy allocation,
partial input across timeout, protocol refusal, close on invalid/expired deadlines,
and terminal partial-send failure. Each probe also checks network/DNS denial,
zero connect time, timeout caps, counted URL refusal and preserved failed outputs.

| Gate | Result | Retained log |
| --- | --- | --- |
| Direct LLVM Clang | 6 methods pass | `direct.log` |
| Direct GCC 16 | 6 methods pass | `gcc.log` |
| LLVM ASan/UBSan, leak detection enabled | 6 direct + 7 legacy methods pass | `sanitized.log` |
| Legacy LLVM Make target | 7 methods pass | `legacy.log` |
| New direct Make target | 6 methods pass | `make.log` |
| Cold-cache module packaging and dependency shadows | 1 method passes | `bindings.log` |
| C-seed and updated-Nano VM/native artifacts, ABI refusal | 4 methods pass | `artifacts.log` |

I run `python3 -m unittest -v tests.test_websocket_transport` with
`NANO_WEBSOCKET_CC` selecting `/opt/homebrew/opt/llvm/bin/clang` or
`/opt/homebrew/bin/gcc-16`. My sanitizer run adds
`NANO_WEBSOCKET_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'`
and `ASAN_OPTIONS=detect_leaks=1`, and includes `tests.test_websocket_client`.
My Make gate is `make -f Makefile.gnu test-websocket-transport
CC=/opt/homebrew/opt/llvm/bin/clang`.

My artifact run uses `tests.test_websocket_artifacts.WebSocketArtifacts`,
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang` and
`NANO_WEBSOCKET_DRIVER='/Users/jordanh/Src/nanolang/bin/nano_vm
/private/tmp/nl51-string-compiler-final.nvm --'` (a single-line command value).
The compiler artifact and its retained host cache are from my checked-string
qualification. This run rebuilds the changed WebSocket module. The artifact log
names its retained command/output directory. `inputs.sha256` pins changed inputs.

These are Darwin tests of the shared C transport and legacy source module.
My four-method owned catalog still lacks affine runtime values, string-bearing
records/results, verified dispatch and paired source admission. I do not claim
that these tests establish public owned service execution or release readiness.

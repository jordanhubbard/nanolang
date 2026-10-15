# My paired WebSocket source lowering

I build on `462ca0cb7` under #990. My C and Nano lowerers now emit the exact
seven-type, four-method WebSocket catalog. I preserve string-bearing Message
records, source-order field evaluation, typed temporary slots, URL/message/deadline
operands, borrowed helper calls, consuming close and method indices. My C path
selects the explicit WebSocket codec and derives stack bounds from its checked
flow facts. My Nano writer independently constructs the same wire format.

My first source run fails serialization: the shared private-wire validator
requires the File catalog's eight layouts, while WebSocket has seven. I pass
the minimum from the trusted codec, retaining the ordinary validator's existing
minimum and exact nominal checks. The next run executes the C product but finds
one differing ownership byte: my Nano writer emits an eighth flag into padding.
I derive flags from the catalog type count and require zero padding in a shadow.
I also retain and fix my initial incorrect conditional syntax in that shadow's
implementation.

My WebSocket test compares exact bytes from the C lowerer and the current Nano
lowerer compiled into VM and LLVM ASan/UBSan native products. Direct and indirect
Message helpers, embedded NUL strings and selected shadows agree. All four
source selections execute through the public VM and generated-native providers.
The grant denies connections, so the main returns the expected rights error;
the Message shadow returns zero. This passes in 60.026 seconds. My source includes
send/receive loops and consuming close, but denial prevents their runtime
execution. I do not count this as a successful network lifecycle.

My final two-method run passes in 58.970 seconds, adding reversed-field TCP
Endpoint construction and exact C/Nano wire agreement without network traffic.

My File wire and source corpus passes all 15 methods in 124.889 seconds, including
reference maps, aliasing, cleanup, allocation refusal, shadows and native products.
WebSocket hosted preparation passes 504 linked and 4,554 instrumented checks,
including 61 allocation-failure prefixes per mapping. The ordinary WebSocket
boundary and File/TCP/multi-catalog nominal regressions also pass.

I add `make -f Makefile.gnu test-websocket-source-lowering`. I run with LLVM
selected through `CC`/`NANO_NATIVE_TEST_CC` and `NMS_RUNTIME_CLANG` plus
`NMS_RUNTIME_OPT` for managed modules. My runtime link uses pkg-config's crypto
flags rather than a machine-specific library path.

Public CLI product dispatch still refuses WebSocket. Explicit CLI policy,
shadow dispatch and installed product selection, mixed-service WebSocket support,
live-network source execution on a host permitting local listeners, DNS integration
and full release/platform qualification remain open. This checkpoint does not
establish a fresh compiler fixed point or release readiness.

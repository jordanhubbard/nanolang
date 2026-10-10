# TCP runtime carrier

I retain this batch against parent `4b79edbfb`; `source-sha256.txt` identifies
my changed implementation and test sources. I instantiate the same private
frame, ownership, borrow, fuel and cleanup machinery for separate File and TCP
opaque APIs. My File value adapter preserves its previous payloads and ABI.
My TCP adapter consumes Endpoint, preserves eleven SocketError fields and maps
the Socket core's SEND/CONNECT ordinal order explicitly.

I run 2,004 allocation-instrumented and 1,849 linked TCP carrier assertions with
LLVM ASan/UBSan, then the same corpus with Apple Clang and GCC 16 without
sanitizers. I test both VM/native arena layouts, real IPv4/IPv6 loopback connect,
send/receive and EOF, invalid byte Results retaining Conn, shared/formal borrow
release, consuming close and abandonment with a live borrow. The invalid
Endpoint control observes zero adapter socket acquisition attempts.

I inject a close error after the harness really closes the descriptor. The
carrier retains the unknown-closure fields, does not retry close and withholds
successful final scalar output. This checks reporting and ownership behavior;
it does not claim to reproduce every kernel close-failure mode.

I manually drive the invalid-Endpoint Error path through cyclic and indirect
frame variants and check exact charged instruction counts. Zero-fuel cases
refuse before effects. The TCP fixture does not yet exercise actual indirect
callee execution or a positive connection through a matched CODE dispatcher.
The new carrier allocation-prefix walk instruments carrier arena allocations;
it does not instrument every linked query/core provider.

My unchanged File carrier, frame, cyclic and indirect suites pass with ordinary
flags; carrier/frame suites also pass under LLVM ASan/UBSan. I retain separate
File cyclic/indirect dispatcher results and their generated-native commands.
These are File compatibility evidence, not TCP dispatch acceptance.

I retain the original fixture helper-name compilation collision and the failed
File sanitizer invocation that omitted the SDK `ffi.h` include path. I renamed
the test helper, used the common service descriptor type in shared code and
supplied LLVM plus the SDK include path to the sanitizer runner. I did not
suppress warnings or remove assertions. The normalized shared-engine diff
records the bounded Endpoint path changes and extraction of value adapters.

I retain full commands, output and statuses in compressed logs and command
artifacts. My public source lowerers still refuse TCP publication. Matched TCP
dispatch, native emission, grants, paired source lowering, network shadows,
fresh bootstrap and the full 5.1 platform/release contract remain open.

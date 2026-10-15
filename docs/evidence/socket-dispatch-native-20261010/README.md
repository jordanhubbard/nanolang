# Matched TCP VM and generated-native execution

I retain this batch against parent `468779c39`. `source-sha256.txt` identifies my
changed sources. I share the existing indirect-capable File dispatcher and C
emitter with an explicitly selected TCP profile. I add Endpoint consumption to
both handlers and its domain-check obligation to the independent coverage
inventory. The shared runtime performs the check before acquiring a socket.

I use one fuel-bounded TCP profile for ordinary bodies, loops and indirect calls.
My explicit private providers are `socket_vm_indirect_private.c` and
`nvm2c_socket_indirect_private.c`, enabled by their corresponding
`NVM_SOCKET_INDIRECT_VM_PRIVATE` and `NVM_SOCKET_INDIRECT_NATIVE_PRIVATE` macros.
They do not authorize the ordinary VM, CLI or public source compiler.

I run `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-socket-dispatch`
with LLVM on PATH. The final run passes in 31.442 seconds under ASan/UBSan. I
also run the same target with GCC 16 and `NANO_SOCKET_DISPATCH_SANITIZERS=0`;
that ordinary-compiler run passes in 31.538 seconds. Each retains 86 successful
subprocess commands, output logs and exit statuses.

My four bytecode programs cover direct IPv4 calls, owned and borrowed indirect
IPv6 calls with permuted nominal identities, invalid Endpoint and an assertion
failure with a borrowed live connection. Real listeners receive byte165 and
reply with byte90. The programs check send counts, receive values and EOF flags.
I compile generated C separately at O0 and O2, without the TCP VM or emitter
providers, and check executable symbols for their absence.

For normal execution I compare scalar result, status, cleanup and real
acquisition/close counts. Polling readiness can change loop counts; I do not
claim identical instruction counts across independent network runs. Explicit
fuel0 and fuel25 cases compare exact charged counts and unchanged output on
refusal. Injected close errors occur after the harness really closes the socket;
both paths retain unknown closure, close once and withhold successful output.

I alter native ABI revision, checked callee sets and Endpoint pending-check
facts. Every control refuses before runtime acquisition or socket creation.
Truncated input refuses VM execution and emission while preserving prior
outputs. `tcp-command-artifacts.tar.gz` retains all generated C, serialized
fixtures, compiler commands, symbol checks and execution reports. Its encoded
ports belong to that run; the checked-in runner creates fresh listeners.

My File indirect dispatcher and public-grant suites pass in 99.988 and117.637
seconds with ordinary compiler flags. I compare the previous batch's generated
File sources to this batch: all78 linked and78 instrumented programs are
byte-identical. The identity log retains their hashes. The normalized shared
review diff shows only profile selection, Endpoint handling and refusal of the
unimplemented TCP public-emission surface.

These are local Darwin checks. I have not connected TCP public grants, paired
source lowering or selected network shadows, performed a fresh bootstrap, or
qualified the full 5.1 platform/release contract. DNS/WebSocket and other open
release requirements remain required under #990 and the release umbrella.

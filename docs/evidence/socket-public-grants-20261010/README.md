# Explicit TCP public grants and installed runtime

I expose checked TCP bytecode execution and generated C through an opaque
outbound-connection grant. My grant retains ABI 1, catalog 2 and a separate
runtime identity. Creation allocates policy without opening a socket. Revocation
is idempotent; destruction clears caller storage. My grant covers outbound
IPv4/IPv6 TCP, not listeners or arbitrary foreign calls.

I use the File public-call gate for both services. Concurrent or reentrant
public calls return BUSY before inspecting their arguments. My tests exercise
poison pointers while that gate is held, including a second thread and reentry
from socket acquisition. I publish a scalar only after runtime teardown reports
successful execution and cleanup. Missing and revoked grants preserve output.

I build and install the C99 package with:

```sh
make -f Makefile.gnu socket-public-runtime
make -f Makefile.gnu PREFIX=/desired/prefix install-socket-public-runtime
```

I install `lib/libnano_socket_runtime.a` and headers beneath
`include/nanolang/socket`. My execution entry is
`nvm_socket_execute_indirect_bytes`; my nonexecuting emitter is
`nvm2c_emit_socket_indirect_bytes`. Generated entries accept the same grant,
options and scalar output. My emitter requires a 1–63 character ASCII identifier
beginning with a letter, followed by letters, digits or underscores. Callers
free its successful C text with `free`.

I retain commands, results and source hashes alongside this report. My final
public suite passes in 41.211 seconds and retains 145 subprocess commands.
I run `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-socket-public`
with LLVM on PATH. The network corpus uses ASan/UBSan; installed archive
consumers use the ordinary C99 package. My public
network corpus covers real IPv4/IPv6, direct and indirect ownership transfers,
borrowed calls, explicit fuel exhaustion, invalid Endpoint, assertion failure,
unknown closure and native ABI/plan tampering. Generated C runs at O0 and O2.
Installed consumers build outside my checkout with only installed headers and
the archive: two generated programs and one public VM/emitter consumer. Native
symbol checks exclude VM/emitter providers from the generated executables.

My first wrapper build called nonexistent socket-prefixed internal helpers.
I retain that diagnostic and use the existing shared engine helper names.

These are local Darwin checks. My grant API follows ordinary C pointer lifetime
rules; it is not a hostile-memory sandbox. This batch does not complete paired
source lowering, CLI policy, mixed File/TCP execution, DNS/WebSocket, a fresh
bootstrap or the full 5.1 platform and publication contract. I keep #990 open.

My private TCP regression passes in 29.869 seconds after the shared test refactor.
My File public indirect sanitizer and installed-package tests both pass;
`regressions.log` retains the terminal result and trace counts.

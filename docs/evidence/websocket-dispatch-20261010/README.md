# My matched private WebSocket execution

I execute the same checked WebSocket wire through my VM adapter and separately
compiled generated C. Both consume ordered arguments, including every timeout,
use exclusive references for send/receive and consume the Connection on close.
I match pending masks to catalog parameter domains. Explicit copied host policy
is required before either engine starts the runtime.

I retain three complete bytecode programs: direct string helper calls, permuted
imports/layouts with indirect string calls and hostname resolution, and consuming
close with an invalid deadline. Each runs a two-iteration send/receive loop,
constructs binary Message records containing `a\0b`, checks the received binary
flag, counted length and string equality, and branches on each Result arm.
Successful normal executions return 77. Generated native code has its own
functions and branches; symbol checks find no private VM executor, emitter or
generic VM execution entrypoints in the standalone native products.

For each program, both native optimization levels (`-O0`, `-O2`) agree with VM
reports on normal, denied-policy, missing-policy, fuel 0/20/40, protocol-error and
close-failure cases. That is 48 VM and 48 native executions per toolchain. I
compare status, result, preservation, instruction count, fuel, acquisition and
cleanup. Every observed socket acquisition has a matching close attempt. These
are tested cases, not a proof for all programs or host failures.

I also alter generated calls to substitute a different valid import and swap
argument order. Both refuse before socket acquisition and preserve prior output.
Removing the timeout-obligation bit from generated agreement refuses before
runtime acquisition. Truncated wire refuses in both VM and emitter; failed
emission preserves the previous output pointer.

Commands:

- `make -f Makefile.gnu test-websocket-dispatch CC=/opt/homebrew/opt/llvm/bin/clang` passes with LLVM ASan/UBSan/LSan enabled by the harness.
- `NANO_WEBSOCKET_SANITIZERS=0 make -f Makefile.gnu test-websocket-dispatch CC=/opt/homebrew/bin/gcc-16` passes with GCC.
- `make -f Makefile.gnu test-socket-dispatch test-services-dispatch CC=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_CLANG=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_OPT=/opt/homebrew/opt/llvm/bin/opt` passes TCP and mixed File/TCP VM/native regressions.
- `make -f Makefile.gnu test-file-indirect-dispatch CC=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_CLANG=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_OPT=/opt/homebrew/opt/llvm/bin/opt` passes both linked and instrumented File methods.

I retain every command/status/output from the two full WebSocket runs as JSON,
plus source and artifact hashes. I subsequently strengthen the generated mask
control to clear specifically bit 1024 and replay just the three altered-native
controls against both retained provider closures. Those commands/results are
separate. The first replay script confused `/var` and `/private/var` spellings
while locating a command argument, before running any compilation; I preserve
that failure and resolve the argument by its filename. Product tests did not fail.

This evidence is Darwin-only and uses explicit private providers. Ordinary
catalog-3 admission remains refused. Public host grants, paired source lowering,
DNS service integration and exact release/platform gates remain required under
#990; this checkpoint does not establish release 5.1 completion.

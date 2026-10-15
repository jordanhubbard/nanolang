# WebSocket artifact ABI and capability entropy

I replace failed capability entropy with explicit refusal instead of predictable
bytes. I also refuse repeated zero secrets, low Forth cells and duplicate live
Forth cells after bounded sampling. My injected boundary writes bytes before
reporting failure, so my refusal checks the return status. Failed mint,
attenuation and transfer preserve output and prior authority. File acquisition
closes its acquired stream on failure; partial Socket mint rolls back the first
capability before any host socket acquisition. File/Socket report an I/O error.

I admit seven exact legacy WebSocket artifact signatures in my Nano emitter and
C AOT translator. My existing artifact selection remains authoritative. I copy
borrowed string results before subsequent foreign calls can overwrite them.
Real local peers exercise connect, status, send, both receive methods, errors
and close, and retain both received strings after subsequent calls and closure.
Wrong parameter/result/arity and unknown-symbol declarations preserve prior
output on refusal.

## Evidence

- My capability/File/Socket adjacency and sanitizer logs pass on Darwin. My final
  entropy log adds File rollback and Socket partial-mint/transfer failures under
  LLVM Clang ASan/UBSan/LSan; my GCC log passes the same final entropy fixture.
- My four artifact tests pass using the newly compiled Nano compiler in NanoVM;
  both updated-compiler methods also pass with that compiler translated to C.
  I retain their per-command records in `vm-compiler/` and `native-compiler/`.
- I built that compiler from `src_nano/nanoc_v06.nano` with the C seed, selected
  dependency shadows and `NANO_SHADOW_TIMEOUT_SECONDS=300`. I retain the compiler
  hashes and build logs. This is updated-source execution, not a fresh Stage 1 /
  Stage 2 fixed point or exact-candidate Linux qualification.
- My first broader translator command selected Make's `cc` despite an environment
  `CC` setting. Apple Clang refused leak detection in the array-pop prerequisite;
  I retain that failed log. My corrected command passes `CC` as a Make argument
  and passes 2,438 translator checks plus the adjacent prerequisite suites.
- That command then catches my nonexistent `nanoisa` Make prerequisite. I use
  the actual `bin/nanoisa` target. My strengthened wrong-operand assertion first
  expects the later declaration diagnostic; I correct it to require the earlier
  exact-operand refusal actually emitted. I retain both failures. The final Make
  target passes all four methods; `make-final/` retains its command records.
  The target also forwards effective link flags for instrumented runtime builds.

My source hashes identify the implementation and fixtures. Generated native
artifact test programs enable ASan/UBSan; their separately built module libraries
are not claimed to have been instrumented by that command. The earlier protocol
checkpoint contains dedicated instrumented wrapper/protocol tests.

Public affine WebSocket bindings, checked service string transport, supervised
DNS and exact release-candidate platform gates remain open under #990.

## Commands

```sh
make -f Makefile.gnu test-nvm2c test-websocket-artifacts CC=/opt/homebrew/opt/llvm/bin/clang NANO_WEBSOCKET_DRIVER=/private/tmp/nl51-ws-compiler-native
make -f Makefile.gnu test-websocket-artifacts CC=/opt/homebrew/opt/llvm/bin/clang NANO_WEBSOCKET_DRIVER=/private/tmp/nl51-ws-compiler-native
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 make -f Makefile.gnu test-nsi-cap CC=/opt/homebrew/opt/llvm/bin/clang OBJ_DIR=/private/tmp/nl51-cap-entropy-sanitized CFLAGS='-Wall -Wextra -Werror -std=c11 -g -O1 -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-fsanitize=address,undefined'
make -f Makefile.gnu test-nsi-cap CC=/opt/homebrew/bin/gcc-16 OBJ_DIR=/private/tmp/nl51-cap-entropy-gcc
```

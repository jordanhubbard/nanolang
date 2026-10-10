# Installed resolver and WebSocket DNS integration

I continue #990 after my checked-string and private DNS-supervisor checkpoints.
My ordinary build now includes `bin/nano-resolver`; `install-resolver` installs it
under `PREFIX/bin`, full `install` depends on that target, and `uninstall` removes
it. I do not run the complete bootstrap to test this isolated helper installer.

My selector uses an absolute `NANOLANG_RESOLVER` override, otherwise an absolute
`NANOLANG_ROOT` plus `/bin/nano-resolver`, otherwise the helper beside my actual
running executable. I resolve the selected path and require an executable regular
file. Invalid configured values refuse without falling through to another source.
I do not search PATH or CWD. Failure preserves the destination buffer. The host
must trust the selected executable and environment, and keep the path stable.

My legacy WebSocket wrapper still explicitly grants hostname lookup as part of
its unsafe host API. Numeric addresses bypass helper selection. DNS now uses the
supervisor and the remaining ten-second connect/upgrade budget; missing/failed/
hung workers refuse the connection without synchronous fallback. I add the
supervisor provider to the module's ordinary C-source closure.

## Darwin checks

- Clang installed/relocated selection: one method, 5.538 seconds. GCC selector:
  one method, 4.782 seconds. I run the actual `install-resolver` target into a
  prefix with spaces, move it, resolve localhost from the relocated executable,
  check explicit helper/root selection and invalid overrides, and verify that
  neither PATH nor CWD supplies a missing helper. Short output storage remains
  unchanged on refusal.
- Production WebSocket real-peer suite: seven methods, 11.250 seconds. The LLVM
  ASan/UBSan/leak run passes the same seven methods in 13.540 seconds. A helper
  that sleeps for thirty seconds is refused under the ten-second connection
  deadline. Real hostname, numeric IPv4/IPv6, framing and cleanup cases pass.
- Cold-cache module packaging and dependency shadows pass through C-native and
  bytecode drivers (one method, 3.994 seconds).
- Exact artifact ABI suite: four methods, 4.412 seconds. C-seed and updated Nano
  compiler outputs both execute real numeric and localhost WebSocket exchanges
  through VM and native C. Wrong ABI inputs preserve prior outputs. I use the
  retained `/private/tmp/nl51-string-compiler-final.nvm` compiler, not a fresh
  Stage1/Stage2 bootstrap. I retain its referenced host cache.

My raw logs and compiler commands are beside this file; `hashes.json` identifies
implementation, helper and compiler inputs. This is Darwin evidence, not Linux
or final-release qualification. Full installed-compiler qualification remains
separate from the tested resolver installer.

## Still required

I have not admitted an affine public DNS or WebSocket service catalog. Versioned
string-bearing fields/results and explicit lookup/deadline host policy still
need matched verifier, source, VM/native and installed-runtime integration.
Legacy WebSocket integer registry identities remain compatibility behavior, not
affine source ownership. These results do not close issue #990 or release 5.1.

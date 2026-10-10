# I repair two CI integration failures

Under #982/#989 I inspect run `38016633196` at `796a8cf1d`. Both Linux
build jobs fail the indirect carrier gate; the sanitizer job fails the C
NanoISA tooling sweep. I retain the observed job states, the arm64 carrier
command/diagnostic/status and the original sanitizer failure excerpt.

The carrier fixture puts an unconditional `break` on the same indented line as
a conditional move. GCC rejects this with `-Werror=misleading-indentation`.
I reproduce the diagnostic locally with Homebrew GCC 16.2.0, including through
the new dispatch fixture, then put the unconditional break on its own line.
I change neither control flow nor warning policy.

The C opcode tooling sweep still ends its private File range at
`FILE_END_BORROW`. The next opcode, `FILE_CALL_REFS`, is also a private fragment:
whole-module disassembly must refuse its incomplete authority, while raw
instruction disassembly must retain its mnemonic. I extend the existing test
branch through that opcode. I preserve both assertions and production policy.

After correction, GCC accepts the indirect VM adapter, native emitter, carrier
fixture and dispatch fixture with `-Wall -Wextra -Werror -fsyntax-only`. It also
accepts all 53 retained generated C products from the preceding dispatch batch.
These are local GCC syntax checks, not Linux execution. The complete compiler
command and source list are in `gcc-generated-status.json`.

I run one local LLVM Clang batch with its bin directory on PATH and
`OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5`:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-nanoisa test-file-indirect-runtime test-file-indirect-dispatch
```

I retain the terminal results in `corrected-batch.log`. Exact-revision remote
CI and full 5.1 acceptance remain required; these repairs do not close them.

# I qualify private callable arguments and results

Under #989, over `a142b6018`, I extend my private target analysis with bounded
same-module formal/result summaries. I describe the implementation and its
remaining boundaries in [my callable contract](../../NANOISA_FILE_CALLABLE_TRANSPORT.md).
The existing carrier preserves identity through argument and return transport;
its VM and native dispatchers consume the newly checked plans.

My complete indirect query gate passes all six methods: target analysis in
2.770 seconds, ownership in 2.944 seconds, and hosted preparation in 6.922
seconds. Existing malformed inputs, late incompatible candidates, initialization,
recursion, copied facts and allocation controls remain active.

Both expanded sanitizer dispatch methods pass in 131.322 seconds on Darwin.
Each configuration captures 68 exact serialized modules and emits 65 C products;
the remaining three modules deliberately refuse. VM and native O0/O2 agree
byte for byte on 838 instrumented traces and 832 linked-provider traces. The new
cases cover a returned factory callable, a callable identity function, a
higher-order helper, direct/indirect wrappers, both targets, catalog permutation,
scalar/File/OpenResult transfers, every lower fuel budget, denial, cleanup error,
unresolved formal and recursive-graph refusal. The higher-order site's candidate
set is checked explicitly. The richer owned fixture exercises 436 preparation
allocation refusals with fresh recovery; emitter allocation controls retain354
refusals. Exact command endpoints are successful, reaped and group-cleaned.

The runner instruments its listed providers and generated C with ASan/UBSan and
enables leak detection; common linked objects retain their prior builds. I retain
the precise inputs, original command log, serialized manifests, generated-source
hashes, command statuses and both reference trace streams. This is not a claim
that every common linked object was rebuilt with sanitizers.

My old File-flow fixtures pass 3,477 instrumented and 3,332 linked checks. Both
unchanged cyclic dispatch methods pass in 38.356 seconds. GCC 16.2.0 accepts the
modified flow implementation, expanded fixture and all65 generated C products
with `-Wall -Wextra -Werror -fsyntax-only`. That is local compiler checking, not
Linux execution.

I run the query/dispatch batch with Homebrew LLVM Clang on PATH and
`OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5`:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-indirect-queries test-file-indirect-dispatch-sanitize
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-flow test-file-cyclic-dispatch
```

During this batch, a142b6018 Linux CI exposes an archive ordering failure before
carrier execution. I retain and repair it in the neighboring link-order evidence;
these Darwin results do not establish that remote gate. Paired source/full shadows,
indirect borrowed formals, public/installed consumers, Linux and the complete
5.1 release requirements remain open.

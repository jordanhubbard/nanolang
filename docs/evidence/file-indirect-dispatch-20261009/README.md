# I match indirect File VM and native execution

Under #989, over `796a8cf1d`, I implement the separate private adapters described
in [my dispatch contract](../../NANOISA_FILE_INDIRECT_DISPATCH.md). The VM uses
the checked indirect carrier; generated C selects real generated functions from
the prepared candidate set. No public/source route changes in this checkpoint.

## My execution results

My first ordinary corpus passes both methods in65.593s. I then add changing
targets across zero/four loop iterations and a generated-call mutation control.
The sanitizer run captures54 exact modules in each configuration, emitting53
and refusing one deliberately unsupported module. Each mode's VM and native
O0/O2 traces agree byte for byte:

| Provider mode | Traces in each execution | Result |
| --- | ---: | --- |
| Allocator/host instrumented | 536 | VM = native O0 = native O2 |
| Linked providers | 530 | VM = native O0 = native O2 |

I compare status, original failure site, fuel, output, File operations and
cleanup details. The corpus includes both candidates, scalar/File/OpenResult
transfers, changing callable locals, preserved caller operands, nested direct
calls, initializer fuel, every lower fuel limit on the new positive cases,
denial, assertion failure, invalid identity and cleanup faults. Existing direct
borrow/loop cases execute through the new plan as well.

I retain424 preparation allocation refusals with fresh recovery in the
instrumented VM and each native replay, plus354 emitter allocation refusals.
These counts describe the measured fixture sweeps, not every possible input.
Tracked carrier execution allocates no project heap. The runner instruments its
listed providers and generated C with ASan/UBSan and enables leak detection;
common linked objects outside that list retain their prior builds.

## My retained failures and corrections

The first setup used the old native-object environment name and ran no tests.
I correct the Make binding and retain [that failure](setup-failure.log).

The expanded sanitizer command finishes both VM captures and all four native
replays successfully, then fails the final generated-target mutation control.
That control selected the zero-iteration module, whose altered call never ran.
I preserve [the original command result](sanitizers-first.log) and its nonzero
terminal in [the summary](summary.json). I select the positive scalar-call
module and rerun the native isolation controls against the same generated
artifacts and providers. All corrected controls pass in12.728s. I do not relabel
the first wrapper as a pass or rebuild the already-passing corpus unnecessarily.

Isolation covers altered ABI revision/size, later variants, references, edges,
candidate sets and selected generated calls. Exact native links omit the VM,
FFI and emitter providers; symbol checks reject VM execution dependencies.

Both unchanged cyclic dispatch methods then pass in38.475s. All five workflow
checks and `git diff --check` also pass. I wire ordinary indirect dispatch into
the platform CI gate; its remote result is not established here.

## My reproducible boundary

I run on Darwin with Homebrew LLVM Clang, its `bin` directory on PATH and
`OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5`:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-indirect-dispatch-sanitize test-file-cyclic-dispatch
```

The first invocation stops at the fixture issue above, before the cyclic
neighbor. I resume corrected isolation separately, then run the unreached
cyclic target. I retain the [ordinary result](ordinary.log),
[corrected isolation](isolation-corrected.log), [cyclic neighbor](preserved-cyclic.log),
[source hashes](sources.json), [provider configuration](sanitizer-inputs.json),
generated-source hashes, module manifests and both exact trace streams.

Private matched execution is now tested on Darwin. Public grants, installed
consumers, paired source/full shadows, callable parameters/results, indirect
borrowed formals, Linux and the complete 5.1 release requirements remain open.

# I qualify my original managed-string acceptance

I record task_5792220dc3654ddcbe7e47ec0253f8ea under original parent
`task_51da49b39230468784da3481b893563b` before changing any fixture. My base is
actual grant merge dc83a0c92432534226d50b422a9d0b8e78657f79. This is a test and
acceptance reconciliation, not a new opcode or executable admission.

My original [contract](NANOISA_MANAGED_STRINGS.md#my-acceptance-evidence) promises
VM, native LLVM before and after optimization, and Wasm comparisons. Core,
frame cleanup, concat, substring and portable conversions are merged. The
Darwin parser child7ba is also complete through its retained evidence; its old
open wording is historical. Aggregate/cycle488, host linkage2d2 and full
applicable-language coverage remain separate requirements.

Static review of `tests/test_llvm_managed_strings.py` shows LLVM verification,
explicit ASan instrumentation and llc object generation. The C harness is built
with -O1, but that does not apply a LLVM optimization pipeline to the already
emitted module. I add a test-only selector for the exact `default<O2>` pipeline
to the common emitted managed-string fixture. The absent selector keeps the
existing route. Unsupported selector values fail setup instead of silently
selecting another mode. I retain the original and optimized native/wasm32 IR,
verify both, and run the same unchanged behavioral assertions. Structural
checks of emitted cleanup/traps apply to original IR before optimization; they
are not claims about an optimizer's symbol or call-count preservation.

I apply optimization before marking/instrumenting the selected native IR for
ASan. I keep the explicit ASan pass and checks for actual instrumentation.
Native allocation controls continue to replace only generated-module malloc;
the instrumented C harness and linked engine scope remain accurately distinct.
Wasm stays import-free and uses the same runtime exports, bounded memory and
actual Node/Wasmtime controls. No expected failure becomes a success, and no
publication sentinel, lifetime counter or page bound is removed.

I qualify the original11 emitted string methods and the existing decimal and
portable float-format conversion methods in both modes. The unchanged core,
package and binary64 parser/formatter acceptance supplies allocator, growth,
reference-bit and production-mode coverage. Before choosing the exact runner
I inspect those modules and all prerequisites; shared helper inheritance must
not accidentally omit a promised method. Existing scalar/literal profile and
verifier refusal neighbors remain part of the original contract.

I use fresh isolated Linux and Darwin tools with explicit supported Clang/LLVM
selection. Every command retains its actual status, bounded process-group
cleanup and output before assertions. Temporary sources, IR, Wasm, binaries
and module-generated objects remain archived; source/tool/provider maps
distinguish immutable inputs from generated products. Failed terminals are
preserved without replay. The source and fixture changes receive independent
review before execution. No complete managed-string or parent closure follows
until an original-criterion evidence matrix supports every claimed boundary.

The unresolved historical evaluator incident791a retains its own evidence.
It does not establish a current managed translator defect, nor does a passing
new gate retrospectively explain it. Full product/fixed-point and release
acceptance remain required.

## I resolve the independent review inventory before code

The selected optimized Wasm artifact itself must run through Wasmtime as well
as the existing Node assertions. The separately published nvm2wasm artifact
remains an additional CLI control; it does not prove the selected O2 route.
The default-main native fixture also uses the same explicit optimization
selector before its ordinary executable link. Its existing link flags do not
retroactively prove IR memory instrumentation; the separate marked-IR harness
retains that claim.

The original eleven string methods have only a successful initializer. I add
one bounded string-only failed-initializer control: retain a committed dynamic
global, allocate a separate local owner, fail ASSERT, refuse entry publication,
release temporary roots and retain exactly the committed global until explicit
disposal. Both selected native and Wasm routes must report ASSERT with exact
live object/byte counts and zero counts after disposal. This is the original
failed-initializer promise, not aggregate admission.

The unchanged core fixture defaults to detect_leaks=0 on Darwin. I record that
limitation and add an explicit strict leak-check selector for the supported
Homebrew compiler qualification; the release run must choose1. I do not call
a default Darwin run leak-checked. Inherited LSAN_OPTIONS is cleared by the
qualification runner. Invalid selectors fail setup.

Conversion classes copy methods rather than inheriting the common class.
The optimization helper is module-level, so every copied compile method uses
the same implementation without missing aliases. An external retained runner
archives each command's overwritten outputs before any assertion, as well as
actual statuses/timeouts and original TemporaryDirectory products. Keeping
only the final directory would lose earlier subcase IR and is insufficient.

## I retain a direct, prepared-provider runner

My review checkpoint includes `scripts/qualify_managed_strings.py`. I prepare
providers separately before freezing a phase; this runner invokes no Make,
bootstrap or implicit provider rebuild. It requires my five public CLIs,
`obj/binary64_parser_vm`, `obj/scalar_global_lifetime`,
`obj/literal_string_aliases`, `obj/generic_numeric_bits`, and
`obj/test_verifier_profiles`. Their current Make recipes define preparation;
this checkpoint does not claim preparation has happened on either host.

I select exactly four direct unittest phases:

- `original`: the complete string, decimal, scalar-format, binary64-format and
  binary64-parse modules, with the original emitted IR route.
- `O2`: those same five modules with `default<O2>` applied by the reviewed
  common helper; tests which compile their own C retain their existing flags.
- `core-package`: the complete core and runtime-package modules, including
  strict core leak checking and actual package generation inside retained
  directories. I do not silently substitute all aggregate Make dependencies
  for this bounded string acceptance.
- `neighbors`: complete scalar-global, literal-string, enum-scalar,
  generic-numeric and verifier-profile modules. These remain unoptimized
  neighbor controls, not an additional O2 claim.

I retain the discovered unittest IDs, actual counts and skips; a skip or
expected-failure result cannot qualify a phase. I stop after the first failed
test (including its unittest cleanup), while expected negative subprocess
statuses remain required assertions within their methods.

An explicit JSON selection provides absolute executable paths for `clang`,
`cc`, `opt`, `llc`, `llvm-as`, `lli`, `wasm-ld`, `node`, `wasmtime`, and `python3`, plus absolute paths to the
actual selected sanitizer/runtime libraries under `libraries`. I launch with
that exact Python. My private PATH aliases preserve the fixtures' literal tool
names; CC and the runtime/core selectors select the supplied Clang. The
selection may include other tools. I pin the scalar helper CLI environment
overrides to this checkout rather than inheriting another checkout. Host preparation records
SDK and actual compiler resource/library resolution before this phase; a list
of hashed libraries alone is not proof that a loader selected each one. The
runner hashes the aliases, actual executables and explicit libraries and
retains version commands. It does not claim every transitive system tool is
inventoried. I clear LSAN_OPTIONS and select detect_leaks=1 on both hosts.

For example, after reviewed preparation I invoke a phase with:

```
/absolute/selected/python3 scripts/qualify_managed_strings.py \
  --phase original --tools /absolute/host-tools.json \
  --output /absolute/new-evidence-directory --phase-seconds 14400
```

I intercept direct fixture `subprocess.run` calls before importing the test
modules. Every supported call captures separate file-backed stdout/stderr,
argv, selected environment, status and duration. A bounded process group gets
TERM, a five-second wait, then KILL and a separate five-second final wait,
even when its leader has exited. I bound group-disappearance polling at two
seconds, retain the exact cleanup outcome, and refuse an unconfirmed cleanup. A phase deadline and TERM handler pass through the same
cleanup. An unexpected subprocess calling convention fails closed.

At phase entry and exit I freshly hash all tracked non-documentation inputs,
the acceptance contract/roadmap, prepared bin/obj files and selected tools into
a content-addressed store. I retain the full tracked path list and explicitly
list excluded documentation, including historical docs/evidence artifacts; no
participating compiler, build or fixture source is omitted. Per-command maps
use a digest cache keyed by resolved path, device, inode, size, mtime_ns and
ctime_ns, checking those fields again after a fresh read. Alias paths share one
read of the same resolved file. Changed identities trigger fresh hashes; phase
endpoint hashing clears the cache. Product maps use the same identity cache,
with a forced fresh final product map, and preserve all archived bytes. Every overwritten IR/C/object/binary version seen at
these boundaries stays available before fixture assertions run. I preserve
fixture TemporaryDirectory products. A private sitecustomize hook inherited
by nvm2wasm copies CLI intermediates into a separate archive directory before
performing its normal cleanup; the unchanged refusal tests still require no
`.nano-wasm-*` leftovers. Inner CLI/compiler subprocess
commands retain the outer command's output/status; I do not claim separate
traces for every transitive compiler process. The CLI's copied intermediates retain their original path in an origin record
and appear in the outer command's post-map. A child killed before cleanup
leaves its original directory under the retained fixture directory instead.

Per-command maps reject changed source/provider/tool identities or content.
They combine stat observations with previously measured digests; they are not
independent byte reads on every command. Fresh phase endpoint maps establish
byte equality at those endpoints, not continuous observation between them. Generated test products are separate from immutable
inputs. I record failures before propagating them, preserve the original
phase, and use a new output directory for any reviewed correction. Final
report digests seal maps/logs/statuses; product maps name archived content by
SHA-256. This is a runner checkpoint only: no build or qualification has run.

## I correct the runner review findings before any gate

Root review of6c310 found an unbounded final wait after SIGKILL and excessive
repeated hashing of historical evidence plus repeated LLVM aliases. I replace
the wait with the bounded, explicitly reported cleanup above and introduce the
stat-keyed per-command cache with fresh phase-boundary hashing. An unconfirmed
cleanup cannot pass. I preserve all product boundary snapshots and archive
bytes; this corrects retention machinery without changing the two fixtures or
any compiler/runtime source. No failed build or fixture exists for this static
review finding, and no qualification has yet run.

## I pin the native GCC installation for Clang23

The first fresh Linux preparation atcd312 stopped before provider qualification:
`/tmp/nanolang-managed-string-cd312-linux-prepare/terminal.json` retains return2
at0.114840524s, confirmed process cleanup, and unchanged source/tool maps.
Clang23 selected GCC14 but diagnosed its missing libstdc++ include directories
under `-Werror,-Wgcc-install-dir-libstdcxx`; it identified the installed GCC13
alternative. No test ran. I retain this original tree and terminal.

Before corrected preparation I extend the explicit selection JSON with optional
`native_clang_flags`, an array of nonempty, NUL-free strings (absent means empty).
Linux selects `--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`. I form CC
with `shlex.join([selected_clang, *flags])` and NMS_NATIVE_CLANG_FLAGS with
`shlex.join(flags)`. The complete original selection remains in the evidence.
The literal clang/wasm tool and NMS_WASM_CC stay unflagged; native-only settings
must not contaminate the wasm32 toolchain. This selects a real installation;
it does not suppress the warning. The external preparation driver uses the same
selection for its native Make commands. Independent review precedes corrected
execution; no compiler, runtime or test assertion changes here.

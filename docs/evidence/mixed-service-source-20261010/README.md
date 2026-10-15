# My paired mixed source checkpoint

I extend `418c38879` under #990 with declaration-qualified C and Nano lowering.
My [source contract](../../SERVICE_MULTI_TRANSPORT.md#paired-source-lowering)
records instance mapping, single-catalog compatibility and the remaining CLI
publication boundary. My full 5.1 scope remains required.

## Mixed fixture

My focused test passes in 95.172 seconds on Darwin with Homebrew LLVM Clang.
I import File/TCP/File from distinct real source modules. I retain all three
owners concurrently, move owners through helpers, call a File writer indirectly,
forward borrows, write/read distinct File bytes, and perform loopback TCP
finish/send/receive/close. Endpoint fields appear in reverse order in source.
I compare exact serialized bytes from C and the independently compiled Nano
lowerer, executing that lowerer in both NanoVM and generated native code.
I then execute the resulting mixed program in the public VM and generated C.
IPv4 and IPv6 main entries return 90; selected source shadows return zero after
checking that result. The peer observes byte 165. This is a direct lowerer/public
API test, not CLI publication or a fresh compiler bootstrap.

## Failures retained

My first source fixture omitted consumption of outer live owners when a nested
acquisition failed. The ownership checker correctly refused it. I add explicit
close operations on both failure paths. The full inline fixture then reaches
the established 256-instruction function limit. I move File-read and TCP-exchange
operations into borrowed helpers, retaining all operations and simultaneous
ownership. A temporary instrumented flow query identifies the limit at
`service_code.inc:71`; I do not change that bound.

The resulting C fixture executes, but the Nano parser refuses the same source.
A retained token diagnostic locates the brace following `tcp.Endpoint`. I add
recognition based on the capitalized type component after the lowercase alias,
with bare, grouped, argument and tuple-literal controls. Both VM and generated
native executions of the parenthesized-parser regression pass. I retain the
original parse refusal separately from the corrected source run.

## Validation scope

My standalone sanitized C corpus passes all 13 methods in 133.326 seconds.
GCC 16 accepts the lowerer and fixture with C11 and strict warnings. Generated
native probes/products use ASan/UBSan; the C sanitized runner instruments its
included lowerer and File runtime but not every linked compiler/runtime object.
I do not claim whole-program sanitizer coverage or a release-candidate fixed
point. The full paired regression terminal is retained separately.

I coordinate this batch through [issue #990](https://github.com/jordanhubbard/nanolang/issues/990#issuecomment-6096357198).
My next batch must select the mixed public emitter/runtime in the source product,
require the existing File and TCP opt-ins independently for every declared
instance, create fresh per-instance grants for supervised shadows, and preserve
staged output on denial or shadow failure. Raw-module CLI admission and installed
product linkage must use the same checked boundary. I keep DNS/WebSocket and the
remaining compiler/platform/release requirements open.

## Final gate

My [full paired terminal](paired-final.log) passes all 17 methods in 294.757
seconds, including single File/TCP, mixed execution, selected/all shadows,
owned/borrowed indirect calls, alias cleanup, early terminal operands and wire
refusal. I retain [the final generated inputs/products](paired-final-artifacts.tar.gz),
[development artifacts](development-artifacts.tar.gz), [source hashes](sources.sha256)
and [commands](commands.txt). My final Make build exits zero after removing
redundant mixed-query archive entries; no product source changed during this
regression. These gates complete this lowerer batch, not CLI or 5.1 release.

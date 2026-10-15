# Execution-time shadow trace correction

I retain each selected target at a dedicated NOP offset in advisory metadata.
My ordinary and owned emitters write the marker before the selected shadow;
NanoVM reports it to stderr only when reached and `NANO_SHADOW_TRACE` is present.
I do not grant execution or ownership authority through these diagnostics.
The format and relocation limits are in `docs/NANOISA_ADVISORY_METADATA.md`.

My final nine-method emitter suite passes in 79.621 seconds. It covers repeated
names, suffix/empty selection, absence of tracing by default, owned shadows,
unreachable shadows after a failure, invalid markers and a nonempty textual
round trip. My initial suite run passes the new trace controls but fails two
existing stdout assertions because I changed their helper to use the shadow
supervisor. I restore their original direct execution; the new trace controls
use the actual supervised route. I retain both results.

`make -j2 test-nanoisa` passes all 2,988 checks.
`make -j2 test-nanovm` passes all 274,642 checks plus its allocation/recovery
controls. `make -j2 test-advisory-metadata` passes 188 advisory checks, allocation
controls and its public artifact/native method. My initial direct invocation
of its Python method fails because the required C fixture was not built; the
Make target builds and runs the complete dependencies. I retain that failure.

My source hashes record the implementation tested over base `53d01164d`.
Fresh bootstrap and the complete paired parser corpus remain required under
GitHub issue #978. This checkpoint does not close parser or release acceptance.

# I name ordinary native invariant failures

My product startup evidence included an abort without a compiler diagnostic.
I recorded `task_fb49a88a8ab84385a52858243214d237` and roadmap contract
`d10264f5` before implementing this diagnostic foundation. I neither attribute
that failure nor claim to repair startup.

My ordinary `nvm2c` emitter now reports the generated C function and line to
stderr before its existing `abort()`. I preserve every condition and cleanup
sequence, including ordinary map storage/root helpers and the newly merged
boxed numeric helper. This does not cover the separate owned-profile emitter
or foreign code. The line identifies generated C, not a NanoLang source line;
I must retain that generated source to interpret it.

At source `32c83d96`, the existing full native gate passes 2,422 native and
1,092 shape checks. Two ordinary controls check a failing helper assertion's
exact SIGABRT, diagnostic function and matching generated line, and a passing
assertion's unchanged stdout and silent stderr. Generated-code ASan/UBSan/leak
checks pass with GCC in 3.063 seconds and Clang in 0.346 seconds.

My first bare Clang invocation stopped before compilation because its GCC
installation-selection warning became an error under `-Werror`. I retain
`/tmp/nanolang-native-invariant-clang.log`. Selecting the existing GCC 13
installation explicitly with
`clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13` passes without
disabling warnings. This is a toolchain-selection result, not a product repair.

Independent review normalized the emitted macro calls back to `abort()` and
confirmed that the initial production delta then reduces to the diagnostic
macro and its standard header. Final source `40d78215` integrates main through
PR611/612 and applies the same replacement to its numeric helper and ordinary
map include files. Its focused integration gates are still running.

I retain logs under `/tmp/nanolang-native-invariant-*`. The separate product
startup and export-shadow holds remain open. I did not replay their artifacts.

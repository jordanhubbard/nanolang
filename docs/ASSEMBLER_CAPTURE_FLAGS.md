# My assembler capture build flags

My Linux capture helper is a shared library loaded into an external assembler.
Its old recipe discarded all caller CFLAGS, including Clang's explicit GCC
installation selection. Public cyclic installed-package qualification retained
the resulting warning-as-error failure under task_befc0deb04bc4a73b4ef2931ca7b604d.

I pass CPPFLAGS and introduce NANO_AS_CAPTURE_CFLAGS and
NANO_AS_CAPTURE_LDFLAGS. Their defaults inherit CFLAGS and LDFLAGS except
-fsanitize=... driver switches. Those switches instrument the compiler/runtime
process; automatically loading the resulting sanitizer runtime into an unrelated
assembler changes that host's requirements. Explicit helper variables remain
available for a compatible dedicated instrumentation check. This is the existing
helper instrumentation boundary, now stated rather than imposed by dropping all
flags. I retain caller optimization, warnings, include paths, target/sysroot and
linker selection, then append the required -fPIC/-shared and -ldl. I guard the
source's _GNU_SOURCE definition so the ordinary project's definition is accepted.

I review the production delta before qualification. I build using the actual Make
recipe with default GCC and strict Clang plus its GCC13 selection in CFLAGS,
verify caller preprocessing with a required forced-include probe, and run the
existing assembler capture/replay corpus against each produced helper. I also
check explicit UBSan helper instrumentation through that corpus. Linux owns this
recipe; Darwin's target selection remains unchanged. I do not claim that an
arbitrary sanitizer can safely interpose an arbitrary external assembler.

My [independent evidence review](evidence/assembler-capture-flags-independent-review.json) checks all34 actual Git report blobs, four current helper hashes, five unchanged inputs and eight successful build/test terminals. Each corpus ran10 tests; ELF initial data independently confirms the three probe values are42. No helper was executed again for this review.

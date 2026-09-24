# Nominal cycle validation

My canonical path previously bypassed the by-value layout check in the old C emitter. At `54e15bce4`, the original record and mixed-cycle tests still failed on both native stages. My expanded thirteen-method baseline adds generic payload, expanding generic recursion and forward generic-dependency cases; it records ten failing subcases (`native-baseline.log`).

I now validate declaration dependencies before canonical program and shadow emission. A monotone parameter-demand pass determines which generic arguments are actually stored by value. I then topologically remove the resulting record/union dependency graph. Phantom arguments do not create storage dependencies. Arrays, lists and maps are pointer-backed boundaries; finite tuple components remain by-value. I do not generate C to validate layouts, unfold infinite generic instantiations, or use a nesting cutoff.

Emitter shadows test parameter demand, finite nested generics, phantom arguments, name shadowing, pointer-backed recursion and cyclic declarations. The bytecode publication regression checks eight cycle forms and six finite forms, preserving prior output on rejection and running accepted modules in NanoVM. All thirteen emitter-driver methods pass (`emitter-suite.log`). The native suite also includes a 151-record finite chain.

I separately preserve the terminal historical d48 CI snapshot. Units-13 job `107552439774` times out after 180 seconds while compiling enum-metadata tests on both native stages (`hosted-enum-timeout.log`). I added these observations to deadline task `task_4251e719e9634aa7b597dbdd080b6b9b`; I have not assigned a cause or increased the deadline. Units-12 was cancelled and does not count as qualified. The negative partition passed.

Final hosted acceptance, final-source fixed points and the remaining PR gates are still open.

Review found two additional gaps before landing: tuple commas inside a generic argument were split incorrectly, and a nominal declaration named `List` was mistaken for a pointer-backed container. The retained source probes and empty successful compiler logs record both accepted cycles; `probe-modules.json` records the incorrectly published module hashes. I corrected balanced generic-argument splitting and made the container exception distinguish declared nominal types. The final matrix includes these failures and a finite record named `List`. Initial bootstrap/emitter logs are checkpoints, not evidence for the subsequent corrections.

The reviewed source passes the complete fresh bootstrap, including installed/no-C-seed smoke checks (`bootstrap-reviewed.log`). Earlier `bootstrap.log` and `bootstrap-final.log` are superseded checkpoints; each completed before the next rebuild began. I explicitly invalidated the bootstrap source timestamp after the mid-build reviews so both final stages include every correction.

All thirteen emitter-driver methods pass on the reviewed source (`emitter-reviewed-suite.log`, 1.828 seconds). The complete 31-method One IR compiler gate passes (`one-ir.log`, 339.314 seconds), including C-seed and self-hosted full compiler-to-native-to-product paths. These use ordinary host-runtime objects; they do not qualify the final instrumented-host-runtime or hosted gates.

All thirteen native nominal-order methods pass on both final stages, including the original output-preservation assertions, primitive-list fields, generic cycles, finite/phantom/pointer cases and the 151-record chain (`native-reviewed.log`). The same complete suite passes with ASan/UBSan native products, leak detection and stack-use-after-return detection (`native-instrumented.log`). Compiler executables are ordinary builds. The earlier scalar/list VM/native regressions also pass with instrumented native products (`adjacent-instrumented.log`).

Commands:

```sh
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make nanoisa_emit
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 make -j2 bootstrap3
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_nanoisa_emit_driver -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_SHADOW_TIMEOUT_SECONDS=60 python3 -m unittest tests.test_one_ir_compiler -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest tests.test_native_nominal_order -v
NANO_CC=/opt/homebrew/opt/llvm/bin/clang NANO_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' NANO_LDFLAGS='-fsanitize=address,undefined' ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m unittest tests.test_native_nominal_order -v
```

I confirmed cancellation of superseded CI runs `35983152810` and `35985131037` after retaining their snapshots. The historical d48 run is terminal failure; cancellation is not qualification. I preserve the current candidate's hosted gate separately.

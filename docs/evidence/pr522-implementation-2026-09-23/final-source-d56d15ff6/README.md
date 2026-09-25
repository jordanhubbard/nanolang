# Final compiler-source instrumentation and compatibility

At `d56d15ff6934dcd4872aa0f90bfe7ea828cf79fa`, my complete instrumented One IR suite passes all 32 methods in 432.838 seconds. Both full compiler paths and the real std artifact execute with the private ASan/UBSan host runtime and instrumented generated native products. Leak and stack-use-after-return checks remain enabled. The retained runner changes only the selected runtime object and compiler/link environment; it retains all methods and deadlines.

The runtime build provenance remains in `../instrumented-host-runtime`. Its source and the translator are unchanged between the preceding complete 32-method stack-repair pass and this final-source run; `hashes.json` records the current runtime sources and exact private runtime object. The seed compiler, VM and translator executables are ordinary builds. This does not substitute for hosted instrumentation of every provider.

My unchanged generic-function and C-seed-union-signature suites pass all 27 methods in 40.606 seconds. My unchanged generic/nongeneric selected-ownership suites pass all 36 methods in 51.799 seconds. They cover the C seed and both native stages, admitted values and transfers, refusals, and preservation of prior output. I retain generated-program ASan/UBSan, leak/UAR checks and the original 120-second compiler deadlines.

For the compatibility runs I select Homebrew Clang with `NANO_CC`, use `NANO_CFLAGS=-O0 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`, and link with `NANO_LDFLAGS=-fsanitize=address,undefined -L/opt/homebrew/opt/openssl@3/lib`. I retain the CI shadow deadline of 60 seconds. The commands are `python3 -m unittest -v tests.test_generic_function_values tests.test_cseed_union_signatures` and `python3 -m unittest -v tests.test_generic_selected_ownership tests.test_selected_variant_ownership`.

These completed local gates do not establish final hosted acceptance or extend the admitted callback/ownership profiles. VM and standalone-native fixed points have their own evidence records.

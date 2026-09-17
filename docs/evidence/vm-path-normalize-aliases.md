# My VM path-normalization aliases

I recognize `path_normalize` and `nl_os_path_normalize` only as exact builtin FFI names in the empty namespace. I validate their string-to-string signature and argument before calling my shared lexical normalizer. I copy its owned result into my VM heap and free the temporary on both successful and failed copying.

I preserve explicitly selected artifact symbols, including a `path_normalize` symbol returning a borrowed literal. I do not infer artifact result ownership from its name.

At code checkpoint `b576c12756dd1a465b572d765ad529cc222c21d1` on Linux ARM64, I pass:

- `make -j8 nano_vm nvm2c test-vm-ffi test-vm-path-normalize-sanitizers`: all 28 VM FFI cases and the focused sanitizer case.
- `python3 -m unittest -v tests.test_native_host_strings`: four methods in 1.241 seconds, including both aliases through VM and native execution and selected artifact identity.

My direct bridge test makes 1,000 successful calls across empty, rooted, relative, parent, and Unicode paths. After releasing each input and result, my VM heap object count returns to zero. My paired bytecode test loops 10,000 times for each alias.

My sanitizer target instruments the FFI bridge and test translation units with AddressSanitizer and UndefinedBehaviorSanitizer, with LeakSanitizer enabled on Linux. The other linked objects use their normal build flags. This evidence does not establish whole-runtime instrumentation or Darwin qualification.

My integrated log is `/tmp/nanolang-vm-path-alias-integrated.log`. My earlier missing-alias evidence remains `/tmp/nanolang-host-adoption-tests.log`. MAC: `task_4c21884a6a8b417eb6b07b7f82c80d55`.

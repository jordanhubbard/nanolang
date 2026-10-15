# I qualify the complete File driver corpus

I continue #989/#982 on `a66283a75` after shared-reference publication and the
CI portability repair. My earlier shared-reference checkpoint selected four
Nano-driver methods. This batch covers the complete eleven-method corpus.
I retain individual methods and terminal results in the driver logs; I do not
infer a full release gate from those methods.

## Compiler artifacts and deadlines

My first fresh compiler build uses `bin/nano_virt src_nano/nanoc_v06.nano
--emit-nvm -o /private/tmp/nanolang-file-all-driver.nvm`, with
`NANO_SHADOW_TRACE=1` and `NANO_CC=/opt/homebrew/opt/llvm/bin/clang`. The compiler
override rebuilds native module cache generations. I retain the default
ten-second dependency-shadow failure in `compiler-default-deadline.log`:
`$shadow_862_assemble_save` is the last traced shadow. No compiler module is
published by that failed attempt.

I inspect the supervisor: it builds native modules before starting one timer
for the entire selected dependency-shadow suite. The last trace does not
identify the cause of elapsed time. My isolated NanoISA import probe passes
all 25 selected shadows with the unchanged ten-second deadline, including
`assemble_save`; I retain its source and log.

I then timestamp a diagnostic build with the existing explicit
`NANO_SHADOW_TIMEOUT_SECONDS=30` override. It passes all 1,066 selected shadows
and publishes the fresh module. Its total wall time is 71.449 seconds, and
the first-to-last trace spans 8.578 seconds. The interval after `assemble_save`
is 0.370 seconds. These are observed trace intervals, not isolated CPU timings
or proof of the original timeout's cause. `compiler-timing-summary.json`
retains the largest intervals. I keep default-deadline acceptance open; I do
not change the product deadline or any shadow assertion.

The failed build overlaps the complete C-driver run. The diagnostic build runs
after that suite finishes and reuses the native module generations built by
the first attempt. These are not controlled load/cache conditions, so I do not
attribute the difference solely to the timeout override.

I translate that module with `bin/nvm2c` and compile its 14 MB generated C with
Homebrew Clang, `-O1 -g`, `bin/nano_aot_runtime.o` and `-lm`. The native build
exits successfully. I reuse the existing AOT runtime object; the linker reports
debug-map timestamp mismatches for three constituent runtime objects. I retain
those warnings in `native-build.log` and do not qualify debug symbols here.
Both forms reference the immutable File library generation
`.nano-gen-Kkwqkb`. I check all 254 recorded project source/dependency hashes
against current contents. My artifact digest file identifies the executables,
module, runtime archive and selected library.

## Driver coverage

I run `python3 -m unittest -v tests.test_service_drivers` with
`NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`. All eleven methods pass
in 427.097 seconds, each exercising both C compiler drivers.

For the Nano-driver run I set `NANO_FILE_DRIVER_MODULE` and
`NANO_FILE_DRIVER_NATIVE` to the fresh temporary artifacts above and run
`python3 -m unittest -v tests.test_nano_service_driver.NanoServiceDriver`.
I set the same native test compiler. I do not inherit the diagnostic compiler
build's timeout override: File publication shadows retain their normal bound.
All eleven methods pass in 621.709 seconds, each exercising both VM and native
compiler forms. I retain the complete terminal result in `nano-drivers.log`.

These methods cover helpers and loops, generated shadows and byte parity,
native invocation grants, shared aliases and forwarded loans, exclusive
multi-borrow maps, mixed owned operands, overlap refusals, import/root shadow
selection, failed-shadow output preservation, protected output aliases,
compiler failure cleanup, and invalid-body refusal. They do not qualify File
indirect execution, a self-hosting fixed point, or Linux/Darwin Stage 1/Stage 2
equivalence. GitHub DNS failures prevent publication and refreshed remote CI
or worker status during this batch. The full 5.1 contract remains open.

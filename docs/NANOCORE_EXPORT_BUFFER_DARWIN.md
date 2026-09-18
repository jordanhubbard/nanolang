# My Darwin export-buffer fixture boundary

I repair task_43dea95525b24546b4b3e259a3148205 from canonical f761af4c.
My active Darwin SDK defines `vsnprintf` as a fortified macro. My fixture includes
stdio first, then defines the same macro for deterministic format-failure
interception; strict compilation refuses that redefinition.

I preserve the original failure log through its owning Darwin gate. I do not
rerun the failed source. Before code, I record this bounded correction: after
`buffer_vsnprintf` has been preprocessed against the platform's ordinary stdio
API, I conditionally undefine an existing `vsnprintf` macro immediately before
my test-local definition. The included exporter still calls my fault wrapper;
the wrapper still calls the ordinary platform formatter. I do not disable
fortification globally, suppress warnings, alter production, remove assertions,
or change first/second formatting failure budgets.

I retain strict `-Werror` and run the corrected export-buffer target on Darwin.
I also compile/run the included exporter fixture at O1 under GCC and Clang
ASan/UBSan on Linux and Clang on Darwin, with leak checking where supported by
the established target harness. Common linked objects retain their ordinary
build; sanitizer claims cover the included exporter/fixture, not all common
runtime objects. I preserve exact commands, tool identities and source hashes.
This fixture correction does not close the adjacent reference-evaluator leak,
shared-match policy acceptance, full NanoCore acceptance, or release.

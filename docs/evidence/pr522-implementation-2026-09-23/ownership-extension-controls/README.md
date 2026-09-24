# Ownership extension refusal controls

I reproduce the hosted failure in the unchanged `make test-ownership-contracts`: `check_union_transport` expects unknown-format rejection when extension kind 3 replaces the required union facts. Kind 3 now names `SCALAR_GLOBALS`, so framing recognizes it and rejects the missing union declaration with `NVM_V2_ERR_SECTION_TYPE`.

I preserve the original unknown-kind refusal with `UINT16_MAX`. I explicitly retain the recognized scalar-global kind as a separate malformed-placement check. In the extension suffix, I retain the scalar-global rejection under ownership version 3 and add a separate unknown-kind rejection. I change no validator behavior, accepted payload, revision check, duplicate-kind check, or output-preservation assertion.

The complete corrected owning gate passes (one Python method, 0.562 seconds), exercising the C transport checks, ordinary VM/native output, reference/owned-union refusals and unchanged prior output. The same C harness passes against private ASan/UBSan objects with `detect_leaks=1:detect_stack_use_after_return=1` and `halt_on_error=1`; its build command is in `instrumented.log`. VM and translator executables in the owning Python gate are ordinary builds. This does not qualify the full instrumented host runtime.

I track final hosted acceptance under `task_6fa5997a55b946a3a61ec4aa786dba24`.

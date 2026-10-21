# My immutable instrumented worker receipt

I qualify the local Linux ARM64 image transport and bounded instrumentation controls, not the full hosted amd64 bootstrap. My final container smoke passes all 18 checkpoint/container controls, actual GCC ASan/UBSan and gcov/lcov execution, exact provider restoration across different outer runner labels, changed flags/archive refusal, read-only root, GitHub output/summary mapping and exact exit 23 propagation. Private compiler temporary files are disk-backed rather than consuming the worker RAM budget.

- Image content ID: `sha256:885454cfdb893f15290671d474a2a51793a1532a40bbf03440e51e4a1fa12f02`
- Config ID: `sha256:ddb2f18feb98b229b38330c17ba4210285ec6259497ab0338f3fa8725fcdcf4d`
- Archive SHA256: `2d903777775a8f0933517ca4e1758eaee28f49bd2df49d51b14f3806baff258c`
- Actual platform: `linux/arm64`; GCC/gcov 13.3.0, lcov 2.0.

I retain the initial missing-CA failure and subsequent Docker 29 OCI-manifest/config mismatch. The corrected recipe bootstraps pinned CA bytes with TLS/signature verification enabled; archive validation checks both OCI and classic Docker identity forms. These failures are not counted as passes.

I retain the actual [image build](image-build.log), [container smoke](container-smoke.log), [unit controls](controls.log), [CA failure](initial-ca-failure.log) and [OCI address failure](oci-address-failure.log). Full exact-head hosted amd64 image, sanitizer Stage 1/continuation and coverage bootstrap/continuation qualification remain open. The original run 39330564550 remains failed; these local controls do not overwrite its evidence.

I also pass all 33 existing CI workflow/change/shadow/checkpoint/container methods after adapting four obsolete workflow assumptions. The [initial failure](workflow-initial-failure.log) retained host-package and first-download assumptions; the [corrected gate](workflow-controls.log) asserts pinned image dependencies and exact checkpoint artifact selection without weakening continuation order or budgets.

I integrate both component commits at aggregate `2152a0380` and pass all 33 workflow, changed-file, timeout, checkpoint and container controls together. I preserve the exact checkpoint download selectors and image dependency assertions. Full hosted amd64 sanitizer/coverage continuation remains pending; local ARM64 smoke is not substituted for it.


## I retain the hosted coverage producer milestone

Component run39544892945/job119042711803 succeeds inside the exact verified linux/amd64 image. Both raw generations are byte-identical, native/install checks complete, and the complete checkpoint packages/uploads as artifact12323204442 (1,879,084,924bytes, upload zip SHA256 afa226ef5b914bf636baec7d084d3190fe8d5bfc8068bdbed2f499bc94735853). `hosted-amd64-coverage-producer.log` retains the actual image/archive identities and commands. This is component-head evidence, not final aggregate acceptance. Coverage continuation119061472864 remains queued at this checkpoint; sanitizer Stage1 remains in progress.

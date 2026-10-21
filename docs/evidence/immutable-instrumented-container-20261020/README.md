# My immutable instrumented worker receipt

I qualify the local Linux ARM64 image transport and bounded instrumentation controls, not the full hosted amd64 bootstrap. My final container smoke passes all 18 checkpoint/container controls, actual GCC ASan/UBSan and gcov/lcov execution, exact provider restoration across different outer runner labels, changed flags/archive refusal, read-only root, GitHub output/summary mapping and exact exit 23 propagation. Private compiler temporary files are disk-backed rather than consuming the worker RAM budget.

- Image content ID: `sha256:885454cfdb893f15290671d474a2a51793a1532a40bbf03440e51e4a1fa12f02`
- Config ID: `sha256:ddb2f18feb98b229b38330c17ba4210285ec6259497ab0338f3fa8725fcdcf4d`
- Archive SHA256: `2d903777775a8f0933517ca4e1758eaee28f49bd2df49d51b14f3806baff258c`
- Actual platform: `linux/arm64`; GCC/gcov 13.3.0, lcov 2.0.

I retain the initial missing-CA failure and subsequent Docker 29 OCI-manifest/config mismatch. The corrected recipe bootstraps pinned CA bytes with TLS/signature verification enabled; archive validation checks both OCI and classic Docker identity forms. These failures are not counted as passes.

I retain the actual [image build](image-build.log), [container smoke](container-smoke.log), [unit controls](controls.log), [CA failure](initial-ca-failure.log) and [OCI address failure](oci-address-failure.log). Full exact-head hosted amd64 image, sanitizer Stage 1/continuation and coverage bootstrap/continuation qualification remain open. The original run 39330564550 remains failed; these local controls do not overwrite its evidence.

I also pass all 33 existing CI workflow/change/shadow/checkpoint/container methods after adapting four obsolete workflow assumptions. The [initial failure](workflow-initial-failure.log) retained host-package and first-download assumptions; the [corrected gate](workflow-controls.log) asserts pinned image dependencies and exact checkpoint artifact selection without weakening continuation order or budgets.

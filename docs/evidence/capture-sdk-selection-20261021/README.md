# I select the qualified SDK explicitly

I retain macOS qualification failures from jobs119125624147 and119125624408 at af4d870f9. Both failed before compilation because generic `xcrun --sdk macosx` selected the Xcode SDK, while the prior qualified capture receipt names the CommandLineTools SDK. The local probe reproduces that distinction and the separate Xcode catalog version-query failure.

I query the exact absolute qualified SDK and validate its own SDKSettings.json version, recording its hash. Compiler/backend hashes, compiler report, wrapper arguments and qualified SDK identity remain unchanged. I do not fall back to another SDK.

Seven selector tests pass, including an actual local xcrun/path/settings check. All68 selector/workflow/checkpoint/container/schema controls pass in2.001seconds. This local SDK check does not claim qualification of the hosted pinned compiler; exact hosted capture and platform suites remain pending.

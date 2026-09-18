# My product acceptance at 84e04f5d

I pin compiler production to `84e04f5d65896b383151b4e5a089b637d887b72d`. My later `e7bccd99` changes only the introspection acceptance import and documentation. These results qualify this source on Linux ARM64; they do not qualify later main or complete my release.

## Product gates

My fresh bootstrap passes. After rebuilding local helpers, all 39 focused product methods pass in 18.721 seconds. All 17 core examples pass. My first complete `test-quick` attempt stops at a missing direct module-facts import in the introspection acceptance program. I preserve that failure and the [fixture correction](product-module-facts-import.md).

At `e7bccd99`, the corrected introspection program passes and all 244 eligible VM examples compile (248 on disk, four existing declared exclusions). The subsequent affine self-host gate stops with 18 checked refusal subcases across Stage1 and Stage2: exact `Box<int>`, `Box<string>`, `Box<array<int>>` and `Choice<int,string>` instances are unsupported. I retain the existing tests as acceptance under `task_6550ccf97eb44f4d8c08f02cd4189cd7`; I have not completed `test-quick`.

My retained logs on the qualification host are:

- `/tmp/nanolang-product-core-integration-bootstrap.log`
- `/tmp/nanolang-product-core-integration-focused.log`
- `/tmp/nanolang-product-core-integration-test-quick.log`
- `/tmp/nanolang-product-facts-import-test-quick.log`

## VM fixed point

My initial module and both VM generations contain exactly 398,464 raw bytes with SHA-256 `60ce960b78a043647e45f1df28be1b566bf97355afa1af3c3342a1cdab4d7ec0`. Both raw comparisons and the declared host-library closure comparison pass. Stage1 takes 760.545 seconds and Stage2 takes 761.447 seconds. Each generated module verifies; the final compiler emits, verifies and executes hello successfully.

My [run manifest](product-vm-fixedpoint-84e04f5d.json) records argv, budgets, durations and hashes. My [post-run integrity record](product-vm-fixedpoint-84e04f5d-integrity.json) confirms unchanged source, capture helper, translator and declared host libraries. I retain `/tmp/nanolang-product-vm-fixedpoint-84e04f5d.log` and its artifact directory.

## Native fixed point

My separately pinned native run passes. The initial module and both native-generated stages contain exactly 398,488 raw bytes with SHA-256 `ba9e46c9ae4cb93a8231073fb67e8d38dfd939346ad4831f2f1fc7464dd1b5fe`. Both raw comparisons and the declared host-library closure comparison pass. Stage1 takes 1,090.732 seconds and Stage2 takes 1,086.731 seconds. All translation, native build, help, verification and hello checks complete successfully.

My [native manifest](product-native-fixedpoint-84e04f5d.json) and [post-run integrity record](product-native-fixedpoint-84e04f5d-integrity.json) preserve the exact source, tools, closure, argv and results. Source, helper, translator and declared host-library hashes remain unchanged. I retain `/tmp/nanolang-product-native-fixedpoint-84e04f5d.log` and its artifact directory.

Separate embedded artifact paths prevent a VM-versus-native cross-run raw-byte equality claim. Darwin acceptance remains pending. PR522 and publication remain held pending complete product and release gates.

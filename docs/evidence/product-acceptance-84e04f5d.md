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

Separate embedded artifact paths prevent a VM-versus-native cross-run raw-byte equality claim. PR522 and publication remain held pending complete product and release gates.


## Darwin qualification

My peer completed isolated macOS 26.6.2 ARM64 qualification with Apple Clang 21.0.0. At exact `84e04f5d65896b383151b4e5a089b637d887b72d`, three-stage bootstrap and 17 core examples pass; the full gate stops at the direct module-facts import fixture issue (exit 2 after 454.28 seconds). The preserved peer log is `/tmp/nanolang-product-84e04f5d-darwin-b2b605a9-test-quick.log`, SHA-256 `c6a6cebd3f38ce1c9888c61c5ff668449cb2dae26573dc3bbd0626c070f4d906`.

At exact `e7bccd998d2af89fa0686701e62ac4613d4a79b1`, the peer confirms identical compiler source trees, successful bootstrap and corrected introspection, all 17 core examples, and compilation of all 244 eligible VM examples. The complete run stops at the same 18 checked generic-union affine refusals as Linux (exit 2 after 752.98 seconds). The preserved peer log is `/tmp/nanolang-product-e7bccd99-darwin-a86b339f-test-quick.log`, SHA-256 `1489385a06b37cf6ba8f0dae59a54b18f30a13be98675f9ae5af5f0cb28d5502`.

My peer attached these results to the existing fixture and generic-union tasks. Neither run qualifies later canonical integrations. Neither run completes the full product gate.

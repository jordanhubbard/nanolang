# xcode-27 capture toolchain requalification (2026-10-26)

Hosted retirement runs on the `xcode-27` runner (image `xcode-27-arm64`
20261023.0307.1) failed with `FileNotFoundError: /Applications/Xcode_27.2_beta_2.app`.
The image now carries Xcode 27.0, 27.1 and 27.2 (release candidate) instead.

Probe (`image-probe-log.txt`, run 40420853610) found these clang builds:

| Xcode | clang banner | clang sha256 |
|---|---|---|
| 27.0 | clang-2100.3.34.2 | 5922f5f7843fee699497c45cc98134b6686e0a5e2fa8a5866ab74c5482fa87a5 |
| 27.1 | clang-2100.3.34.2 | 06e3d64f541c65a93acd176690bdaae13a3d7db092c8b98b392d4c6d0d03f176 |
| 27.2 | clang-2100.3.34.1 | c596e98dbf1218d8a989f66ba638ac8ed57fac3e3719ccdf1b8605544a5ba4e2 |

`as` is identical in all of them (06770364...). The previously pinned clang
(cb5fd7fc..., Xcode 27.2 beta 2) is gone.

The existing capture diagnostic (run 40425119253) probed every installed
Xcode with real external assembly, fresh shared assembly, debug-text
expansion and debug-text replay. All nine entries returned 0 for all four
probes (`capture-selection.json`, `commands.json`), and the eight integrated
capture tests passed (`integrated-capture.log`).

I pinned Xcode 27.2, the release successor of the beta that was qualified:
`scripts/ci_select_capture_clang.py` now names `Xcode_27.2.app`, clang
c596e98d... and banner `clang-2100.3.34.1`. The SDK pin (CLT 27.0) is unchanged.

Not yet shown: a full hosted retirement run through the new pin. The
`xcode-27` legs stay `continue-on-error` until one passes.

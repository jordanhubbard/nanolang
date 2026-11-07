# My refreshed Darwin capture selection

My previous pinned clang executable refuses on the current xcode-27 image before
any qualification corpus executes. I retain that failure under #1049; I do not
replace a tool pin based on its version label alone.

[My actual diagnostic run](https://github.com/jordanhubbard/nanolang/actions/runs/42382072384)
at `8c9584231` passes external assembly, fresh shared linking, debug expansion and
debug replay for the selected Xcode 27.2.0 installation. All eight integrated
source-capture methods pass in 518.363 seconds. My selected compiler resolves to
Xcode_27.2.app, reports clang-2100.4.1.1, and has SHA-256
`04988adaddb282c8c074a1a55f8221452f18f64df3ca0006135f62031710377b`.
The selected assembler retains SHA-256
`06770364632b454317238e79d670c094e27301c7322c3a488323bd42e4d0bcbc`.
The captured cc1as command names target SDK version 27.2 from the selected
Xcode SDK path. I select that tested path rather than the former CLT SDK.

I retain the selected command reports, exact backend hashes and complete
integrated test log here. `receipt.json` records the checksum-verified local
archive of every downloaded diagnostic artifact, including objects and all
installed-tool probes. This receipt records observations; it is not a native
lifecycle or release admission receipt.

I update the selector only after this passing diagnostic. Its hash, path,
SDK version, compiler report and changed-tool refusal controls remain active.
Six local fixture/workflow/wrapper controls pass; the real pinned-Xcode SDK
control is skipped because that installation is absent on this local host.
The next hosted selector and full Linux/Darwin qualification remain required.

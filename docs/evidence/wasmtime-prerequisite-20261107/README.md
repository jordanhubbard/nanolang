# My Wasmtime test prerequisite

I pin Wasmtime 50.0.1 and all four official Linux/macOS x86-64/ARM64 archive
SHA-256 values in `.github/containers/instrumented/install_wasmtime.py`.
The digests come from the [v50.0.1 release assets](https://github.com/bytecodealliance/wasmtime/releases/tag/v50.0.1).
I read only the checked regular executable member, check its version before
atomic replacement, and preserve an existing runtime on refusal. My test image
installs the runtime; developers and retained launchers can provision an explicit
prefix with `make install-test-wasmtime` before resolving execution inputs.

## My retained observations

- My two fixture methods pass, covering missing-runtime guidance and valid,
  checksum-mismatched, symlink and wrong-version installation. Refusals preserve
  the previous output and clean staged executables.
- The actual checksum-pinned Darwin ARM installer reports version 50.0.1.
- In an existing Linux ARM Python 3.13 container, the actual checksum-pinned
  installer succeeds, reports version 50.0.1 and executes a Wasm `nano_entry`
  returning zero. This is a Linux ARM installer/engine smoke, not a build of the
  complete Ubuntu instrumented image or x86-64 qualification.
- With installed LLVM on PATH and the installed OpenSSL library directory in
  LIBRARY_PATH, `make test-managed-record-eligibility` passes all 16 unchanged
  methods in 8.087 seconds, including actual generated-Wasm execution.
- All 15 existing workflow/container controls pass. My fast gate passes 223
  Clang and 223 GCC compilations, the installer controls and bootstrap-tool control.

I retain the first failure where LLVM `opt` was absent from PATH, then the fixture
link failure without OpenSSL's library path. After those environment corrections,
`test-ordinary-record-authority` still fails at its old ordinary `OWN_PACK`
admission assertion, before any Wasm execution. I keep that existing #1054
contract mismatch open and preserve its assertions in this dependency repair.

My first Linux installer attempt rejects the official executable at a 64 MiB
bound. I correct that finite executable bound to 256 MiB. My read-only container
smoke then needs an executable installation tmpfs and writable engine cache;
with those explicit owned paths it passes without changing product assertions.
I do not claim that the complete sanitizer image or retained receipt is qualified.

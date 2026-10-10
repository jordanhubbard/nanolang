# My opt-in read-text execution

I lower exact empty-namespace `file_read`, `vm_file_read`, and
`nl_os_file_read` imports with one STRING argument and one STRING result through
`nvm2llvm --portable-read-text`. My default closed profiles still refuse imports.
I check the operand tag before invoking the host, borrow the rooted input, and
copy the returned bytes into an independently owned managed string. Declaration
validation is not a proof of operand types; my emitted runtime checks remain
necessary.

For native LLVM, I link the generated IR with `libnano_portable_read.a`.
My embedding creates an explicit allowlisted `NprFileHost`, binds
`{npr_file_read, host}` through `npr_module_bind`, and calls `nano_try_entry`.
The high 32 result bits hold my managed status; the low 32 bits hold the program
result. `npr_module_host_status` reports a separate host failure. I deny calls
without a binding. I permit revocation with `npr_module_bind(NULL)` only while
inactive. I require the embedding to keep its callback/context alive until
revocation or disposal; these trusted native APIs are serialized, not a sandbox.

For Wasm, I use `nvm2wasm --portable-read-text`. I compile the adapter with
`NANO_WASM_CLANG` (default `clang`), lower IR with `NANO_LLC`, and link with
`NANO_WASM_LD`. I allow only `npr_wasm_host_read_text` to remain unresolved.
The resulting import is `nanolang_host_v1.read_text`, with five i32 arguments
and an i32 result. I retain the adapter's bounded offset ABI and its 2 MiB initial,
64 MiB maximum memory envelope. My Node and Wasmtime embeddings take an explicit
copied path allowlist; the Wasm file contains no filesystem grant.

`make -f Makefile.gnu install-portable-read-runtime PREFIX=...` installs my native
archive, public headers, translator binaries, Wasm adapter sources and trusted
Node/Python embeddings. I resolve installed adapter sources relative to the
translator, so relocation does not require the original source checkout.

I test generated native LLVM and Node/Wasm reads, denied and revoked access,
repeated entry, nested-entry refusal, disposal, malformed callback results,
wrong operand tags, native allocation-prefix recovery, retained unused imports, exact import refusals and prior-output preservation in
`tests/test_portable_read_execution.py`. Its source fixture also runs through
NanoVM and C AOT, with optional fresh self-hosted producers selected through
`NANO_PORTABLE_DRIVER_MODULE` and `NANO_PORTABLE_DRIVER_NATIVE`. I have checked relocated native and Node execution locally.

This is my first executable portable binding. Full generated Wasmtime execution,
Wasm allocation-failure qualification, the full source boundary corpus and exact-candidate
platform gates remain open. Byte/aggregate results and remaining compiler host
capabilities remain required by my 5.1 scope.

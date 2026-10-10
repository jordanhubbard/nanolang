# My portable read execution checkpoint

I connect exact read-text declarations to opt-in LLVM/Wasm execution. My
[execution log](execution.log) records four integration methods: actual source
through NanoVM/C/LLVM/Node, native binding and malformed callback handling,
Wasm permission and wrong-tag guards, and exact-import/default refusals with
preserved output. My [regression log](regression.log) records 26 existing scalar
and managed-string methods. My [declaration log](declaration.log) retains the
private descriptive-query mutation and allocation checks; those checks do not
execute bytecode or qualify allocation failure in the new execution bridge.

I ran on Darwin with Homebrew LLVM. Execution used `NANO_NATIVE_CLANG` and
`NANO_WASM_CLANG` set to `/opt/homebrew/opt/llvm/bin/clang`, `NANO_LLC` set to
`/opt/homebrew/opt/llvm/bin/llc`, and `/opt/homebrew/bin/wasm-ld`.
I separately installed with `install-portable-read-runtime`, renamed the prefix,
and ran native and Node read programs against the relocated archive/headers,
translator and adapter sources. Both read the allowed six-byte file.

My [execution contract](../../NANOISA_PORTABLE_READ_TEXT_EXECUTION.md) records
usage and remaining acceptance work. I do not claim the broad 5.1 portable host
capability requirement complete. GitHub issue creation failed with an API
connection error; parent #976 remains my existing task reference.

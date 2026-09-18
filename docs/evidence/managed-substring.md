# My matched managed substring evidence

I implement bounded child `task_2d21632d4d5241299c5a0e9967948efd` after merged
VM prerequisite PR640 (`7cc0e9f8`). My contract preceded code in `036e4e72`.
I retain my broad managed-runtime parent and full LLVM/Wasm coverage obligations.

My consuming helper checks remaining byte length, copies before release or
replacement of the descriptor table, consumes one input owner on every path,
and leaves output unchanged on failure. The emitted frame tracks three operands:
source transfers to that helper; both indices are released through normal
consumed-operand cleanup. Integer indices truncate to uint32; other tags supply
zero. Only my managed closed profile gains STR_SUBSTR.

On Linux ARM64, my frozen production source passed 54 integrated methods in
67.091 seconds (`/tmp/nanolang-managed-substring-integrated-final.log`): runtime
core, emitted managed strings, ordinary LLVM, Wasm, scalar globals and generic
comparisons. The new controls include embedded NUL and empty slices, clipped
ranges, integer truncation, non-integer index aliases, explicitly typed helper
calls, repeated globals, native allocation-error recovery, and VM/native/Wasm
type-error cleanup. Emitted native functions are instrumented using the existing
explicit LLVM ASan pass before linking the sanitizer harness; leak detection is
active. Wasm tests check zero imports and actual Node/Wasmtime execution.

The runtime core additionally checks byte and descriptor allocation failures on
a full table, retained source aliases, unchanged output/table, successful growth
and reclamation on native and Wasm. I passed package validation (two methods)
and 14 shared profile cases; the profile probe now reports scalar, literal and
managed decisions separately. Logs: `/tmp/nanolang-managed-substring-prereqs.log`
and `/tmp/nanolang-managed-substring-profiles-final.log`.

My first broad test attempt lacked fresh-worktree nvm2c and scalar-global test
binaries. I built those prerequisites before the final run. My initial new helper
fixture mixed parameter tags across calls; I corrected it to two explicit
signatures, preserving parameter checks. I replaced obsolete substring-refusal
fixtures with still-unsupported STR_CONTAINS controls; conversions and reserved
entry refusal remain tested. Historical failed artifacts remain unexecuted.
I claim no Darwin acceptance or completion of portable string conversions.

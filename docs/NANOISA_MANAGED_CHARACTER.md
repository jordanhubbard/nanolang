# My managed character-byte access contract

I implement child `task_32b265f80d604826adc9ac29716598ab` after the reviewed VM
operand-lifetime correction62caf. I admit STR_CHAR_AT only to my managed
LLVM/Wasm profile; ordinary scalar and literal profile refusals remain.

My operation returns an integer unsigned stored byte, not a character string
or Unicode codepoint. An integer index is interpreted as signed64: negative
or out-of-range indices return -1. A non-integer index uses zero, matching my
existing low-level VM behavior. Source-language builtin typing is separate.
NUL and high bytes retain values0 and128..255; empty input returns -1.

My private helper borrows the source handle, accepts index bits plus an
explicit integer-tag flag, and writes its output only on success. It validates
its view, excludes negative indices before comparing widened unsigned bounds,
and indexes only within the stored length. It allocates nothing, changes no
reference count, and leaves the output untouched on invalid arguments.

My LLVM wrapper validates the source tag and reports errors through the
existing first-error latch. Both popped operands remain ordinary FrameOutput
owners and are released exactly once after the helper reads them, including
when they refer to the same dynamic string. No operand is transferred to the
borrowed helper. Failure then follows the existing frame cleanup protocol.

I require actual VM/native LLVM/import-free Node/Wasmtime execution over
empty/NUL/high-byte inputs, signed limits, exact range edges and non-integer
fallback. Literal and dynamic values, aliases, calls, globals, repeated entry
and disposal must agree. Native/Wasm core controls disable allocation while
checking byte results and unchanged reference counts. Corrected ordinary
source-type refusal must release transient operands while retaining committed
globals; old output preservation and profile controls remain required.

This is not a source-frontend expansion. Full managed-runtime51da,
Darwin managed-sanitizer7ba, historical evaluator791a, aggregate/cycle488 and
host-linkage2d2 obligations remain open.

# My string literal encoding

My parser retains source escape spellings. I first decode the C seed's string
contract, including preserved unknown escapes and its first-NUL string boundary.
I intern those decoded strings, then encode quotes, backslashes and controls for
my assembler. Other UTF-8 bytes retain their spelling. Hexadecimal assembly
escapes cover other control bytes without changing source escape semantics.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline comparisons
and 18 focused Python cases. Fourteen new bytecode/function checks match the C
seed. Both emitted modules verify and execute quote/backslash, newline, carriage
return, tab, UTF-8 bytes, unknown source escapes and the C-string NUL-prefix
behavior in NanoVM and strict native C11. Repeated assembly emission is byte
identical. Shadows also check hexadecimal encoding of control byte 1 and DEL.

Raw `src_nano/nanoc_v06.nano` emission now passes the literal boundary and first
refuses the imported `parser_decode_import_path` definition. That raw-source
path does not merge imports; this result does not identify a missing builtin.
The canonical frontend's merged-parser route is the next full-closure probe.
Task `task_98ec870924bf4b82b9ba5e2591576488` records this slice. Full compiler
emission and bytecode bootstrap equality remain open.

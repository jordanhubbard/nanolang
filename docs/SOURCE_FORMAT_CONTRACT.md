# My runtime-template format contract

I preserve the existing `vm_format` substitution rule for source `format`.
I read my template and evaluate every remaining argument once in source order,
converting each with raw `CAST_STRING` before scanning. Extra arguments still
execute. Raw float conversion retains its existing spelling; source
`float_to_string`'s decimal suffix is a separate contract.

I substitute the next converted argument for `%s`, `%d`, `%f` or `%g` while
arguments remain. The letter does not select a conversion type. I copy every
other byte literally, including unknown markers, a trailing percent and
recognized markers after arguments run out. Thus `%%s` with `"value"` becomes
`%value`; I do not add printf escaping, width or precision rules. Unicode bytes
and empty strings retain their current values.

I require a string template and at least one argument. Bound function values named `format` retain their own signatures. My C
frontend keeps its existing refusal of a direct function definition named
`format`. I test a legal wrapper that directly returns formatted text. The existing variadic
operand signature does not imply native acceptance for every raw value:
`CAST_STRING` keeps its consumer's checked capability boundary.

Before implementation I retain the complete original `tests/test_format.nano`
through C-frontend NanoISA and NanoVM, then measure native translation. My
self-hosted producer currently lacks this operation. The proposed source
lowering uses existing checked string/array instructions, saves converted
arguments, scans template bytes and concatenates literal spans at substitutions.
It does not widen a raw foreign-array import ABI. Acceptance requires both
frontends, both fresh stages, sanitized native execution, original malformed
and prior-output controls, complete translator/units and integrated platforms.

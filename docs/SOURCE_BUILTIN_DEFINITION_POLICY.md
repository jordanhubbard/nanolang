# My reserved function-definition policy

I share my existing C definition policy through
`spec/reserved_builtin_names.json`. It contains 142 effective names. My former
C table contained 144 entries, with map and reduce excluded by its array-operation
guard. I retain that same effective set; I do not reserve every runtime builtin.
My source-baseline evidence records the original table, guard and source pin.

I reject a non-extern function definition with one of these exact names before
emission. My C and Nano source checkers use generated tables from the same
specification. `python3 scripts/gen_reserved_builtins.py --check` refuses stale
tables. I test reserved print/println/format/min definitions, legal ordinary
str_trim/exp/map/reduce names, extern declarations and bound function values.
Module-qualified identities retain their existing binding rules.

A definition refusal does not forbid a local function value named print or
format. An accepted extern declaration does not establish a native host ABI.
I retain the existing consumer capability checks and prior-output assertions.
My actual source checker component and original source suites exercise these
boundaries; fresh self-hosted and integrated platform qualification remain
separate requirements.

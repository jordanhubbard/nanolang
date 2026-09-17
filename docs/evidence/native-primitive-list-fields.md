# I preserve primitive List fields in native records

My native field-token helper recognized `List<Name>` only when `Name` was an
identifier token. `int` and `string` are keyword tokens. A record containing
`List<int>` therefore emitted `nl_List values` and failed C compilation during
shadow execution. I reproduced that failure with Stage 2 at `91709f44`; the
session log is `/tmp/nanolang-primitive-list-field-baseline.log`.

I now accept my supported integer and string List keyword elements alongside
named elements. I retain `List_int*` and `List_string*`, matching their
anonymous runtime typedefs. I do not rewrite them to unrelated struct tags or
add support for other primitive List element types.

My helper shadow checks each emitted spelling and the next-token cursor for
`List<int>`, `List<string>`, and the existing `List<LexerToken>` case. My native
nominal regression constructs integer/string Lists inside records and unions,
mutates them through a record field, reads their values through a function,
and checks the original List alias and a trailing scalar field. Existing
union-only primitive List controls and named-list forward layout controls
remain in that regression.

Validation: fresh `make bootstrap` passes. All ten methods in
`python3 -m unittest -v tests.test_native_nominal_order` pass against both
self-hosted stages (24 compile decisions, including four expected cyclic-layout
refusals with prior-output preservation). The new integer/string record
fixtures also compile and run with the C seed. Logs are retained under
`/tmp/nanolang-primitive-list-fields-{bootstrap,native,cseed}.log`.

After integration with the nested generic identity repair at `d1827351`,
I rebuilt all bootstrap stages and reran nominal plus nested-generic checks:
all eleven methods passed in 76.807 seconds (27 compiler decisions across the
selected stages). The integrated logs are
`/tmp/nanolang-primitive-list-fields-integrated-{bootstrap,native}.log`.

# Mixed native nominal layouts

I collect record and union definitions with their exact emitted field types,
then emit a definition only after its by-value dependencies. Stable scanning
keeps independent definitions deterministic. I reuse existing generic union
instantiation and source-owner naming; this change does not relax ownership
checking or claim new generic-resource support.

A union containing an ordinary record previously failed native shadow C
compilation with an unknown record type. Reversing the global order would break
records containing unions. My ordering supports both directions and alternating
chains. An unresolved by-value cycle produces a first-person compile error;
I do not retry indefinitely or replace an existing output artifact.

Array and List handles do not require complete element layouts. I spell List
fields using their existing named C struct tags so a pointer-backed recursive
field does not require the later specialization typedef.

My focused cases cover record payload execution, both dependency directions,
an alternating chain, reverse declaration order, array/List recursive handles,
and record/mixed cycle rejection with prior-output preservation.

Independent review caught that my integer/string runtime lists use anonymous
struct typedefs. I preserve those exact typedefs and test both through union
construction and runtime list operations. The separate pre-existing record
`List<int>` token-parser defect is tracked as
`task_f0a59def9afe4a08ae63c55d4ba22109`.

Fresh native bootstrap passes. Nine test methods (ten source cases) pass in both
self-hosted stages in 56.802 seconds. Logs:
`/tmp/nanolang-nominal-order-reviewed-bootstrap.log` and
`/tmp/nanolang-nominal-order-reviewed-paired.log`. The ordinary prior compiler
failure is in `/tmp/nanolang-native-nominal-order-baseline.log`.

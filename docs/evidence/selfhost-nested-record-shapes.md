# My finite nested compiler records

On 2026-09-16 I extended supported record shapes to finite nested records and
typed-list fields. The measured compiler blocker was
`List<CompilerDiagnostic>`, whose location is a nested source-location record.
I validate each field shape with an ancestry path, reject cycles and unknown
types, and copy that path before descending so repeated sibling types remain
valid. Integer, boolean, string, scalar-array and typed-list fields retain
their existing representations.

Nested construction uses the same aggregate packing as other records. Field
expression types retain nominal identity; mismatched nested values refuse
output. Existing chained projection, parameters and results preserve nested
values and list identity. This does not add recursive type support or new
foreign ABI signatures.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 28 integration methods.
The nested diagnostic/report fixture adds 10 C-seed bytecode comparisons and
executes under NanoVM and strict C11 AOT, including a returned record with a
mutable record list and two sibling fields of the same nested type. The former
nested-result refusal is now a positive VM/AOT test. Five direct, list-mediated
or mutual cycles and mismatched nested-value programs refuse output.

Full compiler emission and matching Stage 1/Stage 2 bytecode remain acceptance
requirements beyond this shape slice.

A fresh actual canonical-driver probe now first refuses
`unsupported local type array<Symbol>`. That compiler record includes nested `NSType`
records and an array of `NSType`; `NSType.kind` is an integer field. The next
record-array continuation is
recorded separately. The probe publishes no compiler module.

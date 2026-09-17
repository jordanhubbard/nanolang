# My arrays of finite records

On 2026-09-16 I added `array<Record>` for supported finite record shapes.
Empty and computed record literals carry tag 8; element types survive locals,
fields, parameters, results and direct projection. Nested record-array fields
use the same finite shape validation and reject cycles.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 29 integration methods.
The record-array fixture adds 12 C-seed bytecode comparisons and executes under
NanoVM and strict C11 AOT. It represents the measured `Symbol` requirement with
a nested location and `array<NSType>` field, checks empty record-array returns
and fields, and verifies shared identity after pushing a record through a field.
`NSType` is a record with integer/string fields, not an enum; I corrected the
earlier continuation description after inspecting the generated contracts.

The C seed currently rejects direct record-element field projection without an
intermediate declared local; task `task_8ddcdb5c824e4a8eb6cc0e2d1bc9ebe3`
tracks that nominal metadata repair. My cross-compiler fixture uses explicit
`Symbol` locals. A separate emitted-source variant verifies direct projection
under VM/AOT, without claiming C-seed acceptance of that form.

Source enum member metadata remains separate under
`task_e66b50097fe343e3b78e6b750a5c7315`. Full compiler emission and the
matching bytecode bootstrap gate remain open.

A fresh actual canonical compiler probe now first refuses
`unsupported local type CollectResult`. Its remaining unsupported field is
`HashMap<string,int>`, recorded as the next map-bearing record slice. No compiler
module is published by the probe.

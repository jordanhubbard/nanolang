# My optional record fields across array boundaries

I apply the existing record-field conversion to record-array arguments,
returns and tail returns. A present string can feed tagged string-or-missing
storage without equating its source shape to an optional shape. I constrain
the optional payload to the source string shape instead.

I merge record element field facts with the same string-to-tagged rule used
for direct records. Map key/value facts retain their exact merge rule. I do
not clone arrays or rewrite their records at these boundaries; my existing
aggregate extraction checks and boxes the actual stored field tag.

My argument tests vary caller/callee order and call order, check missing and
present tagged values, and then read the original caller array as strings.
My return tests exercise present and missing branches through ordinary and
tail returns, and reject an optional integer payload mixed with a string.
These are boundary tests, not a claim of complete mutable-array inference.

Normal and fresh ASan/UBSan suites pass 1,523 AOT and 994 shape checks each.
Sanitizer leak detection remains disabled; I do not infer leak freedom.

I run `make -j1 test-nvm2c`, `make test-nvm2c-sanitizers` and
`make -j1 test-one-ir-compiler`. The full compiler clears the former argument
and return field conflicts but still fails at function 163,
`parser_store_union_construct`, during `AGG_PACK` classification. The focused
source empty-array return fixture passes. Full compiler acceptance remains
open under the roadmap and MAC task `task_a478c928daf246129a8b8fb82da27fd6`.

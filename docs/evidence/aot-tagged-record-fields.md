# Tagged lookup values in records

I store a tagged record field's runtime value tag separately from its native
representation tag. Its integer and string payloads use the record's existing
scalar slots. Packing and extracting the field copies the value without
unboxing it. Missing remains void, rather than becoming zero or an empty string.

Record value copies, nested snapshots and record-array elements carry the
runtime tag with the payload. Fetched string storage remains owned by the
entry-lifetime lookup registry, so deleting the source map entry does not
invalidate a record field. Early reclamation remains a separate requirement;
these tests do not establish bounded memory retention.

The new nested-array test exposed an inference-order defect in `TYPE_CHECK`.
An unresolved field projection can acquire its optional shape from a later
function. I now defer that unresolved case until emission, after all final
shape constraints are collected. The emitter requires a tagged representation;
it does not silently box a still-ordinary operand to satisfy `TYPE_CHECK`.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,142
AOT checks and 990 shape checks. The sanitizer driver verifies fresh ASan/UBSan
translator and graph objects; opcode case-parity and driver tests pass.
`git diff --check` passes. My new
execution tests cover integer and string lookup fields, missing and present
values, nested records returned through a forwarding tail call, returned
record arrays, extraction and tag inspection after deleting the original map
entry. A negative regression keeps mixed ordinary/optional record fields
rejected until compatible field conversion is implemented.

Full compiler acceptance advances from unsupported packing to a result-field
conflict in function 282 at offset 120: result field 1 has `const char *` and
`nmap_value` representations. That is remaining work. I must join compatible
field representations without erasing tags or treating a plain string's shape
as an optional value everywhere it is used. Tagged scalar returns also remain
unfinished.

MAC still refuses my claim with `agent_status_unavailable`; the parent task
remains open.

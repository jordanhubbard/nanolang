# Canonical DEBUG transport

I preserve accepted DEBUG source-map records through canonical text under MAC
`task_1466d452d48c4a558c3be8a51765dd8f`. These facts are advisory. They do not
supply executable types, ownership authority or source correctness evidence.

I add `.debug <absolute-bytecode-offset> <source-line> <source-column>` outside
functions. Each operand is an unsigned 32-bit integer, matching my in-memory
record. I preserve order, duplicate offsets and zero values exactly; I do not
sort, infer or reinterpret records. Source-file identity remains the existing
ordered advisory metadata convention. An explicit `.flag debug_info` can retain
an empty DEBUG section in canonical v2 transport. Stripping debug information
still removes the section and source-file advisory entries.

I emit one directive per record. Reassembly must either retain every record or
report an allocation error, never silently publish a truncated debug table.
Existing source producers may ignore the checked append result as before; this
transport slice checks failure in assembly and v2 conversion. The v2 bridge
uses its existing INDEX_RANGE allocation-error convention; I add no new error code. Changing general
source-producer allocation policy is outside this contract.

I retain unknown advisory metadata keys byte-for-byte through the existing
transport. Unknown wire section types remain rejected by the v2 loader. I add
no opaque section storage, feature bit, execution eligibility or authority.

I require full canonical v2 byte equality for actual source artifacts and for
assembly fixtures carrying debug entries plus unknown advisory metadata. I test
multiple functions, duplicate/unordered offsets, zero/max operand values, empty
DEBUG presence, stripping, malformed directive diagnostics and allocation
failure cleanup. I retain ordinary v1 debug roundtrip compatibility. Canonical
source bytes, VM execution and generated native execution must remain unchanged.
Other unrelated unrepresented wire fields are not claimed complete by this slice.

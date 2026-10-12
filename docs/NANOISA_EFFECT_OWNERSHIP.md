# My effect ownership declarations

I retain operation value-transfer signatures in ownership format version 3, extension kind 4 (`EFFECTS`), revision 1. This is a declaration prerequisite for resource-bearing effects. My affine instruction verifier still refuses handler, perform and resume instructions; these declarations do not enable execution.

My extension payload uses little-endian integers:

| Field | Encoding |
| --- | --- |
| Operation count | u32, 1 through 256 |
| Operation key | u32 string-pool index |
| Parameter count | u16, 0 through 256 |
| Reserved | u16 zero |
| Result | One value descriptor |
| Parameters | Parameter-count value descriptors in declaration order |

A value descriptor contains a u8 value tag, u8 zero mode, u16 zero reserved word, and u32 layout index. Scalars and strings use `NO_INDEX`. A void result also uses `NO_INDEX`; a void parameter is invalid. Records and unions require an exact retained layout of the matching kind with COMPLETE authority. Resource flags propagate through existing aggregate declaration validation. Collection and borrowed-continuation signatures require additional contracts and are not admitted by this revision.

Rows have strictly increasing key indexes. Keys must be nonempty strings without embedded zero bytes, and distinct indexes cannot repeat the same string identity. Lookup matches counted string contents, as operation dispatch matches names. Unsupported revisions, extension kinds, flags, tags, layouts, counts, trailing data and truncations fail validation.

`nvm_ownership_effect` validates the complete ownership payload before returning one signature. A missing operation or any invalid extension leaves the caller's output unchanged. An unrelated malformed global or union extension cannot be bypassed by querying an effect. Binary and explicit unverified assembly transport preserve the declarations; ordinary assembly still requires executable verification.

My [contract evidence](evidence/effect-ownership-contracts-20261011/README.md) records positive resource and nested-union signatures, capacity boundaries, mutation/truncation refusals, unchanged outputs, and the existing executable refusal. Producer emission, affine handler-entry states, suspended-owner obligations, resumption transfer and runtime cleanup remain required under #976.

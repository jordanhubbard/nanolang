# My retained layout transport

I continue task_a794de400f5b400bbc11a6d251e7bd94 under affine IR parent ed702.
On the PR532 source, an ordinary one-field `Handle` layout loses both its
field and name through `NvmV2Module -> NvmModule -> NvmV2Module`:

```text
before fields=1 name=0; after fields=0 name=4294967295
```

I now retain owned canonical LAYOUTS bytes in the execution module. I copy
fields and names without reordering layout indices. A per-kind executable
definition index remains its ordinal among layouts of that kind; this does
not replace it with a global layout index. Future frontend emission must map
its own declaration order into this retained order explicitly.

My required v2 feature bit 7 marks retained layouts. Complete named/fielded
layout tables require that bit; old readers reject it. A producer can set it
explicitly for complete zero-field layouts as well. Count-only legacy tables
remain compatible and do not gain an authority claim merely by conversion.
V1 serialization refuses retained layouts rather than erasing them.

Canonical `.layouts "hex"` chunks preserve the existing LAYOUTS payload.
I validate the complete reconstructed table against type counts and string
indices before admitting the assembled module. The table still supplies
layout facts only: resource status, function borrow modes, live ownership,
reference provenance and lifetime verification are separate work.

I passed 46 focused checks, including source-teardown independence in both
bridge directions, exact byte reconstruction, same-shaped nominal records,
nested fields, invalid replacement preserving the old table, and legacy
count-only compatibility. The executable fixture adds three file-output
checks and runs one artifact through NanoVM and nvm2c; both print `42` after
a nested record field read. This is an ordinary value program, not a borrowed
call. The focused bridge/layout/storage code and test also pass ASan/UBSan.

My existing format, layout, module, bridge, 20 end-to-end and 2,691 NanoISA
checks pass, along with the 75 reference-place checks. No compiler frontend
or `.nano` compiler source changed. I do not claim a compiler bootstrap or
complete ownership acceptance from these tests. Both NanoISA borrow refusals
and my release publication hold remain.

# My declared purity metadata

I preserve frontend `is_pure` declarations in `nano.purity.v1` through the C
and selfhosted NanoISA producers. I retain the declaration separately from
inferred effects and verified execution authority. My
[metadata contract](../../NANOISA_ADVISORY_METADATA.md) defines indices, exact
bytes, unknown/duplicate/malformed lookup behavior and rewrite obligations.

My source matrix covers declared and undeclared purity, generic specialization,
preserved refusal of unconstrained generic pure inputs, imported declarations,
same-spelling functions in direct and transitive imports, named nested functions
and anonymous captures. I require every emitted function in these fixtures to
have exactly one record. I compare original-wire canonical roundtrips and run
VM/native outputs. Stripped, duplicated, unknown-version and forged records
retain the same VM output and byte-identical generated C. Generated programs
run under ASan/UBSan with leak detection.

This matrix exposed duplicate C assembly symbols for same-spelling imported
functions. `collision/before.nasm` retains two distinct function indices named
`value`; canonical assembly refuses the duplicate. I separate wire symbols
from source lookup aliases for direct and transitive imports. This fix does
not rename source declarations or merge their purity records.

My C codegen serialization regression now requires v1 refusal for the new
metadata-bearing module, then checks declaration retention and execution in v2.
I preserve v1 support for ordinary metadata-free modules in the existing codec
suite.

My initial seed build reaches the ten-second supervised shadow deadline without
an assertion report. I preserve that trace. The same compiler sources complete
all 1,140 shadows with zero failures under a 60-second deadline; summed shadow
execution is 9.327 seconds, excluding surrounding overhead. I keep the deadline
failure distinct from a demonstrated assertion or compiler defect.

I pass 359 advisory codec/lookup checks, 22 allocation-failure boundaries,
90 codegen checks and 210 canonical roundtrip checks. I instrument the changed
codec and assembler directly with ASan/UBSan and leak detection.

My fresh compiler passes all eight product-audit steps: paired source checks,
self-emission, module verification, wrapper-free C translation, strict native
linking, CLI execution, resulting-product metadata checks and the five-method
purity-contract suite. The eight source methods pass through the C producer,
fresh seed and resulting compiler. All five pinned production sources remain
unchanged during the audit; the manifest retains their hashes and product hashes.
Self-emission takes 314.4 seconds on this host.

Full #976 remains open: declared
purity is only one frontend fact. Inferred effects, import annotations, affine,
generic and exhaustiveness facts, fact-aware linking/reconstruction, complete
producer coverage and exact-candidate platform qualification remain required.

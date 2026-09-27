# My selected owned-union verification checkpoint

I add local-normalized format-4 union construction, whole-value transfer and
selected extraction. Live resource unions cannot be overwritten or forgotten
at joins. Different live variants join to unknown selection. Whole stack moves
clear local selection; nested extraction does not invent an inner variant.
Record operations cannot interpret flattened union fields as a record.

I encode `OWN_UNPACK_VARIANT` as `97 local:u16 variant:u16 count:u16`, with
little-endian operands. My private bytecode analysis checks exact count, live
selection, nominal child layouts and resource tokens. Rooted observations
refine their source on a successful match edge and prevent moving or replacing
that owner until observation ends. Dropping the observation does not consume
its owner. Wrong variants/counts, duplicate consumption, stale selection,
mismatched joins, scalar-for-owner construction and dropped payload owners
remain refusals. Exact bytes and every truncated encoding are tested.

Public execution remains refused. I have not implemented VM/native extraction
or enabled format-4 executable admission. The schema explicitly classifies this
new opcode with other decoded, non-executing instructions. Portable native
planning and the owned-transfer detector recognize the new boundary.

My retained gates pass:

- 33 schema consistency tests; 2,999 ISA checks; 98 verifier tests.
- 1,500 shape and 2,428 complete translator assertions.
- Affine state: 425 normal and 457 allocation-injection checks.
- Affine bytecode: 793 normal and 1,155 allocation-injection checks, including
  every allocation in the new selected-owner analysis route.
- Existing scalar-union runtime, owned-transfer and ownership transport gates.

The final instrumented state and bytecode runs pass the same four assertion
counts with ASan/UBSan, halt-on-error and leak detection. The script instruments
state, bytecode, ownership contracts, verifier and both test fixtures; other
dependency objects are ordinary builds. The backend gate preceded a final
change from scalar-only diagnostic wording to union wording; final bytecode
and instrumented checks requalify that edit. This is not a fully instrumented
runtime or a new canonical bootstrap.

`logs.json` seals the uncompressed terminal bytes. VM/native selected transfer,
trap/early-return cleanup, canonical emitter identity and moves, and the full
unchanged generic ownership acceptance matrix remain required for PR #522.

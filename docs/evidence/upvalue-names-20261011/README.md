# Captured upvalue names

I preserve closure capture names as `nano.upvalue.v1`, independent of local-slot
names. I reuse the 16-byte binding header and borrowed view API shape, validate
against upvalue counts, and require full-function intervals. Duplicate capture
indices are unusable even for empty functions. I leave malformed/unknown advisory
records transportable and keep VM admission independent of optional names.

My assembler accepts `.upvalue_name`; the C producer publishes resolved capture
names and the selfhost producer emits them from its completed capture frames.
My source suite checks grandparent captures through intervening environments,
arrays, records, tuples, strings and callbacks. It keeps child locals and
upvalues separate, checks exact original-wire and second-cycle roundtrips,
and strips both naming tables while retaining VM results and identical native C.

I pass ten source methods through both the C producer and fresh selfhosted
seed with ASan/UBSan/leak checks. I pass 244 binding codec/lookup checks, also
with the changed codec and assembler directly instrumented; 90 bytecode tests;
210 canonical disassembly checks; and existing allocation-failure/recovery gates.
Duplicate and unknown capture tables also preserve VM results and byte-identical
native C in the source suite.

The metadata tests cover missing and unknown versions, duplicate entries,
invalid indices, reserved bits, invalid intervals, empty functions, unchanged
lookup output on invalid data, independent local/upvalue tables and malformed
metadata transport. I retain exact commands and outcomes in the accompanying
logs and compiler product manifest.

I do not claim outer-declaration identity, mutation rights, ownership facts or
live-stack locations from capture names. Effect-handler naming, remaining
producer forms, frontend facts, reconstruction and exact-candidate platform
qualification remain open under #976.

My fresh compiler product passes all seven audit steps: self-emission, module
verification, wrapper-free C translation, strict native linking, CLI execution,
ten sanitized metadata methods and 22 nominal/layout methods. All five pinned
production source hashes remain unchanged during the audit.

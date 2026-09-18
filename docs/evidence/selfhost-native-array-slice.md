# My native array_slice helper qualification

I qualify task_c9927df8c5674cf3bcf6ecdf7032f505 under my [reviewed contract](../SELFHOST_NATIVE_ARRAY_SLICE.md). Production630ae6da adds the missing self-hosted native helper definition; independent review confirmed all26 emitted C string lines match the existing C-seed helper byte for byte. Call lowering, source typing, VM slicing and C-seed behavior are unchanged.

My fresh original minimal source baseline passes C-seed compilation/execution, while the qualified781 Stage1/Stage2 tools report undefined `nl_array_slice` during native shadow linking. I retain those first logs and exact tool hashes. No failed artifact is executed.

Frozen1f988e91 passes actual bootstrap in250.036s, including Stage1/Stage2 hello and normal compiler shadows. All three focused methods pass48.573s (phase48.613s). Six source programs compile with normal selected shadows and execute through all three native compiler stages; five compare with verified NanoVirt/NanoVM execution. The original accepted reproducer now passes every native route. Other sources cover clipping, empty arrays, independent outer mutation, scalar/string/float/record leaves, retained nested identity and exactly one evaluation of each argument. I do not claim a new C argument order.

I extract the helper from the actual self-hosted runtime strings and compile it under strict C11 GCC and Clang with ASan/UBSan. Both executions cover all seven element kinds, signed extreme start/length bounds, null fallback, outer-copy independence, shallow nested/record references and exact negative-zero, subnormal and NaN payload bits. `gc_shutdown` qualifies normal cleanup; this child does not claim injected allocation-failure coverage.

I verify2136 source hashes unchanged before/after and current, plus both seven-tool inventories. Independent review inspected all55 retained command logs. My [manifest](selfhost-native-array-slice/reports.sha256.json) seals77 reports, including source fixtures, extracted helper, command logs, original baseline and successful gates. Compiler identities are recorded after qualification; the retained Clang wrapper selects GCC13 headers/libraries. Existing sanitizers and shadow deadlines remain intact.

This is Linux component qualification at the stated pin, not a combined-main bootstrap, Darwin acceptance, fixed point or full product release gate. Parent array/ownership obligations and PR522/publication remain separate.

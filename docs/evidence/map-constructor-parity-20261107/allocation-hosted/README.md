# My generated snapshot allocation checkpoint

At 7a320e0851bcdb9a8dd023d9159101757914279f, run 42419811888 passes all 37 original scalar-map, storage, lifetime, byte/collection-debt and optional-array methods with zero skips on both fresh Linux GCC and Clang. The complete recorded Git trees match the pinned commit; all assembler, VM and C AOT providers remain unchanged.

I measure allocations in my actual generated snapshot programs under forced collection at O0 and O2: INT/INT has 16 allocation positions, INT/STRING 19, STRING/INT 19, STRING/STRING 22 at each optimization. Each compiler family records 344 commands: 40 successful assembly, translation, compilation, baseline or fresh retry commands and 304 intentional production SIGABRT allocation refusals across transient and persistent denial sweeps. Success checks zero live pointers and bytes; refusal checks require my exact invariant diagnostic without sanitizer errors. Process termination does not establish recoverable allocation cleanup.

I retain original generated C, assembly, probe headers, allocation counts, command/status journals, complete suite logs and source/provider identity here. Full member inventories and archive SHA256 values locate every original binary and diagnostic in the complete private archives. Source constructor/field/fresh-stage and integrated platform acceptance remain open under #1083.

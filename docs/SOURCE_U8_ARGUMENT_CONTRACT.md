# My directed scalar call argument contract

I accept a U8 scalar at a declared INT source-function parameter and retain an explicit `CAST_INT` before invocation. I apply the same boundary to direct calls, bound function values and computed callees. I save a computed callee before evaluating its arguments and evaluate each argument once.

I preserve my existing U8 destination policy: literal range checks remain exact and computed INT values narrow modulo 256. I retain that conversion for indirect calls as well. I do not make U8 and INT generally equal, reinterpret `array<u8>` storage as `array<int>`, change generic binding identity or widen a foreign host descriptor. Existing host ABI guards remain in place.

My canonical frontend already converts direct numeric arguments. I extend its indirect argument emission to the saved signature's INT/U8 destinations. I leave other argument tags on their existing path.

I use independent direct/indirect and typed byte-element caller controls alongside the complete original corpus. These controls do not establish complete `std/binary` module or foreign-host acceptance; distinct module/host refusals remain recorded under #1081. I require actual checker production cases, VM/native execution, explicit conversion opcodes and unchanged negative prior-output assertions. Fresh self-hosted and integrated platform acceptance remain separate.

# My unsigned-byte scalar contract

I track this work under task_ad1c498801b34aa38d82f58a588f1d5a.

My ISA declares TAG_U8 as an unsigned byte and PUSH_U8 carries an unsigned
one-byte immediate. My VM already preserves that tag, compares equal bytes by
value, converts them to int without sign extension, and treats zero as false.
My source-language arithmetic table does not introduce a separate byte
arithmetic overload. I do not add one here.

I interpret same-U8 ordering as unsigned numeric ordering over 0 through 255.
CAST_FLOAT produces the exact binary64 representation of that value, matching
the existing numeric CAST_INT conversion. The current VM omissions (default
zero float and default equal ordering) do not implement that numeric contract.
I repair those two omissions before backend admission.

I then admit tagged byte values within the closed scalar backend profile:
constants, locals, direct call parameters/results, supported joins, TYPE_CHECK,
CAST_INT/FLOAT/BOOL and explicit CAST_INT followed by typed I64 comparisons.
My VM and C backend also implement same-U8 generic comparisons; LLVM/Wasm
continue to refuse generic comparison opcodes until task01d or a separate
static operand-proof prerequisite establishes their complete admitted contract. Typed I64/F64/BOOL operations keep
their exact tag checks. I do not add a CAST_U8 instruction, byte arithmetic,
heap or host byte transport, or a byte executable-entry ABI.

Mixed-tag arithmetic and comparison policy remains task_01d3e7ca2b6a47b0bef9504f2fd04006.
I preserve current VM mixed-tag observations and explicitly refuse profiles I
cannot match; I do not infer a numeric promotion rule from tag ordering.

My acceptance requires ordinary boundary values (0, 1, 127, 128, 254, 255),
all same-byte order relations, exact conversion/result tags, call/branch
transport, and previous-output preservation for unsupported profiles. I retain
separate VM, native C, LLVM and Wasm evidence and do not claim full backend
coverage from this scalar slice.

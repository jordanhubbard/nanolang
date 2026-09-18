# My managed primitive numeric formatting contract

I implement `task_b029d365b4334e7d9399e7230e37eeac` after checked VM formatting
and operand lifetime4ba/PR677. I admit STR_FROM_INT/FLOAT only to my managed
LLVM/Wasm profile, preserving scalar/literal-only refusals.

I normalize an operand to the operation's exact expected tag. STR_FROM_INT
retains bits only for TAG_INT; STR_FROM_FLOAT retains bits only for TAG_FLOAT.
Every other tag uses positive-zero bits with that expected tag. In particular,
an owned string becomes numeric zero text, not an identity string result.
I reuse the existing checked portable scalar formatter on that normalized
value. Integer decimal and binary64 %g agreement remains under my established
C-locale/default-rounding contract, including signed zero and nonfinite bytes.
I neither change the process locale nor claim arbitrary locale/rounding parity.

The normalized value owns no heap reference. My existing formatter produces
one fresh managed string owner or returns failure through the status latch.
The original popped operand remains a FrameOutput owner and is released once
after formatting, including allocation failure. Other locals, stack roots and
committed globals retain ordinary cleanup semantics. VM interning can reuse
equal strings; I do not claim matching allocation events or physical handles.

I require ordinary VM/native LLVM/import-free Node/Wasmtime numeric boundaries,
C-reference binary64 bytes, every admitted fallback tag, same-string aliases,
calls/globals/reentry and disposal. Emitted allocation failure must consume
transient input owners, retain committed aliases and permit recovery. Shared
profile and still-unsupported-operation output preservation remain checked.
The existing portable formatter core/reference suite remains an adjacent gate;
no new allocator or numeric formatter implementation is needed. Full runtime51da,
Darwin7ba and evaluator791a remain open.

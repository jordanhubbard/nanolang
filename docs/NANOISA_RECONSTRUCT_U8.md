# My exact `u8` reconstruction contract

I implement MAC `task_580d836e3666437ab1bccff3ff301244` as one bounded
extension of my structured scalar reconstruction. I admit `PUSH_U8`, exact
`u8` local slots, direct `u8` parameters and direct `u8` results. I preserve
the byte tag and the range 0 through 255; I do not silently reconstruct a byte
as an `int` merely because both fit in one host register.

I admit two explicit conversions. `CAST_INT` widens a byte to its unsigned
integer value. `CAST_BOOL` is false for zero and true for every other byte.
This does not admit generic `NOT`, `AND`, `OR`, arithmetic, or
`EQ`/`NE`/ordering for byte operands. Those operations retain their existing
checked refusals until their semantics receive separate contracts.

My retained-facts boundary must recognize exact byte signatures before the
region analyzer runs. Both emitters consume the same typed tree. C uses
`uint8_t`; Nano source uses `u8`. Byte constants and default local
initialization remain representable in both languages. An explicit widening
expression supplies `CAST_INT` and the comparison with zero used by
`CAST_BOOL`. Direct calls retain once-only argument and result snapshots.

I require endpoint constants 0, 1, 127, 128, 254 and 255, local stores and
loads, direct parameter/result calls, both structured branch outcomes and an
explicit conversion in a loop. From the same verified module I run NanoVM,
strict sanitized C, and reconstructed Nano source through the available
current compilers. I retain prior output for generic comparison, arithmetic,
logic, globals, imports, aggregates, ownership and wrong-signature refusals.
I record exact source and selected tool identities around the focused gate.

This extension does not admit a `u8` entry result: my executable entry remains
an arity-zero `int` function. It does not change NanoISA, NanoVM, `nvm2c`, LLVM
or Wasm semantics. Full high-level reconstruction and the v5.1.0 release gates
remain open.

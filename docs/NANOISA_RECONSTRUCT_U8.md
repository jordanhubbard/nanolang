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

## Checked-source prerequisite

My first focused reconstruction run established the existing
`task_b5d82f5ad53745e4896eafdf6112234d` boundary before publication: the C seed
accepts a numeric literal in an explicitly declared `u8` binding, assignment,
parameter or result context, while the self-hosted checker retains the
literal's unconstrained `int` type and refuses the same source. Merely hiding
that diagnostic would be wrong. My canonical NanoISA emitter also currently
emits every numeric literal as `PUSH_I64`, even when the checked destination is
`u8`.

I repair those two sides together. The self-hosted checker applies an exact
`u8` contextual hint only to integer literals used by a declared byte binding,
assignment, direct call parameter or result. The canonical emitter consumes
that same declared context, requires the literal to be in the closed range
0 through 255 and emits `PUSH_U8`. It does not coerce computed integers,
negative literals, floats or unresolved expressions. Direct `u8` values keep
their existing exact argument and return checks. Fresh C-seed, Stage 1 and
Stage 2 source controls must observe the byte tag; a checked out-of-range
literal must retain prior output.

This extension does not admit a `u8` entry result: my executable entry remains
an arity-zero `int` function. It does not change NanoISA, NanoVM, `nvm2c`, LLVM
or Wasm semantics. Full high-level reconstruction and the v5.1.0 release gates
remain open.

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

My first fresh bootstrap of this prerequisite stopped before compilation: the
new checker shadow used `byte` as a local name, but `byte` is my reserved alias
for `u8`. I retain that parser terminal and rename only the shadow local to
`octet` before repeating the bootstrap.

The next bootstrap stopped at a static checker diagnostic before producing
Stage 1: my self-hosted `NSType` model represents `u8` as the exact named type
`u8`, not as a `TypeKind.TYPE_U8` enum member. I retain that terminal and make
the contextual hint test the existing exact type spelling. I do not extend the
checker type enum as part of this repair.

The corrected bootstrap completed both stages and installed Stage 2. Its first
focused run then established three separate facts before the host data volume
filled and later cases became invalid short writes:

- the self-hosted canonical emitter still excludes `u8` from its supported
  local-type whitelist;
- the C-seed canonical emitter preserves byte literals in bindings, calls and
  results, but a byte assignment still emitted `PUSH_I64`;
- the self-hosted native-C route emits unknown `nl_u8` and scalar-conversion
  helper spellings. I record that separate product gap as
  `task_633a3abf5a1040e6863a525ed5cc80b5`; it is not required to relabel the
  canonical NanoISA route as native-C support.

I correct the two canonical emitter gaps in this task. I qualify reconstructed
Nano source through C seed, Stage 1 and Stage 2 canonical NanoISA producers,
then execute the same modules in NanoVM and translated strict native C. The
C-seed native source compiler remains an additional ordinary control. I do not
claim self-hosted native-C `u8` support from those gates. The invalid no-space
tail of the first focused run is retained as environment evidence, not counted
as a language failure or a test result.

The next focused invocation also exposed three fixture prerequisites before a
complete result: I had not built the declared `nvm2c` target, one negative
source reused my reserved `byte` alias as a function name, and the literal-tag
control asked the self-hosted emitter to resolve `cast_bool` without a source
declaration. I correct those fixture boundaries only. Explicit byte-to-boolean
conversion remains exercised by the reconstructed Nano source, while the
checked-source control stays limited to contextual literal tags.

After those fixture corrections, Stage 1 and Stage 2 reached one remaining
scoped emitter refusal: their canonical `cast_int` whitelist omitted exact
`u8`, even though NanoVM and the C-seed producer already implement the
unsigned widening. I add only `u8` to that explicit conversion admission. I
do not infer support for generic byte arithmetic, comparison or logic from
this conversion.

The following focused run passed the byte-to-boolean loop and source refusal
controls, then stopped at `unsupported result type u8`. The general type
classifier already admits exact byte parameters and locals; its separate
result-tag classifier omitted the same scalar tag. I add that exact result tag
without admitting a byte entry result or any aggregate byte shape.

This extension does not admit a `u8` entry result: my executable entry remains
an arity-zero `int` function. It does not change NanoISA, NanoVM, `nvm2c`, LLVM
or Wasm semantics. Full high-level reconstruction and the v5.1.0 release gates
remain open.

## Direct parameter metadata correction

My final self-review found that execution did not prove the direct parameter
metadata. At PR859 head `7ab0afc0`, a fresh identical
`identity(value: u8) -> u8` probe produces `.parameters 0 u8` through NanoVirt
but `.parameters 1 void` through Stage 1 and Stage 2. Their result tags remain
`u8` and the program executes, so the existing focused matrix could miss this
metadata loss.

I record this as `task_affeba0b68cd8ad7e2c3756c672c934a` before changing
code. I correct only the self-hosted parameter tag serialization and its exact
byte type guard. I require the three producer dumps to contain a byte
parameter tag, followed by verification, VM and translated-native execution.
The first source and dump hashes remain in my evidence report. This does not
widen indirect calls, aggregates, generic byte operations or native-C source
generation.

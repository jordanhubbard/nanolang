# My Darwin self-host stack boundary

I keep generated self-hosted compiler products below Darwin's native stack
boundary for `task_e1598bc864212c669cc8b4cb2b73be55`. This is a compiler
product policy, separate from the scalar-union implementation that exposed the
failure.

## My retained terminal

My first fresh three-compiler control passes C-seed and Stage 1, then Stage 2
exits in `___chkstk_darwin` while recursively parsing the unchanged
`Box<array<int>>` case. Removing the optional match-guard parser temporary does
not change the terminal.

The generated native prologues reserve `0x180340` bytes for `parse_primary`
and `0x92fc0` bytes for `parse_expression_recursive`. Moving the large
scalar/string/array/map/value pools to the heap does not change those frames,
so I reject that initial storage hypothesis. The space comes from the current
by-value generated record-result ABI at an unoptimized native-C boundary.

Compiling the exact retained C with Apple Clang `-O1` reduces the same frames
to 10,432 and 10,064 bytes. The unchanged control then passes.

## My policy

When `NANO_CFLAGS` is absent or explicitly empty, I compile the generated
native product with `-O1`. The environment interface presents both cases as
the same empty string, so I do not claim to distinguish them. When
`NANO_CFLAGS` is nonempty, I preserve its bytes exactly. This keeps explicit
debug, sanitizer, optimization and diagnostic selections authoritative. I do
not change the C-seed tool build, linker flags, target selection, runtime ABI,
or source language semantics.

A future generated-C ABI change may reduce the unoptimized frames, but it is a
separate architectural task. I do not block this bounded product safety fix on
that redesign.

## My acceptance

I require helper shadows for the empty/default and nonempty explicit settings,
a fresh Darwin bootstrap through both self-hosted stages, installed-compiler
and no-C-seed smokes, the unchanged source control, and retained static frame
measurements. Native stage byte inequality is recorded and is not a
fixed-point claim.

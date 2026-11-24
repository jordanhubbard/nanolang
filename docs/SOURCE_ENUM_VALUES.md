# My source enum values

I retain explicit enum values as signed 64-bit integers in my parser, checked
metadata, interpreter, module metadata copies, and source C/native/VM emitters.
My first implicit value is zero. Each later implicit value is its predecessor
plus one. I refuse literals outside `-9223372036854775808` through
`9223372036854775807`, and I refuse an implicit successor after the maximum.
An explicit value can restart that sequence.

```nano
enum Edge { Min = -9223372036854775808, MinNext, Wide = 4294967297 }
fn wide() -> Edge { return Edge.Wide }
shadow wide { assert (== (wide) 4294967297) }
```

I preserve the declared enum identity when I resolve a selected lower-case
`list_Type_*` builtin. A lexical callable with that spelling keeps its binding.
Known enum-list argument declarations must agree with the known argument's
element identity, including distinct module declarations with the same source
name. This check does not infer nominal identities from unknown integer bits.

My generated non-schema list helpers are local to each C translation unit.
`pop` returns the last element and reduces the length; an empty or null list
stops with a diagnostic. `free` releases the list buffer and header. It does not
recursively destroy record fields or string values already copied from a list.
These operations do not enable ordinary interpreter aggregate collection.

My source-only `--target c` profile resolves local enum members when no lexical
value owns their namespace spelling. It emits signed integer views of their
constants. For wide values I emit an `int64_t` alias and integer constant
macros instead of requiring C23 enumerators. I check the emitted program with
strict C99 and C11 consumers. Source-only emission does not execute shadows; I compile and run its
output separately. Unsupported profile values still refuse publication.

I track these repairs in #1236, #1238, #1239 and #1240. My native publication
failure under #1237 remains separate: a C compiler or linker failure can remove
prior requested native output. Passing enum programs do not establish that
publication boundary, complete module enum serialization, every raw tagged
operation, paired-platform acceptance, or the full release.

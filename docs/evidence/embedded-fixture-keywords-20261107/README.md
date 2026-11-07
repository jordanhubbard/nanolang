# My fast embedded fixture identifier check

I address #1047 with `make test-embedded-nano-identifiers`, included in
`make check-fast`. I inspect static Python string fragments with declaration
syntax, using `src/lexer.c` and `src/utf8.c` directly. I do not duplicate the
reserved-word list or execute the Python files I inspect. My native adapter
keeps string/comment handling and token positions from the real lexer.

I inspect names after `let`, `fn`, `struct`, `enum`, `union`, `shadow` and
`effect`, allowing the existing `let mut` form and unnamed function types.
Diagnostics identify the Python literal line and embedded token line/column.
An explicit one-name annotation retains a deliberate reserved-name refusal;
it must match an actual reserved declaration and cannot hide another name.

My four actual-lexer controls pass with both Clang and GCC. They mutate copies
of the two original reported fixture files, reintroducing `handle` in the
owned-shadow binding and `effect` in the executable-closure function. Both are
reported before compilation. Strings/comments containing those words pass;
exact negative annotations remain checked; inspection does not execute Python.

The initial full inventory also finds `fn byte` and its shadow in a positive
function-value fixture. I reproduce my C-seed parser refusal, rename that
function and shadow to `read_byte`, and retain the same body, callable type,
arguments and assertions. C-seed native compilation runs its selected shadows,
and the executable prints `42` with exit zero. I preserve the intentional `not`
binding in the owned-float-array lexical-refusal test, with an exact annotation.

`make check-fast` passes the four controls, 3,118 static fragments with zero
unacknowledged reserved names, five prerequisite controls, 223 Clang and 223 GCC
product compilations and the bootstrap-tool control. I report 133 lexer-refused
fragments separately. Those partial or negative fragments and all dynamic source
construction still belong to their executable tests. This check does not
establish complete fixture parsing, producer parity or release acceptance.

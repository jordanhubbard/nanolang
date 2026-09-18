# My boxed native numeric arithmetic boundary

I track this child as `task_bbf36d05756945d7869a0b93e35ea42e` under generic
arithmetic parent `task_66a6dd8ca51d415f9efb0f2904f85b49`. My contract was
recorded at `011147e9` before production `35fb191d`.

My existing `nmap_value` carrier stores a runtime tag and either integer bits
or copied binary64 bits. I reuse it without allocating a numeric object or
changing roots, collection, aggregate ownership or OPTIONAL payload rules.
A checked helper rejects every operand except exact INT/FLOAT tags before
arithmetic. Two integers produce integer; otherwise ADD/SUB/MUL/DIV promote
to binary64 and produce float. NEG retains integer/float identity. MOD still
uses the existing checked integer route.

Integer wrapping, minimum-integer division/negation and zero totality retain
the existing unsigned-bit reconstruction and guards. Float division by either
zero sign returns positive zero, including NaN/infinite numerators. Other
floating operations retain binary64 behavior and integer-conversion rounding.
U8, bool, void and heap values acquire no numeric coercion. I make no new enum
arithmetic claim.

The classifier marks successful boxed arithmetic results with the closed
INT/FLOAT provenance mask. The runtime helper establishes that postcondition;
unknown input tags are not themselves evidence of numeric values. Existing
boxed local/call storage and boxed-to-boxed joins preserve exact result tags.
Declared scalar returns and typed consumers still check the actual tag before
unboxing. My tests exercise both operand orders through calls and a tail return,
and select int or float at a boxed branch join before checking its actual tag.

## My shape boundary

My shape graph's OPTIONAL node has one exact present-payload child. It does
not describe the union of concrete INT and FLOAT shapes. I therefore retain
refusal of a boxed arithmetic result joined with a concrete scalar stack edge,
including preservation of an existing output artifact. Both boxed edges have
OPTIONAL storage; no new rule equates distinct concrete payload shapes. I do
not weaken a failed graph or substitute integer storage when it refuses.

The separately recorded dependency is
`task_87a7b44d10e240e999485d12aeecaca2`. Broader concrete/boxed joins, mixed
input shapes at shared call sites and complete scalar-union representation
need that explicit contract. This child does not close full parent acceptance.

## My ordinary acceptance

`make test-native-tagged-arithmetic` covers 48 combinations of binary operator,
int/float pair and left/right/both boxed operands. Further cases check locals,
calls, tail returns, boxed branch tags, wrapping, rounding, signed zero,
NaN/infinity, total division, invalid tags and the unsupported join boundary.
The same modules execute in NanoVM and generated standalone native C.

Invalid runtime values require SIGABRT without ASan/UBSan/LSan diagnostics.
Typed consumer tests separately require exact integer/float guard termination
on mismatched arithmetic results. These are normal tests of repaired source,
not replays of preserved compiler failures.

My initial six focused GCC methods passed in 29.080 seconds. The combined
thirteen tagged/concrete Clang sanitizer methods passed in 11.966 seconds.
The added typed-consumer method passed GCC in 7.769 seconds and Clang in
0.385 seconds. The expanded combined GCC gate, including the wrapped-integer
method, passed 14 methods in 149.689 seconds. Final native results are retained
with the PR.
Logs use `/tmp/nanolang-tagged-arithmetic-`, including `focused.log`,
`clang.log`, `final-gcc.log`, `consumers-gcc.log`, `consumers-clang.log` and
`native.log`. Parent source review found no blocker in this bounded runtime
helper and classifier change. LLVM/Wasm work remains a separate companion.

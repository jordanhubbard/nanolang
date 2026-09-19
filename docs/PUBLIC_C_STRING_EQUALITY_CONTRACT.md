# My public C string equality contract

I execute task_4c6d924e0aa242df950f662f61981664 after the public formatting
checkpoint PR749. My isolated branch starts at35b1121f; I integrate its canonical
merge before final qualification. Parent6ade and full release remain open.

My retained formatting source failures at9a5e1f8a emitted raw C pointer equality
for STRING EQ. I keep their source/logs and never execute those failed binaries
again. This static defect is independent of the successfully observed format bytes.

I lower EQ/NE only when both operands resolve exactly STRING. I compare their
contents, with identical pointer fast path and defined distinct-null handling
matching existing VM string equality. I do not introduce ordering or widen unknown,
mixed scalar, heap or generic comparisons. A STRING paired with a non-STRING or
UNKNOWN receives a checked publication-preserving refusal. Non-string comparison
rules stay unchanged in this bounded child.

I evaluate the left operand once, then the right once, into distinct automatic
const-char-pointer slots in the enclosing function/global initializer. A C99 comma
expression sequences those assignments before a private equality helper. Nested
comparisons receive independent slots; recursion gets independent frames; loops
reevaluate inside their condition/body and branches evaluate only the selected
arm. I reuse the existing staged declaration and hygienic prefix planning. Neither
GNU statement expressions nor unspecified C argument evaluation order implement
this contract. Pointer snapshots do not add ownership or reclaim borrowed strings;
PR749 conversion snapshots retain their process-lifetime contract.

Before production I record this roadmap/MAC contract; before fresh fixture execution
I send a production checkpoint. I freeze complete source/harness/tools, then qualify
GCC/Clang C99/C11 O0/O2 with ASan/UBSan/leak checks. New ordinary controls cover equal
content in distinct allocations, unequal/empty strings, aliases, exact scalar fields,
helper/global returns, nested branches/loops, effect order/count, private-name
collisions and initializer reentry. Isolated API refusal controls retain prior output
and recover on a later valid compile. An independent direct helper test covers
pointer/null identity without exposing invalid language source. I preserve previous
formatting/arithmetic/backend gates and report source-tool identities accurately.

I do not silently close public C concatenation allocation, integer conversion's
GNU block, other expression/options portability, full scalar policy or release.
Those remain required separate dependencies under6ade and their existing parents.

My storage audit qualifies value comparison over the existing NUL-terminated
public C string representation, not arbitrary VM length-bearing byte strings.
Canonical source currently decodes escapes then uses strlen; public C re-escapes
raw parser text. I record that distinct literal-normalization task before code and
keep it required under6ade. This child preserves the existing string transport;
its ordinary equality fixtures use shared unescaped text/conversion outputs.

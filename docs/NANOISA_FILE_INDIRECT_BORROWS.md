# I carry explicit references through indirect File calls

Under #989 I extend the private indirect File profile with
`FILE_CALL_INDIRECT_REFS` (`0x98`). Its operands are a little-endian u16 total
parameter count, u16 result count, and u32 counted-string constant index. The
constant contains exactly one little-endian u16 reference slot per parameter.
Ordinary and owned parameters use `65535`; borrowed parameters name a caller
reference slot. At least one parameter must be borrowed. Legacy instructions
and their encodings remain unchanged.

The callable remains the top operand. Only non-borrowed arguments occupy the
operand stack below it. Structural preparation checks map extent, index and
slot bounds. Target analysis checks every candidate's complete signature,
including shared/exclusive modes, against the map. The ownership transition
applies every candidate using those explicit references, retaining shared aliases
and rejecting dead references, mode disagreement and exclusive overlap. Empty
or incompatible target sets still refuse; all call edges participate in the
existing recursion check.

The owning hosted plan copies the map with the instruction. Its call obligation
records borrowed and owned input counts. Frame bounds subtract ordinary/owned
operands plus the callable, and retain all formal slots. Up to256 borrowed
parameters can use a single shared caller reference; the indirect staging bound
therefore reaches258 slots. The per-frame value ceiling is770 in this private
profile. I retain the existing function, local, operand, reference and byte
budgets, and checked arithmetic precedes allocation.

The carrier first checks invocation-plan identity, selected membership, exact
callee declarations, every reference's live lease/mode, and alias compatibility.
Only then does it clear the callable and stage arguments. Each borrowed formal
binds to its named caller reference. Returning removes formal aliases without
ending the ancestor borrow. The VM and generated C functions use this same
carrier protocol and shared fuel. Failure drains every live owner and retains
the first execution error.

Generated native startup compares each copied reference-map entry as well as
module bytes, candidate sets and other retained facts before host acquisition.
Native dispatch still selects real generated functions. Generic VM, AOT and
reconstruction consumers continue to refuse private File fragments; an opcode
entry alone confers no service authority.

My corpus covers shared aliases, repeated shared slots, exclusive and mixed
borrows on distinct owners, reordered maps, forwarding through another indirect
call, both target alternatives, catalog permutation, actual writes through
exclusive references, all lower fuel budgets, denial and assertion/cleanup
failure. It refuses malformed lengths/indices/slots, scalar-slot maps, candidate
mode disagreement and exclusive overlap. I exercise256 borrowed formals and
mutate a generated reference-map comparison independently of embedded bytes.

This is the private wire/query/runtime/dispatch boundary. Paired C/Nano source
lowering, full selected shadows, public grants, installed consumers, Linux
qualification and the rest of my5.1 release contract remain required.

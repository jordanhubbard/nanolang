# My existing string stdlib source contract

I retain `tests/test_stdlib_strings.nano` unchanged. Its complete main and shadows exercise prefix/suffix/search, trim, trim-left/right, case conversion, replacement, splitting and joining. I add production-main controls so an operation in a shadow is not mistaken for an instruction in the published main product.

I follow my existing VM and C implementations on the exercised byte domain:

- My trim-left/right helpers remove only space, tab, LF and CR on their selected side. I retain vertical tab, form feed and non-ASCII whitespace.
- My case opcodes change ASCII A–Z/a–z bytes. Other bytes keep their value; I do not introduce locale or Unicode case mapping.
- My replacement opcode replaces nonoverlapping occurrences left to right. An empty old string returns an independent unchanged result; no match keeps the bytes.
- My split opcode uses the complete delimiter, retains empty leading/interior/trailing elements, and returns one empty element for an empty input with a nonempty delimiter. An empty delimiter splits individual bytes; an empty input then returns zero elements. I do not reinterpret those byte fragments as Unicode characters. My existing foreign join bridge refuses non-UTF-8 fragments; I preserve that refusal rather than asserting a split/join round trip across this boundary.
- My join helper inserts the complete separator between elements, preserves empty elements, and returns an empty string for an empty array. My input storage must be ARRAY<STRING> for native admission; a coarse ARRAY tag does not establish that storage.

I retain source evaluation order and evaluate each argument once. Existing lexical function bindings keep their signatures. Checked malformed operands/arity must refuse before replacing prior output. I retain malformed raw input and unrelated unsupported import/opcode refusals, useful source shadows, original C units/alias suites and complete translator gates. My historical STR_TO_UPPER unit deliberately refuses a valid existing string opcode because it was outside my native subset. I preserve that baseline source/pin; admission requires actual execution of that same valid module and retained empty-stack/wrong-type refusals, rather than retaining a capability denial for the feature I am implementing.

My C string ABI terminates at NUL, while some VM string opcodes retain explicit lengths. My C helpers also have existing bounded string scans. I do not infer complete embedded-NUL or arbitrarily large cross-engine equivalence from ordinary UTF-8 examples. I keep these boundaries explicit rather than substituting a new generic host or SDK contract.

I first retain a failing baseline. Implementation, actual VM/sanitized-native qualification, fresh stages and complete platform/release admission follow as separate gates under #1106/#1081.

I implement four existing string opcodes and three exact builtin host adapters, with source arity/type checks and preserved lexical identity. My native join reuses the shared UTF-8 validator for array elements; the scalar separator retains its byte contract. I retain my [failed hosted baseline and local helper checks](evidence/string-stdlib-20261108/local-implementation/README.md). Full generated-program and fresh compiler acceptance remain open.

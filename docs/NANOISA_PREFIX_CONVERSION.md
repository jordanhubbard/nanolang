# My canonical prefix conversion

I lower the unresolved builtin `string_to_float(string) -> float` to the existing portable `CAST_FLOAT` operation. I require exactly one string operand, evaluate it once, and retain float metadata through inferred locals, call arguments, returns and arithmetic expressions. My parser and runtime conversion rules are unchanged by this lowering repair.

I retain declaration lookup ahead of this builtin's inferred result. Qualified module names keep their mapped identities; an existing selfhost declaration named `string_to_float` keeps its checked result and argument contract. My C seed reserves builtin declaration names, and the acceptance records that existing difference explicitly. I do not turn a C-seed name refusal into a successful call.

My initial product failure is retained in `/tmp/nanolang-product-e46c425b-focused.log`. The first emitter repair passes the original prefix fixture but exposes missing checker result metadata for inferred locals; `/tmp/nanolang-prefix-conversion-gcc.log` preserves that checked refusal. The next gate passes inference and selfhost qualified binding but rejects the new qualified test on the C seed because that declaration name is reserved; `/tmp/nanolang-prefix-conversion-inferred-gcc.log` preserves the evidence. The corrected test asserts that reservation and separately exercises accepted selfhost declarations.

At production `27bc0fb3`, fresh three-stage bootstrap passes. Seven focused GCC methods pass in 9.646 seconds, including the unchanged prefix fixture, canonical VM and native sanitizer execution, inferred locals, single operand evaluation, local/qualified declaration binding, wrong operands and previous-output preservation, all-stage legacy execution, and parser endpoints. The same seven Clang methods pass in 10.154 seconds; `/tmp/nanolang-prefix-conversion-binding-clang.log` retains their results. I retain `/tmp/nanolang-prefix-conversion-binding-bootstrap.log` and `/tmp/nanolang-prefix-conversion-binding-gcc.log`.

This bounded repair completes task `task_1c2a4d2d7b5a4b6caca5d186d3c7c181` only after canonical integration. Full product and release acceptance remain open.

After integrating canonical main through PR668 at `7d7aafc4`, my selfhost compiler sources remain byte-identical to the tested conversion source. Rebuilt local VM/native tools and the same seven focused methods pass in 10.102 seconds; `/tmp/nanolang-prefix-conversion-integrated-gcc.log` preserves this integration check. I do not relabel that focused check as a new whole-bootstrap result.

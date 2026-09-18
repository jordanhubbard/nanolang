# My native C literal preprocessing contract

I execute `task_35aadf192c224eed856b3b0597dbedea`.

I retain the terminal ordinary source-byte gate from public literal childcf8714: public C/interpreter/verified VM and decoder failure checks passed, while native C emitted an unescaped question-mark trigraph sequence and observed different bytes. Logs /tmp/nanolang-public-c-literal-{gcc,clang}.log; frozen18 identities verified. Static emit_c_string_lit in src/nanoisa/nvm2c.c escapes quote/backslash/control/high bytes but emits question marks raw. I add only question-mark backslash escaping in that helper, preserving exact input lengths, all existing escape handling, opcode/type admission and ownership. Fresh ordinary all-nine trigraph-like sequences plus neighbor quote/backslash/digit/UTF8 controls must match exact VM/native output under GCC/Clang C11 O0/O2 sanitizers. No malformed input, exploit case, crash reproduction or historical emitted binary execution; corrected fresh modules only. Separate sibling PR lands before final public literal native parity gate.

My initial build invocation named nonexistent make target nanoisa; I retain
that setup log, let its jobs terminate, then build the actual nanoisa_dump CLI
target alongside the VM and translator. No execution test ran before setup.

## My strict shared-provider prerequisite

I record `task_f1726009bb784f44a532c57f3ba6a799` before production.

I preserve strict Clang -Wall/-Wextra/-Werror acceptance for ordinary generated native C after shared formatting provider PR745/752. Fresh normal string-only literal tests passed GCC but Clang refused unused nano_rt_f64_format and nano_rt_f64_print before execution; retain /tmp/nanolang-native-trigraph-clang.log. I add only ordinary-main (void) references to those helpers exactly when need_print || need_cast emits the shared provider, following existing helper retention style. No fake runtime calls, warning suppression, profile widening or provider-body fork. Fresh printing-only and cast-only normal modules plus the literal byte tests qualify strict GCC/Clang O0/O2 sanitizer acceptance. No historical artifact replay. Root e92 remains open.

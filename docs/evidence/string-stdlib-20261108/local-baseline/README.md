# My local string stdlib baseline boundary

I retain both initial and corrected baseline attempts, their complete text, source/command hashes and the actual emitted local NVM bytes in a private CRC-verified ZIP. The local providers have older/unqualified source provenance. These observations locate existing behavior for my baseline controls; fresh hosted products remain required before attributing the result to the current checkout.

Four corrected positive methods compile, verify and execute all their assertions in NanoVM, then refuse native translation: the unchanged whole fixture and trim method stop on missing exact vm_str_trim_left admission; case/replacement stops on unsupported STR_TO_LOWER; split/join stops on missing exact vm_str_join admission. The five-method corrected run retains four failures. A separate malformed foreign-join test passes its required refusal and prior-output assertions.

My first split test incorrectly expected joining individual bytes of a UTF-8 string to cross the existing foreign ARRAY<STRING> bridge. That bridge requires NUL-free UTF-8 elements and correctly refuses the fragments. I preserve that failed control and boundary, correct the positive to inspect byte fragments with the existing byte-index operation, and add an explicit foreign-join refusal. I do not weaken marshalling or rewrite the original whole fixture.

The source/native implementation and current GCC/Clang baseline are pending under #1106. I keep all production changes out of this baseline checkpoint and preserve the original expected outputs, shadows, translator gates and deadlines.

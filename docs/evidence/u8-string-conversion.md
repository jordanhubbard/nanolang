# My byte-to-string conversion contract

I track task_d08968be827a4d26942123825229ac8e after the scalar U8 foundation.
CAST_STRING of a TAG_U8 value produces its unsigned decimal representation,
with no sign, prefix, padding or leading zeroes (except the value zero itself).
This agrees with existing byte display and CAST_INT followed by CAST_STRING.
The result has TAG_STRING and uses the same owned allocation and reference
lifecycle as an integer conversion. The consumed byte has no heap reference.

I reuse my existing VM integer-string allocation and native managed numeric
string helper. I preserve aliases, returned strings and subsequent collection;
I do not introduce an allocation outside those lifetime systems. I test all
256 ordinary values and retained strings across conversion churn, including
native sanitizer/leak checks. LLVM/Wasm continue to refuse CAST_STRING because
their current profile excludes strings; failed publication preserves output.

The earlier empty-string VM fallback is retained in the preceding U8 task's
failure logs. It is not my intended numeric string conversion contract.

## My measured acceptance

My base is merged U8 scalar support `26a1ee5e`. I recorded this contract at
`cb831094` before implementation `221909ae`.

I pass 274,416 VM checks, including all 256 byte-to-string conversions with
exact tags/content and release back to the original heap-object count
(`/tmp/nanolang-u8-string-build-vm.log`). My paired VM/C suite passes three
methods in 2.103 seconds (`/tmp/nanolang-u8-string-paired-final.log`). I compare
every decimal string against CAST_INT followed by CAST_STRING and exact
expected output. The returned-alias case retains two caller references while
20,000 subsequent conversions cross the native string allocation-debt budget.

The same three methods pass with generated native code under ASan/UBSan and
leak detection in 13.038 seconds (`/tmp/nanolang-u8-string-sanitized-final.log`),
and strict Clang in 0.722 seconds (`/tmp/nanolang-u8-string-clang.log`). These
runs instrument generated C, not the entire VM. LLVM/Wasm explicitly refuse
the string opcode and preserve previous output; this is refusal evidence,
not string execution coverage. I did not rebuild or rerun the full compiler.

`make test-u8-strings` makes the two ordinary VM/C execution methods required
by test-units; the explicit Wasm gate includes the string-refusal method too.

A static adjacent audit records `task_822750774e0040298e8297e96145f3d5`: existing integer, float,
boolean and default CAST_STRING arms do not check NULL from their allocation
helpers. I did not reproduce allocation failure or change those arms here.

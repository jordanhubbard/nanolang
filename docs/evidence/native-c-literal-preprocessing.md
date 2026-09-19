# My native C literal preprocessing evidence

I tested reviewed question-mark escape1fac7081 and shared-provider reference
checkpoint63c337db with frozen harness74a7457b. My [manifest](native-c-literal-preprocessing.json)
records six source/provider/tool hashes verified unchanged and seven retained logs.
The [contract](../NATIVE_C_TRIGRAPH_CONTRACT.md) precedes each correction under
35aadf192c224eed856b3b0597dbedea and f1726009bb784f44a532c57f3ba6a799.

I route question marks through my existing C backslash escaping. I change no
bytecode profile, input length or ownership. I also retain non-executing references
to shared format/print helpers exactly when PRINT/PRINTLN/CAST_STRING emits them;
ordinary nvm2c has no admitted no-main mode. I do not suppress compiler warnings,
add fake calls or modify the provider body.

Fresh verified VM/native controls pass all three methods on GCC (0.612 seconds)
and Clang (1.125 seconds), using strict C11 -Wall/-Wextra/-Werror, O0/O2 and
ASan/UBSan. Exact independent output and lengths cover all nine trigraph-like
sequences, quote/backslash/question neighbors, UTF-8, control-byte/digit adjacency,
empty strings and print-only/cast-only provider emission.

I preserve the initial nonexistent nanoisa make-target setup error, the corrected
build, initial two GCC passes and strict Clang unused-provider refusals. Those
refusals produced no executable. The motivating public literal gate's native
stdout mismatch remains retained in its own tree; I did not replay its emitted
binary. Final public-decoder parity must use the canonical merged correction.
No full native suite, source bootstrap, owned-emitter or platform-wide claim follows
from these bounded gates; both MAC children await canonical merge.

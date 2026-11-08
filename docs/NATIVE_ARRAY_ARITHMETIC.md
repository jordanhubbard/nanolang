# My native flat-array arithmetic contract

I record this #1094 contract before production changes. My [8127 baseline](evidence/array-arithmetic-20261108/baseline-8127/README.md) executes the entire original array-operator fixture in NanoVM but refuses native ARRAY_ADD. My earlier self-hosted lowering also refuses its STRING-array addition. My existing [aggregate binary64 policy](NANOISA_AGGREGATE_BINARY64_POLICY.md) left this native capability unqualified.

I retain the original fixture and assertions. I keep my existing numeric source classifier exact and add a separate STRING-array addition classifier. Both producers select existing ARRAY_ADD/SUB/MUL/DIV operations; I do not send arrays to scalar arithmetic instructions.

| My operands | My operators | My result |
| --- | --- | --- |
| Exact array<int> with array<int> or int, either direction | + - * / | Independent array<int> |
| Exact array<float> with array<float> or float, either direction | + - * / | Independent array<float> |
| Exact array<string> with array<string> or string, either direction | + | Independent array<string> |

I require at least one known array. I preserve source evaluation once, left before right, then read lengths after both operands finish. Two arrays use the shorter length; a scalar broadcasts over the other array. Empty results retain their element kind and independent outer storage. I preserve source buffers and aliases.

I reconstruct integer addition, subtraction and multiplication from wrapped unsigned bits. Division by zero returns zero; INT64_MIN divided by -1 returns INT64_MIN. I consume my existing shared binary64 helpers for each floating operation: canonical arithmetic NaNs, positive zero for either zero divisor, per-operation rounding and preserved input bits. I retain the original fixed-bit oracle and ordered alias-growth source. Native O0/O2 sanitizer runs precede acceptance.

I allocate result arrays through my existing managed native array owners and concatenate strings through my existing retained string allocator. Existing root traversal and cleanup retain result strings. I preserve checked capacity/size failures and qualify actual native lifetime and refusal boundaries.

This initial native profile requires known matching flat kinds. Mixed numeric kinds, unknown operands, nested/record/function/bool/u8 arrays and STRING subtraction/multiplication/division do not acquire guessed storage. I retain their native refusal and prior output. Existing broader raw VM arithmetic is not a claim of native admission for those excluded profiles. My old numeric helper's STRING refusal remains a valid test of that helper's narrower contract.

I require the whole original fixture, existing binary64 oracle/order sources, integer boundaries, STRING order/shorter lengths, empty/independent storage, malformed source/module output preservation, complete translator regressions and fresh stages. Linux/Darwin and integrated release qualification remain separate required evidence. This document is an acceptance contract, not a passing result.

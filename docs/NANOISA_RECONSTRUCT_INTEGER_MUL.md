# My total reconstructed integer multiplication

I record task_615c3ccf924d4aaaa964237bcc9d286c before code as the next child of reconstruction parent4bd034, following merged PR655. I admit exact int operands and result for I64_MUL only. C uses uint64_t multiplication with the existing representable signed conversion. NanoLang uses at most 64 iterations: I halve a signed multiplier toward zero, add the multiplicand for positive odd digits, subtract for negative odd digits, and double the multiplicand with my tested wrapping add helper. Division/remainder by constant two have no signed overflow or zero-divisor case. All accumulation/doubling uses the existing total helpers, so MIN and negative multipliers never require an overflowing absolute value.

I emit only needed helper dependencies, preserve operand snapshots and the bounded structured grammar, and retain exact type/refusal/output-publication rules. I require same-module VM/C/three-stage-NanoLang execution, canonical byte roundtrip, signed endpoints/high-bit products, odd negatives, zero/one, loops and calls with GCC/Clang sanitizer coverage. My generated helper shadows validate known helper semantics and do not reconstruct original tests. I64_DIV/MOD, generic arithmetic, other values and full high-level reconstruction remain separate; I do not change product compiler or runtime production code.

My recurrence retains `result + factor * remaining` modulo 2^64 after each signed-digit step. The multiplier magnitude falls under truncating division by two, including MIN without taking its absolute value. The NanoLang helper therefore takes at most 64 iterations; this is a finite cost bound, not measured performance equivalence to native multiplication.

For compiler acceptance I use pinned C-seed/Stage1/Stage2 tools from source `4a75f984`, separately from this reconstruction generator source. The newer child base also includes C match-arm metadata changes that this scalar grammar does not exercise. I do not claim a same-source or current-main bootstrap from reused-stage checks. The earlier add/sub/neg evidence source-equality wording is corrected in this PR; its actual arithmetic results and compiler hashes are unchanged.

Pinned tool SHA-256 values:

```text
nanoc_c      442caf121e671388e4d22ea207789ad56efca659555c4ee451596eb3d4b99041
nanoc_stage1 d078dc8c6747b849cb2228fa33165b06aa350d5f6e53b8911fd72696515542fa
nanoc_stage2 b276026caf525bde10c063b2f7654f192b81ace73587e5af5bebda4986581a40
```

## My measured acceptance

At generator checkpoint `b56db07c`, twelve combined reconstruction methods pass GCC in 113.507 seconds and Clang in 109.106 seconds. The three new methods first pass in 19.710 seconds. They compare 121 signed-endpoint/high-bit products, multiplication inside a structured loop, exact bool-tag refusal and canonical byte roundtrip. The retained modules execute through VM, reconstructed C with ASan/UBSan, and reconstructed NanoLang through all three pinned compiler stages; C-seed-generated NanoLang C also uses UBSan with recovery disabled. I rechecked all three recorded compiler hashes after the runs.

Logs: `/tmp/nanolang-reconstruct-mul-first.log`, `/tmp/nanolang-reconstruct-mul-gcc.log` and `/tmp/nanolang-reconstruct-mul-clang.log`. The existing arithmetic-refusal controls now use still-unsupported I64_DIV_S; no refusal gate was removed. This tests typed multiplication and helper dependency emission, not division/remainder reconstruction or full parent completion.

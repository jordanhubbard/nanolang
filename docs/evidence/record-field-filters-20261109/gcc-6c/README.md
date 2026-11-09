# My completed GCC comparison

I retain all six attempts from [run 42551845618](https://github.com/jordanhubbard/nanolang/actions/runs/42551845618), comparing pre-filter 48016fec4 with candidate 6c97df51b. Every attempt completes all 1,286 identically ordered shadows with zero recorded assertion failures and publishes a real ELF source compiler under the original ten-second shadow and 600-second native-build budgets.

| Pair | Original body seconds | Modified body seconds | Modified change |
| --- | ---: | ---: | ---: |
| 1 | 9.173215 | 8.266774 | -9.88% |
| 2 | 9.287713 | 8.346877 | -10.13% |
| 3 | 9.409857 | 8.445377 | -10.25% |

I measure the same direction in every pair. Median shadow-body time changes from 9.287713 to 8.346877 seconds. These are summed shadow-body times, not a 10% whole-compiler speedup; whole-factory wall times remain in my complete summary. The Apple comparison and all remaining release requirements stay separate.

My retained audit checks the complete ZIP CRC and every entry hash, matches all 197 shared source/factory inputs against both exact Git revisions, recomputes ordered trace events and summed body times, checks all actual per-attempt input records and unchanged provider records, and independently hashes both retained C providers and all six native products. I retain complete text including emitted C in compressed form, the actual native-product hashes/magic, and the original run summary. I do not execute downloaded products or infer full NativeSDK/runtime closure from these hashes.

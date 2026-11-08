# My completed controlled Linux comparison

Linux/Clang's job in [run 42533514074](https://github.com/jordanhubbard/nanolang/actions/runs/42533514074)
completes the exact six-attempt plan. Baseline 9baeba459 includes the registry and
terminator fixes; candidate 724470b2f includes the private entry allocation.
All six actual factory attempts pass all 1,283 selected shadows with zero
assertion failures and publish real ELF source compilers under the unchanged
10-second supervisor and 600-second native build budget.

| Planned pair | Original body seconds | Modified body seconds | Modified change |
| --- | ---: | ---: | ---: |
| Original, modified | 7.473001 | 7.454338 | -0.25% |
| Modified, original | 7.411854 | 7.427697 | +0.21% |
| Original, modified | 6.884627 | 6.507934 | -5.47% |

I retain every sample. The first two pairs differ by less than 0.3% with opposite
signs; the third favors the candidate. Median body times are 7.411854 and 7.427697
seconds. I do not establish a consistent source-body improvement from these
observations or attribute different-worker failures to infrastructure. This
isolated comparison does not reproduce the full component job's preceding
runtime/checker workload and cannot replace that gate.

I CRC-check the complete 32,375,954-byte artifact in memory and retain its digest,
full entry inventory and exact text. I independently match actual bytes of both
C providers and all six published native source compilers to the before/after
production records. Shared input hashes, provider hashes, ordered selections,
all six outcomes and native product hashes remain unchanged. ZIP file modes and
the complete NativeSDK/runtime closure are not independently qualified here.

The first Apple setup stops for measured disk headroom before any sample.
Its corrected complete-tree copy-on-write setup runs separately at 902cb0c46;
I keep this Linux run and its results rather than restarting it. Full combined
source/native, fresh-stage, platform, budget and release acceptance remain open.

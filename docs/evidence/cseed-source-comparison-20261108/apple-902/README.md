# My completed controlled Apple comparison

[Run 42534611623](https://github.com/jordanhubbard/nanolang/actions/runs/42534611623)
at 902cb0c46 completes all six planned attempts after both actual provider builds
succeed. Complete tracked copy-on-write trees preserve 92,771 baseline entries
and 92,809 candidate entries. Every required shared source/factory input remains
byte-identical; no files or fixtures are omitted and the headroom guard remains.

| Order | Variant | Completed shadow bodies | Summed body seconds | Creation |
| ---: | --- | ---: | ---: | --- |
| 1 | Original | 1,283 | 9.795437 | Failed |
| 2 | Modified | 1,283 | 8.063339 | Passed |
| 3 | Modified | 1,283 | 7.119250 | Passed |
| 4 | Original | 1,283 | 6.470367 | Passed |
| 5 | Original | 1,283 | 9.774502 | Failed |
| 6 | Modified | 1,169 | 9.859791 | Failed |

I retain every result. The first and fifth attempts complete every shadow body
but still fail under the unchanged ten-second parent supervisor. I do not call
completed bodies a successful publication or compare a truncated body's total
as though it covers the complete selection. No completed-shadow assertion fails.
The large within-variant variation does not establish a reliable allocator
benefit or explain the failures as infrastructure. The diagnostic correctly
finishes with failure because three planned creations fail.

I CRC-check the complete hosted artifact in memory, retaining exact text and
entry hashes. I independently match actual Mach-O bytes for both C providers and
all three published native source compilers to their execution records. Provider,
source and host-compiler hashes remain unchanged. I do not independently qualify
ZIP modes, the complete NativeSDK/runtime closure, the original combined gate,
fresh stages, budgets, platform integration or release.

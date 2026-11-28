# My late clean shadow admission counterexample

I reproduce [#1249](https://github.com/jordanhubbard/nanolang/issues/1249) with the actual unchanged C seed from source `63248db641ccca41bfa5ccfb1bff0d1b32574d39`. My native image and assembler-capture helper match the [qualified original tools](../fresh-record-adoption-linux-20261124/README.md). I use a new external-only source fixture and preserve all old roots.

My explicitly public foreign function ignores its own SIGALRM and sleeps1.2s. The observer waits for its ready marker and for the parent's actual `hrtimer_nanosleep` state, SIGSTOPs only that parent, observes the real shadow child's exit with Linux pidfd polling, then SIGCONTs the parent. The compiler retains its existing one-second control override, real source shadow and assertion. I change no product source or deadline for this counterexample.

My completion receipt arrives1.200896s after parent start; my supervisor reports reaped=true, completed=true, exit0, signal0, parent_timeout=false under deadline_seconds1. I publish an actual ARM64 native product that runs with exit0. My parent loop breaks when waitpid reaps the child before checking the clock again. This admits the late result after a delayed polling wakeup. The shipped runtime header has the same ordering; its separate regression and repair remain tracked in #1249.

This is a deadline correctness failure, separate from #1229's full-source refusal. My [bounded earlier controls](../shadow-reap-controls-linux-20261124/README.md) show that later receipt/exit status cannot establish when completion happened. I do not attribute the historical or current full-source refusal to this counterexample and do not weaken its deadline.

I retain two setup refusals in separate roots: a private external declaration and an omitted original `nano_as_capture.so`. Both guards refuse before the timing case runs. Their captured error text remains in exception logs; the admitted controller retains a standalone complete stderr file. I repair only these new fixture inputs, then retain the actual compiler/native/helper/library/source guards and all produced capture artifacts.

I archive12403 file records, all three external roots, source/tool images, native product, process observations and original capture files. Manifest SHA-256: `ce964f53a1878153910d8fb957fd069c6f725bdf67cfd9d3c78d10a2fa91a02d`.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/shadow-late-exit-linux-20261124/read_archive.py
```

My reader verifies every archived byte, actual source and qualified image correspondence, controller observations, late receipt time and native product execution record. It does not claim full source, paired Darwin or whole-release acceptance.

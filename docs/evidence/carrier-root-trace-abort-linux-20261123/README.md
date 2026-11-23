# I abort incomplete carrier tracing before destruction

I qualify source `36f7ec1a18061d245e6de80b7f171a40057bb24f`, extending
my [private engine receipt](../carrier-root-engine-linux-20261123/README.md).
The actual source adapter is still pending under #1211/#1116/#1140.
I change no evaluator/environment/GC path, original source test, source
admission, Make recipe, public runtime layout or retained compiler/VM/AOT image.

A source adapter inspecting legacy borrowed containers may need a visited set
or encounter another incomplete-walk condition. My private trace callback now
returns whether it completed. A false result aborts before sweeping any
allocation, releases the mark queue and retains the complete ownership graph.
A subsequent complete collection starts a new mark pass and can reclaim the
unreachable nodes. Destructors and mutation guards keep their prior policy.

Both fresh strict GCC13 Linux ARM64 normal and ASan/UBSan profiles pass the
whole focused target, with leak detection enabled and empty stderr. My new
control stops after marking a child, proves the root/child/unreachable payloads
are all still valid and owned, then completes tracing and reclaims only the
unreachable node before final teardown. The unchanged prior sharing/cycle,
list-root identity, transfer/readoption/cross-heap, reentry and deep-graph/churn
controls also pass. I balance67,667 payload objects and their owned text
buffers. The unchanged fault fixture passes all six actual allocation failures
and releases all registry metadata. I preserve every earlier executable check.

I verify my source/products/archive with:

```sh
python3 docs/evidence/carrier-root-trace-abort-linux-20261123/read_archive.py
```

My reader passes1,417 records/1,398 exact Git inputs and four actual native
ARM64 products. It verifies raw/compressed hashes, object coverage, source and
image guards, complete commands/streams, retained prior assertions, unchanged
Make/original source inputs, exact change scope and the hash-bound preceding
engine archive. Objects and tool profiles are shared with the preceding
capsule, which retains all earlier development cohorts and their products.

I do not accept the interpreter's typed-list source UAF, full source carrier
reclamation, paired current-head bootstrap or the69-requirement release.
Actual fresh allocation/clone registration, binding/evaluation/argument/
aggregate/typed-list/host/async roots, early-discard and GC lifetime adapters
remain required before enabling collection in the interpreter. Both earlier
full-source LSan failures remain preserved and un-retried. Puck retains
#1204/#1190/nvm2c/merge/regex/capture/checkpoint/CI; all frozen roots/audits/
stashes and NAS/TCC/migration pauses remain intact. Native qualification is
terminal, with no Darwin reservation or NAS action.

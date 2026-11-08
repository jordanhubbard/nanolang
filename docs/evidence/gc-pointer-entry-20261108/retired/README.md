# My private GC-entry candidate retirement

I retire the corrected private tail-storage optimization after my fixed
[Linux](../../cseed-source-comparison-20261108/linux-724/README.md) and
[Apple](../../cseed-source-comparison-20261108/apple-902/README.md) comparisons
fail to establish a reliable source-creation benefit. I keep all original and
corrected candidate results. A reduced allocation count alone does not qualify
this change.

I restore `src/runtime/gc.c` and `tests/test_gc_struct.c` byte-for-byte from
correctness checkpoint `9baeba459e6ea4c7ebd6c0694cb05fef4d36f401`.
I retain header/payload and STRING terminator overflow refusals, shutdown
registry retirement after finalization and before freeing, accounting and
all thirteen GC control families. I remove only the candidate-specific tail
arithmetic assertion with its corresponding implementation.

My [actual local commands and hashes](checks.json) record unchanged inputs,
three native sanitizer products and a successful run of the existing GC,
interpreted BSTRING and STRING bounds Make targets. My [complete log](gc-and-bstring.log)
retains all assertions and diagnostics. I use the existing Darwin leak-detector
setting; this is not Linux leak-detector or full source/platform acceptance.
I hold full local provider builds while disk headroom is below policy.

Source-creation reliability, the frozen compiler budget, record-frame lifetime,
full corpus, hosted platform gates and native lifecycle admission remain open.

My source checkpoint `91735dabc` is a real descendant of the previous
`043b52d1a` source baseline with exactly the two reviewed GC files changed.
Merge `3857fdf8a` retains its ancestry without further implementation changes.
My [custody check](custody-precommit.json) preserves all 92,816 reviewed
entries. My [metadata commands](metadata-checks.json) pass the two real Git
controls, eight integration locks, four delegates, supported authority review
and canonical validation. The native receipt guard still refuses
`project.test_receipt_policy_unconfigured`; no native receipt is created.

I preserve the [initial authority refusal](first-authority-refusal.json): a
new Markdown link pointed outside the declared documentation catalog. I use
a plain retained-evidence path in my adoption record and leave the catalog
boundary unchanged. I freeze catalog inputs before recording authority.
